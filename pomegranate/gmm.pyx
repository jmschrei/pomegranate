#cython: boundscheck=False
#cython: cdivision=True
# gmm.pyx
# Contact: Jacob Schreiber ( jmschreiber91@gmail.com )

cimport numpy
import numpy
import json
import sys

if sys.version_info[0] > 2:
	xrange = range

import numpy
cimport numpy

from joblib import Parallel
from joblib import delayed

from .base cimport Model
from .base cimport GraphModel

from .kmeans import Kmeans

from .distributions cimport Distribution
from .distributions import DiscreteDistribution
from .utils cimport _log
from .utils cimport pair_lse

ctypedef numpy.npy_intp SIZE_t

from libc.stdlib cimport calloc
from libc.stdlib cimport free
from libc.string cimport memset
from libc.math cimport exp as cexp

import time

DEF NEGINF = float("-inf")
DEF INF = float("inf")

cpdef numpy.ndarray _check_input(X, dict keymap):
	"""Check the input to make sure that it is a properly formatted array."""

	cdef numpy.ndarray X_ndarray
	if isinstance(X, numpy.ndarray) and (X.dtype == 'float64'):
		return X

	try:
		X_ndarray = numpy.array(X, dtype='float64')
	except ValueError:
		X_ndarray = numpy.empty(X.shape, dtype='float64')

		if X.ndim == 1:
			for i in range(X.shape[0]):
				X_ndarray[i] = keymap[X[i]]
		else:
			for i in range(X.shape[0]):
				for j in range(X.shape[1]):
					X_ndarray[i, j] = keymap[X[i][j]]

	return X_ndarray

cdef double log_probability( Distribution model, double* samples, int n, int d ):
	cdef double logp = 0.0
	cdef int i, j

	for i in range(n):
		if d > 1:
			logp += model._mv_log_probability( samples + i*d )
		else:
			logp += model._log_probability( samples[i] )

	return logp

cdef class GeneralMixtureModel( Model ):
	"""A General Mixture Model.

	This mixture model can be a mixture of any distribution as long as
	they are all of the same dimensionality. Any object can serve as a
	distribution as long as it has fit(X, weights), log_probability(X),
	and summarize(X, weights)/from_summaries() methods if out of core
	training is desired.

	Parameters
	----------
	distributions : array-like, shape (n_components,) or callable
		The components of the model. If array, corresponds to the initial
		distributions of the components. If callable, must also pass in the
		number of components and kmeans++ will be used to initialize them.

	weights : array-like, optional, shape (n_components,)
		The prior probabilities corresponding to each component. Does not
		need to sum to one, but will be normalized to sum to one internally.
		Defaults to None.

	n_components : int, optional
		If a callable is passed into distributions then this is the number
		of components to initialize using the kmeans++ algorithm. Defaults
		to None.


	Attributes
	----------
	distributions : array-like, shape (n_components,)
		The component distribution objects.

	weights : array-like, shape (n_components,)
		The learned prior weight of each object


	Examples
	--------
	>>> from pomegranate import *
	>>> clf = GeneralMixtureModel([NormalDistribution(5, 2), NormalDistribution(1, 1)])
	>>> clf.log_probability(5)
	-2.304562194038089
	>>> clf.predict_proba([[5], [7], [1]])
	array([[ 0.99932952,  0.00067048],
	       [ 0.99999995,  0.00000005],
	       [ 0.06337894,  0.93662106]])
	>>> clf.fit([[1], [5], [7], [8], [2]])
	>>> clf.predict_proba([[5], [7], [1]])
	array([[ 1.        ,  0.        ],
	       [ 1.        ,  0.        ],
	       [ 0.00004383,  0.99995617]])
	>>> clf.distributions
	array([ {
	    "frozen" :false,
	    "class" :"Distribution",
	    "parameters" :[
	        6.6571359101390755,
	        1.2639830514274502
	    ],
	    "name" :"NormalDistribution"
	},
	       {
	    "frozen" :false,
	    "class" :"Distribution",
	    "parameters" :[
	        1.498707696758334,
	        0.4999983303277837
	    ],
	    "name" :"NormalDistribution"
	}], dtype=object)
	"""

	cdef public numpy.ndarray distributions
	cdef object distribution_callable
	cdef public numpy.ndarray weights
	cdef numpy.ndarray summaries_ndarray
	cdef void** distributions_ptr
	cdef double* weights_ptr
	cdef double* summaries_ptr
	cdef dict keymap
	cdef int n
	cdef public int hmm

	def __init__( self, distributions, weights=None, n_components=None ):
		if not callable(distributions) and not isinstance(distributions, list):
			raise ValueError("must either give initial distributions or constructor")

		self.d = 0
		self.hmm = 0

		if callable(distributions):
			if distributions == DiscreteDistribution:
				raise ValueError("cannot fit a discrete GMM without pre-initialized distributions")
			self.n = n_components
			self.distribution_callable = distributions
		else:
			if len(distributions) < 2:
				raise ValueError("must have at least two distributions for general mixture models")

			for dist in distributions:
				if callable(dist):
					raise TypeError("must have initialized distributions in list")
				elif self.d == 0:
					self.d = dist.d
				elif self.d != dist.d:
					raise TypeError("mis-matching dimensions between distributions in list")
				if dist.model == 'HiddenMarkovModel':
					self.hmm = 1

			if weights is None:
				weights = numpy.ones_like(distributions, dtype=float) / len( distributions )
			else:
				weights = numpy.asarray(weights) / weights.sum()

			self.weights = numpy.log( weights )
			self.weights_ptr = <double*> self.weights.data

			self.distributions = numpy.array( distributions )
			self.distributions_ptr = <void**> self.distributions.data

			self.summaries_ndarray = numpy.zeros_like(weights, dtype='float64')
			self.summaries_ptr = <double*> self.summaries_ndarray.data

			self.n = len(distributions)

		if self.d > 0 and isinstance( self.distributions[0], DiscreteDistribution ):
			keys = []
			for d in self.distributions:
				keys.extend( d.keys() )
			self.keymap = { key: i for i, key in enumerate(set(keys)) }
			for d in self.distributions:
				d.bake( tuple(set(keys)) )


	def __reduce__( self ):
		"""Serialize model for pickling."""
		return self.__class__, (self.distributions.tolist(), self.weights, self.n)

	def sample( self, n=1 ):
		"""Generate a sample from the model.

		First, randomly select a component weighted by the prior probability,
		Then, use the sample method from that component to generate a sample.

		Parameters
		----------
		n : int, optional
			The number of samples to generate. Defaults to 1.

		Returns
		-------
		sample : array-like or object
			A randomly generated sample from the model of the type modelled
			by the emissions. An integer if using most distributions, or an
			array if using multivariate ones, or a string for most discrete
			distributions. If n=1 return an object, if n>1 return an array
			of the samples.
		"""

		samples = []
		for i in xrange(n):
			d = numpy.random.choice( self.distributions, p=numpy.exp(self.weights) )
			samples.append(d.sample())

		return samples if n > 1 else samples[0]

	cpdef log_probability( self, X ):
		"""Calculate the log probability of a point under the distribution.

		The probability of a point is the sum of the probabilities of each
		distribution multiplied by the weights. Thus, the log probability
		is the sum of the log probability plus the log prior.

		This is the python interface.

		Parameters
		----------
		X : numpy.ndarray, shape=(n, d) or (n, m, d)
			The samples to calculate the log probability of. Each row is a
			sample and each column is a dimension. If emissions are HMMs then
			shape is (n, m, d) where m is variable length for each obervation,
			and X becomes an array of n (m, d)-shaped arrays.  

		Returns
		-------
		log_probability : double
			The log probabiltiy of the point under the distribution.
		"""

		if self.d == 0:
			raise ValueError("must first fit model before using log probability method.")

		cdef int i, j, n, d, m

		if self.hmm == 1:
			n, d = len(X), self.d
		elif self.d == 1:
			n, d = X.shape[0], 1
		elif self.d > 1 and X.ndim == 1:
			n, d = 1, len(X)
		else:
			n, d = X.shape

		cdef numpy.ndarray logp_ndarray = numpy.zeros(n)
		cdef double* logp = <double*> logp_ndarray.data

		cdef numpy.ndarray X_ndarray
		cdef double* X_ptr 

		if self.hmm == 0:
			X_ndarray = numpy.array(X)
			X_ptr = <double*> X_ndarray.data

		with nogil:
			for i in range(n):
				if self.hmm == 1:
					with gil:
						X_ndarray = numpy.array(X[i])
						X_ptr = <double*> X_ndarray.data
					logp[i] = self._vl_log_probability(X_ptr, n)
				elif d == 1:
					logp[i] = self._log_probability(X_ptr[i])
				else:
					logp[i] = self._mv_log_probability(X_ptr+i*d)

		return logp_ndarray

	cdef double _log_probability(self, double X) nogil:
		cdef int i
		cdef double log_probability_sum = NEGINF
		cdef double log_probability

		for i in range( self.n ):
			log_probability = ( <Model> self.distributions_ptr[i] )._log_probability(X) + self.weights_ptr[i]
			log_probability_sum = pair_lse( log_probability_sum, log_probability )

		return log_probability_sum

	cdef double _mv_log_probability(self, double* X) nogil:
		cdef int i
		cdef double log_probability_sum = NEGINF
		cdef double log_probability

		for i in range( self.n ):
			log_probability = ( <Model> self.distributions_ptr[i] )._mv_log_probability(X) + self.weights_ptr[i]
			log_probability_sum = pair_lse( log_probability_sum, log_probability )

		return log_probability_sum

	cdef double _vl_log_probability(self, double* X, int n) nogil:
		cdef int i
		cdef double log_probability_sum = NEGINF
		cdef double log_probability

		for i in range( self.n ):
			log_probability = ( <Model> self.distributions_ptr[i] )._vl_log_probability(X, n) + self.weights_ptr[i]
			log_probability_sum = pair_lse( log_probability_sum, log_probability )

		return log_probability_sum


	def predict_proba(self, X):
		"""Calculate the posterior P(M|D) for data.

		Calculate the probability of each item having been generated from
		each component in the model. This returns normalized probabilities
		such that each row should sum to 1.

		Since calculating the log probability is much faster, this is just
		a wrapper which exponentiates the log probability matrix.

		Parameters
		----------
		X : array-like, shape (n_samples, n_dimensions)
			The samples to do the prediction on. Each sample is a row and each
			column corresponds to a dimension in that sample. For univariate
			distributions, a single array may be passed in.

		Returns
		-------
		probability : array-like, shape (n_samples, n_components)
			The normalized probability P(M|D) for each sample. This is the
			probability that the sample was generated from each component.
		"""

		if self.d == 0:
			raise ValueError("must first fit model before using predict proba method.")

		return numpy.exp( self.predict_log_proba(X) )

	def predict_log_proba( self, X ):
		"""Calculate the posterior log P(M|D) for data.

		Calculate the log probability of each item having been generated from
		each component in the model. This returns normalized log probabilities
		such that the probabilities should sum to 1

		This is a sklearn wrapper for the original posterior function.

		Parameters
		----------
		X : array-like, shape (n_samples, n_dimensions)
			The samples to do the prediction on. Each sample is a row and each
			column corresponds to a dimension in that sample. For univariate
			distributions, a single array may be passed in.

		Returns
		-------
		y : array-like, shape (n_samples, n_components)
			The normalized log probability log P(M|D) for each sample. This is
			the probability that the sample was generated from each component.
		"""

		if self.d == 0:
			raise ValueError("must first fit model before using predict log proba method.")

		cdef int i, n, d
		cdef numpy.ndarray X_ndarray
		cdef double* X_ptr
		cdef numpy.ndarray y
		cdef double* y_ptr

		if self.hmm == 1:
			n, d = len(X), self.d
		elif self.d == 1:
			n, d = X.shape[0], 1
		elif self.d > 1 and X.ndim == 1:
			n, d = 1, len(X)
		else:
			n, d = X.shape

		y = numpy.zeros((n, self.n), dtype='float64')
		y_ptr = <double*> y.data

		if self.hmm == 0:
			X_ndarray = _check_input(X, self.keymap)
			X_ptr = <double*> X_ndarray.data

		with nogil:
			if self.hmm == 0:
				self._predict_log_proba( X_ptr, y_ptr, n, d )
			else:
				for i in range(n):
					with gil:
						X_ndarray = _check_input(X[i], self.keymap)
						X_ptr = <double*> X_ndarray.data
						d = len(X_ndarray)

					self._predict_log_proba( X_ptr, y_ptr+i*self.n, 1, d )
		
		return y if self.hmm == 1 else y.reshape(self.n, n).T

	cdef void _predict_log_proba( self, double* X, double* y, int n, int d ) nogil:
		cdef double y_sum, logp
		cdef int i, j

		for j in range(self.n):
			if self.hmm == 1:
				y[j] = (<Model> self.distributions_ptr[j])._vl_log_probability(X, d)
			else:
				(<Model> self.distributions_ptr[j])._v_log_probability(X, y+j*n, n)

		for i in range(n):
			y_sum = NEGINF

			for j in range(self.n):
				y[j*n + i] += self.weights_ptr[j]
				y_sum = pair_lse(y_sum, y[j*n + i])

			for j in range(self.n):
				y[j*n + i] -= y_sum

	cpdef predict( self, X ):
		"""Predict the most likely component which generated each sample.

		Calculate the posterior P(M|D) for each sample and return the index
		of the component most likely to fit it. This corresponds to a simple
		argmax over the responsibility matrix.

		This is a sklearn wrapper for the maximum_a_posteriori method.

		Parameters
		----------
		X : array-like, shape (n_samples, n_dimensions)
			The samples to do the prediction on. Each sample is a row and each
			column corresponds to a dimension in that sample. For univariate
			distributions, a single array may be passed in.

		Returns
		-------
		y : array-like, shape (n_samples,)
			The predicted component which fits the sample the best.
		"""

		cdef int i, n, d

		if self.d == 0:
			raise ValueError("must first fit model before using predict method.")

		if self.hmm == 1:
			n, d = len(X), self.d
		elif self.d == 1:
			n, d = X.shape[0], 1
		elif self.d > 1 and X.ndim == 1:
			n, d = 1, len(X)
		else:
			n, d = X.shape

		cdef numpy.ndarray X_ndarray
		cdef double* X_ptr

		cdef numpy.ndarray y = numpy.zeros(n, dtype='int32')
		cdef int* y_ptr = <int*> y.data

		if self.hmm == 0:
			X_ndarray = _check_input(X, self.keymap)
			X_ptr = <double*> X_ndarray.data

		with nogil:
			if self.hmm == 0:
				self._predict(X_ptr, y_ptr, n, d)
			else:
				for i in range(n):
					with gil:
						X_ndarray = _check_input(X[i], self.keymap)
						X_ptr = <double*> X_ndarray.data
						d = len(X_ndarray)

					self._predict(X_ptr, y_ptr+i, 1, d)

		return y

	cdef void _predict( self, double* X, int* y, int n, int d) nogil:
		cdef int i, j
		cdef double max_logp, logp
		cdef double* r = <double*> calloc(n*self.n, sizeof(double))

		for j in range(self.n):
			if self.hmm == 1:
				r[j] = (<Model> self.distributions_ptr[j])._vl_log_probability(X, d)
			else:
				(<Model> self.distributions_ptr[j])._v_log_probability(X, r+j*n, n)

		for i in range(n):
			max_logp = NEGINF

			for j in range(self.n):
				logp = r[j*n + i] + self.weights_ptr[j]
				if logp > max_logp:
					max_logp = logp
					y[i] = j

		free(r)

	def fit( self, X, weights=None, inertia=0.0, stop_threshold=0.1,
		max_iterations=1e8, verbose=False ):
		"""Fit the model to new data using EM.

		This method fits the components of the model to new data using the EM
		method. It will iterate until either max iterations has been reached,
		or the stop threshold has been passed.

		This is a sklearn wrapper for train method.

		Parameters
		----------
		X : array-like, shape (n_samples, n_dimensions)
			This is the data to train on. Each row is a sample, and each column
			is a dimension to train on.

		weights : array-like, shape (n_samples,), optional
			The initial weights of each sample in the matrix. If nothing is
			passed in then each sample is assumed to be the same weight.
			Default is None.

		inertia : double, optional
			The weight of the previous parameters of the model. The new
			parameters will roughly be old_param*inertia + new_param*(1-inertia),
			so an inertia of 0 means ignore the old parameters, whereas an
			inertia of 1 means ignore the new parameters. Default is 0.0.

		stop_threshold : double, optional, positive
			The threshold at which EM will terminate for the improvement of
			the model. If the model does not improve its fit of the data by
			a log probability of 0.1 then terminate. Default is 0.1.

		max_iterations : int, optional, positive
			The maximum number of iterations to run EM for. If this limit is
			hit then it will terminate training, regardless of how well the
			model is improving per iteration. Default is 1e8.

		verbose : bool, optional
			Whether or not to print out improvement information over iterations.
			Default is False.

		Returns
		-------
		improvement : double
			The total improvement in log probability P(D|M)
		"""

		initial_log_probability_sum = NEGINF
		iteration, improvement = 0, INF

		if weights is None:
			weights = numpy.ones(len(X), dtype='float64')
		else:
			weights = numpy.array(weights, dtype='float64')

		while improvement > stop_threshold and iteration < max_iterations + 1:
			self.from_summaries(inertia)
			log_probability_sum = self.summarize(X, weights)

			if iteration == 0:
				initial_log_probability_sum = log_probability_sum
			else:
				improvement = log_probability_sum - last_log_probability_sum

				if verbose:
					print( "Improvement: {}".format(improvement) )

			iteration += 1
			last_log_probability_sum = log_probability_sum

		self.clear_summaries()

		if verbose:
			print( "Total Improvement: {}".format(last_log_probability_sum - initial_log_probability_sum) )

		return last_log_probability_sum - initial_log_probability_sum

	def summarize( self, X, weights=None ):
		"""Summarize a batch of data and store sufficient statistics.

		This will run the expectation step of EM and store sufficient
		statistics in the appropriate distribution objects. The summarization
		can be thought of as a chunk of the E step, and the from_summaries
		method as the M step.

		Parameters
		----------
		X : array-like, shape (n_samples, n_dimensions)
			This is the data to train on. Each row is a sample, and each column
			is a dimension to train on.

		weights : array-like, shape (n_samples,), optional
			The initial weights of each sample in the matrix. If nothing is
			passed in then each sample is assumed to be the same weight.
			Default is None.

		Returns
		-------
		logp : double
			The log probability of the data given the current model. This is
			used to speed up EM.
		"""

		cdef int i, n, d
		cdef numpy.ndarray X_ndarray
		cdef numpy.ndarray weights_ndarray
		cdef double log_probability

		if self.hmm == 1:
			n, d = len(X), self.d
		elif self.d == 1:
			n, d = X.shape[0], 1
		elif self.d > 1 and X.ndim == 1:
			n, d = 1, len(X)
		else:
			n, d = X.shape

		if weights is None:
			weights_ndarray = numpy.ones(n, dtype='float64')
		else:
			weights_ndarray = numpy.array(weights, dtype='float64')

		# If not initialized then we need to do kmeans initialization.
		if self.d == 0:
			X_ndarray = _check_input(X, self.keymap)
			kmeans = Kmeans(self.n)
			kmeans.fit(X_ndarray, max_iterations=1)
			y = kmeans.predict(X_ndarray)

			distributions = [ self.distribution_callable.from_samples( X_ndarray[y==i] ) for i in range(self.n) ]
			self.d = distributions[0].d

			self.distributions = numpy.array(distributions)
			self.distributions_ptr = <void**> self.distributions.data

			self.weights = numpy.log(numpy.ones(self.n, dtype='float64') / self.n)
			self.weights_ptr = <double*> self.weights.data

			self.summaries_ndarray = numpy.zeros_like(self.weights, dtype='float64')
			self.summaries_ptr = <double*> self.summaries_ndarray.data

		cdef double* X_ptr
		cdef double* weights_ptr = <double*> weights_ndarray.data

		if self.hmm == 0:
			X_ndarray = _check_input(X, self.keymap)
			X_ptr = <double*> X_ndarray.data
			
			with nogil:
				log_probability = self._summarize(X_ptr, weights_ptr, n)
		else:
			log_probability = 0.0
			for i in range(n):
				X_ndarray = _check_input(X[i], self.keymap)
				X_ptr = <double*> X_ndarray.data
				d = len(X_ndarray)
				with nogil:
					log_probability += self._summarize(X_ptr, weights_ptr+i, d)

		return log_probability


	cdef double _summarize(self, double* X, double* weights, int n) nogil:
		cdef double* r = <double*> calloc(self.n*n, sizeof(double))
		cdef double* summaries = <double*> calloc(self.n, sizeof(double))
		cdef int i, j
		cdef double total, logp, log_probability_sum = 0.0

		memset(summaries, 0, self.n*sizeof(double))
		cdef double tic

		for j in range(self.n):
			if self.hmm == 1:
				r[j*n] = (<Model> self.distributions_ptr[j])._vl_log_probability(X, n)
			else:
				(<Model> self.distributions_ptr[j])._v_log_probability(X, r+j*n, n)

		for i in range(n):
			total = NEGINF

			for j in range(self.n):
				r[j*n + i] += self.weights_ptr[j]
				total = pair_lse(total, r[j*n + i])

			for j in range(self.n):
				r[j*n + i] = cexp(r[j*n + i] - total) * weights[i]
				summaries[j] += r[j*n + i]
			
			log_probability_sum += total * weights[i]

			if self.hmm == 1:
				break

		for j in range(self.n):
			(<Model> self.distributions_ptr[j])._summarize(X, r+j*n, n)

		with gil:
			for j in range(self.n):
				self.summaries_ptr[j] += summaries[j]

		free(r)
		free(summaries)
		return log_probability_sum

	def from_summaries( self, inertia=0.0, **kwargs ):
		"""Fit the model to the collected sufficient statistics.

		Fit the parameters of the model to the sufficient statistics gathered
		during the summarize calls. This should return an exact update.

		Parameters
		----------
		inertia : double, optional
			The weight of the previous parameters of the model. The new
			parameters will roughly be old_param*inertia + new_param*(1-inertia),
			so an inertia of 0 means ignore the old parameters, whereas an
			inertia of 1 means ignore the new parameters. Default is 0.0.

		Returns
		-------
		None
		"""

		if self.d == 0 or self.summaries_ndarray.sum() == 0:
			return

		self.summaries_ndarray /= self.summaries_ndarray.sum()
		for i, distribution in enumerate(self.distributions):
			distribution.from_summaries(inertia, **kwargs)
			self.weights[i] = _log(self.summaries_ndarray[i])
			self.summaries_ndarray[i] = 0.

	def clear_summaries( self ):
		"""Clear the summary statistics stored in the object.

		Parameters
		----------
		None

		Returns
		-------
		None
		"""

		self.summaries_ndarray *= 0
		for distribution in self.distributions:
			distribution.clear_summaries()

	def to_json( self, separators=(',', ' : '), indent=4 ):
		"""Serialize the model to a JSON.

		Parameters
		----------
		separators : tuple, optional
			The two separaters to pass to the json.dumps function for formatting.
			Default is (',', ' : ').

		indent : int, optional
			The indentation to use at each level. Passed to json.dumps for
			formatting. Default is 4.

		Returns
		-------
		json : str
			A properly formatted JSON object.
		"""

		model = {
					'class' : 'GeneralMixtureModel',
					'distributions'  : [ json.loads( dist.to_json() ) for dist in self.distributions ],
					'weights' : self.weights.tolist()
				}

		return json.dumps( model, separators=separators, indent=indent )

	@classmethod
	def from_json( cls, s ):
		"""Read in a serialized model and return the appropriate classifier.

		Parameters
		----------
		s : str
			A JSON formatted string containing the file.
		Returns
		-------
		model : object
			A properly initialized and baked model.
		"""

		d = json.loads( s )
		distributions = [ Distribution.from_json( json.dumps(j) ) for j in d['distributions'] ]
		model = GeneralMixtureModel( distributions, numpy.array( d['weights'] ) )
		return model

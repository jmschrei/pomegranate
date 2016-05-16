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

from .distributions cimport Distribution
from .distributions import DiscreteDistribution
from .utils cimport _log
from .utils cimport pair_lse

ctypedef numpy.npy_intp SIZE_t

from libc.stdlib cimport calloc
from libc.stdlib cimport free
from libc.string cimport memset
from libc.math cimport exp as cexp

# Define some useful constants
DEF NEGINF = float("-inf")
DEF INF = float("inf")

cpdef numpy.ndarray _check_input( X, keymap ):
	"""Check the input to make sure that it is a properly formatted array."""

	cdef int n = len(X), d = len(X[0])
	cdef numpy.ndarray X_ndarray

	if isinstance( X, numpy.ndarray ) and ( X.dtype == 'float64' ):
		return X

	try:
		X_ndarray = numpy.array( X, dtype='float64' )
	except ValueError:
		X_ndarray = numpy.empty((n, d), dtype='float64')
		for i in range(n):
			for j in range(d):
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

cdef class Kmeans( object ):
	"""A kmeans model.

	Kmeans is not a probabilistic model, but it is used in the kmeans++
	initialization for GMMs. In essence, a point is selected as the center
	for one component and then remaining points are selected

	Parameters
	----------
	k : int
		The number of centroids.

	Attributes
	----------
	k : int
		The number of centroids

	centroids : array-like, shape (k, n_dim)
		The means of the centroid points.
	"""

	cdef public int k, d
	cdef public numpy.ndarray centroids
	cdef double* centroids_ptr
	cdef int initialized
	cdef double* summary_sizes
	cdef double* summary_weights

	def __init__( self, k ):
		self.k = k
		self.d = 0
		self.initialized = False


	def __dealloc__( self ):
		free(self.summary_sizes)
		free(self.summary_weights)

	cpdef fit( self, X, int max_iterations=10 ):
		"""Fit the model to the data using k centroids.
		
		Parameters
		----------
		X : array-like, shape (n_samples, n_dim)
			The data to fit to.

		max_iterations : int, optional
			The maximum number of iterations to run for. Default is 10.

		Returns
		-------
		None
		"""

		cdef int iterations = 0
		while iterations < max_iterations:
			iterations += 1

			self.summarize( X )
			self.from_summaries()

	cpdef summarize( self, X ):
		"""Summarize the points into sufficient statistics for a future update.

		Parameters
		----------
		X : array-like, shape (n_samples, n_dim)
			The data to fit to.

		Returns
		-------
		None
		"""

		cdef numpy.ndarray X_ndarray = numpy.array(X, dtype='float64')
		cdef int n = X_ndarray.shape[0], d = X_ndarray.shape[1], i, j
		cdef double* X_ptr = <double*> X_ndarray.data

		if not self.initialized:
			self.d = d
			self.centroids = numpy.zeros((self.k, d))
			self.centroids_ptr = <double*> self.centroids.data
			self.summary_sizes = <double*> calloc(self.k, sizeof(double))
			self.summary_weights = <double*> calloc(self.k*d, sizeof(double))

			self.initialized = True
			for i in range(self.k):
				for j in range(d):
					self.centroids_ptr[i*d + j] = X_ptr[i*d + j]

		self._summarize( X_ptr, n, d, self.k )

	cdef void _summarize( self, double* X, int n, int d, int k ) nogil:
		cdef int i, j, l, y
		cdef double min_dist, dist
		cdef double* summary_sizes = <double*> calloc(k, sizeof(double))
		cdef double* summary_weights = <double*> calloc(k*d, sizeof(double))

		for i in range(n):
			min_dist = INF

			for j in range(k):
				dist = 0.0

				for l in range(d):
					dist += ( self.centroids_ptr[j*d + l] - X[i*d + l] ) ** 2.0

				if dist < min_dist:
					min_dist = dist
					y = j

			summary_sizes[y] += 1

			for l in range(d):
				summary_weights[y*d + l] += X[i*d + l]

		with gil:
			for j in range(k):
				self.summary_sizes[j] += summary_sizes[j]

				for l in range(d):
					self.summary_weights[j*d + l] += summary_weights[j*d + l]

		free(summary_sizes)
		free(summary_weights)

	cpdef from_summaries( self ):
		"""Fit the model to the sufficient statistics.

		Parameters
		----------
		None

		Returns
		-------
		None
		"""

		cdef int l, j, k = self.k, d = self.d

		for j in range(k):
			for l in range(d):
				self.centroids_ptr[j*d + l] = self.summary_weights[j*d + l] / self.summary_sizes[j]
				self.summary_weights[j*d + l] = 0.0
			self.summary_sizes[j] = 0.0

	cpdef predict( self, X ):
		"""Predict nearest centroid for each point.

		Parameters
		----------
		X : array-like, shape (n_samples, n_dim)
			The data to fit to.

		Returns
		-------
		y : array-like, shape (n_samples,)
			The index of the nearest centroid.
		"""

		X = numpy.array(X)
		cdef double* X_ptr = <double*> (<numpy.ndarray> X).data
		cdef int n, d

		n, d = X.shape
		
		cdef numpy.ndarray y = numpy.zeros(n, dtype='int32')
		cdef int* y_ptr = <int*> y.data

		self._predict( X_ptr, y_ptr, n, d, self.k )
		return y

	cdef void _predict( self, double* X, int* y, int n, int d, int k ) nogil:
		cdef int i, j, l
		cdef double dist, min_dist

		for i in range(n):
			min_dist = INF

			for j in range(k):
				dist = 0.0

				for l in range(d):
					dist += ( self.centroids_ptr[j*d + l] - X[i*d + l] ) ** 2.0

				if dist < min_dist:
					min_dist = dist
					y[i] = j


cdef class GeneralMixtureModel( Distribution ):
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
	cdef int initialized

	def __init__( self, distributions, weights=None, n_components=None ):
		if callable(distributions):
			self.initialized = False
			self.n = n_components
			self.distribution_callable = distributions
		else:
			self.initialized = True

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
			self.d = distributions[0].d

		if self.initialized and isinstance( self.distributions[0], DiscreteDistribution ):
			keys = []
			for d in self.distributions:
				keys.extend( d.keys() )
			self.keymap = { key: i for i, key in enumerate(set(keys)) }
			for d in self.distributions:
				d.encode( tuple(set(keys)) )

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

	def log_probability( self, point ):
		"""Calculate the log probability of a point under the distribution.

		The probability of a point is the sum of the probabilities of each
		distribution multiplied by the weights. Thus, the log probability
		is the sum of the log probability plus the log prior.

		This is the python interface.

		Parameters
		----------
		point : object
			The sample to calculate the log probability of. This is usually an
			integer, but can also be an array of size (n_dims,) or any
			object.

		Returns
		-------
		log_probability : double
			The log probabiltiy of the point under the distribution.
		"""

		if not self.initialized:
			raise ValueError("must first fit model before using log probability method.")

		n = len( self.distributions )
		log_probability_sum = NEGINF

		for i in xrange( n ):
			d = self.distributions[i]
			log_probability = d.log_probability( point ) + self.weights[i]
			log_probability_sum = pair_lse( log_probability_sum,log_probability )

		return log_probability_sum

	cdef double _log_probability( self, double symbol ) nogil:
		"""Calculate the log probability of a point under the distribution.

		The probability of a point is the sum of the probabilities of each
		distribution multiplied by the weights. Thus, the log probability
		is the sum of the log probability plus the log prior.

		This is the cython nogil interface for univariate emissions.

		Parameters
		----------
		point : object
			The sample to calculate the log probability of. This is usually an
			integer, but can also be an array of size (n_components,) or any
			object.

		Returns
		-------
		log_probability : double
			The log probabiltiy of the point under the distribution.
		"""

		cdef int i
		cdef double log_probability_sum = NEGINF
		cdef double log_probability

		for i in range( self.n ):
			log_probability = ( <Distribution> self.distributions_ptr[i] )._log_probability( symbol ) + self.weights_ptr[i]
			log_probability_sum = pair_lse( log_probability_sum, log_probability )

		return log_probability_sum

	cdef double _mv_log_probability( self, double* symbol ) nogil:
		"""Calculate the log probability of a point under the distribution.

		The probability of a point is the sum of the probabilities of each
		distribution multiplied by the weights. Thus, the log probability
		is the sum of the log probability plus the log prior.

		This is the cython nogil interface for multivariate emissions.

		Parameters
		----------
		point : object
			The sample to calculate the log probability of. This is usually an
			integer, but can also be an array of size (n_components,) or any
			object.

		Returns
		-------
		log_probability : double
			The log probabiltiy of the point under the distribution.
		"""

		cdef int i
		cdef double log_probability_sum = NEGINF
		cdef double log_probability

		for i in range( self.n ):
			log_probability = ( <Distribution> self.distributions_ptr[i] )._mv_log_probability( symbol ) + self.weights_ptr[i]
			log_probability_sum = pair_lse( log_probability_sum, log_probability )

		return log_probability_sum


	def predict_proba( self, items ):
		"""Calculate the posterior P(M|D) for data.

		Calculate the probability of each item having been generated from
		each component in the model. This returns normalized probabilities
		such that each row should sum to 1.

		Since calculating the log probability is much faster, this is just
		a wrapper which exponentiates the log probability matrix.

		Parameters
		----------
		items : array-like, shape (n_samples, n_dimensions)
			The samples to do the prediction on. Each sample is a row and each
			column corresponds to a dimension in that sample. For univariate
			distributions, a single array may be passed in.

		Returns
		-------
		probability : array-like, shape (n_samples, n_components)
			The normalized probability P(M|D) for each sample. This is the
			probability that the sample was generated from each component.
		"""

		if not self.initialized:
			raise ValueError("must first fit model before using predict proba method.")

		return numpy.exp( self.predict_log_proba( items ) )

	def predict_log_proba( self, items ):
		"""Calculate the posterior log P(M|D) for data.

		Calculate the log probability of each item having been generated from
		each component in the model. This returns normalized log probabilities
		such that the probabilities should sum to 1

		This is a sklearn wrapper for the original posterior function.

		Parameters
		----------
		items : array-like, shape (n_samples, n_dimensions)
			The samples to do the prediction on. Each sample is a row and each
			column corresponds to a dimension in that sample. For univariate
			distributions, a single array may be passed in.

		Returns
		-------
		y : array-like, shape (n_samples, n_components)
			The normalized log probability log P(M|D) for each sample. This is
			the probability that the sample was generated from each component.
		"""

		if not self.initialized:
			raise ValueError("must first fit model before using predict log proba method.")

		cdef int n = len(items), d = len(items[0])
		cdef numpy.ndarray items_ndarray = _check_input( items, self.keymap )

		cdef double* items_ptr = <double*> items_ndarray.data
		cdef numpy.ndarray y = numpy.zeros((n, self.n))
		cdef double* y_ptr = <double*> y.data
		self._predict_log_proba( items_ptr, n, d, self.n, y_ptr )
		return y

	cdef void _predict_log_proba( self, double* items, int n, int d, int m, double* y) nogil:
		cdef double y_sum, logp
		cdef int i, j

		for i in range(n):
			y_sum = NEGINF

			for j in range(m):
				if d > 1:
					logp = (<Distribution> self.distributions_ptr[j])._mv_log_probability(items + i*d)
				else:
					logp = (<Distribution> self.distributions_ptr[j])._log_probability(items[i])

				y[i*m + j] = logp + self.weights_ptr[j]
				y_sum = pair_lse(y_sum, y[i*m + j])

			for j in range(m):
				y[i*m + j] = y[i*m + j] - y_sum

	cpdef predict( self, items ):
		"""Predict the most likely component which generated each sample.

		Calculate the posterior P(M|D) for each sample and return the index
		of the component most likely to fit it. This corresponds to a simple
		argmax over the responsibility matrix. 

		This is a sklearn wrapper for the maximum_a_posteriori method.

		Parameters
		----------
		items : array-like, shape (n_samples, n_dimensions)
			The samples to do the prediction on. Each sample is a row and each
			column corresponds to a dimension in that sample. For univariate
			distributions, a single array may be passed in.

		Returns
		-------
		indexes : array-like, shape (n_samples,)
			The index of the component which fits the sample the best.
		"""

		if not self.initialized:
			raise ValueError("must first fit model before using predict method.")

		n, d = len(items), len(items[0])
		cdef numpy.ndarray items_ndarray = _check_input( items, self.keymap )

		cdef double* items_ptr = <double*> items_ndarray.data
		cdef int* y_ptr = self._predict( items_ptr, n, d, self.n )
		
		y = numpy.zeros(n, dtype='int')
		for i in range(n):
			y[i] = y_ptr[i]

		free(y_ptr)
		return y

	cdef int* _predict( self, double* items, int n, int d, int m ) nogil:
		cdef int i, j
		cdef int* y = <int*> calloc(n, sizeof(int))
		cdef double max_logp, logp

		for i in range(n):
			max_logp = NEGINF

			for j in range(m):
				if d > 1:
					logp = (<Distribution> self.distributions_ptr[j])._mv_log_probability(items + i*d) + self.weights_ptr[j]
				else:
					logp = (<Distribution> self.distributions_ptr[j])._log_probability(items[i]) + self.weights_ptr[j]

				if logp > max_logp:
					max_logp = logp
					y[i] = j

		return y

	def fit( self, items, weights=None, inertia=0.0, stop_threshold=0.1, 
		max_iterations=1e8, verbose=False ):
		"""Fit the model to new data using EM.

		This method fits the components of the model to new data using the EM
		method. It will iterate until either max iterations has been reached,
		or the stop threshold has been passed.

		This is a sklearn wrapper for train method.

		Parameters
		----------
		items : array-like, shape (n_samples, n_dimensions)
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

		cdef int n = len(items), d = len(items[0])
		cdef numpy.ndarray items_ndarray = _check_input( items, self.keymap )

		# If not initialized then we need to do kmeans initialization.
		if self.initialized == False or d != self.d:
			kmeans = Kmeans(self.n)
			kmeans.fit(items_ndarray, max_iterations=1)
			y = kmeans.predict(items)
			distributions = [ self.distribution_callable.from_samples( items_ndarray[y==i] ) for i in range(self.n) ]

			self.distributions = numpy.array( distributions )
			self.distributions_ptr = <void**> self.distributions.data

			self.weights = numpy.log( numpy.ones(self.n) / self.n )
			self.weights_ptr = <double*> self.weights.data

			self.summaries_ndarray = numpy.zeros_like(self.weights, dtype='float64')
			self.summaries_ptr = <double*> self.summaries_ndarray.data

			self.n = len(distributions)
			self.d = distributions[0].d
			self.initialized = True

		cdef double* X_ptr = <double*> items_ndarray.data

		initial_log_probability_sum = NEGINF
		iteration, improvement = 0, INF 

		while improvement > stop_threshold and iteration < max_iterations + 1:
			self.from_summaries(inertia)
			log_probability_sum = self.summarize(items_ndarray, weights)

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

	def summarize( self, items, weights=None ):
		"""Summarize a batch of data and store sufficient statistics.

		This will run the expectation step of EM and store sufficient
		statistics in the appropriate distribution objects. The summarization
		can be thought of as a chunk of the E step, and the from_summaries
		method as the M step.

		Parameters
		----------
		items : array-like, shape (n_samples, n_dimensions)
			This is the data to train on. Each row is a sample, and each column
			is a dimension to train on.

		weights : array-like, shape (n_samples,), optional
			The initial weights of each sample in the matrix. If nothing is
			passed in then each sample is assumed to be the same weight.
			Default is None.

		Returns
		-------
		None
		"""

		cdef int n = len(items), d = len(items[0])
		cdef numpy.ndarray items_ndarray = _check_input( items, self.keymap )
		cdef numpy.ndarray weights_ndarray

		if weights is None:
			weights_ndarray = numpy.ones( items.shape[0] )
		else:
			weights_ndarray = numpy.array( weights )

		cdef double* items_ptr = <double*> items_ndarray.data
		cdef double* weights_ptr = <double*> weights_ndarray.data

		return self._summarize(items_ptr, weights_ptr, n)

	cdef double _summarize( self, double* items, double* weights, SIZE_t n ) nogil:
		cdef double* r = <double*> calloc(self.n*n, sizeof(double))
		cdef int i, j
		cdef double total, logp, log_probability_sum = 0.0, p

		for i in range(n):
			total = NEGINF

			for j in range(self.n):
				if self.d == 1:
					logp = (<Distribution> self.distributions_ptr[j])._log_probability(items[i])
				else:
					logp = (<Distribution> self.distributions_ptr[j])._mv_log_probability(items+i*self.d)

				r[j*n + i] = logp + self.weights_ptr[j]
				total = pair_lse(total, r[j*n + i])

			for j in range( self.n ):
				r[j*n + i] = cexp(r[j*n + i] - total)
				self.summaries_ptr[j] += r[j*n + i]

			log_probability_sum += total 

		for j in range( self.n ):
			(<Distribution> self.distributions_ptr[j])._summarize(items, &r[j*n], n)

		free(r)
		return log_probability_sum

	def from_summaries( self, inertia=0.0 ):
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

		if self.summaries_ndarray.sum() == 0:
			return

		self.summaries_ndarray /= self.summaries_ndarray.sum()

		for i, distribution in enumerate( self.distributions ):
			distribution.from_summaries( inertia )
			self.weights[i] = _log( self.summaries_ndarray[i] )
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

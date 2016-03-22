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

def log_probability( model, samples ):
	'''Return the log probability of samples given a model.'''
	
	return sum( map( model.log_probability, samples ) )

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

	n_components : int, optional
		If a callable is passed into distributions then this is the number
		of components to initialize using the kmeans++ algorithm.


	Attributes
	----------
	distributions : array-like, shape (n_components,)
		The component distribution objects.

	weights : array-like, shape (n_components,)
		The learned prior weight of each object


	Examples
	--------
	>>> from pomegranate import *
	>>> clf = GeneralMixtureModel([NormalDistributon(5, 2), NormalDistribution(1, 1)])
	>>> clf.log_probability(5)
	-2.304562194038089
	>>> clf.predict_proba([5, 7, 1])
	array([[ 0.99932952,  0.00067048],
	       [ 0.99999995,  0.00000005],
	       [ 0.06337894,  0.93662106]])
	>>> clf.fit([1, 5, 7, 8, 2])
	>>> clf.predict_proba([5, 7, 1])
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
	cdef public numpy.ndarray weights
	cdef numpy.ndarray summaries_ndarray
	cdef void** distributions_ptr
	cdef double* weights_ptr
	cdef double* summaries_ptr
	cdef int n

	def __init__( self, distributions, weights=None, n_components=None ):
		"""Take in a list of initial distributions."""

		if weights is None:
			# Weight everything 1 if no weights specified
			weights = numpy.ones_like(distributions, dtype=float) / len( distributions )
		else:
			# Force whatever we have to be a Numpy array
			weights = numpy.asarray(weights) / weights.sum()

		self.weights = numpy.log( weights )
		self.weights_ptr = <double*> self.weights.data

		self.distributions = numpy.array( distributions )
		self.distributions_ptr = <void**> self.distributions.data
		
		self.summaries_ndarray = numpy.zeros_like(weights, dtype='float64')
		self.summaries_ptr = <double*> self.summaries_ndarray.data

		self.n = len(distributions)
		self.d = distributions[0].d

	def sample( self, n=1 ):
		"""Generate a sample from the model.

		First, randomly select a component weighted by the prior probability,
		Then, use the sample method from that component to generate a sample.
		
		Parameters
		----------
		n : int, optional
			The number of samples to generate.

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
			d = numpy.random.choice( self.distributions, p=numpy.exp(self.weights), size=n )
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
		log_probability : array-like, shape (n_samples, n_components)
			The normalized log probability log P(M|D) for each sample. This is
			the probability that the sample was generated from each component.
		"""

		return numpy.array( self._predict_log_proba( numpy.array(items) ) )

	cdef double [:,:] _predict_log_proba( self, numpy.ndarray items ):
		cdef int m = len( self.distributions ), n = items.shape[0]
		cdef double [:] priors = self.weights
		cdef double [:,:] r = numpy.empty((n, m))
		cdef double r_sum 
		cdef int i, j
		cdef Distribution d

		for i in range(n):
			r_sum = NEGINF

			for j in range(m):
				d = self.distributions[j]
				r[i, j] = d.log_probability(items[i]) + priors[j]
				r_sum = pair_lse(r_sum, r[i, j])

			for j in range(m):
				r[i, j] = r[i, j] - r_sum

		return r

	def predict( self, items ):
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

		return self.predict_log_proba( items ).argmax( axis=1 )

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

		inertia : double, optional
			The weight of the previous parameters of the model. The new
			parameters will roughly be old_param*inertia + new_param*(1-inertia), 
			so an inertia of 0 means ignore the old parameters, whereas an
			inertia of 1 means ignore the new parameters.

		stop_threshold : double, optional, positive
			The threshold at which EM will terminate for the improvement of
			the model. If the model does not improve its fit of the data by
			a log probability of 0.1 then terminate.

		max_iterations : int, optional, positive
			The maximum number of iterations to run EM for. If this limit is
			hit then it will terminate training, regardless of how well the
			model is improving per iteration.

		verbose : bool, optional
			Whether or not to print out improvement information over iterations.

		Returns
		-------
		improvement : double
			The total improvement in log probability P(D|M)
		"""

		initial_log_probability_sum = log_probability( self, items )
		last_log_probability_sum = initial_log_probability_sum
		iteration, improvement = 0, INF 

		while improvement > stop_threshold and iteration < max_iterations:
			self.summarize( items, weights )
			self.from_summaries( inertia )

			trained_log_probability_sum = log_probability( self, items )
			improvement = trained_log_probability_sum - last_log_probability_sum 

			if verbose:
				print( "Improvement: {}".format( improvement ) )

			iteration += 1
			last_log_probability_sum = trained_log_probability_sum

		return trained_log_probability_sum - initial_log_probability_sum

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

		Returns
		-------
		None
		"""

		items = numpy.array(items, dtype='float64')

		if weights is None:
			weights = numpy.ones( items.shape[0] )
		else:
			weights = numpy.array( weights )

		r = self.predict_proba( items )

		for i, distribution in enumerate( self.distributions ):
			distribution.summarize( items, weights=r[:,i]*weights )
			self.summaries_ndarray[i] += r[:,i].sum()

	cdef void _summarize( self, double* items, double* weights, SIZE_t n ) nogil:
		cdef double* r = <double*> calloc( self.n * n, sizeof(double) )
		cdef int i, j
		cdef double total

		for i in range( n ):
			total = 0.0

			for j in range( self.n ):
				if self.d == 1:
					r[j*n + i] = ( <Distribution> self.distributions_ptr[j] )._log_probability( items[i] )
				else:
					r[j*n + i] = ( <Distribution> self.distributions_ptr[j] )._mv_log_probability( items+i*self.d )

				r[j*n + i] = cexp( r[j*n + i] + self.weights_ptr[j] )
				self.summaries_ptr[j] += r[j*n + i]
				total += r[j*n + i]

			for j in range( self.n ):
				r[j*n + i] = weights[i] * r[j*n + i] / total

		for j in range( self.n ):
			( <Distribution> self.distributions_ptr[j] )._summarize( items, &r[j*n], n )

		free(r)

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
			inertia of 1 means ignore the new parameters.

		Returns
		-------
		None
		"""

		self.summaries_ndarray /= self.summaries_ndarray.sum()

		for i, distribution in enumerate( self.distributions ):
			distribution.from_summaries( inertia )
			self.weights[i] = _log( self.summaries_ndarray[i] )
			self.summaries_ndarray[i] = 0.

	def to_json( self, separators=(',', ' : '), indent=4 ):
		"""Serialize the model to a JSON.

		Parameters
		----------
		separators : tuple, optional 
			The two separaters to pass to the json.dumps function for formatting.

		indent : int, optional
			The indentation to use at each level. Passed to json.dumps for
			formatting.
		
		Returns
		-------
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
		"""

		d = json.loads( s )

		distributions = [ Distribution.from_json( json.dumps(j) ) for j in d['distributions'] ] 

		model = GeneralMixtureModel( distributions, numpy.array( d['weights'] ) )
		return model
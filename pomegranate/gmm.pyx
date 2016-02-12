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
	'''
	Return the log probability of samples given a model.
	'''

	return sum( map( model.log_probability, samples ) )

cdef class GeneralMixtureModel( Distribution ):
	"""
	A General Mixture Model. Can be a mixture of any distributions, as long
	as they are all the same dimensionality, have a log_probability method,
	and a from_sample method.
	"""

	cdef public numpy.ndarray distributions
	cdef public numpy.ndarray weights 
	cdef void** distributions_ptr
	cdef double* weights_ptr
	cdef int n

	def __init__( self, distributions, weights=None ):
		"""
		Take in a list of MultivariateDistributions to be optimized.
		"""

		if weights is None:
			# Weight everything 1 if no weights specified
			weights = numpy.ones_like(distributions, dtype=float) / len( distributions )
		else:
			# Force whatever we have to be a Numpy array
			weights = numpy.asarray(weights) / weights.sum()

		self.weights = numpy.log( weights )
		self.weights_ptr = <double*> self.weights.data
		self.distributions = numpy.array( distributions )
		self.summaries = []
		self.n = len(distributions)
		self.d = distributions[0].d
		self.distributions_ptr = <void**> self.distributions.data

	def sample( self ):
		"""
		Sample a point under this mixture by first selecting a component of the
		mixture based on their weights, and then sampling that component.
		"""

		d = numpy.random.choice( self.distributions, p=numpy.exp(self.weights) )
		return d.sample()

	cpdef double log_probability( self, point ):
		"""
		Calculate the probability of a point given the model. The probability
		of a point is the sum of the probabilities of each distribution.
		"""

		cdef int i, n = len( self.distributions )
		cdef double log_probability_sum = NEGINF
		cdef double log_probability
		cdef Distribution d

		for i in xrange( n ):
			d = self.distributions[i]
			log_probability = d.log_probability( point ) + self.weights[i]
			log_probability_sum = pair_lse( log_probability_sum,log_probability )

		return log_probability_sum

	cdef double _log_probability( self, double symbol ) nogil:
		"""
		Calculate the probability of a point in one dimension under the model,
		assuming a one dimensional model.
		"""

		cdef int i
		cdef double log_probability_sum = NEGINF
		cdef double log_probability

		for i in range( self.n ):
			log_probability = ( <Distribution> self.distributions_ptr[i] )._log_probability( symbol ) + self.weights_ptr[i]
			log_probability_sum = pair_lse( log_probability_sum, log_probability )

		return log_probability_sum

	cdef double _mv_log_probability( self, double* symbol ) nogil:
		"""
		Calculate the probability of a point in multiple dimensions under the model,
		assuming multiple dimensions.
		"""

		cdef int i
		cdef double log_probability_sum = NEGINF
		cdef double log_probability

		for i in range( self.n ):
			log_probability = ( <Distribution> self.distributions_ptr[i] )._mv_log_probability( symbol ) + self.weights_ptr[i]
			log_probability_sum = pair_lse( log_probability_sum, log_probability )

		return log_probability_sum


	def predict_proba( self, items ):
		"""sklearn wrapper for the probability of each component for each point."""
		
		return numpy.exp( self.predict_log_proba( items ) )

	def predict_log_proba( self, items ):
		"""sklearn wrapper for the log probability of each component for each point."""

		return self.posterior( items )

	def posterior( self, items ):
		"""Return the posterior log probability of each point under each distribution."""

		return numpy.array( self._posterior( items ) )

	cdef double [:,:] _posterior( self, numpy.ndarray items ):
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
				r[i, j] = d.log_probability( items[i] )
				r[i, j] += priors[j]
				r_sum = pair_lse( r_sum, r[i, j] )

			for j in range(m):
				r[i, j] = r[i, j] - r_sum

		return r

	def predict( self, items ):
		"""sklearn wrapper for posterior method."""

		return self.maximum_a_posteriori( items )

	def maximum_a_posteriori( self, items ):
		"""
		Return the most likely distribution given the posterior
		matrix. 
		"""

		return self.posterior( items ).argmax( axis=1 )

	def fit( self, items, weights=None, stop_threshold=0.1, max_iterations=1e8,
		verbose=False ):
		"""sklearn wrapper for train method."""

		return self.train( items, weights, stop_threshold, max_iterations,
			verbose )

	def train( self, items, weights=None, stop_threshold=0.1, max_iterations=1e8,
		verbose=False ):
		"""
		Take in a list of data points and their respective weights. These are
		most likely uniformly weighted, but the option exists if you want to
		add a second layer of weights on top of the ones learned in the
		expectation step.
		"""

		if weights is None:
			weights = numpy.ones(items.shape[0], dtype=numpy.float64)
		else:
			weights = numpy.array(weights, dtype=numpy.float64)

		initial_log_probability_sum = log_probability( self, items )
		last_log_probability_sum = initial_log_probability_sum
		iteration, improvement = 0, INF 

		while improvement > stop_threshold and iteration < max_iterations:
			# The responsibility matrix
			r = self.predict_proba( items )
			r_sum = r.sum()

			# Update the distribution based on the responsibility matrix
			for i, distribution in enumerate( self.distributions ):
				distribution.fit( items, weights=r[:,i]*weights )
				self.weights[i] = _log( r[:,i].sum() / r_sum )

			trained_log_probability_sum = log_probability( self, items )
			improvement = trained_log_probability_sum - last_log_probability_sum 

			if verbose:
				print( "Improvement: {}".format( improvement ) )

			iteration += 1
			last_log_probability_sum = trained_log_probability_sum

		return trained_log_probability_sum - initial_log_probability_sum

	cdef void _summarize( self, double* items, double* weights, SIZE_t n ) nogil:
		"""
		Summarization function for one step of EM.
		"""

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
				total += r[j*n + i]

			for j in range( self.n ):
				r[j*n + i] = weights[i] * r[j*n + i] / total

		for j in range( self.n ):
			( <Distribution> self.distributions_ptr[j] )._summarize( items, &r[j*n], n )

	def from_summaries( self, inertia=0.0 ):
		"""
		Update all distributions from the summaries.
		"""

		for distribution in self.distributions:
			distribution.from_summaries( inertia )

	def to_json( self, separators=(',', ' : '), indent=4 ):
		"""
		Write out the HMM to JSON format, recursively including state and
		distribution information.
		"""
		
		model = { 
					'class' : 'GeneralMixtureModel',
					'distributions'  : [ json.loads( dist.to_json() ) for dist in self.distributions ],
					'weights' : self.weights.tolist()
				}

		return json.dumps( model, separators=separators, indent=indent )

	@classmethod
	def from_json( cls, s, verbose=False ):
		"""
		Read a GMM from the given JSON, build the model, and bake it.
		"""

		d = json.loads( s )

		distributions = [ Distribution.from_json( json.dumps(j) ) for j in d['distributions'] ] 

		model = GeneralMixtureModel( distributions, numpy.array( d['weights'] ) )
		return model
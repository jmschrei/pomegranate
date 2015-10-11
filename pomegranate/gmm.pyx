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

# Define some useful constants
DEF NEGINF = float("-inf")
DEF INF = float("inf")

def log_probability( model, samples ):
	'''
	Return the log probability of samples given a model.
	'''

	return sum( map( model.log_probability, samples ) )

def weight_set( items, weights ):
	"""
	Set the weights to a numpy array of whatever is passed in, or an array of
	1's if nothing is passed in.
	"""

	items = numpy.array(items, dtype=numpy.float64)
	if weights is None:
		# Weight everything 1 if no weights specified
		weights = numpy.ones(items.shape[0], dtype=numpy.float64)
	else:
		# Force whatever we have to be a Numpy array
		weights = numpy.array(weights, dtype=numpy.float64)
	
	return items, weights

cdef class GeneralMixtureModel:
	"""
	A General Mixture Model. Can be a mixture of any distributions, as long
	as they are all the same dimensionality, have a log_probability method,
	and a from_sample method.
	"""

	cdef public list distributions
	cdef list summaries
	cdef public numpy.ndarray weights 

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
		self.distributions = distributions
		self.summaries = []

	def log_probability( self, point ):
		"""
		Calculate the probability of a point given the model. The probability
		of a point is the sum of the probabilities of each distribution.
		"""

		return self._log_probability( numpy.array( point ) )

	cdef double _log_probability( self, numpy.ndarray point ):
		"""
		Cython optimized function for calculating log probabilities.
		"""

		cdef int n=len(self.distributions), i=0
		cdef double log_probability_sum=NEGINF, log_probability
		cdef Distribution d

		for i in xrange( n ):
			d = self.distributions[i]
			log_probability = d.log_probability( point ) + self.weights[i]
			log_probability_sum = pair_lse( log_probability_sum,
											log_probability )

		return log_probability_sum

	def predict_proba( self, items ):
		"""sklearn wrapper for the probability of each component for each point."""
		
		return numpy.exp( self.predict_log_proba( items ) )

	def predict_log_proba( self, items ):
		"""sklearn wrapper for the log probability of each component for each point."""

		return self.posterior( items )

	def posterior( self, items ):
		"""Return the posterior probability of each point under each distribution."""

		return numpy.array( self._posterior( items, items.shape[0] ) )

	cdef double [:,:] _posterior( self, items, int n ):
		cdef int m = len( self.distributions )
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

		items, weights = weight_set( items, weights )

		initial_log_probability_sum = log_probability( self, items )
		last_log_probability_sum = initial_log_probability_sum
		iteration, improvement = 0, INF 

		priors = self.weights

		while improvement > stop_threshold and iteration < max_iterations:
			# The responsibility matrix
			r = self.posterior( items )

			# Update the distribution based on the responsibility matrix
			for i, distribution in enumerate( self.distributions ):
				distribution.train( items, weights=r[:,i]*weights )
				priors[i] = _log( r[:,i].sum() / r.sum() )

			trained_log_probability_sum = log_probability( self, items )
			improvement = last_log_probability_sum - trained_log_probability_sum
			last_log_probability_sum = trained_log_probability_sum

			if verbose:
				print( "Improvement: {}".format( improvement ) )

			iteration += 1

		self.weights = priors
		return initial_log_probability_sum - last_log_probability_sum

	def to_json( self ):
		"""
		Write out the HMM to JSON format, recursively including state and
		distribution information.
		"""
		
		model = { 
					'class' : 'GeneralMixtureModel',
					'distributions'  : map( str, self.distributions ),
					'weights' : list( self.weights )
				}

		return json.dumps( model, separators=(',', ' : '), indent=4 )

	@classmethod
	def from_json( cls, s, verbose=False ):
		"""
		Read a GMM from the given JSON, build the model, and bake it.
		"""

		# Load a dictionary from a JSON formatted string
		d = json.loads( s )

		distributions = [ Distribution.from_json( j ) for j in d['distributions'] ] 
		# Make a new generic HMM
		model = GeneralMixtureModel( distributions, numpy.array( d['weights'] ) )
		return model
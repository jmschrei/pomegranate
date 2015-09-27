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

cdef class GeneralMixtureModel:
	"""
	A General Mixture Model. Can be a mixture of any distributions, as long
	as they are all the same dimensionality, have a log_probability method,
	and a from_sample method.
	"""

	cdef public list distributions
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

		self.weights = weights
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

		cdef n=len(self.distributions), i=0
		cdef double log_probability_sum=NEGINF, log_probability
		cdef Distribution d

		for i in xrange( n ):
			d = self.distributions[i]
			log_probability = d.log_probability( point ) + _log( self.weights[i] )
			log_probability_sum = pair_lse( log_probability_sum,
											log_probability )

		return log_probability_sum

	def posterior( self, items ):
		"""
		Return the posterior probability of each distribution given the data.
		"""

		n, m = len( items ), len( self.distributions )
		priors = self.weights
		r = numpy.zeros( (n, m) ) 

		for i, item in enumerate( items ):
			# Counter for summation over the row
			r_sum = NEGINF

			# Calculate the log probability of the point over each distribution
			for j, distribution in enumerate( self.distributions ):
				# Calculate the log probability of the item under the distribution
				r[i, j] = distribution.log_probability( item )

				# Add the weights of the model
				r[i, j] += priors[j]

				# Add to the summation
				r_sum = pair_lse( r_sum, r[i, j] )

			# Normalize the row
			for j in xrange( m ):
				r[i, j] = r[i, j] - r_sum

		return r

	def maximum_a_posteriori( self, items ):
		"""
		Return the most likely distribution given the posterior
		matrix. 
		"""

		posterior = self.posterior( items )
		return numpy.argmax( axis=1 )

	def train( self, items, weights=None, stop_threshold=0.1, max_iterations=1e8,
		diagonal=False, verbose=False, inertia=None ):
		"""
		Take in a list of data points and their respective weights. These are
		most likely uniformly weighted, but the option exists if you want to
		add a second layer of weights on top of the ones learned in the
		expectation step.
		"""

		weights = numpy.array( weights ) or numpy.ones_like( items )
		n = len( items )
		m = len( self.distributions )
		last_log_probability_sum = log_probability( self, items )

		iteration, improvement = 0, INF
		priors = numpy.log( self.weights )

		while improvement > stop_threshold and iteration < max_iterations:
			# The responsibility matrix
			r = self.posterior( items )

			# Update the distribution based on the responsibility matrix
			for i, distribution in enumerate( self.distributions ):
				distribution.train( items, weights=r[:,i]*weights )
				priors[i] = r[:,i].sum() / r.sum()

			trained_log_probability_sum = log_probability( self, items )
			improvement = trained_log_probability_sum - last_log_probability_sum
			last_log_probability_sum = trained_log_probability_sum

			if verbose:
				print( "Improvement: {}".format( improvement ) )

			iteration += 1

		self.weights = priors

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
		Read a HMM from the given JSON, build the model, and bake it.
		"""

		# Load a dictionary from a JSON formatted string
		d = json.loads( s )

		distributions = [ Distribution.from_json( j ) for j in d['distributions'] ] 
		# Make a new generic HMM
		model = GeneralMixtureModel( distributions, numpy.array( d['weights'] ) )
		return model
# distributions.pyx
# Contact: Jacob Schreiber ( jmschreiber91@gmail.com )

cimport cython
from cython.view cimport array as cvarray
from libc.math cimport log as clog, sqrt as csqrt, exp as cexp
import math, random, itertools as it, sys, bisect, json
import networkx
import scipy.stats, scipy.sparse, scipy.special, scipy.linalg

if sys.version_info[0] > 2:
	# Set up for Python 3
	from functools import reduce
	xrange = range
	izip = zip
else:
	izip = it.izip

import numpy
cimport numpy

import matplotlib.pyplot as plt

cimport utils
from utils cimport *

# Define some useful constants
DEF NEGINF = float("-inf")
DEF INF = float("inf")
DEF SQRT_2_PI = 2.50662827463

def log(value):
	"""
	Return the natural log of the given value, or - infinity if the value is 0.
	Can handle both scalar floats and numpy arrays.
	"""

	if isinstance( value, numpy.ndarray ):
		to_return = numpy.zeros(( value.shape ))
		to_return[ value > 0 ] = numpy.log( value[ value > 0 ] )
		to_return[ value == 0 ] = NEGINF
		return to_return
	return _log( value )

cdef class Distribution:
	"""
	Represents a probability distribution over whatever the HMM you're making is
	supposed to emit. Ought to be subclassed and have log_probability(), 
	sample(), and from_sample() overridden. Distribution.name should be 
	overridden and replaced with a unique name for the distribution type. The 
	distribution should be registered by calling register() on the derived 
	class, so that Distribution.read() can read it. Any distribution parameters 
	need to be floats stored in self.parameters, so they will be properly 
	written by write().
	"""

	def __init__( self ):
		"""
		Make a new Distribution with the given parameters. All parameters must 
		be floats.
		
		Storing parameters in self.parameters instead of e.g. self.mean on the 
		one hand makes distribution code ugly, because we don't get to call them
		self.mean. On the other hand, it means we don't have to override the 
		serialization code for every derived class.
		"""

		self.name = "Distribution"
		self.frozen = False
		self.parameters = []
		self.summaries = []

	def __str__( self ):
		"""
		Represent this distribution in JSON.
		"""
		
		return self.to_json()

	def __repr__( self ):
		"""
		Represent this distribution in the same format as string.
		"""

		return self.to_json()

	def marginal( self, *args, **kwargs ):
		'''
		The marginal of a non-conditional distribution is just the actual
		distribution, so return that.
		'''

		return self

	def copy( self ):
		"""
		Return a copy of this distribution, untied. 
		"""

		return self.__class__( *self.parameters ) 

	def freeze( self ):
		"""
		Freeze the distribution, preventing training from changing any of the
		parameters of the distribution.
		"""

		self.frozen = True

	def thaw( self ):
		"""
		Thaw the distribution, allowing training to change the parameters of
		the distribution again.
		"""

		self.frozen = False 

	def log_probability( self, symbol ):
		"""
		Return the log probability of the given symbol under this distribution.
		"""
		
		raise NotImplementedError

	def sample( self ):
		"""
		Return a random item sampled from this distribution.
		"""
		
		raise NotImplementedError
	
	def train( self, *args, **kwargs ):
		"""
		A wrapper for from_sample in order to homogenize calls more.
		"""

		self.from_sample( *args, **kwargs )

	def from_sample( self, items, weights=None, inertia=0.0 ):
		"""
		Set the parameters of this Distribution to maximize the likelihood of 
		the given sample. Items holds some sort of sequence. If weights is 
		specified, it holds a sequence of value to weight each item by.
		"""
		
		if self.frozen == True:
			return
		raise NotImplementedError

	def summarize( self, items, weights=None ):
		"""
		Summarize the incoming items into a summary statistic to be used to
		update the parameters upon usage of the `from_summaries` method. By
		default, this will simply store the items and weights into a large
		sample, and call the `from_sample` method.
		"""

		# If no previously stored summaries, just store the incoming data
		if len( self.summaries ) == 0:
			self.summaries = [ items, weights ]

		# Otherwise, append the items and weights
		else:
			prior_items, prior_weights = self.summaries
			items = numpy.concatenate( [prior_items, items] )

			# If even one summary lacks weights, then weights can't be assigned
			# to any of the points.
			if weights is not None:
				weights = numpy.concatenate( [prior_weights, weights] )

			self.summaries = [ items, weights ]

	def from_summaries( self, inertia=0.0 ):
		"""
		Update the parameters of the distribution based on the summaries stored
		previously. 
		"""

		# If the distribution is frozen, don't bother with any calculation
		if self.frozen == True:
			return

		self.train( *self.summaries, inertia=inertia )
		self.summaries = []

	def plot( self, n=1000, **kwargs ):
		"""
		Plot the distribution by sampling from it n times and plotting
		either a histogram or a kernel density. 
		"""

		samples = [ self.sample() for i in xrange( n ) ]
		plt.hist( samples, **kwargs )

	def to_json( self ):
		"""
		Convert the distribution to JSON format.
		"""

		return json.dumps( {
								'class' : 'Distribution',
								'name'  : self.name,
								'parameters' : self.parameters,
								'frozen' : self.frozen
						   }, separators=(',', ' : ' ), indent=4 )

	@classmethod
	def from_json( cls, s ):
		"""
		Read in a JSON and produce an appropriate distribution.
		"""

		d = json.loads( s )

		# Put in some simple checking before we evaluate the distribution
		# component to ensure arbitrary malicious code isn't being evaluated.
		if ' ' in d['class'] or 'Distribution' not in d['class']:
			raise SyntaxError( "Distribution object attempting to read invalid object." )

		dist = eval( "{}( [0], 0 )".format( d['name'] ) )
		dist.parameters = d['parameters']
		dist.frozen = d['frozen']
		return dist

cdef class UniformDistribution( Distribution ):
	"""
	A uniform distribution between two values.
	"""

	def __init__( self, start, end, frozen=False ):
		"""
		Make a new Uniform distribution over floats between start and end, 
		inclusive. Start and end must not be equal.
		"""
		
		# Store the parameters
		self.parameters = [start, end]
		self.summaries = []
		self.name = "UniformDistribution"
		self.frozen = frozen
		
	def log_probability( self, symbol ):
		"""
		What's the probability of the given float under this distribution?
		"""
		
		return self._log_probability( self.parameters[0], self.parameters[1], symbol )

	cdef double _log_probability( self, double a, double b, double symbol ):
		if symbol == a and symbol == b:
			return 0
		if symbol >= a and symbol <= b:
			return _log( 1.0 / ( b - a ) )
		return NEGINF
			
	def sample( self ):
		"""
		Sample from this uniform distribution and return the value sampled.
		"""
		
		return random.uniform(self.parameters[0], self.parameters[1])
		
	def from_sample (self, items, weights=None, inertia=0.0 ):
		"""
		Set the parameters of this Distribution to maximize the likelihood of 
		the given sample. Items holds some sort of sequence. If weights is 
		specified, it holds a sequence of value to weight each item by.
		"""
		
		# If the distribution is frozen, don't bother with any calculation
		if self.frozen == True:
			return

		# Calculate weights. If none are provided, give uniform weights
		if weights is None:
			weights = numpy.ones_like( items )
		else:
			weights = numpy.asarray( weights )

		if weights.sum() == 0:
			return
		
		if len(items) == 0:
			# No sample, so just ignore it and keep our old parameters.
			return
		
		# The ML uniform distribution is just min to max. Weights don't matter
		# for this.
		# Calculate the new parameters, respecting inertia, with an inertia
		# of 0 being completely replacing the parameters, and an inertia of
		# 1 being to ignore new training data.
		prior_min, prior_max = self.parameters
		self.parameters[0] = prior_min*inertia + numpy.min(items)*(1-inertia)
		self.parameters[1] = prior_max*inertia + numpy.max(items)*(1-inertia)

	def summarize( self, items, weights=None ):
		"""
		Take in a series of items and their weights and reduce it down to a
		summary statistic to be used in training later.
		"""

		if weights is None:
			weights = numpy.ones_like( items )
		else:
			weights = numpy.asarray( weights )

		if weights.sum() == 0:
			return

		if len( items ) == 0:
			# No sample, so just ignore it and keep our own parameters.
			return

		items = numpy.asarray( items )

		# Record the min and max, which are the summary statistics for a
		# uniform distribution.
		self.summaries.append([ items.min(), items.max() ])
		
	def from_summaries( self, inertia=0.0 ):
		"""
		Takes in a series of summaries, consisting of the minimum and maximum
		of a sample, and determine the global minimum and maximum.
		"""

		# If the distribution is frozen, don't bother with any calculation
		if self.frozen == True:
			return

		summaries = numpy.asarray( self.summaries )

		# Load the prior parameters
		prior_min, prior_max = self.parameters

		# Calculate the new parameters, respecting inertia, with an inertia
		# of 0 being completely replacing the parameters, and an inertia of
		# 1 being to ignore new training data.
		self.parameters = [ prior_min*inertia + summaries[:,0].min()*(1-inertia), 
							prior_max*inertia + summaries[:,1].max()*(1-inertia) ]
		self.summaries = []

cdef class NormalDistribution( Distribution ):
	"""
	A normal distribution based on a mean and standard deviation.
	"""

	def __init__( self, mean, std, frozen=False ):
		"""
		Make a new Normal distribution with the given mean mean and standard 
		deviation std.
		"""
		
		# Store the parameters
		self.parameters = [mean, std]
		self.summaries = []
		self.name = "NormalDistribution"
		self.frozen = frozen

	def log_probability( self, symbol, epsilon=1E-4 ):
		"""
		What's the probability of the given float under this distribution?
		
		For distributions with 0 std, epsilon is the distance within which to 
		consider things equal to the mean.
		"""

		return self._log_probability( symbol, epsilon )

	cdef double _log_probability( self, double symbol, double epsilon ):
		"""
		Do the actual math here.
		"""

		cdef double mu = self.parameters[0], sigma = self.parameters[1]
		if sigma == 0.0:
			if abs( symbol - mu ) < epsilon:
				return 0
			else:
				return NEGINF
  
		return _log( 1.0 / ( sigma * SQRT_2_PI ) ) - ((symbol - mu) ** 2) /\
			(2 * sigma ** 2)

	def sample( self ):
		"""
		Sample from this normal distribution and return the value sampled.
		"""
		
		# This uses the same parameterization
		return random.normalvariate(*self.parameters)
		
	def from_sample( self, items, weights=None, inertia=0.0, min_std=0.01 ):
		"""
		Set the parameters of this Distribution to maximize the likelihood of 
		the given sample. Items holds some sort of sequence. If weights is 
		specified, it holds a sequence of value to weight each item by.
		
		min_std specifieds a lower limit on the learned standard deviation.
		"""

		# If the distribution is frozen, don't bother with any calculation
		if len(items) == 0 or self.frozen == True:
			# No sample, so just ignore it and keep our old parameters.
			return

		# Make it be a numpy array
		items = numpy.asarray(items)
		
		if weights is None:
			# Weight everything 1 if no weights specified
			weights = numpy.ones_like(items)
		else:
			# Force whatever we have to be a Numpy array
			weights = numpy.asarray(weights)
		
		if weights.sum() == 0:
			# Since negative weights are banned, we must have no data.
			# Don't change the parameters at all.
			return
		# The ML uniform distribution is just sample mean and sample std.
		# But we have to weight them. average does weighted mean for us, but 
		# weighted std requires a trick from Stack Overflow.
		# http://stackoverflow.com/a/2415343/402891
		# Take the mean
		mean = numpy.average(items, weights=weights)

		if len(weights[weights != 0]) > 1:
			# We want to do the std too, but only if more than one thing has a 
			# nonzero weight
			# First find the variance
			variance = (numpy.dot(items ** 2 - mean ** 2, weights) / 
				weights.sum())
				
			if variance >= 0:
				std = csqrt(variance)
			else:
				# May have a small negative variance on accident. Ignore and set
				# to 0.
				std = 0
		else:
			# Only one data point, can't update std
			std = self.parameters[1]    
		
		# Enforce min std
		std = max( numpy.array([std, min_std]) )
		
		# Calculate the new parameters, respecting inertia, with an inertia
		# of 0 being completely replacing the parameters, and an inertia of
		# 1 being to ignore new training data.
		prior_mean, prior_std = self.parameters
		self.parameters = [ prior_mean*inertia + mean*(1-inertia), 
							prior_std*inertia + std*(1-inertia) ]

	def summarize( self, items, weights=None ):
		"""
		Take in a series of items and their weights and reduce it down to a
		summary statistic to be used in training later.
		"""

		if numpy.sum( weights ) == 0:
			return

		items = numpy.asarray( items )

		# Calculate weights. If none are provided, give uniform weights
		if weights is None:
			weights = numpy.ones_like( items )
		else:
			weights = numpy.asarray( weights )

		if weights.sum() == 0:
			return

		# Save the mean and variance, the summary statistics for a normal
		# distribution.
		mean = numpy.average( items, weights=weights )
		variance = numpy.dot( items**2 - mean**2, weights ) / weights.sum()

		# Append the mean, variance, and sum of the weights to give the weights
		# of these statistics.
		self.summaries.append( [ mean, variance, weights.sum() ] )
		
	def from_summaries( self, inertia=0.0 ):
		"""
		Takes in a series of summaries, represented as a mean, a variance, and
		a weight, and updates the underlying distribution. Notes on how to do
		this for a Gaussian distribution were taken from here:
		http://math.stackexchange.com/questions/453113/how-to-merge-two-gaussians
		"""

		# If no summaries stored or the summary is frozen, don't do anything.
		if len( self.summaries ) == 0 or self.frozen == True:
			return

		summaries = numpy.asarray( self.summaries )

		# Calculate the new mean and variance.
		mean = numpy.average( summaries[:,0], weights=summaries[:,2] )
		variance = numpy.sum( [(v+m**2)*w for m, v, w in summaries] ) \
			/ summaries[:,2].sum() - mean**2

		if variance >= 0:
			std = csqrt(variance)
		else:
			std = 0

		# Get the previous parameters.
		prior_mean, prior_std = self.parameters

		# Calculate the new parameters, respecting inertia, with an inertia
		# of 0 being completely replacing the parameters, and an inertia of
		# 1 being to ignore new training data.
		self.parameters = [ prior_mean*inertia + mean*(1-inertia),
							prior_std*inertia + std*(1-inertia) ]
		self.summaries = []

cdef class LogNormalDistribution( Distribution ):
	"""
	Represents a lognormal distribution over non-negative floats.
	"""

	def __init__( self, mu, sigma, frozen=False ):
		"""
		Make a new lognormal distribution. The parameters are the mu and sigma
		of the normal distribution, which is the the exponential of the log
		normal distribution.
		"""
		self.parameters = [ mu, sigma ]
		self.summaries = []
		self.name = "LogNormalDistribution"
		self.frozen = frozen

	def log_probability( self, symbol ):
		"""
		What's the probability of the given float under this distribution?
		"""

		return self._log_probability( symbol )

	cdef double _log_probability( self, double symbol ):
		"""
		Actually perform the calculations here, in the Cython-optimized
		function.
		"""

		mu, sigma = self.parameters
		return -clog( symbol * sigma * SQRT_2_PI ) \
			- 0.5 * ( ( clog( symbol ) - mu ) / sigma ) ** 2

	def sample( self ):
		"""
		Return a sample from this distribution.
		"""

		return numpy.random.lognormal( *self.parameters )

	def from_sample( self, items, weights=None, inertia=0.0, min_std=0.01 ):
		"""
		Set the parameters of this distribution to maximize the likelihood of
		the given samples. Items hold some sort of sequence over floats. If
		weights is specified, hold a sequence of values to weight each item by.
		"""

		# If the distribution is frozen, don't bother with any calculation
		if len(items) == 0 or self.frozen == True:
			# No sample, so just ignore it and keep our old parameters.
			return

		# Make it be a numpy array
		items = numpy.asarray(items)
		
		if weights is None:
			# Weight everything 1 if no weights specified
			weights = numpy.ones_like(items)
		else:
			# Force whatever we have to be a Numpy array
			weights = numpy.asarray(weights)
		
		if weights.sum() == 0:
			# Since negative weights are banned, we must have no data.
			# Don't change the parameters at all.
			return

		# The ML uniform distribution is just the mean of the log of the samples
		# and sample std the variance of the log of the samples.
		# But we have to weight them. average does weighted mean for us, but 
		# weighted std requires a trick from Stack Overflow.
		# http://stackoverflow.com/a/2415343/402891
		# Take the mean
		mean = numpy.average( numpy.log(items), weights=weights)

		if len(weights[weights != 0]) > 1:
			# We want to do the std too, but only if more than one thing has a 
			# nonzero weight
			# First find the variance
			variance = ( numpy.dot( numpy.log(items) ** 2 - mean ** 2, weights) / 
				weights.sum() )
				
			if variance >= 0:
				std = csqrt(variance)
			else:
				# May have a small negative variance on accident. Ignore and set
				# to 0.
				std = 0
		else:
			# Only one data point, can't update std
			std = self.parameters[1]    
		
		# Enforce min std
		std = max( numpy.array([std, min_std]) )
		
		# Calculate the new parameters, respecting inertia, with an inertia
		# of 0 being completely replacing the parameters, and an inertia of
		# 1 being to ignore new training data.
		prior_mean, prior_std = self.parameters
		self.parameters = [ prior_mean*inertia + mean*(1-inertia), 
							prior_std*inertia + std*(1-inertia) ]

	def summarize( self, items, weights=None ):
		"""
		Take in a series of items and their weights and reduce it down to a
		summary statistic to be used in training later.
		"""

		# If no weights are specified, use uniform weights.
		if weights is None:
			weights = numpy.ones_like( items )
		else:
			weights = numpy.asarray( weights )

		if weights.sum() == 0:
			return

		# Calculate the mean and variance, which are the summary statistics
		# for a log-normal distribution.
		mean = numpy.average( numpy.log(items), weights=weights )
		variance = numpy.dot( numpy.log(items)**2 - mean**2, weights ) / weights.sum()
		
		# Save the summary statistics and the weights.
		self.summaries.append( [ mean, variance, weights.sum() ] )
		

	def from_summaries( self, inertia=0.0 ):
		"""
		Takes in a series of summaries, represented as a mean, a variance, and
		a weight, and updates the underlying distribution. Notes on how to do
		this for a Gaussian distribution were taken from here:
		http://math.stackexchange.com/questions/453113/how-to-merge-two-gaussians
		"""

		# If no summaries are provided or the distribution is frozen, 
		# don't do anything.
		if len( self.summaries ) == 0 or self.frozen == True:
			return

		summaries = numpy.asarray( self.summaries )

		# Calculate the mean and variance from the summary statistics.
		mean = numpy.average( summaries[:,0], weights=summaries[:,2] )
		variance = numpy.sum( [(v+m**2)*w for m, v, w in summaries] ) \
			/ summaries[:,2].sum() - mean**2

		if variance >= 0:
			std = csqrt(variance)
		else:
			std = 0

		# Load the previous parameters
		prior_mean, prior_std = self.parameters

		# Calculate the new parameters, respecting inertia, with an inertia
		# of 0 being completely replacing the parameters, and an inertia of
		# 1 being to ignore new training data.
		self.parameters = [ prior_mean*inertia + mean*(1-inertia), 
							prior_std*inertia + std*(1-inertia) ]
		self.summaries = []

cdef class ExtremeValueDistribution( Distribution ):
	"""
	Represent a generalized extreme value distribution over floats.
	"""

	def __init__( self, mu, sigma, epsilon, frozen=True ):
		"""
		Make a new extreme value distribution, where mu is the location
		parameter, sigma is the scale parameter, and epsilon is the shape
		parameter. 
		"""

		self.parameters = [ float(mu), float(sigma), float(epsilon) ]
		self.name = "ExtremeValueDistribution"
		self.frozen = frozen

	def log_probability( self, symbol ):
		"""
		What's the probability of the given float under this distribution?
		"""

		return self._log_probability( symbol )

	cdef double _log_probability( self, double symbol ):
		"""
		Actually perform the calculations here, in the Cython-optimized
		function.
		"""

		mu, sigma, epsilon = self.parameters
		t = ( symbol - mu ) / sigma
		if epsilon == 0:
			return -clog( sigma ) - t - cexp( -t )
		return -clog( sigma ) + clog( 1 + epsilon * t ) * (-1. / epsilon - 1) \
			- ( 1 + epsilon * t ) ** ( -1. / epsilon )

cdef class ExponentialDistribution( Distribution ):
	"""
	Represents an exponential distribution on non-negative floats.
	"""
	
	def __init__( self, rate, frozen=False ):
		"""
		Make a new inverse gamma distribution. The parameter is called "rate" 
		because lambda is taken.
		"""

		self.parameters = [rate]
		self.summaries = []
		self.name = "ExponentialDistribution"
		self.frozen = frozen

	def log_probability( self, symbol ):
		"""
		What's the probability of the given float under this distribution?
		"""
		
		return _log(self.parameters[0]) - self.parameters[0] * symbol
		
	def sample( self ):
		"""
		Sample from this exponential distribution and return the value
		sampled.
		"""
		
		return random.expovariate(*self.parameters)
		
	def from_sample( self, items, weights=None, inertia=0.0 ):
		"""
		Set the parameters of this Distribution to maximize the likelihood of 
		the given sample. Items holds some sort of sequence. If weights is 
		specified, it holds a sequence of value to weight each item by.
		"""
		
		# If the distribution is frozen, don't bother with any calculation
		if len(items) == 0 or self.frozen == True:
			# No sample, so just ignore it and keep our old parameters.
			return
		
		# Make it be a numpy array
		items = numpy.asarray(items)
		
		if weights is None:
			# Weight everything 1 if no weights specified
			weights = numpy.ones_like(items)
		else:
			# Force whatever we have to be a Numpy array
			weights = numpy.asarray(weights)
		
		if weights.sum() == 0:
			# Since negative weights are banned, we must have no data.
			# Don't change the parameters at all.
			return
		
		# Parameter MLE = 1/sample mean, easy to weight
		# Compute the weighted mean
		weighted_mean = numpy.average(items, weights=weights)
		
		# Calculate the new parameters, respecting inertia, with an inertia
		# of 0 being completely replacing the parameters, and an inertia of
		# 1 being to ignore new training data.
		prior_rate = self.parameters[0]
		rate = 1.0 / weighted_mean

		self.parameters[0] = prior_rate*inertia + rate*(1-inertia)

	def summarize( self, items, weights=None ):
		"""
		Take in a series of items and their weights and reduce it down to a
		summary statistic to be used in training later.
		"""

		items = numpy.asarray( items )

		# Either store the weights, or assign uniform weights to each item
		if weights is None:
			weights = numpy.ones_like( items )
		else:
			weights = numpy.asarray( weights )

		if weights.sum() == 0:
			return

		# Calculate the summary statistic, which in this case is the mean.
		mean = numpy.average( items, weights=weights )
		self.summaries.append( [ mean, weights.sum() ] )

	def from_summaries( self, inertia=0.0 ):
		"""
		Takes in a series of summaries, represented as a mean, a variance, and
		a weight, and updates the underlying distribution. Notes on how to do
		this for a Gaussian distribution were taken from here:
		http://math.stackexchange.com/questions/453113/how-to-merge-two-gaussians
		"""

		# If no summaries or the distribution is frozen, do nothing.
		if len( self.summaries ) == 0 or self.frozen == True:
			return

		summaries = numpy.asarray( self.summaries )

		# Calculate the new parameter from the summary statistics.
		mean = numpy.average( summaries[:,0], weights=summaries[:,1] )

		# Get the parameters
		prior_rate = self.parameters[0]
		rate = 1.0 / mean

		# Calculate the new parameters, respecting inertia, with an inertia
		# of 0 being completely replacing the parameters, and an inertia of
		# 1 being to ignore new training data.
		self.parameters[0] = prior_rate*inertia + rate*(1-inertia)
		self.summaries = []

cdef class BetaDistribution( Distribution ):
	"""
	This distribution represents a beta distribution, parameterized using
	alpha/beta, which are both shape parameters. ML estimation is done
	"""

	def __init__( self, alpha, beta, frozen=False ):
		"""
		Make a new beta distribution. Both alpha and beta are both shape
		parameters.
		"""

		self.parameters = [alpha, beta]
		self.summaries = []
		self.name = "BetaDistribution"
		self.frozen = frozen

	def log_probability( self, symbol ):
		"""
		What is the probability of the given float under this distribution?
		"""

		a, b = self.parameters
		return ( _log(math.lgamma(a+b)) - _log(math.lgamma(a)) - 
			_log(math.lgamma(b)) + (a-1)*_log(symbol) +
			(b-1)*_log(1.-symbol) ) 

	def sample( self ):
		"""
		Return a random sample from the beta distribution.
		"""

		return random.betavariate( self.parameters[0], self.parameters[1] )

	def from_sample( self, items, weights=None, inertia=0.0 ):
		"""
		Take in a series of binary data and update the parameters. Simply put,
		the posteriors are:

		alpha = alpha + new_successes
		beta = beta + new_failures
		"""

		# If the distribution is frozen, don't bother with any calculation
		if len(items) == 0 or self.frozen == True:
			# No sample, so just ignore it and keep our old parameters.
			return
		
		# Make it be a numpy array
		items = numpy.asarray(items)
		
		if weights is None:
			# Weight everything 1 if no weights specified
			weights = numpy.ones_like(items)
		else:
			# Force whatever we have to be a Numpy array
			weights = numpy.asarray(weights)
		
		if weights.sum() == 0:
			# Since negative weights are banned, we must have no data.
			# Don't change the parameters at all.
			return

		successes = items.dot( weights )
		failures = (1-items).dot( weights )

		alpha = self.parameters[0]*inertia + successes*(1-inertia)
		beta = self.parameters[1]*inertia + failures*(1-inertia)

		self.parameters = [ alpha, beta ]

	def summarize( self, items, weights=None ):
		"""
		Summarize some new data and store it to summaries.
		"""

		# If the distribution is frozen, don't bother with any calculation
		if len(items) == 0 or self.frozen == True:
			# No sample, so just ignore it and keep our old parameters.
			return
		
		# Make it be a numpy array
		items = numpy.asarray(items)
		
		if weights is None:
			# Weight everything 1 if no weights specified
			weights = numpy.ones_like(items)
		else:
			# Force whatever we have to be a Numpy array
			weights = numpy.asarray(weights)
		
		if weights.sum() == 0:
			# Since negative weights are banned, we must have no data.
			# Don't change the parameters at all.
			return

		successes = items.dot( weights )
		failures = (1-items).dot( weights )

		self.summaries.append( [ successes, failures ] )

	def from_summaries( self, inertia=0.0 ):
		"""
		Use the summaries in order to update the distribution.
		"""

		summaries = numpy.array( self.summaries )

		alpha = self.parameters[0]*inertia + summaries[:,0].sum( axis=0 )*(1-inertia)
		beta = self.parameters[1]*inertia + summaries[:,1].sum( axis=0 )*(1-inertia)

		self.parameters = [ alpha, beta ]
		self.summaries = []


cdef class GammaDistribution( Distribution ):
	"""
	This distribution represents a gamma distribution, parameterized in the 
	alpha/beta (shape/rate) parameterization. ML estimation for a gamma 
	distribution, taking into account weights on the data, is nontrivial, and I 
	was unable to find a good theoretical source for how to do it, so I have 
	cobbled together a solution here from less-reputable sources.
	"""
	
	def __init__( self, alpha, beta, frozen=False ):
		"""
		Make a new gamma distribution. Alpha is the shape parameter and beta is 
		the rate parameter.
		"""
		
		self.parameters = [alpha, beta]
		self.summaries = []
		self.name = "GammaDistribution"
		self.frozen = frozen
		
	def log_probability( self, symbol ):
		"""
		What's the probability of the given float under this distribution?
		"""
		
		# Gamma pdf from Wikipedia (and stats class)
		return (_log(self.parameters[1]) * self.parameters[0] - 
			math.lgamma(self.parameters[0]) + 
			_log(symbol) * (self.parameters[0] - 1) - 
			self.parameters[1] * symbol)
		
	def sample( self ):
		"""
		Sample from this gamma distribution and return the value sampled.
		"""
		
		# We have a handy sample from gamma function. Unfortunately, while we 
		# use the alpha, beta parameterization, and this function uses the 
		# alpha, beta parameterization, our alpha/beta are shape/rate, while its
		# alpha/beta are shape/scale. So we have to mess with the parameters.
		return random.gammavariate(self.parameters[0], 1.0 / self.parameters[1])
		
	def from_sample( self, items, weights=None, inertia=0.0, epsilon=1E-9, 
		iteration_limit=1000 ):
		"""
		Set the parameters of this Distribution to maximize the likelihood of 
		the given sample. Items holds some sort of sequence. If weights is 
		specified, it holds a sequence of value to weight each item by.
		
		In the Gamma case, likelihood maximization is necesarily numerical, and 
		the extension to weighted values is not trivially obvious. The algorithm
		used here includes a Newton-Raphson step for shape parameter estimation,
		and analytical calculation of the rate parameter. The extension to 
		weights is constructed using vital information found way down at the 
		bottom of an Experts Exchange page.
		
		Newton-Raphson continues until the change in the parameter is less than 
		epsilon, or until iteration_limit is reached
		
		See:
		http://en.wikipedia.org/wiki/Gamma_distribution
		http://www.experts-exchange.com/Other/Math_Science/Q_23943764.html
		"""
		
		# If the distribution is frozen, don't bother with any calculation
		if len(items) == 0 or self.frozen == True:
			# No sample, so just ignore it and keep our old parameters.
			return

		# Make it be a numpy array
		items = numpy.asarray(items)
		
		if weights is None:
			# Weight everything 1 if no weights specified
			weights = numpy.ones_like(items)
		else:
			# Force whatever we have to be a Numpy array
			weights = numpy.asarray(weights)

		if weights.sum() == 0:
			# Since negative weights are banned, we must have no data.
			# Don't change the parameters at all.
			return

		# First, do Newton-Raphson for shape parameter.
		
		# Calculate the sufficient statistic s, which is the log of the average 
		# minus the average log. When computing the average log, we weight 
		# outside the log function. (In retrospect, this is actually pretty 
		# obvious.)
		statistic = (log(numpy.average(items, weights=weights)) - 
			numpy.average(log(items), weights=weights))

		# Start our Newton-Raphson at what Wikipedia claims a 1969 paper claims 
		# is a good approximation.
		# Really, start with new_shape set, and shape set to be far away from it
		shape = float("inf")
		
		if statistic != 0:
			# Not going to have a divide by 0 problem here, so use the good
			# estimate
			new_shape =  (3 - statistic + csqrt((statistic - 3) ** 2 + 24 * 
				statistic)) / (12 * statistic)
		if statistic == 0 or new_shape <= 0:
			# Try the current shape parameter
			new_shape = self.parameters[0]

		# Count the iterations we take
		iteration = 0
			
		# Now do the update loop.
		# We need the digamma (gamma derivative over gamma) and trigamma 
		# (digamma derivative) functions. Luckily, scipy.special.polygamma(0, x)
		# is the digamma function (0th derivative of the digamma), and 
		# scipy.special.polygamma(1, x) is the trigamma function.
		while abs(shape - new_shape) > epsilon and iteration < iteration_limit:
			shape = new_shape
			
			new_shape = shape - (log(shape) - 
				scipy.special.polygamma(0, shape) -
				statistic) / (1.0 / shape - scipy.special.polygamma(1, shape))
			
			# Don't let shape escape from valid values
			if abs(new_shape) == float("inf") or new_shape == 0:
				# Hack the shape parameter so we don't stop the loop if we land
				# near it.
				shape = new_shape
				
				# Re-start at some random place.
				new_shape = random.random()
				
			iteration += 1
			
		# Might as well grab the new value
		shape = new_shape
				
		# Now our iterative estimation of the shape parameter has converged.
		# Calculate the rate parameter
		rate = 1.0 / (1.0 / (shape * weights.sum()) * items.dot(weights) )

		# Get the previous parameters
		prior_shape, prior_rate = self.parameters

		# Calculate the new parameters, respecting inertia, with an inertia
		# of 0 being completely replacing the parameters, and an inertia of
		# 1 being to ignore new training data.
		self.parameters = [ prior_shape*inertia + shape*(1-inertia), 
							prior_rate*inertia + rate*(1-inertia) ]    

	def summarize( self, items, weights=None ):
		"""
		Take in a series of items and their weights and reduce it down to a
		summary statistic to be used in training later.
		"""

		if len(items) == 0:
			# No sample, so just ignore it and keep our old parameters.
			return

		# Make it be a numpy array
		items = numpy.asarray(items)
		
		if weights is None:
			# Weight everything 1 if no weights specified
			weights = numpy.ones_like(items)
		else:
			# Force whatever we have to be a Numpy array
			weights = numpy.asarray(weights)

		if weights.sum() == 0:
			# Since negative weights are banned, we must have no data.
			# Don't change the parameters at all.
			return

		# Save the weighted average of the items, and the weighted average of
		# the log of the items.
		self.summaries.append( [ numpy.average( items, weights=weights ),
								 numpy.average( log(items), weights=weights ),
								 items.dot( weights ),
								 weights.sum() ] )

	def from_summaries( self, inertia=0.0, epsilon=1E-9, 
		iteration_limit=1000 ):
		'''
		Set the parameters of this Distribution to maximize the likelihood of 
		the given sample given the summaries which have been stored.
		
		In the Gamma case, likelihood maximization is necesarily numerical, and 
		the extension to weighted values is not trivially obvious. The algorithm
		used here includes a Newton-Raphson step for shape parameter estimation,
		and analytical calculation of the rate parameter. The extension to 
		weights is constructed using vital information found way down at the 
		bottom of an Experts Exchange page.
		
		Newton-Raphson continues until the change in the parameter is less than 
		epsilon, or until iteration_limit is reached

		See:
		http://en.wikipedia.org/wiki/Gamma_distribution
		http://www.experts-exchange.com/Other/Math_Science/Q_23943764.html
		'''

		# If the distribution is frozen, don't bother with any calculation
		if len(self.summaries) == 0 or self.frozen == True:
			return

		# First, do Newton-Raphson for shape parameter.
		
		# Calculate the sufficient statistic s, which is the log of the average 
		# minus the average log. When computing the average log, we weight 
		# outside the log function. (In retrospect, this is actually pretty 
		# obvious.)
		summaries = numpy.array( self.summaries )

		statistic = math.log( numpy.average( summaries[:,0], 
											 weights=summaries[:,3] ) ) - \
					numpy.average( summaries[:,1], 
								   weights=summaries[:,3] )

		# Start our Newton-Raphson at what Wikipedia claims a 1969 paper claims 
		# is a good approximation.
		# Really, start with new_shape set, and shape set to be far away from it
		shape = float("inf")
		
		if statistic != 0:
			# Not going to have a divide by 0 problem here, so use the good
			# estimate
			new_shape =  (3 - statistic + csqrt((statistic - 3) ** 2 + 24 * 
				statistic)) / (12 * statistic)
		if statistic == 0 or new_shape <= 0:
			# Try the current shape parameter
			new_shape = self.parameters[0]

		# Count the iterations we take
		iteration = 0
			
		# Now do the update loop.
		# We need the digamma (gamma derivative over gamma) and trigamma 
		# (digamma derivative) functions. Luckily, scipy.special.polygamma(0, x)
		# is the digamma function (0th derivative of the digamma), and 
		# scipy.special.polygamma(1, x) is the trigamma function.
		while abs(shape - new_shape) > epsilon and iteration < iteration_limit:
			shape = new_shape
			
			new_shape = shape - (_log(shape) - 
				scipy.special.polygamma(0, shape) -
				statistic) / (1.0 / shape - scipy.special.polygamma(1, shape))
			
			# Don't let shape escape from valid values
			if abs(new_shape) == float("inf") or new_shape == 0:
				# Hack the shape parameter so we don't stop the loop if we land
				# near it.
				shape = new_shape
				
				# Re-start at some random place.
				new_shape = random.random()
				
			iteration += 1
			
		# Might as well grab the new value
		shape = new_shape
				
		# Now our iterative estimation of the shape parameter has converged.
		# Calculate the rate parameter
		rate = 1.0 / (1.0 / (shape * summaries[:,3].sum()) * \
			numpy.sum( summaries[:,2] ) )

		# Get the previous parameters
		prior_shape, prior_rate = self.parameters

		# Calculate the new parameters, respecting inertia, with an inertia
		# of 0 being completely replacing the parameters, and an inertia of
		# 1 being to ignore new training data.
		self.parameters = [ prior_shape*inertia + shape*(1-inertia), 
							prior_rate*inertia + rate*(1-inertia) ]
		self.summaries = []  	

cdef class InverseGammaDistribution( GammaDistribution ):
	"""
	This distribution represents an inverse gamma distribution (1/the RV ~ gamma
	with the same parameters). A distribution over non-negative floats.
	
	We cheat and don't have to do much work by inheriting from the 
	GammaDistribution.
	"""
	
	def __init__( self, alpha, beta, frozen=False ):
		"""
		Make a new inverse gamma distribution. Alpha is the shape parameter and 
		beta is the scale parameter.
		"""
		
		self.parameters = [alpha, beta]
		self.summaries = []
		self.name = "InverseGammaDistribution"
		self.frozen = frozen
		
	def log_probability( self, symbol ):
		"""
		What's the probability of the given float under this distribution?
		"""
		
		return super(InverseGammaDistribution, self).log_probability(
			1.0 / symbol)
			
	def sample( self ):
		"""
		Sample from this inverse gamma distribution and return the value
		sampled.
		"""
		
		# Invert the sample from the gamma distribution.
		return 1.0 / super( InverseGammaDistribution, self ).sample()
		
	def from_sample( self, items, weights=None, inertia=0.0 ):
		"""
		Set the parameters of this Distribution to maximize the likelihood of 
		the given sample. Items holds some sort of sequence. If weights is 
		specified, it holds a sequence of value to weight each item by.
		"""
		
		# Fit the gamma distribution on the inverted items.
		super( InverseGammaDistribution, self ).from_sample( 1.0 / 
			numpy.asarray( items ), weights=weights, inertia=inertia )

	def summarize( self, items, weights=None ):
		"""
		Take in a series of items and their weights and reduce it down to a
		summary statistic to be used in training later.
		"""

		super( InverseGammaDistribution, self ).summarize( 
			1.0 / numpy.asarray( items ),  weights=weights )

	def from_summaries( self, inertia=0.0, epsilon=1E-9, iteration_limit=1000 ):
		'''
		Update the parameters based on the summaries stored.
		'''

		super( InverseGammaDistribution, self ).from_summaries(
			epsilon=epsilon, iteration_limit=iteration_limit, inertia=inertia )

cdef class DiscreteDistribution(Distribution):
	"""
	A discrete distribution, made up of characters and their probabilities,
	assuming that these probabilities will sum to 1.0. 
	"""
	
	def __init__( self, characters, frozen=False ):
		"""
		Make a new discrete distribution with a dictionary of discrete
		characters and their probabilities, checking to see that these
		sum to 1.0. Each discrete character can be modelled as a
		Bernoulli distribution.
		"""
		
		# Store the parameters
		self.parameters = [ characters ]
		self.summaries = [ {}, 0 ]
		self.name = "DiscreteDistribution"
		self.frozen = frozen

	def __len__( self ):
		"""
		Return the length of the underlying dictionary
		"""

		return len( self.parameters[0] )

	def __mul__( self, other ):
		"""
		Multiply this by another discrete distribution sharing the same keys.
		This returns the probability of a key being the product of the probability
		of the key under both distributions normalized by the sum over all keys
		of the product that key given the distributions.
		"""

		assert set( self.keys() ) == set( other.keys() )

		# Create the new dictionary
		p, total = {}, NEGINF

		# Go through each key and multiply them
		for key in self.keys():
			p[key] = self.log_probability( key ) + other.log_probability( key )
			total = pair_lse( total, p[key] )

		for key in self.keys():
			p[key] -= total 
			p[key] = math.exp( p[key] )

		return DiscreteDistribution( p )

	def equals( self, other ):
		"""
		Determine if this distribution is equal to another discrete distribution
		to 4 digits.
		"""

		# If we're not even comparing to a discrete distribution, then it cannot
		# be the same.
		if not isinstance( other, DiscreteDistribution ):
			return False

		# If the key sets aren't the same, it cannot be the same and will cause
		# crashing in the next step.
		if set( self.keys() ) != set( other.keys() ):
			return False

		# Go through and make sure the log probabilities for each key are the same.
		for key in self.keys():
			if round( self.log_probability( key ), 12 ) != round( other.log_probability( key ), 12 ):
				return False

		return True

	def clamp( self, key ):
		"""
		Return a distribution clamped to a particular value.
		"""

		d = { k : 0. if k != key else 1. for k in self.keys() }
		return DiscreteDistribution( d )

	def keys( self ):
		"""
		Return the keys of the underlying dictionary.
		"""

		return self.parameters[0].keys()

	def items( self ):
		"""
		Return items of the underlying dictionary,
		"""

		return self.parameters[0].items()

	def values( self ):
		"""
		Return values of the underlying dictionary.
		"""

		return self.parameters[0].values()

	def log_probability( self, symbol ):
		"""
		What's the probability of the given symbol under this distribution?
		Simply the log probability value given at initiation. If the symbol
		is not part of the discrete distribution, return a log probability
		of NEGINF.
		"""

		return _log( self.parameters[0].get( symbol, 0 ) )
			
	def sample( self ):
		"""
		Sample randomly from the discrete distribution, returning the character
		which was randomly generated.
		"""
		
		rand = random.random()
		for key, value in self.parameters[0].items():
			if value >= rand:
				return key
			rand -= value
	
	def from_sample( self, items, weights=None, inertia=0.0 ):
		"""
		Takes in an iterable representing samples from a distribution and
		turn it into a discrete distribution. If no weights are provided,
		each sample is weighted equally. If weights are provided, they are
		normalized to sum to 1 and used.
		"""

		# If the distribution is frozen, don't bother with any calculation
		if len( items ) == 0 or self.frozen == True:
			return

		n = len( items )

		# Normalize weights, or assign uniform probabilities
		if weights is None:
			weights = numpy.ones( n ) / n
		else:
			weights = numpy.array(weights) / numpy.sum(weights)

		# Sum the weights seen for each character
		characters = {}
		for character, weight in izip( items, weights ):
			try:
				characters[character] += weight
			except KeyError:
				characters[character] = weight

		# Adjust the new weights by the inertia
		for character, weight in characters.items():
			characters[character] = weight * (1-inertia)

		# Adjust the old weights by the inertia
		prior_characters = self.parameters[0]
		for character, weight in prior_characters.items():
			try:
				characters[character] += weight * inertia
			except KeyError:
				characters[character] = weight * inertia

		self.parameters = [ characters ]

	def summarize( self, items, weights=None ):
		"""
		Take in a series of items and their weights and reduce it down to a
		summary statistic to be used in training later.
		"""

		n = len( items )
		if weights is None:
			weights = numpy.ones( n )
		else:
			weights = numpy.asarray( weights )

		if weights.sum() == 0:
			return

		characters = self.summaries[0]
		for character, weight in izip( items, weights ):
			try:
				characters[character] += weight
			except KeyError:
				characters[character] = weight

		self.summaries[0] = characters
		self.summaries[1] += weights.sum()

	def from_summaries( self, inertia=0.0 ):
		"""
		Takes in a series of summaries and merge them.
		"""

		# If the distribution is frozen, don't bother with any calculation
		if len( self.summaries ) == 0 or self.frozen == True:
			return

		# Unpack the variables
		prior_characters = self.parameters[0]
		characters, total_weight = self.summaries 

		# Scale the characters by both the total number of weights and by
		# the inertia.
		for character, prob in characters.items():
			characters[character] = ( prob / total_weight ) * (1-inertia)

		# Adjust the old weights by the inertia
		if inertia > 0.0:
			for character, weight in prior_characters.items():
				try:
					characters[character] += weight * inertia
				except KeyError:
					characters[character] = weight * inertia

		self.parameters = [ characters ]
		self.summaries = [ {}, 0 ]


cdef class LambdaDistribution(Distribution):
	"""
	A distribution which takes in an arbitrary lambda function, and returns
	probabilities associated with whatever that function gives. For example...

	func = lambda x: log(1) if 2 > x > 1 else log(0)
	distribution = LambdaDistribution( func )
	print distribution.log_probability( 1 ) # 1
	print distribution.log_probability( -100 ) # 0

	This assumes the lambda function returns the log probability, not the
	untransformed probability.
	"""
	
	def __init__(self, lambda_funct, frozen=True ):
		"""
		Takes in a lambda function and stores it. This function should return
		the log probability of seeing a certain input.
		"""

		# Store the parameters
		self.parameters = [lambda_funct]
		self.name = "LambdaDistribution"
		self.frozen = frozen
		
	def log_probability(self, symbol):
		"""
		What's the probability of the given float under this distribution?
		"""

		return self.parameters[0](symbol)

cdef class GaussianKernelDensity( Distribution ):
	"""
	A quick way of storing points to represent a Gaussian kernel density in one
	dimension. Takes in the points at initialization, and calculates the log of
	the sum of the Gaussian distance of the new point from every other point.
	"""

	def __init__( self, points, bandwidth=1, weights=None, frozen=False ):
		"""
		Take in points, bandwidth, and appropriate weights. If no weights
		are provided, a uniform weight of 1/n is provided to each point.
		Weights are scaled so that they sum to 1. 
		"""

		points = numpy.asarray( points )
		n = len(points)
		
		if weights:
			weights = numpy.array(weights) / numpy.sum(weights)
		else:
			weights = numpy.ones( n ) / n 

		self.parameters = [ points, bandwidth, weights ]
		self.summaries = []
		self.name = "GaussianKernelDensity"
		self.frozen = frozen

	def log_probability( self, symbol ):
		"""
		What's the probability of a given float under this distribution? It's
		the sum of the distances of the symbol from every point stored in the
		density. Bandwidth is defined at the beginning. A wrapper for the
		cython function which does math.
		"""

		return self._log_probability( symbol )

	cdef double _log_probability( self, double symbol ):
		"""
		Actually calculate it here.
		"""
		cdef double bandwidth = self.parameters[1]
		cdef double mu, scalar = 1.0 / SQRT_2_PI
		cdef int i = 0, n = len(self.parameters[0])
		cdef double distribution_prob = 0, point_prob

		for i in xrange( n ):
			# Go through each point sequentially
			mu = self.parameters[0][i]

			# Calculate the probability under that point
			point_prob = scalar * \
				cexp( -0.5 * (( mu-symbol ) / bandwidth) ** 2 )

			# Scale that point according to the weight 
			distribution_prob += point_prob * self.parameters[2][i]

		# Return the log of the sum of the probabilities
		return _log( distribution_prob )

	def sample( self ):
		"""
		Generate a random sample from this distribution. This is done by first
		selecting a random point, weighted by weights if the points are weighted
		or uniformly if not, and then randomly sampling from that point's PDF.
		"""

		mu = numpy.random.choice( self.parameters[0], p=self.parameters[2] )
		return random.gauss( mu, self.parameters[1] )

	def from_sample( self, points, weights=None, inertia=0.0 ):
		"""
		Replace the points, allowing for inertia if specified.
		"""

		# If the distribution is frozen, don't bother with any calculation
		if self.frozen == True:
			return

		points = numpy.asarray( points )
		n = len(points)

		# Get the weights, or assign uniform weights
		if weights:
			weights = numpy.array(weights) / numpy.sum(weights)
		else:
			weights = numpy.ones( n ) / n 

		# If no inertia, get rid of the previous points
		if inertia == 0.0:
			self.parameters[0] = points
			self.parameters[2] = weights

		# Otherwise adjust weights appropriately
		else: 
			self.parameters[0] = numpy.concatenate( ( self.parameters[0],
													  points ) )
			self.parameters[2] = numpy.concatenate( ( self.parameters[2]*inertia,
													  weights*(1-inertia) ) )

cdef class UniformKernelDensity( Distribution ):
	"""
	A quick way of storing points to represent an Exponential kernel density in
	one dimension. Takes in points at initialization, and calculates the log of
	the sum of the Gaussian distances of the new point from every other point.
	"""

	def __init__( self, points, bandwidth=1, weights=None, frozen=False ):
		"""
		Take in points, bandwidth, and appropriate weights. If no weights
		are provided, a uniform weight of 1/n is provided to each point.
		Weights are scaled so that they sum to 1. 
		"""

		points = numpy.asarray( points )
		n = len(points)
		if weights:
			weights = numpy.array(weights) / numpy.sum(weights)
		else:
			weights = numpy.ones( n ) / n 

		self.parameters = [ points, bandwidth, weights ]
		self.summaries = []
		self.name = "UniformKernelDensity"
		self.frozen = frozen

	def log_probability( self, symbol ):
		"""
		What's the probability ofa given float under this distribution? It's
		the sum of the distances from the symbol calculated under individual
		exponential distributions. A wrapper for the cython function.
		"""

		return self._log_probability( symbol )

	cdef _log_probability( self, double symbol ):
		"""
		Actually do math here.
		"""

		cdef double mu
		cdef double distribution_prob=0, point_prob
		cdef int i = 0, n = len(self.parameters[0])

		for i in xrange( n ):
			# Go through each point sequentially
			mu = self.parameters[0][i]

			# The good thing about uniform distributions if that
			# you just need to check to make sure the point is within
			# a bandwidth.
			if abs( mu - symbol ) <= self.parameters[1]:
				point_prob = 1
			else:
				point_prob = 0

			# Properly weight the point before adding it to the sum
			distribution_prob += point_prob * self.parameters[2][i]

		# Return the log of the sum of probabilities
		return _log( distribution_prob )
	
	def sample( self ):
		"""
		Generate a random sample from this distribution. This is done by first
		selecting a random point, weighted by weights if the points are weighted
		or uniformly if not, and then randomly sampling from that point's PDF.
		"""

		mu = numpy.random.choice( self.parameters[0], p=self.parameters[2] )
		bandwidth = self.parameters[1]
		return random.uniform( mu-bandwidth, mu+bandwidth )

	def from_sample( self, points, weights=None, inertia=0.0 ):
		"""
		Replace the points, allowing for inertia if specified.
		"""

		# If the distribution is frozen, don't bother with any calculation
		if self.frozen == True:
			return

		points = numpy.asarray( points )
		n = len(points)

		# Get the weights, or assign uniform weights
		if weights:
			weights = numpy.array(weights) / numpy.sum(weights)
		else:
			weights = numpy.ones( n ) / n 

		# If no inertia, get rid of the previous points
		if inertia == 0.0:
			self.parameters[0] = points
			self.parameters[2] = weights

		# Otherwise adjust weights appropriately
		else: 
			self.parameters[0] = numpy.concatenate( ( self.parameters[0],
													  points ) )
			self.parameters[2] = numpy.concatenate( ( self.parameters[2]*inertia,
													  weights*(1-inertia) ) )

cdef class TriangleKernelDensity( Distribution ):
	"""
	A quick way of storing points to represent an Exponential kernel density in
	one dimension. Takes in points at initialization, and calculates the log of
	the sum of the Gaussian distances of the new point from every other point.
	"""

	def __init__( self, points, bandwidth=1, weights=None, frozen=False ):
		"""
		Take in points, bandwidth, and appropriate weights. If no weights
		are provided, a uniform weight of 1/n is provided to each point.
		Weights are scaled so that they sum to 1. 
		"""

		points = numpy.asarray( points )
		n = len(points)
		if weights:
			weights = numpy.array(weights) / numpy.sum(weights)
		else:
			weights = numpy.ones( n ) / n 

		self.parameters = [ points, bandwidth, weights ]
		self.summaries = []
		self.name = "TriangleKernelDensity"
		self.frozen = frozen

	def log_probability( self, symbol ):
		"""
		What's the probability of a given float under this distribution? It's
		the sum of the distances from the symbol calculated under individual
		exponential distributions. A wrapper for the cython function.
		""" 

		return self._log_probability( symbol )

	cdef double _log_probability( self, double symbol ):
		"""
		Actually do math here.
		"""

		cdef double bandwidth = self.parameters[1]
		cdef double mu
		cdef double distribution_prob=0, point_prob
		cdef int i = 0, n = len(self.parameters[0])

		for i in xrange( n ):
			# Go through each point sequentially
			mu = self.parameters[0][i]

			# Calculate the probability for each point
			point_prob = bandwidth - abs( mu - symbol ) 
			if point_prob < 0:
				point_prob = 0 

			# Properly weight the point before adding to the sum
			distribution_prob += point_prob * self.parameters[2][i]

		# Return the log of the sum of probabilities
		return _log( distribution_prob )

	def sample( self ):
		"""
		Generate a random sample from this distribution. This is done by first
		selecting a random point, weighted by weights if the points are weighted
		or uniformly if not, and then randomly sampling from that point's PDF.
		"""

		mu = numpy.random.choice( self.parameters[0], p=self.parameters[2] )
		bandwidth = self.parameters[1]
		return random.triangular( mu-bandwidth, mu+bandwidth, mu )

	def from_sample( self, points, weights=None, inertia=0.0 ):
		"""
		Replace the points, allowing for inertia if specified.
		"""

		# If the distribution is frozen, don't bother with any calculation
		if self.frozen == True:
			return

		points = numpy.asarray( points )
		n = len(points)

		# Get the weights, or assign uniform weights
		if weights:
			weights = numpy.array(weights) / numpy.sum(weights)
		else:
			weights = numpy.ones( n ) / n 

		# If no inertia, get rid of the previous points
		if inertia == 0.0:
			self.parameters[0] = points
			self.parameters[2] = weights

		# Otherwise adjust weights appropriately
		else: 
			self.parameters[0] = numpy.concatenate( ( self.parameters[0],
													  points ) )
			self.parameters[2] = numpy.concatenate( ( self.parameters[2]*inertia,
													  weights*(1-inertia) ) )

cdef class MixtureDistribution( Distribution ):
	"""
	Allows you to create an arbitrary mixture of distributions. There can be
	any number of distributions, include any permutation of types of
	distributions. Can also specify weights for the distributions.
	"""

	def __init__( self, distributions, weights=None, frozen=False ):
		"""
		Take in the distributions and appropriate weights. If no weights
		are provided, a uniform weight of 1/n is provided to each point.
		Weights are scaled so that they sum to 1. 
		"""
		n = len(distributions)
		if weights:
			weights = numpy.array( weights ) / numpy.sum( weights )
		else:
			weights = numpy.ones(n) / n

		self.parameters = [ distributions, weights ]
		self.name = "MixtureDistribution"
		self.frozen = frozen

	def __str__( self ):
		"""
		Return a string representation of this mixture.
		"""

		distributions, weights = self.parameters
		distributions = map( str, distributions )
		return "MixtureDistribution( {}, {} )".format(
			distributions, list(weights) ).replace( "'", "" )

	def log_probability( self, symbol ):
		"""
		What's the probability of a given float under this mixture? It's
		the log-sum-exp of the distances from the symbol calculated under all
		distributions. Currently in python, not cython, to allow for dovetyping
		of both numeric and not-necessarily-numeric distributions. 
		"""

		(d, w), n = self.parameters, len(self.parameters)
		return _log( numpy.sum([ cexp( d[i].log_probability(symbol) ) \
			* w[i] for i in xrange(n) ]) )

	def sample( self ):
		"""
		Sample from the mixture. First, choose a distribution to sample from
		according to the weights, then sample from that distribution. 
		"""

		i = random.random()
		for d, w in zip( *self.parameters ):
			if w > i:
				return d.sample()
			i -= w 

	def from_sample( self, items, weights=None ):
		"""
		Perform EM to estimate the parameters of each distribution
		which is a part of this mixture.
		"""

		if weights is None:
			weights = numpy.ones( len(items) )
		else:
			weights = numpy.asarray( weights )

		if weights.sum() == 0:
			return

		distributions, w = self.parameters
		n, k = len(items), len(distributions)

		# The responsibility matrix
		r = numpy.zeros( (n, k) )

		# Calculate the log probabilities of each p
		for i, distribution in enumerate( distributions ):
			for j, item in enumerate( items ):
				r[j, i] = distribution.log_probability( item )

		r = numpy.exp( r )

		# Turn these log probabilities into responsibilities by
		# normalizing on a row-by-row manner.
		for i in xrange( n ):
			r[i] = r[i] / r[i].sum()

		# Weight the responsibilities by the given weights
		for i in xrange( k ):
			r[:,i] = r[:,i]*weights

		# Update the emissions of each distribution
		for i, distribution in enumerate( distributions ):
			distribution.from_sample( items, weights=r[:,i] )

		# Update the weight of each distribution
		self.parameters[1] = r.sum( axis=0 ) / r.sum()

	def summarize( self, items, weights=None ):
		"""
		Performs the summary step of the EM algorithm to estimate
		parameters of each distribution which is a part of this mixture.
		"""

		if weights is None:
			weights = numpy.ones( len(items) )
		else:
			weights = numpy.asarray( weights )

		if weights.sum() == 0:
			return

		distributions, w = self.parameters
		n, k = len(items), len(distributions)

		# The responsibility matrix
		r = numpy.zeros( (n, k) )

		# Calculate the log probabilities of each p
		for i, distribution in enumerate( distributions ):
			for j, item in enumerate( items ):
				r[j, i] = distribution.log_probability( item )

		r = numpy.exp( r )

		# Turn these log probabilities into responsibilities by
		# normalizing on a row-by-row manner.
		for i in xrange( n ):
			r[i] = r[i] / r[i].sum()

		# Weight the responsibilities by the given weights
		for i in xrange( k ):
			r[:,i] = r[:,i]*weights

		# Save summary statistics on the emission distributions
		for i, distribution in enumerate( distributions ):
			distribution.summarize( items, weights=r[:,i]*weights )

		# Save summary statistics for weight updates
		self.summaries.append( r.sum( axis=0 ) / r.sum() )

	def from_summaries( self, inertia=0.0 ):
		"""
		Performs the actual update step for the EM algorithm.
		"""

		# If this distribution is frozen, don't do anything.
		if self.frozen == True:
			return

		# Update the emission distributions
		for d in self.parameters[0]:
			d.from_summaries( inertia=inertia )

		# Update the weights
		weights = numpy.array( self.summaries )
		self.parameters[1] = weights.sum( axis=0 ) / weights.sum()

cdef class MultivariateDistribution( Distribution ):
	"""
	An object to easily identify multivariate distributions such as tables.
	"""

	pass

cdef class IndependentComponentsDistribution( MultivariateDistribution ):
	"""
	Allows you to create a multivariate distribution, where each distribution
	is independent of the others. Distributions can be any type, such as
	having an exponential represent the duration of an event, and a normal
	represent the mean of that event. Observations must now be tuples of
	a length equal to the number of distributions passed in.

	s1 = MultivariateDistribution([ ExponentialDistribution( 0.1 ), 
									NormalDistribution( 5, 2 ) ])
	s1.log_probability( (5, 2 ) )
	"""

	def __init__( self, distributions, weights=None, frozen=False ):
		"""
		Take in the distributions and appropriate weights. If no weights
		are provided, a uniform weight of 1/n is provided to each point.
		Weights are scaled so that they sum to 1. 
		"""
		n = len(distributions)
		if weights:
			weights = numpy.array( weights )
		else:
			weights = numpy.ones(n)

		self.parameters = [ distributions, weights ]
		self.name = "MultivariateDistribution"
		self.frozen = frozen

	def __str__( self ):
		"""
		Return a string representation of the MultivariateDistribution.
		"""

		distributions = map( str, self.parameters[0] )
		return "MultivariateDistribution({})".format(
			distributions ).replace( "'", "" )

	def log_probability( self, symbol ):
		"""
		What's the probability of a given tuple under this mixture? It's the
		product of the probabilities of each symbol in the tuple under their
		respective distribution, which is the sum of the log probabilities.
		"""

		return sum( d.log_probability( obs )*w for d, obs, w in zip( 
			self.parameters[0], symbol, self.parameters[1] ) )

	def sample( self ):
		"""
		Sample from the mixture. First, choose a distribution to sample from
		according to the weights, then sample from that distribution. 
		"""

		return [ d.sample() for d in self.parameters[0] ]

	def from_sample( self, items, weights=None, inertia=0.0 ):
		"""
		Items are tuples, and so each distribution can be trained
		independently of each other. 
		"""

		# If the distribution is frozen, don't bother with any calculation
		if self.frozen == True:
			return

		items = numpy.asarray( items )

		for i, d in enumerate( self.parameters[0] ):
			d.from_sample( items[:,i], weights=weights, inertia=inertia )

	def summarize( self, items, weights=None ):
		"""
		Take in an array of items and reduce it down to summary statistics. For
		a multivariate distribution, this involves just passing the appropriate
		data down to the appropriate distributions.
		"""

		items = numpy.asarray( items )

		for i, d in enumerate( self.parameters[0] ):
			d.summarize( items[:,i], weights=weights )

	def from_summaries( self, inertia=0.0 ):
		"""
		Use the collected summary statistics in order to update the
		distributions.
		"""

		# If the distribution is frozen, don't bother with any calculation
		if self.frozen == True:
			return

		for d in self.parameters[0]:
			d.from_summaries( inertia=inertia )

cdef class MultivariateGaussianDistribution( MultivariateDistribution ):
	"""
	A multivariate gaussian distribution, with a mean vector and a covariance
	matrix. This is mostly a wrapper for scipy.stats.multivariate_gaussian.
	"""

	def __init__( self, means, covariance, frozen=False, diagonal=False ):
		"""
		Take in the mean vector and the covariance matrix. 
		"""

		self.parameters = [ numpy.array(means), numpy.array(covariance) ]
		self.name = "MultivariateGaussianDistribution"
		self.frozen = frozen
		self.diagonal = diagonal
		self.summaries = []

	def log_probability( self, symbol, min_std=0.01 ):
		"""
		What's the probability of a given tuple under this mixture? It's the
		product of the probabilities of each symbol in the tuple under their
		respective distribution, which is the sum of the log probabilities.
		"""

		# Taken from sklearn.mixture.gmm._log_multivariate_normal_density_full
		mean, covar = self.parameters
		d = mean.shape[0]

		try:
			cv_chol = scipy.linalg.cholesky(covar, lower=True)
		except scipy.linalg.LinAlgError:
			# The model is most probably stuck in a component with too
			# few observations, we need to reinitialize this components
			try:
				cv_chol = scipy.linalg.cholesky( covar + (min_std ** 2.) * numpy.eye(d), lower=True )
			except scipy.linalg.LinAlgError:
				raise ValueError("Covariance matrix must be symmetric positive-definite")

		cv_log_det = 2 * numpy.sum( numpy.log( numpy.diagonal(cv_chol) ) )
		cv_sol = scipy.linalg.solve_triangular( cv_chol, (symbol - mean).T, lower=True).T
		logp = - .5 * (numpy.sum(cv_sol ** 2) + d * numpy.log(2 * numpy.pi) + cv_log_det)
		return logp

	def sample( self ):
		"""
		Sample from the mixture. First, choose a distribution to sample from
		according to the weights, then sample from that distribution. 
		"""

		return scipy.stats.multivariate_normal.rvs(
			self.parameters[0], self.parameters[1] )

	def from_sample( self, items, weights=None, inertia=0.0, min_std=0.01, diagonal=None, l=0. ):
		"""
		Set the parameters of this Distribution to maximize the likelihood of 
		the given sample. Items holds some sort of sequence. If weights is 
		specified, it holds a sequence of value to weight each item by.
		"""

		# If the distribution is frozen, don't bother with any calculation
		if len(items) == 0 or self.frozen == True:
			# No sample, so just ignore it and keep our old parameters.
			return

		# Make it be a numpy array
		items = numpy.asarray(items)

		if weights is None:
			# Weight everything 1 if no weights specified
			weights = numpy.ones( items.shape[0], dtype=float ) / len( items )
		elif numpy.sum( weights ) == 0:
			# Since negative weights are banned, we must have no data.
			# Don't change parameters at all.
			return

		# Take the mean
		means = numpy.average( items, weights=weights, axis=0 )

		# Calculate the biased weight sample since we don't know the scaling
		# of the weights
		n, m = len(items), self.parameters[0].shape[0]

		cov = numpy.zeros( (m, m) )
		for i in xrange( n ):
			diff = numpy.matrix( items[i] - means )
			cov += weights[i] * numpy.array( diff.T.dot( diff ) )
		cov /= numpy.sum( weights )

        # We can shrink our estimates if needed to prevent getting a singular matrix
		cov = ( 1.-l )*cov + l*numpy.eye( cov.shape[0] )

		if diagonal == True or diagonal == None and self.diagonal == True:
			cov = numpy.diag( numpy.diag( cov ) )

		# Calculate the new parameters, respecting inertia, with an inertia
		# of 0 being completely replacing the parameters, and an inertia of
		# 1 being to ignore new training data.
		prior_means, prior_covs = self.parameters

		self.parameters = [ prior_means*inertia + means*(1-inertia), 
							prior_covs*inertia + cov*(1-inertia) ]

cdef class ConditionalProbabilityTable( MultivariateDistribution ):
	"""
	A conditional probability table, which is dependent on values from at
	least one previous distribution but up to as many as you want to
	encode for.
	"""

	def __init__( self, distribution, parents, hashes=None, n=None ):
		"""
		Take in the distribution represented as a list of lists, where each
		inner list represents a row.
		"""

		# If we're passed a formed distribution already, just store it. Otherwise
		# generate it from a table.
		if isinstance( distribution, numpy.ndarray ) and hashes and n:
			self.parameters = [ distribution, parents, hashes, n ]
		else:
			# Take the list of lists and invert it so that a numpy array represents
			# each column, since each column is a homogenous data type and we are
			# unsure if the distribution is over integers or strings.
			table = zip( *distribution )

			d_keys = list( set( table[-2] ) )
			d_map = { key: i for i, key in enumerate( d_keys )}

			# Create a mapping from values in the table to integers to be stored
			# in a compressed array
			hashes = hashes or [{ key: i for i, key in enumerate( parent.keys() ) } for parent in parents ] + [ d_map ]
			n = n or map( len, parents+[d_keys] )
			m = numpy.cumprod( [1] + n )

			values = numpy.zeros( len( distribution ) )

			# Go through each row and put it into the compressed array
			for row in distribution:
				i = sum( hashes[j][val]*m[j] for j, val in enumerate( row[:-1] ) )
				values[i] = _log( row[-1] )

			# Store all the information
			self.parameters = [ values, parents, hashes, n ]

	def __len__( self ):
		"""
		The length of the distribution is the number of keys.
		"""

		return len( self.keys() )

	def keys( self ):
		"""
		Return the keys of the probability distribution which has parents,
		the child variable.
		"""

		return self.parameters[2][-1].keys()

	def log_probability( self, value, parent_values ):
		"""
		Return the log probability of a value given some number of parent
		values, which can be a subset of the full number of parents. If this
		is the case, then marginalize over unknown values.
		"""

		# Unpack the parameters
		values, parents, hashes, n = self.parameters
		m = numpy.cumprod( [1]+n )

		# Assume parent values are given in the same order that parents
		# were specified
		i = sum( hashes[j][val]*m[j] for j, val in enumerate( parent_values ) )

		# Return the array element with that identity
		return values[i]
		
	def joint( self, neighbor_values=None ):
		"""
		This will turn a conditional probability table into a joint
		probability table. If the data is already a joint, it will likely
		mess up the data. It does so by scaling the parameters the probabilities
		by the parent distributions.
		"""

		# Unpack the parameters
		values, parents, hashes, n = self.parameters
		r_hashes = [ d.keys() for d in hashes ]
		m = numpy.cumprod( [1]+n )

		neighbor_values = neighbor_values or parents+[None]
		
		# If given a dictionary, then decode it
		if isinstance( neighbor_values, dict ):
			nv = [ None for i in xrange( len( neighbor_values)+1 ) ]

			# Go through each parent and find the appropriate value
			for i, parent in enumerate( parents ):
				nv[i] = neighbor_values.get( parent, None )

			if len(neighbor_values) == len(parents):
				# We've already gotten the value for this marginal, because it
				# was encoded as a separate factor.
				pass
			else:
				# Get values for this marginal
				nv[-1] = neighbor_values.get( self, None )

			neighbor_values = nv

		# The growing table
		table = []

		# Create the table row by row
		for keys in it.product( *[ xrange(i) for i in n ] ):
			i = sum( key*k for key, k in it.izip( keys, m ) )

			# Scale the probability by the weights on the marginals
			scaled_val = values[i]
			for j, k in enumerate( keys ):
				if neighbor_values[j] == None:
					continue
				scaled_val += neighbor_values[j].log_probability( r_hashes[j][k] )

			table.append( [ r_hashes[i][key] for i, key in enumerate( keys ) ] + [ cexp(scaled_val) ] )

		# Normalize the values
		total = sum( row[-1] for row in table )
		for i, row in enumerate( table ):
			table[i][-1] = row[-1] / total

		return JointProbabilityTable( table, parents, hashes )

	def marginal( self, neighbor_values=None ):
		"""
		Calculate the marginal of the CPT. This involves normalizing to turn it
		into a joint probability table, and then summing over the desired
		value. 
		"""

		# Convert from a dictionary to a list if necessary
		if isinstance( neighbor_values, dict ):
			neighbor_values = [ neighbor_values.get( d, None ) for d in self.parameters[1] ]

		# Get the index we're marginalizing over
		i = -1 if neighbor_values == None else neighbor_values.index( None )
		return self.joint( neighbor_values ).marginal( i )

cdef class JointProbabilityTable( MultivariateDistribution ):
	"""
	A joint probability table. The primary difference between this and the
	conditional table is that the final column sums to one here. The joint
	table can be thought of as the conditional probability table normalized
	by the marginals of each parent.
	"""

	def __init__( self, distribution, neighbors, hashes=None, n=None ):
		"""
		Take in the distribution represented as a list of lists, where each
		inner list represents a row.
		"""

		# If passed all the information, just store it, otherwise we need to
		# process it a little.
		if isinstance( distribution, numpy.ndarray ) and hashes and n:
			self.parameters = [ distribution, neighbors, hashes, n ]
		else:
			# Take the list of lists and invert it so that a numpy array represents
			# each column, since each column is a homogenous data type and we are
			# unsure if the distribution is over integers or strings.
			table = zip( *distribution )

			infer = len( neighbors ) == len( distribution[0] ) - 2
			if infer:
				d_keys = list( set( table[-2] ) )
				d_map = { key: i for i, key in enumerate( d_keys )}

			if hashes is None:
				hashes = [{ key: i for i, key in enumerate( neighbor.keys() ) } for neighbor in neighbors ]

				if infer:
					hashes += [ d_map ]

			if n is None and infer:
				n = map( len, neighbors + [d_keys] )
			elif n is None and not infer:
				n = map( len, neighbors )
				
			m = numpy.cumprod( [1] + n )

			values = numpy.zeros( len( distribution ) )

			# Go through each row and put it into the compressed array
			for row in distribution:
				i = sum( hashes[j][val]*m[j] for j, val in enumerate( row[:-1] ) )
				values[i] = _log( row[-1] )

			# Store all the information
			self.parameters = [ values, neighbors, hashes, n ]

	def log_probability( self, neighbor_values ):
		"""
		Return the log probability of a value given some number of parent
		values, which can be a subset of the full number of parents. If this
		is the case, then marginalize over unknown values.
		"""

		# Unpack the parameters
		values, neighbors, hashes, n = self.parameters
		m = numpy.cumprod( [1]+n )

		# Assume parent values are given in the same order that parents
		# were specified
		i = sum( hashes[j][val]*m[j] for j, val in enumerate( neighbor_values ) )

		# Return the array element with that identity
		return values[i]

	def marginal( self, wrt=-1, neighbor_values=None ):
		"""
		Determine the marginal of this table with respect to the index of one
		variable. The parents are index 0..n-1 for n parents, and the final
		variable is either the appropriate value or -1. For example:
		table = 
		A    B    C    p(C)
		... data ...
		table.marginal(0) gives the marginal wrt A
		table.marginal(1) gives the marginal wrt B
		table.marginal(2) gives the marginal wrt C
		table.marginal(-1) gives the marginal wrt C
		"""

		# Unpack the parameters
		values, neighbors, hashes, n = self.parameters

		# If given a dictionary, convert to a list
		if isinstance( neighbor_values, dict ):
			neighbor_values = [ neighbor_values.get( d, None ) for d in neighbors ]
		if isinstance( neighbor_values, list ):
			wrt = neighbor_values.index( None )

		r_hashes = [ d.keys() for d in hashes ]
		m = numpy.cumprod( [1]+n )

		# Determine the keys for the respective parent distribution
		d = { k: 0 for k in hashes[wrt].keys() }

		for ki, keys in enumerate( it.product( *[ xrange(i) for i in n ] ) ):
			i = sum( key*k for key, k in it.izip( keys, m ) )
			p = values[i]

			if neighbor_values is not None:
				for j, k in enumerate( keys ):
					if j == wrt:
						continue

					p += neighbor_values[j].log_probability( r_hashes[j][k] )

			d[ r_hashes[wrt][ keys[wrt] ] ] += cexp( p )

		total = sum( d.values() )
		for key, value in d.items():
			d[key] = value / total
		
		return DiscreteDistribution( d )
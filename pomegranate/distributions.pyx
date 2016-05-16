#!python
#cython: boundscheck=False
#cython: cdivision=True
# distributions.pyx
# Contact: Jacob Schreiber ( jmschreiber91@gmail.com )

from libc.stdlib cimport calloc
from libc.stdlib cimport free
from libc.string cimport memset
from libc.math cimport exp as cexp
from libc.math cimport fabs
from libc.math cimport sqrt as csqrt 

import itertools as it
import json
import numpy
import random
import scipy.special
import sys

from .utils cimport pair_lse
from .utils cimport _log
from .utils cimport lgamma

from collections import OrderedDict

if sys.version_info[0] > 2:
	# Set up for Python 3
	xrange = range
	izip = zip
else:
	izip = it.izip

# Define some useful constants
DEF NEGINF = float("-inf")
DEF INF = float("inf")
DEF SQRT_2_PI = 2.50662827463
DEF LOG_2_PI = 1.83787706641

def log(value):
	"""Return the natural log of the given value, or - nf if the value is 0."""

	if isinstance( value, numpy.ndarray ):
		to_return = numpy.zeros(( value.shape ))
		to_return[ value > 0 ] = numpy.log( value[ value > 0 ] )
		to_return[ value == 0 ] = NEGINF
		return to_return
	return _log( value )

def weight_set( items, weights ):
	"""Converts both items and weights to appropriate numpy arrays. 

	Convert the items into a numpy array with 64-bit floats, and the weight
	array to the same. If no weights are passed in, then return a numpy array
	with uniform weights.
	"""

	items = numpy.array(items, dtype=numpy.float64)
	if weights is None:
		# Weight everything 1 if no weights specified
		weights = numpy.ones(items.shape[0], dtype=numpy.float64)
	else:
		# Force whatever we have to be a Numpy array
		weights = numpy.array(weights, dtype=numpy.float64)
	
	return items, weights

cdef class Distribution:
	"""A probability distribution.

	Represents a probability distribution over the defined support. This is
	the base class which must be subclassed to specific probability
	distributions. All distributions have the below methods exposed.

	Parameters
	----------
	Varies on distribution.

	Attributes
	----------
	name : str
		The name of the type of distributioon.

	summaries : list
		Sufficient statistics to store the update.

	frozen : bool
		Whether or not the distribution will be updated during training.

	d : int
		The dimensionality of the data. Univariate distributions are all
		1, while multivariate distributions are > 1.
	"""

	def __cinit__( self ):
		self.name = "Distribution"
		self.frozen = False
		self.summaries = []
		self.d = 1

	def __str__( self ):
		"""Represent this distribution in JSON."""
		
		return self.to_json()

	def __repr__( self ):
		"""Represent this distribution in the same format as string."""

		return self.to_json()

	def marginal( self, *args, **kwargs ):
		"""Return the marginal of the distribution.

		Parameters
		----------
		*args : optional
			Arguments to pass in to specific distributions

		**kwargs : optional
			Keyword arguments to pass in to specific distributions

		Returns
		-------
		distribution : Distribution
			The marginal distribution. If this is a multivariate distribution
			then this method is filled in. Otherwise returns self. 
		"""

		return self

	def copy( self ):
		"""Return a deep copy of this distribution object.

		This object will not be tied to any other distribution or connected
		in any form.

		Paramters
		---------
		None

		Returns
		-------
		distribution : Distribution
			A copy of the distribution with the same parameters.
		"""

		return self.__class__( *self.parameters ) 

	def freeze( self ):
		"""Freeze the distribution, preventing updates from occuring.

		Parameters
		----------
		None

		Returns
		-------
		None
		"""

		self.frozen = True

	def thaw( self ):
		"""Thaw the distribution, re-allowing updates to occur.

		Parameters
		----------
		None

		Returns
		-------
		None
		"""

		self.frozen = False 

	def log_probability( self, double symbol ):
		"""Return the log probability of the given symbol under this distribution.

		Parameters
		----------
		symbol : double
			The symbol to calculate the log probability of (overriden for
			DiscreteDistributions)
		
		Returns
		-------
		logp : double
			The log probability of that point under the distribution.
		"""
		
		cdef double logp
		with nogil: 
			logp = self._log_probability( symbol )
		return logp

	cdef double _log_probability( self, double symbol ) nogil:
		"""Placeholder for the log probability calculation."""
		return NEGINF

	cdef double _mv_log_probability( self, double* symbol ) nogil:
		"""Placeholder for log probability calculation of a vector."""
		return NEGINF

	def sample( self ):
		"""Return a random item sampled from this distribution.

		Parameters
		----------
		None

		Returns
		-------
		sample : double or object
			Returns a sample from the distribution of a type in the support
			of the distribution.
		"""
		
		raise NotImplementedError
	
	def fit( self, items, weights=None, inertia=0.0 ):
		"""Fit the distribution to new data using MLE estimates.

		Parameters
		----------
		items : array-like, shape (n_samples, n_dimensions)
			This is the data to train on. Each row is a sample, and each column
			is a dimension to train on. For univariate distributions an array
			is used, while for multivariate distributions a 2d matrix is used.

		weights : array-like, shape (n_samples,), optional
			The initial weights of each sample in the matrix. If nothing is
			passed in then each sample is assumed to be the same weight.
			Default is None.

		inertia : double, optional
			The weight of the previous parameters of the model. The new
			parameters will roughly be old_param*inertia + new_param*(1-inertia), 
			so an inertia of 0 means ignore the old parameters, whereas an
			inertia of 1 means ignore the new parameters. Default is 0.0.

		Returns
		-------
		None
		"""

		if self.frozen == True:
			return
		raise NotImplementedError
		

	def train( self, items, weights=None, inertia=0.0 ):
		"""A wrapper for from_sample in order to homogenize calls more."""

		raise Warning("Deprecated. Use fit instead")
	

	def summarize( self, items, weights=None ):
		"""Summarize a batch of data into sufficient statistics for a later update.


		Parameters
		----------
		items : array-like, shape (n_samples, n_dimensions)
			This is the data to train on. Each row is a sample, and each column
			is a dimension to train on. For univariate distributions an array
			is used, while for multivariate distributions a 2d matrix is used.

		weights : array-like, shape (n_samples,), optional
			The initial weights of each sample in the matrix. If nothing is
			passed in then each sample is assumed to be the same weight.
			Default is None.

		Returns
		-------
		None
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

	cdef double _summarize( self, double* items, double* weights, SIZE_t n ) nogil:

		pass

	def from_summaries( self, inertia=0.0 ):
		"""Fit the distribution to the stored sufficient statistics.

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

		# If the distribution is frozen, don't bother with any calculation
		if self.frozen == True:
			return

		self.fit( *self.summaries, inertia=inertia )
		self.summaries = []

	def clear_summaries( self ):
		"""Clear the summary statistics stored in the object.

		Parameters 
		----------
		None

		Returns
		-------
		None
		"""

		self.summaries = []

	def plot( self, n=1000, **kwargs ):
		"""Plot the distribution by sampling from it.

		This function will plot a histogram of samples drawn from a distribution
		on the current open figure.

		Parameters
		----------
		n : int, optional
			The number of samples to draw from the distribution. Default is
			1000.

		**kwargs : arguments, optional
			Arguments to pass to matplotlib's histogram function.

		Returns
		-------
		None
		"""

		import matplotlib.pyplot as plt
		samples = [ self.sample() for i in xrange( n ) ]
		plt.hist( samples, **kwargs )

	def to_json( self, separators=(',', ' :'), indent=4 ):
		"""Serialize the distribution to a JSON.

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

		return json.dumps( {
								'class' : 'Distribution',
								'name'  : self.name,
								'parameters' : self.parameters,
								'frozen' : self.frozen
						   }, separators=separators, indent=indent )

	@classmethod
	def from_json( cls, s ):
		"""Read in a serialized distribution and return the appropriate object.
		
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

		if ' ' in d['class'] or 'Distribution' not in d['class']:
			raise SyntaxError( "Distribution object attempting to read invalid object." )

		if d['name'] == 'IndependentComponentsDistribution':
			d['parameters'][0] = [cls.from_json( json.dumps(dist) ) for dist in d['parameters'][0]]
			return IndependentComponentsDistribution( d['parameters'][0], d['parameters'][1], d['frozen'] )
		elif d['name'] == 'MixtureDistribution':
			d['parameters'][0] = [cls.from_json( json.dumps(dist) ) for dist in d['parameters'][0]]
			return MixtureDistribution( d['parameters'][0], d['parameters'][1], d['frozen'] )
		elif 'Table' in d['name']:
			parents = [ Distribution.from_json( json.dumps(j) ) for j in d['parents'] ]
			keys = [ (tuple(c), b) for c, b in d['keys'] ]
			if d['name'] == 'ConditionalProbabilityTable':
				model = ConditionalProbabilityTable( d['values'], parents, OrderedDict(keys) )
			elif d['name'] == 'JointProbabilityTable':
				model = JointProbabilityTable( d['values'], parents, OrderedDict(keys) )
			return model
		else:
			dist = eval( "{}( {}, frozen={} )".format( d['name'],
			                                    ','.join( map( str, d['parameters'] ) ),
			                                     d['frozen']) )
			return dist


cdef class UniformDistribution( Distribution ):
	"""A uniform distribution between two values."""

	property parameters:
		def __get__( self ):
			return [self.start, self.end]
		def __set__( self, parameters ):
			self.start, self.end = parameters

	def __cinit__( UniformDistribution self, double start, double end, bint frozen=False ):
		"""
		Make a new Uniform distribution over floats between start and end, 
		inclusive. Start and end must not be equal.
		"""
		
		# Store the parameters
		self.start = start
		self.end = end
		self.summaries = [INF, NEGINF]
		self.name = "UniformDistribution"
		self.frozen = frozen

	cdef double _log_probability( self, double symbol ) nogil:
		"""Cython optimized function for log probability calculation."""

		cdef double start = self.start
		cdef double end = self.end

		if symbol == start and symbol == end:
			return 0
		if symbol >= start and symbol <= end:
			return _log( 1.0 / ( end - start ) )
		return NEGINF
			
	def sample( self ):
		"""Sample from this uniform distribution and return the value sampled."""
		return random.uniform(self.start, self.end)
		
	def fit( self, items, weights=None, inertia=0.0 ):
		"""
		Set the parameters of this Distribution to maximize the likelihood of 
		the given sample. Items holds some sort of sequence. If weights is 
		specified, it holds a sequence of value to weight each item by.
		"""
	
		if self.frozen:
			return

		self.summarize( items, weights )
		self.from_summaries( inertia )

	cdef double _summarize( self, double* items, double* weights, SIZE_t n ) nogil:
		"""Cython optimized training."""

		cdef double minimum = INF, maximum = NEGINF 
		cdef int i

		for i in range(n):
			if items[i] < minimum:
				minimum = items[i]
			if items[i] > maximum:
				maximum = items[i]

		with gil:
			if maximum > self.summaries[1]:
				self.summaries[1] = maximum
			if minimum < self.summaries[0]:
				self.summaries[0] = minimum
	
	def summarize( self, items, weights=None ):
		"""
		Take in a series of items and their weights and reduce it down to a
		summary statistic to be used in training later.
		"""

		items, weights = weight_set(items, weights)
		if weights.sum() <= 0:
			return

		cdef double* items_p = <double*> (<numpy.ndarray> items).data
		cdef double* weights_p = NULL
		cdef SIZE_t n = items.shape[0]

		with nogil:
			self._summarize( items_p, weights_p, n )
		
	def from_summaries( self, inertia=0.0 ):
		"""
		Takes in a series of summaries, consisting of the minimum and maximum
		of a sample, and determine the global minimum and maximum.
		"""

		# If the distribution is frozen, don't bother with any calculation
		if self.frozen == True:
			return

		minimum, maximum = self.summaries
		self.start = minimum*(1-inertia) + self.start*inertia
		self.end = maximum*(1-inertia) + self.end*inertia

		self.summaries = [INF, NEGINF]

	def clear_summaries( self ):
		"""Clear the summary statistics stored in the object."""

		self.summaries = [INF, NEGINF]

	@classmethod
	def from_samples( cls, items, weights=None ):
		"""Fit a distribution to some data without pre-specifying it."""

		d = cls(0, 1)
		d.fit(items, weights)
		return d

cdef class NormalDistribution( Distribution ):
	"""
	A normal distribution based on a mean and standard deviation.
	"""

	property parameters:
		def __get__( self ):
			return [self.mu, self.sigma]
		def __set__( self, parameters ):
			self.mu, self.sigma = parameters

	def __cinit__( self, double mean, double std, bint frozen=False ):
		"""
		Make a new Normal distribution with the given mean mean and standard 
		deviation std.
		"""

		self.mu = mean
		self.sigma = std
		self.name = "NormalDistribution"
		self.frozen = frozen
		self.summaries = [0, 0, 0]
		self.log_sigma_sqrt_2_pi = -_log(std * SQRT_2_PI)
		self.two_sigma_squared = 2 * std ** 2

	cdef double _log_probability( self, double symbol ) nogil:
		"""Cython optimized function for log probability calculation."""

		return self.log_sigma_sqrt_2_pi - ((symbol - self.mu) ** 2) /\
			self.two_sigma_squared

	def sample( self ):
		"""Sample from this normal distribution and return the value sampled."""
		return random.normalvariate( self.mu, self.sigma )
		
	def fit( self, items, weights=None, inertia=0.0, min_std=0.01 ):
		"""
		Set the parameters of this Distribution to maximize the likelihood of 
		the given sample. Items holds some sort of sequence. If weights is 
		specified, it holds a sequence of value to weight each item by.
		"""
	
		if self.frozen:
			return

		self.summarize( items, weights )
		self.from_summaries( inertia, min_std )

	cdef double _summarize( self, double* items, double* weights, SIZE_t n ) nogil:
		"""Cython function to get the MLE estimate for a Gaussian."""

		cdef SIZE_t i
		cdef double x_sum = 0.0, x2_sum = 0.0, w_sum = 0.0

		# Calculate sufficient statistics for an update.
		for i in range(n):
			w_sum += weights[i]
			x_sum += weights[i] * items[i]
			x2_sum += weights[i] * items[i] * items[i]

		with gil:
			self.summaries[0] += w_sum
			self.summaries[1] += x_sum
			self.summaries[2] += x2_sum
		
	def summarize( self, items, weights=None ):
		"""
		Take in a series of items and their weights and reduce it down to a
		summary statistic to be used in training later.
		"""

		items, weights = weight_set(items, weights)
		if weights.sum() <= 0:
			return

		cdef double* items_p = <double*> (<numpy.ndarray> items).data
		cdef double* weights_p = <double*> (<numpy.ndarray> weights).data
		cdef SIZE_t n = items.shape[0]

		with nogil:
			self._summarize( items_p, weights_p, n )
		
	def from_summaries( self, inertia=0.0, min_std=0.01 ):
		"""
		Takes in a series of summaries, represented as a mean, a variance, and
		a weight, and updates the underlying distribution. Notes on how to do
		this for a Gaussian distribution were taken from here:
		http://math.stackexchange.com/questions/453113/how-to-merge-two-gaussians
		"""

		# If no summaries stored or the summary is frozen, don't do anything.
		if self.summaries[0] == 0 or self.frozen == True:
			return

		mu = self.summaries[1] / self.summaries[0]
		var = self.summaries[2] / self.summaries[0] - self.summaries[1] ** 2.0 / self.summaries[0] ** 2.0

		sigma = csqrt(var)
		if sigma < min_std:
			sigma = min_std

		self.mu = self.mu*inertia + mu*(1-inertia)
		self.sigma = self.sigma*inertia + sigma*(1-inertia)
		self.summaries = [0, 0, 0]
		self.log_sigma_sqrt_2_pi = -_log(sigma * SQRT_2_PI)
		self.two_sigma_squared = 2 * sigma ** 2

	def clear_summaries( self ):
		"""Clear the summary statistics stored in the object."""

		self.summaries = [0, 0, 0]

	@classmethod
	def from_samples( cls, items, weights=None ):
		"""Fit a distribution to some data without pre-specifying it."""

		d = cls(0, 1)
		d.fit(items, weights)
		return d

cdef class LogNormalDistribution( Distribution ):
	"""
	Represents a lognormal distribution over non-negative floats.
	"""

	property parameters:
		def __get__( self ):
			return [self.mu, self.sigma]
		def __set__( self, parameters ):
			self.mu, self.sigma = parameters

	def __cinit__( self, double mu, double sigma, frozen=False ):
		"""
		Make a new lognormal distribution. The parameters are the mu and sigma
		of the normal distribution, which is the the exponential of the log
		normal distribution.
		"""

		self.mu = mu
		self.sigma = sigma
		self.summaries = [0, 0, 0]
		self.name = "LogNormalDistribution"
		self.frozen = frozen

	cdef double _log_probability( self, double symbol ) nogil:
		"""Cython optimized function for log probability calculation."""

		cdef double mu = self.mu, sigma = self.sigma

		return -_log( symbol * sigma * SQRT_2_PI ) \
			- 0.5 * ( ( _log( symbol ) - mu ) / sigma ) ** 2

	def sample( self ):
		"""Return a sample from this distribution."""
		return numpy.random.lognormal( self.mu, self.sigma )

	def fit( self, items, weights=None, inertia=0.0, min_std=0.01 ):
		"""
		Set the parameters of this Distribution to maximize the likelihood of 
		the given sample. Items holds some sort of sequence. If weights is 
		specified, it holds a sequence of value to weight each item by.
		"""
	
		if self.frozen:
			return

		self.summarize( items, weights )
		self.from_summaries( inertia, min_std )

	cdef double _summarize( self, double* items, double* weights, SIZE_t n ) nogil:
		"""Cython function to get the MLE estimate for a Gaussian."""
		
		cdef SIZE_t i
		cdef double x_sum = 0.0, x2_sum = 0.0, w_sum = 0.0
		cdef double log_item

		# Calculate the average, which is the MLE mu estimate
		for i in range(n):
			log_item = _log(items[i])
			w_sum += weights[i]
			x_sum += weights[i] * log_item
			x2_sum += weights[i] * log_item * log_item

		with gil:
			self.summaries[0] += w_sum
			self.summaries[1] += x_sum
			self.summaries[2] += x2_sum
		
	def summarize( self, items, weights=None ):
		"""
		Take in a series of items and their weights and reduce it down to a
		summary statistic to be used in training later.
		"""

		items, weights = weight_set( items, weights )
		if weights.sum() <= 0:
			return

		cdef double* items_p = <double*> (<numpy.ndarray> items).data
		cdef double* weights_p = <double*> (<numpy.ndarray> weights).data
		cdef SIZE_t n = items.shape[0]

		with nogil:
			self._summarize( items_p, weights_p, n )
		
	def from_summaries( self, inertia=0.0, min_std=0.01 ):
		"""
		Takes in a series of summaries, represented as a mean, a variance, and
		a weight, and updates the underlying distribution. Notes on how to do
		this for a Gaussian distribution were taken from here:
		http://math.stackexchange.com/questions/453113/how-to-merge-two-gaussians
		"""

		# If no summaries stored or the summary is frozen, don't do anything.
		if len( self.summaries ) == 0 or self.frozen == True:
			return

		mu = self.summaries[1] / self.summaries[0]
		var = self.summaries[2] / self.summaries[0] - self.summaries[1] ** 2.0 / self.summaries[0] ** 2.0

		sigma = csqrt(var)
		if sigma < min_std:
			sigma = min_std

		self.mu = self.mu*inertia + mu*(1-inertia)
		self.sigma = self.sigma*inertia + sigma*(1-inertia)
		self.summaries = [0, 0, 0]

	def clear_summaries( self ):
		"""Clear the summary statistics stored in the object."""

		self.summaries = [0, 0, 0]

	@classmethod
	def from_samples( cls, items, weights=None ):
		"""Fit a distribution to some data without pre-specifying it."""

		d = cls(0, 1)
		d.fit(items, weights)
		return d

cdef class ExponentialDistribution( Distribution ):
	"""
	Represents an exponential distribution on non-negative floats.
	"""
	
	property parameters:
		def __get__( self ):
			return [ self.rate ]
		def __set__( self, parameters ):
			self.rate = parameters[0]

	def __cinit__( self, double rate, bint frozen=False ):
		"""
		Make a new inverse gamma distribution. The parameter is called "rate" 
		because lambda is taken.
		"""

		self.rate = rate
		self.summaries = [0, 0]
		self.name = "ExponentialDistribution"
		self.frozen = frozen

	cdef double _log_probability( self, double symbol ) nogil:
		"""Cython optimized function for log probability calculation."""
		return _log(self.rate) - self.rate * symbol
		
	def sample( self ):
		"""Sample from this exponential distribution."""
		return random.expovariate(*self.parameters)
		
	def fit( self, items, weights=None, inertia=0.0 ):
		"""
		Set the parameters of this Distribution to maximize the likelihood of 
		the given sample. Items holds some sort of sequence. If weights is 
		specified, it holds a sequence of value to weight each item by.
		"""
	
		if self.frozen:
			return

		self.summarize( items, weights )
		self.from_summaries( inertia )

	def summarize( self, items, weights=None ):
		"""
		Take in a series of items and their weights and reduce it down to a
		summary statistic to be used in training later.
		"""

		items, weights = weight_set( items, weights )

		cdef double* items_p = <double*> (<numpy.ndarray> items).data
		cdef double* weights_p = <double*> (<numpy.ndarray> weights).data
		cdef SIZE_t n = items.shape[0]

		with nogil:
			self._summarize( items_p, weights_p, n )

	cdef double _summarize( self, double* items, double* weights, SIZE_t n ) nogil:
		"""Cython function to get the MLE estimate for an exponential."""
		
		cdef double xw_sum = 0, w = 0
		cdef SIZE_t i

		# Calculate the average, which is the MLE mu estimate
		for i in range(n):
			xw_sum += items[i] * weights[i]
			w += weights[i]
		
		with gil:
			self.summaries[0] += w
			self.summaries[1] += xw_sum

	def from_summaries( self, inertia=0.0 ):
		"""
		Takes in a series of summaries, represented as a mean, a variance, and
		a weight, and updates the underlying distribution. Notes on how to do
		this for a Gaussian distribution were taken from here:
		http://math.stackexchange.com/questions/453113/how-to-merge-two-gaussians
		"""

		if self.frozen == True or self.summaries[0] == 0.0:
			return

		self.rate = self.summaries[0] / self.summaries[1]
		self.summaries = [0, 0]

	def clear_summaries( self ):
		"""Clear the summary statistics stored in the object."""

		self.summaries = [0, 0]

	@classmethod
	def from_samples( cls, items, weights=None ):
		"""Fit a distribution to some data without pre-specifying it."""

		d = cls(1)
		d.fit(items, weights)
		return d


cdef class BetaDistribution( Distribution ):
	"""
	This distribution represents a beta distribution, parameterized using
	alpha/beta, which are both shape parameters. ML estimation is done
	"""

	property parameters:
		def __get__( self ):
			return [ self.alpha, self.beta ]
		def __set__( self, parameters ):
			self.alpha, self.beta = parameters

	def __init__( self, alpha=1, beta=1, frozen=False ):
		"""
		Make a new beta distribution. Both alpha and beta are both shape
		parameters.
		"""

		self.alpha = alpha
		self.beta = beta
		self.summaries = []
		self.name = "BetaDistribution"
		self.frozen = frozen

	cdef double _log_probability( self, double symbol ) nogil:
		"""Cython optimized function for log probability calculation."""

		cdef double a = self.alpha, b = self.beta

		return ( _log(lgamma(a+b)) - _log(lgamma(a)) - 
			_log(lgamma(b)) + (a-1)*_log(symbol) +
			(b-1)*_log(1.-symbol) )

	def sample( self ):
		"""Return a random sample from the beta distribution."""
		return random.betavariate( self.alpha, self.beta )

	def fit( self, items, weights=None, inertia=0.0 ):
		"""
		Set the parameters of this Distribution to maximize the likelihood of 
		the given sample. Items holds some sort of sequence. If weights is 
		specified, it holds a sequence of value to weight each item by.
		"""
	
		if self.frozen:
			return

		self.summarize( items, weights )
		self.from_summaries( inertia )

	def summarize( self, items, weights=None ):
		"""
		Take in a series of items and their weights and reduce it down to a
		summary statistic to be used in training later.
		"""

		items, weights = weight_set( items, weights )

		cdef double* items_p = <double*> (<numpy.ndarray> items).data
		cdef double* weights_p = <double*> (<numpy.ndarray> weights).data
		cdef SIZE_t n = items.shape[0]

		with nogil:
			self._summarize( items_p, weights_p, n )

	cdef double _summarize( self, double* items, double* weights, SIZE_t n ) nogil:
		"""Cython optimized function for summarizing some data."""

		cdef double successes = 0, failures = 0
		cdef SIZE_t i

		for i in range(n):
			if items[i] == 1:
				successes += weights[i]
			else:
				failures += weights[i]

		with gil:
			self.summaries.append( (successes, failures) )

	def from_summaries( self, inertia=0.0 ):
		"""Use the summaries in order to update the distribution."""

		summaries = numpy.array( self.summaries )

		successes, failures = 0, 0
		for alpha, beta in self.summaries:
			successes += alpha
			failures += beta

		self.alpha = self.alpha*inertia + successes*(1-inertia)
		self.beta = self.beta*inertia + failures*(1-inertia)

		self.summaries = []

	def clear_summaries( self ):
		"""Clear the summary statistics stored in the object."""

		self.summaries = []

	@classmethod
	def from_samples( cls, items, weights=None ):
		"""Fit a distribution to some data without pre-specifying it."""

		d = cls(1, 1)
		d.fit(items, weights)
		return d


cdef class GammaDistribution( Distribution ):
	"""
	This distribution represents a gamma distribution, parameterized in the 
	alpha/beta (shape/rate) parameterization. ML estimation for a gamma 
	distribution, taking into account weights on the data, is nontrivial, and I 
	was unable to find a good theoretical source for how to do it, so I have 
	cobbled together a solution here from less-reputable sources.
	"""
	
	property parameters:
		def __get__( self ):
			return [ self.alpha, self.beta ]
		def __set__( self, parameters ):
			self.alpha, self.beta = parameters

	def __cinit__( self, double alpha, double beta, bint frozen=False ):
		"""
		Make a new gamma distribution. Alpha is the shape parameter and beta is 
		the rate parameter.
		"""

		self.alpha = alpha
		self.beta = beta
		self.summaries = []
		self.name = "GammaDistribution"
		self.frozen = frozen

	cdef double _log_probability( self, double symbol ) nogil:
		"""Cython optimized function for log probability calculation."""

		cdef double alpha = self.alpha, beta = self.beta
		
		return (_log(beta) * alpha - lgamma(alpha) + _log(symbol) 
			* (alpha - 1) - beta * symbol)
		
	def sample( self ):
		"""
		Sample from this gamma distribution and return the value sampled.
		"""
		
		# We have a handy sample from gamma function. Unfortunately, while we 
		# use the alpha, beta parameterization, and this function uses the 
		# alpha, beta parameterization, our alpha/beta are shape/rate, while its
		# alpha/beta are shape/scale. So we have to mess with the parameters.
		return random.gammavariate(self.parameters[0], 1.0 / self.parameters[1])
		
	def fit( self, items, weights=None, inertia=0.0, epsilon=1E-9, 
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
		self.alpha = prior_shape*inertia + shape*(1-inertia)
		self.beta = prior_rate*inertia + rate*(1-inertia)

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
		"""
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
		"""

		# If the distribution is frozen, don't bother with any calculation
		if len(self.summaries) == 0 or self.frozen == True:
			return

		# First, do Newton-Raphson for shape parameter.
		
		# Calculate the sufficient statistic s, which is the log of the average 
		# minus the average log. When computing the average log, we weight 
		# outside the log function. (In retrospect, this is actually pretty 
		# obvious.)
		summaries = numpy.array( self.summaries )

		statistic = _log( numpy.average( summaries[:,0], 
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
		self.alpha = prior_shape*inertia + shape*(1-inertia) 
		self.beta =	prior_rate*inertia + rate*(1-inertia)
		self.summaries = []

	def clear_summaries( self ):
		"""Clear the summary statistics stored in the object."""

		self.summaries = []	

	@classmethod
	def from_samples( cls, items, weights=None ):
		"""Fit a distribution to some data without pre-specifying it."""

		d = cls(1)
		d.fit(items, weights)
		return d


cdef class DiscreteDistribution( Distribution ):
	"""
	A discrete distribution, made up of characters and their probabilities,
	assuming that these probabilities will sum to 1.0. 
	"""

	property parameters:
		def __get__( self ):
			return [ self.dist ]
		def __set__( self, parameters ):
			self.dist = parameters[0]
			
	def __cinit__( self, dict characters, bint frozen=False ):
		"""
		Make a new discrete distribution with a dictionary of discrete
		characters and their probabilities, checking to see that these
		sum to 1.0. Each discrete character can be modelled as a
		Bernoulli distribution.
		"""
		
		self.name = "DiscreteDistribution"
		self.frozen = frozen

		self.dist = characters.copy()
		self.log_dist = { key: _log(value) for key, value in characters.items() }
		self.summaries =[ { key: 0 for key in characters.keys() }, 0 ]

		self.encoded_summary = 0
		self.encoded_keys = None
		self.encoded_counts = NULL
		self.encoded_log_probability = NULL

	def __dealloc__( self ):
		if self.encoded_keys is not None:
			free( self.encoded_counts )
			free( self.encoded_log_probability )

	def __len__( self ):
		"""Return the length of the underlying dictionary"""
		return len( self.dist )

	def __mul__( self, other ):
		"""Multiply this by another distribution sharing the same keys."""

		assert set( self.keys() ) == set( other.keys() )
		distribution, total = {}, 0.0

		for key in self.keys():
			distribution[key] = self.log_probability( key ) + other.log_probability( key )
			total += cexp( distribution[key] )

		for key in self.keys():
			distribution[key] = cexp( distribution[key] ) / total

		return DiscreteDistribution( distribution )

	def equals( self, other ):
		"""Return if the keys and values are equal"""

		if not isinstance( other, DiscreteDistribution ):
			return False

		if set( self.keys() ) != set( other.keys() ):
			return False

		for key in self.keys():
			self_prob = round( self.log_probability( key ), 12 )
			other_prob = round( other.log_probability( key ), 12 )
			if self_prob != other_prob:
				return False

		return True

	def clamp( self, key ):
		"""Return a distribution clamped to a particular value."""
		return DiscreteDistribution( { k : 0. if k != key else 1. for k in self.keys() } )

	def keys( self ):
		"""Return the keys of the underlying dictionary."""
		return self.dist.keys()

	def items( self ):
		"""Return items of the underlying dictionary."""
		return self.dist.items()

	def values( self ):
		"""Return values of the underlying dictionary."""
		return self.dist.values()

	def mle( self ):
		"""Return the maximally likely key."""

		max_key, max_value = None, 0
		for key, value in self.items():
			if value > max_value:
				max_key, max_value = key, value

		return max_key

	def encode( self, encoded_keys ):
		"""Encoding the distribution into integers."""

		if encoded_keys is None:
			return

		n = len(encoded_keys)
		self.encoded_keys = encoded_keys
		self.encoded_counts = <double*> calloc( n, sizeof(double) )
		self.encoded_log_probability = <double*> calloc( n, sizeof(double) )
		self.n = n

		for i in range(n):
			key = encoded_keys[i]
			self.encoded_counts[i] = 0
			self.encoded_log_probability[i] = self.log_dist.get( key, NEGINF )

	def log_probability( self, symbol ):
		"""Return the log prob of the symbol under this distribution."""

		return self.__log_probability( symbol )

	cdef double __log_probability( self, symbol ):
		"""Cython optimized lookup."""

		return self.log_dist.get( symbol, NEGINF )

	cdef public double _log_probability( self, double symbol ) nogil:
		"""Cython optimized lookup."""

		if symbol < 0 or symbol > self.n:
			return NEGINF
		return self.encoded_log_probability[<SIZE_t> symbol]

	def sample( self ):
		"""Sample randomly from the discrete distribution."""
		
		rand = random.random()
		for key, value in self.items():
			if value >= rand:
				return key
			rand -= value
	
	def fit( self, items, weights=None, inertia=0.0 ):
		"""
		Set the parameters of this Distribution to maximize the likelihood of 
		the given sample. Items holds some sort of sequence. If weights is 
		specified, it holds a sequence of value to weight each item by.
		"""
	
		if self.frozen:
			return

		self.summarize( items, weights )
		self.from_summaries( inertia )

	def summarize( self, items, weights=None ):
		"""Reduce a set of obervations to sufficient statistics."""

		if weights is None:
			weights = numpy.ones(len(items))
		else:
			weights = numpy.asarray(weights)

		self.summaries[1] += weights.sum()
		characters = self.summaries[0]
		for i in xrange( len(items) ):
			characters[items[i]] += weights[i]

	cdef double _summarize( self, double* items, double* weights, SIZE_t n ) nogil:
		"""Cython version of summarize."""

		cdef int i
		self.encoded_summary = 1

		encoded_counts = <double*> calloc( self.n, sizeof(double) )
		memset( encoded_counts, 0, self.n*sizeof(double) )

		for i in range(n):
			encoded_counts[<SIZE_t> items[i]] += weights[i]

		with gil:
			for i in range(self.n):
				self.encoded_counts[i] += encoded_counts[i]
				self.summaries[1] += encoded_counts[i]

		free( encoded_counts )

	def from_summaries( self, inertia=0.0 ):
		"""Use the summaries in order to update the distribution."""

		if self.summaries[1] == 0:
			return

		if self.encoded_summary == 0:
			_sum = sum( self.summaries[0].values() )
			characters = {}
			for key, value in self.summaries[0].items():
				self.dist[key] = self.dist[key]*inertia + (1-inertia)*value / _sum
				self.log_dist[key] = _log( self.dist[key] )

			self.encode( self.encoded_keys )
		else:
			n = len(self.encoded_keys)
			for i in range(n):
				key = self.encoded_keys[i]
				self.dist[key] = (self.dist[key]*inertia + 
					(1-inertia)*self.encoded_counts[i] / self.summaries[1])
				self.log_dist[key] = _log( self.dist[key] )
				self.encoded_counts[i] = 0

			self.encode( self.encoded_keys )

		self.summaries = [{ key: 0 for key in self.keys() }, 0]

	def clear_summaries( self ):
		"""Clear the summary statistics stored in the object."""

		self.summaries = [{ key: 0 for key in self.keys() }, 0]
		if self.encoded_summary == 1:
			for i in range(len(self.encoded_keys)):
				self.encoded_counts[i] = 0

	@classmethod
	def from_samples( cls, items, weights=None ):
		"""Fit a distribution to some data without pre-specifying it."""

		if weights is None:
			weights = numpy.ones( len(items) )

		characters = {}
		for character, weight in it.izip(items, weights):
			if character in characters:
				characters[character] += weight
			else:
				characters[character] = weight

		d = cls(characters)
		return d


cdef class PoissonDistribution(Distribution):
	"""
	A discrete probability distribution which expresses the probability of a 
	number of events occuring in a fixed time window. It assumes these events
	occur with at a known rate, and independently of each other.
	"""

	property parameters:
		def __get__( self ):
			return [self.l]
		def __set__( self, parameters ):
			self.l = parameters[0]

	def __cinit__(self, l, frozen=False):
		self.l = l
		self.logl = _log(l)
		self.name = "PoissonDistribution"
		self.summaries = [0, 0]
		self.frozen = frozen

	cdef double _log_probability( self, double symbol ) nogil:
		"""Cython optimized function for log probability calculation."""

		cdef double factorial = 1.0
		cdef int i

		if symbol < 0:
			return NEGINF

		elif symbol > 1:
			for i in range(2, <int>symbol+1):
				factorial *= i

		return symbol * self.logl - self.l - _log(factorial)

	def sample( self ):
		"""Sample from the poisson distribution."""

		return numpy.random.poisson( self.l )

	def fit( self, items, weights=None, inertia=0.0 ):
		"""
		Update the parameters of this distribution to maximize the likelihood
		of the current samples. If weights are passed in, perform weighted
		MLE, otherwise unweighted.
		"""

		if self.frozen:
			return

		self.summarize( items, weights )
		self.from_summaries( inertia )

	def summarize( self, items, weights=None ):
		"""
		Take in a series of items and their weights and reduce it down to a
		summary statistic to be used in training later.
		"""

		items, weights = weight_set(items, weights)
		if weights.sum() <= 0:
			return

		cdef double* items_p = <double*> (<numpy.ndarray> items).data
		cdef double* weights_p = <double*> (<numpy.ndarray> weights).data
		cdef SIZE_t n = items.shape[0]

		with nogil:
			self._summarize( items_p, weights_p, n )

	cdef double _summarize( self, double* items, double* weights, SIZE_t n ) nogil:
		"""Cython optimized function to calculate the summary statistics."""

		cdef double x_sum = 0.0, w_sum = 0.0
		cdef int i

		for i in range(n):
			x_sum += items[i] * weights[i]
			w_sum += weights[i]

		with gil:
			self.summaries[0] += x_sum
			self.summaries[1] += w_sum

	def from_summaries( self, inertia=0.0 ):
		"""
		Takes in a series of summaries, consisting of the minimum and maximum
		of a sample, and determine the global minimum and maximum.
		"""

		# If the distribution is frozen, don't bother with any calculation
		if self.frozen == True:
			return

		x_sum, w_sum = self.summaries
		mu = x_sum / w_sum 

		self.l = mu*(1-inertia) + self.l*inertia
		self.logl = _log(self.l)
		self.summaries = [0, 0]

	def clear_summaries( self ):
		"""Clear the summary statistics stored in the object."""

		self.summaries = [0, 0]

	@classmethod
	def from_samples( cls, items, weights=None ):
		"""Fit a distribution to some data without pre-specifying it."""

		d = cls(0)
		d.fit(items, weights)
		return d


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
	
	def __init__(self, lambda_funct=None, frozen=True ):
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

cdef class KernelDensity( Distribution ):
	"""An abstract kernel density, with shared properties and methods."""

	property parameters:
		def __get__( self ):
			return [ self.points_ndarray.tolist(), self.bandwidth, self.weights_ndarray.tolist() ]
		def __set__( self, parameters ):
			self.points_ndarray = numpy.array( parameters[0] )
			self.points = <double*> self.points_ndarray.data

			self.bandwidth = parameters[1]

			self.weights_ndarray = numpy.array( parameters[2] )
			self.weights = <double*> self.weights_ndarray.data

	def __cinit__( self, points=[], bandwidth=1, weights=None, frozen=False ):
		"""
		Take in points, bandwidth, and appropriate weights. If no weights
		are provided, a uniform weight of 1/n is provided to each point.
		Weights are scaled so that they sum to 1. 
		"""

		points = numpy.asarray( points, dtype=numpy.float64 )
		n = points.shape[0]
		
		if weights is not None:
			weights = numpy.array(weights, dtype=numpy.float64) / numpy.sum(weights)
		else:
			weights = numpy.ones( n, dtype=numpy.float64 ) / n 

		self.n = n
		self.points_ndarray = points
		self.points = <double*> self.points_ndarray.data
		
		self.weights_ndarray = weights
		self.weights = <double*> self.weights_ndarray.data

		self.bandwidth = bandwidth
		self.summaries = []
		self.name = "KernelDensity"
		self.frozen = frozen

	def fit( self, points, weights=None, inertia=0.0 ):
		"""Replace the points, allowing for inertia if specified."""

		# If the distribution is frozen, don't bother with any calculation
		if self.frozen == True:
			return

		points = numpy.asarray( points, dtype=numpy.float64 )
		n = points.shape[0]

		# Get the weights, or assign uniform weights
		if weights is not None:
			weights = numpy.array( weights, dtype=numpy.float64 ) / numpy.sum(weights)
		else:
			weights = numpy.ones( n, dtype=numpy.float64 ) / n 

		# If no inertia, get rid of the previous points
		if inertia == 0.0:
			self.points_ndarray = points
			self.weights_ndarray = weights
			self.n = points.shape[0]

		# Otherwise adjust weights appropriately
		else: 
			self.points_ndarray = numpy.concatenate( ( self.points_ndarray, points ) )
			self.weights_ndarray = numpy.concatenate( ( self.weights_ndarray*inertia, weights*(1-inertia) ) )
			self.n = points.shape[0]

		self.points = <double*> self.points_ndarray.data
		self.weights = <double*> self.weights_ndarray.data

	@classmethod
	def from_samples( cls, items, weights=None ):
		"""Fit a distribution to some data without pre-specifying it."""

		d = cls([])
		d.fit(items, weights)
		return d


cdef class GaussianKernelDensity( KernelDensity ):
	"""
	A quick way of storing points to represent a Gaussian kernel density in one
	dimension. Takes in the points at initialization, and calculates the log of
	the sum of the Gaussian distance of the new point from every other point.
	"""

	def __cinit__( self, points=[], bandwidth=1, weights=None, frozen=False ):
		"""
		Take in points, bandwidth, and appropriate weights. If no weights
		are provided, a uniform weight of 1/n is provided to each point.
		Weights are scaled so that they sum to 1. 
		"""

		self.name = "GaussianKernelDensity"

	cdef double _log_probability( self, double symbol ) nogil:
		"""Cython optimized function for log probability calculation."""

		cdef double mu, w
		cdef double scalar = 1.0 / SQRT_2_PI
		cdef int i, n = self.n
		cdef double prob = 0.0

		for i in range( n ):
			mu = self.points[i]
			w = self.weights[i]

			prob += w * scalar * cexp( -0.5 * (( mu-symbol ) / self.bandwidth) ** 2 )

		return _log( prob )

	def sample( self ):
		"""
		Generate a random sample from this distribution. This is done by first
		selecting a random point, weighted by weights if the points are weighted
		or uniformly if not, and then randomly sampling from that point's PDF.
		"""

		mu = numpy.random.choice( self.parameters[0], p=self.parameters[2] )
		return random.gauss( mu, self.parameters[1] )

cdef class UniformKernelDensity( KernelDensity ):
	"""
	A quick way of storing points to represent an Exponential kernel density in
	one dimension. Takes in points at initialization, and calculates the log of
	the sum of the Gaussian distances of the new point from every other point.
	"""

	def __cinit__( self, points=[], bandwidth=1, weights=None, frozen=False ):
		"""
		Take in points, bandwidth, and appropriate weights. If no weights
		are provided, a uniform weight of 1/n is provided to each point.
		Weights are scaled so that they sum to 1. 
		"""

		self.name = "UniformKernelDensity"

	cdef double _log_probability( self, double symbol ) nogil:
		"""Cython optimized function for log probability calculation."""

		cdef double bandwidth = self.bandwidth
		cdef double mu, w

		cdef int i, n = self.n
		cdef double prob = 0.0 

		for i in range( n ):
			# Go through each point sequentially
			mu = self.points[i]
			w = self.weights[i]

			# The good thing about uniform distributions if that
			# you just need to check to make sure the point is within
			# a bandwidth.
			if fabs( mu - symbol ) <= bandwidth:
				prob += w

		# Return the log of the sum of probabilities
		return _log( prob )
	
	def sample( self ):
		"""
		Generate a random sample from this distribution. This is done by first
		selecting a random point, weighted by weights if the points are weighted
		or uniformly if not, and then randomly sampling from that point's PDF.
		"""

		mu = numpy.random.choice( self.parameters[0], p=self.parameters[2] )
		bandwidth = self.parameters[1]
		return random.uniform( mu-bandwidth, mu+bandwidth )

	@classmethod
	def from_samples( cls, items, weights=None ):
		"""Fit a distribution to some data without pre-specifying it."""

		d = cls([])
		d.fit(items, weights)
		return d


cdef class TriangleKernelDensity( KernelDensity ):
	"""
	A quick way of storing points to represent an Exponential kernel density in
	one dimension. Takes in points at initialization, and calculates the log of
	the sum of the Gaussian distances of the new point from every other point.
	"""

	def __cinit__( self, points=[], bandwidth=1, weights=None, frozen=False ):
		"""
		Take in points, bandwidth, and appropriate weights. If no weights
		are provided, a uniform weight of 1/n is provided to each point.
		Weights are scaled so that they sum to 1. 
		"""

		self.name = "TriangleKernelDensity"

	cdef double _log_probability( self, double symbol ) nogil:
		"""Cython optimized function for log probability calculation."""

		cdef double bandwidth = self.bandwidth
		cdef double mu, w

		cdef int i, n = self.n
		cdef double prob = 0.0, hinge

		for i in range( n ):
			# Go through each point sequentially
			mu = self.points[i]
			w = self.weights[i]

			# Calculate the probability for each point
			hinge = bandwidth - fabs( mu - symbol ) 
			if hinge > 0:
				prob += hinge * w 

		# Return the log of the sum of probabilities
		return _log( prob )

	def sample( self ):
		"""
		Generate a random sample from this distribution. This is done by first
		selecting a random point, weighted by weights if the points are weighted
		or uniformly if not, and then randomly sampling from that point's PDF.
		"""

		mu = numpy.random.choice( self.parameters[0], p=self.parameters[2] )
		bandwidth = self.parameters[1]
		return random.triangular( mu-bandwidth, mu+bandwidth, mu )

	@classmethod
	def from_samples( cls, items, weights=None ):
		"""Fit a distribution to some data without pre-specifying it."""

		d = cls([])
		d.fit(items, weights)
		return d

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

	s1 = IndependentComponentsDistribution([ ExponentialDistribution( 0.1 ), 
									NormalDistribution( 5, 2 ) ])
	s1.log_probability( (5, 2 ) )
	"""

	property parameters:
		def __get__( self ):
			return [ self.distributions.tolist(), numpy.exp(self.weights).tolist() ]
		def __set__( self, parameters ):
			self.distributions = numpy.asarray( parameters[0], dtype=numpy.object_ )
			self.weights = numpy.log( parameters[1] )

	def __cinit__( self, distributions=[], weights=None, frozen=False ):
		"""
		Take in the distributions and appropriate weights. If no weights
		are provided, a uniform weight of 1/n is provided to each point.
		Weights are scaled so that they sum to 1. 
		"""

		self.distributions = numpy.array( distributions )
		self.distributions_ptr = <void**> self.distributions.data

		self.d = len(distributions)

		if weights:
			weights = numpy.array( weights, dtype=numpy.float64 )
		else:
			weights = numpy.ones( self.d, dtype=numpy.float64 )

		self.weights = numpy.log( weights )
		self.weights_ptr = <double*> self.weights.data
		self.name = "IndependentComponentsDistribution"
		self.frozen = frozen

	def log_probability( self, symbol ):
		"""
		What's the probability of a given tuple under this mixture? It's the
		product of the probabilities of each symbol in the tuple under their
		respective distribution, which is the sum of the log probabilities.
		"""

		cdef numpy.ndarray symbol_ndarray = numpy.array(symbol).astype('float64')
		cdef double* symbol_ptr = <double*> symbol_ndarray.data
		cdef double logp

		with nogil:
			logp = self._mv_log_probability( symbol_ptr )

		return logp

	cdef double _mv_log_probability( self, double* symbol ) nogil:
		"""Cython optimized function for log probability calculation."""

		cdef int i, d = self.d
		cdef double w, logp = 0.0

		for i in range(d):
			logp += ( <Distribution> self.distributions_ptr[i] )._log_probability( symbol[i] )
			logp += self.weights_ptr[i]

		return logp

	def sample( self ):
		"""
		Sample from the mixture. First, choose a distribution to sample from
		according to the weights, then sample from that distribution. 
		"""

		return [ d.sample() for d in self.parameters[0] ]

	def fit( self, items, weights=None, inertia=0 ):
		"""
		Set the parameters of this Distribution to maximize the likelihood of 
		the given sample. Items holds some sort of sequence. If weights is 
		specified, it holds a sequence of value to weight each item by.
		"""
	
		if self.frozen:
			return

		self.summarize( items, weights )
		self.from_summaries( inertia )

	def summarize( self, items, weights=None ):
		"""
		Take in an array of items and reduce it down to summary statistics. For
		a multivariate distribution, this involves just passing the appropriate
		data down to the appropriate distributions.
		"""

		items, weights = weight_set( items, weights )
		cdef double* items_ptr = <double*> (<numpy.ndarray> items).data
		cdef double* weights_ptr = <double*> (<numpy.ndarray> weights).data

		self._summarize( items_ptr, weights_ptr, items.shape[0] )

	cdef double _summarize( self, double* items, double* weights, SIZE_t n ) nogil:
		cdef SIZE_t i, j, d = self.d

		for i in range(n):
			for j in range(d):
				( <Distribution> self.distributions_ptr[j] )._summarize( items+i*d+j, weights+i, 1 )

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

	def clear_summaries( self ):
		"""Clear the summary statistics stored in the object."""

		for d in self.parameters[0]:
			d.clear_summaries()

	def to_json( self, separators=(',', ' : '), indent=4 ):
		"""Convert the distribution to JSON format."""

		return json.dumps( {
								'class' : 'Distribution',
								'name'  : self.name,
								'parameters' : [[ json.loads( dist.to_json() ) for dist in self.parameters[0] ],
								                 self.parameters[1] ],
								'frozen' : self.frozen
						   }, separators=separators, indent=indent )


cdef class MultivariateGaussianDistribution( MultivariateDistribution ):
	property parameters:
		def __get__( self ):
			return [ self.mu.tolist(), self.cov.tolist() ]
		def __set__( self, parameters ):
			self.mu = numpy.array( parameters[0] )
			self.cov = numpy.array( parameters[1] )

	def __cinit__( self, means=[], covariance=[], frozen=False ):
		"""
		Take in the mean vector and the covariance matrix. 
		"""

		self.name = "MultivariateGaussianDistribution"
		self.frozen = frozen
		self.mu = numpy.array(means, dtype=numpy.float64)
		self.cov = numpy.array(covariance, dtype=numpy.float64)

		det = numpy.linalg.det(covariance)

		if self.mu.shape[0] != self.cov.shape[0]:
			raise ValueError("mu shape is {} while covariance shape is {}".format( self.mu.shape[0], self.cov.shape[0] ))
		if self.cov.shape[0] != self.cov.shape[1]:
			raise ValueError("covariance is not a square matrix, dimensions are ({}, {})".format( self.cov.shape[0], self.cov.shape[1] ) )
		if det == 0:
			raise ValueError("covariance matrix is not invertible.")
		
		d = self.mu.shape[0]
		self.d = d

		self.inv_cov_ndarray = numpy.linalg.inv(covariance).astype('float64')
		self.inv_cov = <double*> (<numpy.ndarray> self.inv_cov_ndarray).data
		self._mu = <double*> (<numpy.ndarray> self.mu).data
		self._cov = <double*> (<numpy.ndarray> self.cov).data
		self._log_det = _log(det)

		self.w_sum = 0.0
		self.column_sum = <double*> calloc( d, sizeof(double) )
		self.pair_sum = <double*> calloc( d*d, sizeof(double) )
		memset( self.column_sum, 0, d*sizeof(double) )
		memset( self.pair_sum, 0, d*d*sizeof(double) )

		self._mu_new = <double*> calloc( d, sizeof(double) )
		self._cov_new = <double*> calloc( d*d, sizeof(double) )

	def __dealloc__(self):
		free(self._mu_new)
		free(self._cov_new)
		free(self.column_sum)
		free(self.pair_sum)

	def log_probability( self, symbol ):
		"""
		What's the probability of a given tuple under this mixture? It's the
		product of the probabilities of each symbol in the tuple under their
		respective distribution, which is the sum of the log probabilities.
		"""

		cdef numpy.ndarray symbol_ndarray = numpy.array(symbol).astype(numpy.float64)
		cdef double* symbol_ptr = <double*> symbol_ndarray.data
		cdef double logp

		with nogil:
			logp = self._mv_log_probability(symbol_ptr)
		return logp

	cdef double _mv_log_probability( self, double* symbol ) nogil:
		"""Cython optimized function for log probability calculation."""

		cdef SIZE_t i, j, d = self.d
		cdef double log_det = self._log_det
		cdef double logp = 0.0

		for i in range(d):
			for j in range(d):
				logp += (symbol[i] - self._mu[i]) * (symbol[j] - self._mu[j]) * self.inv_cov[i + j*d]

		return -0.5 * (d * LOG_2_PI + log_det + logp)

	def sample( self ):
		"""
		Sample from the mixture. First, choose a distribution to sample from
		according to the weights, then sample from that distribution. 
		"""

		return numpy.random.multivariate_normal( self.parameters[0], 
			self.parameters[1] )

	def fit( self, items, weights=None, inertia=0 ):
		"""
		Set the parameters of this Distribution to maximize the likelihood of 
		the given sample. Items holds some sort of sequence. If weights is 
		specified, it holds a sequence of value to weight each item by.
		"""
	
		if self.frozen:
			return

		self.summarize( items, weights )
		self.from_summaries( inertia )

	def summarize( self, items, weights=None ):
		"""
		Take in a series of items and their weights and reduce it down to a
		summary statistic to be used in training later.
		"""

		items, weights = weight_set( items, weights )

		cdef double* items_p = <double*> (<numpy.ndarray> items).data
		cdef double* weights_p = <double*> (<numpy.ndarray> weights).data
		
		cdef SIZE_t n = items.shape[0]
		d = items.shape[1]

		if self.d != d:
			self.d = d

			free(self.column_sum)
			self.column_sum = <double*> calloc( d, sizeof(double) )
			memset( self.column_sum, 0, d*sizeof(double) )

			free(self.pair_sum)
			self.pair_sum = <double*> calloc( d*d, sizeof(double) )
			memset( self.pair_sum, 0, d*d*sizeof(double) )

			self._mu_new = <double*> calloc( d, sizeof(double) )
			self._cov_new = <double*> calloc( d*d, sizeof(double) )

		with nogil:
			self._summarize( items_p, weights_p, n )

	cdef double _summarize( self, double* items, double* weights, SIZE_t n ) nogil:
		"""Calculate sufficient statistics for a minibatch.

		The sufficient statistics for a multivariate gaussian update is the sum of
		each column, and the sum of the outer products of the vectors.
		"""

		cdef SIZE_t i, j, k, d = self.d
		cdef double w_sum = 0.0
		cdef double* column_sum = <double*> calloc(d, sizeof(double))
		cdef double* pair_sum = <double*> calloc(d*d, sizeof(double))
		memset( column_sum, 0, d*sizeof(double) )
		memset( pair_sum, 0, d*d*sizeof(double) )

		for i in range(n):
			w_sum += weights[i]

			for j in range(d):
				column_sum[j] += items[i*d + j] * weights[i]

				for k in range(d):
					pair_sum[j*d + k] += (weights[i] * items[i*d + j] * 
						items[i*d + k])

		with gil:
			self.w_sum += w_sum

			for j in range(d):
				self.column_sum[j] += column_sum[j]

				for k in range(d):
					self.pair_sum[j*d + k] += pair_sum[j*d + k]

		free(column_sum)
		free(pair_sum)

	def from_summaries( self, inertia=0.0 ):
		"""
		Set the parameters of this Distribution to maximize the likelihood of 
		the given sample. Items holds some sort of sequence. If weights is 
		specified, it holds a sequence of value to weight each item by.
		"""

		# If no summaries stored or the summary is frozen, don't do anything.
		if self.frozen == True or self.w_sum < 1e-7:
			return

		cdef SIZE_t d = self.d, i, j, k
		cdef double* column_sum = self.column_sum
		cdef double* pair_sum = self.pair_sum
		cdef double* u = self._mu_new

		if self.cov.shape[0] != self.d:
			self.cov = numpy.zeros((self.d, self.d))
			self._cov = <double*> (<numpy.ndarray> self.cov).data

			self.mu = numpy.zeros(self.d)
			self._mu = <double*> (<numpy.ndarray> self.mu).data

		for i in range(d):
			u[i] = self.column_sum[i] / self.w_sum
			self._mu[i] = self._mu[i] * inertia + u[i] * (1-inertia)

		for j in range(d):
			for k in range(d):
				self._cov_new[j*d + k] = (pair_sum[j*d + k] - column_sum[j]*u[k] 
					- column_sum[k]*u[j] + self.w_sum*u[j]*u[k]) / self.w_sum
				self._cov[j*d + k] = self._cov[j*d + k] * inertia + self._cov_new[j*d +k] * (1-inertia)

		memset( column_sum, 0, d*sizeof(double) )
		memset( pair_sum, 0, d*d*sizeof(double) )
		self.w_sum = 0.0

		self.inv_cov_ndarray = numpy.linalg.inv(self.cov).astype('float64')
		self.inv_cov = <double*> (<numpy.ndarray> self.inv_cov_ndarray).data
		_, self._log_det = numpy.linalg.slogdet(self.cov)

	def clear_summaries( self ):
		"""Clear the summary statistics stored in the object."""

		memset( self.column_sum, 0, self.d*sizeof(double) )
		memset( self.pair_sum, 0, self.d*self.d*sizeof(double) )
		self.w_sum = 0.0

	@classmethod
	def from_samples( cls, items, weights=None ):
		"""Fit a distribution to some data without pre-specifying it."""

		n = len(items[0])
		d = cls( numpy.ones(n), numpy.eye(n) )
		d.fit(items, weights)
		return d


cdef class ConditionalProbabilityTable( MultivariateDistribution ):
	"""
	A conditional probability table, which is dependent on values from at
	least one previous distribution but up to as many as you want to
	encode for.
	"""

	def __init__( self, table=None, parents=None, keys=None, frozen=False ):
		"""
		Take in the distribution represented as a list of lists, where each
		inner list represents a row.
		"""

		self.name = "ConditionalProbabilityTable"
		self.frozen = False

		if keys:
			self.parameters = [ table, parents, keys ]
		else:
			keys = []
			values = numpy.zeros( len(table) )

			for i, row in enumerate( table ):
				keys.append( ( tuple(row[:-1]), i ) )
				values[i] = _log( row[-1] )

			self.key_dict = dict(keys)
			keys = OrderedDict( keys )
			self.parameters = [ values, parents, keys ]

		self.summaries = [{}, {}]

	def __str__( self ):
		"""
		Regenerate the table.
		"""

		values, parents, keys = self.parameters
		return "\n".join( 
					"\t".join( map( str, key + (cexp( values[idx] ),) ) )
							for key, idx in keys.items() ) 

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

		return tuple(set(row[-1] for row in self.parameters[2].keys()))

	def sample( self, parent_values={} ):
		"""Return a random sample from the conditional probability table."""

		values, parents, keys = self.parameters
		keys = keys.keys()

		for parent in parents:
			if parent not in parent_values:
				parent_values[parent] = parent.sample()

		n = len(keys)
		idxs = []
		values_ = []

		for i in range(n):
			for j, parent in enumerate( parents ):
				if parent_values[parent] != keys[i][j]:
					break
			else:
				idxs.append(i)
				values_.append(cexp(values[i]))

		values_ = numpy.cumsum(values_)
		a = numpy.random.uniform(0, 1)
		for i in range(len(values_)):
			if values_[i] > a:
				return keys[idxs[i]][-1] 

	def log_probability( self, symbol ):
		"""
		Return the log probability of a value, which is a tuple in proper
		ordering, like the training data.
		"""

		# Unpack the parameters
		values, _, keys = self.parameters

		# Return the array element with that identity
		return values[ keys[symbol] ]
		
	def joint( self, neighbor_values=None ):
		"""
		This will turn a conditional probability table into a joint
		probability table. If the data is already a joint, it will likely
		mess up the data. It does so by scaling the parameters the probabilities
		by the parent distributions.
		"""

		# Unpack the parameters
		values, parents, hashes = self.parameters

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
		for key, idx in hashes.items():

			# Scale the probability by the weights on the marginals
			scaled_val = values[idx]
			for j, k in enumerate( key ):
				if neighbor_values[j] is not None:
					scaled_val += neighbor_values[j].log_probability( k )

			table.append( key + (cexp(scaled_val),) )

		# Normalize the values
		total = sum( row[-1] for row in table )
		table = [ row[:-1] + (row[-1] / total,) for row in table ]

		return JointProbabilityTable( table, parents )

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

	def summarize(self, items, weights=None):
		"""Summarize the data into sufficient statistics to store."""

		if len(items) == 0 or self.frozen == True:
			return

		if weights is None:
			weights = numpy.ones( len(items), dtype='float64' )
		elif numpy.sum( weights ) == 0:
			return
		else:
			weights = numpy.asarray(weights, dtype='float64' )

		self._table_summarize(items, weights)

	cdef void _table_summarize(self, items, double [:] weights):
		cdef int i, n = len(items)
		cdef tuple item

		for i in range(n):
			item = tuple(items[i])
			self.summaries[0][item] = self.summaries[0].get(item, 0) + weights[i]
			self.summaries[1][item[:-1]] = self.summaries[1].get(item[:-1], 0) + weights[i]

	def from_summaries( self, inertia=0.0, pseudocount=0.0 ):
		"""Update the parameters of the distribution using sufficient statistics."""

		values = numpy.zeros_like(self.parameters[0])
		keys = self.key_dict

		for key in self.parameters[2].keys():
			count = self.summaries[0].get( key, 0.0 )
			marginal_count = self.summaries[1].get( key[:-1], 0.0 )

			probability = count / marginal_count if marginal_count > 0 else 1. / len(self)
			values[keys[key]] = probability

		self.parameters[0] = numpy.log(numpy.exp(self.parameters[0])*inertia + values*(1-inertia) + pseudocount)
		self.summaries = [{}, {}]

	def clear_summaries( self ):
		"""Clear the summary statistics stored in the object."""

		self.summaries = [{}, {}]

	def fit( self, items, weights=None, inertia=0.0, pseudocount=0.0 ):
		"""Update the parameters of the table based on the data."""

		self.summarize( items, weights )
		self.from_summaries( inertia )

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
					'class' : 'Distribution',
		            'name' : 'ConditionalProbabilityTable',
		            'values' : self.parameters[0].tolist(),
		            'parents' : [ json.loads( dist.to_json() ) for dist in self.parameters[1] ],
		            'keys' : self.parameters[2].items()
		        }

		return json.dumps( model, separators=separators, indent=indent )

cdef class JointProbabilityTable( MultivariateDistribution ):
	"""
	A joint probability table. The primary difference between this and the
	conditional table is that the final column sums to one here. The joint
	table can be thought of as the conditional probability table normalized
	by the marginals of each parent.
	"""

	def __cinit__( self, table=None, neighbors=None, keys=None, frozen=False ):
		"""
		Take in the distribution represented as a list of lists, where each
		inner list represents a row.
		"""

		self.summaries = [{}, 0]
		self.name = "JointProbabilityTable"
		self.frozen = False

		if keys:
			self.parameters = [ table, neighbors, keys ]
		else:
			keys = []
			values = numpy.zeros( len(table) )

			for i, row in enumerate( table ):
				keys.append( ( tuple(row[:-1]), i ) )
				values[i] = _log( row[-1] )

			self.key_dict = dict(keys)
			keys = OrderedDict( keys[::-1] )
			self.parameters = [ values, neighbors, keys ]

	def __str__( self ):
		"""Regenerate the table."""

		values, parents, keys = self.parameters
		return "\n".join( 
					"\t".join( map( str, key + (cexp( values[idx] ),) ) )
							for key, idx in keys.items() ) 

	def __len__( self ):
		"""The length of the distribution is the number of keys."""

		return len( self.keys() )

	def sample( self ):
		"""Return a sample from the table."""

		values, neighbors, keys = self.parameters
		values = numpy.cumsum(numpy.exp(values))

		a = numpy.random.uniform(0, 1)
		for i in range(len(values)):
			if values[i] > a:
				return keys.keys()[i][-1]

	def keys( self ):
		"""
		Return the keys of the probability distribution which has parents,
		the child variable.
		"""

		return tuple(set(row[-1] for row in self.parameters[2].keys()))

	def log_probability( self, symbol ):
		"""
		Return the log probability of a value, which is a tuple in proper
		ordering, like the training data.
		"""

		# Unpack the parameters
		values, _, keys = self.parameters

		# Return the array element with that identity
		return values[ keys[symbol] ]

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
		values, neighbors, keys = self.parameters

		# If given a dictionary, convert to a list
		if isinstance( neighbor_values, dict ):
			neighbor_values = [ neighbor_values.get( d, None ) for d in neighbors ]
		if isinstance( neighbor_values, list ):
			wrt = neighbor_values.index( None )

		# Determine the keys for the respective parent distribution
		d = { k: 0 for k in neighbors[wrt].keys() }
		total = 0.0

		for key, idx in keys.items():
			logp = values[idx]

			if neighbor_values is not None:
				for j, k in enumerate( key ):
					if j == wrt:
						continue

					logp += neighbor_values[j].log_probability( k )

			p = cexp( logp )
			d[ key[wrt] ] += p
			total += p

		for key, value in d.items():
			d[key] = value / total
		
		return DiscreteDistribution( d )

	def summarize( self, items, weights=None ):
		"""Summarize the data into sufficient statistics to store."""

		if len(items) == 0 or self.frozen == True:
			return

		if weights is None:
			weights = numpy.ones( len(items), dtype='float64' )
		elif numpy.sum( weights ) == 0:
			return
		else:
			weights = numpy.asarray(weights, dtype='float64' )

		self._table_summarize(items, weights)

	cdef void _table_summarize(self, items, double [:] weights):
		cdef int i, n = len(items)
		cdef tuple item

		for i in range(n):
			item = tuple(items[i])
			self.summaries[0][item] = self.summaries[0].get(item, 0) + weights[i]
			self.summaries[1] += weights[i]

	def from_summaries( self, inertia=0.0, pseudocount=0.0 ):
		"""Update the parameters of the distribution using sufficient statistics."""

		values = numpy.zeros_like(self.parameters[0])
		keys = self.key_dict

		for key in self.parameters[2].keys():
			values[keys[key]] = self.summaries[0].get( key, 0.0 ) / self.summaries[1]

		self.parameters[0] = numpy.log(numpy.exp(self.parameters[0])*inertia + values*(1-inertia) + pseudocount)
		self.summaries = [{}, 0]

	def clear_summaries( self ):
		"""Clear the summary statistics stored in the object."""

		self.summaries = [{}, 0]

	def fit( self, items, weights=None, inertia=0.0, pseudocount=0.0 ):
		"""Update the parameters of the table based on the data."""

		self.summarize( items, weights )
		self.from_summaries( inertia, pseudocount )

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
					'class' : 'Distribution',
		            'name' : 'JointProbabilityTable',
		            'values' : self.parameters[0].tolist(),
		            'parents' : [ json.loads( dist.to_json() ) for dist in self.parameters[1] ],
		            'keys' : self.parameters[2].items()
		        }

		return json.dumps( model, separators=separators, indent=indent )

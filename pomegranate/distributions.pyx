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

import time
import scipy
from scipy.linalg.cython_blas cimport dgemm

import itertools as it
import json
import numpy
import random
import scipy.special
import sys

from .utils cimport pair_lse
from .utils cimport _log
from .utils cimport lgamma
from .utils cimport mdot

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

cdef class Distribution( Model ):
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
		plt.hist( self.sample(n), **kwargs )

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
		elif d['name'] == 'DiscreteDistribution':
			try:
				dist = {float(key) : value for key, value in d['parameters'][0].items()}
			except:
				dist = d['parameters'][0]

			return DiscreteDistribution(dist, frozen=d['frozen'])

		elif 'Table' in d['name']:
			parents = [ Distribution.from_json( json.dumps(j) ) for j in d['parents'] ]
			table = []
			
			for row in d['table']:
				table.append([])
				for item in row:
					try:
						table[-1].append(float(item))
					except:
						table[-1].append(item)

			if d['name'] == 'JointProbabilityTable':
				return JointProbabilityTable(table, parents)
			elif d['name'] == 'ConditionalProbabilityTable':
				return ConditionalProbabilityTable(table, parents)

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
		self.logp = -_log(end-start)

	def __reduce__( self ):
		"""Serialize distribution for pickling."""
		return self.__class__, (self.start, self.end, self.frozen)

	cdef double _log_probability( self, double symbol ) nogil:
		cdef double logp
		self._v_log_probability(&symbol, &logp, 1)
		return logp

	cdef void _v_log_probability(self, double* symbol, double* log_probability, int n) nogil:
		cdef int i
		for i in range(n):
			if symbol[i] >= self.start and symbol[i] <= self.end:
				log_probability[i] = self.logp
			else:
				log_probability[i] = NEGINF


	def sample( self, n=None ):
		"""Sample from this uniform distribution and return the value sampled."""
		return numpy.random.uniform(self.start, self.end, n)

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

		items, weights = weight_set(items, weights)
		if weights.sum() <= 0:
			return

		cdef double* items_p = <double*> (<numpy.ndarray> items).data
		cdef double* weights_p = <double*> (<numpy.ndarray> weights).data
		cdef SIZE_t n = items.shape[0]

		with nogil:
			self._summarize( items_p, weights_p, n )

	cdef double _summarize( self, double* items, double* weights, SIZE_t n ) nogil:
		cdef double minimum = INF, maximum = NEGINF
		cdef int i

		for i in range(n):
			if weights[i] > 0:
				if items[i] < minimum:
					minimum = items[i]
				if items[i] > maximum:
					maximum = items[i]

		with gil:
			if maximum > self.summaries[1]:
				self.summaries[1] = maximum
			if minimum < self.summaries[0]:
				self.summaries[0] = minimum

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
		self.logp = -_log(self.end - self.start)

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

cdef class BernoulliDistribution( Distribution ):
	"""A Bernoulli distribution describing the probability of a binary variable."""

	property parameters:
		def __get__( self ):
			return [self.p]
		def __set__( self, parameters ):
			self.p = parameters[0]
			self.logp[0] = _log(1-self.p)
			self.logp[1] = _log(self.p)

	def __cinit__(self, p, frozen=False):
		self.p = p
		self.name = "BernoulliDistribution"
		self.frozen = frozen
		self.logp = <double*> calloc(2, sizeof(double))
		self.logp[0] = _log(1-p)
		self.logp[1] = _log(p)
		self.summaries = [0.0, 0.0]

	def __dealloc__(self):
		free(self.logp)

	def __reduce__(self):
		"""Serialize distribution for pickling."""
		return self.__class__, (self.p, self.frozen)

	cdef double _log_probability(self, double symbol) nogil:
		cdef double logp
		self._v_log_probability(&symbol, &logp, 1)
		return logp

	cdef void _v_log_probability(self, double* symbol, double* log_probability, int n) nogil:
		cdef int i
		for i in range(n):
			log_probability[i] = self.logp[<int> symbol[i]]

	def sample( self, n=None ):
		return numpy.random.choice(2, p=[1-self.p, self.p], size=n)

	def summarize(self, items, weights=None):
		items, weights = weight_set(items, weights)
		if weights.sum() <= 0:
			return

		cdef double* items_p = <double*> (<numpy.ndarray> items).data
		cdef double* weights_p = <double*> (<numpy.ndarray> weights).data
		cdef SIZE_t n = items.shape[0]

		with nogil:
			self._summarize( items_p, weights_p, n )

	cdef double _summarize( self, double* items, double* weights, SIZE_t n ) nogil:
		cdef SIZE_t i
		cdef double w_sum = 0, x_sum = 0

		for i in range(n):
			w_sum += weights[i]
			if items[i] == 1:
				x_sum += weights[i]

		with gil:
			self.summaries[0] += w_sum
			self.summaries[1] += x_sum

	def from_summaries(self, inertia=0.0):
		"""Update the parameters of the distribution from the summaries."""

		p = self.summaries[1] / self.summaries[0] 
		self.p = self.p * inertia + p * (1-inertia)
		self.logp[0] = _log(1-p)
		self.logp[1] = _log(p)
		self.summaries = [0.0, 0.0]

	def fit(self, items, weights=None, inertia=0.0):
		"""Fit the parameter to maximize the likelihood of the samples."""

		self.summarize(items, weights)
		self.from_summaries(inertia)

	@classmethod
	def from_samples(self, items, weights=None):
		d = BernoulliDistribution(0.5)
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

	def __cinit__( self, mean, std, frozen=False, min_std=None ):
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
		self.min_std = min_std

	def __reduce__( self ):
		"""Serialize distribution for pickling."""
		return self.__class__, (self.mu, self.sigma, self.frozen)

	cdef double _log_probability( self, double symbol ) nogil:
		cdef double logp
		self._v_log_probability(&symbol, &logp, 1)
		return logp

	cdef void _v_log_probability(self, double* symbol, double* log_probability, int n) nogil:
		cdef int i
		for i in range(n):
			log_probability[i] = self.log_sigma_sqrt_2_pi - ((symbol[i] - self.mu) ** 2) /\
				self.two_sigma_squared

	def sample( self, n=None ):
		return numpy.random.normal(self.mu, self.sigma, n)

	def fit( self, items, weights=None, inertia=0.0, min_std=1e-5 ):
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
		cdef SIZE_t i
		cdef double x_sum = 0.0, x2_sum = 0.0, w_sum = 0.0

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

		min_std = self.min_std if self.min_std is not None else min_std

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
	def from_samples( cls, items, weights=None, min_std=1e-5 ):
		"""Fit a distribution to some data without pre-specifying it."""

		d = cls(0, 1)
		d.fit(items, weights, min_std=min_std)
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

	def __reduce__( self ):
		"""Serialize distribution for pickling."""
		return self.__class__, (self.mu, self.sigma, self.frozen)

	cdef double _log_probability( self, double symbol ) nogil:
		cdef double logp
		self._v_log_probability(&symbol, &logp, 1)
		return logp

	cdef void _v_log_probability(self, double* symbol, double* log_probability, int n) nogil:
		cdef int i
		for i in range(n):
			log_probability[i] = -_log( symbol[i] * self.sigma * SQRT_2_PI ) \
				- 0.5 * ((_log(symbol[i]) - self.mu) / self.sigma) ** 2			

	def sample( self, n=None ):
		"""Return a sample from this distribution."""
		return numpy.random.lognormal( self.mu, self.sigma, n )

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

	def clear_summaries( self ):
		"""Clear the summary statistics stored in the object."""

		self.summaries = [0, 0, 0]

	@classmethod
	def from_samples( cls, items, weights=None, min_std=1e-5 ):
		"""Fit a distribution to some data without pre-specifying it."""

		d = cls(0, 1)
		d.fit(items, weights, min_std=min_std)
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

	def __init__( self, double rate, bint frozen=False ):
		"""
		Make a new inverse gamma distribution. The parameter is called "rate"
		because lambda is taken.
		"""

		self.rate = rate
		self.summaries = [0, 0]
		self.name = "ExponentialDistribution"
		self.frozen = frozen
		self.log_rate = _log(rate)

	def __reduce__( self ):
		"""Serialize distribution for pickling."""
		return self.__class__, (self.rate, self.frozen)

	cdef double _log_probability( self, double symbol ) nogil:
		cdef double logp
		self._v_log_probability(&symbol, &logp, 1)
		return logp

	cdef void _v_log_probability( self, double* symbol, double* log_probability, int n ) nogil:
		cdef int i
		for i in range(n):
			log_probability[i] = self.log_rate - self.rate * symbol[i]

	def sample( self, n=None ):
		return numpy.random.exponential( 1. / self.parameters[0], n )

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
		self.log_rate = _log(self.rate)
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
			alpha, beta = parameters
			self.alpha, self.beta = alpha, beta
			self.beta_norm = lgamma(alpha+beta) - lgamma(alpha) - lgamma(beta)

	def __init__( self, alpha, beta, frozen=False ):
		"""
		Make a new beta distribution. Both alpha and beta are both shape
		parameters.
		"""

		self.alpha = alpha
		self.beta = beta
		self.beta_norm = lgamma(alpha+beta) - lgamma(alpha) - lgamma(beta)
		self.summaries = [0, 0]
		self.name = "BetaDistribution"
		self.frozen = frozen

	def __reduce__( self ):
		"""Serialize distribution for pickling."""
		return self.__class__, (self.alpha, self.beta, self.frozen)

	cdef double _log_probability( self, double symbol ) nogil:
		cdef double logp
		self._v_log_probability(&symbol, &logp, 1)
		return logp

	cdef void _v_log_probability( self, double* symbol, double* log_probability, int n ) nogil:
		cdef double alpha = self.alpha
		cdef double beta = self.beta
		cdef double beta_norm = self.beta_norm
		cdef int i

		for i in range(n):
			log_probability[i] = beta_norm + (alpha-1)*_log(symbol[i]) + \
				(beta-1)*_log(1-symbol[i])

	def sample( self, n=None ):
		"""Return a random sample from the beta distribution."""
		return numpy.random.beta( self.alpha, self.beta, n )

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

		cdef double alpha = 0, beta = 0
		cdef SIZE_t i

		for i in range(n):
			if items[i] == 1:
				alpha += weights[i]
			else:
				beta += weights[i]

		with gil:
			self.summaries[0] += alpha
			self.summaries[1] += beta

	def from_summaries( self, inertia=0.0 ):
		"""Use the summaries in order to update the distribution."""

		if self.frozen == True:
			return

		alpha, beta = self.summaries

		self.alpha = self.alpha*inertia + alpha*(1-inertia)
		self.beta = self.beta*inertia + beta*(1-inertia)
		self.beta_norm = lgamma(self.alpha+self.beta) - lgamma(self.alpha) - lgamma(self.beta)

		self.summaries = [0, 0]

	def clear_summaries( self ):
		"""Clear the summary statistics stored in the object."""

		self.summaries = [0, 0]

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
		self.summaries = [0, 0, 0]
		self.name = "GammaDistribution"
		self.frozen = frozen

	def __reduce__( self ):
		"""Serialize distribution for pickling."""
		return self.__class__, (self.alpha, self.beta, self.frozen)

	cdef double _log_probability( self, double symbol ) nogil:
		cdef double logp
		self._v_log_probability(&symbol, &logp, 1)
		return logp

	cdef void _v_log_probability( self, double* symbol, double* log_probability, int n ) nogil:
		cdef double alpha = self.alpha
		cdef double beta = self.beta
		cdef int i

		for i in range(n):
			log_probability[i] = (_log(beta) * alpha - lgamma(alpha) + 
				_log(symbol[i]) * (alpha - 1) - beta * symbol[i])

	def sample( self, n=None ):
		return numpy.random.gamma(self.parameters[0], 1.0 / self.parameters[1])

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

		self.summarize(items, weights)
		self.from_summaries(inertia, epsilon, iteration_limit)

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
		self.summaries[0] += items.dot(weights)
		self.summaries[1] += numpy.log(items).dot(weights)
		self.summaries[2] += weights.sum()

	cdef double _summarize(self, double* items, double* weights, int n) nogil:
		cdef int i
		cdef double xw = 0, logxw = 0, w = 0

		for i in range(n):
			w += weights[i]
			xw = items[i] * weights[i]
			logxw = _log(items[i]) * weights[i]

		with gil:
			self.summaries[0] += xw
			self.summaries[1] += logxw
			self.summaries[2] += w

	def from_summaries( self, inertia=0.0, epsilon=1e-4,
		iteration_limit=100 ):
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
		if self.summaries[2] < 1e-7 or self.frozen == True:
			return

		# First, do Newton-Raphson for shape parameter.

		# Calculate the sufficient statistic s, which is the log of the average
		# minus the average log. When computing the average log, we weight
		# outside the log function. (In retrospect, this is actually pretty
		# obvious.)
		statistic = _log(self.summaries[0] / self.summaries[2]) - \
			self.summaries[1] / self.summaries[2]

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
		rate = 1.0 / (1.0 / (shape * self.summaries[2]) * self.summaries[0])

		# Get the previous parameters
		prior_shape, prior_rate = self.parameters

		# Calculate the new parameters, respecting inertia, with an inertia
		# of 0 being completely replacing the parameters, and an inertia of
		# 1 being to ignore new training data.
		self.alpha = prior_shape*inertia + shape*(1-inertia)
		self.beta =	prior_rate*inertia + rate*(1-inertia)
		self.summaries = [0, 0, 0]

	def clear_summaries( self ):
		"""Clear the summary statistics stored in the object."""

		self.summaries = [0, 0, 0]

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
			return [self.dist]
		def __set__( self, parameters ):
			d = parameters[0]
			self.dist = d
			self.log_dist = {key: _log(value) for key, value in d.items()}

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

	def __reduce__( self ):
		"""Serialize the distribution for pickle."""
		return self.__class__, (self.dist, self.frozen)

	def __len__( self ):
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
		return tuple(self.dist.keys())

	def items( self ):
		"""Return items of the underlying dictionary."""
		return tuple(self.dist.items())

	def values( self ):
		"""Return values of the underlying dictionary."""
		return tuple(self.dist.values())

	def mle( self ):
		"""Return the maximally likely key."""

		max_key, max_value = None, 0
		for key, value in self.items():
			if value > max_value:
				max_key, max_value = key, value

		return max_key

	def bake( self, keys ):
		"""Encoding the distribution into integers."""

		if keys is None:
			return

		n = len(keys)
		self.encoded_keys = keys

		free(self.encoded_counts)
		free(self.encoded_log_probability)

		self.encoded_counts = <double*> calloc( n, sizeof(double) )
		self.encoded_log_probability = <double*> calloc( n, sizeof(double) )
		self.n = n

		for i in range(n):
			key = keys[i]
			self.encoded_counts[i] = 0
			self.encoded_log_probability[i] = self.log_dist.get( key, NEGINF )

	def log_probability( self, symbol ):
		"""Return the log prob of the symbol under this distribution."""

		return self.__log_probability( symbol )

	cdef double __log_probability( self, symbol ):
		return self.log_dist.get( symbol, NEGINF )

	cdef public double _log_probability( self, double symbol ) nogil:
		cdef double logp
		self._v_log_probability(&symbol, &logp, 1)
		return logp

	cdef void _v_log_probability(self, double* symbol, double* log_probability, int n) nogil:
		cdef int i
		for i in range(n):
			if symbol[i] < 0 or symbol[i] > self.n:
				log_probability[i] = NEGINF
			else:
				log_probability[i] = self.encoded_log_probability[<int> symbol[i]]

	def sample( self, n=None ):
		if n is None:
			rand = random.random()
			for key, value in self.items():
				if value >= rand:
					return key
				rand -= value		
		else:
			samples = [self.sample() for i in range(n)]
			return numpy.array(samples)
			

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

		if self.summaries[1] == 0 or self.frozen == True:
			return

		if self.encoded_summary == 0:
			_sum = sum( self.summaries[0].values() )
			characters = {}
			for key, value in self.summaries[0].items():
				self.dist[key] = self.dist[key]*inertia + (1-inertia)*value / _sum
				self.log_dist[key] = _log( self.dist[key] )

			self.bake( self.encoded_keys )
		else:
			n = len(self.encoded_keys)
			for i in range(n):
				key = self.encoded_keys[i]
				self.dist[key] = (self.dist[key]*inertia +
					(1-inertia)*self.encoded_counts[i] / self.summaries[1])
				self.log_dist[key] = _log( self.dist[key] )
				self.encoded_counts[i] = 0

			self.bake( self.encoded_keys )

		self.summaries = [{ key: 0 for key in self.keys() }, 0]

	def clear_summaries( self ):
		"""Clear the summary statistics stored in the object."""

		self.summaries = [{ key: 0 for key in self.keys() }, 0]
		if self.encoded_summary == 1:
			for i in range(len(self.encoded_keys)):
				self.encoded_counts[i] = 0

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
								'parameters' : [{str(key): value for key, value in self.dist.items()}],
								'frozen' : self.frozen
						   }, separators=separators, indent=indent )

	@classmethod
	def from_samples( cls, items, weights=None ):
		"""Fit a distribution to some data without pre-specifying it."""

		if weights is None:
			weights = numpy.ones( len(items) )

		characters = {}
		total = 0

		for character, weight in it.izip(items, weights):
			total += weight
			if character in characters:
				characters[character] += weight
			else:
				characters[character] = weight

		for character, weight in characters.items():
			characters[character] = weight / total

		d = DiscreteDistribution(characters)
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

	def __reduce__( self ):
		"""Serialize the distribution for pickle."""
		return self.__class__, (self.l, self.frozen)

	cdef double _log_probability( self, double symbol ) nogil:
		cdef double logp
		self._v_log_probability(&symbol, &logp, 1)
		return logp

	cdef void _v_log_probability(self, double* symbol, double* log_probability, int n) nogil:
		cdef double f
		cdef int i, j

		for i in range(n):
			f = 1.0

			if symbol[i] < 0 or self.l == 0:
				log_probability[i] = NEGINF
			elif symbol[i] > 0:
				for j in range(2, <int>symbol[i] + 1):
					f *= j
				log_probability[i] = symbol[i] * self.logl - self.l - _log(f)

	def sample( self, n=None ):
		return numpy.random.poisson( self.l, n )

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
		if self.frozen == True or self.summaries[0] < 1e-7:
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

	def __reduce__( self ):
		"""Serialize the distribution for pickle."""
		return self.__class__, (self.points_ndarray, self.bandwidth, self.weights_ndarray, self.frozen)

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
		self.name = "GaussianKernelDensity"

	cdef double _log_probability( self, double symbol ) nogil:
		cdef double logp
		self._v_log_probability(&symbol, &logp, 1)
		return logp

	cdef void _v_log_probability(self, double* symbol, double* log_probability, int n) nogil:
		cdef double mu, w, scalar = 1.0 / SQRT_2_PI, prob, b = self.bandwidth
		cdef int i, j

		for i in range(n):
			prob = 0.0

			for j in range(self.n):
				mu = self.points[j]
				w = self.weights[j]
				prob += w * scalar * cexp(-0.5*((mu-symbol[i]) / b) ** 2)

			log_probability[i] = _log(prob)

	def sample( self, n=None ):
		sigma = self.parameters[1]
		if n is None:
			mu = numpy.random.choice( self.parameters[0], p=self.parameters[2] )
			return numpy.random.normal( mu, sigma )
		else:
			mus = numpy.random.choice( self.parameters[0], n, p=self.parameters[2])
			samples = [numpy.random.normal(mu, sigma) for mu in mus]
			return numpy.array(samples)


cdef class UniformKernelDensity( KernelDensity ):
	"""
	A quick way of storing points to represent an Exponential kernel density in
	one dimension. Takes in points at initialization, and calculates the log of
	the sum of the Gaussian distances of the new point from every other point.
	"""

	def __cinit__( self, points=[], bandwidth=1, weights=None, frozen=False ):
		self.name = "UniformKernelDensity"

	cdef double _log_probability( self, double symbol ) nogil:
		cdef double logp
		self._v_log_probability(&symbol, &logp, 1)
		return logp

	cdef void _v_log_probability(self, double* symbol, double* log_probability, int n) nogil:
		cdef double mu, w, scalar = 1.0 / SQRT_2_PI, prob, b = self.bandwidth
		cdef int i, j

		for i in range(n):
			prob = 0.0

			for j in range(self.n):
				mu = self.points[j]
				w = self.weights[j]

				if fabs(mu - symbol[i]) <= b:
					prob += w

			log_probability[i] = _log(prob)

	def sample( self, n=None ):
		band = self.parameters[1]
		if n is None:
			mu = numpy.random.choice( self.parameters[0], p=self.parameters[2] )
			return numpy.random.uniform( mu-band, mu+band )
		else:
			mus = numpy.random.choice( self.parameters[0], n, p=self.parameters[2])
			samples = [numpy.random.uniform(mu-band, mu+band) for mu in mus]
			return numpy.array(samples)

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
		self.name = "TriangleKernelDensity"

	cdef double _log_probability( self, double symbol ) nogil:
		cdef double logp
		self._v_log_probability(&symbol, &logp, 1)
		return logp

	cdef void _v_log_probability(self, double* symbol, double* log_probability, int n) nogil:
		cdef double mu, w, scalar = 1.0 / SQRT_2_PI, prob
		cdef double hinge, b = self.bandwidth
		cdef int i, j

		for i in range(n):
			prob = 0.0

			for j in range(self.n):
				mu = self.points[j]
				w = self.weights[j]
				hinge = b - fabs(mu - symbol[i])
				if hinge > 0:
					prob += hinge * w

			log_probability[i] = _log(prob)

	def sample( self, n=None ):
		band = self.parameters[1]
		if n is None:
			mu = numpy.random.choice( self.parameters[0], p=self.parameters[2] )
			return numpy.random.triangular( mu-band, mu+band, mu )
		else:
			mus = numpy.random.choice( self.parameters[0], n, p=self.parameters[2])
			samples = [numpy.random.triangular(mu-band, mu+band, mu) for mu in mus]
			return numpy.array(samples)

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
		self.discrete = isinstance(distributions[0], DiscreteDistribution)

		if weights is not None:
			weights = numpy.array( weights, dtype=numpy.float64 )
		else:
			weights = numpy.ones( self.d, dtype=numpy.float64 )

		self.weights = numpy.log( weights )
		self.weights_ptr = <double*> self.weights.data
		self.name = "IndependentComponentsDistribution"
		self.frozen = frozen

	def __reduce__( self ):
		"""Serialize the distribution for pickle."""
		return self.__class__, (self.distributions, numpy.exp(self.weights), self.frozen)

	def log_probability( self, symbol ):
		"""
		What's the probability of a given tuple under this mixture? It's the
		product of the probabilities of each symbol in the tuple under their
		respective distribution, which is the sum of the log probabilities.
		"""

		cdef numpy.ndarray symbol_ndarray = numpy.array(symbol).astype('float64')
		cdef double* symbol_ptr = <double*> symbol_ndarray.data
		cdef double logp

		if self.discrete:
			logp = 0
			for i in range(self.d):
				logp += self.distributions[i].log_probability(symbol[i]) + self.weights[i]

		else:
			with nogil:
				logp = self._mv_log_probability( symbol_ptr )

		return logp

	cdef double _mv_log_probability( self, double* symbol ) nogil:
		cdef double logp
		self._v_log_probability(symbol, &logp, 1)
		return logp

	cdef void _v_log_probability(self, double* symbol, double* log_probability, int n) nogil:
		cdef int i, j, d = self.d
		cdef double logp

		for i in range(n):
			logp = 0.0

			for j in range(d):
				logp += (<Model> self.distributions_ptr[j])._log_probability(symbol[i*d+j])
				logp += self.weights_ptr[j]	

			log_probability[i] = logp

	def sample( self, n=None ):
		if n is None:
			return numpy.array([ d.sample() for d in self.parameters[0] ])
		else:
			return numpy.array([self.sample() for i in range(n)])

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
		cdef int n = items.shape[0]

		with nogil:
			self._summarize(items_ptr, weights_ptr, n)

	cdef double _summarize( self, double* items, double* weights, SIZE_t n ) nogil:
		cdef SIZE_t i, j, d = self.d
		cdef double logp = 0.0

		for i in range(n):
			for j in range(d):
				( <Model> self.distributions_ptr[j] )._summarize( items+i*d+j, weights+i, 1 )

	def from_summaries( self, inertia=0.0 ):
		"""
		Use the collected summary statistics in order to update the
		distributions.
		"""

		# If the distribution is frozen, don't bother with any calculation
		if self.frozen == True:
			return

		for d in self.parameters[0]:
			d.from_summaries(inertia=inertia)

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
		self.mu = numpy.array(means, dtype='float64')
		self._mu = <double*> self.mu.data
		self.cov = numpy.array(covariance, dtype='float64')
		self._cov = <double*> self.cov.data
		_, self._log_det = numpy.linalg.slogdet(self.cov)

		if self.mu.shape[0] != self.cov.shape[0]:
			raise ValueError("mu shape is {} while covariance shape is {}".format( self.mu.shape[0], self.cov.shape[0] ))
		if self.cov.shape[0] != self.cov.shape[1]:
			raise ValueError("covariance is not a square matrix, dimensions are ({}, {})".format( self.cov.shape[0], self.cov.shape[1] ) )
		if self._log_det == NEGINF:
			raise ValueError("covariance matrix is not invertible.")

		d = self.mu.shape[0]
		self.d = d
		self._inv_dot_mu = <double*> calloc(d, sizeof(double))

		chol = scipy.linalg.cholesky(self.cov, lower=True)
		self.inv_cov = scipy.linalg.solve_triangular(chol, numpy.eye(d), 
			lower=True).T
		self._inv_cov = <double*> self.inv_cov.data
		mdot(self._mu, self._inv_cov, self._inv_dot_mu, 1, d, d)

		self.w_sum = 0.0
		self.column_sum = <double*> calloc( d, sizeof(double) )
		self.pair_sum = <double*> calloc( d*d, sizeof(double) )
		memset( self.column_sum, 0, d*sizeof(double) )
		memset( self.pair_sum, 0, d*d*sizeof(double) )
		self._mu_new = <double*> calloc( d, sizeof(double) )

	def __reduce__( self ):
		"""Serialize the distribution for pickle."""
		return self.__class__, (self.mu, self.cov, self.frozen)

	def __dealloc__(self):
		free(self._mu_new)
		free(self.column_sum)
		free(self.pair_sum)

	def log_probability( self, symbol ):
		cdef numpy.ndarray symbol_ndarray = numpy.array(symbol).astype(numpy.float64)
		cdef double* symbol_ptr = <double*> symbol_ndarray.data
		cdef double logp

		with nogil:
			logp = self._mv_log_probability(symbol_ptr)
		return logp


	cdef double _mv_log_probability( self, double* symbol ) nogil:
		"""Cython optimized function for log probability calculation."""

		cdef SIZE_t i, j, d = self.d
		cdef double logp
		self._v_log_probability(symbol, &logp, 1)
		return logp

	cdef void _v_log_probability( self, double* symbol, double* logp, int n) nogil:
		cdef int i, j, d = self.d

		cdef double* dot = <double*> calloc(n*d, sizeof(double))
		mdot(symbol, self._inv_cov, dot, n, d, d)

		for i in range(n):
			logp[i] = 0
			for j in range(d):
				logp[i] += (dot[i*d + j] - self._inv_dot_mu[j])**2

			logp[i] = -0.5 * (d * LOG_2_PI + logp[i]) - 0.5 * self._log_det

		free(dot)

	def sample( self, n=None ):
		"""
		Sample from the mixture. First, choose a distribution to sample from
		according to the weights, then sample from that distribution.
		"""

		return numpy.random.multivariate_normal( self.parameters[0],
			self.parameters[1], n )

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
			raise ValueError("trying to fit data with {} dimensions to distribution \
				with {} distributions".format(d, self.d))

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


		cdef double* y = <double*> calloc(n*d, sizeof(double))

		cdef double alpha = 1
		cdef double beta = 0

		for i in range(n):
			w_sum += weights[i]

			for j in range(d):
				y[i*d + j] = items[i*d + j] * weights[i]
				column_sum[j] += y[i*d + j]

		dgemm('N', 'T', &d, &d, &n, &alpha, y, &d, items, &d, &beta, pair_sum, &d)

		with gil:
			self.w_sum += w_sum

			for j in range(d):
				self.column_sum[j] += column_sum[j]

				for k in range(d):
					self.pair_sum[j*d + k] += pair_sum[j*d + k]

		free(column_sum)
		free(pair_sum)
		free(y)

	def from_summaries( self, inertia=0.0, min_covar=1e-5 ):
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
		cdef double cov
		cdef numpy.ndarray chol

		for i in range(d):
			u[i] = self.column_sum[i] / self.w_sum
			self._mu[i] = self._mu[i] * inertia + u[i] * (1-inertia)

		for j in range(d):
			for k in range(d):
				cov = (pair_sum[j*d + k] - column_sum[j]*u[k]- column_sum[k]*u[j] + 
					self.w_sum*u[j]*u[k]) / self.w_sum
				self._cov[j*d + k] = self._cov[j*d + k] * inertia + cov * (1-inertia)

		memset( column_sum, 0, d*sizeof(double) )
		memset( pair_sum, 0, d*d*sizeof(double) )
		self.w_sum = 0.0

		try:
			chol = scipy.linalg.cholesky(self.cov, lower=True)
		except:
			# Taken from sklearn.gmm, it's possible there are not enough observations
			# to get a good measurement, so reinitialize this component.
			self.cov += min_covar * numpy.eye(d)
			chol = scipy.linalg.cholesky(self.cov, lower=True)

		_, self._log_det = numpy.linalg.slogdet(self.cov)

		self.inv_cov = scipy.linalg.solve_triangular(chol, numpy.eye(d), 
			lower=True).T
		self._inv_cov = <double*> self.inv_cov.data
		mdot(self._mu, self._inv_cov, self._inv_dot_mu, 1, d, d)

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


cdef class DirichletDistribution( MultivariateDistribution ):
	"""A Dirichlet distribution, usually a prior for the multinomial distributions."""

	property parameters:
		def __get__( self ):
			return [ self.alphas.tolist() ]
		def __set__( self, alphas ):
			self.alphas = numpy.array(alphas, dtype='float64')
			self.alphas_ptr = <double*> self.alphas.data
			self.beta_norm = lgamma(sum(alphas)) - sum([lgamma(alpha) for alpha in alphas])

	def __init__(self, alphas, frozen=False):
		self.name = "DirichletDistribution"
		self.frozen = frozen
		self.d = len(alphas)
		
		self.alphas = numpy.array(alphas, dtype='float64')
		self.alphas_ptr = <double*> self.alphas.data
		self.beta_norm = lgamma(sum(alphas)) - sum([lgamma(alpha) for alpha in alphas])
		self.summaries_ndarray = numpy.zeros(self.d, dtype='float64')
		self.summaries_ptr = <double*> self.summaries_ndarray.data

	def log_probability( self, symbol ):
		cdef numpy.ndarray symbol_ndarray = numpy.array(symbol, dtype='float64')
		cdef double* symbol_ptr = <double*> symbol_ndarray.data
		cdef double logp

		with nogil:
			logp = self._mv_log_probability(symbol_ptr)
		return logp

	cdef double _mv_log_probability( self, double* symbol ) nogil:
		cdef double logp
		self._v_log_probability(symbol, &logp, 1)
		return logp

	cdef void _v_log_probability(self, double* symbol, double* log_probability, int n) nogil:
		cdef int i, j, d = self.d
		cdef double logp

		for i in range(n):
			log_probability[i] = self.beta_norm

			for j in range(d):
				log_probability[i] += self.alphas_ptr[j] * _log(symbol[i*d + j])

	def sample( self, n=None ):
		return numpy.random.dirichlet(self.alphas, n)

	def summarize( self, items, weights=None ):
		"""
		Take in a series of items and their weights and reduce it down to a
		summary statistic to be used in training later.
		"""

		items, weights = weight_set( items, weights )

		cdef double* items_ptr = <double*> (<numpy.ndarray> items).data
		cdef double* weights_ptr = <double*> (<numpy.ndarray> weights).data

		cdef SIZE_t n = items.shape[0]

		with nogil:
			self._summarize( items_ptr, weights_ptr, n )

	cdef double _summarize( self, double* items, double* weights, SIZE_t n ) nogil:
		"""Calculate sufficient statistics for a minibatch.

		The sufficient statistics for a dirichlet distribution is just the
		weighted count of the times each thing appears.
		"""

		cdef int i, j, d = self.d

		for i in range(n):
			for j in range(d):
				self.summaries_ptr[j] += items[i*d + j] * weights[i]

	def from_summaries( self, inertia=0.0, pseudocount=0.0 ):
		"""Update the internal parameters of the distribution."""

		if self.frozen == True:
			return

		self.summaries_ndarray += pseudocount
		alphas = self.summaries_ndarray * (1-inertia) + self.alphas * inertia 
		
		self.alphas = alphas
		self.alphas_ptr = <double*> self.alphas.data
		self.beta_norm = lgamma(sum(alphas)) - sum([lgamma(alpha) for alpha in alphas])
		self.summaries_ndarray *= 0

	def clear_summaries( self ):
		self.summaries_ndarray *= 0

	def fit( self, items, weights=None, inertia=0.0, pseudocount=0.0 ):
		self.summarize(items, weights)
		self.from_summaries(inertia, pseudocount)

	@classmethod
	def from_samples( cls, items, weights=None ):
		d = len(items[0])
		dist = DirichletDistribution(numpy.zeros(d))
		dist.fit(items, weights)
		return dist


cdef class ConditionalProbabilityTable( MultivariateDistribution ):
	"""
	A conditional probability table, which is dependent on values from at
	least one previous distribution but up to as many as you want to
	encode for.
	"""

	def __init__( self, table, parents, frozen=False ):
		"""
		Take in the distribution represented as a list of lists, where each
		inner list represents a row.
		"""

		self.name = "ConditionalProbabilityTable"
		self.frozen = False
		self.m = len(parents)
		self.n = len(table)
		self.k = len(set(row[-2] for row in table))
		self.idxs = <int*> calloc(self.m+1, sizeof(int))
		self.marginal_idxs = <int*> calloc(self.m, sizeof(int))

		self.values = <double*> calloc(self.n, sizeof(double))
		self.counts = <double*> calloc(self.n, sizeof(double))
		self.marginal_counts = <double*> calloc(self.n / self.k, sizeof(double))

		memset(self.counts, 0, self.n*sizeof(double))
		memset(self.marginal_counts, 0, self.n*sizeof(double)/self.k)
		
		self.idxs[0] = 1
		self.idxs[1] = self.k
		for i in range(self.m-1):
			self.idxs[i+2] = len(parents[self.m-i-1])

		self.marginal_idxs[0] = 1
		for i in range(self.m-1):
			self.marginal_idxs[i+1] = len(parents[self.m-i-1])

		keys = []
		for i, row in enumerate( table ):
			keys.append( ( tuple(row[:-1]), i ) )
			self.values[i] = _log( row[-1] )

		self.keymap = OrderedDict(keys)

		marginal_keys = []
		for i, row in enumerate( table[::self.k] ):
			marginal_keys.append( ( tuple(row[:-2]), i ) ) 

		self.marginal_keymap = OrderedDict(marginal_keys)
		self.parents = parents
		self.parameters = [ table, self.parents ]

	def __dealloc__(self):
		free(self.idxs)
		free(self.values)
		free(self.counts)
		free(self.marginal_idxs)
		free(self.marginal_counts)

	def __reduce__( self ):
		"""Serialize the distribution for pickle."""
		return self.__class__, (self.parameters[0], self.parents, self.frozen)

	def __str__( self ):
		return "\n".join(
					"\t".join( map( str, key + (cexp( self.values[idx] ),)))
							for key, idx in self.keymap.items() )

	def __len__( self ):
		return self.k

	def keys( self ):
		"""
		Return the keys of the probability distribution which has parents,
		the child variable.
		"""

		return tuple(set(row[-1] for row in self.keymap.keys()))

	def bake( self, keys ):
		"""Order the inputs according to some external global ordering."""

		keymap, marginal_keymap, values = [], set([]), []
		for i, key in enumerate(keys):
			keymap.append((key, i))
			idx = self.keymap[key]
			values.append(self.values[idx])

		marginal_keys = []
		for i, row in enumerate( keys[::self.k] ):
			marginal_keys.append( ( tuple(row[:-1]), i ) ) 

		self.marginal_keymap = OrderedDict(marginal_keys)

		for i in range(len(keys)):
			self.values[i] = values[i]

		self.keymap = OrderedDict(keymap)

	def sample( self, parent_values={} ):
		"""Return a random sample from the conditional probability table."""

		keys = self.keymap.keys()
		for parent in self.parents:
			if parent not in parent_values:
				parent_values[parent] = parent.sample()

		n = len(keys)
		idxs = []
		values_ = []

		for i in range(n):
			for j, parent in enumerate(self.parents):
				if parent_values[parent] != keys[i][j]:
					break
			else:
				idxs.append(i)
				values_.append(cexp(self.values[i]))

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

		idx = self.keymap[tuple(symbol)]
		return self.values[idx]

	cdef double _mv_log_probability( self, double* symbol ) nogil:
		cdef int i, idx = 0

		for i in range(self.m+1):
			idx += self.idxs[i] * <int> symbol[self.m-i]

		return self.values[idx]

	cdef void _v_log_probability( self, double* symbol, double* log_probability, int n ) nogil:
		cdef int i, j, idx

		for i in range(n):
			idx = 0
			for j in range(self.m+1):
				idx += self.idxs[j] * <int> symbol[self.m-j]

			log_probability[i] = self.values[idx]

	def joint( self, neighbor_values=None ):
		"""
		This will turn a conditional probability table into a joint
		probability table. If the data is already a joint, it will likely
		mess up the data. It does so by scaling the parameters the probabilities
		by the parent distributions.
		"""

		neighbor_values = neighbor_values or self.parents+[None]
		if isinstance( neighbor_values, dict ):
			neighbor_values = [ neighbor_values.get( p, None ) for p in self.parents + [self]]

		table, total = [], 0
		for key, idx in self.keymap.items():
			scaled_val = self.values[idx]

			for j, k in enumerate( key ):
				if neighbor_values[j] is not None:
					scaled_val += neighbor_values[j].log_probability( k )

			scaled_val = cexp(scaled_val)
			total += scaled_val
			table.append( key + (scaled_val,) )

		table = [ row[:-1] + (row[-1] / total,) for row in table ]
		return JointProbabilityTable( table, self.parents )

	def marginal( self, neighbor_values=None ):
		"""
		Calculate the marginal of the CPT. This involves normalizing to turn it
		into a joint probability table, and then summing over the desired
		value.
		"""

		# Convert from a dictionary to a list if necessary
		if isinstance( neighbor_values, dict ):
			neighbor_values = [ neighbor_values.get( d, None ) for d in self.parents ]

		# Get the index we're marginalizing over
		i = -1 if neighbor_values == None else neighbor_values.index( None )
		return self.joint(neighbor_values).marginal(i)

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

		self.__summarize(items, weights)

	cdef void __summarize(self, items, double [:] weights):
		cdef int i, n = len(items)
		cdef tuple item 

		for i in range(n):
			key = self.keymap[tuple(items[i])]
			self.counts[key] += weights[i]
			
			key = self.marginal_keymap[tuple(items[i][:-1])]
			self.marginal_counts[key] += weights[i]

	cdef double _summarize(self, double* items, double* weights, int n ) nogil:
		cdef int i, j, idx
		cdef double* counts = <double*> calloc(self.n, sizeof(double))
		cdef double* marginal_counts = <double*> calloc(self.n / self.k, sizeof(double))

		for i in range(n):
			idx = 0
			for j in range(self.m+1):
				idx += self.idxs[i] * <int> items[self.m-i]

			counts[idx] += weights[i]

			idx = 0
			for j in range(self.m):
				idx += self.marginal_idxs[i] * <int> items[self.m-1-i]

			marginal_counts[idx] += weights[i]

		with gil:
			for i in range(n):
				self.counts[i] += counts[i]
				if i < self.n / self.k:
					self.marginal_counts[i] += marginal_counts[i]

		free(counts)
		free(marginal_counts)

	def from_summaries( self, double inertia=0.0, double pseudocount=0.0 ):
		"""Update the parameters of the distribution using sufficient statistics."""

		cdef int i, k

		with nogil:
			for i in range(self.n):
				k = i / self.k

				if self.marginal_counts[k] > 0:
					probability = ((self.counts[i] + pseudocount) / 
						(self.marginal_counts[k] + pseudocount * self.k))

					self.values[i] = _log(cexp(self.values[i])*inertia + 
						probability*(1-inertia))

				else:
					self.values[i] = -_log(self.k)

		for i in range(self.n):
			self.parameters[0][i][-1] = cexp(self.values[i])

		self.clear_summaries()

	def clear_summaries( self ):
		"""Clear the summary statistics stored in the object."""

		with nogil:
			memset(self.counts, 0, self.n*sizeof(double))
			memset(self.marginal_counts, 0, self.n*sizeof(double)/self.k)

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

		table = [list(key + tuple([cexp(self.values[i])])) for key, i in self.keymap.items() ]
		table = [[str(item) for item in row] for row in table]

		model = {
					'class' : 'Distribution',
		            'name' : 'ConditionalProbabilityTable',
		            'table' : table,
		            'parents' : [ json.loads( dist.to_json() ) for dist in self.parents ]
		        }

		return json.dumps( model, separators=separators, indent=indent )

	@classmethod
	def from_samples(cls, X, parents, weights=None):
		"""Learn the table from data."""

		X = numpy.array(X)
		n, d = X.shape

		keys = [numpy.unique(X[:,i]) for i in range(d)]

		table = []
		for key in it.product(*keys):
			table.append( list(key) + [1./len(keys[-1]),] )

		d = ConditionalProbabilityTable(table, parents)
		d.fit(X, weights)
		return d

cdef class JointProbabilityTable( MultivariateDistribution ):
	"""
	A joint probability table. The primary difference between this and the
	conditional table is that the final column sums to one here. The joint
	table can be thought of as the conditional probability table normalized
	by the marginals of each parent.
	"""

	def __cinit__( self, table, parents, frozen=False ):
		"""
		Take in the distribution represented as a list of lists, where each
		inner list represents a row.
		"""

		self.name = "JointProbabilityTable"
		self.frozen = False
		self.m = len(parents)
		self.n = len(table)
		self.k = len(set(row[-2] for row in table))
		self.idxs = <int*> calloc(self.m+1, sizeof(int))

		self.values = <double*> calloc(self.n, sizeof(double))
		self.counts = <double*> calloc(self.n, sizeof(double))
		self.count = 0

		memset(self.counts, 0, self.n*sizeof(double))
		
		self.idxs[0] = 1
		self.idxs[1] = self.k
		for i in range(self.m-1):
			self.idxs[i+2] = len(parents[self.m-i-1])

		keys = []
		for i, row in enumerate( table ):
			keys.append( ( tuple(row[:-1]), i ) )
			self.values[i] = _log( row[-1] )

		self.keymap = OrderedDict(keys)
		self.parents = parents
		self.parameters = [ table, self.parents ]

	def __dealloc__(self):
		free(self.values)
		free(self.counts)

	def __reduce__( self ):
		return self.__class__, (self.parameters[0], self.parents, self.frozen)

	def __str__( self ):
		return "\n".join(
					"\t".join( map( str, key + (cexp(self.values[idx] ),) ) )
							for key, idx in self.keymap.items() )

	def __len__( self ):
		return self.k

	def sample( self, n=None ):
		a = numpy.random.uniform(0, 1)
		for i in range(self.n):
			if cexp(self.values[i]) > a:
				return self.keymap.keys()[i][-1]

	def bake( self, keys ):
		"""Order the inputs according to some external global ordering."""

		keymap, values = [], []
		for i, key in enumerate(keys):
			keymap.append((key, i))
			idx = self.keymap[key]
			values.append(self.values[idx])

		for i in range(len(keys)):
			self.values[i] = values[i]

		self.keymap = OrderedDict(keymap)

	def keys( self ):
		return tuple(set(row[-1] for row in self.parameters[2].keys()))

	def log_probability( self, symbol ):
		"""
		Return the log probability of a value, which is a tuple in proper
		ordering, like the training data.
		"""

		key = self.keymap[tuple(symbol)]
		return self.values[key]

	cdef double _mv_log_probability( self, double* symbol ) nogil:
		cdef int i, idx = 0

		for i in range(self.m+1):
			idx += self.idxs[i] * <int> symbol[self.m-i]

		return self.values[idx]

	cdef void _v_log_probability( self, double* symbol, double* log_probability, int n ) nogil:
		cdef int i, j, idx

		for i in range(n):
			idx = 0
			for j in range(self.m+1):
				idx += self.idxs[j] * <int> symbol[self.m-j]

			log_probability[i] = self.values[idx]

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

		if isinstance(neighbor_values, dict):
			neighbor_values = [ neighbor_values.get( d, None ) for d in self.parents ]
		
		if isinstance(neighbor_values, list):
			wrt = neighbor_values.index(None)

		# Determine the keys for the respective parent distribution
		d = { k: 0 for k in self.parents[wrt].keys() }
		total = 0.0

		for key, idx in self.keymap.items():
			logp = self.values[idx]

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

		return DiscreteDistribution(d)

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

	cdef void __summarize(self, items, double [:] weights):
		cdef int i, n = len(items)
		cdef tuple item 

		for i in range(n):
			key = self.keymap[tuple(items[i])]
			self.counts[key] += weights[i]

	cdef double _summarize(self, double* items, double* weights, int n ) nogil:
		cdef int i, j, idx
		cdef double count = 0
		cdef double* counts = <double*> calloc(self.n, sizeof(double))

		for i in range(n):
			idx = 0
			for j in range(self.m+1):
				idx += self.idxs[i] * <int> items[self.m-i]

			counts[idx] += weights[i]
			count += weights[i]

		with gil:
			self.count += count
			for i in range(n):
				self.counts[i] += counts[i]

		free(counts)

	def from_summaries( self, double inertia=0.0, double pseudocount=0.0 ):
		"""Update the parameters of the distribution using sufficient statistics."""

		cdef int i, k
		cdef double p = pseudocount

		with nogil:
			for i in range(self.n):
				probability = ((self.counts[i] + p) / (self.count + p * self.k))
				self.values[i] = _log(cexp(self.values[i])*inertia + 
					probability*(1-inertia))

		for i in range(self.n):
			self.parameters[0][i][-1] = cexp(self.values[i])

		self.clear_summaries()

	def clear_summaries( self ):
		"""Clear the summary statistics stored in the object."""

		self.count = 0
		with nogil:
			memset(self.counts, 0, self.n*sizeof(double))

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

		table = [list(key + tuple([cexp(self.values[i])])) for key, i in self.keymap.items() ]

		model = {
					'class' : 'Distribution',
		            'name' : 'JointProbabilityTable',
		            'table' : table,
		            'parents' : [ json.loads( dist.to_json() ) for dist in self.parameters[1] ]
		        }

		return json.dumps( model, separators=separators, indent=indent )

	@classmethod
	def from_samples(cls, X, parents, weights=None):
		"""Learn the table from data."""

		X = numpy.array(X)
		n, d = X.shape

		keys = [numpy.unique(X[:,i]) for i in range(d)]
		m = numpy.prod([k.shape[0] for k in keys])

		table = []
		for key in it.product(*keys):
			table.append( list(key) + [1./m,] )

		d = ConditionalProbabilityTable(table, parents)
		d.fit(X, weights)
		return d

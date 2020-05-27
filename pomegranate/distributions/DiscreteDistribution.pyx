#!python
#cython: boundscheck=False
#cython: cdivision=True
# DiscreteDistribution.pyx
# Contact: Jacob Schreiber <jmschreiber91@gmail.com>

import numpy
import itertools as it
import json
import random

from libc.stdlib cimport calloc
from libc.stdlib cimport free
from libc.stdlib cimport malloc

from ..utils cimport _log
from ..utils cimport isnan
from ..utils import check_random_state
from ..utils import _check_nan

from libc.math cimport sqrt as csqrt

# Define some useful constants
DEF NEGINF = float("-inf")
DEF INF = float("inf")
eps = numpy.finfo(numpy.float64).eps

cdef class DiscreteDistribution(Distribution):
	"""
	A discrete distribution, made up of characters and their probabilities,
	assuming that these probabilities will sum to 1.0.
	"""

	property parameters:
		def __get__(self):
			return [self.dist]
		def __set__(self, parameters):
			d = parameters[0]
			self.dist = d
			self.log_dist = {key: _log(value) for key, value in d.items()}

	def __cinit__(self, dict characters, bint frozen=False):
		"""
		Make a new discrete distribution with a dictionary of discrete
		characters and their probabilities, checking to see that these
		sum to 1.0. Each discrete character can be modelled as a
		Bernoulli distribution.
		"""

		if len(characters) == 0:
			raise ValueError("Must pass in a dictionary with at least one value.")

		self.name = "DiscreteDistribution"
		self.frozen = frozen
		self.dtype = str(type(list(characters.keys())[0])).split()[-1].strip('>').strip("'")

		self.dist = characters.copy()
		self.log_dist = { key: _log(value) for key, value in characters.items() }
		self.summaries =[{ key: 0 for key in characters.keys() }, 0]

		self.encoded_summary = 0
		self.encoded_keys = None
		self.encoded_counts = NULL
		self.encoded_log_probability = NULL

	def __dealloc__(self):
		if self.encoded_keys is not None:
			free(self.encoded_counts)
			free(self.encoded_log_probability)

	def __reduce__(self):
		"""Serialize the distribution for pickle."""
		return self.__class__, (self.dist, self.frozen)

	def __len__(self):
		return len(self.dist)

	def __mul__(self, other):
		"""Multiply this by another distribution sharing the same keys."""

		self_keys = self.keys()
		other_keys = other.keys()
		distribution, total = {}, 0.0

		if isinstance(other, DiscreteDistribution) and self_keys == other_keys:
			self_values = (<DiscreteDistribution>self).dist.values()
			other_values = (<DiscreteDistribution>other).dist.values()
			for key, x, y in zip(self_keys, self_values, other_values):
				if _check_nan(key):
					distribution[key] = (1 + eps) * (1 + eps)
				else:
					distribution[key] = (x + eps) * (y + eps)
				total += distribution[key]
		else:
			assert set(self_keys) == set(other_keys)
			self_items = (<DiscreteDistribution>self).dist.items()
			for key, x in self_items:
				if _check_nan(key):
					x = 1.
				y = other.probability(key)
				distribution[key] = (x + eps) * (y + eps)
				total += distribution[key]

		for key in self_keys:
			distribution[key] /= total

			if distribution[key] <= eps / total:
				distribution[key] = 0.0
			elif distribution[key] >= 1 - eps / total:
				distribution[key] = 1.0

		return DiscreteDistribution(distribution)


	def equals(self, other):
		"""Return if the keys and values are equal"""

		if not isinstance(other, DiscreteDistribution):
			return False

		self_keys = self.keys()
		other_keys = other.keys()

		if self_keys == other_keys:
			self_values = (<DiscreteDistribution>self).log_dist.values()
			other_values = (<DiscreteDistribution>other).log_dist.values()
			for key, self_prob, other_prob in zip(self_keys, self_values, other_values):
				if _check_nan(key):
					continue
				self_prob = round(self_prob, 12)
				other_prob = round(other_prob, 12)
				if self_prob != other_prob:
					return False
		elif set(self_keys) == set(other_keys):
			self_items = (<DiscreteDistribution>self).log_dist.items()
			for key, self_prob in self_items:
				if _check_nan(key):
					self_prob = 0.
				else:
					self_prob = round(self_prob, 12)
				other_prob = round(other.log_probability(key), 12)
				if self_prob != other_prob:
					return False
		else:
			return False

		return True

	def clamp(self, key):
		"""Return a distribution clamped to a particular value."""
		return DiscreteDistribution({ k : 0. if k != key else 1. for k in self.keys() })

	def keys(self):
		"""Return the keys of the underlying dictionary."""
		return tuple(self.dist.keys())

	def items(self):
		"""Return items of the underlying dictionary."""
		return tuple(self.dist.items())

	def values(self):
		"""Return values of the underlying dictionary."""
		return tuple(self.dist.values())

	def mle(self):
		"""Return the maximally likely key."""

		max_key, max_value = None, 0
		for key, value in self.items():
			if value > max_value:
				max_key, max_value = key, value

		return max_key

	def bake(self, keys):
		"""Encoding the distribution into integers."""

		if keys is None:
			return

		n = len(keys)
		self.encoded_keys = keys

		free(self.encoded_counts)
		free(self.encoded_log_probability)

		self.encoded_counts = <double*> calloc(n, sizeof(double))
		self.encoded_log_probability = <double*> malloc(n * sizeof(double))
		self.n = n

		for i in range(n):
			key = keys[i]
			self.encoded_log_probability[i] = self.log_dist.get(key, NEGINF)

	def probability(self, X):
		"""Return the prob of the X under this distribution."""

		return self.__probability(X)

	cdef double __probability(self, X):
		if _check_nan(X):
			return 1.
		else:
			return self.dist.get(X, 0)

	def log_probability(self, X):
		"""Return the log prob of the X under this distribution."""

		return self.__log_probability(X)

	cdef double __log_probability(self, X):
		if _check_nan(X):
			return 0.
		else:
			return self.log_dist.get(X, NEGINF)

	cdef void _log_probability(self, double* X, double* log_probability, int n) nogil:
		cdef int i
		for i in range(n):
			if isnan(X[i]):
				log_probability[i] = 0.
			elif X[i] < 0 or X[i] > self.n:
				log_probability[i] = NEGINF
			else:
				log_probability[i] = self.encoded_log_probability[<int> X[i]]

	def sample(self, n=None, random_state=None):
		random_state = check_random_state(random_state)

		keys = list(self.dist.keys())
		probabilities = list(self.dist.values())

		if n is None:
			return random_state.choice(keys, p=probabilities)
		else:
			return random_state.choice(keys, p=probabilities, size=n)

	def fit(self, items, weights=None, inertia=0.0, pseudocount=0.0,
		column_idx=0):
		"""
		Set the parameters of this Distribution to maximize the likelihood of
		the given sample. Items holds some sort of sequence. If weights is
		specified, it holds a sequence of value to weight each item by.
		"""

		if self.frozen:
			return

		self.summarize(items, weights, column_idx)
		self.from_summaries(inertia, pseudocount)

	def summarize(self, items, weights=None, column_idx=0):
		"""Reduce a set of observations to sufficient statistics."""

		if weights is None:
			weights = numpy.ones(len(items))
		else:
			weights = numpy.asarray(weights)
			
		for i in range(len(items)):
			x = items[i]
			if _check_nan(x) == False:
				self.summaries[0][x] += weights[i]
				self.summaries[1] += weights[i]

	cdef double _summarize(self, double* items, double* weights, int n,
		int column_idx, int d) nogil:
		cdef int i
		cdef double item
		self.encoded_summary = 1

		encoded_counts = <double*> calloc(self.n, sizeof(double))

		for i in range(n):
			item = items[i*d + column_idx]
			if isnan(item):
				continue

			encoded_counts[<int> item] += weights[i]

		with gil:
			for i in range(self.n):
				self.encoded_counts[i] += encoded_counts[i]
				self.summaries[1] += encoded_counts[i]

		free(encoded_counts)

	def from_summaries(self, inertia=0.0, pseudocount=0.0):
		"""Use the summaries in order to update the distribution."""

		if self.summaries[1] == 0 or self.frozen == True:
			return

		if self.encoded_summary == 0:
			values = self.summaries[0].values()
			_sum = sum(values) + pseudocount * len(values)
			characters = {}
			for key, value in self.summaries[0].items():
				value += pseudocount
				self.dist[key] = self.dist[key]*inertia + (1-inertia)*(value / _sum)
				self.log_dist[key] = _log(self.dist[key])

			self.bake(self.encoded_keys)
		else:
			n = len(self.encoded_keys)
			for i in range(n):
				_sum = self.summaries[1] + pseudocount * n
				value = self.encoded_counts[i] + pseudocount

				key = self.encoded_keys[i]
				self.dist[key] = self.dist[key]*inertia + (1-inertia)*(value / _sum)
				self.log_dist[key] = _log(self.dist[key])
				self.encoded_counts[i] = 0

			self.bake(self.encoded_keys)

		self.summaries = [{ key: 0 for key in self.keys() }, 0]

	def clear_summaries(self):
		"""Clear the summary statistics stored in the object."""

		self.summaries = [{ key: 0 for key in self.keys() }, 0]
		if self.encoded_summary == 1:
			for i in range(len(self.encoded_keys)):
				self.encoded_counts[i] = 0

	def to_json(self, separators=(',', ' :'), indent=4):
		"""Serialize the distribution to a JSON.

		Parameters
		----------
		separators : tuple, optional
			The two separators to pass to the json.dumps function for formatting.
			Default is (',', ' : ').

		indent : int, optional
			The indentation to use at each level. Passed to json.dumps for
			formatting. Default is 4.

		Returns
		-------
		json : str
			A properly formatted JSON object.
		"""

		return json.dumps({
								'class' : 'Distribution',
								'dtype' : self.dtype,
								'name'  : self.name,
								'parameters' : [{str(key): value for key, value in self.dist.items()}],
								'frozen' : self.frozen
						   }, separators=separators, indent=indent)

	@classmethod
	def from_samples(cls, items, weights=None, pseudocount=0, keys=None):
		"""Fit a distribution to some data without pre-specifying it."""

		if weights is None:
			weights = numpy.ones(len(items))

		keys = keys or numpy.unique(items)
		counts = {key: 0 for key in keys if not _check_nan(key)}
		total = 0

		for X, weight in zip(items, weights):
			if _check_nan(X):
				continue

			total += weight
			counts[X] += weight

		n = len(counts)
		for X, weight in counts.items():
			counts[X] = (weight + pseudocount) / (total + pseudocount * n)

		return cls(counts)

	@classmethod
	def blank(cls):
		return cls({'None': 1.0})

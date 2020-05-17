#!python
#cython: boundscheck=False
#cython: cdivision=True
# ExponentialDistribution.pyx
# Contact: Jacob Schreiber <jmschreiber91@gmail.com>

import numpy

from ..utils cimport _log
from ..utils cimport isnan
from ..utils import check_random_state

from libc.math cimport sqrt as csqrt

# Define some useful constants
DEF NEGINF = float("-inf")
DEF INF = float("inf")

cdef class ExponentialDistribution(Distribution):
	"""Represents an exponential distribution on non-negative floats."""

	property parameters:
		def __get__(self):
			return [self.rate]
		def __set__(self, parameters):
			self.rate = parameters[0]

	def __init__(self, double rate, bint frozen=False):
		self.rate = rate
		self.summaries = [0, 0]
		self.name = "ExponentialDistribution"
		self.frozen = frozen
		self.log_rate = _log(rate)

	def __reduce__(self):
		"""Serialize distribution for pickling."""
		return self.__class__, (self.rate, self.frozen)

	cdef void _log_probability(self, double* X, double* log_probability, int n) nogil:
		cdef int i
		for i in range(n):
			if isnan(X[i]):
				log_probability[i] = 0.
			else:
				log_probability[i] = self.log_rate - self.rate * X[i]

	def sample(self, n=None, random_state=None):
		random_state = check_random_state(random_state)
		return random_state.exponential(1. / self.parameters[0], n)

	cdef double _summarize(self, double* items, double* weights, int n,
		int column_idx, int d) nogil:
		"""Cython function to get the MLE estimate for an exponential."""

		cdef int i
		cdef double xw_sum = 0, w = 0
		cdef double item

		# Calculate the average, which is the MLE mu estimate
		for i in range(n):
			item = items[i*d + column_idx]
			if isnan(item):
				continue

			xw_sum += item * weights[i]
			w += weights[i]

		with gil:
			self.summaries[0] += w
			self.summaries[1] += xw_sum

	def from_summaries(self, inertia=0.0):
		"""
		Takes in a series of summaries, represented as a mean, a variance, and
		a weight, and updates the underlying distribution. Notes on how to do
		this for a Gaussian distribution were taken from here:
		http://math.stackexchange.com/questions/453113/how-to-merge-two-gaussians
		"""

		if self.frozen == True or self.summaries[0] < 1e-7:
			return

		self.rate = (self.summaries[0] + 1e-7) / (self.summaries[1] + 1e-7)
		self.log_rate = _log(self.rate)
		self.summaries = [0, 0]

	def clear_summaries(self):
		"""Clear the summary statistics stored in the object."""

		self.summaries = [0, 0]

	@classmethod
	def blank(cls):
		return cls(1)

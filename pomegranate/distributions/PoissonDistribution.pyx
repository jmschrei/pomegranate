#!python
#cython: boundscheck=False
#cython: cdivision=True
# PoissonDistribution.pyx
# Contact: Jacob Schreiber <jmschreiber91@gmail.com>

import numpy

from ..utils cimport _log
from ..utils cimport isnan
from ..utils cimport lgamma
from ..utils import check_random_state

from libc.math cimport sqrt as csqrt

from .distributions cimport Distribution

# Define some useful constants
DEF NEGINF = float("-inf")
DEF INF = float("inf")

cdef class PoissonDistribution(Distribution):
	"""The probability of a number of events occuring in a fixed time window.

	A probability distribution which expresses the probability of a
	number of events occurring in a fixed time window. It assumes these events
	occur with at a known rate, and independently of each other. It can
	operate over both integer and float values.
	"""

	property parameters:
		def __get__(self):
			return [self.l]
		def __set__(self, parameters):
			self.l = parameters[0]

	def __init__(self, l, frozen=False):
		self.l = l
		self.logl = _log(l)
		self.name = "PoissonDistribution"
		self.summaries = [0, 0]
		self.frozen = frozen

	def __reduce__(self):
		"""Serialize the distribution for pickle."""
		return self.__class__, (self.l, self.frozen)

	cdef void _log_probability(self, double* X, double* log_probability, int n) nogil:
		cdef int i

		for i in range(n):
			if isnan(X[i]):
				log_probability[i] = 0.
			elif X[i] < 0 or self.l == 0:
				log_probability[i] = NEGINF
			else:
				log_probability[i] = X[i] * self.logl - self.l - lgamma(X[i]+1)

	def sample(self, n=None, random_state=None):
		random_state = check_random_state(random_state)
		return random_state.poisson(self.l, n)

	cdef double _summarize(self, double* items, double* weights, int n,
		int column_idx, int d) nogil:
		"""Cython optimized function to calculate the summary statistics."""

		cdef int i
		cdef double x_sum = 0.0, w_sum = 0.0
		cdef double item

		for i in range(n):
			item = items[i*d + column_idx]
			if isnan(item):
				continue

			x_sum += item * weights[i]
			w_sum += weights[i]

		with gil:
			self.summaries[0] += x_sum
			self.summaries[1] += w_sum

	def from_summaries(self, inertia=0.0):
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

	def clear_summaries(self):
		"""Clear the summary statistics stored in the object."""

		self.summaries = [0, 0]

	@classmethod
	def blank(cls):
		return PoissonDistribution(0)

#!python
#cython: boundscheck=False
#cython: cdivision=True
# UniformDistribution.pyx
# Contact: Jacob Schreiber <jmschreiber91@gmail.com>

import numpy

from ..utils cimport _log
from ..utils cimport isnan
from ..utils import check_random_state

# Define some useful constants
DEF NEGINF = float("-inf")
DEF INF = float("inf")

cdef class UniformDistribution(Distribution):
	"""A uniform distribution between two values."""

	property parameters:
		def __get__(self):
			return [self.start, self.end]
		def __set__(self, parameters):
			self.start, self.end = parameters

	def __init__(UniformDistribution self, double start, double end, bint frozen=False):
		self.start = start
		self.end = end
		self.summaries = [INF, NEGINF, 0]
		self.name = "UniformDistribution"
		self.frozen = frozen
		self.logp = -_log(end-start)

	def __reduce__(self):
		"""Serialize distribution for pickling."""
		return self.__class__, (self.start, self.end, self.frozen)

	cdef void _log_probability(self, double* X, double* log_probability, int n) nogil:
		cdef int i
		for i in range(n):
			if isnan(X[i]):
				log_probability[i] = 0.
			elif X[i] >= self.start and X[i] <= self.end:
				log_probability[i] = self.logp
			else:
				log_probability[i] = NEGINF

	def sample(self, n=None, random_state=None):
		"""Sample from this uniform distribution and return the value sampled."""
		random_state = check_random_state(random_state)
		return random_state.uniform(self.start, self.end, n)

	cdef double _summarize(self, double* items, double* weights, int n,
		int column_idx, int d) nogil:
		cdef int i
		cdef double minimum = INF, maximum = NEGINF
		cdef double item, weight = 0.0

		for i in range(n):
			item = items[i*d + column_idx]
			if isnan(item):
				continue

			weight += weights[i]
			if weights[i] > 0:
				if item < minimum:
					minimum = item
				if item > maximum:
					maximum = item

		with gil:
			self.summaries[2] += weight
			if maximum > self.summaries[1]:
				self.summaries[1] = maximum
			if minimum < self.summaries[0]:
				self.summaries[0] = minimum

	def from_summaries(self, inertia=0.0):
		"""
		Takes in a series of summaries, consisting of the minimum and maximum
		of a sample, and determine the global minimum and maximum.
		"""

		# If the distribution is frozen, don't bother with any calculation
		if self.frozen == True or self.summaries[2] == 0:
			return

		minimum, maximum = self.summaries[:2]
		self.start = minimum*(1-inertia) + self.start*inertia
		self.end = maximum*(1-inertia) + self.end*inertia
		self.logp = -_log(self.end - self.start)

		self.summaries = [INF, NEGINF, 0]

	def clear_summaries(self):
		"""Clear the summary statistics stored in the object."""

		self.summaries = [INF, NEGINF, 0]

	@classmethod
	def blank(cls):
		return cls(0, 0)

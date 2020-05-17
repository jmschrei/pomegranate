#!python
#cython: boundscheck=False
#cython: cdivision=True
# BernoulliDistribution.pyx
# Contact: Jacob Schreiber <jmschreiber91@gmail.com>

import numpy

from libc.stdlib cimport malloc
from libc.stdlib cimport free

from ..utils cimport _log
from ..utils cimport isnan
from ..utils import check_random_state

# Define some useful constants
DEF NEGINF = float("-inf")
DEF INF = float("inf")

cdef class BernoulliDistribution(Distribution):
	"""A Bernoulli distribution describing the probability of a binary variable."""

	property parameters:
		def __get__(self):
			return [self.p]
		def __set__(self, parameters):
			self.p = parameters[0]
			self.logp[0] = _log(1-self.p)
			self.logp[1] = _log(self.p)

	def __cinit__(self, p, frozen=False):
		self.p = p
		self.name = "BernoulliDistribution"
		self.frozen = frozen
		self.logp = <double*> malloc(2*sizeof(double))
		self.logp[0] = _log(1-p)
		self.logp[1] = _log(p)
		self.summaries = [0.0, 0.0]

	def __dealloc__(self):
		free(self.logp)

	def __reduce__(self):
		"""Serialize distribution for pickling."""
		return self.__class__, (self.p, self.frozen)

	cdef void _log_probability(self, double* X, double* log_probability, int n) nogil:
		cdef int i
		for i in range(n):
			if isnan(X[i]):
				log_probability[i] = 0.
			else:
				log_probability[i] = self.logp[<int> X[i]]

	def sample(self, n=None, random_state=None):
		random_state = check_random_state(random_state)
		return random_state.choice(2, p=[1-self.p, self.p], size=n)

	cdef double _summarize(self, double* items, double* weights, int n,
		int column_idx, int d) nogil:
		cdef int i
		cdef double w_sum = 0, x_sum = 0
		cdef double item

		for i in range(n):
			item = items[i*d + column_idx]
			if isnan(item):
				continue

			w_sum += weights[i]
			if item == 1:
				x_sum += weights[i]

		with gil:
			self.summaries[0] += w_sum
			self.summaries[1] += x_sum

	def from_summaries(self, inertia=0.0):
		"""Update the parameters of the distribution from the summaries."""

		if self.summaries[0] < 1e-8 or self.frozen:
			return

		p = self.summaries[1] / self.summaries[0]
		self.p = self.p * inertia + p * (1-inertia)
		self.logp[0] = _log(1-p)
		self.logp[1] = _log(p)
		self.summaries = [0.0, 0.0]

	@classmethod
	def blank(cls):
		return cls(0)

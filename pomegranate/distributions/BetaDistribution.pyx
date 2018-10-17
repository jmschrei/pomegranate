#!python
#cython: boundscheck=False
#cython: cdivision=True
# BetaDistribution.pyx
# Contact: Jacob Schreiber <jmschreiber91@gmail.com>

import numpy

from ..utils cimport _log
from ..utils cimport isnan
from ..utils cimport lgamma
from ..utils import check_random_state

# Define some useful constants
DEF NEGINF = float("-inf")
DEF INF = float("inf")

cdef class BetaDistribution(Distribution):
	"""
	This distribution represents a beta distribution, parameterized using
	alpha/beta, which are both shape parameters. ML estimation is done
	"""

	property parameters:
		def __get__(self):
			return [self.alpha, self.beta]
		def __set__(self, parameters):
			alpha, beta = parameters
			self.alpha, self.beta = alpha, beta
			self.beta_norm = lgamma(alpha+beta) - lgamma(alpha) - lgamma(beta)

	def __init__(self, alpha, beta, frozen=False):
		self.alpha = alpha
		self.beta = beta
		self.beta_norm = lgamma(alpha+beta) - lgamma(alpha) - lgamma(beta)
		self.summaries = [0, 0]
		self.name = "BetaDistribution"
		self.frozen = frozen

	def __reduce__(self):
		"""Serialize distribution for pickling."""
		return self.__class__, (self.alpha, self.beta, self.frozen)

	cdef void _log_probability(self, double* X, double* log_probability, int n) nogil:
		cdef int i
		cdef double alpha = self.alpha
		cdef double beta = self.beta
		cdef double beta_norm = self.beta_norm

		for i in range(n):
			if isnan(X[i]):
				log_probability[i] = 0.
			else:
				log_probability[i] = beta_norm + (alpha-1)*_log(X[i]) + \
					(beta-1)*_log(1-X[i])

	def sample(self, n=None, random_state=None):
		random_state = check_random_state(random_state)
		return random_state.beta(self.alpha, self.beta, n)

	cdef double _summarize(self, double* items, double* weights, int n,
		int column_idx, int d) nogil:
		"""Cython optimized function for summarizing some data."""

		cdef int i
		cdef double alpha = 0, beta = 0
		cdef double item

		for i in range(n):
			item = items[i*d + column_idx]
			if isnan(item):
				continue

			if item == 1:
				alpha += weights[i]
			else:
				beta += weights[i]

		with gil:
			self.summaries[0] += alpha
			self.summaries[1] += beta

	def from_summaries(self, inertia=0.0):
		"""Use the summaries in order to update the distribution."""

		if self.frozen == True:
			return

		alpha, beta = self.summaries

		self.alpha = self.alpha*inertia + alpha*(1-inertia)
		self.beta = self.beta*inertia + beta*(1-inertia)
		self.beta_norm = lgamma(self.alpha+self.beta) - lgamma(self.alpha) - lgamma(self.beta)

		self.summaries = [0, 0]

	def clear_summaries(self):
		"""Clear the summary statistics stored in the object."""

		self.summaries = [0, 0]

	@classmethod
	def blank(cls):
		return BetaDistribution(0, 0)

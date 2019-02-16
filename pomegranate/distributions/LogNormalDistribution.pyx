#!python
#cython: boundscheck=False
#cython: cdivision=True
# LogNormalDistribution.pyx
# Contact: Jacob Schreiber <jmschreiber91@gmail.com>

import numpy

from ..utils cimport _log
from ..utils cimport isnan
from ..utils import check_random_state

from libc.math cimport sqrt as csqrt

# Define some useful constants
DEF NEGINF = float("-inf")
DEF INF = float("inf")
DEF SQRT_2_PI = 2.50662827463
DEF LOG_2_PI = 1.83787706641

cdef class LogNormalDistribution(Distribution):
	"""A lognormal distribution over non-negative floats.

	The parameters are the mu and sigma of the normal distribution, which 
	is the the exponential of the log normal distribution.
	"""

	property parameters:
		def __get__(self):
			return [self.mu, self.sigma]
		def __set__(self, parameters):
			self.mu, self.sigma = parameters

	def __init__(self, double mu, double sigma, double min_std=0.0, frozen=False):
		self.mu = mu
		self.sigma = sigma
		self.summaries = [0, 0, 0]
		self.name = "LogNormalDistribution"
		self.frozen = frozen
		self.min_std = min_std

	def __reduce__(self):
		"""Serialize distribution for pickling."""
		return self.__class__, (self.mu, self.sigma, self.frozen)

	cdef void _log_probability(self, double* X, double* log_probability, int n) nogil:
		cdef int i
		for i in range(n):
			if isnan(X[i]):
				log_probability[i] = 0.
			else:
				log_probability[i] = -_log(X[i] * self.sigma * SQRT_2_PI) - 0.5\
					* ((_log(X[i]) - self.mu) / self.sigma) ** 2

	def sample(self, n=None, random_state=None):
		random_state = check_random_state(random_state)
		return random_state.lognormal(self.mu, self.sigma, n)

	cdef double _summarize(self, double* items, double* weights, int n,
		int column_idx, int d) nogil:
		"""Cython function to get the MLE estimate for a Gaussian."""

		cdef int i
		cdef double x_sum = 0.0, x2_sum = 0.0, w_sum = 0.0
		cdef double item, log_item

		for i in range(n):
			item = items[i*d + column_idx]
			if isnan(item):
				continue

			log_item = _log(item)
			w_sum += weights[i]
			x_sum += weights[i] * log_item
			x2_sum += weights[i] * log_item * log_item

		with gil:
			self.summaries[0] += w_sum
			self.summaries[1] += x_sum
			self.summaries[2] += x2_sum

	def from_summaries(self, inertia=0.0):
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
		if sigma < self.min_std:
			sigma = self.min_std

		self.mu = self.mu*inertia + mu*(1-inertia)
		self.sigma = self.sigma*inertia + sigma*(1-inertia)
		self.summaries = [0, 0, 0]

	def clear_summaries(self):
		"""Clear the summary statistics stored in the object."""

		self.summaries = [0, 0, 0]

	@classmethod
	def blank(cls):
		return LogNormalDistribution(0, 1)


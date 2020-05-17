#!python
#cython: boundscheck=False
#cython: cdivision=True
# NormalDistribution.pyx
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

cdef class NormalDistribution(Distribution):
	"""A normal distribution based on a mean and standard deviation."""

	property parameters:
		def __get__(self):
			return [self.mu, self.sigma]
		def __set__(self, parameters):
			self.mu, self.sigma = parameters

	def __init__(self, mean, std, frozen=False, min_std=0.0):
		self.mu = mean
		self.sigma = std
		self.name = "NormalDistribution"
		self.frozen = frozen
		self.summaries = [0, 0, 0]
		self.log_sigma_sqrt_2_pi = -_log(std * SQRT_2_PI)
		self.two_sigma_squared = 1. / (2 * std ** 2)
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
				log_probability[i] = self.log_sigma_sqrt_2_pi - ((X[i] - self.mu) ** 2) *\
					self.two_sigma_squared

	def sample(self, n=None, random_state=None):
		random_state = check_random_state(random_state)
		return random_state.normal(self.mu, self.sigma, n)

	cdef double _summarize(self, double* items, double* weights, int n,
		int column_idx, int d) nogil:
		cdef int i, j
		cdef double x_sum = 0.0, x2_sum = 0.0, w_sum = 0.0
		cdef double item

		for i in range(n):
			item = items[i*d + column_idx]
			if isnan(item):
				continue

			w_sum += weights[i]
			x_sum += weights[i] * item
			x2_sum += weights[i] * item * item

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
		if self.summaries[0] < 1e-8 or self.frozen == True:
			return

		mu = self.summaries[1] / self.summaries[0]
		var = self.summaries[2] / self.summaries[0] - self.summaries[1] ** 2.0 / self.summaries[0] ** 2.0

		sigma = csqrt(var)
		if sigma < self.min_std:
			sigma = self.min_std

		self.mu = self.mu*inertia + mu*(1-inertia)
		self.sigma = self.sigma*inertia + sigma*(1-inertia)
		self.summaries = [0, 0, 0]
		self.log_sigma_sqrt_2_pi = -_log(sigma * SQRT_2_PI)
		self.two_sigma_squared = 1. / (2 * sigma ** 2) if sigma > 0 else 0

	def clear_summaries(self):
		"""Clear the summary statistics stored in the object."""

		self.summaries = [0, 0, 0]

	@classmethod
	def blank(cls):
		return cls(0, 1)

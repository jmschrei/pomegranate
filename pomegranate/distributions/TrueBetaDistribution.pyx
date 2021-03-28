#!python
#cython: boundscheck=False
#cython: cdivision=True
# NormalDistribution.pyx

import numpy

from ..utils cimport _log
from ..utils cimport isnan
from ..utils import check_random_state
from ..utils cimport lgamma


from libc.math cimport sqrt as csqrt

# Define some useful constants
DEF NEGINF = float("-inf")
DEF INF = float("inf")


cdef class TrueBetaDistribution(Distribution):
	"""beta distribution https://en.wikipedia.org/wiki/Beta_distribution"""

	property parameters:
		def __get__(self):
			return [self.alpha, self.beta]
		def __set__(self, parameters):
			self.alpha, self.beta = parameters

	def __init__(self, alpha, beta, frozen=False, min_alpha_beta=0.0, x_eps=1e-6):
		self.alpha = alpha
		self.beta = beta
		self.min_alpha_beta = min_alpha_beta
		self.x_eps = x_eps
		self.name = "TrueBetaDistribution"
		self.frozen = frozen
		self.summaries = [0, 0, 0]
		

	def __reduce__(self):
		"""Serialize distribution for pickling."""
		return self.__class__, (self.alpha, self.beta, self.frozen)

	cdef void _log_probability(self, double* X, double* log_probability, int n) nogil:
		cdef int i
		cdef double x_i

		for i in range(n):
			if isnan(X[i]):
				log_probability[i] = 0.
			else:
				x_i = X[i]
				if x_i < self.x_eps:
					x_i = self.x_eps
				if x_i > 1 - self.x_eps:
					x_i = 1 - self.x_eps
				log_probability[i] = (self.alpha - 1) * _log(x_i) + (self.beta - 1) * _log(1 - x_i) + lgamma(self.alpha + self.beta) - lgamma(self.alpha) - lgamma(self.beta)
									 
	def sample(self, n=None, random_state=None):
		random_state = check_random_state(random_state)
		return random_state.beta(self.alpha, self.beta, size=n)

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

		x_bar = self.summaries[1] / self.summaries[0]
		s2 = self.summaries[2] / self.summaries[0] - self.summaries[1] ** 2.0 / self.summaries[0] ** 2.0


		alpha = x_bar * ((x_bar * (1 - x_bar)) / s2 - 1)
		beta = alpha * (1 - x_bar) /x_bar

		if alpha < self.min_alpha_beta:
			alpha = self.min_alpha_beta
		
		if beta < self.min_alpha_beta:
			beta = self.min_alpha_beta

		self.alpha = self.alpha * inertia + alpha * (1-inertia)
		self.beta = self.beta * inertia+ beta * (1-inertia)
		self.summaries = [0, 0, 0]

	def clear_summaries(self):
		"""Clear the summary statistics stored in the object."""

		self.summaries = [0, 0, 0]

	@classmethod
	def blank(cls):
		return cls(2, 2)

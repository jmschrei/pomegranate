#!python
#cython: boundscheck=False
#cython: cdivision=True
# NormalDistribution.pyx
# Contact: Jacob Schreiber <jmschreiber91@gmail.com>

import numpy

from ..utils cimport lgamma
from ..utils cimport _log
from ..utils cimport isnan


# Constants
DEF LOG_2_PI = 1.83787706641 #@TODO: not true, cp from NormalDist
DEF LOG_PI = 0.49714987269





cdef class TDistribution(Distribution):
	"""
	A Student's t-distribution based on degree of freedom df.
	"""

	property parameters:
		def __get__(self):
			return [self.df]
		def __set__(self, parameters):
			self.df = parameters

	def __cinit__(self, df, frozen=False):
		"""
		Make a new Student's T-distribution with the given degree of freedom df.
		"""

		self.df = df
		self.name = "TDistribution"
		self.frozen = frozen
		self.summaries = [0]

	def __reduce__(self):
		"""Serialize distribution for pickling."""
		return self.__class__, (self.df)

	cdef void _log_probability(self, double* X, double* log_probability, int n) nogil:
		cdef int i
		for i in range(n):
			if isnan(X[i]):
				log_probability[i] = 0.
			else:
				log_probability[i] = lgamma(0.5 * (self.df + 1)) - lgamma(0.5 * self.df) \
                             - 0.5 * _log(self.df) - 0.5 * LOG_PI \
                             - 0.5 * (self.df + 1) + _log(1 + X[i] ** 2 / self.df)

	def sample(self, n=None):
		return numpy.random.standard_t(self.df, n)

	cdef double _summarize(self, double* items, double* weights, int n,
		int column_idx, int d) nogil:

		with gil:
			self.summaries[0] += n


	def from_summaries(self, inertia=0.0):
		""

		# If no summaries stored or the summary is frozen, don't do anything.
		if self.summaries[0] < 1e-8 or self.frozen == True:
			return

		self.df = self.summaries[0]
		self.summaries = [0]


	def clear_summaries(self):
		"""Clear the summary statistics stored in the object."""

		self.summaries = [0]

	@classmethod
	def blank(cls):
		return TDistribution(0)

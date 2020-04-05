#!python
#cython: boundscheck=False
#cython: cdivision=True
# DirichletDistribution.pyx
# Contact: Jacob Schreiber <jmschreiber91@gmail.com>

import numpy

from ..utils cimport pair_lse
from ..utils cimport _log
from ..utils cimport lgamma
from ..utils cimport isnan
from ..utils import check_random_state


cdef class DirichletDistribution(MultivariateDistribution):
	"""A Dirichlet distribution, usually a prior for the multinomial distributions."""

	property parameters:
		def __get__(self):
			return [self.alphas.tolist()]
		def __set__(self, alphas):
			self.alphas = numpy.array(alphas, dtype='float64')
			self.alphas_ptr = <double*> self.alphas.data
			self.beta_norm = lgamma(sum(alphas)) - sum([lgamma(alpha) for alpha in alphas])

	def __init__(self, alphas, frozen=False):
		self.name = "DirichletDistribution"
		self.frozen = frozen
		self.d = len(alphas)

		self.alphas = numpy.array(alphas, dtype='float64')
		self.alphas_ptr = <double*> self.alphas.data
		self.beta_norm = lgamma(sum(alphas)) - sum([lgamma(alpha) for alpha in alphas])
		self.summaries_ndarray = numpy.zeros(self.d, dtype='float64')
		self.summaries_ptr = <double*> self.summaries_ndarray.data

	cdef void _log_probability(self, double* X, double* log_probability, int n) nogil:
		cdef int i, j, d = self.d
		cdef double logp

		for i in range(n):
			log_probability[i] = self.beta_norm

			for j in range(d):
				log_probability[i] += self.alphas_ptr[j] * _log(X[i*d + j])

	def sample(self, n=None, random_state=None):
		random_state = check_random_state(random_state)
		return random_state.dirichlet(self.alphas, n)

	cdef double _summarize(self, double* X, double* weights, int n,
		int column_idx, int d) nogil:
		"""Calculate sufficient statistics for a minibatch.

		The sufficient statistics for a dirichlet distribution is just the
		weighted count of the times each thing appears.
		"""

		cdef int i, j

		for i in range(n):
			for j in range(d):
				self.summaries_ptr[j] += X[i*d + j] * weights[i]

	def from_summaries(self, inertia=0.0, pseudocount=0.0):
		"""Update the internal parameters of the distribution."""

		if self.frozen == True:
			return

		self.summaries_ndarray += pseudocount
		alphas = self.summaries_ndarray * (1-inertia) + self.alphas * inertia

		self.alphas = alphas
		self.alphas_ptr = <double*> self.alphas.data
		self.beta_norm = lgamma(sum(alphas)) - sum([lgamma(alpha) for alpha in alphas])
		self.summaries_ndarray *= 0

	def clear_summaries(self):
		self.summaries_ndarray *= 0

	def fit(self, X, weights=None, inertia=0.0, pseudocount=0.0):
		self.summarize(X, weights)
		self.from_summaries(inertia, pseudocount)

	@classmethod
	def from_samples(cls, X, weights=None, **kwargs):
		"""Fit a distribution to some data without pre-specifying it."""

		distribution = cls.blank(X.shape[1])
		distribution.fit(X, weights, **kwargs)
		return distribution

	@classmethod
	def blank(cls, d=2):
		return DirichletDistribution(numpy.zeros(d))

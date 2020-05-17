#!python
#cython: boundscheck=False
#cython: cdivision=True
# MultivariateGaussianDistribution.pyx
# Contact: Jacob Schreiber <jmschreiber91@gmail.com>

import numpy
import scipy
import json

try:
	import cupy
except:
	cupy = object

from libc.stdlib cimport calloc
from libc.stdlib cimport free
from libc.string cimport memset

from scipy.linalg.cython_blas cimport dgemm

from ..utils cimport _log
from ..utils cimport mdot
from ..utils cimport ndarray_wrap_cpointer
from ..utils cimport _is_gpu_enabled
from ..utils cimport isnan
from ..utils import check_random_state

from libc.math cimport sqrt as csqrt

# Define some useful constants
DEF NEGINF = float("-inf")
DEF INF = float("inf")
DEF SQRT_2_PI = 2.50662827463
DEF LOG_2_PI = 1.83787706641

eps = 1e-8

cdef class MultivariateGaussianDistribution(MultivariateDistribution):
	property parameters:
		def __get__(self):
			return [self.mu.tolist(), self.cov.tolist()]
		def __set__(self, parameters):
			self.mu = numpy.array(parameters[0])
			self.cov = numpy.array(parameters[1])

	def __cinit__(self, means=[], covariance=[], frozen=False):
		"""
		Take in the mean vector and the covariance matrix.
		"""

		self.name = "MultivariateGaussianDistribution"
		self.frozen = frozen
		self.mu = numpy.array(means, dtype='float64')
		self._mu = <double*> self.mu.data
		self.cov = numpy.array(covariance, dtype='float64')
		self._cov = <double*> self.cov.data
		_, self._log_det = numpy.linalg.slogdet(self.cov)

		if self.mu.shape[0] != self.cov.shape[0]:
			raise ValueError("mu shape is {} while covariance shape is {}".format(self.mu.shape[0], self.cov.shape[0]))
		if self.cov.shape[0] != self.cov.shape[1]:
			raise ValueError("covariance is not a square matrix, dimensions are ({}, {})".format(self.cov.shape[0], self.cov.shape[1]))
		if self._log_det == NEGINF:
			raise ValueError("covariance matrix is not invertible.")

		d = self.mu.shape[0]
		self.d = d
		self._inv_dot_mu = <double*> calloc(d, sizeof(double))
		self._mu_new = <double*> calloc(d, sizeof(double))

		chol = scipy.linalg.cholesky(self.cov, lower=True)
		self.inv_cov = scipy.linalg.solve_triangular(chol, numpy.eye(d), lower=True).T
		self._inv_cov = <double*> self.inv_cov.data
		mdot(self._mu, self._inv_cov, self._inv_dot_mu, 1, d, d)

		self.column_sum = <double*> calloc(d*d, sizeof(double))
		self.column_w_sum = <double*> calloc(d, sizeof(double))
		self.pair_sum = <double*> calloc(d*d, sizeof(double))
		self.pair_w_sum = <double*> calloc(d*d, sizeof(double))
		self.clear_summaries()

	def __reduce__(self):
		"""Serialize the distribution for pickle."""
		return self.__class__, (self.mu, self.cov, self.frozen)

	def __dealloc__(self):
		free(self._inv_dot_mu)
		free(self._mu_new)
		free(self.column_sum)
		free(self.column_w_sum)
		free(self.pair_sum)
		free(self.pair_w_sum)

	cdef void _log_probability(self, double* X, double* logp, int n) nogil:
		cdef int i, j, d = self.d
		cdef double* dot

		if _is_gpu_enabled():
			with gil:
				x = ndarray_wrap_cpointer(X, n*d).reshape(n, d)
				x1 = cupy.array(x)
				x2 = cupy.array(self.inv_cov)
				dot_ndarray = cupy.dot(x1, x2).get()
				dot = <double*> (<numpy.ndarray> dot_ndarray).data
		else:
			dot = <double*> calloc(n*d, sizeof(double))
			mdot(X, self._inv_cov, dot, n, d, d)

		for i in range(n):
			logp[i] = 0
			for j in range(d):
				if isnan(X[i*d + j]):
					logp[i] = self._log_probability_missing(X+i*d)
					break
				else:
					logp[i] += (dot[i*d + j] - self._inv_dot_mu[j])**2
			else:
				logp[i] = -0.5 * (d * LOG_2_PI + logp[i]) - 0.5 * self._log_det

		if not _is_gpu_enabled():
			free(dot)

	cdef double _log_probability_missing(self, double* X) nogil:
		cdef double logp

		with gil:
			X_ndarray = ndarray_wrap_cpointer(X, self.d)
			avail = ~numpy.isnan(X_ndarray)
			if avail.sum() == 0:
				return 0

			a = numpy.ix_(avail, avail)


			d1 = MultivariateGaussianDistribution(self.mu[avail], self.cov[a])
			logp = d1.log_probability(X_ndarray[avail])

		return logp

	def sample(self, n=None, random_state=None):
		random_state = check_random_state(random_state)
		return random_state.multivariate_normal(self.parameters[0],
			self.parameters[1], n)

	cdef double _summarize(self, double* X, double* weights, int n,
		int column_idx, int d) nogil:
		"""Calculate sufficient statistics for a minibatch.

		The sufficient statistics for a multivariate gaussian update is the sum of
		each column, and the sum of the outer products of the vectors.
		"""

		cdef int i, j, k
		cdef double x, w, sqrt_weight, w_sum = 0.0
		cdef double* column_sum = <double*> calloc(d*d, sizeof(double))
		cdef double* column_w_sum = <double*> calloc(d, sizeof(double))
		cdef double* pair_sum
		cdef double* pair_w_sum = <double*> calloc(d*d, sizeof(double))

		cdef double* y = <double*> calloc(n*d, sizeof(double))
		cdef double alpha = 1
		cdef double beta = 0

		for i in range(n):
			w = weights[i]
			w_sum += w
			sqrt_weight = csqrt(w)

			for j in range(d):
				x = X[i*d + j]
				if isnan(x):
					y[i*d + j] = 0.

					for k in range(d):
						pair_w_sum[j*d + k] -= w
						if not isnan(X[i*d + k]):
							pair_w_sum[k*d + j] -= w
							column_sum[k*d + j] -= X[i*d + k] * w

				else:
					y[i*d + j] = x * sqrt_weight
					column_sum[j*d + j] += x * w
					column_w_sum[j] += w

		if _is_gpu_enabled():
			with gil:
				x_ndarray = ndarray_wrap_cpointer(y, n*d).reshape(n, d)
				x_gpu = cupy.array(x_ndarray, copy=False)
				pair_sum_ndarray = cupy.dot(x_gpu.T, x_gpu).get()

				for j in range(d):
					self.column_w_sum[j] += column_w_sum[j]

					for k in range(d):
						self.pair_sum[j*d + k] += pair_sum_ndarray[j, k]
						self.pair_w_sum[j*d + k] += pair_w_sum[j*d + k] + w_sum
						self.column_sum[j*d + k] += column_sum[j*d + k]

		else:
			pair_sum = <double*> calloc(d*d, sizeof(double))

			dgemm('N', 'T', &d, &d, &n, &alpha, y, &d, y, &d, &beta, pair_sum, &d)

			with gil:
				for j in range(d):
					self.column_w_sum[j] += column_w_sum[j]

					for k in range(d):
						self.pair_sum[j*d + k] += pair_sum[j*d + k]
						self.pair_w_sum[j*d + k] += pair_w_sum[j*d + k] + w_sum
						self.column_sum[j*d + k] += column_sum[j*d + k]

			free(pair_sum)

		free(column_sum)
		free(column_w_sum)
		free(pair_w_sum)
		free(y)

	def from_summaries(self, inertia=0.0, min_covar=1e-5):
		"""
		Set the parameters of this Distribution to maximize the likelihood of
		the given sample. Items holds some sort of sequence. If weights is
		specified, it holds a sequence of value to weight each item by.
		"""

		cdef int d = self.d, i, j, k
		cdef double* column_sum = self.column_sum
		cdef double pair_sum
		cdef double* mu = self._mu_new
		cdef double cov
		cdef numpy.ndarray chol
		cdef double w_sum = 0.0

		for i in range(self.d):
			w_sum += self.column_w_sum[i]

		# If no summaries stored or the summary is frozen, don't do anything.
		if self.frozen == True or w_sum < 1e-7:
			return


		for i in range(d):
			mu[i] = self.column_sum[i*d + i] / self.column_w_sum[i]
			self._mu[i] = self._mu[i] * inertia + mu[i] * (1-inertia)

		for j in range(d):
			for k in range(d):
				x_jk = self.pair_sum[j*d + k]
				w_jk = self.pair_w_sum[j*d + k]

				if j == k:
					x_j = self.column_sum[j*d + j]
					x_k = self.column_sum[k*d + k]
				else:
					x_j = self.column_sum[j*d + j] + self.column_sum[j*d + k]
					x_k = self.column_sum[k*d + k] + self.column_sum[k*d + j]

				cov = (x_jk - x_j*x_k/w_jk) / w_jk if w_jk > 0.0 else 0
				self._cov[j*d + k] = self._cov[j*d + k] * inertia + cov * (1-inertia)

		try:
			chol = scipy.linalg.cholesky(self.cov, lower=True)
			self.inv_cov = scipy.linalg.solve_triangular(chol, numpy.eye(d),
				lower=True).T
		except:
			if self.cov.sum() == 0:
				self.cov += numpy.eye(d) * min_covar
			else:
				min_eig = numpy.linalg.eig(self.cov)[0].min()
				self.cov -= numpy.eye(d) * (min_eig - eps)

			chol = scipy.linalg.cholesky(self.cov, lower=True)
			self.inv_cov = scipy.linalg.solve_triangular(chol, numpy.eye(d),
				lower=True).T

		_, self._log_det = numpy.linalg.slogdet(self.cov)
		self._inv_cov = <double*> self.inv_cov.data

		mdot(self._mu, self._inv_cov, self._inv_dot_mu, 1, d, d)
		self.clear_summaries()

	def clear_summaries(self):
		"""Clear the summary statistics stored in the object."""

		memset(self.column_sum, 0, self.d*self.d*sizeof(double))
		memset(self.column_w_sum, 0, self.d*sizeof(double))
		memset(self.pair_sum, 0, self.d*self.d*sizeof(double))
		memset(self.pair_w_sum, 0, self.d*self.d*sizeof(double))

	@classmethod
	def from_samples(cls, X, weights=None, **kwargs):
		"""Fit a distribution to some data without pre-specifying it."""

		distribution = cls.blank(X.shape[1])
		distribution.fit(X, weights, **kwargs)
		return distribution

	@classmethod
	def blank(cls, d=2):
		mu = numpy.zeros(d)
		cov = numpy.eye(d)
		return cls(mu, cov)

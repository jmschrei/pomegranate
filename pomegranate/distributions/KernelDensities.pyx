#!python
#cython: boundscheck=False
#cython: cdivision=True
# KernelDensities.pyx
# Contact: Jacob Schreiber <jmschreiber91@gmail.com>

import numpy

from ..utils cimport _log
from ..utils cimport isnan
from ..utils import check_random_state

from libc.math cimport sqrt as csqrt
from libc.math cimport fabs
from libc.math cimport exp as cexp

# Define some useful constants
DEF NEGINF = float("-inf")
DEF INF = float("inf")
DEF SQRT_2_PI = 2.50662827463
DEF LOG_2_PI = 1.83787706641

cdef class KernelDensity(Distribution):
	"""An abstract kernel density, with shared properties and methods."""

	property parameters:
		def __get__(self):
			return [self.points_ndarray.tolist(), self.bandwidth, self.weights_ndarray.tolist()]
		def __set__(self, parameters):
			self.points_ndarray = numpy.array(parameters[0])
			self.points = <double*> self.points_ndarray.data

			self.bandwidth = parameters[1]

			self.weights_ndarray = numpy.array(parameters[2])
			self.weights = <double*> self.weights_ndarray.data

	def __cinit__(self, points=[], bandwidth=1, weights=None, frozen=False):
		"""
		Take in points, bandwidth, and appropriate weights. If no weights
		are provided, a uniform weight of 1/n is provided to each point.
		Weights are scaled so that they sum to 1.
		"""

		points = numpy.asarray(points, dtype=numpy.float64)
		n = points.shape[0]

		if weights is not None:
			weights = numpy.array(weights, dtype=numpy.float64) / numpy.sum(weights)
		else:
			weights = numpy.ones(n, dtype=numpy.float64) / n

		self.n = n
		self.points_ndarray = points
		self.points = <double*> self.points_ndarray.data

		self.weights_ndarray = weights
		self.weights = <double*> self.weights_ndarray.data

		self.bandwidth = bandwidth
		self.summaries = []
		self.name = "KernelDensity"
		self.frozen = frozen

	def __reduce__(self):
		"""Serialize the distribution for pickle."""
		return self.__class__, (self.points_ndarray, self.bandwidth, self.weights_ndarray, self.frozen)

	def fit(self, points, weights=None, inertia=0.0, column_idx=0):
		"""Replace the points, allowing for inertia if specified."""

		# If the distribution is frozen, don't bother with any calculation
		if self.frozen == True:
			return

		points = numpy.asarray(points, dtype=numpy.float64)
		n = points.shape[0]

		# Get the weights, or assign uniform weights
		if weights is not None:
			weights = numpy.array(weights, dtype=numpy.float64) / numpy.sum(weights)
		else:
			weights = numpy.ones(n, dtype=numpy.float64) / n

		# If no inertia, get rid of the previous points
		if inertia == 0.0:
			self.points_ndarray = points
			self.weights_ndarray = weights
			self.n = points.shape[0]

		# Otherwise adjust weights appropriately
		else:
			self.points_ndarray = numpy.concatenate((self.points_ndarray, points))
			self.weights_ndarray = numpy.concatenate((self.weights_ndarray*inertia, weights*(1-inertia)))
			self.n = points.shape[0]

		self.points = <double*> self.points_ndarray.data
		self.weights = <double*> self.weights_ndarray.data

	def summarize(self, items, weights=None, column_idx=0):
		"""Summarize a batch of data into sufficient statistics for a later update.

		Parameters
		----------
		items : array-like, shape (n_samples, n_dimensions)
			This is the data to train on. Each row is a sample, and each column
			is a dimension to train on. For univariate distributions an array
			is used, while for multivariate distributions a 2d matrix is used.

		weights : array-like, shape (n_samples,), optional
			The initial weights of each sample in the matrix. If nothing is
			passed in then each sample is assumed to be the same weight.
			Default is None.

		Returns
		-------
		None
		"""

		# If no previously stored summaries, just store the incoming data
		if len(self.summaries) == 0:
			self.summaries = [items, weights]

		# Otherwise, append the items and weights
		else:
			prior_items, prior_weights = self.summaries
			items = numpy.concatenate([prior_items, items])

			# If even one summary lacks weights, then weights can't be assigned
			# to any of the points.
			if weights is not None:
				weights = numpy.concatenate([prior_weights, weights])

			self.summaries = [items, weights]

	@classmethod
	def blank(cls):
		return cls([])


cdef class GaussianKernelDensity(KernelDensity):
	"""
	A quick way of storing points to represent a Gaussian kernel density in one
	dimension. Takes in the points at initialization, and calculates the log of
	the sum of the Gaussian distance of the new point from every other point.
	"""

	def __cinit__(self, points=[], bandwidth=1, weights=None, frozen=False):
		self.name = "GaussianKernelDensity"

	cdef void _log_probability(self, double* X, double* log_probability, int n) nogil:
		cdef double mu, w, scalar = 1.0 / SQRT_2_PI, prob, b = self.bandwidth
		cdef int i, j

		for i in range(n):
			prob = 0.0

			for j in range(self.n):
				mu = self.points[j]
				w = self.weights[j]
				prob += w * scalar * cexp(-0.5*((mu-X[i]) / b) ** 2)

			log_probability[i] = _log(prob)

	def sample(self, n=None, random_state=None):
		random_state = check_random_state(random_state)
		sigma = self.parameters[1]
		if n is None:
			mu = random_state.choice(self.parameters[0], p=self.parameters[2])
			return random_state.normal(mu, sigma)
		else:
			mus = random_state.choice(self.parameters[0], n, p=self.parameters[2])
			samples = [random_state.normal(mu, sigma) for mu in mus]
			return numpy.array(samples)


cdef class UniformKernelDensity(KernelDensity):
	"""
	A quick way of storing points to represent an Exponential kernel density in
	one dimension. Takes in points at initialization, and calculates the log of
	the sum of the Gaussian distances of the new point from every other point.
	"""

	def __cinit__(self, points=[], bandwidth=1, weights=None, frozen=False):
		self.name = "UniformKernelDensity"

	cdef void _log_probability(self, double* X, double* log_probability, int n) nogil:
		cdef double mu, w, scalar = 1.0 / SQRT_2_PI, prob, b = self.bandwidth
		cdef int i, j

		for i in range(n):
			if isnan(X[i]):
				log_probability[i] = 0.0
				continue

			prob = 0.0

			for j in range(self.n):
				mu = self.points[j]
				w = self.weights[j]

				if fabs(mu - X[i]) <= b:
					prob += w

			log_probability[i] = _log(prob)

	def sample(self, n=None, random_state=None):
		random_state = check_random_state(random_state)
		band = self.parameters[1]
		if n is None:
			mu = random_state.choice(self.parameters[0], p=self.parameters[2])
			return random_state.uniform(mu-band, mu+band)
		else:
			mus = random_state.choice(self.parameters[0], n, p=self.parameters[2])
			samples = [random_state.uniform(mu-band, mu+band) for mu in mus]
			return numpy.array(samples)


cdef class TriangleKernelDensity(KernelDensity):
	"""
	A quick way of storing points to represent an Exponential kernel density in
	one dimension. Takes in points at initialization, and calculates the log of
	the sum of the Gaussian distances of the new point from every other point.
	"""

	def __cinit__(self, points=[], bandwidth=1, weights=None, frozen=False):
		self.name = "TriangleKernelDensity"

	cdef void _log_probability(self, double* X, double* log_probability, int n) nogil:
		cdef double mu, w, scalar = 1.0 / SQRT_2_PI, prob
		cdef double hinge, b = self.bandwidth
		cdef int i, j

		for i in range(n):
			if isnan(X[i]):
				log_probability[i] = 0.0
				continue

			prob = 0.0

			for j in range(self.n):
				mu = self.points[j]
				w = self.weights[j]
				hinge = b - fabs(mu - X[i])
				if hinge > 0:
					prob += hinge * w

			log_probability[i] = _log(prob)

	def sample(self, n=None, random_state=None):
		random_state = check_random_state(random_state)
		band = self.parameters[1]
		if n is None:
			mu = random_state.choice(self.parameters[0], p=self.parameters[2])
			return random_state.triangular(mu-band, mu, mu+band)
		else:
			mus = random_state.choice(self.parameters[0], n, p=self.parameters[2])
			samples = [random_state.triangular(mu-band, mu, mu+band) for mu in mus]
			return numpy.array(samples)

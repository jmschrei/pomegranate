#!python
#cython: boundscheck=False
#cython: cdivision=True
# IndependentComponentsDistribution.pyx
# Contact: Jacob Schreiber <jmschreiber91@gmail.com>

import numpy
import json

from libc.stdlib cimport calloc
from libc.stdlib cimport free
from libc.string cimport memset

from ..utils cimport _log
from ..utils cimport isnan
from ..utils cimport python_log_probability
from ..utils cimport python_summarize

from ..utils import check_random_state
from ..utils import weight_set

from .DiscreteDistribution import DiscreteDistribution


cimport numpy
numpy.import_array()

cdef extern from "numpy/ndarraytypes.h":
	void PyArray_ENABLEFLAGS(numpy.ndarray X, int flags)

cdef class IndependentComponentsDistribution(MultivariateDistribution):
	"""
	Allows you to create a multivariate distribution, where each distribution
	is independent of the others. Distributions can be any type, such as
	having an exponential represent the duration of an event, and a normal
	represent the mean of that event. Observations must now be tuples of
	a length equal to the number of distributions passed in.

	s1 = IndependentComponentsDistribution([ExponentialDistribution(0.1),
									NormalDistribution(5, 2)])
	s1.log_probability((5, 2))
	"""

	property parameters:
		def __get__(self):
			return [self.distributions.tolist(), list(self.weights)]
		def __set__(self, parameters):
			self.distributions = numpy.asarray(parameters[0], dtype=numpy.object_)
			self.weights = parameters[1]

	def __cinit__(self, distributions=[], weights=None, frozen=False):
		"""
		Take in the distributions and appropriate weights. If no weights
		are provided, a uniform weight of 1/n is provided to each point.
		Weights are scaled so that they sum to 1.
		"""

		self.distributions = numpy.array(distributions)
		self.distributions_ptr = <void**> self.distributions.data

		self.d = len(distributions)
		self.discrete = isinstance(distributions[0], DiscreteDistribution)
		self.cython = 1
		for dist in distributions:
			if not isinstance(dist, Distribution) and not isinstance(dist, Model):
				self.cython = 0

		if weights is not None:
			self.weights = numpy.array(weights, dtype=numpy.float64)
		else:
			self.weights = numpy.ones(self.d, dtype=numpy.float64)

		self.weights_ptr = <double*> self.weights.data
		self.name = "IndependentComponentsDistribution"
		self.frozen = frozen

	def __getitem__(self, idx):
		"""Return the distribution at idx dimension."""

		return self.distributions[idx]

	def __reduce__(self):
		"""Serialize the distribution for pickle."""
		return self.__class__, (self.distributions, self.weights, self.frozen)

	def bake(self, keys):
		for i, distribution in enumerate(self.distributions):
			if isinstance(distribution, DiscreteDistribution):
				distribution.bake(keys[i])

	def log_probability(self, X):
		"""
		What's the probability of a given tuple under this mixture? It's the
		product of the probabilities of each X in the tuple under their
		respective distribution, which is the sum of the log probabilities.
		"""

		cdef int i, j, n
		cdef numpy.ndarray X_ndarray
		cdef double* X_ptr
		cdef double logp
		cdef numpy.ndarray logp_array
		cdef double* logp_ptr

		if self.discrete:
			if not isinstance(X[0], (list, tuple, numpy.ndarray)) or len(X) == 1:
				n = 1
			else:
				n = len(X)

			logp_array = numpy.zeros(n)
			for i in range(n):
				for j in range(self.d):
					logp_array[i] += self.distributions[j].log_probability(X[i][j]) * self.weights[j]

			if n == 1:
				return logp_array[0]
			else:
				return logp_array

		else:
			if isinstance(X[0], (int, float)) or len(X) == 1:
				n = 1
			else:
				n = len(X)

			X_ndarray = numpy.array(X, dtype='float64')
			X_ptr = <double*> X_ndarray.data

			logp_array = numpy.empty(n, dtype='float64')
			logp_ptr = <double*> logp_array.data

			with nogil:
				self._log_probability(X_ptr, logp_ptr, n)

			if n == 1:
				return logp_array[0]
			else:
				return logp_array

	cdef void _log_probability(self, double* X, double* log_probability, int n) nogil:
		cdef int i, j
		cdef double logp

		memset(log_probability, 0, n*sizeof(double))

		for i in range(n):
			for j in range(self.d):
				if self.cython == 1:
					(<Model> self.distributions_ptr[j])._log_probability(X+i*self.d+j, &logp, 1)
				else:
					with gil:
						python_log_probability(self.distributions[j], X+i*self.d+j, &logp, 1)

				log_probability[i] += logp * self.weights_ptr[j]

	def sample(self, n=None, random_state=None):
		if n is None:
			return numpy.array([d.sample(random_state=random_state) 
				for d in self.parameters[0]])
		else:
			return numpy.array([d.sample(n, random_state=random_state)
				for d in self.parameters[0]]).T.copy()

	def fit(self, X, weights=None, inertia=0, pseudocount=0.0):
		"""
		Set the parameters of this Distribution to maximize the likelihood of
		the given sample. Items holds some sort of sequence. If weights is
		specified, it holds a sequence of value to weight each item by.
		"""

		if self.frozen:
			return

		self.summarize(X, weights)
		self.from_summaries(inertia, pseudocount)

	def summarize(self, X, weights=None):
		"""
		Take in an array of items and reduce it down to summary statistics. For
		a multivariate distribution, this involves just passing the appropriate
		data down to the appropriate distributions.
		"""

		X, weights = weight_set(X, weights)
		cdef double* X_ptr = <double*> (<numpy.ndarray> X).data
		cdef double* weights_ptr = <double*> (<numpy.ndarray> weights).data
		cdef int n = X.shape[0]
		cdef int d = X.shape[1]

		with nogil:
			self._summarize(X_ptr, weights_ptr, n, 0, d)

	cdef double _summarize(self, double* X, double* weights, int n,
		int column_idx, int d) nogil:
		cdef int i, j
		cdef numpy.npy_intp dim = n * self.d
		cdef numpy.npy_intp n_elements = n

		if self.cython == 1:
			for i in range(d):
				(<Model> self.distributions_ptr[i])._summarize(X, weights, n, 
					i, d)
		else:
			with gil:
				X_ndarray = numpy.PyArray_SimpleNewFromData(1, &dim, 
					numpy.NPY_FLOAT64, X)
				X_ndarray = X_ndarray.reshape(n, self.d)

				w_ndarray = numpy.PyArray_SimpleNewFromData(1, &n_elements, 
					numpy.NPY_FLOAT64, weights)

				for i in range(d):
					self.distributions[i].summarize(X_ndarray[:,i], w_ndarray)

	def from_summaries(self, inertia=0.0, pseudocount=0.0):
		"""
		Use the collected summary statistics in order to update the
		distributions.
		"""

		# If the distribution is frozen, don't bother with any calculation
		if self.frozen == True:
			return

		for d in self.parameters[0]:
			if isinstance(d, DiscreteDistribution):
				d.from_summaries(inertia, pseudocount)
			else:
				d.from_summaries(inertia)

	def clear_summaries(self):
		"""Clear the summary statistics stored in the object."""

		for d in self.parameters[0]:
			d.clear_summaries()

	def to_json(self, separators=(',', ' : '), indent=4):
		"""Convert the distribution to JSON format."""

		return json.dumps({
								'class' : 'Distribution',
								'name'  : self.name,
								'parameters' : [[json.loads(dist.to_json()) for dist in self.parameters[0]],
								                 self.parameters[1]],
								'frozen' : self.frozen
						   }, separators=separators, indent=indent)

	@classmethod
	def from_samples(self, X, weights=None, distribution_weights=None,
		pseudocount=0.0, distributions=None):
		"""Create a new independent components distribution from data."""

		if distributions is None:
			raise ValueError("must pass in a list of distribution callables")

		X, weights = weight_set(X, weights)
		n, d = X.shape

		if callable(distributions):
			distributions = [distributions.from_samples(X[:,i], weights) for i in range(d)]
		else:
			distributions = [distributions[i].from_samples(X[:,i], weights) for i in range(d)]

		return IndependentComponentsDistribution(distributions, distribution_weights)


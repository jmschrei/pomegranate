#!python
#cython: boundscheck=False
#cython: cdivision=True
# CustomDistribution.pyx
# Contact: Aaron Meyer

import numpy

from ..utils cimport isnan

from libc.math cimport sqrt as csqrt


cdef class CustomDistribution(Distribution):
	""" The generic distribution class. """

	def __init__(self, N, frozen=False):
		self.frozen = frozen
		self.summaries = None
		self.weightsIn = numpy.ones(N, dtype='float64')
		self.logWeights = numpy.log(self.weights)
		self.weightsIn_ptr = <double*> self.weightsIn.data
		self.logWeights_ptr = <double*> self.logWeights.data

	cdef void _log_probability(self, double* X, double* log_probability, int n) nogil:
		cdef int i
		for i in range(n):
			if isnan(X[i]):
				log_probability[i] = 0.
			else:
				log_probability[i] = self.logWeights_ptr[int(X[i])]

	cdef double _summarize(self, double* X, double* weights, int n,
		int column_idx, int d) nogil:
		"""Calculate sufficient statistics for a minibatch.
		For this distribution we'll just store the weights.
		"""

		cdef int i

		for i in range(n):
			self.weightsIn_ptr[i] = weights[i]

	def clear_summaries(self):
		""" Clear the summary statistics stored in the object. Not needed here. """
		return

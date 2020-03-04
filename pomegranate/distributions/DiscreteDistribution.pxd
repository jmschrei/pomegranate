# DiscreteDistribution.pxd
# Contact: Jacob Schreiber <jmschreiber91@gmail.com>

import numpy
cimport numpy

from .distributions cimport Distribution

cdef class DiscreteDistribution(Distribution):
	cdef bint encoded_summary
	cdef int n
	cdef str dtype
	cdef dict dist, log_dist
	cdef tuple encoded_keys
	cdef double* encoded_counts
	cdef double* encoded_log_probability
	cdef double __probability(self, symbol)
	cdef double __log_probability(self, symbol)


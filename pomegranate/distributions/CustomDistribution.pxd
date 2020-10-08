# CustomDistribution.pxd
# Contact: Aaron Meyer

import numpy
cimport numpy

from .distributions cimport Distribution

cdef class CustomDistribution(Distribution):
	cdef public numpy.ndarray logWeights
	cdef public numpy.ndarray weightsIn
	cdef double* logWeights_ptr
	cdef double* weightsIn_ptr

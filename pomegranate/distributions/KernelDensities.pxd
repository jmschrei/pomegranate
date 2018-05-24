# KernelDensities.pxd
# Contact: Jacob Schreiber <jmschreiber91@gmail.com>

import numpy
cimport numpy

from .distributions cimport Distribution

cdef class KernelDensity(Distribution):
	cdef numpy.ndarray points_ndarray, weights_ndarray
	cdef double* points
	cdef double* weights
	cdef int n
	cdef double bandwidth

cdef class GaussianKernelDensity(KernelDensity):
	pass

cdef class UniformKernelDensity(KernelDensity):
	pass

cdef class TriangleKernelDensity(KernelDensity):
	pass

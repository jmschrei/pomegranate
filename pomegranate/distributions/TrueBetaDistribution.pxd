import numpy
cimport numpy

from .distributions cimport Distribution

cdef class TrueBetaDistribution(Distribution):
	cdef double alpha, beta
	cdef object min_alpha_beta, x_eps
	
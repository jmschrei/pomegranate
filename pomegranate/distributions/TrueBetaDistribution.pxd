import numpy
cimport numpy

from .distributions cimport Distribution

cdef class TrueBetaDistribution(Distribution):
	cdef double alpha, beta
	cdef double min_alpha_beta, x_eps
	
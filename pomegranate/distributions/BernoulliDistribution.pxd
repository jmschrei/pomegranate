# BernoulliDistribution.pxd
# Contact: Jacob Schreiber <jmschreiber91@gmail.com>

import numpy
cimport numpy

from .distributions cimport Distribution

cdef class BernoulliDistribution(Distribution):
	cdef double p
	cdef double* logp

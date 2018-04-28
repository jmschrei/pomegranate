# BetaDistribution.pxd
# Contact: Jacob Schreiber <jmschreiber91@gmail.com>

import numpy
cimport numpy

from .distributions cimport Distribution

cdef class BetaDistribution(Distribution):
	cdef double alpha, beta, beta_norm

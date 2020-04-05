# LogNormalDistribution.pxd
# Contact: Jacob Schreiber <jmschreiber91@gmail.com>

import numpy
cimport numpy

from .distributions cimport Distribution

cdef class LogNormalDistribution(Distribution):
	cdef double mu, sigma, min_std

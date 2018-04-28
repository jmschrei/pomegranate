# PoissonDistribution.pxd
# Contact: Jacob Schreiber <jmschreiber91@gmail.com>

import numpy
cimport numpy

from .distributions cimport Distribution

cdef class PoissonDistribution(Distribution):
	cdef double l, logl

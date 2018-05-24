# UniformDistribution.pxd
# Contact: Jacob Schreiber <jmschreiber91@gmail.com>

import numpy
cimport numpy

from .distributions cimport Distribution

cdef class UniformDistribution(Distribution):
	cdef double start, end, logp

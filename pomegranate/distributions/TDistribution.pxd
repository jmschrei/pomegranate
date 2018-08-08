# NormalDistribution.pxd
# Contact: Jacob Schreiber <jmschreiber91@gmail.com>

import numpy
cimport numpy

from .distributions cimport Distribution

cdef class TDistribution(Distribution):
    # cdef double mu, sigma, two_sigma_squared, log_sigma_sqrt_2_pi
    cdef double df
    # cdef object min_std

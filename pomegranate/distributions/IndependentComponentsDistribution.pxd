# IndependentComponentsDistribution.pxd
# Contact: Jacob Schreiber <jmschreiber91@gmail.com>

import numpy
cimport numpy

from ..base cimport Model

from .distributions cimport Distribution
from .distributions cimport MultivariateDistribution

cdef class IndependentComponentsDistribution(MultivariateDistribution):
	cdef public numpy.ndarray distributions, weights
	cdef public int discrete
	cdef public int cython
	cdef double* weights_ptr
	cdef void** distributions_ptr


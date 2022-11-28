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
	cdef void _log_probability_cython(self, double* symbol, double* log_probability, int n) nogil
	cdef void _log_probability_python(self, double* symbol, double* log_probability, int n) nogil

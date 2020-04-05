# DirichletDistribution.pxd
# Contact: Jacob Schreiber <jmschreiber91@gmail.com>

import numpy
cimport numpy

from .distributions cimport MultivariateDistribution

cdef class DirichletDistribution(MultivariateDistribution):
	cdef public numpy.ndarray alphas
	cdef double* alphas_ptr
	cdef double beta_norm
	cdef numpy.ndarray summaries_ndarray
	cdef double* summaries_ptr


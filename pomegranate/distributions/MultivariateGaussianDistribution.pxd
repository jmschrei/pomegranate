# MultivariateGaussianDistribution.pxd
# Contact: Jacob Schreiber <jmschreiber91@gmail.com>

import numpy
cimport numpy

from ..base cimport Model

from .distributions cimport Distribution
from .distributions cimport MultivariateDistribution

cdef class MultivariateGaussianDistribution(MultivariateDistribution):
	cdef public numpy.ndarray mu, cov, inv_cov
	cdef double* _mu
	cdef double* _mu_new
	cdef double* _cov
	cdef double _log_det
	cdef double* column_sum
	cdef double* column_w_sum
	cdef double* pair_sum
	cdef double* pair_w_sum
	cdef double* chol_dot_mu
	cdef double* _inv_cov
	cdef double* _inv_dot_mu
	cdef double _log_probability_missing(self, double* X) nogil


# gmm.pxd
# Contact: Jacob Schreiber ( jmschreiber91@gmail.com )

cimport numpy
from .base cimport Model

cdef class BayesModel(Model):
	cdef public numpy.ndarray distributions
	cdef object distribution_callable
	cdef void** distributions_ptr

	cdef public numpy.ndarray weights
	cdef double* weights_ptr

	cdef public numpy.ndarray summaries
	cdef double* summaries_ptr

	cdef object keymap
	cdef public int n
	cdef public bint is_vl_
	cdef public int cython

	cdef void _predict_log_proba(self, double* X, double* y, int n, int d) nogil
	cdef void _predict( self, double* X, int* y, int n, int d) nogil

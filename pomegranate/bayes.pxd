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

	cdef numpy.ndarray summaries
	cdef double* summaries_ptr
	
	cdef object keymap
	cdef public int n
	cdef public bint is_vl_

	cdef double _log_probability(self, double X) nogil
	cdef double _mv_log_probability(self, double* X) nogil
	cdef double _vl_log_probability(self, double* X, int n) nogil
	cdef void _predict_log_proba(self, double* X, double* y, int n, int d) nogil
	cdef void _predict( self, double* X, int* y, int n, int d) nogil
	cdef double _summarize(self, double* X, double* weights, int n) nogil

cdef class BayesClassifier(BayesModel):
	pass

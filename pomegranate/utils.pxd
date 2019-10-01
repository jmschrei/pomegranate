# utils.pxd
# Contact: Jacob Schreiber ( jmschreiber91@gmail.com )

cimport numpy

cdef int* GPU

cdef extern from "numpy/npy_math.h":
	bint npy_isnan(double x) nogil

cdef inline bint isnan(double x) nogil:
	return npy_isnan(x)

cdef int _is_gpu_enabled() nogil
cdef python_log_probability(model, double* X, double* log_probability, int n)
cdef python_summarize(model, double* X, double* weights, int n)
cdef ndarray_wrap_cpointer(void* data, int n)
cdef void mdot(double* X, double* Y, double* A, int m, int n, int k) nogil
cdef double _log (double x) nogil
cdef double pair_lse(double x, double y) nogil
cdef double gamma(double x) nogil
cdef double lgamma(double x) nogil

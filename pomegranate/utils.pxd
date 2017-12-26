# utils.pxd
# Contact: Jacob Schreiber ( jmschreiber91@gmail.com )

cimport numpy

cdef int* GPU

cdef extern from "numpy/npy_math.h":
	bint npy_isnan(double x) nogil

cdef bint isnan(double x) nogil
cdef int _is_gpu_enabled() nogil
cdef ndarray_wrap_cpointer(void* data, int n)
cdef void mdot(double* X, double* Y, double* A, int m, int n, int k) nogil
cdef double _log (double x) nogil
cdef double pair_lse(double x, double y) nogil
cdef double gamma(double x) nogil
cdef double lgamma(double x) nogil

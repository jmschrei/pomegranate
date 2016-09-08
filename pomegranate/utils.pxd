# utils.pxd
# Contact: Jacob Schreiber ( jmschreiber91@gmail.com )

cdef void mdot(double* X, double* Y, double* A, int m, int n, int k) nogil
cdef double _log (double x) nogil
cdef double pair_lse(double x, double y) nogil
cdef double gamma(double x) nogil
cdef double lgamma(double x) nogil
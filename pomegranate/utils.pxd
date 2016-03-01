# utils.pxd
# Contact: Jacob Schreiber ( jmschreiber91@gmail.com )

cdef double _log(double x) nogil
cdef double pair_lse(double x, double y) nogil
cdef double gamma(double x) nogil
cdef double lgamma(double x) nogil

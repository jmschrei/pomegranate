# utils.pxd
# Contact: Jacob Schreiber ( jmschreiber91@gmail.com )

cdef inline double _log ( double x ) nogil
cdef inline int pair_int_max( int x, int y ) nogil
cdef inline double pair_lse( double x, double y ) nogil
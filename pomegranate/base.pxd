# base.pxd
# Contact: Jacob Schreiber ( jmschreiber91@gmail.com )

cimport numpy

ctypedef numpy.npy_intp SIZE_t


cdef class Model(object):
	cdef public str name
	cdef public int d
	cdef public bint frozen
	cdef public str model

	cdef double _log_probability( self, double symbol ) nogil
	cdef double _mv_log_probability( self, double* symbol ) nogil
	cdef double _vl_log_probability( self, double* symbol, int n ) nogil
	cdef void _v_log_probability( self, double* symbol,
	                              double* log_probability, int n ) nogil
	cdef double _summarize( self, double* items, double* weights,
	                        SIZE_t n ) nogil


cdef class GraphModel(Model):
	cdef public list states, edges
	cdef public object graph
	cdef int n_edges, n_states


cdef class State(object):
	cdef public Model distribution
	cdef public str name
	cdef public double weight

# base.pxd
# Contact: Jacob Schreiber ( jmschreiber91@gmail.com )

cimport distributions
from distributions cimport Distribution

cdef class State( object ):
	cdef public Distribution distribution
	cdef public str name
	cdef public double weight

cdef class Model( object ):
	cdef public str name
	cdef public list states, edges
	cdef public object graph
	cdef int n_edges, n_states
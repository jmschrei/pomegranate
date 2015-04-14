# base.pxd
# Contact: Jacob Schreiber ( jmschreiber91@gmail.com )

cimport distributions
from distributions cimport Distribution

cdef class State( object ):
	cdef public Distribution distribution
	cdef public str name
	cdef public str identity
	cdef public double weight

cdef class Model( object ):
	cdef public str name
	cdef public list states, edges
	cdef object graph 
	cdef int [:] in_edge_count, in_transitions, out_edge_count, out_transitions
	cdef double [:] in_transition_log_probabilities
	cdef double [:] out_transition_log_probabilities
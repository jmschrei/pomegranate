# base.pxd
# Contact: Jacob Schreiber ( jmschreiber91@gmail.com )

cdef class Model( object ):
	cdef public str name
	cdef public object graph
	cdef public list states
	cdef int [:] in_edge_count, in_transitions, out_edge_count, out_transitions
	cdef double [:] in_transition_log_probabilities
	cdef double [:] out_transition_log_probabilities

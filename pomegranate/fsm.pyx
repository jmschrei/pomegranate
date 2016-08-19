# bayesnet.pyx
# Contact: Jacob Schreiber ( jmschreiber91@gmail.com )

from libc.stdlib cimport calloc
from libc.stdlib cimport free
from libc.string cimport memset

import json
import numpy
import sys

from .base cimport GraphModel
from .base cimport Model
from .base cimport State

cdef class FiniteStateMachine( GraphModel ):
	"""A Finite State Machine

	A finite state machine is a model which can be in one of many 'states',
	and transitions between these states upon seeing various observations. It
	is not a probabilistic model, but is highly useful and very related to
	HMMs.

	Parameters
	----------
	name : str, optional
		The name of the model. Default is None.

	start : state, optional
		The start of the model. Default is None.

	Attributes
	----------
	start : state
		The start object

	current_state : state
		The current state that the FSM is in.

	start_index : int
		The index of the state which is the starting state

	silent_start : int
		The start of the indexing of silent states

	current_index : int
		The index of current state the model is in

	Examples
	--------
	>>> from pomegranate import *
	>>> s1 = State( None, "s1" )
	>>> s2 = State( None, "s2" )
	>>> model = FiniteStateMachine( "example" )
	>>> model.add_states([s1, s2])
	>>> model.add_transition( model.start, s1, 'A' )
	>>> model.add_transition( model.start, s2, 'B' )
	>>> model.add_transition( s1, s2, 'B' )
	>>> model.add_transition( s1, s1, 'A' )
	>>> model.add_transition( s2, s1, 'A' )
	>>> model.add_transition( s2, s2, 'B' )
	>>> model.bake()
	>>> print model.current_state.name
	example-start
	>>> model.step('A')
	>>> print model.current_state.name, "after seeing 'A'"
	s1 after seeing 'A'
	>>> model.step('A')
	>>> print model.current_state.name, "after seeing 'A'"
	s1 after seeing 'A'
	>>> model.step('B')
	>>> print model.current_state.name, "after seeing 'B'"
	s2 after seeing 'B'
	>>> model.step('A')
	>>> print model.current_state.name, "after seeing 'A'"
	s1 after seeing 'A'
	"""

	cdef public object start
	cdef public State current_state
	cdef public int start_index, silent_start, current_index
	cdef object [:] edge_keys
	cdef dict indices

	cdef int* out_edge_count
	cdef int [:] out_transitions

	def __init__( self, name=None, start=None ):
		self.name = name or str( id(self) )

		# Create the starting state that we begin in
		self.start = start or State( None, name=self.name + "-start" )
		self.current_state = self.start
		self.current_index = 0

		# Create the list of all states in this model
		self.states = [ self.start ]
		self.edges = []

		self.out_edge_count = NULL

	def __dealloc__( self ):
		"""Destructor."""

		free( self.out_edge_count )

	def add_transition( self, a, b, key ):
		"""Add a transition for the model from a -> b.

		Add a transition from state a to state b. Since this is a FSM,
		instead of probabilities we have a key by which this edge is
		traversed.
		
		Parameters
		----------
		a : state
			The state the edge originates at

		b : state
			The state the edge ends up at
		
		key : object
			The key which causes the model to transition from a to b

		Returns
		-------
		None
		"""

		# Add the transition
		self.edges.append( (a, b, key) )

	def add_edge( self, a, b, key ):
		"""Wrapper for add_transition."""

		self.add_transition( a, b, key )

	def bake( self ): 
		"""Finalize the topology of the model.

		Assign a numerical index to every state and create the underlying arrays
		corresponding to the states and edges between the states. This method 
		must be called before any of the probability-calculating methods. This 
		is the same as the HMM bake, except that at the end it sets current
		state information.

		Parameters
		----------
		None

		Returns
		-------
		None
		"""
		
		# We need a mapping of states to their index. 
		indices = { self.states[i]: i for i in xrange(len(self.states)) }
		self.indices = indices

		n = len(self.states)
		# This holds numpy array indexed [a, b] to transition log probabilities 
		# from a to b, where a and b are state indices. It starts out saying all
		# transitions are impossible.
		self.out_edge_count = <int*> calloc( n+1, sizeof(int) )
		memset( self.out_edge_count, 0, (n+1)*sizeof(int) )

		# Now we need to find a way of storing in-edges for a state in a manner
		# that can be called in the cythonized methods below. This is basically
		# an inversion of the graph. We will do this by having two lists, one
		# list size number of nodes + 1, and one list size number of edges.
		# The node size list will store the beginning and end values in the
		# edge list that point to that node. The edge list will be ordered in
		# such a manner that all edges pointing to the same node are grouped
		# together. This will allow us to run the algorithms in time
		# nodes*edges instead of nodes*nodes.

		for a, b, key in self.edges:
			# Increment the total number of edges leaving node a.
			self.out_edge_count[ indices[a]+1 ] += 1

		# Take the cumulative sum so that we can associate array indices with
		# out transitions
		for i in xrange(1, n+1):
			self.out_edge_count[i] += self.out_edge_count[i-1]

		self.out_transitions = numpy.zeros( len(self.edges), dtype=numpy.int32 ) - 1
		self.edge_keys = numpy.empty( len(self.edges), dtype=object )

		# Now we go through the edges again in order to both fill in the
		# transition probability matrix, and also to store the indices sorted
		# by the end-node.
		for a, b, key in self.edges:
			start = self.out_edge_count[ indices[a] ]

			while self.out_transitions[ start ] != -1:
				if start == self.out_edge_count[ indices[a]+1 ]:
					break
				start += 1

			self.out_transitions[ start ] = indices[b]
			self.edge_keys[start] = key

		# This holds the index of the start state
		try:
			self.start_index = indices[self.start]

			# Set current state information
			self.current_state = self.start
			self.current_index = self.start_index 
		except KeyError:
			raise SyntaxError( "model.start has been deleted, leaving the \
				model with no start. Please ensure it has a start." )

	def to_json( self, separators=(',', ' : '), indent=4 ):
		"""Serialize the model to a JSON.

		Parameters
		----------
		separators : tuple, optional 
			The two separaters to pass to the json.dumps function for formatting.
			Default is (',', ' : ').

		indent : int, optional
			The indentation to use at each level. Passed to json.dumps for
			formatting. Default is 4.
		
		Returns
		-------
		json : str
			A properly formatted JSON object.
		"""
		
		model = { 
					'class' : 'FiniteStateMachine',
					'name'  : self.name,
					'start' : json.loads( self.start.to_json() ),
					'states' : [ json.loads( state.to_json() ) for state in self.states ],
					'start_index' : self.start_index,
					'silent_index' : self.silent_start
				}

		indices = { state: i for i, state in enumerate( self.states )}
		# Get all the edges from the graph
		edges = []
		for start, end, key in self.edges:
			s, e = indices[start], indices[end]
			edge = (s, e)
			edges.append( ( s, e, key ) )

		model['edges'] = edges

		return json.dumps( model, separators=separators, indent=indent )
			
	@classmethod
	def from_json( cls, s, verbose=False ):
		"""Read in a serialized model and return the appropriate classifier.
		
		Parameters
		----------
		s : str
			A JSON formatted string containing the file.

		Returns
		-------
		model : object
			A properly initialized and baked model.
		"""

		# Load a dictionary from a JSON formatted string
		d = json.loads( s )

		# Make a new generic HMM
		model = FiniteStateMachine( str(d['name']) )

		# Load all the states from JSON formatted strings
		states = [ State.from_json( json.dumps(j) ) for j in d['states'] ]

		# Add all the states to the model
		model.add_states( states )

		# Indicate appropriate start and end states
		model.start = states[ d['start_index'] ]

		# Add all the edges to the model
		for start, end, key in d['edges']:
			model.add_transition( states[start], states[end], key )

		# Bake the model
		model.bake()
		return model

	def step( self, symbol ):
		"""Update the internal state based on a symbol.

		Take in a symbol and move according to the edges in the model. This
		updates the current state according to which edge could be passed.
		
		Parameters
		----------
		symbol : object
			The next object in the sequence.

		Returns
		-------
		self : object
			The new object. 
		"""

		self._step( symbol )
		return self

	cdef void _step( self, object symbol ):
		cdef int i, k, ki
		cdef int* out_edges = self.out_edge_count 


		i = self.current_index
		for k in xrange( out_edges[i], out_edges[i+1] ):
			ki = self.out_transitions[k]
			if self.edge_keys[k] == symbol:
				self.current_index = ki
				self.current_state = self.states[ki]
				break
		else:
			raise SyntaxError( "No edges leaving state {} with key {}"
				.format( self.states[i].name, symbol ) )

	def reset( self ):
		"""Reset the internal state to the start.
		
		Parameters
		----------
		None

		Returns
		-------
		self : object
			The reset object
		"""

		self.current_index = 0
		self.current_state = self.start
		return self

# base.pyx
# Contact: Jacob Schreiber ( jmschreiber91@gmail.com )

from .distributions cimport Distribution
from .utils cimport *

import itertools as it
import json
import numpy
import networkx
import sys

if sys.version_info[0] > 2:
	# Set up for Python 3
	xrange = range

# Define some useful constants
DEF NEGINF = float("-inf")
DEF INF = float("inf")

def log(value):
	"""
	Return the natural log of the given value, or - infinity if the value is 0.
	Can handle both scalar floats and numpy arrays.
	"""

	if isinstance( value, numpy.ndarray ):
		to_return = numpy.zeros(( value.shape ))
		to_return[ value > 0 ] = numpy.log( value[ value > 0 ] )
		to_return[ value == 0 ] = NEGINF
		return to_return
	return _log( value )

cdef class State( object ):
	"""
	Represents a state in an HMM. Holds emission distribution, but not
	transition distribution, because that's stored in the graph edges.
	"""

	def __init__( self, distribution, name=None, weight=None ):
		"""
		Make a new State emitting from the given distribution. If distribution 
		is None, this state does not emit anything. A name, if specified, will 
		be the state's name when presented in output. Name may not contain 
		spaces or newlines, and must be unique within a model.
		"""
		
		# Save the distribution
		self.distribution = distribution
		
		# Save the name
		self.name = name or str(id(name))

		# Save the weight, or default to the unit weight
		self.weight = weight or 1.

	def __str__( self ):
		"""
		The string representation of a state is the json, so call that format.
		"""

		return self.to_json()

	def __repr__( self ):
		"""
		The string representation of a state is the json, so call that format.
		"""

		return self.__str__()

	def tie( self, state ):
		"""
		Tie this state to another state by just setting the distribution of the
		other state to point to this states distribution.
		"""

		state.distribution = self.distribution

	def is_silent( self ):
		"""
		Return True if this state is silent (distribution is None) and False 
		otherwise.
		"""
		
		return self.distribution is None

	def tied_copy( self ):
		"""
		Return a copy of this state where the distribution is tied to the
		distribution of this state.
		"""

		return State( distribution=self.distribution, name=self.name+'-tied' )
		
	def copy( self ):
		"""
		Return a hard copy of this state.
		"""

		return State( **self.__dict__ )
	
	def to_json( self, separators=(',', ' : '), indent=4 ):
		"""
		Convert this state to JSON format.
		"""

		return json.dumps( { 
							    'class' : 'State',
								'distribution' : None if self.is_silent() else json.loads( self.distribution.to_json() ),
								'name' : self.name,
								'weight' : self.weight  
							}, separators=separators, indent=indent )

	@classmethod
	def from_json( cls, s ):
		"""
		Read a State from a given string formatted in JSON.
		"""

		# Load a dictionary from a JSON formatted string
		d = json.loads( s )

		# If we're not decoding a state, we're decoding the wrong thing
		if d['class'] != 'State':
			raise IOError( "State object attempting to decode {} object".format( d['class'] ) )

		# If this is a silent state, don't decode the distribution
		if d['distribution'] is None:
			return cls( None, str(d['name']), d['weight'] )

		# Otherwise it has a distribution, so decode that
		return cls( Distribution.from_json( json.dumps( d['distribution'] ) ),
					name=str(d['name']), weight=d['weight'] )


Node = State

cdef class Model( object ):
	"""
	Represents an generic graphical model.
	"""

	def __init__( self, name=None ):
		"""
		Make a new graphical model. Name is an optional string used to name
		the model when output. Name may not contain spaces or newlines.
		"""
		
		# Save the name or make up a name.
		self.name = name or str( id(self) )
		self.states = []
		self.edges = []
		self.n_edges = 0
		self.n_states = 0
	
	def __str__(self):
		"""
		Represent this model with it's name and states.
		"""
		
		return "{}:{}".format(self.name, "".join(map(str, self.states)))

	def add_node( self, n ):
		"""
		Add a node to the graph.
		"""

		self.states.append( n )
		self.n_states += 1

	def add_nodes( self, n ):
		"""
		Add multiple states to the graph.
		"""

		for node in n:
			self.add_node( node )

	def add_state( self, s ):
		"""
		Another name for a node.
		"""

		self.add_node( s )

	def add_states( self, s ):
		"""
		Another name for a node.
		"""

		self.add_nodes( s )

	def add_edge( self, a, b ):
		"""
		Add a transition from state a to state b which indicates that B is
		dependent on A in ways specified by the distribution. 
		"""

		# Add the transition
		self.edges.append( ( a, b ) )
		self.n_edges += 1

	def add_transition( self, a, b ):
		"""
		Transitions and edges are the same.
		"""

		self.add_edge( a, b )

	def node_count( self):
		"""
		Returns the number of nodes/states in the model
		"""

		return self.n_states

	def state_count( self ):
		"""
		Returns the number of states present in the model.
		"""

		return self.n_states

	def edge_count( self ):
		"""
		Returns the number of edges present in the model.
		"""

		return self.n_edges

	def dense_transition_matrix( self ):
		"""
		Returns the dense transition matrix. Useful if the transitions of
		somewhat small models need to be analyzed.
		"""

		m = len(self.states)
		transition_log_probabilities = numpy.zeros( (m, m) ) + NEGINF

		for i in xrange(m):
			for n in xrange( self.out_edge_count[i], self.out_edge_count[i+1] ):
				transition_log_probabilities[i, self.out_transitions[n]] = \
					self.out_transition_log_probabilities[n]

		return transition_log_probabilities 

	def bake( self, verbose=False ): 
		"""
		Finalize the topology of the model, and assign a numerical index to
		every node. This method must be called before any of the probability-
		calculating or sampling methods.
		
		This fills in self.states (a list of all states in order), the sparse
		matrices of transitions and their weights, and also will merge silent
		states.
		"""

		n, m = len(self.states), len(self.edges)

		# We need a good way to get transition probabilities by state index that
		# isn't N^2 to build or store. So we will need a reverse of the above
		# mapping. It's awkward but asymptotically fine.
		indices = { self.states[i]: i for i in xrange(n) }
		self.edges = [ ( indices[a], indices[b] ) for a, b in self.edges ]

		# This holds numpy array indexed [a, b] to transition log probabilities 
		# from a to b, where a and b are state indices. It starts out saying all
		# transitions are impossible.
		self.in_transitions = numpy.zeros( m, dtype=numpy.int32 ) - 1
		self.in_edge_count = numpy.zeros( n+1, dtype=numpy.int32 ) 
		self.out_transitions = numpy.zeros( m, dtype=numpy.int32 ) - 1
		self.out_edge_count = numpy.zeros( n+1, dtype=numpy.int32 )

		# Now we need to find a way of storing in-edges for a state in a manner
		# that can be called in the cythonized methods below. This is basically
		# an inversion of the graph. We will do this by having two lists, one
		# list size number of nodes + 1, and one list size number of edges.
		# The node size list will store the beginning and end values in the
		# edge list that point to that node. The edge list will be ordered in
		# such a manner that all edges pointing to the same node are grouped
		# together. This will allow us to run the algorithms in time
		# nodes*edges instead of nodes*nodes.

		for a, b in self.edges:
			# Increment the total number of edges going to node b.
			self.in_edge_count[ indices[b]+1 ] += 1
			# Increment the total number of edges leaving node a.
			self.out_edge_count[ indices[a]+1 ] += 1

		# Take the cumulative sum so that we can associate array indices with
		# in or out transitions
		self.in_edge_count = numpy.cumsum(self.in_edge_count, 
			dtype=numpy.int32)
		self.out_edge_count = numpy.cumsum(self.out_edge_count, 
			dtype=numpy.int32 )

		# Now we go through the edges again in order to both fill in the
		# transition probability matrix, and also to store the indices sorted
		# by the end-node.
		for a, b in self.edges:
			# Put the edge in the dict. Its weight is log-probability
			start = self.in_edge_count[ indices[b] ]

			# Start at the beginning of the section marked off for node b.
			# If another node is already there, keep walking down the list
			# until you find a -1 meaning a node hasn't been put there yet.
			while self.in_transitions[ start ] != -1:
				if start == self.in_edge_count[ indices[b]+1 ]:
					break
				start += 1


			# Store transition info in an array where the in_edge_count shows
			# the mapping stuff.
			self.in_transitions[ start ] = indices[a]

			# Now do the same for out edges
			start = self.out_edge_count[ indices[a] ]

			while self.out_transitions[ start ] != -1:
				if start == self.out_edge_count[ indices[a]+1 ]:
					break
				start += 1

			self.out_transitions[ start ] = indices[b]

		self.edges = []
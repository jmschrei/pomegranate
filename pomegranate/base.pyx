# base.pyx
# Contact: Jacob Schreiber ( jmschreiber91@gmail.com )

cimport distributions
from distributions import Distribution

cimport utils
from utils cimport *

import numpy
cimport numpy

import networkx, sys
import itertools as it

if sys.version_info[0] > 2:
	# Set up for Python 3
	from functools import reduce
	xrange = range
	izip = zip
else:
	izip = it.izip

# Define some useful constants
DEF NEGINF = float("-inf")
DEF INF = float("inf")
DEF SQRT_2_PI = 2.50662827463

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

	def __init__( self, distribution, name=None, weight=None, identity=None ):
		"""
		Make a new State emitting from the given distribution. If distribution 
		is None, this state does not emit anything. A name, if specified, will 
		be the state's name when presented in output. Name may not contain 
		spaces or newlines, and must be unique within an HMM. Identity is a
		store of the id property, to allow for multiple states to have the same
		name but be uniquely identifiable. 
		"""
		
		# Save the distribution
		self.distribution = distribution
		
		# Save the name
		self.name = name or str(id(str))

		# Save the id
		if identity is not None:
			self.identity = str(identity)
		else:
			self.identity = str(id(self))

		self.weight = weight or 1.

	def __str__(self):
		"""
		Represent this state with it's name, weight, and identity.
		"""
		
		return "State( {}, name={}, weight={}, identity={} )".format(
			str(self.distribution), self.name, self.weight, self.identity )

	def is_silent(self):
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

		return State( distribution=self.distribution, name=self.name )
		
	def copy( self ):
		"""
		Return a hard copy of this state.
		"""

		return State( **self.__dict__ )
		
	def write(self, stream):
		"""
		Write this State (and its Distribution) to the given stream.
		
		Format: name, followed by "*" if the state is silent.
		If not followed by "*", the next line contains the emission
		distribution.
		"""
		
		name = self.name.replace( " ", "_" ) 
		stream.write( "{} {} {} {}\n".format( 
			self.identity, name, self.weight, str( self.distribution ) ) )
		
	@classmethod
	def read(cls, stream):
		"""
		Read a State from the given stream, in the format output by write().
		"""
		
		# Read a line
		line = stream.readline()
		
		if line == "":
			raise EOFError("End of file while reading state.")
			
		# Spilt the line up
		parts = line.strip().split()
		
		# parts[0] holds the state's name, and parts[1] holds the rest of the
		# state information, so we can just evaluate it.
		identity, name, weight, distribution = \
			parts[0], parts[1], parts[2], ' '.join( parts[3:] )
		return eval( "State( {}, name='{}', weight={}, identity='{}' )".format( 
			distribution, name, weight, identity ) )

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

		# This holds a directed graph between states. Nodes in that graph are
		# State objects, so they're guaranteed never to conflict when composing
		# two distinct models
		self.graph = networkx.DiGraph()
		
	
	def __str__(self):
		"""
		Represent this HMM with it's name and states.
		"""
		
		return "{}:\n\t{}".format(self.name, "\n\t".join(map(str, self.states)))

	def state_count( self ):
		"""
		Returns the number of states present in the model.
		"""

		return len( self.states )

	def edge_count( self ):
		"""
		Returns the number of edges present in the model.
		"""

		return len( self.out_transition_log_probabilities )

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

	def add_state(self, state):
		"""
		Adds the given State to the model. It must not already be in the model,
		nor may it be part of any other model that will eventually be combined
		with this one.
		"""
		
		# Put it in the graph
		self.graph.add_node(state)

	def add_states( self, states ):
		"""
		Adds multiple states to the model at the same time. Basically just a
		helper function for the add_state method.
		"""

		for state in states:
			self.add_state( state )
		
	def add_transition( self, a, b, probability ):
		"""
		Add a transition from state a to state b with the given (non-log)
		probability. Both states must be in the HMM already. self.start and
		self.end are valid arguments here. Probabilities will be normalized
		such that every node has edges summing to 1. leaving that node, but
		only when the model is baked. 

		By specifying a group as a string, you can tie edges together by giving
		them the same group. This means that a transition across one edge in the
		group counts as a transition across all edges in terms of training.
		"""

		# Add the transition
		self.graph.add_edge(a, b, weight=probability )

	def bake( self ): 
		"""
		Finalize the topology of the model, and assign a numerical index to
		every node. This method must be called before any of the probability-
		calculating or sampling methods.
		
		This fills in self.states (a list of all states in order) and 
		self.transition_log_probabilities (log probabilities for transitions).
		The option verbose will return a log of the changes made to the model
		due to normalization or merging. 
		"""

		# Go through the model and delete any nodes which have no edges leading
		# to it, or edges leading out of it. This gets rid of any states with
		# no edges in or out, as well as recursively removing any chains which
		# are impossible for the viterbi path to touch.
		self.in_edge_count = numpy.zeros( len( self.graph.nodes() ), 
			dtype=numpy.int32 ) 
		self.out_edge_count = numpy.zeros( len( self.graph.nodes() ), 
			dtype=numpy.int32 )


		self.states = self.graph.nodes()
		n, m = len(self.states), len(self.graph.edges())

		# We need a good way to get transition probabilities by state index that
		# isn't N^2 to build or store. So we will need a reverse of the above
		# mapping. It's awkward but asymptotically fine.
		indices = { self.states[i]: i for i in xrange(n) }

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

		for a, b in self.graph.edges_iter():
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
		for a, b, data in self.graph.edges_iter( data=True ):
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

cdef class StructuredModel( Model ):
	'''
	A basic structured model. 
	'''

	def add_transition( self, a, b, probability, pseudocount=None, group=None ):
		"""
		Add a transition from state a to state b with the given (non-log)
		probability. Both states must be in the HMM already. self.start and
		self.end are valid arguments here. Probabilities will be normalized
		such that every node has edges summing to 1. leaving that node, but
		only when the model is baked. 

		By specifying a group as a string, you can tie edges together by giving
		them the same group. This means that a transition across one edge in the
		group counts as a transition across all edges in terms of training.
		"""
		
		# If a pseudocount is specified, use it, otherwise use the probability.
		# The pseudocounts come up during training, when you want to specify
		# custom pseudocount weighting schemes per edge, in order to make the
		# model converge to that scheme given no observations. 
		pseudocount = pseudocount or probability

		# Add the transition
		self.graph.add_edge(a, b, weight=log(probability), 
			pseudocount=pseudocount, group=group )

	def add_transitions( self, a, b, probabilities=None, pseudocounts=None,
		groups=None ):
		"""
		Add many transitions at the same time, in one of two forms. 

		(1) If both a and b are lists, then create transitions from the i-th 
		element of a to the i-th element of b with a probability equal to the
		i-th element of probabilities.

		Example: 
		model.add_transitions([model.start, s1], [s1, model.end], [1., 1.])

		(2) If either a or b are a state, and the other is a list, create a
		transition from all states in the list to the single state object with
		probabilities and pseudocounts specified appropriately.

		Example:
		model.add_transitions([model.start, s1, s2, s3], s4, [0.2, 0.4, 0.3, 0.9])
		model.add_transitions(model.start, [s1, s2, s3], [0.6, 0.2, 0.05])

		If a single group is given, it's assumed all edges should belong to that
		group. Otherwise, either groups can be a list of group identities, or
		simply None if no group is meant.
		"""

		# If a pseudocount is specified, use it, otherwise use the probability.
		# The pseudocounts come up during training, when you want to specify
		# custom pseudocount weighting schemes per edge, in order to make the
		# model converge to that scheme given no observations. 
		pseudocounts = pseudocounts or probabilities

		n = len(a) if isinstance( a, list ) else len(b)
		if groups is None or isinstance( groups, str ):
			groups = [ groups ] * n

		# Allow addition of many transitions from many states
		if isinstance( a, list ) and isinstance( b, list ):
			# Set up an iterator across all edges
			edges = izip( a, b, probabilities, pseudocounts, groups )
			
			for start, end, probability, pseudocount, group in edges:
				self.add_transition( start, end, probability, pseudocount, group )

		# Allow for multiple transitions to a specific state 
		elif isinstance( a, list ) and isinstance( b, State ):
			# Set up an iterator across all edges to b
			edges = izip( a, probabilities, pseudocounts, groups )

			for start, probability, pseudocount, group in edges:
				self.add_transition( start, b, probability, pseudocount, group )

		# Allow for multiple transitions from a specific state
		elif isinstance( a, State ) and isinstance( b, list ):
			# Set up an iterator across all edges from a
			edges = izip( b, probabilities, pseudocounts, groups )

			for end, probability, pseudocount, group in edges:
				self.add_transition( a, end, probability, pseudocount, group )

	def freeze_distributions( self ):
		"""
		Freeze all the distributions in model. This means that upon training,
		only edges will be updated. The parameters of distributions will not
		be affected.
		"""

		for state in self.states:
			state.distribution.freeze()

	def thaw_distributions( self ):
		"""
		Thaw all distributions in the model. This means that upon training,
		distributions will be updated again.
		"""

		for state in self.states:
			state.distribution.thaw()
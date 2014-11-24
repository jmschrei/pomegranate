# base.pyx
# Contact: Jacob Schreiber ( jmschreiber91@gmail.com )

cimport distributions
from distributions import Distribution

import numpy
cimport numpy

import networkx

# Define some useful constants
DEF NEGINF = float("-inf")
DEF INF = float("inf")
DEF SQRT_2_PI = 2.50662827463

cdef class Model(object):
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


		self.nodes = self.graph.nodes()
		n, m = len(self.nodes), len(self.graph.edges())

		# We need a good way to get transition probabilities by state index that
		# isn't N^2 to build or store. So we will need a reverse of the above
		# mapping. It's awkward but asymptotically fine.
		indices = { self.nodes[i]: i for i in xrange(n) }

		# This holds numpy array indexed [a, b] to transition log probabilities 
		# from a to b, where a and b are state indices. It starts out saying all
		# transitions are impossible.
		self.in_transitions = numpy.zeros( m, dtype=numpy.int32 ) - 1
		self.in_edge_count = numpy.zeros( n+1, dtype=numpy.int32 ) 
		self.out_transitions = numpy.zeros( m, dtype=numpy.int32 ) - 1
		self.out_edge_count = numpy.zeros( n+1, dtype=numpy.int32 )
		self.in_transition_weights = numpy.zeros( m )
		self.out_transition_weights = numpy.zeros( m )

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

			self.in_transition_weights[ start ] = data['weight']

			# Store transition info in an array where the in_edge_count shows
			# the mapping stuff.
			self.in_transitions[ start ] = indices[a]

			# Now do the same for out edges
			start = self.out_edge_count[ indices[a] ]

			while self.out_transitions[ start ] != -1:
				if start == self.out_edge_count[ indices[a]+1 ]:
					break
				start += 1

			self.out_transition_weights[ start ] = data['weight']
			self.out_transitions[ start ] = indices[b]  
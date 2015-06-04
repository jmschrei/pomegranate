# bayesnet.pyx
# Contact: Jacob Schreiber ( jmschreiber91@gmail.com )

cimport cython
from cython.view cimport array as cvarray
from libc.math cimport log as clog, sqrt as csqrt, exp as cexp
import math, random, itertools as it, sys, bisect, json
import networkx

if sys.version_info[0] > 2:
	# Set up for Python 3
	from functools import reduce
	xrange = range
	izip = zip
else:
	izip = it.izip

import numpy
cimport numpy

cimport utils 
from utils cimport *

cimport distributions
from distributions cimport *

cimport base
from base cimport *

# Define some useful constants
DEF NEGINF = float("-inf")
DEF INF = float("inf")
DEF SQRT_2_PI = 2.50662827463

# Useful python-based array-intended operations
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
		
def exp(value):
	"""
	Return e^value, or 0 if the value is - infinity.
	"""
	
	return numpy.exp(value)

cdef class FiniteStateMachine( Model ):
	'''
	A finite state machine. 
	'''

	cdef public object start
	cdef public State current_state
	cdef public int start_index, silent_start, current_index
	cdef double [:] in_transition_pseudocounts
	cdef double [:] out_transition_pseudocounts
	cdef double [:] state_weights
	cdef int [:] tied_state_count
	cdef int [:] tied
	cdef int [:] tied_edge_group_size
	cdef int [:] tied_edges_starts
	cdef int [:] tied_edges_ends
	cdef dict indices

	def __init__( self, name=None, start=None ):
		"""
		Make a new Finite State Machine. Name is an optional string used to name
		the model when output. Name may not contain spaces or newlines.
		
		If start is specified, the machine will start in that state and not generate
		a new state for it.
		"""
		
		# Save the name or make up a name.
		self.name = name or str( id(self) )

		# This holds a directed graph between states. Nodes in that graph are
		# State objects, so they're guaranteed never to conflict when composing
		# two distinct models
		self.graph = networkx.DiGraph()
		
		# Save the start or make up a start
		self.start = start or State( None, name=self.name + "-start" )
		self.current_state = self.start
		
		# Put start and end in the graph
		self.graph.add_node(self.start)

	def add_transition( self, a, b ):
		"""
		Add a transition from state a to state b. Since this is a FSM,
		there aren't any probabilities.
		"""

		# Add the transition
		self.graph.add_edge( a, b, weight=0 )

	def add_transitions( self, a, b ):
		"""
		Add many transitions at the same time, in one of two forms. 

		(1) If both a and b are lists, then create transitions from the i-th 
		element of a to the i-th element of b

		Example: 
		model.add_transitions([model.start, s1], [s1, model.end] )

		(2) If either a or b are a state, and the other is a list, create a
		transition from all states in the list to the single state object

		Example:
		model.add_transitions([model.start, s1, s2, s3], s4)
		model.add_transitions(model.start, [s1, s2, s3] )
		"""

		n = len(a) if isinstance( a, list ) else len(b)

		# Allow addition of many transitions from many states
		if isinstance( a, list ) and isinstance( b, list ):
			# Set up an iterator across all edges
			edges = izip( a, b )
			
			for start, end, probability, pseudocount, group in edges:
				self.add_transition( start, end, 1 )

		# Allow for multiple transitions to a specific state 
		elif isinstance( a, list ) and isinstance( b, State ):
			for start in a:
				self.add_transition( start, b, 1 )

		# Allow for multiple transitions from a specific state
		elif isinstance( a, State ) and isinstance( b, list ):
			for end in b:
				self.add_transition( a, end, 1 )

	def bake( self, verbose=False, merge="all" ): 
		"""
		Finalize the topology of the model, and assign a numerical index to
		every state. This method must be called before any of the probability-
		calculating methods. This is the same as the HMM bake, except that at
		the end it sets current state information.
		
		This fills in self.states (a list of all states in order) and 
		self.transition_log_probabilities (log probabilities for transitions), 
		as well as self.start_index and self.end_index, and self.silent_start 
		(the index of the first silent state).

		The option verbose will return a log of the changes made to the model
		due to normalization or merging. 

		Merging has three options:
			"None": No modifications will be made to the model.
			"Partial": A silent state which only has a probability 1 transition
				to another silent state will be merged with that silent state.
				This means that if silent state "S1" has a single transition
				to silent state "S2", that all transitions to S1 will now go
				to S2, with the same probability as before, and S1 will be
				removed from the model.
			"All": A silent state with a probability 1 transition to any other
				state, silent or symbol emitting, will be merged in the manner
				described above. In addition, any orphan states will be removed
				from the model. An orphan state is a state which does not have
				any transitions to it OR does not have any transitions from it,
				except for the start and end of the model. This will iteratively
				remove orphan chains from the model. This is sometimes desirable,
				as all states should have both a transition in to get to that
				state, and a transition out, even if it is only to itself. If
				the state does not have either, the HMM will likely not work as
				intended.
		"""

		# Go through the model and delete any nodes which have no edges leading
		# to it, or edges leading out of it. This gets rid of any states with
		# no edges in or out, as well as recursively removing any chains which
		# are impossible for the viterbi path to touch.
		self.in_edge_count = numpy.zeros( len( self.graph.nodes() ), 
			dtype=numpy.int32 ) 
		self.out_edge_count = numpy.zeros( len( self.graph.nodes() ), 
			dtype=numpy.int32 )
		
		merge = merge.lower() if merge else None
		while merge == 'all':
			merge_count = 0

			# Reindex the states based on ones which are still there
			prestates = self.graph.nodes()
			indices = { prestates[i]: i for i in xrange( len( prestates ) ) }

			# Go through all the edges, summing in and out edges
			for a, b in self.graph.edges():
				self.out_edge_count[ indices[a] ] += 1
				self.in_edge_count[ indices[b] ] += 1
				
			# Go through each state, and if either in or out edges are 0,
			# remove the edge.
			for i in xrange( len( prestates ) ):
				if prestates[i] is self.start:
					continue

				if self.in_edge_count[i] == 0:
					merge_count += 1
					self.graph.remove_node( prestates[i] )

					if verbose:
						print "Orphan state {} removed due to no edges \
							leading to it".format(prestates[i].name )

				elif self.out_edge_count[i] == 0:
					merge_count += 1
					self.graph.remove_node( prestates[i] )

					if verbose:
						print "Orphan state {} removed due to no edges \
							leaving it".format(prestates[i].name )

			if merge_count == 0:
				break

		# Go through the model checking to make sure out edges sum to 1.
		# Normalize them to 1 if this is not the case.
		for state in self.graph.nodes():

			# Perform log sum exp on the edges to see if they properly sum to 1
			out_edges = round( sum( numpy.e**x['weight'] 
				for x in self.graph.edge[state].values() ), 8 )

			# The end state has no out edges, so will be 0
			if out_edges != 1.:
				# Issue a notice if verbose is activated
				if verbose:
					print "{} : {} summed to {}, normalized to 1.0"\
						.format( self.name, state.name, out_edges )

				# Reweight the edges so that the probability (not logp) sums
				# to 1.
				for edge in self.graph.edge[state].values():
					edge['weight'] = edge['weight'] - log( out_edges )

		# Automatically merge adjacent silent states attached by a single edge
		# of 1.0 probability, as that adds nothing to the model. Traverse the
		# edges looking for 1.0 probability edges between silent states.
		while merge in ['all', 'partial']:
			# Repeatedly go through the model until no merges take place.
			merge_count = 0

			for a, b, e in self.graph.edges( data=True ):
				# Since we may have removed a or b in a previous iteration,
				# a simple fix is to just check to see if it's still there
				if a not in self.graph.nodes() or b not in self.graph.nodes():
					continue

				if a == self.start:
					continue

				# If a silent state has a probability 1 transition out
				if e['weight'] == 0.0 and a.is_silent():

					# Make sure the transition is an appropriate merger
					if merge=='all' or ( merge=='partial' and b.is_silent() ):

						# Go through every transition to that state 
						for x, y, d in self.graph.edges( data=True ):

							# Make sure that the edge points to the current node
							if y is a:
								# Increment the edge counter
								merge_count += 1

								# Remove the edge going to that node
								self.graph.remove_edge( x, y )

								pseudo = max( e['pseudocount'], d['pseudocount'] )
								group = e['group'] if e['group'] == d['group'] else None
								# Add a new edge going to the new node
								self.graph.add_edge( x, b, weight=d['weight'],
									pseudocount=pseudo,
									group=group )

								# Log the event
								if verbose:
									print "{} : {} - {} merged".format(
										self.name, a, b)

						# Remove the state now that all edges are removed
						self.graph.remove_node( a )

			if merge_count == 0:
				break

		# Detect whether or not there are loops of silent states by going
		# through every pair of edges, and ensure that there is not a cycle
		# of silent states.		
		for a, b, e in self.graph.edges( data=True ):
			for x, y, d in self.graph.edges( data=True ):
				if a is y and b is x and a.is_silent() and b.is_silent():
					print "Loop: {} - {}".format( a.name, b.name )

		states = self.graph.nodes()
		n, m = len(states), len(self.graph.edges())
		silent_states, normal_states = [], []

		for state in states:
			if state.is_silent():
				silent_states.append(state)
			else:
				normal_states.append(state)

		# We need the silent states to be in topological sort order: any
		# transition between silent states must be from a lower-numbered state
		# to a higher-numbered state. Since we ban loops of silent states, we
		# can get away with this.
		
		# Get the subgraph of all silent states
		silent_subgraph = self.graph.subgraph(silent_states)
		
		# Get the sorted silent states. Isn't it convenient how NetworkX has
		# exactly the algorithm we need?
		silent_states_sorted = networkx.topological_sort(silent_subgraph)
		
		# What's the index of the first silent state?
		self.silent_start = len(normal_states)

		# Save the master state ordering. Silent states are last and in
		# topological order, so when calculationg forward algorithm
		# probabilities we can just go down the list of states.
		self.states = normal_states + silent_states_sorted 
		
		# We need a good way to get transition probabilities by state index that
		# isn't N^2 to build or store. So we will need a reverse of the above
		# mapping. It's awkward but asymptotically fine.
		indices = { self.states[i]: i for i in xrange(n) }
		self.indices = indices

		# This holds numpy array indexed [a, b] to transition log probabilities 
		# from a to b, where a and b are state indices. It starts out saying all
		# transitions are impossible.
		self.in_transitions = numpy.zeros( len(self.graph.edges()), 
			dtype=numpy.int32 ) - 1
		self.in_edge_count = numpy.zeros( len(self.states)+1, 
			dtype=numpy.int32 ) 
		self.out_transitions = numpy.zeros( len(self.graph.edges()), 
			dtype=numpy.int32 ) - 1
		self.out_edge_count = numpy.zeros( len(self.states)+1, 
			dtype=numpy.int32 )
		self.in_transition_log_probabilities = numpy.zeros(
			len( self.graph.edges() ) )
		self.out_transition_log_probabilities = numpy.zeros(
			len( self.graph.edges() ) )
		self.in_transition_pseudocounts = numpy.zeros( 
			len( self.graph.edges() ) )
		self.out_transition_pseudocounts = numpy.zeros(
			len( self.graph.edges() ) )

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
		for a, b, data in self.graph.edges_iter(data=True):
			# Put the edge in the dict. Its weight is log-probability
			start = self.in_edge_count[ indices[b] ]

			# Start at the beginning of the section marked off for node b.
			# If another node is already there, keep walking down the list
			# until you find a -1 meaning a node hasn't been put there yet.
			while self.in_transitions[ start ] != -1:
				if start == self.in_edge_count[ indices[b]+1 ]:
					break
				start += 1

			self.in_transition_log_probabilities[ start ] = data['weight']
			self.in_transition_pseudocounts[ start ] = data['pseudocount']

			# Store transition info in an array where the in_edge_count shows
			# the mapping stuff.
			self.in_transitions[ start ] = indices[a]

			# Now do the same for out edges
			start = self.out_edge_count[ indices[a] ]

			while self.out_transitions[ start ] != -1:
				if start == self.out_edge_count[ indices[a]+1 ]:
					break
				start += 1

			self.out_transition_log_probabilities[ start ] = data['weight']
			self.out_transition_pseudocounts[ start ] = data['pseudocount']
			self.out_transitions[ start ] = indices[b]  


		# This holds the index of the start state
		try:
			self.start_index = indices[self.start]

			# Set current state information
			self.current_state = self.start
			self.current_index = self.start_index 
		except KeyError:
			raise SyntaxError( "Model.start has been deleted, leaving the \
				model with no start. Please ensure it has a start." )

	def to_json( self ):
		"""
		Write out the HMM to JSON format, recursively including state and
		distribution information.
		"""
		
		model = { 
					'class' : 'FiniteStateMachine',
					'name'  : self.name,
					'start' : str(self.start),
					'states' : map( str, self.states ),
					'start_index' : self.start_index,
					'silent_index' : self.silent_start
				}

		indices = { state: i for i, state in enumerate( self.states )}
		# Get all the edges from the graph
		edges = []
		for start, end, data in self.graph.edges_iter( data=True ):
			s, e = indices[start], indices[end]
			prob, pseudocount = math.e**data['weight'], data['pseudocount']
			edge = (s, e)
			edges.append( ( s, e, prob, pseudocount ) )

		model['edges'] = edges
		return json.dumps( model)
#		return json.dumps( model, separators=(',', ' : '), indent=4 )
			
	@classmethod
	def from_json( cls, s, verbose=False ):
		"""
		Read a HMM from the given JSON, build the model, and bake it.
		"""

		# Load a dictionary from a JSON formatted string
		d = json.loads( s )

		# Make a new generic HMM
		model = FiniteStateMachine( str(d['name']) )

		# Load all the states from JSON formatted strings
		states = [ State.from_json( j ) for j in d['states'] ]

		# Add all the states to the model
		model.add_states( states )

		# Indicate appropriate start and end states
		model.start = states[ d['start_index'] ]

		# Add all the edges to the model
		for start, end, probability, pseudocount, group in d['edges']:
			model.add_transition( states[start], states[end], probability, 
				pseudocount, group )

		# Bake the model
		model.bake( verbose=verbose )
		return model

	def step( self, symbol ):
		'''
		Take in a sequence of symbols, and update the internal state.
		It will take the best step given the current state in a greedy manner.
		'''

		self._step( symbol )

	cdef void _step( self, object symbol ):
		'''
		Find the best next state to go to, and make the transition.
		'''

		cdef unsigned int D_SIZE = sizeof( double )
		cdef int m=len(self.states), k, ki, i, l
		cdef int index = self.indices[ self.current_state ]
		cdef double log_probability
		cdef int [:] in_edges = self.in_edge_count

		s = numpy.zeros( (2, m) ) + NEGINF
		s[ 0, self.current_index ] = 0
		e = numpy.zeros( self.silent_start ) + NEGINF
		for i in xrange( self.silent_start ):
			e[i] = self.states[i].distribution.log_probability(
				symbol ) + self.state_weights[i]

		for l in xrange( self.silent_start, m ):
			# Handle transitions between silent states before the first symbol
			# is emitted. No non-silent states have non-zero probability yet, so
			# we can ignore them.
			if l == self.start_index:
				# Start state log-probability is already right. Don't touch it.
				continue

			# This holds the log total transition probability in from 
			# all current-step silent states that can have transitions into 
			# this state.  
			log_probability = NEGINF
			for k in xrange( in_edges[l], in_edges[l+1] ):
				ki = self.in_transitions[k]
				if ki < self.silent_start or ki >= l:
					continue

				# For each current-step preceeding silent state k
				#log_probability = pair_lse( log_probability, 
				#	f[0, k] + self.transition_log_probabilities[k, l] )
				log_probability = pair_lse( log_probability,
					s[0, ki] + self.in_transition_log_probabilities[k] )

			# Update the table entry
			s[0, l] = log_probability

		for l in xrange( self.silent_start ):
			# Do the recurrence for non-silent states l
			# This holds the log total transition probability in from 
			# all previous states

			log_probability = NEGINF
			for k in xrange( in_edges[l], in_edges[l+1] ):
				ki = self.in_transitions[k]

				# For each previous state k
				log_probability = pair_lse( log_probability,
					s[0, ki] + self.in_transition_log_probabilities[k] )

			# Now set the table entry for log probability of emitting 
			# index+1 characters and ending in state l
			s[1, l] = log_probability + e[l]

		for l in xrange( self.silent_start, m ):
			# Now do the first pass over the silent states
			# This holds the log total transition probability in from 
			# all current-step non-silent states
			log_probability = NEGINF
			for k in xrange( in_edges[l], in_edges[l+1] ):
				ki = self.in_transitions[k]
				if ki >= self.silent_start:
					continue

				# For each current-step non-silent state k
				log_probability = pair_lse( log_probability,
					s[1, ki] + self.in_transition_log_probabilities[k] )

			# Set the table entry to the partial result.
			s[1, l] = log_probability

		for l in xrange( self.silent_start, m ):
			# Now the second pass through silent states, where we account
			# for transitions between silent states.

			# This holds the log total transition probability in from 
			# all current-step silent states that can have transitions into 
			# this state.
			log_probability = NEGINF
			for k in xrange( in_edges[l], in_edges[l+1] ):
				ki = self.in_transitions[k]
				if ki < self.silent_start or ki >= l:
					continue

				# For each current-step preceeding silent state k
				log_probability = pair_lse( log_probability,
					s[1, ki] + self.in_transition_log_probabilities[k] )

			# Add the previous partial result and update the table entry
			s[1, l] = pair_lse( s[1, l], log_probability )

		self.current_index = numpy.argmax( s[1] )
		self.current_state = self.states[ self.current_index ]

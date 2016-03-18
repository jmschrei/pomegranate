#cython: boundscheck=False
#cython: cdivision=True
# hmm.pyx: Yet Another Hidden Markov Model library
# Contact: Jacob Schreiber ( jmschreiber91@gmail.com )
#          Adam Novak ( anovak1@ucsc.edu )

from __future__ import print_function

from cython.view cimport array as cvarray
from libc.math cimport exp as cexp
import math, random, itertools as it, sys, json
import networkx
import tempfile
import warnings

from .base cimport Model
from .base cimport State
from .distributions cimport Distribution
from .distributions cimport MultivariateDistribution
from .distributions cimport DiscreteDistribution
from .gmm import GeneralMixtureModel
from .utils cimport _log
from .utils cimport pair_lse

from libc.stdlib cimport calloc
from libc.stdlib cimport free
from libc.string cimport memcpy
from libc.string cimport memset

import numpy
cimport numpy

from joblib import Parallel
from joblib import delayed

if sys.version_info[0] > 2:
	# Set up for Python 3
	xrange = range
	izip = zip
else:
	izip = it.izip

try:
	import pygraphviz
	from matplotlib import pyplot
	import matplotlib.image
except ImportError:
	pygraphviz = None

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

def log_probability( model, samples, n_jobs=1, parallel=None, check_input=True ):
	"""Return the log probability of samples given a model."""

	if parallel is not None:
		return sum( parallel( delayed( model.log_probability, check_pickle=False )
			( sample, check_input ) for sample in samples) )
	else:
		return sum( Parallel( n_jobs=n_jobs, backend='threading')( 
			delayed( model.log_probability, check_pickle=False )
				( sample, check_input ) for sample in samples) )

cdef class HiddenMarkovModel( Model ):
	"""
	Represents a Hidden Markov Model.
	"""

	cdef public object start, end
	cdef public int start_index
	cdef public int end_index
	cdef public int silent_start
	cdef double* in_transition_pseudocounts
	cdef double* out_transition_pseudocounts
	cdef double [:] state_weights
	cdef bint discrete
	cdef bint multivariate
	cdef int d
	cdef int* tied_state_count
	cdef int* tied
	cdef int* tied_edge_group_size
	cdef int* tied_edges_starts
	cdef int* tied_edges_ends
	cdef double* in_transition_log_probabilities
	cdef double* out_transition_log_probabilities
	cdef double* expected_transitions
	cdef int* in_edge_count
	cdef int* in_transitions
	cdef int* out_edge_count
	cdef int* out_transitions
	cdef int finite, n_tied_edge_groups
	cdef dict keymap
	cdef object state_names
	cdef numpy.ndarray distributions

	def __init__( self, name=None, start=None, end=None ):
		"""
		Make a new Hidden Markov Model. Name is an optional string used to name
		the model when output. Name may not contain spaces or newlines.
		
		If start and end are specified, they are used as start and end states 
		and new start and end states are not generated.
		"""

		# Save the name or make up a name.
		self.name = str(name) or str( id(self) )

		# This holds a directed graph between states. Nodes in that graph are
		# State objects, so they're guaranteed never to conflict when composing
		# two distinct models
		self.graph = networkx.DiGraph()
		
		# Save the start and end or mae one up
		self.start = start or State( None, name=self.name + "-start" )
		self.end = end or State( None, name=self.name + "-end" )

		self.n_edges = 0
		self.n_states = 0
		self.discrete = 0
		self.multivariate = 0
		
		# Put start and end in the graph
		self.graph.add_node(self.start)
		self.graph.add_node(self.end)

		self.in_edge_count = NULL
		self.in_transitions = NULL
		self.in_transition_pseudocounts = NULL
		self.in_transition_log_probabilities = NULL
		self.out_edge_count = NULL
		self.out_transitions = NULL
		self.out_transition_pseudocounts = NULL
		self.out_transition_log_probabilities = NULL
		self.expected_transitions = NULL

		self.tied_state_count = NULL
		self.tied = NULL
		self.tied_edge_group_size = NULL
		self.tied_edges_starts = NULL
		self.tied_edges_ends = NULL

		self.state_names = set()

	def add_state(self, state):
		"""
		Adds the given State to the model. It must not already be in the model,
		nor may it be part of any other model that will eventually be combined
		with this one.
		"""
		
		state_name = state.name
		if state_name in self.state_names:
			raise ValueError("A state with name '%s' already exists" % state_name)
		# Put it in the graph
		self.graph.add_node(state)
		self.state_names.add(state_name)

	def add_states( self, *states ):
		"""
		Adds multiple states to the model at the same time. Basically just a
		helper function for the add_state method.
		"""

		for state in states:
			if isinstance( state, list ):
				for s in state:
					self.add_state( s )
			else:
				self.add_state( state )

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
		self.graph.add_edge(a, b, probability=log(probability), 
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

	def copy( self ):
		"""Returns a deep copy of the HMM."""

		return HiddenMarkovModel.from_json( self.to_json() )

	def freeze_distributions( self ):
		"""
		Freeze all the distributions in model. This means that upon training,
		only edges will be updated. The parameters of distributions will not
		be affected.
		"""

		for state in self.states:
			if not state.is_silent():
				state.distribution.freeze()

	def thaw_distributions( self ):
		"""
		Thaw all distributions in the model. This means that upon training,
		distributions will be updated again.
		"""

		for state in self.states:
			if not state.is_silent():
				state.distribution.thaw()

	def is_infinite( self ):
		"""
		Returns whether or not the HMM is infinite, or finite. An infinite HMM
		is a HMM which does not have explicit transitions to an end state,
		meaning that it can end in any symbol emitting state. This is
		determined in the bake method, based on if there are any edges to the
		end state or not. Can only be used after a model is baked.
		"""

		return self.finite == 0

	def add_model( self, other ):
		"""
		Given another Model, add that model's contents to us. Its start and end
		states become silent states in our model.
		"""
		
		# Unify the graphs (requiring disjoint states)
		self.graph = networkx.union(self.graph, other.graph)
		
		# Since the nodes in the graph are references to Python objects,
		# other.start and other.end and self.start and self.end still mean the
		# same State objects in the new combined graph.

	def concatenate( self, other, suffix='', prefix='' ):
		"""
		Concatenate this model to another model in such a way that a single
		probability 1 edge is added between self.end and other.start. Rename
		all other states appropriately by adding a suffix or prefix if needed.
		"""

		other.name = "{}{}{}".format( prefix, other.name, suffix )
		for state in other.states:
			state.name = "{}{}{}".format( prefix, state.name, suffix )

		self.graph = networkx.union( self.graph, other.graph )
		self.add_transition( self.end, other.start, 1.00 )
		self.end = other.end


	def concatenate_model( self, other ):
		"""Deprecated. Use concatenate.
		"""

		raise Warning("Deprecated. Use concatenate.")


	def draw( self, **kwargs ):
		"""
		Draw this model's graph using NetworkX and matplotlib. Blocks until the
		window displaying the graph is closed.
		
		Note that this relies on networkx's built-in graphing capabilities (and 
		not Graphviz) and thus can't draw self-loops.

		See networkx.draw_networkx() for the keywords you can pass in.
		"""
		
		if pygraphviz is not None:
			G = pygraphviz.AGraph(directed=True)
			done_nodes = set()
			add_nodes = set()
			def add_node(node):
				if node not in add_nodes:
					if node.is_silent():
						color = 'grey'
					else:
						if node.distribution.frozen:
							color = 'blue'
						else:
							color = 'red'
					G.add_node(node.name, color=color)
					add_nodes.add(node)
			def proc_node(n):
				add_node(n)
				done_nodes.add(n)
				for node in self.graph.successors_iter(n):
					add_node(node)
					G.add_edge(n.name, node.name, label="%g" % (math.e ** self.graph.get_edge_data(n, node)['probability']))
					if node not in done_nodes:
						proc_node(node)
			for node in self.graph.nodes_iter():
				if isinstance(node, State):
					proc_node(node)
			with tempfile.NamedTemporaryFile() as tf:
				G.draw(tf.name, format='png', prog='dot')
				img = matplotlib.image.imread(tf.name)
				pyplot.imshow(img)
				pyplot.axis('off')
		else:
			warnings.warn("Install pygraphviz for nicer visualizations")
			networkx.draw()
		pyplot.show()

	def bake( self, verbose=False, merge="all" ): 
		"""
		Finalize the topology of the model, and assign a numerical index to
		every state. This method must be called before any of the probability-
		calculating methods.
		
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
 
		in_edge_count = numpy.zeros( len( self.graph.nodes() ), 
			dtype=numpy.int32 ) 
		out_edge_count = numpy.zeros( len( self.graph.nodes() ), 
			dtype=numpy.int32 )
		
		merge = merge.lower() if merge else None
		while merge == 'all':
			merge_count = 0

			# Reindex the states based on ones which are still there
			prestates = self.graph.nodes()
			indices = { prestates[i]: i for i in range( len( prestates ) ) }

			# Go through all the edges, summing in and out edges
			for a, b in self.graph.edges():
				out_edge_count[ indices[a] ] += 1
				in_edge_count[ indices[b] ] += 1
				
			# Go through each state, and if either in or out edges are 0,
			# remove the edge.
			for i in range( len( prestates ) ):
				if prestates[i] is self.start or prestates[i] is self.end:
					continue

				if in_edge_count[i] == 0:
					merge_count += 1
					self.graph.remove_node( prestates[i] )

					if verbose:
						print( "Orphan state {} removed due to no edges \
							leading to it".format(prestates[i].name ) )

				elif out_edge_count[i] == 0:
					merge_count += 1
					self.graph.remove_node( prestates[i] )

					if verbose:
						print( "Orphan state {} removed due to no edges \
							leaving it".format(prestates[i].name ) )

			if merge_count == 0:
				break

		# Go through the model checking to make sure out edges sum to 1.
		# Normalize them to 1 if this is not the case.
		for state in self.graph.nodes():

			# Perform log sum exp on the edges to see if they properly sum to 1
			out_edges = round( sum( numpy.e**x['probability'] 
				for x in self.graph.edge[state].values() ), 8 )

			# The end state has no out edges, so will be 0
			if out_edges != 1. and state != self.end:
				# Issue a notice if verbose is activated
				if verbose:
					print( "{} : {} summed to {}, normalized to 1.0"\
						.format( self.name, state.name, out_edges ) )

				# Reweight the edges so that the probability (not logp) sums
				# to 1.
				for edge in self.graph.edge[state].values():
					edge['probability'] = edge['probability'] - log( out_edges )

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

				if a == self.start or b == self.end:
					continue

				# If a silent state has a probability 1 transition out
				if e['probability'] == 0.0 and a.is_silent():

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
								self.graph.add_edge( x, b, probability=d['probability'],
									pseudocount=pseudo,
									group=group )

								# Log the event
								if verbose:
									print( "{} : {} - {} merged".format(
										self.name, a, b) )

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
					print( "Loop: {} - {}".format( a.name, b.name ) )

		states = self.graph.nodes()
		n, m = len(states), len(self.graph.edges())

		self.n_edges = m
		self.n_states = n

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
		indices = { self.states[i]: i for i in range(n) }

		# Create a sparse representation of the tied states in the model. This
		# is done in the same way of the transition, by having a vector of
		# counts, and a vector of the IDs that the state is tied to.
		self.tied_state_count = <int*> calloc( self.silent_start+1, sizeof(int) )
		for i in range( self.silent_start+1 ):
			self.tied_state_count[i] = 0

		for i in range( self.silent_start ): 
			for j in range( self.silent_start ):
				if i == j:
					continue
				if self.states[i].distribution is self.states[j].distribution:
					self.tied_state_count[i+1] += 1

		for i in range( 1, self.silent_start+1 ):
			self.tied_state_count[i] += self.tied_state_count[i-1]

		self.tied = <int*> calloc( self.tied_state_count[self.silent_start], sizeof(int) )
		for i in range( self.tied_state_count[self.silent_start] ):
			self.tied[i] = -1

		for i in range( self.silent_start ):
			for j in range( self.silent_start ):
				if i == j:
					continue
					
				if self.states[i].distribution is self.states[j].distribution:
					# Begin at the first index which belongs to state i...
					start = self.tied_state_count[i]

					# Find the first non -1 entry in order to put our index.
					while self.tied[start] != -1:
						start += 1

					# Now that we've found a non -1 entry, put the index of the
					# state which this state is tied to in!
					self.tied[start] = j

		# Unpack the state weights
		self.state_weights = numpy.zeros( self.silent_start )
		for i in range( self.silent_start ):
			self.state_weights[i] = _log( self.states[i].weight )

		# This holds numpy array indexed [a, b] to transition log probabilities 
		# from a to b, where a and b are state indices. It starts out saying all
		# transitions are impossible.
		self.in_transitions = <int*> calloc( m, sizeof(int) )
		self.in_edge_count = <int*> calloc( n+1, sizeof(int) )
		self.in_transition_pseudocounts = <double*> calloc( m,
			sizeof(double) )
		self.in_transition_log_probabilities = <double*> calloc( m, 
			sizeof(double) )

		self.out_transitions = <int*> calloc( m, sizeof(int) )
		self.out_edge_count = <int*> calloc( n+1, sizeof(int) )
		self.out_transition_pseudocounts = <double*> calloc( m,
			sizeof(double) )
		self.out_transition_log_probabilities = <double*> calloc( m, 
			sizeof(double) )

		self.expected_transitions =  <double*> calloc( self.n_edges, sizeof(double) )

		memset( self.in_transitions, -1, m*sizeof(int) )
		memset( self.in_edge_count, 0, (n+1)*sizeof(int) )
		memset( self.in_transition_pseudocounts, 0, m*sizeof(double) )
		memset( self.in_transition_log_probabilities, 0, m*sizeof(double) )

		memset( self.out_transitions, -1, m*sizeof(int) )
		memset( self.out_edge_count, 0, (n+1)*sizeof(int) )
		memset( self.out_transition_pseudocounts, 0, m*sizeof(double) )
		memset( self.out_transition_log_probabilities, 0, m*sizeof(double) )
		
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

		# Determine if the model is infinite or not based on the number of edges
		# to the end state
		if self.in_edge_count[ indices[ self.end ]+1 ] == 0:
			self.finite = 0
		else:
			self.finite = 1
		# Take the cumulative sum so that we can associate array indices with
		# in or out transitions
		for i in xrange( 1, n+1 ):
			self.in_edge_count[i] += self.in_edge_count[i-1]
			self.out_edge_count[i] += self.out_edge_count[i-1]

		# We need to store the edge groups as name : set pairs.
		edge_groups = {}

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

			self.in_transition_log_probabilities[start] = <double>data['probability']
			self.in_transition_pseudocounts[start] = data['pseudocount']

			# Store transition info in an array where the in_edge_count shows
			# the mapping stuff.
			self.in_transitions[start] = <int>indices[a]

			# Now do the same for out edges
			start = self.out_edge_count[indices[a]]

			while self.out_transitions[ start ] != -1:
				if start == self.out_edge_count[ indices[a]+1 ]:
					break
				start += 1

			self.out_transition_log_probabilities[start] = <double>data['probability']
			self.out_transition_pseudocounts[start] = data['pseudocount']
			self.out_transitions[start] = <int>indices[b]  

			# If this edge belongs to a group, we need to add it to the
			# dictionary. We only care about forward representations of
			# the edges. 
			group = data['group']
			if group != None:
				if group in edge_groups:
					edge_groups[group].append( ( indices[a], indices[b] ) )
				else:
					edge_groups[ group ] = [ ( indices[a], indices[b] ) ]

		# We will organize the tied edges using three arrays. The first will be
		# the cumulative number of members in each group, to slice the later
		# arrays in the same manner as the transition arrays. The second will
		# be the index of the state the edge starts in. The third will be the
		# index of the state the edge ends in. This way, iterating across the
		# second and third lists in the slices indicated by the first list will
		# give all the edges in a group.
		total_grouped_edges = sum( map( len, edge_groups.values() ) )

		self.n_tied_edge_groups = len(edge_groups.keys())+1
		self.tied_edge_group_size = <int*> calloc(len(edge_groups.keys())+1,
			sizeof(int) )
		self.tied_edge_group_size[0] = 0

		self.tied_edges_starts = <int*> calloc( total_grouped_edges, sizeof(int))
		self.tied_edges_ends = <int*> calloc( total_grouped_edges, sizeof(int))

		# Iterate across all the grouped edges and bin them appropriately.
		for i, (name, edges) in enumerate( edge_groups.items() ):
			# Store the cumulative number of edges so far, which requires
			# adding the current number of edges (m) to the previous
			# number of edges (n)
			n = self.tied_edge_group_size[i]
			self.tied_edge_group_size[i+1] = n + len(edges)

			for j, (start, end) in enumerate( edges ):
				self.tied_edges_starts[n+j] = start
				self.tied_edges_ends[n+j] = end

		dist = self.states[0].distribution 
		if isinstance( dist, DiscreteDistribution ):
			self.discrete = 1

			keys = []
			for state in self.states[:self.silent_start]:
				keys.extend( state.distribution.keys() )
			keys = tuple( set( keys ) )

			self.keymap = { keys[i] : i for i in xrange(len(keys)) }

			for state in self.states:
				for state in self.states[:self.silent_start]:
					state.distribution.encode( keys )

		if isinstance( dist, MultivariateDistribution ):
			self.multivariate = 1
			self.d = dist.d
		if isinstance( dist, GeneralMixtureModel ) and dist.d > 1:
			self.multivariate = 1
			self.d = dist.d

		self.distributions = numpy.empty(self.silent_start, dtype='object')
		for i in range(self.silent_start):
			self.distributions[i] = self.states[i].distribution

		# This holds the index of the start state
		try:
			self.start_index = indices[self.start]
		except KeyError:
			raise SyntaxError( "Model.start has been deleted, leaving the \
				model with no start. Please ensure it has a start." )
		# And the end state
		try:
			self.end_index = indices[self.end]
		except KeyError:
			raise SyntaxError( "Model.end has been deleted, leaving the \
				model with no end. Please ensure it has an end." )

	def sample( self, length=0, path=False ):
		"""
		Generate a sequence from the model. Returns the sequence generated, as a
		list of emitted items. The model must have been baked first in order to 
		run this method.

		If a length is specified and the HMM is infinite (no edges to the
		end state), then that number of samples will be randomly generated.
		If the length is specified and the HMM is finite, the method will
		attempt to generate a prefix of that length. Currently it will force
		itself to not take an end transition unless that is the only path,
		making it not a true random sample on a finite model.

		WARNING: If the HMM is infinite, must specify a length to use.

		If path is True, will return a tuple of ( sample, path ), where path is
		the path of hidden states that the sample took. Otherwise, the method
		will just return the path. Note that the path length may not be the same
		length as the samples, as it will return silent states it visited, but
		they will not generate an emission.
		"""
		
		return self._sample( length, path )

	cdef list _sample( self, int length, int path ):
		"""
		Perform a run of sampling.
		"""

		cdef int i, j, k, l, li, m=len(self.states)
		cdef double cumulative_probability
		cdef double [:,:] transition_probabilities = numpy.zeros( (m,m) )
		cdef double [:] cum_probabilities = numpy.zeros( self.n_edges )

		cdef int*  out_edges = self.out_edge_count

		for k in xrange( m ):
			cumulative_probability = 0.
			for l in xrange( out_edges[k], out_edges[k+1] ):
				cumulative_probability += cexp( 
					self.out_transition_log_probabilities[l] )
				cum_probabilities[l] = cumulative_probability 

		# This holds the numerical index of the state we are currently in.
		# Start in the start state
		i = self.start_index
		
		# Record the number of samples
		cdef int n = 0
		# Define the list of emissions, and the path of hidden states taken
		cdef list emissions = [], sequence_path = []
		cdef State state
		cdef double sample

		while i != self.end_index:
			# Get the object associated with this state
			state = self.states[i]

			# Add the state to the growing path
			sequence_path.append( state )
			
			if not state.is_silent():
				# There's an emission distribution, so sample from it
				emissions.append( state.distribution.sample() )
				n += 1

			# If we've reached the specified length, return the appropriate
			# values
			if length != 0 and n >= length:
				if path:
					return [emissions, sequence_path]
				return emissions

			# What should we pick as our next state?
			# Generate a number between 0 and 1 to make a weighted decision
			# as to which state to jump to next.
			sample = random.random()
			
			# Save the last state id we were in
			j = i

			# Find out which state we're supposed to go to by comparing the
			# random number to the list of cumulative probabilities for that
			# state, and then picking the selected state.
			for k in xrange( out_edges[i], out_edges[i+1] ):
				if cum_probabilities[k] > sample:
					i = self.out_transitions[k]
					break

			# If the user specified a length, and we're not at that length, and
			# we're in an infinite HMM, we want to avoid going to the end state
			# if possible. If there is only a single probability 1 end to the
			# end state we can't avoid it, otherwise go somewhere else.
			if length != 0 and self.finite == 1 and i == self.end_index:
				# If there is only one transition...
				if len( xrange( out_edges[j], out_edges[j+1] ) ) == 1:
					# ...and that transition goes to the end of the model...
					if self.out_transitions[ out_edges[j] ] == self.end_index:
						# ... then end the sampling, as nowhere else to go.
						break

				# Take the cumulative probability of not going to the end state
				cumulative_probability = 0.
				for k in xrange( out_edges[k], out_edges[k+1] ):
					if self.out_transitions[k] != self.end_index:
						cumulative_probability += cum_probabilities[k]

				# Randomly select a number in that probability range
				sample = random.uniform( 0, cumulative_probability )

				# Select the state is corresponds to
				for k in xrange( out_edges[i], out_edges[i+1] ):
					if cum_probabilities[k] > sample:
						i = self.out_transitions[k]
						break
		
		# Done! Return either emissions, or emissions and path.
		if path:
			sequence_path.append( self.end )
			return [emissions, sequence_path]
		return emissions

	cpdef double log_probability( self, sequence, check_input=True ):
		'''
		Calculate the log probability of a single sequence. If a path is
		provided, calculate the log probability of that sequence given
		the path.
		'''

		cdef numpy.ndarray sequence_ndarray
		cdef double* sequence_data
		cdef double* f
		cdef double log_probability_sum
		cdef int n = len(sequence), m = len(self.states)
		cdef void** distributions = <void**> self.distributions.data
		cdef int mv = self.multivariate
		cdef int is_str_type

		if check_input:
			if mv and not isinstance( sequence[0][0], str ):
				sequence_ndarray = numpy.array( sequence, dtype=numpy.float64 )
			elif not mv and not isinstance( sequence[0], str ):
				sequence_ndarray = numpy.array( sequence, dtype=numpy.float64 )
			else:
				sequence = list( map( self.keymap.__getitem__, sequence ) )
				sequence_ndarray = numpy.array( sequence, dtype=numpy.float64 )
		else:
			sequence_ndarray = sequence

		sequence_data = <double*> sequence_ndarray.data

		with nogil:
			f = self._forward( sequence_data, distributions, n, NULL )

		if self.finite == 1:
			log_probability_sum = f[n*m + self.end_index]
		else:
			log_probability_sum = NEGINF
			for i in range( self.silent_start ):
				log_probability_sum = pair_lse( log_probability_sum, f[n*m + i] )

		return log_probability_sum

	cpdef numpy.ndarray forward( self, sequence ):
		'''
		Python wrapper for the forward algorithm, calculating probability by
		going forward through a sequence. Returns the full forward DP matrix.
		Each index i, j corresponds to the sum-of-all-paths log probability
		of starting at the beginning of the sequence, and aligning observations
		to hidden states in such a manner that observation i was aligned to
		hidden state j. Uses row normalization to dynamically scale each row
		to prevent underflow errors.

		If the sequence is impossible, will return a matrix of nans.

		input
			sequence: a list (or numpy array) of observations
			use_cache: Use the already calculated emissions matrix.

		output
			A n-by-m matrix of floats, where n = len( sequence ) and
			m = len( self.states ). This is the DP matrix for the
			forward algorithm.

		See also: 
			- Silent state handling taken from p. 71 of "Biological
		Sequence Analysis" by Durbin et al., and works for anything which
		does not have loops of silent states.
			- Row normalization technique explained by 
		http://www.cs.sjsu.edu/~stamp/RUA/HMM.pdf on p. 14.
		'''

		cdef numpy.ndarray sequence_ndarray
		cdef double* sequence_data
		cdef double* f
		cdef int n = len(sequence), m = len(self.states)
		cdef void** distributions = <void**> self.distributions.data
		cdef numpy.ndarray f_ndarray = numpy.zeros( (n+1, m), dtype=numpy.float64 )

		try:
			sequence_ndarray = numpy.array( sequence, dtype=numpy.float64 )
		except ValueError:
			sequence = list( map( self.keymap.__getitem__, sequence ) )
			sequence_ndarray = numpy.array( sequence, dtype=numpy.float64)

		sequence_data = <double*> sequence_ndarray.data

		with nogil:
			f = self._forward( sequence_data, distributions, n, NULL )

		for i in range(n+1):
			for j in range(m):
				f_ndarray[i, j] = f[i*m + j]

		free(f)
		return f_ndarray

	cdef double* _forward( self, double* sequence, void** distributions, 
		int n, double* emissions ) nogil:
		"""Run the forward algorithm using optimized cython code."""

		cdef int i, k, ki, l, li
		cdef int p = self.silent_start, m = self.n_states
		cdef int dim = self.d

		cdef double log_probability
		cdef int* in_edges = self.in_edge_count

		cdef double* e = NULL
		cdef double* f = <double*> calloc( m*(n+1), sizeof(double) )

		# Either fill in a new emissions matrix, or use the one which has
		# been provided from a previous call.
		if emissions is NULL:
			e = <double*> calloc( n*self.silent_start, sizeof(double) )
			for l in range( self.silent_start ):
				for i in range( n ):
					if self.multivariate:
						e[l*n + i] = (( <Distribution> distributions[l] )._mv_log_probability( sequence+i*dim ) + 
							self.state_weights[l] ) 
					else:
						e[l*n + i] = (( <Distribution> distributions[l] )._log_probability( sequence[i] ) + 
							self.state_weights[l] )
		else:
			e = emissions

		# We must start in the start state, having emitted 0 symbols        
		for i in range(m):
			f[i] = NEGINF
		f[self.start_index] = 0.

		for l in range( self.silent_start, m ):
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
			for k in range( in_edges[l], in_edges[l+1] ):
				ki = self.in_transitions[k]
				if ki < self.silent_start or ki >= l:
					continue

				# For each current-step preceeding silent state k
				log_probability = pair_lse( log_probability,
					f[ki] + self.in_transition_log_probabilities[k] )

			# Update the table entry
			f[l] = log_probability

		for i in range( n ):
			for l in range( self.silent_start ):
				# Do the recurrence for non-silent states l
				# This holds the log total transition probability in from 
				# all previous states

				log_probability = NEGINF
				for k in range( in_edges[l], in_edges[l+1] ):
					ki = self.in_transitions[k]

					# For each previous state k
					log_probability = pair_lse( log_probability,
						f[i*m + ki] + self.in_transition_log_probabilities[k] )

				# Now set the table entry for log probability of emitting 
				# index+1 characters and ending in state l
				f[(i+1)*m + l] = log_probability + e[i + l*n]

			for l in range( self.silent_start, m ):
				# Now do the first pass over the silent states
				# This holds the log total transition probability in from 
				# all current-step non-silent states
				log_probability = NEGINF
				for k in range( in_edges[l], in_edges[l+1] ):
					ki = self.in_transitions[k]
					if ki >= self.silent_start:
						continue

					# For each current-step non-silent state k
					log_probability = pair_lse( log_probability,
						f[(i+1)*m + ki] + self.in_transition_log_probabilities[k] )

				# Set the table entry to the partial result.
				f[(i+1)*m + l] = log_probability

			for l in range( self.silent_start, m ):
				# Now the second pass through silent states, where we account
				# for transitions between silent states.

				# This holds the log total transition probability in from 
				# all current-step silent states that can have transitions into 
				# this state.
				log_probability = NEGINF
				for k in range( in_edges[l], in_edges[l+1] ):
					ki = self.in_transitions[k]
					if ki < self.silent_start or ki >= l:
						continue
					# For each current-step preceeding silent state k
					log_probability = pair_lse( log_probability,
						f[(i+1)*m + ki] + self.in_transition_log_probabilities[k] )

				# Add the previous partial result and update the table entry
				f[(i+1)*m + l] = pair_lse( f[(i+1)*m + l], log_probability )

		if emissions is NULL:
			free(e)
		return f

	cpdef numpy.ndarray backward( self, sequence ):
		'''
		Python wrapper for the backward algorithm, calculating probability by
		going backward through a sequence. Returns the full forward DP matrix.
		Each index i, j corresponds to the sum-of-all-paths log probability
		of starting with observation i aligned to hidden state j, and aligning
		observations to reach the end. Uses row normalization to dynamically 
		scale each row to prevent underflow errors.

		If the sequence is impossible, will return a matrix of nans.

		input
			sequence: a list (or numpy array) of observations

		output
			A n-by-m matrix of floats, where n = len( sequence ) and
			m = len( self.states ). This is the DP matrix for the
			backward algorithm.

		See also: 
			- Silent state handling is "essentially the same" according to
		Durbin et al., so they don't bother to explain *how to actually do it*.
		Algorithm worked out from first principles.
			- Row normalization technique explained by 
		http://www.cs.sjsu.edu/~stamp/RUA/HMM.pdf on p. 14.
		'''

		cdef numpy.ndarray sequence_ndarray
		cdef double* sequence_data
		cdef double* b
		cdef int n = len(sequence), m = len(self.states)
		cdef void** distributions = <void**> self.distributions.data
		cdef numpy.ndarray b_ndarray = numpy.zeros( (n+1, m), dtype=numpy.float64 )

		try:
			sequence_ndarray = numpy.array( sequence, dtype=numpy.float64 )
		except ValueError:
			sequence = list( map( self.keymap.__getitem__, sequence ) )
			sequence_ndarray = numpy.array( sequence, dtype=numpy.float64)

		sequence_data = <double*> sequence_ndarray.data

		with nogil:
			b = self._backward( sequence_data, distributions, n, NULL )

		for i in range(n+1):
			for j in range(m):
				b_ndarray[i, j] = b[i*m + j]

		free(b)
		return b_ndarray

	cdef double* _backward( self, double* sequence, void** distributions, 
		int n, double* emissions ) nogil:
		"""Run the backward algorithm using optimized cython code."""

		cdef int i, ir, k, kr, l, li
		cdef int p = self.silent_start, m = self.n_states
		cdef int dim = self.d

		cdef double log_probability
		cdef int* out_edges = self.out_edge_count

		cdef double* e = NULL
		cdef double* b = <double*> calloc( (n+1)*m, sizeof(double) )

		# Either fill in a new emissions matrix, or use the one which has
		# been provided from a previous call.
		if emissions is NULL:
			e = <double*> calloc( n*self.silent_start, sizeof(double) )
			for l in range( self.silent_start ):
				for i in range( n ):
					if self.multivariate:
						e[l*n + i] = ((<Distribution>distributions[l])._mv_log_probability( sequence+i*dim ) + 
							self.state_weights[l])
					else:
						e[l*n + i] = ((<Distribution>distributions[l])._log_probability( sequence[i] ) + 
							self.state_weights[l])
		else:
			e = emissions

		# We must end in the end state, having emitted len(sequence) symbols
		if self.finite == 1:
			for i in range(m):
				b[n*m + i] = NEGINF
			b[n*m + self.end_index] = 0
		else:
			for i in range(self.silent_start):
				b[n*m + i] = 0.
			for i in range(self.silent_start, m):
				b[n*m + i] = NEGINF

		for kr in range( m-self.silent_start ):
			if self.finite == 0:
				break
			# Cython arrays cannot go backwards, so modify the loop to account
			# for this.
			k = m - kr - 1

			# Do the silent states' dependencies on each other.
			# Doing it in reverse order ensures that anything we can 
			# possibly transition to is already done.
			
			if k == self.end_index:
				# We already set the log-probability for this, so skip it
				continue

			# This holds the log total probability that we go to
			# current-step silent states and then continue from there to
			# finish the sequence.
			log_probability = NEGINF
			for l in range( out_edges[k], out_edges[k+1] ):
				li = self.out_transitions[l]
				if li < k+1:
					continue

				# For each possible current-step silent state we can go to,
				# take into account just transition probability
				log_probability = pair_lse( log_probability,
					b[n*m + li] + self.out_transition_log_probabilities[l] )

			# Now this is the probability of reaching the end state given we are
			# in this silent state.
			b[n*m + k] = log_probability

		for k in range( self.silent_start ):
			if self.finite == 0:
				break
			# Do the non-silent states in the last step, which depend on
			# current-step silent states.
			
			# This holds the total accumulated log probability of going
			# to such states and continuing from there to the end.
			log_probability = NEGINF
			for l in range( out_edges[k], out_edges[k+1] ):
				li = self.out_transitions[l]
				if li < self.silent_start:
					continue

				# For each current-step silent state, add in the probability
				# of going from here to there and then continuing on to the
				# end of the sequence.
				log_probability = pair_lse( log_probability,
					b[n*m + li] + self.out_transition_log_probabilities[l] )

			# Now we have summed the probabilities of all the ways we can
			# get from here to the end, so we can fill in the table entry.
			b[n*m + k] = log_probability

		# Now that we're done with the base case, move on to the recurrence
		for ir in range( n ):
			#if self.finite == 0 and ir == 0:
			#	continue
			# Cython xranges cannot go backwards properly, redo to handle
			# it properly
			i = n - ir - 1
			for kr in range( m-self.silent_start ):
				k = m - kr - 1

				# Do the silent states' dependency on subsequent non-silent
				# states, iterating backwards to match the order we use later.
				
				# This holds the log total probability that we go to some
				# subsequent state that emits the right thing, and then continue
				# from there to finish the sequence.
				log_probability = NEGINF
				for l in range( out_edges[k], out_edges[k+1] ):
					li = self.out_transitions[l]
					if li >= self.silent_start:
						continue

					# For each subsequent non-silent state l, take into account
					# transition and emission emission probability.
					log_probability = pair_lse( log_probability,
						b[(i+1)*m + li] + self.out_transition_log_probabilities[l] +
						e[i + li*n] )

				# We can't go from a silent state here to a silent state on the
				# next symbol, so we're done finding the probability assuming we
				# transition straight to a non-silent state.
				b[i*m + k] = log_probability

			for kr in range( m-self.silent_start ):
				k = m - kr - 1

				# Do the silent states' dependencies on each other.
				# Doing it in reverse order ensures that anything we can 
				# possibly transition to is already done.
				
				# This holds the log total probability that we go to
				# current-step silent states and then continue from there to
				# finish the sequence.
				log_probability = NEGINF
				for l in range( out_edges[k], out_edges[k+1] ):
					li = self.out_transitions[l]
					if li < k+1:
						continue

					# For each possible current-step silent state we can go to,
					# take into account just transition probability
					log_probability = pair_lse( log_probability,
						b[i*m + li] + self.out_transition_log_probabilities[l] )

				# Now add this probability in with the probability accumulated
				# from transitions to subsequent non-silent states.
				b[i*m + k] = pair_lse( log_probability, b[i*m + k] )

			for k in range( self.silent_start ):
				# Do the non-silent states in the current step, which depend on
				# subsequent non-silent states and current-step silent states.
				
				# This holds the total accumulated log probability of going
				# to such states and continuing from there to the end.
				log_probability = NEGINF
				for l in range( out_edges[k], out_edges[k+1] ):
					li = self.out_transitions[l]
					if li >= self.silent_start:
						continue

					# For each subsequent non-silent state l, take into account
					# transition and emission emission probability.
					log_probability = pair_lse( log_probability,
						b[(i+1)*m + li] + self.out_transition_log_probabilities[l] +
						e[i + li*n] )

				for l in range( out_edges[k], out_edges[k+1] ):
					li = self.out_transitions[l]
					if li < self.silent_start:
						continue

					# For each current-step silent state, add in the probability
					# of going from here to there and then continuing on to the
					# end of the sequence.
					log_probability = pair_lse( log_probability,
						b[i*m + li] + self.out_transition_log_probabilities[l] )

				# Now we have summed the probabilities of all the ways we can
				# get from here to the end, so we can fill in the table entry.
				b[i*m + k] = log_probability

		if emissions is NULL:
			free(e)
		return b

	def forward_backward( self, sequence ):
		"""
		Implements the forward-backward algorithm. This is the sum-of-all-paths
		log probability that you start at the beginning of the sequence, align
		observation i to silent state j, and then continue on to the end.
		Simply, it is the probability of emitting the observation given the
		state and then transitioning one step.

		If the sequence is impossible, will return (None, None)

		input
			sequence: a list (or numpy array) of observations

		output
			A tuple of the estimated log transition probabilities, and
			the DP matrix for the FB algorithm. The DP matrix has
			n rows and m columns where n is the number of observations,
			and m is the number of non-silent states.

			* The estimated log transition probabilities are a m-by-m 
			matrix where index i, j indicates the log probability of 
			transitioning from state i to state j.

			* The DP matrix for the FB algorithm contains the sum-of-all-paths
			probability as described above.

		See also: 
			- Forward and backward algorithm implementations. A comprehensive
			description of the forward, backward, and forward-background
			algorithm is here: 
			http://en.wikipedia.org/wiki/Forward%E2%80%93backward_algorithm
		"""

		cdef numpy.ndarray sequence_ndarray
		cdef double* sequence_data
		cdef int n = len(sequence), m = len(self.states)
		cdef void** distributions = <void**> self.distributions.data

		try:
			sequence_ndarray = numpy.array( sequence, dtype=numpy.float64 )
		except ValueError:
			sequence = list( map( self.keymap.__getitem__, sequence ) )
			sequence_ndarray = numpy.array( sequence, dtype=numpy.float64)

		sequence_data = <double*> sequence_ndarray.data

		return self._forward_backward( sequence_data, distributions, n )

	cdef tuple _forward_backward( self, double* sequence, void** distributions, int n ):
		"""
		Actually perform the math here.
		"""

		cdef int i, k, j, l, ki, li
		cdef int m=len(self.states)
		cdef int dim = self.d
		cdef double* e = <double*> calloc(n*self.silent_start, sizeof(double))
		cdef double* f
		cdef double* b

		cdef numpy.ndarray expected_transitions_ndarray = numpy.zeros((m, m))
		cdef double* expected_transitions = <double*> expected_transitions_ndarray.data

		cdef numpy.ndarray emission_weights_ndarray = numpy.zeros((n, self.silent_start))
		cdef double* emission_weights = <double*> emission_weights_ndarray.data

		cdef double log_sequence_probability, log_probability
		cdef double log_transition_emission_probability_sum

		cdef int* out_edges = self.out_edge_count
		cdef int* tied_states = self.tied_state_count

		# Calculate the emissions table
		for l in range( self.silent_start ):
			for i in range( n ):
				if self.multivariate:
					e[l*n + i] = ((<Distribution>distributions[l])._mv_log_probability( sequence+i*dim ) + 
						self.state_weights[l])
				else:
					e[l*n + i] = ((<Distribution>distributions[l])._log_probability( sequence[i] ) + 
						self.state_weights[l])

		f = self._forward( sequence, distributions, n, e )
		b = self._backward( sequence, distributions, n, e )

		if self.finite == 1:
			log_sequence_probability = f[n*m + self.end_index]
		else:
			log_sequence_probability = NEGINF
			for i in range( self.silent_start ):
				log_sequence_probability = pair_lse( 
					log_sequence_probability, f[n*m + i] )
		
		# Is the sequence impossible? If so, don't bother calculating any more.
		if log_sequence_probability == NEGINF:
			print( "Warning: Sequence is impossible." )
			return ( None, None )

		for k in range( m ):
			# For each state we could have come from
			for l in range( out_edges[k], out_edges[k+1] ):
				li = self.out_transitions[l]
				if li >= self.silent_start:
					continue

				# For each state we could go to (and emit a character)
				# Sum up probabilities that we later normalize by 
				# probability of sequence.
				log_transition_emission_probability_sum = NEGINF

				for i in range( n ):
					# For each character in the sequence
					# Add probability that we start and get up to state k, 
					# and go k->l, and emit the symbol from l, and go from l
					# to the end.
					log_transition_emission_probability_sum = pair_lse( 
						log_transition_emission_probability_sum, 
						f[i*m + k] + self.out_transition_log_probabilities[l] +
						e[i + li*n] + b[(i+1)*m + li] )

				# Now divide by probability of the sequence to make it given
				# this sequence, and add as this sequence's contribution to 
				# the expected transitions matrix's k, l entry.
				expected_transitions[k*m + li] += cexp(
					log_transition_emission_probability_sum - 
					log_sequence_probability )

			for l in range( out_edges[k], out_edges[k+1] ):
				li = self.out_transitions[l]
				if li < self.silent_start:
					continue

				# For each silent state we can go to on the same character
				# Sum up probabilities that we later normalize by 
				# probability of sequence.

				log_transition_emission_probability_sum = NEGINF
				for i in range( n+1 ):
					# For each row in the forward DP table (where we can
					# have transitions to silent states) of which we have 1 
					# more than we have symbols...
						
					# Add probability that we start and get up to state k, 
					# and go k->l, and go from l to the end. In this case, 
					# we use forward and backward entries from the same DP 
					# table row, since no character is being emitted.
					log_transition_emission_probability_sum = pair_lse( 
						log_transition_emission_probability_sum, 
						f[i*m + k] + self.out_transition_log_probabilities[l]
						+ b[i*m + li] )
					
				# Now divide by probability of the sequence to make it given
				# this sequence, and add as this sequence's contribution to 
				# the expected transitions matrix's k, l entry.
				expected_transitions[k*m + li] += cexp(
					log_transition_emission_probability_sum -
					log_sequence_probability )
				
			if k < self.silent_start:
				# Now think about emission probabilities from this state
						  
				for i in xrange( n ):
					# For each symbol that came out
		   
					# What's the weight of this symbol for that state?
					# Probability that we emit index characters and then 
					# transition to state l, and that from state l we  
					# continue on to emit len(sequence) - (index + 1) 
					# characters, divided by the probability of the 
					# sequence under the model.
					# According to http://www1.icsi.berkeley.edu/Speech/
					# docs/HTKBook/node7_mn.html, we really should divide by
					# sequence probability.

					emission_weights[i*self.silent_start + k] = f[(i+1)*m + k] + b[(i+1)*m + k] - \
						log_sequence_probability


		free(e)

		return expected_transitions_ndarray, emission_weights_ndarray

	cpdef tuple viterbi( self, sequence ):
		'''
		Run the Viterbi algorithm on the sequence given the model. This finds
		the ML path of hidden states given the sequence. Returns a tuple of the
		log probability of the ML path, or (-inf, None) if the sequence is
		impossible under the model. If a path is returned, it is a list of
		tuples of the form (sequence index, state object).

		This is fundamentally the same as the forward algorithm using max
		instead of sum, except the traceback is more complicated, because
		silent states in the current step can trace back to other silent states
		in the current step as well as states in the previous step.

		input
			sequence: a list (or numpy array) of observations

		output
			A tuple of the log probabiliy of the ML path, and the sequence of
			hidden states that comprise the ML path.

		See also: 
			- Viterbi implementation described well in the wikipedia article
			http://en.wikipedia.org/wiki/Viterbi_algorithm
		'''

		cdef numpy.ndarray sequence_ndarray
		cdef double* sequence_data
		cdef tuple b
		cdef int n = len(sequence), m = len(self.states)
		cdef void** distributions = <void**> self.distributions.data
		cdef numpy.ndarray b_ndarray = numpy.zeros( (n+1, m), dtype=numpy.float64 )

		try:
			sequence_ndarray = numpy.array( sequence, dtype=numpy.float64 )
		except ValueError:
			sequence = list( map( self.keymap.get, sequence, [-1] * len(sequence) ) )
			sequence_ndarray = numpy.array( sequence, dtype=numpy.float64)

		sequence_data = <double*> sequence_ndarray.data

		return self._viterbi( sequence_data, distributions, n )

	cdef tuple _viterbi(self, double* sequence, void** distributions, int n ):
		"""		
		This fills in self.v, the Viterbi algorithm DP table.
		
		This is fundamentally the same as the forward algorithm using max
		instead of sum, except the traceback is more complicated, because silent
		states in the current step can trace back to other silent states in the
		current step as well as states in the previous step.
		"""

		cdef int m = len(self.states)
		cdef int p = self.silent_start
		cdef int i, l, k, ki
		cdef int dim = self.d

		cdef int* tracebackx = <int*> calloc( (n+1)*m, sizeof(int) )
		cdef int* tracebacky = <int*> calloc( (n+1)*m, sizeof(int) )
		cdef double* v = <double*> calloc( (n+1)*m, sizeof(double) )
		cdef double* e = <double*> calloc( (n*self.silent_start), sizeof(double) )

		cdef double state_log_probability
		cdef int* in_edges = self.in_edge_count

		# Fill in the emission table
		for l in range( self.silent_start ):
			for i in range( n ):
				if self.multivariate:
					e[l*n + i] = ((<Distribution>distributions[l])._mv_log_probability( sequence+i*dim ) + 
						self.state_weights[l])
				else:
					e[l*n + i] = ((<Distribution>distributions[l])._log_probability( sequence[i] ) + 
						self.state_weights[l])

		for i in range( m ):
			v[i] = NEGINF
		v[self.start_index] = 0

		for l in range( self.silent_start, m ):
			# Handle transitions between silent states before the first symbol
			# is emitted. No non-silent states have non-zero probability yet, so
			# we can ignore them.
			if l == self.start_index:
				# Start state log-probability is already right. Don't touch it.
				continue

			for k in range( in_edges[l], in_edges[l+1] ):
				ki = self.in_transitions[k]
				if ki < self.silent_start or ki >= l:
					continue

				# For each current-step preceeding silent state k
				# This holds the log-probability coming that way
				state_log_probability = v[ki] + self.in_transition_log_probabilities[k]

				if state_log_probability > v[l]:
					v[l] = state_log_probability
					tracebackx[l] = 0
					tracebacky[l] = ki

		for i in range( n ):
			for l in range( self.silent_start ):
				# Do the recurrence for non-silent states l
				# Start out saying the best likelihood we have is -inf
				v[(i+1)*m + l] = NEGINF
				
				for k in range( in_edges[l], in_edges[l+1] ):
					ki = self.in_transitions[k]

					# For each previous state k
					# This holds the log-probability coming that way
					state_log_probability = v[i*m + ki] + \
						self.in_transition_log_probabilities[k] + e[i + l*n]

					if state_log_probability > v[(i+1)*m + l]:
						v[(i+1)*m + l] = state_log_probability
						tracebackx[(i+1)*m + l] = i
						tracebacky[(i+1)*m + l] = ki

			for l in range( self.silent_start, m ):
				# Now do the first pass over the silent states, finding the best
				# current-step non-silent state they could come from.
				# Start out saying the best likelihood we have is -inf
				v[(i+1)*m + l] = NEGINF

				for k in range( in_edges[l], in_edges[l+1] ):
					ki = self.in_transitions[k]
					if ki >= self.silent_start:
						continue

					# For each current-step non-silent state k
					# This holds the log-probability coming that way
					state_log_probability = v[(i+1)*m + ki] + \
						self.in_transition_log_probabilities[k]

					if state_log_probability > v[(i+1)*m + l]:
						v[(i+1)*m + l] = state_log_probability
						tracebackx[(i+1)*m + l] = i+1
						tracebacky[(i+1)*m + l] = ki

			for l in range( self.silent_start, m ):
				# Now the second pass through silent states, where we check the
				# silent states that could potentially reach here and see if
				# they're better than the non-silent states we found.

				for k in range( in_edges[l], in_edges[l+1] ):
					ki = self.in_transitions[k]
					if ki < self.silent_start or ki >= l:
						continue

					# For each current-step preceeding silent state k
					# This holds the log-probability coming that way
					state_log_probability = v[(i+1)*m + ki] + \
						self.in_transition_log_probabilities[k]

					if state_log_probability > v[(i+1)*m + l]:
						v[(i+1)*m + l] = state_log_probability
						tracebackx[(i+1)*m + l] = i+1
						tracebacky[(i+1)*m + l] = ki

		# Now the DP table is filled in. If this is a finite model, get the
		# log likelihood of ending up in the end state after following the
		# ML path through the model. If an infinite sequence, find the state
		# which the ML path ends in, and begin there.
		cdef int end_index
		cdef double log_likelihood

		if self.finite == 1:
			log_likelihood = v[n*m + self.end_index]
			end_index = self.end_index
		else:
			end_index = -1
			log_likelihood = NEGINF
			for i in range(m):
				if v[n*m + i] > log_likelihood:
					log_likelihood = v[n*m + i]
					end_index = i

		if log_likelihood == NEGINF:
			# The path is impossible, so don't even try a traceback. 
			return ( log_likelihood, None )

		# Otherwise, do the traceback
		# This holds the path, which we construct in reverse order
		cdef list path = []
		cdef int px = n, py = end_index, npx

		# This holds our current position (character, state) AKA (i, k).
		# We start at the end state
		while px != 0 or py != self.start_index:
			# Until we've traced back to the start...
			# Put the position in the path, making sure to look up the state
			# object to use instead of the state index.
			path.append( ( py, self.states[py] ) )

			# Go backwards
			npx = tracebackx[px*m + py]
			py = tracebacky[px*m + py]
			px = npx

		# We've now reached the start (if we didn't raise an exception because
		# we messed up the traceback)
		# Record that we start at the start
		path.append( (py, self.states[py] ) )

		# Flip the path the right way around
		path.reverse()

		# Return the log-likelihood and the right-way-arounded path
		return ( log_likelihood, path )

	def predict_proba( self, sequence ):
		"""sklearn interface for the posterior calculation."""

		return numpy.exp( self.predict_log_proba( sequence ) )

	def predict_log_proba( self, sequence ):
		"""sklearn interface for the log posterior calculation."""

		return self.posterior( sequence )

	def posterior( self, sequence ):
		"""
		Calculate the posterior of each state given each observation. This
		is the emission matrix from the forward-backward algorithm. Maximum
		a posterior, or posterior decoding, will decode this matrix.
		"""

		return numpy.array( self._posterior( numpy.array( sequence ) ) )

	cdef double [:,:] _posterior( self, numpy.ndarray sequence ):
		"""
		Fill out the responsibility/posterior/emission matrix. There are a lot
		of names for this.
		"""

		cdef int i, k, l, li
		cdef int m=len(self.states), n=len(sequence)
		cdef double [:,:] f, b
		cdef double [:,:] emission_weights = numpy.zeros((n, self.silent_start))
		cdef double log_sequence_probability


		# Fill in both the F and B DP matrices.
		f = self.forward( sequence )
		b = self.backward( sequence )

		# Find out the probability of the sequence
		if self.finite == 1:
			log_sequence_probability = f[ n, self.end_index ]
		else:
			log_sequence_probability = NEGINF
			for i in xrange( self.silent_start ):
				log_sequence_probability = pair_lse( 
					log_sequence_probability, f[ n, i ] )
		
		# Is the sequence impossible? If so, don't bother calculating any more.
		if log_sequence_probability == NEGINF:
			print( "Warning: Sequence is impossible." )
			return ( None, None )

		for k in xrange( m ):				
			if k < self.silent_start:				  
				for i in xrange( n ):
					# For each symbol that came out
					# What's the weight of this symbol for that state?
					# Probability that we emit index characters and then 
					# transition to state l, and that from state l we  
					# continue on to emit len(sequence) - (index + 1) 
					# characters, divided by the probability of the 
					# sequence under the model.
					# According to http://www1.icsi.berkeley.edu/Speech/
					# docs/HTKBook/node7_mn.html, we really should divide by
					# sequence probability.
					emission_weights[i,k] = f[i+1, k] + b[i+1, k] - \
						log_sequence_probability

		return emission_weights

	def predict( self, sequence, algorithm='map' ):
		"""sklearn interface. algorithm can be `map` or `viterbi`"""

		if algorithm == 'map':
			return [ state_id for state_id, state in self.maximum_a_posteriori( sequence )[1] ]
		return [ state_id for state_id, state in self.viterbi( sequence )[1] ]

	def maximum_a_posteriori( self, sequence ):
		"""
		MAP decoding is an alternative to viterbi decoding, which returns the
		most likely state for each observation, based on the forward-backward
		algorithm. This is also called posterior decoding. This method is
		described on p. 14 of http://ai.stanford.edu/~serafim/CS262_2007/
		notes/lecture5.pdf

		WARNING: This may produce impossible sequences.
		"""

		return self._maximum_a_posteriori( numpy.array( sequence ) )

	
	cdef tuple _maximum_a_posteriori( self, numpy.ndarray sequence ):
		"""
		Actually perform the math here. Instead of calling forward-backward
		to get the emission weights, it's calculated here so that time isn't
		wasted calculating the transition counts. 
		"""

		cdef int i, k, l, li
		cdef int m=len(self.states), n=len(sequence)
		cdef double [:,:] emission_weights = self._posterior( sequence )

		cdef list path = []
		cdef double maximum_emission_weight
		cdef double log_probability_sum = 0
		cdef int maximum_index

		# Go through each symbol and determine what the most likely state
		# that it came from is.
		for k in xrange( n ):
			maximum_index = -1
			maximum_emission_weight = NEGINF

			# Go through each hidden state and see which one has the maximal
			# weight for emissions. Tied states are not taken into account
			# here, because we are not performing training.
			for l in xrange( self.silent_start ):
				if emission_weights[k, l] > maximum_emission_weight:
					maximum_emission_weight = emission_weights[k, l]
					maximum_index = l

			path.append( ( maximum_index, self.states[maximum_index] ) )
			log_probability_sum += maximum_emission_weight 

		return log_probability_sum, path

	def fit( self, sequences, stop_threshold=1E-9, min_iterations=0,
		max_iterations=1e8, algorithm='baum-welch', verbose=True,
		transition_pseudocount=0, use_pseudocount=False, edge_inertia=0.0,
		distribution_inertia=0.0, n_jobs=1  ):
		"""sklearn interface to the train method."""

		return self.train( sequences, stop_threshold, min_iterations,
			max_iterations, algorithm, verbose, transition_pseudocount,
			use_pseudocount, edge_inertia, distribution_inertia, n_jobs=1 )

	cpdef train( self, sequences, stop_threshold=1E-9, min_iterations=0,
		max_iterations=1e8, algorithm='baum-welch', verbose=True,
		transition_pseudocount=0, use_pseudocount=False, edge_inertia=0.0,
		distribution_inertia=0.0, n_jobs=1 ):
		"""
		Given a list of sequences, performs re-estimation on the model
		parameters. The two supported algorithms are "baum-welch" and
		"viterbi," indicating their respective algorithm. 

		Use either a uniform transition_pseudocount, or the
		previously specified ones by toggling use_pseudocount if pseudocounts
		are needed. edge_inertia can make the new edge parameters be a mix of
		new parameters and the old ones, and distribution_inertia does the same
		thing for distributions instead of transitions.

		Baum-Welch: Iterates until the log of the "score" (total likelihood of 
		all sequences) changes by less than stop_threshold. Returns the final 
		log score.
	
		Always trains for at least min_iterations, and terminate either when
		reaching max_iterations, or the training improvement is smaller than
		stop_threshold.

		Viterbi: Training performed by running each sequence through the
		viterbi decoding algorithm. Edge weight re-estimation is done by 
		recording the number of times a hidden state transitions to another 
		hidden state, and using the percentage of time that edge was taken.
		Emission re-estimation is done by retraining the distribution on
		every sample tagged as belonging to that state.

		Baum-Welch training is usually the more accurate method, but takes
		significantly longer. Viterbi is a good for situations in which
		accuracy can be sacrificed for time.
		"""

		# Convert the boolean into an integer for downstream use.
		use_pseudocount = int( use_pseudocount )

		if isinstance( sequences, numpy.ndarray ):
			sequences = list(sequences)

		if algorithm.lower() == 'labelled' or algorithm.lower() == 'labeled':
			labels = []
			for i, sequence in enumerate(sequences):
				labels.append( sequence[1] )
				sequences[i] = numpy.array( sequence[0] )

			# If calling the labelled training algorithm, then sequences is a
			# list of tuples of sequence, path pairs, not a list of sequences.
			# The log probability sum is the log-sum-exp of the log
			# probabilities of all members of the sequence. In this case,
			# sequences is made up of ( sequence, path ) tuples, instead of
			# just sequences.
			improvement = self._train_labelled( sequences, labels, transition_pseudocount, 
				use_pseudocount, edge_inertia, distribution_inertia )

		if algorithm.lower() == 'viterbi':
			improvement = self._train_loop( sequences, 'viterbi', stop_threshold,
				min_iterations, max_iterations, verbose, 
				transition_pseudocount, use_pseudocount, edge_inertia,
				distribution_inertia, n_jobs )

		elif algorithm.lower() == 'baum-welch':
			improvement = self._train_loop( sequences, 'baum-welch', stop_threshold,
				min_iterations, max_iterations, verbose, 
				transition_pseudocount, use_pseudocount, edge_inertia,
				distribution_inertia, n_jobs )

		out_edges = self.out_edge_count

		for k in range( self.silent_start ):
			for l in range( out_edges[k], out_edges[k+1] ):
				li = self.out_transitions[l]
				prob = self.out_transition_log_probabilities[l]

				self.graph[self.states[k]][self.states[li]]['probability'] = prob

		if verbose:
			print( "Total Training Improvement: {}".format( improvement ) )
		return improvement

	cdef double _train_loop( self, list sequences, str algorithm, double stop_threshold,
		int min_iterations, int max_iterations, bint verbose, double transition_pseudocount,
		bint use_pseudocount, double edge_inertia, double distribution_inertia, int n_jobs ):
		"""Wrapper for the training loop of unsupervised learning.

		This is used for both Viterbi and Baum-Welch training, where Viterbi produces
		hard assignments of edge transitions and state probabilities, and Baum-Welch
		uses soft assignments. This is the outer loop which runs until convergence,
		calling for one iteration of either training algorithm.

		Convergence is defined by running for at least min_iterations and not
		increasing until the log likelihood of the dataset changes by less than
		stop_threshold.
		"""

		cdef int iteration = 0
		cdef double improvement = INF
		cdef double initial_log_probability_sum
		cdef double trained_log_probability_sum
		cdef double last_log_probability_sum
		cdef bint check_input = algorithm.lower() == 'viterbi'

		if algorithm.lower() == 'viterbi':
			sequences = list( map( numpy.array, sequences ) )
		else:
			for i in range( len(sequences) ):
				try:
					sequences[i] = numpy.array( sequences[i], dtype=numpy.float64 )
				except: 
					sequences[i] = numpy.array( list( map( self.keymap.__getitem__, 
						                                   sequences[i] ) ), 
					                            dtype=numpy.float64 )

		with Parallel( n_jobs=n_jobs, backend='threading' ) as parallel:
			initial_log_probability_sum = log_probability( self, sequences, n_jobs, parallel, check_input=check_input )
			trained_log_probability_sum = initial_log_probability_sum

			while improvement > stop_threshold or iteration < min_iterations:
				if iteration >= max_iterations:
					break 

				iteration +=1
				last_log_probability_sum = trained_log_probability_sum

				if algorithm.lower() == 'baum-welch':
					self._train_baum_welch_once( sequences, transition_pseudocount,
						                         use_pseudocount, edge_inertia,
						                         distribution_inertia, n_jobs, parallel )
				elif algorithm.lower() == 'viterbi':
					self._train_viterbi_once( sequences, transition_pseudocount,
						                      use_pseudocount, edge_inertia,
						                      distribution_inertia, n_jobs, parallel )

				trained_log_probability_sum = log_probability( self, sequences, n_jobs, parallel, check_input=check_input )
				improvement = trained_log_probability_sum - last_log_probability_sum
				
				if verbose:
					print( "Training improvement: {}".format(improvement) )

		return trained_log_probability_sum - initial_log_probability_sum


	cdef void _train_baum_welch_once(self, list sequences, 
		double transition_pseudocount, bint use_pseudocount, double edge_inertia, 
		double distribution_inertia, int n_jobs, object parallel ):
		"""
		Given a list of sequences, perform Baum-Welch iterative re-estimation on
		the model parameters.
		
		Iterates until the log of the "score" (total likelihood of all 
		sequences) changes by less than stop_threshold. Returns the final log
		score.
		
		Always trains for at least min_iterations.
		"""

		cdef double* expected_transitions = self.expected_transitions
		memset( expected_transitions, 0, self.n_edges*sizeof(double) )
		
		parallel( delayed( self._baum_welch_summarize, check_pickle=False )(
			sequence, self.distributions ) for sequence in sequences )

		self._baum_welch_update( transition_pseudocount, use_pseudocount, 
			edge_inertia, distribution_inertia )


	cpdef _baum_welch_summarize( self, numpy.ndarray sequence_ndarray, 
		numpy.ndarray distributions_ndarray ):
		"""Python wrapper for the summarization step.

		This is done to ensure compatibility with joblib's multithreading
		API. It just calls the cython update, but provides a Python wrapper
		which joblib can easily wrap.
		"""

		cdef double* sequence = <double*> sequence_ndarray.data
		cdef void** distributions = <void**> distributions_ndarray.data
		cdef int n = sequence_ndarray.shape[0]

		with nogil:
			self.__baum_welch_summarize( sequence, distributions, n)

	cdef void __baum_welch_summarize( self, double* sequence, void** distributions, 
		int n ) nogil:
		"""Collect sufficient statistics on a single sequence."""

		cdef int i, k, l, li
		cdef int m = self.n_states
		cdef int dim = self.d

		cdef double log_sequence_probability
		cdef double log_transition_emission_probability_sum 

		cdef double* expected_transitions = <double*> calloc(self.n_edges, sizeof(double))
		cdef double* f
		cdef double* b
		cdef double* e

		cdef int* tied_edges = self.tied_edge_group_size
		cdef int* tied_states = self.tied_state_count
		cdef int* out_edges = self.out_edge_count

		cdef int* visited = <int*> calloc( self.silent_start, sizeof(int) )
		cdef double* weights = <double*> calloc( n, sizeof(double) )

		e = <double*> calloc( n*self.silent_start, sizeof(double) )
		for l in range( self.silent_start ):
			for i in range( n ):
				if self.multivariate:
					e[l*n + i] = ((<Distribution>distributions[l])._mv_log_probability( sequence+i*dim ) + 
						self.state_weights[l])
				else:
					e[l*n + i] = ((<Distribution>distributions[l])._log_probability( sequence[i] ) + 
						self.state_weights[l])

		f = self._forward( sequence, distributions, n, e )
		b = self._backward( sequence, distributions, n, e )

		if self.finite == 1:
			log_sequence_probability = f[n*m + self.end_index]
		else:
			log_sequence_probability = NEGINF
			for i in range( self.silent_start ):
				log_sequence_probability = pair_lse( f[n*m + i],
					log_sequence_probability )

		# Is the sequence impossible? If so, we can't train on it, so skip 
		# it
		if log_sequence_probability != NEGINF:
			for k in range( m ):
				# For each state we could have come from
				for l in range( out_edges[k], out_edges[k+1] ):
					li = self.out_transitions[l]
					if li >= self.silent_start:
						continue

					# For each state we could go to (and emit a character)
					# Sum up probabilities that we later normalize by 
					# probability of sequence.
					log_transition_emission_probability_sum = NEGINF
					for i in range( n ):
						# For each character in the sequence
						# Add probability that we start and get up to state k, 
						# and go k->l, and emit the symbol from l, and go from l
						# to the end.
						log_transition_emission_probability_sum = pair_lse( 
							log_transition_emission_probability_sum, 
							f[i*m + k] + 
							self.out_transition_log_probabilities[l] + 
							e[i + li*n] + b[(i+1)*m + li] )

					# Now divide by probability of the sequence to make it given
					# this sequence, and add as this sequence's contribution to 
					# the expected transitions matrix's k, l entry.
					expected_transitions[l] += cexp(
						log_transition_emission_probability_sum - 
						log_sequence_probability)

				for l in range( out_edges[k], out_edges[k+1] ):
					li = self.out_transitions[l]
					if li < self.silent_start:
						continue
					# For each silent state we can go to on the same character
					# Sum up probabilities that we later normalize by 
					# probability of sequence.
					log_transition_emission_probability_sum = NEGINF
					for i in range( n+1 ):
						# For each row in the forward DP table (where we can
						# have transitions to silent states) of which we have 1 
						# more than we have symbols...

						# Add probability that we start and get up to state k, 
						# and go k->l, and go from l to the end. In this case, 
						# we use forward and backward entries from the same DP 
						# table row, since no character is being emitted.
						log_transition_emission_probability_sum = pair_lse( 
							log_transition_emission_probability_sum, 
							f[i*m + k] + self.out_transition_log_probabilities[l] 
							+ b[i*m + li] )

					# Now divide by probability of the sequence to make it given
					# this sequence, and add as this sequence's contribution to 
					# the expected transitions matrix's k, l entry.
					expected_transitions[l] += cexp(
						log_transition_emission_probability_sum -
						log_sequence_probability )

				memset( visited, 0, self.silent_start*sizeof(int) )
				if k < self.silent_start:
					# If another state in the set of tied states has already
					# been visited, we don't want to retrain.
					if visited[k] == 1:
						continue

					# Mark that we've visited this state
					visited[k] = 1

					# Mark that we've visited all other states in this state
					# group.
					for l in range( tied_states[k], tied_states[k+1] ):
						li = self.tied[l]
						visited[li] = 1

					for i in range( n ):
						# For each symbol that came out
						# What's the weight of this symbol for that state?
						# Probability that we emit index characters and then 
						# transition to state l, and that from state l we  
						# continue on to emit len(sequence) - (index + 1) 
						# characters, divided by the probability of the 
						# sequence under the model.
						# According to http://www1.icsi.berkeley.edu/Speech/
						# docs/HTKBook/node7_mn.html, we really should divide by
						# sequence probability.
						weights[i] = cexp( f[(i+1)*m + k] + b[(i+1)*m + k] - 
							log_sequence_probability )

						with gil:
							print( "hmm: ", f[(i+1)*m + k], b[(i+1)*m + k], log_sequence_probability )
					
						for l in range( tied_states[k], tied_states[k+1] ):
							li = self.tied[l]
							weights[i] += cexp( f[(i+1)*m + li] + b[(i+1)*m + li] -
								log_sequence_probability )


					(<Distribution>distributions[k])._summarize(sequence, weights, n)

			# Update the master expected transitions vector representing the sparse matrix.
			with gil:
				for i in range(self.n_edges):
					self.expected_transitions[i] += expected_transitions[i]

			free(expected_transitions)
			free(e)
			free(visited)
			free(weights)

	cdef void _baum_welch_update(self, double transition_pseudocount, 
		bint use_pseudocount, double edge_inertia, double distribution_inertia ):
		"""Update the transition matrix and emission distributions."""      

		cdef int k, i, l, li, m = len( self.states ), n, idx
		cdef int* in_edges = self.in_edge_count
		cdef int* out_edges = self.out_edge_count

		# Define several helped variables.
		cdef int* tied_states = self.tied_state_count

		cdef double* norm
		cdef int* visited = <int*> calloc( self.silent_start, sizeof(int) )

		cdef double probability, tied_edge_probability
		cdef int start, end
		cdef int* tied_edges = self.tied_edge_group_size

		cdef double* expected_transitions = <double*> calloc( m*m, sizeof(double) )

		with nogil:
			for k in range( m ):
				for l in range( out_edges[k], out_edges[k+1] ):
					li = self.out_transitions[l]
					expected_transitions[k*m + li] = self.expected_transitions[l]

			# We now have expected_transitions taking into account all sequences.
			# And a list of all emissions, and a weighting of each emission for each
			# state
			# Normalize transition expectations per row (so it becomes transition 
			# probabilities)
			# See http://stackoverflow.com/a/8904762/402891
			# Only modifies transitions for states a transition was observed from.
			norm = <double*> calloc( m, sizeof(double) )

			# Go through the tied state groups and add transitions from each member
			# in the group to the other members of the group.
			# For each group defined.
			for k in range( self.n_tied_edge_groups-1 ):
				tied_edge_probability = 0.

				# For edge in this group, get the sum of the edges
				for l in range( tied_edges[k], tied_edges[k+1] ):
					start = self.tied_edges_starts[l]
					end = self.tied_edges_ends[l]
					tied_edge_probability += expected_transitions[start*m + end]

				# Update each entry
				for l in range( tied_edges[k], tied_edges[k+1] ):
					start = self.tied_edges_starts[l]
					end = self.tied_edges_ends[l]
					expected_transitions[start*m + end] = tied_edge_probability

			# Calculate the regularizing norm for each node
			for k in range( m ):
				for l in range( out_edges[k], out_edges[k+1] ):
					li = self.out_transitions[l]
					norm[k] += expected_transitions[k*m + li] + \
						transition_pseudocount + \
						self.out_transition_pseudocounts[l] * use_pseudocount

			# For every node, update the transitions appropriately
			for k in range( m ):
				# Recalculate each transition out from that node and update
				# the vector of out transitions appropriately
				if norm[k] > 0:
					for l in range( out_edges[k], out_edges[k+1] ):
						li = self.out_transitions[l]
						probability = ( expected_transitions[k*m + li] +
							transition_pseudocount + 
							self.out_transition_pseudocounts[l] * use_pseudocount)\
							/ norm[k]
						self.out_transition_log_probabilities[l] = _log(
							cexp( self.out_transition_log_probabilities[l] ) * 
							edge_inertia + probability * ( 1 - edge_inertia ) )

				# Recalculate each transition in to that node and update the
				# vector of in transitions appropriately 
				for l in range( in_edges[k], in_edges[k+1] ):
					li = self.in_transitions[l]
					if norm[li] > 0:
						probability = ( expected_transitions[li*m + k] +
							transition_pseudocount +
							self.in_transition_pseudocounts[l] * use_pseudocount )\
							/ norm[li]
						self.in_transition_log_probabilities[l] = _log( 
							cexp( self.in_transition_log_probabilities[l] ) *
							edge_inertia + probability * ( 1 - edge_inertia ) )

			memset( visited, 0, self.silent_start*sizeof(int) )
			for k in range( self.silent_start ):
				# If this distribution has already been trained because it is tied
				# to an earlier state, don't bother retraining it as that would
				# waste time.
				if visited[k] == 1:
					continue
				
				# Mark that we've visited this state
				visited[k] = 1

				# Mark that we've visited all states in this tied state group.
				for l in range( tied_states[k], tied_states[k+1] ):
					li = self.tied[l]
					visited[li] = 1

				# Re-estimate the emission distribution for every non-silent state.
				# Take each emission weighted by the probability that we were in 
				# this state when it came out, given that the model generated the 
				# sequence that the symbol was part of. Take into account tied
				# states by only training that distribution one time, since many
				# states are pointing to the same distribution object.
				with gil:
					self.states[k].distribution.from_summaries( 
						inertia=distribution_inertia )

		free(norm)
		free(visited)

	cdef void _train_viterbi_once( self, list sequences, 
		double transition_pseudocount, int use_pseudocount, 
		double edge_inertia, double distribution_inertia, int n_jobs,
		object parallel ):
		"""
		Performs a simple viterbi training algorithm. Each sequence is tagged
		using the viterbi algorithm, and both emissions and transitions are
		updated based on the probabilities in the observations.
		"""

		cdef numpy.ndarray sequence
		cdef numpy.ndarray path
		cdef list labels = []

		for sequence in sequences:
			# Run the viterbi decoding on each observed sequence
			log_sequence_probability, sequence_path = self.viterbi( sequence )
			path = numpy.empty( len(sequence_path), dtype=object )

			if log_sequence_probability == NEGINF:
				print( "Warning: skipped impossible sequence {}".format(sequence) )
				continue

			# Strip off the ID
			for i, (index, state) in enumerate( sequence_path ):
				path[i] = state

			labels.append( path )

		self._train_labelled( sequences, labels, transition_pseudocount, use_pseudocount, 
			edge_inertia, distribution_inertia )

	cdef double _train_labelled( self, list sequences, list sequence_labels,
		double transition_pseudocount, int use_pseudocount, double edge_inertia, 
		double distribution_inertia ):
		"""
		Perform training on a set of sequences where the state path is known,
		thus, labelled. Pass in a list of tuples, where each tuple is of the
		form (sequence, labels).
		"""

		cdef int i, j, m=len(self.states), n, a, b, k, l, li
		cdef numpy.ndarray sequence 
		cdef numpy.ndarray labels
		cdef State label
		cdef list symbols = [ [] for i in xrange(m) ]
		cdef int* tied_states = self.tied_state_count
		cdef double initial_log_probability = log_probability( self, sequences )

		# Define matrices for the transitions between states, and the weight of
		# each emission for each state for training later.
		cdef int [:,:] transition_counts
		transition_counts = numpy.zeros((m,m), dtype=numpy.int32)

		cdef int* in_edges = self.in_edge_count
		cdef int* out_edges = self.out_edge_count

		# Define a mapping of state objects to index 
		cdef dict indices = { self.states[i]: i for i in xrange( m ) }

		# Keep track of the log score across all sequences
		for sequence, labels in izip( sequences, sequence_labels ):
			n = sequence.shape[0]

			# Keep track of the number of transitions from one state to another
			transition_counts[ self.start_index, indices[labels[0]] ] += 1
			for i in xrange( len(labels)-1 ):
				a = indices[labels[i]]
				b = indices[labels[i+1]]
				transition_counts[ a, b ] += 1
			transition_counts[ indices[labels[-1]], self.end_index ] += 1

			# Indicate whether or not an emission came from a state or not.
			i = 0
			for label in labels:
				if label.is_silent():
					continue
				
				# Add the symbol to the list of symbols emitted from a given
				# state.
				k = indices[label]
				symbols[k].append( sequence[i] )

				# Also add the symbol to the list of symbols emitted from any
				# tied states to the current state.
				for l in xrange( tied_states[k], tied_states[k+1] ):
					li = self.tied[l]
					symbols[li].append( sequence[i] )

				# Move to the next observation.
				i += 1

		cdef double [:] norm = numpy.zeros( m )
		cdef double probability

		cdef int* tied_edges = self.tied_edge_group_size
		cdef int tied_edge_probability 
		# Go through the tied state groups and add transitions from each member
		# in the group to the other members of the group.
		# For each group defined.
		for k in xrange( self.n_tied_edge_groups-1 ):
			tied_edge_probability = 0

			# For edge in this group, get the sum of the edges
			for l in xrange( tied_edges[k], tied_edges[k+1] ):
				start = self.tied_edges_starts[l]
				end = self.tied_edges_ends[l]
				tied_edge_probability += transition_counts[start, end]

			# Update each entry
			for l in xrange( tied_edges[k], tied_edges[k+1] ):
				start = self.tied_edges_starts[l]
				end = self.tied_edges_ends[l]
				transition_counts[start, end] = tied_edge_probability

		# Calculate the regularizing norm for each node for normalizing the
		# transition probabilities.
		for k in xrange( m ):
			for l in xrange( out_edges[k], out_edges[k+1] ):
				li = self.out_transitions[l]
				norm[k] += transition_counts[k, li] + transition_pseudocount +\
					self.out_transition_pseudocounts[l] * use_pseudocount

		# For every node, update the transitions appropriately
		for k in xrange( m ):
			# Recalculate each transition out from that node and update
			# the vector of out transitions appropriately
			if norm[k] > 0:
				for l in xrange( out_edges[k], out_edges[k+1] ):
					li = self.out_transitions[l]
					probability = ( transition_counts[k, li] +
						transition_pseudocount + 
						self.out_transition_pseudocounts[l] * use_pseudocount)\
						/ norm[k]
					self.out_transition_log_probabilities[l] = _log(
						cexp( self.out_transition_log_probabilities[l] ) * 
						edge_inertia + probability * ( 1 - edge_inertia ) )

			# Recalculate each transition in to that node and update the
			# vector of in transitions appropriately 
			for l in xrange( in_edges[k], in_edges[k+1] ):
				li = self.in_transitions[l]
				if norm[li] > 0:
					probability = ( transition_counts[li, k] +
						transition_pseudocount +
						self.in_transition_pseudocounts[l] * use_pseudocount )\
						/ norm[li]
					self.in_transition_log_probabilities[l] = _log( 
						cexp( self.in_transition_log_probabilities[l] ) *
						edge_inertia + probability * ( 1 - edge_inertia ) )

		cdef int [:] visited = numpy.zeros( self.silent_start,
			dtype=numpy.int32 )

		for k in xrange( self.silent_start ):
			# If this distribution has already been trained because it is tied
			# to an earlier state, don't bother retraining it as that would
			# waste time.
			if visited[k] == 1:
				continue
			visited[k] = 1

			# We only want to train each distribution object once, and so we
			# don't want to visit states where the distribution has already
			# been retrained.
			for l in xrange( tied_states[k], tied_states[k+1] ):
				li = self.tied[l]
				visited[li] = 1

			# Now train this distribution on the symbols collected. If there
			# are tied states, this will be done once per set of tied states
			# in order to save time.
			self.states[k].distribution.train( symbols[k], 
				inertia=distribution_inertia )

		return log_probability( self, sequences ) - initial_log_probability

	def to_json( self, separators=(',', ' : '), indent=4 ):
		"""
		Write out the HMM to JSON format, recursively including state and
		distribution information.
		"""
		
		model = { 
					'class' : 'HiddenMarkovModel',
					'name'  : self.name,
					'start' : json.loads( self.start.to_json() ),
					'end'   : json.loads( self.end.to_json() ),
					'states' : [ json.loads( state.to_json() ) for state in self.states ],
					'end_index' : self.end_index,
					'start_index' : self.start_index,
					'silent_index' : self.silent_start
				}

		indices = { state: i for i, state in enumerate( self.states )}

		# Get the number of groups of edges which are tied
		groups = []
		n = self.n_tied_edge_groups-1
		
		# Go through each group one at a time
		for i in xrange(n):
			# Create an empty list for that group
			groups.append( [] )

			# Go through each edge in that group
			start, end = self.tied_edge_group_size[i], self.tied_edge_group_size[i+1]

			# Add each edge as a tuple of indices
			for j in xrange( start, end ):
				groups[i].append( ( self.tied_edges_starts[j], self.tied_edges_ends[j] ) )

		# Now reverse this into a dictionary, such that each pair of edges points
		# to a label (a number in this case)
		d = { tup : i for i in xrange(n) for tup in groups[i] }

		# Get all the edges from the graph
		edges = []
		for start, end, data in self.graph.edges_iter( data=True ):
			# If this edge is part of a group of tied edges, annotate this group
			# it is a part of
			s, e = indices[start], indices[end]
			prob, pseudocount = math.e**data['probability'], data['pseudocount']
			edge = (s, e)
			edges.append( ( s, e, prob, pseudocount, d.get( edge, None ) ) )

		model['edges'] = edges

		# Get distribution tie information
		ties = []
		for i in xrange( self.silent_start ):
			start, end = self.tied_state_count[i], self.tied_state_count[i+1]

			for j in xrange( start, end ):
				ties.append( ( i, self.tied[j] ) )

		model['distribution ties'] = ties
		return json.dumps( model, separators=separators, indent=indent )

			
	@classmethod
	def from_json( cls, s, verbose=False ):
		"""
		Read a HMM from the given JSON, build the model, and bake it.
		"""

		# Load a dictionary from a JSON formatted string
		try:
			d = json.loads( s )
		except:
			try:
				with open( s, 'r' ) as infile:
					d = json.load( infile )
			except:
				raise IOError("String must be properly formatted JSON or filename of properly formatted JSON.")

		# Make a new generic HMM
		model = HiddenMarkovModel( str(d['name']) )

		# Load all the states from JSON formatted strings
		states = [ State.from_json( json.dumps(j) ) for j in d['states'] ]
		for i, j in d['distribution ties']:
			# Tie appropriate states together
			states[i].tie( states[j] )

		# Add all the states to the model
		model.add_states( states )

		# Indicate appropriate start and end states
		model.start = states[ d['start_index'] ]
		model.end = states[ d['end_index'] ]

		# Add all the edges to the model
		for start, end, probability, pseudocount, group in d['edges']:
			model.add_transition( states[start], states[end], probability, 
				pseudocount, group )

		# Bake the model
		model.bake( verbose=verbose )
		return model
	
	@classmethod
	def from_matrix( cls, transition_probabilities, distributions, starts, ends,
		state_names=None, name=None ):
		"""
		Take in a 2D matrix of floats of size n by n, which are the transition
		probabilities to go from any state to any other state. May also take in
		a list of length n representing the names of these nodes, and a model
		name. Must provide the matrix, and a list of size n representing the
		distribution you wish to use for that state, a list of size n indicating
		the probability of starting in a state, and a list of size n indicating
		the probability of ending in a state.

		For example, if you wanted a model with two states, A and B, and a 0.5
		probability of switching to the other state, 0.4 probability of staying
		in the same state, and 0.1 probability of ending, you'd write the HMM
		like this:

		matrix = [ [ 0.4, 0.5 ], [ 0.4, 0.5 ] ]
		distributions = [NormalDistribution(1, .5), NormalDistribution(5, 2)]
		starts = [ 1., 0. ]
		ends = [ .1., .1 ]
		state_names= [ "A", "B" ]

		model = Model.from_matrix( matrix, distributions, starts, ends, 
			state_names, name="test_model" )
		"""

		# Build the initial model
		model = Model( name=name )

		# Build state objects for every state with the appropriate distribution
		states = [ State( distribution, name=name ) for name, distribution in
			izip( state_names, distributions) ]

		n = len( states )

		# Add all the states to the model
		for state in states:
			model.add_state( state )

		# Connect the start of the model to the appropriate state
		for i, prob in enumerate( starts ):
			if prob != 0:
				model.add_transition( model.start, states[i], prob )

		# Connect all states to each other if they have a non-zero probability
		for i in xrange( n ):
			for j, prob in enumerate( transition_probabilities[i] ):
				if prob != 0.:
					model.add_transition( states[i], states[j], prob )

		# Connect states to the end of the model if a non-zero probability 
		for i, prob in enumerate( ends ):
			if prob != 0:
				model.add_transition( states[j], model.end, prob )

		model.bake()
		return model

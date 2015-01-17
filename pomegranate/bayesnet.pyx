# bayesnet.pyx
# Contact: Jacob Schreiber ( jmschreiber91@gmail.com )

cimport cython
from cython.view cimport array as cvarray
from libc.math cimport log as clog, sqrt as csqrt, exp as cexp
import math, random, itertools as it, sys, bisect
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

def merge_marginals( marginals ):
	'''
	Merge multiple marginals of the same distribution to form a more informed
	distribution.
	'''

	probabilities = { key: _log( value ) for key, value in marginals[0].parameters[0].items() }

	for marginal in marginals[1:]:
		for key, value in marginal.parameters[0].items():
			probabilities[key] += _log( value )

	total = NEGINF
	for key, value in probabilities.items():
		total = pair_lse( total, probabilities[key] )

	for key, value in probabilities.items():
		probabilities[key] = cexp( value - total )

	return DiscreteDistribution( probabilities )

cdef class BayesianNetwork( Model ):
	"""
	Represents a Bayesian Network
	"""

	def add_transition( self, a, b ):
		"""
		Add a transition from state a to state b which indicates that B is
		dependent on A in ways specified by the distribution. 
		"""

		# Add the transition
		self.graph.add_edge(a, b )

	def add_transitions( self, a, b ):
		"""
		Add multiple conditional dependencies at the same time.
		"""

		n = len(a) if isinstance( a, list ) else len(b)

		# Allow addition of many transitions from many states
		if isinstance( a, list ) and isinstance( b, list ):
			# Set up an iterator across all edges
			for start, end in izip( a, b ):
				self.add_transition( start, end )

		# Allow for multiple transitions to a specific state 
		elif isinstance( a, list ) and isinstance( b, State ):
			# Set up an iterator across all edges to b
			for start in a:
				self.add_transition( start, b )

		# Allow for multiple transitions from a specific state
		elif isinstance( a, State ) and isinstance( b, list ):
			# Set up an iterator across all edges from a
			for end in b:
				self.add_transition( a, end )

	def bake( self, verbose=False ): 
		"""
		Finalize the topology of the model, and assign a numerical index to
		every node. This method must be called before any of the probability-
		calculating or sampling methods.
		
		This fills in self.states (a list of all states in order), the sparse
		matrices of transitions and their weights, and also will merge silent
		states.
		"""

		# Go through the model and delete any nodes which have no edges leading
		# to it, or edges leading out of it. This gets rid of any states with
		# no edges in or out, as well as recursively removing any chains which
		# are impossible for the viterbi path to touch.
		self.in_edge_count = numpy.zeros( len( self.graph.nodes() ), 
			dtype=numpy.int32 ) 
		self.out_edge_count = numpy.zeros( len( self.graph.nodes() ), 
			dtype=numpy.int32 )

		# Go through all edges which exist looking for silent states to merge
		while True:
			# Set the number of merged states to 0
			merged = 0

			for a, b in self.graph.edges():

				# If the receiver is a silent state, then it is a placeholder 
				if b.is_silent():

					# Go through all edges again looking for edges which begin with
					# this silent placeholder
					for c, d in self.graph.edges():

						# If we find two pairs where the middle node is both
						# the same, and silent, then we need to merge the
						# leftmost and rightmost nodes and remove the middle,
						# silent one.
						# A -> B/C -> D becomes A -> D.
						# If A is silent as well, it will be merged out next
						# iteration.
						if b is c:
							pd = d.distribution.parameters[1]
							
							# Go through all parent distributions for this CD
							# to find the 'None' which we are replacing
							for i, parent in enumerate( pd ):
								if parent is None:
									pd[i] = a.distribution

									if verbose:
										print( "{} and {} merged, removing state {}"\
											.format( a.name, d.name, c.name ) )
									break

							# If no 'Nones' are in the conditional distribution
							# then we don't know which to replace
							else:
								raise SyntaxError( "Uncertainty in which parent {}\
									should replace for {}".format( a.name, d.name ))

							# Add an edge directly from A to D, removing the
							# middle silent state
							self.graph.add_edge( a, d )
							self.graph.remove_node( b )
							merged += 1

			if merged == 0:
				break

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



	def log_probability( self, data ):
		'''
		Determine the log probability of the data given the model. The data is
		expected to come in as a dictionary, with the indexes being the names
		of the states, and the keys being the value the data takes at that state.
		'''

		names = { state.name: state.distribution for state in self.states }
		data = { names[state]: value for state, value in data.items() }
		log_probability = 0
 
		for state in self.states:
			d = state.distribution

			# If the distribution is conditional on other distributions, then
			# pass the full dictionary of values in
			if isinstance( d, ConditionalDistribution ):
				log_probability += d.log_probability( data[d], data )

			# Otherwise calculate the log probability of this value
			# independently of the other values.
			else:
				log_probability += d.log_probability( data[d] )

		return log_probability

	def forward( self, data={} ):
		'''
		Propogate messages forward through the network from observed data to
		distributions which depend on that data. This is not the full belief
		propogation algorithm.
		'''

		# Go from state names:data to distribution object:data
		names = { state.name: state.distribution for state in self.states }
		data = { names[state]: value for state, value in data.items() }

		# List of factors
		factors = [ data[ s.distribution ] if s.distribution in data else None for s in self.states ]

		# Unpack the edges
		in_edges = numpy.array( self.in_edge_count )
		out_edges = numpy.array( self.out_edge_count )

		# Figure out the roots of the graph, meaning they're independent of the
		# remainder of the graph and have been visited
		roots = numpy.where( in_edges[1:] - in_edges[:-1] == 0 )[0]
		visited = numpy.zeros( len( self.states ) )
		for i, state in enumerate( self.states ):
			if state.distribution in data.keys():
				visited[i] = 1

		# For each of your roots, unpack observed data or use the prior
		for root in roots:
			visited[ root ] = 1
			if factors[ root ] is not None:
				continue

			state = self.states[ root ]
			d = state.distribution

			if state.name in data:
				factors[ root ] = data[ d ]
			else:
				factors[ root ] = d

		# Go through all of the states and 
		while True:
			for i, state in enumerate( self.states ):
				if visited[ i ] == 1:
					continue

				state = self.states[ i ]
				d = state.distribution

				for k in xrange( in_edges[i], in_edges[i+1] ):
					ki = self.in_transitions[k]

					if visited[ki] == 0:
						break
				else:
					parents = {}
					for k in xrange( in_edges[i], in_edges[i+1] ):
						ki = self.in_transitions[k]
						d = self.states[ki].distribution

						parents[d] = factors[ki]

					factors[i] = state.distribution.marginal( parents )
					visited[i] = 1

			if visited.sum() == visited.shape[0]:
				break

		return factors

	def backward( self, data={} ):
		'''
		Propogate messages backwards through the network from observed data to
		distributions which that data depends on. This is not the full belief
		propogation algorithm.
		'''

		# Go from state names:data to distribution object:data
		names = { state.name: state.distribution for state in self.states }
		data = { names[state]: value for state, value in data.items() }

		# List of factors
		factors = [ data[ s.distribution ] if s.distribution in data else s.distribution.marginal() for s in self.states ]
		new_factors = [ i for i in factors ]

		# Unpack the edges
		in_edges = numpy.array( self.in_edge_count )
		out_edges = numpy.array( self.out_edge_count )

		# Record the message passed along each edge
		messages = [ None for i in in_edges ]

		# Figure out the leaves of the graph, which are independent of the other
		# nodes using the backwards algorithm, and say we've visited them.
		leaves = numpy.where( out_edges[1:] - out_edges[:-1] == 0 )[0]
		visited = numpy.zeros( len( self.states ) )
		visited[leaves] = 1
		for i, s in enumerate( self.states ):
			if s.distribution in data and not isinstance( data[ s.distribution ], Distribution ): 
				visited[i] = 1 

		# Go through the nodes we haven't yet visited and update their beliefs
		# iteratively if we've seen all the data which depends on it. 
		while True:
			for i, state in enumerate( self.states ):
				# If we've already visited the state, then don't visit
				# it again.
				if visited[i] == 1:
					continue

				# Unpack the state and the distribution
				state = self.states[i]
				d = state.distribution

				# Make sure we've seen all the distributions which depend on
				# this one, otherwise break.
				for k in xrange( out_edges[i], out_edges[i+1] ):
					ki = self.out_transitions[k]
					if visited[ki] == 0:
						break
				else:
					for k in xrange( out_edges[i], out_edges[i+1] ):
						ki = self.out_transitions[k]

						# Update the parent information
						parents = {}
						for l in xrange( in_edges[ki], in_edges[ki+1] ):
							li = self.in_transitions[l]
							parents[ self.states[li].distribution ] = factors[li]

						# Get the messages for each of those states
						messages[k] = self.states[ki].distribution.marginal( parents, wrt=d, value=new_factors[ki] )
					else:
						# Find the local messages which influence these
						local_messages = [ factors[i] ] + [ messages[k] for k in xrange( out_edges[i], out_edges[i+1] ) ]
						
						# Merge marginals of each of these, and the prior information
						new_factors[i] = merge_marginals( local_messages )

					# Mark that we've visited this state.
					visited[i] = 1

			# If we've visited all states, we're done
			if visited.sum() == visited.shape[0]:
				break

		return new_factors 


	def forward_backward( self, data={} ):
		'''
		Propogate messages forward through the network to update beliefs in
		each state, then backwards from those beliefs to the remainder of
		the network. This is the sum-product belief propogation algorithm.
		'''

		factors = self.forward( data )
		data = { self.states[i].name: factors[i] for i in xrange( len(factors) ) }
		return self.backward( data )
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

def merge_marginals( multiply, divide=[] ):
	'''
	Merge multiple marginals of the same distribution to form a more informed
	distribution.
	'''

	if len(multiply) == 1:
		return multiply[0]

	probabilities = { key: _log( value ) for key, value in multiply[0].parameters[0].items() }

	for marginal in multiply[1:]:
		if marginal == 1:
			continue
		for key, value in marginal.parameters[0].items():
			probabilities[key] += _log( value )

	for marginal in divide:
		if marginal == 1:
			continue
		for key, value in marginal.parameters[0].items():
			probabilities[key] -= _log( value )

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

	def add_dependency( self, a, b ):
		"""
		Add a transition from state a to state b which indicates that B is
		dependent on A in ways specified by the distribution. 
		"""

		# Add the transition
		self.graph.add_edge(a, b )

	def add_dependencies( self, a, b ):
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

	def marginal( self ):
		"""
		Return the marginal of the graph. This is equivilant to a pass of
		belief propogation given that no data is given; or a single forward
		pass of the sum-product algorithm.
		"""

		return self.forward()

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
		distributions which depend on that data. This is not the full sum
		product algorithm.
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
		for i, s in enumerate( self.states ):
			d = s.distribution
			if d in data and not isinstance( data[ d ], Distribution ): 
				visited[i] = 1 

		# Go through all of the states and 
		while True:
			for i, state in enumerate( self.states ):
				if visited[ i ] == 1:
					continue

				state = self.states[i]

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
		factors = [ data[ s.distribution ] if s.distribution in data else None for s in self.states ]
		new_factors = [ i for i in factors ]

		# Unpack the edges
		in_edges = numpy.array( self.in_edge_count )
		out_edges = numpy.array( self.out_edge_count )

		# Figure out the leaves of the graph, which are independent of the other
		# nodes using the backwards algorithm, and say we've visited them.
		leaves = numpy.where( out_edges[1:] - out_edges[:-1] == 0 )[0]

		visited = numpy.zeros( len( self.states ) )
		visited[leaves] = 1
		for i, s in enumerate( self.states ):
			d = s.distribution
			if d in data and not isinstance( data[ d ], Distribution ): 
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
					local_messages = [ factors[i] ]
					for k in xrange( out_edges[i], out_edges[i+1] ):
						ki = self.out_transitions[k]
						# The message being passed back from this state needs
						# to be weighted by all other parents of this child
						# according to the sum-product algorithm.
						if self.states[ki].distribution in data:
							parents = {}
							for l in xrange( in_edges[ki], in_edges[ki+1] ):
								li = self.in_transitions[l]

								dli = self.states[li].distribution
								parents[dli] = factors[li]

							# Get the weighted message from the state, which is the
							# marginal with respect to state we're trying to update
							# weighted by the marginal of the other parent
							# distributions.
							factor = self.states[ki].distribution.marginal( parents, wrt=d, value=new_factors[ki] )
							local_messages.append( factor )
						else:
							local_messages.append( 1 )
						
					else:
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

		for i in xrange( 3 ):
			factors = self.forward( data )
			data = { self.states[i].name: factors[i] for i in xrange( len(factors) ) }
			factors = self.backward( data )	
			data = { self.states[i].name: factors[i] for i in xrange( len(factors) ) }

		return factors

	def loopy_belief_propogation( self, data={} ):
		'''
		Propogate messages through the network using a flooding schedule, where
		messages are sent from nodes who get updates.
		'''

		# Go from state names:data to distribution object:data
		names = { state.name: state.distribution for state in self.states }
		data = { names[state]: value for state, value in data.items() }

		# List of factors after each pass
		backward_factors = [ data[ s.distribution ] if s.distribution in data else None for s in self.states ]
		forward_factors  = [ data[ s.distribution ] if s.distribution in data else None for s in self.states ] 

		# Unpack the edges
		in_edges = numpy.array( self.in_edge_count )
		out_edges = numpy.array( self.out_edge_count )

		# Figure out the roots of the graph, meaning they're independent of the
		# remainder of the graph and have been visited
		roots = numpy.where( in_edges[1:] - in_edges[:-1] == 0 )[0]

		# Figure out the leaves of the graph, which are independent of the other
		# nodes using the backwards algorithm, and say we've visited them.
		leaves = numpy.where( out_edges[1:] - out_edges[:-1] == 0 )[0]

		for i in roots:
			forward_factors[i] = self.states[i].distribution.marginal()

		for i in xrange( 10 ):
			# First do a forward pass, where all nodes which are highlighted to
			# send messages to their children
			visited = numpy.zeros( len( self.states ) )
			visited[roots] = 1
			for i, s in enumerate( self.states ):
				d = s.distribution
				if d in data and not isinstance( data[ d ], Distribution ): 
					visited[i] = 1 

			while visited.sum() != visited.shape[0]:
				for i, state in enumerate( self.states ):
					if visited[ i ] == 1:
						continue

					state = self.states[i]

					for k in xrange( in_edges[i], in_edges[i+1] ):
						ki = self.in_transitions[k]

						if visited[ki] == 0:
							break
					else:
						parents = {}
						for k in xrange( in_edges[i], in_edges[i+1] ):
							ki = self.in_transitions[k]
							d = self.states[ki].distribution

							parents[d] = forward_factors[ki]

						forward_factors[i] = state.distribution.marginal( parents )
						visited[i] = 1


			print
			print forward_factors
			print backward_factors
			print

			# Now do a backward pass, where nodes are highlighted to send
			# messages to their children
			visited = numpy.zeros( len( self.states ) )
			visited[leaves] = 1
			for i, s in enumerate( self.states ):
				d = s.distribution
				if d in data and not isinstance( data[ d ], Distribution ): 
					visited[i] = 1 

			# Go through the nodes we haven't yet visited and update their beliefs
			# iteratively if we've seen all the data which depends on it. 
			while visited.sum() != visited.shape[0]:
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
						local_messages = [ forward_factors[i] ]
						for k in xrange( out_edges[i], out_edges[i+1] ):
							ki = self.out_transitions[k]
							# The message being passed back from this state needs
							# to be weighted by all other parents of this child
							# according to the sum-product algorithm.
							parents = {}
							for l in xrange( in_edges[ki], in_edges[ki+1] ):
								li = self.in_transitions[l]

								dli = self.states[li].distribution
								parents[dli] = forward_factors[li]

							# Get the weighted message from the state, which is the
							# marginal with respect to state we're trying to update
							# weighted by the marginal of the other parent
							# distributions.
							factor = self.states[ki].distribution.marginal( parents, wrt=d, value=backward_factors[ki] )
							local_messages.append( factor )
							
						else:
							backward_factors[i] = merge_marginals( local_messages )

						# Mark that we've visited this state.
						visited[i] = 1

			print "derp"
			print backward_factors
			print "hello"
			print 

			forward_factors = backward_factors[:]

		return forward_factors 



	@classmethod
	def from_bif( self, filename ):
		'''
		Create a Bayesian network from a BIF file.
		'''

		# Initialize some useful variables
		distributions, states, model = {}, {}, None
		d_objects = {}
		v_read, p_read = False, False

		with open( filename, 'r' ) as infile: 
			for line in infile:
				# Get the name of the model
				if line.startswith( 'network' ):
					model_name = line.split()[1]
					model = BayesianNetwork( model_name )

				# Get the name of a variable, which is a distribution
				elif line.startswith( 'variable' ):
					# Indicate we are currently reading a variable
					v_read = True
					
					# Read in the name of the state
					d_name = line.split()[1].strip()

					# Initialize the distribution
					distribution = []
					distributions[d_name] = distribution

				# Read the domain of a variable
				elif line.strip().startswith('type') and v_read:
					# Read the domain from the file
					domain = line.split("{")[1][:-2].strip().split(',')

					# Encode it in the distribution dictionary
					for value in domain:
						distributions[ d_name ].append( value.replace('}', '').strip() )

					# Stop reading the variable
					v_read = False
					d_name = None

				# Begin reading a probability table
				elif line.startswith( 'probability' ):
					# Indicate that we're reading a probability table now
					p_read = True

					joint = line.split('(')[1].split(')')[0]
					cpt = None

					# If this is a marginal it is easy to create the distribution
					if '|' not in joint:
						d_name = joint.strip()
						c_deps = None
					else:
						d_name = joint.split('|')[0].strip()
						c_deps = [ c.strip() for c in joint.split('|')[1].split(',') ]

						if len(c_deps) == 1:
							cpt = { i : {} for i in distributions[c_deps[0].strip()] }
						else:
							cpt = {}
							for vals in it.product( *[ distributions[dep] for dep in c_deps ] ):
								d = cpt
								for val in vals:
									if val not in d:
										d[val] = {}
									d = d[val]

				# If we're reading a marginal table for a root variable, this
				# case is easy
				elif line.strip().startswith( 'table' ):
					# Strip out the non-probability elements
					probs = line.replace(';', '').replace('table ', '' )

					# Encode the distribution and add it to the network
					probs = map( float, probs.split(',') )

					d = { name: prob for name, prob in zip( distributions[d_name], probs ) }
					d = DiscreteDistribution( d )
					d_objects[d_name] = d 

					# Add this to the states in the model
					s = State( d, name=d_name )
					model.add_state( s )
					states[d] = s
					
					# End this variable reading
					p_read, d_name, c_deps = None, None, None 

				# Now the hard case; reading in a full CPT
				elif line.strip().startswith('(') and p_read:
					# Strip out the parenthesis
					values, probs = line.replace('(', '').replace(';', '').split(')')
					values = [ v.strip() for v in values.split(',') ]
					probs = map( float, probs.split(',') )

					# Initialize a new CPT or use a growing CPT if not in the
					# first line of the growing CPT 
					d = cpt
					for dep, value in zip( c_deps, values ):
						d = d[value]

						if dep == c_deps[-1]:
							for val, prob in zip( distributions[d_name], probs ):
								d[val] = prob

				# If we're done reading a probability
				elif line.startswith( '}' ) and p_read:
					# Stop reading
					p_read = False

					# Make the final nested dictionary into DiscreteDistributions
					d = cpt
					if len(c_deps) == 1:
						for key, value in d.items():
							d[key] = DiscreteDistribution( value )
					else:
						product = it.product( *[distributions[dep] for dep in c_deps[:-1] ] )
						for values in product:
							d = cpt
							for value in values:
								d = d[value]
							for key, value in d.items():
								d[key] = DiscreteDistribution( value )

					# Make the conditional object 
					dependencies = [ d_objects[dep] if dep in d_objects.keys() else dep for dep in c_deps ]
					dist = ConditionalDiscreteDistribution( cpt, dependencies )
					d_objects[d_name] = dist
					
					# Make the state object and add it to the graph
					s = State( dist, d_name )
					model.add_state( s )
					states[dist] = s 

					p_read, d_name, c_dep = None, None, None

		# Go through each distribution and fill in parents which were undetermined
		# at the first pass, and also add the dependencies.
		for name, distribution in d_objects.items():

			# If this distribution is dependent on others, make sure we fill in
			# the parents properly and add the directed edges on the graph
			if isinstance( distribution, ConditionalDiscreteDistribution ):
				# Pull out the parents
				parents = distribution.parameters[1]
				s = states[distribution]

				# Go through the parents and correct them and add the edges
				for i, parent in enumerate( parents ):
					# If we have a string, we need to correct them
					if isinstance( parent, str ):
						distribution.parameters[1][i] = d_objects[parent]
						ps = states[ d_objects[parent] ]
					else:
						ps = states[ parent ]
					
					model.add_dependency( ps, s )

		model.bake()
		return model
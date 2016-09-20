# FactorGraph.pyx
# Contact: Jacob Schreiber ( jmschreiber91@gmail.com )

cimport numpy
import numpy
import sys

try:
	import pygraphviz
	import tempfile
	import matplotlib
	import matplotlib.pyplot as plt
except:
	pygraphviz = None

from .base cimport GraphModel
from .base cimport Model
from .base cimport State
from .distributions cimport Distribution
from .distributions cimport MultivariateDistribution

if sys.version_info[0] > 2:
	xrange = range

cdef class FactorGraph( GraphModel ):
	"""A Factor Graph model.

	A biparte graph where conditional probability tables are on one side,
	and marginals for each of the variables involved are on the other
	side.

	Parameters
	----------
	name : str, optional
		The name of the model. Default is None.
	"""

	cdef numpy.ndarray transitions, edge_count, marginals

	def __init__( self, name=None ):
		"""
		Make a new graphical model. Name is an optional string used to name
		the model when output. Name may not contain spaces or newlines.
		"""
		
		self.name = name or str( id(self) )
		self.states = []
		self.edges = []

	def plot( self, **kwargs ):
		"""Draw this model's graph using NetworkX and matplotlib.

		Note that this relies on networkx's built-in graphing capabilities (and
		not Graphviz) and thus can't draw self-loops.

		See networkx.draw_networkx() for the keywords you can pass in.

		Parameters
		----------
		**kwargs : any
			The arguments to pass into networkx.draw_networkx()

		Returns
		-------
		None
		"""

		if pygraphviz is not None:
			G = pygraphviz.AGraph(directed=True)

			for state in self.states:
				G.add_node(state.name, color='red')

			for parent, child in self.edges:
				G.add_edge(self.states[parent].name, self.states[child].name)

			with tempfile.NamedTemporaryFile() as tf:
				G.draw(tf.name, format='png', prog='dot')
				img = matplotlib.image.imread(tf.name)
				plt.imshow(img)
				plt.axis('off')
		else:
			raise ValueError("must have pygraphviz installed for visualization")

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

		n, m = len(self.states), len(self.edges)

		# Initialize the arrays
		self.marginals = numpy.zeros( n )

		# We need a good way to get transition probabilities by state index that
		# isn't N^2 to build or store. So we will need a reverse of the above
		# mapping. It's awkward but asymptotically fine.
		indices = { self.states[i]: i for i in xrange(n) }
		self.edges = [ ( indices[a], indices[b] ) for a, b in self.edges ]

		# We need a new array for an undirected model which will store all
		# edges involving this state. There is no direction, and so it will
		# be a single array of twice the length of the number of edges,
		# since each edge belongs to two nodes.
		self.transitions = numpy.zeros( m*2, dtype=numpy.int32 ) - 1
		self.edge_count = numpy.zeros( n+1, dtype=numpy.int32 )

		# Go through each node and classify it as either a marginal node or a
		# factor node.
		for i, node in enumerate( self.states ):
			if not isinstance( node.distribution, MultivariateDistribution ):
				self.marginals[i] = 1
			if node.name.endswith( '-joint' ):
				self.marginals[i] = 0 

		# Now we need to find a way of storing edges for a state in a manner
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
			self.edge_count[ b+1 ] += 1
			# Increment the total number of edges leaving node a.
			self.edge_count[ a+1 ] += 1

		# Take the cumulative sum so that we can associate array indices with
		# in or out transitions
		self.edge_count = numpy.cumsum( self.edge_count, dtype=numpy.int32 )

		# Now we go through the edges again in order to both fill in the
		# transition probability matrix, and also to store the indices sorted
		# by the end-node.
		for a, b in self.edges:
			# Put the edge in the dict. Its weight is log-probability
			start = self.edge_count[ b ]

			# Start at the beginning of the section marked off for node b.
			# If another node is already there, keep walking down the list
			# until you find a -1 meaning a node hasn't been put there yet.
			while self.transitions[ start ] != -1:
				if start == self.edge_count[ b+1 ]:
					break
				start += 1

			# Store transition info in an array where the edge_count shows
			# the mapping stuff.
			self.transitions[ start ] = a

			# Now do the same for out edges
			start = self.edge_count[ a ]

			while self.transitions[ start ] != -1:
				if start == self.edge_count[ a+1 ]:
					break
				start += 1

			self.transitions[ start ] = b

	def marginal( self ):
		"""Return the marginal probabilities of each variable in the graph.

		This is equivalent to a pass of belief propogation on a graph where
		no data has been given. This will calculate the probability of each
		variable being in each possible emission when nothing is known.

		Parameters
		----------
		None

		Returns
		-------
		marginals : array-like, shape (n_nodes)
			An array of univariate distribution objects showing the marginal
			probabilities of that variable.
		"""

		return self.predict_proba( {} )

	def predict_proba( self, data, max_iterations=10, verbose=False ):
		"""Returns the probabilities of each variable in the graph given evidence.

		This calculates the marginal probability distributions for each state given
		the evidence provided through loopy belief propogation. Loopy belief
		propogation is an approximate algorithm which is exact for certain graph
		structures.

		Parameters
		----------
		data : dict or array-like, shape <= n_nodes, optional
			The evidence supplied to the graph. This can either be a dictionary
			with keys being state names and values being the observed values
			(either the emissions or a distribution over the emissions) or an
			array with the values being ordered according to the nodes incorporation
			in the graph (the order fed into .add_states/add_nodes) and None for
			variables which are unknown. If nothing is fed in then calculate the
			marginal of the graph.
		max_iterations : int, optional
			The number of iterations with which to do loopy belief propogation.
			Usually requires only 1.
		check_input : bool, optional
			Check to make sure that the observed symbol is a valid symbol for that
			distribution to produce.

		Returns
		-------
		probabilities : array-like, shape (n_nodes)
			An array of univariate distribution objects showing the probabilities
			of each variable.
		"""

		n, m = len( self.states ), len( self.transitions )

		# Save our original distributions so that we don't permanently overwrite
		# them as we do belief propogation.
		distributions = numpy.empty( n, dtype=Distribution )

		# Clamp values down to evidence if we have observed them
		for i, state in enumerate( self.states ):
			if state.name in data:
				val = data[state.name]
				if isinstance( val, Distribution ):
					distributions[i] = val
				else:
					distributions[i] = state.distribution.clamp( val )
			else:
				distributions[i] = state.distribution

		# Create a buffer for each marginal node for messages coming into the
		# node and messages leaving the node.
		out_messages = numpy.empty( m, dtype=Distribution )
		in_messages = numpy.empty( m, dtype=Distribution )

		# Explicitly calculate the distributions at each round so we can test
		# for convergence. 
		prior_distributions = distributions.copy()
		current_distributions = numpy.empty( m, dtype=Distribution )

		# Go through and initialize messages from the states to be whatever
		# we set the marginal to be. For edges which are encoded as leaving
		# a marginal, set it to that marginal, otherwise follow the edge from
		# the factor to the marginal and set it to the marginal.
		for i, state in enumerate( self.states ):
			# Go through and set edges which are encoded as leaving the
			# marginal distributions as the marginal distribution
			if self.marginals[i] == 1:
				for k in xrange( self.edge_count[i], self.edge_count[i+1] ):
					out_messages[k] = distributions[i]
					in_messages[i] = distributions[i]
			# Otherwise follow the edge, then set the message to be
			# the marginal on the other side.
			else:
				for k in xrange( self.edge_count[i], self.edge_count[i+1] ):
					kl = self.transitions[k]
					out_messages[k] = distributions[kl]
					in_messages[k] = distributions[kl]

		# We're going to iterate two steps here:
		# 	(1) send messages from variable nodes to factor nodes, containing
		#   evidence and beliefs about the marginals
		#   (2) send messages from the factors to the variables, containing
		#   the factors belief about each marginal.
		# This is the flooding message schedule for loopy belief propogation.
		iteration = 0
		while iteration < max_iterations:
			# UPDATE MESSAGES LEAVING THE MARGINAL NODES 
			for i, state in enumerate( self.states ):
				# Ignore factor nodes for now
				if self.marginals[i] == 0 or iteration == 0:
					continue

				# We are trying to calculate a new message for each edge leaving
				# this marginal node. So we start by looping over each edge, and
				# for each edge loop over all other edges and multiply the factors
				# together.
				for k in xrange( self.edge_count[i], self.edge_count[i+1] ):
					ki = self.transitions[k]
					# Start off by weighting by the distribution at this factor--
					# keep in mind that this is a uniform distribution unless evidence
					# is provided by the user, at which point it is clamped to a
					# specific value, acting as a filter.
					message = distributions[i]

					for l in xrange( self.edge_count[i], self.edge_count[i+1] ):
						# Don't include the previous message received from here
						if k == l:
							continue

						# Update the out message by multiplying the factors
						# together.
						message *= in_messages[l]

					for l in xrange( self.edge_count[ki], self.edge_count[ki+1] ):
						li = self.transitions[l]

						if li == i:
							out_messages[l] = message

			# We have now updated all of the messages leaving the marginal node,
			# now we have to update all the messages going to the marginal node.
			for i, state in enumerate( self.states ):
				# Now we ignore the marginal nodes
				if self.marginals[i] == 1:
					continue

				# We need to calculate the new in messages for the marginals.
				# This involves taking in all messages from all edges except the
				# message from the marginal we are trying to send a message to.
				for k in xrange( self.edge_count[i], self.edge_count[i+1] ):
					ki = self.transitions[k]

					# We can simply calculate this by turning the CPT into a
					# joint probability table using the other messages, and
					# then summing out those variables.
					d = {}
					for l in xrange( self.edge_count[i], self.edge_count[i+1] ):
						# Don't take in messages from the marginal we are trying
						# to send a message to.
						if k == l:
							continue

						li = self.transitions[l]
						d[ self.states[li].distribution ] = out_messages[l]

					for l in xrange( self.edge_count[ki], self.edge_count[ki+1] ):
						li = self.transitions[l]

						if li == i:
							in_messages[l] = state.distribution.marginal( neighbor_values=d )

			# Calculate the current estimates on the marginals to compare to the
			# last iteration, so that we can stop if we reach convergence.
			done = 1
			for i in xrange( len( self.states ) ):
				if self.marginals[i] == 0:
					continue

				current_distributions[i] = distributions[i]
				# Multiply the factors together by the original marginal to
				# calculate the new estimate of the marginal
				for k in xrange( self.edge_count[i], self.edge_count[i+1] ):
					current_distributions[i] *= in_messages[k]

				if not current_distributions[i].equals( prior_distributions[i] ):
					done = 0

			# If we have converged, then we're done!
			if done == 1:
				break

			# Set this list of distributions to the prior observations of the
			# marginals
			prior_distributions = current_distributions.copy()

			# Increment our iteration calculator
			iteration += 1
			
		# We've already computed the current belief about the marginals, so
		# we can just return that.
		return current_distributions[ numpy.where( self.marginals == 1 ) ]

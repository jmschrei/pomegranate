# BayesianNetwork.pyx
# Contact: Jacob Schreiber ( jmschreiber91@gmail.com )

import numpy

from .base cimport Model
from .base cimport State
from .distributions cimport DiscreteDistribution
from .distributions cimport ConditionalProbabilityTable
from .distributions cimport JointProbabilityTable
from .FactorGraph import FactorGraph
from .utils cimport _log

cdef class BayesianNetwork( Model ):
	"""
	Represents a Bayesian Network
	"""

	def bake( self, verbose=False ): 
		"""
		The Bayesian Network is going to be mostly a wrapper for the Factor
		Graph, as probabilities, inference, and training can be done more
		efficiently on them.
		"""

		# Initialize the factor graph
		self.graph = FactorGraph( self.name+'-fg' )

		# Build the factor graph structure we'll use for all the important
		# calculations.
		j = 0

		# Create two mappings, where edges which previously went to a
		# conditional distribution now go to a factor, and those which left
		# a conditional distribution now go to a marginal
		f_mapping, m_mapping = {}, {}
		d_mapping = {}
		fa_mapping = {}

		# Go through each state and add in the state if it is a marginal
		# distribution, otherwise add in the appropriate marginal and
		# conditional distribution as separate nodes.
		for i, state in enumerate( self.states ):
			# For every state (ones with conditional distributions or those
			# encoding marginals) we need to create a marginal node in the
			# underlying factor graph. 
			keys = state.distribution.keys()
			d = DiscreteDistribution({ key: 1./len(keys) for key in keys })
			m = State( d, state.name )

			# Add the state to the factor graph
			self.graph.add_node( m )

			# Now we need to copy the distribution from the node into the
			# factor node. This could be the conditional table, or the
			# marginal.
			f = State( state.distribution.copy(), state.name+'-joint' )

			if isinstance( state.distribution, ConditionalProbabilityTable ):
				fa_mapping[f.distribution] = m.distribution 

			self.graph.add_node( f )
			self.graph.add_edge( m, f )

			f_mapping[state] = f
			m_mapping[state] = m
			d_mapping[state.distribution] = d

			# Progress the counter by one
			j += 1

		for a, b in self.edges:
			self.graph.add_edge( m_mapping[a], f_mapping[b] )

		# Now go back and redirect parent pointers to the appropriate
		# objects.
		for state in self.graph.states:
			d = state.distribution
			if isinstance( d, ConditionalProbabilityTable ):
				dist = fa_mapping[d]
				d.parameters[1] = [ d_mapping[parent] for parent in d.parameters[1] ]
				state.distribution = d.joint()
				state.distribution.parameters[1].append( dist )

		# Finalize the factor graph structure
		self.graph.bake()

	def log_probability( self, sample ):
		"""
		Return the log probability of the sample under the Bayesian network.
		Currently only supports samples which are fully observed.
		"""

		indices = { state.distribution: i for i, state in enumerate( self.states ) }
		logp = 0.0

		# Go through each state and pass in the appropriate data for the
		# update to the states
		for i, state in enumerate( self.states ):
			if isinstance( state.distribution, ConditionalProbabilityTable ):
				idx = [ indices[ dist ] for dist in state.distribution.parameters[1] ] + [i]
				data = tuple( sample[i] for i in idx )
				logp += state.distribution.log_probability( data )
			else:
				logp += state.distribution.log_probability( sample[i] )
		
		return logp

	def marginal( self ):
		"""
		Return the marginal of the graph. This is equivilant to a pass of
		belief propogation given that no data is given; or a single forward
		pass of the sum-product algorithm.
		"""

		return self.graph.marginal()

	def predict_proba( self, data={}, max_iterations=100, check_input=True ):
		"""sklearn wrapper for loopy belief propogation."""

		return self.forward_backward( data, max_iterations, check_input )

	def forward_backward( self, data={}, max_iterations=100, check_input=True ):
		"""
		Run loopy belief propogation on the underlying factor graph until
		convergence, and return the marginals.
		"""

		if check_input:
			indices = { state.name: state.distribution for state in self.states }

			for key, value in data.items():
				if value not in indices[key].keys() and not isinstance( value, DiscreteDistribution ):
					raise ValueError( "State '{}' does not have key '{}'".format( key, value ) )

		return self.graph.forward_backward( data, max_iterations )

	def from_sample( self, items, weights=None, inertia=0.0 ):
		"""Another name for the train method."""

		self.train( items, weights, inertia )

	def fit( self, items, weights=None, inertia=0.0 ):
		"""sklearn wrapper for the train method."""

		self.train( items, weights, inertia )

	def train( self, items, weights=None, inertia=0.0 ):
		"""
		Take in data, with each column corresponding to observations of the
		state, as ordered by self.states.
		"""

		indices = { state.distribution: i for i, state in enumerate( self.states ) }

		# Go through each state and pass in the appropriate data for the
		# update to the states
		for i, state in enumerate( self.states ):
			if isinstance( state.distribution, ConditionalProbabilityTable ):
				idx = [ indices[ dist ] for dist in state.distribution.parameters[1] ] + [i]
				data = [ [ item[i] for i in idx ] for item in items ]
				state.distribution.from_sample( data, weights, inertia )
			else:
				state.distribution.from_sample( [ item[i] for item in items ], weights, inertia )

		self.bake()

	def impute( self, items, max_iterations=100 ):
		"""
		Take in a matrix of data and impute all nan values to their MLE
		values. 
		"""

		for i in range( len(items) ):
			obs = {}

			for j, state in enumerate( self.states ):
				item = items[i][j]

				if item not in (None, 'nan'):
					try:
						if not numpy.isnan(item):
							obs[ state.name ] = item
					except:
						obs[ state.name ] = item

			imputation = self.forward_backward( obs  )

			for j in range( len( self.states) ):
				items[i][j] = imputation[j].mle()

		return items 

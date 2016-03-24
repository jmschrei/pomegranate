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
	"""A Bayesian Network Model.

	A Bayesian network is a directed graph where nodes represent variables, edges
	represent conditional dependencies of the children on their parents, and the
	lack of an edge represents a conditional independence. 

	Parameters
	----------
	name : str, optional
		The name of the model. Default is None

	Attributes
	----------
	states : list, shape (n_states,)
		A list of all the state objects in the model

	graph : networkx.DiGraph
		The underlying graph object.

	Example
	-------
	>>> from pomegranate import *
	>>> d1 = DiscreteDistribution({'A': 0.2, 'B': 0.8})
	>>> d2 = ConditionalProbabilityTable([['A', 'A', 0.1],
												 ['A', 'B', 0.9],
												 ['B', 'A', 0.4],
												 ['B', 'B', 0.6]], [d1])
	>>> s1 = Node( d1, name="s1" )
	>>> s2 = Node( d2, name="s2" )
	>>> model = BayesianNetwork()
	>>> model.add_nodes([s1, s2])
	>>> model.add_edge(s1, s2)
	>>> model.bake()
	>>> print model.log_probability(['A', 'B'])
	-1.71479842809
	>>> print model.predict_proba({'s2' : 'A'})
	array([ {
	    "frozen" :false,
	    "class" :"Distribution",
	    "parameters" :[
	        {
	            "A" :0.05882352941176471,
	            "B" :0.9411764705882353
	        }
	    ],
	    "name" :"DiscreteDistribution"
	},
	       {
	    "frozen" :false,
	    "class" :"Distribution",
	    "parameters" :[
	        {
	            "A" :1.0,
	            "B" :0.0
	        }
	    ],
	    "name" :"DiscreteDistribution"
	}], dtype=object)
	>>> print model.impute([[None, 'A']])
	[['B', 'A']]
	"""

	def bake( self ): 
		"""Finalize the topology of the model.

		Assign a numerical index to every state and create the underlying arrays
		corresponding to the states and edges between the states. This method 
		must be called before any of the probability-calculating methods. This
		includes converting conditional probability tables into joint probability
		tables and creating a list of both marginal and table nodes.

		Parameters
		----------
		None

		Returns
		-------
		None
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
		"""Return the log probability of a sample under the Bayesian network model.
		
		The log probability is just the sum of the log probabilities under each of
		the components. The log probability of a sample under the graph A -> B is 
		just P(A)*P(B|A).

		Parameters
		----------
		sample : array-like, shape (n_nodes)
			The sample is a vector of points where each dimension represents the
			same variable as added to the graph originally. It doesn't matter what
			the connections between these variables are, just that they are all
			ordered the same.

		Returns
		-------
		logp : double
			The log probability of that sample.
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

		return self.graph.marginal()

	def predict_proba( self, data={}, max_iterations=100, check_input=True ):
		"""Returns the probabilities of each variable in the graph given evidence.

		This calculates the marginal probability distributions for each state given
		the evidence provided through loopy belief propogation. Loopy belief
		propogation is an approximate algorithm which is exact for certain graph
		structures.

		This is a sklearn wrapper for the forward_backward method.

		Parameters
		----------
		data : dict or array-like, shape <= n_nodes, optional
			The evidence supplied to the graph. This can either be a dictionary
			with keys being state names and values being the observed values
			(either the emissions or a distribution over the emissions) or an
			array with the values being ordered according to the nodes incorporation
			in the graph (the order fed into .add_states/add_nodes) and None for
			variables which are unknown. If nothing is fed in then calculate the
			marginal of the graph. Default is {}.

		max_iterations : int, optional
			The number of iterations with which to do loopy belief propogation.
			Usually requires only 1. Default is 100.

		check_input : bool, optional
			Check to make sure that the observed symbol is a valid symbol for that
			distribution to produce. Default is True.

		Returns
		-------
		probabilities : array-like, shape (n_nodes)
			An array of univariate distribution objects showing the probabilities
			of each variable.
		"""

		return self.forward_backward( data, max_iterations, check_input )

	def forward_backward( self, data={}, max_iterations=100, check_input=True ):
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
			marginal of the graph. Default is {}.

		max_iterations : int, optional
			The number of iterations with which to do loopy belief propogation.
			Usually requires only 1. Default is 100.

		check_input : bool, optional
			Check to make sure that the observed symbol is a valid symbol for that
			distribution to produce. Default is True.

		Returns
		-------
		probabilities : array-like, shape (n_nodes)
			An array of univariate distribution objects showing the probabilities
			of each variable.
		"""

		if check_input:
			indices = { state.name: state.distribution for state in self.states }

			for key, value in data.items():
				if value not in indices[key].keys() and not isinstance( value, DiscreteDistribution ):
					raise ValueError( "State '{}' does not have key '{}'".format( key, value ) )

		return self.graph.forward_backward( data, max_iterations )

	def fit( self, items, weights=None, inertia=0.0 ):
		"""Fit the model to data using MLE estimates.

		Fit the model to the data by updating each of the components of the model,
		which are univariate or multivariate distributions. This uses a simple
		MLE estimate to update the distributions according to their summarize or
		fit methods.

		sklearn wrapper for the train method.

		Parameters
		----------
		items : array-like, shape (n_samples, n_nodes)
			The data to train on, where each row is a sample and each column
			corresponds to the associated variable.

		weights : array-like, shape (n_nodes), optional
			The weight of each sample as a positive double. Default is None.

		inertia : double, optional
			The inertia for updating the distributions, passed along to the
			distribution method. Default is 0.0.

		Returns
		-------
		self : object
			The fit Bayesian network object.
		"""

		indices = { state.distribution: i for i, state in enumerate( self.states ) }

		# Go through each state and pass in the appropriate data for the
		# update to the states
		for i, state in enumerate( self.states ):
			if isinstance( state.distribution, ConditionalProbabilityTable ):
				idx = [ indices[ dist ] for dist in state.distribution.parameters[1] ] + [i]
				data = [ [ item[i] for i in idx ] for item in items ]
				state.distribution.fit( data, weights, inertia )
			else:
				state.distribution.fit( [ item[i] for item in items ], weights, inertia )

		self.bake()
		return self

	def impute( self, items, max_iterations=100 ):
		"""Impute missing values of a data matrix using MLE.

		Impute the missing values of a data matrix using the maximally likely
		predictions according to the forward-backward algorithm. Run each
		sample through the algorithm (predict_proba) and replace missing values
		with the maximally likely predicted emission. 
		
		Parameters
		----------
		items : array-like, shape (n_samples, n_nodes)
			Data matrix to impute. Missing values must be either None (if lists)
			or np.nan (if numpy.ndarray). Will fill in these values with the
			maximally likely ones.

		max_iterations : int, optional
			Number of iterations to run loopy belief propogation for. Default
			is 100.

		Returns
		-------
		items : array-like, shape (n_samples, n_nodes)
			This is the data matrix with the missing values imputed.
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

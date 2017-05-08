# BayesianNetwork.pyx
# Contact: Jacob Schreiber ( jmschreiber91@gmail.com )

import itertools as it
import json
import time
import networkx as nx
import numpy
cimport numpy
import sys

from joblib import Parallel
from joblib import delayed

from libc.stdlib cimport calloc
from libc.stdlib cimport free
from libc.string cimport memset

from .base cimport GraphModel
from .base cimport Model
from .base cimport State
from .distributions cimport MultivariateDistribution
from .distributions cimport DiscreteDistribution
from .distributions cimport ConditionalProbabilityTable
from .distributions cimport JointProbabilityTable
from .FactorGraph import FactorGraph
from .utils cimport _log
from .utils cimport lgamma
from .utils import PriorityQueue
from .utils import plot_networkx


try:
	import tempfile
	import pygraphviz
	import matplotlib.pyplot as plt
	import matplotlib.image
except ImportError:
	pygraphviz = None

if sys.version_info[0] > 2:
	# Set up for Python 3
	xrange = range
	izip = zip
else:
	izip = it.izip

DEF INF = float("inf")
DEF NEGINF = float("-inf")

cdef class BayesianNetwork( GraphModel ):
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

	cdef list idxs
	cdef numpy.ndarray keymap
	cdef int* parent_count
	cdef int* parent_idxs
	cdef numpy.ndarray distributions
	cdef void** distributions_ptr

	@property
	def structure( self ):
		structure = [() for i in range(self.d)]
		indices = { distribution : i for i, distribution in enumerate(self.distributions) }

		for i, state in enumerate(self.states):
			d = state.distribution
			if isinstance(d, MultivariateDistribution):
				structure[i] = tuple(indices[parent] for parent in d.parents)

		return tuple(structure)

	def __dealloc__( self ):
		free(self.parent_count)
		free(self.parent_idxs)

	def plot(self, filename=None, **kwargs):
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
				G.add_edge(parent.name, child.name)

			if filename is None:
				with tempfile.NamedTemporaryFile() as tf:
					G.draw(tf.name, format='png', prog='dot')
					img = matplotlib.image.imread(tf.name)
					plt.imshow(img)
					plt.axis('off')
			else:
				G.draw(filename, format='pdf', prog='dot')

		else:
			raise ValueError("must have pygraphviz installed for visualization")

	def bake(self):
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

		self.d = len(self.states)

		# Initialize the factor graph
		self.graph = FactorGraph( self.name+'-fg' )

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

		for a, b in self.edges:
			self.graph.add_edge(m_mapping[a], f_mapping[b])

		# Now go back and redirect parent pointers to the appropriate
		# objects.
		for state in self.graph.states:
			d = state.distribution
			if isinstance( d, ConditionalProbabilityTable ):
				dist = fa_mapping[d]
				d.parents = [ d_mapping[parent] for parent in d.parents ]
				d.parameters[1] = d.parents
				state.distribution = d.joint()
				state.distribution.parameters[1].append( dist )

		# Finalize the factor graph structure
		self.graph.bake()

		indices = {state.distribution : i for i, state in enumerate(self.states)}

		n, self.idxs = 0, []
		self.keymap = numpy.array([state.distribution.keys() for state in self.states])
		for i, state in enumerate(self.states):
			d = state.distribution

			if isinstance(d, MultivariateDistribution):
				idxs = tuple(indices[parent] for parent in d.parents) + (i,)
				self.idxs.append(idxs)
				d.bake(tuple(it.product(*[self.keymap[idx] for idx in idxs])))
				n += len(idxs)
			else:
				self.idxs.append(i)
				d.bake(tuple(self.keymap[i]))
				n += 1

		self.distributions = numpy.array([state.distribution for state in self.states])
		self.distributions_ptr = <void**> self.distributions.data

		self.parent_count = <int*> calloc(self.d+1, sizeof(int))
		self.parent_idxs = <int*> calloc(n, sizeof(int))

		j = 0
		for i, state in enumerate(self.states):
			if isinstance(state.distribution, MultivariateDistribution):
				self.parent_count[i+1] = len(state.distribution.parents) + 1
				for parent in state.distribution.parents:
					self.parent_idxs[j] = indices[parent]
					j += 1

				self.parent_idxs[j] = i
				j += 1
			else:
				self.parent_count[i+1] = 1
				self.parent_idxs[j] = i
				j += 1

			if i > 0:
				self.parent_count[i+1] += self.parent_count[i]

	def log_probability(self, X):
		"""Return the log probability of samples under the Bayesian network.

		The log probability is just the sum of the log probabilities under each of
		the components. The log probability of a sample under the graph A -> B is
		just P(A)*P(B|A). This will return a vector of log probabilities, one for each
		sample.

		Parameters
		----------
		X : array-like, shape (n_samples, n_dim)
			The sample is a vector of points where each dimension represents the
			same variable as added to the graph originally. It doesn't matter what
			the connections between these variables are, just that they are all
			ordered the same.

		Returns
		-------
		logp : double
			The log probability of that sample.
		"""

		if self.d == 0:
			raise ValueError("must bake model before computing probability")

		X = numpy.array(X, ndmin=2)
		n, d = X.shape

		logp = numpy.zeros(n, dtype='float64')
		for i in range(n):
			for j, state in enumerate(self.states):
				logp[i] += state.distribution.log_probability(X[i, self.idxs[j]])

		return logp

	cdef void _log_probability( self, double* symbol, double* log_probability, int n ) nogil:
		cdef int i, j, l, li, k
		cdef double logp
		cdef double* sym = <double*> calloc(self.d, sizeof(double))
		memset(log_probability, 0, n*sizeof(double))

		for i in range(n):
			for j in range(self.d):
				memset(sym, 0, self.d*sizeof(double))
				logp = 0.0

				for l in range(self.parent_count[j], self.parent_count[j+1]):
					li = self.parent_idxs[l]
					k = l - self.parent_count[j]
					sym[k] = symbol[i*self.d + li]

				(<Model> self.distributions_ptr[j])._log_probability(sym, &logp, 1)
				log_probability[i] += logp


	def marginal(self):
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

		if self.d == 0:
			raise ValueError("must bake model before computing marginal")

		return self.graph.marginal()

	def predict_proba(self, data={}, max_iterations=100, check_input=True):
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

		if self.d == 0:
			raise ValueError("must bake model before using forward-backward algorithm")

		if check_input:
			indices = { state.name: state.distribution for state in self.states }

			for key, value in data.items():
				if value not in indices[key].keys() and not isinstance( value, DiscreteDistribution ):
					raise ValueError( "State '{}' does not have key '{}'".format( key, value ) )

		return self.graph.predict_proba( data, max_iterations )

	def summarize(self, items, weights=None):
		"""Summarize a batch of data and store the sufficient statistics.

		This will partition the dataset into columns which belong to their
		appropriate distribution. If the distribution has parents, then multiple
		columns are sent to the distribution. This relies mostly on the summarize
		function of the underlying distribution.

		Parameters
		----------
		items : array-like, shape (n_samples, n_nodes)
			The data to train on, where each row is a sample and each column
			corresponds to the associated variable.

		weights : array-like, shape (n_nodes), optional
			The weight of each sample as a positive double. Default is None.

		Returns
		-------
		None
		"""

		if self.d == 0:
			raise ValueError("must bake model before summarizing data")

		indices = { state.distribution: i for i, state in enumerate( self.states ) }

		# Go through each state and pass in the appropriate data for the
		# update to the states
		for i, state in enumerate( self.states ):
			if isinstance( state.distribution, ConditionalProbabilityTable ):
				idx = [ indices[ dist ] for dist in state.distribution.parents ] + [i]
				data = [ [ item[i] for i in idx ] for item in items ]
				state.distribution.summarize( data, weights )
			else:
				state.distribution.summarize( [ item[i] for item in items ], weights )

	def from_summaries(self, inertia=0.0, pseudocount=0.0):
		"""Use MLE on the stored sufficient statistics to train the model.

		This uses MLE estimates on the stored sufficient statistics to train
		the model.

		Parameters
		----------
		inertia : double, optional
			The inertia for updating the distributions, passed along to the
			distribution method. Default is 0.0.

		pseudocount : double, optional
			A pseudocount to add to the emission of each distribution. This
			effectively smoothes the states to prevent 0. probability symbols
			if they don't happen to occur in the data. Default is 0.

		Returns
		-------
		None
		"""

		for state in self.states:
			state.distribution.from_summaries(inertia, pseudocount)

		self.bake()

	def fit(self, items, weights=None, inertia=0.0, pseudocount=0.0):
		"""Fit the model to data using MLE estimates.

		Fit the model to the data by updating each of the components of the model,
		which are univariate or multivariate distributions. This uses a simple
		MLE estimate to update the distributions according to their summarize or
		fit methods.

		This is a wrapper for the summarize and from_summaries methods.

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

		pseudocount : double, optional
			A pseudocount to add to the emission of each distribution. This
			effectively smoothes the states to prevent 0. probability symbols
			if they don't happen to occur in the data. Only effects hidden
			Markov models defined over discrete distributions. Default is 0.

		Returns
		-------
		None
		"""

		self.summarize(items, weights)
		self.from_summaries(inertia, pseudocount)
		self.bake()

	def predict(self, items, max_iterations=100):
		"""Predict missing values of a data matrix using MLE.

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
		items : numpy.ndarray, shape (n_samples, n_nodes)
			This is the data matrix with the missing values imputed.
		"""

		if self.d == 0:
			raise ValueError("must bake model before using impute")

		imputations = numpy.copy(items)
		for i in range(len(items)):
			obs = {}

			for j, state in enumerate(self.states):
				item = items[i][j]

				if item is None:
					continue
				if isinstance(item, float) and numpy.isnan(item):
					continue
				obs[state.name] = item

			imputation = self.predict_proba(obs, max_iterations)

			for j in range(len(self.states)):
				imputations[i][j] = imputation[j].mle()

		return imputations

	def impute(self, *args, **kwargs):
		raise Warning("method 'impute' has been depricated, please use 'predict' instead")

	def to_json(self, separators=(',', ' : '), indent=4):
		"""Serialize the model to a JSON.

		Parameters
		----------
		separators : tuple, optional
			The two separaters to pass to the json.dumps function for formatting.

		indent : int, optional
			The indentation to use at each level. Passed to json.dumps for
			formatting.

		Returns
		-------
		json : str
			A properly formatted JSON object.
		"""

		states = [ state.copy() for state in self.states ]
		for state in states:
			if isinstance(state.distribution, MultivariateDistribution):
				state.distribution.parents = []

		model = {
					'class' : 'BayesianNetwork',
					'name'  : self.name,
					'structure' : self.structure,
					'states' : [ json.loads( state.to_json() ) for state in states ]
				}

		return json.dumps( model, separators=separators, indent=indent )

	@classmethod
	def from_json(cls, s):
		"""Read in a serialized Bayesian Network and return the appropriate object.

		Parameters
		----------
		s : str
			A JSON formatted string containing the file.

		Returns
		-------
		model : object
			A properly initialized and baked model.
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

		# Make a new generic Bayesian Network
		model = BayesianNetwork( str(d['name']) )


		# Load all the states from JSON formatted strings
		states = [ State.from_json( json.dumps(j) ) for j in d['states'] ]
		structure = d['structure']
		for state, parents in zip(states, structure):
			if len(parents) > 0:
				state.distribution.parents = [states[parent].distribution for parent in parents]
				state.distribution.parameters[1] = state.distribution.parents
				state.distribution.m = len(parents)

		model.add_states(*states)
		for i, parents in enumerate(structure):
			for parent in parents:
				model.add_edge(states[parent], states[i])

		model.bake()
		return model

	@classmethod
	def from_structure(cls, X, structure, weights=None, pseudocount=0.0, 
		name=None, state_names=None):
		"""Return a Bayesian network from a predefined structure.

		Pass in the structure of the network as a tuple of tuples and get a fit
		network in return. The tuple should contain n tuples, with one for each
		node in the graph. Each inner tuple should be of the parents for that
		node. For example, a three node graph where both node 0 and 1 have node
		2 as a parent would be specified as ((2,), (2,), ()).

		Parameters
		----------
		X : array-like, shape (n_samples, n_nodes)
			The data to fit the structure too, where each row is a sample and each column
			corresponds to the associated variable.

		structure : tuple of tuples
			The parents for each node in the graph. If a node has no parents,
			then do not specify any parents.

		weights : array-like, shape (n_nodes), optional
			The weight of each sample as a positive double. Default is None.

		pseudocount : double, optional
			A pseudocount to add to the emission of each distribution. This
			effectively smoothes the states to prevent 0. probability symbols
			if they don't happen to occur in the data. Default is 0.

		name : str, optional
			The name of the model. Default is None.

		state_names : array-like, shape (n_nodes), optional
			A list of meaningful names to be applied to nodes

		Returns
		-------
		model : BayesianNetwoork
			A Bayesian network with the specified structure.
		"""

		X = numpy.array(X)
		d = len(structure)

		nodes = [None for i in range(d)]

		if weights is None:
			weights_ndarray = numpy.ones(X.shape[0], dtype='float64')
		else:
			weights_ndarray = numpy.array(weights, dtype='float64')

		for i, parents in enumerate(structure):
			if len(parents) == 0:
				nodes[i] = DiscreteDistribution.from_samples(X[:,i], weights=weights_ndarray,
					pseudocount=pseudocount)

		while True:
			for i, parents in enumerate(structure):
				if nodes[i] is None:
					for parent in parents:
						if nodes[parent] is None:
							break
					else:
						nodes[i] = ConditionalProbabilityTable.from_samples(X[:,parents+(i,)],
							parents=[nodes[parent] for parent in parents],
							weights=weights_ndarray, pseudocount=pseudocount)
						break
			else:
				break

		if state_names is not None:
			states = [State(node, name=name) for node, name in izip(nodes,state_names)]
		else:
			states = [State(node, name=str(i)) for i, node in enumerate(nodes)]

		model = BayesianNetwork(name=name)
		model.add_nodes(*states)

		for i, parents in enumerate(structure):
			d = states[i].distribution

			for parent in parents:
				model.add_edge(states[parent], states[i])

		model.bake()
		return model

	@classmethod
	def from_samples(cls, X, weights=None, algorithm='greedy', max_parents=-1,
		 root=0, constraint_graph=None, pseudocount=0.0, state_names=None, name=None,
		 n_jobs=1):
		"""Learn the structure of the network from data.

		Find the structure of the network from data using a Bayesian structure
		learning score. This currently enumerates all the exponential number of
		structures and finds the best according to the score. This allows
		weights on the different samples as well. The score that is optimized
		is the minimum description length (MDL).

		Parameters
		----------
		X : array-like, shape (n_samples, n_nodes)
			The data to fit the structure too, where each row is a sample and
			each column corresponds to the associated variable.

		weights : array-like, shape (n_nodes), optional
			The weight of each sample as a positive double. Default is None.

		algorithm : str, one of 'chow-liu', 'greedy', 'exact', 'exact-dp' optional
			The algorithm to use for learning the Bayesian network. Default is
			'greedy' that greedily attempts to find the best structure, and
			frequently can identify the optimal structure. 'exact' uses DP/A* 
			to find the optimal Bayesian network, and 'exact-dp' tries to find
			the shortest path on the entire order lattice, which is more memory
			and computationally expensive. 'exact' and 'exact-dp' should give
			identical results, with 'exact-dp' remaining an option mostly for
			debugging reasons. 'chow-liu' will return the optimal tree-like
			structure for the Bayesian network, which is a very fast
			approximation but not always the best network.

		max_parents : int, optional
			The maximum number of parents a node can have. If used, this means
			using the k-learn procedure. Can drastically speed up algorithms.
			If -1, no max on parents. Default is -1.

		root : int, optional
			For algorithms which require a single root ('chow-liu'), this is the
			root for which all edges point away from. User may specify which
			column to use as the root. Default is the first column.

		constraint_graph : networkx.DiGraph or None, optional
			A directed graph showing valid parent sets for each variable. Each
			node is a set of variables, and edges represent which variables can
			be valid parents of those variables. The naive structure learning
			task is just all variables in a single node with a self edge,
			meaning that you know nothing about

		pseudocount : double, optional
			A pseudocount to add to the emission of each distribution. This
			effectively smoothes the states to prevent 0. probability symbols
			if they don't happen to occur in the data. Default is 0.

		state_names : array-like, shape (n_nodes), optional
			A list of meaningful names to be applied to nodes

		name : str, optional
			The name of the model. Default is None.

		n_jobs : int, optional
			The number of threads to use when learning the structure of the
			network. If a constraint graph is provided, this will parallelize
			the tasks as directed by the constraint graph. If one is not
			provided it will parallelize the building of the parent graphs.
			Both cases will provide large speed gains.

		Returns
		-------
		model : BayesianNetwork
			The learned BayesianNetwork.
		"""

		X = numpy.array(X)
		n, d = X.shape

		if max_parents == -1 or max_parents > _log(2*n / _log(n)):
			max_parents = int(_log(2*n / _log(n)))

		keys = [numpy.unique(X[:,i]) for i in range(X.shape[1])]
		keymap = numpy.array([{key: i for i, key in enumerate(keys[j])} for j in range(X.shape[1])])

		X_int = numpy.zeros((n, d), dtype='int32')
		for i in range(n):
			for j in range(d):
				X_int[i, j] = keymap[j][X[i, j]]

		key_count = numpy.array([len(keymap[i]) for i in range(d)], dtype='int32')

		if weights is None:
			weights = numpy.ones(X.shape[0], dtype='float64')
		else:
			weights = numpy.array(weights, dtype='float64')

		if algorithm == 'chow-liu':
			structure = discrete_chow_liu_tree(X_int, weights, key_count,
				pseudocount, root)
		elif algorithm == 'exact' and constraint_graph is not None:
			structure = discrete_exact_with_constraints(X_int, weights,
				key_count, pseudocount, max_parents, constraint_graph, n_jobs)
		elif algorithm == 'exact':
			structure = discrete_exact_a_star(X_int, weights, key_count,
				pseudocount, max_parents, n_jobs)
		elif algorithm == 'greedy':
			structure = discrete_greedy(X_int, weights, key_count,
				pseudocount, max_parents, n_jobs)
		elif algorithm == 'exact-dp':
			structure = discrete_exact_dp(X_int, weights, key_count,
				pseudocount, max_parents, n_jobs)
		else:
			raise ValueError("Invalid algorithm type passed in. Must be one of 'chow-liu', 'exact', 'exact-dp', 'greedy'")

		return BayesianNetwork.from_structure(X, structure, weights, pseudocount, name, 
			state_names)


cdef class ParentGraph(object):
	"""
	Generate a parent graph for a single variable over its parents.

	This will generate the parent graph for a single parents given the data.
	A parent graph is the dynamically generated best parent set and respective
	score for each combination of parent variables. For example, if we are
	generating a parent graph for x1 over x2, x3, and x4, we may calculate that
	having x2 as a parent is better than x2,x3 and so store the value
	of x2 in the node for x2,x3. 

	Parameters
	----------
	X : numpy.ndarray, shape=(n, d)
		The data to fit the structure too, where each row is a sample and
		each column corresponds to the associated variable.

	weights : numpy.ndarray, shape=(n,)
		The weight of each sample as a positive double. Default is None.

	key_count : numpy.ndarray, shape=(d,)
		The number of unique keys in each column.

	pseudocount : double
		A pseudocount to add to each possibility.

	max_parents : int
		The maximum number of parents a node can have. If used, this means
		using the k-learn procedure. Can drastically speed up algorithms.
		If -1, no max on parents. Default is -1.

	parent_set : tuple, default ()
		The variables which are possible parents for this variable. If nothing
		is passed in then it defaults to all other variables, as one would
		expect in the naive case. This allows for cases where we want to build
		a parent graph over only a subset of the variables.

	Returns
	-------
	structure : tuple, shape=(d,)
		The parents for each variable in this SCC
	"""

	cdef int i, n, d, max_parents
	cdef tuple parent_set
	cdef double pseudocount
	cdef public double all_parents_score
	cdef dict values
	cdef numpy.ndarray X
	cdef numpy.ndarray weights
	cdef numpy.ndarray key_count
	cdef int* m
	cdef int* parents

	def __init__(self, X, weights, key_count, i, pseudocount, max_parents):
		self.X = X
		self.weights = weights
		self.key_count = key_count
		self.i = i
		self.pseudocount = pseudocount
		self.max_parents = max_parents
		self.values = {}
		self.n = X.shape[0]
		self.d = X.shape[1]
		self.m = <int*> calloc(self.d+2, sizeof(int))
		self.parents = <int*> calloc(self.d, sizeof(int))

	def __len__(self):
		return len(self.values)

	def __dealloc__(self):
		free(self.m)
		free(self.parents)

	def calculate_value(self, value):
		cdef int k, parent, l = len(value)

		cdef int* X = <int*> self.X.data
		cdef int* key_count = <int*> self.key_count.data
		cdef int* m = self.m
		cdef int* parents = self.parents

		cdef double* weights = <double*> self.weights.data
		cdef double score

		m[0] = 1
		for k, parent in enumerate(value):
			m[k+1] = m[k] * key_count[parent]
			parents[k] = parent

		parents[l] = self.i
		m[l+1] = m[l] * key_count[self.i]
		m[l+2] = m[l] * (key_count[self.i] - 1)

		with nogil:
			score = discrete_score_node(X, weights, m, parents, self.n, 
				l+1, self.d, self.pseudocount)

		return score

	def __getitem__(self, value):
		if value in self.values:
			return self.values[value]

		if len(value) > self.max_parents:
			best_parents, best_score = (), NEGINF
		else:
			best_parents, best_score = value, self.calculate_value(value)
		
		for variable in value:
			parent_subset = tuple(parent for parent in value if parent != variable)
			parents, score = self[parent_subset]

			if score > best_score:
				best_score = score
				best_parents = parents

		self.values[value] = (best_parents, best_score)
		return self.values[value]


def discrete_chow_liu_tree(numpy.ndarray X_ndarray, numpy.ndarray weights_ndarray,
	numpy.ndarray key_count_ndarray, double pseudocount, int root):
	cdef int i, j, k, l, lj, lk, Xj, Xk, xj, xk
	cdef int n = X_ndarray.shape[0], d = X_ndarray.shape[1]
	cdef int max_keys = key_count_ndarray.max()

	cdef int* X = <int*> X_ndarray.data
	cdef double* weights = <double*> weights_ndarray.data
	cdef int* key_count = <int*> key_count_ndarray.data

	cdef numpy.ndarray mutual_info_ndarray = numpy.zeros((d, d), dtype='float64')
	cdef double* mutual_info = <double*> mutual_info_ndarray.data

	cdef double* marg_j = <double*> calloc(max_keys, sizeof(double))
	cdef double* marg_k = <double*> calloc(max_keys, sizeof(double))
	cdef double* joint_count = <double*> calloc(max_keys**2, sizeof(double))

	for j in range(d):
		for k in range(j):
			if j == k:
				continue

			lj = key_count[j]
			lk = key_count[k]

			for i in range(max_keys):
				marg_j[i] = pseudocount
				marg_k[i] = pseudocount

				for l in range(max_keys):
					joint_count[i*max_keys + l] = pseudocount

			for i in range(n):
				Xj = X[i*d + j]
				Xk = X[i*d + k]

				joint_count[Xj * lk + Xk] += weights[i]
				marg_j[Xj] += weights[i]
				marg_k[Xk] += weights[i]

			for xj in range(lj):
				for xk in range(lk):
					if joint_count[xj*lk+xk] > 0:
						mutual_info[j*d + k] -= joint_count[xj*lk+xk] * _log(
							joint_count[xj*lk+xk] / (marg_j[xj] * marg_k[xk]))
						mutual_info[k*d + j] = mutual_info[j*d + k]


	structure = [() for i in range(d)]
	visited = [root]
	unvisited = list(range(d))
	unvisited.remove(root)

	while len(unvisited) > 0:
		min_score, min_x, min_y = INF, -1, -1

		for x in visited:
			for y in unvisited:
				score = mutual_info_ndarray[x, y]
				if score < min_score:
					min_score, min_x, min_y = score, x, y

		structure[min_y] += (min_x,)
		visited.append(min_y)
		unvisited.remove(min_y)

	free(marg_j)
	free(marg_k)
	free(joint_count)
	return tuple(structure)


def discrete_exact_with_constraints(numpy.ndarray X, numpy.ndarray weights,
	numpy.ndarray key_count, double pseudocount, int max_parents,
	object constraint_graph, int n_jobs):
	"""
	This returns the optimal Bayesian network given a set of constraints.

	This function controls the process of learning the Bayesian network by
	taking in a constraint graph, identifying the strongly connected
	components (SCCs) and solving each one using the appropriate algorithm.
	This is mostly an internal function.

	Parameters
	----------
	X : numpy.ndarray, shape=(n, d)
		The data to fit the structure too, where each row is a sample and
		each column corresponds to the associated variable.

	weights : numpy.ndarray, shape=(n,)
		The weight of each sample as a positive double. Default is None.

	key_count : numpy.ndarray, shape=(d,)
		The number of unique keys in each column.

	pseudocount : double
		A pseudocount to add to each possibility.

	max_parents : int
		The maximum number of parents a node can have. If used, this means
		using the k-learn procedure. Can drastically speed up algorithms.
		If -1, no max on parents. Default is -1.

	constraint_graph : networkx.DiGraph
		A directed graph showing valid parent sets for each variable. Each
		node is a set of variables, and edges represent which variables can
		be valid parents of those variables. The naive structure learning
		task is just all variables in a single node with a self edge,
		meaning that you know nothing about

	n_jobs : int
		The number of threads to use when learning the structure of the
		network. This parallelized both the creation of the parent
		graphs for each variable and the solving of the SCCs.

	Returns
	-------
	structure : tuple, shape=(d,)
		The parents for each variable in the network.
	"""

	n, d = X.shape[0], X.shape[1]
	l = len(constraint_graph.nodes())
	parent_sets = { node : tuple() for node in constraint_graph.nodes() }
	structure = [None for i in range(d)]

	for parents, children in constraint_graph.edges():
		parent_sets[children] += parents

	tasks = []
	components = nx.strongly_connected_components(constraint_graph)
	for component in components:
		component = list(component)

		if len(component) == 1:
			children = component[0]
			parents = parent_sets[children]

			if children == parents:
				tasks.append((0, parents, children))
			elif set(children).issubset(set(parents)):
				tasks.append((1, parents, children))
			else:
				if len(parents) > 0:
					for child in children:
						tasks.append((2, parents, child))
		else:
			parents = [parent_sets[children] for children in component]
			tasks.append((3, parents, component))

	with Parallel(n_jobs=n_jobs, backend='threading') as parallel:
		local_structures = parallel( delayed(discrete_exact_with_constraints_task)(
			X, weights, key_count, pseudocount, max_parents, task, n_jobs) 
			for task in tasks)

	structure = [[] for i in range(d)]
	for local_structure in local_structures:
		for i in range(d):
			structure[i] += list(local_structure[i])

	return tuple(tuple(node) for node in structure)


def discrete_exact_with_constraints_task(numpy.ndarray X, numpy.ndarray weights,
	numpy.ndarray key_count, double pseudocount, int max_parents, tuple task,
	int n_jobs):
	"""
	This is a wrapper for the function to be parallelzied by joblib.

	This function takes in a single task as an id and a set of parents and
	children and calls the appropriate function. This is mostly a wrapper for
	joblib to parallelize.

	Parameters
	----------
	X : numpy.ndarray, shape=(n, d)
		The data to fit the structure too, where each row is a sample and
		each column corresponds to the associated variable.

	weights : numpy.ndarray, shape=(n,)
		The weight of each sample as a positive double. Default is None.

	key_count : numpy.ndarray, shape=(d,)
		The number of unique keys in each column.

	pseudocount : double
		A pseudocount to add to each possibility.

	max_parents : int
		The maximum number of parents a node can have. If used, this means
		using the k-learn procedure. Can drastically speed up algorithms.
		If -1, no max on parents. Default is -1.

	task : tuple
		A 3-tuple containing the id, the set of parents and the set of children
		to learn a component of the Bayesian network over. The cases represent
		a SCC of the following:

			0 - Self loop and no parents
			1 - Self loop and parents
			2 - Parents and no self loop
			3 - Multiple nodes

	n_jobs : int
		The number of threads to use when learning the structure of the
		network. This parallelizes the creation of the parent graphs
		for each task or the finding of best parents in case 2.

	Returns
	-------
	structure : tuple, shape=(d,)
		The parents for each variable in this SCC
	"""


	d = X.shape[1]
	structure = [() for i in range(d)]
	case, parents, children = task

	if case == 0:
		local_structure = discrete_exact_a_star(X[:,parents].copy(), weights,
			key_count[list(parents)], pseudocount, max_parents, False, n_jobs)

		for i, parent in enumerate(parents):
			structure[parent] = tuple([parents[k] for k in local_structure[i]])

	elif case == 1:
		structure = discrete_exact_slap(X, weights, task, key_count, 
			pseudocount, max_parents, n_jobs)

	elif case == 2:
		logp, local_structure = discrete_find_best_parents(X, weights,
			key_count, pseudocount, max_parents, parents, children)

		structure[children] = local_structure

	elif case == 3:
		structure = discrete_exact_component(X, weights,
			task, key_count, pseudocount, max_parents, n_jobs)

	return tuple(structure)


def discrete_exact_dp(X, weights, key_count, pseudocount, max_parents, n_jobs):
	"""
	Find the optimal graph over a set of variables with no other knowledge.

	This is the naive dynamic programming structure learning task where the 
	optimal graph is identified from a set of variables using an order graph
	and parent graphs. This can be used either when no constraint graph is
	provided or for a SCC which is made up of a node containing a self-loop.
	This is a reference implementation that uses the naive shortest path
	algorithm over the entire order graph. The 'exact' option uses the A* path
	in order to avoid considering the full order graph.

	Parameters
	----------
	X : numpy.ndarray, shape=(n, d)
		The data to fit the structure too, where each row is a sample and
		each column corresponds to the associated variable.

	weights : numpy.ndarray, shape=(n,)
		The weight of each sample as a positive double. Default is None.

	key_count : numpy.ndarray, shape=(d,)
		The number of unique keys in each column.

	pseudocount : double
		A pseudocount to add to each possibility.

	max_parents : int
		The maximum number of parents a node can have. If used, this means
		using the k-learn procedure. Can drastically speed up algorithms.
		If -1, no max on parents. Default is -1.

	n_jobs : int
		The number of threads to use when learning the structure of the
		network. This parallelizes the creation of the parent graphs.

	Returns
	-------
	structure : tuple, shape=(d,)
		The parents for each variable in this SCC
	"""

	cdef int i, n = X.shape[0], d = X.shape[1]
	cdef list parent_graphs = []

	parent_graphs = Parallel(n_jobs=n_jobs, backend='threading')( 
		delayed(generate_parent_graph)(X, weights, key_count, i, pseudocount, 
			max_parents) for i in range(d) )

	order_graph = nx.DiGraph()

	for i in range(d+1):
		for subset in it.combinations(range(d), i):
			order_graph.add_node(subset)

			for variable in subset:
				parent = tuple(v for v in subset if v != variable)

				structure, weight = parent_graphs[variable][parent]
				weight = -weight if weight < 0 else 0
				order_graph.add_edge(parent, subset, weight=weight,
					structure=structure)

	path = nx.shortest_path(order_graph, source=(), target=tuple(range(d)),
		weight='weight')

	score, structure = 0, list( None for i in range(d) )
	for u, v in zip(path[:-1], path[1:]):
		idx = list(set(v) - set(u))[0]
		parents = order_graph.get_edge_data(u, v)['structure']
		structure[idx] = parents
		score -= order_graph.get_edge_data(u, v)['weight']

	return tuple(structure)


def discrete_exact_a_star(X, weights, key_count, pseudocount, max_parents, n_jobs):
	"""
	Find the optimal graph over a set of variables with no other knowledge.

	This is the naive dynamic programming structure learning task where the 
	optimal graph is identified from a set of variables using an order graph
	and parent graphs. This can be used either when no constraint graph is
	provided or for a SCC which is made up of a node containing a self-loop.
	It uses DP/A* in order to find the optimal graph without considering all
	possible topological sorts. A greedy version of the algorithm can be used 
	that massively reduces both the computational and memory cost while frequently 
	producing the optimal graph. 

	Parameters
	----------
	X : numpy.ndarray, shape=(n, d)
		The data to fit the structure too, where each row is a sample and
		each column corresponds to the associated variable.

	weights : numpy.ndarray, shape=(n,)
		The weight of each sample as a positive double. Default is None.

	key_count : numpy.ndarray, shape=(d,)
		The number of unique keys in each column.

	pseudocount : double
		A pseudocount to add to each possibility.

	max_parents : int
		The maximum number of parents a node can have. If used, this means
		using the k-learn procedure. Can drastically speed up algorithms.
		If -1, no max on parents. Default is -1.

	n_jobs : int
		The number of threads to use when learning the structure of the
		network. This parallelizes the creation of the parent graphs.

	Returns
	-------
	structure : tuple, shape=(d,)
		The parents for each variable in this SCC
	"""

	cdef int i, n = X.shape[0], d = X.shape[1]
	cdef list parent_graphs = []

	parent_graphs = [ParentGraph(X, weights, key_count, i, pseudocount, max_parents) for i in range(d)]

	other_variables = {}
	for i in range(d):
		other_variables[i] = tuple(j for j in range(d) if j != i)

	o = PriorityQueue()
	closed = {}

	h = sum(parent_graphs[i][other_variables[i]][1] for j in range(d))
	o.push(((), h, [() for i in range(d)]), 0)
	while not o.empty():
		weight, (variables, g, structure) = o.pop()

		if variables in closed:
			continue
		else:
			closed[variables] = 1

		if len(variables) == d:
			return tuple(structure)

		out_set = tuple(i for i in range(d) if i not in variables)
		for i in out_set:
			pg = parent_graphs[i]
			parents, c = pg[variables]

			e = g - c
			f = weight - c + pg[other_variables[i]][1]

			local_structure = structure[:]
			local_structure[i] = parents

			new_variables = tuple(sorted(variables + (i,)))
			entry = (new_variables, e, local_structure)

			prev_entry = o.get(new_variables)
			if prev_entry is not None:
				if prev_entry[0] > f:
					o.delete(new_variables)
					o.push(entry, f)
			else:
				o.push(entry, f)


def discrete_greedy(X, weights, key_count, pseudocount, max_parents, n_jobs):
	"""
	Find the optimal graph over a set of variables with no other knowledge.

	This is the naive dynamic programming structure learning task where the 
	optimal graph is identified from a set of variables using an order graph
	and parent graphs. This can be used either when no constraint graph is
	provided or for a SCC which is made up of a node containing a self-loop.
	It uses DP/A* in order to find the optimal graph without considering all
	possible topological sorts. A greedy version of the algorithm can be used 
	that massively reduces both the computational and memory cost while frequently 
	producing the optimal graph. 

	Parameters
	----------
	X : numpy.ndarray, shape=(n, d)
		The data to fit the structure too, where each row is a sample and
		each column corresponds to the associated variable.

	weights : numpy.ndarray, shape=(n,)
		The weight of each sample as a positive double. Default is None.

	key_count : numpy.ndarray, shape=(d,)
		The number of unique keys in each column.

	pseudocount : double
		A pseudocount to add to each possibility.

	max_parents : int
		The maximum number of parents a node can have. If used, this means
		using the k-learn procedure. Can drastically speed up algorithms.
		If -1, no max on parents. Default is -1.

	greedy : bool, default is True
		Whether the use a heuristic in order to massive reduce computation
		and memory time, but without the guarantee of finding the best
		network.

	n_jobs : int
		The number of threads to use when learning the structure of the
		network. This parallelizes the creation of the parent graphs.

	Returns
	-------
	structure : tuple, shape=(d,)
		The parents for each variable in this SCC
	"""

	cdef int i, n = X.shape[0], d = X.shape[1]
	cdef list parent_graphs = []

	parent_graphs = [ParentGraph(X, weights, key_count, i, pseudocount, max_parents) for i in range(d)]
	structure, seen_variables, unseen_variables = [() for i in range(d)], (), set(range(d))

	for i in range(d):
		best_score = NEGINF
		best_variable = -1
		best_parents = None

		for j in unseen_variables:
			parents, score = parent_graphs[j][seen_variables]

			if score > best_score:
				best_score = score
				best_variable = j
				best_parents = parents

		structure[best_variable] = best_parents
		seen_variables = tuple(sorted(seen_variables + (best_variable,)))
		unseen_variables = unseen_variables - set([best_variable])

	return tuple(structure)

def discrete_exact_slap(X, weights, task, key_count, pseudocount, max_parents, 
	n_jobs):
	"""
	Find the optimal graph in a node with a Self Loop And Parents (SLAP).

	Instead of just performing exact BNSL over the set of all parents and
	removing the offending edges there are efficiencies that can be gained
	by considering the structure. In particular, parents not coming from the
	main node do not need to be considered in the order graph but simply
	added to each entry after creation of the order graph. This is because
	those variables occur earlier in the topological ordering but it doesn't
	matter how they occur otherwise. Parent graphs must be defined over all
	variables however.

	Parameters
	----------
	X : numpy.ndarray, shape=(n, d)
		The data to fit the structure too, where each row is a sample and
		each column corresponds to the associated variable.

	weights : numpy.ndarray, shape=(n,)
		The weight of each sample as a positive double. Default is None.

	key_count : numpy.ndarray, shape=(d,)
		The number of unique keys in each column.

	pseudocount : double
		A pseudocount to add to each possibility.

	max_parents : int
		The maximum number of parents a node can have. If used, this means
		using the k-learn procedure. Can drastically speed up algorithms.
		If -1, no max on parents. Default is -1.

	n_jobs : int
		The number of threads to use when learning the structure of the
		network. This parallelizes the creation of the parent graphs.

	Returns
	-------
	structure : tuple, shape=(d,)
		The parents for each variable in this SCC
	"""

	cdef tuple parents = task[0], children = task[1]
	cdef tuple outside_parents = tuple(i for i in parents if i not in children)
	cdef int i, n = X.shape[0], d = X.shape[1]
	cdef list parent_graphs = [None for i in range(max(parents)+1)]

	graphs = Parallel(n_jobs=n_jobs, backend='threading')( 
		delayed(generate_parent_graph)(X, weights, key_count, i, pseudocount, 
			max_parents) for i in children)

	for i, child in enumerate(children):
		parent_graphs[child] = graphs[i]

	order_graph = nx.DiGraph()
	for i in range(d+1):
		for subset in it.combinations(children, i):
			order_graph.add_node(subset + outside_parents)

			for variable in subset:
				parent = tuple(v for v in subset if v != variable)
				parent += outside_parents

				structure, weight = parent_graphs[variable][parent]
				weight = -weight if weight < 0 else 0
				order_graph.add_edge(parent, subset + outside_parents, weight=weight,
					structure=structure)

	path = nx.shortest_path(order_graph, source=outside_parents, target=parents,
		weight='weight')

	score, structure = 0, list(() for i in range(d))
	for u, v in zip(path[:-1], path[1:]):
		idx = list(set(v) - set(u))[0]
		parents = order_graph.get_edge_data(u, v)['structure']
		structure[idx] = parents
		score -= order_graph.get_edge_data(u, v)['weight']

	return tuple(structure)


def discrete_exact_component(X, weights, task, key_count, pseudocount, 
	max_parents, n_jobs):
	"""
	Find the optimal graph over a multi-node component of the constaint graph.

	The general algorithm in this case is to begin with each variable and add
	all possible single children for that entry recursively until completion.
	This will result in a far sparser order graph than before. In addition, one
	can eliminate entries from the parent graphs that contain invalid parents
	as they are a fast of computational time. 

	Parameters
	----------
	X : numpy.ndarray, shape=(n, d)
		The data to fit the structure too, where each row is a sample and
		each column corresponds to the associated variable.

	weights : numpy.ndarray, shape=(n,)
		The weight of each sample as a positive double. Default is None.

	key_count : numpy.ndarray, shape=(d,)
		The number of unique keys in each column.

	pseudocount : double
		A pseudocount to add to each possibility.

	max_parents : int
		The maximum number of parents a node can have. If used, this means
		using the k-learn procedure. Can drastically speed up algorithms.
		If -1, no max on parents. Default is -1.

	n_jobs : int
		The number of threads to use when learning the structure of the
		network. This parallelizes the creation of the parent graphs.

	Returns
	-------
	structure : tuple, shape=(d,)
		The parents for each variable in this SCC
	"""

	cdef int i, n = X.shape[0], d = X.shape[1]
	cdef double weight

	cdef int variable, child, parent
	cdef tuple parent_set
	cdef tuple entry, parent_entry, filtered_entry
	cdef list last_layer, last_layer_child_sets, layer, layer_child_sets
	cdef set child_set

	_, parent_vars, children_vars = task
	variable_set = set([])
	for parents in parent_vars:
		variable_set = variable_set.union(parents)
	for children in children_vars:
		variable_set = variable_set.union(children)

	parent_sets = {variable: () for variable in variable_set}
	child_sets = {variable: () for variable in variable_set}
	for parents, children in zip(parent_vars, children_vars):
		for child in children:
			parent_sets[child] += parents
		for parent in parents:
			child_sets[parent] += children

	graphs = Parallel(n_jobs=n_jobs, backend='threading')( 
		delayed(generate_parent_graph)(X, weights, key_count, child, pseudocount, 
			max_parents, parents) for child, parents in parent_sets.items())

	parent_graphs = [None for i in range(d)]
	for (child, _), graph in zip(parent_sets.items(), graphs):
		parent_graphs[child] = graph

	last_layer = []
	last_layer_children = []

	order_graph = nx.DiGraph()
	order_graph.add_node(())
	for variable in variable_set:
		order_graph.add_node((variable,))

		structure, weight = parent_graphs[variable][()]
		weight = -weight if weight < 0 else 0
		order_graph.add_edge((), (variable,), weight=weight, structure=structure)

		last_layer.append((variable,))
		last_layer_children.append(set(child_sets[variable]))

	layer = [] 
	layer_children = []

	seen_entries = {(variable,): 1 for variable in variable_set} 

	for i in range(len(variable_set)-1):
		for parent_entry, child_set in zip(last_layer, last_layer_children):
			for child in child_set:
				parent_set = parent_sets[child]
				entry = tuple(sorted(parent_entry + (child,)))

				filtered_entry = tuple(variable for variable in parent_entry if variable in parent_set)
				structure, weight = parent_graphs[child][filtered_entry]
				weight = -weight if weight < 0 else 0

				order_graph.add_edge(parent_entry, entry, weight=round(weight, 4), structure=structure)

				new_child_set = child_set - set([child])
				for grandchild in child_sets[child]:
					if grandchild not in entry:
						new_child_set.add(grandchild)

				if entry not in seen_entries:
					seen_entries[entry] = 1
					layer.append(entry)
					layer_children.append(new_child_set)

		last_layer = layer
		last_layer_children = layer_children

		layer = []
		layer_children = []

	path = nx.shortest_path(order_graph, source=(), target=tuple(sorted(variable_set)),
		weight='weight')

	score, structure = 0, list(() for i in range(d))
	for u, v in zip(path[:-1], path[1:]):
		idx = list(set(v) - set(u))[0]
		parents = order_graph.get_edge_data(u, v)['structure']
		structure[idx] = parents
		score -= order_graph.get_edge_data(u, v)['weight']

	return tuple(structure)


def generate_parent_graph(numpy.ndarray X_ndarray,
	numpy.ndarray weights_ndarray, numpy.ndarray key_count_ndarray,
	int i, double pseudocount, int max_parents, tuple parent_set=()):
	"""
	Generate a parent graph for a single variable over its parents.

	This will generate the parent graph for a single parents given the data.
	A parent graph is the dynamically generated best parent set and respective
	score for each combination of parent variables. For example, if we are
	generating a parent graph for x1 over x2, x3, and x4, we may calculate that
	having x2 as a parent is better than x2,x3 and so store the value
	of x2 in the node for x2,x3. 

	Parameters
	----------
	X : numpy.ndarray, shape=(n, d)
		The data to fit the structure too, where each row is a sample and
		each column corresponds to the associated variable.

	weights : numpy.ndarray, shape=(n,)
		The weight of each sample as a positive double. Default is None.

	key_count : numpy.ndarray, shape=(d,)
		The number of unique keys in each column.

	pseudocount : double
		A pseudocount to add to each possibility.

	max_parents : int
		The maximum number of parents a node can have. If used, this means
		using the k-learn procedure. Can drastically speed up algorithms.
		If -1, no max on parents. Default is -1.

	parent_set : tuple, default ()
		The variables which are possible parents for this variable. If nothing
		is passed in then it defaults to all other variables, as one would
		expect in the naive case. This allows for cases where we want to build
		a parent graph over only a subset of the variables.

	Returns
	-------
	structure : tuple, shape=(d,)
		The parents for each variable in this SCC
	"""


	cdef int j, k, variable, l
	cdef int n = X_ndarray.shape[0], d = X_ndarray.shape[1]

	cdef int* X = <int*> X_ndarray.data
	cdef int* key_count = <int*> key_count_ndarray.data
	cdef int* m = <int*> calloc(d+2, sizeof(int))
	cdef int* parents = <int*> calloc(d, sizeof(int))

	cdef double* weights = <double*> weights_ndarray.data
	cdef dict parent_graph = {}
	cdef double best_score, score
	cdef tuple subset, parent_subset, best_structure, structure

	if parent_set == ():
		parent_set = tuple(set(range(d)) - set([i]))

	cdef int n_parents = len(parent_set)

	m[0] = 1
	for j in range(n_parents+1):
		for subset in it.combinations(parent_set, j):
			if j <= max_parents:
				for k, variable in enumerate(subset):
					m[k+1] = m[k] * key_count_ndarray[variable]
					parents[k] = variable

				parents[j] = i
				m[j+1] = m[j] * key_count[i]
				m[j+2] = m[j] * (key_count[i] - 1)

				best_structure = subset

				with nogil:
					best_score = discrete_score_node(X, weights, m, parents, n, j+1,
						d, pseudocount)
			else:
				best_structure, best_score = (), NEGINF

			for k, variable in enumerate(subset):
				parent_subset = tuple(l for l in subset if l != variable)
				structure, score = parent_graph[parent_subset]

				if score > best_score:
					best_score = score
					best_structure = structure

			parent_graph[subset] = (best_structure, best_score)

	free(m)
	free(parents)
	return parent_graph

cdef double discrete_score_node(int* X, double* weights, int* m, int* parents, 
	int n, int d, int l, double pseudocount) nogil:
	cdef int i, j, k, idx
	cdef double logp = -_log(n) / 2 * m[d+1]
	cdef double count, marginal_count
	cdef double* counts = <double*> calloc(m[d], sizeof(double))
	cdef double* marginal_counts = <double*> calloc(m[d-1], sizeof(double))

	memset(counts, 0, m[d]*sizeof(double))
	memset(marginal_counts, 0, m[d-1]*sizeof(double))

	for i in range(n):
		idx = 0
		for j in range(d-1):
			k = parents[j]
			idx += X[i*l+k] * m[j]

		marginal_counts[idx] += weights[i]
		k = parents[d-1]
		idx += X[i*l+k] * m[d-1]

		counts[idx] += weights[i]

	for i in range(m[d]):
		count = pseudocount + counts[i]
		marginal_count = pseudocount * (m[d] / m[d-1]) + marginal_counts[i%m[d-1]]

		if count > 0:
			logp += count * _log(count / marginal_count)

	free(counts)
	free(marginal_counts)
	return logp


cdef discrete_find_best_parents(numpy.ndarray X_ndarray,
	numpy.ndarray weights_ndarray, numpy.ndarray key_count_ndarray,
	double pseudocount, int max_parents, tuple parent_set, int i):
	cdef int j, k
	cdef int n = X_ndarray.shape[0], l = X_ndarray.shape[1]

	cdef int* X = <int*> X_ndarray.data
	cdef int* key_count = <int*> key_count_ndarray.data
	cdef int* m = <int*> calloc(l+2, sizeof(int))
	cdef int* combs = <int*> calloc(l, sizeof(int))

	cdef double* weights = <double*> weights_ndarray.data

	cdef double best_score = NEGINF, score
	cdef tuple best_parents, parents

	m[0] = 1
	for k in range(min(max_parents, len(parent_set))+1):
		for parents in it.combinations(parent_set, k):
			for j in range(k):
				combs[j] = parents[j]
				m[j+1] = m[j] * key_count[combs[j]]

			combs[k] = i
			m[k+1] = m[k] * key_count[i]
			m[k+2] = m[k] * (key_count[i] - 1)

			with nogil:
				score = discrete_score_node(X, weights, m, combs, n, k+1, l, 
					pseudocount)

			if score > best_score:
				best_score = score
				best_parents = parents

	free(m)
	free(combs)
	return best_score, best_parents

# BayesianNetwork.pyx
# Contact: Jacob Schreiber ( jmschreiber91@gmail.com )

import itertools as it
import json
import time
import networkx as nx
import numpy
cimport numpy

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

try:
	import tempfile
	import pygraphviz
	import matplotlib.pyplot as plt
	import matplotlib.image
except ImportError:
	pygraphviz = None

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
				G.add_edge(parent.name, child.name)

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

		if self.d == 0:
			raise ValueError("must bake model before computing probability")

		sample = numpy.array(sample, ndmin=2)
		logp = 0.0
		for i, state in enumerate( self.states ):
			logp += state.distribution.log_probability( sample[0, self.idxs[i]] )
		
		return logp

	cdef double _mv_log_probability(self, double* symbol) nogil:
		cdef double logp
		self._v_log_probability(symbol, &logp, 1)
		return logp

	cdef void _v_log_probability( self, double* symbol, double* log_probability, int n ) nogil:
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

				(<Model> self.distributions_ptr[j])._v_log_probability(sym, &logp, 1)
				log_probability[i] += logp


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

		if self.d == 0:
			raise ValueError("must bake model before computing marginal")

		return self.graph.marginal()

	def predict_proba( self, data={}, max_iterations=100, check_input=True ):
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

	def from_summaries( self, inertia=0.0 ):
		"""Use MLE on the stored sufficient statistics to train the model.

		This uses MLE estimates on the stored sufficient statistics to train
		the model.

		Parameters
		----------
		inertia : double, optional
			The inertia for updating the distributions, passed along to the
			distribution method. Default is 0.0.

		Returns
		-------
		None
		"""

		for state in self.states:
			state.distribution.from_summaries(inertia)

	def fit( self, items, weights=None, inertia=0.0 ):
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

		Returns
		-------
		None
		"""

		self.summarize(items, weights)
		self.from_summaries(inertia)
		self.bake()

	def predict( self, items, max_iterations=100 ):
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
		items : array-like, shape (n_samples, n_nodes)
			This is the data matrix with the missing values imputed.
		"""

		if self.d == 0:
			raise ValueError("must bake model before using impute")

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

			imputation = self.predict_proba( obs  )

			for j in range( len( self.states) ):
				items[i][j] = imputation[j].mle()

		return items 

	def impute( self, *args, **kwargs ):
		raise Warning("method 'impute' has been depricated, please use 'predict' instead") 

	def to_json( self, separators=(',', ' : '), indent=4 ):
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
	def from_json( cls, s ):
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
	def from_structure( cls, X, structure, weights=None, name=None ):
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

		name : str, optional
			The name of the model. Default is None.

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
				nodes[i] = DiscreteDistribution.from_samples(X[:,i], weights=weights_ndarray)

		while True:
			for i, parents in enumerate(structure):
				if nodes[i] is None:
					for parent in parents:
						if nodes[parent] is None:
							break
					else:
						nodes[i] = ConditionalProbabilityTable.from_samples(X[:,parents+(i,)], 
							parents=[nodes[parent] for parent in parents], 
							weights=weights_ndarray)
						break
			else:
				break

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
	def from_samples( cls, X, weights=None, algorithm='chow-liu', max_parents=-1,
		 root=0, constraint_graph=None, pseudocount=0.0 ):
		"""Learn the structure of the network from data.

		Find the structure of the network from data using a Bayesian structure
		learning score. This currently enumerates all the exponential number of
		structures and finds the best according to the score. This allows
		weights on the different samples as well.

		Parameters
		----------
		X : array-like, shape (n_samples, n_nodes)
			The data to fit the structure too, where each row is a sample and 
			each column corresponds to the associated variable.

		weights : array-like, shape (n_nodes), optional
			The weight of each sample as a positive double. Default is None.

		algorithm : str, one of 'chow-liu', 'exact' optional
			The algorithm to use for learning the Bayesian network. Default is
			'chow-liu' which returns a tree structure.

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
			A pseudocount to add to each possibility.

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
				key_count, pseudocount, max_parents, constraint_graph) 
		elif algorithm == 'exact':
			structure = discrete_exact_graph(X_int, weights, key_count, 
				pseudocount, max_parents)

		return BayesianNetwork.from_structure(X, structure, weights)


cdef tuple discrete_chow_liu_tree(numpy.ndarray X_ndarray, numpy.ndarray weights_ndarray, 
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
	

cdef discrete_exact_graph(numpy.ndarray X, numpy.ndarray weights, 
	numpy.ndarray key_count, double pseudocount, int max_parents):
	
	cdef int n = X.shape[0], d = X.shape[1]
	cdef list parent_graphs = [{} for i in range(d)]

	generate_parent_graphs(X, weights, key_count, parent_graphs, max_parents, 
		pseudocount)

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


cdef void generate_parent_graphs(numpy.ndarray X_ndarray, 
	numpy.ndarray weights_ndarray, numpy.ndarray key_count_ndarray,
	list parent_graphs, int max_parents, double pseudocount):

	cdef int i, j, k
	cdef int n = X_ndarray.shape[0], l = X_ndarray.shape[1]

	cdef int* X = <int*> X_ndarray.data
	cdef int* key_count = <int*> key_count_ndarray.data
	cdef int* m = <int*> calloc(l+2, sizeof(int))
	cdef int* parents = <int*> calloc(l, sizeof(int))
	cdef int* combs = <int*> calloc(l, sizeof(int))

	cdef double* weights = <double*> weights_ndarray.data
	cdef double* scores = <double*> calloc(2**(l-1), sizeof(double))

	cdef list structures = [None for i in range(2**(l-1))]

	m[0] = 1
	for i in range(l):
		j = 0
		for k in range(l):
			if k != i:
				parents[j] = k
				j += 1

		for k in range(l):
			generate_parent_layer(X, weights, key_count, n, l, m, scores, 
				structures, parent_graphs, max_parents, pseudocount, i, parents, 
				combs, l-1, k, k, 0)

	free(m)
	free(scores)
	del structures


cdef void generate_parent_layer(int* X, double* weights, int* key_count, int n, 
	int l, int* m, double* scores, list structures, list parent_graphs, 
	int max_parents, double pseudocount, int i, int* parents, int* combs, 
	int n_parents, int k, int length, int start):

	cdef int ii, j, ij, idx
	cdef double best_score
	cdef tuple parent_tuple, best_parents

	if length == 0:
		parent_tuple = tuple(combs[j] for j in range(k))

		for j in range(k):
			m[j+1] = m[j] * key_count[combs[j]]

		combs[k] = i
		m[k+1] = m[k] * key_count[i]
		m[k+2] = m[k] * (key_count[i] - 1)

		if k <= max_parents: 
			best_parents = parent_tuple
			best_score = score_node(X, weights, m, combs, n, k+1, l, pseudocount)
		else:
			best_parents = ()
			best_score = NEGINF

		for j in range(k):
			ij, idx = 0, 0
			for ii in range(l):
				if ii == i:
					continue

				if ij < k and ij != j and combs[ij] == ii:
					idx = idx * 2 + 1
					ij += 1
				else:
					idx = idx * 2

			if scores[idx] >= best_score:
				best_score = scores[idx]
				best_parents = structures[idx]

		idx, ij = 0, 0
		for ii in range(l):
			if ii == i:
				continue
			if ij < k and combs[ij] == ii:
				idx = idx * 2 + 1
				ij += 1
			else:
				idx = idx * 2

		scores[idx] = best_score
		structures[idx] = best_parents

		parent_graphs[i][parent_tuple] = (best_parents, best_score)
		return

	for ii in range(start, n_parents-length+1):
		combs[k - length] = parents[ii]
		generate_parent_layer(X, weights, key_count, n, l, m, scores, 
			structures, parent_graphs, max_parents, pseudocount, i, parents, 
			combs, n_parents, k, length-1, ii+1)


cdef double score_graph(int* X, double* weights, int* key_count, int n, int l, 
	tuple structure, double pseudocount):
	cdef int i, j, d
	cdef double logp

	cdef int* m = <int*> calloc(l+2, sizeof(int))
	cdef int* idxs = <int*> calloc(l, sizeof(int))

	logp = 0.0
	m[0] = 1
	for i in range(l):
		parents = structure[i] + (i,)
		d = len(parents)

		for j in range(d):
			idxs[j] = parents[j]
			m[j+1] = m[j] * key_count[parents[j]]
		m[j+2] = m[j] * (key_count[parents[j]] - 1)

		logp += score_node(X, weights, m, idxs, n, d, l, pseudocount)

	free(m)
	free(idxs)
	return logp


cdef double score_node(int* X, double* weights, int* m, int* parents, int n, int d, int l, double pseudocount) nogil:
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
			logp += count * _log( count / marginal_count )

	free(counts)
	free(marginal_counts)
	return logp


cpdef discrete_exact_with_constraints(numpy.ndarray X, numpy.ndarray weights, 
	numpy.ndarray key_count, double pseudocount, int max_parents, 
	object constraint_graph):

	n, d = X.shape[0], X.shape[1]
	l = len(constraint_graph.nodes())
	parent_sets = { node : tuple() for node in constraint_graph.nodes() }
	indices = { node : i for i, node in enumerate(constraint_graph.nodes()) }
	cycle = numpy.zeros(l)
	structure = [None for i in range(d)]

	for parent, child in constraint_graph.edges():
		parent_sets[child] += parent
		if child == parent:
			cycle[indices[child]] = 1

	for children, parents in parent_sets.items():
		if cycle[indices[children]] == 1:
			local_structure = discrete_exact_graph(X[:,parents].copy(), weights, 
				key_count[list(parents)], pseudocount, max_parents)

			for i, parent in enumerate(parents):
				if parent in children:
					structure[parent] = tuple([parents[k] for k in local_structure[i]])

		else:
			for child in children:
				logp, node_parents = discrete_find_best_parents(X, weights,
					key_count, child, parents, max_parents, pseudocount)

				structure[child] = node_parents

	return tuple(structure)


cdef discrete_find_best_parents(numpy.ndarray X_ndarray, 
	numpy.ndarray weights_ndarray, numpy.ndarray key_count_ndarray,
	int i, tuple parent_set, int max_parents, double pseudocount):

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

			score = score_node(X, weights, m, combs, n, k+1, l, pseudocount)

			if score > best_score:
				best_score = score
				best_parents = parents

	free(m)
	free(combs)
	return best_score, best_parents

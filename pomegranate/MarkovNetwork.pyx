import itertools
import json
import time
import numpy
cimport numpy

from scipy.special import logsumexp

from joblib import Parallel
from joblib import delayed

from libc.stdlib cimport calloc
from libc.stdlib cimport free
from libc.stdlib cimport malloc
from libc.string cimport memset

from .base cimport Model
from .base import State

from distributions import Distribution
from distributions.DiscreteDistribution cimport DiscreteDistribution
from distributions.JointProbabilityTable cimport JointProbabilityTable

from .FactorGraph import FactorGraph
from .utils cimport _log
from .utils cimport isnan

from .utils import _check_nan

nan = numpy.nan

cdef class MarkovNetwork(Model):
	"""A Markov Network Model.

	A Markov network is an undirected graph where nodes represent variables,
	edges represent associations between the variables, and the lack of an edge
	represents a conditional independence.

	Parameters
	----------
	distributions : list, tuple, or numpy.ndarray
		A collection of joint probability distributions that represent the 

	name : str, optional
		The name of the model. Default is None
	"""

	cdef object graph
	cdef list idxs
	cdef public list keys_
	cdef public numpy.ndarray keymap
	cdef int* parent_count
	cdef int* parent_idxs
	cdef public numpy.ndarray distributions
	cdef void** distributions_ptr
	cdef public float partition

	def __init__(self, distributions, name=None):
		self.distributions = numpy.array(distributions)
		if len(self.distributions) == 0:
			raise ValueError("Must pass in at least one distribution to initialize a Markov Network.")

		self.name = 'MarkovNetwork'
		self.d = len(set(numpy.concatenate([d.parents for d in distributions])))

	@property
	def structure(self):
		return tuple(tuple(d.parents) for d in self.distributions)

	def __dealloc__(self):
		free(self.parent_count)
		free(self.parent_idxs)

	def bake(self, calculate_partition=True):
		"""Finalize the topology of the underlying factor graph model.

		Assign a numerical index to every clique and create the underlying
		factor graph model. This method must be called before any of the 
		probability-calculating or inference methods because the probability
		calculating methods rely on the partition function and the inference
		methods rely on the factor graph.

		Parameters
		----------
		calculate_partition : bool, optional
			Whether to calculate the partition function. This is not necessary if
			the goal is simply to perform inference, but is required if the goal
			is to calculate the probability of examples under the model.

		Returns
		-------
		None
		"""

		# Initialize the factor graph
		self.graph = FactorGraph(self.name+'-fg')
		self.idxs = []
		self.keys_ = [None for i in range(self.d)]

		marginal_nodes = numpy.empty(self.d, dtype=object)
		factor_nodes = {}
		n = 0

		# Determine all marginal nodes and their distributions
		for i, d in enumerate(self.distributions):
			keys = numpy.array(d.keys(), dtype=object)

			for j, parent in enumerate(d.parents):
				keys_ = numpy.unique(keys[:,j])
				self.keys_[parent] = keys_

				d_ = DiscreteDistribution({key: 1. / len(keys_) for key in keys_})
				m = State(d_, str(parent)+"-marginal")
				marginal_nodes[parent] = m

		# Add each marginal node to the graph
		for i in range(self.d):
			self.graph.add_node(marginal_nodes[i])

		# Add each factor node to the graph with respective edges
		for i, d in enumerate(self.distributions):
			f = State(d.copy(), str(i)+"-joint")
			factor_nodes[i] = f
			self.graph.add_node(f)

			for j, parent in enumerate(d.parents):
				m = marginal_nodes[parent]
				self.graph.add_edge(m, f)

			idxs = tuple(d.parents)
			self.idxs.append(idxs)
			n += len(idxs)

			f.distribution.parents = [marginal_nodes[parent].distribution for parent in d.parents]
			d.n_columns = self.d

		# Finalize the factor graph structure
		self.graph.bake()

		#self.keymap = numpy.array([{key: i for i, key in enumerate(keys)} for keys in self.keymap])
		self.parent_count = <int*> calloc(self.d+1, sizeof(int))
		self.parent_idxs = <int*> calloc(n, sizeof(int))

		self.partition = float("inf")
		if calculate_partition == True:
			X_ = list(itertools.product(*self.keys_))
			self.partition = logsumexp(self.log_probability(X_, 
				unnormalized=True))

	def probability(self, X, n_jobs=1, unnormalized=False):
		"""Return the probability of samples under the Markov network.

		This is just a wrapper that exponentiates the result from the log
		probability method.

		Parameters
		----------
		X : array-like, shape (n_samples, n_dim)
			The sample is a vector of points where each dimension represents the
			same variable as added to the graph originally. It doesn't matter what
			the connections between these variables are, just that they are all
			ordered the same.

		n_jobs : int, optional
			The number of jobs to use to parallelize, either the number of threads
			or the number of processes to use. -1 means use all available resources.
			Default is 1.

		unnormalized : bool, optional
			Whether to return the unnormalized or normalized probabilities. The
			normalized probabilities requires the partition function to be
			calculated.

		Returns
		-------
		prob : numpy.ndarray or double
			The log probability of the samples if many, or the single log probability.
		"""

		return numpy.exp(self.log_probability(X, n_jobs=n_jobs, 
			unnormalized=unnormalized))

	def log_probability(self, X, n_jobs=1, unnormalized=False):
		"""Return the log probability of samples under the Markov network.

		The log probability is just the sum of the log probabilities under 
		each of the components minus the partition function. This method will 
		return a vector of log probabilities, one for each sample.

		Parameters
		----------
		X : array-like, shape (n_samples, n_dim)
			The sample is a vector of points where each dimension represents the
			same variable as added to the graph originally. It doesn't matter what
			the connections between these variables are, just that they are all
			ordered the same.

		n_jobs : int, optional
			The number of jobs to use to parallelize, either the number of threads
			or the number of processes to use. -1 means use all available resources.
			Default is 1.

		unnormalized : bool, optional
			Whether to return the unnormalized or normalized probabilities. The
			normalized probabilities requires the partition function to be
			calculated.

		Returns
		-------
		logp : numpy.ndarray or double
			The log probability of the samples if many, or the single log probability.
		"""

		if self.d == 0:
			raise ValueError("Must bake model before computing probability")
		if self.partition == float("inf") and unnormalized == False:
			raise ValueError("Must calculate partition before computing probability")

		X = numpy.array(X, ndmin=2, dtype=object)
		n, d = X.shape
		logp = numpy.zeros(n, dtype='float64')

		if unnormalized == False:
			logp -= self.partition

		for i in range(n):
			for j, d in enumerate(self.distributions):
				logp[i] += d.log_probability(X[i, self.idxs[j]])

		return logp if n > 1 else logp[0]

	cdef void _log_probability(self, double* symbol, double* log_probability, 
		int n) nogil:
		cdef int i, j, l, li, k
		cdef double logp
		cdef double* sym = <double*> malloc(self.d*sizeof(double))
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

			log_probability[i] -= self.partition

		free(sym)


	def marginal(self):
		"""Return the marginal probabilities of each variable in the graph.

		This is equivalent to a pass of belief propagation on a graph where
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

	def predict(self, X, max_iterations=100, n_jobs=1):
		"""Predict missing values of a data matrix using MLE.

		Impute the missing values of a data matrix using the maximally likely
		predictions according to the loopy belief propagation (also known as the
		forward-backward) algorithm. Run each example through the algorithm 
		(predict_proba) and replace missing values with the maximally likely 
		predicted emission.

		Parameters
		----------
		X : array-like, shape (n_samples, n_nodes)
			Data matrix to impute. Missing values must be either None (if lists)
			or np.nan (if numpy.ndarray). Will fill in these values with the
			maximally likely ones.

		max_iterations : int, optional
			Number of iterations to run loopy belief propagation for. Default
			is 100.

		n_jobs : int
			The number of jobs to use to parallelize, either the number of threads
			or the number of processes to use. -1 means use all available resources.
			Default is 1.

		Returns
		-------
		y_hat : numpy.ndarray, shape (n_samples, n_nodes)
			This is the data matrix with the missing values imputed.
		"""

		if self.d == 0:
			raise ValueError("Must bake model before calling predict.")

		y_hat = self.predict_proba(X, max_iterations=max_iterations,
			n_jobs=n_jobs)

		for i in range(len(y_hat)):
			for j in range(len(y_hat[i])):
				if isinstance(y_hat[i][j], Distribution):
					y_hat[i][j] = y_hat[i][j].mle()

		return y_hat

	def predict_proba(self, X, max_iterations=100, check_input=True, n_jobs=1):
		"""Returns the probabilities of each variable in the graph given evidence.

		This calculates the marginal probability distributions for each state given
		the evidence provided through loopy belief propagation. Loopy belief
		propagation is an approximate algorithm which is exact for certain graph
		structures.

		Parameters
		----------
		X : dict or array-like, shape <= n_nodes
			The evidence supplied to the graph. This can either be a dictionary
			with keys being state names and values being the observed values
			(either the emissions or a distribution over the emissions) or an
			array with the values being ordered according to the nodes 
			incorporation in the graph and None for variables which are unknown.
			It can also be vectorized, so a list of dictionaries can be passed 
			in where each dictionary is a single sample, or a list of lists where 
			each list is a single sample, both formatted as mentioned before. 
			The preferred method is as an numpy array.

		max_iterations : int, optional
			The number of iterations with which to do loopy belief propagation.
			Usually requires only 1. Default is 100.

		check_input : bool, optional
			Check to make sure that the observed symbol is a valid symbol for that
			distribution to produce. Default is True.

		n_jobs : int, optional
			The number of threads to use when parallelizing the job. This
			parameter is passed directly into joblib. Default is 1, indicating
			no parallelism.

		Returns
		-------
		y_hat : array-like, shape (n_samples, n_nodes)
			An array of univariate distribution objects showing the probabilities
			of each variable.
		"""

		if self.d == 0:
			raise ValueError("Must bake model before calling predict_proba")

		if isinstance(X, dict):
			return self.graph.predict_proba(X, max_iterations)

		elif isinstance(X, (list, numpy.ndarray)) and not isinstance(X[0],
			(list, numpy.ndarray, dict)):

			data = {}
			for i in range(self.d):
				if not _check_nan(X[i]):
					data[str(i)+"-marginal"] = X[i]

			return self.graph.predict_proba(data, max_iterations)

		else:
			y_hat = []
			for x in X:
				y_ = self.predict_proba(x, max_iterations=max_iterations,
					check_input=False, n_jobs=1)
				y_hat.append(y_)

			return y_hat


	def fit(self, X, weights=None, inertia=0.0, pseudocount=0.0, verbose=False,
		calculate_partition=True, n_jobs=1):
		"""Fit the model to data using MLE estimates.

		Fit the model to the data by updating each of the components of the model,
		which are univariate or multivariate distributions. This uses a simple
		MLE estimate to update the distributions according to their summarize or
		fit methods.

		This is a wrapper for the summarize and from_summaries methods.

		Parameters
		----------
		X : array-like, shape (n_samples, n_nodes)
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

		verbose : bool, optional
			Whether or not to print out improvement information over
			iterations. Only required if doing semisupervised learning.
			Default is False.

		calculate_partition : bool, optional
			Whether to calculate the partition function. This is not necessary if
			the goal is simply to perform inference, but is required if the goal
			is to calculate the probability of examples under the model.

		n_jobs : int
			The number of jobs to use to parallelize, either the number of threads
			or the number of processes to use. -1 means use all available resources.
			Default is 1.

		Returns
		-------
		self : MarkovNetwork
			The fit Markov network object with updated model parameters.
		"""

		training_start_time = time.time()

		if weights is None:
			weights = numpy.ones(len(X), dtype='float64')
		else:
			weights = numpy.array(weights, dtype='float64')

		starts = [int(i*len(X)/n_jobs) for i in range(n_jobs)]
		ends = [int(i*len(X)/n_jobs) for i in range(1, n_jobs+1)]

		with Parallel(n_jobs=n_jobs, backend='threading') as parallel:
			parallel( delayed(self.summarize, check_pickle=False)(
				X[start:end], weights[start:end]) for start, end in zip(starts, ends))

		self.from_summaries(inertia, pseudocount)
		self.bake(calculate_partition=calculate_partition)

		if verbose:
			total_time_spent = time.time() - training_start_time
			print("Total Time (s): {:.4f}".format(total_time_spent))

		return self

	def summarize(self, X, weights=None):
		"""Summarize a batch of data and store the sufficient statistics.

		This will partition the dataset into columns which belong to their
		appropriate distribution. If the distribution has parents, then multiple
		columns are sent to the distribution. This relies mostly on the summarize
		function of the underlying distribution.

		Parameters
		----------
		X : array-like, shape (n_samples, n_nodes)
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

		n, d = len(X), len(X[0])
		X_int = numpy.empty((n, d), dtype='float64')

		for i in range(n):
			for j in range(d):
				if X[i][j] == 'nan' or X[i][j] == None or X[i][j] == nan:
					X_int[i, j] = nan
				else:
					X_int[i, j] = self.keymap[j][X[i][j]]

		if weights is None:
			weights = numpy.ones(n, dtype='float64')
		else:
			weights = numpy.array(weights, dtype='float64')

		for i, d in enumerate(self.distributions):
			d.summarize(X, weights)

	def from_summaries(self, inertia=0.0, pseudocount=0.0,
		calculate_partition=True):
		"""Use MLE on the stored sufficient statistics to train the model.

		Parameters
		----------
		inertia : double, optional
			The inertia for updating the distributions, passed along to the
			distribution method. Default is 0.0.

		pseudocount : double, optional
			A pseudocount to add to the emission of each distribution. This
			effectively smoothes the states to prevent 0. probability symbols
			if they don't happen to occur in the data. Default is 0.

		calculate_partition : bool, optional
			Whether to calculate the partition function. This is not necessary if
			the goal is simply to perform inference, but is required if the goal
			is to calculate the probability of examples under the model.

		Returns
		-------
		None
		"""

		for d in self.distributions:
			d.from_summaries(inertia, pseudocount)

		self.bake(calculate_partition=calculate_partition)

	def to_json(self, separators=(',', ' : '), indent=4):
		"""Serialize the model to a JSON.

		Parameters
		----------
		separators : tuple, optional
			The two separators to pass to the json.dumps function for formatting.

		indent : int, optional
			The indentation to use at each level. Passed to json.dumps for
			formatting.

		Returns
		-------
		json : str
			A properly formatted JSON object.
		"""

		states = [distribution.copy() for distribution in self.distributions]

		model = {
					'class' : 'MarkovNetwork',
					'name'  : self.name,
					'distributions' : [json.loads(d.to_json()) 
						for d in self.distributions]
		}

		return json.dumps(model, separators=separators, indent=indent)

	@classmethod
	def from_json(cls, s):
		"""Read in a serialized Markov Network and return the appropriate object.

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
			d = json.loads(s)
		except:
			try:
				with open(s, 'r') as infile:
					d = json.load(infile)
			except:
				raise IOError("String must be properly formatted JSON or filename of properly formatted JSON.")

		distributions = []
		for j in d['distributions']:
			distribution = JointProbabilityTable.from_json(json.dumps(j))
			distributions.append(distribution)

		model = cls(distributions, str(d['name']))
		model.bake()
		return model

	@classmethod
	def from_structure(cls, X, structure, weights=None, pseudocount=0.0,
		name=None, calculate_partition=True):
		"""Return a Markov network from a predefined structure.

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

		Returns
		-------
		model : MarkovNetwork
			A Markov network with the specified structure.
		"""

		X = numpy.array(X)
		n, d = X.shape
		distributions = []

		if weights is None:
			weights = numpy.ones(X.shape[0], dtype='float64')
		else:
			weights = numpy.array(weights, dtype='float64')

		for i, parents in enumerate(structure):
			distribution = JointProbabilityTable.from_samples(X[:, parents],
				parents=parents, weights=weights, pseudocount=pseudocount)
			distributions.append(distribution)

		model = cls(distributions)
		model.bake(calculate_partition=calculate_partition)
		return model

	@classmethod
	def from_samples(cls, X, weights=None, algorithm='chow-liu', max_parents=-1,
		 pseudocount=0.0, name=None, reduce_dataset=True, 
		 calculate_partition=True, n_jobs=1):
		"""Learn the structure of the network from data.

		Find the structure of the network from data using a Markov structure
		learning score. This currently enumerates all the exponential number of
		structures and finds the best according to the score. This allows
		weights on the different samples as well. The score that is optimized
		is the minimum description length (MDL).

		If not all states for a variable appear in the supplied data, this
		function can not gurantee that the returned Markov Network is optimal
		when 'exact' or 'exact-dp' is used. This is because the number of
		states for each node is derived only from the data provided, and the
		scoring function depends on the number of states of a variable.

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

		name : str, optional
			The name of the model. Default is None.

		reduce_dataset : bool, optional
			Given the discrete nature of these datasets, frequently a user
			will pass in a dataset that has many identical samples. It is time
			consuming to go through these redundant samples and a far more
			efficient use of time to simply calculate a new dataset comprised
			of the subset of unique observed samples weighted by the number of
			times they occur in the dataset. This typically will speed up all
			algorithms, including when using a constraint graph. Default is
			True.

		n_jobs : int, optional
			The number of threads to use when learning the structure of the
			network. If a constraint graph is provided, this will parallelize
			the tasks as directed by the constraint graph. If one is not
			provided it will parallelize the building of the parent graphs.
			Both cases will provide large speed gains.

		Returns
		-------
		model : MarkovNetwork
			The learned Markov Network.
		"""

		X = numpy.array(X)
		n, d = X.shape

		keys = [set([x for x in X[:,i] if not _check_nan(x)]) for i in range(d)]
		keymap = numpy.array([{key: i for i, key in enumerate(keys[j])} for j in range(d)])
		key_count = numpy.array([len(keymap[i]) for i in range(d)], dtype='int32')

		if weights is None:
			weights = numpy.ones(X.shape[0], dtype='float64')
		else:
			weights = numpy.array(weights, dtype='float64')

		if reduce_dataset:
			X_count = {}

			for x, weight in zip(X, weights):
				# Convert NaN to None because two tuples containing
				# (1.0, 2.0, 3.0, nan) are not considered equal, but two tuples
				# containing (1.0, 2.0, 3.0, None) are considered equal
				x = tuple(None if isnan(xn) else xn for xn in x)
				if x in X_count:
					X_count[x] += weight
				else:
					X_count[x] = weight

			weights = numpy.array(list(X_count.values()), dtype='float64')
			X = numpy.array(list(X_count.keys()), dtype=X.dtype)
			n, d = X.shape


		X_int = numpy.zeros((n, d), dtype='float64')
		for i in range(n):
			for j in range(d):
				if _check_nan(X[i, j]):
					X_int[i, j] = nan
				else:
					X_int[i, j] = keymap[j][X[i, j]]

		w_sum = weights.sum()

		if max_parents == -1 or max_parents > _log(2*w_sum / _log(w_sum)):
			max_parents = int(_log(2*w_sum / _log(w_sum)))

		if algorithm == 'chow-liu':
			if numpy.any(numpy.isnan(X_int)):
				raise ValueError("Chow-Liu tree learning does not current support missing values")

			structure = discrete_chow_liu_tree(X_int, weights, key_count,
				pseudocount)
		else:
			raise ValueError("Invalid algorithm type passed in. Must be one of 'chow-liu', 'exact', 'exact-dp', 'greedy'")

		return cls.from_structure(X, structure=structure, 
			weights=weights, pseudocount=pseudocount, name=name,
			calculate_partition=calculate_partition)

def discrete_chow_liu_tree(numpy.ndarray X_ndarray, numpy.ndarray weights_ndarray,
	numpy.ndarray key_count_ndarray, double pseudocount):
	"""Find the Chow-Liu tree that spans a data set.

	The Chow-Liu algorithm first calculates the mutual information between each
	pair of variables and then constructs a maximum spanning tree given that.
	This algorithm slightly from the one implemented for Bayesian networks
	because Bayesian networks are directed and need a node to be the root.
	In contrast, the structure here is undirected and so is a simple maximum
	spanning tree.
	"""

	cdef int i, j, k, l, lj, lk, Xj, Xk, xj, xk
	cdef int n = X_ndarray.shape[0], d = X_ndarray.shape[1]
	cdef int max_keys = key_count_ndarray.max()

	cdef double* X = <double*> X_ndarray.data
	cdef double* weights = <double*> weights_ndarray.data
	cdef int* key_count = <int*> key_count_ndarray.data

	cdef double* mutual_info = <double*> calloc(d * d, sizeof(double))

	cdef double* marg_j = <double*> malloc(max_keys*sizeof(double))
	cdef double* marg_k = <double*> malloc(max_keys*sizeof(double))
	cdef double* joint_count = <double*> malloc(max_keys**2*sizeof(double))

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
				Xj = <int> X[i*d + j]
				Xk = <int> X[i*d + k]

				joint_count[Xj * lk + Xk] += weights[i]
				marg_j[Xj] += weights[i]
				marg_k[Xk] += weights[i]

			for xj in range(lj):
				for xk in range(lk):
					if joint_count[xj*lk+xk] > 0:
						mutual_info[j*d + k] -= joint_count[xj*lk+xk] * _log(
							joint_count[xj*lk+xk] / (marg_j[xj] * marg_k[xk]))
						mutual_info[k*d + j] = mutual_info[j*d + k]

	structure = []

	cdef int x, y, min_x, min_y
	cdef double min_score, score

	for x in range(d):
		min_score = float("inf")
		min_x = -1
		min_y = -1

		for y in range(d):
			if x == y:
				continue

			score = mutual_info[x*d + y]
			if score < min_score:
				min_score = score
				min_x = x
				min_y = y

	structure.append([min_x, min_y])

	visited = [min_y, min_x]
	unvisited = list(range(d))
	unvisited.remove(min_y)
	unvisited.remove(min_x)

	for i in range(d-2):
		min_score = float("inf")
		min_x = -1
		min_y = -1

		for x in visited:
			for y in unvisited:
				score = mutual_info[x*d + y]
				if score < min_score:
					min_score = score
					min_x = x
					min_y = y

		structure.append([min_x, min_y])
		visited.append(min_y)
		unvisited.remove(min_y)

	free(mutual_info)
	free(marg_j)
	free(marg_k)
	free(joint_count)
	return tuple([tuple(neighbors) for neighbors in structure])

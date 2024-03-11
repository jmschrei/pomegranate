# factor_graph.py
# Author: Jacob Schreiber <jmschreiber91@gmail.com>

import time
import torch
import itertools

from ._utils import _cast_as_tensor
from ._utils import _update_parameter
from ._utils import _check_parameter
from ._utils import _reshape_weights

from .distributions._distribution import Distribution
from .distributions._distribution import ConditionalDistribution

from .distributions import Categorical
from .distributions import JointCategorical


class FactorGraph(Distribution):
	"""A factor graph object.

	A factor graph represents a probability distribution as a bipartite graph
	where marginal distributions of each dimension in the distribution are on
	one side of the graph and factors are on the other side. The distributions
	on the factor side encode probability estimates from the model, whereas the
	distributions on the marginal side encode probability estimates from the
	data. 

	Inference is done on the factor graph using the loopy belief propagation
	algorithm. This is an iterative algorithm where "messages" are passed
	along each edge between the marginals and the factors until the estimates
	for the marginals converges. In brief: each message represents what the
	generating node thinks its marginal distribution is with respect to the
	child node. Calculating each message involves marginalizing the parent
	node with respect to *every other* node. When the parent node is already
	a univariate distribution -- either because it is a marginal node or
	a univariate factor node -- no marginalization is needed and it sends
	itself as the message. Basically, a joint probability table will receive
	messages from all the marginal nodes that comprise its dimensions and,
	to each of those marginal nodes, it will send a message back saying what
	it (the joint probability table) thinks its marginal distribution is with
	respect to the messages from the OTHER marginals. More concretely, if a
	joint probability table has two dimensions with marginal node parents
	A and B, it will send a message to A that is itself after marginalizing out
	B, and will send a message to B that is itself after marginalizing out A. 

	..note:: It is worth noting that this algorithm is exact when the structure
	is a tree. If there exist any loops in the factors, i.e., you can draw a
	circle beginning with a factor and then hopping between marginals and
	factors and make it back to the factor without crossing any edges twice,
	the probabilities returned are approximate.


	Parameters
	----------
	factors: tuple or list or None
		A set of distribution objects. These do not need to be initialized,
		i.e. can be "Categorical()". Currently, they must be either Categorical
		or JointCategorical distributions. Default is None.

	marginals: tuple or list or None
		A set of distribution objects. These must be initialized and be
		Categorical distributions.

	edges: list or tuple or None
		A set of edges. Critically, the items in this list must be the
		distribution objects themselves, and the order that edges must match
		the order distributions in a multivariate distribution. Specifically,
		if you have a multivariate distribution, the first edge that includes
		it must correspond to the first dimension, the second edge must
		correspond to the second dimension, etc, and the total number of
		edges cannot exceed the number of dimensions. Default is None.

	max_iter: int, optional
		The number of iterations to do in the inference step as distributions
		are converging. Default is 10.

	tol: float, optional
		The threshold at which to stop during fitting when the improvement
		goes under. Default is 1e-6.

	inertia: float, [0, 1], optional
		Indicates the proportion of the update to apply to the parameters
		during training. When the inertia is 0.0, the update is applied in
		its entirety and the previous parameters are ignored. When the
		inertia is 1.0, the update is entirely ignored and the previous
		parameters are kept, equivalently to if the parameters were frozen.

	frozen: bool, optional
		Whether all the parameters associated with this distribution are frozen.
		If you want to freeze individual pameters, or individual values in those
		parameters, you must modify the `frozen` attribute of the tensor or
		parameter directly. Default is False.

	check_data: bool, optional
		Whether to check properties of the data and potentially recast it to
		torch.tensors. This does not prevent checking of parameters but can
		slightly speed up computation when you know that your inputs are valid.
		Setting this to False is also necessary for compiling.

	verbose: bool, optional
		Whether to print the improvement and timings during training.
	"""

	def __init__(self, factors=None, marginals=None, edges=None, max_iter=20, 
		tol=1e-6, inertia=0.0, frozen=False, check_data=True, verbose=False):
		super().__init__(inertia=inertia, frozen=frozen, check_data=check_data)
		self.name = "FactorGraph"

		self.factors = torch.nn.ModuleList([])
		self.marginals = torch.nn.ModuleList([])

		self._factor_idxs = {}
		self._marginal_idxs = {}

		self._factor_edges = []
		self._marginal_edges = []

		self.max_iter = _check_parameter(max_iter, "max_iter", min_value=1, 
			dtypes=[int, torch.int16, torch.int32, torch.int64])
		self.tol = _check_parameter(tol, "tol", min_value=0)
		self.verbose = verbose

		self.d = 0
		self._initialized = factors is not None and factors[0]._initialized

		if factors is not None:
			_check_parameter(factors, "factors", dtypes=(list, tuple))

			for factor in factors:
				self.add_factor(factor)

		if marginals is not None:
			_check_parameter(marginals, "marginals", dtypes=(list, tuple))

			for marginal in marginals:
				self.add_marginal(marginal)

		if edges is not None:
			_check_parameter(edges, "edges", dtypes=(list, tuple))

			for marginal, factor in edges:
				self.add_edge(marginal, factor)

		self._initialized = not factors

	def _initialize(self, d):
		self._initialized = True
		super()._initialize(d)

	def _reset_cache(self):
		return

	def add_factor(self, distribution):
		"""Adds a distribution to the set of factors.

		
		Parameters
		----------
		distribution: pomegranate.distributions.Distribution
			A distribution object to include as a node.
		"""

		if not isinstance(distribution, (Categorical, JointCategorical)):
			raise ValueError("Must be a Categorical or a JointCategorical"
				" distribution.")

		self.factors.append(distribution)
		self._factor_edges.append([])
		self._factor_idxs[distribution] = len(self.factors) - 1

		self._initialized = distribution._initialized

	def add_marginal(self, distribution):
		"""Adds a distribution to the set of marginals.

		This adds a distribution to the marginal side of the bipartate graph.
		This distribution must be univariate. 

		Parameters
		----------
		distribution: pomegranate.distributions.Distribution
			A distribution object to include as a node.
		"""

		if not isinstance(distribution, Categorical):
			raise ValueError("Must be a Categorical distribution.")
		
		self.marginals.append(distribution)
		self._marginal_edges.append([])
		self._marginal_idxs[distribution] = len(self.marginals) - 1

		self.d += 1

	def add_edge(self, marginal, factor):
		"""Adds an undirected edge to the set of edges.

		Because a factor graph is a bipartite graph, one of the edges must be
		a marginal distribution and the other edge must be a factor 
		distribution.

		Parameters
		----------
		marginal: pomegranate.distributions.Distribution
			The marginal distribution to include in the edge.

		factor: pomegranate.distributions.Distribution
			The factor distribution to include in the edge.
		"""

		if marginal not in self._marginal_idxs:
			raise ValueError("Marginal distribution does not exist in graph.")
		
		if factor not in self._factor_idxs:
			raise ValueError("Factor distribution does not exist in graph.")

		m_idx = self._marginal_idxs[marginal]
		f_idx = self._factor_idxs[factor]

		self._factor_edges[f_idx].append(m_idx)
		self._marginal_edges[m_idx].append(f_idx)

	def log_probability(self, X):
		"""Calculate the log probability of each example.

		This method calculates the log probability of each example given the
		parameters of the distribution. The examples must be given in a 2D
		format.


		Parameters
		----------
		X: list, tuple, numpy.ndarray, torch.Tensor, shape=(-1, self.d)
			A set of examples to evaluate.

		Returns
		-------
		logp: torch.Tensor, shape=(-1,)
			The log probability of each example.
		"""

		X = _check_parameter(_cast_as_tensor(X), "X", ndim=2, 
			check_parameter=self.check_data)
		
		logps = torch.zeros(X.shape[0], device=X.device, dtype=torch.float32)
		for idxs, factor in zip(self._factor_edges, self.factors):
			logps += factor.log_probability(X[:, idxs])

		for i, marginal in enumerate(self.marginals):
			logps += marginal.log_probability(X[:, i:i+1])

		return logps

	def predict(self, X):
		"""Infers the maximum likelihood value for each missing value.

		This method infers a probability distribution for each of the missing
		values in the data. First, the sum-product algorithm is run to infer
		a probability distribution for each variable. Then, the maximum
		likelihood value is returned from that distribution.

		The input to this method must be a torch.masked.MaskedTensor where the
		mask specifies which variables are observed (mask = True) and which ones
		are not observed (mask = False) for each of the values. When setting
		mask = False, it does not matter what the corresponding value in the
		tensor is. Different sets of variables can be observed or missing in
		different examples. 

		Unlike the `predict_proba` and `predict_log_proba` methods, this
		method preserves the dimensions of the original data because it does
		not matter how many categories a variable can take when you're only
		returning the maximally likely one.


		Parameters
		----------
		X: torch.masked.MaskedTensor
			A masked tensor where the observed values are available and the
			unobserved values are missing, i.e., the mask is True for
			observed values and the mask is False for missing values. It does
			not matter what the underlying value in the tensor is for the 
			missing values.
		"""

		y = [t.argmax(dim=1) for t in self.predict_proba(X)]
		return torch.vstack(y).T.contiguous()

	def predict_proba(self, X):
		"""Predict the probability of each variable given some evidence.

		Given some evidence about the value that each variable takes, infer
		the value that each other variable takes. If no evidence is given,
		this returns the marginal value of each variable given the dependence
		structure in the network.

		Currently, only hard evidence is supported, where the evidence takes
		the form of a discrete value. The evidence is represented as a
		masked tensor where the masked out values are considered missing.


		Parameters
		----------
		X: torch.masked.MaskedTensor
			A masked tensor where the observed values are available and the
			unobserved values are missing, i.e., the mask is True for
			observed values and the mask is False for missing values. It does
			not matter what the underlying value in the tensor is for the 
			missing values.
		"""

		nm = len(self.marginals)
		nf = len(self.factors)

		if X.shape[1] != nm:
			raise ValueError("X.shape[1] must match the number of marginals.")

		# Pull out the current distributions from factors and marginals
		factors = []
		marginals = []
		prior_marginals = []
		current_marginals = []

		for i, m in enumerate(self.marginals):
			p = torch.clone(m.probs[0])
			p = p.repeat((X.shape[0],) + tuple(1 for _ in p.shape))

			# Use the evidence in the data to set marginal distributions
			for j in range(X.shape[0]):
				if X._masked_mask[j, i] == True:
					value = X._masked_data[j, i]

					p[j] = 0
					p[j, value] = 1.0

			marginals.append(p)
			prior_marginals.append(torch.clone(p))
			current_marginals.append(torch.clone(p))

		for i, f in enumerate(self.factors):
			if not isinstance(f, Categorical):
				p = torch.clone(f.probs)
			else:
				p = torch.clone(f.probs[0])

			p = p.repeat((X.shape[0],) + tuple(1 for _ in p.shape))
			factors.append(p)

		# Set the original in and out messages along the edges
		in_messages, out_messages = [], []

		for i, m in enumerate(marginals):
			k = len(self._marginal_edges[i])
			in_messages.append([])

			for j in range(k):
				in_messages[-1].append(m)

		for i in range(len(factors)):
			k = len(self._factor_edges[i])
			out_messages.append([])
			
			for j in range(k):
				marginal_idx = self._factor_edges[i][j]
				d_j = marginals[marginal_idx]
				out_messages[-1].append(d_j)


		# Begin iterations
		iteration = 0
		while iteration < self.max_iter:
			# Update the messages going into the nodes.
			for i, f in enumerate(factors):
				ni_edges = len(self._factor_edges[i])

				for k in range(ni_edges):
					message = torch.clone(f)

					shape = torch.ones(len(message.shape), dtype=torch.int32)
					shape[0] = X.shape[0]

					for l in range(ni_edges):
						if k == l:
							continue

						shape[l+1] = message.shape[l+1]						
						message *= out_messages[i][l].reshape(*shape)
						message = torch.sum(message, dim=l+1, keepdims=True)
						shape[l+1] = 1

					else:
						message = message.squeeze()
						if len(message.shape) == 1:
							message = message.unsqueeze(0)

					j = self._factor_edges[i][k]
					for ik, parent in enumerate(self._marginal_edges[j]):
						if parent == i:
							dims = tuple(range(1, len(message.shape)))
							in_messages[j][ik] = message / message.sum(
								dim=dims, keepdims=True)

							break


			# Calculate the current estimates of the marginals
			loss = 0
			for i, m in enumerate(marginals):
				current_marginals[i] = torch.clone(m)

				for k in range(len(self._marginal_edges[i])):
					current_marginals[i] *= in_messages[i][k]


				dims = tuple(range(1, len(current_marginals[i].shape)))
				current_marginals[i] /= current_marginals[i].sum(dim=dims, 
					keepdims=True)

				loss += torch.nn.KLDivLoss(reduction="batchmean")(torch.log(
					current_marginals[i] + 1e-8), prior_marginals[i])

			if self.verbose:
				print(iteration, loss.item())

			if loss < self.tol:
				break


			# Update the messages leaving based on the new marginals
			for i, m in enumerate(marginals):
				ni_edges = len(self._marginal_edges[i])

				for k in range(ni_edges):
					message = torch.clone(m)

					for l in range(ni_edges):
						if k == l:
							continue

						message *= in_messages[i][l]

					j = self._marginal_edges[i][k]
					for ik, parent in enumerate(self._factor_edges[j]):
						if parent == i:
							dims = tuple(range(1, len(message.shape)))
							out_messages[j][ik] = message / message.sum(
								dim=dims, keepdims=True)
							break

			prior_marginals = [torch.clone(d) for d in current_marginals]
			iteration += 1


		return current_marginals

	def predict_log_proba(self, X):
		"""Infers the probability of each category given the model and data.

		This method is a wrapper around the `predict_proba` method and simply
		takes the log of each returned tensor.

		This method infers a log probability distribution for each of the 
		missing  values in the data. It uses the factor graph representation of 
		the Bayesian network to run the sum-product/loopy belief propagation
		algorithm.

		The input to this method must be a torch.masked.MaskedTensor where the
		mask specifies which variables are observed (mask = True) and which ones
		are not observed (mask = False) for each of the values. When setting
		mask = False, it does not matter what the corresponding value in the
		tensor is. Different sets of variables can be observed or missing in
		different examples. 

		An important note is that, because each variable can have a different
		number of categories in the categorical setting, the return is a list
		of tensors where each element in that list is the marginal probability
		distribution for that variable. More concretely: the first element will
		be the distribution of values for the first variable across all
		examples. When the first variable has been provided as evidence, the
		distribution will be clamped to the value provided as evidence.

		..warning:: This inference is exact given a Bayesian network that has
		a tree-like structure, but is only approximate for other cases. When
		the network is acyclic, this procedure will converge, but if the graph
		contains cycles then there is no guarantee on convergence.


		Parameters
		----------
		X: torch.masked.MaskedTensor, shape=(-1, d)
			The data to predict values for. The mask should correspond to
			whether the variable is observed in the example. 
		

		Returns
		-------
		y: list of tensors, shape=(d,)
			A list of tensors where each tensor contains the distribution of
			values for that dimension.
		"""

		return [torch.log(t) for t in self.predict_proba(X)]

	def fit(self, X, sample_weight=None):
		"""Fit the factors of the model to optionally weighted examples.

		This method will fit the provided factor distributions to the given
		data and their optional weights. It will not update the marginal
		distributions, as that information is already encoded in the joint
		probabilities.

		..note:: A structure must already be provided. Currently, structure
		learning of factor graphs is not supported.


		Parameters
		----------
		X: list, tuple, numpy.ndarray, torch.Tensor, shape=(-1, self.d)
			A set of examples to evaluate. 

		sample_weight: list, tuple, numpy.ndarray, torch.Tensor, optional
			A set of weights for the examples. This can be either of shape
			(-1, self.d) or a vector of shape (-1,). Default is ones.


		Returns
		-------
		self
		"""

		self.summarize(X, sample_weight=sample_weight)
		self.from_summaries()
		return self

	def summarize(self, X, sample_weight=None):
		"""Extract the sufficient statistics from a batch of data.

		This method calculates the sufficient statistics from optionally
		weighted data and adds them to the stored cache for each distribution
		in the network. Sample weights can either be provided as one
		value per example or as a 2D matrix of weights for each feature in
		each example.


		Parameters
		----------
		X: list, tuple, numpy.ndarray, torch.Tensor, shape=(-1, len, self.d)
			A set of examples to summarize.

		sample_weight: list, tuple, numpy.ndarray, torch.Tensor, optional
			A set of weights for the examples. This can be either of shape
			(-1, self.d) or a vector of shape (-1,). Default is ones.


		Returns
		-------
		logp: torch.Tensor, shape=(-1,)
			The log probability of each example.
		"""

		if self.frozen:
			return

		X, sample_weight = super().summarize(X, sample_weight=sample_weight)
		X = _check_parameter(X, "X", ndim=2, check_parameter=self.check_data)

		for i, factor in enumerate(self.factors):
			factor.summarize(X[:, self._factor_edges[i]], 
				sample_weight=sample_weight[:,i])

	def from_summaries(self):
		if self.frozen:
			return

		for distribution in self.factors:
			distribution.from_summaries()

# markov_chain.py
# Author: Jacob Schreiber <jmschreiber91@gmail.com>

import torch

from ._utils import _cast_as_tensor
from ._utils import _update_parameter
from ._utils import _check_parameter
from ._utils import _reshape_weights

from .distributions._distribution import Distribution
from .distributions import Categorical
from .distributions import ConditionalCategorical


class MarkovChain(Distribution):
	"""A Markov chain.

	A Markov chain is the simplest sequential model which factorizes the
	joint probability distribution P(X_{0} ... X_{t}) along a chain into the
	product of a marginal distribution P(X_{0}) P(X_{1} | X_{0}) ... with
	k conditional probability distributions for a k-th order Markov chain.

	Despite sometimes being thought of as an independent model, Markov chains
	are probability distributions over sequences just like hidden Markov
	models. Because a Markov chain has the same theoretical properties as a
	probability distribution, it can be used in any situation that a simpler 
	distribution could, such as an emission distribution for a HMM or a 
	component of a Bayes classifier.


	Parameters
	----------
	distributions: tuple or list or None
		A set of distribution objects. These objects do not need to be
		initialized, i.e., can be "Categorical()". 

	k: int or None
		The number of conditional distributions to include in the chain, also
		the number of steps back to model in the sequence. This must be passed
		in if the distributions are not passed in.

	n_categories: list, tuple, or None
		A list or tuple containing the number of categories that each feature
		has. 

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
	"""

	def __init__(self, distributions=None, k=None, n_categories=None, 
		inertia=0.0, frozen=False, check_data=True):
		super().__init__(inertia=inertia, frozen=frozen, check_data=check_data)
		self.name = "MarkovChain"

		self.distributions = _check_parameter(distributions, "distributions",
			dtypes=(list, tuple))
		self.k = _check_parameter(_cast_as_tensor(k, dtype=torch.int32), "k",
			ndim=0)
		self.n_categories = _check_parameter(n_categories, "n_categories",
			dtypes=(list, tuple))

		if distributions is None and k is None:
			raise ValueError("Must provide one of 'distributions', or 'k'.")

		if distributions is not None:
			self.k = len(distributions) - 1

		self.d = None
		self._initialized = distributions is not None and distributions[0]._initialized
		self._reset_cache()

	def _initialize(self, d, n_categories):
		"""Initialize the probability distribution.

		This method is meant to only be called internally. It initializes the
		parameters of the distribution and stores its dimensionality. For more
		complex methods, this function will do more.


		Parameters
		----------
		d: int
			The dimensionality the distribution is being initialized to.

		n_categories: int
			The maximum number of categories to model. This single number is
			used as the maximum across all features and all timesteps.
		"""

		if self.distributions is None:
			self.distributions = [Categorical()]
			self.distributions[0]._initialize(d, max(n_categories))

			for i in range(self.k):
				distribution = ConditionalCategorical()
				distribution._initialize(d, [[n_categories[j]]*(i+2) 
					for j in range(d)])

				self.distributions.append(distribution)

		self.n_categories = n_categories
		self._initialized = True
		super()._initialize(d)

	def _reset_cache(self):
		"""Reset the internally stored statistics.

		This method is meant to only be called internally. It resets the
		stored statistics used to update the model parameters as well as
		recalculates the cached values meant to speed up log probability
		calculations.
		"""

		if self._initialized:
			for distribution in self.distributions:
				distribution._reset_cache()

	def sample(self, n):
		"""Sample from the probability distribution.

		This method will return `n` samples generated from the underlying
		probability distribution. For a mixture model, this involves first
		sampling the component using the prior probabilities, and then sampling
		from the chosen distribution.


		Parameters
		----------
		n: int
			The number of samples to generate.
		

		Returns
		-------
		X: torch.tensor, shape=(n, self.d)
			Randomly generated samples.
		"""

		X = [self.distributions[0].sample(n)]

		for distribution in self.distributions[1:]:
			X_ = torch.stack(X).permute(1, 0, 2)
			samples = distribution.sample(n, X_[:, -self.k-1:])
			X.append(samples)

		return torch.stack(X).permute(1, 0, 2)

	def log_probability(self, X):
		"""Calculate the log probability of each example.

		This method calculates the log probability of each example given the
		parameters of the distribution. The examples must be given in a 3D
		format.


		Parameters
		----------
		X: list, tuple, numpy.ndarray, torch.Tensor, shape=(-1, length, self.d)
			A set of examples to evaluate.

		Returns
		-------
		logp: torch.Tensor, shape=(-1,)
			The log probability of each example.
		"""


		X = _check_parameter(_cast_as_tensor(X), "X", ndim=3, 
			check_parameter=self.check_data)
		self.d = X.shape[1]

		logps = self.distributions[0].log_probability(X[:, 0])
		for i, distribution in enumerate(self.distributions[1:-1]):
			logps += distribution.log_probability(X[:, :i+2])

		for i in range(X.shape[1] - self.k):
			j = i + self.k + 1
			logps += self.distributions[-1].log_probability(X[:, i:j])

		return logps

	def fit(self, X, sample_weight=None):
		"""Fit the model to optionally weighted examples.

		This method will fit the provided distributions given the data and
		their weights. If only `k` has been provided, the relevant set of
		distributions will be initialized.


		Parameters
		----------
		X: list, tuple, numpy.ndarray, torch.Tensor, shape=(-1, length, self.d)
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
		X: list, tuple, numpy.ndarray, torch.Tensor, shape=(-1, length, self.d)
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

		X = _check_parameter(_cast_as_tensor(X), "X", ndim=3, 
			check_parameter=self.check_data)
		sample_weight = _check_parameter(_cast_as_tensor(sample_weight), 
			"sample_weight", min_value=0, ndim=(1, 2), 
			check_parameter=self.check_data)

		if not self._initialized:
			if self.n_categories is not None:
				n_keys = self.n_categories
			elif isinstance(X, torch.masked.MaskedTensor):
				n_keys = (torch.max(torch.max(X._masked_data, dim=0)[0], 
					dim=0)[0] + 1).type(torch.int32)
			else:
				n_keys = (torch.max(torch.max(X, dim=0)[0], dim=0)[0] + 1).type(
					torch.int32)

			self._initialize(len(X[0][0]), n_keys)

		if sample_weight is None:
			sample_weight = torch.ones_like(X[:, 0])
		elif len(sample_weight.shape) == 1: 
			sample_weight = sample_weight.reshape(-1, 1).expand(-1, X.shape[2])
		elif sample_weight.shape[1] == 1:
			sample_weight = sample_weight.expand(-1, X.shape[2])

		_check_parameter(_cast_as_tensor(sample_weight), "sample_weight", 
			min_value=0, ndim=2, shape=(X.shape[0], X.shape[2]), 
			check_parameter=self.check_data)

		self.distributions[0].summarize(X[:, 0], sample_weight=sample_weight)
		for i, distribution in enumerate(self.distributions[1:-1]):
			distribution.summarize(X[:, :i+2], sample_weight=sample_weight)

		distribution = self.distributions[-1]
		for i in range(X.shape[1] - self.k):
			j = i + self.k + 1
			distribution.summarize(X[:, i:j], sample_weight=sample_weight)

	def from_summaries(self):
		"""Update the model parameters given the extracted statistics.

		This method uses calculated statistics from calls to the `summarize`
		method to update the distribution parameters. Hyperparameters for the
		update are passed in at initialization time.

		Note: Internally, a call to `fit` is just a successive call to the
		`summarize` method followed by the `from_summaries` method.
		"""

		if self.frozen:
			return

		for distribution in self.distributions:
			distribution.from_summaries()

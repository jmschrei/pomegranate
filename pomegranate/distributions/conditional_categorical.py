# conditional_categorical.py
# Contact: Jacob Schreiber <jmschreiber91@gmail.com>

import numpy
import torch
import itertools

from .._utils import _cast_as_tensor
from .._utils import _cast_as_parameter
from .._utils import _update_parameter
from .._utils import _check_parameter
from .._utils import _reshape_weights

from .._utils import BufferList

from ._distribution import ConditionalDistribution
from .categorical import Categorical


class ConditionalCategorical(ConditionalDistribution):
	"""A conditional categorical distribution.

	This is a categorical distribution that is conditioned on previous
	emissions, meaning that the probability of each character depends on the
	observed character earlier in the sequence. Each feature is conditioned
	independently of the others like a `Categorical` distribution. 

	This conditioning makes the shape of the distribution a bit more
	complicated than the `JointCategorical` distribution. Specifically, a 
	`JointCategorical` distribution is multivariate by definition but a
	`ConditionalCategorical` does not have to be. Although both may appear 
	similar in that they both take in a vector of characters and return 
	probabilities, the vector fed into the JointCategorical are all observed 
	together without some notion of time, whereas the ConditionalCategorical 
	explicitly requires a notion of timing, where the probability of later 
	characters depend on the composition of characters seen before.


	Parameters
	----------
	probs: list of numpy.ndarray, torch.tensor or None, shape=(k, k), optional
		A list of conditional probabilities with one tensor for each feature
		in the data being modeled. Each tensor should have `k+1` dimensions 
		where `k` is the number of timesteps to condition on. Each dimension
		should span the number of keys in that dimension. For example, if
		specifying a univariate conditional categorical distribution where
		k=2, a valid tensor shape would be [(2, 3, 4)]. Default is None.

	n_categories: list, numpy.ndarray, torch.tensor or None, optional
		The number of categories for each feature in the data. Only needs to
		be provided when the parameters will be learned directly from data and
		you want to make sure that right number of keys are included in each
		dimension. Unlike the `Categorical` distribution, this needs to be
		a list of shapes with one shape for each feature and the shape matches
		that specified in `probs`. Default is None.

	pseudocount: float, optional
		A value to add to the observed counts of each feature when training.
		Setting this to a positive value ensures that no probabilities are
		truly zero. Default is 0.

	inertia: float, (0, 1), optional
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

	def __init__(self, probs=None, n_categories=None, pseudocount=0, 
		inertia=0.0, frozen=False, check_data=True):
		super().__init__(inertia=inertia, frozen=frozen, check_data=check_data)
		self.name = "ConditionalCategorical"

		if probs is not None:
			self.n_categories = []
			self.probs = torch.nn.ParameterList([])
			
			for prob in probs:
				prob = _check_parameter(_cast_as_parameter(prob), "probs",
					min_value=0, max_value=1)
				
				self.probs.append(prob)
				self.n_categories.append(tuple(prob.shape))

		else:
			self.probs = None
			self.n_categories = n_categories
		
		self.pseudocount = _check_parameter(pseudocount, "pseudocount")

		self._initialized = probs is not None
		self.d = len(self.probs) if self._initialized else None
		self.n_parents = len(self.probs[0].shape) if self._initialized else None
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

		n_categories: list of tuples
			The shape of each conditional distribution, one per feature.
		"""

		self.n_categories = []
		for n_cat in n_categories:
			if isinstance(n_cat, (list, tuple)):
				self.n_categories.append(tuple(n_cat))
			elif isinstance(n_cat, (numpy.ndarray, torch.Tensor)):
				self.n_categories.append(tuple(n_cat.tolist()))

		self.n_parents = len(self.n_categories[0])
		self.probs = torch.nn.ParameterList([_cast_as_parameter(torch.zeros(
			*cats, dtype=self.dtype, device=self.device, requires_grad=False)) 
				for cats in self.n_categories])

		self._initialized = True
		super()._initialize(d)

	def _reset_cache(self):
		"""Reset the internally stored statistics.

		This method is meant to only be called internally. It resets the
		stored statistics used to update the model parameters as well as
		recalculates the cached values meant to speed up log probability
		calculations.
		"""

		if self._initialized == False:
			return

		_w_sum = []
		_xw_sum = []

		for n_categories in self.n_categories:
			_w_sum.append(torch.zeros(*n_categories[:-1], 
				dtype=self.probs[0].dtype, device=self.device))
			_xw_sum.append(torch.zeros(*n_categories, 
				dtype=self.probs[0].dtype, device=self.device))

		self._w_sum = BufferList(_w_sum)
		self._xw_sum = BufferList(_xw_sum)

		self._log_probs = BufferList([torch.log(prob) for prob in self.probs])

	def sample(self, n, X):
		"""Sample from the probability distribution.

		This method will return `n` samples generated from the underlying
		probability distribution. For a mixture model, this involves first
		sampling the component using the prior probabilities, and then sampling
		from the chosen distribution.


		Parameters
		----------
		n: int
			The number of samples to generate.
		
		X: list, numpy.ndarray, torch.tensor, shape=(n, d, *self.probs.shape-1) 
			The values to be conditioned on when generating the samples.

		Returns
		-------
		X: torch.tensor, shape=(n, self.d)
			Randomly generated samples.
		"""

		X = _check_parameter(_cast_as_tensor(X), "X", ndim=3, 
			shape=(-1, self.n_parents-1, self.d))

		y = []
		for i in range(n):
			y.append([])

			for j in range(self.d):
				idx = tuple(X[i, :, j])
				if len(idx) == 1:
					idx = idx[0].item()
				
				probs = self.probs[j][idx]

				y_ = torch.multinomial(probs, 1).item()
				y[-1].append(y_)

		return torch.tensor(y)

	def log_probability(self, X):
		X = _check_parameter(_cast_as_tensor(X), "X", ndim=3, 
			shape=(-1, self.n_parents, self.d), check_parameter=self.check_data)

		logps = torch.zeros(len(X), dtype=self.probs[0].dtype, device=X.device, 
			requires_grad=False)

		for i in range(len(X)):
			for j in range(self.d):
				logps[i] += self._log_probs[j][tuple(X[i, :, j])]

		return logps

	def summarize(self, X, sample_weight=None):
		if self.frozen == True:
			return

		X = _check_parameter(_cast_as_tensor(X), "X", ndim=3, 
			dtypes=(torch.int32, torch.int64), check_parameter=self.check_data)

		if not self._initialized:
			self._initialize(len(X[0][0]), torch.max(X, dim=0)[0].T+1)

		X = _check_parameter(X, "X", shape=(-1, self.n_parents, self.d),
			check_parameter=self.check_data)
		sample_weight = _check_parameter(_cast_as_tensor(sample_weight, 
			dtype=torch.float32), "sample_weight", min_value=0, ndim=(1, 2))

		if sample_weight is None:
			sample_weight = torch.ones(X[:, 0].shape[0], X[:, 0].shape[-1], 
				dtype=self.probs[0].dtype)
		elif len(sample_weight.shape) == 1: 
			sample_weight = sample_weight.reshape(-1, 1).expand(-1, X.shape[2])
		elif sample_weight.shape[1] == 1 and self.d > 1:
			sample_weight = sample_weight.expand(-1, X.shape[2])

		_check_parameter(sample_weight, "sample_weight", 
			min_value=0, ndim=2, shape=(X.shape[0], X.shape[2]))

		for j in range(self.d):
			strides = torch.tensor(self._xw_sum[j].stride(), device=X.device)
			X_ = torch.sum(X[:, :, j] * strides, dim=-1)

			self._xw_sum[j].view(-1).scatter_add_(0, X_, sample_weight[:,j])
			self._w_sum[j][:] = self._xw_sum[j].sum(dim=-1)

	def from_summaries(self):
		if self.frozen == True:
			return

		for i in range(self.d):
			probs = self._xw_sum[i] / self._w_sum[i].unsqueeze(-1)
			probs = torch.nan_to_num(probs, 1. / probs.shape[-1])

			_update_parameter(self.probs[i], probs, self.inertia)

		self._reset_cache()


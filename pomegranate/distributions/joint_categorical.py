# joint_categorical.py
# Contact: Jacob Schreiber <jmschreiber91@gmail.com>

import numpy
import torch

from .._utils import _cast_as_tensor
from .._utils import _cast_as_parameter
from .._utils import _update_parameter
from .._utils import _check_parameter
from .._utils import _reshape_weights

from ._distribution import Distribution
from .categorical import Categorical


class JointCategorical(Distribution):
	"""A joint categorical distribution.

	A joint categorical distribution models the probability of a vector of
	categorical values occurring without assuming that the dimensions are
	independent from each other. Essentially, it is a Categorical distribution
	without the assumption that the dimensions are independent of each other. 

	There are two ways to initialize this object. The first is to pass in
	the tensor of probability parameters, at which point they can immediately be
	used. The second is to not pass in the rate parameters and then call
	either `fit` or `summary` + `from_summaries`, at which point the 
	probability parameters will be learned from data.


	Parameters
	----------
	probs: list, numpy.ndarray, torch.tensor, or None, shape=*n_categories
		A tensor where each dimension corresponds to one column in the data
		set being modeled and the size of each dimension is the number of
		categories in that column, e.g., if the data being modeled is binary 
		and has shape (5, 4), this will be a tensor with shape (2, 2, 2, 2).
		Default is None.

	n_categories: list, numpy.ndarray, torch.tensor, or None, shape=(d,)
		A vector with the maximum number of categories that each column
		can have. If not given, this will be inferred from the data. Default
		is None.

	inertia: float, [0, 1], optional
		Indicates the proportion of the update to apply to the parameters
		during training. When the inertia is 0.0, the update is applied in
		its entirety and the previous parameters are ignored. When the
		inertia is 1.0, the update is entirely ignored and the previous
		parameters are kept, equivalently to if the parameters were frozen.

	pseudocount: float, optional
		A number of observations to add to each entry in the probability
		distribution during training. A higher value will smooth the 
		distributions more. Default is 0.

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
	
	def __init__(self, probs=None, n_categories=None, pseudocount=0, 
		inertia=0.0, frozen=False, check_data=True):
		super().__init__(inertia=inertia, frozen=frozen, check_data=check_data)
		self.name = "JointCategorical"

		self.probs = _check_parameter(_cast_as_parameter(probs), "probs", 
			min_value=0, max_value=1, value_sum=1)

		self.n_categories = _check_parameter(n_categories, "n_categories", 
			min_value=2)
		self.pseudocount = _check_parameter(pseudocount, "pseudocount")

		self._initialized = probs is not None
		self.d = len(self.probs.shape) if self._initialized else None

		if self._initialized:
			if n_categories is None:
				self.n_categories = tuple(self.probs.shape)
			elif isinstance(n_categories, int):
				self.n_categories = (n_categories for i in range(n_categories))
			else:
				self.n_categories = tuple(n_categories)
		else:
			self.n_categories = None

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

		n_categories: list, numpy.ndarray, torch.tensor, or None, shape=(d,)
			A vector with the maximum number of categories that each column
			can have. If not given, this will be inferred from the data. 
			Default is None.
		"""

		self.probs = _cast_as_parameter(torch.zeros(*n_categories, 
			dtype=self.dtype, device=self.device))

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

		if self._initialized == False:
			return

		self._w_sum = torch.zeros(self.d, dtype=self.probs.dtype)
		self._xw_sum = torch.zeros(*self.n_categories, dtype=self.probs.dtype)

		self._log_probs = torch.log(self.probs)

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

		idxs = torch.multinomial(self.probs.flatten(), num_samples=n, 
			replacement=True)

		X = numpy.unravel_index(idxs.numpy(), self.n_categories)
		X = numpy.stack(X).T
		return torch.from_numpy(X)

	def log_probability(self, X):
		"""Calculate the log probability of each example.

		This method calculates the log probability of each example given the
		parameters of the distribution. The examples must be given in a 2D
		format. For a joint categorical distribution, each value must be an
		integer category that is smaller than the maximum number of categories
		for each feature.

		Note: This differs from some other log probability calculation
		functions, like those in torch.distributions, because it is not
		returning the log probability of each feature independently, but rather
		the total log probability of the entire example.


		Parameters
		----------
		X: list, tuple, numpy.ndarray, torch.Tensor, shape=(-1, self.d)
			A set of examples to evaluate.


		Returns
		-------
		logp: torch.Tensor, shape=(-1,)
			The log probability of each example.
		"""

		X = _check_parameter(_cast_as_tensor(X), "X", 
			value_set=tuple(range(max(self.n_categories)+1)), ndim=2, 
			shape=(-1, self.d), check_parameter=self.check_data)

		logps = torch.zeros(len(X), dtype=self.probs.dtype)
		for i in range(len(X)):
			logps[i] = self._log_probs[tuple(X[i])]

		return logps


	def summarize(self, X, sample_weight=None):
		"""Extract the sufficient statistics from a batch of data.

		This method calculates the sufficient statistics from optionally
		weighted data and adds them to the stored cache. The examples must be
		given in a 2D format. Sample weights can either be provided as one
		value per example or as a 2D matrix of weights for each feature in
		each example.


		Parameters
		----------
		X: list, tuple, numpy.ndarray, torch.Tensor, shape=(-1, self.d)
			A set of examples to summarize.

		sample_weight: list, tuple, numpy.ndarray, torch.Tensor, optional
			A set of weights for the examples. This can be either of shape
			(-1, self.d) or a vector of shape (-1,). Default is ones.
		"""

		if self.frozen == True:
			return

		X = _check_parameter(_cast_as_tensor(X), "X", ndim=2, 
			dtypes=(torch.int32, torch.int64), check_parameter=self.check_data)

		if not self._initialized:
			self._initialize(len(X[0]), torch.max(X, dim=0)[0]+1)

		X = _check_parameter(X, "X", shape=(-1, self.d), 
			value_set=tuple(range(max(self.n_categories)+1)), 
			check_parameter=self.check_data)

		sample_weight = _reshape_weights(X, _cast_as_tensor(sample_weight, 
			dtype=torch.float32))[:,0]

		self._w_sum += torch.sum(sample_weight, dim=0)
		for i in range(len(X)):
			self._xw_sum[tuple(X[i])] += sample_weight[i]

	def from_summaries(self):
		"""Update the model parameters given the extracted statistics.

		This method uses calculated statistics from calls to the `summarize`
		method to update the distribution parameters. Hyperparameters for the
		update are passed in at initialization time.

		Note: Internally, a call to `fit` is just a successive call to the
		`summarize` method followed by the `from_summaries` method.
		"""

		if self.frozen == True:
			return

		probs = self._xw_sum / self._w_sum[0]

		_update_parameter(self.probs, probs, self.inertia)
		self._reset_cache()

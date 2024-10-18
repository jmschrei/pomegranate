# categorical.py
# Contact: Jacob Schreiber <jmschreiber91@gmail.com>

import torch

from .._utils import _inplace_add
from .._utils import _cast_as_tensor
from .._utils import _cast_as_parameter
from .._utils import _update_parameter
from .._utils import _check_parameter
from .._utils import _reshape_weights

from ._distribution import Distribution


class Categorical(Distribution):
	"""A categorical distribution object.

	A categorical distribution models the probability of a set of distinct
	values happening. It is an extension of the Bernoulli distribution to
	multiple values. Sometimes it is referred to as a discrete distribution,
	but this distribution does not enforce that the numeric values used for the
	keys have any relationship based on their identity. Permuting the keys will
	have no effect on the calculation. This distribution assumes that the
	features are independent from each other.

	The keys must be contiguous non-negative integers that begin at zero. 
	Because the probabilities are represented as a single tensor, each feature
	must have values for all keys up to the maximum key of any one distribution.
	Specifically, if one feature has 10 keys and a second feature has only 4,
	the tensor must go out to 10 for each feature but encode probabilities of
	zero for the second feature. 


	Parameters
	----------
	probs: list, numpy.ndarray, torch.tensor or None, shape=(k, d), optional
		Probabilities for each key for each feature, where k is the largest
		number of keys across all features. Default is None

	n_categories: list, numpy.ndarray, torch.tensor or None, optional
		The number of categories for each feature in the data. Only needs to
		be provided when the parameters will be learned directly from data and
		you want to make sure that right number of keys are included in each
		dimension. Default is None.

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

	def __init__(self, probs=None, n_categories=None, pseudocount=0.0, 
		inertia=0.0, frozen=False, check_data=True):
		super().__init__(inertia=inertia, frozen=frozen, check_data=check_data)
		self.name = "Categorical"

		self.probs = _check_parameter(_cast_as_parameter(probs), "probs", 
			min_value=0, max_value=1, ndim=2)

		self.pseudocount = pseudocount

		self._initialized = probs is not None
		self.d = self.probs.shape[-2] if self._initialized else None

		if n_categories is not None:
			self.n_keys = n_categories
		else:
			self.n_keys = self.probs.shape[-1] if self._initialized else None
		
		self._reset_cache()

	def _initialize(self, d, n_keys):
		"""Initialize the probability distribution.

		This method is meant to only be called internally. It initializes the
		parameters of the distribution and stores its dimensionality. For more
		complex methods, this function will do more.


		Parameters
		----------
		d: int
			The dimensionality the distribution is being initialized to.

		n_keys: int
			The number of keys the distribution is being initialized with.
		"""

		self.probs = _cast_as_parameter(torch.zeros(d, n_keys, 
			dtype=self.dtype, device=self.device))

		self.n_keys = n_keys
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

		self.register_buffer("_w_sum", torch.zeros(self.d, device=self.device))
		self.register_buffer("_xw_sum", torch.zeros(self.d, self.n_keys, 
			device=self.device))

		self.register_buffer("_log_probs", torch.log(self.probs))

	def sample(self, n):
		"""Sample from the probability distribution.

		This method will return `n` samples generated from the underlying
		probability distribution.


		Parameters
		----------
		n: int
			The number of samples to generate.
		

		Returns
		-------
		X: torch.tensor, shape=(n, self.d)
			Randomly generated samples.
		"""

		return torch.distributions.Categorical(self.probs).sample([n])

	def log_probability(self, X):
		"""Calculate the log probability of each example.

		This method calculates the log probability of each example given the
		parameters of the distribution. The examples must be given in a 2D
		format. For a categorical distribution, each entry in the data must
		be an integer in the range [0, n_keys).

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

		X = _check_parameter(_cast_as_tensor(X), "X", min_value=0.0,
			max_value=self.n_keys-1, ndim=2, shape=(-1, self.d),
			check_parameter=self.check_data)

		logps = torch.zeros(X.shape[0], dtype=self.probs.dtype, 
			device=self.device)
		
		for i in range(self.d):
			if isinstance(X, torch.masked.MaskedTensor):
				logp_ = self._log_probs[i][X[:, i]._masked_data]
				logp_[~X[:, i]._masked_mask] = 0
				logps += logp_
			else:
				logps += self._log_probs[i][X[:, i]]

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

		X = _cast_as_tensor(X)
		if not self._initialized:
			if self.n_keys is not None:
				n_keys = self.n_keys
			elif isinstance(X, torch.masked.MaskedTensor):
				n_keys = int(torch.max(X._masked_data)) + 1
			else:
				n_keys = int(torch.max(X)) + 1

			self._initialize(X.shape[1], n_keys)

		X = _check_parameter(X, "X", min_value=0, max_value=self.n_keys-1, 
			ndim=2, shape=(-1, self.d), check_parameter=self.check_data)
		sample_weight = _reshape_weights(X, _cast_as_tensor(sample_weight))

		_inplace_add(self._w_sum, torch.sum(sample_weight, dim=0))
		for i in range(self.n_keys):
			_inplace_add(self._xw_sum[:, i], torch.sum((X == i) * sample_weight, 
				dim=0))

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

		probs = (self._xw_sum + self.pseudocount) / (self._w_sum + 
			self.pseudocount * self.n_keys).unsqueeze(1)

		_update_parameter(self.probs, probs, self.inertia)
		self._reset_cache()

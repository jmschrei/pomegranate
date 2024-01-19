# exponential.py
# Contact: Jacob Schreiber <jmschreiber91@gmail.com>

import torch
from torch.distributions import Exponential as tExponential

from .._utils import _cast_as_tensor
from .._utils import _cast_as_parameter
from .._utils import _update_parameter
from .._utils import _check_parameter

from ._distribution import Distribution


class Exponential(Distribution):
	"""An exponential distribution object.

	An exponential distribution models scales of discrete events, and has a
	rate parameter describing the average time between event occurrences.
	This distribution assumes that each feature is independent of the others.
	Although the object is meant to operate on discrete counts, it can be used
	on any non-negative continuous data.

	There are two ways to initialize this object. The first is to pass in
	the tensor of rate parameters, at which point they can immediately be
	used. The second is to not pass in the rate parameters and then call
	either `fit` or `summary` + `from_summaries`, at which point the rate
	parameter will be learned from data.


	Parameters
	----------
	scales: list, numpy.ndarray, torch.Tensor or None, shape=(d,), optional
		The rate parameters for each feature. Default is None.

	inertia: float, (0, 1), optional
		Indicates the proportion of the update to apply to the parameters
		during training. When the inertia is 0.0, the update is applied in
		its entirety and the previous parameters are ignored. When the
		inertia is 1.0, the update is entirely ignored and the previous
		parameters are kept, equivalently to if the parameters were frozen.

	frozen: bool, optional
		Whether all the parameters associated with this distribution are 
		frozen. If you want to freeze individual pameters, or individual values 
		in those parameters, you must modify the `frozen` attribute of the 
		tensor or parameter directly. Default is False.

	check_data: bool, optional
		Whether to check properties of the data and potentially recast it to
		torch.tensors. This does not prevent checking of parameters but can
		slightly speed up computation when you know that your inputs are valid.
		Setting this to False is also necessary for compiling.
	"""

	def __init__(self, scales=None, inertia=0.0, frozen=False, check_data=True):
		super().__init__(inertia=inertia, frozen=frozen, check_data=check_data)
		self.name = "Exponential"

		self.scales = _check_parameter(_cast_as_parameter(scales), "scales", 
			min_value=0, ndim=1)

		self._initialized = scales is not None
		self.d = self.scales.shape[-1] if self._initialized else None
		self._reset_cache()

	def _initialize(self, d):
		"""Initialize the probability distribution.

		This method is meant to only be called internally. It initializes the
		parameters of the distribution and stores its dimensionality. For more
		complex methods, this function will do more.


		Parameters
		----------
		d: int
			The dimensionality the distribution is being initialized to.
		"""

		self.scales = _cast_as_parameter(torch.zeros(d, dtype=self.dtype,
			device=self.device))

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
		self.register_buffer("_xw_sum", torch.zeros(self.d, device=self.device))

		self.register_buffer("_log_scales", torch.log(self.scales))

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

		return tExponential(1. / self.scales).sample([n])

	def log_probability(self, X):
		"""Calculate the log probability of each example.

		This method calculates the log probability of each example given the
		parameters of the distribution. The examples must be given in a 2D
		format. For an exponential distribution, the data must be non-negative.

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
			ndim=2, shape=(-1, self.d), check_parameter=self.check_data)
		
		return torch.sum(-self._log_scales - (1. / self.scales) * X, dim=1)

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
			
		X, sample_weight = super().summarize(X, sample_weight=sample_weight)
		_check_parameter(X, "X", min_value=0, check_parameter=self.check_data)

		self._w_sum[:] = self._w_sum + torch.sum(sample_weight, dim=0)
		self._xw_sum[:] = self._xw_sum + torch.sum(X * sample_weight, dim=0)

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

		scales = self._xw_sum / self._w_sum
		_update_parameter(self.scales, scales, self.inertia)
		self._reset_cache()

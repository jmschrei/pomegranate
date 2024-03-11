# uniform.py
# Contact: Jacob Schreiber <jmschreiber91@gmail.com>

import torch

from .._utils import _cast_as_tensor
from .._utils import _cast_as_parameter
from .._utils import _update_parameter
from .._utils import _check_parameter
from .._utils import _check_shapes

from ._distribution import Distribution

inf = float("inf")


class Uniform(Distribution):
	"""A uniform distribution.

	A uniform distribution models the probability of a variable occurring given
	a range that has the same probability within it and no probability outside
	it. It is described by a vector of minimum and maximum values for this
	range.  This distribution assumes that the features are independent of
	each other.

	There are two ways to initialize this object. The first is to pass in
	the tensor of probability parameters, at which point they can immediately be
	used. The second is to not pass in the rate parameters and then call
	either `fit` or `summary` + `from_summaries`, at which point the probability
	parameter will be learned from data.


	Parameters
	----------
	mins: list, numpy.ndarray, torch.Tensor or None, shape=(d,), optional
		The minimum values of the range.

	maxs: list, numpy.ndarray, torch.Tensor, or None, optional
		The maximum values of the range.

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

	def __init__(self, mins=None, maxs=None, inertia=0.0, frozen=False, 
		check_data=True):
		super().__init__(inertia=inertia, frozen=frozen, check_data=check_data)
		self.name = "Uniform"

		self.mins = _check_parameter(_cast_as_parameter(mins), "mins", ndim=1)
		self.maxs = _check_parameter(_cast_as_parameter(maxs), "maxs", ndim=1)

		_check_shapes([self.mins, self.maxs], ["mins", "maxs"])

		self._initialized = (mins is not None) and (maxs is not None)
		self.d = self.mins.shape[-1] if self._initialized else None
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

		self.mins = _cast_as_parameter(torch.zeros(d, dtype=self.dtype,
			device=self.device))
		self.maxs = _cast_as_parameter(torch.zeros(d, dtype=self.dtype,
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

		self.register_buffer("_x_mins", torch.full((self.d,), inf, 
			device=self.device))
		self.register_buffer("_x_maxs", torch.full((self.d,), -inf,
			device=self.device))
		self.register_buffer("_logps", -torch.log(self.maxs - self.mins))

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

		return torch.distributions.Uniform(self.mins, self.maxs).sample([n])

	def log_probability(self, X):
		"""Calculate the log probability of each example.

		This method calculates the log probability of each example given the
		parameters of the distribution. The examples must be given in a 2D
		format. For a Bernoulli distribution, each entry in the data must
		be either 0 or 1.

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

		X = _check_parameter(_cast_as_tensor(X), "X", ndim=2, 
			shape=(-1, self.d), check_parameter=self.check_data)

		return torch.where((X >= self.mins) & (X <= self.maxs), self._logps, 
			float("-inf")).sum(dim=1)

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

		self._x_mins = torch.minimum(self._x_mins, X.min(dim=0).values)
		self._x_maxs = torch.maximum(self._x_maxs, X.max(dim=0).values)

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

		_update_parameter(self.mins, self._x_mins, self.inertia)
		_update_parameter(self.maxs, self._x_maxs, self.inertia)
		self._reset_cache()

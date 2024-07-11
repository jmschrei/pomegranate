# normal.py
# Contact: Jacob Schreiber <jmschreiber91@gmail.com>

import torch

from .._utils import _cast_as_tensor
from .._utils import _cast_as_parameter
from .._utils import _update_parameter
from .._utils import _check_parameter
from .._utils import _check_shapes

from ._distribution import Distribution


# Define some useful constants
NEGINF = float("-inf")
INF = float("inf")
SQRT_2_PI = 2.50662827463
LOG_2_PI = 1.83787706641


class Normal(Distribution):
	"""A normal distribution object.

	A normal distribution models the probability of a variable occurring under
	a bell-shaped curve. It is described by a vector of mean values and a
	covariance value that can be zero, one, or two dimensional. This
	distribution can assume that features are independent of the others if
	the covariance type is 'diag' or 'sphere', but if the type is 'full' then
	the features are not independent.

	There are two ways to initialize this object. The first is to pass in
	the tensor of probability parameters, at which point they can immediately be
	used. The second is to not pass in the rate parameters and then call
	either `fit` or `summary` + `from_summaries`, at which point the probability
	parameter will be learned from data.


	Parameters
	----------
	means: list, numpy.ndarray, torch.Tensor or None, shape=(d,), optional
		The mean values of the distributions. Default is None.

	covs: list, numpy.ndarray, torch.Tensor, or None, optional
		The variances and covariances of the distribution. If covariance_type
		is 'full', the shape should be (self.d, self.d); if 'diag', the shape
		should be (self.d,); if 'sphere', it should be (1,). Note that this is
		the variances or covariances in all settings, and not the standard
		deviation, as may be more common for diagonal covariance matrices.
		Default is None.

	covariance_type: str, optional
		The type of covariance matrix. Must be one of 'full', 'diag', or
		'sphere'. Default is 'full'. 

	min_cov: float or None, optional
		The minimum variance or covariance.

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
	"""

	def __init__(self, means=None, covs=None, covariance_type='full', 
		min_cov=None, inertia=0.0, frozen=False, check_data=True):
		super().__init__(inertia=inertia, frozen=frozen, check_data=check_data)
		self.name = "Normal"

		self.means = _check_parameter(_cast_as_parameter(means), "means", 
			ndim=1)
		self.covs = _check_parameter(_cast_as_parameter(covs), "covs", 
			ndim=(1, 2))

		_check_shapes([self.means, self.covs], ["means", "covs"])

		self.min_cov = _check_parameter(min_cov, "min_cov", min_value=0, ndim=0)
		self.covariance_type = covariance_type

		self._initialized = (means is not None) and (covs is not None)
		self.d = self.means.shape[-1] if self._initialized else None
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

		self.means = _cast_as_parameter(torch.zeros(d, dtype=self.dtype,
			device=self.device))
		
		if self.covariance_type == 'full':
			self.covs = _cast_as_parameter(torch.zeros(d, d, 
				dtype=self.dtype, device=self.device))
		elif self.covariance_type == 'diag':
			self.covs = _cast_as_parameter(torch.zeros(d, dtype=self.dtype,
				device=self.device))
		elif self.covariance_type == 'sphere':
			self.covs = _cast_as_parameter(torch.tensor(0, dtype=self.dtype,
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

		self.register_buffer("_w_sum", torch.zeros(self.d, dtype=self.dtype, 
			device=self.device))
		self.register_buffer("_xw_sum", torch.zeros(self.d, dtype=self.dtype,
			device=self.device))

		if self.covariance_type == 'full':
			self.register_buffer("_xxw_sum", torch.zeros(self.d, self.d, 
				dtype=self.dtype, device=self.device))

			if self.covs.sum() > 0.0:
				chol = torch.linalg.cholesky(self.covs)
				_inv_cov = torch.linalg.solve_triangular(chol, torch.eye(
					len(self.covs), dtype=self.dtype, device=self.device), 
					upper=False).T
				_inv_cov_dot_mu = torch.matmul(self.means, _inv_cov)
				_log_det = -0.5 * torch.linalg.slogdet(self.covs)[1]
				_theta = _log_det - 0.5 * (self.d * LOG_2_PI)

				self.register_buffer("_inv_cov", _inv_cov)
				self.register_buffer("_inv_cov_dot_mu", _inv_cov_dot_mu)
				self.register_buffer("_log_det", _log_det)
				self.register_buffer("_theta", _theta)

		elif self.covariance_type in ('diag', 'sphere'):
			self.register_buffer("_xxw_sum", torch.zeros(self.d, 
				dtype=self.dtype, device=self.device))

			if self.covs.sum() > 0.0:
				_log_sigma_sqrt_2pi = -torch.log(torch.sqrt(self.covs) * 
					SQRT_2_PI)
				_inv_two_sigma = 1. / (2 * self.covs)

				self.register_buffer("_log_sigma_sqrt_2pi", _log_sigma_sqrt_2pi)
				self.register_buffer("_inv_two_sigma", _inv_two_sigma)

			if torch.any(self.covs < 0):
				raise ValueError("Variances must be positive.")

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

		if self.covariance_type == 'diag':
			return torch.distributions.Normal(self.means, self.covs).sample([n])
		elif self.covariance_type == 'full':
			return torch.distributions.MultivariateNormal(self.means, 
				self.covs).sample([n])

	def log_probability(self, X):
		"""Calculate the log probability of each example.

		This method calculates the log probability of each example given the
		parameters of the distribution. The examples must be given in a 2D
		format.

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

		X = _check_parameter(_cast_as_tensor(X, dtype=self.means.dtype), "X", 
			ndim=2, shape=(-1, self.d), check_parameter=self.check_data)

		if self.covariance_type == 'full':
			logp = torch.matmul(X, self._inv_cov) - self._inv_cov_dot_mu
			logp = self.d * LOG_2_PI + torch.sum(logp ** 2, dim=-1)
			logp = self._log_det - 0.5 * logp
			return logp
		
		elif self.covariance_type in ('diag', 'sphere'):
			return torch.sum(self._log_sigma_sqrt_2pi - ((X - self.means) ** 2) 
				* self._inv_two_sigma, dim=-1)

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
		X = _cast_as_tensor(X, dtype=self.means.dtype)
		sample_weight = _cast_as_tensor(sample_weight, dtype=self.means.dtype)
		if self.covariance_type == 'full':
			self._w_sum += torch.sum(sample_weight, dim=0)
			self._xw_sum += torch.sum(X * sample_weight, axis=0)
			self._xxw_sum += torch.matmul((X * sample_weight).T, X)

		elif self.covariance_type in ('diag', 'sphere'):
			self._w_sum[:] = self._w_sum + torch.sum(sample_weight, dim=0)
			self._xw_sum[:] = self._xw_sum + torch.sum(X * sample_weight, dim=0)
			self._xxw_sum[:] = self._xxw_sum + torch.sum(X ** 2 * 
				sample_weight, dim=0)

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

		means = self._xw_sum / self._w_sum

		if self.covariance_type == 'full':
			v = self._xw_sum.unsqueeze(0) * self._xw_sum.unsqueeze(1)
			covs = self._xxw_sum / self._w_sum -  v / self._w_sum ** 2.0

		elif self.covariance_type in ['diag', 'sphere']:
			covs = self._xxw_sum / self._w_sum - \
				self._xw_sum ** 2.0 / self._w_sum ** 2.0
			if self.covariance_type == 'sphere':
				covs = covs.mean(dim=-1)

		_update_parameter(self.means, means, self.inertia)
		_update_parameter(self.covs, covs, self.inertia)
		self._reset_cache()

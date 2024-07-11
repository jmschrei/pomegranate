# _distribution.py
# Jacob Schreiber <jmschreiber91@gmail.com>

import torch

from .._utils import _cast_as_tensor
from .._utils import _update_parameter
from .._utils import _cast_as_parameter
from .._utils import _check_parameter
from .._utils import _reshape_weights


class Distribution(torch.nn.Module):
	"""A base distribution object.

	This distribution is inherited by all the other distributions.
	"""

	def __init__(self, inertia, frozen, check_data):
		super(Distribution, self).__init__()
		self._device = _cast_as_parameter([0.0])

		_check_parameter(inertia, "inertia", min_value=0, max_value=1, ndim=0)
		_check_parameter(frozen, "frozen", value_set=[True, False], ndim=0)
		_check_parameter(check_data, "check_data", value_set=[True, False],
			ndim=0)

		self.register_buffer("inertia", _cast_as_tensor(inertia))
		self.register_buffer("frozen", _cast_as_tensor(frozen))
		self.register_buffer("check_data", _cast_as_tensor(check_data))

		self._initialized = False

	@property
	def device(self):
		try:
			return next(self.parameters()).device
		except:
			return 'cpu'

	@property
	def dtype(self):
		return next(self.parameters()).dtype

	def freeze(self):
		self.register_buffer("frozen", _cast_as_tensor(True))
		return self

	def unfreeze(self):
		self.register_buffer("frozen", _cast_as_tensor(False))
		return self

	def forward(self, X):
		self.summarize(X)
		return self.log_probability(X)

	def backward(self, X):
		self.from_summaries()
		return X

	def _initialize(self, d):
		self.d = d
		self._reset_cache()

	def _reset_cache(self):
		raise NotImplementedError

	def probability(self, X):
		return torch.exp(self.log_probability(X))

	def log_probability(self, X):
		raise NotImplementedError

	def fit(self, X, sample_weight=None):
		self.summarize(X, sample_weight=sample_weight)
		self.from_summaries()
		return self

	def summarize(self, X, sample_weight=None):
		if not self._initialized:
			self._initialize(len(X[0]))

		X = _cast_as_tensor(X)
		_check_parameter(X, "X", ndim=2, shape=(-1, self.d), 
			check_parameter=self.check_data)

		sample_weight = _reshape_weights(X, _cast_as_tensor(sample_weight), 
			device=self.device)

		return X, sample_weight

	def from_summaries(self):
		raise NotImplementedError


class ConditionalDistribution(Distribution):
	def __init__(self, inertia, frozen, check_data):
		super().__init__(inertia=inertia, frozen=frozen, check_data=check_data)

	def marginal(self, dim):
		raise NotImplementedError
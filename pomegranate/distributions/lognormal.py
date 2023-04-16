# lognormal.py
# Contact: Jacob Schreiber <jmschreiber91@gmail.com>

import torch

from .._utils import _cast_as_tensor
from .._utils import _update_parameter
from .._utils import _check_parameter
from .._utils import _check_shapes

from ._distribution import Distribution

from .normal import Normal

class LogNormal(Normal):
	"""Still under development."""
	
	def __init__(self, means=None, covs=None, covariance_type='diag', 
		min_cov=0.0, frozen=False, inertia=0.0):
		super(LogNormal, self).__init__(means=means, covs=covs, 
			covariance_type=covariance_type, min_cov=min_cov,
			frozen=frozen, inertia=inertia)

	def log_probability(self, X):
		X = _check_parameter(_cast_as_tensor(X, dtype=self.means.dtype), "X", 
			ndim=2, shape=(-1, self.d), min_value=0.0)

		return super(LogNormal, self).log_probability(torch.log(X))

	def summarize(self, X, sample_weight=None):
		X = _cast_as_tensor(X)

		super(LogNormal, self).summarize(torch.log(X), 
			sample_weight=sample_weight)

	def from_summaries(self):
		super(LogNormal, self).from_summaries()

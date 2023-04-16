# negative_binomial.py
# Contact: Jacob Schreiber <jmschreiber91@gmail.com>

import torch

from .._utils import _cast_as_tensor
from .._utils import _update_parameter
from .._utils import _check_parameter

from ._distribution import Distribution


class NegativeBinomial(Distribution):
	"""An exponential distribution object.

	An exponential distribution models scales of discrete events, and has a
	rate parameter describing the average time between event occurances.
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
	scales: torch.tensor or None, shape=(d,), optional
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


	Examples
	--------
	>>> # Create a distribution with known parameters
	>>> scales = torch.tensor([1.2, 0.4])
	>>> X = torch.tensor([[0.3, 0.2], [0.8, 0.1]])
	>>>
	>>> d = Exponential(scales)
	>>> d.log_probability(X)
	tensor([-1.1740, -1.7340])
	>>>
	>>>
	>>> # Fit a distribution to data
	>>> torch.manual_seed(0)
	>>>
	>>> X = torch.exp(torch.randn(100, 10) * 15)
	>>> X.shape
	torch.Size([100, 10])
	>>> 
	>>> d = Exponential()
	>>> d.fit(X)
	>>> d.scales
	tensor([2.5857e-13, 1.7420e-14, 6.8009e-21, 1.9106e-25, 5.1296e-19, 
		2.4965e-15, 4.7202e-10, 2.5022e-24, 7.7177e-21, 6.0313e-21])
	>>>
	>>>
	>>> # Fit a distribution using the summarize API
	>>> d = Exponential()
	>>> d.summarize(X[:50])
	>>> d.summarize(X[50:])
	>>> d.from_summaries()
	>>> d.scales
	tensor([2.5857e-13, 1.7420e-14, 6.8009e-21, 1.9106e-25, 5.1296e-19, 
		2.4965e-15, 4.7202e-10, 2.5022e-24, 7.7177e-21, 6.0313e-21])
	>>> 
	>>>
	>>>
	>>> # As a loss function for a neural network
	>>> class ToyNet(torch.nn.Module):
	>>> 	def __init__(self, d):
	>>>			super(ToyNet, self).__init__()
	>>>			self.fc1 = torch.nn.Linear(d, 32)
	>>>			self.scales = torch.nn.Linear(32, d)
	>>>			self.relu = torch.nn.ReLU()
	>>>
	>>>		def forward(self, X):
	>>>			X = self.fc1(X)
	>>>			X = self.relu(X)
	>>>			scales = self.scales(X)
	>>>			return self.relu(scales) + 0.01
	>>>
	>>> model = ToyNet(10)
	>>> optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
	>>>
	>>> for i in range(100):
	>>>		optimizer.zero_grad()
	>>>
	>>>		scales = model(X)
	>>> 	loss = -Exponential(scales).log_probability(X).sum()
	>>>		loss.backward()
	>>>		optimizer.step()
	"""

	def __init__(self, n_successes=None, probs=None, inertia=0.0, frozen=False):
		super().__init__(inertia=inertia, frozen=frozen)
		self.name = "Exponential"

		self.n_successes = _check_parameter(_cast_as_tensor(n_successes),
			"n_successes", min_value=0, ndim=1)
		self.probs = _check_parameter(_cast_as_tensor(probs), "probs",
			min_value=0, max_value=1, ndim=1)

		self._initialized = (n_successes is not None) and (probs is not None)
		self.d = len(self.scales) if self._initialized else None
		self._reset_cache()

	def _initialize(self, d):
		self.n_successes = torch.zeros(self.d)
		self.probs = torch.zeros(self.d)

		self._initialized = True
		super()._initialize(d)

	def _reset_cache(self):
		if self._initialized == False:
			return

		self._w_sum = torch.zeros(self.d)
		self._xw_sum = torch.zeros(self.d)

		self._log_scales = torch.log(self.scales)
		self._lgamma_n = torch.lgamma(self.n_successes)

	def log_probability(self, X):
		X = _check_parameter(_cast_as_tensor(X), "X", min_value=0.0, 
			ndim=2, shape=(-1, self.d))

		return torch.lgamma(X + self.n_successess) - self._lgamma_n - \
			torch.lgamma(X+1)

	def summarize(self, X, sample_weight=None):
		if self.frozen == True:
			return

		X, sample_weight = super().summarize(X, sample_weight=sample_weight)
		X = _check_parameter(X, "X", min_value=0)

		self._w_sum += torch.sum(sample_weight, dim=0)
		self._xw_sum += torch.sum(X * sample_weight, dim=0)

	def from_summaries(self):
		if self.frozen == True:
			return

		probs = (self._w_sum * r) / (self._w_sum * r + self._xw_sum)

		scales = self._xw_sum / self._w_sum
		_update_parameter(self.scales, scales, self.inertia)
		self._reset_cache()

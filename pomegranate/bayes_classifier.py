# BayesClassifier.py
# Author: Jacob Schreiber <jmschreiber91@gmail.com>

import numpy
import torch

from ._utils import _cast_as_tensor
from ._utils import _cast_as_parameter
from ._utils import _update_parameter
from ._utils import _check_parameter
from ._utils import _reshape_weights

from ._bayes import BayesMixin

from .distributions._distribution import Distribution


class BayesClassifier(BayesMixin, Distribution):
	"""A Bayes classifier object.

	A simple way to produce a classifier using probabilistic models is to plug
	them into Bayes' rule. Basically, inference is the same as the 'E' step in
	EM for mixture models. However, fitting can be significantly faster because
	instead of having to iteratively infer labels and learn parameters, you can
	just learn the parameters given the known labels. Because the learning step
	for most models are simple MLE estimates, this can be done extremely
	quickly.

	Although the most common distribution to use is a Gaussian with a diagonal
	covariance matrix, termed the Gaussian naive Bayes model, any probability
	distribution can be used. Here, you can just drop any distributions or
	probabilistic model in as long as it has the `log_probability`, `summarize`,
	and `from_samples` methods implemented.

	Further, the probabilistic models do not even need to be simple
	distributions. The distributions can be mixture models or hidden Markov
	models or Bayesian networks.


	Parameters
	----------
	distributions: tuple or list
		A set of distribution objects. These objects do not need to be
		initialized, i.e., can be "Normal()". 

	priors: tuple, numpy.ndarray, torch.Tensor, or None. shape=(k,), optional
		The prior probabilities over the given distributions. Default is None.

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
		Setting this to False is also necessary for compiling. Default is True.
	"""

	def __init__(self, distributions, priors=None, inertia=0.0, frozen=False,
		check_data=True):
		super().__init__(inertia=inertia, frozen=frozen, check_data=check_data)
		self.name = "BayesClassifier"

		_check_parameter(distributions, "distributions", dtypes=(list, tuple, 
			numpy.array, torch.nn.ModuleList))
		self.distributions = torch.nn.ModuleList(distributions)

		self.priors = _check_parameter(_cast_as_parameter(priors), "priors", 
			min_value=0, max_value=1, ndim=1, value_sum=1.0, 
			shape=(len(distributions),))

		self.k = len(distributions)

		if all(d._initialized for d in distributions):
			self._initialized = True
			self.d = distributions[0].d
			if self.priors is None:
				self.priors = _cast_as_parameter(torch.ones(self.k) / self.k)

		else:
			self._initialized = False
			self.d = None
		
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

		self.priors = _cast_as_parameter(torch.ones(self.k, dtype=self.dtype, 
			device=self.device) / self.k)

		self._initialized = True
		super()._initialize(d)

	def fit(self, X, y, sample_weight=None):
		"""Fit the model to optionally weighted examples.

		This method implements the core of the learning process. For a
		general Bayes model, this involves fitting each component of the model
		using the labels that are provided. 

		This method is largely a wrapper around the `summarize` and
		`from_summaries` methods. It's primary contribution is serving as a
		loop around these functions and to monitor convergence.


		Parameters
		----------
		X: list, tuple, numpy.ndarray, torch.Tensor, shape=(-1, self.d)
			A set of examples to evaluate. 

		y: list, tuple, numpy.ndarray, torch.Tensor, shape=(-1,)
			A set of labels, one per example.

		sample_weight: list, tuple, numpy.ndarray, torch.Tensor, optional
			A set of weights for the examples. This can be either of shape
			(-1, self.d) or a vector of shape (-1,). Default is ones.


		Returns
		-------
		self
		"""

		self.summarize(X, y, sample_weight=sample_weight)
		self.from_summaries()
		return self

	def summarize(self, X, y, sample_weight=None):
		"""Extract the sufficient statistics from a batch of data.

		This method calculates the sufficient statistics from optionally
		weighted data and adds them to the stored cache. The examples must be
		given in a 2D format. Sample weights can either be provided as one
		value per example or as a 2D matrix of weights for each feature in
		each example.

		For a Bayes' classifier, this step involves partitioning the data
		according to the labels and then training each component using MLE
		estimates.


		Parameters
		----------
		X: list, tuple, numpy.ndarray, torch.Tensor, shape=(-1, self.d)
			A set of examples to summarize.

		y: list, tuple, numpy.ndarray, torch.Tensor, shape=(-1,)
			A set of labels, one per example.

		sample_weight: list, tuple, numpy.ndarray, torch.Tensor, optional
			A set of weights for the examples. This can be either of shape
			(-1, self.d) or a vector of shape (-1,). Default is ones.
		"""

		X, sample_weight = super().summarize(X, sample_weight=sample_weight)
		y = _check_parameter(_cast_as_tensor(y), "y", min_value=0, 
			max_value=self.k-1, ndim=1, shape=(len(X),), 
			check_parameter=self.check_data)
		sample_weight = _check_parameter(sample_weight, "sample_weight", 
			min_value=0, shape=(-1, self.d), check_parameter=self.check_data)

		for j, d in enumerate(self.distributions):
			idx = y == j
			d.summarize(X[idx], sample_weight[idx])

			if self.frozen == False:
				self._w_sum[j] = self._w_sum[j] + sample_weight[idx].mean(
					dim=-1).sum()

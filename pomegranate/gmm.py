# gmm.py
# Author: Jacob Schreiber <jmschreiber91@gmail.com>

import time
import numpy
import torch

from ._utils import _cast_as_tensor
from ._utils import _cast_as_parameter
from ._utils import _update_parameter
from ._utils import _check_parameter
from ._utils import _reshape_weights

from .distributions._distribution import Distribution

from ._bayes import BayesMixin

from .kmeans import KMeans


class GeneralMixtureModel(BayesMixin, Distribution):
	"""A general mixture model.

	Frequently, data is generated from multiple components. A mixture model
	is a probabilistic model that explicitly models data as having come from
	a set of probability distributions rather than a single one. Usually, the
	abbreviation "GMM" refers to a Gaussian mixture model, but any probability
	distribution or heterogeneous set of distributions can be included in the
	mixture, making it a "general" mixture model.

	However, a mixture model itself has all the same theoretical properties as a
	probability distribution because it is one. Hence, it can be used in any
	situation that a simpler distribution could, such as an emission
	distribution for a HMM or a component of a Bayes classifier.

	Conversely, many models that are usually thought of as composed of
	probability distributions but distinct from them, e.g. hidden Markov models,
	Markov chains, and Bayesian networks, can in theory be passed into this
	object and incorporated into the mixture.

	If the distributions included in the mixture are not initialized, the
	fitting step will first initialize them by running k-means for a small
	number of iterations and fitting the distributions to the clusters that
	are discovered.


	Parameters
	----------
	distributions: tuple or list
		A set of distribution objects. These objects do not need to be
		initialized, i.e., can be "Normal()". 

	priors: tuple, numpy.ndarray, torch.Tensor, or None. shape=(k,), optional
		The prior probabilities over the given distributions. Default is None.

	max_iter: int, optional
		The number of iterations to do in the EM step of fitting the
		distribution. Default is 10.

	tol: float, optional
		The threshold at which to stop during fitting when the improvement
		goes under. Default is 0.1.

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

	verbose: bool, optional
		Whether to print the improvement and timings during training.
	"""

	def __init__(self, distributions, priors=None, init='random', max_iter=1000, 
		tol=0.1, inertia=0.0, frozen=False, random_state=None, check_data=True, 
		verbose=False):
		super().__init__(inertia=inertia, frozen=frozen, check_data=check_data)
		self.name = "GeneralMixtureModel"

		_check_parameter(distributions, "distributions", dtypes=(list, tuple, 
			numpy.array, torch.nn.ModuleList))
		self.distributions = torch.nn.ModuleList(distributions)

		self.priors = _check_parameter(_cast_as_parameter(priors), "priors", 
			min_value=0, max_value=1, ndim=1, value_sum=1.0, 
			shape=(len(distributions),))

		self.verbose = verbose

		self.k = len(distributions)

		if all(d._initialized for d in distributions):
			self._initialized = True
			self.d = distributions[0].d
			if self.priors is None:
				self.priors = _cast_as_parameter(torch.ones(self.k) / self.k)

		else:
			self._initialized = False
			self.d = None
		
		self.init = init
		self.max_iter = max_iter
		self.tol = tol
		self.random_state = random_state
		self._reset_cache()

	def _initialize(self, X, sample_weight=None):
		"""Initialize the probability distribution.

		This method is meant to only be called internally. It initializes the
		parameters of the distribution and stores its dimensionality. For more
		complex methods, this function will do more.


		Parameters
		----------
		X: list, numpy.ndarray, torch.Tensor, shape=(-1, self.d)
			The data to use to initialize the model.

		sample_weight: list, tuple, numpy.ndarray, torch.Tensor, optional
			A set of weights for the examples. This can be either of shape
			(-1, self.d) or a vector of shape (-1,). Default is ones.
		"""

		X = _check_parameter(_cast_as_tensor(X), "X", ndim=2)

		if sample_weight is None:
			sample_weight = torch.ones(1, dtype=self.dtype, 
				device=self.device).expand(X.shape[0], 1)
		else:
			sample_weight = _check_parameter(_cast_as_tensor(sample_weight), 
				"sample_weight", min_value=0., check_parameter=self.check_data)

		model = KMeans(self.k, init=self.init, max_iter=3, 
			random_state=self.random_state)

		if self.device != model.device:
			model.to(self.device)

		y_hat = model.fit_predict(X, sample_weight=sample_weight)

		self.priors = _cast_as_parameter(torch.empty(self.k, 
			dtype=self.dtype, device=self.device))

		sample_weight_sum = sample_weight.sum()
		for i in range(self.k):
			idx = y_hat == i

			sample_weight_idx = sample_weight[idx]
			self.distributions[i].fit(X[idx], sample_weight=sample_weight_idx)
			self.priors[i] = sample_weight_idx.sum() / sample_weight_sum

		self._initialized = True
		self._reset_cache()
		super()._initialize(X.shape[1])

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

		X = []
		for distribution in self.distributions:
			X_ = distribution.sample(n)
			X.append(X_)

		X = torch.stack(X)
		idxs = torch.multinomial(self.priors, num_samples=n, replacement=True)
		return X[idxs, torch.arange(n)]

	def fit(self, X, sample_weight=None, priors=None):
		"""Fit the model to optionally weighted examples.

		This method implements the core of the learning process. For a
		mixture model, this involves performing EM until the distributions that
		are being fit converge according to the threshold set by `tol`, or
		until the maximum number of iterations has been hit.

		This method is largely a wrapper around the `summarize` and
		`from_summaries` methods. It's primary contribution is serving as a
		loop around these functions and to monitor convergence.


		Parameters
		----------
		X: list, tuple, numpy.ndarray, torch.Tensor, shape=(-1, self.d)
			A set of examples to evaluate. 

		sample_weight: list, tuple, numpy.ndarray, torch.Tensor optional
			A set of weights for the examples. This can be either of shape
			(-1, self.d) or a vector of shape (-1,). Default is ones.

		priors: list, numpy.ndarray, torch.Tensor, shape=(-1, self.k)
			Prior probabilities of assigning each symbol to each node. If not
			provided, do not include in the calculations (conceptually
			equivalent to a uniform probability, but without scaling the
			probabilities). This can be used to assign labels to observatons
			by setting one of the probabilities for an observation to 1.0.
			This can be used when only some labels are known by using a
			uniform distribution when the labels are not known. Note that 
			this can be used to assign hard labels, but does not have the 
			same semantics for soft labels, in that it only influences the 
			initial estimate of an observation being generated by a component, 
			not gives a target. Default is None.


		Returns
		-------
		self
		"""

		logp = None
		for i in range(self.max_iter):
			start_time = time.time()

			last_logp = logp
			logp = self.summarize(X, sample_weight=sample_weight, 
				priors=priors)

			if i > 0:
				improvement = logp - last_logp
				duration = time.time() - start_time

				if self.verbose:
					print("[{}] Improvement: {}, Time: {:4.4}s".format(i, 
						improvement, duration))

				if improvement < self.tol:
					break

			self.from_summaries()

		self._reset_cache()
		return self

	def summarize(self, X, sample_weight=None, priors=None):
		"""Extract the sufficient statistics from a batch of data.

		This method calculates the sufficient statistics from optionally
		weighted data and adds them to the stored cache. The examples must be
		given in a 2D format. Sample weights can either be provided as one
		value per example or as a 2D matrix of weights for each feature in
		each example. Labels can be provided for examples but, if provided,
		must be incomplete such that semi-supervised learning can be performed.

		For a mixture model, this step is essentially performing the 'E' part
		of the EM algorithm on a batch of data, where examples are soft-assigned
		to distributions in the model and summaries are derived from that.


		Parameters
		----------
		X: list, tuple, numpy.ndarray, torch.Tensor, shape=(-1, self.d)
			A set of examples to summarize.

		sample_weight: list, tuple, numpy.ndarray, torch.Tensor, optional
			A set of weights for the examples. This can be either of shape
			(-1, self.d) or a vector of shape (-1,). Default is ones.

		priors: list, numpy.ndarray, torch.Tensor, shape=(-1, self.k)
			Prior probabilities of assigning each symbol to each node. If not
			provided, do not include in the calculations (conceptually
			equivalent to a uniform probability, but without scaling the
			probabilities). This can be used to assign labels to observatons
			by setting one of the probabilities for an observation to 1.0.
			Note that this can be used to assign hard labels, but does not
			have the same semantics for soft labels, in that it only
			influences the initial estimate of an observation being generated
			by a component, not gives a target. Default is None.


		Returns
		-------
		logp: float
			The log probability of X given the model.
		"""

		X = _check_parameter(_cast_as_tensor(X), "X", ndim=2)
		if not self._initialized:
			self._initialize(X, sample_weight=sample_weight)

		sample_weight = _reshape_weights(X, _cast_as_tensor(sample_weight, 
			dtype=torch.float32), device=self.device)

		e = self._emission_matrix(X, priors=priors)
		logp = torch.logsumexp(e, dim=1, keepdims=True)
		y = torch.exp(e - logp)

		z = torch.clone(self._w_sum)

		for i, d in enumerate(self.distributions):
			d.summarize(X, y[:, i:i+1] * sample_weight)

			if self.frozen == False:
				self._w_sum[i] = self._w_sum[i] + (y[:, i:i+1] * 
					sample_weight).mean(dim=-1).sum()

		return torch.sum(logp)

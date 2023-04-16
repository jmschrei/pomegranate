# kmeans.py
# Author: Jacob Schreiber

import time
import torch

from ._utils import _cast_as_tensor
from ._utils import _cast_as_parameter
from ._utils import _update_parameter
from ._utils import _check_parameter
from ._utils import _reshape_weights
from ._utils import _initialize_centroids

from ._utils import eps

class KMeans(torch.nn.Module):
	"""A KMeans clustering object.

	Although K-means clustering is not a probabilistic model by itself,
	necessarily, it can be a useful initialization for many probabilistic
	methods, and can also be thought of as a specific formulation of a mixture
	model. Specifically, if you have a Gaussian mixture model with diagonal
	covariances set to 1/inf, in theory you will get the exact same results
	as k-means clustering.

	The implementation is provided here not necessarily to compete with other
	implementations, but simply to use as a consistent initialization.


	Parameters
	----------
	k: int or None, optional
		The number of clusters to initialize. Default is None

	centroids: list, numpy.ndarray, torch.Tensor, or None, optional
		A set of centroids to use to initialize the algorithm. Default is None.

	init: str, optional
		The initialization to use if `centroids` are not provided. Default is
		'first-k'. Must be one of:

			'first-k': Use the first k examples from the data set
			'random': Use a random set of k examples from the data set
			'submodular-facility-location': Use a facility location submodular
				objective to initialize the k-means algorithm
			'submodular-feature-based': Use a feature-based submodular objective
				to initialize the k-means algorithm.

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

	verbose: bool, optional
		Whether to print the improvement and timings during training.
	"""

	def __init__(self, k=None, centroids=None, init='first-k', max_iter=10, 
		tol=0.1, inertia=0.0, frozen=False, random_state=None, verbose=False):
		super().__init__()
		self.name = "KMeans"
		self._device = _cast_as_parameter([0.0])

		self.centroids = _check_parameter(_cast_as_parameter(centroids, 
			dtype=torch.float32), "centroids", ndim=2)

		self.k = _check_parameter(_cast_as_parameter(k), "k", ndim=0, 
			min_value=2, dtypes=(int, torch.int32, torch.int64))

		self.init = _check_parameter(init, "init", value_set=("random", 
			"first-k", "submodular-facility-location", 
			"submodular-feature-based"), ndim=0, dtypes=(str,))
		self.max_iter = _check_parameter(_cast_as_tensor(max_iter), "max_iter",
			ndim=0, min_value=1, dtypes=(int, torch.int32, torch.int64))
		self.tol = _check_parameter(_cast_as_tensor(tol), "tol", ndim=0,
			min_value=0)
		self.inertia = _check_parameter(_cast_as_tensor(inertia), "inertia",
			ndim=0, min_value=0.0, max_value=1.0)
		self.frozen = _check_parameter(_cast_as_tensor(frozen), "frozen",
			ndim=0, value_set=(True, False))
		self.random_state = random_state
		self.verbose = _check_parameter(verbose, "verbose", 
			value_set=(True, False))

		if self.k is None and self.centroids is None:
			raise ValueError("Must specify one of `k` or `centroids`.")

		self.k = len(centroids) if centroids is not None else self.k
		self.d = len(centroids[0]) if centroids is not None else None
		self._initialized = centroids is not None
		self._reset_cache()

	@property
	def device(self):
		try:
			return next(self.parameters()).device
		except:
			return 'cpu'

	def _initialize(self, X):
		"""Initialize the probability distribution.

		This method is meant to only be called internally. It initializes the
		parameters of the distribution and stores its dimensionality. For more
		complex methods, this function will do more.


		Parameters
		----------
		X: list, numpy.ndarray, torch.Tensor, shape=(-1, self.d)
			The data to use to initialize the model.
		"""

		X = _check_parameter(_cast_as_tensor(X), "X", ndim=2)
		centroids = _initialize_centroids(X, self.k, algorithm=self.init, 
			random_state=self.random_state)
		
		if isinstance(centroids, torch.masked.MaskedTensor):
			centroids = centroids._masked_data * centroids._masked_mask

		if centroids.device != self.device:
			centroids = centroids.to(self.device)

		self.centroids = _cast_as_parameter(centroids)

		self.d = X.shape[1]
		self._initialized = True
		self._reset_cache()

	def _reset_cache(self):
		"""Reset the internally stored statistics.

		This method is meant to only be called internally. It resets the
		stored statistics used to update the model parameters as well as
		recalculates the cached values meant to speed up log probability
		calculations.
		"""

		if self._initialized == False:
			return

		self.register_buffer("_w_sum", torch.zeros(self.k, self.d, 
			device=self.device))
		self.register_buffer("_xw_sum", torch.zeros(self.k, self.d,
			device=self.device))
		self.register_buffer("_centroid_sum", torch.sum(self.centroids**2, 
			dim=1).unsqueeze(0))

	def _distances(self, X):
		"""Calculate the distances between each example and each centroid.

		This method calculates the distance between each example and each
		centroid in the model. These distances make up the backbone of the
		k-means learning and prediction steps.


		Parameters
		----------
		X: list, tuple, numpy.ndarray, torch.Tensor, shape=(-1, self.d)
			A set of examples to summarize.


		Returns
		-------
		distances: torch.Tensor, shape=(-1, self.k)
			The Euclidean distance between each example and each cluster.
		"""

		X = _check_parameter(_cast_as_tensor(X, dtype=torch.float32), "X", 
			ndim=2, shape=(-1, self.d))

		XX = torch.sum(X**2, dim=1).unsqueeze(1)

		if isinstance(X, torch.masked.MaskedTensor):
			n = X._masked_mask.sum(dim=1).unsqueeze(1)
			Xc = torch.matmul(X._masked_data * X._masked_mask, self.centroids.T)
		else:
			n = X.shape[1]
			Xc = torch.matmul(X, self.centroids.T)
		
		distances = torch.empty(X.shape[0], self.k, dtype=X.dtype, 
			device=self.device)
		distances[:] = torch.clamp(XX - 2*Xc + self._centroid_sum, min=0)
		return torch.sqrt(distances / n)

	def predict(self, X):
		"""Calculate the cluster assignment for each example.

		This method calculates cluster assignment for each example as the
		nearest centroid according to the Euclidean distance.


		Parameters
		----------
		X: list, tuple, numpy.ndarray, torch.Tensor, shape=(-1, self.d)
			A set of examples to summarize.


		Returns
		-------
		y: torch.Tensor, shape=(-1,)
			The predicted label for each example.
		"""

		return self._distances(X).argmin(dim=1)

	def summarize(self, X, sample_weight=None):
		"""Extract the sufficient statistics from a batch of data.

		This method calculates the sufficient statistics from optionally
		weighted data and adds them to the stored cache. The examples must be
		given in a 2D format. Sample weights can either be provided as one
		value per example or as a 2D matrix of weights for each feature in
		each example.

		For k-means clustering, this step is essentially performing the 'E' part
		of the EM algorithm on a batch of data, where examples are hard-assigned
		to distributions in the model and summaries are derived from that.


		Parameters
		----------
		X: list, tuple, numpy.ndarray, torch.Tensor, shape=(-1, self.d)
			A set of examples to summarize.

		sample_weight: list, tuple, numpy.ndarray, torch.Tensor, optional
			A set of weights for the examples. This can be either of shape
			(-1, self.d) or a vector of shape (-1,). Default is ones.
		"""

		if self.frozen:
			return 0

		if not self._initialized:
			self._initialize(X)

		X = _check_parameter(_cast_as_tensor(X, dtype=torch.float32), "X", 
			ndim=2, shape=(-1, self.d))
		sample_weight = _reshape_weights(X, _cast_as_tensor(sample_weight, 
			dtype=torch.float32), device=self.device)

		distances = self._distances(X)
		y_hat = distances.argmin(dim=1)

		if isinstance(X, torch.masked.MaskedTensor):
			for i in range(self.k):
				idx = y_hat == i

				self._w_sum[i][:] = self._w_sum[i] + sample_weight[idx].sum(dim=0)
				self._xw_sum[i][:] = self._xw_sum[i] + (X[idx] * 
					sample_weight[idx]).sum(dim=0)

		else:
			y_hat = y_hat.unsqueeze(1).expand(-1, self.d)
			self._w_sum.scatter_add_(0, y_hat, sample_weight)
			self._xw_sum.scatter_add_(0, y_hat, X * sample_weight)
		
		return distances.min(dim=1).values.sum()

	def from_summaries(self):
		"""Update the model parameters given the extracted statistics.

		This method uses calculated statistics from calls to the `summarize`
		method to update the distribution parameters. Hyperparameters for the
		update are passed in at initialization time.

		Note: Internally, a call to `fit` is just a successive call to the
		`summarize` method followed by the `from_summaries` method.
		"""

		if self.frozen:
			return

		centroids = self._xw_sum / self._w_sum
		_update_parameter(self.centroids, centroids, self.inertia)
		self._reset_cache()

	def fit(self, X, sample_weight=None):
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

		sample_weight: list, tuple, numpy.ndarray, torch.Tensor, optional
			A set of weights for the examples. This can be either of shape
			(-1, self.d) or a vector of shape (-1,). Default is ones.


		Returns
		-------
		self
		"""

		d_current = None
		for i in range(self.max_iter):
			start_time = time.time()

			d_previous = d_current
			d_current = self.summarize(X, sample_weight=sample_weight)

			if i > 0:
				improvement = d_previous - d_current
				duration = time.time() - start_time

				if self.verbose:
					print("[{}] Improvement: {}, Time: {:4.4}s".format(i, 
						improvement, duration))

				if improvement < self.tol:
					break

			self.from_summaries()

		self._reset_cache()
		return self

	def fit_predict(self, X, sample_weight=None):
		"""Fit the model and then return the predictions.

		This function wraps a call to the `fit` function and then to the
		`predict` function. Essentially, k-means will be fit to the data and
		the resulting clustering assignments will be returned.


		Parameters
		----------
		X: list, tuple, numpy.ndarray, torch.Tensor, shape=(-1, self.d)
			A set of examples to evaluate. 

		sample_weight: list, tuple, numpy.ndarray, torch.Tensor, optional
			A set of weights for the examples. This can be either of shape
			(-1, self.d) or a vector of shape (-1,). Default is ones.


		Returns
		-------
		y: torch.Tensor, shape=(-1,)
			Cluster assignments for each example.
		"""

		self.fit(X, sample_weight=sample_weight)
		return self.predict(X)
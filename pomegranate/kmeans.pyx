#cython: boundscheck=False
#cython: cdivision=True
# kmeans.pyx
# Contact: Jacob Schreiber ( jmschreiber91@gmail.com )

from libc.stdlib cimport calloc
from libc.stdlib cimport free
from libc.string cimport memset
from libc.string cimport memcpy
from libc.math cimport log10 as clog10
from libc.math cimport sqrt as csqrt

from scipy.linalg.cython_blas cimport ddot

from .base cimport Model

from .utils cimport ndarray_wrap_cpointer
from .utils cimport mdot
from .utils cimport _is_gpu_enabled
from .utils cimport isnan

import time
import json
import numpy
cimport numpy

from joblib import Parallel
from joblib import delayed

try:
	import cupy
except:
	cupy = object

DEF NEGINF = float("-inf")
DEF INF = float("inf")

cdef double distance(double* X, double* centroid, int d) nogil:
	cdef int i
	cdef double distance = 0.0

	for i in range(d):
		if not isnan(X[i]):
			distance += (X[i] - centroid[i]) ** 2

	return distance

cpdef numpy.ndarray initialize_centroids(numpy.ndarray X, weights, int k,
	init='first-k', double oversampling_factor=0.95):
	"""Initialize the centroids for kmeans given a dataset.

	This function will take in a dataset and return the centroids found using
	some method. This is the initialization step of kmeans.

	Parameters
	----------
	X : numpy.ndarray, shape=(n_samples, n_dim)
		The dataset to identify centroids in.

	weights : numpy.ndarray, shape=(n_samples,)
		The weights associated with each of the samples.

	k : int
		The number of centroids to extract.

	init : str, one of 'first-k', 'random', 'kmeans++', 'kmeans||'
		'first-k' : use the first k samples as the centroids
		'random' : randomly select k samples as the centroids
		'kmeans++' : use the kmeans++ initialization algorithm, which
			iteratively selects the next centroid randomly, but weighted
			based on distance to nearest centroid, to be likely to choose
			good initializations
		'kmeans||' : use the scalable kmeans++ initialization algorithm,
			as described in http://theory.stanford.edu/~sergei/papers/vldb12-kmpar.pdf

	Returns
	-------
	centroids : numpy.ndarray, shape=(k, d)
		The centroids to use initially for clustering. There are k centroids,
		one for each cluster, and they span d dimensions.
	"""

	cdef int n = X.shape[0], d = X.shape[1]
	cdef int count, i, j, l, m
	cdef double dist
	cdef numpy.ndarray centroids
	cdef numpy.ndarray min_distance
	cdef numpy.ndarray min_idxs
	cdef numpy.ndarray mask

	cdef double* X_ptr = <double*> X.data
	cdef double* weights_ptr
	cdef double* min_distance_ptr
	cdef double* centroids_ptr
	cdef double phi

	if weights is None:
		weights = numpy.ones(len(X), dtype='float64') / len(X)
	else:
		weights = weights.astype('float64') / sum(weights)

	weights_ptr = <double*> (<numpy.ndarray> weights).data

	if init not in ('first-k', 'random', 'kmeans++', 'kmeans||'):
		raise ValueError("initialization must be one of 'first-k', 'random', 'kmeans++', or 'kmeans||'")

	if init == 'first-k':
		centroids = X[:k].copy()

	elif init == 'random':
		idxs = numpy.random.choice(n, size=k, replace=False, p=weights)
		centroids = X[idxs].copy()

	elif init == 'kmeans++':
		centroids = numpy.zeros((k, d), dtype='float64')
		centroids_ptr = <double*> centroids.data

		idx = numpy.random.choice(n, p=weights)
		centroids[0] = X[idx]
		centroids[0][numpy.isnan(centroids[0])] = 0.0

		min_distance = numpy.zeros(n, dtype='float64') + INF
		min_distance_ptr = <double*> min_distance.data

		for m in range(k-1):
			for i in range(n):
				dist = distance(X_ptr + i*d, centroids_ptr + m*d, d)
				dist *= weights_ptr[i]

				if dist < min_distance_ptr[i]:
					min_distance_ptr[i] = dist

			idx = numpy.random.choice(n, p=min_distance / min_distance.sum())
			centroids[m+1] = X[idx]
			centroids[m+1][numpy.isnan(centroids[m+1])] = 0.0

	elif init == 'kmeans||':
		centroids = numpy.zeros((1, d))
		idx = numpy.random.choice(n, p=weights)
		centroids[0] = X[idx]
		centroids[0][numpy.isnan(centroids[0])] = 0
		centroids_ptr = <double*> centroids.data

		min_distance = numpy.zeros(n, dtype='float64')
		min_distance_ptr = <double*> min_distance.data

		for i in range(n):
			min_distance_ptr[i] = distance(X_ptr + i*d, centroids_ptr, d)

		min_idxs = numpy.zeros(n, dtype='int32')
		min_idxs_ptr = <int*> min_idxs.data

		phi = min_distance.sum()
		count = 1

		for iteration in range(int(clog10(phi))):
			prob = numpy.random.uniform(0, 1, size=(n,))
			thresh = oversampling_factor * min_distance / phi

			centroids = numpy.concatenate((centroids, X[prob < thresh]))
			centroids[numpy.isnan(centroids)] = 0.0
			centroids_ptr = <double*> centroids.data
			m = centroids.shape[0] - count

			if m > 0:
				for i in range(n):
					for l in range(m):
						dist = distance(X_ptr + i*d, centroids_ptr + (l+count)*d, d)
						if dist < min_distance_ptr[i]:
							phi += dist - min_distance_ptr[i]
							min_distance_ptr[i] = dist
							min_idxs_ptr[i] = l + count

				count = centroids.shape[0]

		w = numpy.bincount(min_idxs)

		clf = Kmeans(k, 'kmeans++', 1)
		clf.fit(centroids, w)
		centroids = clf.centroids

	centroids[numpy.isnan(centroids)] = 0.0
	return numpy.array(centroids)

cdef class Kmeans(Model):
	"""A kmeans model.

	Kmeans is not a probabilistic model, but it is used in the kmeans++
	initialization for GMMs. In essence, a point is selected as the center for
	one component and then remaining points are selected.

	Parameters
	----------
	k : int
		The number of centroids.

	init : str or list, optional
		The initialization scheme for k-means. If str, must be one of 'first-k',
		'random', 'kmeans++', or 'kmeans||'. First-k uses the first k samples
		as the centroids. Random will randomly assign points to be the
		centroids. Kmeans++ uses the kmeans++ initialization algorithm, and
		kmeans|| uses the scalable kmeans algorithm. If a list or array is
		passed in, these values are used as the centroids. Default is kmeans++.

	n_init : int, optional
		The number of initializations to do before taking the best. This can
		help reduce the effect of converging to a local optima by sampling
		multiple local optima and taking the best.


	Attributes
	----------
	k : int
		The number of centroids

	centroids : array-like, shape (k, n_dim)
		The means of the centroid points.
	"""

	cdef public int k
	cdef int n_init
	cdef str init
	cdef public numpy.ndarray centroids
	cdef numpy.ndarray centroids_T
	cdef double* centroids_ptr
	cdef double* centroids_T_ptr
	cdef double* summary_sizes
	cdef double* summary_weights
	cdef double* centroid_norms

	def __init__(self, k, init='kmeans++', n_init=10):
		self.k = k
		self.d = 0
		self.n_init = n_init
		self.centroid_norms = <double*> calloc(self.k, sizeof(double))

		if isinstance(init, (list, numpy.ndarray)):
			self.centroids = numpy.array(init, dtype='float64', ndmin=2)
			self.centroids_ptr = <double*> self.centroids.data
			self.centroids_T = self.centroids.T.copy()
			self.centroids_T_ptr = <double*> self.centroids_T.data
			self.d = self.centroids.shape[1]

			for i in range(self.k):
				self.centroid_norms[i] = self.centroids[i].dot(self.centroids[i])

			self.init = 'fixed'
		elif isinstance(init, str):
			self.init = init

	def __dealloc__(self):
		free(self.centroid_norms)

	def predict(self, X, n_jobs=1):
		"""Predict nearest centroid for each point.

		Parameters
		----------
		X : array-like, shape (n_samples, n_dim)
			The data to fit to.

		n_jobs : int
			The number of jobs to use to parallelize, either the number of threads
			or the number of processes to use. -1 means use all available resources.
			Default is 1.

		Returns
		-------
		y : array-like, shape (n_samples,)
			The index of the nearest centroid.
		"""

		starts = [int(i*len(X)/n_jobs) for i in range(n_jobs)]
		ends = [int(i*len(X)/n_jobs) for i in range(1, n_jobs+1)]

		if n_jobs > 1:
			with Parallel(n_jobs=n_jobs, backend='threading') as parallel:
				y_pred = parallel(delayed(self.predict, check_pickle=False)(
					X[start:end]) for start, end in zip(starts, ends))

				return numpy.concatenate(y_pred)

		X = numpy.array(X, dtype='float64')
		cdef double* X_ptr = <double*> (<numpy.ndarray> X).data
		cdef int n = len(X)

		cdef numpy.ndarray y = numpy.zeros(n, dtype='int32')
		cdef int* y_ptr = <int*> y.data

		self.d = X.shape[1]

		with nogil:
			self._predict(X_ptr, y_ptr, n)

		return y

	cdef void _predict(self, double* X, int* y, int n) nogil:
		cdef int i, j, l, k = self.k, d = self.d
		cdef double dist, min_dist
		cdef double* centroid

		for i in range(n):
			min_dist = INF

			for j in range(k):
				centroid = self.centroids_ptr + j*d
				dist = distance(X + i*d, centroid, d)

				if dist < min_dist:
					min_dist = dist
					y[i] = j

	def distance(self, X):
		"""Calculate the distance to each centroid.

		Parameters
		----------
		X : array-like, shape (n_samples, n_dim)
			The data to fit to.

		Returns
		-------
		y : array-like, shape (n_samples, n_centroids)
			The index of the nearest centroid.
		"""

		cdef int i, j, n = X.shape[0], d = self.d
		cdef double* X_ptr
		cdef double* centroid

		X = numpy.array(X, dtype='float64')
		X_ptr = <double*> (<numpy.ndarray> X).data

		dist = numpy.zeros((X.shape[0], self.k))


		for i in range(X.shape[0]):
			for j in range(self.k):
				centroid = self.centroids_ptr + j*d
				dist[i, j] = distance(X_ptr + i*d, centroid, d)

		return dist

	def fit(self, X, weights=None, inertia=0.0, stop_threshold=1e-3,
                max_iterations=1e3, batch_size=None, batches_per_epoch=None,
                clear_summaries=False, verbose=False, n_jobs=1):
		"""Fit the model to the data using k centroids.

		Parameters
		----------
		X : array-like, shape (n_samples, n_dim)
			The data to fit to.

		weights : array-like, shape (n_samples,), optional
			The initial weights of each sample in the matrix. If nothing is
			passed in then each sample is assumed to be the same weight.
			Default is None.

		inertia : double, optional
			The weight of the previous parameters of the model. The new
			parameters will roughly be
			old_param*inertia + new_param*(1-inertia), so an inertia of 0 means
			ignore the old parameters, whereas an inertia of 1 means ignore
			the new parameters. Default is 0.0.

		stop_threshold : double, optional, positive
			The threshold at which EM will terminate for the improvement of
			the model. If the model does not improve its fit of the data by a
			log probability of 0.1 then terminate. Default is 0.1.

		max_iterations : int, optional
			The maximum number of iterations to run for. Default is 1e3.

		batch_size : int or None, optional
			The number of samples in a batch to summarize on. This controls
			the size of the set sent to `summarize` and so does not make the
			update any less exact. This is useful when training on a memory
			map and cannot load all the data into memory. If set to None,
			batch_size is 1 / n_jobs. Default is None.

		batches_per_epoch : int or None, optional
			The number of batches in an epoch. This is the number of batches to
			summarize before calling `from_summaries` and updating the model
			parameters. This allows one to do minibatch updates by updating the
			model parameters before setting the full dataset. If set to None,
			uses the full dataset. Default is None.

		clear_summaries : bool or 'auto', optional
			Whether to clear the stored sufficient statistics after an update.
			Typically this would be set to True as you want to recollect new
			statistics after each epoch. However, in either the streaming or
			the minibatching setting one may want to set this to False to
			collect statistics on the entire dataset. 'auto' means set to True
			if batches_per_epoch is None, otherwise False.

		verbose : bool, optional
			Whether or not to print out improvement information over
			iterations. Default is False.

		n_jobs : int, optional
			The number of threads to use when processing data. Default to 1,
			meaning no parallelism.

		Returns
		-------
		self : Kmeans
			This is the fit kmeans object.
		"""

		if not isinstance(X, numpy.ndarray):
			X = numpy.array(X, dtype='float64')

		n, d = X.shape

		best_centroids, best_distance = None, INF
		training_start_time = time.time()

		self.d = d
		self.summary_sizes = <double*> calloc(self.k*d, sizeof(double))
		self.summary_weights = <double*> calloc(self.k*d, sizeof(double))

		if batch_size is None:
			starts = [int(i*len(X)/n_jobs) for i in range(n_jobs)]
			ends = [int(i*len(X)/n_jobs) for i in range(1, n_jobs+1)]
		else:
			starts = list(range(0, n, batch_size))
			if starts[-1] == n:
				del starts[-1]
			ends = list(range(batch_size, n, batch_size)) + [n]

		if clear_summaries == 'auto':
			clear_summaries = True if batches_per_epoch == None else False

		batches_per_epoch = batches_per_epoch or len(starts)
		n_seen_batches = 0

		if self.centroids is not None:
			initial_centroids = self.centroids.copy()
		else:
			initial_centroids = None

		with Parallel(n_jobs=n_jobs, backend='threading') as parallel:
			for i in range(self.n_init):
				self.clear_summaries()
				initial_distance_sum = INF
				iteration, improvement = 0, INF

				if weights is not None and initial_centroids is None:
					self.centroids = initialize_centroids(X[:ends[0]],
						weights[:ends[0]], self.k, self.init)
				elif weights is None and initial_centroids is None:
					self.centroids = initialize_centroids(X[:ends[0]],
						None, self.k, self.init)
				else:
					self.centroids = initial_centroids.copy()

				self.centroids_ptr = <double*> self.centroids.data
				self.centroids_T = self.centroids.T.copy()
				self.centroids_T_ptr = <double*> self.centroids_T.data

				for i in range(self.k):
					self.centroid_norms[i] = self.centroids[i].dot(self.centroids[i])

				while improvement > stop_threshold and iteration < max_iterations + 1:
					epoch_start_time = time.time()

					epoch_starts = starts[n_seen_batches:n_seen_batches+batches_per_epoch]
					epoch_ends = ends[n_seen_batches:n_seen_batches+batches_per_epoch]

					n_seen_batches += batches_per_epoch
					if n_seen_batches >= len(starts):
						n_seen_batches = 0

					self.from_summaries(inertia, clear_summaries)

					if weights is not None:
						distance_sum = sum(parallel(delayed(self.summarize,
							check_pickle=False)(X[start:end], weights[start:end])
							for start, end in zip(epoch_starts, epoch_ends)))
					else:
						distance_sum = sum(parallel(delayed(self.summarize,
							check_pickle=False)(X[start:end]) for start, end in zip(
								epoch_starts, epoch_ends)))

					if iteration == 0:
						initial_distance_sum = distance_sum
					else:
						improvement = last_distance_sum - distance_sum
						time_spent = time.time() - epoch_start_time

						if verbose:
							print("[{}] Improvement: {}\tTime (s): {:4.4}".format(iteration,
								improvement, time_spent))

					iteration += 1
					last_distance_sum = distance_sum

				total_improvement = initial_distance_sum - last_distance_sum

				if verbose:
					total_time_spent = time.time() - training_start_time
					print("Total Improvement: {}".format(total_improvement))
					print("Total Time (s): {:.4f}".format(total_time_spent))

				if last_distance_sum < best_distance:
					best_centroids = self.centroids.copy()
					best_distance = last_distance_sum

		free(self.summary_sizes)
		free(self.summary_weights)

		self.centroids = best_centroids
		self.centroids_ptr = <double*> self.centroids.data
		self.centroids_T = self.centroids.T.copy()
		self.centroids_T_ptr = <double*> self.centroids_T.data

		for i in range(self.k):
			self.centroid_norms[i] = self.centroids[i].dot(self.centroids[i])

		return self

	def summarize(self, X, weights=None):
		"""Summarize the points into sufficient statistics for a future update.

		Parameters
		----------
		X : array-like, shape (n_samples, n_dim)
			The data to fit to.

		weights : array-like, shape (n_samples,), optional
			The initial weights of each sample in the matrix. If nothing is
			passed in then each sample is assumed to be the same weight.
			Default is None.

		Returns
		-------
		dist : double
			The negative total euclidean distance between each point and its
			nearest centroid. This is not a probability, and the negative is
			returned to fit in with the idea of large negative numbers being
			worse than smaller negative numbers, such as with log
			probabilities.
		"""

		cdef numpy.ndarray X_ndarray = numpy.array(X, dtype='float64')
		cdef double* X_ptr = <double*> X_ndarray.data

		cdef numpy.ndarray weights_ndarray
		cdef int i, j, n = X_ndarray.shape[0], d = X_ndarray.shape[1]
		cdef double dist

		if weights is None:
			weights_ndarray = numpy.ones(n, dtype='float64')
		else:
			weights_ndarray = numpy.array(weights, dtype='float64')

		cdef double* weights_ptr = <double*> weights_ndarray.data

		with nogil:
			dist = self._summarize(X_ptr, weights_ptr, n, 0, self.d)

		return dist

	cdef double _summarize(self, double* X, double* weights, int n,
		int column_idx, int d) nogil:
		cdef int i, j, l, y, k = self.k, inc = 1
		cdef double min_dist, dist, total_dist = 0.0, pdist
		cdef double* summary_sizes = <double*> calloc(k*d, sizeof(double))
		cdef double* summary_weights = <double*> calloc(k*d, sizeof(double))
		memset(summary_sizes, 0, k*d*sizeof(double))
		memset(summary_weights, 0, k*d*sizeof(double))

		cdef double* dists = <double*> calloc(n*k, sizeof(double))
		memset(dists, 0, n*k*sizeof(double))

		cdef double* X_ = <double*> calloc(n*d, sizeof(double))
		memset(X_, 0, n*d*sizeof(double))

		cdef double* bias = <double*> calloc(n*k, sizeof(double))
		memset(bias, 0, n*k*sizeof(double))

		for i in range(n):
			for j in range(d):
				idx = i*d + j
				if isnan(X[idx]):
					X_[idx] = 0.0
					for l in range(k):
						bias[i*k + l] += self.centroids_T_ptr[j*k + l] ** 2

				else:
					X_[idx] = X[idx]

		mdot(X_, self.centroids_T_ptr, dists, n, k, d)

		for i in range(n):
			min_dist = INF
			pdist = ddot(&d, X_ + i*d, &inc, X_ + i*d, &inc)

			for j in range(k):
				dist = self.centroid_norms[j] + pdist - 2*dists[i*k + j] - bias[i*k + j]
				if dist < min_dist:
					min_dist = dist
					y = j

			total_dist += min_dist

			for l in range(d):
				if not isnan(X[i*d + l]):
					summary_sizes[y*d + l] += weights[i]
					summary_weights[y*d + l] += X[i*d + l] * weights[i]

		with gil:
			for j in range(k):
				for l in range(d):
					self.summary_sizes[j*d + l] += summary_sizes[j*d + l]
					self.summary_weights[j*d + l] += summary_weights[j*d + l]

		free(summary_sizes)
		free(summary_weights)
		free(dists)
		free(X_)
		free(bias)
		return total_dist

	def from_summaries(self, double inertia=0.0, clear_summaries=True):
		"""Fit the model to the collected sufficient statistics.

		Fit the parameters of the model to the sufficient statistics gathered
		during the summarize calls. This should return an exact update.

		Parameters
		----------
		inertia : double, optional
			The weight of the previous parameters of the model. The new
			parameters will roughly be
			old_param*inertia + new_param*(1-inertia),
			so an inertia of 0 means ignore the old parameters, whereas an
			inertia of 1 means ignore the new parameters. Default is 0.0.

		clear_summaries : boolean, optional
			Whether to clear the stored summaries after updating the models
			based on them. If set to False, this can be used for streaming
			updates where one updates the parameters of the model while seeing
			more data. Default is True.

		Returns
		-------
		None
		"""

		cdef int i, l, j, k = self.k, d = self.d
		cdef double w_sum = 0

		with nogil:
			for i in range(k*d):
				w_sum += self.summary_sizes[i]

			if w_sum > 1e-8:
				for j in range(k):
					for l in range(d):
						i = j*d + l

						if self.summary_weights[i] == 0:
							continue

						self.centroids_ptr[i] = \
							inertia * self.centroids_ptr[i] \
							+ (1-inertia) * self.summary_weights[i] \
							/ self.summary_sizes[i]

		self.centroids_T = self.centroids.T.copy()
		self.centroids_T_ptr = <double*> self.centroids_T.data

		for i in range(self.k):
			self.centroid_norms[i] = self.centroids[i].dot(self.centroids[i])

		if clear_summaries:
			self.clear_summaries()

	def clear_summaries(self):
		memset(self.summary_sizes, 0, self.k*self.d*sizeof(double))
		memset(self.summary_weights, 0, self.k*self.d*sizeof(double))

	def to_json(self, separators=(',', ' : '), indent=4):
		model = {
					'class' : 'Kmeans',
					'k' : self.k,
					'centroids'  : self.centroids.tolist()
				}

		return json.dumps(model, separators=separators, indent=indent)

	@classmethod
	def from_json(cls, s):
		d = json.loads(s)
		model = Kmeans(d['k'], d['centroids'])
		return model

	@classmethod
	def from_samples(cls, k, X, weights=None, init='kmeans++', n_init=10,
		inertia=0.0, stop_threshold=0.1, max_iterations=1e3, batch_size=None,
		batches_per_epoch=None, clear_summaries=False, verbose=False, n_jobs=1):
		"""
		Fit a k-means object to the data directly.

		Parameters
		----------
		k : int
			The number of centroids.

		X : array-like, shape (n_samples, n_dim)
			The data to fit to.

		weights : array-like, shape (n_samples,), optional
			The initial weights of each sample in the matrix. If nothing is
			passed in then each sample is assumed to be the same weight.
			Default is None.

		init : str or list, optional
			The initialization scheme for k-means. If str, must be one of 'first-k',
			'random', 'kmeans++', or 'kmeans||'. First-k uses the first k samples
			as the centroids. Random will randomly assign points to be the
			centroids. Kmeans++ uses the kmeans++ initialization algorithm, and
			kmeans|| uses the scalable kmeans algorithm. If a list or array is
			passed in, these values are used as the centroids. Default is kmeans++.

		n_init : int, optional
			The number of initializations to do before taking the best. This can
			help reduce the effect of converging to a local optima by sampling
			multiple local optima and taking the best.

		inertia : double, optional
			The weight of the previous parameters of the model. The new
			parameters will roughly be
			old_param*inertia + new_param*(1-inertia), so an inertia of 0 means
			ignore the old parameters, whereas an inertia of 1 means ignore
			the new parameters. Default is 0.0.

		stop_threshold : double, optional, positive
			The threshold at which EM will terminate for the improvement of
			the model. If the model does not improve its fit of the data by a
			log probability of 0.1 then terminate. Default is 0.1.

		max_iterations : int, optional
			The maximum number of iterations to run for. Default is 1e3.

		batch_size : int or None, optional
			The number of samples in a batch to summarize on. This controls
			the size of the set sent to `summarize` and so does not make the
			update any less exact. This is useful when training on a memory
			map and cannot load all the data into memory. If set to None,
			batch_size is 1 / n_jobs. Default is None.

		batches_per_epoch : int or None, optional
			The number of batches in an epoch. This is the number of batches to
			summarize before calling `from_summaries` and updating the model
			parameters. This allows one to do minibatch updates by updating the
			model parameters before setting the full dataset. If set to None,
			uses the full dataset. Default is None.

		clear_summaries : bool or 'auto', optional
			Whether to clear the stored sufficient statistics after an update.
			Typically this would be set to True as you want to recollect new
			statistics after each epoch. However, in either the streaming or
			the minibatching setting one may want to set this to False to
			collect statistics on the entire dataset. 'auto' means set to True
			if batches_per_epoch is None, otherwise False.

		verbose : bool, optional
			Whether or not to print out improvement information over
			iterations. Default is False.

		n_jobs : int, optional
			The number of threads to use. Default is 1, indicating no
			parallelism is used.
		"""

		if not isinstance(X, numpy.ndarray):
			X = numpy.array(X, dtype='float64')

		n, d = X.shape

		model = Kmeans(k=k, init=init, n_init=n_init)
		model.fit(X, weights, inertia=inertia, stop_threshold=stop_threshold,
			max_iterations=max_iterations, batch_size=batch_size,
			batches_per_epoch=batches_per_epoch, clear_summaries=clear_summaries,
			verbose=verbose, n_jobs=n_jobs)

		return model

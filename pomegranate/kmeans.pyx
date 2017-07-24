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


cpdef numpy.ndarray initialize_centroids(numpy.ndarray X, numpy.ndarray weights, int k,
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

	init : str, one of 'first-k', 'random', 'kmeans++'
		'first-k' : use the first k samples as the centroids
		'random' : randomly select k samples as the centroids
		'kmeans++' : use the kmeans++ initialization algorithm, which
			iteratively selects the next centroid randomly, but weighted
			based on distance to nearest centroid, to be likely to choose
			good initializations

	Returns
	-------
	centroids : numpy.ndarray, shape=(k, d)
		The centroids to use initially for clustering. There are k centroids,
		one for each cluster, and they span d dimensions.
	"""

	cdef int n = X.shape[0], d = X.shape[1]
	cdef int count, i, j, l, m
	cdef double distance
	cdef numpy.ndarray centroids
	cdef numpy.ndarray min_distance
	cdef numpy.ndarray min_idxs

	cdef double* X_ptr = <double*> X.data
	cdef double* weights_ptr = <double*> weights.data
	cdef double* min_distance_ptr
	cdef double* centroids_ptr
	cdef int* min_idxs_ptr
	cdef double phi

	weights = weights / weights.sum()

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

		min_distance = numpy.zeros(n, dtype='float64') + INF
		min_distance_ptr = <double*> min_distance.data

		for m in range(k-1):
			for i in range(n):
				distance = 0
				for j in range(d):
					distance += (X_ptr[i*d + j] - centroids_ptr[m*d + j]) ** 2
				distance *= weights_ptr[i]

				if distance < min_distance_ptr[i]:
					min_distance_ptr[i] = distance

			idx = numpy.random.choice(n, p=min_distance / min_distance.sum())
			centroids[m+1] = X[idx]

	elif init == 'kmeans||':
		phi = 0

		centroids = numpy.zeros((1, d))
		idx = numpy.random.choice(n, p=weights)
		centroids[0] = X[idx]

		min_distance = ((X - centroids[0]) ** 2).sum(axis=1)
		min_distance_ptr = <double*> min_distance.data

		min_idxs = numpy.zeros(n, dtype='int32')
		min_idxs_ptr = <int*> min_idxs.data

		phi = min_distance.sum()
		count = 1

		for iteration in range(int(clog10(phi))):
			prob = numpy.random.uniform(0, 1, size=(n,))
			thresh = oversampling_factor * min_distance / phi

			centroids = numpy.concatenate((centroids, X[prob < thresh]))
			centroids_ptr = <double*> centroids.data
			m = centroids.shape[0] - count

			if m > 0:
				for i in range(n):
					for l in range(m):
						distance = 0
						for j in range(d):
							distance += (X_ptr[i*d + j] - centroids_ptr[(l + count)*d + j]) ** 2

						if distance < min_distance_ptr[i]:
							phi += distance - min_distance_ptr[i]
							min_distance_ptr[i] = distance
							min_idxs_ptr[i] = l + count

				count = centroids.shape[0]

		w = numpy.bincount(min_idxs)

		clf = Kmeans(k, 'kmeans++', 1)
		clf.fit(centroids, w)
		centroids = clf.centroids

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

		for i in range(n):
			min_dist = INF

			for j in range(k):
				dist = 0.0

				for l in range(d):
					dist += (X[i*d + l] - self.centroids_ptr[j*d + l]) ** 2.0

				if dist < min_dist:
					min_dist = dist
					y[i] = j

	def fit(self, X, weights=None, inertia=0.0, stop_threshold=1e-3,
                max_iterations=1e3, verbose=False, n_jobs=1):
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

		X = numpy.array(X, dtype='float64')
		n, d = X.shape

		if weights is None:
			weights = numpy.ones(n, dtype='float64')
		else:
			weights = numpy.array(weights, dtype='float64')

		best_centroids, best_distance = None, INF
		training_start_time = time.time()

		self.d = d
		self.summary_sizes = <double*> calloc(self.k, sizeof(double))
		self.summary_weights = <double*> calloc(self.k*d, sizeof(double))

		starts = [int(i*len(X)/n_jobs) for i in range(n_jobs)]
		ends = [int(i*len(X)/n_jobs) for i in range(1, n_jobs+1)]

		with Parallel(n_jobs=n_jobs, backend='threading') as parallel:
			for i in range(self.n_init):
				self.clear_summaries()
				initial_distance_sum = INF
				iteration, improvement = 0, INF

				self.centroids = initialize_centroids(X, weights, self.k, self.init)
				self.centroids_ptr = <double*> self.centroids.data
				self.centroids_T = self.centroids.T.copy()
				self.centroids_T_ptr = <double*> self.centroids_T.data

				for i in range(self.k):
					self.centroid_norms[i] = self.centroids[i].dot(self.centroids[i])

				while improvement > stop_threshold and iteration < max_iterations + 1:
					epoch_start_time = time.time()

					self.from_summaries(inertia)

					distance_sum = sum( parallel(delayed(self.summarize, 
						check_pickle=False)(X[start:end], weights[start:end]) 
						for start, end in zip(starts, ends) ))

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
			nearest centroid. This is not a probabilitity, and the negative is
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
			dist = self._summarize(X_ptr, weights_ptr, n)

		return dist

	cdef double _summarize(self, double* X, double* weights, int n) nogil:
		cdef int i, j, l, y, k = self.k, d = self.d, inc = 1
		cdef double min_dist, dist, total_dist, pdist = 0.0
		cdef double* summary_sizes = <double*> calloc(k, sizeof(double))
		cdef double* summary_weights = <double*> calloc(k*d, sizeof(double))
		memset(summary_sizes, 0, k*sizeof(double))
		memset(summary_weights, 0, k*d*sizeof(double))

		cdef double* dists = <double*> calloc(n*k, sizeof(double))
		memset(dists, 0, n*k*sizeof(double))
		mdot(X, self.centroids_T_ptr, dists, n, k, d)

		for i in range(n):
			min_dist = INF
			pdist = ddot(&d, X + i*d, &inc, X + i*d, &inc)

			for j in range(k):
				dist = self.centroid_norms[j] + pdist - 2*dists[i*k + j]

				if dist < min_dist:
					min_dist = dist
					y = j
			
			total_dist += min_dist
			summary_sizes[y] += weights[i]
			
			for l in range(d):
				summary_weights[y*d + l] += X[i*d + l] * weights[i]

		with gil:
			for j in range(k):
				self.summary_sizes[j] += summary_sizes[j]

				for l in range(d):
					self.summary_weights[j*d + l] += summary_weights[j*d + l]

		free(summary_sizes)
		free(summary_weights)
		free(dists)
		return total_dist

	def from_summaries(self, double inertia=0.0):
		cdef int l, j, k = self.k, d = self.d
		cdef double w_sum = 0

		with nogil:
			for i in range(k):
				w_sum += self.summary_sizes[i]

			if w_sum > 1e-8:
				for j in range(k):
					for l in range(d):
						self.centroids_ptr[j*d + l] = \
							inertia * self.centroids_ptr[j*d + l] \
							+ (1-inertia) * self.summary_weights[j*d + l] \
							/ self.summary_sizes[j]

		self.centroids_T = self.centroids.T.copy()
		self.centroids_T_ptr = <double*> self.centroids_T.data

		for i in range(self.k):
			self.centroid_norms[i] = self.centroids[i].dot(self.centroids[i])

		self.clear_summaries()

	def clear_summaries(self):
		memset(self.summary_sizes, 0, self.k*sizeof(double))
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
	def from_samples(cls, k, X, weights=None, init='kmeans++', n_init=10, inertia=0.0,
		stop_threshold=0.1, max_iterations=1e3, verbose=False, n_jobs=1):
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

		verbose : bool, optional
			Whether or not to print out improvement information over
			iterations. Default is False.

		n_jobs : int, optional
			The number of threads to use. Default is 1, indicating no
			parallelism is used.
		"""

		X = numpy.array(X, dtype='float64')
		n, d = X.shape

		if weights is None:
			weights = numpy.ones(n, dtype='float64')
		else:
			weights = numpy.array(weights, dtype='float64')

		model = Kmeans(k=k, init=init, n_init=n_init)
		model.fit(X, weights, inertia=inertia, stop_threshold=stop_threshold,
			max_iterations=max_iterations, verbose=verbose, n_jobs=n_jobs)

		return model
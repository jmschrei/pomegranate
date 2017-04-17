#cython: boundscheck=False
#cython: cdivision=True
# kmeans.pyx
# Contact: Jacob Schreiber ( jmschreiber91@gmail.com )

from libc.stdlib cimport calloc
from libc.stdlib cimport free
from libc.string cimport memset
from libc.string cimport memcpy
from libc.math cimport log10 as clog10

from .base cimport Model

import json
import numpy
cimport numpy


DEF NEGINF = float("-inf")
DEF INF = float("inf")


def initialize_centroids(numpy.ndarray X, numpy.ndarray weights, int k, 
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

	return centroids

cdef class Kmeans(Model):
	"""A kmeans model.

	Kmeans is not a probabilistic model, but it is used in the kmeans++
	initialization for GMMs. In essence, a point is selected as the center for
	one component and then remaining points are selected.

	Parameters
	----------
	k : int
		The number of centroids.

	centroids : numpy.ndarray or None, optional
		The centroids to be used for this kmeans clusterer, if known ahead of
		time. These centroids can either be refined on future data, or used
		for predictions in the future. If None, then it will begin clustering
		on the first batch of data that it sees.
		Default is None.

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
	cdef double* centroids_ptr
	cdef double* summary_sizes
	cdef double* summary_weights

	def __init__(self, k, init='kmeans++', n_init=10):
		self.k = k
		self.d = 0
		self.n_init = n_init

		if isinstance(init, (list, numpy.ndarray)):
			self.centroids = numpy.array(init, dtype='float64', ndmin=2)
			self.centroids_ptr = <double*> self.centroids.data
		elif isinstance(init, str):
			self.init = init

	def __dealloc__(self):
		free(self.summary_sizes)
		free(self.summary_weights)

	def predict(self, X):
		"""Predict nearest centroid for each point.

		Parameters
		----------
		X : array-like, shape (n_samples, n_dim)
			The data to fit to.

		Returns
		-------
		y : array-like, shape (n_samples,)
			The index of the nearest centroid.
		"""

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
		    max_iterations=1e3, verbose=False):
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
			the new parameters.
			Default is 0.0.

		stop_threshold : double, optional, positive
			The threshold at which EM will terminate for the improvement of
			the model. If the model does not improve its fit of the data by a
			log probability of 0.1 then terminate.
			Default is 0.1.

		max_iterations : int, optional
			The maximum number of iterations to run for. Default is 1e3.

		verbose : bool, optional
			Whether or not to print out improvement information over
			iterations.
			Default is False.

		Returns
		-------
		None
		"""

		best_centroids, best_distance = None, INF

		for i in range(self.n_init):
			self.d = 0
			initial_distance_sum = INF
			iteration, improvement = 0, INF

			while improvement > stop_threshold and iteration < max_iterations + 1:
				self.from_summaries(inertia)
				distance_sum = self.summarize(X, weights)

				if iteration == 0:
					initial_distance_sum = distance_sum
				else:
					improvement = distance_sum - last_distance_sum

					if verbose:
						print("Improvement: {}".format(improvement))

				iteration += 1
				last_distance_sum = distance_sum

			self.clear_summaries()
			total_improvement = last_distance_sum - initial_distance_sum

			if verbose:
				print("Total Improvement: {}".format(total_improvement))

			if last_distance_sum < best_distance:
				best_centroids = self.centroids.copy()
				best_distance = last_distance_sum

		self.centroids = best_centroids
		self.centroids_ptr = <double*> self.centroids.data
		return best_distance

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
		cdef numpy.ndarray weights_ndarray
		cdef int i, j, n = X_ndarray.shape[0], d = X_ndarray.shape[1]
		cdef double* X_ptr = <double*> X_ndarray.data
		cdef double dist

		if weights is None:
			weights_ndarray = numpy.ones(n, dtype='float64')
		else:
			weights_ndarray = numpy.array(weights, dtype='float64')

		cdef double* weights_ptr = <double*> weights_ndarray.data

		if self.d == 0:
			self.d = d
			self.centroids = numpy.zeros((self.k, d))
			self.centroids_ptr = <double*> self.centroids.data

			self.centroids = initialize_centroids(X, weights_ndarray, self.k, self.init)
			self.centroids_ptr = <double*> self.centroids.data

			self.summary_sizes = <double*> calloc(self.k, sizeof(double))
			self.summary_weights = <double*> calloc(self.k*d, sizeof(double))
			memset(self.summary_sizes, 0 ,self.k*sizeof(double))
			memset(self.summary_weights, 0, self.k*d*sizeof(double))

		with nogil:
			dist = self._summarize(X_ptr, weights_ptr, n)

		return dist

	cdef double _summarize(self, double* X, double* weights, int n) nogil:
		cdef int i, j, l, y, k = self.k, d = self.d
		cdef double min_dist, dist, total_dist = 0.0
		cdef double* summary_sizes = <double*> calloc(k, sizeof(double))
		cdef double* summary_weights = <double*> calloc(k*d, sizeof(double))
		memset(summary_sizes, 0, k*sizeof(double))
		memset(summary_weights, 0, k*d*sizeof(double))

		for i in range(n):
			min_dist = INF

			for j in range(k):
				dist = 0.0

				for l in range(d):
					dist += (self.centroids_ptr[j*d + l] - X[i*d + l]) ** 2.0

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
		return total_dist

	def from_summaries(self, double inertia=0.0):
		if self.d == 0:
			return

		cdef int l, j, k = self.k, d = self.d

		with nogil:
			for j in range(k):
				for l in range(d):
					self.centroids_ptr[j*d + l] = \
						inertia * self.centroids_ptr[j*d + l] \
						+ (1-inertia) * self.summary_weights[j*d + l] \
						/ self.summary_sizes[j] \

			memset(self.summary_sizes, 0, self.k * sizeof(int))
			memset(self.summary_weights, 0, self.k * self.d * sizeof(int))

	def clear_summaries(self):
		memset(self.summary_sizes, 0, self.k*sizeof(int))
		memset(self.summary_weights, 0, self.k*self.d*sizeof(int))

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

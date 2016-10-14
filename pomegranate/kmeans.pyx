#cython: boundscheck=False
#cython: cdivision=True
# kmeans.pyx
# Contact: Jacob Schreiber ( jmschreiber91@gmail.com )

from libc.stdlib cimport calloc
from libc.stdlib cimport free
from libc.string cimport memset
from libc.string cimport memcpy

from .base cimport Model

import json
import numpy
cimport numpy


DEF NEGINF = float("-inf")
DEF INF = float("inf")


cdef class Kmeans(Model):
	"""A kmeans model.

	Kmeans is not a probabilistic model, but it is used in the kmeans++
	initialization for GMMs. In essence, a point is selected as the center
	for one component and then remaining points are selected

	Parameters
	----------
	k : int
		The number of centroids.

	centroids : numpy.ndarray or None, optional
		The centroids to be used for this kmeans clusterer, if known ahead of
		time. These centroids can either be refined on future data, or used
		for predictions in the future. If None, then it will begin clustering
		on the first batch of data that it sees. Default is None. 

	Attributes
	----------
	k : int
		The number of centroids

	centroids : array-like, shape (k, n_dim)
		The means of the centroid points.
	"""

	cdef public int k
	cdef public numpy.ndarray centroids
	cdef double* centroids_ptr
	cdef double* summary_sizes
	cdef double* summary_weights

	def __init__(self, k, centroids=None):
		self.k = k
		self.d = 0

		if centroids is not None:
			self.centroids = numpy.array(centroids, dtype='float64')
			self.centroids_ptr = <double*> self.centroids.data
			self.d = self.centroids.shape[1]

	def __str__(self):
		return self.to_json()

	def __repr__(self):
		return self.to_json()

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
					dist += ( self.centroids_ptr[j*d + l] - X[i*d + l] ) ** 2.0

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
			parameters will roughly be old_param*inertia + new_param*(1-inertia),
			so an inertia of 0 means ignore the old parameters, whereas an
			inertia of 1 means ignore the new parameters. Default is 0.0.

		stop_threshold : double, optional, positive
			The threshold at which EM will terminate for the improvement of
			the model. If the model does not improve its fit of the data by
			a log probability of 0.1 then terminate. Default is 0.1.

		max_iterations : int, optional
			The maximum number of iterations to run for. Default is 1e3.

		verbose : bool, optional
			Whether or not to print out improvement information over iterations.
			Default is False.

		Returns
		-------
		None
		"""

		initial_log_probability_sum = NEGINF
		iteration, improvement = 0, INF

		while improvement > stop_threshold and iteration < max_iterations + 1:
			self.from_summaries(inertia)
			log_probability_sum = self.summarize(X, weights)

			if iteration == 0:
				initial_log_probability_sum = log_probability_sum
			else:
				improvement = log_probability_sum - last_log_probability_sum

				if verbose:
					print("Improvement: {}".format(improvement))

			iteration += 1
			last_log_probability_sum = log_probability_sum

		self.clear_summaries()

		if verbose:
			print("Total Improvement: {}".format(
				last_log_probability_sum - initial_log_probability_sum))

		return last_log_probability_sum - initial_log_probability_sum

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
			self.summary_sizes = <double*> calloc(self.k, sizeof(double))
			self.summary_weights = <double*> calloc(self.k*d, sizeof(double))

			memcpy(self.centroids_ptr, X_ptr, self.k*d*sizeof(double))
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
					dist += ( self.centroids_ptr[j*d + l] - X[i*d + l] ) ** 2.0

				if dist < min_dist:
					min_dist = dist
					y = j

			total_dist -= min_dist
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
		"""Fit the model to the sufficient statistics.

		Parameters
		----------
		inertia : double, optional
			The weight of the previous parameters of the model. The new
			parameters will roughly be old_param*inertia
			+ new_param * (1-inertia), so an inertia of 0 means ignore the old
			parameters, whereas an inertia of 1 means ignore the new
			parameters. Default is 0.0.

		Returns
		-------
		None
		"""

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
		"""Clear the stored sufficient statistics.

		Parameters
		----------
		None

		Returns
		-------
		None
		"""

		memset(self.summary_sizes, 0, self.k*sizeof(int))
		memset(self.summary_weights, 0, self.k*self.d*sizeof(int))

	def to_json(self, separators=(',', ' : '), indent=4):
		"""Serialize the model to a JSON.

		Parameters
		----------
		separators : tuple, optional
			The two separaters to pass to the json.dumps function for formatting.
			Default is (',', ' : ').

		indent : int, optional
			The indentation to use at each level. Passed to json.dumps for
			formatting. Default is 4.

		Returns
		-------
		json : str
			A properly formatted JSON object.
		"""

		model = {
					'class' : 'Kmeans',
					'k' : self.k,
					'centroids'  : self.centroids.tolist()
				}

		return json.dumps(model, separators=separators, indent=indent)

	@classmethod
	def from_json(cls, s):
		"""Read in a serialized model and return the appropriate classifier.

		Parameters
		----------
		s : str
			A JSON formatted string containing the file.
		Returns
		-------
		model : object
			A properly initialized and baked model.
		"""

		d = json.loads(s)
		model = Kmeans(d['k'], d['centroids'])
		return model

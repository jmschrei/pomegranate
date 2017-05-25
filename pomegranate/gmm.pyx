#cython: boundscheck=False
#cython: cdivision=True
# gmm.pyx
# Contact: Jacob Schreiber ( jmschreiber91@gmail.com )

from libc.stdlib cimport calloc
from libc.stdlib cimport free
from libc.string cimport memset
from libc.math cimport exp as cexp

import json

import numpy
cimport numpy

from .base cimport Model
from .kmeans import Kmeans
from .distributions cimport Distribution
from .distributions import DiscreteDistribution, IndependentComponentsDistribution
from .bayes cimport BayesModel
from .utils cimport _log
from .utils cimport pair_lse
from .utils import _check_input

DEF NEGINF = float("-inf")
DEF INF = float("inf")

cdef class GeneralMixtureModel(BayesModel):
	"""A General Mixture Model.

	This mixture model can be a mixture of any distribution as long as they
	are all of the same dimensionality. Any object can serve as a distribution
	as long as it has fit(X, weights), log_probability(X), and summarize(X,
	weights)/from_summaries() methods if out of core training is desired.

	Parameters
	----------
	distributions : array-like, shape (n_components,) or callable
		The components of the model. If array, corresponds to the initial
		distributions of the components. If callable, must also pass in the
		number of components and kmeans++ will be used to initialize them.

	weights : array-like, optional, shape (n_components,)
		The prior probabilities corresponding to each component. Does not
		need to sum to one, but will be normalized to sum to one internally.
		Defaults to None.

	Attributes
	----------
	distributions : array-like, shape (n_components,)
		The component distribution objects.

	weights : array-like, shape (n_components,)
		The learned prior weight of each object

	Examples
	--------
	>>> from pomegranate import *
	>>> clf = GeneralMixtureModel([
	>>>     NormalDistribution(5, 2),
	>>>     NormalDistribution(1, 1)])
	>>> clf.log_probability(5)
	-2.304562194038089
	>>> clf.predict_proba([[5], [7], [1]])
	array([[ 0.99932952,  0.00067048],
	       [ 0.99999995,  0.00000005],
	       [ 0.06337894,  0.93662106]])
	>>> clf.fit([[1], [5], [7], [8], [2]])
	>>> clf.predict_proba([[5], [7], [1]])
	array([[ 1.        ,  0.        ],
	       [ 1.        ,  0.        ],
	       [ 0.00004383,  0.99995617]])
	>>> clf.distributions
	array([ {
	    "frozen" :false,
	    "class" :"Distribution",
	    "parameters" :[
	        6.6571359101390755,
	        1.2639830514274502
	    ],
	    "name" :"NormalDistribution"
	},
	       {
	    "frozen" :false,
	    "class" :"Distribution",
	    "parameters" :[
	        1.498707696758334,
	        0.4999983303277837
	    ],
	    "name" :"NormalDistribution"
	}], dtype=object)
	"""

	def __init__(self, distributions, weights=None):
		super(GeneralMixtureModel, self).__init__(distributions, weights)

	def __reduce__(self):
		return self.__class__, (self.distributions.tolist(), numpy.exp(self.weights))

	def fit(self, X, weights=None, inertia=0.0, stop_threshold=0.1,
		max_iterations=1e8, pseudocount=0.0, verbose=False):
		"""Fit the model to new data using EM.

		This method fits the components of the model to new data using the EM
		method. It will iterate until either max iterations has been reached,
		or the stop threshold has been passed.

		This is a sklearn wrapper for train method.

		Parameters
		----------
		X : array-like, shape (n_samples, n_dimensions)
			This is the data to train on. Each row is a sample, and each column
			is a dimension to train on.

		weights : array-like, shape (n_samples,), optional
			The initial weights of each sample in the matrix. If nothing is
			passed in then each sample is assumed to be the same weight.
			Default is None.

		inertia : double, optional
			The weight of the previous parameters of the model. The new
			parameters will roughly be
			old_param*inertia + new_param*(1-inertia),
			so an inertia of 0 means ignore the old parameters, whereas an
			inertia of 1 means ignore the new parameters.
			Default is 0.0.

		stop_threshold : double, optional, positive
			The threshold at which EM will terminate for the improvement of
			the model. If the model does not improve its fit of the data by
			a log probability of 0.1 then terminate.
			Default is 0.1.

		max_iterations : int, optional, positive
			The maximum number of iterations to run EM for. If this limit is
			hit then it will terminate training, regardless of how well the
			model is improving per iteration.
			Default is 1e8.

		pseudocount : double, optional, positive
            A pseudocount to add to the emission of each distribution. This
            effectively smoothes the states to prevent 0. probability symbols
            if they don't happen to occur in the data. Only effects mixture
            models defined over discrete distributions. Default is 0.

		verbose : bool, optional
			Whether or not to print out improvement information over
			iterations.
			Default is False.

		Returns
		-------
		improvement : double
			The total improvement in log probability P(D|M)
		"""

		initial_log_probability_sum = NEGINF
		iteration, improvement = 0, INF

		while improvement > stop_threshold and iteration < max_iterations + 1:
			self.from_summaries(inertia, pseudocount)
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
		"""Summarize a batch of data and store sufficient statistics.

		This will run the expectation step of EM and store sufficient
		statistics in the appropriate distribution objects. The summarization
		can be thought of as a chunk of the E step, and the from_summaries
		method as the M step.

		Parameters
		----------
		X : array-like, shape (n_samples, n_dimensions)
			This is the data to train on. Each row is a sample, and each column
			is a dimension to train on.

		weights : array-like, shape (n_samples,), optional
			The initial weights of each sample in the matrix. If nothing is
			passed in then each sample is assumed to be the same weight.
			Default is None.

		Returns
		-------
		logp : double
			The log probability of the data given the current model. This is
			used to speed up EM.
		"""

		cdef int i, n, d
		cdef numpy.ndarray X_ndarray
		cdef numpy.ndarray weights_ndarray
		cdef double log_probability

		if self.is_vl_:
			n, d = len(X), self.d
		elif self.d == 1:
			n, d = X.shape[0], 1
		elif self.d > 1 and X.ndim == 1:
			n, d = 1, len(X)
		else:
			n, d = X.shape

		if weights is None:
			weights_ndarray = numpy.ones(n, dtype='float64')
		else:
			weights_ndarray = numpy.array(weights, dtype='float64')

		cdef double* X_ptr
		cdef double* weights_ptr = <double*> weights_ndarray.data

		if not self.is_vl_:
			X_ndarray = _check_input(X, self.keymap)
			X_ptr = <double*> X_ndarray.data
			
			with nogil:
				log_probability = self._summarize(X_ptr, weights_ptr, n)
		else:
			log_probability = 0.0
			for i in range(n):
				X_ndarray = _check_input(X[i], self.keymap)
				X_ptr = <double*> X_ndarray.data
				d = len(X_ndarray)
				with nogil:
					log_probability += self._summarize(X_ptr, weights_ptr+i, d)

		return log_probability

	cdef double _summarize(self, double* X, double* weights, int n) nogil:
		cdef double* r = <double*> calloc(self.n*n, sizeof(double))
		cdef double* summaries = <double*> calloc(self.n, sizeof(double))
		cdef int i, j
		cdef double total, logp, log_probability_sum = 0.0

		memset(summaries, 0, self.n*sizeof(double))
		cdef double tic

		for j in range(self.n):
			if self.is_vl_:
				r[j*n] = (<Model> self.distributions_ptr[j])._vl_log_probability(X, n)
			else:
				(<Model> self.distributions_ptr[j])._log_probability(X, r+j*n, n)

		for i in range(n):
			total = NEGINF

			for j in range(self.n):
				r[j*n + i] += self.weights_ptr[j]
				total = pair_lse(total, r[j*n + i])

			for j in range(self.n):
				r[j*n + i] = cexp(r[j*n + i] - total) * weights[i]
				summaries[j] += r[j*n + i]
			
			log_probability_sum += total * weights[i]

			if self.is_vl_:
				break

		for j in range(self.n):
			(<Model> self.distributions_ptr[j])._summarize(X, r+j*n, n)

		with gil:
			for j in range(self.n):
				self.summaries_ptr[j] += summaries[j]

		free(r)
		free(summaries)
		return log_probability_sum

	def to_json(self):
		separators=(',', ' : ')
		indent=4

		model = {
					'class' : 'GeneralMixtureModel',
					'distributions'  : [ json.loads(dist.to_json())
					                     for dist in self.distributions ],
					'weights' : self.weights.tolist()
				}

		return json.dumps(model, separators=separators, indent=indent)

	@classmethod
	def from_json(cls, s):
		d = json.loads(s)
		distributions = [ Distribution.from_json(json.dumps(j))
		                  for j in d['distributions'] ]
		model = GeneralMixtureModel(distributions, numpy.array( d['weights'] ))
		return model

	@classmethod
	def from_samples(self, distributions, n_components, X, weights=None, 
		n_init=1, init='kmeans++', max_kmeans_iterations=1, inertia=0.0, 
		stop_threshold=0.1, max_iterations=1e8, pseudocount=0.0, verbose=False):
		"""Create a mixture model directly from the given dataset.

		First, k-means will be run using the given initializations, in order to
		define initial clusters for the points. These clusters are used to
		initialize the distributions used. Then, EM is run to refine the
		parameters of these distributions.

		A homogenous mixture can be defined by passing in a single distribution
		callable as the first parameter and specifying the number of components,
		while a heterogeneous mixture can be defined by passing in a list of
		callables of the appropriate type.

		Parameters
		----------
		distributions : array-like, shape (n_components,) or callable
			The components of the model. If array, corresponds to the initial
			distributions of the components. If callable, must also pass in the
			number of components and kmeans++ will be used to initialize them.

		n_components : int
			If a callable is passed into distributions then this is the number
			of components to initialize using the kmeans++ algorithm.

		X : array-like, shape (n_samples, n_dimensions)
			This is the data to train on. Each row is a sample, and each column
			is a dimension to train on.

		weights : array-like, shape (n_samples,), optional
			The initial weights of each sample in the matrix. If nothing is
			passed in then each sample is assumed to be the same weight.
			Default is None.

		n_init : int, optional
			The number of initializations of k-means to do before choosing
			the best. Default is 1.

		init : str, optional
			The initialization algorithm to use for the initial k-means
			clustering. Must be one of 'first-k', 'random', 'kmeans++',
			or 'kmeans||'. Default is 'kmeans++'.

		max_kmeans_iterations : int, optional
			The maximum number of iterations to run kmeans for in the
			initialization step. Default is 1.

		inertia : double, optional
			The weight of the previous parameters of the model. The new
			parameters will roughly be
			old_param*inertia + new_param*(1-inertia),
			so an inertia of 0 means ignore the old parameters, whereas an
			inertia of 1 means ignore the new parameters.
			Default is 0.0.

		stop_threshold : double, optional, positive
			The threshold at which EM will terminate for the improvement of
			the model. If the model does not improve its fit of the data by
			a log probability of 0.1 then terminate.
			Default is 0.1.

		max_iterations : int, optional, positive
			The maximum number of iterations to run EM for. If this limit is
			hit then it will terminate training, regardless of how well the
			model is improving per iteration.
			Default is 1e8.

		pseudocount : double, optional, positive
            A pseudocount to add to the emission of each distribution. This
            effectively smoothes the states to prevent 0. probability symbols
            if they don't happen to occur in the data. Only effects mixture
            models defined over discrete distributions. Default is 0.

		verbose : bool, optional
			Whether or not to print out improvement information over
			iterations.
			Default is False.
		"""

		if not callable(distributions) and not isinstance(distributions, list):
			raise ValueError("must either give initial distributions "
			                 "or constructor")

		if callable(distributions):
			if distributions == DiscreteDistribution:
				raise ValueError("cannot fit a discrete GMM "
				                 "without pre-initialized distributions")
			
			distributions = [distributions for i in range(n_components)]

		else:
			n_components = len(distributions)
			if n_components < 2:
				raise ValueError("must have at least two distributions "
				                 "for general mixture models")

			for dist in distributions:
				if not callable(dist):
					raise ValueError("must pass in uninitialized distributions")

		X = numpy.array(X)
		n, d = X.shape

		kmeans = Kmeans(n_components, init=init, n_init=n_init)
		kmeans.fit(X, weights=weights, max_iterations=max_kmeans_iterations)
		y = kmeans.predict(X)

		distributions = [distribution.from_samples(X[y == i]) for i, distribution in enumerate(distributions)]
		class_weights = numpy.array([(y == i).mean() for i in range(n_components)])

		model = GeneralMixtureModel(distributions, class_weights)
		model.fit(X, weights, inertia=inertia, stop_threshold=stop_threshold, 
			max_iterations=max_iterations, pseudocount=pseudocount, 
			verbose=verbose)

		return model
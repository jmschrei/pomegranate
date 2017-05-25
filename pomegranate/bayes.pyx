#cython: boundscheck=False
#cython: cdivision=True
# bayes.pyx
# Contact: Jacob Schreiber ( jmschreiber91@gmail.com )

from libc.stdlib cimport calloc
from libc.stdlib cimport free
from libc.string cimport memset
from libc.math cimport exp as cexp

import json

import numpy
cimport numpy

from .base cimport Model
from .distributions cimport Distribution
from .distributions import DiscreteDistribution
from .distributions import IndependentComponentsDistribution

from .utils cimport _log
from .utils cimport pair_lse
from .utils import _check_input
from .utils import _convert

DEF NEGINF = float("-inf")

cdef class BayesModel(Model):
	"""A simple implementation of Bayes Rule as a base model.

	Bayes rule is foundational to many models. Here, it is used as a base
	class for naive Bayes, Bayes classifiers, and mixture models, that are
	all fundamentally similar and rely on bayes rule.

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

	d : int
		The number of dimensionals the model is built to consider.

	is_vl_ : bool
		Whether this model is built for variable length sequences or not.
	"""

	def __init__(self, distributions, weights=None):
		self.d = 0
		self.is_vl_ = 0

		self.n = len(distributions)
		if len(distributions) < 2:
			raise ValueError("must pass in at least two distributions")

		self.d = distributions[0].d
		for dist in distributions:
			if callable(dist):
				raise TypeError("must have initialized distributions in list")
			elif self.d != dist.d:
				raise TypeError("mis-matching dimensions between distributions in list")
			if dist.model == 'HiddenMarkovModel':
				self.is_vl_ = 1
				self.keymap = dist.keymap

		if weights is None:
			weights = numpy.ones_like(distributions, dtype='float64') / self.n
		else:
			weights = numpy.array(weights, dtype='float64') / weights.sum()

		self.weights = numpy.log(weights)
		self.weights_ptr = <double*> self.weights.data

		self.distributions = numpy.array(distributions)
		self.distributions_ptr = <void**> self.distributions.data

		self.summaries = numpy.zeros_like(weights, dtype='float64')
		self.summaries_ptr = <double*> self.summaries.data

		dist = distributions[0]
		if self.is_vl_ == 1:
			pass
		elif isinstance(dist, DiscreteDistribution):
			keys = []
			for distribution in distributions:
				keys.extend(distribution.keys())
			self.keymap = [{key: i for i, key in enumerate(set(keys))}]
			for distribution in distributions:
				distribution.bake(tuple(set(keys)))

		elif isinstance(dist, IndependentComponentsDistribution) and dist.discrete:
			self.keymap = []
			for i in range(self.d):
				if isinstance(distributions[i], DiscreteDistribution):
					keys = dist.distributions[i].keys()
					self.keymap.append({ key: i for i, key in enumerate(set(keys)) })
				else:
					self.keymap.append(None)

			for distribution in distributions:
				for i in range(self.d):
					d = distribution.distributions[i]
					if isinstance(d, DiscreteDistribution):
						d.bake( tuple(set(self.keymap[i].keys())) )

	def __reduce__(self):
		return self.__class__, (self.distributions.tolist(),
		                        numpy.exp(self.weights),
		                        self.n)

	def sample(self, n=1):
		"""Generate a sample from the model.

		First, randomly select a component weighted by the prior probability,
		Then, use the sample method from that component to generate a sample.

		Parameters
		----------
		n : int, optional
			The number of samples to generate. Defaults to 1.

		Returns
		-------
		sample : array-like or object
			A randomly generated sample from the model of the type modelled
			by the emissions. An integer if using most distributions, or an
			array if using multivariate ones, or a string for most discrete
			distributions. If n=1 return an object, if n>1 return an array
			of the samples.
		"""

		samples = []
		for i in range(n):
			d = numpy.random.choice(self.distributions,
			                        p=numpy.exp(self.weights))
			samples.append(d.sample())

		return samples if n > 1 else samples[0]

	def log_probability(self, X):
		"""Calculate the log probability of a point under the distribution.

		The probability of a point is the sum of the probabilities of each
		distribution multiplied by the weights. Thus, the log probability is
		the sum of the log probability plus the log prior.

		This is the python interface.

		Parameters
		----------
		X : numpy.ndarray, shape=(n, d) or (n, m, d)
			The samples to calculate the log probability of. Each row is a
			sample and each column is a dimension. If emissions are HMMs then
			shape is (n, m, d) where m is variable length for each obervation,
			and X becomes an array of n (m, d)-shaped arrays.

		Returns
		-------
		log_probability : double
			The log probabiltiy of the point under the distribution.
		"""

		cdef int i, j, n, d, m

		if self.is_vl_ or self.d == 1:
			n, d = len(X), self.d
		elif self.d > 1 and X.ndim == 1:
			n, d = 1, len(X)
		else:
			n, d = X.shape

		cdef numpy.ndarray logp_ndarray = numpy.zeros(n)
		cdef double* logp = <double*> logp_ndarray.data

		cdef numpy.ndarray X_ndarray
		cdef double* X_ptr 

		if not self.is_vl_:
			X_ndarray = _check_input(X, self.keymap)
			X_ptr = <double*> X_ndarray.data
			if d != self.d:
				raise ValueError("sample has {} dimensions but model has {} dimensions".format(d, self.d))

		with nogil:
			if self.is_vl_:
				for i in range(n):
					with gil:
						X_ndarray = numpy.array(X[i])
						X_ptr = <double*> X_ndarray.data
					logp[i] = self._vl_log_probability(X_ptr, n)
			else:
				self._log_probability(X_ptr, logp, n)

		return logp_ndarray

	cdef void _log_probability(self, double* X, double* log_probability, int n) nogil:
		cdef int i, j, d = self.d
		cdef double* logp = <double*> calloc(n, sizeof(double))

		(<Model> self.distributions_ptr[0])._log_probability(X, log_probability, n)
		for i in range(n):
			log_probability[i] += self.weights_ptr[0]

		for j in range(1, self.n):
			(<Model> self.distributions_ptr[j])._log_probability(X, logp, n)
			for i in range(n):
				log_probability[i] = pair_lse(log_probability[i], logp[i] + self.weights_ptr[j])

	cdef double _vl_log_probability(self, double* X, int n) nogil:
		cdef int i
		cdef double log_probability_sum = NEGINF
		cdef double log_probability

		for i in range(self.n):
			log_probability = (<Model> self.distributions_ptr[i])._vl_log_probability(X, n) + self.weights_ptr[i]
			log_probability_sum = pair_lse(log_probability_sum, log_probability)

		return log_probability_sum

	def predict_proba(self, X):
		"""Calculate the posterior P(M|D) for data.

		Calculate the probability of each item having been generated from
		each component in the model. This returns normalized probabilities
		such that each row should sum to 1.

		Since calculating the log probability is much faster, this is just
		a wrapper which exponentiates the log probability matrix.

		Parameters
		----------
		X : array-like, shape (n_samples, n_dimensions)
			The samples to do the prediction on. Each sample is a row and each
			column corresponds to a dimension in that sample. For univariate
			distributions, a single array may be passed in.

		Returns
		-------
		probability : array-like, shape (n_samples, n_components)
			The normalized probability P(M|D) for each sample. This is the
			probability that the sample was generated from each component.
		"""

		return numpy.exp(self.predict_log_proba(X))

	def predict_log_proba(self, X):
		"""Calculate the posterior log P(M|D) for data.

		Calculate the log probability of each item having been generated from
		each component in the model. This returns normalized log probabilities
		such that the probabilities should sum to 1

		This is a sklearn wrapper for the original posterior function.

		Parameters
		----------
		X : array-like, shape (n_samples, n_dimensions)
			The samples to do the prediction on. Each sample is a row and each
			column corresponds to a dimension in that sample. For univariate
			distributions, a single array may be passed in.

		Returns
		-------
		y : array-like, shape (n_samples, n_components)
			The normalized log probability log P(M|D) for each sample. This is
			the probability that the sample was generated from each component.
		"""

		cdef int i, n, d
		cdef numpy.ndarray X_ndarray
		cdef double* X_ptr

		cdef numpy.ndarray y
		cdef double* y_ptr

		if not self.is_vl_:
			X_ndarray = _check_input(X, self.keymap)
			X_ptr = <double*> X_ndarray.data
			n, d = X_ndarray.shape[0], X_ndarray.shape[1]
			if d != self.d:
				raise ValueError("sample only has {} dimensions but should have {} dimensions".format(d, self.d))
		else:
			X_ndarray = X
			n, d = len(X_ndarray), self.d

		y = numpy.zeros((n, self.n), dtype='float64')
		y_ptr = <double*> y.data

		if not self.is_vl_:
			X_ndarray = _check_input(X, self.keymap)
			X_ptr = <double*> X_ndarray.data
			if d != self.d:
				raise ValueError("sample only has {} dimensions but should have {} dimensions".format(d, self.d))

		with nogil:
			if not self.is_vl_:
				self._predict_log_proba(X_ptr, y_ptr, n, d)
			else:
				for i in range(n):
					with gil:
						X_ndarray = _check_input(X[i], self.keymap)
						X_ptr = <double*> X_ndarray.data
						d = len(X_ndarray)

					self._predict_log_proba(X_ptr, y_ptr+i*self.n, 1, d)

		return y if self.is_vl_ else y.reshape(self.n, n).T

	cdef void _predict_log_proba(self, double* X, double* y, int n, int d) nogil:
		cdef double y_sum, logp
		cdef int i, j

		for j in range(self.n):
			if self.is_vl_:
				y[j] = (<Model> self.distributions_ptr[j])._vl_log_probability(X, d)
			else:
				(<Model> self.distributions_ptr[j])._log_probability(X, y+j*n, n)

		for i in range(n):
			y_sum = NEGINF

			for j in range(self.n):
				y[j*n + i] += self.weights_ptr[j]
				y_sum = pair_lse(y_sum, y[j*n + i])

			for j in range(self.n):
				y[j*n + i] -= y_sum

	def predict(self, X):
		"""Predict the most likely component which generated each sample.

		Calculate the posterior P(M|D) for each sample and return the index
		of the component most likely to fit it. This corresponds to a simple
		argmax over the responsibility matrix.

		This is a sklearn wrapper for the maximum_a_posteriori method.

		Parameters
		----------
		X : array-like, shape (n_samples, n_dimensions)
			The samples to do the prediction on. Each sample is a row and each
			column corresponds to a dimension in that sample. For univariate
			distributions, a single array may be passed in.

		Returns
		-------
		y : array-like, shape (n_samples,)
			The predicted component which fits the sample the best.
		"""

		cdef int i, n, d
		cdef numpy.ndarray X_ndarray
		cdef double* X_ptr

		cdef numpy.ndarray y
		cdef int* y_ptr

		if not self.is_vl_:
			X_ndarray = _check_input(X, self.keymap)
			X_ptr = <double*> X_ndarray.data
			n, d = len(X_ndarray), len(X_ndarray[0])
			if d != self.d:
				raise ValueError("sample only has {} dimensions but should have {} dimensions".format(d, self.d))
		else:
			X_ndarray = X
			n, d = len(X_ndarray), self.d


		y = numpy.zeros(n, dtype='int32')
		y_ptr = <int*> y.data

		with nogil:
			if not self.is_vl_:
				self._predict(X_ptr, y_ptr, n, d)
			else:
				for i in range(n):
					with gil:
						X_ndarray = _check_input(X[i], self.keymap)
						X_ptr = <double*> X_ndarray.data
						d = len(X_ndarray)

					self._predict(X_ptr, y_ptr+i, 1, d)

		return y

	cdef void _predict( self, double* X, int* y, int n, int d) nogil:
		cdef int i, j
		cdef double max_logp, logp
		cdef double* r = <double*> calloc(n*self.n, sizeof(double))

		for j in range(self.n):
			if self.is_vl_:
				r[j] = (<Model> self.distributions_ptr[j])._vl_log_probability(X, d)
			else:
				(<Model> self.distributions_ptr[j])._log_probability(X, r+j*n, n)

		for i in range(n):
			max_logp = NEGINF

			for j in range(self.n):
				logp = r[j*n + i] + self.weights_ptr[j]
				if logp > max_logp:
					max_logp = logp
					y[i] = j

		free(r)

	def fit(self, X, weights=None, inertia=0.0, pseudocount=0.0):
		"""Fit the model to some data. Implemented in subclasses.

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

		pseudocount : double, optional, positive
            A pseudocount to add to the emission of each distribution. This
            effectively smoothes the states to prevent 0. probability symbols
            if they don't happen to occur in the data. Only effects mixture
            models defined over discrete distributions. Default is 0.

		Returns
		-------
		improvement : double
			The total improvement in log probability P(D|M)
		"""

		raise NotImplementedError

	def summarize(self, X, weights=None):
		"""Summarize a batch of data and store sufficient statistics.

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

		raise NotImplementedError

	cdef double _summarize(self, double* X, double* weights, int n) nogil:
		return -1

	def from_summaries(self, inertia=0.0, pseudocount=0.0, **kwargs):
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

		pseudocount : double, optional
			A pseudocount to add to the emission of each distribution. This
			effectively smoothes the states to prevent 0. probability symbols
			if they don't happen to occur in the data. If discrete data, will
			smooth both the prior probabilities of each component and the
			emissions of each component. Otherwise, will only smooth the prior
			probabilities of each component. Default is 0.

		Returns
		-------
		None
		"""

		if self.summaries.sum() == 0:
			return

		summaries = self.summaries + pseudocount
		summaries /= summaries.sum()

		for i, distribution in enumerate(self.distributions):
			if isinstance(distribution, DiscreteDistribution):
				distribution.from_summaries(inertia, pseudocount)
			else:
				distribution.from_summaries(inertia, **kwargs)
			
			self.weights[i] = _log(summaries[i])
			self.summaries[i] = 0.
		
		return self

	def clear_summaries(self):
		"""Remove the stored sufficient statistics.

		Parameters
		----------
		None

		Returns
		-------
		None
		"""

		self.summaries *= 0
		for distribution in self.distributions:
			distribution.clear_summaries()

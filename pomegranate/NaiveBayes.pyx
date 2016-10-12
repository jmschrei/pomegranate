# NaiveBayes.pyx
# Authors: Jacob Schreiber <jmschreiber91@gmail.com>

"""
Naive Bayes estimator, for anything with a log_probability method.
"""

import numpy
cimport numpy

from libc.math cimport exp as cexp

from .distributions cimport Distribution
from .distributions import DiscreteDistribution
from .gmm import GeneralMixtureModel
from .hmm import HiddenMarkovModel
from .BayesianNetwork import BayesianNetwork

from .base cimport Model
from .base cimport GraphModel

from libc.stdlib cimport calloc
from libc.stdlib cimport free

from .utils cimport pair_lse
from .utils import _convert
import json

import sys

from joblib import Parallel
from joblib import delayed

cpdef numpy.ndarray _check_input(X, dict keymap):
	"""Check the input to make sure that it is a properly formatted array."""

	cdef numpy.ndarray X_ndarray

	if isinstance(X, numpy.ndarray) and (X.dtype == 'float64'):
		return X
	elif isinstance(X, (int, float)):
		X_ndarray = numpy.array([X], dtype='float64')
	elif not isinstance(X, (numpy.ndarray, list, tuple)):
		X_ndarray = numpy.array([keymap[X]], dtype='float64')
	else:
		try:
			X_ndarray = numpy.array(X, dtype='float64')
		except ValueError:
			X = numpy.array(X)
			X_ndarray = numpy.empty(X.shape, dtype='float64')

			if X.ndim == 1:
				for i in range(X.shape[0]):
					X_ndarray[i] = keymap[X[i]]
			else:
				for i in range(X.shape[0]):
					for j in range(X.shape[1]):
						X_ndarray[i, j] = keymap[X[i][j]]

	return X_ndarray

cdef double NEGINF = float("-inf")

cdef class NaiveBayes( Model ):
	"""A Naive Bayes model, a supervised alternative to GMM.

	Parameters
	----------
	models : list or constructor
		Must either be a list of initialized distribution/model objects, or
		the constructor for a distribution object:

		* Initialized : NaiveBayes([NormalDistribution(1, 2), NormalDistribution(0, 1)])
		* Constructor : NaiveBayes(NormalDistribution)

	weights : list or numpy.ndarray or None, default None
		The prior probabilities of the components. If None is passed in then
		defaults to the uniformly distributed priors.

	Attributes
	----------
	models : list
		The model objects, either initialized by the user or fit to data.

	weights : numpy.ndarray
		The prior probability of each component of the model.

	Examples
	--------
	>>> from pomegranate import *
	>>> clf = NaiveBayes( NormalDistribution )
	>>> X = [0, 2, 0, 1, 0, 5, 6, 5, 7, 6]
	>>> y = [0, 0, 0, 0, 0, 1, 1, 0, 1, 1]
	>>> clf.fit(X, y)
	>>> clf.predict_proba([6])
	array([[ 0.01973451,  0.98026549]])

	>>> from pomegranate import *
	>>> clf = NaiveBayes([NormalDistribution(1, 2), NormalDistribution(0, 1)])
	>>> clf.predict_log_proba([[0], [1], [2], [-1]])
	array([[-1.1836569 , -0.36550972],
		   [-0.79437677, -0.60122959],
		   [-0.26751248, -1.4493653 ],
		   [-1.09861229, -0.40546511]])
	"""

	cdef object distribution_callable
	cdef public numpy.ndarray distributions
	cdef void** distributions_ptr
	cdef numpy.ndarray summaries
	cdef double* summaries_ptr
	cdef public numpy.ndarray weights
	cdef double* weights_ptr
	cdef public int n
	cdef int hmm
	cdef dict keymap

	def __init__( self, distributions=None, weights=None ):
		if not callable(distributions) and not isinstance(distributions, (list, numpy.ndarray)):
			raise ValueError("must either give initial distributions or constructor")

		self.d = 0
		self.hmm = 0

		if callable(distributions):
			self.distribution_callable = distributions
		else:
			self.n = len(distributions)
			if len(distributions) < 2:
				raise ValueError("must have at least two distributions for general mixture models")

			for dist in distributions:
				if callable(dist):
					raise TypeError("must have initialized distributions in list")
				elif self.d == 0:
					self.d = dist.d
				elif self.d != dist.d:
					raise TypeError("mis-matching dimensions between distributions in list")
				if dist.model == 'HiddenMarkovModel':
					self.hmm = 1
					self.keymap = dist.keymap

			if weights is None:
				weights = numpy.ones_like(distributions, dtype=float) / len( distributions )
			else:
				weights = numpy.asarray(weights) / weights.sum()

			self.weights = numpy.log( weights )
			self.weights_ptr = <double*> self.weights.data

			self.distributions = numpy.array( distributions )
			self.distributions_ptr = <void**> self.distributions.data

			self.summaries = numpy.zeros_like(weights, dtype='float64')
			self.summaries_ptr = <double*> self.summaries.data

		if self.d > 0 and isinstance( self.distributions[0], DiscreteDistribution ):
			keys = []
			for d in self.distributions:
				keys.extend( d.keys() )
			self.keymap = { key: i for i, key in enumerate(set(keys)) }
			for d in self.distributions:
				d.bake( tuple(set(keys)) )

	def __reduce__( self ):
		return self.__class__, (self.distributions, self.weights)

	def __str__( self ):
		try:
			return self.to_json()
		except:
			return self.__repr__()

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

		if self.d == 0:
			raise ValueError("must first fit model before using predict proba method.")

		return numpy.exp( self.predict_log_proba(X) )

	def predict_log_proba( self, X ):
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

		if self.d == 0:
			raise ValueError("must first fit model before using predict log proba method.")

		cdef int i, n, d
		cdef numpy.ndarray X_ndarray
		cdef double* X_ptr
		cdef numpy.ndarray y
		cdef double* y_ptr

		if self.hmm == 0:
			X_ndarray = _check_input(X, self.keymap)
			X_ptr = <double*> X_ndarray.data

			if not (self.d == 1 and X_ndarray.ndim == 1) and X_ndarray.shape[1] != self.d:
				raise ValueError("dimensionality of model does not match data")

		if self.hmm == 1:
			n, d = len(X), self.d
		elif self.d > 1 and X_ndarray.ndim == 1:
			n, d = 1, X_ndarray.shape[0]
		else: 
			n, d = len(X_ndarray), self.d

		y = numpy.zeros((n, self.n), dtype='float64')
		y_ptr = <double*> y.data

		with nogil:
			if self.hmm == 0:
				self._predict_log_proba( X_ptr, y_ptr, n, d )
			else:
				for i in range(n):
					with gil:
						X_ndarray = _check_input(X[i], self.keymap)
						X_ptr = <double*> X_ndarray.data
						d = len(X_ndarray)

					self._predict_log_proba( X_ptr, y_ptr+i*self.n, 1, d )

		return y if self.hmm == 1 else y.reshape(self.n, n).T

	cdef void _predict_log_proba( self, double* X, double* y, int n, int d ) nogil:
		cdef double y_sum, logp
		cdef int i, j

		for j in range(self.n):
			if self.hmm == 1:
				y[j] = (<Model> self.distributions_ptr[j])._vl_log_probability(X, d)
			else:
				(<Model> self.distributions_ptr[j])._v_log_probability(X, y+j*n, n)

		for i in range(n):
			y_sum = NEGINF

			for j in range(self.n):
				y[j*n + i] += self.weights_ptr[j]
				y_sum = pair_lse(y_sum, y[j*n + i])

			for j in range(self.n):
				y[j*n + i] -= y_sum

	def predict( self, X ):
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

		if self.d == 0:
			raise ValueError("must first fit model before using predict method.")

		if self.hmm == 0:
			X_ndarray = _check_input(X, self.keymap)
			X_ptr = <double*> X_ndarray.data

			if not (self.d == 1 and X_ndarray.ndim == 1) and X_ndarray.shape[1] != self.d:
				raise ValueError("dimensionality of model does not match data")

		if self.hmm == 1:
			n, d = len(X), self.d
		elif self.d > 1 and X_ndarray.ndim == 1:
			n, d = 1, X_ndarray.shape[0]
		else: 
			n, d = len(X_ndarray), self.d

		cdef numpy.ndarray y = numpy.zeros(n, dtype='int32')
		cdef int* y_ptr = <int*> y.data

		with nogil:
			if self.hmm == 0:
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
			if self.hmm == 1:
				r[j] = (<Model> self.distributions_ptr[j])._vl_log_probability(X, d)
			else:
				(<Model> self.distributions_ptr[j])._v_log_probability(X, r+j*n, n)

		for i in range(n):
			max_logp = NEGINF

			for j in range(self.n):
				logp = r[j*n + i] + self.weights_ptr[j]
				if logp > max_logp:
					max_logp = logp
					y[i] = j

		free(r)

	def fit( self, X, y, weights=None, n_jobs=1, inertia=0.0 ):
		"""Fit the Naive Bayes model to the data by passing data to their components.

		Parameters
		----------
		X : numpy.ndarray or list
			The dataset to operate on. For most models this is a numpy array with
			columns corresponding to features and rows corresponding to samples.
			For markov chains and HMMs this will be a list of variable length
			sequences.

		y : numpy.ndarray or list or None, optional
			Data labels for supervised training algorithms. Default is None

		weights : array-like or None, shape (n_samples,), optional
			The initial weights of each sample in the matrix. If nothing is
			passed in then each sample is assumed to be the same weight.
			Default is None.

		n_jobs : int
			The number of jobs to use to parallelize, either the number of threads
			or the number of processes to use. Default is 1.

		inertia : double, optional
			Inertia used for the training the distributions.

		Returns
		-------
		self : object
			Returns the fitted model
		"""

		self.summarize( X, y, weights, n_jobs=n_jobs )
		self.from_summaries( inertia )
		return self

	def summarize( self, X, y, weights=None, n_jobs=1 ):
		"""Summarize data into stored sufficient statistics for out-of-core training.

		Parameters
		----------
		X : array-like, shape (n_samples, variable)
			Array of the samples, which can be either fixed size or variable depending
			on the underlying components.

		y : array-like, shape (n_samples,)
			Array of the known labels as integers

		weights : array-like, shape (n_samples,) optional
			Array of the weight of each sample, a positive float

		n_jobs : int
			The number of jobs to use to parallelize, either the number of threads
			or the number of processes to use. Default is 1.

		Returns
		-------
		None
		"""

		X = _convert(X)
		y = _convert(y)

		if self.d > 0 and not isinstance( self.distributions[0], HiddenMarkovModel ):
			if X.ndim > 2:
				raise ValueError("input data has too many dimensions")
			elif X.ndim == 2 and self.d != X.shape[1]:
				raise ValueError("input data rows do not match model dimension")

		if weights is None:
			weights = numpy.ones(X.shape[0]) / X.shape[0]
		else:
			weights = numpy.array(weights) / numpy.sum(weights)

		if len(X.shape) == 1 and not isinstance(X[0], list):
			X = X.reshape( X.shape[0], 1 )

		n = numpy.unique(y).shape[0]
		self.n = n

		if self.d == 0:
			self.distributions = numpy.array([self.distribution_callable] * n)
			self.d = X.shape[1]
			self.weights = numpy.ones(n, dtype=numpy.float64) / n
			self.weights_ptr = <double*> (<numpy.ndarray> self.weights).data
			self.summaries = numpy.zeros(n)

			if self.distributions[0] is DiscreteDistribution:
				self.keymap = { key: i for i, key in enumerate(numpy.unique(X)) }

		elif n != len(self.distributions):
			self.distributions = numpy.array([self.distributions[0].__class__] * n)
			self.d = X.shape[1]
			self.weights = numpy.ones(n, dtype=numpy.float64) / n
			self.weights_ptr = <double*> (<numpy.ndarray> self.weights).data
			self.summaries = numpy.zeros(n)

		for i in range(n):
			self.summaries[i] += weights[y==i].sum()
			if callable(self.distributions[i]):
				self.distributions[i] = self.distributions[i].from_samples(X[0:2])

			if isinstance(self.distributions[i], HiddenMarkovModel):
				self.distributions[i].fit( list(X[y==i]), weights=weights[y==i], n_jobs=n_jobs )

		delay = delayed(lambda model, x, weights: model.summarize(x, weights), check_pickle=False)
		with Parallel(n_jobs=n_jobs, backend='threading') as parallel:
			parallel( delay(self.distributions[i], X[y==i], weights[y==i]) for i in range(n) )


	def from_summaries( self, inertia=0.0 ):
		"""Fit the Naive Bayes model to the stored sufficient statistics.

		Parameters
		----------
		inertia : double, optional
			Inertia used for the training the distributions.

		Returns
		-------
		self : object
			Returns the fitted model
		"""

		n = len(self.distributions)
		self.summaries /= self.summaries.sum()

		self.distributions = numpy.array( self.distributions )
		self.distributions_ptr = <void**> self.distributions.data

		for i in range(n):
			if not isinstance( self.distributions[i], HiddenMarkovModel ):
				self.distributions[i].from_summaries(inertia=inertia)

			self.weights[i] = self.summaries[i]

		self.summaries = numpy.zeros(n)
		return self

	def to_json( self, separators=(',', ' : '), indent=4 ):
		if self.d == 0:
			raise ValueError("must fit components to the data before prediction")

		nb = {
			'class' : 'NaiveBayes',
			'models' : [ json.loads( model.to_json() ) for model in self.distributions ],
			'weights' : self.weights.tolist()
		}

		return json.dumps( nb, separators=separators, indent=indent )

	def from_json( cls, s ):
		try:
			d = json.loads( s )
		except:
			try:
				with open( s, 'r' ) as f:
					d = json.load( f )
			except:
				raise IOError("String must be properly formatted JSON or filename of properly formatted JSON.")

		models = list()
		for j in d['models']:
			if j['class'] == 'Distribution':
				models.append( Distribution.from_json( json.dumps(j) ) )
			elif j['class'] == 'GeneralMixtureModel':
				models.append( GeneralMixtureModel.from_json( json.dumps(j) ) )
			elif j['class'] == 'HiddenMarkovModel':
				models.append( HiddenMarkovModel.from_json( json.dumps(j) ) )
			elif j['class'] == 'BayesianNetwork':
				models.append( BayesianNetwork.from_json( json.dumps(j) ) )

		nb = NaiveBayes( models, numpy.array( d['weights'] ) )
		return nb

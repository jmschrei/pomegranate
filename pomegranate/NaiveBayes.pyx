# NaiveBayes.pyx
# Authors: Jacob Schreiber <jmschreiber91@gmail.com>

"""
Naive Bayes estimator, for anything with a log_probability method.
"""

import numpy
cimport numpy

from .hmm import HiddenMarkovModel
from .utils cimport pair_lse 

cdef double NEGINF = float("-inf")

cdef class NaiveBayes( object ):
	"""A Naive Bayes object, a supervised alternative to GMM.

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

	cdef int initialized
	cdef public object models
	cdef numpy.ndarray summaries
	cdef public numpy.ndarray weights

	def __init__( self, models=None, weights=None ):
		if not callable(models) and not isinstance(models, list): 
			raise ValueError("must either give initial models or constructor")

		self.summaries = None
		self.initialized = False

		if not callable(models):
			self.initialized = True
			self.summaries = numpy.zeros(len(models))

			if weights is None:
				self.weights = numpy.ones(len(models), dtype='float64') / len(models)
			else:
				self.weights = numpy.array(weights) / numpy.sum(weights)

		self.models = models

	def fit( self, X, y, weights=None, inertia=0.0 ):
		"""
		Fit the Naive Bayes model to the data in a supervised manner. Do this
		by passing the points associated with each model to only that model for
		training.
		"""

		self.summarize( X, y )
		self.from_summaries( inertia )


	def summarize( self, X, y, weights=None ):
		"""Summarize the values by storing the sufficient statistics in each model."""

		X = numpy.array(X)
		y = numpy.array(y)

		if weights is None:
			weights = numpy.ones(X.shape[0]) / X.shape[0]
		else:
			weights = numpy.array(weights, dtype='float64') / numpy.sum(weights)

		if len(X.shape) == 1:
			X = X.reshape( X.shape[0], 1 )

		if not self.initialized:
			n = numpy.unique(y).shape[0]
			self.models = [self.models] * n
			self.weights = numpy.ones(n, dtype=numpy.float64) / n
			self.summaries = numpy.zeros(n)
			self.initialized = True

		n = len(self.models)

		for i in range(n):
			if callable(self.models[i]):
				self.models[i] = self.models[i].from_samples(X[0:1])

			if isinstance( self.models[i], HiddenMarkovModel ):
				self.models[i].summarize( list(X[y==i]) )
			else:
				self.models[i].summarize( X[y==i], weights[y==i] )

			self.summaries[i] += weights[y==i].shape

	def from_summaries( self, inertia=0.0 ):
		"""Update the underlying distributions using the stored sufficient statistics."""

		n = len(self.models)
		self.summaries /= self.summaries.sum()

		for i in range(n):
			self.models[i].from_summaries(inertia=inertia)
			self.weights[i] = self.summaries[i]

		self.summaries = numpy.zeros(n) 
	
	cpdef predict_log_proba( self, X ):
		"""Predict the log probability of each sample under each model using Bayes Rule."""

		X = numpy.array(X)

		if self.initialized == 0:
			raise ValueError("must fit components to the data before prediction,")

		n, m = X.shape[0], len(self.models)
		r = numpy.zeros( (n, m), dtype=numpy.float64 )
		logw = numpy.log( self.weights )

		for i in range(n):
			total = NEGINF
			
			for j in range(m):
				r[i, j] = self.models[j].log_probability( X[i] ) + logw[j]
				total = pair_lse( total, r[i, j] )

			for j in range(m):
				r[i, j] -= total

		return r

	cpdef predict_proba( self, X ):
		"""Predict the probability of each sample under each model using Bayes Rule."""

		X = numpy.array(X)

		if self.initialized == 0:
			raise ValueError("must fit components to the data before prediction,")

		n, m = X.shape[0], len(self.models)
		r = numpy.zeros( (n, m), dtype=numpy.float64 )

		for i in range(n):
			total = 0.
			
			for j in range(m):
				r[i, j] = numpy.e ** self.models[j].log_probability( X[i] ) * self.weights[j]
				total += r[i, j]

			for j in range(m):
				r[i, j] /= total

		return r

	cpdef predict( self, X ):
		"""Predict the class label for each sample under each model."""

		if self.initialized == 0:
			raise ValueError("must fit components to the data before prediction,")

		return self.predict_proba(X).argmax( axis=1 )

	#def to_json( self, separators=(',', ' : '), indent=4 ):

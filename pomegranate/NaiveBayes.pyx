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

	This model can be initialized either by passing in initial distribution
	objects or by passing in the constructor and the number of components.

	>>> dist = NaiveBayes( [NormalDistribution(5, 1), NormalDistribution(6, 2)] )
	>>> dist = NaiveBayes( NormalDistribution, n_components=2 )
	"""

	cdef public object models
	cdef numpy.ndarray summaries
	cdef public numpy.ndarray weights

	def __init__( self, models=None, weights=None ):
		if not callable(models) and not isinstance(models, list): 
			raise ValueError("must either give initial models or constructor")

		if weights is None and not callable(models):
			self.weights = numpy.ones( len(models), dtype=numpy.float64 ) / len(models)
			self.summaries = numpy.zeros( len(models) )
		elif weights is not None and not callable(models):
			self.weights = numpy.array(weights) / numpy.sum(weights)
			self.summaries = numpy.zeros( len(models) )
		else:
			self.summaries = None

		self.models = models

	cpdef fit( self, X, numpy.ndarray y ):
		"""
		Fit the Naive Bayes model to the data in a supervised manner. Do this
		by passing the points associated with each model to only that model for
		training.
		"""

		cdef int i, n
		cdef numpy.ndarray data, X_ndarray
		cdef list data_list

		if callable(self.models):
			n = numpy.unique(y).shape[0]
			self.models = [ self.models ] * n
			self.weights = numpy.ones( n, dtype=numpy.float64 ) / n
			self.summaries = numpy.zeros(n)
		else:
			n = len(self.models)

		X_ndarray = numpy.array(X)

		for i in range(n):
			data = X_ndarray[ y==i ]

			if isinstance( self.models[i], HiddenMarkovModel ):
				data_list = list( data )
				self.models[i].fit( data_list )
			elif callable( self.models[i] ):
				self.models[i] = self.models[i].from_samples( data )
			else:
				self.models[i].fit( data )

			self.weights[i] = float(data.shape[0]) / X.shape[0]

	def summarize( self, X, y ):
		"""Summarize the values by storing the sufficient statistics in each model."""

		X_ndarray = numpy.array(X)
		n = len(self.models)

		for i in range(n):
			data = X_ndarray[ y==i ]

			if isinstance( self.models[i], HiddenMarkovModel ):
				data_list = list( data )
				self.models[i].summarize( data_list )
			else:
				self.models[i].summarize( data )

			self.summaries[i] += data.shape[0]

	def from_summaries( self, inertia=0.0 ):
		"""Update the underlying distributions using the stored sufficient statistics."""

		n = len(self.models)

		self.summaries /= self.summaries.sum()

		for i in range(n):
			self.models[i].from_summaries(inertia=inertia)
			self.weights[i] = self.summaries[i]

		self.summaries = numpy.zeros(n) 
	
	cpdef predict_log_proba( self, numpy.ndarray X ):
		"""Predict the log probability of each sample under each model using Bayes Rule."""

		if callable(self.models):
			raise ValueError("must fit components to the data before prediction,")

		cdef int i, j, n = X.shape[0], m = len(self.models)
		cdef numpy.ndarray r = numpy.zeros( (n, m), dtype=numpy.float64 )
		cdef double total
		cdef double [:] logw = numpy.log( self.weights )

		for i in range(n):
			total = NEGINF
			
			for j in range(m):
				r[i, j] = self.models[j].log_probability( X[i] ) + logw[j]
				total = pair_lse( total, r[i, j] )

			for j in range(m):
				r[i, j] -= total

		return r

	cpdef predict_proba( self, numpy.ndarray X ):
		"""Predict the probability of each sample under each model using Bayes Rule."""

		if callable(self.models):
			raise ValueError("must fit components to the data before prediction,")

		cdef int i, j, n = X.shape[0], m = len(self.models)
		cdef numpy.ndarray r = numpy.zeros( (n, m), dtype=numpy.float64 )
		cdef double total

		for i in range(n):
			total = 0.
			
			for j in range(m):
				r[i, j] = numpy.e ** self.models[j].log_probability( X[i] ) * self.weights[j]
				total += r[i, j]

			for j in range(m):
				r[i, j] /= total

		return r

	cpdef predict( self, numpy.ndarray X ):
		"""Predict the class label for each sample under each model."""

		if callable(self.models):
			raise ValueError("must fit components to the data before prediction,")

		return self.predict_proba( X ).argmax( axis=1 )
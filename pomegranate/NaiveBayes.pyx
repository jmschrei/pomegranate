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

	cdef list models
	cdef numpy.ndarray weights_ndarray
	cdef double* weights

	def __init__( self, models=[], n_components=-1 ):
		if models == [] and n_components == -1:
			raise ValueError("Must either give initial models or constructor and n_components")

		if n_components > 0:
			models = [ models ] * n_components

		self.models = models
		self.weights_ndarray = numpy.zeros( len(models), dtype=numpy.float64 ) + 1. / len(models)
		self.weights = <double*> self.weights_ndarray.data

	cpdef fit( self, X, numpy.ndarray y ):
		"""
		Fit the Naive Bayes model to the data in a supervised manner. Do this
		by passing the points associated with each model to only that model for
		training.
		"""

		cdef int i, n = len(self.models)
		cdef numpy.ndarray data, X_ndarray
		cdef list data_list

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
	
	cpdef predict_log_proba( self, numpy.ndarray X ):
		"""
		Predict the log probability of each sample under each model using Bayes Rule.
		"""

		cdef int i, j, n = X.shape[0], m = len(self.models)
		cdef numpy.ndarray r = numpy.zeros( (n, m), dtype=numpy.float64 )
		cdef double total
		cdef double [:] logw = numpy.log( self.weights_ndarray )

		for i in range(n):
			total = NEGINF
			
			for j in range(m):
				r[i, j] = self.models[j].log_probability( X[i] ) + logw[j]
				total = pair_lse( total, r[i, j] )

			for j in range(m):
				r[i, j] -= total

		return r

	cpdef predict_proba( self, numpy.ndarray X ):
		"""
		Predict the probability of each sample under each model using Bayes Rule.
		"""

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
		"""
		Predict the class label for each sample under each model.
		"""

		return self.predict_proba( X ).argmax( axis=1 )
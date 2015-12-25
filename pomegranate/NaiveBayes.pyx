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
	"""A Naive Bayes object, a supervised alternative to GMM."""

	cdef list models
	cdef numpy.ndarray weights_ndarray
	cdef double* weights

	def __init__( self, models ):
		self.models = models
		self.weights_ndarray = numpy.zeros( len(models), dtype=numpy.float64 ) + 1. / len(models)
		self.weights = <double*> self.weights_ndarray.data

	cpdef fit( self, numpy.ndarray X, numpy.ndarray y ):
		"""
		Fit the Naive Bayes model to the data in a supervised manner. Do this
		by passing the points associated with each model to only that model for
		training.
		"""

		cdef int i, n = len(self.models)
		cdef numpy.ndarray data
		cdef list data_list

		for i in range(n):
			data = X[ y==i ]

			if isinstance( self.models[i], HiddenMarkovModel ):
				data_list = list( data )
				self.models[i].fit( data_list )
			else:
				self.models[i].fit( data )

			self.weights[i] = float(data.shape[0]) / X.shape[0]
			print self.weights_ndarray
	
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
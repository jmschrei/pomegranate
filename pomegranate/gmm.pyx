# gmm.pyx: Yet Another Hidden Markov Model library
# Contact: Jacob Schreiber ( jmschreiber91@gmail.com )

cimport cython
from cython.view cimport array as cvarray
from libc.math cimport log as clog, sqrt as csqrt, exp as cexp
import math, random, itertools as it, sys, bisect
import networkx

if sys.version_info[0] > 2:
	# Set up for Python 3
	from functools import reduce
	xrange = range
	izip = zip
else:
	izip = it.izip

import numpy
cimport numpy

cimport distributions
from distributions cimport *

cimport utils
from utils cimport *

cimport base
from base cimport *

# Define some useful constants
DEF NEGINF = float("-inf")
DEF INF = float("inf")
DEF SQRT_2_PI = 2.50662827463

# Useful python-based array-intended operations
def log(value):
	"""
	Return the natural log of the given value, or - infinity if the value is 0.
	Can handle both scalar floats and numpy arrays.
	"""

	if isinstance( value, numpy.ndarray ):
		to_return = numpy.zeros(( value.shape ))
		to_return[ value > 0 ] = numpy.log( value[ value > 0 ] )
		to_return[ value == 0 ] = NEGINF
		return to_return
	return _log( value )
		
def exp(value):
	"""
	Return e^value, or 0 if the value is - infinity.
	"""
	
	return numpy.exp(value)

cdef class GaussianMixtureModel( object ):
	"""
	A Gaussian Mixture Model. Currently assumes a diagonal covariance matrix.
	"""

	cdef list means 

	def __init__( self, means, covariance ):
		"""
		Take in a list of MultivariateDistributions to be optimized.
		"""

		self.distributions = distributions

	def from_sample( self, items, weights=None ):
		"""
		Take in a list of data points and their respective weights. These are
		most likely uniformly weighted, but the option exists if you want to
		add a second layer of weights on top of the ones learned in the
		expectation step.
		"""

		pass




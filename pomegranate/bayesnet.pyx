# bayesnet.pyx
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

cimport utils 
from utils cimport *

cimport distributions
from distributions cimport *

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

cdef class BayesianNetwork( Model ):
	"""
	Represents a Bayesian Network
	"""

	def add_edges( self, a, b, weights ):
		"""
		Add many transitions at the same time, in one of two forms. 

		(1) If both a and b are lists, then create transitions from the i-th 
		element of a to the i-th element of b with a probability equal to the
		i-th element of probabilities.

		Example: 
		model.add_transitions([model.start, s1], [s1, model.end], [1., 1.])

		(2) If either a or b are a state, and the other is a list, create a
		transition from all states in the list to the single state object with
		probabilities and pseudocounts specified appropriately.

		Example:
		model.add_transitions([model.start, s1, s2, s3], s4, [0.2, 0.4, 0.3, 0.9])
		model.add_transitions(model.start, [s1, s2, s3], [0.6, 0.2, 0.05])

		If a single group is given, it's assumed all edges should belong to that
		group. Otherwise, either groups can be a list of group identities, or
		simply None if no group is meant.
		"""

		n = len(a) if isinstance( a, list ) else len(b)

		# Allow addition of many transitions from many states
		if isinstance( a, list ) and isinstance( b, list ):
			# Set up an iterator across all edges
			for start, end, weight in izip( a, b, weights ):
				self.add_transition( start, end, weight )

		# Allow for multiple transitions to a specific state 
		elif isinstance( a, list ) and isinstance( b, State ):
			# Set up an iterator across all edges to b
			for start, weight in izip( a, weights ):
				self.add_transition( start, b, weight )

		# Allow for multiple transitions from a specific state
		elif isinstance( a, State ) and isinstance( b, list ):
			# Set up an iterator across all edges from a
			for end, weight in izip( b, weights ):
				self.add_transition( a, end, weight )

	def sample( self ):
		'''
		Generate a random sample from this model.
		'''

		n = len( self.States )
		samples = [-1]*n
		visited = 0

		while True:
			for i in xrange( n ):
				if samples[i] != -1:
					continue

				sample = 0

				for k in xrange( self.in_edge_count[i], self.in_edge_count[i+1] ):
					ki = self.in_transitions[k]

					if samples[ki] == -1:
						sample = -1
						break
					else:
						sample += samples[ki]*self.in_transition_weights[k]  

				if sample >= 0:
					samples[i] = sample + self.States[i].distribution.sample()
					visited += 1

			if visited == n:
				break

		return samples
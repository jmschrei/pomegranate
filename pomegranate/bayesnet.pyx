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

def merge_marginals( marginals ):
	'''
	Merge multiple marginals of the same distribution to form a more informed
	distribution.
	'''

	probabilities = { key: _log( value ) for key, value in marginals[0].parameters[0].items() }

	for marginal in marginals[1:]:
		for key, value in marginal.parameters[0].items():
			probabilities[key] += _log( value )

	total = NEGINF
	for key, value in probabilities.items():
		total = pair_lse( total, probabilities[key] )

	for key, value in probabilities.items():
		probabilities[key] = cexp( value - total )

	return DiscreteDistribution( probabilities )

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

	def log_probability( self, data ):
		'''
		Determine the log probability of the data given the model. The data is
		expected to come in as a dictionary, with the indexes being the names
		of the states, and the keys being the value the data takes at that state.
		'''

		names = { state.name: state.distribution for state in self.states }
		data = { names[state]: value for state, value in data.items() }
		log_probability = 0
 
		for state in self.states:
			d = state.distribution

			# If the distribution is conditional on other distributions, then
			# pass the full dictionary of values in
			if isinstance( d, ConditionalDistribution ):
				log_probability += d.log_probability( data[d], data )

			# Otherwise calculate the log probability of this value
			# independently of the other values.
			else:
				log_probability += d.log_probability( data[d] )

		return log_probability

	def forward( self, data={} ):
		'''
		Calculate the posterior probabilities of each hidden variable, which
		are the variables not observed in the data.
		'''

		names = { state.name: state.distribution for state in self.states }
		data = { names[state]: value for state, value in data.items() }

		factors = [ data[ s.distribution ] if s.distribution in data else None for s in self.states ]

		in_edges = numpy.array( self.in_edge_count )
		out_edges = numpy.array( self.out_edge_count )

		roots = numpy.where( in_edges[1:] - in_edges[:-1] == 0 )[0]
		visited = numpy.zeros( len( self.states ) )
		for i, state in enumerate( self.states ):
			if state.distribution in data.keys():
				visited[i] = 1

		for root in roots:
			visited[ root ] = 1
			if factors[ root ] is not None:
				continue

			state = self.states[ root ]
			d = state.distribution

			if state.name in data:
				factors[ root ] = data[ d ]
			else:
				factors[ root ] = d

		while True:
			for i, state in enumerate( self.states ):
				if visited[ i ] == 1:
					continue

				state = self.states[ i ]
				d = state.distribution

				for k in xrange( in_edges[i], in_edges[i+1] ):
					ki = self.in_transitions[k]

					if visited[ki] == 0:
						break
				else:
					parents = {}
					for k in xrange( in_edges[i], in_edges[i+1] ):
						ki = self.in_transitions[k]
						d = self.states[ki].distribution

						parents[d] = factors[ki]

					factors[i] = state.distribution.marginal( parents )
					visited[i] = 1

			if visited.sum() == visited.shape[0]:
				break

		return factors

	def backward( self, data={} ):
		'''
		'''

		names = { state.name: state.distribution for state in self.states }
		data = { names[state]: value for state, value in data.items() }

		factors = [ data[ s.distribution ] if s.distribution in data else s.distribution.marginal() for s in self.states ]
		new_factors = [ i for i in factors ]

		in_edges = numpy.array( self.in_edge_count )
		out_edges = numpy.array( self.out_edge_count )

		messages = [ None for i in in_edges ]

		leaves = numpy.where( out_edges[1:] - out_edges[:-1] == 0 )[0]
		visited = numpy.zeros( len( self.states ) )
		visited[leaves] = 1
		for i, s in enumerate( self.states ):
			if s.distribution in data and not isinstance( data[ s.distribution ], Distribution ): 
				visited[i] = 1 

		while True:
			for i, state in enumerate( self.states ):
				if visited[i] == 1:
					continue

				state = self.states[i]
				d = state.distribution

				for k in xrange( out_edges[i], out_edges[i+1] ):
					ki = self.out_transitions[k]
					if visited[ki] == 0:
						break
				else:
					for k in xrange( out_edges[i], out_edges[i+1] ):
						ki = self.out_transitions[k]

						parents = {}
						for l in xrange( in_edges[ki], in_edges[ki+1] ):
							li = self.in_transitions[l]
							parents[ self.states[li].distribution ] = factors[li]

						messages[k] = self.states[ki].distribution.marginal( parents, wrt=d, value=new_factors[ki] )
					else:
						local_messages = [ factors[i] ] + [ messages[k] for k in xrange( out_edges[i], out_edges[i+1] ) ]
						new_factors[i] = merge_marginals( local_messages )

					visited[i] = 1

			if visited.sum() == visited.shape[0]:
				break

		return new_factors 


	def forward_backward( self, data={} ):
		'''
		...
		'''

		print "FORWARD LOGS"
		factors = self.forward( data )
		data = { self.states[i].name: factors[i] for i in xrange( len(factors) ) }

		print "\n".join( map( str, factors ) )
		print
		print "BACKWARD LOGS"
		return self.backward( data )
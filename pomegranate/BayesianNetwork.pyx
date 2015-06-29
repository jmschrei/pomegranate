# BayesianNetwork.pyx
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
from base cimport Model, State

from hmm import *
from FactorGraph import *

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

	def bake( self, verbose=False ): 
		"""
		The Bayesian Network is going to be mostly a wrapper for the Factor
		Graph, as probabilities, inference, and training can be done more
		efficiently on them.
		"""

		# Initialize the factor graph
		self.graph = FactorGraph( self.name+'-fg' )

		# Build the factor graph structure we'll use for all the important
		# calculations.
		j = 0

		# Create two mappings, where edges which previously went to a
		# conditional distribution now go to a factor, and those which left
		# a conditional distribution now go to a marginal
		f_mapping, m_mapping = {}, {}
		d_mapping = {}
		fa_mapping = {}

		# Go through each state and add in the state if it is a marginal
		# distribution, otherwise add in the appropriate marginal and
		# conditional distribution as separate nodes.
		for i, state in enumerate( self.states ):
			# For every state (ones with conditional distributions or those
			# encoding marginals) we need to create a marginal node in the
			# underlying factor graph. 
			keys = state.distribution.keys()
			d = DiscreteDistribution({ key: 1./len(keys) for key in keys })
			m = State( d, state.name )

			# Add the state to the factor graph
			self.graph.add_node( m )

			# Now we need to copy the distribution from the node into the
			# factor node. This could be the conditional table, or the
			# marginal.
			f = State( state.distribution.copy(), state.name+'-joint' )

			if isinstance( state.distribution, ConditionalProbabilityTable ):
				fa_mapping[f.distribution] = m.distribution 

			self.graph.add_node( f )
			self.graph.add_edge( m, f )

			f_mapping[state] = f
			m_mapping[state] = m
			d_mapping[state.distribution] = d

			# Progress the counter by one
			j += 1

		for a, b in self.edges:
			self.graph.add_edge( m_mapping[a], f_mapping[b] )

		# Now go back and redirect parent pointers to the appropriate
		# objects.
		for state in self.graph.states:
			d = state.distribution
			if isinstance( d, ConditionalProbabilityTable ):
				dist = fa_mapping[d]
				d.parameters[1] = [ d_mapping[parent] for parent in d.parameters[1] ]
				state.distribution = d.joint()
				state.distribution.parameters[1].append( dist )
				state.distribution.parameters[2][-1] = { key: i for i, key in enumerate( dist.keys() ) }
				state.distribution.parameters[3][-1] = len( dist )

		# Finalize the factor graph structure
		self.graph.bake()

	def marginal( self ):
		"""
		Return the marginal of the graph. This is equivilant to a pass of
		belief propogation given that no data is given; or a single forward
		pass of the sum-product algorithm.
		"""

		return self.graph.marginal()

	def forward_backward( self, data={}, max_iterations=100 ):
		"""
		Run loopy belief propogation on the underlying factor graph until
		convergence, and return the marginals.
		"""

		return self.graph.forward_backward( data, max_iterations )

	def log_probability( self, data ):
		"""
		Return the probability of the data given the model. This is just a
		product of the factors in the factor graph, so call the underlying
		factor graph representation to do that.
		"""

		return self.graph.log_probability( data )

	def from_sample( self, items, weights=None, inertia=0.0, pseudocount=0.0 ):
		"""
		Another name for the train method.
		"""

		self.train( items, weights, intertia, pseudocount )


	def train( self, items, weights=None, inertia=0.0, pseudocount=0.0 ):
		"""
		Take in data, with each column corresponding to observations of the
		state, as ordered by self.states.
		"""

		indices = { state.distribution: i for i, state in enumerate( self.states ) }

		# Go through each state and pass in the appropriate data for the
		# update to the states
		for i, state in enumerate( self.states ):
			if isinstance( state.distribution, ConditionalProbabilityTable ):
				idx = [ indices[ dist ] for dist in state.distribution.parameters[1] ] + [i]
				data = [ [ item[i] for i in idx ] for item in items ]
				state.distribution.from_sample( data, weights, inertia, pseudocount )
			else:
				state.distribution.from_sample( [ item[i] for item in items ], 
					weights, inertia, pseudocount )

		self.bake()
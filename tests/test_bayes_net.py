# test_bayes_net_monty.py
# Contact: Jacob Schreiber
#          jmschreiber91@gmail.com

'''
These are unit tests for the Bayesian network part of pomegranate.
'''

from __future__ import  division

from pomegranate import *
from nose.tools import with_setup
from nose.tools import assert_equal
from nose.tools import assert_raises
import random
import numpy

nan = numpy.nan

def setup():
	global network, monty_index, prize_index, guest_index

	random.seed(0)

	# Friends emisisons are completely random
	guest = DiscreteDistribution( { 'A': 1./3, 'B': 1./3, 'C': 1./3 } )

	# The actual prize is independent of the other distributions
	prize = DiscreteDistribution( { 'A': 1./3, 'B': 1./3, 'C': 1./3 } )

	# Monty is dependent on both the guest and the prize. 
	monty = ConditionalProbabilityTable(
		[[ 'A', 'A', 'A', 0.0 ],
		 [ 'A', 'A', 'B', 0.5 ],
		 [ 'A', 'A', 'C', 0.5 ],
		 [ 'A', 'B', 'A', 0.0 ],
		 [ 'A', 'B', 'B', 0.0 ],
		 [ 'A', 'B', 'C', 1.0 ],
		 [ 'A', 'C', 'A', 0.0 ],
		 [ 'A', 'C', 'B', 1.0 ],
		 [ 'A', 'C', 'C', 0.0 ],
		 [ 'B', 'A', 'A', 0.0 ],
		 [ 'B', 'A', 'B', 0.0 ],
		 [ 'B', 'A', 'C', 1.0 ],
		 [ 'B', 'B', 'A', 0.5 ],
		 [ 'B', 'B', 'B', 0.0 ],
		 [ 'B', 'B', 'C', 0.5 ],
		 [ 'B', 'C', 'A', 1.0 ],
		 [ 'B', 'C', 'B', 0.0 ],
		 [ 'B', 'C', 'C', 0.0 ],
		 [ 'C', 'A', 'A', 0.0 ],
		 [ 'C', 'A', 'B', 1.0 ],
		 [ 'C', 'A', 'C', 0.0 ],
		 [ 'C', 'B', 'A', 1.0 ],
		 [ 'C', 'B', 'B', 0.0 ],
		 [ 'C', 'B', 'C', 0.0 ],
		 [ 'C', 'C', 'A', 0.5 ],
		 [ 'C', 'C', 'B', 0.5 ],
		 [ 'C', 'C', 'C', 0.0 ]], [guest, prize] )  


	# Make the states
	s1 = State( guest, name="guest" )
	s2 = State( prize, name="prize" )
	s3 = State( monty, name="monty" )

	# Make the bayes net, add the states, and the conditional dependencies.
	network = BayesianNetwork( "test" )
	network.add_nodes(s1, s2, s3)
	network.add_edge( s1, s3 )
	network.add_edge( s2, s3 )
	network.bake()

	monty_index = network.states.index( s3 )
	prize_index = network.states.index( s2 )
	guest_index = network.states.index( s1 )

def teardown():
	pass

def discrete_equality( x, y, z=8 ):
	xd, yd = x.parameters[0], y.parameters[0]
	for key, value in xd.items():
		if round( yd[key], z ) != round( value, z ):
			raise ValueError( "{} != {}".format( yd[key], value ) )

def test_guest():
	a = network.predict_proba( {'guest' : 'A'} )
	b = network.predict_proba( {'guest' : 'B'} )
	c = network.predict_proba( {'guest' : 'C'} )

	prize_correct = DiscreteDistribution({'A' : 1./3, 'B' : 1./3, 'C' : 1./3 })

	discrete_equality( a[prize_index], b[prize_index] )
	discrete_equality( a[prize_index], c[prize_index] )
	discrete_equality( a[prize_index], prize_correct )

	discrete_equality( a[monty_index], DiscreteDistribution({'A': 0.0, 'B' : 1./2, 'C' : 1./2}) )
	discrete_equality( b[monty_index], DiscreteDistribution({'A': 1./2, 'B' : 0.0, 'C' : 1./2}) )
	discrete_equality( c[monty_index], DiscreteDistribution({'A': 1./2, 'B' : 1./2, 'C' : 0.0}) )

def test_guest_monty():
	b = network.predict_proba( { 'guest' : 'A', 'monty' : 'B' } )
	c = network.predict_proba( { 'guest' : 'A', 'monty' : 'C' } )

	discrete_equality( b[guest_index], DiscreteDistribution({'A': 1., 'B': 0., 'C': 0. }) )
	discrete_equality( b[monty_index], DiscreteDistribution({'A': 0., 'B': 1., 'C': 0. }) )
	discrete_equality( b[prize_index], DiscreteDistribution({'A': 1./3, 'B': 0.0, 'C': 2./3 }) )
	discrete_equality( c[guest_index], DiscreteDistribution({'A': 1., 'B': 0., 'C': 0. }) )
	discrete_equality( c[monty_index], DiscreteDistribution({'A': 0., 'B': 0., 'C': 1. }) )
	discrete_equality( c[prize_index], DiscreteDistribution({'A': 1./3, 'B': 2./3, 'C': 0.0 }) )

def test_monty():
	a = network.predict_proba({ 'monty' : 'A' })

	discrete_equality( a[monty_index], DiscreteDistribution({'A': 1.0, 'B': 0.0, 'C': 0.0}) )
	discrete_equality( a[guest_index], a[prize_index] )
	discrete_equality( a[guest_index], DiscreteDistribution({'A' : 0.0, 'B' : 1./2, 'C' : 1./2}) )

def test_imputation():
	obs = [['A', None, 'B'],
	       ['A', None, 'C'],
	       ['A', 'B', 'C']]

	network.predict( obs )

	assert_equal( obs[0], ['A', 'C', 'B'] )
	assert_equal( obs[1], ['A', 'B', 'C'] )
	assert_equal( obs[2], ['A', 'B', 'C'] )

def test_numpy_imputation():
	obs = numpy.array([['A', None, 'B'],
	       			   ['A', None, 'C'],
	       			   ['A', 'B', 'C']])

	network.predict( obs )

	assert_equal( obs[0, 0], 'A' )
	assert_equal( obs[0, 1], 'C' )
	assert_equal( obs[0, 2], 'B' )
	assert_equal( obs[1, 0], 'A' )
	assert_equal( obs[1, 1], 'B' )
	assert_equal( obs[1, 2], 'C' )
	assert_equal( obs[2, 0], 'A' )
	assert_equal( obs[2, 1], 'B' )
	assert_equal( obs[2, 2], 'C' )

def test_raise_error():
	obs = [['green', 'cat', None]]
	assert_raises( ValueError, network.predict, obs )

	obs = [['A', 'b', None]]
	assert_raises( ValueError, network.predict, obs )

	obs = [['none', 'B', None]]
	assert_raises( ValueError, network.predict, obs )

	obs = [['NaN', 'B', None]]
	assert_raises( ValueError, network.predict, obs )

	obs = [['A', 'C', 'D']]
	assert_raises( ValueError, network.predict, obs )

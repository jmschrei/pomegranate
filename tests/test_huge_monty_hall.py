# test_huge_monty_hall.py
# Contact: Jacob Schreiber
#          jmschr@cs.washington.edu

'''
These are unit tests for the Bayesian network part of pomegranate.
'''

from __future__ import  (division, print_function)

from pomegranate import *
from nose.tools import with_setup
from nose.tools import assert_equal
import numpy as np

def setup():
	'''
	Build the huge monty hall network. This is an example I made up with which
	may not exactly flow logically, but tests a varied type of tables ensures
	heterogeneous types of data work together.
	'''

	global network, friend, guest, remaining, randomize, prize, monty 

	# Friend
	friend = DiscreteDistribution( { True: 0.5, False: 0.5 } )

	# Guest emisisons are completely random
	guest = ConditionalProbabilityTable(
		[[ True, 'A', 0.50 ],
		 [ True, 'B', 0.25 ],
		 [ True, 'C', 0.25 ],
		 [ False, 'A', 0.0 ],
		 [ False, 'B', 0.7 ],
		 [ False, 'C', 0.3 ]], [friend] )

	# Number of remaining cars
	remaining = DiscreteDistribution( { 0: 0.1, 1: 0.7, 2: 0.2, } )

	# Whether they randomize is dependent on the numnber of remaining cars
	randomize = ConditionalProbabilityTable(
		[[ 0, True , 0.05 ],
	     [ 0, False, 0.95 ],
	     [ 1, True , 0.8 ],
	     [ 1, False, 0.2 ],
	     [ 2, True , 0.5 ],
	     [ 2, False, 0.5 ]], [remaining] )

	# Where the prize is depends on if they randomize or not and also the guests friend
	prize = ConditionalProbabilityTable(
		[[ True, True, 'A', 0.3 ],
		 [ True, True, 'B', 0.4 ],
		 [ True, True, 'C', 0.3 ],
		 [ True, False, 'A', 0.2 ],
		 [ True, False, 'B', 0.4 ],
		 [ True, False, 'C', 0.4 ],
		 [ False, True, 'A', 0.1 ],
		 [ False, True, 'B', 0.9 ],
		 [ False, True, 'C', 0.0 ],
		 [ False, False, 'A', 0.0 ],
		 [ False, False, 'B', 0.4 ],
		 [ False, False, 'C', 0.6]], [randomize, friend] )

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
	s0 = State( friend, name="friend")
	s1 = State( guest, name="guest" )
	s2 = State( prize, name="prize" )
	s3 = State( monty, name="monty" )
	s4 = State( remaining, name="remaining" )
	s5 = State( randomize, name="randomize" )

	# Make the bayes net, add the states, and the conditional dependencies.
	network = BayesianNetwork( "test" )
	network.add_nodes(s0, s1, s2, s3, s4, s5)
	network.add_transition( s0, s1 )
	network.add_transition( s1, s3 )
	network.add_transition( s2, s3 )
	network.add_transition( s4, s5 )
	network.add_transition( s5, s2 )
	network.add_transition( s0, s2 )
	network.bake()

def teardown():
	pass

@with_setup( setup, teardown )
def test_monty():
	assert_equal( monty.log_probability( ('A', 'A', 'C') ), np.log(0.5) )
	assert_equal( monty.log_probability( ('B', 'B', 'C') ), np.log(0.5) )
	assert_equal( monty.log_probability( ('C', 'C', 'C') ), float("-inf") )

	data = [[ True,  'A', 'A', 'C', 1, True  ],
			[ True,  'A', 'A', 'C', 0, True  ],
			[ False, 'A', 'A', 'B', 1, False ],
			[ False, 'A', 'A', 'A', 2, False ],
			[ False, 'A', 'A', 'C', 1, False ],
			[ False, 'B', 'B', 'B', 2, False ],
			[ False, 'B', 'B', 'C', 0, False ],
			[ True,  'C', 'C', 'A', 2, True  ],
			[ True,  'C', 'C', 'C', 1, False ],
			[ True,  'C', 'C', 'C', 0, False ],
			[ True,  'C', 'C', 'C', 2, True  ],
			[ True,  'C', 'B', 'A', 1, False ]]

	network.fit( data )

	assert_equal( monty.log_probability( ('A', 'A', 'C') ), np.log(0.6) )
	assert_equal( monty.log_probability( ('B', 'B', 'C') ), np.log(0.5) )
	assert_equal( monty.log_probability( ('C', 'C', 'C') ), np.log(0.75) )

@with_setup( setup, teardown )
def test_friend():
	assert_equal( friend.log_probability(True), np.log(0.5) )
	assert_equal( friend.log_probability(False), np.log(0.5) )

	data = [[ True,  'A', 'A', 'C', 1, True  ],
			[ True,  'A', 'A', 'C', 0, True  ],
			[ False, 'A', 'A', 'B', 1, False ],
			[ False, 'A', 'A', 'A', 2, False ],
			[ False, 'A', 'A', 'C', 1, False ],
			[ False, 'B', 'B', 'B', 2, False ],
			[ False, 'B', 'B', 'C', 0, False ],
			[ True,  'C', 'C', 'A', 2, True  ],
			[ True,  'C', 'C', 'C', 1, False ],
			[ True,  'C', 'C', 'C', 0, False ],
			[ True,  'C', 'C', 'C', 2, True  ],
			[ True,  'C', 'B', 'A', 1, False ]]

	network.fit( data )

	assert_equal( friend.log_probability(True), np.log(7./12) )
	assert_equal( friend.log_probability(False), np.log(5./12) )


@with_setup( setup, teardown )
def test_remaining():
	assert_equal( remaining.log_probability(0), np.log( 0.1 ) )
	assert_equal( remaining.log_probability(1), np.log( 0.7 ) )
	assert_equal( remaining.log_probability(2), np.log( 0.2 ) )

	data = [[ True,  'A', 'A', 'C', 1, True  ],
			[ True,  'A', 'A', 'C', 0, True  ],
			[ False, 'A', 'A', 'B', 1, False ],
			[ False, 'A', 'A', 'A', 2, False ],
			[ False, 'A', 'A', 'C', 1, False ],
			[ False, 'B', 'B', 'B', 2, False ],
			[ False, 'B', 'B', 'C', 0, False ],
			[ True,  'C', 'C', 'A', 2, True  ],
			[ True,  'C', 'C', 'C', 1, False ],
			[ True,  'C', 'C', 'C', 0, False ],
			[ True,  'C', 'C', 'C', 2, True  ],
			[ True,  'C', 'B', 'A', 1, False ]]

	network.fit( data )

	assert_equal( remaining.log_probability(0), np.log(3./12) )
	assert_equal( remaining.log_probability(1), np.log(5./12) )
	assert_equal( remaining.log_probability(2), np.log(4./12) )

@with_setup( setup, teardown )
def test_prize():
	assert_equal( prize.log_probability( (True,  True,  'A') ), np.log(0.3) )
	assert_equal( prize.log_probability( (True,  False, 'C') ), np.log(0.4) )
	assert_equal( prize.log_probability( (False, True,  'B') ), np.log(0.9) )
	assert_equal( prize.log_probability( (False, False, 'A') ), float("-inf") )

	data = [[ True,  'A', 'A', 'C', 1, True  ],
			[ True,  'A', 'A', 'C', 0, True  ],
			[ False, 'A', 'A', 'B', 1, False ],
			[ False, 'A', 'A', 'A', 2, False ],
			[ False, 'A', 'A', 'C', 1, False ],
			[ False, 'B', 'B', 'B', 2, False ],
			[ False, 'B', 'B', 'C', 0, False ],
			[ True,  'C', 'C', 'A', 2, True  ],
			[ True,  'C', 'C', 'C', 1, False ],
			[ True,  'C', 'C', 'C', 0, False ],
			[ True,  'C', 'C', 'C', 2, True  ],
			[ True,  'C', 'B', 'A', 1, False ]]

	network.fit( data )

	assert_equal( prize.log_probability( (True, True, 'C') ), np.log(0.5) )
	assert_equal( prize.log_probability( (True, True, 'B') ), float("-inf") )

	a = prize.log_probability( (True, False, 'A' ) )
	b = prize.log_probability( (True, False, 'B' ) )
	c = prize.log_probability( (True, False, 'C' ) )

	assert_equal( a, b )
	assert_equal( b, c )

	assert_equal( prize.log_probability( (False, False, 'C') ), float("-inf") )
	assert_equal( prize.log_probability( (False, True, 'C' ) ), np.log(2./3) )

# test_bayes_net_monty.py
# Contact: Jacob Schreiber
#          jmschr@cs.washington.edu

'''
Test a simplified version inspired by the Titanic bayes net, submitted as a test
case by a user.
'''

from __future__ import  (division, print_function)

from pomegranate import *
from nose.tools import with_setup
from nose.tools import assert_equal
import random
import numpy as np
import time

def setup():
	"""Build a simple model referring to the titanic disaster."""

	global network, passenger, gender, tclass

	# Passengers on the Titanic either survive or perish
	passenger = DiscreteDistribution( { 'survive': 0.6, 'perish': 0.4 } )

	# Gender, given survival data
	gender = ConditionalProbabilityTable(
	            [[ 'survive', 'male',   0.0 ],
	             [ 'survive', 'female', 1.0 ],
	             [ 'perish', 'male',    1.0 ],
		         [ 'perish', 'female',  0.0]], [passenger] )


	# Class of travel, given survival data
	tclass = ConditionalProbabilityTable(
	            [[ 'survive', 'first',  0.0 ],
	             [ 'survive', 'second', 1.0 ],
	             [ 'survive', 'third',  0.0 ],
	             [ 'perish', 'first',  1.0 ],
	             [ 'perish', 'second', 0.0 ],
		         [ 'perish', 'third',  0.0]], [passenger] )


	# State objects hold both the distribution, and a high level name.
	s1 = State( passenger, name = "passenger" )
	s2 = State( gender, name = "gender" )
	s3 = State( tclass, name = "class" )

	# Create the Bayesian network object with a useful name
	network = BayesianNetwork( "Titanic Disaster" )

	# Add the three nodes to the network
	network.add_nodes( [ s1, s2, s3 ] )

	# Add transitions which represent conditional depesndencies, where the second
	# node is conditionally dependent on the first node (Monty is dependent on both guest and prize)
	network.add_edge( s1, s2 )
	network.add_edge( s1, s3 )
	network.bake()

def teardown():
	pass

def test_network():
	assert_equal( passenger.log_probability('survive'), np.log(0.6) )
	assert_equal( passenger.log_probability('survive'), np.log(0.6) )

	assert_equal( gender.log_probability( ('survive', 'male') ),   float("-inf") )
	assert_equal( gender.log_probability( ('survive', 'female') ), 0.0 )
	assert_equal( gender.log_probability( ('perish', 'male') ),    0.0 )
	assert_equal( gender.log_probability( ('perish', 'female') ),  float("-inf") )

	assert_equal( tclass.log_probability( ('survive', 'first') ), float("-inf") )
	assert_equal( tclass.log_probability( ('survive', 'second') ), 0.0 )
	assert_equal( tclass.log_probability( ('survive', 'third') ), float("-inf") )
	assert_equal( tclass.log_probability( ('perish', 'first') ), 0.0 )
	assert_equal( tclass.log_probability( ('perish', 'second') ), float("-inf") )
	assert_equal( tclass.log_probability( ('perish', 'third') ), float("-inf") )

def test_guest():
	male   = network.forward_backward( {'gender' : 'male'   } )
	female = network.forward_backward( {'gender' : 'female' } )

	assert_equal( female[0].log_probability( "survive" ), 0.0 )
	assert_equal( female[0].log_probability( "perish" ), float("-inf") )

	assert_equal( female[1].log_probability( "male" ), float("-inf") )
	assert_equal( female[1].log_probability( "female" ), 0.0 )

	assert_equal( female[2].log_probability( "first" ), float("-inf") )
	assert_equal( female[2].log_probability( "second" ), 0.0 )
	assert_equal( female[2].log_probability( "third" ), float("-inf") )

	assert_equal( male[0].log_probability( "survive" ), float("-inf") )
	assert_equal( male[0].log_probability( "perish" ), 0.0 )

	assert_equal( male[1].log_probability( "male" ), 0.0 )
	assert_equal( male[1].log_probability( "female" ), float("-inf") )

	assert_equal( male[2].log_probability( "first" ), 0.0 )
	assert_equal( male[2].log_probability( "second" ), float("-inf") )
	assert_equal( male[2].log_probability( "third" ), float("-inf") )

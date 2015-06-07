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
import random
import numpy as np
import time

def setup():
	'''
	Build the model which corresponds to the Cold network.
	'''

	global network

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
	'''
	Tear down the network, aka do nothing.
	'''

def discrete_equality( x, y, z=8 ):
	'''
	Test to see if two discrete distributions are equal to each other to
	z decimal points.
	'''

	xd, yd = x.parameters[0], y.parameters[0]
	for key, value in xd.items():
		if round( yd[key], z ) != round( value, z ):
			return False
	return True

def test_guest():
	'''
	See what happens when the guest says something.
	'''

	male   = network.forward_backward( {'gender' : 'male'   } )
	female = network.forward_backward( {'gender' : 'female' } ) 

	assert female[0].log_probability( "survive" ) == 0.0
	assert female[0].log_probability( "perish" ) == float("-inf")

	assert female[1].log_probability( "male" ) == float("-inf")
	assert female[1].log_probability( "female" ) == 0.0

	assert female[2].log_probability( "first" ) == float("-inf")
	assert female[2].log_probability( "second" ) == 0.0
	assert female[2].log_probability( "third" ) == float("-inf")

	assert male[0].log_probability( "survive" ) == float("-inf")
	assert male[0].log_probability( "perish" )  == 0.0

	assert male[1].log_probability( "male" )   == 0.0
	assert male[1].log_probability( "female" ) == float("-inf")

	assert male[2].log_probability( "first" )  == 0.0
	assert male[2].log_probability( "second" ) == float("-inf")
	assert male[2].log_probability( "third" )  == float("-inf")
# test_bayes_net_monty.py
# Contact: Jacob Schreiber
#          jmschr@cs.washington.edu

'''
These are unit tests for the Finite State Machine part of pomegranate.
'''

from __future__ import  (division, print_function)

from pomegranate import *
from nose.tools import with_setup

def setup():
	'''
	Build a FSM which corresponds to a turnstile.
	'''

	global model

	# Create the states in the same way as you would an HMM
	a = State( None, "5"  )
	b = State( None, "10" )
	c = State( None, "15" )
	d = State( None, "20" )
	e = State( None, "25" )

	# Create a FiniteStateMachine object 
	model = FiniteStateMachine( "Turnstile" )

	# Add the states in the same way
	model.add_states( [a, b, c, d, e] )

	# Add in transitions by using nickels
	model.add_transition( model.start, a, 5 )
	model.add_transition( a, b, 5 )
	model.add_transition( b, c, 5 )
	model.add_transition( c, d, 5 )
	model.add_transition( d, e, 5 )

	# Add in transitions using dimes
	model.add_transition( model.start, b, 10 )
	model.add_transition( a, c, 10 )
	model.add_transition( b, d, 10 )
	model.add_transition( c, e, 10 )

	# Add in transitions using quarters
	model.add_transition( model.start, e, 25 )

	# Bake the model in the same way
	model.bake()

def teardown():
	'''
	Tear down the network, aka do nothing.
	'''

	model = None

@with_setup( setup, teardown )
def test_nickles():
	"""
	Test a valid sequence.
	"""

	seq = [ 5, 5, 5, 5, 5 ]

	assert model.current_index == 0

	for i, symbol in enumerate( seq ):
		model.step( symbol )
		assert model.current_index == i + 1

@with_setup( setup, teardown )
def test_dimes():
	"""
	Test a valid sequence.
	"""

	seq = [ 5, 10, 10 ]
	states = [ 1, 3, 5 ]

	assert model.current_index == 0

	for symbol, state in zip( seq, states ):
		model.step( symbol )
		assert model.current_index == state

@with_setup( setup, teardown )
def test_quarter():
	"""
	Test a valid sequence.
	"""

	assert model.current_index == 0
	model.step( 25 )
	assert model.current_index == 5

@with_setup( setup, teardown )
def test_invalid():
	"""
	Test a valid sequence.
	"""

	assert model.current_index == 0
	model.step( 10 )
	assert model.current_index == 2
	model.step( 25 )
	assert model.current_index == 2
	model.step( 5 )
	assert model.current_index == 3
	model.step( 10 )
	assert model.current_index == 5
	model.step( 25 )
	assert model.current_index == 5
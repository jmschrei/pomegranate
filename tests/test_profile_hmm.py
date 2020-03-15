from __future__ import  (division, print_function)

from pomegranate import *
from nose.tools import with_setup
from nose.tools import assert_equal
from nose.tools import assert_not_equal
from nose.tools import assert_raises
from nose.tools import assert_almost_equal
import random
import numpy as np
import json

def setup():
	'''
	Build a model that we want to use to test sequences. This model will
	be somewhat complicated, in order to extensively test YAHMM. This will be
	a three state global sequence alignment HMM. The HMM models a reference of
	'ACT', with pseudocounts to allow for slight deviations from this
	reference.
	'''

	random.seed(0)

	global model
	model = HiddenMarkovModel( "Global Alignment")

	# Define the distribution for insertions
	i_d = DiscreteDistribution( { 'A': 0.25, 'C': 0.25, 'G': 0.25, 'T': 0.25 } )

	# Create the insert states
	i0 = State( i_d, name="I0" )
	i1 = State( i_d, name="I1" )
	i2 = State( i_d, name="I2" )
	i3 = State( i_d, name="I3" )

	# Create the match states
	m1 = State( DiscreteDistribution({ "A": 0.95, 'C': 0.01, 'G': 0.01, 'T': 0.02 }) , name="M1" )
	m2 = State( DiscreteDistribution({ "A": 0.003, 'C': 0.99, 'G': 0.003, 'T': 0.004 }) , name="M2" )
	m3 = State( DiscreteDistribution({ "A": 0.01, 'C': 0.01, 'G': 0.01, 'T': 0.97 }) , name="M3" )

	# Create the delete states
	d1 = State( None, name="D1" )
	d2 = State( None, name="D2" )
	d3 = State( None, name="D3" )

	# Add all the states to the model
	model.add_states( [i0, i1, i2, i3, m1, m2, m3, d1, d2, d3 ] )

	# Create transitions from match states
	model.add_transition( model.start, m1, 0.9 )
	model.add_transition( model.start, i0, 0.1 )
	model.add_transition( m1, m2, 0.9 )
	model.add_transition( m1, i1, 0.05 )
	model.add_transition( m1, d2, 0.05 )
	model.add_transition( m2, m3, 0.9 )
	model.add_transition( m2, i2, 0.05 )
	model.add_transition( m2, d3, 0.05 )
	model.add_transition( m3, model.end, 0.9 )
	model.add_transition( m3, i3, 0.1 )

	# Create transitions from insert states
	model.add_transition( i0, i0, 0.70 )
	model.add_transition( i0, d1, 0.15 )
	model.add_transition( i0, m1, 0.15 )

	model.add_transition( i1, i1, 0.70 )
	model.add_transition( i1, d2, 0.15 )
	model.add_transition( i1, m2, 0.15 )

	model.add_transition( i2, i2, 0.70 )
	model.add_transition( i2, d3, 0.15 )
	model.add_transition( i2, m3, 0.15 )

	model.add_transition( i3, i3, 0.85 )
	model.add_transition( i3, model.end, 0.15 )

	# Create transitions from delete states
	model.add_transition( d1, d2, 0.15 )
	model.add_transition( d1, i1, 0.15 )
	model.add_transition( d1, m2, 0.70 )

	model.add_transition( d2, d3, 0.15 )
	model.add_transition( d2, i2, 0.15 )
	model.add_transition( d2, m3, 0.70 )

	model.add_transition( d3, i3, 0.30 )
	model.add_transition( d3, model.end, 0.70 )

	# Call bake to finalize the structure of the model.
	model.bake()


def multitransition_setup():
	'''
	Build a model that we want to use to test sequences. This is the same as the
	above model, except that it uses the multiple transition methods for building.
	'''

	random.seed(0)

	global model
	model = HiddenMarkovModel( "Global Alignment")

	# Define the distribution for insertions
	i_d = DiscreteDistribution( { 'A': 0.25, 'C': 0.25, 'G': 0.25, 'T': 0.25 } )

	# Create the insert states
	i0 = State( i_d, name="I0" )
	i1 = State( i_d, name="I1" )
	i2 = State( i_d, name="I2" )
	i3 = State( i_d, name="I3" )

	# Create the match states
	m1 = State( DiscreteDistribution({ "A": 0.95, 'C': 0.01, 'G': 0.01, 'T': 0.02 }) , name="M1" )
	m2 = State( DiscreteDistribution({ "A": 0.003, 'C': 0.99, 'G': 0.003, 'T': 0.004 }) , name="M2" )
	m3 = State( DiscreteDistribution({ "A": 0.01, 'C': 0.01, 'G': 0.01, 'T': 0.97 }) , name="M3" )

	# Create the delete states
	d1 = State( None, name="D1" )
	d2 = State( None, name="D2" )
	d3 = State( None, name="D3" )

	# Add all the states to the model
	model.add_states( [i0, i1, i2, i3, m1, m2, m3, d1, d2, d3 ] )

	# Create transitions from match states
	model.add_transitions( model.start, [m1, i0], [0.9, 0.1] )

	model.add_transitions( m1, [m2, i1, d2], [0.9, 0.05, 0.05] )
	model.add_transitions( m2, [m3, i2, d3], [0.9, 0.05, 0.05] )
	model.add_transitions( m3, [model.end, i3], [0.9, 0.1] )

	# Create transitions from insert states
	model.add_transitions( i0, [i0, d1, m1], [0.7, 0.15, 0.15] )
	model.add_transitions( i1, [i1, d2, m2], [0.7, 0.15, 0.15] )
	model.add_transitions( i2, [i2, d3, m3], [0.7, 0.15, 0.15] )
	model.add_transitions( [i3, i3], [i3, model.end], [0.85, 0.15] )

	# Create transitions from delete states
	model.add_transitions( d1, [d2, i1, m2], [0.15, 0.15, 0.70] )
	model.add_transitions( [d2, d2, d2, d3, d3], [d3, i2, m3, i3, model.end],
		[0.15, 0.15, 0.70, 0.30, 0.70 ] )

	# Call bake to finalize the structure of the model.
	model.bake()


def tied_edge_setup():
	'''
	Build a model that we want to use to test sequences. This model has
	tied edges.
	'''

	random.seed(0)

	global model
	model = HiddenMarkovModel( "Global Alignment")

	# Define the distribution for insertions
	i_d = DiscreteDistribution( { 'A': 0.25, 'C': 0.25, 'G': 0.25, 'T': 0.25 } )

	# Create the insert states
	i0 = State( i_d, name="I0" )
	i1 = State( i_d, name="I1" )
	i2 = State( i_d, name="I2" )
	i3 = State( i_d, name="I3" )

	# Create the match states
	m1 = State( DiscreteDistribution({ "A": 0.95, 'C': 0.01, 'G': 0.01, 'T': 0.02 }) , name="M1" )
	m2 = State( DiscreteDistribution({ "A": 0.003, 'C': 0.99, 'G': 0.003, 'T': 0.004 }) , name="M2" )
	m3 = State( DiscreteDistribution({ "A": 0.01, 'C': 0.01, 'G': 0.01, 'T': 0.97 }) , name="M3" )

	# Create the delete states
	d1 = State( None, name="D1" )
	d2 = State( None, name="D2" )
	d3 = State( None, name="D3" )

	# Add all the states to the model
	model.add_states( [i0, i1, i2, i3, m1, m2, m3, d1, d2, d3 ] )

	# Create transitions from match states
	model.add_transition( model.start, m1, 0.9 )
	model.add_transition( model.start, i0, 0.1 )
	model.add_transition( m1, m2, 0.9 )
	model.add_transition( m1, i1, 0.05 )
	model.add_transition( m1, d2, 0.05 )
	model.add_transition( m2, m3, 0.9 )
	model.add_transition( m2, i2, 0.05 )
	model.add_transition( m2, d3, 0.05 )
	model.add_transition( m3, model.end, 0.9 )
	model.add_transition( m3, i3, 0.1 )

	# Create transitions from insert states
	model.add_transition( i0, i0, 0.70, group="i_a" )
	model.add_transition( i0, d1, 0.15, group="i_b" )
	model.add_transition( i0, m1, 0.15, group="i_c" )

	model.add_transition( i1, i1, 0.70, group="i_a" )
	model.add_transition( i1, d2, 0.15, group="i_b" )
	model.add_transition( i1, m2, 0.15, group="i_c" )

	model.add_transition( i2, i2, 0.70, group="i_a" )
	model.add_transition( i2, d3, 0.15, group="i_b" )
	model.add_transition( i2, m3, 0.15, group="i_c" )

	model.add_transition( i3, i3, 0.85, group="i_a" )
	model.add_transition( i3, model.end, 0.15 )

	# Create transitions from delete states
	model.add_transition( d1, d2, 0.15, group="d_a" )
	model.add_transition( d1, i1, 0.15, group="d_b" )
	model.add_transition( d1, m2, 0.70, group="d_c" )

	model.add_transition( d2, d3, 0.15, group="d_a" )
	model.add_transition( d2, i2, 0.15, group="d_b" )
	model.add_transition( d2, m3, 0.70, group="d_c" )

	model.add_transition( d3, i3, 0.30 )
	model.add_transition( d3, model.end, 0.70 )

	# Call bake to finalize the structure of the model.
	model.bake()


def teardown():
	'''
	Remove the model at the end of the unit testing. Since it is stored in a
	global variance, simply delete it.
	'''

	pass


@with_setup( setup, teardown )
def test_same_length_viterbi():
	scores = [ -0.5132449003570658, -11.048101241343396, -9.125519674022627,
		-5.0879558788604475 ]
	sequences = [ list(x) for x in [ 'ACT', 'GGC', 'GAT', 'ACC' ] ]

	for seq, score in zip( sequences, scores ):
		assert_almost_equal( model.viterbi( seq )[0], score )

	assert_raises( ValueError, model.viterbi, list('XXX') )


@with_setup( setup, teardown )
def test_variable_length_viterbi():
	scores = [ -5.406181012423981, -10.88681993576597, -3.6244718790494277,
	-3.644880750680635, -10.674332964640293, -10.393824835172445,
	-8.67126440174503, -16.903451796110275, -16.451699654050792 ]
	sequences = [ list(x) for x in ('A', 'GA', 'AC', 'AT', 'ATCC',
		'ACGTG', 'ATTT', 'TACCCTC', 'TGTCAACACT') ]

	for seq, score in zip( sequences, scores ):
		assert_almost_equal( model.viterbi( seq )[0], score )


@with_setup( setup, teardown )
def test_log_probability():
	scores = [ -5.3931, -0.5052, -11.8478, -14.3482 ]
	sequences = [ list(x) for x in ( 'A', 'ACT', 'GGCA', 'TACCTGT' ) ]

	for seq, score in zip( sequences, scores ):
		assert_equal( round( model.log_probability( seq ), 4 ), score )


@with_setup( setup, teardown )
def test_posterior_transitions():
	a_scores = [ 0.0, 0.0021, 0.2017, 1.5105 ]
	b_scores = [ 0.013, 0.0036, 1.9836, 2.145 ]
	c_scores = [ 0.013, 0.0035, 0.817, 0.477 ]
	d_scores = [ 1.0, 0.0023, 0.2636, 0.3682 ]
	t_scores = [ 4.013, 4.0083, 6.457, 8.9812 ]
	sequences = [ list(x) for x in ( 'A', 'ACT', 'GGCA', 'TACCTGT' ) ]

	indices = { state.name: i for i, state in enumerate( model.states ) }
	i, j, k, l = indices['I2'], indices['I0'], indices['D1'], indices['D2']

	scores = zip( sequences, a_scores, b_scores, c_scores, d_scores, t_scores )
	for seq, a, b, c, d, t in scores:
		trans, ems = model.forward_backward( seq )

		assert_equal( round( trans[i].sum(), 4 ), a )
		assert_equal( round( trans[j].sum(), 4 ), b )
		assert_equal( round( trans[k].sum(), 4 ), c )
		assert_equal( round( trans[l].sum(), 4 ), d )
		assert_equal( round( trans.sum(), 4 ), t )


@with_setup( setup, teardown )
def test_posterior_transitions_w_training():
	sequences = [ list(x) for x in ( 'A', 'ACT', 'GGCA', 'TACCTGT' ) ]
	indices = { state.name: i for i, state in enumerate( model.states ) }

	transitions = model.dense_transition_matrix()
	i0, i1, i2 = indices['I0'], indices['I1'], indices['I2']
	d1, d2, d3 = indices['D1'], indices['D2'], indices['D3']
	m1, m2, m3 = indices['M1'], indices['M2'], indices['M3']

	assert_equal( transitions[d1, i1], transitions[d2, i2] )
	assert_equal( transitions[i0, i0], transitions[i1, i1] )
	assert_equal( transitions[i0, i0], transitions[i2, i2] )
	assert_equal( transitions[i0, m1], transitions[i1, m2] )
	assert_equal( transitions[d1, d2], transitions[d2, d3] )
	assert_equal( transitions[i0, d1], transitions[i1, d2] )
	assert_equal( transitions[i0, d1], transitions[i2, d3] )

	model.fit( sequences, verbose=False )
	transitions = model.dense_transition_matrix()

	assert_not_equal( transitions[d1, i1], transitions[d2, i2] )
	assert_not_equal( transitions[i0, m1], transitions[i1, m2] )
	assert_not_equal( transitions[d1, d2], transitions[d2, d3] )
	assert_not_equal( transitions[i0, d1], transitions[i1, d2] )
	assert_not_equal( transitions[i0, d1], transitions[i2, d3] )


@with_setup( setup, teardown )
def test_posterior_transitions_w_vtraining():
	sequences = [ list(x) for x in ( 'A', 'ACT', 'GGCA', 'TACCTGT' ) ]
	indices = { state.name: i for i, state in enumerate( model.states ) }

	transitions = model.dense_transition_matrix()
	i0, i1, i2, i3 = indices['I0'], indices['I1'], indices['I2'], indices['I3']
	d1, d2, d3 = indices['D1'], indices['D2'], indices['D3']
	m1, m2, m3 = indices['M1'], indices['M2'], indices['M3']

	assert_equal( transitions[d1, i1], transitions[d2, i2] )
	assert_equal( transitions[i0, i0], transitions[i1, i1] )
	assert_equal( transitions[i0, i0], transitions[i2, i2] )
	assert_equal( transitions[i0, m1], transitions[i1, m2] )
	assert_equal( transitions[d1, d2], transitions[d2, d3] )
	assert_equal( transitions[i0, d1], transitions[i1, d2] )
	assert_equal( transitions[i0, d1], transitions[i2, d3] )

	model.fit( sequences, verbose=False, algorithm='viterbi' )
	transitions = model.dense_transition_matrix()

	assert_not_equal( transitions[i0, i0], transitions[i1, i1] )
	assert_not_equal( transitions[d1, d2], transitions[d2, d3] )
	assert_not_equal( transitions[i0, d1], transitions[i1, d2] )
	assert_not_equal( transitions[i0, d1], transitions[i2, d3] )


@with_setup( tied_edge_setup, teardown )
def test_posterior_transitions_w_tied_training():
	sequences = [ list(x) for x in ( 'A', 'ACT', 'GGCA', 'TACCTGT' ) ]
	indices = { state.name: i for i, state in enumerate( model.states ) }

	transitions = model.dense_transition_matrix()
	i0, i1, i2, i3 = indices['I0'], indices['I1'], indices['I2'], indices['I3']
	d1, d2, d3 = indices['D1'], indices['D2'], indices['D3']
	m1, m2, m3 = indices['M1'], indices['M2'], indices['M3']

	assert_equal( transitions[d1, i1], transitions[d2, i2] )
	assert_equal( transitions[i0, i0], transitions[i1, i1] )
	assert_equal( transitions[i0, i0], transitions[i2, i2] )
	assert_equal( transitions[i0, m1], transitions[i1, m2] )
	assert_equal( transitions[d1, d2], transitions[d2, d3] )
	assert_equal( transitions[i0, d1], transitions[i1, d2] )
	assert_equal( transitions[i0, d1], transitions[i2, d3] )

	model.fit( sequences, verbose=False )
	transitions = model.dense_transition_matrix()

	assert_equal( transitions[i0, i0], transitions[i1, i1] )
	assert_equal( transitions[d1, d2], transitions[d2, d3] )
	assert_equal( transitions[i0, d1], transitions[i1, d2] )
	assert_equal( transitions[i0, d1], transitions[i2, d3] )


@with_setup( tied_edge_setup, teardown )
def test_posterior_transitions_w_tied_vtraining():
	sequences = [ list(x) for x in ( 'A', 'ACT', 'GGCA', 'TACCTGT' ) ]
	indices = { state.name: i for i, state in enumerate( model.states ) }

	transitions = model.dense_transition_matrix()
	i0, i1, i2 = indices['I0'], indices['I1'], indices['I2']
	d1, d2, d3 = indices['D1'], indices['D2'], indices['D3']
	m1, m2, m3 = indices['M1'], indices['M2'], indices['M3']

	assert_equal( transitions[d1, i1], transitions[d2, i2] )
	assert_equal( transitions[i0, i0], transitions[i1, i1] )
	assert_equal( transitions[i0, i0], transitions[i2, i2] )
	assert_equal( transitions[i0, m1], transitions[i1, m2] )
	assert_equal( transitions[d1, d2], transitions[d2, d3] )
	assert_equal( transitions[i0, d1], transitions[i1, d2] )
	assert_equal( transitions[i0, d1], transitions[i2, d3] )

	model.fit( sequences, verbose=False, algorithm='viterbi' )
	transitions = model.dense_transition_matrix()

	assert_equal( transitions[d1, i1], transitions[d2, i2] )
	assert_equal( transitions[i0, i0], transitions[i1, i1] )
	assert_equal( transitions[i0, i0], transitions[i2, i2] )
	assert_equal( transitions[i0, m1], transitions[i1, m2] )
	assert_equal( transitions[d1, d2], transitions[d2, d3] )
	assert_equal( transitions[i0, d1], transitions[i1, d2] )
	assert_equal( transitions[i0, d1], transitions[i2, d3] )


@with_setup( setup, teardown )
def test_posterior_emissions():
	a_scores = [ 0.987, 0.9965, 0.183, 0.523 ]
	b_scores = [ 0.0, 0.9977, 0.7364, 0.6318 ]
	c_scores = [ 0.0, 0.9975, 0.6237, 0.8641 ]
	d_scores = [ 0.0, 0.0021, 0.2017, 1.5105 ]
	sequences = [ list(x) for x in ( 'A', 'ACT', 'GGCA', 'TACCTGT' ) ]

	indices = { state.name: i for i, state in enumerate( model.states ) }
	i, j, k, l = indices['M1'], indices['M2'], indices['M3'], indices['I2']

	for seq, a, b, c, d in zip( sequences, a_scores, b_scores, c_scores, d_scores ):
		trans, ems = model.forward_backward( seq )
		ems = np.exp( ems )

		assert_equal( round( ems[:,i].sum(), 4 ), a )
		assert_equal( round( ems[:,j].sum(), 4 ), b )
		assert_equal( round( ems[:,k].sum(), 4 ), c )
		assert_equal( round( ems[:,l].sum(), 4 ), d )
		assert_equal( round( ems.sum() ), len( seq ) )


@with_setup( multitransition_setup, teardown )
def test_posterior_emissions_w_multitransition_setup():
	a_scores = [ 0.987, 0.9965, 0.183, 0.523 ]
	b_scores = [ 0.0, 0.9977, 0.7364, 0.6318 ]
	c_scores = [ 0.0, 0.9975, 0.6237, 0.8641 ]
	d_scores = [ 0.0, 0.0021, 0.2017, 1.5105 ]
	sequences = [ list(x) for x in ( 'A', 'ACT', 'GGCA', 'TACCTGT' ) ]

	indices = { state.name: i for i, state in enumerate( model.states ) }
	i, j, k, l = indices['M1'], indices['M2'], indices['M3'], indices['I2']

	for seq, a, b, c, d in zip( sequences, a_scores, b_scores, c_scores, d_scores ):
		trans, ems = model.forward_backward( seq )
		ems = np.exp( ems )

		assert_equal( round( ems[:,i].sum(), 4 ), a )
		assert_equal( round( ems[:,j].sum(), 4 ), b )
		assert_equal( round( ems[:,k].sum(), 4 ), c )
		assert_equal( round( ems[:,l].sum(), 4 ), d )
		assert_equal( round( ems.sum() ), len( seq ) )


@with_setup( tied_edge_setup, teardown )
def test_posterior_emissions_w_tied_edge_setup():
	a_scores = [ 0.987, 0.9965, 0.183, 0.523 ]
	b_scores = [ 0.0, 0.9977, 0.7364, 0.6318 ]
	c_scores = [ 0.0, 0.9975, 0.6237, 0.8641 ]
	d_scores = [ 0.0, 0.0021, 0.2017, 1.5105 ]
	sequences = [ list(x) for x in ( 'A', 'ACT', 'GGCA', 'TACCTGT' ) ]

	indices = { state.name: i for i, state in enumerate( model.states ) }
	i, j, k, l = indices['M1'], indices['M2'], indices['M3'], indices['I2']

	for seq, a, b, c, d in zip( sequences, a_scores, b_scores, c_scores, d_scores ):
		trans, ems = model.forward_backward( seq )
		ems = np.exp( ems )

		assert_equal( round( ems[:,i].sum(), 4 ), a )
		assert_equal( round( ems[:,j].sum(), 4 ), b )
		assert_equal( round( ems[:,k].sum(), 4 ), c )
		assert_equal( round( ems[:,l].sum(), 4 ), d )
		assert_equal( round( ems.sum() ), len( seq ) )


@with_setup( setup, teardown )
def test_properties():
	assert_equal( model.edge_count(), 29 )
	assert_equal( model.state_count(), 12 )
	assert_equal( model.name, "Global Alignment" )


@with_setup( setup, teardown )
def test_to_json():
	b = json.loads(model.to_json())

	assert_equal(b['name'], 'Global Alignment')
	assert_equal(len(b['edges']), 29)
	assert_equal(len(b['states']), 12)
	assert_equal(b['silent_index'], 7)


@with_setup( setup, teardown )
def test_from_json():
	hmm = HiddenMarkovModel.from_json( model.to_json() )

	assert_equal(hmm.edge_count(), 29)
	assert_equal(hmm.state_count(), 12)
	assert_equal(hmm.name, "Global Alignment")

from __future__ import  (division, print_function)

from pomegranate import *
from nose.tools import with_setup
import random
import numpy as np
import time

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
	global m1, m2, m3
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

def teardown():
	'''
	Remove the model at the end of the unit testing. Since it is stored in a
	global variance, simply delete it.
	'''

	pass

@with_setup( setup, teardown )
def test_viterbi_train():
	'''
	Test the model using various parameter settings for Viterbi training.
	'''

	seqs = [ list(x) for x in [ 'ACT', 'ACT', 'ACC', 'ACTC', 'ACT', 'ACT', 'CCT', 
		'CCC', 'AAT', 'CT', 'AT', 'CT', 'CT', 'CT', 'CT', 'CT', 'CT', 
		'ACT', 'ACT', 'CT', 'ACT', 'CT', 'CT', 'CT', 'CT' ] ]

	total_improvement = model.train( seqs, 
									 algorithm='viterbi', 
									 verbose=False, 
									 use_pseudocount=True )

	assert round( total_improvement, 4 ) == 0.4654

@with_setup( setup, teardown )
def test_viterbi_train_no_pseudocount():
	'''
	Test the model using various parameter settings for Viterbi training.
	'''

	seqs = [ list(x) for x in [ 'ACT', 'ACT', 'ACC', 'ACTC', 'ACT', 'ACT', 'CCT', 
		'CCC', 'AAT', 'CT', 'AT', 'CT', 'CT', 'CT', 'CT', 'CT', 'CT', 
		'ACT', 'ACT', 'CT', 'ACT', 'CT', 'CT', 'CT', 'CT' ] ]

	total_improvement = model.train( seqs, 
									 algorithm='viterbi', 
									 verbose=False, 
									 use_pseudocount=False )

	assert round( total_improvement, 4 ) == 0.577

@with_setup( setup, teardown )
def test_viterbi_train_w_pseudocount():
	'''
	Test the model using various parameter settings for Viterbi training.
	'''

	seqs = [ list(x) for x in [ 'ACT', 'ACT', 'ACC', 'ACTC', 'ACT', 'ACT', 'CCT', 
		'CCC', 'AAT', 'CT', 'AT', 'CT', 'CT', 'CT', 'CT', 'CT', 'CT', 
		'ACT', 'ACT', 'CT', 'ACT', 'CT', 'CT', 'CT', 'CT' ] ]

	total_improvement = model.train( seqs, 
									 algorithm='viterbi', 
									 verbose=False, 
									 transition_pseudocount=1. )

	assert round( total_improvement, 4 ) == 0.2124

@with_setup( setup, teardown )
def test_viterbi_train_w_pseudocount_priors():
	'''
	Test the model using various parameter settings for Viterbi training.
	'''

	seqs = [ list(x) for x in [ 'ACT', 'ACT', 'ACC', 'ACTC', 'ACT', 'ACT', 'CCT', 
		'CCC', 'AAT', 'CT', 'AT', 'CT', 'CT', 'CT', 'CT', 'CT', 'CT', 
		'ACT', 'ACT', 'CT', 'ACT', 'CT', 'CT', 'CT', 'CT' ] ]

	total_improvement = model.train( seqs, 
									 algorithm='viterbi', 
									 verbose=False, 
									 transition_pseudocount=0.278,
									 use_pseudocount=True )

	assert round( total_improvement, 4 ) == 0.3635

@with_setup( setup, teardown )
def test_viterbi_train_w_inertia():
	'''
	Test the model using various parameter settings for Viterbi training.
	'''

	seqs = [ list(x) for x in [ 'ACT', 'ACT', 'ACC', 'ACTC', 'ACT', 'ACT', 'CCT', 
		'CCC', 'AAT', 'CT', 'AT', 'CT', 'CT', 'CT', 'CT', 'CT', 'CT', 
		'ACT', 'ACT', 'CT', 'ACT', 'CT', 'CT', 'CT', 'CT' ] ]

	total_improvement = model.train( seqs, 
									 algorithm='viterbi', 
									 verbose=False, 
									 edge_inertia=0.193 )

	assert round( total_improvement, 4 ) == 0.2808

@with_setup( setup, teardown )
def test_viterbi_train_w_inertia2():
	'''
	Test the model using various parameter settings for Viterbi training.
	'''

	seqs = [ list(x) for x in [ 'ACT', 'ACT', 'ACC', 'ACTC', 'ACT', 'ACT', 'CCT', 
		'CCC', 'AAT', 'CT', 'AT', 'CT', 'CT', 'CT', 'CT', 'CT', 'CT', 
		'ACT', 'ACT', 'CT', 'ACT', 'CT', 'CT', 'CT', 'CT' ] ]

	total_improvement = model.train( seqs, 
									 algorithm='viterbi', 
									 verbose=False, 
									 edge_inertia=0.82 )

	assert round( total_improvement, 4 ) == -0.1789

@with_setup( setup, teardown )
def test_viterbi_train_w_pseudocount_inertia():
	'''
	Test the model using various parameter settings for Viterbi training.
	'''

	seqs = [ list(x) for x in [ 'ACT', 'ACT', 'ACC', 'ACTC', 'ACT', 'ACT', 'CCT', 
		'CCC', 'AAT', 'CT', 'AT', 'CT', 'CT', 'CT', 'CT', 'CT', 'CT', 
		'ACT', 'ACT', 'CT', 'ACT', 'CT', 'CT', 'CT', 'CT' ] ]

	total_improvement = model.train( seqs, 
									 algorithm='viterbi', 
									 verbose=False, 
									 edge_inertia=0.23,
									 use_pseudocount=True )
	print( round( total_improvement, 4 ) )
	assert round( total_improvement, 4 ) == 0.1573

@with_setup( setup, teardown )
def test_bw_train():
	'''
	Test the model using various parameter settings for Baum-Welch training.
	'''

	seqs = [ list(x) for x in [ 'ACT', 'ACT', 'ACC', 'ACTC', 'ACT', 'ACT', 'CCT', 
		'CCC', 'AAT', 'CT', 'AT', 'CT', 'CT', 'CT', 'CT', 'CT', 'CT', 
		'ACT', 'ACT', 'CT', 'ACT', 'CT', 'CT', 'CT', 'CT' ] ]

	total_improvement = model.train( seqs, 
									 algorithm='baum-welch', 
									 verbose=False, 
									 use_pseudocount=True,
									 max_iterations=5 )
	
	assert round( total_improvement, 4 ) == 0.4398

@with_setup( setup, teardown )
def test_bw_train_no_pseudocount():
	'''
	Test the model using various parameter settings for Baum-Welch training.
	'''

	seqs = [ list(x) for x in [ 'ACT', 'ACT', 'ACC', 'ACTC', 'ACT', 'ACT', 'CCT', 
		'CCC', 'AAT', 'CT', 'AT', 'CT', 'CT', 'CT', 'CT', 'CT', 'CT', 
		'ACT', 'ACT', 'CT', 'ACT', 'CT', 'CT', 'CT', 'CT' ] ]

	total_improvement = model.train( seqs, 
									 algorithm='baum-welch', 
									 verbose=False, 
									 use_pseudocount=False,
									 max_iterations=5 )
	
	assert round( total_improvement, 4 ) == 0.5718

@with_setup( setup, teardown )
def test_bw_train_w_pseudocount():
	'''
	Test the model using various parameter settings for Baum-Welch training.
	'''

	seqs = [ list(x) for x in [ 'ACT', 'ACT', 'ACC', 'ACTC', 'ACT', 'ACT', 'CCT', 
		'CCC', 'AAT', 'CT', 'AT', 'CT', 'CT', 'CT', 'CT', 'CT', 'CT', 
		'ACT', 'ACT', 'CT', 'ACT', 'CT', 'CT', 'CT', 'CT' ] ]

	total_improvement = model.train( seqs, 
									 algorithm='baum-welch', 
									 verbose=False, 
									 transition_pseudocount=0.123,
									 max_iterations=5 )
	
	assert round( total_improvement, 4 ) == 0.521

@with_setup( setup, teardown )
def test_bw_train_w_pseudocount_priors():
	'''
	Test the model using various parameter settings for Baum-Welch training.
	'''

	seqs = [ list(x) for x in [ 'ACT', 'ACT', 'ACC', 'ACTC', 'ACT', 'ACT', 'CCT', 
		'CCC', 'AAT', 'CT', 'AT', 'CT', 'CT', 'CT', 'CT', 'CT', 'CT', 
		'ACT', 'ACT', 'CT', 'ACT', 'CT', 'CT', 'CT', 'CT' ] ]

	total_improvement = model.train( seqs, 
									 algorithm='baum-welch', 
									 verbose=False, 
									 transition_pseudocount=0.278,
									 use_pseudocount=True,
									 max_iterations=5 )
	
	assert round( total_improvement, 4 ) == 0.3277

@with_setup( setup, teardown )
def test_bw_train_w_inertia():
	'''
	Test the model using various parameter settings for Baum-Welch training.
	'''

	seqs = [ list(x) for x in [ 'ACT', 'ACT', 'ACC', 'ACTC', 'ACT', 'ACT', 'CCT', 
		'CCC', 'AAT', 'CT', 'AT', 'CT', 'CT', 'CT', 'CT', 'CT', 'CT', 
		'ACT', 'ACT', 'CT', 'ACT', 'CT', 'CT', 'CT', 'CT' ] ]

	total_improvement = model.train( seqs, 
									 algorithm='baum-welch', 
									 verbose=False, 
									 edge_inertia=0.193,
									 max_iterations=5 )
	
	assert round( total_improvement, 4 ) == 0.5398

@with_setup( setup, teardown )
def test_bw_train_w_inertia2():
	'''
	Test the model using various parameter settings for Baum-Welch training.
	'''

	seqs = [ list(x) for x in [ 'ACT', 'ACT', 'ACC', 'ACTC', 'ACT', 'ACT', 'CCT', 
		'CCC', 'AAT', 'CT', 'AT', 'CT', 'CT', 'CT', 'CT', 'CT', 'CT', 
		'ACT', 'ACT', 'CT', 'ACT', 'CT', 'CT', 'CT', 'CT' ] ]

	total_improvement = model.train( seqs, 
									 algorithm='baum-welch', 
									 verbose=False, 
									 edge_inertia=0.82,
									 max_iterations=5 )
	
	assert round( total_improvement, 4 ) == -0.3285

@with_setup( setup, teardown )
def test_bw_train_w_pseudocount_inertia():
	'''
	Test the model using various parameter settings for Baum-Welch training.
	'''

	seqs = [ list(x) for x in [ 'ACT', 'ACT', 'ACC', 'ACTC', 'ACT', 'ACT', 'CCT', 
		'CCC', 'AAT', 'CT', 'AT', 'CT', 'CT', 'CT', 'CT', 'CT', 'CT', 
		'ACT', 'ACT', 'CT', 'ACT', 'CT', 'CT', 'CT', 'CT' ] ]

	total_improvement = model.train( seqs, 
									 algorithm='baum-welch', 
									 verbose=False, 
									 edge_inertia=0.02,
									 use_pseudocount=True,
									 max_iterations=5 )
	
	assert round( total_improvement, 4 ) == 0.4363

@with_setup( setup, teardown )
def test_bw_train_w_frozen_distributions():
	'''
	Test the model using various parameter settings for Baum-Welch training.
	'''

	seqs = [ list(x) for x in [ 'ACT', 'ACT', 'ACC', 'ACTC', 'ACT', 'ACT', 'CCT', 
		'CCC', 'AAT', 'CT', 'AT', 'CT', 'CT', 'CT', 'CT', 'CT', 'CT', 
		'ACT', 'ACT', 'CT', 'ACT', 'CT', 'CT', 'CT', 'CT' ] ]

	total_improvement = model.train( seqs, 
									 algorithm='baum-welch', 
									 verbose=False, 
									 distribution_inertia=1.00,
									 max_iterations=5 )

	assert round( total_improvement, 4 ) == -0.1698

@with_setup( setup, teardown )
def test_bw_train_w_frozen_edges():
	'''
	Test the model using various parameter settings for Baum-Welch training.
	'''

	seqs = [ list(x) for x in [ 'ACT', 'ACT', 'ACC', 'ACTC', 'ACT', 'ACT', 'CCT', 
		'CCC', 'AAT', 'CT', 'AT', 'CT', 'CT', 'CT', 'CT', 'CT', 'CT', 
		'ACT', 'ACT', 'CT', 'ACT', 'CT', 'CT', 'CT', 'CT' ] ]

	total_improvement = model.train( seqs, 
									 algorithm='baum-welch', 
									 verbose=False, 
									 edge_inertia=1.00,
									 max_iterations=5 )

	assert round( total_improvement, 4 ) == -0.3488

@with_setup( setup, teardown )
def test_bw_train_w_edge_a_distribution_inertia():
	'''
	Test the model using various parameter settings for Baum-Welch training.
	'''

	seqs = [ list(x) for x in [ 'ACT', 'ACT', 'ACC', 'ACTC', 'ACT', 'ACT', 'CCT', 
		'CCC', 'AAT', 'CT', 'AT', 'CT', 'CT', 'CT', 'CT', 'CT', 'CT', 
		'ACT', 'ACT', 'CT', 'ACT', 'CT', 'CT', 'CT', 'CT' ] ]

	total_improvement = model.train( seqs, 
									 algorithm='baum-welch', 
									 verbose=False, 
									 edge_inertia=0.5,
									 distribution_inertia=0.5,
									 max_iterations=5 )

	assert round( total_improvement, 4 ) == -0.1702

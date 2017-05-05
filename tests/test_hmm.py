from __future__ import (division)

from pomegranate import *
from pomegranate.parallel import log_probability
from nose.tools import with_setup
from nose.tools import assert_almost_equal
from nose.tools import assert_equal
from nose.tools import assert_not_equal
from nose.tools import assert_less_equal
from nose.tools import assert_raises
from nose.tools import assert_greater
import pickle
import random
import numpy as np
import time

np.random.seed(0)
random.seed(0)

def setup():
	'''
	Build a model that we want to use to test sequences. This model will
	be somewhat complicated, in order to extensively test YAHMM. This will be
	a three state global sequence alignment HMM. The HMM models a reference of
	'ACT', with pseudocounts to allow for slight deviations from this
	reference.
	'''

	global model
	global m1, m2, m3
	model = HiddenMarkovModel("Global Alignment")

	# Define the distribution for insertions
	i_d = DiscreteDistribution({ 'A': 0.25, 'C': 0.25, 'G': 0.25, 'T': 0.25 })

	# Create the insert states
	i0 = State(i_d, name="I0")
	i1 = State(i_d, name="I1")
	i2 = State(i_d, name="I2")
	i3 = State(i_d, name="I3")

	# Create the match states
	m1 = State(DiscreteDistribution({ "A": 0.95, 'C': 0.01, 'G': 0.01, 'T': 0.02 }) , name="M1")
	m2 = State(DiscreteDistribution({ "A": 0.003, 'C': 0.99, 'G': 0.003, 'T': 0.004 }) , name="M2")
	m3 = State(DiscreteDistribution({ "A": 0.01, 'C': 0.01, 'G': 0.01, 'T': 0.97 }) , name="M3")

	# Create the delete states
	d1 = State(None, name="D1")
	d2 = State(None, name="D2")
	d3 = State(None, name="D3")

	# Add all the states to the model
	model.add_states([i0, i1, i2, i3, m1, m2, m3, d1, d2, d3])

	# Create transitions from match states
	model.add_transition(model.start, m1, 0.9)
	model.add_transition(model.start, i0, 0.1)
	model.add_transition(m1, m2, 0.9)
	model.add_transition(m1, i1, 0.05)
	model.add_transition(m1, d2, 0.05)
	model.add_transition(m2, m3, 0.9)
	model.add_transition(m2, i2, 0.05)
	model.add_transition(m2, d3, 0.05)
	model.add_transition(m3, model.end, 0.9)
	model.add_transition(m3, i3, 0.1)

	# Create transitions from insert states
	model.add_transition(i0, i0, 0.70)
	model.add_transition(i0, d1, 0.15)
	model.add_transition(i0, m1, 0.15)

	model.add_transition(i1, i1, 0.70)
	model.add_transition(i1, d2, 0.15)
	model.add_transition(i1, m2, 0.15)

	model.add_transition(i2, i2, 0.70)
	model.add_transition(i2, d3, 0.15)
	model.add_transition(i2, m3, 0.15)

	model.add_transition(i3, i3, 0.85)
	model.add_transition(i3, model.end, 0.15)

	# Create transitions from delete states
	model.add_transition(d1, d2, 0.15)
	model.add_transition(d1, i1, 0.15)
	model.add_transition(d1, m2, 0.70) 

	model.add_transition(d2, d3, 0.15)
	model.add_transition(d2, i2, 0.15)
	model.add_transition(d2, m3, 0.70)

	model.add_transition(d3, i3, 0.30)
	model.add_transition(d3, model.end, 0.70)

	# Call bake to finalize the structure of the model.
	model.bake()

def setup_multivariate_discrete():
	'''
	Build a model that we want to use to test sequences. This model will
	be somewhat complicated, in order to extensively test YAHMM. This will be
	a three state global sequence alignment HMM. The HMM models a reference of
	'ACT', with pseudocounts to allow for slight deviations from this
	reference.
	'''

	global model
	global m1, m2, m3
	model = HiddenMarkovModel("Global Alignment")

	i1 = DiscreteDistribution({'A': 0.25, 'C': 0.25, 'G': 0.25, 'T': 0.25})
	i2 = DiscreteDistribution({'A': 0.25, 'C': 0.25, 'G': 0.25, 'T': 0.25})
	# Define the distribution for insertions
	i_d = IndependentComponentsDistribution([i1, i2])

	# Create the insert states
	i0 = State(i_d, name="I0")
	i1 = State(i_d, name="I1")
	i2 = State(i_d, name="I2")
	i3 = State(i_d, name="I3")

	# Create the match states
	m11 = DiscreteDistribution({ "A": 0.95, 'C': 0.01, 'G': 0.01, 'T': 0.02 })
	m12 = DiscreteDistribution({ "A": 0.92, 'C': 0.02, 'G': 0.02, 'T': 0.03 })

	m21 = DiscreteDistribution({ "A": 0.005, 'C': 0.96, 'G': 0.005, 'T': 0.003 })
	m22 = DiscreteDistribution({ "A": 0.003, 'C': 0.99, 'G': 0.003, 'T': 0.004 })

	m31 = DiscreteDistribution({ "A": 0.01, 'C': 0.01, 'G': 0.01, 'T': 0.97 })
	m32 = DiscreteDistribution({ "A": 0.05, 'C': 0.03, 'G': 0.02, 'T': 0.90 })

	m1 = State(IndependentComponentsDistribution([m11, m12]), name="M1")
	m2 = State(IndependentComponentsDistribution([m21, m22]), name="M2")
	m3 = State(IndependentComponentsDistribution([m31, m32]), name="M3")

	# Create the delete states
	d1 = State(None, name="D1")
	d2 = State(None, name="D2")
	d3 = State(None, name="D3")

	# Add all the states to the model
	model.add_states([i0, i1, i2, i3, m1, m2, m3, d1, d2, d3])

	# Create transitions from match states
	model.add_transition(model.start, m1, 0.9)
	model.add_transition(model.start, i0, 0.1)
	model.add_transition(m1, m2, 0.9)
	model.add_transition(m1, i1, 0.05)
	model.add_transition(m1, d2, 0.05)
	model.add_transition(m2, m3, 0.9)
	model.add_transition(m2, i2, 0.05)
	model.add_transition(m2, d3, 0.05)
	model.add_transition(m3, model.end, 0.9)
	model.add_transition(m3, i3, 0.1)

	# Create transitions from insert states
	model.add_transition(i0, i0, 0.70)
	model.add_transition(i0, d1, 0.15)
	model.add_transition(i0, m1, 0.15)

	model.add_transition(i1, i1, 0.70)
	model.add_transition(i1, d2, 0.15)
	model.add_transition(i1, m2, 0.15)

	model.add_transition(i2, i2, 0.70)
	model.add_transition(i2, d3, 0.15)
	model.add_transition(i2, m3, 0.15)

	model.add_transition(i3, i3, 0.85)
	model.add_transition(i3, model.end, 0.15)

	# Create transitions from delete states
	model.add_transition(d1, d2, 0.15)
	model.add_transition(d1, i1, 0.15)
	model.add_transition(d1, m2, 0.70) 

	model.add_transition(d2, d3, 0.15)
	model.add_transition(d2, i2, 0.15)
	model.add_transition(d2, m3, 0.70)

	model.add_transition(d3, i3, 0.30)
	model.add_transition(d3, model.end, 0.70)

	# Call bake to finalize the structure of the model.
	model.bake()

def setup_multivariate_gaussian():
	'''
	Build a model that we want to use to test sequences. This model will
	be somewhat complicated, in order to extensively test YAHMM. This will be
	a three state global sequence alignment HMM. The HMM models a reference of
	'ACT', with pseudocounts to allow for slight deviations from this
	reference.
	'''

	global model
	global m1, m2, m3
	model = HiddenMarkovModel("Global Alignment")

	i1 = UniformDistribution(-20, 20)
	i2 = UniformDistribution(-20, 20)
	# Define the distribution for insertions
	i_d = IndependentComponentsDistribution([i1, i2])

	# Create the insert states
	i0 = State(i_d, name="I0")
	i1 = State(i_d, name="I1")
	i2 = State(i_d, name="I2")
	i3 = State(i_d, name="I3")

	# Create the match states
	m11 = NormalDistribution(5, 1)
	m12 = NormalDistribution(7, 1)

	m21 = NormalDistribution(13, 1)
	m22 = NormalDistribution(17, 1)

	m31 = NormalDistribution(-2, 1)
	m32 = NormalDistribution(-5, 1)

	m1 = State(IndependentComponentsDistribution([m11, m12]), name="M1")
	m2 = State(IndependentComponentsDistribution([m21, m22]), name="M2")
	m3 = State(IndependentComponentsDistribution([m31, m32]), name="M3")

	# Create the delete states
	d1 = State(None, name="D1")
	d2 = State(None, name="D2")
	d3 = State(None, name="D3")

	# Add all the states to the model
	model.add_states([i0, i1, i2, i3, m1, m2, m3, d1, d2, d3])

	# Create transitions from match states
	model.add_transition(model.start, m1, 0.9)
	model.add_transition(model.start, i0, 0.1)
	model.add_transition(m1, m2, 0.9)
	model.add_transition(m1, i1, 0.05)
	model.add_transition(m1, d2, 0.05)
	model.add_transition(m2, m3, 0.9)
	model.add_transition(m2, i2, 0.05)
	model.add_transition(m2, d3, 0.05)
	model.add_transition(m3, model.end, 0.9)
	model.add_transition(m3, i3, 0.1)

	# Create transitions from insert states
	model.add_transition(i0, i0, 0.70)
	model.add_transition(i0, d1, 0.15)
	model.add_transition(i0, m1, 0.15)

	model.add_transition(i1, i1, 0.70)
	model.add_transition(i1, d2, 0.15)
	model.add_transition(i1, m2, 0.15)

	model.add_transition(i2, i2, 0.70)
	model.add_transition(i2, d3, 0.15)
	model.add_transition(i2, m3, 0.15)

	model.add_transition(i3, i3, 0.85)
	model.add_transition(i3, model.end, 0.15)

	# Create transitions from delete states
	model.add_transition(d1, d2, 0.15)
	model.add_transition(d1, i1, 0.15)
	model.add_transition(d1, m2, 0.70) 

	model.add_transition(d2, d3, 0.15)
	model.add_transition(d2, i2, 0.15)
	model.add_transition(d2, m3, 0.70)

	model.add_transition(d3, i3, 0.30)
	model.add_transition(d3, model.end, 0.70)

	# Call bake to finalize the structure of the model.
	model.bake()

def dense_model(d1, d2, d3, d4):
	s1 = State(d1, "s1")
	s2 = State(d2, "s2")
	s3 = State(d3, "s3")
	s4 = State(d4, "s4")

	model = HiddenMarkovModel()
	model.add_states(s1, s2, s3, s4)
	model.add_transition(model.start, s1, 0.1)
	model.add_transition(model.start, s2, 0.3)
	model.add_transition(model.start, s3, 0.2)
	model.add_transition(model.start, s4, 0.4)
	model.add_transition(s1, s1, 0.5)
	model.add_transition(s1, s2, 0.1)
	model.add_transition(s1, s3, 0.1)
	model.add_transition(s1, s4, 0.2)
	model.add_transition(s2, s1, 0.2)
	model.add_transition(s2, s2, 0.1)
	model.add_transition(s2, s3, 0.4)
	model.add_transition(s2, s4, 0.2)
	model.add_transition(s3, s1, 0.1)
	model.add_transition(s3, s2, 0.1)
	model.add_transition(s3, s3, 0.3)
	model.add_transition(s3, s4, 0.4)
	model.add_transition(s4, s1, 0.2)
	model.add_transition(s4, s2, 0.2)
	model.add_transition(s4, s3, 0.1)
	model.add_transition(s4, s4, 0.4)
	model.add_transition(s1, model.end, 0.1)
	model.add_transition(s2, model.end, 0.1)
	model.add_transition(s3, model.end, 0.1)
	model.add_transition(s4, model.end, 0.1)
	model.bake()
	return model

def setup_discrete_dense():
	global model

	d1 = DiscreteDistribution({'A': 0.90, 'B': 0.02, 'C': 0.03, 'D': 0.05})
	d2 = DiscreteDistribution({'A': 0.02, 'B': 0.90, 'C': 0.03, 'D': 0.05})
	d3 = DiscreteDistribution({'A': 0.03, 'B': 0.02, 'C': 0.90, 'D': 0.05})
	d4 = DiscreteDistribution({'A': 0.05, 'B': 0.02, 'C': 0.03, 'D': 0.90})
	model = dense_model(d1, d2, d3, d4)

def setup_gaussian_dense():
	global model

	d1 = NormalDistribution(5, 1)
	d2 = NormalDistribution(1, 1)
	d3 = NormalDistribution(13, 2)
	d4 = NormalDistribution(16, 0.5)
	model = dense_model(d1, d2, d3, d4)

def setup_multivariate_gaussian_dense():
	global model

	mu = numpy.random.randn(4, 5)
	d1 = MultivariateGaussianDistribution(mu[0], numpy.eye(5))
	d2 = MultivariateGaussianDistribution(mu[1], numpy.eye(5))
	d3 = MultivariateGaussianDistribution(mu[2], numpy.eye(5))
	d4 = MultivariateGaussianDistribution(mu[3], numpy.eye(5))
	model = dense_model(d1, d2, d3, d4)

def setup_poisson_dense():
	global model

	d1 = PoissonDistribution(12.1)
	d2 = PoissonDistribution(8.7)
	d3 = PoissonDistribution(1)
	d4 = PoissonDistribution(5)
	model = dense_model(d1, d2, d3, d4)

def teardown():
	'''
	Remove the model at the end of the unit testing. Since it is stored in a
	global variance, simply delete it.
	'''

	pass


@with_setup(setup, teardown)
def test_viterbi_train():
	seqs = [list(x) for x in ['ACT', 'ACT', 'ACC', 'ACTC', 'ACT', 'ACT', 'CCT', 
	'CCC', 'AAT', 'CT', 'AT', 'CT', 'CT', 'CT', 'CT', 'CT', 'CT', 
	'ACT', 'ACT', 'CT', 'ACT', 'CT', 'CT', 'CT', 'CT']]

	total_improvement = model.fit(seqs, 
									 algorithm='viterbi', 
									 verbose=False, 
									 use_pseudocount=True)

	assert_equal(round(total_improvement, 4), 83.2834)


@with_setup(setup, teardown)
def test_viterbi_train_no_pseudocount():
	seqs = [list(x) for x in ['ACT', 'ACT', 'ACC', 'ACTC', 'ACT', 'ACT', 'CCT', 
	'CCC', 'AAT', 'CT', 'AT', 'CT', 'CT', 'CT', 'CT', 'CT', 'CT', 
	'ACT', 'ACT', 'CT', 'ACT', 'CT', 'CT', 'CT', 'CT']]

	total_improvement = model.fit(seqs, 
									 algorithm='viterbi', 
									 verbose=False, 
									 use_pseudocount=False)

	assert_equal(round(total_improvement, 4), 84.9318)


@with_setup(setup, teardown)
def test_viterbi_train_w_pseudocount():
	seqs = [list(x) for x in ['ACT', 'ACT', 'ACC', 'ACTC', 'ACT', 'ACT', 'CCT', 
	'CCC', 'AAT', 'CT', 'AT', 'CT', 'CT', 'CT', 'CT', 'CT', 'CT', 
	'ACT', 'ACT', 'CT', 'ACT', 'CT', 'CT', 'CT', 'CT']]

	total_improvement = model.fit(seqs, 
									 algorithm='viterbi', 
									 verbose=False, 
									 transition_pseudocount=1.)

	assert_equal(round(total_improvement, 4), 79.4713)


@with_setup(setup, teardown)
def test_viterbi_train_w_pseudocount_priors():
	seqs = [list(x) for x in ['ACT', 'ACT', 'ACC', 'ACTC', 'ACT', 'ACT', 'CCT', 
	'CCC', 'AAT', 'CT', 'AT', 'CT', 'CT', 'CT', 'CT', 'CT', 'CT', 
	'ACT', 'ACT', 'CT', 'ACT', 'CT', 'CT', 'CT', 'CT']]

	total_improvement = model.fit(seqs, 
									 algorithm='viterbi', 
									 verbose=False, 
									 transition_pseudocount=0.278,
									 use_pseudocount=True)

	assert_equal(round(total_improvement, 4), 81.7439)


@with_setup(setup, teardown)
def test_viterbi_train_w_inertia():
	seqs = [list(x) for x in ['ACT', 'ACT', 'ACC', 'ACTC', 'ACT', 'ACT', 'CCT', 
	'CCC', 'AAT', 'CT', 'AT', 'CT', 'CT', 'CT', 'CT', 'CT', 'CT', 
	'ACT', 'ACT', 'CT', 'ACT', 'CT', 'CT', 'CT', 'CT']]

	total_improvement = model.fit(seqs, 
									 algorithm='viterbi', 
									 verbose=False, 
									 edge_inertia=0.193)

	assert_equal(round(total_improvement, 4), 84.9318)


@with_setup(setup, teardown)
def test_viterbi_train_w_inertia2():
	seqs = [list(x) for x in ['ACT', 'ACT', 'ACC', 'ACTC', 'ACT', 'ACT', 'CCT', 
	'CCC', 'AAT', 'CT', 'AT', 'CT', 'CT', 'CT', 'CT', 'CT', 'CT', 
	'ACT', 'ACT', 'CT', 'ACT', 'CT', 'CT', 'CT', 'CT']]

	total_improvement = model.fit(seqs, 
									 algorithm='viterbi', 
									 verbose=False, 
									 edge_inertia=0.82)

	assert_equal(round(total_improvement, 4), 84.9318)


@with_setup(setup, teardown)
def test_viterbi_train_w_pseudocount_inertia():
	seqs = [list(x) for x in ['ACT', 'ACT', 'ACC', 'ACTC', 'ACT', 'ACT', 'CCT', 
	'CCC', 'AAT', 'CT', 'AT', 'CT', 'CT', 'CT', 'CT', 'CT', 'CT', 
	'ACT', 'ACT', 'CT', 'ACT', 'CT', 'CT', 'CT', 'CT']]

	total_improvement = model.fit(seqs, 
									 algorithm='viterbi', 
									 verbose=False, 
									 edge_inertia=0.23,
									 use_pseudocount=True) 

	assert_equal(round(total_improvement, 4), 83.2834)


@with_setup(setup, teardown)
def test_bw_train():
	seqs = [list(x) for x in ['ACT', 'ACT', 'ACC', 'ACTC', 'ACT', 'ACT', 'CCT', 
		'CCC', 'AAT', 'CT', 'AT', 'CT', 'CT', 'CT', 'CT', 'CT', 'CT', 
		'ACT', 'ACT', 'CT', 'ACT', 'CT', 'CT', 'CT', 'CT']]

	total_improvement = model.fit(seqs, 
									 algorithm='baum-welch', 
									 verbose=False, 
									 use_pseudocount=True,
									 max_iterations=5)

	assert_equal(round(total_improvement, 4), 83.1132)


@with_setup(setup_multivariate_discrete, teardown)
def test_bw_multivariate_discrete_train():
	seqs = [[['A', 'A'], ['A', 'C'], ['C', 'T']], [['A', 'A'], ['C', 'C'], ['T', 'T']],
			[['A', 'A'], ['A', 'C'], ['C', 'C'], ['T', 'T']], [['A', 'A'], ['C', 'C']]]

	total_improvement = model.fit(seqs, 
									 algorithm='baum-welch', 
									 verbose=False, 
									 use_pseudocount=True,
									 max_iterations=5)

	assert_equal(round(total_improvement, 4), 13.3622)


@with_setup(setup_multivariate_gaussian, teardown)
def test_bw_multivariate_gaussian_train():
	seqs = [[[5, 8], [8, 10], [13, 17], [-3, -4]], [[6, 7], [13, 16], [12, 11], [-6, -7]], 
			[[4, 6], [13, 15], [-4, -7]], [[6, 5], [14, 18], [-7, -5]]]

	total_improvement = model.fit(seqs, 
									 algorithm='baum-welch', 
									 verbose=False, 
									 use_pseudocount=True,
									 max_iterations=5)

	assert_equal(round(total_improvement, 4), 24.7013)

@with_setup(setup, teardown)
def test_bw_train_json():
	seqs = [list(x) for x in ['ACT', 'ACT', 'ACC', 'ACTC', 'ACT', 'ACT', 'CCT', 
		'CCC', 'AAT', 'CT', 'AT', 'CT', 'CT', 'CT', 'CT', 'CT', 'CT', 
		'ACT', 'ACT', 'CT', 'ACT', 'CT', 'CT', 'CT', 'CT']]

	total_improvement = model.fit(seqs, 
									 algorithm='baum-welch', 
									 verbose=False, 
									 use_pseudocount=True,
									 max_iterations=5)

	assert_equal(round(total_improvement, 4), 83.1132)
	assert_almost_equal(sum(model.log_probability(seq) for seq in seqs), -42.2341, 4)

	hmm = HiddenMarkovModel.from_json(model.to_json())
	assert_almost_equal(sum(model.log_probability(seq) for seq in seqs), -42.2341, 4)


@with_setup(setup, teardown)
def test_bw_train_no_pseudocount():
	seqs = [list(x) for x in ['ACT', 'ACT', 'ACC', 'ACTC', 'ACT', 'ACT', 'CCT', 
		'CCC', 'AAT', 'CT', 'AT', 'CT', 'CT', 'CT', 'CT', 'CT', 'CT', 
		'ACT', 'ACT', 'CT', 'ACT', 'CT', 'CT', 'CT', 'CT']]
	
	total_improvement = model.fit(seqs, 
									 algorithm='baum-welch', 
									 verbose=False, 
									 use_pseudocount=False,
									 max_iterations=5)
 
	assert_equal(round(total_improvement, 4), 85.681)


@with_setup(setup, teardown)
def test_bw_train_w_pseudocount():
	seqs = [list(x) for x in ['ACT', 'ACT', 'ACC', 'ACTC', 'ACT', 'ACT', 'CCT', 
		'CCC', 'AAT', 'CT', 'AT', 'CT', 'CT', 'CT', 'CT', 'CT', 'CT', 
		'ACT', 'ACT', 'CT', 'ACT', 'CT', 'CT', 'CT', 'CT']]

	total_improvement = model.fit(seqs, 
									 algorithm='baum-welch', 
									 verbose=False, 
									 transition_pseudocount=0.123,
									 max_iterations=5)
	
	assert_equal(round(total_improvement, 4), 84.9408)


@with_setup(setup, teardown)
def test_bw_train_w_pseudocount_priors():
	seqs = [list(x) for x in ['ACT', 'ACT', 'ACC', 'ACTC', 'ACT', 'ACT', 'CCT', 
		'CCC', 'AAT', 'CT', 'AT', 'CT', 'CT', 'CT', 'CT', 'CT', 'CT', 
		'ACT', 'ACT', 'CT', 'ACT', 'CT', 'CT', 'CT', 'CT']]

	total_improvement = model.fit(seqs, 
									 algorithm='baum-welch', 
									 verbose=False, 
									 transition_pseudocount=0.278,
									 use_pseudocount=True,
									 max_iterations=5)
	 
	assert_equal(round(total_improvement, 4), 81.2265)


@with_setup(setup, teardown)
def test_bw_train_w_inertia():
	seqs = [list(x) for x in ['ACT', 'ACT', 'ACC', 'ACTC', 'ACT', 'ACT', 'CCT', 
		'CCC', 'AAT', 'CT', 'AT', 'CT', 'CT', 'CT', 'CT', 'CT', 'CT', 
		'ACT', 'ACT', 'CT', 'ACT', 'CT', 'CT', 'CT', 'CT']]

	total_improvement = model.fit(seqs, 
									 algorithm='baum-welch', 
									 verbose=False, 
									 edge_inertia=0.193,
									 max_iterations=5)
	 
	assert_equal(round(total_improvement, 4), 85.0528)


@with_setup(setup, teardown)
def test_bw_train_w_inertia2():
	seqs = [list(x) for x in ['ACT', 'ACT', 'ACC', 'ACTC', 'ACT', 'ACT', 'CCT', 
		'CCC', 'AAT', 'CT', 'AT', 'CT', 'CT', 'CT', 'CT', 'CT', 'CT', 
		'ACT', 'ACT', 'CT', 'ACT', 'CT', 'CT', 'CT', 'CT']]

	total_improvement = model.fit(seqs, 
									 algorithm='baum-welch', 
									 verbose=False, 
									 edge_inertia=0.82,
									 max_iterations=5)
  
	assert_equal(round(total_improvement, 4), 72.5134)


@with_setup(setup, teardown)
def test_bw_train_w_pseudocount_inertia():
	seqs = [list(x) for x in ['ACT', 'ACT', 'ACC', 'ACTC', 'ACT', 'ACT', 'CCT', 
		'CCC', 'AAT', 'CT', 'AT', 'CT', 'CT', 'CT', 'CT', 'CT', 'CT', 
		'ACT', 'ACT', 'CT', 'ACT', 'CT', 'CT', 'CT', 'CT']]

	total_improvement = model.fit(seqs, 
									 algorithm='baum-welch', 
									 verbose=False, 
									 edge_inertia=0.02,
									 use_pseudocount=True,
									 max_iterations=5)
 
	assert_equal(round(total_improvement, 4), 83.0764)


@with_setup(setup, teardown)
def test_bw_train_w_frozen_distributions():
	seqs = [list(x) for x in ['ACT', 'ACT', 'ACC', 'ACTC', 'ACT', 'ACT', 'CCT', 
		'CCC', 'AAT', 'CT', 'AT', 'CT', 'CT', 'CT', 'CT', 'CT', 'CT', 
		'ACT', 'ACT', 'CT', 'ACT', 'CT', 'CT', 'CT', 'CT']]

	total_improvement = model.fit(seqs, 
									 algorithm='baum-welch', 
									 verbose=False, 
									 distribution_inertia=1.00,
									 max_iterations=5)
  
	assert_equal(round(total_improvement, 4), 64.474)


@with_setup(setup, teardown)
def test_bw_train_w_frozen_edges():
	seqs = [list(x) for x in ['ACT', 'ACT', 'ACC', 'ACTC', 'ACT', 'ACT', 'CCT', 
		'CCC', 'AAT', 'CT', 'AT', 'CT', 'CT', 'CT', 'CT', 'CT', 'CT', 
		'ACT', 'ACT', 'CT', 'ACT', 'CT', 'CT', 'CT', 'CT']]

	total_improvement = model.fit(seqs, 
									 algorithm='baum-welch', 
									 verbose=False, 
									 edge_inertia=1.00,
									 max_iterations=5)

	assert_equal(round(total_improvement, 4), 44.0208)


@with_setup(setup, teardown)
def test_bw_train_w_edge_a_distribution_inertia():
	seqs = [list(x) for x in ['ACT', 'ACT', 'ACC', 'ACTC', 'ACT', 'ACT', 'CCT', 
		'CCC', 'AAT', 'CT', 'AT', 'CT', 'CT', 'CT', 'CT', 'CT', 'CT', 
		'ACT', 'ACT', 'CT', 'ACT', 'CT', 'CT', 'CT', 'CT']]

	total_improvement = model.fit(seqs, 
									 algorithm='baum-welch', 
									 verbose=False, 
									 edge_inertia=0.5,
									 distribution_inertia=0.5,
									 max_iterations=5)
 
	assert_equal(round(total_improvement, 4), 81.5447)


@with_setup(setup, teardown)
def test_bw_train_parallel():
	seqs = [list(x) for x in ['ACT', 'ACT', 'ACC', 'ACTC', 'ACT', 'ACT', 'CCT', 
		'CCC', 'AAT', 'CT', 'AT', 'CT', 'CT', 'CT', 'CT', 'CT', 'CT', 
		'ACT', 'ACT', 'CT', 'ACT', 'CT', 'CT', 'CT', 'CT']]

	total_improvement = model.fit(seqs, 
									 algorithm='baum-welch', 
									 verbose=False, 
									 use_pseudocount=True,
									 max_iterations=5,
									 n_jobs=2)

	assert_equal(round(total_improvement, 4), 83.1132)


@with_setup(setup, teardown)
def test_bw_train_no_pseudocount_parallel():
	seqs = [list(x) for x in ['ACT', 'ACT', 'ACC', 'ACTC', 'ACT', 'ACT', 'CCT', 
		'CCC', 'AAT', 'CT', 'AT', 'CT', 'CT', 'CT', 'CT', 'CT', 'CT', 
		'ACT', 'ACT', 'CT', 'ACT', 'CT', 'CT', 'CT', 'CT']]
	
	total_improvement = model.fit(seqs, 
									 algorithm='baum-welch', 
									 verbose=False, 
									 use_pseudocount=False,
									 max_iterations=5,
									 n_jobs=2)
 
	assert_equal(round(total_improvement, 4), 85.681)


@with_setup(setup, teardown)
def test_bw_train_w_pseudocount_parallel():
	seqs = [list(x) for x in ['ACT', 'ACT', 'ACC', 'ACTC', 'ACT', 'ACT', 'CCT', 
		'CCC', 'AAT', 'CT', 'AT', 'CT', 'CT', 'CT', 'CT', 'CT', 'CT', 
		'ACT', 'ACT', 'CT', 'ACT', 'CT', 'CT', 'CT', 'CT']]

	total_improvement = model.fit(seqs, 
									 algorithm='baum-welch', 
									 verbose=False, 
									 transition_pseudocount=0.123,
									 max_iterations=5,
									 n_jobs=2)
	
	assert_equal(round(total_improvement, 4), 84.9408)


@with_setup(setup, teardown)
def test_bw_train_w_pseudocount_priors_parallel():
	seqs = [list(x) for x in ['ACT', 'ACT', 'ACC', 'ACTC', 'ACT', 'ACT', 'CCT', 
		'CCC', 'AAT', 'CT', 'AT', 'CT', 'CT', 'CT', 'CT', 'CT', 'CT', 
		'ACT', 'ACT', 'CT', 'ACT', 'CT', 'CT', 'CT', 'CT']]

	total_improvement = model.fit(seqs, 
									 algorithm='baum-welch', 
									 verbose=False, 
									 transition_pseudocount=0.278,
									 use_pseudocount=True,
									 max_iterations=5,
									 n_jobs=2)
	 
	assert_equal(round(total_improvement, 4), 81.2265)


@with_setup(setup, teardown)
def test_bw_train_w_inertia_parallel():
	seqs = [list(x) for x in ['ACT', 'ACT', 'ACC', 'ACTC', 'ACT', 'ACT', 'CCT', 
		'CCC', 'AAT', 'CT', 'AT', 'CT', 'CT', 'CT', 'CT', 'CT', 'CT', 
		'ACT', 'ACT', 'CT', 'ACT', 'CT', 'CT', 'CT', 'CT']]

	total_improvement = model.fit(seqs, 
									 algorithm='baum-welch', 
									 verbose=False, 
									 edge_inertia=0.193,
									 max_iterations=5,
									 n_jobs=2)
	 
	assert_equal(round(total_improvement, 4), 85.0528)


@with_setup(setup, teardown)
def test_bw_train_w_inertia2_parallel():
	seqs = [list(x) for x in ['ACT', 'ACT', 'ACC', 'ACTC', 'ACT', 'ACT', 'CCT', 
		'CCC', 'AAT', 'CT', 'AT', 'CT', 'CT', 'CT', 'CT', 'CT', 'CT', 
		'ACT', 'ACT', 'CT', 'ACT', 'CT', 'CT', 'CT', 'CT']]

	total_improvement = model.fit(seqs, 
									 algorithm='baum-welch', 
									 verbose=False, 
									 edge_inertia=0.82,
									 max_iterations=5,
									 n_jobs=2)
  
	assert_equal(round(total_improvement, 4), 72.5134)


@with_setup(setup, teardown)
def test_bw_train_w_pseudocount_inertia_parallel():
	seqs = [list(x) for x in ['ACT', 'ACT', 'ACC', 'ACTC', 'ACT', 'ACT', 'CCT', 
		'CCC', 'AAT', 'CT', 'AT', 'CT', 'CT', 'CT', 'CT', 'CT', 'CT', 
		'ACT', 'ACT', 'CT', 'ACT', 'CT', 'CT', 'CT', 'CT']]

	total_improvement = model.fit(seqs, 
									 algorithm='baum-welch', 
									 verbose=False, 
									 edge_inertia=0.02,
									 use_pseudocount=True,
									 max_iterations=5,
									 n_jobs=2)
 
	assert_equal(round(total_improvement, 4), 83.0764)


@with_setup(setup, teardown)
def test_bw_train_w_frozen_distributions_parallel():
	seqs = [list(x) for x in ['ACT', 'ACT', 'ACC', 'ACTC', 'ACT', 'ACT', 'CCT', 
		'CCC', 'AAT', 'CT', 'AT', 'CT', 'CT', 'CT', 'CT', 'CT', 'CT', 
		'ACT', 'ACT', 'CT', 'ACT', 'CT', 'CT', 'CT', 'CT']]

	total_improvement = model.fit(seqs, 
									 algorithm='baum-welch', 
									 verbose=False, 
									 distribution_inertia=1.00,
									 max_iterations=5,
									 n_jobs=2)
  
	assert_equal(round(total_improvement, 4), 64.474)


@with_setup(setup, teardown)
def test_bw_train_w_frozen_edges_parallel():
	seqs = [list(x) for x in ['ACT', 'ACT', 'ACC', 'ACTC', 'ACT', 'ACT', 'CCT', 
		'CCC', 'AAT', 'CT', 'AT', 'CT', 'CT', 'CT', 'CT', 'CT', 'CT', 
		'ACT', 'ACT', 'CT', 'ACT', 'CT', 'CT', 'CT', 'CT']]

	total_improvement = model.fit(seqs, 
									 algorithm='baum-welch', 
									 verbose=False, 
									 edge_inertia=1.00,
									 max_iterations=5,
									 n_jobs=2)

	assert_equal(round(total_improvement, 4), 44.0208)


@with_setup(setup, teardown)
def test_bw_train_w_edge_a_distribution_inertia():
	seqs = [list(x) for x in ['ACT', 'ACT', 'ACC', 'ACTC', 'ACT', 'ACT', 'CCT', 
		'CCC', 'AAT', 'CT', 'AT', 'CT', 'CT', 'CT', 'CT', 'CT', 'CT', 
		'ACT', 'ACT', 'CT', 'ACT', 'CT', 'CT', 'CT', 'CT']]

	total_improvement = model.fit(seqs, 
									 algorithm='baum-welch', 
									 verbose=False, 
									 edge_inertia=0.5,
									 distribution_inertia=0.5,
									 max_iterations=5,
									 n_jobs=2)
 
	assert_equal(round(total_improvement, 4), 81.5447)


@with_setup(setup, teardown)
def test_dimension():
	# single dimensions
	s1d1 = State(NormalDistribution(5, 2))
	s2d1 = State(NormalDistribution(0, 1))
	s3d1 = State(UniformDistribution(0, 10))

	hmmd1 = HiddenMarkovModel()

	assert_equal(hmmd1.d, 0)

	hmmd1.add_transition(hmmd1.start, s1d1, 0.5)
	hmmd1.add_transition(hmmd1.start, s2d1, 0.5)
	hmmd1.add_transition(s1d1, s3d1, 1)
	hmmd1.add_transition(s2d1, s3d1, 1)
	hmmd1.add_transition(s3d1, s1d1, 0.5)
	hmmd1.add_transition(s3d1, s2d1, 0.5)

	assert_equal(hmmd1.d, 0)

	hmmd1.bake()

	assert_equal(hmmd1.d, 1)

	# multiple dimensions
	s1d3 = State(MultivariateGaussianDistribution([1, 4, 3], [[3, 0, 1],[0, 3, 0],[1, 0, 3]]))
	s2d3 = State(MultivariateGaussianDistribution([7, 7, 7], [[1, 0, 0],[0, 5, 0],[0, 0, 3]]))
	s3d3 = State(IndependentComponentsDistribution([UniformDistribution(0, 10), UniformDistribution(0, 10), UniformDistribution(0, 10)]))

	hmmd3 = HiddenMarkovModel()

	assert_equal(hmmd3.d, 0)

	hmmd3.add_transition(hmmd3.start, s1d3, 0.5)
	hmmd3.add_transition(hmmd3.start, s2d3, 0.5)
	hmmd3.add_transition(s1d3, s3d3, 1)
	hmmd3.add_transition(s2d3, s3d3, 1)
	hmmd3.add_transition(s3d3, s1d3, 0.5)
	hmmd3.add_transition(s3d3, s2d3, 0.5)

	assert_equal(hmmd3.d, 0)

	hmmd3.bake()

	assert_equal(hmmd3.d, 3)

	sbd1 = State(UniformDistribution(0, 10))
	sbd3 = State(MultivariateGaussianDistribution([1, 4, 3], [[3, 0, 1],[0, 3, 0],[1, 0, 3]]))

	hmmb = HiddenMarkovModel()
	hmmb.add_transition(hmmb.start, sbd1, 0.5)
	hmmb.add_transition(hmmb.start, sbd3, 0.5)
	hmmb.add_transition(sbd1, sbd1, 0.5)
	hmmb.add_transition(sbd1, sbd3, 0.5)
	hmmb.add_transition(sbd3, sbd1, 0.5)
	hmmb.add_transition(sbd3, sbd3, 0.5)

	assert_raises(ValueError, hmmb.bake)

@with_setup(setup, teardown)
def test_serialization():
	s1 = State(MultivariateGaussianDistribution([1, 4], [[.1, 0], [0, 1]]))
	s2 = State(MultivariateGaussianDistribution([-1, 1], [[1, 0], [0, 7]]))
	s3 = State(MultivariateGaussianDistribution([3, 5], [[.5, 0], [0, 6]]))

	hmm1 = HiddenMarkovModel()

	hmm1.add_transition(hmm1.start, s1, 1)
	hmm1.add_transition(s1, s1, 0.8)
	hmm1.add_transition(s1, s2, 0.2)
	hmm1.add_transition(s2, s2, 0.9)
	hmm1.add_transition(s2, s3, 0.1)
	hmm1.add_transition(s3, s3, 0.95)
	hmm1.add_transition(s3, hmm1.end, 0.05)

	hmm1.bake()

	hmm2 = pickle.loads(pickle.dumps(hmm1))

	for i in range(10):
		sequence = hmm1.sample(10)
		assert_almost_equal(hmm1.log_probability(sequence), hmm2.log_probability(sequence))

@with_setup(setup_discrete_dense, teardown)
def test_discrete_from_samples():
	X = [model.sample() for i in range(25)]
	model2 = HiddenMarkovModel.from_samples(DiscreteDistribution, 4, X, max_iterations=25)
	model3 = HiddenMarkovModel.from_samples(DiscreteDistribution, 2, X, max_iterations=25)

	logp1 = sum(map(model.log_probability, X))
	logp2 = sum(map(model2.log_probability, X))
	logp3 = sum(map(model3.log_probability, X))

	assert_greater(logp2, logp1)
	assert_greater(logp2, logp1)

@with_setup(setup_gaussian_dense, teardown)
def test_gaussian_from_samples():
	X = [model.sample() for i in range(25)]
	model2 = HiddenMarkovModel.from_samples(NormalDistribution, 4, X, max_iterations=25)
	model3 = HiddenMarkovModel.from_samples(NormalDistribution, 2, X, max_iterations=25)

	logp1 = sum(map(model.log_probability, X))
	logp2 = sum(map(model2.log_probability, X))
	logp3 = sum(map(model3.log_probability, X))

	assert_greater(logp2, logp1)
	assert_greater(logp2, logp1)

@with_setup(setup_multivariate_gaussian_dense, teardown)
def test_multivariate_gaussian_from_samples():
	X = [model.sample() for i in range(25)]
	model2 = HiddenMarkovModel.from_samples(MultivariateGaussianDistribution, 4, X, max_iterations=25)
	model3 = HiddenMarkovModel.from_samples(MultivariateGaussianDistribution, 2, X, max_iterations=25)

	logp1 = sum(map(model.log_probability, X))
	logp2 = sum(map(model2.log_probability, X))
	logp3 = sum(map(model3.log_probability, X))

	assert_greater(logp2, logp1)
	assert_greater(logp2, logp1)

@with_setup(setup_poisson_dense, teardown)
def test_poisson_from_samples():
	X = [model.sample() for i in range(25)]
	model2 = HiddenMarkovModel.from_samples(PoissonDistribution, 4, X, max_iterations=25)
	model3 = HiddenMarkovModel.from_samples(PoissonDistribution, 2, X, max_iterations=25)

	logp1 = sum(map(model.log_probability, X))
	logp2 = sum(map(model2.log_probability, X))
	logp3 = sum(map(model3.log_probability, X))

	assert_greater(logp2, logp1)
	assert_greater(logp2, logp1)

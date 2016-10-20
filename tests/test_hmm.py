from __future__ import (division)

from pomegranate import *
from nose.tools import with_setup
from nose.tools import assert_almost_equal
from nose.tools import assert_equal
from nose.tools import assert_not_equal
from nose.tools import assert_less_equal
from nose.tools import assert_raises
import pickle
import random
import numpy as np

np.random.seed(0)

def setup():
	pass

def teardown():
	pass

@with_setup( setup, teardown )
def test_dimension():
	# single dimensions
	s1d1 = State( NormalDistribution( 5, 2 ) )
	s2d1 = State( NormalDistribution( 0, 1 ) )
	s3d1 = State( UniformDistribution( 0, 10 ) )

	hmmd1 = HiddenMarkovModel()

	assert_equal( hmmd1.d, 0 )

	hmmd1.add_transition( hmmd1.start, s1d1, 0.5 )
	hmmd1.add_transition( hmmd1.start, s2d1, 0.5 )
	hmmd1.add_transition( s1d1, s3d1, 1 )
	hmmd1.add_transition( s2d1, s3d1, 1 )
	hmmd1.add_transition( s3d1, s1d1, 0.5 )
	hmmd1.add_transition( s3d1, s2d1, 0.5 )

	assert_equal( hmmd1.d, 0 )

	hmmd1.bake()

	assert_equal( hmmd1.d, 1 )

	# multiple dimensions
	s1d3 = State( MultivariateGaussianDistribution([1, 4, 3], [[3, 0, 1],[0, 3, 0],[1, 0, 3]]) )
	s2d3 = State( MultivariateGaussianDistribution([7, 7, 7], [[1, 0, 0],[0, 5, 0],[0, 0, 3]]) )
	s3d3 = State( IndependentComponentsDistribution([ UniformDistribution(0, 10), UniformDistribution(0, 10), UniformDistribution(0, 10) ]) )

	hmmd3 = HiddenMarkovModel()

	assert_equal( hmmd3.d, 0 )

	hmmd3.add_transition( hmmd3.start, s1d3, 0.5 )
	hmmd3.add_transition( hmmd3.start, s2d3, 0.5 )
	hmmd3.add_transition( s1d3, s3d3, 1 )
	hmmd3.add_transition( s2d3, s3d3, 1 )
	hmmd3.add_transition( s3d3, s1d3, 0.5 )
	hmmd3.add_transition( s3d3, s2d3, 0.5 )

	assert_equal( hmmd3.d, 0 )

	hmmd3.bake()

	assert_equal( hmmd3.d, 3 )

	sbd1 = State( UniformDistribution( 0, 10 ) )
	sbd3 = State( MultivariateGaussianDistribution([1, 4, 3], [[3, 0, 1],[0, 3, 0],[1, 0, 3]]) )

	hmmb = HiddenMarkovModel()
	hmmb.add_transition( hmmb.start, sbd1, 0.5 )
	hmmb.add_transition( hmmb.start, sbd3, 0.5 )
	hmmb.add_transition( sbd1, sbd1, 0.5 )
	hmmb.add_transition( sbd1, sbd3, 0.5 )
	hmmb.add_transition( sbd3, sbd1, 0.5 )
	hmmb.add_transition( sbd3, sbd3, 0.5 )

	assert_raises( ValueError, hmmb.bake )

@with_setup( setup, teardown )
def test_serialization():
	s1 = State( MultivariateGaussianDistribution( [1, 4], [[.1, 0], [0, 1]] ) )
	s2 = State( MultivariateGaussianDistribution( [-1, 1], [[1, 0], [0, 7]] ) )
	s3 = State( MultivariateGaussianDistribution( [3, 5], [[.5, 0], [0, 6]] ) )

	hmm1 = HiddenMarkovModel()

	hmm1.add_transition( hmm1.start, s1, 1 )
	hmm1.add_transition( s1, s1, 0.8 )
	hmm1.add_transition( s1, s2, 0.2 )
	hmm1.add_transition( s2, s2, 0.9 )
	hmm1.add_transition( s2, s3, 0.1 )
	hmm1.add_transition( s3, s3, 0.95 )
	hmm1.add_transition( s3, hmm1.end, 0.05 )

	hmm1.bake()

	hmm2 = pickle.loads(pickle.dumps(hmm1))

	for i in range(10):
		sequence = hmm1.sample(10)
		assert_almost_equal(hmm1.log_probability(sequence), hmm2.log_probability(sequence))

@with_setup( setup, teardown )
def test_pruning():
	# single dimensions
	s1 = State( NormalDistribution( 1, 10 ) )
	s2 = State( NormalDistribution( 2, 10 ) )
	s3 = State( UniformDistribution( 3, 10 ) )
	s4 = State( UniformDistribution( 4, 10 ) )
	s5 = State( UniformDistribution( 5, 10 ) )

	hmm = HiddenMarkovModel()

	hmm.add_transition( hmm.start, s1, 0.5 )
	hmm.add_transition( hmm.start, s2, 0.5 )
	hmm.add_transition( s1, s3, 1 )
	hmm.add_transition( s2, s3, 1 )
	hmm.add_transition( s3, s3, 0.25 )
	hmm.add_transition( s3, s4, 0.5 )
	hmm.add_transition( s3, s5, 0.25 )
	hmm.add_transition( s4, hmm.end, 1 )
	hmm.add_transition( s5, hmm.end, 1 )

	hmm.bake()

	assert_equal( len( hmm.graph.edges() ), 9 )
	assert_equal( hmm.graph.get_edge_data( s3, s3 )['pseudocount'], 0.25 )

	# Pruning
	hmm.prune_transition( s3, s4 )

	assert_equal( len( hmm.graph.edges() ), 8 )
	assert_equal( hmm.graph.get_edge_data( s3, s3 )['pseudocount'], 0.5 )

	hmm.bake()

	assert_equal( len( hmm.graph.edges() ), 7 )
	assert_equal( hmm.graph.get_edge_data( s3, s3 )['pseudocount'], 0.5 )

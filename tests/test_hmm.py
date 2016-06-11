from __future__ import (division)

from pomegranate import *
from nose.tools import with_setup
from nose.tools import assert_almost_equal
from nose.tools import assert_equal
from nose.tools import assert_not_equal
from nose.tools import assert_less_equal
from nose.tools import assert_raises
import random
import numpy as np

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
	s2d3 = State( MultivariateGaussianDistribution([7, 7, 7], [[1, 3, 1],[3, 5, 0],[1, 0, 3]]) )
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
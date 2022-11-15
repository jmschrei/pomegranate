# test_bayes_net.py
# Authors: Jacob Schreiber <jmschreiber91@gmail.com>

'''
These are unit tests for the Markov network model of pomegranate.
'''

from __future__ import division

from pomegranate import JointProbabilityTable
from pomegranate import MarkovNetwork

from .assert_tools import assert_almost_equal

from numpy.testing import assert_array_equal

import numpy

import pytest


@pytest.fixture
def markov_network_int():
	d1 = JointProbabilityTable([
		[0, 0, 0.1],
		[0, 1, 0.2],
		[1, 0, 0.4],
		[1, 1, 0.3]], [0, 1])

	d2 = JointProbabilityTable([
		[0, 0, 0, 0.05],
		[0, 0, 1, 0.15],
		[0, 1, 0, 0.07],
		[0, 1, 1, 0.03],
		[1, 0, 0, 0.12],
		[1, 0, 1, 0.18],
		[1, 1, 0, 0.10],
		[1, 1, 1, 0.30]], [1, 2, 3])

	d3 = JointProbabilityTable([
		[0, 0, 0, 0.08],
		[0, 0, 1, 0.12],
		[0, 1, 0, 0.11],
		[0, 1, 1, 0.19],
		[1, 0, 0, 0.04],
		[1, 0, 1, 0.06],
		[1, 1, 0, 0.23],
		[1, 1, 1, 0.17]], [2, 3, 4])

	model1 = MarkovNetwork([d1])
	model1.bake()

	model2 = MarkovNetwork([d1, d2])
	model2.bake()

	model3 = MarkovNetwork([d1, d2, d3])
	model3.bake()

	model4 = MarkovNetwork([d1, d3])
	model4.bake()
	return d1, d2, d3, model1, model2, model3, model4


@pytest.fixture
def markov_network_str():
	d1 = JointProbabilityTable([
		['0', '0', 0.1],
		['0', '1', 0.2],
		['1', '0', 0.4],
		['1', '1', 0.3]], [0, 1])

	d2 = JointProbabilityTable([
		['0', '0', '0', 0.05],
		['0', '0', '1', 0.15],
		['0', '1', '0', 0.07],
		['0', '1', '1', 0.03],
		['1', '0', '0', 0.12],
		['1', '0', '1', 0.18],
		['1', '1', '0', 0.10],
		['1', '1', '1', 0.30]], [1, 2, 3])

	d3 = JointProbabilityTable([
		['0', '0', '0', 0.08],
		['0', '0', '1', 0.12],
		['0', '1', '0', 0.11],
		['0', '1', '1', 0.19],
		['1', '0', '0', 0.04],
		['1', '0', '1', 0.06],
		['1', '1', '0', 0.23],
		['1', '1', '1', 0.17]], [2, 3, 4])

	model1 = MarkovNetwork([d1])
	model1.bake()

	model2 = MarkovNetwork([d1, d2])
	model2.bake()

	model3 = MarkovNetwork([d1, d2, d3])
	model3.bake()

	model4 = MarkovNetwork([d1, d3])
	model4.bake()
	return d1, d2, d3, model1, model2, model3, model4


@pytest.fixture
def markov_network_bool():
	d1 = JointProbabilityTable([
		[False, False, 0.1],
		[False, True,  0.2],
		[True,  False, 0.4],
		[True,  True,  0.3]], [0, 1])

	d2 = JointProbabilityTable([
		[False, False, False, 0.05],
		[False, False, True,  0.15],
		[False, True,  False, 0.07],
		[False, True,  True,  0.03],
		[True,  False, False, 0.12],
		[True,  False, True,  0.18],
		[True,  True,  False, 0.10],
		[True,  True,  True,  0.30]], [1, 2, 3])

	d3 = JointProbabilityTable([
		[False, False, False, 0.08],
		[False, False, True,  0.12],
		[False, True,  False, 0.11],
		[False, True,  True,  0.19],
		[True,  False, False, 0.04],
		[True,  False, True,  0.06],
		[True,  True,  False, 0.23],
		[True,  True,  True,  0.17]], [2, 3, 4])

	model1 = MarkovNetwork([d1])
	model1.bake()

	model2 = MarkovNetwork([d1, d2])
	model2.bake()

	model3 = MarkovNetwork([d1, d2, d3])
	model3.bake()

	model4 = MarkovNetwork([d1, d3])
	model4.bake()
	return d1, d2, d3, model1, model2, model3, model4


@pytest.fixture
def markov_network_mixed():
	d1 = JointProbabilityTable([
		[False, 'blue', 0.1],
		[False, 'red',  0.2],
		[True,  'blue', 0.4],
		[True,  'red',  0.3]], [0, 1])

	d2 = JointProbabilityTable([
		['blue', False, 0, 0.05],
		['blue', False, 1, 0.15],
		['blue', True,  0, 0.07],
		['blue', True,  1, 0.03],
		['red',  False, 0, 0.12],
		['red',  False, 1, 0.18],
		['red',  True,  0, 0.10],
		['red',  True,  1, 0.30]], [1, 2, 3])

	d3 = JointProbabilityTable([
		[False, 0, 'a', 0.08],
		[False, 0, 'b', 0.12],
		[False, 1, 'a', 0.11],
		[False, 1, 'b', 0.19],
		[True,  0, 'a', 0.04],
		[True,  0, 'b', 0.06],
		[True,  1, 'a', 0.23],
		[True,  1, 'b', 0.17]], [2, 3, 4])

	model1 = MarkovNetwork([d1])
	model1.bake()

	model2 = MarkovNetwork([d1, d2])
	model2.bake()

	model3 = MarkovNetwork([d1, d2, d3])
	model3.bake()

	model4 = MarkovNetwork([d1, d3])
	model4.bake()
	return d1, d2, d3, model1, model2, model3, model4


def test_initialize():
	with pytest.raises(ValueError):
		MarkovNetwork([])

	d1 = JointProbabilityTable([
		[0, 0, 0.2],
		[0, 1, 0.2],
		[1, 0, 0.4],
		[1, 1, 0.2]], [0, 1])

	model = MarkovNetwork([d1])

def test_structure(markov_network_int):
	d1, d2, d3, model1, model2, model3, model4 = markov_network_int
	assert model1.structure == ((0, 1),)
	assert model2.structure == ((0, 1), (1, 2, 3))
	assert model3.structure == ((0, 1), (1, 2, 3), (2, 3, 4))
	assert model4.structure == ((0, 1), (2, 3, 4))

def test_partition(markov_network_int):
	d1, d2, d3, model1, model2, model3, model4 = markov_network_int
	model3.bake()
	assert model3.partition != float("inf")

	model3.bake(calculate_partition=False)
	assert model3.partition == float("inf")

def test_d(markov_network_int):
	d1, d2, d3, model1, model2, model3, model4 = markov_network_int
	assert model1.d == 2
	assert model2.d == 4
	assert model3.d == 5
	assert model4.d == 5

def test_d_mixed(markov_network_mixed):
	d1, d2, d3, model1, model2, model3, model4 = markov_network_mixed
	assert model1.d == 2
	assert model2.d == 4
	assert model3.d == 5
	assert model4.d == 5

def test_log_probability_int(markov_network_int):
	d1, d2, d3, model1, model2, model3, model4 = markov_network_int
	x = [1, 0]
	logp1 = model1.log_probability(x)
	logp2 = d1.log_probability(x)

	assert_almost_equal(logp1, logp2)
	assert_almost_equal(logp1, numpy.log(0.4))

	x = [1, 0, 1, 1]
	logp1 = model2.log_probability(x)
	logp2 = d1.log_probability(x[:2]) + d2.log_probability(x[1:])

	with pytest.raises(AssertionError):
		assert_almost_equal(logp1, logp2)
	assert_almost_equal(logp1, -3.7297014467295373)
	
	x = [1, 0, 1, 0, 1]
	logp1 = model3.log_probability(x)
	logp2 = (d1.log_probability(x[:2]) + d2.log_probability(x[1:4])
		+ d3.log_probability(x[2:]))

	with pytest.raises(AssertionError):
		assert_almost_equal(logp1, logp2)
	assert_almost_equal(logp1, -4.429966143312331)

	logp3 = model4.log_probability(x)
	logp4 = d1.log_probability(x[:2]) + d3.log_probability(x[2:])

	assert_almost_equal(logp3, logp4)
	assert_almost_equal(logp3, -3.7297014486341915)
	with pytest.raises(AssertionError):
		assert_almost_equal(logp1, logp3)

def test_log_probability_str(markov_network_str):
	d1, d2, d3, model1, model2, model3, model4 = markov_network_str
	x = ['1', '0']
	logp1 = model1.log_probability(x)
	logp2 = d1.log_probability(x)

	assert_almost_equal(logp1, logp2)
	assert_almost_equal(logp1, numpy.log(0.4))

	x = ['1', '0', '1', '1']
	logp1 = model2.log_probability(x)
	logp2 = d1.log_probability(x[:2]) + d2.log_probability(x[1:])

	with pytest.raises(AssertionError):
		assert_almost_equal(logp1, logp2)
	assert_almost_equal(logp1, -3.7297014467295373)
	
	x = ['1', '0', '1', '0', '1']
	logp1 = model3.log_probability(x)
	logp2 = (d1.log_probability(x[:2]) + d2.log_probability(x[1:4])
		+ d3.log_probability(x[2:]))

	with pytest.raises(AssertionError):
		assert_almost_equal(logp1, logp2)
	assert_almost_equal(logp1, -4.429966143312331)

	logp3 = model4.log_probability(x)
	logp4 = d1.log_probability(x[:2]) + d3.log_probability(x[2:])

	assert_almost_equal(logp3, logp4)
	assert_almost_equal(logp3, -3.7297014486341915)
	with pytest.raises(AssertionError):
		assert_almost_equal(logp1, logp3)

def test_log_probability_bool(markov_network_bool):
	d1, d2, d3, model1, model2, model3, model4 = markov_network_bool
	x = [True, False]
	logp1 = model1.log_probability(x)
	logp2 = d1.log_probability(x)

	assert_almost_equal(logp1, logp2)
	assert_almost_equal(logp1, numpy.log(0.4))

	x = [True, False, True, True]
	logp1 = model2.log_probability(x)
	logp2 = d1.log_probability(x[:2]) + d2.log_probability(x[1:])

	with pytest.raises(AssertionError):
		assert_almost_equal(logp1, logp2)
	assert_almost_equal(logp1, -3.7297014467295373)
	
	x = [True, False, True, False, True]
	logp1 = model3.log_probability(x)
	logp2 = (d1.log_probability(x[:2]) + d2.log_probability(x[1:4])
		+ d3.log_probability(x[2:]))

	with pytest.raises(AssertionError):
		assert_almost_equal(logp1, logp2)
	assert_almost_equal(logp1, -4.429966143312331)

	logp3 = model4.log_probability(x)
	logp4 = d1.log_probability(x[:2]) + d3.log_probability(x[2:])

	assert_almost_equal(logp3, logp4)
	assert_almost_equal(logp3, -3.7297014486341915)
	with pytest.raises(AssertionError):
		assert_almost_equal(logp1, logp3)

def test_log_probability_mixed(markov_network_mixed):
	d1, d2, d3, model1, model2, model3, model4 = markov_network_mixed
	x = [True, 'blue']

	logp1 = model1.log_probability(x)
	logp2 = d1.log_probability(x)

	assert_almost_equal(logp1, logp2)
	assert_almost_equal(logp1, numpy.log(0.4))

	x = [True, 'blue', True, 1]
	logp1 = model2.log_probability(x)
	logp2 = d1.log_probability(x[:2]) + d2.log_probability(x[1:])

	with pytest.raises(AssertionError):
		assert_almost_equal(logp1, logp2)
	assert_almost_equal(logp1, -3.7297014467295373)
	
	x = [1, 'blue', True, 0, 'b']
	logp1 = model3.log_probability(x)
	logp2 = (d1.log_probability(x[:2]) + d2.log_probability(x[1:4])
		+ d3.log_probability(x[2:]))

	with pytest.raises(AssertionError):
		assert_almost_equal(logp1, logp2)
	assert_almost_equal(logp1, -4.429966143312331)

	logp3 = model4.log_probability(x)
	logp4 = d1.log_probability(x[:2]) + d3.log_probability(x[2:])

	assert_almost_equal(logp3, logp4)
	assert_almost_equal(logp3, -3.7297014486341915)
	with pytest.raises(AssertionError):
		assert_almost_equal(logp1, logp3)

def test_log_probability_unnormalized_int(markov_network_int):
	d1, d2, d3, model1, model2, model3, model4 = markov_network_int
	x = [1, 0]
	logp1 = model1.log_probability(x, unnormalized=True)
	logp2 = d1.log_probability(x)

	assert_almost_equal(logp1, logp2)
	assert_almost_equal(logp1, numpy.log(0.4))

	x = [1, 0, 1, 1]
	logp1 = model2.log_probability(x, unnormalized=True)
	logp2 = d1.log_probability(x[:2]) + d2.log_probability(x[1:])

	assert_almost_equal(logp1, logp2)
	assert_almost_equal(logp1, numpy.log(0.4 * 0.03))
	
	x = [1, 0, 1, 0, 1]
	logp1 = model3.log_probability(x, unnormalized=True)
	logp2 = (d1.log_probability(x[:2]) + d2.log_probability(x[1:4])
		+ d3.log_probability(x[2:]))

	assert_almost_equal(logp1, logp2)
	assert_almost_equal(logp1, numpy.log(0.4 * 0.07 * 0.06))

	logp3 = model4.log_probability(x, unnormalized=True)
	logp4 = d1.log_probability(x[:2]) + d3.log_probability(x[2:])

	assert_almost_equal(logp3, logp4)
	assert_almost_equal(logp3, numpy.log(0.4 * 0.06))
	with pytest.raises(AssertionError):
		assert_almost_equal(logp1, logp3)

def test_log_probability_unnormalized_str(markov_network_str):
	d1, d2, d3, model1, model2, model3, model4 = markov_network_str
	x = ['1', '0']
	logp1 = model1.log_probability(x, unnormalized=True)
	logp2 = d1.log_probability(x)

	assert_almost_equal(logp1, logp2)
	assert_almost_equal(logp1, numpy.log(0.4))

	x = ['1', '0', '1', '1']
	logp1 = model2.log_probability(x, unnormalized=True)
	logp2 = d1.log_probability(x[:2]) + d2.log_probability(x[1:])

	assert_almost_equal(logp1, logp2)
	assert_almost_equal(logp1, numpy.log(0.4 * 0.03))
	
	x = ['1', '0', '1', '0', '1']
	logp1 = model3.log_probability(x, unnormalized=True)
	logp2 = (d1.log_probability(x[:2]) + d2.log_probability(x[1:4])
		+ d3.log_probability(x[2:]))

	assert_almost_equal(logp1, logp2)
	assert_almost_equal(logp1, numpy.log(0.4 * 0.07 * 0.06))

	logp3 = model4.log_probability(x, unnormalized=True)
	logp4 = d1.log_probability(x[:2]) + d3.log_probability(x[2:])

	assert_almost_equal(logp3, logp4)
	assert_almost_equal(logp3, numpy.log(0.4 * 0.06))
	with pytest.raises(AssertionError):
		assert_almost_equal(logp1, logp3)

def test_log_probability_unnormalized_bool(markov_network_bool):
	d1, d2, d3, model1, model2, model3, model4 = markov_network_bool
	x = [True, False]
	logp1 = model1.log_probability(x, unnormalized=True)
	logp2 = d1.log_probability(x)

	assert_almost_equal(logp1, logp2)
	assert_almost_equal(logp1, numpy.log(0.4))

	x = [True, False, True, True]
	logp1 = model2.log_probability(x, unnormalized=True)
	logp2 = d1.log_probability(x[:2]) + d2.log_probability(x[1:])

	assert_almost_equal(logp1, logp2)
	assert_almost_equal(logp1, numpy.log(0.4 * 0.03))
	
	x = [True, False, True, False, True]
	logp1 = model3.log_probability(x, unnormalized=True)
	logp2 = (d1.log_probability(x[:2]) + d2.log_probability(x[1:4])
		+ d3.log_probability(x[2:]))

	assert_almost_equal(logp1, logp2)
	assert_almost_equal(logp1, numpy.log(0.4 * 0.07 * 0.06))

	logp3 = model4.log_probability(x, unnormalized=True)
	logp4 = d1.log_probability(x[:2]) + d3.log_probability(x[2:])

	assert_almost_equal(logp3, logp4)
	assert_almost_equal(logp3, numpy.log(0.4 * 0.06))
	with pytest.raises(AssertionError):
		assert_almost_equal(logp1, logp3)

def test_log_probability_unnormalized_mixed(markov_network_mixed):
	d1, d2, d3, model1, model2, model3, model4 = markov_network_mixed
	x = [True, 'blue']
	logp1 = model1.log_probability(x, unnormalized=True)
	logp2 = d1.log_probability(x)

	assert_almost_equal(logp1, logp2)
	assert_almost_equal(logp1, numpy.log(0.4))

	x = [True, 'blue', True, 1]
	logp1 = model2.log_probability(x, unnormalized=True)
	logp2 = d1.log_probability(x[:2]) + d2.log_probability(x[1:])

	assert_almost_equal(logp1, logp2)
	assert_almost_equal(logp1, numpy.log(0.4 * 0.03))
	
	x = [1, 'blue', True, 0, 'b']
	logp1 = model3.log_probability(x, unnormalized=True)
	logp2 = (d1.log_probability(x[:2]) + d2.log_probability(x[1:4])
		+ d3.log_probability(x[2:]))

	assert_almost_equal(logp1, logp2)
	assert_almost_equal(logp1, numpy.log(0.4 * 0.07 * 0.06))

	logp3 = model4.log_probability(x, unnormalized=True)
	logp4 = d1.log_probability(x[:2]) + d3.log_probability(x[2:])

	assert_almost_equal(logp3, logp4)
	assert_almost_equal(logp3, numpy.log(0.4 * 0.06))
	with pytest.raises(AssertionError):
		assert_almost_equal(logp1, logp3)

def test_predict_int(markov_network_int):
	d1, d2, d3, model1, model2, model3, model4 = markov_network_int
	assert_array_equal(model1.predict([[1, None]]), [[1, 0]])
	assert_array_equal(model1.predict([[None, 1]]), [[1, 1]])

	assert_array_equal(model2.predict([[1, 0, None, None]]), [[1, 0, 0, 1]])
	assert_array_equal(model2.predict([[0, 0, None, None]]), [[0, 0, 0, 1]])
	assert_array_equal(model2.predict([[None, 1, None, None]]), [[1, 1, 1, 1]])
	assert_array_equal(model2.predict([[None, 1, 1, None]]), [[1, 1, 1, 1]])
	assert_array_equal(model2.predict([[None, 1, None, 0]]), [[1, 1, 0, 0]])

	assert_array_equal(model3.predict([[1, 0, None, 1, None]]), 
		[[1, 0, 0, 1, 1]])
	assert_array_equal(model3.predict([[None, 0, None, 1, None]]), 
		[[1, 0, 0, 1, 1]])
	assert_array_equal(model3.predict([[1, 0, None, 0, 1]]), 
		[[1, 0, 0, 0, 1]])
	assert_array_equal(model3.predict([[None, None, None, None, None]]), 
		[[1, 1, 1, 1, 1]])
	assert_array_equal(model3.predict([[None, None, None, None, 1]]), 
		[[1, 1, 0, 1, 1]])

def test_predict_str(markov_network_str):
	d1, d2, d3, model1, model2, model3, model4 = markov_network_str
	assert_array_equal(model1.predict([['1', None]]), [['1', '0']])
	assert_array_equal(model1.predict([[None, '1']]), [['1', '1']])

	assert_array_equal(model2.predict([['1', '0', None, None]]), [['1', '0', '0', '1']])
	assert_array_equal(model2.predict([['0', '0', None, None]]), [['0', '0', '0', '1']])
	assert_array_equal(model2.predict([[None, '1', None, None]]), [['1', '1', '1', '1']])
	assert_array_equal(model2.predict([[None, '1', '1', None]]), [['1', '1', '1', '1']])
	assert_array_equal(model2.predict([[None, '1', None, '0']]), [['1', '1', '0', '0']])

	assert_array_equal(model3.predict([['1', '0', None, '1', None]]), 
		[['1', '0', '0', '1', '1']])
	assert_array_equal(model3.predict([[None, '0', None, '1', None]]), 
		[['1', '0', '0', '1', '1']])
	assert_array_equal(model3.predict([['1', '0', None, '0', '1']]), 
		[['1', '0', '0', '0', '1']])
	assert_array_equal(model3.predict([[None, None, None, None, None]]), 
		[['1', '1', '1', '1', '1']])
	assert_array_equal(model3.predict([[None, None, None, None, '1']]), 
		[['1', '1', '0', '1', '1']])

def test_predict_bool(markov_network_bool):
	d1, d2, d3, model1, model2, model3, model4 = markov_network_bool
	assert_array_equal(model1.predict([[True, None]]), [[True, False]])
	assert_array_equal(model1.predict([[None, True]]), [[True, True]])

	assert_array_equal(model2.predict([[True, False, None, None]]), [[True, False, False, True]])
	assert_array_equal(model2.predict([[False, False, None, None]]), [[False, False, False, True]])
	assert_array_equal(model2.predict([[None, True, None, None]]), [[True, True, True, True]])
	assert_array_equal(model2.predict([[None, True, True, None]]), [[True, True, True, True]])
	assert_array_equal(model2.predict([[None, True, None, False]]), [[True, True, False, False]])

	assert_array_equal(model3.predict([[True, False, None, True, None]]), 
		[[True, False, False, True, True]])
	assert_array_equal(model3.predict([[None, False, None, True, None]]), 
		[[True, False, False, True, True]])
	assert_array_equal(model3.predict([[True, False, None, False, True]]), 
		[[True, False, False, False, True]])
	assert_array_equal(model3.predict([[None, None, None, None, None]]), 
		[[True, True, True, True, True]])
	assert_array_equal(model3.predict([[None, None, None, None, True]]), 
		[[True, True, False, True, True]])


def test_predict_mixed(markov_network_mixed):
	d1, d2, d3, model1, model2, model3, model4 = markov_network_mixed
	assert_array_equal(model1.predict([[True, None]]), 
		numpy.array([[True, 'blue']], dtype=object))
	assert_array_equal(model1.predict([[None, 'red']]), 
		numpy.array([[True, 'red']], dtype=object))

	assert_array_equal(model2.predict([[True, 'blue', None, None]]), 
		numpy.array([[True, 'blue', False, 1]], dtype=object))
	assert_array_equal(model2.predict([[False, 'blue', None, None]]), 
		numpy.array([[False, 'blue', False, 1]], dtype=object))
	assert_array_equal(model2.predict([[None, 'red', None, None]]),
		numpy.array([[True, 'red', True, 1]], dtype=object))
	assert_array_equal(model2.predict([[None, 'red', True, None]]),
		numpy.array([[True, 'red', True, 1]], dtype=object))
	assert_array_equal(model2.predict([[None, 'red', None, 0]]),
		numpy.array([[True, 'red', False, 0]], dtype=object))

	assert_array_equal(model3.predict([[True, 'blue', None, 1, None]]), 
		numpy.array([[True, 'blue', False, 1, 'b']], dtype=object))

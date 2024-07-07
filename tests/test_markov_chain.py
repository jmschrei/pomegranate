# test_bayes_classifier.py
# Contact: Jacob Schreiber <jmschreiber91@gmail.com>

import numpy
import torch
import pytest

from pomegranate.markov_chain import MarkovChain
from pomegranate.distributions import Categorical
from pomegranate.distributions import ConditionalCategorical

from .distributions._utils import _test_initialization_raises_one_parameter
from .distributions._utils import _test_initialization
from .distributions._utils import _test_predictions
from .distributions._utils import _test_efd_from_summaries
from .distributions._utils import _test_raises

from .tools import assert_raises
from numpy.testing import assert_array_almost_equal


MIN_VALUE = 0
MAX_VALUE = 2

inf = float("inf")


@pytest.fixture
def X():
	return [[[1], [2], [0], [0]],
	     [[0], [0], [1], [1]],
	     [[1], [1], [2], [0]],
	     [[2], [2], [2], [1]],
	     [[0], [1], [0], [0]],
	     [[1], [1], [0], [1]],
	     [[2], [1], [0], [1]],
	     [[1], [0], [2], [1]],
	     [[1], [1], [0], [0]],
	     [[0], [2], [1], [0]],
	     [[0], [0], [0], [0]]]


@pytest.fixture
def w():
	return [[1], [2], [0], [0], [5], [1], [2], [1], [1], [2], [0]]


@pytest.fixture
def model():
	d0 = Categorical([[0.3, 0.1, 0.6]])
	d1 = ConditionalCategorical(
		[[[0.4, 0.2, 0.4],
		  [0.1, 0.2, 0.7],
		  [0.2, 0.5, 0.3]]])

	d2 = ConditionalCategorical(
		[[[[0.3, 0.1, 0.6],
          [0.1, 0.4, 0.5],
          [0.6, 0.2, 0.2]],

         [[0.2, 0.3, 0.5],
          [0.7, 0.1, 0.2],
          [0.8, 0.1, 0.1]],

         [[0.6, 0.3, 0.1],
          [0.4, 0.1, 0.5],
          [0.6, 0.2, 0.2]]]])

	return MarkovChain([d0, d1, d2])


###


def test_initialization_raises():
	d = [Categorical(),  ConditionalCategorical()]

	assert_raises(ValueError, MarkovChain)
	assert_raises(ValueError, MarkovChain, d, [0.2, 0.2, 0.6])
	assert_raises(ValueError, MarkovChain, d, [0.2, 1.0])
	assert_raises(ValueError, MarkovChain, d, [-0.2, 1.2])

	assert_raises(ValueError, MarkovChain, Categorical)
	assert_raises(ValueError, MarkovChain, d, inertia=-0.4)
	assert_raises(ValueError, MarkovChain, d, inertia=1.2)
	assert_raises(ValueError, MarkovChain, d, inertia=1.2, frozen="true")
	assert_raises(ValueError, MarkovChain, d, inertia=1.2, frozen=3)


def test_initialize(X):
	d = [Categorical(), ConditionalCategorical()]
	model = MarkovChain(d)
	assert model.d is None
	assert model.k == 1
	assert model._initialized == False

	model._initialize(2, ((3,)))
	assert model._initialized == True
	assert model.d == 2
	assert model.k == 1


###


@pytest.mark.sample
def test_sample(model):
	torch.manual_seed(0)

	X = model.sample(1)
	assert_array_almost_equal(X, [[[2], [1], [2]]])

	X = model.sample(5)
	assert_array_almost_equal(X, 
		[[[1], [2], [0]],
         [[2], [2], [2]],
         [[0], [2], [0]],
         [[2], [1], [0]],
         [[0], [0], [2]]], 3)


###



def test_log_probability(model, X):
	logp = model.log_probability(X)
	assert_array_almost_equal(logp, [-3.3932, -5.3391, -5.7446, -4.9337, 
		-6.7254, -5.4727, -3.3242, -6.9078, -5.8781, -4.646 , -4.5282], 4)


def test_log_probability_raises(model, X):
	assert_raises(ValueError, model.log_probability, [X])
	assert_raises(ValueError, model.log_probability, X[0])
	assert_raises((ValueError, TypeError), model.log_probability, X[0][0])

	assert_raises(ValueError, model.log_probability, 
		[[MIN_VALUE-0.1 for i in range(len(X[0]))]])
	assert_raises(ValueError, model.log_probability, 
		[[MAX_VALUE+0.1 for i in range(len(X[0]))]])


def test_probability(model, X):
	p = model.probability(X)
	assert_array_almost_equal(p, [0.0336, 0.0048, 0.0032, 0.0072, 0.0012, 
		0.0042, 0.036, 0.001, 0.0028, 0.0096, 0.0108], 4)


def test_probability_raises(model, X):
	assert_raises(ValueError, model.probability, [X])
	assert_raises(ValueError, model.probability, X[0])
	assert_raises((ValueError, TypeError), model.probability, X[0][0])

	assert_raises(ValueError, model.probability, 
		[[MIN_VALUE-0.1 for i in range(len(X[0]))]])
	assert_raises(ValueError, model.probability, 
		[[MAX_VALUE+0.1 for i in range(len(X[0]))]])


###


def test_partial_summarize(model, X):
	model.summarize(X[:4])
	assert_array_almost_equal(model.distributions[0]._xw_sum, [[1., 2., 1.]])
	assert_array_almost_equal(model.distributions[1]._xw_sum[0], 
		[[1., 0., 0.],
         [0., 1., 1.],
         [0., 0., 1.]])
	assert_array_almost_equal(model.distributions[2]._xw_sum[0],
		[[[0., 1., 0.],
          [0., 1., 0.],
          [0., 0., 0.]],

         [[0., 0., 0.],
          [0., 0., 1.],
          [2., 0., 0.]],

         [[1., 0., 0.],
          [0., 0., 0.],
          [0., 1., 1.]]])

	model.summarize(X[4:])
	assert_array_almost_equal(model.distributions[0]._xw_sum, [[4., 5., 2.]])
	assert_array_almost_equal(model.distributions[1]._xw_sum[0],
		[[2., 1., 1.],
         [1., 3., 1.],
         [0., 1., 1.]])
	assert_array_almost_equal(model.distributions[2]._xw_sum[0],
		[[[2., 1., 0.],
          [1., 1., 0.],
          [0., 2., 0.]],

         [[2., 2., 1.],
          [2., 0., 1.],
          [2., 0., 0.]],

         [[1., 0., 0.],
          [2., 0., 0.],
          [0., 1., 1.]]])


	model = MarkovChain(k=2)
	model.summarize(X[:4])
	assert_array_almost_equal(model.distributions[0]._xw_sum, [[1., 2., 1.]])
	assert_array_almost_equal(model.distributions[1]._xw_sum[0], 
		[[1., 0., 0.],
         [0., 1., 1.],
         [0., 0., 1.]])
	assert_array_almost_equal(model.distributions[2]._xw_sum[0],
		[[[0., 1., 0.],
          [0., 1., 0.],
          [0., 0., 0.]],

         [[0., 0., 0.],
          [0., 0., 1.],
          [2., 0., 0.]],

         [[1., 0., 0.],
          [0., 0., 0.],
          [0., 1., 1.]]])


	model.summarize(X[4:])
	assert_array_almost_equal(model.distributions[0]._xw_sum, [[4., 5., 2.]])
	assert_array_almost_equal(model.distributions[1]._xw_sum[0],
		[[2., 1., 1.],
         [1., 3., 1.],
         [0., 1., 1.]])
	assert_array_almost_equal(model.distributions[2]._xw_sum[0],
		[[[2., 1., 0.],
          [1., 1., 0.],
          [0., 2., 0.]],

         [[2., 2., 1.],
          [2., 0., 1.],
          [2., 0., 0.]],

         [[1., 0., 0.],
          [2., 0., 0.],
          [0., 1., 1.]]])


def test_full_summarize(model, X):
	model.summarize(X)
	assert_array_almost_equal(model.distributions[0]._xw_sum, [[4., 5., 2.]])
	assert_array_almost_equal(model.distributions[1]._xw_sum[0],
		[[2., 1., 1.],
         [1., 3., 1.],
         [0., 1., 1.]])
	assert_array_almost_equal(model.distributions[2]._xw_sum[0],
		[[[2., 1., 0.],
          [1., 1., 0.],
          [0., 2., 0.]],

         [[2., 2., 1.],
          [2., 0., 1.],
          [2., 0., 0.]],

         [[1., 0., 0.],
          [2., 0., 0.],
          [0., 1., 1.]]])


	model = MarkovChain(k=2)
	model.summarize(X)
	assert_array_almost_equal(model.distributions[0]._xw_sum, [[4., 5., 2.]])
	assert_array_almost_equal(model.distributions[1]._xw_sum[0],
		[[2., 1., 1.],
         [1., 3., 1.],
         [0., 1., 1.]])
	assert_array_almost_equal(model.distributions[2]._xw_sum[0],
		[[[2., 1., 0.],
          [1., 1., 0.],
          [0., 2., 0.]],

         [[2., 2., 1.],
          [2., 0., 1.],
          [2., 0., 0.]],

         [[1., 0., 0.],
          [2., 0., 0.],
          [0., 1., 1.]]])


def test_summarize_weighted(model, X, w):
	model.summarize(X, sample_weight=w)
	assert_array_almost_equal(model.distributions[0]._xw_sum, [[9., 4., 2.]])
	assert_array_almost_equal(model.distributions[1]._xw_sum[0],
		[[2., 5., 2.],
         [1., 2., 1.],
         [0., 2., 0.]])
	assert_array_almost_equal(model.distributions[2]._xw_sum[0],
		[[[0., 2., 0.],
          [5., 2., 0.],
          [0., 3., 0.]],

         [[6., 3., 1.],
          [2., 0., 0.],
          [1., 0., 0.]],

         [[1., 0., 0.],
          [4., 0., 0.],
          [0., 0., 0.]]])

	model = MarkovChain(k=2)
	model.summarize(X, sample_weight=w)
	assert_array_almost_equal(model.distributions[0]._xw_sum, [[9., 4., 2.]])
	assert_array_almost_equal(model.distributions[1]._xw_sum[0],
		[[2., 5., 2.],
         [1., 2., 1.],
         [0., 2., 0.]])
	assert_array_almost_equal(model.distributions[2]._xw_sum[0],
		[[[0., 2., 0.],
          [5., 2., 0.],
          [0., 3., 0.]],

         [[6., 3., 1.],
          [2., 0., 0.],
          [1., 0., 0.]],

         [[1., 0., 0.],
          [4., 0., 0.],
          [0., 0., 0.]]])


def test_summarize_weighted_flat(model, X, w):
	w = numpy.array(w)[:,0] 

	model.summarize(X, sample_weight=w)
	assert_array_almost_equal(model.distributions[0]._xw_sum, [[9., 4., 2.]])
	assert_array_almost_equal(model.distributions[1]._xw_sum[0],
		[[2., 5., 2.],
         [1., 2., 1.],
         [0., 2., 0.]])
	assert_array_almost_equal(model.distributions[2]._xw_sum[0],
		[[[0., 2., 0.],
          [5., 2., 0.],
          [0., 3., 0.]],

         [[6., 3., 1.],
          [2., 0., 0.],
          [1., 0., 0.]],

         [[1., 0., 0.],
          [4., 0., 0.],
          [0., 0., 0.]]])

	model = MarkovChain(k=2)
	model.summarize(X, sample_weight=w)
	assert_array_almost_equal(model.distributions[0]._xw_sum, [[9., 4., 2.]])
	assert_array_almost_equal(model.distributions[1]._xw_sum[0],
		[[2., 5., 2.],
         [1., 2., 1.],
         [0., 2., 0.]])
	assert_array_almost_equal(model.distributions[2]._xw_sum[0],
		[[[0., 2., 0.],
          [5., 2., 0.],
          [0., 3., 0.]],

         [[6., 3., 1.],
          [2., 0., 0.],
          [1., 0., 0.]],

         [[1., 0., 0.],
          [4., 0., 0.],
          [0., 0., 0.]]])


def test_summarize_raises(model, X, w):
	assert_raises(ValueError, model.summarize, [X])
	assert_raises(ValueError, model.summarize, X[0])
	assert_raises((ValueError, TypeError), model.summarize, X[0][0])
	assert_raises(ValueError, model.summarize, 
		[[-0.1 for i in range(3)] for x in X])

	assert_raises(ValueError, model.summarize, [X], w)
	assert_raises(ValueError, model.summarize, X, [w])
	assert_raises(ValueError, model.summarize, [X], [w])
	assert_raises(ValueError, model.summarize, X[:len(X)-1], w)
	assert_raises(ValueError, model.summarize, X, w[:len(w)-1])


def test_from_summaries(model, X):
	model.summarize(X)
	model.from_summaries()

	assert_array_almost_equal(model.distributions[0]._xw_sum, [[0., 0., 0]])
	assert_array_almost_equal(model.distributions[1]._xw_sum[0],
		[[0., 0., 0.],
         [0., 0., 0.],
         [0., 0., 0.]])
	assert_array_almost_equal(model.distributions[2]._xw_sum[0],
		[[[0., 0., 0.],
          [0., 0., 0.],
          [0., 0., 0.]],
 
         [[0., 0., 0.],
          [0., 0., 0.],
          [0., 0., 0.]],

         [[0., 0., 0.],
          [0., 0., 0.],
          [0., 0., 0.]]])

	assert_array_almost_equal(model.distributions[0].probs, 
		[[0.363636, 0.454545, 0.181818]])
	assert_array_almost_equal(model.distributions[1].probs[0],
		[[0.5 , 0.25, 0.25],
         [0.2 , 0.6 , 0.2 ],
         [0.  , 0.5 , 0.5 ]])
	assert_array_almost_equal(model.distributions[2].probs[0],
		[[[0.6667, 0.3333, 0.0000],
          [0.5000, 0.5000, 0.0000],
          [0.0000, 1.0000, 0.0000]],

         [[0.4000, 0.4000, 0.2000],
          [0.6667, 0.0000, 0.3333],
          [1.0000, 0.0000, 0.0000]],

         [[1.0000, 0.0000, 0.0000],
          [1.0000, 0.0000, 0.0000],
          [0.0000, 0.5000, 0.5000]]], 4)

	assert_array_almost_equal(model.distributions[0]._log_probs,
		[[-1.011601, -0.788457, -1.704748]])
	assert_array_almost_equal(model.distributions[1]._log_probs[0],
		[[-0.693147, -1.386294, -1.386294],
         [-1.609438, -0.510826, -1.609438],
         [-inf, -0.693147, -0.693147]])
	assert_array_almost_equal(model.distributions[2]._log_probs[0],
		[[[-0.4055, -1.0986,    -inf],
          [-0.6931, -0.6931,    -inf],
          [   -inf,  0.0000,    -inf]],

         [[-0.9163, -0.9163, -1.6094],
          [-0.4055,    -inf, -1.0986],
          [ 0.0000,    -inf,    -inf]],

         [[ 0.0000,    -inf,    -inf],
          [ 0.0000,    -inf,    -inf],
          [   -inf, -0.6931, -0.6931]]], 4)


	X_ = numpy.array(X)
	d = Categorical().fit(X_[:, 0])
	d2 = ConditionalCategorical().fit(X_[:, :2])

	assert_array_almost_equal(d.probs, model.distributions[0].probs)
	assert_array_almost_equal(d2.probs[0], model.distributions[1].probs[0])

	model = MarkovChain([Categorical(), ConditionalCategorical(),
		ConditionalCategorical()])
	model.summarize(X)
	model.from_summaries()

	assert_array_almost_equal(model.distributions[0]._xw_sum, [[0., 0., 0]])
	assert_array_almost_equal(model.distributions[1]._xw_sum[0],
		[[0., 0., 0.],
         [0., 0., 0.],
         [0., 0., 0.]])
	assert_array_almost_equal(model.distributions[2]._xw_sum[0],
		[[[0., 0., 0.],
          [0., 0., 0.],
          [0., 0., 0.]],
 
         [[0., 0., 0.],
          [0., 0., 0.],
          [0., 0., 0.]],

         [[0., 0., 0.],
          [0., 0., 0.],
          [0., 0., 0.]]])

	assert_array_almost_equal(model.distributions[0].probs, 
		[[0.363636, 0.454545, 0.181818]])
	assert_array_almost_equal(model.distributions[1].probs[0],
		[[0.5 , 0.25, 0.25],
         [0.2 , 0.6 , 0.2 ],
         [0.  , 0.5 , 0.5 ]])
	assert_array_almost_equal(model.distributions[2].probs[0],
		[[[0.6667, 0.3333, 0.0000],
          [0.5000, 0.5000, 0.0000],
          [0.0000, 1.0000, 0.0000]],

         [[0.4000, 0.4000, 0.2000],
          [0.6667, 0.0000, 0.3333],
          [1.0000, 0.0000, 0.0000]],

         [[1.0000, 0.0000, 0.0000],
          [1.0000, 0.0000, 0.0000],
          [0.0000, 0.5000, 0.5000]]], 4)

	assert_array_almost_equal(model.distributions[0]._log_probs,
		[[-1.011601, -0.788457, -1.704748]])
	assert_array_almost_equal(model.distributions[1]._log_probs[0],
		[[-0.693147, -1.386294, -1.386294],
         [-1.609438, -0.510826, -1.609438],
         [-inf, -0.693147, -0.693147]])
	assert_array_almost_equal(model.distributions[2]._log_probs[0],
		[[[-0.4055, -1.0986,    -inf],
          [-0.6931, -0.6931,    -inf],
          [   -inf,  0.0000,    -inf]],

         [[-0.9163, -0.9163, -1.6094],
          [-0.4055,    -inf, -1.0986],
          [ 0.0000,    -inf,    -inf]],

         [[ 0.0000,    -inf,    -inf],
          [ 0.0000,    -inf,    -inf],
          [   -inf, -0.6931, -0.6931]]], 4)


def test_from_summaries_weighted(model, X, w):
	model.summarize(X, sample_weight=w)
	model.from_summaries()

	assert_array_almost_equal(model.distributions[0]._xw_sum, [[0., 0., 0]])
	assert_array_almost_equal(model.distributions[1]._xw_sum[0],
		[[0., 0., 0.],
         [0., 0., 0.],
         [0., 0., 0.]])
	assert_array_almost_equal(model.distributions[2]._xw_sum[0],
		[[[0., 0., 0.],
          [0., 0., 0.],
          [0., 0., 0.]],
 
         [[0., 0., 0.],
          [0., 0., 0.],
          [0., 0., 0.]],

         [[0., 0., 0.],
          [0., 0., 0.],
          [0., 0., 0.]]])

	assert_array_almost_equal(model.distributions[0].probs, 
		[[0.6     , 0.266667, 0.133333]])
	assert_array_almost_equal(model.distributions[1].probs[0],
		[[0.222222, 0.555556, 0.222222],
         [0.25    , 0.5     , 0.25    ], 
         [0.      , 1.      , 0.      ]])
	assert_array_almost_equal(model.distributions[2].probs[0],
		[[[0.0000, 1.0000, 0.0000],
          [0.7143, 0.2857, 0.0000],
          [0.0000, 1.0000, 0.0000]],

         [[0.6000, 0.3000, 0.1000],
          [1.0000, 0.0000, 0.0000],
          [1.0000, 0.0000, 0.0000]],

         [[1.0000, 0.0000, 0.0000],
          [1.0000, 0.0000, 0.0000],
          [0.3333, 0.3333, 0.3333]]], 4)

	assert_array_almost_equal(model.distributions[0]._log_probs,
		[[-0.510826, -1.321756, -2.014903]])
	assert_array_almost_equal(model.distributions[1]._log_probs[0],
		[[-1.504077, -0.587787, -1.504077],
         [-1.386294, -0.693147, -1.386294],
         [     -inf,  0.      ,      -inf]])
	assert_array_almost_equal(model.distributions[2]._log_probs[0],
		[[[   -inf,  0.0000,    -inf],
          [-0.3365, -1.2528,    -inf],
          [   -inf,  0.0000,    -inf]],

         [[-0.5108, -1.2040, -2.3026],
          [ 0.0000,    -inf,    -inf],
          [ 0.0000,    -inf,    -inf]],

         [[ 0.0000,    -inf,    -inf],
          [ 0.0000,    -inf,    -inf],
          [-1.0986, -1.0986, -1.0986]]], 4)


	X_ = numpy.array(X)
	d = Categorical().fit(X_[:, 0], sample_weight=w)
	d2 = ConditionalCategorical().fit(X_[:, :2], sample_weight=w)

	assert_array_almost_equal(d.probs, model.distributions[0].probs)
	assert_array_almost_equal(d2.probs[0], model.distributions[1].probs[0])


	model = MarkovChain([Categorical(), ConditionalCategorical(),
		ConditionalCategorical()])
	model.summarize(X, sample_weight=w)
	model.from_summaries()

	assert_array_almost_equal(model.distributions[0]._xw_sum, [[0., 0., 0]])
	assert_array_almost_equal(model.distributions[1]._xw_sum[0],
		[[0., 0., 0.],
         [0., 0., 0.],
         [0., 0., 0.]])
	assert_array_almost_equal(model.distributions[2]._xw_sum[0],
		[[[0., 0., 0.],
          [0., 0., 0.],
          [0., 0., 0.]],
 
         [[0., 0., 0.],
          [0., 0., 0.],
          [0., 0., 0.]],

         [[0., 0., 0.],
          [0., 0., 0.],
          [0., 0., 0.]]])

	assert_array_almost_equal(model.distributions[0].probs, 
		[[0.6     , 0.266667, 0.133333]])
	assert_array_almost_equal(model.distributions[1].probs[0],
		[[0.222222, 0.555556, 0.222222],
         [0.25    , 0.5     , 0.25    ], 
         [0.      , 1.      , 0.      ]])
	assert_array_almost_equal(model.distributions[2].probs[0],
		[[[0.0000, 1.0000, 0.0000],
          [0.7143, 0.2857, 0.0000],
          [0.0000, 1.0000, 0.0000]],

         [[0.6000, 0.3000, 0.1000],
          [1.0000, 0.0000, 0.0000],
          [1.0000, 0.0000, 0.0000]],

         [[1.0000, 0.0000, 0.0000],
          [1.0000, 0.0000, 0.0000],
          [0.3333, 0.3333, 0.3333]]], 4)

	assert_array_almost_equal(model.distributions[0]._log_probs,
		[[-0.510826, -1.321756, -2.014903]])
	assert_array_almost_equal(model.distributions[1]._log_probs[0],
		[[-1.504077, -0.587787, -1.504077],
         [-1.386294, -0.693147, -1.386294],
         [     -inf,  0.      ,      -inf]])
	assert_array_almost_equal(model.distributions[2]._log_probs[0],
		[[[   -inf,  0.0000,    -inf],
          [-0.3365, -1.2528,    -inf],
          [   -inf,  0.0000,    -inf]],

         [[-0.5108, -1.2040, -2.3026],
          [ 0.0000,    -inf,    -inf],
          [ 0.0000,    -inf,    -inf]],

         [[ 0.0000,    -inf,    -inf],
          [ 0.0000,    -inf,    -inf],
          [-1.0986, -1.0986, -1.0986]]], 4)


def test_from_summaries_frozen(model, X):
	p0 = [[0.3, 0.1, 0.6]]

	p1 = [[[0.4, 0.2, 0.4],
		   [0.1, 0.2, 0.7],
		   [0.2, 0.5, 0.3]]]

	p2 = [[[[0.3, 0.1, 0.6],
            [0.1, 0.4, 0.5],
            [0.6, 0.2, 0.2]],

           [[0.2, 0.3, 0.5],
            [0.7, 0.1, 0.2],
            [0.8, 0.1, 0.1]],

           [[0.6, 0.3, 0.1],
            [0.4, 0.1, 0.5],
            [0.6, 0.2, 0.2]]]]

	d0 = Categorical(p0)
	d1 = ConditionalCategorical(p1)
	d2 = ConditionalCategorical(p2)

	model = MarkovChain([d0, d1, d2], frozen=True)
	model.summarize(X)
	model.from_summaries()

	assert_array_almost_equal(model.distributions[0].probs, p0)
	assert_array_almost_equal(model.distributions[1].probs[0], p1[0])
	assert_array_almost_equal(model.distributions[2].probs[0], p2[0])

	d2 = ConditionalCategorical(p2, frozen=True)

	model = MarkovChain([d0, d1, d2])
	model.summarize(X)
	model.from_summaries()

	assert_raises(AssertionError, assert_array_almost_equal, 
		model.distributions[0].probs, p0)
	assert_raises(AssertionError, assert_array_almost_equal, 
		model.distributions[1].probs[0], p1[0])
	assert_array_almost_equal(model.distributions[2].probs[0], p2[0])

	d0 = Categorical(p0, frozen=True)
	d2 = ConditionalCategorical(p2)

	model = MarkovChain([d0, d1, d2])
	model.summarize(X)
	model.from_summaries()

	assert_array_almost_equal(model.distributions[0].probs, p0)
	assert_raises(AssertionError, assert_array_almost_equal, 
		model.distributions[1].probs[0], p1[0])
	assert_raises(AssertionError, assert_array_almost_equal,
		model.distributions[2].probs[0], p2[0])


def test_fit(model, X):
	model.fit(X)

	assert_array_almost_equal(model.distributions[0]._xw_sum, [[0., 0., 0]])
	assert_array_almost_equal(model.distributions[1]._xw_sum[0],
		[[0., 0., 0.],
         [0., 0., 0.],
         [0., 0., 0.]])
	assert_array_almost_equal(model.distributions[2]._xw_sum[0],
		[[[0., 0., 0.],
          [0., 0., 0.],
          [0., 0., 0.]],
 
         [[0., 0., 0.],
          [0., 0., 0.],
          [0., 0., 0.]],

         [[0., 0., 0.],
          [0., 0., 0.],
          [0., 0., 0.]]])

	assert_array_almost_equal(model.distributions[0].probs, 
		[[0.363636, 0.454545, 0.181818]])
	assert_array_almost_equal(model.distributions[1].probs[0],
		[[0.5 , 0.25, 0.25],
         [0.2 , 0.6 , 0.2 ],
         [0.  , 0.5 , 0.5 ]])
	assert_array_almost_equal(model.distributions[2].probs[0],
		[[[0.6667, 0.3333, 0.0000],
          [0.5000, 0.5000, 0.0000],
          [0.0000, 1.0000, 0.0000]],

         [[0.4000, 0.4000, 0.2000],
          [0.6667, 0.0000, 0.3333],
          [1.0000, 0.0000, 0.0000]],

         [[1.0000, 0.0000, 0.0000],
          [1.0000, 0.0000, 0.0000],
          [0.0000, 0.5000, 0.5000]]], 4)

	assert_array_almost_equal(model.distributions[0]._log_probs,
		[[-1.011601, -0.788457, -1.704748]])
	assert_array_almost_equal(model.distributions[1]._log_probs[0],
		[[-0.693147, -1.386294, -1.386294],
         [-1.609438, -0.510826, -1.609438],
         [-inf, -0.693147, -0.693147]])
	assert_array_almost_equal(model.distributions[2]._log_probs[0],
		[[[-0.4055, -1.0986,    -inf],
          [-0.6931, -0.6931,    -inf],
          [   -inf,  0.0000,    -inf]],

         [[-0.9163, -0.9163, -1.6094],
          [-0.4055,    -inf, -1.0986],
          [ 0.0000,    -inf,    -inf]],

         [[ 0.0000,    -inf,    -inf],
          [ 0.0000,    -inf,    -inf],
          [   -inf, -0.6931, -0.6931]]], 4)


	X_ = numpy.array(X)
	d = Categorical().fit(X_[:, 0])
	d2 = ConditionalCategorical().fit(X_[:, :2])

	assert_array_almost_equal(d.probs, model.distributions[0].probs)
	assert_array_almost_equal(d2.probs[0], model.distributions[1].probs[0])

	model = MarkovChain([Categorical(), ConditionalCategorical(),
		ConditionalCategorical()])
	model.fit(X)

	assert_array_almost_equal(model.distributions[0]._xw_sum, [[0., 0., 0]])
	assert_array_almost_equal(model.distributions[1]._xw_sum[0],
		[[0., 0., 0.],
         [0., 0., 0.],
         [0., 0., 0.]])
	assert_array_almost_equal(model.distributions[2]._xw_sum[0],
		[[[0., 0., 0.],
          [0., 0., 0.],
          [0., 0., 0.]],
 
         [[0., 0., 0.],
          [0., 0., 0.],
          [0., 0., 0.]],

         [[0., 0., 0.],
          [0., 0., 0.],
          [0., 0., 0.]]])

	assert_array_almost_equal(model.distributions[0].probs, 
		[[0.363636, 0.454545, 0.181818]])
	assert_array_almost_equal(model.distributions[1].probs[0],
		[[0.5 , 0.25, 0.25],
         [0.2 , 0.6 , 0.2 ],
         [0.  , 0.5 , 0.5 ]])
	assert_array_almost_equal(model.distributions[2].probs[0],
		[[[0.6667, 0.3333, 0.0000],
          [0.5000, 0.5000, 0.0000],
          [0.0000, 1.0000, 0.0000]],

         [[0.4000, 0.4000, 0.2000],
          [0.6667, 0.0000, 0.3333],
          [1.0000, 0.0000, 0.0000]],

         [[1.0000, 0.0000, 0.0000],
          [1.0000, 0.0000, 0.0000],
          [0.0000, 0.5000, 0.5000]]], 4)

	assert_array_almost_equal(model.distributions[0]._log_probs,
		[[-1.011601, -0.788457, -1.704748]])
	assert_array_almost_equal(model.distributions[1]._log_probs[0],
		[[-0.693147, -1.386294, -1.386294],
         [-1.609438, -0.510826, -1.609438],
         [-inf, -0.693147, -0.693147]])
	assert_array_almost_equal(model.distributions[2]._log_probs[0],
		[[[-0.4055, -1.0986,    -inf],
          [-0.6931, -0.6931,    -inf],
          [   -inf,  0.0000,    -inf]],

         [[-0.9163, -0.9163, -1.6094],
          [-0.4055,    -inf, -1.0986],
          [ 0.0000,    -inf,    -inf]],

         [[ 0.0000,    -inf,    -inf],
          [ 0.0000,    -inf,    -inf],
          [   -inf, -0.6931, -0.6931]]], 4)


def test_fit_k(model, X):
	numpy.random.seed(137)
	seq_data = numpy.random.randint(0, 10, (1,10,1))


	model = MarkovChain(k=1)
	model.fit(seq_data)

	assert_array_almost_equal(model.distributions[1].probs[0],
	   [[0.0000, 1.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
         0.0000],
        [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
         1.0000],
        [0.1000, 0.1000, 0.1000, 0.1000, 0.1000, 0.1000, 0.1000, 0.1000, 0.1000,
         0.1000],
        [1.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
         0.0000],
        [0.1000, 0.1000, 0.1000, 0.1000, 0.1000, 0.1000, 0.1000, 0.1000, 0.1000,
         0.1000],
        [0.1000, 0.1000, 0.1000, 0.1000, 0.1000, 0.1000, 0.1000, 0.1000, 0.1000,
         0.1000],
        [0.0000, 0.0000, 0.0000, 0.3333, 0.0000, 0.0000, 0.3333, 0.3333, 0.0000,
         0.0000],
        [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
         1.0000],
        [0.1000, 0.1000, 0.1000, 0.1000, 0.1000, 0.1000, 0.1000, 0.1000, 0.1000,
         0.1000],
        [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.5000, 0.5000, 0.0000,
         0.0000]], 4)


def test_fit_weighted(model, X, w):
	model.fit(X, sample_weight=w)

	assert_array_almost_equal(model.distributions[0]._xw_sum, [[0., 0., 0]])
	assert_array_almost_equal(model.distributions[1]._xw_sum[0],
		[[0., 0., 0.],
         [0., 0., 0.],
         [0., 0., 0.]])
	assert_array_almost_equal(model.distributions[2]._xw_sum[0],
		[[[0., 0., 0.],
          [0., 0., 0.],
          [0., 0., 0.]],
 
         [[0., 0., 0.],
          [0., 0., 0.],
          [0., 0., 0.]],

         [[0., 0., 0.],
          [0., 0., 0.],
          [0., 0., 0.]]])

	assert_array_almost_equal(model.distributions[0].probs, 
		[[0.6     , 0.266667, 0.133333]])
	assert_array_almost_equal(model.distributions[1].probs[0],
		[[0.222222, 0.555556, 0.222222],
         [0.25    , 0.5     , 0.25    ], 
         [0.      , 1.      , 0.      ]])
	assert_array_almost_equal(model.distributions[2].probs[0],
		[[[0.0000, 1.0000, 0.0000],
          [0.7143, 0.2857, 0.0000],
          [0.0000, 1.0000, 0.0000]],

         [[0.6000, 0.3000, 0.1000],
          [1.0000, 0.0000, 0.0000],
          [1.0000, 0.0000, 0.0000]],

         [[1.0000, 0.0000, 0.0000],
          [1.0000, 0.0000, 0.0000],
          [0.3333, 0.3333, 0.3333]]], 4)

	assert_array_almost_equal(model.distributions[0]._log_probs,
		[[-0.510826, -1.321756, -2.014903]])
	assert_array_almost_equal(model.distributions[1]._log_probs[0],
		[[-1.504077, -0.587787, -1.504077],
         [-1.386294, -0.693147, -1.386294],
         [     -inf,  0.      ,      -inf]])
	assert_array_almost_equal(model.distributions[2]._log_probs[0],
		[[[   -inf,  0.0000,    -inf],
          [-0.3365, -1.2528,    -inf],
          [   -inf,  0.0000,    -inf]],

         [[-0.5108, -1.2040, -2.3026],
          [ 0.0000,    -inf,    -inf],
          [ 0.0000,    -inf,    -inf]],

         [[ 0.0000,    -inf,    -inf],
          [ 0.0000,    -inf,    -inf],
          [-1.0986, -1.0986, -1.0986]]], 4)


	X_ = numpy.array(X)
	d = Categorical().fit(X_[:, 0], sample_weight=w)
	d2 = ConditionalCategorical().fit(X_[:, :2], sample_weight=w)

	assert_array_almost_equal(d.probs, model.distributions[0].probs)
	assert_array_almost_equal(d2.probs[0], model.distributions[1].probs[0])


	model = MarkovChain([Categorical(), ConditionalCategorical(),
		ConditionalCategorical()])
	model.fit(X, sample_weight=w)

	assert_array_almost_equal(model.distributions[0]._xw_sum, [[0., 0., 0]])
	assert_array_almost_equal(model.distributions[1]._xw_sum[0],
		[[0., 0., 0.],
         [0., 0., 0.],
         [0., 0., 0.]])
	assert_array_almost_equal(model.distributions[2]._xw_sum[0],
		[[[0., 0., 0.],
          [0., 0., 0.],
          [0., 0., 0.]],
 
         [[0., 0., 0.],
          [0., 0., 0.],
          [0., 0., 0.]],

         [[0., 0., 0.],
          [0., 0., 0.],
          [0., 0., 0.]]])

	assert_array_almost_equal(model.distributions[0].probs, 
		[[0.6     , 0.266667, 0.133333]])
	assert_array_almost_equal(model.distributions[1].probs[0],
		[[0.222222, 0.555556, 0.222222],
         [0.25    , 0.5     , 0.25    ], 
         [0.      , 1.      , 0.      ]])
	assert_array_almost_equal(model.distributions[2].probs[0],
		[[[0.0000, 1.0000, 0.0000],
          [0.7143, 0.2857, 0.0000],
          [0.0000, 1.0000, 0.0000]],

         [[0.6000, 0.3000, 0.1000],
          [1.0000, 0.0000, 0.0000],
          [1.0000, 0.0000, 0.0000]],

         [[1.0000, 0.0000, 0.0000],
          [1.0000, 0.0000, 0.0000],
          [0.3333, 0.3333, 0.3333]]], 4)

	assert_array_almost_equal(model.distributions[0]._log_probs,
		[[-0.510826, -1.321756, -2.014903]])
	assert_array_almost_equal(model.distributions[1]._log_probs[0],
		[[-1.504077, -0.587787, -1.504077],
         [-1.386294, -0.693147, -1.386294],
         [     -inf,  0.      ,      -inf]])
	assert_array_almost_equal(model.distributions[2]._log_probs[0],
		[[[   -inf,  0.0000,    -inf],
          [-0.3365, -1.2528,    -inf],
          [   -inf,  0.0000,    -inf]],

         [[-0.5108, -1.2040, -2.3026],
          [ 0.0000,    -inf,    -inf],
          [ 0.0000,    -inf,    -inf]],

         [[ 0.0000,    -inf,    -inf],
          [ 0.0000,    -inf,    -inf],
          [-1.0986, -1.0986, -1.0986]]], 4)


def test_fit_chain(X):
	d0 = Categorical([[0.3, 0.1, 0.6]])
	d1 = ConditionalCategorical(
		[[[0.4, 0.2, 0.4],
		  [0.1, 0.2, 0.7],
		  [0.2, 0.5, 0.3]]])

	d2 = ConditionalCategorical(
		[[[[0.3, 0.1, 0.6],
           [0.1, 0.4, 0.5],
           [0.6, 0.2, 0.2]],

          [[0.2, 0.3, 0.5],
           [0.7, 0.1, 0.2],
           [0.8, 0.1, 0.1]],

          [[0.6, 0.3, 0.1],
           [0.4, 0.1, 0.5],
           [0.6, 0.2, 0.2]]]])

	model = MarkovChain([d0, d1, d2]).fit(X)

	assert_array_almost_equal(model.distributions[0]._xw_sum, [[0., 0., 0]])
	assert_array_almost_equal(model.distributions[1]._xw_sum[0],
		[[0., 0., 0.],
         [0., 0., 0.],
         [0., 0., 0.]])
	assert_array_almost_equal(model.distributions[2]._xw_sum[0],
		[[[0., 0., 0.],
          [0., 0., 0.],
          [0., 0., 0.]],
 
         [[0., 0., 0.],
          [0., 0., 0.],
          [0., 0., 0.]],

         [[0., 0., 0.],
          [0., 0., 0.],
          [0., 0., 0.]]])

	assert_array_almost_equal(model.distributions[0].probs, 
		[[0.363636, 0.454545, 0.181818]])
	assert_array_almost_equal(model.distributions[1].probs[0],
		[[0.5 , 0.25, 0.25],
         [0.2 , 0.6 , 0.2 ],
         [0.  , 0.5 , 0.5 ]])
	assert_array_almost_equal(model.distributions[2].probs[0],
		[[[0.6667, 0.3333, 0.0000],
          [0.5000, 0.5000, 0.0000],
          [0.0000, 1.0000, 0.0000]],

         [[0.4000, 0.4000, 0.2000],
          [0.6667, 0.0000, 0.3333],
          [1.0000, 0.0000, 0.0000]],

         [[1.0000, 0.0000, 0.0000],
          [1.0000, 0.0000, 0.0000],
          [0.0000, 0.5000, 0.5000]]], 4)

	assert_array_almost_equal(model.distributions[0]._log_probs,
		[[-1.011601, -0.788457, -1.704748]])
	assert_array_almost_equal(model.distributions[1]._log_probs[0],
		[[-0.693147, -1.386294, -1.386294],
         [-1.609438, -0.510826, -1.609438],
         [-inf, -0.693147, -0.693147]])
	assert_array_almost_equal(model.distributions[2]._log_probs[0],
		[[[-0.4055, -1.0986,    -inf],
          [-0.6931, -0.6931,    -inf],
          [   -inf,  0.0000,    -inf]],

         [[-0.9163, -0.9163, -1.6094],
          [-0.4055,    -inf, -1.0986],
          [ 0.0000,    -inf,    -inf]],

         [[ 0.0000,    -inf,    -inf],
          [ 0.0000,    -inf,    -inf],
          [   -inf, -0.6931, -0.6931]]], 4)


def test_fit_raises(model, X, w):
	assert_raises(ValueError, model.fit, [X])
	assert_raises(ValueError, model.fit, X[0])
	assert_raises((ValueError, TypeError), model.fit, X[0][0])
	assert_raises(ValueError, model.fit, 
		[[-0.1 for i in range(3)] for x in X])

	assert_raises(ValueError, model.fit, [X], w)
	assert_raises(ValueError, model.fit, X, [w])
	assert_raises(ValueError, model.fit, [X], [w])
	assert_raises(ValueError, model.fit, X[:len(X)-1], w)
	assert_raises(ValueError, model.fit, X, w[:len(w)-1])

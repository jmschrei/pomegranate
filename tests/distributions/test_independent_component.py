# test_independent_component.py
# Contact: Jacob Schreiber <jmschreiber91@gmail.com>

import os
import numpy
import torch
import pytest

from torchegranate.distributions import Exponential
from torchegranate.distributions import Gamma
from torchegranate.distributions import Categorical
from torchegranate.distributions import IndependentComponents

from ._utils import _test_initialization_raises_one_parameter
from ._utils import _test_initialization
from ._utils import _test_predictions
from ._utils import _test_efd_from_summaries
from ._utils import _test_raises

from nose.tools import assert_raises
from numpy.testing import assert_array_almost_equal


MIN_VALUE = 0
MAX_VALUE = None
VALID_VALUE = 1.2


@pytest.fixture
def X():
	return [[1, 2, 0],
		 [0, 3, 1],
		 [1, 1, 0],
		 [2, 2, 1],
		 [3, 1, 0],
		 [5, 1, 1],
		 [2, 1, 0]]


@pytest.fixture
def X_masked(X):
	mask = torch.tensor(numpy.array([
		[False, True,  True ],
		[True,  True,  False],
		[False, False, False],
		[True,  True,  True ],
		[False, True,  False],
		[True,  True,  True ],
		[True,  False, True ]]))

	X = torch.tensor(numpy.array(X))
	return torch.masked.MaskedTensor(X, mask=mask)


@pytest.fixture
def X2():
	return [[1.2, 0.5, 1.1, 1.9],
		 [6.2, 1.1, 2.4, 1.1]] 


@pytest.fixture
def w():
	return [[1], [2], [0], [0], [5], [1], [2]]


@pytest.fixture
def w2():
	return [[1.1], [3.5]]


@pytest.fixture
def distributions():
	d1 = Exponential([1.0])
	d2 = Gamma([1.1], [2.0])
	d3 = Categorical([[0.3, 0.7]])
	return d1, d2, d3


###


def test_initialization():
	d1, d2, d3 = Exponential(), Gamma(), Categorical()
	d = IndependentComponents([d1, d2, d3])

	d1, d2, d3 = Exponential([1]), Gamma([1], [2]), Categorical([[0.1, 0.9]])
	d = IndependentComponents([d1, d2, d3])

	d1, d2, d3 = Exponential(), 5, Gamma()
	assert_raises(ValueError, IndependentComponents, [d1, d2, d3])
	assert_raises(ValueError, IndependentComponents, [d1])
	assert_raises(TypeError, IndependentComponents, d1, d2, d3)


def test_reset_cache(X):
	d1, d2 = Exponential([1]), Gamma([1], [2])
	d = IndependentComponents([d1, d2])
	d1._w_sum[0] = 5
	d1._xw_sum[0] = 21
	d2._w_sum[0] = 1
	d2._xw_sum[0] = 8

	assert_array_almost_equal(d1._w_sum, [5])
	assert_array_almost_equal(d1._xw_sum, [21])
	assert_array_almost_equal(d2._w_sum, [1])
	assert_array_almost_equal(d2._xw_sum, [8])

	d._reset_cache()
	assert_array_almost_equal(d1._w_sum, [0])
	assert_array_almost_equal(d1._xw_sum, [0])
	assert_array_almost_equal(d2._w_sum, [0])
	assert_array_almost_equal(d2._xw_sum, [0])


###


@pytest.mark.sample
def test_sample(distributions):
	torch.manual_seed(0)

	X = IndependentComponents(distributions).sample(1)
	assert_array_almost_equal(X, [[3.508326, 0.269697, 1.]], 4)

	X = IndependentComponents(distributions).sample(5)
	assert_array_almost_equal(X,
		[[1.5661, 0.0797, 1.0000],
		 [0.1968, 0.5958, 0.0000],
		 [0.4325, 0.8809, 1.0000],
		 [0.8707, 0.0992, 0.0000],
		 [0.3400, 0.5702, 1.0000]], 3)


###


def test_probability(X, distributions):
	y = [0.004881, 0.004364, 0.033653, 0.00419 , 0.004555, 0.001438, 0.01238]

	d = IndependentComponents(distributions)
	_test_predictions(X, y, d.probability(X), torch.float32)

	d1 = Exponential([1.2])
	d2 = Exponential([7.5])
	d3 = Exponential([1.2, 7.5])
	d = IndependentComponents([d1, d2])

	x = [[1., 2.],
		 [2., 2.],
		 [0., 1.],
		 [0., 0.]]
	y = [0.036986, 0.016074, 0.097241, 0.111111]
	_test_predictions(x, y, d.probability(x), torch.float32)
	_test_predictions(x, d.probability(x), d3.probability(x), torch.float32)


def test_probability_raises(X, distributions):
	d = IndependentComponents(distributions)
	_test_raises(d, "probability", X, min_value=MIN_VALUE)


def test_log_probability(X, distributions):
	y = [-5.322324, -5.43448, -3.391639, -5.475026, -5.391639, -6.544341,
		   -4.391639]

	d = IndependentComponents(distributions)
	_test_predictions(X, y, d.log_probability(X), torch.float32)

	d1 = Exponential([1.2])
	d2 = Exponential([7.5])
	d3 = Exponential([1.2, 7.5])
	d = IndependentComponents([d1, d2])

	x = [[1., 2.],
		 [2., 2.],
		 [0., 1.],
		 [0., 0.]]
	y = [-3.297225, -4.130558, -2.330558, -2.197225]
	_test_predictions(x, y, d.log_probability(x), torch.float32)
	_test_predictions(x, d.log_probability(x), d3.log_probability(x), 
		torch.float32)


def test_log_probability_raises(X, distributions):
	d = IndependentComponents(distributions)
	_test_raises(d, "log_probability", X, min_value=MIN_VALUE)


###


def test_summarize(X, distributions):
	d = IndependentComponents(distributions)
	d.summarize(X[:4])
	assert_array_almost_equal(d.distributions[0]._w_sum, [4.0])
	assert_array_almost_equal(d.distributions[0]._xw_sum, [4.0])
	assert_array_almost_equal(d.distributions[1]._w_sum, [4.0])
	assert_array_almost_equal(d.distributions[1]._xw_sum, [8.0])
	assert_array_almost_equal(d.distributions[2]._w_sum, [4.0])
	assert_array_almost_equal(d.distributions[2]._xw_sum, [[2.0, 2.0]])

	d.summarize(X[4:])
	assert_array_almost_equal(d.distributions[0]._w_sum, [7.0])
	assert_array_almost_equal(d.distributions[0]._xw_sum, [14.0])
	assert_array_almost_equal(d.distributions[1]._w_sum, [7.0])
	assert_array_almost_equal(d.distributions[1]._xw_sum, [11.0])
	assert_array_almost_equal(d.distributions[2]._w_sum, [7.0])
	assert_array_almost_equal(d.distributions[2]._xw_sum, [[4.0, 3.0]])

	d1 = Exponential([1.0])
	d2 = Gamma([1.1], [2.0])
	d3 = Categorical([[0.3, 0.7]])
	d = IndependentComponents([d1, d2, d3])
	d.summarize(X)
	assert_array_almost_equal(d.distributions[0]._w_sum, [7.0])
	assert_array_almost_equal(d.distributions[0]._xw_sum, [14.0])
	assert_array_almost_equal(d.distributions[1]._w_sum, [7.0])
	assert_array_almost_equal(d.distributions[1]._xw_sum, [11.0])
	assert_array_almost_equal(d.distributions[2]._w_sum, [7.0])
	assert_array_almost_equal(d.distributions[2]._xw_sum, [[4.0, 3.0]])

	d1, d2, d3 = Exponential(), Gamma(), Categorical()
	d = IndependentComponents([d1, d2, d3])
	d.summarize(X[:4])
	assert_array_almost_equal(d.distributions[0]._w_sum, [4.0])
	assert_array_almost_equal(d.distributions[0]._xw_sum, [4.0])
	assert_array_almost_equal(d.distributions[1]._w_sum, [4.0])
	assert_array_almost_equal(d.distributions[1]._xw_sum, [8.0])
	assert_array_almost_equal(d.distributions[2]._w_sum, [4.0])
	assert_array_almost_equal(d.distributions[2]._xw_sum, [[2.0, 2.0]])

	d.summarize(X[4:])
	assert_array_almost_equal(d.distributions[0]._w_sum, [7.0])
	assert_array_almost_equal(d.distributions[0]._xw_sum, [14.0])
	assert_array_almost_equal(d.distributions[1]._w_sum, [7.0])
	assert_array_almost_equal(d.distributions[1]._xw_sum, [11.0])
	assert_array_almost_equal(d.distributions[2]._w_sum, [7.0])
	assert_array_almost_equal(d.distributions[2]._xw_sum, [[4.0, 3.0]])

	d1, d2, d3 = Exponential(), Gamma(), Categorical()
	d = IndependentComponents([d1, d2, d3])
	d.summarize(X)
	assert_array_almost_equal(d.distributions[0]._w_sum, [7.0])
	assert_array_almost_equal(d.distributions[0]._xw_sum, [14.0])
	assert_array_almost_equal(d.distributions[1]._w_sum, [7.0])
	assert_array_almost_equal(d.distributions[1]._xw_sum, [11.0])
	assert_array_almost_equal(d.distributions[2]._w_sum, [7.0])
	assert_array_almost_equal(d.distributions[2]._xw_sum, [[4.0, 3.0]])


def test_summarize_weighted(X, w, distributions):
	d = IndependentComponents(distributions)
	d.summarize(X[:4], sample_weight=w[:4])
	assert_array_almost_equal(d.distributions[0]._w_sum, [3.0])
	assert_array_almost_equal(d.distributions[0]._xw_sum, [1.0])
	assert_array_almost_equal(d.distributions[1]._w_sum, [3.0])
	assert_array_almost_equal(d.distributions[1]._xw_sum, [8.0])
	assert_array_almost_equal(d.distributions[2]._w_sum, [3.0])
	assert_array_almost_equal(d.distributions[2]._xw_sum, [[1.0, 2.0]])

	d.summarize(X[4:], sample_weight=w[4:])
	assert_array_almost_equal(d.distributions[0]._w_sum, [11.0])
	assert_array_almost_equal(d.distributions[0]._xw_sum, [25.0])
	assert_array_almost_equal(d.distributions[1]._w_sum, [11.0])
	assert_array_almost_equal(d.distributions[1]._xw_sum, [16.0])
	assert_array_almost_equal(d.distributions[2]._w_sum, [11.0])
	assert_array_almost_equal(d.distributions[2]._xw_sum, [[8.0, 3.0]])

	d1 = Exponential([1.0])
	d2 = Gamma([1.1], [2.0])
	d3 = Categorical([[0.3, 0.7]])
	d = IndependentComponents([d1, d2, d3])
	d.summarize(X, sample_weight=w)
	assert_array_almost_equal(d.distributions[0]._w_sum, [11.0])
	assert_array_almost_equal(d.distributions[0]._xw_sum, [25.0])
	assert_array_almost_equal(d.distributions[1]._w_sum, [11.0])
	assert_array_almost_equal(d.distributions[1]._xw_sum, [16.0])
	assert_array_almost_equal(d.distributions[2]._w_sum, [11.0])
	assert_array_almost_equal(d.distributions[2]._xw_sum, [[8.0, 3.0]])

	d1, d2, d3 = Exponential(), Gamma(), Categorical()
	d = IndependentComponents([d1, d2, d3])
	d.summarize(X[:4], sample_weight=w[:4])
	assert_array_almost_equal(d.distributions[0]._w_sum, [3.0])
	assert_array_almost_equal(d.distributions[0]._xw_sum, [1.0])
	assert_array_almost_equal(d.distributions[1]._w_sum, [3.0])
	assert_array_almost_equal(d.distributions[1]._xw_sum, [8.0])
	assert_array_almost_equal(d.distributions[2]._w_sum, [3.0])
	assert_array_almost_equal(d.distributions[2]._xw_sum, [[1.0, 2.0]])

	d.summarize(X[4:], sample_weight=w[4:])
	assert_array_almost_equal(d.distributions[0]._w_sum, [11.0])
	assert_array_almost_equal(d.distributions[0]._xw_sum, [25.0])
	assert_array_almost_equal(d.distributions[1]._w_sum, [11.0])
	assert_array_almost_equal(d.distributions[1]._xw_sum, [16.0])
	assert_array_almost_equal(d.distributions[2]._w_sum, [11.0])
	assert_array_almost_equal(d.distributions[2]._xw_sum, [[8.0, 3.0]])

	d1, d2, d3 = Exponential(), Gamma(), Categorical()
	d = IndependentComponents([d1, d2, d3])
	d.summarize(X, sample_weight=w)
	assert_array_almost_equal(d.distributions[0]._w_sum, [11.0])
	assert_array_almost_equal(d.distributions[0]._xw_sum, [25.0])
	assert_array_almost_equal(d.distributions[1]._w_sum, [11.0])
	assert_array_almost_equal(d.distributions[1]._xw_sum, [16.0])
	assert_array_almost_equal(d.distributions[2]._w_sum, [11.0])
	assert_array_almost_equal(d.distributions[2]._xw_sum, [[8.0, 3.0]])


def test_summarize_weighted_flat(X, w, distributions):
	w = numpy.array(w)[:,0] 

	d = IndependentComponents(distributions)
	d.summarize(X[:4], sample_weight=w[:4])
	assert_array_almost_equal(d.distributions[0]._w_sum, [3.0])
	assert_array_almost_equal(d.distributions[0]._xw_sum, [1.0])
	assert_array_almost_equal(d.distributions[1]._w_sum, [3.0])
	assert_array_almost_equal(d.distributions[1]._xw_sum, [8.0])
	assert_array_almost_equal(d.distributions[2]._w_sum, [3.0])
	assert_array_almost_equal(d.distributions[2]._xw_sum, [[1.0, 2.0]])

	d.summarize(X[4:], sample_weight=w[4:])
	assert_array_almost_equal(d.distributions[0]._w_sum, [11.0])
	assert_array_almost_equal(d.distributions[0]._xw_sum, [25.0])
	assert_array_almost_equal(d.distributions[1]._w_sum, [11.0])
	assert_array_almost_equal(d.distributions[1]._xw_sum, [16.0])
	assert_array_almost_equal(d.distributions[2]._w_sum, [11.0])
	assert_array_almost_equal(d.distributions[2]._xw_sum, [[8.0, 3.0]])

	d1 = Exponential([1.0])
	d2 = Gamma([1.1], [2.0])
	d3 = Categorical([[0.3, 0.7]])
	d = IndependentComponents([d1, d2, d3])
	d.summarize(X, sample_weight=w)
	assert_array_almost_equal(d.distributions[0]._w_sum, [11.0])
	assert_array_almost_equal(d.distributions[0]._xw_sum, [25.0])
	assert_array_almost_equal(d.distributions[1]._w_sum, [11.0])
	assert_array_almost_equal(d.distributions[1]._xw_sum, [16.0])
	assert_array_almost_equal(d.distributions[2]._w_sum, [11.0])
	assert_array_almost_equal(d.distributions[2]._xw_sum, [[8.0, 3.0]])

	d1, d2, d3 = Exponential(), Gamma(), Categorical()
	d = IndependentComponents([d1, d2, d3])
	d.summarize(X[:4], sample_weight=w[:4])
	assert_array_almost_equal(d.distributions[0]._w_sum, [3.0])
	assert_array_almost_equal(d.distributions[0]._xw_sum, [1.0])
	assert_array_almost_equal(d.distributions[1]._w_sum, [3.0])
	assert_array_almost_equal(d.distributions[1]._xw_sum, [8.0])
	assert_array_almost_equal(d.distributions[2]._w_sum, [3.0])
	assert_array_almost_equal(d.distributions[2]._xw_sum, [[1.0, 2.0]])

	d.summarize(X[4:], sample_weight=w[4:])
	assert_array_almost_equal(d.distributions[0]._w_sum, [11.0])
	assert_array_almost_equal(d.distributions[0]._xw_sum, [25.0])
	assert_array_almost_equal(d.distributions[1]._w_sum, [11.0])
	assert_array_almost_equal(d.distributions[1]._xw_sum, [16.0])
	assert_array_almost_equal(d.distributions[2]._w_sum, [11.0])
	assert_array_almost_equal(d.distributions[2]._xw_sum, [[8.0, 3.0]])

	d1, d2, d3 = Exponential(), Gamma(), Categorical()
	d = IndependentComponents([d1, d2, d3])
	d.summarize(X, sample_weight=w)
	assert_array_almost_equal(d.distributions[0]._w_sum, [11.0])
	assert_array_almost_equal(d.distributions[0]._xw_sum, [25.0])
	assert_array_almost_equal(d.distributions[1]._w_sum, [11.0])
	assert_array_almost_equal(d.distributions[1]._xw_sum, [16.0])
	assert_array_almost_equal(d.distributions[2]._w_sum, [11.0])
	assert_array_almost_equal(d.distributions[2]._xw_sum, [[8.0, 3.0]])


def test_summarize_raises(X, w, distributions):
	d = IndependentComponents(distributions)
	_test_raises(d, "summarize", X, w=w, min_value=MIN_VALUE, 
		max_value=MAX_VALUE)


def test_from_summaries(X, distributions):
	d1, d2, d3 = distributions
	d = IndependentComponents(distributions)
	d.summarize(X)
	assert_array_almost_equal(d1.scales, [1.])
	assert_array_almost_equal(d2.rates, [2.])
	assert_array_almost_equal(d2.shapes, [1.1])
	assert_array_almost_equal(d3.probs, [[0.3, 0.7]])

	d.from_summaries()
	assert_array_almost_equal(d1.scales, [2.])
	assert_array_almost_equal(d2.rates, [3.382766], 4)
	assert_array_almost_equal(d2.shapes, [5.315776], 4)
	assert_array_almost_equal(d3.probs, [[0.571429, 0.428571]], 4)


def test_from_summaries_inertia(X):
	d1 = Exponential([0.3])
	d2 = Exponential([0.7])
	d3 = Exponential([1.1])
	
	d = IndependentComponents([d1, d2, d3])
	d.summarize(X)
	d.from_summaries()

	assert_array_almost_equal(d1.scales, [2.0])
	assert_array_almost_equal(d2.scales, [1.571429], 4)
	assert_array_almost_equal(d3.scales, [0.428571], 4)


	d1 = Exponential([0.3], inertia=0.3)
	d2 = Exponential([0.7], inertia=0.6)
	d3 = Exponential([1.1])
	
	d = IndependentComponents([d1, d2, d3])
	d.summarize(X)
	d.from_summaries()

	assert_array_almost_equal(d1.scales, [1.49])
	assert_array_almost_equal(d2.scales, [1.048571], 4)
	assert_array_almost_equal(d3.scales, [0.428571], 4)


def test_from_summaries_frozen(X):
	d1 = Exponential([0.3])
	d2 = Exponential([0.7])
	d3 = Exponential([1.1])
	
	d = IndependentComponents([d1, d2, d3])
	d.summarize(X)
	d.from_summaries()

	assert_array_almost_equal(d1.scales, [2.0])
	assert_array_almost_equal(d2.scales, [1.571429], 4)
	assert_array_almost_equal(d3.scales, [0.428571], 4)


	d1 = Exponential([0.3], frozen=True)
	d2 = Exponential([0.7], frozen=True)
	d3 = Exponential([1.1])
	
	d = IndependentComponents([d1, d2, d3])
	d.summarize(X)
	d.from_summaries()

	assert_array_almost_equal(d1.scales, [0.3])
	assert_array_almost_equal(d2.scales, [0.7])
	assert_array_almost_equal(d3.scales, [0.428571], 4)


def test_fit(X, distributions):
	d1, d2, d3 = distributions
	d = IndependentComponents(distributions)
	d.fit(X)
	assert_array_almost_equal(d1.scales, [2.])
	assert_array_almost_equal(d2.rates, [3.382766], 4)
	assert_array_almost_equal(d2.shapes, [5.315776], 4)
	assert_array_almost_equal(d3.probs, [[0.571429, 0.428571]], 4)


def test_fit_weighted(X, w, distributions):
	d1, d2, d3 = distributions
	d = IndependentComponents(distributions)
	d.fit(X, sample_weight=w)
	assert_array_almost_equal(d1.scales, [2.272727], 4)
	assert_array_almost_equal(d2.rates, [3.181162], 4)
	assert_array_almost_equal(d2.shapes, [4.627145], 4)
	assert_array_almost_equal(d3.probs, [[0.727273, 0.272727]], 4)


def test_fit_chain(X, w, distributions):
	d1, d2, d3 = distributions
	d = IndependentComponents(distributions).fit(X, sample_weight=w)
	assert_array_almost_equal(d1.scales, [2.272727], 4)
	assert_array_almost_equal(d2.rates, [3.181162], 4)
	assert_array_almost_equal(d2.shapes, [4.627145], 4)
	assert_array_almost_equal(d3.probs, [[0.727273, 0.272727]], 4)


def test_fit_raises(X, w, distributions):
	d = IndependentComponents(distributions)
	_test_raises(d, "fit", X, w=w, min_value=MIN_VALUE, 
		max_value=MAX_VALUE)


def test_serialization(X, distributions):
	d = IndependentComponents(distributions)
	d.summarize(X[:4])
	assert_array_almost_equal(d.distributions[0]._w_sum, [4.0])
	assert_array_almost_equal(d.distributions[0]._xw_sum, [4.0])
	assert_array_almost_equal(d.distributions[1]._w_sum, [4.0])
	assert_array_almost_equal(d.distributions[1]._xw_sum, [8.0])
	assert_array_almost_equal(d.distributions[2]._w_sum, [4.0])
	assert_array_almost_equal(d.distributions[2]._xw_sum, [[2.0, 2.0]])

	torch.save(d, ".pytest.torch")
	d2 = torch.load(".pytest.torch")
	os.system("rm .pytest.torch")

	assert_array_almost_equal(d2.distributions[0]._w_sum, [4.0])
	assert_array_almost_equal(d2.distributions[0]._xw_sum, [4.0])
	assert_array_almost_equal(d2.distributions[1]._w_sum, [4.0])
	assert_array_almost_equal(d2.distributions[1]._xw_sum, [8.0])
	assert_array_almost_equal(d2.distributions[2]._w_sum, [4.0])
	assert_array_almost_equal(d2.distributions[2]._xw_sum, [[2.0, 2.0]])

	assert_array_almost_equal(d.log_probability(X), d2.log_probability(X))


def test_masked_probability(distributions, X, X_masked):
	X = torch.tensor(numpy.array(X))
	y = [0.004068, 0.001214, 0.028045, 0.001165, 0.003795, 0.0004  ,
           0.010317]

	d3 = Exponential([4.])
	d = IndependentComponents([distributions[0], distributions[1], d3])
	mask = torch.ones_like(X).type(torch.bool)
	X_ = torch.masked.MaskedTensor(X, mask=mask)

	assert_array_almost_equal(y, d.probability(X_))

	y =  [1.105751e-02, 6.233579e-03, 1.000000e+00, 1.165453e-03,
           3.049322e-01, 4.000343e-04, 3.383382e-02]

	assert_array_almost_equal(y, d.probability(X_masked))


def test_masked_log_probability(distributions, X, X_masked):
	X = torch.tensor(numpy.array(X))
	y = [-5.504645, -6.714099, -3.57396 , -6.754645, -5.57396 , -7.82396 ,
           -4.57396]

	d3 = Exponential([4.])
	d = IndependentComponents([distributions[0], distributions[1], d3])
	mask = torch.ones_like(X).type(torch.bool)
	X_ = torch.masked.MaskedTensor(X, mask=mask)

	assert_array_almost_equal(y, d.log_probability(X_))

	y = [-4.504645, -5.077805,  0.      , -6.754645, -1.187666, -7.82396 ,
           -3.386294]

	assert_array_almost_equal(y, d.log_probability(X_masked))


def test_masked_summarize(X, X_masked, w):
	X = torch.tensor(numpy.array(X))
	mask = torch.ones_like(X).type(torch.bool)
	X_ = torch.masked.MaskedTensor(X, mask=mask)

	d1 = Exponential([1.])
	d2 = Gamma([1.], [2.])
	d3 = Exponential([4.])
	d = IndependentComponents([d1, d2, d3])
	d.summarize(X_masked)
	assert_array_almost_equal(d1._w_sum, [4.0])
	assert_array_almost_equal(d1._xw_sum, [9.0])
	assert_array_almost_equal(d2._w_sum, [5.0])
	assert_array_almost_equal(d2._xw_sum, [9.0])
	assert_array_almost_equal(d3._w_sum, [4.0])
	assert_array_almost_equal(d3._xw_sum, [2.0])


def test_masked_from_summaries(X, X_masked):
	X = torch.tensor(numpy.array(X))
	mask = torch.ones_like(X).type(torch.bool)
	X_ = torch.masked.MaskedTensor(X, mask=mask)

	d1 = Exponential([1.])
	d2 = Gamma([1.], [2.])
	d3 = Exponential([4.])
	d = IndependentComponents([d1, d2, d3])
	d.summarize(X_masked)
	d.from_summaries()
	assert_array_almost_equal(d1.scales, [2.25])
	assert_array_almost_equal(d2.rates, [3.14873], 4)
	assert_array_almost_equal(d2.shapes, [5.667715], 4)
	assert_array_almost_equal(d3.scales, [0.5])


def test_masked_fit(X, X_masked):
	X = torch.tensor(numpy.array(X))
	mask = torch.ones_like(X).type(torch.bool)
	X_ = torch.masked.MaskedTensor(X, mask=mask)

	d1 = Exponential([1.])
	d2 = Gamma([1.], [2.])
	d3 = Exponential([4.])
	d = IndependentComponents([d1, d2, d3]).fit(X_masked)
	assert_array_almost_equal(d1.scales, [2.25])
	assert_array_almost_equal(d2.rates, [3.14873], 4)
	assert_array_almost_equal(d2.shapes, [5.667715], 4)
	assert_array_almost_equal(d3.scales, [0.5])
	
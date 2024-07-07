# test_bernoulli.py
# Contact: Jacob Schreiber <jmschreiber91@gmail.com>

import os
import numpy
import torch
import pytest

from pomegranate.distributions import Bernoulli

from ._utils import _test_initialization_raises_one_parameter
from ._utils import _test_initialization
from ._utils import _test_predictions
from ._utils import _test_efd_from_summaries
from ._utils import _test_raises

from ..tools import assert_raises
from numpy.testing import assert_array_almost_equal


MIN_VALUE = 0
MAX_VALUE = 1
VALID_VALUE = 0.55


@pytest.fixture
def X():
	return [[1, 0, 0],
	     [0, 0, 1],
	     [1, 0, 1],
	     [0, 1, 1],
	     [1, 1, 0],
	     [1, 1, 0],
	     [1, 1, 1]]


@pytest.fixture
def X2():
	return [[1.0, 0.0, 1.0, 1.0],
	     [1.0, 1.0, 0.0, 1.0]] 


@pytest.fixture
def w():
	return [[1], [2], [0], [0], [5], [1], [2]]


@pytest.fixture
def w2():
	return [[1.1], [3.5]]


@pytest.fixture
def probs():
	return [0.1, 0.87, 0.37]


###


def test_initialization():
	d = Bernoulli()
	_test_initialization(d, None, "probs", 0.0, False, None)
	assert_raises(AttributeError, getattr, d, "_w_sum")
	assert_raises(AttributeError, getattr, d, "_xw_sum")
	assert_raises(AttributeError, getattr, d, "_log_probs")
	assert_raises(AttributeError, getattr, d, "_log_inv_probs")


def test_initialization_float():
	funcs = (lambda x: x, tuple, 
		lambda x: numpy.array(x, dtype=numpy.float32), 
		lambda x: torch.tensor(x, dtype=torch.float32), 
		lambda x: torch.nn.Parameter(torch.tensor(x), requires_grad=False))

	x = [0.9, 0.51, 0.33, 0.19, 0.01]
	for func in funcs:
		y = func(x)
		_test_initialization(Bernoulli(y, inertia=0.0, frozen=False), 
			y, "probs", 0.0, False, torch.float32)
		_test_initialization(Bernoulli(y, inertia=0.3, frozen=False), 
			y, "probs", 0.3, False, torch.float32)
		_test_initialization(Bernoulli(y, inertia=1.0, frozen=True), 
			y, "probs", 1.0, True, torch.float32)
		_test_initialization(Bernoulli(y, inertia=1.0, frozen=False), 
			y, "probs", 1.0, False, torch.float32)

	x = numpy.array(x, dtype=numpy.float64)
	_test_initialization(Bernoulli(x, inertia=0.0, frozen=False), 
		x, "probs", 0.0, False, torch.float64)
	_test_initialization(Bernoulli(x, inertia=0.3, frozen=False), 
		x, "probs", 0.3, False, torch.float64)
	_test_initialization(Bernoulli(x, inertia=1.0, frozen=True), 
		x, "probs", 1.0, True, torch.float64)
	_test_initialization(Bernoulli(x, inertia=1.0, frozen=False), 
		x, "probs", 1.0, False, torch.float64)


def test_initialization_raises():
	_test_initialization_raises_one_parameter(Bernoulli, VALID_VALUE, 
		min_value=MIN_VALUE)


def test_reset_cache(X):
	d = Bernoulli()
	d.summarize(X)
	assert_array_almost_equal(d._w_sum, [7.0, 7.0, 7.0])
	assert_array_almost_equal(d._xw_sum, [5.0, 4.0, 4.0])

	d._reset_cache()
	assert_array_almost_equal(d._w_sum, [0.0, 0.0, 0.0])
	assert_array_almost_equal(d._xw_sum, [0.0, 0.0, 0.0])	

	d = Bernoulli()
	assert_raises(AttributeError, getattr, d, "_w_sum")
	assert_raises(AttributeError, getattr, d, "_xw_sum")
	assert_raises(AttributeError, getattr, d, "_log_probs")
	assert_raises(AttributeError, getattr, d, "_log_inv_probs")

	d._reset_cache()
	assert_raises(AttributeError, getattr, d, "_w_sum")
	assert_raises(AttributeError, getattr, d, "_xw_sum")
	assert_raises(AttributeError, getattr, d, "_log_probs")
	assert_raises(AttributeError, getattr, d, "_log_inv_probs")


def test_initialize(X):
	d = Bernoulli()
	assert d.d is None
	assert d.probs is None
	assert d._initialized == False
	assert_raises(AttributeError, getattr, d, "_w_sum")
	assert_raises(AttributeError, getattr, d, "_xw_sum")
	assert_raises(AttributeError, getattr, d, "_log_probs")
	assert_raises(AttributeError, getattr, d, "_log_inv_probs")

	d._initialize(3)
	assert d._initialized == True
	assert d.probs.shape[0] == 3
	assert d.d == 3
	assert_array_almost_equal(d.probs, [0.0, 0.0, 0.0])
	assert_array_almost_equal(d._w_sum, [0.0, 0.0, 0.0])
	assert_array_almost_equal(d._xw_sum, [0.0, 0.0, 0.0])	

	d._initialize(2)
	assert d._initialized == True
	assert d.probs.shape[0] == 2
	assert d.d == 2
	assert_array_almost_equal(d.probs, [0.0, 0.0])
	assert_array_almost_equal(d._w_sum, [0.0, 0.0])
	assert_array_almost_equal(d._xw_sum, [0.0, 0.0])	

	d = Bernoulli([0.1, 0.67])
	assert d._initialized == True
	assert d.d == 2

	d._initialize(3)
	assert d._initialized == True
	assert d.probs.shape[0] == 3
	assert d.d == 3
	assert_array_almost_equal(d.probs, [0.0, 0.0, 0.0])

	d = Bernoulli()
	d.summarize(X)
	d._initialize(4)
	assert d._initialized == True
	assert d.probs.shape[0] == 4
	assert d.d == 4
	assert_array_almost_equal(d._w_sum, [0.0, 0.0, 0.0, 0.0])
	assert_array_almost_equal(d._xw_sum, [0.0, 0.0, 0.0, 0.0])	


###


@pytest.mark.sample
def test_sample(probs):
	torch.manual_seed(0)

	X = Bernoulli(probs).sample(1)
	assert_array_almost_equal(X, [[0, 1, 1]])

	X = Bernoulli(probs).sample(5)
	assert_array_almost_equal(X, 
		[[0., 1., 0.],
         [0., 0., 0.],
         [0., 1., 0.],
         [1., 1., 1.],
         [0., 1., 0.]])


###


def test_probability(X, probs):
	p = [0.32]
	x = [[1.0], [0.0], [1.0], [1.0], [0.0]]
	y = [0.32, 0.68, 0.32, 0.32, 0.68]

	d1 = Bernoulli(p)
	d2 = Bernoulli(numpy.array(p, dtype=numpy.float64))

	_test_predictions(x, y, d1.probability(x), torch.float32)
	_test_predictions(x, y, d2.probability(x), torch.float64)

	x = [[1.0, 0.0, 1.0]]
	y = [0.00481]

	d1 = Bernoulli(probs)
	d2 = Bernoulli(numpy.array(probs, dtype=numpy.float64))

	_test_predictions(x, y, d1.probability(x), torch.float32)
	_test_predictions(x, y, d2.probability(x), torch.float64)

	y = [0.00819, 0.04329, 0.00481, 0.28971, 0.05481, 0.05481, 0.03219]

	d1 = Bernoulli(probs)
	d2 = Bernoulli(numpy.array(probs, dtype=numpy.float64))

	_test_predictions(X, y, d1.probability(X), torch.float32)
	_test_predictions(X, y, d2.probability(X), torch.float64)


def test_probability_dtypes(X, probs):
	X = numpy.array(X)
	X_int = numpy.array(X, dtype=numpy.int32)

	y = Bernoulli(probs).probability(X)
	assert y.dtype == torch.float32

	y = Bernoulli(probs).probability(X_int)
	assert y.dtype == torch.float32

	X = X.astype(numpy.float64)
	X_int = X.astype('int32')

	y = Bernoulli(numpy.array(probs, dtype=numpy.float64)).probability(X)
	assert y.dtype == torch.float64

	y = Bernoulli(numpy.array(probs, dtype=numpy.float64)).probability(X_int)
	assert y.dtype == torch.float64


def test_probability_raises(X, probs):
	_test_raises(Bernoulli(probs), "probability", X, min_value=MIN_VALUE, 
		max_value=MAX_VALUE)

	_test_raises(Bernoulli([VALID_VALUE]), "probability", X, 
		min_value=MIN_VALUE, max_value=MAX_VALUE)


def test_log_probability(X, probs):
	p = [0.32]
	x = [[1.0], [0.0], [1.0], [1.0], [0.0]]
	y = [-1.139434, -0.385662, -1.139434, -1.139434, -0.385662]

	x_torch = torch.tensor(numpy.array(x))	
	p_torch = torch.tensor(numpy.array(p))

	d1 = Bernoulli(p)
	d2 = Bernoulli(numpy.array(p, dtype=numpy.float64))
	d3 = torch.distributions.Bernoulli(p_torch)

	_test_predictions(x, y, d1.log_probability(x), torch.float32)
	_test_predictions(x, y, d2.log_probability(x), torch.float64)
	_test_predictions(x, d3.log_prob(x_torch).sum(axis=1), 
		d2.log_probability(x), torch.float64)

	x = [[1.0, 0.0, 1.0]]
	y = [-5.337058]
	p_torch = torch.tensor(numpy.array(probs))
	x_torch = torch.tensor(numpy.array(x))

	d1 = Bernoulli(probs)
	d2 = Bernoulli(numpy.array(probs, dtype=numpy.float64))
	d3 = torch.distributions.Bernoulli(p_torch)

	_test_predictions(x, y, d1.log_probability(x), torch.float32)
	_test_predictions(x, y, d2.log_probability(x), torch.float64)
	_test_predictions(x, d3.log_prob(x_torch).sum(axis=1), 
		d2.log_probability(x), torch.float64)

	p_torch = torch.tensor(numpy.array(probs))
	y = [-4.804842, -3.139834, -5.337058, -1.238875, -2.903883, -2.903883,
           -3.436099]

	d1 = Bernoulli(probs)
	d2 = Bernoulli(numpy.array(probs, dtype=numpy.float64))
	d3 = torch.distributions.Exponential(p_torch)

	_test_predictions(X, y, d1.log_probability(X), torch.float32)
	_test_predictions(X, y, d2.log_probability(X), torch.float64)


def test_log_probability_dtypes(X, probs):
	X = numpy.array(X)
	X_int = numpy.array(X, dtype=numpy.int32)

	y = Bernoulli(probs).log_probability(X)
	assert y.dtype == torch.float32

	y = Bernoulli(probs).log_probability(X_int)
	assert y.dtype == torch.float32

	X = X.astype(numpy.float64)
	X_int = X.astype('int32')

	y = Bernoulli(numpy.array(probs, dtype=numpy.float64)).log_probability(X)
	assert y.dtype == torch.float64

	y = Bernoulli(numpy.array(probs, dtype=numpy.float64)).log_probability(X_int)
	assert y.dtype == torch.float64


def test_log_probability_raises(X, probs):
	_test_raises(Bernoulli(probs), "log_probability", X, min_value=MIN_VALUE, 
		max_value=MAX_VALUE)

	_test_raises(Bernoulli([VALID_VALUE]), "log_probability", X, 
		min_value=MIN_VALUE, max_value=MAX_VALUE)


###


def test_summarize(X, X2, probs):
	d = Bernoulli(probs)
	d.summarize(X[:4])
	assert_array_almost_equal(d._w_sum, [4.0, 4.0, 4.0])
	assert_array_almost_equal(d._xw_sum, [2.0, 1.0, 3.0])

	d.summarize(X[4:])
	assert_array_almost_equal(d._w_sum, [7.0, 7.0, 7.0])
	assert_array_almost_equal(d._xw_sum, [5.0, 4.0, 4.0])

	d = Bernoulli(probs)
	d.summarize(X)
	assert_array_almost_equal(d._w_sum, [7.0, 7.0, 7.0])
	assert_array_almost_equal(d._xw_sum, [5.0, 4.0, 4.0])


	d = Bernoulli()
	d.summarize(X[:4])
	assert_array_almost_equal(d._w_sum, [4.0, 4.0, 4.0])
	assert_array_almost_equal(d._xw_sum, [2.0, 1.0, 3.0])

	d.summarize(X[4:])
	assert_array_almost_equal(d._w_sum, [7.0, 7.0, 7.0])
	assert_array_almost_equal(d._xw_sum, [5.0, 4.0, 4.0])

	d = Bernoulli()
	d.summarize(X)
	assert_array_almost_equal(d._w_sum, [7.0, 7.0, 7.0])
	assert_array_almost_equal(d._xw_sum, [5.0, 4.0, 4.0])


def test_summarize_weighted(X, X2, w, w2, probs):
	d = Bernoulli(probs)
	d.summarize(X[:4], sample_weight=w[:4])
	assert_array_almost_equal(d._w_sum, [3., 3., 3.])
	assert_array_almost_equal(d._xw_sum, [1., 0., 2.])

	d.summarize(X[4:], sample_weight=w[4:])
	assert_array_almost_equal(d._w_sum, [11.0, 11.0, 11.0])
	assert_array_almost_equal(d._xw_sum, [9.0, 8.0, 4.0])

	d = Bernoulli(probs)
	d.summarize(X, sample_weight=w)
	assert_array_almost_equal(d._w_sum, [11.0, 11.0, 11.0])
	assert_array_almost_equal(d._xw_sum, [9.0, 8.0, 4.0])


	d = Bernoulli()
	d.summarize(X2, sample_weight=w2)
	assert_array_almost_equal(d._w_sum, [4.6, 4.6, 4.6, 4.6])
	assert_array_almost_equal(d._xw_sum, [4.6, 3.5, 1.1, 4.6])

	d = Bernoulli([0.1, 0.1, 0.1, 0.1])
	d.summarize(X2, sample_weight=w2)
	assert_array_almost_equal(d._w_sum, [4.6, 4.6, 4.6, 4.6])
	assert_array_almost_equal(d._xw_sum, [4.6, 3.5, 1.1, 4.6])


def test_summarize_weighted_flat(X, X2, w, w2, probs):
	w = numpy.array(w)[:,0] 

	d = Bernoulli(probs)
	d.summarize(X[:4], sample_weight=w[:4])
	assert_array_almost_equal(d._w_sum, [3., 3., 3.])
	assert_array_almost_equal(d._xw_sum, [1., 0., 2.])

	d.summarize(X[4:], sample_weight=w[4:])
	assert_array_almost_equal(d._w_sum, [11.0, 11.0, 11.0])
	assert_array_almost_equal(d._xw_sum, [9.0, 8.0, 4.0])

	d = Bernoulli(probs)
	d.summarize(X, sample_weight=w)
	assert_array_almost_equal(d._w_sum, [11.0, 11.0, 11.0])
	assert_array_almost_equal(d._xw_sum, [9.0, 8.0, 4.0])


	d = Bernoulli()
	d.summarize(X2, sample_weight=w2)
	assert_array_almost_equal(d._w_sum, [4.6, 4.6, 4.6, 4.6])
	assert_array_almost_equal(d._xw_sum, [4.6, 3.5, 1.1, 4.6])

	d = Bernoulli([0.1, 0.1, 0.1, 0.1])
	d.summarize(X2, sample_weight=w2)
	assert_array_almost_equal(d._w_sum, [4.6, 4.6, 4.6, 4.6])
	assert_array_almost_equal(d._xw_sum, [4.6, 3.5, 1.1, 4.6])


def test_summarize_weighted_2d(X):
	d = Bernoulli()
	d.summarize(X[:4], sample_weight=X[:4])
	assert_array_almost_equal(d._w_sum, [2., 1., 3.])
	assert_array_almost_equal(d._xw_sum, [2., 1., 3.])

	d.summarize(X[4:], sample_weight=X[4:])
	assert_array_almost_equal(d._w_sum, [5., 4., 4.])
	assert_array_almost_equal(d._xw_sum, [5., 4., 4.])

	d = Bernoulli()
	d.summarize(X, sample_weight=X)
	assert_array_almost_equal(d._w_sum, [5., 4., 4.])
	assert_array_almost_equal(d._xw_sum, [5., 4., 4.])


def test_summarize_dtypes(X, w, probs):
	X = numpy.array(X)
	X = X.astype(numpy.float32)

	p = numpy.array(probs, dtype=numpy.float32)
	d = Bernoulli(p)
	assert d._xw_sum.dtype == torch.float32
	d.summarize(X)
	assert d._xw_sum.dtype == torch.float32

	X = X.astype(numpy.float64)
	p = numpy.array(probs, dtype=numpy.float32)
	d = Bernoulli(p)
	assert d._xw_sum.dtype == torch.float32
	d.summarize(X)
	assert d._xw_sum.dtype == torch.float32

	X = X.astype(numpy.int32)
	p = numpy.array(probs, dtype=numpy.float32)
	d = Bernoulli(p)
	assert d._xw_sum.dtype == torch.float32
	d.summarize(X)
	assert d._xw_sum.dtype == torch.float32

	X = X.astype(numpy.int64)
	p = numpy.array(probs, dtype=numpy.float32)
	d = Bernoulli(p)
	assert d._xw_sum.dtype == torch.float32
	d.summarize(X)
	assert d._xw_sum.dtype == torch.float32


def test_summarize_raises(X, w, probs):
	_test_raises(Bernoulli(probs), "summarize", X, w=w, 
		min_value=MIN_VALUE, max_value=MAX_VALUE)

	_test_raises(Bernoulli(), "summarize", X, w=w, 
		min_value=MIN_VALUE, max_value=MAX_VALUE)

	_test_raises(Bernoulli([VALID_VALUE]), "summarize", X, w=w, 
		min_value=MIN_VALUE, max_value=MAX_VALUE)


def test_from_summaries(X, probs):
	d = Bernoulli(probs)
	d.summarize(X)
	d.from_summaries()
	assert_raises(AssertionError, assert_array_almost_equal, probs, d.probs)

	for param in probs, None:
		d = Bernoulli(param)
		d.summarize(X[:4])
		d.from_summaries()
		_test_efd_from_summaries(d, "probs", "_log_probs", [0.5 , 0.25, 0.75])

		d.summarize(X[4:])
		d.from_summaries()
		_test_efd_from_summaries(d, "probs", "_log_probs", 
			[1.      , 1.      , 0.333333])

		d = Bernoulli(param)
		d.summarize(X[:4])
		d.summarize(X[4:])
		d.from_summaries()
		_test_efd_from_summaries(d, "probs", "_log_probs", 
			[0.714286, 0.571429, 0.571429])

		d = Bernoulli(param)
		d.summarize(X)
		d.from_summaries()
		_test_efd_from_summaries(d, "probs", "_log_probs", 
			[0.714286, 0.571429, 0.571429])


def test_from_summaries_weighted(X, w, probs):
	for param in probs, None:
		d = Bernoulli(probs)
		d.summarize(X[:4], sample_weight=w[:4])
		d.from_summaries()
		_test_efd_from_summaries(d, "probs", "_log_probs", 
			[0.333333, 0.      , 0.666667])

		d.summarize(X[4:], sample_weight=w[4:])
		d.from_summaries()
		_test_efd_from_summaries(d, "probs", "_log_probs", 
			[1.  , 1.  , 0.25])

		d = Bernoulli(probs)
		d.summarize(X, sample_weight=w)
		d.from_summaries()
		_test_efd_from_summaries(d, "probs", "_log_probs", 
			[0.818182, 0.727273, 0.363636])


def test_from_summaries_null():
	d = Bernoulli([0.2, 0.8])
	d.from_summaries()
	assert d.probs[0] != 0.2 and d.probs[1] != 0.8 
	assert_array_almost_equal(d._w_sum, [0.0, 0.0])
	assert_array_almost_equal(d._xw_sum, [0.0, 0.0])

	d = Bernoulli([0.2, 0.8], inertia=0.5)
	d.from_summaries()
	assert d.probs[0] != 0.2 and d.probs[1] != 0.8
	assert_array_almost_equal(d._w_sum, [0.0, 0.0])
	assert_array_almost_equal(d._xw_sum, [0.0, 0.0])

	d = Bernoulli([0.2, 0.8], inertia=0.5, frozen=True)
	d.from_summaries()
	_test_efd_from_summaries(d, "probs", "_log_probs", [0.2, 0.8])


def test_from_summaries_inertia(X, w, probs):
	d = Bernoulli(probs, inertia=0.3)
	d.summarize(X[:4])
	d.from_summaries()
	_test_efd_from_summaries(d, "probs", "_log_probs", [0.38 , 0.436, 0.636])

	d.summarize(X[4:])
	d.from_summaries()
	_test_efd_from_summaries(d, "probs", "_log_probs", 
		[0.814   , 0.8308  , 0.424133])

	d = Bernoulli(probs, inertia=0.3)
	d.summarize(X)
	d.from_summaries()
	_test_efd_from_summaries(d, "probs", "_log_probs", 
		[0.53 , 0.661, 0.511])


def test_from_summaries_weighted_inertia(X, w, probs):
	d = Bernoulli(probs, inertia=0.3)
	d.summarize(X, sample_weight=w)
	d.from_summaries()
	_test_efd_from_summaries(d, "probs", "_log_probs", 
		[0.602727, 0.770091, 0.365545])

	d = Bernoulli(probs, inertia=1.0)
	d.summarize(X[:4])
	d.from_summaries()
	_test_efd_from_summaries(d, "probs", "_log_probs", probs)

	d.summarize(X[4:])
	d.from_summaries()
	_test_efd_from_summaries(d, "probs", "_log_probs", probs)

	d = Bernoulli(probs, inertia=1.0)
	d.summarize(X)
	d.from_summaries()
	_test_efd_from_summaries(d, "probs", "_log_probs", probs)


def test_from_summaries_frozen(X, w, probs):
	d = Bernoulli(probs, frozen=True)
	d.summarize(X[:4])
	assert_array_almost_equal(d._w_sum, [0.0, 0.0, 0.0])
	assert_array_almost_equal(d._xw_sum, [0.0, 0.0, 0.0])

	d.from_summaries()
	_test_efd_from_summaries(d, "probs", "_log_probs", probs)

	d.summarize(X[4:])
	assert_array_almost_equal(d._w_sum, [0.0, 0.0, 0.0])
	assert_array_almost_equal(d._xw_sum, [0.0, 0.0, 0.0])

	d.from_summaries()
	_test_efd_from_summaries(d, "probs", "_log_probs", probs)

	d = Bernoulli(probs, frozen=True)
	d.summarize(X)
	assert_array_almost_equal(d._w_sum, [0.0, 0.0, 0.0])
	assert_array_almost_equal(d._xw_sum, [0.0, 0.0, 0.0])

	d.from_summaries()
	_test_efd_from_summaries(d, "probs", "_log_probs", probs)

	d = Bernoulli(probs, frozen=True)
	d.summarize(X, sample_weight=w)
	assert_array_almost_equal(d._w_sum, [0.0, 0.0, 0.0])
	assert_array_almost_equal(d._xw_sum, [0.0, 0.0, 0.0])

	d.from_summaries()
	_test_efd_from_summaries(d, "probs", "_log_probs", probs)


def test_from_summaries_dtypes(X, probs):
	X = numpy.array(X)
	X = X.astype(numpy.float32)

	p = numpy.array(probs, dtype=numpy.float32)
	d = Bernoulli(p)
	d.summarize(X)
	d.from_summaries()
	assert d.probs.dtype == torch.float32
	assert d._log_probs.dtype == torch.float32

	p = numpy.array(probs, dtype=numpy.float64)
	d = Bernoulli(p)
	d.summarize(X)
	d.from_summaries()
	assert d.probs.dtype == torch.float64
	assert d._log_probs.dtype == torch.float64


def test_from_summaries_raises():
	assert_raises(AttributeError, Bernoulli().from_summaries)


def test_fit(X, probs):
	for param in probs, None:
		d = Bernoulli(param)
		d.fit(X[:4])
		_test_efd_from_summaries(d, "probs", "_log_probs", 
			[0.5 , 0.25, 0.75])

		d.fit(X[4:])
		_test_efd_from_summaries(d, "probs", "_log_probs", 
			[1.      , 1.      , 0.333333])

		d = Bernoulli(param)
		d.fit(X)
		_test_efd_from_summaries(d, "probs", "_log_probs", 
			[0.714286, 0.571429, 0.571429])


def test_fit_weighted(X, w, probs):
	for param in probs, None:
		d = Bernoulli(param)
		d.fit(X[:4], sample_weight=w[:4])
		_test_efd_from_summaries(d, "probs", "_log_probs", 
			[0.333333, 0.      , 0.666667])

		d.fit(X[4:], sample_weight=w[4:])
		_test_efd_from_summaries(d, "probs", "_log_probs", 
			[1.  , 1.  , 0.25])

		d = Bernoulli(param)
		d.fit(X, sample_weight=w)
		_test_efd_from_summaries(d, "probs", "_log_probs", 
			[0.818182, 0.727273, 0.363636])

	d = Bernoulli()
	d.fit(X, sample_weight=w)
	_test_efd_from_summaries(d, "probs", "_log_probs", 
		[0.818182, 0.727273, 0.363636])


def test_fit_chain(X):
	d = Bernoulli().fit(X[:4])
	_test_efd_from_summaries(d, "probs", "_log_probs", [0.5 , 0.25, 0.75])

	d.fit(X[4:])
	_test_efd_from_summaries(d, "probs", "_log_probs", 
		[1.      , 1.      , 0.333333])

	d = Bernoulli().fit(X)
	_test_efd_from_summaries(d, "probs", "_log_probs", 
		[0.714286, 0.571429, 0.571429])


def test_fit_dtypes(X, probs):
	X = numpy.array(X)
	X = X.astype(numpy.float32)

	p = numpy.array(probs, dtype=numpy.float32)
	d = Bernoulli(p).fit(X)
	assert d.probs.dtype == torch.float32
	assert d._log_probs.dtype == torch.float32

	p = numpy.array(probs, dtype=numpy.float64)
	d = Bernoulli(p).fit(X)
	assert d.probs.dtype == torch.float64
	assert d._log_probs.dtype == torch.float64


def test_fit_raises(X, w, probs):
	_test_raises(Bernoulli(probs), "fit", X, w=w, min_value=MIN_VALUE, 
		max_value=MAX_VALUE)

	_test_raises(Bernoulli(), "fit", X, w=w, min_value=MIN_VALUE, 
		max_value=MAX_VALUE)

	_test_raises(Bernoulli([VALID_VALUE]), "fit", X, w=w, 
		min_value=MIN_VALUE, max_value=MAX_VALUE)


def test_serialization(X):
	d = Bernoulli().fit(X[:4])
	d.summarize(X[4:])

	probs = [0.5, 0.25, 0.75]

	assert_array_almost_equal(d.probs, probs)
	assert_array_almost_equal(d._log_probs, numpy.log(probs))

	torch.save(d, ".pytest.torch")
	d2 = torch.load(".pytest.torch")
	os.system("rm .pytest.torch")

	assert_array_almost_equal(d2.probs, probs)
	assert_array_almost_equal(d2._log_probs, numpy.log(probs))

	assert_array_almost_equal(d2._w_sum, [3., 3., 3.])
	assert_array_almost_equal(d2._xw_sum, [3, 3, 1.])
	assert_array_almost_equal(d.log_probability(X), d2.log_probability(X))

# test_gamma.py
# Contact: Jacob Schreiber <jmschreiber91@gmail.com>

import os
import numpy
import torch
import pytest

from pomegranate.distributions import Gamma

from ._utils import _test_initialization_raises_two_parameters
from ._utils import _test_initialization
from ._utils import _test_predictions
from ._utils import _test_efd_from_summaries
from ._utils import _test_raises

from ..tools import assert_raises
from numpy.testing import assert_array_almost_equal


MIN_VALUE = 0
MAX_VALUE = None
VALID_VALUE = 1.2


@pytest.fixture
def X():
	return [[1.1, 2.1, 0.5],
	     [0.5, 0.2, 1.3],
	     [1.4, 1.1, 2.2],
	     [3.1, 2.1, 2.2],
	     [3.4, 1.0, 0.3],
	     [5.4, 1.9, 4.0],
	     [2.2, 1.3, 0.1]]


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
def shapes():
	return [1.2, 1.8, 2.1]


@pytest.fixture
def rates():
	return [0.3, 3.1, 1.2]


###


def test_initialization():
	d = Gamma()
	
	_test_initialization(d, None, "shapes", 0.0, False, None)
	_test_initialization(d, None, "rates", 0.0, False, None)

	assert_raises(AttributeError, getattr, d, "_w_sum")
	assert_raises(AttributeError, getattr, d, "_xw_sum")
	assert_raises(AttributeError, getattr, d, "_logx_w_sum")
	assert_raises(AttributeError, getattr, d, "_log_rates")
	assert_raises(AttributeError, getattr, d, "_lgamma_shapes")
	assert_raises(AttributeError, getattr, d, "_thetas")


def test_initialization_int():
	funcs = (lambda x: x, tuple, numpy.array, torch.tensor, 
		lambda x: torch.nn.Parameter(torch.tensor(x), requires_grad=False))

	x = [1, 2, 3, 8, 1]
	for func in funcs:
		y = func(x)

		_test_initialization(Gamma(y, y, inertia=0.0, frozen=False), 
			y, "shapes", 0.0, False, torch.int64)
		_test_initialization(Gamma(y, y, inertia=0.3, frozen=False), 
			y, "shapes", 0.3, False, torch.int64)
		_test_initialization(Gamma(y, y, inertia=1.0, frozen=True), 
			y, "shapes", 1.0, True, torch.int64)
		_test_initialization(Gamma(y, y, inertia=1.0, frozen=False), 
			y, "shapes", 1.0, False, torch.int64)

		_test_initialization(Gamma(y, y, inertia=0.0, frozen=False), 
			y, "rates", 0.0, False, torch.int64)
		_test_initialization(Gamma(y, y, inertia=0.3, frozen=False), 
			y, "rates", 0.3, False, torch.int64)
		_test_initialization(Gamma(y, y, inertia=1.0, frozen=True), 
			y, "rates", 1.0, True, torch.int64)
		_test_initialization(Gamma(y, y, inertia=1.0, frozen=False), 
			y, "rates", 1.0, False, torch.int64)

	x = numpy.array(x, dtype=numpy.int32)
	for func in funcs[2:]:
		y = func(x)
		_test_initialization(Gamma(y, y, inertia=0.0, frozen=False), 
			y, "shapes", 0.0, False, torch.int32)
		_test_initialization(Gamma(y, y, inertia=0.3, frozen=False), 
			y, "shapes", 0.3, False, torch.int32)
		_test_initialization(Gamma(y, y, inertia=1.0, frozen=True), 
			y, "shapes", 1.0, True, torch.int32)
		_test_initialization(Gamma(y, y, inertia=1.0, frozen=False), 
			y, "shapes", 1.0, False, torch.int32)

		_test_initialization(Gamma(y, y, inertia=0.0, frozen=False), 
			y, "rates", 0.0, False, torch.int32)
		_test_initialization(Gamma(y, y, inertia=0.3, frozen=False), 
			y, "rates", 0.3, False, torch.int32)
		_test_initialization(Gamma(y, y, inertia=1.0, frozen=True), 
			y, "rates", 1.0, True, torch.int32)
		_test_initialization(Gamma(y, y, inertia=1.0, frozen=False), 
			y, "rates", 1.0, False, torch.int32)


def test_initialization_float():
	funcs = (lambda x: x, tuple, 
		lambda x: numpy.array(x, dtype=numpy.float32), 
		lambda x: torch.tensor(x, dtype=torch.float32), 
		lambda x: torch.nn.Parameter(torch.tensor(x), requires_grad=False))

	x = [1.0, 2.2, 3.9, 8.1, 1.0]
	for func in funcs:
		y = func(x)

		_test_initialization(Gamma(y, y, inertia=0.0, frozen=False), 
			y, "shapes", 0.0, False, torch.float32)
		_test_initialization(Gamma(y, y, inertia=0.3, frozen=False), 
			y, "shapes", 0.3, False, torch.float32)
		_test_initialization(Gamma(y, y, inertia=1.0, frozen=True), 
			y, "shapes", 1.0, True, torch.float32)
		_test_initialization(Gamma(y, y, inertia=1.0, frozen=False), 
			y, "shapes", 1.0, False, torch.float32)

		_test_initialization(Gamma(y, y, inertia=0.0, frozen=False), 
			y, "rates", 0.0, False, torch.float32)
		_test_initialization(Gamma(y, y, inertia=0.3, frozen=False), 
			y, "rates", 0.3, False, torch.float32)
		_test_initialization(Gamma(y, y, inertia=1.0, frozen=True), 
			y, "rates", 1.0, True, torch.float32)
		_test_initialization(Gamma(y, y, inertia=1.0, frozen=False), 
			y, "rates", 1.0, False, torch.float32)

	y = numpy.array(x, dtype=numpy.float64)

	_test_initialization(Gamma(y, y, inertia=0.0, frozen=False), 
		y, "shapes", 0.0, False, torch.float64)
	_test_initialization(Gamma(y, y, inertia=0.3, frozen=False), 
		y, "shapes", 0.3, False, torch.float64)
	_test_initialization(Gamma(y, y, inertia=1.0, frozen=True), 
		y, "shapes", 1.0, True, torch.float64)
	_test_initialization(Gamma(y, y, inertia=1.0, frozen=False), 
		y, "shapes", 1.0, False, torch.float64)

	_test_initialization(Gamma(y, y, inertia=0.0, frozen=False), 
		y, "rates", 0.0, False, torch.float64)
	_test_initialization(Gamma(y, y, inertia=0.3, frozen=False), 
		y, "rates", 0.3, False, torch.float64)
	_test_initialization(Gamma(y, y, inertia=1.0, frozen=True), 
		y, "rates", 1.0, True, torch.float64)
	_test_initialization(Gamma(y, y, inertia=1.0, frozen=False), 
		y, "rates", 1.0, False, torch.float64)


def test_initialization_raises():
	_test_initialization_raises_two_parameters(Gamma, VALID_VALUE, VALID_VALUE,
		min_value1=MIN_VALUE, min_value2=MIN_VALUE)


def test_reset_cache(X):
	d = Gamma()
	d.summarize(X)
	assert_array_almost_equal(d._w_sum, [7.0, 7.0, 7.0])
	assert_array_almost_equal(d._xw_sum, [17.099998,  9.7     , 10.599999])
	assert_array_almost_equal(d._logx_w_sum, [4.568669,  0.873965, -0.974132])

	d._reset_cache()
	assert_array_almost_equal(d._w_sum, [0.0, 0.0, 0.0])
	assert_array_almost_equal(d._xw_sum, [0.0, 0.0, 0.0])
	assert_array_almost_equal(d._logx_w_sum, [0.0, 0.0, 0.0])	

	d = Gamma()
	assert_raises(AttributeError, getattr, d, "_w_sum")
	assert_raises(AttributeError, getattr, d, "_xw_sum")
	assert_raises(AttributeError, getattr, d, "_logx_w_sum")

	assert_raises(AttributeError, getattr, d, "_log_rates")
	assert_raises(AttributeError, getattr, d, "_lgamma_shapes")
	assert_raises(AttributeError, getattr, d, "_thetas")

	d._reset_cache()
	assert_raises(AttributeError, getattr, d, "_w_sum")
	assert_raises(AttributeError, getattr, d, "_xw_sum")
	assert_raises(AttributeError, getattr, d, "_logx_w_sum")

	assert_raises(AttributeError, getattr, d, "_log_rates")
	assert_raises(AttributeError, getattr, d, "_lgamma_shapes")
	assert_raises(AttributeError, getattr, d, "_thetas")


def test_initialize(X):
	d = Gamma()
	assert d.d is None
	assert d.shapes is None
	assert d.rates is None
	assert d._initialized == False
	assert_raises(AttributeError, getattr, d, "_w_sum")
	assert_raises(AttributeError, getattr, d, "_xw_sum")
	assert_raises(AttributeError, getattr, d, "_logx_w_sum")

	assert_raises(AttributeError, getattr, d, "_log_rates")
	assert_raises(AttributeError, getattr, d, "_lgamma_shapes")
	assert_raises(AttributeError, getattr, d, "_thetas")

	d = Gamma([1.2], None)
	assert d.d is None
	assert d.rates is None
	assert d._initialized == False
	assert_raises(AttributeError, getattr, d, "_w_sum")
	assert_raises(AttributeError, getattr, d, "_xw_sum")
	assert_raises(AttributeError, getattr, d, "_logx_w_sum")

	assert_raises(AttributeError, getattr, d, "_log_rates")
	assert_raises(AttributeError, getattr, d, "_lgamma_shapes")
	assert_raises(AttributeError, getattr, d, "_thetas")

	d = Gamma(None, [1.2])
	assert d.d is None
	assert d.shapes is None
	assert d._initialized == False
	assert_raises(AttributeError, getattr, d, "_w_sum")
	assert_raises(AttributeError, getattr, d, "_xw_sum")
	assert_raises(AttributeError, getattr, d, "_logx_w_sum")

	assert_raises(AttributeError, getattr, d, "_log_rates")
	assert_raises(AttributeError, getattr, d, "_lgamma_shapes")
	assert_raises(AttributeError, getattr, d, "_thetas")

	d._initialize(3)
	assert d._initialized == True
	assert d.shapes.shape[0] == 3
	assert d.d == 3
	assert_array_almost_equal(d.shapes, [0.0, 0.0, 0.0])
	assert_array_almost_equal(d.rates, [0.0, 0.0, 0.0])
	assert_array_almost_equal(d._w_sum, [0.0, 0.0, 0.0])
	assert_array_almost_equal(d._xw_sum, [0.0, 0.0, 0.0])
	assert_array_almost_equal(d._logx_w_sum, [0.0, 0.0, 0.0])	


	d._initialize(2)
	assert d._initialized == True
	assert d.shapes.shape[0] == 2
	assert d.d == 2
	assert_array_almost_equal(d.shapes, [0.0, 0.0])
	assert_array_almost_equal(d.rates, [0.0, 0.0])
	assert_array_almost_equal(d._w_sum, [0.0, 0.0])
	assert_array_almost_equal(d._xw_sum, [0.0, 0.0])	
	assert_array_almost_equal(d._logx_w_sum, [0.0, 0.0])	


	d = Gamma([1.2, 9.3], [1.1, 9.2])
	assert d._initialized == True
	assert d.d == 2

	d._initialize(3)
	assert d._initialized == True
	assert d.shapes.shape[0] == 3
	assert d.d == 3
	assert_array_almost_equal(d.shapes, [0.0, 0.0, 0.0])
	assert_array_almost_equal(d.rates, [0.0, 0.0, 0.0])

	d = Gamma()
	d.summarize(X)
	d._initialize(4)
	assert d._initialized == True
	assert d.shapes.shape[0] == 4
	assert d.d == 4
	assert_array_almost_equal(d._w_sum, [0.0, 0.0, 0.0, 0.0])
	assert_array_almost_equal(d._xw_sum, [0.0, 0.0, 0.0, 0.0])
	assert_array_almost_equal(d._logx_w_sum, [0.0, 0.0, 0.0, 0.0])	


###


@pytest.mark.sample
def test_sample(shapes, rates):
	torch.manual_seed(0)

	X = Gamma(shapes, rates).sample(1)
	assert_array_almost_equal(X, [[10.794655,  0.367495,  0.568067]])

	X = Gamma(shapes, rates).sample(5)
	assert_array_almost_equal(X, 
		[[0.3594, 0.2441, 1.0691],
         [0.9625, 1.0465, 0.9298],
         [4.2083, 0.6331, 3.7465],
         [3.5261, 1.1585, 1.4492],
         [1.1837, 0.5522, 2.1590]], 3)


###

def test_probability(X, shapes, rates):
	s, r = [1.7], [1.3]
	x = [[1.0], [2.0], [8.0], [3.7], [1.9]]
	y = [4.685216e-01, 2.074282e-01, 2.242917e-04, 3.500235e-02,
           2.278939e-01]

	d1 = Gamma(s, r)
	d2 = Gamma(numpy.array(s, dtype=numpy.float64), 
		numpy.array(r, dtype=numpy.float64))

	_test_predictions(x, y, d1.probability(x), torch.float32)
	_test_predictions(x, y, d2.probability(x), torch.float64)

	x = [[1.0, 2.0, 4]]
	y = [0.000293]

	d1 = Gamma(shapes, rates)
	d2 = Gamma(numpy.array(shapes, dtype=numpy.float64), 
		numpy.array(rates, dtype=numpy.float64))

	_test_predictions(x, y, d1.probability(x), torch.float32)
	_test_predictions(x, y, d2.probability(x), torch.float64)

	s, r = [1, 2, 4], [2, 1, 3]
	x = [[1, 2, 1],
	     [2, 2, 1],
	     [3, 1, 1],
	     [2, 1, 2]]
	y = [0.049242, 0.006664, 0.001226, 0.003608]

	d1 = Gamma(s, r)
	d2 = Gamma(numpy.array(s, dtype=numpy.int32), 
		numpy.array(r, dtype=numpy.int32))

	_test_predictions(x, y, d1.probability(x), torch.float32)
	_test_predictions(x, y, d2.probability(x), torch.float32)

	s, r = [1.0, 2.0, 4.0], [2.0, 1.0, 3.0]
	d1 = Gamma(s, r)
	d2 = Gamma(numpy.array(s, dtype=numpy.float64), 
		numpy.array(r, dtype=numpy.float64))
	_test_predictions(x, y, d1.probability(x), torch.float32)
	_test_predictions(x, y, d2.probability(x), torch.float64)

	y = [0.001497, 0.092372, 0.012607, 0.000671, 0.011402, 0.000144,
           0.002768]

	d1 = Gamma(shapes, rates)
	d2 = Gamma(numpy.array(shapes, dtype=numpy.float64),
		numpy.array(rates, dtype=numpy.float64))

	_test_predictions(X, y, d1.probability(X), torch.float32)
	_test_predictions(X, y, d2.probability(X), torch.float64)


def test_probability_dtypes():
	X = numpy.random.uniform(0, 5, size=(10, 3)).astype(numpy.float32)
	X_int = X.astype('int32')
	p = numpy.array([0.1, 5.3, 2.5], dtype=numpy.float32)
	
	y = Gamma(p, p).probability(X)
	assert y.dtype == torch.float32

	y = Gamma(p, p).probability(X_int)
	assert y.dtype == torch.float32

	X = numpy.random.uniform(0, 5, size=(10, 3)).astype(numpy.float64)
	X_int = X.astype('int32')
	p = numpy.array([0.1, 5.3, 2.5], dtype=numpy.float64)

	y = Gamma(p, p).probability(X)
	assert y.dtype == torch.float64

	y = Gamma(p, p).probability(X_int)
	assert y.dtype == torch.float64


def test_probability_raises(X, shapes, rates):
	_test_raises(Gamma(shapes, rates), "probability", X, min_value=MIN_VALUE, 
		max_value=MAX_VALUE)

	_test_raises(Gamma([VALID_VALUE], [VALID_VALUE]), "probability", X, 
		min_value=MIN_VALUE, max_value=MAX_VALUE)


def test_log_probability(X, shapes, rates):
	s, r = [1.7], [1.3]
	x = [[1.0], [2.0], [8.0], [3.0], [2.0]]
	y = [-0.758173, -1.57297 , -8.402563, -2.589144, -1.57297]

	x_torch = torch.tensor(numpy.array(x))	
	s_torch = torch.tensor(numpy.array(s))
	r_torch = torch.tensor(numpy.array(r))

	d1 = Gamma(s, r)
	d2 = Gamma(numpy.array(s, dtype=numpy.float64),
		numpy.array(r, dtype=numpy.float64))
	d3 = torch.distributions.Gamma(s_torch, r_torch)

	_test_predictions(x, y, d1.log_probability(x), torch.float32)
	_test_predictions(x, y, d2.log_probability(x), torch.float64)
	_test_predictions(x, d3.log_prob(x_torch).sum(axis=1), 
		d2.log_probability(x), torch.float64)

	p = [1.7, 2.3, 1.0, 1.7, 4.1]
	x = [[1.0, 2.0, 8.0, 3.0, 2.0]]
	y = [-16.157602]

	p_torch = torch.tensor(numpy.array(p))
	x_torch = torch.tensor(numpy.array(x))

	d1 = Gamma(p, p)
	d2 = Gamma(numpy.array(p, dtype=numpy.float64),
		numpy.array(p, dtype=numpy.float64))
	d3 = torch.distributions.Gamma(p_torch, p_torch)

	_test_predictions(x, y, d1.log_probability(x), torch.float32)
	_test_predictions(x, y, d2.log_probability(x), torch.float64)
	_test_predictions(x, d3.log_prob(x_torch).sum(axis=1), 
		d2.log_probability(x), torch.float64)

	p = [1, 2, 4]
	x = [[1, 2, 1],
	     [2, 2, 1],
	     [1, 1, 3],
	     [2, 1, 2]]
	y = [-3.16714 , -4.16714 , -6.564451, -4.780846]

	p_torch = torch.tensor(numpy.array(p))
	x_torch = torch.tensor(numpy.array(x))

	d1 = Gamma(p, p)
	d2 = Gamma(numpy.array(p, dtype=numpy.float64),
		numpy.array(p, dtype=numpy.float64))
	d3 = torch.distributions.Gamma(p_torch, p_torch)

	_test_predictions(x, y, d1.log_probability(x), torch.float32)
	_test_predictions(x, y, d2.log_probability(x), torch.float64)
	_test_predictions(x, d3.log_prob(x_torch).sum(axis=1), 
		d2.log_probability(x), torch.float64)

	p = [1.0, 2.0, 4.0]
	p_torch = torch.tensor(numpy.array(p))

	d1 = Gamma(p, p)
	d2 = Gamma(numpy.array(p, dtype=numpy.float64),
		numpy.array(p, dtype=numpy.float64))
	d3 = torch.distributions.Gamma(p_torch, p_torch)

	_test_predictions(x, y, d1.log_probability(x), torch.float32)
	_test_predictions(x, y, d2.log_probability(x), torch.float64)
	_test_predictions(x, d3.log_prob(x_torch).sum(axis=1), 
		d2.log_probability(x), torch.float64)

	s_torch = torch.tensor(numpy.array(shapes))
	r_torch = torch.tensor(numpy.array(rates))
	x_torch = torch.tensor(numpy.array(X))

	y = [-6.504198, -2.381927, -4.373503, -7.307214, -4.473963, -8.848661,
           -5.889609]

	d1 = Gamma(shapes, rates)
	d2 = Gamma(numpy.array(shapes, dtype=numpy.float64),
		numpy.array(rates, dtype=numpy.float64))
	d3 = torch.distributions.Gamma(s_torch, r_torch)

	_test_predictions(X, y, d1.log_probability(X), torch.float32)
	_test_predictions(X, y, d2.log_probability(X), torch.float64)
	_test_predictions(X, d3.log_prob(x_torch).sum(axis=1), 
		d2.log_probability(X), torch.float64)


def test_log_probability_dtypes():
	X = numpy.random.uniform(0, 5, size=(10, 3)).astype(numpy.float32)
	X_int = X.astype('int32')
	p = numpy.array([0.1, 5.3, 2.5], dtype=numpy.float32)
	
	y = Gamma(p, p).log_probability(X)
	assert y.dtype == torch.float32

	y = Gamma(p, p).log_probability(X_int)
	assert y.dtype == torch.float32

	X = numpy.random.uniform(0, 5, size=(10, 3)).astype(numpy.float64)
	X_int = X.astype('int32')
	p = numpy.array([0.1, 5.3, 2.5], dtype=numpy.float64)

	y = Gamma(p, p).log_probability(X)
	assert y.dtype == torch.float64

	y = Gamma(p, p).log_probability(X_int)
	assert y.dtype == torch.float64


def test_log_probability_raises(X, shapes, rates):
	_test_raises(Gamma(shapes, rates), "log_probability", X, 
		min_value=MIN_VALUE, max_value=MAX_VALUE)

	_test_raises(Gamma([VALID_VALUE], [VALID_VALUE]), "log_probability", X, 
		min_value=MIN_VALUE, max_value=MAX_VALUE)


###


def test_summarize(X, shapes, rates):
	d = Gamma(shapes, rates)
	d.summarize(X[:4])
	assert_array_almost_equal(d._w_sum, [4.0, 4.0, 4.0])
	assert_array_almost_equal(d._xw_sum, [6.1, 5.5, 6.2])
	assert_array_almost_equal(d._logx_w_sum, [0.870037, -0.030253,  1.146132])

	d.summarize(X[4:])
	assert_array_almost_equal(d._w_sum, [7.0, 7.0, 7.0])
	assert_array_almost_equal(d._xw_sum, [17.1,  9.7, 10.6])
	assert_array_almost_equal(d._logx_w_sum, [4.568669,  0.873965, -0.974132])


	d = Gamma(shapes, rates)
	d.summarize(X)
	assert_array_almost_equal(d._w_sum, [7.0, 7.0, 7.0])
	assert_array_almost_equal(d._xw_sum, [17.099998,  9.7     , 10.599999])
	assert_array_almost_equal(d._logx_w_sum, [4.568669,  0.873965, -0.974132])


	d = Gamma()
	d.summarize(X[:4])
	assert_array_almost_equal(d._w_sum, [4.0, 4.0, 4.0])
	assert_array_almost_equal(d._xw_sum, [6.1, 5.5, 6.2])
	assert_array_almost_equal(d._logx_w_sum, [0.870037, -0.030253,  1.146132])


	d.summarize(X[4:])
	assert_array_almost_equal(d._w_sum, [7.0, 7.0, 7.0])
	assert_array_almost_equal(d._xw_sum, [17.1,  9.7, 10.6])
	assert_array_almost_equal(d._logx_w_sum, [4.568669,  0.873965, -0.974132])

	d = Gamma()
	d.summarize(X)
	assert_array_almost_equal(d._w_sum, [7.0, 7.0, 7.0])
	assert_array_almost_equal(d._xw_sum, [17.099998,  9.7     , 10.599999])
	assert_array_almost_equal(d._logx_w_sum, [4.568669,  0.873965, -0.974132])


def test_summarize_weighted(X, X2, w, w2, shapes, rates):
	d = Gamma(shapes, rates)
	d.summarize(X[:4], sample_weight=w[:4])
	assert_array_almost_equal(d._w_sum, [3., 3., 3.])
	assert_array_almost_equal(d._xw_sum, [2.1, 2.5, 3.1])
	assert_array_almost_equal(d._logx_w_sum, [-1.290984, -2.476939, -0.168419])

	d.summarize(X[4:], sample_weight=w[4:])
	assert_array_almost_equal(d._w_sum, [11.0, 11.0, 11.0])
	assert_array_almost_equal(d._xw_sum, [28.9     , 12.      ,  8.799999])
	assert_array_almost_equal(d._logx_w_sum, [8.091207, -1.310356, -9.407159])

	d = Gamma(shapes, rates)
	d.summarize(X, sample_weight=w)
	assert_array_almost_equal(d._w_sum, [11.0, 11.0, 11.0])
	assert_array_almost_equal(d._xw_sum, [28.9     , 12.      ,  8.799999])
	assert_array_almost_equal(d._logx_w_sum, [8.091207, -1.310356, -9.407158])

	d = Gamma()
	d.summarize(X2, sample_weight=w2)
	assert_array_almost_equal(d._w_sum, [4.6, 4.6, 4.6, 4.6])
	assert_array_almost_equal(d._xw_sum, [23.02, 4.4, 9.61, 5.94])
	assert_array_almost_equal(d._logx_w_sum, [6.586476, -0.428876,  3.168982,  
		1.039625])

	d = Gamma([0, 0, 0, 0], [0, 0, 0, 0])
	d.summarize(X2, sample_weight=w2)
	assert_array_almost_equal(d._w_sum, [4.6, 4.6, 4.6, 4.6])
	assert_array_almost_equal(d._xw_sum, [23.02, 4.4, 9.61, 5.94])
	assert_array_almost_equal(d._logx_w_sum, [6.586476, -0.428876,  3.168982,  
		1.039625])


def test_summarize_weighted_flat(X, X2, w, w2, shapes, rates):
	w = numpy.array(w)[:,0] 

	d = Gamma(shapes, rates)
	d.summarize(X[:4], sample_weight=w[:4])
	assert_array_almost_equal(d._w_sum, [3., 3., 3.])
	assert_array_almost_equal(d._xw_sum, [2.1, 2.5, 3.1])
	assert_array_almost_equal(d._logx_w_sum, [-1.290984, -2.476939, -0.168419])

	d.summarize(X[4:], sample_weight=w[4:])
	assert_array_almost_equal(d._w_sum, [11.0, 11.0, 11.0])
	assert_array_almost_equal(d._xw_sum, [28.9     , 12.      ,  8.799999])
	assert_array_almost_equal(d._logx_w_sum, [8.091207, -1.310356, -9.407159])

	d = Gamma(shapes, rates)
	d.summarize(X, sample_weight=w)
	assert_array_almost_equal(d._w_sum, [11.0, 11.0, 11.0])
	assert_array_almost_equal(d._xw_sum, [28.9     , 12.      ,  8.799999])
	assert_array_almost_equal(d._logx_w_sum, [8.091207, -1.310356, -9.407158])

	d = Gamma()
	d.summarize(X2, sample_weight=w2)
	assert_array_almost_equal(d._w_sum, [4.6, 4.6, 4.6, 4.6])
	assert_array_almost_equal(d._xw_sum, [23.02, 4.4, 9.61, 5.94])
	assert_array_almost_equal(d._logx_w_sum, [6.586476, -0.428876,  3.168982,  
		1.039625])

	d = Gamma([0, 0, 0, 0], [0, 0, 0, 0])
	d.summarize(X2, sample_weight=w2)
	assert_array_almost_equal(d._w_sum, [4.6, 4.6, 4.6, 4.6])
	assert_array_almost_equal(d._xw_sum, [23.02, 4.4, 9.61, 5.94])
	assert_array_almost_equal(d._logx_w_sum, [6.586476, -0.428876,  3.168982,  
		1.039625])


def test_summarize_weighted_2d(X):
	d = Gamma()
	d.summarize(X[:4], sample_weight=X[:4])
	assert_array_almost_equal(d._w_sum, [6.1, 5.5, 6.2])
	assert_array_almost_equal(d._xw_sum, [13.03    , 10.069999, 11.62])
	assert_array_almost_equal(d._logx_w_sum, [3.736675, 2.89909 , 3.463712])

	d.summarize(X[4:], sample_weight=X[4:])
	assert_array_almost_equal(d._w_sum, [17.1,  9.7, 10.6])
	assert_array_almost_equal(d._xw_sum, [58.59    , 16.369999, 27.720001])
	assert_array_almost_equal(d._logx_w_sum, [18.738672,  4.459686,  8.417439])

	d = Gamma()
	d.summarize(X, sample_weight=X)
	assert_array_almost_equal(d._w_sum, [17.099998,  9.7     , 10.599999])
	assert_array_almost_equal(d._xw_sum, [58.59    , 16.369999, 27.720001])
	assert_array_almost_equal(d._logx_w_sum, [18.73867 ,  4.459686,  8.417439], 
		4)


def test_summarize_dtypes(X, w):
	X = numpy.array(X)
	X = X.astype(numpy.float32)

	p = numpy.array([3.0, 1.1, 2.8], dtype=numpy.float32)
	d = Gamma(p, p)
	assert d._xw_sum.dtype == torch.float32
	d.summarize(X)
	assert d._xw_sum.dtype == torch.float32

	X = X.astype(numpy.float64)
	p = numpy.array([3.0, 1.1, 2.8], dtype=numpy.float32)
	d = Gamma(p, p)
	assert d._xw_sum.dtype == torch.float32
	d.summarize(X)
	assert d._xw_sum.dtype == torch.float32

	X = X.astype(numpy.int32)
	p = numpy.array([3.0, 1.1, 2.8], dtype=numpy.float32)
	d = Gamma(p, p)
	assert d._xw_sum.dtype == torch.float32
	d.summarize(X)
	assert d._xw_sum.dtype == torch.float32

	X = X.astype(numpy.int64)
	p = numpy.array([3.0, 1.1, 2.8], dtype=numpy.float32)
	d = Gamma(p, p)
	assert d._xw_sum.dtype == torch.float32
	d.summarize(X)
	assert d._xw_sum.dtype == torch.float32


def test_summarize_raises(X, w, shapes, rates):
	_test_raises(Gamma(shapes, rates), "summarize", X, w=w, 
		min_value=MIN_VALUE, max_value=MAX_VALUE)

	_test_raises(Gamma(), "summarize", X, w=w, 
		min_value=MIN_VALUE, max_value=MAX_VALUE)

	_test_raises(Gamma([VALID_VALUE], [VALID_VALUE]), "summarize", X, w=w, 
		min_value=MIN_VALUE, max_value=MAX_VALUE)


def _test_fit_params(d, shapes, rates, thetas):
	assert_array_almost_equal(d.shapes, shapes, 4)
	assert_array_almost_equal(d.rates, rates, 4)

	assert_array_almost_equal(d._w_sum, numpy.zeros(d.d))
	assert_array_almost_equal(d._xw_sum, numpy.zeros(d.d))
	assert_array_almost_equal(d._logx_w_sum, numpy.zeros(d.d))

	assert_array_almost_equal(d._log_rates, numpy.log(rates), 4)
	assert_array_almost_equal(d._lgamma_shapes, torch.lgamma(torch.tensor(
		shapes)), 4)
	assert_array_almost_equal(d._thetas, thetas, 4)


def test_from_summaries(X, shapes, rates):
	d = Gamma(shapes, rates)
	d.summarize(X)
	d.from_summaries()
	assert_raises(AssertionError, assert_array_almost_equal, shapes, d.shapes)

	for param1, param2 in (shapes, rates), (None, None):
		d = Gamma(param1, param2)
		d.summarize(X[:4])
		d.from_summaries()
		_test_fit_params(d, [2.599745, 1.681025, 3.453269], 
			[1.704751, 1.222564, 2.227916], [1.029533, 0.437424, 1.616514] )	

		d.summarize(X[4:])
		d.from_summaries()
		_test_fit_params(d, [7.692329, 14.423422,  0.571159], 
			[2.097908, 10.302444,  0.389426], [-2.211836,  9.980053, -0.98287])

		d = Gamma(param1, param2)
		d.summarize(X[:4])
		d.summarize(X[4:])
		d.from_summaries()
		_test_fit_params(d, [2.231405, 2.63784 , 1.037289],
			[0.913441, 1.903595, 0.685002], [-0.316346,  1.31193 , -0.372041])

		d = Gamma(param1, param2)
		d.summarize(X)
		d.from_summaries()
		_test_fit_params(d, [2.231407, 2.637839 , 1.037289],
			[0.913441, 1.903595, 0.685002], [-0.316346,  1.31193 , -0.372041])


def test_from_summaries_weighted(X, w, shapes, rates):
	for param in shapes, None:
		d = Gamma(shapes, rates)
		d.summarize(X[:4], sample_weight=w[:4])
		d.from_summaries()
		_test_fit_params(d, [6.951008, 0.907765, 5.783966],
			[9.930012, 1.089317, 5.597386], [9.468783, 0.017088, 5.538523])
	
		d.summarize(X[4:], sample_weight=w[4:])
		d.from_summaries()
		_test_fit_params(d, [13.98193 , 19.375662,  0.73567],
			[4.17371 , 16.316347,  1.032519], [-2.527685, 16.604717, -0.195563])

		d = Gamma(shapes, rates)
		d.summarize(X, sample_weight=w)
		d.from_summaries()
		_test_fit_params(d, [2.323329, 2.580077, 0.922164],
			[0.884312, 2.365071, 1.152706], [-0.453977,  1.878409,  0.08094])
	

def test_from_summaries_null():
	d = Gamma([1, 2], [1, 2])
	d.from_summaries()
	assert d.shapes[0] != 1 and d.shapes[1] != 2
	assert d.rates[0] != 1 and d.rates[1] != 2 
	assert_array_almost_equal(d._w_sum, [0.0, 0.0])
	assert_array_almost_equal(d._xw_sum, [0.0, 0.0])
	assert_array_almost_equal(d._logx_w_sum, [0.0, 0.0])

	d = Gamma([1, 2], [1, 2], inertia=0.5)
	d.from_summaries()
	assert d.shapes[0] != 1 and d.shapes[1] != 2
	assert d.rates[0] != 1 and d.rates[1] != 2 
	assert_array_almost_equal(d._w_sum, [0.0, 0.0])
	assert_array_almost_equal(d._xw_sum, [0.0, 0.0])
	assert_array_almost_equal(d._logx_w_sum, [0.0, 0.0])


	d = Gamma([1, 2], [1, 2], inertia=0.5, frozen=True)
	d.from_summaries()
	_test_fit_params(d, [1, 2], [1, 2], [0, 1.386294])


def test_from_summaries_inertia(X, w, shapes, rates):
	d = Gamma(shapes, rates, inertia=0.3)
	d.summarize(X[:4])
	d.from_summaries()
	_test_fit_params(d, [2.179822, 1.716718, 3.047288],
		[1.283326, 1.785795, 1.919541], [0.457686, 1.087673, 1.249872])

	d.summarize(X[4:])
	d.from_summaries()
	_test_fit_params(d, [6.038577, 10.611411,  1.313998],
		[1.853533, 7.747449, 0.848461], [-1.127075,  7.527596, -0.105499])

	d = Gamma(shapes, rates, inertia=0.3)
	d.summarize(X)
	d.from_summaries()
	_test_fit_params(d, [1.921985, 2.386487, 1.356102],
		[0.729409, 2.262517, 0.839501], [-0.575438,  1.74043 , -0.12133])


def test_from_summaries_weighted_inertia(X, w, shapes, rates):
	d = Gamma(shapes, rates, inertia=0.3)
	d.summarize(X, sample_weight=w)
	d.from_summaries()
	_test_fit_params(d, [1.98633 , 2.346054, 1.275515],
		[0.709018, 2.58555 , 1.166894], [-0.677328,  2.046211,  0.300559])

	d = Gamma(shapes, rates, inertia=1.0)
	d.summarize(X[:4])
	d.from_summaries()
	_test_fit_params(d, shapes, rates, [-1.359393,  2.107607,  0.337438])

	d.summarize(X[4:])
	d.from_summaries()
	_test_fit_params(d, shapes, rates, [-1.359393,  2.107607,  0.337438])

	d = Gamma(shapes, rates, inertia=1.0)
	d.summarize(X)
	d.from_summaries()
	_test_fit_params(d, shapes, rates, [-1.359393,  2.107607,  0.337438])


def test_from_summaries_frozen(X, w, shapes, rates):
	d = Gamma(shapes, rates, frozen=True)
	d.summarize(X[:4])
	assert_array_almost_equal(d._w_sum, [0.0, 0.0, 0.0])
	assert_array_almost_equal(d._xw_sum, [0.0, 0.0, 0.0])

	d.from_summaries()
	_test_fit_params(d, shapes, rates, [-1.359393,  2.107607,  0.337438])

	d.summarize(X[4:])
	assert_array_almost_equal(d._w_sum, [0.0, 0.0, 0.0])
	assert_array_almost_equal(d._xw_sum, [0.0, 0.0, 0.0])

	d.from_summaries()
	_test_fit_params(d, shapes, rates, [-1.359393,  2.107607,  0.337438])

	d = Gamma(shapes, rates, frozen=True)
	d.summarize(X)
	assert_array_almost_equal(d._w_sum, [0.0, 0.0, 0.0])
	assert_array_almost_equal(d._xw_sum, [0.0, 0.0, 0.0])

	d.from_summaries()
	_test_fit_params(d, shapes, rates, [-1.359393,  2.107607,  0.337438])

	d = Gamma(shapes, rates, frozen=True)
	d.summarize(X, sample_weight=w)
	assert_array_almost_equal(d._w_sum, [0.0, 0.0, 0.0])
	assert_array_almost_equal(d._xw_sum, [0.0, 0.0, 0.0])

	d.from_summaries()
	_test_fit_params(d, shapes, rates, [-1.359393,  2.107607,  0.337438])

def test_from_summaries_dtypes(X, shapes):
	X = numpy.array(X)
	X = X.astype(numpy.float32)

	p = numpy.array(shapes, dtype=numpy.float32)
	d = Gamma(p, p)
	d.summarize(X)
	d.from_summaries()
	assert d.shapes.dtype == torch.float32
	assert d.rates.dtype == torch.float32
	assert d._log_rates.dtype == torch.float32
	assert d._lgamma_shapes.dtype == torch.float32


	p = numpy.array(shapes, dtype=numpy.float64)
	d = Gamma(p, p)
	d.summarize(X)
	d.from_summaries()
	assert d.shapes.dtype == torch.float64
	assert d.rates.dtype == torch.float64
	assert d._log_rates.dtype == torch.float64
	assert d._lgamma_shapes.dtype == torch.float64


	p = numpy.array(shapes, dtype=numpy.int32)
	d = Gamma(p, p)
	d.summarize(X)
	d.from_summaries()
	assert d.shapes.dtype == torch.int32
	assert d.rates.dtype == torch.int32
	assert d._log_rates.dtype == torch.float32
	assert d._lgamma_shapes.dtype == torch.float32


	X = numpy.array(X)
	X = X.astype(numpy.float64)

	p = numpy.array(shapes, dtype=numpy.float64)
	d = Gamma(p, p)
	d.summarize(X)
	d.from_summaries()
	assert d.shapes.dtype == torch.float64
	assert d.rates.dtype == torch.float64
	assert d._log_rates.dtype == torch.float64
	assert d._lgamma_shapes.dtype == torch.float64


def test_from_summaries_raises():
	assert_raises(AttributeError, Gamma().from_summaries)


def test_fit(X, shapes, rates):
	d = Gamma(shapes, rates)
	d.fit(X)
	assert_raises(AssertionError, assert_array_almost_equal, shapes, d.shapes)

	for param1, param2 in (shapes, rates), (None, None):
		d = Gamma(param1, param2)
		d.fit(X[:4])
		_test_fit_params(d, [2.599745, 1.681025, 3.453269], 
			[1.704751, 1.222564, 2.227916], [1.029533, 0.437424, 1.616514] )	

		d.fit(X[4:])
		_test_fit_params(d, [7.692329, 14.423422,  0.571159], 
			[2.097908, 10.302444,  0.389426], [-2.211836,  9.980053, -0.98287])

		d = Gamma(param1, param2)
		d.fit(X)
		_test_fit_params(d, [2.231407, 2.637839 , 1.037289],
			[0.913441, 1.903595, 0.685002], [-0.316346,  1.31193 , -0.372041])


def test_fit_weighted(X, w, shapes, rates):
	for param in shapes, None:
		d = Gamma(shapes, rates)
		d.fit(X[:4], sample_weight=w[:4])
		_test_fit_params(d, [6.951008, 0.907765, 5.783966],
			[9.930012, 1.089317, 5.597386], [9.468783, 0.017088, 5.538523])
	
		d.fit(X[4:], sample_weight=w[4:])
		_test_fit_params(d, [13.98193 , 19.375662,  0.73567],
			[4.17371 , 16.316347,  1.032519], [-2.527685, 16.604717, -0.195563])

		d = Gamma(shapes, rates)
		d.fit(X, sample_weight=w)
		_test_fit_params(d, [2.323329, 2.580077, 0.922164],
			[0.884312, 2.365071, 1.152706], [-0.453977,  1.878409,  0.08094])
	
	X = [[1.2, 0.5, 1.1, 1.9],
	     [6.2, 1.1, 2.4, 1.1]] 

	w = [[1.1], [3.5]]

	d = Gamma()
	d.fit(X, sample_weight=w)
	_test_fit_params(d, [2.957831, 10.413525, 10.615726, 17.029907],
		[0.591052, 10.886867,  5.081409, 13.188144], 
		[-2.209965, 11.121039,  3.048994, 13.169821])


def test_fit_chain(X):
	d = Gamma().fit(X[:4])
	_test_fit_params(d, [2.599745, 1.681025, 3.453269],
			[1.704751, 1.222564, 2.227916], [1.029533, 0.437424, 1.616514])

	d.fit(X[4:])
	_test_fit_params(d, [7.692329, 14.423422,  0.571159],
			[2.097908, 10.302444,  0.389426], [-2.211836,  9.980053, -0.98287])

	d = Gamma().fit(X)
	_test_fit_params(d, [2.231407, 2.637839, 1.037289],
			[0.913442, 1.903595, 0.685002], [-0.316346,  1.31193 , -0.372041])


def test_fit_dtypes(X, shapes):
	X = numpy.array(X)
	X = X.astype(numpy.float32)

	p = numpy.array(shapes, dtype=numpy.float32)
	d = Gamma(p, p)
	d.fit(X)
	assert d.shapes.dtype == torch.float32
	assert d.rates.dtype == torch.float32
	assert d._log_rates.dtype == torch.float32
	assert d._lgamma_shapes.dtype == torch.float32

	p = numpy.array(shapes, dtype=numpy.float64)
	d = Gamma(p, p)
	d.fit(X)
	assert d.shapes.dtype == torch.float64
	assert d.rates.dtype == torch.float64
	assert d._log_rates.dtype == torch.float64
	assert d._lgamma_shapes.dtype == torch.float64


	p = numpy.array(shapes, dtype=numpy.int32)
	d = Gamma(p, p)
	d.fit(X)
	assert d.shapes.dtype == torch.int32
	assert d.rates.dtype == torch.int32
	assert d._log_rates.dtype == torch.float32
	assert d._lgamma_shapes.dtype == torch.float32


	X = numpy.array(X)
	X = X.astype(numpy.float64)

	p = numpy.array(shapes, dtype=numpy.float64)
	d = Gamma(p, p)
	d.fit(X)
	assert d.shapes.dtype == torch.float64
	assert d.rates.dtype == torch.float64
	assert d._log_rates.dtype == torch.float64
	assert d._lgamma_shapes.dtype == torch.float64


def test_serialization(X):
	d = Gamma().fit(X[:4])
	d.summarize(X[4:])

	rates = [1.704751, 1.222564, 2.227916]

	assert_array_almost_equal(d.rates, rates, 4)
	assert_array_almost_equal(d._log_rates, numpy.log(rates), 4)

	torch.save(d, ".pytest.torch")
	d2 = torch.load(".pytest.torch")
	os.system("rm .pytest.torch")

	assert_array_almost_equal(d2.rates, rates)
	assert_array_almost_equal(d2._log_rates, numpy.log(rates))

	assert_array_almost_equal(d2._w_sum, [3., 3., 3.])
	assert_array_almost_equal(d2._xw_sum, [11. ,  4.2,  4.4])
	assert_array_almost_equal(d.log_probability(X), d2.log_probability(X), 4)
	

def test_masked_probability(shapes, rates, X, X_masked):
	X = torch.tensor(numpy.array(X))
	y = [0.001497, 0.092372, 0.012607, 0.000671, 0.011402, 0.000144,
           0.002768]

	d = Gamma(shapes, rates)
	mask = torch.ones_like(X).type(torch.bool)
	X_ = torch.masked.MaskedTensor(X, mask=mask)

	assert_array_almost_equal(y, d.probability(X_)._masked_data)

	y =  [7.955700e-03, 2.350480e-01, 1.000000e+00, 6.706825e-04,
           3.706888e-01, 1.435738e-04, 1.534282e-02]

	assert_array_almost_equal(y, d.probability(X_masked)._masked_data)


def test_masked_log_probability(shapes, rates, X, X_masked):
	X = torch.tensor(numpy.array(X))
	y = [-6.504198, -2.381927, -4.373503, -7.307214, -4.473963, -8.848661,
           -5.889609]

	d = Gamma(shapes, rates)
	mask = torch.ones_like(X).type(torch.bool)
	X_ = torch.masked.MaskedTensor(X, mask=mask)

	assert_array_almost_equal(y, d.log_probability(X_)._masked_data)

	y = [-4.833867, -1.447965,  0.      , -7.307215, -0.992392, -8.848662,
           -4.177108]

	assert_array_almost_equal(y, d.log_probability(X_masked)._masked_data)


def test_masked_summarize(X, X_masked, w):
	X = torch.tensor(numpy.array(X))
	mask = torch.ones_like(X).type(torch.bool)
	X_ = torch.masked.MaskedTensor(X, mask=mask)

	d = Gamma()
	d.summarize(X, sample_weight=w)
	assert_array_almost_equal(d._w_sum, [11.0, 11.0, 11.0])
	assert_array_almost_equal(d._xw_sum, [28.9, 12. ,  8.8])

	d = Gamma()
	d.summarize(X_masked)
	assert_array_almost_equal(d._w_sum, [4.0, 5.0, 4.0])
	assert_array_almost_equal(d._xw_sum, [11.2,  7.3,  6.8])


def test_masked_from_summaries(X, X_masked):
	X = torch.tensor(numpy.array(X))
	mask = torch.ones_like(X).type(torch.bool)
	X_ = torch.masked.MaskedTensor(X, mask=mask)

	d = Gamma()
	d.summarize(X_)
	d.from_summaries()
	_test_fit_params(d, [2.231407, 2.637839 , 1.037289],
		[0.913441, 1.903595, 0.685002], [-0.316346,  1.31193 , -0.372041])

	d = Gamma()
	d.summarize(X_masked)
	d.from_summaries()
	_test_fit_params(d, [1.8081, 1.9674, 0.8058], [0.6457, 1.3475, 0.474], 
		[-0.722 ,  0.6002, -0.7481])


def test_masked_fit(X, X_masked):
	X = torch.tensor(numpy.array(X))
	mask = torch.ones_like(X).type(torch.bool)
	X_ = torch.masked.MaskedTensor(X, mask=mask)

	d = Gamma()
	d.fit(X_)
	_test_fit_params(d, [2.231407, 2.637839 , 1.037289],
		[0.913441, 1.903595, 0.685002], [-0.316346,  1.31193 , -0.372041])

	d = Gamma()
	d.fit(X_masked)
	_test_fit_params(d, [1.8081, 1.9674, 0.8058], [0.6457, 1.3475, 0.474], 
		[-0.722 ,  0.6002, -0.7481])

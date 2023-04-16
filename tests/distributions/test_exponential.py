# test_exponential.py
# Contact: Jacob Schreiber <jmschreiber91@gmail.com>

import os
import numpy
import torch
import pytest

from torchegranate.distributions import Exponential

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
	     [0, 0, 1],
	     [1, 1, 2],
	     [2, 2, 2],
	     [3, 1, 0],
	     [5, 1, 4],
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
def scales():
	return [1.2, 1.8, 2.1]


###


def test_initialization():
	d = Exponential()
	_test_initialization(d, None, "scales", 0.0, False, None)
	assert_raises(AttributeError, getattr, d, "_w_sum")
	assert_raises(AttributeError, getattr, d, "_xw_sum")
	assert_raises(AttributeError, getattr, d, "_log_scales")


def test_initialization_int():
	funcs = (lambda x: x, tuple, numpy.array, torch.tensor, 
		lambda x: torch.nn.Parameter(torch.tensor(x), requires_grad=False))

	x = [1, 2, 3, 8, 1]
	for func in funcs:
		y = func(x)
		_test_initialization(Exponential(y, inertia=0.0, frozen=False), 
			y, "scales", 0.0, False, torch.int64)
		_test_initialization(Exponential(y, inertia=0.3, frozen=False), 
			y, "scales", 0.3, False, torch.int64)
		_test_initialization(Exponential(y, inertia=1.0, frozen=True), 
			y, "scales", 1.0, True, torch.int64)
		_test_initialization(Exponential(y, inertia=1.0, frozen=False), 
			y, "scales", 1.0, False, torch.int64)

	x = numpy.array(x, dtype=numpy.int32)
	for func in funcs[2:]:
		y = func(x)
		_test_initialization(Exponential(y, inertia=0.0, frozen=False), 
			y, "scales", 0.0, False, torch.int32)
		_test_initialization(Exponential(y, inertia=0.3, frozen=False), 
			y, "scales", 0.3, False, torch.int32)
		_test_initialization(Exponential(y, inertia=1.0, frozen=True), 
			y, "scales", 1.0, True, torch.int32)
		_test_initialization(Exponential(y, inertia=1.0, frozen=False), 
			y, "scales", 1.0, False, torch.int32)


def test_initialization_float():
	funcs = (lambda x: x, tuple, 
		lambda x: numpy.array(x, dtype=numpy.float32), 
		lambda x: torch.tensor(x, dtype=torch.float32), 
		lambda x: torch.nn.Parameter(torch.tensor(x), requires_grad=False))

	x = [1.0, 2.2, 3.9, 8.1, 1.0]
	for func in funcs:
		y = func(x)
		_test_initialization(Exponential(y, inertia=0.0, frozen=False), 
			y, "scales", 0.0, False, torch.float32)
		_test_initialization(Exponential(y, inertia=0.3, frozen=False), 
			y, "scales", 0.3, False, torch.float32)
		_test_initialization(Exponential(y, inertia=1.0, frozen=True), 
			y, "scales", 1.0, True, torch.float32)
		_test_initialization(Exponential(y, inertia=1.0, frozen=False), 
			y, "scales", 1.0, False, torch.float32)

	x = numpy.array(x, dtype=numpy.float64)
	_test_initialization(Exponential(x, inertia=0.0, frozen=False), 
		x, "scales", 0.0, False, torch.float64)
	_test_initialization(Exponential(x, inertia=0.3, frozen=False), 
		x, "scales", 0.3, False, torch.float64)
	_test_initialization(Exponential(x, inertia=1.0, frozen=True), 
		x, "scales", 1.0, True, torch.float64)
	_test_initialization(Exponential(x, inertia=1.0, frozen=False), 
		x, "scales", 1.0, False, torch.float64)


def test_initialization_raises():
	_test_initialization_raises_one_parameter(Exponential, VALID_VALUE, 
		min_value=MIN_VALUE)


def test_reset_cache(X):
	d = Exponential()
	d.summarize(X)
	assert_array_almost_equal(d._w_sum, [7.0, 7.0, 7.0])
	assert_array_almost_equal(d._xw_sum, [14.0, 8.0, 9.0])

	d._reset_cache()
	assert_array_almost_equal(d._w_sum, [0.0, 0.0, 0.0])
	assert_array_almost_equal(d._xw_sum, [0.0, 0.0, 0.0])	

	d = Exponential()
	assert_raises(AttributeError, getattr, d, "_w_sum")
	assert_raises(AttributeError, getattr, d, "_xw_sum")
	assert_raises(AttributeError, getattr, d, "_log_scales")

	d._reset_cache()
	assert_raises(AttributeError, getattr, d, "_w_sum")
	assert_raises(AttributeError, getattr, d, "_xw_sum")
	assert_raises(AttributeError, getattr, d, "_log_scales")


def test_initialize(X):
	d = Exponential()
	assert d.d is None
	assert d.scales is None
	assert d._initialized == False
	assert_raises(AttributeError, getattr, d, "_w_sum")
	assert_raises(AttributeError, getattr, d, "_xw_sum")
	assert_raises(AttributeError, getattr, d, "_log_scales")

	d._initialize(3)
	assert d._initialized == True
	assert d.scales.shape[0] == 3
	assert d.d == 3
	assert_array_almost_equal(d.scales, [0.0, 0.0, 0.0])
	assert_array_almost_equal(d._w_sum, [0.0, 0.0, 0.0])
	assert_array_almost_equal(d._xw_sum, [0.0, 0.0, 0.0])	

	d._initialize(2)
	assert d._initialized == True
	assert d.scales.shape[0] == 2
	assert d.d == 2
	assert_array_almost_equal(d.scales, [0.0, 0.0])
	assert_array_almost_equal(d._w_sum, [0.0, 0.0])
	assert_array_almost_equal(d._xw_sum, [0.0, 0.0])	

	d = Exponential([1.2, 9.3])
	assert d._initialized == True
	assert d.d == 2

	d._initialize(3)
	assert d._initialized == True
	assert d.scales.shape[0] == 3
	assert d.d == 3
	assert_array_almost_equal(d.scales, [0.0, 0.0, 0.0])

	d = Exponential()
	d.summarize(X)
	d._initialize(4)
	assert d._initialized == True
	assert d.scales.shape[0] == 4
	assert d.d == 4
	assert_array_almost_equal(d._w_sum, [0.0, 0.0, 0.0, 0.0])
	assert_array_almost_equal(d._xw_sum, [0.0, 0.0, 0.0, 0.0])	


###


@pytest.mark.sample
def test_sample(scales):
	torch.manual_seed(0)

	X = Exponential(scales).sample(1)
	assert_array_almost_equal(X, [[4.209991, 2.214692, 1.291593]])

	X = Exponential(scales).sample(5)
	assert_array_almost_equal(X, 
		[[3.0421, 1.8643, 3.2889],
         [0.2361, 0.7785, 1.8285],
         [0.4080, 1.0855, 0.4086],
         [0.5267, 1.7507, 1.3807],
         [0.6975, 0.9415, 0.4829]], 3)


###


def test_probability(X, scales):
	p = [1.7]
	x = [[1.0], [2.0], [8.0], [3.7], [1.9]]
	y = [0.326651, 0.181391, 0.005319, 0.06673 , 0.192381]

	d1 = Exponential(p)
	d2 = Exponential(numpy.array(p, dtype=numpy.float64))

	_test_predictions(x, y, d1.probability(x), torch.float32)
	_test_predictions(x, y, d2.probability(x), torch.float64)

	x = [[1.0, 2.0, 4]]
	y = [0.004695]

	d1 = Exponential(scales)
	d2 = Exponential(numpy.array(scales, dtype=numpy.float64))

	_test_predictions(x, y, d1.probability(x), torch.float32)
	_test_predictions(x, y, d2.probability(x), torch.float64)

	p = [1, 2, 4]
	x = [[1, 2, 1],
	     [2, 2, 1],
	     [0, 1, 0],
	     [0, 0, 2]]
	y = [0.013175, 0.004847, 0.075816, 0.075816]

	d1 = Exponential(p)
	d2 = Exponential(numpy.array(p, dtype=numpy.float64))

	_test_predictions(x, y, d1.probability(x), torch.float32)
	_test_predictions(x, y, d2.probability(x), torch.float64)

	p = [1.0, 2.0, 4.0]
	d1 = Exponential(p)
	d2 = Exponential(numpy.array(p, dtype=numpy.float64))
	_test_predictions(x, y, d1.probability(x), torch.float32)
	_test_predictions(x, y, d2.probability(x), torch.float64)

	p_torch = torch.tensor(numpy.array(scales))
	y = [0.03154 , 0.136937, 0.021209, 0.005289, 0.010383, 0.000292,
           0.023891]

	d1 = Exponential(scales)
	d2 = Exponential(numpy.array(scales, dtype=numpy.float64))

	_test_predictions(X, y, d1.probability(X), torch.float32)
	_test_predictions(X, y, d2.probability(X), torch.float64)


def test_probability_dtypes():
	X = numpy.random.uniform(0, 5, size=(10, 3)).astype(numpy.float32)
	X_int = X.astype('int32')
	p = numpy.array([0.1, 5.3, 2.5], dtype=numpy.float32)
	
	y = Exponential(p).probability(X)
	assert y.dtype == torch.float32

	y = Exponential(p).probability(X_int)
	assert y.dtype == torch.float32

	X = numpy.random.uniform(0, 5, size=(10, 3)).astype(numpy.float64)
	X_int = X.astype('int32')
	p = numpy.array([0.1, 5.3, 2.5], dtype=numpy.float64)

	y = Exponential(p).probability(X)
	assert y.dtype == torch.float64

	y = Exponential(p).probability(X_int)
	assert y.dtype == torch.float64


def test_probability_raises(X, scales):
	_test_raises(Exponential(scales), "probability", X, min_value=MIN_VALUE, 
		max_value=MAX_VALUE)

	_test_raises(Exponential([VALID_VALUE]), "probability", X, 
		min_value=MIN_VALUE, max_value=MAX_VALUE)


def test_log_probability(X, scales):
	p = [1.7]
	x = [[1.0], [2.0], [8.0], [3.7], [1.9]]
	y = [-1.118864, -1.707099, -5.23651 , -2.707099, -1.648275]

	x_torch = torch.tensor(numpy.array(x))	
	p_torch = torch.tensor(numpy.array(p))

	d1 = Exponential(p)
	d2 = Exponential(numpy.array(p, dtype=numpy.float64))
	d3 = torch.distributions.Exponential(1. / p_torch)

	_test_predictions(x, y, d1.log_probability(x), torch.float32)
	_test_predictions(x, y, d2.log_probability(x), torch.float64)
	_test_predictions(x, d3.log_prob(x_torch).sum(axis=1), 
		d2.log_probability(x), torch.float64)

	p = [1.7, 2.3, 0.9, 1.7, 4.1]
	x = [[1.0, 2.0, 8.0, 3.7, 1.9]]
	y = [-16.186367]

	p_torch = torch.tensor(numpy.array(p))
	x_torch = torch.tensor(numpy.array(x))

	d1 = Exponential(p)
	d2 = Exponential(numpy.array(p, dtype=numpy.float64))
	d3 = torch.distributions.Exponential(1. / p_torch)

	_test_predictions(x, y, d1.log_probability(x), torch.float32)
	_test_predictions(x, y, d2.log_probability(x), torch.float64)
	_test_predictions(x, d3.log_prob(x_torch).sum(axis=1), 
		d2.log_probability(x), torch.float64)

	p = [1, 2, 4]
	x = [[1, 2, 1],
	     [2, 2, 1],
	     [0, 1, 0],
	     [0, 0, 2]]
	y = [-4.329442, -5.329442, -2.579442, -2.579442]

	p_torch = torch.tensor(numpy.array(p))
	x_torch = torch.tensor(numpy.array(x))

	d1 = Exponential(p)
	d2 = Exponential(numpy.array(p, dtype=numpy.float64))
	d3 = torch.distributions.Exponential(1. / p_torch)

	_test_predictions(x, y, d1.log_probability(x), torch.float32)
	_test_predictions(x, y, d2.log_probability(x), torch.float64)
	_test_predictions(x, d3.log_prob(x_torch).sum(axis=1), 
		d2.log_probability(x), torch.float64)

	p = [1.0, 2.0, 4.0]
	d1 = Exponential(p)
	d2 = Exponential(numpy.array(p, dtype=numpy.float64))
	d3 = torch.distributions.Exponential(1. / p_torch)

	p_torch = torch.tensor(numpy.array(p))
	x_torch = torch.tensor(numpy.array(x))

	_test_predictions(x, y, d1.log_probability(x), torch.float32)
	_test_predictions(x, y, d2.log_probability(x), torch.float64)
	_test_predictions(x, d3.log_prob(x_torch).sum(axis=1), 
		d2.log_probability(x), torch.float64)

	p_torch = torch.tensor(numpy.array(scales))
	y = [-3.45649 , -1.988236, -3.853315, -5.242204, -4.567601, -8.13903,
           -3.734268]

	d1 = Exponential(scales)
	d2 = Exponential(numpy.array(scales, dtype=numpy.float64))
	d3 = torch.distributions.Exponential(1. / p_torch)

	_test_predictions(X, y, d1.log_probability(X), torch.float32)
	_test_predictions(X, y, d2.log_probability(X), torch.float64)
	_test_predictions(x, d3.log_prob(x_torch).sum(axis=1), 
		d2.log_probability(x), torch.float64)


def test_log_probability_dtypes():
	X = numpy.random.uniform(0, 5, size=(10, 3)).astype(numpy.float32)
	X_int = X.astype('int32')
	p = numpy.array([0.1, 5.3, 2.5], dtype=numpy.float32)
	
	y = Exponential(p).log_probability(X)
	assert y.dtype == torch.float32

	y = Exponential(p).log_probability(X_int)
	assert y.dtype == torch.float32

	X = numpy.random.uniform(0, 5, size=(10, 3)).astype(numpy.float64)
	X_int = X.astype('int32')
	p = numpy.array([0.1, 5.3, 2.5], dtype=numpy.float64)

	y = Exponential(p).log_probability(X)
	assert y.dtype == torch.float64

	y = Exponential(p).log_probability(X_int)
	assert y.dtype == torch.float64


def test_log_probability_raises(X, scales):
	_test_raises(Exponential(scales), "log_probability", X, min_value=MIN_VALUE, 
		max_value=MAX_VALUE)

	_test_raises(Exponential([VALID_VALUE]), "log_probability", X, 
		min_value=MIN_VALUE, max_value=MAX_VALUE)


###


def test_summarize(X, X2, scales):
	d = Exponential(scales)
	d.summarize(X[:4])
	assert_array_almost_equal(d._w_sum, [4.0, 4.0, 4.0])
	assert_array_almost_equal(d._xw_sum, [4.0, 5.0, 5.0])

	d.summarize(X[4:])
	assert_array_almost_equal(d._w_sum, [7.0, 7.0, 7.0])
	assert_array_almost_equal(d._xw_sum, [14.0, 8.0, 9.0])

	d = Exponential(scales)
	d.summarize(X)
	assert_array_almost_equal(d._w_sum, [7.0, 7.0, 7.0])
	assert_array_almost_equal(d._xw_sum, [14.0, 8.0, 9.0])


	d = Exponential()
	d.summarize(X[:4])
	assert_array_almost_equal(d._w_sum, [4.0, 4.0, 4.0])
	assert_array_almost_equal(d._xw_sum, [4.0, 5.0, 5.0])

	d.summarize(X[4:])
	assert_array_almost_equal(d._w_sum, [7.0, 7.0, 7.0])
	assert_array_almost_equal(d._xw_sum, [14.0, 8.0, 9.0])

	d = Exponential()
	d.summarize(X)
	assert_array_almost_equal(d._w_sum, [7.0, 7.0, 7.0])
	assert_array_almost_equal(d._xw_sum, [14.0, 8.0, 9.0])


def test_summarize_weighted(X, X2, w, w2, scales):
	d = Exponential(scales)
	d.summarize(X[:4], sample_weight=w[:4])
	assert_array_almost_equal(d._w_sum, [3., 3., 3.])
	assert_array_almost_equal(d._xw_sum, [1., 2., 2.])

	d.summarize(X[4:], sample_weight=w[4:])
	assert_array_almost_equal(d._w_sum, [11.0, 11.0, 11.0])
	assert_array_almost_equal(d._xw_sum, [25.0, 10.0, 6.0])

	d = Exponential(scales)
	d.summarize(X, sample_weight=w)
	assert_array_almost_equal(d._w_sum, [11.0, 11.0, 11.0])
	assert_array_almost_equal(d._xw_sum, [25.0, 10.0, 6.0])


	d = Exponential()
	d.summarize(X2, sample_weight=w2)
	assert_array_almost_equal(d._w_sum, [4.6, 4.6, 4.6, 4.6])
	assert_array_almost_equal(d._xw_sum, [23.02, 4.4, 9.61, 5.94])

	d = Exponential([0, 0, 0, 0])
	d.summarize(X2, sample_weight=w2)
	assert_array_almost_equal(d._w_sum, [4.6, 4.6, 4.6, 4.6])
	assert_array_almost_equal(d._xw_sum, [23.02, 4.4, 9.61, 5.94])


def test_summarize_weighted_flat(X, X2, w, w2, scales):
	w = numpy.array(w)[:,0] 

	d = Exponential(scales)
	d.summarize(X[:4], sample_weight=w[:4])
	assert_array_almost_equal(d._w_sum, [3., 3., 3.])
	assert_array_almost_equal(d._xw_sum, [1., 2., 2.])

	d.summarize(X[4:], sample_weight=w[4:])
	assert_array_almost_equal(d._w_sum, [11.0, 11.0, 11.0])
	assert_array_almost_equal(d._xw_sum, [25.0, 10.0, 6.0])

	d = Exponential(scales)
	d.summarize(X, sample_weight=w)
	assert_array_almost_equal(d._w_sum, [11.0, 11.0, 11.0])
	assert_array_almost_equal(d._xw_sum, [25.0, 10.0, 6.0])


	d = Exponential()
	d.summarize(X2, sample_weight=w2)
	assert_array_almost_equal(d._w_sum, [4.6, 4.6, 4.6, 4.6])
	assert_array_almost_equal(d._xw_sum, [23.02, 4.4, 9.61, 5.94])

	d = Exponential([0, 0, 0, 0])
	d.summarize(X2, sample_weight=w2)
	assert_array_almost_equal(d._w_sum, [4.6, 4.6, 4.6, 4.6])
	assert_array_almost_equal(d._xw_sum, [23.02, 4.4, 9.61, 5.94])


def test_summarize_weighted_2d(X):
	d = Exponential()
	d.summarize(X[:4], sample_weight=X[:4])
	assert_array_almost_equal(d._w_sum, [4., 5., 5.])
	assert_array_almost_equal(d._xw_sum, [6., 9., 9.])

	d.summarize(X[4:], sample_weight=X[4:])
	assert_array_almost_equal(d._w_sum, [14., 8., 9.])
	assert_array_almost_equal(d._xw_sum, [44., 12., 25.])

	d = Exponential()
	d.summarize(X, sample_weight=X)
	assert_array_almost_equal(d._w_sum, [14., 8., 9.])
	assert_array_almost_equal(d._xw_sum, [44., 12., 25.])


def test_summarize_dtypes(X, w):
	X = numpy.array(X)
	X = X.astype(numpy.float32)

	p = numpy.array([3.0, 1.1, 2.8], dtype=numpy.float32)
	d = Exponential(p)
	assert d._xw_sum.dtype == torch.float32
	d.summarize(X)
	assert d._xw_sum.dtype == torch.float32

	X = X.astype(numpy.float64)
	p = numpy.array([3.0, 1.1, 2.8], dtype=numpy.float32)
	d = Exponential(p)
	assert d._xw_sum.dtype == torch.float32
	d.summarize(X)
	assert d._xw_sum.dtype == torch.float32

	X = X.astype(numpy.int32)
	p = numpy.array([3.0, 1.1, 2.8], dtype=numpy.float32)
	d = Exponential(p)
	assert d._xw_sum.dtype == torch.float32
	d.summarize(X)
	assert d._xw_sum.dtype == torch.float32

	X = X.astype(numpy.int64)
	p = numpy.array([3.0, 1.1, 2.8], dtype=numpy.float32)
	d = Exponential(p)
	assert d._xw_sum.dtype == torch.float32
	d.summarize(X)
	assert d._xw_sum.dtype == torch.float32


def test_summarize_raises(X, w, scales):
	_test_raises(Exponential(scales), "summarize", X, w=w, 
		min_value=MIN_VALUE, max_value=MAX_VALUE)

	_test_raises(Exponential(), "summarize", X, w=w, 
		min_value=MIN_VALUE, max_value=MAX_VALUE)

	_test_raises(Exponential([VALID_VALUE]), "summarize", X, w=w, 
		min_value=MIN_VALUE, max_value=MAX_VALUE)


def test_from_summaries(X, scales):
	d = Exponential(scales)
	d.summarize(X)
	d.from_summaries()
	assert_raises(AssertionError, assert_array_almost_equal, scales, d.scales)

	for param in scales, None:
		d = Exponential(param)
		d.summarize(X[:4])
		d.from_summaries()
		_test_efd_from_summaries(d, "scales", "_log_scales", [1.  , 1.25, 1.25])

		d.summarize(X[4:])
		d.from_summaries()
		_test_efd_from_summaries(d, "scales", "_log_scales", 
			[3.333333, 1.      , 1.333333])

		d = Exponential(param)
		d.summarize(X[:4])
		d.summarize(X[4:])
		d.from_summaries()
		_test_efd_from_summaries(d, "scales", "_log_scales", 
			[2.      , 1.142857, 1.285714])

		d = Exponential(param)
		d.summarize(X)
		d.from_summaries()
		_test_efd_from_summaries(d, "scales", "_log_scales", 
			[2.      , 1.142857, 1.285714])


def test_from_summaries_weighted(X, w, scales):
	for param in scales, None:
		d = Exponential(scales)
		d.summarize(X[:4], sample_weight=w[:4])
		d.from_summaries()
		_test_efd_from_summaries(d, "scales", "_log_scales", 
			[0.333333, 0.666667, 0.666667])

		d.summarize(X[4:], sample_weight=w[4:])
		d.from_summaries()
		_test_efd_from_summaries(d, "scales", "_log_scales", 
			[3. , 1. , 0.5])

		d = Exponential(scales)
		d.summarize(X, sample_weight=w)
		d.from_summaries()
		_test_efd_from_summaries(d, "scales", "_log_scales", 
			[2.272727, 0.909091, 0.545455])


def test_from_summaries_null():
	d = Exponential([1, 2])
	d.from_summaries()
	assert d.scales[0] != 1 and d.scales[1] != 2 
	assert_array_almost_equal(d._w_sum, [0.0, 0.0])
	assert_array_almost_equal(d._xw_sum, [0.0, 0.0])

	d = Exponential([1, 2], inertia=0.5)
	d.from_summaries()
	assert d.scales[0] != 1 and d.scales[1] != 2 
	assert_array_almost_equal(d._w_sum, [0.0, 0.0])
	assert_array_almost_equal(d._xw_sum, [0.0, 0.0])

	d = Exponential([1, 2], inertia=0.5, frozen=True)
	d.from_summaries()
	_test_efd_from_summaries(d, "scales", "_log_scales", [1, 2])


def test_from_summaries_inertia(X, w, scales):
	d = Exponential(scales, inertia=0.3)
	d.summarize(X[:4])
	d.from_summaries()
	_test_efd_from_summaries(d, "scales", "_log_scales", [1.06 , 1.415, 1.505])

	d.summarize(X[4:])
	d.from_summaries()
	_test_efd_from_summaries(d, "scales", "_log_scales", 
		[2.651333, 1.1245  , 1.384833])

	d = Exponential(scales, inertia=0.3)
	d.summarize(X)
	d.from_summaries()
	_test_efd_from_summaries(d, "scales", "_log_scales", 
		[1.76, 1.34, 1.53])


def test_from_summaries_weighted_inertia(X, w, scales):
	d = Exponential(scales, inertia=0.3)
	d.summarize(X, sample_weight=w)
	d.from_summaries()
	_test_efd_from_summaries(d, "scales", "_log_scales", 
		[1.950909, 1.176364, 1.011818])

	d = Exponential(scales, inertia=1.0)
	d.summarize(X[:4])
	d.from_summaries()
	_test_efd_from_summaries(d, "scales", "_log_scales", scales)

	d.summarize(X[4:])
	d.from_summaries()
	_test_efd_from_summaries(d, "scales", "_log_scales", scales)

	d = Exponential(scales, inertia=1.0)
	d.summarize(X)
	d.from_summaries()
	_test_efd_from_summaries(d, "scales", "_log_scales", scales)


def test_from_summaries_frozen(X, w, scales):
	d = Exponential(scales, frozen=True)
	d.summarize(X[:4])
	assert_array_almost_equal(d._w_sum, [0.0, 0.0, 0.0])
	assert_array_almost_equal(d._xw_sum, [0.0, 0.0, 0.0])

	d.from_summaries()
	_test_efd_from_summaries(d, "scales", "_log_scales", scales)

	d.summarize(X[4:])
	assert_array_almost_equal(d._w_sum, [0.0, 0.0, 0.0])
	assert_array_almost_equal(d._xw_sum, [0.0, 0.0, 0.0])

	d.from_summaries()
	_test_efd_from_summaries(d, "scales", "_log_scales", scales)

	d = Exponential(scales, frozen=True)
	d.summarize(X)
	assert_array_almost_equal(d._w_sum, [0.0, 0.0, 0.0])
	assert_array_almost_equal(d._xw_sum, [0.0, 0.0, 0.0])

	d.from_summaries()
	_test_efd_from_summaries(d, "scales", "_log_scales", scales)

	d = Exponential(scales, frozen=True)
	d.summarize(X, sample_weight=w)
	assert_array_almost_equal(d._w_sum, [0.0, 0.0, 0.0])
	assert_array_almost_equal(d._xw_sum, [0.0, 0.0, 0.0])

	d.from_summaries()
	_test_efd_from_summaries(d, "scales", "_log_scales", scales)


def test_from_summaries_dtypes(X, scales):
	X = numpy.array(X)
	X = X.astype(numpy.float32)

	p = numpy.array(scales, dtype=numpy.float32)
	d = Exponential(p)
	d.summarize(X)
	d.from_summaries()
	assert d.scales.dtype == torch.float32
	assert d._log_scales.dtype == torch.float32

	p = numpy.array(scales, dtype=numpy.float64)
	d = Exponential(p)
	d.summarize(X)
	d.from_summaries()
	assert d.scales.dtype == torch.float64
	assert d._log_scales.dtype == torch.float64

	p = numpy.array(scales, dtype=numpy.int32)
	d = Exponential(p)
	d.summarize(X)
	d.from_summaries()
	assert d.scales.dtype == torch.int32
	assert d._log_scales.dtype == torch.float32

	X = numpy.array(X)
	X = X.astype(numpy.float64)

	p = numpy.array(scales, dtype=numpy.float64)
	d = Exponential(p)
	d.summarize(X)
	d.from_summaries()
	assert d.scales.dtype == torch.float64
	assert d._log_scales.dtype == torch.float64


def test_from_summaries_raises():
	assert_raises(AttributeError, Exponential().from_summaries)


def test_fit(X, scales):
	for param in scales, None:
		d = Exponential(param)
		d.fit(X[:4])
		_test_efd_from_summaries(d, "scales", "_log_scales", 
			[1.  , 1.25, 1.25])

		d.fit(X[4:])
		_test_efd_from_summaries(d, "scales", "_log_scales", 
			[3.333333, 1.      , 1.333333])

		d = Exponential(param)
		d.fit(X)
		_test_efd_from_summaries(d, "scales", "_log_scales", 
			[2.      , 1.142857, 1.285714])


def test_fit_weighted(X, w, scales):
	for param in scales, None:
		d = Exponential(param)
		d.fit(X[:4], sample_weight=w[:4])
		_test_efd_from_summaries(d, "scales", "_log_scales", 
			[0.333333, 0.666667, 0.666667])

		d.fit(X[4:], sample_weight=w[4:])
		_test_efd_from_summaries(d, "scales", "_log_scales", 
			[3., 1. , 0.5])

		d = Exponential(param)
		d.fit(X, sample_weight=w)
		_test_efd_from_summaries(d, "scales", "_log_scales", 
			[2.272727, 0.909091, 0.545455])


	X = [[1.2, 0.5, 1.1, 1.9],
	     [6.2, 1.1, 2.4, 1.1]] 

	w = [[1.1], [3.5]]

	d = Exponential()
	d.fit(X, sample_weight=w)
	_test_efd_from_summaries(d, "scales", "_log_scales", 
		[5.004348, 0.956522, 2.089131, 1.291304])


def test_fit_chain(X):
	d = Exponential().fit(X[:4])
	_test_efd_from_summaries(d, "scales", "_log_scales", [1.  , 1.25, 1.25])

	d.fit(X[4:])
	_test_efd_from_summaries(d, "scales", "_log_scales", 
		[3.333333, 1.0, 1.3333333])

	d = Exponential().fit(X)
	_test_efd_from_summaries(d, "scales", "_log_scales", 
		[2.      , 1.142857, 1.285714])


def test_fit_dtypes(X, scales):
	X = numpy.array(X)
	X = X.astype(numpy.float32)

	p = numpy.array(scales, dtype=numpy.float32)
	d = Exponential(p).fit(X)
	assert d.scales.dtype == torch.float32
	assert d._log_scales.dtype == torch.float32

	p = numpy.array(scales, dtype=numpy.float64)
	d = Exponential(p).fit(X)
	assert d.scales.dtype == torch.float64
	assert d._log_scales.dtype == torch.float64

	p = numpy.array(scales, dtype=numpy.int32)
	d = Exponential(p).fit(X)
	assert d.scales.dtype == torch.int32
	assert d._log_scales.dtype == torch.float32

	X = numpy.array(X)
	X = X.astype(numpy.float64)

	p = numpy.array(scales, dtype=numpy.float64)
	d = Exponential(p).fit(X)
	assert d.scales.dtype == torch.float64
	assert d._log_scales.dtype == torch.float64


def test_fit_raises(X, w, scales):
	_test_raises(Exponential(scales), "fit", X, w=w, min_value=MIN_VALUE, 
		max_value=MAX_VALUE)

	_test_raises(Exponential(), "fit", X, w=w, min_value=MIN_VALUE, 
		max_value=MAX_VALUE)

	_test_raises(Exponential([VALID_VALUE]), "fit", X, w=w, 
		min_value=MIN_VALUE, max_value=MAX_VALUE)


def test_serialization(X):
	d = Exponential().fit(X[:4])
	d.summarize(X[4:])

	scales = [1.  , 1.25, 1.25]

	assert_array_almost_equal(d.scales, scales)
	assert_array_almost_equal(d._log_scales, numpy.log(scales))

	torch.save(d, ".pytest.torch")
	d2 = torch.load(".pytest.torch")
	os.system("rm .pytest.torch")

	assert_array_almost_equal(d2.scales, scales)
	assert_array_almost_equal(d2._log_scales, numpy.log(scales))

	assert_array_almost_equal(d2._w_sum, [3., 3., 3.])
	assert_array_almost_equal(d2._xw_sum, [10., 3., 4.])
	assert_array_almost_equal(d.log_probability(X), d2.log_probability(X))


def test_masked_probability(scales, X, X_masked):
	X = torch.tensor(numpy.array(X))
	y = [0.03154 , 0.136937, 0.021209, 0.005289, 0.010383, 0.000292,
           0.023891]

	d = Exponential(scales)
	mask = torch.ones_like(X).type(torch.bool)
	X_ = torch.masked.MaskedTensor(X, mask=mask)

	assert_array_almost_equal(y, d.probability(X_)._masked_data)

	y =  [8.708810e-02, 4.629630e-01, 1.000000e+00, 5.288587e-03,
           3.187519e-01, 2.919201e-04, 7.495064e-02]

	assert_array_almost_equal(y, d.probability(X_masked)._masked_data)


def test_masked_log_probability(scales, X, X_masked):
	X = torch.tensor(numpy.array(X))
	y = [-3.45649 , -1.988236, -3.853315, -5.242204, -4.567601, -8.13903,
           -3.734268]

	d = Exponential(scales)
	mask = torch.ones_like(X).type(torch.bool)
	X_ = torch.masked.MaskedTensor(X, mask=mask)

	assert_array_almost_equal(y, d.log_probability(X_)._masked_data)

	y = [-2.440835, -0.770108,  0.      , -5.242204, -1.143342, -8.13903 ,
           -2.590925]

	assert_array_almost_equal(y, d.log_probability(X_masked)._masked_data)


def test_masked_summarize(X, X_masked, w):
	X = torch.tensor(numpy.array(X))
	mask = torch.ones_like(X).type(torch.bool)
	X_ = torch.masked.MaskedTensor(X, mask=mask)

	d = Exponential()
	d.summarize(X, sample_weight=w)
	assert_array_almost_equal(d._w_sum, [11.0, 11.0, 11.0])
	assert_array_almost_equal(d._xw_sum, [25.0, 10.0, 6.0])

	d = Exponential()
	d.summarize(X_masked)
	assert_array_almost_equal(d._w_sum, [4.0, 5.0, 4.0])
	assert_array_almost_equal(d._xw_sum, [9.0, 6.0, 6.0])


def test_masked_from_summaries(X, X_masked):
	X = torch.tensor(numpy.array(X))
	mask = torch.ones_like(X).type(torch.bool)
	X_ = torch.masked.MaskedTensor(X, mask=mask)

	d = Exponential()
	d.summarize(X_)
	d.from_summaries()
	_test_efd_from_summaries(d, "scales", "_log_scales", 
			[2.      , 1.142857, 1.285714])

	d = Exponential()
	d.summarize(X_masked)
	d.from_summaries()
	_test_efd_from_summaries(d, "scales", "_log_scales", 
			[2.25, 1.2 , 1.5])


def test_masked_fit(X, X_masked):
	X = torch.tensor(numpy.array(X))
	mask = torch.ones_like(X).type(torch.bool)
	X_ = torch.masked.MaskedTensor(X, mask=mask)

	d = Exponential()
	d.fit(X_)
	_test_efd_from_summaries(d, "scales", "_log_scales", 
			[2.      , 1.142857, 1.285714])

	d = Exponential()
	d.fit(X_masked)
	_test_efd_from_summaries(d, "scales", "_log_scales", 
			[2.25, 1.2 , 1.5])
	
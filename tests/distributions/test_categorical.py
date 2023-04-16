# test_categorical.py
# Contact: Jacob Schreiber <jmschreiber91@gmail.com>

import os
import numpy
import torch
import pytest

from torchegranate.distributions import Categorical

from ._utils import _test_initialization_raises_one_parameter
from ._utils import _test_predictions
from ._utils import _test_raises

from nose.tools import assert_raises
from numpy.testing import assert_array_almost_equal


MIN_VALUE = 0
MAX_VALUE = 3
VALID_VALUE = 1


@pytest.fixture
def X():
	return [[1, 2, 0],
	     [3, 0, 1],
	     [1, 1, 2],
	     [2, 2, 2],
	     [0, 1, 0],
	     [1, 1, 2],
	     [2, 1, 0]]


@pytest.fixture
def w():
	return [[1], [2], [0], [0], [5], [1], [2]]


@pytest.fixture
def probs():
	return [
		[0.1, 0.2, 0.2, 0.5],
		[0.3, 0.1, 0.3, 0.3],
		[0.7, 0.05, 0.05, 0.2]
	]


###


def _test_initialization(d, x, name, inertia, frozen, dtype):
	assert d.inertia == inertia
	assert d.frozen == frozen
	param = getattr(d, name)

	if x is not None:
		assert param.shape == (len(x), len(x[0]))
		assert param.dtype == dtype
		assert_array_almost_equal(param, x)
	else:
		assert param == x


def test_initialization():
	d = Categorical()
	_test_initialization(d, None, "probs", 0.0, False, None)
	assert_raises(AttributeError, getattr, d, "_w_sum")
	assert_raises(AttributeError, getattr, d, "_xw_sum")
	assert_raises(AttributeError, getattr, d, "_log_probs")


def test_initialization_float(probs):
	funcs = (lambda x: x, tuple, 
		lambda x: numpy.array(x, dtype=numpy.float32), 
		lambda x: torch.tensor(x, dtype=torch.float32), 
		lambda x: torch.nn.Parameter(torch.tensor(x), requires_grad=False))

	for func in funcs:
		y = func(probs)
		_test_initialization(Categorical(y, inertia=0.0, frozen=False), 
			y, "probs", 0.0, False, torch.float32)
		_test_initialization(Categorical(y, inertia=0.3, frozen=False), 
			y, "probs", 0.3, False, torch.float32)
		_test_initialization(Categorical(y, inertia=1.0, frozen=True), 
			y, "probs", 1.0, True, torch.float32)
		_test_initialization(Categorical(y, inertia=1.0, frozen=False), 
			y, "probs", 1.0, False, torch.float32)

	x = numpy.array(probs, dtype=numpy.float64)
	_test_initialization(Categorical(x, inertia=0.0, frozen=False), 
		x, "probs", 0.0, False, torch.float64)
	_test_initialization(Categorical(x, inertia=0.3, frozen=False), 
		x, "probs", 0.3, False, torch.float64)
	_test_initialization(Categorical(x, inertia=1.0, frozen=True), 
		x, "probs", 1.0, True, torch.float64)
	_test_initialization(Categorical(x, inertia=1.0, frozen=False), 
		x, "probs", 1.0, False, torch.float64)


def test_initialization_raises():
	_test_initialization_raises_one_parameter(Categorical, VALID_VALUE, 
		min_value=MIN_VALUE)


def test_reset_cache(X):
	d = Categorical()
	d.summarize(X)
	assert_array_almost_equal(d._w_sum, [7.0, 7.0, 7.0])
	assert_array_almost_equal(d._xw_sum,
		[[1., 3., 2., 1.],
         [1., 4., 2., 0.],
         [3., 1., 3., 0.]])

	d._reset_cache()
	assert_array_almost_equal(d._w_sum, [0.0, 0.0, 0.0])
	assert_array_almost_equal(d._xw_sum, 
		[[0., 0., 0., 0.],
         [0., 0., 0., 0.],
         [0., 0., 0., 0.]])	

	d = Categorical()
	assert_raises(AttributeError, getattr, d, "_w_sum")
	assert_raises(AttributeError, getattr, d, "_xw_sum")
	assert_raises(AttributeError, getattr, d, "_log_probs")

	d._reset_cache()
	assert_raises(AttributeError, getattr, d, "_w_sum")
	assert_raises(AttributeError, getattr, d, "_xw_sum")
	assert_raises(AttributeError, getattr, d, "_log_probs")


def test_initialize(X, probs):
	d = Categorical()
	assert d.d is None
	assert d.probs is None
	assert d._initialized == False
	assert_raises(AttributeError, getattr, d, "_w_sum")
	assert_raises(AttributeError, getattr, d, "_xw_sum")
	assert_raises(AttributeError, getattr, d, "_log_probs")

	d._initialize(3, 5)
	assert d._initialized == True
	assert d.probs.shape == (3, 5)
	assert d.d == 3
	assert d.n_keys == 5
	assert_array_almost_equal(d.probs, 
		[[0.0, 0.0, 0.0, 0.0, 0.0],
		 [0.0, 0.0, 0.0, 0.0, 0.0],
		 [0.0, 0.0, 0.0, 0.0, 0.0]])
	assert_array_almost_equal(d._w_sum, [0.0, 0.0, 0.0])
	assert_array_almost_equal(d._xw_sum,
		[[0.0, 0.0, 0.0, 0.0, 0.0],
		 [0.0, 0.0, 0.0, 0.0, 0.0],
		 [0.0, 0.0, 0.0, 0.0, 0.0]])	

	d._initialize(2, 2)
	assert d._initialized == True
	assert d.probs.shape == (2, 2)
	assert d.d == 2
	assert d.n_keys == 2
	assert_array_almost_equal(d.probs,
		[[0.0, 0.0],
		 [0.0, 0.0]])
	assert_array_almost_equal(d._w_sum, [0.0, 0.0])
	assert_array_almost_equal(d._xw_sum,
		[[0.0, 0.0],
		 [0.0, 0.0]])	

	d = Categorical(probs)
	assert d._initialized == True
	assert d.d == 3
	assert d.n_keys == 4

	d._initialize(3, 4)
	assert d._initialized == True
	assert d.probs.shape == (3, 4)
	assert d.d == 3
	assert d.n_keys == 4
	assert_array_almost_equal(d.probs,
		[[0.0, 0.0, 0.0, 0.0],
		 [0.0, 0.0, 0.0, 0.0],
		 [0.0, 0.0, 0.0, 0.0]])

	d = Categorical()
	d.summarize(X)
	d._initialize(4, 2)
	assert d._initialized == True
	assert d.probs.shape == (4, 2)
	assert d.d == 4
	assert d.n_keys == 2
	assert_array_almost_equal(d._w_sum, [0.0, 0.0, 0.0, 0.0])
	assert_array_almost_equal(d._xw_sum,
		[[0.0, 0.0],
		 [0.0, 0.0],
		 [0.0, 0.0],
		 [0.0, 0.0]])	


###


@pytest.mark.sample
def test_sample(probs):
	torch.manual_seed(0)

	X = Categorical(probs).sample(1)
	assert_array_almost_equal(X, [[3, 3, 0]])

	X = Categorical(probs).sample(5)
	assert_array_almost_equal(X, 
		[[3, 2, 0],
         [3, 0, 0],
         [3, 2, 0],
         [1, 0, 0],
         [2, 1, 0]])


###


def test_probability(X, probs):
	p = [[0.2, 0.8]]
	x = [[1], [0], [0], [1], [0]]
	y = [0.8, 0.2, 0.2, 0.8, 0.2]

	d1 = Categorical(p)
	d2 = Categorical(numpy.array(p, dtype=numpy.float64))

	_test_predictions(x, y, d1.probability(x), torch.float32)
	_test_predictions(x, y, d2.probability(x), torch.float64)

	x = [[1, 2, 0]]
	y = [0.2*0.3*0.7]

	d1 = Categorical(probs)
	d2 = Categorical(numpy.array(probs, dtype=numpy.float64))

	_test_predictions(x, y, d1.probability(x), torch.float32)
	_test_predictions(x, y, d2.probability(x), torch.float64)

	y = [0.042 , 0.0075, 0.001 , 0.003 , 0.007 , 0.001 , 0.014 ]

	d1 = Categorical(probs)
	d2 = Categorical(numpy.array(probs, dtype=numpy.float64))

	_test_predictions(X, y, d1.probability(X), torch.float32)
	_test_predictions(X, y, d2.probability(X), torch.float64)


def test_probability_dtypes(X, probs):
	y = Categorical(probs).probability(X)
	assert y.dtype == torch.float32

	probs = numpy.array(probs, dtype=numpy.float64)

	y = Categorical(probs).probability(X)
	assert y.dtype == torch.float64


def test_probability_raises(X, probs):
	_test_raises(Categorical(probs), "probability", X, min_value=MIN_VALUE, 
		max_value=MAX_VALUE)


def test_log_probability(X, probs):
	p = [[0.2, 0.8]]
	x = [[1], [0], [0], [1], [0]]
	y = numpy.log([0.8, 0.2, 0.2, 0.8, 0.2])

	x_torch = torch.tensor(numpy.array(x))	
	p_torch = torch.tensor(numpy.array(p))

	d1 = Categorical(p)
	d2 = Categorical(numpy.array(p, dtype=numpy.float64))
	d3 = torch.distributions.Categorical(p_torch)

	_test_predictions(x, y, d1.log_probability(x), torch.float32)
	_test_predictions(x, y, d2.log_probability(x), torch.float64)
	_test_predictions(x, d3.log_prob(x_torch).sum(axis=1), 
		d2.log_probability(x), torch.float64)

	x = [[1, 2, 0]]
	y = [-3.170086]

	p_torch = torch.tensor(numpy.array(probs))
	x_torch = torch.tensor(numpy.array(x))

	d1 = Categorical(probs)
	d2 = Categorical(numpy.array(probs, dtype=numpy.float64))
	d3 = torch.distributions.Categorical(p_torch)

	_test_predictions(x, y, d1.log_probability(x), torch.float32)
	_test_predictions(x, y, d2.log_probability(x), torch.float64)
	_test_predictions(x, d3.log_prob(x_torch).sum(axis=1), 
		d2.log_probability(x), torch.float64)

	p_torch = torch.tensor(numpy.array(probs))
	x_torch = torch.tensor(numpy.array(X))

	y = [-3.170086, -4.892852, -6.907755, -5.809143, -4.961845, -6.907755,
           -4.268698]

	d1 = Categorical(probs)
	d2 = Categorical(numpy.array(probs, dtype=numpy.float64))
	d3 = torch.distributions.Categorical(p_torch)

	_test_predictions(X, y, d1.log_probability(X), torch.float32)
	_test_predictions(X, y, d2.log_probability(X), torch.float64)
	_test_predictions(X, d3.log_prob(x_torch).sum(axis=1), 
		d2.log_probability(X), torch.float64)


def test_log_probability_dtypes(X, probs):
	y = Categorical(probs).log_probability(X)
	assert y.dtype == torch.float32

	probs = numpy.array(probs, dtype=numpy.float64)

	y = Categorical(probs).log_probability(X)
	assert y.dtype == torch.float64


def test_log_probability_raises(X, probs):
	_test_raises(Categorical(probs), "log_probability", X, min_value=MIN_VALUE, 
		max_value=MAX_VALUE)


###


def test_summarize(X, probs):
	for param in probs, None:
		d = Categorical(param)
		d.summarize(X[:4])
		assert_array_almost_equal(d._w_sum, [4.0, 4.0, 4.0])
		assert_array_almost_equal(d._xw_sum, 
			[[0., 2., 1., 1.],
	         [1., 1., 2., 0.],
	         [1., 1., 2., 0.]])

		d.summarize(X[4:])
		assert_array_almost_equal(d._w_sum, [7.0, 7.0, 7.0])
		assert_array_almost_equal(d._xw_sum, 
			[[1., 3., 2., 1.],
             [1., 4., 2., 0.],
             [3., 1., 3., 0.]])

		d = Categorical(param)
		d.summarize(X)
		assert_array_almost_equal(d._w_sum, [7.0, 7.0, 7.0])
		assert_array_almost_equal(d._xw_sum,
			[[1., 3., 2., 1.],
             [1., 4., 2., 0.],
             [3., 1., 3., 0.]])


def test_summarize_weighted(X, w, probs):
	for param in probs, None:
		d = Categorical(param)
		d.summarize(X[:4], sample_weight=w[:4])
		assert_array_almost_equal(d._w_sum, [3., 3., 3.])
		assert_array_almost_equal(d._xw_sum,
			[[0., 1., 0., 2.],
             [2., 0., 1., 0.],
             [1., 2., 0., 0.]])

		d.summarize(X[4:], sample_weight=w[4:])
		assert_array_almost_equal(d._w_sum, [11.0, 11.0, 11.0])
		assert_array_almost_equal(d._xw_sum,
			[[5., 2., 2., 2.],
             [2., 8., 1., 0.],
             [8., 2., 1., 0.]],)

		d = Categorical(param)
		d.summarize(X, sample_weight=w)
		assert_array_almost_equal(d._w_sum, [11.0, 11.0, 11.0])
		assert_array_almost_equal(d._xw_sum,
			[[5., 2., 2., 2.],
             [2., 8., 1., 0.],
             [8., 2., 1., 0.]])


def test_summarize_weighted_flat(X, w, probs):
	w = numpy.array(w)[:,0] 

	for param in probs, None:
		d = Categorical(param)
		d.summarize(X[:4], sample_weight=w[:4])
		assert_array_almost_equal(d._w_sum, [3., 3., 3.])
		assert_array_almost_equal(d._xw_sum,
			[[0., 1., 0., 2.],
             [2., 0., 1., 0.],
             [1., 2., 0., 0.]])

		d.summarize(X[4:], sample_weight=w[4:])
		assert_array_almost_equal(d._w_sum, [11.0, 11.0, 11.0])
		assert_array_almost_equal(d._xw_sum,
			[[5., 2., 2., 2.],
             [2., 8., 1., 0.],
             [8., 2., 1., 0.]],)

		d = Categorical(param)
		d.summarize(X, sample_weight=w)
		assert_array_almost_equal(d._w_sum, [11.0, 11.0, 11.0])
		assert_array_almost_equal(d._xw_sum,
			[[5., 2., 2., 2.],
             [2., 8., 1., 0.],
             [8., 2., 1., 0.]])


def test_summarize_weighted_2d(X):
	d = Categorical()
	d.summarize(X[:4], sample_weight=X[:4])
	assert_array_almost_equal(d._w_sum, [7., 5., 5.])
	assert_array_almost_equal(d._xw_sum,
		[[0., 2., 2., 3.],
         [0., 1., 4., 0.],
         [0., 1., 4., 0.]])

	d.summarize(X[4:], sample_weight=X[4:])
	assert_array_almost_equal(d._w_sum, [10., 8., 7.])
	assert_array_almost_equal(d._xw_sum,
		[[0., 3., 4., 3.],
         [0., 4., 4., 0.],
         [0., 1., 6., 0.]])

	d = Categorical()
	d.summarize(X, sample_weight=X)
	assert_array_almost_equal(d._w_sum, [10., 8., 7.])
	assert_array_almost_equal(d._xw_sum,
		[[0., 3., 4., 3.],
         [0., 4., 4., 0.],
         [0., 1., 6., 0.]])


def test_summarize_dtypes(X, probs):
	X = numpy.array(X, dtype=numpy.float32)
	p = numpy.array(probs, dtype=numpy.float32)

	d = Categorical(p)
	assert d._xw_sum.dtype == torch.float32
	d.summarize(X)
	assert d._xw_sum.dtype == torch.float32

	X = numpy.array(X, dtype=numpy.float64)
	p = numpy.array(probs, dtype=numpy.float64)

	d = Categorical(p)
	assert d._xw_sum.dtype == torch.float32
	d.summarize(X)
	assert d._xw_sum.dtype == torch.float32


def test_summarize_raises(X, w, probs):
	_test_raises(Categorical(probs), "summarize", X, w=w, 
		min_value=MIN_VALUE, max_value=MAX_VALUE)

	_test_raises(Categorical(), "summarize", X, w=w, 
		min_value=MIN_VALUE, max_value=MAX_VALUE)


def _test_fit_params(d, values):
	assert_array_almost_equal(d.probs, values)
	assert_array_almost_equal(d._log_probs, numpy.log(values), 4)
	assert_array_almost_equal(d._w_sum, numpy.zeros(d.d))
	assert_array_almost_equal(d._xw_sum, numpy.zeros((d.d, d.n_keys)))


def test_from_summaries(X, probs):
	d = Categorical(probs)
	d.summarize(X)
	d.from_summaries()
	assert_raises(AssertionError, assert_array_almost_equal, probs, d.probs)

	for param in probs, None:
		d = Categorical(param)
		d.summarize(X[:4])
		d.from_summaries()
		_test_fit_params(d,
			[[0.  , 0.5 , 0.25, 0.25],
             [0.25, 0.25, 0.5 , 0.  ],
             [0.25, 0.25, 0.5 , 0.  ]])

		d.summarize(X[4:])
		d.from_summaries()
		_test_fit_params(d,
			[[0.333333, 0.333333, 0.333333, 0.      ],
             [0.      , 1.      , 0.      , 0.      ],
             [0.666667, 0.      , 0.333333, 0.      ]])

		d = Categorical(param)
		d.summarize(X[:4])
		d.summarize(X[4:])
		d.from_summaries()
		_test_fit_params(d, 
			[[0.142857, 0.428571, 0.285714, 0.142857],
             [0.142857, 0.571429, 0.285714, 0.      ],
             [0.428571, 0.142857, 0.428571, 0.      ]])

		d = Categorical(param)
		d.summarize(X)
		d.from_summaries()
		_test_fit_params(d, 
			[[0.142857, 0.428571, 0.285714, 0.142857],
             [0.142857, 0.571429, 0.285714, 0.      ],
             [0.428571, 0.142857, 0.428571, 0.      ]])


def test_from_summaries_weighted(X, w, probs):
	for param in probs, None:
		d = Categorical(probs)
		d.summarize(X[:4], sample_weight=w[:4])
		d.from_summaries()
		_test_fit_params(d, 
			[[0.      , 0.333333, 0.      , 0.666667],
             [0.666667, 0.      , 0.333333, 0.      ],
             [0.333333, 0.666667, 0.      , 0.      ]])

		d.summarize(X[4:], sample_weight=w[4:])
		d.from_summaries()
		_test_fit_params(d, 
			[[0.625, 0.125, 0.25 , 0.   ],
             [0.   , 1.   , 0.   , 0.   ],
             [0.875, 0.   , 0.125, 0.   ]])

		d = Categorical(probs)
		d.summarize(X, sample_weight=w)
		d.from_summaries()
		_test_fit_params(d, 
			[[0.454545, 0.181818, 0.181818, 0.181818],
             [0.181818, 0.727273, 0.090909, 0.      ],
             [0.727273, 0.181818, 0.090909, 0.      ]])


def test_from_summaries_null(X, probs):
	d = Categorical(probs)
	d.from_summaries()
	assert d.probs[1, 0] != probs[1][0] and d.probs[0][1] != probs[0][1] 
	assert_array_almost_equal(d._w_sum, [0.0, 0.0, 0.0])
	assert_array_almost_equal(d._xw_sum, 
		[[0.0, 0.0, 0.0, 0.0],
		 [0.0, 0.0, 0.0, 0.0],
		 [0.0, 0.0, 0.0, 0.0]])

	d = Categorical(probs, inertia=0.5)
	d.from_summaries()
	assert d.probs[1, 0] != probs[1][0] and d.probs[0][1] != probs[0][1] 
	assert_array_almost_equal(d._w_sum, [0.0, 0.0, 0.0])
	assert_array_almost_equal(d._xw_sum, 
		[[0.0, 0.0, 0.0, 0.0],
		 [0.0, 0.0, 0.0, 0.0],
		 [0.0, 0.0, 0.0, 0.0]])

	d = Categorical(probs, inertia=0.5, frozen=True)
	d.from_summaries()
	_test_fit_params(d,
		[[0.1 , 0.2 , 0.2 , 0.5 ],
         [0.3 , 0.1 , 0.3 , 0.3 ],
         [0.7 , 0.05, 0.05, 0.2 ]])


def test_from_summaries_inertia(X, w, probs):
	d = Categorical(probs, inertia=0.3)
	d.summarize(X[:4])
	d.from_summaries()
	_test_fit_params(d,
		[[0.03 , 0.41 , 0.235, 0.325],
         [0.265, 0.205, 0.44 , 0.09 ],
         [0.385, 0.19 , 0.365, 0.06 ]])

	d.summarize(X[4:])
	d.from_summaries()
	_test_fit_params(d, 
		[[0.242333, 0.356333, 0.303833, 0.0975  ],
         [0.0795  , 0.7615  , 0.132   , 0.027   ],
         [0.582167, 0.057   , 0.342833, 0.018   ]])

	d = Categorical(probs, inertia=0.3)
	d.summarize(X)
	d.from_summaries()
	_test_fit_params(d, 
		[[0.13 , 0.36 , 0.26 , 0.25 ],
         [0.19 , 0.43 , 0.29 , 0.09 ],
         [0.51 , 0.115, 0.315, 0.06 ]])


def test_from_summaries_weighted_inertia(X, w, probs):
	d = Categorical(probs, inertia=0.5)
	d.summarize(X, sample_weight=w)
	d.from_summaries()
	_test_fit_params(d, 
		[[0.277273, 0.190909, 0.190909, 0.340909],
         [0.240909, 0.413636, 0.195455, 0.15    ],
         [0.713636, 0.115909, 0.070455, 0.1     ]])

	d = Categorical(probs, inertia=1.0)
	d.summarize(X[:4])
	d.from_summaries()
	_test_fit_params(d, probs)

	d.summarize(X[4:])
	d.from_summaries()
	_test_fit_params(d, probs)

	d = Categorical(probs, inertia=1.0)
	d.summarize(X)
	d.from_summaries()
	_test_fit_params(d, probs)


def test_from_summaries_frozen(X, w, probs):
	d = Categorical(probs, frozen=True)
	d.summarize(X[:4])
	assert_array_almost_equal(d._w_sum, [0.0, 0.0, 0.0])
	assert_array_almost_equal(d._xw_sum,
		[[0., 0., 0., 0.],
         [0., 0., 0., 0.],
         [0., 0., 0., 0.]])

	d.from_summaries()
	_test_fit_params(d, probs)

	d.summarize(X[4:])
	assert_array_almost_equal(d._w_sum, [0.0, 0.0, 0.0])
	assert_array_almost_equal(d._xw_sum,
		[[0., 0., 0., 0.],
         [0., 0., 0., 0.],
         [0., 0., 0., 0.]])

	d.from_summaries()
	_test_fit_params(d, probs)

	d = Categorical(probs, frozen=True)
	d.summarize(X)
	assert_array_almost_equal(d._w_sum, [0.0, 0.0, 0.0])
	assert_array_almost_equal(d._xw_sum,
		[[0., 0., 0., 0.],
         [0., 0., 0., 0.],
         [0., 0., 0., 0.]])

	d.from_summaries()
	_test_fit_params(d, probs)

	d = Categorical(probs, frozen=True)
	d.summarize(X, sample_weight=w)
	assert_array_almost_equal(d._w_sum, [0.0, 0.0, 0.0])
	assert_array_almost_equal(d._xw_sum,
		[[0., 0., 0., 0.],
         [0., 0., 0., 0.],
         [0., 0., 0., 0.]])

	d.from_summaries()
	_test_fit_params(d, probs)


def test_from_summaries_dtypes(X, probs):
	X = numpy.array(X)
	X = X.astype(numpy.float32)

	p = numpy.array(probs, dtype=numpy.float32)
	d = Categorical(p)
	d.summarize(X)
	d.from_summaries()
	assert d.probs.dtype == torch.float32
	assert d._log_probs.dtype == torch.float32

	p = numpy.array(probs, dtype=numpy.float64)
	d = Categorical(p)
	d.summarize(X)
	d.from_summaries()
	assert d.probs.dtype == torch.float64
	assert d._log_probs.dtype == torch.float64


def test_from_summaries_raises():
	assert_raises(AttributeError, Categorical().from_summaries)


def test_fit(X, probs):
	d = Categorical(probs)
	d.fit(X)
	assert_raises(AssertionError, assert_array_almost_equal, probs, d.probs)

	for param in probs, None:
		d = Categorical(param)
		d.fit(X[:4])
		_test_fit_params(d,
			[[0.  , 0.5 , 0.25, 0.25],
             [0.25, 0.25, 0.5 , 0.  ],
             [0.25, 0.25, 0.5 , 0.  ]])

		d.fit(X[4:])
		_test_fit_params(d,
			[[0.333333, 0.333333, 0.333333, 0.      ],
             [0.      , 1.      , 0.      , 0.      ],
             [0.666667, 0.      , 0.333333, 0.      ]])

		d = Categorical(param)
		d.fit(X)
		_test_fit_params(d, 
			[[0.142857, 0.428571, 0.285714, 0.142857],
             [0.142857, 0.571429, 0.285714, 0.      ],
             [0.428571, 0.142857, 0.428571, 0.      ]])


def test_fit_weighted(X, w, probs):
	for param in probs, None:
		d = Categorical(probs)
		d.fit(X[:4], sample_weight=w[:4])
		_test_fit_params(d, 
			[[0.      , 0.333333, 0.      , 0.666667],
             [0.666667, 0.      , 0.333333, 0.      ],
             [0.333333, 0.666667, 0.      , 0.      ]])

		d.fit(X[4:], sample_weight=w[4:])
		_test_fit_params(d, 
			[[0.625, 0.125, 0.25 , 0.   ],
             [0.   , 1.   , 0.   , 0.   ],
             [0.875, 0.   , 0.125, 0.   ]])

		d = Categorical(probs)
		d.fit(X, sample_weight=w)
		_test_fit_params(d, 
			[[0.454545, 0.181818, 0.181818, 0.181818],
             [0.181818, 0.727273, 0.090909, 0.      ],
             [0.727273, 0.181818, 0.090909, 0.      ]])


def test_fit_chain(X):
	d = Categorical().fit(X[:4])
	_test_fit_params(d,
		[[0.  , 0.5 , 0.25, 0.25],
         [0.25, 0.25, 0.5 , 0.  ],
         [0.25, 0.25, 0.5 , 0.  ]])

	d.fit(X[4:])
	_test_fit_params(d, 
		[[0.333333, 0.333333, 0.333333, 0.      ],
         [0.      , 1.      , 0.      , 0.      ],
         [0.666667, 0.      , 0.333333, 0.      ]])

	d = Categorical().fit(X)
	_test_fit_params(d, 
		[[0.142857, 0.428571, 0.285714, 0.142857],
         [0.142857, 0.571429, 0.285714, 0.      ],
         [0.428571, 0.142857, 0.428571, 0.      ]])


def test_fit_dtypes(X, probs):
	X = numpy.array(X)
	X = X.astype(numpy.float32)

	p = numpy.array(probs, dtype=numpy.float32)
	d = Categorical(p).fit(X)
	assert d.probs.dtype == torch.float32
	assert d._log_probs.dtype == torch.float32

	p = numpy.array(probs, dtype=numpy.float64)
	d = Categorical(p).fit(X)
	assert d.probs.dtype == torch.float64
	assert d._log_probs.dtype == torch.float64


def test_fit_raises(X, w, probs):
	_test_raises(Categorical(probs), "fit", X, w=w, min_value=MIN_VALUE, 
		max_value=MAX_VALUE)

	_test_raises(Categorical(), "fit", X, w=w, min_value=MIN_VALUE, 
		max_value=MAX_VALUE)


def test_serialization(X):
	d = Categorical().fit(X[:4])
	d.summarize(X[4:])

	p = [[0.  , 0.5 , 0.25, 0.25],
         [0.25, 0.25, 0.5 , 0.  ],
         [0.25, 0.25, 0.5 , 0.  ]]

	assert_array_almost_equal(d.probs, p)
	assert_array_almost_equal(d._log_probs, numpy.log(p))

	torch.save(d, ".pytest.torch")
	d2 = torch.load(".pytest.torch")
	os.system("rm .pytest.torch")

	assert_array_almost_equal(d2.probs, p)
	assert_array_almost_equal(d2._log_probs, numpy.log(p))

	assert_array_almost_equal(d2._w_sum, [3., 3., 3.])
	assert_array_almost_equal(d2._xw_sum, 
		[[1., 1., 1., 0.],
         [0., 3., 0., 0.],
         [2., 0., 1., 0.]])
	assert_array_almost_equal(d.log_probability(X), d2.log_probability(X))

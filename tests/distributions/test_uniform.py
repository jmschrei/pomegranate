# test_uniform.py
# Contact: Jacob Schreiber <jmschreiber91@gmail.com>

import os
import numpy
import torch
import pytest

from pomegranate.distributions import Uniform

from ._utils import _test_initialization_raises_two_parameters
from ._utils import _test_initialization
from ._utils import _test_predictions
from ._utils import _test_efd_from_summaries
from ._utils import _test_raises

from nose.tools import assert_raises
from numpy.testing import assert_array_almost_equal


MIN_VALUE = None
MAX_VALUE = None
VALID_VALUE = 1.2

INF = float("inf")
NEGINF = float("-inf")


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
def w():
	return [[1], [2], [0], [0], [5], [1], [2]]


@pytest.fixture
def mins():
	return [0.3, 0.2, 1.0]


@pytest.fixture
def maxs():
	return [2.7, 3.1, 2.5]


###


def test_initialization():
	d = Uniform()
	
	_test_initialization(d, None, "mins", 0.0, False, None)
	_test_initialization(d, None, "maxs", 0.0, False, None)

	assert_raises(AttributeError, getattr, d, "_x_mins")
	assert_raises(AttributeError, getattr, d, "_x_maxs")
	assert_raises(AttributeError, getattr, d, "_logps")


def test_initialization_int():
	funcs = (lambda x: x, tuple, numpy.array, torch.tensor, 
		lambda x: torch.nn.Parameter(torch.tensor(x), requires_grad=False))

	mins = [1, 2, 3, 8, 1]
	maxs = [2, 3, 4, 9, 2]
	for func in funcs:
		mins_ = func(mins)
		maxs_ = func(maxs)

		_test_initialization(Uniform(mins_, maxs_, inertia=0.0, frozen=False), 
			mins_, "mins", 0.0, False, torch.int64)
		_test_initialization(Uniform(mins_, maxs_, inertia=0.3, frozen=False), 
			mins_, "mins", 0.3, False, torch.int64)
		_test_initialization(Uniform(mins_, maxs_, inertia=1.0, frozen=True), 
			mins_, "mins", 1.0, True, torch.int64)
		_test_initialization(Uniform(mins_, maxs_, inertia=1.0, frozen=False), 
			mins_, "mins", 1.0, False, torch.int64)

		_test_initialization(Uniform(mins_, maxs_, inertia=0.0, frozen=False), 
			maxs_, "maxs", 0.0, False, torch.int64)
		_test_initialization(Uniform(mins_, maxs_, inertia=0.3, frozen=False), 
			maxs_, "maxs", 0.3, False, torch.int64)
		_test_initialization(Uniform(mins_, maxs_, inertia=1.0, frozen=True), 
			maxs_, "maxs", 1.0, True, torch.int64)
		_test_initialization(Uniform(mins_, maxs_, inertia=1.0, frozen=False), 
			maxs_, "maxs", 1.0, False, torch.int64)

	mins = numpy.array(mins, dtype=numpy.int32)
	maxs = numpy.array(maxs, dtype=numpy.int32)
	for func in funcs[2:]:
		mins_ = func(mins)
		maxs_ = func(maxs)

		_test_initialization(Uniform(mins_, maxs_, inertia=0.0, frozen=False), 
			mins_, "mins", 0.0, False, torch.int32)
		_test_initialization(Uniform(mins_, maxs_, inertia=0.3, frozen=False), 
			mins_, "mins", 0.3, False, torch.int32)
		_test_initialization(Uniform(mins_, maxs_, inertia=1.0, frozen=True), 
			mins_, "mins", 1.0, True, torch.int32)
		_test_initialization(Uniform(mins_, maxs_, inertia=1.0, frozen=False), 
			mins_, "mins", 1.0, False, torch.int32)

		_test_initialization(Uniform(mins_, maxs_, inertia=0.0, frozen=False), 
			maxs_, "maxs", 0.0, False, torch.int32)
		_test_initialization(Uniform(mins_, maxs_, inertia=0.3, frozen=False), 
			maxs_, "maxs", 0.3, False, torch.int32)
		_test_initialization(Uniform(mins_, maxs_, inertia=1.0, frozen=True), 
			maxs_, "maxs", 1.0, True, torch.int32)
		_test_initialization(Uniform(mins_, maxs_, inertia=1.0, frozen=False), 
			maxs_, "maxs", 1.0, False, torch.int32)


def test_initialization_float():
	funcs = (lambda x: x, tuple, 
		lambda x: numpy.array(x, dtype=numpy.float32), 
		lambda x: torch.tensor(x, dtype=torch.float32), 
		lambda x: torch.nn.Parameter(torch.tensor(x), requires_grad=False))

	mins = [1.0, 2.2, 3.9, 8.1, 1.0]
	maxs = [2.0, 3.2, 4.9, 9.1, 2.0]
	for func in funcs:
		mins_ = func(mins)
		maxs_ = func(maxs)

		_test_initialization(Uniform(mins_, maxs_, inertia=0.0, frozen=False), 
			mins_, "mins", 0.0, False, torch.float32)
		_test_initialization(Uniform(mins_, maxs_, inertia=0.3, frozen=False), 
			mins_, "mins", 0.3, False, torch.float32)
		_test_initialization(Uniform(mins_, maxs_, inertia=1.0, frozen=True), 
			mins_, "mins", 1.0, True, torch.float32)
		_test_initialization(Uniform(mins_, maxs_, inertia=1.0, frozen=False), 
			mins_, "mins", 1.0, False, torch.float32)

		_test_initialization(Uniform(mins_, maxs_, inertia=0.0, frozen=False), 
			maxs_, "maxs", 0.0, False, torch.float32)
		_test_initialization(Uniform(mins_, maxs_, inertia=0.3, frozen=False), 
			maxs_, "maxs", 0.3, False, torch.float32)
		_test_initialization(Uniform(mins_, maxs_, inertia=1.0, frozen=True), 
			maxs_, "maxs", 1.0, True, torch.float32)
		_test_initialization(Uniform(mins_, maxs_, inertia=1.0, frozen=False), 
			maxs_, "maxs", 1.0, False, torch.float32)

	mins_ = numpy.array(mins, dtype=numpy.float64)
	maxs_ = numpy.array(maxs, dtype=numpy.float64)

	_test_initialization(Uniform(mins_, maxs_, inertia=0.0, frozen=False), 
		mins_, "mins", 0.0, False, torch.float64)
	_test_initialization(Uniform(mins_, maxs_, inertia=0.3, frozen=False), 
		mins_, "mins", 0.3, False, torch.float64)
	_test_initialization(Uniform(mins_, maxs_, inertia=1.0, frozen=True), 
		mins_, "mins", 1.0, True, torch.float64)
	_test_initialization(Uniform(mins_, maxs_, inertia=1.0, frozen=False), 
		mins_, "mins", 1.0, False, torch.float64)

	_test_initialization(Uniform(mins_, maxs_, inertia=0.0, frozen=False), 
		maxs_, "maxs", 0.0, False, torch.float64)
	_test_initialization(Uniform(mins_, maxs_, inertia=0.3, frozen=False), 
		maxs_, "maxs", 0.3, False, torch.float64)
	_test_initialization(Uniform(mins_, maxs_, inertia=1.0, frozen=True), 
		maxs_, "maxs", 1.0, True, torch.float64)
	_test_initialization(Uniform(mins_, maxs_, inertia=1.0, frozen=False), 
		maxs_, "maxs", 1.0, False, torch.float64)


def test_initialization_raises():
	_test_initialization_raises_two_parameters(Uniform, VALID_VALUE, 
		VALID_VALUE, min_value1=MIN_VALUE, min_value2=MIN_VALUE)


def test_reset_cache(X):
	d = Uniform()
	d.summarize(X)
	assert_array_almost_equal(d._x_mins, [0.5, 0.2, 0.1])
	assert_array_almost_equal(d._x_maxs, [5.4, 2.1, 4.])

	d._reset_cache()
	assert_array_almost_equal(d._x_mins, [INF, INF, INF])
	assert_array_almost_equal(d._x_maxs, [NEGINF, NEGINF, NEGINF])

	d = Uniform()
	assert_raises(AttributeError, getattr, d, "_x_mins")
	assert_raises(AttributeError, getattr, d, "_x_maxs")
	assert_raises(AttributeError, getattr, d, "_logps")

	d._reset_cache()
	assert_raises(AttributeError, getattr, d, "_x_mins")
	assert_raises(AttributeError, getattr, d, "_x_maxs")
	assert_raises(AttributeError, getattr, d, "_logps")


def test_initialize(X):
	d = Uniform()
	assert d.d is None
	assert d.mins is None
	assert d.maxs is None
	assert d._initialized == False
	assert_raises(AttributeError, getattr, d, "_x_mins")
	assert_raises(AttributeError, getattr, d, "_x_maxs")
	assert_raises(AttributeError, getattr, d, "_logps")

	d = Uniform([1.2], None)
	assert d.d is None
	assert d.maxs is None
	assert d._initialized == False
	assert_raises(AttributeError, getattr, d, "_x_mins")
	assert_raises(AttributeError, getattr, d, "_x_maxs")
	assert_raises(AttributeError, getattr, d, "_logps")

	d = Uniform(None, [1.2])
	assert d.d is None
	assert d.mins is None
	assert d._initialized == False
	assert_raises(AttributeError, getattr, d, "_x_mins")
	assert_raises(AttributeError, getattr, d, "_x_maxs")
	assert_raises(AttributeError, getattr, d, "_logps")

	d._initialize(3)
	assert d._initialized == True
	assert d.mins.shape[0] == 3
	assert d.d == 3
	assert_array_almost_equal(d._x_mins, [INF, INF, INF])
	assert_array_almost_equal(d._x_maxs, [NEGINF, NEGINF, NEGINF])


	d._initialize(2)
	assert d._initialized == True
	assert d.mins.shape[0] == 2
	assert d.d == 2
	assert_array_almost_equal(d._x_mins, [INF, INF])
	assert_array_almost_equal(d._x_maxs, [NEGINF, NEGINF])	


	d = Uniform([1.2, 9.3], [1.1, 9.2])
	assert d._initialized == True
	assert d.d == 2

	d._initialize(3)
	assert d._initialized == True
	assert d.mins.shape[0] == 3
	assert d.d == 3
	assert_array_almost_equal(d._x_mins, [INF, INF, INF])
	assert_array_almost_equal(d._x_maxs, [NEGINF, NEGINF, NEGINF])

	d = Uniform()
	d.summarize(X)
	d._initialize(4)
	assert d._initialized == True
	assert d.mins.shape[0] == 4
	assert d.d == 4
	assert_array_almost_equal(d._x_mins, [INF, INF, INF, INF])
	assert_array_almost_equal(d._x_maxs, [NEGINF, NEGINF, NEGINF, NEGINF])	


###


@pytest.mark.sample
def test_sample(mins, maxs):
	torch.manual_seed(0)

	X = Uniform(mins, maxs).sample(1)
	assert_array_almost_equal(X, [[1.491016, 2.427843, 1.132716]])

	X = Uniform(mins, maxs).sample(5)
	assert_array_almost_equal(X, 
		[[0.6169, 1.0915, 1.9511],
         [1.4762, 2.7997, 1.6834],
         [1.8175, 1.2118, 1.6026],
         [0.3536, 0.6897, 1.4408],
         [1.5445, 2.2232, 2.2000]], 3)


###


def test_probability(X, mins, maxs):
	mins_, maxs_ = [1.3], [2.3]
	x = [[1.0], [2.0], [8.0], [3.7], [1.9]]
	y = [0., 1., 0., 0., 1.]

	d1 = Uniform(mins_, maxs_)
	d2 = Uniform(numpy.array(mins_, dtype=numpy.float64), 
		numpy.array(maxs_, dtype=numpy.float64))

	_test_predictions(x, y, d1.probability(x), torch.float32)
	_test_predictions(x, y, d2.probability(x), torch.float64)

	x = [[1.0, 2.0, 1.5]]
	y = [0.095785]

	d1 = Uniform(mins, maxs)
	d2 = Uniform(numpy.array(mins, dtype=numpy.float64), 
		numpy.array(maxs, dtype=numpy.float64))

	_test_predictions(x, y, d1.probability(x), torch.float32)
	_test_predictions(x, y, d2.probability(x), torch.float64)

	s, r = [0, 2, 4], [2, 4, 8]
	x = [[1, 3, 5],
	     [1, 3, 4],
	     [3, 5, 5],
	     [2, 4, 1]]
	y = [0.0625, 0.0625, 0.    , 0. ]

	d1 = Uniform(s, r)
	d2 = Uniform(numpy.array(s, dtype=numpy.int32), 
		numpy.array(r, dtype=numpy.int32))

	_test_predictions(x, y, d1.probability(x), torch.float32)
	_test_predictions(x, y, d2.probability(x), torch.float32)

	s, r = [0.0, 2.0, 4.0], [2.0, 4.0, 8.0]
	d1 = Uniform(s, r)
	d2 = Uniform(numpy.array(s, dtype=numpy.float64), 
		numpy.array(r, dtype=numpy.float64))
	_test_predictions(x, y, d1.probability(x), torch.float32)
	_test_predictions(x, y, d2.probability(x), torch.float64)

	y = [0.      , 0.095785, 0.095785, 0.      , 0.      , 0.      ,
           0.  ]

	d1 = Uniform(mins, maxs)
	d2 = Uniform(numpy.array(mins, dtype=numpy.float64),
		numpy.array(maxs, dtype=numpy.float64))

	_test_predictions(X, y, d1.probability(X), torch.float32)
	_test_predictions(X, y, d2.probability(X), torch.float64)


def test_probability_dtypes():
	X = numpy.random.uniform(0, 5, size=(10, 3)).astype(numpy.float32)
	X_int = X.astype('int32')
	p = numpy.array([0.1, 5.3, 2.5], dtype=numpy.float32)
	
	y = Uniform(p, p+1).probability(X)
	assert y.dtype == torch.float32

	y = Uniform(p, p+1).probability(X_int)
	assert y.dtype == torch.float32

	X = numpy.random.uniform(0, 5, size=(10, 3)).astype(numpy.float64)
	X_int = X.astype('int32')
	p = numpy.array([0.1, 5.3, 2.5], dtype=numpy.float64)

	y = Uniform(p, p+1).probability(X)
	assert y.dtype == torch.float64

	y = Uniform(p, p+1).probability(X_int)
	assert y.dtype == torch.float64


def test_probability_raises(X, mins, maxs):
	_test_raises(Uniform(mins, maxs), "probability", X, min_value=MIN_VALUE, 
		max_value=MAX_VALUE)

	_test_raises(Uniform([VALID_VALUE], [VALID_VALUE+1]), "probability", X, 
		min_value=MIN_VALUE, max_value=MAX_VALUE)


def test_log_probability(X, mins, maxs):
	mins_, maxs_ = [1.3], [2.3]
	x = [[1.0], [2.0], [8.0], [3.7], [1.9]]
	y = [NEGINF, 0., NEGINF, NEGINF, 0.]

	d1 = Uniform(mins_, maxs_)
	d2 = Uniform(numpy.array(mins_, dtype=numpy.float64), 
		numpy.array(maxs_, dtype=numpy.float64))

	_test_predictions(x, y, d1.log_probability(x), torch.float32)
	_test_predictions(x, y, d2.log_probability(x), torch.float64)

	x = [[1.0, 2.0, 1.5]]
	y = [-2.345645]

	d1 = Uniform(mins, maxs)
	d2 = Uniform(numpy.array(mins, dtype=numpy.float64), 
		numpy.array(maxs, dtype=numpy.float64))

	_test_predictions(x, y, d1.log_probability(x), torch.float32)
	_test_predictions(x, y, d2.log_probability(x), torch.float64)

	s, r = [0, 2, 4], [2, 4, 8]
	x = [[1, 3, 5],
	     [1, 3, 4],
	     [3, 5, 5],
	     [2, 4, 1]]
	y = [-2.772589, -2.772589,      NEGINF,      NEGINF ]

	d1 = Uniform(s, r)
	d2 = Uniform(numpy.array(s, dtype=numpy.int32), 
		numpy.array(r, dtype=numpy.int32))

	_test_predictions(x, y, d1.log_probability(x), torch.float32)
	_test_predictions(x, y, d2.log_probability(x), torch.float32)

	s, r = [0.0, 2.0, 4.0], [2.0, 4.0, 8.0]
	d1 = Uniform(s, r)
	d2 = Uniform(numpy.array(s, dtype=numpy.float64), 
		numpy.array(r, dtype=numpy.float64))
	_test_predictions(x, y, d1.log_probability(x), torch.float32)
	_test_predictions(x, y, d2.log_probability(x), torch.float64)

	y = [NEGINF, -2.345645, -2.345645,      NEGINF,      NEGINF,      NEGINF,
                NEGINF]

	d1 = Uniform(mins, maxs)
	d2 = Uniform(numpy.array(mins, dtype=numpy.float64),
		numpy.array(maxs, dtype=numpy.float64))

	_test_predictions(X, y, d1.log_probability(X), torch.float32)
	_test_predictions(X, y, d2.log_probability(X), torch.float64)


def test_log_probability_dtypes():
	X = numpy.random.uniform(0, 5, size=(10, 3)).astype(numpy.float32)
	X_int = X.astype('int32')
	p = numpy.array([0.1, 5.3, 2.5], dtype=numpy.float32)
	
	y = Uniform(p, p+1).log_probability(X)
	assert y.dtype == torch.float32

	y = Uniform(p, p+1).log_probability(X_int)
	assert y.dtype == torch.float32

	X = numpy.random.uniform(0, 5, size=(10, 3)).astype(numpy.float64)
	X_int = X.astype('int32')
	p = numpy.array([0.1, 5.3, 2.5], dtype=numpy.float64)

	y = Uniform(p, p+1).log_probability(X)
	assert y.dtype == torch.float64

	y = Uniform(p, p+1).log_probability(X_int)
	assert y.dtype == torch.float64


def test_log_probability_raises(X, mins, maxs):
	_test_raises(Uniform(mins, maxs), "log_probability", X, 
		min_value=MIN_VALUE, max_value=MAX_VALUE)

	_test_raises(Uniform([VALID_VALUE], [VALID_VALUE+1]), "log_probability", X, 
		min_value=MIN_VALUE, max_value=MAX_VALUE)


###


def test_summarize(X, mins, maxs):
	for mins_, maxs_ in ((mins, maxs), (None, None)):
		d = Uniform(mins_, maxs_)
		d.summarize(X[:4])
		assert_array_almost_equal(d._x_mins, [0.5, 0.2, 0.5])
		assert_array_almost_equal(d._x_maxs, [3.1, 2.1, 2.2])

		d.summarize(X[4:])
		assert_array_almost_equal(d._x_mins, [0.5, 0.2, 0.1])
		assert_array_almost_equal(d._x_maxs, [5.4, 2.1, 4.])

		d = Uniform(mins_, maxs_)
		d.summarize(X)
		assert_array_almost_equal(d._x_mins, [0.5, 0.2, 0.1])
		assert_array_almost_equal(d._x_maxs, [5.4, 2.1, 4.])


def test_summarize_weighted(X, w, mins, maxs):
	d = Uniform(mins, maxs)
	d.summarize(X[:4], sample_weight=w[:4])
	assert_array_almost_equal(d._x_mins, [0.5, 0.2, 0.5])
	assert_array_almost_equal(d._x_maxs, [3.1, 2.1, 2.2])

	d.summarize(X[4:], sample_weight=w[4:])
	assert_array_almost_equal(d._x_mins, [0.5, 0.2, 0.1])
	assert_array_almost_equal(d._x_maxs, [5.4, 2.1, 4.])

	d = Uniform(mins, maxs)
	d.summarize(X, sample_weight=w)
	assert_array_almost_equal(d._x_mins, [0.5, 0.2, 0.1])
	assert_array_almost_equal(d._x_maxs, [5.4, 2.1, 4.])


def test_summarize_weighted_flat(X, w, mins, maxs):
	w = numpy.array(w)[:,0] 

	d = Uniform(mins, maxs)
	d.summarize(X[:4], sample_weight=w[:4])
	assert_array_almost_equal(d._x_mins, [0.5, 0.2, 0.5])
	assert_array_almost_equal(d._x_maxs, [3.1, 2.1, 2.2])

	d.summarize(X[4:], sample_weight=w[4:])
	assert_array_almost_equal(d._x_mins, [0.5, 0.2, 0.1])
	assert_array_almost_equal(d._x_maxs, [5.4, 2.1, 4.])

	d = Uniform(mins, maxs)
	d.summarize(X, sample_weight=w)
	assert_array_almost_equal(d._x_mins, [0.5, 0.2, 0.1])
	assert_array_almost_equal(d._x_maxs, [5.4, 2.1, 4.])


def test_summarize_weighted_2d(X):
	d = Uniform()
	d.summarize(X[:4], sample_weight=X[:4])
	assert_array_almost_equal(d._x_mins, [0.5, 0.2, 0.5])
	assert_array_almost_equal(d._x_maxs, [3.1, 2.1, 2.2])

	d.summarize(X[4:], sample_weight=X[4:])
	assert_array_almost_equal(d._x_mins, [0.5, 0.2, 0.1])
	assert_array_almost_equal(d._x_maxs, [5.4, 2.1, 4.])

	d = Uniform()
	d.summarize(X, sample_weight=X)
	assert_array_almost_equal(d._x_mins, [0.5, 0.2, 0.1])
	assert_array_almost_equal(d._x_maxs, [5.4, 2.1, 4.])


def test_summarize_dtypes(X, w):
	X = numpy.array(X)
	X = X.astype(numpy.float32)

	p = numpy.array([3.0, 1.1, 2.8], dtype=numpy.float32)
	d = Uniform(p, p+1)
	assert d._x_maxs.dtype == torch.float32
	d.summarize(X)
	assert d._x_maxs.dtype == torch.float32

	X = X.astype(numpy.float64)
	p = numpy.array([3.0, 1.1, 2.8], dtype=numpy.float32)
	d = Uniform(p, p+1)
	assert d._x_maxs.dtype == torch.float32
	d.summarize(X)
	assert d._x_maxs.dtype == torch.float64

	X = X.astype(numpy.int32)
	p = numpy.array([3.0, 1.1, 2.8], dtype=numpy.float32)
	d = Uniform(p, p+1)
	assert d._x_maxs.dtype == torch.float32
	d.summarize(X)
	assert d._x_maxs.dtype == torch.float32

	X = X.astype(numpy.int64)
	p = numpy.array([3.0, 1.1, 2.8], dtype=numpy.float32)
	d = Uniform(p, p+1)
	assert d._x_maxs.dtype == torch.float32
	d.summarize(X)
	assert d._x_maxs.dtype == torch.float32


def test_summarize_raises(X, w, mins, maxs):
	_test_raises(Uniform(mins, maxs), "summarize", X, w=w, 
		min_value=MIN_VALUE, max_value=MAX_VALUE)

	_test_raises(Uniform(), "summarize", X, w=w, 
		min_value=MIN_VALUE, max_value=MAX_VALUE)

	_test_raises(Uniform([VALID_VALUE], [VALID_VALUE]), "summarize", X, w=w, 
		min_value=MIN_VALUE, max_value=MAX_VALUE)


def _test_fit_params(d, mins, maxs):
	assert_array_almost_equal(d.mins, mins)
	assert_array_almost_equal(d.maxs, maxs)

	assert_array_almost_equal(d._x_mins, numpy.zeros(d.d) + INF)
	assert_array_almost_equal(d._x_maxs, numpy.zeros(d.d) + NEGINF)
	

def test_from_summaries(X, mins, maxs):
	d = Uniform(mins, maxs)
	d.summarize(X)
	d.from_summaries()
	assert_raises(AssertionError, assert_array_almost_equal, mins, d.mins)

	for param1, param2 in (mins, maxs), (None, None):
		d = Uniform(param1, param2)
		d.summarize(X[:4])
		d.from_summaries()
		_test_fit_params(d, [0.5, 0.2, 0.5], [3.1, 2.1, 2.2])	

		d.summarize(X[4:])
		d.from_summaries()
		_test_fit_params(d, [2.2, 1. , 0.1], [5.4, 1.9, 4.])

		d = Uniform(param1, param2)
		d.summarize(X[:4])
		d.summarize(X[4:])
		d.from_summaries()
		_test_fit_params(d, [0.5, 0.2, 0.1], [5.4, 2.1, 4.])

		d = Uniform(param1, param2)
		d.summarize(X)
		d.from_summaries()
		_test_fit_params(d, [0.5, 0.2, 0.1], [5.4, 2.1, 4.])


def test_from_summaries_weighted(X, w, mins, maxs):
	for param in mins, None:
		d = Uniform(mins, maxs)
		d.summarize(X[:4], sample_weight=w[:4])
		d.from_summaries()
		_test_fit_params(d, [0.5, 0.2, 0.5], [3.1, 2.1, 2.2])	
	
		d.summarize(X[4:], sample_weight=w[4:])
		d.from_summaries()
		_test_fit_params(d, [2.2, 1. , 0.1], [5.4, 1.9, 4.])

		d = Uniform(mins, maxs)
		d.summarize(X, sample_weight=w)
		d.from_summaries()
		_test_fit_params(d, [0.5, 0.2, 0.1], [5.4, 2.1, 4.])
	

def test_from_summaries_null():
	d = Uniform([1, 2], [1, 2])
	d.from_summaries()
	assert d.mins[0] != 1 and d.mins[1] != 2
	assert d.maxs[0] != 1 and d.maxs[1] != 2 
	assert_array_almost_equal(d._x_mins, [INF, INF])
	assert_array_almost_equal(d._x_maxs, [NEGINF, NEGINF])

	d = Uniform([1, 2], [1, 2], inertia=0.5)
	d.from_summaries()
	assert d.mins[0] != 1 and d.mins[1] != 2
	assert d.maxs[0] != 1 and d.maxs[1] != 2 
	assert_array_almost_equal(d._x_mins, [INF, INF])
	assert_array_almost_equal(d._x_maxs, [NEGINF, NEGINF])


	d = Uniform([1, 2], [1, 2], inertia=0.5, frozen=True)
	d.from_summaries()
	_test_fit_params(d, [1, 2], [1, 2])


def test_from_summaries_inertia(X, w, mins, maxs):
	d = Uniform(mins, maxs, inertia=0.3)
	d.summarize(X[:4])
	d.from_summaries()
	_test_fit_params(d, [0.44, 0.2 , 0.65], [2.98, 2.4 , 2.29])	

	d.summarize(X[4:])
	d.from_summaries()
	_test_fit_params(d, [1.672, 0.76 , 0.265], [4.674, 2.05 , 3.487])

	d = Uniform(mins, maxs, inertia=0.3)
	d.summarize(X)
	d.from_summaries()
	_test_fit_params(d, [0.44, 0.2 , 0.37], [4.59, 2.4 , 3.55])


def test_from_summaries_weighted_inertia(X, w, mins, maxs):
	d = Uniform(mins, maxs, inertia=0.3)
	d.summarize(X, sample_weight=w)
	d.from_summaries()
	_test_fit_params(d, [0.44, 0.2 , 0.37], [4.59, 2.4 , 3.55])

	d = Uniform(mins, maxs, inertia=1.0)
	d.summarize(X[:4])
	d.from_summaries()
	_test_fit_params(d, mins, maxs)

	d.summarize(X[4:])
	d.from_summaries()
	_test_fit_params(d, mins, maxs)

	d = Uniform(mins, maxs, inertia=1.0)
	d.summarize(X)
	d.from_summaries()
	_test_fit_params(d, mins, maxs)


def test_from_summaries_frozen(X, w, mins, maxs):
	d = Uniform(mins, maxs, frozen=True)
	d.summarize(X[:4])
	assert_array_almost_equal(d._x_mins, [INF, INF, INF])
	assert_array_almost_equal(d._x_maxs, [NEGINF, NEGINF, NEGINF])

	d.from_summaries()
	_test_fit_params(d, mins, maxs)

	d.summarize(X[4:])
	assert_array_almost_equal(d._x_mins, [INF, INF, INF])
	assert_array_almost_equal(d._x_maxs, [NEGINF, NEGINF, NEGINF])

	d.from_summaries()
	_test_fit_params(d, mins, maxs)

	d = Uniform(mins, maxs, frozen=True)
	d.summarize(X)
	assert_array_almost_equal(d._x_mins, [INF, INF, INF])
	assert_array_almost_equal(d._x_maxs, [NEGINF, NEGINF, NEGINF])

	d.from_summaries()
	_test_fit_params(d, mins, maxs)

	d = Uniform(mins, maxs, frozen=True)
	d.summarize(X, sample_weight=w)
	assert_array_almost_equal(d._x_mins, [INF, INF, INF])
	assert_array_almost_equal(d._x_maxs, [NEGINF, NEGINF, NEGINF])

	d.from_summaries()
	_test_fit_params(d, mins, maxs)

def test_from_summaries_dtypes(X, mins):
	X = numpy.array(X)
	X = X.astype(numpy.float32)

	p = numpy.array(mins, dtype=numpy.float32)
	d = Uniform(p, p+1)
	d.summarize(X)
	d.from_summaries()
	assert d.mins.dtype == torch.float32
	assert d.maxs.dtype == torch.float32


	p = numpy.array(mins, dtype=numpy.float64)
	d = Uniform(p, p+1)
	d.summarize(X)
	d.from_summaries()
	assert d.mins.dtype == torch.float64
	assert d.maxs.dtype == torch.float64


	p = numpy.array(mins, dtype=numpy.int32)
	d = Uniform(p, p+1)
	d.summarize(X)
	d.from_summaries()
	assert d.mins.dtype == torch.int32
	assert d.maxs.dtype == torch.int32


	X = numpy.array(X)
	X = X.astype(numpy.float64)

	p = numpy.array(mins, dtype=numpy.float64)
	d = Uniform(p, p+1)
	d.summarize(X)
	d.from_summaries()
	assert d.mins.dtype == torch.float64
	assert d.maxs.dtype == torch.float64


def test_from_summaries_raises():
	assert_raises(AttributeError, Uniform().from_summaries)


def test_fit(X, mins, maxs):
	d = Uniform(mins, maxs)
	d.fit(X)
	assert_raises(AssertionError, assert_array_almost_equal, mins, d.mins)

	for param1, param2 in (mins, maxs), (None, None):
		d = Uniform(param1, param2)
		d.fit(X[:4])
		_test_fit_params(d, [0.5, 0.2, 0.5], [3.1, 2.1, 2.2])	

		d.fit(X[4:])
		_test_fit_params(d, [2.2, 1. , 0.1], [5.4, 1.9, 4.])

		d = Uniform(param1, param2)
		d.fit(X)
		_test_fit_params(d, [0.5, 0.2, 0.1], [5.4, 2.1, 4.])


def test_fit_weighted(X, w, mins, maxs):
	d = Uniform(mins, maxs)
	d.fit(X[:4], sample_weight=w[:4])
	_test_fit_params(d, [0.5, 0.2, 0.5], [3.1, 2.1, 2.2])	

	d.fit(X[4:], sample_weight=w[4:])
	_test_fit_params(d, [2.2, 1. , 0.1], [5.4, 1.9, 4.])

	d = Uniform(mins, maxs)
	d.fit(X, sample_weight=w)
	_test_fit_params(d, [0.5, 0.2, 0.1], [5.4, 2.1, 4.])
	
	X = [[1.2, 0.5, 1.1, 1.9],
	     [6.2, 1.1, 2.4, 1.1]] 

	w = [[1.1], [3.5]]

	d = Uniform()
	d.fit(X, sample_weight=w)
	_test_fit_params(d, [1.2, 0.5, 1.1, 1.1], [6.2, 1.1, 2.4, 1.9])


def test_fit_chain(X):
	d = Uniform().fit(X[:4])
	_test_fit_params(d, [0.5, 0.2, 0.5], [3.1, 2.1, 2.2])	

	d.fit(X[4:])
	_test_fit_params(d, [2.2, 1. , 0.1], [5.4, 1.9, 4.])

	d = Uniform().fit(X)
	_test_fit_params(d, [0.5, 0.2, 0.1], [5.4, 2.1, 4.])


def test_fit_dtypes(X, mins):
	X = numpy.array(X)
	X = X.astype(numpy.float32)

	p = numpy.array(mins, dtype=numpy.float32)
	d = Uniform(p, p)
	d.fit(X)
	assert d.mins.dtype == torch.float32
	assert d.maxs.dtype == torch.float32

	p = numpy.array(mins, dtype=numpy.float64)
	d = Uniform(p, p)
	d.fit(X)
	assert d.mins.dtype == torch.float64
	assert d.maxs.dtype == torch.float64


	p = numpy.array(mins, dtype=numpy.int32)
	d = Uniform(p, p)
	d.fit(X)
	assert d.mins.dtype == torch.int32
	assert d.maxs.dtype == torch.int32


	X = numpy.array(X)
	X = X.astype(numpy.float64)

	p = numpy.array(mins, dtype=numpy.float64)
	d = Uniform(p, p)
	d.fit(X)
	assert d.mins.dtype == torch.float64
	assert d.maxs.dtype == torch.float64


def test_fit_raises(X, w, mins, maxs):
	_test_raises(Uniform(mins, maxs), "fit", X, w=w, min_value=MIN_VALUE, 
		max_value=MAX_VALUE)

	_test_raises(Uniform(), "fit", X, w=w, min_value=MIN_VALUE, 
		max_value=MAX_VALUE)

	_test_raises(Uniform([VALID_VALUE], [VALID_VALUE+1]), "fit", X, w=w, 
		min_value=MIN_VALUE, max_value=MAX_VALUE)


def test_serialization(X):
	d = Uniform().fit(X[:4])
	d.summarize(X[4:])

	assert_array_almost_equal(d.mins, [0.5, 0.2, 0.5])
	assert_array_almost_equal(d.maxs, [3.1, 2.1, 2.2])

	torch.save(d, ".pytest.torch")
	d2 = torch.load(".pytest.torch")
	os.system("rm .pytest.torch")

	assert_array_almost_equal(d2.mins, [0.5, 0.2, 0.5])
	assert_array_almost_equal(d2.maxs, [3.1, 2.1, 2.2])

	assert_array_almost_equal(d2._x_mins, [2.2, 1.0, 0.1])
	assert_array_almost_equal(d2._x_maxs, [5.4, 1.9, 4.0])
	assert_array_almost_equal(d2._logps, [-0.955511, -0.641854, -0.530628])
	assert_array_almost_equal(d.log_probability(X), d2.log_probability(X))

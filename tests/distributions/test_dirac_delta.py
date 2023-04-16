# test_dirac_delta.py
# Contact: Jacob Schreiber <jmschreiber91@gmail.com>

import os
import numpy
import torch
import pytest

from torchegranate.distributions import DiracDelta

from ._utils import _test_initialization_raises_one_parameter
from ._utils import _test_initialization
from ._utils import _test_predictions
from ._utils import _test_efd_from_summaries
from ._utils import _test_raises

from nose.tools import assert_raises
from numpy.testing import assert_array_almost_equal


MIN_VALUE = None
MAX_VALUE = None
VALID_VALUE = 1.0

NEGINF = float("-inf")


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
def w():
	return [[1], [2], [0], [0], [5], [1], [2]]


@pytest.fixture
def alphas():
	return [1.2, 1.8, 2.1]


###


def test_initialization():
	d = DiracDelta()
	_test_initialization(d, None, "alphas", 0.0, False, None)
	assert_raises(AttributeError, getattr, d, "_log_alphas")


def test_initialization_int():
	funcs = (lambda x: x, tuple, numpy.array, torch.tensor, 
		lambda x: torch.nn.Parameter(torch.tensor(x), requires_grad=False))

	x = [1, 2, 3, 8, 1]
	for func in funcs:
		y = func(x)
		_test_initialization(DiracDelta(y, inertia=0.0, frozen=False), 
			y, "alphas", 0.0, False, torch.int64)
		_test_initialization(DiracDelta(y, inertia=0.3, frozen=False), 
			y, "alphas", 0.3, False, torch.int64)
		_test_initialization(DiracDelta(y, inertia=1.0, frozen=True), 
			y, "alphas", 1.0, True, torch.int64)
		_test_initialization(DiracDelta(y, inertia=1.0, frozen=False), 
			y, "alphas", 1.0, False, torch.int64)

	x = numpy.array(x, dtype=numpy.int32)
	for func in funcs[2:]:
		y = func(x)
		_test_initialization(DiracDelta(y, inertia=0.0, frozen=False), 
			y, "alphas", 0.0, False, torch.int32)
		_test_initialization(DiracDelta(y, inertia=0.3, frozen=False), 
			y, "alphas", 0.3, False, torch.int32)
		_test_initialization(DiracDelta(y, inertia=1.0, frozen=True), 
			y, "alphas", 1.0, True, torch.int32)
		_test_initialization(DiracDelta(y, inertia=1.0, frozen=False), 
			y, "alphas", 1.0, False, torch.int32)


def test_initialization_float():
	funcs = (lambda x: x, tuple, 
		lambda x: numpy.array(x, dtype=numpy.float32), 
		lambda x: torch.tensor(x, dtype=torch.float32), 
		lambda x: torch.nn.Parameter(torch.tensor(x), requires_grad=False))

	x = [1.0, 2.2, 3.9, 8.1, 1.0]
	for func in funcs:
		y = func(x)
		_test_initialization(DiracDelta(y, inertia=0.0, frozen=False), 
			y, "alphas", 0.0, False, torch.float32)
		_test_initialization(DiracDelta(y, inertia=0.3, frozen=False), 
			y, "alphas", 0.3, False, torch.float32)
		_test_initialization(DiracDelta(y, inertia=1.0, frozen=True), 
			y, "alphas", 1.0, True, torch.float32)
		_test_initialization(DiracDelta(y, inertia=1.0, frozen=False), 
			y, "alphas", 1.0, False, torch.float32)

	x = numpy.array(x, dtype=numpy.float64)
	_test_initialization(DiracDelta(x, inertia=0.0, frozen=False), 
		x, "alphas", 0.0, False, torch.float64)
	_test_initialization(DiracDelta(x, inertia=0.3, frozen=False), 
		x, "alphas", 0.3, False, torch.float64)
	_test_initialization(DiracDelta(x, inertia=1.0, frozen=True), 
		x, "alphas", 1.0, True, torch.float64)
	_test_initialization(DiracDelta(x, inertia=1.0, frozen=False), 
		x, "alphas", 1.0, False, torch.float64)


def test_initialization_raises():
	_test_initialization_raises_one_parameter(DiracDelta, VALID_VALUE, 
		min_value=MIN_VALUE)


def test_initialize(X):
	d = DiracDelta()
	assert d.d is None
	assert d.alphas is None
	assert d._initialized == False

	d._initialize(3)
	assert d._initialized == True
	assert d.alphas.shape[0] == 3
	assert d.d == 3
	assert_array_almost_equal(d.alphas, [1.0, 1.0, 1.0])
	assert_array_almost_equal(d._log_alphas, [0.0, 0.0, 0.0])

	d._initialize(2)
	assert d._initialized == True
	assert d.alphas.shape[0] == 2
	assert d.d == 2
	assert_array_almost_equal(d.alphas, [1.0, 1.0])
	assert_array_almost_equal(d._log_alphas, [0.0, 0.0])

	d = DiracDelta([1.2, 9.3])
	assert d._initialized == True
	assert d.d == 2

	d._initialize(3)
	assert d._initialized == True
	assert d.alphas.shape[0] == 3
	assert d.d == 3
	assert_array_almost_equal(d.alphas, [1.0, 1.0, 1.0])
	assert_array_almost_equal(d._log_alphas, [0.0, 0.0, 0.0])

	d = DiracDelta()
	d.summarize(X)
	d._initialize(4)
	assert d._initialized == True
	assert d.alphas.shape[0] == 4
	assert d.d == 4
	assert_array_almost_equal(d._log_alphas, [0.0, 0.0, 0.0, 0.0])


###


def test_probability(X, alphas):
	x = [[0.0], [2.0], [8.0], [0.0], [1.9]]
	y = [1.0, 0.0, 0.0, 1.0, 0.0]

	d1 = DiracDelta([1.0])

	_test_predictions(x, y, d1.probability(x), torch.float32)

	x = [[1.0, 2.0, 4],
	     [0.0, 0.0, 0.0]]
	y = [0.0, numpy.prod(alphas)]

	d1 = DiracDelta(alphas)
	d2 = DiracDelta(numpy.array(alphas, dtype=numpy.float64))

	_test_predictions(x, y, d1.probability(x), torch.float32)
	_test_predictions(x, y, d2.probability(x), torch.float64)

	p = [1, 2, 4]
	x = [[1, 2, 1],
	     [2, 2, 1],
	     [0, 0, 0],
	     [0, 0, 2]]
	y = [0.0, 0.0, numpy.prod(p), 0.0]

	d1 = DiracDelta(p)
	d2 = DiracDelta(numpy.array(p, dtype=numpy.float64))

	_test_predictions(x, y, d1.probability(x), torch.float32)
	_test_predictions(x, y, d2.probability(x), torch.float64)

	p = [1.0, 2.0, 4.0]
	d1 = DiracDelta(p)
	d2 = DiracDelta(numpy.array(p, dtype=numpy.float64))
	_test_predictions(x, y, d1.probability(x), torch.float32)
	_test_predictions(x, y, d2.probability(x), torch.float64)


def test_probability_dtypes():
	X = numpy.random.uniform(0, 5, size=(10, 3)).astype(numpy.float32)
	X_int = X.astype('int32')
	p = numpy.array([0.1, 5.3, 2.5], dtype=numpy.float32)
	
	y = DiracDelta(p).probability(X)
	assert y.dtype == torch.float32

	y = DiracDelta(p).probability(X_int)
	assert y.dtype == torch.float32

	X = numpy.random.uniform(0, 5, size=(10, 3)).astype(numpy.float64)
	X_int = X.astype('int32')
	p = numpy.array([0.1, 5.3, 2.5], dtype=numpy.float64)

	y = DiracDelta(p).probability(X)
	assert y.dtype == torch.float64

	y = DiracDelta(p).probability(X_int)
	assert y.dtype == torch.float64


def test_probability_raises(X, alphas):
	_test_raises(DiracDelta(alphas), "probability", X, min_value=MIN_VALUE, 
		max_value=MAX_VALUE)

	_test_raises(DiracDelta([VALID_VALUE]), "probability", X, 
		min_value=MIN_VALUE, max_value=MAX_VALUE)


def test_log_probability(X, alphas):
	x = [[0.0], [2.0], [8.0], [0.0], [1.9]]
	y = [0.0, NEGINF, NEGINF, 0.0, NEGINF]

	d1 = DiracDelta([1.0])

	_test_predictions(x, y, d1.log_probability(x), torch.float32)

	x = [[1.0, 2.0, 4],
	     [0.0, 0.0, 0.0]]
	y = [NEGINF, numpy.log(numpy.prod(alphas))]

	d1 = DiracDelta(alphas)
	d2 = DiracDelta(numpy.array(alphas, dtype=numpy.float64))

	_test_predictions(x, y, d1.log_probability(x), torch.float32)
	_test_predictions(x, y, d2.log_probability(x), torch.float64)

	p = [1, 2, 4]
	x = [[1, 2, 1],
	     [2, 2, 1],
	     [0, 0, 0],
	     [0, 0, 2]]
	y = [NEGINF, NEGINF, numpy.log(numpy.prod(p)), NEGINF]

	d1 = DiracDelta(p)
	d2 = DiracDelta(numpy.array(p, dtype=numpy.float64))

	_test_predictions(x, y, d1.log_probability(x), torch.float32)
	_test_predictions(x, y, d2.log_probability(x), torch.float64)

	p = [1.0, 2.0, 4.0]
	d1 = DiracDelta(p)
	d2 = DiracDelta(numpy.array(p, dtype=numpy.float64))
	_test_predictions(x, y, d1.log_probability(x), torch.float32)
	_test_predictions(x, y, d2.log_probability(x), torch.float64)


def test_log_probability_dtypes():
	X = numpy.random.uniform(0, 5, size=(10, 3)).astype(numpy.float32)
	X_int = X.astype('int32')
	p = numpy.array([0.1, 5.3, 2.5], dtype=numpy.float32)
	
	y = DiracDelta(p).log_probability(X)
	assert y.dtype == torch.float32

	y = DiracDelta(p).log_probability(X_int)
	assert y.dtype == torch.float32

	X = numpy.random.uniform(0, 5, size=(10, 3)).astype(numpy.float64)
	X_int = X.astype('int32')
	p = numpy.array([0.1, 5.3, 2.5], dtype=numpy.float64)

	y = DiracDelta(p).log_probability(X)
	assert y.dtype == torch.float64

	y = DiracDelta(p).log_probability(X_int)
	assert y.dtype == torch.float64


def test_log_probability_raises(X, alphas):
	_test_raises(DiracDelta(alphas), "log_probability", X, min_value=MIN_VALUE, 
		max_value=MAX_VALUE)

	_test_raises(DiracDelta([VALID_VALUE]), "log_probability", X, 
		min_value=MIN_VALUE, max_value=MAX_VALUE)


###


def test_summarize(X, alphas):
	d = DiracDelta(alphas)
	d.summarize(X)
	assert_array_almost_equal(d.alphas, alphas)

	d = DiracDelta()
	d.summarize(X)
	assert_array_almost_equal(d.alphas, [1, 1, 1])


def test_summarize_weighted(X, w, alphas):
	d = DiracDelta(alphas)
	d.summarize(X, sample_weight=w)
	assert_array_almost_equal(d.alphas, alphas)


def test_summarize_weighted_flat(X, w, alphas):
	w = numpy.array(w)[:,0] 

	d = DiracDelta(alphas)
	d.summarize(X, sample_weight=w)
	assert_array_almost_equal(d.alphas, alphas)


def test_summarize_weighted_2d(X, alphas):
	d = DiracDelta(alphas)
	d.summarize(X, sample_weight=X)
	assert_array_almost_equal(d.alphas, alphas)


def test_summarize_raises(X, w, alphas):
	_test_raises(DiracDelta(alphas), "summarize", X, w=w, 
		min_value=MIN_VALUE, max_value=MAX_VALUE)

	_test_raises(DiracDelta(), "summarize", X, w=w, 
		min_value=MIN_VALUE, max_value=MAX_VALUE)

	_test_raises(DiracDelta([VALID_VALUE]), "summarize", X, w=w, 
		min_value=MIN_VALUE, max_value=MAX_VALUE)


def test_from_summaries(X, alphas):
	d = DiracDelta(alphas)
	d.summarize(X)
	d.from_summaries()
	assert_array_almost_equal(d.alphas, alphas)
	assert_array_almost_equal(d._log_alphas, numpy.log(alphas))

	d = DiracDelta()
	d.summarize(X)
	d.from_summaries()
	assert_array_almost_equal(d.alphas, [1., 1., 1.])
	assert_array_almost_equal(d._log_alphas, [0., 0., 0.])


def test_from_summaries_weighted(X, w, alphas):
	d = DiracDelta(alphas)
	d.summarize(X, sample_weight=w)
	d.from_summaries()
	assert_array_almost_equal(d.alphas, alphas)
	assert_array_almost_equal(d._log_alphas, numpy.log(alphas))

	d = DiracDelta()
	d.summarize(X, sample_weight=w)
	d.from_summaries()
	assert_array_almost_equal(d.alphas, [1., 1., 1.])
	assert_array_almost_equal(d._log_alphas, [0., 0., 0.])


def test_from_summaries_null():
	d = DiracDelta([1, 2])
	d.from_summaries()
	assert_array_almost_equal(d.alphas, [1, 2])
	assert_array_almost_equal(d._log_alphas, numpy.log([1, 2]))

	d = DiracDelta([1, 2], inertia=0.5)
	d.from_summaries()
	assert_array_almost_equal(d.alphas, [1, 2])
	assert_array_almost_equal(d._log_alphas, numpy.log([1, 2]))

	d = DiracDelta([1, 2], inertia=0.5, frozen=True)
	d.from_summaries()
	assert_array_almost_equal(d.alphas, [1, 2])
	assert_array_almost_equal(d._log_alphas, numpy.log([1, 2]))


def test_from_summaries_inertia(X, w, alphas):
	d = DiracDelta(alphas, inertia=0.3)
	d.summarize(X[:4])
	d.from_summaries()
	assert_array_almost_equal(d.alphas, alphas)
	assert_array_almost_equal(d._log_alphas, numpy.log(alphas))

	d.summarize(X[4:])
	d.from_summaries()
	assert_array_almost_equal(d.alphas, alphas)
	assert_array_almost_equal(d._log_alphas, numpy.log(alphas))

	d = DiracDelta(alphas, inertia=0.3)
	d.summarize(X)
	d.from_summaries()
	assert_array_almost_equal(d.alphas, alphas)
	assert_array_almost_equal(d._log_alphas, numpy.log(alphas))


def test_from_summaries_weighted_inertia(X, w, alphas):
	d = DiracDelta(alphas, inertia=0.3)
	d.summarize(X, sample_weight=w)
	d.from_summaries()
	assert_array_almost_equal(d.alphas, alphas)
	assert_array_almost_equal(d._log_alphas, numpy.log(alphas))

	d = DiracDelta(alphas, inertia=1.0)
	d.summarize(X[:4])
	d.from_summaries()
	assert_array_almost_equal(d.alphas, alphas)
	assert_array_almost_equal(d._log_alphas, numpy.log(alphas))

	d.summarize(X[4:])
	d.from_summaries()
	assert_array_almost_equal(d.alphas, alphas)
	assert_array_almost_equal(d._log_alphas, numpy.log(alphas))

	d = DiracDelta(alphas, inertia=1.0)
	d.summarize(X)
	d.from_summaries()
	assert_array_almost_equal(d.alphas, alphas)
	assert_array_almost_equal(d._log_alphas, numpy.log(alphas))


def test_from_summaries_frozen(X, w, alphas):
	d = DiracDelta(alphas, frozen=True)
	d.summarize(X[:4])
	d.from_summaries()
	assert_array_almost_equal(d.alphas, alphas)
	assert_array_almost_equal(d._log_alphas, numpy.log(alphas))

	d.summarize(X[4:])
	d.from_summaries()
	assert_array_almost_equal(d.alphas, alphas)
	assert_array_almost_equal(d._log_alphas, numpy.log(alphas))

	d = DiracDelta(alphas, frozen=True)
	d.summarize(X)
	assert_array_almost_equal(d.alphas, alphas)
	assert_array_almost_equal(d._log_alphas, numpy.log(alphas))


def test_from_summaries_dtypes(X, alphas):
	X = numpy.array(X)
	X = X.astype(numpy.float32)

	p = numpy.array(alphas, dtype=numpy.float32)
	d = DiracDelta(p)
	d.summarize(X)
	d.from_summaries()
	assert d.alphas.dtype == torch.float32
	assert d._log_alphas.dtype == torch.float32

	p = numpy.array(alphas, dtype=numpy.float64)
	d = DiracDelta(p)
	d.summarize(X)
	d.from_summaries()
	assert d.alphas.dtype == torch.float64
	assert d._log_alphas.dtype == torch.float64

	p = numpy.array(alphas, dtype=numpy.int32)
	d = DiracDelta(p)
	d.summarize(X)
	d.from_summaries()
	assert d.alphas.dtype == torch.int32
	assert d._log_alphas.dtype == torch.float32

	X = numpy.array(X)
	X = X.astype(numpy.float64)

	p = numpy.array(alphas, dtype=numpy.float64)
	d = DiracDelta(p)
	d.summarize(X)
	d.from_summaries()
	assert d.alphas.dtype == torch.float64
	assert d._log_alphas.dtype == torch.float64


def test_fit(X, alphas):
	d = DiracDelta(alphas)
	d.fit(X)
	assert_array_almost_equal(d.alphas, alphas)
	assert_array_almost_equal(d._log_alphas, numpy.log(alphas))

	d = DiracDelta()
	d.fit(X)
	assert_array_almost_equal(d.alphas, [1, 1, 1])
	assert_array_almost_equal(d._log_alphas, [0, 0, 0])


def test_fit_weighted(X, w, alphas):
	d = DiracDelta(alphas)
	d.fit(X, sample_weight=w)
	assert_array_almost_equal(d.alphas, alphas)
	assert_array_almost_equal(d._log_alphas, numpy.log(alphas))

	d = DiracDelta()
	d.fit(X, sample_weight=w)
	assert_array_almost_equal(d.alphas, [1, 1, 1])
	assert_array_almost_equal(d._log_alphas, [0, 0, 0])


def test_fit_chain(X, alphas):
	d = DiracDelta(alphas).fit(X)
	assert_array_almost_equal(d.alphas, alphas)
	assert_array_almost_equal(d._log_alphas, numpy.log(alphas))

	d = DiracDelta().fit(X)
	assert_array_almost_equal(d.alphas, [1, 1, 1])
	assert_array_almost_equal(d._log_alphas, [0, 0, 0])


def test_fit_dtypes(X, alphas):
	X = numpy.array(X)
	X = X.astype(numpy.float32)

	p = numpy.array(alphas, dtype=numpy.float32)
	d = DiracDelta(p).fit(X)
	assert d.alphas.dtype == torch.float32
	assert d._log_alphas.dtype == torch.float32

	p = numpy.array(alphas, dtype=numpy.float64)
	d = DiracDelta(p).fit(X)
	assert d.alphas.dtype == torch.float64
	assert d._log_alphas.dtype == torch.float64

	p = numpy.array(alphas, dtype=numpy.int32)
	d = DiracDelta(p).fit(X)
	assert d.alphas.dtype == torch.int32
	assert d._log_alphas.dtype == torch.float32

	X = numpy.array(X)
	X = X.astype(numpy.float64)

	p = numpy.array(alphas, dtype=numpy.float64)
	d = DiracDelta(p).fit(X)
	assert d.alphas.dtype == torch.float64
	assert d._log_alphas.dtype == torch.float64


def test_fit_raises(X, w, alphas):
	_test_raises(DiracDelta(alphas), "fit", X, w=w, min_value=MIN_VALUE, 
		max_value=MAX_VALUE)

	_test_raises(DiracDelta(), "fit", X, w=w, min_value=MIN_VALUE, 
		max_value=MAX_VALUE)

	_test_raises(DiracDelta([VALID_VALUE]), "fit", X, w=w, 
		min_value=MIN_VALUE, max_value=MAX_VALUE)

'''
def test_serialization():
	d = DiracDelta([1., 1., 2.])
	assert_array_almost_equal(d.alphas, [1., 1., 2.])
	assert_array_almost_equal(d._log_alphas, numpy.log([1., 1., 2.]))

	torch.save(d, ".pytest.torch")
	d2 = torch.load(".pytest.torch")
	os.system("rm .pytest.torch")

	assert_array_almost_equal(d2.alphas, [1., 1., 2.])
	assert_array_almost_equal(d2._log_alphas, numpy.log([1., 1., 2.]))

	assert_array_almost_equal(d.log_probability(X), d2.log_probability(X))
'''
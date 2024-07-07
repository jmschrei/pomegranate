# test_normal_diagonal.py
# Contact: Jacob Schreiber <jmschreiber91@gmail.com>

import os
import numpy
import torch
import pytest

from pomegranate.distributions import Normal

from ._utils import _test_initialization_raises_two_parameters
from ._utils import _test_initialization
from ._utils import _test_predictions
from ._utils import _test_efd_from_summaries
from ._utils import _test_raises

from ..tools import assert_raises
from numpy.testing import assert_array_almost_equal

SQRT_2_PI = 2.50662827463
LOG_2_PI = 1.83787706641

MIN_VALUE = None
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
def w():
	return [[1], [2], [0], [0], [5], [1], [2]]


@pytest.fixture
def means():
	return [1.2, 1.8, 2.1]


@pytest.fixture
def covs():
	return [0.3, 3.1, 1.2]


###


def test_initialization():
	d = Normal(covariance_type='diag')
	
	_test_initialization(d, None, "means", 0.0, False, None)
	_test_initialization(d, None, "covs", 0.0, False, None)

	assert_raises(AttributeError, getattr, d, "_w_sum")
	assert_raises(AttributeError, getattr, d, "_xw_sum")
	assert_raises(AttributeError, getattr, d, "_xxw_sum")

	assert_raises(AttributeError, getattr, d, "_inv_cov")
	assert_raises(AttributeError, getattr, d, "_inv_cov_dot_mu")
	assert_raises(AttributeError, getattr, d, "_log_det")
	assert_raises(AttributeError, getattr, d, "_log_sigma_sqrt_2pi")


def test_initialization_int():
	funcs = (lambda x: x, tuple, numpy.array, torch.tensor, 
		lambda x: torch.nn.Parameter(torch.tensor(x), requires_grad=False))

	x = [1, 2, 3, 8, 1]
	for func in funcs:
		y = func(x)

		_test_initialization(Normal(y, y, covariance_type='diag', 
			inertia=0.0, frozen=False), y, "means", 0.0, False, torch.int64)
		_test_initialization(Normal(y, y, covariance_type='diag', 
			inertia=0.3, frozen=False), y, "means", 0.3, False, torch.int64)
		_test_initialization(Normal(y, y, covariance_type='diag', 
			inertia=1.0, frozen=True), y, "means", 1.0, True, torch.int64)
		_test_initialization(Normal(y, y, covariance_type='diag',
			inertia=1.0, frozen=False), y, "means", 1.0, False, torch.int64)

		_test_initialization(Normal(y, y, covariance_type='diag',
			inertia=0.0, frozen=False), y, "covs", 0.0, False, torch.int64)
		_test_initialization(Normal(y, y, covariance_type='diag',
			inertia=0.3, frozen=False), y, "covs", 0.3, False, torch.int64)
		_test_initialization(Normal(y, y, covariance_type='diag',
			inertia=1.0, frozen=True), y, "covs", 1.0, True, torch.int64)
		_test_initialization(Normal(y, y, covariance_type='diag',
			inertia=1.0, frozen=False), y, "covs", 1.0, False, torch.int64)

	x = numpy.array(x, dtype=numpy.int32)
	for func in funcs[2:]:
		y = func(x)
		_test_initialization(Normal(y, y, covariance_type='diag', 
			inertia=0.0, frozen=False), y, "means", 0.0, False, torch.int32)
		_test_initialization(Normal(y, y, covariance_type='diag', 
			inertia=0.3, frozen=False), y, "means", 0.3, False, torch.int32)
		_test_initialization(Normal(y, y, covariance_type='diag', 
			inertia=1.0, frozen=True), y, "means", 1.0, True, torch.int32)
		_test_initialization(Normal(y, y, covariance_type='diag', 
			inertia=1.0, frozen=False), y, "means", 1.0, False, torch.int32)

		_test_initialization(Normal(y, y, covariance_type='diag', 
			inertia=0.0, frozen=False), y, "covs", 0.0, False, torch.int32)
		_test_initialization(Normal(y, y, covariance_type='diag',
			inertia=0.3, frozen=False), y, "covs", 0.3, False, torch.int32)
		_test_initialization(Normal(y, y, covariance_type='diag', 
			inertia=1.0, frozen=True), y, "covs", 1.0, True, torch.int32)
		_test_initialization(Normal(y, y, covariance_type='diag',
			inertia=1.0, frozen=False), y, "covs", 1.0, False, torch.int32)


def test_initialization_float():
	funcs = (lambda x: x, tuple, 
		lambda x: numpy.array(x, dtype=numpy.float32), 
		lambda x: torch.tensor(x, dtype=torch.float32), 
		lambda x: torch.nn.Parameter(torch.tensor(x), requires_grad=False))

	x = [1.0, 2.2, 3.9, 8.1, 1.0]
	for func in funcs:
		y = func(x)

		_test_initialization(Normal(y, y, covariance_type='diag', 
			inertia=0.0, frozen=False), y, "means", 0.0, False, torch.float32)
		_test_initialization(Normal(y, y, covariance_type='diag',
			inertia=0.3, frozen=False), y, "means", 0.3, False, torch.float32)
		_test_initialization(Normal(y, y, covariance_type='diag',
			inertia=1.0, frozen=True), y, "means", 1.0, True, torch.float32)
		_test_initialization(Normal(y, y, covariance_type='diag',
			inertia=1.0, frozen=False), y, "means", 1.0, False, torch.float32)

		_test_initialization(Normal(y, y, covariance_type='diag',
			inertia=0.0, frozen=False), y, "covs", 0.0, False, torch.float32)
		_test_initialization(Normal(y, y, covariance_type='diag',
			inertia=0.3, frozen=False), y, "covs", 0.3, False, torch.float32)
		_test_initialization(Normal(y, y, covariance_type='diag',
			inertia=1.0, frozen=True), y, "covs", 1.0, True, torch.float32)
		_test_initialization(Normal(y, y, covariance_type='diag',
			inertia=1.0, frozen=False), y, "covs", 1.0, False, torch.float32)

	y = numpy.array(x, dtype=numpy.float64)

	_test_initialization(Normal(y, y, covariance_type='diag',
		inertia=0.0, frozen=False), y, "means", 0.0, False, torch.float64)
	_test_initialization(Normal(y, y, covariance_type='diag',
		inertia=0.3, frozen=False), y, "means", 0.3, False, torch.float64)
	_test_initialization(Normal(y, y, covariance_type='diag',
		inertia=1.0, frozen=True), y, "means", 1.0, True, torch.float64)
	_test_initialization(Normal(y, y, covariance_type='diag',
		inertia=1.0, frozen=False), y, "means", 1.0, False, torch.float64)

	_test_initialization(Normal(y, y, covariance_type='diag',
		inertia=0.0, frozen=False), y, "covs", 0.0, False, torch.float64)
	_test_initialization(Normal(y, y, covariance_type='diag',
		inertia=0.3, frozen=False), y, "covs", 0.3, False, torch.float64)
	_test_initialization(Normal(y, y, covariance_type='diag',
		inertia=1.0, frozen=True), y, "covs", 1.0, True, torch.float64)
	_test_initialization(Normal(y, y, covariance_type='diag',
		inertia=1.0, frozen=False), y, "covs", 1.0, False, torch.float64)


def test_initialization_raises():
	_test_initialization_raises_two_parameters(Normal, VALID_VALUE, VALID_VALUE,
		min_value1=MIN_VALUE, min_value2=MIN_VALUE)


def test_reset_cache(X):
	d = Normal(covariance_type='diag')
	d.summarize(X)
	assert_array_almost_equal(d._w_sum, [7.0, 7.0, 7.0])
	assert_array_almost_equal(d._xw_sum, [17.099998,  9.7     , 10.599999])
	assert_array_almost_equal(d._xxw_sum, [58.59    , 16.369999, 27.720001])

	d._reset_cache()
	assert_array_almost_equal(d._w_sum, [0.0, 0.0, 0.0])
	assert_array_almost_equal(d._xw_sum, [0.0, 0.0, 0.0])
	assert_array_almost_equal(d._xxw_sum, [0.0, 0.0, 0.0])

	d = Normal()
	assert_raises(AttributeError, getattr, d, "_w_sum")
	assert_raises(AttributeError, getattr, d, "_xw_sum")
	assert_raises(AttributeError, getattr, d, "_xxw_sum")

	assert_raises(AttributeError, getattr, d, "_inv_cov")
	assert_raises(AttributeError, getattr, d, "_inv_cov_dot_mu")
	assert_raises(AttributeError, getattr, d, "_log_det")
	assert_raises(AttributeError, getattr, d, "_log_sigma_sqrt_2pi")

	d._reset_cache()
	assert_raises(AttributeError, getattr, d, "_w_sum")
	assert_raises(AttributeError, getattr, d, "_xw_sum")
	assert_raises(AttributeError, getattr, d, "_xxw_sum")

	assert_raises(AttributeError, getattr, d, "_inv_cov")
	assert_raises(AttributeError, getattr, d, "_inv_cov_dot_mu")
	assert_raises(AttributeError, getattr, d, "_log_det")
	assert_raises(AttributeError, getattr, d, "_log_sigma_sqrt_2pi")


def test_initialize(X):
	d = Normal(covariance_type='diag')
	assert d.d is None
	assert d.means is None
	assert d.covs is None
	assert d._initialized == False
	assert_raises(AttributeError, getattr, d, "_w_sum")
	assert_raises(AttributeError, getattr, d, "_xw_sum")
	assert_raises(AttributeError, getattr, d, "_xxw_sum")

	assert_raises(AttributeError, getattr, d, "_inv_cov")
	assert_raises(AttributeError, getattr, d, "_inv_cov_dot_mu")
	assert_raises(AttributeError, getattr, d, "_log_det")
	assert_raises(AttributeError, getattr, d, "_log_sigma_sqrt_2pi")

	d = Normal([1.2], None, covariance_type='diag')
	assert d.d is None
	assert d.covs is None
	assert d._initialized == False
	assert_raises(AttributeError, getattr, d, "_w_sum")
	assert_raises(AttributeError, getattr, d, "_xw_sum")
	assert_raises(AttributeError, getattr, d, "_xxw_sum")

	assert_raises(AttributeError, getattr, d, "_inv_cov")
	assert_raises(AttributeError, getattr, d, "_inv_cov_dot_mu")
	assert_raises(AttributeError, getattr, d, "_log_det")
	assert_raises(AttributeError, getattr, d, "_log_sigma_sqrt_2pi")

	d = Normal(None, [1.2], covariance_type='diag')
	assert d.d is None
	assert d.means is None
	assert d._initialized == False
	assert_raises(AttributeError, getattr, d, "_w_sum")
	assert_raises(AttributeError, getattr, d, "_xw_sum")
	assert_raises(AttributeError, getattr, d, "_xxw_sum")

	assert_raises(AttributeError, getattr, d, "_inv_cov")
	assert_raises(AttributeError, getattr, d, "_inv_cov_dot_mu")
	assert_raises(AttributeError, getattr, d, "_log_det")
	assert_raises(AttributeError, getattr, d, "_log_sigma_sqrt_2pi")

	d._initialize(3)
	assert d._initialized == True
	assert d.means.shape[0] == 3
	assert d.d == 3
	assert_array_almost_equal(d.means, [0.0, 0.0, 0.0])
	assert_array_almost_equal(d.covs, [0.0, 0.0, 0.0])
	assert_array_almost_equal(d._w_sum, [0.0, 0.0, 0.0])
	assert_array_almost_equal(d._xw_sum, [0.0, 0.0, 0.0])
	assert_array_almost_equal(d._xw_sum, [0.0, 0.0, 0.0])
	assert_array_almost_equal(d._xxw_sum, [0.0, 0.0, 0.0])	


	d._initialize(2)
	assert d._initialized == True
	assert d.means.shape[0] == 2
	assert d.d == 2
	assert_array_almost_equal(d.means, [0.0, 0.0])
	assert_array_almost_equal(d.covs, [0.0, 0.0])
	assert_array_almost_equal(d._w_sum, [0.0, 0.0])
	assert_array_almost_equal(d._xw_sum, [0.0, 0.0])
	assert_array_almost_equal(d._xw_sum, [0.0, 0.0])
	assert_array_almost_equal(d._xxw_sum, [0.0, 0.0])


	d = Normal([1.2, 9.3], [1.1, 9.2], covariance_type='diag')
	assert d._initialized == True
	assert d.d == 2

	d._initialize(3)
	assert d._initialized == True
	assert d.means.shape[0] == 3
	assert d.d == 3
	assert_array_almost_equal(d.means, [0.0, 0.0, 0.0])
	assert_array_almost_equal(d.covs, [0.0, 0.0, 0.0])

	d = Normal(covariance_type='diag')
	d.summarize(X)
	d._initialize(4)
	assert d._initialized == True
	assert d.means.shape[0] == 4
	assert d.d == 4
	assert_array_almost_equal(d._w_sum, [0.0, 0.0, 0.0, 0.0])
	assert_array_almost_equal(d._xw_sum, [0.0, 0.0, 0.0, 0.0])
	assert_array_almost_equal(d._xw_sum, [0.0, 0.0, 0.0, 0.0])
	assert_array_almost_equal(d._xxw_sum, [0.0, 0.0, 0.0, 0.0])	


###


@pytest.mark.sample
def test_sample(means, covs):
	torch.manual_seed(0)

	X = Normal(means, covs, covariance_type='diag').sample(1)
	assert_array_almost_equal(X, [[1.662299,  0.89037 , -0.514547]])

	X = Normal(means, covs, covariance_type='diag').sample(5)
	assert_array_almost_equal(X, 
		[[ 1.3705, -1.5620,  0.4217],
         [ 1.3210,  4.3979,  1.2369],
         [ 1.0790, -0.0496,  2.3184],
         [ 0.9430,  5.2119,  0.8146],
         [ 1.2368,  0.0444,  2.5477]], 3)


###


def test_probability(X, means, covs):
	m, c = [1.7], [1.3]
	x = [[1.0], [2.0], [4.0], [3.7], [1.9]]
	y = [0.289795, 0.337991, 0.045742, 0.075126, 0.344554]

	d1 = Normal(m, c, covariance_type='diag')
	d2 = Normal(numpy.array(m, dtype=numpy.float64), 
		numpy.array(c, dtype=numpy.float64), covariance_type='diag')

	_test_predictions(x, y, d1.probability(x), torch.float32)
	_test_predictions(x, y, d2.probability(x), torch.float64)

	x = [[1.0, 2.0, 4]]
	y = [0.012413]

	d1 = Normal(means, covs, covariance_type='diag')
	d2 = Normal(numpy.array(means, dtype=numpy.float64), 
		numpy.array(covs, dtype=numpy.float64), covariance_type='diag')

	_test_predictions(x, y, d1.probability(x), torch.float32)
	_test_predictions(x, y, d2.probability(x), torch.float64)

	m, c = [1, 2, 4], [2, 1, 3]
	x = [[1, 2, 1],
	     [2, 2, 1],
	     [3, 1, 1],
	     [2, 1, 2]]
	y = [0.005784, 0.004504, 0.001291, 0.006286]

	d1 = Normal(m, c, covariance_type='diag')
	d2 = Normal(numpy.array(m, dtype=numpy.int32), 
		numpy.array(c, dtype=numpy.int32), covariance_type='diag')

	_test_predictions(x, y, d1.probability(x), torch.float32)
	_test_predictions(x, y, d2.probability(x), torch.float32)

	m, c = [1.0, 2.0, 4.0], [2.0, 1.0, 3.0]
	d1 = Normal(m, c, covariance_type='diag')
	d2 = Normal(numpy.array(m, dtype=numpy.float64), 
		numpy.array(c, dtype=numpy.float64), covariance_type='diag')
	_test_predictions(x, y, d1.probability(x), torch.float32)
	_test_predictions(x, y, d2.probability(x), torch.float64)

	y = [2.004972e-02, 1.346142e-02, 5.173831e-02, 1.438069e-04,
           4.410233e-06, 2.273481e-15, 2.059388e-03]

	d1 = Normal(means, covs, covariance_type='diag')
	d2 = Normal(numpy.array(means, dtype=numpy.float64),
		numpy.array(covs, dtype=numpy.float64), covariance_type='diag')

	_test_predictions(X, y, d1.probability(X), torch.float32)
	_test_predictions(X, y, d2.probability(X), torch.float64)


def test_probability_dtypes():
	X = numpy.random.uniform(0, 5, size=(10, 3)).astype(numpy.float32)
	X_int = X.astype('int32')
	p = numpy.array([0.1, 5.3, 2.5], dtype=numpy.float32)
	
	y = Normal(p, p, covariance_type='diag').probability(X)
	assert y.dtype == torch.float32

	y = Normal(p, p, covariance_type='diag').probability(X_int)
	assert y.dtype == torch.float32

	X = numpy.random.uniform(0, 5, size=(10, 3)).astype(numpy.float64)
	X_int = X.astype('int32')
	p = numpy.array([0.1, 5.3, 2.5], dtype=numpy.float64)

	y = Normal(p, p, covariance_type='diag').probability(X)
	assert y.dtype == torch.float64

	y = Normal(p, p, covariance_type='diag').probability(X_int)
	assert y.dtype == torch.float64


def test_probability_raises(X, means, covs):
	_test_raises(Normal(means, covs, covariance_type='diag'), "probability", X, 
		min_value=MIN_VALUE, max_value=MAX_VALUE)

	_test_raises(Normal([VALID_VALUE], [VALID_VALUE], covariance_type='diag'), 
		"probability", X, min_value=MIN_VALUE, max_value=MAX_VALUE)


def test_log_probability(X, means, covs):
	m, c = [1.7], [1.3]
	x = [[1.0], [2.0], [4.0], [3.0], [2.0]]
	y = [-1.238582, -1.084736, -3.084736, -1.700121, -1.084736]

	x_torch = torch.tensor(numpy.array(x))	
	m_torch = torch.tensor(numpy.array(m))
	c_torch = torch.sqrt(torch.tensor(numpy.array(c)))

	d1 = Normal(m, c, covariance_type='diag')
	d2 = Normal(numpy.array(m, dtype=numpy.float64),
		numpy.array(c, dtype=numpy.float64), covariance_type='diag')
	d3 = torch.distributions.Normal(m_torch, c_torch)

	_test_predictions(x, y, d1.log_probability(x), torch.float32)
	_test_predictions(x, y, d2.log_probability(x), torch.float64)
	_test_predictions(x, d3.log_prob(x_torch).sum(axis=1), 
		d2.log_probability(x), torch.float64)

	p = [1.7, 2.3, 1.0, 1.7, 4.1]
	x = [[1.0, 2.0, 4.0, 3.0, 2.0]]
	y = [-11.945815]

	p_torch = torch.tensor(numpy.array(p))
	x_torch = torch.tensor(numpy.array(x))

	d1 = Normal(p, p, covariance_type='diag')
	d2 = Normal(numpy.array(p, dtype=numpy.float64),
		numpy.array(p, dtype=numpy.float64), covariance_type='diag')
	d3 = torch.distributions.Normal(p_torch, torch.sqrt(p_torch))

	_test_predictions(x, y, d1.log_probability(x), torch.float32)
	_test_predictions(x, y, d2.log_probability(x), torch.float64)
	_test_predictions(x, d3.log_prob(x_torch).sum(axis=1), 
		d2.log_probability(x), torch.float64)

	p = [1, 2, 4]
	x = [[1, 2, 1],
	     [2, 2, 1],
	     [1, 1, 3],
	     [2, 1, 2]]
	y = [-4.921536, -5.421536, -4.171536, -5.046536]

	p_torch = torch.tensor(numpy.array(p))
	x_torch = torch.tensor(numpy.array(x))

	d1 = Normal(p, p, covariance_type='diag')
	d2 = Normal(numpy.array(p, dtype=numpy.float64),
		numpy.array(p, dtype=numpy.float64), covariance_type='diag')
	d3 = torch.distributions.Normal(p_torch, torch.sqrt(p_torch))

	_test_predictions(x, y, d1.log_probability(x), torch.float32)
	_test_predictions(x, y, d2.log_probability(x), torch.float64)
	_test_predictions(x, d3.log_prob(x_torch).sum(axis=1), 
		d2.log_probability(x), torch.float64)

	p = [1.0, 2.0, 4.0]
	p_torch = torch.tensor(numpy.array(p))

	d1 = Normal(p, p, covariance_type='diag')
	d2 = Normal(numpy.array(p, dtype=numpy.float64),
		numpy.array(p, dtype=numpy.float64), covariance_type='diag')
	d3 = torch.distributions.Normal(p_torch, torch.sqrt(p_torch))

	_test_predictions(x, y, d1.log_probability(x), torch.float32)
	_test_predictions(x, y, d2.log_probability(x), torch.float64)
	_test_predictions(x, d3.log_prob(x_torch).sum(axis=1), 
		d2.log_probability(x), torch.float64)

	m_torch = torch.tensor(numpy.array(means))
	c_torch = torch.tensor(numpy.array(covs))
	x_torch = torch.tensor(numpy.array(X))

	y = [-3.90954 ,  -4.307928,  -2.961557,  -8.84704 , -12.331584,
           -33.717472,  -6.185347]

	d2 = Normal(numpy.array(means, dtype=numpy.float64),
		numpy.array(covs, dtype=numpy.float64), covariance_type='diag')
	d3 = torch.distributions.Normal(m_torch, torch.sqrt(c_torch))

	_test_predictions(X, y, d2.log_probability(X), torch.float64)
	_test_predictions(X, d3.log_prob(x_torch).sum(axis=1), 
		d2.log_probability(X), torch.float64)


def test_log_probability_dtypes():
	X = numpy.random.uniform(0, 5, size=(10, 3)).astype(numpy.float32)
	X_int = X.astype('int32')
	p = numpy.array([0.1, 5.3, 2.5], dtype=numpy.float32)
	
	y = Normal(p, p, covariance_type='diag').log_probability(X)
	assert y.dtype == torch.float32

	y = Normal(p, p, covariance_type='diag').log_probability(X_int)
	assert y.dtype == torch.float32

	X = numpy.random.uniform(0, 5, size=(10, 3)).astype(numpy.float64)
	X_int = X.astype('int32')
	p = numpy.array([0.1, 5.3, 2.5], dtype=numpy.float64)

	y = Normal(p, p, covariance_type='diag').log_probability(X)
	assert y.dtype == torch.float64

	y = Normal(p, p, covariance_type='diag').log_probability(X_int)
	assert y.dtype == torch.float64


def test_log_probability_raises(X, means, covs):
	_test_raises(Normal(means, covs, covariance_type='diag'), "log_probability",
		X, min_value=MIN_VALUE, max_value=MAX_VALUE)

	_test_raises(Normal([VALID_VALUE], [VALID_VALUE], covariance_type='diag'), 
		"log_probability", X, min_value=MIN_VALUE, max_value=MAX_VALUE)


###


def test_summarize(X, means, covs):
	for m, c in ((means, covs), (None, None)):
		d = Normal(m, c, covariance_type='diag')
		d.summarize(X[:4])
		assert_array_almost_equal(d._w_sum, [4.0, 4.0, 4.0])
		assert_array_almost_equal(d._xw_sum, [6.1, 5.5, 6.2])
		assert_array_almost_equal(d._xxw_sum, [13.03    , 10.069999, 11.62])

		d.summarize(X[4:])
		assert_array_almost_equal(d._w_sum, [7.0, 7.0, 7.0])
		assert_array_almost_equal(d._xw_sum, [17.1,  9.7, 10.6])
		assert_array_almost_equal(d._xxw_sum, [58.59    , 16.369999, 27.720001])


		d = Normal(m, c, covariance_type='diag')
		d.summarize(X)
		assert_array_almost_equal(d._w_sum, [7.0, 7.0, 7.0])
		assert_array_almost_equal(d._xw_sum, [17.099998,  9.7     , 10.599999])
		assert_array_almost_equal(d._xxw_sum, [58.59    , 16.369999, 27.720001])


def test_summarize_weighted(X, w, means, covs):
	d = Normal(means, covs, covariance_type='diag')
	d.summarize(X[:4], sample_weight=w[:4])
	assert_array_almost_equal(d._w_sum, [3., 3., 3.])
	assert_array_almost_equal(d._xw_sum, [2.1, 2.5, 3.1])
	assert_array_almost_equal(d._xxw_sum, [1.71    , 4.489999, 3.63])

	d.summarize(X[4:], sample_weight=w[4:])
	assert_array_almost_equal(d._w_sum, [11.0, 11.0, 11.0])
	assert_array_almost_equal(d._xw_sum, [28.9     , 12.      ,  8.799999])
	assert_array_almost_equal(d._xxw_sum, [98.350006, 16.48    , 20.1])

	d = Normal(means, covs, covariance_type='diag')
	d.summarize(X, sample_weight=w)
	assert_array_almost_equal(d._w_sum, [11.0, 11.0, 11.0])
	assert_array_almost_equal(d._xw_sum, [28.9     , 12.      ,  8.799999])
	assert_array_almost_equal(d._xxw_sum, [98.350006, 16.48    , 20.1])


def test_summarize_weighted_flat(X, w, means, covs):
	w = numpy.array(w)[:,0] 

	d = Normal(means, covs, covariance_type='diag')
	d.summarize(X[:4], sample_weight=w[:4])
	assert_array_almost_equal(d._w_sum, [3., 3., 3.])
	assert_array_almost_equal(d._xw_sum, [2.1, 2.5, 3.1])
	assert_array_almost_equal(d._xxw_sum, [1.71    , 4.489999, 3.63])

	d.summarize(X[4:], sample_weight=w[4:])
	assert_array_almost_equal(d._w_sum, [11.0, 11.0, 11.0])
	assert_array_almost_equal(d._xw_sum, [28.9     , 12.      ,  8.799999])
	assert_array_almost_equal(d._xxw_sum, [98.350006, 16.48    , 20.1])

	d = Normal(means, covs, covariance_type='diag')
	d.summarize(X, sample_weight=w)
	assert_array_almost_equal(d._w_sum, [11.0, 11.0, 11.0])
	assert_array_almost_equal(d._xw_sum, [28.9     , 12.      ,  8.799999])
	assert_array_almost_equal(d._xxw_sum, [98.350006, 16.48    , 20.1])


def test_summarize_weighted_2d(X):
	d = Normal(covariance_type='diag')
	d.summarize(X[:4], sample_weight=X[:4])
	assert_array_almost_equal(d._w_sum, [6.1, 5.5, 6.2])
	assert_array_almost_equal(d._xw_sum, [13.03    , 10.069999, 11.62])
	assert_array_almost_equal(d._xxw_sum, [33.990997, 19.860998, 23.618])

	d.summarize(X[4:], sample_weight=X[4:])
	assert_array_almost_equal(d._w_sum, [17.1,  9.7, 10.6])
	assert_array_almost_equal(d._xw_sum, [58.59    , 16.369999, 27.720001])
	assert_array_almost_equal(d._xxw_sum, [241.40701 ,  29.916998,  87.645996],
		5)

	d = Normal(covariance_type='diag')
	d.summarize(X, sample_weight=X)
	assert_array_almost_equal(d._w_sum, [17.099998,  9.7     , 10.599999])
	assert_array_almost_equal(d._xw_sum, [58.59    , 16.369999, 27.720001])
	assert_array_almost_equal(d._xxw_sum, [241.40703 ,  29.916996,  87.646], 5)


def test_summarize_dtypes(X, w):
	X = numpy.array(X)
	X = X.astype(numpy.float32)

	p = numpy.array([3.0, 1.1, 2.8], dtype=numpy.float32)
	d = Normal(p, p, covariance_type='diag')
	assert d._xw_sum.dtype == torch.float32
	d.summarize(X)
	assert d._xw_sum.dtype == torch.float32

	X = X.astype(numpy.float64)
	p = numpy.array([3.0, 1.1, 2.8], dtype=numpy.float32)
	d = Normal(p, p, covariance_type='diag')
	assert d._xw_sum.dtype == torch.float32
	d.summarize(X)
	assert d._xw_sum.dtype == torch.float32

	X = X.astype(numpy.int32)
	p = numpy.array([3.0, 1.1, 2.8], dtype=numpy.float32)
	d = Normal(p, p, covariance_type='diag')
	assert d._xw_sum.dtype == torch.float32
	d.summarize(X)
	assert d._xw_sum.dtype == torch.float32

	X = X.astype(numpy.int64)
	p = numpy.array([3.0, 1.1, 2.8], dtype=numpy.float32)
	d = Normal(p, p, covariance_type='diag')
	assert d._xw_sum.dtype == torch.float32
	d.summarize(X)
	assert d._xw_sum.dtype == torch.float32


def test_summarize_raises(X, w, means, covs):
	_test_raises(Normal(means, covs, covariance_type='diag'), "summarize", X, 
		w=w, min_value=MIN_VALUE, max_value=MAX_VALUE)

	_test_raises(Normal(covariance_type='diag'), "summarize", X, w=w, 
		min_value=MIN_VALUE, max_value=MAX_VALUE)

	_test_raises(Normal([VALID_VALUE], [VALID_VALUE], covariance_type='diag'), 
		"summarize", X, w=w, min_value=MIN_VALUE, max_value=MAX_VALUE)


def _test_fit_params(d, means, covs):
	assert_array_almost_equal(d.means, means)
	assert_array_almost_equal(d.covs, covs)

	assert_array_almost_equal(d._w_sum, numpy.zeros(d.d))
	assert_array_almost_equal(d._xw_sum, numpy.zeros(d.d))
	assert_array_almost_equal(d._xxw_sum, numpy.zeros(d.d))

	assert_raises(AttributeError, getattr, d, "_inv_cov")
	assert_raises(AttributeError, getattr, d, "_inv_cov_dot_mu")
	assert_raises(AttributeError, getattr, d, "_log_det")

	assert_array_almost_equal(d._log_sigma_sqrt_2pi, -numpy.log(numpy.sqrt(
		covs) * SQRT_2_PI), 4)
	assert_array_almost_equal(d._inv_two_sigma, 1. / (2 * numpy.array(covs,
		dtype=numpy.float32)), 4)


def test_from_summaries(X, means, covs):
	d = Normal(means, covs, covariance_type='diag')
	d.summarize(X)
	d.from_summaries()
	assert_raises(AssertionError, assert_array_almost_equal, means, d.means)
	assert_array_almost_equal(d.covs, torch.diag(torch.cov(torch.tensor(X).T, 
		correction=0)))

	for param1, param2 in (means, covs), (None, None):
		d = Normal(param1, param2, covariance_type='diag')
		d.summarize(X[:4])
		d.from_summaries()
		_test_fit_params(d, [1.525, 1.375, 1.55], [0.931875, 0.626875, 0.5025])	

		d.summarize(X[4:])
		d.from_summaries()
		_test_fit_params(d, [3.666667, 1.4     , 1.466667], 
			[1.742223, 0.14    , 3.215556])

		d = Normal(param1, param2, covariance_type='diag')
		d.summarize(X[:4])
		d.summarize(X[4:])
		d.from_summaries()
		_test_fit_params(d, [2.442857, 1.385714, 1.514286],
			[2.402449, 0.418367, 1.666939])

		d = Normal(param1, param2, covariance_type='diag')
		d.summarize(X)
		d.from_summaries()
		_test_fit_params(d, [2.442857, 1.385714, 1.514286],
			[2.402449, 0.418367, 1.666939])


def test_from_summaries_weighted(X, w, means, covs):
	for param in means, None:
		d = Normal(means, covs, covariance_type='diag')
		d.summarize(X[:4], sample_weight=w[:4])
		d.from_summaries()
		_test_fit_params(d, [0.7     , 0.833333, 1.033333],
			[0.08    , 0.802222, 0.142222])
	
		d.summarize(X[4:], sample_weight=w[4:])
		d.from_summaries()
		_test_fit_params(d, [3.35  , 1.1875, 0.7125],
			[0.857502, 0.088594, 1.551094])

		d = Normal(means, covs, covariance_type='diag')
		d.summarize(X, sample_weight=w)
		d.from_summaries()
		_test_fit_params(d, [2.627273, 1.090909, 0.8],
			[2.038348, 0.308099, 1.187273])
	

#def test_from_summaries_null():
#	d = Normal([1, 2], [1, 2], covariance_type='diag')
#	d.from_summaries()
#	assert d.means[0] != 1 and d.means[1] != 2
#	assert d.covs[0] != 1 and d.covs[1] != 2 
#	assert_array_almost_equal(d._w_sum, [0.0, 0.0])
#	assert_array_almost_equal(d._xw_sum, [0.0, 0.0])
#	assert_array_almost_equal(d._xxw_sum, [0.0, 0.0])
#
#	d = Normal([1, 2], [1, 2], covariance_type='diag', inertia=0.5)
#	d.from_summaries()
#	assert d.means[0] != 1 and d.means[1] != 2
#	assert d.covs[0] != 1 and d.covs[1] != 2 
#	assert_array_almost_equal(d._w_sum, [0.0, 0.0])
#	assert_array_almost_equal(d._xw_sum, [0.0, 0.0])
#	assert_array_almost_equal(d._xxw_sum, [0.0, 0.0])
#
#
#	d = Normal([1, 2], [1, 2], covariance_type='diag', inertia=0.5, frozen=True)
#	d.from_summaries()
#	_test_fit_params(d, [1, 2], [1, 2])


def test_from_summaries_inertia(X, w, means, covs):
	d = Normal(means, covs, covariance_type='diag', inertia=0.3)
	d.summarize(X[:4])
	d.from_summaries()
	_test_fit_params(d, [1.4275, 1.5025, 1.715], [0.742312, 1.368812, 0.71175])

	d.summarize(X[4:])
	d.from_summaries()
	_test_fit_params(d, [2.994917, 1.43075 , 1.541167],
		[1.44225 , 0.508644, 2.464414])

	d = Normal(means, covs, covariance_type='diag', inertia=0.3)
	d.summarize(X)
	d.from_summaries()
	_test_fit_params(d, [2.07, 1.51, 1.69], [1.771715, 1.222857, 1.526857])


def test_from_summaries_weighted_inertia(X, w, means, covs):
	d = Normal(means, covs, covariance_type='diag', inertia=0.3)
	d.summarize(X, sample_weight=w)
	d.from_summaries()
	_test_fit_params(d, [2.199091, 1.303636, 1.19],
		[1.516843, 1.145669, 1.191091])

	d = Normal(means, covs, covariance_type='diag', inertia=1.0)
	d.summarize(X[:4])
	d.from_summaries()
	_test_fit_params(d, means, covs)

	d.summarize(X[4:])
	d.from_summaries()
	_test_fit_params(d, means, covs)

	d = Normal(means, covs, covariance_type='diag', inertia=1.0)
	d.summarize(X)
	d.from_summaries()
	_test_fit_params(d, means, covs)


def test_from_summaries_frozen(X, w, means, covs):
	d = Normal(means, covs, covariance_type='diag', frozen=True)
	d.summarize(X[:4])
	assert_array_almost_equal(d._w_sum, [0.0, 0.0, 0.0])
	assert_array_almost_equal(d._xw_sum, [0.0, 0.0, 0.0])
	assert_array_almost_equal(d._xxw_sum, [0.0, 0.0, 0.0])

	d.from_summaries()
	_test_fit_params(d, means, covs)

	d.summarize(X[4:])
	assert_array_almost_equal(d._w_sum, [0.0, 0.0, 0.0])
	assert_array_almost_equal(d._xw_sum, [0.0, 0.0, 0.0])
	assert_array_almost_equal(d._xxw_sum, [0.0, 0.0, 0.0])

	d.from_summaries()
	_test_fit_params(d, means, covs)

	d = Normal(means, covs, covariance_type='diag', frozen=True)
	d.summarize(X)
	assert_array_almost_equal(d._w_sum, [0.0, 0.0, 0.0])
	assert_array_almost_equal(d._xw_sum, [0.0, 0.0, 0.0])
	assert_array_almost_equal(d._xxw_sum, [0.0, 0.0, 0.0])

	d.from_summaries()
	_test_fit_params(d, means, covs)

	d = Normal(means, covs, covariance_type='diag', frozen=True)
	d.summarize(X, sample_weight=w)
	assert_array_almost_equal(d._w_sum, [0.0, 0.0, 0.0])
	assert_array_almost_equal(d._xw_sum, [0.0, 0.0, 0.0])
	assert_array_almost_equal(d._xxw_sum, [0.0, 0.0, 0.0])

	d.from_summaries()
	_test_fit_params(d, means, covs)

def test_from_summaries_dtypes(X, means):
	X = numpy.array(X)
	X = X.astype(numpy.float32)

	p = numpy.array(means, dtype=numpy.float32)
	d = Normal(p, p, covariance_type='diag')
	d.summarize(X)
	d.from_summaries()
	assert d.means.dtype == torch.float32
	assert d.covs.dtype == torch.float32

	p = numpy.array(means, dtype=numpy.float64)
	d = Normal(p, p, covariance_type='diag')
	d.summarize(X)
	d.from_summaries()
	assert d.means.dtype == torch.float64
	assert d.covs.dtype == torch.float64

	p = numpy.array(means, dtype=numpy.int32)
	d = Normal(p, p, covariance_type='diag')
	d.summarize(X)
	d.from_summaries()
	assert d.means.dtype == torch.int32
	assert d.covs.dtype == torch.int32

	X = numpy.array(X)
	X = X.astype(numpy.float64)

	p = numpy.array(means, dtype=numpy.float64)
	d = Normal(p, p, covariance_type='diag')
	d.summarize(X)
	d.from_summaries()
	assert d.means.dtype == torch.float64
	assert d.covs.dtype == torch.float64


def test_from_summaries_raises():
	assert_raises(AttributeError, Normal().from_summaries)


def test_fit(X, means, covs):
	d = Normal(means, covs, covariance_type='diag')
	d.fit(X)
	assert_raises(AssertionError, assert_array_almost_equal, means, d.means)
	assert_array_almost_equal(d.covs, torch.diag(torch.cov(torch.tensor(X).T, 
		correction=0)))

	for param1, param2 in (means, covs), (None, None):
		d = Normal(param1, param2, covariance_type='diag')
		d.fit(X[:4])
		_test_fit_params(d, [1.525, 1.375, 1.55], [0.931875, 0.626875, 0.5025])	

		d.fit(X[4:])
		_test_fit_params(d, [3.666667, 1.4     , 1.466667], 
			[1.742223, 0.14    , 3.215556])

		d = Normal(param1, param2, covariance_type='diag')
		d.fit(X)
		_test_fit_params(d, [2.442857, 1.385714, 1.514286],
			[2.402449, 0.418367, 1.666939])


def test_fit_weighted(X, w, means, covs):
	for param in means, None:
		d = Normal(means, covs, covariance_type='diag')
		d.fit(X[:4], sample_weight=w[:4])
		_test_fit_params(d, [0.7     , 0.833333, 1.033333],
			[0.08    , 0.802222, 0.142222])
	
		d.fit(X[4:], sample_weight=w[4:])
		_test_fit_params(d, [3.35  , 1.1875, 0.7125],
			[0.857502, 0.088594, 1.551094])

		d = Normal(means, covs, covariance_type='diag')
		d.fit(X, sample_weight=w)
		_test_fit_params(d, [2.627273, 1.090909, 0.8],
			[2.038348, 0.308099, 1.187273])
	
	X = [[1.2, 0.5, 1.1, 1.9],
	     [6.2, 1.1, 2.4, 1.1]] 

	w = [[1.1], [3.5]]

	d = Normal(covariance_type='diag')
	d.fit(X, sample_weight=w)
	_test_fit_params(d, [5.004348, 0.956522, 2.089131, 1.291304],
		[4.548677, 0.065501, 0.30749 , 0.116446])


def test_fit_chain(X):
	d = Normal(covariance_type='diag').fit(X[:4])
	_test_fit_params(d, [1.525, 1.375, 1.55], [0.931875, 0.626875, 0.5025])	

	d.fit(X[4:])
	_test_fit_params(d, [3.666667, 1.4     , 1.466667], 
			[1.742223, 0.14    , 3.215556])

	d = Normal(covariance_type='diag').fit(X)
	_test_fit_params(d, [2.442857, 1.385714, 1.514286],
			[2.402449, 0.418367, 1.666939])


def test_fit_dtypes(X, means):
	X = numpy.array(X)
	X = X.astype(numpy.float32)

	p = numpy.array(means, dtype=numpy.float32)
	d = Normal(p, p, covariance_type='diag')
	d.fit(X)
	assert d.means.dtype == torch.float32
	assert d.covs.dtype == torch.float32

	p = numpy.array(means, dtype=numpy.float64)
	d = Normal(p, p, covariance_type='diag')
	d.fit(X)
	assert d.means.dtype == torch.float64
	assert d.covs.dtype == torch.float64


	p = numpy.array(means, dtype=numpy.int32)
	d = Normal(p, p, covariance_type='diag')
	d.fit(X)
	assert d.means.dtype == torch.int32
	assert d.covs.dtype == torch.int32

	X = numpy.array(X)
	X = X.astype(numpy.float64)

	p = numpy.array(means, dtype=numpy.float64)
	d = Normal(p, p, covariance_type='diag')
	d.fit(X)
	assert d.means.dtype == torch.float64
	assert d.covs.dtype == torch.float64


def test_fit_raises(X, w, means):
	_test_raises(Normal(means, covariance_type='diag'), "fit", X, w=w, 
		min_value=MIN_VALUE, max_value=MAX_VALUE)

	_test_raises(Normal(covariance_type='diag'), "fit", X, w=w, 
		min_value=MIN_VALUE, max_value=MAX_VALUE)

	_test_raises(Normal([VALID_VALUE], covariance_type='diag'), "fit", X, w=w, 
		min_value=MIN_VALUE, max_value=MAX_VALUE)


def test_serialization(X):
	d = Normal(covariance_type='diag').fit(X[:4])
	d.summarize(X[4:])

	means = [1.525, 1.375, 1.55 ]
	covs = [0.931875, 0.626875, 0.5025  ]

	assert_array_almost_equal(d.means, means)
	assert_array_almost_equal(d.covs, covs)

	torch.save(d, ".pytest.torch")
	d2 = torch.load(".pytest.torch")
	os.system("rm .pytest.torch")

	assert_array_almost_equal(d2.means, means)
	assert_array_almost_equal(d2.covs, covs)

	assert_array_almost_equal(d2._w_sum, [3., 3., 3.])
	assert_array_almost_equal(d2._xw_sum, [11. ,  4.2,  4.4])
	assert_array_almost_equal(d2._xxw_sum, [45.56    ,  6.299999, 16.1     ])
	assert_array_almost_equal(d.log_probability(X), d2.log_probability(X))
	

def test_masked_probability(means, covs, X, X_masked):
	X = torch.tensor(numpy.array(X))
	y = [2.004972e-02, 1.346142e-02, 5.173831e-02, 1.438069e-04,
           4.410233e-06, 2.273481e-15, 2.059388e-03]

	d = Normal(means, covs, covariance_type='diag')
	mask = torch.ones_like(X).type(torch.bool)
	X_ = torch.masked.MaskedTensor(X, mask=mask)

	assert_array_almost_equal(y, d.probability(X_)._masked_data)

	y =  [2.798962e-02, 4.825954e-02, 1.000000e+00, 1.438069e-04,
           2.043614e-01, 2.273480e-15, 9.462826e-03]

	assert_array_almost_equal(y, d.probability(X_masked)._masked_data)


def test_masked_log_probability(means, covs, X, X_masked):
	X = torch.tensor(numpy.array(X))
	y = [-3.90954 ,  -4.307928,  -2.961557,  -8.847039 , -12.331583,
           -33.717464,  -6.185347]

	d = Normal(means, covs, covariance_type='diag')
	mask = torch.ones_like(X).type(torch.bool)
	X_ = torch.masked.MaskedTensor(X, mask=mask)

	assert_array_almost_equal(y, d.log_probability(X_)._masked_data)

	y = [-3.575922,  -3.031162,   0.      ,  -8.847039,  -1.587865,
           -33.717464,  -4.660384]

	assert_array_almost_equal(y, d.log_probability(X_masked)._masked_data)


def test_masked_summarize(X, X_masked, w):
	X = torch.tensor(numpy.array(X))
	mask = torch.ones_like(X).type(torch.bool)
	X_ = torch.masked.MaskedTensor(X, mask=mask)

	d = Normal(covariance_type='diag')
	d.summarize(X, sample_weight=w)
	assert_array_almost_equal(d._w_sum, [11.0, 11.0, 11.0])
	assert_array_almost_equal(d._xw_sum, [28.9     , 12.      ,  8.799999])

	d = Normal(covariance_type='diag')
	d.summarize(X_masked)
	assert_array_almost_equal(d._w_sum, [4.0, 5.0, 4.0])
	assert_array_almost_equal(d._xw_sum, [11.200001,  7.3     ,  6.8])


def test_masked_from_summaries(X, X_masked):
	X = torch.tensor(numpy.array(X))
	mask = torch.ones_like(X).type(torch.bool)
	X_ = torch.masked.MaskedTensor(X, mask=mask)

	d = Normal(covariance_type='diag')
	d.summarize(X_)
	d.from_summaries()
	_test_fit_params(d, [2.442857, 1.385714, 1.514286],
		[2.402449, 0.418367, 1.666939])

	d = Normal(covariance_type='diag')
	d.summarize(X_masked)
	d.from_summaries()
	_test_fit_params(d, [2.8 , 1.46, 1.7], [3.124999, 0.5624  , 2.385])


def test_masked_fit(X, X_masked):
	X = torch.tensor(numpy.array(X))
	mask = torch.ones_like(X).type(torch.bool)
	X_ = torch.masked.MaskedTensor(X, mask=mask)

	d = Normal(covariance_type='diag')
	d.fit(X_)
	_test_fit_params(d, [2.442857, 1.385714, 1.514286],
		[2.402449, 0.418367, 1.666939])

	d = Normal(covariance_type='diag')
	d.fit(X_masked)
	_test_fit_params(d, [2.8 , 1.46, 1.7], [3.124999, 0.5624  , 2.385])
	
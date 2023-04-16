# test_normal_full.py
# Contact: Jacob Schreiber <jmschreiber91@gmail.com>

import os
import numpy
import torch
import pytest

from torchegranate.distributions import Normal

from ._utils import _test_initialization_raises_two_parameters
from ._utils import _test_initialization
from ._utils import _test_predictions
from ._utils import _test_efd_from_summaries
from ._utils import _test_raises

from nose.tools import assert_raises
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
def w():
	return [[1], [2], [0], [0], [5], [1], [2]]


@pytest.fixture
def means():
	return [1.2, 1.8, 2.1]


@pytest.fixture
def covs():
	return [
		[0.3, 0.1, 0.0],
		[0.1, 2.1, 0.6],
		[0.0, 0.6, 1.2]
	]


###


def test_initialization():
	d = Normal(covariance_type='full')
	
	_test_initialization(d, None, "means", 0.0, False, None)
	_test_initialization(d, None, "covs", 0.0, False, None)

	assert_raises(AttributeError, getattr, d, "_w_sum")
	assert_raises(AttributeError, getattr, d, "_xw_sum")
	assert_raises(AttributeError, getattr, d, "_xxw_sum")

	assert_raises(AttributeError, getattr, d, "_inv_cov")
	assert_raises(AttributeError, getattr, d, "_inv_cov_dot_mu")
	assert_raises(AttributeError, getattr, d, "_log_det")
	assert_raises(AttributeError, getattr, d, "_log_sigma_sqrt_2pi")


def test_initialization_float(means, covs):
	funcs = (lambda x: x, tuple, 
		lambda x: numpy.array(x, dtype=numpy.float32), 
		lambda x: torch.tensor(x, dtype=torch.float32), 
		lambda x: torch.nn.Parameter(torch.tensor(x), requires_grad=False))

	for func in funcs:
		means_, covs_ = func(means), func(covs)

		_test_initialization(Normal(means_, covs_, covariance_type='full', 
			inertia=0.0, frozen=False), means_, "means", 0.0, False, 
			torch.float32)
		_test_initialization(Normal(means_, covs_, covariance_type='full',
			inertia=0.3, frozen=False), means_, "means", 0.3, False, 
			torch.float32)
		_test_initialization(Normal(means_, covs_, covariance_type='full',
			inertia=1.0, frozen=True), means_, "means", 1.0, True, 
			torch.float32)
		_test_initialization(Normal(means_, covs_, covariance_type='full',
			inertia=1.0, frozen=False), means_, "means", 1.0, False, 
			torch.float32)

		_test_initialization(Normal(means_, covs_, covariance_type='full',
			inertia=0.0, frozen=False), covs_, "covs", 0.0, False, 
			torch.float32)
		_test_initialization(Normal(means_, covs_, covariance_type='full',
			inertia=0.3, frozen=False), covs_, "covs", 0.3, False, 
			torch.float32)
		_test_initialization(Normal(means_, covs_, covariance_type='full',
			inertia=1.0, frozen=True), covs_, "covs", 1.0, True, 
			torch.float32)
		_test_initialization(Normal(means_, covs_, covariance_type='full',
			inertia=1.0, frozen=False), covs_, "covs", 1.0, False, 
			torch.float32)

	means_ = numpy.array(means, dtype=numpy.float64)
	covs_ = numpy.array(covs, dtype=numpy.float64)

	_test_initialization(Normal(means_, covs_, covariance_type='full',
		inertia=0.0, frozen=False), means_, "means", 0.0, False, torch.float64)
	_test_initialization(Normal(means_, covs_, covariance_type='full',
		inertia=0.3, frozen=False), means_, "means", 0.3, False, torch.float64)
	_test_initialization(Normal(means_, covs_, covariance_type='full',
		inertia=1.0, frozen=True), means_, "means", 1.0, True, torch.float64)
	_test_initialization(Normal(means_, covs_, covariance_type='full',
		inertia=1.0, frozen=False), means_, "means", 1.0, False, torch.float64)

	_test_initialization(Normal(means_, covs_, covariance_type='full',
		inertia=0.0, frozen=False), covs_, "covs", 0.0, False, torch.float64)
	_test_initialization(Normal(means_, covs_, covariance_type='full',
		inertia=0.3, frozen=False), covs_, "covs", 0.3, False, torch.float64)
	_test_initialization(Normal(means_, covs_, covariance_type='full',
		inertia=1.0, frozen=True), covs_, "covs", 1.0, True, torch.float64)
	_test_initialization(Normal(means_, covs_, covariance_type='full',
		inertia=1.0, frozen=False), covs_, "covs", 1.0, False, torch.float64)


def test_initialization_raises():
	_test_initialization_raises_two_parameters(Normal, VALID_VALUE, VALID_VALUE,
		min_value1=MIN_VALUE, min_value2=MIN_VALUE)


def test_reset_cache(X):
	d = Normal(covariance_type='full')
	d.summarize(X)
	assert_array_almost_equal(d._w_sum, [7.0, 7.0, 7.0])
	assert_array_almost_equal(d._xw_sum, [17.099998,  9.7     , 10.599999])
	assert_array_almost_equal(d._xxw_sum,
		[[58.59    , 26.98    , 33.940002],
		 [26.98    , 16.369999, 16.38    ],
		 [33.940002, 16.38    , 27.720001]],)

	d._reset_cache()
	assert_array_almost_equal(d._w_sum, [0.0, 0.0, 0.0])
	assert_array_almost_equal(d._xw_sum, [0.0, 0.0, 0.0])
	assert_array_almost_equal(d._xxw_sum,
		[[0., 0., 0.],
		 [0., 0., 0.],
		 [0., 0., 0.]])

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
	d = Normal(covariance_type='full')
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

	d = Normal([1.2], None, covariance_type='full')
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

	d = Normal(None, [[1.2]], covariance_type='full')
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
	assert_array_almost_equal(d.covs,
		[[0., 0., 0.],
		 [0., 0., 0.],
		 [0., 0., 0.]])
	assert_array_almost_equal(d._w_sum, [0.0, 0.0, 0.0])
	assert_array_almost_equal(d._xw_sum, [0.0, 0.0, 0.0])
	assert_array_almost_equal(d._xxw_sum,
		[[0., 0., 0.],
		 [0., 0., 0.],
		 [0., 0., 0.]])	


	d._initialize(2)
	assert d._initialized == True
	assert d.means.shape[0] == 2
	assert d.d == 2
	assert_array_almost_equal(d.means, [0.0, 0.0])
	assert_array_almost_equal(d.covs,
		[[0., 0.],
		 [0., 0.]])
	assert_array_almost_equal(d._w_sum, [0.0, 0.0])
	assert_array_almost_equal(d._xw_sum, [0.0, 0.0])
	assert_array_almost_equal(d._xxw_sum,
		[[0., 0.],
		 [0., 0.]])


	d = Normal([1.2, 9.3], [[1.1, 1.2], [1.2, 1.5]], covariance_type='full')
	assert d._initialized == True
	assert d.d == 2

	d._initialize(3)
	assert d._initialized == True
	assert d.means.shape[0] == 3
	assert d.d == 3
	assert_array_almost_equal(d.means, [0.0, 0.0, 0.0])
	assert_array_almost_equal(d.covs,
		[[0., 0., 0.],
		 [0., 0., 0.],
		 [0., 0., 0.]])

	d = Normal(covariance_type='full')
	d.summarize(X)
	d._initialize(4)
	assert d._initialized == True
	assert d.means.shape[0] == 4
	assert d.d == 4
	assert_array_almost_equal(d._w_sum, [0.0, 0.0, 0.0, 0.0])
	assert_array_almost_equal(d._xw_sum, [0.0, 0.0, 0.0, 0.0])
	assert_array_almost_equal(d._xxw_sum,
		[[0., 0., 0., 0.],
		 [0., 0., 0., 0.],
		 [0., 0., 0., 0.],
		 [0., 0., 0., 0.]])
	

###


@pytest.mark.sample
def test_sample(means, covs):
	torch.manual_seed(0)

	X = Normal(means, covs, covariance_type='full').sample(1)
	assert_array_almost_equal(X, [[2.044038,  1.659515, -0.229191]])

	X = Normal(means, covs, covariance_type='full').sample(5)
	assert_array_almost_equal(X, 
		[[1.5113, 0.3447, 0.2308],
         [1.4209, 3.0784, 1.7213],
         [0.9791, 0.8686, 2.0354],
         [0.7308, 3.2258, 1.4744],
         [1.2672, 1.0083, 2.2415]], 3)


###


def test_probability(X, means, covs):
	m, c = [1.7], [[1.3]]
	x = [[1.0], [2.0], [4.0], [3.7], [1.9]]
	y = [0.289795, 0.337991, 0.045742, 0.075126, 0.344554]

	d1 = Normal(m, c, covariance_type='full')
	d2 = Normal(numpy.array(m, dtype=numpy.float64), 
		numpy.array(c, dtype=numpy.float64), covariance_type='full')

	_test_predictions(x, y, d1.probability(x), torch.float32)
	_test_predictions(x, y, d2.probability(x), torch.float64)

	x = [[1.0, 2.0, 4.0]]
	y = [0.014501]

	d1 = Normal(means, covs, covariance_type='full')
	d2 = Normal(numpy.array(means, dtype=numpy.float64), 
		numpy.array(covs, dtype=numpy.float64), covariance_type='full')

	_test_predictions(x, y, d1.probability(x), torch.float32)
	_test_predictions(x, y, d2.probability(x), torch.float64)

	y = [1.873443e-02, 2.068522e-02, 6.141333e-02, 1.854071e-04,
		   5.782201e-06, 7.198527e-16, 2.817989e-03]

	d1 = Normal(means, covs, covariance_type='full')
	d2 = Normal(numpy.array(means, dtype=numpy.float64),
		numpy.array(covs, dtype=numpy.float64), covariance_type='full')

	_test_predictions(X, y, d1.probability(X), torch.float32)
	_test_predictions(X, y, d2.probability(X), torch.float64)


def test_probability_dtypes(X, means, covs):
	y = Normal(means, covs, covariance_type='full').probability(X)
	assert y.dtype == torch.float32


def test_probability_raises(X, means, covs):
	_test_raises(Normal(means, covs, covariance_type='full'), "probability", X, 
		min_value=MIN_VALUE, max_value=MAX_VALUE)

	_test_raises(Normal([VALID_VALUE], [[VALID_VALUE]], covariance_type='full'), 
		"probability", X, min_value=MIN_VALUE, max_value=MAX_VALUE)


def test_log_probability(X, means, covs):
	m, c = [1.7], [[1.3]]
	x = [[1.0], [2.0], [4.0], [3.0], [2.0]]
	y = [-1.238582, -1.084736, -3.084736, -1.700121, -1.084736]

	x_torch = torch.tensor(numpy.array(x))	
	m_torch = torch.tensor(numpy.array(m))
	c_torch = torch.tensor(numpy.array(c))

	d1 = Normal(m, c, covariance_type='full')
	d2 = Normal(numpy.array(m, dtype=numpy.float64),
		numpy.array(c, dtype=numpy.float64), covariance_type='full')
	d3 = torch.distributions.MultivariateNormal(m_torch, c_torch)

	_test_predictions(x, y, d1.log_probability(x), torch.float32)
	_test_predictions(x, y, d2.log_probability(x), torch.float64)
	_test_predictions(x, d3.log_prob(x_torch), d2.log_probability(x), 
		torch.float64)

	p = [1.7, 2.3, 1.0, 1.7, 4.1]
	x = [[1.0, 2.0, 4.0, 3.0, 2.0]]
	y = [-12.434692]

	c = numpy.eye(5).astype(numpy.float32)
	p_torch = torch.tensor(numpy.array(p))
	x_torch = torch.tensor(numpy.array(x))

	d1 = Normal(p, c, covariance_type='full')
	d2 = Normal(numpy.array(p, dtype=numpy.float64),
		numpy.array(c, dtype=numpy.float64), covariance_type='full')
	d3 = torch.distributions.MultivariateNormal(p_torch, torch.eye(5))

	_test_predictions(x, y, d1.log_probability(x), torch.float32)
	_test_predictions(x, y, d2.log_probability(x), torch.float64)
	_test_predictions(x, d3.log_prob(x_torch), d2.log_probability(x), 
		torch.float64)

	y = [-3.977393,  -3.878336,  -2.790128,  -8.592959, -12.060726,
		-34.867487,  -5.871732]

	m_torch = torch.tensor(numpy.array(means))
	c_torch = torch.tensor(numpy.array(covs))
	x_torch = torch.tensor(numpy.array(X))

	d2 = Normal(numpy.array(means, dtype=numpy.float64),
		numpy.array(covs, dtype=numpy.float64), covariance_type='full')
	d3 = torch.distributions.MultivariateNormal(m_torch, c_torch)

	_test_predictions(X, y, d2.log_probability(X), torch.float64)
	_test_predictions(X, d3.log_prob(x_torch), d2.log_probability(X), 
		torch.float64)


def test_log_probability_dtypes(X, means, covs):
	y = Normal(means, covs, covariance_type='full').log_probability(X)
	assert y.dtype == torch.float32


def test_log_probability_raises(X, means, covs):
	_test_raises(Normal(means, covs, covariance_type='full'), "log_probability",
		X, min_value=MIN_VALUE, max_value=MAX_VALUE)

	_test_raises(Normal([VALID_VALUE], [[VALID_VALUE]], covariance_type='full'), 
		"log_probability", X, min_value=MIN_VALUE, max_value=MAX_VALUE)


###


def test_summarize(X, means, covs):
	for m, c in ((means, covs), (None, None)):
		d = Normal(m, c, covariance_type='full')
		d.summarize(X[:4])
		assert_array_almost_equal(d._w_sum, [4.0, 4.0, 4.0])
		assert_array_almost_equal(d._xw_sum, [6.1, 5.5, 6.2])
		assert_array_almost_equal(d._xxw_sum,
			[[13.03    , 10.459999, 11.1     ],
             [10.459999, 10.07    ,  8.35    ],
             [11.1     ,  8.35    , 11.620001]])

		d.summarize(X[4:])
		assert_array_almost_equal(d._w_sum, [7.0, 7.0, 7.0])
		assert_array_almost_equal(d._xw_sum, [17.1,  9.7, 10.6])
		assert_array_almost_equal(d._xxw_sum, 
			[[58.59    , 26.98    , 33.940002],
             [26.98    , 16.369999, 16.380001],
             [33.940002, 16.380001, 27.720001]])


		d = Normal(m, c, covariance_type='full')
		d.summarize(X)
		assert_array_almost_equal(d._w_sum, [7.0, 7.0, 7.0])
		assert_array_almost_equal(d._xw_sum, [17.099998,  9.7     , 10.599999])
		assert_array_almost_equal(d._xxw_sum,
			[[58.59    , 26.98    , 33.940002],
             [26.98    , 16.369999, 16.38    ],
             [33.940002, 16.38    , 27.720001]])


def test_summarize_weighted(X, w, means, covs):
	d = Normal(means, covs, covariance_type='full')
	d.summarize(X[:4], sample_weight=w[:4])
	assert_array_almost_equal(d._w_sum, [3., 3., 3.])
	assert_array_almost_equal(d._xw_sum, [2.1, 2.5, 3.1])
	assert_array_almost_equal(d._xxw_sum,
		[[1.71    , 2.51    , 1.85    ],
         [2.51    , 4.489999, 1.57    ],
         [1.85    , 1.57    , 3.63    ]])

	d.summarize(X[4:], sample_weight=w[4:])
	assert_array_almost_equal(d._w_sum, [11.0, 11.0, 11.0])
	assert_array_almost_equal(d._xw_sum, [28.9     , 12.      ,  8.799999])
	assert_array_almost_equal(d._xxw_sum,
		[[98.350006, 35.489998, 28.990002],
         [35.489998, 16.48    , 10.93    ],
         [28.990002, 10.93    , 20.1     ]], 5)

	d = Normal(means, covs, covariance_type='full')
	d.summarize(X, sample_weight=w)
	assert_array_almost_equal(d._w_sum, [11.0, 11.0, 11.0])
	assert_array_almost_equal(d._xw_sum, [28.9     , 12.      ,  8.799999])
	assert_array_almost_equal(d._xxw_sum,
		[[98.350006, 35.49    , 28.99    ],
         [35.49    , 16.48    , 10.93    ],
         [28.99    , 10.93    , 20.1     ]], 5)


def test_summarize_weighted_flat(X, w, means, covs):
	w = numpy.array(w)[:,0] 

	d = Normal(means, covs, covariance_type='full')
	d.summarize(X[:4], sample_weight=w[:4])
	assert_array_almost_equal(d._w_sum, [3., 3., 3.])
	assert_array_almost_equal(d._xw_sum, [2.1, 2.5, 3.1])
	assert_array_almost_equal(d._xxw_sum,
		[[1.71    , 2.51    , 1.85    ],
         [2.51    , 4.489999, 1.57    ],
         [1.85    , 1.57    , 3.63    ]])

	d.summarize(X[4:], sample_weight=w[4:])
	assert_array_almost_equal(d._w_sum, [11.0, 11.0, 11.0])
	assert_array_almost_equal(d._xw_sum, [28.9     , 12.      ,  8.799999])
	assert_array_almost_equal(d._xxw_sum,
		[[98.350006, 35.489998, 28.990002],
         [35.489998, 16.48    , 10.93    ],
         [28.990002, 10.93    , 20.1     ]], 5)

	d = Normal(means, covs, covariance_type='full')
	d.summarize(X, sample_weight=w)
	assert_array_almost_equal(d._w_sum, [11.0, 11.0, 11.0])
	assert_array_almost_equal(d._xw_sum, [28.9     , 12.      ,  8.799999])
	assert_array_almost_equal(d._xxw_sum,
		[[98.350006, 35.49    , 28.99    ],
         [35.49    , 16.48    , 10.93    ],
         [28.99    , 10.93    , 20.1     ]], 5)

def test_summarize_weighted_2d(X):
	d = Normal(covariance_type='full')
	d.summarize(X[:4], sample_weight=X[:4])
	assert_array_almost_equal(d._w_sum, [6.1, 5.5, 6.2])
	assert_array_almost_equal(d._xw_sum, [13.03    , 10.069999, 11.62])
	assert_array_almost_equal(d._xxw_sum,
		[[33.990997, 24.928   , 26.383999],
         [20.235998, 19.860996, 14.620998],
         [22.9     , 16.351   , 23.618   ]], 5)

	d.summarize(X[4:], sample_weight=X[4:])
	assert_array_almost_equal(d._w_sum, [17.1,  9.7, 10.6])
	assert_array_almost_equal(d._xw_sum, [58.59    , 16.369999, 27.720001])
	assert_array_almost_equal(d._xxw_sum,
		[[241.40701,  98.18401, 146.97601],
         [ 46.848  ,  29.917  ,  29.53   ],
         [109.62801,  46.854  ,  87.646  ]], 5)

	d = Normal(covariance_type='full')
	d.summarize(X, sample_weight=X)
	assert_array_almost_equal(d._w_sum, [17.099998,  9.7     , 10.599999])
	assert_array_almost_equal(d._xw_sum, [58.59    , 16.369999, 27.720001])
	assert_array_almost_equal(d._xxw_sum,
		[[241.40701,  98.18401, 146.97601],
         [ 46.848  ,  29.917  ,  29.53   ],
         [109.62801,  46.854  ,  87.646  ]], 5)


def test_summarize_dtypes(X, w, means, covs):
	X = numpy.array(X)
	X = X.astype(numpy.float32)

	d = Normal(means, covs, covariance_type='full')
	assert d._xw_sum.dtype == torch.float32
	d.summarize(X)
	assert d._xw_sum.dtype == torch.float32

	X = X.astype(numpy.float64)
	d = Normal(means, covs, covariance_type='full').type(torch.float64)
	assert d._xw_sum.dtype == torch.float64
	d.summarize(X)
	assert d._xw_sum.dtype == torch.float64

	X = X.astype(numpy.int32)
	d = Normal(means, covs, covariance_type='full')
	assert d._xw_sum.dtype == torch.float32
	d.summarize(X)
	assert d._xw_sum.dtype == torch.float32

	X = X.astype(numpy.int64)
	d = Normal(means, covs, covariance_type='full')
	assert d._xw_sum.dtype == torch.float32
	d.summarize(X)
	assert d._xw_sum.dtype == torch.float32


def test_summarize_raises(X, w, means, covs):
	_test_raises(Normal(means, covs, covariance_type='full'), "summarize", X, 
		w=w, min_value=MIN_VALUE, max_value=MAX_VALUE)

	_test_raises(Normal(covariance_type='full'), "summarize", X, w=w, 
		min_value=MIN_VALUE, max_value=MAX_VALUE)

	_test_raises(Normal([VALID_VALUE], [[VALID_VALUE]], covariance_type='full'), 
		"summarize", X, w=w, min_value=MIN_VALUE, max_value=MAX_VALUE)


def _test_fit_params(d, means, covs):
	assert_array_almost_equal(d.means, means)
	assert_array_almost_equal(d.covs, covs)

	assert_array_almost_equal(d._w_sum, numpy.zeros(d.d))
	assert_array_almost_equal(d._xw_sum, numpy.zeros(d.d))
	assert_array_almost_equal(d._xxw_sum, numpy.zeros((d.d, d.d)))

	assert_raises(AttributeError, getattr, d, "_log_sigma_sqrt_2pi")
	assert_raises(AttributeError, getattr, d, "_inv_two_sigma")


def test_from_summaries(X, means, covs):
	d = Normal(means, covs, covariance_type='full')
	d.summarize(X)
	d.from_summaries()
	assert_raises(AssertionError, assert_array_almost_equal, means, d.means)
	assert_array_almost_equal(d.covs, torch.cov(torch.tensor(X).T, 
		correction=0))

	for param1, param2 in (means, covs), (None, None):
		d = Normal(param1, param2, covariance_type='full')
		d.summarize(X[:4])
		d.from_summaries()
		_test_fit_params(d, [1.525, 1.375, 1.55],
			[[ 0.931875,  0.518125,  0.41125 ],
             [ 0.518125,  0.626875, -0.04375 ],		
             [ 0.41125 , -0.04375 ,  0.5025  ]])	

		d.summarize(X[4:])
		d.from_summaries()
		_test_fit_params(d, [3.666667, 1.4     , 1.466667], 
			[[1.742223, 0.373333, 2.235555],
             [0.373333, 0.14    , 0.623333],
             [2.235555, 0.623333, 3.215556]])

		d = Normal(param1, param2, covariance_type='full')
		d.summarize(X[:4])
		d.summarize(X[4:])
		d.from_summaries()
		_test_fit_params(d, [2.442857, 1.385714, 1.514286],
			[[2.402449, 0.469184, 1.149388],
             [0.469184, 0.418367, 0.241633],
             [1.149388, 0.241633, 1.666939]])

		d = Normal(param1, param2, covariance_type='full')
		d.summarize(X)
		d.from_summaries()
		_test_fit_params(d, [2.442857, 1.385714, 1.514286],
			[[2.40245 , 0.469184, 1.149389],
             [0.469184, 0.418367, 0.241633],
             [1.149389, 0.241633, 1.666939]])


def test_from_summaries_weighted(X, w, means, covs):
	for param in means, None:
		d = Normal(means, covs, covariance_type='full')
		d.summarize(X, sample_weight=w)
		d.from_summaries()
		_test_fit_params(d, [2.627273, 1.090909, 0.8],
			[[2.038348, 0.360248, 0.533636],
             [0.360248, 0.308099, 0.120909],
             [0.533636, 0.120909, 1.187273]])
	

def test_from_summaries_null():
	d = Normal([1., 2.], [[1., 0.], [0., 2.]], covariance_type='full')
	d.from_summaries()
	assert d.means[0] != 1 and d.means[1] != 2
	assert d.covs[0, 0] != 1 and d.covs[1, 1] != 2 
	assert_array_almost_equal(d._w_sum, [0.0, 0.0])
	assert_array_almost_equal(d._xw_sum, [0.0, 0.0])
	assert_array_almost_equal(d._xxw_sum, [[0.0, 0.0], [0.0, 0.0]])

	d = Normal([1., 2.], [[1., 0.], [0., 2.]], covariance_type='full', inertia=0.5)
	d.from_summaries()
	assert d.means[0] != 1 and d.means[1] != 2
	assert d.covs[0, 0] != 1 and d.covs[1, 1] != 2 
	assert_array_almost_equal(d._w_sum, [0.0, 0.0])
	assert_array_almost_equal(d._xw_sum, [0.0, 0.0])
	assert_array_almost_equal(d._xxw_sum, [[0.0, 0.0], [0.0, 0.0]])


	d = Normal([1., 2.], [[1., 0.], [0., 2.]], covariance_type='full', inertia=0.5, frozen=True)
	d.from_summaries()
	_test_fit_params(d, [1, 2], [[1., 0.], [0., 2.]])


def test_from_summaries_inertia(X, w, means, covs):
	d = Normal(means, covs, covariance_type='full', inertia=0.3)
	d.summarize(X[:4])
	d.from_summaries()
	_test_fit_params(d, [1.4275, 1.5025, 1.715],
		[[0.742312, 0.392687, 0.287875],
         [0.392687, 1.068812, 0.149375],
         [0.287875, 0.149375, 0.71175 ]])

	d.summarize(X[4:])
	d.from_summaries()
	_test_fit_params(d, [2.994917, 1.43075 , 1.541167],
		[[1.44225 , 0.37914 , 1.651251],
         [0.37914 , 0.418644, 0.481146],
         [1.651251, 0.481146, 2.464414]])

	d = Normal(means, covs, covariance_type='full', inertia=0.3)
	d.summarize(X)
	d.from_summaries()
	_test_fit_params(d, [2.07, 1.51, 1.69],
		[[1.771715, 0.358429, 0.804572],
         [0.358429, 0.922857, 0.349143],
         [0.804572, 0.349143, 1.526857]])


def test_from_summaries_weighted_inertia(X, w, means, covs):
	d = Normal(means, covs, covariance_type='full', inertia=0.3)
	d.summarize(X, sample_weight=w)
	d.from_summaries()
	_test_fit_params(d, [2.199091, 1.303636, 1.19],
		[[1.516843, 0.282174, 0.373545],
         [0.282174, 0.845669, 0.264636],
         [0.373545, 0.264636, 1.191091]])

	d = Normal(means, covs, covariance_type='full', inertia=1.0)
	d.summarize(X[:4])
	d.from_summaries()
	_test_fit_params(d, means, covs)

	d.summarize(X[4:])
	d.from_summaries()
	_test_fit_params(d, means, covs)

	d = Normal(means, covs, covariance_type='full', inertia=1.0)
	d.summarize(X)
	d.from_summaries()
	_test_fit_params(d, means, covs)


def test_from_summaries_frozen(X, w, means, covs):
	d = Normal(means, covs, covariance_type='full', frozen=True)
	d.summarize(X[:4])
	assert_array_almost_equal(d._w_sum, [0.0, 0.0, 0.0])
	assert_array_almost_equal(d._xw_sum, [0.0, 0.0, 0.0])
	assert_array_almost_equal(d._xxw_sum,
		[[0., 0., 0.],
         [0., 0., 0.],
         [0., 0., 0.]])

	d.from_summaries()
	_test_fit_params(d, means, covs)

	d.summarize(X[4:])
	assert_array_almost_equal(d._w_sum, [0.0, 0.0, 0.0])
	assert_array_almost_equal(d._xw_sum, [0.0, 0.0, 0.0])
	assert_array_almost_equal(d._xxw_sum,
		[[0., 0., 0.],
         [0., 0., 0.],
         [0., 0., 0.]])

	d.from_summaries()
	_test_fit_params(d, means, covs)

	d = Normal(means, covs, covariance_type='full', frozen=True)
	d.summarize(X)
	assert_array_almost_equal(d._w_sum, [0.0, 0.0, 0.0])
	assert_array_almost_equal(d._xw_sum, [0.0, 0.0, 0.0])
	assert_array_almost_equal(d._xxw_sum,
		[[0., 0., 0.],
         [0., 0., 0.],
         [0., 0., 0.]])

	d.from_summaries()
	_test_fit_params(d, means, covs)

	d = Normal(means, covs, covariance_type='full', frozen=True)
	d.summarize(X, sample_weight=w)
	assert_array_almost_equal(d._w_sum, [0.0, 0.0, 0.0])
	assert_array_almost_equal(d._xw_sum, [0.0, 0.0, 0.0])
	assert_array_almost_equal(d._xxw_sum,
		[[0., 0., 0.],
         [0., 0., 0.],
         [0., 0., 0.]])

	d.from_summaries()
	_test_fit_params(d, means, covs)

def test_from_summaries_dtypes(X, means,):
	X = numpy.array(X)
	X = X.astype(numpy.float32)

	p = numpy.array(means, dtype=numpy.float32)
	d = Normal(p, [p, p, p], covariance_type='full')
	d.summarize(X)
	d.from_summaries()
	assert d.means.dtype == torch.float32
	assert d.covs.dtype == torch.float32

	p = numpy.array(means, dtype=numpy.float64)
	d = Normal(p, [p, p, p], covariance_type='full')
	d.summarize(X)
	d.from_summaries()
	assert d.means.dtype == torch.float64
	assert d.covs.dtype == torch.float64


def test_from_summaries_raises():
	assert_raises(AttributeError, Normal().from_summaries)


def test_fit(X, means, covs):
	d = Normal(means, covs, covariance_type='full')
	d.fit(X)
	assert_raises(AssertionError, assert_array_almost_equal, means, d.means)
	assert_array_almost_equal(d.covs, torch.cov(torch.tensor(X).T, 
		correction=0))

	for param1, param2 in (means, covs), (None, None):
		d = Normal(param1, param2, covariance_type='full')
		d.fit(X[:4])
		_test_fit_params(d, [1.525, 1.375, 1.55],
			[[ 0.931875,  0.518125,  0.41125 ],
             [ 0.518125,  0.626875, -0.04375 ],		
             [ 0.41125 , -0.04375 ,  0.5025  ]])	

		d.fit(X[4:])
		_test_fit_params(d, [3.666667, 1.4     , 1.466667], 
			[[1.742223, 0.373333, 2.235555],
             [0.373333, 0.14    , 0.623333],
             [2.235555, 0.623333, 3.215556]])

		d = Normal(param1, param2, covariance_type='full')
		d.fit(X)
		_test_fit_params(d, [2.442857, 1.385714, 1.514286],
			[[2.40245 , 0.469184, 1.149389],
             [0.469184, 0.418367, 0.241633],
             [1.149389, 0.241633, 1.666939]])


def test_fit_weighted(X, w, means, covs):
	for param in means, None:
		d = Normal(means, covs, covariance_type='full')
		d.fit(X, sample_weight=w)
		_test_fit_params(d, [2.627273, 1.090909, 0.8],
			[[2.038348, 0.360248, 0.533636],
             [0.360248, 0.308099, 0.120909],
             [0.533636, 0.120909, 1.187273]])


def test_fit_chain(X):
	d = Normal(covariance_type='full').fit(X[:4])
	_test_fit_params(d, [1.525, 1.375, 1.55],
		[[ 0.931875,  0.518125,  0.41125 ],
         [ 0.518125,  0.626875, -0.04375 ],
         [ 0.41125 , -0.04375 ,  0.5025  ]])	

	d.fit(X[4:])
	_test_fit_params(d, [3.666667, 1.4     , 1.466667], 
		[[1.742223, 0.373333, 2.235555],
         [0.373333, 0.14    , 0.623333],
         [2.235555, 0.623333, 3.215556]])

	d = Normal(covariance_type='full').fit(X)
	_test_fit_params(d, [2.442857, 1.385714, 1.514286],
		[[2.40245 , 0.469184, 1.149389],
         [0.469184, 0.418367, 0.241633],
         [1.149389, 0.241633, 1.666939]])


def test_fit_dtypes(X, means):
	X = numpy.array(X)
	X = X.astype(numpy.float32)

	p = numpy.array(means, dtype=numpy.float32)
	d = Normal(p, [p, p, p], covariance_type='full')
	d.fit(X)
	assert d.means.dtype == torch.float32
	assert d.covs.dtype == torch.float32

	p = numpy.array(means, dtype=numpy.float64)
	d = Normal(p, [p, p, p], covariance_type='full')
	d.fit(X)
	assert d.means.dtype == torch.float64
	assert d.covs.dtype == torch.float64


def test_fit_raises(X, w, means):
	_test_raises(Normal(means, covariance_type='full'), "fit", X, w=w, 
		min_value=MIN_VALUE, max_value=MAX_VALUE)

	_test_raises(Normal(covariance_type='full'), "fit", X, w=w, 
		min_value=MIN_VALUE, max_value=MAX_VALUE)

	_test_raises(Normal([VALID_VALUE], covariance_type='full'), "fit", X, w=w, 
		min_value=MIN_VALUE, max_value=MAX_VALUE)


def test_serialization(X):
	d = Normal(covariance_type='full').fit(X[:4])
	d.summarize(X[4:])

	means = [1.525, 1.375, 1.55 ]
	covs = [[ 0.931875,  0.518125,  0.41125 ],
            [ 0.518125,  0.626875, -0.04375 ],
            [ 0.41125 , -0.04375 ,  0.5025  ]]

	assert_array_almost_equal(d.means, means)
	assert_array_almost_equal(d.covs, covs)

	torch.save(d, ".pytest.torch")
	d2 = torch.load(".pytest.torch")
	os.system("rm .pytest.torch")

	assert_array_almost_equal(d2.means, means)
	assert_array_almost_equal(d2.covs, covs)

	assert_array_almost_equal(d2._w_sum, [3., 3., 3.])
	assert_array_almost_equal(d2._xw_sum, [11. ,  4.2,  4.4])
	assert_array_almost_equal(d2._xxw_sum,
		[[45.56, 16.52, 22.84],
         [16.52,  6.3 ,  8.03],
         [22.84,  8.03, 16.1 ]])
	assert_array_almost_equal(d.log_probability(X), d2.log_probability(X))
	
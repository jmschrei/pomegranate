# test_ConditionalCategorical.py
# Contact: Jacob Schreiber <jmschreiber91@gmail.com>

import numpy
import torch
import pytest

from torchegranate.distributions import ConditionalCategorical

from ._utils import _test_initialization_raises_one_parameter
from ._utils import _test_initialization
from ._utils import _test_predictions
from ._utils import _test_raises

from nose.tools import assert_raises
from numpy.testing import assert_array_almost_equal


MIN_VALUE = 0
MAX_VALUE = 2
VALID_VALUE = numpy.array([1, 2, 0])


@pytest.fixture
def X():
	return [[[1, 2, 0],
        [0, 1, 0]],
       [[1, 2, 1],
        [0, 0, 1]],
       [[1, 2, 0],
        [1, 1, 0]],
       [[1, 2, 0],
        [0, 0, 0]],
       [[1, 1, 0],
        [0, 1, 0]],
       [[1, 1, 1],
        [1, 0, 1]],
       [[0, 1, 0],
        [0, 0, 0]]] 


@pytest.fixture
def w():
	return [[1.1], [2.8], [0], [0], [5.5], [1.8], [2.3]]



@pytest.fixture
def probs():
	return [[[0.25, 0.75],
			 [0.32, 0.68]],

			[[0.1, 0.9],
			 [0.3, 0.7],
			 [0.24, 0.76]],

			[[0.35, 0.65],
			 [0.90, 0.10]]]


###


def _test_initialization(d, x, inertia, frozen, dtype):
	assert d.inertia == inertia
	assert d.frozen == frozen
	
	if d._initialized:
		assert len(d.probs) == len(x)

		if isinstance(x, torch.Tensor):
			assert d.probs[0].shape == x[0].shape
			assert d.probs[0].dtype == x[0].dtype

		for i in range(len(d.probs)):
			assert_array_almost_equal(d.probs[i], x[i])


def test_initialization():
	d = ConditionalCategorical()
	_test_initialization(d, None, 0.0, False, None)
	assert_raises(AttributeError, getattr, d, "_w_sum")
	assert_raises(AttributeError, getattr, d, "_xw_sum")
	assert_raises(AttributeError, getattr, d, "_log_probs")

def test_initialization_float(probs):
	funcs = (
		lambda x: x, 
		lambda x: [tuple(x_) for x_ in x], 
		lambda x: [numpy.array(x_, dtype=numpy.float32) for x_ in x], 
		lambda x: [torch.tensor(x_, dtype=torch.float32, requires_grad=False) for x_ in x],
		lambda x: [torch.nn.Parameter(torch.tensor(x_), requires_grad=False) for x_ in x]
	)

	for func in funcs:
		y = func(probs)
		_test_initialization(ConditionalCategorical(y, inertia=0.0, frozen=False), 
			y, 0.0, False, torch.float32)
		_test_initialization(ConditionalCategorical(y, inertia=0.3, frozen=False), 
			y, 0.3, False, torch.float32)
		_test_initialization(ConditionalCategorical(y, inertia=1.0, frozen=True), 
			y, 1.0, True, torch.float32)
		_test_initialization(ConditionalCategorical(y, inertia=1.0, frozen=False), 
			y, 1.0, False, torch.float32)

	x = [numpy.array(prob, dtype=numpy.float64) for prob in probs]
	_test_initialization(ConditionalCategorical(x, inertia=0.0, frozen=False), 
		x, 0.0, False, torch.float64)
	_test_initialization(ConditionalCategorical(x, inertia=0.3, frozen=False), 
		x, 0.3, False, torch.float64)
	_test_initialization(ConditionalCategorical(x, inertia=1.0, frozen=True), 
		x, 1.0, True, torch.float64)
	_test_initialization(ConditionalCategorical(x, inertia=1.0, frozen=False), 
		x, 1.0, False, torch.float64)


def test_initialization_raises(probs):	
	assert_raises(TypeError, ConditionalCategorical, 0.3)
	assert_raises(ValueError, ConditionalCategorical, probs, inertia=-0.4)
	assert_raises(ValueError, ConditionalCategorical, probs, inertia=1.2)
	assert_raises(ValueError, ConditionalCategorical, probs, inertia=1.2, 
		frozen="true")
	assert_raises(ValueError, ConditionalCategorical, probs, inertia=1.2, 
		frozen=3)
	
	assert_raises(ValueError, ConditionalCategorical, inertia=-0.4)
	assert_raises(ValueError, ConditionalCategorical, inertia=1.2)
	assert_raises(ValueError, ConditionalCategorical, inertia=1.2, frozen="true")
	assert_raises(ValueError, ConditionalCategorical, inertia=1.2, frozen=3)

	#assert_raises(ValueError, ConditionalCategorical, numpy.array(probs)+0.001) FIXME
	#assert_raises(ValueError, ConditionalCategorical, numpy.array(probs)-0.001) FIXME

	p = [numpy.array(prob) for prob in probs]
	p[0][0, 0] = -0.03
	assert_raises(ValueError, ConditionalCategorical, p)

	p = [numpy.array(prob) for prob in probs]
	p[0][0, 0] = 1.03
	assert_raises(ValueError, ConditionalCategorical, p)


def test_reset_cache(X):
	d = ConditionalCategorical()
	d.summarize(X)
	assert_array_almost_equal(d._w_sum[0], [1., 6.])
	assert_array_almost_equal(d._xw_sum[0], [[1., 0.], [4., 2.]])

	assert_array_almost_equal(d._w_sum[1], [0., 3., 4.])
	assert_array_almost_equal(d._xw_sum[1], [[0., 0.], [2., 1.], [2., 2.]])

	d._reset_cache()
	assert_array_almost_equal(d._w_sum[0], [0., 0.])
	assert_array_almost_equal(d._xw_sum[0], [[0., 0.], [0., 0.]])

	assert_array_almost_equal(d._w_sum[1], [0., 0., 0.])
	assert_array_almost_equal(d._xw_sum[1], [[0., 0.], [0., 0.], [0., 0.]])

	d = ConditionalCategorical()
	assert_raises(AttributeError, getattr, d, "_w_sum")
	assert_raises(AttributeError, getattr, d, "_xw_sum")
	assert_raises(AttributeError, getattr, d, "_log_probs")

	d._reset_cache()
	assert_raises(AttributeError, getattr, d, "_w_sum")
	assert_raises(AttributeError, getattr, d, "_xw_sum")
	assert_raises(AttributeError, getattr, d, "_log_probs")


def test_initialize(X, probs):
	d = ConditionalCategorical()
	assert d.d is None
	assert d.probs is None
	assert d._initialized == False
	assert_raises(AttributeError, getattr, d, "_w_sum")
	assert_raises(AttributeError, getattr, d, "_xw_sum")
	assert_raises(AttributeError, getattr, d, "_log_probs")

	d._initialize(2, [(2, 2), (2, 3)])
	assert d._initialized == True
	assert d.probs[0].shape == (2, 2)
	assert d.probs[1].shape == (2, 3)
	assert d.d == 2

	assert_array_almost_equal(d.probs[0], [[0., 0.], [0., 0.]])
	assert_array_almost_equal(d._w_sum[0], [0., 0.])
	assert_array_almost_equal(d._xw_sum[0], [[0., 0.], [0., 0.]])

	assert_array_almost_equal(d.probs[1], [[0., 0., 0.], [0., 0., 0.]])
	assert_array_almost_equal(d._w_sum[1], [0., 0.])
	assert_array_almost_equal(d._xw_sum[1], [[0., 0., 0.], [0., 0., 0.]])


	d._initialize(1, [(3, 2, 3)])
	assert d._initialized == True
	assert d.probs[0].shape == (3, 2, 3)
	assert d.d == 1

	assert_array_almost_equal(d.probs[0], 
		[[[0., 0., 0.],
          [0., 0., 0.]],
         [[0., 0., 0.],
          [0., 0., 0.]],
         [[0., 0., 0.],
          [0., 0., 0.]]])
	assert_array_almost_equal(d._w_sum[0], [[0., 0.], [0., 0.], [0., 0.]])
	assert_array_almost_equal(d._xw_sum[0],
		[[[0., 0., 0.],
          [0., 0., 0.]],
         [[0., 0., 0.],
          [0., 0., 0.]],
         [[0., 0., 0.],
          [0., 0., 0.]]])	


	d = ConditionalCategorical(probs)
	assert d._initialized == True
	assert d.d == 3
	assert d.n_categories == [(2, 2), (3, 2), (2, 2)]

	d._initialize(3, [(2, 2), (3, 2), (2, 2)])
	d.summarize(X)
	assert d._initialized == True
	assert d.probs[0].shape == (2, 2)
	assert d.probs[1].shape == (3, 2)
	assert d.probs[2].shape == (2, 2)
	assert d.d == 3

	assert_array_almost_equal(d._w_sum[0], [1., 6.])
	assert_array_almost_equal(d._xw_sum[0], [[1., 0.], [4., 2.]])

	assert_array_almost_equal(d._w_sum[1], [0., 3., 4.])
	assert_array_almost_equal(d._xw_sum[1], [[0., 0.], [2., 1.], [2., 2.]])

	assert_array_almost_equal(d._w_sum[2], [5., 2.])
	assert_array_almost_equal(d._xw_sum[2], [[5., 0.], [0., 2.]])


###


@pytest.mark.sample
def test_sample(probs, X):
	torch.manual_seed(0)

	X = ConditionalCategorical(probs).sample(1, [[[0, 1, 0]]])
	assert_array_almost_equal(X, [[1, 0, 1]])

	x = [
		[[0, 1, 0]],
		[[1, 1, 0]],
		[[0, 0, 0]],
		[[0, 0, 0]],
		[[1, 0, 1]]
	]

	X = ConditionalCategorical(probs).sample(5, x)
	assert_array_almost_equal(X, 
		[[1, 1, 1],
         [0, 1, 1],
         [1, 1, 0],
         [0, 1, 1],
         [1, 1, 0]], 3)


###


def test_probability(X, probs):
	y = [0.08512, 0.00768, 0.18088, 0.02688, 0.0784 , 0.0204 , 0.02625]

	d1 = ConditionalCategorical(probs)
	d2 = ConditionalCategorical([numpy.array(prob, dtype=numpy.float64) for prob in probs])

	_test_predictions(X, y, d1.probability(X), torch.float32)
	_test_predictions(X, y, d2.probability(X), torch.float64)


def test_probability_dtypes(X, probs):	
	y = ConditionalCategorical(probs).probability(X)
	assert y.dtype == torch.float32

	p = [numpy.array(prob, dtype=numpy.float64) for prob in probs]
	y = ConditionalCategorical(p).probability(X)
	assert y.dtype == torch.float64


def test_probability_raises(X, probs):
	_test_raises(ConditionalCategorical(probs), "probability", X, min_value=MIN_VALUE, 
		max_value=MAX_VALUE)

	_test_raises(ConditionalCategorical(probs), "probability", X, 
		min_value=MIN_VALUE, max_value=MAX_VALUE)


def test_log_probability(X, probs):
	y = [-2.463693, -4.869136, -1.709921, -3.616373, -2.545931, -3.89222 ,
           -3.640089]

	d1 = ConditionalCategorical(probs)
	d2 = ConditionalCategorical([numpy.array(prob, dtype=numpy.float64) for prob in probs])

	_test_predictions(X, y, d1.log_probability(X), torch.float32)
	_test_predictions(X, y, d2.log_probability(X), torch.float64)


def test_log_probability_dtypes(X, probs):
	y = ConditionalCategorical(probs).log_probability(X)
	assert y.dtype == torch.float32

	p = [numpy.array(prob, dtype=numpy.float64) for prob in probs]
	y = ConditionalCategorical(p).log_probability(X)
	assert y.dtype == torch.float64


def test_log_probability_raises(X, probs):
	_test_raises(ConditionalCategorical(probs), "log_probability", X, 
		min_value=MIN_VALUE, max_value=MAX_VALUE)

	_test_raises(ConditionalCategorical(probs), "log_probability", X, 
		min_value=MIN_VALUE, max_value=MAX_VALUE)


###


def test_summarize(X, probs):
	for probs_ in probs, None:
		d = ConditionalCategorical(probs_)
		d.summarize(X[:4])
		assert_array_almost_equal(d._w_sum[0], [0., 4.])
		assert_array_almost_equal(d._w_sum[1], [0., 0., 4.])
		assert_array_almost_equal(d._w_sum[2], [3., 1.])

		assert_array_almost_equal(d._xw_sum[0], [[0., 0.], [3., 1.]])
		assert_array_almost_equal(d._xw_sum[1], [[0., 0.], [0., 0.], [2., 2.]])
		assert_array_almost_equal(d._xw_sum[2], [[3., 0.], [0., 1.]])

		d.summarize(X[4:])
		assert_array_almost_equal(d._w_sum[0], [1., 6.])
		assert_array_almost_equal(d._w_sum[1], [0., 3., 4.])
		assert_array_almost_equal(d._w_sum[2], [5., 2.])

		assert_array_almost_equal(d._xw_sum[0], [[1., 0.], [4., 2.]])
		assert_array_almost_equal(d._xw_sum[1], [[0., 0.], [2., 1.], [2., 2.]])
		assert_array_almost_equal(d._xw_sum[2], [[5., 0.], [0., 2.]])


		d = ConditionalCategorical(probs_)
		d.summarize(X)
		assert_array_almost_equal(d._w_sum[0], [1., 6.])
		assert_array_almost_equal(d._w_sum[1], [0., 3., 4.])
		assert_array_almost_equal(d._w_sum[2], [5., 2.])

		assert_array_almost_equal(d._xw_sum[0], [[1., 0.], [4., 2.]])
		assert_array_almost_equal(d._xw_sum[1], [[0., 0.], [2., 1.], [2., 2.]])
		assert_array_almost_equal(d._xw_sum[2], [[5., 0.], [0., 2.]])


def test_summarize_weighted(X, w, probs):
	for probs_ in probs, None:
		d = ConditionalCategorical(probs_)
		d.summarize(X[:4], sample_weight=w[:4])
		assert_array_almost_equal(d._w_sum[0], [0.0, 3.9])
		assert_array_almost_equal(d._w_sum[1], [0.0, 0.0, 3.9])
		assert_array_almost_equal(d._w_sum[2], [1.1, 2.8])

		assert_array_almost_equal(d._xw_sum[0], [[0.0, 0.0], [3.9, 0.0]])
		assert_array_almost_equal(d._xw_sum[1], [[0.0, 0.0], [0.0, 0.0], [2.8, 1.1]])
		assert_array_almost_equal(d._xw_sum[2], [[1.1, 0.0], [0.0, 2.8]])

		d.summarize(X[4:], sample_weight=w[4:])
		assert_array_almost_equal(d._w_sum[0], [2.3, 11.2])
		assert_array_almost_equal(d._w_sum[1], [0.0, 9.6, 3.9])
		assert_array_almost_equal(d._w_sum[2], [8.9, 4.6])

		assert_array_almost_equal(d._xw_sum[0], [[2.3, 0.0], [9.4, 1.8]])
		assert_array_almost_equal(d._xw_sum[1], [[0.0, 0.0], [4.1, 5.5], [2.8, 1.1]])
		assert_array_almost_equal(d._xw_sum[2], [[8.9, 0.0], [0.0, 4.6]])


		d = ConditionalCategorical(probs_)
		d.summarize(X, sample_weight=w)
		assert_array_almost_equal(d._w_sum[0], [2.3, 11.2])
		assert_array_almost_equal(d._w_sum[1], [0.0, 9.6, 3.9])
		assert_array_almost_equal(d._w_sum[2], [8.9, 4.6])

		assert_array_almost_equal(d._xw_sum[0], [[2.3, 0.0], [9.4, 1.8]])
		assert_array_almost_equal(d._xw_sum[1], [[0.0, 0.0], [4.1, 5.5], [2.8, 1.1]])
		assert_array_almost_equal(d._xw_sum[2], [[8.9, 0.0], [0.0, 4.6]])


def test_summarize_weighted_flat(X, w, probs):
	w = numpy.array(w)[:,0] 

	for probs_ in probs, None:
		d = ConditionalCategorical(probs_)
		d.summarize(X[:4], sample_weight=w[:4])
		assert_array_almost_equal(d._w_sum[0], [0.0, 3.9])
		assert_array_almost_equal(d._w_sum[1], [0.0, 0.0, 3.9])
		assert_array_almost_equal(d._w_sum[2], [1.1, 2.8])

		assert_array_almost_equal(d._xw_sum[0], [[0.0, 0.0], [3.9, 0.0]])
		assert_array_almost_equal(d._xw_sum[1], [[0.0, 0.0], [0.0, 0.0], [2.8, 1.1]])
		assert_array_almost_equal(d._xw_sum[2], [[1.1, 0.0], [0.0, 2.8]])

		d.summarize(X[4:], sample_weight=w[4:])
		assert_array_almost_equal(d._w_sum[0], [2.3, 11.2])
		assert_array_almost_equal(d._w_sum[1], [0.0, 9.6, 3.9])
		assert_array_almost_equal(d._w_sum[2], [8.9, 4.6])

		assert_array_almost_equal(d._xw_sum[0], [[2.3, 0.0], [9.4, 1.8]])
		assert_array_almost_equal(d._xw_sum[1], [[0.0, 0.0], [4.1, 5.5], [2.8, 1.1]])
		assert_array_almost_equal(d._xw_sum[2], [[8.9, 0.0], [0.0, 4.6]])


		d = ConditionalCategorical(probs_)
		d.summarize(X, sample_weight=w)
		assert_array_almost_equal(d._w_sum[0], [2.3, 11.2])
		assert_array_almost_equal(d._w_sum[1], [0.0, 9.6, 3.9])
		assert_array_almost_equal(d._w_sum[2], [8.9, 4.6])

		assert_array_almost_equal(d._xw_sum[0], [[2.3, 0.0], [9.4, 1.8]])
		assert_array_almost_equal(d._xw_sum[1], [[0.0, 0.0], [4.1, 5.5], [2.8, 1.1]])
		assert_array_almost_equal(d._xw_sum[2], [[8.9, 0.0], [0.0, 4.6]])


def test_summarize_dtypes(X, w, probs):
	X = numpy.array(X)
	probs = [numpy.array(prob, dtype=numpy.float32) for prob in probs]

	X = X.astype(numpy.int32)
	d = ConditionalCategorical(probs)
	assert d._xw_sum.dtype == torch.float32
	d.summarize(X)
	assert d._xw_sum.dtype == torch.float32

	X = X.astype(numpy.int64)
	d = ConditionalCategorical(probs)
	assert d._xw_sum.dtype == torch.float32
	d.summarize(X)
	assert d._xw_sum.dtype == torch.float32

	X = X.astype(numpy.int32)
	probs = [numpy.array(prob, dtype=numpy.float64) for prob in probs]

	d = ConditionalCategorical(probs)
	assert d._xw_sum.dtype == torch.float64
	d.summarize(X)
	assert d._xw_sum.dtype == torch.float64

	X = X.astype(numpy.int64)
	d = ConditionalCategorical(probs)
	assert d._xw_sum.dtype == torch.float64
	d.summarize(X)
	assert d._xw_sum.dtype == torch.float64


def test_summarize_raises(X, w, probs):
	_test_raises(ConditionalCategorical(probs), "summarize", X, w=w, 
		min_value=MIN_VALUE, max_value=MAX_VALUE)

	_test_raises(ConditionalCategorical(), "summarize", X, w=w, 
		min_value=MIN_VALUE, max_value=MAX_VALUE)


def _test_efd_from_summaries(d, name1, name2, values):
	for i in range(d.d):
		assert_array_almost_equal(getattr(d, name1)[i], values[i], 4)
		assert_array_almost_equal(getattr(d, name2)[i], numpy.log(values[i]), 2)
		assert_array_almost_equal(d._w_sum[i], numpy.zeros(d.probs[i].shape[:-1]))
		assert_array_almost_equal(d._xw_sum[i], numpy.zeros(d.probs[i].shape))


def test_from_summaries(X, probs):
	d = ConditionalCategorical(probs)
	d.summarize(X)
	d.from_summaries()
	for i in range(d.d):
		assert_raises(AssertionError, assert_array_almost_equal, probs[i], 
			d.probs[i])

	for param in probs, None:
		d = ConditionalCategorical(param)
		d.summarize(X[:4])
		d.from_summaries()
		_test_efd_from_summaries(d, "probs", "_log_probs",
			[[[0.50, 0.50],
	          [0.75, 0.25]],

	         [[0.50, 0.50],
	          [0.50, 0.50],
	          [0.50, 0.50]],

	         [[1.00, 0.00],
	          [0.00, 1.00]]])


		d.summarize(X[4:])
		d.from_summaries()
		_test_efd_from_summaries(d, "probs", "_log_probs", 
			[[[1.00, 0.00],
	          [0.50, 0.50]],

	         [[0.50      , 0.50      ],
	          [0.6666667 , 0.33333334],
	          [0.50      , 0.50      ]],

	         [[1.00, 0.00],
	          [0.00, 1.00]]])

		d = ConditionalCategorical(param)
		d.summarize(X[:4])
		d.summarize(X[4:])
		d.from_summaries()
		_test_efd_from_summaries(d, "probs", "_log_probs", 
			[[[1.0       , 0.0       ],
	          [0.6666667 , 0.33333334]],

	         [[0.50      , 0.50      ],
	          [0.6666667 , 0.33333334],
	          [0.50      , 0.50      ]],

	         [[1.00, 0.00],
	          [0.00, 1.00]]])

		d = ConditionalCategorical(param)
		d.summarize(X)
		d.from_summaries()
		_test_efd_from_summaries(d, "probs", "_log_probs", 
			[[[1.0       , 0.0       ],
	          [0.6666667 , 0.33333334]],

	         [[0.50      , 0.50      ],
	          [0.6666667 , 0.33333334],
	          [0.50      , 0.50      ]],

	         [[1.00, 0.00],
	          [0.00, 1.00]]])


def test_from_summaries_weighted(X, w, probs):
	for param in probs, None:
		d = ConditionalCategorical(param)
		d.summarize(X[:4], sample_weight=w[:4])
		d.from_summaries()
		_test_efd_from_summaries(d, "probs", "_log_probs",
			[[[0.50, 0.50],
	          [1.00, 0.00]],

	         [[0.50     , 0.50     ],
	          [0.50     , 0.50     ],
	          [0.7179487, 0.2820513]],

	         [[1.00, 0.00],
	          [0.00, 1.00]]])


		d.summarize(X[4:], sample_weight=w[4:])
		d.from_summaries()
		_test_efd_from_summaries(d, "probs", "_log_probs", 
			[[[1.00      , 0.00      ],
	          [0.75342464, 0.24657533]],

	         [[0.50     , 0.50     ],
	          [0.4270833, 0.5729166],
	          [0.50     , 0.50     ]],

	         [[1.00, 0.00],
	          [0.00, 1.00]]])

		d = ConditionalCategorical(param)
		d.summarize(X[:4], sample_weight=w[:4])
		d.summarize(X[4:], sample_weight=w[4:])
		d.from_summaries()
		_test_efd_from_summaries(d, "probs", "_log_probs", 
			[[[1.0       , 0.0       ],
	          [0.8392857 , 0.16071428]],

	         [[0.50      , 0.50      ],
	          [0.4270833 , 0.5729166 ],
	          [0.7179487 , 0.2820513 ]],

	         [[1.00, 0.00],
	          [0.00, 1.00]]])

		d = ConditionalCategorical(param)
		d.summarize(X, sample_weight=w)
		d.from_summaries()
		_test_efd_from_summaries(d, "probs", "_log_probs", 
			[[[1.0       , 0.0       ],
	          [0.8392857 , 0.16071428]],

	         [[0.50      , 0.50      ],
	          [0.4270833 , 0.5729166 ],
	          [0.7179487 , 0.2820513 ]],

	         [[1.00, 0.00],
	          [0.00, 1.00]]])


def test_from_summaries_null(probs):
	d = ConditionalCategorical(probs)
	d.from_summaries()
	for i in range(d.d):
		assert_raises(AssertionError, assert_array_almost_equal, d.probs[i], 
			probs[i])

	assert_array_almost_equal(d._w_sum[0], [0.0, 0.0])
	assert_array_almost_equal(d._w_sum[1], [0.0, 0.0, 0.0])
	assert_array_almost_equal(d._w_sum[2], [0.0, 0.0])

	assert_array_almost_equal(d._xw_sum[0], [[0.0, 0.0], [0.0, 0.0]])
	assert_array_almost_equal(d._xw_sum[1], [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]])
	assert_array_almost_equal(d._xw_sum[2], [[0.0, 0.0], [0.0, 0.0]])

	d = ConditionalCategorical(probs, inertia=0.5)
	d.from_summaries()
	for i in range(d.d):
		assert_raises(AssertionError, assert_array_almost_equal, d.probs[i], 
			probs[i])

	assert_array_almost_equal(d._w_sum[0], [0.0, 0.0])
	assert_array_almost_equal(d._w_sum[1], [0.0, 0.0, 0.0])
	assert_array_almost_equal(d._w_sum[2], [0.0, 0.0])

	assert_array_almost_equal(d._xw_sum[0], [[0.0, 0.0], [0.0, 0.0]])
	assert_array_almost_equal(d._xw_sum[1], [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]])
	assert_array_almost_equal(d._xw_sum[2], [[0.0, 0.0], [0.0, 0.0]])

	d = ConditionalCategorical(probs, inertia=0.5, frozen=True)
	d.from_summaries()
	_test_efd_from_summaries(d, "probs", "_log_probs", probs)


def test_from_summaries_inertia(X, w, probs):
	d = ConditionalCategorical(probs, inertia=0.3)
	d.summarize(X[:4])
	d.from_summaries()
	_test_efd_from_summaries(d, "probs", "_log_probs", 
		[[[0.425, 0.575],
          [0.621, 0.379]],

         [[0.380, 0.620],
          [0.440, 0.560],
          [0.422, 0.578]],

         [[0.805, 0.195],
          [0.270, 0.730]]])

	d.summarize(X[4:])
	d.from_summaries()
	_test_efd_from_summaries(d, "probs", "_log_probs", 
		[[[0.8275, 0.1725],
          [0.5363, 0.4637]],

         [[0.464     , 0.536     ],
          [0.59866667, 0.40133333],
          [0.4766    , 0.5234    ]],

         [[0.9415, 0.0585],
          [0.0810, 0.9190]]])

	d = ConditionalCategorical(probs, inertia=0.3)
	d.summarize(X)
	d.from_summaries()
	_test_efd_from_summaries(d, "probs", "_log_probs", 
		[[[0.775     , 0.225       ],
          [0.56266665, 0.43733335]],

         [[0.38      , 0.62      ],
          [0.5566667 , 0.44333333],
          [0.422     , 0.578     ]],

         [[0.805, 0.195],
          [0.270, 0.730]]])


def test_from_summaries_weighted_inertia(X, w, probs):
	d = ConditionalCategorical(probs, inertia=0.3)
	d.summarize(X, sample_weight=w)
	d.from_summaries()
	_test_efd_from_summaries(d, "probs", "_log_probs", 
		[[[0.7750, 0.2250],
          [0.6835, 0.3165]],

         [[0.38      , 0.62      ],
          [0.3889583 , 0.61104167],
          [0.5745641 , 0.4254359 ]],

         [[0.805, 0.195],
          [0.270, 0.730]]])

	d = ConditionalCategorical(probs, inertia=1.0)
	d.summarize(X[:4])
	d.from_summaries()
	_test_efd_from_summaries(d, "probs", "_log_probs", probs)

	d.summarize(X[4:])
	d.from_summaries()
	_test_efd_from_summaries(d, "probs", "_log_probs", probs)

	d = ConditionalCategorical(probs, inertia=1.0)
	d.summarize(X)
	d.from_summaries()
	_test_efd_from_summaries(d, "probs", "_log_probs", probs)


def test_from_summaries_frozen(X, w, probs):
	d = ConditionalCategorical(probs, frozen=True)
	d.summarize(X[:4])
	assert_array_almost_equal(d._w_sum[0], [0.0, 0.0])
	assert_array_almost_equal(d._w_sum[1], [0.0, 0.0, 0.0])
	assert_array_almost_equal(d._w_sum[2], [0.0, 0.0])

	assert_array_almost_equal(d._xw_sum[0], [[0.0, 0.0], [0.0, 0.0]])
	assert_array_almost_equal(d._xw_sum[1], [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]])
	assert_array_almost_equal(d._xw_sum[2], [[0.0, 0.0], [0.0, 0.0]])

	d.from_summaries()
	_test_efd_from_summaries(d, "probs", "_log_probs", probs)

	d.summarize(X[4:])
	assert_array_almost_equal(d._w_sum[0], [0.0, 0.0])
	assert_array_almost_equal(d._w_sum[1], [0.0, 0.0, 0.0])
	assert_array_almost_equal(d._w_sum[2], [0.0, 0.0])

	assert_array_almost_equal(d._xw_sum[0], [[0.0, 0.0], [0.0, 0.0]])
	assert_array_almost_equal(d._xw_sum[1], [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]])
	assert_array_almost_equal(d._xw_sum[2], [[0.0, 0.0], [0.0, 0.0]])

	d.from_summaries()
	_test_efd_from_summaries(d, "probs", "_log_probs", probs)

	d = ConditionalCategorical(probs, frozen=True)
	d.summarize(X)
	assert_array_almost_equal(d._w_sum[0], [0.0, 0.0])
	assert_array_almost_equal(d._w_sum[1], [0.0, 0.0, 0.0])
	assert_array_almost_equal(d._w_sum[2], [0.0, 0.0])

	assert_array_almost_equal(d._xw_sum[0], [[0.0, 0.0], [0.0, 0.0]])
	assert_array_almost_equal(d._xw_sum[1], [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]])
	assert_array_almost_equal(d._xw_sum[2], [[0.0, 0.0], [0.0, 0.0]])

	d.from_summaries()
	_test_efd_from_summaries(d, "probs", "_log_probs", probs)

	d = ConditionalCategorical(probs, frozen=True)
	d.summarize(X, sample_weight=w)
	assert_array_almost_equal(d._w_sum[0], [0.0, 0.0])
	assert_array_almost_equal(d._w_sum[1], [0.0, 0.0, 0.0])
	assert_array_almost_equal(d._w_sum[2], [0.0, 0.0])

	assert_array_almost_equal(d._xw_sum[0], [[0.0, 0.0], [0.0, 0.0]])
	assert_array_almost_equal(d._xw_sum[1], [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]])
	assert_array_almost_equal(d._xw_sum[2], [[0.0, 0.0], [0.0, 0.0]])

	d.from_summaries()
	_test_efd_from_summaries(d, "probs", "_log_probs", probs)


def test_from_summaries_dtypes(X, probs):
	p = [numpy.array(prob, dtype=numpy.float32) for prob in probs]
	d = ConditionalCategorical(p)
	d.summarize(X)
	d.from_summaries()
	assert d.probs[0].dtype == torch.float32
	assert d._log_probs.dtype == torch.float32

	p = [numpy.array(prob, dtype=numpy.float64) for prob in probs]
	d = ConditionalCategorical(p)
	d.summarize(X)
	d.from_summaries()
	assert d.probs[0].dtype == torch.float64
	assert d._log_probs.dtype == torch.float64


def test_from_summaries_raises():
	assert_raises(TypeError, ConditionalCategorical().from_summaries)


def test_fit(X, w, probs):
	d = ConditionalCategorical(probs)
	d.fit(X)
	for i in range(d.d):
		assert_raises(AssertionError, assert_array_almost_equal, probs[i], 
			d.probs[i])

	for param in probs, None:
		d = ConditionalCategorical(param)
		d.fit(X[:4])
		_test_efd_from_summaries(d, "probs", "_log_probs",
			[[[0.50, 0.50],
	          [0.75, 0.25]],

	         [[0.50, 0.50],
	          [0.50, 0.50],
	          [0.50, 0.50]],

	         [[1.00, 0.00],
	          [0.00, 1.00]]])


		d.fit(X[4:])
		_test_efd_from_summaries(d, "probs", "_log_probs", 
			[[[1.00, 0.00],
	          [0.50, 0.50]],

	         [[0.50      , 0.50      ],
	          [0.6666667 , 0.33333334],
	          [0.50      , 0.50      ]],

	         [[1.00, 0.00],
	          [0.00, 1.00]]])

		d = ConditionalCategorical(param)
		d.fit(X)
		_test_efd_from_summaries(d, "probs", "_log_probs", 
			[[[1.0       , 0.0       ],
	          [0.6666667 , 0.33333334]],

	         [[0.50      , 0.50      ],
	          [0.6666667 , 0.33333334],
	          [0.50      , 0.50      ]],

	         [[1.00, 0.00],
	          [0.00, 1.00]]])


def test_fit_weighted(X, w, probs):
	for param in probs, None:
		d = ConditionalCategorical(param)
		d.fit(X[:4], sample_weight=w[:4])
		_test_efd_from_summaries(d, "probs", "_log_probs",
			[[[0.50, 0.50],
	          [1.00, 0.00]],

	         [[0.50     , 0.50     ],
	          [0.50     , 0.50     ],
	          [0.7179487, 0.2820513]],

	         [[1.00, 0.00],
	          [0.00, 1.00]]])


		d.fit(X[4:], sample_weight=w[4:])
		_test_efd_from_summaries(d, "probs", "_log_probs", 
			[[[1.00      , 0.00      ],
	          [0.75342464, 0.24657533]],

	         [[0.50     , 0.50     ],
	          [0.4270833, 0.5729166],
	          [0.50     , 0.50     ]],

	         [[1.00, 0.00],
	          [0.00, 1.00]]])


		d = ConditionalCategorical(param)
		d.fit(X, sample_weight=w)
		_test_efd_from_summaries(d, "probs", "_log_probs", 
			[[[1.0       , 0.0       ],
	          [0.8392857 , 0.16071428]],

	         [[0.50      , 0.50      ],
	          [0.4270833 , 0.5729166 ],
	          [0.7179487 , 0.2820513 ]],

	         [[1.00, 0.00],
	          [0.00, 1.00]]])


def test_fit_chain(X):
	d = ConditionalCategorical().fit(X[:4])
	_test_efd_from_summaries(d, "probs", "_log_probs",
		[[[0.50, 0.50],
          [0.75, 0.25]],

         [[0.50, 0.50],
          [0.50, 0.50],
          [0.50, 0.50]],

         [[1.00, 0.00],
          [0.00, 1.00]]])


	d.fit(X[4:])
	_test_efd_from_summaries(d, "probs", "_log_probs", 
		[[[1.00, 0.00],
          [0.50, 0.50]],

         [[0.50      , 0.50      ],
          [0.6666667 , 0.33333334],
          [0.50      , 0.50      ]],

         [[1.00, 0.00],
          [0.00, 1.00]]])

	d = ConditionalCategorical().fit(X)
	_test_efd_from_summaries(d, "probs", "_log_probs", 
		[[[1.0       , 0.0       ],
          [0.6666667 , 0.33333334]],

         [[0.50      , 0.50      ],
          [0.6666667 , 0.33333334],
          [0.50      , 0.50      ]],

         [[1.00, 0.00],
          [0.00, 1.00]]])


def test_fit_dtypes(X, probs):
	p = [numpy.array(prob, dtype=numpy.float32) for prob in probs]
	d = ConditionalCategorical(p)
	d.fit(X)
	assert d.probs[0].dtype == torch.float32
	assert d._log_probs.dtype == torch.float32

	p = [numpy.array(prob, dtype=numpy.float64) for prob in probs]
	d = ConditionalCategorical(p)
	d.fit(X)
	assert d.probs[0].dtype == torch.float64
	assert d._log_probs.dtype == torch.float64


def test_fit_raises(X, w, probs):
	_test_raises(ConditionalCategorical(probs), "fit", X, w=w, min_value=MIN_VALUE, 
		max_value=MAX_VALUE)

	_test_raises(ConditionalCategorical(), "fit", X, w=w, min_value=MIN_VALUE, 
		max_value=MAX_VALUE)

# test_JointCategorical.py
# Contact: Jacob Schreiber <jmschreiber91@gmail.com>

import numpy
import torch
import pytest

from torchegranate.distributions import JointCategorical

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
	return [[1, 2, 0],
		 [0, 0, 1],
		 [1, 2, 0],
		 [1, 2, 0],
		 [1, 1, 0],
		 [1, 1, 0],
		 [0, 1, 0]]


@pytest.fixture
def w():
	return [[1.1], [2.8], [0], [0], [5.5], [1.8], [2.3]]



@pytest.fixture
def probs():
	return [[[0.25 / 6., 0.75 / 6.],
			 [0.32 / 6., 0.68 / 6.],
			 [0.5 / 6., 0.5 / 6.]],

			[[0.1 / 6., 0.9 / 6.],
			 [0.3 / 6., 0.7 / 6.],
			 [0.24 / 6., 0.76 / 6.]]]


###


def test_initialization():
	d = JointCategorical()
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
		_test_initialization(JointCategorical(y, inertia=0.0, frozen=False), 
			y, "probs", 0.0, False, torch.float32)
		_test_initialization(JointCategorical(y, inertia=0.3, frozen=False), 
			y, "probs", 0.3, False, torch.float32)
		_test_initialization(JointCategorical(y, inertia=1.0, frozen=True), 
			y, "probs", 1.0, True, torch.float32)
		_test_initialization(JointCategorical(y, inertia=1.0, frozen=False), 
			y, "probs", 1.0, False, torch.float32)

	x = numpy.array(probs, dtype=numpy.float64)
	_test_initialization(JointCategorical(x, inertia=0.0, frozen=False), 
		x, "probs", 0.0, False, torch.float64)
	_test_initialization(JointCategorical(x, inertia=0.3, frozen=False), 
		x, "probs", 0.3, False, torch.float64)
	_test_initialization(JointCategorical(x, inertia=1.0, frozen=True), 
		x, "probs", 1.0, True, torch.float64)
	_test_initialization(JointCategorical(x, inertia=1.0, frozen=False), 
		x, "probs", 1.0, False, torch.float64)


def test_initialization_raises(probs):	
	assert_raises(ValueError, JointCategorical, 0.3)
	assert_raises(ValueError, JointCategorical, probs, inertia=-0.4)
	assert_raises(ValueError, JointCategorical, probs, inertia=1.2)
	assert_raises(ValueError, JointCategorical, probs, inertia=1.2, 
		frozen="true")
	assert_raises(ValueError, JointCategorical, probs, inertia=1.2, 
		frozen=3)
	
	assert_raises(ValueError, JointCategorical, inertia=-0.4)
	assert_raises(ValueError, JointCategorical, inertia=1.2)
	assert_raises(ValueError, JointCategorical, inertia=1.2, frozen="true")
	assert_raises(ValueError, JointCategorical, inertia=1.2, frozen=3)

	assert_raises(ValueError, JointCategorical, numpy.array(probs)+0.001)
	assert_raises(ValueError, JointCategorical, numpy.array(probs)-0.001)

	p = numpy.array(probs)
	p[0, 0] = -0.03
	assert_raises(ValueError, JointCategorical, p)

	p = numpy.array(probs)
	p[0, 0] = 1.03
	assert_raises(ValueError, JointCategorical, p)


def test_reset_cache(X):
	d = JointCategorical()
	d.summarize(X)
	assert_array_almost_equal(d._w_sum, [7.0, 7.0, 7.0])
	assert_array_almost_equal(d._xw_sum, 
		[[[0., 1.],
		  [1., 0.],
		  [0., 0.]],

		 [[0., 0.],
		  [2., 0.],
		  [3., 0.]]])

	d._reset_cache()
	assert_array_almost_equal(d._w_sum, [0.0, 0.0, 0.0])
	assert_array_almost_equal(d._xw_sum,
		[[[0., 0.],
		  [0., 0.],
		  [0., 0.]],

		 [[0., 0.],
		  [0., 0.],
		  [0., 0.]]])

	d = JointCategorical()
	assert_raises(AttributeError, getattr, d, "_w_sum")
	assert_raises(AttributeError, getattr, d, "_xw_sum")
	assert_raises(AttributeError, getattr, d, "_log_probs")

	d._reset_cache()
	assert_raises(AttributeError, getattr, d, "_w_sum")
	assert_raises(AttributeError, getattr, d, "_xw_sum")
	assert_raises(AttributeError, getattr, d, "_log_probs")


def test_initialize(X, probs):
	d = JointCategorical()
	assert d.d is None
	assert d.probs is None
	assert d._initialized == False
	assert_raises(AttributeError, getattr, d, "_w_sum")
	assert_raises(AttributeError, getattr, d, "_xw_sum")
	assert_raises(AttributeError, getattr, d, "_log_probs")

	d._initialize(3, (2, 3, 2))
	assert d._initialized == True
	assert d.probs.shape == (2, 3, 2)
	assert d.d == 3
	assert_array_almost_equal(d.probs,
		[[[0., 0.],
		  [0., 0.],
		  [0., 0.]],

		 [[0., 0.],
		  [0., 0.],
		  [0., 0.]]])
	assert_array_almost_equal(d._w_sum, [0.0, 0.0, 0.0])
	assert_array_almost_equal(d._xw_sum,
		[[[0., 0.],
		  [0., 0.],
		  [0., 0.]],

		 [[0., 0.],
		  [0., 0.],
		  [0., 0.]]])	

	d._initialize(2, (2, 2))
	assert d._initialized == True
	assert d.probs.shape == (2, 2)
	assert d.d == 2
	assert_array_almost_equal(d.probs,
		[[0., 0.],
		 [0., 0.]])
	assert_array_almost_equal(d._w_sum, [0.0, 0.0])
	assert_array_almost_equal(d._xw_sum,
		[[0., 0.],
		 [0., 0.]])	

	d = JointCategorical(probs)
	assert d._initialized == True
	assert d.d == 3
	assert d.n_categories == (2, 3, 2)

	d._initialize(3, (2, 2, 4))
	assert d._initialized == True
	assert d.probs.shape == (2, 2, 4)
	assert d.d == 3
	assert_array_almost_equal(d.probs,
		[[[0., 0., 0., 0.],
		  [0., 0., 0., 0.]],

		 [[0., 0., 0., 0.],
		  [0., 0., 0., 0.]]])

	d = JointCategorical()
	d.summarize(X)
	assert d._initialized == True
	assert d.probs.shape == (2, 3, 2)
	assert d.d == 3
	assert_array_almost_equal(d._w_sum, [7.0, 7.0, 7.0])
	assert_array_almost_equal(d._xw_sum,
		[[[0., 1.],
		  [1., 0.],
		  [0., 0.]],

		 [[0., 0.],
		  [2., 0.],
		  [3., 0.]]])	

	d = JointCategorical()
	d.summarize(X)
	d._initialize(4, (2, 2, 2, 2))
	assert d._initialized == True
	assert d.probs.shape == (2, 2, 2, 2)
	assert d.d == 4
	assert_array_almost_equal(d._w_sum, [0.0, 0.0, 0.0, 0.0])
	assert_array_almost_equal(d._xw_sum,
		[[[[0., 0.],
		   [0., 0.]],

		  [[0., 0.],
		   [0., 0.]]],


		 [[[0., 0.],
		   [0., 0.]],

		  [[0., 0.],
		   [0., 0.]]]])	


###


@pytest.mark.sample
def test_sample(probs):
	torch.manual_seed(0)

	X = JointCategorical(probs).sample(1)
	assert_array_almost_equal(X, [[1, 2, 1]])

	X = JointCategorical(probs).sample(5)
	assert_array_almost_equal(X, 
		[[1, 1, 0],
         [0, 2, 1],
         [1, 2, 1],
         [1, 0, 1],
         [1, 1, 1]], 3)


###


def test_probability(X, probs):
	y = [0.04, 0.125, 0.04, 0.04, 0.05, 0.05, 0.053333]

	d1 = JointCategorical(probs)
	d2 = JointCategorical(numpy.array(probs, dtype=numpy.float64))

	_test_predictions(X, y, d1.probability(X), torch.float32)
	_test_predictions(X, y, d2.probability(X), torch.float64)


def test_probability_dtypes(X, probs):	
	y = JointCategorical(probs).probability(X)
	assert y.dtype == torch.float32

	p = numpy.array(probs, dtype=numpy.float64)
	y = JointCategorical(p).probability(X)
	assert y.dtype == torch.float64


def test_probability_raises(X, probs):
	_test_raises(JointCategorical(probs), "probability", X, min_value=MIN_VALUE, 
		max_value=MAX_VALUE)

	_test_raises(JointCategorical(probs), "probability", X, 
		min_value=MIN_VALUE, max_value=MAX_VALUE)


def test_log_probability(X, probs):
	y = [-3.218876, -2.079442, -3.218876, -3.218876, -2.995732, -2.995732,
           -2.931194]

	d1 = JointCategorical(probs)
	d2 = JointCategorical(numpy.array(probs, dtype=numpy.float64))

	_test_predictions(X, y, d1.log_probability(X), torch.float32)
	_test_predictions(X, y, d2.log_probability(X), torch.float64)


def test_log_probability_dtypes(X, probs):
	y = JointCategorical(probs).log_probability(X)
	assert y.dtype == torch.float32

	p = numpy.array(probs, dtype=numpy.float64)
	y = JointCategorical(p).log_probability(X)
	assert y.dtype == torch.float64


def test_log_probability_raises(X, probs):
	_test_raises(JointCategorical(probs), "log_probability", X, 
		min_value=MIN_VALUE, max_value=MAX_VALUE)

	_test_raises(JointCategorical(probs), "log_probability", X, 
		min_value=MIN_VALUE, max_value=MAX_VALUE)


###


def test_summarize(X, probs):
	d = JointCategorical(probs)
	d.summarize(X[:4])
	assert_array_almost_equal(d._w_sum, [4.0, 4.0, 4.0])
	assert_array_almost_equal(d._xw_sum,
		[[[0., 1.],
          [0., 0.],
          [0., 0.]],

         [[0., 0.],
          [0., 0.],
          [3., 0.]]])

	d.summarize(X[4:])
	assert_array_almost_equal(d._w_sum, [7.0, 7.0, 7.0])
	assert_array_almost_equal(d._xw_sum,
		[[[0., 1.],
          [1., 0.],
          [0., 0.]],

         [[0., 0.],
          [2., 0.],
          [3., 0.]]])

	d = JointCategorical(probs)
	d.summarize(X)
	assert_array_almost_equal(d._w_sum, [7.0, 7.0, 7.0])
	assert_array_almost_equal(d._xw_sum,
		[[[0., 1.],
          [1., 0.],
          [0., 0.]],

         [[0., 0.],
          [2., 0.],
          [3., 0.]]])


	d = JointCategorical()
	d.summarize(X[:4])
	assert_array_almost_equal(d._w_sum, [4.0, 4.0, 4.0])
	assert_array_almost_equal(d._xw_sum,
		[[[0., 1.],
          [0., 0.],
          [0., 0.]],

         [[0., 0.],
          [0., 0.],
          [3., 0.]]])

	d.summarize(X[4:])
	assert_array_almost_equal(d._w_sum, [7.0, 7.0, 7.0])
	assert_array_almost_equal(d._xw_sum,
		[[[0., 1.],
          [1., 0.],
          [0., 0.]],

         [[0., 0.],
          [2., 0.],
          [3., 0.]]])

	d = JointCategorical()
	d.summarize(X)
	assert_array_almost_equal(d._w_sum, [7.0, 7.0, 7.0])
	assert_array_almost_equal(d._xw_sum,
		[[[0., 1.],
          [1., 0.],
          [0., 0.]],

         [[0., 0.],
          [2., 0.],
          [3., 0.]]])


def test_summarize_weighted(X, w, probs):
	d = JointCategorical(probs)
	d.summarize(X[:4], sample_weight=w[:4])
	assert_array_almost_equal(d._w_sum, [3.9, 3.9, 3.9])
	assert_array_almost_equal(d._xw_sum,
		[[[0.0000, 2.8000],
          [0.0000, 0.0000],
          [0.0000, 0.0000]],

         [[0.0000, 0.0000],
          [0.0000, 0.0000],
          [1.1000, 0.0000]]])


	d.summarize(X[4:], sample_weight=w[4:])
	assert_array_almost_equal(d._w_sum, [13.5, 13.5, 13.5])
	assert_array_almost_equal(d._xw_sum,
		[[[0.0000, 2.8000],
          [2.3000, 0.0000],
          [0.0000, 0.0000]],

         [[0.0000, 0.0000],
          [7.3000, 0.0000],
          [1.1000, 0.0000]]])

	d = JointCategorical(probs)
	d.summarize(X, sample_weight=w)
	assert_array_almost_equal(d._w_sum, [13.5, 13.5, 13.5])
	assert_array_almost_equal(d._xw_sum,
		[[[0.0000, 2.8000],
          [2.3000, 0.0000],
          [0.0000, 0.0000]],

         [[0.0000, 0.0000],
          [7.3000, 0.0000],
          [1.1000, 0.0000]]])


def test_summarize_weighted_flat(X, w, probs):
	w = numpy.array(w)[:,0] 

	d = JointCategorical(probs)
	d.summarize(X[:4], sample_weight=w[:4])
	assert_array_almost_equal(d._w_sum, [3.9, 3.9, 3.9])
	assert_array_almost_equal(d._xw_sum,
		[[[0.0000, 2.8000],
          [0.0000, 0.0000],
          [0.0000, 0.0000]],

         [[0.0000, 0.0000],
          [0.0000, 0.0000],
          [1.1000, 0.0000]]])

	d.summarize(X[4:], sample_weight=w[4:])
	assert_array_almost_equal(d._w_sum, [13.5, 13.5, 13.5])
	assert_array_almost_equal(d._xw_sum,
		[[[0.0000, 2.8000],
          [2.3000, 0.0000],
          [0.0000, 0.0000]],

         [[0.0000, 0.0000],
          [7.3000, 0.0000],
          [1.1000, 0.0000]]])

	d = JointCategorical(probs)
	d.summarize(X, sample_weight=w)
	assert_array_almost_equal(d._w_sum, [13.5, 13.5, 13.5])
	assert_array_almost_equal(d._xw_sum,
		[[[0.0000, 2.8000],
          [2.3000, 0.0000],
          [0.0000, 0.0000]],

         [[0.0000, 0.0000],
          [7.3000, 0.0000],
          [1.1000, 0.0000]]])


def test_summarize_dtypes(X, w, probs):
	X = numpy.array(X)
	probs = numpy.array(probs, dtype=numpy.float32)

	X = X.astype(numpy.int32)
	d = JointCategorical(probs)
	assert d._xw_sum.dtype == torch.float32
	d.summarize(X)
	assert d._xw_sum.dtype == torch.float32

	X = X.astype(numpy.int64)
	d = JointCategorical(probs)
	assert d._xw_sum.dtype == torch.float32
	d.summarize(X)
	assert d._xw_sum.dtype == torch.float32

	X = X.astype(numpy.int32)
	d = JointCategorical(probs.astype(numpy.float64))
	assert d._xw_sum.dtype == torch.float64
	d.summarize(X)
	assert d._xw_sum.dtype == torch.float64

	X = X.astype(numpy.int64)
	d = JointCategorical(probs.astype(numpy.float64))
	assert d._xw_sum.dtype == torch.float64
	d.summarize(X)
	assert d._xw_sum.dtype == torch.float64


def test_summarize_raises(X, w, probs):
	_test_raises(JointCategorical(probs), "summarize", X, w=w, 
		min_value=MIN_VALUE, max_value=MAX_VALUE)

	_test_raises(JointCategorical(), "summarize", X, w=w, 
		min_value=MIN_VALUE, max_value=MAX_VALUE)


def _test_efd_from_summaries(d, name1, name2, values):
	assert_array_almost_equal(getattr(d, name1), values, 4)
	assert_array_almost_equal(getattr(d, name2), numpy.log(values), 2)
	assert_array_almost_equal(d._w_sum, numpy.zeros(d.d))
	assert_array_almost_equal(d._xw_sum, numpy.zeros(d.probs.shape))


def test_from_summaries(X, probs):
	d = JointCategorical(probs)
	d.summarize(X)
	d.from_summaries()
	assert_raises(AssertionError, assert_array_almost_equal, probs, d.probs)

	for param in probs, None:
		d = JointCategorical(param)
		d.summarize(X[:4])
		d.from_summaries()
		_test_efd_from_summaries(d, "probs", "_log_probs",
			[[[0.0000, 0.2500],
	          [0.0000, 0.0000],
	          [0.0000, 0.0000]],

	         [[0.0000, 0.0000],
	          [0.0000, 0.0000],
	          [0.7500, 0.0000]]])

		d.summarize(X[4:])
		d.from_summaries()
		_test_efd_from_summaries(d, "probs", "_log_probs", 
			[[[0.0000, 0.0000],
			  [0.3333, 0.0000],
			  [0.0000, 0.0000]],

			 [[0.0000, 0.0000],
			  [0.6667, 0.0000],
			  [0.0000, 0.0000]]])

		d = JointCategorical(param)
		d.summarize(X[:4])
		d.summarize(X[4:])
		d.from_summaries()
		_test_efd_from_summaries(d, "probs", "_log_probs", 
			[[[0.0000, 0.1429],
       		  [0.1429, 0.0000],
     	      [0.0000, 0.0000]],

        	 [[0.0000, 0.0000],
         	  [0.2857, 0.0000],
        	  [0.4286, 0.0000]]])

		d = JointCategorical(param)
		d.summarize(X)
		d.from_summaries()
		_test_efd_from_summaries(d, "probs", "_log_probs", 
			[[[0.0000, 0.1429],
       		  [0.1429, 0.0000],
     	      [0.0000, 0.0000]],

        	 [[0.0000, 0.0000],
         	  [0.2857, 0.0000],
        	  [0.4286, 0.0000]]])


def test_from_summaries_weighted(X, w, probs):
	for param in probs, None:
		d = JointCategorical(probs)
		d.summarize(X[:4], sample_weight=w[:4])
		d.from_summaries()
		_test_efd_from_summaries(d, "probs", "_log_probs", 
			[[[0.0000, 0.7179],
         	  [0.0000, 0.0000],
      	      [0.0000, 0.0000]],

      	     [[0.0000, 0.0000],
              [0.0000, 0.0000],
              [0.2821, 0.0000]]])

		d.summarize(X[4:], sample_weight=w[4:])
		d.from_summaries()
		_test_efd_from_summaries(d, "probs", "_log_probs", 
			[[[0.0000, 0.0000],
       		  [0.2396, 0.0000],
      	      [0.0000, 0.0000]],

      	     [[0.0000, 0.0000],
              [0.7604, 0.0000],
              [0.0000, 0.0000]]])

		d = JointCategorical(probs)
		d.summarize(X, sample_weight=w)
		d.from_summaries()
		_test_efd_from_summaries(d, "probs", "_log_probs", 
			[[[0.0000, 0.2074],
        	 [0.1704, 0.0000],
      	     [0.0000, 0.0000]],

     	    [[0.0000, 0.0000],
     	     [0.5407, 0.0000],
      	     [0.0815, 0.0000]]])


def test_from_summaries_null(probs):
	d = JointCategorical(probs)
	d.from_summaries()
	assert_raises(AssertionError, assert_array_almost_equal, d.probs, probs)
	assert_array_almost_equal(d._w_sum, [0.0, 0.0, 0.0])
	assert_array_almost_equal(d._xw_sum, numpy.zeros_like(probs))

	d = JointCategorical(probs, inertia=0.5)
	d.from_summaries()
	assert_raises(AssertionError, assert_array_almost_equal, d.probs, probs)
	assert_array_almost_equal(d._w_sum, [0.0, 0.0, 0.0])
	assert_array_almost_equal(d._xw_sum, numpy.zeros_like(probs))

	d = JointCategorical(probs, inertia=0.5, frozen=True)
	d.from_summaries()
	_test_efd_from_summaries(d, "probs", "_log_probs", probs)


def test_from_summaries_inertia(X, w, probs):
	d = JointCategorical(probs, inertia=0.3)
	d.summarize(X[:4])
	d.from_summaries()
	_test_efd_from_summaries(d, "probs", "_log_probs", 
		[[[0.0125, 0.2125],
          [0.0160, 0.0340],
          [0.0250, 0.0250]],

         [[0.0050, 0.0450],
          [0.0150, 0.0350],
          [0.5370, 0.0380]]])

	d.summarize(X[4:])
	d.from_summaries()
	_test_efd_from_summaries(d, "probs", "_log_probs", 
		[[[0.0038, 0.0638],
          [0.2381, 0.0102],
          [0.0075, 0.0075]],

         [[0.0015, 0.0135],
          [0.4712, 0.0105],
          [0.1611, 0.0114]]])

	d = JointCategorical(probs, inertia=0.3)
	d.summarize(X)
	d.from_summaries()
	_test_efd_from_summaries(d, "probs", "_log_probs", 
		[[[0.0125, 0.1375],
          [0.1160, 0.0340],
          [0.0250, 0.0250]],

         [[0.0050, 0.0450],
          [0.2150, 0.0350],
          [0.3120, 0.0380]]])


def test_from_summaries_weighted_inertia(X, w, probs):
	d = JointCategorical(probs, inertia=0.3)
	d.summarize(X, sample_weight=w)
	d.from_summaries()
	_test_efd_from_summaries(d, "probs", "_log_probs", 
		[[[0.0125, 0.1827],
          [0.1353, 0.0340],
          [0.0250, 0.0250]],

         [[0.0050, 0.0450],
          [0.3935, 0.0350],
          [0.0690, 0.0380]]])

	d = JointCategorical(probs, inertia=1.0)
	d.summarize(X[:4])
	d.from_summaries()
	_test_efd_from_summaries(d, "probs", "_log_probs", probs)

	d.summarize(X[4:])
	d.from_summaries()
	_test_efd_from_summaries(d, "probs", "_log_probs", probs)

	d = JointCategorical(probs, inertia=1.0)
	d.summarize(X)
	d.from_summaries()
	_test_efd_from_summaries(d, "probs", "_log_probs", probs)


def test_from_summaries_frozen(X, w, probs):
	d = JointCategorical(probs, frozen=True)
	d.summarize(X[:4])
	assert_array_almost_equal(d._w_sum, [0.0, 0.0, 0.0])
	assert_array_almost_equal(d._xw_sum, numpy.zeros_like(probs))

	d.from_summaries()
	_test_efd_from_summaries(d, "probs", "_log_probs", probs)

	d.summarize(X[4:])
	assert_array_almost_equal(d._w_sum, [0.0, 0.0, 0.0])
	assert_array_almost_equal(d._xw_sum, numpy.zeros_like(probs))

	d.from_summaries()
	_test_efd_from_summaries(d, "probs", "_log_probs", probs)

	d = JointCategorical(probs, frozen=True)
	d.summarize(X)
	assert_array_almost_equal(d._w_sum, [0.0, 0.0, 0.0])
	assert_array_almost_equal(d._xw_sum, numpy.zeros_like(probs))

	d.from_summaries()
	_test_efd_from_summaries(d, "probs", "_log_probs", probs)

	d = JointCategorical(probs, frozen=True)
	d.summarize(X, sample_weight=w)
	assert_array_almost_equal(d._w_sum, [0.0, 0.0, 0.0])
	assert_array_almost_equal(d._xw_sum, numpy.zeros_like(probs))

	d.from_summaries()
	_test_efd_from_summaries(d, "probs", "_log_probs", probs)


def test_from_summaries_dtypes(X, probs):
	p = numpy.array(probs, dtype=numpy.float32)
	d = JointCategorical(p)
	d.summarize(X)
	d.from_summaries()
	assert d.probs.dtype == torch.float32
	assert d._log_probs.dtype == torch.float32

	p = numpy.array(probs, dtype=numpy.float64)
	d = JointCategorical(p)
	d.summarize(X)
	d.from_summaries()
	assert d.probs.dtype == torch.float64
	assert d._log_probs.dtype == torch.float64


def test_from_summaries_raises():
	assert_raises(AttributeError, JointCategorical().from_summaries)


def test_fit(X, w, probs):
	d = JointCategorical(probs)
	d.fit(X)
	assert_raises(AssertionError, assert_array_almost_equal, probs, d.probs)

	for param in probs, None:
		d = JointCategorical(param)
		d.fit(X[:4])
		_test_efd_from_summaries(d, "probs", "_log_probs",
			[[[0.0000, 0.2500],
	          [0.0000, 0.0000],
	          [0.0000, 0.0000]],

	         [[0.0000, 0.0000],
	          [0.0000, 0.0000],
	          [0.7500, 0.0000]]])

		d.fit(X[4:])
		_test_efd_from_summaries(d, "probs", "_log_probs", 
			[[[0.0000, 0.0000],
			  [0.3333, 0.0000],
			  [0.0000, 0.0000]],

			 [[0.0000, 0.0000],
			  [0.6667, 0.0000],
			  [0.0000, 0.0000]]])

		d = JointCategorical(param)
		d.fit(X)
		_test_efd_from_summaries(d, "probs", "_log_probs", 
			[[[0.0000, 0.1429],
       		  [0.1429, 0.0000],
     	      [0.0000, 0.0000]],

        	 [[0.0000, 0.0000],
         	  [0.2857, 0.0000],
        	  [0.4286, 0.0000]]])


def test_fit_weighted(X, w, probs):
	for param in probs, None:
		d = JointCategorical(probs)
		d.fit(X[:4], sample_weight=w[:4])
		_test_efd_from_summaries(d, "probs", "_log_probs", 
			[[[0.0000, 0.7179],
         	  [0.0000, 0.0000],
      	      [0.0000, 0.0000]],

      	     [[0.0000, 0.0000],
              [0.0000, 0.0000],
              [0.2821, 0.0000]]])

		d.fit(X[4:], sample_weight=w[4:])
		_test_efd_from_summaries(d, "probs", "_log_probs", 
			[[[0.0000, 0.0000],
       		  [0.2396, 0.0000],
      	      [0.0000, 0.0000]],

      	     [[0.0000, 0.0000],
              [0.7604, 0.0000],
              [0.0000, 0.0000]]])

		d = JointCategorical(probs)
		d.fit(X, sample_weight=w)
		_test_efd_from_summaries(d, "probs", "_log_probs", 
			[[[0.0000, 0.2074],
           	  [0.1704, 0.0000],
      	      [0.0000, 0.0000]],

     	     [[0.0000, 0.0000],
     	      [0.5407, 0.0000],
      	      [0.0815, 0.0000]]])


def test_fit_chain(X):
	d = JointCategorical().fit(X[:4])
	_test_efd_from_summaries(d, "probs", "_log_probs",
		[[[0.0000, 0.2500],
          [0.0000, 0.0000],
          [0.0000, 0.0000]],

         [[0.0000, 0.0000],
          [0.0000, 0.0000],
          [0.7500, 0.0000]]])

	d.fit(X[4:])
	_test_efd_from_summaries(d, "probs", "_log_probs", 
		[[[0.0000, 0.0000],
		  [0.3333, 0.0000],
		  [0.0000, 0.0000]],

		 [[0.0000, 0.0000],
		  [0.6667, 0.0000],
		  [0.0000, 0.0000]]])


	d = JointCategorical().fit(X)
	_test_efd_from_summaries(d, "probs", "_log_probs", 
		[[[0.0000, 0.1429],
   		  [0.1429, 0.0000],
 	      [0.0000, 0.0000]],

    	 [[0.0000, 0.0000],
     	  [0.2857, 0.0000],
    	  [0.4286, 0.0000]]])


def test_fit_dtypes(X, probs):
	p = numpy.array(probs, dtype=numpy.float32)
	d = JointCategorical(p)
	d.fit(X)
	assert d.probs.dtype == torch.float32
	assert d._log_probs.dtype == torch.float32

	p = numpy.array(probs, dtype=numpy.float64)
	d = JointCategorical(p)
	d.fit(X)
	assert d.probs.dtype == torch.float64
	assert d._log_probs.dtype == torch.float64


def test_fit_raises(X, w, probs):
	_test_raises(JointCategorical(probs), "fit", X, w=w, min_value=MIN_VALUE, 
		max_value=MAX_VALUE)

	_test_raises(JointCategorical(), "fit", X, w=w, min_value=MIN_VALUE, 
		max_value=MAX_VALUE)

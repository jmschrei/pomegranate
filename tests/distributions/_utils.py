# _utils.py
# Contact: Jacob Schreiber <jmschreiber91@gmail.com>

import numpy
import torch

from nose.tools import assert_raises
from numpy.testing import assert_array_almost_equal


def _test_initialization(d, x, name, inertia, frozen, dtype):
	assert d.inertia == inertia
	assert d.frozen == frozen
	param = getattr(d, name)

	if x is not None:
		assert param.shape[0] == len(x)
		assert param.dtype == dtype
		assert_array_almost_equal(param, x)
	else:
		assert param == x


def _test_initialization_raises_one_parameter(distribution, valid_value, 
	min_value=None, max_value=None):
	assert_raises(ValueError, distribution, valid_value)
	assert_raises(ValueError, distribution, [valid_value], inertia=-0.4)
	assert_raises(ValueError, distribution, [valid_value], inertia=1.2)
	assert_raises(ValueError, distribution, [valid_value], inertia=1.2, 
		frozen="true")
	assert_raises(ValueError, distribution, [valid_value], inertia=1.2, 
		frozen=3)
	
	assert_raises(ValueError, distribution, inertia=-0.4)
	assert_raises(ValueError, distribution, inertia=1.2)
	assert_raises(ValueError, distribution, inertia=1.2, frozen="true")
	assert_raises(ValueError, distribution, inertia=1.2, frozen=3)

	if min_value is not None:
		assert_raises(ValueError, distribution, [valid_value, min_value-0.1])

	if max_value is not None:
		assert_raises(ValueError, distribution, [valid_value, max_value+0.1])


def _test_initialization_raises_two_parameters(distribution, valid_value1,
	valid_value2, min_value1=None, min_value2=None, max_value1=None,
	max_value2=None):
	
	assert_raises(ValueError, distribution, valid_value1)
	assert_raises(ValueError, distribution, None, valid_value2)
	assert_raises(ValueError, distribution, valid_value1, valid_value2)
	assert_raises(ValueError, distribution, [valid_value1], 
		[valid_value2, valid_value2])
	assert_raises(ValueError, distribution, [valid_value1, valid_value1], 
		[valid_value2])

	assert_raises(ValueError, distribution, [valid_value1, valid_value2], 
		inertia=-0.4)
	assert_raises(ValueError, distribution, [valid_value1, valid_value2], 
		inertia=1.2)
	assert_raises(ValueError, distribution, [valid_value1, valid_value2], 
		inertia=1.2, frozen="true")
	assert_raises(ValueError, distribution, [valid_value1, valid_value2], 
		inertia=1.2, frozen=3)

	assert_raises(ValueError, distribution, inertia=-0.4)
	assert_raises(ValueError, distribution, inertia=1.2)
	assert_raises(ValueError, distribution, inertia=1.2, frozen="true")
	assert_raises(ValueError, distribution, inertia=1.2, frozen=3)

	if min_value1 is not None:
		assert_raises(ValueError, distribution, [valid_value1, min_value1-0.1],
			[valid_value2, valid_value2])

	if min_value2 is not None:
		assert_raises(ValueError, distribution, [valid_value1, valid_value1],
			[valid_value2, min_value2-0.1])

	if max_value1 is not None:
		assert_raises(ValueError, distribution, [valid_value1, max_value1+0.1],
			[valid_value2, valid_value2])

	if max_value2 is not None:
		assert_raises(ValueError, distribution, [valid_value1, valid_value1],
			[valid_value2, max_value2+0.1])


def _test_predictions(x, y, y_hat, dtype):
	assert isinstance(y_hat, torch.Tensor)
	assert y_hat.dtype == dtype
	assert y_hat.shape == (len(x),)
	assert_array_almost_equal(y, y_hat)


def _test_raises(d, name, X, w=None, min_value=None, max_value=None):
	f = getattr(d, name)

	assert_raises(ValueError, f, [X])
	assert_raises(ValueError, f, X[0])
	assert_raises((ValueError, TypeError, RuntimeError), f, X[0][0])

	if d._initialized == True:
		assert_raises(ValueError, f, [x[:-1] for x in X])

		if min_value is not None:
			assert_raises(ValueError, f, [[min_value-0.1 for i in range(d.d)]])
		
		if max_value is not None:
			assert_raises(ValueError, f, [[max_value+0.1 for i in range(d.d)]])
	else:	
		if min_value is not None:
			assert_raises(ValueError, f, [[min_value-0.1 for i in range(3)]])
		
		if max_value is not None:
			assert_raises(ValueError, f, [[max_value+0.1 for i in range(3)]])


	if w is not None:
		assert_raises(ValueError, f, [X], w)
		assert_raises(ValueError, f, X, [w])
		assert_raises(ValueError, f, [X], [w])
		assert_raises(ValueError, f, X, w[:len(w)-1])
		assert_raises(ValueError, f, X[:len(X)-1], w)


def _test_efd_from_summaries(d, name1, name2, values):
	assert_array_almost_equal(getattr(d, name1), values)
	assert_array_almost_equal(getattr(d, name2), numpy.log(values))
	assert_array_almost_equal(d._w_sum, numpy.zeros(d.d))
	assert_array_almost_equal(d._xw_sum, numpy.zeros(d.d))

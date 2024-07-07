# test_utils.py
# Contact: Jacob Schreiber <jmschreiber91@gmail.com>

import numpy
import torch
import pytest

from pomegranate._utils import _cast_as_tensor
from pomegranate._utils import _update_parameter
from pomegranate._utils import _check_parameter
from pomegranate._utils import partition_sequences

from .tools import assert_almost_equal
from .tools import assert_equal
from .tools import assert_raises
from numpy.testing import assert_array_equal
from numpy.testing import assert_array_almost_equal


def _test_cast(y, dtype, ndim):
	x = _cast_as_tensor(y)
	assert isinstance(x, torch.Tensor)
	assert x.dtype == dtype
	assert x.ndim == ndim

	if type(x) not in (numpy.ndarray, torch.Tensor) and ndim == 0:
		if type(x) in float:
			assert_almost_equal(x, y)
		else:
			assert_equal(x, y)
	else:
		assert_array_almost_equal(x, y)


def test_cast_as_tensor_bool():
	_test_cast(False, torch.bool, 0)
	_test_cast(True, torch.bool, 0)


def test_cast_as_tensor_int():
	_test_cast(5, torch.int64, 0)
	_test_cast(0, torch.int64, 0)
	_test_cast(-1, torch.int64, 0)


def test_cast_as_tensor_float():
	_test_cast(1.2, torch.float32, 0)
	_test_cast(0.0, torch.float32, 0)
	_test_cast(-8.772, torch.float32, 0)


def test_cast_as_tensor_numpy_bool():
	_test_cast(numpy.array(False), torch.bool, 0)
	_test_cast(numpy.array(True), torch.bool, 0)


def test_cast_as_tensor_numpy_int():
	_test_cast(numpy.array(5), torch.int64, 0)
	_test_cast(numpy.array(0), torch.int64, 0)
	_test_cast(numpy.array(-1), torch.int64, 0)


def test_cast_as_tensor_numpy_float():
	_test_cast(numpy.array(1.2), torch.float64, 0)
	_test_cast(numpy.array(0.0), torch.float64, 0)
	_test_cast(numpy.array(-8.772), torch.float64, 0)


def test_cast_as_tensor_torch_bool():
	_test_cast(torch.tensor(False), torch.bool, 0)
	_test_cast(torch.tensor(True), torch.bool, 0)


def test_cast_as_tensor_torch_int():
	_test_cast(torch.tensor(5), torch.int64, 0)
	_test_cast(torch.tensor(0), torch.int64, 0)
	_test_cast(torch.tensor(-1), torch.int64, 0)


def test_cast_as_tensor_torch_float():
	_test_cast(torch.tensor(1.2), torch.float32, 0)
	_test_cast(torch.tensor(0.0), torch.float32, 0)
	_test_cast(torch.tensor(-8.772), torch.float32, 0)


def test_cast_as_tensor_check_wrong():
	assert_raises(AssertionError, _test_cast, True, torch.int64, 0)
	assert_raises(AssertionError, _test_cast, True, torch.bool, 1)
	assert_raises(AssertionError, _test_cast, 1, torch.int32, 0)
	assert_raises(AssertionError, _test_cast, 1, torch.int64, 1)
	assert_raises(AssertionError, _test_cast, 1.2, torch.int64, 0)
	assert_raises(AssertionError, _test_cast, 1.2, torch.float64, 1)



def test_cast_as_tensor_numpy_bool_1d():
	_test_cast(numpy.array([True, False, True, True]), torch.bool, 1)
	_test_cast(numpy.array([True, True, True]), torch.bool, 1)
	_test_cast(numpy.array([False]), torch.bool, 1)


def test_cast_as_tensor_numpy_int_1d():
	_numpy_dtypes = numpy.int32, numpy.int64
	_torch_dtypes = torch.int32, torch.int64

	for _dtype1, _dtype2 in zip(_numpy_dtypes, _torch_dtypes):
		_test_cast(numpy.array([1, 2, 3], dtype=_dtype1), _dtype2, 1)
		_test_cast(numpy.array([0, -3, 0], dtype=_dtype1), _dtype2, 1)
		_test_cast(numpy.array([0], dtype=_dtype1), _dtype2, 1)


def test_cast_as_tensor_numpy_float_1d():
	_numpy_dtypes = numpy.float16, numpy.float32, numpy.float64
	_torch_dtypes = torch.float16, torch.float32, torch.float64

	for _dtype1, _dtype2 in zip(_numpy_dtypes, _torch_dtypes):
		_test_cast(numpy.array([1.2, 2.0, 3.1], dtype=_dtype1), _dtype2, 1)
		_test_cast(numpy.array([0.0, -3.0, 0.0], dtype=_dtype1), _dtype2, 1)
		_test_cast(numpy.array([0.0], dtype=_dtype1), _dtype2, 1)


def test_cast_as_tensor_numpy_check_wrong_1d():
	x1 = numpy.array([True, True, False])
	x2 = numpy.array([1, 2, 3])
	x3 = numpy.array([1.0, 1.1, 1.1])

	assert_raises(AssertionError, _test_cast, x1, torch.int64, 1)
	assert_raises(AssertionError, _test_cast, x1, torch.bool, 0)
	assert_raises(AssertionError, _test_cast, x2, torch.int32, 1)
	assert_raises(AssertionError, _test_cast, x2, torch.int64, 0)
	assert_raises(AssertionError, _test_cast, x3, torch.int64, 1)
	assert_raises(AssertionError, _test_cast, x3, torch.float64, 0)



def test_cast_as_tensor_numpy_bool_2d():
	_test_cast(numpy.array([[True, False, True], [True, True, False]]), 
		torch.bool, 2)
	_test_cast(numpy.array([[True, True, True]]), torch.bool, 2)
	_test_cast(numpy.array([[False]]), torch.bool, 2)


def test_cast_as_tensor_numpy_int_2d():
	_numpy_dtypes = numpy.int32, numpy.int64
	_torch_dtypes = torch.int32, torch.int64

	for _dtype1, _dtype2 in zip(_numpy_dtypes, _torch_dtypes):
		_test_cast(numpy.array([[1, 2, 3], [3, 2, 1]], dtype=_dtype1), 
			_dtype2, 2)
		_test_cast(numpy.array([[0, -3, 0]], dtype=_dtype1), _dtype2, 2)
		_test_cast(numpy.array([[0]], dtype=_dtype1), _dtype2, 2)


def test_cast_as_tensor_numpy_float_2d():
	_numpy_dtypes = numpy.float16, numpy.float32, numpy.float64
	_torch_dtypes = torch.float16, torch.float32, torch.float64

	for _dtype1, _dtype2 in zip(_numpy_dtypes, _torch_dtypes):
		_test_cast(numpy.array([[1.2, 2.0, 3.1], [-1.4, -1.3, -0.0]], 
			dtype=_dtype1), _dtype2, 2)
		_test_cast(numpy.array([[0.0, -3.0, 0.0]], dtype=_dtype1), _dtype2, 2)
		_test_cast(numpy.array([[0.0]], dtype=_dtype1), _dtype2, 2)


def test_cast_as_tensor_numpy_check_wrong_2d():
	x1 = numpy.array([[True, True, False]])
	x2 = numpy.array([[1, 2, 3]])
	x3 = numpy.array([[1.0, 1.1, 1.1]])

	assert_raises(AssertionError, _test_cast, x1, torch.int64, 2)
	assert_raises(AssertionError, _test_cast, x1, torch.bool, 1)
	assert_raises(AssertionError, _test_cast, x2, torch.int32, 2)
	assert_raises(AssertionError, _test_cast, x2, torch.int64, 1)
	assert_raises(AssertionError, _test_cast, x3, torch.int64, 2)
	assert_raises(AssertionError, _test_cast, x3, torch.float64, 1)



def test_cast_as_tensor_torch_bool_1d():
	_test_cast(torch.tensor([True, False, True, True]), torch.bool, 1)
	_test_cast(torch.tensor([True, True, True]), torch.bool, 1)
	_test_cast(torch.tensor([False]), torch.bool, 1)


def test_cast_as_tensor_torch_int_1d():
	_torch_dtypes = torch.int32, torch.int64

	for _dtype in _torch_dtypes:
		_test_cast(torch.tensor([1, 2, 3], dtype=_dtype), _dtype, 1)
		_test_cast(torch.tensor([0, -3, 0], dtype=_dtype), _dtype, 1)
		_test_cast(torch.tensor([0], dtype=_dtype), _dtype, 1)


def test_cast_as_tensor_torch_float_1d():
	_torch_dtypes = torch.float16, torch.float32, torch.float64

	for _dtype in _torch_dtypes:
		_test_cast(torch.tensor([1.2, 2.0, 3.1], dtype=_dtype), _dtype, 1)
		_test_cast(torch.tensor([0.0, -3.0, 0.0], dtype=_dtype), _dtype, 1)
		_test_cast(torch.tensor([0.0], dtype=_dtype), _dtype, 1)


def test_cast_as_tensor_torch_check_wrong_1d():
	x1 = torch.tensor([True, True, False])
	x2 = torch.tensor([1, 2, 3])
	x3 = torch.tensor([1.0, 1.1, 1.1])

	assert_raises(AssertionError, _test_cast, x1, torch.int64, 1)
	assert_raises(AssertionError, _test_cast, x1, torch.bool, 0)
	assert_raises(AssertionError, _test_cast, x2, torch.int32, 1)
	assert_raises(AssertionError, _test_cast, x2, torch.int64, 0)
	assert_raises(AssertionError, _test_cast, x3, torch.int64, 1)
	assert_raises(AssertionError, _test_cast, x3, torch.float64, 0)



def test_cast_as_tensor_torch_bool_2d():
	_test_cast(torch.tensor([[True, False, True], [True, True, False]]), 
		torch.bool, 2)
	_test_cast(torch.tensor([[True, True, True]]), torch.bool, 2)
	_test_cast(torch.tensor([[False]]), torch.bool, 2)


def test_cast_as_tensor_torch_int_2d():
	_torch_dtypes = torch.int32, torch.int64

	for _dtype in _torch_dtypes:
		_test_cast(torch.tensor([[1, 2, 3], [3, 2, 1]], dtype=_dtype), 
			_dtype, 2)
		_test_cast(torch.tensor([[0, -3, 0]], dtype=_dtype), _dtype, 2)
		_test_cast(torch.tensor([[0]], dtype=_dtype), _dtype, 2)


def test_cast_as_tensor_torch_float_2d():
	_torch_dtypes = torch.float16, torch.float32, torch.float64

	for _dtype in _torch_dtypes:
		_test_cast(torch.tensor([[1.2, 2.0, 3.1], [-1.4, -1.3, -0.0]], 
			dtype=_dtype), _dtype, 2)
		_test_cast(torch.tensor([[0.0, -3.0, 0.0]], dtype=_dtype), _dtype, 2)
		_test_cast(torch.tensor([[0.0]], dtype=_dtype), _dtype, 2)


def test_cast_as_tensor_torch_check_wrong_2d():
	x1 = torch.tensor([[True, True, False]])
	x2 = torch.tensor([[1, 2, 3]])
	x3 = torch.tensor([[1.0, 1.1, 1.1]])

	assert_raises(AssertionError, _test_cast, x1, torch.int64, 2)
	assert_raises(AssertionError, _test_cast, x1, torch.bool, 1)
	assert_raises(AssertionError, _test_cast, x2, torch.int32, 2)
	assert_raises(AssertionError, _test_cast, x2, torch.int64, 1)
	assert_raises(AssertionError, _test_cast, x3, torch.int64, 2)
	assert_raises(AssertionError, _test_cast, x3, torch.float64, 1)



#

def _test_update(inertia):
	x1 = torch.tensor([1.0, 1.4, 1.8, -1.1, 0.0])
	x2 = torch.tensor([2.2, 8.2, 0.1, 105.2, 0.0])
	y = x1 * inertia + x2 * (1-inertia) 

	_update_parameter(x1, x2, inertia=inertia)
	assert_array_almost_equal(x1, y)


def test_update_parameter():
	_test_update(inertia=0.0)


def test_update_parameter_inertia():
	_test_update(inertia=0.1)
	_test_update(inertia=0.5)
	_test_update(inertia=1.0)



#



def test_check_parameters_min_values_bool():
	x = torch.tensor([True, True, False], dtype=torch.bool)
	dtypes = [torch.bool]

	_check_parameter(x, "x", min_value=0)
	_check_parameter(x, "x", min_value=-1.0)

	assert_raises(ValueError, _check_parameter, x, "x", min_value=1)
	assert_raises(ValueError, _check_parameter, x, "x", min_value=1000.0)


def test_check_parameters_min_values_int():
	x = torch.tensor([1, 6, 24], dtype=torch.int32)
	dtypes = [torch.bool]

	_check_parameter(x, "x", min_value=1)
	_check_parameter(x, "x", min_value=-1.0)

	assert_raises(ValueError, _check_parameter, x, "x", min_value=2)
	assert_raises(ValueError, _check_parameter, x, "x", min_value=25.0)


def test_check_parameters_min_values_float():
	x = torch.tensor([1, 6, 24], dtype=torch.float32)
	dtypes = [torch.bool]

	_check_parameter(x, "x", min_value=1)
	_check_parameter(x, "x", min_value=-1.0)

	assert_raises(ValueError, _check_parameter, x, "x", min_value=2)
	assert_raises(ValueError, _check_parameter, x, "x", min_value=25.0)


def test_check_parameters_max_values_bool():
	x = torch.tensor([True, True, False], dtype=torch.bool)
	dtypes = [torch.bool]

	_check_parameter(x, "x", max_value=1)
	_check_parameter(x, "x", max_value=100.0)

	assert_raises(ValueError, _check_parameter, x, "x", max_value=0)
	assert_raises(ValueError, _check_parameter, x, "x", max_value=-2.7)


def test_check_parameters_max_values_int():
	x = torch.tensor([1, 6, 24], dtype=torch.int32)
	dtypes = [torch.bool]

	_check_parameter(x, "x", max_value=24)
	_check_parameter(x, "x", max_value=24.1)

	assert_raises(ValueError, _check_parameter, x, "x", max_value=2)
	assert_raises(ValueError, _check_parameter, x, "x", max_value=8.0)


def test_check_parameters_max_values_float():
	x = torch.tensor([1, 6, 24], dtype=torch.float32)
	dtypes = [torch.bool]

	_check_parameter(x, "x", max_value=24)
	_check_parameter(x, "x", max_value=24.1)

	assert_raises(ValueError, _check_parameter, x, "x", max_value=2)
	assert_raises(ValueError, _check_parameter, x, "x", max_value=8.0)


def test_check_parameters_minmax_values_float():
	x = torch.tensor([1.1, 2.3, 7.8], dtype=torch.float32)
	dtypes = [torch.bool]

	_check_parameter(x, "x", min_value=1.0, max_value=24)

	assert_raises(ValueError, _check_parameter, x, "x", min_value=1.2, 
		max_value=24)
	assert_raises(ValueError, _check_parameter, x, "x", min_value=0.0,
		max_value=6)


def test_check_parameters_value_sum_float():
	x = torch.tensor([1.1, 2.3, 7.8], dtype=torch.float32)
	_check_parameter(x, "x", value_sum=torch.sum(x))

	assert_raises(ValueError, _check_parameter, x, "x", value_sum=-0.2)
	assert_raises(ValueError, _check_parameter, x, "x", value_sum=torch.sum(x)
		+2e-6)


	x = 1.5
	_check_parameter(x, "x", value_sum=x)

	assert_raises(ValueError, _check_parameter, x, "x", value_sum=-0.2)
	assert_raises(ValueError, _check_parameter, x, "x", value_sum=x+2e-6)


def test_check_parameters_value_set_bool():
	x = torch.tensor([True, True, True], dtype=torch.bool)
	value_set = [True]

	_check_parameter(x, "x", value_set=tuple(value_set))
	_check_parameter(x, "x", value_set=list(value_set))

	assert_raises(ValueError, _check_parameter, x, "x", value_set=[False])
	assert_raises(ValueError, _check_parameter, x, "x", value_set=[5.2])


def test_check_parameters_value_set_int():
	x = torch.tensor([2, 6, 24], dtype=torch.int32)
	value_set = [2, 6, 24, 26]

	_check_parameter(x, "x", value_set=tuple(value_set))
	_check_parameter(x, "x", value_set=list(value_set))

	assert_raises(ValueError, _check_parameter, x, "x", value_set=[True, False])
	assert_raises(ValueError, _check_parameter, x, "x", value_set=[5.2, 1, 6])


#def test_check_parameters_value_set_float():
#	x = torch.tensor([1.1, 6.0, 24.3], dtype=torch.float32)
#	value_set = [1.1, 6.0, 24.3, 17.8]
#
#	_check_parameter(x, "x", value_set=tuple(value_set))
#	_check_parameter(x, "x", value_set=list(value_set))
#
#	assert_raises(ValueError, _check_parameter, x, "x", value_set=[True, False])
#	assert_raises(ValueError, _check_parameter, x, "x", value_set=[5.2, 1, 6])


def test_check_parameters_dtypes_bool():
	x = torch.tensor([True, True, False], dtype=torch.bool)
	dtypes = [torch.bool]

	_check_parameter(x, "x", dtypes=tuple(dtypes))
	_check_parameter(x, "x", dtypes=list(dtypes))
	_check_parameter(x, "x", dtypes=set(dtypes))

	assert_raises(ValueError, _check_parameter, x, "x", dtypes=[torch.int64])
	assert_raises(ValueError, _check_parameter, x, "x", dtypes=[torch.float64])


def test_check_parameters_dtypes_int():
	x = torch.tensor([1, 2, 3], dtype=torch.int32)
	dtypes = [torch.int32, torch.int64]

	_check_parameter(x, "x", dtypes=tuple(dtypes))
	_check_parameter(x, "x", dtypes=list(dtypes))
	_check_parameter(x, "x", dtypes=set(dtypes))

	assert_raises(ValueError, _check_parameter, x, "x", dtypes=[torch.int64])
	assert_raises(ValueError, _check_parameter, x, "x", dtypes=[torch.float32])


def test_check_parameters_dtypes_float():
	x = torch.tensor([1, 2, 3], dtype=torch.float32)
	dtypes = [torch.float32, torch.float64]

	_check_parameter(x, "x", dtypes=tuple(dtypes))
	_check_parameter(x, "x", dtypes=list(dtypes))
	_check_parameter(x, "x", dtypes=set(dtypes))

	assert_raises(ValueError, _check_parameter, x, "x", dtypes=[torch.int64])
	assert_raises(ValueError, _check_parameter, x, "x", dtypes=[torch.float64])


def test_check_parameters_ndim_0():
	x = torch.tensor(1.1)

	_check_parameter(x, "x", ndim=0)
	assert_raises(ValueError, _check_parameter, x, "x", ndim=1)
	assert_raises(ValueError, _check_parameter, x, "x", ndim=2)


def test_check_parameters_ndim_1():
	x = torch.tensor([1.1])

	_check_parameter(x, "x", ndim=1)
	assert_raises(ValueError, _check_parameter, x, "x", ndim=0)
	assert_raises(ValueError, _check_parameter, x, "x", ndim=2)


def test_check_parameters_ndim_2():
	x = torch.tensor([[1.1]])

	_check_parameter(x, "x", ndim=2)
	assert_raises(ValueError, _check_parameter, x, "x", ndim=1)
	assert_raises(ValueError, _check_parameter, x, "x", ndim=0)


def test_check_parameters_ndim_tuple():
	x = torch.tensor([1.1])

	_check_parameter(x, "x", ndim=(1,))
	_check_parameter(x, "x", ndim=(0, 1))
	_check_parameter(x, "x", ndim=(1, 2))

	assert_raises(ValueError, _check_parameter, x, "x", ndim=0)
	assert_raises(ValueError, _check_parameter, x, "x", ndim=(0, 2))

	x = torch.tensor(1.1)

	_check_parameter(x, "x", ndim=(0,))
	_check_parameter(x, "x", ndim=(0, 1))
	_check_parameter(x, "x", ndim=(0, 2))

	assert_raises(ValueError, _check_parameter, x, "x", ndim=1)
	assert_raises(ValueError, _check_parameter, x, "x", ndim=(1, 2))

	x = torch.tensor([[1.1]])

	_check_parameter(x, "x", ndim=(2,))
	_check_parameter(x, "x", ndim=(2, 1))
	_check_parameter(x, "x", ndim=(0, 2))

	assert_raises(ValueError, _check_parameter, x, "x", ndim=1)
	assert_raises(ValueError, _check_parameter, x, "x", ndim=(0, 1))


def test_check_parameters_ndim_list():
	x = torch.tensor([1.1])

	_check_parameter(x, "x", ndim=[1])
	_check_parameter(x, "x", ndim=[0, 1])
	_check_parameter(x, "x", ndim=[1, 2])

	assert_raises(ValueError, _check_parameter, x, "x", ndim=0)
	assert_raises(ValueError, _check_parameter, x, "x", ndim=[0, 2])

	x = torch.tensor(1.1)

	_check_parameter(x, "x", ndim=[0])
	_check_parameter(x, "x", ndim=[0, 1])
	_check_parameter(x, "x", ndim=[0, 2])

	assert_raises(ValueError, _check_parameter, x, "x", ndim=1)
	assert_raises(ValueError, _check_parameter, x, "x", ndim=[1, 2])

	x = torch.tensor([[1.1]])

	_check_parameter(x, "x", ndim=[2])
	_check_parameter(x, "x", ndim=[2, 1])
	_check_parameter(x, "x", ndim=[0, 2])

	assert_raises(ValueError, _check_parameter, x, "x", ndim=1)
	assert_raises(ValueError, _check_parameter, x, "x", ndim=[0, 1])


def test_check_parameters_shape():
	x = torch.tensor([[1.1]])

	_check_parameter(x, "x", shape=(1, 1))
	_check_parameter(x, "x", shape=(-1, 1))
	_check_parameter(x, "x", shape=(1, -1))
	_check_parameter(x, "x", shape=(-1, -1))
	assert_raises(ValueError, _check_parameter, x, "x", shape=(2, 1))
	assert_raises(ValueError, _check_parameter, x, "x", shape=(1, 2))
	assert_raises(ValueError, _check_parameter, x, "x", shape=(1,))
	assert_raises(ValueError, _check_parameter, x, "x", shape=(1, 2, 1))
	assert_raises(ValueError, _check_parameter, x, "x", shape=(1, 2, -1))

	x = torch.tensor([
		[1.1, 1.2, 1.3, 1.8],
		[2.1, 1.1, 1.4, 0.9]
	])

	_check_parameter(x, "x", shape=(2, 4))
	_check_parameter(x, "x", shape=(-1, 4))
	_check_parameter(x, "x", shape=(2, -1))
	_check_parameter(x, "x", shape=(-1, -1))
	assert_raises(ValueError, _check_parameter, x, "x", shape=(2, 1))
	assert_raises(ValueError, _check_parameter, x, "x", shape=(1, 2))
	assert_raises(ValueError, _check_parameter, x, "x", shape=(1,))
	assert_raises(ValueError, _check_parameter, x, "x", shape=(1, 2, 1))
	assert_raises(ValueError, _check_parameter, x, "x", shape=(1, 2, -1))
	assert_raises(ValueError, _check_parameter, x, "x", shape=(2, -1, -1))


###


@pytest.fixture
def X1():
	return [[[0.5, 0.2], 
		     [0.2, 0.1]],

		    [[0.1, 0.2],
		     [0.7, 0.2]],
 
		    [[0.1, 0.4],
		     [0.3, 0.1]],

		    [[0.3, 0.1],
		     [0.1, 0.6]]]


@pytest.fixture
def w1():
	return [[0.1, 0.5], [0.1, 0.3], [0.5, 0.2], [0.2, 0.1]]


@pytest.fixture
def p1():
	return [[[0.5, 0.5], 
		  [0.5, 0.5]],

		 [[0.7, 0.3],
		  [0.3, 0.7]],

		 [[1.0, 0.0],
		  [0.5, 0.5]],

		 [[0.0, 1.0],
		  [0.0, 1.0]]]


def test_partition_3d_X(X1):
	funcs = (lambda x: x, tuple, numpy.array, 
		lambda x: torch.from_numpy(numpy.array(x)))

	for func in funcs:
		y, _, _ = partition_sequences(func(X1))

		assert isinstance(y, list)
		assert len(y) == 1

		assert isinstance(y[0], torch.Tensor)
		assert y[0].ndim == 3
		assert y[0].shape == (4, 2, 2)
		assert_array_almost_equal(X1, y[0])


def test_partition_3d_Xw(X1, w1):
	funcs = (lambda x: x, tuple, numpy.array, 
		lambda x: torch.from_numpy(numpy.array(x)))

	for func in funcs:
		y, w, p = partition_sequences(func(X1), sample_weight=func(w1))

		assert isinstance(y, list)
		assert len(y) == 1

		assert isinstance(w, list)
		assert len(w) == 1

		assert p is None

		assert isinstance(y[0], torch.Tensor)
		assert y[0].ndim == 3
		assert y[0].shape == (4, 2, 2)
		assert_array_almost_equal(X1, y[0])

		assert isinstance(w[0], torch.Tensor)
		assert w[0].ndim == 2
		assert w[0].shape == (4, 2)
		assert_array_almost_equal(w1, w[0])


def test_partition_3d_Xp(X1, p1):
	funcs = (lambda x: x, tuple, numpy.array, 
		lambda x: torch.from_numpy(numpy.array(x)))

	for func in funcs:
		y, w, p = partition_sequences(func(X1), priors=func(p1), n_dists=2)

		assert isinstance(y, list)
		assert len(y) == 1

		assert w is None

		assert isinstance(p, list)
		assert len(p) == 1

		assert isinstance(y[0], torch.Tensor)
		assert y[0].ndim == 3
		assert y[0].shape == (4, 2, 2)
		assert_array_almost_equal(X1, y[0])

		assert isinstance(p[0], torch.Tensor)
		assert p[0].ndim == 3
		assert p[0].shape == (4, 2, 2)
		assert_array_almost_equal(p1, p[0])


def test_partition_3d_Xwp(X1, w1, p1):
	funcs = (lambda x: x, tuple, numpy.array, 
		lambda x: torch.from_numpy(numpy.array(x)))

	for func in funcs:
		y, w, p = partition_sequences(func(X1), sample_weight=func(w1),
									  priors=func(p1), n_dists=2)

		assert isinstance(y, list)
		assert len(y) == 1

		assert isinstance(w, list)
		assert len(w) == 1

		assert isinstance(p, list)
		assert len(p) == 1

		assert isinstance(y[0], torch.Tensor)
		assert y[0].ndim == 3
		assert y[0].shape == (4, 2, 2)
		assert_array_almost_equal(X1, y[0])

		assert isinstance(w[0], torch.Tensor)
		assert w[0].ndim == 2
		assert w[0].shape == (4, 2)
		assert_array_almost_equal(w1, w[0])

		assert isinstance(p[0], torch.Tensor)
		assert p[0].ndim == 3
		assert p[0].shape == (4, 2, 2)
		assert_array_almost_equal(p1, p[0])


@pytest.fixture
def X2():
	return [[[[0.5, 0.2], 
		      [0.2, 0.1]],

		     [[0.1, 0.2],
		      [0.7, 0.2]]],

		    [[[0.1, 0.4],
		      [0.3, 0.1],
		      [0.1, 0.7]],

		     [[0.3, 0.1],
		      [0.1, 0.6],
		      [0.1, 0.1]]]]


@pytest.fixture
def w2():
	return [[[0.1, 0.5], [0.1, 0.3]], [[0.5, 0.2, 0.1], [0.2, 0.1, 0.0]]]


@pytest.fixture
def p2():
	return [[[[0.5, 0.5], 
		      [1.0, 0.0]],

		     [[0.5, 0.5],
		      [0.5, 0.5]]],

		    [[[0.5, 0.5],
		      [0.7, 0.3],
		      [0.1, 0.9]],

		     [[1.0, 0.0],
		      [1.0, 0.0],
		      [1.0, 0.0]]]]


def test_partition_3ds_X(X2):
	funcs = (lambda x: x, tuple, numpy.array, 
		lambda x: torch.from_numpy(numpy.array(x)))

	for func in funcs:
		X2_ = [func(x) for x in X2]

		y, _, _ = partition_sequences(X2)

		assert isinstance(y, list)
		assert len(y) == 2

		for i, y_ in enumerate(y):
			assert isinstance(y_, torch.Tensor)
			assert y_.ndim == 3
			assert y_.shape == (2, i+2, 2)
			assert_array_almost_equal(y_, X2[i])


def test_partition_3ds_Xw(X2, w2):
	funcs = (lambda x: x, tuple, numpy.array, 
		lambda x: torch.from_numpy(numpy.array(x)))

	for func in funcs:
		X2_ = [func(x) for x in X2]
		w2_ = [func(w) for w in w2]

		y, w, p = partition_sequences(X2_, sample_weight=w2_) 

		assert isinstance(y, list)
		assert len(y) == 2

		assert isinstance(w, list)
		assert len(w) == 2

		assert p == None

		for i, y_ in enumerate(y):
			assert isinstance(y_, torch.Tensor)
			assert y_.ndim == 3
			assert y_.shape == (2, i+2, 2)
			assert_array_almost_equal(y_, X2[i])

		for i, w_ in enumerate(w):
			assert isinstance(w_, torch.Tensor)
			assert w_.ndim == 2
			assert w_.shape == (2, i+2)
			assert_array_almost_equal(w_, w2[i])


def test_partition_3ds_Xp(X2, p2):
	funcs = (lambda x: x, tuple, numpy.array, 
		lambda x: torch.from_numpy(numpy.array(x)))

	for func in funcs:
		X2_ = [func(x) for x in X2]
		p2_ = [func(p) for p in p2]

		y, w, p = partition_sequences(X2_, priors=p2_, n_dists=2)

		assert isinstance(y, list)
		assert len(y) == 2

		assert isinstance(p, list)
		assert len(p) == 2

		assert w is None

		for i, y_ in enumerate(y):
			assert isinstance(y_, torch.Tensor)
			assert y_.ndim == 3
			assert y_.shape == (2, i+2, 2)
			assert_array_almost_equal(y_, X2[i])

		for i, p_ in enumerate(p):
			assert isinstance(p_, torch.Tensor)
			assert p_.ndim == 3
			assert p_.shape == (2, i+2, 2)
			assert_array_almost_equal(p_, p2[i])


def test_partition_3ds_Xwp(X2, w2, p2):
	funcs = (lambda x: x, tuple, numpy.array, 
		lambda x: torch.from_numpy(numpy.array(x)))

	for func in funcs:
		X2_ = [func(x) for x in X2]
		w2_ = [func(w) for w in w2]
		p2_ = [func(p) for p in p2]

		y, w, p = partition_sequences(X2_,
			sample_weight=w2_, priors=p2_, n_dists=2)

		assert isinstance(y, list)
		assert len(y) == 2

		assert isinstance(p, list)
		assert len(p) == 2

		assert isinstance(w, list)
		assert len(w) == 2

		for i, y_ in enumerate(y):
			assert isinstance(y_, torch.Tensor)
			assert y_.ndim == 3
			assert y_.shape == (2, i+2, 2)
			assert_array_almost_equal(y_, X2[i])

		for i, w_ in enumerate(w):
			assert isinstance(w_, torch.Tensor)
			assert w_.ndim == 2
			assert w_.shape == (2, i+2)
			assert_array_almost_equal(w_, w2[i])

		for i, p_ in enumerate(p):
			assert isinstance(p_, torch.Tensor)
			assert p_.ndim == 3
			assert p_.shape == (2, i+2, 2)
			assert_array_almost_equal(p_, p2[i])


@pytest.fixture
def X3():
	return [[[0.5, 0.2], 
		     [0.2, 0.1]],

		    [[0.1, 0.2],
		     [0.7, 0.2]],

		    [[0.1, 0.4],
		     [0.3, 0.1],
		     [0.1, 0.7]],

		    [[0.3, 0.1],
		     [0.1, 0.6],
		     [0.1, 0.1]],

		    [[0.5, 0.1]],

		    [[0.5, 0.2],
		     [0.3, 0.4]]]


@pytest.fixture
def w3():
	return [[0.1, 0.5], [0.1, 0.3], [0.5, 0.2, 0.1], [0.2, 0.1, 0.3],
		[0.1], [0.6, 0.4]]


@pytest.fixture
def p3():
	return [[[0.5, 0.5], 
		     [0.5, 0.5]],

		    [[0.9, 0.1],
		     [0.0, 1.0]],

		    [[0.5, 0.5],
		     [0.5, 0.5],
		     [0.0, 1.0]],

		    [[0.0, 1.0],
		     [0.0, 1.0],
		     [1.0, 0.0]],

		    [[0.5, 0.5]],

		    [[0.7, 0.3],
		     [0.3, 0.7]]]


def test_partition_2ds_X(X3):
	X_batches = [
		[[[0.5, 0.1]]],

		[[[0.5, 0.2], 
		  [0.2, 0.1]],

		 [[0.1, 0.2],
		  [0.7, 0.2]],

		 [[0.5, 0.2],
		  [0.3, 0.4]]],

		[[[0.1, 0.4],
		  [0.3, 0.1],
		  [0.1, 0.7]],

		  [[0.3, 0.1],
		   [0.1, 0.6],
		   [0.1, 0.1]]]]


	funcs = (lambda x: x, tuple, numpy.array, 
		lambda x: torch.from_numpy(numpy.array(x)))

	for func in funcs:
		X3_ = [func(x) for x in X3]

		y, _, _ = partition_sequences(X3_)

		assert isinstance(y, list)
		assert len(y) == 3

		for i, y_ in enumerate(y):
			assert isinstance(y_, torch.Tensor)
			assert y_.ndim == 3
			assert y_.shape == ([1, 3, 2][i], i+1, 2)
			assert_array_almost_equal(y_, X_batches[i])


def test_partition_2ds_Xw(X3, w3):
	w_batches = [[[0.1]], [[0.1, 0.5], [0.1, 0.3], [0.6, 0.4]], 
		[[0.5, 0.2, 0.1], [0.2, 0.1, 0.3]]]


	funcs = (lambda x: x, tuple, numpy.array, 
		lambda x: torch.from_numpy(numpy.array(x)))

	for func in funcs:
		X3_ = [func(x) for x in X3]
		w3_ = [func(w) for w in w3]

		y, w, p = partition_sequences(X3_, sample_weight=w3_) 

		assert isinstance(y, list)
		assert len(y) == 3

		assert isinstance(w, list)
		assert len(w) == 3

		assert p == None

		for i, y_ in enumerate(y):
			assert isinstance(y_, torch.Tensor)
			assert y_.ndim == 3
			assert y_.shape == ([1, 3, 2][i], i+1, 2)

		for i, w_ in enumerate(w):
			assert isinstance(w_, torch.Tensor)
			assert w_.ndim == 2
			assert w_.shape == ([1, 3, 2][i], i+1)
			assert_array_almost_equal(w_, w_batches[i])


def test_partition_2ds_Xp(X3, p3):
	p_batches = [
		[[[0.5, 0.5]]],

		[[[0.5, 0.5], 
		  [0.5, 0.5]],

		 [[0.9, 0.1],
		  [0.0, 1.0]],

		 [[0.7, 0.3],
		  [0.3, 0.7]]],

		[[[0.5, 0.5],
		  [0.5, 0.5],
		  [0.0, 1.0]],

		 [[0.0, 1.0],
		  [0.0, 1.0],
		  [1.0, 0.0]]]]

	funcs = (lambda x: x, tuple, numpy.array, 
		lambda x: torch.from_numpy(numpy.array(x)))

	for func in funcs:
		X3_ = [func(x) for x in X3]
		p3_ = [func(p) for p in p3]

		y, w, p = partition_sequences(X3_, priors=p3_, n_dists=2)

		assert isinstance(y, list)
		assert len(y) == 3

		assert isinstance(p, list)
		assert len(p) == 3

		assert w is None

		for i, y_ in enumerate(y):
			assert isinstance(y_, torch.Tensor)
			assert y_.ndim == 3
			assert y_.shape == ([1, 3, 2][i], i+1, 2)

		for i, p_ in enumerate(p):
			assert isinstance(p_, torch.Tensor)
			assert p_.ndim == 3
			assert p_.shape == ([1, 3, 2][i], i+1, 2)
			assert_array_almost_equal(p_, p_batches[i])


def test_partition_2ds_Xwp(X3, w3, p3):
	funcs = (lambda x: x, tuple, numpy.array, 
		lambda x: torch.from_numpy(numpy.array(x)))

	for func in funcs:
		X3_ = [func(x) for x in X3]
		w3_ = [func(w) for w in w3]
		p3_ = [func(p) for p in p3]

		y, w, p = partition_sequences(X3_, sample_weight=w3_, priors=p3_, n_dists=2)

		assert isinstance(y, list)
		assert len(y) == 3

		assert isinstance(p, list)
		assert len(w) == 3

		assert isinstance(p, list)
		assert len(p) == 3

		for i, y_ in enumerate(y):
			assert isinstance(y_, torch.Tensor)
			assert y_.ndim == 3
			assert y_.shape == ([1, 3, 2][i], i+1, 2)

		for i, w_ in enumerate(w):
			assert isinstance(w_, torch.Tensor)
			assert w_.ndim == 2
			assert w_.shape == ([1, 3, 2][i], i+1)

		for i, p_ in enumerate(p):
			assert isinstance(p_, torch.Tensor)
			assert p_.ndim == 3
			assert p_.shape == ([1, 3, 2][i], i+1, 2)


@pytest.mark.parametrize("X", [torch.ones((1, 10, 2)), [torch.ones((1, 10, 2)), torch.ones((2, 10, 2))]])
@pytest.mark.parametrize("invalid", ["sample_weight", "priors"])
def test_dont_hide_errors_for_priors_and_sample_weight(X, invalid):
	"""Test that we get the correct error message when we don't pass data in case 3."""

	with pytest.raises(ValueError) as excinfo:
		partition_sequences(X, **{invalid: numpy.zeros((1, 1)) - 1}, n_dists=10)

	assert invalid in str(excinfo.value)
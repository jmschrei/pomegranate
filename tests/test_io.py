from pomegranate import *
from pomegranate.io import DataGenerator
from pomegranate.io import SequenceGenerator
from pomegranate.io import DataFrameGenerator

from nose.tools import with_setup
from nose.tools import assert_true
from nose.tools import assert_equal
from nose.tools import assert_raises
from numpy.testing import assert_almost_equal
from numpy.testing import assert_array_equal
from numpy.testing import assert_array_almost_equal

import random
import numpy
import pandas

numpy.random.seed(0)
random.seed(0)

nan = numpy.nan

def test_io_datagenerator_shape():
	X = numpy.random.randn(500, 13)
	data = DataGenerator(X)

	assert_array_equal(data.shape, X.shape)

def test_io_datagenerator_classes_fail():
	X = numpy.random.randn(500, 13)
	data = DataGenerator(X)

	assert_raises(ValueError, lambda data: data.classes, data)

def test_io_datagenerator_classes():
	X = numpy.random.randn(500, 13)
	y = numpy.random.randint(5, size=500)
	data = DataGenerator(X, y=y)

	assert_array_equal(data.classes, [0, 1, 2, 3, 4])

def test_io_datagenerator_x_batches():
	X = numpy.random.randn(500, 13)
	w = numpy.ones(500)

	data = DataGenerator(X)
	X_ = numpy.concatenate([batch[0] for batch in data.batches()])
	w_ = numpy.concatenate([batch[1] for batch in data.batches()])
	assert_almost_equal(X, X_)
	assert_almost_equal(w, w_)

	data = DataGenerator(X, batch_size=123)
	X_ = numpy.concatenate([batch[0] for batch in data.batches()])
	w_ = numpy.concatenate([batch[1] for batch in data.batches()])
	assert_almost_equal(X, X_)
	assert_almost_equal(w, w_)

	data = DataGenerator(X, batch_size=1)
	X_ = numpy.concatenate([batch[0] for batch in data.batches()])
	w_ = numpy.concatenate([batch[1] for batch in data.batches()])
	assert_almost_equal(X, X_)
	assert_almost_equal(w, w_)

	data = DataGenerator(X, batch_size=506)
	X_ = numpy.concatenate([batch[0] for batch in data.batches()])
	w_ = numpy.concatenate([batch[1] for batch in data.batches()])
	assert_almost_equal(X, X_)
	assert_almost_equal(w, w_)

def test_io_datagenerator_w_batches():
	X = numpy.random.randn(500, 13)
	w = numpy.abs(numpy.random.randn(500))

	data = DataGenerator(X, w)
	w_ = numpy.concatenate([batch[1] for batch in data.batches()])
	assert_almost_equal(w, w_)

	data = DataGenerator(X, w, batch_size=123)
	w_ = numpy.concatenate([batch[1] for batch in data.batches()])
	assert_almost_equal(w, w_)

	data = DataGenerator(X, w, batch_size=1)
	w_ = numpy.concatenate([batch[1] for batch in data.batches()])
	assert_almost_equal(w, w_)

	data = DataGenerator(X, w, batch_size=506)
	w_ = numpy.concatenate([batch[1] for batch in data.batches()])
	assert_almost_equal(w, w_)

def test_io_datagenerator_y_batches():
	X = numpy.random.randn(500, 13)
	w = numpy.abs(numpy.random.randn(500))
	y = numpy.random.randint(5, size=500)

	data = DataGenerator(X, y=y)
	y_ = numpy.concatenate([batch[1] for batch in data.batches()])
	assert_almost_equal(y, y_)

	data = DataGenerator(X, y=y, batch_size=123)
	y_ = numpy.concatenate([batch[1] for batch in data.batches()])
	assert_almost_equal(y, y_)

	data = DataGenerator(X, y=y, batch_size=1)
	y_ = numpy.concatenate([batch[1] for batch in data.batches()])
	assert_almost_equal(y, y_)

	data = DataGenerator(X, y=y, batch_size=506)
	y_ = numpy.concatenate([batch[1] for batch in data.batches()])
	assert_almost_equal(y, y_)

def test_io_datagenerator_wy_batches():
	X = numpy.random.randn(500, 13)
	w = numpy.abs(numpy.random.randn(500))
	y = numpy.random.randint(5, size=500)

	data = DataGenerator(X, w, y)
	y_ = numpy.concatenate([batch[1] for batch in data.batches()])
	w_ = numpy.concatenate([batch[2] for batch in data.batches()])
	assert_almost_equal(y, y_)
	assert_almost_equal(w, w_)

	data = DataGenerator(X, w, y, batch_size=123)
	y_ = numpy.concatenate([batch[1] for batch in data.batches()])
	w_ = numpy.concatenate([batch[2] for batch in data.batches()])
	assert_almost_equal(y, y_)
	assert_almost_equal(w, w_)

	data = DataGenerator(X, w, y, batch_size=1)
	y_ = numpy.concatenate([batch[1] for batch in data.batches()])
	w_ = numpy.concatenate([batch[2] for batch in data.batches()])
	assert_almost_equal(y, y_)
	assert_almost_equal(w, w_)

	data = DataGenerator(X, w, y, batch_size=506)
	y_ = numpy.concatenate([batch[1] for batch in data.batches()])
	w_ = numpy.concatenate([batch[2] for batch in data.batches()])
	assert_almost_equal(y, y_)
	assert_almost_equal(w, w_)

def test_io_datagenerator_wy_unlabeled():
	X = numpy.random.randn(500, 13)
	w = numpy.abs(numpy.random.randn(500))
	y = numpy.random.randint(5, size=500) - 1

	data = DataGenerator(X, w, y)
	X_ = numpy.concatenate([batch[0] for batch in data.unlabeled_batches()])
	w_ = numpy.concatenate([batch[1] for batch in data.unlabeled_batches()])

	assert_true(X.shape[0] > X_.shape[0])
	assert_almost_equal(X[y == -1], X_)
	assert_almost_equal(w[y == -1], w_)

def test_io_datagenerator_wy_labeled():
	X = numpy.random.randn(500, 13)
	w = numpy.abs(numpy.random.randn(500))
	y = numpy.random.randint(5, size=500) - 1

	data = DataGenerator(X, w, y)
	X_ = numpy.concatenate([batch[0] for batch in data.labeled_batches()])
	y_ = numpy.concatenate([batch[1] for batch in data.labeled_batches()])
	w_ = numpy.concatenate([batch[2] for batch in data.labeled_batches()])

	assert_true(X.shape[0] > X_.shape[0])
	assert_almost_equal(X[y != -1], X_)
	assert_almost_equal(y[y != -1], y_)
	assert_almost_equal(w[y != -1], w_)

def test_io_seqgenerator_uni_shape():
	X = [numpy.random.randn(15) for i in range(500)] 
	data = SequenceGenerator(X)

	assert_array_equal(data.shape, (len(X), 1))

def test_io_seqgenerator_symbol_shape():
	X = [[numpy.random.choice(['A', 'C', 'G', 'T']) for j in range(15)] for i in range(500)] 
	data = SequenceGenerator(X)

	assert_array_equal(data.shape, (len(X), 1))

def test_io_seqgenerator_shape():
	X = [numpy.random.randn(15, 8) for i in range(500)] 
	data = SequenceGenerator(X)

	assert_array_equal(data.shape, (len(X), 8))

def test_io_seqgenerator_x_batches():
	X = [numpy.random.randn(15, 8) for i in range(500)] 
	w = numpy.ones(500)

	data = SequenceGenerator(X)
	X_ = numpy.concatenate([batch[0] for batch in data.batches()])
	w_ = numpy.concatenate([batch[1] for batch in data.batches()])
	assert_almost_equal(X, X_)
	assert_almost_equal(w, w_)

def test_io_seqgenerator_x_symbol_batches():
	X = [[numpy.random.choice(["A", "C", "G", "T"]) for i in range(18)] for i in range(500)] 
	w = numpy.ones(500)

	data = SequenceGenerator(X)
	X_ = numpy.concatenate([batch[0] for batch in data.batches()])
	w_ = numpy.concatenate([batch[1] for batch in data.batches()])
	assert_array_equal(X, X_)
	assert_almost_equal(w, w_)

def test_io_seqgenerator_w_batches():
	X = [numpy.random.randn(15, 8) for i in range(500)] 
	w = numpy.abs(numpy.random.randn(500))

	data = SequenceGenerator(X, w)
	w_ = numpy.concatenate([batch[1] for batch in data.batches()])
	assert_almost_equal(w, w_)

def test_io_seqgenerator_w_symbol_batches():
	X = [[numpy.random.choice(["A", "C", "G", "T"]) for i in range(18)] for i in range(500)] 
	w = numpy.abs(numpy.random.randn(500))

	data = SequenceGenerator(X, w)
	w_ = numpy.concatenate([batch[1] for batch in data.batches()])
	assert_almost_equal(w, w_)

def test_io_seqgenerator_y_batches():
	X = [numpy.random.randn(15, 8) for i in range(500)] 
	y = [numpy.random.randint(6, size=15) for i in range(500)]

	data = SequenceGenerator(X, y=y)
	X_ = numpy.concatenate([batch[0] for batch in data.batches()])
	y_ = numpy.concatenate([batch[2] for batch in data.batches()])
	assert_array_equal(X, X_)
	assert_almost_equal(y, y_)

def test_io_seqgenerator_y_symbol_batches():
	X = [[numpy.random.choice(["A", "C", "G", "T"]) for i in range(18)] for i in range(500)] 
	y = [numpy.random.randint(6, size=15) for i in range(500)]

	data = SequenceGenerator(X, y=y)
	X_ = numpy.concatenate([batch[0] for batch in data.batches()])
	y_ = numpy.concatenate([batch[2] for batch in data.batches()])
	assert_array_equal(X, X_)
	assert_almost_equal(y, y_)

def test_io_seqgenerator_wy_batches():
	X = [numpy.random.randn(15, 8) for i in range(500)] 
	y = [numpy.random.randint(6, size=15) for i in range(500)]
	w = numpy.abs(numpy.random.randn(500))

	data = SequenceGenerator(X, w, y)
	X_ = numpy.concatenate([batch[0] for batch in data.batches()])
	w_ = numpy.concatenate([batch[1] for batch in data.batches()])
	y_ = numpy.concatenate([batch[2] for batch in data.batches()])
	assert_almost_equal(X, X_)
	assert_almost_equal(w, w_)
	assert_almost_equal(y, y_)

def test_io_seqgenerator_wy_symbol_batches():
	X = [[numpy.random.choice(["A", "C", "G", "T"]) for i in range(18)] for i in range(500)] 
	y = [numpy.random.randint(6, size=15) for i in range(500)]
	w = numpy.abs(numpy.random.randn(500))

	data = SequenceGenerator(X, w, y)
	X_ = numpy.concatenate([batch[0] for batch in data.batches()])
	w_ = numpy.concatenate([batch[1] for batch in data.batches()])
	y_ = numpy.concatenate([batch[2] for batch in data.batches()])
	assert_array_equal(X, X_)
	assert_almost_equal(w, w_)
	assert_almost_equal(y, y_)

def test_io_seqgenerator_wy_unlabeled():
	X = [numpy.random.randn(15, 8) for i in range(500)] 
	y = [numpy.random.randint(6, size=15) for i in range(500)]
	w = numpy.abs(numpy.random.randn(500))

	idx = numpy.random.choice(500, size=100, replace=False)
	for i in idx:
		y[i] = None

	data = SequenceGenerator(X, w, y)
	X_ = numpy.concatenate([batch[0] for batch in data.unlabeled_batches()])
	w_ = numpy.concatenate([batch[1] for batch in data.unlabeled_batches()])

	assert_true(len(X) > len(X_))
	assert_true(len(w) > len(w_))

	i = 0
	for j in range(500):
		if y[j] is None:
			assert_almost_equal(X[j], X_[i])
			assert_almost_equal(w[j], w_[i])
			i += 1

def test_io_seqgenerator_wy_symbol_unlabeled():
	X = [[numpy.random.choice(["A", "C", "G", "T"]) for i in range(18)] for i in range(500)]  
	y = [numpy.random.randint(6, size=15) for i in range(500)]
	w = numpy.abs(numpy.random.randn(500))

	idx = numpy.random.choice(500, size=100, replace=False)
	for i in idx:
		y[i] = None

	data = SequenceGenerator(X, w, y)
	X_ = numpy.concatenate([batch[0] for batch in data.unlabeled_batches()])
	w_ = numpy.concatenate([batch[1] for batch in data.unlabeled_batches()])

	assert_true(len(X) > len(X_))
	assert_true(len(w) > len(w_))

	i = 0
	for j in range(500):
		if y[j] is None:
			assert_array_equal(X[j], X_[i])
			assert_almost_equal(w[j], w_[i])
			i += 1

def test_io_seqgenerator_wy_labeled():
	X = [numpy.random.randn(15, 8) for i in range(500)]
	y = [numpy.random.randint(6, size=15) for i in range(500)]
	w = numpy.abs(numpy.random.randn(500))

	idx = numpy.random.choice(500, size=100, replace=False)
	for i in idx:
		y[i] = None

	data = SequenceGenerator(X, w, y)
	X_ = numpy.concatenate([batch[0] for batch in data.labeled_batches()])
	w_ = numpy.concatenate([batch[1] for batch in data.labeled_batches()])
	y_ = numpy.concatenate([batch[2] for batch in data.labeled_batches()])

	assert_true(len(X) > len(X_))
	assert_true(len(w) > len(w_))
	assert_true(len(y) > len(y_))

	i = 0
	for j in range(500):
		if y[j] is not None:
			assert_almost_equal(X[j], X_[i])
			assert_almost_equal(w[j], w_[i])
			assert_almost_equal(y[j], y_[i])
			i += 1

def test_io_seqgenerator_wy_symbol_labeled():
	X = [[numpy.random.choice(["A", "C", "G", "T"]) for i in range(18)] for i in range(500)] 
	y = [numpy.random.randint(6, size=15) for i in range(500)]
	w = numpy.abs(numpy.random.randn(500))

	idx = numpy.random.choice(500, size=100, replace=False)
	for i in idx:
		y[i] = None

	data = SequenceGenerator(X, w, y)
	X_ = numpy.concatenate([batch[0] for batch in data.labeled_batches()])
	w_ = numpy.concatenate([batch[1] for batch in data.labeled_batches()])
	y_ = numpy.concatenate([batch[2] for batch in data.labeled_batches()])

	assert_true(len(X) > len(X_))
	assert_true(len(w) > len(w_))
	assert_true(len(y) > len(y_))

	i = 0
	for j in range(500):
		if y[j] is not None:
			assert_array_equal(X[j], X_[i])
			assert_almost_equal(w[j], w_[i])
			assert_almost_equal(y[j], y_[i])
			i += 1

def test_io_dfgenerator_shape():
	X = pandas.DataFrame(numpy.random.randn(500, 13))
	data = DataFrameGenerator(X)

	assert_array_equal(data.shape, X.shape)

def test_io_dfgenerator_classes_fail():
	X = pandas.DataFrame(numpy.random.randn(500, 13))
	data = DataFrameGenerator(X)

	assert_raises(ValueError, lambda data: data.classes, data)

def test_io_dfgenerator_numpy_classes():
	X = pandas.DataFrame(numpy.random.randn(500, 13))
	y = numpy.random.randint(5, size=500)
	data = DataFrameGenerator(X, y=y)

	assert_array_equal(data.classes, [0, 1, 2, 3, 4])

def test_io_dfgenerator_pandas_classes():
	X = pandas.DataFrame(numpy.random.randn(500, 13))
	y = pandas.Series(numpy.random.randint(5, size=500))
	data = DataFrameGenerator(X, y=y)

	assert_array_equal(data.classes, [0, 1, 2, 3, 4])

def test_io_dfgenerator_str_classes():
	X = pandas.DataFrame(numpy.random.randn(500, 13))
	X['y'] = numpy.random.randint(5, size=500)
	data = DataFrameGenerator(X, y='y')

	assert_array_equal(data.classes, [0, 1, 2, 3, 4])

def test_io_dfgenerator_x_batches():
	X = pandas.DataFrame(numpy.random.randn(500, 13))
	w = numpy.ones(500)

	data = DataFrameGenerator(X)
	X_ = numpy.concatenate([batch[0] for batch in data.batches()])
	w_ = numpy.concatenate([batch[1] for batch in data.batches()])
	assert_almost_equal(X.values, X_)
	assert_almost_equal(w, w_)

	data = DataFrameGenerator(X, batch_size=123)
	X_ = numpy.concatenate([batch[0] for batch in data.batches()])
	w_ = numpy.concatenate([batch[1] for batch in data.batches()])
	assert_almost_equal(X.values, X_)
	assert_almost_equal(w, w_)

	data = DataFrameGenerator(X, batch_size=1)
	X_ = numpy.concatenate([batch[0] for batch in data.batches()])
	w_ = numpy.concatenate([batch[1] for batch in data.batches()])
	assert_almost_equal(X.values, X_)
	assert_almost_equal(w, w_)

	data = DataFrameGenerator(X, batch_size=506)
	X_ = numpy.concatenate([batch[0] for batch in data.batches()])
	w_ = numpy.concatenate([batch[1] for batch in data.batches()])
	assert_almost_equal(X.values, X_)
	assert_almost_equal(w, w_)

def test_io_dfgenerator_w_batches():
	X = pandas.DataFrame(numpy.random.randn(500, 13))
	w = numpy.abs(numpy.random.randn(500))

	data = DataFrameGenerator(X, w)
	X_ = numpy.concatenate([batch[0] for batch in data.batches()])
	w_ = numpy.concatenate([batch[1] for batch in data.batches()])
	assert_almost_equal(X.values, X_)
	assert_almost_equal(w, w_)

	data = DataFrameGenerator(X, w, batch_size=123)
	X_ = numpy.concatenate([batch[0] for batch in data.batches()])
	w_ = numpy.concatenate([batch[1] for batch in data.batches()])
	assert_almost_equal(X.values, X_)
	assert_almost_equal(w, w_)

	data = DataFrameGenerator(X, w, batch_size=1)
	X_ = numpy.concatenate([batch[0] for batch in data.batches()])
	w_ = numpy.concatenate([batch[1] for batch in data.batches()])
	assert_almost_equal(X.values, X_)
	assert_almost_equal(w, w_)

	data = DataFrameGenerator(X, w, batch_size=506)
	X_ = numpy.concatenate([batch[0] for batch in data.batches()])
	w_ = numpy.concatenate([batch[1] for batch in data.batches()])
	assert_almost_equal(X.values, X_)
	assert_almost_equal(w, w_)

def test_io_dfgenerator_w_str_batches():
	X = pandas.DataFrame(numpy.random.randn(500, 13))
	X2 = X.copy()
	w = numpy.abs(numpy.random.randn(500))
	X['w'] = w

	data = DataFrameGenerator(X, 'w')
	X_ = numpy.concatenate([batch[0] for batch in data.batches()])
	w_ = numpy.concatenate([batch[1] for batch in data.batches()])
	assert_almost_equal(X2.values, X_)
	assert_almost_equal(w, w_)

	data = DataFrameGenerator(X, 'w', batch_size=123)
	X_ = numpy.concatenate([batch[0] for batch in data.batches()])
	w_ = numpy.concatenate([batch[1] for batch in data.batches()])
	assert_almost_equal(X2.values, X_)
	assert_almost_equal(w, w_)

	data = DataFrameGenerator(X, 'w', batch_size=1)
	X_ = numpy.concatenate([batch[0] for batch in data.batches()])
	w_ = numpy.concatenate([batch[1] for batch in data.batches()])
	assert_almost_equal(X2.values, X_)
	assert_almost_equal(w, w_)

	data = DataFrameGenerator(X, 'w', batch_size=506)
	X_ = numpy.concatenate([batch[0] for batch in data.batches()])
	w_ = numpy.concatenate([batch[1] for batch in data.batches()])
	assert_almost_equal(X2.values, X_)
	assert_almost_equal(w, w_)

def test_io_dfgenerator_y_batches():
	X = pandas.DataFrame(numpy.random.randn(500, 13))
	y = numpy.random.randint(5, size=500)

	data = DataFrameGenerator(X, y=y)
	X_ = numpy.concatenate([batch[0] for batch in data.batches()])
	y_ = numpy.concatenate([batch[1] for batch in data.batches()])
	assert_almost_equal(X.values, X_)
	assert_almost_equal(y, y_)

	data = DataFrameGenerator(X, y=y, batch_size=123)
	X_ = numpy.concatenate([batch[0] for batch in data.batches()])
	y_ = numpy.concatenate([batch[1] for batch in data.batches()])
	assert_almost_equal(X.values, X_)
	assert_almost_equal(y, y_)

	data = DataFrameGenerator(X, y=y, batch_size=1)
	X_ = numpy.concatenate([batch[0] for batch in data.batches()])
	y_ = numpy.concatenate([batch[1] for batch in data.batches()])
	assert_almost_equal(X.values, X_)
	assert_almost_equal(y, y_)

	data = DataFrameGenerator(X, y=y, batch_size=506)
	X_ = numpy.concatenate([batch[0] for batch in data.batches()])
	y_ = numpy.concatenate([batch[1] for batch in data.batches()])
	assert_almost_equal(X.values, X_)
	assert_almost_equal(y, y_)

def test_io_dfgenerator_y_str_batches():
	X = pandas.DataFrame(numpy.random.randn(500, 13))
	X2 = X.copy()
	y = numpy.random.randint(5, size=500)
	X['y'] = y

	data = DataFrameGenerator(X, y='y')
	X_ = numpy.concatenate([batch[0] for batch in data.batches()])
	y_ = numpy.concatenate([batch[1] for batch in data.batches()])
	assert_almost_equal(X2.values, X_)
	assert_almost_equal(y, y_)

	data = DataFrameGenerator(X, y='y', batch_size=123)
	X_ = numpy.concatenate([batch[0] for batch in data.batches()])
	y_ = numpy.concatenate([batch[1] for batch in data.batches()])
	assert_almost_equal(X2.values, X_)
	assert_almost_equal(y, y_)

	data = DataFrameGenerator(X, y='y', batch_size=1)
	X_ = numpy.concatenate([batch[0] for batch in data.batches()])
	y_ = numpy.concatenate([batch[1] for batch in data.batches()])
	assert_almost_equal(X2.values, X_)
	assert_almost_equal(y, y_)

	data = DataFrameGenerator(X, y='y', batch_size=506)
	X_ = numpy.concatenate([batch[0] for batch in data.batches()])
	y_ = numpy.concatenate([batch[1] for batch in data.batches()])
	assert_almost_equal(X2.values, X_)
	assert_almost_equal(y, y_)

def test_io_dfgenerator_wy_batches():
	X = pandas.DataFrame(numpy.random.randn(500, 13))
	w = numpy.abs(numpy.random.randn(500))
	y = numpy.random.randint(5, size=500)

	data = DataFrameGenerator(X, w, y)
	X_ = numpy.concatenate([batch[0] for batch in data.batches()])
	y_ = numpy.concatenate([batch[1] for batch in data.batches()])
	w_ = numpy.concatenate([batch[2] for batch in data.batches()])
	assert_almost_equal(X.values, X_)
	assert_almost_equal(y, y_)
	assert_almost_equal(w, w_)

	data = DataFrameGenerator(X, w, y, batch_size=123)
	X_ = numpy.concatenate([batch[0] for batch in data.batches()])
	y_ = numpy.concatenate([batch[1] for batch in data.batches()])
	w_ = numpy.concatenate([batch[2] for batch in data.batches()])
	assert_almost_equal(X.values, X_)
	assert_almost_equal(y, y_)
	assert_almost_equal(w, w_)

	data = DataFrameGenerator(X, w, y, batch_size=1)
	X_ = numpy.concatenate([batch[0] for batch in data.batches()])
	y_ = numpy.concatenate([batch[1] for batch in data.batches()])
	w_ = numpy.concatenate([batch[2] for batch in data.batches()])
	assert_almost_equal(X.values, X_)
	assert_almost_equal(y, y_)
	assert_almost_equal(w, w_)

	data = DataFrameGenerator(X, w, y, batch_size=506)
	X_ = numpy.concatenate([batch[0] for batch in data.batches()])
	y_ = numpy.concatenate([batch[1] for batch in data.batches()])
	w_ = numpy.concatenate([batch[2] for batch in data.batches()])
	assert_almost_equal(X.values, X_)
	assert_almost_equal(y, y_)
	assert_almost_equal(w, w_)

def test_io_dfgenerator_wy_str_batches():
	X = pandas.DataFrame(numpy.random.randn(500, 13))
	X2 = X.copy()
	w = numpy.abs(numpy.random.randn(500))
	y = numpy.random.randint(5, size=500)

	X['w'] = w
	X['y'] = y

	data = DataFrameGenerator(X, 'w', 'y')
	X_ = numpy.concatenate([batch[0] for batch in data.batches()])
	y_ = numpy.concatenate([batch[1] for batch in data.batches()])
	w_ = numpy.concatenate([batch[2] for batch in data.batches()])
	assert_almost_equal(X2.values, X_)
	assert_almost_equal(y, y_)
	assert_almost_equal(w, w_)

	data = DataFrameGenerator(X, 'w', 'y', batch_size=123)
	X_ = numpy.concatenate([batch[0] for batch in data.batches()])
	y_ = numpy.concatenate([batch[1] for batch in data.batches()])
	w_ = numpy.concatenate([batch[2] for batch in data.batches()])
	assert_almost_equal(X2.values, X_)
	assert_almost_equal(y, y_)
	assert_almost_equal(w, w_)

	data = DataFrameGenerator(X, 'w', 'y', batch_size=1)
	X_ = numpy.concatenate([batch[0] for batch in data.batches()])
	y_ = numpy.concatenate([batch[1] for batch in data.batches()])
	w_ = numpy.concatenate([batch[2] for batch in data.batches()])
	assert_almost_equal(X2.values, X_)
	assert_almost_equal(y, y_)
	assert_almost_equal(w, w_)

	data = DataFrameGenerator(X, 'w', 'y', batch_size=506)
	X_ = numpy.concatenate([batch[0] for batch in data.batches()])
	y_ = numpy.concatenate([batch[1] for batch in data.batches()])
	w_ = numpy.concatenate([batch[2] for batch in data.batches()])
	assert_almost_equal(X2.values, X_)
	assert_almost_equal(y, y_)
	assert_almost_equal(w, w_)

def test_io_dfgenerator_wy_unlabeled():
	X = pandas.DataFrame(numpy.random.randn(500, 13))
	w = numpy.abs(numpy.random.randn(500))
	y = numpy.random.randint(5, size=500) - 1

	data = DataFrameGenerator(X, w, y)
	X_ = numpy.concatenate([batch[0] for batch in data.unlabeled_batches()])
	w_ = numpy.concatenate([batch[1] for batch in data.unlabeled_batches()])

	assert_true(X.shape[0] > X_.shape[0])
	assert_almost_equal(X.loc[y == -1], X_)
	assert_almost_equal(w[y == -1], w_)

def test_io_dfgenerator_wy_str_unlabeled():
	X = pandas.DataFrame(numpy.random.randn(500, 13))
	X2 = X.copy()
	w = numpy.abs(numpy.random.randn(500))
	y = numpy.random.randint(5, size=500) - 1

	X['w'] = w
	X['y'] = y

	data = DataFrameGenerator(X, 'w', 'y')
	X_ = numpy.concatenate([batch[0] for batch in data.unlabeled_batches()])
	w_ = numpy.concatenate([batch[1] for batch in data.unlabeled_batches()])

	assert_true(X.shape[0] > X_.shape[0])
	assert_almost_equal(X2.loc[y == -1], X_)
	assert_almost_equal(w[y == -1], w_)

def test_io_dfgenerator_wy_labeled():
	X = pandas.DataFrame(numpy.random.randn(500, 13))
	w = numpy.abs(numpy.random.randn(500))
	y = numpy.random.randint(5, size=500) - 1

	data = DataFrameGenerator(X, w, y)
	X_ = numpy.concatenate([batch[0] for batch in data.labeled_batches()])
	y_ = numpy.concatenate([batch[1] for batch in data.labeled_batches()])
	w_ = numpy.concatenate([batch[2] for batch in data.labeled_batches()])

	assert_true(X.shape[0] > X_.shape[0])
	assert_almost_equal(X.loc[y != -1], X_)
	assert_almost_equal(y[y != -1], y_)
	assert_almost_equal(w[y != -1], w_)

def test_io_dfgenerator_wy_str_labeled():
	X = pandas.DataFrame(numpy.random.randn(500, 13))
	X2 = X.copy()
	w = numpy.abs(numpy.random.randn(500))
	y = numpy.random.randint(5, size=500) - 1

	X['w'] = w
	X['y'] = y

	data = DataFrameGenerator(X, 'w', 'y')
	X_ = numpy.concatenate([batch[0] for batch in data.labeled_batches()])
	y_ = numpy.concatenate([batch[1] for batch in data.labeled_batches()])
	w_ = numpy.concatenate([batch[2] for batch in data.labeled_batches()])

	assert_true(X.shape[0] > X_.shape[0])
	assert_almost_equal(X2.loc[y != -1], X_)
	assert_almost_equal(y[y != -1], y_)
	assert_almost_equal(w[y != -1], w_)

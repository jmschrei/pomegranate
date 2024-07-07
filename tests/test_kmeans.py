# test_kmeans.py
# Contact: Jacob Schreiber <jmschreiber91@gmail.com>

import os
import numpy
import torch
import pytest

from pomegranate.kmeans import KMeans

from .distributions._utils import _test_initialization_raises_one_parameter
from .distributions._utils import _test_initialization
from .distributions._utils import _test_predictions
from .distributions._utils import _test_efd_from_summaries
from .distributions._utils import _test_raises

from .tools import assert_raises
from numpy.testing import assert_array_almost_equal


MIN_VALUE = None
MAX_VALUE = None
VALID_VALUE = 1.2
inf = float("inf")


@pytest.fixture
def X():
	return [[1, 2, 0],
	     [0, 0, 1],
	     [1, 1, 2],
	     [2, 2, 2],
	     [3, 1, 0],
	     [5, 1, 4],
	     [2, 1, 0],
	     [1, 0, 2],
	     [1, 1, 0],
	     [0, 2, 1],
	     [0, 0, 0]]


@pytest.fixture
def X_masked(X):
	mask = torch.tensor(numpy.array([
		[False, True,  True ],
		[True,  True,  False],
		[False, False, False],
		[True,  True,  True ],
		[False, True,  False],
		[True,  True,  True ],
		[False, False, False],
		[True,  False, True ],
		[True,  True,  True ],
		[True,  True,  True ],
		[True,  False, True ]]))

	X = torch.tensor(numpy.array(X))
	return torch.masked.MaskedTensor(X, mask=mask)


@pytest.fixture
def w():
	return [[1], [2], [0], [0], [5], [1], [2], [1], [1], [2], [0]]


@pytest.fixture
def model():
	centroids = [[1, 0, 1], [2, 1, -1]]
	return KMeans(centroids=centroids)


###


def test_initialization():
	model = KMeans(3)

	assert_raises(AttributeError, getattr, model, "_w_sum")
	assert_raises(AttributeError, getattr, model, "_xw_sum")


def test_initialization_raises():
	d = 2

	assert_raises(ValueError, KMeans)
	assert_raises(ValueError, KMeans, k=1)
	assert_raises(ValueError, KMeans, k=0)
	assert_raises(ValueError, KMeans, k=-1)
	assert_raises(ValueError, KMeans, k=2.4)

	assert_raises(ValueError, KMeans, d, inertia=-0.4)
	assert_raises(ValueError, KMeans, d, inertia=1.2)
	assert_raises(ValueError, KMeans, d, inertia=1.2, frozen="true")
	assert_raises(ValueError, KMeans, d, inertia=1.2, frozen=3)
	
	assert_raises(ValueError, KMeans, d, tol=-2)

	assert_raises(ValueError, KMeans, d, max_iter=0)
	assert_raises(ValueError, KMeans, d, max_iter=1.3)
	assert_raises(ValueError, KMeans, d, max_iter=-2)

	assert_raises(ValueError, KMeans, d, init="aaa")
	assert_raises(ValueError, KMeans, d, init=False)


def test_reset_cache(model, X):
	model.summarize(X)
	
	assert_array_almost_equal(model._w_sum, [[8., 8., 8.], [3., 3., 3.]])
	assert_array_almost_equal(model._xw_sum, [[10., 7., 12.], [6., 4., 0.,]])

	model._reset_cache()
	assert_array_almost_equal(model._w_sum, [[0., 0., 0.], [0., 0., 0.]])
	assert_array_almost_equal(model._xw_sum, [[0., 0., 0.,], [0., 0., 0.]])	


def test_initialize(X):
	d = 4
	model = KMeans(d)

	assert model.d is None
	assert model._initialized == False
	assert_raises(AttributeError, getattr, model, "_w_sum")
	assert_raises(AttributeError, getattr, model, "_xw_sum")

	model._initialize(X)
	assert model._initialized == True
	assert model.centroids.shape == (4, 3)
	assert model.d == 3
	assert_array_almost_equal(model._w_sum,
		[[0., 0., 0.],
		 [0., 0., 0.],
		 [0., 0., 0.],
		 [0., 0., 0.]])
	assert_array_almost_equal(model._xw_sum,
		[[0., 0., 0.],
		 [0., 0., 0.],
		 [0., 0., 0.],
		 [0., 0., 0.]])



###


def test_predict(model, X):
	y_hat = model.predict(X)
	assert_array_almost_equal(y_hat, [1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0], 4)


def test_predict_raises(model, X):
	_test_raises(model, "predict", X)

	model = KMeans(2)
	_test_raises(model, "predict", X)


def test_distances(model, X):
	y_hat = model._distances(X)
	assert_array_almost_equal(y_hat,
		[[1.2910, 1.0000],
         [0.5774, 1.7321],
         [0.8165, 1.8257],
         [1.4142, 1.8257],
         [1.4142, 0.8165],
         [2.9439, 3.3665],
         [1.0000, 0.5774],
         [0.5774, 1.9149],
         [0.8165, 0.8165],
         [1.2910, 1.7321],
         [0.8165, 1.4142]], 4)


def test_distances_raises(model, X):
	_test_raises(model, "_distances", X)

	model = KMeans(2)
	_test_raises(model, "_distances", X)


###


def test_partial_summarize(model, X):
	model.summarize(X[:4])
	assert_array_almost_equal(model._w_sum, [[3., 3., 3.], [1., 1., 1.]])
	assert_array_almost_equal(model._xw_sum, [[3., 3., 5.], [1., 2., 0.]])

	model.summarize(X[4:])
	assert_array_almost_equal(model._w_sum, [[8., 8., 8.], [3., 3., 3.]])
	assert_array_almost_equal(model._xw_sum, [[10.,  7., 12.], [ 6.,  4.,  0.]])


def test_full_summarize(model, X):
	model.summarize(X)
	assert_array_almost_equal(model._w_sum, [[8., 8., 8.], [3., 3., 3.]])
	assert_array_almost_equal(model._xw_sum, [[10.,  7., 12.], [ 6.,  4.,  0.]])


def test_summarize_weighted(model, X, w):
	model.summarize(X, sample_weight=w)
	assert_array_almost_equal(model._w_sum, [[7., 7., 7.], [8., 8., 8.]])
	assert_array_almost_equal(model._xw_sum, [[ 7.,  6., 10.], [20.,  9.,  0.]])


def test_summarize_weighted_flat(model, X, w):
	w = numpy.array(w)[:,0] 

	model.summarize(X, sample_weight=w)
	assert_array_almost_equal(model._w_sum, [[7., 7., 7.], [8., 8., 8.]])
	assert_array_almost_equal(model._xw_sum, [[ 7.,  6., 10.], [20.,  9.,  0.]])


def test_summarize_weighted_2d(model, X):
	model.summarize(X, sample_weight=X)
	assert_array_almost_equal(model._w_sum, [[10.,  7., 12.], [6.,  4.,  0.]])
	assert_array_almost_equal(model._xw_sum, [[32., 11., 30.], [14.,  6.,  0.]])


def test_summarize_raises(model, X, w):
	assert_raises(ValueError, model.summarize, [X])
	assert_raises(ValueError, model.summarize, X[0])
	assert_raises((ValueError, TypeError), model.summarize, X[0][0])
	assert_raises(ValueError, model.summarize, [x[:-1] for x in X])

	assert_raises(ValueError, model.summarize, [X], w)
	assert_raises(ValueError, model.summarize, X, [w])
	assert_raises(ValueError, model.summarize, [X], [w])
	assert_raises(ValueError, model.summarize, X[:len(X)-1], w)
	assert_raises(ValueError, model.summarize, X, w[:len(w)-1])


def test_from_summaries(model, X):
	model.summarize(X)
	model.from_summaries()

	assert_array_almost_equal(model._w_sum, [[0., 0., 0.], [0., 0., 0.]])
	assert_array_almost_equal(model._xw_sum, [[0., 0., 0.], [0., 0., 0.]])
	assert_array_almost_equal(model.centroids, 
		[[1.25    , 0.875   , 1.5     ],
         [2.      , 1.333333, 0.      ]])


def test_from_summaries_weighted(model, X, w):
	model.summarize(X, sample_weight=w)
	model.from_summaries()

	assert_array_almost_equal(model._w_sum, [[0., 0., 0.], [0., 0., 0.]])
	assert_array_almost_equal(model._xw_sum, [[0., 0., 0.], [0., 0., 0.]])
	assert_array_almost_equal(model.centroids, 
		[[1.      , 0.857143, 1.428571],
         [2.5     , 1.125   , 0.      ]])


def test_from_summaries_null(model):
	centroids = torch.clone(model.centroids)

	model.from_summaries()

	assert_array_almost_equal(model._w_sum, [[0., 0., 0.], [0., 0., 0.]])
	assert_array_almost_equal(model._xw_sum, [[0., 0., 0.], [0., 0., 0.]])
	assert_raises(AssertionError, assert_array_almost_equal, model.centroids, centroids)


def test_from_summaries_inertia(X):
	model = KMeans(2, init='first-k', inertia=0.3)
	model.summarize(X)
	model.from_summaries()

	assert_array_almost_equal(model._w_sum, [[0., 0., 0.], [0., 0., 0.]])
	assert_array_almost_equal(model._xw_sum, [[0., 0., 0.], [0., 0., 0.]])
	assert_array_almost_equal(model.centroids, 
		[[1.7  , 1.6  , 0.7  ],
         [0.35 , 0.175, 1.175]])


	centroids = X[:2]
	model = KMeans(centroids=centroids, inertia=1.0)
	model.summarize(X)
	model.from_summaries()

	assert_array_almost_equal(model._w_sum, [[0., 0., 0.], [0., 0., 0.]])
	assert_array_almost_equal(model._xw_sum, [[0., 0., 0.], [0., 0., 0.]])
	assert_array_almost_equal(model.centroids, centroids)


def test_from_summaries_weighted_inertia(X, w):
	model = KMeans(2, init='first-k', inertia=0.3)
	model.summarize(X, sample_weight=w)
	model.from_summaries()

	assert_array_almost_equal(model._w_sum, [[0., 0., 0.], [0., 0., 0.]])
	assert_array_almost_equal(model._xw_sum, [[0., 0., 0.], [0., 0., 0.]])
	assert_array_almost_equal(model.centroids, 
		[[1.816667, 1.475   , 0.35    ],
         [0.233333, 0.      , 1.233333]])


	centroids = X[:2]
	model = KMeans(centroids=centroids, inertia=1.0)
	model.summarize(X, sample_weight=w)
	model.from_summaries()

	assert_array_almost_equal(model._w_sum, [[0., 0., 0.], [0., 0., 0.]])
	assert_array_almost_equal(model._xw_sum, [[0., 0., 0.], [0., 0., 0.]])
	assert_array_almost_equal(model.centroids, centroids)


def test_from_summaries_frozen(X):
	centroids = X[:2]
	model = KMeans(centroids=centroids, frozen=True)
	model.summarize(X, sample_weight=w)
	model.from_summaries()

	assert_array_almost_equal(model._w_sum, [[0., 0., 0.], [0., 0., 0.]])
	assert_array_almost_equal(model._xw_sum, [[0., 0., 0.], [0., 0., 0.]])
	assert_array_almost_equal(model.centroids, centroids)


def test_fit(model, X):
	model.fit(X)

	assert_array_almost_equal(model._w_sum, [[0., 0., 0.], [0., 0., 0.]])
	assert_array_almost_equal(model._xw_sum, [[0., 0., 0.], [0., 0., 0.]])
	assert_array_almost_equal(model.centroids, 
		[[1.5, 1. , 2. ],
         [1.4, 1. , 0. ]])


def test_fit_weighted(model, X, w):
	model.fit(X, sample_weight=w)

	assert_array_almost_equal(model._w_sum, [[0., 0., 0.], [0., 0., 0.]])
	assert_array_almost_equal(model._xw_sum, [[0., 0., 0.], [0., 0., 0.]])
	assert_array_almost_equal(model.centroids, 
		[[0.428571, 1.      , 0.857143],
         [3.      , 1.      , 0.5     ]])


def test_fit_chain(X):
	model = KMeans(k=2, init='first-k').fit(X)

	assert_array_almost_equal(model._w_sum, [[0., 0., 0.], [0., 0., 0.]])
	assert_array_almost_equal(model._xw_sum, [[0., 0., 0.], [0., 0., 0.]])
	assert_array_almost_equal(model.centroids, 
		[[2.6     , 1.4     , 1.2     ],
         [0.5     , 0.666667, 1.      ]])


def test_fit_raises(model, X, w):
	assert_raises(ValueError, model.fit, [X])
	assert_raises(ValueError, model.fit, X[0])
	assert_raises((ValueError, TypeError), model.fit, X[0][0])
	assert_raises(ValueError, model.fit, [x[:-1] for x in X])

	assert_raises(ValueError, model.fit, [X], w)
	assert_raises(ValueError, model.fit, X, [w])
	assert_raises(ValueError, model.fit, [X], [w])
	assert_raises(ValueError, model.fit, X[:len(X)-1], w)
	assert_raises(ValueError, model.fit, X, w[:len(w)-1])


def test_serialization(X, model):
	torch.save(model, ".pytest.torch")
	model2 = torch.load(".pytest.torch")
	os.system("rm .pytest.torch")

	assert model is not model2

	assert_array_almost_equal(model2.centroids, model.centroids)
	assert_array_almost_equal(model2._w_sum, model._w_sum)
	assert_array_almost_equal(model2._xw_sum, model._xw_sum)
	assert_array_almost_equal(model2._centroid_sum, model._centroid_sum)

	assert_array_almost_equal(model2._distances(X), model._distances(X))


def test_masked_distances(model, X, X_masked):
	X = torch.tensor(numpy.array(X))
	mask = torch.ones_like(X).type(torch.bool)
	X_ = torch.masked.MaskedTensor(X, mask=mask)

	y_hat = model._distances(X_)
	assert_array_almost_equal(y_hat,
		[[1.2910, 1.0000],
         [0.5774, 1.7321],
         [0.8165, 1.8257],
         [1.4142, 1.8257],
         [1.4142, 0.8165],
         [2.9439, 3.3665],
         [1.0000, 0.5774],
         [0.5774, 1.9149],
         [0.8165, 0.8165],
         [1.2910, 1.7321],
         [0.8165, 1.4142]], 4)

	y_hat = model._distances(X_masked)
	assert_array_almost_equal(y_hat,
		[[1.7321, 1.7321],
         [1.0000, 1.7321],
         [   inf,    inf],
         [1.4142, 1.8257],
         [1.7321, 2.2361],
         [2.9439, 3.3665],
         [   inf,    inf],
         [0.7071, 2.3452],
         [0.8165, 0.8165],
         [1.2910, 1.7321],
         [1.0000, 1.7321]], 4)


def test_masked_summarize(model, X, X_masked, w):
	X = torch.tensor(numpy.array(X))
	mask = torch.ones_like(X).type(torch.bool)
	X_ = torch.masked.MaskedTensor(X, mask=mask)

	model.summarize(X_, sample_weight=w)
	assert_array_almost_equal(model._w_sum, [[7., 7., 7.], [8., 8., 8.]])
	assert_array_almost_equal(model._xw_sum, [[ 7.,  6., 10.], [20.,  9.,  0.]])

	model = KMeans(2)
	model.summarize(X_masked, sample_weight=w)
	assert_array_almost_equal(model._w_sum, [[ 4., 10.,  5.], [ 3.,  2.,  1.]])
	assert_array_almost_equal(model._xw_sum, [[ 6., 13.,  6.], [ 1.,  0.,  2.]])


def test_masked_from_summaries(model, X, X_masked, w):
	X = torch.tensor(numpy.array(X))
	mask = torch.ones_like(X).type(torch.bool)
	X_ = torch.masked.MaskedTensor(X, mask=mask)

	model.summarize(X_, sample_weight=w)
	model.from_summaries()

	assert_array_almost_equal(model._w_sum, [[0., 0., 0.], [0., 0., 0.]])
	assert_array_almost_equal(model._xw_sum, [[0., 0., 0.], [0., 0., 0.]])
	assert_array_almost_equal(model.centroids, 
		[[1.      , 0.857143, 1.428571],
         [2.5     , 1.125   , 0.      ]])

	model = KMeans(centroids=[[1, 0, 1], [2, 1, 0]])
	model.summarize(X_masked, sample_weight=w)
	model.from_summaries()

	assert_array_almost_equal(model._w_sum, [[0., 0., 0.], [0., 0., 0.]])
	assert_array_almost_equal(model._xw_sum, [[0., 0., 0.], [0., 0., 0.]])
	assert_array_almost_equal(model.centroids, 
		[[0.2     , 1.      , 1.333333],
         [3.      , 1.333333, 1.333333]])


def test_masked_fit(model, X, X_masked):
	X = torch.tensor(numpy.array(X))
	mask = torch.ones_like(X).type(torch.bool)
	X_ = torch.masked.MaskedTensor(X, mask=mask)

	model.fit(X_)

	assert_array_almost_equal(model._w_sum, [[0., 0., 0.], [0., 0., 0.]])
	assert_array_almost_equal(model._xw_sum, [[0., 0., 0.], [0., 0., 0.]])
	assert_array_almost_equal(model.centroids, 
		[[1.5, 1. , 2. ],
         [1.4, 1. , 0. ]])

	model = KMeans(centroids=[[1, 0, 1], [2, 1, 0]])
	model.fit(X_masked)

	assert_array_almost_equal(model._w_sum, [[0., 0., 0.], [0., 0., 0.]])
	assert_array_almost_equal(model._xw_sum, [[0., 0., 0.], [0., 0., 0.]])
	assert_array_almost_equal(model.centroids, 
		[[0.4, 1.2, 0.6],
         [3.5, 1.5, 3. ]])
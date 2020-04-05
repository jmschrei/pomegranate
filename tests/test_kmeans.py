from pomegranate import *
from nose.tools import with_setup
from nose.tools import assert_true
from nose.tools import assert_equal
from nose.tools import assert_greater_equal
from nose.tools import assert_greater
from nose.tools import assert_raises
from nose.tools import assert_not_equal
from numpy.testing import assert_almost_equal
from numpy.testing import assert_array_almost_equal
from numpy.testing import assert_array_equal
import random
import pickle
import numpy as np

numpy.random.seed(0)

def setup_three_dimensions():
	global X
	X = numpy.array([[-0.13174492,  0.51895916, -1.13141796],
		 [ 7.92260379,  7.86325294,  7.9884075 ],
         [-0.63378039, -0.96394236, -1.34125012],
         [ 8.16216236,  8.04655182,  6.68825619],
         [-0.69595565, -0.19004012,  0.40768949],
         [ 7.76271281,  8.94969945,  7.03687617],
         [-1.92481462, -1.03905815, -0.44048926],
         [ 7.90926091,  7.21944418,  7.15989354],
         [-0.97493454, -0.04714556, -0.38607725],
         [ 9.65781658,  7.04832845,  6.47613347]])

	idxs = numpy.array([29, 19, 26, 11,  8, 27, 21,  7, 14, 13])
	i, j = idxs // 3, idxs % 3

	global X_nan
	X_nan = X.copy()
	X_nan[i, j] = numpy.nan


	global centroids
	centroids = numpy.array([[0, 0, 0],
							 [8, 8, 8]])

	global model
	model = Kmeans(2, centroids)


def setup_five_dimensions():
	global X
	X = numpy.array([[-0.04320239,  2.25402395, -0.3075753 ,  0.01710706,  2.88816037],
	      [ 3.6483074 ,  5.03958367,  3.14457941,  4.94180558,  4.32880698],
	      [ 7.48485345,  8.54100011,  7.90936486,  8.12260819,  6.6466098 ],
	      [ 12.15394848,  10.52091121,  13.55495735,  10.48190106, 10.94417476],
          [ 1.21068778,  0.77311369, -0.31479566, -0.51865649,  0.4408653 ],
          [-0.62796182, -0.34947675, -1.09050772, -0.34591408,  0.78866514],
          [ 0.5661847 ,  0.30785453,  0.38823634,  1.99717206, -0.99415221],
          [ 0.10871016,  2.06244903, -0.19580087, -0.22100353, -0.43777027],
	      [ 3.06987578,  4.8633418 ,  4.23645519,  4.20563589,  3.40046883],
          [ 3.0471144 ,  3.43070459,  3.88690894,  3.61962816,  3.52399965],
          [ 3.3020318 ,  5.16491752,  3.85249134,  2.7075964 ,  4.03831846],
          [ 3.55266908,  2.69803949,  4.13340743,  5.72527752,  4.9840009 ],
	      [ 7.27689336,  8.99614296,  7.10109146,  7.81354687,  7.27320546],
          [ 9.55443921,  7.70358635,  8.9762396 ,  7.8054752 ,  7.95933534],
          [ 7.55150108,  9.09523173,  8.38379803,  8.18932292,  7.70853   ],
          [ 9.59329137,  8.26811547,  9.82226673,  8.35257773,  8.21768809],
          [ 11.77294852,  12.33135372,  13.02160394,  12.05536766, 11.96375761],
          [ 11.08768408,  13.15689157,  12.59002102,  11.16137415, 9.84335332],
          [ 11.41978669,  11.45646564,  11.77622614,  11.96590564, 12.33083825],
          [ 12.13323296,  11.89683824,  12.18373541,  13.21432431, 11.79987739]])

	idxs = numpy.array([77, 26, 61, 46, 18, 30, 94, 96, 45, 67,  4, 20, 23, 73, 37, 21, 58,
       99, 51,  7, 69, 53, 81, 85, 95,  9, 98, 24, 28, 38])
	i, j = idxs // 5, idxs % 5

	global X_nan
	X_nan = X.copy()
	X_nan[i, j] = numpy.nan

	global centroids
	centroids = numpy.array([[0, 0, 0, 0, 0],
							 [4, 4, 4, 4, 4],
							 [8, 8, 8, 8, 8],
							 [12, 12, 12, 12, 12]])

	global model
	model = Kmeans(4, centroids)


def test_kmeans_init():
	centroids = [[2, 3], [5, 7]]
	model = Kmeans(2, centroids)
	assert_equal(model.d, 2)
	assert_equal(model.k, 2)
	assert_array_equal(model.centroids, centroids)


@with_setup(setup_three_dimensions)
def test_kmeans_from_samples():
	model = Kmeans.from_samples(2, X, init='first-k')
	centroids = [[-0.872246, -0.344245, -0.578309],
      			 [ 8.282911,  7.825455,  7.069913]]

	assert_array_almost_equal(model.centroids, centroids)


@with_setup(setup_three_dimensions)
def test_kmeans_from_samples_parallel():
	model = Kmeans.from_samples(2, X, init='first-k', n_jobs=2)
	centroids = [[-0.872246, -0.344245, -0.578309],
      			 [ 8.282911,  7.825455,  7.069913]]

	assert_array_almost_equal(model.centroids, centroids)


@with_setup(setup_three_dimensions)
def test_kmeans_predict():
	y = numpy.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
	y_hat = model.predict(X)
	assert_array_equal(y, y_hat)


@with_setup(setup_three_dimensions)
def test_kmeans_predict_parallel():
	y = numpy.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
	y_hat = model.predict(X, n_jobs=2)
	assert_array_equal(y, y_hat)

	y_hat = model.predict(X, n_jobs=4)
	assert_array_equal(y, y_hat)


@with_setup(setup_five_dimensions)
def test_kmeans_predict_large():
	y = [0, 1, 2, 3, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3]
	y_hat = model.predict(X)
	assert_array_equal(y, y_hat)


@with_setup(setup_five_dimensions)
def test_kmeans_predict_large_parallel():
	y = [0, 1, 2, 3, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3]
	y_hat = model.predict(X, n_jobs=2)
	assert_array_equal(y, y_hat)

	y_hat = model.predict(X, n_jobs=4)
	assert_array_equal(y, y_hat)


@with_setup(setup_three_dimensions)
def test_kmeans_fit():
	model.fit(X)

	centroids = [[-0.872246, -0.344245, -0.578309],
       			 [ 8.282911,  7.825455,  7.069913]]

	assert_array_almost_equal(model.centroids, centroids)


@with_setup(setup_three_dimensions)
def test_kmeans_fit_parallel():
	model.fit(X, n_jobs=2)

	centroids = [[-0.872246, -0.344245, -0.578309],
       			 [ 8.282911,  7.825455,  7.069913]]

	assert_array_almost_equal(model.centroids, centroids)

	model.fit(X, n_jobs=4)

	centroids = [[-0.872246, -0.344245, -0.578309],
       			 [ 8.282911,  7.825455,  7.069913]]

	assert_array_almost_equal(model.centroids, centroids)


@with_setup(setup_five_dimensions)
def test_kmeans_multiple_init():
	model1 = Kmeans.from_samples(4, X, init='kmeans++', n_init=1)
	model2 = Kmeans.from_samples(4, X, init='kmeans++', n_init=25)

	dist1 = model1.distance(X).min(axis=1).sum()
	dist2 = model2.distance(X).min(axis=1).sum()

	assert_greater_equal(dist1, dist2)

	model1 = Kmeans.from_samples(4, X, init='first-k', n_init=1)
	model2 = Kmeans.from_samples(4, X, init='first-k', n_init=5)

	dist1 = model1.distance(X).min(axis=1).sum()
	dist2 = model2.distance(X).min(axis=1).sum()

	assert_equal(dist1, dist2)


@with_setup(setup_five_dimensions)
def test_kmeans_ooc_from_samples():
	numpy.random.seed(0)

	model1 = Kmeans.from_samples(5, X, init='first-k', batch_size=20)
	model2 = Kmeans.from_samples(5, X, init='first-k', batch_size=None)

	assert_array_equal(model1.centroids, model2.centroids)


@with_setup(setup_three_dimensions)
def test_kmeans_ooc_fit():
	centroids_copy = numpy.copy(centroids)
	model1 = Kmeans(2, centroids_copy, n_init=1)
	model1.fit(X)

	centroids_copy = numpy.copy(centroids)
	model2 = Kmeans(2, centroids_copy, n_init=1)
	model2.fit(X, batch_size=10)

	centroids_copy = numpy.copy(centroids)
	model3 = Kmeans(2, centroids_copy, n_init=1)
	model3.fit(X, batch_size=1)

	assert_array_almost_equal(model1.centroids, model2.centroids)
	assert_array_almost_equal(model1.centroids, model3.centroids)


@with_setup(setup_five_dimensions)
def test_kmeans_minibatch_from_samples():
	model1 = Kmeans.from_samples(4, X, init='first-k', batch_size=10)
	model2 = Kmeans.from_samples(4, X, init='first-k', batch_size=None)
	model3 = Kmeans.from_samples(4, X, init='first-k', batch_size=10, batches_per_epoch=1)

	assert_array_almost_equal(model1.centroids, model2.centroids)
	assert_raises(AssertionError, assert_array_equal, model1.centroids, model3.centroids)


@with_setup(setup_five_dimensions)
def test_kmeans_minibatch_fit():
	centroids_copy = numpy.copy(centroids)
	model1 = Kmeans(4, centroids_copy)
	model1.fit(X, batch_size=10)

	centroids_copy = numpy.copy(centroids)
	model2 = Kmeans(4, centroids_copy)
	model2.fit(X, batch_size=None)

	centroids_copy = numpy.copy(centroids)
	model3 = Kmeans(4, centroids_copy)
	model3.fit(X, batch_size=5, batches_per_epoch=1)

	assert_array_almost_equal(model1.centroids, model2.centroids)
	assert_raises(AssertionError, assert_array_equal, model1.centroids, model3.centroids)


@with_setup(setup_three_dimensions)
def test_kmeans_nan_from_samples():
	model = Kmeans.from_samples(2, X_nan, init='first-k')
	centroids = [[-0.872246,  0.235907, -0.785954],
      			 [ 7.94916 ,  7.825455,  7.395059]]

	assert_array_almost_equal(model.centroids, centroids)


@with_setup(setup_three_dimensions)
def test_kmeans_nan_from_samples_parallel():
	model = Kmeans.from_samples(2, X_nan, init='first-k', n_jobs=2)
	centroids = [[-0.872246,  0.235907, -0.785954],
      			 [ 7.94916 ,  7.825455,  7.395059]]

	assert_array_almost_equal(model.centroids, centroids)


@with_setup(setup_three_dimensions)
def test_kmeans_nan_fit():
	model.fit(X_nan)

	centroids = [[-0.872246,  0.235907, -0.785954],
      			 [ 7.94916 ,  7.825455,  7.395059]]

	assert_array_almost_equal(model.centroids, centroids)


@with_setup(setup_three_dimensions)
def test_kmeans_nan_fit_parallel():
	model.fit(X_nan, n_jobs=2)

	centroids = [[-0.872246,  0.235907, -0.785954],
      			 [ 7.94916 ,  7.825455,  7.395059]]

	assert_array_almost_equal(model.centroids, centroids)


@with_setup(setup_five_dimensions)
def test_kmeans_nan_fit_large():
	model.fit(X_nan)

	centroids = [[ -0.187485,   1.541443,  -0.331161,   1.00714 ,  -0.214419],
		         [  3.393221,   4.200322,   4.027316,   4.25569 ,   3.986697],
		         [  8.292196,   8.401983,   7.798085,   8.023552,   7.461508],
		         [ 11.782228,  11.711423,  12.625309,  11.727549,  10.917095]]

	assert_array_almost_equal(model.centroids, centroids)


@with_setup(setup_five_dimensions)
def test_kmeans_nan_fit_large_parallel():
	model.fit(X_nan, n_jobs=2)

	centroids = [[ -0.187485,   1.541443,  -0.331161,   1.00714 ,  -0.214419],
		         [  3.393221,   4.200322,   4.027316,   4.25569 ,   3.986697],
		         [  8.292196,   8.401983,   7.798085,   8.023552,   7.461508],
		         [ 11.782228,  11.711423,  12.625309,  11.727549,  10.917095]]

	assert_array_almost_equal(model.centroids, centroids)


@with_setup(setup_three_dimensions)
def test_kmeans_nan_predict():
	y = numpy.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
	y_hat = model.predict(X_nan)

	assert_array_almost_equal(y, y_hat)


@with_setup(setup_three_dimensions)
def test_kmeans_nan_predict_parallel():
	y = numpy.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
	y_hat = model.predict(X_nan, n_jobs=2)

	assert_array_almost_equal(y, y_hat)


@with_setup(setup_five_dimensions)
def test_kmeans_nan_large_predict():
	y = numpy.array([0, 1, 2, 3, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3])
	y_hat = model.predict(X_nan)

	assert_array_almost_equal(y, y_hat)


@with_setup(setup_five_dimensions)
def test_kmeans_nan_large_predict_parallel():
	y = numpy.array([0, 1, 2, 3, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3])
	y_hat = model.predict(X_nan, n_jobs=2)

	assert_array_almost_equal(y, y_hat)


@with_setup(setup_five_dimensions)
def test_kmeans_nan_multiple_init():
	numpy.random.seed(0)
	model1 = Kmeans.from_samples(4, X_nan, init='kmeans++', n_init=1)
	
	numpy.random.seed(0)
	model2 = Kmeans.from_samples(4, X_nan, init='kmeans++', n_init=25)

	dist1 = model1.distance(X).min(axis=1).sum()
	dist2 = model2.distance(X).min(axis=1).sum()

	assert_greater_equal(dist1, dist2)

	model1 = Kmeans.from_samples(4, X_nan, init='first-k', n_init=1)
	model2 = Kmeans.from_samples(4, X_nan, init='first-k', n_init=5)

	dist1 = model1.distance(X).min(axis=1).sum()
	dist2 = model2.distance(X).min(axis=1).sum()

	assert_equal(dist1, dist2)


@with_setup(setup_five_dimensions)
def test_kmeans_ooc_nan_from_samples():
	model1 = Kmeans.from_samples(4, X_nan, init='first-k', batch_size=20)
	model2 = Kmeans.from_samples(4, X_nan, init='first-k', batch_size=None)

	assert_array_almost_equal(model1.centroids, model2.centroids)


@with_setup(setup_five_dimensions)
def test_kmeans_ooc_nan_fit():
	centroids_copy = numpy.copy(centroids)
	model1 = Kmeans(4, centroids_copy, n_init=1)
	model1.fit(X_nan)

	centroids_copy = numpy.copy(centroids)
	model2 = Kmeans(4, centroids_copy, n_init=1)
	model2.fit(X_nan, batch_size=10)

	centroids_copy = numpy.copy(centroids)
	model3 = Kmeans(4, centroids_copy, n_init=1)
	model3.fit(X_nan, batch_size=1)

	assert_array_almost_equal(model1.centroids, model2.centroids, 4)
	assert_array_almost_equal(model1.centroids, model3.centroids, 4)


@with_setup(setup_five_dimensions)
def test_kmeans_minibatch_nan_from_samples():
	model1 = Kmeans.from_samples(4, X_nan, init='first-k', batch_size=10)
	model2 = Kmeans.from_samples(4, X_nan, init='first-k', batch_size=None)
	model3 = Kmeans.from_samples(4, X_nan, init='first-k', batch_size=10, batches_per_epoch=1)
	model4 = Kmeans.from_samples(4, X_nan, init='first-k', batch_size=10, batches_per_epoch=2)

	assert_array_almost_equal(model1.centroids, model2.centroids)
	assert_array_almost_equal(model1.centroids, model4.centroids)
	assert_raises(AssertionError, assert_array_equal, model1.centroids, model3.centroids)


@with_setup(setup_five_dimensions)
def test_kmeans_minibatch_nan_fit():
	centroids_copy = numpy.copy(centroids)
	model1 = Kmeans(4, centroids_copy, n_init=1)
	model1.fit(X, batch_size=10)

	centroids_copy = numpy.copy(centroids)
	model2 = Kmeans(4, centroids_copy, n_init=1)
	model2.fit(X, batch_size=None)

	centroids_copy = numpy.copy(centroids)
	model3 = Kmeans(4, centroids_copy, n_init=1)
	model3.fit(X, batch_size=10, batches_per_epoch=1)

	centroids_copy = numpy.copy(centroids)
	model4 = Kmeans(4, centroids_copy, n_init=1)
	model4.fit(X, batch_size=10, batches_per_epoch=2)

	assert_array_almost_equal(model1.centroids, model2.centroids)
	assert_array_almost_equal(model1.centroids, model4.centroids)
	assert_raises(AssertionError, assert_array_equal, model1.centroids, model3.centroids)

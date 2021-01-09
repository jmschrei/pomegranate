from __future__ import (division)

from pomegranate import *
from pomegranate.io import DataGenerator
from pomegranate.io import DataFrameGenerator

from nose.tools import with_setup
from nose.tools import assert_almost_equal
from nose.tools import assert_equal
from nose.tools import assert_not_equal
from nose.tools import assert_less_equal
from nose.tools import assert_raises
from nose.tools import assert_true
from numpy.testing import assert_array_equal
from numpy.testing import assert_array_almost_equal

import pandas
import random
import pickle
import numpy as np

nan = numpy.nan

def setup_univariate_mixed():
	normal = NormalDistribution(5, 2)
	uniform = UniformDistribution(0, 10)

	global model
	model = NaiveBayes([normal, uniform])

	global X
	X = numpy.array([[5], [3], [1], [-1]])

def setup_multivariate_gaussian():
	d11 = NormalDistribution(0.0, 1)
	d12 = NormalDistribution(0.5, 1)
	d13 = NormalDistribution(0.3, 1)
	d1 = IndependentComponentsDistribution([d11, d12, d13])

	d21 = NormalDistribution(1.0, 1)
	d22 = NormalDistribution(1.2, 1)
	d23 = NormalDistribution(1.5, 1)
	d2 = IndependentComponentsDistribution([d21, d22, d23])

	global model
	model = NaiveBayes([d1, d2])

	global X
	X = numpy.array([[0.3, 0.5, 0.1],
					 [0.8, 1.4, 0.5],
					 [1.4, 2.6, 1.8],
					 [4.2, 3.3, 3.7],
					 [2.6, 3.6, 3.3]])

	global y
	y = [0, 0, 0, 1, 1]

	global X_nan
	X_nan = numpy.array([[0.3, nan, 0.1],
		     			 [nan, 1.4, nan],
			     		 [1.4, 2.6, nan],
				    	 [nan, nan, nan],
					     [nan, 3.6, 3.3]])


def setup_multivariate_mixed():
	d11 = ExponentialDistribution(5)
	d12 = LogNormalDistribution(0.5, 0.78)
	d13 = PoissonDistribution(4)
	d1 = IndependentComponentsDistribution([d11, d12, d13])

	d21 = ExponentialDistribution(35)
	d22 = LogNormalDistribution(1.8, 1.33)
	d23 = PoissonDistribution(6)
	d2 = IndependentComponentsDistribution([d21, d22, d23])

	global model
	model = NaiveBayes([d1, d2])

	global X
	X = numpy.array([[0.3, 0.5, 0.1],
					 [0.8, 1.4, 0.5],
					 [1.4, 2.6, 1.8],
					 [4.2, 3.3, 3.7],
					 [2.6, 3.6, 3.3]])

	global y
	y = [0, 0, 0, 1, 1]

	global X_nan
	X_nan = numpy.array([[0.3, nan, 0.1],
		     			 [nan, 1.4, nan],
			     		 [1.4, 2.6, nan],
				    	 [nan, nan, nan],
					     [nan, 3.6, 3.3]])


def teardown():
	pass


@with_setup(setup_univariate_mixed, teardown)
def test_nb_univariate_initialization():
	assert_equal(model.d, 1)
	assert_equal(model.n, 2)


@with_setup(setup_multivariate_mixed, teardown)
def test_nb_multivariate_initialization():
	assert_equal(model.d, 3)
	assert_equal(model.n, 2)


def test_nb_univariate_constructors():
	d1 = NormalDistribution(0.5, 1)
	d2 = MultivariateGaussianDistribution([0, 0], [[1, 0], [0, 1]])
	d3 = IndependentComponentsDistribution([NormalDistribution(0, 1),
		NormalDistribution(2, 1), NormalDistribution(3, 1)])

	assert_raises(TypeError, NaiveBayes, [d1, d2])
	assert_raises(TypeError, NaiveBayes, [d1, d3])
	assert_raises(ValueError, NaiveBayes, [NormalDistribution])


def test_nb_multivariate_constructors():
	d1 = MultivariateGaussianDistribution([0, 0], [[1, 0], [0, 1]])
	d2 = IndependentComponentsDistribution([NormalDistribution(0, 1),
		NormalDistribution(2, 1), NormalDistribution(3, 1)])
	d3 = IndependentComponentsDistribution([NormalDistribution(0, 1),
		NormalDistribution(2, 1)])

	NaiveBayes([d1, d3])
	assert_raises(TypeError, NaiveBayes, [d2, d3])
	assert_raises(TypeError, NaiveBayes, [d2, d1])
	assert_raises(ValueError, NaiveBayes, [MultivariateGaussianDistribution])
	assert_raises(ValueError, NaiveBayes, [IndependentComponentsDistribution])


@with_setup(setup_univariate_mixed, teardown)
def test_nb_univariate_mixed_predict_log_proba():
	y_hat = model.predict_log_proba(X)
	y = [[-0.4063484, -1.096847],
	     [-0.6024268, -0.792926],
   	  	 [-1.5484819, -0.238981],
		 [ 0.0,   float('-inf')]]

	assert_array_almost_equal(y, y_hat)


@with_setup(setup_multivariate_gaussian, teardown)
def test_nb_multivariate_gaussian_predict_log_proba():
	y_hat = model.predict_log_proba(X)
	y = [[ -2.194303e-01,    -1.624430e+00],
 		 [ -8.00891133e-01,  -5.95891133e-01],
		 [ -3.24475797e+00,  -3.97579742e-02],
 		 [ -8.77515454e+00,  -1.54536960e-04],
 		 [ -6.90600226e+00,  -1.00225665e-03]]

	assert_array_almost_equal(y, y_hat)


@with_setup(setup_multivariate_mixed, teardown)
def test_nb_multivariate_mixed_predict_log_proba():
	y_hat = model.predict_log_proba(X)
	y = [[ -3.96979060e-05,  -1.01342320e+01],
		 [ -1.43325352e-11,  -2.49684574e+01],
		 [  0.00000000e+00,  -4.18889545e+01],
		 [  0.00000000e+00,  -1.24795606e+02],
		 [  0.00000000e+00,  -7.68246547e+01]]

	assert_array_almost_equal(y, y_hat)


@with_setup(setup_multivariate_gaussian, teardown)
def test_nb_multivariate_gaussian_nan_predict_log_proba():
	y_hat = model.predict_log_proba(X_nan)
	y = [[-0.27268481, -1.43268481],
		 [-0.90406199, -0.51906199],
		 [-2.23782228, -0.11282228],
		 [-0.69314718, -0.69314718],
		 [-4.81315536, -0.00815536]]

	assert_array_almost_equal(y, y_hat)


@with_setup(setup_multivariate_mixed, teardown)
def test_nb_multivariate_mixed_nan_predict_log_proba():
	y_hat = model.predict_log_proba(X_nan)
	y = [[ -1.21742279e-04,  -9.01366508e+00],
		 [ -2.83092062e-01,  -1.40019217e+00],
		 [  0.00000000e+00,  -4.06187917e+01],
		 [ -6.93147181e-01,  -6.93147181e-01],
		 [ -3.80319311e-01,  -1.15088421e+00]]

	assert_array_almost_equal(y, y_hat)


@with_setup(setup_univariate_mixed, teardown)
def test_nb_univariate_mixed_predict_log_proba_parallel():
	y_hat = model.predict_log_proba(X, n_jobs=2)
	y = [[-0.4063484, -1.096847],
	     [-0.6024268, -0.792926],
   	  	 [-1.5484819, -0.238981],
		 [ 0.0,   float('-inf')]]

	assert_array_almost_equal(y, y_hat)


@with_setup(setup_multivariate_gaussian, teardown)
def test_nb_multivariate_gaussian_predict_log_proba_parallel():
	y_hat = model.predict_log_proba(X, n_jobs=2)
	y = [[ -2.194303e-01,    -1.624430e+00],
 		 [ -8.00891133e-01,  -5.95891133e-01],
		 [ -3.24475797e+00,  -3.97579742e-02],
 		 [ -8.77515454e+00,  -1.54536960e-04],
 		 [ -6.90600226e+00,  -1.00225665e-03]]

	assert_array_almost_equal(y, y_hat)


@with_setup(setup_multivariate_mixed, teardown)
def test_nb_multivariate_mixed_predict_log_proba_parallel():
	y_hat = model.predict_log_proba(X, n_jobs=2)
	y = [[ -3.96979060e-05,  -1.01342320e+01],
		 [ -1.43325352e-11,  -2.49684574e+01],
		 [  0.00000000e+00,  -4.18889545e+01],
		 [  0.00000000e+00,  -1.24795606e+02],
		 [  0.00000000e+00,  -7.68246547e+01]]

	assert_array_almost_equal(y, y_hat)


@with_setup(setup_univariate_mixed, teardown)
def test_nb_univariate_mixed_predict_proba():
	y_hat = model.predict_proba(X)
	y = [[ 0.66607801,  0.33392199],
		 [ 0.54748134,  0.45251866],
		 [ 0.21257042,  0.78742958],
		 [ 1.,          0.        ]]

	assert_array_almost_equal(y, y_hat)


@with_setup(setup_multivariate_gaussian, teardown)
def test_nb_multivariate_gaussian_predict_proba():
	y_hat = model.predict_proba(X)
	y =	[[  8.02976114e-01,   1.97023886e-01],
		 [  4.48928731e-01,   5.51071269e-01],
		 [  3.89779969e-02,   9.61022003e-01],
		 [  1.54525019e-04,   9.99845475e-01],
		 [  1.00175456e-03,   9.98998245e-01]]

	assert_array_almost_equal(y, y_hat)


@with_setup(setup_multivariate_mixed, teardown)
def test_nb_multivariate_mixed_predict_proba():
	y_hat = model.predict_proba(X)
	y = [[  9.99960303e-01,   3.96971181e-05],
		 [  1.00000000e+00,   1.43329876e-11],
		 [  1.00000000e+00,   6.42477904e-19],
		 [  1.00000000e+00,   6.33806932e-55],
		 [  1.00000000e+00,   4.31992661e-34]]

	assert_array_almost_equal(y, y_hat)


@with_setup(setup_multivariate_gaussian, teardown)
def test_nb_multivariate_gaussian_nan_predict_proba():
	y_hat = model.predict_proba(X_nan)
	y = [[ 0.76133271,  0.23866729],
		 [ 0.40492153,  0.59507847],
		 [ 0.10669059,  0.89330941],
		 [ 0.5,         0.5       ],
		 [ 0.00812219,  0.99187781]]

	assert_array_almost_equal(y, y_hat)


@with_setup(setup_multivariate_mixed, teardown)
def test_nb_multivariate_mixed_nan_predict_proba():
	y_hat = model.predict_proba(X_nan)
	y = [[  9.99878265e-01,   1.21734869e-04],
		 [  7.53450421e-01,   2.46549579e-01],
		 [  1.00000000e+00,   2.28814158e-18],
		 [  5.00000000e-01,   5.00000000e-01],
		 [  6.83643080e-01,   3.16356920e-01]]

	assert_array_almost_equal(y, y_hat)


@with_setup(setup_univariate_mixed, teardown)
def test_nb_univariate_mixed_predict_proba_parallel():
	y_hat = model.predict_proba(X, n_jobs=2)
	y =  [[ 0.66607801,  0.33392199],
		  [ 0.54748134,  0.45251866],
		  [ 0.21257042,  0.78742958],
		  [ 1.,          0.        ]]

	assert_array_almost_equal(y, y_hat)


@with_setup(setup_multivariate_gaussian, teardown)
def test_nb_multivariate_gaussian_predict_proba_parallel():
	y_hat = model.predict_proba(X, n_jobs=2)
	y = [[  8.02976114e-01,   1.97023886e-01],
		 [  4.48928731e-01,   5.51071269e-01],
		 [  3.89779969e-02,   9.61022003e-01],
		 [  1.54525019e-04,   9.99845475e-01],
		 [  1.00175456e-03,   9.98998245e-01]]

	assert_array_almost_equal(y, y_hat)


@with_setup(setup_multivariate_mixed, teardown)
def test_nb_multivariate_mixed_predict_proba_parallel():
	y_hat = model.predict_proba(X, n_jobs=2)
	y = [[  9.99960303e-01,   3.96971181e-05,],
		 [  1.00000000e+00,   1.43329876e-11,],
		 [  1.00000000e+00,   6.42477904e-19,],
		 [  1.00000000e+00,   6.33806932e-55,],
		 [  1.00000000e+00,   4.31992661e-34,]]

	assert_array_almost_equal(y, y_hat)


@with_setup(setup_univariate_mixed, teardown)
def test_nb_univariate_mixed_predict():
	y_hat = model.predict(X)
	y = [0, 0, 1, 0]

	assert_array_almost_equal(y, y_hat)


@with_setup(setup_multivariate_gaussian, teardown)
def test_nb_multivariate_gaussian_predict():
	y_hat = model.predict(X)
	y = [0, 1, 1, 1, 1]

	assert_array_almost_equal(y, y_hat)


@with_setup(setup_multivariate_mixed, teardown)
def test_nb_multivariate_mixed_predict():
	y_hat = model.predict(X)
	y = [0, 0, 0, 0, 0]

	assert_array_almost_equal(y, y_hat)


@with_setup(setup_multivariate_gaussian, teardown)
def test_nb_multivariate_gaussian_nan_predict():
	y_hat = model.predict(X_nan)
	y = [0, 1, 1, 0, 1]

	assert_array_almost_equal(y, y_hat)


@with_setup(setup_multivariate_mixed, teardown)
def test_nb_multivariate_mixed_nan_predict():
	y_hat = model.predict(X_nan)
	y = [0, 0, 0, 0, 0]

	assert_array_almost_equal(y, y_hat)


@with_setup(setup_univariate_mixed, teardown)
def test_nb_univariate_mixed_predict_parallel():
	y_hat = model.predict(X, n_jobs=2)
	y = [0, 0, 1, 0]

	assert_array_almost_equal(y, y_hat)


@with_setup(setup_multivariate_gaussian, teardown)
def test_nb_multivariate_gaussian_predict_parallel():
	y_hat = model.predict(X, n_jobs=2)
	y = [0, 1, 1, 1, 1]

	assert_array_almost_equal(y, y_hat)


@with_setup(setup_multivariate_mixed, teardown)
def test_nb_multivariate_mixed_predict_parallel():
	y_hat = model.predict(X, n_jobs=2)
	y = [0, 0, 0, 0, 0]

	assert_array_almost_equal(y, y_hat)


@with_setup(setup_univariate_mixed, teardown)
def test_nb_univariate_mixed_fit():
	X = np.array([5, 4, 5, 4, 6, 5, 6, 5, 4, 6, 5, 4, 0, 0,
		1, 9, 8, 2, 0, 1, 1, 8, 10, 0]).reshape(-1, 1)
	y = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1,
		1, 1, 1, 1, 1, 1, 1, 1, 1, 1])

	model.fit(X, y)

	d1 = model.distributions[0]
	d2 = model.distributions[1]

	assert_array_almost_equal(d1.parameters, [4.916666666666667, 0.7592027982620252])
	assert_array_almost_equal(d2.parameters, [0.0, 10.0])

@with_setup(setup_multivariate_gaussian, teardown)
def test_nb_multivariate_gaussian_fit():
	model.fit(X, y)

	d11 = model.distributions[0].distributions[0]
	d12 = model.distributions[0].distributions[1]
	d13 = model.distributions[0].distributions[2]
	d21 = model.distributions[1].distributions[0]
	d22 = model.distributions[1].distributions[1]
	d23 = model.distributions[1].distributions[2]

	assert_array_almost_equal(d11.parameters, [0.8333333333333334, 0.4496912521077347])
	assert_array_almost_equal(d12.parameters, [1.5, 0.8602325267042628])
	assert_array_almost_equal(d13.parameters, [0.7999999999999999, 0.725718035235908])
	assert_array_almost_equal(d21.parameters, [3.4000000000000004, 0.7999999999999993])
	assert_array_almost_equal(d22.parameters, [3.45, 0.1499999999999969])
	assert_array_almost_equal(d23.parameters, [3.5, 0.19999999999999787])


@with_setup(setup_multivariate_mixed, teardown)
def test_nb_multivariate_mixed_fit():
	model.fit(X, y)

	d11 = model.distributions[0].distributions[0]
	d12 = model.distributions[0].distributions[1]
	d13 = model.distributions[0].distributions[2]
	d21 = model.distributions[1].distributions[0]
	d22 = model.distributions[1].distributions[1]
	d23 = model.distributions[1].distributions[2]

	assert_array_almost_equal(d11.parameters, [1.1999999920000004])
	assert_array_almost_equal(d12.parameters, [0.199612167029568, 0.6799837375101412])
	assert_array_almost_equal(d13.parameters, [0.7999999999999999])
	assert_array_almost_equal(d21.parameters, [0.2941176574394461])
	assert_array_almost_equal(d22.parameters, [1.2374281569672494, 0.04350568849481522])
	assert_array_almost_equal(d23.parameters, [3.5])


@with_setup(setup_multivariate_gaussian, teardown)
def test_nb_multivariate_gaussian_nan_fit():
	model.fit(X_nan, y)

	d11 = model.distributions[0].distributions[0]
	d12 = model.distributions[0].distributions[1]
	d13 = model.distributions[0].distributions[2]
	d21 = model.distributions[1].distributions[0]
	d22 = model.distributions[1].distributions[1]
	d23 = model.distributions[1].distributions[2]

	assert_array_almost_equal(d11.parameters, [0.85, 0.55])
	assert_array_almost_equal(d12.parameters, [2.0, 0.6000000000000003])
	assert_array_almost_equal(d13.parameters, [0.1, 0.0])
	assert_array_almost_equal(d21.parameters, [1.0, 1.0])
	assert_array_almost_equal(d22.parameters, [3.6, 0.0])
	assert_array_almost_equal(d23.parameters, [3.3, 0.0])


@with_setup(setup_multivariate_mixed, teardown)
def test_nb_multivariate_mixed_nan_fit():
	model.fit(X_nan, y)

	d11 = model.distributions[0].distributions[0]
	d12 = model.distributions[0].distributions[1]
	d13 = model.distributions[0].distributions[2]
	d21 = model.distributions[1].distributions[0]
	d22 = model.distributions[1].distributions[1]
	d23 = model.distributions[1].distributions[2]

	assert_array_almost_equal(d11.parameters, [1.1764705778546718])
	assert_array_almost_equal(d12.parameters, [0.645991, 0.3095196])
	assert_array_almost_equal(d13.parameters, [0.1])
	assert_array_almost_equal(d21.parameters, [35.0])
	assert_array_almost_equal(d22.parameters, [1.2809338454620642, 0.0])
	assert_array_almost_equal(d23.parameters, [3.3])


@with_setup(setup_univariate_mixed, teardown)
def test_nb_univariate_mixed_fit_parallel():
	X = np.array([5, 4, 5, 4, 6, 5, 6, 5, 4, 6, 5, 4, 0, 0,
		1, 9, 8, 2, 0, 1, 1, 8, 10, 0]).reshape(-1, 1)
	y = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1,
		1, 1, 1, 1, 1, 1, 1, 1, 1, 1])

	model.fit(X, y, n_jobs=2)

	d1 = model.distributions[0]
	d2 = model.distributions[1]

	assert_array_almost_equal(d1.parameters, [4.916666666666667, 0.7592027982620252])
	assert_array_almost_equal(d2.parameters, [0.0, 10.0])

@with_setup(setup_multivariate_gaussian, teardown)
def test_nb_multivariate_gaussian_fit_parallel():
	model.fit(X, y, n_jobs=2)

	d11 = model.distributions[0].distributions[0]
	d12 = model.distributions[0].distributions[1]
	d13 = model.distributions[0].distributions[2]
	d21 = model.distributions[1].distributions[0]
	d22 = model.distributions[1].distributions[1]
	d23 = model.distributions[1].distributions[2]

	assert_array_almost_equal(d11.parameters, [0.8333333333333334, 0.4496912521077347])
	assert_array_almost_equal(d12.parameters, [1.5, 0.8602325267042628])
	assert_array_almost_equal(d13.parameters, [0.7999999999999999, 0.725718035235908])
	assert_array_almost_equal(d21.parameters, [3.4000000000000004, 0.7999999999999993])
	assert_array_almost_equal(d22.parameters, [3.45, 0.1499999999999969])
	assert_array_almost_equal(d23.parameters, [3.5, 0.19999999999999787])


@with_setup(setup_multivariate_mixed, teardown)
def test_nb_multivariate_mixed_fit_parallel():
	model.fit(X, y, n_jobs=2)

	d11 = model.distributions[0].distributions[0]
	d12 = model.distributions[0].distributions[1]
	d13 = model.distributions[0].distributions[2]
	d21 = model.distributions[1].distributions[0]
	d22 = model.distributions[1].distributions[1]
	d23 = model.distributions[1].distributions[2]

	assert_array_almost_equal(d11.parameters, [1.1999999920000004])
	assert_array_almost_equal(d12.parameters, [0.199612167029568, 0.6799837375101412])
	assert_array_almost_equal(d13.parameters, [0.7999999999999999])
	assert_array_almost_equal(d21.parameters, [0.2941176574394461])
	assert_array_almost_equal(d22.parameters, [1.2374281569672494, 0.04350568849481522])
	assert_array_almost_equal(d23.parameters, [3.5])


@with_setup(setup_univariate_mixed, teardown)
def test_nb_univariate_mixed_from_samples():
	X = np.array([5, 4, 5, 4, 6, 5, 6, 5, 4, 6, 5, 4, 0, 0,
		1, 9, 8, 2, 0, 1, 1, 8, 10, 0]).reshape(-1, 1)
	y = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1,
		1, 1, 1, 1, 1, 1, 1, 1, 1, 1])

	model = NaiveBayes.from_samples([NormalDistribution, UniformDistribution],
		X, y)

	d1 = model.distributions[0]
	d2 = model.distributions[1]

	assert_array_almost_equal(d1.parameters, [4.916666666666667, 0.7592027982620252])
	assert_array_almost_equal(d2.parameters, [0.0, 10.0])

@with_setup(setup_multivariate_gaussian, teardown)
def test_nb_multivariate_gaussian_from_samples():
	model = NaiveBayes.from_samples(NormalDistribution, X, y)

	d11 = model.distributions[0].distributions[0]
	d12 = model.distributions[0].distributions[1]
	d13 = model.distributions[0].distributions[2]
	d21 = model.distributions[1].distributions[0]
	d22 = model.distributions[1].distributions[1]
	d23 = model.distributions[1].distributions[2]

	assert_array_almost_equal(d11.parameters, [0.8333333333333334, 0.4496912521077347])
	assert_array_almost_equal(d12.parameters, [1.5, 0.8602325267042628])
	assert_array_almost_equal(d13.parameters, [0.7999999999999999, 0.725718035235908])
	assert_array_almost_equal(d21.parameters, [3.4000000000000004, 0.7999999999999993])
	assert_array_almost_equal(d22.parameters, [3.45, 0.1499999999999969])
	assert_array_almost_equal(d23.parameters, [3.5, 0.19999999999999787])


@with_setup(setup_multivariate_mixed, teardown)
def test_nb_multivariate_mixed_from_samples():
	d = [ExponentialDistribution, LogNormalDistribution, PoissonDistribution]
	model = NaiveBayes.from_samples(d, X, y)

	d11 = model.distributions[0].distributions[0]
	d12 = model.distributions[0].distributions[1]
	d13 = model.distributions[0].distributions[2]
	d21 = model.distributions[1].distributions[0]
	d22 = model.distributions[1].distributions[1]
	d23 = model.distributions[1].distributions[2]

	assert_array_almost_equal(d11.parameters, [1.1999999920000004])
	assert_array_almost_equal(d12.parameters, [0.199612167029568, 0.6799837375101412])
	assert_array_almost_equal(d13.parameters, [0.7999999999999999])
	assert_array_almost_equal(d21.parameters, [0.2941176574394461])
	assert_array_almost_equal(d22.parameters, [1.2374281569672494, 0.04350568849481522])
	assert_array_almost_equal(d23.parameters, [3.5])


@with_setup(setup_multivariate_gaussian, teardown)
def test_nb_multivariate_gaussian_nan_from_samples():
	model = NaiveBayes.from_samples(NormalDistribution, X_nan, y)

	d11 = model.distributions[0].distributions[0]
	d12 = model.distributions[0].distributions[1]
	d13 = model.distributions[0].distributions[2]
	d21 = model.distributions[1].distributions[0]
	d22 = model.distributions[1].distributions[1]
	d23 = model.distributions[1].distributions[2]

	assert_array_almost_equal(d11.parameters, [0.85, 0.55])
	assert_array_almost_equal(d12.parameters, [2.0, 0.6000000000000003])
	assert_array_almost_equal(d13.parameters, [0.1, 0.0])
	assert_array_almost_equal(d21.parameters, [0.0, 1.0])
	assert_array_almost_equal(d22.parameters, [3.6, 0.0])
	assert_array_almost_equal(d23.parameters, [3.3, 0.0])


@with_setup(setup_multivariate_mixed, teardown)
def test_nb_multivariate_mixed_nan_from_samples():
	d = [ExponentialDistribution, LogNormalDistribution, PoissonDistribution]
	model = NaiveBayes.from_samples(d, X_nan, y)

	d11 = model.distributions[0].distributions[0]
	d12 = model.distributions[0].distributions[1]
	d13 = model.distributions[0].distributions[2]
	d21 = model.distributions[1].distributions[0]
	d22 = model.distributions[1].distributions[1]
	d23 = model.distributions[1].distributions[2]

	assert_array_almost_equal(d11.parameters, [1.1764705778546718])
	assert_array_almost_equal(d12.parameters, [0.645991, 0.3095196])
	assert_array_almost_equal(d13.parameters, [0.1])
	assert_array_almost_equal(d21.parameters, [1])
	assert_array_almost_equal(d22.parameters, [1.2809338454620642, 0.0])
	assert_array_almost_equal(d23.parameters, [3.3])


@with_setup(setup_univariate_mixed, teardown)
def test_nb_univariate_mixed_from_samples_parallel():
	X = np.array([5, 4, 5, 4, 6, 5, 6, 5, 4, 6, 5, 4, 0, 0,
		1, 9, 8, 2, 0, 1, 1, 8, 10, 0]).reshape(-1, 1)
	y = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1,
		1, 1, 1, 1, 1, 1, 1, 1, 1, 1])

	model = NaiveBayes.from_samples([NormalDistribution, UniformDistribution],
		X, y, n_jobs=2)

	d1 = model.distributions[0]
	d2 = model.distributions[1]

	assert_array_almost_equal(d1.parameters, [4.916666666666667, 0.7592027982620252])
	assert_array_almost_equal(d2.parameters, [0.0, 10.0])

@with_setup(setup_multivariate_gaussian, teardown)
def test_nb_multivariate_gaussian_from_samples_parallel():
	model = NaiveBayes.from_samples(NormalDistribution, X, y, n_jobs=2)

	d11 = model.distributions[0].distributions[0]
	d12 = model.distributions[0].distributions[1]
	d13 = model.distributions[0].distributions[2]
	d21 = model.distributions[1].distributions[0]
	d22 = model.distributions[1].distributions[1]
	d23 = model.distributions[1].distributions[2]

	assert_array_almost_equal(d11.parameters, [0.8333333333333334, 0.4496912521077347])
	assert_array_almost_equal(d12.parameters, [1.5, 0.8602325267042628])
	assert_array_almost_equal(d13.parameters, [0.7999999999999999, 0.725718035235908])
	assert_array_almost_equal(d21.parameters, [3.4000000000000004, 0.7999999999999993])
	assert_array_almost_equal(d22.parameters, [3.45, 0.1499999999999969])
	assert_array_almost_equal(d23.parameters, [3.5, 0.19999999999999787])


@with_setup(setup_multivariate_mixed, teardown)
def test_nb_multivariate_mixed_from_samples_parallel():
	d = [ExponentialDistribution, LogNormalDistribution, PoissonDistribution]
	model = NaiveBayes.from_samples(d, X, y, n_jobs=2)

	d11 = model.distributions[0].distributions[0]
	d12 = model.distributions[0].distributions[1]
	d13 = model.distributions[0].distributions[2]
	d21 = model.distributions[1].distributions[0]
	d22 = model.distributions[1].distributions[1]
	d23 = model.distributions[1].distributions[2]

	assert_array_almost_equal(d11.parameters, [1.1999999920000004])
	assert_array_almost_equal(d12.parameters, [0.199612167029568, 0.6799837375101412])
	assert_array_almost_equal(d13.parameters, [0.7999999999999999])
	assert_array_almost_equal(d21.parameters, [0.2941176574394461])
	assert_array_almost_equal(d22.parameters, [1.2374281569672494, 0.04350568849481522])
	assert_array_almost_equal(d23.parameters, [3.5])


@with_setup(setup_univariate_mixed, teardown)
def test_raise_errors():
	# check raises no errors when converting values
	model.predict_log_proba([[5]])
	model.predict_log_proba([[4.5]])
	model.predict_log_proba([[5], [6]])
	model.predict_log_proba(np.array([[5], [6]]) )

	model.predict_proba([[5]])
	model.predict_proba([[4.5]])
	model.predict_proba([[5], [6]])
	model.predict_proba(np.array([[5], [6]]))

	model.predict([[5]])
	model.predict([[4.5]])
	model.predict([[5], [6]])
	model.predict(np.array([[5], [6]]))

@with_setup(setup_univariate_mixed, teardown)
def test_pickling():
	j_univ = pickle.dumps(model)

	new_univ = pickle.loads(j_univ)
	assert_true(isinstance(new_univ.distributions[0], NormalDistribution))
	assert_true(isinstance(new_univ.distributions[1], UniformDistribution))
	assert_true(isinstance(new_univ, NaiveBayes))
	numpy.testing.assert_array_equal(model.weights, new_univ.weights)

@with_setup(setup_univariate_mixed, teardown)
def test_json():
	j_univ = model.to_json()

	new_univ = model.from_json(j_univ)
	assert_true(isinstance(new_univ.distributions[0], NormalDistribution))
	assert_true(isinstance(new_univ.distributions[1], UniformDistribution))
	assert_true(isinstance(new_univ, NaiveBayes))
	numpy.testing.assert_array_equal( model.weights, new_univ.weights)

@with_setup(setup_univariate_mixed, teardown)
def test_robust_from_json():
	j_univ = model.to_json()

	new_univ = from_json(j_univ)
	assert_true(isinstance(new_univ.distributions[0], NormalDistribution))
	assert_true(isinstance(new_univ.distributions[1], UniformDistribution))
	assert_true(isinstance(new_univ, NaiveBayes))
	numpy.testing.assert_array_equal( model.weights, new_univ.weights)

@with_setup(setup_multivariate_gaussian, teardown)
def test_io_log_probability():
	X2 = DataGenerator(X)
	X3 = DataFrameGenerator(pandas.DataFrame(X))
	logp1 = model.log_probability(X)
	logp2 = model.log_probability(X2)
	logp3 = model.log_probability(X3)
	assert_array_almost_equal(logp1, logp2)
	assert_array_almost_equal(logp1, logp3)

@with_setup(setup_multivariate_gaussian, teardown)
def test_io_predict():
	X2 = DataGenerator(X)
	X3 = DataFrameGenerator(pandas.DataFrame(X))
	y_hat1 = model.predict(X)
	y_hat2 = model.predict(X2)
	y_hat3 = model.predict(X3)
	assert_array_almost_equal(y_hat1, y_hat2)
	assert_array_almost_equal(y_hat1, y_hat3)

@with_setup(setup_multivariate_gaussian, teardown)
def test_io_predict_proba():
	X2 = DataGenerator(X)
	X3 = DataFrameGenerator(pandas.DataFrame(X))
	y_hat1 = model.predict_proba(X)
	y_hat2 = model.predict_proba(X2)
	y_hat3 = model.predict_proba(X3)
	assert_array_almost_equal(y_hat1, y_hat2)
	assert_array_almost_equal(y_hat1, y_hat3)

@with_setup(setup_multivariate_gaussian, teardown)
def test_io_predict_log_proba():
	X2 = DataGenerator(X)
	X3 = DataFrameGenerator(pandas.DataFrame(X))
	y_hat1 = model.predict_log_proba(X)
	y_hat2 = model.predict_log_proba(X2)
	y_hat3 = model.predict_log_proba(X3)
	assert_array_almost_equal(y_hat1, y_hat2)
	assert_array_almost_equal(y_hat1, y_hat3)

def test_io_fit():
	X = numpy.random.randn(100, 5) + 0.5
	weights = numpy.abs(numpy.random.randn(100))
	y = numpy.random.randint(2, size=100)
	data_generator = DataGenerator(X, weights, y)

	d1 = IndependentComponentsDistribution([
		NormalDistribution(0, 1) for i in range(5)])
	d2 = IndependentComponentsDistribution([
		NormalDistribution(1, 1) for i in range(5)])

	nb1 = NaiveBayes([d1, d2])
	nb1.fit(X, y, weights)

	d1 = IndependentComponentsDistribution([
		NormalDistribution(0, 1) for i in range(5)])
	d2 = IndependentComponentsDistribution([
		NormalDistribution(1, 1) for i in range(5)])

	nb2 = NaiveBayes([d1, d2])
	nb2.fit(data_generator)

	logp1 = nb1.log_probability(X)
	logp2 = nb2.log_probability(X)

	assert_array_almost_equal(logp1, logp2)

def test_io_from_samples():
	X = numpy.random.randn(100, 5) + 0.5
	weights = numpy.abs(numpy.random.randn(100))
	y = numpy.random.randint(2, size=100)
	data_generator = DataGenerator(X, weights, y)

	d = NormalDistribution

	nb1 = NaiveBayes.from_samples(d, X=X, weights=weights, y=y)
	nb2 = NaiveBayes.from_samples(d, X=data_generator)

	logp1 = nb1.log_probability(X)
	logp2 = nb2.log_probability(X)

	assert_array_almost_equal(logp1, logp2)

def test_discrete_distribution():
	"""
	Test fit NaiveBayes to discrete variables.
	"""
	X = np.array([
	[0, 0],
	[1, 1],
	[2, 0],
	[0, 0],
	[1, 1],
	[2, 0],
	])
	y = np.array([0, 0, 0, 1, 1, 1])
	m = NaiveBayes.from_samples(
		DiscreteDistribution,
		X,
		y,
	)
	p_y0 = m.distributions[0].parameters[0]
	assert_almost_equal(list(p_y0[0].parameters[0].values()), [1/3, 1/3, 1/3])
	assert_almost_equal(list(p_y0[1].parameters[0].values()), [2/3, 1/3])

	p_y1 = m.distributions[1].parameters[0]
	assert_almost_equal(list(p_y1[0].parameters[0].values()), [1/3, 1/3, 1/3])
	assert_almost_equal(list(p_y1[1].parameters[0].values()), [2/3, 1/3])

	# Check the probability calculation for a test variable.
	X_test = np.array([[1, 0]])
	p_groundtruth = np.array([[1/2, 1/2]])
	assert_array_almost_equal(m.predict_log_proba(X_test), np.log(p_groundtruth))
	assert_array_almost_equal(m.log_probability(X_test), [np.log(2/9)])
	assert_array_almost_equal(m.predict_proba(X_test), p_groundtruth)
	assert_array_almost_equal(m.predict(X_test), [0])

def test_bernoulli_discrete_distribution():
	"""
	Test model composed of a Bernoulli and discrete distribution.
	"""
	X = np.array([
		# y = 0.
		[0, 1],
		[0, 1],
		[1, 0],
		[1, 0],
		[1, 0],
		# y = 1.
		[0, 0],
		[1, 0],
		[1, 1],
		[1, 1],
	])
	y = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1])

	# Since the data is composed of binary values, `DiscreteDistribution` and
	# `BernoulliDistribution` should coincide.
	m1 = NaiveBayes.from_samples(DiscreteDistribution, X, y)
	m2 = NaiveBayes.from_samples(BernoulliDistribution, X, y)
	m3 = NaiveBayes.from_samples([DiscreteDistribution, BernoulliDistribution], X, y)
	m4 = NaiveBayes.from_samples([BernoulliDistribution, DiscreteDistribution], X, y)

	# Work out expression analytically for a single test point.
	X_test = np.array([[0, 1]])
	py0_x01 = (2/5) * (2/5) * (5/9)
	py1_x01 = (1/4) * (1/2) * (4/9)
	py0_cond_x01 = py0_x01 / (py0_x01 + py1_x01)
	p_groundtruth = np.array([[py0_cond_x01, 1- py0_cond_x01]])
	assert_array_almost_equal(m1.predict_proba(X_test), p_groundtruth)
	assert_array_almost_equal(m1.predict_log_proba(X_test), np.log(p_groundtruth))
	assert_array_almost_equal(m1.log_probability(X_test), [np.log(py0_x01 + py1_x01)])
	assert_array_almost_equal(m1.predict(X_test), [0])

	# Check that all predictions are identical.
	assert_array_almost_equal(m1.predict_proba(X), m2.predict_proba(X))
	assert_array_almost_equal(m2.predict_proba(X), m3.predict_proba(X))
	assert_array_almost_equal(m3.predict_proba(X), m4.predict_proba(X))

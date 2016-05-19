from __future__ import  (division, print_function)

from pomegranate import *
from nose.tools import with_setup
from nose.tools import assert_true
from nose.tools import assert_equal
from nose.tools import assert_greater
from nose.tools import assert_raises
from numpy.testing import assert_almost_equal
import random
import numpy as np

np.random.seed(0)
random.seed(0)

def setup_multivariate_gaussian():
	"""
	Set up a five component Gaussian mixture model, where each component
	is a multivariate Gaussian distribution.
	"""

	global gmm

	mu = np.arange(5)
	cov = np.eye(5)

	mgs = [ MultivariateGaussianDistribution( mu*i, cov ) for i in range(5) ]
	gmm = GeneralMixtureModel( mgs )


def teardown():
	"""
	Teardown the model, so delete it.
	"""
	pass


@with_setup(setup_multivariate_gaussian, teardown)
def test_multivariate_gmm_log_probability():
	X = np.array([1.1, 2.7, 3.0, 4.8, 6.2])
	assert_equal( round(gmm.log_probability(X), 4), -9.8406 )

	X = np.array([1.8, 2.1, 3.1, 5.2, 6.5])
	assert_equal( round(gmm.log_probability(X), 4), -9.6717 )

	X = np.array([0.9, 2.2, 3.2, 5.0, 5.8])
	assert_equal( round(gmm.log_probability(X), 4), -9.7162 )

	X = np.array([1.0, 2.1, 3.5, 4.3, 5.2])
	assert_equal( round(gmm.log_probability(X), 4), -9.894 )

	X = np.array([1.2, 2.9, 3.1, 4.2, 5.5])
	assert_equal( round(gmm.log_probability(X), 4), -10.9381 )

	X = np.array([1.8, 1.9, 3.0, 4.9, 5.7])
	assert_equal( round(gmm.log_probability(X), 4), -11.0661 )

	X = np.array([1.2, 3.1, 2.9, 4.2, 5.9])
	assert_equal( round(gmm.log_probability(X), 4), -11.3147 )

	X = np.array([1.0, 2.9, 3.9, 4.1, 6.0])
	assert_equal( round(gmm.log_probability(X), 4), -10.7922 )


def test_multivariate_gmm_json():
	gmm_2 = GeneralMixtureModel.from_json( gmm.to_json() )

	X = np.array([1.1, 2.7, 3.0, 4.8, 6.2])
	assert_equal( round(gmm_2.log_probability(X), 4), -9.8406 )

	X = np.array([1.8, 2.1, 3.1, 5.2, 6.5])
	assert_equal( round(gmm_2.log_probability(X), 4), -9.6717 )

	X = np.array([0.9, 2.2, 3.2, 5.0, 5.8])
	assert_equal( round(gmm_2.log_probability(X), 4), -9.7162 )

	X = np.array([1.0, 2.1, 3.5, 4.3, 5.2])
	assert_equal( round(gmm_2.log_probability(X), 4), -9.894 )

	X = np.array([1.2, 2.9, 3.1, 4.2, 5.5])
	assert_equal( round(gmm_2.log_probability(X), 4), -10.9381 )

	X = np.array([1.8, 1.9, 3.0, 4.9, 5.7])
	assert_equal( round(gmm_2.log_probability(X), 4), -11.0661 )

	X = np.array([1.2, 3.1, 2.9, 4.2, 5.9])
	assert_equal( round(gmm_2.log_probability(X), 4), -11.3147 )

	X = np.array([1.0, 2.9, 3.9, 4.1, 6.0])
	assert_equal( round(gmm_2.log_probability(X), 4), -10.7922 )


@with_setup(setup_multivariate_gaussian, teardown)
def test_multivariate_gmm_posterior():
	posterior = np.array([[ -2.10001234e+01, -1.23402948e-04, -9.00012340e+00, -4.80001234e+01, -1.17000123e+02],
                          [ -2.30009115e+01, -9.11466556e-04, -7.00091147e+00, -4.40009115e+01, -1.11000911e+02]])

	X = np.array([[ 2., 5., 7., 3., 2. ],
		          [ 1., 2., 5., 2., 5. ]])

	assert_almost_equal( gmm.predict_log_proba(X), posterior, 4)
	assert_almost_equal( numpy.exp(gmm.predict_log_proba(X)), gmm.predict_proba(X), 4 )


@with_setup(setup_multivariate_gaussian, teardown)
def test_multivariate_gmm_maximum_a_posteriori():
	X = np.array([[ 2., 5., 7., 3., 2. ],
		          [ 1., 2., 5., 2., 5. ],
				  [ 2., 1., 8., 2., 1. ],
				  [ 4., 3., 8., 1., 2. ]])

	assert_almost_equal( gmm.predict(X), gmm.predict_proba(X).argmax(axis=1) )


def test_multivariate_gmm_train():
	d1 = MultivariateGaussianDistribution( [0, 0], [[1, 0], [0, 1]] )
	d2 = MultivariateGaussianDistribution( [2, 2], [[1, 0], [0, 1]] )
	gmm = GeneralMixtureModel( [d1, d2] )

	X = np.array([[ 0.1,  0.7],
		          [ 1.8,  2.1],
		          [-0.9, -1.2],
		          [-0.0,  0.2],
		          [ 1.4,  2.9],
		          [ 1.8,  2.5],
		          [ 1.4,  3.1],
		          [ 1.0,  1.0]])

	assert_almost_equal( gmm.fit(X, verbose=True), 15.242416, 4 )


def test_multivariate_gmm_train_iterations():
	X = numpy.concatenate([ numpy.random.randn(1000, 3) + i for i in range(2) ])

	mu = np.ones(3) * 2
	cov = np.eye(3)
	mgs = [ MultivariateGaussianDistribution( mu*i, cov ) for i in range(2) ]
	gmm = GeneralMixtureModel( mgs )

	improvement = gmm.fit(X)

	mgs = [ MultivariateGaussianDistribution( mu*i, cov ) for i in range(2) ]
	gmm = GeneralMixtureModel( mgs )

	assert_greater( improvement, gmm.fit(X, max_iterations=1) )


def test_initialization_error():
	X = numpy.concatenate((numpy.random.randn(100, 5) + 2, numpy.random.randn(100, 5)))

	gmm = GeneralMixtureModel( MultivariateGaussianDistribution, n_components=2 )
	assert_raises( ValueError, gmm.predict, X )
	assert_raises( ValueError, gmm.predict_proba, X )
	assert_raises( ValueError, gmm.predict_log_proba, X )
	assert_raises( ValueError, gmm.log_probability, X )

	gmm.fit(X)
	gmm.predict(X)
	gmm.predict_proba(X)
	gmm.predict_log_proba(X)
	gmm.log_probability(X)


def test_initialization():
	X = numpy.concatenate((numpy.random.randn(100, 5) + 2, numpy.random.randn(100, 5)))
	gmm1 = GeneralMixtureModel( MultivariateGaussianDistribution, n_components=2 )
	gmm2 = GeneralMixtureModel( MultivariateGaussianDistribution, n_components=2 )
	assert_greater( gmm.fit(X), gmm.fit(X, max_iterations=1) )

	assert_raises( TypeError, GeneralMixtureModel, [NormalDistribution, NormalDistribution(5, 2)] )
	assert_raises( TypeError, GeneralMixtureModel, [NormalDistribution(5, 2), NormalDistribution] )
	assert_raises( TypeError, GeneralMixtureModel, [NormalDistribution(5, 2), MultivariateGaussianDistribution([5, 2], [[1, 0], [0, 1]])] )

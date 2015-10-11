from __future__ import  (division, print_function)

from pomegranate import *
from nose.tools import with_setup
from nose.tools import assert_true
from nose.tools import assert_equal
from nose.tools import assert_greater
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


@with_setup(setup_multivariate_gaussian, teardown)
def test_multivariate_gmm_posterior():
	posterior = np.array([[ -2.10001234e+01, -1.23402948e-04, -9.00012340e+00, -4.80001234e+01, -1.17000123e+02],
                          [ -2.30009115e+01, -9.11466556e-04, -7.00091147e+00, -4.40009115e+01, -1.11000911e+02]])

	X = np.array([[ 2., 5., 7., 3., 2. ],
		          [ 1., 2., 5., 2., 5. ]])

	assert_almost_equal( gmm.posterior(X), posterior, 4)
	assert_almost_equal( numpy.exp( gmm.posterior(X) ), gmm.predict_proba(X), 7 )
	assert_almost_equal( gmm.posterior(X), gmm.predict_log_proba(X) )


@with_setup(setup_multivariate_gaussian, teardown)
def test_multivariate_gmm_maximum_a_posteriori():
	posterior_argmax = np.array([1, 1])

	X = np.array([[ 2., 5., 7., 3., 2. ],
		          [ 1., 2., 5., 2., 5. ]])

	assert_almost_equal( gmm.maximum_a_posteriori(X), posterior_argmax )
	assert_almost_equal( gmm.maximum_a_posteriori(X), gmm.predict(X), 7 )


@with_setup(setup_multivariate_gaussian, teardown)
def test_multivariate_gmm_train():
	X = np.array([[1.1, 2.7, 3.0, 4.8, 6.2],
		          [1.8, 2.1, 3.1, 5.2, 6.5],
		          [0.9, 2.2, 3.2, 5.0, 5.8],
		          [1.0, 2.1, 3.5, 4.3, 5.2],
		          [1.2, 2.9, 3.1, 4.2, 5.5],
		          [1.8, 1.9, 3.0, 4.9, 5.7],
		          [1.2, 3.1, 2.9, 4.2, 5.9],
		          [1.0, 2.9, 3.9, 4.1, 6.0]])

	assert_almost_equal( gmm.train(X, verbose=True), 83.396986, 4 )


def test_multivariate_gmm_train_iterations():
	X = np.array([[1.5, 3.7, 3.0, 4.8, 6.2],
		          [1.8, 2.1, 3.1, 5.2, 6.5],
		          [0.9, 2.2, 3.2, 5.0, 5.8],
		          [1.0, .1, 3.5, 4.3, 5.2],
		          [1.7, 3.9, 6.1, 7.2, 10.5],
		          [1.6, 2.9, 6.0, 7.9, 10.7],
		          [1.8, 6.1, 4.9, 8.2, 9.9],
		          [1.9, 4.9, 6.9, 7.1, 10.0]])

	mu = np.arange(5) * 2
	cov = np.eye(5)
	mgs = [ MultivariateGaussianDistribution( mu*i, cov ) for i in range(5) ]
	gmm = GeneralMixtureModel( mgs )

	improvement = gmm.train(X)

	mgs = [ MultivariateGaussianDistribution( mu*i, cov ) for i in range(5) ]
	gmm = GeneralMixtureModel( mgs )

	assert_greater( improvement, gmm.train(X, max_iterations=1) )

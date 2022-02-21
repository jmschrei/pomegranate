from __future__ import (division)

from pomegranate import *
from pomegranate.bayes import BayesModel
from pomegranate.io import DataGenerator
from pomegranate.io import DataFrameGenerator

from nose.tools import with_setup
from nose.tools import assert_almost_equal
from nose.tools import assert_equal
from nose.tools import assert_not_equal
from nose.tools import assert_less_equal
from nose.tools import assert_raises
from nose.tools import assert_true
from numpy.testing import assert_array_almost_equal

import pandas
import random
import pickle
import numpy as np

nan = numpy.nan

def setup_multivariate_gaussian():
	mu, cov = [0, 0, 0], numpy.eye(3)
	d1 = MultivariateGaussianDistribution(mu, cov)

	mu, cov = [2, 2, 2], numpy.eye(3)
	d2 = MultivariateGaussianDistribution(mu, cov)

	global model
	model = BayesClassifier([d1, d2])

	global X
	X = numpy.array([[ 0.3,  0.5,  0.1],
					 [ 0.8,  1.4,  0.5],
					 [ 1.4,  2.6,  1.8],
					 [ 4.2,  3.3,  3.7],
					 [ 2.6,  3.6,  3.3],
					 [ 3.1,  2.2,  1.7],
					 [ 1.8,  2.2,  1.8],
					 [-1.2, -1.8, -1.5],
					 [-1.8,  0.3,  0.5],
					 [ 0.7, -1.3, -0.1]])

	global y
	y = [0, 0, 0, 1, 1, 1, 1, 0, 0, 0]

	global X_nan
	X_nan = numpy.array([[ 0.3,  nan,  0.1],
		     			 [ nan,  1.4,  nan],
			     		 [ 1.4,  2.6,  nan],
				    	 [ nan,  nan,  nan],
					     [ nan,  3.6,  3.3],
					     [ 3.1,  nan,  1.7],
						 [ nan,  nan,  1.8],
						 [-1.2, -1.8, -1.5],
						 [ nan,  0.3,  0.5],
						 [ nan, -1.3,  nan]])


def setup_multivariate_mixed():
	mu, cov = [0, 0, 0], numpy.eye(3)
	d1 = MultivariateGaussianDistribution(mu, cov)

	d21 = ExponentialDistribution(5)
	d22 = LogNormalDistribution(0.2, 0.8)
	d23 = PoissonDistribution(3)
	d2 = IndependentComponentsDistribution([d21, d22, d23])

	global model
	model = BayesClassifier([d1, d2])

	global X
	X = numpy.array([[ 0.3,  0.5,  0.1],
					 [ 0.8,  1.4,  0.5],
					 [ 1.4,  2.6,  1.8],
					 [ 4.2,  3.3,  3.7],
					 [ 2.6,  3.6,  3.3],
					 [ 3.1,  2.2,  1.7],
					 [ 1.8,  2.2,  1.8],
					 [ 1.2,  1.8,  1.5],
					 [ 1.8,  0.3,  0.5],
					 [ 0.7,  1.3,  0.1]])

	global y
	y = [0, 0, 0, 1, 1, 1, 1, 0, 0, 0]

	global X_nan
	X_nan = numpy.array([[ 0.3,  nan,  0.1],
		     			 [ nan,  1.4,  nan],
			     		 [ 1.4,  2.6,  nan],
				    	 [ nan,  nan,  nan],
					     [ nan,  3.6,  3.3],
					     [ 3.1,  nan,  1.7],
						 [ nan,  nan,  1.8],
						 [ 1.2,  1.8,  1.5],
						 [ nan,  0.3,  0.5],
						 [ nan,  1.3,  nan]])


def setup_hmm():
	global model
	global hmm1
	global hmm2
	global hmm3

	rigged = State( DiscreteDistribution({ 'H': 0.8, 'T': 0.2 }) )
	unrigged = State( DiscreteDistribution({ 'H': 0.5, 'T':0.5 }) )

	hmm1 = HiddenMarkovModel()
	hmm1.start = rigged
	hmm1.add_transition(rigged, rigged, 1)
	hmm1.bake()

	hmm2 = HiddenMarkovModel()
	hmm2.start = unrigged
	hmm2.add_transition(unrigged, unrigged, 1)
	hmm2.bake()

	hmm3 = HiddenMarkovModel()
	hmm3.add_transition(hmm3.start, unrigged, 0.5)
	hmm3.add_transition(hmm3.start, rigged, 0.5)
	hmm3.add_transition(rigged, rigged, 0.5)
	hmm3.add_transition(rigged, unrigged, 0.5)
	hmm3.add_transition(unrigged, rigged, 0.5)
	hmm3.add_transition(unrigged, unrigged, 0.5)
	hmm3.bake()

	model = BayesClassifier([hmm1, hmm2, hmm3])


def setup_multivariate():
	pass


def teardown():
	pass

def test_unpickle_bayes_model():
	"""Test that `BayesModel` can be pickled and unpickled."""
	dists = [BernoulliDistribution(0.2), BernoulliDistribution(0.3)]
	model = BayesModel(distributions=dists)
	unpickled_model = pickle.loads(pickle.dumps(model))
	np.testing.assert_almost_equal(model.weights, unpickled_model.weights)
	# Check weights of individual distributions.
	np.testing.assert_almost_equal(
		model.distributions[0].parameters,
		unpickled_model.distributions[0].parameters,
	)
	np.testing.assert_almost_equal(
		model.distributions[1].parameters,
		unpickled_model.distributions[1].parameters,
	)

@with_setup(setup_multivariate_gaussian, teardown)
def test_bc_multivariate_gaussian_initialization():
	assert_equal(model.d, 3)
	assert_equal(model.n, 2)
	assert_equal(model.is_vl_, False)

@with_setup(setup_multivariate_mixed, teardown)
def test_bc_multivariate_mixed_initialization():
	assert_equal(model.d, 3)
	assert_equal(model.n, 2)
	assert_equal(model.is_vl_, False)


@with_setup(setup_multivariate_gaussian, teardown)
def test_bc_multivariate_gaussian_predict_log_proba():
	y_hat = model.predict_log_proba(X)
	y = [[ -1.48842547e-02,  -4.21488425e+00],
		 [ -4.37487950e-01,  -1.03748795e+00],
		 [ -5.60369104e+00,  -3.69104343e-03],
		 [ -1.64000001e+01,  -7.54345812e-08],
		 [ -1.30000023e+01,  -2.26032685e-06],
		 [ -8.00033541e+00,  -3.35406373e-04],
		 [ -5.60369104e+00,  -3.69104343e-03],
		 [ -3.05902274e-07,  -1.50000003e+01],
		 [ -3.35406373e-04,  -8.00033541e+00],
		 [ -6.11066022e-04,  -7.40061107e+00]]

	assert_array_almost_equal(y, y_hat)


@with_setup(setup_multivariate_mixed, teardown)
def test_bc_multivariate_mixed_predict_log_proba():
	y_hat = model.predict_log_proba(X)
	y = [[ -5.03107596e-01,  -9.27980626e-01],
		 [ -1.86355320e-01,  -1.77183117e+00],
		 [ -5.58542088e-01,  -8.48731256e-01],
		 [ -7.67315597e-01,  -6.24101927e-01],
		 [ -2.32860808e+00,  -1.02510436e-01],
		 [ -3.06641866e-03,  -5.78877778e+00],
		 [ -9.85292840e-02,  -2.36626165e+00],
		 [ -2.61764180e-01,  -1.46833995e+00],
		 [ -2.01640009e-03,  -6.20744952e+00],
		 [ -1.47371167e-01,  -1.98758175e+00]]

	assert_array_almost_equal(y, y_hat)


@with_setup(setup_multivariate_gaussian, teardown)
def test_bc_multivariate_gaussian_nan_predict_log_proba():
	y_hat = model.predict_log_proba(X_nan)
	y = [[ -3.99533332e-02,  -3.23995333e+00],
		 [ -1.17110067e+00,  -3.71100666e-01],
		 [ -4.01814993e+00,  -1.81499279e-02],
		 [ -6.93147181e-01,  -6.93147181e-01],
		 [ -9.80005545e+00,  -5.54500620e-05],
		 [ -5.60369104e+00,  -3.69104343e-03],
		 [ -1.78390074e+00,  -1.83900741e-01],
		 [ -3.05902274e-07,  -1.50000003e+01],
		 [ -8.68361522e-02,  -2.48683615e+00],
		 [ -1.00016521e-02,  -4.61000165e+00]]

	assert_array_almost_equal(y, y_hat)


@with_setup(setup_multivariate_mixed, teardown)
def test_bc_multivariate_mixed_nan_predict_log_proba():
	y_hat = model.predict_log_proba(X_nan)
	y = [[ -3.57980882e-01,  -1.20093223e+00],
		 [ -1.20735130e+00,  -3.55230506e-01],
		 [ -2.43174286e-01,  -1.53310132e+00],
		 [ -6.93147181e-01,  -6.93147181e-01],
		 [ -9.31781101e+00,  -8.98143220e-05],
		 [ -6.29755079e-04,  -7.37049444e+00],
		 [ -1.31307006e+00,  -3.13332194e-01],
		 [ -2.61764180e-01,  -1.46833995e+00],
		 [ -2.29725479e-01,  -1.58353505e+00],
		 [ -1.17299253e+00,  -3.70251760e-01]]

	assert_array_almost_equal(y, y_hat)


@with_setup(setup_multivariate_gaussian, teardown)
def test_bc_multivariate_gaussian_predict_log_proba_parallel():
	y_hat = model.predict_log_proba(X, n_jobs=2)
	y = [[ -1.48842547e-02,  -4.21488425e+00],
		 [ -4.37487950e-01,  -1.03748795e+00],
		 [ -5.60369104e+00,  -3.69104343e-03],
		 [ -1.64000001e+01,  -7.54345812e-08],
		 [ -1.30000023e+01,  -2.26032685e-06],
		 [ -8.00033541e+00,  -3.35406373e-04],
		 [ -5.60369104e+00,  -3.69104343e-03],
		 [ -3.05902274e-07,  -1.50000003e+01],
		 [ -3.35406373e-04,  -8.00033541e+00],
		 [ -6.11066022e-04,  -7.40061107e+00]]

	assert_array_almost_equal(y, y_hat)


@with_setup(setup_multivariate_mixed, teardown)
def test_bc_multivariate_mixed_predict_log_proba_parallel():
	y_hat = model.predict_log_proba(X, n_jobs=2)
	y = [[ -5.03107596e-01,  -9.27980626e-01],
		 [ -1.86355320e-01,  -1.77183117e+00],
		 [ -5.58542088e-01,  -8.48731256e-01],
		 [ -7.67315597e-01,  -6.24101927e-01],
		 [ -2.32860808e+00,  -1.02510436e-01],
		 [ -3.06641866e-03,  -5.78877778e+00],
		 [ -9.85292840e-02,  -2.36626165e+00],
		 [ -2.61764180e-01,  -1.46833995e+00],
		 [ -2.01640009e-03,  -6.20744952e+00],
		 [ -1.47371167e-01,  -1.98758175e+00]]

	assert_array_almost_equal(y, y_hat)


@with_setup(setup_multivariate_gaussian, teardown)
def test_bc_multivariate_gaussian_predict_proba():
	y_hat = model.predict_proba(X)
	y =	[[  9.85225968e-01,   1.47740317e-02],
		 [  6.45656306e-01,   3.54343694e-01],
		 [  3.68423990e-03,   9.96315760e-01],
		 [  7.54345778e-08,   9.99999925e-01],
		 [  2.26032430e-06,   9.99997740e-01],
		 [  3.35350130e-04,   9.99664650e-01],
		 [  3.68423990e-03,   9.96315760e-01],
		 [  9.99999694e-01,   3.05902227e-07],
		 [  9.99664650e-01,   3.35350130e-04],
		 [  9.99389121e-01,   6.10879359e-04]]

	assert_array_almost_equal(y, y_hat)


@with_setup(setup_multivariate_mixed, teardown)
def test_bc_multivariate_mixed_predict_proba():
	y_hat = model.predict_proba(X)
	y = [[ 0.60464873,  0.39535127],
		 [ 0.82997863,  0.17002137],
		 [ 0.57204244,  0.42795756],
		 [ 0.46425765,  0.53574235],
		 [ 0.09743127,  0.90256873],
		 [ 0.99693828,  0.00306172],
		 [ 0.90616916,  0.09383084],
		 [ 0.76969251,  0.23030749],
		 [ 0.99798563,  0.00201437],
		 [ 0.86297361,  0.13702639]]

	assert_array_almost_equal(y, y_hat)


@with_setup(setup_multivariate_gaussian, teardown)
def test_bc_multivariate_gaussian_nan_predict_proba():
	y_hat = model.predict_proba(X_nan)
	y = [[  9.60834277e-01,   3.91657228e-02],
		 [  3.10025519e-01,   6.89974481e-01],
		 [  1.79862100e-02,   9.82013790e-01],
		 [  5.00000000e-01,   5.00000000e-01],
		 [  5.54485247e-05,   9.99944551e-01],
		 [  3.68423990e-03,   9.96315760e-01],
		 [  1.67981615e-01,   8.32018385e-01],
		 [  9.99999694e-01,   3.05902227e-07],
		 [  9.16827304e-01,   8.31726965e-02],
		 [  9.90048198e-01,   9.95180187e-03]]

	assert_array_almost_equal(y, y_hat)


@with_setup(setup_multivariate_mixed, teardown)
def test_bc_multivariate_mixed_nan_predict_proba():
	y_hat = model.predict_proba(X_nan)
	y = [[  6.99086440e-01,   3.00913560e-01],
		 [  2.98988163e-01,   7.01011837e-01],
		 [  7.84134838e-01,   2.15865162e-01],
		 [  5.00000000e-01,   5.00000000e-01],
		 [  8.98102888e-05,   9.99910190e-01],
		 [  9.99370443e-01,   6.29556825e-04],
		 [  2.68992964e-01,   7.31007036e-01],
		 [  7.69692511e-01,   2.30307489e-01],
		 [  7.94751748e-01,   2.05248252e-01],
		 [  3.09439547e-01,   6.90560453e-01]]

	assert_array_almost_equal(y, y_hat)


@with_setup(setup_multivariate_gaussian, teardown)
def test_bc_multivariate_gaussian_predict_proba_parallel():
	y_hat = model.predict_proba(X, n_jobs=2)
	y = [[  9.85225968e-01,   1.47740317e-02],
		 [  6.45656306e-01,   3.54343694e-01],
		 [  3.68423990e-03,   9.96315760e-01],
		 [  7.54345778e-08,   9.99999925e-01],
		 [  2.26032430e-06,   9.99997740e-01],
		 [  3.35350130e-04,   9.99664650e-01],
		 [  3.68423990e-03,   9.96315760e-01],
		 [  9.99999694e-01,   3.05902227e-07],
		 [  9.99664650e-01,   3.35350130e-04],
		 [  9.99389121e-01,   6.10879359e-04]]

	assert_array_almost_equal(y, y_hat)


@with_setup(setup_multivariate_mixed, teardown)
def test_bc_multivariate_mixed_predict_proba_parallel():
	y_hat = model.predict_proba(X, n_jobs=2)
	y = [[ 0.60464873,  0.39535127],
		 [ 0.82997863,  0.17002137],
		 [ 0.57204244,  0.42795756],
		 [ 0.46425765,  0.53574235],
		 [ 0.09743127,  0.90256873],
		 [ 0.99693828,  0.00306172],
		 [ 0.90616916,  0.09383084],
		 [ 0.76969251,  0.23030749],
		 [ 0.99798563,  0.00201437],
		 [ 0.86297361,  0.13702639]]

	assert_array_almost_equal(y, y_hat)


@with_setup(setup_multivariate_gaussian, teardown)
def test_bc_multivariate_gaussian_predict():
	y_hat = model.predict(X)
	y = [0, 0, 1, 1, 1, 1, 1, 0, 0, 0]

	assert_array_almost_equal(y, y_hat)


@with_setup(setup_multivariate_mixed, teardown)
def test_bc_multivariate_mixed_predict():
	y_hat = model.predict(X)
	y = [0, 0, 0, 1, 1, 0, 0, 0, 0, 0]

	assert_array_almost_equal(y, y_hat)


@with_setup(setup_multivariate_gaussian, teardown)
def test_bc_multivariate_gaussian_nan_predict():
	y_hat = model.predict(X_nan)
	y = [0, 1, 1, 0, 1, 1, 1, 0, 0, 0]

	assert_array_almost_equal(y, y_hat)


@with_setup(setup_multivariate_mixed, teardown)
def test_bc_multivariate_mixed_nan_predict():
	y_hat = model.predict(X_nan)
	y = [0, 1, 0, 0, 1, 0, 1, 0, 0, 1]

	assert_array_almost_equal(y, y_hat)


@with_setup(setup_multivariate_gaussian, teardown)
def test_bc_multivariate_gaussian_predict_parallel():
	y_hat = model.predict(X, n_jobs=2)
	y = [0, 0, 1, 1, 1, 1, 1, 0, 0, 0]

	assert_array_almost_equal(y, y_hat)


@with_setup(setup_multivariate_mixed, teardown)
def test_bc_multivariate_mixed_predict_parallel():
	y_hat = model.predict(X, n_jobs=2)
	y = [0, 0, 0, 1, 1, 0, 0, 0, 0, 0]

	assert_array_almost_equal(y, y_hat)


@with_setup(setup_multivariate_gaussian, teardown)
def test_bc_multivariate_gaussian_fit_parallel():
	model.fit(X, y, n_jobs=2)

	mu1 = model.distributions[0].parameters[0]
	cov1 = model.distributions[0].parameters[1]
	mu1_t = [0.03333333, 0.28333333, 0.21666666]
	cov1_t = [[1.3088888, 0.9272222, 0.6227777],
			  [0.9272222, 2.2513888, 1.3402777],
			  [0.6227777, 1.3402777, 0.9547222]]

	mu2 = model.distributions[1].parameters[0]
	cov2 = model.distributions[1].parameters[1]
	mu2_t = [2.925, 2.825, 2.625]
	cov2_t = [[0.75687499, 0.23687499, 0.4793750],
			  [0.23687499, 0.40187499, 0.5318749],
			  [0.47937500, 0.53187499, 0.7868750]]


	assert_array_almost_equal(mu1, mu1_t)
	assert_array_almost_equal(cov1, cov1_t)
	assert_array_almost_equal(mu2, mu2_t)
	assert_array_almost_equal(cov2, cov2_t)


@with_setup(setup_multivariate_mixed, teardown)
def test_bc_multivariate_mixed_fit_parallel():
	model.fit(X, y, n_jobs=2)

	mu1 = model.distributions[0].parameters[0]
	cov1 = model.distributions[0].parameters[1]
	mu1_t = [1.033333, 1.3166667, 0.75]
	cov1_t = [[0.242222, 0.0594444, 0.178333],
			  [0.059444, 0.5980555, 0.414166],
			  [0.178333, 0.4141666, 0.439166]]

	d21 = model.distributions[1].distributions[0]
	d22 = model.distributions[1].distributions[1]
	d23 = model.distributions[1].distributions[2]

	assert_array_almost_equal(mu1, mu1_t)
	assert_array_almost_equal(cov1, cov1_t)
	assert_array_almost_equal(d21.parameters, [0.34188034])
	assert_array_almost_equal(d22.parameters, [1.01294275, 0.22658346])
	assert_array_almost_equal(d23.parameters, [2.625])


@with_setup(setup_multivariate_gaussian, teardown)
def test_bc_multivariate_gaussian_from_samples():
	model = BayesClassifier.from_samples(MultivariateGaussianDistribution, X, y)

	mu1 = model.distributions[0].parameters[0]
	cov1 = model.distributions[0].parameters[1]
	mu1_t = [0.03333333, 0.2833333, 0.21666666]
	cov1_t = [[1.308888888, 0.9272222222, 0.6227777777],
			  [0.927222222, 2.251388888, 1.340277777],
			  [0.622777777, 1.340277777, 0.9547222222]]

	mu2 = model.distributions[1].parameters[0]
	cov2 = model.distributions[1].parameters[1]
	mu2_t = [2.925, 2.825, 2.625]
	cov2_t = [[0.75687500, 0.23687499, 0.47937500],
			  [0.23687499, 0.40187499, 0.53187499],
			  [0.47937500, 0.53187499, 0.78687500]]

	assert_array_almost_equal(mu1, mu1_t)
	assert_array_almost_equal(cov1, cov1_t)
	assert_array_almost_equal(mu2, mu2_t)
	assert_array_almost_equal(cov2, cov2_t)


@with_setup(setup_multivariate_gaussian, teardown)
def test_bc_multivariate_gaussian_pickle():
	model2 = pickle.loads(pickle.dumps(model))

	assert_true(isinstance(model2, BayesClassifier))
	assert_true(isinstance(model2.distributions[0], MultivariateGaussianDistribution))
	assert_true(isinstance(model2.distributions[1], MultivariateGaussianDistribution))
	assert_array_almost_equal(model.weights, model2.weights)


@with_setup(setup_multivariate_mixed, teardown)
def test_bc_multivariate_mixed_pickle():
	model2 = pickle.loads(pickle.dumps(model))

	assert_true(isinstance(model2, BayesClassifier))
	assert_true(isinstance(model2.distributions[0], MultivariateGaussianDistribution))
	assert_true(isinstance(model2.distributions[1], IndependentComponentsDistribution))
	assert_array_almost_equal(model.weights, model2.weights)


@with_setup(setup_multivariate_gaussian, teardown)
def test_bc_multivariate_gaussian_to_json():
	model2 = BayesClassifier.from_json(model.to_json())

	assert_true(isinstance(model2, BayesClassifier))
	assert_true(isinstance(model2.distributions[0], MultivariateGaussianDistribution))
	assert_true(isinstance(model2.distributions[1], MultivariateGaussianDistribution))
	assert_array_almost_equal(model.weights, model2.weights)


@with_setup(setup_multivariate_mixed, teardown)
def test_bc_multivariate_mixed_to_json():
	model2 = BayesClassifier.from_json(model.to_json())

	assert_true(isinstance(model2, BayesClassifier))
	assert_true(isinstance(model2.distributions[0], MultivariateGaussianDistribution))
	assert_true(isinstance(model2.distributions[1], IndependentComponentsDistribution))
	assert_array_almost_equal(model.weights, model2.weights)


@with_setup(setup_multivariate_gaussian, teardown)
def test_bc_multivariate_gaussian_robust_from_json():
	model2 = from_json(model.to_json())

	assert_true(isinstance(model2, BayesClassifier))
	assert_true(isinstance(model2.distributions[0], MultivariateGaussianDistribution))
	assert_true(isinstance(model2.distributions[1], MultivariateGaussianDistribution))
	assert_array_almost_equal(model.weights, model2.weights)


@with_setup(setup_multivariate_mixed, teardown)
def test_bc_multivariate_mixed_robust_from_json():
	model2 = from_json(model.to_json())

	assert_true(isinstance(model2, BayesClassifier))
	assert_true(isinstance(model2.distributions[0], MultivariateGaussianDistribution))
	assert_true(isinstance(model2.distributions[1], IndependentComponentsDistribution))
	assert_array_almost_equal(model.weights, model2.weights)


@with_setup(setup_hmm, teardown)
def test_model():
	assert_almost_equal(hmm1.log_probability(list('H')), -0.2231435513142097 )
	assert_almost_equal(hmm1.log_probability(list('T')), -1.6094379124341003 )
	assert_almost_equal(hmm1.log_probability(list('HHHH')), -0.8925742052568388 )
	assert_almost_equal(hmm1.log_probability(list('THHH')), -2.2788685663767296 )
	assert_almost_equal(hmm1.log_probability(list('TTTT')), -6.437751649736401 )

	assert_almost_equal(hmm2.log_probability(list('H')), -0.6931471805599453 )
	assert_almost_equal(hmm2.log_probability(list('T')), -0.6931471805599453 )
	assert_almost_equal(hmm2.log_probability(list('HHHH')), -2.772588722239781 )
	assert_almost_equal(hmm2.log_probability(list('THHH')), -2.772588722239781 )
	assert_almost_equal(hmm2.log_probability(list('TTTT')), -2.772588722239781 )

	assert_almost_equal(hmm3.log_probability(list('H')), -0.43078291609245417)
	assert_almost_equal(hmm3.log_probability(list('T')), -1.0498221244986776)
	assert_almost_equal(hmm3.log_probability(list('HHHH')), -1.7231316643698167)
	assert_almost_equal(hmm3.log_probability(list('THHH')), -2.3421708727760397)
	assert_almost_equal(hmm3.log_probability(list('TTTT')), -4.1992884979947105)
	assert_almost_equal(hmm3.log_probability(list('THTHTHTHTHTH')), -8.883630243546788)
	assert_almost_equal(hmm3.log_probability(list('THTHHHHHTHTH')), -7.645551826734343)

	assert_equal(model.d, 1)


@with_setup(setup_hmm, teardown)
def test_hmm_log_proba():
	logs = model.predict_log_proba(np.array([list('H'), list('THHH'), list('TTTT'), list('THTHTHTHTHTH'), list('THTHHHHHTHTH')]))

	assert_almost_equal(logs[0][0], -0.89097292388986515)
	assert_almost_equal(logs[0][1], -1.3609765531356006)
	assert_almost_equal(logs[0][2], -1.0986122886681096)

	assert_almost_equal(logs[1][0], -0.93570553121744293)
	assert_almost_equal(logs[1][1], -1.429425687080494)
	assert_almost_equal(logs[1][2], -0.9990078376167526)

	assert_almost_equal(logs[2][0], -3.9007882563128864)
	assert_almost_equal(logs[2][1], -0.23562532881626597)
	assert_almost_equal(logs[2][2], -1.6623251045711958)

	assert_almost_equal(logs[3][0], -3.1703366478831185)
	assert_almost_equal(logs[3][1], -0.49261403211260379)
	assert_almost_equal(logs[3][2], -1.058478108940049)

	assert_almost_equal(logs[4][0], -1.3058441172130273)
	assert_almost_equal(logs[4][1], -1.4007102236822906)
	assert_almost_equal(logs[4][2], -0.7284958836972919)


@with_setup(setup_hmm, teardown)
def test_hmm_proba():
	probs = model.predict_proba(np.array([list('H'), list('THHH'), list('TTTT'), list('THTHTHTHTHTH'), list('THTHHHHHTHTH')]))

	assert_almost_equal(probs[0][0], 0.41025641025641024)
	assert_almost_equal(probs[0][1], 0.25641025641025639)
	assert_almost_equal(probs[0][2], 0.33333333333333331)

	assert_almost_equal(probs[1][0], 0.39230898163446098)
	assert_almost_equal(probs[1][1], 0.23944639992337707)
	assert_almost_equal(probs[1][2], 0.36824461844216183)

	assert_almost_equal(probs[2][0], 0.020225961918306088)
	assert_almost_equal(probs[2][1], 0.79007663743383105)
	assert_almost_equal(probs[2][2], 0.18969740064786292)

	assert_almost_equal(probs[3][0], 0.041989459861032523)
	assert_almost_equal(probs[3][1], 0.61102706038265642)
	assert_almost_equal(probs[3][2], 0.346983479756311)

	assert_almost_equal(probs[4][0], 0.27094373022369794)
	assert_almost_equal(probs[4][1], 0.24642188711704707)
	assert_almost_equal(probs[4][2], 0.48263438265925512)


@with_setup(setup_hmm, teardown)
def test_hmm_prediction():
	predicts = model.predict(np.array([list('H'), list('THHH'), list('TTTT'), list('THTHTHTHTHTH'), list('THTHHHHHTHTH')]))

	assert_equal(predicts[0], 0)
	assert_equal(predicts[1], 0)
	assert_equal(predicts[2], 1)
	assert_equal(predicts[3], 1)
	assert_equal(predicts[4], 2)

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

	mu1 = numpy.array([0, 0, 0, 0, 0])
	mu2 = numpy.array([1, 1, 1, 1, 1])
	cov = numpy.eye(5)

	d1 = MultivariateGaussianDistribution(mu1, cov)
	d2 = MultivariateGaussianDistribution(mu2, cov)
	bc1 = BayesClassifier([d1, d2])
	bc1.fit(X, y, weights)

	d1 = MultivariateGaussianDistribution(mu1, cov)
	d2 = MultivariateGaussianDistribution(mu2, cov)
	bc2 = BayesClassifier([d1, d2])
	bc2.fit(data_generator)

	logp1 = bc1.log_probability(X)
	logp2 = bc2.log_probability(X)

	assert_array_almost_equal(logp1, logp2)

def test_io_from_samples():
	X = numpy.random.randn(100, 5) + 0.5
	weights = numpy.abs(numpy.random.randn(100))
	y = numpy.random.randint(2, size=100)
	data_generator = DataGenerator(X, weights, y)

	d = MultivariateGaussianDistribution

	bc1 = BayesClassifier.from_samples(d, X=X, y=y, weights=weights)
	bc2 = BayesClassifier.from_samples(d, X=data_generator)

	logp1 = bc1.log_probability(X)
	logp2 = bc2.log_probability(X)

	assert_array_almost_equal(logp1, logp2)
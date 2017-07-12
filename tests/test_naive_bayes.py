from __future__ import (division)

from pomegranate import *
from nose.tools import with_setup
from nose.tools import assert_almost_equal
from nose.tools import assert_equal
from nose.tools import assert_not_equal
from nose.tools import assert_less_equal
from nose.tools import assert_raises
from nose.tools import assert_true
from numpy.testing import assert_array_equal
import random
import pickle
import numpy as np

def setup_univariate():
	global normal
	global uniform
	global multi
	global model

	normal = NormalDistribution(5, 2)
	uniform = UniformDistribution(0, 10)
	multi = IndependentComponentsDistribution([NormalDistribution(5, 2), NormalDistribution(7, 1)])
	model = NaiveBayes([normal, uniform])


def teardown():
	pass

@with_setup(setup_univariate, teardown)
def test_univariate_distributions():
	assert_almost_equal(normal.log_probability(11), -6.112085713764219)
	assert_almost_equal(normal.log_probability(9), -3.612085713764219)
	assert_almost_equal(normal.log_probability(7), -2.112085713764219)
	assert_almost_equal(normal.log_probability(5), -1.612085713764219)
	assert_almost_equal(normal.log_probability(3), -2.112085713764219)
	assert_almost_equal(normal.log_probability(1), -3.612085713764219)
	assert_almost_equal(normal.log_probability(-1), -6.112085713764219)

	assert_almost_equal(uniform.log_probability(11), -float('inf') )
	assert_almost_equal(uniform.log_probability(10), -2.3025850929940455)
	assert_almost_equal(uniform.log_probability(5), -2.3025850929940455)
	assert_almost_equal(uniform.log_probability(0), -2.3025850929940455)
	assert_almost_equal(uniform.log_probability(-1), -float('inf') )

	assert_equal(model.d, 1)

def test_constructors():
	assert_raises(TypeError, NaiveBayes, [normal, multi])
	assert_raises(TypeError, NaiveBayes, [NormalDistribution, normal])

	assert_raises(ValueError, NaiveBayes, [NormalDistribution])
	assert_raises(ValueError, NaiveBayes, [MultivariateGaussianDistribution])
	assert_raises(ValueError, NaiveBayes, [IndependentComponentsDistribution])


@with_setup(setup_univariate, teardown)
def test_univariate_log_proba():
	logs = model.predict_log_proba(np.array([[5], [3], [1], [-1]]))

	assert_almost_equal(logs[0][0], -0.40634848776410526)
	assert_almost_equal(logs[0][1], -1.0968478669939319)

	assert_almost_equal(logs[1][0], -0.60242689998689203)
	assert_almost_equal(logs[1][1], -0.79292627921671865)

	assert_almost_equal(logs[2][0], -1.5484819556996032)
	assert_almost_equal(logs[2][1], -0.23898133492942986)

	assert_almost_equal(logs[3][0], 0.0)
	assert_almost_equal(logs[3][1], -float('inf'))

@with_setup(setup_univariate, teardown)
def test_univariate_proba():
	probs = model.predict_proba(np.array([[5], [3], [1], [-1]]))

	assert_almost_equal(probs[0][0], 0.66607800693933361)
	assert_almost_equal(probs[0][1], 0.33392199306066628)

	assert_almost_equal(probs[1][0], 0.54748134004225524)
	assert_almost_equal(probs[1][1], 0.45251865995774476)

	assert_almost_equal(probs[2][0], 0.21257042033580209)
	assert_almost_equal(probs[2][1], 0.78742957966419791)

	assert_almost_equal(probs[3][0], 1.0)
	assert_almost_equal(probs[3][1], 0.0)

@with_setup(setup_univariate, teardown)
def test_univariate_prediction():
	predicts = model.predict(np.array([[5], [3], [1], [-1]]))

	assert_equal(predicts[0], 0)
	assert_equal(predicts[1], 0)
	assert_equal(predicts[2], 1)
	assert_equal(predicts[3], 0)

def test_multivariate_prediction():
	X = numpy.concatenate([numpy.random.normal(2, 1, (10, 3)), numpy.random.normal(7, 2, (20, 3))])
	y = numpy.concatenate([numpy.zeros(10), numpy.ones(20)])

	model = NaiveBayes.from_samples(NormalDistribution, X, y)
	y_hat = model.predict(X)

	assert_array_equal(y, y_hat)

@with_setup(setup_univariate, teardown)
def test_univariate_fit():
	X = np.array([5, 4, 5, 4, 6, 5, 6, 5, 4, 6, 5, 4, 0, 0, 1, 9, 8, 2, 0, 1, 1, 8, 10, 0]).reshape(-1, 1)
	y = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])

	model.fit(X, y)

	data = np.array([[5], [3], [1], [-1]])

	# test univariate log probabilities
	logs = model.predict_log_proba(data)

	assert_almost_equal(logs[0][0], -0.1751742330621151)
	assert_almost_equal(logs[0][1], -1.8282830423560459)
	assert_almost_equal(logs[1][0], -1.7240463796541046)
	assert_almost_equal(logs[1][1], -0.19643229738177137)
	assert_almost_equal(logs[2][0], -11.64810474561418)
	assert_almost_equal(logs[2][1], -8.7356310092268075e-06)
	assert_almost_equal(logs[3][0], 0.0)
	assert_almost_equal(logs[3][1], -float('inf'))

	# test univariate probabilities
	probs = model.predict_proba(data)

	assert_almost_equal(probs[0][0], 0.83931077234299012)
	assert_almost_equal(probs[0][1], 0.1606892276570098)
	assert_almost_equal(probs[1][0], 0.17834304226047082)
	assert_almost_equal(probs[1][1], 0.82165695773952918)
	assert_almost_equal(probs[2][0], 8.7355928537720986e-06)
	assert_almost_equal(probs[2][1], 0.99999126440714625)
	assert_almost_equal(probs[3][0], 1.0)
	assert_almost_equal(probs[3][1], 0.0)

	# test univariate classifications
	predicts = model.predict(data)

	assert_equal(predicts[0], 0)
	assert_equal(predicts[1], 1)
	assert_equal(predicts[2], 1)
	assert_equal(predicts[3], 0)

@with_setup(setup_univariate, teardown)
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

@with_setup(setup_univariate, teardown)
def test_pickling():
	j_univ = pickle.dumps(model)

	new_univ = pickle.loads(j_univ)
	assert_true(isinstance(new_univ.distributions[0], NormalDistribution))
	assert_true(isinstance(new_univ.distributions[1], UniformDistribution))
	assert_true(isinstance(new_univ, NaiveBayes))
	numpy.testing.assert_array_equal(model.weights, new_univ.weights)

@with_setup(setup_univariate, teardown)
def test_json():
	j_univ = model.to_json()

	new_univ = model.from_json(j_univ)
	assert_true(isinstance(new_univ.distributions[0], NormalDistribution))
	assert_true(isinstance(new_univ.distributions[1], UniformDistribution))
	assert_true(isinstance(new_univ, NaiveBayes))
	numpy.testing.assert_array_equal( model.weights, new_univ.weights)

def test_from_samples_uni():
	X = numpy.concatenate([numpy.random.normal(5, 1, (50, 1)), numpy.random.normal(1, 1, (20, 1))])
	y = numpy.concatenate([numpy.zeros(50), numpy.ones(20)])
	
	NaiveBayes.from_samples(NormalDistribution, X, y)
	NaiveBayes.from_samples(UniformDistribution, X, y)
	model = NaiveBayes.from_samples(ExponentialDistribution, X, y)

	assert_raises(ValueError, NaiveBayes.from_samples, MultivariateGaussianDistribution, X, y)
	assert_equal(model.d, 1)
	assert_equal(len(model.distributions), 2)
	assert_true(isinstance(model.distributions[0], ExponentialDistribution))
	assert_true(isinstance(model.distributions[1], ExponentialDistribution))

def test_from_samples_multi():
	X = numpy.concatenate([numpy.random.normal(5, 1, (50, 3)), numpy.random.normal(1, 1, (20, 3))])
	y = numpy.concatenate([numpy.zeros(50), numpy.ones(20)])
	
	NaiveBayes.from_samples(NormalDistribution, X, y)
	NaiveBayes.from_samples(UniformDistribution, X, y)
	model = NaiveBayes.from_samples(ExponentialDistribution, X, y)

	assert_raises(ValueError, NaiveBayes.from_samples, MultivariateGaussianDistribution, X, y)
	assert_equal(model.d, 3)
	assert_equal(len(model.distributions), 2)
	assert_equal(len(model.distributions[0].distributions), 3)
	assert_true(isinstance(model.distributions[0], IndependentComponentsDistribution))
	assert_true(isinstance(model.distributions[1], IndependentComponentsDistribution))
	assert_true(isinstance(model.distributions[0].distributions[0], ExponentialDistribution))
	assert_true(isinstance(model.distributions[0].distributions[1], ExponentialDistribution))
	assert_true(isinstance(model.distributions[0].distributions[2], ExponentialDistribution))
	assert_true(isinstance(model.distributions[1].distributions[0], ExponentialDistribution))
	assert_true(isinstance(model.distributions[1].distributions[1], ExponentialDistribution))
	assert_true(isinstance(model.distributions[1].distributions[2], ExponentialDistribution))

	model = NaiveBayes.from_samples(PoissonDistribution, X, y)
	assert_true(isinstance(model.distributions[0], IndependentComponentsDistribution))
	assert_true(isinstance(model.distributions[1], IndependentComponentsDistribution))
	assert_true(isinstance(model.distributions[0].distributions[0], PoissonDistribution))
	assert_true(isinstance(model.distributions[0].distributions[1], PoissonDistribution))
	assert_true(isinstance(model.distributions[0].distributions[2], PoissonDistribution))
	assert_true(isinstance(model.distributions[1].distributions[0], PoissonDistribution))
	assert_true(isinstance(model.distributions[1].distributions[1], PoissonDistribution))
	assert_true(isinstance(model.distributions[1].distributions[2], PoissonDistribution))


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
from numpy.testing import assert_array_almost_equal

import numpy
import scipy.stats

nan = numpy.nan

class NormalDistribution2():
	def __init__(self, mu, std):
		self.mu = mu
		self.std = std
		self.d = 1
		self.summaries = numpy.zeros(3)

	def log_probability(self, X):
		return scipy.stats.norm.logpdf(X, self.mu, self.std)

	def summarize(self, X, w=None):
		if w is None:
			w = numpy.ones(X.shape[0])

		X = X.reshape(X.shape[0])
		self.summaries[0] += w.sum()
		self.summaries[1] += X.dot(w)
		self.summaries[2] += (X ** 2.).dot(w)

	def from_summaries(self, inertia=0.0):
		self.mu = self.summaries[1] / self.summaries[0]
		self.std = self.summaries[2] / self.summaries[0] - self.summaries[1] ** 2 / (self.summaries[0] ** 2)
		self.std = numpy.sqrt(self.std)
		self.clear_summaries()

	def clear_summaries(self, inertia=0.0):
		self.summaries = numpy.zeros(3)

	@classmethod
	def from_samples(cls, X, weights=None):
		d = NormalDistribution2(0, 0)
		d.summarize(X, weights)
		d.from_summaries()
		return d

	@classmethod
	def blank(cls):
		return NormalDistribution2(0, 0)

class MultivariateGaussianDistribution2():
	def __init__(self, mu, cov):
		self.mu = mu
		self.cov = cov
		self.d = len(mu)
		self.summaries = [0, numpy.zeros(self.d), numpy.zeros((self.d, self.d))]

	def log_probability(self, X):
		return scipy.stats.multivariate_normal.logpdf(X, self.mu, self.cov)

	def summarize(self, X, w=None):
		if w is None:
			w = numpy.ones(X.shape[0])

		self.summaries[0] += w.sum()
		self.summaries[1] += X.T.dot(w)
		self.summaries[2] += (X.T * numpy.sqrt(w)).dot((X.T * numpy.sqrt(w)).T)

	def from_summaries(self, inertia=0.0):
		self.mu = self.summaries[1] / self.summaries[0]

		w = self.summaries[0]
		x_ij = self.summaries[2]
		x_i = self.summaries[1].reshape(self.d, 1)
		self.cov = (x_ij - x_i.T * x_i / w) / w 
		self.clear_summaries()

	def clear_summaries(self, inertia=0.0):
		self.summaries = [0, numpy.zeros(self.d), numpy.zeros((self.d, self.d))]

	@classmethod
	def from_samples(cls, X, weights=None):
		mu = numpy.zeros(X.shape[1])
		cov = numpy.eye(X.shape[1])

		d = MultivariateGaussianDistribution2(mu, cov)
		d.summarize(X, weights)
		d.from_summaries()
		return d

	@classmethod
	def blank(cls, d=2):
		mu = numpy.zeros(d)
		cov = numpy.eye(d)
		return MultivariateGaussianDistribution2(mu, cov)

def build_model(d1, d2):
	s1 = State(d1, "s1")
	s2 = State(d2, "d2")

	model = HiddenMarkovModel()
	model.add_states(s1, s2)
	model.add_transition(model.start, s1, 0.8)
	model.add_transition(model.start, s2, 0.2)
	model.add_transition(s1, s1, 0.6)
	model.add_transition(s1, s2, 0.4)
	model.add_transition(s2, s2, 0.3)
	model.add_transition(s2, s1, 0.7)
	model.bake()
	return model

def setup_normal_hmm():
	global model1
	global model2

	d1 = NormalDistribution(0, 1)
	d2 = NormalDistribution(1, 1)
	model1 = build_model(d1, d2)

	d3 = NormalDistribution2(0, 1)
	d4 = NormalDistribution2(1, 1)
	model2 = build_model(d3, d4)

def setup_mgd_hmm():
	global model1
	global model2

	d1 = MultivariateGaussianDistribution(numpy.zeros(3), numpy.eye(3))
	d2 = MultivariateGaussianDistribution(numpy.ones(3), numpy.eye(3))
	model1 = build_model(d1, d2)
	

	d3 = MultivariateGaussianDistribution2(numpy.zeros(3), numpy.eye(3))
	d4 = MultivariateGaussianDistribution2(numpy.ones(3), numpy.eye(3))
	model2 = build_model(d3, d4)

def setup_icd_hmm():
	global model1
	global model2

	d1 = IndependentComponentsDistribution([NormalDistribution(0, 1) for _ in range(5)])
	d2 = IndependentComponentsDistribution([NormalDistribution(1, 1) for _ in range(5)])
	model1 = build_model(d1, d2)

	d3 = IndependentComponentsDistribution([NormalDistribution2(0, 1) for _ in range(5)])
	d4 = IndependentComponentsDistribution([NormalDistribution2(1, 1) for _ in range(5)])
	model2 = build_model(d3, d4)

def test_custom_normal_gmm_init():
	d1 = NormalDistribution2(0, 1)
	d2 = NormalDistribution2(1, 1)
	model = GeneralMixtureModel([d1, d2])

	assert_equal(d1.d, 1)
	assert_equal(d2.d, 1)
	assert_equal(model.d, 1)

def test_custom_normal_gmm_logp():
	X = numpy.random.normal(0.5, 1, size=(20, 1))

	d1 = NormalDistribution(0, 1)
	d2 = NormalDistribution(1, 1)
	model1 = GeneralMixtureModel([d1, d2])
	

	d3 = NormalDistribution(0, 1)
	d4 = NormalDistribution(1, 1)
	model2 = GeneralMixtureModel([d3, d4])

	assert_array_almost_equal(model1.log_probability(X), model2.log_probability(X))

def test_custom_normal_gmm_predict():
	X = numpy.random.normal(0.5, 1, size=(20, 1))

	d1 = NormalDistribution(0, 1)
	d2 = NormalDistribution(1, 1)
	model1 = GeneralMixtureModel([d1, d2])
	

	d3 = NormalDistribution(0, 1)
	d4 = NormalDistribution(1, 1)
	model2 = GeneralMixtureModel([d3, d4])

	assert_array_equal(model1.predict(X), model2.predict(X))

def test_custom_normal_gmm_predict_proba():
	X = numpy.random.normal(0.5, 1, size=(20, 1))

	d1 = NormalDistribution(0, 1)
	d2 = NormalDistribution(1, 1)
	model1 = GeneralMixtureModel([d1, d2])
	

	d3 = NormalDistribution(0, 1)
	d4 = NormalDistribution(1, 1)
	model2 = GeneralMixtureModel([d3, d4])

	assert_array_almost_equal(model1.predict_proba(X), model2.predict_proba(X))

def test_custom_normal_gmm_predict_log_proba():
	X = numpy.random.normal(0.5, 1, size=(20, 1))

	d1 = NormalDistribution(0, 1)
	d2 = NormalDistribution(1, 1)
	model1 = GeneralMixtureModel([d1, d2])
	

	d3 = NormalDistribution(0, 1)
	d4 = NormalDistribution(1, 1)
	model2 = GeneralMixtureModel([d3, d4])

	assert_array_almost_equal(model1.predict_log_proba(X), model2.predict_log_proba(X))

def test_custom_normal_gmm_fit():
	X = numpy.random.normal(0.5, 1, size=(20, 1))

	d1 = NormalDistribution(0, 1)
	d2 = NormalDistribution(1, 1)
	model1 = GeneralMixtureModel([d1, d2])
	model1.fit(X)

	d3 = NormalDistribution2(0, 1)
	d4 = NormalDistribution2(1, 1)
	model2 = GeneralMixtureModel([d3, d4])
	model2.fit(X)

	assert_array_almost_equal(model1.log_probability(X), model2.log_probability(X))

def test_custom_normal_gmm_from_samples():
	X = numpy.random.normal(0.5, 1, size=(200, 1))
	X[::2] += 1

	model1 = GeneralMixtureModel.from_samples(NormalDistribution, 2, X, init='first-k', max_iterations=5)
	model2 = GeneralMixtureModel.from_samples(NormalDistribution2, 2, X, init='first-k', max_iterations=5)

	assert_array_almost_equal(model1.log_probability(X), model2.log_probability(X))

def test_custom_mgd_gmm_init():
	d1 = MultivariateGaussianDistribution2(numpy.zeros(3), numpy.eye(3))
	d2 = MultivariateGaussianDistribution2(numpy.ones(3), numpy.eye(3))
	model = GeneralMixtureModel([d1, d2])

	assert_equal(d1.d, 3)
	assert_equal(d2.d, 3)
	assert_equal(model.d, 3)

def test_custom_mgd_gmm_logp():
	X = numpy.random.normal(0.5, 1, size=(20,3))

	d1 = MultivariateGaussianDistribution(numpy.zeros(3), numpy.eye(3))
	d2 = MultivariateGaussianDistribution(numpy.ones(3), numpy.eye(3))
	model1 = GeneralMixtureModel([d1, d2])
	

	d3 = MultivariateGaussianDistribution2(numpy.zeros(3), numpy.eye(3))
	d4 = MultivariateGaussianDistribution2(numpy.ones(3), numpy.eye(3))
	model2 = GeneralMixtureModel([d3, d4])

	assert_array_almost_equal(model1.log_probability(X), model2.log_probability(X))

def test_custom_mgd_gmm_predict():
	X = numpy.random.normal(0.5, 1, size=(20,3))

	d1 = MultivariateGaussianDistribution(numpy.zeros(3), numpy.eye(3))
	d2 = MultivariateGaussianDistribution(numpy.ones(3), numpy.eye(3))
	model1 = GeneralMixtureModel([d1, d2])
	

	d3 = MultivariateGaussianDistribution2(numpy.zeros(3), numpy.eye(3))
	d4 = MultivariateGaussianDistribution2(numpy.ones(3), numpy.eye(3))
	model2 = GeneralMixtureModel([d3, d4])

	assert_array_equal(model1.predict(X), model2.predict(X))

def test_custom_mgd_gmm_predict_proba():
	X = numpy.random.normal(0.5, 1, size=(20,3))

	d1 = MultivariateGaussianDistribution(numpy.zeros(3), numpy.eye(3))
	d2 = MultivariateGaussianDistribution(numpy.ones(3), numpy.eye(3))
	model1 = GeneralMixtureModel([d1, d2])
	

	d3 = MultivariateGaussianDistribution2(numpy.zeros(3), numpy.eye(3))
	d4 = MultivariateGaussianDistribution2(numpy.ones(3), numpy.eye(3))
	model2 = GeneralMixtureModel([d3, d4])

	assert_array_almost_equal(model1.predict_proba(X), model2.predict_proba(X))

def test_custom_mgd_gmm_predict_log_proba():
	X = numpy.random.normal(0.5, 1, size=(20,3))

	d1 = MultivariateGaussianDistribution(numpy.zeros(3), numpy.eye(3))
	d2 = MultivariateGaussianDistribution(numpy.ones(3), numpy.eye(3))
	model1 = GeneralMixtureModel([d1, d2])
	

	d3 = MultivariateGaussianDistribution2(numpy.zeros(3), numpy.eye(3))
	d4 = MultivariateGaussianDistribution2(numpy.ones(3), numpy.eye(3))
	model2 = GeneralMixtureModel([d3, d4])

	assert_array_almost_equal(model1.predict_log_proba(X), model2.predict_log_proba(X))

def test_custom_mgd_gmm_fit():
	X = numpy.random.normal(0, 1, size=(500,3))
	X[::2] += 1

	d1 = MultivariateGaussianDistribution(numpy.zeros(3), numpy.eye(3))
	d2 = MultivariateGaussianDistribution(numpy.ones(3), numpy.eye(3))
	model1 = GeneralMixtureModel([d1, d2])
	model1.fit(X)

	d3 = MultivariateGaussianDistribution2(numpy.zeros(3), numpy.eye(3))
	d4 = MultivariateGaussianDistribution2(numpy.ones(3), numpy.eye(3))
	model2 = GeneralMixtureModel([d3, d4])
	model2.fit(X)

	assert_array_almost_equal(model1.log_probability(X), model2.log_probability(X))

def test_custom_mgd_gmm_from_samples():
	X = numpy.random.normal(0, 1, size=(500,3))
	X[::2] += 1

	model1 = GeneralMixtureModel.from_samples(MultivariateGaussianDistribution, 2, X, init='first-k', max_iterations=5)
	model2 = GeneralMixtureModel.from_samples(MultivariateGaussianDistribution2, 2, X, init='first-k', max_iterations=5)

	assert_array_almost_equal(model1.log_probability(X), model2.log_probability(X))

def test_custom_icd_gmm_init():
	d1 = IndependentComponentsDistribution([NormalDistribution2(0, 1) for _ in range(5)])
	d2 = IndependentComponentsDistribution([NormalDistribution2(1, 1) for _ in range(5)])
	model = GeneralMixtureModel([d1, d2])

	assert_equal(d1.d, 5)
	assert_equal(d2.d, 5)
	assert_equal(model.d, 5)

def test_custom_icd_gmm_logp():
	X = numpy.random.normal(0.5, 1, size=(20,5))

	d1 = IndependentComponentsDistribution([NormalDistribution(0, 1) for _ in range(5)])
	d2 = IndependentComponentsDistribution([NormalDistribution(1, 1) for _ in range(5)])
	model1 = GeneralMixtureModel([d1, d2])
	

	d3 = IndependentComponentsDistribution([NormalDistribution2(0, 1) for _ in range(5)])
	d4 = IndependentComponentsDistribution([NormalDistribution2(1, 1) for _ in range(5)])
	model2 = GeneralMixtureModel([d3, d4])

	assert_array_almost_equal(model1.log_probability(X), model2.log_probability(X))

def test_custom_icd_gmm_predict():
	X = numpy.random.normal(0.5, 1, size=(20,5))

	d1 = IndependentComponentsDistribution([NormalDistribution(0, 1) for _ in range(5)])
	d2 = IndependentComponentsDistribution([NormalDistribution(1, 1) for _ in range(5)])
	model1 = GeneralMixtureModel([d1, d2])
	

	d3 = IndependentComponentsDistribution([NormalDistribution2(0, 1) for _ in range(5)])
	d4 = IndependentComponentsDistribution([NormalDistribution2(1, 1) for _ in range(5)])
	model2 = GeneralMixtureModel([d3, d4])

	assert_array_almost_equal(model1.predict(X), model2.predict(X))

def test_custom_icd_gmm_predict_proba():
	X = numpy.random.normal(0.5, 1, size=(20,5))

	d1 = IndependentComponentsDistribution([NormalDistribution(0, 1) for _ in range(5)])
	d2 = IndependentComponentsDistribution([NormalDistribution(1, 1) for _ in range(5)])
	model1 = GeneralMixtureModel([d1, d2])
	

	d3 = IndependentComponentsDistribution([NormalDistribution2(0, 1) for _ in range(5)])
	d4 = IndependentComponentsDistribution([NormalDistribution2(1, 1) for _ in range(5)])
	model2 = GeneralMixtureModel([d3, d4])

	assert_array_almost_equal(model1.predict_proba(X), model2.predict_proba(X))

def test_custom_icd_gmm_predict_log_proba():
	X = numpy.random.normal(0.5, 1, size=(20,5))

	d1 = IndependentComponentsDistribution([NormalDistribution(0, 1) for _ in range(5)])
	d2 = IndependentComponentsDistribution([NormalDistribution(1, 1) for _ in range(5)])
	model1 = GeneralMixtureModel([d1, d2])
	

	d3 = IndependentComponentsDistribution([NormalDistribution2(0, 1) for _ in range(5)])
	d4 = IndependentComponentsDistribution([NormalDistribution2(1, 1) for _ in range(5)])
	model2 = GeneralMixtureModel([d3, d4])

	assert_array_almost_equal(model1.predict_log_proba(X), model2.predict_log_proba(X))

def test_custom_icd_gmm_fit():
	X = numpy.random.normal(0.5, 1, size=(20,5))

	d1 = IndependentComponentsDistribution([NormalDistribution(0, 1) for _ in range(5)])
	d2 = IndependentComponentsDistribution([NormalDistribution(1, 1) for _ in range(5)])
	model1 = GeneralMixtureModel([d1, d2])
	model1.fit(X)

	d3 = IndependentComponentsDistribution([NormalDistribution2(0, 1) for _ in range(5)])
	d4 = IndependentComponentsDistribution([NormalDistribution2(1, 1) for _ in range(5)])
	model2 = GeneralMixtureModel([d3, d4])
	model2.fit(X)

	assert_array_almost_equal(model1.log_probability(X), model2.log_probability(X))

def test_custom_icd_gmm_from_samples():
	X = numpy.random.normal(0.0, 1, size=(200,5))
	X[::2] += 1

	model1 = GeneralMixtureModel.from_samples(NormalDistribution, 2, X, init='first-k', max_iterations=5)
	model2 = GeneralMixtureModel.from_samples(NormalDistribution2, 2, X, init='first-k', max_iterations=5)

	assert_array_almost_equal(model1.log_probability(X), model2.log_probability(X))

def test_custom_normal_nb_init():
	d1 = NormalDistribution2(0, 1)
	d2 = NormalDistribution2(1, 1)
	model = NaiveBayes([d1, d2])

	assert_equal(d1.d, 1)
	assert_equal(d2.d, 1)
	assert_equal(model.d, 1)

def test_custom_normal_nb_logp():
	X = numpy.random.normal(0.5, 1, size=(20, 1))

	d1 = NormalDistribution(0, 1)
	d2 = NormalDistribution(1, 1)
	model1 = NaiveBayes([d1, d2])
	

	d3 = NormalDistribution(0, 1)
	d4 = NormalDistribution(1, 1)
	model2 = NaiveBayes([d3, d4])

	assert_array_almost_equal(model1.log_probability(X), model2.log_probability(X))

def test_custom_normal_nb_predict():
	X = numpy.random.normal(0.5, 1, size=(20, 1))

	d1 = NormalDistribution(0, 1)
	d2 = NormalDistribution(1, 1)
	model1 = NaiveBayes([d1, d2])
	

	d3 = NormalDistribution(0, 1)
	d4 = NormalDistribution(1, 1)
	model2 = NaiveBayes([d3, d4])

	assert_array_equal(model1.predict(X), model2.predict(X))

def test_custom_normal_nb_predict_proba():
	X = numpy.random.normal(0.5, 1, size=(20, 1))

	d1 = NormalDistribution(0, 1)
	d2 = NormalDistribution(1, 1)
	model1 = NaiveBayes([d1, d2])
	

	d3 = NormalDistribution(0, 1)
	d4 = NormalDistribution(1, 1)
	model2 = NaiveBayes([d3, d4])

	assert_array_almost_equal(model1.predict_proba(X), model2.predict_proba(X))

def test_custom_normal_nb_predict_log_proba():
	X = numpy.random.normal(0.5, 1, size=(20, 1))

	d1 = NormalDistribution(0, 1)
	d2 = NormalDistribution(1, 1)
	model1 = NaiveBayes([d1, d2])
	

	d3 = NormalDistribution(0, 1)
	d4 = NormalDistribution(1, 1)
	model2 = NaiveBayes([d3, d4])

	assert_array_almost_equal(model1.predict_log_proba(X), model2.predict_log_proba(X))

def test_custom_normal_nb_fit():
	X = numpy.random.normal(0.5, 1, size=(20, 1))
	y = numpy.zeros(20)
	y[::2] = 1

	d1 = NormalDistribution(0, 1)
	d2 = NormalDistribution(1, 1)
	model1 = NaiveBayes([d1, d2])
	model1.fit(X, y)

	d3 = NormalDistribution2(0, 1)
	d4 = NormalDistribution2(1, 1)
	model2 = NaiveBayes([d3, d4])
	model2.fit(X, y)

	assert_array_almost_equal(model1.log_probability(X), model2.log_probability(X))

def test_custom_normal_nb_from_samples():
	X = numpy.random.normal(0.5, 1, size=(200, 1))
	X[::2] += 1
	y = numpy.zeros(200)
	y[::2] = 1

	model1 = NaiveBayes.from_samples(NormalDistribution, X, y)
	model2 = NaiveBayes.from_samples(NormalDistribution2, X, y)

	assert_array_almost_equal(model1.log_probability(X), model2.log_probability(X))

def test_custom_mgd_bc_init():
	d1 = MultivariateGaussianDistribution2(numpy.zeros(3), numpy.eye(3))
	d2 = MultivariateGaussianDistribution2(numpy.ones(3), numpy.eye(3))
	model = BayesClassifier([d1, d2])

	assert_equal(d1.d, 3)
	assert_equal(d2.d, 3)
	assert_equal(model.d, 3)

def test_custom_mgd_bc_logp():
	X = numpy.random.normal(0.5, 1, size=(20,3))

	d1 = MultivariateGaussianDistribution(numpy.zeros(3), numpy.eye(3))
	d2 = MultivariateGaussianDistribution(numpy.ones(3), numpy.eye(3))
	model1 = BayesClassifier([d1, d2])
	

	d3 = MultivariateGaussianDistribution2(numpy.zeros(3), numpy.eye(3))
	d4 = MultivariateGaussianDistribution2(numpy.ones(3), numpy.eye(3))
	model2 = BayesClassifier([d3, d4])

	assert_array_almost_equal(model1.log_probability(X), model2.log_probability(X))

def test_custom_mgd_bc_predict():
	X = numpy.random.normal(0.5, 1, size=(20,3))

	d1 = MultivariateGaussianDistribution(numpy.zeros(3), numpy.eye(3))
	d2 = MultivariateGaussianDistribution(numpy.ones(3), numpy.eye(3))
	model1 = BayesClassifier([d1, d2])

	d3 = MultivariateGaussianDistribution2(numpy.zeros(3), numpy.eye(3))
	d4 = MultivariateGaussianDistribution2(numpy.ones(3), numpy.eye(3))
	model2 = BayesClassifier([d3, d4])

	assert_array_equal(model1.predict(X), model2.predict(X))

def test_custom_mgd_bc_predict_proba():
	X = numpy.random.normal(0.5, 1, size=(20,3))

	d1 = MultivariateGaussianDistribution(numpy.zeros(3), numpy.eye(3))
	d2 = MultivariateGaussianDistribution(numpy.ones(3), numpy.eye(3))
	model1 = BayesClassifier([d1, d2])
	

	d3 = MultivariateGaussianDistribution2(numpy.zeros(3), numpy.eye(3))
	d4 = MultivariateGaussianDistribution2(numpy.ones(3), numpy.eye(3))
	model2 = BayesClassifier([d3, d4])

	assert_array_almost_equal(model1.predict_proba(X), model2.predict_proba(X))

def test_custom_mgd_bc_predict_log_proba():
	X = numpy.random.normal(0.5, 1, size=(20,3))

	d1 = MultivariateGaussianDistribution(numpy.zeros(3), numpy.eye(3))
	d2 = MultivariateGaussianDistribution(numpy.ones(3), numpy.eye(3))
	model1 = BayesClassifier([d1, d2])
	

	d3 = MultivariateGaussianDistribution2(numpy.zeros(3), numpy.eye(3))
	d4 = MultivariateGaussianDistribution2(numpy.ones(3), numpy.eye(3))
	model2 = BayesClassifier([d3, d4])

	assert_array_almost_equal(model1.predict_log_proba(X), model2.predict_log_proba(X))

def test_custom_mgd_bc_fit():
	X = numpy.random.normal(0, 1, size=(500,3))
	X[::2] += 1
	y = numpy.zeros(500)
	y[::2] = 1

	d1 = MultivariateGaussianDistribution(numpy.zeros(3), numpy.eye(3))
	d2 = MultivariateGaussianDistribution(numpy.ones(3), numpy.eye(3))
	model1 = BayesClassifier([d1, d2])
	model1.fit(X, y)

	d3 = MultivariateGaussianDistribution2(numpy.zeros(3), numpy.eye(3))
	d4 = MultivariateGaussianDistribution2(numpy.ones(3), numpy.eye(3))
	model2 = BayesClassifier([d3, d4])
	model2.fit(X, y)

	assert_array_almost_equal(model1.log_probability(X), model2.log_probability(X))

def test_custom_mgd_bc_from_samples():
	X = numpy.random.normal(0, 1, size=(500,3))
	X[::2] += 1
	y = numpy.zeros(500)
	y[::2] = 1

	model1 = BayesClassifier.from_samples(MultivariateGaussianDistribution, X, y)
	model2 = BayesClassifier.from_samples(MultivariateGaussianDistribution2, X, y)

	assert_array_almost_equal(model1.log_probability(X), model2.log_probability(X))

def test_custom_icd_nb_init():
	d1 = IndependentComponentsDistribution([NormalDistribution2(0, 1) for _ in range(5)])
	d2 = IndependentComponentsDistribution([NormalDistribution2(1, 1) for _ in range(5)])
	model = NaiveBayes([d1, d2])

	assert_equal(d1.d, 5)
	assert_equal(d2.d, 5)
	assert_equal(model.d, 5)

def test_custom_icd_nb_logp():
	X = numpy.random.normal(0.5, 1, size=(20,5))

	d1 = IndependentComponentsDistribution([NormalDistribution(0, 1) for _ in range(5)])
	d2 = IndependentComponentsDistribution([NormalDistribution(1, 1) for _ in range(5)])
	model1 = NaiveBayes([d1, d2])
	

	d3 = IndependentComponentsDistribution([NormalDistribution2(0, 1) for _ in range(5)])
	d4 = IndependentComponentsDistribution([NormalDistribution2(1, 1) for _ in range(5)])
	model2 = NaiveBayes([d3, d4])

	assert_array_almost_equal(model1.log_probability(X), model2.log_probability(X))

def test_custom_icd_nb_predict():
	X = numpy.random.normal(0.5, 1, size=(20,5))

	d1 = IndependentComponentsDistribution([NormalDistribution(0, 1) for _ in range(5)])
	d2 = IndependentComponentsDistribution([NormalDistribution(1, 1) for _ in range(5)])
	model1 = NaiveBayes([d1, d2])
	

	d3 = IndependentComponentsDistribution([NormalDistribution2(0, 1) for _ in range(5)])
	d4 = IndependentComponentsDistribution([NormalDistribution2(1, 1) for _ in range(5)])
	model2 = NaiveBayes([d3, d4])

	assert_array_almost_equal(model1.predict(X), model2.predict(X))

def test_custom_icd_nb_predict_proba():
	X = numpy.random.normal(0.5, 1, size=(20,5))

	d1 = IndependentComponentsDistribution([NormalDistribution(0, 1) for _ in range(5)])
	d2 = IndependentComponentsDistribution([NormalDistribution(1, 1) for _ in range(5)])
	model1 = NaiveBayes([d1, d2])
	

	d3 = IndependentComponentsDistribution([NormalDistribution2(0, 1) for _ in range(5)])
	d4 = IndependentComponentsDistribution([NormalDistribution2(1, 1) for _ in range(5)])
	model2 = NaiveBayes([d3, d4])

	assert_array_almost_equal(model1.predict_proba(X), model2.predict_proba(X))

def test_custom_icd_nb_predict_log_proba():
	X = numpy.random.normal(0.5, 1, size=(20,5))

	d1 = IndependentComponentsDistribution([NormalDistribution(0, 1) for _ in range(5)])
	d2 = IndependentComponentsDistribution([NormalDistribution(1, 1) for _ in range(5)])
	model1 = NaiveBayes([d1, d2])
	

	d3 = IndependentComponentsDistribution([NormalDistribution2(0, 1) for _ in range(5)])
	d4 = IndependentComponentsDistribution([NormalDistribution2(1, 1) for _ in range(5)])
	model2 = NaiveBayes([d3, d4])

	assert_array_almost_equal(model1.predict_log_proba(X), model2.predict_log_proba(X))

def test_custom_icd_nb_fit():
	X = numpy.random.normal(0.5, 1, size=(20,5))
	y = numpy.zeros(20)
	y[::2] = 1

	d1 = IndependentComponentsDistribution([NormalDistribution(0, 1) for _ in range(5)])
	d2 = IndependentComponentsDistribution([NormalDistribution(1, 1) for _ in range(5)])
	model1 = NaiveBayes([d1, d2])
	model1.fit(X, y)

	d3 = IndependentComponentsDistribution([NormalDistribution2(0, 1) for _ in range(5)])
	d4 = IndependentComponentsDistribution([NormalDistribution2(1, 1) for _ in range(5)])
	model2 = NaiveBayes([d3, d4])
	model2.fit(X, y)

	assert_array_almost_equal(model1.log_probability(X), model2.log_probability(X))

def test_custom_icd_nb_from_samples():
	X = numpy.random.normal(0.0, 1, size=(200,5))
	X[::2] += 1
	y = numpy.zeros(200)
	y[::2] = 1

	model1 = NaiveBayes.from_samples(NormalDistribution, X, y)
	model2 = NaiveBayes.from_samples(NormalDistribution2, X, y)

	assert_array_almost_equal(model1.log_probability(X), model2.log_probability(X))

@with_setup(setup_normal_hmm)
def test_custom_normal_hmm_init():
	assert_equal(model1.d, 1)
	assert_equal(model2.d, 1)

@with_setup(setup_normal_hmm)
def test_custom_normal_hmm_logp():
	X = numpy.random.normal(0.5, 1, size=(5, 20, 1))
	assert_array_almost_equal(model1.log_probability(X), model2.log_probability(X))

@with_setup(setup_normal_hmm)
def test_custom_normal_hmm_predict():
	X = numpy.random.normal(0.5, 1, size=(5, 20, 1))
	assert_array_equal(model1.predict(X), model2.predict(X))

@with_setup(setup_normal_hmm)
def test_custom_normal_hmm_predict_proba():
	X = numpy.random.normal(0.5, 1, size=(5, 20, 1))
	assert_array_almost_equal(model1.predict_proba(X), model2.predict_proba(X))

@with_setup(setup_normal_hmm)
def test_custom_normal_hmm_predict_log_proba():
	X = numpy.random.normal(0.5, 1, size=(5, 20, 1))
	assert_array_almost_equal(model1.predict_log_proba(X), model2.predict_log_proba(X))

@with_setup(setup_normal_hmm)
def test_custom_normal_hmm_fit():
	X = numpy.random.normal(0.5, 1, size=(5, 20, 1))

	model1.fit(X, max_iterations=5)
	model2.fit(X, max_iterations=5)

	assert_array_almost_equal(model1.log_probability(X), model2.log_probability(X))

def test_custom_normal_hmm_from_samples():
	X = numpy.random.normal(0.5, 1, size=(2, 200, 1))
	X[::2] += 1

	model1 = HiddenMarkovModel.from_samples(NormalDistribution, 2, X, init='first-k', max_iterations=5)
	model2 = HiddenMarkovModel.from_samples(NormalDistribution2, 2, X, init='first-k', max_iterations=5)

	assert_array_almost_equal(model1.log_probability(X), model2.log_probability(X))

@with_setup(setup_mgd_hmm)
def test_custom_mgd_hmm_init():
	assert_equal(model1.d, 3)
	assert_equal(model2.d, 3)

@with_setup(setup_mgd_hmm)
def test_custom_mgd_hmm_logp():
	X = numpy.random.normal(0.5, 1, size=(5, 20, 3))
	assert_array_almost_equal(model1.log_probability(X), model2.log_probability(X))

@with_setup(setup_mgd_hmm)
def test_custom_mgd_hmm_predict():
	X = numpy.random.normal(0.5, 1, size=(5, 20, 3))
	assert_array_equal(model1.predict(X), model2.predict(X))

@with_setup(setup_mgd_hmm)
def test_custom_mgd_hmm_predict_proba():
	X = numpy.random.normal(0.5, 1, size=(5, 20, 3))
	assert_array_almost_equal(model1.predict_proba(X), model2.predict_proba(X))

@with_setup(setup_mgd_hmm)
def test_custom_mgd_hmm_predict_log_proba():
	X = numpy.random.normal(0.5, 1, size=(5, 20, 3))
	assert_array_almost_equal(model1.predict_log_proba(X), model2.predict_log_proba(X))

@with_setup(setup_mgd_hmm)
def test_custom_mgd_hmm_fit():
	X = numpy.random.normal(0, 0.1, size=(2, 50, 3))
	X[:, ::2] += 1

	model1.fit(X, max_iterations=5)
	model2.fit(X, max_iterations=5)

	assert_array_almost_equal(model1.log_probability(X), model2.log_probability(X))

def test_custom_mgd_hmm_from_samples():
	X = numpy.random.normal(0, 1, size=(2, 50, 3))
	X[:, ::2] += 1

	model1 = HiddenMarkovModel.from_samples(MultivariateGaussianDistribution, 2, X, init='first-k', max_iterations=5)
	model2 = HiddenMarkovModel.from_samples(MultivariateGaussianDistribution2, 2, X, init='first-k', max_iterations=5)

	assert_array_almost_equal(model1.log_probability(X), model2.log_probability(X))

@with_setup(setup_icd_hmm)
def test_custom_icd_hmm_init():
	assert_equal(model1.d, 5)
	assert_equal(model2.d, 5)

@with_setup(setup_icd_hmm)
def test_custom_icd_hmm_logp():
	X = numpy.random.normal(0.5, 1, size=(5, 20, 5))
	assert_array_almost_equal(model1.log_probability(X), model2.log_probability(X))

@with_setup(setup_icd_hmm)
def test_custom_icd_hmm_predict():
	X = numpy.random.normal(0.5, 1, size=(5, 20, 5))
	assert_array_almost_equal(model1.predict(X), model2.predict(X))

@with_setup(setup_icd_hmm)
def test_custom_icd_hmm_predict_proba():
	X = numpy.random.normal(0.5, 1, size=(5, 20, 5))
	assert_array_almost_equal(model1.predict_proba(X), model2.predict_proba(X))

@with_setup(setup_icd_hmm)
def test_custom_icd_hmm_predict_log_proba():
	X = numpy.random.normal(0.5, 1, size=(5, 20, 5))
	assert_array_almost_equal(model1.predict_log_proba(X), model2.predict_log_proba(X))

@with_setup(setup_icd_hmm)
def test_custom_icd_hmm_fit():
	X = numpy.random.normal(0.5, 1, size=(3, 20, 5))

	model1.fit(X, max_iterations=5)
	model2.fit(X, max_iterations=5)

	assert_array_almost_equal(model1.log_probability(X), model2.log_probability(X))

def test_custom_icd_hmm_from_samples():
	X = numpy.random.normal(0.0, 1, size=(2, 20, 5))
	X[::2] += 1

	model1 = HiddenMarkovModel.from_samples(NormalDistribution, 2, X, init='first-k', max_iterations=5)
	model2 = HiddenMarkovModel.from_samples(NormalDistribution2, 2, X, init='first-k', max_iterations=5)

	assert_array_almost_equal(model1.log_probability(X), model2.log_probability(X))


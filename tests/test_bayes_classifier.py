from __future__ import (division)

from pomegranate import *
from nose.tools import with_setup
from nose.tools import assert_almost_equal
from nose.tools import assert_equal
from nose.tools import assert_not_equal
from nose.tools import assert_less_equal
from nose.tools import assert_raises
from nose.tools import assert_true
import random
import pickle
import numpy as np

def setup_multivariate():
	global model
	global multi
	global indie

	multi = MultivariateGaussianDistribution([5, 5], [[2, 0], [0, 2]])
	indie = IndependentComponentsDistribution([UniformDistribution(0, 10), UniformDistribution(0, 10)])
	model = NaiveBayes([multi, indie])

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

def teardown():
	pass

@with_setup(setup_multivariate, teardown)
def test_multivariate_distributions():
	assert_almost_equal(multi.log_probability([11, 11]), -20.531024246969945)
	assert_almost_equal(multi.log_probability([9, 9]), -10.531024246969945)
	assert_almost_equal(multi.log_probability([7, 7]), -4.531024246969945)
	assert_almost_equal(multi.log_probability([5, 5]), -2.5310242469699453)
	assert_almost_equal(multi.log_probability([3, 3]), -4.531024246969945)
	assert_almost_equal(multi.log_probability([1, 1]), -10.531024246969945)
	assert_almost_equal(multi.log_probability([-1, -1]), -20.531024246969945)

	assert_almost_equal(indie.log_probability([11, 11]), -float('inf'))
	assert_almost_equal(indie.log_probability([10, 10]), -4.605170185988091)
	assert_almost_equal(indie.log_probability([5, 5]), -4.605170185988091)
	assert_almost_equal(indie.log_probability([0, 0]), -4.605170185988091)
	assert_almost_equal(indie.log_probability([-1, -1]), -float('inf'))

	assert_equal(model.d, 2)

@with_setup(setup_multivariate, teardown)
def test_multivariate_log_proba():
	logs = model.predict_log_proba(np.array([[5, 5], [2, 9], [10, 7], [-1 ,7]]))

	assert_almost_equal(logs[0][0], -0.11837282271439786)
	assert_almost_equal(logs[0][1], -2.1925187617325435)

	assert_almost_equal(logs[1][0], -4.1910993250497741)
	assert_almost_equal(logs[1][1], -0.015245264067919706)

	assert_almost_equal(logs[2][0], -5.1814895400206806)
	assert_almost_equal(logs[2][1], -0.0056354790388262188)

	assert_almost_equal(logs[3][0], 0.0)
	assert_almost_equal(logs[3][1], -float('inf'))

@with_setup(setup_multivariate, teardown)
def test_multivariate_proba():
	probs = model.predict_proba(np.array([[5, 5], [2, 9], [10, 7], [-1, 7]]))

	assert_almost_equal(probs[0][0], 0.88836478829527532)
	assert_almost_equal(probs[0][1], 0.11163521170472469)

	assert_almost_equal(probs[1][0], 0.015129643331582699)
	assert_almost_equal(probs[1][1], 0.98487035666841727)

	assert_almost_equal(probs[2][0], 0.0056196295140261846)
	assert_almost_equal(probs[2][1], 0.99438037048597383)

	assert_almost_equal(probs[3][0], 1.0)
	assert_almost_equal(probs[3][1], 0.0)

@with_setup(setup_multivariate, teardown)
def test_multivariate_prediction():
	predicts = model.predict(np.array([[5, 5], [2, 9], [10, 7], [-1, 7]]))

	assert_equal(predicts[0], 0)
	assert_equal(predicts[1], 1)
	assert_equal(predicts[2], 1)
	assert_equal(predicts[3], 0)

@with_setup(setup_multivariate, teardown)
def test_multivariate_fit():
	X = np.array([[6, 5], [3.5, 4], [4, 6], [8, 6.5], [3.5, 4], [4.5, 5.5],
				  [0, 7], [0.5, 7.5], [9.5, 8], [5, 0.5], [7.5, 1.5], [7, 7]])
	y = np.array([0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1])

	model.fit( X, y )

	data = np.array([[5, 5], [2, 3], [10, 7], [-1, 7]])

	# test multivariate log probabilities
	logs = model.predict_log_proba(data)

	assert_almost_equal(logs[0][0], -0.09672086616254516)
	assert_almost_equal(logs[0][1], -2.3838967922368868)
	assert_almost_equal(logs[1][0], -0.884636340789835)
	assert_almost_equal(logs[1][1], -0.53249928976493077)
	assert_almost_equal(logs[2][0], 0.0 )
	assert_almost_equal(logs[2][1], -float('inf'))
	assert_almost_equal(logs[3][0], 0.0 )
	assert_almost_equal(logs[3][1], -float('inf'))

	# test multivariate probabilities
	probs = model.predict_proba( data )

	assert_almost_equal(probs[0][0], 0.90780937108369053)
	assert_almost_equal(probs[0][1], 0.092190628916309608)
	assert_almost_equal(probs[1][0], 0.41286428788295315)
	assert_almost_equal(probs[1][1], 0.58713571211704685)
	assert_almost_equal(probs[2][0], 1.0)
	assert_almost_equal(probs[2][1], 0.0)
	assert_almost_equal(probs[3][0], 1.0)
	assert_almost_equal(probs[3][1], 0.0)

	# test multivariate classifications
	predicts = model.predict(data)

	assert_equal(predicts[0], 0)
	assert_equal(predicts[1], 1)
	assert_equal(predicts[2], 0)
	assert_equal(predicts[3], 0)

@with_setup(setup_multivariate, teardown)
def test_raise_errors():
	assert_raises(ValueError, model.predict_log_proba, 5)
	assert_raises(ValueError, model.predict_log_proba, [5] )
	assert_raises(ValueError, model.predict_log_proba, [[1], [2], [3], [4]] )
	assert_raises(ValueError, model.predict_log_proba, [[1, 2, 3], [4, 5, 6]] )

	assert_raises(ValueError, model.predict_proba, 5 )
	assert_raises(ValueError, model.predict_proba, [5] )
	assert_raises(ValueError, model.predict_proba, [[1], [2], [3], [4]] )
	assert_raises(ValueError, model.predict_proba, [[1, 2, 3],[4, 5, 6]] )

	assert_raises( ValueError, model.predict, 5 )
	assert_raises( ValueError, model.predict, [5] )
	assert_raises( ValueError, model.predict, [[1], [2], [3], [4]] )
	assert_raises( ValueError, model.predict, [[1, 2, 3], [4, 5, 6]] )

@with_setup(setup_multivariate, teardown)
def test_pickling():
	j_multi = pickle.dumps(model)
	new_multi = pickle.loads(j_multi)
	assert_true(isinstance(new_multi.distributions[0], MultivariateGaussianDistribution))
	assert_true(isinstance(new_multi.distributions[1], IndependentComponentsDistribution))
	assert_true(isinstance(new_multi, NaiveBayes))
	numpy.testing.assert_array_equal(model.weights, new_multi.weights)

@with_setup(setup_multivariate, teardown)
def test_json():
	j_multi = model.to_json()
	new_multi = model.from_json(j_multi)
	assert_true(isinstance(new_multi.distributions[0], MultivariateGaussianDistribution))
	assert_true(isinstance(new_multi.distributions[1], IndependentComponentsDistribution))
	assert_true(isinstance(new_multi, NaiveBayes))
	numpy.testing.assert_array_equal(model.weights, new_multi.weights)

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
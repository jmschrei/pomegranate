from __future__ import (division)

from pomegranate import *
from nose.tools import with_setup
from nose.tools import assert_almost_equal
from nose.tools import assert_equal
from nose.tools import assert_not_equal
from nose.tools import assert_less_equal
from nose.tools import assert_raises
import random
import pickle
import numpy as np

def setup_univariate():
	global normal
	global uniform
	global univariate

	normal = NormalDistribution( 5, 2 )
	uniform = UniformDistribution( 0, 10 )
	univariate = NaiveBayes([ normal, uniform ])

def setup_multivariate():
	global multivariate
	global multi
	global indie

	multi = MultivariateGaussianDistribution( means=[ 5, 5 ], covariance=[[ 2, 0 ], [ 0, 2 ]] )
	indie = IndependentComponentsDistribution( distributions=[ UniformDistribution(0, 10), UniformDistribution(0, 10) ])
	multivariate = NaiveBayes([ multi, indie ])

def setup_hmm():
	global hmms
	global dumb
	global fair
	global smart

	rigged = State( DiscreteDistribution({ 'H': 0.8, 'T': 0.2 }) )
	unrigged = State( DiscreteDistribution({ 'H': 0.5, 'T':0.5 }) )

	dumb = HiddenMarkovModel()
	dumb.start = rigged
	dumb.add_transition( rigged, rigged, 1 )
	dumb.bake()

	fair = HiddenMarkovModel()
	fair.start = unrigged
	fair.add_transition( unrigged, unrigged, 1 )
	fair.bake()

	smart = HiddenMarkovModel()
	smart.add_transition( smart.start, unrigged, 0.5 )
	smart.add_transition( smart.start, rigged, 0.5 )
	smart.add_transition( rigged, rigged, 0.5 )
	smart.add_transition( rigged, unrigged, 0.5 )
	smart.add_transition( unrigged, rigged, 0.5 )
	smart.add_transition( unrigged, unrigged, 0.5 )
	smart.bake()

	hmms = NaiveBayes([ dumb, fair, smart ])

def setup_bayesnet():
	# TODO: create test cases for bayesian networks
	pass

def setup_all():
	setup_univariate()
	setup_multivariate()
	setup_hmm()
	setup_bayesnet()

def teardown():
	pass

@with_setup( setup_univariate, teardown )
def test_univariate_distributions():
	assert_almost_equal( normal.log_probability( 11 ), -6.112085713764219 )
	assert_almost_equal( normal.log_probability( 9 ), -3.612085713764219 )
	assert_almost_equal( normal.log_probability( 7 ), -2.112085713764219 )
	assert_almost_equal( normal.log_probability( 5 ), -1.612085713764219 )
	assert_almost_equal( normal.log_probability( 3 ), -2.112085713764219 )
	assert_almost_equal( normal.log_probability( 1 ), -3.612085713764219 )
	assert_almost_equal( normal.log_probability( -1 ), -6.112085713764219 )

	assert_almost_equal( uniform.log_probability( 11 ), -float('inf') )
	assert_almost_equal( uniform.log_probability( 10 ), -2.3025850929940455 )
	assert_almost_equal( uniform.log_probability( 5 ), -2.3025850929940455 )
	assert_almost_equal( uniform.log_probability( 0 ), -2.3025850929940455 )
	assert_almost_equal( uniform.log_probability( -1 ), -float('inf') )

	assert_equal( univariate.d, 1 )

@with_setup( setup_multivariate, teardown )
def test_multivariate_distributions():
	assert_almost_equal( multi.log_probability([ 11, 11 ]), -20.531024246969945 )
	assert_almost_equal( multi.log_probability([ 9, 9 ]), -10.531024246969945 )
	assert_almost_equal( multi.log_probability([ 7, 7 ]), -4.531024246969945 )
	assert_almost_equal( multi.log_probability([ 5, 5 ]), -2.5310242469699453 )
	assert_almost_equal( multi.log_probability([ 3, 3 ]), -4.531024246969945 )
	assert_almost_equal( multi.log_probability([ 1, 1 ]), -10.531024246969945 )
	assert_almost_equal( multi.log_probability([ -1, -1 ]), -20.531024246969945 )

	assert_almost_equal( indie.log_probability([ 11, 11 ]), -float('inf') )
	assert_almost_equal( indie.log_probability([ 10, 10 ]), -4.605170185988091 )
	assert_almost_equal( indie.log_probability([ 5, 5 ]), -4.605170185988091 )
	assert_almost_equal( indie.log_probability([ 0, 0 ]), -4.605170185988091 )
	assert_almost_equal( indie.log_probability([ -1, -1 ]), -float('inf') )

	assert_equal( multivariate.d, 2 )

@with_setup( setup_hmm, teardown )
def test_hmms():
	assert_almost_equal( dumb.log_probability( list('H') ), -0.2231435513142097 )
	assert_almost_equal( dumb.log_probability( list('T') ), -1.6094379124341003 )
	assert_almost_equal( dumb.log_probability( list('HHHH') ), -0.8925742052568388 )
	assert_almost_equal( dumb.log_probability( list('THHH') ), -2.2788685663767296 )
	assert_almost_equal( dumb.log_probability( list('TTTT') ), -6.437751649736401 )

	assert_almost_equal( fair.log_probability( list('H') ), -0.6931471805599453 )
	assert_almost_equal( fair.log_probability( list('T') ), -0.6931471805599453 )
	assert_almost_equal( fair.log_probability( list('HHHH') ), -2.772588722239781 )
	assert_almost_equal( fair.log_probability( list('THHH') ), -2.772588722239781 )
	assert_almost_equal( fair.log_probability( list('TTTT') ), -2.772588722239781 )

	assert_almost_equal( smart.log_probability( list('H') ), -0.43078291609245417 )
	assert_almost_equal( smart.log_probability( list('T') ), -1.0498221244986776 )
	assert_almost_equal( smart.log_probability( list('HHHH') ), -1.7231316643698167 )
	assert_almost_equal( smart.log_probability( list('THHH') ), -2.3421708727760397 )
	assert_almost_equal( smart.log_probability( list('TTTT') ), -4.1992884979947105 )
	assert_almost_equal( smart.log_probability( list('THTHTHTHTHTH') ), -8.883630243546788 )
	assert_almost_equal( smart.log_probability( list('THTHHHHHTHTH') ), -7.645551826734343 )

	assert_equal( hmms.d, 1 )

def test_constructors():
	# check for value error when supplied no information
	assert_raises( ValueError, NaiveBayes )

	# check error is not thrown
	NaiveBayes( NormalDistribution )
	NaiveBayes( MultivariateGaussianDistribution )
	NaiveBayes( HiddenMarkovModel )

	# check if error is thrown
	assert_raises( TypeError, NaiveBayes, [ normal, multi ] )
	assert_raises( TypeError, NaiveBayes, [ NormalDistribution, normal ] )

	# TODO:nick check that predict throws error if input is mismatched and not hmms

@with_setup( setup_univariate, teardown )
def test_univariate_log_proba():
	logs = univariate.predict_log_proba(np.array([ 5, 3, 1, -1 ]))

	assert_almost_equal( logs[0][0], -0.40634848776410526 )
	assert_almost_equal( logs[0][1], -1.0968478669939319 )

	assert_almost_equal( logs[1][0], -0.60242689998689203 )
	assert_almost_equal( logs[1][1], -0.79292627921671865 )

	assert_almost_equal( logs[2][0], -1.5484819556996032 )
	assert_almost_equal( logs[2][1], -0.23898133492942986 )

	assert_almost_equal( logs[3][0], 0.0 )
	assert_almost_equal( logs[3][1], -float('inf') )

@with_setup( setup_multivariate, teardown )
def test_multivariate_log_proba():
	logs = multivariate.predict_log_proba(np.array([[ 5, 5 ], [ 2, 9 ], [ 10, 7 ], [ -1 ,7 ]]))

	assert_almost_equal( logs[0][0], -0.11837282271439786 )
	assert_almost_equal( logs[0][1], -2.1925187617325435 )

	assert_almost_equal( logs[1][0], -4.1910993250497741 )
	assert_almost_equal( logs[1][1], -0.015245264067919706 )

	assert_almost_equal( logs[2][0], -5.1814895400206806 )
	assert_almost_equal( logs[2][1], -0.0056354790388262188 )

	assert_almost_equal( logs[3][0], 0.0 )
	assert_almost_equal( logs[3][1], -float('inf') )

@with_setup( setup_hmm, teardown )
def test_hmm_log_proba():
	logs = hmms.predict_log_proba(np.array([list('H'), list('THHH'), list('TTTT'), list('THTHTHTHTHTH'), list('THTHHHHHTHTH')]))

	assert_almost_equal( logs[0][0], -0.89097292388986515 )
	assert_almost_equal( logs[0][1], -1.3609765531356006 )
	assert_almost_equal( logs[0][2], -1.0986122886681096 )

	assert_almost_equal( logs[1][0], -0.93570553121744293 )
	assert_almost_equal( logs[1][1], -1.429425687080494 )
	assert_almost_equal( logs[1][2], -0.9990078376167526 )

	assert_almost_equal( logs[2][0], -3.9007882563128864 )
	assert_almost_equal( logs[2][1], -0.23562532881626597 )
	assert_almost_equal( logs[2][2], -1.6623251045711958 )

	assert_almost_equal( logs[3][0], -3.1703366478831185 )
	assert_almost_equal( logs[3][1], -0.49261403211260379 )
	assert_almost_equal( logs[3][2], -1.058478108940049 )

	assert_almost_equal( logs[4][0], -1.3058441172130273 )
	assert_almost_equal( logs[4][1], -1.4007102236822906 )
	assert_almost_equal( logs[4][2], -0.7284958836972919 )

@with_setup( setup_univariate, teardown )
def test_univariate_proba():
	probs = univariate.predict_proba(np.array([ 5, 3, 1, -1 ]))

	assert_almost_equal( probs[0][0], 0.66607800693933361 )
	assert_almost_equal( probs[0][1], 0.33392199306066628 )

	assert_almost_equal( probs[1][0], 0.54748134004225524 )
	assert_almost_equal( probs[1][1], 0.45251865995774476 )

	assert_almost_equal( probs[2][0], 0.21257042033580209 )
	assert_almost_equal( probs[2][1], 0.78742957966419791 )

	assert_almost_equal( probs[3][0], 1.0 )
	assert_almost_equal( probs[3][1], 0.0 )

@with_setup( setup_multivariate, teardown )
def test_multivariate_proba():
	probs = multivariate.predict_proba(np.array([[ 5, 5 ], [ 2, 9 ], [ 10, 7 ], [ -1, 7 ]]))

	assert_almost_equal( probs[0][0], 0.88836478829527532 )
	assert_almost_equal( probs[0][1], 0.11163521170472469 )

	assert_almost_equal( probs[1][0], 0.015129643331582699 )
	assert_almost_equal( probs[1][1], 0.98487035666841727 )

	assert_almost_equal( probs[2][0], 0.0056196295140261846 )
	assert_almost_equal( probs[2][1], 0.99438037048597383 )

	assert_almost_equal( probs[3][0], 1.0 )
	assert_almost_equal( probs[3][1], 0.0 )

@with_setup( setup_hmm, teardown )
def test_hmm_proba():
	probs = hmms.predict_proba(np.array([list('H'), list('THHH'), list('TTTT'), list('THTHTHTHTHTH'), list('THTHHHHHTHTH')]))

	assert_almost_equal( probs[0][0], 0.41025641025641024 )
	assert_almost_equal( probs[0][1], 0.25641025641025639 )
	assert_almost_equal( probs[0][2], 0.33333333333333331 )

	assert_almost_equal( probs[1][0], 0.39230898163446098 )
	assert_almost_equal( probs[1][1], 0.23944639992337707 )
	assert_almost_equal( probs[1][2], 0.36824461844216183 )

	assert_almost_equal( probs[2][0], 0.020225961918306088 )
	assert_almost_equal( probs[2][1], 0.79007663743383105 )
	assert_almost_equal( probs[2][2], 0.18969740064786292 )

	assert_almost_equal( probs[3][0], 0.041989459861032523 )
	assert_almost_equal( probs[3][1], 0.61102706038265642 )
	assert_almost_equal( probs[3][2], 0.346983479756311 )

	assert_almost_equal( probs[4][0], 0.27094373022369794 )
	assert_almost_equal( probs[4][1], 0.24642188711704707 )
	assert_almost_equal( probs[4][2], 0.48263438265925512 )

@with_setup( setup_univariate, teardown )
def test_univariate_prediction():
	predicts = univariate.predict(np.array([ 5, 3, 1, -1 ]))

	assert_equal( predicts[0], 0 )
	assert_equal( predicts[1], 0 )
	assert_equal( predicts[2], 1 )
	assert_equal( predicts[3], 0 )

@with_setup( setup_multivariate, teardown )
def test_multivariate_prediction():
	predicts = multivariate.predict(np.array([[ 5, 5 ], [ 2, 9 ], [ 10, 7 ], [ -1, 7 ]]))

	assert_equal( predicts[0], 0 )
	assert_equal( predicts[1], 1 )
	assert_equal( predicts[2], 1 )
	assert_equal( predicts[3], 0 )

@with_setup( setup_hmm, teardown )
def test_hmm_prediction():
	predicts = hmms.predict(np.array([list('H'), list('THHH'), list('TTTT'), list('THTHTHTHTHTH'), list('THTHHHHHTHTH')]))

	assert_equal( predicts[0], 0 )
	assert_equal( predicts[1], 0 )
	assert_equal( predicts[2], 1 )
	assert_equal( predicts[3], 1 )
	assert_equal( predicts[4], 2 )

@with_setup( setup_univariate, teardown )
def test_univariate_fit():
	X = np.array([ 5, 4, 5, 4, 6, 5, 6, 5, 4, 6, 5, 4, 0, 0, 1, 9, 8, 2, 0, 1, 1, 8, 10, 0 ])
	y = np.array([ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 ])

	univariate.fit( X, y )

	data = np.array([ 5, 3, 1, -1 ])

	# test univariate log probabilities
	logs = univariate.predict_log_proba( data )

	assert_almost_equal( logs[0][0], -0.1751742330621151 )
	assert_almost_equal( logs[0][1], -1.8282830423560459 )
	assert_almost_equal( logs[1][0], -1.7240463796541046 )
	assert_almost_equal( logs[1][1], -0.19643229738177137 )
	assert_almost_equal( logs[2][0], -11.64810474561418 )
	assert_almost_equal( logs[2][1], -8.7356310092268075e-06 )
	assert_almost_equal( logs[3][0], 0.0 )
	assert_almost_equal( logs[3][1], -float('inf') )

	# test univariate probabilities
	probs = univariate.predict_proba( data )

	assert_almost_equal( probs[0][0], 0.83931077234299012 )
	assert_almost_equal( probs[0][1], 0.1606892276570098 )
	assert_almost_equal( probs[1][0], 0.17834304226047082 )
	assert_almost_equal( probs[1][1], 0.82165695773952918 )
	assert_almost_equal( probs[2][0], 8.7355928537720986e-06 )
	assert_almost_equal( probs[2][1], 0.99999126440714625 )
	assert_almost_equal( probs[3][0], 1.0 )
	assert_almost_equal( probs[3][1], 0.0 )

	# test univariate classifications
	predicts = univariate.predict( data )

	assert_equal( predicts[0], 0 )
	assert_equal( predicts[1], 1 )
	assert_equal( predicts[2], 1 )
	assert_equal( predicts[3], 0 )


@with_setup( setup_multivariate, teardown )
def test_multivariate_fit():
	X = np.array([[ 6, 5 ], [ 3.5, 4 ], [ 4, 6 ], [ 8, 6.5 ], [ 3.5, 4 ], [ 4.5, 5.5 ],
				  [ 0, 7 ], [ 0.5, 7.5 ], [ 9.5, 8 ], [ 5, 0.5 ], [ 7.5, 1.5 ], [ 7, 7 ]])
	y = np.array([ 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1 ])

	multivariate.fit( X, y )

	data = np.array([[ 5, 5 ], [ 2, 3 ], [ 10, 7 ], [ -1, 7 ]])

	# test multivariate log probabilities
	logs = multivariate.predict_log_proba( data )

	assert_almost_equal( logs[0][0], -0.09672086616254516 )
	assert_almost_equal( logs[0][1], -2.3838967922368868 )
	assert_almost_equal( logs[1][0], -0.884636340789835 )
	assert_almost_equal( logs[1][1], -0.53249928976493077 )
	assert_almost_equal( logs[2][0], 0.0 )
	assert_almost_equal( logs[2][1], -float('inf') )
	assert_almost_equal( logs[3][0], 0.0 )
	assert_almost_equal( logs[3][1], -float('inf') )

	# test multivariate probabilities
	probs = multivariate.predict_proba( data )

	assert_almost_equal( probs[0][0], 0.90780937108369053 )
	assert_almost_equal( probs[0][1], 0.092190628916309608 )
	assert_almost_equal( probs[1][0], 0.41286428788295315 )
	assert_almost_equal( probs[1][1], 0.58713571211704685 )
	assert_almost_equal( probs[2][0], 1.0 )
	assert_almost_equal( probs[2][1], 0.0 )
	assert_almost_equal( probs[3][0], 1.0 )
	assert_almost_equal( probs[3][1], 0.0 )

	# test multivariate classifications
	predicts = multivariate.predict( data )

	assert_equal( predicts[0], 0 )
	assert_equal( predicts[1], 1 )
	assert_equal( predicts[2], 0 )
	assert_equal( predicts[3], 0 )

@with_setup( setup_hmm, teardown )
def test_hmm_fit():
	X = np.array([list( 'HHHHHTHTHTTTTH' ),
				  list( 'HHTHHTTHHHHHTH' ),
				  list( 'TH' ), list( 'HHHHT' ),])
	y = np.array([ 2, 2, 1, 0 ])

	hmms.fit( X, y )

	data = np.array([list('H'), list('THHH'), list('HHHH'), list('THTHTHTHTHTH'), list('THTHHHHHTHTH')])

	# test hmm log probabilities
	logs = hmms.predict_log_proba( data )

	assert_almost_equal( logs[0][0], -1.2745564715121378 )
	assert_almost_equal( logs[0][1], -1.8242710481193862 )
	assert_almost_equal( logs[0][2], -0.58140929189714219 )
	assert_almost_equal( logs[1][0], -148.69787686172131 )
	assert_almost_equal( logs[1][1], 0.0 )
	assert_almost_equal( logs[1][2], -148.90503555416012 )
	assert_almost_equal( logs[2][0], -0.65431299454112646 )
	assert_almost_equal( logs[2][1], -2.8531712971903209 )
	assert_almost_equal( logs[2][2], -0.86147167008351733 )
	assert_almost_equal( logs[3][0], -898.78383614856 )
	assert_almost_equal( logs[3][1], 0.0 )
	assert_almost_equal( logs[3][2], -149.77063003020317 )
	assert_almost_equal( logs[4][0], -596.9903657729626 )
	assert_almost_equal( logs[4][1], 0.0 )
	assert_almost_equal( logs[4][2], -149.77062996265178 )

	# test hmm probabilities
	probs = hmms.predict_proba( data )

	assert_almost_equal( probs[0][0], 0.27955493104058637 )
	assert_almost_equal( probs[0][1], 0.16133520687824079 )
	assert_almost_equal( probs[0][2], 0.55910986208117275 )
	assert_almost_equal( probs[1][0], 2.709300362358663e-09 )
	assert_almost_equal( probs[1][1], 0.99999999508833481 )
	assert_almost_equal( probs[1][2], 2.2023649431371957e-09 )
	assert_almost_equal( probs[2][0], 0.51979904472991378 )
	assert_almost_equal( probs[2][1], 0.057661169909141198 )
	assert_almost_equal( probs[2][2], 0.422539785360945 )
	assert_almost_equal( probs[3][0], 5.3986709867980443e-55 )
	assert_almost_equal( probs[3][1], 0.999999999073242 )
	assert_almost_equal( probs[3][2], 9.2675809768728048e-10 )
	assert_almost_equal( probs[4][0], 5.9769084150373497e-36 )
	assert_almost_equal( probs[4][1], 0.99999999907324189 )
	assert_almost_equal( probs[4][2], 9.267581150732276e-10 )

	# test hmm classifications
	predicts = hmms.predict( data )

	assert_equal( predicts[0], 2 )
	assert_equal( predicts[1], 1 )
	assert_equal( predicts[2], 0 )
	assert_equal( predicts[3], 1 )
	assert_equal( predicts[4], 1 )

@with_setup( setup_all, teardown )
def test_raise_errors():
	# check if fit first ValueError is thrown
	train_error = NaiveBayes( MultivariateGaussianDistribution )
	assert_raises( ValueError, train_error.predict, [ 1, 2, 3 ] )
	assert_raises( ValueError, train_error.predict_proba, [ 1, 2, 3 ] )
	assert_raises( ValueError, train_error.predict_log_proba, [ 1, 2, 3 ] )

	# check raises no errors when converting values
	univariate.predict_log_proba( 5 )
	univariate.predict_log_proba( 4.5 )
	univariate.predict_log_proba([ 5, 6 ])
	univariate.predict_log_proba( np.array([ 5, 6 ]) )

	univariate.predict_proba( 5 )
	univariate.predict_proba( 4.5 )
	univariate.predict_proba([ 5, 6 ])
	univariate.predict_proba( np.array([ 5, 6 ]) )

	univariate.predict( 5 )
	univariate.predict( 4.5 )
	univariate.predict([ 5, 6 ])
	univariate.predict( np.array([ 5, 6 ]) )

	# check raises error when wrong dimension data is input
	assert_raises( ValueError, multivariate.predict_log_proba, 5 )
	assert_raises( ValueError, multivariate.predict_log_proba, [ 5 ] )
	assert_raises( ValueError, multivariate.predict_log_proba, [[ 1 ], [ 2 ], [ 3 ], [ 4 ]] )
	assert_raises( ValueError, multivariate.predict_log_proba, [[ 1, 2, 3 ], [ 4, 5, 6 ]] )

	assert_raises( ValueError, multivariate.predict_proba, 5 )
	assert_raises( ValueError, multivariate.predict_proba, [ 5 ] )
	assert_raises( ValueError, multivariate.predict_proba, [[ 1 ], [ 2 ], [ 3 ], [ 4 ]] )
	assert_raises( ValueError, multivariate.predict_proba, [[ 1, 2, 3 ],[ 4, 5, 6 ]] )

	assert_raises( ValueError, multivariate.predict, 5 )
	assert_raises( ValueError, multivariate.predict, [ 5 ] )
	assert_raises( ValueError, multivariate.predict, [[ 1 ], [ 2 ], [ 3 ], [ 4 ]] )
	assert_raises( ValueError, multivariate.predict, [[ 1, 2, 3 ], [ 4, 5, 6 ]] )

	# special case for hmm's

@with_setup( setup_all, teardown )
def test_pickling():
	j_univ = pickle.dumps(univariate)
	j_multi = pickle.dumps(multivariate)

	new_univ = pickle.loads( j_univ )
	assert isinstance( new_univ.models[0], NormalDistribution )
	assert isinstance( new_univ.models[1], UniformDistribution )
	numpy.testing.assert_array_equal( univariate.weights, new_univ.weights )
	assert isinstance( new_univ, NaiveBayes )

	new_multi = pickle.loads( j_multi )
	assert isinstance( new_multi.models[0], MultivariateGaussianDistribution )
	assert isinstance( new_multi.models[1], IndependentComponentsDistribution )
	numpy.testing.assert_array_equal( multivariate.weights, new_multi.weights )
	assert isinstance( new_multi, NaiveBayes )

	# hmms throw an error if the same hmm is recreated in the same space

@with_setup( setup_all, teardown )
def test_json():
	j_univ = univariate.to_json()
	j_multi = multivariate.to_json()

	new_univ = univariate.from_json( j_univ )
	assert isinstance( new_univ.models[0], NormalDistribution )
	assert isinstance( new_univ.models[1], UniformDistribution )
	numpy.testing.assert_array_equal( univariate.weights, new_univ.weights )
	assert isinstance( new_univ, NaiveBayes )

	new_multi = multivariate.from_json( j_multi )
	assert isinstance( new_multi.models[0], MultivariateGaussianDistribution )
	assert isinstance( new_multi.models[1], IndependentComponentsDistribution )
	numpy.testing.assert_array_equal( multivariate.weights, new_multi.weights )
	assert isinstance( new_multi, NaiveBayes )

	# hmms throw an error if the same hmm is recreated in the same space

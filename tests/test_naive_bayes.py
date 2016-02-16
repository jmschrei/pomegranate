from __future__ import (division)

from pomegranate import *
from nose.tools import with_setup
from nose.tools import assert_almost_equal
from nose.tools import assert_equal
from nose.tools import assert_not_equal
from nose.tools import assert_less_equal
from nose.tools import assert_raises
import random
import numpy as np

def setup_models():
	"""
	split into several methods, one for predictions, one for fitting then predictions, one for constructors
	use setup to create naive bayes models. test to see if distributions are correct
	"""
	# create univarite distributions, test values of distributions
	normal = NormalDistribution( 5, 2 )
	uniform = UniformDistribution( 0, 10 )

	# make sure univariate distributions are working
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
	
	# create multivariate distributions, test values of distributions
	multi = MultivariateGaussianDistribution( means=[ 5, 5 ], covariance=[[ 2, 0 ], [ 0, 2 ]] )
	indie = IndependentComponentsDistribution( distributions=[ UniformDistribution(0, 10), UniformDistribution(0, 10) ])

	# make sure multivariate distributions are working
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

	# create simple hmms, test values of hmms
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

	# make sure hmm's are working
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

	# to be used in each test
	global univariate
	global multivariate
	global hmms

	# no errors should be thrown
	univariate = NaiveBayes([ normal, uniform ])
	multivariate = NaiveBayes([ multi, indie ])
	hmms = NaiveBayes([ dumb, fair, smart ])

def teardown():
	pass

@with_setup( setup_models, teardown )
def test_constructors():
	"""
	Test to see if constructors work properly.
	"""
	# check for value error when supplied no information
	assert_raises( ValueError, NaiveBayes )

	# check error is not thrown
	NaiveBayes( NormalDistribution, 3 )
	NaiveBayes( MultivariateGaussianDistribution, 5 )
	NaiveBayes( HiddenMarkovModel, 2 )

	# check if error is thrown
	#NaiveBayes([ normal, multi ])
	#NaiveBayes([ indie, dumb ])
	#NaiveBayes([ fait, uniform ])

	# TODO:nick check for proper error if two distributions with mismatching inputs are used
	# TODO:nick check that predict throws error if input is mismatched and not hmms

@with_setup( setup_models, teardown )
def test_log_proba():
	"""
	Test predict log probabilities method
	"""
	# outputs correct log probability values for univariate distributions
	assert_almost_equal( univariate.predict_log_proba(np.array([ 5 ]))[0][0], -0.40634848776410526 )
	assert_almost_equal( univariate.predict_log_proba(np.array([ 5 ]))[0][1], -1.0968478669939319 )
	
	assert_almost_equal( univariate.predict_log_proba(np.array([ 3 ]))[0][0], -0.60242689998689203 )
	assert_almost_equal( univariate.predict_log_proba(np.array([ 3 ]))[0][1], -0.79292627921671865 )
	
	assert_almost_equal( univariate.predict_log_proba(np.array([ 1 ]))[0][0], -1.5484819556996032 )
	assert_almost_equal( univariate.predict_log_proba(np.array([ 1 ]))[0][1], -0.23898133492942986 )
	
	assert_almost_equal( univariate.predict_log_proba(np.array([ -1 ]))[0][0], 0.0 )
	assert_almost_equal( univariate.predict_log_proba(np.array([ -1 ]))[0][1], -float('inf') )
	

	# outputs correct log probability values for multivariate distributions
	assert_almost_equal( multivariate.predict_log_proba(np.array([[ 5, 5 ]]))[0][0], -0.11837282271439786 )
	assert_almost_equal( multivariate.predict_log_proba(np.array([[ 5, 5 ]]))[0][1], -2.1925187617325435 )
	
	assert_almost_equal( multivariate.predict_log_proba(np.array([[ 2, 9 ]]))[0][0], -4.1910993250497741 )
	assert_almost_equal( multivariate.predict_log_proba(np.array([[ 2, 9 ]]))[0][1], -0.015245264067919706 )
	
	assert_almost_equal( multivariate.predict_log_proba(np.array([[ 10, 7 ]]))[0][0], -5.1814895400206806 )
	assert_almost_equal( multivariate.predict_log_proba(np.array([[ 10, 7 ]]))[0][1], -0.0056354790388262188 )
	
	assert_almost_equal( multivariate.predict_log_proba(np.array([[ -1, 7 ]]))[0][0], 0.0 )
	assert_almost_equal( multivariate.predict_log_proba(np.array([[ -1, 7 ]]))[0][1], -float('inf') )


	# outputs correct log probability values for hmm's
	assert_almost_equal( hmms.predict_log_proba(np.array([list('H')]))[0][0], -0.89097292388986515 )
	assert_almost_equal( hmms.predict_log_proba(np.array([list('H')]))[0][1], -1.3609765531356006 )
	assert_almost_equal( hmms.predict_log_proba(np.array([list('H')]))[0][2], -1.0986122886681096 )
	
	assert_almost_equal( hmms.predict_log_proba(np.array([list('THHH')]))[0][0], -0.93570553121744293 )
	assert_almost_equal( hmms.predict_log_proba(np.array([list('THHH')]))[0][1], -1.429425687080494 )
	assert_almost_equal( hmms.predict_log_proba(np.array([list('THHH')]))[0][2], -0.9990078376167526 )
	
	assert_almost_equal( hmms.predict_log_proba(np.array([list('TTTT')]))[0][0], -3.9007882563128864 )
	assert_almost_equal( hmms.predict_log_proba(np.array([list('TTTT')]))[0][1], -0.23562532881626597 )
	assert_almost_equal( hmms.predict_log_proba(np.array([list('TTTT')]))[0][2], -1.6623251045711958 )
	
	assert_almost_equal( hmms.predict_log_proba(np.array([list('THTHTHTHTHTH')]))[0][0], -3.1703366478831185 )
	assert_almost_equal( hmms.predict_log_proba(np.array([list('THTHTHTHTHTH')]))[0][1], -0.49261403211260379 )
	assert_almost_equal( hmms.predict_log_proba(np.array([list('THTHTHTHTHTH')]))[0][2], -1.058478108940049 )
	
	assert_almost_equal( hmms.predict_log_proba(np.array([list('THTHHHHHTHTH')]))[0][0], -1.3058441172130273 )
	assert_almost_equal( hmms.predict_log_proba(np.array([list('THTHHHHHTHTH')]))[0][1], -1.4007102236822906 )
	assert_almost_equal( hmms.predict_log_proba(np.array([list('THTHHHHHTHTH')]))[0][2], -0.7284958836972919 )
	

@with_setup( setup_models, teardown )
def test_proba():
	"""
	Test predict probability method
	"""
	# outputs correct probability values for univariate distributions
	assert_almost_equal( univariate.predict_proba(np.array([ 5 ]))[0][0], 0.66607800693933361 )
	assert_almost_equal( univariate.predict_proba(np.array([ 5 ]))[0][1], 0.33392199306066628 )
	
	assert_almost_equal( univariate.predict_proba(np.array([ 3 ]))[0][0], 0.54748134004225524 )
	assert_almost_equal( univariate.predict_proba(np.array([ 3 ]))[0][1], 0.45251865995774476 )
	
	assert_almost_equal( univariate.predict_proba(np.array([ 1 ]))[0][0], 0.21257042033580209 )
	assert_almost_equal( univariate.predict_proba(np.array([ 1 ]))[0][1], 0.78742957966419791 )
	
	assert_almost_equal( univariate.predict_proba(np.array([ -1 ]))[0][0], 1.0 )
	assert_almost_equal( univariate.predict_proba(np.array([ -1 ]))[0][1], 0.0 )
	

	# outputs correct probability values for multivariate distributions
	assert_almost_equal( multivariate.predict_proba(np.array([[ 5, 5 ]]))[0][0], 0.88836478829527532 )
	assert_almost_equal( multivariate.predict_proba(np.array([[ 5, 5 ]]))[0][1], 0.11163521170472469 )
	
	assert_almost_equal( multivariate.predict_proba(np.array([[ 2, 9 ]]))[0][0], 0.015129643331582699 )
	assert_almost_equal( multivariate.predict_proba(np.array([[ 2, 9 ]]))[0][1], 0.98487035666841727 )
	
	assert_almost_equal( multivariate.predict_proba(np.array([[ 10, 7 ]]))[0][0], 0.0056196295140261846 )
	assert_almost_equal( multivariate.predict_proba(np.array([[ 10, 7 ]]))[0][1], 0.99438037048597383 )
	
	assert_almost_equal( multivariate.predict_proba(np.array([[ -1, 7 ]]))[0][0], 1.0 )
	assert_almost_equal( multivariate.predict_proba(np.array([[ -1, 7 ]]))[0][1], 0.0 )


	# outputs correct probability values for hmm's
	assert_almost_equal( hmms.predict_proba(np.array([list('H')]))[0][0], 0.41025641025641024 )
	assert_almost_equal( hmms.predict_proba(np.array([list('H')]))[0][1], 0.25641025641025639 )
	assert_almost_equal( hmms.predict_proba(np.array([list('H')]))[0][2], 0.33333333333333331 )
	
	assert_almost_equal( hmms.predict_proba(np.array([list('THHH')]))[0][0], 0.39230898163446098 )
	assert_almost_equal( hmms.predict_proba(np.array([list('THHH')]))[0][1], 0.23944639992337707 )
	assert_almost_equal( hmms.predict_proba(np.array([list('THHH')]))[0][2], 0.36824461844216183 )
	
	assert_almost_equal( hmms.predict_proba(np.array([list('TTTT')]))[0][0], 0.020225961918306088 )
	assert_almost_equal( hmms.predict_proba(np.array([list('TTTT')]))[0][1], 0.79007663743383105 )
	assert_almost_equal( hmms.predict_proba(np.array([list('TTTT')]))[0][2], 0.18969740064786292 )
	
	assert_almost_equal( hmms.predict_proba(np.array([list('THTHTHTHTHTH')]))[0][0], 0.041989459861032523 )
	assert_almost_equal( hmms.predict_proba(np.array([list('THTHTHTHTHTH')]))[0][1], 0.61102706038265642 )
	assert_almost_equal( hmms.predict_proba(np.array([list('THTHTHTHTHTH')]))[0][2], 0.346983479756311 )
	
	assert_almost_equal( hmms.predict_proba(np.array([list('THTHHHHHTHTH')]))[0][0], 0.27094373022369794 )
	assert_almost_equal( hmms.predict_proba(np.array([list('THTHHHHHTHTH')]))[0][1], 0.24642188711704707 )
	assert_almost_equal( hmms.predict_proba(np.array([list('THTHHHHHTHTH')]))[0][2], 0.48263438265925512 )

@with_setup( setup_models, teardown )
def test_prediction():
	# outputs correct probability values for univariate distributions
	assert_equal( univariate.predict(np.array([ 5 ]))[0], 0 )
	assert_equal( univariate.predict(np.array([ 3 ]))[0], 0 )
	assert_equal( univariate.predict(np.array([ 1 ]))[0], 1 )
	assert_equal( univariate.predict(np.array([ -1 ]))[0], 0 )
	
	# outputs correct probability values for multivariate distributions
	assert_equal( multivariate.predict(np.array([[ 5, 5 ]]))[0], 0 )
	assert_equal( multivariate.predict(np.array([[ 2, 9 ]]))[0], 1 )
	assert_equal( multivariate.predict(np.array([[ 10, 7 ]]))[0], 1 )
	assert_equal( multivariate.predict(np.array([[ -1, 7 ]]))[0], 0 )

	# outputs correct probability values for hmm's
	assert_equal( hmms.predict(np.array([list('H')]))[0], 0 )
	assert_equal( hmms.predict(np.array([list('THHH')]))[0], 0 )
	assert_equal( hmms.predict(np.array([list('TTTT')]))[0], 1 )
	assert_equal( hmms.predict(np.array([list('THTHTHTHTHTH')]))[0], 1 )
	assert_equal( hmms.predict(np.array([list('THTHHHHHTHTH')]))[0], 2 )
	
@with_setup( setup_models, teardown )
def test_fit():
	"""
	Test fit method by fitting NaiveBayes classifiers with data
	"""
	# fit data to univariate
	uni_X = np.array([ 5, 4, 5, 4, 6, 5, 6, 5, 4, 6, 5, 4, 0, 0, 1, 9, 8, 2, 0, 1, 1, 8, 10, 0 ])
	uni_y = np.array([ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 ])

	univariate.fit( uni_X, uni_y )

	# test univariate log probabilities
	assert_almost_equal( univariate.predict_log_proba(np.array([ 5 ]))[0][0], -0.1751742330621151 )
	assert_almost_equal( univariate.predict_log_proba(np.array([ 5 ]))[0][1], -1.8282830423560459 )
	assert_almost_equal( univariate.predict_log_proba(np.array([ 3 ]))[0][0], -1.7240463796541046 )
	assert_almost_equal( univariate.predict_log_proba(np.array([ 3 ]))[0][1], -0.19643229738177137 )
	assert_almost_equal( univariate.predict_log_proba(np.array([ 1 ]))[0][0], -11.64810474561418 )
	assert_almost_equal( univariate.predict_log_proba(np.array([ 1 ]))[0][1], -8.7356310092268075e-06 )
	assert_almost_equal( univariate.predict_log_proba(np.array([ -1 ]))[0][0], 0.0 )
	assert_almost_equal( univariate.predict_log_proba(np.array([ -1 ]))[0][1], -float('inf') )
	
	# test univariate probabilities
	assert_almost_equal( univariate.predict_proba(np.array([ 5 ]))[0][0], 0.83931077234299012 )
	assert_almost_equal( univariate.predict_proba(np.array([ 5 ]))[0][1], 0.1606892276570098 )
	assert_almost_equal( univariate.predict_proba(np.array([ 3 ]))[0][0], 0.17834304226047082 )
	assert_almost_equal( univariate.predict_proba(np.array([ 3 ]))[0][1], 0.82165695773952918 )
	assert_almost_equal( univariate.predict_proba(np.array([ 1 ]))[0][0], 8.7355928537720986e-06 )
	assert_almost_equal( univariate.predict_proba(np.array([ 1 ]))[0][1], 0.99999126440714625 )
	assert_almost_equal( univariate.predict_proba(np.array([ -1 ]))[0][0], 1.0 )
	assert_almost_equal( univariate.predict_proba(np.array([ -1 ]))[0][1], 0.0 )

	# test univariate classifications
	assert_equal( univariate.predict(np.array([ 5 ]))[0], 0 )
	assert_equal( univariate.predict(np.array([ 3 ]))[0], 1 )
	assert_equal( univariate.predict(np.array([ 1 ]))[0], 1 )
	assert_equal( univariate.predict(np.array([ -1 ]))[0], 0 )

	# fit data to multivariate
	multi_X = np.array([[ 6, 5 ], [ 3.5, 4 ], [ 4, 6 ], [ 8, 6.5 ], [ 3.5, 4 ], [ 4.5, 5.5 ],
				  [ 0, 7 ], [ 0.5, 7.5 ], [ 9.5, 8 ], [ 5, 0.5 ], [ 7.5, 1.5 ], [ 7, 7 ]])
	multi_y = np.array([ 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1 ])

	multivariate.fit( multi_X, multi_y )

	# test multivariate log probabilities
	assert_almost_equal( multivariate.predict_log_proba(np.array([[ 5, 5 ]]))[0][0], -0.09672086616254516 )
	assert_almost_equal( multivariate.predict_log_proba(np.array([[ 5, 5 ]]))[0][1], -2.3838967922368868 )
	assert_almost_equal( multivariate.predict_log_proba(np.array([[ 2, 3 ]]))[0][0], -0.884636340789835 )
	assert_almost_equal( multivariate.predict_log_proba(np.array([[ 2, 3 ]]))[0][1], -0.53249928976493077 )
	assert_almost_equal( multivariate.predict_log_proba(np.array([[ 10, 7 ]]))[0][0], 0.0 )
	assert_almost_equal( multivariate.predict_log_proba(np.array([[ 10, 7 ]]))[0][1], -float('inf') )
	assert_almost_equal( multivariate.predict_log_proba(np.array([[ -1, 7 ]]))[0][0], 0.0 )
	assert_almost_equal( multivariate.predict_log_proba(np.array([[ -1, 7 ]]))[0][1], -float('inf') )

	# test multivariate probabilities
	assert_almost_equal( multivariate.predict_proba(np.array([[ 5, 5 ]]))[0][0], 0.90780937108369053 )
	assert_almost_equal( multivariate.predict_proba(np.array([[ 5, 5 ]]))[0][1], 0.092190628916309608 )
	
	assert_almost_equal( multivariate.predict_proba(np.array([[ 2, 3 ]]))[0][0], 0.41286428788295315 )
	assert_almost_equal( multivariate.predict_proba(np.array([[ 2, 3 ]]))[0][1], 0.58713571211704685 )
	
	assert_almost_equal( multivariate.predict_proba(np.array([[ 10, 7 ]]))[0][0], 1.0 )
	assert_almost_equal( multivariate.predict_proba(np.array([[ 10, 7 ]]))[0][1], 0.0 )
	
	assert_almost_equal( multivariate.predict_proba(np.array([[ -1, 7 ]]))[0][0], 1.0 )
	assert_almost_equal( multivariate.predict_proba(np.array([[ -1, 7 ]]))[0][1], 0.0 )

	# test multivariate classifications
	assert_equal( multivariate.predict(np.array([[ 5, 5 ]]))[0], 0 )
	assert_equal( multivariate.predict(np.array([[ 2, 3 ]]))[0], 1 )
	assert_equal( multivariate.predict(np.array([[ 10, 7 ]]))[0], 0 )
	assert_equal( multivariate.predict(np.array([[ -1, 7 ]]))[0], 0 )

	# fit data to hmm's
	hmm_X = np.array([list( 'HHHHHTHTHTTTTH' ),
				  list( 'HHTHHTTHHHHHTH' ),
				  list( 'TH' ), list( 'HHHHT' ),])
	hmm_y = np.array([ 2, 2, 1, 0 ])

	hmms.fit( hmm_X, hmm_y )

	# test hmm log probabilities
	assert_almost_equal( hmms.predict_log_proba(np.array([list('H')]))[0][0], -1.2745564715121378 )
	assert_almost_equal( hmms.predict_log_proba(np.array([list('H')]))[0][1], -1.8242710481193862 )
	assert_almost_equal( hmms.predict_log_proba(np.array([list('H')]))[0][2], -0.58140929189714219 )
	assert_almost_equal( hmms.predict_log_proba(np.array([list('THHH')]))[0][0], -148.69787686172131 )
	assert_almost_equal( hmms.predict_log_proba(np.array([list('THHH')]))[0][1], 0.0 )
	assert_almost_equal( hmms.predict_log_proba(np.array([list('THHH')]))[0][2], -148.90503555416012 )
	assert_almost_equal( hmms.predict_log_proba(np.array([list('HHHH')]))[0][0], -0.65431299454112646 )
	assert_almost_equal( hmms.predict_log_proba(np.array([list('HHHH')]))[0][1], -2.8531712971903209 )
	assert_almost_equal( hmms.predict_log_proba(np.array([list('HHHH')]))[0][2], -0.86147167008351733 )
	assert_almost_equal( hmms.predict_log_proba(np.array([list('THTHTHTHTHTH')]))[0][0], -898.78383614856 )
	assert_almost_equal( hmms.predict_log_proba(np.array([list('THTHTHTHTHTH')]))[0][1], 0.0 )
	assert_almost_equal( hmms.predict_log_proba(np.array([list('THTHTHTHTHTH')]))[0][2], -149.77063003020317 )
	assert_almost_equal( hmms.predict_log_proba(np.array([list('THTHHHHHTHTH')]))[0][0], -596.9903657729626 )
	assert_almost_equal( hmms.predict_log_proba(np.array([list('THTHHHHHTHTH')]))[0][1], 0.0 )
	assert_almost_equal( hmms.predict_log_proba(np.array([list('THTHHHHHTHTH')]))[0][2], -149.77062996265178 )

	# test hmm probabilities
	assert_almost_equal( hmms.predict_proba(np.array([list('H')]))[0][0], 0.27955493104058637 )
	assert_almost_equal( hmms.predict_proba(np.array([list('H')]))[0][1], 0.16133520687824079 )
	assert_almost_equal( hmms.predict_proba(np.array([list('H')]))[0][2], 0.55910986208117275 )
	assert_almost_equal( hmms.predict_proba(np.array([list('THHH')]))[0][0], 2.709300362358663e-09 )
	assert_almost_equal( hmms.predict_proba(np.array([list('THHH')]))[0][1], 0.99999999508833481 )
	assert_almost_equal( hmms.predict_proba(np.array([list('THHH')]))[0][2], 2.2023649431371957e-09 )
	assert_almost_equal( hmms.predict_proba(np.array([list('HHHH')]))[0][0], 0.51979904472991378 )
	assert_almost_equal( hmms.predict_proba(np.array([list('HHHH')]))[0][1], 0.057661169909141198 )
	assert_almost_equal( hmms.predict_proba(np.array([list('HHHH')]))[0][2], 0.422539785360945 )
	assert_almost_equal( hmms.predict_proba(np.array([list('THTHTHTHTHTH')]))[0][0], 5.3986709867980443e-55 )
	assert_almost_equal( hmms.predict_proba(np.array([list('THTHTHTHTHTH')]))[0][1], 0.999999999073242 )
	assert_almost_equal( hmms.predict_proba(np.array([list('THTHTHTHTHTH')]))[0][2], 9.2675809768728048e-10 )
	assert_almost_equal( hmms.predict_proba(np.array([list('THTHHHHHTHTH')]))[0][0], 5.9769084150373497e-36 )
	assert_almost_equal( hmms.predict_proba(np.array([list('THTHHHHHTHTH')]))[0][1], 0.99999999907324189 )
	assert_almost_equal( hmms.predict_proba(np.array([list('THTHHHHHTHTH')]))[0][2], 9.267581150732276e-10 )

	# test hmm classifications
	assert_equal( hmms.predict(np.array([list('H')]))[0], 2 )
	assert_equal( hmms.predict(np.array([list('THHH')]))[0], 1 )
	assert_equal( hmms.predict(np.array([list('HHHH')]))[0], 0 )
	assert_equal( hmms.predict(np.array([list('THTHTHTHTHTH')]))[0], 1 )
	assert_equal( hmms.predict(np.array([list('THTHHHHHTHTH')]))[0], 1 )

@with_setup( setup_models, teardown )
def test_raise_errors():
	"""
	Tests edge cases for naive bayes that should or should not raise errors
	"""
	# run on all other cases that should raise errors in combination
	# example, constructing with objects with no parameters,
	#	then running predict methods before fit
	pass
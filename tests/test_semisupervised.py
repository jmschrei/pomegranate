# test_bayes_classifier.py
# Contact: Jacob Schreiber <jmschreiber91@gmail.com>

import os
import numpy
import torch
import pytest

from pomegranate.gmm import GeneralMixtureModel
from pomegranate.hmm import DenseHMM
from pomegranate.distributions import Exponential

from .tools import assert_raises
from numpy.testing import assert_array_equal
from numpy.testing import assert_array_almost_equal


inf = float("inf")


@pytest.fixture
def X():
	return [[1, 2, 0],
	     [0, 0, 1],
	     [1, 1, 2],
	     [2, 2, 2],
	     [3, 1, 0],
	     [5, 1, 4],
	     [2, 1, 0],
	     [1, 0, 2],
	     [1, 1, 0],
	     [0, 2, 1],
	     [0, 0, 0]]


@pytest.fixture
def priors():
	return torch.tensor(numpy.array([
		[0.5, 0.5],
	    [0.5, 0.5],
	    [0.5, 0.5],
	    [0.0, 1.0],
	    [0.5, 0.5],
	    [0.3, 0.7],
	    [0.6, 0.4],
	    [0.5, 0.5],
	    [0.0, 1.0],
	    [1.0, 0.0],
	    [0.5, 0.5]
	]))


@pytest.fixture
def gmm():
	d = [Exponential([2.1, 0.3, 0.1]), Exponential([1.5, 3.1, 2.2])]
	return GeneralMixtureModel(d, priors=[0.7, 0.3])


@pytest.fixture
def hmm():
	d = [Exponential([2.1, 0.3, 0.1]), Exponential([1.5, 3.1, 2.2])]
	return DenseHMM(d, edges=[[0.8, 0.2], [0.4, 0.6]], starts=[0.4, 0.6],
		ends=[0.5, 0.5])


###


def _test_gmm_raises(func, X, priors):
	X = numpy.array(X, dtype='float32')
	priors = numpy.array(priors, dtype='float32')

	assert_raises(ValueError, func, X, priors+1)
	assert_raises(ValueError, func, X, priors-1)
	assert_raises(ValueError, func, X, priors/2.0)
	assert_raises(ValueError, func, X, priors[:5])
	assert_raises(ValueError, func, X[:5], priors)
	assert_raises(ValueError, func, X, priors[:,0])
	assert_raises(ValueError, func, X, priors[:, :1])


def test_gmm_emission_matrix(gmm, X, priors):
	y_hat = gmm._emission_matrix(X)
	assert_array_almost_equal(y_hat, [
		[ -4.7349,  -4.8411],
        [ -7.5921,  -3.9838],
        [-21.4016,  -5.4276],
        [-25.2111,  -6.4169],
        [ -2.3540,  -5.8519],
        [-43.3063,  -9.0034],
        [ -1.8778,  -5.1852],
        [-18.0682,  -5.1051],
        [ -1.4016,  -4.5185],
        [-14.2587,  -4.6290],
        [  2.4079,  -3.5293]], 4)

	y_hat = gmm._emission_matrix(X, priors=priors)
	assert_array_almost_equal(y_hat, [
		[ -5.4281,  -5.5343],
        [ -8.2852,  -4.6770],
        [-22.0947,  -6.1208],
        [    -inf,  -6.4169],
        [ -3.0471,  -6.5450],
        [-44.5103,  -9.3601],
        [ -2.3886,  -6.1015],
        [-18.7614,  -5.7982],
        [    -inf,  -4.5185],
        [-14.2587,     -inf],
        [  1.7148,  -4.2224]], 4)


def test_gmm_emission_matrix_raises(gmm, X, priors):
	_test_gmm_raises(gmm._emission_matrix, X, priors)


def test_gmm_probability(gmm, X, priors):
	y_hat = gmm.probability(X)
	assert_array_almost_equal(y_hat, numpy.exp([-4.0935, -3.9571, -5.4276, 
		-6.4169, -2.3241, -9.0034, -1.8418, -5.1051, -1.3582, -4.6289,  
		2.4106]), 3)

	y_hat = gmm.probability(X, priors=priors)
	assert_array_almost_equal(y_hat, [8.3407e-03, 9.5592e-03, 2.1967e-03, 
		1.6337e-03, 4.8933e-02, 8.6094e-05, 9.3998e-02, 3.0330e-03, 
		1.0905e-02, 6.4197e-07, 5.5702e+00], 3)	


def test_gmm_probability_raises(gmm, X, priors):
	_test_gmm_raises(gmm.probability, X, priors)


def test_gmm_log_probability(gmm, X, priors):
	y_hat = gmm.log_probability(X)
	assert_array_almost_equal(y_hat, [-4.0935, -3.9571, -5.4276, -6.4169, 
		-2.3241, -9.0034, -1.8418, -5.1051, -1.3582, -4.6289,  2.4106], 4)

	y_hat = gmm.log_probability(X, priors=priors)
	assert_array_almost_equal(y_hat, [-4.7866, -4.6503, -6.1208, -6.4169,
		-3.0173, -9.3601, -2.3645, -5.7982, -4.5185, -14.2587, 1.7174], 4)	


def test_gmm_log_probability_raises(gmm, X, priors):
	_test_gmm_raises(gmm.log_probability, X, priors)


def test_gmm_predict(gmm, X, priors):
	y_hat = gmm.predict(X)
	assert_array_equal(y_hat, [0, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0])

	y_hat = gmm.predict(X, priors=priors)
	assert_array_equal(y_hat, [0, 1, 1, 1, 0, 1, 0, 1, 1, 0, 0])	


def test_gmm_predict_raises(gmm, X, priors):
	_test_gmm_raises(gmm.predict, X, priors)


def test_gmm_predict_proba(gmm, X, priors):
	y_hat = gmm.predict_proba(X)
	assert_array_almost_equal(y_hat, [
		[5.2653e-01, 4.7347e-01],
        [2.6385e-02, 9.7361e-01],
        [1.1551e-07, 1.0000e+00],
        [6.8830e-09, 1.0000e+00],
        [9.7063e-01, 2.9372e-02],
        [1.2660e-15, 1.0000e+00],
        [9.6468e-01, 3.5317e-02],
        [2.3451e-06, 1.0000e+00],
        [9.5759e-01, 4.2413e-02],
        [6.5741e-05, 9.9993e-01],
        [9.9737e-01, 2.6323e-03]], 4)

	y_hat = gmm.predict_proba(X, priors=priors)
	assert_array_almost_equal(y_hat, [
		[5.2653e-01, 4.7347e-01],
        [2.6385e-02, 9.7361e-01],
        [1.1551e-07, 1.0000e+00],
        [0.0000e+00, 1.0000e+00],
        [9.7063e-01, 2.9372e-02],
        [5.4256e-16, 1.0000e+00],
        [9.7618e-01, 2.3825e-02],
        [2.3451e-06, 1.0000e+00],
        [0.0000e+00, 1.0000e+00],
        [1.0000e+00, 0.0000e+00],
        [9.9737e-01, 2.6323e-03]], 4)	


def test_gmm_predict_proba_raises(gmm, X, priors):
	_test_gmm_raises(gmm.predict_proba, X, priors)


def test_gmm_predict_log_proba(gmm, X, priors):
	y_hat = gmm.predict_log_proba(X)
	assert_array_almost_equal(y_hat, [
		[-6.4145e-01, -7.4766e-01],
        [-3.6350e+00, -2.6740e-02],
        [-1.5974e+01,  0.0000e+00],
        [-1.8794e+01,  0.0000e+00],
        [-2.9812e-02, -3.5277e+00],
        [-3.4303e+01,  0.0000e+00],
        [-3.5955e-02, -3.3434e+00],
        [-1.2963e+01, -2.3842e-06],
        [-4.3338e-02, -3.1603e+00],
        [-9.6298e+00, -6.5804e-05],
        [-2.6357e-03, -5.9399e+00]], 3)

	y_hat = gmm.predict_log_proba(X, priors=priors)
	assert_array_almost_equal(y_hat, [
		[-6.4145e-01, -7.4766e-01],
        [-3.6350e+00, -2.6740e-02],
        [-1.5974e+01,  0.0000e+00],
        [       -inf,  0.0000e+00],
        [-2.9812e-02, -3.5277e+00],
        [-3.5150e+01,  0.0000e+00],
        [-2.4113e-02, -3.7370e+00],
        [-1.2963e+01, -2.3842e-06],
        [       -inf,  0.0000e+00],
        [ 0.0000e+00,        -inf],
        [-2.6358e-03, -5.9399e+00]], 3)	


def test_gmm_predict_log_proba_raises(gmm, X, priors):
	_test_gmm_raises(gmm.predict_log_proba, X, priors)


def test_gmm_summarize(gmm, X, priors):
	gmm.summarize(X, priors=priors)

	assert_array_almost_equal(gmm._w_sum, [4.497087, 6.502913], 4)

	assert_array_almost_equal(gmm.distributions[0]._w_sum, [4.497087, 4.497087, 
		4.497087], 4)
	assert_array_almost_equal(gmm.distributions[0]._xw_sum, [5.390766, 4.999861,
		1.02639], 4)

	assert_array_almost_equal(gmm.distributions[1]._w_sum, [6.502912, 6.502912, 
		6.502912], 4)
	assert_array_almost_equal(gmm.distributions[1]._xw_sum, [10.609234, 
		6.000139, 10.97361], 4)


def test_gmm_summarize_raises(gmm, X, priors):
	X = numpy.array(X, dtype='float32')
	priors = numpy.array(priors, dtype='float32')
	func = gmm.summarize

	assert_raises(ValueError, func, X, None, priors+1)
	assert_raises(ValueError, func, X, None, priors-1)
	assert_raises(ValueError, func, X, None, priors/2.0)
	assert_raises(ValueError, func, X, None, priors[:5])
	assert_raises(ValueError, func, X[:5], None, priors)
	assert_raises(ValueError, func, X, None, priors[:,0])
	assert_raises(ValueError, func, X, None, priors[:, :1])


def test_gmm_fit(X, priors):
	X = numpy.array(X) + 1

	d1 = Exponential([2.1, 0.3, 0.1])
	d2 = Exponential([1.5, 3.1, 2.2])
	gmm = GeneralMixtureModel([d1, d2], priors=[0.7, 0.3])
	gmm.fit(X)

	assert_array_almost_equal(gmm._log_priors, [-8.9908e+00, -1.2458e-04], 4)	
	assert_array_almost_equal(d1.scales, [1.7997, 1.7256, 1.5104], 4)
	assert_array_almost_equal(d2.scales, [2.4546, 2.    , 2.091], 4)


	d1 = Exponential([2.1, 0.3, 0.1])
	d2 = Exponential([1.5, 3.1, 2.2])
	gmm = GeneralMixtureModel([d1, d2], priors=[0.7, 0.3])
	gmm.fit(X, priors=priors)

	assert_array_almost_equal(gmm._log_priors, [-1.607879, -0.223534], 4)	
	assert_array_almost_equal(d1.scales, [1.5753, 2.2913, 1.8507], 4)
	assert_array_almost_equal(d2.scales, [2.6748, 1.927 , 2.1511], 4)


def test_gmm_fit_raises(gmm, X, priors):
	X = numpy.array(X, dtype='float32')
	priors = numpy.array(priors, dtype='float32')
	func = gmm.fit

	assert_raises(ValueError, func, X, None, priors+1)
	assert_raises(ValueError, func, X, None, priors-1)
	assert_raises(ValueError, func, X, None, priors/2.0)
	assert_raises(ValueError, func, X, None, priors[:5])
	assert_raises(ValueError, func, X[:5], None, priors)
	assert_raises(ValueError, func, X, None, priors[:,0])
	assert_raises(ValueError, func, X, None, priors[:, :1])



#


def _test_hmm_raises(func, X, priors):
	X = numpy.array(X, dtype='float32')
	priors = numpy.array(priors, dtype='float32')[None, :, :]

	assert_raises(ValueError, func, X, None, priors+1)
	assert_raises(ValueError, func, X, None, priors-1)
	assert_raises(ValueError, func, X, None, priors/2.0)
	assert_raises(ValueError, func, X, None, priors[:, :5])
	assert_raises(ValueError, func, X[:, :5], None, priors)
	assert_raises(ValueError, func, X, None, priors[:, :, 0])
	assert_raises(ValueError, func, X, None, priors[:, :, :1])


def test_hmm_emission_matrix(hmm, X, priors):
	X = numpy.array([X])
	priors = priors.unsqueeze(0)

	y_hat = hmm._emission_matrix(X)
	assert_array_almost_equal(y_hat, [
		[[ -4.3782,  -3.6372],
         [ -7.2354,  -2.7799],
         [-21.0449,  -4.2237],
         [-24.8544,  -5.2129],
         [ -1.9973,  -4.6479],
         [-42.9497,  -7.7994],
         [ -1.5211,  -3.9812],
         [-17.7116,  -3.9011],
         [ -1.0449,  -3.3146],
         [-13.9020,  -3.4250],
         [  2.7646,  -2.3253]]], 4)

	y_hat = hmm._emission_matrix(X, priors=priors)
	assert_array_almost_equal(y_hat, [
		[[ -5.0714,  -4.3303],
         [ -7.9285,  -3.4730],
         [-21.7381,  -4.9168],
         [    -inf,  -5.2129],
         [ -2.6904,  -5.3411],
         [-44.1536,  -8.1561],
         [ -2.0319,  -4.8975],
         [-18.4047,  -4.5942],
         [    -inf,  -3.3146],
         [-13.9020,     -inf],
         [  2.0715,  -3.0185]]], 4)


def test_hmm_emission_matrix_raises(hmm, X, priors):
	X = numpy.array(X, dtype='float32')
	priors = numpy.array(priors, dtype='float32')[None, :, :]
	func = hmm._emission_matrix

	assert_raises(ValueError, func, X, priors+1)
	assert_raises(ValueError, func, X, priors-1)
	assert_raises(ValueError, func, X, priors/2.0)
	assert_raises(ValueError, func, X, priors[:, :5])
	assert_raises(ValueError, func, X[:, :5], priors)
	assert_raises(ValueError, func, X, priors[:, :, 0])
	assert_raises(ValueError, func, X, priors[:, :, :1])


def test_hmm_forward(hmm, X, priors):
	X = numpy.array([X])
	priors = priors.unsqueeze(0)

	y_hat = hmm.forward(X)
	assert_array_almost_equal(y_hat, [
		[[ -5.2945,  -4.1480],
         [-11.8077,  -7.3380],
         [-29.2766, -12.0687],
         [-37.8394, -17.7924],
         [-20.7060, -22.9511],
         [-63.8272, -29.8389],
         [-32.2763, -34.3310],
         [-50.1489, -37.4616],
         [-39.4228, -41.2870],
         [-53.4733, -44.0753],
         [-42.2268, -46.9115]]], 4)

	y_hat = hmm.forward(X, priors=priors)
	assert_array_almost_equal(y_hat, [
		[[ -5.9877,  -4.8411],
         [-13.1940,  -8.7243],
         [-31.3560, -14.1481],
         [    -inf, -19.8719],
         [-23.4786, -25.7237],
         [-67.8038, -32.9682],
         [-35.9164, -38.3766],
         [-54.5024, -41.8919],
         [    -inf, -45.7173],
         [-60.5357,     -inf],
         [-58.6873, -65.1636]]], 4)


def test_hmm_forward_raises(hmm, X, priors):
	_test_hmm_raises(hmm.forward, X, priors)


def test_hmm_backward(hmm, X, priors):
	X = numpy.array([X])
	priors = priors.unsqueeze(0)

	y_hat = hmm.backward(X)
	assert_array_almost_equal(y_hat, [
		[[-39.9505, -38.8647],
         [-36.6752, -35.5766],
         [-31.9407, -30.8421],
         [-24.6495, -25.1184],
         [-22.4807, -21.3821],
         [-12.6419, -13.0719],
         [-10.9597,  -9.8611],
         [ -5.0633,  -5.4492],
         [ -3.8699,  -2.7714],
         [  1.8499,   1.1644],
         [ -0.6931,  -0.6931]]], 4)

	y_hat = hmm.backward(X, priors=priors)
	assert_array_almost_equal(y_hat, [
		[[-55.7255, -54.6397],
         [-51.7571, -50.6585],
         [-46.3294, -45.2308],
         [-39.0382, -39.5071],
         [-36.1763, -35.0777],
         [-25.9039, -26.4107],
         [-23.6907, -22.5921],
         [-18.5856, -17.4870],
         [-12.9685, -13.6616],
         [  1.1567,   0.4712],
         [ -0.6931,  -0.6931]]], 4)


def test_hmm_backward_raises(hmm, X, priors):
	_test_hmm_raises(hmm.backward, X, priors)


def test_hmm_forward_backward(hmm, X, priors):
	X = numpy.array([X])
	priors = priors.unsqueeze(0)

	t, r, starts, ends, logp = hmm.forward_backward(X)
	assert_array_almost_equal(t, [[[1.8028e-03, 2.2629e+00],
         [3.1568e+00, 4.5785e+00]]], 4)

	assert_array_almost_equal(r, [
		[[-2.3343e+00, -1.0190e-01],
         [-5.5721e+00, -3.8109e-03],
         [-1.8306e+01,  0.0000e+00],
         [-1.9578e+01,  0.0000e+00],
         [-2.7591e-01, -1.4225e+00],
         [-3.3558e+01,  0.0000e+00],
         [-3.2527e-01, -1.2813e+00],
         [-1.2301e+01, -3.8147e-06],
         [-3.8184e-01, -1.1476e+00],
         [-8.7126e+00, -1.6403e-04],
         [-9.1934e-03, -4.6938e+00]]], 3)

	assert_array_almost_equal(starts, [[0.0969, 0.9031]], 4)
	assert_array_almost_equal(ends, [[0.9908, 0.0092]], 4)
	assert_array_almost_equal(logp, [-42.9108], 4)

	t, r, starts, ends, logp = hmm.forward_backward(X, priors=priors)
	assert_array_almost_equal(t, [[[0.9999, 1.6556], [2.5572, 4.7872]]], 4)

	assert_array_almost_equal(r, [
		[[-2.3343e+00, -1.0190e-01],
         [-5.5721e+00, -3.8109e-03],
         [-1.8306e+01,  0.0000e+00],
         [       -inf,  0.0000e+00],
         [-2.7591e-01, -1.4225e+00],
         [-3.4329e+01,  0.0000e+00],
         [-2.2815e-01, -1.5897e+00],
         [-1.3709e+01,  0.0000e+00],
         [       -inf,  0.0000e+00],
         [ 0.0000e+00,        -inf],
         [-1.5373e-03, -6.4778e+00]]], 3)

	assert_array_almost_equal(starts, [[0.0969, 0.9031]], 4)
	assert_array_almost_equal(ends, [[0.9985, 0.0015]], 4)
	assert_array_almost_equal(logp, [-59.3789], 4)


def test_hmm_forward_backward_raises(hmm, X, priors):
	_test_hmm_raises(hmm.forward_backward, X, priors)


def test_hmm_probability(hmm, X, priors):
	X = numpy.array([X])
	priors = priors.unsqueeze(0)

	y_hat = hmm.probability(X)
	assert_array_almost_equal(y_hat, [2.3125e-19], 4)

	y_hat = hmm.probability(X, priors=priors)
	assert_array_almost_equal(y_hat, [1.6295e-26], 4)


def test_hmm_probability_raises(hmm, X, priors):
	X = numpy.array(X, dtype='float32')
	priors = numpy.array(priors, dtype='float32')[None, :, :]
	func = hmm.probability

	assert_raises(ValueError, func, X, priors+1)
	assert_raises(ValueError, func, X, priors-1)
	assert_raises(ValueError, func, X, priors/2.0)
	assert_raises(ValueError, func, X, priors[:, :5])
	assert_raises(ValueError, func, X[:, :5], priors)
	assert_raises(ValueError, func, X, priors[:, :, 0])
	assert_raises(ValueError, func, X, priors[:, :, :1])


def test_hmm_log_probability(hmm, X, priors):
	X = numpy.array([X])
	priors = priors.unsqueeze(0)

	y_hat = hmm.log_probability(X)
	assert_array_almost_equal(y_hat, [-42.9108], 4)

	y_hat = hmm.log_probability(X, priors=priors)
	assert_array_almost_equal(y_hat, [-59.3789], 4)


def test_hmm_log_probability_raises(hmm, X, priors):
	X = numpy.array(X, dtype='float32')
	priors = numpy.array(priors, dtype='float32')[None, :, :]
	func = hmm.log_probability

	assert_raises(ValueError, func, X, priors+1)
	assert_raises(ValueError, func, X, priors-1)
	assert_raises(ValueError, func, X, priors/2.0)
	assert_raises(ValueError, func, X, priors[:, :5])
	assert_raises(ValueError, func, X[:, :5], priors)
	assert_raises(ValueError, func, X, priors[:, :, 0])
	assert_raises(ValueError, func, X, priors[:, :, :1])


def test_hmm_predict(hmm, X, priors):
	X = numpy.array([X])
	priors = priors.unsqueeze(0)

	y_hat = hmm.predict(X)
	assert_array_almost_equal(y_hat, [[1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0]], 4)

	y_hat = hmm.predict(X, priors=priors)
	assert_array_almost_equal(y_hat, [[1, 1, 1, 1, 0, 1, 0, 1, 1, 0, 0]], 4)


def test_hmm_predict_raises(hmm, X, priors):
	X = numpy.array(X, dtype='float32')
	priors = numpy.array(priors, dtype='float32')[None, :, :]
	func = hmm.predict

	assert_raises(ValueError, func, X, priors+1)
	assert_raises(ValueError, func, X, priors-1)
	assert_raises(ValueError, func, X, priors/2.0)
	assert_raises(ValueError, func, X, priors[:, :5])
	assert_raises(ValueError, func, X[:, :5], priors)
	assert_raises(ValueError, func, X, priors[:, :, 0])
	assert_raises(ValueError, func, X, priors[:, :, :1])


def test_hmm_predict_proba(hmm, X, priors):
	X = numpy.array([X])
	priors = priors.unsqueeze(0)

	y_hat = hmm.predict_proba(X)
	assert_array_almost_equal(y_hat, [
		[[9.6881e-02, 9.0312e-01],
         [3.8024e-03, 9.9620e-01],
         [1.1210e-08, 1.0000e+00],
         [3.1428e-09, 1.0000e+00],
         [7.5888e-01, 2.4112e-01],
         [2.6658e-15, 1.0000e+00],
         [7.2233e-01, 2.7767e-01],
         [4.5453e-06, 1.0000e+00],
         [6.8261e-01, 3.1739e-01],
         [1.6449e-04, 9.9984e-01],
         [9.9085e-01, 9.1517e-03]]], 4)

	y_hat = hmm.predict_proba(X, priors=priors)
	assert_array_almost_equal(y_hat, [
		[[9.6881e-02, 9.0312e-01],
         [3.8023e-03, 9.9620e-01],
         [1.1210e-08, 1.0000e+00],
         [0.0000e+00, 1.0000e+00],
         [7.5888e-01, 2.4112e-01],
         [1.2337e-15, 1.0000e+00],
         [7.9601e-01, 2.0399e-01],
         [1.1122e-06, 1.0000e+00],
         [0.0000e+00, 1.0000e+00],
         [1.0000e+00, 0.0000e+00],
         [9.9846e-01, 1.5372e-03]]], 4)


def test_hmm_predict_proba_raises(hmm, X, priors):
	X = numpy.array(X, dtype='float32')
	priors = numpy.array(priors, dtype='float32')[None, :, :]
	func = hmm.predict_proba

	assert_raises(ValueError, func, X, priors+1)
	assert_raises(ValueError, func, X, priors-1)
	assert_raises(ValueError, func, X, priors/2.0)
	assert_raises(ValueError, func, X, priors[:, :5])
	assert_raises(ValueError, func, X[:, :5], priors)
	assert_raises(ValueError, func, X, priors[:, :, 0])
	assert_raises(ValueError, func, X, priors[:, :, :1])


def test_hmm_predict_log_proba(hmm, X, priors):
	X = numpy.array([X])
	priors = priors.unsqueeze(0)

	y_hat = hmm.predict_log_proba(X)
	assert_array_almost_equal(y_hat, [
		[[-2.3343e+00, -1.0190e-01],
         [-5.5721e+00, -3.8109e-03],
         [-1.8306e+01,  0.0000e+00],
         [-1.9578e+01,  0.0000e+00],
         [-2.7591e-01, -1.4225e+00],
         [-3.3558e+01,  0.0000e+00],
         [-3.2527e-01, -1.2813e+00],
         [-1.2301e+01, -3.8147e-06],
         [-3.8184e-01, -1.1476e+00],
         [-8.7126e+00, -1.6403e-04],
         [-9.1934e-03, -4.6938e+00]]], 3)

	y_hat = hmm.predict_log_proba(X, priors=priors)
	assert_array_almost_equal(y_hat, [
		[[-2.3343e+00, -1.0190e-01],
         [-5.5721e+00, -3.8109e-03],
         [-1.8306e+01,  0.0000e+00],
         [       -inf,  0.0000e+00],
         [-2.7591e-01, -1.4225e+00],
         [-3.4329e+01,  0.0000e+00],
         [-2.2815e-01, -1.5897e+00],
         [-1.3709e+01,  0.0000e+00],
         [       -inf,  0.0000e+00],
         [ 0.0000e+00,        -inf],
         [-1.5373e-03, -6.4778e+00]]], 3)


def test_hmm_predict_log_proba_raises(hmm, X, priors):
	X = numpy.array(X, dtype='float32')
	priors = numpy.array(priors, dtype='float32')[None, :, :]
	func = hmm.predict_log_proba

	assert_raises(ValueError, func, X, priors+1)
	assert_raises(ValueError, func, X, priors-1)
	assert_raises(ValueError, func, X, priors/2.0)
	assert_raises(ValueError, func, X, priors[:, :5])
	assert_raises(ValueError, func, X[:, :5], priors)
	assert_raises(ValueError, func, X, priors[:, :, 0])
	assert_raises(ValueError, func, X, priors[:, :, :1])


def test_hmm_summarize(hmm, X, priors):
	X = numpy.array([X])
	priors = priors.unsqueeze(0)

	y_hat = hmm.summarize(X)
	assert_array_almost_equal(hmm._xw_starts_sum, [0.0969, 0.9031], 4)
	assert_array_almost_equal(hmm._xw_ends_sum, [0.9908, 0.0092], 4)
	assert_array_almost_equal(hmm._xw_sum, [[1.8028e-03, 2.2629e+00],
		[3.1568e+00, 4.5785e+00]], 4)


	y_hat = hmm.summarize(X, priors=priors)
	assert_array_almost_equal(hmm._xw_starts_sum, [0.1938, 1.8062], 4)
	assert_array_almost_equal(hmm._xw_ends_sum, [1.9893, 0.0107], 4)
	assert_array_almost_equal(hmm._xw_sum, [[1.0017, 3.9185],
		[5.714 , 9.3657]], 4)


def test_hmm_summarize_raises(hmm, X, priors):
	_test_hmm_raises(hmm, X, priors)


def test_hmm_fit_raises(hmm, X, priors):
	_test_hmm_raises(hmm, X, priors)

# test_bayes_classifier.py
# Contact: Jacob Schreiber <jmschreiber91@gmail.com>

import os
import numpy
import torch
import pytest

from pomegranate.bayes_classifier import BayesClassifier
from pomegranate.distributions import Exponential

from .distributions._utils import _test_initialization_raises_one_parameter
from .distributions._utils import _test_initialization
from .distributions._utils import _test_predictions
from .distributions._utils import _test_efd_from_summaries
from .distributions._utils import _test_raises

from .tools import assert_raises
from numpy.testing import assert_array_almost_equal


MIN_VALUE = 0
MAX_VALUE = None
VALID_VALUE = 1.2


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
def X_masked(X):
	mask = torch.tensor(numpy.array([
		[False, True,  True ],
		[True,  True,  False],
		[False, False, False],
		[True,  True,  True ],
		[False, True,  False],
		[True,  True,  True ],
		[False, False, False],
		[True,  False, True ],
		[True,  True,  True ],
		[True,  True,  True ],
		[True,  False, True ]]))

	X = torch.tensor(numpy.array(X))
	return torch.masked.MaskedTensor(X, mask=mask)


@pytest.fixture
def w():
	return [[1], [2], [0], [0], [5], [1], [2], [1], [1], [2], [0]]


@pytest.fixture
def y():
	return [1, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1]


@pytest.fixture
def model():
	d = [Exponential([2.1, 0.3, 0.1]), Exponential([1.5, 3.1, 2.2])]
	return BayesClassifier(d, priors=[0.7, 0.3])


###


def test_initialization():
	d = [Exponential(), Exponential()]
	model = BayesClassifier(d)

	_test_initialization(model, None, "priors", 0.0, False, None)
	assert_raises(AttributeError, getattr, model, "_w_sum")
	assert_raises(AttributeError, getattr, model, "_log_priors")


def test_initialization_raises():
	d = [Exponential(), Exponential()]

	assert_raises(TypeError, BayesClassifier)
	assert_raises(ValueError, BayesClassifier, d, [0.2, 0.2, 0.6])
	assert_raises(ValueError, BayesClassifier, d, [0.2, 1.0])
	assert_raises(ValueError, BayesClassifier, d, [-0.2, 1.2])

	assert_raises(ValueError, BayesClassifier, Exponential)
	assert_raises(ValueError, BayesClassifier, d, inertia=-0.4)
	assert_raises(ValueError, BayesClassifier, d, inertia=1.2)
	assert_raises(ValueError, BayesClassifier, d, inertia=1.2, frozen="true")
	assert_raises(ValueError, BayesClassifier, d, inertia=1.2, frozen=3)
	

def test_reset_cache(X, y):
	d = [Exponential(), Exponential()]
	
	model = BayesClassifier(d)
	model.summarize(X, y)
	
	assert_array_almost_equal(model._w_sum, [6.0, 5.0])
	assert_array_almost_equal(model._log_priors, [-0.693147, -0.693147])

	model._reset_cache()
	assert_array_almost_equal(model._w_sum, [0.0, 0.0])
	assert_array_almost_equal(model._log_priors, [-0.693147, -0.693147])	


def test_initialize(X):
	d = [Exponential(), Exponential()]
	model = BayesClassifier(d)

	assert model.d is None
	assert model.k == 2
	assert model._initialized == False
	assert_raises(AttributeError, getattr, model, "_w_sum")
	assert_raises(AttributeError, getattr, model, "_log_priors")

	model._initialize(3)
	assert model._initialized == True
	assert model.priors.shape[0] == 2
	assert model.d == 3
	assert model.k == 2
	assert_array_almost_equal(model.priors, [0.5, 0.5])
	assert_array_almost_equal(model._w_sum, [0.0, 0.0])

	model._initialize(2)
	assert model._initialized == True
	assert model.priors.shape[0] == 2
	assert model.d == 2
	assert model.k == 2
	assert_array_almost_equal(model.priors, [0.5, 0.5])
	assert_array_almost_equal(model._w_sum, [0.0, 0.0])

	d = [Exponential([0.4, 2.1]), Exponential([3, 1]), Exponential([0.2, 1])]
	model = BayesClassifier(d)
	assert model._initialized == True
	assert model.d == 2
	assert model.k == 3

	model._initialize(3)
	assert model._initialized == True
	assert model.priors.shape[0] == 3
	assert model.d == 3
	assert model.k == 3
	assert_array_almost_equal(model.priors, [1./3, 1./3, 1./3])
	assert_array_almost_equal(model._w_sum, [0.0, 0.0, 0.0])



###


def test_emission_matrix(model, X):
	e = model._emission_matrix(X)

	assert_array_almost_equal(e, 
		[[ -4.7349,  -4.8411],
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
	assert_array_almost_equal(e[:, 0], model.distributions[0].log_probability(X) 
		- 0.3567, 4)
	assert_array_almost_equal(e[:, 1], model.distributions[1].log_probability(X) 
		- 1.2040, 4)


def test_emission_matrix_raises(model, X):
	_test_raises(model, "_emission_matrix", X, min_value=MIN_VALUE)

	d = [Exponential(), Exponential()]
	model = BayesClassifier(d)
	_test_raises(model, "_emission_matrix", X, min_value=MIN_VALUE)


def test_log_probability(model, X):
	logp = model.log_probability(X)
	assert_array_almost_equal(logp, [-4.0935, -3.9571, -5.4276, -6.4169, 
		-2.3241, -9.0034, -1.8418, -5.1051, -1.3582, -4.6289,  2.4106], 4)


def test_log_probability_raises(model, X):
	_test_raises(model, "log_probability", X, min_value=MIN_VALUE)

	d = [Exponential(), Exponential()]
	model = BayesClassifier(d)
	_test_raises(model, "log_probability", X, min_value=MIN_VALUE)


def test_predict(model, X):
	y_hat = model.predict(X)
	assert_array_almost_equal(y_hat, [0, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0], 4)


def test_predict_raises(model, X):
	_test_raises(model, "predict", X, min_value=MIN_VALUE)

	d = [Exponential(), Exponential()]
	model = BayesClassifier(d)
	_test_raises(model, "predict", X, min_value=MIN_VALUE)


def test_predict_proba(model, X):
	y_hat = model.predict_proba(X)
	assert_array_almost_equal(y_hat,
		[[5.2653e-01, 4.7347e-01],
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

	model2 = BayesClassifier(model.distributions)
	y_hat2 = model2.predict_proba(X)
	assert_array_almost_equal(y_hat2,
		[[3.2277e-01, 6.7723e-01],
         [1.1481e-02, 9.8852e-01],
         [4.9503e-08, 1.0000e+00],
         [2.9498e-09, 1.0000e+00],
         [9.3405e-01, 6.5951e-02],
         [5.4255e-16, 1.0000e+00],
         [9.2130e-01, 7.8700e-02],
         [1.0050e-06, 1.0000e+00],
         [9.0633e-01, 9.3666e-02],
         [2.8176e-05, 9.9997e-01],
         [9.9388e-01, 6.1207e-03]], 4)	


def test_predict_proba_raises(model, X):
	_test_raises(model, "predict_proba", X, min_value=MIN_VALUE)

	d = [Exponential(), Exponential()]
	model = BayesClassifier(d)
	_test_raises(model, "predict_proba", X, min_value=MIN_VALUE)


def test_predict_log_proba(model, X):
	y_hat = model.predict_log_proba(X)
	assert_array_almost_equal(y_hat,
		[[-6.4145e-01, -7.4766e-01],
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

	model2 = BayesClassifier(model.distributions)
	y_hat2 = model2.predict_log_proba(X)
	assert_array_almost_equal(y_hat2,
		[[-1.1308e+00, -3.8974e-01],
         [-4.4671e+00, -1.1548e-02],
         [-1.6821e+01,  0.0000e+00],
         [-1.9642e+01,  0.0000e+00],
         [-6.8226e-02, -2.7188e+00],
         [-3.5150e+01,  0.0000e+00],
         [-8.1969e-02, -2.5421e+00],
         [-1.3810e+01, -9.5367e-07],
         [-9.8348e-02, -2.3680e+00],
         [-1.0477e+01, -2.8133e-05],
         [-6.1395e-03, -5.0961e+00]], 3)	


def test_predict_log_proba_raises(model, X):
	_test_raises(model, "predict_log_proba", X, min_value=MIN_VALUE)

	d = [Exponential(), Exponential()]
	model = BayesClassifier(d)
	_test_raises(model, "predict_log_proba", X, min_value=MIN_VALUE)

###


def test_partial_summarize(model, X, y):
	model.summarize(X[:4], y[:4])
	assert_array_almost_equal(model._w_sum, [2., 2.])

	model.summarize(X[4:], y[4:])
	assert_array_almost_equal(model._w_sum, [6., 5.])


	model = BayesClassifier([Exponential(), Exponential()])
	model.summarize(X[:4], y[:4])
	assert_array_almost_equal(model._w_sum, [2., 2.])

	model.summarize(X[4:], y[4:])
	assert_array_almost_equal(model._w_sum, [6., 5.])


def test_full_summarize(model, X, y):
	model.summarize(X, y)
	assert_array_almost_equal(model._w_sum, [6., 5.])


	model = BayesClassifier([Exponential(), Exponential()])
	model.summarize(X, y)
	assert_array_almost_equal(model._w_sum, [6., 5.])


def test_summarize_weighted(model, X, y, w):
	model.summarize(X, y, sample_weight=w)
	assert_array_almost_equal(model._w_sum, [7., 8.])

	model = BayesClassifier([Exponential(), Exponential()])
	model.summarize(X, y, sample_weight=w)
	assert_array_almost_equal(model._w_sum, [7., 8.])


def test_summarize_weighted_flat(model, X, y, w):
	w = numpy.array(w)[:,0] 

	model.summarize(X, y, sample_weight=w)
	assert_array_almost_equal(model._w_sum, [7., 8.])

	model = BayesClassifier([Exponential(), Exponential()])
	model.summarize(X, y, sample_weight=w)
	assert_array_almost_equal(model._w_sum, [7., 8.])


def test_summarize_weighted_2d(model, X, y):
	model.summarize(X, y, sample_weight=X)
	assert_array_almost_equal(model._w_sum, [7.666667, 5.333333])

	model = BayesClassifier([Exponential(), Exponential()])
	model.summarize(X, y, sample_weight=X)
	assert_array_almost_equal(model._w_sum, [7.666667, 5.333333])


def test_summarize_raises(model, X, y, w):
	assert_raises(ValueError, model.summarize, [X], y)
	assert_raises(ValueError, model.summarize, X[0], y)
	assert_raises((ValueError, TypeError), model.summarize, X[0][0], y)
	assert_raises(ValueError, model.summarize, [x[:-1] for x in X], y)
	assert_raises(ValueError, model.summarize, 
		[[-0.1 for i in range(3)] for x in X], y)

	assert_raises(ValueError, model.summarize, [X], y, w)
	assert_raises(ValueError, model.summarize, X, [y], w)
	assert_raises(ValueError, model.summarize, X, y, [w])
	assert_raises(ValueError, model.summarize, [X], y, [w])
	assert_raises(ValueError, model.summarize, X[:len(X)-1], y, w)
	assert_raises(ValueError, model.summarize, X, y[:len(y)-1], w)
	assert_raises(ValueError, model.summarize, X, y, w[:len(w)-1])


def test_from_summaries(model, X, y):
	model.summarize(X, y)
	model.from_summaries()

	assert_array_almost_equal(model._w_sum, [0., 0.])
	assert_array_almost_equal(model.priors, [6./11, 5./11])
	assert_array_almost_equal(model._log_priors, numpy.log([6./11, 5./11]))

	X_ = numpy.array(X)[numpy.array(y) == 0]
	d = Exponential().fit(X_)

	assert_array_almost_equal(d.scales, model.distributions[0].scales)

	model = BayesClassifier([Exponential(), Exponential()])
	model.summarize(X, y)
	model.from_summaries()

	assert_array_almost_equal(model._w_sum, [0., 0.])
	assert_array_almost_equal(model.priors, [6./11, 5./11])
	assert_array_almost_equal(model._log_priors, numpy.log([6./11, 5./11]))

	assert_array_almost_equal(d.scales, model.distributions[0].scales)


def test_from_summaries_weighted(model, X, y, w):
	model.summarize(X, y, sample_weight=w)
	model.from_summaries()

	assert_array_almost_equal(model._w_sum, [0., 0.])
	assert_array_almost_equal(model.priors, [0.466667, 0.533333])
	assert_array_almost_equal(model._log_priors, 
		numpy.log([0.466667, 0.533333]))

	idxs = numpy.array(y) == 0
	X_ = numpy.array(X)[idxs]
	w_ = numpy.array(w)[idxs]
	d = Exponential().fit(X_, sample_weight=w_)

	assert_array_almost_equal(d.scales, model.distributions[0].scales)

	model = BayesClassifier([Exponential(), Exponential()])
	model.summarize(X, y, sample_weight=w)
	model.from_summaries()

	assert_array_almost_equal(model._w_sum, [0., 0.])
	assert_array_almost_equal(model.priors, [0.466667, 0.533333])
	assert_array_almost_equal(model._log_priors, 
		numpy.log([0.466667, 0.533333]))

	assert_array_almost_equal(d.scales, model.distributions[0].scales)


def test_from_summaries_null(model):
	model.from_summaries()

	assert model.distributions[0].scales[0] != 2.1 
	assert model.distributions[1].scales[0] != 1.5

	assert_array_almost_equal(model._w_sum, [0., 0.])


def test_from_summaries_inertia(X, y):
	d = [Exponential([2.1, 0.3, 0.1]), Exponential([1.5, 3.1, 2.2])]
	model = BayesClassifier(d, priors=[0.2, 0.8], inertia=0.3)
	model.summarize(X, y)
	model.from_summaries()

	assert_array_almost_equal(model._w_sum, [0., 0.])
	assert_array_almost_equal(model.priors, [0.441818, 0.558182])
	assert_array_almost_equal(model._log_priors, 
		numpy.log([0.441818, 0.558182]))


	d = [Exponential([2.1, 0.3, 0.1]), Exponential([1.5, 3.1, 2.2])]
	model = BayesClassifier(d, priors=[0.2, 0.8], inertia=1.0)
	model.summarize(X, y)
	model.from_summaries()

	assert_array_almost_equal(model._w_sum, [0., 0.])
	assert_array_almost_equal(model.priors, [0.2, 0.8])
	assert_array_almost_equal(model._log_priors, numpy.log([0.2, 0.8]))


	s1, s2 = [2.1, 0.3, 0.1], [1.5, 3.1, 2.2] 
	d = [Exponential(s1, inertia=1.0), Exponential(s2, inertia=1.0)]
	model = BayesClassifier(d, priors=[0.2, 0.8])
	model.summarize(X, y)
	model.from_summaries()

	assert_array_almost_equal(model.distributions[0].scales, s1)
	assert_array_almost_equal(model.distributions[1].scales, s2)

	assert_array_almost_equal(model._w_sum, [0., 0.])
	assert_array_almost_equal(model.priors, [0.545455, 0.454545])
	assert_array_almost_equal(model._log_priors, 
		numpy.log([0.545455, 0.454545]))


	d = [Exponential(s1, inertia=1.0), Exponential(s2)]
	model = BayesClassifier(d, priors=[0.2, 0.8], inertia=1.0)
	model.summarize(X, y)
	model.from_summaries()

	assert_array_almost_equal(model.distributions[0].scales, s1)
	assert_array_almost_equal(model.distributions[1].scales, [1.2, 1.4, 0.6])

	assert_array_almost_equal(model._w_sum, [0., 0.])
	assert_array_almost_equal(model.priors, [0.2, 0.8])
	assert_array_almost_equal(model._log_priors, numpy.log([0.2, 0.8]))


def test_from_summaries_weighted_inertia(X, y, w):
	d = [Exponential([2.1, 0.3, 0.1]), Exponential([1.5, 3.1, 2.2])]
	model = BayesClassifier(d, priors=[0.2, 0.8], inertia=0.3)
	model.summarize(X, y, sample_weight=w)
	model.from_summaries()

	assert_array_almost_equal(model._w_sum, [0., 0.])
	assert_array_almost_equal(model.priors, [0.386667, 0.613333])
	assert_array_almost_equal(model._log_priors, 
		numpy.log([0.386667, 0.613333]))


	d = [Exponential([2.1, 0.3, 0.1]), Exponential([1.5, 3.1, 2.2])]
	model = BayesClassifier(d, priors=[0.2, 0.8], inertia=1.0)
	model.summarize(X, y, sample_weight=w)
	model.from_summaries()

	assert_array_almost_equal(model._w_sum, [0., 0.])
	assert_array_almost_equal(model.priors, [0.2, 0.8])
	assert_array_almost_equal(model._log_priors, numpy.log([0.2, 0.8]))


def test_from_summaries_frozen(model, X, y):
	d = [Exponential([2.1, 0.3, 0.1]), Exponential([1.5, 3.1, 2.2])]
	model = BayesClassifier(d, priors=[0.2, 0.8], frozen=True)
	model.summarize(X, y)
	model.from_summaries()

	assert_array_almost_equal(model._w_sum, [0., 0.])
	assert_array_almost_equal(model.priors, [0.2, 0.8])
	assert_array_almost_equal(model._log_priors, numpy.log([0.2, 0.8]))


	s1, s2 = [2.1, 0.3, 0.1], [1.5, 3.1, 2.2] 
	d = [Exponential(s1, frozen=True), Exponential(s2, frozen=True)]
	model = BayesClassifier(d, priors=[0.2, 0.8])
	model.summarize(X, y)
	model.from_summaries()

	assert_array_almost_equal(model.distributions[0].scales, s1)
	assert_array_almost_equal(model.distributions[1].scales, s2)

	assert_array_almost_equal(model._w_sum, [0., 0.])
	assert_array_almost_equal(model.priors, [0.545455, 0.454545])
	assert_array_almost_equal(model._log_priors, 
		numpy.log([0.545455, 0.454545]))


	d = [Exponential(s1, frozen=True), Exponential(s2)]
	model = BayesClassifier(d, priors=[0.2, 0.8], frozen=True)
	model.summarize(X, y)
	model.from_summaries()

	assert_array_almost_equal(model.distributions[0].scales, s1)
	assert_array_almost_equal(model.distributions[1].scales, [1.2, 1.4, 0.6])

	assert_array_almost_equal(model._w_sum, [0., 0.])
	assert_array_almost_equal(model.priors, [0.2, 0.8])
	assert_array_almost_equal(model._log_priors, numpy.log([0.2, 0.8]))


def test_fit(model, X, y):
	model.fit(X, y)

	assert_array_almost_equal(model._w_sum, [0., 0.])
	assert_array_almost_equal(model.priors, [6./11, 5./11])
	assert_array_almost_equal(model._log_priors, numpy.log([6./11, 5./11]))

	X_ = numpy.array(X)[numpy.array(y) == 0]
	d = Exponential().fit(X_)

	assert_array_almost_equal(d.scales, model.distributions[0].scales)

	model = BayesClassifier([Exponential(), Exponential()])
	model.fit(X, y)

	assert_array_almost_equal(model._w_sum, [0., 0.])
	assert_array_almost_equal(model.priors, [6./11, 5./11])
	assert_array_almost_equal(model._log_priors, numpy.log([6./11, 5./11]))

	assert_array_almost_equal(d.scales, model.distributions[0].scales)


def test_fit_weighted(model, X, y, w):
	model.fit(X, y, sample_weight=w)

	assert_array_almost_equal(model._w_sum, [0., 0.])
	assert_array_almost_equal(model.priors, [0.466667, 0.533333])
	assert_array_almost_equal(model._log_priors, 
		numpy.log([0.466667, 0.533333]))

	idxs = numpy.array(y) == 0
	X_ = numpy.array(X)[idxs]
	w_ = numpy.array(w)[idxs]
	d = Exponential().fit(X_, sample_weight=w_)

	assert_array_almost_equal(d.scales, model.distributions[0].scales)

	model = BayesClassifier([Exponential(), Exponential()])
	model.fit(X, y, sample_weight=w)

	assert_array_almost_equal(model._w_sum, [0., 0.])
	assert_array_almost_equal(model.priors, [0.466667, 0.533333])
	assert_array_almost_equal(model._log_priors, 
		numpy.log([0.466667, 0.533333]))

	assert_array_almost_equal(d.scales, model.distributions[0].scales)


def test_fit_chain(X, y):
	model = BayesClassifier([Exponential(), Exponential()]).fit(X, y)

	assert_array_almost_equal(model._w_sum, [0., 0.])
	assert_array_almost_equal(model.priors, [6./11, 5./11])
	assert_array_almost_equal(model._log_priors, numpy.log([6./11, 5./11]))


def test_fit_raises(model, X, w, y):
	assert_raises(ValueError, model.fit, [X], y)
	assert_raises(ValueError, model.fit, X[0], y)
	assert_raises((ValueError, TypeError), model.fit, X[0][0], y)
	assert_raises(ValueError, model.fit, [x[:-1] for x in X], y)
	assert_raises(ValueError, model.fit, 
		[[-0.1 for i in range(3)] for x in X], y)

	assert_raises(ValueError, model.fit, [X], y, w)
	assert_raises(ValueError, model.fit, X, [y], w)
	assert_raises(ValueError, model.fit, X, y, [w])
	assert_raises(ValueError, model.fit, [X], y, [w])
	assert_raises(ValueError, model.fit, X[:len(X)-1], y, w)
	assert_raises(ValueError, model.fit, X, y[:len(y)-1], w)
	assert_raises(ValueError, model.fit, X, y, w[:len(w)-1])


def test_serialization(X, model):
	torch.save(model, ".pytest.torch")
	model2 = torch.load(".pytest.torch")
	os.system("rm .pytest.torch")

	assert_array_almost_equal(model2.priors, model.priors)
	assert_array_almost_equal(model2._log_priors, model._log_priors)

	assert_array_almost_equal(model2.predict_proba(X), model.predict_proba(X))

	m1d1, m1d2 = model.distributions
	m2d1, m2d2 = model2.distributions

	assert m1d1 is not m2d1
	assert m1d2 is not m2d2

	assert_array_almost_equal(m1d1.scales, m2d1.scales)
	assert_array_almost_equal(m1d2.scales, m2d2.scales)


def test_masked_probability(model, X, X_masked):
	X = torch.tensor(numpy.array(X))
	y = [1.668138e-02, 1.911842e-02, 4.393471e-03, 1.633741e-03,
           9.786682e-02, 1.229918e-04, 1.585297e-01, 6.066021e-03,
           2.571130e-01, 9.765117e-03, 1.114044e+01]

	mask = torch.ones_like(X).type(torch.bool)
	X_ = torch.masked.MaskedTensor(X, mask=mask)

	assert_array_almost_equal(y, model.probability(X_), 5)

	y =  [5.277007e-02, 1.175627e+00, 1.000000e+00, 1.633741e-03,
           1.533307e-01, 1.229918e-04, 1.000000e+00, 1.880462e-02,
           2.571130e-01, 9.765117e-03, 3.424242e+00]

	assert_array_almost_equal(y, model.probability(X_masked), 5)


def test_masked_log_probability(model, X, X_masked):
	X = torch.tensor(numpy.array(X))
	y = [-4.09346, -3.9571 , -5.42764, -6.41688, -2.32415, -9.00339,
           -1.84181, -5.10505, -1.35824, -4.62894,  2.41058]

	mask = torch.ones_like(X).type(torch.bool)
	X_ = torch.masked.MaskedTensor(X, mask=mask)

	assert_array_almost_equal(y, model.log_probability(X_), 5)

	y = [-2.94181,  0.1618 ,  0.     , -6.41688, -1.87516, -9.00339,
            0.     , -3.97365, -1.35824, -4.62894,  1.23088]

	assert_array_almost_equal(y, model.log_probability(X_masked), 5)


def test_masked_emission_matrix(model, X, X_masked):
	X = torch.tensor(numpy.array(X))
	mask = torch.ones_like(X).type(torch.bool)
	X_ = torch.masked.MaskedTensor(X, mask=mask)

	e = model._emission_matrix(X_)

	assert_array_almost_equal(e, 
		[[ -4.7349,  -4.8411],
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

	d = [Exponential([2.1, 0.3, 0.1]), Exponential([1.5, 3.1, 2.2])]
	model = BayesClassifier(d)
	e = model._emission_matrix(X_masked)

	assert_array_almost_equal(e, 
		[[ -3.8533,  -3.2582],
         [ -0.2311,  -2.2300],
         [ -0.6931,  -0.6931],
         [-25.5476,  -5.9061],
         [ -2.8225,  -2.1471],
         [-43.6428,  -8.4926],
         [ -0.6931,  -0.6931],
         [-19.6087,  -3.4628],
         [ -1.7381,  -4.0077],
         [-14.5952,  -4.1182],
         [  0.8675,  -1.8871]], 4)


def test_masked_summarize(model, X, X_masked, w, y):
	X = torch.tensor(numpy.array(X))
	mask = torch.ones_like(X).type(torch.bool)
	X_ = torch.masked.MaskedTensor(X, mask=mask)

	d = [Exponential([2.1, 0.3, 0.1]), Exponential([1.5, 3.1, 2.2])]
	model = BayesClassifier(d, priors=[0.7, 0.3])
	model.summarize(X_, y, sample_weight=w)
	assert_array_almost_equal(model._w_sum, [7.0, 8.0])

	d = [Exponential([2.1, 0.3, 0.1]), Exponential([1.5, 3.1, 2.2])]
	model = BayesClassifier(d, priors=[0.7, 0.3])
	model.summarize(X_masked, y, sample_weight=w)
	assert_array_almost_equal(model._w_sum, [5.0, 8.0])


def test_masked_from_summaries(model, X, X_masked, y):
	X = torch.tensor(numpy.array(X))
	mask = torch.ones_like(X).type(torch.bool)
	X_ = torch.masked.MaskedTensor(X, mask=mask)

	model.summarize(X_, y)
	model.from_summaries()

	assert_array_almost_equal(model._w_sum, [0., 0.])
	assert_array_almost_equal(model.priors, [0.545455, 0.454545])
	assert_array_almost_equal(model._log_priors, 
		numpy.log([0.545455, 0.454545]))

	d = [Exponential([2.1, 0.3, 0.1]), Exponential([1.5, 3.1, 2.2])]
	model = BayesClassifier(d, priors=[0.7, 0.3])
	model.summarize(X_masked, y, sample_weight=w)
	model.from_summaries()

	assert_array_almost_equal(model._w_sum, [0., 0.])
	assert_array_almost_equal(model.priors, [0.444444, 0.555556])
	assert_array_almost_equal(model._log_priors, 
		numpy.log([0.444444, 0.555556]))


def test_masked_fit(X, X_masked, y):
	X = torch.tensor(numpy.array(X))
	mask = torch.ones_like(X).type(torch.bool)
	X_ = torch.masked.MaskedTensor(X, mask=mask)

	d = [Exponential([2.1, 0.3, 1.1]), Exponential([1.5, 3.1, 2.2])]
	model = BayesClassifier(d)
	model.fit(X_, y)

	assert_array_almost_equal(model._w_sum, [0., 0.])
	assert_array_almost_equal(model.priors, [0.545455, 0.454545])
	assert_array_almost_equal(model._log_priors, 
		numpy.log([0.545455, 0.454545]))

	d = [Exponential([2.1, 0.3, 0.1]), Exponential([1.5, 3.1, 2.2])]
	model = BayesClassifier(d)
	model.fit(X_masked, y)

	assert_array_almost_equal(model._w_sum, [0., 0.])
	assert_array_almost_equal(model.priors, [0.444444, 0.555556])
	assert_array_almost_equal(model._log_priors, 
		numpy.log([0.444444, 0.555556]))
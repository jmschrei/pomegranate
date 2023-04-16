# test_bayes_classifier.py
# Contact: Jacob Schreiber <jmschreiber91@gmail.com>

import numpy
import torch
import pytest

from pomegranate.hmm import DenseHMM
from pomegranate.distributions import Exponential
from pomegranate.distributions import Gamma

from ..distributions._utils import _test_initialization_raises_one_parameter
from ..distributions._utils import _test_initialization
from ..distributions._utils import _test_predictions
from ..distributions._utils import _test_efd_from_summaries
from ..distributions._utils import _test_raises

from nose.tools import assert_raises
from numpy.testing import assert_array_almost_equal
from numpy.testing import assert_array_equal


MIN_VALUE = 0
MAX_VALUE = None
VALID_VALUE = 1.2


@pytest.fixture
def X():
	return [[[1, 2, 0],
	      [0, 0, 1],
	      [1, 1, 2],
	      [2, 2, 2],
	      [3, 1, 0]],
	     [[5, 1, 4],
	      [2, 1, 0],
	      [1, 0, 2],
	      [1, 1, 0],
	      [0, 2, 1]]]


@pytest.fixture
def X_masked(X):
	X = torch.tensor(numpy.array(X))

	mask = [[[True , True , True ],
	         [True , False, False],
	         [False, False, False],
	         [True , True , True ],
	         [True , True , True ]],
	        [[True , True , True ],
	         [False, True , True ],
	         [True , False, True ],
	         [True , True , True ],
	         [False, False, False]]]

	mask = torch.tensor(numpy.array(mask), dtype=torch.bool)
	return torch.masked.MaskedTensor(X, mask=mask)


@pytest.fixture
def w():
	return [1, 2.3]


@pytest.fixture
def model():
	starts = [0.2, 0.8]
	ends = [0.1, 0.1]

	edges = [[0.1, 0.8],
	         [0.3, 0.6]]

	d = [Exponential([2.1, 0.3, 0.1]), Exponential([1.5, 3.1, 2.2])]
	model = DenseHMM(distributions=d, edges=edges, starts=starts, ends=ends, 
		random_state=0)
	return model


@pytest.fixture
def model2():
	d1 = Exponential([2.1, 0.3, 0.1])
	d2 = Exponential([1.5, 3.1, 2.2])

	model = DenseHMM()
	model.add_distributions([d1, d2])
	model.add_edge(model.start, d1, 0.2)
	model.add_edge(model.start, d2, 0.8)
	
	model.add_edge(d1, d1, 0.1)
	model.add_edge(d1, d2, 0.8)
	model.add_edge(d1, model.end, 0.1)

	model.add_edge(d2, d1, 0.3)
	model.add_edge(d2, d2, 0.6)
	model.add_edge(d2, model.end, 0.1)
	return model


###


def test_initialization(X):
	d = [Exponential(), Exponential()]
	model = DenseHMM(d)

	assert model.inertia == 0.0
	assert model.frozen == False
	assert model.n_distributions == 2

	assert_raises(AttributeError, getattr, model, "_xw_sum")
	assert_raises(AttributeError, getattr, model, "_xw_starts_sum")
	assert_raises(AttributeError, getattr, model, "_xw_ends_sum")

	assert model.starts is None
	assert model.ends is None
	assert model.edges is None


def test_add_distribution():
	d1 = Exponential()
	d2 = Gamma()

	model = DenseHMM()
	assert model.distributions == None
	assert model.n_distributions == 0

	model.add_distribution(d1)
	assert model.distributions == [d1]
	assert model.n_distributions == 1

	model.add_distribution(d2)
	assert model.distributions == [d1, d2]
	assert model.n_distributions == 2

	model = DenseHMM([d1, d2])
	assert model.distributions == [d1, d2]
	assert model.n_distributions == 2

	model.add_distribution(d1)
	assert model.distributions == [d1, d2, d1]
	assert model.n_distributions == 3


def test_add_distributions():
	d1 = Exponential()
	d2 = Gamma()

	model = DenseHMM()
	assert model.distributions == None
	assert model.n_distributions == 0

	model.add_distributions([d1, d2])
	assert model.distributions == [d1, d2]
	assert model.n_distributions == 2

	model = DenseHMM([d1, d2])
	assert model.distributions == [d1, d2]
	assert model.n_distributions == 2

	model.add_distributions([d1, d2])
	assert model.distributions == [d1, d2, d1, d2]
	assert model.n_distributions == 4


def test_add_start_edge():
	d1 = Exponential()
	d2 = Gamma()

	model = DenseHMM([d1, d2])
	model.add_edge(model.start, d1, 0.3)
	model.add_edge(model.start, d2, 0.7)
	model._initialize()

	assert_array_almost_equal(model.edges, numpy.log([[0.5, 0.5], [0.5, 0.5]]))
	assert_array_almost_equal(model.starts, numpy.log([0.3, 0.7]))
	assert_array_almost_equal(model.ends, numpy.log([0.5, 0.5]))


def test_add_end_edge():
	d1 = Exponential()
	d2 = Gamma()

	model = DenseHMM([d1, d2])
	model.add_edge(d1, model.end, 0.2)
	model.add_edge(d2, model.end, 0.3)
	model._initialize()

	assert_array_almost_equal(model.edges, numpy.log([[0.5, 0.5], [0.5, 0.5]]))
	assert_array_almost_equal(model.starts, numpy.log([0.5, 0.5]))
	assert_array_almost_equal(model.ends, numpy.log([0.2, 0.3]))


def test_add_edge():
	d1 = Exponential()
	d2 = Gamma()

	model = DenseHMM([d1, d2])
	model.add_edge(d1, d1, 0.2)
	model.add_edge(d1, d2, 0.8)
	model.add_edge(d2, d1, 0.3)
	model.add_edge(d2, d2, 0.4)
	model.add_edge(d2, model.end, 0.3)
	model._initialize()

	assert_array_almost_equal(model.edges, numpy.log([[0.2, 0.8], [0.3, 0.4]]))
	assert_array_almost_equal(model.starts, numpy.log([0.5, 0.5]))
	assert_array_almost_equal(model.ends, numpy.log([0.0, 0.3]))


def test_initialization_raises():
	d = [Exponential(), Exponential()]

	assert_raises(ValueError, DenseHMM, d, edges=[0.2, 0.2, 0.6])
	assert_raises(ValueError, DenseHMM, d, edges=[0.2, 1.0])
	assert_raises(ValueError, DenseHMM, d, edges=[[-0.2, 0.9], [0.2, 0.8]])
	assert_raises(ValueError, DenseHMM, d, edges=[[0.3, 1.1], [0.2, 0.8]])
	assert_raises(ValueError, DenseHMM, d, edges=[[0.2, 0.6, 0.2], 
		[0.2, 0.7, 0.1], [0.1, 0.3, 0.6]])
	assert_raises(ValueError, DenseHMM, d, edges=[[[0.2, 0.8], [0.2, 0.8]]])

	assert_raises(ValueError, DenseHMM, d, starts=[0.1, 0.3])
	assert_raises(ValueError, DenseHMM, d, starts=[0.1, 1.2])
	assert_raises(ValueError, DenseHMM, d, starts=[-0.1, 1.1])
	assert_raises(ValueError, DenseHMM, d, starts=[0.5, 0.6])
	assert_raises(ValueError, DenseHMM, d, starts=[0.1, 0.3, 0.3, 0.3])
	assert_raises(ValueError, DenseHMM, d, starts=[[0.1, 0.9]])
	assert_raises(ValueError, DenseHMM, d, starts=[[0.1], [0.9]])

	assert_raises(ValueError, DenseHMM, d, ends=[0.1, 1.2])
	assert_raises(ValueError, DenseHMM, d, ends=[-0.1, 1.1])
	assert_raises(ValueError, DenseHMM, d, ends=[0.1, 0.3, 0.3, 0.3])
	assert_raises(ValueError, DenseHMM, d, ends=[[0.1, 0.9]])
	assert_raises(ValueError, DenseHMM, d, ends=[[0.1], [0.9]])

	assert_raises(ValueError, DenseHMM, d, max_iter=0)
	assert_raises(ValueError, DenseHMM, d, max_iter=-1)
	assert_raises(ValueError, DenseHMM, d, max_iter=1.3)
	
	assert_raises(ValueError, DenseHMM, d, tol=-1)

	assert_raises((ValueError, TypeError), DenseHMM, Exponential)
	assert_raises(ValueError, DenseHMM, d, inertia=-0.4)
	assert_raises(ValueError, DenseHMM, d, inertia=1.2)
	assert_raises(ValueError, DenseHMM, d, inertia=1.2, frozen="true")
	assert_raises(ValueError, DenseHMM, d, inertia=1.2, frozen=3)
	

def test_reset_cache(model, X):
	model.summarize(X)
	assert_array_almost_equal(model._xw_sum, 
		[[2.666838e-04, 1.895245e+00], [2.635103e+00, 3.469387e+00]], 4)
	assert_array_almost_equal(model._xw_starts_sum, [0.136405, 1.863595], 4)
	assert_array_almost_equal(model._xw_ends_sum, [0.876264, 1.123736], 4)

	model._reset_cache()
	assert_array_almost_equal(model._xw_sum, [[0., 0.], [0., 0.]])
	assert_array_almost_equal(model._xw_starts_sum, [0., 0.])
	assert_array_almost_equal(model._xw_ends_sum, [0., 0.])


def test_initialize(X):
	d = [Exponential(), Exponential()]
	model = DenseHMM(d, random_state=0)

	d1 = model.distributions[0]
	d2 = model.distributions[1]

	assert model.d is None
	assert model.n_distributions == 2
	assert model._initialized == False

	assert d1._initialized == False
	assert d2._initialized == False

	assert_raises(AttributeError, getattr, model, "_xw_sum")
	assert_raises(AttributeError, getattr, model, "_xw_starts_sum")
	assert_raises(AttributeError, getattr, model, "_xw_ends_sum")

	model._initialize(X)
	assert model._initialized == True
	assert model.d == 3
	assert isinstance(model, DenseHMM)

	assert d1._initialized == True
	assert d2._initialized == True

	assert_array_almost_equal(d1.scales, [1.5, 1. , 2. ])
	assert_array_almost_equal(d2.scales, [1.75, 1.25, 0.  ])


###


@pytest.mark.sample
def test_sample(model):
	torch.manual_seed(0)

	X = model.sample(1)
	assert_array_almost_equal(X[0],
		[[5.2625, 3.8142, 1.3531],
         [3.8027, 3.2107, 3.4455],
         [0.2951, 1.3407, 1.9155],
         [0.5100, 1.8695, 0.4280],
         [0.6584, 3.0151, 1.4465],
         [0.8719, 1.6214, 0.5059],
         [1.6406, 4.7632, 0.5196],
         [1.6939, 0.3603, 1.6320]], 4)

	X = model.sample(3)
	assert_array_almost_equal(X[0],
		[[0.3843, 2.5327, 1.9483],
         [1.7183, 4.2407, 0.5683],
         [2.4902, 3.0871, 0.4796],
         [0.2486, 2.0362, 5.4891]], 3)
	assert_array_almost_equal(X[1],
		[[1.7823e+00, 1.9666e-01, 1.5899e-01],
         [9.6352e-01, 9.5312e-02, 4.2323e-03],
         [2.9002e+00, 1.2545e-01, 7.7268e-02],
         [9.4207e-01, 6.1731e+00, 9.2237e-02],
         [8.8499e+00, 6.5642e+00, 3.1673e+00],
         [4.4337e-01, 6.5174e-01, 9.6875e+00]], 3)
	assert_array_almost_equal(X[2],
		[[2.2685e+00, 1.3926e+00, 1.8356e+00],
         [2.0275e+00, 1.3866e-02, 1.9275e+00],
         [1.2650e+00, 1.5342e-01, 3.9146e-01],
         [1.3925e+01, 2.7539e-01, 1.6151e-02],
         [7.5165e-01, 6.1712e+00, 2.5927e-01],
         [3.4823e+00, 4.6675e-01, 1.5978e-01]], 3)


def test_sample_length(model):
	starts = [0.2, 0.8]
	ends = [0.1, 0.1]

	edges = [[0.1, 0.8],
	         [0.3, 0.6]]

	d = [Exponential([2.1, 0.3, 0.1]), Exponential([1.5, 3.1, 2.2])]
	model = DenseHMM(distributions=d, edges=edges, starts=starts, ends=ends,
		sample_length=3, random_state=0)

	X = model.sample(25)
	assert max([len(x) for x in X]) <= 3


def test_sample_paths(model):
	torch.manual_seed(0)

	starts = [0.2, 0.8]
	ends = [0.1, 0.1]

	edges = [[0.4, 0.5],
	         [0.6, 0.3]]

	d = [Exponential([2.1, 0.3, 0.1]), Exponential([1.5, 3.1, 2.2])]
	model = DenseHMM(distributions=d, edges=edges, starts=starts, ends=ends,
		return_sample_paths=True, random_state=0)

	X, path = model.sample(1)
	assert_array_equal(path[0],
		[1, 1, 1, 0, 1, 1, 0, 1])


###


def test_emission_matrix(model, X):
	e = model._emission_matrix(X)

	assert_array_almost_equal(e, 
		[[[ -4.3782,  -3.6372],
          [ -7.2354,  -2.7799],
          [-21.0449,  -4.2237],
          [-24.8544,  -5.2129],
          [ -1.9973,  -4.6479]],

         [[-42.9497,  -7.7994],
          [ -1.5211,  -3.9812],
          [-17.7116,  -3.9011],
          [ -1.0449,  -3.3146],
          [-13.9020,  -3.4250]]], 4)


def test_emission_matrix_raises(model, X):
	f = getattr(model, "_emission_matrix")

	assert_raises(ValueError, f, [X])
	assert_raises(ValueError, f, X[0])
	assert_raises((ValueError, TypeError, RuntimeError), f, X[0][0])

	if MIN_VALUE is not None:
		assert_raises(ValueError, f, 
			[[[MIN_VALUE-0.1 for i in range(model.d)] for j in range(4)]])
	
	if MAX_VALUE is not None:
		assert_raises(ValueError, f, 
			[[[MAX_VALUE+0.1 for i in range(model.d)] for j in range(4)]])


def test_log_probability(model, X):
	logp = model.log_probability(X)
	assert_array_almost_equal(logp, [-22.8266, -22.8068], 4)


def test_log_probability_raises(model, X):
	f = getattr(model, "log_probability")

	assert_raises(ValueError, f, [X])
	assert_raises(ValueError, f, X[0])
	assert_raises((ValueError, TypeError, RuntimeError), f, X[0][0])

	if MIN_VALUE is not None:
		assert_raises(ValueError, f, 
			[[[MIN_VALUE-0.1 for i in range(model.d)] for j in range(4)]])
	
	if MAX_VALUE is not None:
		assert_raises(ValueError, f, 
			[[[MAX_VALUE+0.1 for i in range(model.d)] for j in range(4)]])


def test_probability(model, X):
	logp = model.probability(X)
	assert_array_almost_equal(logp, [1.2205e-09, 1.2449e-09], 4)


def test_probability_raises(model, X):
	f = getattr(model, "probability")

	assert_raises(ValueError, f, [X])
	assert_raises(ValueError, f, X[0])
	assert_raises((ValueError, TypeError, RuntimeError), f, X[0][0])

	if MIN_VALUE is not None:
		assert_raises(ValueError, f, 
			[[[MIN_VALUE-0.1 for i in range(model.d)] for j in range(4)]])
	
	if MAX_VALUE is not None:
		assert_raises(ValueError, f, 
			[[[MAX_VALUE+0.1 for i in range(model.d)] for j in range(4)]])


def test_forward(model, X):
	y_hat = model.forward(X)

	assert_array_almost_equal(y_hat,
		[[[ -5.9877,  -3.8603],
          [-12.2607,  -7.0036],
          [-29.2507, -11.7311],
          [-37.7895, -17.4549],
          [-20.6561, -22.6136]],

         [[-44.5591,  -8.0226],
          [-10.7476, -12.5146],
          [-30.3480, -14.7513],
          [-17.0002, -18.5767],
          [-32.7223, -20.5042]]], 4)


def test_forward_raises(model, X):
	f = getattr(model, "forward")

	assert_raises(ValueError, f, [X])
	assert_raises(ValueError, f, X[0])
	assert_raises((ValueError, TypeError, RuntimeError), f, X[0][0])

	if MIN_VALUE is not None:
		assert_raises(ValueError, f, 
			[[[MIN_VALUE-0.1 for i in range(model.d)] for j in range(4)]])
	
	if MAX_VALUE is not None:
		assert_raises(ValueError, f, 
			[[[MAX_VALUE+0.1 for i in range(model.d)] for j in range(4)]])


def test_backward(model, X):
	y_hat = model.backward(X)

	assert_array_almost_equal(y_hat,
		[[[-18.8311, -19.1130],
          [-15.5423, -15.8300],
          [-10.8078, -11.0955],
          [ -6.1547,  -5.3717],
          [ -2.3026,  -2.3026]],

         [[-15.5896, -14.7842],
          [-12.1797, -12.4674],
          [ -8.8158,  -8.0555],
          [ -5.9508,  -6.2384],
          [ -2.3026,  -2.3026]]], 4)


def test_backward_raises(model, X):
	f = getattr(model, "backward")

	assert_raises(ValueError, f, [X])
	assert_raises(ValueError, f, X[0])
	assert_raises((ValueError, TypeError, RuntimeError), f, X[0][0])

	if MIN_VALUE is not None:
		assert_raises(ValueError, f, 
			[[[MIN_VALUE-0.1 for i in range(model.d)] for j in range(4)]])
	
	if MAX_VALUE is not None:
		assert_raises(ValueError, f, 
			[[[MAX_VALUE+0.1 for i in range(model.d)] for j in range(4)]])


def test_forward_backward(model, X):
	expected_transitions, fb, starts, ends, logp = model.forward_backward(X)

	assert_array_almost_equal(expected_transitions,
		[[[2.6353e-04, 1.4304e-01], [8.8289e-01, 2.9738e+00]],
         [[3.1500e-06, 1.7522e+00], [1.7522e+00, 4.9559e-01]]], 3)

	assert_array_almost_equal(fb,
		[[[-1.9921e+00, -1.4665e-01],
          [-4.9764e+00, -6.9228e-03],
          [-1.7232e+01, -3.2831e-08],
          [-2.1118e+01, -6.7415e-10],
          [-1.3209e-01, -2.0896e+00]],

         [[-3.7342e+01,  0.0000e+00],
          [-1.2056e-01, -2.1752e+00],
          [-1.6357e+01, -7.8752e-08],
          [-1.4412e-01, -2.0083e+00],
          [-1.2218e+01, -4.9402e-06]]], 3)

	assert_array_almost_equal(starts, 
		[[1.3641e-01, 8.6359e-01],
         [6.0619e-17, 1.0000e+00]], 3)

	assert_array_almost_equal(ends,
		[[8.7626e-01, 1.2374e-01],
         [4.9402e-06, 1.0000e+00]], 3)

	assert_array_almost_equal(logp, [-22.8266, -22.8068], 3)


def test_predict(model, X):
	y_hat = model.predict(X)
	assert_array_almost_equal(y_hat, 
		[[1, 1, 1, 1, 0],
         [1, 0, 1, 0, 1]], 4)


def test_predict_raises(model, X):
	f = getattr(model, "predict")

	assert_raises(ValueError, f, [X])
	assert_raises(ValueError, f, X[0])
	assert_raises((ValueError, TypeError, RuntimeError), f, X[0][0])

	if MIN_VALUE is not None:
		assert_raises(ValueError, f, 
			[[[MIN_VALUE-0.1 for i in range(model.d)] for j in range(4)]])
	
	if MAX_VALUE is not None:
		assert_raises(ValueError, f, 
			[[[MAX_VALUE+0.1 for i in range(model.d)] for j in range(4)]])


def test_predict_proba(model, X):
	y_hat = model.predict_proba(X)

	assert_array_almost_equal(y_hat,
		[[[1.3641e-01, 8.6359e-01],
          [6.8989e-03, 9.9310e-01],
          [3.2831e-08, 1.0000e+00],
          [6.7415e-10, 1.0000e+00],
          [8.7626e-01, 1.2374e-01]],

         [[6.0619e-17, 1.0000e+00],
          [8.8642e-01, 1.1358e-01],
          [7.8752e-08, 1.0000e+00],
          [8.6578e-01, 1.3422e-01],
          [4.9402e-06, 1.0000e+00]]], 4)

	assert_array_almost_equal(torch.sum(y_hat, dim=-1),
		[[1.0000, 1.0000, 1.0000, 1.0000, 1.0000],
         [1.0000, 1.0000, 1.0000, 1.0000, 1.0000]])


def test_predict_proba_raises(model, X):
	f = getattr(model, "predict_proba")

	assert_raises(ValueError, f, [X])
	assert_raises(ValueError, f, X[0])
	assert_raises((ValueError, TypeError, RuntimeError), f, X[0][0])

	if MIN_VALUE is not None:
		assert_raises(ValueError, f, 
			[[[MIN_VALUE-0.1 for i in range(model.d)] for j in range(4)]])
	
	if MAX_VALUE is not None:
		assert_raises(ValueError, f, 
			[[[MAX_VALUE+0.1 for i in range(model.d)] for j in range(4)]])


def test_predict_log_proba(model, X):
	y_hat = model.predict_log_proba(X)

	assert_array_almost_equal(y_hat,
		[[[-1.9921e+00, -1.4665e-01],
          [-4.9764e+00, -6.9228e-03],
          [-1.7232e+01, -3.2831e-08],
          [-2.1118e+01, -6.7415e-10],
          [-1.3209e-01, -2.0896e+00]],

         [[-3.7342e+01,  0.0000e+00],
          [-1.2056e-01, -2.1752e+00],
          [-1.6357e+01, -7.8752e-08],
          [-1.4412e-01, -2.0083e+00],
          [-1.2218e+01, -4.9402e-06]]], 3)


def test_predict_log_proba_raises(model, X):
	f = getattr(model, "predict_log_proba")

	assert_raises(ValueError, f, [X])
	assert_raises(ValueError, f, X[0])
	assert_raises((ValueError, TypeError, RuntimeError), f, X[0][0])

	if MIN_VALUE is not None:
		assert_raises(ValueError, f, 
			[[[MIN_VALUE-0.1 for i in range(model.d)] for j in range(4)]])
	
	if MAX_VALUE is not None:
		assert_raises(ValueError, f, 
			[[[MAX_VALUE+0.1 for i in range(model.d)] for j in range(4)]])
###


def test_partial_summarize(model, X):
	d1 = model.distributions[0]
	d2 = model.distributions[1]
	model.summarize(X[:1])

	assert_array_almost_equal(model._xw_sum,
		[[2.635337e-04, 1.430405e-01], [8.828942e-01, 2.973798e+00]], 4)
	assert_array_almost_equal(model._xw_starts_sum, [0.136405, 0.863595], 4)
	assert_array_almost_equal(model._xw_ends_sum, [0.876259, 0.123741], 4)

	assert_array_almost_equal(d1._w_sum, [1.019563, 1.019563, 1.019563], 4)
	assert_array_almost_equal(d1._xw_sum, [2.765183, 1.149069, 0.006899], 4)

	assert_array_almost_equal(d2._w_sum, [3.980437, 3.980437, 3.980437], 4)
	assert_array_almost_equal(d2._xw_sum, [4.234818, 4.850933, 4.9931], 4)	

	model.summarize(X[1:])
	assert_array_almost_equal(model._xw_sum, 
		[[2.666838e-04, 1.895245e+00], [2.635103e+00, 3.469387e+00]], 4)
	assert_array_almost_equal(model._xw_starts_sum, [0.136405, 1.863595], 4)
	assert_array_almost_equal(model._xw_ends_sum, [0.876264, 1.123736], 4)

	assert_array_almost_equal(d1._w_sum, [2.771773, 2.771773, 2.771773], 4)
	assert_array_almost_equal(d1._xw_sum, [5.403805, 2.901283, 0.006904], 4)

	assert_array_almost_equal(d2._w_sum, [7.228226, 7.228226, 7.228226], 4)
	assert_array_almost_equal(d2._xw_sum, [10.596193,  8.098717, 11.993094], 4)	


def test_summarize(model, X):
	d1 = model.distributions[0]
	d2 = model.distributions[1]
	model.summarize(X)

	assert_array_almost_equal(model._xw_sum, 
		[[2.666838e-04, 1.895245e+00], [2.635103e+00, 3.469387e+00]], 4)
	assert_array_almost_equal(model._xw_starts_sum, [0.136405, 1.863595], 4)
	assert_array_almost_equal(model._xw_ends_sum, [0.876264, 1.123736], 4)

	assert_array_almost_equal(d1._w_sum, [2.771771, 2.771771, 2.771771], 4)
	assert_array_almost_equal(d1._xw_sum, [5.403805, 2.901283, 0.006904], 4)

	assert_array_almost_equal(d2._w_sum, [7.228226, 7.228226, 7.228226], 4)
	assert_array_almost_equal(d2._xw_sum, [10.596193,  8.098717, 11.993094], 4)	


def test_summarize_weighted(model, X, w):
	d1 = model.distributions[0]
	d2 = model.distributions[1]
	model.summarize(X, sample_weight=w)

	assert_array_almost_equal(model._xw_sum, 
		[[2.707788e-04, 4.173112e+00], [4.912973e+00, 4.113652e+00]], 4)
	assert_array_almost_equal(model._xw_starts_sum, [0.136405, 3.163595], 4)
	assert_array_almost_equal(model._xw_ends_sum, [0.876271, 2.423729], 4)

	assert_array_almost_equal(d1._w_sum, [5.049643, 5.049643, 5.049643], 4)
	assert_array_almost_equal(d1._xw_sum, [8.834015e+00, 5.179160e+00, 
		6.910696e-03], 4)

	assert_array_almost_equal(d2._w_sum, [11.450353, 11.450353, 11.450353], 4)
	assert_array_almost_equal(d2._xw_sum, [18.865982, 12.320835, 21.093086], 4)	


def test_summarize_raises(model, X, w):
	assert_raises(ValueError, model.summarize, [X])
	assert_raises(ValueError, model.summarize, X[0])
	assert_raises((ValueError, TypeError), model.summarize, X[0][0])
	assert_raises(ValueError, model.summarize, 
		[[-0.1 for i in range(3)] for x in X])

	assert_raises(ValueError, model.summarize, [X], w)
	assert_raises(ValueError, model.summarize, X, [w])
	assert_raises(ValueError, model.summarize, [X], [w])
	assert_raises(ValueError, model.summarize, X[:len(X)-1], w)
	assert_raises(ValueError, model.summarize, X, w[:len(w)-1])


def test_from_summaries(model, X):
	d1 = model.distributions[0]
	d2 = model.distributions[1]

	model.summarize(X)
	model.from_summaries()

	assert_array_almost_equal(model.starts, [-2.685274, -0.07064])
	assert_array_almost_equal(model.ends, [-1.151575, -1.861335])
	assert_array_almost_equal(model.edges, 
		[[-9.248936, -0.38014 ], [-1.009071, -0.734016]], 4)

	assert_array_almost_equal(d1.scales, [1.949585, 1.046725, 0.002491])
	assert_array_almost_equal(d1._w_sum, [0., 0., 0.])
	assert_array_almost_equal(d1._xw_sum, [0., 0., 0.])

	assert_array_almost_equal(d2.scales, [1.465946, 1.120429, 1.659203])
	assert_array_almost_equal(d2._w_sum, [0., 0., 0.])
	assert_array_almost_equal(d2._xw_sum, [0., 0., 0.])


def test_from_summaries_weighted(model, X, w):
	d1 = model.distributions[0]
	d2 = model.distributions[1]

	model.summarize(X, sample_weight=w)
	model.from_summaries()

	assert_array_almost_equal(model.starts, [-3.186049, -0.042213])
	assert_array_almost_equal(model.ends, [-1.7514  , -1.552714])
	assert_array_almost_equal(model.edges, 
		[[-9.833528, -0.190657], [-0.846141, -1.023709]], 4)

	assert_array_almost_equal(d1.scales, 
		[1.749434e+00, 1.025649e+00, 1.368553e-03])
	assert_array_almost_equal(d1._w_sum, [0., 0., 0.])
	assert_array_almost_equal(d1._xw_sum, [0., 0., 0.])

	assert_array_almost_equal(d2.scales, [1.647633, 1.076022, 1.842134])
	assert_array_almost_equal(d2._w_sum, [0., 0., 0.])
	assert_array_almost_equal(d2._xw_sum, [0., 0., 0.])


def test_from_summaries_inertia(X):
	d = [Exponential([2.1, 0.3, 0.1]), Exponential([1.5, 3.1, 2.2])]
	model = DenseHMM(distributions=d, edges=[[0.1, 0.8], [0.3, 0.6]], 
		starts=[0.2, 0.8], ends=[0.1, 0.1], inertia=0.3)

	d1 = model.distributions[0]
	d2 = model.distributions[1]

	model.summarize(X)
	model.from_summaries()

	assert_array_almost_equal(model.starts, [-2.362523, -0.116391])
	assert_array_almost_equal(model.ends, [-1.496878, -1.99371])
	assert_array_almost_equal(model.edges, 
		[[-7.16503 , -0.333041], [-1.067542, -0.667059]], 4)

	assert_array_almost_equal(d1.scales, [1.949585, 1.046725, 0.002491])
	assert_array_almost_equal(d1._w_sum, [0., 0., 0.])
	assert_array_almost_equal(d1._xw_sum, [0., 0., 0.])

	assert_array_almost_equal(d2.scales, [1.465946, 1.120429, 1.659203])
	assert_array_almost_equal(d2._w_sum, [0., 0., 0.])
	assert_array_almost_equal(d2._xw_sum, [0., 0., 0.])


	d = [Exponential([2.1, 0.3, 0.1], inertia=0.25), 
	     Exponential([1.5, 3.1, 2.2], inertia=0.83)]
	model = DenseHMM(distributions=d, edges=[[0.1, 0.8], [0.3, 0.6]], 
		starts=[0.2, 0.8], ends=[0.1, 0.1], inertia=0.0)

	d1 = model.distributions[0]
	d2 = model.distributions[1]

	model.summarize(X)
	model.from_summaries()

	assert_array_almost_equal(model.starts, [-2.685274, -0.07064])
	assert_array_almost_equal(model.ends, [-1.151575, -1.861335])
	assert_array_almost_equal(model.edges, 
		[[-9.248936, -0.38014 ], [-1.009071, -0.734016]], 4)

	assert_array_almost_equal(d1.scales, [1.987189, 0.860044, 0.026868])
	assert_array_almost_equal(d1._w_sum, [0., 0., 0.])
	assert_array_almost_equal(d1._xw_sum, [0., 0., 0.])

	assert_array_almost_equal(d2.scales, [1.494211, 2.763473, 2.108064])
	assert_array_almost_equal(d2._w_sum, [0., 0., 0.])
	assert_array_almost_equal(d2._xw_sum, [0., 0., 0.])


def test_from_summaries_weighted_inertia(X, w):
	d = [Exponential([2.1, 0.3, 0.1]), Exponential([1.5, 3.1, 2.2])]
	model = DenseHMM(distributions=d, edges=[[0.1, 0.8], [0.3, 0.6]], 
		starts=[0.2, 0.8], ends=[0.1, 0.1], inertia=0.3)

	d1 = model.distributions[0]
	d2 = model.distributions[1]

	model.summarize(X, sample_weight=w)
	model.from_summaries()

	assert_array_almost_equal(model.starts, [-2.713066, -0.096492])
	assert_array_almost_equal(model.ends, [-1.916754, -1.777675])
	assert_array_almost_equal(model.edges, 
		[[-7.574243, -0.200404], [-0.953491, -0.869844]], 4)

	assert_array_almost_equal(d1.scales, 
		[1.749434e+00, 1.025649e+00, 1.368553e-03], 3)
	assert_array_almost_equal(d1._w_sum, [0., 0., 0.])
	assert_array_almost_equal(d1._xw_sum, [0., 0., 0.])

	assert_array_almost_equal(d2.scales, [1.647633, 1.076022, 1.842134])
	assert_array_almost_equal(d2._w_sum, [0., 0., 0.])
	assert_array_almost_equal(d2._xw_sum, [0., 0., 0.])


	d = [Exponential([2.1, 0.3, 0.1], inertia=0.25), 
	     Exponential([1.5, 3.1, 2.2], inertia=0.83)]
	model = DenseHMM(distributions=d, edges=[[0.1, 0.8], [0.3, 0.6]], 
		starts=[0.2, 0.8], ends=[0.1, 0.1], inertia=0.0)

	d1 = model.distributions[0]
	d2 = model.distributions[1]

	model.summarize(X, sample_weight=w)
	model.from_summaries()

	assert_array_almost_equal(model.starts, [-3.186049, -0.042213])
	assert_array_almost_equal(model.ends, [-1.7514  , -1.552714])
	assert_array_almost_equal(model.edges, 
		[[-9.833528, -0.190657], [-0.846141, -1.023709]], 4)

	assert_array_almost_equal(d1.scales, [1.837075, 0.844237, 0.026026])
	assert_array_almost_equal(d1._w_sum, [0., 0., 0.])
	assert_array_almost_equal(d1._xw_sum, [0., 0., 0.])

	assert_array_almost_equal(d2.scales, [1.525098, 2.755924, 2.139163])
	assert_array_almost_equal(d2._w_sum, [0., 0., 0.])
	assert_array_almost_equal(d2._xw_sum, [0., 0., 0.])


def test_from_summaries_frozen(model, X):
	d = [Exponential([2.1, 0.3, 0.1]), Exponential([1.5, 3.1, 2.2])]
	model = DenseHMM(distributions=d, edges=[[0.1, 0.8], [0.3, 0.6]], 
		starts=[0.2, 0.8], ends=[0.1, 0.1], frozen=True)

	d1 = model.distributions[0]
	d2 = model.distributions[1]

	model.summarize(X)
	model.from_summaries()

	assert_array_almost_equal(model.starts, [-1.609438, -0.223144])
	assert_array_almost_equal(model.ends, [-2.302585, -2.302585])
	assert_array_almost_equal(model.edges, 
		[[-2.302585, -0.223144], [-1.203973, -0.510826]], 4)

	assert_array_almost_equal(d1.scales, [1.949585, 1.046725, 0.002491])
	assert_array_almost_equal(d1._w_sum, [0., 0., 0.])
	assert_array_almost_equal(d1._xw_sum, [0., 0., 0.])

	assert_array_almost_equal(d2.scales, [1.465946, 1.120429, 1.659203])
	assert_array_almost_equal(d2._w_sum, [0., 0., 0.])
	assert_array_almost_equal(d2._xw_sum, [0., 0., 0.])


	d = [Exponential([2.1, 0.3, 0.1], frozen=True), 
	     Exponential([1.5, 3.1, 2.2], frozen=True)]
	model = DenseHMM(distributions=d, edges=[[0.1, 0.8], [0.3, 0.6]], 
		starts=[0.2, 0.8], ends=[0.1, 0.1], inertia=0.0)

	d1 = model.distributions[0]
	d2 = model.distributions[1]

	model.summarize(X)
	model.from_summaries()

	assert_array_almost_equal(model.starts, [-2.685274, -0.07064])
	assert_array_almost_equal(model.ends, [-1.151575, -1.861335])
	assert_array_almost_equal(model.edges, 
		[[-9.248936, -0.38014 ], [-1.009071, -0.734016]], 4)

	assert_array_almost_equal(d1.scales, [2.1, 0.3, 0.1])
	assert_array_almost_equal(d1._w_sum, [0., 0., 0.])
	assert_array_almost_equal(d1._xw_sum, [0., 0., 0.])

	assert_array_almost_equal(d2.scales, [1.5, 3.1, 2.2])
	assert_array_almost_equal(d2._w_sum, [0., 0., 0.])
	assert_array_almost_equal(d2._xw_sum, [0., 0., 0.])


def test_fit(X):
	X = torch.tensor(numpy.array(X) + 1)

	d = [Exponential([2.1, 0.3, 0.1]), Exponential([1.5, 3.1, 2.2])]
	model = DenseHMM(distributions=d, edges=[[0.1, 0.8], [0.3, 0.6]], 
		starts=[0.2, 0.8], ends=[0.1, 0.1], max_iter=1)
	model.fit(X)
	
	d1 = model.distributions[0]
	d2 = model.distributions[1]

	assert_array_almost_equal(model.starts, [-1.489857e+01, -3.385568e-07], 4)
	assert_array_almost_equal(model.ends, [-1.110725, -1.609444])
	assert_array_almost_equal(model.edges, 
		[[-23.442368,  -0.399464], [-11.607552,  -0.223154]], 4)

	assert_array_almost_equal(d1.scales, [3.021216, 2.007029, 1.000361])
	assert_array_almost_equal(d1._w_sum, [0., 0., 0.])
	assert_array_almost_equal(d1._xw_sum, [0., 0., 0.])

	assert_array_almost_equal(d2.scales, [2.599996, 2.100001, 2.200011])
	assert_array_almost_equal(d2._w_sum, [0., 0., 0.])
	assert_array_almost_equal(d2._xw_sum, [0., 0., 0.])

	d = [Exponential([2.1, 0.3, 0.1]), Exponential([1.5, 3.1, 2.2])]
	model = DenseHMM(distributions=d, edges=[[0.1, 0.8], [0.3, 0.6]], 
		starts=[0.2, 0.8], ends=[0.1, 0.1], max_iter=5)
	model.fit(X)

	
	d1 = model.distributions[0]
	d2 = model.distributions[1]

	assert_array_almost_equal(model.starts, [-1.545504e+01, -1.940718e-07], 4)
	assert_array_almost_equal(model.ends, [-0.758036, -1.609449])
	assert_array_almost_equal(model.edges, 
		[[-23.906055,  -0.632214], [-11.732582,  -0.223151]], 4)

	assert_array_almost_equal(d1.scales, [2.603264, 2.076076, 1.532971])
	assert_array_almost_equal(d1._w_sum, [0., 0., 0.])
	assert_array_almost_equal(d1._xw_sum, [0., 0., 0.])

	assert_array_almost_equal(d2.scales, [2.6     , 2.1     , 2.200005])
	assert_array_almost_equal(d2._w_sum, [0., 0., 0.])
	assert_array_almost_equal(d2._xw_sum, [0., 0., 0.])


def test_fit_3ds(X):
	X = [torch.tensor(numpy.array(X) + 1)]

	d = [Exponential([2.1, 0.3, 0.1]), Exponential([1.5, 3.1, 2.2])]
	model = DenseHMM(distributions=d, edges=[[0.1, 0.8], [0.3, 0.6]], 
		starts=[0.2, 0.8], ends=[0.1, 0.1], max_iter=1)
	model.fit(X)
	
	d1 = model.distributions[0]
	d2 = model.distributions[1]

	assert_array_almost_equal(model.starts, [-1.489857e+01, -3.385568e-07], 4)
	assert_array_almost_equal(model.ends, [-1.110725, -1.609444])
	assert_array_almost_equal(model.edges, 
		[[-23.442368,  -0.399464], [-11.607552,  -0.223154]], 4)

	assert_array_almost_equal(d1.scales, [3.021216, 2.007029, 1.000361])
	assert_array_almost_equal(d1._w_sum, [0., 0., 0.])
	assert_array_almost_equal(d1._xw_sum, [0., 0., 0.])

	assert_array_almost_equal(d2.scales, [2.599996, 2.100001, 2.200011])
	assert_array_almost_equal(d2._w_sum, [0., 0., 0.])
	assert_array_almost_equal(d2._xw_sum, [0., 0., 0.])

	d = [Exponential([2.1, 0.3, 0.1]), Exponential([1.5, 3.1, 2.2])]
	model = DenseHMM(distributions=d, edges=[[0.1, 0.8], [0.3, 0.6]], 
		starts=[0.2, 0.8], ends=[0.1, 0.1], max_iter=5)
	model.fit(X)

	
	d1 = model.distributions[0]
	d2 = model.distributions[1]

	assert_array_almost_equal(model.starts, [-1.545504e+01, -1.940718e-07], 4)
	assert_array_almost_equal(model.ends, [-0.758036, -1.609449])
	assert_array_almost_equal(model.edges, 
		[[-23.906055,  -0.632214], [-11.732582,  -0.223151]], 4)

	assert_array_almost_equal(d1.scales, [2.603264, 2.076076, 1.532971])
	assert_array_almost_equal(d1._w_sum, [0., 0., 0.])
	assert_array_almost_equal(d1._xw_sum, [0., 0., 0.])

	assert_array_almost_equal(d2.scales, [2.6     , 2.1     , 2.200005])
	assert_array_almost_equal(d2._w_sum, [0., 0., 0.])
	assert_array_almost_equal(d2._xw_sum, [0., 0., 0.])


def test_fit_2ds(X):
	X = [x for x in torch.tensor(numpy.array(X) + 1)]

	d = [Exponential([2.1, 0.3, 0.1]), Exponential([1.5, 3.1, 2.2])]
	model = DenseHMM(distributions=d, edges=[[0.1, 0.8], [0.3, 0.6]], 
		starts=[0.2, 0.8], ends=[0.1, 0.1], max_iter=1)
	model.fit(X)
	
	d1 = model.distributions[0]
	d2 = model.distributions[1]

	assert_array_almost_equal(model.starts, [-1.489857e+01, -3.385568e-07], 4)
	assert_array_almost_equal(model.ends, [-1.110725, -1.609444])
	assert_array_almost_equal(model.edges, 
		[[-23.442368,  -0.399464], [-11.607552,  -0.223154]], 4)

	assert_array_almost_equal(d1.scales, [3.021216, 2.007029, 1.000361])
	assert_array_almost_equal(d1._w_sum, [0., 0., 0.])
	assert_array_almost_equal(d1._xw_sum, [0., 0., 0.])

	assert_array_almost_equal(d2.scales, [2.599996, 2.100001, 2.200011])
	assert_array_almost_equal(d2._w_sum, [0., 0., 0.])
	assert_array_almost_equal(d2._xw_sum, [0., 0., 0.])

	d = [Exponential([2.1, 0.3, 0.1]), Exponential([1.5, 3.1, 2.2])]
	model = DenseHMM(distributions=d, edges=[[0.1, 0.8], [0.3, 0.6]], 
		starts=[0.2, 0.8], ends=[0.1, 0.1], max_iter=5)
	model.fit(X)

	
	d1 = model.distributions[0]
	d2 = model.distributions[1]

	assert_array_almost_equal(model.starts, [-1.545504e+01, -1.940718e-07], 4)
	assert_array_almost_equal(model.ends, [-0.758036, -1.609449])
	assert_array_almost_equal(model.edges, 
		[[-23.906055,  -0.632214], [-11.732582,  -0.223151]], 4)

	assert_array_almost_equal(d1.scales, [2.603264, 2.076076, 1.532971])
	assert_array_almost_equal(d1._w_sum, [0., 0., 0.])
	assert_array_almost_equal(d1._xw_sum, [0., 0., 0.])

	assert_array_almost_equal(d2.scales, [2.6     , 2.1     , 2.200005])
	assert_array_almost_equal(d2._w_sum, [0., 0., 0.])
	assert_array_almost_equal(d2._xw_sum, [0., 0., 0.])


def test_fit_weighted(X, w):
	X = torch.tensor(numpy.array(X) + 1)

	d = [Exponential([2.1, 0.3, 0.1]), Exponential([1.5, 3.1, 2.2])]
	model = DenseHMM(distributions=d, edges=[[0.1, 0.8], [0.3, 0.6]], 
		starts=[0.2, 0.8], ends=[0.1, 0.1], max_iter=1)
	model.fit(X, sample_weight=w)
	
	d1 = model.distributions[0]
	d2 = model.distributions[1]

	assert_array_almost_equal(model.starts, [-1.5399e+01, -2.0519e-07], 3)
	assert_array_almost_equal(model.ends, [-1.732272, -1.609437])
	assert_array_almost_equal(model.edges, 
		[[-23.970318,  -0.194656], [-11.483337,  -0.223157]], 5)

	assert_array_almost_equal(d1.scales, [2.801925, 2.003776, 1.000194])
	assert_array_almost_equal(d1._w_sum, [0., 0., 0.])
	assert_array_almost_equal(d1._xw_sum, [0., 0., 0.])

	assert_array_almost_equal(d2.scales, [2.678787, 2.060607, 2.278801])
	assert_array_almost_equal(d2._w_sum, [0., 0., 0.])
	assert_array_almost_equal(d2._xw_sum, [0., 0., 0.])

	d = [Exponential([2.1, 0.3, 0.1]), Exponential([1.5, 3.1, 2.2])]
	model = DenseHMM(distributions=d, edges=[[0.1, 0.8], [0.3, 0.6]], 
		starts=[0.2, 0.8], ends=[0.1, 0.1], max_iter=5)
	model.fit(X, sample_weight=w)

	
	d1 = model.distributions[0]
	d2 = model.distributions[1]

	assert_array_almost_equal(model.starts, [-1.6093e+01, -1.0250e-07], 3)
	assert_array_almost_equal(model.ends, [-1.469704, -1.609439])
	assert_array_almost_equal(model.edges, 
		[[-24.481024,  -0.261356], [-11.632328,  -0.223154]], 5)

	assert_array_almost_equal(d1.scales, [2.324057, 2.012569, 1.522347])
	assert_array_almost_equal(d1._w_sum, [0., 0., 0.])
	assert_array_almost_equal(d1._xw_sum, [0., 0., 0.])

	assert_array_almost_equal(d2.scales, [2.678791, 2.060607, 2.278795])
	assert_array_almost_equal(d2._w_sum, [0., 0., 0.])
	assert_array_almost_equal(d2._xw_sum, [0., 0., 0.])


def test_fit_weighted_3ds(X, w):
	X = [torch.tensor(numpy.array(X) + 1)]

	d = [Exponential([2.1, 0.3, 0.1]), Exponential([1.5, 3.1, 2.2])]
	model = DenseHMM(distributions=d, edges=[[0.1, 0.8], [0.3, 0.6]], 
		starts=[0.2, 0.8], ends=[0.1, 0.1], max_iter=1)
	model.fit(X, sample_weight=[w])
	
	d1 = model.distributions[0]
	d2 = model.distributions[1]

	assert_array_almost_equal(model.starts, [-1.5399e+01, -2.0519e-07], 3)
	assert_array_almost_equal(model.ends, [-1.732272, -1.609437])
	assert_array_almost_equal(model.edges, 
		[[-23.970318,  -0.194656], [-11.483337,  -0.223157]], 5)

	assert_array_almost_equal(d1.scales, [2.801925, 2.003776, 1.000194])
	assert_array_almost_equal(d1._w_sum, [0., 0., 0.])
	assert_array_almost_equal(d1._xw_sum, [0., 0., 0.])

	assert_array_almost_equal(d2.scales, [2.678787, 2.060607, 2.278801])
	assert_array_almost_equal(d2._w_sum, [0., 0., 0.])
	assert_array_almost_equal(d2._xw_sum, [0., 0., 0.])

	d = [Exponential([2.1, 0.3, 0.1]), Exponential([1.5, 3.1, 2.2])]
	model = DenseHMM(distributions=d, edges=[[0.1, 0.8], [0.3, 0.6]], 
		starts=[0.2, 0.8], ends=[0.1, 0.1], max_iter=5)
	model.fit(X, sample_weight=[w])

	
	d1 = model.distributions[0]
	d2 = model.distributions[1]

	assert_array_almost_equal(model.starts, [-1.6093e+01, -1.0250e-07], 3)
	assert_array_almost_equal(model.ends, [-1.469704, -1.609439])
	assert_array_almost_equal(model.edges, 
		[[-24.481024,  -0.261356], [-11.632328,  -0.223154]], 5)

	assert_array_almost_equal(d1.scales, [2.324057, 2.012569, 1.522347])
	assert_array_almost_equal(d1._w_sum, [0., 0., 0.])
	assert_array_almost_equal(d1._xw_sum, [0., 0., 0.])

	assert_array_almost_equal(d2.scales, [2.678791, 2.060607, 2.278795])
	assert_array_almost_equal(d2._w_sum, [0., 0., 0.])
	assert_array_almost_equal(d2._xw_sum, [0., 0., 0.])


def test_fit_weighted_2ds(X, w):
	X = [x for x in torch.tensor(numpy.array(X) + 1)]

	d = [Exponential([2.1, 0.3, 0.1]), Exponential([1.5, 3.1, 2.2])]
	model = DenseHMM(distributions=d, edges=[[0.1, 0.8], [0.3, 0.6]], 
		starts=[0.2, 0.8], ends=[0.1, 0.1], max_iter=1)
	model.fit(X, sample_weight=w)
	
	d1 = model.distributions[0]
	d2 = model.distributions[1]

	assert_array_almost_equal(model.starts, [-1.5399e+01, -2.0519e-07], 3)
	assert_array_almost_equal(model.ends, [-1.732272, -1.609437])
	assert_array_almost_equal(model.edges, 
		[[-23.970318,  -0.194656], [-11.483337,  -0.223157]], 5)

	assert_array_almost_equal(d1.scales, [2.801925, 2.003776, 1.000194])
	assert_array_almost_equal(d1._w_sum, [0., 0., 0.])
	assert_array_almost_equal(d1._xw_sum, [0., 0., 0.])

	assert_array_almost_equal(d2.scales, [2.678787, 2.060607, 2.278801])
	assert_array_almost_equal(d2._w_sum, [0., 0., 0.])
	assert_array_almost_equal(d2._xw_sum, [0., 0., 0.])

	d = [Exponential([2.1, 0.3, 0.1]), Exponential([1.5, 3.1, 2.2])]
	model = DenseHMM(distributions=d, edges=[[0.1, 0.8], [0.3, 0.6]], 
		starts=[0.2, 0.8], ends=[0.1, 0.1], max_iter=5)
	model.fit(X, sample_weight=w)

	
	d1 = model.distributions[0]
	d2 = model.distributions[1]

	assert_array_almost_equal(model.starts, [-1.6093e+01, -1.0250e-07], 3)
	assert_array_almost_equal(model.ends, [-1.469704, -1.609439])
	assert_array_almost_equal(model.edges, 
		[[-24.481024,  -0.261356], [-11.632328,  -0.223154]], 5)

	assert_array_almost_equal(d1.scales, [2.324057, 2.012569, 1.522347])
	assert_array_almost_equal(d1._w_sum, [0., 0., 0.])
	assert_array_almost_equal(d1._xw_sum, [0., 0., 0.])

	assert_array_almost_equal(d2.scales, [2.678791, 2.060607, 2.278795])
	assert_array_almost_equal(d2._w_sum, [0., 0., 0.])
	assert_array_almost_equal(d2._xw_sum, [0., 0., 0.])


def test_fit_raises(model, X, w):
	assert_raises(ValueError, model.fit, [[X]])
	assert_raises(ValueError, model.fit, X[0])
	assert_raises((ValueError, TypeError), model.fit, X[0][0])
	assert_raises(ValueError, model.fit, 
		[[-0.1 for i in range(3)] for x in X])

	assert_raises(ValueError, model.fit, [X], w)
	assert_raises(ValueError, model.fit, X, [w])
	assert_raises(ValueError, model.fit, X[:len(X)-1], w)
	assert_raises(ValueError, model.fit, X, w[:len(w)-1])


###

def test_masked_emission_matrix(model, X, X_masked):
	X = torch.tensor(numpy.array(X))
	mask = torch.ones_like(X).type(torch.bool)
	X_ = torch.masked.MaskedTensor(X, mask=mask)

	e = model._emission_matrix(X_)
	assert_array_almost_equal(e, 
		[[[ -4.3782,  -3.6372],
          [ -7.2354,  -2.7799],
          [-21.0449,  -4.2237],
          [-24.8544,  -5.2129],
          [ -1.9973,  -4.6479]],

         [[-42.9497,  -7.7994],
          [ -1.5211,  -3.9812],
          [-17.7116,  -3.9011],
          [ -1.0449,  -3.3146],
          [-13.9020,  -3.4250]]], 4)

	e = model._emission_matrix(X_masked)
	assert_array_almost_equal(e, 
		[[[ -4.3782,  -3.6372],
          [ -0.7419,  -0.4055],
          [  0.0000,   0.0000],
          [-24.8544,  -5.2129],
          [ -1.9973,  -4.6479]],

         [[-42.9497,  -7.7994],
          [  0.1732,  -2.2424],
          [-18.9155,  -2.7697],
          [ -1.0449,  -3.3146],
          [  0.0000,   0.0000]]], 4)

###

def test_masked_log_probability(model, X, X_masked):
	X = torch.tensor(numpy.array(X))
	mask = torch.ones_like(X).type(torch.bool)
	X_ = torch.masked.MaskedTensor(X, mask=mask)

	logp = model.log_probability(X_)
	assert_array_almost_equal(logp, [-22.8266, -22.8068], 4)

	logp = model.log_probability(X_masked)
	assert_array_almost_equal(logp, [-15.463 , -16.3894], 4)


def test_masked_probability(model, X, X_masked):
	X = torch.tensor(numpy.array(X))
	mask = torch.ones_like(X).type(torch.bool)
	X_ = torch.masked.MaskedTensor(X, mask=mask)

	p = model.probability(X_)
	assert_array_almost_equal(p, [1.2205e-09, 1.2449e-09], 4)

	p = model.probability(X_masked)
	assert_array_almost_equal(p, [1.9253e-07, 7.6242e-08], 4)


def test_masked_forward(model, X, X_masked):
	X = torch.tensor(numpy.array(X))
	mask = torch.ones_like(X).type(torch.bool)
	X_ = torch.masked.MaskedTensor(X, mask=mask)

	f = model.forward(X_)
	assert_array_almost_equal(f,
		[[[ -5.9877,  -3.8603],
          [-12.2607,  -7.0036],
          [-29.2507, -11.7311],
          [-37.7895, -17.4549],
          [-20.6561, -22.6136]],

         [[-44.5591,  -8.0226],
          [-10.7476, -12.5146],
          [-30.3480, -14.7513],
          [-17.0002, -18.5767],
          [-32.7223, -20.5042]]], 4)

	f = model.forward(X_masked)
	assert_array_almost_equal(f,
		[[[ -5.9877,  -3.8603],
          [ -5.7673,  -4.6291],
          [ -5.7316,  -4.7842],
          [-30.7211, -10.0912],
          [-13.2925, -15.2500]],

         [[-44.5591,  -8.0226],
          [ -9.0533, -10.7758],
          [-29.8424, -11.9204],
          [-14.1693, -15.7458],
          [-15.9894, -14.2483]]], 4)



def test_masked_backward(model, X, X_masked):
	X = torch.tensor(numpy.array(X))
	mask = torch.ones_like(X).type(torch.bool)
	X_ = torch.masked.MaskedTensor(X, mask=mask)

	b = model.backward(X_)
	assert_array_almost_equal(b,
		[[[-18.8311, -19.1130],
          [-15.5423, -15.8300],
          [-10.8078, -11.0955],
          [ -6.1547,  -5.3717],
          [ -2.3026,  -2.3026]],

         [[-15.5896, -14.7842],
          [-12.1797, -12.4674],
          [ -8.8158,  -8.0555],
          [ -5.9508,  -6.2384],
          [ -2.3026,  -2.3026]]], 4)

	b = model.backward(X_masked)
	assert_array_almost_equal(b,
		[[[-11.6441, -11.7241],
          [-11.1645, -11.0955],
          [-10.8078, -11.0955],
          [ -6.1547,  -5.3717],
          [ -2.3026,  -2.3026]],

         [[ -9.1620,  -8.3668],
          [ -7.4618,  -7.7494],
          [ -5.1529,  -4.4689],
          [ -2.4079,  -2.4079],
          [ -2.3026,  -2.3026]]], 4)


def test_masked_forward_backward(model, X, X_masked):
	X = torch.tensor(numpy.array(X))
	mask = torch.ones_like(X).type(torch.bool)
	X_ = torch.masked.MaskedTensor(X, mask=mask)

	t, r, starts, ends, logp = model.forward_backward(X_)
	assert_array_almost_equal(t,
		[[[2.6353e-04, 1.4304e-01], 
		  [8.8289e-01, 2.9738e+00]],
         
         [[3.1500e-06, 1.7522e+00], 
          [1.7522e+00, 4.9559e-01]]], 3)

	assert_array_almost_equal(r,
		[[[-1.9921e+00, -1.4665e-01],
          [-4.9764e+00, -6.9228e-03],
          [-1.7232e+01, -3.2831e-08],
          [-2.1118e+01, -6.7415e-10],
          [-1.3209e-01, -2.0896e+00]],

         [[-3.7342e+01,  0.0000e+00],
          [-1.2056e-01, -2.1752e+00],
          [-1.6357e+01, -7.8752e-08],
          [-1.4412e-01, -2.0083e+00],
          [-1.2218e+01, -4.9402e-06]]], 3)

	assert_array_almost_equal(starts, 
		[[1.3641e-01, 8.6359e-01],
         [6.0619e-17, 1.0000e+00]], 3)

	assert_array_almost_equal(ends,
		[[8.7626e-01, 1.2374e-01],
         [4.9402e-06, 1.0000e+00]], 3)

	assert_array_almost_equal(logp, [-22.8266, -22.8068], 3)


	t, r, starts, ends, logp = model.forward_backward(X_masked)
	assert_array_almost_equal(t,
		[[[0.0417, 0.6437],
          [1.4056, 1.9091]],

         [[0.0921, 1.6185],
          [1.7677, 0.5218]]], 3)

	assert_array_almost_equal(r,
		[[[ -2.1687,  -0.1214],
          [ -1.4687,  -0.2616],
          [ -1.0765,  -0.4167],
          [-21.4128,   0.0000],
          [ -0.1321,  -2.0896]],

         [[-37.3318,   0.0000],
          [ -0.1257,  -2.1359],
          [-18.6059,   0.0000],
          [ -0.1879,  -1.7644],
          [ -1.9026,  -0.1615]]], 3)

	assert_array_almost_equal(starts, 
		[[1.1432e-01, 8.8568e-01],
         [6.1237e-17, 1.0000e+00]], 3)

	assert_array_almost_equal(ends,
		[[0.8763, 0.1237],
         [0.1492, 0.8508]], 3)

	assert_array_almost_equal(logp, [-15.4630, -16.3894], 3)


def test_masked_predict(model, X, X_masked):
	X = torch.tensor(numpy.array(X))
	mask = torch.ones_like(X).type(torch.bool)
	X_ = torch.masked.MaskedTensor(X, mask=mask)

	y_hat = model.predict(X_)
	assert_array_almost_equal(y_hat, 
		[[1, 1, 1, 1, 0],
         [1, 0, 1, 0, 1]], 4)

	y_hat = model.predict(X_masked)
	assert_array_almost_equal(y_hat, 
		[[1, 1, 1, 1, 0],
         [1, 0, 1, 0, 1]], 4)


def test_masked_predict_proba(model, X, X_masked):
	X = torch.tensor(numpy.array(X))
	mask = torch.ones_like(X).type(torch.bool)
	X_ = torch.masked.MaskedTensor(X, mask=mask)

	y_hat = model.predict_proba(X_)
	assert_array_almost_equal(y_hat,
		[[[1.3641e-01, 8.6359e-01],
          [6.8989e-03, 9.9310e-01],
          [3.2831e-08, 1.0000e+00],
          [6.7415e-10, 1.0000e+00],
          [8.7626e-01, 1.2374e-01]],

         [[6.0619e-17, 1.0000e+00],
          [8.8642e-01, 1.1358e-01],
          [7.8752e-08, 1.0000e+00],
          [8.6578e-01, 1.3422e-01],
          [4.9402e-06, 1.0000e+00]]], 4)

	assert_array_almost_equal(torch.sum(y_hat, dim=-1),
		[[1.0000, 1.0000, 1.0000, 1.0000, 1.0000],
         [1.0000, 1.0000, 1.0000, 1.0000, 1.0000]])


	y_hat = model.predict_proba(X_masked)
	assert_array_almost_equal(y_hat,
		[[[1.1432e-01, 8.8568e-01],
          [2.3021e-01, 7.6979e-01],
          [3.4080e-01, 6.5920e-01],
          [5.0183e-10, 1.0000e+00],
          [8.7626e-01, 1.2374e-01]],

         [[6.1237e-17, 1.0000e+00],
          [8.8186e-01, 1.1814e-01],
          [8.3094e-09, 1.0000e+00],
          [8.2871e-01, 1.7129e-01],
          [1.4918e-01, 8.5083e-01]]], 4)

	assert_array_almost_equal(torch.sum(y_hat, dim=-1),
		[[1.0000, 1.0000, 1.0000, 1.0000, 1.0000],
         [1.0000, 1.0000, 1.0000, 1.0000, 1.0000]])


def test_masked_predict_log_proba(model, X, X_masked):
	X = torch.tensor(numpy.array(X))
	mask = torch.ones_like(X).type(torch.bool)
	X_ = torch.masked.MaskedTensor(X, mask=mask)

	y_hat = model.predict_log_proba(X_)
	assert_array_almost_equal(y_hat,
		[[[-1.9921e+00, -1.4665e-01],
          [-4.9764e+00, -6.9228e-03],
          [-1.7232e+01, -3.2831e-08],
          [-2.1118e+01, -6.7415e-10],
          [-1.3209e-01, -2.0896e+00]],

         [[-3.7342e+01,  0.0000e+00],
          [-1.2056e-01, -2.1752e+00],
          [-1.6357e+01, -7.8752e-08],
          [-1.4412e-01, -2.0083e+00],
          [-1.2218e+01, -4.9402e-06]]], 3)

	y_hat = model.predict_log_proba(X_masked)
	assert_array_almost_equal(y_hat,
		[[[ -2.1687,  -0.1214],
          [ -1.4687,  -0.2616],
          [ -1.0765,  -0.4167],
          [-21.4128,   0.0000],
          [ -0.1321,  -2.0896]],

         [[-37.3318,   0.0000],
          [ -0.1257,  -2.1359],
          [-18.6059,   0.0000],
          [ -0.1879,  -1.7644],
          [ -1.9026,  -0.1615]]], 3)


def test_masked_ones_summarize(model, X, w):
	X = torch.tensor(numpy.array(X))
	mask = torch.ones_like(X).type(torch.bool)
	X_ = torch.masked.MaskedTensor(X, mask=mask)

	d1 = model.distributions[0]
	d2 = model.distributions[1]
	model.summarize(X_, sample_weight=w)

	assert_array_almost_equal(model._xw_sum, 
		[[2.707788e-04, 4.173112e+00], [4.912973e+00, 4.113652e+00]], 4)
	assert_array_almost_equal(model._xw_starts_sum, [0.136405, 3.163595], 4)
	assert_array_almost_equal(model._xw_ends_sum, [0.876271, 2.423729], 4)

	assert_array_almost_equal(d1._w_sum, [5.049643, 5.049643, 5.049643], 4)
	assert_array_almost_equal(d1._xw_sum, [8.834015e+00, 5.179160e+00, 
		6.910696e-03], 4)

	assert_array_almost_equal(d2._w_sum, [11.450353, 11.450353, 11.450353], 4)
	assert_array_almost_equal(d2._xw_sum, [18.865982, 12.320835, 21.093086], 4)	


def test_masked_summarize(model, X, X_masked, w):
	d1 = model.distributions[0]
	d2 = model.distributions[1]
	model.summarize(X_masked, sample_weight=w)

	assert_array_almost_equal(model._xw_sum, 
		[[0.2535, 4.3662], [5.4712, 3.1091]], 4)
	assert_array_almost_equal(model._xw_starts_sum, [0.1143, 3.1857], 4)
	assert_array_almost_equal(model._xw_ends_sum, [1.2194, 2.0806], 4)

	assert_array_almost_equal(d1._w_sum, [3.1268, 4.9249, 4.9249], 4)
	assert_array_almost_equal(d1._xw_sum, [4.6491e+00, 5.0392e+00, 3.9227e-08],
		4)

	assert_array_almost_equal(d2._w_sum, [7.7732, 4.9751, 7.2751], 4)
	assert_array_almost_equal(d2._xw_sum, [17.4509,  6.8608, 15.8000], 4)


def test_masked_ones_from_summaries(model, X, w):
	X = torch.tensor(numpy.array(X))
	mask = torch.ones_like(X).type(torch.bool)
	X_ = torch.masked.MaskedTensor(X, mask=mask)

	d1 = model.distributions[0]
	d2 = model.distributions[1]

	model.summarize(X_, sample_weight=w)
	model.from_summaries()

	assert_array_almost_equal(model.starts, [-3.186049, -0.042213])
	assert_array_almost_equal(model.ends, [-1.7514  , -1.552714])
	assert_array_almost_equal(model.edges, 
		[[-9.833528, -0.190657], [-0.846141, -1.023709]], 4)

	assert_array_almost_equal(d1.scales, 
		[1.749434e+00, 1.025649e+00, 1.368553e-03])
	assert_array_almost_equal(d1._w_sum, [0., 0., 0.])
	assert_array_almost_equal(d1._xw_sum, [0., 0., 0.])

	assert_array_almost_equal(d2.scales, [1.647633, 1.076022, 1.842134])
	assert_array_almost_equal(d2._w_sum, [0., 0., 0.])
	assert_array_almost_equal(d2._xw_sum, [0., 0., 0.])


def test_masked_from_summaries(model, X_masked, w):
	d1 = model.distributions[0]
	d2 = model.distributions[1]

	model.summarize(X_masked, sample_weight=w)
	model.from_summaries()

	assert_array_almost_equal(model.starts, [-3.3627, -0.0353], 4)
	assert_array_almost_equal(model.ends, [-1.5662, -1.6339], 4)
	assert_array_almost_equal(model.edges, 
		[[-3.1371, -0.2907], [-0.6671, -1.2323]], 4)

	assert_array_almost_equal(d1.scales, 
		[1.4869e+00, 1.0232e+00, 7.9650e-09], 4)
	assert_array_almost_equal(d1._w_sum, [0., 0., 0.])
	assert_array_almost_equal(d1._xw_sum, [0., 0., 0.])

	assert_array_almost_equal(d2.scales, [2.2450, 1.3790, 2.1718], 4)
	assert_array_almost_equal(d2._w_sum, [0., 0., 0.])
	assert_array_almost_equal(d2._xw_sum, [0., 0., 0.])


def test_masked_fit(X, X_masked):
	X = torch.tensor(numpy.array(X) + 1)
	mask = torch.ones_like(X).type(torch.bool)
	X_ = torch.masked.MaskedTensor(X, mask=mask)

	d = [Exponential([2.1, 0.3, 0.1]), Exponential([1.5, 3.1, 2.2])]
	model = DenseHMM(distributions=d, edges=[[0.1, 0.8], [0.3, 0.6]], 
		starts=[0.2, 0.8], ends=[0.1, 0.1], max_iter=5)
	model.fit(X_)

	
	d1 = model.distributions[0]
	d2 = model.distributions[1]

	assert_array_almost_equal(model.starts, [-1.545504e+01, -1.940718e-07], 4)
	assert_array_almost_equal(model.ends, [-0.758036, -1.609449])
	assert_array_almost_equal(model.edges, 
		[[-23.906055,  -0.632214], [-11.732582,  -0.223151]], 4)

	assert_array_almost_equal(d1.scales, [2.603264, 2.076076, 1.532971])
	assert_array_almost_equal(d1._w_sum, [0., 0., 0.])
	assert_array_almost_equal(d1._xw_sum, [0., 0., 0.])

	assert_array_almost_equal(d2.scales, [2.6     , 2.1     , 2.200005])
	assert_array_almost_equal(d2._w_sum, [0., 0., 0.])
	assert_array_almost_equal(d2._xw_sum, [0., 0., 0.])


	d = [Exponential([2.1, 0.3, 0.1]), Exponential([1.5, 3.1, 2.2])]
	model = DenseHMM(distributions=d, edges=[[0.1, 0.8], [0.3, 0.6]], 
		starts=[0.2, 0.8], ends=[0.1, 0.1], max_iter=5)
	model.fit(X_masked + 1)

	d1 = model.distributions[0]
	d2 = model.distributions[1]

	assert_array_almost_equal(model.starts, [-16.6664,   0.0000], 4)
	assert_array_almost_equal(model.ends, [-0.9247, -1.7014], 4)
	assert_array_almost_equal(model.edges, 
		[[-3.5068, -0.5563], [-2.4457, -0.3135]], 4)

	assert_array_almost_equal(d1.scales, [2.5729, 2.1118, 1.4393], 4)
	assert_array_almost_equal(d1._w_sum, [0., 0., 0.])
	assert_array_almost_equal(d1._xw_sum, [0., 0., 0.])

	assert_array_almost_equal(d2.scales, [2.8777, 2.3498, 2.1939], 4)
	assert_array_almost_equal(d2._w_sum, [0., 0., 0.])
	assert_array_almost_equal(d2._xw_sum, [0., 0., 0.])

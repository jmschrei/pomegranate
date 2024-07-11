# test_bayesian_network_structure_learning.py
# Contact: Jacob Schreiber <jmschreiber91@gmail.com>

import numpy
import torch
import pytest

from pomegranate.bayesian_network import BayesianNetwork

from pomegranate.bayesian_network import _from_structure
from pomegranate.bayesian_network import _learn_structure
from pomegranate.bayesian_network import _categorical_exact
from pomegranate.bayesian_network import _categorical_chow_liu

from pomegranate.distributions import Categorical
from pomegranate.distributions import ConditionalCategorical

from .tools import assert_raises
from numpy.testing import assert_array_almost_equal


def assert_tuple_equal(x, y):
	assert len(x) == len(y)

	for i in range(len(x)):
		assert len(x[i]) == len(y[i])

		if len(x[i]) > 0:
			for j in range(len(x[i])):
				assert x[i][j] == y[i][j]


@pytest.fixture
def X():
	return [[1, 2, 0, 0],
	     [0, 0, 1, 0],
	     [1, 1, 1, 0],
	     [1, 2, 1, 1],
	     [1, 1, 0, 1],
	     [0, 1, 0, 1],
	     [0, 1, 0, 0],
	     [1, 0, 1, 1],
	     [1, 1, 0, 1],
	     [0, 2, 1, 0],
	     [0, 0, 0, 1]]


@pytest.fixture
def X_masked(X):
	mask = torch.tensor(numpy.array([
		[False, True,  True,  True ],
		[True,  True,  False, False],
		[False, False, False, True ],
		[True,  True,  True,  True ],
		[False, True,  False, False],
		[True,  True,  True,  False],
		[False, False, False, True ],
		[True,  False, True,  True ],
		[True,  True,  True,  False],
		[True,  True,  True,  True ],
		[True,  False, True,  False]]))

	X = torch.tensor(numpy.array(X))
	return torch.masked.MaskedTensor(X, mask=mask)


@pytest.fixture
def w():
	return torch.tensor([1.0, 0.3, 1.0, 0.0, 0.0, 2.1, 3.1, 1.0, 1.2, 1.1, 2.0])


###


def test_categorical_chow_liu(X):
	structure = _categorical_chow_liu(X)
	assert_tuple_equal(structure, ((), (0,), (0,), (0,)))

	structure = _categorical_chow_liu(X, root=1)
	assert_tuple_equal(structure, ((1,), (), (1,), (1,)))

	structure = _categorical_chow_liu(X, root=2)
	assert_tuple_equal(structure, ((2,), (2,), (), (2,)))


def test_categorical_chow_liu_weighted(X, w):
	structure = _categorical_chow_liu(X, w)
	assert_tuple_equal(structure, ((), (0,), (0,), (0,)))

	structure = _categorical_chow_liu(X, w, root=1)
	assert_tuple_equal(structure, ((1,), (), (1,), (1,)))

	structure = _categorical_chow_liu(X, w, root=2)
	assert_tuple_equal(structure, ((2,), (2,), (), (2,)))


def test_categorical_chow_liu_large():
	numpy.random.seed(0)
	X = numpy.random.randint(3, size=(50, 10))

	structure = _categorical_chow_liu(X)
	assert_tuple_equal(structure, ((), (0,), (1,), (7,), (0,), (0,), (7,), (2,),
		(7,), (0,)))

	structure = _categorical_chow_liu(X, root=1)
	assert_tuple_equal(structure, ((1,), (), (1,), (7,), (0,), (0,), (7,), (2,), 
		(7,), (0,)))

	structure = _categorical_chow_liu(X, root=2)
	assert_tuple_equal(structure, ((1,), (2,), (), (7,), (0,), (0,), (7,), (2,), 
		(7,), (0,)))


def test_categorical_chow_liu_large_pseudocount():
	numpy.random.seed(0)
	X = numpy.random.randint(3, size=(50, 10))

	structure = _categorical_chow_liu(X, pseudocount=10)
	assert_tuple_equal(structure, ((), (2,), (4,), (2,), (0,), (3,), (3,), (2,), 
		(2,), (6,)))

	structure = _categorical_chow_liu(X, root=1, pseudocount=10)
	assert_tuple_equal(structure, ((4,), (), (1,), (2,), (2,), (3,), (3,), (2,), 
		(2,), (6,)))

	structure = _categorical_chow_liu(X, root=2, pseudocount=10)
	assert_tuple_equal(structure, ((4,), (2,), (), (2,), (2,), (3,), (3,), (2,),
		(2,), (6,)))


def test_categorical_chow_liu_raises(X, w):
	assert_raises(ValueError, _categorical_chow_liu, X, w, -1)
	assert_raises(TypeError, _categorical_chow_liu, X, w, [1.2, 1.1])
	assert_raises(ValueError, _categorical_chow_liu, X, w, None, -1)
	assert_raises(ValueError, _categorical_chow_liu, X, w, None, 0.3)
	assert_raises(ValueError, _categorical_chow_liu, X, w, None, [1, 2])

	w = torch.tensor(w)
	assert_raises(ValueError, _categorical_chow_liu, X, w.unsqueeze(0))
	assert_raises(ValueError, _categorical_chow_liu, X, w.unsqueeze(1))

	X = numpy.array(X)
	n, d = X.shape

	assert_raises(ValueError, _categorical_chow_liu, X + 0.3, w)
	assert_raises(ValueError, _categorical_chow_liu, X.reshape(1, n, d), w)
	assert_raises(ValueError, _categorical_chow_liu, X - 1, w)


###


def test_categorical_exact(X):
	structure = _categorical_exact(X)
	assert_tuple_equal(structure, ((), (0,), (0, 1), ()))

	structure = _categorical_exact(X, max_parents=1)
	assert_tuple_equal(structure, ((), (0,), (), ()))


def test_categorical_exact_weighted(X, w):
	structure = _categorical_exact(X, w)
	assert_tuple_equal(structure, ((), (0,), (0, 1), ()))

	structure = _categorical_exact(X, w, max_parents=1)
	assert_tuple_equal(structure, ((), (0,), (), ()))


def test_categorical_exact_exclude_parents(X):
	exclude_parents = ((), (2,), (), (1,))
	structure = _categorical_exact(X, exclude_parents=exclude_parents)
	assert_tuple_equal(structure, ((), (), (0,), (0, 2)))

	structure = _categorical_exact(X, exclude_parents=exclude_parents, 
		max_parents=1)
	assert_tuple_equal(structure, ((), (0,), (), ()))

	exclude_parents = ((), (2,), (), (0, 1))
	structure = _categorical_exact(X, exclude_parents=exclude_parents)
	assert_tuple_equal(structure, ((2, 3), (), (), (2,)))


def test_categorical_exact_large():
	numpy.random.seed(0)
	X = numpy.random.randint(3, size=(50, 8))

	structure = _categorical_exact(X)
	assert_tuple_equal(structure, ((), (0, 3), (0, 1, 3), (), (0, 1, 2, 3), (), 
		(), ()))

	structure = _categorical_exact(X, max_parents=1)
	assert_tuple_equal(structure, ((), (), (), (), (), (0,), (), ()))

	structure = _categorical_exact(X, max_parents=2)
	assert_tuple_equal(structure, ((), (), (0, 1), (), (), (), (), ()))

	structure = _categorical_exact(X, max_parents=3)
	assert_tuple_equal(structure, ((), (), (0, 1), (), (0, 1, 2), (), (), ()))

	structure = _categorical_exact(X, max_parents=4)
	assert_tuple_equal(structure, ((), (0, 3), (0, 1, 3), (), (0, 1, 2, 3), (), 
		(), ()))


def test_categorical_exact_large_pseudocount():
	numpy.random.seed(0)
	X = numpy.random.randint(3, size=(50, 8))

	structure = _categorical_exact(X, pseudocount=10)
	assert_tuple_equal(structure, ((), (), (), (), (), (), (), ()))

	structure = _categorical_exact(X, pseudocount=5)
	assert_tuple_equal(structure, ((), (), (), (), (), (), (), ()))

	structure = _categorical_exact(X, pseudocount=2)
	assert_tuple_equal(structure, ((), (), (), (), (), (), (), ()))

	structure = _categorical_exact(X, pseudocount=1)
	assert_tuple_equal(structure, ((), (), (), (), (), (), (), ()))

	structure = _categorical_exact(X, pseudocount=0.1)
	assert_tuple_equal(structure, ((), (), (), (), (), (), (), ()))

	structure = _categorical_exact(X, pseudocount=1e-8)
	assert_tuple_equal(structure, ((), (), (), (), (), (0,), (), ()))


###


def test_categorical_learn_structure_chow_liu():
	numpy.random.seed(0)
	X = numpy.random.randint(3, size=(50, 7))
	w = numpy.abs(numpy.random.randn(50))

	structure1 = _categorical_chow_liu(X)
	structure2 = _learn_structure(X, algorithm='chow-liu')
	assert_tuple_equal(structure1, structure2)

	structure1 = _categorical_chow_liu(X, root=1)
	structure2 = _learn_structure(X, root=1, algorithm='chow-liu')
	assert_tuple_equal(structure1, structure2)

	structure1 = _categorical_chow_liu(X, pseudocount=50)
	structure2 = _learn_structure(X, pseudocount=50, algorithm='chow-liu')
	assert_tuple_equal(structure1, structure2)

	structure1 = _categorical_chow_liu(X, w)
	structure2 = _learn_structure(X, w, algorithm='chow-liu')
	assert_tuple_equal(structure1, structure2)


def test_categorical_learn_structure_exact():
	numpy.random.seed(0)
	X = numpy.random.randint(3, size=(50, 7))
	w = numpy.abs(numpy.random.randn(50))

	structure1 = _categorical_exact(X)
	structure2 = _learn_structure(X, algorithm='exact')
	assert_tuple_equal(structure1, structure2)

	structure1 = _categorical_exact(X, max_parents=1)
	structure2 = _learn_structure(X, max_parents=1, algorithm='exact')
	assert_tuple_equal(structure1, structure2)

	structure1 = _categorical_exact(X, pseudocount=50)
	structure2 = _learn_structure(X, pseudocount=50, algorithm='exact')
	assert_tuple_equal(structure1, structure2)

	structure1 = _categorical_exact(X, w)
	structure2 = _learn_structure(X, w, algorithm='exact')
	assert_tuple_equal(structure1, structure2)


###


def test_categorical_from_structure(X):
	structure = ((), (0,), (1, 3), ())
	distributions = _from_structure(X, structure=structure)
	model = BayesianNetwork(distributions, structure=structure)

	assert isinstance(model.distributions[0], Categorical)
	assert isinstance(model.distributions[1], ConditionalCategorical)
	assert isinstance(model.distributions[2], ConditionalCategorical)
	assert isinstance(model.distributions[3], Categorical)

	p0 = model.distributions[0].probs
	p1 = model.distributions[1].probs[0]
	p2 = model.distributions[2].probs[0]
	p3 = model.distributions[3].probs

	assert_array_almost_equal(p0, [[0.454545, 0.545455]])
	assert_array_almost_equal(p1, [[0.4     , 0.4     , 0.2     ], 
		[0.166667, 0.5     , 0.333333]])
	assert_array_almost_equal(p2, [[[0.0000, 1.0000],
         [0.5000, 0.5000]],

        [[0.5000, 0.5000],
         [1.0000, 0.0000]],

        [[0.5000, 0.5000],
         [0.0000, 1.0000]]])
	assert_array_almost_equal(p3, [[0.4545, 0.5455]], 4)


def test_categorical_from_structure_weighted(X, w):
	structure = ((), (0,), (1, 3), ())
	distributions = _from_structure(X, sample_weight=w, structure=structure)
	model = BayesianNetwork(distributions, structure=structure)

	assert isinstance(model.distributions[0], Categorical)
	assert isinstance(model.distributions[1], ConditionalCategorical)
	assert isinstance(model.distributions[2], ConditionalCategorical)
	assert isinstance(model.distributions[3], Categorical)

	p0 = model.distributions[0].probs
	p1 = model.distributions[1].probs[0]
	p2 = model.distributions[2].probs[0]
	p3 = model.distributions[3].probs

	assert_array_almost_equal(p0, [[0.6719, 0.3281]], 4)
	assert_array_almost_equal(p1, [[0.2674, 0.6047, 0.1279], 
		[0.2381, 0.5238, 0.2381]], 4)
	assert_array_almost_equal(p2, [[[0.0000, 1.0000],
         [0.6667, 0.3333]],

        [[0.7561, 0.2439],
         [1.0000, 0.0000]],

        [[0.4762, 0.5238],
         [0.5000, 0.5000]]], 4)
	assert_array_almost_equal(p3, [[0.5078, 0.4922]], 4)


def test_categorical_from_structure_null(X):
	structure = ((), (), (), ())
	distributions = _from_structure(X, structure=structure)
	model = BayesianNetwork(distributions, structure=structure)

	assert isinstance(model.distributions[0], Categorical)
	assert isinstance(model.distributions[1], Categorical)
	assert isinstance(model.distributions[2], Categorical)
	assert isinstance(model.distributions[3], Categorical)

	p0 = model.distributions[0].probs
	p1 = model.distributions[1].probs
	p2 = model.distributions[2].probs
	p3 = model.distributions[3].probs

	assert_array_almost_equal(p0, [[0.454545, 0.545455]])
	assert_array_almost_equal(p1, [[0.272727, 0.454545, 0.272727]])
	assert_array_almost_equal(p2, [[0.5455, 0.4545]], 4)
	assert_array_almost_equal(p3, [[0.4545, 0.5455]], 4)

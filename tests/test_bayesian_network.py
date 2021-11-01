# test_bayes_net.py
# Authors: Jacob Schreiber <jmschreiber91@gmail.com>
#          Nelson Liu <nelson@nelsonliu.me>
#
'''
These are unit tests for the Bayesian network part of pomegranate.
'''

from __future__ import division

from pomegranate import from_json
from pomegranate import DiscreteDistribution
from pomegranate import ConditionalProbabilityTable
from pomegranate import State, Node
from pomegranate import BayesianNetwork
from pomegranate.BayesianNetwork import _check_input
from pomegranate.io import DataGenerator
from pomegranate.io import DataFrameGenerator

from nose.tools import with_setup
from nose.tools import assert_equal
from nose.tools import assert_not_equal
from nose.tools import assert_raises
from nose.tools import assert_almost_equal

from networkx import DiGraph

from numpy.testing import assert_array_equal
from numpy.testing import assert_array_almost_equal

import pandas
import random, numpy
import sys

nan = numpy.nan
numpy.random.seed(1)

datasets = [numpy.random.randint(2, size=(10, 4)),
            numpy.random.randint(2, size=(100, 5)),
            numpy.random.randint(2, size=(1000, 7)),
            numpy.random.randint(2, size=(100, 9))]

datasets_nan = []
for dataset in datasets:
    X = dataset.copy().astype('float64')
    n, d = X.shape

    idx = numpy.random.choice(n*d, replace=False, size=n*d//5)
    i, j = idx // d, idx % d
    X[i, j] = numpy.nan

    datasets_nan.append(X)

def setup_monty():
    # Build a model of the Monty Hall Problem
    global monty_network, monty_index, prize_index, guest_index

    random.seed(0)

    # Friends emissions are completely random
    guest = DiscreteDistribution({'A': 1. / 3, 'B': 1. / 3, 'C': 1. / 3})

    # The actual prize is independent of the other distributions
    prize = DiscreteDistribution({'A': 1. / 3, 'B': 1. / 3, 'C': 1. / 3})
    # Monty is dependent on both the guest and the prize.
    monty = ConditionalProbabilityTable(
        [['A', 'A', 'A', 0.0],
         ['A', 'A', 'B', 0.5],
         ['A', 'A', 'C', 0.5],
         ['A', 'B', 'A', 0.0],
         ['A', 'B', 'B', 0.0],
         ['A', 'B', 'C', 1.0],
         ['A', 'C', 'A', 0.0],
         ['A', 'C', 'B', 1.0],
         ['A', 'C', 'C', 0.0],
         ['B', 'A', 'A', 0.0],
         ['B', 'A', 'B', 0.0],
         ['B', 'A', 'C', 1.0],
         ['B', 'B', 'A', 0.5],
         ['B', 'B', 'B', 0.0],
         ['B', 'B', 'C', 0.5],
         ['B', 'C', 'A', 1.0],
         ['B', 'C', 'B', 0.0],
         ['B', 'C', 'C', 0.0],
         ['C', 'A', 'A', 0.0],
         ['C', 'A', 'B', 1.0],
         ['C', 'A', 'C', 0.0],
         ['C', 'B', 'A', 1.0],
         ['C', 'B', 'B', 0.0],
         ['C', 'B', 'C', 0.0],
         ['C', 'C', 'A', 0.5],
         ['C', 'C', 'B', 0.5],
         ['C', 'C', 'C', 0.0]], [guest, prize])

    # Make the states
    s1 = State(guest, name="guest")
    s2 = State(prize, name="prize")
    s3 = State(monty, name="monty")

    # Make the bayes net, add the states, and the conditional dependencies.
    monty_network = BayesianNetwork("test")
    monty_network.add_nodes(s1, s2, s3)
    monty_network.add_edge(s1, s3)
    monty_network.add_edge(s2, s3)
    monty_network.bake()

    monty_index = monty_network.states.index(s3)
    prize_index = monty_network.states.index(s2)
    guest_index = monty_network.states.index(s1)


def setup_titanic():
    # Build a model of the titanic disaster
    global titanic_network, passenger, gender, tclass

    # Passengers on the Titanic either survive or perish
    passenger = DiscreteDistribution({'survive': 0.6, 'perish': 0.4})

    # Gender, given survival data
    gender = ConditionalProbabilityTable(
        [['survive', 'male',   0.0],
         ['survive', 'female', 1.0],
         ['perish', 'male',    1.0],
         ['perish', 'female',  0.0]], [passenger])

    # Class of travel, given survival data
    tclass = ConditionalProbabilityTable(
        [['survive', 'first',  0.0],
         ['survive', 'second', 1.0],
         ['survive', 'third',  0.0],
         ['perish', 'first',  1.0],
         ['perish', 'second', 0.0],
         ['perish', 'third',  0.0]], [passenger])

    # State objects hold both the distribution, and a high level name.
    s1 = State(passenger, name="passenger")
    s2 = State(gender, name="gender")
    s3 = State(tclass, name="class")

    # Create the Bayesian network object with a useful name
    titanic_network = BayesianNetwork("Titanic Disaster")

    # Add the three nodes to the network
    titanic_network.add_nodes(s1, s2, s3)

    # Add transitions which represent conditional dependencies, where the
    # second node is conditionally dependent on the first node (Monty is
    # dependent on both guest and prize)
    titanic_network.add_edge(s1, s2)
    titanic_network.add_edge(s1, s3)
    titanic_network.bake()


def setup_large_monty():
    # Build the huge monty hall large_monty_network. This is an example I made
    # up with which may not exactly flow logically, but tests a varied type of
    # tables ensures heterogeneous types of data work together.
    global large_monty_network, large_monty_friend, large_monty_guest, large_monty
    global large_monty_remaining, large_monty_randomize, large_monty_prize

    # large_monty_Friend
    large_monty_friend = DiscreteDistribution({True: 0.5, False: 0.5})

    # large_monty_Guest emissions are completely random
    large_monty_guest = ConditionalProbabilityTable(
        [[True, 'A', 0.50],
         [True, 'B', 0.25],
         [True, 'C', 0.25],
         [False, 'A', 0.0],
         [False, 'B', 0.7],
         [False, 'C', 0.3]], [large_monty_friend])

    # Number of large_monty_remaining cars
    large_monty_remaining = DiscreteDistribution({0: 0.1, 1: 0.7, 2: 0.2})

    # Whether they large_monty_randomize is dependent on the number of
    # large_monty_remaining cars
    large_monty_randomize = ConditionalProbabilityTable(
        [[0, True, 0.05],
         [0, False, 0.95],
         [1, True, 0.8],
         [1, False, 0.2],
         [2, True, 0.5],
         [2, False, 0.5]], [large_monty_remaining])

    # Where the large_monty_prize is depends on if they large_monty_randomize or
    # not and also the large_monty_guests large_monty_friend
    large_monty_prize = ConditionalProbabilityTable(
        [[True, True, 'A', 0.3],
         [True, True, 'B', 0.4],
         [True, True, 'C', 0.3],
         [True, False, 'A', 0.2],
         [True, False, 'B', 0.4],
         [True, False, 'C', 0.4],
         [False, True, 'A', 0.1],
         [False, True, 'B', 0.9],
         [False, True, 'C', 0.0],
         [False, False, 'A', 0.0],
         [False, False, 'B', 0.4],
         [False, False, 'C', 0.6]], [large_monty_randomize, large_monty_friend])

    # Monty is dependent on both the large_monty_guest and the large_monty_prize.
    large_monty = ConditionalProbabilityTable(
        [['A', 'A', 'A', 0.0],
         ['A', 'A', 'B', 0.5],
         ['A', 'A', 'C', 0.5],
         ['A', 'B', 'A', 0.0],
         ['A', 'B', 'B', 0.0],
         ['A', 'B', 'C', 1.0],
         ['A', 'C', 'A', 0.0],
         ['A', 'C', 'B', 1.0],
         ['A', 'C', 'C', 0.0],
         ['B', 'A', 'A', 0.0],
         ['B', 'A', 'B', 0.0],
         ['B', 'A', 'C', 1.0],
         ['B', 'B', 'A', 0.5],
         ['B', 'B', 'B', 0.0],
         ['B', 'B', 'C', 0.5],
         ['B', 'C', 'A', 1.0],
         ['B', 'C', 'B', 0.0],
         ['B', 'C', 'C', 0.0],
         ['C', 'A', 'A', 0.0],
         ['C', 'A', 'B', 1.0],
         ['C', 'A', 'C', 0.0],
         ['C', 'B', 'A', 1.0],
         ['C', 'B', 'B', 0.0],
         ['C', 'B', 'C', 0.0],
         ['C', 'C', 'A', 0.5],
         ['C', 'C', 'B', 0.5],
         ['C', 'C', 'C', 0.0]], [large_monty_guest, large_monty_prize])

    # Make the states
    s0 = State(large_monty_friend, name="large_monty_friend")
    s1 = State(large_monty_guest, name="large_monty_guest")
    s2 = State(large_monty_prize, name="large_monty_prize")
    s3 = State(large_monty, name="large_monty")
    s4 = State(large_monty_remaining, name="large_monty_remaining")
    s5 = State(large_monty_randomize, name="large_monty_randomize")

    # Make the bayes net, add the states, and the conditional dependencies.
    large_monty_network = BayesianNetwork("test")
    large_monty_network.add_nodes(s0, s1, s2, s3, s4, s5)
    large_monty_network.add_transition(s0, s1)
    large_monty_network.add_transition(s1, s3)
    large_monty_network.add_transition(s2, s3)
    large_monty_network.add_transition(s4, s5)
    large_monty_network.add_transition(s5, s2)
    large_monty_network.add_transition(s0, s2)
    large_monty_network.bake()

def setup_random_mixed():
    numpy.random.seed(0)
    global X
    X = numpy.array([
        numpy.random.choice([True, False], size=50),
        numpy.random.choice(['A', 'B'], size=50),
        numpy.random.choice(2, size=50)
    ], dtype=object).T.copy()

    global weights
    weights = numpy.abs(numpy.random.randn(50))

    global data_generator
    data_generator = DataGenerator(X, weights)

    global model
    model = BayesianNetwork.from_samples(X)

def teardown():
    pass


@with_setup(setup_monty, teardown)
def test_check_input_dict():
    obs = {'guest' : 'A'}
    _check_input(obs, monty_network)

    obs = {'guest' : 'NaN'}
    assert_raises(ValueError, _check_input, obs, monty_network)

    obs = {'guest' : None}
    assert_raises(ValueError, _check_input, obs, monty_network)

    obs = {'guest' : numpy.nan}
    assert_raises(ValueError, _check_input, obs, monty_network)

    obs = {'guest' : 'NaN', 'prize' : 'B'}
    assert_raises(ValueError, _check_input, obs, monty_network)

    obs = {'guest' : 'A', 'prize' : 'C'}
    _check_input(obs, monty_network)

    obs = {'guest' : 'A', 'prize' : 'C', 'monty' : 'C'}
    _check_input(obs, monty_network)

    obs = {'guest' : DiscreteDistribution({'A' : 0.25,
        'B' : 0.25, 'C' : 0.50})}
    _check_input(obs, monty_network)

    obs = {'hello' : 'A', 'prize' : 'B'}
    assert_raises(ValueError, _check_input, obs, monty_network)


@with_setup(setup_monty, teardown)
def test_check_input_list_of_dicts():
    obs = {'guest' : 'A'}
    _check_input([obs], monty_network)

    obs = {'guest' : 'NaN'}
    assert_raises(ValueError, _check_input, [obs], monty_network)

    obs = {'guest' : None}
    assert_raises(ValueError, _check_input, [obs], monty_network)

    obs = {'guest' : numpy.nan}
    assert_raises(ValueError, _check_input, [obs], monty_network)

    obs = {'guest' : 'NaN', 'prize' : 'B'}
    assert_raises(ValueError, _check_input, [obs], monty_network)

    obs = {'guest' : 'A', 'prize' : 'C'}
    _check_input([obs], monty_network)

    obs = {'guest' : 'A', 'prize' : 'C', 'monty' : 'C'}
    _check_input([obs], monty_network)

    obs = {'guest' : DiscreteDistribution({'A' : 0.25,
        'B' : 0.25, 'C' : 0.50})}
    _check_input([obs], monty_network)

    obs = {'hello' : 'A', 'prize' : 'B'}
    assert_raises(ValueError, _check_input, [obs], monty_network)

    obs = [{'guest' : 'A'}, {'guest' : 'A', 'prize' : 'C'},
        {'guest' : 'A', 'prize' : 'C', 'monty' : 'C'},
        {'guest' : DiscreteDistribution({'A' : 0.25,
            'B' : 0.25, 'C' : 0.50})}]
    _check_input(obs, monty_network)

    obs.append({'guest' : 'NaN', 'prize' : 'B'})
    assert_raises(ValueError, _check_input, obs, monty_network)


@with_setup(setup_monty, teardown)
def test_check_input_list_of_lists():
    obs = ['A', None, None]
    _check_input([obs], monty_network)

    obs = ['A', numpy.nan, numpy.nan]
    _check_input([obs], monty_network)

    obs = numpy.array(['A', None, None])
    _check_input([obs], monty_network)

    obs = numpy.array(['A', numpy.nan, numpy.nan])
    _check_input([obs], monty_network)

    obs = numpy.array(['A', 'B', 'C'])
    _check_input([obs], monty_network)

    obs = numpy.array(['NaN', numpy.nan, numpy.nan])
    assert_raises(ValueError, _check_input, [obs], monty_network)

    obs = numpy.array(['A', 'B', 'D'])
    assert_raises(ValueError, _check_input, [obs], monty_network)

    obs = ['A']
    assert_raises(ValueError, _check_input, [obs], monty_network)

    obs = ['A', 'C', 'E', 'F']
    assert_raises(ValueError, _check_input, [obs], monty_network)

    d = DiscreteDistribution({'A': 0.25, 'B': 0.25, 'C': 0.25})
    obs = [d, None, None]
    _check_input([obs], monty_network)

    e = DiscreteDistribution({'A': 0.25, 'B': 0.25, 'D': 0.25})
    obs = [e, None, None]
    assert_raises(ValueError, _check_input, [obs], monty_network)

    obs = [['A', None, None], ['A', numpy.nan, numpy.nan], ['A', 'B', 'C'],
        ['A', None, 'C'], [None, 'B', 'C'], [d, None, None]]
    _check_input(obs, monty_network)

    obs.append([e, None, None])
    assert_raises(ValueError, _check_input, obs, monty_network)


@with_setup(setup_titanic, teardown)
def test_titanic_network():
    assert_almost_equal(passenger.log_probability('survive'), numpy.log(0.6))
    assert_almost_equal(passenger.log_probability('survive'), numpy.log(0.6))

    assert_almost_equal(gender.log_probability(('survive', 'male')),   float("-inf"))
    assert_almost_equal(gender.log_probability(('survive', 'female')), 0.0)
    assert_almost_equal(gender.log_probability(('perish', 'male')),    0.0)
    assert_almost_equal(gender.log_probability(('perish', 'female')),  float("-inf"))

    assert_almost_equal(tclass.log_probability(('survive', 'first')), float("-inf"))
    assert_almost_equal(tclass.log_probability(('survive', 'second')), 0.0)
    assert_almost_equal(tclass.log_probability(('survive', 'third')), float("-inf"))
    assert_almost_equal(tclass.log_probability(('perish', 'first')), 0.0)
    assert_almost_equal(tclass.log_probability(('perish', 'second')), float("-inf"))
    assert_almost_equal(tclass.log_probability(('perish', 'third')), float("-inf"))


@with_setup(setup_titanic, teardown)
def test_guest_titanic():
    male = titanic_network.predict_proba({'gender': 'male'})
    female = titanic_network.predict_proba({'gender': 'female'})

    assert_equal(female[0].log_probability("survive"), 0.0)
    assert_equal(female[0].log_probability("perish"), float("-inf"))

    assert_equal(female[1], 'female')
    assert_equal(female[2].log_probability("first"), float("-inf"))
    assert_equal(female[2].log_probability("second"), 0.0)
    assert_equal(female[2].log_probability("third"), float("-inf"))

    assert_equal(male[0].log_probability("survive"), float("-inf"))
    assert_equal(male[0].log_probability("perish"), 0.0)

    assert_equal(male[1], 'male')

    assert_equal(male[2].log_probability("first"), 0.0)
    assert_equal(male[2].log_probability("second"), float("-inf"))
    assert_equal(male[2].log_probability("third"), float("-inf"))

    titanic_network2 = BayesianNetwork.from_json(titanic_network.to_json())


@with_setup(setup_large_monty, teardown)
def test_large_monty():
    assert_almost_equal(large_monty.log_probability(('A', 'A', 'C')), numpy.log(0.5))
    assert_almost_equal(large_monty.log_probability(('B', 'B', 'C')), numpy.log(0.5))
    assert_equal(large_monty.log_probability(('C', 'C', 'C')), float("-inf"))

    data = [[True,  'A', 'A', 'C', 1, True],
            [True,  'A', 'A', 'C', 0, True],
            [False, 'A', 'A', 'B', 1, False],
            [False, 'A', 'A', 'A', 2, False],
            [False, 'A', 'A', 'C', 1, False],
            [False, 'B', 'B', 'B', 2, False],
            [False, 'B', 'B', 'C', 0, False],
            [True,  'C', 'C', 'A', 2, True],
            [True,  'C', 'C', 'C', 1, False],
            [True,  'C', 'C', 'C', 0, False],
            [True,  'C', 'C', 'C', 2, True],
            [True,  'C', 'B', 'A', 1, False]]

    large_monty_network.fit(data)

    assert_almost_equal(large_monty.log_probability(('A', 'A', 'C')), numpy.log(0.6))
    assert_almost_equal(large_monty.log_probability(('B', 'B', 'C')), numpy.log(0.5))
    assert_almost_equal(large_monty.log_probability(('C', 'C', 'C')), numpy.log(0.75))


@with_setup(setup_large_monty, teardown)
def test_large_monty_friend():
    assert_almost_equal(large_monty_friend.log_probability(True), numpy.log(0.5))
    assert_almost_equal(large_monty_friend.log_probability(False), numpy.log(0.5))

    data = [[True,  'A', 'A', 'C', 1, True],
            [True,  'A', 'A', 'C', 0, True],
            [False, 'A', 'A', 'B', 1, False],
            [False, 'A', 'A', 'A', 2, False],
            [False, 'A', 'A', 'C', 1, False],
            [False, 'B', 'B', 'B', 2, False],
            [False, 'B', 'B', 'C', 0, False],
            [True,  'C', 'C', 'A', 2, True],
            [True,  'C', 'C', 'C', 1, False],
            [True,  'C', 'C', 'C', 0, False],
            [True,  'C', 'C', 'C', 2, True],
            [True,  'C', 'B', 'A', 1, False]]

    large_monty_network.fit(data)

    assert_almost_equal(large_monty_friend.log_probability(True), numpy.log(7. / 12))
    assert_almost_equal(large_monty_friend.log_probability(False), numpy.log(5. / 12))


@with_setup(setup_large_monty, teardown)
def test_large_monty_remaining():
    model = large_monty_remaining

    assert_almost_equal(model.log_probability(0), numpy.log(0.1))
    assert_almost_equal(model.log_probability(1), numpy.log(0.7))
    assert_almost_equal(model.log_probability(2), numpy.log(0.2))

    data = [[True,  'A', 'A', 'C', 1, True],
            [True,  'A', 'A', 'C', 0, True],
            [False, 'A', 'A', 'B', 1, False],
            [False, 'A', 'A', 'A', 2, False],
            [False, 'A', 'A', 'C', 1, False],
            [False, 'B', 'B', 'B', 2, False],
            [False, 'B', 'B', 'C', 0, False],
            [True,  'C', 'C', 'A', 2, True],
            [True,  'C', 'C', 'C', 1, False],
            [True,  'C', 'C', 'C', 0, False],
            [True,  'C', 'C', 'C', 2, True],
            [True,  'C', 'B', 'A', 1, False]]

    large_monty_network.fit(data)

    assert_almost_equal(model.log_probability(0), numpy.log(3. / 12))
    assert_almost_equal(model.log_probability(1), numpy.log(5. / 12))
    assert_almost_equal(model.log_probability(2), numpy.log(4. / 12))

@with_setup(setup_large_monty, teardown)
def test_large_monty_network_log_probability():
    model = large_monty_network

    data = numpy.array([[True,  'A', 'A', 'C', 1, True],
            [True,  'A', 'A', 'C', 0, True],
            [False, 'A', 'A', 'B', 1, False],
            [False, 'A', 'A', 'A', 2, False],
            [False, 'A', 'A', 'C', 1, False],
            [False, 'B', 'B', 'B', 2, False],
            [False, 'B', 'B', 'C', 0, False],
            [True,  'C', 'C', 'A', 2, True],
            [True,  'C', 'C', 'C', 1, False],
            [True,  'C', 'C', 'C', 0, False],
            [True,  'C', 'C', 'C', 2, True],
            [True,  'C', 'B', 'A', 1, False]],
            dtype=object)

    logp = [-3.863233, -8.581732, float("-inf"), float("-inf"), float("-inf"),
        float("-inf"), -5.013138, -6.279147, float("-inf"), float("-inf"),
        float("-inf"), -4.150915]
    logp1 = model.log_probability(data)

    assert_array_almost_equal(logp, logp1)

    model.fit(data)

    logp = [-5.480639, -5.480639, -4.60517 , -5.298317, -3.506558, -5.192957,
       -5.192957, -4.74667 , -2.667228, -3.360375, -3.648057, -3.072693]
    logp2 = model.log_probability(data)

    assert_array_almost_equal(logp2, logp)

@with_setup(setup_large_monty, teardown)
def test_large_monty_network_log_probability_parallel():
    model = large_monty_network

    data = numpy.array([[True,  'A', 'A', 'C', 1, True],
            [True,  'A', 'A', 'C', 0, True],
            [False, 'A', 'A', 'B', 1, False],
            [False, 'A', 'A', 'A', 2, False],
            [False, 'A', 'A', 'C', 1, False],
            [False, 'B', 'B', 'B', 2, False],
            [False, 'B', 'B', 'C', 0, False],
            [True,  'C', 'C', 'A', 2, True],
            [True,  'C', 'C', 'C', 1, False],
            [True,  'C', 'C', 'C', 0, False],
            [True,  'C', 'C', 'C', 2, True],
            [True,  'C', 'B', 'A', 1, False]], dtype=object)

    logp = [-3.863233, -8.581732, float("-inf"), float("-inf"), float("-inf"),
        float("-inf"), -5.013138, -6.279147, float("-inf"), float("-inf"),
        float("-inf"), -4.150915]

    logp1 = model.log_probability(data, n_jobs=2)

    assert_array_almost_equal(logp, logp1)

    model.fit(data)

    logp = [-5.480639, -5.480639, -4.60517 , -5.298317, -3.506558, -5.192957,
       -5.192957, -4.74667 , -2.667228, -3.360375, -3.648057, -3.072693]
    logp2 = model.log_probability(data, n_jobs=2)

    assert_array_almost_equal(logp2, logp)

@with_setup(setup_large_monty, teardown)
def test_large_monty_prize():
    assert_almost_equal(large_monty_prize.log_probability(
        (True,  True,  'A')), numpy.log(0.3))
    assert_almost_equal(large_monty_prize.log_probability(
        (True,  False, 'C')), numpy.log(0.4))
    assert_almost_equal(large_monty_prize.log_probability(
        (False, True,  'B')), numpy.log(0.9))
    assert_almost_equal(large_monty_prize.log_probability(
        (False, False, 'A')), float("-inf"))

    data = [[True,  'A', 'A', 'C', 1, True],
            [True,  'A', 'A', 'C', 0, True],
            [False, 'A', 'A', 'B', 1, False],
            [False, 'A', 'A', 'A', 2, False],
            [False, 'A', 'A', 'C', 1, False],
            [False, 'B', 'B', 'B', 2, False],
            [False, 'B', 'B', 'C', 0, False],
            [True,  'C', 'C', 'A', 2, True],
            [True,  'C', 'C', 'C', 1, False],
            [True,  'C', 'C', 'C', 0, False],
            [True,  'C', 'C', 'C', 2, True],
            [True,  'C', 'B', 'A', 1, False]]

    large_monty_network.fit(data)

    assert_almost_equal(large_monty_prize.log_probability(
        (True, True, 'C')), numpy.log(0.5))
    assert_equal(large_monty_prize.log_probability(
        (True, True, 'B')), float("-inf"))

    a = large_monty_prize.log_probability((True, False, 'A'))
    b = large_monty_prize.log_probability((True, False, 'B'))
    c = large_monty_prize.log_probability((True, False, 'C'))

    assert_almost_equal(a, b)
    assert_almost_equal(b, c)

    assert_equal(large_monty_prize.log_probability(
        (False, False, 'C')), float("-inf"))
    assert_almost_equal(large_monty_prize.log_probability(
        (False, True, 'C')), numpy.log(2. / 3))


def assert_discrete_equal(x, y, z=8):
    xd, yd = x.parameters[0], y.parameters[0]
    for key, value in xd.items():
        if round(yd[key], z) != round(value, z):
            raise ValueError("{} != {}".format(yd[key], value))


@with_setup(setup_monty, teardown)
def test_guest_monty():
    a = monty_network.predict_proba({'guest': 'A'})
    b = monty_network.predict_proba({'guest': 'B'})
    c = monty_network.predict_proba({'guest': 'C'})

    prize_correct = DiscreteDistribution(
        {'A': 1. / 3, 'B': 1. / 3, 'C': 1. / 3})

    assert_discrete_equal(a[prize_index], b[prize_index])
    assert_discrete_equal(a[prize_index], c[prize_index])
    assert_discrete_equal(a[prize_index], prize_correct)

    assert_discrete_equal(a[monty_index], DiscreteDistribution(
        {'A': 0.0, 'B': 1. / 2, 'C': 1. / 2}))
    assert_discrete_equal(b[monty_index], DiscreteDistribution(
        {'A': 1. / 2, 'B': 0.0, 'C': 1. / 2}))
    assert_discrete_equal(c[monty_index], DiscreteDistribution(
        {'A': 1. / 2, 'B': 1. / 2, 'C': 0.0}))


@with_setup(setup_monty, teardown)
def test_guest_with_monty():
    b = monty_network.predict_proba({'guest': 'A', 'monty': 'B'})
    c = monty_network.predict_proba({'guest': 'A', 'monty': 'C'})

    assert_equal(b[guest_index], 'A')
    assert_equal(b[monty_index], 'B')
    assert_discrete_equal(b[prize_index], DiscreteDistribution(
        {'A': 1. / 3, 'B': 0.0, 'C': 2. / 3}))

    assert_equal(c[guest_index], 'A')
    assert_equal(c[monty_index], 'C')
    assert_discrete_equal(c[prize_index], DiscreteDistribution(
        {'A': 1. / 3, 'B': 2. / 3, 'C': 0.0}))


@with_setup(setup_monty, teardown)
def test_monty():
    a = monty_network.predict_proba({'monty': 'A'})

    assert_equal(a[monty_index], 'A')
    assert_discrete_equal(a[guest_index], a[prize_index])
    assert_discrete_equal(a[guest_index], DiscreteDistribution(
        {'A': 0.0, 'B': 1. / 2, 'C': 1. / 2}))


@with_setup(setup_monty, teardown)
def test_predict():
    obs = [['A', None, 'B'],
           ['A', None, 'C'],
           ['A', 'B', 'C']]

    predictions = monty_network.predict(obs)

    assert_array_equal(predictions,
                       [
                         ['A', 'C', 'B'],
                         ['A', 'B', 'C'],
                         ['A', 'B', 'C']
                       ])

    assert_array_equal(obs,
                       [
                         ['A', None, 'B'],
                         ['A', None, 'C'],
                         ['A', 'B', 'C']
                       ])

@with_setup(setup_monty, teardown)
def test_rejection_sampling():
    numpy.random.seed(0)
    predictions = monty_network._rejection(n=10,evidences=[{'guest':'A', 'monty':'B'}])
    (unique, counts) = numpy.unique(predictions[:,1], return_counts=True)
    assert_array_equal(unique, ['A', 'C'])
    assert counts[0] > 0 and counts[0] < 4
    # Need to find where random seed is changed so next test can work 
    # assert_array_equal(predictions,
    #                    [['A', 'C', 'B'],
    #                     ['A', 'C', 'B'],
    #                     ['A', 'A', 'B'],
    #                     ['A', 'C', 'B'],
    #                     ['A', 'C', 'B'],
    #                     ['A', 'C', 'B'],
    #                     ['A', 'C', 'B'],
    #                     ['A', 'C', 'B'],
    #                     ['A', 'C', 'B'],
    #                     ['A', 'C', 'B']])

@with_setup(setup_monty, teardown)
def test_gibbs_sampling():
    # evidences = [['A', None, 'B'],
    #              ['A', None, 'C'],
    #              ['A', 'B', 'C' ]]
    predictions = monty_network._gibbs(n=1000,evidences=[{'guest': 'A', 'monty': 'B'}])
    values, counts = numpy.unique(predictions[:, 1], return_counts=True)
    # will fail from time to time
    # need to fix the seed
    assert(abs(counts[0]-340) < 34)

@with_setup(setup_monty, teardown)
def test_predict_parallel():
    obs = [['A', None, 'B'],
           ['A', None, 'C'],
           ['A', 'B', 'C']]

    predictions = monty_network.predict(obs, n_jobs=2)
    assert_array_equal(predictions,
                       [
                         ['A', 'C', 'B'],
                         ['A', 'B', 'C'],
                         ['A', 'B', 'C']
                       ])

    assert_array_equal(obs,
                       [
                         ['A', None, 'B'],
                         ['A', None, 'C'],
                         ['A', 'B', 'C']
                       ])

@with_setup(setup_monty, teardown)
def test_predict_datagenerator():
    obs = [['A', None, 'B'],
           ['A', None, 'C'],
           ['A', 'B', 'C']]

    X = DataGenerator(obs)

    predictions = monty_network.predict(X)

    assert_array_equal(predictions,
                       [
                         ['A', 'C', 'B'],
                         ['A', 'B', 'C'],
                         ['A', 'B', 'C']
                       ])

    assert_array_equal(obs,
                       [
                         ['A', None, 'B'],
                         ['A', None, 'C'],
                         ['A', 'B', 'C']
                       ])

@with_setup(setup_monty, teardown)
def test_numpy_predict():
    obs = numpy.array([['A', None, 'B'],
                    ['A', None, 'C'],
                    ['A', 'B', 'C']])

    predictions = monty_network.predict(obs)

    assert_array_equal(predictions,
                       [
                         ['A', 'C', 'B'],
                         ['A', 'B', 'C'],
                         ['A', 'B', 'C']
                       ])

    assert_array_equal(obs,
                       [
                         ['A', None, 'B'],
                         ['A', None, 'C'],
                         ['A', 'B', 'C']
                       ])

@with_setup(setup_monty, teardown)
def test_numpy_predict_parallel():
    obs = numpy.array([['A', None, 'B'],
                    ['A', None, 'C'],
                    ['A', 'B', 'C']])

    predictions = monty_network.predict(obs, n_jobs=2)

    assert_array_equal(predictions,
                       [
                         ['A', 'C', 'B'],
                         ['A', 'B', 'C'],
                         ['A', 'B', 'C']
                       ])

    assert_array_equal(obs,
                       [
                         ['A', None, 'B'],
                         ['A', None, 'C'],
                         ['A', 'B', 'C']
                       ])

@with_setup(setup_monty, teardown)
def test_numpy_predict_datagenerator():
    obs = numpy.array([['A', None, 'B'],
                    ['A', None, 'C'],
                    ['A', 'B', 'C']])

    X = DataGenerator(obs)

    predictions = monty_network.predict(X)

    assert_array_equal(predictions,
                       [
                         ['A', 'C', 'B'],
                         ['A', 'B', 'C'],
                         ['A', 'B', 'C']
                       ])

    assert_array_equal(obs,
                       [
                         ['A', None, 'B'],
                         ['A', None, 'C'],
                         ['A', 'B', 'C']
                       ])


@with_setup(setup_monty, teardown)
def test_single_dict_predict_proba():
    obs = {'guest': 'A',  'monty': 'B'}
    y = DiscreteDistribution({'A': 1./3, 'B': 0., 'C': 2./3})
    y_hat = monty_network.predict_proba(obs)

    assert_equal(y_hat[0], 'A')
    assert_equal(y_hat[2], 'B')
    assert_discrete_equal(y_hat[1], y)


@with_setup(setup_large_monty, teardown)
def test_single_dict_large_predict_proba():
    obs = {'large_monty_friend' : True,  'large_monty_guest': 'A',
        'large_monty_prize': 'A', 'large_monty': 'C'}
    y1 = DiscreteDistribution({0: 0.0472, 1: 0.781, 2: 0.17167})
    y2 = DiscreteDistribution({True: 0.8562, False: 0.143776})
    y_hat = large_monty_network.predict_proba(obs)

    assert_equal(y_hat[0], True)
    assert_equal(y_hat[1], 'A')
    assert_equal(y_hat[2], 'A')
    assert_equal(y_hat[3], 'C')
    assert_discrete_equal(y_hat[4], y1, 3)
    assert_discrete_equal(y_hat[5], y2, 3)

    obs = {'large_monty_friend' : True, 'large_monty_prize': 'A',
        'large_monty': 'C', 'large_monty_remaining' : 2}
    y1 = DiscreteDistribution({'A': 0.5, 'B': 0.5, 'C': 0.0})
    y2 = DiscreteDistribution({True: 0.75, False: 0.25})
    y_hat = large_monty_network.predict_proba(obs)

    assert_equal(y_hat[0], True)
    assert_equal(y_hat[2], 'A')
    assert_equal(y_hat[3], 'C')
    assert_equal(y_hat[4], 2)
    assert_discrete_equal(y_hat[1], y1)
    assert_discrete_equal(y_hat[5], y2)


@with_setup(setup_monty, teardown)
def test_list_of_lists_predict_proba():
    obs = [['A', None, 'B']]
    y = DiscreteDistribution({'A': 1./3, 'B': 0., 'C': 2./3})
    y_hat = monty_network.predict_proba(obs)

    assert_equal(y_hat[0][0], 'A')
    assert_equal(y_hat[0][2], 'B')
    assert_discrete_equal(y_hat[0][1], y)


@with_setup(setup_large_monty, teardown)
def test_list_of_lists_large_predict_proba():
    obs = [[True,  'A', 'A', 'C', None, None]]
    y1 = DiscreteDistribution({0: 0.0472, 1: 0.781, 2: 0.17167})
    y2 = DiscreteDistribution({True: 0.8562, False: 0.143776})
    y_hat = large_monty_network.predict_proba(obs)

    assert_equal(y_hat[0][0], True)
    assert_equal(y_hat[0][1], 'A')
    assert_equal(y_hat[0][2], 'A')
    assert_equal(y_hat[0][3], 'C')
    assert_discrete_equal(y_hat[0][4], y1, 3)
    assert_discrete_equal(y_hat[0][5], y2, 3)

    obs = [[True, None, 'A', 'C', 2, None]]
    y1 = DiscreteDistribution({'A': 0.5, 'B': 0.5, 'C': 0.0})
    y2 = DiscreteDistribution({True: 0.75, False: 0.25})
    y_hat = large_monty_network.predict_proba(obs)

    assert_equal(y_hat[0][0], True)
    assert_equal(y_hat[0][2], 'A')
    assert_equal(y_hat[0][3], 'C')
    assert_equal(y_hat[0][4], 2)
    assert_discrete_equal(y_hat[0][1], y1)
    assert_discrete_equal(y_hat[0][5], y2)


@with_setup(setup_monty, teardown)
def test_list_of_dicts_predict_proba():
    obs = [{'guest': 'A',  'monty': 'B'}]
    y = DiscreteDistribution({'A': 1./3, 'B': 0., 'C': 2./3})
    y_hat = monty_network.predict_proba(obs)

    assert_equal(y_hat[0][0], 'A')
    assert_equal(y_hat[0][2], 'B')
    assert_discrete_equal(y_hat[0][1], y)


@with_setup(setup_large_monty, teardown)
def test_list_of_dicts_large_predict_proba():
    obs = [{'large_monty_friend' : True,  'large_monty_guest': 'A',
        'large_monty_prize': 'A', 'large_monty': 'C'}]
    y1 = DiscreteDistribution({0: 0.0472, 1: 0.781, 2: 0.17167})
    y2 = DiscreteDistribution({True: 0.8562, False: 0.143776})
    y_hat = large_monty_network.predict_proba(obs)

    assert_equal(y_hat[0][0], True)
    assert_equal(y_hat[0][1], 'A')
    assert_equal(y_hat[0][2], 'A')
    assert_equal(y_hat[0][3], 'C')
    assert_discrete_equal(y_hat[0][4], y1, 3)
    assert_discrete_equal(y_hat[0][5], y2, 3)

    obs = [{'large_monty_friend' : True, 'large_monty_prize': 'A',
        'large_monty': 'C', 'large_monty_remaining' : 2}]
    y1 = DiscreteDistribution({'A': 0.5, 'B': 0.5, 'C': 0.0})
    y2 = DiscreteDistribution({True: 0.75, False: 0.25})
    y_hat = large_monty_network.predict_proba(obs)

    assert_equal(y_hat[0][0], True)
    assert_equal(y_hat[0][2], 'A')
    assert_equal(y_hat[0][3], 'C')
    assert_equal(y_hat[0][4], 2)
    assert_discrete_equal(y_hat[0][1], y1)
    assert_discrete_equal(y_hat[0][5], y2)


@with_setup(setup_monty, teardown)
def test_list_of_dicts_predict_proba_parallel():
    obs = [{'guest': 'A',  'monty': 'B'},
           {'guest': 'B', 'prize': 'A'},
           {'monty': 'C', 'prize': 'B'},
           {'monty': 'B'}, {'prize': 'A'}]
    y = DiscreteDistribution({'A': 1./3, 'B': 0., 'C': 2./3})
    y_hat = monty_network.predict_proba(obs, n_jobs=2)

    assert_equal(y_hat[0][0], 'A')
    assert_equal(y_hat[0][2], 'B')
    assert_discrete_equal(y_hat[0][1], y)

    assert_equal(y_hat[1][0], 'B')
    assert_equal(y_hat[1][1], 'A')

    assert_equal(y_hat[3][2], 'B')
    assert_equal(y_hat[4][1], 'A')


@with_setup(setup_monty, teardown)
def test_raise_error():
    obs = [['green', 'cat', None]]
    assert_raises(ValueError, monty_network.predict, obs)

    obs = [['A', 'b', None]]
    assert_raises(ValueError, monty_network.predict, obs)

    obs = [['none', 'B', None]]
    assert_raises(ValueError, monty_network.predict, obs)

    obs = [['NaN', 'B', None]]
    assert_raises(ValueError, monty_network.predict, obs)

    obs = [['A', 'C', 'D']]
    assert_raises(ValueError, monty_network.predict, obs)


def test_exact_structure_learning():
    logps = -19.8282, -345.9527, -4847.59688, -604.0190
    for X, logp in zip(datasets, logps):
        model = BayesianNetwork.from_samples(X, algorithm='exact')
        model2 = BayesianNetwork.from_samples(X, algorithm='exact-dp')
        assert_almost_equal(model.log_probability(X).sum(), model2.log_probability(X).sum())
        assert_almost_equal(model.log_probability(X).sum(), logp, 4)

def test_exact_low_memory_structure_learning():
    logps = -19.8282, -345.9527, -4847.59688, -604.0190
    for X, logp in zip(datasets, logps):
        model = BayesianNetwork.from_samples(X, low_memory=True, algorithm='exact')
        model2 = BayesianNetwork.from_samples(X, low_memory=True, algorithm='exact-dp')
        assert_almost_equal(model.log_probability(X).sum(), model2.log_probability(X).sum())
        assert_almost_equal(model.log_probability(X).sum(), logp, 4)

def test_exact_penalized_structure_learning():
    n_parents = [(5, 3, 4), (10, 0, 1), (21, 0, 8), (26, 3, 21)]
    for X, n_parents in zip(datasets, n_parents):
        model = BayesianNetwork.from_samples(X, algorithm='exact', penalty=0)
        model2 = BayesianNetwork.from_samples(X, algorithm='exact-dp', penalty=0)
        assert_equal(sum(map(len, model.structure)), n_parents[0])
        assert_equal(sum(map(len, model2.structure)), n_parents[0])

        model = BayesianNetwork.from_samples(X, algorithm='exact')
        model2 = BayesianNetwork.from_samples(X, algorithm='exact-dp')
        assert_equal(sum(map(len, model.structure)), n_parents[1])
        assert_equal(sum(map(len, model2.structure)), n_parents[1])

        model = BayesianNetwork.from_samples(X, algorithm='exact', penalty=1)
        model2 = BayesianNetwork.from_samples(X, algorithm='exact-dp', penalty=1)
        assert_equal(sum(map(len, model.structure)), n_parents[2])
        assert_equal(sum(map(len, model2.structure)), n_parents[2])

        model = BayesianNetwork.from_samples(X, algorithm='exact', penalty=100)
        model2 = BayesianNetwork.from_samples(X, algorithm='exact-dp', penalty=100)
        assert_equal(sum(map(len, model.structure)), 0)
        assert_equal(sum(map(len, model2.structure)), 0)

def test_exact_penalized_low_memory_structure_learning():
    n_parents = [(5, 3, 4), (10, 0, 1), (21, 0, 8), (26, 3, 21)]
    for X, n_parents in zip(datasets, n_parents):
        model = BayesianNetwork.from_samples(X, low_memory=True, algorithm='exact', penalty=0)
        model2 = BayesianNetwork.from_samples(X, low_memory=True, algorithm='exact-dp', penalty=0)
        assert_equal(sum(map(len, model.structure)), n_parents[0])
        assert_equal(sum(map(len, model2.structure)), n_parents[0])

        model = BayesianNetwork.from_samples(X, low_memory=True, algorithm='exact')
        model2 = BayesianNetwork.from_samples(X, low_memory=True, algorithm='exact-dp')
        assert_equal(sum(map(len, model.structure)), n_parents[1])
        assert_equal(sum(map(len, model2.structure)), n_parents[1])

        model = BayesianNetwork.from_samples(X, low_memory=True, algorithm='exact', penalty=1)
        model2 = BayesianNetwork.from_samples(X, low_memory=True, algorithm='exact-dp', penalty=1)
        assert_equal(sum(map(len, model.structure)), n_parents[2])
        assert_equal(sum(map(len, model2.structure)), n_parents[2])

        model = BayesianNetwork.from_samples(X, low_memory=True, algorithm='exact', penalty=100)
        model2 = BayesianNetwork.from_samples(X, low_memory=True, algorithm='exact-dp', penalty=100)
        assert_equal(sum(map(len, model.structure)), 0)
        assert_equal(sum(map(len, model2.structure)), 0)

def test_exact_structure_learning_include_edges():
    for X in datasets:
        model = BayesianNetwork.from_samples(X, algorithm='exact', 
            include_edges=[(1, 3)])
        assert_equal(model.structure[3], (1,))

        model = BayesianNetwork.from_samples(X, algorithm='exact')
        assert_not_equal(model.structure[3], (1,))

def test_exact_low_memory_structure_learning_include_edges():
    for X in datasets:
        model = BayesianNetwork.from_samples(X, algorithm='exact', 
            low_memory=True, include_edges=[(1, 3)])
        assert_equal(model.structure[3], (1,))

        model = BayesianNetwork.from_samples(X, low_memory=True,
            algorithm='exact')
        assert_not_equal(model.structure[3], (1,))

def test_exact_dp_structure_learning_include_edges():
    for X in datasets:
        model = BayesianNetwork.from_samples(X, algorithm='exact-dp', 
            include_edges=[(1, 3)])
        assert_equal(model.structure[3], (1,))

        model = BayesianNetwork.from_samples(X, algorithm='exact-dp')
        assert_not_equal(model.structure[3], (1,))

def test_exact_structure_learning_exclude_edges():
    for X in datasets:
        X = X.copy()
        X[:,1] = X[:,-1]
        X[:,-2] = X[:,-1]

        d = X.shape[1]

        # Learn constrained network
        model = BayesianNetwork.from_samples(X, algorithm='exact', 
            exclude_edges=[(1, d-1), (d-1, d-2)])    
        assert_not_equal(model.structure[-1], (1,))
        assert_not_equal(model.structure[-2], (d-1,))
        assert_equal(model.structure[-2], (1,))

def test_exact_low_memory_structure_learning_exclude_edges():
    for X in datasets:
        X = X.copy()
        X[:,1] = X[:,-1]
        X[:,-2] = X[:,-1]

        d = X.shape[1]

        # Learn constrained network
        model = BayesianNetwork.from_samples(X, algorithm='exact', 
            low_memory=True, exclude_edges=[(1, d-1), (d-1, d-2)])    
        assert_not_equal(model.structure[-1], (1,))
        assert_not_equal(model.structure[-2], (d-1,))
        assert_equal(model.structure[-2], (1,))


def test_exact_dp_structure_learning_exclude_edges():
    for X in datasets:
        X = X.copy()
        X[:,1] = X[:,-1]
        X[:,-2] = X[:,-1]

        d = X.shape[1]

        # Learn constrained network
        model = BayesianNetwork.from_samples(X, algorithm='exact-dp', 
            exclude_edges=[(1, d-1), (d-1, d-2)])    
        assert_not_equal(model.structure[-1], (1,))
        assert_not_equal(model.structure[-2], (d-1,))

def test_constrained_sl_structure_learning_exclude_edges():
    for X in datasets:
        X = X.copy()
        X[:,1] = X[:,-1]
        X[:,-2] = X[:,-1]

        d = X.shape[1]

        cg = DiGraph()
        n = tuple(range(d))
        cg.add_edge(n, n)

        # Learn constrained network
        model = BayesianNetwork.from_samples(X, algorithm='greedy', 
            constraint_graph=cg, exclude_edges=[(1, d-1), (d-1, d-2)])    
        assert_not_equal(model.structure[-1], (1,))
        assert_not_equal(model.structure[-2], (d-1,))
        assert_equal(model.structure[-2], (1,))

def test_low_memory_constrained_sl_structure_learning_exclude_edges():
    for X in datasets:
        X = X.copy()
        X[:,1] = X[:,-1]
        X[:,-2] = X[:,-1]

        d = X.shape[1]

        cg = DiGraph()
        n = tuple(range(d))
        cg.add_edge(n, n)

        # Learn constrained network
        model = BayesianNetwork.from_samples(X, algorithm='greedy', 
            low_memory=True, constraint_graph=cg, 
            exclude_edges=[(1, d-1), (d-1, d-2)])    
        assert_not_equal(model.structure[-1], (1,))
        assert_not_equal(model.structure[-2], (d-1,))
        assert_equal(model.structure[-2], (1,))

def test_constrained_parents_structure_learning_exclude_edges():
    for X in datasets:
        X = X.copy()
        X[:,1] = X[:,-1]
        X[:,-2] = X[:,-1]

        d = X.shape[1]
        d1 = int(numpy.ceil(d / 2))
        
        # Node groups
        g1 = tuple(range(0, d1))
        g2 = tuple(range(d1, d))
        
        # Constraint graph:
        cg = DiGraph()
        cg.add_edge(g1, g2)

        # Learn constrained network
        model1 = BayesianNetwork.from_samples(X, algorithm='exact', 
            constraint_graph=cg, exclude_edges=[(1, d-1)])
        assert_not_equal(model1.structure[-1], (1,))
        assert_equal(model1.structure[-2], (1,))

        model2 = BayesianNetwork.from_samples(X, algorithm='exact',
            constraint_graph=cg)
        assert_equal(model2.structure[-1], (1,))
        assert_equal(model2.structure[-2], (1,))

    X = numpy.random.randint(2, size=(50, 8))
    X[:,0] = X[:,4]
    X[:,1] = X[:,7]
    X[:,2] = X[:,7]

    cg = DiGraph()
    n1 = (0, 2, 3, 5, 6)
    n2 = (1, 4, 7)
    cg.add_edge(n1, n2)

    model = BayesianNetwork.from_samples(X, algorithm='exact', 
        constraint_graph=cg, exclude_edges=[(0, 4), (2, 7)])
    assert_not_equal(model.structure[7], (2,))
    assert_not_equal(model.structure[4], (0,))

    model = BayesianNetwork.from_samples(X, algorithm='exact',
        constraint_graph=cg)
    assert_equal(model.structure[7], (2,))
    assert_equal(model.structure[4], (0,))

def test_low_memory_constrained_parents_structure_learning_exclude_edges():
    for X in datasets:
        X = X.copy()
        X[:,1] = X[:,-1]
        X[:,-2] = X[:,-1]

        d = X.shape[1]
        d1 = int(numpy.ceil(d / 2))
        
        # Node groups
        g1 = tuple(range(0, d1))
        g2 = tuple(range(d1, d))
        
        # Constraint graph:
        cg = DiGraph()
        cg.add_edge(g1, g2)

        # Learn constrained network
        model1 = BayesianNetwork.from_samples(X, algorithm='exact', 
            low_memory=True, constraint_graph=cg, exclude_edges=[(1, d-1)])
        assert_not_equal(model1.structure[-1], (1,))
        assert_equal(model1.structure[-2], (1,))

        model2 = BayesianNetwork.from_samples(X, algorithm='exact',
            low_memory=True, constraint_graph=cg)
        assert_equal(model2.structure[-1], (1,))
        assert_equal(model2.structure[-2], (1,))

    X = numpy.random.randint(2, size=(50, 8))
    X[:,0] = X[:,4]
    X[:,1] = X[:,7]
    X[:,2] = X[:,7]

    cg = DiGraph()
    n1 = (0, 2, 3, 5, 6)
    n2 = (1, 4, 7)
    cg.add_edge(n1, n2)

    model = BayesianNetwork.from_samples(X, algorithm='exact', 
        low_memory=True, constraint_graph=cg, exclude_edges=[(0, 4), (2, 7)])
    assert_not_equal(model.structure[7], (2,))
    assert_not_equal(model.structure[4], (0,))

    model = BayesianNetwork.from_samples(X, algorithm='exact',
        low_memory=True, constraint_graph=cg)
    assert_equal(model.structure[7], (2,))
    assert_equal(model.structure[4], (0,))

def test_constrained_slap_structure_learning_exclude_edges():
    for X in datasets:
        X = X.copy()
        X[:,1] = X[:,-1]
        X[:,-2] = X[:,-1]

        d = X.shape[1]
        d1 = int(numpy.ceil(d / 2))
        
        # Node groups
        g1 = tuple(range(0, d1))
        g2 = tuple(range(d1, d))
        
        # Constraint graph:
        cg = DiGraph()
        cg.add_edge(g1, g2)
        cg.add_edge(g2, g2)

        # Learn constrained network
        model1 = BayesianNetwork.from_samples(X, algorithm='exact', 
            constraint_graph=cg, exclude_edges=[(1, d-1)])
        assert_not_equal(model1.structure[-1], (1,))
        assert_equal(model1.structure[-1], (d-2,))

        #model2 = BayesianNetwork.from_samples(X, algorithm='exact',
        #    constraint_graph=cg)
        #assert_equal(model2.structure[-1], (d-2,))

    X = numpy.random.randint(2, size=(50, 8))
    X[:,0] = X[:,4]
    X[:,1] = X[:,7]
    X[:,2] = X[:,7]

    cg = DiGraph()
    n1 = (0, 2, 3, 5, 6)
    n2 = (1, 4, 7)
    cg.add_edge(n1, n2)
    cg.add_edge(n2, n2)

    model = BayesianNetwork.from_samples(X, algorithm='exact', 
        constraint_graph=cg, exclude_edges=[(0, 4), (2, 7)])
    assert_not_equal(model.structure[7], (2,))
    assert_not_equal(model.structure[4], (0,))

    model = BayesianNetwork.from_samples(X, algorithm='exact',
        constraint_graph=cg)
    assert_equal(model.structure[7], (2,))
    assert_equal(model.structure[4], (0,))

def test_constrained_parents_structure_learning():
    logps1 = [-12.2173, -207.3633, -3462.7469, -480.0970]
    logps2 = [-10.8890, -207.3633, -3462.7469, -480.0970]

    for X, logp1, logp2 in zip(datasets, logps1, logps2):
        X = X.copy()
        X[:,1] = X[:,-1]
        X[:,-2] = X[:,-1]

        d = X.shape[1]
        d1 = int(numpy.ceil(d / 2))
        
        # Node groups
        g1 = tuple(range(0, d1))
        g2 = tuple(range(d1, d))
        
        # Constraint graph:
        cg = DiGraph()
        cg.add_edge(g1, g2)

        # Learn constrained network
        model1 = BayesianNetwork.from_samples(X, algorithm='exact', 
            constraint_graph=cg)
        assert_almost_equal(model1.log_probability(X).sum(), logp1, 4)
        
        # Check structure constraints satisfied
        for node in g1:
            assert_equal(0, len(model1.structure[node]))
        
        assert_equal(model1.structure[-1], (1,))
        assert_equal(model1.structure[-2], (1,))

        model2 = BayesianNetwork.from_samples(X, algorithm='exact')
        assert_almost_equal(model2.log_probability(X).sum(), logp2, 4)
        assert_equal(model2.structure[-1], (d-2,))
        assert_equal(model2.structure[-2], (1,))

def test_constrained_slap_structure_learning():
    logps = [-21.7780, -345.9527, -4847.5969, -611.0356]

    for X, logp in zip(datasets, logps):
        d = X.shape[1]
        d1 = int(numpy.ceil(d / 2))
        
        # Node groups
        g1 = tuple(range(0, d1))
        g2 = tuple(range(d1, d))
        
        # Constraint graph:
        cg = DiGraph()
        cg.add_edge(g1, g2)
        cg.add_edge(g2, g2)

        # Learn constrained network
        model = BayesianNetwork.from_samples(X, algorithm='exact', constraint_graph=cg)
        assert_almost_equal(model.log_probability(X).sum(), logp, 4)
        
        # Check structure constraints satisfied
        for node in g1:
            assert_equal(0, len(model.structure[node]))

def test_from_structure():
    X = datasets[1]
    structure = ((1, 2), (4,), (), (), (3,))
    model = BayesianNetwork.from_structure(X, structure=structure)

    assert_equal(model.structure, structure)
    assert_almost_equal(model.log_probability(X).sum(), -344.38287, 4)

    model2 = BayesianNetwork.from_json(model.to_json())
    assert_equal(model2.structure, structure)
    assert_almost_equal(model.log_probability(X).sum(), -344.38287, 4)

    model_dtype = type(model.states[0].distribution.parameters[0][0][0])
    model2_dtype = type(model2.states[0].distribution.parameters[0][0][0])
    assert_equal(model_dtype, model2_dtype)

def test_robust_from_structure():
    X = datasets[1]
    structure = ((1, 2), (4,), (), (), (3,))
    model = BayesianNetwork.from_structure(X, structure=structure)

    assert_equal(model.structure, structure)
    assert_almost_equal(model.log_probability(X).sum(), -344.38287, 4)

    model2 = from_json(model.to_json())
    assert_equal(model2.structure, structure)
    assert_almost_equal(model.log_probability(X).sum(), -344.38287, 4)

    model_dtype = type(model.states[0].distribution.parameters[0][0][0])
    model2_dtype = type(model2.states[0].distribution.parameters[0][0][0])
    assert_equal(model_dtype, model2_dtype)

@with_setup(setup_random_mixed)
def test_from_json():
    model2 = BayesianNetwork.from_json(model.to_json())

    logp1 = model.log_probability(X)
    logp2 = model2.log_probability(X)
    logp = [-2.304186, -1.898721, -1.898721, -2.224144, -1.898721, -1.978764,
        -1.898721, -1.898721, -1.898721, -1.898721, -1.818679, -2.384229,
        -2.304186, -1.978764, -2.304186, -2.384229, -2.304186, -2.384229,
        -2.304186, -1.978764, -2.224144, -1.818679, -1.898721, -2.304186,
        -2.304186, -1.898721, -1.818679, -1.898721, -1.818679, -2.304186,
        -1.978764, -2.224144, -1.898721, -2.304186, -1.898721, -1.818679,
        -2.304186, -1.898721, -1.898721, -2.384229, -2.224144, -1.818679,
        -2.384229, -1.978764, -1.818679, -1.978764, -1.898721, -1.818679,
        -2.224144, -1.898721]

    assert_array_almost_equal(logp1, logp2)
    assert_array_almost_equal(logp1, logp)
    assert_array_almost_equal(logp2, logp)

    model_dtype = type(list(model.states[0].distribution.parameters[0].keys())[0])
    model2_dtype = type(list(model2.states[0].distribution.parameters[0].keys())[0])
    assert_equal(model_dtype, model2_dtype)

@with_setup(setup_random_mixed)
def test_robust_from_json():
    model2 = from_json(model.to_json())

    logp1 = model.log_probability(X)
    logp2 = model2.log_probability(X)
    logp = [-2.304186, -1.898721, -1.898721, -2.224144, -1.898721, -1.978764,
        -1.898721, -1.898721, -1.898721, -1.898721, -1.818679, -2.384229,
        -2.304186, -1.978764, -2.304186, -2.384229, -2.304186, -2.384229,
        -2.304186, -1.978764, -2.224144, -1.818679, -1.898721, -2.304186,
        -2.304186, -1.898721, -1.818679, -1.898721, -1.818679, -2.304186,
        -1.978764, -2.224144, -1.898721, -2.304186, -1.898721, -1.818679,
        -2.304186, -1.898721, -1.898721, -2.384229, -2.224144, -1.818679,
        -2.384229, -1.978764, -1.818679, -1.978764, -1.898721, -1.818679,
        -2.224144, -1.898721]

    assert_array_almost_equal(logp1, logp2)
    assert_array_almost_equal(logp1, logp)
    assert_array_almost_equal(logp2, logp)

    model_dtype = type(list(model.states[0].distribution.parameters[0].keys())[0])
    model2_dtype = type(list(model2.states[0].distribution.parameters[0].keys())[0])
    assert_equal(model_dtype, model2_dtype)

def test_float64_from_json():
    X = datasets[1].astype('float64')
    structure = ((1, 2), (4,), (), (), (3,))
    model = BayesianNetwork.from_structure(X, structure=structure)

    assert_equal(model.structure, structure)
    assert_almost_equal(model.log_probability(X).sum(), -344.38287, 4)

    model2 = BayesianNetwork.from_json(model.to_json())
    assert_equal(model2.structure, structure)
    assert_almost_equal(model.log_probability(X).sum(), -344.38287, 4)

    model_dtype = type(model.states[0].distribution.parameters[0][0][0])
    model2_dtype = type(model2.states[0].distribution.parameters[0][0][0])
    assert_equal(model_dtype, model2_dtype)

def test_robust_float64_from_json():
    X = datasets[1].astype('float64')
    structure = ((1, 2), (4,), (), (), (3,))
    model = BayesianNetwork.from_structure(X, structure=structure)

    assert_equal(model.structure, structure)
    assert_almost_equal(model.log_probability(X).sum(), -344.38287, 4)

    model2 = from_json(model.to_json())
    assert_equal(model2.structure, structure)
    assert_almost_equal(model.log_probability(X).sum(), -344.38287, 4)

    model_dtype = type(model.states[0].distribution.parameters[0][0][0])
    model2_dtype = type(model2.states[0].distribution.parameters[0][0][0])
    assert_equal(model_dtype, model2_dtype)

def test_parallel_structure_learning():
    logps = -19.8282, -345.9527, -4847.59688, -604.0190
    for X, logp in zip(datasets, logps):
        model = BayesianNetwork.from_samples(X, algorithm='exact')
        model2 = BayesianNetwork.from_samples(X, algorithm='exact', n_jobs=2)
        assert_equal(model.log_probability(X).sum(), model2.log_probability(X).sum())
        assert_almost_equal(model.log_probability(X).sum(), logp, 4)


def test_greedy_structure_learning():
    logps = -19.8282, -345.9527, -4847.59688, -611.0356
    for X, logp in zip(datasets, logps):
        model = BayesianNetwork.from_samples(X, algorithm='greedy')
        assert_almost_equal(model.log_probability(X).sum(), logp, 4)

def test_greedy_structure_learning_include_edges():
    for X in datasets:
        model = BayesianNetwork.from_samples(X, algorithm='greedy', 
            include_edges=[(1, 3)])
        assert_equal(model.structure[3], (1,))

        model = BayesianNetwork.from_samples(X, algorithm='greedy')
        assert_not_equal(model.structure[3], (1,))

def test_greedy_structure_learning_exclude_edges():
    for X in datasets:
        X = X.copy()
        X[:,1] = X[:,-1]
        X[:,-2] = X[:,-1]

        d = X.shape[1]

        # Learn constrained network
        model = BayesianNetwork.from_samples(X, algorithm='greedy', 
            exclude_edges=[(1, d-1), (d-1, d-2)])    
        assert_not_equal(model.structure[-1], (1,))
        assert_not_equal(model.structure[-2], (d-1,))
        assert_equal(model.structure[-2], (1,))

def test_greedy_penalized_structure_learning():
    n_parents = [(5, 3, 4), (10, 0, 1), (21, 0, 5), (26, 1, 21)]
    for X, n_parents in zip(datasets, n_parents):
        model = BayesianNetwork.from_samples(X, algorithm='greedy', penalty=0)
        assert_equal(sum(map(len, model.structure)), n_parents[0])

        model = BayesianNetwork.from_samples(X, algorithm='greedy')
        assert_equal(sum(map(len, model.structure)), n_parents[1])

        model = BayesianNetwork.from_samples(X, algorithm='greedy', penalty=1)
        assert_equal(sum(map(len, model.structure)), n_parents[2])

        model = BayesianNetwork.from_samples(X, algorithm='greedy', penalty=100)
        assert_equal(sum(map(len, model.structure)), 0)

def test_chow_liu_structure_learning():
    logps = -19.8282, -344.248785, -4842.40158, -603.2370
    for X, logp in zip(datasets, logps):
        model = BayesianNetwork.from_samples(X, algorithm='chow-liu')
        assert_almost_equal(model.log_probability(X).sum(), logp, 4)


def test_exact_nan_structure_learning():
    logps = -6.13764, -159.6505, -2055.76364, -201.73615
    for X, logp in zip(datasets_nan, logps):
        model = BayesianNetwork.from_samples(X, algorithm='exact')
        model2 = BayesianNetwork.from_samples(X, algorithm='exact-dp')

        assert_equal(model.log_probability(X).sum(), model2.log_probability(X).sum())
        assert_almost_equal(model.log_probability(X).sum(), logp, 4)


def test_greedy_nan_structure_learning():
    logps = -7.5239, -159.6505, -2058.5706, -203.7662
    for X, logp in zip(datasets_nan, logps):
        model = BayesianNetwork.from_samples(X, algorithm='greedy')
        assert_almost_equal(model.log_probability(X).sum(), logp, 4)

@with_setup(setup_random_mixed, teardown)
def test_io_log_probability():
    X2 = DataGenerator(X)
    X3 = DataFrameGenerator(pandas.DataFrame(X))

    logp1 = model.log_probability(X)
    logp2 = model.log_probability(X2)
    logp3 = model.log_probability(X3)

    assert_array_almost_equal(logp1, logp2)
    assert_array_almost_equal(logp1, logp3)

@with_setup(setup_random_mixed, teardown)
def test_io_predict():
    X2 = DataGenerator(X)
    X3 = DataFrameGenerator(pandas.DataFrame(X))

    y_hat1 = model.predict(X)
    y_hat2 = model.predict(X2)
    y_hat3 = model.predict(X3)

    assert_array_equal(y_hat1, y_hat2)
    assert_array_equal(y_hat1, y_hat3)

@with_setup(setup_random_mixed, teardown)
def test_io_fit():
    d1 = DiscreteDistribution({True: 0.6, False: 0.4})
    d2 = ConditionalProbabilityTable([
        [True, 'A', 0.2],
        [True, 'B', 0.8],
        [False, 'A', 0.3],
        [False, 'B', 0.7]], [d1])
    d3 = ConditionalProbabilityTable([
        ['A', 0, 0.3],
        ['A', 1, 0.7],
        ['B', 0, 0.8],
        ['B', 1, 0.2]], [d2])

    n1 = Node(d1)
    n2 = Node(d2)
    n3 = Node(d3)

    model1 = BayesianNetwork()
    model1.add_nodes(n1, n2, n3)
    model1.add_edge(n1, n2)
    model1.add_edge(n2, n3)
    model1.bake()
    model1.fit(X, weights=weights)

    d1 = DiscreteDistribution({True: 0.2, False: 0.8})
    d2 = ConditionalProbabilityTable([
        [True, 'A', 0.7],
        [True, 'B', 0.2],
        [False, 'A', 0.4],
        [False, 'B', 0.6]], [d1])
    d3 = ConditionalProbabilityTable([
        ['A', 0, 0.9],
        ['A', 1, 0.1],
        ['B', 0, 0.0],
        ['B', 1, 1.0]], [d2])

    n1 = Node(d1)
    n2 = Node(d2)
    n3 = Node(d3)

    model2 = BayesianNetwork()
    model2.add_nodes(n1, n2, n3)
    model2.add_edge(n1, n2)
    model2.add_edge(n2, n3)
    model2.bake()
    model2.fit(data_generator)

    logp1 = model1.log_probability(X)
    logp2 = model2.log_probability(X)

    assert_array_almost_equal(logp1, logp2)

@with_setup(setup_random_mixed, teardown)
def test_io_from_samples():
    model1 = BayesianNetwork.from_samples(X, weights=weights)
    model2 = BayesianNetwork.from_samples(data_generator)

    logp1 = model1.log_probability(X)
    logp2 = model2.log_probability(X)

    assert_array_almost_equal(logp1, logp2)

@with_setup(setup_random_mixed, teardown)
def test_io_from_structure():
    structure = ((2,), (0, 2), ())

    model1 = BayesianNetwork.from_structure(X=X, weights=weights,
        structure=structure)
    model2 = BayesianNetwork.from_structure(X=data_generator,
        structure=structure)

    logp1 = model1.log_probability(X)
    logp2 = model2.log_probability(X)

    assert_array_almost_equal(logp1, logp2)

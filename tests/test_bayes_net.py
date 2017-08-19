# test_bayes_net.py
# Authors: Jacob Schreiber <jmschreiber91@gmail.com>
#          Nelson Liu <nelson@nelsonliu.me>
#
'''
These are unit tests for the Bayesian network part of pomegranate.
'''

from __future__ import division

from pomegranate import DiscreteDistribution
from pomegranate import ConditionalProbabilityTable
from pomegranate import State
from pomegranate import BayesianNetwork

from nose.tools import with_setup
from nose.tools import assert_equal
from nose.tools import assert_raises
from nose.tools import assert_almost_equal

import random, numpy
from numpy.testing import assert_array_equal
import sys

nan = numpy.nan
numpy.random.seed(1)

datasets = [numpy.random.randint(2, size=(10, 4)),
            numpy.random.randint(2, size=(100, 5)),
            numpy.random.randint(2, size=(1000, 7)),
            numpy.random.randint(2, size=(100, 9))]

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


def setup_huge_monty():
    # Build the huge monty hall huge_monty_network. This is an example I made
    # up with which may not exactly flow logically, but tests a varied type of
    # tables ensures heterogeneous types of data work together.
    global huge_monty_network, huge_monty_friend, huge_monty_guest, huge_monty
    global huge_monty_remaining, huge_monty_randomize, huge_monty_prize

    # Huge_Monty_Friend
    huge_monty_friend = DiscreteDistribution({True: 0.5, False: 0.5})

    # Huge_Monty_Guest emisisons are completely random
    huge_monty_guest = ConditionalProbabilityTable(
        [[True, 'A', 0.50],
         [True, 'B', 0.25],
         [True, 'C', 0.25],
         [False, 'A', 0.0],
         [False, 'B', 0.7],
         [False, 'C', 0.3]], [huge_monty_friend])

    # Number of huge_monty_remaining cars
    huge_monty_remaining = DiscreteDistribution({0: 0.1, 1: 0.7, 2: 0.2, })

    # Whether they huge_monty_randomize is dependent on the numnber of
    # huge_monty_remaining cars
    huge_monty_randomize = ConditionalProbabilityTable(
        [[0, True, 0.05],
         [0, False, 0.95],
         [1, True, 0.8],
         [1, False, 0.2],
         [2, True, 0.5],
         [2, False, 0.5]], [huge_monty_remaining])

    # Where the huge_monty_prize is depends on if they huge_monty_randomize or
    # not and also the huge_monty_guests huge_monty_friend
    huge_monty_prize = ConditionalProbabilityTable(
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
         [False, False, 'C', 0.6]], [huge_monty_randomize, huge_monty_friend])

    # Monty is dependent on both the huge_monty_guest and the huge_monty_prize.
    huge_monty = ConditionalProbabilityTable(
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
         ['C', 'C', 'C', 0.0]], [huge_monty_guest, huge_monty_prize])

    # Make the states
    s0 = State(huge_monty_friend, name="huge_monty_friend")
    s1 = State(huge_monty_guest, name="huge_monty_guest")
    s2 = State(huge_monty_prize, name="huge_monty_prize")
    s3 = State(huge_monty, name="huge_monty")
    s4 = State(huge_monty_remaining, name="huge_monty_remaining")
    s5 = State(huge_monty_randomize, name="huge_monty_randomize")

    # Make the bayes net, add the states, and the conditional dependencies.
    huge_monty_network = BayesianNetwork("test")
    huge_monty_network.add_nodes(s0, s1, s2, s3, s4, s5)
    huge_monty_network.add_transition(s0, s1)
    huge_monty_network.add_transition(s1, s3)
    huge_monty_network.add_transition(s2, s3)
    huge_monty_network.add_transition(s4, s5)
    huge_monty_network.add_transition(s5, s2)
    huge_monty_network.add_transition(s0, s2)
    huge_monty_network.bake()


def teardown():
    pass


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

    assert_equal(female[1].log_probability("male"), float("-inf"))
    assert_equal(female[1].log_probability("female"), 0.0)

    assert_equal(female[2].log_probability("first"), float("-inf"))
    assert_equal(female[2].log_probability("second"), 0.0)
    assert_equal(female[2].log_probability("third"), float("-inf"))

    assert_equal(male[0].log_probability("survive"), float("-inf"))
    assert_equal(male[0].log_probability("perish"), 0.0)

    assert_equal(male[1].log_probability("male"), 0.0)
    assert_equal(male[1].log_probability("female"), float("-inf"))

    assert_equal(male[2].log_probability("first"), 0.0)
    assert_equal(male[2].log_probability("second"), float("-inf"))
    assert_equal(male[2].log_probability("third"), float("-inf"))

    titanic_network2 = BayesianNetwork.from_json(titanic_network.to_json())


@with_setup(setup_huge_monty, teardown)
def test_huge_monty():
    assert_almost_equal(huge_monty.log_probability(('A', 'A', 'C')), numpy.log(0.5))
    assert_almost_equal(huge_monty.log_probability(('B', 'B', 'C')), numpy.log(0.5))
    assert_equal(huge_monty.log_probability(('C', 'C', 'C')), float("-inf"))

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

    huge_monty_network.fit(data)

    assert_almost_equal(huge_monty.log_probability(('A', 'A', 'C')), numpy.log(0.6))
    assert_almost_equal(huge_monty.log_probability(('B', 'B', 'C')), numpy.log(0.5))
    assert_almost_equal(huge_monty.log_probability(('C', 'C', 'C')), numpy.log(0.75))


@with_setup(setup_huge_monty, teardown)
def test_huge_monty_friend():
    assert_almost_equal(huge_monty_friend.log_probability(True), numpy.log(0.5))
    assert_almost_equal(huge_monty_friend.log_probability(False), numpy.log(0.5))

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

    huge_monty_network.fit(data)

    assert_almost_equal(huge_monty_friend.log_probability(True), numpy.log(7. / 12))
    assert_almost_equal(huge_monty_friend.log_probability(False), numpy.log(5. / 12))


@with_setup(setup_huge_monty, teardown)
def test_huge_monty_remaining():
    assert_almost_equal(huge_monty_remaining.log_probability(0), numpy.log(0.1))
    assert_almost_equal(huge_monty_remaining.log_probability(1), numpy.log(0.7))
    assert_almost_equal(huge_monty_remaining.log_probability(2), numpy.log(0.2))

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

    huge_monty_network.fit(data)

    assert_almost_equal(huge_monty_remaining.log_probability(0), numpy.log(3. / 12))
    assert_almost_equal(huge_monty_remaining.log_probability(1), numpy.log(5. / 12))
    assert_almost_equal(huge_monty_remaining.log_probability(2), numpy.log(4. / 12))


@with_setup(setup_huge_monty, teardown)
def test_huge_monty_prize():
    assert_almost_equal(huge_monty_prize.log_probability(
        (True,  True,  'A')), numpy.log(0.3))
    assert_almost_equal(huge_monty_prize.log_probability(
        (True,  False, 'C')), numpy.log(0.4))
    assert_almost_equal(huge_monty_prize.log_probability(
        (False, True,  'B')), numpy.log(0.9))
    assert_almost_equal(huge_monty_prize.log_probability(
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

    huge_monty_network.fit(data)

    assert_almost_equal(huge_monty_prize.log_probability(
        (True, True, 'C')), numpy.log(0.5))
    assert_equal(huge_monty_prize.log_probability(
        (True, True, 'B')), float("-inf"))

    a = huge_monty_prize.log_probability((True, False, 'A'))
    b = huge_monty_prize.log_probability((True, False, 'B'))
    c = huge_monty_prize.log_probability((True, False, 'C'))

    assert_almost_equal(a, b)
    assert_almost_equal(b, c)

    assert_equal(huge_monty_prize.log_probability(
        (False, False, 'C')), float("-inf"))
    assert_almost_equal(huge_monty_prize.log_probability(
        (False, True, 'C')), numpy.log(2. / 3))


def discrete_equality(x, y, z=8):
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

    discrete_equality(a[prize_index], b[prize_index])
    discrete_equality(a[prize_index], c[prize_index])
    discrete_equality(a[prize_index], prize_correct)

    discrete_equality(a[monty_index], DiscreteDistribution(
        {'A': 0.0, 'B': 1. / 2, 'C': 1. / 2}))
    discrete_equality(b[monty_index], DiscreteDistribution(
        {'A': 1. / 2, 'B': 0.0, 'C': 1. / 2}))
    discrete_equality(c[monty_index], DiscreteDistribution(
        {'A': 1. / 2, 'B': 1. / 2, 'C': 0.0}))


@with_setup(setup_monty, teardown)
def test_guest_with_monty():
    b = monty_network.predict_proba({'guest': 'A', 'monty': 'B'})
    c = monty_network.predict_proba({'guest': 'A', 'monty': 'C'})

    discrete_equality(b[guest_index], DiscreteDistribution(
        {'A': 1., 'B': 0., 'C': 0.}))
    discrete_equality(b[monty_index], DiscreteDistribution(
        {'A': 0., 'B': 1., 'C': 0.}))
    discrete_equality(b[prize_index], DiscreteDistribution(
        {'A': 1. / 3, 'B': 0.0, 'C': 2. / 3}))
    discrete_equality(c[guest_index], DiscreteDistribution(
        {'A': 1., 'B': 0., 'C': 0.}))
    discrete_equality(c[monty_index], DiscreteDistribution(
        {'A': 0., 'B': 0., 'C': 1.}))
    discrete_equality(c[prize_index], DiscreteDistribution(
        {'A': 1. / 3, 'B': 2. / 3, 'C': 0.0}))


@with_setup(setup_monty, teardown)
def test_monty():
    a = monty_network.predict_proba({'monty': 'A'})

    discrete_equality(a[monty_index], DiscreteDistribution(
        {'A': 1.0, 'B': 0.0, 'C': 0.0}))
    discrete_equality(a[guest_index], a[prize_index])
    discrete_equality(a[guest_index], DiscreteDistribution(
        {'A': 0.0, 'B': 1. / 2, 'C': 1. / 2}))


@with_setup(setup_monty, teardown)
def test_imputation():
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
def test_numpy_imputation():
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
        assert_equal(model.log_probability(X).sum(), model2.log_probability(X).sum())
        assert_almost_equal(model.log_probability(X).sum(), logp, 4)

def test_from_structure():
    X = datasets[1]
    structure = ((1, 2), (4,), (), (), (3,))
    model = BayesianNetwork.from_structure(X, structure=structure)

    assert_equal(model.structure, structure)
    assert_almost_equal(model.log_probability(X).sum(), -344.38287, 4)

    model2 = BayesianNetwork.from_json(model.to_json())
    assert_equal(model2.structure, structure)
    assert_almost_equal(model.log_probability(X).sum(), -344.38287, 4)


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


def test_chow_liu_structure_learning():
    logps = -19.8282, -344.248785, -4842.40158, -603.2370
    for X, logp in zip(datasets, logps):
        model = BayesianNetwork.from_samples(X, algorithm='chow-liu')
        assert_almost_equal(model.log_probability(X).sum(), logp, 4)

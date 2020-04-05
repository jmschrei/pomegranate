from pomegranate import *

from nose.tools import assert_equal

def setup():
    '''
    No setup or teardown needs to be done in this case.
    '''
    pass


def teardown():
    '''
    No setup or teardown needs to be done in this case.
    '''
    pass

def test_json():
    d1 = DiscreteDistribution({"A": 0.1, "B": 0.9})
    d2 = ConditionalProbabilityTable(
        [["A", "A", 0.1], ["A", "B", 0.9], ["B", "A", 0.7], ["B", "B", 0.3]],
        [d1])
    bayes_net = BayesianNetwork("test")
    s1 = Node(d1, name="d1")
    s2 = Node(d2, name="d2")
    bayes_net.add_states(s1, s2)
    bayes_net.add_edge(s1, s2)
    bayes_net.bake()
    fg = bayes_net.graph
    also_fg = FactorGraph.from_json(fg.to_json())
    assert_equal(fg.to_json(), also_fg.to_json())
    assert_equal(len(fg.edges), len(also_fg.edges))
    for e1, e2 in zip(fg.edges, also_fg.edges):
        assert_equal(e1[0], e2[0])
        assert_equal(e1[1], e2[1])

def test_robust_json():
    d1 = DiscreteDistribution({"A": 0.1, "B": 0.9})
    d2 = ConditionalProbabilityTable(
        [["A", "A", 0.1], ["A", "B", 0.9], ["B", "A", 0.7], ["B", "B", 0.3]],
        [d1])
    bayes_net = BayesianNetwork("test")
    s1 = Node(d1, name="d1")
    s2 = Node(d2, name="d2")
    bayes_net.add_states(s1, s2)
    bayes_net.add_edge(s1, s2)
    bayes_net.bake()
    fg = bayes_net.graph
    also_fg = from_json(fg.to_json())
    assert_equal(fg.to_json(), also_fg.to_json())
    assert_equal(len(fg.edges), len(also_fg.edges))
    for e1, e2 in zip(fg.edges, also_fg.edges):
        assert_equal(e1[0], e2[0])
        assert_equal(e1[1], e2[1])

# test_hmm_gmm.py
# Authors: Nelson Liu <nelson@nelsonliu.me>
#
'''
These are unit tests for a mixture of HMMs, or a GMM with HMM emissions.
'''

from __future__ import division

from pomegranate import (GeneralMixtureModel,
                         NormalDistribution,
                         HiddenMarkovModel,
                         State)
from nose.tools import with_setup
from nose.tools import assert_greater
from nose.tools import assert_equal
from nose.tools import assert_raises
import random
import numpy as np

nan = np.nan
np.random.seed(0)
random.seed(0)

def setup_hmm_gmm():
    # Build a GMM with HMM emissions
    global hmm_a, hmm_b, gmm
    hmm_a = build_hmm(1, 7, "hmm_a")
    hmm_b = build_hmm(3, 10, "hmm_b")
    gmm = GeneralMixtureModel([hmm_a, hmm_b])

def teardown():
    pass

def build_hmm(mean_a, mean_b, name):
    d1 = State(NormalDistribution(mean_a, 1), "d1")
    d2 = State(NormalDistribution(mean_b, 1), "d2")

    hmm = HiddenMarkovModel(name)
    hmm.add_states(d1, d2)

    hmm.add_transition(hmm.start, d1, 0.9)
    hmm.add_transition(hmm.start, d2, 0.1)
    hmm.add_transition(d1, d2, 0.65)
    hmm.add_transition(d1, d1, 0.20)
    hmm.add_transition( d1, hmm.end, 0.15)
    hmm.add_transition(d2, d1, 0.20)
    hmm.add_transition(d2, d2, 0.70)
    hmm.add_transition(d2, hmm.end, 0.10)

    hmm.bake()
    return hmm

@with_setup(setup_hmm_gmm, teardown)
def test_hmm_gmm_prior():
    hmm_a_sample = [hmm_a.sample() for i in range(5)]
    hmm_b_sample = [hmm_b.sample() for i in range(10)]
    X = hmm_a_sample + hmm_b_sample
    probs = gmm.predict_proba(X)
    for i in probs[:5]:
        assert_greater(i[0], i[1])
    for i in probs[5:]:
        assert_greater(i[1], i[0])

@with_setup(setup_hmm_gmm, teardown)
def test_hmm_gmm_posterior():
    hmm_a_sample = [hmm_a.sample() for i in range(1000)]
    hmm_b_sample = [hmm_b.sample() for i in range(2000)]
    X = hmm_a_sample + hmm_b_sample
    gmm.fit(X, verbose=True)
    print np.exp(gmm.weights)
    np.testing.assert_array_almost_equal(np.exp(gmm.weights),
                                         np.array([0.33, 0.66]),
                                         decimal=2)

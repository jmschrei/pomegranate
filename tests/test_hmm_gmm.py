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
    hmm.add_transition(d1, hmm.end, 0.15)
    hmm.add_transition(d2, d1, 0.20)
    hmm.add_transition(d2, d2, 0.70)
    hmm.add_transition(d2, hmm.end, 0.10)

    hmm.bake()
    return hmm


@with_setup(setup_hmm_gmm, teardown)
def test_hmm_gmm_log_probability():
    np.testing.assert_array_almost_equal(gmm.log_probability([5]),
                                         np.array([-4.09641629]))


@with_setup(setup_hmm_gmm, teardown)
def test_hmm_gmm_predict_proba():
    np.testing.assert_array_almost_equal(gmm.predict_proba([[5], [7]]),
                                         np.array([[0.071109, 0.928891],
                                                   [0.984603, 0.015397]]))


@with_setup(setup_hmm_gmm, teardown)
def test_hmm_gmm_predict_log_proba():
    np.testing.assert_array_almost_equal(gmm.predict_log_proba([[5], [7]]),
                                         np.array([[-2.64354, -0.073764],
                                                   [-0.015517, -4.173585]]))


@with_setup(setup_hmm_gmm, teardown)
def test_hmm_gmm_fit():
    X = [[1], [7], [8], [2]]
    gmm.fit(X, verbose=True)
    np.testing.assert_array_almost_equal(gmm.predict_log_proba([[5], [7]]),
                                         np.array([[0., 0.928891],
                                                   [0., 0.015397]]))

from __future__ import (division)

from pomegranate import *
from pomegranate.parallel import log_probability
from pomegranate.io import SequenceGenerator

from nose.tools import with_setup
from nose.tools import assert_almost_equal
from nose.tools import assert_equal
from nose.tools import assert_not_equal
from nose.tools import assert_less_equal
from nose.tools import assert_raises
from nose.tools import assert_greater
from numpy.testing import assert_array_almost_equal
from numpy.testing import assert_array_equal

import pickle
import random
import numpy
import time

numpy.random.seed(0)
random.seed(0)

inf = float("inf")
nan = numpy.nan

def sparse_model(d1, d2, d3, i_d):
    model = HiddenMarkovModel("Global Alignment")

    # Create the insert states
    i0 = State(i_d, name="I0")
    i1 = State(i_d, name="I1")
    i2 = State(i_d, name="I2")
    i3 = State(i_d, name="I3")

    # Create the match states
    m1 = State(d1, name="M1")
    m2 = State(d2, name="M2")
    m3 = State(d3, name="M3")

    # Create the delete states
    d1 = State(None, name="D1")
    d2 = State(None, name="D2")
    d3 = State(None, name="D3")

    # Add all the states to the model
    model.add_states(i0, i1, i2, i3, m1, m2, m3, d1, d2, d3)

    # Create transitions from match states
    model.add_transition(model.start, m1, 0.9)
    model.add_transition(model.start, i0, 0.1)
    model.add_transition(m1, m2, 0.9)
    model.add_transition(m1, i1, 0.05)
    model.add_transition(m1, d2, 0.05)
    model.add_transition(m2, m3, 0.9)
    model.add_transition(m2, i2, 0.05)
    model.add_transition(m2, d3, 0.05)
    model.add_transition(m3, model.end, 0.9)
    model.add_transition(m3, i3, 0.1)

    # Create transitions from insert states
    model.add_transition(i0, i0, 0.70)
    model.add_transition(i0, d1, 0.15)
    model.add_transition(i0, m1, 0.15)

    model.add_transition(i1, i1, 0.70)
    model.add_transition(i1, d2, 0.15)
    model.add_transition(i1, m2, 0.15)

    model.add_transition(i2, i2, 0.70)
    model.add_transition(i2, d3, 0.15)
    model.add_transition(i2, m3, 0.15)

    model.add_transition(i3, i3, 0.85)
    model.add_transition(i3, model.end, 0.15)

    # Create transitions from delete states
    model.add_transition(d1, d2, 0.15)
    model.add_transition(d1, i1, 0.15)
    model.add_transition(d1, m2, 0.70)

    model.add_transition(d2, d3, 0.15)
    model.add_transition(d2, i2, 0.15)
    model.add_transition(d2, m3, 0.70)

    model.add_transition(d3, i3, 0.30)
    model.add_transition(d3, model.end, 0.70)

    # Call bake to finalize the structure of the model.
    model.bake()
    return model


def setup():
    global model

    i_d = DiscreteDistribution({ 'A': 0.25, 'C': 0.25, 'G': 0.25, 'T': 0.25 })
    d1 = DiscreteDistribution({ "A": 0.95, 'C': 0.01, 'G': 0.01, 'T': 0.02 })
    d2 = DiscreteDistribution({ "A": 0.003, 'C': 0.99, 'G': 0.003, 'T': 0.004 })
    d3 = DiscreteDistribution({ "A": 0.01, 'C': 0.01, 'G': 0.01, 'T': 0.97 })

    model = sparse_model(d1, d2, d3, i_d)


def setup_multivariate_discrete_sparse():
    global model

    i1 = DiscreteDistribution({'A': 0.25, 'C': 0.25, 'G': 0.25, 'T': 0.25})
    i2 = DiscreteDistribution({'A': 0.25, 'C': 0.25, 'G': 0.25, 'T': 0.25})
    i_d = IndependentComponentsDistribution([i1, i2])

    d11 = DiscreteDistribution({ "A": 0.95, 'C': 0.01, 'G': 0.01, 'T': 0.02 })
    d12 = DiscreteDistribution({ "A": 0.92, 'C': 0.02, 'G': 0.02, 'T': 0.03 })

    d21 = DiscreteDistribution({ "A": 0.005, 'C': 0.96, 'G': 0.005, 'T': 0.003 })
    d22 = DiscreteDistribution({ "A": 0.003, 'C': 0.99, 'G': 0.003, 'T': 0.004 })

    d31 = DiscreteDistribution({ "A": 0.01, 'C': 0.01, 'G': 0.01, 'T': 0.97 })
    d32 = DiscreteDistribution({ "A": 0.05, 'C': 0.03, 'G': 0.02, 'T': 0.90 })

    d1 = IndependentComponentsDistribution([d11, d12])
    d2 = IndependentComponentsDistribution([d21, d22])
    d3 = IndependentComponentsDistribution([d31, d32])

    model = sparse_model(d1, d2, d3, i_d)


def setup_multivariate_gaussian_sparse():
    global model

    i1 = UniformDistribution(-20, 20)
    i2 = UniformDistribution(-20, 20)
    i_d = IndependentComponentsDistribution([i1, i2])

    d11 = NormalDistribution(5, 1)
    d12 = NormalDistribution(7, 1)

    d21 = NormalDistribution(13, 1)
    d22 = NormalDistribution(17, 1)

    d31 = NormalDistribution(-2, 1)
    d32 = NormalDistribution(-5, 1)

    d1 = IndependentComponentsDistribution([d11, d12])
    d2 = IndependentComponentsDistribution([d21, d22])
    d3 = IndependentComponentsDistribution([d31, d32])

    model = sparse_model(d1, d2, d3, i_d)


def dense_model(d1, d2, d3, d4):
    s1 = State(d1, "s1")
    s2 = State(d2, "s2")
    s3 = State(d3, "s3")
    s4 = State(d4, "s4")

    model = HiddenMarkovModel()
    model.add_states(s1, s2, s3, s4)
    model.add_transition(model.start, s1, 0.1)
    model.add_transition(model.start, s2, 0.3)
    model.add_transition(model.start, s3, 0.2)
    model.add_transition(model.start, s4, 0.4)
    model.add_transition(s1, s1, 0.5)
    model.add_transition(s1, s2, 0.1)
    model.add_transition(s1, s3, 0.1)
    model.add_transition(s1, s4, 0.2)
    model.add_transition(s2, s1, 0.2)
    model.add_transition(s2, s2, 0.1)
    model.add_transition(s2, s3, 0.4)
    model.add_transition(s2, s4, 0.2)
    model.add_transition(s3, s1, 0.1)
    model.add_transition(s3, s2, 0.1)
    model.add_transition(s3, s3, 0.3)
    model.add_transition(s3, s4, 0.4)
    model.add_transition(s4, s1, 0.2)
    model.add_transition(s4, s2, 0.2)
    model.add_transition(s4, s3, 0.1)
    model.add_transition(s4, s4, 0.4)
    model.add_transition(s1, model.end, 0.1)
    model.add_transition(s2, model.end, 0.1)
    model.add_transition(s3, model.end, 0.1)
    model.add_transition(s4, model.end, 0.1)
    model.bake()
    return model


def setup_univariate_discrete_dense():
    global model

    d1 = DiscreteDistribution({'A': 0.90, 'B': 0.02, 'C': 0.03, 'D': 0.05})
    d2 = DiscreteDistribution({'A': 0.02, 'B': 0.90, 'C': 0.03, 'D': 0.05})
    d3 = DiscreteDistribution({'A': 0.03, 'B': 0.02, 'C': 0.90, 'D': 0.05})
    d4 = DiscreteDistribution({'A': 0.05, 'B': 0.02, 'C': 0.03, 'D': 0.90})

    model = dense_model(d1, d2, d3, d4)


def setup_univariate_gaussian_dense():
    global model

    d1 = NormalDistribution(5, 1)
    d2 = NormalDistribution(1, 1)
    d3 = NormalDistribution(13, 2)
    d4 = NormalDistribution(16, 0.5)

    model = dense_model(d1, d2, d3, d4)


def setup_univariate_poisson_dense():
    global model

    d1 = PoissonDistribution(12.1)
    d2 = PoissonDistribution(8.7)
    d3 = PoissonDistribution(1)
    d4 = PoissonDistribution(5)

    model = dense_model(d1, d2, d3, d4)


def setup_multivariate_mixed_dense():
    global model

    d11 = NormalDistribution(1, 1)
    d12 = ExponentialDistribution(5)
    d13 = LogNormalDistribution(0.5, 0.78)
    d14 = NormalDistribution(0.4, 0.8)
    d15 = PoissonDistribution(4)
    d1 = IndependentComponentsDistribution([d11, d12, d13, d14, d15])

    d21 = NormalDistribution(3, 1)
    d22 = ExponentialDistribution(35)
    d23 = LogNormalDistribution(1.8, 1.33)
    d24 = NormalDistribution(0.5, 0.7)
    d25 = PoissonDistribution(6)
    d2 = IndependentComponentsDistribution([d21, d22, d23, d24, d25])

    d31 = NormalDistribution(5, 1)
    d32 = ExponentialDistribution(7)
    d33 = LogNormalDistribution(1.2, 0.38)
    d34 = NormalDistribution(0.2, 0.8)
    d35 = PoissonDistribution(8)
    d3 = IndependentComponentsDistribution([d31, d32, d33, d34, d35])

    d41 = NormalDistribution(9, 1)
    d42 = ExponentialDistribution(15)
    d43 = LogNormalDistribution(2.2, 1.0)
    d44 = NormalDistribution(2.3, 0.2)
    d45 = PoissonDistribution(10)
    d4 = IndependentComponentsDistribution([d41, d42, d43, d44, d45])

    model = dense_model(d1, d2, d3, d4)


def setup_multivariate_gaussian_dense():
    global model

    random_state = numpy.random.RandomState(0)
    mu = random_state.normal(0, 1, size=(4, 5))
    d1 = MultivariateGaussianDistribution(mu[0], numpy.eye(5))
    d2 = MultivariateGaussianDistribution(mu[1], numpy.eye(5))
    d3 = MultivariateGaussianDistribution(mu[2], numpy.eye(5))
    d4 = MultivariateGaussianDistribution(mu[3], numpy.eye(5))

    model = dense_model(d1, d2, d3, d4)


def setup_general_mixture_gaussian():
    global model

    # should be able to pass list of weights
    gmm1 = GeneralMixtureModel([NormalDistribution(5, 2), NormalDistribution(1, 2)], weights=[0.33, 0.67])
    gmm2 = GeneralMixtureModel([NormalDistribution(3, 2), NormalDistribution(-1, 2)], weights=numpy.array([0.67, 0.33]))
    s1 = State(gmm1, "s1")
    s2 = State(gmm2, "s2")
    model = HiddenMarkovModel()
    model.add_states([s1, s2])
    model.add_transition(model.start, s1, 0.5)
    model.add_transition(model.start, s2, 0.5)
    model.add_transition(s1, s1, 0.7)
    model.add_transition(s1, s2, 0.3)
    model.add_transition(s2, s2, 0.8)
    model.add_transition(s2, s1, 0.2)
    model.bake()


def teardown():
    '''
    Remove the model at the end of the unit testing. Since it is stored in a
    global variance, simply delete it.
    '''

    pass


@with_setup(setup_univariate_discrete_dense)
def test_hmm_univariate_discrete_dense_forward():
    f = model.forward(['A', 'B', 'D', 'D', 'C'])
    logp = numpy.array([[-inf, -inf, -inf, -inf, 0., -inf],
                [-2.40794561, -5.11599581, -5.11599581, -3.91202301, -inf, -4.40631933],
                [-6.89188193, -4.35987383, -8.09848286, -7.43200392, -inf, -6.52303724],
                [-8.73634472, -9.47926612, -8.22377759, -5.87605232, -inf, -8.01305991],
                [-10.28388158, -10.39501067, -10.80077009, -6.76858029, -inf, -8.99969597],
                [-11.780373, -11.84820305, -9.00308599, -11.14654124, -inf, -11.09251037]])

    assert_array_almost_equal(f, logp)


@with_setup(setup_univariate_gaussian_dense)
def test_hmm_univariate_gaussian_dense_forward():
    f = model.forward([3, 5, 8, 19, 13])
    logp = numpy.array([[-inf, -inf, -inf, -inf, 0.0, -inf],
        [-5.221523626198319, -4.122911337530209, -15.72152362619832, -339.14208208451845, -inf, -6.137807473983832],
        [-6.045149476287824, -15.056746007188107, -14.571238720950003, -247.67044476202696, -inf, -8.347414404915495],
        [-12.157146753448874, -33.766352938119766, -13.083738234853534, -135.8798604314077, -inf, -14.126191869688759],
        [-111.69303081629936, -177.04513040289305, -19.78896563212587, -31.409154369913637, -inf, -22.091541742268795],
        [-55.01047129270264, -95.01047129270263, -22.605021155923353, -38.93103873379323, -inf, -24.90760616769035]])

    assert_array_almost_equal(f, logp)


@with_setup(setup_univariate_poisson_dense)
def test_hmm_univariate_poisson_dense_forward():
    f = model.forward([5, 8, 2, 4, 7, 8, 2])
    logp = numpy.array([[-inf, -inf, -inf, -inf, 0.0, -inf],
        [-6.724049572762615, -3.874849418805291, -7.396929655216146, -2.6565929124856993, -inf, -4.680333421931903],
        [-6.730147189571273, -6.114939268804046, -15.76343567997591, -6.1491172641470415, -inf, -7.498440536740427],
        [-14.331825716875938, -12.23889386200279, -8.404629462451046, -8.953498331843619, -inf, -10.236032891838605],
        [-15.218653579546471, -13.152885951823485, -13.58596603773139, -10.59764914181018, -inf, -12.771059542322048],
        [-15.25983657200538, -14.222315575321783, -22.039008189195016, -13.683075848076015, -inf, -15.403406295745581],
        [-17.30957501126636, -16.957613004675117, -26.32610484066788, -16.995462506806398, -inf, -18.279525087145316],
        [-25.05974969431663, -23.03771236899512, -19.218787612604835, -19.75231189091333, -inf, -21.044274521213687]])

    assert_array_almost_equal(f, logp)


@with_setup(setup_multivariate_mixed_dense)
def test_hmm_multivariate_mixed_dense_forward():
    f = model.forward([[0, 1, 5, 2, 3], [2, 4, 1, 5, 6], [4, 6, 2, 0, 1]])
    logp = numpy.array([[ -inf,  -inf,  -inf,  -inf,  0.,  -inf],
         [ -14.73222089,  -46.16604623,  -29.00420739,  -62.64844211, -inf, -17.03480535],
         [ -55.5993327,  -180.07255998,  -74.2437335,  -195.58972567, -inf, -57.90191778],
         [ -94.93076444, -273.03467427, -107.55659346, -233.79896288, -inf, -97.23334625]])

    assert_array_almost_equal(f, logp)


@with_setup(setup_multivariate_gaussian_dense)
def test_hmm_multivariate_gaussian_dense_forward():
    f = model.forward([[0, 1, 5, 2, 3], [2, 4, 1, 5, 6], [-4, 6, -2, 0, 1]])
    logp = numpy.array([[-inf, -inf, -inf, -inf, 0.0, -inf],
        [-17.388625105797296, -25.109952452723487, -20.33305760532214, -28.085443290422877, -inf, -19.639474084287432],
        [-41.518178585896166, -62.62215498003203, -56.463304390840634, -63.21955480161304, -inf, -43.820763354673296],
        [-86.42085732368164, -67.62244823776697, -71.30752972353059, -70.64290643117542, -inf, -69.85376066028503]])

    assert_array_almost_equal(f, logp)


@with_setup(setup_univariate_discrete_dense)
def test_hmm_univariate_discrete_dense_nan_forward():
    f = model.forward(['A', nan, 'D', nan, 'C'])
    logp = numpy.array([[       -inf,        -inf,        -inf,        -inf,  0.,                -inf],
         [-2.40794561, -5.11599581, -5.11599581, -3.91202301,        -inf, -4.40631933],
         [-2.97985892, -4.25451331, -4.18645985, -3.51998092,        -inf, -4.51167984],
         [-6.32889724, -7.26872515, -6.99767999, -3.58171257,        -inf, -5.76918537],
         [-5.0073806,  -5.13193889, -5.65094338, -4.42343213,        -inf, -5.87454588],
         [-8.42983637, -9.02567905, -5.34834914, -8.24851276,        -inf, -7.53208902]])

    assert_array_almost_equal(f, logp)


@with_setup(setup_univariate_gaussian_dense)
def test_hmm_univariate_gaussian_dense_nan_forward():
    f = model.forward([3, 5, 8, nan, 13])
    logp = numpy.array([[         -inf,          -inf, -inf, -inf, 0., -inf],
         [  -5.22152363,   -4.12291134,  -15.72152363, -339.14208208, -inf, -6.13780747],
         [  -6.04514948,  -15.05674601,  -14.57123872, -247.67044476, -inf, -8.3474144 ],
         [ -12.15714675,  -33.76635294,  -13.08373823, -135.87986043, -inf, -14.12619187],
         [ -12.77409228,  -14.12619187,  -13.67687992,  -13.18336302, -inf, -14.23155239],
         [ -46.01446276,  -86.89953909,  -15.32388795,  -31.38618129, -inf, -17.62647294]])

    assert_array_almost_equal(f, logp)


@with_setup(setup_univariate_poisson_dense)
def test_hmm_univariate_poisson_dense_nan_forward():
    f = model.forward([5, 8, 2, nan, 7, nan, 2])
    logp = numpy.array([[        -inf,         -inf, -inf, -inf, 0., -inf],
         [ -6.72404957,  -3.87484942,  -7.39692966,  -2.65659291, -inf, -4.68033342],
         [ -6.73014719,  -6.11493927, -15.76343568,  -6.14911726, -inf, -7.49844054],
         [-14.33182572, -12.23889386,  -8.40462946,  -8.95349833, -inf, -10.23603289],
         [ -9.91342156,  -9.92812422,  -9.40791221,  -8.85734696, -inf, -10.34139341],
         [-12.72297734, -12.0579037,  -19.19249097, -11.37941729, -inf, -13.11175535],
         [-12.21880511, -12.66362307, -12.49060859, -11.97032983, -inf, -13.21711586],
         [-20.06552788, -17.98505855, -14.24996743, -14.55695817, -inf, -15.98581404]])

    assert_array_almost_equal(f, logp)


@with_setup(setup_multivariate_mixed_dense)
def test_hmm_multivariate_mixed_dense_nan_forward():
    f = model.forward([[0, nan, 5, nan, 3], [nan, 4, 1, 5, 6], [4, nan, 2, nan, 1]])
    logp = numpy.array([[-inf, -inf, -inf, -inf, 0.0, -inf],
        [-8.64586382, -11.86321233, -20.72307256, -49.92199169, -inf, -10.90916394],
        [-48.07813868, -172.52798004, -62.58979428, -164.0451409, -inf, -50.38072328],
        [-58.19821326, -58.2514622, -59.25426314, -73.55131746, -inf, -59.66964241]])

    assert_array_almost_equal(f, logp)


@with_setup(setup_multivariate_gaussian_dense)
def test_hmm_multivariate_gaussian_dense_nan_forward():
    f = model.forward([[0, nan, 5, nan, 3], [nan, 4, 1, 5, 6], [-4, nan, -2, nan, 1]])
    logp = numpy.array([[        -inf,         -inf,        -inf, -inf,   0.,  -inf],
         [-15.34182759, -21.05906503, -16.62794596, -24.70263887, -inf, -17.39777437],
         [-38.48018922, -55.02937919, -51.31243927, -58.5238134,  -inf, -40.78277157],
         [-63.35506959, -49.99044883, -56.0924389,  -55.56636784, -inf, -52.28702405]])

    assert_array_almost_equal(f, logp)


@with_setup(setup_univariate_discrete_dense)
def test_hmm_univariate_discrete_dense_backward():
    f = model.backward(['A', 'B', 'D', 'D', 'C'])
    logp = numpy.array([[-9.86805902419294, -10.666561769922483, -11.09973677168472, -10.617074536069564, -11.092510372852566, -inf],
        [-9.120551817416588, -9.07513780778706, -9.061129343592423, -8.517934491110973, -8.143527673240188, -inf],
        [-6.950743680969357, -6.918207210615943, -6.3171849919534315, -6.328223424806821, -6.314201940019015, -inf],
        [-5.926238762190273, -5.832923518979429, -5.343210135438772, -5.352351255040521, -5.296020007499352, -inf],
        [-4.474141923581687, -3.2834143460057716, -3.547379891840237, -4.474141923581686, -3.8922203781319653, -inf],
        [-2.3025850929940455, -2.3025850929940455, -2.3025850929940455, -2.3025850929940455, -inf, 0.0]])

    assert_array_almost_equal(f, logp)


@with_setup(setup_univariate_gaussian_dense)
def test_hmm_univariate_gaussian_dense_backward():
    f = model.backward([3, 5, 8, 19, 13])
    logp = numpy.array([[-24.010022764471987, -24.820878919065986, -25.359784144328874, -24.666641886986977, -24.907606167690343, -inf],
        [-20.47495458748052, -21.390489786220005, -22.08295469697486, -21.390527588974052, -22.081667004696868, -inf],
        [-18.86308443640411, -18.00715991329825, -18.32109321121691, -19.183852463904262, -18.700307092256082, -inf],
        [-13.533310680731095, -12.147019061523556, -12.434699610689767, -13.533307024859651, -12.840163500171148, -inf],
        [-6.217255777912353, -4.830961508172448, -5.1186435298576365, -6.217255656072613, -5.524108597352521, -inf],
        [-2.3025850929940455, -2.3025850929940455, -2.3025850929940455, -2.3025850929940455, -inf, 0.0]])

    assert_array_almost_equal(f, logp)


@with_setup(setup_univariate_poisson_dense)
def test_hmm_univariate_poisson_dense_backward():
    f = model.backward([5, 8, 2, 4, 7, 8, 2])
    logp = numpy.array([[-21.691907586187032, -21.73799328948721, -21.138366991878907, -21.083291604178275, -21.044274521213687, -inf],
        [-18.8959690490499, -19.147279755417475, -18.96895836891973, -18.554384880883024, -18.344028493972242, -inf],
        [-16.436767297608355, -15.50289984292892, -15.520463678389667, -16.044844087024796, -15.741349651348344, -inf],
        [-13.746179478239421, -13.67833968609671, -13.103892290217432, -13.104515632505755, -13.064497897488891, -inf],
        [-10.900016194765097, -11.134592598694933, -10.70914236553333, -10.534402907712945, -10.472655441966987, -inf],
        [-8.08666946806842, -8.338743855593863, -8.159657556177146, -7.746210358522981, -7.536781914211483, -inf],
        [-5.624801928499422, -4.698024820874441, -4.71562372030326, -5.232043004902404, -4.927999971986549, -inf],
        [-2.3025850929940455, -2.3025850929940455, -2.3025850929940455, -2.3025850929940455, -inf, 0.0]])

    assert_array_almost_equal(f, logp)


@with_setup(setup_multivariate_mixed_dense)
def test_hmm_multivariate_mixed_dense_backward():
    f = model.backward([[0, 1, 5, 2, 3], [2, 4, 1, 5, 6], [4, 6, 2, 0, 1]])
    logp = numpy.array([[-95.62390845, -96.54019907, -97.23334619, -96.54019916, -97.23334625, -inf],
        [-82.50112549, -83.41741621, -84.11056338, -83.41741622, -84.11056339, -inf],
        [-41.63401355, -42.55027471, -43.24340546, -42.55029936, -43.24342189, -inf],
        [-2.30258509, -2.30258509, -2.30258509, -2.30258509, -inf, 0.0]])


    assert_array_almost_equal(f, logp)

@with_setup(setup_multivariate_gaussian_dense)
def test_hmm_multivariate_gaussian_dense_backward():
    f = model.backward([[0, 1, 5, 2, 3], [2, 4, 1, 5, 6], [-4, 6, -2, 0, 1]])
    logp = numpy.array([[-68.2539137567533, -69.16076639610863, -69.84868309756291, -69.16857768159394, -69.85376066028503, -inf],
        [-52.47579136451719, -53.392079325925124, -54.08522496164164, -53.392081629466915, -54.08522649261687, -inf],
        [-28.335582428803374, -28.267824434424867, -28.247424322493245, -27.65418843714621, -27.260167817399534, -inf],
        [-2.3025850929940455, -2.3025850929940455, -2.3025850929940455, -2.3025850929940455, -inf, 0.0]])
    assert_array_almost_equal(f, logp)


@with_setup(setup_univariate_discrete_dense)
def test_hmm_univariate_discrete_dense_nan_backward():
    f = model.backward(['A', nan, 'D', nan, 'C'])
    logp = numpy.array([[-6.2351892, -7.03937052, -7.53995041, -7.0284468, -7.53208902, -inf],
        [-5.47529921, -5.28136881, -5.22459211, -5.34120086, -5.21037792, -inf],
        [-5.61180968, -5.59762678, -5.01422924, -5.0155398, -4.99812389, -inf],
        [-4.22604905, -3.92055934, -4.01239598, -4.06168378, -3.78494992, -inf],
        [-4.47414192, -3.28341435, -3.54737989, -4.47414192, -3.89222038, -inf],
        [-2.30258509, -2.30258509, -2.30258509, -2.30258509, -inf, 0.0]])

    assert_array_almost_equal(f, logp)


@with_setup(setup_univariate_gaussian_dense)
def test_hmm_univariate_gaussian_dense_nan_backward():
    f = model.backward([3, 5, 8, nan, 13])
    logp = numpy.array([[-16.72877261, -17.53965174, -18.0785865, -17.38544424, -17.62647294, -inf],
        [-13.19368605, -14.10946925, -14.80215433, -14.10948639, -14.80126502, -inf],
        [-11.58174695, -11.10331706, -11.44733525, -12.0888188, -11.79646424, -inf],
        [-5.88078356, -5.52410863, -5.62946913, -5.68662755, -5.38434669, -inf],
        [-6.21725578, -4.83096151, -5.11864353, -6.21725566, -5.5241086, -inf],
        [-2.30258509, -2.30258509, -2.30258509, -2.30258509, -inf, 0.0]])

    assert_array_almost_equal(f, logp)


@with_setup(setup_univariate_poisson_dense)
def test_hmm_univariate_poisson_dense_nan_backward():
    f = model.backward([5, 8, 2, nan, 7, nan, 2])
    logp = numpy.array([[-16.63348003, -16.67956771, -16.0798998, -16.02482554, -15.98581404, -inf],
        [-13.83793763, -14.08893409, -13.91233676, -13.49588981, -13.28439885, -inf],
        [-11.37983721, -10.43981663, -10.46546269, -10.99407821, -10.68304212, -inf],
        [-8.14995891, -8.11105833, -8.03603323, -8.07865567, -7.99445884, -inf],
        [-8.13575931, -8.359288, -8.01322363, -7.75426045, -7.62580822, -inf],
        [-5.36468956, -5.06990563, -5.10022463, -5.18918048, -4.95862159, -inf],
        [-5.62480193, -4.69802482, -4.71562372, -5.232043, -4.92799997, -inf],
        [-2.30258509, -2.30258509, -2.30258509, -2.30258509, -inf, 0.0]])

    assert_array_almost_equal(f, logp)


@with_setup(setup_multivariate_mixed_dense)
def test_hmm_multivariate_mixed_dense_nan_backward():
    f = model.backward([[0, nan, 5, nan, 3], [nan, 4, 1, 5, 6], [4, nan, 2, nan, 1]])
    logp = numpy.array([[-58.07503525, -58.9897255, -59.68021202, -58.98706625, -59.66964241, -inf],
        [-51.03967719, -51.95596417, -52.64910927, -51.95596729, -52.64911135, -inf],
        [-11.5915042, -11.41490257, -11.63801707, -11.45032367, -11.10355162, -inf],
        [-2.30258509, -2.30258509, -2.30258509, -2.30258509, -inf, 0.0]])

    assert_array_almost_equal(f, logp)


@with_setup(setup_multivariate_gaussian_dense)
def test_hmm_multivariate_gaussian_dense_nan_backward():
    f = model.backward([[0, nan, 5, nan, 3], [nan, 4, 1, 5, 6], [-4, nan, -2, nan, 1]])
    logp = numpy.array([[-50.72705835, -51.59493939, -52.26201549, -51.63477927, -52.28702405, -inf],
        [-37.00027105, -37.91654779, -38.60968708, -37.91655924, -38.60969427, -inf],
        [-13.80683759, -13.80018574, -13.79865713, -13.11480477, -12.71022551, -inf],
        [-2.30258509, -2.30258509, -2.30258509, -2.30258509, -inf, 0.0]])

    assert_array_almost_equal(f, logp)


@with_setup(setup_univariate_discrete_dense)
def test_hmm_univariate_discrete_dense_predict_log_proba():
    f = model.predict_log_proba(['A', 'B', 'D', 'D', 'C'])
    logp = numpy.array([[-0.43598705, -3.09862324, -3.08461478, -1.33744712],
        [-2.75011524, -0.18557067, -3.32315748, -2.66771698],
        [-3.57007311, -4.21967926, -2.47447735, -0.1358932],
        [-3.66551313, -2.58591465, -3.25563961, -0.15021184],
        [-2.99044772, -3.05827777, -0.21316071, -2.35661596]])

    assert_array_almost_equal(f, logp)


@with_setup(setup_univariate_gaussian_dense)
def test_hmm_univariate_gaussian_dense_predict_log_proba():
    f = model.predict_log_proba([3, 5, 8, 19, 13])
    logp = numpy.array([[-0.78887205, -0.60579496, -12.89687216, -335.62500351],
        [-0.00062775, -8.15629975, -7.98472576, -241.94669106],
        [-0.78285127, -21.00576583, -0.61083168, -124.50556129],
        [-93.00268043, -156.96848574, -2.99e-06, -12.71880386],
        [-32.40545022, -72.40545022, -8e-08, -16.32601766]])

    assert_array_almost_equal(f, logp)


@with_setup(setup_univariate_poisson_dense)
def test_hmm_univariate_poisson_dense_predict_log_proba():
    f = model.predict_log_proba([5, 8, 2, 4, 7, 8, 2])
    logp = numpy.array([[-4.5757441, -1.97785465, -5.3216135, -0.16670327],
        [-2.12263997, -0.57356459, -10.23962484, -1.14968683],
        [-7.03373067, -4.87295903, -0.46424723, -1.01373944],
        [-5.07439525, -3.24320403, -3.25083388, -0.08777753],
        [-2.30223152, -1.51678491, -9.15439122, -0.38501169],
        [-1.89010242, -0.6113633, -9.99745404, -1.18323099],
        [-6.31806027, -4.29602294, -0.47709818, -1.01062246]])

    assert_array_almost_equal(f, logp)


@with_setup(setup_multivariate_mixed_dense)
def test_hmm_multivariate_mixed_dense_predict_log_proba():
    f = model.predict_log_proba([[0, 1, 5, 2, 3], [2, 4, 1, 5, 6], [4, 6, 2, 0, 1]])
    logp = numpy.array([[-1.3e-07, -32.35011618, -15.88142452, -48.83251208],
        [-0.0, -125.38948844, -20.25379271, -140.90667878],
        [-3.29e-06, -178.10391312, -12.6258323, -138.86820172]])

    assert_array_almost_equal(f, logp)

@with_setup(setup_multivariate_mixed_dense)
def test_hmm_multivariate_mixed_dense_predict_log_proba_from_json():
    logp = numpy.array([[-1.3e-07, -32.35011618, -15.88142452, -48.83251208],
        [-0.0, -125.38948844, -20.25379271, -140.90667878],
        [-3.29e-06, -178.10391312, -12.6258323, -138.86820172]])

    s = [[0, 1, 5, 2, 3], [2, 4, 1, 5, 6], [4, 6, 2, 0, 1]]
    hmm_json = HiddenMarkovModel.from_json(model.to_json())
    f = hmm_json.predict_log_proba(s)
    assert_array_almost_equal(f, logp)

@with_setup(setup_multivariate_mixed_dense)
def test_hmm_multivariate_mixed_dense_predict_log_proba_from_yaml():
    logp = numpy.array([[-1.3e-07, -32.35011618, -15.88142452, -48.83251208],
        [-0.0, -125.38948844, -20.25379271, -140.90667878],
        [-3.29e-06, -178.10391312, -12.6258323, -138.86820172]])

    s = [[0, 1, 5, 2, 3], [2, 4, 1, 5, 6], [4, 6, 2, 0, 1]]
    hmm_yaml = HiddenMarkovModel.from_yaml(model.to_yaml())
    f = hmm_yaml.predict_log_proba(s)
    assert_array_almost_equal(f, logp)

@with_setup(setup_multivariate_mixed_dense)
def test_hmm_multivariate_mixed_dense_robust_from_json():
    logp = numpy.array([[-1.3e-07, -32.35011618, -15.88142452, -48.83251208],
        [-0.0, -125.38948844, -20.25379271, -140.90667878],
        [-3.29e-06, -178.10391312, -12.6258323, -138.86820172]])

    s = [[0, 1, 5, 2, 3], [2, 4, 1, 5, 6], [4, 6, 2, 0, 1]]
    hmm_json = from_json(model.to_json())
    f = hmm_json.predict_log_proba(s)
    assert_array_almost_equal(f, logp)

@with_setup(setup_multivariate_gaussian_dense)
def test_hmm_multivariate_gaussian_dense_predict_log_proba():
    f = model.predict_log_proba([[0, 1, 5, 2, 3], [2, 4, 1, 5, 6], [-4, 6, -2, 0, 1]])
    logp = numpy.array([[-0.01065581, -8.64827112, -4.56452191, -11.62376426],
        [-3.5e-07, -21.03621875, -14.85696805, -21.01998258],
        [-18.86968176, -0.07127267, -3.75635416, -3.09173086]])

    assert_array_almost_equal(f, logp)


@with_setup(setup_multivariate_gaussian_dense)
def test_hmm_multivariate_gaussian_dense_predict_log_proba_from_json():
    logp = numpy.array([[-0.01065581, -8.64827112, -4.56452191, -11.62376426],
        [-3.5e-07, -21.03621875, -14.85696805, -21.01998258],
        [-18.86968176, -0.07127267, -3.75635416, -3.09173086]])

    s = [[0, 1, 5, 2, 3], [2, 4, 1, 5, 6], [-4, 6, -2, 0, 1]]
    hmm_json = HiddenMarkovModel.from_json(model.to_json())
    f = hmm_json.predict_log_proba(s)
    assert_array_almost_equal(f, logp)

@with_setup(setup_multivariate_gaussian_dense)
def test_hmm_multivariate_gaussian_dense_predict_log_proba_from_yaml():
    logp = numpy.array([[-0.01065581, -8.64827112, -4.56452191, -11.62376426],
        [-3.5e-07, -21.03621875, -14.85696805, -21.01998258],
        [-18.86968176, -0.07127267, -3.75635416, -3.09173086]])

    s = [[0, 1, 5, 2, 3], [2, 4, 1, 5, 6], [-4, 6, -2, 0, 1]]
    hmm_yaml = HiddenMarkovModel.from_yaml(model.to_yaml())
    f = hmm_yaml.predict_log_proba(s)
    assert_array_almost_equal(f, logp)

@with_setup(setup_multivariate_gaussian_dense)
def test_hmm_multivariate_gaussian_dense_robust_from_json():
    logp = numpy.array([[-0.01065581, -8.64827112, -4.56452191, -11.62376426],
        [-3.5e-07, -21.03621875, -14.85696805, -21.01998258],
        [-18.86968176, -0.07127267, -3.75635416, -3.09173086]])

    s = [[0, 1, 5, 2, 3], [2, 4, 1, 5, 6], [-4, 6, -2, 0, 1]]
    hmm_json = from_json(model.to_json())
    f = hmm_json.predict_log_proba(s)
    assert_array_almost_equal(f, logp)

@with_setup(setup_univariate_discrete_dense)
def test_hmm_univariate_discrete_dense_nan_predict_log_proba():
    f = model.predict_log_proba(['A', nan, 'D', nan, 'C'])
    logp = numpy.array([[-0.35115579, -2.8652756, -2.8084989, -1.72113484],
        [-1.05957958, -2.32005107, -1.66860008, -1.0034317],
        [-3.02285728, -3.65719546, -3.47798695, -0.11130733],
        [-1.9494335, -0.88326422, -1.66623425, -1.36548504],
        [-3.20033245, -3.79617513, -0.11884521, -3.01900883]])

    assert_array_almost_equal(f, logp)


@with_setup(setup_univariate_gaussian_dense)
def test_hmm_univariate_gaussian_dense_nan_predict_log_proba():
    f = model.predict_log_proba([3, 5, 8, nan, 13])
    logp = numpy.array([[-0.78873673, -0.60590764, -12.89720502, -335.62509553],
        [-0.00042349, -8.53359013, -8.39210104, -242.13279062],
        [-0.41145737, -21.66398863, -1.08673442, -123.94001504],
        [-1.36487512, -1.33068044, -1.16905051, -1.77414573],
        [-30.69057492, -71.57565125, -1.1e-07, -16.06229345]])

    assert_array_almost_equal(f, logp)


@with_setup(setup_univariate_poisson_dense)
def test_hmm_univariate_poisson_dense_nan_predict_log_proba():
    f = model.predict_log_proba([5, 8, 2, nan, 7, nan, 2])
    logp = numpy.array([[-4.57617316, -1.97796947, -5.32345238, -0.16666868],
        [-2.12417036, -0.56894186, -10.24308433, -1.15738144],
        [-6.49597059, -4.36413816, -0.45484866, -1.04633997],
        [-2.06336683, -2.30159819, -1.4353218, -0.62579337],
        [-2.10185287, -1.14199529, -8.30690156, -0.58278374],
        [-1.857793, -1.37583385, -1.22041828, -1.2165588],
        [-6.38229894, -4.3018296, -0.56673849, -0.87372922]])

    assert_array_almost_equal(f, logp)


@with_setup(setup_multivariate_mixed_dense)
def test_hmm_multivariate_mixed_dense_nan_predict_log_proba():
    f = model.predict_log_proba([[0, nan, 5, nan, 3], [nan, 4, 1, 5, 6], [4, nan, 2, nan, 1]])
    logp = numpy.array([[-0.0158986, -4.14953409, -13.70253942, -42.20831658],
        [-4.8e-07, -124.2732402, -14.55816895, -115.82582217],
        [-0.83115595, -0.88440489, -1.88720582, -16.18426015]])

    assert_array_almost_equal(f, logp)


@with_setup(setup_multivariate_gaussian_dense)
def test_hmm_multivariate_gaussian_dense_nan_predict_log_proba():
    f = model.predict_log_proba([[0, nan, 5, nan, 3], [nan, 4, 1, 5, 6], [-4, nan, -2, nan, 1]])
    logp = numpy.array([[-0.05507459, -6.68858877, -2.95060899, -10.33217406],
        [-2.76e-06, -16.54254089, -12.82407236, -19.35159412],
        [-13.37063063, -0.00600988, -6.10799995, -5.58192889]])

    assert_array_almost_equal(f, logp)


@with_setup(setup_univariate_discrete_dense)
def test_hmm_univariate_discrete_dense_predict_proba():
    f = model.predict_proba(['A', 'B', 'D', 'D', 'C'])
    logp = numpy.array([[0.6466261, 0.04511127, 0.04574765, 0.26251498],
        [0.06392049, 0.83063013, 0.03603886, 0.06941051],
        [0.0281538, 0.01470336, 0.08420699, 0.87293586],
        [0.02559104, 0.07532715, 0.03855615, 0.86052566],
        [0.05026493, 0.04696852, 0.80802627, 0.09474029]])

    assert_array_almost_equal(f, logp)


@with_setup(setup_univariate_gaussian_dense)
def test_hmm_univariate_gaussian_dense_predict_proba():
    f = model.predict_proba([3, 5, 8, 19, 13])
    logp = numpy.array([[0.454357, 0.54564049, 2.51e-06, 0.0],
        [0.99937245, 0.00028692, 0.00034063, 0.0],
        [0.45710084, 0.0, 0.54289916, 0.0],
        [0.0, 0.0, 0.99999701, 2.99e-06],
        [0.0, 0.0, 0.99999992, 8e-08]])

    assert_array_almost_equal(f, logp)


@with_setup(setup_univariate_poisson_dense)
def test_hmm_univariate_poisson_dense_predict_proba():
    f = model.predict_proba([5, 8, 2, 4, 7, 8, 2])
    logp = numpy.array([[0.01029863, 0.13836576, 0.00488487, 0.84645074],
        [0.11971517, 0.56351316, 3.573e-05, 0.31673595],
        [0.00088164, 0.00765069, 0.62860812, 0.36285955],
        [0.00625487, 0.03903861, 0.03874189, 0.91596463],
        [0.10003536, 0.2194162, 0.00010575, 0.68044268],
        [0.15105634, 0.54261062, 4.552e-05, 0.30628753],
        [0.00180344, 0.01362263, 0.62058159, 0.36399234]])

    assert_array_almost_equal(f, logp)


@with_setup(setup_multivariate_mixed_dense)
def test_hmm_multivariate_mixed_dense_predict_proba():
    f = model.predict_proba([[0, 1, 5, 2, 3], [2, 4, 1, 5, 6], [4, 6, 2, 0, 1]])
    logp = numpy.array([[0.99999987, 0.0, 1.3e-07, 0.0],
        [1.0, 0.0, 0.0, 0.0],
        [0.99999671, 0.0, 3.29e-06, 0.0]])

    assert_array_almost_equal(f, logp)

@with_setup(setup_multivariate_gaussian_dense)
def test_hmm_multivariate_gaussian_dense_predict_proba():
    f = model.predict_proba([[0, 1, 5, 2, 3], [2, 4, 1, 5, 6], [-4, 6, -2, 0, 1]])
    logp = numpy.array([[0.98940076, 0.00017543, 0.01041486, 8.95e-06],
        [0.99999965, 0.0, 3.5e-07, 0.0],
        [1e-08, 0.93120794, 0.02336878, 0.04542326]])

    assert_array_almost_equal(f, logp)


@with_setup(setup_univariate_discrete_dense)
def test_hmm_univariate_discrete_dense_nan_predict_proba():
    f = model.predict_proba(['A', nan, 'D', nan, 'C'])
    logp = numpy.array([[0.70387409, 0.05696743, 0.06029543, 0.17886305],
        [0.3466015, 0.09826857, 0.18851078, 0.36661915],
        [0.04866198, 0.02580478, 0.03086949, 0.89466375],
        [0.14235469, 0.41343118, 0.18895729, 0.25525684],
        [0.04074865, 0.0224565, 0.88794523, 0.04884961]])

    assert_array_almost_equal(f, logp)


@with_setup(setup_univariate_gaussian_dense)
def test_hmm_univariate_gaussian_dense_nan_predict_proba():
    f = model.predict_proba([3, 5, 8, nan, 13])
    logp = numpy.array([[0.45441848, 0.54557901, 2.51e-06, 0.0],
        [0.9995766, 0.00019675, 0.00022665, 0.0],
        [0.66268377, 0.0, 0.33731623, 0.0],
        [0.25541257, 0.26429736, 0.31066177, 0.1696283],
        [0.0, 0.0, 0.99999989, 1.1e-07]])

    assert_array_almost_equal(f, logp)


@with_setup(setup_univariate_poisson_dense)
def test_hmm_univariate_poisson_dense_nan_predict_proba():
    f = model.predict_proba([5, 8, 2, nan, 7, nan, 2])
    logp = numpy.array([[0.01029422, 0.13834988, 0.00487589, 0.84648002],
        [0.1195321, 0.56612416, 3.56e-05, 0.31430814],
        [0.00150951, 0.01272562, 0.63454399, 0.35122088],
        [0.12702558, 0.10009874, 0.23803875, 0.53483693],
        [0.12222974, 0.31918153, 0.00024681, 0.55834192],
        [0.15601658, 0.25262885, 0.29510671, 0.29624786],
        [0.00169123, 0.01354376, 0.56737292, 0.4173921]])

    assert_array_almost_equal(f, logp)


@with_setup(setup_multivariate_mixed_dense)
def test_hmm_multivariate_mixed_dense_nan_predict_proba():
    f = model.predict_proba([[0, nan, 5, nan, 3], [nan, 4, 1, 5, 6], [4, nan, 2, nan, 1]])
    logp = numpy.array([[0.98422712, 0.01577176, 1.12e-06, 0.0],
        [0.99999952, 0.0, 4.8e-07, 0.0],
        [0.43554553, 0.41295986, 0.15149452, 9e-08]])

    assert_array_almost_equal(f, logp)


@with_setup(setup_multivariate_gaussian_dense)
def test_hmm_multivariate_gaussian_dense_nan_predict_proba():
    f = model.predict_proba([[0, nan, 5, nan, 3], [nan, 4, 1, 5, 6], [-4, nan, -2, nan, 1]])
    logp = numpy.array([[0.94641455, 0.00124504, 0.05230784, 3.257e-05],
        [0.99999724, 7e-08, 2.7e-06, 0.0],
        [1.56e-06, 0.99400815, 0.002225, 0.0037653]])

    assert_array_almost_equal(f, logp)


@with_setup(setup_univariate_discrete_dense)
def test_hmm_univariate_discrete_dense_predict():
    f = model.predict(['A', 'B', 'D', 'D', 'C'])
    path = [0, 1, 3, 3, 2]

    assert_array_almost_equal(f, path)


@with_setup(setup_univariate_gaussian_dense)
def test_hmm_univariate_gaussian_dense_predict():
    f = model.predict([3, 5, 8, 19, 13])
    path = [1, 0, 2, 2, 2]

    assert_array_almost_equal(f, path)


@with_setup(setup_univariate_poisson_dense)
def test_hmm_univariate_poisson_dense_predict():
    f = model.predict([5, 8, 2, 4, 7, 8, 2])
    path = [3, 1, 2, 3, 3, 1, 2]

    assert_array_almost_equal(f, path)


@with_setup(setup_multivariate_mixed_dense)
def test_hmm_multivariate_mixed_dense_predict():
    f = model.predict([[0, 1, 5, 2, 3], [2, 4, 1, 5, 6], [4, 6, 2, 0, 1]])
    path = [0, 0, 0]

    assert_array_almost_equal(f, path)

@with_setup(setup_multivariate_gaussian_dense)
def test_hmm_multivariate_gaussian_dense_predict():
    f = model.predict([[0, 1, 5, 2, 3], [2, 4, 1, 5, 6], [-4, 6, -2, 0, 1]])
    path = [0, 0, 1]

    assert_array_almost_equal(f, path)


@with_setup(setup_univariate_discrete_dense)
def test_hmm_univariate_discrete_dense_nan_predict():
    f = model.predict(['A', nan, 'D', nan, 'C'])
    path = [0, 3, 3, 1, 2]

    assert_array_almost_equal(f, path)


@with_setup(setup_univariate_gaussian_dense)
def test_hmm_univariate_gaussian_dense_nan_predict():
    f = model.predict([3, 5, 8, nan, 13])
    path = [1, 0, 0, 2, 2]

    assert_array_almost_equal(f, path)


@with_setup(setup_univariate_poisson_dense)
def test_hmm_univariate_poisson_dense_nan_predict():
    f = model.predict([5, 8, 2, nan, 7, nan, 2])
    path = [3, 1, 2, 3, 3, 3, 2]

    assert_array_almost_equal(f, path)


@with_setup(setup_multivariate_mixed_dense)
def test_hmm_multivariate_mixed_dense_nan_predict():
    f = model.predict([[0, nan, 5, nan, 3], [nan, 4, 1, 5, 6], [4, nan, 2, nan, 1]])
    path = [0, 0, 0]

    assert_array_almost_equal(f, path)


@with_setup(setup_multivariate_gaussian_dense)
def test_hmm_multivariate_gaussian_dense_nan_predict():
    f = model.predict([[0, nan, 5, nan, 3], [nan, 4, 1, 5, 6], [-4, nan, -2, nan, 1]])
    path = [0, 0, 1]

    assert_array_almost_equal(f, path)


@with_setup(setup_univariate_discrete_dense)
def test_hmm_univariate_discrete_dense_predict_viterbi():
    f = model.predict(['A', 'B', 'D', 'D', 'C'], algorithm='viterbi')
    path = [4, 0, 1, 3, 3, 2, 5]

    assert_array_almost_equal(f, path)


@with_setup(setup_univariate_gaussian_dense)
def test_hmm_univariate_gaussian_dense_predict_viterbi():
    f = model.predict([3, 5, 8, 19, 13], algorithm='viterbi')
    path = [4, 1, 0, 2, 2, 2, 5]

    assert_array_almost_equal(f, path)


@with_setup(setup_univariate_poisson_dense)
def test_hmm_univariate_poisson_dense_predict_viterbi():
    f = model.predict([5, 8, 2, 4, 7, 8, 2], algorithm='viterbi')
    path = [4, 3, 1, 2, 3, 3, 1, 2, 5]

    assert_array_almost_equal(f, path)


@with_setup(setup_multivariate_mixed_dense)
def test_hmm_multivariate_mixed_dense_predict_viterbi():
    f = model.predict([[0, 1, 5, 2, 3], [2, 4, 1, 5, 6], [4, 6, 2, 0, 1]],
        algorithm='viterbi')
    path = [4, 0, 0, 0, 5]

    assert_array_almost_equal(f, path)

@with_setup(setup_multivariate_gaussian_dense)
def test_hmm_multivariate_gaussian_dense_predict_viterbi():
    f = model.predict([[0, 1, 5, 2, 3], [2, 4, 1, 5, 6], [-4, 6, -2, 0, 1]],
        algorithm='viterbi')
    path = [4, 0, 0, 1, 5]

    assert_array_almost_equal(f, path)


@with_setup(setup_univariate_discrete_dense)
def test_hmm_univariate_discrete_dense_nan_predict_viterbi():
    f = model.predict(['A', nan, 'D', nan, 'C'], algorithm='viterbi')
    path = [4, 0, 0, 3, 1, 2, 5]

    assert_array_almost_equal(f, path)


@with_setup(setup_univariate_gaussian_dense)
def test_hmm_univariate_gaussian_dense_nan_predict_viterbi():
    f = model.predict([3, 5, 8, nan, 13], algorithm='viterbi')
    path = [4, 1, 0, 0, 0, 2, 5]

    assert_array_almost_equal(f, path)


@with_setup(setup_univariate_poisson_dense)
def test_hmm_univariate_poisson_dense_nan_predict_viterbi():
    f = model.predict([5, 8, 2, nan, 7, nan, 2], algorithm='viterbi')
    path = [4, 3, 1, 2, 3, 3, 1, 2, 5]

    assert_array_almost_equal(f, path)


@with_setup(setup_multivariate_mixed_dense)
def test_hmm_multivariate_mixed_dense_nan_predict_viterbi():
    f = model.predict([[0, nan, 5, nan, 3], [nan, 4, 1, 5, 6], [4, nan, 2, nan, 1]],
        algorithm='viterbi')
    path = [4, 0, 0, 0, 5]

    assert_array_almost_equal(f, path)


@with_setup(setup_multivariate_gaussian_dense)
def test_hmm_multivariate_gaussian_dense_nan_predict_viterbi():
    f = model.predict([[0, nan, 5, nan, 3], [nan, 4, 1, 5, 6], [-4, nan, -2, nan, 1]],
        algorithm='viterbi')
    path = [4, 0, 0, 1, 5]

    assert_array_almost_equal(f, path)


@with_setup(setup, teardown)
def test_hmm_viterbi_fit():
    seqs = [list(x) for x in ['ACT', 'ACT', 'ACC', 'ACTC', 'ACT', 'ACT', 'CCT',
    'CCC', 'AAT', 'CT', 'AT', 'CT', 'CT', 'CT', 'CT', 'CT', 'CT',
    'ACT', 'ACT', 'CT', 'ACT', 'CT', 'CT', 'CT', 'CT']]

    _, history = model.fit(seqs,
                               return_history=True,
                               algorithm='viterbi',
                               verbose=False,
                               use_pseudocount=True)

    total_improvement = history.total_improvement[-1]
    assert_equal(round(total_improvement, 4), 83.2834)


@with_setup(setup, teardown)
def test_hmm_viterbi_fit_no_pseudocount():
    seqs = [list(x) for x in ['ACT', 'ACT', 'ACC', 'ACTC', 'ACT', 'ACT', 'CCT',
    'CCC', 'AAT', 'CT', 'AT', 'CT', 'CT', 'CT', 'CT', 'CT', 'CT',
    'ACT', 'ACT', 'CT', 'ACT', 'CT', 'CT', 'CT', 'CT']]

    _, history = model.fit(seqs,
                               return_history=True,
                                     algorithm='viterbi',
                                     verbose=False,
                                     use_pseudocount=False)

    total_improvement = history.total_improvement[-1]
    assert_equal(round(total_improvement, 4), 84.9318)


@with_setup(setup, teardown)
def test_hmm_viterbi_fit_w_pseudocount():
    seqs = [list(x) for x in ['ACT', 'ACT', 'ACC', 'ACTC', 'ACT', 'ACT', 'CCT',
    'CCC', 'AAT', 'CT', 'AT', 'CT', 'CT', 'CT', 'CT', 'CT', 'CT',
    'ACT', 'ACT', 'CT', 'ACT', 'CT', 'CT', 'CT', 'CT']]

    _, history = model.fit(seqs,
                               return_history=True,
                                     algorithm='viterbi',
                                     verbose=False,
                                     transition_pseudocount=1.)

    total_improvement = history.total_improvement[-1]
    assert_equal(round(total_improvement, 4), 79.4713)


@with_setup(setup, teardown)
def test_hmm_viterbi_fit_w_pseudocount_priors():
    seqs = [list(x) for x in ['ACT', 'ACT', 'ACC', 'ACTC', 'ACT', 'ACT', 'CCT',
    'CCC', 'AAT', 'CT', 'AT', 'CT', 'CT', 'CT', 'CT', 'CT', 'CT',
    'ACT', 'ACT', 'CT', 'ACT', 'CT', 'CT', 'CT', 'CT']]

    _, history = model.fit(seqs,
                               return_history=True,
                                     algorithm='viterbi',
                                     verbose=False,
                                     transition_pseudocount=0.278,
                                     use_pseudocount=True)

    total_improvement = history.total_improvement[-1]
    assert_equal(round(total_improvement, 4), 81.7439)


@with_setup(setup, teardown)
def test_hmm_viterbi_fit_w_inertia():
    seqs = [list(x) for x in ['ACT', 'ACT', 'ACC', 'ACTC', 'ACT', 'ACT', 'CCT',
    'CCC', 'AAT', 'CT', 'AT', 'CT', 'CT', 'CT', 'CT', 'CT', 'CT',
    'ACT', 'ACT', 'CT', 'ACT', 'CT', 'CT', 'CT', 'CT']]

    _, history = model.fit(seqs,
                               return_history=True,
                                     algorithm='viterbi',
                                     verbose=False,
                                     edge_inertia=0.193)

    total_improvement = history.total_improvement[-1]
    assert_equal(round(total_improvement, 4), 84.9318)


@with_setup(setup, teardown)
def test_hmm_viterbi_fit_w_inertia2():
    seqs = [list(x) for x in ['ACT', 'ACT', 'ACC', 'ACTC', 'ACT', 'ACT', 'CCT',
    'CCC', 'AAT', 'CT', 'AT', 'CT', 'CT', 'CT', 'CT', 'CT', 'CT',
    'ACT', 'ACT', 'CT', 'ACT', 'CT', 'CT', 'CT', 'CT']]

    _, history = model.fit(seqs,
                               return_history=True,
                                     algorithm='viterbi',
                                     verbose=False,
                                     edge_inertia=0.82)

    total_improvement = history.total_improvement[-1]
    assert_equal(round(total_improvement, 4), 84.9318)


@with_setup(setup, teardown)
def test_hmm_viterbi_fit_w_pseudocount_inertia():
    seqs = [list(x) for x in ['ACT', 'ACT', 'ACC', 'ACTC', 'ACT', 'ACT', 'CCT',
    'CCC', 'AAT', 'CT', 'AT', 'CT', 'CT', 'CT', 'CT', 'CT', 'CT',
    'ACT', 'ACT', 'CT', 'ACT', 'CT', 'CT', 'CT', 'CT']]

    _, history = model.fit(seqs,
                               return_history=True,
                                     algorithm='viterbi',
                                     verbose=False,
                                     edge_inertia=0.23,
                                     use_pseudocount=True)

    total_improvement = history.total_improvement[-1]
    assert_equal(round(total_improvement, 4), 83.2834)

@with_setup(setup, teardown)
def test_hmm_viterbi_fit_one_check_input():
    seqs = [list(x) for x in ['ACT', 'ACT', 'ACC', 'ACTC', 'ACT', 'ACT', 'CCT',
    'CCC', 'AAT', 'CT', 'AT', 'CT', 'CT', 'CT', 'CT', 'CT', 'CT',
    'ACT', 'ACT', 'CT', 'ACT', 'CT', 'CT', 'CT', 'CT']]

    _, history = model.fit(seqs,
                               return_history=True,
                               algorithm='viterbi',
                               verbose=False,
                               use_pseudocount=True,
                               multiple_check_input=False)

    total_improvement = history.total_improvement[-1]
    assert_equal(round(total_improvement, 4), 83.2834)

@with_setup(setup, teardown)
def test_hmm_bw_fit():
    seqs = [list(x) for x in ['ACT', 'ACT', 'ACC', 'ACTC', 'ACT', 'ACT', 'CCT',
        'CCC', 'AAT', 'CT', 'AT', 'CT', 'CT', 'CT', 'CT', 'CT', 'CT',
        'ACT', 'ACT', 'CT', 'ACT', 'CT', 'CT', 'CT', 'CT']]

    _, history = model.fit(seqs,
                               return_history=True,
                                     algorithm='baum-welch',
                                     verbose=False,
                                     use_pseudocount=True,
                                     max_iterations=5)

    total_improvement = history.total_improvement[-1]
    assert_equal(round(total_improvement, 4), 83.1132)


@with_setup(setup_multivariate_discrete_sparse, teardown)
def test_hmm_bw_multivariate_discrete_fit():
    seqs = [[['A', 'A'], ['A', 'C'], ['C', 'T']], [['A', 'A'], ['C', 'C'], ['T', 'T']],
            [['A', 'A'], ['A', 'C'], ['C', 'C'], ['T', 'T']], [['A', 'A'], ['C', 'C']]]

    _, history = model.fit(seqs,
                               return_history=True,
                                     algorithm='baum-welch',
                                     verbose=False,
                                     use_pseudocount=True,
                                     max_iterations=5)

    total_improvement = history.total_improvement[-1]
    assert_equal(round(total_improvement, 4), 13.3622)


@with_setup(setup_multivariate_discrete_sparse, teardown)
def test_hmm_bw_multivariate_discrete_fit_json_yaml():
    seqs = [[['A', 'A'], ['A', 'C'], ['C', 'T']], [['A', 'A'], ['C', 'C'], ['T', 'T']],
            [['A', 'A'], ['A', 'C'], ['C', 'C'], ['T', 'T']], [['A', 'A'], ['C', 'C']]]

    hmm_json = HiddenMarkovModel.from_json(model.to_json())
    _, history = hmm_json.fit(seqs,
                               return_history=True,
                                     algorithm='baum-welch',
                                     verbose=False,
                                     use_pseudocount=True,
                                     max_iterations=5)

    total_improvement = history.total_improvement[-1]
    assert_equal(round(total_improvement, 4), 13.3622)

@with_setup(setup_multivariate_discrete_sparse, teardown)
def test_hmm_bw_multivariate_discrete_fit_robust_from_json():
    seqs = [[['A', 'A'], ['A', 'C'], ['C', 'T']], [['A', 'A'], ['C', 'C'], ['T', 'T']],
            [['A', 'A'], ['A', 'C'], ['C', 'C'], ['T', 'T']], [['A', 'A'], ['C', 'C']]]

    hmm_json = from_json(model.to_json())
    _, history = hmm_json.fit(seqs,
                               return_history=True,
                                     algorithm='baum-welch',
                                     verbose=False,
                                     use_pseudocount=True,
                                     max_iterations=5)

    total_improvement = history.total_improvement[-1]
    assert_equal(round(total_improvement, 4), 13.3622)

@with_setup(setup_multivariate_discrete_sparse, teardown)
def test_hmm_bw_multivariate_discrete_fit_from_yaml():
    seqs = [[['A', 'A'], ['A', 'C'], ['C', 'T']], [['A', 'A'], ['C', 'C'], ['T', 'T']],
            [['A', 'A'], ['A', 'C'], ['C', 'C'], ['T', 'T']], [['A', 'A'], ['C', 'C']]]

    hmm_yaml = HiddenMarkovModel.from_yaml(model.to_yaml())
    _, history = hmm_yaml.fit(seqs,
                              return_history=True,
                              algorithm='baum-welch',
                              verbose=False,
                              use_pseudocount=True,
                              max_iterations=5)

    total_improvement = history.total_improvement[-1]
    assert_equal(round(total_improvement, 4), 13.3622)


@with_setup(setup_multivariate_gaussian_sparse, teardown)
def test_hmm_bw_multivariate_gaussian_fit():
    seqs = [[[5, 8], [8, 10], [13, 17], [-3, -4]], [[6, 7], [13, 16], [12, 11], [-6, -7]],
            [[4, 6], [13, 15], [-4, -7]], [[6, 5], [14, 18], [-7, -5]]]

    _, history = model.fit(seqs,
                               return_history=True,
                                     algorithm='baum-welch',
                                     verbose=False,
                                     use_pseudocount=True,
                                     max_iterations=5)

    total_improvement = history.total_improvement[-1]
    assert_equal(round(total_improvement, 4), 24.7013)

@with_setup(setup_multivariate_gaussian_sparse, teardown)
def test_hmm_bw_multivariate_gaussian_from_json():
    seqs = [[[5, 8], [8, 10], [13, 17], [-3, -4]], [[6, 7], [13, 16], [12, 11], [-6, -7]],
            [[4, 6], [13, 15], [-4, -7]], [[6, 5], [14, 18], [-7, -5]]]

    hmm_json = HiddenMarkovModel.from_json(model.to_json())
    _, history = hmm_json.fit(seqs,
                               return_history=True,
                                     algorithm='baum-welch',
                                     verbose=False,
                                     use_pseudocount=True,
                                     max_iterations=5)

    total_improvement = history.total_improvement[-1]
    assert_equal(round(total_improvement, 4), 24.7013)

@with_setup(setup_multivariate_gaussian_sparse, teardown)
def test_hmm_bw_multivariate_gaussian_robust_from_json():
    seqs = [[[5, 8], [8, 10], [13, 17], [-3, -4]], [[6, 7], [13, 16], [12, 11], [-6, -7]],
            [[4, 6], [13, 15], [-4, -7]], [[6, 5], [14, 18], [-7, -5]]]

    hmm_json = from_json(model.to_json())
    _, history = hmm_json.fit(seqs,
                               return_history=True,
                                     algorithm='baum-welch',
                                     verbose=False,
                                     use_pseudocount=True,
                                     max_iterations=5)

    total_improvement = history.total_improvement[-1]
    assert_equal(round(total_improvement, 4), 24.7013)

@with_setup(setup_multivariate_gaussian_sparse, teardown)
def test_hmm_bw_multivariate_gaussian_from_yaml(): 
    seqs = [[[5, 8], [8, 10], [13, 17], [-3, -4]], [[6, 7], [13, 16], [12, 11], [-6, -7]],
            [[4, 6], [13, 15], [-4, -7]], [[6, 5], [14, 18], [-7, -5]]]
            
    hmm_yaml = HiddenMarkovModel.from_yaml(model.to_yaml())
    _, history = hmm_yaml.fit(seqs,
                              return_history=True,
                              algorithm='baum-welch',
                              verbose=False,
                              use_pseudocount=True,
                              max_iterations=5)

    total_improvement = history.total_improvement[-1]
    assert_equal(round(total_improvement, 4), 24.7013)

@with_setup(setup, teardown)
def test_hmm_bw_fit_json():
    seqs = [list(x) for x in ['ACT', 'ACT', 'ACC', 'ACTC', 'ACT', 'ACT', 'CCT',
        'CCC', 'AAT', 'CT', 'AT', 'CT', 'CT', 'CT', 'CT', 'CT', 'CT',
        'ACT', 'ACT', 'CT', 'ACT', 'CT', 'CT', 'CT', 'CT']]

    _, history = model.fit(seqs,
                               return_history=True,
                                     algorithm='baum-welch',
                                     verbose=False,
                                     use_pseudocount=True,
                                     max_iterations=5)

    total_improvement = history.total_improvement[-1]

    assert_equal(round(total_improvement, 4), 83.1132)
    assert_almost_equal(sum(model.log_probability(seq) for seq in seqs), -42.2341, 4)

    hmm = HiddenMarkovModel.from_json(model.to_json())
    assert_almost_equal(sum(model.log_probability(seq) for seq in seqs), -42.2341, 4)

@with_setup(setup, teardown)
def test_hmm_bw_fit_robust_from_json():
    seqs = [list(x) for x in ['ACT', 'ACT', 'ACC', 'ACTC', 'ACT', 'ACT', 'CCT',
        'CCC', 'AAT', 'CT', 'AT', 'CT', 'CT', 'CT', 'CT', 'CT', 'CT',
        'ACT', 'ACT', 'CT', 'ACT', 'CT', 'CT', 'CT', 'CT']]

    _, history = model.fit(seqs,
                               return_history=True,
                                     algorithm='baum-welch',
                                     verbose=False,
                                     use_pseudocount=True,
                                     max_iterations=5)

    total_improvement = history.total_improvement[-1]

    assert_equal(round(total_improvement, 4), 83.1132)
    assert_almost_equal(sum(model.log_probability(seq) for seq in seqs), -42.2341, 4)

    hmm = from_json(model.to_json())
    assert_almost_equal(sum(model.log_probability(seq) for seq in seqs), -42.2341, 4)

@with_setup(setup, teardown)
def test_hmm_bw_fit_no_pseudocount():
    seqs = [list(x) for x in ['ACT', 'ACT', 'ACC', 'ACTC', 'ACT', 'ACT', 'CCT',
        'CCC', 'AAT', 'CT', 'AT', 'CT', 'CT', 'CT', 'CT', 'CT', 'CT',
        'ACT', 'ACT', 'CT', 'ACT', 'CT', 'CT', 'CT', 'CT']]

    _, history = model.fit(seqs,
                               return_history=True,
                                     algorithm='baum-welch',
                                     verbose=False,
                                     use_pseudocount=False,
                                     max_iterations=5)

    total_improvement = history.total_improvement[-1]
    assert_equal(round(total_improvement, 4), 85.681)


@with_setup(setup, teardown)
def test_hmm_bw_fit_w_pseudocount():
    seqs = [list(x) for x in ['ACT', 'ACT', 'ACC', 'ACTC', 'ACT', 'ACT', 'CCT',
        'CCC', 'AAT', 'CT', 'AT', 'CT', 'CT', 'CT', 'CT', 'CT', 'CT',
        'ACT', 'ACT', 'CT', 'ACT', 'CT', 'CT', 'CT', 'CT']]

    _, history = model.fit(seqs,
                               return_history=True,
                                     algorithm='baum-welch',
                                     verbose=False,
                                     transition_pseudocount=0.123,
                                     max_iterations=5)

    total_improvement = history.total_improvement[-1]
    assert_equal(round(total_improvement, 4), 84.9408)


@with_setup(setup, teardown)
def test_hmm_bw_fit_w_pseudocount_priors():
    seqs = [list(x) for x in ['ACT', 'ACT', 'ACC', 'ACTC', 'ACT', 'ACT', 'CCT',
        'CCC', 'AAT', 'CT', 'AT', 'CT', 'CT', 'CT', 'CT', 'CT', 'CT',
        'ACT', 'ACT', 'CT', 'ACT', 'CT', 'CT', 'CT', 'CT']]

    _, history = model.fit(seqs,
                               return_history=True,
                                     algorithm='baum-welch',
                                     verbose=False,
                                     transition_pseudocount=0.278,
                                     use_pseudocount=True,
                                     max_iterations=5)

    total_improvement = history.total_improvement[-1]
    assert_equal(round(total_improvement, 4), 81.2265)


@with_setup(setup, teardown)
def test_hmm_bw_fit_w_inertia():
    seqs = [list(x) for x in ['ACT', 'ACT', 'ACC', 'ACTC', 'ACT', 'ACT', 'CCT',
        'CCC', 'AAT', 'CT', 'AT', 'CT', 'CT', 'CT', 'CT', 'CT', 'CT',
        'ACT', 'ACT', 'CT', 'ACT', 'CT', 'CT', 'CT', 'CT']]

    _, history = model.fit(seqs,
                               return_history=True,
                                     algorithm='baum-welch',
                                     verbose=False,
                                     edge_inertia=0.193,
                                     max_iterations=5)

    total_improvement = history.total_improvement[-1]
    assert_equal(round(total_improvement, 4), 85.0528)


@with_setup(setup, teardown)
def test_hmm_bw_fit_w_inertia2():
    seqs = [list(x) for x in ['ACT', 'ACT', 'ACC', 'ACTC', 'ACT', 'ACT', 'CCT',
        'CCC', 'AAT', 'CT', 'AT', 'CT', 'CT', 'CT', 'CT', 'CT', 'CT',
        'ACT', 'ACT', 'CT', 'ACT', 'CT', 'CT', 'CT', 'CT']]

    _, history = model.fit(seqs,
                               return_history=True,
                                     algorithm='baum-welch',
                                     verbose=False,
                                     edge_inertia=0.82,
                                     max_iterations=5)

    total_improvement = history.total_improvement[-1]
    assert_equal(round(total_improvement, 4), 72.5134)


@with_setup(setup, teardown)
def test_hmm_bw_fit_w_pseudocount_inertia():
    seqs = [list(x) for x in ['ACT', 'ACT', 'ACC', 'ACTC', 'ACT', 'ACT', 'CCT',
        'CCC', 'AAT', 'CT', 'AT', 'CT', 'CT', 'CT', 'CT', 'CT', 'CT',
        'ACT', 'ACT', 'CT', 'ACT', 'CT', 'CT', 'CT', 'CT']]

    _, history = model.fit(seqs,
                               return_history=True,
                                     algorithm='baum-welch',
                                     verbose=False,
                                     edge_inertia=0.02,
                                     use_pseudocount=True,
                                     max_iterations=5)

    total_improvement = history.total_improvement[-1]
    assert_equal(round(total_improvement, 4), 83.0764)


@with_setup(setup, teardown)
def test_hmm_bw_fit_w_frozen_distributions():
    seqs = [list(x) for x in ['ACT', 'ACT', 'ACC', 'ACTC', 'ACT', 'ACT', 'CCT',
        'CCC', 'AAT', 'CT', 'AT', 'CT', 'CT', 'CT', 'CT', 'CT', 'CT',
        'ACT', 'ACT', 'CT', 'ACT', 'CT', 'CT', 'CT', 'CT']]

    _, history = model.fit(seqs,
                               return_history=True,
                                     algorithm='baum-welch',
                                     verbose=False,
                                     distribution_inertia=1.00,
                                     max_iterations=5)

    total_improvement = history.total_improvement[-1]
    assert_equal(round(total_improvement, 4), 64.474)


@with_setup(setup, teardown)
def test_hmm_bw_fit_w_frozen_edges():
    seqs = [list(x) for x in ['ACT', 'ACT', 'ACC', 'ACTC', 'ACT', 'ACT', 'CCT',
        'CCC', 'AAT', 'CT', 'AT', 'CT', 'CT', 'CT', 'CT', 'CT', 'CT',
        'ACT', 'ACT', 'CT', 'ACT', 'CT', 'CT', 'CT', 'CT']]

    _, history = model.fit(seqs,
                               return_history=True,
                                     algorithm='baum-welch',
                                     verbose=False,
                                     edge_inertia=1.00,
                                     max_iterations=5)

    total_improvement = history.total_improvement[-1]
    assert_equal(round(total_improvement, 4), 44.0208)


@with_setup(setup, teardown)
def test_hmm_bw_fit_w_edge_a_distribution_inertia():
    seqs = [list(x) for x in ['ACT', 'ACT', 'ACC', 'ACTC', 'ACT', 'ACT', 'CCT',
        'CCC', 'AAT', 'CT', 'AT', 'CT', 'CT', 'CT', 'CT', 'CT', 'CT',
        'ACT', 'ACT', 'CT', 'ACT', 'CT', 'CT', 'CT', 'CT']]

    _, history = model.fit(seqs,
                               return_history=True,
                                     algorithm='baum-welch',
                                     verbose=False,
                                     edge_inertia=0.5,
                                     distribution_inertia=0.5,
                                     max_iterations=5)

    total_improvement = history.total_improvement[-1]
    assert_equal(round(total_improvement, 4), 81.5447)


@with_setup(setup, teardown)
def test_hmm_bw_fit_parallel():
    seqs = [list(x) for x in ['ACT', 'ACT', 'ACC', 'ACTC', 'ACT', 'ACT', 'CCT',
        'CCC', 'AAT', 'CT', 'AT', 'CT', 'CT', 'CT', 'CT', 'CT', 'CT',
        'ACT', 'ACT', 'CT', 'ACT', 'CT', 'CT', 'CT', 'CT']]

    _, history = model.fit(seqs,
                               return_history=True,
                                     algorithm='baum-welch',
                                     verbose=False,
                                     use_pseudocount=True,
                                     max_iterations=5,
                                     n_jobs=2)

    total_improvement = history.total_improvement[-1]
    assert_equal(round(total_improvement, 4), 83.1132)


@with_setup(setup, teardown)
def test_hmm_bw_fit_no_pseudocount_parallel():
    seqs = [list(x) for x in ['ACT', 'ACT', 'ACC', 'ACTC', 'ACT', 'ACT', 'CCT',
        'CCC', 'AAT', 'CT', 'AT', 'CT', 'CT', 'CT', 'CT', 'CT', 'CT',
        'ACT', 'ACT', 'CT', 'ACT', 'CT', 'CT', 'CT', 'CT']]

    _, history = model.fit(seqs,
                               return_history=True,
                                     algorithm='baum-welch',
                                     verbose=False,
                                     use_pseudocount=False,
                                     max_iterations=5,
                                     n_jobs=2)

    total_improvement = history.total_improvement[-1]
    assert_equal(round(total_improvement, 4), 85.681)


@with_setup(setup, teardown)
def test_hmm_bw_fit_w_pseudocount_parallel():
    seqs = [list(x) for x in ['ACT', 'ACT', 'ACC', 'ACTC', 'ACT', 'ACT', 'CCT',
        'CCC', 'AAT', 'CT', 'AT', 'CT', 'CT', 'CT', 'CT', 'CT', 'CT',
        'ACT', 'ACT', 'CT', 'ACT', 'CT', 'CT', 'CT', 'CT']]

    _, history = model.fit(seqs,
                               return_history=True,
                                     algorithm='baum-welch',
                                     verbose=False,
                                     transition_pseudocount=0.123,
                                     max_iterations=5,
                                     n_jobs=2)

    total_improvement = history.total_improvement[-1]
    assert_equal(round(total_improvement, 4), 84.9408)


@with_setup(setup, teardown)
def test_hmm_bw_fit_w_pseudocount_priors_parallel():
    seqs = [list(x) for x in ['ACT', 'ACT', 'ACC', 'ACTC', 'ACT', 'ACT', 'CCT',
        'CCC', 'AAT', 'CT', 'AT', 'CT', 'CT', 'CT', 'CT', 'CT', 'CT',
        'ACT', 'ACT', 'CT', 'ACT', 'CT', 'CT', 'CT', 'CT']]

    _, history = model.fit(seqs,
                               return_history=True,
                                     algorithm='baum-welch',
                                     verbose=False,
                                     transition_pseudocount=0.278,
                                     use_pseudocount=True,
                                     max_iterations=5,
                                     n_jobs=2)

    total_improvement = history.total_improvement[-1]
    assert_equal(round(total_improvement, 4), 81.2265)


@with_setup(setup, teardown)
def test_hmm_bw_fit_w_inertia_parallel():
    seqs = [list(x) for x in ['ACT', 'ACT', 'ACC', 'ACTC', 'ACT', 'ACT', 'CCT',
        'CCC', 'AAT', 'CT', 'AT', 'CT', 'CT', 'CT', 'CT', 'CT', 'CT',
        'ACT', 'ACT', 'CT', 'ACT', 'CT', 'CT', 'CT', 'CT']]

    _, history = model.fit(seqs,
                               return_history=True,
                                     algorithm='baum-welch',
                                     verbose=False,
                                     edge_inertia=0.193,
                                     max_iterations=5,
                                     n_jobs=2)

    total_improvement = history.total_improvement[-1]
    assert_equal(round(total_improvement, 4), 85.0528)


@with_setup(setup, teardown)
def test_hmm_bw_fit_w_inertia2_parallel():
    seqs = [list(x) for x in ['ACT', 'ACT', 'ACC', 'ACTC', 'ACT', 'ACT', 'CCT',
        'CCC', 'AAT', 'CT', 'AT', 'CT', 'CT', 'CT', 'CT', 'CT', 'CT',
        'ACT', 'ACT', 'CT', 'ACT', 'CT', 'CT', 'CT', 'CT']]

    _, history = model.fit(seqs,
                               return_history=True,
                                     algorithm='baum-welch',
                                     verbose=False,
                                     edge_inertia=0.82,
                                     max_iterations=5,
                                     n_jobs=2)

    total_improvement = history.total_improvement[-1]
    assert_equal(round(total_improvement, 4), 72.5134)


@with_setup(setup, teardown)
def test_hmm_bw_fit_w_pseudocount_inertia_parallel():
    seqs = [list(x) for x in ['ACT', 'ACT', 'ACC', 'ACTC', 'ACT', 'ACT', 'CCT',
        'CCC', 'AAT', 'CT', 'AT', 'CT', 'CT', 'CT', 'CT', 'CT', 'CT',
        'ACT', 'ACT', 'CT', 'ACT', 'CT', 'CT', 'CT', 'CT']]

    _, history = model.fit(seqs,
                               return_history=True,
                                     algorithm='baum-welch',
                                     verbose=False,
                                     edge_inertia=0.02,
                                     use_pseudocount=True,
                                     max_iterations=5,
                                     n_jobs=2)

    total_improvement = history.total_improvement[-1]
    assert_equal(round(total_improvement, 4), 83.0764)


@with_setup(setup, teardown)
def test_hmm_bw_fit_w_frozen_distributions_parallel():
    seqs = [list(x) for x in ['ACT', 'ACT', 'ACC', 'ACTC', 'ACT', 'ACT', 'CCT',
        'CCC', 'AAT', 'CT', 'AT', 'CT', 'CT', 'CT', 'CT', 'CT', 'CT',
        'ACT', 'ACT', 'CT', 'ACT', 'CT', 'CT', 'CT', 'CT']]

    _, history = model.fit(seqs,
                               return_history=True,
                                     algorithm='baum-welch',
                                     verbose=False,
                                     distribution_inertia=1.00,
                                     max_iterations=5,
                                     n_jobs=2)

    total_improvement = history.total_improvement[-1]
    assert_equal(round(total_improvement, 4), 64.474)


@with_setup(setup, teardown)
def test_hmm_bw_fit_w_frozen_edges_parallel():
    seqs = [list(x) for x in ['ACT', 'ACT', 'ACC', 'ACTC', 'ACT', 'ACT', 'CCT',
        'CCC', 'AAT', 'CT', 'AT', 'CT', 'CT', 'CT', 'CT', 'CT', 'CT',
        'ACT', 'ACT', 'CT', 'ACT', 'CT', 'CT', 'CT', 'CT']]

    _, history = model.fit(seqs,
                               return_history=True,
                                     algorithm='baum-welch',
                                     verbose=False,
                                     edge_inertia=1.00,
                                     max_iterations=5,
                                     n_jobs=2)

    total_improvement = history.total_improvement[-1]
    assert_equal(round(total_improvement, 4), 44.0208)


@with_setup(setup, teardown)
def test_hmm_bw_fit_w_edge_a_distribution_inertia():
    seqs = [list(x) for x in ['ACT', 'ACT', 'ACC', 'ACTC', 'ACT', 'ACT', 'CCT',
        'CCC', 'AAT', 'CT', 'AT', 'CT', 'CT', 'CT', 'CT', 'CT', 'CT',
        'ACT', 'ACT', 'CT', 'ACT', 'CT', 'CT', 'CT', 'CT']]

    _, history = model.fit(seqs,
                               return_history=True,
                                     algorithm='baum-welch',
                                     verbose=False,
                                     edge_inertia=0.5,
                                     distribution_inertia=0.5,
                                     max_iterations=5,
                                     n_jobs=2)

    total_improvement = history.total_improvement[-1]
    assert_equal(round(total_improvement, 4), 81.5447)

@with_setup(setup, teardown)
def test_hmm_bw_fit_one_check_input():
    seqs = [list(x) for x in ['ACT', 'ACT', 'ACC', 'ACTC', 'ACT', 'ACT', 'CCT',
        'CCC', 'AAT', 'CT', 'AT', 'CT', 'CT', 'CT', 'CT', 'CT', 'CT',
        'ACT', 'ACT', 'CT', 'ACT', 'CT', 'CT', 'CT', 'CT']]

    _, history = model.fit(seqs,
                               return_history=True,
                                     algorithm='baum-welch',
                                     verbose=False,
                                     use_pseudocount=True,
                                     max_iterations=5,
                                     multiple_check_input=False)

    total_improvement = history.total_improvement[-1]
    assert_equal(round(total_improvement, 4), 83.1132)

def test_hmm_initialization():
    hmmd1 = HiddenMarkovModel()
    assert_equal(hmmd1.d, 0)


def test_hmm_univariate_initialization():
    s1d1 = State(NormalDistribution(5, 2))
    s2d1 = State(NormalDistribution(0, 1))
    s3d1 = State(UniformDistribution(0, 10))

    hmmd1 = HiddenMarkovModel()
    hmmd1.add_transition(hmmd1.start, s1d1, 0.5)
    hmmd1.add_transition(hmmd1.start, s2d1, 0.5)
    hmmd1.add_transition(s1d1, s3d1, 1)
    hmmd1.add_transition(s2d1, s3d1, 1)
    hmmd1.add_transition(s3d1, s1d1, 0.5)
    hmmd1.add_transition(s3d1, s2d1, 0.5)

    assert_equal(hmmd1.d, 0)

    hmmd1.bake()
    assert_equal(hmmd1.d, 1)


def test_hmm_multivariate_initialization():
    s1d3 = State(MultivariateGaussianDistribution([1, 4, 3], [[3, 0, 1],[0, 3, 0],[1, 0, 3]]))
    s2d3 = State(MultivariateGaussianDistribution([7, 7, 7], [[1, 0, 0],[0, 5, 0],[0, 0, 3]]))
    s3d3 = State(IndependentComponentsDistribution([UniformDistribution(0, 10), UniformDistribution(0, 10), UniformDistribution(0, 10)]))

    hmmd3 = HiddenMarkovModel()
    assert_equal(hmmd3.d, 0)

    hmmd3.add_transition(hmmd3.start, s1d3, 0.5)
    hmmd3.add_transition(hmmd3.start, s2d3, 0.5)
    hmmd3.add_transition(s1d3, s3d3, 1)
    hmmd3.add_transition(s2d3, s3d3, 1)
    hmmd3.add_transition(s3d3, s1d3, 0.5)
    hmmd3.add_transition(s3d3, s2d3, 0.5)
    assert_equal(hmmd3.d, 0)

    hmmd3.bake()
    assert_equal(hmmd3.d, 3)


def test_hmm_initialization_error():
    sbd1 = State(UniformDistribution(0, 10))
    sbd3 = State(MultivariateGaussianDistribution([1, 4, 3], [[3, 0, 1],[0, 3, 0],[1, 0, 3]]))

    hmmb = HiddenMarkovModel()
    hmmb.add_transition(hmmb.start, sbd1, 0.5)
    hmmb.add_transition(hmmb.start, sbd3, 0.5)
    hmmb.add_transition(sbd1, sbd1, 0.5)
    hmmb.add_transition(sbd1, sbd3, 0.5)
    hmmb.add_transition(sbd3, sbd1, 0.5)
    hmmb.add_transition(sbd3, sbd3, 0.5)

    assert_raises(ValueError, hmmb.bake)


@with_setup(setup_univariate_gaussian_dense)
def test_hmm_pickle_univariate():
    model2 = pickle.loads(pickle.dumps(model))

    random_state = numpy.random.RandomState(0)
    for i in range(10):
        sequence = random_state.normal(0, 1, size=10)

        logp1 = model.log_probability(sequence)
        logp2 = model2.log_probability(sequence)

        assert_almost_equal(logp1, logp2)


@with_setup(setup_univariate_gaussian_dense)
def test_hmm_json_univariate():
    model2 = HiddenMarkovModel.from_json(model.to_json())

    random_state = numpy.random.RandomState(0)
    for i in range(10):
        sequence = random_state.normal(0, 1, size=10)

        logp1 = model.log_probability(sequence)
        logp2 = model2.log_probability(sequence)

        assert_almost_equal(logp1, logp2)

@with_setup(setup_univariate_gaussian_dense)
def test_hmm_robust_from_json_univariate():
    model2 = from_json(model.to_json())

    random_state = numpy.random.RandomState(0)
    for i in range(10):
        sequence = random_state.normal(0, 1, size=10)

        logp1 = model.log_probability(sequence)
        logp2 = model2.log_probability(sequence)

        assert_almost_equal(logp1, logp2)

@with_setup(setup_multivariate_gaussian_dense)
def test_hmm_pickle_multivariate():
    model2 = pickle.loads(pickle.dumps(model))

    random_state = numpy.random.RandomState(0)
    for i in range(10):
        sequence = random_state.normal(0, 1, size=(10, 5))

        logp1 = model.log_probability(sequence)
        logp2 = model2.log_probability(sequence)

        assert_almost_equal(logp1, logp2)


@with_setup(setup_multivariate_gaussian_dense)
def test_hmm_json_multivariate():
    model2 = HiddenMarkovModel.from_json(model.to_json())

    random_state = numpy.random.RandomState(0)
    for i in range(10):
        sequence = random_state.normal(0, 1, size=(10, 5))

        logp1 = model.log_probability(sequence)
        logp2 = model2.log_probability(sequence)

        assert_almost_equal(logp1, logp2)

@with_setup(setup_multivariate_gaussian_dense)
def test_hmm_robust_from_json_multivariate():
    model2 = from_json(model.to_json())

    random_state = numpy.random.RandomState(0)
    for i in range(10):
        sequence = random_state.normal(0, 1, size=(10, 5))

        logp1 = model.log_probability(sequence)
        logp2 = model2.log_probability(sequence)

        assert_almost_equal(logp1, logp2)

@with_setup(setup_univariate_discrete_dense, teardown)
def test_hmm_univariate_discrete_from_samples():
    X = [model.sample(random_state=0) for i in range(25)]
    model2 = HiddenMarkovModel.from_samples(DiscreteDistribution, 4, X, max_iterations=25)

    logp1 = sum(map(model.log_probability, X))
    logp2 = sum(map(model2.log_probability, X))

    assert_greater(logp2, logp1)


@with_setup(setup_univariate_discrete_dense, teardown)
def test_hmm_univariate_discrete_from_samples_with_labels():
    X, y = zip(*model.sample(25, path=True, random_state=0))
    y = [[state.name for state in seq if not state.is_silent()] for seq in y]

    model2 = HiddenMarkovModel.from_samples(DiscreteDistribution, 4, X, 
        max_iterations=25, labels=y)

    logp1 = sum(map(model.log_probability, X))
    logp2 = sum(map(model2.log_probability, X))

    assert_greater(logp2, logp1)

@with_setup(setup_univariate_discrete_dense, teardown)
def test_hmm_univariate_discrete_from_samples_one_check_input():
    X = [model.sample(random_state=0) for i in range(25)]
    model2 = HiddenMarkovModel.from_samples(DiscreteDistribution, 4, X, 
                                            max_iterations=25,
                                            multiple_check_input=False)

    logp1 = sum(map(model.log_probability, X))
    logp2 = sum(map(model2.log_probability, X))

    assert_greater(logp2, logp1)

@with_setup(setup_univariate_gaussian_dense, teardown)
def test_hmm_univariate_gaussian_from_samples():
    X = model.sample(n=25, random_state=0)
    model2 = HiddenMarkovModel.from_samples(NormalDistribution, 4, X, max_iterations=25)

    logp1 = sum(map(model.log_probability, X))
    logp2 = sum(map(model2.log_probability, X))

    assert_greater(logp2, logp1)


@with_setup(setup_multivariate_gaussian_dense, teardown)
def test_hmm_multivariate_gaussian_from_samples():
    X = model.sample(n=25, random_state=0)
    model2 = HiddenMarkovModel.from_samples(MultivariateGaussianDistribution, 4, X, max_iterations=25)

    logp1 = sum(map(model.log_probability, X))
    logp2 = sum(map(model2.log_probability, X))

    assert_greater(logp2, logp1)

@with_setup(setup_univariate_discrete_dense, teardown)
def test_hmm_univariate_discrete_from_samples_end_state():
    X = model.sample(n=25, random_state=0)
    model2 = HiddenMarkovModel.from_samples(DiscreteDistribution, 4, X, max_iterations=25, end_state=True)

    #We get non-zero end probabilities for each state
    assert_greater(model2.dense_transition_matrix()[0][model2.end_index],0)
    assert_greater(model2.dense_transition_matrix()[1][model2.end_index],0)
    assert_greater(model2.dense_transition_matrix()[2][model2.end_index],0)
    assert_greater(model2.dense_transition_matrix()[3][model2.end_index],0)

@with_setup(setup_univariate_discrete_dense, teardown)
def test_hmm_univariate_discrete_from_samples_no_end_state():
    X = [model.sample(random_state=0) for i in range(25)]
    model2 = HiddenMarkovModel.from_samples(DiscreteDistribution, 4, X, max_iterations=25, end_state=False)

    #We don't have end probabilities for each state
    assert_equal(model2.dense_transition_matrix()[0][model2.end_index],0)
    assert_equal(model2.dense_transition_matrix()[1][model2.end_index],0)
    assert_equal(model2.dense_transition_matrix()[2][model2.end_index],0)
    assert_equal(model2.dense_transition_matrix()[3][model2.end_index],0)

@with_setup(setup_general_mixture_gaussian, teardown)
def test_hmm_json_general_mixture_gaussian():
    model2 = HiddenMarkovModel.from_json(model.to_json())
    random_state = numpy.random.RandomState(0)
    for i in range(10):
        sequence = random_state.normal(0, 1, size=10)

        logp1 = model.log_probability(sequence)
        logp2 = model2.log_probability(sequence)

        assert_almost_equal(logp1, logp2)

@with_setup(setup_general_mixture_gaussian, teardown)
def test_hmm_robust_from_json_general_mixture_gaussian():
    model2 = from_json(model.to_json())
    random_state = numpy.random.RandomState(0)
    for i in range(10):
        sequence = random_state.normal(0, 1, size=10)

        logp1 = model.log_probability(sequence)
        logp2 = model2.log_probability(sequence)

        assert_almost_equal(logp1, logp2)

def test_io_fit():
    X = [numpy.random.choice(['A', 'B'], size=25) for i in range(20)]
    weights = numpy.abs(numpy.random.randn(20))
    data_generator = SequenceGenerator(X, weights)

    s1 = State(DiscreteDistribution({'A': 0.3, 'B': 0.7}))
    s2 = State(DiscreteDistribution({'A': 0.9, 'B': 0.1}))
    model1 = HiddenMarkovModel()
    model1.add_states(s1, s2)
    model1.add_transition(model1.start, s1, 0.8)
    model1.add_transition(model1.start, s2, 0.2)
    model1.add_transition(s1, s1, 0.3)
    model1.add_transition(s1, s2, 0.7)
    model1.add_transition(s2, s1, 0.4)
    model1.add_transition(s2, s2, 0.6)
    model1.bake()
    model1.fit(X, weights=weights, max_iterations=5)

    s1 = State(DiscreteDistribution({'A': 0.3, 'B': 0.7}))
    s2 = State(DiscreteDistribution({'A': 0.9, 'B': 0.1}))
    model2 = HiddenMarkovModel()
    model2.add_states(s1, s2)
    model2.add_transition(model2.start, s1, 0.8)
    model2.add_transition(model2.start, s2, 0.2)
    model2.add_transition(s1, s1, 0.3)
    model2.add_transition(s1, s2, 0.7)
    model2.add_transition(s2, s1, 0.4)
    model2.add_transition(s2, s2, 0.6)
    model2.bake()
    model2.fit(X, weights=weights, max_iterations=5)

    logp1 = [model1.log_probability(x) for x in X]
    logp2 = [model2.log_probability(x) for x in X]

    assert_array_almost_equal(logp1, logp2)

def test_io_from_samples_hmm():
    X = [numpy.random.choice(['A', 'B'], size=25) for i in range(20)]
    weights = numpy.abs(numpy.random.randn(20))
    data_generator = SequenceGenerator(X, weights)

    model1 = HiddenMarkovModel.from_samples(DiscreteDistribution,
        n_components=2, X=X, weights=weights, max_iterations=5,
        init='first-k', random_state=1)
    model2 = HiddenMarkovModel.from_samples(DiscreteDistribution,
        n_components=2, X=data_generator, max_iterations=5,
        init='first-k', random_state=1)

    logp1 = [model1.log_probability(x) for x in X]
    logp2 = [model2.log_probability(x) for x in X]

    assert_array_almost_equal(logp1, logp2)

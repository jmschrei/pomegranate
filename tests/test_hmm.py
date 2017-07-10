from __future__ import (division)

from pomegranate import *
from pomegranate.parallel import log_probability
from nose.tools import with_setup
from nose.tools import assert_almost_equal
from nose.tools import assert_equal
from nose.tools import assert_not_equal
from nose.tools import assert_less_equal
from nose.tools import assert_raises
from nose.tools import assert_greater
from numpy.testing import assert_array_almost_equal
import pickle
import random
import numpy as np
import time

np.random.seed(0)
random.seed(0)

NEGINF = float("-inf")

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
	model = HiddenMarkovModel("Global Alignment")

	i_d = DiscreteDistribution({ 'A': 0.25, 'C': 0.25, 'G': 0.25, 'T': 0.25 })

	d1 = DiscreteDistribution({ "A": 0.95, 'C': 0.01, 'G': 0.01, 'T': 0.02 })
	d2 = DiscreteDistribution({ "A": 0.003, 'C': 0.99, 'G': 0.003, 'T': 0.004 })
	d3 = DiscreteDistribution({ "A": 0.01, 'C': 0.01, 'G': 0.01, 'T': 0.97 })

	model = sparse_model(d1, d2, d3, i_d)

def setup_multivariate_discrete():
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

def setup_multivariate_gaussian():
	'''
	Build a model that we want to use to test sequences. This model will
	be somewhat complicated, in order to extensively test YAHMM. This will be
	a three state global sequence alignment HMM. The HMM models a reference of
	'ACT', with pseudocounts to allow for slight deviations from this
	reference.
	'''

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

def setup_discrete_dense():
	global model

	d1 = DiscreteDistribution({'A': 0.90, 'B': 0.02, 'C': 0.03, 'D': 0.05})
	d2 = DiscreteDistribution({'A': 0.02, 'B': 0.90, 'C': 0.03, 'D': 0.05})
	d3 = DiscreteDistribution({'A': 0.03, 'B': 0.02, 'C': 0.90, 'D': 0.05})
	d4 = DiscreteDistribution({'A': 0.05, 'B': 0.02, 'C': 0.03, 'D': 0.90})
	model = dense_model(d1, d2, d3, d4)

def setup_gaussian_dense():
	global model

	d1 = NormalDistribution(5, 1)
	d2 = NormalDistribution(1, 1)
	d3 = NormalDistribution(13, 2)
	d4 = NormalDistribution(16, 0.5)
	model = dense_model(d1, d2, d3, d4)

def setup_multivariate_gaussian_dense():
	global model

	mu = numpy.random.randn(4, 5)
	d1 = MultivariateGaussianDistribution(mu[0], numpy.eye(5))
	d2 = MultivariateGaussianDistribution(mu[1], numpy.eye(5))
	d3 = MultivariateGaussianDistribution(mu[2], numpy.eye(5))
	d4 = MultivariateGaussianDistribution(mu[3], numpy.eye(5))
	model = dense_model(d1, d2, d3, d4)

def setup_poisson_dense():
	global model

	d1 = PoissonDistribution(12.1)
	d2 = PoissonDistribution(8.7)
	d3 = PoissonDistribution(1)
	d4 = PoissonDistribution(5)
	model = dense_model(d1, d2, d3, d4)

def teardown():
	'''
	Remove the model at the end of the unit testing. Since it is stored in a
	global variance, simply delete it.
	'''

	pass

@with_setup(setup_discrete_dense)
def test_discrete_forward():
	f = model.forward(['A', 'B', 'D', 'D', 'C'])
	logp = numpy.array([[NEGINF, NEGINF, NEGINF, NEGINF, 0., NEGINF],
				[-2.40794561, -5.11599581, -5.11599581, -3.91202301, NEGINF, -4.40631933],
				[-6.89188193, -4.35987383, -8.09848286, -7.43200392, NEGINF, -6.52303724],
				[-8.73634472, -9.47926612, -8.22377759, -5.87605232, NEGINF, -8.01305991],
				[-10.28388158, -10.39501067, -10.80077009, -6.76858029, NEGINF, -8.99969597],
				[-11.780373, -11.84820305, -9.00308599, -11.14654124, NEGINF, -11.09251037]])
	assert_array_almost_equal(f, logp)

@with_setup(setup_gaussian_dense)
def test_gaussian_forward():
	f = model.forward([3, 5, 8, 19, 13])
	logp = numpy.array([[NEGINF, NEGINF, NEGINF, NEGINF, 0.0, NEGINF], 
		[-5.221523626198319, -4.122911337530209, -15.72152362619832, -339.14208208451845, NEGINF, -6.137807473983832], 
		[-6.045149476287824, -15.056746007188107, -14.571238720950003, -247.67044476202696, NEGINF, -8.347414404915495], 
		[-12.157146753448874, -33.766352938119766, -13.083738234853534, -135.8798604314077, NEGINF, -14.126191869688759], 
		[-111.69303081629936, -177.04513040289305, -19.78896563212587, -31.409154369913637, NEGINF, -22.091541742268795],
		[-55.01047129270264, -95.01047129270263, -22.605021155923353, -38.93103873379323, NEGINF, -24.90760616769035]])
	assert_array_almost_equal(f, logp)


@with_setup(setup_multivariate_gaussian_dense)
def test_multivariate_gaussian_forward():
	f = model.forward([[0, 1, 5, 2, 3], [2, 4, 1, 5, 6], [-4, 6, -2, 0, 1]])
	logp = numpy.array([[NEGINF, NEGINF, NEGINF, NEGINF, 0.0, NEGINF], 
		[-17.388625105797296, -25.109952452723487, -20.33305760532214, -28.085443290422877, NEGINF, -19.639474084287432], 
		[-41.518178585896166, -62.62215498003203, -56.463304390840634, -63.21955480161304, NEGINF, -43.820763354673296], 
		[-86.42085732368164, -67.62244823776697, -71.30752972353059, -70.64290643117542, NEGINF, -69.85376066028503]])
	assert_array_almost_equal(f, logp)


@with_setup(setup_poisson_dense)
def test_poisson_forward():
	f = model.forward([5, 8, 2, 4, 7, 8, 2])
	logp = numpy.array([[NEGINF, NEGINF, NEGINF, NEGINF, 0.0, NEGINF], 
		[-6.724049572762615, -3.874849418805291, -7.396929655216146, -2.6565929124856993, NEGINF, -4.680333421931903], 
		[-6.730147189571273, -6.114939268804046, -15.76343567997591, -6.1491172641470415, NEGINF, -7.498440536740427], 
		[-14.331825716875938, -12.23889386200279, -8.404629462451046, -8.953498331843619, NEGINF, -10.236032891838605], 
		[-15.218653579546471, -13.152885951823485, -13.58596603773139, -10.59764914181018, NEGINF, -12.771059542322048], 
		[-15.25983657200538, -14.222315575321783, -22.039008189195016, -13.683075848076015, NEGINF, -15.403406295745581], 
		[-17.30957501126636, -16.957613004675117, -26.32610484066788, -16.995462506806398, NEGINF, -18.279525087145316], 
		[-25.05974969431663, -23.03771236899512, -19.218787612604835, -19.75231189091333, NEGINF, -21.044274521213687]])
	assert_array_almost_equal(f, logp)

@with_setup(setup_discrete_dense)
def test_discrete_backward():
	f = model.backward(['A', 'B', 'D', 'D', 'C'])
	logp = numpy.array([[-9.86805902419294, -10.666561769922483, -11.09973677168472, -10.617074536069564, -11.092510372852566, NEGINF], 
		[-9.120551817416588, -9.07513780778706, -9.061129343592423, -8.517934491110973, -8.143527673240188, NEGINF], 
		[-6.950743680969357, -6.918207210615943, -6.3171849919534315, -6.328223424806821, -6.314201940019015, NEGINF], 
		[-5.926238762190273, -5.832923518979429, -5.343210135438772, -5.352351255040521, -5.296020007499352, NEGINF], 
		[-4.474141923581687, -3.2834143460057716, -3.547379891840237, -4.474141923581686, -3.8922203781319653, NEGINF], 
		[-2.3025850929940455, -2.3025850929940455, -2.3025850929940455, -2.3025850929940455, NEGINF, 0.0]])
	assert_array_almost_equal(f, logp)

@with_setup(setup_gaussian_dense)
def test_gaussian_backward():
	f = model.backward([3, 5, 8, 19, 13])
	logp = numpy.array([[-24.010022764471987, -24.820878919065986, -25.359784144328874, -24.666641886986977, -24.907606167690343, NEGINF], 
		[-20.47495458748052, -21.390489786220005, -22.08295469697486, -21.390527588974052, -22.081667004696868, NEGINF], 
		[-18.86308443640411, -18.00715991329825, -18.32109321121691, -19.183852463904262, -18.700307092256082, NEGINF], 
		[-13.533310680731095, -12.147019061523556, -12.434699610689767, -13.533307024859651, -12.840163500171148, NEGINF], 
		[-6.217255777912353, -4.830961508172448, -5.1186435298576365, -6.217255656072613, -5.524108597352521, NEGINF], 
		[-2.3025850929940455, -2.3025850929940455, -2.3025850929940455, -2.3025850929940455, NEGINF, 0.0]])
	assert_array_almost_equal(f, logp)


@with_setup(setup_multivariate_gaussian_dense)
def test_multivariate_gaussian_backward():
	f = model.backward([[0, 1, 5, 2, 3], [2, 4, 1, 5, 6], [-4, 6, -2, 0, 1]])
	logp = numpy.array([[-68.2539137567533, -69.16076639610863, -69.84868309756291, -69.16857768159394, -69.85376066028503, NEGINF], 
		[-52.47579136451719, -53.392079325925124, -54.08522496164164, -53.392081629466915, -54.08522649261687, NEGINF], 
		[-28.335582428803374, -28.267824434424867, -28.247424322493245, -27.65418843714621, -27.260167817399534, NEGINF], 
		[-2.3025850929940455, -2.3025850929940455, -2.3025850929940455, -2.3025850929940455, NEGINF, 0.0]])
	assert_array_almost_equal(f, logp)


@with_setup(setup_poisson_dense)
def test_poisson_backward():
	f = model.backward([5, 8, 2, 4, 7, 8, 2])
	logp = numpy.array([[-21.691907586187032, -21.73799328948721, -21.138366991878907, -21.083291604178275, -21.044274521213687, NEGINF], 
		[-18.8959690490499, -19.147279755417475, -18.96895836891973, -18.554384880883024, -18.344028493972242, NEGINF], 
		[-16.436767297608355, -15.50289984292892, -15.520463678389667, -16.044844087024796, -15.741349651348344, NEGINF], 
		[-13.746179478239421, -13.67833968609671, -13.103892290217432, -13.104515632505755, -13.064497897488891, NEGINF], 
		[-10.900016194765097, -11.134592598694933, -10.70914236553333, -10.534402907712945, -10.472655441966987, NEGINF], 
		[-8.08666946806842, -8.338743855593863, -8.159657556177146, -7.746210358522981, -7.536781914211483, NEGINF], 
		[-5.624801928499422, -4.698024820874441, -4.71562372030326, -5.232043004902404, -4.927999971986549, NEGINF], 
		[-2.3025850929940455, -2.3025850929940455, -2.3025850929940455, -2.3025850929940455, NEGINF, 0.0]])
	assert_array_almost_equal(f, logp)

@with_setup(setup, teardown)
def test_viterbi_train():
	seqs = [list(x) for x in ['ACT', 'ACT', 'ACC', 'ACTC', 'ACT', 'ACT', 'CCT', 
	'CCC', 'AAT', 'CT', 'AT', 'CT', 'CT', 'CT', 'CT', 'CT', 'CT', 
	'ACT', 'ACT', 'CT', 'ACT', 'CT', 'CT', 'CT', 'CT']]

	total_improvement = model.fit(seqs, 
									 algorithm='viterbi', 
									 verbose=False, 
									 use_pseudocount=True)

	assert_equal(round(total_improvement, 4), 83.2834)


@with_setup(setup, teardown)
def test_viterbi_train_no_pseudocount():
	seqs = [list(x) for x in ['ACT', 'ACT', 'ACC', 'ACTC', 'ACT', 'ACT', 'CCT', 
	'CCC', 'AAT', 'CT', 'AT', 'CT', 'CT', 'CT', 'CT', 'CT', 'CT', 
	'ACT', 'ACT', 'CT', 'ACT', 'CT', 'CT', 'CT', 'CT']]

	total_improvement = model.fit(seqs, 
									 algorithm='viterbi', 
									 verbose=False, 
									 use_pseudocount=False)

	assert_equal(round(total_improvement, 4), 84.9318)


@with_setup(setup, teardown)
def test_viterbi_train_w_pseudocount():
	seqs = [list(x) for x in ['ACT', 'ACT', 'ACC', 'ACTC', 'ACT', 'ACT', 'CCT', 
	'CCC', 'AAT', 'CT', 'AT', 'CT', 'CT', 'CT', 'CT', 'CT', 'CT', 
	'ACT', 'ACT', 'CT', 'ACT', 'CT', 'CT', 'CT', 'CT']]

	total_improvement = model.fit(seqs, 
									 algorithm='viterbi', 
									 verbose=False, 
									 transition_pseudocount=1.)

	assert_equal(round(total_improvement, 4), 79.4713)


@with_setup(setup, teardown)
def test_viterbi_train_w_pseudocount_priors():
	seqs = [list(x) for x in ['ACT', 'ACT', 'ACC', 'ACTC', 'ACT', 'ACT', 'CCT', 
	'CCC', 'AAT', 'CT', 'AT', 'CT', 'CT', 'CT', 'CT', 'CT', 'CT', 
	'ACT', 'ACT', 'CT', 'ACT', 'CT', 'CT', 'CT', 'CT']]

	total_improvement = model.fit(seqs, 
									 algorithm='viterbi', 
									 verbose=False, 
									 transition_pseudocount=0.278,
									 use_pseudocount=True)

	assert_equal(round(total_improvement, 4), 81.7439)


@with_setup(setup, teardown)
def test_viterbi_train_w_inertia():
	seqs = [list(x) for x in ['ACT', 'ACT', 'ACC', 'ACTC', 'ACT', 'ACT', 'CCT', 
	'CCC', 'AAT', 'CT', 'AT', 'CT', 'CT', 'CT', 'CT', 'CT', 'CT', 
	'ACT', 'ACT', 'CT', 'ACT', 'CT', 'CT', 'CT', 'CT']]

	total_improvement = model.fit(seqs, 
									 algorithm='viterbi', 
									 verbose=False, 
									 edge_inertia=0.193)

	assert_equal(round(total_improvement, 4), 84.9318)


@with_setup(setup, teardown)
def test_viterbi_train_w_inertia2():
	seqs = [list(x) for x in ['ACT', 'ACT', 'ACC', 'ACTC', 'ACT', 'ACT', 'CCT', 
	'CCC', 'AAT', 'CT', 'AT', 'CT', 'CT', 'CT', 'CT', 'CT', 'CT', 
	'ACT', 'ACT', 'CT', 'ACT', 'CT', 'CT', 'CT', 'CT']]

	total_improvement = model.fit(seqs, 
									 algorithm='viterbi', 
									 verbose=False, 
									 edge_inertia=0.82)

	assert_equal(round(total_improvement, 4), 84.9318)


@with_setup(setup, teardown)
def test_viterbi_train_w_pseudocount_inertia():
	seqs = [list(x) for x in ['ACT', 'ACT', 'ACC', 'ACTC', 'ACT', 'ACT', 'CCT', 
	'CCC', 'AAT', 'CT', 'AT', 'CT', 'CT', 'CT', 'CT', 'CT', 'CT', 
	'ACT', 'ACT', 'CT', 'ACT', 'CT', 'CT', 'CT', 'CT']]

	total_improvement = model.fit(seqs, 
									 algorithm='viterbi', 
									 verbose=False, 
									 edge_inertia=0.23,
									 use_pseudocount=True) 

	assert_equal(round(total_improvement, 4), 83.2834)


@with_setup(setup, teardown)
def test_bw_train():
	seqs = [list(x) for x in ['ACT', 'ACT', 'ACC', 'ACTC', 'ACT', 'ACT', 'CCT', 
		'CCC', 'AAT', 'CT', 'AT', 'CT', 'CT', 'CT', 'CT', 'CT', 'CT', 
		'ACT', 'ACT', 'CT', 'ACT', 'CT', 'CT', 'CT', 'CT']]

	total_improvement = model.fit(seqs, 
									 algorithm='baum-welch', 
									 verbose=False, 
									 use_pseudocount=True,
									 max_iterations=5)

	assert_equal(round(total_improvement, 4), 83.1132)


@with_setup(setup_multivariate_discrete, teardown)
def test_bw_multivariate_discrete_train():
	seqs = [[['A', 'A'], ['A', 'C'], ['C', 'T']], [['A', 'A'], ['C', 'C'], ['T', 'T']],
			[['A', 'A'], ['A', 'C'], ['C', 'C'], ['T', 'T']], [['A', 'A'], ['C', 'C']]]

	total_improvement = model.fit(seqs, 
									 algorithm='baum-welch', 
									 verbose=False, 
									 use_pseudocount=True,
									 max_iterations=5)

	assert_equal(round(total_improvement, 4), 13.3622)


@with_setup(setup_multivariate_gaussian, teardown)
def test_bw_multivariate_gaussian_train():
	seqs = [[[5, 8], [8, 10], [13, 17], [-3, -4]], [[6, 7], [13, 16], [12, 11], [-6, -7]], 
			[[4, 6], [13, 15], [-4, -7]], [[6, 5], [14, 18], [-7, -5]]]

	total_improvement = model.fit(seqs, 
									 algorithm='baum-welch', 
									 verbose=False, 
									 use_pseudocount=True,
									 max_iterations=5)

	assert_equal(round(total_improvement, 4), 24.7013)

@with_setup(setup, teardown)
def test_bw_train_json():
	seqs = [list(x) for x in ['ACT', 'ACT', 'ACC', 'ACTC', 'ACT', 'ACT', 'CCT', 
		'CCC', 'AAT', 'CT', 'AT', 'CT', 'CT', 'CT', 'CT', 'CT', 'CT', 
		'ACT', 'ACT', 'CT', 'ACT', 'CT', 'CT', 'CT', 'CT']]

	total_improvement = model.fit(seqs, 
									 algorithm='baum-welch', 
									 verbose=False, 
									 use_pseudocount=True,
									 max_iterations=5)

	assert_equal(round(total_improvement, 4), 83.1132)
	assert_almost_equal(sum(model.log_probability(seq) for seq in seqs), -42.2341, 4)

	hmm = HiddenMarkovModel.from_json(model.to_json())
	assert_almost_equal(sum(model.log_probability(seq) for seq in seqs), -42.2341, 4)


@with_setup(setup, teardown)
def test_bw_train_no_pseudocount():
	seqs = [list(x) for x in ['ACT', 'ACT', 'ACC', 'ACTC', 'ACT', 'ACT', 'CCT', 
		'CCC', 'AAT', 'CT', 'AT', 'CT', 'CT', 'CT', 'CT', 'CT', 'CT', 
		'ACT', 'ACT', 'CT', 'ACT', 'CT', 'CT', 'CT', 'CT']]
	
	total_improvement = model.fit(seqs, 
									 algorithm='baum-welch', 
									 verbose=False, 
									 use_pseudocount=False,
									 max_iterations=5)
 
	assert_equal(round(total_improvement, 4), 85.681)


@with_setup(setup, teardown)
def test_bw_train_w_pseudocount():
	seqs = [list(x) for x in ['ACT', 'ACT', 'ACC', 'ACTC', 'ACT', 'ACT', 'CCT', 
		'CCC', 'AAT', 'CT', 'AT', 'CT', 'CT', 'CT', 'CT', 'CT', 'CT', 
		'ACT', 'ACT', 'CT', 'ACT', 'CT', 'CT', 'CT', 'CT']]

	total_improvement = model.fit(seqs, 
									 algorithm='baum-welch', 
									 verbose=False, 
									 transition_pseudocount=0.123,
									 max_iterations=5)
	
	assert_equal(round(total_improvement, 4), 84.9408)


@with_setup(setup, teardown)
def test_bw_train_w_pseudocount_priors():
	seqs = [list(x) for x in ['ACT', 'ACT', 'ACC', 'ACTC', 'ACT', 'ACT', 'CCT', 
		'CCC', 'AAT', 'CT', 'AT', 'CT', 'CT', 'CT', 'CT', 'CT', 'CT', 
		'ACT', 'ACT', 'CT', 'ACT', 'CT', 'CT', 'CT', 'CT']]

	total_improvement = model.fit(seqs, 
									 algorithm='baum-welch', 
									 verbose=False, 
									 transition_pseudocount=0.278,
									 use_pseudocount=True,
									 max_iterations=5)
	 
	assert_equal(round(total_improvement, 4), 81.2265)


@with_setup(setup, teardown)
def test_bw_train_w_inertia():
	seqs = [list(x) for x in ['ACT', 'ACT', 'ACC', 'ACTC', 'ACT', 'ACT', 'CCT', 
		'CCC', 'AAT', 'CT', 'AT', 'CT', 'CT', 'CT', 'CT', 'CT', 'CT', 
		'ACT', 'ACT', 'CT', 'ACT', 'CT', 'CT', 'CT', 'CT']]

	total_improvement = model.fit(seqs, 
									 algorithm='baum-welch', 
									 verbose=False, 
									 edge_inertia=0.193,
									 max_iterations=5)
	 
	assert_equal(round(total_improvement, 4), 85.0528)


@with_setup(setup, teardown)
def test_bw_train_w_inertia2():
	seqs = [list(x) for x in ['ACT', 'ACT', 'ACC', 'ACTC', 'ACT', 'ACT', 'CCT', 
		'CCC', 'AAT', 'CT', 'AT', 'CT', 'CT', 'CT', 'CT', 'CT', 'CT', 
		'ACT', 'ACT', 'CT', 'ACT', 'CT', 'CT', 'CT', 'CT']]

	total_improvement = model.fit(seqs, 
									 algorithm='baum-welch', 
									 verbose=False, 
									 edge_inertia=0.82,
									 max_iterations=5)
  
	assert_equal(round(total_improvement, 4), 72.5134)


@with_setup(setup, teardown)
def test_bw_train_w_pseudocount_inertia():
	seqs = [list(x) for x in ['ACT', 'ACT', 'ACC', 'ACTC', 'ACT', 'ACT', 'CCT', 
		'CCC', 'AAT', 'CT', 'AT', 'CT', 'CT', 'CT', 'CT', 'CT', 'CT', 
		'ACT', 'ACT', 'CT', 'ACT', 'CT', 'CT', 'CT', 'CT']]

	total_improvement = model.fit(seqs, 
									 algorithm='baum-welch', 
									 verbose=False, 
									 edge_inertia=0.02,
									 use_pseudocount=True,
									 max_iterations=5)
 
	assert_equal(round(total_improvement, 4), 83.0764)


@with_setup(setup, teardown)
def test_bw_train_w_frozen_distributions():
	seqs = [list(x) for x in ['ACT', 'ACT', 'ACC', 'ACTC', 'ACT', 'ACT', 'CCT', 
		'CCC', 'AAT', 'CT', 'AT', 'CT', 'CT', 'CT', 'CT', 'CT', 'CT', 
		'ACT', 'ACT', 'CT', 'ACT', 'CT', 'CT', 'CT', 'CT']]

	total_improvement = model.fit(seqs, 
									 algorithm='baum-welch', 
									 verbose=False, 
									 distribution_inertia=1.00,
									 max_iterations=5)
  
	assert_equal(round(total_improvement, 4), 64.474)


@with_setup(setup, teardown)
def test_bw_train_w_frozen_edges():
	seqs = [list(x) for x in ['ACT', 'ACT', 'ACC', 'ACTC', 'ACT', 'ACT', 'CCT', 
		'CCC', 'AAT', 'CT', 'AT', 'CT', 'CT', 'CT', 'CT', 'CT', 'CT', 
		'ACT', 'ACT', 'CT', 'ACT', 'CT', 'CT', 'CT', 'CT']]

	total_improvement = model.fit(seqs, 
									 algorithm='baum-welch', 
									 verbose=False, 
									 edge_inertia=1.00,
									 max_iterations=5)

	assert_equal(round(total_improvement, 4), 44.0208)


@with_setup(setup, teardown)
def test_bw_train_w_edge_a_distribution_inertia():
	seqs = [list(x) for x in ['ACT', 'ACT', 'ACC', 'ACTC', 'ACT', 'ACT', 'CCT', 
		'CCC', 'AAT', 'CT', 'AT', 'CT', 'CT', 'CT', 'CT', 'CT', 'CT', 
		'ACT', 'ACT', 'CT', 'ACT', 'CT', 'CT', 'CT', 'CT']]

	total_improvement = model.fit(seqs, 
									 algorithm='baum-welch', 
									 verbose=False, 
									 edge_inertia=0.5,
									 distribution_inertia=0.5,
									 max_iterations=5)
 
	assert_equal(round(total_improvement, 4), 81.5447)


@with_setup(setup, teardown)
def test_bw_train_parallel():
	seqs = [list(x) for x in ['ACT', 'ACT', 'ACC', 'ACTC', 'ACT', 'ACT', 'CCT', 
		'CCC', 'AAT', 'CT', 'AT', 'CT', 'CT', 'CT', 'CT', 'CT', 'CT', 
		'ACT', 'ACT', 'CT', 'ACT', 'CT', 'CT', 'CT', 'CT']]

	total_improvement = model.fit(seqs, 
									 algorithm='baum-welch', 
									 verbose=False, 
									 use_pseudocount=True,
									 max_iterations=5,
									 n_jobs=2)

	assert_equal(round(total_improvement, 4), 83.1132)


@with_setup(setup, teardown)
def test_bw_train_no_pseudocount_parallel():
	seqs = [list(x) for x in ['ACT', 'ACT', 'ACC', 'ACTC', 'ACT', 'ACT', 'CCT', 
		'CCC', 'AAT', 'CT', 'AT', 'CT', 'CT', 'CT', 'CT', 'CT', 'CT', 
		'ACT', 'ACT', 'CT', 'ACT', 'CT', 'CT', 'CT', 'CT']]
	
	total_improvement = model.fit(seqs, 
									 algorithm='baum-welch', 
									 verbose=False, 
									 use_pseudocount=False,
									 max_iterations=5,
									 n_jobs=2)
 
	assert_equal(round(total_improvement, 4), 85.681)


@with_setup(setup, teardown)
def test_bw_train_w_pseudocount_parallel():
	seqs = [list(x) for x in ['ACT', 'ACT', 'ACC', 'ACTC', 'ACT', 'ACT', 'CCT', 
		'CCC', 'AAT', 'CT', 'AT', 'CT', 'CT', 'CT', 'CT', 'CT', 'CT', 
		'ACT', 'ACT', 'CT', 'ACT', 'CT', 'CT', 'CT', 'CT']]

	total_improvement = model.fit(seqs, 
									 algorithm='baum-welch', 
									 verbose=False, 
									 transition_pseudocount=0.123,
									 max_iterations=5,
									 n_jobs=2)
	
	assert_equal(round(total_improvement, 4), 84.9408)


@with_setup(setup, teardown)
def test_bw_train_w_pseudocount_priors_parallel():
	seqs = [list(x) for x in ['ACT', 'ACT', 'ACC', 'ACTC', 'ACT', 'ACT', 'CCT', 
		'CCC', 'AAT', 'CT', 'AT', 'CT', 'CT', 'CT', 'CT', 'CT', 'CT', 
		'ACT', 'ACT', 'CT', 'ACT', 'CT', 'CT', 'CT', 'CT']]

	total_improvement = model.fit(seqs, 
									 algorithm='baum-welch', 
									 verbose=False, 
									 transition_pseudocount=0.278,
									 use_pseudocount=True,
									 max_iterations=5,
									 n_jobs=2)
	 
	assert_equal(round(total_improvement, 4), 81.2265)


@with_setup(setup, teardown)
def test_bw_train_w_inertia_parallel():
	seqs = [list(x) for x in ['ACT', 'ACT', 'ACC', 'ACTC', 'ACT', 'ACT', 'CCT', 
		'CCC', 'AAT', 'CT', 'AT', 'CT', 'CT', 'CT', 'CT', 'CT', 'CT', 
		'ACT', 'ACT', 'CT', 'ACT', 'CT', 'CT', 'CT', 'CT']]

	total_improvement = model.fit(seqs, 
									 algorithm='baum-welch', 
									 verbose=False, 
									 edge_inertia=0.193,
									 max_iterations=5,
									 n_jobs=2)
	 
	assert_equal(round(total_improvement, 4), 85.0528)


@with_setup(setup, teardown)
def test_bw_train_w_inertia2_parallel():
	seqs = [list(x) for x in ['ACT', 'ACT', 'ACC', 'ACTC', 'ACT', 'ACT', 'CCT', 
		'CCC', 'AAT', 'CT', 'AT', 'CT', 'CT', 'CT', 'CT', 'CT', 'CT', 
		'ACT', 'ACT', 'CT', 'ACT', 'CT', 'CT', 'CT', 'CT']]

	total_improvement = model.fit(seqs, 
									 algorithm='baum-welch', 
									 verbose=False, 
									 edge_inertia=0.82,
									 max_iterations=5,
									 n_jobs=2)
  
	assert_equal(round(total_improvement, 4), 72.5134)


@with_setup(setup, teardown)
def test_bw_train_w_pseudocount_inertia_parallel():
	seqs = [list(x) for x in ['ACT', 'ACT', 'ACC', 'ACTC', 'ACT', 'ACT', 'CCT', 
		'CCC', 'AAT', 'CT', 'AT', 'CT', 'CT', 'CT', 'CT', 'CT', 'CT', 
		'ACT', 'ACT', 'CT', 'ACT', 'CT', 'CT', 'CT', 'CT']]

	total_improvement = model.fit(seqs, 
									 algorithm='baum-welch', 
									 verbose=False, 
									 edge_inertia=0.02,
									 use_pseudocount=True,
									 max_iterations=5,
									 n_jobs=2)
 
	assert_equal(round(total_improvement, 4), 83.0764)


@with_setup(setup, teardown)
def test_bw_train_w_frozen_distributions_parallel():
	seqs = [list(x) for x in ['ACT', 'ACT', 'ACC', 'ACTC', 'ACT', 'ACT', 'CCT', 
		'CCC', 'AAT', 'CT', 'AT', 'CT', 'CT', 'CT', 'CT', 'CT', 'CT', 
		'ACT', 'ACT', 'CT', 'ACT', 'CT', 'CT', 'CT', 'CT']]

	total_improvement = model.fit(seqs, 
									 algorithm='baum-welch', 
									 verbose=False, 
									 distribution_inertia=1.00,
									 max_iterations=5,
									 n_jobs=2)
  
	assert_equal(round(total_improvement, 4), 64.474)


@with_setup(setup, teardown)
def test_bw_train_w_frozen_edges_parallel():
	seqs = [list(x) for x in ['ACT', 'ACT', 'ACC', 'ACTC', 'ACT', 'ACT', 'CCT', 
		'CCC', 'AAT', 'CT', 'AT', 'CT', 'CT', 'CT', 'CT', 'CT', 'CT', 
		'ACT', 'ACT', 'CT', 'ACT', 'CT', 'CT', 'CT', 'CT']]

	total_improvement = model.fit(seqs, 
									 algorithm='baum-welch', 
									 verbose=False, 
									 edge_inertia=1.00,
									 max_iterations=5,
									 n_jobs=2)

	assert_equal(round(total_improvement, 4), 44.0208)


@with_setup(setup, teardown)
def test_bw_train_w_edge_a_distribution_inertia():
	seqs = [list(x) for x in ['ACT', 'ACT', 'ACC', 'ACTC', 'ACT', 'ACT', 'CCT', 
		'CCC', 'AAT', 'CT', 'AT', 'CT', 'CT', 'CT', 'CT', 'CT', 'CT', 
		'ACT', 'ACT', 'CT', 'ACT', 'CT', 'CT', 'CT', 'CT']]

	total_improvement = model.fit(seqs, 
									 algorithm='baum-welch', 
									 verbose=False, 
									 edge_inertia=0.5,
									 distribution_inertia=0.5,
									 max_iterations=5,
									 n_jobs=2)
 
	assert_equal(round(total_improvement, 4), 81.5447)


@with_setup(setup, teardown)
def test_dimension():
	# single dimensions
	s1d1 = State(NormalDistribution(5, 2))
	s2d1 = State(NormalDistribution(0, 1))
	s3d1 = State(UniformDistribution(0, 10))

	hmmd1 = HiddenMarkovModel()

	assert_equal(hmmd1.d, 0)

	hmmd1.add_transition(hmmd1.start, s1d1, 0.5)
	hmmd1.add_transition(hmmd1.start, s2d1, 0.5)
	hmmd1.add_transition(s1d1, s3d1, 1)
	hmmd1.add_transition(s2d1, s3d1, 1)
	hmmd1.add_transition(s3d1, s1d1, 0.5)
	hmmd1.add_transition(s3d1, s2d1, 0.5)

	assert_equal(hmmd1.d, 0)

	hmmd1.bake()

	assert_equal(hmmd1.d, 1)

	# multiple dimensions
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


def test_serialization_univariate():
	s1 = State(NormalDistribution(1, 2))
	s2 = State(NormalDistribution(5, 1))
	s3 = State(NormalDistribution(8, 2))

	hmm1 = HiddenMarkovModel()

	hmm1.add_transition(hmm1.start, s1, 1)
	hmm1.add_transition(s1, s1, 0.8)
	hmm1.add_transition(s1, s2, 0.2)
	hmm1.add_transition(s2, s2, 0.9)
	hmm1.add_transition(s2, s3, 0.1)
	hmm1.add_transition(s3, s3, 0.95)
	hmm1.add_transition(s3, hmm1.end, 0.05)

	hmm1.bake()

	hmm2 = pickle.loads(pickle.dumps(hmm1))

	for i in range(10):
		sequence = numpy.random.randn(10)
		assert_almost_equal(hmm1.log_probability(sequence), hmm2.log_probability(sequence))


def test_serialization_multivariate():
	s1 = State(MultivariateGaussianDistribution([1, 4], [[.1, 0], [0, 1]]))
	s2 = State(MultivariateGaussianDistribution([-1, 1], [[1, 0], [0, 7]]))
	s3 = State(MultivariateGaussianDistribution([3, 5], [[.5, 0], [0, 6]]))

	hmm1 = HiddenMarkovModel()

	hmm1.add_transition(hmm1.start, s1, 1)
	hmm1.add_transition(s1, s1, 0.8)
	hmm1.add_transition(s1, s2, 0.2)
	hmm1.add_transition(s2, s2, 0.9)
	hmm1.add_transition(s2, s3, 0.1)
	hmm1.add_transition(s3, s3, 0.95)
	hmm1.add_transition(s3, hmm1.end, 0.05)

	hmm1.bake()

	hmm2 = pickle.loads(pickle.dumps(hmm1))

	for i in range(10):
		sequence = numpy.random.randn(10, 2)
		assert_almost_equal(hmm1.log_probability(sequence), hmm2.log_probability(sequence))


@with_setup(setup_discrete_dense, teardown)
def test_discrete_from_samples():
	X = [model.sample() for i in range(25)]
	model2 = HiddenMarkovModel.from_samples(DiscreteDistribution, 4, X, max_iterations=25)

	logp1 = sum(map(model.log_probability, X))
	logp2 = sum(map(model2.log_probability, X))

	assert_greater(logp2, logp1)


@with_setup(setup_gaussian_dense, teardown)
def test_gaussian_from_samples():
	X = [model.sample() for i in range(25)]
	model2 = HiddenMarkovModel.from_samples(NormalDistribution, 4, X, max_iterations=25)

	logp1 = sum(map(model.log_probability, X))
	logp2 = sum(map(model2.log_probability, X))

	assert_greater(logp2, logp1)


@with_setup(setup_multivariate_gaussian_dense, teardown)
def test_multivariate_gaussian_from_samples():
	X = [model.sample() for i in range(25)]
	model2 = HiddenMarkovModel.from_samples(MultivariateGaussianDistribution, 4, X, max_iterations=25)

	logp1 = sum(map(model.log_probability, X))
	logp2 = sum(map(model2.log_probability, X))

	assert_greater(logp2, logp1)

@with_setup(setup_discrete_dense, teardown)
def test_discrete_from_samples_end_state():
    X = [model.sample() for i in range(25)]
    model2 = HiddenMarkovModel.from_samples(DiscreteDistribution, 4, X, max_iterations=25, end_state=True)

    #We get non-zero end probabilities for each state
    assert_greater(model2.dense_transition_matrix()[0][model2.end_index],0)
    assert_greater(model2.dense_transition_matrix()[1][model2.end_index],0)
    assert_greater(model2.dense_transition_matrix()[2][model2.end_index],0)
    assert_greater(model2.dense_transition_matrix()[3][model2.end_index],0)

@with_setup(setup_discrete_dense, teardown)
def test_discrete_from_samples_no_end_state():
    X = [model.sample() for i in range(25)]
    model2 = HiddenMarkovModel.from_samples(DiscreteDistribution, 4, X, max_iterations=25, end_state=False)

    #We don't have end probabilities for each state
    assert_equal(model2.dense_transition_matrix()[0][model2.end_index],0)
    assert_equal(model2.dense_transition_matrix()[1][model2.end_index],0)
    assert_equal(model2.dense_transition_matrix()[2][model2.end_index],0)
    assert_equal(model2.dense_transition_matrix()[3][model2.end_index],0)
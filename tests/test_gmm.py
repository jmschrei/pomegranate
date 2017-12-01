from pomegranate import *
from nose.tools import with_setup
from nose.tools import assert_true
from nose.tools import assert_equal
from nose.tools import assert_greater
from nose.tools import assert_raises
from nose.tools import assert_not_equal
from numpy.testing import assert_almost_equal
from numpy.testing import assert_array_equal
from numpy.testing import assert_array_almost_equal
import random
import pickle
import numpy as np

np.random.seed(0)
random.seed(0)

def setup_nothing():
	pass

def setup_multivariate_gaussian():
	"""
	Set up a five component Gaussian mixture model, where each component
	is a multivariate Gaussian distribution.
	"""

	global gmm

	mu = np.arange(5)
	cov = np.eye(5)

	mgs = [MultivariateGaussianDistribution(mu*i, cov) for i in range(5)]
	gmm = GeneralMixtureModel(mgs)


def setup_multivariate_mixed():
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

	global gmm
	gmm = GeneralMixtureModel([d1, d2])


def setup_univariate_gaussian():
	"""
	Set up a three component univariate Gaussian model.
	"""

	global gmm
	gmm = GeneralMixtureModel([NormalDistribution(i*3, 1) for i in range(3)])


def setup_univariate_mixed():
	"""Set up a four component univariate mixed model."""

	global gmm
	d1 = ExponentialDistribution(5)
	d2 = NormalDistribution(0, 1.2)
	d3 = LogNormalDistribution(0.3, 1.4)
	d4 = PoissonDistribution(5)
	gmm = GeneralMixtureModel([d1, d2, d3, d4])


def teardown():
	"""
	Teardown the model, so delete it.
	"""

	pass


@with_setup(setup_multivariate_gaussian, teardown)
def test_gmm_multivariate_gaussian_log_probability():
	X = numpy.array([[1.1, 2.7, 3.0, 4.8, 6.2],
					[1.8, 2.1, 3.1, 5.2, 6.5],
					[0.9, 2.2, 3.2, 5.0, 5.8],
					[1.0, 2.1, 3.5, 4.3, 5.2],
					[1.2, 2.9, 3.1, 4.2, 5.5],
					[1.8, 1.9, 3.0, 4.9, 5.7],
					[1.2, 3.1, 2.9, 4.2, 5.9],
					[1.0, 2.9, 3.9, 4.1, 6.0]])

	logp_t = [ -9.8405678, -9.67171158, -9.71615297, -9.89404726, 
	-10.93812212, -11.06611533, -11.31473392, -10.79220257]
	logp = gmm.log_probability(X)

	assert_array_almost_equal(logp, logp_t)


@with_setup(setup_multivariate_mixed, teardown)
def test_gmm_multivariate_mixed_log_probability():
	X = numpy.array([[1.1, 2.7, 3.0, 4.8, 6.2],
					[1.8, 2.1, 3.1, 5.2, 6.5],
					[0.9, 2.2, 3.2, 5.0, 5.8],
					[1.0, 2.1, 3.5, 4.3, 5.2],
					[1.2, 2.9, 3.1, 4.2, 5.5],
					[1.8, 1.9, 3.0, 4.9, 5.7],
					[1.2, 3.1, 2.9, 4.2, 5.9],
					[1.0, 2.9, 3.9, 4.1, 6.0]])

	logp_t = [-33.75384631, -34.1714099,  -32.59702495, -27.39375394, 
	-30.66715208, -30.52489174, -31.71056782, -30.79589904]
	logp = gmm.log_probability(X)

	assert_array_almost_equal(logp, logp_t)


@with_setup(setup_univariate_gaussian, teardown)
def test_gmm_univariate_gaussian_log_probability():
	X = np.array([[1.1], [2.7], [3.0], [4.8], [6.2]])
	logp = [-2.35925975, -2.03120691, -1.99557605, -2.39638244, -2.03147258]
	assert_array_almost_equal(gmm.log_probability(X), logp)

	X = np.array([[1.8], [2.1], [3.1], [5.2], [6.5]])
	logp = [-2.39618117, -2.26893273, -1.9995911,  -2.22202965, -2.14007514]
	assert_array_almost_equal(gmm.log_probability(X), logp)

	X = np.array([[0.9], [2.2], [3.2], [5.0], [5.8]])
	logp = [-2.26957032, -2.22113386, -2.01155305, -2.31613252, -2.01751101]
	assert_array_almost_equal(gmm.log_probability(X), logp)

	X = np.array([[1.0], [2.1], [3.5], [4.3], [5.2]])
	logp = [-2.31613252, -2.26893273, -2.09160506, -2.42491769, -2.22202965]
	assert_array_almost_equal(gmm.log_probability(X), logp)

	X = np.array([[1.2], [2.9], [3.1], [4.2], [5.5]])
	logp = [-2.39638244, -1.9995911,  -1.9995911,  -2.39618117, -2.09396318]
	assert_array_almost_equal(gmm.log_probability(X), logp)

	X = np.array([[1.8], [1.9], [3.0], [4.9], [5.7]])
	logp = [-2.39618117, -2.35895351, -1.99557605, -2.35925975, -2.03559364]
	assert_array_almost_equal(gmm.log_probability(X), logp)

	X = np.array([[1.2], [3.1], [2.9], [4.2], [5.9]])
	logp = [-2.39638244, -1.9995911,  -1.9995911,  -2.39618117, -2.00766654]
	assert_array_almost_equal(gmm.log_probability(X), logp)

	X = np.array([[1.0], [2.9], [3.9], [4.1], [6.0]])
	logp = [-2.31613252, -1.9995911,  -2.26893273, -2.35895351, -2.00650306]
	assert_array_almost_equal(gmm.log_probability(X), logp)


@with_setup(setup_univariate_mixed, teardown)
def test_gmm_mixed_log_probability():
	X = np.array([[1.1], [2.7], [3.0], [4.8], [6.2]])
	logp = [-2.01561437061559, -2.7951359521294536, -2.8314639809821918, 
			-2.9108132001193265, -3.1959940375620945]
	assert_array_almost_equal(gmm.log_probability(X), logp)

	X = np.array([[1.8], [2.1], [3.1], [5.2], [6.5]])
	logp = [-2.4758296236378774, -2.6201420691379314, -2.8383034405278975, 
			-2.966292939154318, -3.2891059316267657]
	assert_array_almost_equal(gmm.log_probability(X), logp)

	X = np.array([[0.9], [2.2], [3.2], [5.0], [5.8]])
	logp = [-1.8326789033955484, -2.660084457680723, -2.8432332382831653, 
			-2.9359363402629808, -3.0888100501157312]
	assert_array_almost_equal(gmm.log_probability(X), logp)

	X = np.array([[1.0], [2.1], [3.5], [4.3], [5.2]])
	logp = [-1.9296484854978633, -2.6201420691379314, -2.850660742142904, 
			-2.8698462265150058, -2.966292939154318]
	assert_array_almost_equal(gmm.log_probability(X), logp)

	X = np.array([[1.2], [2.9], [3.1], [4.2], [5.5]])
	logp = [-2.0938404063492411, -2.8222781320451804, -2.8383034405278975, 
			-2.8650550479352965, -3.0216937107055277]
	assert_array_almost_equal(gmm.log_probability(X), logp)

	X = np.array([[1.8], [1.9], [3.0], [4.9], [5.7]])
	logp = [-2.4758296236378774, -2.527780058213545, -2.8314639809821918, 
			-2.9227254358998933, -3.0651517040535605]
	assert_array_almost_equal(gmm.log_probability(X), logp)

	X = np.array([[1.2], [3.1], [2.9], [4.2], [5.9]])
	logp = [-2.0938404063492411, -2.8383034405278975, -2.8222781320451804, 
			-2.8650550479352965, -3.1137381280223324]
	assert_array_almost_equal(gmm.log_probability(X), logp)

	X = np.array([[1.0], [2.9], [3.9], [4.1], [6.0]])
	logp = [-1.9296484854978633, -2.8222781320451804, -2.8560160466782816, 
			-2.8612291168066575, -3.1399218398841575]
	assert_array_almost_equal(gmm.log_probability(X), logp)


@with_setup(setup_multivariate_gaussian, teardown)
def test_gmm_multivariate_gaussian_json():
	gmm_2 = GeneralMixtureModel.from_json(gmm.to_json())

	X = np.array([[1.1, 2.7, 3.0, 4.8, 6.2]])
	assert_almost_equal(gmm_2.log_probability(X).sum(), -9.8406, 4)

	X = np.array([[1.8, 2.1, 3.1, 5.2, 6.5]])
	assert_almost_equal(gmm_2.log_probability(X).sum(), -9.6717, 4)

	X = np.array([[0.9, 2.2, 3.2, 5.0, 5.8]])
	assert_almost_equal(gmm_2.log_probability(X).sum(), -9.7162, 4)

	X = np.array([[1.0, 2.1, 3.5, 4.3, 5.2]])
	assert_almost_equal(gmm_2.log_probability(X).sum(), -9.894, 4)

	X = np.array([[1.2, 2.9, 3.1, 4.2, 5.5]])
	assert_almost_equal(gmm_2.log_probability(X).sum(), -10.9381, 4)

	X = np.array([[1.8, 1.9, 3.0, 4.9, 5.7]])
	assert_almost_equal(gmm_2.log_probability(X).sum(), -11.0661, 4)

	X = np.array([[1.2, 3.1, 2.9, 4.2, 5.9]])
	assert_almost_equal(gmm_2.log_probability(X).sum(), -11.3147, 4)

	X = np.array([[1.0, 2.9, 3.9, 4.1, 6.0]])
	assert_almost_equal(gmm_2.log_probability(X).sum(), -10.7922, 4)

@with_setup(setup_multivariate_mixed, teardown)
def test_gmm_multivariate_mixed_json():
	gmm2 = GeneralMixtureModel.from_json(gmm.to_json())

	X = numpy.array([[1.1, 2.7, 3.0, 4.8, 6.2],
					[1.8, 2.1, 3.1, 5.2, 6.5],
					[0.9, 2.2, 3.2, 5.0, 5.8],
					[1.0, 2.1, 3.5, 4.3, 5.2],
					[1.2, 2.9, 3.1, 4.2, 5.5],
					[1.8, 1.9, 3.0, 4.9, 5.7],
					[1.2, 3.1, 2.9, 4.2, 5.9],
					[1.0, 2.9, 3.9, 4.1, 6.0]])

	logp_t = [-33.75384631, -34.1714099,  -32.59702495, -27.39375394, 
	-30.66715208, -30.52489174, -31.71056782, -30.79589904]
	logp1 = gmm.log_probability(X)
	logp2 = gmm2.log_probability(X)

	assert_array_almost_equal(logp2, logp_t)
	assert_array_almost_equal(logp1, logp2)


@with_setup(setup_multivariate_gaussian, teardown)
def test_gmm_multivariate_gaussian_predict_log_proba():
	posterior = np.array([[-2.10001234e+01, -1.23402948e-04, -9.00012340e+00, -4.80001234e+01, -1.17000123e+02],
                          [-2.30009115e+01, -9.11466556e-04, -7.00091147e+00, -4.40009115e+01, -1.11000911e+02]])

	X = np.array([[2., 5., 7., 3., 2.],
		          [1., 2., 5., 2., 5.]])

	assert_almost_equal(gmm.predict_log_proba(X), posterior, 4)
	assert_almost_equal(numpy.exp(gmm.predict_log_proba(X)), gmm.predict_proba(X), 4)


@with_setup(setup_multivariate_gaussian, teardown)
def test_gmm_multivariate_gaussian_predict():
	X = np.array([[2., 5., 7., 3., 2.],
		          [1., 2., 5., 2., 5.],
				  [2., 1., 8., 2., 1.],
				  [4., 3., 8., 1., 2.]])

	assert_almost_equal(gmm.predict(X), gmm.predict_proba(X).argmax(axis=1))


def test_gmm_multivariate_gaussian_fit():
	d1 = MultivariateGaussianDistribution([0, 0], [[1, 0], [0, 1]])
	d2 = MultivariateGaussianDistribution([2, 2], [[1, 0], [0, 1]])
	gmm = GeneralMixtureModel([d1, d2])

	X = np.array([[0.1,  0.7],
		          [1.8,  2.1],
		          [-0.9, -1.2],
		          [-0.0,  0.2],
		          [1.4,  2.9],
		          [1.8,  2.5],
		          [1.4,  3.1],
		          [1.0,  1.0]])

	assert_almost_equal(gmm.fit(X), 15.242416, 4)


def test_gmm_multivariate_gaussian_fit_iterations():
	X = numpy.concatenate([numpy.random.randn(1000, 3) + i for i in range(2)])

	mu = np.ones(3) * 2
	cov = np.eye(3)
	mgs = [MultivariateGaussianDistribution(mu*i, cov) for i in range(2)]
	gmm = GeneralMixtureModel(mgs)

	improvement = gmm.fit(X)

	mgs = [MultivariateGaussianDistribution(mu*i, cov) for i in range(2)]
	gmm = GeneralMixtureModel(mgs)

	assert_greater(improvement, gmm.fit(X, max_iterations=1))

def test_gmm_initialization():
	assert_raises(ValueError, GeneralMixtureModel, [])

	assert_raises(TypeError, GeneralMixtureModel, [NormalDistribution(5, 2), MultivariateGaussianDistribution([5, 2], [[1, 0], [0, 1]])])
	assert_raises(TypeError, GeneralMixtureModel, [NormalDistribution(5, 2), NormalDistribution])

	X = numpy.concatenate((numpy.random.randn(300, 5) + 0.5, numpy.random.randn(200, 5)))

	MGD = MultivariateGaussianDistribution

	gmm1 = GeneralMixtureModel.from_samples(MGD, 2, X, init='first-k')
	gmm2 = GeneralMixtureModel.from_samples(MGD, 2, X, init='first-k', max_iterations=1)
	assert_greater(gmm1.log_probability(X).sum(), gmm2.log_probability(X).sum())

	assert_equal(gmm1.d, 5)
	assert_equal(gmm2.d, 5)

@with_setup(setup_multivariate_gaussian, teardown)
def test_gmm_dimension():
	gmm1 = GeneralMixtureModel([NormalDistribution(0, 1), UniformDistribution(0, 10)])

	assert_equal(gmm.d, 5)
	assert_equal(gmm1.d, 1)

@with_setup(setup_multivariate_gaussian, teardown)
def test_gmm_json():
	univariate = GeneralMixtureModel([NormalDistribution(5, 2), UniformDistribution(0, 10)])

	j_univ = univariate.to_json()
	j_multi = gmm.to_json()

	new_univ = univariate.from_json(j_univ)
	assert_true(isinstance(new_univ.distributions[0], NormalDistribution))
	assert_true(isinstance(new_univ.distributions[1], UniformDistribution))
	assert_true(isinstance(new_univ, GeneralMixtureModel))
	assert_array_equal(univariate.weights, new_univ.weights)

	new_multi = gmm.from_json(j_multi)
	for i in range(5):
		assert_true(isinstance(new_multi.distributions[i], MultivariateGaussianDistribution))

	assert_true(isinstance(new_multi, GeneralMixtureModel))
	assert_array_almost_equal(gmm.weights, new_multi.weights)

@with_setup(setup_multivariate_gaussian, teardown)
def test_gmm_pickling():
	univariate = GeneralMixtureModel(
		[NormalDistribution(5, 2), UniformDistribution(0, 10)],
        weights=np.array([1.0, 2.0]))

	j_univ = pickle.dumps(univariate)
	j_multi = pickle.dumps(gmm)

	new_univ = pickle.loads(j_univ)
	assert_true(isinstance(new_univ.distributions[0], NormalDistribution))
	assert_true(isinstance(new_univ.distributions[1], UniformDistribution))
	assert_true(isinstance(new_univ, GeneralMixtureModel))
	assert_array_equal(univariate.weights, new_univ.weights)

	new_multi = pickle.loads(j_multi)
	for i in range(5):
		assert_true(isinstance(new_multi.distributions[i], MultivariateGaussianDistribution))

	assert_true(isinstance(new_multi, GeneralMixtureModel))
	assert_array_almost_equal(gmm.weights, new_multi.weights)

@with_setup(setup_multivariate_gaussian, teardown)
def test_gmm_multivariate_gaussian_ooc():
	X = numpy.concatenate([numpy.random.randn(1000, 3) + i for i in range(3)])

	gmm = GeneralMixtureModel.from_samples(MultivariateGaussianDistribution,
		3, X, init='first-k', max_iterations=5)
	gmm2 = GeneralMixtureModel.from_samples(MultivariateGaussianDistribution, 
		3, X, init='first-k', max_iterations=5, batch_size=3000)
	gmm3 = GeneralMixtureModel.from_samples(MultivariateGaussianDistribution,
		3, X, init='first-k', max_iterations=5, batch_size=500, batches_per_epoch=6)
	gmm4 = GeneralMixtureModel.from_samples(MultivariateGaussianDistribution, 
		3, X, init='first-k', max_iterations=5, batch_size=500, batches_per_epoch=2)

	assert_almost_equal(gmm.log_probability(X).sum(), gmm2.log_probability(X).sum())
	assert_almost_equal(gmm.log_probability(X).sum(), gmm3.log_probability(X).sum(), -2)
	assert_not_equal(gmm.log_probability(X).sum(), gmm4.log_probability(X).sum())


@with_setup(setup_multivariate_gaussian, teardown)
def test_gmm_multivariate_gaussian_minibatch():
	X = numpy.concatenate([numpy.random.randn(1000, 3) + i for i in range(3)])

	gmm = GeneralMixtureModel.from_samples(MultivariateGaussianDistribution,
		3, X, init='first-k', max_iterations=5)
	gmm2 = GeneralMixtureModel.from_samples(MultivariateGaussianDistribution,
		3, X, init='first-k', max_iterations=5, batch_size=500, batches_per_epoch=1)
	gmm3 = GeneralMixtureModel.from_samples(MultivariateGaussianDistribution, 
		3, X, init='first-k', max_iterations=5, batch_size=500, batches_per_epoch=6)
	gmm4 = GeneralMixtureModel.from_samples(MultivariateGaussianDistribution,
		3, X, init='first-k', max_iterations=5, batch_size=3000, batches_per_epoch=1)

	assert_not_equal(gmm.log_probability(X).sum(), gmm2.log_probability(X).sum())
	assert_not_equal(gmm2.log_probability(X).sum(), gmm3.log_probability(X).sum())
	assert_raises(AssertionError, assert_array_almost_equal, gmm3.log_probability(X), 
		gmm.log_probability(X))

	assert_array_equal(gmm.log_probability(X), gmm4.log_probability(X))


def test_gmm_multivariate_gaussian_nan_from_samples():
	numpy.random.seed(1)
	X = numpy.concatenate([numpy.random.normal(0, 1, size=(300, 3)), 
						   numpy.random.normal(8, 1, size=(300, 3))])
	numpy.random.shuffle(X)
	idxs = numpy.random.choice(numpy.arange(1800), replace=False, size=500)
	i, j = idxs // 3, idxs % 3
	
	X_nan = X.copy()
	X_nan[i, j] = numpy.nan

	mu1t = [-0.036813615311095164, 0.05802948506749107, 0.09725454186262805]
	cov1t = [[ 1.02529437, -0.11391075,  0.03146951],
 		  	[-0.11391075,  1.03553592, -0.07852064],
 		 	[ 0.03146951, -0.07852064,  0.83874547]]

	mu2t = [8.088079704231793, 7.927924504375215, 8.000474719123183]
	cov2t = [[ 0.95559825, -0.02582016,  0.07491681],
 			[-0.02582016,  0.99427793,  0.03304442],
 			[ 0.07491681,  0.03304442,  1.15403456]]

	for init in 'first-k', 'random', 'kmeans++':
		model = GeneralMixtureModel.from_samples(MultivariateGaussianDistribution, 2, 
			X_nan, init=init, n_init=1)

		mu1 = model.distributions[0].parameters[0]
		cov1 = model.distributions[0].parameters[1]

		mu2 = model.distributions[1].parameters[0]
		cov2 = model.distributions[1].parameters[1]

		assert_array_almost_equal(mu1, mu1t)
		assert_array_almost_equal(mu2, mu2t)
		assert_array_almost_equal(cov1, cov1t)
		assert_array_almost_equal(cov2, cov2t)


def test_gmm_multivariate_gaussian_nan_fit():
	mu1, mu2, mu3 = numpy.zeros(3), numpy.zeros(3)+3, numpy.zeros(3)+5
	cov1, cov2, cov3 = numpy.eye(3), numpy.eye(3)*2, numpy.eye(3)*0.5

	d1 = MultivariateGaussianDistribution(mu1, cov1)
	d2 = MultivariateGaussianDistribution(mu2, cov2)
	d3 = MultivariateGaussianDistribution(mu3, cov3)
	model = GeneralMixtureModel([d1, d2, d3])

	numpy.random.seed(1)
	X = numpy.concatenate([numpy.random.normal(0, 1, size=(300, 3)), 
						   numpy.random.normal(2.5, 1, size=(300, 3)),
						   numpy.random.normal(6, 1, size=(300, 3))])
	numpy.random.shuffle(X)
	idxs = numpy.random.choice(numpy.arange(2700), replace=False, size=1000)
	i, j = idxs // 3, idxs % 3

	model.fit(X)

	mu1t = [-0.003165176330948316, 0.07462401273020161, 0.04001352280548061]
	cov1t = [[ 0.98556769, -0.10062447,  0.08213565],
				[-0.10062447,  1.06955989, -0.03085883],
				[ 0.08213565, -0.03085883,  0.89728992]]

	mu2t = [2.601485766170187, 2.48231424824341, 2.52771758325412]
	cov2t = [[ 0.94263451, -0.00361101, -0.02668448],
	 		[-0.00361101,  1.06339061, -0.00408865],
			 [-0.02668448, -0.00408865,  1.14789789]]

	mu3t = [5.950490843670593, 5.9572969419328725, 6.025950220056731]
	cov3t = [[ 1.03991941, -0.0232587,  -0.02457755],
				[-0.0232587,   1.01047466, -0.04948464],
				[-0.02457755, -0.04948464,  0.85671553]]

	mu1 = model.distributions[0].parameters[0]
	cov1 = model.distributions[0].parameters[1]

	mu2 = model.distributions[1].parameters[0]
	cov2 = model.distributions[1].parameters[1]

	mu3 = model.distributions[2].parameters[0]
	cov3 = model.distributions[2].parameters[1]

	assert_array_almost_equal(mu1, mu1t)
	assert_array_almost_equal(mu2, mu2t)
	assert_array_almost_equal(mu3, mu3t)
	assert_array_almost_equal(cov1, cov1t)
	assert_array_almost_equal(cov2, cov2t)
	assert_array_almost_equal(cov3, cov3t)


@with_setup(setup_multivariate_gaussian, teardown)
def test_gmm_multivariate_gaussian_nan_log_probability():
	numpy.random.seed(1)

	X = numpy.concatenate([numpy.random.normal(0, 1, size=(5, 5)), 
						   numpy.random.normal(2.5, 1, size=(5, 5)),
						   numpy.random.normal(6, 1, size=(5, 5))])
	numpy.random.shuffle(X)
	idxs = numpy.random.choice(numpy.arange(75), replace=False, size=35)
	i, j = idxs // 5, idxs % 5
	X[i, j] = numpy.nan

	logp_t = [ -7.73923053e+00,  -7.81725880e+00,  -4.55182482e+00,  -2.45359578e+01,
			   -8.01289941e+00,  -1.08630517e+01,  -3.10356303e+00,  -3.06193233e+01,
 			   -5.46424483e+00,  -1.84952128e+01,  -3.15420910e+01,  -1.01635415e+01,
  			    1.66533454e-16,  -2.70671185e+00,  -5.69860159e+00]

	logp = gmm.log_probability(X)

	assert_array_almost_equal(logp, logp_t)


def test_gmm_multivariate_gaussian_nan_predict():
	X = numpy.concatenate([numpy.random.normal(0, 1, size=(300, 5)), 
						   numpy.random.normal(8, 1, size=(300, 5))])
	numpy.random.shuffle(X)
	idxs = numpy.random.choice(numpy.arange(3000), replace=False, size=900)
	i, j = idxs // 5, idxs % 5
	
	X_nan = X.copy()
	X_nan[i, j] = numpy.nan

	for init in 'first-k', 'random', 'kmeans++':
		model = GeneralMixtureModel.from_samples(MultivariateGaussianDistribution, 2, 
			X_nan, init=init, n_init=1)

		for d in model.distributions:
			assert_equal(numpy.isnan(d.parameters[0]).sum(), 0)
			assert_equal(numpy.isnan(d.parameters[1]).sum(), 0)

		y_hat = model.predict(X)
		assert_equal(y_hat.sum(), 300)


def test_gmm_multivariate_gaussian_ooc_nan_from_samples():
	numpy.random.seed(2)
	X = numpy.concatenate([numpy.random.normal(i*3, 0.5, size=(200, 3)) for i in range(2)])
	numpy.random.shuffle(X)
	idxs = numpy.random.choice(numpy.arange(1200), replace=False, size=100)
	i, j = idxs // 3, idxs % 3
	X[i, j] = numpy.nan

	model1 = GeneralMixtureModel.from_samples(MultivariateGaussianDistribution,
		2, X, init='first-k', batch_size=None, max_iterations=3)
	model2 = GeneralMixtureModel.from_samples(MultivariateGaussianDistribution, 
		2, X, init='first-k', batch_size=400, max_iterations=3)
	model3 = GeneralMixtureModel.from_samples(MultivariateGaussianDistribution, 
		2, X, init='first-k', batch_size=100, max_iterations=3)

	cov1 = model1.distributions[0].parameters[1]
	cov2 = model2.distributions[0].parameters[1]
	cov3 = model3.distributions[0].parameters[1]

	assert_array_almost_equal(cov1, cov2)
	assert_array_almost_equal(cov1, cov3)


def test_gmm_multivariate_gaussian_ooc_nan_fit():
	X = numpy.concatenate([numpy.random.normal(i*3, 0.5, size=(100, 3)) for i in range(2)])
	numpy.random.shuffle(X)
	idxs = numpy.random.choice(numpy.arange(600), replace=False, size=100)
	i, j = idxs // 3, idxs % 3
	X[i, j] = numpy.nan

	mus = [numpy.ones(3)*i*3 for i in range(2)]
	covs = [numpy.eye(3) for i in range(2)]


	distributions = [MultivariateGaussianDistribution(mu, cov) for mu, cov in zip(mus, covs)]
	model1 = GeneralMixtureModel(distributions)
	model1.fit(X, max_iterations=3)

	distributions = [MultivariateGaussianDistribution(mu, cov) for mu, cov in zip(mus, covs)]
	model2 = GeneralMixtureModel(distributions)
	model2.fit(X, batch_size=10, max_iterations=3)

	distributions = [MultivariateGaussianDistribution(mu, cov) for mu, cov in zip(mus, covs)]
	model3 = GeneralMixtureModel(distributions)
	model3.fit(X, batch_size=1, max_iterations=3)

	cov1 = model1.distributions[0].parameters[1]
	cov2 = model2.distributions[0].parameters[1]
	cov3 = model3.distributions[0].parameters[1]

	assert_array_almost_equal(cov1, cov2)
	assert_array_almost_equal(cov1, cov3)


def test_gmm_multivariate_gaussian_minibatch_nan_from_samples():
	X = numpy.concatenate([numpy.random.normal(i*2, 1, size=(100, 3)) for i in range(2)])
	numpy.random.shuffle(X)
	idxs = numpy.random.choice(numpy.arange(600), replace=False, size=100)
	i, j = idxs // 3, idxs % 3
	X[i, j] = numpy.nan

	model1 = GeneralMixtureModel.from_samples(MultivariateGaussianDistribution,
		2, X, init='first-k', batch_size=None, max_iterations=5)
	model2 = GeneralMixtureModel.from_samples(MultivariateGaussianDistribution,
		2, X, init='first-k', batch_size=200, max_iterations=5)
	model3 = GeneralMixtureModel.from_samples(MultivariateGaussianDistribution,
		2, X, init='first-k', batch_size=50, batches_per_epoch=4,
		max_iterations=5)
	model4 = GeneralMixtureModel.from_samples(MultivariateGaussianDistribution,
		2, X, init='first-k', batch_size=50, batches_per_epoch=1,
		max_iterations=5)

	cov1 = model1.distributions[0].parameters[1]
	cov2 = model2.distributions[0].parameters[1]
	cov3 = model3.distributions[0].parameters[1]
	cov4 = model4.distributions[0].parameters[1]

	assert_array_almost_equal(cov1, cov2)
	assert_raises(AssertionError, assert_array_almost_equal, cov1, cov3)
	assert_raises(AssertionError, assert_array_almost_equal, cov1, cov4)


def test_gmm_multivariate_gaussian_minibatch_nan_fit():
	X = numpy.concatenate([numpy.random.normal(i*3, 0.5, size=(100, 3)) for i in range(2)])
	numpy.random.shuffle(X)
	idxs = numpy.random.choice(numpy.arange(600), replace=False, size=100)
	i, j = idxs // 3, idxs % 3
	X[i, j] = numpy.nan

	mus = [numpy.ones(3)*i*3 for i in range(2)]
	covs = [numpy.eye(3) for i in range(2)]


	distributions = [MultivariateGaussianDistribution(mu, cov) for mu, cov in zip(mus, covs)]
	model1 = GeneralMixtureModel(distributions)
	model1.fit(X, batch_size=None, max_iterations=3)

	distributions = [MultivariateGaussianDistribution(mu, cov) for mu, cov in zip(mus, covs)]
	model2 = GeneralMixtureModel(distributions)
	model2.fit(X, batch_size=10, max_iterations=3)

	distributions = [MultivariateGaussianDistribution(mu, cov) for mu, cov in zip(mus, covs)]
	model3 = GeneralMixtureModel(distributions)
	model3.fit(X, batch_size=10, batches_per_epoch=5, max_iterations=3)

	cov1 = model1.distributions[0].parameters[1]
	cov2 = model2.distributions[0].parameters[1]
	cov3 = model3.distributions[0].parameters[1]

	assert_array_almost_equal(cov1, cov2)
	assert_raises(AssertionError, assert_array_equal, cov1, cov3)

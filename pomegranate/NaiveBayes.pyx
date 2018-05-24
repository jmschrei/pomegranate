#cython: boundscheck=False
#cython: cdivision=True
# NaiveBayes.pyx
# Contact: Jacob Schreiber ( jmschreiber91@gmail.com )

import time
import json
import numpy
cimport numpy

from .base cimport Model
from .bayes cimport BayesModel

from distributions.distributions cimport Distribution
from distributions import DiscreteDistribution
from distributions import IndependentComponentsDistribution
from distributions import MultivariateGaussianDistribution
from distributions import DirichletDistribution

from .gmm import GeneralMixtureModel
from .utils import _convert
from .callbacks import History

from joblib import Parallel
from joblib import delayed

DEF NEGINF = float("-inf")
DEF INF = float("inf")

cdef class NaiveBayes(BayesModel):
	"""A naive Bayes model, a supervised alternative to GMM.

	A naive Bayes classifier, that treats each dimension independently from
	each other. This is a simpler version of the Bayes Classifier, that can
	use any distribution with any covariance structure, including Bayesian
	networks and hidden Markov models.

	Parameters
	----------
	models : list
		A list of initialized distributions.

	weights : list or numpy.ndarray or None, default None
		The prior probabilities of the components. If None is passed in then
		defaults to the uniformly distributed priors.

	Attributes
	----------
	models : list
		The model objects, either initialized by the user or fit to data.

	weights : numpy.ndarray
		The prior probability of each component of the model.

	Examples
	--------
	>>> from pomegranate import *
	>>> X = [0, 2, 0, 1, 0, 5, 6, 5, 7, 6]
	>>> y = [0, 0, 0, 0, 0, 1, 1, 0, 1, 1]
	>>> clf = NaiveBayes.from_samples(NormalDistribution, X, y)
	>>> clf.predict_proba([6])
	array([[0.01973451,  0.98026549]])

	>>> from pomegranate import *
	>>> clf = NaiveBayes([NormalDistribution(1, 2), NormalDistribution(0, 1)])
	>>> clf.predict_log_proba([[0], [1], [2], [-1]])
	array([[-1.1836569 , -0.36550972],
		   [-0.79437677, -0.60122959],
		   [-0.26751248, -1.4493653],
		   [-1.09861229, -0.40546511]])
	"""

	def __init__(self, distributions, weights=None):
		super(NaiveBayes, self).__init__(distributions, weights)

	def __reduce__(self):
		return self.__class__, (self.distributions, self.weights)

	def to_json(self, separators=(',', ' : '), indent=4):
		if self.d == 0:
			raise ValueError("must fit components to the data before prediction")

		nb = {
			'class' : 'NaiveBayes',
			'models' : [json.loads(model.to_json()) for model in self.distributions],
			'weights' : numpy.exp(self.weights).tolist()
		}

		return json.dumps(nb, separators=separators, indent=indent)

	@classmethod
	def from_json(cls, s):
		try:
			d = json.loads(s)
		except:
			try:
				with open(s, 'r') as f:
					d = json.load(f)
			except:
				raise IOError("String must be properly formatted JSON or filename of properly formatted JSON.")

		models = list()
		for j in d['models']:
			if j['class'] == 'Distribution':
				models.append(Distribution.from_json(json.dumps(j)))
			elif j['class'] == 'GeneralMixtureModel':
				models.append(GeneralMixtureModel.from_json(json.dumps(j)))

		nb = NaiveBayes(models, numpy.array(d['weights']))
		return nb

	@classmethod
	def from_samples(self, distributions, X, y, weights=None,
		pseudocount=0.0, stop_threshold=0.1, max_iterations=1e8,
		callbacks=[], return_history=False, verbose=False, n_jobs=1):
		"""Create a naive Bayes classifier directly from the given dataset.

		This will initialize the distributions using maximum likelihood estimates
		derived by partitioning the dataset using the label vector. If any labels
		are missing, the model will be trained using EM in a semi-supervised
		setting.

		A homogeneous model can be defined by passing in a single distribution
		callable as the first parameter and specifying the number of components,
		while a heterogeneous model can be defined by passing in a list of
		callables of the appropriate type.

		A naive Bayes classifier is a subrset of the Bayes classifier in that
		the math is identical, but the distributions are independent for each
		feature. Simply put, one can create a multivariate Gaussian Bayes
		classifier with a full covariance matrix, but a Gaussian naive Bayes
		would require a diagonal covariance matrix.

		Parameters
		----------
		distributions : array-like, shape (n_components,) or callable
			The components of the model. This should either be a single callable
			if all components will be the same distribution, or an array of
			callables, one for each feature.

		X : array-like, shape (n_samples, n_dimensions)
			This is the data to train on. Each row is a sample, and each column
			is a dimension to train on.

		y : array-like, shape (n_samples,)
			The labels for each sample. The labels should be integers between
			0 and k-1 for a problem with k classes, or -1 if the label is not
			known for that sample.

		weights : array-like, shape (n_samples,), optional
			The initial weights of each sample in the matrix. If nothing is
			passed in then each sample is assumed to be the same weight.
			Default is None.

		pseudocount : double, optional, positive
			A pseudocount to add to the emission of each distribution. This
			effectively smoothes the states to prevent 0. probability symbols
			if they don't happen to occur in the data. Only effects mixture
			models defined over discrete distributions. Default is 0.

		stop_threshold : double, optional, positive
			The threshold at which EM will terminate for the improvement of
			the model. If the model does not improve its fit of the data by
			a log probability of 0.1 then terminate. Only required if doing
			semisupervised learning. Default is 0.1.

		max_iterations : int, optional, positive
			The maximum number of iterations to run EM for. If this limit is
			hit then it will terminate training, regardless of how well the
			model is improving per iteration. Only required if doing
			semisupervised learning. Default is 1e8.

        callbacks : list, optional
            A list of callback objects that describe functionality that should
            be undertaken over the course of training.

        return_history : bool, optional
            Whether to return the history during training as well as the model.

		verbose : bool, optional
			Whether or not to print out improvement information over
			iterations. Only required if doing semisupervised learning.
			Default is False.

		n_jobs : int
			The number of jobs to use to parallelize, either the number of threads
			or the number of processes to use. Default is 1.

		Returns
		-------
		model : NaiveBayes
			The fit naive Bayes model.
		"""

		ICD = IndependentComponentsDistribution
		if distributions in (MultivariateGaussianDistribution, DirichletDistribution):
			raise ValueError("naive Bayes only supports independent features. Use BayesClassifier instead")
		elif isinstance(distributions, (list, numpy.ndarray, tuple)):
			for distribution in distributions:
				if not callable(distribution):
					raise ValueError("must pass in class constructors, not initiated distributions (i.e. NormalDistribution)")
				elif distribution in (MultivariateGaussianDistribution, DirichletDistribution):
					raise ValueError("naive Bayes only supported independent features. Use BayesClassifier instead")


		X = numpy.array(X)
		y = numpy.array(y)

		n, d = X.shape
		n_components = numpy.unique(y[y != -1]).shape[0]
		if callable(distributions):
			if d > 1:
				distributions = [ICD([distributions.blank() for j in range(d)]) for i in range(n_components)]
			else:
				distributions = [distributions.blank() for i in range(n_components)]
		else:
			if d > 1:
				distributions = [ICD([distribution.blank() for distribution in distributions]) for i in range(n_components)]
			else:
				distributions = [distribution.blank() for distribution in distributions]

		model = NaiveBayes(distributions)
		_, history = model.fit(X, y, weights=weights, pseudocount=pseudocount,
			stop_threshold=stop_threshold, max_iterations=max_iterations,
			verbose=verbose, callbacks=callbacks, return_history=True, n_jobs=n_jobs)

		if return_history:
			return model, history
		return model

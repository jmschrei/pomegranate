#cython: boundscheck=False
#cython: cdivision=True
# BayesClassifier.pyx
# Contact: Jacob Schreiber ( jmschreiber91@gmail.com )

import time
import json
import numpy
cimport numpy

from .bayes cimport BayesModel
from distributions import Distribution
from .gmm import GeneralMixtureModel
from .hmm import HiddenMarkovModel
from .BayesianNetwork import BayesianNetwork

from .utils import _convert

from joblib import Parallel
from joblib import delayed

DEF NEGINF = float("-inf")
DEF INF = float("inf")

cdef class BayesClassifier(BayesModel):
	"""A Bayes classifier, a more general form of a naive Bayes classifier.

	A Bayes classifier, like a naive Bayes classifier, uses Bayes' rule in
	order to calculate the posterior probability of the classes, which are
	used for the predictions. However, a naive Bayes classifier assumes that
	each of the features are independent of each other and so can be modelled
	as independent distributions. A generalization of that, the Bayes
	classifier, allows for an arbitrary covariance between the features. This
	allows for more complicated components to be used, up to and including
	even HMMs to form a classifier over sequences, or mixtures to form a
	classifier with complex emissions.

	Parameters
	----------
	models : list
		A list of initialized distribution objects to use as the components
		in the model.

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
	>>>
	>>> d1 = NormalDistribution(3, 2)
	>>> d2 = NormalDistribution(5, 1.5)
	>>>
	>>> clf = BayesClassifier([d1, d2])
	>>> clf.predict_proba([[6]])
	array([[ 0.2331767,  0.7668233]])
	>>> X = [[0], [2], [0], [1], [0], [5], [6], [5], [7], [6]]
	>>> y = [0, 0, 0, 0, 0, 1, 1, 0, 1, 1]
	>>> clf.fit(X, y)
	>>> clf.predict_proba([[6]])
	array([[ 0.01973451,  0.98026549]])
	"""

	def __init__(self, distributions, weights=None):
		super(BayesClassifier, self).__init__(distributions, weights)

	def __reduce__(self):
		return self.__class__, (self.distributions, self.weights)

	def to_json( self, separators=(',', ' : '), indent=4 ):
		if self.d == 0:
			raise ValueError("must fit components to the data before prediction")

		model = {
			'class' : 'BayesClassifier',
			'models' : [ json.loads( model.to_json() ) for model in self.distributions ],
			'weights' : self.weights.tolist()
		}

		return json.dumps(model, separators=separators, indent=indent)

	@classmethod
	def from_json( cls, s ):
		try:
			d = json.loads( s )
		except:
			try:
				with open( s, 'r' ) as f:
					d = json.load( f )
			except:
				raise IOError("String must be properly formatted JSON or filename of properly formatted JSON.")

		models = list()
		for j in d['models']:
			if j['class'] == 'Distribution':
				models.append( Distribution.from_json( json.dumps(j) ) )
			elif j['class'] == 'GeneralMixtureModel':
				models.append( GeneralMixtureModel.from_json( json.dumps(j) ) )
			elif j['class'] == 'HiddenMarkovModel':
				models.append( HiddenMarkovModel.from_json( json.dumps(j) ) )
			elif j['class'] == 'BayesianNetwork':
				models.append( BayesianNetwork.from_json( json.dumps(j) ) )

		nb = BayesClassifier( models, numpy.array( d['weights'] ) )
		return nb

	@classmethod
	def from_samples(self, distributions, X, y, weights=None,
		inertia=0.0, pseudocount=0.0, stop_threshold=0.1, max_iterations=1e8,
		callbacks=[], return_history=False, verbose=False, n_jobs=1):
		"""Create a Bayes classifier directly from the given dataset.

		This will initialize the distributions using maximum likelihood estimates
		derived by partitioning the dataset using the label vector. If any labels
		are missing, the model will be trained using EM in a semi-supervised
		setting.

		A homogeneous model can be defined by passing in a single distribution
		callable as the first parameter and specifying the number of components,
		while a heterogeneous model can be defined by passing in a list of
		callables of the appropriate type.

		A Bayes classifier is a superset of the naive Bayes classifier in that
		the math is identical, but the distributions used do not have to be
		independent for each feature. Simply put, one can create a multivariate
		Gaussian Bayes classifier with a full covariance matrix, but a Gaussian
		naive Bayes would require a diagonal covariance matrix.

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

		inertia : double, optional
			Inertia used for the training the distributions.

		pseudocount : double, optional
			A pseudocount to add to the emission of each distribution. This
			effectively smoothes the states to prevent 0. probability symbols
			if they don't happen to occur in the data. Default is 0.

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
			or the number of processes to use. -1 means use all available resources.
			Default is 1.

		Returns
		-------
		model : BayesClassifier
			The fit Bayes classifier model.
		"""

		if isinstance(distributions, (list, numpy.ndarray, tuple)):
			for distribution in distributions:
				if not callable(distribution):
					raise ValueError("must pass in class constructors, not initiated distributions (i.e. NormalDistribution)")

		X = numpy.array(X)
		y = numpy.array(y)

		n, d = X.shape
		n_components = numpy.unique(y[y != -1]).shape[0]
		if callable(distributions):
			if d > 1:
				distributions = [distributions.blank(d) for i in range(n_components)]
			else:
				distributions = [distribution.blank() for i in range(n_components)]
		else:
			distributions = [distribution.blank() for distribution in distributions]

		model = BayesClassifier(distributions)
		_, history = model.fit(X, y, weights=weights, inertia=inertia, pseudocount=pseudocount,
			stop_threshold=stop_threshold, max_iterations=max_iterations,
			callbacks=callbacks, return_history=True, verbose=verbose, n_jobs=n_jobs)

		if return_history:
			return model, history
		return model

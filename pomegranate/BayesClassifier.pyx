#cython: boundscheck=False
#cython: cdivision=True
# BayesClassifier.pyx
# Contact: Jacob Schreiber ( jmschreiber91@gmail.com )

import time
import json
import numpy
cimport numpy

from .bayes cimport BayesModel
from .distributions import Distribution
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

	def fit(self, X, y, weights=None, inertia=0.0, pseudocount=0.0,
		stop_threshold=0.1, max_iterations=1e8, verbose=False, n_jobs=1):
		"""Fit the Bayes classifier to the data by passing data to its components.

		The fit step for a Bayes classifier with purely labeled data is a simple
		MLE update on the underlying distributions, grouped by the labels. However,
		in the semi-supervised the model is trained on a mixture of both labeled
		and unlabeled data, where the unlabeled data uses the label -1. In this
		setting, EM is used to train the model. The model is initialized using the
		labeled data and then sufficient statistics are gathered for both the
		labeled and unlabeled data, combined, and used to update the parameters.

		Parameters
		----------
		X : numpy.ndarray or list
			The dataset to operate on. For most models this is a numpy array with
			columns corresponding to features and rows corresponding to samples.
			For markov chains and HMMs this will be a list of variable length
			sequences.

		y : numpy.ndarray or list or None, optional
			Data labels for supervised training algorithms. Default is None

		weights : array-like or None, shape (n_samples,), optional
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

		verbose : bool, optional
			Whether or not to print out improvement information over
			iterations. Only required if doing semisupervised learning.
			Default is False.

		n_jobs : int
			The number of jobs to use to parallelize, either the number of threads
			or the number of processes to use. Default is 1.

		Returns
		-------
		self : object
			Returns the fitted model
		"""

		training_start_time = time.time()

		X = numpy.array(X, dtype='float64')
		n, d = X.shape

		if weights is None:
			weights = numpy.ones(n, dtype='float64')
		else:
			weights = numpy.array(weights, dtype='float64')

		starts = [int(i*len(X)/n_jobs) for i in range(n_jobs)]
		ends = [int(i*len(X)/n_jobs) for i in range(1, n_jobs+1)]

		with Parallel(n_jobs=n_jobs, backend='threading') as parallel:
			parallel( delayed(self.summarize, check_pickle=False)(X[start:end], 
				y[start:end], weights[start:end]) for start, end in zip(starts, ends) )

			self.from_summaries(inertia, pseudocount)

			semisupervised = -1 in y
			if semisupervised:
				initial_log_probability_sum = NEGINF
				iteration, improvement = 0, INF
				n_classes = numpy.unique(y).shape[0]

				unsupervised = GeneralMixtureModel(self.distributions)

				X_labeled = X[y != -1]
				y_labeled = y[y != -1]
				weights_labeled = None if weights is None else weights[y != -1]

				X_unlabeled = X[y == -1]
				weights_unlabeled = None if weights is None else weights[y == -1]

				labeled_starts = [int(i*len(X_labeled)/n_jobs) for i in range(n_jobs)]
				labeled_ends = [int(i*len(X_labeled)/n_jobs) for i in range(1, n_jobs+1)]

				unlabeled_starts = [int(i*len(X_unlabeled)/n_jobs) for i in range(n_jobs)]
				unlabeled_ends = [int(i*len(X_unlabeled)/n_jobs) for i in range(1, n_jobs+1)]

				while improvement > stop_threshold and iteration < max_iterations + 1:
					epoch_start_time = time.time()
					self.from_summaries(inertia, pseudocount)
					unsupervised.weights[:] = self.weights

					parallel( delayed(self.summarize, 
						check_pickle=False)(X_labeled[start:end], 
						y_labeled[start:end], weights_labeled[start:end]) 
						for start, end in zip(labeled_starts, labeled_ends))

					unsupervised.summaries[:] = self.summaries

					log_probability_sum = sum(parallel( delayed(unsupervised.summarize, 
						check_pickle=False)(X_unlabeled[start:end], weights_unlabeled[start:end]) 
						for start, end in zip(unlabeled_starts, unlabeled_ends)))

					self.summaries[:] = unsupervised.summaries

					if iteration == 0:
						initial_log_probability_sum = log_probability_sum
					else:
						improvement = log_probability_sum - last_log_probability_sum

						time_spent = time.time() - epoch_start_time
						if verbose:
							print("[{}] Improvement: {}\tTime (s): {:4.4}".format(iteration,
								improvement, time_spent))

					iteration += 1
					last_log_probability_sum = log_probability_sum

				self.clear_summaries()

				if verbose:
					total_imp = last_log_probability_sum - initial_log_probability_sum
					print("Total Improvement: {}".format(total_imp))

		if verbose:
			total_time_spent = time.time() - training_start_time
			print("Total Time (s): {:.4f}".format(total_time_spent))

		return self

	def summarize( self, X, y, weights=None, n_jobs=1 ):
		"""Summarize data into stored sufficient statistics for out-of-core training.

		Parameters
		----------
		X : array-like, shape (n_samples, variable)
			Array of the samples, which can be either fixed size or variable depending
			on the underlying components.

		y : array-like, shape (n_samples,)
			Array of the known labels as integers

		weights : array-like, shape (n_samples,) optional
			Array of the weight of each sample, a positive float

		n_jobs : int
			The number of jobs to use to parallelize, either the number of threads
			or the number of processes to use. Default is 1.

		Returns
		-------
		None
		"""

		X = _convert(X)
		y = _convert(y)

		if self.d > 0 and not isinstance( self.distributions[0], HiddenMarkovModel ):
			if X.ndim > 2:
				raise ValueError("input data has too many dimensions")
			elif X.ndim == 2 and self.d != X.shape[1]:
				raise ValueError("input data rows do not match model dimension")

		if weights is None:
			weights = numpy.ones(X.shape[0], dtype='float64')
		else:
			weights = numpy.array(weights, dtype='float64')

		if self.is_vl_:
			for i in range(self.n):
				self.distributions[i].fit( list(X[y==i]), weights=weights[y==i], n_jobs=n_jobs )
		else:
			delay = delayed(lambda model, x, weights: model.summarize(x, weights), check_pickle=False)
			with Parallel(n_jobs=n_jobs, backend='threading') as parallel:
				parallel( delay(self.distributions[i], X[y==i], weights[y==i]) for i in range(self.n) )

		for i in range(self.n):
			self.summaries[i] += weights[y == i].sum()

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
		verbose=False, n_jobs=1):
		"""Create a mixture model directly from the given dataset.

		First, k-means will be run using the given initializations, in order to
		define initial clusters for the points. These clusters are used to
		initialize the distributions used. Then, EM is run to refine the
		parameters of these distributions.

		A homogenous mixture can be defined by passing in a single distribution
		callable as the first parameter and specifying the number of components,
		while a heterogeneous mixture can be defined by passing in a list of
		callables of the appropriate type.

		Parameters
		----------
		distributions : array-like, shape (n_components,) or callable
			The components of the model. If array, corresponds to the initial
			distributions of the components. If callable, must also pass in the
			number of components and kmeans++ will be used to initialize them.

		n_components : int
			If a callable is passed into distributions then this is the number
			of components to initialize using the kmeans++ algorithm.

		X : array-like, shape (n_samples, n_dimensions)
			This is the data to train on. Each row is a sample, and each column
			is a dimension to train on.

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

		if isinstance(distributions, (list, numpy.ndarray, tuple)):
			for distribution in distributions:
				if not callable(distribution):
					raise ValueError("must pass in class constructors, not initiated distributions (i.e. NormalDistribution)")

		X = numpy.array(X)
		y = numpy.array(y)

		n, d = X.shape
		n_components = numpy.unique(y).shape[0]
		if callable(distributions):
			if d > 1:
				distributions = [distributions.blank(d) for i in range(n_components)]
			else:
				distributions = [distribution.blank() for i in range(n_components)]
		else:
			distributions = [distribution.blank() for distribution in distributions]

		model = BayesClassifier(distributions)
		model.fit(X, y, weights=weights, inertia=inertia, pseudocount=pseudocount, 
			stop_threshold=stop_threshold, max_iterations=max_iterations,
			verbose=verbose, n_jobs=n_jobs)

		return model

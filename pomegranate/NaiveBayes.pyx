# NaiveBayes.pyx
# Authors: Jacob Schreiber <jmschreiber91@gmail.com>

"""
Naive Bayes estimator, for anything with a log_probability method.
"""

import numpy
cimport numpy

from libc.math cimport exp as cexp

from .distributions cimport Distribution
from .hmm import HiddenMarkovModel
from .utils cimport pair_lse

cdef double NEGINF = float("-inf")

cdef class NaiveBayes( object ):
    """A Naive Bayes model, a supervised alternative to GMM.

    Parameters
    ----------
    models : list or constructor
        Must either be a list of initialized distribution/model objects, or
        the constructor for a distribution object:

        * Initialized : NaiveBayes([NormalDistribution(1, 2), NormalDistribution(0, 1)])
        * Constructor : NaiveBayes(NormalDistribution)

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
    >>> clf = NaiveBayes( NormalDistribution )
    >>> X = [0, 2, 0, 1, 0, 5, 6, 5, 7, 6]
    >>> y = [0, 0, 0, 0, 0, 1, 1, 0, 1, 1]
    >>> clf.fit(X, y)
    >>> clf.predict_proba([6])
    array([[ 0.01973451,  0.98026549]])

    >>> from pomegranate import *
    >>> clf = NaiveBayes([NormalDistribution(1, 2), NormalDistribution(0, 1)])
    >>> clf.predict_log_proba([[0], [1], [2], [-1]])
    array([[-1.1836569 , -0.36550972],
           [-0.79437677, -0.60122959],
           [-0.26751248, -1.4493653 ],
           [-1.09861229, -0.40546511]])
    """

    cdef int initialized
    cdef public object models
    cdef void** models_ptr
    cdef numpy.ndarray summaries
    cdef public numpy.ndarray weights
    cdef double* weights_ptr

    def __init__( self, models=None, weights=None ):
        if not callable(models) and not isinstance(models, list): 
            raise ValueError("must either give initial models or constructor")

        self.summaries = None
        self.initialized = False

        if not callable(models):
            self.initialized = True
            self.summaries = numpy.zeros(len(models))

            self.models = numpy.array( models )
            self.models_ptr = <void**> (<numpy.ndarray> self.models).data

            if weights is None:
                self.weights = numpy.ones(len(models), dtype='float64') / len(models)
            else:
                self.weights = numpy.array(weights) / numpy.sum(weights)

            self.weights_ptr = <double*> (<numpy.ndarray> self.weights).data

        self.models = models

    def fit( self, X, y, weights=None, inertia=0.0 ):
        """Fit the Naive Bayes model to the data by passing data to their components.

        Parameters
        ----------
        X : array-like, shape (n_samples, variable)
            Array of the samples, which can be either fixed size or variable depending
            on the underlying components.

        y : array-like, shape (n_samples,)
            Array of the known labels as integers

        weights : array-like, shape (n_samples,) optional
            Array of the weight of each sample, a positive float

        inertia : double, optional
            Inertia used for the training the distributions.

        Returns
        -------
        self : object
            Returns the fitted model 
        """

        self.summarize( X, y, weights )
        self.from_summaries( inertia )
        return self


    def summarize( self, X, y, weights=None ):
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

        Returns
        -------
        None
        """

        X = numpy.array(X)
        y = numpy.array(y)

        if weights is None:
            weights = numpy.ones(X.shape[0]) / X.shape[0]
        else:
            weights = numpy.array(weights) / numpy.sum(weights)

        if len(X.shape) == 1 and not isinstance(X[0], list):
            X = X.reshape( X.shape[0], 1 )

        n = numpy.unique(y).shape[0]

        if not self.initialized:
            self.models = [self.models] * n
            self.weights = numpy.ones(n, dtype=numpy.float64) / n
            self.weights_ptr = <double*> (<numpy.ndarray> self.weights).data
            self.summaries = numpy.zeros(n)
            self.initialized = True
        elif n != len(self.models):
            self.models = [self.models[0].__class__] * n
            self.weights = numpy.ones(n, dtype=numpy.float64) / n
            self.weights_ptr = <double*> (<numpy.ndarray> self.weights).data
            self.summaries = numpy.zeros(n)
            self.initialized = True

        n = len(self.models)

        for i in range(n):
            if callable(self.models[i]):
                self.models[i] = self.models[i].from_samples(X[0:2])

            if isinstance( self.models[i], HiddenMarkovModel ):
                self.models[i].fit( list(X[y==i]) )
            else:
                self.models[i].summarize( X[y==i], weights[y==i] )

            self.summaries[i] += weights[y==i].sum()

    def from_summaries( self, inertia=0.0 ):
        """Fit the Naive Bayes model to the stored sufficient statistics.

        Parameters
        ----------
        inertia : double, optional
            Inertia used for the training the distributions.

        Returns
        -------
        self : object
            Returns the fitted model 
        """

        n = len(self.models)
        self.summaries /= self.summaries.sum()

        self.models = numpy.array( self.models )
        self.models_ptr = <void**> (<numpy.ndarray> self.models).data

        for i in range(n):
            if not isinstance( self.models[i], HiddenMarkovModel ):
                self.models[i].from_summaries(inertia=inertia)

            self.weights[i] = self.summaries[i]

        self.summaries = numpy.zeros(n)
        return self
    
    cpdef predict_log_proba( self, X ):
        """Return the normalized log probability of samples under the model.

        Parameters
        ----------
        X : array-like, shape (n_samples, variable)
            Array of the samples, which can be either fixed size or variable depending
            on the underlying components.

        Returns
        -------
        r : array-like, shape (n_samples, n_components)
            Returns the normalized log probabilities of the sample under the
            components of the model. The normalized log probability is the
            log of the posterior probability P(M|D) from Bayes rule.
        """

        X = numpy.array(X)

        if self.initialized == 0:
            raise ValueError("must fit components to the data before prediction,")

        n, m = X.shape[0], len(self.models)
        r = numpy.zeros( (n, m), dtype=numpy.float64 )
        logw = numpy.log( self.weights )

        for i in range(n):
            total = NEGINF
            
            for j in range(m):
                r[i, j] = self.models[j].log_probability( X[i] ) + logw[j]
                total = pair_lse( total, r[i, j] )

            for j in range(m):
                r[i, j] -= total

        return r

    cpdef predict_proba( self, X ):
        """Return the normalized probability of samples under the model.

        Parameters
        ----------
        X : array-like, shape (n_samples, variable)
            Array of the samples, which can be either fixed size or variable depending
            on the underlying components.

        Returns
        -------
        r : array-like, shape (n_samples, n_components)
            Returns the normalized probabilities of the sample under the
            components of the model. The normalized log probability is the
            posterior probability P(M|D) from Bayes rule.
        """

        X = numpy.array(X)

        if self.initialized == 0:
            raise ValueError("must fit components to the data before prediction,")

        n, m = X.shape[0], len(self.models)
        r = numpy.zeros( (n, m), dtype=numpy.float64 )

        for i in range(n):
            total = 0.
            
            for j in range(m):
                r[i, j] = cexp(self.models[j].log_probability( X[i] )) * self.weights[j]
                total += r[i, j]

            for j in range(m):
                r[i, j] /= total

        return r

    def predict( self, X ):
        """Return the most likely component corresponding to each sample.

        Parameters
        ----------
        X : array-like, shape (n_samples, variable)
            Array of the samples, which can be either fixed size or variable depending
            on the underlying components.

        Returns
        -------
        r : array-like, shape (n_samples, n_components)
            Returns the normalized probabilities of the sample under the
            components of the model. The normalized log probability is the
            posterior probability P(M|D) from Bayes rule.
        """

        if self.initialized == 0:
            raise ValueError("must fit components to the data before prediction,")

        return self._predict( numpy.array(X) )

    cdef numpy.ndarray _predict( self, numpy.ndarray X ):
        cdef int i, j, m = len(self.models), n = X.shape[0]
        cdef numpy.ndarray y = numpy.zeros(n, dtype='int32')
        cdef int* y_ptr = <int*> y.data
        cdef double logp, max_logp

        for i in range(n):
            max_logp = NEGINF

            for j in range(m):
                logp = self.models[j].log_probability(X[i]) + self.weights_ptr[j]

                if logp > max_logp:
                    max_logp = logp
                    y_ptr[i] = j

        return y

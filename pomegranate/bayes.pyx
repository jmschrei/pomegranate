#cython: boundscheck=False
#cython: cdivision=True
# bayes.pyx
# Contact: Jacob Schreiber ( jmschreiber91@gmail.com )

from libc.stdlib cimport calloc
from libc.stdlib cimport free
from libc.string cimport memset
from libc.math cimport exp as cexp

import json
import time

import numpy
cimport numpy

from .base cimport Model
from distributions.distributions cimport Distribution
from distributions import DiscreteDistribution
from distributions import IndependentComponentsDistribution
from .hmm import HiddenMarkovModel
from .gmm import GeneralMixtureModel
from .callbacks import History

from .utils cimport _log
from .utils cimport pair_lse
from .utils cimport python_log_probability
from .utils import _check_input
from .utils import _convert
from .utils import check_random_state

from joblib import Parallel
from joblib import delayed

DEF INF = float("inf")
DEF NEGINF = float("-inf")

cdef class BayesModel(Model):
    """A simple implementation of Bayes Rule as a base model.

    Bayes rule is foundational to many models. Here, it is used as a base
    class for naive Bayes, Bayes classifiers, and mixture models, that are
    all fundamentally similar and rely on bayes rule.

    Parameters
    ----------
    distributions : array-like, shape (n_components,)
        The initialized components of the model.
    weights : array-like, optional, shape (n_components,)
        The prior probabilities corresponding to each component. Does not
        need to sum to one, but will be normalized to sum to one internally.
        Defaults to None.

    Attributes
    ----------
    distributions : array-like, shape (n_components,)
        The component distribution objects.

    weights : array-like, shape (n_components,)
        The learned prior weight of each object

    d : int
        The number of dimensionals the model is built to consider.

    is_vl_ : bool
        Whether this model is built for variable length sequences or not.
    """

    def __init__(self, distributions, weights=None):
        self.d = 0
        self.is_vl_ = 0
        self.cython = 1

        self.n = len(distributions)
        if len(distributions) < 2:
            raise ValueError("must pass in at least two distributions")

        self.d = distributions[0].d
        for dist in distributions:
            if callable(dist):
                raise TypeError("must have initialized distributions in list")
            elif self.d != dist.d:
                raise TypeError("mis-matching dimensions between distributions in list")

            if not isinstance(dist, Distribution) and not isinstance(dist, Model):
                self.cython = 0
            elif dist.model == 'HiddenMarkovModel':
                self.is_vl_ = 1
                self.keymap = dist.keymap

        if weights is None:
            weights = numpy.ones_like(distributions, dtype='float64') / self.n
        else:
            weights = numpy.array(weights, dtype='float64') / sum(weights)

        self.weights = numpy.log(weights)
        self.weights_ptr = <double*> self.weights.data

        self.distributions = numpy.array(distributions)
        self.distributions_ptr = <void**> self.distributions.data

        self.summaries = numpy.zeros_like(weights, dtype='float64')
        self.summaries_ptr = <double*> self.summaries.data

        dist = distributions[0]
        if self.is_vl_ == 1:
            pass

        elif isinstance(dist, DiscreteDistribution):
            keys = []
            for distribution in distributions:
                keys.extend(distribution.keys())
            self.keymap = [{key: i for i, key in enumerate(keys)}]
            for distribution in distributions:
                distribution.bake(tuple(keys))

        elif isinstance(dist, IndependentComponentsDistribution):
            self.keymap = [{} for i in range(self.d)]
            keymap_tuples = [tuple() for i in range(self.d)]

            for distribution in distributions:
                for i in range(self.d):
                    if isinstance(distribution[i], DiscreteDistribution):
                        for key in distribution[i].keys():
                            if key not in self.keymap[i]:
                                self.keymap[i][key] = len(self.keymap[i])
                                keymap_tuples[i] += (key,)

            for distribution in distributions:
                for i in range(self.d):
                    d = distribution[i]
                    if isinstance(d, DiscreteDistribution):
                        d.bake(keymap_tuples[i])

        

    def __reduce__(self):
        return self.__class__, (self.distributions.tolist(),
                                numpy.exp(self.weights),
                                self.n)

    def sample(self, n=1, random_state=None):
        """Generate a sample from the model.

        First, randomly select a component weighted by the prior probability,
        Then, use the sample method from that component to generate a sample.

        Parameters
        ----------
        n : int, optional
            The number of samples to generate. Defaults to 1.

        random_state : int, numpy.random.RandomState, or None
            The random state used for generating samples. If set to none, a
            random seed will be used. If set to either an integer or a
            random seed, will produce deterministic outputs.

        Returns
        -------
        sample : array-like or object
            A randomly generated sample from the model of the type modelled
            by the emissions. An integer if using most distributions, or an
            array if using multivariate ones, or a string for most discrete
            distributions. If n=1 return an object, if n>1 return an array
            of the samples.
        """

        random_state = check_random_state(random_state)

        samples = []
        random_seeds = random_state.randint(1000000, size=n)

        for i in range(n):
            d = random_state.choice(self.distributions,
                                    p=numpy.exp(self.weights))
            samples.append(d.sample(random_state=random_seeds[i]))

        return numpy.array(samples) if n > 1 else samples[0]

    def log_probability(self, X, n_jobs=1):
        """Calculate the log probability of a point under the distribution.

        The probability of a point is the sum of the probabilities of each
        distribution multiplied by the weights. Thus, the log probability is
        the sum of the log probability plus the log prior.

        This is the python interface.

        Parameters
        ----------
        X : numpy.ndarray, shape=(n, d) or (n, m, d)
            The samples to calculate the log probability of. Each row is a
            sample and each column is a dimension. If emissions are HMMs then
            shape is (n, m, d) where m is variable length for each observation,
            and X becomes an array of n (m, d)-shaped arrays.

        n_jobs : int
            The number of jobs to use to parallelize, either the number of threads
            or the number of processes to use. -1 means use all available resources.
            Default is 1.

        Returns
        -------
        log_probability : double
            The log probability of the point under the distribution.
        """

        cdef int i, j, n, d, m

        if n_jobs > 1:
            starts = [int(i*len(X)/n_jobs) for i in range(n_jobs)]
            ends = [int(i*len(X)/n_jobs) for i in range(1, n_jobs+1)]

            with Parallel(n_jobs=n_jobs, backend='threading') as parallel:
                logp_arrays = parallel(delayed(self.log_probability, check_pickle=False)(
                    X[start:end]) for start, end in zip(starts, ends))

            return numpy.concatenate(logp_arrays)

        if self.is_vl_ or self.d == 1:
            n, d = len(X), self.d
        elif self.d > 1 and X.ndim == 1:
            n, d = 1, len(X)
        else:
            n, d = X.shape

        cdef numpy.ndarray logp_ndarray = numpy.zeros(n)
        cdef double* logp = <double*> logp_ndarray.data

        cdef numpy.ndarray X_ndarray
        cdef double* X_ptr

        if not self.is_vl_:
            X_ndarray = _check_input(X, self.keymap)
            X_ptr = <double*> X_ndarray.data
            if d != self.d:
                raise ValueError("sample has {} dimensions but model has {} dimensions".format(d, self.d))

        with nogil:
            if self.is_vl_:
                for i in range(n):
                    with gil:
                        X_ndarray = numpy.array(X[i])
                        X_ptr = <double*> X_ndarray.data
                    logp[i] = self._vl_log_probability(X_ptr, n)
            else:
                self._log_probability(X_ptr, logp, n)

        return logp_ndarray

    cdef void _log_probability(self, double* X, double* log_probability, int n) nogil:
        cdef int i, j, d = self.d
        cdef double* logp = <double*> calloc(n, sizeof(double))

        if self.cython == 1:
            (<Model> self.distributions_ptr[0])._log_probability(X, log_probability, n)
        else:
            with gil:
                python_log_probability(self.distributions[0], X, log_probability, n)

        for i in range(n):
            log_probability[i] += self.weights_ptr[0]

        for j in range(1, self.n):
            if self.cython == 1:
                (<Model> self.distributions_ptr[j])._log_probability(X, logp, n)
            else:
                with gil:
                    python_log_probability(self.distributions[j], X, logp, n)

            for i in range(n):
                log_probability[i] = pair_lse(log_probability[i], logp[i] + self.weights_ptr[j])

        free(logp)

    cdef double _vl_log_probability(self, double* X, int n) nogil:
        cdef int i
        cdef double log_probability_sum = NEGINF
        cdef double log_probability

        for i in range(self.n):
            log_probability = (<Model> self.distributions_ptr[i])._vl_log_probability(X, n) + self.weights_ptr[i]
            log_probability_sum = pair_lse(log_probability_sum, log_probability)

        return log_probability_sum

    def predict_proba(self, X, n_jobs=1):
        """Calculate the posterior P(M|D) for data.

        Calculate the probability of each item having been generated from
        each component in the model. This returns normalized probabilities
        such that each row should sum to 1.

        Since calculating the log probability is much faster, this is just
        a wrapper which exponentiates the log probability matrix.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_dimensions)
            The samples to do the prediction on. Each sample is a row and each
            column corresponds to a dimension in that sample. For univariate
            distributions, a single array may be passed in.

        n_jobs : int
            The number of jobs to use to parallelize, either the number of threads
            or the number of processes to use. -1 means use all available resources.
            Default is 1.

        Returns
        -------
        probability : array-like, shape (n_samples, n_components)
            The normalized probability P(M|D) for each sample. This is the
            probability that the sample was generated from each component.
        """

        return numpy.exp(self.predict_log_proba(X, n_jobs=n_jobs))

    def predict_log_proba(self, X, n_jobs=1):
        """Calculate the posterior log P(M|D) for data.

        Calculate the log probability of each item having been generated from
        each component in the model. This returns normalized log probabilities
        such that the probabilities should sum to 1

        This is a sklearn wrapper for the original posterior function.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_dimensions)
            The samples to do the prediction on. Each sample is a row and each
            column corresponds to a dimension in that sample. For univariate
            distributions, a single array may be passed in.

        n_jobs : int
            The number of jobs to use to parallelize, either the number of threads
            or the number of processes to use. -1 means use all available resources.
            Default is 1.

        Returns
        -------
        y : array-like, shape (n_samples, n_components)
            The normalized log probability log P(M|D) for each sample. This is
            the probability that the sample was generated from each component.
        """

        cdef int i, n, d
        cdef numpy.ndarray X_ndarray
        cdef double* X_ptr

        cdef numpy.ndarray y
        cdef double* y_ptr

        if n_jobs > 1:
            starts = [int(i*len(X)/n_jobs) for i in range(n_jobs)]
            ends = [int(i*len(X)/n_jobs) for i in range(1, n_jobs+1)]

            with Parallel(n_jobs=n_jobs, backend='threading') as parallel:
                y_arrays = parallel(delayed(self.predict_log_proba, check_pickle=False)(
                    X[start:end]) for start, end in zip(starts, ends))

            return numpy.concatenate(y_arrays)

        if not self.is_vl_:
            X_ndarray = _check_input(X, self.keymap)
            X_ptr = <double*> X_ndarray.data
            n, d = X_ndarray.shape[0], X_ndarray.shape[1]
            if d != self.d:
                raise ValueError("sample only has {} dimensions but should have {} dimensions".format(d, self.d))
        else:
            X_ndarray = X
            n, d = len(X_ndarray), self.d

        y = numpy.zeros((n, self.n), dtype='float64')
        y_ptr = <double*> y.data

        with nogil:
            if not self.is_vl_:
                self._predict_log_proba(X_ptr, y_ptr, n, d)
            else:
                for i in range(n):
                    with gil:
                        X_ndarray = _check_input(X[i], self.keymap)
                        X_ptr = <double*> X_ndarray.data
                        d = len(X_ndarray)

                    self._predict_log_proba(X_ptr, y_ptr+i*self.n, 1, d)

        return y if self.is_vl_ else y.reshape(self.n, n).T

    cdef void _predict_log_proba(self, double* X, double* y, int n, int d) nogil:
        cdef double y_sum, logp
        cdef int i, j

        for j in range(self.n):
            if self.is_vl_:
                y[j] = (<Model> self.distributions_ptr[j])._vl_log_probability(X, d)
            else:
                if self.cython == 1:
                    (<Model> self.distributions_ptr[j])._log_probability(X, y+j*n, n)
                else:
                    with gil:
                        python_log_probability(self.distributions[j], X, y+j*n, n)

        for i in range(n):
            y_sum = NEGINF

            for j in range(self.n):
                y[j*n + i] += self.weights_ptr[j]
                y_sum = pair_lse(y_sum, y[j*n + i])

            for j in range(self.n):
                y[j*n + i] -= y_sum

    def predict(self, X, n_jobs=1):
        """Predict the most likely component which generated each sample.

        Calculate the posterior P(M|D) for each sample and return the index
        of the component most likely to fit it. This corresponds to a simple
        argmax over the responsibility matrix.

        This is a sklearn wrapper for the maximum_a_posteriori method.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_dimensions)
            The samples to do the prediction on. Each sample is a row and each
            column corresponds to a dimension in that sample. For univariate
            distributions, a single array may be passed in.

        n_jobs : int
            The number of jobs to use to parallelize, either the number of threads
            or the number of processes to use. -1 means use all available resources.
            Default is 1.

        Returns
        -------
        y : array-like, shape (n_samples,)
            The predicted component which fits the sample the best.
        """

        cdef int i, n, d
        cdef numpy.ndarray X_ndarray
        cdef double* X_ptr

        cdef numpy.ndarray y
        cdef int* y_ptr

        if n_jobs > 1:
            starts = [int(i*len(X)/n_jobs) for i in range(n_jobs)]
            ends = [int(i*len(X)/n_jobs) for i in range(1, n_jobs+1)]

            with Parallel(n_jobs=n_jobs, backend='threading') as parallel:
                y_arrays = parallel(delayed(self.predict, check_pickle=False)(X[start:end])
                    for start, end in zip(starts, ends))

            return numpy.concatenate(y_arrays)

        if not self.is_vl_:
            X_ndarray = _check_input(X, self.keymap)
            X_ptr = <double*> X_ndarray.data
            n, d = len(X_ndarray), len(X_ndarray[0])
            if d != self.d:
                raise ValueError("sample only has {} dimensions but should have {} dimensions".format(d, self.d))
        else:
            X_ndarray = X
            n, d = len(X_ndarray), self.d


        y = numpy.zeros(n, dtype='int32')
        y_ptr = <int*> y.data

        with nogil:
            if not self.is_vl_:
                self._predict(X_ptr, y_ptr, n, d)
            else:
                for i in range(n):
                    with gil:
                        X_ndarray = _check_input(X[i], self.keymap)
                        X_ptr = <double*> X_ndarray.data
                        d = len(X_ndarray)

                    self._predict(X_ptr, y_ptr+i, 1, d)

        return y

    cdef void _predict( self, double* X, int* y, int n, int d) nogil:
        cdef int i, j
        cdef double max_logp, logp
        cdef double* r = <double*> calloc(n*self.n, sizeof(double))

        for j in range(self.n):
            if self.is_vl_:
                r[j] = (<Model> self.distributions_ptr[j])._vl_log_probability(X, d)
            else:
                if self.cython == 1:
                    (<Model> self.distributions_ptr[j])._log_probability(X, r+j*n, n)
                else:
                    with gil:
                        python_log_probability(self.distributions[j], X, r+j*n, n)

        for i in range(n):
            max_logp = NEGINF

            for j in range(self.n):
                logp = r[j*n + i] + self.weights_ptr[j]
                if logp > max_logp:
                    max_logp = logp
                    y[i] = j

        free(r)

    def fit(self, X, y, weights=None, inertia=0.0, pseudocount=0.0,
        stop_threshold=0.1, max_iterations=1e8, callbacks=[],
        return_history=False, verbose=False, n_jobs=1):
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

        callbacks : list, optional
            A list of callback objects that describe functionality that should
            be undertaken over the course of training. Only used for
            semi-supervised learning.

        return_history : bool, optional
            Whether to return the history during training as well as the model.
            Only used for semi-supervised learning.

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
        self : object
            Returns the fitted model
        """

        training_start_time = time.time()
        total_improvement = 0

        X = numpy.array(X, dtype='float64')
        n, d = X.shape

        if weights is None:
            weights = numpy.ones(n, dtype='float64')
        else:
            weights = numpy.array(weights, dtype='float64')

        starts = [int(i*len(X)/n_jobs) for i in range(n_jobs)]
        ends = [int(i*len(X)/n_jobs) for i in range(1, n_jobs+1)]

        callbacks = [History()] + callbacks
        for callback in callbacks:
            callback.model = self
            callback.on_training_begin()

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
                        epoch_end_time = time.time()
                        time_spent = epoch_end_time - epoch_start_time

                        if verbose:
                            print("[{}] Improvement: {}\tTime (s): {:4.4}".format(iteration,
                                improvement, time_spent))

                        total_improvement += improvement

                        logs = {'learning_rate': None,
                                'n_seen_batches' : None,
                                'epoch' : iteration,
                                'improvement' : improvement,
                                'total_improvement' : total_improvement,
                                'log_probability' : log_probability_sum,
                                'last_log_probability' : last_log_probability_sum,
                                'initial_log_probability' : initial_log_probability_sum,
                                'epoch_start_time' : epoch_start_time,
                                'epoch_end_time' : epoch_end_time,
                                'duration' : time_spent }

                        for callback in callbacks:
                            callback.on_epoch_end(logs)

                    iteration += 1
                    last_log_probability_sum = log_probability_sum

                for callback in callbacks:
                    callback.on_training_end(logs)

                self.clear_summaries()

                if verbose:
                    print("Total Improvement: {}".format(total_improvement))


        if verbose:
            total_time_spent = time.time() - training_start_time
            print("Total Time (s): {:.4f}".format(total_time_spent))

        if return_history:
            history = callbacks[0]
            return self, history
        return self

    def summarize(self, X, y, weights=None):
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
            for i, distribution in enumerate(self.distributions):
                distribution.summarize(list(X[y==i]), weights[y==i])
        else:
            for i, distribution in enumerate(self.distributions):
                distribution.summarize(X[y==i], weights[y==i])

        for i in range(self.n):
            weight = weights[y==i].sum()
            self.summaries[i] += weight

    cdef double _summarize(self, double* X, double* weights, int n,
        int column_idx, int d) nogil:
        return -1

    def from_summaries(self, inertia=0.0, pseudocount=0.0, **kwargs):
        """Fit the model to the collected sufficient statistics.

        Fit the parameters of the model to the sufficient statistics gathered
        during the summarize calls. This should return an exact update.

        Parameters
        ----------
        inertia : double, optional
            The weight of the previous parameters of the model. The new
            parameters will roughly be
            old_param*inertia + new_param*(1-inertia),
            so an inertia of 0 means ignore the old parameters, whereas an
            inertia of 1 means ignore the new parameters. Default is 0.0.

        pseudocount : double, optional
            A pseudocount to add to the emission of each distribution. This
            effectively smoothes the states to prevent 0. probability symbols
            if they don't happen to occur in the data. If discrete data, will
            smooth both the prior probabilities of each component and the
            emissions of each component. Otherwise, will only smooth the prior
            probabilities of each component. Default is 0.

        Returns
        -------
        None
        """

        if self.summaries.sum() == 0:
            return

        summaries = self.summaries + pseudocount
        summaries /= summaries.sum()

        for i, distribution in enumerate(self.distributions):
            if isinstance(distribution, DiscreteDistribution):
                distribution.from_summaries(inertia, pseudocount)
            else:
                distribution.from_summaries(inertia, **kwargs)

            self.weights[i] = _log(summaries[i])
            self.summaries[i] = 0.

        return self

    def clear_summaries(self):
        """Remove the stored sufficient statistics.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """

        self.summaries *= 0
        for distribution in self.distributions:
            distribution.clear_summaries()

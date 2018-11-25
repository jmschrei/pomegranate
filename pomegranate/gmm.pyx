#cython: boundscheck=False
#cython: cdivision=True
# gmm.pyx
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
from .kmeans import Kmeans

from distributions.distributions cimport Distribution
from distributions import DiscreteDistribution
from distributions import MultivariateDistribution
from distributions import IndependentComponentsDistribution

from .bayes cimport BayesModel
from .utils cimport _log
from .utils cimport pair_lse
from .utils cimport python_log_probability
from .utils cimport python_summarize
from .utils import _check_input

from .callbacks import History

from joblib import Parallel
from joblib import delayed

DEF NEGINF = float("-inf")
DEF INF = float("inf")

cdef class GeneralMixtureModel(BayesModel):
    """A General Mixture Model.

    This mixture model can be a mixture of any distribution as long as they
    are all of the same dimensionality. Any object can serve as a distribution
    as long as it has fit(X, weights), log_probability(X), and summarize(X,
    weights)/from_summaries() methods if out of core training is desired.

    Parameters
    ----------
    distributions : array-like, shape (n_components,)
        The components of the model as initialized distributions.

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

    Examples
    --------
    >>> from pomegranate import *
    >>>
    >>> d1 = NormalDistribution(5, 2)
    >>> d2 = NormalDistribution(1, 1)
    >>>
    >>> clf = GeneralMixtureModel([d1, d2])
    >>> clf.log_probability(5)
    -2.304562194038089
    >>> clf.predict_proba([[5], [7], [1]])
    array([[ 0.99932952,  0.00067048],
           [ 0.99999995,  0.00000005],
           [ 0.06337894,  0.93662106]])
    >>> clf.fit([[1], [5], [7], [8], [2]])
    >>> clf.predict_proba([[5], [7], [1]])
    array([[ 1.        ,  0.        ],
           [ 1.        ,  0.        ],
           [ 0.00004383,  0.99995617]])
    >>> clf.distributions
    array([ {
        "frozen" :false,
        "class" :"Distribution",
        "parameters" :[
            6.6571359101390755,
            1.2639830514274502
        ],
        "name" :"NormalDistribution"
    },
           {
        "frozen" :false,
        "class" :"Distribution",
        "parameters" :[
            1.498707696758334,
            0.4999983303277837
        ],
        "name" :"NormalDistribution"
    }], dtype=object)
    """

    def __init__(self, distributions, weights=None):
        super(GeneralMixtureModel, self).__init__(distributions, weights)

    def __reduce__(self):
        return self.__class__, (self.distributions.tolist(), numpy.exp(self.weights))

    def fit(self, X, weights=None, inertia=0.0, pseudocount=0.0,
        stop_threshold=0.1, max_iterations=1e8, batch_size=None,
        batches_per_epoch=None, lr_decay=0.0, callbacks=[],
        return_history=False, verbose=False, n_jobs=1):
        """Fit the model to new data using EM.

        This method fits the components of the model to new data using the EM
        method. It will iterate until either max iterations has been reached,
        or the stop threshold has been passed.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_dimensions)
            This is the data to train on. Each row is a sample, and each column
            is a dimension to train on.

        weights : array-like, shape (n_samples,), optional
            The initial weights of each sample in the matrix. If nothing is
            passed in then each sample is assumed to be the same weight.
            Default is None.

        inertia : double, optional
            The weight of the previous parameters of the model. The new
            parameters will roughly be
            old_param*inertia + new_param*(1-inertia),
            so an inertia of 0 means ignore the old parameters, whereas an
            inertia of 1 means ignore the new parameters.
            Default is 0.0.

        pseudocount : double, optional, positive
            A pseudocount to add to the emission of each distribution. This
            effectively smoothes the states to prevent 0. probability symbols
            if they don't happen to occur in the data. Only effects mixture
            models defined over discrete distributions. Default is 0.

        stop_threshold : double, optional, positive
            The threshold at which EM will terminate for the improvement of
            the model. If the model does not improve its fit of the data by
            a log probability of 0.1 then terminate.
            Default is 0.1.

        max_iterations : int, optional, positive
            The maximum number of iterations to run EM for. If this limit is
            hit then it will terminate training, regardless of how well the
            model is improving per iteration.
            Default is 1e8.

        batch_size : int or None, optional
            The number of samples in a batch to summarize on. This controls
            the size of the set sent to `summarize` and so does not make the
            update any less exact. This is useful when training on a memory
            map and cannot load all the data into memory. If set to None,
            batch_size is 1 / n_jobs. Default is None.

        batches_per_epoch : int or None, optional
            The number of batches in an epoch. This is the number of batches to
            summarize before calling `from_summaries` and updating the model
            parameters. This allows one to do minibatch updates by updating the
            model parameters before setting the full dataset. If set to None,
            uses the full dataset. Default is None.

        lr_decay : double, optional, positive
            The step size decay as a function of the number of iterations.
            Functionally, this sets the inertia to be (2+k)^{-lr_decay}
            where k is the number of iterations. This causes initial
            iterations to have more of an impact than later iterations,
            and is frequently used in minibatch learning. This value is
            suggested to be between 0.5 and 1. Default is 0, meaning no
            decay.

        callbacks : list, optional
            A list of callback objects that describe functionality that should
            be undertaken over the course of training.

        return_history : bool, optional
            Whether to return the history during training as well as the model.

        verbose : bool, optional
            Whether or not to print out improvement information over
            iterations.
            Default is False.

        n_jobs : int, optional
            The number of threads to use when parallelizing the job. This
            parameter is passed directly into joblib. Default is 1, indicating
            no parallelism.

        Returns
        -------
        self : GeneralMixtureModel
            The fit mixture model.
        """

        initial_log_probability_sum = NEGINF
        total_improvement = 0
        iteration, improvement = 0, INF
        n = len(X)

        training_start_time = time.time()

        if batch_size is None:
            starts = [int(i*len(X)/n_jobs) for i in range(n_jobs)]
            ends = [int(i*len(X)/n_jobs) for i in range(1, n_jobs+1)]
        else:
            starts = list(range(0, n, batch_size))
            if starts[-1] == n:
                starts = starts[:-1]
            ends = list(range(batch_size, n, batch_size)) + [n]

        minibatching = batches_per_epoch is not None
        batches_per_epoch = batches_per_epoch or len(starts)
        n_seen_batches = 0
        epoch_starts, epoch_ends = None, None

        callbacks = [History()] + callbacks
        for callback in callbacks:
            callback.model = self
            callback.on_training_begin()

        with Parallel(n_jobs=n_jobs, backend='threading') as parallel:
            while improvement > stop_threshold and iteration < max_iterations + 1:
                epoch_start_time = time.time()
                step_size = 1 - ((1 - inertia) * (2 + iteration) ** -lr_decay)
                self.from_summaries(step_size, pseudocount)

                if epoch_starts is not None and minibatching:
                    updated_log_probability_sum = sum(self.log_probability(X[start:end]).sum()
                        for start, end in zip(epoch_starts, epoch_ends))
                    improvement = updated_log_probability_sum - log_probability_sum

                epoch_starts = starts[n_seen_batches:n_seen_batches+batches_per_epoch]
                epoch_ends = ends[n_seen_batches:n_seen_batches+batches_per_epoch]

                n_seen_batches += batches_per_epoch
                if n_seen_batches >= len(starts):
                    n_seen_batches = 0

                if weights is not None:
                    log_probability_sum = sum(parallel(delayed(self.summarize,
                        check_pickle=False)(X[start:end], weights[start:end])
                        for start, end in zip(epoch_starts, epoch_ends)))
                else:
                    log_probability_sum = sum(parallel(delayed(self.summarize,
                        check_pickle=False)(X[start:end]) for start, end in zip(
                            epoch_starts, epoch_ends)))

                if iteration == 0:
                    initial_log_probability_sum = log_probability_sum
                else:
                    epoch_end_time = time.time()
                    time_spent = epoch_end_time - epoch_start_time

                    if not minibatching:
                        improvement = log_probability_sum - last_log_probability_sum

                    if verbose:
                        print("[{}] Improvement: {}\tTime (s): {:.4}".format(
                            iteration, improvement, time_spent))

                    total_improvement += improvement

                    logs = {'learning_rate': step_size,
                            'n_seen_batches' : n_seen_batches,
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
            total_time_spent = time.time() - training_start_time
            print("Total Improvement: {}".format(total_improvement))
            print("Total Time (s): {:.4f}".format(total_time_spent))

        history = callbacks[0]

        if return_history:
            return self, history
        return self

    def summarize(self, X, weights=None):
        """Summarize a batch of data and store sufficient statistics.

        This will run the expectation step of EM and store sufficient
        statistics in the appropriate distribution objects. The summarization
        can be thought of as a chunk of the E step, and the from_summaries
        method as the M step.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_dimensions)
            This is the data to train on. Each row is a sample, and each column
            is a dimension to train on.

        weights : array-like, shape (n_samples,), optional
            The initial weights of each sample in the matrix. If nothing is
            passed in then each sample is assumed to be the same weight.
            Default is None.

        Returns
        -------
        logp : double
            The log probability of the data given the current model. This is
            used to speed up EM.
        """

        cdef int i, n, d
        cdef numpy.ndarray X_ndarray
        cdef numpy.ndarray weights_ndarray
        cdef double log_probability

        if self.is_vl_:
            n, d = len(X), self.d
        elif self.d == 1:
            n, d = X.shape[0], 1
        elif self.d > 1 and X.ndim == 1:
            n, d = 1, len(X)
        else:
            n, d = X.shape

        if weights is None:
            weights_ndarray = numpy.ones(n, dtype='float64')
        else:
            weights_ndarray = numpy.array(weights, dtype='float64')

        cdef double* X_ptr
        cdef double* weights_ptr = <double*> weights_ndarray.data

        if not self.is_vl_:
            X_ndarray = _check_input(X, self.keymap)
            X_ptr = <double*> X_ndarray.data

            with nogil:
                log_probability = self._summarize(X_ptr, weights_ptr, n,
                    0, self.d)

        else:
            log_probability = 0.0
            for i in range(n):
                X_ndarray = _check_input(X[i], self.keymap)
                X_ptr = <double*> X_ndarray.data
                d = len(X_ndarray)
                with nogil:
                    log_probability += self._summarize(X_ptr, weights_ptr+i,
                        d, 0, self.d)

        return log_probability

    cdef double _summarize(self, double* X, double* weights, int n,
        int column_idx, int d) nogil:
        cdef double* r = <double*> calloc(self.n*n, sizeof(double))
        cdef double* summaries = <double*> calloc(self.n, sizeof(double))
        cdef int i, j
        cdef double total, logp, log_probability_sum = 0.0

        memset(summaries, 0, self.n*sizeof(double))

        for j in range(self.n):
            if self.cython == 0:
                with gil:
                    python_log_probability(self.distributions[j], X, r+j*n, n)
            elif self.is_vl_:
                r[j*n] = (<Model> self.distributions_ptr[j])._vl_log_probability(X, n)
            else:
                (<Model> self.distributions_ptr[j])._log_probability(X, r+j*n, n)

        for i in range(n):
            total = NEGINF

            for j in range(self.n):
                r[j*n + i] += self.weights_ptr[j]
                total = pair_lse(total, r[j*n + i])

            for j in range(self.n):
                r[j*n + i] = cexp(r[j*n + i] - total) * weights[i]
                summaries[j] += r[j*n + i]

            log_probability_sum += total * weights[i]

            if self.is_vl_:
                break

        for j in range(self.n):
            if self.cython == 0:
                with gil:
                    python_summarize(self.distributions[j], X, r+j*n, n)
            else:
                (<Model> self.distributions_ptr[j])._summarize(X, r+j*n, n,
                    0, d)

        with gil:
            for j in range(self.n):
                self.summaries_ptr[j] += summaries[j]

        free(r)
        free(summaries)
        return log_probability_sum

    def to_json(self):
        separators=(',', ' : ')
        indent=4

        model = {
                    'class' : 'GeneralMixtureModel',
                    'distributions'  : [ json.loads(dist.to_json())
                                         for dist in self.distributions ],
                    'weights' : numpy.exp(self.weights).tolist()
                }

        return json.dumps(model, separators=separators, indent=indent)

    @classmethod
    def from_json(cls, s):
        d = json.loads(s)
        distributions = [ Distribution.from_json(json.dumps(j))
                          for j in d['distributions'] ]
        model = GeneralMixtureModel(distributions, numpy.array( d['weights'] ))
        return model

    @classmethod
    def from_samples(self, distributions, n_components, X, weights=None,
        n_init=1, init='kmeans++', max_kmeans_iterations=1, inertia=0.0,
        pseudocount=0.0, stop_threshold=0.1, max_iterations=1e8, batch_size=None,
        batches_per_epoch=None, lr_decay=0.0, callbacks=[], return_history=False,
        verbose=False, n_jobs=1):
        """Create a mixture model directly from the given dataset.

        First, k-means will be run using the given initializations, in order to
        define initial clusters for the points. These clusters are used to
        initialize the distributions used. Then, EM is run to refine the
        parameters of these distributions.

        A homogeneous mixture can be defined by passing in a single distribution
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

        n_init : int, optional
            The number of initializations of k-means to do before choosing
            the best. Default is 1.

        init : str, optional
            The initialization algorithm to use for the initial k-means
            clustering. Must be one of 'first-k', 'random', 'kmeans++',
            or 'kmeans||'. Default is 'kmeans++'.

        max_kmeans_iterations : int, optional
            The maximum number of iterations to run kmeans for in the
            initialization step. Default is 1.

        inertia : double, optional
            The weight of the previous parameters of the model. The new
            parameters will roughly be
            old_param*inertia + new_param*(1-inertia),
            so an inertia of 0 means ignore the old parameters, whereas an
            inertia of 1 means ignore the new parameters. Default is 0.0.

        pseudocount : double, optional, positive
            A pseudocount to add to the emission of each distribution. This
            effectively smoothes the states to prevent 0. probability symbols
            if they don't happen to occur in the data. Only effects mixture
            models defined over discrete distributions. Default is 0.

        stop_threshold : double, optional, positive
            The threshold at which EM will terminate for the improvement of
            the model. If the model does not improve its fit of the data by
            a log probability of 0.1 then terminate. Default is 0.1.

        max_iterations : int, optional, positive
            The maximum number of iterations to run EM for. If this limit is
            hit then it will terminate training, regardless of how well the
            model is improving per iteration. Default is 1e8.

        batch_size : int or None, optional
            The number of samples in a batch to summarize on. This controls
            the size of the set sent to `summarize` and so does not make the
            update any less exact. This is useful when training on a memory
            map and cannot load all the data into memory. If set to None,
            batch_size is 1 / n_jobs. Default is None.

        batches_per_epoch : int or None, optional
            The number of batches in an epoch. This is the number of batches to
            summarize before calling `from_summaries` and updating the model
            parameters. This allows one to do minibatch updates by updating the
            model parameters before setting the full dataset. If set to None,
            uses the full dataset. Default is None.

        lr_decay : double, optional, positive
            The step size decay as a function of the number of iterations.
            Functionally, this sets the inertia to be (2+k)^{-lr_decay}
            where k is the number of iterations. This causes initial
            iterations to have more of an impact than later iterations,
            and is frequently used in minibatch learning. This value is
            suggested to be between 0.5 and 1. Default is 0, meaning no
            decay.

        callbacks : list, optional
            A list of callback objects that describe functionality that should
            be undertaken over the course of training.

        return_history : bool, optional
            Whether to return the history during training as well as the model.

        verbose : bool, optional
            Whether or not to print out improvement information over
            iterations. Default is False.

        n_jobs : int, optional
            The number of threads to use when parallelizing the job. This
            parameter is passed directly into joblib. Default is 1, indicating
            no parallelism.
        """

        icd = False
        if not isinstance(X, numpy.ndarray):
            X = numpy.array(X)

        n, d = X.shape

        if not callable(distributions) and not isinstance(distributions, list):
            raise ValueError("must either give initial distributions "
                             "or constructor")

        if callable(distributions):
            d_ = distributions

            if d_ == DiscreteDistribution:
                raise ValueError("cannot fit a discrete GMM "
                                 "without pre-initialized distributions")

            if d == 1:
                distributions = [d_ for i in range(n_components)]
            elif isinstance(d_.blank(), MultivariateDistribution):
                distributions = [d_ for i in range(n_components)]
            elif d_.blank().d > 1:
                distributions = [d_ for i in range(n_components)]
            else:
                icd = True
                distributions = [d_ for i in range(d)]

        else:
            if d == len(distributions):
                icd = True
            else:
                n_components = len(distributions)

            if n_components < 2:
                raise ValueError("must have at least two distributions "
                                 "for general mixture models")

            for dist in distributions:
                if not callable(dist):
                    raise ValueError("must pass in uninitialized distributions")

        kmeans_batch_size = batch_size or len(X)
        X_kmeans = X[:batch_size]

        kmeans = Kmeans(n_components, init=init, n_init=n_init)
        kmeans.fit(X_kmeans, weights=weights, max_iterations=max_kmeans_iterations,
            batch_size=kmeans_batch_size, batches_per_epoch=batches_per_epoch,
            n_jobs=n_jobs)

        y = kmeans.predict(X_kmeans)

        if icd:
            distributions = [IndependentComponentsDistribution.from_samples(X_kmeans[y == i],
                distributions=distributions) for i in range(n_components)]
        else:
            distributions = [distribution.from_samples(X_kmeans[y == i])
                for i, distribution in enumerate(distributions)]

        class_weights = numpy.array([(y == i).mean() for i in range(n_components)])

        model = GeneralMixtureModel(distributions, class_weights)
        _, history = model.fit(X, weights, inertia=inertia, stop_threshold=stop_threshold,
            max_iterations=max_iterations, pseudocount=pseudocount,
            batch_size=batch_size, batches_per_epoch=batches_per_epoch,
            lr_decay=lr_decay, callbacks=callbacks, return_history=True,
            verbose=verbose, n_jobs=n_jobs)

        if return_history:
            return model, history
        return model

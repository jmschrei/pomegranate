# parallel.py
# Contact: Jacob Schreiber <jmschreiber91@gmail.com>

import numpy, time
cimport numpy

import sys

from .base cimport Model
from .hmm import HiddenMarkovModel
from .NaiveBayes import NaiveBayes
from .distributions import Distribution

from joblib import Parallel
from joblib import delayed

NEGINF = float("-inf")
INF = float("inf")

def parallelize(model, X, func, n_jobs, backend):
	"""The main parallelization function.

	This function takes in a model, a dataset, a function to parallelize, the
	number of jobs to do, and the backend, and will chunk up the dataset and
	parallelize the function.

	Parameters
	----------
	model : base.Model
		This is any pomegranate model. All pomegranate models have a cython
		backend which releases the GIL and allows for multithreaded
		parallelization.

	X : numpy.ndarray or list
		The dataset to operate on. For most models this is a numpy array with
		columns corresponding to features and rows corresponding to samples.
		For markov chains and HMMs this will be a list of variable length
		sequences.

	func : <function>
		The function to parallelize. Typically this is one of the model methods
		for prediction or fitting.

	n_jobs : int
		The number of jobs to use to parallelize, either the number of threads
		or the number of processes to use.

	backend : str, 'multiprocessing' or 'threading'
		The parallelization backend of joblib to use. If 'multiprocessing' then
		use the processing backend, if 'threading' then use the threading
		backend.

	Returns
	-------
	results : object
		The results of the method concatenated together across processes.   
	"""

	delay = delayed(getattr(model, func), check_pickle=False)
	with Parallel(n_jobs=n_jobs, backend=backend) as parallel:
		if isinstance(model, HiddenMarkovModel):
			y = parallel(delay(x) for x in X)
		else:
			n = len(X)
			starts = [n/n_jobs*i for i in range(n_jobs)]
			ends = starts[1:] + [n]
			y = parallel(delay(X[start:end]) for start, end in zip(starts, ends))

	return numpy.concatenate(y) if n_jobs > 1 and n_jobs != len(X) else y

def predict(model, X, n_jobs=1, backend='threading'):
	"""Provides for a parallelized predict function.

	This function takes in a model, a dataset, the number of jobs to do, 
	and the backend, and will chunk up the dataset and parallelize the predict
	function.

	Parameters
	----------
	model : base.Model
		This is any pomegranate model. All pomegranate models have a cython
		backend which releases the GIL and allows for multithreaded
		parallelization.

	X : numpy.ndarray or list
		The dataset to operate on. For most models this is a numpy array with
		columns corresponding to features and rows corresponding to samples.
		For markov chains and HMMs this will be a list of variable length
		sequences.

	n_jobs : int
		The number of jobs to use to parallelize, either the number of threads
		or the number of processes to use.

	backend : str, 'multiprocessing' or 'threading'
		The parallelization backend of joblib to use. If 'multiprocessing' then
		use the processing backend, if 'threading' then use the threading
		backend.

	Returns
	-------
	results : object
		The predictions concatenated together across processes.   
	"""

	return parallelize(model, X, 'predict', n_jobs, backend)

def predict_proba(model, X, n_jobs=1, backend='threading'):
	"""Provides for a parallelized predict_proba function.

	This function takes in a model, a dataset, the number of jobs to do, 
	and the backend, and will chunk up the dataset and parallelize the 
	predict_proba function.

	Parameters
	----------
	model : base.Model
		This is any pomegranate model. All pomegranate models have a cython
		backend which releases the GIL and allows for multithreaded
		parallelization.

	X : numpy.ndarray or list
		The dataset to operate on. For most models this is a numpy array with
		columns corresponding to features and rows corresponding to samples.
		For markov chains and HMMs this will be a list of variable length
		sequences.

	n_jobs : int
		The number of jobs to use to parallelize, either the number of threads
		or the number of processes to use.

	backend : str, 'multiprocessing' or 'threading'
		The parallelization backend of joblib to use. If 'multiprocessing' then
		use the processing backend, if 'threading' then use the threading
		backend.

	Returns
	-------
	results : object
		The predictions concatenated together across processes.   
	"""

	return parallelize(model, X, 'predict_proba', n_jobs, backend)

def predict_log_proba(model, X, n_jobs=1, backend='threading'):
	"""Provides for a parallelized predict_log_proba function.

	This function takes in a model, a dataset, the number of jobs to do, 
	and the backend, and will chunk up the dataset and parallelize the 
	predict_log_proba function.

	Parameters
	----------
	model : base.Model
		This is any pomegranate model. All pomegranate models have a cython
		backend which releases the GIL and allows for multithreaded
		parallelization.

	X : numpy.ndarray or list
		The dataset to operate on. For most models this is a numpy array with
		columns corresponding to features and rows corresponding to samples.
		For markov chains and HMMs this will be a list of variable length
		sequences.

	n_jobs : int
		The number of jobs to use to parallelize, either the number of threads
		or the number of processes to use.

	backend : str, 'multiprocessing' or 'threading'
		The parallelization backend of joblib to use. If 'multiprocessing' then
		use the processing backend, if 'threading' then use the threading
		backend.

	Returns
	-------
	results : object
		The predictions concatenated together across processes.   
	"""

	return parallelize(model, X, 'predict_log_proba', n_jobs, backend)

def log_probability(model, X, n_jobs=1, backend='threading'):
	"""Provides for a parallelized log_probability function.

	This function takes in a model, a dataset, the number of jobs to do, 
	and the backend, and will chunk up the dataset and parallelize the 
	log_probability function.

	Parameters
	----------
	model : base.Model
		This is any pomegranate model. All pomegranate models have a cython
		backend which releases the GIL and allows for multithreaded
		parallelization.

	X : numpy.ndarray or list
		The dataset to operate on. For most models this is a numpy array with
		columns corresponding to features and rows corresponding to samples.
		For markov chains and HMMs this will be a list of variable length
		sequences.

	n_jobs : int
		The number of jobs to use to parallelize, either the number of threads
		or the number of processes to use.

	backend : str, 'multiprocessing' or 'threading'
		The parallelization backend of joblib to use. If 'multiprocessing' then
		use the processing backend, if 'threading' then use the threading
		backend.

	Returns
	-------
	results : object
		The log probabilities concatenated together across processes.   
	"""

	return parallelize(model, X, 'log_probability', n_jobs, backend)

def probability(model, X, n_jobs=1, backend='threading'):
	"""Provides for a parallelized probability function.

	This function takes in a model, a dataset, the number of jobs to do, 
	and the backend, and will chunk up the dataset and parallelize the 
	log_probability function followed by exponentiation.

	Parameters
	----------
	model : base.Model
		This is any pomegranate model. All pomegranate models have a cython
		backend which releases the GIL and allows for multithreaded
		parallelization.

	X : numpy.ndarray or list
		The dataset to operate on. For most models this is a numpy array with
		columns corresponding to features and rows corresponding to samples.
		For markov chains and HMMs this will be a list of variable length
		sequences.

	n_jobs : int
		The number of jobs to use to parallelize, either the number of threads
		or the number of processes to use.

	backend : str, 'multiprocessing' or 'threading'
		The parallelization backend of joblib to use. If 'multiprocessing' then
		use the processing backend, if 'threading' then use the threading
		backend.

	Returns
	-------
	results : object
		The probabilities concatenated together across processes.   
	"""

	return numpy.exp(parallelize(model, X, 'log_probability', n_jobs, backend))

def summarize(model, X, weights=None, y=None, n_jobs=1, backend='threading', parallel=None):
	"""Provides for a parallelized summarization function.

	This function takes in a model, a dataset, the number of jobs to do, 
	and the backend, and will chunk up the dataset and parallelize the 
	summarization function.

	Parameters
	----------
	model : base.Model
		This is any pomegranate model. All pomegranate models have a cython
		backend which releases the GIL and allows for multithreaded
		parallelization.

	X : numpy.ndarray or list
		The dataset to operate on. For most models this is a numpy array with
		columns corresponding to features and rows corresponding to samples.
		For markov chains and HMMs this will be a list of variable length
		sequences.

	y : numpy.ndarray or list or None, optional
		Data labels for supervised training algorithms. Default is None

	n_jobs : int
		The number of jobs to use to parallelize, either the number of threads
		or the number of processes to use.

	backend : str, 'multiprocessing' or 'threading'
		The parallelization backend of joblib to use. If 'multiprocessing' then
		use the processing backend, if 'threading' then use the threading
		backend.

	parallel : joblib.Parallel or None
		The worker pool. If you're calling summarize multiple times, it may be
		more efficient to reuse the worker pool rather than create a new one
		each time it is called.

	Returns
	-------
	logp : double
		The log probability of the dataset being summarized. 
	"""

	if isinstance(X, list) and isinstance(model, HiddenMarkovModel):
		n, n_jobs = len(X), len(X)
	elif isinstance(X, list):
		n, d = len(X), model.d
	elif X.ndim == 1 and model.d > 1:
		n, d = 1, X.shape[0]
	elif X.ndim == 1 and model.d == 1:
		n, d = X.shape[0], 1
	else:
		n, d = X.shape

	if weights is None:
		weights = numpy.ones(len(X), dtype='float64')
	else:
		weights = numpy.array(weights, dtype='float64')

	starts = [n/n_jobs*i for i in range(n_jobs)]
	ends = starts[1:] + [n]

	parallel = parallel or Parallel(n_jobs=n_jobs, backend=backend)
	delay = delayed(model.summarize, check_pickle=False)

	if isinstance(model, NaiveBayes):
		y = parallel(delay(X[start:end], y[start:end], weights[start:end]) for start, end in zip(starts, ends))
	else:
		y = parallel(delay(X[start:end], weights[start:end]) for start, end in zip(starts, ends))
	
	return sum(y)

def fit(model, X, weights=None, y=None, n_jobs=1, backend='threading', stop_threshold=1e-3, 
	max_iterations=1e8, inertia=0.0, verbose=False, **kwargs):
	"""Provides for a parallelized fit function.

	This function takes in a model, a dataset, the number of jobs to do, 
	and the backend, and appropriate arguments for fitting, and will chunk 
	up the dataset and parallelize the fit function.

	Parameters
	----------
	model : base.Model
		This is any pomegranate model. All pomegranate models have a cython
		backend which releases the GIL and allows for multithreaded
		parallelization.

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

	n_jobs : int
		The number of jobs to use to parallelize, either the number of threads
		or the number of processes to use. Default is 1.

	backend : str, 'multiprocessing' or 'threading'
		The parallelization backend of joblib to use. If 'multiprocessing' then
		use the processing backend, if 'threading' then use the threading
		backend. Default is 'threading'

	stop_threshold : double, optional, positive
		The threshold at which EM will terminate for the improvement of
		the model. If the model does not improve its fit of the data by
		a log probability of 0.1 then terminate. Default is 1e-3.

	max_iterations : int, optional, positive
		The maximum number of iterations to run EM for. If this limit is
		hit then it will terminate training, regardless of how well the
		model is improving per iteration. Default is 1e8.

	inertia : double, optional
		The weight of the previous parameters of the model. The new
		parameters will roughly be old_param*inertia + new_param*(1-inertia),
		so an inertia of 0 means ignore the old parameters, whereas an
		inertia of 1 means ignore the new parameters. Default is 0.0.

	verbose : bool, optional
		Whether or not to print out improvement information over iterations.
		Default is False.

	Returns
	-------
	improvement : double
		The improvement in log probability after fitting a model, or None if
		fitting a basic distribution.
	"""

	if weights is None:
		weights = numpy.ones(len(X), dtype='float64')
	else:
		weights = numpy.array(weights, dtype='float64')

	if isinstance(model, HiddenMarkovModel):
		return model.fit(X, weights=weights, n_jobs=n_jobs, stop_threshold=stop_threshold, 
			max_iterations=max_iterations, inertia=inertia, verbose=verbose, **kwargs)

	elif isinstance(model, Distribution):
		summarize(model, X, weights, n_jobs, backend)
		model.from_summaries(inertia)

	elif isinstance(model, NaiveBayes):
		model.fit(X, y, weights, n_jobs=n_jobs, inertia=inertia)

	else:
		if isinstance(X, list):
			n, d = len(X), model.d
		elif X.ndim == 1 and model.d == 1:
			n, d = X.shape[0], 1
		else:
			n, d = X.shape

		starts = [n/n_jobs*i for i in range(n_jobs)]
		ends = starts[1:] + [n]

		delay = delayed(model.summarize, check_pickle=False)

		initial_log_probability_sum = NEGINF
		iteration, improvement = 0, INF

		with Parallel(n_jobs=n_jobs, backend=backend) as parallel:
			while improvement > stop_threshold and iteration < max_iterations + 1:
				if model.d == 0:
					log_probability_sum = model.summarize(X, weights)
					initial_log_probability_sum = log_probability_sum
				elif iteration == 0:
					log_probability_sum = sum(parallel(delay(X[start:end], weights[start:end]) for start, end in zip(starts, ends)))
					initial_log_probability_sum = log_probability_sum
				else:
					model.from_summaries(inertia, **kwargs)
					log_probability_sum = sum(parallel(delay(X[start:end], weights[start:end]) for start, end in zip(starts, ends)))
					improvement = log_probability_sum - last_log_probability_sum
					if verbose:
						print( "Improvement: {}".format(improvement) )

				iteration += 1
				last_log_probability_sum = log_probability_sum

		model.clear_summaries()

		if verbose:
			print( "Total Improvement: {}".format(last_log_probability_sum - initial_log_probability_sum) )

		return last_log_probability_sum - initial_log_probability_sum

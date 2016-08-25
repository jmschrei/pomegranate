# parallel.py
# Contact: Jacob Schreiber <jmschreiber91@gmail.com>

import numpy

from .hmm import HiddenMarkovModel
from .NaiveBayes import NaiveBayes
from .distributions import Distribution

from joblib import Parallel
from joblib import delayed

NEGINF = float("-inf")
INF = float("inf")

def parallelize(model, X, func, n_jobs, backend):
	if isinstance(X, list) and isinstance(model, HiddenMarkovModel):
		n, n_jobs = len(X), len(X)
	elif X.ndim == 1 and model.d > 1:
		n, d = 1, X.shape[0]
	elif X.ndim == 1 and model.d == 1:
		n, d = X.shape[0], 1
	else:
		n, d = X.shape

	starts = [n/n_jobs*i for i in range(n_jobs)]
	ends = starts[1:] + [n]

	parallel = Parallel(n_jobs=n_jobs, backend=backend)
	delay = delayed(getattr(model, func), check_pickle=False)

	y = parallel(delay(X[start:end]) for start, end in zip(starts, ends))
	return numpy.concatenate(y) if n_jobs > 1 and n_jobs != n else y

def predict(model, X, n_jobs=1, backend='threading'):
	return parallelize(model, X, 'predict', n_jobs, backend)

def predict_proba(model, X, n_jobs=1, backend='threading'):
	return parallelize(model, X, 'predict_proba', n_jobs, backend)

def predict_log_proba(model, X, n_jobs=1, backend='threading'):
	return parallelize(model, X, 'predict_log_proba', n_jobs, backend)

def log_probability(model, X, n_jobs=1, backend='threading'):
	return parallelize(model, X, 'log_probability', n_jobs, backend)

def summarize(model, X, weights=None, n_jobs=1, backend='threading'):
	if isinstance(X, list) and isinstance(model, HiddenMarkovModel):
		n, n_jobs = len(X), len(X)
	elif X.ndim == 1 and model.d > 1:
		n, d = 1, X.shape[0]
	elif X.ndim == 1 and model.d == 1:
		n, d = X.shape[0], 1
	else:
		n, d = X.shape

	if weights is None:
		weights = numpy.ones(X.shape[0])
	else:
		weights = numpy.array(weights)

	starts = [n/n_jobs*i for i in range(n_jobs)]
	ends = starts[1:] + [n]

	parallel = Parallel(n_jobs=n_jobs, backend=backend)
	delay = delayed(model.summarize, check_pickle=False)

	y = parallel(delay(X[start:end], weights[start:end]) for start, end in zip(starts, ends))
	return sum(y)

def fit(model, X, weights=None, n_jobs=1, backend='threading', stop_threshold=1e-3, max_iterations=1e8, inertia=0.0, verbose=False, **kwargs):
	if weights is None:
		weights = numpy.ones(X.shape[0])
	else:
		weights = numpy.array(weights)

	if isinstance(model, HiddenMarkovModel):
		return model.fit(X, weights=weights, n_jobs=n_jobs, stop_threshold=stop_threshold, 
			max_iterations=max_iterations, inertia=inertia, **kwargs)

	elif isinstance(model, (Distribution, NaiveBayes)):
		summarize(model, X, weights, n_jobs, backend)
		model.from_summaries(inertia)

	else:
		initial_log_probability_sum = NEGINF
		iteration, improvement = 0, INF 

		while improvement > stop_threshold and iteration < max_iterations + 1:
			model.from_summaries(inertia)
			log_probability_sum = summarize(model, X, weights, n_jobs, backend)

			if iteration == 0:
				initial_log_probability_sum = log_probability_sum
			else:
				improvement = log_probability_sum - last_log_probability_sum

				if verbose:
					print( "Improvement: {}".format(improvement) )

			iteration += 1
			last_log_probability_sum = log_probability_sum

		model.clear_summaries()

		if verbose:
			print( "Total Improvement: {}".format(last_log_probability_sum - initial_log_probability_sum) )

		return last_log_probability_sum - initial_log_probability_sum

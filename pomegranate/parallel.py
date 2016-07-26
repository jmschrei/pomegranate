# parallel.py
# Contact: Jacob Schreiber <jmschreiber91@gmail.com>

import numpy

from .hmm import HiddenMarkovModel

from joblib import Parallel
from joblib import delayed

def parallelize(clf, X, func, n_jobs, backend):
	if isinstance(X, list) and isinstance(clf, HiddenMarkovModel):
		n, n_jobs = len(X), len(X)
	elif X.ndim == 1 and self.d > 1:
		n, d = 1, X.shape[0]
	elif X.ndim == 1 and self.d == 1:
		n, d = X.shape[0], 1
	else:
		n, d = X.shape

	starts = [n/n_jobs*i for i in range(n_jobs)]
	ends = starts[1:] + [n]

	parallel = Parallel(n_jobs=n_jobs, backend=backend)
	delay = delayed(getattr(clf, func), check_pickle=False)

	y = parallel(delay(X[start:end]) for start, end in zip(starts, ends))
	return numpy.concatenate(y) if n_jobs > 1 and n_jobs != n else y

def predict(clf, X, n_jobs=1, backend='threading'):
	return parallelize(clf, X, 'predict', n_jobs, backend)

def predict_proba(clf, X, n_jobs=1, backend='threading'):
	return parallelize(clf, X, 'predict_proba', n_jobs, backend)

def predict_log_proba(clf, X, n_jobs=1, backend='threading'):
	return parallelize(clf, X, 'predict_log_proba', n_jobs, backend)

def log_probability(clf, X, n_jobs=1, backend='threading'):
	return parallelize(clf, X, 'log_probability', n_jobs, backend)

def summarize(clf, X, n_jobs=1, backend='threading'):
	return parallelize(clf, X, 'summarize', n_jobs, backend)

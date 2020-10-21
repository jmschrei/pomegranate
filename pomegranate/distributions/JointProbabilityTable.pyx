#!python
#cython: boundscheck=False
#cython: cdivision=True
# JointProbabilityTable.pyx
# Contact: Jacob Schreiber <jmschreiber91@gmail.com>

from libc.stdlib cimport calloc
from libc.stdlib cimport free
from libc.stdlib cimport malloc
from libc.string cimport memset
from libc.math cimport exp as cexp

from ..utils cimport _log
from ..utils cimport isnan
from ..utils import check_random_state
from ..utils import _check_nan

import itertools as it
import numpy
import random
import scipy

from .DiscreteDistribution import DiscreteDistribution

cdef class JointProbabilityTable(MultivariateDistribution):
	"""
	A joint probability table. The primary difference between this and the
	conditional table is that the final column sums to one here. The joint
	table can be thought of as the conditional probability table normalized
	by the marginals of each parent.
	"""

	def __cinit__(self, table, parents=None, frozen=False):
		"""
		Take in the distribution represented as a list of lists, where each
		inner list represents a row.
		"""

		self.name = "JointProbabilityTable"
		self.d = len(parents) if parents is not None else len(table[0]) - 1
		self.m = len(parents) if parents is not None else len(table[0]) - 1
		self.n = len(table)
		self.k = len(set(row[-2] for row in table))
		self.idxs = <int*> malloc((self.m+1)*sizeof(int))

		self.values = <double*> malloc(self.n*sizeof(double))
		self.counts = <double*> calloc(self.n, sizeof(double))

		self.n_columns = self.d

		self.dtypes = []
		for column in table[0]:
			dtype = str(type(column)).split()[-1].strip('>').strip("'")
			self.dtypes.append(dtype)

		self.idxs[0] = 1
		for i in range(self.m-1):
			self.idxs[i+1] = len(set(row[self.m-i-2] for row in table))
		self.idxs[self.m] = 0

		self.keymap = {}
		for i, row in enumerate(table):
			self.keymap[tuple(row[:-1])] = i
			self.values[i] = _log(row[-1])

		self.parents = list(parents) if parents is not None else None
		self.parameters = [[list(row) for row in table], self.parents, self.keymap]

	def __dealloc__(self):
		free(self.idxs)
		free(self.values)
		free(self.counts)

	def __reduce__(self):
		return self.__class__, (self.parameters[0], self.parents, self.frozen)

	def __str__(self):
		return "\n".join(
					"\t".join(map(str, key + (cexp(self.values[idx]),)))
							for key, idx in self.keymap.items())

	def __len__(self):
		return self.k

	def sample(self, n=None, random_state=None):
		random_state = check_random_state(random_state)
		a = random_state.uniform(0, 1)
		values = numpy.cumsum(numpy.exp([self.values[i] for i in range(self.n)]))
		for i in range(self.n):
			if values[i] > a:
				return list(self.keymap.keys())[i]

	def bake(self, keys):
		"""Order the inputs according to some external global ordering."""

		keymap, values = [], []
		for i, key in enumerate(keys):
			keymap.append((key, i))
			idx = self.keymap[key]
			values.append(self.values[idx])

		for i in range(len(keys)):
			self.values[i] = values[i]

		self.keymap = dict(keymap)

	def keys(self):
		return tuple(row for row in self.parameters[2].keys())

	def log_probability(self, X):
		"""
		Return the log probability of a value, which is a tuple in proper
		ordering, like the training data.
		"""

		X = numpy.array(X, ndmin=2, dtype=object)

		log_probabilities = numpy.zeros(X.shape[0])
		for i, x in enumerate(X):
			x = tuple(x)

			for x_ in x:
				if _check_nan(x_):
					break
			else:
				key = self.keymap[x]
				log_probabilities[i] = self.values[key]

		if X.shape[0] == 1:
			return log_probabilities[0]
		return log_probabilities

	cdef void _log_probability(self, double* X, double* log_probability, 
		int n) nogil:
		cdef int i, j, idx

		for i in range(n):
			idx = 0
			for j in range(self.m+1):
				if isnan(X[self.m-j]):
					log_probability[i] = 0.
					break

				idx += self.idxs[j] * <int> X[self.m-j]
			else:
				log_probability[i] = self.values[idx]

	def marginal(self, wrt=-1, neighbor_values=None):
		"""
		Determine the marginal of this table with respect to the index of one
		variable. The parents are index 0..n-1 for n parents, and the final
		variable is either the appropriate value or -1. For example:
		table =
		A    B    C    p(C)
		... data ...
		table.marginal(0) gives the marginal wrt A
		table.marginal(1) gives the marginal wrt B
		table.marginal(2) gives the marginal wrt C
		table.marginal(-1) gives the marginal wrt C
		"""

		if isinstance(neighbor_values, dict):
			neighbor_values = [neighbor_values.get(d, None) for d in self.parents]

		if isinstance(neighbor_values, list):
			wrt = neighbor_values.index(None)

		# Determine the keys for the respective parent distribution
		d = {k: 0 for k in self.parents[wrt].keys()}
		total = 0.0

		for key, idx in self.keymap.items():
			logp = self.values[idx]

			if neighbor_values is not None:
				for j, k in enumerate(key):
					if j == wrt:
						continue

					logp += neighbor_values[j].log_probability(k)

			p = cexp(logp)
			d[key[wrt]] += p
			total += p

		for key, value in d.items():
			d[key] = value / total if total > 0 else 1. / len(self.parents[wrt].keys())

		return DiscreteDistribution(d)

	def summarize(self, items, weights=None):
		"""Summarize the data into sufficient statistics to store."""

		if len(items) == 0 or self.frozen == True:
			return

		if weights is None:
			weights = numpy.ones(len(items), dtype='float64')
		elif numpy.sum(weights) == 0:
			return
		else:
			weights = numpy.asarray(weights, dtype='float64')

		self.__summarize(items, weights)

	cdef void __summarize(self, items, double [:] weights):
		cdef int i, n = len(items)
		cdef tuple item

		for i in range(n):
			item = tuple(items[i])

			if _check_nan(item):
				continue

			key = self.keymap[item]
			self.counts[key] += weights[i]
			self.count += weights[i]

	cdef double _summarize(self, double* items, double* weights, int n,
		int column_idx, int d) nogil:
		cdef int i, j, idx
		cdef double count = 0
		cdef double* counts = <double*> calloc(self.n, sizeof(double))

		for i in range(n):
			idx = 0
			for j in range(self.m+1):
				if isnan(items[self.m-i]):
					break

				idx += self.idxs[i] * <int> items[self.m-i]
			else:
				counts[idx] += weights[i]
				count += weights[i]

		with gil:
			self.count += count
			for i in range(self.n):
				self.counts[i] += counts[i]

		free(counts)

	def from_summaries(self, inertia=0.0, pseudocount=0.0):
		"""Update the parameters of the table."""

		cdef int i, k

		w_sum = sum(self.counts[i] for i in range(self.n))
		if w_sum < 1e-7:
			return

		for i in range(self.n):
			probability = ((self.counts[i] + pseudocount) / 
				(self.count + pseudocount * self.k))
			
			self.values[i] = _log(cexp(self.values[i])*inertia +
				probability*(1-inertia))

		for i in range(self.n):
			self.parameters[0][i][-1] = cexp(self.values[i])

		self.clear_summaries()

	def clear_summaries(self):
		"""Clear the summary statistics stored in the object."""

		self.count = 0
		memset(self.counts, 0, self.n*sizeof(double))

	def fit(self, items, weights=None, inertia=0.0, pseudocount=0.0):
		"""Update the parameters of the table based on the data."""

		self.summarize(items, weights)
		self.from_summaries(inertia, pseudocount)

	def to_dict(self):
		table = [list(key + tuple([cexp(self.values[i])])) for key, i in self.keymap.items()]

		return {
			'class' : 'Distribution',
			'name' : 'JointProbabilityTable',
			'table' : table,
			'dtypes' : self.dtypes,
			'parents' : [dist if isinstance(dist, int) else dist.to_dict() for dist in self.parameters[1]]
		}

	@classmethod
	def from_samples(cls, X, parents=None, weights=None, pseudocount=0.0, 
		keys=None):
		"""Learn the table from data."""

		X = numpy.array(X)
		n, d = X.shape

		if parents is None:
			parents = list(range(X.shape[1]))

		keys = keys or [numpy.unique(X[:,i]).tolist() for i in range(d)]
		m = numpy.prod([len(k) for k in keys])

		table = []
		for key in it.product(*keys):
			table.append(list(key) + [1./m,])

		d = cls(table, parents)
		d.fit(X, weights, pseudocount=pseudocount)
		return d

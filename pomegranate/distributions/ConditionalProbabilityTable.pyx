#!python
#cython: boundscheck=False
#cython: cdivision=True
# ConditionalProbabilityTable.pyx
# Contact: Jacob Schreiber <jmschreiber91@gmail.com>

from libc.stdlib cimport calloc
from libc.stdlib cimport free
from libc.string cimport memset
from libc.math cimport exp as cexp

from ..utils cimport _log
from ..utils cimport isnan
from ..utils import _check_nan
from ..utils import check_random_state

import itertools as it
import json
import numpy
import random
import scipy

from collections import OrderedDict

from .JointProbabilityTable import JointProbabilityTable

cdef class ConditionalProbabilityTable(MultivariateDistribution):
	"""
	A conditional probability table, which is dependent on values from at
	least one previous distribution but up to as many as you want to
	encode for.
	"""

	def __init__(self, table, parents, frozen=False):
		"""
		Take in the distribution represented as a list of lists, where each
		inner list represents a row.
		"""

		self.name = "ConditionalProbabilityTable"
		self.frozen = False
		self.m = len(parents)
		self.n = len(table)
		self.k = len(set(row[-2] for row in table))
		self.idxs = <int*> calloc(self.m+1, sizeof(int))
		self.marginal_idxs = <int*> calloc(self.m, sizeof(int))

		self.values = <double*> calloc(self.n, sizeof(double))
		self.counts = <double*> calloc(self.n, sizeof(double))
		self.marginal_counts = <double*> calloc(self.n / self.k, sizeof(double))

		self.column_idxs = numpy.arange(len(parents)+1, dtype='int32')
		self.column_idxs_ptr = <int*> self.column_idxs.data
		self.n_columns = len(parents) + 1

		self.dtypes = []
		for column in table[0]:
			dtype = str(type(column)).split()[-1].strip('>').strip("'")
			self.dtypes.append(dtype)

		memset(self.counts, 0, self.n*sizeof(double))
		memset(self.marginal_counts, 0, self.n*sizeof(double)/self.k)

		self.idxs[0] = 1
		self.idxs[1] = self.k
		for i in range(self.m-1):
			self.idxs[i+2] = self.idxs[i+1] * len(parents[self.m-i-1])

		self.marginal_idxs[0] = 1
		for i in range(self.m-1):
			self.marginal_idxs[i+1] = self.marginal_idxs[i] * len(parents[self.m-i-1])

		keys = []
		for i, row in enumerate(table):
			keys.append((tuple(row[:-1]), i))
			self.values[i] = _log(row[-1])

		self.keymap = OrderedDict(keys)

		marginal_keys = []
		for i, row in enumerate(table[::self.k]):
			marginal_keys.append((tuple(row[:-2]), i))

		self.marginal_keymap = OrderedDict(marginal_keys)
		self.parents = parents
		self.parameters = [table, self.parents]

	def __dealloc__(self):
		free(self.idxs)
		free(self.values)
		free(self.counts)
		free(self.marginal_idxs)
		free(self.marginal_counts)

	def __reduce__(self):
		"""Serialize the distribution for pickle."""
		return self.__class__, (self.parameters[0], self.parents, self.frozen)

	def __str__(self):
		return "\n".join(
					"\t".join(map(str, key + (cexp(self.values[idx]),)))
							for key, idx in self.keymap.items())

	def __len__(self):
		return self.k

	def keys(self):
		"""
		Return the keys of the probability distribution which has parents,
		the child variable.
		"""

		return tuple(set(row[-1] for row in self.keymap.keys()))

	def bake(self, keys):
		"""Order the inputs according to some external global ordering."""

		keymap, marginal_keymap, values = [], set([]), []
		for i, key in enumerate(keys):
			keymap.append((key, i))
			idx = self.keymap[key]
			values.append(self.values[idx])

		marginal_keys = []
		for i, row in enumerate(keys[::self.k]):
			marginal_keys.append((tuple(row[:-1]), i))

		self.marginal_keymap = OrderedDict(marginal_keys)

		for i in range(len(keys)):
			self.values[i] = values[i]

		self.keymap = OrderedDict(keymap)

	def sample(self, parent_values=None, n=None, random_state=None):
		"""Return a random sample from the conditional probability table."""
		random_state = check_random_state(random_state)

		if parent_values is None:
			parent_values = {}

		for parent in self.parents:
			if parent not in parent_values:
				parent_values[parent] = parent.sample(
					random_state=random_state)

		sample_cands = []
		sample_vals = []
		for key, ind in self.keymap.items():
			for j, parent in enumerate(self.parents):
				if parent_values[parent] != key[j]:
					break
			else:
				sample_cands.append(key[-1])
				sample_vals.append(cexp(self.values[ind]))

		sample_vals /= numpy.sum(sample_vals)
		sample_ind = numpy.where(random_state.multinomial(1, sample_vals))[0][0]

		if n is None:
			return sample_cands[sample_ind]
		else:
			states = random_state.randint(1000000, size=n)
			return [self.sample(parent_values, n=None, random_state=state)
				for state in states]

	def log_probability(self, X):
		"""
		Return the log probability of a value, which is a tuple in proper
		ordering, like the training data.
		"""

		X = tuple(X)

		for x in X:
			if isinstance(x, str):
				if x == 'nan':
					return 0.0
			elif x is None or numpy.isnan(x):
				return 0.0

		idx = self.keymap[X]
		return self.values[idx]

	cdef void _log_probability(self, double* X, double* log_probability, int n) nogil:
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

	def joint(self, neighbor_values=None):
		"""
		This will turn a conditional probability table into a joint
		probability table. If the data is already a joint, it will likely
		mess up the data. It does so by scaling the parameters the probabilities
		by the parent distributions.
		"""

		neighbor_values = neighbor_values or self.parents+[None]
		if isinstance(neighbor_values, dict):
			neighbor_values = [neighbor_values.get(p, None) for p in self.parents + [self]]

		table, total = [], 0
		for key, idx in self.keymap.items():
			scaled_val = self.values[idx]

			for j, k in enumerate(key):
				if neighbor_values[j] is not None:
					scaled_val += neighbor_values[j].log_probability(k)

			scaled_val = cexp(scaled_val)
			total += scaled_val
			table.append(key + (scaled_val,))

		table = [row[:-1] + (row[-1] / total if total > 0 else 1. / self.n,) for row in table]
		return JointProbabilityTable(table, self.parents)

	def marginal(self, neighbor_values=None):
		"""
		Calculate the marginal of the CPT. This involves normalizing to turn it
		into a joint probability table, and then summing over the desired
		value.
		"""

		# Convert from a dictionary to a list if necessary
		if isinstance(neighbor_values, dict):
			neighbor_values = [neighbor_values.get(d, None) for d in self.parents]

		# Get the index we're marginalizing over
		i = -1 if neighbor_values == None else neighbor_values.index(None)
		return self.joint(neighbor_values).marginal(i)

	def fit(self, items, weights=None, inertia=0.0, pseudocount=0.0):
		"""Update the parameters of the table based on the data."""

		self.summarize(items, weights)
		self.from_summaries(inertia, pseudocount)

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

			for symbol in item:
				if _check_nan(symbol):
					break
			else:
				key = self.keymap[item]
				self.counts[key] += weights[i]

				key = self.marginal_keymap[item[:-1]]
				self.marginal_counts[key] += weights[i]

	cdef double _summarize(self, double* items, double* weights, int n,
		int column_idx, int d) nogil:
		cdef int i, j, idx, k
		cdef double* counts = <double*> calloc(self.n, sizeof(double))
		cdef double* marginal_counts = <double*> calloc(self.n / self.k, sizeof(double))

		memset(counts, 0, self.n*sizeof(double))
		memset(marginal_counts, 0, self.n / self.k * sizeof(double))

		for i in range(n):
			idx = 0
			for j in range(self.m+1):
				k = i*self.n_columns + self.column_idxs_ptr[self.m-j]
				if isnan(items[k]):
					break

				idx += self.idxs[j] * <int> items[k]
			else:
				counts[idx] += weights[i]

				idx = 0
				for j in range(self.m):
					k = i*self.n_columns + self.column_idxs_ptr[self.m-1-j]
					idx += self.marginal_idxs[j] * <int> items[k]

				marginal_counts[idx] += weights[i]

		with gil:
			for i in range(self.n / self.k):
				self.marginal_counts[i] += marginal_counts[i]

			for i in range(self.n):
				self.counts[i] += counts[i]

		free(counts)
		free(marginal_counts)

	def from_summaries(self, double inertia=0.0, double pseudocount=0.0):
		"""Update the parameters of the distribution using sufficient statistics."""

		cdef int i, k, idx

		w_sum = sum(self.counts[i] for i in range(self.n))
		if w_sum < 1e-7:
			return

		with nogil:
			for i in range(self.n):
				k = i / self.k

				if self.marginal_counts[k] > 0:
					probability = ((self.counts[i] + pseudocount) /
						(self.marginal_counts[k] + pseudocount * self.k))

					self.values[i] = _log(cexp(self.values[i])*inertia +
						probability*(1-inertia))

				else:
					self.values[i] = -_log(self.k)

		for i in range(self.n):
			idx = self.keymap[tuple(self.parameters[0][i][:-1])]
			self.parameters[0][i][-1] = cexp(self.values[idx])

		self.clear_summaries()

	def clear_summaries(self):
		"""Clear the summary statistics stored in the object."""

		with nogil:
			memset(self.counts, 0, self.n*sizeof(double))
			memset(self.marginal_counts, 0, self.n*sizeof(double)/self.k)

	def to_json(self, separators=(',', ' : '), indent=4):
		"""Serialize the model to a JSON.

		Parameters
		----------
		separators : tuple, optional
		    The two separators to pass to the json.dumps function for formatting.
		    Default is (',', ' : ').

		indent : int, optional
		    The indentation to use at each level. Passed to json.dumps for
		    formatting. Default is 4.

		Returns
		-------
		json : str
		    A properly formatted JSON object.
		"""

		table = [list(key + tuple([cexp(self.values[i])])) for key, i in self.keymap.items()]
		table = [[str(item) for item in row] for row in table]

		model = {
					'class' : 'Distribution',
		            'name' : 'ConditionalProbabilityTable',
		            'table' : table,
		            'dtypes' : self.dtypes,
		            'parents' : [json.loads(dist.to_json()) for dist in self.parents]
		        }

		return json.dumps(model, separators=separators, indent=indent)

	@classmethod
	def from_samples(cls, X, parents, weights=None, pseudocount=0.0):
		"""Learn the table from data."""

		X = numpy.array(X)
		n, d = X.shape

		keys = [numpy.unique(X[:,i]) for i in range(d)]

		for i in range(d):
			keys_ = []
			for key in keys[i]:
				if _check_nan(key):
					continue

				keys_.append(key)

			keys[i] = keys_

		table = []
		for key in it.product(*keys):
			table.append(list(key) + [1./len(keys[-1]),])

		d = ConditionalProbabilityTable(table, parents)
		d.fit(X, weights, pseudocount=pseudocount)
		return d

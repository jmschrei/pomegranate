#cython: boundscheck=False
#cython: cdivision=True
# MarkovChain.pyx
# Contact: Jacob Schreiber <jmschreiber91@gmail.com>

import numpy
import json
import itertools as it

from .distributions import Distribution
from .distributions import DiscreteDistribution
from .distributions import ConditionalProbabilityTable


cdef class MarkovChain(object):
	"""A Markov Chain.

	Implemented as a series of conditional distributions, the Markov chain
	models P(X_i | X_i-1...X_i-k) for a k-th order Markov network. The
	conditional dependencies are directly on the emissions, and not on a
	hidden state as in a hidden Markov model.

	Parameters
	----------
	distributions : list, shape (k+1)
		A list of the conditional distributions which make up the markov chain.
		Begins with P(X_i), then P(X_i | X_i-1). For a k-th order markov chain
		you must put in k+1 distributions.

	Attributes
	----------
	distributions : list, shape (k+1)
		The distributions which make up the chain.

	Examples
	--------
	>>> from pomegranate import *
	>>> d1 = DiscreteDistribution({'A': 0.25, 'B': 0.75})
	>>> d2 = ConditionalProbabilityTable([['A', 'A', 0.33],
		                             ['B', 'A', 0.67],
		                             ['A', 'B', 0.82],
		                             ['B', 'B', 0.18]], [d1])
	>>> mc = MarkovChain([d1, d2])
	>>> mc.log_probability(list('ABBAABABABAABABA'))
	-8.9119890701808213
	"""

	cdef int k
	cdef public list distributions

	def __init__(self, distributions):
		self.k = len(distributions) - 1
		self.distributions = distributions

	def __reduce__(self):
		return self.__class__, (self.distributions,)

	def log_probability(self, sequence):
		"""Calculate the log probability of the sequence under the model.

		This calculates the first slices of increasing size under the
		corresponding first few components of the model until size k is
		reached, at which all slices are evaluated under the final component.

		Parameters
		----------
		sequence : array-like
			An array of observations

		Returns
		-------
		logp : double
			The log probability of the sequence under the model.
		"""

		n, k = len(sequence), self.k
		l = min(k, n)
		logp = 0.0

		for i in range(l):
			key = sequence[0] if i == 0 else tuple(sequence[:i+1])
			logp += self.distributions[i].log_probability(key)

		for j in range(n-l):
			key = tuple(sequence[j:j+k+1]) if k > 0 else sequence[j]
			logp += self.distributions[-1].log_probability(key)

		return logp

	def sample(self, length):
		"""Create a random sample from the model.

		Parameters
		----------
		length : int or Distribution
			Give either the length of the sample you want to generate, or a
			distribution object which will be randomly sampled for the length.
			Continuous distributions will have their sample rounded to the
			nearest integer, minimum 1.

		Returns
		-------
		sequence : array-like, shape = (length,)
			A sequence randomly generated from the markov chain.
		"""

		if isinstance(length, Distribution):
			length = int(length.sample())

		length = max(length, 1)

		sequence = [ self.distributions[0].sample() ]
		if length > self.k:
			for j, distribution in enumerate(self.distributions[1:]):
				parents = { self.distributions[l] : sequence[l]
				            for l in range(j+1)}
				sequence.append( distribution.sample(parents) )
			for l in range(length - self.k - 1):
				parents = { self.distributions[k] : sequence[l+k+1]
				            for k in range(self.k)}
				sequence.append( self.distributions[-1].sample(parents) )

		return sequence

	def fit(self, sequences, weights=None, inertia=0.0):
		"""Fit the model to new data using MLE.

		The underlying distributions are fed in their appropriate points and
		weights and are updated.

		Parameters
		----------
		sequences : array-like, shape (n_samples, variable)
			This is the data to train on. Each row is a sample which contains
			a sequence of variable length

		weights : array-like, shape (n_samples,), optional
			The initial weights of each sample. If nothing is passed in then
			each sample is assumed to be the same weight. Default is None.

		inertia : double, optional
			The weight of the previous parameters of the model. The new
			parameters will roughly be
			old_param*inertia + new_param*(1-inertia), so an inertia of 0 means
			ignore the old parameters, whereas an inertia of 1 means ignore the
			new parameters. Default is 0.0.

		Returns
		-------
		None
		"""

		self.summarize(sequences, weights)
		self.from_summaries(inertia)

	def summarize(self, sequences, weights=None):
		"""Summarize a batch of data and store sufficient statistics.

		This will summarize the sequences into sufficient statistics stored in
		each distribution.

		Parameters
		----------
		sequences : array-like, shape (n_samples, variable)
			This is the data to train on. Each row is a sample which contains
			a sequence of variable length

		weights : array-like, shape (n_samples,), optional
			The initial weights of each sample. If nothing is passed in then
			each sample is assumed to be the same weight. Default is None.

		Returns
		-------
		None
		"""

		if weights is None:
			weights = numpy.ones(len(sequences), dtype='float64')
		else:
			weights = numpy.array(weights)

		n = max( map(len, sequences) )
		for i in range(self.k):
			if i == 0:
				symbols = [ sequence[0] for sequence in sequences ]
			else:
				symbols = [ sequence[:i+1]
				            for sequence in sequences if len(sequence) > i ]
			self.distributions[i].summarize(symbols, weights)

		for j in range(n-self.k):
			if self.k == 0:
				symbols = [ sequence[j]
				            for sequence in sequences
				            if len(sequence) > j+self.k ]
			else:
				symbols = [ sequence[j:j+self.k+1]
				            for sequence in sequences
				            if len(sequence) > j+self.k ]

			self.distributions[-1].summarize(symbols, weights)

	def from_summaries(self, inertia=0.0):
		"""Fit the model to the collected sufficient statistics.

		Fit the parameters of the model to the sufficient statistics gathered
		during the summarize calls. This should return an exact update.

		Parameters
		----------
		inertia : double, optional
			The weight of the previous parameters of the model. The new
			parameters will roughly be
			old_param*inertia + new_param * (1-inertia), so an inertia of 0
			means ignore the old parameters, whereas an inertia of 1 means
			ignore the new parameters. Default is 0.0.

		Returns
		-------
		None
		"""

		for i in range(self.k+1):
			self.distributions[i].from_summaries(inertia=inertia)

	def to_json(self, separators=(',', ' : '), indent=4):
		"""Serialize the model to a JSON.

		Parameters
		----------
		separators : tuple, optional
		    The two separaters to pass to the json.dumps function for
		    formatting. Default is (',', ' : ').

		indent : int, optional
		    The indentation to use at each level. Passed to json.dumps for
		    formatting. Default is 4.

		Returns
		-------
		json : str
		    A properly formatted JSON object.
		"""

		model = {
		            'class' : 'MarkovChain',
		            'distributions' : [ json.loads( d.to_json() )
		                                for d in self.distributions ]
		        }

		return json.dumps(model, separators=separators, indent=indent)

	@classmethod
	def from_json(cls, s):
		"""Read in a serialized model and return the appropriate classifier.

		Parameters
		----------
		s : str
		    A JSON formatted string containing the file.

		Returns
		-------
		model : object
		    A properly initialized and baked model.
		"""

		d = json.loads(s)
		distributions = [ Distribution.from_json( json.dumps(j) )
		                  for j in d['distributions'] ]
		model = MarkovChain(distributions)
		return model

	@classmethod
	def from_samples(cls, X, weights=None, k=1):
		"""Learn the Markov chain from data.

		Takes in the memory of the chain (k) and learns the initial
		distribution and probability tables associated with the proper
		parameters.

		Parameters
		----------
		X : array-like, list or numpy.array
			The data to fit the structure too as a list of sequences of
			variable length. Since the data will be of variable length,
			there is no set form 

		weights : array-like, shape (n_nodes), optional
			The weight of each sample as a positive double. Default is None.

		k : int, optional
			The number of samples back to condition on in the model. Default
			is 1.

		Returns
		-------
		model : MarkovChain
			The learned markov chain model.
		"""

		symbols = set()
		for seq in X:
			for symbol in seq:
				symbols.add(symbol)

		n = len(symbols)

		d = DiscreteDistribution({ symbol: 1./n for symbol in symbols })
		distributions = [d]

		for i in range(1, k+1):
			table = []
			for key in it.product(symbols, repeat=i+1):
				table.append( list(key) + [1./n] )

			d = ConditionalProbabilityTable(table, distributions[:])
			distributions.append(d)

		model = MarkovChain(distributions)
		model.fit(X)
		return model

#cython: boundscheck=False
#cython: cdivision=True
# MarkovChain.pyx
# Contact: Jacob Schreiber <jmschreiber91@gmail.com>


cdef class MarkovChain(object):
	"""A Markov Chain.

	Implemented as a series of conditional distributions, the Markov chain
	models P(X_i | X_i-1...X_i-k) for a k-th order Markov network."""

	cdef int k
	cdef list distributions

	def __init__(self, distributions):
		self.k = len(distributions) - 1
		self.distributions = distributions

	def log_probability(self, sequence):
		"""Calculate the log probability of the sequence under the model.

		This is the first slices under the first few components of the model,
		and then the remaining slices under the last component."""

		n, k = len(sequence), self.k
		l = min(k, n)
		logp = 0.0

		for i in range(l):
			key = sequence[0] if i == 0 else tuple(sequence[:i+1])
			logp += self.distributions[i].log_probability(key)

		for j in range(n-l):
			key = tuple(sequence[j:j+k+1])
			logp += self.distributions[-1].log_probability(key)

		return logp

	def fit(self, sequences):
		"""Fit the model to new data."""

		assert isinstance(sequence, list), "sequences must be of type list"

		n = max( map( len, sequences ) )
		for i in range(self.k):
			symbols = [ sequence[:i+1] for sequence in sequences if len(sequence) > i ]
			self.distributions[i].fit(symbols)
		for j in range(n-self.k):
			symbols = [ sequence[j:j+self.k+1] for sequence in sequences if len(sequence) > j+self.k ]
			self.distributions[-1].summarize(symbols)

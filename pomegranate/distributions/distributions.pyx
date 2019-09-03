#!python
#cython: boundscheck=False
#cython: cdivision=True
# distributions.pyx
# Contact: Jacob Schreiber <jmschreiber91@gmail.com>


import json
import numpy
import sys

from ..utils import weight_set

from .UniformDistribution import UniformDistribution
from .BernoulliDistribution import BernoulliDistribution
from .NormalDistribution import NormalDistribution
from .LogNormalDistribution import LogNormalDistribution
from .ExponentialDistribution import ExponentialDistribution
from .BetaDistribution import BetaDistribution
from .GammaDistribution import GammaDistribution
from .DiscreteDistribution import DiscreteDistribution
from .PoissonDistribution import PoissonDistribution

from .KernelDensities import KernelDensity
from .KernelDensities import UniformKernelDensity
from .KernelDensities import GaussianKernelDensity
from .KernelDensities import TriangleKernelDensity

from .IndependentComponentsDistribution import IndependentComponentsDistribution
from .MultivariateGaussianDistribution import MultivariateGaussianDistribution
from .DirichletDistribution import DirichletDistribution
from .ConditionalProbabilityTable import ConditionalProbabilityTable
from .JointProbabilityTable import JointProbabilityTable

from .NeuralNetworkWrapper import NeuralNetworkWrapper

cdef class Distribution(Model):
	"""A probability distribution.

	Represents a probability distribution over the defined support. This is
	the base class which must be subclassed to specific probability
	distributions. All distributions have the below methods exposed.

	Parameters
	----------
	Varies on distribution.

	Attributes
	----------
	name : str
		The name of the type of distributioon.

	summaries : list
		Sufficient statistics to store the update.

	frozen : bool
		Whether or not the distribution will be updated during training.

	d : int
		The dimensionality of the data. Univariate distributions are all
		1, while multivariate distributions are > 1.
	"""

	def __cinit__(self):
		self.name = "Distribution"
		self.frozen = False
		self.summaries = []
		self.d = 1

	def marginal(self, *args, **kwargs):
		"""Return the marginal of the distribution.

		Parameters
		----------
		*args : optional
			Arguments to pass in to specific distributions

		**kwargs : optional
			Keyword arguments to pass in to specific distributions

		Returns
		-------
		distribution : Distribution
			The marginal distribution. If this is a multivariate distribution
			then this method is filled in. Otherwise returns self.
		"""

		return self

	def copy(self):
		"""Return a deep copy of this distribution object.

		This object will not be tied to any other distribution or connected
		in any form.

		Paramters
		---------
		None

		Returns
		-------
		distribution : Distribution
			A copy of the distribution with the same parameters.
		"""

		return self.__class__(*self.parameters)

	def log_probability(self, X):
		"""Return the log probability of the given X under this distribution.

		Parameters
		----------
		X : double
			The X to calculate the log probability of (overridden for
			DiscreteDistributions)

		Returns
		-------
		logp : double
			The log probability of that point under the distribution.
		"""

		cdef int i, n
		cdef double logp
		cdef numpy.ndarray logp_array
		cdef numpy.ndarray X_ndarray
		cdef double* X_ptr
		cdef double* logp_ptr

		n = 1 if isinstance(X, (int, float)) else len(X)

		logp_array = numpy.empty(n, dtype='float64')
		logp_ptr = <double*> logp_array.data

		X_ndarray = numpy.array(X, dtype='float64')
		X_ptr = <double*> X_ndarray.data

		self._log_probability(X_ptr, logp_ptr, n)

		if n == 1:
			return logp_array[0]
		else:
			return logp_array

	def fit(self, items, weights=None, inertia=0.0, column_idx=0):
		"""
		Set the parameters of this Distribution to maximize the likelihood of
		the given sample. Items holds some sort of sequence. If weights is
		specified, it holds a sequence of value to weight each item by.
		"""

		if self.frozen:
			return

		self.summarize(items, weights, column_idx)
		self.from_summaries(inertia)

	def summarize(self, items, weights=None, column_idx=0):
		"""
		Take in a series of items and their weights and reduce it down to a
		summary statistic to be used in training later.
		"""

		items, weights = weight_set(items, weights)
		if weights.sum() <= 0:
			return

		cdef double* items_p = <double*> (<numpy.ndarray> items).data
		cdef double* weights_p = <double*> (<numpy.ndarray> weights).data
		cdef int n = items.shape[0]
		cdef int d = 1
		cdef int column_id = <int> column_idx

		if items.ndim == 2:
			d = items.shape[1]

		with nogil:
			self._summarize(items_p, weights_p, n, column_id, d)

	cdef double _summarize(self, double* items, double* weights, int n,
		int column_idx, int d) nogil:
		pass

	def from_summaries(self, inertia=0.0):
		"""Fit the distribution to the stored sufficient statistics.
		Parameters
		----------
		inertia : double, optional
			The weight of the previous parameters of the model. The new
			parameters will roughly be old_param*inertia + new_param*(1-inertia),
			so an inertia of 0 means ignore the old parameters, whereas an
			inertia of 1 means ignore the new parameters. Default is 0.0.

		Returns
		-------
		None
		"""

		# If the distribution is frozen, don't bother with any calculation
		if self.frozen == True:
			return

		self.fit(*self.summaries, inertia=inertia)
		self.summaries = []

	def clear_summaries(self):
		"""Clear the summary statistics stored in the object.
		Parameters
		----------
		None
		Returns
		-------
		None
		"""

		self.summaries = []

	def plot(self, n=1000, **kwargs):
		"""Plot the distribution by sampling from it.

		This function will plot a histogram of samples drawn from a distribution
		on the current open figure.

		Parameters
		----------
		n : int, optional
			The number of samples to draw from the distribution. Default is
			1000.

		**kwargs : arguments, optional
			Arguments to pass to matplotlib's histogram function.

		Returns
		-------
		None
		"""

		import matplotlib.pyplot as plt
		plt.hist(self.sample(n), **kwargs)

	def to_json(self, separators=(',', ' :'), indent=4):
		"""Serialize the distribution to a JSON.

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

		return json.dumps({
								'class' : 'Distribution',
								'name'  : self.name,
								'parameters' : self.parameters,
								'frozen' : self.frozen
						   }, separators=separators, indent=indent)

	@classmethod
	def from_json(cls, s):
		"""Read in a serialized distribution and return the appropriate object.

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

		if ' ' in d['class'] or 'Distribution' not in d['class']:
			raise SyntaxError("Distribution object attempting to read invalid object.")

		if d['name'] == 'IndependentComponentsDistribution':
			d['parameters'][0] = [cls.from_json(json.dumps(dist)) for dist in d['parameters'][0]]
			return IndependentComponentsDistribution(d['parameters'][0], d['parameters'][1], d['frozen'])
		elif d['name'] == 'DiscreteDistribution':
			if d['dtype'] in ('str', 'unicode', 'numpy.string_'):
				dist = {str(key) : value for key, value in d['parameters'][0].items()}
			elif d['dtype'] == 'int':
				dist = {int(key) : value for key, value in d['parameters'][0].items()}
			elif d['dtype'] == 'float':
				dist = {float(key) : value for key, value in d['parameters'][0].items()}
			else:
				dist = d['parameters'][0]

			return DiscreteDistribution(dist, frozen=d['frozen'])

		elif 'Table' in d['name']:
			parents = [Distribution.from_json(json.dumps(j)) for j in d['parents']]
			table = []

			for row in d['table']:
				table.append([])
				for dtype, item in zip(d['dtypes'], row):
					if dtype in ('str', 'unicode', 'numpy.string_'):
						table[-1].append(str(item))
					elif dtype == 'int':
						table[-1].append(int(item))
					elif dtype == 'float':
						table[-1].append(float(item))
					else:
						table[-1].append(item)

			if d['name'] == 'JointProbabilityTable':
				return JointProbabilityTable(table, parents)
			elif d['name'] == 'ConditionalProbabilityTable':
				return ConditionalProbabilityTable(table, parents)

		else:
			dist = eval("{}({}, frozen={})".format(d['name'],
			                                    ','.join(map(str, d['parameters'])),
			                                     d['frozen']))
			return dist

	@classmethod
	def from_samples(cls, items, weights=None, **kwargs):
		"""Fit a distribution to some data without pre-specifying it."""

		distribution = cls.blank()
		distribution.fit(items, weights, **kwargs)
		return distribution




cdef class MultivariateDistribution(Distribution):
	"""
	An object to easily identify multivariate distributions such as tables.
	"""

	def log_probability(self, X):
		"""Return the log probability of the given X under this distribution.

		Parameters
		----------
		X : list or numpy.ndarray
			The point or points to calculate the log probability of. If one
			point is passed in, then it will return a single log probability.
			If a vector of points is passed in, then it will return a vector
			of log probabilities.

		Returns
		-------
		logp : double or numpy.ndarray
			The log probability of that point under the distribution. If a
			single point is passed in, it will return a single double
			corresponding to that point. If a vector of points is passed in
			then it will return a numpy array of log probabilities for each
			point.
		"""

		cdef int i, n
		cdef numpy.ndarray logp_array
		cdef numpy.ndarray X_ndarray
		cdef double* X_ptr
		cdef double* logp_ptr

		if isinstance(X[0], (int, float)) or len(X) == 1:
			n = 1
		else:
			n = len(X)

		X_ndarray = numpy.array(X, dtype='float64')
		X_ptr = <double*> X_ndarray.data

		logp_array = numpy.empty(n, dtype='float64')
		logp_ptr = <double*> logp_array.data

		with nogil:
			self._log_probability(X_ptr, logp_ptr, n)

		if n == 1:
			return logp_array[0]
		else:
			return logp_array

#!python
#cython: boundscheck=False
#cython: cdivision=True
# GammaDistribution.pyx
# Contact: Jacob Schreiber <jmschreiber91@gmail.com>

import numpy
import scipy
import random


from ..utils cimport _log
from ..utils cimport isnan
from ..utils cimport lgamma
from ..utils import check_random_state

from libc.math cimport sqrt as csqrt

# Define some useful constants
DEF NEGINF = float("-inf")
DEF INF = float("inf")

cdef class GammaDistribution(Distribution):
	"""
	This distribution represents a gamma distribution, parameterized in the
	alpha/beta (shape/rate) parameterization. ML estimation for a gamma
	distribution, taking into account weights on the data, is nontrivial, and I
	was unable to find a good theoretical source for how to do it, so I have
	cobbled together a solution here from less-reputable sources.
	"""

	property parameters:
		def __get__(self):
			return [self.alpha, self.beta]
		def __set__(self, parameters):
			self.alpha, self.beta = parameters

	def __init__(self, double alpha, double beta, bint frozen=False):
		self.alpha = alpha
		self.beta = beta
		self.summaries = [0, 0, 0]
		self.name = "GammaDistribution"
		self.frozen = frozen

	def __reduce__(self):
		"""Serialize distribution for pickling."""
		return self.__class__, (self.alpha, self.beta, self.frozen)

	cdef void _log_probability(self, double* X, double* log_probability, int n) nogil:
		cdef int i
		cdef double alpha = self.alpha
		cdef double beta = self.beta

		for i in range(n):
			if isnan(X[i]):
				log_probability[i] = 0.
			else:
				log_probability[i] = (_log(beta) * alpha - lgamma(alpha) +
					_log(X[i]) * (alpha - 1) - beta * X[i])

	def sample(self, n=None, random_state=None):
		random_state = check_random_state(random_state)
		return random_state.gamma(self.parameters[0], 1.0 / self.parameters[1], n)

	def fit(self, items, weights=None, inertia=0.0, epsilon=1E-9,
		iteration_limit=1000, column_idx=0):
		"""
		Set the parameters of this Distribution to maximize the likelihood of
		the given sample. Items holds some sort of sequence. If weights is
		specified, it holds a sequence of value to weight each item by.
		In the Gamma case, likelihood maximization is necesarily numerical, and
		the extension to weighted values is not trivially obvious. The algorithm
		used here includes a Newton-Raphson step for shape parameter estimation,
		and analytical calculation of the rate parameter. The extension to
		weights is constructed using vital information found way down at the
		bottom of an Experts Exchange page.
		Newton-Raphson continues until the change in the parameter is less than
		epsilon, or until iteration_limit is reached
		See:
		http://en.wikipedia.org/wiki/Gamma_distribution
		http://www.experts-exchange.com/Other/Math_Science/Q_23943764.html
		"""

		self.summarize(items, weights, column_idx)
		self.from_summaries(inertia, epsilon, iteration_limit)

	def summarize(self, items, weights=None, column_idx=0):
		"""
		Take in a series of items and their weights and reduce it down to a
		summary statistic to be used in training later.
		"""

		if len(items) == 0:
			# No sample, so just ignore it and keep our old parameters.
			return

		# Make it be a numpy array
		items = numpy.asarray(items)

		if weights is None:
			# Weight everything 1 if no weights specified
			weights = numpy.ones_like(items)
		else:
			# Force whatever we have to be a Numpy array
			weights = numpy.asarray(weights)

		if weights.sum() == 0:
			# Since negative weights are banned, we must have no data.
			# Don't change the parameters at all.
			return

		# Save the weighted average of the items, and the weighted average of
		# the log of the items.
		self.summaries[0] += items.dot(weights)
		self.summaries[1] += numpy.log(items).dot(weights)
		self.summaries[2] += weights.sum()

	cdef double _summarize(self, double* items, double* weights, int n,
		int column_idx, int d) nogil:
		cdef int i
		cdef double xw = 0, logxw = 0, w = 0
		cdef double item

		for i in range(n):
			item = items[i*d + column_idx]
			if isnan(item):
				continue

			w += weights[i]
			xw += item * weights[i]
			logxw += _log(item) * weights[i]

		with gil:
			self.summaries[0] += xw
			self.summaries[1] += logxw
			self.summaries[2] += w

	def from_summaries(self, inertia=0.0, epsilon=1e-4,
		iteration_limit=100):
		"""
		Set the parameters of this Distribution to maximize the likelihood of
		the given sample given the summaries which have been stored.
		In the Gamma case, likelihood maximization is necesarily numerical, and
		the extension to weighted values is not trivially obvious. The algorithm
		used here includes a Newton-Raphson step for shape parameter estimation,
		and analytical calculation of the rate parameter. The extension to
		weights is constructed using vital information found way down at the
		bottom of an Experts Exchange page.
		Newton-Raphson continues until the change in the parameter is less than
		epsilon, or until iteration_limit is reached
		See:
		http://en.wikipedia.org/wiki/Gamma_distribution
		http://www.experts-exchange.com/Other/Math_Science/Q_23943764.html
		"""

		# If the distribution is frozen, don't bother with any calculation
		if self.summaries[2] < 1e-7 or self.frozen == True:
			return

		# First, do Newton-Raphson for shape parameter.

		# Calculate the sufficient statistic s, which is the log of the average
		# minus the average log. When computing the average log, we weight
		# outside the log function. (In retrospect, this is actually pretty
		# obvious.)
		statistic = _log(self.summaries[0] / self.summaries[2]) - \
			self.summaries[1] / self.summaries[2]

		# Start our Newton-Raphson at what Wikipedia claims a 1969 paper claims
		# is a good approximation.
		# Really, start with new_shape set, and shape set to be far away from it
		shape = float("inf")

		if statistic != 0:
			# Not going to have a divide by 0 problem here, so use the good
			# estimate
			new_shape =  (3 - statistic + csqrt((statistic - 3) ** 2 + 24 *
				statistic)) / (12 * statistic)
		if statistic == 0 or new_shape <= 0:
			# Try the current shape parameter
			new_shape = self.parameters[0]

		# Count the iterations we take
		iteration = 0

		# Now do the update loop.
		# We need the digamma (gamma derivative over gamma) and trigamma
		# (digamma derivative) functions. Luckily, scipy.special.polygamma(0, x)
		# is the digamma function (0th derivative of the digamma), and
		# scipy.special.polygamma(1, x) is the trigamma function.

		while abs(shape - new_shape) > epsilon and iteration < iteration_limit:
			shape = new_shape

			new_shape = shape - (_log(shape) -
				scipy.special.polygamma(0, shape) -
				statistic) / (1.0 / shape - scipy.special.polygamma(1, shape))

			#print new_shape, scipy.special.polygamma(1, shape)

			# Don't let shape escape from valid values
			if abs(new_shape) == float("inf") or new_shape == 0:
				# Hack the shape parameter so we don't stop the loop if we land
				# near it.
				shape = new_shape

				# Re-start at some random place.
				new_shape = random.random()

			iteration += 1

		# Might as well grab the new value
		shape = new_shape

		# Now our iterative estimation of the shape parameter has converged.
		# Calculate the rate parameter
		rate = 1.0 / (1.0 / (shape * self.summaries[2]) * self.summaries[0])

		# Get the previous parameters
		prior_shape, prior_rate = self.parameters


		# Calculate the new parameters, respecting inertia, with an inertia
		# of 0 being completely replacing the parameters, and an inertia of
		# 1 being to ignore new training data.
		self.alpha = prior_shape*inertia + shape*(1.0-inertia)
		self.beta =	prior_rate*inertia + rate*(1.0-inertia)
		self.summaries = [0, 0, 0]

	def clear_summaries(self):
		"""Clear the summary statistics stored in the object."""

		self.summaries = [0, 0, 0]

	@classmethod
	def blank(cls):
		return GammaDistribution(0, 0)

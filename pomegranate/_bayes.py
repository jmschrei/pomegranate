# _bayes.py
# Author: Jacob Schreiber <jmschreiber91@gmail.com>


import torch

from ._utils import _cast_as_tensor
from ._utils import _update_parameter
from ._utils import _check_parameter
from ._utils import _reshape_weights

from .distributions._distribution import Distribution


class BayesMixin(torch.nn.Module):
	def _reset_cache(self):
		"""Reset the internally stored statistics.

		This method is meant to only be called internally. It resets the
		stored statistics used to update the model parameters as well as
		recalculates the cached values meant to speed up log probability
		calculations.
		"""

		if self._initialized == False:
			return

		self.register_buffer("_w_sum", torch.zeros(self.k, device=self.device))
		self.register_buffer("_log_priors", torch.log(self.priors))

	def _emission_matrix(self, X, priors=None):
		"""Return the emission/responsibility matrix.

		This method returns the log probability of each example under each
		distribution contained in the model with the log prior probability
		of each component added.


		Parameters
		----------
		X: list, tuple, numpy.ndarray, torch.Tensor, shape=(-1, self.d)
			A set of examples to evaluate. 

		priors: list, numpy.ndarray, torch.Tensor, shape=(-1, self.k)
			Prior probabilities of assigning each symbol to each node. If not
			provided, do not include in the calculations (conceptually
			equivalent to a uniform probability, but without scaling the
			probabilities). This can be used to assign labels to observatons
			by setting one of the probabilities for an observation to 1.0.
			Note that this can be used to assign hard labels, but does not
			have the same semantics for soft labels, in that it only
			influences the initial estimate of an observation being generated
			by a component, not gives a target. Default is None.

	
		Returns
		-------
		e: torch.Tensor, shape=(-1, self.k)
			A set of log probabilities for each example under each distribution.
		"""

		X = _check_parameter(_cast_as_tensor(X), "X", ndim=2, 
			shape=(-1, self.d), check_parameter=self.check_data)

		priors = _check_parameter(_cast_as_tensor(priors), "priors",
			ndim=2, shape=(X.shape[0], self.k), min_value=0.0, max_value=1.0,
			value_sum=1.0, value_sum_dim=-1, check_parameter=self.check_data)

		d = X.shape[0]
		e = torch.empty(d, self.k, device=self.device, dtype=self.dtype)
		for i, d in enumerate(self.distributions):
			e[:, i] = d.log_probability(X)

		if priors is not None:
			e += torch.log(priors)

		return e + self._log_priors

	def probability(self, X, priors=None):
		"""Calculate the probability of each example.

		This method calculates the probability of each example given the
		parameters of the distribution. The examples must be given in a 2D
		format.

		Note: This differs from some other probability calculation
		functions, like those in torch.distributions, because it is not
		returning the probability of each feature independently, but rather
		the total probability of the entire example.


		Parameters
		----------
		X: list, tuple, numpy.ndarray, torch.Tensor, shape=(-1, self.d)
			A set of examples to evaluate.

		priors: list, numpy.ndarray, torch.Tensor, shape=(-1, self.k)
			Prior probabilities of assigning each symbol to each node. If not
			provided, do not include in the calculations (conceptually
			equivalent to a uniform probability, but without scaling the
			probabilities). This can be used to assign labels to observatons
			by setting one of the probabilities for an observation to 1.0.
			Note that this can be used to assign hard labels, but does not
			have the same semantics for soft labels, in that it only
			influences the initial estimate of an observation being generated
			by a component, not gives a target. Default is None.


		Returns
		-------
		prob: torch.Tensor, shape=(-1,)
			The probability of each example.
		"""

		return torch.exp(self.log_probability(X, priors=priors))

	def log_probability(self, X, priors=None):
		"""Calculate the log probability of each example.

		This method calculates the log probability of each example given the
		parameters of the distribution. The examples must be given in a 2D
		format. For a Bernoulli distribution, each entry in the data must
		be either 0 or 1.

		Note: This differs from some other log probability calculation
		functions, like those in torch.distributions, because it is not
		returning the log probability of each feature independently, but rather
		the total log probability of the entire example.


		Parameters
		----------
		X: list, tuple, numpy.ndarray, torch.Tensor, shape=(-1, self.d)
			A set of examples to evaluate.

		priors: list, numpy.ndarray, torch.Tensor, shape=(-1, self.k)
			Prior probabilities of assigning each symbol to each node. If not
			provided, do not include in the calculations (conceptually
			equivalent to a uniform probability, but without scaling the
			probabilities). This can be used to assign labels to observatons
			by setting one of the probabilities for an observation to 1.0.
			Note that this can be used to assign hard labels, but does not
			have the same semantics for soft labels, in that it only
			influences the initial estimate of an observation being generated
			by a component, not gives a target. Default is None.


		Returns
		-------
		logp: torch.Tensor, shape=(-1,)
			The log probability of each example.
		"""

		e = self._emission_matrix(X, priors=priors)
		return torch.logsumexp(e, dim=1)

	def predict(self, X, priors=None):
		"""Calculate the label assignment for each example.

		This method calculates the label for each example as the most likely
		component after factoring in the prior probability.


		Parameters
		----------
		X: list, tuple, numpy.ndarray, torch.Tensor, shape=(-1, self.d)
			A set of examples to summarize.

		priors: list, numpy.ndarray, torch.Tensor, shape=(-1, self.k)
			Prior probabilities of assigning each symbol to each node. If not
			provided, do not include in the calculations (conceptually
			equivalent to a uniform probability, but without scaling the
			probabilities). This can be used to assign labels to observatons
			by setting one of the probabilities for an observation to 1.0.
			Note that this can be used to assign hard labels, but does not
			have the same semantics for soft labels, in that it only
			influences the initial estimate of an observation being generated
			by a component, not gives a target. Default is None.


		Returns
		-------
		y: torch.Tensor, shape=(-1,)
			The predicted label for each example.
		"""

		e = self._emission_matrix(X, priors=priors)
		return torch.argmax(e, dim=1)

	def predict_proba(self, X, priors=None):
		"""Calculate the posterior probabilities for each example.

		This method calculates the posterior probabilities for each example
		under each component of the model after factoring in the prior 
		probability and normalizing across all the components.


		Parameters
		----------
		X: list, tuple, numpy.ndarray, torch.Tensor, shape=(-1, self.d)
			A set of examples to summarize.

		priors: list, numpy.ndarray, torch.Tensor, shape=(-1, self.k)
			Prior probabilities of assigning each symbol to each node. If not
			provided, do not include in the calculations (conceptually
			equivalent to a uniform probability, but without scaling the
			probabilities). This can be used to assign labels to observatons
			by setting one of the probabilities for an observation to 1.0.
			Note that this can be used to assign hard labels, but does not
			have the same semantics for soft labels, in that it only
			influences the initial estimate of an observation being generated
			by a component, not gives a target. Default is None.


		Returns
		-------
		y: torch.Tensor, shape=(-1, self.k)
			The posterior probabilities for each example under each component.
		"""

		e = self._emission_matrix(X, priors=priors)
		return torch.exp(e - torch.logsumexp(e, dim=1, keepdims=True))
		
	def predict_log_proba(self, X, priors=None):
		"""Calculate the log posterior probabilities for each example.

		This method calculates the log posterior probabilities for each example
		under each component of the model after factoring in the prior 
		probability and normalizing across all the components.


		Parameters
		----------
		X: list, tuple, numpy.ndarray, torch.Tensor, shape=(-1, self.d)
			A set of examples to summarize.

		priors: list, numpy.ndarray, torch.Tensor, shape=(-1, self.k)
			Prior probabilities of assigning each symbol to each node. If not
			provided, do not include in the calculations (conceptually
			equivalent to a uniform probability, but without scaling the
			probabilities). This can be used to assign labels to observatons
			by setting one of the probabilities for an observation to 1.0.
			Note that this can be used to assign hard labels, but does not
			have the same semantics for soft labels, in that it only
			influences the initial estimate of an observation being generated
			by a component, not gives a target. Default is None.


		Returns
		-------
		y: torch.Tensor, shape=(-1, self.k)
			The log posterior probabilities for each example under each 
			component.
		"""

		e = self._emission_matrix(X, priors=priors) 
		return e - torch.logsumexp(e, dim=1, keepdims=True)

	def from_summaries(self):
		"""Update the model parameters given the extracted statistics.

		This method uses calculated statistics from calls to the `summarize`
		method to update the distribution parameters. Hyperparameters for the
		update are passed in at initialization time.

		Note: Internally, a call to `fit` is just a successive call to the
		`summarize` method followed by the `from_summaries` method.
		"""

		for d in self.distributions:
			d.from_summaries()

		if self.frozen == True:
			return

		priors = self._w_sum / torch.sum(self._w_sum)

		_update_parameter(self.priors, priors, self.inertia)
		self._reset_cache()

# base.pyx
# Contact: Jacob Schreiber ( jmschreiber91@gmail.com )

from .utils cimport *

from .distributions import Distribution

import json
import numpy
import uuid
import yaml


# Define some useful constants
DEF NEGINF = float("-inf")
DEF INF = float("inf")


cdef class Model(object):
	"""The abstract building block for all distributions."""

	def __cinit__(self):
		self.name = "Model"

	def __str__(self):
		return self.to_json()

	def __repr__(self):
		return self.to_json()

	def get_params(self, *args, **kwargs):
		return self.__getstate__()

	def set_params(self, state):
		self.__setstate__(state)

	def to_dict(self):
		"""Serialize this object to a dictionary of parameters."""
		raise NotImplementedError

	def to_json(self, separators=(',', ' : '), indent=4):
		"""Serialize the model to JSON.

		Parameters
		----------
		separators : tuple, optional
			The two separators to pass to the json.dumps function for
			formatting.
			Default is (',', ' : ').

		indent : int, optional
			The indentation to use at each level. Passed to json.dumps for
			formatting. Default is 4.

		Returns
		-------
		json : str
			A properly formatted JSON object.
		"""

		return json.dumps( self.to_dict(), separators=separators, indent=indent )

	def to_yaml(self):
		"""Serialize the model to YAML for compactness."""
		return yaml.safe_dump(json.loads(self.to_json()))

	@classmethod
	def from_dict( cls, dictionary ):
		"""Deserialize this object from a dictionary of parameters."""
		raise NotImplementedError

	@classmethod
	def from_json( cls, s ):
		"""Deserialize this object from its JSON representation.

		Parameters
		----------
		s : str
			A JSON formatted string containing the file.

		Returns
		-------
		model : object
			A properly initialized and baked model.
		"""

		# Load a dictionary from a JSON formatted string
		try:
			d = json.loads(s)
		except:
			try:
				with open(s, 'r') as infile:
					d = json.load(infile)
			except:
				raise IOError("String must be properly formatted JSON or filename of properly formatted JSON.")

		return cls.from_dict(d)

	@classmethod
	def from_yaml( cls, yml ):
		"""Deserialize this object from its YAML representation."""
		return cls.from_dict(yaml.load(yml, Loader=yaml.SafeLoader))

	def copy(self):
		"""Return a deep copy of this distribution object.

		This object will not be tied to any other distribution or connected
		in any form.

		Parameters
		----------
		None

		Returns
		-------
		distribution : Distribution
			A copy of the distribution with the same parameters.
		"""

		return self.__class__.from_json( self.to_json() )

	def freeze(self):
		"""Freeze the distribution, preventing updates from occurring."""
		self.frozen = True

	def thaw(self):
		"""Thaw the distribution, re-allowing updates to occur."""
		self.frozen = False

	def copy(self):
		"""Return a deep copy of this distribution object.

		This object will not be tied to any other distribution or connected
		in any form.

		Parameters
		----------
		None

		Returns
		-------
		distribution : Distribution
			A copy of the distribution with the same parameters.
		"""

		return self.__class__(*self.parameters)

	def sample(self, n=None):
		"""Return a random item sampled from this distribution.

		Parameters
		----------
		n : int or None, optional
			The number of samples to return. Default is None, which is to
			generate a single sample.

		Returns
		-------
		sample : double or object
			Returns a sample from the distribution of a type in the support
			of the distribution.
		"""

		raise NotImplementedError

	def probability(self, symbol):
		"""Return the probability of the given symbol under this distribution.

		Parameters
		----------
		symbol : object
			The symbol to calculate the probability of

		Returns
		-------
		probability : double
			The probability of that point under the distribution.
		"""

		return numpy.exp(self.log_probability(symbol))

	def log_probability(self, symbol):
		"""Return the log probability of the given symbol under this
		distribution.

		Parameters
		----------
		symbol : double
			The symbol to calculate the log probability of (overridden for
			DiscreteDistributions)

		Returns
		-------
		logp : double
			The log probability of that point under the distribution.
		"""

		raise NotImplementedError

	def score(self, X, y):
		"""Return the accuracy of the model on a data set.

		Parameters
		----------
		X : numpy.ndarray, shape=(n, d)
			The values of the data set

		y : numpy.ndarray, shape=(n,)
			The labels of each value
		"""

		return (self.predict(X) == y).mean()


	def sample(self, n=None):
		"""Return a random item sampled from this distribution.

		Parameters
		----------
		n : int or None, optional
			The number of samples to return. Default is None, which is to
			generate a single sample.

		Returns
		-------
		sample : double or object
			Returns a sample from the distribution of a type in the support
			of the distribution.
		"""

		raise NotImplementedError

	def fit(self, items, weights=None, inertia=0.0):
		"""Fit the distribution to new data using MLE estimates.

		Parameters
		----------
		items : array-like, shape (n_samples, n_dimensions)
			This is the data to train on. Each row is a sample, and each column
			is a dimension to train on. For univariate distributions an array
			is used, while for multivariate distributions a 2d matrix is used.

		weights : array-like, shape (n_samples,), optional
			The initial weights of each sample in the matrix. If nothing is
			passed in then each sample is assumed to be the same weight.
			Default is None.

		inertia : double, optional
			The weight of the previous parameters of the model. The new
			parameters will roughly be
			old_param * inertia + new_param * (1-inertia), so an inertia of 0
			means ignore the old parameters, whereas an inertia of 1 means
			ignore the new parameters. Default is 0.0.

		Returns
		-------
		None
		"""

		raise NotImplementedError

	def summarize(self, items, weights=None):
		"""Summarize a batch of data into sufficient statistics for a later
		update.

		Parameters
		----------
		items : array-like, shape (n_samples, n_dimensions)
			This is the data to train on. Each row is a sample, and each column
			is a dimension to train on. For univariate distributions an array
			is used, while for multivariate distributions a 2d matrix is used.

		weights : array-like, shape (n_samples,), optional
			The initial weights of each sample in the matrix. If nothing is
			passed in then each sample is assumed to be the same weight.
			Default is None.

		Returns
		-------
		None
		"""

		return NotImplementedError

	def from_summaries(self, inertia=0.0):
		"""Fit the distribution to the stored sufficient statistics.

		Parameters
		----------
		inertia : double, optional
			The weight of the previous parameters of the model. The new
			parameters will roughly be
			old_param * inertia + new_param * (1-inertia), so an inertia of 0
			means ignore the old parameters, whereas an inertia of 1 means
			ignore the new parameters. Default is 0.0.

		Returns
		-------
		None
		"""

		return NotImplementedError

	def clear_summaries(self):
		"""Clear the summary statistics stored in the object."""
		return NotImplementedError

	cdef void _log_probability(self, double* symbol, double* log_probability,
		int n) nogil:
		pass

	cdef double _vl_log_probability(self, double* symbol, int n) nogil:
		return NEGINF

	cdef double _summarize(self, double* items, double* weights, int n,
		int column_idx, int d) nogil:
		pass


cdef class GraphModel(Model):
	"""Represents an generic graphical model."""

	def __init__(self, name=None):
		"""
		Make a new graphical model. Name is an optional string used to name
		the model when output. Name may not contain spaces or newlines.
		"""

		# Save the name or make up a name.
		self.name = name or str(id(self))
		self.states = []
		self.edges = []

	def __str__(self):
		"""Represent this model with it's name and states."""
		if self.states is not None:
			state_str = "".join(map(str, self.states))
		else:
			state_str = ""

		return "{}:{}".format(self.name, state_str)

	def add_node(self, node):
		"""Add a node to the graph."""
		self.states.append(node)
		self.n_states += 1

	def add_nodes(self, *nodes):
		"""Add multiple states to the graph."""
		for node in nodes:
			self.add_node(node)

	def add_state(self, state):
		"""Another name for a node."""
		self.add_node(state)

	def add_states(self, *states):
		"""Another name for a node."""
		for state in states:
			self.add_state(state)

	def add_edge(self, a, b):
		"""
		Add a transition from state a to state b which indicates that B is
		dependent on A in ways specified by the distribution.
		"""

		# Add the transition
		self.edges.append((a, b))
		self.n_edges += 1

	def add_transition(self, a, b):
		"""Transitions and edges are the same."""
		self.add_edge(a, b)

	def node_count(self):
		"""Returns the number of nodes/states in the model"""
		return self.n_states

	def state_count(self):
		"""Returns the number of states present in the model."""
		return self.n_states

	def edge_count(self):
		"""Returns the number of edges present in the model."""
		return self.n_edges

	def dense_transition_matrix(self):
		"""
		Returns the dense transition matrix. Useful if the transitions of
		somewhat small models need to be analyzed.
		"""

		m = len(self.states)
		transition_log_probabilities = numpy.full((m, m), NEGINF)

		for i in range(m):
			for n in range( self.out_edge_count[i], self.out_edge_count[i+1] ):
				transition_log_probabilities[i, self.out_transitions[n]] = \
					self.out_transition_log_probabilities[n]

		return transition_log_probabilities


cdef class State(object):
	"""
	Represents a state in an HMM. Holds emission distribution, but not
	transition distribution, because that's stored in the graph edges.
	"""

	def __init__( self, distribution, name=None, weight=None ):
		"""
		Make a new State emitting from the given distribution. If distribution
		is None, this state does not emit anything. A name, if specified, will
		be the state's name when presented in output. Name may not contain
		spaces or newlines, and must be unique within a model.
		"""

		# Save the distribution
		self.distribution = distribution

		# Save the name
		self.name = name or str(uuid.uuid4())

		# Save the weight, or default to the unit weight
		self.weight = weight or 1.

	def __reduce__(self):
		return self.__class__, (self.distribution, self.name, self.weight)

	def __str__(self):
		"""
		The string representation of a state is the json, so call that format.
		"""
		return self.to_json()

	def __repr__(self):
		"""
		The string representation of a state is the json, so call that format.
		"""
		return self.__str__()

	def tie( self, state ):
		"""
		Tie this state to another state by just setting the distribution of the
		other state to point to this states distribution.
		"""
		state.distribution = self.distribution

	def is_silent(self):
		"""
		Return True if this state is silent (distribution is None) and False
		otherwise.
		"""
		return self.distribution is None

	def tied_copy(self):
		"""
		Return a copy of this state where the distribution is tied to the
		distribution of this state.
		"""
		return State( distribution=self.distribution, name=self.name+'-tied' )

	def copy( self ):
		"""Return a hard copy of this state."""
		return State( distribution=self.distribution.copy(), name=self.name )

	def to_dict(self):
		"""Convert this state to a dictionary of parameters."""
		return {
			'class' : 'State',
			'distribution' : None if self.is_silent() else self.distribution.to_dict(),
			'name' : self.name,
			'weight' : self.weight
		}

	def to_json(self, separators=(',', ' : '), indent=4):
		"""Convert this state to JSON format."""
		return json.dumps(self.to_dict(), separators=separators, indent=indent)

	def to_yaml(self):
		"""Convert this state to YAML format."""
		return yaml.safe_dump(json.loads(self.to_json()))

	@classmethod
	def from_dict( cls, d ):
		"""Read a State from a given dictionary of parameters."""

		# If we're not decoding a state, we're decoding the wrong thing
		if d['class'] != 'State':
			raise IOError( "State object attempting to decode "
			               "{} object".format( d['class'] ) )

		# If this is a silent state, don't decode the distribution
		if d['distribution'] is None:
			return cls( None, str(d['name']), d['weight'] )

		# Otherwise it has a distribution, so decode that
		name = str(d['name'])
		weight = d['weight']
		from .gmm import GeneralMixtureModel

		c = d['distribution']['class']
		dist = eval(c).from_dict( d['distribution'] )
		return cls( dist, name, weight )

	@classmethod
	def from_json( cls, s ):
		"""Read a State from a given string formatted in JSON."""

		# Load a dictionary from a JSON formatted string
		return cls.from_dict(json.loads(s))

	@classmethod
	def from_yaml( cls, yml ):
		"""Read a State from a given string formatted in YAML."""
		return cls.from_dict(yaml.load(yml, Loader=yaml.SafeLoader))

# Create a convenient alias
Node = State

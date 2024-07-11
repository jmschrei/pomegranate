# _utils.py
# Jacob Schreiber <jmschreiber91@gmail.com>

import numpy
import torch
import collections

from apricot import FacilityLocationSelection
from apricot import FeatureBasedSelection


eps = torch.finfo(torch.float32).eps

class BufferList(torch.nn.Module):
	"""A buffer list."""

	def __init__(self, buffers):
		super(BufferList, self).__init__()

		self.buffers = []
		for i, b in enumerate(buffers):
			name = "_buffer_{}".format(i)
			
			self.register_buffer(name, b)
			self.buffers.append(getattr(self, name))

	def __repr__(self):
		return str(self.buffers)

	def __getitem__(self, i):
		return self.buffers[i]

	@property
	def dtype(self):
		return self.buffers[0].dtype


def _inplace_add(X, Y):
	"""Do an in-place addition on X accounting for if Y is a masked tensor."""

	if isinstance(Y, torch.masked.MaskedTensor):
		X += Y._masked_data
	else:
		X += Y

def _cast_as_tensor(value, dtype=None):
	"""Set the parameter.""" 

	if value is None:
		return None

	if isinstance(value, (torch.nn.Parameter, torch.Tensor, torch.masked.MaskedTensor)):
		if dtype is None:
			return value
		elif value.dtype == dtype:
			return value
		else:
			return value.type(dtype)
			
	if isinstance(value, list) and all(isinstance(v, numpy.ndarray) for v in value):
		value = numpy.array(value)
		
	if isinstance(value, (float, int, list, tuple, numpy.ndarray)):
		if dtype is None:
			return torch.tensor(value)
		else:
			return torch.tensor(value, dtype=dtype)


def _cast_as_parameter(value, dtype=None, requires_grad=False):
	"""Set the parameter.""" 

	if value is None:
		return None

	value = _cast_as_tensor(value, dtype=dtype)
	return torch.nn.Parameter(value, requires_grad=requires_grad)


def _update_parameter(value, new_value, inertia=0.0, frozen=None):
	"""Update a parameter unles.
	"""

	if hasattr(value, "frozen") and getattr(value, "frozen") == True:
		return

	if inertia == 0.0:
		value[...] = _cast_as_parameter(new_value)

	elif inertia < 1.0:
		value_ = inertia*value + (1-inertia)*new_value

		inf_idx = torch.isinf(value)
		inf_idx_new = torch.isinf(new_value)

		value_[inf_idx] = value[inf_idx].type(value_.dtype)
		value_[inf_idx_new] = new_value[inf_idx_new].type(value_.dtype)
		
		value[:] = _cast_as_parameter(value_)


def _check_parameter(parameter, name, min_value=None, max_value=None, 
	value_sum=None, value_sum_dim=None, value_set=None, dtypes=None, ndim=None, 
	shape=None, check_parameter=True, epsilon=1e-6):
	"""Ensures that the parameter falls within a valid range.

	This check accepts several optional conditions that can be used to ensure
	that the value has the desired properties. If these conditions are set to
	`None`, they are not checked. 

	Note: `parameter` can be a single value or it can be a whole tensor/array
	of values. These checks are run either against the entire tensor, e.g.
	ndims, or against each of the values in the parameter.

	
	Parameters
	----------
	parameter: anything
		The parameter meant to be checked

	name: str
		The name of the parameter for error logging purposes.

	min_value: float or None, optional
		The minimum numeric value that any values in the parameter can take.
		Default is None.

	max_value: float or None, optional
		The maximum numeric value that any values in the parameter can take.
		Default is None.

	value_sum: float or None, optional
		The approximate sum, within eps, of the parameter. Default is None.

	value_sum_dim: int or float, optional
		What dimension to sum over. If None, sum over entire tensor. Default
		is None.

	value_set: tuple or list or set or None, optional
		The set of values that each element in the parameter can take. Default
		is None.

	dtypes: tuple or list or set or None, optional
		The set of dtypes that the parameter can take. Default is None.

	ndim: int or list or tuple or None, optional
		The number of dimensions of the tensor. Should not be used when the
		parameter is a single value. Default is None.

	shape: tuple or None, optional
		The shape of the parameter. -1 can be used to accept any value for that
		dimension.

	epsilon: float, optional
		When using `value_sum`, this is the maximum difference acceptable.
		Default is 1e-6.
	"""

	vector = (numpy.ndarray, torch.Tensor, torch.nn.Parameter)

	if parameter is None:
		return None

	if check_parameter == False:
		return parameter

	if dtypes is not None:
		if isinstance(parameter, vector):
			if parameter.dtype not in dtypes:
				raise ValueError("Parameter {} dtype must be one of {}".format(
					name, dtypes))
		else:
			if type(parameter) not in dtypes:
				raise ValueError("Parameter {} dtype must be one of {}".format(
					name, dtypes))


	if min_value is not None:
		if isinstance(parameter, vector):
			if (parameter < min_value).sum() > 0:
				raise ValueError("Parameter {} must have a minimum value above"
					" {}".format(name, min_value))
		else:
			if parameter < min_value:
				raise ValueError("Parameter {} must have a minimum value above"
					" {}".format(name, min_value))


	if max_value is not None:
		if isinstance(parameter, vector):
			if (parameter > max_value).sum() > 0:
				raise ValueError("Parameter {} must have a maximum value below"
					" {}".format(name, max_value))
		else:
			if parameter > max_value:
				raise ValueError("Parameter {} must have a maximum value below"
					" {}".format(name, max_value))


	if value_sum is not None:
		if isinstance(parameter, vector):
			if value_sum_dim is None:
				delta = torch.sum(parameter) - value_sum
			else:
				delta = torch.sum(parameter, dim=value_sum_dim) - value_sum

			if torch.any(torch.abs(delta) > epsilon):
				raise ValueError("Parameter {} must sum to {}".format(name, 
					value_sum))
		else:
			if abs(parameter - value_sum) > epsilon:
				raise ValueError("Parameter {} must sum to {}".format(name, 
					value_sum))


	if value_set is not None:
		if isinstance(parameter, vector):
			if (~numpy.isin(parameter, value_set)).sum() > 0:
				raise ValueError("Parameter {} must contain values in set"
					" {}".format(name, value_set))
		else:
			if parameter not in value_set:
				raise ValueError("Parameter {} must contain values in set"
					" {}".format(name, value_set))


	if ndim is not None:
		if isinstance(parameter, vector):
			if isinstance(ndim, int):
				if len(parameter.shape) != ndim:
					raise ValueError("Parameter {} must have {} dims".format(
						name, ndim))
			else:
				if len(parameter.shape) not in ndim:
					raise ValueError("Parameter {} must have {} dims".format(
						name, ndim))
		else:
			if ndim != 0:
				raise ValueError("Parameter {} must have {} dims".format(
					name, ndim))

	if shape is not None:
		if isinstance(parameter, vector):
			if len(parameter.shape) != len(shape):
				raise ValueError("Parameter {} must have shape {}".format(
					name, shape))

			for i in range(len(shape)):
				if shape[i] != -1 and shape[i] != parameter.shape[i]:
					raise ValueError("Parameter {} must have shape {}".format(
						name, shape))

	return parameter


def _check_shapes(parameters, names):
	"""Check the shapes of a set of parameters.

	This function takes in a set of parameters, as well as their names, and
	checks that the shape is correct. It will raise an error if the lengths
	of the parameters without the value of None are not equal.


	Parameters
	----------
	parameters: list or tuple
		A set of parameters, which can be None, to check the shape of.

	names: list or tuple
		A set of parameter names to refer to if something is wrong.
	"""

	n = len(parameters)

	for i in range(n):
		for j in range(n):
			if parameters[i] is None:
				continue

			if parameters[j] is None:
				continue

			n1, n2 = names[i], names[j]
			if len(parameters[i]) != len(parameters[j]):
				raise ValueError("Parameters {} and {} must be the same "
					"shape.".format(names[i], names[j]))


def _reshape_weights(X, sample_weight, device='cpu'):
	"""Handle a sample weight tensor by creating and reshaping it.

	This function will take any weight input, including None, 1D weights, and
	2D weights, and shape it into a 2D matrix with the same shape as the data
	X also passed in.

	Both elements must be PyTorch tensors.


	Parameters
	----------
	X: torch.tensor, ndims=2
		The data being weighted. The contents of this tensor are not used, only
		the shape is.

	sample_weight: torch.tensor or None
		The weight for each element in the data or None.


	Returns
	-------
	sample_weight: torch.tensor, shape=X.shape
		A tensor with the same dimensions as X with elements repeated as
		necessary.
	"""

	if sample_weight is None:
		if torch.is_floating_point(X):
			sample_weight = torch.ones(1, device=device, 
				dtype=X.dtype).expand(*X.shape)
		else:
			sample_weight = torch.ones(1, device=device, 
				dtype=torch.float32).expand(*X.shape)
	else:
		if not torch.is_floating_point(sample_weight):
			sample_weight = sample_weight.type(torch.float32)


	if len(sample_weight.shape) == 1: 
		sample_weight = sample_weight.reshape(-1, 1).expand(-1, X.shape[1])
		_check_parameter(sample_weight, "sample_weight", min_value=0)

	elif sample_weight.shape[1] == 1:
		sample_weight = sample_weight.expand(-1, X.shape[1])
		_check_parameter(sample_weight, "sample_weight", min_value=0)

	if isinstance(X, torch.masked.MaskedTensor):
		if not isinstance(sample_weight, torch.masked.MaskedTensor):
			sample_weight = torch.masked.MaskedTensor(sample_weight, 
				mask=X._masked_mask)

	_check_parameter(sample_weight, "sample_weight", shape=X.shape, 
		ndim=X.ndim)
	return sample_weight


def _initialize_centroids(X, k, algorithm='first-k', random_state=None):
	if isinstance(k, torch.Tensor):
		k = k.item()
		
	if not isinstance(random_state, numpy.random.mtrand.RandomState):
		random_state = numpy.random.RandomState(random_state)

	if algorithm == 'first-k':
		return _cast_as_tensor(torch.clone(X[:k]), dtype=torch.float32)

	elif algorithm == 'random':
		idxs = random_state.choice(len(X), size=k, replace=False)
		return _cast_as_tensor(torch.clone(X[idxs]), dtype=torch.float32)

	elif algorithm == 'submodular-facility-location':
		selector = FacilityLocationSelection(k, random_state=random_state)
		return _cast_as_tensor(selector.fit_transform(X), dtype=torch.float32)

	elif algorithm == 'submodular-feature-based':
		selector = FeatureBasedSelection(k, random_state=random_state)
		return selector.fit_transform(X)


def partition_sequences(X, sample_weight=None, priors=None, n_dists=None):
	"""Partition a set of sequences into blobs of equal length.

	This function will take in a list of sequences, where each sequence is
	a different length, and group together sequences of the same length so that
	batched operations can be more efficiently done on them. 

	Alternatively, it can take in sequences in the correct format and simply 
	return them. The correct form is to be either a single tensor that has
	three dimensions or a list of three dimensional tensors, where each
	tensor contains all the sequences of the same length.

	More concretely, the input to this function can be:

		- A single 3D tensor, or blob of data that can be cast into a 3D
		tensor, which gets returned

		- A list of 3D tensors, or elements that can be cast into 3D tensors, 
		which also gets returned

		- A list of 2D tensors, which get partitioned into a list of 3D tensors


	Parameters
	----------
	X: list or tensor
		The input sequences to be partitioned if in an incorrect format.

	sample_weight: list or tensor or None, optional
		The input sequence weights for the sequences or None. If None, return
		None. Default is None.

	priors: list or tensor or None
		The input sequence priors for the sequences or None. If None, return
		None. Default is None.

	n_dists: int or None
		The expected last dimension of the priors tensor. Must be provided if
		`priors` is provided. Default is None.


	Returns
	-------
	X_: list or tensor
		The partitioned and grouped sequences.
	"""

	if priors is not None and n_dists is None:
		raise RuntimeError("If priors are provided, n_dists must be provided as well.")
	
	# If a 3D tensor has been passed in, return it
	try:
		X = [_check_parameter(_cast_as_tensor(X), "X", ndim=3)]
	except:
		pass
	else:
		if sample_weight is not None:
			sample_weight = [_check_parameter(_cast_as_tensor(sample_weight),
				"sample_weight", min_value=0.0).to(X[0].device)]

		if priors is not None:
			priors = [_check_parameter(_cast_as_tensor(priors), "priors",
				ndim=3, shape=(*X[0].shape[:-1], n_dists)).to(X[0].device)]

		return X, sample_weight, priors

	# Otherwise, cast all elements in the list as a tensor
	X = [_cast_as_tensor(x) for x in X]

	# If a list of 3D tensors has been passed in, return it
	try:
		X = [_check_parameter(x, "X", ndim=3) for x in X]
	except:
		pass
	else:
		if sample_weight is not None:
			sample_weight = [_check_parameter(_cast_as_tensor(w_),
				"sample_weight", min_value=0.0).to(X[0].device) 
					for w_ in sample_weight]

		if priors is not None:
			priors = [_check_parameter(_cast_as_tensor(p), "priors",
				ndim=3, shape=(*X[i].shape[:-1], n_dists)).to(X[0].device) 
					for i, p in enumerate(priors)]

		if all([x.ndim == 3 for x in X]):
			return X, sample_weight, priors

	# Otherwise, group together same-sized examples
	X_dict = collections.defaultdict(list)
	sample_weight_dict = collections.defaultdict(list)
	priors_dict = collections.defaultdict(list)

	for i, x in enumerate(X):
		x = _check_parameter(x, "X", ndim=2)
		n = len(x)

		X_dict[n].append(x)

		if sample_weight is not None:
			w = _check_parameter(_cast_as_tensor(sample_weight[i]), 
				"sample_weight", min_value=0.0).to(X[0].device)

			sample_weight_dict[n].append(w)

		if priors is not None:
			p = _check_parameter(_cast_as_tensor(priors[i]), "priors",
				ndim=2, shape=(*x.shape[:-1], n_dists)).to(X[0].device)

			priors_dict[n].append(p)

	keys = sorted(X_dict.keys())
	
	X_ = [torch.stack(X_dict[key]) for key in keys]

	if sample_weight is None:
		sample_weight_ = None
	else:
		sample_weight_ = [torch.stack(sample_weight_dict[key]) for key in keys]

	if priors is None:
		priors_ = None
	else:
		priors_ = [torch.stack(priors_dict[key]) for key in keys]

	return X_, sample_weight_, priors_

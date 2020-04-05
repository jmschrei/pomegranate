import numpy

try:
	import pandas
except:
	pandas = None

class BaseGenerator(object):
	"""The base data generator class.

	This object is inherited by data generator objects in order to specify that
	they are data generators. Do not use this object directly.
	"""
	
	def __init__(self):
		pass

	def __len__(self):
		return NotImplementedError

	@property
	def shape(self):
		return NotImplementedError

	@property
	def classes(self):
		return NotImplementedError

	@property
	def ndim(self):
		return NotImplementedError

class DataGenerator(BaseGenerator):
	"""A generator that returns batches of a data set.

	This object will wrap a data set and optionally a set of labels and will
	return batches of data as requested. When it reaches the end of a data
	set it will not roll over but rather return a batch of data smaller
	than the other batches.

	Parameters
	----------
	X : numpy.ndarray or list
		The data set to iterate over.

	weights : numpy.ndarray or list or None, optional
		The weights for each example. Default is None.

	y: numpy.ndarray or list or None, optional
		The set of labels for each example in the data set. Default is None.

	batch_size : int or None, optional
		The size of the batches to return. If None will return the full data 
		set each time. Default is None

	batches_per_epoch : int or None, optional
		The number of batches to return before resetting the index. If the
		value is too low you may not see all examples from the data set. If
		None, will return enough batches to cover the entire data set. Default
		is None.
	"""
	
	def __init__(self, X, weights=None, y=None, batch_size=None,
			batches_per_epoch=None):
		self.X = numpy.array(X)
		self.y = y
		self.idx = 0

		if y is not None and len(y) != len(X):
			raise ValueError("Size of label vector y does not match size of data.")

		if weights is None:
			self.weights = numpy.ones(len(X), dtype='float64')
		else:
			if len(weights) != len(X):
				raise ValueError("Size of weight vector does not match size of data.")

			self.weights = numpy.array(weights)

		if batch_size is None:
			self.batch_size = len(self)
		else:
			self.batch_size = int(batch_size)

		if batches_per_epoch is None:
			self.batches_per_epoch = float("inf")
		else:
			self.batches_per_epoch = batches_per_epoch

	def __len__(self):
		return len(self.X)

	@property
	def shape(self):
		return self.X.shape

	@property
	def classes(self):
		if self.y is None:
			raise ValueError("Classes cannot be found on an unlabeled data set.")

		return numpy.unique(self.y)

	@property
	def ndim(self):
		return self.X.ndim

	def batches(self):
		if self.batch_size == len(self):
			while True:
				if self.y is not None:
					yield self.X, self.y, self.weights
				else:
					yield self.X, self.weights			
				break
		else:
			start, end = 0, self.batch_size
			iteration = 0

			while start < len(self) and iteration < self.batches_per_epoch:
				if self.y is not None:
					yield (self.X[start:end], self.y[start:end], 
						self.weights[start:end])
				else:
					yield self.X[start:end], self.weights[start:end]

				start += self.batch_size
				end += self.batch_size
				iteration += 1

	def labeled_batches(self):
		X = self.X[self.y != -1]
		weights = self.weights[self.y != -1]
		y = self.y[self.y != -1]

		start, end = 0, self.batch_size
		while start < len(X):
			yield X[start:end], y[start:end], weights[start:end]
			
			start += self.batch_size
			end += self.batch_size 

	def unlabeled_batches(self):
		X = self.X[self.y == -1]
		weights = self.weights[self.y == -1]

		start, end = 0, self.batch_size
		while start < len(X):
			yield X[start:end], weights[start:end]
			
			start += self.batch_size
			end += self.batch_size

class SequenceGenerator(BaseGenerator):
	"""A generator that returns batches of sequences from a data set.

	This object will wrap a data set and optionally a set of labels and will
	return sequences as requested. Due to the processing in pomegranate, only
	batches of size 1 are supported.

	Parameters
	----------
	X : numpy.ndarray or list
		The data set to iterate over.

	weights : numpy.ndarray or list or None, optional
		The weights for each example. Default is None.

	y: numpy.ndarray or list or None, optional
		The set of labels for each example in the data set. Default is None.
	"""
	
	def __init__(self, X, weights=None, y=None, batches_per_epoch=None):
		self.X = X
		self.y = y
		self.idx = 0

		if weights is None:
			self.weights = numpy.ones(len(X), dtype='float64')
		else:
			self.weights = weights

		if batches_per_epoch is None:
			self.batches_per_epoch = float("inf")
		else:
			self.batches_per_epoch = batches_per_epoch

	def __len__(self):
		return len(self.X)

	@property
	def shape(self):
		x_ = numpy.array(self.X[0])

		if x_.ndim == 1:
			return len(self.X), 1
		elif x_.ndim == 2:
			return len(self.X), x_.shape[1]
		else:
			raise ValueError("Data must be passed in as a list of numpy arrays.")

	@property
	def ndim(self):
		return len(self.X[0])

	@property
	def classes(self):
		if self.y is None:
			raise ValueError("No labels found for this data set.")

		return numpy.unique(self.y)

	def batches(self):
		for idx in range(len(self)):
			if self.y is not None:
				yield self.X[idx:idx+1], self.weights[idx:idx+1], self.y[idx:idx+1]
			else:
				yield self.X[idx:idx+1], self.weights[idx:idx+1]

	def labeled_batches(self):
		X = [x for x, y in zip(self.X, self.y) if y is not None]
		weights = [w for w, y in zip(self.weights, self.y) if y is not None]
		y = [y for y in self.y if y is not None]

		for idx in range(len(X)):
			yield X[idx:idx+1], weights[idx:idx+1], y[idx:idx+1]

	def unlabeled_batches(self):
		X = [x for x, y in zip(self.X, self.y) if y is None]
		weights = [w for w, y in zip(self.weights, self.y) if y is None]

		for idx in range(len(X)):
			yield X[idx:idx+1], weights[idx:idx+1]

class DataFrameGenerator(BaseGenerator):
	"""A generator that returns batches of sequences from a DataFrame.

	This object will wrap a DataFrame and generate batches of data
	from it. This gives a natural support for pandas DataFrames to
	all pomegranate models. Weights and labels for examples can either
	be passed in separately (potentially either as Series or array
	objects) or specified as a column in the main DataFrame object.

	Parameters
	----------
	X : pandas.DataFrame
		The DataFrame containing data. The weights and labels (if used)
		can be columns in this DataFrame.

	weights : pandas.Series, numpy.ndarray, str, or None, optional
		The weights for each example. This can either be a series, a numpy
		array, a string key for the column of the DataFrame, or None. None
		means that all weights are uniform. Default is None.

	y : pandas.Series, numpy.ndarray, str, or None, optional
		The labels for each example. This can either be a series, a numpy
		array, a string key for the column of the DataFrame, or None. None
		means that all examples are unlabeled. Default is None.

	batch_size : int or None, optional
		The size of the batches to return. If None will return the full data 
		set each time. Default is None

	batches_per_epoch : int or None, optional
		The number of batches to return before resetting the index. If the
		value is too low you may not see all examples from the data set. If
		None, will return enough batches to cover the entire data set. Default
		is None.
	"""
	
	def __init__(self, X, weights=None, y=None, batch_size=None, 
		batches_per_epoch=None):
		if pandas is None:
			raise ValueError("Must have pandas installed to use DataFrameGenerator.")

		self.X = X

		if isinstance(weights, pandas.Series):
			self.weights = weights.values
		elif isinstance(weights, numpy.ndarray):
			self.weights = weights
		elif isinstance(weights, str):
			self.weights = self.X[weights].values
			self.X = self.X.drop(weights, axis=1)
		else:
			self.weights = numpy.ones(self.X.shape[0])

		if isinstance(y, pandas.Series):
			self.y = y.values
		elif isinstance(y, numpy.ndarray):
			self.y = y
		elif isinstance(y, str):
			self.y = self.X[y].values
			self.X = self.X.drop(y, axis=1)
		else:
			self.y = None

		if len(self.weights) != len(self.X):
			raise ValueError("Size of weights does not match size of data set.")
		if self.y is not None and len(self.y) != len(self.X):
			raise ValueError("Size of labels does not match size of data set.")

		self.batch_size = batch_size or len(self.X)

		if batches_per_epoch is None:
			self.batches_per_epoch = float("inf")
		else:
			self.batches_per_epoch = batches_per_epoch

	def __len__(self):
		return len(self.X)

	@property
	def shape(self):
		return self.X.shape

	@property
	def classes(self):
		if self.y is None:
			raise ValueError("Must specify y to return classes.")

		return numpy.unique(self.y)

	@property
	def ndim(self):
		return self.X.ndim

	def batches(self):
		if self.batch_size == len(self):
			while True:
				if self.y is not None:
					yield self.X.values, self.y, self.weights
				else:
					yield self.X.values, self.weights			
				break
		else:
			start, end = 0, self.batch_size
			iteration = 0

			while start < len(self) and iteration < self.batches_per_epoch:
				x = self.X.iloc[start:end].values

				if self.y is not None:
					yield x, self.y[start:end], self.weights[start:end]
				else:
					yield x, self.weights[start:end]

				start += self.batch_size
				end += self.batch_size
				iteration += 1

	def labeled_batches(self):
		X = self.X.iloc[self.y != -1]
		weights = self.weights[self.y != -1]
		y = self.y[self.y != -1]

		start, end = 0, self.batch_size
		while start < len(X):
			x_ = X.iloc[start:end].values
			yield x_, y[start:end], weights[start:end]
			
			start += self.batch_size
			end += self.batch_size 

	def unlabeled_batches(self):
		X = self.X.iloc[self.y == -1]
		weights = self.weights[self.y == -1]

		start, end = 0, self.batch_size
		while start < len(X):
			x_ = X.iloc[start:end].values
			yield x_, weights[start:end]
			
			start += self.batch_size
			end += self.batch_size

class CSVGenerator(BaseGenerator):
	"""A generator that returns batches of sequences from a data file.

	This object will wrap a file, such as a CSV file, and generate batches
	of data from it. It will not load the entire file into memory except
	for particular model methods that force it to do so. It is mostly a
	wrapper around a call to `pandas.read_csv`.

	Parameters
	----------
	filename : str
		The name of the file to open.

	weight_column : str or int or None, optional 
		The column to use for the weights. If None, assume uniform weights.

	y_column: str or int or None, optional
		The column to use for the labels. If None, assume no labels.

	kwargs : keyword arguments, optional
		Any other argument to pass into `pandas.read_csv`.
	"""
	
	def __init__(self, filename, weight_column=None, y_column=None, 
		batch_size=32, **kwargs):
		self.filename = filename
		self.weight_column = weight_column
		self.y_column = y_column
		self.kwargs = kwargs
		self.file = pandas.read_csv(filename, iterator=True, 
			chunksize=batch_size, **kwargs)

	def __len__(self):
		return len(self.X)

	@property
	def shape(self):
		raise ValueError("Cannot get shape of a file.")

	@property
	def classes(self):
		if self.y_column is None:
			raise ValueError("Must specify y_column to return classes.")

		return numpy.unique(numpy.concatenate([numpy.unique(
			batch[self.y_column]) for batch in self.batches]))

	def batches(self):
		for batch in self.file:
			if self.weight_column is not None:
				weights = batch[self.weight_column].values.astype('float64')
				X = batch.drop(self.weight_column)
			else:
				weights = numpy.ones(batch.shape[0], dtype='float64')

			if self.y_column is not None:
				y = X[self.y_column].values.astype('float64')
				X = X.drop(self.y_column).values.astype('float64')
				yield X, weights, y
			else:
				X = X.values.astype('float64')
				yield X, weights


	def labeled_batches(self):
		X = [x for x, y in zip(self.X, self.y) if y is not None]
		weights = [w for w, y in zip(self.weights, self.y) if y is not None]
		y = [y for y in self.y if y is not None]

		for idx in range(len(X)):
			yield X[idx:idx+1], weights[idx:idx+1], y[idx:idx+1]

	def unlabeled_batches(self):
		X = [x for x, y in zip(self.X, self.y) if y is None]
		weights = [w for w, y in zip(self.weights, self.y) if y is None]

		for idx in range(len(X)):
			yield X[idx:idx+1], weights[idx:idx+1]

# callbacks.py
# Authors: Jacob Schreiber <jmschreiber91@gmail.com>

class Callback(object):
	"""An object that adds functionality during training.

	A callback is a function or group of functions that can be executed during
	the training process for any of pomegranate's models that have iterative
	training procedures. A callback can be called at three stages-- the
	beginning of training, at the end of each epoch (or iteration), and at
	the end of training. Users can define any functions that they wish in
	the corresponding functions.
	"""

	def __init__(self):
		self.model = None
		self.params = None

	def on_training_begin(self):
		"""Functionality to add to the beginning of training.

		This method will be called at the beginning of each model's training
		procedure.
		"""

		pass

	def on_training_end(self, logs):
		"""Functionality to add to the end of training.

		This method will be called at the end of each model's training
		procedure.
		"""

		pass

	def on_epoch_end(self, logs):
		"""Functionality to add to the end of each epoch.

		This method will be called at the end of each epoch during the model's
		iterative training procedure.
		"""

		pass


class ModelCheckpoint(Callback):
	"""This will save the model to disk after each epoch."""

	def __init__(self, name=None, verbose=True):
		self.model = None
		self.params = None
		self.name = None
		self.verbose = verbose

	def on_epoch_end(self, logs):
		"""Save the model to disk at the end of each epoch."""

		model = self.model.to_json()
		epoch = logs['epoch']
		name = self.name if self.name is not None else self.model.name

		if self.verbose:
			print("[{}] Saving checkpoint to {}.{}.json".format(epoch, name, epoch))

		with open('{}.{}.json'.format(name, epoch), 'w') as outfile:
			outfile.write(model)


class History(Callback):
	"""Keeps a history of the loss during training."""

	def on_training_begin(self):
		self.total_improvement = []
		self.improvements = []
		self.log_probabilities = []
		self.epoch_start_times = []
		self.epoch_end_times = []
		self.epoch_durations = []
		self.epochs = []
		self.learning_rates = []
		self.n_seen_batches = []
		self.initial_log_probablity = None

	def on_epoch_end(self, logs):
		"""Save the files to the appropriate lists."""

		self.total_improvement.append(logs['total_improvement'])
		self.improvements.append(logs['improvement'])
		self.log_probabilities.append(logs['log_probability'])
		self.epoch_start_times.append(logs['epoch_start_time'])
		self.epoch_end_times.append(logs['epoch_end_time'])
		self.epoch_durations.append(logs['duration'])
		self.epochs.append(logs['epoch'])
		self.learning_rates.append(logs['learning_rate'])
		self.n_seen_batches.append(logs['n_seen_batches'])
		self.initial_log_probability = logs['initial_log_probability']


class CSVLogger(Callback):
	"""Logs results of training to a CSV file during training."""

	def __init__(self, filename, separator=',', append=False):
		self.filename = filename
		self.separator = separator
		self.append = append
		self.file = None
		self.columns = ['epoch', 'duration', 'total_improvement', 'improvement',
			'log_probability', 'last_log_probability', 'epoch_start_time',
			'epoch_end_time', 'n_seen_batches', 'learning_rate']

	def on_training_begin(self):
		if self.append == False:
			self.file = open(self.filename, 'w')
			self.file.write(self.separator.join(self.columns) + "\n")
		else:
			self.file = open(self.filename, 'a')


	def on_training_end(self, logs):
		self.file.close()

	def on_epoch_end(self, logs):
		self.file.write(self.separator.join(str(logs[col]) for col in self.columns) + "\n")

class LambdaCallback(Callback):
	"""A callback that takes in anonymous functions for any of the methods, for convenience."""

	def __init__(self, on_training_begin=None, on_training_end=None, on_epoch_end=None):
		self.on_training_begin_ = on_training_begin
		self.on_training_end_ = on_training_end
		self.on_epoch_end_ = on_epoch_end

	def on_training_begin(self):
		if self.on_training_begin_ is not None:
			self.on_training_begin_()

	def on_training_end(self, logs):
		if self.on_training_end_ is not None:
			self.on_training_end_(logs)

	def on_epoch_end(self, logs):
		if self.on_epoch_end_ is not None:
			self.on_epoch_end_(logs)

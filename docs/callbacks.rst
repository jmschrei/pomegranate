.. _callbacks:

Callbacks
=========

- `IPython Notebook Tutorial <https://github.com/jmschrei/pomegranate/blob/master/tutorials/C_Feature_Tutorial_8_Callbacks.ipynb>`_

Callback refer to functions that should be executing during the training procedure. These functions can be executed either at the start of training, the end of each epoch, or at the end of training. They mirror in style the callbacks from keras, and so are passed in using the `callbacks` keyword in `fit` and `from_sample` methods.

In pomegranate, a callback is an object that inherits from the `pomegranate.callbacks.Callback` object and has the following three methods implemented or inherited:

* `on_training_begin(self)` : What should happen when training begins.
* `on_epoch_end(self, logs)` : What should happen at the end of an epoch. The model will pass a dictionary of logs to each callback with each call that includes summary information about the training. The logs file is described more in depth below.
* `on_training_end(self, logs)` : What should happen when training ends. The final set of logs is passed in as well.

The log dictionary that is returned has the following entries:

 - `epoch` : `int`, the iteration or epoch that the model is currently on
 - `improvement` : `float`, the improvement since the latest iteration in the training set log probability
 - `total_improvement` : `float`, the total improvement seen in the training set log probability since the beginning of training
 - `log_probability` : `float`, the log probability of the training set after this round of training
 - `last_log_probability` : `float`, the log probability of the training set before this round of training
 - `duration` : `float`, the time in seconds that this epoch took
 - `epoch_start_time` : the time accoding to `time.time()` that this epoch began
 - `epoch_end_time`: the time according to `time.time()` that this epoch eded
 - `n_seen_batches` : `int`, the number of batches that have been seen by the model, only useful for mini-batching
 - `learning_rate` : The learning rate. This is undefined except when a decaying learning rate is set. 

The following callbacks are built in to pomegranate:

1. ``History()``: This will keep track of the above values in respective lists, e.g., `history.epochs` and `history.improvements`. This callback is automatically run by all models, and is returned when `return_history=True` is passed in.

.. code-block:: python

	from pomegranate.callbacks import History
	from pomegranate import *

	model = HiddenMarkovModel.from_samples(X) # No history returned
	model, history = HiddenMarkovModel.from_samples(X, return_history=True)


2. ``ModelCheckpoint(name=None, verbose=True)``: This callback will save the model parameters to a file named `{name}.{epoch}.json` at the end of each epoch. By default the name is the name of the model, but that can be overriden with the name passed in to the callback object. The verbosity flag indicates if it should print a message to the screen indicating that a file was saved, and where to, at the end of each epoch.

.. code-block:: python

	>>> from pomegranate.callbacks import ModelCheckpoint
	>>> from pomegranate import *
	>>> HiddenMarkovModel.from_samples(X, callbacks=[ModelCheckpoint()])

3. ``CSVLogger(filename, separator=',', append=False)``: This callback will save the statistics from the logs dictionary to rows in a file at the end of each epoch. The filename specifies where to save the logs to, the separator is the symbol to separate values, and append indicates whether to save to the end of a file or to overwrite it, if it currently exists.

.. code-block:: python

	>>> from pomegranate.callbacks import CSVLogger, ModelCheckpoint
	>>> from pomegranate import *
	>>> HiddenMarkovModel.from_samples(X, callbacks=[CSVLogger('model.logs'), ModelCheckpoint()])

4. ``LambdaCallback(on_training_begin=None, on_training_end=None, on_epoch_end=None)``: A convenient wrapper that allows you to pass functions in that get executed at the appropriate points. The function `on_epoch_end` and `on_training_end` should accept a single argument, the dictionary of logs, as described above.

.. code-block:: python

	>>> from pomegranate.callbacks import LambdaCheckpoint
	>>> from pomegranate import *
	>>> 
	>>> def on_training_end(logs):
	>>> 	print("Total Improvement: {:4.4}".format(logs['total_improvement']))
	>>> 
	>>> HiddenMarkovModel.from_samples(X, callbacks=[LambdaCheckpoint(on_training_end=on_training_end)])

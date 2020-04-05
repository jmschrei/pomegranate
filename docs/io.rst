.. _io:

Data Generators and IO
======================

- `IPython Notebook Tutorial <https://github.com/jmschrei/pomegranate/blob/master/tutorials/C_Feature_Tutorial_7_Data_Generators.ipynb>`_

The main way that data is fed into most Python machine learning models is formatted as numpy arrays. However, there are some cases where this is not convenient. The first case is when the data doesn't fit into memory. This case was dealt with a little bit in the Out of Core documentation page. The second case is when the data lives in some other format, such as a CSV file or some type of data base, and one doesn't want to create an entire copy of the data formatted as a numpy array.

Fortunately, pomegranate supports the use of data generators as input rather than only taking in numpy arrays. Data generators are objects that wrap data sets and yield batches of data in a manner that is specified by the user. Once the generator is exhausted the epoch is ended. The default data generator is to yield contiguous chunks of examples of a certain batch size until the entire data set has been seen, finish the epoch, and then start over.

The strength of data generators is that they allow the user to have a much greater degree of control over the training process than hardcoding a few training schemes. By specifying how exactly a batch is generated from the data set (and the preprocessing that might go into converting examples for use by the model) and exactly when an epoch ends, users can do a wide variety of out-of-core and mini-batch training schemes without anything needed to be built-in to pomegranate.

See the tutorial for more information about how to use and define your own data generators.

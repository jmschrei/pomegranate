.. _markovnetwork:

Markov Networks
===============

- `IPython Notebook Tutorial <https://github.com/jmschrei/pomegranate/blob/master/tutorials/B_Model_Tutorial_7_Markov_Networks.ipynb>`_

`Markov networks <https://en.wikipedia.org/wiki/Markov_random_field>`_ (sometimes called Markov random fields) are probabilistic models that are typically represented using an undirected graph. Each of the nodes in the graph represents a variable in the data and each of the edges represent an associate. Unlike Bayesian networks which have directed edges and clear directions of causality, Markov networks have undirected edges and only encode associations.

Currently, pomegranate only supports discrete Markov networks, meaning that the values must be categories, i.e. 'apples' and 'oranges', or 1 and 2, where 1 and 2 refer to categories, not numbers, and so 2 is not explicitly 'bigger' than 1. 


Initialization
--------------

Markov networks can be initialized in two ways, depending on whether the underlying graphical structure is known or not: (1) a list of the joint probabilities tables can be passed into the initialization, with one table per clique in the graph, or (2) both the graphical structure and distributions can be learned directly from data. This mirrors the other models that are implemented in pomegranate. However, because finding the optimal Markov network requires enumerating a number of potential graphs that is exponential with the number of dimensions in the data, it can be fairly time intensive to find the exact network.

Let's see an example of creating a Markov network with three cliques in it. 

.. code-block:: python

	from pomegranate import *

	d1 = JointProbabilityTable([
		[0, 0, 0.1],
		[0, 1, 0.2],
		[1, 0, 0.4],
		[1, 1, 0.3]], [0, 1])

	d2 = JointProbabilityTable([
		[0, 0, 0, 0.05],
		[0, 0, 1, 0.15],
		[0, 1, 0, 0.07],
		[0, 1, 1, 0.03],
		[1, 0, 0, 0.12],
		[1, 0, 1, 0.18],
		[1, 1, 0, 0.10],
		[1, 1, 1, 0.30]], [1, 2, 3])

	d3 = JointProbabilityTable([
		[0, 0, 0, 0.08],
		[0, 0, 1, 0.12],
		[0, 1, 0, 0.11],
		[0, 1, 1, 0.19],
		[1, 0, 0, 0.04],
		[1, 0, 1, 0.06],
		[1, 1, 0, 0.23],
		[1, 1, 1, 0.17]], [2, 3, 4])


	model = MarkovNetwork([d1, d2, d3])
	model.bake()

That was fairly simple. Each `JointProbabilityTable` object just had to include the table of all values that the variables can take as well as a list of variable indexes that are included in the table, in the order from left to right that they appear. For example, in d1, the first column of the table corresponds to the first column of data in a data matrix and the second column in the table corresponds to the second column in a data matrix.

One can also initialize a Markov network based completely on data. Currently, the only algorithm that pomegranate supports for this is the Chow-Liu tree-building algorithm. This algorithm first calculates the mutual information between all pairs of variables and then determines the maximum spanning tree through it. This process generally captures the strongest dependencies in the data set. However, because it requires all variables to have at least one connection, it can lead to instances where variables are incorrectly associated with each other. Overall, it generally performs well and it fairly fast to calculate.

.. code-block:: python
	
	from pomegranate import *
	import numpy

	X = numpy.random.randint(2, size=(100, 6))
	model = MarkovNetwork.from_samples(X)

Probability
-----------

The probability of an example under a Markov network is more difficult to calculate than under a Bayesian network. With a Bayesian network, one can simply multiply the probabilities of each variable given its parents to get a probability of the entire example. However, repeating this process for a Markov network (by plugging in the values of each clique and multiplying across all cliques) results in a value called the "unnormalized" probability. This value is called "unnormalized" because the sum of this value across all combinations of values that the variables in an example can take does not sum to 1. 

The normalization of an "unnormalized" probability requires the calculation of a partition function. This function (frequently abbreviated `Z`) is just the sum of the probability of all combinations of values that the variables can take. After calculation, one can just divide the unnormalized probability by this value to get the normalized probability. The only problem is that the calculation of the partition function requires the summation over a number of examples that grows exponentially with the number of dimensions. You can read more about this in the tutorial.

If you have a small number of variables (<30) it shouldn't be a problem to calculate the partition function and then normalized probabilities.

.. code-block:: python
	
	>>> print(model.probability([1, 0, 1, 0, 1]))
	-4.429966143312331

Prediction
----------

Markov networks can be used to predict the value of missing variables given the observed values in a process called "inference." In other predictive models there are typically a single or fixed set of missing values that need to be predicted, commonly referred to as the labels. However, in the case of Markov (or Bayesian) networks, the missing values can be any variables and the inference process will use all of the available data to impute those missing values. For example:

.. code-block:: python

	>>> print(model.predict([[None, 0, None, 1, None]]))
	[[1, 0, 0, 1, 1]]

API Reference
-------------

.. automodule:: pomegranate.MarkovNetwork
	:members:
	:inherited-members:

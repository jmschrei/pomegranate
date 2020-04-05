.. _bayesiannetwork:

Bayesian Networks
=================

- `IPython Notebook Tutorial <https://github.com/jmschrei/pomegranate/blob/master/tutorials/B_Model_Tutorial_4_Bayesian_Networks.ipynb>`_
- `IPython Notebook Structure Learning Tutorial <https://github.com/jmschrei/pomegranate/blob/master/tutorials/B_Model_Tutorial_4b_Bayesian_Network_Structure_Learning.ipynb>`_

`Bayesian networks <http://en.wikipedia.org/wiki/Bayesian_network>`_ are a probabilistic model that are especially good at inference given incomplete data. Much like a hidden Markov model, they consist of a directed graphical model (though Bayesian networks must also be acyclic) and a set of probability distributions. The edges encode dependency statements between the variables, where the lack of an edge between any pair of variables indicates a conditional independence. Each node encodes a probability distribution, where root nodes encode univariate probability distributions and inner/leaf nodes encode conditional probability distributions. Bayesian networks are exceptionally flexible when doing inference, as any subset of variables can be observed, and inference done over all other variables, without needing to define these groups in advance. In fact, the set of observed variables can change from one sample to the next without needing to modify the underlying algorithm at all. 

Currently, pomegranate only supports discrete Bayesian networks, meaning that the values must be categories, i.e. 'apples' and 'oranges', or 1 and 2, where 1 and 2 refer to categories, not numbers, and so 2 is not explicitly 'bigger' than 1. 


Initialization
--------------

Bayesian networks can be initialized in two ways, depending on whether the underlying graphical structure is known or not: (1) the graphical structure can be built one node at a time with pre-initialized distributions set for each node, or (2) both the graphical structure and distributions can be learned directly from data. This mirrors the other models that are implemented in pomegranate. However, typically expectation maximization is used to fit the parameters of the distribution, and so initialization (such as through k-means) is typically fast whereas fitting is slow. For Bayesian networks, the opposite is the case. Fitting can be done quickly by just summing counts through the data, while initialization is hard as it requires an exponential time search through all possible DAGs to identify the optimal graph. More is discussed in the tutorials above and in the fitting section below.

Let's take a look at initializing a Bayesian network in the first manner by quickly implementing the `Monty Hall problem <http://en.wikipedia.org/wiki/Monty_Hall_problem>`_. The Monty Hall problem arose from the gameshow *Let's Make a Deal*, where a guest had to choose which one of three doors had a prize behind it. The twist was that after the guest chose, the host, originally Monty Hall, would then open one of the doors the guest did not pick and ask if the guest wanted to switch which door they had picked. Initial inspection may lead you to believe that if there are only two doors left, there is a 50-50 chance of you picking the right one, and so there is no advantage one way or the other. However, it has been proven both through simulations and analytically that there is in fact a 66% chance of getting the prize if the guest switches their door, regardless of the door they initially went with. 

Our network will have three nodes, one for the guest, one for the prize, and one for the door Monty chooses to open. The door the guest initially chooses and the door the prize is behind are uniform random processes across the three doors, but the door which Monty opens is dependent on both the door the guest chooses (it cannot be the door the guest chooses), and the door the prize is behind (it cannot be the door with the prize behind it). 

.. code-block:: python

	from pomegranate import *

	guest = DiscreteDistribution({'A': 1./3, 'B': 1./3, 'C': 1./3})
	prize = DiscreteDistribution({'A': 1./3, 'B': 1./3, 'C': 1./3})
	monty = ConditionalProbabilityTable(
		[['A', 'A', 'A', 0.0],
		 ['A', 'A', 'B', 0.5],
		 ['A', 'A', 'C', 0.5],
		 ['A', 'B', 'A', 0.0],
		 ['A', 'B', 'B', 0.0],
		 ['A', 'B', 'C', 1.0],
		 ['A', 'C', 'A', 0.0],
		 ['A', 'C', 'B', 1.0],
		 ['A', 'C', 'C', 0.0],
		 ['B', 'A', 'A', 0.0],
		 ['B', 'A', 'B', 0.0],
		 ['B', 'A', 'C', 1.0],
		 ['B', 'B', 'A', 0.5],
		 ['B', 'B', 'B', 0.0],
		 ['B', 'B', 'C', 0.5],
		 ['B', 'C', 'A', 1.0],
		 ['B', 'C', 'B', 0.0],
		 ['B', 'C', 'C', 0.0],
		 ['C', 'A', 'A', 0.0],
		 ['C', 'A', 'B', 1.0],
		 ['C', 'A', 'C', 0.0],
		 ['C', 'B', 'A', 1.0],
		 ['C', 'B', 'B', 0.0],
		 ['C', 'B', 'C', 0.0],
		 ['C', 'C', 'A', 0.5],
		 ['C', 'C', 'B', 0.5],
		 ['C', 'C', 'C', 0.0]], [guest, prize])  

	s1 = Node(guest, name="guest")
	s2 = Node(prize, name="prize")
	s3 = Node(monty, name="monty")

	model = BayesianNetwork("Monty Hall Problem")
	model.add_states(s1, s2, s3)
	model.add_edge(s1, s3)
	model.add_edge(s2, s3)
	model.bake()

.. NOTE::
	The objects 'state' and 'node' are really the same thing and can be used interchangeable. The only difference is the name, as hidden Markov models use 'state' in the literature frequently whereas Bayesian networks use 'node' frequently. 

The conditional distribution must be explicitly spelled out in this example, followed by a list of the parents in the same order as the columns take in the table that is provided (e.g. the columns in the table correspond to guest, prize, monty, probability.)

However, one can also initialize a Bayesian network based completely on data. As mentioned before, the exact version of this algorithm takes exponential time with the number of variables and typically can't be done on more than ~25 variables. This is because there are a super-exponential number of directed acyclic graphs that one could define over a set of variables, but fortunately one can use dynamic programming in order to reduce this complexity down to "simply exponential." The implementation of the exact algorithm actually goes further than the original dynamic programming algorithm by implementing an A* search to somewhat reduce computational time but drastically reduce required memory, sometimes by an order of magnitude.

.. code-block:: python
	
	from pomegranate import *
	import numpy

	X = numpy.load('data.npy')
	model = BayesianNetwork.from_samples(X, algorithm='exact')

The exact algorithm is not the default, though. The default is a novel greedy algorithm that greedily chooses a topological ordering of the variables, but optimally identifies the best parents for each variable given this ordering. It is significantly faster and more memory efficient than the exact algorithm and produces far better estimates than using a Chow-Liu tree. This is set to the default to avoid locking up the computers of users that unintentionally tell their computers to do a near-impossible task.

Probability
-----------

You can calculate the probability of a sample under a Bayesian network as the product of the probability of each variable given its parents, if it has any. This can be expressed as :math:`P = \prod\limits_{i=1}^{d} P(D_{i}|Pa_{i})` for a sample with $d$ dimensions. For example, in the Monty Hal problem, the probability of a show is the probability of the guest choosing the respective door, times the probability of the prize being behind a given door, times the probability of Monty opening a given door given the previous two values. For example, using the manually initialized network above:

.. code-block:: python
	
	>>> print(model.probability([['A', 'A', 'A'],
		                     ['A', 'A', 'B'],
		                     ['C', 'C', 'B']]))
	[ 0.          0.05555556  0.05555556]

Prediction
----------

Bayesian networks are frequently used to infer/impute the value of missing variables given the observed values. In other models, typically there is either a single or fixed set of missing variables, such as latent factors, that need to be imputed, and so returning a fixed vector or matrix as the predictions makes sense. However, in the case of Bayesian networks, we can make no such assumptions, and so when data is passed in for prediction it should be in the format as a matrix with ``None`` in the missing variables that need to be inferred. The return is thus a filled in matrix where the Nones have been replaced with the imputed values. For example:

.. code-block:: python

	>>> print(model.predict([['A', 'B', None],
		                 ['A', 'C', None],
		                 ['C', 'B', None]]))
	[['A' 'B' 'C']
	 ['A' 'C' 'B']
	 ['C' 'B' 'A']]

In this example, the final column is the one that is always missing, but a more complex example is as follows:

.. code-block:: python

	>>> print(model.predict([['A', 'B', None],
	                 ['A', None, 'C'],
	                 [None, 'B', 'A']]))
	[['A' 'B' 'C']
	 ['A' 'B' 'C']
 	 ['C' 'B' 'A']]

Fitting
-------

Fitting a Bayesian network to data is a fairly simple process. Essentially, for each variable, you need consider only that column of data and the columns corresponding to that variables parents. If it is a univariate distribution, then the maximum likelihood estimate is just the count of each symbol divided by the number of samples in the data. If it is a multivariate distribution, it ends up being the probability of each symbol in the variable of interest given the combination of symbols in the parents. For example, consider a binary dataset with two variables, X and Y, where X is a parent of Y. First, we would go through the dataset and calculate P(X=0) and P(X=1). Then, we would calculate P(Y=0|X=0), P(Y=1|X=0), P(Y=0|X=1), and P(Y=1|X=1). Those values encode all of the parameters of the Bayesian network.


API Reference
-------------

.. automodule:: pomegranate.BayesianNetwork
	:members:
	:inherited-members:

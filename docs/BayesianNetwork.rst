.. _bayesiannetwork:

Bayesian Networks
=================

..
	- `IPython Notebook Tutorial <https://github.com/jmschrei/pomegranate/blob/master/tutorials/B_Model_Tutorial_4_Bayesian_Networks.ipynb>`_
	- `IPython Notebook Structure Learning Tutorial <https://github.com/jmschrei/pomegranate/blob/master/tutorials/B_Model_Tutorial_4b_Bayesian_Network_Structure_Learning.ipynb>`_

	`Bayesian networks <http://en.wikipedia.org/wiki/Bayesian_network>`_ are a probabilistic models composed of a set of probability distributions and a directed graph (usually acyclic) that encodes dependencies between the distributions. In order to do exact inference, the directed graph must be acyclic, but one can do approximate inference on a cyclic graph. Each distribution is univariate, but those that are not at roots of the graph are also categorical. 

Bayesian networks are exceptionally flexible when doing inference as any subset of variables can be observed and inference done over the remaining variables. The set of observed variables can change from one sample to the next without needing to modify the underlying algorithm at all. This is in contrast to most classic machine learning algorithms where the observed and unobserved variables are defined in advance and static throughout the learning and inference process.

Currently, pomegranate only supports categorical Bayesian networks where the values are encoded as ordinal numbers beginning at 0 and ending at ``n_categories-1``.


Initialization
--------------

Bayesian networks can be initialized in three ways depending on whether the underlying graph is known or not: (1) the graph can be built programmatically by adding the distributions and then adding one edge at a time, (2) the distributions and the entire graph can be passed in at initialization, or (3) the distributions and the graph can be learned from data. These options are similar to the initialization of other models in pomegranate. However, one noticable exception is that initialization is typically fast for other methods, e.g. using k-means to initialize a hidden Markov model, whereas the structure learning step in option 3 above can be very time-consuming. We will see more of this later.

Let's take a look at initializing a Bayesian network in the first manner by quickly implementing the `Monty Hall problem <http://en.wikipedia.org/wiki/Monty_Hall_problem>`_. The Monty Hall problem arose from the gameshow *Let's Make a Deal*, where a guest had to choose which one of three doors had a prize behind it. The twist was that after the guest chose, the host, originally Monty Hall, would then open one of the doors the guest did not pick and ask if the guest wanted to switch which door they had picked. Initial inspection may lead you to believe that if there are only two doors left, there is a 50-50 chance of you picking the right one, and so there is no advantage one way or the other. However, it has been proven both through simulations and analytically that there is in fact a 66% chance of getting the prize if the guest switches their door, regardless of the door they initially went with. 

Our network will have three distributions: guest, prize, and the door that Monty chooses to open. The door the guest initially chooses and the door the prize is behind are uniform random processes across the three doors, but the door which Monty opens is dependent on both the door the guest chooses (it cannot be the door the guest chooses), and the door the prize is behind (it cannot be the door with the prize behind it). 

.. code-block:: python

	>>> from torchegranate.distributions import Categorical
	>>> from torchegranate.distributions import ConditionalCategorical
	>>> from torchegranate.bayesian_network import BayesianNetwork
	>>>
	>>> guest = Categorical([[1./3, 1./3, 1./3]])
	>>> prize = Categorical([[1./3, 1./3, 1./3]])
	>>>
	>>> probs = numpy.array([[
	     [[0.0, 0.5, 0.5], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0]], 
	     [[0.0, 0.0, 1.0], [0.5, 0.0, 0.5], [1.0, 0.0, 0.0]],
	     [[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.5, 0.5, 0.0]]
	]])
	>>> monty = ConditionalCategorical(probs) 
	>>> model = BayesianNetwork([guest, prize, monty], [(guest, monty), (prize, monty)])

.. NOTE::
	The API of Bayesian networks is different in pomegranate v1.0.0 than it was in previous versions. There is no need to use ``Node`` or ``State`` objects because those objects do not exist anymore, there is no ``bake`` method to call at the end, and the conditional distribution doesn't need to have the parents specified -- that's determined by the edges added to the network and the order that they are added in.


However, one can also initialize a Bayesian network by using data to learn the structure and the parameters of the distributions jointly. The naive exact version of categorical structure learning takes a super-exponential amount of time but pomegranate uses a dynamic programming-based implementation that reduces this down to just an exponential amount of time. Unfortunately, even this faster version cannot scale to more than 20-30 variables, depending on your hardware. The reason the exact algorithm takes so long is because you are considering all combinations of parents for each node and also enforcing the acyclicity constraint, which are both challenging.

If you want to initialize a Bayesian network from data, i.e., perform structure learning, all you need to do is create the object and fit it to data.


.. code-block:: python
	
	>>> from pomegranate import *
	>>> import numpy
	>>>
	>>> X = numpy.load('data.npy')
	>>> model = BayesianNetwork().fit(X)

..
	The exact algorithm is not the default, though. The default is a novel greedy algorithm that greedily chooses a topological ordering of the variables, but optimally identifies the best parents for each variable given this ordering. It is significantly faster and more memory efficient than the exact algorithm and produces far better estimates than using a Chow-Liu tree. This is set to the default to avoid locking up the computers of users that unintentionally tell their computers to do a near-impossible task.


Probability
-----------

You can calculate the probability of a sample under a Bayesian network as the product of the probability of each variable given its parents, if it has any. This can be expressed as :math:`P = \prod\limits_{i=1}^{d} P(D_{i}|Pa_{i})` for a sample with $d$ dimensions. For example, in the Monty Hal problem, the probability of a show is the probability of the guest choosing the respective door, times the probability of the prize being behind a given door, times the probability of Monty opening a given door given the previous two values. For example, using the manually initialized network above:

.. code-block:: python
	
	>>> model.probability([[0, 0, 0],
		                     [0, 0, 1],
		                     [2, 2, 0]]))
	torch.tensor([ 0.          0.05555556  0.05555556])


Prediction
----------

In addition to calculating the probability of examples, Bayesian networks can also be given incomplete data sets and used for imputation. In this setting, the missing values are imputed using the observed values, as well as the structure of the network, for each example. Importantly, because each example is imputed individually, there is no information shared across the column. Regardless, this imputation approach is very powerful because different sets of variables can be missing in each example. 

This is implemented in pomegranate using a `torch.masked.MaskedTensor` to store the observed values and a mask indicating what values are not observed. The tensor of observed values can have any value at the locations for the missing values and, usually, one would want that value to be something easy to construct a mask from. For example, in the below example, the tensor of values has a ``-1`` where the values are missing, and the mask is generated using that value. Observed values are returned as-is.

.. code-block:: python

	>>> X = torch.tensor([[0, 1, -1], [0, 2, -1], [2, 1, -1]])
	>>> X_masked = torch.masked.MaskedTensor(X, mask=X >= 0)
	>>> model.predict(X_masked)
	tensor([[0, 1, 2],
  	        [0, 2, 1],
      	    [2, 1, 0]])


Fitting
-------

Fitting a Bayesian network to data is a fairly simple process. Essentially, for each variable, you need consider only that column of data and the columns corresponding to that variables parents. If the distribution does not have parents then the MLE is just the count of each category divided by the number of samples in the data. If the distribution does have parents then the MLE ends up being the probability of each symbol in the variable of interest given the combination of symbols in the parents. For example, consider a binary dataset with two variables, X and Y, where X is a parent of Y. First, we would go through the dataset and calculate P(X=0) and P(X=1). Then, we would calculate P(Y=0|X=0), P(Y=1|X=0), P(Y=0|X=1), and P(Y=1|X=1). Those values encode all of the parameters of the Bayesian network.


API Reference
-------------

.. automodule:: pomegranate.BayesianNetwork
	:members:
	:inherited-members:

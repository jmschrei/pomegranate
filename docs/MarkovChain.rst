.. _markovchain:

Markov Chains
=============

`IPython Notebook Tutorial <https://github.com/jmschrei/pomegranate/blob/master/tutorials/B_Model_Tutorial_6_Markov_Chain.ipynb>`_

Markov chains are form of structured model over sequences. They represent the probability of each character in the sequence as a conditional probability of the last k symbols. For example, a 3rd order Markov chain would have each symbol depend on the last three symbols. A 0th order Markov chain is a naive predictor where each symbol is independent of all other symbols. Currently pomegranate only supports discrete emission Markov chains where each symbol is a discrete symbol versus a continuous number (like 'A' 'B' 'C' instead of 17.32 or 19.65).

Initialization
--------------

Markov chains can almost be represented by a single conditional probability table (CPT), except that the probability of the first k elements (for a k-th order Markov chain) cannot be appropriately represented except by using special characters. Due to this pomegranate takes in a series of k+1 distributions representing the first k elements. For example for a second order Markov chain:

.. code-block:: python

	from pomegranate import *
	d1 = DiscreteDistribution({'A': 0.25, 'B': 0.75})
	d2 = ConditionalProbabilityTable([['A', 'A', 0.1],
	                                      ['A', 'B', 0.9],
	                                      ['B', 'A', 0.6],
	                                      ['B', 'B', 0.4]], [d1])
	d3 = ConditionalProbabilityTable([['A', 'A', 'A', 0.4],
	                                      ['A', 'A', 'B', 0.6],
	                                      ['A', 'B', 'A', 0.8],
	                                      ['A', 'B', 'B', 0.2],
	                                      ['B', 'A', 'A', 0.9],
	                                      ['B', 'A', 'B', 0.1],
	                                      ['B', 'B', 'A', 0.2],
	                                      ['B', 'B', 'B', 0.8]], [d1, d2])
	model = MarkovChain([d1, d2, d3])

Probability
-----------

The probability of a sequence under the Markov chain is just the probability of the first character under the first distribution times the probability of the second character under the second distribution and so forth until you go past the (k+1)th character, which remains evaluated under the (k+1)th distribution. We can calculate the probability or log probability in the same manner as any of the other models. Given the model shown before:

.. code-block:: python

	>>> model.log_probability(['A', 'B', 'B', 'B'])
	-3.324236340526027
	>>> model.log_probability(['A', 'A', 'A', 'A'])
	-5.521460917862246

Fitting
-------

Markov chains are not very complicated to train. For each sequence the appropriate symbols are sent to the appropriate distributions and maximum likelihood estimates are used to update the parameters of the distributions. There are no latent factors to train and so no expectation maximization or iterative algorithms are needed to train anything.

API Reference
-------------

.. automodule:: pomegranate.MarkovChain
	:members:
	:inherited-members:

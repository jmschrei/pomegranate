.. _hiddenmarkovmodel:

Hidden Markov Models
====================

- `IPython Notebook Tutorial <https://github.com/jmschrei/pomegranate/blob/master/tutorials/Tutorial_3_Hidden_Markov_Models.ipynb>`_
- `IPython Notebook Sequence Alignment Tutorial <http://nbviewer.ipython.org/github/jmschrei/yahmm/blob/master/examples/Global%20Sequence%20Alignment.ipynb>`_

`Hidden Markov models <http://en.wikipedia.org/wiki/Hidden_Markov_model>`_ (HMMs) are a structured probabilistic model that forms a probability distribution of sequences, as opposed to individual symbols. It is similar to a Bayesian network in that it has a directed graphical structure where nodes represent probability distributions, but unlike Bayesian networks in that the edges represent transitions and encode transition probabilities, whereas in Bayesian networks edges encode dependence statements. A HMM can be thought of as a general mixture model plus a transition matrix, where each component in the general Mixture model corresponds to a node in the hidden Markov model, and the transition matrix informs the probability that adjacent symbols in the sequence transition from being generated from one component to another. A strength of HMMs is that they can model variable length sequences whereas other models typically require a fixed feature set. They are extensively used in the fields of natural language processing to model speech, bioinformatics to model biosequences, and robotics to model movement.  

The HMM implementation in pomegranate is based off of the implementation in its predecessor, Yet Another Hidden Markov Model (YAHMM). To convert a script that used YAHMM to a script using pomegranate, you only need to change calls to the ``Model`` class to call ``HiddenMarkovModel``. For example, a script that previously looked like the following:

.. code-block:: python
	
	from yahmm import *
	model = Model()

would now be written as

.. code-block:: python
	
	from pomegranate import *
	model = HiddenMarkovModel()

and the remaining method calls should be identical.


Initialization
--------------

Hidden Markov models can be initialized in one of two ways depending on if you know the initial parameters of the model, either (1) by defining both the distributions and the graphical structure manually, or (2) running the ``from_samples`` method to learn both the structure and distributions directly from data. The first initialization method can be used either to specify a pre-defined model that is ready to make predictions, or as the initialization to a training algorithm such as Baum-Welch. It is flexible enough to allow sparse transition matrices and any type of distribution on each node, i.e. normal distributions on several nodes, but a mixture of normals on some nodes modeling more complex phenomena. The second initialization method is less flexible, in that currently each node must have the same distribution type, and that it will only learn dense graphs. Similar to mixture models, this initialization method starts with k-means to initialize the distributions and a uniform probability transition matrix before running Baum-Welch.

If you are initializing the parameters manually, you can do so either by passing in a list of distributions and a transition matrix, or by building the model line-by-line. Let's first take a look at building the model from a list of distributions and a transition matrix.

.. code-block:: python

	from pomegranate import *
	dists = [NormalDistribution(5, 1), NormalDistribution(1, 7), NormalDistribution(8,2)]
	trans_mat = numpy.array([[0.7, 0.3, 0.0],
	                             [0.0, 0.8, 0.2],
	                             [0.0, 0.0, 0.9]])
	starts = numpy.array([1.0, 0.0, 0.0])
	ends = numpy.array([0.0, 0.0, 0.1])
	model = HiddenMarkovModel.from_matrix(trans_mat, dists, starts, ends)

Next, let's take a look at building the same model line by line.

.. code-block:: python

	from pomegranate import *
	s1 = State(NormalDistribution(5, 1))
	s2 = State(NormalDistribution(1, 7))
	s3 = State(NormalDistribution(8, 2))
	model = HiddenMarkovModel()
	model.add_states(s1, s2, s3)
	model.add_transition(model.start, s1, 1.0)
	model.add_transition(s1, s1, 0.7)
	model.add_transition(s1, s2, 0.3)
	model.add_transition(s2, s2, 0.8)
	model.add_transition(s2, s3, 0.2)
	model.add_transition(s3, s3, 0.9)
	model.add_transition(s3, model.end, 0.1)
	model.bake()

Initially it may seem that the first method is far easier due to it being fewer lines of code. However, when building large sparse models defining a full transition matrix can be cumbersome, especially when it is mostly 0s.

Models built in this manner must be explicitly "baked" at the end. This finalizes the model topology and creates the internal sparse matrix which makes up the model. This step also automatically normalizes all transitions to make sure they sum to 1.0, stores information about tied distributions, edges, pseudocounts, and merges unnecessary silent states in the model for computational efficiency. This can cause the `bake` step to take a little bit of time. If you want to reduce this overhead and are sure you specified the model correctly you can pass in `merge="None"` to the bake step to avoid model checking.

The second way to initialize models is to use the ``from_samples`` class method. The call is identical to initializing a mixture model.

.. code-block:: python

	from pomegranate import *
	model = HiddenMarkovModel.from_samples(NormalDistribution, n_components=5, X=X)

Much like a mixture model, all arguments present in the ``fit`` step can also be passed in to this method. Also like a mixture model, it is initialized by running k-means on the concatenation of all data, ignoring that the symbols are part of a structured sequence. The clusters returned are used to initialize all parameters of the distributions, i.e. both mean and covariances for multivariate Gaussian distributions. The transition matrix is initialized as uniform random probabilities. After the components (distributions on the nodes) are initialized, the given training algorithm is used to refine the parameters of the distributions and learn the appropriate transition probabilities.


Log Probability
---------------

There are two common forms of the log probability which are used. The first is the log probability of the most likely path the sequence can take through the model, called the Viterbi probability. This can be calculated using ``model.viterbi(sequence)``.  However, this is :math:`P(D|S_{ML}, S_{ML}, S_{ML})` not :math:`P(D|M)`. In order to get :math:`P(D|M)` we have to sum over all possible paths instead of just the single most likely path. This can be calculated using ``model.log_probability(sequence)`` and uses the forward algorithm internally. On that note, the full forward matrix can be returned using ``model.forward(sequence)`` and the full backward matrix can be returned using ``model.backward(sequence)``, while the full forward-backward emission and transition matrices can be returned using ``model.forward_backward(sequence)``.

Prediction
----------

A common prediction technique is calculating the Viterbi path, which is the most likely sequence of states that generated the sequence given the full model. This is solved using a simple dynamic programming algorithm similar to sequence alignment in bioinformatics. This can be called using ``model.viterbi(sequence)``. A sklearn wrapper can be called using ``model.predict(sequence, algorithm='viterbi')``. 

Another prediction technique is called maximum a posteriori or forward-backward, which uses the forward and backward algorithms to calculate the most likely state per observation in the sequence given the entire remaining alignment. Much like the forward algorithm can calculate the sum-of-all-paths probability instead of the most likely single path, the forward-backward algorithm calculates the best sum-of-all-paths state assignment instead of calculating the single best path. This can be called using ``model.predict(sequence, algorithm='map')`` and the raw normalized probability matrices can be called using ``model.predict_proba(sequence)``.

Fitting
------- 

A simple fitting algorithm for hidden Markov models is called Viterbi training. In this method, each observation is tagged with the most likely state to generate it using the Viterbi algorithm. The distributions (emissions) of each states are then updated using MLE estimates on the observations which were generated from them, and the transition matrix is updated by looking at pairs of adjacent state taggings. This can be done using ``model.fit(sequence, algorithm='viterbi')``. 

However, this is not the best way to do training and much like the other sections there is a way of doing training using sum-of-all-paths probabilities instead of maximally likely path. This is called Baum-Welch or forward-backward training. Instead of using hard assignments based on the Viterbi path, observations are given weights equal to the probability of them having been generated by that state. Weighted MLE can then be done to update the distributions, and the soft transition matrix can give a more precise probability estimate. This is the default training algorithm, and can be called using either ``model.fit(sequences)`` or explicitly using ``model.fit(sequences, algorithm='baum-welch')``. 

Fitting in pomegranate also has a number of options, including the use of distribution or edge inertia, freezing certain states, tying distributions or edges, and using pseudocounts. See the tutorial linked to at the top of this page for full details on each of these options.

API Reference
-------------

.. automodule:: pomegranate.hmm
    :members:
    :inherited-members:

Hidden Markov Models
====================

[IPython Notebook Tutorial](https://github.com/jmschrei/pomegranate/blob/master/tutorials/Tutorial_3_Hidden_Markov_Models.ipynb)
[IPython Notebook Sequence Alignment Tutorial](http://nbviewer.ipython.org/github/jmschrei/yahmm/blob/master/examples/Global%20Sequence%20Alignment.ipynb)

[Hidden Markov models](http://en.wikipedia.org/wiki/Hidden_Markov_model) (HMMs) are a form of structured learning, in which a sequence of observations are labelled according to the hidden state they belong. A strength of HMMs is their ability to analyze variable length sequences whereas other models require a static set of features. This makes them extensively used in the fields of natural language processing and bioinformatics where data is routinely variable length sequences. They can be thought of as a structured form of General Mixture Models. 

Initialization
--------------

Transitioning from YAHMM to pomegranate is simple because the only change is that the `Model` class is now `HiddenMarkovModel`. You can port your code over by either changing `Model` to `HiddenMarkovModel` or changing your imports at the top of the file as follows:

```python
>>> from pomegranate import *
>>> from pomegranate import HiddenMarkovModel as Model
```

instead of

```python
>>> from yahmm import *
```

Since hidden Markov models are graphical structures, that structure has to be defined. pomegranate allows you to define this structure either through matrices as is common in other packages, or build it up state by state and edge by edge. pomegranate differs from other packages in that it offers both explicit start and end states which you must begin in or end in. Explicit end states give you more control over the model because algorithms require ending there, as opposed to in any state in the model. It also offers silent states, which are states without explicit emission distributions but can be used to significantly simplify the graphical structure. 

```python
>>> from pomegranate import *
>>> dists = [NormalDistribution(5, 1), NormalDistribution(1, 7), NormalDistribution(8,2)]
>>> trans_mat = numpy.array([[0.7, 0.3, 0.0],
                             [0.0, 0.8, 0.2],
                             [0.0, 0.0, 0.9]])
>>> starts = numpy.array([1.0, 0.0, 0.0])
>>> ends = numpy.array([0.0, 0.0, 0.1])
>>> model = HiddenMarkovModel.from_matrix(trans_mat, dists, starts, ends)
```

Alternatively this model could be created edge by edge and state by state. This is helpful for large sparse graphs. You must add transitions using the explicit start and end states where the sum of probabilities leaving a state sums to 1.0.

```python
>>> from pomegranate import *
>>> s1 = State( Distribution( NormalDistribution(5, 1) ) )
>>> s2 = State( Distribution( NormalDistribution(1, 7) ) )
>>> s3 = State( Distribution( NormalDistribution(8, 2) ) )
>>> model = HiddenMarkovModel()
>>> model.add_states(s1, s2, s3)
>>> model.add_transition(model.start, s1, 1.0)
>>> model.add_transition(s1, s1, 0.7)
>>> model.add_transition(s1, s2, 0.3)
>>> model.add_transition(s2, s2, 0.8)
>>> model.add_transition(s2, s3, 0.2)
>>> model.add_transition(s3, s3, 0.9)
>>> model.add_transition(s3, model.end, 0.1)
>>> model.bake()
```

Models built in this manner must be explicitly "baked" at the end. This finalizes the model topology and creates the internal sparse matrix which makes up the model. This removes "orphan" parts of the model, normalizes all transitions to make sure they sum to 1.0, stores information about tied distributions, edges, and pseudocounts, and merges unneccesary silent states in the model for computational efficiency. This can cause the `bake` step to take a little bit of time. If you want to reduce this overhead and are sure you specified the model correctly you can pass in `merge="None"` to the bake step to avoid model checking.

Log Probability
---------------

There are two common forms of the log probability which are used. The first is the log probability of the most likely path the sequence can take through the model, called the Viterbi probability. This can be calculated using `model.viterbi(sequence)`.  However, this is $P(D|S_{ML}_{0}, S_{ML}_{1}, S_{ML}_{2}...)$ not $P(D|M)$. In order to get $P(D|M)$ we have to sum over all possible paths instead of just the single most likely path, which can be calculated using `model.log_probability(sequence)` using the forward or backward algorithms. On that note, the full forward matrix can be returned using `model.forward(sequence)` and the full backward matrix can be returned using `model.backward(sequence)`.

Prediction
----------

A common prediction technique is calculating the Viterbi path, which is the most likely sequence of states that generated the sequence given the full model. This is solved using a simple dynamic programming algorithm similar to sequence alignment in bioinformatics. This can be called using `model.viterbi(sequence)`. A sklearn wrapper can be called using `model.predict(sequence, algorithm='viterbi')`. 

Another prediction technique is called maximum a posteriori or forward-backward, which uses the forward and backward algorithms to calculate the most likely state per observation in the sequence given the entire remaining alignment. Much like the forward algorithm can calculate the sum-of-all-paths probability instead of the most likely single path, the forward-backward algorithm calculates the best sum-of-all-paths state assignment instead of calculating the single best path. This can be called using `model.predict(sequence, algorithm='map')` and the raw normalized probability matrices can be called using `model.predict_proba(sequence)`.

Fitting
------- 

A simple fitting algorithm for hidden Markov models is called Viterbi training. In this method, each observation is tagged with the most likely state to generate it using the Viterbi algorithm. The distributions (emissions) of each states are then updated using MLE estimates on the observations which were generated from them, and the transition matrix is updated by looking at pairs of adjacent state taggings. This can be done using `model.fit(sequence, algorithm='viterbi')`. 

However, this is not the best way to do training and much like the other sections there is a way of doing training using sum-of-all-paths probabilities instead of maximally likely path. This is called Baum-Welch or forward-backward training. Instead of using hard assignments based on the Viterbi path, observations are given weights equal to the probability of them having been generated by that state. Weighted MLE can then be done to updates the distributions, and the soft transition matrix can give a more precise probability estimate. This is the default training algorithm, and can be called using either `model.fit(sequences)` or explicitly using `model.fit(sequences, algorithm='baum-welch')`. 

Fitting in pomegranate also has a number of options, including the use of distribution or edge inertia, freezing certain states, tying distributions or edges, and using pseudocounts. See the tutorial linked to at the top of this page for full details on each of these options.

API Reference
-------------

```eval_rst
.. automodule:: pomegranate.hmm
    :members:
```
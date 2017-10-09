.. currentmodule:: pomegranate


===============
Release History
===============

Version 0.8.0
=============

Highlights
----------

This will serve as a log for the changes added for the release of version 0.7.8.


Changelog
---------

k-means
.......

	- k-means has been changed from using iterative computation to using the alternate formulation of euclidean distance, from ||a - b||^{2} to using ||a||^{2} + ||b||^{2} - 2||a \cdot b||. This allows for the centroid norms to be cached, significantly speeding up computation, and for dgemm to be used to solve the matrix matrix multiplication. Initial attempts to add in GPU support appeared unsuccessful, but in theory it should be something that can be added in.

	- k-means has been refactored to more natively support an out-of-core learning goal, by allowing for data to initially be cast as numpy memorymaps and not coercing them to arrays midway through.


Hidden Markov Models
....................

	- Allowed labels for labeled training to take in string names of the states instead of the state objects themselves.

	- Added in `state_names` and `names` parameters to the `from_samples` method to allow for more control over the creation of the model.

	- Added in semi-supervised learning to the `fit` step that can be activated by passing in a list of labels where sequences that have no labels have a None value. This allows for training to occur where some sequences are fully labeled and others have no labels, not for training to occur on partially labeled sequences.

	- Supervised initialization followed by semi-supervised learning added in to the `from_samples` method similarly to other methods. One should do this by passing in string labels for state names, always starting with <model_name>-start, where model_name is the `name` parameter passed into the `from_samples` method. Sequences that do not have labels should have a None instead of a list of corresponding labels. While semi-supervised learning using the `fit` method can support arbitrary transitions amongst silent states, the `from_samples` method does not produce silent states, and so other than the start and end states, all states should be symbol emitting states. If using semi-supervised learning, one must also pass in a list of the state names using the `state_names` parameter that has been added in.

	- Fixed bug in supervised learning where it would not initialize correctly due to an error in the semi-supervised learning implementation.

	- Fixed bug where model could not be plotted without pygraphviz due to an incorrect call to networkx.draw.

General Mixture Models
......................

	- Changed the initialization step to be done on the first batch of data instead of the entire dataset. If the entire dataset fits in memory this does not change anything. However, this allows for out-of-core updates to be done automatically instead of immediately trying to load the entire dataset into memory. This does mean that out-of-core updates will have a different initialization now, but then yield exact updates after that.

	- Fixed bug where passing in a 1D array would cause an error by recasting all 1D arrays as 2D arrays.

Bayesian Networks
.................

	- Added in a reduce_dataset parameter to the `from_samples` method that will take in a dataset and create a new dataset that is the unique set of samples, weighted by their weighted occurance in the dataset. Essentially, it takes a dataset that may have repeating members, and produces a new dataset that is entirely unique members. This produces an identically scoring Bayesian network as before, but all structure learning algorithms can be significantly sped up. This speed up is proportional to the redundancy of the dataset, so large datasets on a smallish (< 12) number of variables will see massive speed gains (sometimes even 2-3 orders of magnitude!) whereas past that it may not be beneficial. The redundancy of the dataset (and thus the speedup) can be estimated as n_samples / n_possibilities, where n_samples is the number of samples in the dataset and n_possibilities is the product of the number of unique keys per variable, or 2**d for binary data with d variables. It can be calculated exactly as n_samples / n_unique_samples, as many datasets are biased towards repeating elements. 

	- Fixed a premature optimization where the parents were stripped from conditional probability tables when saving the Bayesian Network to a json, causing an error in serialization. The premature optimization is that in theory pomegranate is set up to handle cyclic Bayesian networks and serializing that without first stripping parents would cause an infinite file size. However, a future PR that enabled cyclic Bayesian networks will account for this error.

Naive Bayes
...........

	- Fixed documentation of `from_samples` to actually refer to the naive Bayes model.

	- Added in semi-supervised learning through the EM algorithm for samples that are labeled with -1.

Bayes Classifier
................

	- Fixed documentation of `from_samples` to actually refer to the Bayes classifier model.

	- Added in semi-supervised learning through the EM algorithm for samples that are labeled with -1.

Distributions
.............

	- Multivariate Gaussian Distributions can now use GPUs for both log probability and summarization calculations, speeding up both tasks ~4x for any models that use them. This is added in through CuPy.

Out Of Core
...........

	- The parameter "batch_size" has been added to HMMs, GMMs, and k-means models for built-in out-of-core calculations. Pass in a numpy memory map instead of an array and set the batch size for exact updates (sans initialization).

Minibatching
............

	- The parameter "batches_per_epoch" has been added to HMMs, GMMs, and k-means models for build-in minibatching support. This specifies the number of batches (as defined by "batch_size") to summarize before calculating new parameter updates.

	- The parameter "lr_decay" has been added to HMMs and GMMs that specifies the decay in the learning rate over time. Models may not converge otherwise when doing minibatching.

Parallelization
...............

	- `n_jobs` has been added to all models for both fitting and prediction steps. This allows users to make parallelized predictions with their model without having to do anything more complicated than setting a larger number of jobs.

Tutorials
.........

	- Removed the PyData 2016 Chicago Tutorial due to it's similarity to tutorials_0_pomegranate_overview.
.. currentmodule:: pomegranate


===============
Release History
===============

Version 0.12.1
==============

Highlights
----------

	- A variety of minor bug fixes.


Version 0.12.0
==============

Highlights
----------

	- MarkovNetwork models have been added in and include both inference and structure learning.
	- Support for Python 2 has been depricated.
	- Markov network, data generator, and callback tutorials have been added in
	- A robust `from_json` method has been added in to __init__.py that can deserialize JSONs from any pomegranate model.

MarkovNetwork
-------------
	
	- MarkovNetwork models have been added in as a new probabilistic model.
	- Loopy belief propagation inference has been added in using the FactorGraph backend
	- Structure learning has been added in using Chow-Liu trees

BayesianNetwork
---------------

	- Chow-Liu tree building has been sped up slightly, courtesy of @alexhenrie
	- Chow-Liu tree building was further sped up by almost an order of magnitude
	- Constraint Graphs no longer fail when passing in graphs with self loops, courtesy of @alexhenrie

BayesClassifier
---------------

	- Updated the `from_samples` method to accept BayesianNetwork as an emission. This will build one Bayesian network for each class and use them as the emissions.

Distributions
-------------

	- Added a warning to DiscreteDistribution when the user passes in an empty dictionary.
	- Fixed the sampling procedure for JointProbabilityTables. 
	- GammaDistributions should have their shape issue resolved
	- The documentation for BetaDistributions has been updated to specify that it is a Beta-Bernoulli distribution. 

io
---
	
	- New file added, io.py, that contains data generators that can be operated on
	- Added DataGenerator, DataFrameGenerator, and a BaseGenerator class to inherit from 

HiddenMarkovModel
-----------------

	- Added RandomState parameter to `from_samples` to account for randomness when building discrete models.

Misc
----

	- Unneccessary calls to memset have been removed, courtesy of @alexhenrie
	- Checking for missing values has been slightly refactored to be cleaner, courtesy of @mareksmid-lucid
	- Include the LICENSE file in MANIFEST.in and simplify a bit, courtesy of @toddrme2178
	- Added in a robust from_json method that can be used to deseralize a JSON for any pomegranate model.

docs
----

	- Added io.rst to briefly describe data generators
	- Added MarkovNetwork.rst to describe Markov networks
	- Added links to tutorials that did not have tutorials linked to them.

Tutorials
---------
	
	- Added in a tutorial notebook for Markov networks
	- Added in a tutorial notebook for data generators
	- Added in a tutorial notebook for callbacks

CI
--

	- Removed unit tests for Py2.7 from AppVeyor and Travis
	- Added unit tests for Py3.8 to AppVeyor and Travis 

Version 0.11.2
==============

Highlights
----------

	- Faster BSNL, particularly when there is missing data, courtesy of @alexhenrie
	- GPU acceleration should be fixed

BayesianNetwork
---------------

	- A speed improvement by making `isnan` an inline function, courtesy of @alexhenrie
	- A speed improvement by changing the manner that parent sets are iterated, courtesy of @alexhenrie

Utils
-----

	- The `enable_gpu` call has been moved to the bottom of the GPU checking code and so should not crash anymore.

Version 0.11.1
==============

Highlights
----------

	- Added speed improvements to Bayesian network structure learning when missing data is present.

BayesianNetwork
---------------

	- By default duplicates get merged in a data set so that there are fewer rows with larger weights, dramatically improving speed. However, because `np.nan != np.nan`, rows with missing values don't get merged. This fix changes `np.nan` to `None` so that the rows get merged appropriately.

	- A few misc changes that sometimes improve speed.

	- Changed the probability calculation when a node is being scored given a single row. Previously it would return 0, meaning that sometimes it will return the densest graph possible erroneously. This may change your networks in edge cases, but will reduce their complexity.

Version 0.11.0
==============

Highlights
----------

	- Allowed for user specified custom distributions by implementing a Python fallback option if the distribution object doesn't inherit from the base distribution class.
	- Fixed an issue with GammaDistribution update
	- Removed deterministic seed being set in hmm.bake
	- Made pomegranate compatible with NetworkX v2.0 and above
	- NeuralHMMs and Neural Mixture Models are now possible through the custom distributions
	- Many new tutorials


Distributions
-------------

	- Fixed an error in GammaDistribution's cython level update step where sufficient statistics were incorrectly collected from a data set. This will only affect GammaDistributions that are used as part of a composition model rather than stand-alone ones.

	- Added in support for custom distributions. This is done by checking whether a distribution is inherited from the base pomegranate distribution object. If not, it will use the python methods. 

	- Added in examples of using custom distributions, including neural networks, with pomegranate models.

	- Made NormalDistribution.blank and LogNormalDistribution.blank return distributions with a standard deviation of 1, to avoid DivisionByZero errors.

	- Added in a NeuralNetworkWrapper distribution that should handle wrapping a neural network correctly for use in pomegranate. This assumes a keras-like API.

HiddenMarkovModel
-----------------

	- Removed a deterministic seed being set in hmm.bake. These lines were set because it was thought that there was some randomness in either the internal state generation of the topological sort. However, it appears that this is not necessary, and so it has been removed.

	- Fixed a bug where semi-supervised learning would not work because of an undefined variable.

	- Added in support for networkx v2.0 and above using their new API.

Tutorials
---------
	
	- Revamped the tutorials in the tutorials folder, greatly expanding their scope

	- Added in new tutorials about custom distributions and neural probabilistic models


Version 0.10.0
==============

Highlights
----------

	- Broke distributions into their own files and placed them in their own folder
	- Fixed Bayesian network failing in call to np.isnan when fitting to character data
	- Added in callbacks to all models in the style of keras, with built-ins being History, ModelCheckpoint, and CVLogger. History is calculated for each model. Use `return_history=True` to gt the model and the history object that contains training.
	- Added top-level Makefile for convenience in development to build/test/clean/install/uninstall with multiple conda environments.
	- Added top-level rebuildconda for convenience in development to create or re-create a conda development environment for a given python version, defaulting to 2.7.

Changelog
---------

Callbacks
---------

	- Added in a callbacks module, and the use of callbacks in all iterative training procedures. Callbacks are called at the beginning of training, at the end of each epoch, and at the end of the training procedure, using the respective functions. See the documentation page for more details.


Distributions
-------------
	
	- Broke the distributions.pyx into a folder where each distribution has its own file. This will speed up compilation when the code is modified.

	- Added in a `dtype` attribute to DiscreteDistribution, ConditionalProbabilityTable, and JointProbabilityTable, to prevent automatic casting of keys as floats when converting to and from jsons

	- For MultivariateGaussianDistributions, added in an epsilon when performing a ridge adjustment on a non-positive semidefinite matrix to hopefully completely fix this issue.

	- NormalDistribution update should now check to see if the weights are below an epsilon, rather than equal to 0, resolving some stability issues.

	- Fixed an issue with BernoulliDistribution where it would raise a ZeroDivisionError when `from_summaries` was called with no observations.

	- Fixed an issue where an IndependentComponentsDistribution would print upon calls to `log_probability`


HiddenMarkovModel
-----------------

    - Changed the output to be the fit model, like in scikit-learn, instead of the total improvement, to allow for chaining

	- Added in callback functionality to both the `fit` and `from_samples` methods

	- Added in the `return_history` parameter to both the `fit` and `from_samples` methods, which will return the history callback as well as the fit model

	- Resolved an issue in the `summary` method where default weights were assigned to the wrong variable when not passed in.

	- Resolved an issue where printing an empty model resulted in an error.

GeneralMixtureModel
-------------------

    - Changed the output to be the fit model, like in scikit-learn, instead of the total improvement, to allow for chaining

	- Added in callback functionality to both the `fit` and `from_samples` methods

	- Added in the `return_history` parameter to both the `fit` and `from_samples` methods, which will return the history callback as well as the fit model


NaiveBayes
----------

	- Added in callback functionality to both the `fit` and `from_samples` methods that will be used only in semi-supervised learning

	- Added in the `return_history` parameter to both the `fit` and `from_samples` methods, which will return the history callback as well as the fit model that will be used only in semi-supervised learning


BayesClassifier
---------------

	- Added in callback functionality to both the `fit` and `from_samples` methods that will be used only in semi-supervised learning

	- Added in the `return_history` parameter to both the `fit` and `from_samples` methods, which will return the history callback as well as the fit model that will be used only in semi-supervised learning


BayesianNetwork
---------------

	- Modified the built keymap to be a numpy array of objects to prevent casting of all keys as the type of the first column.

Makefile
---------------

	- There is a new top-level "convenience" Makefile for development to make it easy to develop with two conda environments.  The default is for two conda environments, py2.7 and py3.6, but those could be overridden at run time with, for example, `make PY3_ENV=py3.6.2 biginstall`.  Targets exist for `install, test, bigclean, and nbtest` along with variations of each that first activate either one or both conda environments.  For example, `make biginstall` will install for both `py2.7` and `py3.6` environments.  When developing pomegranate, one frequently wants to do a fully clean build, wipe out all installed targets, and replace them.  This can be done with `make bigclean biguninstall biginstall`.  In addition, there is a target `nbtest` for testing all of the jupyter notebooks to ensure that the cells run.  See the Makefile for a list of additional conda packages to install for this to work.  The default is to stop on first error but you can run `make ALLOW_ERRORS=--allow-errors nbtest` to run all cells and then inspect the html output manually for errors.
	- There is a new top-level "convenience" rebuildconda script which will remove and create a conda environment for development.  Be careful using it that the environment you want to rebuild is the right one.  You can list environments with `conda info --envs`.  The default is to rebuild the `2.7` environment with name `py2.7`.  With this, you can create an alternative environment, test it out, and remove it as in `./rebuildconda 2.7.9 ; make PY2_ENV=py2.7.9 bigclean py2build py2test py2install nbtest ; source deactivate ; conda env remove --name py2.7.9`.

Version 0.9.0
=============

Highlights
----------

	- Missing value support has been added in for all models except factor graphs. This is done by included the string `nan` in string datasets, or `numpy.nan` in numeric datasets. Model fitting and inference is supported for all models for this. The technique is to not collect sufficient statistics from missing data, not to impute the missing values.

	- The unit testing suite has been greatly expanded, from around 140 tests to around 370 tests.

Changelog
---------

HiddenMarkovModel
-----------------

	- The documentation has been fixed so that states are defined as `State(NormalDistribution(0, 1))` instead of incorrectly as `State(Distribution(NormalDistribution(0, 1)))`
	
	- Fixed a bug in `from_samples` that was causing a TypeError if `name` was not specified when using `DiscreteDistribution` with custom labels.

	- Expanded the number of unit tests to include missing value support and be more comprehensive


Distributions
-------------
	
	- Multivariate Gaussian distributions have had their parameter updates simplified. This doesn't lead to a significant change in speed, just less code.

	- Fixed an issue where Poisson Distributions had an overflow issue caused when calculating large factorials by moving the log inside the product.

	- Fixed an issue where Poisson Distributions were not correctly calculating the probability of 0 counts.

	- Fixed an issue where Exponential Distribution would fail when fed integer 0-mode data.

	- Fixed an issue where IndependentComponentDistribution would have incorrect per-dimension weights after serialization.

	- Added in missing value support for fitting and log probability calculations for all univariate distributions, ICD, MGD, and CPTs through calculating sufficient statistics only on data that exists. The only distributions that currently do not support missing values are JointProbabilityTables and DirichletDistributions.

	- Fixed an issue with multivariate Gaussian distributions where the covariance matrix is no longer invertible with enough missing data by subtracting the smallest eigenvalue from the diagonal

K-Means
-------

	- Added in missing value support for k-means clustering by ignoring dimensions that are missing in the data. Can now fit and predict on missing data.

	- Added in missing value support for all initialization strategies

	- Added in a suite of unit tests

	- Added in the `distance` method that returns the distance between each point and each centroid

GeneralMixtureModel
-------------------

	- Added in missing value support for mixture models through updates to the distributions

	- Fixed an issue where passing in a list of distributions to `from_samples` along with a number of components did not produce a mixture of IndependentComponentsDistribution objects

	- Expanded the unit test suite and added tests for missing value support

BayesianNetwork
---------------

	- Vectorized the `predict_proba` method to take either a single sample or a list of samples

	- Changed the output of `predict_proba` to be individual symbols instead of a distribution where one symbol has a probability of 1 when fed in as known prior knowledge.

	- Added in an n_jobs parameter to parallelize the prediction of samples. This does not speed up a single sample, only a batch of samples.

	- Factored out `_check_input` into a function that be used independently

	- Added unit tests to check each of the above functions extensively

	- Missing value support added for the `log_probability`, `fit`, and `from_samples` methods. Chow-Liu trees are not supported for missing values, but using a constraint graph still works.


Version 0.8.1
=============

Highlights
----------

This will serve as a log for the changes added for the release of version 0.8.1.
	
	- Univariate offsets have been added to allow for distributions to be fit to a column of data rather than a vector of numbers. This stops the copying of data that had to be done previously.


Changelog
---------

Base
----
	
	- Parameters `column_idx` and `d` have been added to the `_summarize` method that all models expose. This is only useful for univariate distributions and models that fit univariate distributions and can be ignored by other models. The `column_idx` parameter specifies which column in a data matrix the distribution should be fit to, essentially serving as an offset. `d` refers to the number of dimensions that the data matrix has. This means that a univariate distribution will fit to all samples `i` such that `i*d + column_idx` in a pointer array. Multivariate distributions and models using those can ignore this.

	- A convenience function `to_yaml` was added to `State` and `Model` classes.  `YAML` is a superset of `JSON` that can be 4 to 5 times more compact.  You need the `yaml` package installed to use it.


Distributions
-------------

	- The `summarize` method has been moved from most individual distributions to the `Distribution` base object, as has the `fit` method. 

	- `min_std` has been moved from the `from_summaries` method and the `fit` method to the `__init__` method for the `NormalDistribution` and `LogNormalDistribution` objects.

NaiveBayes
----------

	- Moved the `fit` and `summarize` methods to `BayesModel` due to their similarity with BayesClassifier

BayesClassifier
---------------

	- Moved the `fit` and `summarize` methods to `BayesModel` due to their similarity to NaiveBayes


GeneralMixtureModel
-------------------

	- Fixed a bug where `n_jobs` was ignored in the `from_samples` method because `batch_size` was reset for the k-means initialization

HiddenMarkovModel
-----------------

	- The default name of a HiddenMarkovModel has been changed from "None" to "HiddenMarkovModel"


Version 0.8.0
=============

Highlights
----------

This will serve as a log for the changes added for the release of version 0.8.0.


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

	- Added in a reduce_dataset parameter to the `from_samples` method that will take in a dataset and create a new dataset that is the unique set of samples, weighted by their weighted occurrence in the dataset. Essentially, it takes a dataset that may have repeating members, and produces a new dataset that is entirely unique members. This produces an identically scoring Bayesian network as before, but all structure learning algorithms can be significantly sped up. This speed up is proportional to the redundancy of the dataset, so large datasets on a smallish (< 12) number of variables will see massive speed gains (sometimes even 2-3 orders of magnitude!) whereas past that it may not be beneficial. The redundancy of the dataset (and thus the speedup) can be estimated as n_samples / n_possibilities, where n_samples is the number of samples in the dataset and n_possibilities is the product of the number of unique keys per variable, or 2**d for binary data with d variables. It can be calculated exactly as n_samples / n_unique_samples, as many datasets are biased towards repeating elements. 

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

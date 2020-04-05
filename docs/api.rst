=======
The API
=======

pomegranate has a minimal core API that is made possible because all models are treated as a probability distribution regardless of complexity. Regardless of whether it's a simple probability distribution, or a hidden Markov model that uses a different probability distribution on each feature, these methods can be used. Each model documentation page has an API reference showing the full set of methods and parameters for each method, but generally all models have the following methods and parameters for the methods. 

.. code-block:: python

	>>> model.probability(X)

This method will take in either a single sample and return its probability, or a set of samples and return the probability of each one, given the model.

.. code-block:: python

	>>> model.log_probability(X)

The same as above but returns the log of the probability. This is helpful for numeric stability.

.. code-block:: python

	>>> model.fit(X, weights=None, inertia=0.0)

This will fit the model to the given data with optional weights. If called on a mixture model or a hidden Markov model this runs expectation-maximization to perform iterative updates, otherwise it uses maximum likelihood estimates. The shape of data should be (n, d) where n is the number of samples and d is the dimensionality, with weights being a vector of non-negative numbers of size (n,) when passed in. The inertia shows the proportion of the prior weight to use, defaulting to ignoring the prior values.

.. code-block:: python

	>>> model.summarize(X, weights=None)

This is the first step of the two step out-of-core learning API. It will take in a data set and optional weights and extract the sufficient statistics that allow for an exact update, adding to the cached values. If this is the first time that summarize is called then it will store the extracted values, if it's not the first time then the extracted values are added to those that have already been cached.

.. code-block:: python

	>>> model.from_summaries(inertia=0.0) 

This is the second step in the out-of-core learning API. It will used the extracted and aggregated sufficient statistics to derive exact parameter updates for the model. Afterwards it will reset the stored values.

.. code-block:: python

	>>> model.clear_summaries()

This method clears whatever summaries are left on the model without updating the parameters.

.. code-block:: python

	>>> Model.from_samples(X, weights=None)

This method will initialize a model to a data set. In the case of a simple distribution it will simply extract the parameters from the case. In the more complicated case of a Bayesian network it will jointly find the best structure and the best parameters given that structure. In the case of a hidden Markov model it will first find clusters and then learn a dense transition matrix.

Compositional Methods
---------------------

These methods are available for the compositional models, i.e., mixture models, hidden Markov models, Bayesian networks, naive Bayes classifiers, and Bayes' classifiers. These methods perform inference on the data. In the case of Bayesian networks it will use the forward-backward algorithm to make predictions on all variables for which values are not provided. For all other models, this will return the model component that yields the highest posterior P(M|D) for some sample. This value is calculated using Bayes' rule, where the likelihood of each sample given each component multiplied by the prior of that component is normalized by the likelihood of that sample given all components multiplied by the prior of those components. 

.. code-block:: python

	>>> model.predict(X)

This will return the most likely value for the data. In the case of Bayesian networks this is the most likely value that the variable takes given the structure of the network and the other observed values. In the other cases it is the model component that most likely explains this sample, such as the mixture component that a sample most likely falls under, or the class that is being predicted by a Bayes' classifier.

.. code-block:: python

	>>> model.predict_proba(X)

This returns the matrix of posterior probabilities P(M|D) directly. The predict method is simply running argmax over this matrix.

.. code-block:: python

	>>> model.predict_log_proba(X)

This returns the matrix of log posterior probabilities for numerical stability.

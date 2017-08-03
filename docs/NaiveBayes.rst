.. _naivebayes:

Bayes Classifiers and Naive Bayes
=================================

`IPython Notebook Tutorial <https://github.com/jmschrei/pomegranate/blob/master/tutorials/Tutorial_5_Bayes_Classifiers.ipynb>`_


Bayes classifiers are simple probabilistic classification models based off of Bayes theorem. See the above tutorial for a full primer on how they work, and what the distinction between a naive Bayes classifier and a Bayes classifier is. Essentially, each class is modeled by a probability distribution and classifications are made according to what distribution fits the data the best. They are a supervised version of general mixture models, in that the ``predict``, ``predict_proba``, and ``predict_log_proba`` methods return the same values for the same underlying distributions, but that instead of using expectation-maximization to fit to new data they can use the provided labels directly.

Initialization
--------------

Bayes classifiers and naive Bayes can both be initialized in one of two ways depending on if you know the parameters of the model beforehand or not, (1) passing in a list of pre-initialized distributions to the model, or (2) using the ``from_samples`` class method to initialize the model directly from data. For naive Bayes models on multivariate data, the pre-initialized distributions must be a list of ``IndependentComponentDistribution`` objects since each dimension is modeled independently from the others. For Bayes classifiers on multivariate data a list of any type of multivariate distribution can be provided. For univariate data the two models produce identical results, and can be passed in a list of univariate distributions. For example:

.. code-block:: python

	from pomegranate import *
	d1 = IndependentComponentsDistribution([NormalDistribution(5, 2), NormalDistribution(6, 1), NormalDistribution(9, 1)])
	d2 = IndependentComponentsDistribution([NormalDistribution(2, 1), NormalDistribution(8, 1), NormalDistribution(5, 1)])
	d3 = IndependentComponentsDistribution([NormalDistribution(3, 1), NormalDistribution(5, 3), NormalDistribution(4, 1)])
	model = NaiveBayes([d1, d2, d3])

would create a three class naive Bayes classifier that modeled data with three dimensions. Alternatively, we can initialize a Bayes classifier in the following manner 

.. code-block:: python

	from pomegranate import *
	d1 = MultivariateGaussianDistribution([5, 6, 9], [[2, 0, 0], [0, 1, 0], [0, 0, 1]])
	d2 = MultivariateGaussianDistribution([2, 8, 5], [[1, 0, 0], [0, 1, 0], [0, 0, 1]])
	d3 = MultivariateGaussianDistribution([3, 5, 4], [[1, 0, 0], [0, 3, 0], [0, 0, 1]])
	model = BayesClassifier([d1, d2, d3])

The two examples above functionally create the same model, as the Bayes classifier uses multivariate Gaussian distributions with the same means and a diagonal covariance matrix containing only the variances. However, if we were to fit these models to data later on, the Bayes classifier would learn a full covariance matrix while the naive Bayes would only learn the diagonal.

If we instead wish to initialize our model directly onto data, we use the ``from_samples`` class method.

.. code-block:: python
	
	from pomegranate import *
	import numpy
	X = numpy.load('data.npy')
	y = numpy.load('labels.npy')
	model = NaiveBayes.from_samples(NormalDistribution, X, y)

This would create a naive Bayes model directly from the data with normal distributions modeling each of the dimensions, and a number of components equal to the number of classes in ``y``. Alternatively if we wanted to create a model with different distributions for each dimension we can do the following:

.. code-block:: python

	model = NaiveBayes.from_samples([NormalDistribution, ExponentialDistribution], X, y)

This assumes that your data is two dimensional and that you want to model the first distribution as a normal distribution and the second dimension as an exponential distribution.

We can do pretty much the same thing with Bayes classifiers, except passing in a more complex model.

.. code-block:: python
	
	model = BayesClassifier.from_samples(MultivariateGaussianDistribution, X, y)

One can use much more complex models than just a multivariate Gaussian with a full covariance matrix when using a Bayes classifier. Specifically, you can also have your distributions be general mixture models, hidden Markov models, and Bayesian networks. For example:

.. code-block:: python
	
	model = BayesClassifier.from_samples(BayesianNetwork, X, y)

That would require that the data is only discrete valued currently, and the structure learning task may be too long if not set appropriately. However, it is possible. Currently, one cannot simply put in GeneralMixtureModel or HiddenMarkovModel despite them having a ``from_samples`` method because there is a great deal of flexibility in terms of the structure or emission distributions. The easiest way to set up one of these more complex models is to build each of the components separately and then feed them into the Bayes classifier method using the first initialization method.

.. code-block:: python

	d1 = GeneralMixtureModel.from_samples(MultivariateGaussianDistribution, n_components=5, X=X[y==0])
	d2 = GeneralMixtureModel.from_samples(MultivariateGaussianDistribution, n_components=5, X=X[y==1])
	model = BayesClassifier([d1, d2]) 

Prediction
----------

Bayes classifiers and naive Bayes supports the same three prediction methods that the other models support, ``predict``, ``predict_proba``, and ``predict_log_proba``. These methods return the most likely class given the data (argmax_m P(M|D)), the probability of each class given the data (P(M|D)), and the log probability of each class given the data (log P(M|D)). It is best to always pass in a 2D matrix even for univariate data, where it would have a shape of (n, 1). 

The ``predict`` method takes in samples and returns the most likely class given the data.

.. code-block:: python
	
	from pomegranate import *
	model = NaiveBayes([NormalDistribution(5, 2), UniformDistribution(0, 10), ExponentialDistribution(1.0)])
	model.predict( np.array([[0], [1], [2], [3], [4]]))
	[2, 2, 2, 0, 0]

Calling ``predict_proba`` on five samples for a Naive Bayes with univariate components would look like the following.

.. code-block:: python

	from pomegranate import *
	model = NaiveBayes([NormalDistribution(5, 2), UniformDistribution(0, 10), ExponentialDistribution(1)])
	model.predict_proba(np.array([[0], [1], [2], [3], [4]]))
	[[ 0.00790443  0.09019051  0.90190506]
	 [ 0.05455011  0.20207126  0.74337863]
	 [ 0.21579499  0.33322883  0.45097618]
	 [ 0.44681566  0.36931382  0.18387052]
	 [ 0.59804205  0.33973357  0.06222437]]

Multivariate models work the same way.

.. code-block:: python

	from pomegranate import *
	d1 = MultivariateGaussianDistribution([5, 5], [[1, 0], [0, 1]])
	d2 = IndependentComponentsDistribution([NormalDistribution(5, 2), NormalDistribution(5, 2)])
	model = BayesClassifier([d1, d2])
	clf.predict_proba(np.array([[0, 4],
							 	    [1, 3],
								    [2, 2],
								    [3, 1],
								    [4, 0]]))
	array([[ 0.00023312,  0.99976688],
	       [ 0.00220745,  0.99779255],
	       [ 0.00466169,  0.99533831],
	       [ 0.00220745,  0.99779255],
	       [ 0.00023312,  0.99976688]])

``predict_log_proba`` works the same way, returning the log probabilities instead of the probabilities.

Fitting
-------

Both naive Bayes and Bayes classifiers also have a ``fit`` method that updates the parameters of the model based on new data. The major difference between these methods and the others presented is that these are supervised methods and so need to be passed labels in addition to data. This change propagates also to the ``summarize`` method, where labels are provided as well.

.. code-block:: python

	from pomegranate import *
	d1 = MultivariateGaussianDistribution([5, 5], [[1, 0], [0, 1]])
	d2 = IndependentComponentsDistribution(NormalDistribution(5, 2), NormalDistribution(5, 2)])
	model = BayesClassifier([d1, d2])
	X = np.array([[6.0, 5.0],
		    		  [3.5, 4.0],
			    	  [7.5, 1.5],
				      [7.0, 7.0 ]])
	y = np.array([0, 0, 1, 1])
	model.fit(X, y)

As we can see, there are four samples, with the first two samples labeled as class 0 and the last two samples labeled as class 1. Keep in mind that the training samples must match the input requirements for the models used. So if using a univariate distribution, then each sample must contain one item. A bivariate distribution, two. For hidden markov models, the sample can be a list of observations of any length. An example using hidden markov models would be the following.

.. code-block:: python
	
	d1 = HiddenMarkovModel...
	d2 = HiddenMarkovModel...
	d3 = HiddenMarkovModel...
	model = BayesClassifier([d1, d2, d3])
	X = np.array([list('HHHHHTHTHTTTTH'),
					   	    list('HHTHHTTHHHHHTH'),
					  	    list('TH'), 
					  	    list('HHHHT')])
	y = np.array([2, 2, 1, 0])
	model.fit(X, y)

API Reference
-------------

.. automodule:: pomegranate.NaiveBayes
	:members:
	:inherited-members:

.. automodule:: pomegranate.BayesClassifier
	:members:
	:inherited-members:

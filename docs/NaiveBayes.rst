.. _naivebayes:

Naive Bayes Classifiers
=======================

`IPython Notebook Tutorial <https://github.com/jmschrei/pomegranate/blob/master/tutorials/Tutorial_5_Naive_Bayes.ipynb>`_

The Naive Bayes classifier is a simple probabilistic classification model based on Bayes Theorem. Since Naive Bayes classifiers classifies sets of data by which class has the highest conditional probability, Naive Bayes classifiers can use any distribution or model which has a probabilistic interpretation of the data as one of its components. Basically if it can output a log probability, then it can be used in Naive Bayes.

An IPython notebook example demonstrating a Naive Bayes classifier using multivariate distributions can be `found here <https://github.com/jmschrei/pomegranate/blob/master/examples/naivebayes_multivariate_male_female.ipynb>`_.

Initialization
--------------

Naive Bayes can be initialized in two ways, either by (1) passing in pre-initialized models as a list, or by (2) passing in the constructor and the number of components for simple distributions. For example, here is how you can create a Naive bayes classifier which compares a normal distribution to a uniform distribution to an exponential distribution:

.. code-block:: python

	>>> from pomegranate import *
	>>> clf = NaiveBayes([ NormalDistribution(5, 2), UniformDistribution(0, 10), ExponentialDistribution(1) ])

An advantage of initializing the classifier this way is that you can use pre-trained or known-before-hand models to make predictions. A disadvantage is that if we don't have any prior knowledge as to what the distributions should be then we have to make up distributions to start off with. If all of the models in the classifier use the same type of model then we can pass in the constructor for that model and the number of classes that there are.

.. code-block:: python

	>>> from pomegranate import *
	>>> clf = NaiveBayes(NormalDistribution, n_components=5)

.. warning::
	If we initialize a naive Bayes classifier in this manner we must fit the model before we can use it to predict.

An advantage of doing it this way is that we don't need to make dummy distributions just to train, but a disadvantage is that we have to train the model before we can use it.

Since Naive Bayes classifiers simply compares the likelihood of a sample occurring under different models, it can be initialized with any model in pomegranate. This is assuming that all the models take the same type of input.

.. code-block:: python

	>>> from pomegranate import *
	>>> d1 = MultivariateGaussianDistribution([5, 5], [[1, 0], [0, 1]])
	>>> d2 = IndependentComponentsDistribution([NormalDistribution(5, 2), NormalDistribution(5, 2)])
	>>> clf = NaiveBayes([d1, d2])

.. note::
	This is no longer strictly a "naive" Bayes classifier if we are using more complicated models. However, much of the underlying math still holds.

Prediction
----------

Naive Bayes supports the same three prediction methods that the other models support, namely ``predict``, ``predict_proba``, and ``predict_log_proba``. These methods return the most likely class given the data, the probability of each class given the data, and the log probability of each class given the data.

The ``predict`` method takes in samples and returns the most likely class given the data.

.. code-block:: python
	
	>>> from pomegranate import *
	>>> clf = NaiveBayes([ NormalDistribution( 5, 2 ), UniformDistribution( 0, 10 ), ExponentialDistribution( 1.0 ) ])
	>>> clf.predict( np.array([ 0, 1, 2, 3, 4 ]) )
	[ 2, 2, 2, 0, 0 ]


Calling ``predict_proba`` on five samples for a Naive Bayes with univariate components would look like the following.

.. code-block:: python

	>>> from pomegranate import *
	>>> clf = NaiveBayes([NormalDistribution(5, 2), UniformDistribution(0, 10), ExponentialDistribution(1)])
	>>> clf.predict_proba(np.array([ 0, 1, 2, 3, 4]))
	[[ 0.00790443  0.09019051  0.90190506]
	 [ 0.05455011  0.20207126  0.74337863]
	 [ 0.21579499  0.33322883  0.45097618]
	 [ 0.44681566  0.36931382  0.18387052]
	 [ 0.59804205  0.33973357  0.06222437]]


Multivariate models work the same way except that the input has to have the same number of columns as are represented in the model, like the following.

.. code-block:: python

	>>> from pomegranate import *
	>>> d1 = MultivariateGaussianDistribution([5, 5], [[1, 0], [0, 1]])
	>>> d2 = IndependentComponentsDistribution([NormalDistribution(5, 2), NormalDistribution(5, 2)])
	>>> clf = NaiveBayes([d1, d2])
	>>> clf.predict_proba(np.array([[0, 4],
							 	    [1, 3],
								    [2, 2],
								    [3, 1],
								    [4, 0]]))
	array([[ 0.00023312,  0.99976688],
	       [ 0.00220745,  0.99779255],
	       [ 0.00466169,  0.99533831],
	       [ 0.00220745,  0.99779255],
	       [ 0.00023312,  0.99976688]])


``predict_log_proba`` works in a similar way except that it returns the log probabilities instead of the actual probabilities.

Fitting
-------

Naive Bayes has a fit method, in which the models in the classifier are trained to "fit" to a set of data. The method takes two numpy arrays as input, an array of samples and an array of correct classifications for each sample. Here is an example for a Naive Bayes made up of two bivariate distributions.

.. code-block:: python

	>>> from pomegranate import *
	>>> d1 = MultivariateGaussianDistribution([5, 5], [[1, 0], [0, 1]])
	>>> d2 = IndependentComponentsDistribution(NormalDistribution(5, 2), NormalDistribution(5, 2)])
	>>> clf = NaiveBayes([d1, d2])
	>>> X = np.array([[6.0, 5.0],
		    		  [3.5, 4.0],
			    	  [7.5, 1.5],
				      [7.0, 7.0 ]])
	>>> y = np.array([0, 0, 1, 1])
	>>> clf.fit(X, y)

As we can see, there are four samples, with the first two samples labeled as class 0 and the last two samples labeled as class 1. Keep in mind that the training samples must match the input requirements for the models used. So if using a univariate distribution, then each sample must contain one item. A bivariate distribution, two. For hidden markov models, the sample can be a list of observations of any length. An example using hidden markov models would be the following.

.. code-block:: python
	
	>>> X = np.array([list( 'HHHHHTHTHTTTTH' ),
					   	    list( 'HHTHHTTHHHHHTH' ),
					  	    list( 'TH' ), 
					  	    list( 'HHHHT' )])
	>>> y = np.array([2, 2, 1, 0])
	>>> clf.fit(X, y)

API Reference
-------------

.. automodule:: pomegranate.NaiveBayes
	:members:
	:inherited-members:



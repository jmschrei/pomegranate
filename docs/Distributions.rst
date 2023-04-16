.. _distributions:

Probability Distributions
=========================

..`IPython Notebook Tutorial <https://github.com/jmschrei/pomegranate/blob/master/tutorials/B_Model_Tutorial_1_Distributions.ipynb>`_

Simple probability distributions, such as normal or exponential distributions, are commonly used as components as more complex models, such as mixtures and hidden Markov models, but they can also be used by themselves. Because everything in pomegranate has the same API, you can use these simple probability distributions in the same situations as the more complex models. 

.. NOTE::
    PyTorch has its own built in library of probability distributions but these distributions cannot be fit to data. I would have loved to build upon that library and use it as a base for pomegranate, but fitting distributions is a critical part of the package.

**Distributions**

.. currentmodule:: pomegranate.distributions

.. autosummary::

    Bernoulli
    Categorical
    ConditionalCategorical
    JointCategorical
    DiracDelta
    Exponential
    Gamma
    LogNormal
    Normal
    Poisson
    StudentT
    Uniform
    ZeroInflated


Initialization
--------------

Initializing a distribution can be done in one of two ways. The first way involves passing in parameters when you know them. For example, the parameters of a normal distribution are the mean (mu) and the variance (sigma**2) (Note: not the mean and standard deviation). We can initialize it as follows:

.. code-block:: python

    from pomegranate.distributions import Normal
    a = Normal([5.0], [2.0])

However, frequently we do not know the parameters of the distribution beforehand and would like to learn a distribution directly from data. Because pomegranate is following the scikit-learn API, this is done by not passing in any parameters at initialization (though you can still pass in the learning hyperparameters, e.g., ``max_iterations``) and then calling the ``fit`` method. 

.. code-block:: python

    b = Normal().fit([[1.0], [2.0], [0.0]])

Note that all distributions in pomegranate are capable of handling multivariate distribution and so the data that they are fit to must be in 2D or 3D form (usually 2D), with the examples being the rows and the variables being the columns.

If we want to fit the model to weighted samples, we can just pass in an array of the relative weights of each sample as well.

.. code-block:: python

    b = Normal().fit([[3], [4], [5], [6], [7]], sample_weight=[0.5, 1, 1.5, 1, 0.5])

Probability
-----------

Probability distributions are typically used to evaluate the probability of observations/examples. This can be done using the ``probability`` and ``log_probability`` methods that all distributions, and all other probabilistic models in pomegranate, have. The examples that are evaluated must be 2D or 3D, depending on the distribution, even when you only have one example and/or one variable.

.. code-block:: python

    >>> a = NormalDistribution([5.0], [4.0], covariance_type='diag')
    >>> a.log_probability([[8.0]])
    tensor([-2.7371])
    >>> a.probability([[8.0]])
    tensor([0.0648])
    >>> b = Normal(covariance_type='diag')
    >>> b.fit([[3.0], [4.0], [5.0], [6.0], [7.0]], sample_weight=[0.5, 1, 1.5, 1, 0.5])
    >>> b.log_probability([[8.0]])
    tensor([-4.4378])



API Reference
-------------

.. automodule:: pomegranate.distributions
   :members: Bernoulli, Categorical, ConditionalCategorical, JointCategorical, DiracDelta, Exponential, Gamma, LogNormal, Normal, Poisson, StudentT, Uniform, ZeroInflated
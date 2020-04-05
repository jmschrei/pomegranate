.. _distributions:

Probability Distributions
=========================

`IPython Notebook Tutorial <https://github.com/jmschrei/pomegranate/blob/master/tutorials/B_Model_Tutorial_1_Distributions.ipynb>`_

While probability distributions are frequently used as components of more complex models such as mixtures and hidden Markov models, they can also be used by themselves. Many data science tasks require fitting a distribution to data or generating samples under a distribution. pomegranate has a large library of both univariate and multivariate distributions which can be used with an intuitive interface.

**Univariate Distributions**

.. currentmodule:: pomegranate.distributions

.. autosummary::

    UniformDistribution
    BernoulliDistribution
    NormalDistribution
    LogNormalDistribution
    ExponentialDistribution
    PoissonDistribution
    BetaDistribution
    GammaDistribution
    DiscreteDistribution

**Kernel Densities**

.. autosummary::

    GaussianKernelDensity
    UniformKernelDensity
    TriangleKernelDensity

**Multivariate Distributions**

.. autosummary::
    
    IndependentComponentsDistribution
    MultivariateGaussianDistribution
    DirichletDistribution
    ConditionalProbabilityTable
    JointProbabilityTable

While there are a large variety of univariate distributions, multivariate distributions can be made from univariate distributions by using ```IndependentComponentsDistribution``` with the assumption that each column of data is independent from the other columns (instead of being related by a covariance matrix, like in multivariate gaussians). Here is an example:

.. code-block:: python

    d1 = NormalDistribution(5, 2)
    d2 = LogNormalDistribution(1, 0.3)
    d3 = ExponentialDistribution(4)
    d = IndependentComponentsDistribution([d1, d2, d3])

Use MultivariateGaussianDistribution when you want the full correlation matrix within the feature vector. When you want a strict diagonal correlation (i.e no correlation or "independent"), this is achieved using IndependentComponentsDistribution with NormalDistribution for each feature. There is no implementation of spherical or other variations of correlation.

Initialization
--------------

Initializing a distribution is simple and done just by passing in the distribution parameters. For example, the parameters of a normal distribution are the mean (mu) and the standard deviation (sigma). We can initialize it as follows:

.. code-block:: python

    from pomegranate import *
    a = NormalDistribution(5, 2)

However, frequently we don't know the parameters of the distribution beforehand or would like to directly fit this distribution to some data. We can do this through the `from_samples` class method.

.. code-block:: python

    b = NormalDistribution.from_samples([3, 4, 5, 6, 7])

If we want to fit the model to weighted samples, we can just pass in an array of the relative weights of each sample as well.

.. code-block:: python

    b = NormalDistribution.from_samples([3, 4, 5, 6, 7], weights=[0.5, 1, 1.5, 1, 0.5])

Probability
-----------

Distributions are typically used to calculate the probability of some sample. This can be done using either the `probability` or `log_probability` methods.

.. code-block:: python

    >>> a = NormalDistribution(5, 2)
    >>> a.log_probability(8)
    -2.737085713764219
    >>> a.probability(8)
    0.064758797832971712
    >>> b = NormalDistribution.from_samples([3, 4, 5, 6, 7], weights=[0.5, 1, 1.5, 1, 0.5])
    >>> b.log_probability(8)
    -4.437779569430167

These methods work for univariate distributions, kernel densities, and multivariate distributions all the same. For a multivariate distribution you'll have to pass in an array for the full sample.

.. code-block:: python
    
    >>> d1 = NormalDistribution(5, 2)
    >>> d2 = LogNormalDistribution(1, 0.3)
    >>> d3 = ExponentialDistribution(4)
    >>> d = IndependentComponentsDistribution([d1, d2, d3])
    >>>
    >>> X = [6.2, 0.4, 0.9]
    >>> d.log_probability(X)
    -23.205411733352875

Fitting
-------

We may wish to fit the distribution to new data, either overriding the previous parameters completely or moving the parameters to match the dataset more closely through inertia. Distributions are updated using maximum likelihood estimates (MLE). Kernel densities will either discard previous points or downweight them if inertia is used.

.. code-block:: python

    d = NormalDistribution(5, 2)
    d.fit([1, 5, 7, 3, 2, 4, 3, 5, 7, 8, 2, 4, 6, 7, 2, 4, 5, 1, 3, 2, 1])
    d
    {
        "frozen" :false,
        "class" :"Distribution",
        "parameters" :[
            3.9047619047619047,
            2.13596776114341
        ],
        "name" :"NormalDistribution"
    }

Training can be done on weighted samples by passing an array of weights in along with the data for any of the training functions, like the following:

.. code-block:: python

    d = NormalDistribution(5, 2)
    d.fit([1, 5, 7, 3, 2, 4], weights=[0.5, 0.75, 1, 1.25, 1.8, 0.33])
    d
    {
        "frozen" :false,
        "class" :"Distribution",
        "parameters" :[
            3.538188277087034,
            1.954149818564894
        ],
        "name" :"NormalDistribution"
    }

Training can also be done with inertia, where the new value will be some percentage the old value and some percentage the new value, used like `d.from_samples([5,7,8], inertia=0.5)` to indicate a 50-50 split between old and new values. 

API Reference
-------------

.. automodule:: pomegranate.distributions
   :members: BernoulliDistribution,BetaDistribution,ConditionalProbabilityTable,DirichletDistribution,DiscreteDistribution,ExponentialDistribution,GammaDistribution,IndependentComponentsDistribution,JointProbabilityTable,KernelDensities,LogNormalDistribution,MultivariateGaussianDistribution,NormalDistribution,PoissonDistribution,UniformDistribution

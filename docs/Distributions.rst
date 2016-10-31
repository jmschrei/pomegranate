.. _distributions:

Probability Distributions
=========================

[IPython Notebook Tutorial](https://github.com/jmschrei/pomegranate/blob/master/tutorials/Tutorial_1_Distributions.ipynb)

The probability distribution is the simplest probabilistic modelling component. While these are frequently used as components of more complex models such as General Mixture Models or Hidden Markov Models, they can also be used by themselves. Many data science tasks require calculating the probability of samples under a distribution or fitting a distribution to data and using the parameters. pomegranate has a large library of probability distributions and kernel densities which can also be combined to form multivariate or mixture distributions. 

Here is a full list of currently implemented distributions:

**Univariate Distributions**

(1) UniformDistribution
(2) BernoulliDistribution
(3) NormalDistribution
(4) LogNormalDistribution
(5) ExponentialDistribution
(6) PoissonDistribution
(7) BetaDistribution
(8) GammaDistribution
(9) DiscreteDistribution

**Kernel Densities**

(10) GaussianKernelDensity
(11) UniformKernelDensity
(12) TriangleKernelDensity

**Multivariate Distributions**

(13) IndependentComponentsDistribution
(14) MultivariateGaussianDistribution
(15) DirichletDistribution
(16) ConditionalProbabilityTable
(17) JointProbabilityTable

Initialization
--------------

A widely used model is the Normal distribution.

```python
>>> from pomegranate import *
>>> a = NormalDistribution(5, 2)
```

If we don't know the parameters of the normal distribution beforehand, we can learn them from data using the class method `from_samples`.

```python
>>> b = NormalDistribution.from_samples([3, 4, 5, 6, 7], weights=[0.5, 1, 1.5, 1, 0.5])
```

We can initialize kernel densities by passing in a list of points and their respective weights (equal weighting if no weights are explicitly passed in) like the following:

```python
>>> c = TriangleKernelDensity([1,5,2,3,4], weights=[4,2,6,3,1])
```

Next, we can try to make a mixture of distributions. We can make a mixture of arbitrary distributions with arbitrary weights. Usually people will make mixtures of the same type of distributions, such as a mixture of Gaussians, but this is more general than that. You do this by just passing in a list of initialized distribution objects and their associated weights. For example, we could create a mixture of a normal distribution and an exponential distribution like below. This probably doesn't make sense, but you can still do it.

```python
>>> d = MixtureDistribution([NormalDistribution(2, 4), ExponentialDistribution(8)], weights=[1, 0.01])
```

Log Probability
---------------

Distributions can calculate the log probability of a point under the parameters of the distribution. This is done using the `log_probability` method.

```python
>>> a.log_probability(8)
-2.737085713764219
>>> b.log_probability(8)
-4.437779569430167
```

Since all types of distributions use the same log probability method, we can do the same for the triangle kernel density as well.

```python
>>> print c.log_probability(8)
-inf
>>> print c.log_probability(5.5)
-2.772588722239781
```

This will return -inf because there is no density at 8--no initial samples have been put down.  

Lastly, we can use the same method for a mixture distribution:

```python
print d.log_probability(8)
-3.440183225177332
```

Predictions
-----------

Distributions by themselves do not have any prediction methods at this time.

Fitting
-------

Distributions can be updated using MLE estimates on data or weighted data. Kernel densities will discard previous points and add in the new points, while MixtureDistributions will perform expectation-maximization to update the mixture of distributions.

```python
>>> d.fit([1, 5, 7, 3, 2, 4, 3, 5, 7, 8, 2, 4, 6, 7, 2, 4, 5, 1, 3, 2, 1])
>>> d
{
    "frozen" : false,
    "class" : "Distribution",
    "parameters" : [
        [
            {
                "frozen" : false,
                "class" : "Distribution",
                "parameters" : [
                    3.916017859766298,
                    2.1324213594169596
                ],
                "name" : "NormalDistribution"
            },
            {
                "frozen" : false,
                "class" : "Distribution",
                "parameters" : [
                    0.9995546120033824
                ],
                "name" : "ExponentialDistribution"
            }
        ],
        [
            0.9961393668380545,
            0.003860633161945405
        ]
    ],
    "name" : "MixtureDistribution"
}
``` 

Distributions can also be trained in an out-of-core manner by storing sufficient statistics calculated on each batch to get exact updates. This is done using the `summarize` method on a minibatch of the data, and then `from_summaries` when you want to update the parameters of the data.

```python
>>> d = MixtureDistribution( [ NormalDistribution( 2, 4 ), ExponentialDistribution( 8 ) ], weights=[1, 0.01] )
>>> d.summarize([1, 5, 7, 3, 2, 4, 3])
>>> d.summarize([5, 7, 8])
>>> d.summarize([2, 4, 6, 7, 2, 4, 5, 1, 3, 2, 1])
>>> d.from_summaries()
>>> d
{
    "frozen" : false,
    "class" : "Distribution",
    "parameters" : [
        [
            {
                "frozen" : false,
                "class" : "Distribution",
                "parameters" : [
                    3.916017859766298,
                    2.1324213594169588
                ],
                "name" : "NormalDistribution"
            },
            {
                "frozen" : false,
                "class" : "Distribution",
                "parameters" : [
                    0.9995546120033824
                ],
                "name" : "ExponentialDistribution"
            }
        ],
        [
            0.997075329955115,
            0.0029246700448848878
        ]
    ],
    "name" : "MixtureDistribution"
}

```

This is a scalable way of training on millions or billions of samples without needing to store them in memory. 

Training can be done on weighted samples by passing an array of weights in along with the data for any of the training functions, such as `d.summarize([5,7,8], weights=[1,2,3])`. Training can also be done with inertia, where the new value will be some percentage the old value and some percentage the new value, used like `d.from_sample([5,7,8], inertia=0.5)` to indicate a 50-50 split between old and new values. 

API Reference
-------------

.. automodule:: pomegranate.distributions
	:members: Distribution

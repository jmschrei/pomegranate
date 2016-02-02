Probability Distributions
========================

The probability distribution is one of the simplest probabilistic models used. While these are frequently used as parts of more complex models such as General Mixture Models or Hidden Markov Models, they can also be used by themselves. Many simple analyses require just calculating the probability of samples under a distribution, or fitting a distribution to data and seeing what the distribution parameters are. pomegranate has a large library of probability distributions, kernel densities, and the ability to combine these to form multivariate or mixture distributions. 

An IPython notebook tutorial with visualizations can be [found here](https://github.com/jmschrei/pomegranate/blob/master/tutorials/Tutorial_1_Distributions.ipynb)

Here is a full list of currently implemented distributions:

```
UniformDistribution
NormalDistribution
LogNormalDistribution
ExponentialDistribution
BetaDistribution
GammaDistribution
DiscreteDistribution
LambdaDistribution
GaussianKernelDensity
UniformKernelDensity
TriangleKernelDensity
IndependentComponentsDistribution
MultivariateGaussianDistribution
ConditionalProbabilityTable
JointProbabilityTable
```

All distribution objects have the same methods:

```
copy() : Make a deep copy of the distribution
freeze() : Prevent the distribution from updating on training calls
thaw() : Reallow the distribution to update on training calls
log_probability( symbol ): Return the log probability of the symbol under the distribution
sample() : Return a randomly generated sample from the distribution
fit / train / from_sample( items, weights=None, inertia=None ) : Update the parameters of the distribution
summarize( items, weights=None ) : Store sufficient statistics of a dataset for a future update
from_summaries( inertia=0.0 ) : Update the parameters of the distribution from the sufficient statistics
to_json() : Return a json formatted string representing the distribution
from_json( s ) : Build an appropriate distribution object from the string
```

## Initialization

A widely used model is the Normal distribution. We can easily create one, specifying the parameters if we know them.

```python
from pomegranate import *

a = NormalDistribution(5, 2)
```

If we don't know the parameters of the normal distribution beforehand, we can learn them from data using the class method `from_samples`.

```python
b = NormalDistribution.from_samples([3, 4, 5, 6, 7], weights=[0.5, 1, 1.5, 1, 0.5])
```

We can initialize kernel densities by passing in a list of points and their respective weights (equal weighting if no weights are explicitly passed in) like the following:

```python
c = TriangleKernelDensity([1,5,2,3,4], weights=[4,2,6,3,1])
```

Next, we can try to make a mixture of distributions. We can make a mixture of arbitrary distributions with arbitrary weights. Usually people will make mixtures of the same type of distributions, such as a mixture of Gaussians, but this is more general than that. You do this by just passing in a list of initialized distribution objects and their associated weights. For example, we could create a mixture of a normal distribution and an exponential distribution like below. This probably doesn't make sense, but you can still do it.

```python
d = MixtureDistribution([NormalDistribution(2, 4), ExponentialDistribution(8)], weights=[1, 0.01])
```

## Prediction

The only prediction step which a distribution has is calculating the log probability of a point under the parameters of the distribution. This is done using the `log_probability` method

```python
a.log_probability(8) # This will return -2.737
b.log_probability(8) # This will also return -2.737 as 'a' and 'b' are the same distribution
```

Since all types of distributions use the same log probability method, we can do the same for the triangle kernel density.

```
print c.log_probability(8)
```

This will return -inf because there is no density at 8--no initial samples have been put down.  

We can then evaluate the mixture distribution:

```
print d.log_probability(8)
```

This should return -3.44.   

## Fitting

We can also update these distributions using Maximum Likelihood Estimates for the new values. Kernel densities will discard previous points and add in the new points, while MixtureDistributions will perform expectation-maximization to update the mixture of distributions.

```python
d.from_sample([1, 5, 7, 3, 2, 4, 3, 5, 7, 8, 2, 4, 6, 7, 2, 4, 5, 1, 3, 2, 1])
print d
```

This should result in `MixtureDistribution( [NormalDistribution(3.916, 2.132), ExponentialDistribution(0.99955)], [0.9961, 0.00386] )`. 

In addition to training on a batch of data which can be held in memory, all distributions can be trained out-of-core (online) using summary statistics and still get exact updates. This is done using the `summarize` method on a minibatch of the data, and then `from_summaries` when you want to update the parameters of the data.

```python
d = MixtureDistribution( [ NormalDistribution( 2, 4 ), ExponentialDistribution( 8 ) ], weights=[1, 0.01] )
d.summarize([1, 5, 7, 3, 2, 4, 3])
d.summarize([5, 7, 8])
d.summarize([2, 4, 6, 7, 2, 4, 5, 1, 3, 2, 1])
d.from_summaries()
```

Splitting up the data into batches will still give an exact answer, but allows for out of core training of distributions on massive amounts of data. 

In addition, training can be done on weighted samples by passing an array of weights in along with the data for any of the training functions, such as `d.summarize([5,7,8], weights=[1,2,3])`. Training can also be done with inertia, where the new value will be some percentage the old value and some percentage the new value, used like `d.from_sample([5,7,8], inertia=0.5)` to indicate a 50-50 split between old and new values. 

Probability Distributions
========================

The probability distribution is one of the simplest probabilistic models used. While these are frequently used as parts of more complex models such as General Mixture Models or Hidden Markov Models, they can also be used by themselves. Many simple analyses require just calculating the probability of samples under a distribution, or fitting a distribution to data and seeing what the distribution parameters are. pomegranate has a large library of probability distributions, kernel densities, and the ability to combine these to form multivariate or mixture distributions. 

All distribution objects have the same methods:

```
<b>Distribution.from_sample(sample, weight):</b> Create a new distribution parameterized using weighted MLE estimates on data
<b>d.log_probability(sample):</b> Calculate the log probability of this point under the distribution
<b>d.sample():</b> Return a randomly sample from this distribution
<b>d.fit(sample, weight, inertia):</b> Fit the parameters of the distribution using weighted MLE estimates on the data with inertia as a regularizer
```

A widely used model is the Normal distribution. We can easily create one, specifying the parameters if we know them.

```python
from pomegranate import *

a = NormalDistribution(5, 2)
print a.log_probability(8)
```

This will return -2.737, which is the log probability of 8 under that Normal Distribution.


```
b = TriangleKernelDensity( [1,5,2,3,4], weights=[4,2,6,3,1] )
c = MixtureDistribution( [ NormalDistribution( 2, 4 ), ExponentialDistribution( 8 ) ], weights=[1, 0.01] )

print a.log_probability( 8 )
print b.log_probability( 8 )
print c.log_probability( 8 )
```

This should return `-2.737`, `-inf`, and `-3.44` respectively.  

We can also update these distributions using Maximum Likelihood Estimates for the new values. Kernel densities will discard previous points and add in the new points, while MixtureDistributions will perform expectation-maximization to update the mixture of distributions.

```python
c.from_sample([1, 5, 7, 3, 2, 4, 3, 5, 7, 8, 2, 4, 6, 7, 2, 4, 5, 1, 3, 2, 1])
print c
```

This should result in `MixtureDistribution( [NormalDistribution(3.916, 2.132), ExponentialDistribution(0.99955)], [0.9961, 0.00386] )`. All distributions can be trained either as a batch using `from_sample`, or using summary statistics using `summarize` on lists of numbers until all numbers have been fed in, and then `from_summaries` like in the following example which produces the same result:

```python
c = MixtureDistribution( [ NormalDistribution( 2, 4 ), ExponentialDistribution( 8 ) ], weights=[1, 0.01] )
c.summarize([1, 5, 7, 3, 2, 4, 3])
c.summarize([5, 7, 8])
c.summarize([2, 4, 6, 7, 2, 4, 5, 1, 3, 2, 1])
c.from_summaries()
```

Splitting up the data into batches will still give an exact answer, but allows for out of core training of distributions on massive amounts of data. 

In addition, training can be done on weighted samples by passing an array of weights in along with the data for any of the training functions, such as `c.summarize([5,7,8], weights=[1,2,3])`. Training can also be done with inertia, where the new value will be some percentage the old value and some percentage the new value, used like `c.from_sample([5,7,8], inertia=0.5)` to indicate a 50-50 split between old and new values. 

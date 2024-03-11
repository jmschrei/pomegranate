<img src="https://github.com/jmschrei/pomegranate/blob/master/docs/logo/pomegranate-logo.png" width=300>

[![Downloads](https://pepy.tech/badge/pomegranate)](https://pepy.tech/project/pomegranate) ![](https://github.com/jmschrei/pomegranate/actions/workflows/python-package.yml/badge.svg) ![](https://readthedocs.org/projects/pomegranate/badge/?version=latest)

> **Note**
> IMPORTANT: pomegranate v1.0.0 is a ground-up rewrite of pomegranate using PyTorch as the computational backend instead of Cython. Although the same functionality is supported, the API is significantly different. Please see the tutorials and examples folders for help rewriting your code.

[ReadTheDocs](https://pomegranate.readthedocs.io/en/latest/) | [Tutorials](https://github.com/jmschrei/pomegranate/tree/master/docs/tutorials) | [Examples](https://github.com/jmschrei/pomegranate/tree/master/examples) 

pomegranate is a library for probabilistic modeling defined by its modular implementation and treatment of all models as the probability distributions they are. The modular implementation allows one to easily drop normal distributions into a mixture model to create a Gaussian mixture model just as easily as dropping a gamma and a Poisson distribution into a mixture model to create a heterogeneous mixture. But that's not all! Because each model is treated as a probability distribution, Bayesian networks can be dropped into a mixture just as easily as a normal distribution, and hidden Markov models can be dropped into Bayes classifiers to make a classifier over sequences. Together, these two design choices enable a flexibility not seen in any other probabilistic modeling package.

Recently, pomegranate (v1.0.0) was rewritten from the ground up using PyTorch to replace the outdated Cython backend. This rewrite gave me an opportunity to fix many bad design choices that I made as a bb software engineer. Unfortunately, many of these changes are not backwards compatible and will disrupt workflows. On the flip side, these changes have significantly sped up most methods, improved and simplified the code, fixed many issues raised by the community over the years, and made it significantly easier to contribute. I've written more below, but you're likely here now because your code is broken and this is the tl;dr.

Special shout-out to [NumFOCUS](https://numfocus.org/) for supporting this work with a special development grant.

### Installation

`pip install pomegranate`

If you need the last Cython release before the rewrite, use `pip install pomegranate==0.14.8`. You may need to manually install a version of Cython before v3.

### Why a Rewrite?

This rewrite was motivated by four main reasons:

- <b>Speed</b>: Native PyTorch is usually significantly faster than the hand-tuned Cython code that I wrote.
- <b>Features</b>: PyTorch has many features, such as serialization, mixed precision, and GPU support, that can now be directly used in pomegranate without additional work on my end. 
- <b>Community Contribution</b>: A challenge that many people faced when using pomegranate was that they could not modify or extend it because they did not know Cython. Even if they did know Cython, coding in it is a pain that I felt each time I tried adding a new feature or fixing a bug or releasing a new version. Using PyTorch as the backend significantly reduces the amount of effort needed to add in new features.
- <b>Interoperability</b>: Libraries like PyTorch offer an invaluable opportunity to not just utilize their computational backends but to better integrate into existing resources and communities. This rewrite will make it easier for people to integrate probabilistic models with neural networks as losses, constraints, and structural regularizations, as well as with other projects built on PyTorch.

### High-level Changes 

1. General
- The entire codebase has been rewritten in PyTorch and all models are instances of `torch.nn.Module`
- This codebase is checked by a comprehensive suite of >800 unit tests calling assert statements several thousand times, much more than previous versions.
- Installation issues are now likely to come from PyTorch for which there are countless resources to help out.

2. Features
- All models now have GPU support
- All models now have support for half/mixed precision
- Serialization is now handled by PyTorch, yielding more compact and efficient I/O
- Missing values are now supported through `torch.masked.MaskedTensor` objects
- Prior probabilities can now be passed to all relevant models and methods and enable more comprehensive/flexible semi-supervised learning than before

3. Models
- All distributions are now multivariate by default and treat each feature independently (except Normal)
- "Distribution" has been removed from names so that, for example, `NormalDistribution` is now `Normal`
- `FactorGraph` is now supported as first-class citizens, with all the prediction and training methods
- Hidden Markov models have been split into `DenseHMM` and `SparseHMM` models which differ in how the transition matrix is encoded, with `DenseHMM` objects being significantly faster on truly dense graphs

4. Differences
- `NaiveBayes` has been permanently removed as it is redundant with `BayesClassifier`
- `MarkovNetwork` has not yet been implemented
- Constraint graphs and constrained structure learning for Bayesian networks has not yet been implemented
- Silent states for hidden Markov models have not yet been implemented
- Viterbi for hidden Markov models has not yet been implemented

## Speed

Most models and methods in pomegranate v1.0.0 are faster than their counterparts in earlier versions. This generally scales by complexity, where one sees only small speedups for simple distributions on small data sets but much larger speedups for more complex models on big data sets, e.g. hidden Markov model training or Bayesian network inference. The notable exception for now is that Bayesian network structure learning, other than Chow-Liu tree building, is still incomplete and not much faster. In the examples below, `torchegranate` refers to the temporarily repository used to develop pomegranate v1.0.0 and `pomegranate` refers to pomegranate v0.14.8.

### K-Means

Who knows what's happening here? Wild.

![image](https://user-images.githubusercontent.com/3916816/232371843-66b9d326-b4de-4da0-bbb1-5eab5f9a4492.png)


### Hidden Markov Models

Dense transition matrix (CPU)

![image](https://user-images.githubusercontent.com/3916816/232370752-58969609-5ee4-417f-a0da-1fbb83763d63.png)

Sparse transition matrix (CPU)

![image](https://user-images.githubusercontent.com/3916816/232371006-20a82e07-3553-4257-987b-d8e9b333933a.png)

Training a 125 node model with a dense transition matrix

![image](https://user-images.githubusercontent.com/3916816/232394045-e9a13fd6-19b3-4e78-80ce-734b383157a6.png)


### Bayesian Networks
![image](https://user-images.githubusercontent.com/3916816/232370594-e89e66a8-d9d9-4369-ba64-8902d8ec2fcc.png) 
![image](https://user-images.githubusercontent.com/3916816/232370632-199d5e99-0cd5-415e-9c72-c4ec9fb7a44c.png)


## Features

> **Note**
> Please see the [tutorials](https://github.com/jmschrei/pomegranate/tree/master/docs/tutorials) folder for code examples.

Switching from a Cython backend to a PyTorch backend has enabled or expanded a large number of features. Because the rewrite is a thin wrapper over PyTorch, as new features get released for PyTorch they can be applied to pomegranate models without the need for a new release from me. 

### GPU Support

All distributions and methods in pomegranate now have GPU support. Because each distribution is a `torch.nn.Module` object, the use is identical to other code written in PyTorch. This means that both the model and the data have to be moved to the GPU by the user. For instance:

```python
>>> X = torch.exp(torch.randn(50, 4))

# Will execute on the CPU
>>> d = Exponential().fit(X)
>>> d.scales
Parameter containing:
tensor([1.8627, 1.3132, 1.7187, 1.4957])

# Will execute on a GPU
>>> d = Exponential().cuda().fit(X.cuda())
>>> d.scales
Parameter containing:
tensor([1.8627, 1.3132, 1.7187, 1.4957], device='cuda:0')
```

Likewise, all models are distributions, and so can be used on the GPU similarly. When a model is moved to the GPU, all of the models associated with it (e.g. distributions) are also moved to the GPU.

```python
>>> X = torch.exp(torch.randn(50, 4)).cuda()
>>> model = GeneralMixtureModel([Exponential(), Exponential()]).cuda()
>>> model.fit(X)
[1] Improvement: 1.26068115234375, Time: 0.001134s
[2] Improvement: 0.168121337890625, Time: 0.001097s
[3] Improvement: 0.037841796875, Time: 0.001095s
>>> model.distributions[0].scales
Parameter containing:
>>> model.distributions[1].scales
tensor([0.9141, 1.0835, 2.7503, 2.2475], device='cuda:0')
Parameter containing:
tensor([1.9902, 2.3871, 0.8984, 1.2215], device='cuda:0')
```

### Mixed Precision

pomegranate models can, in theory, operate in the same mixed or low-precision regimes as other PyTorch modules. However, because pomegranate uses more complex operations than most neural networks, this sometimes does not work or help in practice because these operations have not been optimized or implemented in the low-precision regime. So, hopefully this feature will become more useful over time.

```python
>>> X = torch.randn(100, 4)
>>> d = Normal(covariance_type='diag')
>>>
>>> with torch.autocast('cuda', dtype=torch.bfloat16):
>>>     d.fit(X)
```

### Serialization

pomegranate distributions are all instances of `torch.nn.Module` and so serialization is the same as any other PyTorch model.

Saving:
```python
>>> X = torch.exp(torch.randn(50, 4)).cuda()
>>> model = GeneralMixtureModel([Exponential(), Exponential()], verbose=True)
>>> model.cuda()
>>> model.fit(X)
>>> torch.save(model, "test.torch")
```

Loading:
```python
>>> model = torch.load("test.torch")
```

### torch.compile

> **Note**
> `torch.compile` is under active development by the PyTorch team and may rapidly improve. For now, you may need to pass in `check_data=False` when initializing models to avoid one compatibility issue.

In PyTorch v2.0.0, `torch.compile` was introduced as a flexible wrapper around tools that would fuse operations together, use CUDA graphs, and generally try to remove I/O bottlenecks in GPU execution. Because these bottlenecks can be extremely significant in the small-to-medium sized data settings many pomegranate users are faced with, `torch.compile` seems like it will be extremely valuable. Rather than targeting entire models, which mostly just compiles the `forward` method, you should compile individual methods from your objects.

```python
# Create your object as normal
>>> mu = torch.exp(torch.randn(100))
>>> d = Exponential(mu).cuda()

# Create some data
>>> X = torch.exp(torch.randn(1000, 100))
>>> d.log_probability(X)

# Compile the `log_probability` method!
>>> d.log_probability = torch.compile(d.log_probability, mode='reduce-overhead', fullgraph=True)
>>> d.log_probability(X)
```

Unfortunately, I have had difficulty getting `torch.compile` to work when methods are called in a nested manner, e.g., when compiling the `predict` method for a mixture model which, inside it, calls the `log_probability` method of each distribution. I have tried to organize the code in a manner that avoids some of these errors, but because the error messages right now are opaque I have had some difficulty.

### Missing Values

pomegranate supports handling data with missing values through `torch.masked.MaskedTensor` objects. Simply, one needs to just put a mask over the values that are missing.

```python
>>> X = <your tensor with NaN for the missing values>
>>> mask = ~torch.isnan(X)
>>> X_masked = torch.masked.MaskedTensor(X, mask=mask)
>>> d = Normal(covariance_type='diag').fit(X_masked)
>>> d.means
Parameter containing:
tensor([0.2271, 0.0290, 0.0763, 0.0135])
```

All algorithms currently treat missingness as something to ignore. As an example, when calculating the mean of a column with missing values, the mean will simply be the average value of the present values. Missing values are not imputed because improper imputation can bias your data, produce unlikely estimates which distort distributions, and also shrink the variance.

Because not all operations are yet available for MaskedTensors, the following distributions are not yet supported for missing values: Bernoulli, categorical, normal with full covariance, uniform

### Prior Probabilities and Semi-supervised Learning

A new feature in pomegranate v1.0.0 is being able to pass in prior probabilities for each observation for mixture models, Bayes classifiers, and hidden Markov models. These are the prior probability that an observation belongs to a component of the model before evaluating the likelihood and should range between 0 and 1. When these values include a 1.0 for an observation, it is treated as a label, because the likelihood no longer matters in terms of assigning that observation to a state. Hence, one can use these prior probabilities to do labeled training when each observation has a 1.0 for some state, semi-supervised learning when a subset of observations (including when sequences are only partially labeled for hidden Markov models), or more sophisticated forms of weighting when the values are between 0 and 1. 

![image](https://user-images.githubusercontent.com/3916816/232373036-39d591e2-e673-450e-ab1c-98e47f0fa6aa.png)

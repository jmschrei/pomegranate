<img src="https://github.com/jmschrei/pomegranate/blob/master/docs/logo/pomegranate-logo.png" width=300>

[![Downloads](https://pepy.tech/badge/pomegranate)](https://pepy.tech/project/pomegranate) ![](https://github.com/jmschrei/pomegranate/actions/workflows/python-package.yml/badge.svg)

> **Note**
> IMPORTANT: pomegranate v1.0.0 is a ground-up rewrite of pomegranate using PyTorch as the computational backend instead of Cython. Although the same functionality is supported, the API is significantly different. Please see the tutorials and examples folders for help rewriting your code.

pomegranate is a library of probabilistic models defined by its modular implementation and treatment of all probabilistic models as the probability distributions they are. Together, these design choices allow one to easily drop two normal distributions into a mixture model to create a Gaussian mixture model or drop a Poisson distribution and an exponential distribution to just as easily create a heterogeneous mixture, or drop in two Bayesian networks, or drop in two hidden Markov models to create a mixture over sequence models.

Recently, pomegranate (v1.0.0) was rewritten from the ground up using PyTorch to replace the outdated Cython backend. This rewrite gave me an opportunity to fix many bad design choices that I made while I was still a bb software engineer. Unfortunately, many of these changes are not back compatible and will disrupt everyone's workflows. However, the changes have significantly improved/simplified the code, fixed many issues raised by the community over the year, made it easier to contribute, and made the code significantly faster. I've written more below, but you're likely here now because your code is broken and this is the tl;dr.

Special shout-out to [NumFOCUS](https://numfocus.org/) for supporting this work with a special development grant.

### Installation

`pip install pomegranate`

If you need the last Cython release before the rewrite, use `pip install pomegranate==0.14.8`.

### Why a Rewrite?

This rewrite was motivated by four main reasons:

- <b>Speed</b>: Native PyTorch is usually significantly faster than the hand-tuned Cython code that I wrote.
- <b>Features</b>: PyTorch has many features, such as serialization, mixed precision, and GPU support, that can now be directly used in pomegranate without additional work on my end. 
- <b>Community Contribution</b>: A challenge that many people faced when using pomegranate was that they could not extend it because they did not know Cython, and even if they did know it, coding in Cython is a pain. I felt this pain every time I tried adding a new feature or fixing a bug. Using PyTorch as the backend significantly reduces the amount of effort needed to add in new features.
- <b>Interoperability</b>: Libraries like PyTorch offer a unique ability to not just utilize their computational backends but to better integrate into existing deep learning resources. This rewrite should make it easier for people to merge probabilistic models with deep learning models.

## Features

> **Note**
> Please see the [tutorials](https://github.com/jmschrei/pomegranate/tree/master/tutorials) folder for code examples.

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

### Serialization

pomegranate distributions are all instances of `torch.nn.Module` and so serialization is the same as any other model and can use any of the other built-in functionality.

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

In PyTorch v2.0.0, `torch.compile` was introduced as a flexible wrapper around tools that would fuse operations together, use CUDA graphs, and generally try to remove I/O bottlenecks in GPU execution. Because these bottlenecks can be extremely significant in the small-to-medium sized data settings many pomegranate users are faced with, `torch.compile` seems like it will be extremely valuable. Rather than targetting entire models, which mostly just compiles the `forward` method, you should compile individual methods from your objects.

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

### Frequently Asked Questions

> Why can't we just use `torch.distributions`?

`torch.distributions` is a great implementation of the statistical characteristics of many distributions, but does not implement fitting these distributions to data or using them as components of larger functions. If all you need to do is calculate log probabilities, or sample, given parameters (perhaps as output from neural network components), `torch.distributions` is a great, simple, alternative.

> What models are implemented in pomegranatee?

Currently, implementations of many distributions are included, as well as general mixture models, Bayes classifiers (including naive Bayes), hidden Markov models, and Markov chains. Bayesian networks will be added soon but are not yet included.

> How much faster is v1.0.0 over previous versions?

It depends on the method being used. Most individual distributions are approximately 2-3x faster. Some distributions, such as the categorical distributions, can be over 10x faster. These will be even faster if a GPU is used.

<img src="https://github.com/jmschrei/pomegranate/blob/master/docs/logo/pomegranate-logo.png" width=300>

[![Downloads](https://pepy.tech/badge/torchegranate)](https://pepy.tech/project/torchegranate) ![](https://github.com/jmschrei/torchegranate/actions/workflows/python-package.yml/badge.svg)

> **Warning**
> torchegranate is currently under rapid active development and the API has not yet been finalized. Major changes will be noted in the CHANGELOG. 

> **Note**
> This is a temporary repository that will host code until the main functionality of pomegranate is reproduced. Then, the code here will be merged back into the main pomegranate repo.

torchegranate is a rewrite of the [pomegranate](https://github.com/jmschrei/pomegranate) library to use PyTorch as a backend. It implements probabilistic models with a modular implementation, enabling greater flexibility in terms of model creation than most models allow. Specifically, one can drop any probability distribution into any compositional model, e.g., drop Poisson distributions into a mixture model, to create any model desired without needing to explicitly hardcode each potential model. Because one is defining the distributions to use in each of the compositional models, there is no limitation on the models being homogenous -- one can create a mixture of a exponential distribution and a gamma distribution just as easily as creating a mixture entirely composed of gamma distributions. 

But that's not all! A core aspect of pomegranate's philosophy is that every probabilistic model is a probability distribution. Hidden Markov models are simply distributions over sequences and Bayesian networks are joint probability tables that are broken down according to the conditional independences defined in a graph. Functionally, this means that one can drop a mixture model into a hidden Markov model to use as an emission just as easily as one can drop an individual distribution. As part of the roadmap, and made part in possible due to the flexibility of PyTorch, complete cross-functionality should be possible, such as a Bayes classifier with hidden Markov models or a mixture of Bayesian networks.

Special shout-out to [NumFOCUS](https://numfocus.org/) for supporting this work with a special development grant.

### Installation

`pip install torchegranate`

### Why a Rewrite?

This rewrite was motivated by three main reasons:

- <b>Speed</b>: Native PyTorch is just as fast as the hand-tuned Cython code I wrote for pomegranate, if not significantly faster.
- <b>Community Contribution</b>: A challenge that many people faced when using pomegranate was that they could not extend it because they did not know Cython, and even if they did know it, coding in Cython is a pain. I felt this pain every time I tried adding a new feature or fixing a bug. Using PyTorch as the backend significantly reduces this problem.
- <b>Interoperability</b>: Libraries like PyTorch offer a unique ability to not just utilize their computational backends but to better integrate into existing deep learning resources. This rewrite should make it easier for people to merge probabilistic models with deep learning models.

### Roadmap

The ultimate goal is for this repository to include all of the useful features from pomegranate, at which point this repository will be merged back into the main pomegranate library. However, that is quite a far way off. Here are some milestones that I see for the next few releases.

- [x] v0.1.0: Initial draft of most models with basic feature support, only on CPUs
- [x] v0.2.0: Addition of GPU support for all existing operations and serialization via PyTorch
- [x] v0.3.0: Addition of missing value support for all existing algorithms
- [x] v0.4.0: Addition of Bayesian networks and factor graphs
- [ ] v0.5.0: Addition of sampling algorithms for each existing method
- [ ] v0.6.0: Addition of pass-through for forward and backward algorithms to enable direct inclusion of these components into PyTorch models


### GPU Support

All distributions in torchegranate have GPU support. Because each distribution is a `torch.nn.Module` object, the use is identical to other code written in PyTorch. This means that both the model and the data have to be moved to the GPU by the user. For instance:

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

torchegranate objects are all instances of `torch.nn.Module` and so serialization is the same as any other model and can use any of the other built-in functionality.

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

### Missing Values

torchegranate supports handling data with missing values through `torch.masked.MaskedTensor` objects. Simply, one needs to just put a mask over the values that are missing.

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

> What models are implemented in torchegranate?

Currently, implementations of many distributions are included, as well as general mixture models, Bayes classifiers (including naive Bayes), hidden Markov models, and Markov chains. Bayesian networks will be added soon but are not yet included.

> How much faster is this than pomegranate?

It depends on the method being used. Most individual distributions are approximately 2-3x faster. Some distributions, such as the categorical distributions, can be over 10x faster. These will be even faster if a GPU is used.

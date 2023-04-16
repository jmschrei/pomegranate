.. _ooc:

Out of Core Learning
====================

..- `IPython Notebook Tutorial <https://github.com/jmschrei/pomegranate/blob/master/tutorials/C_Feature_Tutorial_2_Out_Of_Core_Learning.ipynb>`_

Sometimes we would like to train models on datasets that cannot fit in memory. A common way to overcome this when training neural networks is to use minibatch approaches, where gradients are calculated with respect to a smaller number of points. However, because probabilistic models usually have fewer or no latent variables, one can calculate updates in a much simpler fashion. Specifically, pomegranate allows you to summarize minibatches into sufficient statistics. These sufficient statistics are additive, meaning that adding the sufficient statistics from two batches is equivalent to calculating them on the concatenation of the two batches. Due to this property, pomegranate can calculate exact sufficient statistics across an entire data set without the need to see the entire data set at once. This can be done through the methods ```model.summarize``` and ```model.from_summaries```. Let's see an example of using it to update a normal distribution.

.. code-block:: python

	>>> import torch
	>>> from torchegranate.distributions import Normal
	>>>
	>>> X = torch.randn(5000, 1) * 5 + 3 
	>>>
	>>> a = Normal([1.0], [1.0], covariance_type='diag')
	>>> a.fit(X)
	>>> a.means, torch.sqrt(a.covs)
	(Parameter containing:
	 tensor([2.8534]),
	 tensor([4.9633]))
	>>>
	>>> b = Normal([1.0], [1.0], covariance_type='diag')
	>>> for i in range(5):
	>>>     b.summarize(X[i*1000:(i+1)*1000])
	>>> b.from_summaries()
	>>> b.means, torch.sqrt(b.covs)
	(Parameter containing:
	 tensor([2.8534]),
	 tensor([4.9633]))

This is a simple example with a simple distribution, but all models support the same API.

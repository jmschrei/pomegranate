.. _nan:

Missing Values
==============

..- `IPython Notebook Tutorial <https://github.com/jmschrei/pomegranate/blob/master/tutorials/C_Feature_Tutorial_4_Missing_Values.ipynb>`_

As of v1.0.0, pomegranate supports missing values for almost all algorithms. This means that models can be fit to incomplete data sets, inference can be done on examples that have missing values, and even Bayesian network structure learning can be done in the presence of missing values. Currently, this support exists in the form of calculating sufficient statistics with respect to only the variables that are present in a sample and ignoring the missing values, in contrast to imputing the missing values and then running the original algorithms on the completed data.

Missing value support was added in a manner that requires the least user thought. Specifically, one just has to replace the tensors that are passed in with a ``torch.masked.MaskedTensor`` object that takes in a tensor and a mask. Mask indices that have a ``False`` value are considered to be missing and so the value that they take are ignored. For instance:

.. code-block:: python

	>>> import torch
	>>> X = torch.tensor([[0, 1, -1], [-1, -1, 0], [-1, 0, 0], [0, 0, 1]])
	>>> X_masked = torch.masked.MaskedTensor(X, mask=X != -1)

This will generate a ``MaskedTensor`` object where the indices where the value is -1 are masked out. These masked tensors can then be passed into any method that previously accepted a normal tensor.


.. code-block:: python

	>>> import torch
	>>> from pomegranate.distributions import Exponential
	>>>
	>>> X = torch.exp(torch.randn(100, 1))
	>>> mask = torch.ones(100, 1dtype=bool)
	>>> mask[75:] = False
	>>> X_masked = torch.masked.MaskedTensor(X, mask=mask)
	>>>
	>>> Exponential().fit(X[:75])
	Parameter containing:
	tensor([1.6111])
	>>>
	>>> Exponential().fit(X_masked)
	Parameter containing:
	tensor([1.6111])

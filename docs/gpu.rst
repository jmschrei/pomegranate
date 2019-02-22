.. _gpu:

GPU Usage
=========

pomegranate has GPU accelerated matrix multiplications to speed up all operations involving multivariate Gaussian distributions and all models that use them. This has led to an approximately 4x speedup for multivariate Gaussian mixture models and HMMs compared to using BLAS only. This speedup seems to scale better with dimensionality, with higher dimensional models seeing a larger speedup than smaller dimensional ones.

By default, pomegranate will activate GPU acceleration if it can import cupy, otherwise it will default to BLAS. You can check whether pomegranate is using GPU acceleration with this built-in function:

.. code-block:: python
	
	import pomegranate
	print(pomegranate.utils.is_gpu_enabled())

If you'd like to deactivate GPU acceleration you can use the following command:

.. code-block:: python
	
	pomegranate.utils.disable_gpu()

Likewise, if you'd like to activate GPU acceleration you can use the following command:

.. code-block:: python

	pomegranate.utils.enable_gpu()


FAQ
---

Q. Why cupy and not Theano?

A. pomegranate only needs to do matrix multiplications using a GPU. While Theano supports an impressive range of more complex operations, it did not have a simple interface to support a matrix-matrix multiplication in the same manner that cupy does.


Q. Why am I not seeing a large speedup with my GPU?

A. There is a cost to transferring data to and from a GPU. It is possible that the GPU isn't fast enough, or that there isn't enough data to utilize the massively parallel aspect of a GPU for your dataset. 


Q. Does pomegranate work using my type of GPU?

A. The supported GPUs will be better documented on the cupy package.


Q. Is multi-GPU supported?

A. Currently, no. In theory it should be possible, though. 

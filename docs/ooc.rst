.. _ooc:

Out of Core Learning
====================

Sometimes datasets which we'd like to train on can't fit in memory but we'd still like to get an exact update. pomegranate supports out of core training to allow this, by allowing models to summarize batches of data into sufficient statistics and then later on using these sufficient statistics to get an exact update for model parameters. These are done through the methods ```model.summarize``` and ```model.from_summaries```. Let's see an example of using it to update a normal distribution.

.. code-block:: python

	>>> from pomegranate import *
	>>> import numpy
	>>>
	>>> a = NormalDistribution(1, 1)
	>>> b = NormalDistribution(1, 1)
	>>> X = numpy.random.normal(3, 5, size=(5000,))
	>>> 
	>>> a.fit(X)
	>>> a
	{
	    "frozen" :false,
	    "class" :"Distribution",
	    "parameters" :[
	        3.012692830297519,
	        4.972082359070984
	    ],
	    "name" :"NormalDistribution"
	}
	>>> for i in range(5):
	>>>     b.summarize(X[i*1000:(i+1)*1000])
	>>> b.from_summaries()
	>>> b
	{
	    "frozen" :false,
	    "class" :"Distribution",
	    "parameters" :[
	        3.01269283029752,
	        4.972082359070983
	    ],
	    "name" :"NormalDistribution"
	}

This is a simple example with a simple distribution, but all models and model stacks support this type of learning. Lets next look at a simple Bayesian network.

.. code-block::python

	>>> from pomegranate import *
	>>> import numpy
	>>>
	>>> d1 = DiscreteDistribution({0: 0.25, 1: 0.75})
	>>> d2 = DiscreteDistribution({0: 0.45, 1: 0.55})
	>>> d3 = ConditionalProbabilityTable([[0, 0, 0, 0.02], 
								  [0, 0, 1, 0.98],
								  [0, 1, 0, 0.15],
								  [0, 1, 1, 0.85],
								  [1, 0, 0, 0.33],
								  [1, 0, 1, 0.67],
								  [1, 1, 0, 0.89],
								  [1, 1, 1, 0.11]], [d1, d2])
	>>>
	>>> d4 = ConditionalProbabilityTable([[0, 0, 0.4], 
                                  [0, 1, 0.6],
                                  [1, 0, 0.3],
                                  [1, 1, 0.7]], [d3]) 
    >>>
	>>> s1 = State(d1, name="s1")
	>>> s2 = State(d2, name="s2")
	>>> s3 = State(d3, name="s3")
	>>> s4 = State(d4, name="s4")
	>>>
	>>> model = BayesianNetwork()
	>>> model.add_nodes(s1, s2, s3, s4)
	>>> model.add_edge(s1, s3)
	>>> model.add_edge(s2, s3)
	>>> model.add_edge(s3, s4)
	>>> model.bake()
	>>> model2 = model.copy()
	>>>
	>>> X = numpy.random.randint(2, size=(10000, 4))
	>>> print model.states[0].distribution.equals( model2.states[0].distribution )
	True
	>>> model.fit(X)
	>>> print model.states[0].distribution.equals( model2.states[0].distribution )
	False
	>>> model2.summarize(X[:2500])
	>>> model2.summarize(X[2500:5000])
	>>> model2.summarize(X[5000:7500])
	>>> model2.summarize(X[7500:])
	>>> model2.from_summaries()
	>>>
	>>> print model.states[0].distribution.equals( model2.states[0].distribution )
	True

We can see that before fitting to any data, the distribution in one of the states is equal for both. After fitting the first distribution they become different as would be expected. After fitting the second one through summarize the distributions become equal again, showing that it is recovering an exact update.

pomegranate provides support for out-of-core computing when one would like to use an iterative method such as EM without needing to rewrite the convergence criterion themselves. This is done through two parameters, the first being `batch_size` and the second being `batches_per_epoch`. Both parameters are set to None by default meaning that the batch size is the full dataset and that one epoch means one run through the full dataset. However, both of these can be set to whatever the user would like. For instance, if one wanted to read only 10,000 samples per batch but still get exact updates as if they had viewed the whole dataset at the same time, they could pass `batch_size=10000` into either the `fit` method or the `from_samples` method. If they wanted to update the parameters of the model after each batch in a minibatch setting, they could set `batches_per_epoch=1`. While this still allows the user to use `n_jobs` as before to speed up calculations, one should note that if `batches_per_epoch != None` then it should be greater than or equal to the number of threads desired.

Here is an example of fitting a mixture model to some data in both a normal and an out-of-core manner.

.. code-block:: python

	>>> from pomegranate import *
	>>> from sklearn.datasets import make_blobs
	>>> import numpy, time
	>>> numpy.random.seed(0)	
	>>>
	>>> n, d, m = 55710, 25, 4
	>>> X, _ = make_blobs(n, d, m, cluster_std=4, shuffle=True)
	>>> 
	>>> tic = time.time()
	>>> 
	>>> n, d, m = 557100, 25, 4
	>>> X, _ = make_blobs(n, d, m, cluster_std=4, shuffle=True)
	>>>
	>>> model = GeneralMixtureModel.from_samples(MultivariateGaussianDistribution, n_components=m, X=X, 
			n_init=1, max_iterations=5, init='first-k', verbose=True)
	[1] Improvement: 2841.61601918	Time (s): 1.052
	[2] Improvement: 830.059409089	Time (s): 1.912
	[3] Improvement: 368.397171594	Time (s): 1.415
	[4] Improvement: 199.537868068	Time (s): 1.119
	[5] Improvement: 121.741913736	Time (s): 1.856
	Total Improvement: 4361.35238167
	Total Time (s): 8.2603
	>>>
	>>> model = GeneralMixtureModel.from_samples(MultivariateGaussianDistribution, n_components=m, X=X, 
			n_init=1, max_iterations=5, init='first-k', verbose=True, batch_size=10000)
	>>>
	>>> print time.time() - tic, model.log_probability(X).sum()
	[1] Improvement: 2841.616018	Time (s): 1.246
	[2] Improvement: 830.059409752	Time (s): 1.285
	[3] Improvement: 368.397172503	Time (s): 1.21
	[4] Improvement: 199.537868194	Time (s): 0.9692
	[5] Improvement: 121.741913162	Time (s): 0.8211
	Total Improvement: 4361.35238161
	Total Time (s): 6.4084

It looks like it takes a similar amount of time while still producing identical results. Of course, to use it in an out-of-core manner one would want to feed in a numpy memory map instead of an in-memory numpy array. The only change would be to do the following:

.. code-block:: python

	from pomegranate import *
	import numpy

	X = numpy.load('X_train.npy', mmap_mode='r')
	model = GeneralMixtureModel.from_samples(MultivariateGaussianDistribution, n_components=10, X=X,
		n_init=1, max_iterations=5, init='first-k', verbose=True, batch_size=10000)

The only change is to the datatype of `X`, which is now a memory map instead of an array.

We can also use parallelism in conjunction with out-of-core learning. We'll attempt to learn a Gaussian mixture model over ~24G of data using a computer with only ~4G of memory. First without parallelism.

.. code-block:: python

	>>> import numpy
	>>> from pomegranate import *
	>>> 
	>>> X = numpy.load("big_datums.npy", mmap_mode="r")
	>>> print X.shape
	(60000000, 50)
	>>> model = GeneralMixtureModel.from_samples(MultivariateGaussianDistribution, 
			3, X, max_iterations=50, batch_size=100000, batches_per_epoch=50, 
			n_jobs=1, verbose=True)
	[1] Improvement: 252989.289729	Time (s): 18.84
	[2] Improvement: 58446.0881071	Time (s): 18.75
	[3] Improvement: 26323.5638447	Time (s): 18.76
	[4] Improvement: 15133.080919	Time (s): 18.8
	[5] Improvement: 10138.1656616	Time (s): 18.91
	[6] Improvement: 7458.30408692	Time (s): 18.86
	[7] Improvement: 5995.06008983	Time (s): 18.89
	[8] Improvement: 4838.79921204	Time (s): 18.91
	[9] Improvement: 4188.59295541	Time (s): 18.97
	[10] Improvement: 3590.57844329	Time (s): 18.93
	...

And now in parallel:

.. code-block:: python

	>>> import numpy
	>>> from pomegranate import *
	>>> 
	>>> X = numpy.load("big_datums.npy", mmap_mode="r")
	>>> print X.shape
	(60000000, 50)
	>>> model = GeneralMixtureModel.from_samples(MultivariateGaussianDistribution, 
			3, X, max_iterations=50, batch_size=100000, batches_per_epoch=50, 
			n_jobs=4, verbose=True)
	[1] Improvement: 252989.289729	Time (s): 9.952
	[2] Improvement: 58446.0881071	Time (s): 9.952
	[3] Improvement: 26323.5638446	Time (s): 9.969
	[4] Improvement: 15133.080919	Time (s): 10.0
	[5] Improvement: 10138.1656617	Time (s): 9.986
	[6] Improvement: 7458.30408692	Time (s): 9.949
	[7] Improvement: 5995.06008989	Time (s): 9.971
	[8] Improvement: 4838.79921204	Time (s): 10.02
	[9] Improvement: 4188.59295535	Time (s): 10.02
	[10] Improvement: 3590.57844335	Time (s): 9.989
	...

The speed improvement may be sub-linear in cases where data loading takes up a substantial portion of time. A solid state drive will likely improve this performance.


FAQ
---

Q. What data storage types are able to be used with out of core training?

A. Currently only stored numpy arrays (.npy files) that can be read as memory maps using `numpy.load('data.npy', mmap_mode='r')` are supported for data that truly can't be loaded into memory.


Q. Are there plans to add in more on-disc data sources?

A. At some point, yes. However, numpy memory maps are extremely convenient and easy to use.


Q. What should I set my batch size to?

A. It should be the largest amount of data that fits in memory. The larger the block of data, the more efficient the calculations can be, particularly if GPU computing is being used.


Q. Can I still do multi-threading / use a GPU with out-of-core learning?

A. Absolutely. No change is needed except to specify the batch size. As said above, the larger the batch size likely the more efficient the calculations, particularly when using a GPU, but one should play with this themselves for their specific datasets.


Q. Does out of core learning give exact or approximate updates?

A. It gives exact updates. Sufficient statistics are collected for each of the batches and are equal to the sufficient statistics that one would get from the full dataset. However, the initialization step is done on only a single batch. This may cause the final models to differ due simply to the different initializations. If one has pre-defined initializations and simply calls `fit`, then the exact same model will be yielded.

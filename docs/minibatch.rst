.. _minibatch:

Minibatch Learning
==================

The most common and natural way to train a model is to derive updates from the full dataset being used for training. A good example is batch gradient descent, which derives the direction and magnitude with which to update the weights of a model from the full dataset. In contrast, minibatch learning involves deriving model updates from only a subset of the data at a time. A popular variant is called stochastic gradient descent and involves training on only a single sample at a time. It has surprisingly good properties in some settings and frequently converges faster than batch gradient descent. In general, as the size of the minibatches increases the noise of the update decreases but the computation time it takes to calculate the update increases. In practice, the increase in computation time can be diminished by utilizing specialized hardware like GPUs or tools like BLAS, making larger batch sizes much more favorable.

Similar to the other training strategies in pomegranate, minibatch learning is made possible by the decoupling the gathering of sufficient statistics from the updating of the model. Out-of-core learning works by aggregating sufficient statistics among many batches and then calculating a model update at the end. Minibatch learning involves calculating sufficient statistics from a single batch of data and then immediately updating the model before moving on to the next batch. In this manner, out-of-core learning can be used to learn the same model one would as if they had seen the entire dataset, whereas minibatch learning distinctly does not learn the same model. 

Minibatch learning in pomegranate is implemented through the use of the `batch_size` and `batches_per_epoch` keywords. Specifically, one must set some batch size and then the number of batches that should be used before calculating an update. Traditionally one may want to set this to 1, specifying that the update is calculated after a single batch, but there is no reason why multiple batches couldn't be used to calculate an update if desired. Here is an example:

.. code-block:: python

	>>> from pomegranate import *
	>>> from sklearn.datasets import make_blobs
	>>> import numpy
	>>> numpy.random.seed(0)
	>>>
	>>> n, d, m = 557100, 25, 4
	>>> X, _ = make_blobs(n, d, m, cluster_std=4, shuffle=True)
	>>>
	>>> model = GeneralMixtureModel.from_samples(MultivariateGaussianDistribution, n_components=m, X=X, 
		n_init=1, max_iterations=5, init='first-k', verbose=True)
	[1] Improvement: 2841.61601918	Time (s): 2.373
	[2] Improvement: 830.059409089	Time (s): 1.866
	[3] Improvement: 368.397171594	Time (s): 1.9
	[4] Improvement: 199.537868068	Time (s): 1.936
	[5] Improvement: 121.741913736	Time (s): 1.297
	Total Improvement: 4361.35238167
	Total Time (s): 11.6352
	>>> print model.log_probability(X).sum()
	-40070118.5672
	>>> model = GeneralMixtureModel.from_samples(MultivariateGaussianDistribution, n_components=m, X=X, 
		n_init=1, max_iterations=5, init='first-k', verbose=True, batch_size=10000, batches_per_epoch=1)
	[1] Improvement: 722.089154968	Time (s): 0.04816
	[2] Improvement: 1285.42216032	Time (s): 0.1187
	[3] Improvement: 1215.87126935	Time (s): 0.1184
	[4] Improvement: 1092.96581448	Time (s): 0.1064
	[5] Improvement: 1222.47873598	Time (s): 0.1127
	Total Improvement: 5538.82713509
	Total Time (s): 0.5239
	>>> print model.log_probability(X).sum()
	-40100556.1073

We can see that, as expected, each batch takes a significantly shorter amount of time than an epoch on the full dataset. The model that is produced seems to give similar, though not quite as good, log probability scores on the full dataset. However, it doesn't quite appear that the minibatch approach is converging. This is a problem that is common to many second order update methods, including EM. A frequent solution is to use "stepwise EM" as described in Bishop (p. 368) where a decay is set on the step size. What this means in pomegranate is that the inertia increases as a function of the number of iterations, ensuring convergence. The step size is calculated as $1 - (1 - inertia) * (2 + k)^{-lr_decay}$ where k is the number of iterations, and inertia is the proportion of the old parameters used (typically 0.0). Let's see what happens when we do and don't use a decay.

Without a decay:

.. code-block:: python

	>>> from pomegranate import *
	>>> from sklearn.datasets import make_blobs
	>>> import numpy
	>>> numpy.random.seed(0)
	>>>
	>>> n, d, m = 557100, 25, 4
	>>> X, _ = make_blobs(n, d, m, cluster_std=0.2, shuffle=True)
	>>> 
	>>> model = GeneralMixtureModel.from_samples(MultivariateGaussianDistribution, n_components=m, X=X, 
		n_init=1, init='first-k', max_iterations=25, verbose=True)
	[1] Improvement: 2829.31191301	Time (s): 2.301
	[2] Improvement: 826.320851962	Time (s): 2.48
	...
	[24] Improvement: 3.06409630994	Time (s): 1.754
	[25] Improvement: 2.80105452973	Time (s): 2.085
	Total Improvement: 4689.45119551
	Total Time (s): 56.0279
	>>> print model.log_probability(X).sum()
	825534.114778
	>>> model = GeneralMixtureModel.from_samples(MultivariateGaussianDistribution, n_components=m, X=X, 
		n_init=1, init='first-k', verbose=True, max_iterations=25, batch_size=100000, batches_per_epoch=1)
	[1] Improvement: 1053.6233468	Time (s): 0.5352
	[2] Improvement: 1376.99673058	Time (s): 0.4793
	[3] Improvement: 1252.96547886	Time (s): 0.4395
	[4] Improvement: 1198.91839751	Time (s): 0.5888
	[5] Improvement: 1215.01340037	Time (s): 0.4509
	...
	[21] Improvement: 1090.46744541	Time (s): 0.3881
	[22] Improvement: 1095.10725289	Time (s): 0.3707
	[23] Improvement: 1131.25851107	Time (s): 0.2651
	[24] Improvement: 854.786135095	Time (s): 0.301
	[25] Improvement: 1514.55094317	Time (s): 0.3382
	Total Improvement: 29235.6544552
	Total Time (s): 10.8482
	>>> print model.log_probability(X).sum()
	823170.438085

With a decay of 0.5:

.. code-block:: python

	>>> from pomegranate import *
	>>> from sklearn.datasets import make_blobs
	>>> import numpy
	>>> numpy.random.seed(0)
	>>>
	>>> n, d, m = 557100, 25, 4
	>>> X, _ = make_blobs(n, d, m, cluster_std=0.2, shuffle=True)
	>>> 
	>>> model = GeneralMixtureModel.from_samples(MultivariateGaussianDistribution, n_components=m, X=X, 
		n_init=1, init='first-k', max_iterations=25, verbose=True)
	[1] Improvement: 2829.31191301	Time (s): 2.181
	[2] Improvement: 826.320851962	Time (s): 2.166
	...
	[24] Improvement: 3.06409630994	Time (s): 2.206
	[25] Improvement: 2.80105452973	Time (s): 2.477
	Total Improvement: 4689.45119551
	Total Time (s): 60.1254
	>>> print model.log_probability(X).sum()
	825534.114778
	>>> model = GeneralMixtureModel.from_samples(MultivariateGaussianDistribution, n_components=m, X=X, 
		n_init=1, init='first-k', verbose=True, max_iterations=25, batch_size=100000, lr_decay=0.5, batches_per_epoch=1)
	[1] Improvement: 796.097334852	Time (s): 0.3315
	[2] Improvement: 752.266746429	Time (s): 0.3974
	[3] Improvement: 638.97759942	Time (s): 0.3137
	[4] Improvement: 555.537951205	Time (s): 0.4384
	[5] Improvement: 494.904335364	Time (s): 0.4975
	...
	[21] Improvement: 223.957448395	Time (s): 0.6416
	[22] Improvement: 230.288166696	Time (s): 0.6673
	[23] Improvement: 205.620445758	Time (s): 0.4929
	[24] Improvement: 197.345235002	Time (s): 0.5871
	[25] Improvement: 210.324390932	Time (s): 0.8142
	Total Improvement: 8921.00813791
	Total Time (s): 14.5593
	>>> print model.log_probability(X).sum()
	825199.540663

It does seem like the model is converging. The primary conceptual difference between using a learning rate decay and a maximum number iterations is that with a decay the updates are smoothed over several batches whereas without the decay the updates are based primarily on a single batch.

In addition, one can use parallelism with minibatching in the same way one would use it in an out-of-core setting. To borrow the example from the ooc section, let's attempt to learn a Gaussian mixture model over ~24G of data using a computer with only ~4G of memory. First without parallelism.

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

Q. Does minibatch learning produce an exact update?

A. No. Since the model is updated after each batch (or group of batches) it will produce a different model than waiting to update the model until the entire dataset is seen.


Q. Is minibatch learning faster?

A. It is typically faster per epoch simply because now an epoch is a subset of the the full dataset, usually a single batch. However, it frequently can take less time total to converge depending on the learning rate decay and has better theoretical properties than batch EM.


Q. Are there other names for minibatch learning?

A. The Bishop textbook refers to minibatch learning as "stepwise EM", and sometimes it is referred to as stochastic EM.


Q. Can minibatch learning be used in an out-of-core manner?

A. Yes! Since only a batch of data is seen at a time there is no reason why the whole dataset needs to be in memory. However, the initialization step will now only use a single batch of data and so may not be as good as if the initialization was done on the full dataset.

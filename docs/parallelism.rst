.. _parallelism:

Parallelism
===========

pomegranate supports multi-threaded parallelism through the joblib library. Typically, python applications use multi-processing in order to get around the Global Interpreter Lock (GIL) that prevents multiple threads from running in the same Python process. However, since pomegranate does most of its computation using only C level primitives, it can release the GIL and enable multiple threads to work at the same time. The main difference that a user will notice is that it is more memory efficient, because instead of copying the data across multiple processes that each have their own memory allocated, each thread in pomegranate can operate on the same single memory allocation.

Using parallelism in pomegranate is as simple as specifying the `n_jobs` parameter in any of the methods-- both fitting and prediction methods!

For example:

.. code-block:: python

	import pomegranate, numpy

	X = numpy.random.randn(1000, 1)
	
	# No parallelism
	model = GeneralMixtureModel.from_samples(NormalDistribution, 3, X)

	# Some parallelism
	model = GeneralMixtureModel.from_samples(NormalDistribution, 3, X, n_jobs=2)

	# Maximum parallelism
	model = GeneralMixtureModel.from_samples(NormalDistribution, 3, X, n_jobs=-1)

If you instead have a fit model and you're just looking to speed up prediction time, you need only pass the n_jobs parameter in to those methods as well.

.. code-block:: python

	model = <fit model>
	X = numpy.random.randn(1000, 1)

	# No parallelism
	y = model.predict_proba(X)

	# Some parallelism
	y = model.predict_proba(X, n_jobs=2)

	# Maximum parallelism
	y = model.predict_proba(X, n_jobs=-1)

FAQ
---

Q. What models support parallelism?

A. All models should support parallel fitting. All models (except for HMMs) support parallel predictions natively through the `n_jobs` parameter. Basic distributions do not support parallelism as they typically take a neglible amount of time to do anything with.


Q. How can I parallelize something that doesn't have built-in parallelism?

A. You can easily write a parallelized prediction wrapper for any model using multiprocessing. It would likely look like the following:

.. code-block:: python

	from joblib import Parallel, delayed
	from pomegranate import BayesianNetwork

	def parallel_predict(name, X):
		"""Load up a pomegranate model and predict a subset of X"""

		model = BayesianNetwork.from_json(name)
		return model.predict(X)

	X_train, X_test = numpy.load("train.data"), numpy.load("test.data")

	model = BayesianNetwork.from_samples(X_train)
	with open("model.json", "w") as outfile:
		outfile.write(model.to_json())

	n = len(X_test)
	starts, ends = [i*n/4 for i in range(4)], [(i+1)*n/4 for i in range(4)]

	y_pred = Parallel(n_jobs=4)( delayed(parallel_predict)(
		X_test[start:end]) for start, end in zip(starts, ends))


Q. What is the difference between multiprocessing and multithreading?

A. Multiprocessing involves creating a whole new Python process and passing the relevant data over to it. Multithreading involves creating multiple threads within the same Python process that all have access to the same memory. Multithreading is frequently more efficient because it doesn't involve copying potentially large amounts of data between different Python processes.


Q. Why don't all modules use multithreading?

A. Python has the Global Interpreter Lock (GIL) enabled which prevents more than one thread to execute per processes. The work-around is multiprocessing, which simply creates multiple processes that each have one thread working. When one uses Cython, they can disable to GIL when using only C-level primitives. Since most of the compute-intensive tasks involve only C-level primitives, multithreading is a natural choice for pomegranate. In situations where the size of the data is small and the cost of transferring it from one process to another is negligible, then multithreading can simply make things more complicated.

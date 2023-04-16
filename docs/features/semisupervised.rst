.. _semisupervised.rst

Semi-Supervised Learning
========================

Semi-supervised learning is a branch of machine learning that deals with training sets that are only partially labeled. These types of datasets are common in the world. For example, consider that one may have a few hundred images that are properly labeled as being various food items. They may wish to augment this dataset with the hundreds of thousands of unlabeled pictures of food floating around the internet, but not wish to incur the cost of having to hand label them. Unfortunately, many machine learning methods are not able to handle both labeled and unlabeled data together and so frequently either the unlabeled data is tossed out in favor of supervised learning, or the labeled data is only used to identify the meaning of clusters learned by unsupervised techniques on the unlabeled data.

Probabilistic modeling offers an intuitive way of incorporating both labeled and unlabeled data into the training process through the expectation-maximization algorithm. Essentially, one will initialize the model on the labeled data, calculate the sufficient statistics of the unlabeled data and labeled data separately, and then add them together. This process can be thought of as vanilla EM on the unlabeled data except that at each iteration the sufficient statistics from the labeled data (MLE estimates) are added.

pomegranate follows the same convention as scikit-learn when it comes to partially labeled datasets. The label vector `y` is still of an equal length to the data matrix `X`, with labeled samples given the appropriate integer label, but unlabeled samples are given the label `-1`. While `np.nan` may be a more intuitive choice for missing labels, it isn't used because `np.nan` is a double and the `y` vector is integers. When doing semi-supervised learning with hidden Markov models, however, one would pass in a list of labels for each labeled sequence, or `None` for each unlabeled sequence, instead of `-1` to indicate an unlabeled sequence. 

All models that support labeled data support semi-supervised learning, including naive Bayes classifiers, general Bayes classifiers, and hidden Markov models. Semi-supervised learning can be done with all extensions of these models natively, including on mixture model Bayes classifiers, mixed-distribution naive Bayes classifiers, using multi-threaded parallelism, and utilizing a GPU. Below is a simple example. Notice that there is no difference in the `from_samples` call, the presence of -1 in the label vector is enough.

.. code-block:: python

	import numpy

	from sklearn.datasets import make_blobs
	from sklearn.model_selection import train_test_split
	from pomegranate import NaiveBayes, NormalDistribution

	n, d, m = 50000, 5, 10
	X, y = make_blobs(n, d, m, cluster_std=10)
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

	n_unlabeled = int(X_train.shape[0] * 0.999)
	idxs = numpy.random.choice(X_train.shape[0], size=n_unlabeled)
	y_train[idxs] = -1

	model = NaiveBayes.from_samples(NormalDistribution, X_train, y_train, verbose=True)

While HMMs can theoretically be trained on sequences of data that are only partially labeled, currently semi-supervised learning for HMMs means that some sequences are fully labeled, and some sequences have no labels at all. This means that instead of passing in a normal label vector as a list of lists such as `[[model.start, s1, s2, model.end], [model.start, s1, s1, model.end]]`, one would pass in a list of mixed list/None types, with lists defining the labels for labeled sequences, and None specifying that a sequence is unlabeled. For example, if the second sequence was unlabeled, one would pass in `[[model.start, s1, s2, model.end], None]` instead.

FAQ
---

Q. What ratio of unlabeled / labeled data is typically best?

A. It's hard to say. However, semi-supervised learning works best when the underlying distributions are **more complicated than the labeled data captures**. If your data is simple Gaussian blobs, not many samples are needed and adding in unlabeled samples likely will not help. However, if the true underlying distributions are some complex mixture of components but your labeled data looks like a simple blob, semi-supervised learning can help significantly. 


Q. If this uses EM, what's the difference between semi-supervised learning and a mixture model?

A. Semi-supervised learning is a middle ground between unsupervised learning and supervised learning. As such, it adds together the sufficient statistics from unsupervised learning (using the EM algorithm) and supervised learning (using MLE) to get the complete model. An immediate benefit of this is that since there is a supervised initialization, the learned components will always align with the intended classes instead of being randomly assigning class values.


Q. Can parallelism be used with semi-supervised learning? 

A. Yes. All aspects of pomegranate that can be used with naive Bayes classifiers or general Bayes classifiers can be used in the context of semi-supervised learning in the same way one would do so in supervised learning. One need only set the `n_jobs` parameters as normal. Literally the only difference for the user that the label vector now contains many `-1` values. 

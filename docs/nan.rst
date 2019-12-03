.. _nan:

Missing Values
==============

- `IPython Notebook Tutorial <https://github.com/jmschrei/pomegranate/blob/master/tutorials/C_Feature_Tutorial_4_Missing_Values.ipynb>`_

As of version 0.9.0, pomegranate supports missing values for almost all methods. This means that models can be fit to data sets that have missing values in them, inference can be done on samples that have missing values, and even structure learning can be done in the presence of missing values. Currently, this support exists in the form of calculating sufficient statistics with respect to only the variables that are present in a sample and ignoring the missing values, in contrast to imputing the missing values and using those for the estimation. 

Missing value support was added in a manner that requires the least user thought. All one has to do is add ``numpy.nan`` to mark an entry as missing for numeric data sets, or the string ``'nan'`` for string data sets. pomegranate will automatically handle missing values appropriately. The functions have been written in such a way to minimize the overhead of missing value support, by only acting differently when a missing value is found. However, it may take some models longer to do calculations in the presence of missing values than on dense data. For example, when calculating the log probability of a sample under a multivariate Gaussian distribution one can typically use BLAS or a GPU since a dot product is taken between the data and the inverse covariance matrix. Unfortunately, since missing data can occur in any of the columns, a new inverse covariance matrix has to be calculated for each sample and BLAS cannot be utilized at all. 

As an example, when fitting a ``NormalDistribution`` to a vector of data, the parameters are estimated simply by ignoring the missing values. A data set with 100 observations and 50 missing values would produce the same model as a data set comprised simply of the 100 observations. This comes into play when fitting multivariate models, like an ``IndependentComponentsDistribution``, because each distribution is fit to only the observations for their specific feature. This means that samples where some values are missing can still be utilized in the dimensions where they are observed. This can lead to more robust estimates that by imputing the missing values using the mean or median of the column.

Here is an example of fitting a univariate distribution to data sets with missing values:

.. code-block:: python

	>>> import numpy
	>>> from pomegranate import *
	>>>
	>>> X = numpy.random.randn(100)
	>>> X[75:] = numpy.nan
	>>>
	>>> NormalDistribution.from_samples(X)
	{
	    "frozen" :false,
	    "class" :"Distribution",
	    "parameters" :[
	        -0.0007138484812874587,
	        1.0288813172046551
	    ],
	    "name" :"NormalDistribution"
	}
	>>> NormalDistribution.from_samples(X[:75])
	{
	    "frozen" :false,
	    "class" :"Distribution",
	    "parameters" :[
	        -0.0007138484812874587,
	        1.0288813172046551
	    ],
	    "name" :"NormalDistribution"
	}

Multivariate Gaussian distributions take a slightly more complex approach. The means of each column are computed using the available data, but the covariance is calculated using sufficient statistics calculated from pairs of variables that exist in a sample. For example, if the sample was (2.0, 1.7, numpy.nan), then sufficient statistics would be calculated for the variance of the first and second variables as well as the covariance between the two, but nothing would be updated about the third variable. 

All univariate distributions return a probability of 1 for missing data. This is done to support inference algorithms in more complex models. For example, when running the forward algorithm in a hidden Markov model in the presence of missing data, one would simply ignore the emission probability for the steps where the symbol is missing. This means that when getting to the step when a missing symbol is being aligned to each of the states, the cost is simply the transition probability to that state, instead of the transition probability multiplied by the likelihood of that symbol under that states' distribution (or, equivalently, having a likelihood of 1.) Under a Bayesian network, the probability of a sample is just the product of probabilities under distributions where the sample is fully observed. 

See the tutorial for more examples of missing value support in pomegranate!


FAQ
---

Q. How do I indicate that a value is missing in a data set?

A. If it is a numeric data set, indicate that a value is missing using ``numpy.nan``. If it is strings (such as 'A', 'B', etc...) use the string ``'nan'``. If your strings are stored in a numpy array, make sure that the full string 'nan' is present. numpy arrays have a tendancy to truncate longer strings if they're defined over shorter strings (like an array containing 'A' and 'B' might truncate 'nan' to be 'n').


Q. Are all algorithms supported?

A. Almost all! The only known non-supported function is Chow-Liu tree building. You can fit a Gaussian Mixture Model, run k-means clustering, decode a sequence using the Viterbi algorithm for a hidden Markov model, and learn the structure of a Bayesian network on data sets with missing values now!


Q. It is much slower to fit models using multivariate Gaussian distributions to missing data. Why?

A. When calculating the log probability of a point with missing values, a new inverse covariance matrix needs to be calculated over the subset of variables that are observed. This is a double whammy for speed because you need to (1) invert a matrix once per sample, and (2) cannot use BLAS for the calculation since there is no fixed sized covariance matrix to operate with.


Q. Performance on data sets without missing values appears to be worse now. What should I do?

A. Please report it on the GitHub issue tracker or email me. I have tried to minimize overhead in as many places as I can, but I have not run speed tests on all cases. Please include a sample script, and the amount of time it took.

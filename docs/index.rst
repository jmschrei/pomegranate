.. Introduction documentation master file, created by
   sphinx-quickstart on Sun Oct 30 18:10:26 2016.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.


.. image:: logo/pomegranate-logo.png
	:width: 300px

|
 
.. image:: https://travis-ci.org/jmschrei/pomegranate.svg?branch=master
   :target: https://travis-ci.org/jmschrei/pomegranate

.. image:: https://ci.appveyor.com/api/projects/status/github/jmschrei/pomegranate?svg=True
   :target: https://ci.appveyor.com/project/JacobSchreiber/pomegranate/branch/master

.. image:: https://readthedocs.org/projects/pomegranate/badge/?version=latest
   :target: http://pomegranate.readthedocs.io/en/latest/?badge=latest

|


Home
====

pomegranate is a Python package that implements fast and flexible probabilistic models ranging from individual probability distributions to compositional models such as Bayesian networks and hidden Markov models. The core philosophy behind pomegranate is that all probabilistic models can be viewed as a probability distribution in that they all yield probability estimates for samples and can be updated given samples and their associated weights. The primary consequence of this view is that the components that are implemented in pomegranate can be stacked more flexibly than other packages. For example, one can build a Gaussian mixture model just as easily as building an exponential or log normal mixture model. But that's not all! One can create a Bayes classifier that uses different types of distributions on each features, perhaps modeling time-associated features using an exponential distribution and counts using a Poisson distribution. Lastly, since these compositional models themselves can be viewed as probability distributions, one can build a mixture of Bayesian networks or a hidden Markov model Bayes' classifier that makes predictions over sequences. 

In addition to a variety of probability distributions and models, pomegranate has a variety of built-in features that are implemented for all of the models. These include different training strategies such as semi-supervised learning, learning with missing values, and mini-batch learning. It also includes support for massive data supports with out-of-core learning, multi-threaded parallelism, and GPU support. 


Thank You
=========

No good project is done alone, and so I'd like to thank all the previous contributors to YAHMM, all the current contributors to pomegranate, and the many graduate students whom I have pestered with ideas and questions. 

Contributions
=============

Contributions are eagerly accepted! If you would like to contribute a feature then fork the master branch and be sure to run the tests before changing any code. Let us know what you want to do on the issue tracker just in case we're already working on an implementation of something similar. Also, please don't forget to add tests for any new functions. Please review the `Code of Conduct <https://pomegranate.readthedocs.io/en/latest/CODE_OF_CONDUCT.html>`_ before contributing. 

.. toctree::
   :maxdepth: 1
   :hidden:
   :caption: Getting Started

   self
   install.rst
   CODE_OF_CONDUCT.rst
   faq.rst
   whats_new.rst

.. toctree::
   :maxdepth: 1
   :hidden:
   :caption: Features

   api.rst
   ooc.rst
   io.rst
   semisupervised.rst
   parallelism.rst
   gpu.rst
   nan.rst
   callbacks.rst

.. toctree::
   :maxdepth: 1
   :hidden:
   :caption: Models

   Distributions.rst
   GeneralMixtureModel.rst
   HiddenMarkovModel.rst
   NaiveBayes.rst
   MarkovChain.rst
   BayesianNetwork.rst
   MarkovNetwork.rst
   FactorGraph.rst

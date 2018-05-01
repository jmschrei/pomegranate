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

.. image:: https://readthedocs.org/projects/pomegranate/badge/?version=latest
   :target: http://pomegranate.readthedocs.io/en/latest/?badge=latest

|


Home
====

pomegranate is a python package that implements fast and flexible probabilistic models ranging from individual probability distributions to compositional models such as Bayesian networks and hidden Markov models. Furthermore, pomegranate is flexible enough to allow a stacking of compositional models so that one can create a mixture of Bayesian networks or a hidden Markov model Bayes' classifier that allows for classification over sequences instead of fixed feature sets. These flexible models are paired with a variety of training features and strategies, including support for out-of-core, mini-batch, semi-supervised, and missing value learning that can all be done with built-in multi-threaded parallelism. Descriptions of each of these models and training strategies can be found in this documentation. 

Below we give a brief description of the models available in pomegranate.

The most basic level of probabilistic modeling is the simple probability distribution. If we're modeling language, this may be a simple distribution of all possible words that a person can say, with values being the frequency with which that person says them.

(1) :ref:`distributions`

The next level up are compositional models which use the simple distributions in more complex ways. A Markov chain can extend a simple probability distribution to say that the probability of a certain word depends on the word(s) which have been said previously. A hidden Markov model may say that the probability of a certain words depends on the latent/hidden state of the previous word, such as a noun usually follows an adjective.

(2) :ref:`markovchain`
(3) :ref:`naivebayes`
(4) :ref:`generalmixturemodel`
(5) :ref:`hiddenmarkovmodel`
(6) :ref:`bayesiannetwork`
(7) :ref:`factorgraph`

The third level are stacks of probabilistic models which can model even more complex phenomena. If a single hidden Markov model can capture a dialect of a language (such as a certain person's speech usage) then a mixture of hidden Markov models may fine tune this to be situation specific. For example, a person may use more formal language at work and more casual language when speaking with friends. By modeling this as a mixture of HMMs, we represent the person's language as a "mixture" of these dialects.

(8) GMM-HMMs
(9) Mixtures of Models
(10) Bayesian Classifiers of Models

Thank You
=========

No good project is done alone, and so I'd like to thank all the previous contributors to YAHMM and all the current contributors to pomegranate as well as the graduate students whom I have pestered with ideas. Contributions are eagerly accepted! If you would like to contribute a feature then fork the master branch and be sure to run the tests before changing any code. Let us know what you want to do on the issue tracker just in case we're already working on an implementation of something similar. Also, please don't forget to add tests for any new functions. 

.. toctree::
   :maxdepth: 1
   :hidden:
   :caption: Getting Started

   self
   install.rst
   faq.rst
   whats_new.rst

.. toctree::
   :maxdepth: 1
   :hidden:
   :caption: Features

   ooc.rst
   semisupervised.rst
   minibatch.rst
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
   FactorGraph.rst

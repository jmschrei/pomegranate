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

pomegranate is a python package which implements fast, efficient, and extremely flexible probabilistic models ranging from probability distributions to Bayesian networks to mixtures of hidden Markov models. The most basic level of probabilistic modeling is the a simple probability distribution. If we're modeling language, this may be a simple distribution over the frequency of all possible words a person can say. 

(1) :ref:`distributions`

The next level up are probabilistic models which use the simple distributions in more complex ways. A markov chain can extend a simple probability distribution to say that the probability of a certain word depends on the word(s) which have been said previously. A hidden Markov model may say that the probability of a certain words depends on the latent/hidden state of the previous word, such as a noun usually follows an adjective.

(2) :ref:`markovchain`
(3) :ref:`naivebayes`
(4) :ref:`generalmixturemodel`
(5) :ref:`hiddenmarkovmodel`
(6) :ref:`bayesiannetwork`
(7) :ref:`factorgraph`

The third level are stacks of probabilistic models which can model even more complex phenomena. If a single hidden Markov model can capture a dialect of a language (such as a certain persons speech usage) then a mixture of hidden Markov models may fine tune this to be situation specific. For example, a person may use more formal language at work and more casual language when speaking with friends. By modeling this as a mixture of HMMs, we represent the persons language as a "mixture" of these dialects.

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

.. Introduction documentation master file, created by
   sphinx-quickstart on Sun Oct 30 18:10:26 2016.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.


.. image:: logo/pomegranate-logo.png
	:width: 300px

|
 
.. image:: https://travis-ci.org/jmschrei/pomegranate.svg?branch=master
	:target: https://travis-ci.org/jmschrei/pomegranate

|


Home
====

pomegranate is a python package which implements fast, efficient, and extremely flexible probabilistic models ranging from probability distributions to Bayesian networks to mixtures of hidden Markov models. It is convenient to think of pomegranate's functionality on three levels. The first level is basic probability distributions:


(1) :ref:`distributions`


The second level are probabilistic models which are made up of probability distributions:

(2) :ref:`markovchain`
(3) :ref:`naivebayes`
(4) :ref:`generalmixturemodel`
(5) :ref:`hiddenmarkovmodel`
(6) :ref:`bayesiannetwork`
(7) :ref:`factorgraph`


The third level are stacks of probabilistic models:

(8) GMM-HMMs
(9) Mixtures of Models
(10) Bayesian Classifiers of Models

No good project is done alone, and so I'd like to thank all the previous contributors to YAHMM and all the current contributors to pomegranate as well as the many graduate students who I have pestered with ideas. Contributions are eagerly accepted! If you would like to contribute a feature then fork the master branch and be sure to run the tests before changing any code. Let us know what you want to do on the issue tracker just in case we're already working on an implementation of something similar. This way we can avoid any needless duplication of effort. Also, please don't forget to add tests for any new functions. 

.. toctree::
   :maxdepth: 0
   :hidden:

   self
   Distributions.rst
   MarkovChain.rst
   NaiveBayes.rst
   GeneralMixtureModel.rst
   HiddenMarkovModel.rst
   BayesianNetwork.rst
   FactorGraph.rst


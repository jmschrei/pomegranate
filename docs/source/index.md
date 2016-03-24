pomegranate
==========

[![Build Status](https://travis-ci.org/jmschrei/pomegranate.svg?branch=master)](https://travis-ci.org/jmschrei/pomegranate)

pomegranate is a package for graphical models and Bayesian statistics for Python, implemented in cython. It grew out of the [YAHMM](https://github.com/jmschrei/yahmm) package, where many of the components used could be rearranged to do other cool things. It currently supports:

* [Probability Distributions](probability.md)
* [Markov Chains](markovchain.md)
* [Naive Bayes](naivebayes.md)
* [General Mixture Models](gmm.md)
* [Hidden Markov Models](hmm.md)
* [Discrete Bayesian Networks](bayesnet.md)
* [Factor Graphs](factorgraph.md)
* [Finite State Machines](fsm.md)

Documentation and tutorials for each of these methods are present on the scrollbar to the left. IPython notebook examples and further tutorials are present in the github repository.

No good project is done alone, and so I'd like to thank all the previous contributors to YAHMM, and all the current contributors to pomegranate, including the graduate students who share my office I annoy on a regular basis by bouncing ideas off of.

## Installation

pomegranate is now pip installable! Install using `pip install pomegranate`. You can get the bleeding edge using the following:

```
git clone https://github.com/jmschrei/pomegranate.git
cd pomegranate
python setup.py install
```

Lastly, you can also download the zip and manually move the files into your site-packages folder (or your PYTHON_PATH, if you've changed it).

On Windows machines you may need to download a C++ compiler. This minimal version of Visual Studio 2008 works well: https://www.microsoft.com/en-us/download/details.aspx?id=44266

## Contributing

If you would like to contribute a feature then fork the master branch (fork the release if you are fixing a bug). Be sure to run the tests before changing any code. You'll need to have [nosetests](https://github.com/nose-devs/nose) installed. The following command will run all the tests:
```
nosetests -s -v tests/
```
Let us know what you want to do just in case we're already working on an implementation of something similar. This way we can avoid any needless duplication of effort. Also, please don't forget to add tests for any new functions. 

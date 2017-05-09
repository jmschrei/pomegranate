<img src="https://github.com/jmschrei/pomegranate/blob/master/docs/logo/pomegranate-logo.png" width=300>

[![Build Status](https://travis-ci.org/jmschrei/pomegranate.svg?branch=master)](https://travis-ci.org/jmschrei/pomegranate) ![Build Status](https://ci.appveyor.com/api/projects/status/github/jmschrei/pomegranate?svg=True)

pomegranate is a package for probabilistic and graphical models for Python, implemented in cython for speed. It grew out of the [YAHMM](https://github.com/jmschrei/yahmm) package, where many of the components used could be rearranged to do other cool things. It currently supports:

* Probability Distributions
* General Mixture Models
* Hidden Markov Models
* Naive Bayes
* Bayes Classifiers
* Markov Chains
* Discrete Bayesian Networks

To support the above algorithms, it has efficient implementations of the following:

* Kmeans
* Factor Graphs

See the tutorial below, or the more in depth tutorials in the `tutorials` folder with examples in IPython notebooks. See [the website](http://pomegranate.readthedocs.org/en/latest/) for further information.

No good project is done alone, and so I'd like to thank all the previous contributors to YAHMM, and all the current contributors to pomegranate, including the graduate students who share my office I annoy on a regular basis by bouncing ideas off of.
## Installation

### Dependencies

pomegranate requires:

```
- Cython (only if building from source)
- NumPy
- SciPy
- NetworkX
- joblib
```

To run the tests, you also must have `nose` installed.

### User Installation

pomegranate is now pip installable! Install using `pip install pomegranate`. You can get the bleeding edge using the following:

```
git clone https://github.com/jmschrei/pomegranate.git
cd pomegranate
python setup.py install
```

Lastly, you can also download the zip and manually move the files into your site-packages folder (or your PYTHON_PATH, if you've changed it).

To build from source on Windows machines, you may need to download a C++ compiler. For Python 2 this minimal version of Visual Studio 2008 works well: https://www.microsoft.com/en-us/download/details.aspx?id=44266. For Python 3 this version of the Visual Studio Build Tools has been reported to work: http://go.microsoft.com/fwlink/?LinkId=691126. 

If those do no work, it has been suggested that https://wiki.python.org/moin/WindowsCompilers may provide more information. Note that your compiler version must fit your python version. Run python --version to tell which python version you use. Don't forget to select the appropriate Windows version API you'd like to use. If you get an error message "ValueError: Unknown MS Compiler version 1900" remove your Python's Lib/distutils/distutil.cfg and retry. See http://stackoverflow.com/questions/34135280/valueerror-unknown-ms-compiler-version-1900 for details.

## Contributing

If you would like to contribute a feature then fork the master branch (fork the release if you are fixing a bug). Be sure to run the tests before changing any code. You'll need to have [nosetests](https://github.com/nose-devs/nose) installed. The following command will run all the tests:
```
python setup.py test
```
Let us know what you want to do just in case we're already working on an implementation of something similar. This way we can avoid any needless duplication of effort. Also, please don't forget to add tests for any new functions.

## Tutorial

Please take a look at the tutorials folder, which includes several tutorials on how to effectively use pomegranate!

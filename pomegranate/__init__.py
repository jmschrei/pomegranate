# __init__.py: pomegranate
# Contact: Jacob Schreiber <jmschreiber91@gmail.com>


"""
For detailed documentation and examples, see the README.
"""

import os

from .base import *
from .parallel import *

from .distributions import *
from .kmeans import Kmeans
from .gmm import GeneralMixtureModel
from .NaiveBayes import NaiveBayes
from .BayesClassifier import BayesClassifier
from .MarkovChain import MarkovChain
from .hmm import HiddenMarkovModel
from .BayesianNetwork import BayesianNetwork
from .FactorGraph import FactorGraph

__version__ = '0.11.2'

os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

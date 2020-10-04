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
from .MarkovNetwork import MarkovNetwork
from .FactorGraph import FactorGraph

__version__ = '0.13.4'

os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

def from_json(s):
	"""A robust loading method.

	This method can load an appropriately formatted JSON object from any model
	in pomegranate and return the appropriate object. This relies mostly on the
	'class' attribute in the JSON.

	Parameters
	----------
	s : str
		Either the filename of a JSON object or a string that is JSON formatted,
		as produced by any of the `to_json` methods in pomegranate.
	"""

	try:
		d = json.loads(s)
	except:
		try:
			with open(s, 'r') as f:
				d = json.load(f)
		except:
			raise IOError("String must be properly formatted JSON or filename of properly formatted JSON.")

	if d['class'] == 'Distribution':
		return Distribution.from_json(s)
	elif d['class'] == 'GeneralMixtureModel':
		return GeneralMixtureModel.from_json(s)
	elif d['class'] == 'HiddenMarkovModel':
		return HiddenMarkovModel.from_json(s)
	elif d['class'] == 'NaiveBayes':
		return NaiveBayes.from_json(s)
	elif d['class'] == 'BayesClassifier':
		return BayesClassifier.from_json(s)
	elif d['class'] == 'BayesianNetwork':
		return BayesianNetwork.from_json(s)
	elif d['class'] == 'MarkovChain':
		return MarkovChain.from_json(s)
	elif d['class'] == 'MarkovNetwork':
		return MarkovNetwork.from_json(s)
	elif d['class'] == 'FactorGraph':
		return FactorGraph.from_json(s)
	else:
		raise ValueError("Must pass in an JSON with a valid model name.")
# NeuralHMM.py
# Contact: Jacob Schreiber <jmschreiber91@gmail.com>

from .hmm import HiddenMarkovModel



class NeuralHMM(HiddenMarkovModel):
	"""This is a Neural HMM! It's cool.

	Stuff.
	"""

	def __init__(self, hmm=None, neural_network=None, name=None):
		self.hmm = hmm
		self.neural_network = neural_network
		self.name = name or str(id(self))

	def log_probability(self)
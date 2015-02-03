# rainy_sunny_hmm.py
# Contact: Jacob Schreiber
#		   jmschreiber91@gmail.com

"""
Example rainy-sunny HMM using yahmm. Example drawn from the wikipedia HMM
article: http://en.wikipedia.org/wiki/Hidden_Markov_model describing what
Bob likes to do on rainy or sunny days.
"""

from pomegranate import *
from pomegranate import HiddenMarkovModel as Model
import random
import math

random.seed(0)

model = Model( name="Rainy-Sunny" )

# Emission probabilities
rainy = State( DiscreteDistribution({ 'walk': 0.1, 'shop': 0.4, 'clean': 0.5 }), name='Rainy' )
sunny = State( DiscreteDistribution({ 'walk': 0.6, 'shop': 0.3, 'clean': 0.1 }), name='Sunny' )

model.add_transition( model.start, rainy, 0.6 )
model.add_transition( model.start, sunny, 0.4 )

# Transition matrix, with 0.05 subtracted from each probability to add to
# the probability of exiting the hmm
model.add_transition( rainy, rainy, 0.65 )
model.add_transition( rainy, sunny, 0.25 )
model.add_transition( sunny, rainy, 0.35 )
model.add_transition( sunny, sunny, 0.55 )

# Add transitions to the end of the model
model.add_transition( rainy, model.end, 0.1 )
model.add_transition( sunny, model.end, 0.1 )

# Finalize the model structure
model.bake( verbose=True )

# Lets sample from this model.
print model.sample()

# Lets call Bob every hour and see what he's doing!
# (aka build up a sequence of observations)
sequence = [ 'walk', 'shop', 'clean', 'clean', 'clean', 'walk', 'clean' ]

# What is the probability of seeing this sequence?
print "Probability of Sequence: ", \
	math.e**model.forward( sequence )[ len(sequence), model.end_index ]
print "Probability of Cleaning at Time Step 3 Given This Sequence: ", \
	math.e**model.forward_backward( sequence )[1][ 2, model.states.index( rainy ) ]
print "Probability of the Sequence Given It's Sunny at Time Step 4: ", \
	math.e**model.backward( sequence )[ 3, model.states.index( sunny ) ]

print " ".join( state.name for i, state in model.maximum_a_posteriori( sequence )[1] )

# cse515_hw2.py
# Contact: Jacob Schreiber
#          jmschr@cs.washington.edu

from pomegranate import *
import math
import numpy
numpy.set_printoptions( suppress=True )

high = State( DiscreteDistribution( {'S':0.7, 'R':0.1, 'C':0.2 } ), "High" )
low  = State( DiscreteDistribution( {'S':0.2, 'R':0.7, 'C':0.1 } ), "Low" )
mid = State( DiscreteDistribution( {'S':0.01, 'R':0.01, 'C':0.98} ), "Mid")

model = HiddenMarkovModel( "rainy-sunny" )
model.add_states([high, low, mid])

model.add_transition( model.start, high, 0.5 )
model.add_transition( model.start, low, 0.5 )
model.add_transition( high, high, 0.6 )
model.add_transition( high,  low, 0.4 )
model.add_transition( low,   low, 0.6 )
model.add_transition( low,  high, 0.4 )
model.bake()

seq = list('RSSSC')

print "PART (A)"
print " ".join( state.name for i, state in model.viterbi( seq )[1] if not state.is_silent() )

print "PART (B)"
print math.e**model.forward_backward( seq )[1]

print "PART (C)"
print "No, it's not."
print " ".join( state.name for i, state in model.maximum_a_posteriori( seq )[1] if not state.is_silent() )
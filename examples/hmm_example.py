#!/usr/bin/env python2.7
# example.py: Yet Another Hidden Markov Model library
# Contact: Jacob Schreiber ( jmschreiber91@gmail.com )

"""
A simple example highlighting how to build a model using states, add
transitions, and then run the algorithms, including showing how training
on a sequence improves the probability of the sequence.
"""

import random
from pomegranate import *
from pomegranate import HiddenMarkovModel as Model

random.seed(0)
model = Model(name="ExampleModel")
distribution = UniformDistribution(0.0, 1.0)
state = State(distribution, name="uniform")
state2 = State(NormalDistribution(0, 2), name="normal")
silent = State(None, name="silent")
model.add_state(state)
model.add_state(state2)

model.add_transition(state, state, 0.4)
model.add_transition(state, state2, 0.4)
model.add_transition(state2, state2, 0.4)
model.add_transition(state2, state, 0.4)

model.add_transition(model.start, state, 0.5)
model.add_transition(model.start, state2, 0.5)
model.add_transition(state, model.end, 0.2)
model.add_transition(state2, model.end, 0.2)

model.bake()
sequence = model.sample()
print sequence
print
print model.forward(sequence)[ len(sequence), model.end_index ]
print model.backward(sequence)[0,model.start_index]
print
trans, ems =  model.forward_backward(sequence)
print trans
print ems
print
model.train( [ sequence ] )

print
print model.forward(sequence)[ len(sequence), model.end_index ]
print model.backward(sequence)[0,model.start_index]
print
trans, ems = model.forward_backward(sequence)
print trans
print ems
print
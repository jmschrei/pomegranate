# infinite_hmm_sampling.py
# Contact: Jacob Schreiber
#          jmschreiber91@gmail.com

'''
This example shows how to use yahmm to sample from an infinite HMM. The premise
is that you have an HMM which does not have transitions to the end state, and
so can continue on forever. This is done by not adding transitions to the end
state. If you bake a model with no transitions to the end state, you get an
infinite model, with no extra work! This change is passed on to all the
algorithms.
'''

from pomegranate import *
from pomegranate import HiddenMarkovModel as Model
import itertools as it
import numpy as np

# Define the states
s1 = State( NormalDistribution( 5, 2 ), name="S1" )
s2 = State( NormalDistribution( 15, 2 ), name="S2" )
s3 = State( NormalDistribution( 25, 2 ), name="S3 ")

# Define the transitions
model = Model( "infinite" )
model.add_transition( model.start, s1, 0.7 )
model.add_transition( model.start, s2, 0.2 )
model.add_transition( model.start, s3, 0.1 )
model.add_transition( s1, s1, 0.6 )
model.add_transition( s1, s2, 0.1 )
model.add_transition( s1, s3, 0.3 )
model.add_transition( s2, s1, 0.4 )
model.add_transition( s2, s2, 0.4 )
model.add_transition( s2, s3, 0.2 )
model.add_transition( s3, s1, 0.05 )
model.add_transition( s3, s2, 0.15 )
model.add_transition( s3, s3, 0.8 )
model.bake()

sequence = [ 4.8, 5.6, 24.1, 25.8, 14.3, 26.5, 15.9, 5.5, 5.1 ]


print model.is_infinite()

print "Algorithms On Infinite Model"
sequence = [ 4.8, 5.6, 24.1, 25.8, 14.3, 26.5, 15.9, 5.5, 5.1 ]
print "Forward"
print model.forward( sequence )

print "\n".join( state.name for state in model.states )
print "Backward"
print model.backward( sequence )

print "Forward-Backward"
trans, emissions = model.forward_backward( sequence )
print trans
print emissions

print "Viterbi"
prob, states = model.viterbi( sequence )
print "Prob: {}".format( prob )
print "\n".join( state[1].name for state in states )
print
print "MAP"
prob, states = model.maximum_a_posteriori( sequence )
print "Prob: {}".format( prob )
print "\n".join( state[1].name for state in states )

print "Showing that sampling can reproduce the original transition probs."
print "Should produce a matrix close to the following: "
print " [ [ 0.60, 0.10, 0.30 ] "
print "   [ 0.40, 0.40, 0.20 ] "
print "   [ 0.05, 0.15, 0.80 ] ] "
print
print "Tranition Matrix From 100000 Samples:"
sample, path = model.sample( 100000, path=True )
trans = np.zeros((3,3))

for state, n_state in it.izip( path[1:-2], path[2:-1] ):
	state_name = float( state.name[1:] )-1
	n_state_name = float( n_state.name[1:] )-1
	trans[ state_name, n_state_name ] += 1

trans = (trans.T / trans.sum( axis=1 )).T
print trans
# tied_state_hmm.py
# Contact: Jacob Schreiber
#		   jmschreiber91@gmail.com

"""
An example of using tied states to represent the same distribution across 
multiple states. This example is a toy example derived from biology, where we 
will look at DNA sequences. The fake structure we will pretend exists is:

start -> background -> CG island -> background -> poly-T region

DNA is comprised of four nucleotides, A, C, G, and T. Lets say that in the
background sequence, all of these occur at the same frequency. In the CG
island, the nucleotides C and G occur more frequently. In the poly T region,
T occurs most frequently.

We need the graph structure, because we fake know that the sequence must return
to the background distribution between the CG island and the poly-T region.
However, we also fake know that both background distributions need to be the same
"""

from pomegranate import *
from pomgranate import HiddenMarkovModel as Model
import random
import numpy 
random.seed(0)

# Lets start off with an example without tied states and see what happens
print "Without Tied States"
print

model = Model( "No Tied States" )

# Define the four states
background_one = State( DiscreteDistribution({'A': 0.25, 'C':0.25, 'G': 0.25, 'T':0.25 }), name="B1" )
CG_island = State( DiscreteDistribution({'A': 0.1, 'C':0.4, 'G': 0.4, 'T':0.1 }), name="CG" )
background_two = State( DiscreteDistribution({'A': 0.25, 'C':0.25, 'G': 0.25, 'T':0.25 }), name="B2" )
poly_T = State( DiscreteDistribution({'A': 0.1, 'C':0.1, 'G': 0.1, 'T':0.7 }), name="PT" )

# Add all the transitions
model.add_transition( model.start, background_one, 1. )
model.add_transition( background_one, background_one, 0.9 )
model.add_transition( background_one, CG_island, 0.1 )
model.add_transition( CG_island, CG_island, 0.8 )
model.add_transition( CG_island, background_two, 0.2 )
model.add_transition( background_two, background_two, 0.8 )
model.add_transition( background_two, poly_T, 0.2 )
model.add_transition( poly_T, poly_T, 0.7 )
model.add_transition( poly_T, model.end, 0.3 )
model.bake( verbose=True )

# Define the sequences. Training must be done on a list of lists, not on a string,
# in order to allow strings of any length.
sequences = [ numpy.array(list("TAGCACATCGCAGCGCATCACGCGCGCTAGCATATAAGCACGATCAGCACGACTGTTTTT")),
	      numpy.array(list("TAGAATCGCTACATAGACGCGCGCTCGCCGCGCTCGATAAGCTACGAACACGATTTTTTA")),
	      numpy.array(list("GATAGCTACGACTACGCGACTCACGCGCGCGCTCCGCATCAGACACGAATATAGATAAGATATTTTTT")) ]


# Print the distributions before training
print
print "\n".join( "{}: {}".format( state.name, state.distribution ) 
	for state in model.states if not state.is_silent() )

# Train
model.train( sequences, stop_threshold=0.01 )

# Print the distributions after training
print
print "\n".join( "{}: {}".format( state.name, state.distribution ) 
	for state in model.states if not state.is_silent() )

print "-"*80

print "With Tied States"
print

model = Model( "Tied States" )

# Define the background distribution
background = DiscreteDistribution({'A': 0.25, 'C':0.25, 'G': 0.25, 'T':0.25 })

# Define the four states. Pass the background distribution to the the two
# background states. This is the only change you need to make.
background_one = State( background, name="B1" )
CG_island = State( DiscreteDistribution({'A': 0.1, 
	'C':0.4, 'G': 0.4, 'T':0.1 }), name="CG" )
background_two = State( background, name="B2" )
poly_T = State( DiscreteDistribution({'A': 0.1, 
	'C':0.1, 'G': 0.1, 'T':0.7 }), name="PT" )

# Add all the transitions
model.add_transition( model.start, background_one, 1. )
model.add_transition( background_one, background_one, 0.9 )
model.add_transition( background_one, CG_island, 0.1 )
model.add_transition( CG_island, CG_island, 0.8 )
model.add_transition( CG_island, background_two, 0.2 )
model.add_transition( background_two, background_two, 0.8 )
model.add_transition( background_two, poly_T, 0.2 )
model.add_transition( poly_T, poly_T, 0.7 )
model.add_transition( poly_T, model.end, 0.3 )
model.bake( verbose=True )

# Define the sequences. Training must be done on a list of lists, not on a string,
# in order to allow strings of any length.
sequences = [ numpy.array(list("TAGCACATCGCAGCGCATCACGCGCGCTAGCATATAAGCACGATCAGCACGACTGTTTTT")),
			  numpy.array(list("TAGAATCGCTACATAGACGCGCGCTCGCCGCGCTCGATAAGCTACGAACACGATTTTTTA")),
			  numpy.array(list("GATAGCTACGACTACGCGACTCACGCGCGCGCTCCGCATCAGACACGAATATAGATAAGATATTTTTT")) ]


# Print the distributions before training
print
print "\n".join( "{}: {}".format( state.name, state.distribution ) 
	for state in model.states if not state.is_silent() )

# Train
model.train( sequences, stop_threshold=0.01 )

# Print the distributions after training
print
print "\n".join( "{}: {}".format( state.name, state.distribution ) 
	for state in model.states if not state.is_silent() )
print
print "Notice that states B1 and B2 are the same after training with tied states, \
	not so without tied states"

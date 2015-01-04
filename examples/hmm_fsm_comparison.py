# Heads-Tails HMM FSM comparison
# Contact: Jacob Schreiber
#          jmschr@cs.washington.edu

from pomegranate import *

fair = State( DiscreteDistribution({ 'H' : 0.5, 'T' : 0.5 }), "fair" )
unfair = State( DiscreteDistribution({ 'H' : 0.75, 'T' : 0.25 }), "unfair" )

# Transition Probabilities
stay_same = 0.5
change = 1. - stay_same

# Create the HiddenMarkovModel instance and add the states
hmm = HiddenMarkovModel( "HT" )
hmm.add_states([fair, unfair])

# We don't know which coin he chose to start off with
hmm.add_transition( hmm.start, fair, 0.5 )
hmm.add_transition( hmm.start, unfair, 0.5 )

# However, we do know it's hard for him to switch
hmm.add_transition( fair, fair, stay_same )
hmm.add_transition( fair, unfair, change )

hmm.add_transition( unfair, unfair, stay_same )
hmm.add_transition( unfair, fair, change )

hmm.bake()


# Create the FiniteStateMachine instance and add the states
fsm = FiniteStateMachine( "HT" )
fsm.add_states([fair, unfair])

# We don't know which coin he chose to start off with
fsm.add_transition( fsm.start, fair, 0.5 )
fsm.add_transition( fsm.start, unfair, 0.5 )

fsm.add_transition( fair, fair, stay_same )
fsm.add_transition( fair, unfair, change )

fsm.add_transition( unfair, unfair, stay_same )
fsm.add_transition( unfair, fair, change )

fsm.bake()

sequence = [ 'H', 'H', 'T', 'T', 'H', 'T', 'H', 'T', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'T', 'T', 'H' ]

print "HMM Path"
print "\t".join( state.name for _, state in hmm.viterbi( sequence )[1] )

print "FSM Path"
for flip in sequence:
	fsm.step( flip )
	print fsm.current_state.name,
print
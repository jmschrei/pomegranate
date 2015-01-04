# Monty Hall Bayes Net Test
# Contact: Jacob Schreiber
#          jmschr@cs.washington.edu

'''
Lets test out the Bayesian Network framework to produce the Monty Hall problem,
but modified a little. The Monty Hall problem is basically a game show where a
guest chooses one of three doors to open, with an unknown one having a prize
behind it. Monty then opens another non-chosen door without a prize behind it,
and asks the guest if they would like to change their answer. Many people were
surprised to find that if the guest changed their answer, there was a 66% chance
of success as opposed to a 50% as might be expected if there were two doors.

This can be modelled as a Bayesian network with three nodes-- guest, prize, and
Monty, each over the domain of door 'A', 'B', 'C'. Monty is dependent on both
guest and prize, in that it can't be either of them. Lets extend this a little
bit to say the guest has an untrustworthy friend whose answer he will not go with.
'''

import math
from pomegranate import *

# Friends emisisons are completely random
guest = DiscreteDistribution( { 'A': 1./3, 'B': 1./3, 'C': 1./3 } )

# The actual prize is independent of the other distributions
prize = DiscreteDistribution( { 'A': 1./3, 'B': 1./3, 'C': 1./3 } )

# Monty is dependent on both the guest and the prize. 
monty = ConditionalDiscreteDistribution( {
	'A' : { 'A' : DiscreteDistribution({ 'A' : 0.0, 'B' : 0.5, 'C' : 0.5 }),
			'B' : DiscreteDistribution({ 'A' : 0.0, 'B' : 0.0, 'C' : 1.0 }),
			'C' : DiscreteDistribution({ 'A' : 0.0, 'B' : 1.0, 'C' : 0.0 }) },
	'B' : { 'A' : DiscreteDistribution({ 'A' : 0.0, 'B' : 0.0, 'C' : 1.0 }), 
			'B' : DiscreteDistribution({ 'A' : 0.5, 'B' : 0.0, 'C' : 0.5 }),
			'C' : DiscreteDistribution({ 'A' : 1.0, 'B' : 0.0, 'C' : 0.0 }) },
	'C' : { 'A' : DiscreteDistribution({ 'A' : 0.0, 'B' : 1.0, 'C' : 0.0 }),
			'B' : DiscreteDistribution({ 'A' : 1.0, 'B' : 0.0, 'C' : 0.0 }),
			'C' : DiscreteDistribution({ 'A' : 0.5, 'B' : 0.5, 'C' : 0.0 }) } 
	}, [guest, prize] )

# Make the states
s1 = State( guest, name="guest" )
s2 = State( prize, name="prize" )
s3 = State( monty, name="monty" )

# Make the bayes net, add the states, and the conditional dependencies.
network = BayesianNetwork( "test" )
network.add_states( [ s1, s2, s3 ] )
network.add_transition( s1, s3 )
network.add_transition( s2, s3 )
network.bake()

print "\t".join([ state.name for state in network.states ])
print
print "Guest says 'A'"
observations = { 'guest' : 'A' }
beliefs = map( str, network.forward_backward( observations ) )
print "\n".join( "{}\t{}".format( state.name, belief ) for state, belief in zip( network.states, beliefs ) )
print
print "Guest says 'A', monty says 'B' (note that prize goes to 66% if you switch)"
observations = { 'guest' : 'A', 'monty' : 'B' }
beliefs = map( str, network.forward_backward( observations ) )
print "\n".join( "{}\t{}".format( state.name, belief ) for state, belief in zip( network.states, beliefs ) )
print
observations = { 'monty' : 'B' }
beliefs = map( str, network.forward_backward( observations ) )
print "\n".join( "{}\t{}".format( state.name, belief ) for state, belief in zip( network.states, beliefs ) )
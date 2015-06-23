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
monty = ConditionalProbabilityTable(
	[[ 'A', 'A', 'A', 0.0 ],
	 [ 'A', 'A', 'B', 0.5 ],
	 [ 'A', 'A', 'C', 0.5 ],
	 [ 'A', 'B', 'A', 0.0 ],
	 [ 'A', 'B', 'B', 0.0 ],
	 [ 'A', 'B', 'C', 1.0 ],
	 [ 'A', 'C', 'A', 0.0 ],
	 [ 'A', 'C', 'B', 1.0 ],
	 [ 'A', 'C', 'C', 0.0 ],
	 [ 'B', 'A', 'A', 0.0 ],
	 [ 'B', 'A', 'B', 0.0 ],
	 [ 'B', 'A', 'C', 1.0 ],
	 [ 'B', 'B', 'A', 0.5 ],
	 [ 'B', 'B', 'B', 0.0 ],
	 [ 'B', 'B', 'C', 0.5 ],
	 [ 'B', 'C', 'A', 1.0 ],
	 [ 'B', 'C', 'B', 0.0 ],
	 [ 'B', 'C', 'C', 0.0 ],
	 [ 'C', 'A', 'A', 0.0 ],
	 [ 'C', 'A', 'B', 1.0 ],
	 [ 'C', 'A', 'C', 0.0 ],
	 [ 'C', 'B', 'A', 1.0 ],
	 [ 'C', 'B', 'B', 0.0 ],
	 [ 'C', 'B', 'C', 0.0 ],
	 [ 'C', 'C', 'A', 0.5 ],
	 [ 'C', 'C', 'B', 0.5 ],
	 [ 'C', 'C', 'C', 0.0 ]], [guest, prize] ) 

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
print "Guest says 'A' and Prize is 'A'"
observations = { 'guest' : 'A', 'prize' : 'A' }
beliefs = map( str, network.forward_backward( observations ) )
print "\n".join( "{}\t{}".format( state.name, belief ) for state, belief in zip( network.states, beliefs ) )

print "Network Training"
data = [[ 'A', 'A', 'A' ],
		[ 'A', 'A', 'A' ],
		[ 'A', 'A', 'A' ],
		[ 'A', 'A', 'A' ],
		[ 'A', 'A', 'A' ],
		[ 'B', 'B', 'B' ],
		[ 'B', 'B', 'C' ],
		[ 'C', 'C', 'A' ],
		[ 'C', 'C', 'C' ],
		[ 'C', 'C', 'C' ],
		[ 'C', 'C', 'C' ],
		[ 'C', 'B', 'A' ]]

network.train( data )

print "Guest says 'A' and Prize is 'A'"
observations = { 'guest' : 'A', 'prize' : 'A' }
beliefs = map( str, network.forward_backward( observations ) )
print "\n".join( "{}\t{}".format( state.name, belief ) for state, belief in zip( network.states, beliefs ) )
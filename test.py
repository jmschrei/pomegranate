import math
from pomegranate import *
'''
a = DiscreteDistribution( { 'T': 0.90, 'H': 0.10 } )
b = DiscreteDistribution( { 'T': 0.40, 'H': 0.60 } )

c = ConditionalDiscreteDistribution(
	{ 'T' : { 'T' : DiscreteDistribution( { 'T' : 0.99, 'H' : 0.01 } ),
			  'H' : DiscreteDistribution( { 'T' : 0.20, 'H' : 0.80 } ) },
	  'H' : { 'T' : DiscreteDistribution( { 'T' : 0.67, 'H' : 0.33 } ),
	  		  'H' : DiscreteDistribution( { 'T' : 0.37, 'H' : 0.63 } ) } 
	}, [ a, b ], [ 'T', 'H' ] )

s1 = State( a, name="s1" )
s2 = State( b, name="s2" )
s3 = State( c, name="s3" )

network = BayesianNetwork( "test" )
network.add_states( [s1, s2, s3] )
network.add_transition( s1, s3, 1.0 )
network.add_transition( s2, s3, 1.0 )
network.bake()

print "RANDOM EXAMPLE"
print
print math.e ** c.log_probability( 'T', { a : 'T', b : 'H', c : 'T' } )
print
print math.e ** network.log_probability( { "s1" : 'T', "s2" : 'H', "s3" : 'T' } )
print
print network.belief_propogation( { "s1": 'T', "s2": 'H' } )
'''


friend = DiscreteDistribution( { 'A': 1./3, 'B': 1./3, 'C': 1./3 } )

guest = ConditionalDiscreteDistribution( {
	'A' : DiscreteDistribution({ 'A' : 0.0, 'B' : 0.5, 'C' : 0.5 }),
	'B' : DiscreteDistribution({ 'A' : 0.5, 'B' : 0.0, 'C' : 0.5 }),
	'C' : DiscreteDistribution({ 'A' : 0.5, 'B' : 0.5, 'C' : 0.0 })
	}, [friend])

prize = DiscreteDistribution( { 'A': 1./3, 'B': 1./3, 'C': 1./3 } )

# Encoding is CHOSEN : ACTUAL : MONTY 
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

s0 = State( friend, name="friend" )
s1 = State( guest, name="guest" )
s2 = State( prize, name="prize" )
s3 = State( monty, name="monty" )

network = BayesianNetwork( "test" )
network.add_states( [ s0, s1, s2, s3 ] )
network.add_transition( s0, s1, 1.0 )
network.add_transition( s1, s3, 1.0 )
network.add_transition( s2, s3, 1.0 )
network.bake()

print "\t".join([ state.name for state in network.states ])


#print monty.marginal( wrt=guest, value='A' )

print "\n".join( map( str, network.forward_backward( { 'monty' : 'A', 'friend' : 'B' } ) ) )


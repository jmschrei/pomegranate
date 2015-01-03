import math
from pomegranate import *

#######################
# MODIFIED MONTY HALL #
#######################
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
print "\n".join( map( str, network.forward_backward( { 'friend' : 'A', 'monty' : 'B' } ) ) )
'''
################
# ASIA EXAMPLE #
################

asia = DiscreteDistribution({ 'True' : 0.5, 'False' : 0.5 })
tuberculosis = ConditionalDiscreteDistribution({
	'True' : DiscreteDistribution({ 'True' : 0.2, 'False' : 0.80 }),
	'False' : DiscreteDistribution({ 'True' : 0.01, 'False' : 0.99 })
	}, [asia])

smoking = DiscreteDistribution({ 'True' : 0.5, 'False' : 0.5 })
lung = ConditionalDiscreteDistribution({
	'True' : DiscreteDistribution({ 'True' : 0.75, 'False' : 0.25 }),
	'False' : DiscreteDistribution({ 'True' : 0.02, 'False' : 0.98 })
	}, [smoking] )
bronchitis = ConditionalDiscreteDistribution({
	'True' : DiscreteDistribution({ 'True' : 0.92, 'False' : 0.08 }),
	'False' : DiscreteDistribution({ 'True' : 0.03, 'False' : 0.97})
	}, [smoking] )

tuberculosis_or_cancer = ConditionalDiscreteDistribution({
	'True' : { 'True' : DiscreteDistribution({ 'True' : 1.0, 'False' : 0.0 }),
			   'False' : DiscreteDistribution({ 'True' : 1.0, 'False' : 0.0 }),
			 },
	'False' : { 'True' : DiscreteDistribution({ 'True' : 1.0, 'False' : 0.0 }),
				'False' : DiscreteDistribution({ 'True' : 0.0, 'False' : 1.0 })
			  }
	}, [tuberculosis, lung] )

xray = ConditionalDiscreteDistribution({
	'True' : DiscreteDistribution({ 'True' : .885, 'False' : .115 }),
	'False' : DiscreteDistribution({ 'True' : 0.04, 'False' : 0.96 })
	}, [tuberculosis_or_cancer] )

dyspnea = ConditionalDiscreteDistribution({
	'True' : { 'True' : DiscreteDistribution({ 'True' : 0.96, 'False' : 0.04 }),
			   'False' : DiscreteDistribution({ 'True' : 0.89, 'False' : 0.11 })
			 },
	'False' : { 'True' : DiscreteDistribution({ 'True' : 0.82, 'False' : 0.18 }),
	            'False' : DiscreteDistribution({ 'True' : 0.4, 'False' : 0.6 })
	          }
	}, [tuberculosis_or_cancer, bronchitis])

s0 = State( asia, name="asia" )
s1 = State( tuberculosis, name="tuberculosis" )
s2 = State( smoking, name="smoker" )
s3 = State( lung, name="cancer" )
s4 = State( bronchitis, name="bronchitis" )
s5 = State( tuberculosis_or_cancer, name="TvC" )
s6 = State( xray, name="xray" )
s7 = State( dyspnea, name='dyspnea' )

network = BayesianNetwork( "asia" )
network.add_states([ s0, s1, s2, s3, s4, s5, s6, s7 ])
network.add_transition( s0, s1, 1.0 )
network.add_transition( s1, s5, 1.0 )
network.add_transition( s2, s3, 1.0 )
network.add_transition( s2, s4, 1.0 )
network.add_transition( s3, s5, 1.0 )
network.add_transition( s5, s6, 1.0 )
network.add_transition( s5, s7, 1.0 )
network.add_transition( s4, s7, 1.0 )
network.bake()

print "\t".join([ state.name for state in network.states ])
print "\n".join( map( str, network.forward_backward({ 'tuberculosis' : 'True', 'smoker' : 'False', 'bronchitis' : DiscreteDistribution({ 'True' : 0.8, 'False' : 0.2 }) }) ) )

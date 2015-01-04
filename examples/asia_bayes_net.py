# Asia Bayes Net
# Contact: Jacob Schreiber
#          jmschr@cs.washington.edu

'''
The Asia Bayesian Network. See a description here:  
http://www.norsys.com/tutorials/netica/secA/tut_A1.htm
'''

from pomegranate import *

# Create the distributions
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

# Make the states. Note the name can be different than the name of the state
# can be different than the name of the distribution
s0 = State( asia, name="asia" )
s1 = State( tuberculosis, name="tuberculosis" )
s2 = State( smoking, name="smoker" )
s3 = State( lung, name="cancer" )
s4 = State( bronchitis, name="bronchitis" )
s5 = State( tuberculosis_or_cancer, name="TvC" )
s6 = State( xray, name="xray" )
s7 = State( dyspnea, name='dyspnea' )

# Create the Bayesian network
network = BayesianNetwork( "asia" )
network.add_states([ s0, s1, s2, s3, s4, s5, s6, s7 ])
network.add_transition( s0, s1 )
network.add_transition( s1, s5 )
network.add_transition( s2, s3 )
network.add_transition( s2, s4 )
network.add_transition( s3, s5 )
network.add_transition( s5, s6 )
network.add_transition( s5, s7 )
network.add_transition( s4, s7 )
network.bake()

print "Has tuberculosis, is not a smoker, 80-20 chance he has bronchitis"
observations = { 'tuberculosis' : 'True', 'smoker' : 'False', 
				 'bronchitis' : DiscreteDistribution({ 'True' : 0.8, 'False' : 0.2 }) }
beliefs = map( str, network.forward_backward( observations ) )
print "\n".join( "{}\t\t{}".format( state.name, belief ) for state, belief in zip( network.states, beliefs ) )
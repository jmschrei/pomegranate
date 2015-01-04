# FSM test
# Contact: Jacob Schreiber
#          jmschr@cs.washington.edu

from pomegranate import *

# Create the states in the same way as you would an HMM
a = State( NormalDistribution( 5, 1 ), "a" )
b = State( NormalDistribution( 23, 1 ), "b" )
c = State( NormalDistribution( 100, 1 ), "c" )

# Create a FiniteStateMachine object 
model = FiniteStateMachine( "test" )

# Add the states in the same way
model.add_states( [a, b, c] )

# Add the transitions in the same manner
model.add_transition( model.start, a, 1.0 )
model.add_transition( a, a, 0.33 )
model.add_transition( a, b, 0.33 )
model.add_transition( b, b, 0.5 )
model.add_transition( b, a, 0.5 )
model.add_transition( a, c, 0.33 )
model.add_transition( c, a, 0.5 )
model.add_transition( c, c, 0.5 )

# Bake the model in the same way
model.bake( verbose=True )

# Take a sequence of observations
seq = [ 5, 5, 5, 5, 23, 23, 5, 23, 23, 100, 23, 23, 23, 23, 5, 5, 100, 5, 23 ]

# Print out the model 
print "\n".join( state.name for state in model.states )

# Print out where you start in the model
print model.current_state.name

# Print out where the model is for each step
for symbol in seq:
	model.step( symbol )
	print symbol, model.current_state.name
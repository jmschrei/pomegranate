# FSM test
# Contact: Jacob Schreiber
#          jmschr@cs.washington.edu

from pomegranate import *

# Create the states in the same way as you would an HMM
a = State( None, "5"  )
b = State( None, "10" )
c = State( None, "15" )
d = State( None, "20" )
e = State( None, "25" )

# Create a FiniteStateMachine object 
model = FiniteStateMachine( "Turnstile" )

# Add the states in the same way
model.add_states( [a, b, c, d, e] )

# Add in transitions by using nickels
model.add_transition( model.start, a, 5 )
model.add_transition( a, b, 5 )
model.add_transition( b, c, 5 )
model.add_transition( c, d, 5 )
model.add_transition( d, e, 5 )

# Add in transitions using dimes
model.add_transition( model.start, b, 10 )
model.add_transition( a, c, 10 )
model.add_transition( b, d, 10 )
model.add_transition( c, e, 10 )

# Add in transitions using quarters
model.add_transition( model.start, e, 25 )

# Bake the model in the same way
model.bake()

# Take a sequence of observations
seq = [ 5, 25, 10 ]


# Print out where you start in the model
print model.current_state.name

# Print out where the model is for each step
for symbol in seq:
	i = model.current_state.name
	model.step( symbol )
	print "Inserted {}: Moving from {} to {}.".format( symbol, i, model.current_state.name )
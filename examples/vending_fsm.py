# Vending Machine FSM
# Contact: Jacob Schreiber
#          jmschr@cs.washington.edu

from pomegranate import *

# Create the states in the same way as you would an HMM
a = State( DiscreteDistribution({  0 : 1.0 }), name="0" )
b = State( DiscreteDistribution({  5 : 1.0 }), name="5" )
c = State( DiscreteDistribution({  5 : 1.0 }), name="10a" )
d = State( DiscreteDistribution({  5 : 1.0 }), name="15a" )
e = State( DiscreteDistribution({ 10 : 1.0 }), name="10b" )
f = State( DiscreteDistribution({ 10 : 1.0 }), name="15b" )

# Create a FiniteStateMachine object 
model = FiniteStateMachine( "Vending Machine", start=a )

# Add the states to the machine
model.add_states( [a, b, c, d, e, f] )

# Connect the states according to possible transitions
model.add_transition( a, b, 0.33 )
model.add_transition( a, a, 0.33 )
model.add_transition( a, e, 0.33 )
model.add_transition( b, c, 0.5 )
model.add_transition( b, f, 0.5 )
model.add_transition( c, e, 1.0 )
model.add_transition( d, a, 1.0 )
model.add_transition( e, d, 0.5 )
model.add_transition( e, f, 0.5 )
model.add_transition( f, a, 1.0 )

# Bake the model in the same way
model.bake( merge=False )

# Take a sequence of observations
seq = [ 5, 5, 5, 0, 0 ]

# Print out where you start in the model
print "Start", model.current_state.name

# Print out where the model is for each step
for symbol in seq:
	model.step( symbol )
	print symbol, model.current_state.name

seq = [ 5, 10, 0 ]

print
# Print out where you start in the model
print "Start", model.current_state.name

# Print out where the model is for each step
for symbol in seq:
	model.step( symbol )
	print symbol, model.current_state.name
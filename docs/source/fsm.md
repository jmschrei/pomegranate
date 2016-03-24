Finite State Machines
====================

[Finite state machines](http://en.wikipedia.org/wiki/Finite-state_machine) are computational machines which can be in one of many states. The machine can be defined as a graphical model where the states are the states of the machine, and the edges define the transitions from each state to the other states in the machine. As the machine receives data, the state which it is in changes in a greedy fashion. Since the machine can be in only one state at a time and is memoryless, it is extremely useful. A classic example is a turnstile, which can take in nickels, dimes, and quarters, but only needs 25 cents to pass.

```python
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
```

In the above example, the name of the states encodes information about the state, and the edges each hold keys as to what cas pass along them. There are no distributions on these states, as this is not a probabilistic model, but distributions can be added without breaking the code if they are useful information to have on each state. 

```python
# Take a sequence of observations
seq = [ 5, 25, 10 ]


# Print out where you start in the model
print model.current_state.name

# Print out where the model is for each step
for symbol in seq:
	i = model.current_state.name
	model.step( symbol )
	print "Inserted {}: Moving from {} to {}.".format( symbol, i, model.current_state.name )
```
yields
```
Turnstile-start
Inserted 5: Moving from Turnstile-start to 5.
Inserted 5: Moving from 5 to 10.
Inserted 5: Moving from 10 to 15.
Inserted 5: Moving from 15 to 20.
Inserted 5: Moving from 20 to 25.
```

As we add nickles, we progress through the machine. But if we restarted and tried to do something invalid, we get the following, without progressing in the state machine:

```python
# Take a sequence of coins to add to the model
seq = [ 5, 25, 10 ]

# Print out where you start in the model
print model.current_state.name

# Print out where the model is for each step
for symbol in seq:
	i = model.current_state.name
	model.step( symbol )
	print "Inserted {}: Moving from {} to {}.".format( symbol, i, model.current_state.name )
```

yields

```
Turnstile-start
Inserted 5: Moving from Turnstile-start to 5.
Exception SyntaxError('No edges leaving state 5 with key 25') in 'pomegranate.fsm.FiniteStateMachine._step' ignored
Inserted 10: Moving from 5 to 15.
```

Presumably there would be client code surrounding the state machine to see where it is at each position, and do something based on which state it is currently in.

API Reference
=============

```eval_rst
.. automodule:: pomegranate.fsm
	:members:
```

pomegranate
==========

pomegranate is a package for graphical models and Bayesian statistics for Python, implemented in cython. It grew out of the [YAHMM](https://github.com/jmschrei/yahmm) package, where many of the components used could be rearranged to do other cool things. It currently supports:

* Probability Distributions
* Finite State Machines
* Hidden Markov Models
* Discrete Bayesian Networks

## Installation

I am currently having issues getting pomgranate to be pip installable, stemming from the use of multiple cython files. If anyone knows the fix, please submit a working pull request and I will send you a picture of a pomegranate and other assorted fruit of varying deliciousness.

However, cloning the repo or downloading the zip and manually moving the files into your site-packages folder does work.

## Contributing

If you would like to contribute a feature then fork the master branch (fork the release if you are fixing a bug). Be sure to run the tests before changing any code. You'll need to have [nosetests](https://github.com/nose-devs/nose) installed. The following command will run all the tests:
```
nosetests -w tests/
```
Let us know what you want to do just in case we're already working on an implementation of something similar. This way we can avoid any needless duplication of effort. Also, please don't forget to add tests for any new functions. 

## Tutorial

# Probability Distributions

There have been many times when I've had to simply calculate the probability of some data points given a distribution, or had to fit a distribution to some given data. pomegranate offers a simple solution, which is an extensible library of univariate distributions and kernel densities, and a multivariate distribution natively built in.

```
from pomegranate import *

a = NormalDistribution( 5, 2 )
b = TriangleKernelDensity( [1,5,2,3,4], weights=[4,2,6,3,1] )
c = MixtureDistribution( [ NormalDistribution( 2, 4 ), ExponentialDistribution( 8 ) ], weights=[1, 0.01] )

print a.log_probability( 8 )
print b.log_probability( 8 )
print c.log_probability( 8 )
```

This should return -2.737, -inf, and -3.44 respectively.  We can also update these distributions using Maximum Likelihood Estimates for the new values. Kernel densities will discard previous points and add in the new points, while MixtureDistributions will perform expectation-maximization.

```
c.from_sample([1, 5, 7, 3, 2, 4, 3, 5, 7, 8, 2, 4, 6, 7, 2, 4, 5, 1, 3, 2, 1])
c
```

This should result in `MixtureDistribution( [NormalDistribution(3.916, 2.132), ExponentialDistribution(0.99955)], [0.9961, 0.00386] )`. All distributions can be trained either as a batch using `from_sample`, or using summary statistics using `summarize` on lists of numbers until all numbers have been fed in, and then `from_summaries` like in the following example which produces the same result:

```
c = MixtureDistribution( [ NormalDistribution( 2, 4 ), ExponentialDistribution( 8 ) ], weights=[1, 0.01] )
c.summarize([1, 5, 7, 3, 2, 4, 3])
c.summarize([5, 7, 8])
c.summarize([2, 4, 6, 7, 2, 4, 5, 1, 3, 2, 1])
c.from_summaries()
```

In addition, training can be done on weighted samples by passing an array of weights in along with the data for any of the training functions, such as `c.summarize([5,7,8], weights=[1,2,3])`. Training can also be done with inertia, where the new value will be some percentage the old value and some percentage the new value, used like `c.from_sample([5,7,8], inertia=0.5)` to indicate a 50-50 split between old and new values. 

# Finite State Machines

[Finite state machines](http://en.wikipedia.org/wiki/Finite-state_machine) are 'machines' which can be in one of many 'states'. These states are defined as a graphical model, and as the machine receives data, the state which it is changes in a greedy fashion. Since the machine can be in only one state at a time and is memoryless, it is extremely useful for some computations. A classic example is a vending machine, which takes in coins and when it reaches the specified amount, will reset and dispense an item.

```
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
```

In the above example, the 'states' we model are the value of the new coin added. We can either add three nickles, or a nickle and a dime, or a dime and a nickle. Each states name is the cumulative amount, and emission is the value of the new coin. When it reaches 15, it will reset by seeing a 0 value coin. For example:
```
# Take a sequence of coins to add to the model
seq = [ 5, 5, 5 ]

# Print out where you start in the model
print "Start", model.current_state.name

# Print out where the model is for each step
for symbol in seq:
	model.step( symbol )
	print symbol, model.current_state.name
```

yields

```
Start 0
5 5
5 10a
5 15a
```

As we add nickles, we the machine progresses to new states until it reaches 15, which is what we want in this example. We can return to the beginning of the machine from there by appending a 0 to the list, as follows:
```
# Take a sequence of coins to add to the model
seq = [ 5, 5, 5 ]

# Print out where you start in the model
print "Start", model.current_state.name

# Print out where the model is for each step
for symbol in seq:
	model.step( symbol )
	print symbol, model.current_state.name
```
yields
```
Start 0
5 5
5 10a
5 15a
0 0
```

We could also have a dime in there.

```
# Take a sequence of coins to add to the model
seq = [ 10, 5, 0 ]

# Print out where you start in the model
print "Start", model.current_state.name

# Print out where the model is for each step
for symbol in seq:
	model.step( symbol )
	print symbol, model.current_state.name
```
yields
```
Start 0
10 10a
5 15b
0 0
```

Presumably the action of dispensing the drink would accompany the machine being in state 15a or 15b, but that's up to the application on top of the state machine.

# Hidden Markov Models

The hidden Markov models in pomegranate support the forward, backward, forward-backward, viterbi decoding,  and maximum-a-posteriori decoding algorithms. They also support labelled sequence, Viterbi, and Baum-Welch training, with tied edges and emissions, edge and emission inertia and edge pseudocounts. Extensive documentation is provided as a part of the [YAHMM](https://github.com/jmschrei/yahmm) project. All previous projects which used YAHMM HMMs can be converted to use pomegranate HMMs by simply writing 

```
from pomegranate import HiddenMarkovModel as Model
```

as the last import from pomegranate instead of

```
from yahmm import *
```

An example of doing pairwise sequence alignment using a profile HMM is shown below. This is an example which is part of YAHMM's example set.

```
from pomegranate import *
model = HiddenMarkovModel( "Global Sequence Aligner" )

# Define the distribution for insertions
i_d = DiscreteDistribution( { 'A': 0.25, 'C': 0.25, 'G': 0.25, 'T': 0.25 } )

# Create the insert states
i0 = State( i_d, name="I0" )
i1 = State( i_d, name="I1" )
i2 = State( i_d, name="I2" )
i3 = State( i_d, name="I3" )

# Create the match states
m1 = State( DiscreteDistribution({ "A": 0.95, 'C': 0.01, 'G': 0.01, 'T': 0.02 }) , name="M1" )
m2 = State( DiscreteDistribution({ "A": 0.003, 'C': 0.99, 'G': 0.003, 'T': 0.004 }) , name="M2" )
m3 = State( DiscreteDistribution({ "A": 0.01, 'C': 0.01, 'G': 0.01, 'T': 0.97 }) , name="M3" )

# Create the delete states
d1 = State( None, name="D1" )
d2 = State( None, name="D2" )
d3 = State( None, name="D3" )

# Add all the states to the model
model.add_states( [i0, i1, i2, i3, m1, m2, m3, d1, d2, d3 ] )

# Create transitions from match states
model.add_transition( model.start, m1, 0.9 )
model.add_transition( model.start, i0, 0.1 )
model.add_transition( m1, m2, 0.9 )
model.add_transition( m1, i1, 0.05 )
model.add_transition( m1, d2, 0.05 )
model.add_transition( m2, m3, 0.9 )
model.add_transition( m2, i2, 0.05 )
model.add_transition( m2, d3, 0.05 )
model.add_transition( m3, model.end, 0.9 )
model.add_transition( m3, i3, 0.1 )

# Create transitions from insert states
model.add_transition( i0, i0, 0.70 )
model.add_transition( i0, d1, 0.15 )
model.add_transition( i0, m1, 0.15 )

model.add_transition( i1, i1, 0.70 )
model.add_transition( i1, d2, 0.15 )
model.add_transition( i1, m2, 0.15 )

model.add_transition( i2, i2, 0.70 )
model.add_transition( i2, d3, 0.15 )
model.add_transition( i2, m3, 0.15 )

model.add_transition( i3, i3, 0.85 )
model.add_transition( i3, model.end, 0.15 )

# Create transitions from delete states
model.add_transition( d1, d2, 0.15 )
model.add_transition( d1, i1, 0.15 )
model.add_transition( d1, m2, 0.70 ) 

model.add_transition( d2, d3, 0.15 )
model.add_transition( d2, i2, 0.15 )
model.add_transition( d2, m3, 0.70 )

model.add_transition( d3, i3, 0.30 )
model.add_transition( d3, model.end, 0.70 )

# Call bake to finalize the structure of the model.
model.bake()

for sequence in map( list, ('ACT', 'GGC', 'GAT', 'ACC') ):
    logp, path = model.viterbi( sequence )
    print "Sequence: '{}'  -- Log Probability: {} -- Path: {}".format(
        ''.join( sequence ), logp, " ".join( state.name for idx, state in path[1:-1] ) )
```

This should produce the following:

Sequence: 'ACT'  -- Log Probability: -0.513244900357 -- Path: M1 M2 M3
Sequence: 'GGC'  -- Log Probability: -11.0481012413 -- Path: I0 I0 D1 M2 D3
Sequence: 'GAT'  -- Log Probability: -9.12551967402 -- Path: I0 M1 D2 M3
Sequence: 'ACC'  -- Log Probability: -5.08795587886 -- Path: M1 M2 M3

Everything is exactly the same as in YAHMM, except the `Model` class is now `HiddenMarkovModel`.

# Bayesian Networks

Currently, only discrete Bayesian networks are supported. The forward, backward, and forward-backward (often called sum-product) algorithms are implemented using a factor-graph representation. 

Lets test out the Bayesian Network framework on the [Monty Hall problem](http://en.wikipedia.org/wiki/Monty_Hall_problem). The Monty Hall problem arose from the gameshow <i>Let's Make a Deal</i>, where a guest had to choose which one of three doors had a prize behind it. The twist was that after the guest chose, the host, originally Monty Hall, would then open one of the doors the guest did not pick and ask if the guest wanted to switch which door they had picked. Initial inspection may lead you to believe that if there are only two doors left, there is a 50-50 chance of you picking the right one, and so there is no advantage one way or the other. However, it has been proven both through simulations and analytically that there is in fact a 66% chance of getting the prize if the guest switches their door, regardless of the door they initially went with. 

We can reproduce this result using Bayesian networks with three nodes, one for the guest, one for the prize, and one for the door Monty chooses to open. The door the guest initially chooses and the door the prize is behind are completely random processes across the three doors, but the door which Monty opens is dependent on both the door the guest chooses (it cannot be the door the guest chooses), and the door the prize is behind (it cannot be the door with the prize behind it). 

```
import math
from pomegranate import *

# The guests initial door selection is completely random
guest = DiscreteDistribution( { 'A': 1./3, 'B': 1./3, 'C': 1./3 } )

# The door the prize is behind is also completely random
prize = DiscreteDistribution( { 'A': 1./3, 'B': 1./3, 'C': 1./3 } )

# However, the door Monty opens is dependent on both of the previous distributions
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

# State objects hold both the distribution, and a high level name.
s1 = State( guest, name="guest" )
s2 = State( prize, name="prize" )
s3 = State( monty, name="monty" )

# Create the Bayesian network object with a useful name
network = BayesianNetwork( "Monty Hall Problem" )

# Add the three states to the network 
network.add_states( [ s1, s2, s3 ] )

# Add transitions which represent conditional dependencies, where the second node is conditionally dependent on the first node (Monty is dependent on both guest and prize)
network.add_transition( s1, s3 )
network.add_transition( s2, s3 )
network.bake()
```

Bayesian Networks introduc a new distribution, the ConditionalDiscreteDistribution. This distribution takes in (1) a dictionary where each nested layer refers to the values one of the distributions it is dependent on takes and (2) a list of the  distribution objects it is dependent on in the order of the nesting in the dictionary. In the Monty Hall example, the monty distribution is dependent on both the guest and the prize distributions in that order. The first layer of nesting accounts for what happens when the guest chooses various doors, and the second layer indicates what happens when the prize is actually behind a certain door, for each of the 9 possibilities. 

In order to reproduce the final result, we need to take advantage of the forward-backward/sum-product algorithm. This algorithm allows the network to calculate posterior probabilities for each distribution in the network when as distributions get clamped to certain values. This means we do not need perfect information to do inference with Bayesian networks. This is done in pomegranate by feeding in a dictionary of state names and their associated values, and running `forward_backward`. 

```
observations = { 'guest' : 'A' }

# beliefs will be an array of posterior distributions or clamped values for each state, indexed corresponding to the order
# in self.states. 
beliefs = network.forward_backward( observations )

# Convert the beliefs into a more readable format
beliefs = map( str, beliefs )

# Print out the state name and belief for each state on individual lines
print "\n".join( "{}\t{}".format( state.name, belief ) for state, belief in zip( network.states, beliefs ) )
```
This will yield: 
```
prize	DiscreteDistribution({'A': 0.3333333333333335, 'C': 0.3333333333333333, 'B': 0.3333333333333333})
guest	A
monty	DiscreteDistribution({'A': 0.0, 'C': 0.5, 'B': 0.5})
```

Since we have clamped the guest distribution to 'A', it returns just that value. The prize distribution is unaffected. 
Since guest is clamped to 'A', it is forced to stay that way. Note that the prize distribution is unaffected, but that the monty distribution now puts a 0 probability on him saying A, since he will not open the same door the guest chose. 

In order to reproduce the final result, we need to see what happens when Monty opens a door. Lets clamp the Monty distribution to 'B' to indicate he has opened that door. 

```
observations = { 'guest' : 'A', 'monty' : 'B' }
beliefs = map( str, network.forward_backward( observations ) )
print "\n".join( "{}\t{}".format( state.name, belief ) for state, belief in zip( network.states, beliefs ) )
```
yields
```
guest	A
monty	B
prize	DiscreteDistribution({'A': 0.3333333333333333, 'C': 0.6666666666666666, 'B': 0.0})
```
Both guest and monty have been clamped to values. However, we see that probability of prize being 'C' is 66% mimicking the mystery behind the Monty hall problem!

This has predominately leveraged forward propogation of messages. If we want to see backward propogation of messages, lets see what happens if we tuned in late and only saw which door Monty opened.
```
observations = { 'monty' : 'B' }
beliefs = map( str, network.forward_backward( observations ) )
print "\n".join( "{}\t{}".format( state.name, belief ) for state, belief in zip( network.states, beliefs ) )
```
yields
```
guest	DiscreteDistribution({'A': 0.49999999999999994, 'C': 0.49999999999999994, 'B': 0.0})
monty	B
prize	DiscreteDistribution({'A': 0.49999999999999994, 'C': 0.49999999999999994, 'B': 0.0})
```

We know that if Monty opened door 'B', that the prize cannot be behind 'B' and that the guest could not have opened 'B'. The posterior guest and prize probabilities show this. 

Useful stuff.

Below is the [Asia example](http://www.norsys.com/tutorials/netica/secA/tut_A1.htm). It is more of the same, but seeing a more complicated network may help you get yours working!
```
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
```

Lets add in some information:

```
observations = { 'tuberculosis' : 'True', 
                 'smoker' : 'False', 
				         'bronchitis' : DiscreteDistribution({ 'True' : 0.8, 'False' : 0.2 }) 
				       }
beliefs = map( str, network.forward_backward( observations ) )
print "\n".join( "{}\t\t{}".format( state.name, belief ) for state, belief in zip( network.states, beliefs ) )
```

yields

```
xray		DiscreteDistribution({'True': 0.885, 'False': 0.115})
dyspnea		DiscreteDistribution({'True': 0.9460000000000002, 'False': 0.05400000000000003})
asia		DiscreteDistribution({'True': 0.9523809523809523, 'False': 0.04761904761904764})
tuberculosis		True
smoker		False
cancer		DiscreteDistribution({'False': 0.98, 'True': 0.02000000000000001})
bronchitis		DiscreteDistribution({'True': 0.811127248750323, 'False': 0.188872751249677})
TvC		DiscreteDistribution({'False': 0.0, 'True': 1.0})
```

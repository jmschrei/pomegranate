pomegranate
==========

pomegranate is a package for graphical models and Bayesian statistics for Python, implemented in cython. It currently supports:

* Probability Distributions
* Finite State Machines
* Hidden Markov Models
* Discrete Bayesian Networks

The hidden Markov model implementation is from the [YAHMM](https://github.com/jmschrei/yahmm). While developing it (with the help of other talented contributors), I noticed it may be useful to break it into components and build other cool graphical models with it, and so I'm presently working on that!

## Installation

I am currently having issues getting pomgranate to be pip installable, stemming from the use of multiple cython files. If anyone knows the fix, please submit a working pull request and I will send you a picture of a pomegranate and other assorted fruit of varying deliciousness.

However, cloning the repo or downloading the zip and manually moving the files into your site-packages folder does appear to work! 

## Contributing

If you would like to contribute a feature then fork the master branch (fork the release if you are fixing a bug). Be sure to run the tests before changing any code. You'll need to have [nosetests](https://github.com/nose-devs/nose) installed. The following command will run all the tests:
```
nosetests -w tests/
```
Let us know what you want to do just in case we're already working on an implementation of something similar. This way we can avoid any needless duplication of effort. Also, please don't forget to add tests for any new functions. 

## Tutorial

# Probability Distributions

pomegranate has an extensible library of univariate distributions and kernel densities along with a multivariate distribution natively built in. Here are a few examples:

```
from pomegranate import *

a = NormalDistribution( 5, 2 )
b = TriangleKernelDensity( [1,5,2,3,4], weights=[4,2,6,3,1] )
c = MixtureDistribution( [ NormalDistribution( 2, 4 ), ExponentialDistribution( 8 ) ], weights=[1, 0.01] )

print a.log_probability( 8 )
print b.log_probability( 8 )
print c.log_probability( 8 )
```

This should produce -2.737, -inf, -3.44 respectively.  We can also update these distributions! Kernel densities simply add the points, or discard previous points and replace them with the new points if training is done without inertia. MixtureDistributions will perform expectation-maximization to perform training.

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

This is useful if you need to quickly determine the probability of a float given a distribution or fit a univariate distribution to some data.

# Finite State Machines

Finite state machines are memoryless graphical models which means that intead of feeding it a sequence of observations, it is fed a single observation at a time, and its current state can be queried. Here is an example of a three state FSM which is not fully connected, showing its greedy behavior.

```
from pomegranate import *

# Create the states in the FSM, comprised of their emission
# distribution and a name
a = State( NormalDistribution( 5, 1 ), "a" )
b = State( NormalDistribution( 23, 1 ), "b" )
c = State( NormalDistribution( 100, 1 ), "c" )

# Create a FiniteStateMachine object 
model = FiniteStateMachine( "test" )

# Add the states to the model
model.add_states( [a, b, c] )

# Add the transitions and their associated probabilities.
model.add_transition( model.start, a, 1.0 )
model.add_transition( a, a, 0.33 )
model.add_transition( a, b, 0.33 )
model.add_transition( b, b, 0.5 )
model.add_transition( b, a, 0.5 )
model.add_transition( a, c, 0.33 )
model.add_transition( c, a, 0.5 )
model.add_transition( c, c, 0.5 )

# Bake the model to finalize the internal structure
model.bake( verbose=True )

# Take a sequence of observations
seq = [ 5, 5, 5, 5, 23, 23, 5, 23, 23, 100, 23, 23, 23, 23, 5, 5, 100, 5, 23 ]

# Print out where you start in the model
print model.current_state.name

# Print out where the model is for each step
for symbol in seq:
	model.step( symbol )
	print symbol, model.current_state.name
```

In the above example, we have a list of items we go iterate through. However, since FSMs are memoryless, they can be 'online' very easily, with data being fed in as actions are taken and the underlying state being used to prompt a response.

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

Currently, only discrete Bayesian networks are supported. The forward, backward, and forward-backward (often called sum-product) algorithms are implemented using a factor-graph representation. Given the conditional nature of the distributions in Bayesian networks, the code is slightly less clean.

Lets test out the Bayesian Network framework to produce the Monty Hall problem, but modified a little. The Monty Hall problem is basically a game show where a guest chooses one of three doors to open, with an unknown one having a prize behind it. Monty then opens another non-chosen door without a prize behind it, and asks the guest if they would like to change their answer. Many people were surprised to find that if the guest changed their answer, there was a 66% chance of success as opposed to a 50% as might be expected if there were two doors.

This can be modelled as a Bayesian network with three nodes-- guest, prize, and Monty, each over the domain of door 'A', 'B', 'C'. Monty is dependent on both guest and prize, in that it can't be either of them. Lets extend this a little bit to say the guest has an untrustworthy friend whose answer he will not go with.

```
import math
from pomegranate import *

# Friends emisisons are completely random
friend = DiscreteDistribution( { 'A': 1./3, 'B': 1./3, 'C': 1./3 } )

# The guest is conditioned on the friend, basically go against the friend
guest = ConditionalDiscreteDistribution( {
	'A' : DiscreteDistribution({ 'A' : 0.0, 'B' : 0.5, 'C' : 0.5 }),
	'B' : DiscreteDistribution({ 'A' : 0.5, 'B' : 0.0, 'C' : 0.5 }),
	'C' : DiscreteDistribution({ 'A' : 0.5, 'B' : 0.5, 'C' : 0.0 })
	}, [friend])

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
s0 = State( friend, name="friend" )
s1 = State( guest, name="guest" )
s2 = State( prize, name="prize" )
s3 = State( monty, name="monty" )

# Make the bayes net, add the states, and the conditional dependencies.
network = BayesianNetwork( "test" )
network.add_states( [ s0, s1, s2, s3 ] )
network.add_transition( s0, s1 )
network.add_transition( s1, s3 )
network.add_transition( s2, s3 )
network.bake()
```

The ConditionalDiscreteDistribution takes in (1) a dictionary, where each nested layer refers to the values one of the distributions it is dependant on takes and (2) a list of the other distribution objects it is dependant on. The `guest` distribution is a good example, where the keys of the dictionary passed in are the values the `friend` distribution can take, and the inner values are the distributions assuming the `friend` took that value. If the friend says 'A', then the guest will not say 'A' but has a 50-50 chance of saying 'B', or 'C'. The `monty` distribution is the same, except two layers of nesting because it is dependant on both the guest and the prize distributions.

Unlike a FSM or a HMM, we do not need to feed in perfect information. We can add in some information, and see how the distributions along the other values change. This is done by feeding in a dictionary of state names and their associated values, and running `forward_backward`. 

Lets see what happens when we clamp the guest to 'A'. 

```
observations = { 'guest' : 'A' }
beliefs = map( str, network.forward_backward( observations ) )
print "\n".join( "{}\t{}".format( state.name, belief ) for state, belief in zip( network.states, beliefs ) )
```

This will yield: 
```
friend	DiscreteDistribution({'A': 0.0, 'C': 0.49999999999999994, 'B': 0.49999999999999994})
prize	DiscreteDistribution({'A': 0.3333333333333335, 'C': 0.3333333333333333, 'B': 0.3333333333333333})
guest	A
monty	DiscreteDistribution({'A': 0.0, 'C': 0.5, 'B': 0.5})
```

Since guest is clamped to 'A', it is forced to stay that way. Note that the prize distribution is unaffected, but that the monty distribution now puts a 0 probability on him saying A, since he will not open the same door the guest chose. Lastly, this inference goes backwards and says that the friend could not have said 'A', because the guest intentionally does not do what the friend said.

Now lets see what the posterior probabilities are after Monty opens door 'B'.

```
observations = { 'guest' : 'A', 'monty' : 'B' }
beliefs = map( str, network.forward_backward( observations ) )
print "\n".join( "{}\t{}".format( state.name, belief ) for state, belief in zip( network.states, beliefs ) )
```
yields
```
friend	DiscreteDistribution({'A': 0.0, 'C': 0.49999999999999994, 'B': 0.49999999999999994})
prize	DiscreteDistribution({'A': 0.3333333333333333, 'C': 0.6666666666666666, 'B': 0.0})
guest	A
monty	B
```
Both guest and monty have been clamped to values. However, we see that probability of 'C' is now 0.667, mimicking the mystery behind the Monty hall problem!

Lastly, the values which are clamped do not need to be a single character, but rather can be a distribution! Lets see what happens if the friend has a 50% chance of saying 'B' and a 50% chance of saying 'A', but we don't know which.

```
observations = { 'friend' : DiscreteDistribution({ 'A' : 0.5, 'B' : 0.5, 'C' : 0.0 }) }
beliefs = map( str, network.forward_backward( observations ) )
print "\n".join( "{}\t{}".format( state.name, belief ) for state, belief in zip( network.states, beliefs ) )
```
yields
```
friend	DiscreteDistribution({'A': 0.5, 'C': 0.0, 'B': 0.5})
prize	DiscreteDistribution({'A': 0.3181818181818182, 'C': 0.36363636363636365, 'B': 0.3181818181818182})
guest	DiscreteDistribution({'A': 0.22727272727272732, 'C': 0.5454545454545452, 'B': 0.22727272727272732})
monty	DiscreteDistribution({'A': 0.37500000000000006, 'C': 0.2500000000000001, 'B': 0.37500000000000006})
```

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

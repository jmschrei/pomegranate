### Bayesian Networks

[Bayesian networks](http://en.wikipedia.org/wiki/Bayesian_network) are a powerful inference tool, in which nodes represent some random variable we care about, edges represent dependencies and a lack of an edge between two nodes represents a conditional independence. A powerful algorithm called the sum-product or forward-backward algorithm allows for inference to be done on this network, calculating posteriors on unobserved ("hidden") variables when limited information is given. The more information is known, the better the inference will be, but there is no requirement on the number of nodes which must be observed. If no information is given, the marginal of the graph is trivially calculated. The hidden and observed variables do not need to be explicitly defined when the network is set, they simply exist based on what information is given. 

An IPython notebook tutorial with visualizations can be [found here](https://github.com/jmschrei/pomegranate/blob/master/tutorials/Tutorial_4_Bayesian_Networks.ipynb)

Lets test out the Bayesian Network framework on the [Monty Hall problem](http://en.wikipedia.org/wiki/Monty_Hall_problem). The Monty Hall problem arose from the gameshow <i>Let's Make a Deal</i>, where a guest had to choose which one of three doors had a prize behind it. The twist was that after the guest chose, the host, originally Monty Hall, would then open one of the doors the guest did not pick and ask if the guest wanted to switch which door they had picked. Initial inspection may lead you to believe that if there are only two doors left, there is a 50-50 chance of you picking the right one, and so there is no advantage one way or the other. However, it has been proven both through simulations and analytically that there is in fact a 66% chance of getting the prize if the guest switches their door, regardless of the door they initially went with. 

We can reproduce this result using Bayesian networks with three nodes, one for the guest, one for the prize, and one for the door Monty chooses to open. The door the guest initially chooses and the door the prize is behind are completely random processes across the three doors, but the door which Monty opens is dependent on both the door the guest chooses (it cannot be the door the guest chooses), and the door the prize is behind (it cannot be the door with the prize behind it). 

```python
import math
from pomegranate import *

# The guests initial door selection is completely random
guest = DiscreteDistribution( { 'A': 1./3, 'B': 1./3, 'C': 1./3 } )

# The door the prize is behind is also completely random
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

In order to reproduce the final result, we need to take advantage of the forward-backward/sum-product algorithm. This algorithm allows the network to calculate posterior probabilities for each distribution in the network when as distributions get clamped to certain values. This is done in pomegranate by feeding in a dictionary of state names and their associated values, and running `forward_backward`.  

Lets say that the guest chooses door 'A'. guest becomes an observed variable, while both prize and monty are hidden variables. 

```python
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
guest	DiscreteDistribution({'A': 1.0, 'C': 0.0, 'B': 0.0})
monty	DiscreteDistribution({'A': 0.0, 'C': 0.5, 'B': 0.5})
```

Since we have clamped the guest distribution to 'A', it returns just that value. The prize distribution is unaffected. 
Since guest is clamped to 'A', it is forced to stay that way. Note that the prize distribution is unaffected, but that the monty distribution now puts a 0 probability on him saying A, since he will not open the same door the guest chose. 

In order to reproduce the final result, we need to see what happens when Monty opens a door. Lets clamp the Monty distribution to 'B' to indicate he has opened that door. Now both guest and monty are observed variables, and prize is the hidden variable.

```python
observations = { 'guest' : 'A', 'monty' : 'B' }
beliefs = map( str, network.forward_backward( observations ) )
print "\n".join( "{}\t{}".format( state.name, belief ) for state, belief in zip( network.states, beliefs ) )
```
yields
```
guest	DiscreteDistribution({'A': 1.0, 'C': 0.0, 'B': 0.0})
monty	DiscreteDistribution({'A': 0.0, 'C': 0.0, 'B': 1.0})
prize	DiscreteDistribution({'A': 0.3333333333333333, 'C': 0.6666666666666666, 'B': 0.0})
```
Both guest and monty have been clamped to values. However, we see that probability of prize being 'C' is 66% mimicking the mystery behind the Monty hall problem!

This has predominately leveraged forward propogation of messages. If we want to see backward propogation of messages, lets see what happens if we tuned in late and only saw which door Monty opened. Monty is an observed variable, while both guest and prize are hidden variables.

```python
observations = { 'monty' : 'B' }
beliefs = map( str, network.forward_backward( observations ) )
print "\n".join( "{}\t{}".format( state.name, belief ) for state, belief in zip( network.states, beliefs ) )
```
yields
```
guest	DiscreteDistribution({'A': 0.49999999999999994, 'C': 0.49999999999999994, 'B': 0.0})
monty	DiscreteDistribution({'A': 0.0, 'C': 0.0, 'B': 1.0})
prize	DiscreteDistribution({'A': 0.49999999999999994, 'C': 0.49999999999999994, 'B': 0.0})
```

We know that if Monty opened door 'B', that the prize cannot be behind 'B' and that the guest could not have opened 'B'. The posterior guest and prize probabilities show this. 

Useful stuff.

API Reference
=============

```eval_rst
.. automodule:: pomegranate.BayesianNetwork
	:members:
```

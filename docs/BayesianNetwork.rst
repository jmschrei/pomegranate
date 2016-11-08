.. _bayesiannetwork:

Bayesian Networks
=================

`IPython Notebook Tutorial <https://github.com/jmschrei/pomegranate/blob/master/tutorials/Tutorial_4_Bayesian_Networks.ipynb>`_

`Bayesian networks <http://en.wikipedia.org/wiki/Bayesian_network>`_ are a powerful inference tool, in which nodes represent some random variable we care about, edges represent dependencies and a lack of an edge between two nodes represents a conditional independence. A powerful algorithm called the sum-product or forward-backward algorithm allows for inference to be done on this network, calculating posteriors on unobserved ("hidden") variables when limited information is given. The more information is known, the better the inference will be, but there is no requirement on the number of nodes which must be observed. If no information is given, the marginal of the graph is trivially calculated. The hidden and observed variables do not need to be explicitly defined when the network is set, they simply exist based on what information is given. 

Lets test out the Bayesian Network framework on the `Monty Hall problem <http://en.wikipedia.org/wiki/Monty_Hall_problem>`_. The Monty Hall problem arose from the gameshow *Let's Make a Deal*, where a guest had to choose which one of three doors had a prize behind it. The twist was that after the guest chose, the host, originally Monty Hall, would then open one of the doors the guest did not pick and ask if the guest wanted to switch which door they had picked. Initial inspection may lead you to believe that if there are only two doors left, there is a 50-50 chance of you picking the right one, and so there is no advantage one way or the other. However, it has been proven both through simulations and analytically that there is in fact a 66% chance of getting the prize if the guest switches their door, regardless of the door they initially went with. 

We can reproduce this result using Bayesian networks with three nodes, one for the guest, one for the prize, and one for the door Monty chooses to open. The door the guest initially chooses and the door the prize is behind are completely random processes across the three doors, but the door which Monty opens is dependent on both the door the guest chooses (it cannot be the door the guest chooses), and the door the prize is behind (it cannot be the door with the prize behind it). 

.. code-block:: python

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

	s1 = State( guest, name="guest" )
	s2 = State( prize, name="prize" )
	s3 = State( monty, name="monty" )

	network = BayesianNetwork( "Monty Hall Problem" )
	network.add_states(s1, s2, s3)
	network.add_edge(s1, s3)
	network.add_edge(s2, s3)
	network.bake()


Bayesian Networks utilize ConditionalProbabilityTable objects to represent conditional distributions. This distribution is made up of a table where each column represents the parent (or self) values except for the last column which represents the probability of the variable taking on that value given its parent values. It also takes in a list of parent distribution objects in the same order that they are used in the table. In the Monty Hall example, the monty distribution is dependent on both the guest and the prize distributions in that order and so the first column of the CPT is the value the guest takes and the second column is the value that the prize takes.

The next step is to make predictions using this model. One of the strengths of Bayesian networks is their ability to infer the values of arbitrary 'hidden variables' given the values from 'observed variables.' These hidden and observed variables do not need to be specified beforehand, and the more variables which are observed the better the inference will be on the hidden variables.

Lets say that the guest chooses door 'A'. guest becomes an observed variable, while both prize and monty are hidden variables. 

... code-block:: python
	
	>>> beliefs = network.predict_proba({ 'guest' : 'A' })
	>>> beliefs = map(str, beliefs)
	>>> print "\n".join( "{}\t{}".format( state.name, belief ) for state, belief in zip( network.states, beliefs ) )
	prize	DiscreteDistribution({'A': 0.3333333333333335, 'C': 0.3333333333333333, 'B': 0.3333333333333333})
	guest	DiscreteDistribution({'A': 1.0, 'C': 0.0, 'B': 0.0})
	monty	DiscreteDistribution({'A': 0.0, 'C': 0.5, 'B': 0.5})

Since we've observed the value that guest takes, we know there is a 100% chance it is that value. The prize distribution is unaffected because it is independent of the guest variable given that we don't know the door that Monty opens.

Now the next step is for Monty to open a door. Let's say that Monty opens door 'b':

.. code-block:: python

	>>> beliefs = network.predict_proba({'guest' : 'A', 'monty' : 'B'})
	>>> print "\n".join( "{}\t{}".format( state.name, str(belief) ) for state, belief in zip( network.states, beliefs ) )
	guest	DiscreteDistribution({'A': 1.0, 'C': 0.0, 'B': 0.0})
	monty	DiscreteDistribution({'A': 0.0, 'C': 0.0, 'B': 1.0})
	prize	DiscreteDistribution({'A': 0.3333333333333333, 'C': 0.6666666666666666, 'B': 0.0})

We've observed both guest and Monty so there is a 100% chance for those values. However, we see that probability of prize being 'C' is 66% mimicking the mystery behind the Monty hall problem!

API Reference
-------------

.. automodule:: pomegranate.BayesianNetwork
	:members:
	:inherited-members:

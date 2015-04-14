# test_table.py

from pomegranate import *
import math

# Define some tables yo
c_table = [[0, 0, 0, 0.6],
		   [0, 0, 1, 0.4],
		   [0, 1, 0, 0.7],
		   [0, 1, 1, 0.3],
		   [1, 0, 0, 0.2],
		   [1, 0, 1, 0.8],
		   [1, 1, 0, 0.9],
		   [1, 1, 1, 0.1]]

d_table = [[ 0, 0, 0.5 ],
		   [ 0, 1, 0.5 ],
		   [ 1, 0, 0.3 ],
		   [ 1, 1, 0.7 ]]

f_table = [[ 0, 0, 0, 0.8 ],
		   [ 0, 0, 1, 0.2 ],
		   [ 0, 1, 0, 0.3 ],
		   [ 0, 1, 1, 0.7 ],
		   [ 1, 0, 0, 0.6 ],
		   [ 1, 0, 1, 0.4 ],
		   [ 1, 1, 0, 0.9 ],
		   [ 1, 1, 1, 0.1 ]]

e_table = [[ 0, 0, 0.7 ],
		   [ 0, 1, 0.3 ],
		   [ 1, 0, 0.2 ],
		   [ 1, 1, 0.8 ]]

g_table = [[ 0, 0, 0, 0.34 ],
		   [ 0, 0, 1, 0.66 ],
		   [ 0, 1, 0, 0.83 ],
		   [ 0, 1, 1, 0.17 ],
		   [ 1, 0, 0, 0.77 ],
		   [ 1, 0, 1, 0.23 ],
		   [ 1, 1, 0, 0.12 ],
		   [ 1, 1, 1, 0.88 ]]

# Turn them into distribution objects
a = DiscreteDistribution({ 0: 0.5, 1: 0.5 })
b = DiscreteDistribution({ 0: 0.7, 1: 0.3 })
e = ConditionalProbabilityTable( e_table, [b] )
c = ConditionalProbabilityTable( c_table, [a,b] )
d = ConditionalProbabilityTable( d_table, [c] )
f = ConditionalProbabilityTable( f_table, [c,e] )
g = ConditionalProbabilityTable( g_table, [c,e] )

# Turn these into states
a_s = State( a, "a" )
b_s = State( b, "b" )
c_s = State( c, "c" )
d_s = State( d, "d" )
e_s = State( e, "e" )
f_s = State( f, "f" )
g_s = State( g, "g" )

# Build the model and add the states
model = BayesianNetwork( "derp" )
model.add_nodes( [a_s, b_s, c_s, d_s, e_s, f_s, g_s] )

# Add the edges in the graph
model.add_edge( a_s, c_s )
model.add_edge( b_s, c_s )
model.add_edge( c_s, d_s )
model.add_edge( c_s, f_s )
model.add_edge( b_s, e_s )
model.add_edge( e_s, f_s )
model.add_edge( c_s, g_s )
model.add_edge( e_s, g_s )

# Finalize the structure
model.bake()

# print "\n".join( "{:10.10} : {}".format( state.name, belief.parameters[0] ) for state, belief in zip( model.states, model.forward_backward( max_iterations=100 ) ) )

# new model
c = DiscreteDistribution({ 0: 0.9, 1: 0.1 })
l = DiscreteDistribution({ 0: 0.3, 1: 0.7 })

rn_table = [[ 0, 0, 0.95 ],
			[ 0, 1, 0.05 ],
			[ 1, 0, 0.50 ],
			[ 1, 1, 0.50 ] ]

s_table = [[ 0, 0, 0, 0.99 ],
		   [ 0, 0, 1, 0.01 ],
		   [ 0, 1, 0, 0.75 ],
		   [ 0, 1, 1, 0.25 ],
		   [ 1, 0, 0, 0.50 ],
		   [ 1, 0, 1, 0.50 ],
		   [ 1, 1, 0, 0.40 ],
		   [ 1, 1, 1, 0.60 ]]

tm_table = [[ 0, 0, 0, 0.99 ],
			[ 0, 0, 1, 0.01 ],
			[ 0, 1, 0, 0.90 ],
			[ 0, 1, 1, 0.10 ],
			[ 1, 0, 0, 0.80 ],
			[ 1, 0, 1, 0.20 ],
			[ 1, 1, 0, 0.50 ],
			[ 1, 1, 1, 0.50 ]]

rn = ConditionalProbabilityTable( rn_table, [c] )
s = ConditionalProbabilityTable( s_table, [c, l] )
tm = ConditionalProbabilityTable( tm_table, [rn, s] )

c_s = State( c, 'c' )
l_s = State( l, 'l' )
rn_s = State( rn, 'rn' )
s_s = State( s, 's' )
tm_s = State( tm, 'tm' )

model = BayesianNetwork( 'sneeze' )
model.add_nodes([ c_s, l_s, rn_s, s_s, tm_s ])

model.add_edge( c_s, rn_s )
model.add_edge( c_s, s_s )
model.add_edge( l_s, s_s )
model.add_edge( rn_s, tm_s )
model.add_edge( s_s, tm_s )

model.bake()
print "\n".join( "{:10.10} : {}".format( state.name, belief.parameters[0] ) for state, belief in zip( model.states, model.forward_backward( max_iterations=10 ) ) )
from __future__ import  (division, print_function)

from pomegranate import *
from nose.tools import with_setup
import random
import numpy as np

def setup():
	'''
	No setup or teardown needs to be done in this case.
	'''

	pass

def teardown():
	'''
	No setup or teardown needs to be done in this case.
	'''

	pass

def discrete_equality( x, y, z=8 ):
	'''
	Test to see if two discrete distributions are equal to each other to
	z decimal points.
	'''

	xd, yd = x.parameters[0], y.parameters[0]
	for key, value in xd.items():
		if round( yd[key], z ) != round( value, z ):
			return False
	return True

@with_setup( setup, teardown )
def test_normal():
	'''
	Test that the normal distribution implementation is correct.
	'''

	d = NormalDistribution( 5, 2 )
	e = NormalDistribution( 5., 2. )

	assert d.log_probability( 5 ) == -1.6120857137642188
	assert d.log_probability( 5 ) == e.log_probability( 5 )
	assert d.log_probability( 5 ) == d.log_probability( 5. )

	assert d.log_probability( 0 ) == -4.737085713764219
	assert d.log_probability( 0 ) == e.log_probability( 0. )

	d.from_sample( [ 5, 4, 5, 4, 6, 5, 6, 5, 4, 6, 5, 4 ] )

	assert round( d.parameters[0], 4 ) == 4.9167
	assert round( d.parameters[1], 4 ) == 0.7592 
	assert d.log_probability( 4 ) != e.log_probability( 4 )
	assert d.log_probability( 4 ) == -1.3723678499651766
	assert d.log_probability( 18 ) == -149.13140399454429
	assert d.log_probability( 1e8 ) == -8674697942168743.0

	d = NormalDistribution( 5, 1e-10 )
	assert d.log_probability( 1e100 ) == -4.9999999999999994e+219

	d.from_sample( [ 0, 2, 3, 2, 100 ], weights=[ 0, 5, 2, 3, 200 ] )
	assert round( d.parameters[0], 4 ) == 95.3429
	assert round( d.parameters[1], 4 ) == 20.8276
	assert round( d.log_probability( 50 ), 8 ) == -6.32501194

	d = NormalDistribution( 5, 2 )
	d.from_sample( [ 0, 5, 3, 5, 7, 3, 4, 5, 2 ], inertia=0.5 )

	assert round( d.parameters[0], 4 ) == 4.3889
	assert round( d.parameters[1], 4 ) == 1.9655 

	d.summarize( [ 0, 2 ], weights=[0, 5] )
	d.summarize( [ 3, 2 ], weights=[2, 3] )
	d.summarize( [ 100 ], weights=[200] )
	d.from_summaries()

	assert round( d.parameters[0], 4 ) == 95.3429
	assert round( d.parameters[1], 4 ) == 20.8276

	d.freeze()
	d.from_sample( [ 0, 1, 1, 2, 3, 2, 1, 2, 2 ] )
	assert round( d.parameters[0], 4 ) == 95.3429
	assert round( d.parameters[1], 4 ) == 20.8276

	d.thaw()
	d.from_sample( [ 5, 4, 5, 4, 6, 5, 6, 5, 4, 6, 5, 4 ] )
	assert round( d.parameters[0], 4 ) == 4.9167 
	assert round( d.parameters[1], 4 ) == 0.7592

@with_setup( setup, teardown )
def test_uniform():
	'''
	Test that the uniform distribution implementation is correct.
	'''

	d = UniformDistribution( 0, 10 )

	assert d.log_probability( 2.34 ) == -2.3025850929940455
	assert d.log_probability( 2 ) == d.log_probability( 8 )
	assert d.log_probability( 10 ) == d.log_probability( 3.4 )
	assert d.log_probability( 1.7 ) == d.log_probability( 9.7 )
	assert d.log_probability( 10.0001 ) == float( "-inf" )
	assert d.log_probability( -0.0001 ) == float( "-inf" )

	for i in xrange( 10 ):
		data = np.random.randn( 100 ) * 100
		d.from_sample( data )
		assert d.parameters == [ data.min(), data.max() ]


	for i in xrange( 100 ):
		sample = d.sample()
		assert data.min() <= sample <= data.max()

	d = UniformDistribution( 0, 10 )
	d.from_sample( [ -5, 20 ], inertia=0.5 )

	assert d.parameters[0] == -2.5
	assert d.parameters[1] == 15

	d.from_sample( [ -100, 100 ], inertia=1.0 )

	assert d.parameters[0] == -2.5
	assert d.parameters[1] == 15

	d.summarize( [ 0, 50, 2, 24, 28 ] )
	d.summarize( [ -20, 7, 8, 4 ] )
	d.from_summaries( inertia=0.75 )

	assert d.parameters[0] == -6.875
	assert d.parameters[1] == 23.75

	d.summarize( [ 0, 100 ] )
	d.summarize( [ 100, 200 ] )
	d.from_summaries()

	assert d.parameters[0] == 0
	assert d.parameters[1] == 200

	d.freeze()
	d.from_sample( [ 0, 1, 6, 7, 8, 3, 4, 5, 2 ] )
	assert d.parameters == [ 0, 200 ]

	d.thaw()
	d.from_sample( [ 0, 1, 6, 7, 8, 3, 4, 5, 2 ] )
	assert d.parameters == [ 0, 8 ]

@with_setup( setup, teardown )
def test_discrete():
	'''
	Test that the discrete distribution implementation is correct.
	'''

	d = DiscreteDistribution( { 'A': 0.25, 'C': 0.25, 'G': 0.25, 'T': 0.25 } )

	assert d.log_probability( 'C' ) == -1.3862943611198906
	assert d.log_probability( 'A' ) == d.log_probability( 'C' )
	assert d.log_probability( 'G' ) == d.log_probability( 'T' )
	assert d.log_probability( 'a' ) == float( '-inf' )

	seq = "ACGTACGTTGCATGCACGCGCTCTCGCGC"
	d.from_sample( list( seq ) )

	assert d.log_probability( 'C' ) == -0.9694005571881036
	assert d.log_probability( 'A' ) == -1.9810014688665833
	assert d.log_probability( 'T' ) == -1.575536360758419

	seq = "ACGTGTG"
	d.from_sample( list( seq ), weights=[0.,1.,2.,3.,4.,5.,6.] )

	assert d.log_probability( 'A' ) == float( '-inf' )
	assert d.log_probability( 'C' ) == -3.044522437723423
	assert d.log_probability( 'G' ) == -0.5596157879354228

	d.summarize( list("ACG"), weights=[0., 1., 2.] )
	d.summarize( list("TGT"), weights=[3., 4., 5.] )
	d.summarize( list("G"), weights=[6.] )
	d.from_summaries()

	assert d.log_probability( 'A' ) == float( '-inf' )
	assert round( d.log_probability( 'C' ), 4 ) == -3.0445
	assert round( d.log_probability( 'G' ), 4 ) == -0.5596

	d = DiscreteDistribution( { 'A': 0.0, 'B': 1.0 } )
	d.summarize( list( "ABABABAB" ) )
	d.summarize( list( "ABAB" ) )
	d.summarize( list( "BABABABABABABABABA" ) )
	d.from_summaries( inertia=0.75 )
	assert d.parameters[0] == { 'A': 0.125, 'B': 0.875 }
 
	d = DiscreteDistribution( { 'A': 0.0, 'B': 1.0 } )
	d.summarize( list( "ABABABAB" ) )
	d.summarize( list( "ABAB" ) )
	d.summarize( list( "BABABABABABABABABA" ) )
	d.from_summaries( inertia=0.5 )
	assert d.parameters[0] == { 'A': 0.25, 'B': 0.75 }

	d.freeze()
	d.from_sample( list('ABAABBAAAAAAAAAAAAAAAAAA') )
	assert d.parameters[0] == { 'A': 0.25, 'B': 0.75 }

@with_setup( setup, teardown )
def test_lognormal():
	'''
	Test that the lognormal distribution implementation is correct.
	'''

	d = LogNormalDistribution( 5, 2 )
	assert round( d.log_probability( 5 ), 4 ) == -4.6585

	d.from_sample( [ 5.1, 5.03, 4.98, 5.05, 4.91, 5.2, 5.1, 5., 4.8, 5.21 ])
	assert round( d.parameters[0], 4 ) == 1.6167
	assert round( d.parameters[1], 4 ) == 0.0237

	d.summarize( [5.1, 5.03, 4.98, 5.05] )
	d.summarize( [4.91, 5.2, 5.1] )
	d.summarize( [5., 4.8, 5.21] )
	d.from_summaries()

	assert round( d.parameters[0], 4 ) == 1.6167
	assert round( d.parameters[1], 4 ) == 0.0237

@with_setup( setup, teardown )
def test_gamma():
	'''
	Test that the gamma distribution implementation is correct.
	'''

	d = GammaDistribution( 5, 2 )
	assert round( d.log_probability( 4 ), 4 ) == -2.1671

	d.from_sample( [ 2.3, 4.3, 2.7, 2.3, 3.1, 3.2, 3.4, 3.1, 2.9, 2.8 ] )
	assert round( d.parameters[0], 4 ) == 31.8806
	assert round( d.parameters[1], 4 ) == 10.5916

	d = GammaDistribution( 2, 7 )
	assert round( d.log_probability( 4 ), 4 ) != -2.1671

	d.summarize( [2.3, 4.3, 2.7] )
	d.summarize( [2.3, 3.1, 3.2] )
	d.summarize( [3.4, 3.1, 2.9, 2.8] )
	d.from_summaries()

	assert round( d.parameters[0], 4 ) == 31.8806
	assert round( d.parameters[1], 4 ) == 10.5916

@with_setup( setup, teardown )
def test_exponential():
	'''
	Test that the beta distribution implementation is correct.
	'''

	d = ExponentialDistribution( 3 )
	assert round( d.log_probability( 8 ), 4 ) == -22.9014

	d.from_sample( [ 2.7, 2.9, 3.8, 1.9, 2.7, 1.6, 1.3, 1.0, 1.9 ] )
	assert round( d.parameters[0], 4 ) == 0.4545

	d = ExponentialDistribution( 4 )
	assert round( d.log_probability( 8 ), 4 ) != -22.9014

	d.summarize( [2.7, 2.9, 3.8] )
	d.summarize( [1.9, 2.7, 1.6] )
	d.summarize( [1.3, 1.0, 1.9] )
	d.from_summaries()

	print (d.parameters)
	assert round( d.parameters[0], 4 ) == 0.4545

@with_setup( setup, teardown )
def test_inverse_gamma():
	'''
	Test that the inverse gamma distribution implementation is correct.
	'''

	d = InverseGammaDistribution( 4, 5 )
	assert round( d.log_probability( 1.06 ), 4 ) == -0.2458

	d.from_sample( [ 0.1, 0.2, 0.015, 0.1, 0.09, 0.08, 0.07, 0.075, 0.044 ] )
	assert round( d.parameters[0], 4 ) == 1.9565
	assert round( d.parameters[1], 4 ) == 0.1063

	d = InverseGammaDistribution( 5, 4 )
	assert round( d.log_probability( 1.06 ), 4 ) != -0.2458 

	d.summarize( [0.1, 0.2, 0.015] )
	d.summarize( [0.1, 0.09, 0.08] )
	d.summarize( [0.07, 0.075] )
	d.summarize( [0.044] )
	d.from_summaries()

	assert round( d.parameters[0], 4 ) == 1.9565
	assert round( d.parameters[1], 4 ) == 0.1063	

@with_setup( setup, teardown )
def test_gaussian_kernel():
	'''
	Test that the Gaussian Kernel Density implementation is correct.
	'''

	d = GaussianKernelDensity( [ 0, 4, 3, 5, 7, 4, 2 ] )
	assert round( d.log_probability( 3.3 ), 4 ) == -1.7042

	d.from_sample( [ 1, 6, 8, 3, 2, 4, 7, 2] )
	assert round( d.log_probability( 1.2 ), 4 ) == -2.0237

	d.from_sample( [ 1, 0, 108 ], weights=[2., 3., 278.] )
	assert round( d.log_probability( 110 ), 4 ) == -2.9368
	assert round( d.log_probability( 0 ), 4 ) == -5.1262

	d.summarize( [1, 6, 8, 3] )
	d.summarize( [2, 4, 7] )
	d.summarize( [2] )
	d.from_summaries()
	assert round( d.log_probability( 1.2 ), 4 ) == -2.0237

	d.summarize( [ 1, 0, 108 ], weights=[2., 3., 278.] )
	d.from_summaries()
	assert round( d.log_probability( 110 ), 4 ) == -2.9368
	assert round( d.log_probability( 0 ), 4 ) == -5.1262

	d.freeze()
	d.from_sample( [ 1, 3, 5, 4, 6, 7, 3, 4, 2 ] )
	assert round( d.log_probability( 110 ), 4 ) == -2.9368
	assert round( d.log_probability( 0 ), 4 ) == -5.1262

@with_setup( setup, teardown )
def test_triangular_kernel():
	'''
	Test that the Triangular Kernel Density implementation is correct.
	'''

	d = TriangleKernelDensity( [ 1, 6, 3, 4, 5, 2 ] )
	assert round( d.log_probability( 6.5 ), 4 ) == -2.4849

	d = TriangleKernelDensity( [1, 8, 100] )
	assert round( d.log_probability( 6.5 ), 4 ) != -2.4849

	d.summarize( [1, 6] )
	d.summarize( [3, 4, 5] )
	d.summarize( [2] )
	d.from_summaries()
	assert round( d.log_probability( 6.5 ), 4 ) == -2.4849

	d.freeze()
	d.from_sample( [ 1, 4, 6, 7, 3, 5, 7, 8, 3, 3, 4 ] )
	assert round( d.log_probability( 6.5 ), 4 ) == -2.4849



@with_setup( setup, teardown )
def test_uniform_kernel():
	'''
	Test that the Uniform Kernel Density implementation is correct.
	'''

	d = UniformKernelDensity( [ 1, 3, 5, 6, 2, 2, 3, 2, 2 ] )

	assert round( d.log_probability( 2.2 ), 4 ) == -0.4055
	assert round( d.log_probability( 6.2 ), 4 ) == -2.1972
	assert d.log_probability( 10 ) == float( '-inf' )

	d = UniformKernelDensity( [ 1, 100, 200 ] )
	assert round( d.log_probability( 2.2 ), 4 ) != -0.4055
	assert round( d.log_probability( 6.2 ), 4 ) != -2.1972

	d.summarize( [1, 3, 5, 6, 2] )
	d.summarize( [2, 3, 2, 2] )
	d.from_summaries()
	assert round( d.log_probability( 2.2 ), 4 ) == -0.4055
	assert round( d.log_probability( 6.2 ), 4 ) == -2.1972

@with_setup( setup, teardown )
def test_mixture():
	'''
	Test that the Mixture Distribution implementation is correct.
	'''

	d = MixtureDistribution( [ NormalDistribution( 5, 1 ), 
							   NormalDistribution( 4, 4 ) ] )

	assert round( d.log_probability( 6 ), 4 ) == -1.8018
	assert round( d.log_probability( 5 ), 4 ) == -1.3951
	assert round( d.log_probability( 4.5 ), 4 ) == -1.4894

	d = MixtureDistribution( [ NormalDistribution( 5, 1 ),
	                           NormalDistribution( 4, 4 ) ],
	                         weights=[1., 7.] )

	assert round( d.log_probability( 6 ), 4 ) == -2.2325
	assert round( d.log_probability( 5 ), 4 ) == -2.0066
	assert round( d.log_probability( 4.5 ), 4 ) == -2.0356

@with_setup( setup, teardown )
def test_independent():
	'''
	Test that the Multivariate Distribution implementation is correct.
	'''

	d = IndependentComponentsDistribution( [ NormalDistribution( 5, 2 ), ExponentialDistribution( 2 ) ] )

	assert round( d.log_probability( (4,1) ), 4 ) == -3.0439
	assert round( d.log_probability( ( 100, 0.001 ) ), 4 ) == -1129.0459

	d = IndependentComponentsDistribution( [ NormalDistribution( 5, 2 ), ExponentialDistribution( 2 ) ],
								  weights=[18., 1.] )

	assert round( d.log_probability( (4,1) ), 4 ) == -32.5744
	assert round( d.log_probability( (100, 0.001) ), 4 ) == -20334.5764

	d.from_sample( [ (5, 1), (5.2, 1.7), (4.7, 1.9), (4.9, 2.4), (4.5, 1.2) ] )

	assert round( d.parameters[0][0].parameters[0], 4 ) == 4.86
	assert round( d.parameters[0][0].parameters[1], 4 ) == 0.2417
	assert round( d.parameters[0][1].parameters[0], 4 ) == 0.6098

	d = IndependentComponentsDistribution( [ NormalDistribution( 5, 2 ),
									UniformDistribution( 0, 10 ) ] )
	d.from_sample( [ ( 0, 0 ), ( 5, 0 ), ( 3, 0 ), ( 5, -5 ), ( 7, 0 ),
				     ( 3, 0 ), ( 4, 0 ), ( 5, 0 ), ( 2, 20) ], inertia=0.5 )

	assert round( d.parameters[0][0].parameters[0], 4 ) == 4.3889
	assert round( d.parameters[0][0].parameters[1], 4 ) == 1.9655 

	assert d.parameters[0][1].parameters[0] == -2.5
	assert d.parameters[0][1].parameters[1] == 15

	d.from_sample( [ ( 0, 0 ), ( 5, 0 ), ( 3, 0 ), ( 5, -5 ), ( 7, 0 ),
				     ( 3, 0 ), ( 4, 0 ), ( 5, 0 ), ( 2, 20 ) ], inertia=0.75 )

	assert round( d.parameters[0][0].parameters[0], 4 ) != 4.3889
	assert round( d.parameters[0][0].parameters[1], 4 ) != 1.9655 

	assert d.parameters[0][1].parameters[0] != -2.5
	assert d.parameters[0][1].parameters[1] != 15

	d = IndependentComponentsDistribution([ NormalDistribution( 5, 2 ),
								   UniformDistribution( 0, 10 ) ])

	d.summarize([ ( 0, 0 ), ( 5, 0 ), ( 3, 0 ) ])
	d.summarize([ ( 5, -5 ), ( 7, 0 ) ])
	d.summarize([ ( 3, 0 ), ( 4, 0 ), ( 5, 0 ), ( 2, 20 ) ])
	d.from_summaries( inertia=0.5 )

	assert round( d.parameters[0][0].parameters[0], 4 ) == 4.3889
	assert round( d.parameters[0][0].parameters[1], 4 ) == 1.9655 

	assert d.parameters[0][1].parameters[0] == -2.5
	assert d.parameters[0][1].parameters[1] == 15

	d.freeze()
	d.from_sample( [ ( 1, 7 ), ( 7, 2 ), ( 2, 4), ( 2, 4 ), ( 1, 4 ) ] )

	assert round( d.parameters[0][0].parameters[0], 4 ) == 4.3889
	assert round( d.parameters[0][0].parameters[1], 4 ) == 1.9655 

	assert d.parameters[0][1].parameters[0] == -2.5
	assert d.parameters[0][1].parameters[1] == 15

def test_conditional():
	'''
	Test a conditional discrete distribution.
	'''

	phditis = DiscreteDistribution({ True : 0.01, False : 0.99 })
	test_result = ConditionalProbabilityTable(
		[[True,  True,  0.95 ],
		 [True,  False, 0.05 ],
		 [False, True,  0.05 ],
		 [False, False, 0.95 ]], [phditis])

	assert discrete_equality( test_result.marginal(),
		DiscreteDistribution({False : 0.941, True : 0.059}) )

def test_monty():
	'''
	Test based on the Monty Hall simulation.
	'''

	guest = DiscreteDistribution( { 'A': 1./3, 'B': 1./3, 'C': 1./3 } )

	# The actual prize is independent of the other distributions
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

	assert monty.log_probability( ('A', 'B', 'C') ) == 0.
	assert monty.log_probability( ('C', 'B', 'A') ) == 0.
	assert monty.log_probability( ('C', 'C', 'C') ) == float("-inf")
	assert monty.log_probability( ('A', 'A', 'A') ) == float("-inf")
	assert monty.log_probability( ('B', 'A', 'C') ) == 0.
	assert monty.log_probability( ('C', 'A', 'B') ) == 0.

	data = [[ 'A', 'A', 'C' ],
			[ 'A', 'A', 'C' ],
			[ 'A', 'A', 'B' ],
			[ 'A', 'A', 'A' ],
			[ 'A', 'A', 'C' ],
			[ 'B', 'B', 'B' ],
			[ 'B', 'B', 'C' ],
			[ 'C', 'C', 'A' ],
			[ 'C', 'C', 'C' ],
			[ 'C', 'C', 'C' ],
			[ 'C', 'C', 'C' ],
			[ 'C', 'B', 'A' ]]

	monty.from_sample( data, weights=[1, 1, 3, 3, 1, 1, 3, 7, 1, 1, 1, 1] )

	assert monty.log_probability( ('A', 'A', 'A') ) == monty.log_probability( ('A', 'A', 'C') )
	assert monty.log_probability( ('A', 'A', 'A') ) == monty.log_probability( ('A', 'A', 'B') )
	assert monty.log_probability( ('B', 'A', 'A') ) == monty.log_probability( ('B', 'A', 'C') )
	assert monty.log_probability( ('B', 'B', 'A') ) == float("-inf")
	assert monty.log_probability( ('C', 'C', 'B') ) == float("-inf")
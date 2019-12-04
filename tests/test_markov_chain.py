from __future__ import (division)

from pomegranate import *
from nose.tools import with_setup
from nose.tools import assert_almost_equal
from nose.tools import assert_equal
from nose.tools import assert_not_equal
from nose.tools import assert_less_equal
from nose.tools import assert_raises
import random
import pickle
import numpy as np

def setup():
	global data
	global weights
	global zeroth_dist
	global first_dist
	global second_dist

	data = [ list('AAACBDAA'), list('DBBCBABD'), list('CBDCCDCBC'),	 list('DDDDC'),
			 list('DADBBCBBBACBC'), list('CBDBAC'), list('CACDDDBCAABA'), list('BBABDAADCABCCAD'),
			 list('DCBBBADBBCBC'), list('DCADDAAA'), list('AABBBCC'), list('CABBADACDBBDCC'),
			 list('AADDCDDDB'), list('CCDBDCDDDABDAC'), list('DDADB'), list('BCDACDBBBBAC'),
			 list('ADCBCCCB'), list('BDDBCA'), list('ACCDCBADCBBBB'), list('ACCBABCBA'),
			 list('DAABCCBBD'), list('CCAADACBDCDBABC'), list('BCDCACBBAD'), list('CBBBCDCA'),
			 list('CBDADACAADBCC'), list('DCDABCC'), list('AAAABCBC'), list('BDDDDACBBCABA'),
			 list('CCBDCBDCBADBBDB'), list('CBDAA'), list('CCDAAAB'), list('ABDBAC'),
			 list('DBCCDCC'), list('CCDCBB'), list('ACDBCDD'), list('CACCACCDBBDABB'),
			 list('DCDDAB'), list('ADDDCDACACA'), list('AAACACADB'), list('BDBBDDBACDCCAAA') ]

	weights = [ 8, 7, 1, 2, 8, 1, 9, 2, 6, 2, 5, 6, 10, 10, 8, 4, 10, 10, 2, 9,
				6, 2, 9, 9, 3, 1, 10, 6, 9, 1, 2, 9, 2, 1, 4, 1, 2, 7, 9, 4]

	zeroth_dist = DiscreteDistribution( { 'A': 0.1, 'B': 0.2, 'C': 0.3, 'D': 0.4 } )

	first_dist = ConditionalProbabilityTable(
		[[ 'A', 'A', 0.8 ], [ 'A', 'B', 0.05 ], [ 'A', 'C', 0.05 ], [ 'A', 'D', 0.1 ],
		 [ 'B', 'A', 0.1 ], [ 'B', 'B', 0.2 ], [ 'B', 'C', 0.6 ], [ 'B', 'D', 0.1 ],
		 [ 'C', 'A', 0.15 ], [ 'C', 'B', 0.1 ], [ 'C', 'C', 0.25 ], [ 'C', 'D', 0.5 ],
		 [ 'D', 'A', 0.25 ], [ 'D', 'B', 0.25 ], [ 'D', 'C', 0.4 ], [ 'D', 'D', 0.1 ]],
		 [ zeroth_dist ] )

	second_dist = ConditionalProbabilityTable(
		[[ 'A', 'A', 'A', 0.05 ], [ 'A', 'A', 'B', 0.25 ], [ 'A', 'A', 'C', 0.15 ], [ 'A', 'A', 'D', 0.55 ],
		 [ 'A', 'B', 'A', 0.05 ], [ 'A', 'B', 'B', 0.05 ], [ 'A', 'B', 'C', 0.85 ], [ 'A', 'B', 'D', 0.05 ],
		 [ 'A', 'C', 'A', 0.7 ], [ 'A', 'C', 'B', 0.1 ], [ 'A', 'C', 'C', 0.1 ], [ 'A', 'C', 'D', 0.1 ],
		 [ 'A', 'D', 'A', 0.2 ], [ 'A', 'D', 'B', 0.4 ], [ 'A', 'D', 'C', 0.35 ], [ 'A', 'D', 'D', 0.05 ],
		 [ 'B', 'A', 'A', 0.3 ], [ 'B', 'A', 'B', 0.05 ], [ 'B', 'A', 'C', 0.15 ], [ 'B', 'A', 'D', 0.5 ],
		 [ 'B', 'B', 'A', 0.8 ], [ 'B', 'B', 'B', 0.1 ], [ 'B', 'B', 'C', 0.1 ], [ 'B', 'B', 'D', 0.0 ],
		 [ 'B', 'C', 'A', 0.1 ], [ 'B', 'C', 'B', 0.35 ], [ 'B', 'C', 'C', 0.3 ], [ 'B', 'C', 'D', 0.25 ],
		 [ 'B', 'D', 'A', 0.3 ], [ 'B', 'D', 'B', 0.1 ], [ 'B', 'D', 'C', 0.2 ], [ 'B', 'D', 'D', 0.4 ],
		 [ 'C', 'A', 'A', 0.2 ], [ 'C', 'A', 'B', 0.3 ], [ 'C', 'A', 'C', 0.3 ], [ 'C', 'A', 'D', 0.2 ],
		 [ 'C', 'B', 'A', 0.35 ], [ 'C', 'B', 'B', 0.45 ], [ 'C', 'B', 'C', 0.0 ], [ 'C', 'B', 'D', 0.2 ],
		 [ 'C', 'C', 'A', 0.25 ], [ 'C', 'C', 'B', 0.0 ], [ 'C', 'C', 'C', 0.6 ], [ 'C', 'C', 'D', 0.15 ],
		 [ 'C', 'D', 'A', 0.8 ], [ 'C', 'D', 'B', 0.1 ], [ 'C', 'D', 'C', 0.05 ], [ 'C', 'D', 'D', 0.05 ],
		 [ 'D', 'A', 'A', 0.5 ], [ 'D', 'A', 'B', 0.0 ], [ 'D', 'A', 'C', 0.5 ], [ 'D', 'A', 'D', 0.0 ],
		 [ 'D', 'B', 'A', 0.35 ], [ 'D', 'B', 'B', 0.5 ], [ 'D', 'B', 'C', 0.1 ], [ 'D', 'B', 'D', 0.05 ],
		 [ 'D', 'C', 'A', 0.1 ], [ 'D', 'C', 'B', 0.45 ], [ 'D', 'C', 'C', 0.0 ], [ 'D', 'C', 'D', 0.45 ],
		 [ 'D', 'D', 'A', 0.2 ], [ 'D', 'D', 'B', 0.1 ], [ 'D', 'D', 'C', 0.1 ], [ 'D', 'D', 'D', 0.6 ]],
		 [ zeroth_dist, first_dist ] )

def teardown():
	pass

@with_setup( setup, teardown )
def test_zeroth_dist():
	assert_almost_equal( zeroth_dist.log_probability( 'A' ), -2.3025850929940455 )
	assert_almost_equal( zeroth_dist.log_probability( 'B' ), -1.6094379124341003 )
	assert_almost_equal( zeroth_dist.log_probability( 'C' ), -1.2039728043259361 )
	assert_almost_equal( zeroth_dist.log_probability( 'D' ), -0.916290731874155 )
	assert_almost_equal( zeroth_dist.log_probability( 'E' ), -float('inf') )

@with_setup( setup, teardown )
def test_first_dist():
	assert_almost_equal( first_dist.log_probability( ('A', 'A') ), -0.22314355131420971 )
	assert_almost_equal( first_dist.log_probability( ('A', 'B') ), -2.9957322735539909 )

	assert_almost_equal( first_dist.log_probability( ('B', 'B') ), -1.6094379124341003 )
	assert_almost_equal( first_dist.log_probability( ('B', 'C') ), -0.51082562376599072 )

	assert_almost_equal( first_dist.log_probability( ('C', 'C') ), -1.3862943611198906 )
	assert_almost_equal( first_dist.log_probability( ('C', 'D') ), -0.69314718055994529 )

	assert_almost_equal( first_dist.log_probability( ('D', 'D') ), -2.3025850929940455 )
	assert_almost_equal( first_dist.log_probability( ('D', 'A') ), -1.3862943611198906 )

@with_setup( setup, teardown )
def test_second_dist():
	assert_almost_equal( second_dist.log_probability( ('A', 'A', 'C') ), -1.89711998489 )
	assert_almost_equal( second_dist.log_probability( ('A', 'A', 'A') ), -2.99573227355 )
	assert_almost_equal( second_dist.log_probability( ('A', 'B', 'A') ), -2.99573227355 )
	assert_almost_equal( second_dist.log_probability( ('A', 'B', 'C') ), -0.162518929498 )

	assert_almost_equal( second_dist.log_probability( ('A', 'C', 'C') ), -2.30258509299 )
	assert_almost_equal( second_dist.log_probability( ('A', 'C', 'D') ), -2.30258509299 )
	assert_almost_equal( second_dist.log_probability( ('A', 'D', 'B') ), -0.916290731874 )
	assert_almost_equal( second_dist.log_probability( ('A', 'D', 'D') ), -2.99573227355 )

	assert_almost_equal( second_dist.log_probability( ('B', 'A', 'B') ), -2.99573227355 )
	assert_almost_equal( second_dist.log_probability( ('B', 'A', 'D') ), -0.69314718056 )
	assert_almost_equal( second_dist.log_probability( ('B', 'B', 'B') ), -2.30258509299 )
	assert_almost_equal( second_dist.log_probability( ('B', 'B', 'D') ), -float('inf') )

	assert_almost_equal( second_dist.log_probability( ('B', 'C', 'A') ), -2.30258509299 )
	assert_almost_equal( second_dist.log_probability( ('B', 'C', 'B') ), -1.0498221245 )
	assert_almost_equal( second_dist.log_probability( ('B', 'D', 'D') ), -0.916290731874 )
	assert_almost_equal( second_dist.log_probability( ('B', 'D', 'B') ), -2.30258509299 )

	assert_almost_equal( second_dist.log_probability( ('C', 'A', 'A') ), -1.60943791243 )
	assert_almost_equal( second_dist.log_probability( ('C', 'A', 'B') ), -1.20397280433 )
	assert_almost_equal( second_dist.log_probability( ('C', 'B', 'C') ), -float('inf') )
	assert_almost_equal( second_dist.log_probability( ('C', 'B', 'A') ), -1.0498221245 )

	assert_almost_equal( second_dist.log_probability( ('C', 'C', 'D') ), -1.89711998489 )
	assert_almost_equal( second_dist.log_probability( ('C', 'C', 'B') ), -float('inf') )
	assert_almost_equal( second_dist.log_probability( ('C', 'D', 'A') ), -0.223143551314 )
	assert_almost_equal( second_dist.log_probability( ('C', 'D', 'C') ), -2.99573227355 )

	assert_almost_equal( second_dist.log_probability( ('D', 'A', 'D') ), -float('inf') )
	assert_almost_equal( second_dist.log_probability( ('D', 'A', 'A') ), -0.69314718056 )
	assert_almost_equal( second_dist.log_probability( ('D', 'B', 'D') ), -2.99573227355 )
	assert_almost_equal( second_dist.log_probability( ('D', 'B', 'C') ), -2.30258509299 )

	assert_almost_equal( second_dist.log_probability( ('D', 'C', 'A') ), -2.30258509299 )
	assert_almost_equal( second_dist.log_probability( ('D', 'C', 'D') ), -0.798507696218 )
	assert_almost_equal( second_dist.log_probability( ('D', 'D', 'D') ), -0.510825623766 )
	assert_almost_equal( second_dist.log_probability( ('D', 'D', 'A') ), -1.60943791243 )

@with_setup( setup, teardown )
def test_constructors():
	# raises no errors
	first_chain = MarkovChain([ zeroth_dist, first_dist ])
	second_chain = MarkovChain([ zeroth_dist, first_dist, second_dist ])

@with_setup( setup, teardown )
def test_first_log_probability():
	# test going one state back
	first_chain = MarkovChain([ zeroth_dist, first_dist ])

	assert_almost_equal( first_chain.log_probability( list('A') ), -2.3025850929940455 )
	assert_almost_equal( first_chain.log_probability( list('B') ), -1.6094379124341003 )

	assert_almost_equal( first_chain.log_probability( list('AC') ), -5.2983173665480363 )
	assert_almost_equal( first_chain.log_probability( list('AD') ), -4.6051701859880909 )
	assert_almost_equal( first_chain.log_probability( list('BD') ), -3.9120230054281455 )
	assert_almost_equal( first_chain.log_probability( list('BA') ), -3.9120230054281455 )
	assert_almost_equal( first_chain.log_probability( list('CA') ), -3.1010927892118172 )
	assert_almost_equal( first_chain.log_probability( list('CB') ), -3.5065578973199818 )
	assert_almost_equal( first_chain.log_probability( list('DB') ), -2.3025850929940455 )
	assert_almost_equal( first_chain.log_probability( list('DC') ), -1.83258146374831 )

	assert_almost_equal( first_chain.log_probability( list('ABDD') ), -9.9034875525361272 )
	assert_almost_equal( first_chain.log_probability( list('CCCB') ), -6.2791466195597625 )
	assert_almost_equal( first_chain.log_probability( list('CCBD') ), -7.1954373514339167 )
	assert_almost_equal( first_chain.log_probability( list('ACAC') ), -10.191169624987909 )

	assert_almost_equal( first_chain.log_probability( list('ABDBCCDC') ), -12.493754717981954 )
	assert_almost_equal( first_chain.log_probability( list('DACCBDCB') ), -14.508657738524217 )

	assert_almost_equal( first_chain.log_probability( list('BCCCACBDBDBABACD') ), -30.75583043175768 )

	assert_almost_equal( first_chain.log_probability( list('DABBCBDACAAADCBDCDBCBDCACBDABBAA') ), -53.616465272807112 )

	# test going two states back
	second_chain = MarkovChain([ zeroth_dist, first_dist, second_dist ])

	assert_almost_equal( second_chain.log_probability( list('A') ), -2.30258509299 )
	assert_almost_equal( second_chain.log_probability( list('B') ), -1.60943791243 )

	assert_almost_equal( second_chain.log_probability( list('AC') ), -5.29831736655 )
	assert_almost_equal( second_chain.log_probability( list('AD') ), -4.60517018599 )
	assert_almost_equal( second_chain.log_probability( list('BD') ), -3.91202300543 )
	assert_almost_equal( second_chain.log_probability( list('BA') ), -3.91202300543 )
	assert_almost_equal( second_chain.log_probability( list('CA') ), -3.10109278921 )
	assert_almost_equal( second_chain.log_probability( list('CB') ), -3.50655789732 )
	assert_almost_equal( second_chain.log_probability( list('DB') ), -2.30258509299 )
	assert_almost_equal( second_chain.log_probability( list('DC') ), -1.83258146375 )

	assert_almost_equal( second_chain.log_probability( list('ABDD') ), -9.21034037198 )
	assert_almost_equal( second_chain.log_probability( list('CCCD') ), -4.9982127741 )
	assert_almost_equal( second_chain.log_probability( list('CCBD') ), -float('inf') )
	assert_almost_equal( second_chain.log_probability( list('ACAC') ), -6.85896511481 )

	assert_almost_equal( second_chain.log_probability( list('ABDBCCDC') ), -18.9960448889 )
	assert_almost_equal( second_chain.log_probability( list('DACCBDCB') ), -float('inf') )

	assert_almost_equal( second_chain.log_probability( list('BCCCACBDBDBABACD') ), -29.1792463442 )

	assert_almost_equal( second_chain.log_probability( list('DABBCBDACAAADCBDCDBCBDCACBDABBAA') ), -float('inf') )

# if summarize and from summaries work, so does fit

@with_setup( setup, teardown )
def test_summarize_no_weights_no_inertia():
	first_chain = MarkovChain([ zeroth_dist, first_dist ])

	# split in four
	first_chain.summarize( data[:10] )
	first_chain.summarize( data[10:20] )
	first_chain.summarize( data[20:30] )
	first_chain.summarize( data[30:] )
	first_chain.from_summaries()

	# check if probabilities are correct
	assert_almost_equal( first_chain.log_probability( list('A') ), -1.29098418132 )
	assert_almost_equal( first_chain.log_probability( list('B') ), -1.89711998489 )

	assert_almost_equal( first_chain.log_probability( list('AC') ), -2.52493781785 )
	assert_almost_equal( first_chain.log_probability( list('AD') ), -2.82721868973 )
	assert_almost_equal( first_chain.log_probability( list('BD') ), -3.35240721749 )
	assert_almost_equal( first_chain.log_probability( list('BA') ), -3.56371631116 )
	assert_almost_equal( first_chain.log_probability( list('CA') ), -2.66812748722 )
	assert_almost_equal( first_chain.log_probability( list('CB') ), -2.31672960038 )
	assert_almost_equal( first_chain.log_probability( list('DB') ), -2.74959920402 )
	assert_almost_equal( first_chain.log_probability( list('DC') ), -2.70514744144 )

	assert_almost_equal( first_chain.log_probability( list('ABDD') ), -5.69233078086 )
	assert_almost_equal( first_chain.log_probability( list('CCCB') ), -5.2049574644 )
	assert_almost_equal( first_chain.log_probability( list('CCBD') ), -5.216130765 )
	assert_almost_equal( first_chain.log_probability( list('ACAC') ), -5.30308884496 )

	assert_almost_equal( first_chain.log_probability( list('ABDBCCDC') ), -11.1281275339 )
	assert_almost_equal( first_chain.log_probability( list('DACCBDCB') ), -10.6827162728 )

	assert_almost_equal( first_chain.log_probability( list('BCCCACBDBDBABACD') ), -23.2162130846 )

	assert_almost_equal( first_chain.log_probability( list('DABBCBDACAAADCBDCDBCBDCACBDABBAA') ), -43.7174844781 )

	second_chain = MarkovChain([ zeroth_dist, first_dist, second_dist ])

	# split into four
	second_chain.summarize( data[:10] )
	second_chain.summarize( data[10:20] )
	second_chain.summarize( data[20:30] )
	second_chain.summarize( data[30:] )
	second_chain.from_summaries()

@with_setup( setup, teardown )
def test_summarize_no_weights_with_inertia():
	first_chain = MarkovChain([ zeroth_dist, first_dist ])

	first_chain.summarize( data[:10] )
	first_chain.summarize( data[10:20] )
	first_chain.summarize( data[20:30] )
	first_chain.summarize( data[30:] )
	first_chain.from_summaries( inertia=0.4 )

	assert_almost_equal( first_chain.log_probability( list('A') ), -1.58474529984 )
	assert_almost_equal( first_chain.log_probability( list('B') ), -1.77195684193 )

	assert_almost_equal( first_chain.log_probability( list('AC') ), -3.22112518823 )
	assert_almost_equal( first_chain.log_probability( list('AD') ), -3.3619279842 )
	assert_almost_equal( first_chain.log_probability( list('BD') ), -3.48675527002 )
	assert_almost_equal( first_chain.log_probability( list('BA') ), -3.6470979201 )
	assert_almost_equal( first_chain.log_probability( list('CA') ), -2.82601794483 )
	assert_almost_equal( first_chain.log_probability( list('CB') ), -2.66015931757 )
	assert_almost_equal( first_chain.log_probability( list('DB') ), -2.54362030796 )
	assert_almost_equal( first_chain.log_probability( list('DC') ), -2.30916483161 )

	assert_almost_equal( first_chain.log_probability( list('ABDD') ), -6.88185009599 )
	assert_almost_equal( first_chain.log_probability( list('CCCB') ), -5.50132618677 )
	assert_almost_equal( first_chain.log_probability( list('CCBD') ), -5.79554118026 )
	assert_almost_equal( first_chain.log_probability( list('ACAC') ), -6.52834038129 )

	assert_almost_equal( first_chain.log_probability( list('ABDBCCDC') ), -11.1045309105 )
	assert_almost_equal( first_chain.log_probability( list('DACCBDCB') ), -11.519936158 )

	assert_almost_equal( first_chain.log_probability( list('BCCCACBDBDBABACD') ), -24.8604337068 )

	assert_almost_equal( first_chain.log_probability( list('DABBCBDACAAADCBDCDBCBDCACBDABBAA') ), -44.8853484278 )

@with_setup( setup, teardown )
def test_summarize_with_weights_no_inertia():
	first_chain = MarkovChain([ zeroth_dist, first_dist ])

	# split in four
	first_chain.summarize( data[:10], weights=weights[:10] )
	first_chain.summarize( data[10:20], weights=weights[10:20] )
	first_chain.summarize( data[20:30], weights=weights[20:30] )
	first_chain.summarize( data[30:], weights=weights[30:] )
	first_chain.from_summaries()

	assert_almost_equal( first_chain.log_probability( list('A') ), -0.961056745744 )
	assert_almost_equal( first_chain.log_probability( list('B') ), -1.82454929205 )

	assert_almost_equal( first_chain.log_probability( list('AC') ), -2.13478966488 )
	assert_almost_equal( first_chain.log_probability( list('AD') ), -2.47837936927 )
	assert_almost_equal( first_chain.log_probability( list('BD') ), -3.41892100613 )
	assert_almost_equal( first_chain.log_probability( list('BA') ), -3.45702085236 )
	assert_almost_equal( first_chain.log_probability( list('CA') ), -2.84952832968 )
	assert_almost_equal( first_chain.log_probability( list('CB') ), -2.43790935599 )
	assert_almost_equal( first_chain.log_probability( list('DB') ), -2.88910833385 )
	assert_almost_equal( first_chain.log_probability( list('DC') ), -2.98200208074 )

	assert_almost_equal( first_chain.log_probability( list('ABDD') ), -5.56874179664 )
	assert_almost_equal( first_chain.log_probability( list('CCCB') ), -5.71252297888 )
	assert_almost_equal( first_chain.log_probability( list('CCBD') ), -5.66958788152 )
	assert_almost_equal( first_chain.log_probability( list('ACAC') ), -4.78548674538 )

	assert_almost_equal( first_chain.log_probability( list('ABDBCCDC') ), -11.3892431338 )
	assert_almost_equal( first_chain.log_probability( list('DACCBDCB') ), -11.0991865874 )

	assert_almost_equal( first_chain.log_probability( list('BCCCACBDBDBABACD') ), -23.5462387667 )

	assert_almost_equal( first_chain.log_probability( list('DABBCBDACAAADCBDCDBCBDCACBDABBAA') ), -44.3127858762 )

@with_setup( setup, teardown )
def test_summarize_with_weights_with_inertia():
	first_chain = MarkovChain([ zeroth_dist, first_dist ])

	# split in four
	first_chain.summarize( data[:10], weights=weights[:10] )
	first_chain.summarize( data[10:20], weights=weights[10:20] )
	first_chain.summarize( data[20:30], weights=weights[20:30] )
	first_chain.summarize( data[30:], weights=weights[30:] )
	first_chain.from_summaries( inertia=0.4 )

	assert_almost_equal( first_chain.log_probability( list('A') ), -1.3112125381 )
	assert_almost_equal( first_chain.log_probability( list('B') ), -1.73288210353 )

	assert_almost_equal( first_chain.log_probability( list('AC') ), -2.89339373397 )
	assert_almost_equal( first_chain.log_probability( list('AD') ), -3.07392432189 )
	assert_almost_equal( first_chain.log_probability( list('BD') ), -3.55414269165 )
	assert_almost_equal( first_chain.log_probability( list('BA') ), -3.58268887356 )
	assert_almost_equal( first_chain.log_probability( list('CA') ), -2.92624445535 )
	assert_almost_equal( first_chain.log_probability( list('CB') ), -2.70099965753 )
	assert_almost_equal( first_chain.log_probability( list('DB') ), -2.596587547 )
	assert_almost_equal( first_chain.log_probability( list('DC') ), -2.43824119019 )

	assert_almost_equal( first_chain.log_probability( list('ABDD') ), -6.77853581842 )
	assert_almost_equal( first_chain.log_probability( list('CCCB') ), -5.75946483735 )
	assert_almost_equal( first_chain.log_probability( list('CCBD') ), -6.05149283556 )
	assert_almost_equal( first_chain.log_probability( list('ACAC') ), -6.10013721195 )

	assert_almost_equal( first_chain.log_probability( list('ABDBCCDC') ), -11.2181867683 )
	assert_almost_equal( first_chain.log_probability( list('DACCBDCB') ), -11.6681121956 )

	assert_almost_equal( first_chain.log_probability( list('BCCCACBDBDBABACD') ), -25.0365515667 )

	assert_almost_equal( first_chain.log_probability( list('DABBCBDACAAADCBDCDBCBDCACBDABBAA') ), -45.1660985662 )

@with_setup( setup, teardown )
def test_raise_errors():
	pass

@with_setup( setup, teardown )
def test_pickling():
	chain1 = MarkovChain([ zeroth_dist, first_dist ])
	chain2 = pickle.loads( pickle.dumps( chain1 ) )

	assert_almost_equal( chain1.log_probability( list('BCCCACBDBDBABACD') ),
	                     chain2.log_probability( list('BCCCACBDBDBABACD') ) )

@with_setup( setup, teardown )
def test_json():
	chain1 = MarkovChain([ zeroth_dist, first_dist ])
	chain2 = MarkovChain.from_json(chain1.to_json())

	assert_almost_equal( chain1.log_probability( list('BCCCACBDBDBABACD') ),
	                     chain2.log_probability( list('BCCCACBDBDBABACD') ) )

@with_setup( setup, teardown )
def test_robust_from_json():
	chain1 = MarkovChain([ zeroth_dist, first_dist ])
	chain2 = from_json(chain1.to_json())

	assert_almost_equal( chain1.log_probability( list('BCCCACBDBDBABACD') ),
	                     chain2.log_probability( list('BCCCACBDBDBABACD') ) )
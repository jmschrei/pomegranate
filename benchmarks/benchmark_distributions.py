# benchmark_distributions.py
# Contact: Jacob Schreiber ( jmschreiber91@gmail.com )

"""
Benchmark the distribution module, printing out the time it takes to do
log probability and training calculations.
"""

from pomegranate import *
import random
import numpy
import time

numpy.random.seed(0)
random.seed(0)

def print_benchmark( distribution, duration ):
	"""Formatted print."""

	print( "{:25}: {:.4}s".format( distribution.__class__.__name__, duration ) )

def bench_log_probability( distribution, n=10000000, symbol=5 ):
	"""Bench a log probability distribution."""

	tic = time.time()
	for i in range(n):
		logp = distribution.log_probability( symbol )
	return time.time() - tic

def bench_from_sample( distribution, sample, n=1000 ):
	"""Bench the training of a probability distribution."""

	tic = time.time()
	for i in range(n):
		distribution.summarize( sample )
	return time.time() - tic

def benchmark_distribution_log_probabilities():
	"""Run log probability benchmarks."""

	distributions = [ UniformDistribution( 0, 17 ),
	                  NormalDistribution( 7, 1 ),
	                  LogNormalDistribution( 7, 1 ),
	                  ExponentialDistribution( 7 ),
	                  GammaDistribution( 7, 3 ),
	                  GaussianKernelDensity([0, 1, 4, 3, 2, 0.5, 2, 1, 2]),
	                  UniformKernelDensity([0, 1, 4, 3, 2, 0.5, 2, 1, 2]),
	                  TriangleKernelDensity([0, 1, 4, 3, 2, 0.5, 2, 1, 2]),
	                  MixtureDistribution( [UniformDistribution( 5, 2 ),
	                  	                    NormalDistribution( 7, 1 ),
	                  	                    NormalDistribution( 3, 0.5 )] )
	                ]

	for distribution in distributions:
		print_benchmark( distribution, bench_log_probability( distribution ) )

	distribution = DiscreteDistribution({'A': 0.2, 'B': 0.27, 'C': 0.3, 'D': 0.23})
	print_benchmark( distribution, bench_log_probability( distribution ) )

	distribution = IndependentComponentsDistribution([ NormalDistribution( 5, 1 ),
		                                               NormalDistribution( 8, 0.5),
		                                               NormalDistribution( 2, 0.1),
		                                               NormalDistribution( 13, 0.1),
		                                               NormalDistribution( 0.5, 0.01) ])

	print_benchmark( distribution, bench_log_probability( distribution, symbol=(5,4,3,2,1) ) )

	mu = np.random.randn(4)
	cov = np.random.randn(4, 4) / 10
	cov = np.abs( cov.dot( cov.T ) ) + np.eye( 4 )
	distribution = MultivariateGaussianDistribution( mu, cov )

	print_benchmark( distribution, bench_log_probability( distribution, n=100000, symbol=(1,2,3,4) ) )

def benchmark_distribution_train():
	"""Run training benchmarks."""

	distributions = [ UniformDistribution( 0, 17 ),
	                  NormalDistribution( 7, 1 ),
	                  LogNormalDistribution( 7, 1 ),
	                  ExponentialDistribution( 7 ),
	                  GammaDistribution( 7, 3 ),
	                  GaussianKernelDensity([0, 1, 4, 3, 2, 0.5, 2, 1, 2]),
	                  UniformKernelDensity([0, 1, 4, 3, 2, 0.5, 2, 1, 2]),
	                  TriangleKernelDensity([0, 1, 4, 3, 2, 0.5, 2, 1, 2]),
	                  MixtureDistribution( [UniformDistribution( 5, 2 ),
	                  	                    NormalDistribution( 7, 1 ),
	                  	                    NormalDistribution( 3, 0.5 )] )
	                ]

	sample = np.random.randn(10000)

	for distribution in distributions:
		print_benchmark( distribution, bench_from_sample( distribution, sample ) )

	sample = ['A']*2500 + ['B']*3000 + ['C']*3500 + ['D']*1000

	distribution = DiscreteDistribution({'A': 0.2, 'B': 0.27, 'C': 0.3, 'D': 0.23})
	print_benchmark( distribution, bench_from_sample( distribution, sample ) )

	sample = np.random.randn(10000, 5)
	distribution = IndependentComponentsDistribution([ NormalDistribution( 5, 1 ),
		                                               NormalDistribution( 8, 0.5),
		                                               NormalDistribution( 2, 0.1),
		                                               NormalDistribution( 13, 0.1),
		                                               NormalDistribution( 0.5, 0.01) ])

	print_benchmark( distribution, bench_from_sample( distribution, sample ) )

	sample = np.random.randn(10000, 4)
	mu = np.random.randn(4)
	cov = np.random.randn(4, 4) / 10
	cov = np.abs( cov.dot( cov.T ) ) + np.eye( 4 )
	distribution = MultivariateGaussianDistribution( mu, cov )

	print_benchmark( distribution, bench_from_sample( distribution, sample ) )

print( "DISTRIBUTION BENCHMARKS" )
print( "-----------------------" )
print()
print( "LOG PROBABILITY (N=10,000,000 iterations, N=100,000 FOR MVG)" )
benchmark_distribution_log_probabilities()
print()
print( "TRAINING (N=1,000 ITERATIONS, BATCHES=10,000 ITEMS)" )
benchmark_distribution_train()

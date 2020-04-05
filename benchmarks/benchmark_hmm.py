# benchmark_hmm.py
# Contact: Jacob Schreiber ( jmschreiber91@gmail.com )

"""
Benchmark the HMM module, including multithreading.
"""

from pomegranate import *
import numpy as np
import time
import random

np.random.seed(0)
random.seed(0)

def global_alignment( match_distributions, insert_distribution ):
	"""Create a global alignment HMM from match distributions.
	"""

	model = HiddenMarkovModel()

	i0 = State( insert_distribution, name="i0" )
	model.add_state(i0)
	model.add_transition( i0, i0, 0.3 )
	model.add_transition( model.start, i0, 0.3 )

	last_match, last_insert, last_delete = model.start, i0, None
	for i, distribution in enumerate( match_distributions ):
		match = State( distribution, name="m{}".format(i+1) )
		insert = State( insert_distribution, name="i{}".format(i+1) )
		delete = State( None, name="d{}".format(i+1) )
		model.add_states([match, insert, delete])

		model.add_transition( last_match, match, 0.5 )
		model.add_transition( last_match, delete, 0.1 )
		model.add_transition( last_insert, match, 0.5 )
		model.add_transition( last_insert, delete, 0.2 )
		if last_delete is not None:
			model.add_transition( last_delete, match, 0.7 )
			model.add_transition( last_delete, delete, 0.2 )

		model.add_transition( insert, insert, 0.3 )
		model.add_transition( match, insert, 0.1 )
		model.add_transition( delete, insert, 0.1 )

		last_match, last_insert, last_delete = match, insert, delete

	model.add_transition( last_match, model.end, 0.6 )
	model.add_transition( last_insert, model.end, 0.7 )
	model.add_transition( last_delete, model.end, 0.9 )

	model.bake()
	return model

def benchmark_forward( model, sample ):
	tic = time.time()
	for i in range(25000):
		logp = model.forward( sample )[-1, model.end_index]
	print("{:16}: time: {:5.5}, logp: {:5.5}".format( "FORWARD", time.time() - tic, logp ))

def benchmark_backward( model, sample ):
	tic = time.time()
	for i in range(25000):
		logp = model.backward( sample )[0, model.start_index]
	print("{:16}: time: {:5.5}, logp: {:5.5}".format( "BACKWARD", time.time() - tic, logp ))

def benchmark_forward_backward( model, sample ):
	tic = time.time()
	for i in range(25000):
		model.forward_backward( sample )
	print("{:16}: time: {:5.5}".format( "FORWARD-BACKWARD", time.time() - tic ))

def benchmark_viterbi( model, sample ):

	tic = time.time()
	for i in range(25000):
		logp, path = model.viterbi( sample )
	print("{:16}: time: {:5.5}, logp: {:5.5}".format( "VITERBI", time.time() - tic, logp ))

def benchmark_training( model, samples, n_jobs ):
	tic = time.time()
	improvement = model.train( samples, max_iterations=10, verbose=False, n_jobs=n_jobs )
	print("{:16}: time: {:5.5}, improvement: {:5.5} ({} jobs)".format( "BW TRAINING", time.time() - tic, improvement, n_jobs ))

def main():
	n = 15

	print("HIDDEN MARKOV MODEL BENCHMARKS")
	print("gaussian emissions")
	sigma = 3
	means = np.random.randn(n)*20
	gaussian_dists = [ NormalDistribution(mean, 1) for mean in means ]
	gaussian_model = global_alignment( gaussian_dists, NormalDistribution(0, 10) )
	gaussian_sample = np.random.randn(n)*sigma + means
	gaussian_batch = np.random.randn(50, n)*sigma + means

	benchmark_forward( gaussian_model, gaussian_sample )
	benchmark_backward( gaussian_model, gaussian_sample )
	benchmark_viterbi( gaussian_model, gaussian_sample )
	benchmark_forward_backward( gaussian_model, gaussian_sample )
	benchmark_training( gaussian_model, gaussian_batch, 1 )

	# Reset the model
	gaussian_dists = [ NormalDistribution(mean, 1) for mean in means ]
	gaussian_model = global_alignment( gaussian_dists, NormalDistribution(0, 10) )

	benchmark_training( gaussian_model, gaussian_batch, 4 )

	print
	print("multivariate gaussian emissions")
	m = 10
	means = np.random.randn(n, m) * 5 + np.arange(m) * 3
	mgd = MultivariateGaussianDistribution
	multivariate_gaussian_dists = [ mgd( mean, np.eye(m) ) for mean in means ]
	multivariate_gaussian_model = global_alignment( multivariate_gaussian_dists, mgd( np.zeros(m), np.eye(m)*3 ) )
	multivariate_gaussian_sample = np.random.randn(n, m)*sigma + means
	multivariate_gaussian_batch = np.random.randn(500, n, m)*sigma + means

	benchmark_forward( multivariate_gaussian_model, multivariate_gaussian_sample )
	benchmark_backward( multivariate_gaussian_model, multivariate_gaussian_sample  )
	benchmark_viterbi( multivariate_gaussian_model, multivariate_gaussian_sample  )
	benchmark_forward_backward( multivariate_gaussian_model, multivariate_gaussian_sample  )
	benchmark_training( multivariate_gaussian_model, multivariate_gaussian_batch, 1 )

	# Reset the model
	multivariate_gaussian_dists = [ mgd( mean, np.eye(m) ) for mean in means ]
	multivariate_gaussian_model = global_alignment( multivariate_gaussian_dists, mgd( np.zeros(m), np.eye(m)*3 ) )

	benchmark_training( multivariate_gaussian_model, multivariate_gaussian_batch, 4 )

	print
	print("discrete distribution")
	probs = np.abs( np.random.randn(n, 4) * 20 ).T
	probs = (probs / probs.sum(axis=0)).T
	discrete_dists = [ DiscreteDistribution({ char: prob for char, prob in zip('ACGT', row)}) for row in probs ]
	discrete_model = global_alignment( discrete_dists, DiscreteDistribution({'A': 0.25, 'C': 0.25, 'G': 0.25, 'T': 0.25}))
	discrete_sample = list('ACGTAGCTACGACATCAGAC')
	discrete_batch = np.array([ np.random.choice(list('ACGT'), size=200, p=row) for row in probs ]).T

	benchmark_forward( discrete_model, discrete_sample )
	benchmark_backward( discrete_model, discrete_sample  )
	benchmark_viterbi( discrete_model, discrete_sample  )
	benchmark_forward_backward( discrete_model, discrete_sample  )
	benchmark_training( discrete_model, discrete_batch, 1 )

	# Reset the model
	discrete_dists = [ DiscreteDistribution({ char: prob for char, prob in zip('ACGT', row)}) for row in probs ]
	discrete_model = global_alignment( discrete_dists, DiscreteDistribution({'A': 0.25, 'C': 0.25, 'G': 0.25, 'T': 0.25}))

	benchmark_training( discrete_model, discrete_batch, 4 )


if __name__ == '__main__':
	main()

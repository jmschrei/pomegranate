# distributions.pxd
# Contact: Jacob Schreiber ( jmschreiber91@gmail.com )

cimport numpy

from .base cimport Model

ctypedef numpy.npy_float64 DOUBLE_t 
ctypedef numpy.npy_intp SIZE_t  

cdef class Distribution( Model ):
	cdef public list summaries

cdef class UniformDistribution( Distribution ):
	cdef double start, end, logp

cdef class BernoulliDistribution( Distribution ):
	cdef double p
	cdef double* logp

cdef class NormalDistribution( Distribution ):
	cdef double mu, sigma, two_sigma_squared, log_sigma_sqrt_2_pi
	cdef object min_std

cdef class LogNormalDistribution( Distribution ):
	cdef double mu, sigma

cdef class ExponentialDistribution( Distribution ):
	cdef double rate, log_rate

cdef class BetaDistribution( Distribution ):
	cdef double alpha, beta, beta_norm

cdef class GammaDistribution( Distribution ):
	cdef double alpha, beta

cdef class DiscreteDistribution( Distribution ):
	cdef bint encoded_summary
	cdef int n
	cdef dict dist, log_dist
	cdef tuple encoded_keys
	cdef double* encoded_counts
	cdef double* encoded_log_probability
	cdef double __log_probability( self, symbol )

cdef class PoissonDistribution( Distribution ):
	cdef double l, logl

cdef class LambdaDistribution( Distribution ):
	pass

cdef class KernelDensity( Distribution ):
	cdef numpy.ndarray points_ndarray, weights_ndarray
	cdef double* points
	cdef double* weights
	cdef int n
	cdef double bandwidth

cdef class GaussianKernelDensity( KernelDensity ):
	pass

cdef class UniformKernelDensity( KernelDensity ):
	pass

cdef class TriangleKernelDensity( KernelDensity ):
	pass

cdef class MixtureDistribution( Distribution ):
	cdef double* weights_p
	cdef numpy.ndarray distributions, weights
	cdef int n

cdef class MultivariateDistribution( Distribution ):
	pass

cdef class IndependentComponentsDistribution( MultivariateDistribution ):
	cdef public numpy.ndarray distributions, weights
	cdef int discrete
	cdef double* weights_ptr
	cdef void** distributions_ptr

cdef class MultivariateGaussianDistribution( MultivariateDistribution ):
	cdef public numpy.ndarray mu, cov, inv_cov
	cdef double* _mu
	cdef double* _mu_new
	cdef double* _cov
	cdef double _log_det
	cdef double w_sum
	cdef double* column_sum
	cdef double* pair_sum
	cdef double* chol_dot_mu
	cdef double* _inv_cov
	cdef double* _inv_dot_mu

cdef class DirichletDistribution( MultivariateDistribution ):
	cdef public numpy.ndarray alphas
	cdef double* alphas_ptr
	cdef double beta_norm
	cdef numpy.ndarray summaries_ndarray
	cdef double* summaries_ptr

cdef class ConditionalProbabilityTable( MultivariateDistribution ):
	cdef double* values
	cdef double* counts
	cdef double* marginal_counts
	cdef int n, k
	cdef int* idxs
	cdef int* marginal_idxs
	cdef public list parents, parameters
	cdef public object keymap
	cdef public object marginal_keymap
	cdef public int m
	cdef void __summarize( self, items, double [:] weights )

cdef class JointProbabilityTable( MultivariateDistribution ):
	cdef double* values
	cdef double* counts
	cdef double count
	cdef int n, k
	cdef int* idxs
	cdef public list parents, parameters
	cdef public object keymap
	cdef public object marginal_keymap
	cdef public int m
	cdef void __summarize( self, items, double [:] weights )
# distributions.pxd
# Contact: Jacob Schreiber ( jmschreiber91@gmail.com )

cimport numpy

ctypedef numpy.npy_float64 DOUBLE_t 
ctypedef numpy.npy_intp SIZE_t  

cdef class Distribution:
	cdef public str name
	cdef public list summaries
	cdef public bint frozen
	cdef double _log_probability( self, double symbol ) nogil
	cdef double _mv_log_probability( self, double* symbol ) nogil
	cdef void _summarize( self, double* items, double* weights, SIZE_t n ) nogil
	cdef public int d

cdef class UniformDistribution( Distribution ):
	cdef double start, end

cdef class NormalDistribution( Distribution ):
	cdef double mu, sigma, two_sigma_squared, log_sigma_sqrt_2_pi

cdef class LogNormalDistribution( Distribution ):
	cdef double mu, sigma

cdef class ExponentialDistribution( Distribution ):
	cdef double rate

cdef class BetaDistribution( Distribution ):
	cdef double alpha, beta

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
	cdef double* weights_ptr
	cdef void** distributions_ptr

cdef class MultivariateGaussianDistribution( MultivariateDistribution ):
	cdef public numpy.ndarray mu, cov, inv_cov_ndarray
	cdef double* inv_cov
	cdef double* _mu
	cdef double* _mu_new
	cdef double* _cov
	cdef double* _cov_new
	cdef double _log_det
	cdef double w_sum
	cdef double* column_sum
	cdef double* pair_sum

cdef class ConditionalProbabilityTable( MultivariateDistribution ):
	cdef dict key_dict
	cdef public list parameters
	cdef void _table_summarize( self, items, double [:] weights )

cdef class JointProbabilityTable( MultivariateDistribution ):
	cdef dict key_dict
	cdef public list parameters
	cdef void _table_summarize( self, items, double [:] weights )
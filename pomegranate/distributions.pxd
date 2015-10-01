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

cdef class LambdaDistribution( Distribution ):
	pass

cdef class GaussianKernelDensity( Distribution ):
	cdef double [:] points, weights
	cdef int n
	cdef double bandwidth

cdef class UniformKernelDensity( Distribution ):
	cdef double [:] points, weights
	cdef int n
	cdef double bandwidth

cdef class TriangleKernelDensity( Distribution ):
	cdef double [:] points, weights
	cdef int n
	cdef double bandwidth

cdef class MixtureDistribution( Distribution ):
	cdef double* weights_p
	cdef double [:] weights
	cdef Distribution [:] distributions
	cdef int n

cdef class MultivariateDistribution( Distribution ):
	cdef public SIZE_t d

cdef class IndependentComponentsDistribution( MultivariateDistribution ):
	cdef double [:] weights
	cdef Distribution [:] distributions

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
	cdef void _from_summaries( self, double inertia )

cdef class ConditionalProbabilityTable( MultivariateDistribution ):
	cdef public list parameters

cdef class JointProbabilityTable( MultivariateDistribution ):
	cdef public list parameters
	cdef double [:] _from_sample( self, int [:,:] items, 
		double [:] weights, double inertia, double pseudocount )
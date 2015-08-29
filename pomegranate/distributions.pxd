# distributions.pxd
# Contact: Jacob Schreiber ( jmschreiber91@gmail.com )

cimport numpy

ctypedef numpy.npy_float64 DOUBLE_t 
ctypedef numpy.npy_intp SIZE_t  

cdef class Distribution:
	cdef public str name
	cdef public list parameters, summaries
	cdef public bint frozen
	cdef double _log_probability( self, double symbol ) nogil
	cdef void _summarize( self, double* items, double* weights,
		SIZE_t n, SIZE_t d ) nogil

cdef class UniformDistribution( Distribution ):
	cdef double start, end

cdef class NormalDistribution( Distribution ):
	cdef double mu, sigma 

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
	pass

cdef class IndependentComponentsDistribution( MultivariateDistribution ):
	cdef double [:] weights
	cdef Distribution [:] distributions
	cdef int n
	cdef double __log_probability( self, symbol )

cdef class MultivariateGaussianDistribution( MultivariateDistribution ):
	cdef public numpy.ndarray mu, cov, cv_chol
	cdef int d
	cdef double cv_log_det

cdef class ConditionalProbabilityTable( MultivariateDistribution ):
	cdef double [:] _from_sample( self, int [:,:] items, 
		double [:] weights, double inertia, double pseudocount )

cdef class JointProbabilityTable( MultivariateDistribution ):
	cdef double [:] _from_sample( self, int [:,:] items, 
		double [:] weights, double inertia, double pseudocount )
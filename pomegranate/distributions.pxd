# distributions.pxd
# Contact: Jacob Schreiber ( jmschreiber91@gmail.com )

cimport numpy

ctypedef numpy.npy_float64 DOUBLE_t 
ctypedef numpy.npy_intp SIZE_t  

cdef class Distribution:
	cdef public str name
	cdef public list parameters, summaries
	cdef public bint frozen
	cdef void _summarize( self, double* items, double* weights,
		SIZE_t n, SIZE_t d ) nogil

cdef class UniformDistribution( Distribution ):
	cdef double start, end
	cdef double _log_probability( self, double symbol )

cdef class NormalDistribution( Distribution ):
	cdef double mu, sigma 
	cdef double _log_probability( self, double symbol )

cdef class LogNormalDistribution( Distribution ):
	cdef double mu, sigma
	cdef double _log_probability( self, double symbol )

cdef class ExponentialDistribution( Distribution ):
	cdef double rate
	cdef double _log_probability( self, double symbol )

cdef class BetaDistribution( Distribution ):
	cdef double alpha, beta
	cdef double _log_probability( self, double symbol )

cdef class GammaDistribution( Distribution ):
	cdef double alpha, beta
	cdef double _log_probability( self, double symbol )

cdef class DiscreteDistribution( Distribution ):
	cdef int n
	cdef double _log_probability( self, symbol )

cdef class LambdaDistribution( Distribution ):
	pass

cdef class GaussianKernelDensity( Distribution ):
	cdef double [:] points, weights
	cdef int n
	cdef double bandwidth
	cdef double _log_probability( self, double symbol )

cdef class UniformKernelDensity( Distribution ):
	cdef double [:] points, weights
	cdef int n
	cdef double bandwidth
	cdef double _log_probability( self, double symbol )

cdef class TriangleKernelDensity( Distribution ):
	cdef double [:] points, weights
	cdef int n
	cdef double bandwidth
	cdef double _log_probability( self, double symbol )

cdef class MixtureDistribution( Distribution ):
	cdef double [:] weights
	cdef Distribution [:] distributions
	cdef int n
	cdef double _log_probability( self, double symbol )


cdef class MultivariateDistribution( Distribution ):
	pass


cdef class IndependentComponentsDistribution( MultivariateDistribution ):
	cdef double [:] weights
	cdef Distribution [:] distributions
	cdef int n
	cdef double _log_probability( self, symbol )

cdef class MultivariateGaussianDistribution( MultivariateDistribution ):
	cdef public numpy.ndarray mu, cov
	cdef int d

cdef class ConditionalProbabilityTable( MultivariateDistribution ):
	cdef double [:] _from_sample( self, int [:,:] items, 
		double [:] weights, double inertia, double pseudocount )

cdef class JointProbabilityTable( MultivariateDistribution ):
	cdef double [:] _from_sample( self, int [:,:] items, 
		double [:] weights, double inertia, double pseudocount )

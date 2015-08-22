# distributions.pxd
# Contact: Jacob Schreiber ( jmschreiber91@gmail.com )

cimport numpy

ctypedef numpy.npy_float64 DOUBLE_t 
ctypedef numpy.npy_intp SIZE_t  

cdef class Distribution:
	cdef public str name
	cdef public list parameters, summaries
	cdef public bint frozen, is_numeric_type
	cdef double _dlog_probability( self, double symbol ) nogil
	cdef double _slog_probability( self, char* symbol ) nogil

cdef class UniformDistribution( Distribution ):
	cdef double start, end
	cdef void _from_sample( self, double [:] items, double [:] weights, 
		DOUBLE_t inertia, SIZE_t size ) nogil
	cdef void _summarize( self, double [:] items, double [:] weights,
		SIZE_t size ) nogil

cdef class NormalDistribution( Distribution ):
	cdef double mu, sigma 
	cdef void _from_sample( self, numpy.ndarray items, numpy.ndarray weights,
		DOUBLE_t inertia, DOUBLE_t min_std, SIZE_t size ) nogil

cdef class LogNormalDistribution( Distribution ):
	cdef double mu, sigma

cdef class ExtremeValueDistribution( Distribution ):
	cdef double mu, sigma, epsilon

cdef class ExponentialDistribution( Distribution ):
	cdef double rate

cdef class BetaDistribution( Distribution ):
	cdef double alpha, beta

cdef class GammaDistribution( Distribution ):
	cdef double alpha, beta

cdef struct KeyValuePair:
	char* key
	double value

cdef class DiscreteDistribution( Distribution ):
	cdef KeyValuePair* records
	cdef int n

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
	cdef double [:] weights
	cdef Distribution [:] distributions
	cdef int n


cdef class MultivariateDistribution( Distribution ):
	pass


#cdef class IndependentComponentsDistribution( MultivariateDistribution ):
#	cdef double [:] weights
#	cdef Distribution [:] distributions
#	cdef int n
#	cdef double _log_probability( self, obs symbol ) nogil

cdef class MultivariateGaussianDistribution( MultivariateDistribution ):
	cdef public int diagonal 

cdef class ConditionalProbabilityTable( MultivariateDistribution ):
	cdef double [:] _from_sample( self, int [:,:] items, 
		double [:] weights, double inertia, double pseudocount )

cdef class JointProbabilityTable( MultivariateDistribution ):
	cdef double [:] _from_sample( self, int [:,:] items, 
		double [:] weights, double inertia, double pseudocount )

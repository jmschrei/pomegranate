# distributions.pxd
# Contact: Jacob Schreiber ( jmschreiber91@gmail.com )

cimport numpy

cdef class Distribution( object ):
	cdef public str name
	cdef public list parameters, summaries
	cdef public bint frozen

cdef class UniformDistribution( Distribution ):
	cdef double start, end
	cdef public double _log_probability( self, double symbol ) nogil

cdef class NormalDistribution( Distribution ):
	cdef double mu, sigma 
	cdef public double _log_probability( self, double symbol ) nogil

cdef class LogNormalDistribution( Distribution ):
	cdef double mu, sigma
	cdef public double _log_probability( self, double symbol ) nogil

cdef class ExtremeValueDistribution( Distribution ):
	cdef double mu, sigma, epsilon
	cdef public double _log_probability( self, double symbol ) nogil

cdef class ExponentialDistribution( Distribution ):
	cdef double rate
	cdef public double _log_probability( self, double symbol ) nogil

cdef class BetaDistribution( Distribution ):
	cdef double alpha, beta
	cdef public double _log_probability( self, double symbol ) nogil

cdef class GammaDistribution( Distribution ):
	cdef double alpha, beta
	cdef public double _log_probability( self, double symbol ) nogil

cdef class InverseGammaDistribution( GammaDistribution ):
	pass

cdef class DiscreteDistribution( Distribution ):
	cdef dict dist

cdef class LambdaDistribution( Distribution ):
	pass

cdef class GaussianKernelDensity( Distribution ):
	cdef double [:] points, weights
	cdef int n
	cdef double bandwidth
	cdef public double _log_probability( self, double symbol ) nogil

cdef class UniformKernelDensity( Distribution ):
	cdef double [:] points, weights
	cdef int n
	cdef double bandwidth
	cdef public double _log_probability( self, double symbol ) nogil

cdef class TriangleKernelDensity( Distribution ):
	cdef double [:] points, weights
	cdef int n
	cdef double bandwidth
	cdef public double _log_probability( self, double symbol ) nogil

cdef class MixtureDistribution( Distribution ):
	cdef double [:] weights
	cdef Distribution [:] distributions
	cdef int n
	cdef public double _log_probability( self, double symbol )

cdef class MultivariateDistribution( Distribution ):
	pass

cdef class IndependentComponentsDistribution( MultivariateDistribution ):
	cdef double [:] weights
	cdef Distribution [:] distributions
	cdef int n
	cdef public double _log_probability( self, numpy.ndarray symbol )

cdef class MultivariateGaussianDistribution( MultivariateDistribution ):
	cdef public int diagonal 

cdef class ConditionalProbabilityTable( MultivariateDistribution ):
	cdef double [:] _from_sample( self, int [:,:] items, 
		double [:] weights, double inertia, double pseudocount )

cdef class JointProbabilityTable( MultivariateDistribution ):
	cdef double [:] _from_sample( self, int [:,:] items, 
		double [:] weights, double inertia, double pseudocount )
# distributions.pxd
# Contact: Jacob Schreiber ( jmschreiber91@gmail.com )

cimport numpy

cdef class Distribution( object ):
	cdef public str name
	cdef public list parameters, summaries
	cdef public bint frozen

cdef class UniformDistribution( Distribution ):
	cdef double _log_probability( self, double a, double b, double symbol )

cdef class NormalDistribution( Distribution ): 
	cdef double _log_probability( self, double symbol, double epsilon )

cdef class LogNormalDistribution( Distribution ):
	cdef double _log_probability( self, double symbol )

cdef class ExtremeValueDistribution( Distribution ):
	cdef double _log_probability( self, double symbol )

cdef class ExponentialDistribution( Distribution ):
	pass

cdef class GammaDistribution( Distribution ):
	pass

cdef class InverseGammaDistribution( GammaDistribution ):
	pass

cdef class DiscreteDistribution( Distribution ):
	pass

cdef class LambdaDistribution( Distribution ):
	pass

cdef class GaussianKernelDensity( Distribution ):
	cdef double _log_probability( self, double symbol )

cdef class UniformKernelDensity( Distribution ):
	cdef double _log_probability( self, double symbol )

cdef class TriangleKernelDensity( Distribution ):
	cdef double _log_probability( self, double symbol )

cdef class MixtureDistribution( Distribution ):
	pass

cdef class MultivariateDistribution( Distribution ):
	pass

cdef class IndependentComponentDistribution( MultivariateDistribution ):
	pass

cdef class MultivariateGaussianDistribution( MultivariateDistribution ):
	cdef public int diagonal 

cdef class ConditionalProbabilityTable( MultivariateDistribution ):
	cdef double [:] _from_sample( self, int [:,:] items, 
		double [:] weights, double inertia, double pseudocount )

cdef class JointProbabilityTable( MultivariateDistribution ):
	cdef double [:] _from_sample( self, int [:,:] items, 
		double [:] weights, double inertia, double pseudocount )
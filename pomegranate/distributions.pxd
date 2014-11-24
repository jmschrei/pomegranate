# distributions.pxd
# Contact: Jacob Schreiber ( jmschreiber91@gmail.com )

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

cdef class DiscreteDistribution( Distribution ):
	pass

cdef class LambdaDistribution( Distribution ):
	pass

cdef class GaussianKernelDensity( Distribution ):
	cdef double _log_probability( self, double symbol )

cdef class UniformKernelDensity( Distribution ):
	cdef _log_probability( self, double symbol )

cdef class TriangleKernelDensity( Distribution ):
	cdef double _log_probability( self, double symbol )

cdef class MixtureDistribution( Distribution ):
	pass

cdef class MultivariateDistribution( Distribution ):
	pass

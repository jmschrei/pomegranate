# distributions.pxd
# Contact: Jacob Schreiber (jmschreiber91@gmail.com)

cimport numpy

from ..base cimport Model

ctypedef numpy.npy_float64 DOUBLE_t
ctypedef numpy.npy_intp SIZE_t

cdef class Distribution(Model):
	cdef public list summaries

cdef class MultivariateDistribution(Distribution):
	pass


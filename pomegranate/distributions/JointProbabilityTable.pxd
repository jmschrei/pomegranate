# JointProbabilityTable.pxd
# Contact: Jacob Schreiber <jmschreiber91@gmail.com>

import numpy
cimport numpy

from .distributions cimport MultivariateDistribution

cdef class JointProbabilityTable(MultivariateDistribution):
	cdef double* values
	cdef double* counts
	cdef double count
	cdef public int n, k, n_columns
	cdef int* idxs
	cdef public list parents, parameters, dtypes, column_keys
	cdef public object keymap
	cdef public object marginal_keymap
	cdef public numpy.ndarray column_idxs
	cdef double* column_idxs_ptr
	cdef public int m
	cdef void __summarize(self, items, double [:] weights)


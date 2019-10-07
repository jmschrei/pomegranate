# utils.pyx
# Contact: Jacob Schreiber ( jmschreiber91@gmail.com )

from libc.math cimport log as clog
from libc.math cimport sqrt as csqrt
from libc.math cimport exp as cexp
from libc.math cimport floor
from libc.math cimport fabs

from libc.stdlib cimport calloc, free
from scipy.linalg.cython_blas cimport dgemm

cimport numpy
import numpy
import numbers

import heapq, itertools

try:
	import tempfile
	import pygraphviz
	import matplotlib.pyplot as plt
	import matplotlib.image
except ImportError:
	pygraphviz = None

cdef int* GPU = <int*> calloc(1, sizeof(int))
cdef int* has_cupy = <int*> calloc(1, sizeof(int))

try:
	import cupy
	from cupy import cuda
	cuda.Device().cublas_handle

	global has_cupy
	has_cupy[0] = 1

	enable_gpu()
except:
	global has_cupy
	has_cupy[0] = 0

numpy.import_array()

cdef extern from "numpy/ndarraytypes.h":
	void PyArray_ENABLEFLAGS(numpy.ndarray X, int flags)

# Define some useful constants
DEF NEGINF = float("-inf")
DEF INF = float("inf")
DEF SQRT_2_PI = 2.50662827463
DEF GAMMA = 0.577215664901532860606512090
DEF HALF_LOG2_PI = 0.91893853320467274178032973640562

cdef class PriorityQueue(object):
	cdef int n
	cdef public list pq
	cdef dict entries

	def __init__(self):
		self.n = 0
		self.pq = []
		self.entries = {}

	def push(self, item, weight):
		entry = [weight, item]
		self.entries[item[0]] = entry
		heapq.heappush(self.pq, entry)
		self.n += 1

	def get(self, variables):
		return self.entries.get(variables, None)

	def delete(self, variables):
		entry = self.entries.pop(variables)
		entry[-1] = ((-1,),)
		self.n -= 1

	def empty(self):
		return self.n == 0

	def pop(self):
		while not self.empty():
			weight, item = heapq.heappop(self.pq)
			if item[0] != (-1,):
				del self.entries[item[0]]
				self.n -= 1
				return weight, item
		else:
			raise KeyError("Attempting to pop from an empty priority queue")


def is_gpu_enabled():
	global GPU
	return bool(GPU[0])

cdef int _is_gpu_enabled() nogil:
	return GPU[0]

cpdef enable_gpu():
	global GPU
	
	if has_cupy[0] == 0:
		raise Warning("Please install cupy before attempting to utilize a GPU.")
	else:
		GPU[0] = 1

cpdef disable_gpu():
	global GPU
	GPU[0] = 0

cdef ndarray_wrap_cpointer(void* data, numpy.npy_intp n):
	cdef numpy.ndarray[numpy.float64_t, ndim=1] X = numpy.PyArray_SimpleNewFromData(1, &n, numpy.NPY_FLOAT64, data)
	return X

cdef python_log_probability(model, double* X, double* log_probability, int n):
	cdef int i
	cdef numpy.npy_intp dim = n * model.d
	cdef numpy.ndarray X_ndarray

	X_ndarray = numpy.PyArray_SimpleNewFromData(1, &dim, numpy.NPY_FLOAT64, X)
	X_ndarray = X_ndarray.reshape(n, model.d)

	logp = model.log_probability(X_ndarray)
	
	if n == 1:
		log_probability[0] = logp
	else:
		for i in range(n):
			log_probability[i] = logp[i]

cdef python_summarize(model, double* X, double* weights, int n):
	cdef int i
	cdef numpy.npy_intp dim = n * model.d
	cdef numpy.npy_intp n_elements = n
	cdef numpy.ndarray X_ndarray
	cdef numpy.ndarray w_ndarray

	X_ndarray = numpy.PyArray_SimpleNewFromData(1, &dim, numpy.NPY_FLOAT64, X)
	if model.d > 1:
		X_ndarray = X_ndarray.reshape(n, model.d)

	w_ndarray = numpy.PyArray_SimpleNewFromData(1, &n_elements, 
		numpy.NPY_FLOAT64, weights)
	
	model.summarize(X_ndarray, w_ndarray)
	

cdef void mdot(double* X, double* Y, double* A, int m, int n, int k) nogil:
	cdef double alpha = 1
	cdef double beta = 0
	dgemm('N', 'N', &n, &m, &k, &alpha, Y, &n, X, &k, &beta, A, &n)

cpdef bdot(numpy.ndarray X_ndarray):
	cdef int n = X_ndarray.shape[0]
	cdef int d = X_ndarray.shape[1]

	cdef double* x = <double*> X_ndarray.data

	cdef double alpha = 1
	cdef double beta = 1

	cdef numpy.ndarray c_ndarray = numpy.zeros((d, d), dtype='float64')
	cdef double* c = <double*> c_ndarray.data

	dgemm('N', 'T', &d, &d, &n, &alpha, x, &d, x, &d, &beta, c, &d)
	#dgemm('T', 'N', &n, &n, &d, &alpha, x, &d, x, &d, &beta, c, &n)
	return c_ndarray


cpdef numpy.ndarray _convert( data ):
	if type(data) is numpy.ndarray:
		return data
	if type(data) is int:
		return numpy.array( [data] )
	if type(data) is float:
		return numpy.array( [data] )
	if type(data) is list:
		return numpy.array( data )

# Useful speed optimized functions
cdef double _log(double x) nogil:
	'''
	A wrapper for the c log function, by returning negative infinity if the
	input is 0.
	'''

	return clog( x ) if x > 0 else NEGINF

cdef double pair_lse(double x, double y) nogil:
	'''
	Perform log-sum-exp on a pair of numbers in log space..  This is calculated
	as z = log( e**x + e**y ). However, this causes underflow sometimes
	when x or y are too negative. A simplification of this is thus
	z = x + log( e**(y-x) + 1 ), where x is the greater number. If either of
	the inputs are infinity, return infinity, and if either of the inputs
	are negative infinity, then simply return the other input.
	'''

	if x == INF or y == INF:
		return INF
	if x == NEGINF:
		return y
	if y == NEGINF:
		return x
	if x > y:
		return x + clog(cexp(y-x) + 1)
	return y + clog(cexp(x-y) + 1)

def logsumexp(X):
	"""Calculate the log-sum-exp of an array to add in log space."""

	X = numpy.array(X, dtype='float64')
	
	cdef double* X_ptr = <double*> (<numpy.ndarray> X).data
	cdef double x
	cdef int i, n = X.shape[0]
	cdef double y = 0.
	cdef double x_max = NEGINF

	with nogil:
		for i in range(n):
			x = X_ptr[i]
			if x > x_max:
				x_max = x

		for i in range(n):
			x = X_ptr[i]
			if x == NEGINF:
				continue

			y += cexp(x - x_max)

	return x_max + clog(y)

def logaddexp(X, Y):
	"""Calculate the log-add-exp of a pair of arrays."""

	X = numpy.array(X, dtype='float64')
	Y = numpy.array(Y, dtype='float64')

	if len(X.shape) != len(Y.shape):
		raise ValueError("Both arrays must be of the same shape.")
	if X.shape[0] != Y.shape[0]:
		raise ValueError("Both arrays must be of the same shape.")
	if len(X.shape) > 1:
		raise ValueError("Both arrays must of one dimensional.")

	Z = numpy.zeros_like(Y)

	cdef int i, n = X.shape[0]
	cdef double* X_ptr = <double*> (<numpy.ndarray> X).data
	cdef double* Y_ptr = <double*> (<numpy.ndarray> Y).data
	cdef double* Z_ptr = <double*> (<numpy.ndarray> Z).data

	with nogil:
		for i in range(n):
			Z_ptr[i] = pair_lse(X_ptr[i], Y_ptr[i])

	return Z


cdef double gamma(double x) nogil:
	"""Calculate the gamma function on a number."""

	# Split the function domain into three intervals:
	# (0, 0.001), [0.001, 12), and (12, infinity).

	# First interval: (0, 0.001).
	# For small x, 1/Gamma(x) has power series x + gamma x^2  - ...
	# So in this range, 1/Gamma(x) = x + gamma x^2 with error
	# on the order of x^3.
	# The relative error over this interval is less than 6e-7.


	cdef double p[8]
	p[0] = -1.71618513886549492533811E+0
	p[1] =	2.47656508055759199108314E+1
	p[2] = -3.79804256470945635097577E+2
	p[3] =  6.29331155312818442661052E+2
	p[4] =  8.66966202790413211295064E+2
	p[5] = -3.14512729688483675254357E+4
	p[6] = -3.61444134186911729807069E+4
	p[7] =  6.64561438202405440627855E+4

	cdef double q[8]
	q[0] = -3.08402300119738975254353E+1
	q[1] =  3.15350626979604161529144E+2
	q[2] = -1.01515636749021914166146E+3
	q[3] = -3.10777167157231109440444E+3
	q[4] =  2.25381184209801510330112E+4
	q[5] =  4.75584627752788110767815E+3
	q[6] = -1.34659959864969306392456E+5
	q[7] = -1.15132259675553483497211E+5

	cdef double den, num, result, z, y
	cdef int i, n, arg_was_less_than_one

	if x == 0.0:
		return INF

	if x < 0.001:
		return 1.0 / (x * (1.0 + GAMMA * x));

	# Second interval: [0.001, 12).

	if x < 12.0:
		# The algorithm directly approximates gamma over (1,2) and uses
		# reduction identities to reduce other arguments to this interval.
		y = x
		n = 0
		arg_was_less_than_one = (y < 1.0)

		# Add or subtract integers as necessary to bring y into (1,2)
		# Will correct for this below */
		if arg_was_less_than_one:
			y += 1.0
		else:
			n = <int>floor(y) - 1
			y -= n

		num = 0.0
		den = 1.0

		z = y - 1
		for i in range(8):
			num = (num + p[i]) * z
			den = den * z + q[i]

		result = num/den + 1.0

		# Apply correction if argument was not initially in (1,2)
		if arg_was_less_than_one:
			# Use identity gamma(z) = gamma(z+1)/z
			# The variable "result" now holds gamma of the original y + 1
			# Thus we use y-1 to get back the original y.
			result /= (y-1.0)
		else:
			# Use the identity gamma(z+n) = z*(z+1)* ... *(z+n-1)*gamma(z)
			for i in range(n):
				result *= y+i

		return result

	# Third interval: [12, infinity).
	if x > 171.624:
	# Correct answer too large to display, force +infinity.
		return INF

	return cexp(lgamma(x));

cdef double lgamma(double x) nogil:
	# Abramowitz and Stegun 6.1.41
	# Asymptotic series should be good to at least 11 or 12 figures
	# For error analysis, see Whittiker and Watson
	# A Course in Modern Analysis (1927), page 252

	cdef double c[8]
	c[0] =  1.0 / 12.0
	c[1] = -1.0 / 360.0
	c[2] =  1.0 / 1260.0
	c[3] = -1.0 / 1680.0
	c[4] =  1.0 / 1188.0
	c[5] = -691.0 / 360360.0
	c[6] =  1.0 / 156.0
	c[7] = -3617.0 / 122400.0

	cdef double z, sum
	cdef int i

	if x < 12.0:
		return clog(fabs(gamma(x)))

	z = 1.0 / (x * x)
	sum = c[7]

	for i in range(7):
		sum *= z
		sum += c[6-i]

	return (x - 0.5) * clog(x) - x + HALF_LOG2_PI + sum / x

def plot_networkx(Q, edge_label=None, filename=None):
	G = pygraphviz.AGraph(directed=True)

	for state in Q.nodes():
		G.add_node(state, color='red')

	for parent, child, data in Q.edges(data=True):
		if edge_label:
			G.add_edge(parent, child, label=data[edge_label])
		else:
			G.add_edge(parent, child)

	if filename is None:
		with tempfile.NamedTemporaryFile() as tf:
			G.draw(tf.name, format='png', prog='dot')
			img = matplotlib.image.imread(tf.name)
			plt.imshow(img)
			plt.axis('off')
	else:
		G.draw(filename, format='pdf', prog='dot')

def _check_input(X, keymap):
	"""Check the input to make sure that it is a properly formatted array."""

	cdef numpy.ndarray X_ndarray

	try:
		X_ndarray = numpy.array(X, dtype='float64', ndmin=2)
	except:
		if not isinstance(X, (numpy.ndarray, list, tuple)):
			X_ndarray = numpy.array(keymap[0][X], dtype='float64', ndmin=2)
		else:
			X = numpy.array(X)
			X_ndarray = numpy.empty(X.shape, dtype='float64')

			if X.ndim == 1:
				for i in range(X.shape[0]):
					X_ndarray[i] = keymap[0][X[i]]
				X_ndarray = X_ndarray.reshape(-1, 1)
			else:
				for i in range(X.shape[0]):
					for j in range(X.shape[1]):
						X_ndarray[i, j] = keymap[j][X[i, j]]


	return X_ndarray

def parallelize_function(X, cls, func, filename, **kwargs):
	"""Parallelize a function using joblib multiprocessing."""

	model = cls.from_json(filename)
	return getattr(model, func)(X, **kwargs)

def weight_set(items, weights):
	"""Converts both items and weights to appropriate numpy arrays.

	Convert the items into a numpy array with 64-bit floats, and the weight
	array to the same. If no weights are passed in, then return a numpy array
	with uniform weights.
	"""

	items = numpy.array(items, dtype=numpy.float64)
	if weights is None: # Weight everything 1 if no weights specified
		weights = numpy.ones(items.shape[0], dtype=numpy.float64)
	else: # Force whatever we have to be a Numpy array
		weights = numpy.array(weights, dtype=numpy.float64)

	return items, weights

def _check_nan(X):
	"""Checks to see if a value is nan, either as a float or a string."""
	
	if isinstance(X, (str, unicode, numpy.string_)):
		if X == 'nan':
			return True
		return False

	if X is None or numpy.isnan(X):
		return True
	return False

def check_random_state(seed):
	"""Turn seed into a np.random.RandomState instance.

	This function will check to see whether the input seed is a valid seed
	for generating random numbers. This is a slightly modified version of
	the code from sklearn.utils.validation.

	Parameters
	----------
	seed : None | int | instance of RandomState
		If seed is None, return the RandomState singleton used by np.random.
		If seed is an int, return a new RandomState instance seeded with seed.
		If seed is already a RandomState instance, return it.
		Otherwise raise ValueError.
	"""

	if seed is None or seed is numpy.random:
		return numpy.random.mtrand._rand
	if isinstance(seed, (numbers.Integral, numpy.integer)):
		return numpy.random.RandomState(seed)
	if isinstance(seed, numpy.random.RandomState):
		return seed
	raise ValueError('%r cannot be used to seed a numpy.random.RandomState'
					 ' instance' % seed)
	

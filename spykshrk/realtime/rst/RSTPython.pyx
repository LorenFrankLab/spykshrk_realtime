cimport cRSTPython as RST

cimport cRSTFunctions as RSTFunc

from cpython cimport PyObject, Py_INCREF

import numpy as np
cimport numpy as np

import struct

from libc.stdlib cimport malloc, free

np.import_array()

BUF_LEN = 10000
FIXED_POINT = 10

def TOFLOAT(x):
	return float(x)/(2**FIXED_POINT)

def TOFIXED(x):
	return int(x*(2**FIXED_POINT))

class kernel_param:
	def __init__(self, mean, stddev, min_val, max_val, interval):
		self.mean = mean
		self.stddev = stddev
		self.min_val = min_val
		self.max_val = max_val
		self.interval = interval

cdef class NPArrayWrapper:
	cdef void* data_ptr
	cdef int size
	cdef int np_data_type

	cdef set_data(self, int size, void* data_ptr, int np_data_type):
		self.data_ptr = data_ptr
		self.size = size
		self.np_data_type = np_data_type

	def __array__(self):
		cdef np.npy_intp shape[1]
		shape[0] = <np.npy_intp> self.size
		ndarray = np.PyArray_SimpleNewFromData(1, shape, self.np_data_type, self.data_ptr)
		return ndarray

	#def __dealloc__(self):
	#	free(<void*> self.data_ptr)

cdef class RSTPython:
	cdef RST.RSTREE rst
	cdef public data
	cdef public unsigned int key
	cdef RSTFunc.eval_buf rec
	cdef tree_name
	cdef kernel_p


	cdef wrap_array(self, int size, void* array, int np_data_type):
		cdef np.ndarray ndarray
		wrapper = NPArrayWrapper()
		wrapper.set_data(size,array,np_data_type)
		ndarray = np.array(wrapper, copy=False)
		ndarray.base = <PyObject*> wrapper
		Py_INCREF(wrapper)

		return ndarray


	def __cinit__(self, tree_name, new_tree, kernel_p):

		self.kernel_p = kernel_p

		# Initialize kernel in underlying data RST data structure
		RSTFunc.init_gaussian_table(
				kernel_p.mean, kernel_p.stddev, 
				kernel_p.min_val, kernel_p.max_val, kernel_p.interval)
		# Initialize Tree
		self.tree_name = tree_name
		if new_tree:
			RST.RemoveRST(tree_name)
			create_success = RST.CreateRST(tree_name,4096, False)
			if create_success:
				print('Created Tree: %s successfully' % (tree_name))
			else:
				raise Exception('Error: Tried to create a new R*-Tree but failed (%s)'%(tree_name))

		RST.NoRSTree(&self.rst)

		open_success = RST.OpenRST(&self.rst, tree_name)
		if open_success:
			print('Opened Tree:%s successfully' % (tree_name))
		else:
			raise Exception('Error: Tried to load a R*-Tree that does not exist (%s)'%(tree_name))

		self.data = {}
		self.key = 1
		# Initializing query record buffer
		self.rec.x1 = 0
		self.rec.x2 = 0
		self.rec.x3 = 0
		self.rec.x4 = 0
		self.rec.len = BUF_LEN
		self.rec.cur = 0
		self.rec.weight = <float *> malloc(BUF_LEN * sizeof(float))
		self.rec.pos = <float *> malloc(BUF_LEN * sizeof(float))

	def __dealloc__(self):
		close_success = RST.CloseRST(&self.rst)
		if close_success:
			print('Closed Tree...')
		else:
			print('Error Closing Tree...')


	def insert_rec(self, x1, x2, x3, x4, pos):
		cdef RST.typrect rect
		cdef RST.typinfo info
		cdef bint insert_success

		rect[0].l = x1
		rect[0].h = x1
		rect[1].l = x2
		rect[1].h = x2
		rect[2].l = x3
		rect[2].h = x3
		rect[3].l = x4
		rect[3].h = x4
		
		# Hide/encode the floating point position value as an integer
		# to maintain info.contents int type.  Must be unpacked as
		# a float.
		info.contents = <int> struct.unpack('i',struct.pack('f', pos))[0]
		#info.contents = <int> pos
		RST.InsertRecord(self.rst,rect,&info,&insert_success)
		return insert_success

	def query_rec(self, l1, l2,l3, l4, h1, h2, h3, h4, x1, x2, x3, x4):
		cdef RST.typrect rect
		cdef RST.typrect unused
		cdef int count
		cdef int ptr = 0
		cdef np.npy_intp shape[1]

		# reset record cursor
		self.rec.cur = 0

		rect[0].l = l1
		rect[0].h = h1
		rect[1].l = l2
		rect[1].h = h2
		rect[2].l = l3
		rect[2].h = h3
		rect[3].l = l4
		rect[3].h = h4

		self.rec.x1 = x1
		self.rec.x2 = x2
		self.rec.x3 = x3
		self.rec.x4 = x4


		RST.RegionQuery(self.rst, rect, unused, RSTFunc.Intersects, RSTFunc.Intersects, RSTFunc.EvalQuery, &self.rec)

		shape[0] = <np.npy_intp> self.rec.cur
		query_result = []
		weights = self.wrap_array(self.rec.cur, self.rec.weight, np.NPY_FLOAT)
		positions = self.wrap_array(self.rec.cur, self.rec.pos, np.NPY_FLOAT)

		return [weights, positions]

	def test_init_gaus(self, mean, stddev, min_val, max_val, interval):
		RSTFunc.init_gaussian_table(mean, stddev, min_val, max_val, interval)
	
	def test_gaus_func(self, x):
		return RSTFunc.gaussian_func(x)


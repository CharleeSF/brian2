#cython: boundscheck=False
#cython: wraparound=False
#cython: cdivision=True
#cython: infer_types=True

import numpy as _numpy
cimport numpy as _numpy
from libc.math cimport sin, cos, tan, sinh, cosh, tanh, exp, log, log10, sqrt, asin, acos, atan, fmod, floor, ceil
cdef extern from "math.h":
    double M_PI
# Import the two versions of std::abs
from libc.stdlib cimport abs, malloc, free  # For integers
from libc.math cimport abs  # For floating point values
from libcpp cimport bool

_numpy.import_array()
cdef extern from "numpy/ndarraytypes.h":
    void PyArray_CLEARFLAGS(_numpy.PyArrayObject *arr, int flags)
from libc.stdlib cimport free

cdef extern from "stdint_compat.h":
    # Longness only used for type promotion
    # Actual compile time size used for conversion
    ctypedef signed int int32_t
    ctypedef signed long int64_t
    ctypedef unsigned long uint64_t
    # It seems we cannot used a fused type here
    cdef int int_(bool)
    cdef int int_(char)
    cdef int int_(short)
    cdef int int_(int)
    cdef int int_(long)
    cdef int int_(float)
    cdef int int_(double)
    cdef int int_(long double)

cdef int function(double * v_array, double * v0_array, int _idx) nogil:
    cdef double v
    cdef double v0
    v0 = v0_array[_idx]
    v = v_array[_idx]
    v_array[_idx] = v
    v0_array[_idx] = v0
    return 0

def main(_namespace):
    cdef int _idx
    cdef int _vectorisation_idx
    _var_N = _namespace["_var_N"]
    cdef int64_t N = _namespace["N"]
    _var_v0 = _namespace["_var_v0"]
    cdef _numpy.ndarray[double, ndim=1, mode='c'] _buf__array_neurongroup_v0 = _namespace['_array_neurongroup_v0']
    cdef double * _array_neurongroup_v0 = <double *> _buf__array_neurongroup_v0.data
    cdef double v0
    _var_v = _namespace["_var_v"]
    cdef _numpy.ndarray[double, ndim=1, mode='c'] _buf__array_neurongroup_v = _namespace['_array_neurongroup_v']
    cdef double * _array_neurongroup_v = <double *> _buf__array_neurongroup_v.data
    cdef double v

    for _idx in range(N):      

        function(_array_neurongroup_v, _array_neurongroup_v0, _idx)


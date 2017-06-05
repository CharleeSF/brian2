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

cdef struct statevariable_container:
    double* v
    double* v0

cdef fill_statevariable_arrays(_namespace, statevariable_container* statevariables, int N):
    cdef _numpy.ndarray[double, ndim=1, mode='c'] _buf__array_neurongroup_v = _namespace['_array_neurongroup_v']    
    cdef double * _array_neurongroup_v = <double *> _buf__array_neurongroup_v.data
    cdef _numpy.ndarray[double, ndim=1, mode='c'] _buf__array_neurongroup_v0 = _namespace['_array_neurongroup_v0']
    cdef double * _array_neurongroup_v0 = <double *> _buf__array_neurongroup_v0.data
    cdef int _idx

    statevariables.v = <double *>malloc(N*sizeof(double))
    statevariables.v0 = <double *>malloc(N*sizeof(double))

    for _idx in range(N):
        statevariables.v[_idx] = _array_neurongroup_v[_idx]
        statevariables.v0[_idx] = _array_neurongroup_v0[_idx]

    return 0


cdef int unpack(int _idx, statevariable_container* statevariables) nogil:
    cdef double v0
    cdef double v
    v0 = statevariables.v0[_idx]
    v = statevariables.v[_idx]
    statevariables.v0[_idx] = v0
    statevariables.v[_idx] = v
    return 0

def main(_namespace):
    cdef int _idx
    _var_N = _namespace["_var_N"]
    cdef int64_t N = _namespace["N"]

    cdef statevariable_container *statevariables = <statevariable_container*>malloc(sizeof(statevariable_container))
    fill_statevariable_arrays(_namespace, statevariables, N)

    for _idx in range(N):    
        unpack(_idx, statevariables)


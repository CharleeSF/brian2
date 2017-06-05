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

cdef fill_statevariable_arrays(_namespace, double ***statevariables, int N):
    cdef _numpy.ndarray[double, ndim=1, mode='c'] _buf__array_neurongroup_v = _namespace['_array_neurongroup_v']    
    cdef double * _array_neurongroup_v = <double *> _buf__array_neurongroup_v.data
    cdef _numpy.ndarray[double, ndim=1, mode='c'] _buf__array_neurongroup_v0 = _namespace['_array_neurongroup_v0']
    cdef double * _array_neurongroup_v0 = <double *> _buf__array_neurongroup_v0.data
    cdef int _idx

    statevariables[0] = <double **>malloc(2*sizeof(double *))
    statevariables[0][0] = _array_neurongroup_v
    statevariables[0][1] = _array_neurongroup_v0

    return 0


cdef int unpack(int _idx, double ** statevariables) nogil:
    cdef double v
    cdef double v0
    v = statevariables[0][_idx]
    v0 = statevariables[1][_idx]
    statevariables[0][_idx] = v
    statevariables[1][_idx] = v0

    return 0

def main(_namespace):
    cdef int _idx
    _var_N = _namespace["_var_N"]
    cdef int64_t N = _namespace["N"]

    cdef double ** statevariables
    fill_statevariable_arrays(_namespace, &statevariables, N)

    for _idx in range(N):    
        unpack(_idx, statevariables)


#cython: language_level=3
#cython: embedsignature=True, cdivision=True, boundscheck=False, wraparound=False, initializedcheck=False
"""
float64 1-volume maximisation (maxvol).

Reduced from maxvolpy 0.3.8 by Alexander Mikhalev, MIT licence -- see
LICENSE.maxvolpy.txt next to this file. Only the float64 ``maxvol`` path is
kept, which is the only one pyace ever calls; the float32/complex variants and
the whole rectangular ``rect_maxvol`` family are gone.

Two deliberate differences from the original:

* the pivot-interchange loops use ``range`` rather than ``prange``. Building
  those swaps in parallel is a data race -- each swap reads entries earlier
  swaps wrote -- and it was only ever safe because nothing passed -fopenmp.
* the scratch buffers are released in a ``finally``, so an error raised by
  dgetrf no longer leaks them.
"""
__all__ = ['c_maxvol']

import numpy as np
cimport numpy as cnp

from libc.stdlib cimport malloc, free
from scipy.linalg.cython_blas cimport dtrsm, dcopy, dger
from scipy.linalg.cython_lapack cimport dgetrf

cnp.import_array()

cdef extern from "math.h" nogil:
    double fabs(double)


def c_maxvol(A, tol=1.05, max_iters=100, top_k_index=-1, int verbose=0):
    """
    Cython implementation of float64 1-volume maximisation.

    Returns (row_indices, coefficients) exactly as maxvolpy.maxvol.c_maxvol did.
    """
    cdef int N, r
    cdef cnp.ndarray lu, coef, basis
    if not isinstance(A, np.ndarray):
        raise TypeError("argument must be of numpy.ndarray type")
    if A.ndim != 2:
        raise ValueError("argument must have 2 dimensions")
    if A.dtype != np.dtype(np.float64):
        raise TypeError("argument must be of float64 type, got {}".format(A.dtype))
    N, r = A.shape
    if N <= r:
        return np.arange(N, dtype=np.int32), np.eye(N, dtype=A.dtype)
    if tol < 1:
        tol = 1.0
    lu = np.copy(A, order='F')
    coef = np.copy(lu, order='F')
    basis = np.ndarray(r, dtype=np.int32)
    dmaxvol(N, r, <double *>lu.data, <double *>coef.data,
            <int *>basis.data, tol, max_iters, top_k_index, verbose)
    return basis, coef


cdef object dmaxvol(int N, int R, double *lu, double *coef, int *basis,
        double tol, int max_iters, int top_k_index, int verbose=0):
    cdef int *ipiv = <int *> malloc(R * sizeof(int))
    cdef int *interchange = <int *> malloc(N * sizeof(int))
    cdef double *tmp_row = <double *> malloc(R * sizeof(double))
    cdef double *tmp_column = <double *> malloc(N * sizeof(double))
    cdef int info = 0, i, j, tmp_int, i_one = 1, iters = 0
    cdef int k_row, k_col
    cdef char cR = b'R', cN = b'N', cU = b'U', cL = b'L'
    cdef double d_one = 1, alpha, max_value
    cdef double abs_max, tmp
    try:
        if (ipiv == NULL or interchange == NULL or tmp_row == NULL or
                tmp_column == NULL):
            raise MemoryError("malloc failed to allocate temporary buffers")
        if top_k_index == -1 or top_k_index > N:
            top_k_index = N
        if top_k_index < R:
            top_k_index = R
        dgetrf(&top_k_index, &R, lu, &N, ipiv, &info)
        if info < 0:
            raise ValueError("Internal maxvol_fullrank error, {} argument of"
                             " dgetrf_ had illegal value".format(info))
        if info > 0:
            raise ValueError("Input matrix must not be singular")
        for i in range(N):
            interchange[i] = i
        for i in range(R):
            j = ipiv[i] - 1
            if j != i:
                tmp_int = interchange[i]
                interchange[i] = interchange[j]
                interchange[j] = tmp_int
        for i in range(R):
            basis[i] = interchange[i]
        dtrsm(&cR, &cU, &cN, &cN, &N, &R, &d_one, lu, &N, coef, &N)
        dtrsm(&cR, &cL, &cN, &cU, &N, &R, &d_one, lu, &N, coef, &N)
        while iters < max_iters:
            abs_max = -1
            for k_row in range(top_k_index):
                for k_col in range(R):
                    tmp = fabs(coef[k_row + k_col * N])
                    if tmp > abs_max:
                        abs_max = tmp
                        j = k_row
                        i = k_col
            max_value = coef[j + i * N]
            if verbose and iters % 10 == 0:
                print('Iter {}/{}: abs_max = {:.3f} (tol = {:.3f})'.format(
                    iters, max_iters, abs_max, tol))
            if abs_max > tol:
                dcopy(&R, coef + j, &N, tmp_row, &i_one)
                tmp_row[i] -= d_one
                dcopy(&N, coef + i * N, &i_one, tmp_column, &i_one)
                basis[i] = j
                alpha = (-d_one) / max_value
                dger(&N, &R, &alpha, tmp_column, &i_one, tmp_row, &i_one,
                     coef, &N)
                iters += i_one
            else:
                break
    finally:
        free(ipiv)
        free(interchange)
        free(tmp_row)
        free(tmp_column)
    return

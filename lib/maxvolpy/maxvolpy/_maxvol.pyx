
#cython: embedsignature=True, cdivision=True, boundscheck=False, wraparound=False, initializedcheck=False
__all__ = ['c_maxvol', 'c_rect_maxvol']
import numpy as np
cimport numpy as cnp
from libc.stdlib cimport malloc, realloc, free
from cython.parallel import prange

from scipy.linalg.cython_blas cimport (
        strsm, dtrsm, ctrsm, ztrsm,
        scopy, ccopy, dcopy, zcopy,
        sgemv, dgemv, cgemv, zgemv,
        sger, dger, cgerc, cgeru, zgerc, zgeru,
        isamax, idamax
)

from scipy.linalg.cython_lapack cimport (
    sgetrf, dgetrf, cgetrf, zgetrf
)

cdef extern from "complex.h" nogil:
    double cabs(double complex)
    float cabsf(float complex)

cdef extern from "math.h" nogil:
    double fabs(double)
    float fabsf(float)

def c_rect_maxvol(A, tol=1., maxK=None, min_add_K=None, minK=None,
        start_maxvol_iters=10, identity_submatrix=True, top_k_index=-1):
    """
    Cython implementation of rectangular 2-volume maximization.
    
    For information see `rect_maxvol` function.
    """
    cdef int N, r, id_sub
    cdef cnp.ndarray lu, coef, basis
    if type(A) != np.ndarray:
        raise TypeError, "argument must be of numpy.ndarray type"
    if len(A.shape) != 2:
        raise ValueError, "argument must have 2 dimensions"
    N, r = A.shape
    if N <= r:
        return np.arange(N, dtype=np.int32), np.eye(N, dtype=A.dtype)
    lu = np.copy(A, order='F')
    if maxK is None or maxK > N:
        maxK = N
    if maxK < r:
        maxK = r
    if minK is None or minK < r:
        minK = r
    if minK > N:
        minK = N
    if min_add_K is not None:
        minK = max(minK, r+min_add_K)
    if minK > maxK:
        minK = maxK
    if identity_submatrix:
        id_sub = 1
    else:
        id_sub = 0
    try:
        if A.dtype is np.dtype(np.float32):
            return srect_maxvol(N, r, <float *>lu.data, tol, minK, maxK,
                start_maxvol_iters, id_sub, top_k_index)
        elif A.dtype is np.dtype(np.float64):
            return drect_maxvol(N, r, <double *>lu.data, tol, minK, maxK,
                start_maxvol_iters, id_sub, top_k_index)
        elif A.dtype is np.dtype(np.complex64):
            return crect_maxvol(N, r, <float complex *>lu.data, tol, minK,
                maxK, start_maxvol_iters, id_sub, top_k_index)
        elif A.dtype is np.dtype(np.complex128):
            return zrect_maxvol(N, r, <double complex*>lu.data, tol, minK,
                maxK, start_maxvol_iters, id_sub, top_k_index)
    except Exception:
        raise

def c_maxvol(A, tol=1.05, max_iters=100, top_k_index=-1):
    """
    Cython implementation of 1-volume maximization.
    
    For information see `maxvol` function.
    """
    cdef int N, r
    cdef cnp.ndarray lu, coef, basis
    if type(A) != np.ndarray:
        raise TypeError, "argument must be of numpy.ndarray type"
    if len(A.shape) != 2:
        raise ValueError, "argument must have 2 dimensions"
    N, r = A.shape
    if N <= r:
        return np.arange(N, dtype=np.int32), np.eye(N, dtype=A.dtype)
    if tol < 1:
        tol = 1.0
    lu = np.copy(A, order='F')
    coef = np.copy(lu, order='F')
    basis = np.ndarray(r, dtype=np.int32)
    try:
        if A.dtype is np.dtype(np.float32):
            smaxvol(N, r, <float *>lu.data, <float *>coef.data,
                <int *>basis.data, tol, max_iters, top_k_index)
        elif A.dtype == np.dtype(np.float64):
            dmaxvol(N, r, <double *>lu.data, <double *>coef.data,
                <int *>basis.data, tol, max_iters, top_k_index)
        elif A.dtype is np.dtype(np.complex64):
            cmaxvol(N, r, <float complex *>lu.data, <float complex *>
                coef.data, <int *>basis.data, tol, max_iters, top_k_index)
        elif A.dtype is np.dtype(np.complex128):
            zmaxvol(N, r, <double complex*>lu.data, <double complex *>
                coef.data, <int *>basis.data, tol, max_iters, top_k_index)
        else:
            raise TypeError("must be of float or complex type")
    except Exception:
        raise
    return basis, coef


cdef object srect_maxvol(int N, int R, float *lu, float tol, int minK,
        int maxK, int start_maxvol_iters, int identity_submatrix,
        int top_k_index):
    cdef char cN = 'N'
    cdef int i, j, i_one = 1, K, size = N*R
    cdef float d_one = 1.0, d_zero = 0.0, l
    cdef float tol2 = tol*tol, tmp, tmp2
    cdef int *basis = <int *> malloc(N * sizeof(int))
    cdef float *chosen = <float *> malloc(N * sizeof(float))
    cdef int [:]basis_buf
    cdef int coef_realloc_step = R, coef_columns = R+coef_realloc_step
    cdef float *coef = <float *> malloc(N * coef_columns * sizeof(float))
    cdef float *tmp_pointer
    cdef float *L = <float *> malloc(N * sizeof(float))
    cdef float *V = <float *> malloc(N * sizeof(float))
    cdef float *tmp_row = <float *> malloc(N * sizeof(float))
    cdef float [:,:] coef_buf
    if top_k_index == -1 or top_k_index > N:
        top_k_index = N
    if top_k_index < R:
        top_k_index = R
    scopy(&size, lu, &i_one, coef, &i_one)
    tmp = 1.05 # tolerance for square maxvol
    smaxvol(N, R, lu, coef, basis, tmp, start_maxvol_iters, top_k_index)
    # compute square length for each vector
    for j in prange(top_k_index, schedule="static", nogil=True):
        L[j] = 0.0
        V[j] = 0.0
        chosen[j] = 1.0
    for i in range(R):
        tmp_pointer = coef+i*N
        for j in prange(top_k_index, schedule="static", nogil=True):
            tmp = fabsf(tmp_pointer[j])
            L[j] += tmp*tmp
    for i in prange(R, schedule="static", nogil=True):
        L[basis[i]] = 0.0
        chosen[basis[i]] = 0.0
    i = isamax(&top_k_index, L, &i_one)-1
    K = R
    while K < minK or (L[i] > tol2 and K < maxK):
        basis[K] = i
        chosen[i] = 0.0
        #scopy(&K, coef+i, &N, tmp_row, &i_one)
        tmp_pointer = coef+i
        for j in prange(K, schedule="static", nogil=True):
            tmp_row[j] = tmp_pointer[j*N]
        sgemv(&cN, &N, &K, &d_one, coef, &N, tmp_row, &i_one, &d_zero, V,
            &i_one)
        l = (-d_one)/(1+V[i])
        sger(&N, &K, &l, V, &i_one, tmp_row, &i_one, coef, &N)
        tmp = -l
        if coef_columns <= K:
            coef_columns += coef_realloc_step
            coef = <float *> realloc(coef, N * coef_columns * sizeof(float))
        tmp_pointer = coef+K*N
        for j in prange(N, schedule="static", nogil=True):
            tmp_pointer[j] = tmp*V[j]
        for j in prange(top_k_index, schedule="static", nogil=True):
            tmp2 = fabsf(V[j])
            L[j] -= tmp2*tmp2*tmp
            L[j] *= chosen[j]
        i = isamax(&top_k_index, L, &i_one)-1
        K += 1
    free(L)
    free(V)
    free(tmp_row)
    C = np.ndarray((N, K), order='F', dtype=np.float32)
    coef_buf = C
    for i in prange(K, schedule="static", nogil=True):
        for j in range(N):
            coef_buf[j, i] = coef[i*N+j]
    free(coef)
    if identity_submatrix == 1:
        for i in prange(K, schedule="static", nogil=True):
            tmp_pointer = &coef_buf[0, 0]+basis[i]
            for j in range(K):
                tmp_pointer[j*N] = 0.0
            tmp_pointer[i*N] = 1.0
    I = np.ndarray(K, dtype=np.int32)
    basis_buf = I
    for i in prange(K, schedule="static", nogil=True):
        basis_buf[i] = basis[i]
    free(basis)
    return I, C

cdef object smaxvol(int N, int R, float *lu, float *coef, int *basis,
        float tol, int max_iters, int top_k_index):
    cdef int *ipiv = <int *> malloc(R * sizeof(int))
    cdef int *interchange = <int *> malloc(N * sizeof(int))
    cdef float *tmp_row = <float *> malloc(R*sizeof(float))
    cdef float *tmp_column = <float *> malloc(N*sizeof(float))
    cdef int info = 0, size = N * R, i, j, tmp_int, i_one = 1, iters = 0
    cdef int k_row, k_col
    cdef char cR = 'R', cN = 'N', cU = 'U', cL = 'L'
    cdef float d_one = 1, alpha, max_value
    cdef float abs_max, tmp
    if (ipiv == NULL or interchange == NULL or tmp_row == NULL or
            tmp_column == NULL):
        raise MemoryError("malloc failed to allocate temporary buffers")
    if top_k_index == -1 or top_k_index > N:
        top_k_index = N
    if top_k_index < R:
        top_k_index = R
    sgetrf(&top_k_index, &R, lu, &N, ipiv, &info)
    if info < 0:
        raise ValueError("Internal maxvol_fullrank error, {} argument of"
            " sgetrf_ had illegal value".format(info))
    if info > 0:
        raise ValueError("Input matrix must not be singular")
    for i in prange(N, schedule="static", nogil=True):
        interchange[i] = i
    for i in prange(R, schedule="static", nogil=True):
        j = ipiv[i]-1
        if j != i:
            tmp_int = interchange[i]
            interchange[i] = interchange[j]
            interchange[j] = tmp_int
    free(ipiv)
    for i in prange(R, schedule="static", nogil=True):
        basis[i] = interchange[i]
    free(interchange)
    strsm(&cR, &cU, &cN, &cN, &N, &R, &d_one, lu, &N, coef, &N)
    strsm(&cR, &cL, &cN, &cU, &N, &R, &d_one, lu, &N, coef, &N)
    while iters < max_iters:
        abs_max = -1
        for k_row in range(top_k_index):
            for k_col in range(R):
                tmp = fabsf(coef[k_row+k_col*N])
                if tmp > abs_max:
                    abs_max = tmp
                    j = k_row
                    i = k_col
        max_value = coef[j+i*N]
        if iters % 10 == 0:
            print('Iter {}/{}: abs_max = {} (tol = {})'.format(iters, max_iters, abs_max, tol))
        if abs_max > tol:
            scopy(&R, coef+j, &N, tmp_row, &i_one)
            tmp_row[i] -= d_one
            scopy(&N, coef+i*N, &i_one, tmp_column, &i_one)
            basis[i] = j
            alpha = (-d_one)/max_value
            sger(&N, &R, &alpha, tmp_column, &i_one, tmp_row, &i_one,
                coef, &N)
            iters += i_one
        else:
            break
    free(tmp_row)
    free(tmp_column)
    return

cdef object drect_maxvol(int N, int R, double *lu, double tol, int minK,
        int maxK, int start_maxvol_iters, int identity_submatrix,
        int top_k_index):
    cdef char cN = 'N'
    cdef int i, j, i_one = 1, K, size = N*R
    cdef double d_one = 1.0, d_zero = 0.0, l
    cdef double tol2 = tol*tol, tmp, tmp2
    cdef int *basis = <int *> malloc(N * sizeof(int))
    cdef double *chosen = <double *> malloc(N * sizeof(double))
    cdef int [:]basis_buf
    cdef int coef_realloc_step = R, coef_columns = R+coef_realloc_step
    cdef double *coef = <double *> malloc(N * coef_columns * sizeof(double))
    cdef double *tmp_pointer
    cdef double *L = <double *> malloc(N * sizeof(double))
    cdef double *V = <double *> malloc(N * sizeof(double))
    cdef double *tmp_row = <double *> malloc(N * sizeof(double))
    cdef double [:,:] coef_buf
    if top_k_index == -1 or top_k_index > N:
        top_k_index = N
    if top_k_index < R:
        top_k_index = R
    dcopy(&size, lu, &i_one, coef, &i_one)
    tmp = 1.05 # tolerance for square maxvol
    dmaxvol(N, R, lu, coef, basis, tmp, start_maxvol_iters, top_k_index)
    # compute square length for each vector
    for j in prange(top_k_index, schedule="static", nogil=True):
        L[j] = 0.0
        V[j] = 0.0
        chosen[j] = 1.0
    for i in range(R):
        tmp_pointer = coef+i*N
        for j in prange(top_k_index, schedule="static", nogil=True):
            tmp = fabs(tmp_pointer[j])
            L[j] += tmp*tmp
    for i in prange(R, schedule="static", nogil=True):
        L[basis[i]] = 0.0
        chosen[basis[i]] = 0.0
    i = idamax(&top_k_index, L, &i_one)-1
    K = R
    while K < minK or (L[i] > tol2 and K < maxK):
        basis[K] = i
        chosen[i] = 0.0
        #dcopy(&K, coef+i, &N, tmp_row, &i_one)
        tmp_pointer = coef+i
        for j in prange(K, schedule="static", nogil=True):
            tmp_row[j] = tmp_pointer[j*N]
        dgemv(&cN, &N, &K, &d_one, coef, &N, tmp_row, &i_one, &d_zero, V,
            &i_one)
        l = (-d_one)/(1+V[i])
        dger(&N, &K, &l, V, &i_one, tmp_row, &i_one, coef, &N)
        tmp = -l
        if coef_columns <= K:
            coef_columns += coef_realloc_step
            coef = <double *> realloc(coef, N * coef_columns * sizeof(double))
        tmp_pointer = coef+K*N
        for j in prange(N, schedule="static", nogil=True):
            tmp_pointer[j] = tmp*V[j]
        for j in prange(top_k_index, schedule="static", nogil=True):
            tmp2 = fabs(V[j])
            L[j] -= tmp2*tmp2*tmp
            L[j] *= chosen[j]
        i = idamax(&top_k_index, L, &i_one)-1
        K += 1
    free(L)
    free(V)
    free(tmp_row)
    C = np.ndarray((N, K), order='F', dtype=np.float64)
    coef_buf = C
    for i in prange(K, schedule="static", nogil=True):
        for j in range(N):
            coef_buf[j, i] = coef[i*N+j]
    free(coef)
    if identity_submatrix == 1:
        for i in prange(K, schedule="static", nogil=True):
            tmp_pointer = &coef_buf[0, 0]+basis[i]
            for j in range(K):
                tmp_pointer[j*N] = 0.0
            tmp_pointer[i*N] = 1.0
    I = np.ndarray(K, dtype=np.int32)
    basis_buf = I
    for i in prange(K, schedule="static", nogil=True):
        basis_buf[i] = basis[i]
    free(basis)
    return I, C

cdef object dmaxvol(int N, int R, double *lu, double *coef, int *basis,
        double tol, int max_iters, int top_k_index):
    cdef int *ipiv = <int *> malloc(R * sizeof(int))
    cdef int *interchange = <int *> malloc(N * sizeof(int))
    cdef double *tmp_row = <double *> malloc(R*sizeof(double))
    cdef double *tmp_column = <double *> malloc(N*sizeof(double))
    cdef int info = 0, size = N * R, i, j, tmp_int, i_one = 1, iters = 0
    cdef int k_row, k_col
    cdef char cR = 'R', cN = 'N', cU = 'U', cL = 'L'
    cdef double d_one = 1, alpha, max_value
    cdef double abs_max, tmp
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
    for i in prange(N, schedule="static", nogil=True):
        interchange[i] = i
    for i in prange(R, schedule="static", nogil=True):
        j = ipiv[i]-1
        if j != i:
            tmp_int = interchange[i]
            interchange[i] = interchange[j]
            interchange[j] = tmp_int
    free(ipiv)
    for i in prange(R, schedule="static", nogil=True):
        basis[i] = interchange[i]
    free(interchange)
    dtrsm(&cR, &cU, &cN, &cN, &N, &R, &d_one, lu, &N, coef, &N)
    dtrsm(&cR, &cL, &cN, &cU, &N, &R, &d_one, lu, &N, coef, &N)
    while iters < max_iters:
        abs_max = -1
        for k_row in range(top_k_index):
            for k_col in range(R):
                tmp = fabs(coef[k_row+k_col*N])
                if tmp > abs_max:
                    abs_max = tmp
                    j = k_row
                    i = k_col
        max_value = coef[j+i*N]
        if iters % 10 == 0:
            print('Iter {}/{}: abs_max = {} (tol = {})'.format(iters, max_iters, abs_max, tol))
        if abs_max > tol:
            dcopy(&R, coef+j, &N, tmp_row, &i_one)
            tmp_row[i] -= d_one
            dcopy(&N, coef+i*N, &i_one, tmp_column, &i_one)
            basis[i] = j
            alpha = (-d_one)/max_value
            dger(&N, &R, &alpha, tmp_column, &i_one, tmp_row, &i_one,
                coef, &N)
            iters += i_one
        else:
            break
    free(tmp_row)
    free(tmp_column)
    return

cdef object crect_maxvol(int N, int R, float complex *lu, float tol, int minK,
        int maxK, int start_maxvol_iters, int identity_submatrix,
        int top_k_index):
    cdef char cN = 'N'
    cdef int i, j, i_one = 1, K, size = N*R
    cdef float complex d_one = 1.0, d_zero = 0.0, l
    cdef float tol2 = tol*tol, tmp, tmp2
    cdef int *basis = <int *> malloc(N * sizeof(int))
    cdef float *chosen = <float *> malloc(N * sizeof(float))
    cdef int [:]basis_buf
    cdef int coef_realloc_step = R, coef_columns = R+coef_realloc_step
    cdef float complex *coef = <float complex *> malloc(N * coef_columns * sizeof(float complex))
    cdef float complex *tmp_pointer
    cdef float *L = <float *> malloc(N * sizeof(float))
    cdef float complex *V = <float complex *> malloc(N * sizeof(float complex))
    cdef float complex *tmp_row = <float complex *> malloc(N * sizeof(float complex))
    cdef float complex [:,:] coef_buf
    if top_k_index == -1 or top_k_index > N:
        top_k_index = N
    if top_k_index < R:
        top_k_index = R
    ccopy(&size, lu, &i_one, coef, &i_one)
    tmp = 1.05 # tolerance for square maxvol
    cmaxvol(N, R, lu, coef, basis, tmp, start_maxvol_iters, top_k_index)
    # compute square length for each vector
    for j in prange(top_k_index, schedule="static", nogil=True):
        L[j] = 0.0
        V[j] = 0.0
        chosen[j] = 1.0
    for i in range(R):
        tmp_pointer = coef+i*N
        for j in prange(top_k_index, schedule="static", nogil=True):
            tmp = cabsf(tmp_pointer[j])
            L[j] += tmp*tmp
    for i in prange(R, schedule="static", nogil=True):
        L[basis[i]] = 0.0
        chosen[basis[i]] = 0.0
    i = isamax(&top_k_index, L, &i_one)-1
    K = R
    while K < minK or (L[i] > tol2 and K < maxK):
        basis[K] = i
        chosen[i] = 0.0
        #ccopy(&K, coef+i, &N, tmp_row, &i_one)
        tmp_pointer = coef+i
        for j in prange(K, schedule="static", nogil=True):
            tmp_row[j] = tmp_pointer[j*N].conjugate()
        cgemv(&cN, &N, &K, &d_one, coef, &N, tmp_row, &i_one, &d_zero, V,
            &i_one)
        l = (-d_one)/(1+V[i])
        cgerc(&N, &K, &l, V, &i_one, tmp_row, &i_one, coef, &N)
        tmp = -l.real
        if coef_columns <= K:
            coef_columns += coef_realloc_step
            coef = <float complex *> realloc(coef, N * coef_columns * sizeof(float complex))
        tmp_pointer = coef+K*N
        for j in prange(N, schedule="static", nogil=True):
            tmp_pointer[j] = tmp*V[j]
        for j in prange(top_k_index, schedule="static", nogil=True):
            tmp2 = cabsf(V[j])
            L[j] -= tmp2*tmp2*tmp
            L[j] *= chosen[j]
        i = isamax(&top_k_index, L, &i_one)-1
        K += 1
    free(L)
    free(V)
    free(tmp_row)
    C = np.ndarray((N, K), order='F', dtype=np.complex64)
    coef_buf = C
    for i in prange(K, schedule="static", nogil=True):
        for j in range(N):
            coef_buf[j, i] = coef[i*N+j]
    free(coef)
    if identity_submatrix == 1:
        for i in prange(K, schedule="static", nogil=True):
            tmp_pointer = &coef_buf[0, 0]+basis[i]
            for j in range(K):
                tmp_pointer[j*N] = 0.0
            tmp_pointer[i*N] = 1.0
    I = np.ndarray(K, dtype=np.int32)
    basis_buf = I
    for i in prange(K, schedule="static", nogil=True):
        basis_buf[i] = basis[i]
    free(basis)
    return I, C

cdef object cmaxvol(int N, int R, float complex *lu, float complex *coef, int *basis,
        float tol, int max_iters, int top_k_index):
    cdef int *ipiv = <int *> malloc(R * sizeof(int))
    cdef int *interchange = <int *> malloc(N * sizeof(int))
    cdef float complex *tmp_row = <float complex *> malloc(R*sizeof(float complex))
    cdef float complex *tmp_column = <float complex *> malloc(N*sizeof(float complex))
    cdef int info = 0, size = N * R, i, j, tmp_int, i_one = 1, iters = 0
    cdef int k_row, k_col
    cdef char cR = 'R', cN = 'N', cU = 'U', cL = 'L'
    cdef float complex d_one = 1, alpha, max_value
    cdef float abs_max, tmp
    if (ipiv == NULL or interchange == NULL or tmp_row == NULL or
            tmp_column == NULL):
        raise MemoryError("malloc failed to allocate temporary buffers")
    if top_k_index == -1 or top_k_index > N:
        top_k_index = N
    if top_k_index < R:
        top_k_index = R
    cgetrf(&top_k_index, &R, lu, &N, ipiv, &info)
    if info < 0:
        raise ValueError("Internal maxvol_fullrank error, {} argument of"
            " cgetrf_ had illegal value".format(info))
    if info > 0:
        raise ValueError("Input matrix must not be singular")
    for i in prange(N, schedule="static", nogil=True):
        interchange[i] = i
    for i in prange(R, schedule="static", nogil=True):
        j = ipiv[i]-1
        if j != i:
            tmp_int = interchange[i]
            interchange[i] = interchange[j]
            interchange[j] = tmp_int
    free(ipiv)
    for i in prange(R, schedule="static", nogil=True):
        basis[i] = interchange[i]
    free(interchange)
    ctrsm(&cR, &cU, &cN, &cN, &N, &R, &d_one, lu, &N, coef, &N)
    ctrsm(&cR, &cL, &cN, &cU, &N, &R, &d_one, lu, &N, coef, &N)
    while iters < max_iters:
        abs_max = -1
        for k_row in range(top_k_index):
            for k_col in range(R):
                tmp = cabsf(coef[k_row+k_col*N])
                if tmp > abs_max:
                    abs_max = tmp
                    j = k_row
                    i = k_col
        max_value = coef[j+i*N]
        if iters % 10 == 0:
            print('Iter {}/{}: abs_max = {} (tol = {})'.format(iters, max_iters, abs_max, tol))
        if abs_max > tol:
            ccopy(&R, coef+j, &N, tmp_row, &i_one)
            tmp_row[i] -= d_one
            ccopy(&N, coef+i*N, &i_one, tmp_column, &i_one)
            basis[i] = j
            alpha = (-d_one)/max_value
            cgeru(&N, &R, &alpha, tmp_column, &i_one, tmp_row, &i_one,
                coef, &N)
            iters += i_one
        else:
            break
    free(tmp_row)
    free(tmp_column)
    return

cdef object zrect_maxvol(int N, int R, double complex *lu, double tol, int minK,
        int maxK, int start_maxvol_iters, int identity_submatrix,
        int top_k_index):
    cdef char cN = 'N'
    cdef int i, j, i_one = 1, K, size = N*R
    cdef double complex d_one = 1.0, d_zero = 0.0, l
    cdef double tol2 = tol*tol, tmp, tmp2
    cdef int *basis = <int *> malloc(N * sizeof(int))
    cdef double *chosen = <double *> malloc(N * sizeof(double))
    cdef int [:]basis_buf
    cdef int coef_realloc_step = R, coef_columns = R+coef_realloc_step
    cdef double complex *coef = <double complex *> malloc(N * coef_columns * sizeof(double complex))
    cdef double complex *tmp_pointer
    cdef double *L = <double *> malloc(N * sizeof(double))
    cdef double complex *V = <double complex *> malloc(N * sizeof(double complex))
    cdef double complex *tmp_row = <double complex *> malloc(N * sizeof(double complex))
    cdef double complex [:,:] coef_buf
    if top_k_index == -1 or top_k_index > N:
        top_k_index = N
    if top_k_index < R:
        top_k_index = R
    zcopy(&size, lu, &i_one, coef, &i_one)
    tmp = 1.05 # tolerance for square maxvol
    zmaxvol(N, R, lu, coef, basis, tmp, start_maxvol_iters, top_k_index)
    # compute square length for each vector
    for j in prange(top_k_index, schedule="static", nogil=True):
        L[j] = 0.0
        V[j] = 0.0
        chosen[j] = 1.0
    for i in range(R):
        tmp_pointer = coef+i*N
        for j in prange(top_k_index, schedule="static", nogil=True):
            tmp = cabs(tmp_pointer[j])
            L[j] += tmp*tmp
    for i in prange(R, schedule="static", nogil=True):
        L[basis[i]] = 0.0
        chosen[basis[i]] = 0.0
    i = idamax(&top_k_index, L, &i_one)-1
    K = R
    while K < minK or (L[i] > tol2 and K < maxK):
        basis[K] = i
        chosen[i] = 0.0
        #zcopy(&K, coef+i, &N, tmp_row, &i_one)
        tmp_pointer = coef+i
        for j in prange(K, schedule="static", nogil=True):
            tmp_row[j] = tmp_pointer[j*N].conjugate()
        zgemv(&cN, &N, &K, &d_one, coef, &N, tmp_row, &i_one, &d_zero, V,
            &i_one)
        l = (-d_one)/(1+V[i])
        zgerc(&N, &K, &l, V, &i_one, tmp_row, &i_one, coef, &N)
        tmp = -l.real
        if coef_columns <= K:
            coef_columns += coef_realloc_step
            coef = <double complex *> realloc(coef, N * coef_columns * sizeof(double complex))
        tmp_pointer = coef+K*N
        for j in prange(N, schedule="static", nogil=True):
            tmp_pointer[j] = tmp*V[j]
        for j in prange(top_k_index, schedule="static", nogil=True):
            tmp2 = cabs(V[j])
            L[j] -= tmp2*tmp2*tmp
            L[j] *= chosen[j]
        i = idamax(&top_k_index, L, &i_one)-1
        K += 1
    free(L)
    free(V)
    free(tmp_row)
    C = np.ndarray((N, K), order='F', dtype=np.complex128)
    coef_buf = C
    for i in prange(K, schedule="static", nogil=True):
        for j in range(N):
            coef_buf[j, i] = coef[i*N+j]
    free(coef)
    if identity_submatrix == 1:
        for i in prange(K, schedule="static", nogil=True):
            tmp_pointer = &coef_buf[0, 0]+basis[i]
            for j in range(K):
                tmp_pointer[j*N] = 0.0
            tmp_pointer[i*N] = 1.0
    I = np.ndarray(K, dtype=np.int32)
    basis_buf = I
    for i in prange(K, schedule="static", nogil=True):
        basis_buf[i] = basis[i]
    free(basis)
    return I, C

cdef object zmaxvol(int N, int R, double complex *lu, double complex *coef, int *basis,
        double tol, int max_iters, int top_k_index):
    cdef int *ipiv = <int *> malloc(R * sizeof(int))
    cdef int *interchange = <int *> malloc(N * sizeof(int))
    cdef double complex *tmp_row = <double complex *> malloc(R*sizeof(double complex))
    cdef double complex *tmp_column = <double complex *> malloc(N*sizeof(double complex))
    cdef int info = 0, size = N * R, i, j, tmp_int, i_one = 1, iters = 0
    cdef int k_row, k_col
    cdef char cR = 'R', cN = 'N', cU = 'U', cL = 'L'
    cdef double complex d_one = 1, alpha, max_value
    cdef double abs_max, tmp
    if (ipiv == NULL or interchange == NULL or tmp_row == NULL or
            tmp_column == NULL):
        raise MemoryError("malloc failed to allocate temporary buffers")
    if top_k_index == -1 or top_k_index > N:
        top_k_index = N
    if top_k_index < R:
        top_k_index = R
    zgetrf(&top_k_index, &R, lu, &N, ipiv, &info)
    if info < 0:
        raise ValueError("Internal maxvol_fullrank error, {} argument of"
            " zgetrf_ had illegal value".format(info))
    if info > 0:
        raise ValueError("Input matrix must not be singular")
    for i in prange(N, schedule="static", nogil=True):
        interchange[i] = i
    for i in prange(R, schedule="static", nogil=True):
        j = ipiv[i]-1
        if j != i:
            tmp_int = interchange[i]
            interchange[i] = interchange[j]
            interchange[j] = tmp_int
    free(ipiv)
    for i in prange(R, schedule="static", nogil=True):
        basis[i] = interchange[i]
    free(interchange)
    ztrsm(&cR, &cU, &cN, &cN, &N, &R, &d_one, lu, &N, coef, &N)
    ztrsm(&cR, &cL, &cN, &cU, &N, &R, &d_one, lu, &N, coef, &N)
    while iters < max_iters:
        abs_max = -1
        for k_row in range(top_k_index):
            for k_col in range(R):
                tmp = cabs(coef[k_row+k_col*N])
                if tmp > abs_max:
                    abs_max = tmp
                    j = k_row
                    i = k_col
        max_value = coef[j+i*N]
        if iters % 10 == 0:
            print('Iter {}/{}: abs_max = {} (tol = {})'.format(iters, max_iters, abs_max, tol))
        if abs_max > tol:
            zcopy(&R, coef+j, &N, tmp_row, &i_one)
            tmp_row[i] -= d_one
            zcopy(&N, coef+i*N, &i_one, tmp_column, &i_one)
            basis[i] = j
            alpha = (-d_one)/max_value
            zgeru(&N, &R, &alpha, tmp_column, &i_one, tmp_row, &i_one,
                coef, &N)
            iters += i_one
        else:
            break
    free(tmp_row)
    free(tmp_column)
    return

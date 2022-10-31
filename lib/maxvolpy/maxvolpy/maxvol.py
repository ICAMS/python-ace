"""
This submodule contains routines to find good submatrices. How good
matrix is depends on special extreme properties of the matrix. Two of
this properties are 1-volume and 2-volume with the following formulas:

:math:`vol_1(A) = \\left|\\det(A)\\right|,\\, vol_2(A) =\
    \\sqrt{\\max(\\det(A^HA), \\det(AA^H))}`
"""

from __future__ import absolute_import, division, print_function

__all__ = ['rect_maxvol', 'maxvol', 'rect_maxvol_svd', 'maxvol_svd',
    'rect_maxvol_qr', 'maxvol_qr']

from .misc import svd_cut

def py_rect_maxvol(A, tol=1., maxK=None, min_add_K=None, minK=None,
        start_maxvol_iters=10, identity_submatrix=True, top_k_index=-1):
    """
    Python implementation of rectangular 2-volume maximization.

    See Also
    --------
    rect_maxvol
    """
    # tol2 - square of parameter tol
    tol2 = tol**2
    # N - number of rows, r - number of columns of matrix A
    N, r = A.shape
    # some work on parameters
    if N <= r:
        return np.arange(N, dtype=np.int32), np.eye(N, dtype=A.dtype)
    if maxK is None or maxK > N:
        maxK = N
    if maxK < r:
        maxK = r
    if minK is None or minK < r:
        minK = r
    if minK > N:
        minK = N
    if min_add_K is not None:
        minK = max(minK, r + min_add_K) 
    if minK > maxK:
        minK = maxK
        #raise ValueError('minK value cannot be greater than maxK value')
    if top_k_index == -1 or top_k_index > N:
        top_k_index = N
    if top_k_index < r:
        top_k_index = r
    # choose initial submatrix and coefficients according to maxvol
    # algorithm
    index = np.zeros(N, dtype=np.int32)
    chosen = np.ones(top_k_index)
    tmp_index, C = py_maxvol(A, 1.05, start_maxvol_iters, top_k_index)
    index[:r] = tmp_index
    chosen[tmp_index] = 0
    C = np.asfortranarray(C)
    # compute square 2-norms of each row in coefficients matrix C
    row_norm_sqr = np.array([chosen[i]*np.linalg.norm(C[i], 2)**2 for
        i in range(top_k_index)])
    # find maximum value in row_norm_sqr
    i = np.argmax(row_norm_sqr)
    K = r
    # set cgeru or zgeru for complex numbers and dger or sger
    # for float numbers
    try:
        ger = get_blas_funcs('geru', [C])
    except:
        ger = get_blas_funcs('ger', [C])
    # augment maxvol submatrix with each iteration
    while (row_norm_sqr[i] > tol2 and K < maxK) or K < minK:
        # add i to index and recompute C and square norms of each row
        # by SVM-formula
        index[K] = i
        chosen[i] = 0
        c = C[i].copy()
        v = C.dot(c.conj())
        l = 1.0/(1+v[i])
        ger(-l,v,c,a=C,overwrite_a=1)
        C = np.hstack([C, l*v.reshape(-1,1)])
        row_norm_sqr -= (l*v[:top_k_index]*v[:top_k_index].conj()).real
        row_norm_sqr *= chosen
        # find maximum value in row_norm_sqr
        i = row_norm_sqr.argmax()
        K += 1
    # parameter identity_submatrix is True, set submatrix,
    # corresponding to maxvol rows, equal to identity matrix
    if identity_submatrix:
        C[index[:K]] = np.eye(K, dtype=C.dtype)
    return index[:K].copy(), C

def py_maxvol(A, tol=1.05, max_iters=100, top_k_index=-1):
    """
    Python implementation of 1-volume maximization.

    See Also
    --------
    maxvol
    """
    # some work on parameters
    if tol < 1:
        tol = 1.0
    N, r = A.shape
    if N <= r:
        return np.arange(N, dtype=np.int32), np.eye(N, dtype=A.dtype)
    if top_k_index == -1 or top_k_index > N:
        top_k_index = N
    if top_k_index < r:
        top_k_index = r
    # set auxiliary matrices and get corresponding *GETRF function
    # from lapack
    B = np.copy(A[:top_k_index], order='F')
    C = np.copy(A.T, order='F')
    H, ipiv, info = get_lapack_funcs('getrf', [B])(B, overwrite_a=1)
    # compute pivots from ipiv (result of *GETRF)
    index = np.arange(N, dtype=np.int32)
    for i in range(r):
        tmp = index[i]
        index[i] = index[ipiv[i]]
        index[ipiv[i]] = tmp
    # solve A = CH, H is in LU format
    B = H[:r]
    # It will be much faster to use *TRSM instead of *TRTRS
    trtrs = get_lapack_funcs('trtrs', [B])
    trtrs(B, C, trans=1, lower=0, unitdiag=0, overwrite_b=1)
    trtrs(B, C, trans=1, lower=1, unitdiag=1, overwrite_b=1)
    # C has shape (r, N) -- it is stored transposed
    # find max value in C
    i, j = divmod(abs(C[:,:top_k_index]).argmax(), top_k_index)
    # set cgeru or zgeru for complex numbers and dger or sger for
    # float numbers
    try:
        ger = get_blas_funcs('geru', [C])
    except:
        ger = get_blas_funcs('ger', [C])
    # set number of iters to 0
    iters = 0
    # check if need to swap rows
    while abs(C[i,j]) > tol and iters < max_iters:
        # add j to index and recompute C by SVM-formula
        index[i] = j
        tmp_row = C[i].copy()
        tmp_column = C[:,j].copy()
        tmp_column[i] -= 1.
        alpha = -1./C[i,j]
        ger(alpha, tmp_column, tmp_row, a=C, overwrite_a=1)
        iters += 1
        i, j = divmod(abs(C[:,:top_k_index]).argmax(), top_k_index)
    return index[:r].copy(), C.T

def rect_maxvol(A, tol=1., maxK=None, min_add_K=None, minK=None,
        start_maxvol_iters=10, identity_submatrix=True, top_k_index=-1):
    """
    Finds good rectangular submatrix.

    Uses greedy iterative maximization of 2-volume to find good
    `K`-by-`r` submatrix in a given `N`-by-`r` matrix `A` of rank `r`.
    Returns good submatrix and least squares coefficients of expansion
    (`N`-by-`K` matrix) of rows of matrix `A` by rows of good submatrix.

    Parameters
    ----------
    A : numpy.ndarray(ndim=2)
        Real or complex matrix of shape `(N, r)`, `N >= r`.
    tol : float, optional
        Upper bound for euclidian norm of coefficients of expansion of
        rows of `A` by rows of good submatrix. Defaults to `1.0`.
    maxK : integer, optional
        Maximum number of rows in good submatrix. Defaults to `N` if
        not set explicitly.
    minK : integer, optional
        Minimum number of rows in good submatrix. Defaults to `r` if
        not set explicitly.
    min_add_K : integer, optional
        Minimum number of rows to add to the square submatrix.
        Resulting good matrix will have minimum of `r+min_add_K` rows.
        Ignored if not set explicitly.
    start_maxvol_iters : integer, optional
        How many iterations of square maxvol (optimization of 1-volume)
        is required to be done before actual rectangular 2-volume
        maximization. Defaults to `10`.
    identity_submatrix : boolean, optional
        Coefficients of expansions are computed as least squares
        solution. If `identity_submatrix` is True, returned matrix of
        coefficients will have submatrix, corresponding to good rows,
        set to identity. Defaults to `True`.
    top_k_index : integer, optional
        Pivot rows for good submatrix will be in range from `0` to
        `(top_k_index-1)`. This restriction is ignored, if `top_k_index`
        is -1. Defaults to `-1`.

    Returns
    -------
    piv : numpy.ndarray(ndim=1, dtype=numpy.int32)
        Rows of matrix `A`, corresponding to submatrix, good in terms
        of 2-volume. Shape is `(K, )`.
    C : numpy.ndarray(ndim=2)
        Matrix of coefficients of expansions of all rows of `A` by good
        rows `piv`. Shape is `(N, K)`.

    Examples
    --------
    >>> import numpy as np
    >>> from maxvolpy.maxvol import rect_maxvol
    >>> np.random.seed(100)
    >>> a = np.random.rand(1000, 30, 2).view(dtype=np.complex128)[:,:,0]
    >>> piv, C = rect_maxvol(a, 1.0)
    >>> np.allclose(a, C.dot(a[piv]))
    True
    >>> print('maximum euclidian norm of row in matrix C: {:.5f}'.
    ... format(max([np.linalg.norm(C[i], 2) for i in range(1000)])))
    maximum euclidian norm of row in matrix C: 1.00000
    >>> piv, C = rect_maxvol(a, 1.5)
    >>> np.allclose(a, C.dot(a[piv]))
    True
    >>> print('maximum euclidian norm of row in matrix C: {:.5f}'.
    ... format(max([np.linalg.norm(C[i], 2) for i in range(1000)])))
    maximum euclidian norm of row in matrix C: 1.49193
    >>> piv, C = rect_maxvol(a, 2.0)
    >>> np.allclose(a, C.dot(a[piv]))
    True
    >>> print('maximum euclidian norm of row in matrix C: {:.5f}'.
    ... format(max([np.linalg.norm(C[i], 2) for i in range(1000)])))
    maximum euclidian norm of row in matrix C: 1.91954
    """
    return rect_maxvol_func(A, tol, maxK, min_add_K, minK, start_maxvol_iters,
            identity_submatrix, top_k_index)

def maxvol(A, tol=1.05, max_iters=100, top_k_index=-1):
    """
    Finds good square submatrix.

    Uses greedy iterative maximization of 1-volume to find good
    `r`-by-`r` submatrix in a given `N`-by-`r` matrix `A` of rank `r`.
    Returns good submatrix and coefficients of expansion
    (`N`-by-`r` matrix) of rows of matrix `A` by rows of good submatrix.

    Parameters
    ----------
    A : numpy.ndarray(ndim=2)
        Real or complex matrix of shape `(N, r)`, `N >= r`.
    tol : float, optional
        Upper bound for infinite norm of coefficients of expansion of
        rows of `A` by rows of good submatrix. Minimum value is 1.
        Default to `1.05`.
    max_iters : integer, optional
        Maximum number of iterations. Each iteration swaps 2 rows.
        Defaults to `100`.
    top_k_index : integer, optional
        Pivot rows for good submatrix will be in range from `0` to
        `(top_k_index-1)`. This restriction is ignored, if `top_k_index`
        is -1. Defaults to `-1`.
    
    Returns
    -------
    piv : numpy.ndarray(ndim=1, dtype=numpy.int32)
        Rows of matrix `A`, corresponding to submatrix, good in terms
        of 1-volume. Shape is `(r, )`.
    C : numpy.ndarray(ndim=2)
        Matrix of coefficients of expansions of all rows of `A` by good
        rows `piv`. Shape is `(N, r)`.

    Examples
    --------
    >>> import numpy as np
    >>> from maxvolpy.maxvol import maxvol
    >>> np.random.seed(100)
    >>> a = np.random.rand(1000, 30, 2).view(dtype=np.complex128)[:,:,0]
    >>> piv, C = maxvol(a, 1.0)
    >>> np.allclose(a, C.dot(a[piv]))
    True
    >>> print('Chebyshev norm of matrix C: {:.5f}'.format(abs(C).max()))
    Chebyshev norm of matrix C: 1.00000
    >>> piv, C = maxvol(a, 1.05)
    >>> np.allclose(a, C.dot(a[piv]))
    True
    >>> print('Chebyshev norm of matrix C: {:.5f}'.format(abs(C).max()))
    Chebyshev norm of matrix C: 1.04641
    >>> piv, C = maxvol(a, 1.10)
    >>> np.allclose(a, C.dot(a[piv]))
    True
    >>> print('Chebyshev norm of matrix C: {:.5f}'.format(abs(C).max()))
    Chebyshev norm of matrix C: 1.07854
    """
    return maxvol_func(A, tol=tol, max_iters=max_iters,
            top_k_index=top_k_index)

def rect_maxvol_svd(A, svd_tol=1e-3, svd_alpha=0., tol=1., maxK=None,
        min_add_K=None, minK=None, start_maxvol_iters=10,
        identity_submatrix=True, job='F', top_k_index=-1):
    """
    Applies SVD truncation and finds good rectangular submatrix.

    Computes SVD for `top_k_index` rows and/or columns of given matrix
    `A`, increases singular values by regularizing parameter
    `svd_alpha`, cuts off singular values, lower than
    `svd_tol` (relatively, getting only highest singular vectors),
    projects rows and/or columns, starting from `top_k_index`, to space
    of first `top_k_index` rows and/or columns and runs `rect_maxvol`
    for left and/or right singular vectors.

    Parameters
    ----------
    A : numpy.ndarray(ndim=2)
        Real or complex matrix.
    svd_tol : float
        Cut-off singular values parameter.
    svd_alpha : float
        Regularizing parameter for `misc.svd_cut`.
    tol : float
        Upper bound for euclidian norm of coefficients of expansion of
        rows/columns of approximant of `A` by good rows/columns of
        approximant.
    maxK : integer
        Maximum number of rows/columns in good submatrix.
    minK : integer
        Minimum number of rows/columns in good submatrix.
    min_add_K : integer
        Minimum number of rows/columns to add to the square submatrix.
        Resulting good matrix will have minimum of `r+min_add_K`
        rows/columns.
    start_maxvol_iters : integer
        How many iterations of square maxvol (optimization of 1-volume)
        is required to be done before actual rectangular 2-volume
        maximization.
    identity_submatrix : boolean
        Coefficients of expansions are computed as least squares
        solution. If `identity_submatrix` is True, returned matrix of
        coefficients will have submatrix, corresponding to good
        rows/columns, set to identity.
    job : character
        'R' to find good rows in approximant, 'C' to find good columns
        in approximant and 'F' for both rows and columns.
    top_k_index : integer
        Pivot rows/columns for good submatrix will be in range from `0`
        to `(top_k_index-1)`. This restriction is ignored, if
        `top_k_index` is -1.

    Returns
    -------
    Depending on `job` parameter, returns result of `rect_maxvol` for\
    left and/or right singular vectors of approximant.
    piv : numpy.ndarray(ndim=1, dtype=numpy.int32)
        Rows/columns of approximant of `A`, corresponding to submatrix,
        good in terms of 2-volume.
    C : numpy.ndarray(ndim=2)
        Coefficients of expansions of all rows/columns of approximant
        by good rows/columns `piv` of approximant.

    Examples
    --------
    >>> import numpy as np
    >>> from maxvolpy.maxvol import rect_maxvol_svd
    >>> np.random.seed(100)
    >>> a = np.random.rand(1000, 30, 2).view(dtype=np.complex128)[:,:,0]
    >>> piv, C = rect_maxvol_svd(a, svd_tol=1e-1, tol=1.0, job='R')
    >>> print('relative maxvol approximation error: {:.5f}'.format(
    ... np.linalg.norm(a-C.dot(a[piv]), 2)/np.linalg.norm(a, 2)))
    relative maxvol approximation error: 0.14497
    >>> print('maximum euclidian norm of row in matrix C: {:.5f}'.
    ... format(max([np.linalg.norm(C[i], 2) for i in range(1000)])))
    maximum euclidian norm of row in matrix C: 1.00000
    >>> piv, C = rect_maxvol_svd(a, svd_tol=1e-1, tol=1.5, job='R')
    >>> print('relative maxvol approximation error: {:.5f}'.format(
    ... np.linalg.norm(a-C.dot(a[piv]), 2)/np.linalg.norm(a, 2)))
    relative maxvol approximation error: 0.19640
    >>> print('maximum euclidian norm of row in matrix C: {:.5f}'.
    ... format(max([np.linalg.norm(C[i], 2) for i in range(1000)])))
    maximum euclidian norm of row in matrix C: 1.49535
    >>> piv, C = rect_maxvol_svd(a, svd_tol=1e-1, tol=2.0, job='R')
    >>> print('relative maxvol approximation error: {:.5f}'.format(
    ... np.linalg.norm(a-C.dot(a[piv]), 2)/np.linalg.norm(a, 2)))
    relative maxvol approximation error: 0.22981
    >>> print('maximum euclidian norm of row in matrix C: {:.5f}'.
    ... format(max([np.linalg.norm(C[i], 2) for i in range(1000)])))
    maximum euclidian norm of row in matrix C: 1.88485
    """
    if job == 'R':
        if top_k_index == -1:
            top_k_index = A.shape[0]
        # compute largest singular values and vectors for first
        # top_k_index rows
        U, S, V = svd_cut(A[:top_k_index], svd_tol, svd_alpha)
        # find projection coefficients of all other rows to subspace
        # of largest singular vectors of first rows
        B = A[top_k_index:].dot(V.T.conj())*(1.0/S).reshape(1,-1)
        # apply rect_maxvol for projection coefficients
        return rect_maxvol(np.vstack([U, B]), tol, maxK, min_add_K, minK,
                start_maxvol_iters, identity_submatrix, top_k_index)
    elif job == 'C':
        if top_k_index == -1:
            top_k_index = A.shape[1]
        # compute largest singular values and vectors for first
        # top_k_index columns
        U, S, V = svd_cut(A[:,:top_k_index], svd_tol, svd_alpha)
        # find projection coefficients of all other columns to subspace
        # of largest singular vectors of first columns
        B = (1.0/S).reshape(-1,1)*U.T.conj().dot(A[:,top_k_index:])
        # apply rect_maxvol for projection coefficients
        return rect_maxvol(np.vstack([V.T.conj(), B.T.conj()]), tol, maxK,
                min_add_K, minK, start_maxvol_iters, identity_submatrix,
                top_k_index)
    elif job == 'F':
        # proceed with job = 'R' and job = 'C' simultaneously
        if top_k_index != -1:
            value0, value1 = rect_maxvol_svd(A, svd_tol, svd_alpha, tol, maxK,
                    min_add_K, minK, start_maxvol_iters, identity_submatrix,
                    'R', top_k_index)
            value2, value3 = rect_maxvol_svd(A, svd_tol, svd_alpha, tol, maxK,
                    min_add_K, minK, start_maxvol_iters, identity_submatrix,
                    'C', top_k_index)
            return value0, value1, value2, value3
        U, S, V = svd_cut(A, svd_tol, svd_alpha)
        value0, value1 = rect_maxvol(U, tol, maxK, min_add_K, minK,
                start_maxvol_iters, identity_submatrix, top_k_index)
        value2, value3 = rect_maxvol(V.T.conj(), tol, maxK, min_add_K, minK,
                start_maxvol_iters, identity_submatrix, top_k_index)
        return value0, value1, value2, value3

def maxvol_svd(A, svd_tol=1e-3, svd_alpha=0., tol=1.05, max_iters=100, job='F',
        top_k_index=-1):
    """
    Applies SVD truncation and finds good square submatrix.

    Computes SVD for `top_k_index` rows and/or columns of given matrix
    `A`, increases singular values by regularizing parameter
    `svd_alpha`, cuts off singular values, lower than
    `svd_tol` (relatively, getting only highest singular vectors),
    projects rows and/or columns, starting from `top_k_index`, to space
    of first `top_k_index` rows and/or columns and runs `maxvol`
    for left and/or right singular vectors.

    Parameters
    ----------
    A : numpy.ndarray(ndim=2)
        Real or complex matrix.
    svd_tol : float
        Cut-off singular values parameter.
    svd_alpha : float
        Regularizing parameter for `misc.svd_cut`.
    tol : float
        Upper bound for infinite norm of coefficients of expansion of
        rows/columns of approximant of `A` good rows/columns of
        approximant. Minimum value is 1.
    max_iters : integer
        Maximum number of iterations. Each iteration swaps 2 rows.
    job : character
        'R' to find good rows in approximant, 'C' to find good columns
        in approximant and 'F' for both rows and columns.
    top_k_index : integer
        Pivot rows/columns for good submatrix will be in range from `0`
        to `(top_k_index-1)`. This restriction is ignored, if
        `top_k_index` is -1.

    Returns
    -------
    Depending on `job` parameter, returns result of `rect_maxvol` for\
    left and/or right singular vectors of approximant.
    piv : numpy.ndarray(ndim=1, dtype=numpy.int32)
        Rows/columns of approximant of `A`, corresponding to submatrix,
        good in terms of 1-volume.
    C : numpy.ndarray(ndim=2)
        Coefficients of expansions of all rows/columns of approximant
        by good rows/columns `piv` of approximant.

    Examples
    --------
    >>> import numpy as np
    >>> from maxvolpy.maxvol import maxvol_svd
    >>> np.random.seed(100)
    >>> a = np.random.rand(1000, 30, 2).view(dtype=np.complex128)[:,:,0]
    >>> piv, C = maxvol_svd(a, svd_tol=1e-1, tol=1.0, job='R')
    >>> print('relative maxvol approximation error: {:.5f}'.format(
    ... np.linalg.norm(a-C.dot(a[piv]), 2)/np.linalg.norm(a, 2)))
    relative maxvol approximation error: 0.24684
    >>> print('Chebyshev norm of matrix C: {:.5f}'.format(abs(C).max()))
    Chebyshev norm of matrix C: 1.00000
    >>> piv, C = maxvol_svd(a, svd_tol=1e-1, tol=1.05, job='R')
    >>> print('relative maxvol approximation error: {:.5f}'.format(
    ... np.linalg.norm(a-C.dot(a[piv]), 2)/np.linalg.norm(a, 2)))
    relative maxvol approximation error: 0.24684
    >>> print('Chebyshev norm of matrix C: {:.5f}'.format(abs(C).max()))
    Chebyshev norm of matrix C: 1.00000
    >>> piv, C = maxvol_svd(a, svd_tol=1e-1, tol=1.10, job='R')
    >>> print('relative maxvol approximation error: {:.5f}'.format(
    ... np.linalg.norm(a-C.dot(a[piv]), 2)/np.linalg.norm(a, 2)))
    relative maxvol approximation error: 0.25485
    >>> print('Chebyshev norm of matrix C: {:.5f}'.format(abs(C).max()))
    Chebyshev norm of matrix C: 1.06825
    """
    if tol < 1:
        tol = 1.0
    if job == 'R':
        if top_k_index == -1:
            top_k_index = A.shape[0]
        # compute largest singular values and vectors for first
        # top_k_index rows
        U, S, V = svd_cut(A[:top_k_index], svd_tol, svd_alpha)
        # find projection coefficients of all other rows to subspace of
        # largest singular vectors of first rows
        B = A[top_k_index:].dot(V.T.conj())*(1.0/S).reshape(1,-1)
        # apply maxvol for projection coefficients
        return maxvol(np.vstack([U, B]), tol, max_iters, top_k_index)
    elif job == 'C':
        if top_k_index == -1:
            top_k_index = A.shape[1]
        # compute largest singular values and vectors for first
        # top_k_index columns
        U, S, V = svd_cut(A[:,:top_k_index], svd_tol, svd_alpha)
        # find projection coefficients of all other columns to subspace
        # of largest singular vectors of first columns
        B = (1.0/S).reshape(-1,1)*U.T.conj().dot(A[:,top_k_index:])
        # apply rect_maxvol for projection coefficients
        return maxvol(np.vstack([V.T.conj(), B.T.conj()]), tol, max_iters,
                top_k_index)
    elif job == 'F':
        # procede with job = 'R' and job = 'C' simultaneously
        if top_k_index != -1:
            value0, value1 = maxvol_svd(A, svd_tol, svd_alpha, tol, max_iters,
                    'R', top_k_index)
            value2, value3 = maxvol_svd(A, svd_tol, svd_alpha, tol, max_iters,
                    'C', top_k_index)
            return value0, value1, value2, value3
        U, S, V = svd_cut(A, svd_tol, svd_alpha)
        value0, value1 = maxvol(U, tol, max_iters, top_k_index)
        value2, value3 = maxvol(V.T.conj(), tol, max_iters, top_k_index)
        return value0, value1, value2, value3

def rect_maxvol_qr(A, tol=1., maxK=None, min_add_K=None, minK=None,
        start_maxvol_iters=10, identity_submatrix=True, top_k_index=-1):
    """
    Finds good rectangular submatrix in Q factor of QR of `A`.

    When rank of `N`-by-`r` matrix `A` is not guaranteed to be
    equal to `r`, good submatrix in `A` can be found as good submatrix
    in Q factor of QR decomposition of `A`.

    Parameters
    ----------
    A : numpy.ndarray(ndim=2)
        Real or complex matrix of shape `(N, r)`, `N >= r`.
    tol : float
        Upper bound for euclidian norm of coefficients of expansion of
        rows of `A` by rows of good submatrix.
    maxK : integer
        Maximum number of rows in good submatrix.
    minK : integer
        Minimum number of rows in good submatrix.
    min_add_K : integer
        Minimum number of rows to add to the square submatrix.
        Resulting good matrix will have minimum of `r+min_add_K` rows.
    start_maxvol_iters : integer
        How many iterations of square maxvol (optimization of 1-volume)
        is required to be done before actual rectangular 2-volume
        maximization.
    identity_submatrix : boolean
        Coefficients of expansions are computed as least squares
        solution. If `identity_submatrix` is True, returned matrix of
        coefficients will have submatrix, corresponding to good rows,
        set to identity.
    top_k_index : integer
        Pivot rows for good submatrix will be in range from `0` to
        `(top_k_index-1)`. This restriction is ignored, if `top_k_index`
        is -1.

    Returns
    -------
    piv : numpy.ndarray(ndim=1, dtype=numpy.int32)
        Rows of matrix `A`, corresponding to submatrix, good in terms
        of 2-volume. Shape is `(K, )`.
    C : numpy.ndarray(ndim=2)
        Matrix of coefficients of expansions of all rows of `A` by good
        rows `piv`. Shape is `(N, K)`.

    Examples
    --------
    >>> import numpy as np
    >>> from maxvolpy.maxvol import rect_maxvol_qr
    >>> np.random.seed(100)
    >>> a = np.random.rand(1000, 30, 2).view(dtype=np.complex128)[:,:,0]
    >>> piv, C = rect_maxvol_qr(a, 1.0)
    >>> np.allclose(a, C.dot(a[piv]))
    True
    >>> print('maximum euclidian norm of row in matrix C: {:.5f}'.
    ... format(max([np.linalg.norm(C[i], 2) for i in range(1000)])))
    maximum euclidian norm of row in matrix C: 1.00000
    >>> piv, C = rect_maxvol_qr(a, 1.5)
    >>> np.allclose(a, C.dot(a[piv]))
    True
    >>> print('maximum euclidian norm of row in matrix C: {:.5f}'.
    ... format(max([np.linalg.norm(C[i], 2) for i in range(1000)])))
    maximum euclidian norm of row in matrix C: 1.49193
    >>> piv, C = rect_maxvol_qr(a, 2.0)
    >>> np.allclose(a, C.dot(a[piv]))
    True
    >>> print('maximum euclidian norm of row in matrix C: {:.5f}'.
    ... format(max([np.linalg.norm(C[i], 2) for i in range(1000)])))
    maximum euclidian norm of row in matrix C: 1.91954
    """
    N, r = A.shape
    if N <= r:
        return np.arange(N, dtype=np.int32), np.eye(N, dtype=A.dtype)
    Q = np.linalg.qr(A)[0]
    return rect_maxvol(Q, tol, maxK, min_add_K, minK, start_maxvol_iters,
            identity_submatrix, top_k_index)

def maxvol_qr(A, tol=1.05, max_iters=100, top_k_index=-1):
    """
    Finds good square submatrix in Q factor of QR of `A`.

    When rank of `N`-by-`r` matrix `A` is not guaranteed to be
    equal to `r`, good submatrix in `A` can be found as good submatrix
    in Q factor of QR decomposition of `A`.

    Parameters
    ----------
    A : numpy.ndarray(ndim=2)
        Real or complex matrix of shape `(N, r)`, `N >= r`.
    tol : float
        Upper bound for infinite norm of coefficients of expansion of rows
        of `A` by rows of good submatrix. Minimum value is 1.
    max_iters : integer
        Maximum number of iterations. Each iteration swaps 2 rows.
    top_k_index : integer
        Pivot rows for good submatrix will be in range from `0` to
        `(top_k_index-1)`. This restriction is ignored, if `top_k_index`
        is -1.
    
    Returns
    -------
    piv : numpy.ndarray(ndim=1, dtype=numpy.int32)
        Rows of matrix `A`, corresponding to submatrix, good in terms
        of 1-volume. Shape is `(r, )`.
    C : numpy.ndarray(ndim=2)
        Matrix of coefficients of expansions of all rows of `A` by good
        rows `piv`. Shape is `(N, r)`.

    Examples
    --------
    >>> import numpy as np
    >>> from maxvolpy.maxvol import maxvol_qr
    >>> np.random.seed(100)
    >>> a = np.random.rand(1000, 30, 2).view(dtype=np.complex128)[:,:,0]
    >>> piv, C = maxvol_qr(a, 1.0)
    >>> np.allclose(a, C.dot(a[piv]))
    True
    >>> print('Chebyshev norm of matrix C: {:.5f}'.format(abs(C).max()))
    Chebyshev norm of matrix C: 1.00000
    >>> piv, C = maxvol_qr(a, 1.05)
    >>> np.allclose(a, C.dot(a[piv]))
    True
    >>> print('Chebyshev norm of matrix C: {:.5f}'.format(abs(C).max()))
    Chebyshev norm of matrix C: 1.04641
    >>> piv, C = maxvol_qr(a, 1.10)
    >>> np.allclose(a, C.dot(a[piv]))
    True
    >>> print('Chebyshev norm of matrix C: {:.5f}'.format(abs(C).max()))
    Chebyshev norm of matrix C: 1.07854
    """
    N, r = A.shape
    if N <= r:
        return np.arange(N, dtype=np.int32), np.eye(N, dtype=A.dtype)
    Q = np.linalg.qr(A)[0]
    return maxvol(Q, tol, max_iters, top_k_index)

import numpy as np

try:
    from ._maxvol import c_rect_maxvol, c_maxvol
    rect_maxvol_func = c_rect_maxvol
    maxvol_func = c_maxvol
    __all__.extend(['c_rect_maxvol', 'c_maxvol'])
except:
    from scipy.linalg import solve_triangular, get_lapack_funcs, get_blas_funcs
    print("warning: fast C maxvol functions are not compiled, continue with"
            " python maxvol functions")
    rect_maxvol_func = py_rect_maxvol
    maxvol_func = py_maxvol

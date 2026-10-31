"""
Row selection by 1-volume maximisation (maxvol).

Reduced from maxvolpy 0.3.8 by Alexander Mikhalev, MIT licence -- see
LICENSE.maxvolpy.txt next to this file. pyace only ever needs the float64
square `maxvol`, so only that is kept.

`maxvol` prefers the compiled `_maxvol.c_maxvol`. If that extension is missing
or fails to load, it falls back to `py_maxvol`, which selects exactly the same
rows via LAPACK/BLAS through scipy but is roughly 3-15x slower depending on
matrix size.
"""
import logging

import numpy as np
from scipy.linalg import get_lapack_funcs, get_blas_funcs

__all__ = ['maxvol', 'py_maxvol']

log = logging.getLogger(__name__)


def py_maxvol(A, tol=1.05, max_iters=100, top_k_index=-1, verbose=False):
    """
    Pure numpy/scipy 1-volume maximisation.

    :param A: (N, r) float64 array with N > r
    :param tol: stop once no coefficient exceeds this, must be >= 1
    :param max_iters: maximum number of row swaps
    :param top_k_index: only consider the first `top_k_index` rows (-1 = all)
    :return: (row_indices, coefficients), the same pair `c_maxvol` returns
    """
    if tol < 1:
        tol = 1.0
    N, r = A.shape
    if N <= r:
        return np.arange(N, dtype=np.int32), np.eye(N, dtype=A.dtype)
    if top_k_index == -1 or top_k_index > N:
        top_k_index = N
    if top_k_index < r:
        top_k_index = r

    B = np.copy(A[:top_k_index], order='F')
    C = np.copy(A.T, order='F')
    H, ipiv, info = get_lapack_funcs('getrf', [B])(B, overwrite_a=1)
    if info < 0:
        raise ValueError("argument {} of getrf had an illegal value".format(-info))
    if info > 0:
        raise ValueError("Input matrix must not be singular")

    index = np.arange(N, dtype=np.int32)
    for i in range(r):
        index[i], index[ipiv[i]] = index[ipiv[i]], index[i]

    # solve A = CH with H in LU form
    B = H[:r]
    trtrs = get_lapack_funcs('trtrs', [B])
    trtrs(B, C, trans=1, lower=0, unitdiag=0, overwrite_b=1)
    trtrs(B, C, trans=1, lower=1, unitdiag=1, overwrite_b=1)

    # C is stored transposed, shape (r, N)
    view = C[:, :top_k_index]
    # reused every iteration -- np.abs(view) would otherwise allocate an
    # N-by-r temporary per swap, which dominates the runtime for large N
    abs_view = np.empty_like(view)
    np.abs(view, out=abs_view)
    i, j = divmod(abs_view.argmax(), top_k_index)

    ger = get_blas_funcs('ger', [C])
    iters = 0
    while abs(C[i, j]) > tol and iters < max_iters:
        if verbose and iters % 10 == 0:
            log.info("Iter %d/%d: abs_max = %.3f (tol = %.3f)",
                     iters, max_iters, abs(C[i, j]), tol)
        index[i] = j
        tmp_row = C[i].copy()
        tmp_column = C[:, j].copy()
        tmp_column[i] -= 1.
        ger(-1. / C[i, j], tmp_column, tmp_row, a=C, overwrite_a=1)
        iters += 1
        np.abs(view, out=abs_view)
        i, j = divmod(abs_view.argmax(), top_k_index)
    return index[:r].copy(), C.T


try:
    from ._maxvol import c_maxvol

    _maxvol_func = c_maxvol
    __all__.append('c_maxvol')
except ImportError as e:  # pragma: no cover - depends on how pyace was built
    _maxvol_func = None
    _import_error = e


def maxvol(A, tol=1.05, max_iters=100, top_k_index=-1, verbose=False):
    """
    Find `r` rows of the (N, r) array `A` whose submatrix has near-maximal
    absolute determinant.

    :param A: (N, r) float64 array
    :param tol: stop once no coefficient exceeds this, must be >= 1
    :param max_iters: maximum number of row swaps
    :param top_k_index: only consider the first `top_k_index` rows (-1 = all)
    :param verbose: log progress every 10 iterations
    :return: (row_indices, coefficients)
    """
    if not isinstance(A, np.ndarray):
        raise TypeError("argument must be of numpy.ndarray type")
    if A.ndim != 2:
        raise ValueError("argument must have 2 dimensions")
    A = np.asarray(A, dtype=np.float64)

    if _maxvol_func is not None:
        return _maxvol_func(A, tol, max_iters, top_k_index, int(verbose))

    log.warning("compiled pyace.maxvol._maxvol is unavailable (%s); "
                "falling back to the slower pure-python maxvol", _import_error)
    return py_maxvol(A, tol, max_iters, top_k_index, verbose)

"""
:py:mod:`maxvolpy.misc` module contains some auxiliary routines,
useful when working with different maxvol functions from
:py:mod:`maxvolpy.maxvol` and when building approximations with help
of functions from :py:mod:`maxvolpy.cross`.
"""

from __future__ import print_function, division, absolute_import

__all__ = ['svd_cut', 'reduce_approximation']

import numpy as np

def svd_cut(A, tol, alpha=0., norm=2):
    """
    Computes SVD and cuts low singular values.
    
    Computes singular values decomposition of matrix `A`, adds
    regularizing parameter `alpha` to each singular value and returns
    only largest singular values and vectors with relative tolerance
    `tol`.

    Parameters
    ----------
    A: numpy.ndarray
        Real or complex matrix or matrix-like object.
    tol: float
        Tolerance of cutting singular values operation.
    alpha: float, optional
        Regularizing parameter.
    norm: {2, 'fro'}, optional
        Defines norm, that is chosen when cutting singular values.

    Returns
    -------
    U: numpy.ndarray
        Left singular vectors, corresponding to largest singular values.
    S: numpy.ndarray
        Largest singular values.
    V: numpy.ndarray
        Right singular vectors, corresponding to largest singular values.
    """
    U, S, V = np.linalg.svd(A, full_matrices=0)
    S_reg = S+alpha
    S1 = S_reg[::-1]
    if norm == 2:
        tmp_S = tol*S1[-1]
        rank = S1.shape[0]-S1.searchsorted(tmp_S, 'left')
    elif norm == 'fro':
        S1 = S1*S1
        for i in range(1, S1.shape[0]):
            S1[i] += S1[i-1]
        tmp_S = S1[-1]*tol*tol
        rank = S1.shape[0]-S1.searchsorted(tmp_S, 'left')
    else:
        raise ValueError("Invalid parameter norm value")
    return U[:,:rank].copy(), S_reg[:rank].copy(), V[:rank].copy()

def reduce_approximation(U, V, tol, alpha=0.):
    """
    Performs `svd_cut` procedure for matrix `U.dot(V)`

    Parameters
    ----------
    U, V: numpy.ndarray
        Two matrices, such that `U.dot(V)` makes sense.
    tol: float
        Relative tolerance of cutting singular values of a matrix
        `U.dot(V)`.
    alpha: float
        Regularizing parameter for `svd_cut`.

    Returns
    -------
    U: numpy.ndarray
        Left singular vectors, corresponding to largest singular values.
    S: numpy.ndarray
        Largest singular values.
    V: numpy.ndarray
        Right singular vectors, corresponding to largest singular values.
    """
    Q1, R1 = np.linalg.qr(U)
    U1, S1, V1 = svd_cut(R1.dot(V), tol, alpha)
    return Q1.dot(U1), S1, V1

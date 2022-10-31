from __future__ import absolute_import, division, print_function
import numpy as np

def py_cross_full(A, rank):
    """returns cross approximation of matrix A, computed with full pivoting."""
    rows = np.ndarray(A.shape[0], dtype = np.int32)
    cols = np.ndarray(A.shape[1], dtype = np.int32)

def py_lu_full_fixed(A, rank):
    N, M = A.shape
    U = np.ndarray((N, rank), dtype = A.dtype)
    V = np.ndarray((rank, M), dtype = A.dtype)
    B = A.copy()
    for i in range(rank):
        maxi, maxj = divmod(abs(B).argmax(), M)
        max_val = B[maxi, maxj]
        U[:,i] = B[:,maxj].copy()
        V[i] = B[maxi]/max_val
        B -= U[:,i].reshape(-1, 1)*V[i].reshape(1, -1)
    return U, V

def py_lu_partial_fixed(A, rank):
    N, M = A.shape
    U = np.ndarray((N, rank), dtype = A.dtype)
    V = np.ndarray((rank, M), dtype = A.dtype)
    P = [i for i in range(N)]
    maxi = 0
    i = 0
    index_in_P = 0
    while i < rank:
        B = A[maxi]-U[maxi,:i].dot(V[:i])
        maxj = abs(B).argmax()
        max_val = B[maxj]
        if max_val == 0:
            if len(P) == 1:
                break
            P.pop(index_in_P)
            
        else:
            U[:,i] = A[:,maxj]-U[:,:i].dot(V[:i,maxj])
            V[i] = B/max_val
            P.pop(index_in_P)
            index_in_P = abs(U[P,i]).argmax()
            maxi = P[index_in_P]
            i += 1
    return U, V

def aca_partial(A, tol):
    N, M = A.shape
    rank = 10
    U = np.ndarray((N, rank), dtype = A.dtype)
    V = np.ndarray((rank, M), dtype = A.dtype)
    P = [i for i in range(N)]
    maxi = 0
    i = 0
    index_in_P = 0
    eps = 1+tol
    while tol < eps:
        B = A[maxi]-U[maxi,:i].dot(V[:i])
        maxj = abs(B).argmax()
        max_val = B[maxj]
        if max_val == 0:
            if len(P) == 1:
                break
            else:
                continue
        U[:,i] = A[:,maxj]-U[:,:i].dot(V[:i,maxj])
        V[i] = B/max_val
        P.pop(index_in_P)
        index_in_P = abs(U[P,i]).argmax()
        maxi = P[index_in_P]
        if i == 0:
            norm = np.linalg.norm(U[:,0])*np.linalg.norm(V[0])
            #print(norm)
        eps = np.linalg.norm(U[:,i])*np.linalg.norm(V[i])/norm
        #print(i, max_val, eps)
        if i+1 == rank:
            new_rank = min(2*rank, M, N)
            tmp_U = np.ndarray((N, new_rank), dtype = A.dtype)
            tmp_V = np.ndarray((new_rank, M), dtype = A.dtype)
            tmp_U[:,:rank] = U
            tmp_V[:rank] = V
            U = tmp_U
            V = tmp_V
            rank = new_rank
        i += 1
    return U[:,:i].copy(), V[:i].copy()

def py_ica(A, tol):
    N, M = A.shape
    tol2 = tol*tol
    max_rank = 100
    U = np.ndarray((N, max_rank), dtype = A.dtype)
    V = np.ndarray((max_rank, M), dtype = A.dtype)
    row_index = np.random.randint(N)
    row = A[row_index]
    rank = 0
    cross_norm = 0.
    skeleton_norm = 0.
    K = min(N, M)
    while rank < K and skeleton_norm >= tol*cross_norm:
        col_index = abs(row).argmax()
        col = A[:, col_index] - U[:, :rank].dot(V[:rank, col_index])
        row_index = abs(col).argmax()
        max_value = col[row_index]
        if max_value == 0:
            break
        #print(row_index, col_index, max_value)
        row = (A[row_index] - U[row_index, :rank].dot(V[:rank])) / max_value
        skeleton_norm = np.linalg.norm(col) * np.linalg.norm(row)
        skeleton_norm *= skeleton_norm
        cross_norm = cross_norm+skeleton_norm+2*(col.conj().dot(U[:, :rank]).dot(V[:rank].dot(row.conj()))).real
        #print(skeleton_norm, cross_norm)
        if max_rank == rank:
            max_rank = min(2*max_rank, K)
            tmp_U = np.ndarray((N, max_rank), dtype = A.dtype)
            tmp_V = np.ndarray((max_rank, M), dtype = A.dtype)
            tmp_U[:,:rank] = U
            tmp_V[:rank] = V
            U, V = tmp_U, tmp_V
        U[:,rank] = col.copy()
        V[rank] = row.copy()
        row[col_index] = 0
        rank += 1
    return U[:,:rank].copy(), V[:rank].copy()

import numpy as np
from scipy.sparse import csr_matrix

def slice_sparse(X, R, C):
    rows, cols = X.shape
    Y = csr_matrix((len(R), len(C)), dtype=X.dtype)
    assert(min(R) >= 0)
    assert(max(R) < rows)
    assert(min(C) >= 0)
    assert(max(C) < cols)
    for i, r in enumerate(R):
        for j, c in enumerate(C):
            Y[i, j] = X[r, c]

    return Y

def slice_matrix(X, R, C):
    return X[np.ix_(R, C)]

def slice_vector(X, R):
    return X[R]

def slice_matrix_dim(X, R, dim):
    if dim == 1:
        return X[R, :]
    elif dim == 2:
        return X[:, R]
    else:
        raise ValueError("Unsupported dimension")

def slice_list(unordered, index_map):
    return [unordered[i] for i in index_map]

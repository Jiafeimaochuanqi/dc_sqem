import numpy as np

def cumsum(X, dim, zero_prefix=False):
    if dim == 1:
        Y = np.cumsum(X, axis=0)
        if zero_prefix:
            Y = np.vstack([np.zeros((1, X.shape[1])), Y])
    elif dim == 2:
        Y = np.cumsum(X, axis=1)
        if zero_prefix:
            Y = np.hstack([np.zeros((X.shape[0], 1)), Y])
    else:
        raise ValueError("dim must be 1 (columns) or 2 (rows)")
    return Y
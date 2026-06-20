import numpy as np

def accumarray(S, V):
    assert S.size == V.size, "S and V should be same size"
    if S.size == 0:
        return np.zeros((0, 1))

    A = np.zeros((S.max() + 1, 1))

    for s in range(S.size):
        A[S[s]] += V[s]
    return A

def accumarray_scalar(S, V):
    if S.size == 0:
        return np.zeros((0, 1))
    A = np.zeros((S.max() + 1, 1))
    for s in range(S.size):
        A[S[s]] += V
    return A
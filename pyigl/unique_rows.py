from pyigl.sortrows import *
import numpy as np

def unique_rows(A):
    sortA,sort_idx=sortrows(A,ascending=True)
    diff = np.diff(sortA, axis=0)
    unique_diff = np.any(diff, axis=1)
    mask = np.concatenate(([True], unique_diff))

    C = sortA[mask]
    IA = sort_idx[mask]
    IC = np.zeros(A.shape[0], dtype=int)
    unique_idx=np.where(mask)[0]
    j = 0
    for i in range(sortA.shape[0]):
        if np.any(sortA[unique_idx[j]]!=sortA[i]):
            j += 1
        IC[sort_idx[i]] = j
    return C, IA, IC

import numpy as np

#Convert nested list to matrix.
def list_to_matrix_2d(V):
    if not V:
        return np.empty((0, 0))
    
    m = len(V)
    n = min(len(row) for row in V)
    
    if any(len(row) != n for row in V):
        raise ValueError("All rows must have the same number of columns.")
    
    M = np.zeros((m, n))
    for i in range(m):
        for j in range(n):
            M[i, j] = V[i][j]
    
    return M

#Convert fixed-size array to matrix.
def list_to_matrix_fixed(V):
    if not V:
        return np.empty((0, 0))
    
    m = len(V)
    n = len(V[0])
    
    M = np.zeros((m, n))
    for i in range(m):
        for j in range(n):
            M[i, j] = V[i][j]
    
    return M

#Convert nested list to matrix (with padding).
def list_to_matrix_padded(V, n, padding):
    m = len(V)
    M = np.full((m, n), padding)
    
    for i in range(m):
        row = V[i]
        if len(row) > n:
            raise ValueError("Row is longer than the specified number of columns.")
        
        M[i, :len(row)] = row
    
    return M

#Convert a single list to a matrix.
def list_to_matrix_1d(V):
    if not V:
        return np.empty((0, 0))
    return np.array(V)
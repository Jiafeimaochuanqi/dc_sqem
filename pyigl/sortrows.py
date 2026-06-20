import numpy as np

def sortrows(X, ascending=True):
    num_rows, num_cols = X.shape
    IX = np.arange(num_rows)
    
    # Sorting rule: sort by row, compare column by column.
    if ascending:
        IX = np.lexsort([X[:, col] for col in range(num_cols-1, -1, -1)])
    else:
        IX = np.lexsort([X[:, col] for col in range(num_cols)])
        IX = IX[::-1]
    
    Y = X[IX, :]
    return Y, IX

def sortrows_simple(X, ascending=True):
    Y, _ = sortrows(X, ascending)
    return Y
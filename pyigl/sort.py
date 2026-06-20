import numpy as np

def sort(X, dim=1, ascending=True):
    num_inner = X.shape[0] if dim == 1 else X.shape[1]
    
    if num_inner == 2:
        return sort2(X, dim, ascending)
    elif num_inner == 3:
        return sort3(X, dim, ascending)
    
    num_outer = X.shape[1] if dim == 1 else X.shape[0]
    Y = np.zeros_like(X)
    IX = np.zeros_like(X, dtype=int)
    
    for i in range(num_outer):
        if dim == 1:
            data = X[:, i]
        else:
            data = X[i, :]
        
        index_map = np.argsort(data)
        
        if not ascending:
            index_map = index_map[::-1]
        
        if dim == 1:
            Y[:, i] = data[index_map]
            IX[:, i] = index_map
        else:
            Y[i, :] = data[index_map]
            IX[i, :] = index_map
            
    return Y, IX

def sort2(X, dim, ascending=True):
    Y = np.copy(X)
    IX = np.zeros_like(X, dtype=int)
    
    if dim == 1:
        for i in range(X.shape[1]):
            if (ascending and X[0, i] > X[1, i]) or (not ascending and X[0, i] < X[1, i]):
                Y[0, i], Y[1, i] = Y[1, i], Y[0, i]
                IX[0, i], IX[1, i] = 1, 0
            else:
                IX[0, i], IX[1, i] = 0, 1
    else:
        for i in range(X.shape[0]):
            if (ascending and X[i, 0] > X[i, 1]) or (not ascending and X[i, 0] < X[i, 1]):
                Y[i, 0], Y[i, 1] = Y[i, 1], Y[i, 0]
                IX[i, 0], IX[i, 1] = 1, 0
            else:
                IX[i, 0], IX[i, 1] = 0, 1
    
    return Y, IX

def sort3(X, dim, ascending=True):
    Y = np.copy(X)
    IX = np.zeros_like(X, dtype=int)
    
    if dim == 1:
        for i in range(X.shape[1]):
            data = list(X[:, i])
            index_map = np.argsort(data)
            if not ascending:
                index_map = index_map[::-1]
            Y[:, i] = np.array(data)[index_map]
            IX[:, i] = index_map
    else:
        for i in range(X.shape[0]):
            data = list(X[i, :])
            index_map = np.argsort(data)
            if not ascending:
                index_map = index_map[::-1]
            Y[i, :] = np.array(data)[index_map]
            IX[i, :] = index_map
    
    return Y, IX
def sort_list(unsorted, ascending=True,return_index_map=False):
    index_map = list(range(len(unsorted)))
    index_map.sort(key=lambda i: unsorted[i])
    if not ascending:
        index_map.reverse()
    sorted_list = [unsorted[i] for i in index_map]
    if return_index_map:
        return sorted_list, index_map
    else:
        return sorted_list

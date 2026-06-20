import numpy as np

def doublearea(V, F):
    if F.shape[1] == 4:
        return doublearea_quad(V, F)

    dim = V.shape[1]
    m = F.shape[0]
    dblA = np.zeros(m)

    if dim == 3:
        for f in range(m):
            for d in range(3):
                rx = V[F[f, 0], d] - V[F[f, 2], d]
                sx = V[F[f, 1], d] - V[F[f, 2], d]
                ry = V[F[f, 0], (d + 1) % 3] - V[F[f, 2], (d + 1) % 3]
                sy = V[F[f, 1], (d + 1) % 3] - V[F[f, 2], (d + 1) % 3]
                dblAd = rx * sy - ry * sx
                dblA[f] += dblAd ** 2
        dblA = np.sqrt(dblA)
    elif dim == 2:
        for f in range(m):
            rx = V[F[f, 0], 0] - V[F[f, 2], 0]
            sx = V[F[f, 1], 0] - V[F[f, 2], 0]
            ry = V[F[f, 0], 1] - V[F[f, 2], 1]
            sy = V[F[f, 1], 1] - V[F[f, 2], 1]
            dblA[f] = rx * sy - ry * sx
    else:
        raise ValueError("Unsupported dimension")

    return dblA

def doublearea_single(A, B, C):
    A = np.array(A, dtype=np.float64)
    B = np.array(B, dtype=np.float64)
    C = np.array(C, dtype=np.float64)
    AB = B - A
    AC = C - A
    cross = np.cross(AB, AC)
    area = np.linalg.norm(cross)
    return area

def doublearea_quad(V, F):
    m = F.shape[0]
    Ft = np.zeros((m * 2, 3), dtype=int)
    for i in range(m):
        Ft[2 * i] = [F[i, 0], F[i, 1], F[i, 2]]
        Ft[2 * i + 1] = [F[i, 2], F[i, 3], F[i, 0]]

    doublearea_tri = doublearea(V, Ft)
    dblA = np.zeros(m)
    for i in range(m):
        dblA[i] = doublearea_tri[2 * i] + doublearea_tri[2 * i + 1]

    return dblA
def doublearea_from_edge_lengths(ul, nan_replacement=np.nan):
    assert ul.shape[1] == 3, "Only support triangles with 3 edges"
    
    m = ul.shape[0]
    dblA = np.zeros((m, 1))
    
    # Sort edge lengths (l) in ascending order for each triangle
    l = np.sort(ul, axis=1)
    
    for i in range(m):
        # Apply Kahan's Heron's formula to compute the area
        arg = (l[i, 0] + (l[i, 1] + l[i, 2])) * \
              (l[i, 2] - (l[i, 0] - l[i, 1])) * \
              (l[i, 2] + (l[i, 0] - l[i, 1])) * \
              (l[i, 0] + (l[i, 1] - l[i, 2]))
        
        dblA[i] = 0.5 * np.sqrt(arg)#2*area
        
        # If the computed area is NaN, replace it with `nan_replacement`
        if np.isnan(dblA[i]):
            dblA[i] = nan_replacement
        
        # Ensure the computed area is valid
        assert not np.isnan(dblA[i]), "DOUBLEAREA() PRODUCED NaN"
    
    return dblA

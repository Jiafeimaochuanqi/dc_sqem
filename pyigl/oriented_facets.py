import numpy as np

def oriented_facets(F):
    # E will hold the oriented facets
    E = np.zeros((F.shape[0] * F.shape[1], F.shape[1] - 1), dtype=F.dtype)

    if F.shape[1] == 4:  # Tetrahedrons
        E[0*F.shape[0]:1*F.shape[0], 0] = F[:, 1]
        E[0*F.shape[0]:1*F.shape[0], 1] = F[:, 3]
        E[0*F.shape[0]:1*F.shape[0], 2] = F[:, 2]
        E[1*F.shape[0]:2*F.shape[0], 0] = F[:, 0]
        E[1*F.shape[0]:2*F.shape[0], 1] = F[:, 2]
        E[1*F.shape[0]:2*F.shape[0], 2] = F[:, 3]
        E[2*F.shape[0]:3*F.shape[0], 0] = F[:, 0]
        E[2*F.shape[0]:3*F.shape[0], 1] = F[:, 3]
        E[2*F.shape[0]:3*F.shape[0], 2] = F[:, 1]
        E[3*F.shape[0]:4*F.shape[0], 0] = F[:, 0]
        E[3*F.shape[0]:4*F.shape[0], 1] = F[:, 1]
        E[3*F.shape[0]:4*F.shape[0], 2] = F[:, 2]

    elif F.shape[1] == 3:  # Triangles
        E[0*F.shape[0]:1*F.shape[0], 0] = F[:, 1]
        E[0*F.shape[0]:1*F.shape[0], 1] = F[:, 2]
        E[1*F.shape[0]:2*F.shape[0], 0] = F[:, 2]
        E[1*F.shape[0]:2*F.shape[0], 1] = F[:, 0]
        E[2*F.shape[0]:3*F.shape[0], 0] = F[:, 0]
        E[2*F.shape[0]:3*F.shape[0], 1] = F[:, 1]

    return E
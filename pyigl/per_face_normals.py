
import numpy as np

def per_face_normals_base(V, F, Z=None):
    if Z is None:
        Z = np.zeros(3)
    
    N = np.zeros((F.shape[0], 3))
    
    for i in range(F.shape[0]):
        v0 = V[F[i, 0]]
        v1 = V[F[i, 1]]
        v2 = V[F[i, 2]]
        edge1 = v1 - v0
        edge2 = v2 - v0

        normal = np.cross(edge1, edge2)

        norm = np.linalg.norm(normal)
        
        if norm == 0:
            N[i] = Z
        else:
            N[i] = normal / norm
    
    return N

def per_face_normals(V, F):
    return per_face_normals_base(V, F, Z=np.zeros(3))
import numpy as np
from scipy.linalg import svd, qr
from .mesh_utils import *
# Constants representing point types
CORNER = 0
EDGE = 1
PLANE = 2
NONETYPE = 4

class DualPoint:
    def __init__(self,point=None,type:int=NONETYPE,dir=None,is_inside=False,radius=0):
        self.m_point=point
        self.m_type=type#0 for CORNER, 1 for EDGE, 2 for PLANE
        self.m_dir=dir#line dir or plane normal
        self.m_is_inside=is_inside
        self.radius=radius

class SQEF:
    def __init__(self, normals,shape_samples,cell_samples):
        """
        Initialize QEF (Quadratic Error Function)
        A: (Mx4) Matrix
        b: (Mx1) Vector
        """
        self.normals=normals
        self.shape_samples=shape_samples
        self.cell_samples=cell_samples
        fixed_column=np.full((len(self.normals), 1), 1)
        self.A=np.c_[self.normals, fixed_column]#[n_{i}^{t},1\\]
                    
        self.b=np.sum(self.normals * self.shape_samples, axis=1)#[n_{i}^{t}p_{i},1\\]
        self.is_plane = False
        self.is_line = False
        self.dir = None
        self.point = None

    def solve_with_group(self,groups,is_cylinder, svd_threshold=0.01,debug=False):
        """
        Solve the QEF using SVD decomposition.
        svd_threshold: threshold for singular value truncation
        """
        # QR decomposition
        if len(self.A)<4:
            self.A=np.vstack((self.A, self.A))
            self.b=np.hstack((self.b, self.b))
        Q, R = qr(self.A, mode='economic')#Q:Mx4,R:Mx4
        Qb = Q.T @ self.b

        # Retain only the first 4 rows (Mx4 to 4x4)
        R = R[:4, :4]
        Qb = Qb[:4]

        # SVD decomposition
        U, s, Vt = svd(R, full_matrices=True)
        #print(Vt)
        if debug:
            print("R:",R)
            print("s:",s)
            print("Vt:",Vt)
        # Compute diagonal inverse of singular values
        mask = s > 1e-6
        diag_s_inv = np.zeros_like(R)
        diag_s_inv[mask, mask] = 1 / s[mask]
        
        for i in range(len(s)):
            if s[i] < s[0] * svd_threshold:
                if i == 1:
                    v1=stableNormalized(Vt[1, :3])[0]
                    v3=stableNormalized(Vt[3, :3])[0]
                    self.dir = stableNormalized(np.cross(v1,v3))[0]
                    self.is_plane=True
                    if len(groups)>=2:
                        self.is_plane=True
                        v2=stableNormalized(Vt[2, :3])[0]
                        v3=stableNormalized(Vt[3, :3])[0]
                        #print(np.dot(v0,v1))
                        self.dir = stableNormalized(np.cross(v2,v3))[0]
                        break
                    else:
                        return None
                elif i == 2:
                    # print(Vt[0])
                    # print(Vt[1])
                    self.is_plane=True
                    v2=stableNormalized(Vt[2, :3])[0]
                    v3=stableNormalized(Vt[3, :3])[0]
                    #print(np.dot(v0,v1))
                    self.dir = stableNormalized(np.cross(v2,v3))[0]
                    break
                elif i==3:
                    v3=stableNormalized(Vt[3, :3])[0]  
                    self.is_line=True                 
                    self.dir=v3
                    break
        
        if is_cylinder:
            self.is_plane=False
            v3=stableNormalized(Vt[3, :3])[0]  
            self.is_line=True                 
            self.dir=v3
        else:
            if len(groups)==1 and self.is_plane:
                v1=stableNormalized(Vt[1, :3])[0]
                v3=stableNormalized(Vt[3, :3])[0]
                self.dir = stableNormalized(np.cross(v1,v3))[0]
                self.is_plane=True
                return None
            if len(groups)<=2 and self.is_line:
                if len(groups)==2:
                    self.is_line=False
                    self.is_plane=True
                    v2=stableNormalized(Vt[2, :3])[0]
                    v3=stableNormalized(Vt[3, :3])[0]
                    #print(np.dot(v0,v1))
                    self.dir = stableNormalized(np.cross(v2,v3))[0]
                else:
                    self.is_line=False
                    self.is_plane=True
                    return None
            if len(groups)<4 and not self.is_plane and not self.is_line:
                if len(groups)==3:
                    self.is_line=True
                    v3=stableNormalized(Vt[3, :3])[0]                           
                    self.dir=v3
                elif len(groups)==2:
                    self.is_plane=True
                    v2=stableNormalized(Vt[2, :3])[0]
                    v3=stableNormalized(Vt[3, :3])[0]
                    #print(np.dot(v0,v1))
                    self.dir = stableNormalized(np.cross(v2,v3))[0]
                else:
                    self.is_line=False
                    self.is_plane=False
                    return None

                
        # Compute solution
        A_inv = Vt.T @ diag_s_inv @ U.T
        self.point = A_inv @ Qb

        #R=[R_{top4}\\0]
        #R_{top4}^{T}R_{top4}=R^{T}R=A^{T}A
        #A=UDV^{T}
        #A^{T}A=VD^{2}V^{T}=> A^{T}A@V=VD^{2}

        return self.point
    def solve_in_corner(self,debug=False):
        """
        Solve the QEF using SVD decomposition.
        svd_threshold: threshold for singular value truncation
        """
        if len(self.A)<4:
            self.A=np.vstack((self.A, self.A))
            self.b=np.hstack((self.b, self.b))
        # QR decomposition
        Q, R = qr(self.A, mode='economic')#Q:Mx4,R:Mx4
        Qb = Q.T @ self.b

        # Retain only the first 4 rows (Mx4 to 4x4)
        R = R[:4, :4]
        Qb = Qb[:4]

        # SVD decomposition
        U, s, Vt = svd(R, full_matrices=True)
        #print(Vt)
        if debug:
            print("R:",R)
            print("s:",s)
            print("Vt:",Vt)
        # Compute diagonal inverse of singular values
        mask = s > 1e-6
        diag_s_inv = np.zeros_like(R)
        diag_s_inv[mask, mask] = 1 / s[mask]
        A_inv = Vt.T @ diag_s_inv @ U.T
        self.point = A_inv @ Qb
        return self.point

    def solve_in_line(self, svd_threshold=0.01,debug=False):
        """
        Solve the QEF using SVD decomposition.
        svd_threshold: threshold for singular value truncation
        """
        if len(self.A)<4:
            self.A=np.vstack((self.A, self.A))
            self.b=np.hstack((self.b, self.b))
        # QR decomposition
        Q, R = qr(self.A, mode='economic')#Q:Mx4,R:Mx4
        Qb = Q.T @ self.b

        # Retain only the first 4 rows (Mx4 to 4x4)
        R = R[:4, :4]
        Qb = Qb[:4]

        # SVD decomposition
        U, s, Vt = svd(R, full_matrices=True)
        #print(Vt)
        if debug:
            print("R:",R)
            print("s:",s)
            print("Vt:",Vt)
        # Compute diagonal inverse of singular values
        mask = s > 1e-6
        diag_s_inv = np.zeros_like(R)
        diag_s_inv[mask, mask] = 1 / s[mask]

        for i in range(len(s)):
            if s[i] < s[0] * svd_threshold:
                if i == 1:
                    v1=stableNormalized(Vt[1, :3])[0]
                    v3=stableNormalized(Vt[3, :3])[0]
                    self.dir = stableNormalized(np.cross(v1,v3))[0]
                    self.is_plane=True
                    break
                elif i == 2:
                    # print(Vt[0])
                    # print(Vt[1])
                    self.is_plane=True
                    v2=stableNormalized(Vt[2, :3])[0]
                    v3=stableNormalized(Vt[3, :3])[0]
                    #print(np.dot(v0,v1))
                    self.dir = stableNormalized(np.cross(v2,v3))[0]
                    break
                elif i==3:
                    self.is_line=True 
                    v3=stableNormalized(Vt[3, :3])[0]                 
                    self.dir=v3
                    break
        if self.is_plane:
            self.is_line=True  
            self.is_plane=False  
            v3=stableNormalized(Vt[3, :3])[0]      
            self.dir=v3

        # Compute solution
        A_inv = Vt.T @ diag_s_inv @ U.T
        self.point = A_inv @ Qb

        #R=[R_{top4}\\0]
        #R_{top4}^{T}R_{top4}=R^{T}R=A^{T}A
        #A=UDV^{T}
        #A^{T}A=VD^{2}V^{T}=> A^{T}A@V=VD^{2}

        return self.point
    def solve_in_plane(self,debug=False):
        """
        Solve the QEF using SVD decomposition.
        svd_threshold: threshold for singular value truncation
        """
        if len(self.A)<4:
            self.A=np.vstack((self.A, self.A))
            self.b=np.hstack((self.b, self.b))
        # QR decomposition
        Q, R = qr(self.A, mode='economic')#Q:Mx4,R:Mx4
        Qb = Q.T @ self.b

        # Retain only the first 4 rows (Mx4 to 4x4)
        R = R[:4, :4]
        Qb = Qb[:4]

        # SVD decomposition
        U, s, Vt = svd(R, full_matrices=True)
        #print(Vt)
        if debug:
            print("R:",R)
            print("s:",s)
            print("Vt:",Vt)
        # Compute diagonal inverse of singular values
        mask = s > 1e-6
        diag_s_inv = np.zeros_like(R)
        diag_s_inv[mask, mask] = 1 / s[mask]

        self.is_plane=True
        v2=stableNormalized(Vt[2, :3])[0]
        v3=stableNormalized(Vt[3, :3])[0]
        #print(np.dot(v0,v1))
        self.dir = stableNormalized(np.cross(v2,v3))[0]

        # Compute solution
        A_inv = Vt.T @ diag_s_inv @ U.T
        self.point = A_inv @ Qb

        #R=[R_{top4}\\0]
        #R_{top4}^{T}R_{top4}=R^{T}R=A^{T}A
        #A=UDV^{T}
        #A^{T}A=VD^{2}V^{T}=> A^{T}A@V=VD^{2}

        return self.point
    def solve(self, svd_threshold=0.01,debug=False):
        """
        Solve the QEF using SVD decomposition.
        svd_threshold: threshold for singular value truncation
        """
        if len(self.A)<4:
            self.A=np.vstack((self.A, self.A))
            self.b=np.hstack((self.b, self.b))
        # QR decomposition
        Q, R = qr(self.A, mode='economic')#Q:Mx4,R:Mx4
        Qb = Q.T @ self.b

        # Retain only the first 4 rows (Mx4 to 4x4)
        R = R[:4, :4]
        Qb = Qb[:4]

        # SVD decomposition
        U, s, Vt = svd(R, full_matrices=True)
        #print(Vt)
        if debug:
            print("R:",R)
            print("s:",s)
            print("Vt:",Vt)
        # Compute diagonal inverse of singular values
        mask = s > 1e-6
        diag_s_inv = np.zeros_like(R)
        diag_s_inv[mask, mask] = 1 / s[mask]
        for i in range(len(s)):
            if s[i] < s[0] * svd_threshold:
                if i == 1:
                    v1=stableNormalized(Vt[1, :3])[0]
                    v3=stableNormalized(Vt[3, :3])[0]
                    self.dir = stableNormalized(np.cross(v1,v3))[0]
                    self.is_plane=True
                    return None
                elif i == 2:
                    # print(Vt[0])
                    # print(Vt[1])
                    self.is_plane=True
                    v2=stableNormalized(Vt[2, :3])[0]
                    v3=stableNormalized(Vt[3, :3])[0]
                    #print(np.dot(v0,v1))
                    self.dir = stableNormalized(np.cross(v2,v3))[0]
                    break
                elif i==3:
                    v3=stableNormalized(Vt[3, :3])[0]  
                    self.is_line=True                 
                    self.dir=v3
                    break
        # Compute solution
        A_inv = Vt.T @ diag_s_inv @ U.T
        self.point = A_inv @ Qb

        #R=[R_{top4}\\0]
        #R_{top4}^{T}R_{top4}=R^{T}R=A^{T}A
        #A=UDV^{T}
        #A^{T}A=VD^{2}V^{T}=> A^{T}A@V=VD^{2}

        return self.point

    def residual(self, v):
        """
        Compute the residual of the QEF solution.
        v: solution vector
        """
        return np.linalg.norm(self.A @ v - self.b) / len(self.b)


def _sqef_clear_type_state(sqef):
    sqef.is_plane = False
    sqef.is_line = False
    sqef.dir = None
    sqef.point = None


def _sqef_factor_once(sqef):
    A = sqef.A
    b = sqef.b

    if len(A) < 4:
        A = np.vstack((A, A))
        b = np.hstack((b, b))

    Q, R = qr(A, mode='economic')
    Qb = Q.T @ b

    R = R[:4, :4]
    Qb = Qb[:4]

    U, s, Vt = svd(R, full_matrices=True)

    diag_s_inv = np.zeros_like(R)
    mask = s > 1e-6
    diag_s_inv[mask, mask] = 1 / s[mask]

    A_inv = Vt.T @ diag_s_inv @ U.T
    point = A_inv @ Qb
    return point, Vt


def solve_x3(A, b, x_known):
    A_known = A[:, :3]          # (n, 3)
    A_unknown = A[:, 3]         # (n,)
    b_adj = b - A_known @ x_known   # (n,)
    numerator = A_unknown @ b_adj    
    denominator = A_unknown @ A_unknown 
    if denominator == 0:
        raise ValueError("A[:, 3] is zero")

    x3 = numerator / denominator
    return x3

def solve_point_pre_sqef_with_group(
    sqef: SQEF,
    min_corner,
    max_corner,
    groups,
    is_cylinder,
    svd_threshold,
    one_ring_boxes,
    two_ring_boxes,
    corner_factor=1e-3,
    edge_factor=1e-3,
    plane_factor=1e-3,
):
    center=(min_corner+max_corner)*0.5
    def _solve_result_from_current_type():
        v = sqef.point
        is_point_inside = is_inside(v[:3], min_corner, max_corner, corner_factor)

        if not sqef.is_line and not sqef.is_plane:
            if is_point_inside:
                radius=v[3]
                return DualPoint(v[:3], CORNER, None, True,radius)
            return DualPoint(None, CORNER, None, False)

        if sqef.is_line:
            intersects, enter_point, exit_point = line_aabb_intersection(
                sqef.point[:3], sqef.dir, min_corner, max_corner,edge_factor
            )
            if intersects:
                best_v = (enter_point + exit_point) * 0.5
                radius=solve_x3(sqef.A,sqef.b,best_v)
                return DualPoint(best_v, EDGE, sqef.dir, True,radius)
            return DualPoint(None, EDGE, sqef.dir, False)

        if sqef.is_plane:
            unique_intersections = plane_aabb_intersection(
                sqef.point[:3], sqef.dir, min_corner, max_corner,plane_factor
            )
            if len(unique_intersections) > 0:
                best_v = np.mean(np.array(unique_intersections), axis=0)
                radius=solve_x3(sqef.A,sqef.b,best_v)
                return DualPoint(best_v, PLANE, sqef.dir, True,radius)
            return DualPoint(None, PLANE, sqef.dir, False)

        return DualPoint(None, PLANE, sqef.dir, False)

    def _get_neighbor_solution_of_current_type(neighbor_boxes):
        if not sqef.is_line and not sqef.is_plane:
            for adj_min_corner, adj_max_corner in neighbor_boxes:
                if is_inside(sqef.point[:3], adj_min_corner, adj_max_corner) and sqef.point[3]>0:
                    return DualPoint(sqef.point[:3], CORNER, sqef.dir, False,sqef.point[3])
            return DualPoint(None, CORNER, sqef.dir, False)

        elif sqef.is_line:
            min_dis=np.inf
            result=None
            for adj_min_corner, adj_max_corner in neighbor_boxes:
                intersects, enter_point, exit_point = line_aabb_intersection(
                    sqef.point[:3], sqef.dir, adj_min_corner, adj_max_corner
                )
                if intersects:
                    best_v=(enter_point+ exit_point)*0.5
                    radius=solve_x3(sqef.A,sqef.b,best_v)
                    dis=np.linalg.norm(best_v-center)
                    if radius>0 and dis<min_dis:
                        result=DualPoint(best_v, EDGE, sqef.dir, False)
                        result.radius=radius
                        min_dis=dis
            if result is not None:
                return result
            return DualPoint(None, EDGE, sqef.dir, False)

        else:
            min_dis=np.inf
            result=None
            for adj_min_corner, adj_max_corner in neighbor_boxes:
                unique_intersections = plane_aabb_intersection(
                    sqef.point[:3], sqef.dir, adj_min_corner, adj_max_corner
                )
                if len(unique_intersections) > 0:
                    best_v=np.mean(np.array(unique_intersections), axis=0)
                    radius=solve_x3(sqef.A,sqef.b,best_v)
                    dis=np.linalg.norm(best_v-center)
                    if radius>0 and dis<min_dis:
                        result=DualPoint(best_v, PLANE, sqef.dir, False)
                        result.radius=radius
                        min_dis=dis
            if result is not None:
                return result
            return DualPoint(None, PLANE, sqef.dir, False)


    def _force_line():
        point, Vt = _sqef_factor_once(sqef)
        _sqef_clear_type_state(sqef)
        sqef.point = point
        sqef.is_line = True
        sqef.dir = stableNormalized(Vt[3, :3])[0]

    def _force_plane():
        point, Vt = _sqef_factor_once(sqef)
        _sqef_clear_type_state(sqef)
        sqef.point = point
        sqef.is_plane = True
        v2 = stableNormalized(Vt[2, :3])[0]
        v3 = stableNormalized(Vt[3, :3])[0]
        sqef.dir = stableNormalized(np.cross(v2, v3))[0]

    _sqef_clear_type_state(sqef)
    v=sqef.solve_with_group(groups, is_cylinder, svd_threshold)
    if v is None:
        return DualPoint(None, PLANE, None, False)
    result = _solve_result_from_current_type()
    if result.m_point is not None:
        return result

    result= _get_neighbor_solution_of_current_type(one_ring_boxes)
    if result.m_point is not None:
        return result    
    # result= _get_neighbor_solution_of_current_type(two_ring_boxes)
    # if result.m_point is not None:
    #     return result

    #corner -> line -> plane
    if not sqef.is_line and not sqef.is_plane:
        _force_line()
        result = _solve_result_from_current_type()
        if result.m_point is not None:
            return result
        result= _get_neighbor_solution_of_current_type(one_ring_boxes)
        if result.m_point is not None:
            return result    
        # result= _get_neighbor_solution_of_current_type(two_ring_boxes)
        # if result.m_point is not None:
        #     return result

        _force_plane()
        result =_solve_result_from_current_type()
        if result.m_point is not None:
            return result
        result= _get_neighbor_solution_of_current_type(one_ring_boxes)
        if result.m_point is not None:
            return result    
        result= _get_neighbor_solution_of_current_type(two_ring_boxes)
        return result
    #line -> plane
    elif sqef.is_line:
        _force_plane()
        result =_solve_result_from_current_type()
        if result.m_point is not None:
            return result
        result= _get_neighbor_solution_of_current_type(one_ring_boxes)
        if result.m_point is not None:
            return result    
        result= _get_neighbor_solution_of_current_type(two_ring_boxes)
        return result
    else:
        result= _get_neighbor_solution_of_current_type(two_ring_boxes)
        return result
    #return result
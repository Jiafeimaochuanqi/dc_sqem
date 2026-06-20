import numpy as np
import trimesh
from collections import defaultdict
from pyigl.doublearea import *
import numpy as np
import numpy as np
import os
def compute_octree_node_edge_length(max_depth):
    min_corner=np.array([-0.525, -0.519, -0.52])
    max_corner=np.array([0.501, 0.574, 0.532])
    #the subdivision depth for sampling in a cell
    sampling_depth=1
    #the sampling resolution on each edge
    m_capacity_per_dim = int(2 ** (max_depth + sampling_depth)) + 1
    #resolution= int((m_capacity_per_dim - 1) / (2 ** sampling_depth))
    # compute the unit edge length
    m_unit_edge_length = (max_corner[0] - min_corner[0]) / (m_capacity_per_dim - 1)
    return m_unit_edge_length

def load_mesh(name, data_directory):
    mesh_path = os.path.join(data_directory, name + ".obj")
    mesh = trimesh.load_mesh(mesh_path)
    return mesh

def indices_in_bbox(points, min_corner, max_corner):
    inside_mask = (points >= min_corner).all(axis=1) & (points <= max_corner).all(axis=1)
    return np.where(inside_mask)[0]


def get_k_smallest_indices(arr, k):
    sorted_with_indices = sorted(enumerate(arr), key=lambda x: x[1])
    return np.array([index for index, value in sorted_with_indices[:k]])

def box_intersection(box1, box2):
    min1, max1 = np.array(box1[0]), np.array(box1[1])
    min2, max2 = np.array(box2[0]), np.array(box2[1])
    intersect_min = np.maximum(min1, min2)
    intersect_max = np.minimum(max1, max2)
    
    if np.all(intersect_min <= intersect_max):
        return (intersect_min, intersect_max)
    else:
        return None


def sample_points_on_edge(edge_vertices, n_samples):
    if n_samples <= 0:
        return np.array([]).reshape(0, 3)
    
    point_a, point_b = edge_vertices[0], edge_vertices[1]
    t_values = np.linspace(0, 1, n_samples + 2)[1:-1]
    samples = point_a + t_values.reshape(-1, 1) * (point_b - point_a)
    
    return samples

def extract_edges(face):
    edges = []
    n = len(face)
    for i in range(n):
        u = face[i]
        v = face[(i+1) % n]
        edges.append(tuple(sorted([u,v])))
        # edges.append((min(u, v), max(u, v)))
    return edges

def update_face_edge_vertex_table(faces,edge_to_faces,vertex_to_edge,deleted_face_flag,face_id):
        face=faces[face_id]
        edge0=tuple(sorted([face[0],face[1]]))
        edge1=tuple(sorted([face[1],face[2]]))
        edge2=tuple(sorted([face[2],face[3]]))
        edge3=tuple(sorted([face[3],face[0]]))
        edges=[edge0,edge1,edge2,edge3] 
        for edge in edges: 
            edge_to_faces[edge].remove(face_id)
            for fid in edge_to_faces[edge]:
                if deleted_face_flag[fid]:
                    edge_to_faces[edge].remove(fid)

            if len(edge_to_faces[edge])==0:
                id0,id1=edge
                vertex_to_edge[id0].remove(edge)
                vertex_to_edge[id1].remove(edge)
def make_dict_edge2face(faces,deleted_flags=None):
    edge_to_faces = {}
    vertex_to_edge={}
    for face_id, face in enumerate(faces):
        if deleted_flags is not None and deleted_flags[face_id]:
            continue
        edges = extract_edges(face)
        for edge in edges:
            if edge not in edge_to_faces:
                edge_to_faces[edge] = set()
            edge_to_faces[edge].add(face_id)
            id0,id1=edge
            if id0 not in vertex_to_edge:
                vertex_to_edge[id0]=set()
            vertex_to_edge[id0].add(edge)
            if id1 not in vertex_to_edge:
                vertex_to_edge[id1]=set()
            vertex_to_edge[id1].add(edge)
    for edge in edge_to_faces:
        edge_to_faces[edge] = list(edge_to_faces[edge])
    for vertex in vertex_to_edge:
        vertex_to_edge[vertex] = list(vertex_to_edge[vertex])    
    return edge_to_faces,vertex_to_edge

def point_projection_on_plane(point, plane_point, plane_normal):
    point = np.array(point, dtype=np.float64)
    plane_point = np.array(plane_point, dtype=np.float64)
    plane_normal = np.array(plane_normal, dtype=np.float64)

    plane_normal = plane_normal / np.linalg.norm(plane_normal)

    vec = point - plane_point

    distance = np.dot(vec, plane_normal)

    projection = point - distance * plane_normal
    
    return projection
def point_projection_on_line(point, line_point, line_vector):
    point_vector = point-line_point

    dot_product = np.dot(point_vector, line_vector)
    line_length_squared = (line_vector[0] ** 2 +
                           line_vector[1] ** 2 +
                           line_vector[2] ** 2)

    projection_scalar = dot_product / line_length_squared

    projection_vector = (projection_scalar * line_vector[0],
                         projection_scalar * line_vector[1],
                         projection_scalar * line_vector[2])

    return line_point+projection_vector


def stableNormalized(vectors):

    vectors = np.asarray(vectors, dtype=float)
    if vectors.ndim == 1:
        vectors = vectors[np.newaxis, :]
    assert vectors.ndim == 2 and vectors.shape[1] == 3, "input must be shape (n,3) array"

    w = np.max(np.abs(vectors), axis=1, keepdims=True)

    w[w == 0] = 1

    scaled = vectors / w

    z = np.sum(scaled**2, axis=1, keepdims=True)

    result = np.zeros_like(vectors)

    mask = (z > 0).flatten()
    
    if np.any(mask):

        normalized = scaled[mask] / np.sqrt(z[mask])
        result[mask] = normalized
    
    return result

def sample_points_in_quad(vertices,n):
    fv0=vertices[:3]
    fv1=np.array([vertices[0],vertices[2],vertices[3]])
    s0=doublearea(fv0,np.array([[0,1,2]]))
    s1=doublearea(fv1,np.array([[0,1,2]]))
    if s0+s1>0:
        n0=int(np.round((s0*n)/(s0+s1)))
        n1=n-n0
        result=[]
        result.extend(sample_points_in_triangle(fv0,n0))
        result.extend(sample_points_in_triangle(fv1,n1))
        return np.array(result)
    else:
        return vertices

def sample_points_in_triangle(vertices, n):
    points=[]
    if(n>7):
        points.extend(vertices)
        points.append((vertices[0]+vertices[1])*0.5)
        points.append((vertices[0]+vertices[2])*0.5)
        points.append((vertices[1]+vertices[2])*0.5)
        points.append((vertices[0]+vertices[1]+vertices[2])/3)
        r1 = np.random.rand(n-7)
        r2 = np.random.rand(n-7)
        u = 1 - np.sqrt(r1)
        v = np.sqrt(r1) * (1 - r2)
        w = np.sqrt(r1) * r2
    else:
        r1 = np.random.rand(n)
        r2 = np.random.rand(n)
        u = 1 - np.sqrt(r1)
        v = np.sqrt(r1) * (1 - r2)
        w = np.sqrt(r1) * r2
    other_points = u[:, np.newaxis] * vertices[0] + v[:, np.newaxis] * vertices[1] + w[:, np.newaxis] * vertices[2]
    points.extend(other_points)
    return np.array(points)


def is_inside(point, min_corner, max_corner, offset=0.0):
    """
    Check if the point is inside the AABB with an offset.
    """
    offset_dir = max_corner - min_corner
    ext_min_corner = min_corner - offset_dir * offset
    ext_max_corner = max_corner + offset_dir * offset
    return np.all(point <= ext_max_corner) and np.all(point >= ext_min_corner)


def line_aabb_intersection(point, dir, min_corner, max_corner,offset=0.0):

    point = np.asarray(point, dtype=np.float64)
    dir = np.asarray(dir, dtype=np.float64)
    min_corner = np.asarray(min_corner, dtype=np.float64)
    max_corner = np.asarray(max_corner, dtype=np.float64)
    offset_dir = max_corner - min_corner
    min_corner = min_corner - offset_dir * offset
    max_corner = max_corner + offset_dir * offset

    t_min = np.full(3, -np.inf)
    t_max = np.full(3, np.inf)

    intersects=True
    for i in range(3):
        if dir[i] == 0:
            if point[i] < min_corner[i] or point[i] > max_corner[i]:
                intersects=False
            continue
            
        t1 = (min_corner[i] - point[i]) / dir[i]
        t2 = (max_corner[i] - point[i]) / dir[i]
        
        if t1 < t2:
            t_min[i] = t1
            t_max[i] = t2
        else:
            t_min[i] = t2
            t_max[i] = t1

    t_enter = np.max(t_min)
    t_exit = np.min(t_max)
    enter_point = point + t_enter * dir
    exit_point = point + t_exit * dir
    intersects = intersects and (t_enter <= t_exit)

    return intersects, enter_point, exit_point
def point_to_plane_distance(point, plane_point, plane_normal):
    point = np.array(point)
    plane_point = np.array(plane_point)
    plane_normal = np.array(plane_normal)
    return np.abs(np.dot(plane_normal, point - plane_point))



from itertools import product

def plane_aabb_intersection(point, dir, min_corner, max_corner,offset=0.0,tol = 1e-10):
    point = np.asarray(point, dtype=np.float64)
    dir = np.asarray(dir, dtype=np.float64)
    norm_dir = dir / np.linalg.norm(dir)
    min_corner = np.asarray(min_corner, dtype=np.float64)
    max_corner = np.asarray(max_corner, dtype=np.float64)
    offset_dir = max_corner - min_corner
    min_corner = min_corner - offset_dir * offset
    max_corner = max_corner + offset_dir * offset
    
    faces = [
        (0, min_corner[0]),  # x = min_corner.x
        (0, max_corner[0]),  # x = max_corner.x
        (1, min_corner[1]),  # y = min_corner.y
        (1, max_corner[1]),  # y = max_corner.y
        (2, min_corner[2]),  # z = min_corner.z
        (2, max_corner[2]),  # z = max_corner.z
    ]
    for axis, val in faces:
        if np.abs(norm_dir[axis]) < 1.0-tol:
            continue
        if np.abs(np.dot(dir, point) - np.dot(dir, [val if i == axis else 0 for i in range(3)])) < tol:
            if axis == 0:  # x = min/max
                y = [min_corner[1], max_corner[1]]
                z = [min_corner[2], max_corner[2]]
                vertices = np.array([[val, yi, zi] for yi, zi in product(y, z)])
            elif axis == 1:  # y = min/max
                x = [min_corner[0], max_corner[0]]
                z = [min_corner[2], max_corner[2]]
                vertices = np.array([[xi, val, zi] for xi, zi in product(x, z)])
            else:  # z = min/max
                x = [min_corner[0], max_corner[0]]
                y = [min_corner[1], max_corner[1]]
                vertices = np.array([[xi, yi, val] for xi, yi in product(x, y)])
            return vertices

    edges = [
        (min_corner, np.array([max_corner[0], min_corner[1], min_corner[2]])),
        (min_corner, np.array([min_corner[0], max_corner[1], min_corner[2]])),
        (min_corner, np.array([min_corner[0], min_corner[1], max_corner[2]])),
        (np.array([min_corner[0], max_corner[1], max_corner[2]]),max_corner),
        (np.array([max_corner[0], min_corner[1], max_corner[2]]),max_corner),
        (np.array([max_corner[0], max_corner[1], min_corner[2]]),max_corner),

        (np.array([min_corner[0], max_corner[1], min_corner[2]]),np.array([max_corner[0], max_corner[1], min_corner[2]])),
        (np.array([min_corner[0], max_corner[1], min_corner[2]]),np.array([min_corner[0], max_corner[1], max_corner[2]])),

        (np.array([min_corner[0], min_corner[1], max_corner[2]]),np.array([max_corner[0], min_corner[1], max_corner[2]])),
        (np.array([min_corner[0], min_corner[1], max_corner[2]]),np.array([min_corner[0], max_corner[1], max_corner[2]])),

        (np.array([max_corner[0], min_corner[1], min_corner[2]]),np.array([max_corner[0], max_corner[1], min_corner[2]])),
        (np.array([max_corner[0], min_corner[1], min_corner[2]]),np.array([max_corner[0], min_corner[1], max_corner[2]]))
    ]
    intersections = []
    for p1, p2 in edges:
        u = p2 - p1
        denom = np.dot(dir, u)
        if np.abs(denom) < tol:
            continue
        t = np.dot(dir, point - p1) / denom
        if 0 <= t <= 1:
            intersection = p1 + t * u
            intersections.append(intersection)

    unique_intersections = []
    for p in intersections:
        if not any(np.allclose(p, q, atol=tol) for q in unique_intersections):
            unique_intersections.append(p)
    
    return np.array(unique_intersections)
def save_mat_obj(filename,cpoint,radii,cedges,cfaces):
    n = len(cpoint)
    used = [False] * n
    
    for tri in cfaces:
        for v in tri:
            used[v] = True
    
    for edge in cedges:
        for v in edge:
            used[v] = True
    
    old_to_new = {}
    new_idx = 0
    for i in range(n):
        if used[i]:
            old_to_new[i] = new_idx
            new_idx += 1
    
    new_vertices = [cpoint[i] for i in range(n) if used[i]]
    new_radii = [radii[i] for i in range(n) if used[i]]
    new_triangles = [[old_to_new[v] for v in tri] for tri in cfaces]

    new_edges = [[old_to_new[v] for v in edge] for edge in cedges]
    
    with open(filename, 'w') as f:
        f.write(f'{len(new_vertices)} {len(new_edges)} {len(new_triangles)}\n')
        for vertex,radius in zip(new_vertices,new_radii):
            f.write(f'v {vertex[0]} {vertex[1]} {vertex[2]} {radius}\n')
        for edge in new_edges:
            f.write(f'e {edge[0]} {edge[1]}\n')
        for face in new_triangles:
            f.write(f'f {face[0]} {face[1]} {face[2]}\n')
    f.close()

def save_obj(obj_filename,vertices,edges,triangles):
    with open(obj_filename, 'w') as obj_file:
        for vertex in vertices:
            obj_file.write(f"v {vertex[0]} {vertex[1]} {vertex[2]} \n")
        if edges is not None:
            for edge in edges:
                if len(edge)==2:
                    obj_file.write(f"l {edge[0]+1} {edge[1]+1}\n")
        for triangle in triangles:
            #triangle=np.unique(triangle)
            if len(triangle)==3:
                obj_file.write(f"f {triangle[0]+1} {triangle[1]+1} {triangle[2]+1}\n")
            elif len(triangle)==4:
                obj_file.write(f"f {triangle[0]+1} {triangle[1]+1} {triangle[2]+1} {triangle[3]+1}\n")
def save_obj_with_normals(obj_filename,vertices,edges,triangles,normals):
    with open(obj_filename, 'w') as obj_file:
        for vertex in vertices:
            obj_file.write(f"v {vertex[0]} {vertex[1]} {vertex[2]} \n")
        for normal in normals:
            obj_file.write(f"n {normal[0]} {normal[1]} {normal[2]} \n")
        if edges is not None:
            for edge in edges:
                if len(edge)==2:
                    obj_file.write(f"l {edge[0]+1} {edge[1]+1}\n")
        for triangle in triangles:
            #triangle=np.unique(triangle)
            if len(triangle)==3:
                obj_file.write(f"f {triangle[0]+1} {triangle[1]+1} {triangle[2]+1}\n")
            elif len(triangle)==4:
                obj_file.write(f"f {triangle[0]+1} {triangle[1]+1} {triangle[2]+1} {triangle[3]+1}\n")

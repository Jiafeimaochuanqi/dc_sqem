import numpy as np
from typing import Dict, Tuple, Optional

def compute_face_normals_basic(vertices, faces):
    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]
    e1 = v1 - v0
    e2 = v2 - v0
    normals = np.cross(e1, e2)
    return normals


def compute_face_normals_normalized(vertices, faces):
    normals = compute_face_normals_basic(vertices, faces)
    lengths = np.linalg.norm(normals, axis=1, keepdims=True)
    lengths = np.where(lengths > 0, lengths, 1.0)
    unit_normals = normals / lengths
    areas = lengths.flatten() / 2.0
    return unit_normals, areas


def triangle_point_from_bary(v0, v1, v2, bary):
    """
    bary = (u,v,w), point = u*v0 + v*v1 + w*v2
    """
    bary = np.asarray(bary, dtype=float)
    return bary[0] * v0 + bary[1] * v1 + bary[2] * v2


def sort_edge(i, j):
    return (i, j) if i < j else (j, i)


def extract_edges(face):
    edges = []
    n = len(face)
    for i in range(n):
        u = face[i]
        v = face[(i+1) % n]
        edges.append(tuple(sorted([u,v])))
        # edges.append((min(u, v), max(u, v)))
    return edges

def extract_unique_edges(faces: np.ndarray):
    edge_set = set()
    for face in faces:
        edges=extract_edges(face)
        edge_set.update(edges)
    edges = sorted(edge_set)
    return edges

def generate_interior_lattice_barycentric(max_m=120):
    cache = {}
    for m in range(1, max_m + 1):
        pts = []
        for a in range(1, m):
            for b in range(1, m - a):
                c = m - a - b
                if c >= 1:
                    pts.append([a / m, b / m, c / m])

        pts = np.array(pts, dtype=float)
        if len(pts) > 0:
            center = np.array([1/3, 1/3, 1/3], dtype=float)
            dist = np.linalg.norm(pts - center[None, :], axis=1)
            order = np.argsort(dist)
            pts = pts[order]

        cache[m] = pts
    return cache


_INTERIOR_LATTICE_CACHE = generate_interior_lattice_barycentric(max_m=200)


def interior_barycentric_lattice_for_m(m: int) -> np.ndarray:
    if m < 1:
        raise ValueError("m must be >= 1")
    if m not in _INTERIOR_LATTICE_CACHE:
        raise ValueError(f"m={m} exceeds current cache size")
    return _INTERIOR_LATTICE_CACHE[m]


def total_sample_count_for_m(num_vertices: int, num_edges: int, num_faces: int, m: int,feature_edge_num: int,total_feature_edge_interior_sample_num: int) -> int:
    if m < 1:
        raise ValueError("m must be >= 1")

    edge_interior = max(0, m - 1)
    face_interior = max(0, (m - 1) * (m - 2) // 2)

    total = num_vertices + (num_edges-feature_edge_num) * edge_interior+total_feature_edge_interior_sample_num + num_faces * face_interior
    return int(total)


def choose_m_from_target_count(num_vertices: int, num_edges: int, num_faces: int, target_count: int,feature_edge_num: int,total_feature_edge_interior_sample_num: int) -> int:
    if target_count <= num_vertices:
        return 1

    m = 1
    while total_sample_count_for_m(num_vertices, num_edges, num_faces, m,feature_edge_num,total_feature_edge_interior_sample_num) < target_count:
        m += 1

    if m == 1:
        return 1

    # c0 = total_sample_count_for_m(num_vertices, num_edges, num_faces, m - 1,feature_edge_num,total_feature_edge_interior_sample_num)
    # c1 = total_sample_count_for_m(num_vertices, num_edges, num_faces, m,feature_edge_num,total_feature_edge_interior_sample_num)

    # if abs(c0 - target_count) <= abs(c1 - target_count):
    #     return m - 1
    return m

def sample_mesh_uniform_lattice(
    vertices: np.ndarray,
    faces: np.ndarray,
    target_count: int,
    feature_edge_num:int,
    total_feature_edge_interior_sample_num:int,
    m: Optional[int] = None,
):
    
    vertices = np.asarray(vertices, dtype=float)
    faces = np.asarray(faces, dtype=int)

    if vertices.ndim != 2 or vertices.shape[1] != 3:
        raise ValueError("vertices must be (V,3)")
    if faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError("faces must be (F,3)")
    if target_count < 0:
        raise ValueError("target_count must be >= 0")

    face_normals, face_areas = compute_face_normals_normalized(vertices, faces)
    edges = extract_unique_edges(faces)

    V = vertices.shape[0]
    E = len(edges)
    F = faces.shape[0]

    if m is None:
        m = choose_m_from_target_count(V, E, F, target_count,feature_edge_num,total_feature_edge_interior_sample_num)

    if m < 1:
        raise ValueError("m must be >= 1")

    edge_interior_count = max(0, m - 1)
    face_interior_count = max(0, (m - 1) * (m - 2) // 2)

    bary_template = interior_barycentric_lattice_for_m(m)  # (k,3), same for every face
    assert bary_template.shape[0] == face_interior_count

    edge_sample_count: Dict[Tuple[int, int], int] = {
        e: edge_interior_count for e in edges
    }

    triangle_interior_samples: Dict[int, np.ndarray] = {}
    triangle_interior_normals: Dict[int, np.ndarray] = {}
    triangle_barycentric: Dict[int, np.ndarray] = {}

    all_samples_list = []
    all_normals_list = []
    all_face_ids_list = []
    all_bary_list = []

    for tid, (i0, i1, i2) in enumerate(faces):
        v0 = vertices[i0]
        v1 = vertices[i1]
        v2 = vertices[i2]
        nrm = face_normals[tid]

        if face_interior_count == 0:
            pts = np.zeros((0, 3), dtype=float)
            nrms = np.zeros((0, 3), dtype=float)
            bary = np.zeros((0, 3), dtype=float)
        else:
            bary = bary_template.copy()
            pts = np.array(
                [triangle_point_from_bary(v0, v1, v2, b) for b in bary],
                dtype=float
            )
            nrms = np.repeat(nrm[None, :], face_interior_count, axis=0)

        triangle_interior_samples[tid] = pts
        triangle_interior_normals[tid] = nrms
        triangle_barycentric[tid] = bary

        if len(pts) > 0:
            all_samples_list.append(pts)
            all_normals_list.append(nrms)
            all_face_ids_list.append(np.full((pts.shape[0],), tid, dtype=np.int32))
            all_bary_list.append(bary)

    if len(all_samples_list) == 0:
        all_samples = np.zeros((0, 3), dtype=float)
        all_normals = np.zeros((0, 3), dtype=float)
        all_face_ids = np.zeros((0,), dtype=np.int32)
        all_barycentric = np.zeros((0, 3), dtype=float)
    else:
        all_samples = np.vstack(all_samples_list)
        all_normals = np.vstack(all_normals_list)
        all_face_ids = np.concatenate(all_face_ids_list)
        all_barycentric = np.vstack(all_bary_list)

    estimated_total_count = total_sample_count_for_m(V, E, F, m,feature_edge_num,total_feature_edge_interior_sample_num)

    return {
        "m": int(m),
        "estimated_total_count": int(estimated_total_count),
        "num_vertices": int(V),
        "num_edges": int(E),
        "num_faces": int(F),
        "edge_interior_count_per_edge": int(edge_interior_count),
        "face_interior_count_per_face": int(face_interior_count),
        "face_normals": face_normals,
        "face_areas": face_areas,
        "edges": edges,
        "edge_sample_count": edge_sample_count,
        "triangle_interior_samples": triangle_interior_samples,
        "triangle_interior_normals": triangle_interior_normals,
        "triangle_barycentric": triangle_barycentric,
        "all_samples": all_samples,
        "all_normals": all_normals,
        "all_face_ids": all_face_ids,
        "all_barycentric": all_barycentric,
    }
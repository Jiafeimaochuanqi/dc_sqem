import numpy as np
import pyvista as pv

def create_lines_pyvista_data(sources,targets):
    all_points = np.vstack((np.array(sources), np.array(targets)))
    n=len(sources)
    lines = []
    for i in range(n):
        lines.append(2)
        lines.append(i)
        lines.append(i + n)

    line_mesh = pv.PolyData(all_points, lines=np.array(lines))
    return line_mesh
def create_normals_pyvista_data(sources,normals,length=0.01):
    targets=sources+normals*length
    return create_lines_pyvista_data(sources,targets)
def create_plane_pyvista_data(point,normal,min_corner,max_corner):

    a, b, c = normal
    d = -np.dot(normal, point)

    x = [min_corner[0], max_corner[0]]
    y = [min_corner[1], max_corner[1]]
    z = [min_corner[2], max_corner[2]]
    bbox_vertices = np.array([[i, j, k] for i in x for j in y for k in z])

    def project_to_plane(pt, a, b, c, d):
        t = -(a*pt[0] + b*pt[1] + c*pt[2] + d) / (a**2 + b**2 + c**2)
        return pt + t * np.array([a, b, c])

    projected_pts = np.array([project_to_plane(pt, a, b, c, d) for pt in bbox_vertices])

    if not np.allclose(normal, [1, 0, 0]):
        u = np.array([1, 0, 0])
    else:
        u = np.array([0, 1, 0])
    u = u - np.dot(u, normal) * normal
    u = u / np.linalg.norm(u)
    v = np.cross(normal, u)
    v = v / np.linalg.norm(v)

    u_coords = np.dot(projected_pts - point, u)
    v_coords = np.dot(projected_pts - point, v)
    u_min, u_max = u_coords.min(), u_coords.max()
    v_min, v_max = v_coords.min(), v_coords.max()

    p0 = point + u_min * u + v_min * v
    p1 = point + u_max * u + v_min * v
    p2 = point + u_max * u + v_max * v
    p3 = point + u_min * u + v_max * v
    vertices = np.array([p0, p1, p2, p3])
    faces = np.array([3, 0, 1, 2,  3, 0, 2, 3])
    plane = pv.PolyData(vertices, faces)
    return plane
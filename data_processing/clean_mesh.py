import open3d as o3d
import numpy as np
import trimesh
import os
import glob
import multiprocessing as mp
from multiprocessing import Pool
INPUT_DIR = '/mnt/WindowsH/datasets/abc_test_2048_100'
# OUTPUT_DIR= 'E:/coding/python/datasets/scaled/ABC'
def clean_mesh(input_path):
    mesh = o3d.io.read_triangle_mesh(input_path)

    mesh_merged = mesh.merge_close_vertices(eps=1e-3)
    print(f"after merger: {len(mesh_merged.vertices)} points, {len(mesh_merged.triangles)} faces")

    mesh_merged.remove_unreferenced_vertices()
    mesh_merged.remove_degenerate_triangles()
    print(f"after remove degenerate: {len(mesh_merged.vertices)} points, {len(mesh_merged.triangles)} faces")

    o3d.io.write_triangle_mesh(input_path, mesh_merged)

if __name__ == '__main__':
    with Pool(mp.cpu_count()) as p:
        print(glob.glob(os.path.join(INPUT_DIR, '*')))
        p.map(clean_mesh, glob.glob(os.path.join(INPUT_DIR, '*')))

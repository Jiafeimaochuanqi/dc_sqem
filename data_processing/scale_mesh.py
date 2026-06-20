import numpy as np
import os
import glob
import igl
import multiprocessing as mp
from multiprocessing import Pool
INPUT_DIR = '/mnt/WindowsH/datasets/original_mesh/other'
OUTPUT_DIR= '/mnt/WindowsH/datasets/scaled/other'
def normalize_vertices(vertices,center=None,diag_length=None):
    if center is None:
        min_corner=np.min(vertices, axis=0)
        max_corner=np.max(vertices, axis=0)
        center = (min_corner + max_corner) / 2
    vertices_centered =vertices -center
    if diag_length is None:
        diag = (np.max(vertices_centered, axis=0, keepdims=True) -
                np.min(vertices_centered, axis=0, keepdims=True))
        diag_length=np.max(diag)
    vertices_normalized =vertices_centered/ diag_length
    min_corner=np.min(vertices_normalized, axis=0)
    max_corner=np.max(vertices_normalized, axis=0)
    print(min_corner)
    print(max_corner)
    return vertices_normalized,center,diag_length
def scale_mesh(input_path):
    V,TC,N, F,FTC,FN = igl.readOBJ(input_path)
    V_NEW,_,_=normalize_vertices(V)
    filename = os.path.basename(input_path)
    name_without_extension, extension = os.path.splitext(filename)
    out_path=os.path.join(OUTPUT_DIR,name_without_extension+".obj")
    igl.writeOBJ(out_path, V_NEW, F)
if __name__ == '__main__':
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with Pool(mp.cpu_count()) as p:
        print(glob.glob(os.path.join(INPUT_DIR, '*')))
        p.map(scale_mesh, glob.glob(os.path.join(INPUT_DIR, '*')))

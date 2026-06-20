
import numpy as np
#from .octree import OctreeNode, Octree
from cpp_module.GRAPH_GROUPING import OctreeNode, Octree
import os
#from sklearn.cluster import DBSCAN
from cpp_module.GRAPH_GROUPING import DBSCAN
from .sqef import *
from .mesh_utils import *
from .neural_utils import * 

import pickle
from .MeshConnectivityAnalyzer import *
import time
from cpp_module.GRAPH_GROUPING import SDFProjectorFunc,batch_compute_face_udf_parallel
g_child_2_vtk_order=np.array([0,4,6,2,1,5,7,3])
g_child_order = np.array([
    [0, 0, 0],
    [0, 0, 1],
    [0, 1, 0],
    [0, 1, 1],
    [1, 0, 0],
    [1, 0, 1],
    [1, 1, 0],
    [1, 1, 1]
],dtype=int)

def run_leaf_node_mat_search_from_npz(octree, data):
    spoints = np.asarray(data["spoints"], dtype=np.float32)
    sns = np.asarray(data["ns"], dtype=np.float32)

    sp_feature_mask = data.get("sp_feature_mask", None)
    if sp_feature_mask is None:
        sp_feature_mask = np.zeros(len(spoints), dtype=bool)
    else:
        sp_feature_mask = np.asarray(sp_feature_mask, dtype=bool)

    patch_ids = data.get("patchIds", None)
    if patch_ids is None:
        patch_ids = np.zeros(len(spoints), dtype=np.int32)
    else:
        patch_ids = np.asarray(patch_ids, dtype=np.int32)

    sp_convex_feature_mask = data.get("sp_convex_feature_mask", None)
    if sp_convex_feature_mask is None:
        sp_convex_feature_mask = np.zeros(len(spoints), dtype=bool)
    else:
        sp_convex_feature_mask = np.asarray(sp_convex_feature_mask, dtype=bool)

    dihedral_angle_threshold = data.get("angle_threshold_degrees", None)
    if dihedral_angle_threshold is None:
        dihedral_angle_threshold = 135.0
    else:
        dihedral_angle_threshold = float(dihedral_angle_threshold)

    min_dihedral_angle_degree = data.get("min_dihedral_angle_degree", None)
    if min_dihedral_angle_degree is None:
        min_dihedral_angle_degree = 0.0
    else:
        min_dihedral_angle_degree = float(min_dihedral_angle_degree)

    return octree.run_leaf_node_mat_search(
        spoints,
        sns,
        sp_feature_mask,
        patch_ids,
        sp_convex_feature_mask,
        dihedral_angle_threshold,
        min_dihedral_angle_degree,
    )

def create_octree_by_sqef(args,data,cloud_sdf_grad_mask_func,sampling_depth=1)->Octree: 
    max_depth=args.max_depth
    margin_threshold=args.margin_threshold
    octree = Octree(max_depth=max_depth,
                    min_corner=np.array([-0.525, -0.519, -0.52]),
                    max_corner=np.array([0.501, 0.574, 0.532]),
                    sampling_depth=sampling_depth)
    octree.angle_threshold0=args.angle_threshold0
    octree.angle_threshold=args.angle_threshold
    spoints=data['spoints']
    
    octree.m_root_node.inside_surface_point_ids=np.arange(len(spoints))
    #octree.sdf_kdtree = KDTree(spoints)
    sp_convex_feature_mask=data.get('sp_convex_feature_mask',None)
    if sp_convex_feature_mask is not None:
        convex_feature_points=spoints[sp_convex_feature_mask]
        octree.convex_feature_points=convex_feature_points
        octree.m_root_node.inside_convex_feature_ids=np.arange(len(convex_feature_points))
    convex_corner_points=data.get('convex_corner_points',None)
    if convex_corner_points is not None:
        octree.convex_corner_points=convex_corner_points
    edge_lengths=data.get('edge_lengths',None)

    if edge_lengths is None:
        octree.position_eps=3.0*octree.box_edge_length
    else:
        octree.position_eps=3.0*max(edge_lengths)
    
    octree.grow_by_max_depth(spoints,cloud_sdf_grad_mask_func,margin_threshold)

    run_leaf_node_mat_search_from_npz(octree, data)
    
    # if args.debug:
    #     with open(args.pkl_path, 'wb') as f:
    #         pickle.dump(octree, f)
    #         print(f"create octree {args.pkl_path}:{len(octree.m_leaf_nodes)} nodes done!")
    return octree

def build_debug_octree_output(octree):
    filtered_mesh_f_ids = octree.filtered_mesh_f_ids
    all_fids = np.arange(len(octree.m_mesh_f))
    excluded_mesh_f_ids = np.setdiff1d(all_fids, filtered_mesh_f_ids)
    excluded_mesh_f_ids = np.sort(excluded_mesh_f_ids)
    data = octree.visualize_octree_cells()
    data["excluded_mesh_f_ids"] = excluded_mesh_f_ids
    return data

def extract_octree_by_sqef(data, args, sampling_depth=1):
    start_time = time.time()

    meshpath = os.path.join(args.datasets_dir, args.dataset + ".obj")
    filename = os.path.basename(meshpath)
    name_without_extension, _ = os.path.splitext(filename)

    os.makedirs(args.output_dir, exist_ok=True)


    pkl_path = os.path.join(
        args.output_dir,
        name_without_extension + ".depth" + str(args.max_depth) + ".sqef.initial.pkl",
    )
    args.pkl_path = pkl_path

    # if os.path.exists(pkl_path):
    #     with open(pkl_path, "rb") as f:
    #         octree = pickle.load(f)
    # else:
    #     octree = create_octree_by_sqef(
    #         args,
    #         data,
    #         cloud_sdf_grad_mask_func,
    #         batch_size,
    #         sampling_depth,
    #     )
    spoints = data["spoints"]
    sp_normals = data["ns"]
    sp_feature_mask = data.get("sp_feature_mask", None)
    if sp_feature_mask is None:
        sp_feature_mask = np.array([], dtype=bool)

    projFunc = SDFProjectorFunc(
        spoints.astype(np.float32),
        sp_normals.astype(np.float32),
        sp_feature_mask,
    )
    octree = create_octree_by_sqef(
            args,
            data,
            projFunc,
            sampling_depth,
        )
    
    if args.debug:
        step1_time = time.time()
        print(f"create octree done!cost:{step1_time - start_time:.6f} sec")

    centroids_coords = octree.get_centroids_of_leaf_nodes().astype(np.float32)

    
    sp_convex_feature_mask = data.get("sp_convex_feature_mask", None)
    if sp_convex_feature_mask is None:
        sp_convex_feature_mask = np.array([], dtype=bool)
    octree.solve_and_locate_vertices(
        spoints,
        sp_convex_feature_mask,
        projFunc,
        centroids_coords,
        args.singular_value_threshold,
    )
    if args.debug:
        step2_time = time.time()
        print(f"solve_and_locate_vertices done!cost:{step2_time - step1_time:.6f} sec")

    octree.build_mesh_and_topology()
    if args.debug:
        step3_time = time.time()
        print(f"build_mesh_and_topology done!cost:{step3_time - step2_time:.6f} sec")
        print(f"generate_mesh:{len(octree.m_mesh_f)} faces")

    octree.compute_face_udf_and_triangulate(projFunc)
    if args.debug:
        step4_time = time.time()
        print(f"compute_face_udf_and_triangulate done!cost:{step4_time - step3_time:.6f} sec")

    octree.clean_and_post_process()
    if args.debug:
        step5_time = time.time()
        print(f"clean_and_post_process done!cost:{step5_time - step4_time:.6f} sec")

    if args.debug:
        data = build_debug_octree_output(octree)
        return data, octree

    return None, octree
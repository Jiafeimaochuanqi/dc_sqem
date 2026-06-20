import numpy as np
from typing import List, Tuple, Optional
from .sqef import *
from .linklist import *
from .priorityqueue import MulPriorityQueue
from pyigl.sharp_feature_detector import EdgeType
import copy
import multiprocessing as mp
from functools import partial
import numpy as np
from functools import partial
from scipy.spatial.distance import cdist
#from sklearn.cluster import DBSCAN
from cpp_module.GRAPH_GROUPING import DBSCAN
from .MeshConnectivityAnalyzer import *
from .mesh_utils import indices_in_bbox
from .plane_group import compute_groups,CPlaneGrouping
from .neural_utils import query_sdf_and_sn_and_proj_and_feature_mask
from cpp_module.GRAPH_GROUPING import MulPriorityQueueInt,is_cylindrical_tinf_node
from cpp_module.GRAPH_GROUPING import PointCloudRVD,KNNSearch,batch_compute_face_udf_parallel

# from .cylinder_util import *
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

edges_table = [(0, 1), (0, 2), (2, 3), (3, 1),
         (4, 5), (4, 6), (6, 7), (7, 5),
         (0, 4), (1, 5), (2, 6), (3, 7)]

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


cellProcFaceMask = np.array([
    [0, 4, 0], [1, 5, 0], [2, 6, 0], [3, 7, 0],
    [0, 2, 1], [4, 6, 1], [1, 3, 1], [5, 7, 1],
    [0, 1, 2], [2, 3, 2], [4, 5, 2], [6, 7, 2]
])


cellProcEdgeMask = np.array([
    [0, 1, 2, 3, 0],
    [4, 5, 6, 7, 0],
    [0, 4, 1, 5, 1],
    [2, 6, 3, 7, 1],
    [0, 2, 4, 6, 2],
    [1, 3, 5, 7, 2]
])


faceProcFaceMask = np.array([
    [[4, 0, 0], [5, 1, 0], [6, 2, 0], [7, 3, 0]],
    [[2, 0, 1], [6, 4, 1], [3, 1, 1], [7, 5, 1]],
    [[1, 0, 2], [3, 2, 2], [5, 4, 2], [7, 6, 2]]
])

faceProcEdgeMask = np.array([
    [[1, 4, 0, 5, 1, 1], [1, 6, 2, 7, 3, 1], [0, 4, 6, 0, 2, 2], [0, 5, 7, 1, 3, 2]],
    [[0, 2, 3, 0, 1, 0], [0, 6, 7, 4, 5, 0], [1, 2, 0, 6, 4, 2], [1, 3, 1, 7, 5, 2]],
    [[1, 1, 0, 3, 2, 0], [1, 5, 4, 7, 6, 0], [0, 1, 5, 0, 4, 1], [0, 3, 7, 2, 6, 1]]
])


edgeProcEdgeMask = np.array([
    [[3, 2, 1, 0, 0], [7, 6, 5, 4, 0]],
    [[5, 1, 4, 0, 1], [7, 3, 6, 2, 1]],
    [[6, 4, 2, 0, 2], [7, 5, 3, 1, 2]]
])
g_child_2_vtk_order=np.array([0,4,6,2,1,5,7,3])


class OctreeNode:
    def __init__(self, min_corner_id: np.ndarray, max_corner_id: np.ndarray, parent=None, depth=0):
        self.id=-1
        self.leaf_node_index=-1
        self.m_parent = parent
        self.inside_surface_point_ids=[]
        self.inside_convex_feature_ids=[]
        self.strict_inside_convex_feature_ids=[]
        self.m_children: List[Optional[OctreeNode]] = []
        self.m_depth = depth
        self.m_min_corner_id = min_corner_id.astype(int)
        self.m_max_corner_id = max_corner_id.astype(int)

        self.m_centroid_id = (self.m_min_corner_id + self.m_max_corner_id) // 2
        self.m_sample_grids = None
        self.m_edge_span = max_corner_id[0] - min_corner_id[0]
        self.m_vertex_idx = -1
        self.quote_leaf_node_index=-1
        self.quote_vertex_id=-1
        self.be_quoted_leaf_node_indexes=[]
        self.be_quoted_vertex_ids=[]

        self.cell_sample_mask=None
        self.groups=None
        self.m_node_type = "empty"
        self.m_node_cell_type=0#0 for empty cell, 1 for virtual cell, 2 for real cell
        self.m_reconstruct_point = None

        self.is_cylinder=False
        
        

    def subdivide(self):
        if len(self.m_children)>0:
            raise ValueError("Node already has children!")

        child_edge_span = self.m_edge_span // 2
        max_corner_id_offset = np.array([child_edge_span] * 3)
        parent_id=self.id
        for child_index,offset in enumerate(g_child_order):
            child_id=parent_id * 8 + 1 + child_index
            child_min_corner_id = self.m_min_corner_id + offset * child_edge_span
            child_max_corner_id = child_min_corner_id + max_corner_id_offset
            self.m_children.append(OctreeNode(child_min_corner_id, child_max_corner_id, parent=self, depth=self.m_depth + 1))
            self.m_children[child_index].id=child_id

    def generate_sample_grids(self, sampling_pattern: np.ndarray) -> np.ndarray:
        return sampling_pattern + self.m_min_corner_id
    
    def set_is_cylinder(self, value: bool):
        self.is_cylinder = bool(value)

    def set_groups(self, groups):
        self.groups = groups

    def set_drop_stage(self, stage: int):
        self.drop_stage = int(stage)

    def set_decision(self, pid: int, sid: int):
        self.decision_pid = int(pid)
        self.decision_sid = int(sid)

def generate_linespace_points(lower_left,upper_right,n):
    x_axis = np.linspace(lower_left[0], upper_right[0], n)
    y_axis = np.linspace(lower_left[1], upper_right[1], n)
    z_axis = np.linspace(lower_left[2], upper_right[2], n)
    X, Y, Z = np.meshgrid(x_axis, y_axis, z_axis,indexing="ij")
    vertices = np.vstack((X.reshape(-1), Y.reshape(-1), Z.reshape(-1))).transpose()
    return vertices

def process_node_chunk_ids(chunk_range, octree,spoints):

    start_idx, end_idx = chunk_range
    subdivide_indices = []
    drop_indices = []
    mask_updates = {}
    group_updates = {}
    inside_surface_point_ids_updates={}
    for i in range(start_idx, end_idx):
        node = octree.m_leaf_nodes[i]
        if node.m_edge_span <= 1:
            raise ValueError("assigned depth is not enough!")
        if node.id==1062177:
            pass
        
        flag = octree.pre_check_node_build(node,spoints)
        
        mask_updates[i] = node.cell_sample_mask
        
        if flag:
            if len(node.inside_surface_point_ids)>0:
                inside_surface_point_ids_updates[i]=node.inside_surface_point_ids
            if node.m_depth < octree.m_max_depth:
                subdivide_indices.append(i)
            elif node.m_depth==octree.m_max_depth:
                if node.groups is not None:
                    group_updates[i]=node.groups
                
        else:
            drop_indices.append(i)
    
    return subdivide_indices, drop_indices, mask_updates,group_updates,inside_surface_point_ids_updates

def find_central_indices(vertices, groups,k=1):
    vertices = np.array(vertices)
    central_indices = []
    new_groups=[]
    for group in groups:
        if len(group) == 0:
            #central_indices.append(None)
            continue
        elif len(group) == 1:
            new_groups.append([len(central_indices)])
            central_indices.append(group[0])
            continue

        group_points = vertices[group]

        distances = cdist(group_points, group_points)
        total_distances = distances.sum(axis=1)
        if k==1:
            min_index_in_group = np.argmin(total_distances)
            central_index = group[min_index_in_group]
            new_groups.append([len(central_indices)])
            central_indices.append(central_index)
        else:
            actual_k = min(k,len(group))
            k_min_index_in_group=get_k_smallest_indices(total_distances,actual_k)
            k_central_index = [group[i] for i in k_min_index_in_group]
            new_groups.append(np.arange(len(central_indices),len(k_central_index)+len(central_indices)))
            central_indices.extend(k_central_index)  
    return central_indices,new_groups

class Octree:
    def __init__(self, max_depth: int, min_corner: np.ndarray = np.full(3, -1.0), max_corner: np.ndarray = np.full(3, 1.0), sampling_depth=1):
        self.dtype=np.float32
        #the max depth of the octree
        self.m_max_depth = max_depth
        #the min corner
        self.m_min_corner = min_corner
        #the max corner
        self.m_max_corner = max_corner
        #the subdivision depth for sampling in a cell
        self.m_sampling_depth = sampling_depth
        #the sampling resolution on each edge
        self.m_capacity_per_dim = int(2 ** (max_depth + sampling_depth)) + 1
        self.resolution= int((self.m_capacity_per_dim - 1) / (2 ** sampling_depth))
        self.occupied = np.zeros((self.resolution, self.resolution, self.resolution), dtype=bool)
        self.occupied_cell=np.full((self.resolution, self.resolution, self.resolution), -1)
        self.occupied_cell_inverse={}
        print(f"Resolution: {self.resolution}")
        print(f"Samples per cell: {(2*sampling_depth + 1) ** 3}")

        root_min_corner_id = np.zeros(3, dtype=int)
        root_max_corner_id = np.full(3, self.m_capacity_per_dim - 1, dtype=int)
        self.m_root_node = OctreeNode(root_min_corner_id, root_max_corner_id)
        # compute the unit edge length
        self.m_unit_edge_length = (max_corner[0] - min_corner[0]) / (self.m_capacity_per_dim - 1)
        self.box_edge_length = self.m_unit_edge_length*(2 ** sampling_depth)
        #the current leaf nodes
        self.m_leaf_nodes = [self.m_root_node]
        self.m_leaf_nodes_adjacency=[]
        self.m_leaf_node_pos_types=[]
        self.drop_nodes=[]
        self.rvd_graph_edges=[]
        #the existence of grids, indexed by the global id
        self.m_grid_existence = np.zeros(self.m_capacity_per_dim**3, dtype=bool)
        # the map from global id to sampling id
        self.m_gid2sid = np.full(self.m_capacity_per_dim**3,-1)
        
        # generate sampling pattern
        self.m_sampling_pattern = self.generate_sample_pattern(2 ** sampling_depth + 1).astype(int)
        self.m_sampling_patterns_for_depth=[None for i in range(self.m_max_depth+1)]

        self.box_edge_length_for_depth=[0 for i in range(self.m_max_depth+1)]
        for i in range(self.m_max_depth+1):
            self.m_sampling_patterns_for_depth[i]=self.generate_sample_pattern_of_depth(i)
        for i in range(self.m_max_depth+1):
            self.box_edge_length_for_depth[i]=self.compute_box_length_of_depth(i)
        #print("m_sampling_pattern:\n",self.m_sampling_pattern)


        self.m_sample_coords=[]
        self.m_sample_mesh_sns=[]
        self.m_sample_mesh_sdfs=[]
        self.m_sample_mesh_projs=[]
        self.m_sample_mesh_point_ids=[]
        self.m_sample_mesh_masks=[]


        #mesh vertices
        self.m_mesh_v = []
        self.m_mesh_v_radii = []
        #cell leaf node index of mesh vertices
        self.m_mesh_v_to_leaf_node_indexes=[]
        self.m_mesh_v_simulated=[]

        self.cell_samples=[]
        self.mesh_samples=[]
        
        #the point types of mesh vertices #0 for CORNER, 1 for EDGE, 2 for PLANE
        self.m_v_type = []
        #mesh faces (quad or triangle)
        self.m_mesh_f = []

        self.m_mesh_f_unique = []
        self.m_mesh_f_tri=[]
        self.m_mesh_f_area2=[]

        self.m_mesh_f_udfs=[]
        #cell id corresponding to the mesh faces
        self.m_mesh_cell_id=[]
        #cell type corresponding to the mesh faces
        self.m_mesh_cell_type = []

        self.cylinder_node_ids=[]
        self.alone_edges=set()

        self.feature_flags=[]#medial axis point feature flags

        #self.sp_normals=[]#surface point normals
        self.cell_quad2face_id={}
        self.cell_quad2face_id_inverse={}
        self.cell_quad2face={}
        self.cavity_face_ids=[]
        self.cavity_leaf_node_indexes=[]
        self.cavity_vertex_indexes=[]

        self.face_to_edge={}
        self.edge_to_faces={}
        self.vertex_to_edge={}
        self.deleted_face_flag=None
        self.static_face_flag=None
        self.first_deleted_face_flag=None
        self.second_deleted_face_flag=None
        self.third_deleted_face_flag=None

        self.cavity_flags=None

        self.quote_vertex_ids=[]
        self.be_quoted_vertex_ids=[]

        self.quote_leaf_node_indexes=[]
        self.be_quoted_leaf_node_indexes=[]



        self.cavity_face_ids=None
        self.cavity_flags=None
        
        self.origin_cavity_leaf_node_indexes=None
        self.origin_cavity_vertex_indexes=None
        self.origin_cavity_face_ids=None
        self.origin_cavity_flags=None

        self.face2cavity=None
        self.vertex2cavity=None
        self.edge2cavity=None
        self.origin_face2cavity=None
        self.origin_edge2cavity=None
        self.origin_vertex2cavity=None

        #add
        self.convex_feature_points=[]
        self.convex_corner_points=[]
        self.position_eps=None
        self.norm_eps=1e-3
        #add
        self.node_contain_corner_ids=[]
        self.strict_node_contain_corner_ids=[]
        self.adj_contain_outer_corner_flags=[]
        self.adj_corner_node_id=[]
        #add
        self.deleted_face_map={}
    
    def run_leaf_node_mat_search(
        self,
        spoints,
        sns,
        sp_feature_mask,
        patch_ids,
        sp_convex_feature_mask,
        dihedral_angle_threshold=135.0,
        min_dihedral_angle_degree=0.0,
    ):
        def _flatten_records_from_nodes(nodes):
            cell_seed_nums = []
            nearest_cloud_ids_flat = []
            cell_node_ids = []

            for node in nodes:
                _, _, _, _, _, shape_point_ids, _ = self.get_group_data(node)
                shape_point_ids = np.asarray(shape_point_ids, dtype=np.int64)

                if len(shape_point_ids) == 0:
                    node.groups = []
                    continue

                cell_seed_nums.append(len(shape_point_ids))
                nearest_cloud_ids_flat.extend(shape_point_ids.tolist())
                cell_node_ids.append(int(node.id))

            if len(cell_seed_nums) == 0:
                return (
                    np.zeros((0,), dtype=np.int32),
                    np.zeros((0,), dtype=np.int32),
                    np.zeros((0,), dtype=np.int64),
                )

            return (
                np.asarray(cell_seed_nums, dtype=np.int32),
                np.asarray(nearest_cloud_ids_flat, dtype=np.int32),
                np.asarray(cell_node_ids, dtype=np.int64),
            )

        def _clusters_to_spoint_ids(clusters, shape_point_ids):
            spoint_ids_cluster = []

            for cluster in clusters:
                spoint_ids = set()
                for cid in cluster:
                    spoint_ids.add(int(shape_point_ids[cid]))
                spoint_ids_cluster.append(list(spoint_ids))

            return spoint_ids_cluster

        spoints = np.asarray(spoints, dtype=np.float32)
        sns = np.asarray(sns, dtype=np.float32)
        sp_feature_mask = np.asarray(sp_feature_mask, dtype=bool)
        patch_ids = np.asarray(patch_ids, dtype=np.int64)
        sp_convex_feature_mask = np.asarray(sp_convex_feature_mask, dtype=bool)

        rvd = PointCloudRVD(spoints, sns, sp_feature_mask)
        knn = KNNSearch(spoints, 16)

        if len(self.convex_feature_points) > 0:
            rvd.buildKnnGraph(
                knn,
                26,
                True,
                180.0 - dihedral_angle_threshold - 1.0,
                min((180.0 - min_dihedral_angle_degree) / 2.0 + 0.1, 90.0),
                1,
            )
        else:
            rvd.buildKnnGraph(
                knn,
                8,
                True,
                180.0 - dihedral_angle_threshold - 1.0,
                min((180.0 - min_dihedral_angle_degree) / 2.0 + 0.1, 90.0),
                1,
            )

        surface_graph_edges = rvd.extractUniqueUndirectedEdges()
        self.rvd_graph_edges = surface_graph_edges

        direct_keep_nodes = []
        judge_nodes = []

        for node in self.m_leaf_nodes:
            has_feature = len(node.inside_convex_feature_ids) > 0

            if has_feature:
                node.decision_pid = -1
                node.decision_sid = -1
                node.groups = []
                direct_keep_nodes.append(node)
            else:
                judge_nodes.append(node)

        if len(judge_nodes) == 0:
            self.m_leaf_nodes = direct_keep_nodes
            self.node_id_map_decision_pid = {}
            self.node_id_map_decision_sid = {}
            self.node_id_map_spoint_clusters = {}

            print(
                f"[run_leaf_node_mat_search] direct_keep={len(direct_keep_nodes)}, "
                f"judged=0, final_kept=0, final_dropped=0"
            )
            return

        cellSeedNums, nearestCloudIdsFlat, cellNodeIds = _flatten_records_from_nodes(
            judge_nodes
        )

        final_kept_judge_nodes = []
        final_dropped_judge_nodes = []

        if len(cellNodeIds) > 0:
            projection_clusters = rvd.computeProjectionClustersNoLabelOneRingExpandedInSamePatch(
                cellSeedNums,
                nearestCloudIdsFlat,
                patch_ids,
            )

            node_id_to_groups = {
                int(node_id): clusters
                for node_id, clusters in zip(cellNodeIds.tolist(), projection_clusters)
            }

            self.node_id_map_cell_sample_clusters = node_id_to_groups
            self.node_id_map_spoint_clusters = {}

            for node in judge_nodes:
                node_id = int(node.id)
                clusters = node_id_to_groups.get(node_id, [])
                node.groups = clusters

                _, sdfs, node_sns, _, _, shape_point_ids, _ = self.get_group_data(node)

                sdfs = np.asarray(sdfs)
                node_sns = np.asarray(node_sns)
                shape_point_ids = np.asarray(shape_point_ids, dtype=np.int64)

                node_patch_ids = patch_ids[shape_point_ids]

                valid_patch_ids = node_patch_ids[node_patch_ids >= 0]
                feature_patch_ids = node_patch_ids[node_patch_ids == -1]

                if len(valid_patch_ids) == 0:
                    radian_threshold = np.radians(self.angle_threshold0)
                    cosine_eps = 1.0 - np.cos(radian_threshold)

                    normal_clustering = DBSCAN(
                        eps=cosine_eps,
                        min_samples=1,
                        metric="cosine",
                    )
                    normal_labels0 = normal_clustering.fit_predict(node_sns)

                    if len(np.unique(normal_labels0)) > 5 or len(node.groups) > 4:
                        ret = is_cylindrical_tinf_node(node_sns, True)
                        node.is_cylinder = bool(ret["is_cylindrical_tinf"])

                else:
                    unique_patch_ids, patch_counts = np.unique(
                        valid_patch_ids,
                        return_counts=True,
                    )
                    unique_patch_count = len(unique_patch_ids)

                    if unique_patch_count < 3 and unique_patch_count > 0:
                        dominant_idx = np.argmax(patch_counts)
                        dominant_patch_id = unique_patch_ids[dominant_idx]

                        dominant_mask = (node_patch_ids == dominant_patch_id) | (
                            node_patch_ids == -1
                        )
                        dominant_sns = node_sns[dominant_mask]

                        if (
                            unique_patch_count == 1
                            and len(feature_patch_ids) > len(valid_patch_ids)
                        ):
                            dominant_mask = node_patch_ids == -1
                            dominant_sns = node_sns[dominant_mask]

                        if len(dominant_sns) >= 4:
                            radian_threshold = np.radians(self.angle_threshold0)
                            cosine_eps = 1.0 - np.cos(radian_threshold)

                            normal_clustering = DBSCAN(
                                eps=cosine_eps,
                                min_samples=1,
                                metric="cosine",
                            )
                            normal_labels0 = normal_clustering.fit_predict(dominant_sns)

                            if len(np.unique(normal_labels0)) > 5 or len(node.groups) > 4:
                                ret = is_cylindrical_tinf_node(
                                    dominant_sns,
                                    len(feature_patch_ids) > 0,
                                )
                                node.is_cylinder = bool(ret["is_cylindrical_tinf"])

                spoint_ids_cluster = _clusters_to_spoint_ids(clusters, shape_point_ids)
                self.node_id_map_spoint_clusters[node_id] = spoint_ids_cluster

                if len(clusters) > 1 or node.is_cylinder:
                    final_kept_judge_nodes.append(node)
                else:
                    if len(feature_patch_ids) > 0 and len(valid_patch_ids) > 0:
                        sp_convex_flags = sp_convex_feature_mask[shape_point_ids]

                        sdfs = sdfs.copy()
                        sdfs[sp_convex_flags] = np.abs(sdfs[sp_convex_flags])
                        mask = sdfs < 0
                        if np.sum(mask) > 1:
                            ids = np.where(mask)[0]

                            radian_threshold = np.radians(
                                min(
                                    (180.0 - min_dihedral_angle_degree) / 2.0 + 0.1,
                                    90.0,
                                )
                            )
                            cosine_eps = 1.0 - np.cos(radian_threshold)

                            normal_clustering = DBSCAN(
                                eps=cosine_eps,
                                min_samples=1,
                                metric="cosine",
                            )
                            normal_labels0 = normal_clustering.fit_predict(node_sns[ids])

                            if len(np.unique(normal_labels0)) > 1:
                                uniq = np.unique(normal_labels0)
                                clusters = []

                                for normal_label in uniq:
                                    normal_mask = normal_labels0 == normal_label
                                    local_cluster_indices = np.where(normal_mask)[0]
                                    cluster_indices = ids[local_cluster_indices]
                                    clusters.append(cluster_indices.tolist())

                                node.groups = clusters
                                node_id_to_groups[node_id] = clusters
                                self.node_id_map_cell_sample_clusters[node_id] = clusters

                                spoint_ids_cluster = _clusters_to_spoint_ids(
                                    clusters,
                                    shape_point_ids,
                                )
                                self.node_id_map_spoint_clusters[node_id] = spoint_ids_cluster

                                final_kept_judge_nodes.append(node)
                                continue

                    node.drop_stage = 2
                    final_dropped_judge_nodes.append(node)

        else:
            self.node_id_map_spoint_clusters = {}

            for node in judge_nodes:
                node.groups = []
                node.drop_stage = 2
                final_dropped_judge_nodes.append(node)

        self.drop_nodes = final_dropped_judge_nodes
        self.m_leaf_nodes = direct_keep_nodes + final_kept_judge_nodes


        print(
            f"[run_leaf_node_mat_search] direct_keep={len(direct_keep_nodes)}, "
            f"judged={len(judge_nodes)}, "
            f"final_kept={len(final_kept_judge_nodes)}, "
            f"final_dropped={len(final_dropped_judge_nodes)}"
        )

    def generate_sample_pattern(self,stop,step=1)-> np.ndarray:
        x, y, z = np.meshgrid(
            np.arange(0,stop,step),
            np.arange(0,stop,step),
            np.arange(0,stop,step),
            indexing='ij'
        )
        sample_grids = np.vstack((x.flatten(), y.flatten(), z.flatten())).T
        return sample_grids
    def generate_sample_pattern_of_depth(self,depth)-> np.ndarray:
        assert(depth<=self.m_max_depth)
        return self.generate_sample_pattern(2 ** (self.m_sampling_depth+self.m_max_depth-depth) + 1,2**(self.m_max_depth-depth))
    
    def compute_box_length_of_depth(self,depth)->float:
        assert(depth<=self.m_max_depth)
        box_edge_length = self.m_unit_edge_length*(2 ** self.m_sampling_depth)*(2**(self.m_max_depth-depth))
        return box_edge_length
    
    def convert_grid_to_coords(self, grid: np.ndarray) -> np.ndarray:
        return self.m_min_corner + grid * self.m_unit_edge_length
    
    def get_centroids_of_leaf_nodes(self) ->np.ndarray:
        print("m_leaf_nodes.size():",len(self.m_leaf_nodes))
        centroids=np.zeros((len(self.m_leaf_nodes),3),dtype=self.dtype)
        for i,node in enumerate(self.m_leaf_nodes):
            centroid_grid=node.m_centroid_id
            centroids[i]=self.convert_grid_to_coords(centroid_grid)
        return centroids 
    
    
    def encode(self,grid):
        return (grid[0]*self.m_capacity_per_dim+grid[1])*self.m_capacity_per_dim+grid[2]
    
    def decode(self,gid:int):
        grid=np.zeros(3,dtype=int)
        grid[0] = gid // (self.m_capacity_per_dim ** 2)
        grid[1] = (gid // self.m_capacity_per_dim) % self.m_capacity_per_dim
        grid[2] = gid % self.m_capacity_per_dim
        return grid
    
    def get_samples_of_new_nodes_optimized(self) -> tuple:
        new_grid_indices_vec = []
        new_grid_coords_vec = []
        id_counter = 0
        for leaf_node in self.m_leaf_nodes:
            sampling_pattern= self.m_sampling_patterns_for_depth[leaf_node.m_depth]
            leaf_node.m_sample_grids=leaf_node.generate_sample_grids(sampling_pattern)
            sampling_grids = leaf_node.m_sample_grids
            leaf_node.cell_sample_mask = np.zeros(len(sampling_grids), dtype=bool)

            gids = (sampling_grids[:, 0] * self.m_capacity_per_dim + sampling_grids[:, 1]) * self.m_capacity_per_dim + sampling_grids[:, 2]

            new_points_mask = ~self.m_grid_existence[gids]
            new_gids = gids[new_points_mask]
            new_coords = self.convert_grid_to_coords(sampling_grids[new_points_mask])

            self.m_grid_existence[new_gids] = True

            new_sids = np.arange(len(self.m_sample_coords) + id_counter, len(self.m_sample_coords) + id_counter + len(new_gids))
            self.m_gid2sid[new_gids] = new_sids
            id_counter += len(new_gids)

            new_grid_indices_vec.extend(new_gids.tolist())
            new_grid_coords_vec.extend(new_coords)


        new_sample_size = len(self.m_sample_coords) + len(new_grid_coords_vec)

        origin_m_sample_coords=copy.deepcopy(self.m_sample_coords)
        origin_m_sample_mesh_grads=copy.deepcopy(self.m_sample_mesh_sns)
        origin_m_sample_mesh_sdfs=copy.deepcopy(self.m_sample_mesh_sdfs)
        origin_m_sample_mesh_projs=copy.deepcopy(self.m_sample_mesh_projs)
        origin_m_sample_mesh_point_ids=copy.deepcopy(self.m_sample_mesh_point_ids)
        
        origin_m_sample_mesh_masks=copy.deepcopy(self.m_sample_mesh_masks)


        self.m_sample_coords = np.zeros((new_sample_size, 3), dtype=self.dtype)
        self.m_sample_mesh_sns = np.zeros((new_sample_size, 3), dtype=self.dtype)
        self.m_sample_mesh_sdfs = np.zeros(new_sample_size, dtype=self.dtype)
        self.m_sample_mesh_projs=np.zeros((new_sample_size, 3), dtype=self.dtype)
        self.m_sample_mesh_point_ids=np.zeros(new_sample_size, dtype=int)
        self.m_sample_mesh_masks = np.zeros(new_sample_size, dtype=bool)

        if len(origin_m_sample_coords)>0:
            self.m_sample_coords[:len(origin_m_sample_coords)] = origin_m_sample_coords
            self.m_sample_mesh_sns[:len(origin_m_sample_mesh_grads)] = origin_m_sample_mesh_grads
            self.m_sample_mesh_sdfs[:len(origin_m_sample_mesh_sdfs)] = origin_m_sample_mesh_sdfs
            self.m_sample_mesh_projs[:len(origin_m_sample_mesh_projs)] = origin_m_sample_mesh_projs
            self.m_sample_mesh_point_ids[:len(origin_m_sample_mesh_point_ids)] = origin_m_sample_mesh_point_ids
            self.m_sample_mesh_masks[:len(origin_m_sample_mesh_masks)] =origin_m_sample_mesh_masks

        if len(new_grid_coords_vec)>0:
            self.m_sample_coords[len(origin_m_sample_coords):] = np.array(new_grid_coords_vec)

        return np.array(new_grid_indices_vec), np.array(new_grid_coords_vec)
    



    def set_new_grid_shape_data(self,new_grid_indices,new_grid_shape_sdfs,new_grid_shape_sns,new_grid_shape_projs,new_grid_shape_point_ids,new_grid_shape_masks):
        data_ids=self.m_gid2sid[new_grid_indices]
        self.m_sample_mesh_sdfs[data_ids]=new_grid_shape_sdfs
        self.m_sample_mesh_sns[data_ids]=new_grid_shape_sns   
        self.m_sample_mesh_projs[data_ids]=new_grid_shape_projs
        self.m_sample_mesh_point_ids[data_ids]=new_grid_shape_point_ids
        self.m_sample_mesh_masks[data_ids]=new_grid_shape_masks

    def build_spoint_id_maps(self):
        self.spoint_id_to_sid_map = {}

        for node in self.m_leaf_nodes:
            sampling_grids = node.m_sample_grids

            gids = (
                (sampling_grids[:, 0] * self.m_capacity_per_dim + sampling_grids[:, 1])
                * self.m_capacity_per_dim
                + sampling_grids[:, 2]
            )

            sids = self.m_gid2sid[gids]
            shape_point_ids = self.m_sample_mesh_point_ids[sids]

            for pid, sid in zip(shape_point_ids, sids):
                pid = int(pid)
                sid = int(sid)
                self.spoint_id_to_sid_map.setdefault(pid, set()).add(sid)  

    def adaptive_subdivide(self, centroid_udfs: np.ndarray, centroid_grads: np.ndarray, margin: float):
        new_nodes=[]
        for i in range(len(self.m_leaf_nodes)):
            node_edge_length = self.m_unit_edge_length * self.m_leaf_nodes[i].m_edge_span
            half_diagonal_length = np.sqrt(3) * node_edge_length * 0.5
            if (self.m_leaf_nodes[i].m_edge_span <= 1):
                raise ValueError("Insufficient pre-allocated depth!")
            centroid_grid = self.m_leaf_nodes[i].m_centroid_id

            sid = self.m_gid2sid[self.encode(centroid_grid)]

            self.m_centroid_mat_udfs[sid] = centroid_udfs[i]
            self.m_centroid_mat_grads[sid] = centroid_grads[i]

            if (self.m_leaf_nodes[i].m_depth < self.m_max_depth):
                if (self.m_centroid_mat_udfs[sid] < half_diagonal_length + margin):
                    self.m_leaf_nodes[i].subdivide()
                    self.m_leaf_nodes[i].m_node_type = "interval"

                    new_nodes.extend(self.m_leaf_nodes[i].m_children)
                else:
                    self.drop_nodes.append(self.m_leaf_nodes[i])
            else:
                new_nodes.append(self.m_leaf_nodes[i])
            
        self.m_leaf_nodes = new_nodes

    def get_group_data(self,node):
        sampling_grids = node.m_sample_grids
        
        gids = (sampling_grids[:, 0] * self.m_capacity_per_dim + sampling_grids[:, 1]) * self.m_capacity_per_dim + sampling_grids[:, 2]
        sids = self.m_gid2sid[gids]
        origin_sdfs = self.m_sample_mesh_sdfs[sids]
        
        origin_sdfs = origin_sdfs.reshape(-1, 1)
        origin_coords = self.m_sample_coords[sids]
        origin_sns=self.m_sample_mesh_sns[sids]
        origin_proj_points = self.m_sample_mesh_projs[sids]
        origin_shape_point_ids=self.m_sample_mesh_point_ids[sids]     
        origin_feature_flags = self.m_sample_mesh_masks[sids]
        mask = origin_sdfs[:, 0] !=0
        
        node.cell_sample_mask = mask
        if np.all(origin_sdfs > 0) and len(node.inside_surface_point_ids)==0:
            return [],[],[],[],[],[],[]
        
        sdfs = origin_sdfs[mask]
        sns=origin_sns[mask]
        coords = origin_coords[mask]
        proj_points = origin_proj_points[mask]
        shape_point_ids = origin_shape_point_ids[mask]
        feature_flags = origin_feature_flags[mask]
        return sids[mask],sdfs,sns,coords,proj_points,shape_point_ids,feature_flags
    
    def remove_isolated_cylinder_nodes(self):
        has_feature=len(self.convex_feature_points)>0
        if not has_feature:
            return
        for leaf_node_index, node in enumerate(self.m_leaf_nodes):
            if not hasattr(node, 'is_cylinder') or not node.is_cylinder:
                continue

            adjacent_cylinder_count = 0
            
            node_pos_type=self.m_leaf_node_pos_types[leaf_node_index]
            x_bits = node_pos_type & 0b000011
            y_bits = node_pos_type & 0b001100
            z_bits = node_pos_type & 0b110000
            adj_node_id0=-1
            adj_node_id1=-1
            if x_bits==0b000011:
                adj_node_id0=self.m_leaf_nodes_adjacency[leaf_node_index,0]
                adj_node_id1=self.m_leaf_nodes_adjacency[leaf_node_index,1]
                if adj_node_id0!=-1 and self.m_leaf_nodes[adj_node_id0].is_cylinder:
                    adjacent_cylinder_count += 1
                if adj_node_id1!=-1 and self.m_leaf_nodes[adj_node_id1].is_cylinder:
                    adjacent_cylinder_count += 1
            if y_bits==0b001100:
                adj_node_id0=self.m_leaf_nodes_adjacency[leaf_node_index,2]
                adj_node_id1=self.m_leaf_nodes_adjacency[leaf_node_index,3]
                if adj_node_id0!=-1 and self.m_leaf_nodes[adj_node_id0].is_cylinder:
                    adjacent_cylinder_count += 1
                if adj_node_id1!=-1 and self.m_leaf_nodes[adj_node_id1].is_cylinder:
                    adjacent_cylinder_count += 1
            if z_bits==0b110000:
                adj_node_id0=self.m_leaf_nodes_adjacency[leaf_node_index,4]
                adj_node_id1=self.m_leaf_nodes_adjacency[leaf_node_index,5]
                if adj_node_id0!=-1 and self.m_leaf_nodes[adj_node_id0].is_cylinder:
                    adjacent_cylinder_count += 1
                if adj_node_id1!=-1 and self.m_leaf_nodes[adj_node_id1].is_cylinder:
                    adjacent_cylinder_count += 1
        
            if adjacent_cylinder_count ==0:
                node.is_cylinder = False

    def compute_inside_surface_point_ids(self,spoints,node):
        if len(node.m_parent.inside_surface_point_ids)>0:
            points=spoints[node.m_parent.inside_surface_point_ids]
            min_corner = self.convert_grid_to_coords(node.m_min_corner_id)
            max_corner = self.convert_grid_to_coords(node.m_max_corner_id)
            local_inside_ids=indices_in_bbox(points,min_corner,max_corner)
            node.inside_surface_point_ids=node.m_parent.inside_surface_point_ids[local_inside_ids]
    def pre_check_node_build(self, node,spoints):
        node.cell_sample_mask = np.ones(len(node.m_sample_grids), dtype=bool)
        
        if node.m_parent is None:
            return True
        box_edge_length = self.box_edge_length_for_depth[node.m_depth]
        diag_length = np.sqrt(3) * box_edge_length

        centroid_grid = node.m_centroid_id
        centroid_sid = self.m_gid2sid[self.encode(centroid_grid)]
        centroid_coord=self.m_sample_coords[centroid_sid]
        centroid_sdf=self.m_sample_mesh_sdfs[centroid_sid]
        
        if centroid_sdf>diag_length*0.5 and len(node.inside_convex_feature_ids)==0:
            return False
        # dists, _ =self.sdf_kdtree.query(np.array([centroid_coord]), k=1)
        # if centroid_sdf>0 and dists[0]>diag_length*0.5 and len(node.inside_convex_feature_ids)==0:
        #     return False
        sampling_grids = node.m_sample_grids
        
        gids = (sampling_grids[:, 0] * self.m_capacity_per_dim + sampling_grids[:, 1]) * self.m_capacity_per_dim + sampling_grids[:, 2]
        sids = self.m_gid2sid[gids]
        origin_sdfs = self.m_sample_mesh_sdfs[sids]
        if node.m_depth<self.m_max_depth:
            if np.max(origin_sdfs)>0:
                self.compute_inside_surface_point_ids(spoints,node)
        else:
            if np.min(origin_sdfs)>0:
                self.compute_inside_surface_point_ids(spoints,node)
        if len(node.inside_convex_feature_ids) > 0:
            return True

        _,sdfs,sns,coords,nearest_points,shape_point_ids,feature_flags=self.get_group_data(node)
        
        if np.isnan(sns).any():
            print(sns)
        if len(coords) > 1:
            norms=np.linalg.norm(sns, axis=1, keepdims=True)
            sns = sns / norms
            
            torch_feature = np.count_nonzero(feature_flags) > 0    
            if not torch_feature and node.m_depth<self.m_max_depth and len(self.convex_feature_points)>0:
                radian_threshold=np.radians(self.angle_threshold0)
                cosine_eps = 1 - np.cos(radian_threshold)
                normal_clustering = DBSCAN(eps=cosine_eps,min_samples=1,metric='cosine')
                normal_labels0 = normal_clustering.fit_predict(sns)
                if len(np.unique(normal_labels0))>1:
                    return True
                else:
                    return False
            elif node.m_depth==self.m_max_depth and len(self.convex_feature_points)==0:
                radian_threshold=np.radians(self.angle_threshold)
                cosine_eps = 1 - np.cos(radian_threshold)
                normal_clustering = DBSCAN(eps=cosine_eps,min_samples=1,metric='cosine')
                normal_labels0 = normal_clustering.fit_predict(sns)
                if len(np.unique(normal_labels0))>1:
                    return True
                else:
                    return False
            return True
        return False
    
    def compute_node_strict_contain_feature_ids(self):
        if len(self.convex_feature_points) > 0:
            for node in self.m_leaf_nodes:
                if len(node.inside_convex_feature_ids)>0:
                    min_corner =self.convert_grid_to_coords(node.m_min_corner_id)
                    max_corner =self.convert_grid_to_coords(node.m_max_corner_id)
                    cadidate_points=self.convex_feature_points[node.inside_convex_feature_ids]
                    local_inside_ids=indices_in_bbox(cadidate_points,min_corner,max_corner)
                    if len(local_inside_ids) > 0:
                        node.inside_convex_feature_ids = node.inside_convex_feature_ids[local_inside_ids]
    def adaptive_subdivide_sqef_parallel(self,spoints):
        total_nodes = len(self.m_leaf_nodes)
        
        if total_nodes <= 1024:
            return self.adaptive_subdivide_sqef(spoints)
        
        num_processes = min(mp.cpu_count(), 2)
        chunk_size = total_nodes // num_processes + 1
        
        pool = mp.Pool(processes=num_processes)
        
        try:
            chunks = []
            for i in range(0, total_nodes, chunk_size):
                chunk_end = min(i + chunk_size, total_nodes)
                chunks.append((i, chunk_end))
            process_func = partial(process_node_chunk_ids, octree=self,spoints=spoints)
            chunk_results = pool.map(process_func, chunks)
            subdivide_indices = []
            drop_indices = []
            mask_updates = {}
            group_updates = {}
            inside_surface_point_ids_updates={}
            for chunk_subdivide, chunk_drop, chunk_masks,chunk_groups,chunk_inside_surface_point_ids in chunk_results:
                subdivide_indices.extend(chunk_subdivide)
                drop_indices.extend(chunk_drop)
                mask_updates.update(chunk_masks)
                group_updates.update(chunk_groups)
                inside_surface_point_ids_updates.update(chunk_inside_surface_point_ids)

            new_nodes = []
            self.drop_nodes = []
        
            for idx, mask in mask_updates.items():
                if idx < len(self.m_leaf_nodes):
                    self.m_leaf_nodes[idx].cell_sample_mask = mask
            for idx, groups in group_updates.items():
                if idx < len(self.m_leaf_nodes):
                    self.m_leaf_nodes[idx].groups = groups
            for idx, inside_surface_point_ids in inside_surface_point_ids_updates.items():
                if idx < len(self.m_leaf_nodes):
                    self.m_leaf_nodes[idx].inside_surface_point_ids = inside_surface_point_ids
            for idx in subdivide_indices:
                node = self.m_leaf_nodes[idx]
                node.subdivide()
                node.m_node_type = "interval"
                new_nodes.extend(node.m_children)
            
            for i, node in enumerate(self.m_leaf_nodes):
                if i not in subdivide_indices and i not in drop_indices:
                    new_nodes.append(node)
            
            for idx in drop_indices:
                self.m_leaf_nodes[idx].drop_stage=0
                self.drop_nodes.append(self.m_leaf_nodes[idx])
                
            self.m_leaf_nodes = new_nodes
            
        finally:
            pool.close()
            pool.join()

    def scan_occupied(self):
        self.occupied = np.zeros((self.resolution, self.resolution, self.resolution), dtype=bool)
        self.occupied_cell = np.full((self.resolution, self.resolution, self.resolution), -1, dtype=np.int64)
        self.occupied_cell_inverse = {}
        for i, node in enumerate(self.m_leaf_nodes):
            node.leaf_node_index = i
            x, y, z = node.m_min_corner_id // (2 ** self.m_sampling_depth)
            self.occupied[x, y, z] = 1
            self.occupied_cell[x, y, z] = node.id
            self.occupied_cell_inverse[node.id] = (int(x), int(y), int(z))
    def _calculate_chunk_size(self, total_nodes):
        if total_nodes <= 2000:
            return 1000
        elif total_nodes <= 10000:
            return 2000
        elif total_nodes <= 40000:
            return 4000
        else:
            return 10000

    def grow_by_max_depth(self,spoints,convex_feature_points,cloud_sdf_grad_mask_func,margin_threshold):
        if len(convex_feature_points)>0:
            inside_convex_feature_ids=np.arange(len(convex_feature_points))
        else:
            inside_convex_feature_ids=[]
        
        self.m_root_node.inside_convex_feature_ids=inside_convex_feature_ids
        self.m_root_node.id=0
        cur_depth = 0

        while cur_depth <= self.m_max_depth:
            new_grid_indices, new_grid_coords = self.get_samples_of_new_nodes_optimized()
            
            if len(new_grid_indices)>0:
                new_grid_shape_sdfs, new_grid_shape_sns, new_grid_shape_projs, \
                new_grid_shape_point_ids, new_grid_shape_masks = \
                    query_sdf_and_sn_and_proj_and_feature_mask(
                        cloud_sdf_grad_mask_func,
                        new_grid_coords.astype(np.float32),
                        -1
                    )
                self.set_new_grid_shape_data(
                    new_grid_indices,
                    new_grid_shape_sdfs.flatten(),
                    new_grid_shape_sns,
                    new_grid_shape_projs,
                    new_grid_shape_point_ids.flatten(),
                    new_grid_shape_masks.flatten()
                )
            
            # if cur_depth <= max_depth - 4:
            #     octree.adaptive_subdivide_sqef(spoints)
            # else:
            #     octree.adaptive_subdivide_sqef_parallel(spoints)
            # octree.adaptive_subdivide_sqef(spoints,margin_threshold)
            self.adaptive_subdivide_sqef(spoints,margin_threshold)

            cur_depth += 1
        self.build_spoint_id_maps()
        self.scan_occupied()
        self.compute_node_strict_contain_feature_ids()

    def adaptive_subdivide_sqef(self,spoints,margin):
        new_nodes=[]
        self.drop_nodes=[]
        for i in range(len(self.m_leaf_nodes)):
            node=self.m_leaf_nodes[i]
            if node.id==391349:
                pass
            if (node.m_edge_span <= 1):
                raise ValueError("Insufficient pre-allocated depth!")
            if node.m_parent is not None:
                parent_inside_convex_feature_ids=node.m_parent.inside_convex_feature_ids
                if len(parent_inside_convex_feature_ids)>0:
                    min_corner = self.convert_grid_to_coords(node.m_min_corner_id)
                    max_corner = self.convert_grid_to_coords(node.m_max_corner_id)
                    points=self.convex_feature_points[parent_inside_convex_feature_ids]
                    offset_dir = max_corner - min_corner
                    ext_min_corner = min_corner - offset_dir * margin
                    ext_max_corner = max_corner + offset_dir * margin
                    in_bbox = (points >= ext_min_corner) & (points <= ext_max_corner)
                    mask = np.all(in_bbox, axis=1)
                    local_inside_convex_feature_ids=np.where(mask)[0]
                    global_inside_convex_feature_ids=parent_inside_convex_feature_ids[local_inside_convex_feature_ids]
                    if len(global_inside_convex_feature_ids)>0:
                        node.inside_convex_feature_ids=global_inside_convex_feature_ids
            flag=self.pre_check_node_build(node,spoints)
            
            if flag:
                if (self.m_leaf_nodes[i].m_depth < self.m_max_depth):
                    self.m_leaf_nodes[i].subdivide()
                    self.m_leaf_nodes[i].m_node_type = "interval"
                    new_nodes.extend(self.m_leaf_nodes[i].m_children)
                else:
                    new_nodes.append(self.m_leaf_nodes[i])
            else:
                self.drop_nodes.append(self.m_leaf_nodes[i])
        self.m_leaf_nodes = new_nodes
    
    
    def dfs_relate(self):
        for leaf_node_index,result in enumerate(self.results):
            node=self.m_leaf_nodes[leaf_node_index]
            if result.m_is_inside and result.m_point is not None:
                node.m_node_cell_type= 2
            elif result.m_point is not None and self.leaf_link_nodes[leaf_node_index].next is not None:
                visited=np.zeros(len(self.results))
                cur_list_node=self.leaf_link_nodes[leaf_node_index]
                while cur_list_node.next is not None and visited[cur_list_node.val]==0:
                    visited[cur_list_node.val]=1
                    cur_list_node=cur_list_node.next
                cur_index=cur_list_node.val
                if self.results[cur_index].m_is_inside:
                    node.m_node_type = "leaf"
                    target=self.results[cur_index].m_point
                    if self.results[cur_index].m_type>result.m_type and result.m_type<PLANE:
                        target_node=self.m_leaf_nodes[cur_index]
                        self.results[cur_index].m_point=result.m_point
                        self.m_mesh_v[target_node.m_vertex_idx]=result.m_point
                        self.m_mesh_v_radii[target_node.m_vertex_idx]=result.radius
                    elif self.results[cur_index].m_type<=result.m_type:
                        result.m_point=target
                        result.radius=self.results[cur_index].radius
                    else:
                        pass
                        
                    node.m_node_cell_type= 1#virtual
                    #========
                    node.m_vertex_idx = len(self.m_mesh_v)
                    self.m_mesh_v.append(result.m_point)
                    self.m_mesh_v_radii.append(result.radius)
                    self.m_v_type.append(result.m_type)
                    self.m_mesh_v_to_leaf_node_indexes.append(leaf_node_index)

                    self.vertex_to_leaf_cell[node.m_vertex_idx]=leaf_node_index
                    self.vertex_to_leaf_cell_inverse[leaf_node_index]=node.m_vertex_idx#leaf node index<-->vertex id
                    #========
                    node.quote_leaf_node_index=cur_index
                    node.quote_vertex_id=self.m_leaf_nodes[cur_index].m_vertex_idx

                    self.m_leaf_nodes[cur_index].be_quoted_leaf_node_indexes.append(node.leaf_node_index)
                    self.m_leaf_nodes[cur_index].be_quoted_vertex_ids.append(node.m_vertex_idx)
                    
                else:
                    node.m_node_type = "leaf"
                    #================================add data for node of cur_index================================
                    node.m_node_cell_type= 1#virtual

                    self.results[cur_index].m_point=result.m_point
                    self.m_leaf_nodes[cur_index].m_vertex_idx=len(self.m_mesh_v)

                    self.vertex_to_leaf_cell[self.m_leaf_nodes[cur_index].m_vertex_idx]=cur_index
                    self.vertex_to_leaf_cell_inverse[cur_index]=self.m_leaf_nodes[cur_index].m_vertex_idx
                    self.m_mesh_v.append(result.m_point)
                    self.m_mesh_v_radii.append(result.radius)
                    self.m_v_type.append(result.m_type)
                    self.m_mesh_v_to_leaf_node_indexes.append(cur_index)
                    self.m_leaf_nodes[cur_index].m_reconstruct_point =result.m_point
                    self.m_leaf_nodes[cur_index].m_node_cell_type= 2
                    #fix bug
                    self.m_leaf_nodes[cur_index].m_node_type='leaf'
                    self.results[cur_index].m_is_inside=True
                    #================================add data for node================================
                    node.m_vertex_idx = len(self.m_mesh_v)
                    self.m_mesh_v.append(result.m_point)
                    self.m_mesh_v_radii.append(result.radius)
                    self.m_v_type.append(result.m_type)
                    self.m_mesh_v_to_leaf_node_indexes.append(leaf_node_index)

                    self.vertex_to_leaf_cell[node.m_vertex_idx]=leaf_node_index
                    self.vertex_to_leaf_cell_inverse[leaf_node_index]=node.m_vertex_idx
                    #================================add quote reference================================
                    node.quote_leaf_node_index=cur_index
                    node.quote_vertex_id=self.m_leaf_nodes[cur_index].m_vertex_idx

                    self.m_leaf_nodes[cur_index].be_quoted_leaf_node_indexes.append(node.leaf_node_index)
                    self.m_leaf_nodes[cur_index].be_quoted_vertex_ids.append(node.m_vertex_idx)
                #add cylinder node
                if node.is_cylinder:
                    self.cylinder_node_ids.append(i)
            else:
                node.m_node_cell_type=0
                node.m_node_type = "empty"
                node.m_vertex_idx = -1
        self.m_mesh_v_to_leaf_node_indexes=np.array(self.m_mesh_v_to_leaf_node_indexes)
        self.quote_leaf_node_indexes = [-1 for _ in range(len(self.m_leaf_nodes))]
        self.be_quoted_leaf_node_indexes = [[] for _ in range(len(self.m_leaf_nodes))]
        for i,result in enumerate(self.m_leaf_nodes):
            node=self.m_leaf_nodes[i]
            if node.quote_leaf_node_index!=-1:
                self.quote_leaf_node_indexes[i]=(node.quote_leaf_node_index)
            if len(node.be_quoted_leaf_node_indexes)>0:
                self.be_quoted_leaf_node_indexes[i].extend(node.be_quoted_leaf_node_indexes)
        self.quote_vertex_ids = [-1 for _ in range(len(self.m_mesh_v))]
        self.be_quoted_vertex_ids = [[] for _ in range(len(self.m_mesh_v))]
        for i in range(len(self.m_mesh_v)):
            leaf_node_index=self.vertex_to_leaf_cell[i]
            node=self.m_leaf_nodes[leaf_node_index]
            self.m_mesh_v_simulated.append(self.convert_grid_to_coords(node.m_centroid_id))
            if node.quote_vertex_id!=-1:
                self.quote_vertex_ids[i]=node.quote_vertex_id
            if len(node.be_quoted_vertex_ids)>0:
                self.be_quoted_vertex_ids[i].extend(node.be_quoted_vertex_ids)
        self.quote_vertex_ids =np.array(self.quote_vertex_ids)
        
        for leaf_node_index in self.cylinder_node_ids:
            node=self.m_leaf_nodes[leaf_node_index]
            cur_vertex_idx=node.m_vertex_idx
            if cur_vertex_idx==-1:
                continue
        self.m_mesh_v_adj_cylinder=np.zeros(len(self.m_mesh_v),dtype=bool)
        for leaf_node_index in self.cylinder_node_ids:
            node=self.m_leaf_nodes[leaf_node_index]
            cur_vertex_idx=node.m_vertex_idx
            if cur_vertex_idx==-1:
                continue
            adj_leaf_node_indexes=self.m_leaf_nodes_adjacency[leaf_node_index]
            for adj_leaf_node_index in adj_leaf_node_indexes:
                if adj_leaf_node_index==-1:
                    continue
                adj_node=self.m_leaf_nodes[adj_leaf_node_index]
                if adj_node.m_vertex_idx==-1:
                    continue
                if self.m_mesh_v_adj_cylinder[adj_node.m_vertex_idx]:
                    continue
                self.m_mesh_v_adj_cylinder[adj_node.m_vertex_idx]=True
    def batch_locate(self):
        self.m_mesh_v=[]
        self.m_mesh_v_radii=[]
        self.m_mesh_v_simulated=[]
        self.m_v_type=[]
        self.vertex_to_leaf_cell={}
        self.vertex_to_leaf_cell_inverse={}
        self.leaf_link_nodes=[ListNode() for _ in range(len(self.m_leaf_nodes))]

        for leaf_node_index,node in enumerate(self.m_leaf_nodes):
            # if leaf_node_index==1770:
            #     print(leaf_node_index)
            result=self.results[leaf_node_index]
            if result.m_is_inside and result.m_point is not None:
                #add cylinder node
                if node.is_cylinder:
                    self.cylinder_node_ids.append(leaf_node_index)
                else:
                    node_pos_type=self.m_leaf_node_pos_types[leaf_node_index]
                    x_bits = node_pos_type & 0b000011
                    y_bits = node_pos_type & 0b001100
                    z_bits = node_pos_type & 0b110000
                    adj_node_id0=-1
                    adj_node_id1=-1
                    if x_bits==0b000011:
                        adj_node_id0=self.m_leaf_nodes_adjacency[leaf_node_index,0]
                        adj_node_id1=self.m_leaf_nodes_adjacency[leaf_node_index,1]
                    elif y_bits==0b001100:
                        adj_node_id0=self.m_leaf_nodes_adjacency[leaf_node_index,2]
                        adj_node_id1=self.m_leaf_nodes_adjacency[leaf_node_index,3]
                    elif z_bits==0b110000:
                        adj_node_id0=self.m_leaf_nodes_adjacency[leaf_node_index,4]
                        adj_node_id1=self.m_leaf_nodes_adjacency[leaf_node_index,5]
                    if adj_node_id0!=-1 and adj_node_id1!=-1:
                        if self.m_leaf_nodes[adj_node_id0].is_cylinder and self.m_leaf_nodes[adj_node_id1].is_cylinder:
                            node.is_cylinder=True
                            self.cylinder_node_ids.append(leaf_node_index)
                self.leaf_link_nodes[leaf_node_index].val=leaf_node_index
                self.leaf_link_nodes[leaf_node_index].next=None

                node.m_node_type = "leaf"
                node.m_vertex_idx = len(self.m_mesh_v)

                self.m_mesh_v.append(self.results[leaf_node_index].m_point)
                self.m_mesh_v_radii.append(self.results[leaf_node_index].radius)
                self.m_v_type.append(self.results[leaf_node_index].m_type)
                self.m_mesh_v_to_leaf_node_indexes.append(leaf_node_index)

                self.vertex_to_leaf_cell[node.m_vertex_idx]=leaf_node_index
                self.vertex_to_leaf_cell_inverse[leaf_node_index]=node.m_vertex_idx

            elif result.m_point is not None:
                sqef=self.sqefs[leaf_node_index]
                _, index = self.center_kdtree.query(result.m_point)

                #adj_node_ids=self.m_leaf_nodes_adjacency[i]
                check=True
                cur_x,cur_y,cur_z=node.m_min_corner_id//(2**self.m_sampling_depth)
                index_x,index_y,index_z=self.m_leaf_nodes[index].m_min_corner_id//(2**self.m_sampling_depth)
                if sqef is not None and (not sqef.is_line and not sqef.is_plane):
                    # if abs(cur_x-index_x)>2 or abs(cur_y-index_y)>2 or abs(cur_z-index_z)>2:
                    if abs(cur_x-index_x)>1 or abs(cur_y-index_y)>1 or abs(cur_z-index_z)>1:
                    #if abs(cur_x-index_x)+abs(cur_y-index_y)+abs(cur_z-index_z)!=1:
                        check=False
                else:
                    if abs(cur_x-index_x)>1 or abs(cur_y-index_y)>1 or abs(cur_z-index_z)>1:
                    #if abs(cur_x-index_x)+abs(cur_y-index_y)+abs(cur_z-index_z)!=1:
                        check=False
                # if sqef is not None:
                #     if (abs(cur_x-index_x)+abs(cur_y-index_y)+ abs(cur_z-index_z))>2:
                #         check=False
 

                if not check:
                    self.leaf_link_nodes[leaf_node_index].val=leaf_node_index
                    self.leaf_link_nodes[leaf_node_index].next=None
                    continue
                min_corner = self.convert_grid_to_coords(self.m_leaf_nodes[index].m_min_corner_id)
                max_corner = self.convert_grid_to_coords(self.m_leaf_nodes[index].m_max_corner_id)
                if is_inside(result.m_point,min_corner,max_corner):
                    self.leaf_link_nodes[leaf_node_index].val=leaf_node_index
                    self.leaf_link_nodes[leaf_node_index].next=self.leaf_link_nodes[index]
                else:
                    self.leaf_link_nodes[leaf_node_index].val=leaf_node_index
                    self.leaf_link_nodes[leaf_node_index].next=None
            else:
                self.leaf_link_nodes[leaf_node_index].val=leaf_node_index
                self.leaf_link_nodes[leaf_node_index].next=None
 
    def contain_feature_ids(self,feature_points,node:OctreeNode,offset=0.0):
        if len(feature_points) == 0:
            return []
        feature_points = np.asarray(feature_points)

        min_corner = self.convert_grid_to_coords(node.m_min_corner_id)
        max_corner = self.convert_grid_to_coords(node.m_max_corner_id)

        offset_dir = max_corner - min_corner
        ext_min_corner = min_corner - offset_dir * offset
        ext_max_corner = max_corner + offset_dir * offset
        inside_mask = np.all((feature_points >= ext_min_corner) & 
                            (feature_points <= ext_max_corner), axis=1)
        
        return np.where(inside_mask)[0].tolist()


    def search_cavity(self):
        self.cellProcContourSearch(self.m_root_node)
    
    def compute_face_rank_global(self,face_id):
        #geo_rank=0
        geo_rank=-min(self.m_mesh_f_udfs[face_id])
        if face_id in self.face2cavity:
            if len(self.face2cavity[face_id])>=2:
                return (np.inf,geo_rank)
        face=self.m_mesh_f[face_id]

        edges=extract_edges(face)
        adj_static_face_edges=[]

        for edge in edges:
            for fid in self.edge_to_faces[edge]:
                if fid!=face_id and len(self.edge_to_faces[edge])==2 and self.static_face_flag[fid]==True:
                    adj_static_face_edges.append(edge)
                    break
                   
        if len(adj_static_face_edges)==4:
            self.static_face_flag[face_id]=True
            return (np.inf,geo_rank)

        for edge in adj_static_face_edges:
            if len(self.edge_to_faces[edge])==2:
                return (np.inf,geo_rank)
        is_pocket=False
        topo_rank=0
        for vid in face:
            leaf_node_index=self.vertex_to_leaf_cell[vid]
            if self.m_leaf_nodes[leaf_node_index].is_cylinder:
                topo_rank+=1

        #bug
        # if np.any(self.m_mesh_v_adj_cylinder[face]):
        #     topo_rank=np.inf
        for edge in edges:
            if len(self.edge_to_faces[edge])==1:
                i0,i1=edge
                leaf_node_index0=self.vertex_to_leaf_cell[i0]
                leaf_node_index1=self.vertex_to_leaf_cell[i1]
                if self.feature_flags[leaf_node_index0] and self.feature_flags[leaf_node_index1]:
                #if self.m_v_type[i0]!=2 and self.m_v_type[i1]!=2:
                    return (np.inf,geo_rank)
                # elif len(self.convex_feature_points)==0 and self.m_leaf_nodes[leaf_node_index0].is_cylinder and self.m_leaf_nodes[leaf_node_index1].is_cylinder:
                elif self.m_leaf_nodes[leaf_node_index0].is_cylinder or self.m_leaf_nodes[leaf_node_index1].is_cylinder:
                    return (np.inf,geo_rank)
                else:
                    is_pocket=True

            # elif len(self.edge_to_faces[edge])==4:
            #     quote_vertex_id0=edge[0]
            #     if self.quote_vertex_ids[edge[0]]!=-1:
            #         quote_vertex_id0=self.quote_vertex_ids[edge[0]]
            #     quote_vertex_id1=edge[1]
            #     if self.quote_vertex_ids[edge[1]]!=-1:
            #         quote_vertex_id1=self.quote_vertex_ids[edge[1]]
            #     if quote_vertex_id0!=quote_vertex_id1:
            #         topo_rank=-np.inf
                


        if not is_pocket:
            if face_id not in self.face2cavity or len(self.face2cavity[face_id])==0:
                return (np.inf,geo_rank)
        return (topo_rank,geo_rank)  
    def compute_face_rank_local(self,face_id):

        geo_rank=-min(self.m_mesh_f_udfs[face_id])
        #geo_rank=0
        if face_id in self.face2cavity:
            if len(self.face2cavity[face_id])>=2:
                return (np.inf,geo_rank)
        face=self.m_mesh_f[face_id]
        edges=extract_edges(face)
        adj_static_face_edges=[]

        for edge in edges:
            for fid in self.edge_to_faces[edge]:
                if fid!=face_id and len(self.edge_to_faces[edge])==2 and self.static_face_flag[fid]==True:
                    adj_static_face_edges.append(edge)
                    break
                   
        if len(adj_static_face_edges)==4:
            self.static_face_flag[face_id]=True
            return (np.inf,geo_rank)

        for edge in adj_static_face_edges:     
            if len(self.edge_to_faces[edge])==2:
                return (np.inf,geo_rank)
        is_pocket=False
        topo_rank=0
        for vid in face:
            leaf_node_index=self.vertex_to_leaf_cell[vid]
            if self.m_leaf_nodes[leaf_node_index].is_cylinder:
                topo_rank+=1
        # if np.any(self.m_mesh_v_adj_cylinder[face]):
        #     topo_rank=np.inf
        for edge in edges:
            if len(self.edge_to_faces[edge])==1:
                i0,i1=edge
                leaf_node_index0=self.vertex_to_leaf_cell[i0]
                leaf_node_index1=self.vertex_to_leaf_cell[i1]
                if self.feature_flags[leaf_node_index0] and self.feature_flags[leaf_node_index1]:
                #if self.m_v_type[i0]!=2 and self.m_v_type[i1]!=2:
                    return (np.inf,geo_rank)
                # elif len(self.convex_feature_points)==0 and self.m_leaf_nodes[leaf_node_index0].is_cylinder and self.m_leaf_nodes[leaf_node_index1].is_cylinder:
                elif self.m_leaf_nodes[leaf_node_index0].is_cylinder or self.m_leaf_nodes[leaf_node_index1].is_cylinder:
                    return (np.inf,geo_rank)
                else:
                    is_pocket=True               
            # elif len(self.edge_to_faces[edge])==4:
            #     quote_vertex_id0=edge[0]
            #     if self.quote_vertex_ids[edge[0]]!=-1:
            #         quote_vertex_id0=self.quote_vertex_ids[edge[0]]
            #     quote_vertex_id1=edge[1]
            #     if self.quote_vertex_ids[edge[1]]!=-1:
            #         quote_vertex_id1=self.quote_vertex_ids[edge[1]]
            #     if quote_vertex_id0!=quote_vertex_id1:
            #         topo_rank=-np.inf
        if not is_pocket:
            if face_id not in self.face2cavity or len(self.face2cavity[face_id])==0:
                return (np.inf,geo_rank)
        return (topo_rank,geo_rank)      
    

    def compute_all_cavity_face_rank_local(self,all_cavity_face_id):
        results=[]
        for face_id in all_cavity_face_id:
            if self.deleted_face_flag[face_id]:
                continue
            priority_tuple=self.compute_face_rank_local(face_id)
            results.append((face_id,priority_tuple))
        return results
    def compute_all_cavity_face_rank_global(self,all_cavity_face_id):
        results=[]
        for face_id in all_cavity_face_id:
            if self.deleted_face_flag[face_id]:
                continue
            priority_tuple=self.compute_face_rank_global(face_id)
            results.append((face_id,priority_tuple))
        return results

    def f_num(self):
        return len(self.deleted_face_flag)-np.count_nonzero(self.deleted_face_flag)           
    def v_num(self):
        count=0
        for i in range(len(self.m_mesh_v)):
            if i not in self.vertex_to_edge or len(self.vertex_to_edge[i])==0:
                continue
            count+=1
        return count
    def e_num(self):
        count=0
        for edge in self.edge_to_faces.keys():
            exist_flag=False
            if edge in self.edge_to_faces and len(self.edge_to_faces[edge])>0:
                for fid in self.edge_to_faces[edge]:
                    if self.deleted_face_flag[fid]==False:
                        exist_flag=True
                        break
                if exist_flag:
                    count+=1
        count+=len(self.alone_edges)
        return count
    def eular_characteristic(self):
        return self.v_num()-self.e_num()+self.f_num()-self.cavity_num()
    def setup_cavity_flags(self,group=None):
        cavity_flags=np.zeros(len(self.m_mesh_f),dtype=int)
        self.face2cavity={}
        self.edge2cavity={}
        self.vertex2cavity={}
        for(cavity_id,cavity_face_id) in enumerate(self.cavity_face_ids):
            if len(cavity_face_id)==6:
                if group is not None and cavity_id not in group:
                    continue
                for face_id in cavity_face_id:
                    face=self.m_mesh_f[face_id]
                    edges = [
                        tuple(sorted([face[0],face[1]])),
                        tuple(sorted([face[1],face[2]])),
                        tuple(sorted([face[2],face[3]])),
                        tuple(sorted([face[3],face[0]]))
                    ]
                    cavity_flags[face_id]=1
                    if face_id not in self.face2cavity:
                        self.face2cavity[face_id]=[]
                    if cavity_id not in self.face2cavity[face_id]:
                        self.face2cavity[face_id].append(cavity_id)
                    for edge in edges:
                        if edge not in self.edge2cavity:
                            self.edge2cavity[edge]=set()
                        if cavity_id not in self.edge2cavity[edge]:
                            self.edge2cavity[edge].add(cavity_id)
                    for vid in face:
                        if vid not in self.vertex2cavity:
                            self.vertex2cavity[vid]=set()
                        if cavity_id not in self.vertex2cavity[vid]:
                            self.vertex2cavity[vid].add(cavity_id)
        for edge in self.edge2cavity.keys():
            self.edge2cavity[edge]=list(self.edge2cavity[edge])
        for vid in self.vertex2cavity.keys():
            self.vertex2cavity[vid]=list(self.vertex2cavity[vid])
        return cavity_flags
    
    def update_cavity_tables(self,face_id):
        if face_id not in self.origin_face2cavity:
            return
        if self.deleted_face_flag[face_id]==1:
            #delete face
            for cavity_id in self.origin_face2cavity[face_id]:
                if face_id in self.cavity_face_ids[cavity_id]:
                    self.cavity_face_ids[cavity_id].remove(face_id)
            cur_cavity_ids=self.face2cavity[face_id]
            for cavity_id in cur_cavity_ids:
                self.face2cavity[face_id].remove(cavity_id)
                if len(self.cavity_face_ids[cavity_id])<6:
                    for fid in self.cavity_face_ids[cavity_id]:
                        self.face2cavity[fid].remove(cavity_id)
                    cavity_vertex_indexes=self.origin_cavity_vertex_indexes[cavity_id]
                    edge_indexes = [(0, 1), (0, 2), (2, 3), (3, 1),
                        (4, 5), (4, 6), (6, 7), (7, 5),
                        (0, 4), (1, 5), (2, 6), (3, 7)]
                    for edge_index in edge_indexes:
                        i0,i1=edge_index
                        edge=tuple(sorted([cavity_vertex_indexes[i0],cavity_vertex_indexes[i1]]))
                        if cavity_id in self.edge2cavity[edge]:
                            self.edge2cavity[edge].remove(cavity_id)
        else:
            #add face
            origin_cavity_ids=self.origin_face2cavity[face_id]
            for cavity_id in origin_cavity_ids:
                if face_id not in self.cavity_face_ids[cavity_id]:
                    self.cavity_face_ids[cavity_id].append(face_id)
                if len(self.cavity_face_ids[cavity_id])==6:
                    #cavity is complete
                    if cavity_id not in self.face2cavity[face_id]:
                        self.face2cavity[face_id].append(cavity_id)
                    for fid in self.cavity_face_ids[cavity_id]:
                        if cavity_id not in self.face2cavity[fid]:
                            self.face2cavity[fid].append(cavity_id)
                    cavity_vertex_indexes=self.origin_cavity_vertex_indexes[cavity_id]
                    edge_indexes = [(0, 1), (0, 2), (2, 3), (3, 1),
                        (4, 5), (4, 6), (6, 7), (7, 5),
                        (0, 4), (1, 5), (2, 6), (3, 7)]
                    for edge_index in edge_indexes:
                        i0,i1=edge_index
                        edge=tuple(sorted([cavity_vertex_indexes[i0],cavity_vertex_indexes[i1]]))
                        if edge not in self.edge2cavity:
                            self.edge2cavity[edge]=[]
                        if cavity_id not in self.edge2cavity[edge]:
                            self.edge2cavity[edge].append(cavity_id)

    def cavity_num(self,group=None):
        count=0
        for i in range(len(self.cavity_face_ids)):
            # if i==306:
            #     print(self.cavity_face_ids[i])
            if group is not None and i not in group:
                continue
            if len(self.cavity_face_ids[i])==6:
                count+=1
        return count
    def static_nonmannifold(self):
        extract_edges(self.m_mesh_f)
        for edge in self.edge_to_faces.keys():
            if len(self.edge_to_faces[edge])>2:
                return True
    def has_pocket(self,queue):
        face_ids=queue.traverse_via_map()
        for face_id in face_ids:
            if(self.is_pocket(face_id)):
                return True
        return False
    def has_avaliable_pocket(self,queue):
        face_ids=queue.traverse_via_map()
        for face_id in face_ids:
            if(self.is_avaliable_pocket(face_id)):
                return face_id
        return None
    def is_avaliable_pocket(self,face_id):
        face=self.m_mesh_f[face_id]
        for i in range(len(face)):
            edge=tuple(sorted([face[i],face[(i+1)%len(face)]]))
            if edge in self.edge_to_faces and len(self.edge_to_faces[edge])==1 :
                if self.deleted_face_flag[face_id]==0:
                    return True
        return False
    def is_pocket(self,face_id):
        face=self.m_mesh_f[face_id]
        for i in range(len(face)):
            edge=tuple(sorted([face[i],face[(i+1)%len(face)]]))
            if edge in self.edge_to_faces and len(self.edge_to_faces[edge])==1 :
                return True
        return False

   
    def update_relate_face_edge_vertex_table(self,face_id):
        face=self.m_mesh_f[face_id]
        edge0=tuple(sorted([face[0],face[1]]))
        edge1=tuple(sorted([face[1],face[2]]))
        edge2=tuple(sorted([face[2],face[3]]))
        edge3=tuple(sorted([face[3],face[0]]))
        edges=[edge0,edge1,edge2,edge3] 
        for edge in edges:
            if self.deleted_face_flag[face_id]==1 and edge in self.edge_to_faces:
                #delete face
                if face_id in self.edge_to_faces[edge]:
                    self.edge_to_faces[edge].remove(face_id)
                for fid in self.edge_to_faces[edge]:
                    if self.deleted_face_flag[fid]:
                        self.edge_to_faces[edge].remove(fid)

                if len(self.edge_to_faces[edge])==0:
                    id0,id1=edge
                    if edge in self.vertex_to_edge[id0]:
                        self.vertex_to_edge[id0].remove(edge)
                    if edge in self.vertex_to_edge[id1]:
                        self.vertex_to_edge[id1].remove(edge)
            else:
                #add face
                if edge not in self.edge_to_faces:
                    self.edge_to_faces[edge]=[]
                if face_id not in self.edge_to_faces[edge]:
                    self.edge_to_faces[edge].append(face_id)
                id0,id1=edge
                if id0 not in self.vertex_to_edge:
                    self.vertex_to_edge[id0]=[]
                if edge not in self.vertex_to_edge[id0]:
                    self.vertex_to_edge[id0].append(edge)
                if id1 not in self.vertex_to_edge:
                    self.vertex_to_edge[id1]=[]
                if edge not in self.vertex_to_edge[id1]:
                    self.vertex_to_edge[id1].append(edge)
    def update_relate_face_rank_local(self,face_id,queue):
        face=self.m_mesh_f[face_id]
        edge0=tuple(sorted([face[0],face[1]]))
        edge1=tuple(sorted([face[1],face[2]]))
        edge2=tuple(sorted([face[2],face[3]]))
        edge3=tuple(sorted([face[3],face[0]]))
        edges=[edge0,edge1,edge2,edge3]
        if self.deleted_face_flag[face_id]==0:
            #restore face
            priority_tuple=self.compute_face_rank_local(face_id)
            queue.update_priority(face_id,priority_tuple)

        for edge in edges:
            for fid in self.edge_to_faces[edge]:
                if fid==face_id:
                    continue
                if self.deleted_face_flag[fid]==1:
                    continue
                if queue.contains(fid):
                    priority_tuple=self.compute_face_rank_local(fid)
                    queue.update_priority(fid,priority_tuple) 
        if face_id in self.face2cavity:
            for old_cavity_id in self.origin_face2cavity[face_id]:
                for fid in self.cavity_face_ids[old_cavity_id]: 
                    if fid==face_id:
                        continue
                    if self.deleted_face_flag[fid]==1:
                        continue
                    priority_tuple=self.compute_face_rank_local(fid)
                    if queue.contains(fid):
                        queue.update_priority(fid,priority_tuple)
    def update_relate_face_rank_global(self,face_id,queue):
        face=self.m_mesh_f[face_id]
        edge0=tuple(sorted([face[0],face[1]]))
        edge1=tuple(sorted([face[1],face[2]]))
        edge2=tuple(sorted([face[2],face[3]]))
        edge3=tuple(sorted([face[3],face[0]]))
        edges=[edge0,edge1,edge2,edge3]
        if self.deleted_face_flag[face_id]==0:
            #restore face
            priority_tuple=self.compute_face_rank_global(face_id)
            queue.update_priority(face_id,priority_tuple)
        for edge in edges:
            for fid in self.edge_to_faces[edge]:
                if fid==face_id:
                    continue
                if self.deleted_face_flag[fid]==1:
                    continue
                if queue.contains(fid):
                    priority_tuple=self.compute_face_rank_global(fid)
                    queue.update_priority(fid,priority_tuple) 
        if face_id in self.face2cavity:
            for old_cavity_id in self.origin_face2cavity[face_id]:
                for fid in self.cavity_face_ids[old_cavity_id]: 
                    if fid==face_id:
                        continue
                    if self.deleted_face_flag[fid]==1:
                        continue
                    priority_tuple=self.compute_face_rank_global(fid)
                    if queue.contains(fid):
                        queue.update_priority(fid,priority_tuple)

    def update_relate_table_and_rank_global(self,face_id,queue):
        self.update_relate_face_edge_vertex_table(face_id)
        self.update_relate_face_rank_global(face_id,queue)    

    def update_relate_table_and_rank_local(self,face_id,queue):
        self.update_relate_face_edge_vertex_table(face_id)
        self.update_relate_face_rank_local(face_id,queue)


    def clean_cavity_face(self):
        self.edge_to_faces,self.vertex_to_edge=make_dict_edge2face(self.m_mesh_f,self.deleted_face_flag)
        self.degree_four_edges=[]
        for edge in self.edge_to_faces.keys():
            if len(self.edge_to_faces[edge])==4:
                self.degree_four_edges.append(edge)

        # cavity_flags=self.setup_cavity_flags()
        cavity_flags=self.origin_cavity_flags
        m_mesh_f_id=np.arange(len(self.m_mesh_f),dtype=int)
        all_cavity_face_id=m_mesh_f_id[cavity_flags.astype(bool)]
        count=0

        before_eular=self.eular_characteristic()
        for cavity_id in range(len(self.cavity_face_ids)):

            face_ids=self.cavity_face_ids[cavity_id]
            new_face_ids=[]
            for face_id in face_ids:
                if self.deleted_face_flag[face_id] or len(self.face2cavity[face_id])==2:
                    continue
                new_face_ids.append(face_id)
            if len(new_face_ids)==0:
                continue
            results=self.compute_all_cavity_face_rank_local(new_face_ids)
            #cavity_deleted_queue=MulPriorityQueue()
            cavity_deleted_queue=MulPriorityQueueInt()
            for result in results:
                cavity_deleted_queue.push(result[0],result[1])
            #--------------------------------
            while not cavity_deleted_queue.is_empty():
                face_id,priority_tuple=cavity_deleted_queue.pop()
                if priority_tuple[0]==np.inf:
                    #print(f'no available face to filter on this cavity:{cavity_id}')
                    break
                if self.deleted_face_flag[face_id]:
                    continue
                #delete_face_ids.append(face_id)
                count+=1
                
                self.deleted_face_flag[face_id]=True
                self.update_cavity_tables(face_id)

                self.update_relate_table_and_rank_local(face_id,cavity_deleted_queue)
                after_eular=self.eular_characteristic()
                if after_eular!=before_eular:
                    print(f'{face_id} eular characteristic not equal')
                    #delete_face_ids.remove(face_id)
                    self.deleted_face_flag[face_id]=False
                    self.update_cavity_tables(face_id)
                    self.update_relate_table_and_rank_local(face_id,cavity_deleted_queue)
                    after_eular=self.eular_characteristic()
                    count-=1
                    if(after_eular!=before_eular):
                        print(face_id)
                        np.save('deleted_face_flag.npy',self.deleted_face_flag)
                        SystemExit('eular characteristic not equal after rollback!')
                else:
                    self.deleted_face_map[face_id]=len(self.deleted_face_map)
        #========for end========
            
        #all cavity break
        # with open('delete_face_ids1.txt', 'w') as f:
        #     for face_id in delete_face_ids:
        #         print(f"{face_id}-->{self.origin_face2cavity[face_id]}:{self.m_mesh_f_udf[face_id]}",file=f)
        new_all_cavity_face_id=[]
        for fid in all_cavity_face_id:
            if self.deleted_face_flag[fid]:
                continue
            new_all_cavity_face_id.append(fid)
        all_cavity_face_id=new_all_cavity_face_id
        results=self.compute_all_cavity_face_rank_global(all_cavity_face_id)
        #deleted_queue=MulPriorityQueue()
        deleted_queue=MulPriorityQueueInt()
        for result in results:
            deleted_queue.push(result[0],result[1])

        while not deleted_queue.is_empty():
            face_id,priority_tuple=deleted_queue.pop()
            
            if priority_tuple[0]==np.inf:
                print('no available face to filter')
                break

            if self.deleted_face_flag[face_id]:
                continue

            #delete_face_ids.append(face_id)
            count+=1

            self.deleted_face_flag[face_id]=True
            self.update_cavity_tables(face_id)

            self.update_relate_table_and_rank_global(face_id,deleted_queue)

            
            after_eular=self.eular_characteristic()
            if after_eular!=before_eular:
                print(f'{face_id} eular characteristic not equal')
                #delete_face_ids.remove(face_id)
                self.deleted_face_flag[face_id]=False
                self.update_cavity_tables(face_id)
                self.update_relate_table_and_rank_global(face_id,deleted_queue)
                after_eular=self.eular_characteristic()
                count-=1
                if(after_eular!=before_eular):
                    print(face_id)
                    np.save('deleted_face_flag.npy',self.deleted_face_flag)
                    SystemExit('eular characteristic not equal')
            else:
                self.deleted_face_map[face_id]=len(self.deleted_face_map)    

                
        flag=(self.cavity_num()>0 or self.has_avaliable_pocket(deleted_queue)) and not deleted_queue.is_empty()
        if flag:
            print(f'has_avaliable_pocket:{self.has_avaliable_pocket(deleted_queue)}')
        print(f'filter {count} face done')
        print(f'remaining cavity num:{self.cavity_num()}')

    
    
    def filter_drop_nodes(self):
        #call in view code
        new_drop_nodes=[]
        for node in self.drop_nodes:
            x,y,z=node.m_min_corner_id//(2**self.m_sampling_depth)
            g_neighbor_offsets=[(-1,0,0),(1,0,0),(0,-1,0),(0,1,0),(0,0,-1),(0,0,1)]
            for offset in g_neighbor_offsets:
                nx,ny,nz=x+offset[0],y+offset[1],z+offset[2]
                if 0<=nx<self.resolution and 0<=ny<self.resolution and 0<=nz<self.resolution:
                    if self.occupied[nx,ny,nz]:
                        new_drop_nodes.append(node)
                        break
        self.drop_nodes=new_drop_nodes
    def scan_occupied_by_index(self):

        self.occupied = np.zeros((self.resolution, self.resolution, self.resolution), dtype=bool)
        self.occupied_cell = np.full((self.resolution, self.resolution, self.resolution), -1, dtype=int)
        self.occupied_cell_inverse = {}

        for leaf_node_index, node in enumerate(self.m_leaf_nodes):
            node.leaf_node_index = leaf_node_index

            x, y, z = node.m_min_corner_id // (2 ** self.m_sampling_depth)
            x, y, z = int(x), int(y), int(z)

            if not (0 <= x < self.resolution and 0 <= y < self.resolution and 0 <= z < self.resolution):
                raise ValueError(
                    f"scan_occupied_by_index: node.id={node.id} has out-of-bound grid coord {(x, y, z)}"
                )

            self.occupied[x, y, z] = True
            self.occupied_cell[x, y, z] = leaf_node_index
            self.occupied_cell_inverse[leaf_node_index] = (x, y, z)
    def build_leaf_nodes_adjacency(self):
        self.m_leaf_node_pos_types=np.zeros(len(self.m_leaf_nodes),dtype=int)
        self.m_leaf_nodes_adjacency=np.full((len(self.m_leaf_nodes), 6), -1)
        self.two_ring_adjacency=[]
        for leaf_node_index,node in enumerate(self.m_leaf_nodes):
            #-x,+x,-y,+y,-z,+z
            x,y,z=node.m_min_corner_id//(2**self.m_sampling_depth)
            g_neighbor_offsets=[(-1,0,0),(1,0,0),(0,-1,0),(0,1,0),(0,0,-1),(0,0,1)]
            #-x,+x,-y,+y,-z,+z
            POS_BIT_MASKS=[0b000001,0b000010,0b000100,0b001000,0b010000,0b100000]
            for index,(offset,pos_bit_mask) in enumerate(zip(g_neighbor_offsets,POS_BIT_MASKS)):
                nx,ny,nz=x+offset[0],y+offset[1],z+offset[2]
                if 0<=nx<self.resolution and 0<=ny<self.resolution and 0<=nz<self.resolution:
                    if self.occupied[nx,ny,nz]:
                        neighbor_index=self.occupied_cell[nx,ny,nz]
                        self.m_leaf_nodes_adjacency[leaf_node_index,index]=neighbor_index
                        self.m_leaf_node_pos_types[leaf_node_index] |= pos_bit_mask
        for leaf_node_index,node in enumerate(self.m_leaf_nodes):
            two_rihg_neighbors=[]
            #-x,+x,-y,+y,-z,+z
            x,y,z=node.m_min_corner_id//(2**self.m_sampling_depth)
            # g_neighbor_offsets=[(-1,-1,0),(-1,0,-1),(0,-1,-1),(1,1,0),(1,0,1),(0,1,1),(-2,0,0),(2,0,0),(0,-2,0),(0,2,0),(0,0,-2),(0,0,2)]
            g_neighbor_offsets=[(-1,-1,0),(-1,0,-1),(0,-1,-1),(1,1,0),(1,0,1),(0,1,1),(-1,1,0),(-1,0,1),(0,-1,1),(1,-1,0),(1,0,-1),(0,1,-1)]
            #-x,+x,-y,+y,-z,+z

            for offset in g_neighbor_offsets:
                nx,ny,nz=x+offset[0],y+offset[1],z+offset[2]
                if 0<=nx<self.resolution and 0<=ny<self.resolution and 0<=nz<self.resolution:
                    if self.occupied[nx,ny,nz]:
                        neighbor_index=self.occupied_cell[nx,ny,nz]
                        two_rihg_neighbors.append(neighbor_index)
            self.two_ring_adjacency.append(two_rihg_neighbors)



    def check_is_corner_type(self,leaf_node_index):
        node_pos_type=self.m_leaf_node_pos_types[leaf_node_index]#-x,+x,-y,+y,-z,+z
        x_bits = node_pos_type & 0b000011
        if x_bits not in [0b000001, 0b000010, 0b000011]:
            return False
        if x_bits==0b000001:
            adj_leaf_node_index=self.m_leaf_nodes_adjacency[leaf_node_index,0]
            if len(self.m_leaf_nodes[adj_leaf_node_index].inside_convex_feature_ids)==0:
                return False
        elif x_bits==0b000010:
            adj_leaf_node_index=self.m_leaf_nodes_adjacency[leaf_node_index,1]
            if len(self.m_leaf_nodes[adj_leaf_node_index].inside_convex_feature_ids)==0:
                return False
        # else:
        #     adj_leaf_node_index0=self.m_leaf_nodes_adjacency[leaf_node_index,0]
        #     adj_leaf_node_index1=self.m_leaf_nodes_adjacency[leaf_node_index,1]
        #     if len(self.m_leaf_nodes[adj_leaf_node_index0].inside_convex_feature_ids)==0 and len(self.m_leaf_nodes[adj_leaf_node_index1].inside_convex_feature_ids)==0:
        #         return False
        #     if len(self.m_leaf_nodes[adj_leaf_node_index0].inside_convex_feature_ids)>0 and len(self.m_leaf_nodes[adj_leaf_node_index1].inside_convex_feature_ids)>0:
        #         return False
        #================

        y_bits = node_pos_type & 0b001100
        if y_bits not in [0b000100, 0b001000,0b001100]:
            return False
        if y_bits==0b000100:
            adj_leaf_node_index=self.m_leaf_nodes_adjacency[leaf_node_index,2]
            if len(self.m_leaf_nodes[adj_leaf_node_index].inside_convex_feature_ids)==0:
                return False
        elif y_bits==0b001000:
            adj_leaf_node_index=self.m_leaf_nodes_adjacency[leaf_node_index,3]
            if len(self.m_leaf_nodes[adj_leaf_node_index].inside_convex_feature_ids)==0:
                return False
        # else:
        #     adj_leaf_node_index2=self.m_leaf_nodes_adjacency[leaf_node_index,2]
        #     adj_leaf_node_index3=self.m_leaf_nodes_adjacency[leaf_node_index,3]
        #     if len(self.m_leaf_nodes[adj_leaf_node_index2].inside_convex_feature_ids)==0 and len(self.m_leaf_nodes[adj_leaf_node_index3].inside_convex_feature_ids)==0:
        #         return False
        #     if len(self.m_leaf_nodes[adj_leaf_node_index2].inside_convex_feature_ids)>0 and len(self.m_leaf_nodes[adj_leaf_node_index3].inside_convex_feature_ids)>0:
        #         return False
        #================

        z_bits = node_pos_type & 0b110000
        if z_bits not in [0b010000, 0b100000,0b110000]:
            return False
        if z_bits==0b010000:
            adj_leaf_node_index=self.m_leaf_nodes_adjacency[leaf_node_index,4]
            if len(self.m_leaf_nodes[adj_leaf_node_index].inside_convex_feature_ids)==0:
                return False
        elif z_bits==0b100000:
            adj_leaf_node_index=self.m_leaf_nodes_adjacency[leaf_node_index,5]
            if len(self.m_leaf_nodes[adj_leaf_node_index].inside_convex_feature_ids)==0:
                return False

        return True




    def _precompute_node_data(self, centroids_coords):
        self.bounding_boxes = []
        for node in self.m_leaf_nodes:
            min_corner = self.convert_grid_to_coords(node.m_min_corner_id)
            max_corner = self.convert_grid_to_coords(node.m_max_corner_id)
            self.bounding_boxes.append((min_corner, max_corner))
        
        self.node_contain_corner_ids = []
        self.strict_node_contain_corner_ids = []
        for i,node in enumerate(self.m_leaf_nodes):
            # if i==10:
            #     print(i)
            self.node_contain_corner_ids.append(self.contain_feature_ids(self.convex_corner_points, node, offset=5e-2))
            self.strict_node_contain_corner_ids.append(self.contain_feature_ids(self.convex_corner_points, node))
        
        self.adj_nearest_leaf_node_indices = [-1 for _ in range(len(self.m_leaf_nodes))]
        for i, node in enumerate(self.m_leaf_nodes):
            if len(self.node_contain_corner_ids[i])>0 and self.check_is_corner_type(i):
                corner_ids = self.node_contain_corner_ids[i]
                corner_points = self.convex_corner_points[corner_ids]
                target_center = np.mean(corner_points, axis=0)
                self.adj_nearest_leaf_node_indices[i] = -1
                adj_node_ids = self.m_leaf_nodes_adjacency[i]
                min_dis = np.inf
                for adj_node_id in adj_node_ids:
                    if adj_node_id != -1 and len(self.m_leaf_nodes[adj_node_id].inside_convex_feature_ids) > 0 and len(self.node_contain_corner_ids[adj_node_id]) == 0:
                        adj_center = centroids_coords[adj_node_id]
                        adj_dis = np.linalg.norm(adj_center - target_center)
                        if adj_dis < min_dis:
                            min_dis = adj_dis
                            self.adj_nearest_leaf_node_indices[i] = adj_node_id
        
        self.adj_contain_outer_corner_flags = np.zeros(len(self.m_leaf_nodes), dtype=bool)
        self.adj_corner_node_id = [-1 for _ in range(len(self.m_leaf_nodes))]
        for i, node in enumerate(self.m_leaf_nodes):
            if len(node.inside_convex_feature_ids)==0:
                continue
            adj_node_ids = self.m_leaf_nodes_adjacency[i]
            for adj_id in adj_node_ids:
                if adj_id != -1 and len(self.node_contain_corner_ids[adj_id]) > 0:
                    self.adj_contain_outer_corner_flags[i] = True
                    self.adj_corner_node_id[i] = adj_id
                    break



    def batch_pre_solve_sqef(self,spoints,sp_convex_feature_mask,cloud_sdf_grad_mask_func,centroids_coords,singular_value_threshold,corner_factor=1e-2, edge_factor=1e-2, plane_factor=1e-2):

        from scipy.spatial import KDTree
        self.center_kdtree = KDTree(centroids_coords)
        self.sqefs=[None for _ in range(len(self.m_leaf_nodes))]
        self.results = [DualPoint() for _ in range(len(self.m_leaf_nodes))]
        self.feature_flags=np.zeros(len(self.m_leaf_nodes),dtype=bool)

        self.m_mesh_v=[]
        self.m_mesh_v_radii=[]
        self.m_mesh_v_simulated=[]
        self.m_v_type=[]
        self.cell_samples=[]
        self.mesh_samples=[]
        self._precompute_node_data(centroids_coords)
        has_feature=len(self.convex_feature_points)>0
        for i,node in enumerate(self.m_leaf_nodes):
            # if i==5210:
            #     pass
            min_corner,max_corner=self.bounding_boxes[i]
            center=centroids_coords[i]
            if len(self.node_contain_corner_ids[i])>0:
                if len(self.strict_node_contain_corner_ids[i])>0:
                    self.feature_flags[i]=True
                    corner_ids=self.strict_node_contain_corner_ids[i]
                    corner_points=self.convex_corner_points[corner_ids]
                    self.results[i]= DualPoint(corner_points[0], CORNER, None, True)
                else:
                    self.feature_flags[i]=True
                    corner_ids=self.node_contain_corner_ids[i]
                    corner_points=self.convex_corner_points[corner_ids]
                    self.results[i]= DualPoint(corner_points[0], CORNER, None, False)
                node.m_reconstruct_point=self.results[i].m_point
                self.cell_samples.append([])
                self.mesh_samples.append([])
            elif len(node.inside_convex_feature_ids)>0 and self.adj_contain_outer_corner_flags[i]==False:
                if len(node.strict_inside_convex_feature_ids)>0:
                    self.feature_flags[i]=True
                    feature_ids=node.strict_inside_convex_feature_ids
                    feature_points=self.convex_feature_points[feature_ids]
                    distances_sq = np.sum((feature_points - center) ** 2, axis=1)
                    min_idx = np.argmin(distances_sq)
                    nearest_point = feature_points[min_idx]             
                    self.results[i]= DualPoint(nearest_point, EDGE, None, True)
                    self.cell_samples.append([])
                    self.mesh_samples.append([])
                else:
                    self.feature_flags[i]=True
                    feature_ids=node.inside_convex_feature_ids
                    feature_points=self.convex_feature_points[feature_ids]
                    distances_sq = np.sum((feature_points - center) ** 2, axis=1)
                    min_idx = np.argmin(distances_sq)
                    nearest_point = feature_points[min_idx]             
                    self.results[i]= DualPoint(nearest_point, EDGE, None, False)
                    self.cell_samples.append([])
                    self.mesh_samples.append([])
                node.m_reconstruct_point=self.results[i].m_point
            elif len(node.inside_convex_feature_ids)>0 and self.adj_contain_outer_corner_flags[i]==True:
                self.feature_flags[i]=True
                corner_node_id=self.adj_corner_node_id[i]
                if self.adj_nearest_leaf_node_indices[corner_node_id]==i:
                    corner_ids=self.node_contain_corner_ids[corner_node_id]
                    corner_points=self.convex_corner_points[corner_ids]
                    best_v=np.mean(corner_points,axis=0)
                    self.results[i]= DualPoint(best_v, CORNER, None, False)
                    node.m_reconstruct_point=self.results[i].m_point
                    self.cell_samples.append([])
                    self.mesh_samples.append([])
                else:
                    if len(node.strict_inside_convex_feature_ids)>0:
                        self.feature_flags[i]=True
                        feature_ids=node.strict_inside_convex_feature_ids
                        feature_points=self.convex_feature_points[feature_ids]
                        distances_sq = np.sum((feature_points - center) ** 2, axis=1)
                        min_idx = np.argmin(distances_sq)
                        nearest_point = feature_points[min_idx]             
                        self.results[i]= DualPoint(nearest_point, EDGE, None, True)
                        self.cell_samples.append([])
                        self.mesh_samples.append([])
                    else:
                        self.feature_flags[i]=True
                        feature_ids=node.inside_convex_feature_ids
                        feature_points=self.convex_feature_points[feature_ids]
                        distances_sq = np.sum((feature_points - center) ** 2, axis=1)
                        min_idx = np.argmin(distances_sq)
                        nearest_point = feature_points[min_idx]             
                        self.results[i]= DualPoint(nearest_point, EDGE, None, False)
                        self.cell_samples.append([])
                        self.mesh_samples.append([])
                    node.m_reconstruct_point=self.results[i].m_point
            else:
                _,sdfs,sns,coords,proj_points,shape_point_ids,feature_flags=self.get_group_data(node)
                
                torch_feature = np.count_nonzero(feature_flags) > 0 
                nearest_points=spoints[shape_point_ids]
                if len(coords) > 1:
                    groups=node.groups
                    if sp_convex_feature_mask is not None and len(sp_convex_feature_mask)>0:
                        sp_convex_flags=sp_convex_feature_mask[shape_point_ids]
                        sdfs[sp_convex_flags]=abs(sdfs[sp_convex_flags])
                    if np.sum(sdfs>0)>0:
                        new_groups=[]
                        for group in groups:
                            group_sdfs=sdfs[group]
                            tmp_mask=group_sdfs<0
                            tmp_mask=tmp_mask.flatten()
                            if np.sum(tmp_mask)>0:
                                new_group=np.array(group)
                                new_group=new_group[tmp_mask]
                                new_groups.append(new_group)
                            # else:
                            #     new_group=group
                        if len(new_groups)>1:    
                            groups=new_groups
                    if has_feature:
                        if not node.is_cylinder and not torch_feature:
                            cosine_eps = 1 - np.cos(np.radians(self.angle_threshold))
                            normal_clustering = DBSCAN(eps=cosine_eps,
                                                        min_samples=1,
                                                        metric='cosine')
                            #normals = stableNormalized(nearest_points - coords) * (-np.sign(sdfs))
                            normal_labels = normal_clustering.fit_predict(sns)
                            uniq = np.unique(normal_labels)
                            if 2 <= len(uniq) < len(groups):
                                groups = []
                                for normal_label in uniq:
                                    normal_mask = (normal_labels == normal_label)
                                    cluster_indices = np.where(normal_mask)[0]
                                    groups.append(cluster_indices.tolist())
                    else:
                        if not node.is_cylinder:
                            cosine_eps = 1 - np.cos(np.radians(self.angle_threshold))
                            normal_clustering = DBSCAN(eps=cosine_eps,
                                                        min_samples=1,
                                                        metric='cosine')
                            #normals = stableNormalized(nearest_points - coords) * (-np.sign(sdfs))
                            normal_labels = normal_clustering.fit_predict(sns)
                            uniq = np.unique(normal_labels)
                            if 2 <= len(uniq) < len(groups):
                                groups = []
                                for normal_label in uniq:
                                    normal_mask = (normal_labels == normal_label)
                                    cluster_indices = np.where(normal_mask)[0]
                                    groups.append(cluster_indices.tolist())

                    # if len(coords)>4 and not node.is_cylinder and ((len(groups)==2 and torch_feature) or len(groups)==3):
                    if len(coords)>4 and not node.is_cylinder and (len(groups)==2 or len(groups)==3):
                        central_indices,new_groups=find_central_indices(nearest_points,groups,1)
                        #central_indices,new_groups=find_central_indices(nearest_points,groups,2)#tmp
                        sdfs=sdfs[central_indices]
                        sns=sns[central_indices]
                        coords=coords[central_indices]
                        proj_points=proj_points[central_indices]
                        nearest_points=nearest_points[central_indices]
                        feature_flags=feature_flags[central_indices]
                        #change groups
                        groups=new_groups
                    
                    normals = stableNormalized(nearest_points - coords) * (-np.sign(sdfs))
                    if has_feature:
                        normals[~feature_flags]=sns[~feature_flags]
                    sqef=SQEF(normals,nearest_points,coords)
                    self.sqefs[i]=sqef
                    one_ring_boxes = []
                    for adj_id in self.m_leaf_nodes_adjacency[i]:
                        if adj_id != -1:
                            one_ring_boxes.append(self.bounding_boxes[adj_id])
                    two_ring_boxes = []
                    for adj_id in self.two_ring_adjacency[i]:
                        if adj_id != -1:
                            two_ring_boxes.append(self.bounding_boxes[adj_id])
                    self.results[i] = solve_point_pre_sqef_with_group(
                        sqef,
                        min_corner,
                        max_corner,
                        groups,
                        node.is_cylinder,
                        singular_value_threshold,
                        one_ring_boxes=one_ring_boxes,
                        two_ring_boxes=two_ring_boxes,
                        corner_factor=corner_factor,
                        edge_factor=edge_factor,
                        plane_factor=plane_factor,
                    )
                    if self.results[i].m_point is not None and self.results[i].radius<0:
                        self.results[i].m_is_inside = False  
                        self.m_leaf_nodes[i].m_reconstruct_point = self.results[i].m_point
                        self.results[i].m_point=None
                    if not self.results[i].m_is_inside:
                        node.is_cylinder=False
                        
                    if self.results[i].m_is_inside and self.results[i].m_point is not None:
                        
                        sdf, _, _, _,_ = query_sdf_and_sn_and_proj_and_feature_mask(cloud_sdf_grad_mask_func, self.results[i].m_point.reshape(-1, 3))
                        sdf = sdf.reshape(-1)
                        if sdf[0] > 5e-3:
                            self.results[i].m_is_inside = False  
                            self.m_leaf_nodes[i].m_reconstruct_point = self.results[i].m_point
                            self.results[i].m_point=None
                            
                    node.m_reconstruct_point=self.results[i].m_point

                self.cell_samples.append(coords)
                self.mesh_samples.append(nearest_points)

    def compute_tangent_plane(self,center,cloud_sdf_grad_mask_func):
        has_feature=len(self.convex_feature_points)>0
        min_corner=center-self.box_edge_length*0.125
        max_corner=center+self.box_edge_length*0.125
        origin_coords=generate_linespace_points(min_corner,max_corner,5)
        origin_sdfs, origin_sns,origin_nearest_points,_,origin_feature_flags=query_sdf_and_sn_and_proj_and_feature_mask(cloud_sdf_grad_mask_func, origin_coords.astype(np.float32)) 

        mask=origin_sdfs[:,0]!=0
        origin_sdfs=origin_sdfs.reshape(-1,1)
        origin_feature_flags=origin_feature_flags.reshape(-1)

        sdfs = origin_sdfs[mask]
        sns=origin_sns[mask]
        coords = origin_coords[mask]
        nearest_points = origin_nearest_points[mask]
        feature_flags = origin_feature_flags[mask]
        if len(sdfs)==0:
            return np.inf
        normals = stableNormalized(nearest_points - coords) * (-np.sign(sdfs))
        #normals[feature_flags]=sns[feature_flags]
        norms = np.linalg.norm(nearest_points - coords, axis=1)
        mask1=norms<self.norm_eps
        normals[mask1]=sns[mask1]

        position_eps=np.radians(40.0)*max(norms)
        distance_threshold=self.box_edge_length*0.1
        enough,groups=compute_groups(sdfs,sns,coords,nearest_points,feature_flags,self.norm_eps,max(self.position_eps,position_eps),distance_threshold)
        if enough:
            sorted_groups = sorted(groups, key=lambda x: len(x), reverse=True)
            plane_groups=sorted_groups[:2]
            if len(coords)>4 and has_feature:
                central_indices,new_groups=find_central_indices(nearest_points,plane_groups)
                sdfs=sdfs[central_indices]
                sns=sns[central_indices]
                coords=coords[central_indices]
                nearest_points=nearest_points[central_indices]
                feature_flags=feature_flags[central_indices]
                #change groups
                groups=new_groups
            else:
                central_indices=plane_groups[0] + plane_groups[1]
                sdfs=sdfs[central_indices]
                sns=sns[central_indices]
                coords=coords[central_indices]
                nearest_points=nearest_points[central_indices]
                feature_flags=feature_flags[central_indices]
                #change groups
                groups=plane_groups
            normals = stableNormalized(nearest_points - coords) * (-np.sign(sdfs))
            norms = np.linalg.norm(nearest_points - coords, axis=1)
            mask1=norms<self.norm_eps

            normals[mask1]=sns[mask1]
            sqef=SQEF(normals,nearest_points,coords)
            sqef.solve_in_plane()
            dis= point_to_plane_distance(center,sqef.point[:3],sqef.dir)
            return dis
        return np.inf
    def batch_compute_face_udf(self,cloud_sdf_grad_mask_func):
        self.m_mesh_f_udfs=np.zeros((len(self.m_mesh_f),2),dtype=float)
        for fid,face in enumerate(self.m_mesh_f):
            face_unique=self.m_mesh_f_unique[fid]
            if(len(np.unique(face_unique))<=2):
                self.m_mesh_f_udfs[fid]=0
            face_vertices=self.m_mesh_v[face]
            #========topo 0========
            tri00=np.array([face_unique[0],face_unique[1],face_unique[2]])
            tri01=np.array([face_unique[0],face_unique[2],face_unique[3]])
            dis00=0
            dis01=0
            if(len(np.unique(tri00))==3):
                center00=(face_vertices[0]+face_vertices[1]+face_vertices[2])/3.0
                dis00=self.compute_tangent_plane(center00,cloud_sdf_grad_mask_func)
            if(len(np.unique(tri01))==3):
                center01=(face_vertices[0]+face_vertices[2]+face_vertices[3])/3.0
                dis01=self.compute_tangent_plane(center01,cloud_sdf_grad_mask_func)
            udf0=max(dis00,dis01)
            #========topo 1========
            tri10=np.array([face_unique[0],face_unique[1],face_unique[3]])
            tri11=np.array([face_unique[1],face_unique[2],face_unique[3]])
            dis10=0
            dis11=0
            if(len(np.unique(tri10))==3):
                center10=(face_vertices[0]+face_vertices[1]+face_vertices[3])/3.0
                dis10=self.compute_tangent_plane(center10,cloud_sdf_grad_mask_func)
            if(len(np.unique(tri11))==3):
                center11=(face_vertices[1]+face_vertices[2]+face_vertices[3])/3.0
                dis11=self.compute_tangent_plane(center11,cloud_sdf_grad_mask_func)
            udf1=max(dis10,dis11)
            self.m_mesh_f_udfs[fid,0]=udf0
            self.m_mesh_f_udfs[fid,1]=udf1
            


    def scan_cavity(self):
        self.cavity_face_ids=[]
        self.cavity_leaf_node_indexes=[]
        self.cavity_vertex_indexes=[]
        for node in self.m_leaf_nodes:
            x,y,z=node.m_min_corner_id//(2**self.m_sampling_depth)
            if x<self.resolution-1 and y<self.resolution-1 and z<self.resolution-1:
                check=True
                cavity_leaf_node_index=[]
                cavity_vertex_index=[]
                for offset in g_child_order:
                    if self.occupied[x+offset[0],y+offset[1],z+offset[2]]==False:
                        check=False
                        break
                    leaf_node_index=self.occupied_cell[x+offset[0],y+offset[1],z+offset[2]]
                    if self.m_leaf_nodes[leaf_node_index].m_node_type=="empty":
                        check=False
                        break
                    cavity_leaf_node_index.append(leaf_node_index)
                    cavity_vertex_index.append(self.m_leaf_nodes[leaf_node_index].m_vertex_idx)
                if check:
                    #x-neg
                    quad0=[cavity_leaf_node_index[0],cavity_leaf_node_index[1],cavity_leaf_node_index[2],cavity_leaf_node_index[3]]
                    #x-pos
                    quad1=[cavity_leaf_node_index[4],cavity_leaf_node_index[5],cavity_leaf_node_index[6],cavity_leaf_node_index[7]]
                    #y-neg
                    quad2=[cavity_leaf_node_index[0],cavity_leaf_node_index[1],cavity_leaf_node_index[5],cavity_leaf_node_index[4]]
                    #y-pos
                    quad3=[cavity_leaf_node_index[2],cavity_leaf_node_index[3],cavity_leaf_node_index[7],cavity_leaf_node_index[6]]
                    #z-neg
                    quad4=[cavity_leaf_node_index[0],cavity_leaf_node_index[2],cavity_leaf_node_index[6],cavity_leaf_node_index[4]]
                    #z-pos
                    quad5=[cavity_leaf_node_index[1],cavity_leaf_node_index[3],cavity_leaf_node_index[7],cavity_leaf_node_index[5]]
                    quads=[quad0,quad1,quad2,quad3,quad4,quad5]
                    cavity_face_id=[]
                    skip=False
                    for quad in quads:
                        if tuple(sorted(quad)) not in self.cell_quad2face_id:
                            print("cavity face id not in cell_quad2face_id,skip it")
                            skip=True
                            break
                        face_id=self.cell_quad2face_id[tuple(sorted(quad))]
                        cavity_face_id.append(face_id)
                    if skip:
                        continue
                    self.cavity_face_ids.append(cavity_face_id)
                    self.cavity_leaf_node_indexes.append(cavity_leaf_node_index)
                    self.cavity_vertex_indexes.append(cavity_vertex_index)
        print(f"scan {len(self.cavity_face_ids)} cavity done!")
    def check_edge_exist(self,edge):
        if edge in self.alone_edges:
            return True
        if edge in self.edge_to_faces and len(self.edge_to_faces[edge])>0:
            for fid in self.edge_to_faces[edge]:
                if self.deleted_face_flag[fid]==False:
                    return True
            return False
        return False
    def add_edge_to_vertex_map(self,edge):
        v0,v1=edge
        if v0 not in self.vertex_to_edge:
            self.vertex_to_edge[v0]=[edge]
        elif edge not in self.vertex_to_edge[v0]:
            self.vertex_to_edge[v0].append(edge)

        if v1 not in self.vertex_to_edge:
            self.vertex_to_edge[v1]=[edge]
        elif edge not in self.vertex_to_edge[v1]:
            self.vertex_to_edge[v1].append(edge)
    def generate_alone_edges_for_cad(self):
        has_feature=len(self.convex_feature_points)>0
        if not has_feature:
            return
        self.alone_edges=set()
        for node_id in self.cylinder_node_ids:
            node=self.m_leaf_nodes[node_id]
            cur_vertex_idx=node.m_vertex_idx
            if cur_vertex_idx==-1:
                continue

            node_pos_type=self.m_leaf_node_pos_types[node_id]
            x_bits = node_pos_type & 0b000011
            y_bits = node_pos_type & 0b001100
            z_bits = node_pos_type & 0b110000
            adj_node_id0=-1
            adj_node_id1=-1
            if x_bits==0b000011:
                adj_node_id0=self.m_leaf_nodes_adjacency[node_id,0]
                adj_node_id1=self.m_leaf_nodes_adjacency[node_id,1]
                if adj_node_id0==-1 or adj_node_id1==-1:
                    continue
            elif y_bits==0b001100:
                adj_node_id0=self.m_leaf_nodes_adjacency[node_id,2]
                adj_node_id1=self.m_leaf_nodes_adjacency[node_id,3]
                if adj_node_id0==-1 or adj_node_id1==-1:
                    continue
            elif z_bits==0b110000:
                adj_node_id0=self.m_leaf_nodes_adjacency[node_id,4]
                adj_node_id1=self.m_leaf_nodes_adjacency[node_id,5]
                if adj_node_id0==-1 or adj_node_id1==-1:
                    continue
            #================================
            if adj_node_id0!=-1 and self.m_leaf_nodes[adj_node_id0].m_vertex_idx!=-1:
                adj_node_vertex_id0=self.m_leaf_nodes[adj_node_id0].m_vertex_idx
                edge=tuple(sorted((cur_vertex_idx, adj_node_vertex_id0)))
                if self.check_edge_exist(edge):
                    continue
                if self.m_leaf_nodes[adj_node_id0].is_cylinder:
                    self.alone_edges.add(edge)
                    self.add_edge_to_vertex_map(edge)
                else:
                    if adj_node_vertex_id0 not in self.vertex_to_edge:
                        if len(self.m_leaf_nodes[adj_node_id0].be_quoted_vertex_ids)>0:
                            be_quoted_vertex_id=self.m_leaf_nodes[adj_node_id0].be_quoted_vertex_ids[0]
                            self.alone_edges.add(edge)
                            self.add_edge_to_vertex_map(edge)
                            connect_edge=tuple(sorted((be_quoted_vertex_id, adj_node_vertex_id0)))
                            if not self.check_edge_exist(connect_edge):
                                self.alone_edges.add(connect_edge)
                                self.add_edge_to_vertex_map(connect_edge)
                    else:
                        self.alone_edges.add(edge)
                        self.vertex_to_edge[adj_node_vertex_id0].append(edge)
            #================================
            if adj_node_id1!=-1 and self.m_leaf_nodes[adj_node_id1].m_vertex_idx!=-1:
                adj_node_vertex_id1=self.m_leaf_nodes[adj_node_id1].m_vertex_idx
                edge=tuple(sorted((cur_vertex_idx, adj_node_vertex_id1)))
                if self.check_edge_exist(edge):
                    continue
                if self.m_leaf_nodes[adj_node_id1].is_cylinder:
                    self.alone_edges.add(edge)
                    self.add_edge_to_vertex_map(edge)
                else:
                    if adj_node_vertex_id1 not in self.vertex_to_edge:
                        if len(self.m_leaf_nodes[adj_node_id1].be_quoted_vertex_ids)>0:
                            be_quoted_vertex_id=self.m_leaf_nodes[adj_node_id1].be_quoted_vertex_ids[0]
                            self.alone_edges.add(edge)
                            self.add_edge_to_vertex_map(edge)
                            connect_edge=tuple(sorted((be_quoted_vertex_id, adj_node_vertex_id1)))
                            if not self.check_edge_exist(connect_edge):
                                self.alone_edges.add(connect_edge)
                                self.add_edge_to_vertex_map(connect_edge)
                        else:
                            continue
                    else:
                        self.alone_edges.add(edge)
                        self.vertex_to_edge[adj_node_vertex_id1].append(edge)

    def generate_alone_edges(self):
        if len(self.convex_feature_points)>0:
            return
        for leaf_node_ndex,node in enumerate(self.m_leaf_nodes):
            if node.m_vertex_idx==-1:
                continue
            if not node.is_cylinder:
                continue
            vid=node.m_vertex_idx
     
            for adj_leaf_node_index in self.m_leaf_nodes_adjacency[leaf_node_ndex]:
                if adj_leaf_node_index!=-1 and self.m_leaf_nodes[adj_leaf_node_index].m_vertex_idx!=-1:
                    
                    adj_leaf_node_vertex_id=self.m_leaf_nodes[adj_leaf_node_index].m_vertex_idx
                    edge=tuple(sorted((vid, adj_leaf_node_vertex_id)))
                    if self.check_edge_exist(edge):
                        continue
                    if self.m_leaf_nodes[adj_leaf_node_index].is_cylinder:
                        self.alone_edges.add(edge)
                        self.add_edge_to_vertex_map(edge)
                    else:
                        if adj_leaf_node_vertex_id not in self.vertex_to_edge:
                            if len(self.m_leaf_nodes[adj_leaf_node_index].be_quoted_vertex_ids)>0:
                                be_quoted_vertex_id=self.m_leaf_nodes[adj_leaf_node_index].be_quoted_vertex_ids[0]
                                self.alone_edges.add(edge)
                                self.add_edge_to_vertex_map(edge)

                                connect_edge=tuple(sorted((be_quoted_vertex_id, adj_leaf_node_vertex_id)))
                                if not self.check_edge_exist(connect_edge):
                                    self.alone_edges.add(connect_edge)
                                    self.add_edge_to_vertex_map(connect_edge)
                        else:
                            self.alone_edges.add(edge)
                            self.add_edge_to_vertex_map(edge)

    def generate_alone_edges_backup(self):
        has_feature=len(self.convex_feature_points)>0
        if has_feature:
            return
        before_eular=self.eular_characteristic()
        uf=UnionFindWithSize(len(self.m_leaf_nodes))
        for cell_quad in self.m_mesh_cell_id:
            uf.union(cell_quad[0],cell_quad[1])
            uf.union(cell_quad[1],cell_quad[2])
            uf.union(cell_quad[2],cell_quad[3])

        for leaf_node_ndex,node in enumerate(self.m_leaf_nodes):
            if node.m_vertex_idx==-1:
                continue
            vid=node.m_vertex_idx
            root_cur=uf.find_iterative(leaf_node_ndex)
            for adj_leaf_node_index in self.m_leaf_nodes_adjacency[leaf_node_ndex]:
                if adj_leaf_node_index!=-1 and self.m_leaf_nodes[adj_leaf_node_index].m_vertex_idx!=-1:
                    adj_leaf_node_vertex_id=self.m_leaf_nodes[adj_leaf_node_index].m_vertex_idx
                    edge=tuple(sorted((vid, adj_leaf_node_vertex_id)))
                    
                    if self.check_edge_exist(edge):
                        continue
                    root_adj=uf.find_iterative(adj_leaf_node_index)
                    if(root_cur==root_adj):
                        continue
                    #add edge data
                    uf.union(leaf_node_ndex,adj_leaf_node_index)
                    self.alone_edges.add(edge)
                    
                    if adj_leaf_node_vertex_id not in self.vertex_to_edge:
                        self.vertex_to_edge[adj_leaf_node_vertex_id]=[edge]
                    elif edge not in self.vertex_to_edge[adj_leaf_node_vertex_id]:
                        self.vertex_to_edge[adj_leaf_node_vertex_id].append(edge)

                    if vid not in self.vertex_to_edge:
                        self.vertex_to_edge[vid]=[edge]
                    elif edge not in self.vertex_to_edge[vid]:
                        self.vertex_to_edge[vid].append(edge)
                    root_cur=uf.find_iterative(leaf_node_ndex)
                    after_eular=self.eular_characteristic()
                    if before_eular!=after_eular:
                        print(before_eular)

        for leaf_node_ndex,node in enumerate(self.m_leaf_nodes):
            if node.m_vertex_idx==-1:
                continue
            vid=node.m_vertex_idx
            root_cur=uf.find_iterative(leaf_node_ndex)
            for adj_leaf_node_index in self.two_ring_adjacency[leaf_node_ndex]:
                if adj_leaf_node_index!=-1 and self.m_leaf_nodes[adj_leaf_node_index].m_vertex_idx!=-1:
                    adj_leaf_node_vertex_id=self.m_leaf_nodes[adj_leaf_node_index].m_vertex_idx
                    edge=tuple(sorted((vid, adj_leaf_node_vertex_id)))
                    if self.check_edge_exist(edge):
                        continue
                    root_adj=uf.find_iterative(adj_leaf_node_index)
                    if(root_cur==root_adj):
                        continue
                    #add edge data
                    uf.union(leaf_node_ndex,adj_leaf_node_index)
                    self.alone_edges.add(edge)
                    
                    if adj_leaf_node_vertex_id not in self.vertex_to_edge:
                        self.vertex_to_edge[adj_leaf_node_vertex_id]=[edge]
                    elif edge not in self.vertex_to_edge[adj_leaf_node_vertex_id]:
                        self.vertex_to_edge[adj_leaf_node_vertex_id].append(edge)

                    if vid not in self.vertex_to_edge:
                        self.vertex_to_edge[vid]=[edge]
                    elif edge not in self.vertex_to_edge[vid]:
                        self.vertex_to_edge[vid].append(edge)
                    root_cur=uf.find_iterative(leaf_node_ndex)
                    after_eular=self.eular_characteristic()
                    if before_eular!=after_eular:
                        print(before_eular)
    def connect_edge_to_adj_node(self,cur_vertex_idx,adj_node_id0,adj_node_id1):
        #================================
        if adj_node_id0!=-1 and self.m_leaf_nodes[adj_node_id0].m_vertex_idx!=-1:
            adj_node_vertex_id0=self.m_leaf_nodes[adj_node_id0].m_vertex_idx
            edge=tuple(sorted((cur_vertex_idx, adj_node_vertex_id0)))
            if not self.check_edge_exist(edge):
                if self.m_leaf_nodes[adj_node_id0].is_cylinder:
                    self.alone_edges.add(edge)
                    self.add_edge_to_vertex_map(edge)
                elif self.m_v_type[adj_node_vertex_id0]!=2:
                #else:
                    if adj_node_vertex_id0 not in self.vertex_to_edge:
                        if len(self.m_leaf_nodes[adj_node_id0].be_quoted_vertex_ids)>0:
                            be_quoted_vertex_id=self.m_leaf_nodes[adj_node_id0].be_quoted_vertex_ids[0]
                            self.alone_edges.add(edge)
                            self.add_edge_to_vertex_map(edge)
                            connect_edge=tuple(sorted((be_quoted_vertex_id, adj_node_vertex_id0)))
                            if not self.check_edge_exist(connect_edge):
                                self.alone_edges.add(connect_edge)
                                self.add_edge_to_vertex_map(connect_edge)
                    else:
                        self.alone_edges.add(edge)
                        self.add_edge_to_vertex_map(edge)
        #================================
        if adj_node_id1!=-1 and self.m_leaf_nodes[adj_node_id1].m_vertex_idx!=-1:
            adj_node_vertex_id1=self.m_leaf_nodes[adj_node_id1].m_vertex_idx
            edge=tuple(sorted((cur_vertex_idx, adj_node_vertex_id1)))
            if not self.check_edge_exist(edge):
                if self.m_leaf_nodes[adj_node_id1].is_cylinder:
                    self.alone_edges.add(edge)
                    self.add_edge_to_vertex_map(edge)
                elif self.m_v_type[adj_node_vertex_id1]!=2:
                #else:
                    if adj_node_vertex_id1 not in self.vertex_to_edge:
                        if len(self.m_leaf_nodes[adj_node_id1].be_quoted_vertex_ids)>0:
                            be_quoted_vertex_id=self.m_leaf_nodes[adj_node_id1].be_quoted_vertex_ids[0]
                            self.alone_edges.add(edge)
                            self.add_edge_to_vertex_map(edge)
                            connect_edge=tuple(sorted((be_quoted_vertex_id, adj_node_vertex_id1)))
                            if not self.check_edge_exist(connect_edge):
                                self.alone_edges.add(connect_edge)
                                self.add_edge_to_vertex_map(connect_edge)
                        else:
                            return
                    else:
                        self.alone_edges.add(edge)
                        self.add_edge_to_vertex_map(edge)

    def generate_alone_edges_for_cad(self):
        has_feature=len(self.convex_feature_points)>0
        if not has_feature:
            return
        self.alone_edges=set()
        for leaf_node_index in self.cylinder_node_ids:
            node=self.m_leaf_nodes[leaf_node_index]
            cur_vertex_idx=node.m_vertex_idx
            if cur_vertex_idx==-1:
                continue

            node_pos_type=self.m_leaf_node_pos_types[leaf_node_index]
            x_bits = node_pos_type & 0b000011
            y_bits = node_pos_type & 0b001100
            z_bits = node_pos_type & 0b110000
            adj_node_id0=-1
            adj_node_id1=-1
            if x_bits==0b000011:
                adj_node_id0=self.m_leaf_nodes_adjacency[leaf_node_index,0]
                adj_node_id1=self.m_leaf_nodes_adjacency[leaf_node_index,1]
                if adj_node_id0!=-1 and adj_node_id1!=-1:
                    if self.m_leaf_nodes[adj_node_id0].is_cylinder or self.m_leaf_nodes[adj_node_id1].is_cylinder:
                        self.connect_edge_to_adj_node(cur_vertex_idx,adj_node_id0,adj_node_id1)
                
            if y_bits==0b001100:
                adj_node_id0=self.m_leaf_nodes_adjacency[leaf_node_index,2]
                adj_node_id1=self.m_leaf_nodes_adjacency[leaf_node_index,3]
                if adj_node_id0!=-1 and adj_node_id1!=-1:
                    if self.m_leaf_nodes[adj_node_id0].is_cylinder or self.m_leaf_nodes[adj_node_id1].is_cylinder:
                        self.connect_edge_to_adj_node(cur_vertex_idx,adj_node_id0,adj_node_id1)

            if z_bits==0b110000:
                adj_node_id0=self.m_leaf_nodes_adjacency[leaf_node_index,4]
                adj_node_id1=self.m_leaf_nodes_adjacency[leaf_node_index,5]
                if adj_node_id0!=-1 and adj_node_id1!=-1:
                    if self.m_leaf_nodes[adj_node_id0].is_cylinder or self.m_leaf_nodes[adj_node_id1].is_cylinder:
                        self.connect_edge_to_adj_node(cur_vertex_idx,adj_node_id0,adj_node_id1)
            
    def generate_mesh(self):
        self.cellProcContour(self.m_root_node)
        self.m_mesh_v=np.array(self.m_mesh_v)
        self.m_v_type=np.array(self.m_v_type)
        self.m_mesh_v_radii=np.array(self.m_mesh_v_radii)
        self.m_mesh_v_simulated=np.array(self.m_mesh_v_simulated)
        self.m_mesh_f=np.array(self.m_mesh_f,dtype=int)
        self.m_mesh_cell_type=np.array(self.m_mesh_cell_type)
        self.m_mesh_cell_id=np.array(self.m_mesh_cell_id)


    def cellProcContour(self, node: OctreeNode):
        if node.m_node_type == "interval":
            for i in range(8):
                self.cellProcContour(node.m_children[i])
            for i in range(12):
                c=[cellProcFaceMask[i,0],cellProcFaceMask[i,1]]
                fcd=[node.m_children[c[0]],node.m_children[c[1]]]
                self.faceProcContour(fcd,cellProcFaceMask[i,2])
            # 6 edge calls
            for i in range(6):
                c = [ cellProcEdgeMask[i,0], cellProcEdgeMask[i,1], cellProcEdgeMask[i,2], cellProcEdgeMask[i,3] ]
                ecd=[]
                for j in range(4):
                    ecd.append(node.m_children[c[j]])

                self.edgeProcContour(ecd, cellProcEdgeMask[i,4])

    def faceProcContour(self,node,dir):
        if (node[0].m_node_type == "empty" or node[1].m_node_type == "empty"):
            return
        if (node[0].m_node_type == "interval" or node[1].m_node_type == "interval"):
            for i in range(4):
                c = [ faceProcFaceMask[dir,i,0], faceProcFaceMask[dir,i,1] ]
                fcd=[]
                for j in range(2):
                    if (node[j].m_node_type == "leaf"):
                        fcd.append(node[j])
                    else:
                        fcd.append(node[j].m_children[c[j]])
                self.faceProcContour(fcd, faceProcFaceMask[dir,i,2])

            orders= np.array([ [ 0, 0, 1, 1], [ 0, 1, 0, 1 ] ])
            #[1, 4, 0, 5, 1, 1]
            #-->
            #ecd=[node[order[0]].m_children[4],node[order[1]].m_children[0],node[order[0]].m_children[5],node[order[1]].m_children[1]]
            for i in range(4):
                c = [ faceProcEdgeMask[dir,i,1], faceProcEdgeMask[dir,i,2],faceProcEdgeMask[dir,i,3], faceProcEdgeMask[dir,i,4] ]
                order = orders[faceProcEdgeMask[dir,i,0]]
                ecd=[]
                for j in range(4):
                    if (node[order[j]].m_node_type == "leaf"):
                        ecd.append(node[order[j]])
                    
                    else:
                        ecd.append(node[order[j]].m_children[c[j]])
                self.edgeProcContour(ecd, faceProcEdgeMask[dir,i,5])
     
    def edgeProcContour(self,nodes,dir):
        for i in range(4):
            if (nodes[i].m_node_type == "empty"):
                return
        
        if nodes[0].m_node_type == "leaf" and nodes[1].m_node_type == "leaf" and nodes[2].m_node_type == "leaf" and nodes[3].m_node_type == "leaf":
            self.processEdgeWrite(nodes,dir)
        else:
            for i in range(2):
                c = [ edgeProcEdgeMask[dir,i,0],
                            edgeProcEdgeMask[dir,i,1],
                            edgeProcEdgeMask[dir,i,2],
                            edgeProcEdgeMask[dir,i,3] ]
                ecd=[]
                for j in range(4):
                    if (nodes[j].m_node_type == "leaf"):
                        ecd.append(nodes[j])
                    else:
                        ecd.append(nodes[j].m_children[c[j]])
                self.edgeProcContour(ecd, edgeProcEdgeMask[dir,i,4])
    
    def processEdgeWrite(self,nodes,dir):
        # min_corner_ids=[]
        # max_corner_ids=[]
        indexes=np.arange(3).tolist()
        indexes.remove(dir)
        inds=[]
        inds_unique=[]
        leaf_node_indexes=[]
        node_cell_types=[]
        # map_node_ids=[]
        feature_count=0
        for i in range(4):
            inds.append(nodes[i].m_vertex_idx)
            if nodes[i].quote_vertex_id!=-1:
                inds_unique.append(nodes[i].quote_vertex_id)
            else:
                inds_unique.append(nodes[i].m_vertex_idx)
            if self.feature_flags[nodes[i].leaf_node_index]:
                feature_count+=1
            leaf_node_indexes.append(nodes[i].leaf_node_index)
            # map_node_ids.append(self.vertex_to_leaf_cell[nodes[i].m_vertex_idx])
            node_cell_types.append(nodes[i].m_node_cell_type)
            # min_corner_ids.append(nodes[i].m_min_corner_id)
            # max_corner_ids.append(nodes[i].m_max_corner_id)

        # min_corner_id=np.min(min_corner_ids,axis=0)
        # max_corner_id=np.max(max_corner_ids,axis=0)
        # edge_ids=np.vstack([min_corner_id,max_corner_id])
        # edge_ids[:,indexes]=(min_corner_id[indexes]+max_corner_id[indexes])// 2
        # edge=self.convert_grid_to_coords(edge_ids)

        
        face_quad=[ inds[0], inds[1], inds[3], inds[2] ]
        face_quad_unique=[ inds_unique[0], inds_unique[1], inds_unique[3], inds_unique[2] ]
        cell_quad=[ leaf_node_indexes[0], leaf_node_indexes[1], leaf_node_indexes[3], leaf_node_indexes[2] ]
        cell_quad_types=[ node_cell_types[0], node_cell_types[1], node_cell_types[3], node_cell_types[2] ]
        self.cell_quad2face_id[tuple(sorted(cell_quad))]=len(self.m_mesh_f)
        self.cell_quad2face_id_inverse[len(self.m_mesh_f)]=cell_quad
        self.cell_quad2face[tuple(sorted(cell_quad))]=face_quad
        self.m_mesh_f.append(face_quad)
        self.m_mesh_f_unique.append(face_quad_unique)
        self.m_mesh_cell_type.append(cell_quad_types)#every four elements in subarray 0 for empty cell, 1 for virtual cell, 2 for real cell
        self.m_mesh_cell_id.append(cell_quad)#every element in subarray represent a node id
       
    
    def print_nodes(self,node,file):
        if node is None:
            return
        file.write(f"{node.m_depth},{node.m_node_type},{node.m_vertex_idx}\n")
        for child in node.m_children:
            self.print_nodes(child, file)
            
    

    def visualize_octree_cells(self):
        print("m_leaf_nodes.size():",len(self.m_leaf_nodes))
        common_points=[]
        common_cells=[]
        common_cell_offset=0
        empty_points=[]
        empty_cells=[]
        empty_cell_offset=0
        cell_pattern=g_child_2_vtk_order
        leaf_to_empty={}
        leaf_to_empty_inverse={}
        leaf_to_inter={}
        leaf_to_inter_inverse={}

        empty_cell_m_type=[]
        common_cell_m_type=[]
        node_cell_types=[]
        flags=[]
        for i,node in enumerate(self.m_leaf_nodes):
            node_cell_types.append(node.m_node_cell_type)
            node_corner_id=g_child_order * (2**(self.m_max_depth + self.m_sampling_depth - node.m_depth)) + node.m_min_corner_id
            node_corners = node_corner_id * self.m_unit_edge_length + self.m_min_corner
            if(node.m_node_type=="empty"):
                leaf_to_empty[i]=len(empty_cells)
                leaf_to_empty_inverse[len(empty_cells)]=i
                
                empty_points.extend(node_corners.tolist())
                empty_cells.append(cell_pattern+empty_cell_offset)
                empty_cell_offset += 8
                empty_cell_m_type.append(self.results[i].m_type)     
            else:
                if node.m_node_cell_type==1:
                    flags.append(0)
                else:
                    flags.append(1)
                leaf_to_inter[i]=len(common_cells)
                leaf_to_inter_inverse[len(common_cells)]=i
                common_points.extend(node_corners.tolist())

                common_cells.append(cell_pattern+common_cell_offset)
                common_cell_offset += 8
                common_cell_m_type.append(self.results[i].m_type) 

        data={}
        data['common_point_mat']=np.array(common_points)
        data['common_cell_mat']=np.array(common_cells)

        data['empty_point_mat']=np.array(empty_points)
        data['empty_cell_mat']=np.array(empty_cells)

        data['leaf_to_empty']=leaf_to_empty
        data['leaf_to_empty_inverse']=leaf_to_empty_inverse
        data['leaf_to_inter']=leaf_to_inter
        data['leaf_to_inter_inverse']=leaf_to_inter_inverse
 
        data['empty_cell_m_type']=np.array(empty_cell_m_type)
        data['common_cell_m_type']=np.array(common_cell_m_type)
        data['flags']=np.array(flags)
        data['node_cell_types']=np.array(node_cell_types)

        return data

    
    def backup_cavity_data(self):
        self.origin_cavity_flags=copy.deepcopy(self.setup_cavity_flags())
        self.origin_cavity_leaf_node_indexes=copy.deepcopy(self.cavity_leaf_node_indexes)
        self.origin_cavity_vertex_indexes=copy.deepcopy(self.cavity_vertex_indexes)
        self.origin_cavity_face_ids=copy.deepcopy(self.cavity_face_ids)
        self.origin_face2cavity=copy.deepcopy(self.face2cavity)
        self.origin_edge2cavity=copy.deepcopy(self.edge2cavity)
        self.origin_vertex2cavity=copy.deepcopy(self.vertex2cavity)
        # with open('cavity_face_ids.txt', 'w') as f:
        #     for i,cavity_face_id in enumerate(self.origin_cavity_face_ids):
        #         print(f"{i}:{cavity_face_id}",file=f)
        # with open('face2cavity.txt', 'w') as f:
        #     for face_id, cavity_ids in self.origin_face2cavity.items():
        #         print(f"{face_id}:{cavity_ids}",file=f)
    def initial_topology_data(self):
        self.deleted_face_flag=np.zeros(len(self.m_mesh_f),dtype=bool)
        self.static_face_flag=np.ones(len(self.m_mesh_f),dtype=bool)
        self.face_adjacency = {}
        for fid in self.face2cavity:
            self.static_face_flag[fid]=False
        for fid in range(len(self.m_mesh_f)):
            self.face_adjacency[fid]={}
        self.cavity_adjacenty={}
        for cavity_id in range(len(self.cavity_face_ids)):
            self.cavity_adjacenty[cavity_id]={}

    def make_cavity_adjacency_dict(self):
        for face_id in self.face2cavity.keys():
            face_cavity_ids=self.face2cavity[face_id]
            for cur_cavity_id in face_cavity_ids:
                adj_cavity_ids = [adj_cavity_id for adj_cavity_id in face_cavity_ids if adj_cavity_id != cur_cavity_id]
                if len(adj_cavity_ids)>0:
                    self.cavity_adjacenty[cur_cavity_id][face_id]=adj_cavity_ids[0]
                else:
                    self.cavity_adjacenty[cur_cavity_id][face_id]=-1

    
    def make_vertex_edge_face_topology(self):
        self.face_to_edge={}
        self.edge_to_faces = {}
        self.vertex_to_edge={}
        for face_id, face in enumerate(self.m_mesh_f):
            if self.deleted_face_flag is not None and self.deleted_face_flag[face_id]:
                continue
            edges = extract_edges(face)
            self.face_to_edge[face_id]=edges
            for edge_idx,edge in enumerate(edges):
                if edge not in self.edge_to_faces:
                    self.edge_to_faces[edge] = []
                if face_id not in self.edge_to_faces[edge]:
                    self.edge_to_faces[edge].append(face_id)
                id0,id1=edge
                if id0 not in self.vertex_to_edge:
                    self.vertex_to_edge[id0]=[]
                if edge not in self.vertex_to_edge[id0]:
                    self.vertex_to_edge[id0].append(edge)
                if id1 not in self.vertex_to_edge:
                    self.vertex_to_edge[id1]=[]
                if edge not in self.vertex_to_edge[id1]:
                    self.vertex_to_edge[id1].append(edge)

                adj_faces = [f for f in self.edge_to_faces[edge] if f != face_id]
                self.face_adjacency[face_id][edge_idx] = adj_faces 
        for edge in self.alone_edges:
            id0,id1=edge
            if id0 not in self.vertex_to_edge:
                self.vertex_to_edge[id0]=[]
            if edge not in self.vertex_to_edge[id0]:
                self.vertex_to_edge[id0].append(edge)
            if id1 not in self.vertex_to_edge:
                self.vertex_to_edge[id1]=[]
            if edge not in self.vertex_to_edge[id1]:
                self.vertex_to_edge[id1].append(edge)


    def update_cavity_data(self):
        for cavity_id,cavity_face_id in enumerate(self.cavity_face_ids):
            cavity_face_id_origin= copy.deepcopy(cavity_face_id)
            cavity_break=False
            for fid in cavity_face_id_origin:
                if self.deleted_face_flag[fid]==1:
                    cavity_break=True
                    cavity_face_id.remove(fid)
                    if fid in self.face2cavity and cavity_id in self.face2cavity[fid]:
                        self.face2cavity[fid].remove(cavity_id)
            if cavity_break:
                cavity_vertex_indexes=self.origin_cavity_vertex_indexes[cavity_id]
                edge_indexes = [(0, 1), (0, 2), (2, 3), (3, 1),
                    (4, 5), (4, 6), (6, 7), (7, 5),
                    (0, 4), (1, 5), (2, 6), (3, 7)]
                for edge_index in edge_indexes:
                    i0,i1=edge_index
                    edge=tuple(sorted([cavity_vertex_indexes[i0],cavity_vertex_indexes[i1]]))
                    if cavity_id in self.edge2cavity[edge]:
                        self.edge2cavity[edge].remove(cavity_id)

    def filter_non_cavity_isolate_patch(self):
        self.edge_to_faces,self.vertex_to_edge=make_dict_edge2face(self.m_mesh_f,self.deleted_face_flag)
        while True:
            count=0
            for face_id in range(len(self.m_mesh_f)):
                if face_id==1508 or face_id==1500:
                    print(face_id)
                if self.deleted_face_flag[face_id]==1:
                    continue
                if self.origin_cavity_flags[face_id]==1:
                    continue
                # if fid in self.face2cavity and len(self.face2cavity[fid])>0:
                #     continue
                face=self.m_mesh_f[face_id]
                boundary_count=0
                non_manifold_count=0
                bd_edges=[]
                # skip=True
                for i in range(len(face)):
                    edge=tuple(sorted([face[i],face[(i+1)%len(face)]]))
                    if len(self.edge_to_faces[edge])==1:
                        boundary_count+=1
                        bd_edges.append(edge)
                    elif len(self.edge_to_faces[edge])>=3:
                        non_manifold_count+=1
                        
                non_manifold_boundary_count=0
                for i in range(len(face)):
                    edge=tuple(sorted([face[i],face[(i+1)%len(face)]]))
                    if len(self.edge_to_faces[edge])>=3:
                        is_non_manifold_boundary=True
                        for relate_fid in self.edge_to_faces[edge]:
                            if relate_fid!=face_id and  self.origin_cavity_flags[relate_fid]==0:
                                is_non_manifold_boundary=False
                                break
                        if is_non_manifold_boundary:
                            non_manifold_boundary_count+=1
                before_eular=self.eular_characteristic()
                if boundary_count==4:
                    count+=self.try_delete_face(face_id,before_eular)
                elif boundary_count==3 and non_manifold_count==1:
                    count+=self.try_delete_face(face_id,before_eular)
                elif boundary_count==2 and non_manifold_count==2:
                    if non_manifold_boundary_count==2:
                        continue
                    count+=self.try_delete_face(face_id,before_eular)
                elif boundary_count==2 and non_manifold_count==1:
                    if non_manifold_boundary_count==1:
                        continue
                    count+=self.try_delete_face(face_id,before_eular)
                elif boundary_count==1 and non_manifold_count==3:
                    if non_manifold_boundary_count>=2:
                        continue

                    count+=self.try_delete_face(face_id,before_eular)
                elif boundary_count==1 and non_manifold_count==2:
                    bd_edge=bd_edges[0]
                    i0,i1=bd_edge
                    if self.m_v_type[i0]!=2 and self.m_v_type[i1]!=2:
                        continue
                    if non_manifold_boundary_count==2:
                        continue
                    count+=self.try_delete_face(face_id,before_eular)
            if count==0:
                break
    def update_face_edge_type_map(self,face_edge_type_map,face_id):
        if not self.deleted_face_flag[face_id]:
            return
        related_face_ids=set()
        for edge in self.face_to_edge[face_id]:
            related_face_ids.update(self.edge_to_faces[edge])
        for fid in related_face_ids:
            if fid==face_id:
                continue
            bd_edges_fid=[]
            non_manifold_edges_fid=[]
            for edge_fid in self.face_to_edge[fid]:
                if len(self.edge_to_faces[edge_fid])==1:
                    bd_edges_fid.append(edge_fid)
                elif len(self.edge_to_faces[edge_fid])>=3:
                    non_manifold_edges_fid.append(edge_fid)
            face_edge_type_map[fid]=(bd_edges_fid,non_manifold_edges_fid)
    def check_is_feature_bd_edge(self,bd_edge):
        vid0,vid1=bd_edge
        leaf_node_index0=self.m_mesh_v_to_leaf_node_indexes[vid0]
        leaf_node_index1=self.m_mesh_v_to_leaf_node_indexes[vid1]
        if self.feature_flags[leaf_node_index0] and self.feature_flags[leaf_node_index1]:
            return True
        # elif len(self.convex_feature_points)==0 and self.m_leaf_nodes[leaf_node_index0].is_cylinder and self.m_leaf_nodes[leaf_node_index1].is_cylinder:
        elif self.m_leaf_nodes[leaf_node_index0].is_cylinder or self.m_leaf_nodes[leaf_node_index1].is_cylinder:
            return True
        return False
    def check_is_degenerate_bd_edge(self,bd_edge):
        vid0,vid1=bd_edge
        if self.quote_vertex_ids[vid0]!=-1:
            quote_vertex_id0=self.quote_vertex_ids[vid0]
        else:
            quote_vertex_id0=vid0
        if self.quote_vertex_ids[vid1]!=-1:
            quote_vertex_id1=self.quote_vertex_ids[vid1]
        else:
            quote_vertex_id1=vid1
        if quote_vertex_id0==quote_vertex_id1:
            return True
        return False
    def remove_mul_sheet_faces(self):
        cur_eular=self.eular_characteristic()
        m_v_type=np.array(self.m_v_type)
        while True:
            count=0
            for face_id in range(len(self.m_mesh_f)):
                if self.static_face_flag[face_id] and self.origin_cavity_flags[face_id]:
                    continue
                if self.deleted_face_flag[face_id]==1:
                    continue
                if not self.origin_cavity_flags[face_id]:
                    continue
                face=self.m_mesh_f[face_id]
                face_types=m_v_type[face]
                if face_types[0]==2 and face_types[2]==2 and face_types[1]!=2 and face_types[3]!=2:
                    if self.try_delete_face(face_id,cur_eular):
                        count+=1
                        self.deleted_face_map[face_id]=len(self.deleted_face_map)
                elif face_types[0]!=2 and face_types[2]!=2 and face_types[1]==2 and face_types[3]==2:
                    if self.try_delete_face(face_id,cur_eular):
                        count+=1
                        self.deleted_face_map[face_id]=len(self.deleted_face_map)
            if count==0:
                break
    def filter_isolate_non_cavity_patch_preprocess(self):
        filter_isolate_patch_count=0
        cur_eular=self.eular_characteristic()
        for face_id in range(len(self.m_mesh_f)):
            if self.static_face_flag[face_id] and self.origin_cavity_flags[face_id]:
                continue
            if self.deleted_face_flag[face_id]==1:
                continue
            if self.origin_cavity_flags[face_id]:
                continue
            bd_edges=[]
            non_manifold_edges=[]
            for edge in self.face_to_edge[face_id]:
                if len(self.edge_to_faces[edge])==1:
                    bd_edges.append(edge)
                elif len(self.edge_to_faces[edge])>=3:
                    non_manifold_edges.append(edge)

            boundary_count=len(bd_edges)
            non_manifold_count=len(non_manifold_edges)

            if boundary_count==4:
                if self.try_delete_face(face_id,cur_eular):
                    filter_isolate_patch_count+=1
                    self.deleted_face_map[face_id]=len(self.deleted_face_map)
            elif boundary_count==3 and non_manifold_count==1:
                if self.try_delete_face(face_id,cur_eular):
                    filter_isolate_patch_count+=1
                    self.deleted_face_map[face_id]=len(self.deleted_face_map)
            elif boundary_count==2 and non_manifold_count==2:
                if self.check_is_feature_bd_edge(bd_edges[0]) and self.check_is_feature_bd_edge(bd_edges[1]):
                    continue
                if self.try_delete_face(face_id,cur_eular):
                    filter_isolate_patch_count+=1
                    self.deleted_face_map[face_id]=len(self.deleted_face_map)
            elif boundary_count==2 and non_manifold_count==1:
                if self.check_is_feature_bd_edge(bd_edges[0]) and self.check_is_feature_bd_edge(bd_edges[1]):
                    continue
                if self.try_delete_face(face_id,cur_eular):
                    filter_isolate_patch_count+=1
                    self.deleted_face_map[face_id]=len(self.deleted_face_map)
            elif boundary_count==1 and non_manifold_count==3:
                if self.try_delete_face(face_id,cur_eular):
                    filter_isolate_patch_count+=1
                    self.deleted_face_map[face_id]=len(self.deleted_face_map)

            elif boundary_count==1 and non_manifold_count==1:
                bd_edge=bd_edges[0]
                if self.check_is_feature_bd_edge(bd_edge):
                    continue
                non_manifold_edge=non_manifold_edges[0]
                vid0,vid1=non_manifold_edge
                if self.m_v_type[vid0]!=2 and self.m_v_type[vid1]!=2 and len(self.edge_to_faces[non_manifold_edge])==3:
                    continue
                if self.try_delete_face(face_id,cur_eular):
                    filter_isolate_patch_count+=1
                    self.deleted_face_map[face_id]=len(self.deleted_face_map)
        return filter_isolate_patch_count
        
    def filter_isolate_patch_one_batch(self,face_edge_type_map):
        filter_isolate_patch_count=0
        cur_eular=self.eular_characteristic()
        for face_id in range(len(self.m_mesh_f)):
            if self.static_face_flag[face_id] and self.origin_cavity_flags[face_id]:
                continue
            if self.deleted_face_flag[face_id]==1:
                continue

            bd_edges,non_manifold_edges=face_edge_type_map[face_id]
            boundary_count=len(bd_edges)
            non_manifold_count=len(non_manifold_edges)

            if boundary_count==4:
                if self.try_delete_face(face_id,cur_eular):
                    filter_isolate_patch_count+=1
                    self.deleted_face_map[face_id]=len(self.deleted_face_map)
                self.update_face_edge_type_map(face_edge_type_map,face_id)
                #update eular
                cur_eular=self.eular_characteristic()
            elif boundary_count==3 and non_manifold_count==1:
                if self.try_delete_face(face_id,cur_eular):
                    filter_isolate_patch_count+=1
                    self.deleted_face_map[face_id]=len(self.deleted_face_map)
                self.update_face_edge_type_map(face_edge_type_map,face_id)
            elif boundary_count==3 and non_manifold_count==0:
                if self.check_is_feature_bd_edge(bd_edges[0]) or self.check_is_feature_bd_edge(bd_edges[1]) or self.check_is_feature_bd_edge(bd_edges[2]):
                    continue
                if self.try_delete_face(face_id,cur_eular):
                    filter_isolate_patch_count+=1
                    self.deleted_face_map[face_id]=len(self.deleted_face_map)
                self.update_face_edge_type_map(face_edge_type_map,face_id)
            elif boundary_count==2 and non_manifold_count==2:
                if self.check_is_feature_bd_edge(bd_edges[0]) or self.check_is_feature_bd_edge(bd_edges[1]):
                    continue
                if self.try_delete_face(face_id,cur_eular):
                    filter_isolate_patch_count+=1
                    self.deleted_face_map[face_id]=len(self.deleted_face_map)
                self.update_face_edge_type_map(face_edge_type_map,face_id)
            elif boundary_count==2 and non_manifold_count==1:
                if self.check_is_feature_bd_edge(bd_edges[0]) or self.check_is_feature_bd_edge(bd_edges[1]):
                    continue
                if self.try_delete_face(face_id,cur_eular):
                    filter_isolate_patch_count+=1
                    self.deleted_face_map[face_id]=len(self.deleted_face_map)
                self.update_face_edge_type_map(face_edge_type_map,face_id)
            elif boundary_count==1 and non_manifold_count==3:
                bd_edge=bd_edges[0]
                if self.try_delete_face(face_id,cur_eular):
                    filter_isolate_patch_count+=1
                    self.deleted_face_map[face_id]=len(self.deleted_face_map)
                self.update_face_edge_type_map(face_edge_type_map,face_id)
            elif boundary_count==1 and non_manifold_count==2:
                bd_edge=bd_edges[0]
                if self.check_is_feature_bd_edge(bd_edge):
                    continue
                non_manifold_edge=non_manifold_edges[0]
                vid0,vid1=non_manifold_edge
                if self.m_v_type[vid0]!=2 and self.m_v_type[vid1]!=2 and len(self.edge_to_faces[non_manifold_edge])==3:
                    continue
                non_manifold_edge=non_manifold_edges[1]
                vid0,vid1=non_manifold_edge
                if self.m_v_type[vid0]!=2 and self.m_v_type[vid1]!=2 and len(self.edge_to_faces[non_manifold_edge])==3:
                    continue
                if self.try_delete_face(face_id,cur_eular):
                    filter_isolate_patch_count+=1
                    self.deleted_face_map[face_id]=len(self.deleted_face_map)
                self.update_face_edge_type_map(face_edge_type_map,face_id)
            elif boundary_count==1 and non_manifold_count==1:
                bd_edge=bd_edges[0]

                if self.m_mesh_f_area2[face_id]!=0 and self.check_is_feature_bd_edge(bd_edge):
                    continue
                non_manifold_edge=non_manifold_edges[0]
                vid0,vid1=non_manifold_edge
                if self.m_v_type[vid0]!=2 and self.m_v_type[vid1]!=2 and len(self.edge_to_faces[non_manifold_edge])==3:
                    continue
                if self.try_delete_face(face_id,cur_eular):
                    filter_isolate_patch_count+=1
                    self.deleted_face_map[face_id]=len(self.deleted_face_map)
                self.update_face_edge_type_map(face_edge_type_map,face_id)
        return filter_isolate_patch_count
    

    def post_process(self):
        has_feature=len(self.convex_feature_points)>0
        if has_feature:
            self.edge_to_faces,self.vertex_to_edge=make_dict_edge2face(self.m_mesh_f,self.deleted_face_flag)
            face_edge_type_map={}
            for face_id in range(len(self.m_mesh_f)):
                if self.static_face_flag[face_id] and self.origin_cavity_flags[face_id]:
                    continue
                if self.deleted_face_flag[face_id]==1:
                    continue
                bd_edges=[]
                non_manifold_edges=[]
                for edge in self.face_to_edge[face_id]:
                    if len(self.edge_to_faces[edge])==1:
                        bd_edges.append(edge)
                    elif len(self.edge_to_faces[edge])>=3:
                        non_manifold_edges.append(edge)
                face_edge_type_map[face_id]=(bd_edges,non_manifold_edges)
            while(True):
                count=self.filter_isolate_patch_one_batch(face_edge_type_map)
                if count==0:
                    break
            cur_eular=self.eular_characteristic()
            
            for face_id in range(len(self.m_mesh_f)):
                if self.static_face_flag[face_id] and self.origin_cavity_flags[face_id]:
                    continue
                if self.deleted_face_flag[face_id]==1:
                    continue

                bd_edges,non_manifold_edges=face_edge_type_map[face_id]
                boundary_count=len(bd_edges)
                #non_manifold_count=len(non_manifold_edges)
                if self.m_mesh_f_area2[face_id]==0 and boundary_count>0:
                    if boundary_count==1:
                        bd_edge=bd_edges[0]
                        vid0,vid1=bd_edge
                        if self.m_v_type[vid0]==0 or self.m_v_type[vid1]==0:
                            continue
                        leaf_node_index0=self.m_mesh_v_to_leaf_node_indexes[vid0]
                        leaf_node_index1=self.m_mesh_v_to_leaf_node_indexes[vid1]
                        if self.m_leaf_nodes[leaf_node_index0].is_cylinder or self.m_leaf_nodes[leaf_node_index1].is_cylinder:
                            continue
                    if self.try_delete_face(face_id,cur_eular):
                        self.deleted_face_map[face_id]=len(self.deleted_face_map)
                    self.update_face_edge_type_map(face_edge_type_map,face_id)
                    continue
            self.make_vertex_edge_face_topology()
            self.generate_alone_edges_for_cad()
        
        #========
        mask=self.deleted_face_flag.astype(bool)
        inverted_mask = ~mask
        all_fids=np.arange(len(self.m_mesh_f))
        origin_filtered_mesh_f_ids=all_fids[inverted_mask]
        filtered_mesh_f_ids=all_fids[inverted_mask]
        alone_edges_list=list(self.alone_edges)
        edge_flags=np.ones(len(alone_edges_list),dtype=bool)
        local_face_marks=np.ones(len(origin_filtered_mesh_f_ids),dtype=bool)
        
        self.third_deleted_face_flag=copy.deepcopy(self.deleted_face_flag)
        _, _, local_face_marks, edge_flags=remove_redundant_patch(self.m_mesh_f[origin_filtered_mesh_f_ids],alone_edges_list)
        filtered_mesh_f_ids=origin_filtered_mesh_f_ids[local_face_marks]
        new_deleted_face_ids=np.setdiff1d(origin_filtered_mesh_f_ids , filtered_mesh_f_ids)
        self.deleted_face_flag[new_deleted_face_ids]=True
        self.deleted_alone_edge_flag=~edge_flags

        
        self.filtered_mesh_f=np.array(self.m_mesh_f[filtered_mesh_f_ids])
        self.filtered_mesh_f_ids=filtered_mesh_f_ids#for debug
        #================================
        #output filtered triangles
        filtered_mesh_tris=[]
        for i,fid in enumerate(filtered_mesh_f_ids):
            f_tri_2k=self.m_mesh_f_tri[fid*2]
            f_tri_2k_1=self.m_mesh_f_tri[fid*2+1]
            filtered_mesh_tris.append(f_tri_2k)
            filtered_mesh_tris.append(f_tri_2k_1)
        
        self.filtered_mesh_f_tri=np.array(filtered_mesh_tris)


    def try_delete_face(self,fid,before_eular):
        self.deleted_face_flag[fid]=1
        self.update_relate_face_edge_vertex_table(fid)
        self.update_cavity_tables(fid)
        after_eular=self.eular_characteristic()
        if before_eular==after_eular:
            return 1
        else:
            self.deleted_face_flag[fid]=0
            self.update_relate_face_edge_vertex_table(fid)
            self.update_cavity_tables(fid)
            after_eular=self.eular_characteristic()
            assert(before_eular==after_eular)
            return 0     
               
    def triangulate_faces_by_udf(self):
        """
        Triangulate quad faces.
        Results:
            self.m_mesh_f_tri
            self.m_mesh_f_area2
        """
        m_mesh_f_udf_max = self.m_mesh_f_udfs

        mesh_tri_f = np.zeros((len(self.m_mesh_f) * 2, 3), dtype=int)
        mesh_f_area = np.zeros(len(self.m_mesh_f), dtype=float)

        new_face_id = 0
        m_v_type = np.asarray(self.m_v_type, dtype=int)

        for fid, face in enumerate(self.m_mesh_f):
            face = np.asarray(face, dtype=int)
            face_v = self.m_mesh_v[face]
            v_types = m_v_type[face]

            if v_types[0] == EDGE and v_types[1] == PLANE and v_types[2] == EDGE and v_types[3] == PLANE:
                mesh_tri_f[new_face_id] = np.array([face[0], face[2], face[3]])
                mesh_tri_f[new_face_id + 1] = np.array([face[0], face[2], face[1]])

                double_area0 = doublearea_single(face_v[0], face_v[1], face_v[2])
                double_area1 = doublearea_single(face_v[2], face_v[3], face_v[0])
                mesh_f_area[fid] = abs(double_area0) + abs(double_area1)

                new_face_id += 2
                continue

            elif v_types[0] == PLANE and v_types[1] == EDGE and v_types[2] == PLANE and v_types[3] == EDGE:
                mesh_tri_f[new_face_id] = np.array([face[0], face[1], face[3]])
                mesh_tri_f[new_face_id + 1] = np.array([face[2], face[3], face[1]])

                double_area0 = doublearea_single(face_v[0], face_v[1], face_v[3])
                double_area1 = doublearea_single(face_v[1], face_v[2], face_v[3])
                mesh_f_area[fid] = abs(double_area0) + abs(double_area1)

                new_face_id += 2
                continue

            udf01 = m_mesh_f_udf_max[fid, 0]
            udf23 = m_mesh_f_udf_max[fid, 1]

            if udf01 < udf23:
                mesh_tri_f[new_face_id] = np.array([face[0], face[1], face[2]])
                mesh_tri_f[new_face_id + 1] = np.array([face[2], face[3], face[0]])

                double_area0 = doublearea_single(face_v[0], face_v[1], face_v[2])
                double_area1 = doublearea_single(face_v[2], face_v[3], face_v[0])
                mesh_f_area[fid] = abs(double_area0) + abs(double_area1)

                new_face_id += 2
            else:
                mesh_tri_f[new_face_id] = np.array([face[0], face[1], face[3]])
                mesh_tri_f[new_face_id + 1] = np.array([face[1], face[2], face[3]])

                double_area0 = doublearea_single(face_v[0], face_v[1], face_v[3])
                double_area1 = doublearea_single(face_v[1], face_v[2], face_v[3])
                mesh_f_area[fid] = abs(double_area0) + abs(double_area1)

                new_face_id += 2

        self.m_mesh_f_tri = mesh_tri_f[:new_face_id]
        self.m_mesh_f_area2 = mesh_f_area

        return self.m_mesh_f_tri, self.m_mesh_f_area2
    def solve_and_locate_vertices(
        self,
        spoints,
        sp_convex_feature_mask,
        cloud_sdf_grad_mask_func,
        centroids_coords,
        singular_value_threshold,
    ):
        self.scan_occupied_by_index()
        self.build_leaf_nodes_adjacency()
        self.batch_pre_solve_sqef(
            spoints,
            sp_convex_feature_mask,
            cloud_sdf_grad_mask_func,
            centroids_coords,
            singular_value_threshold,
        )
        self.batch_locate()
        self.dfs_relate()

    def build_mesh_and_topology(self):
        self.generate_mesh()

        self.scan_cavity()
        self.backup_cavity_data()

        self.initial_topology_data()
        self.make_vertex_edge_face_topology()
        self.make_cavity_adjacency_dict()

        self.generate_alone_edges()
    def compute_face_udf_and_triangulate(self, projFunc):
        self.m_mesh_f_udfs = batch_compute_face_udf_parallel(
            projFunc,
            self.m_mesh_v.astype(np.float32),
            self.m_mesh_f,
            self.m_mesh_f_unique,
            self.box_edge_length,
            self.norm_eps,
            self.position_eps,
            0.5,
        )
        self.triangulate_faces_by_udf()

    def clean_and_post_process(self):
        self.first_deleted_face_flag = copy.deepcopy(self.deleted_face_flag)

        self.update_cavity_data()
        self.make_vertex_edge_face_topology()
        self.make_cavity_adjacency_dict()

        self.clean_cavity_face()

        self.second_deleted_face_flag = copy.deepcopy(self.deleted_face_flag)

        self.update_cavity_data()
        self.make_vertex_edge_face_topology()

        self.post_process()
    def save_medial_mesh(self,savepath):
        save_obj(savepath, self.m_mesh_v,self.alone_edges, self.filtered_mesh_f_tri)
    def save_ma_mesh(self,savepath):
        alone_edges_list = list(self.alone_edges)
        alone_edge_flags = ~self.deleted_alone_edge_flag
        filtered_mesh_tris = self.filtered_mesh_f_tri
        all_mat_edges = set()

        for i in range(len(alone_edges_list)):
            if alone_edge_flags[i]:
                all_mat_edges.add(alone_edges_list[i])

        for tri in filtered_mesh_tris:
            edge01 = tuple(sorted([tri[0], tri[1]]))
            edge02 = tuple(sorted([tri[0], tri[2]]))
            edge12 = tuple(sorted([tri[1], tri[2]]))
            all_mat_edges.add(edge01)
            all_mat_edges.add(edge02)
            all_mat_edges.add(edge12)
        
        save_mat_obj(savepath,self.m_mesh_v,self.m_mesh_v_radii,list(all_mat_edges),self.filtered_mesh_f_tri)
        
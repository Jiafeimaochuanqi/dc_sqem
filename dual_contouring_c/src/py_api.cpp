#include "cloud_rvd.h"
#include "NanoKNNSearch.h"
#include "SegmentOcclusionCheck.h"
#include "cylinder_util.hpp"
#include "tangent.hpp"
#include "sqef.h"
#include "octree.h"
#include "octree_node.h"
#include "plane_grouping/dbscan.hpp"
#include "plane_grouping/plane_grouping.hpp"
#include "MulPriorityQueue.hpp"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>

#include <memory>
#include <string>

namespace py = pybind11;

template class PointCloudRVD<float>;
template class NanoKNNSearch<float>;


using RVDf = PointCloudRVD<float>;
using KNNSearchf = NanoKNNSearch<float>;


static py::list leaf_nodes_as_list(Octree& octree)
{
    py::list out;

    for (OctreeNode* node : octree.get_leaf_nodes()) {
        out.append(py::cast(node, py::return_value_policy::reference));
    }

    return out;
}

static py::list drop_nodes_as_list(Octree& octree)
{
    py::list out;

    for (OctreeNode* node : octree.get_drop_nodes()) {
        out.append(py::cast(node, py::return_value_policy::reference));
    }

    return out;
}

void bind_octree(py::module_& m)
{
    py::enum_<DualPointType>(m, "DualPointType")
    .value("CORNER", DualPointType::CORNER)
    .value("EDGE", DualPointType::EDGE)
    .value("PLANE", DualPointType::PLANE)
    .value("NONETYPE", DualPointType::NONETYPE)
    .export_values();

    py::class_<DualPoint>(m, "DualPoint")
        .def(py::init<>())

        .def_readwrite("m_point", &DualPoint::m_point)
        .def_readwrite("m_type", &DualPoint::m_type)
        .def_readwrite("m_is_inside", &DualPoint::m_is_inside)
        .def_readwrite("radius", &DualPoint::radius);
    py::class_<OctreeNode>(m, "OctreeNode")
        .def(py::init<>())

        .def_readwrite("id", &OctreeNode::id)
        .def_readwrite("leaf_node_index", &OctreeNode::leaf_node_index)

        .def_readwrite("inside_surface_point_ids", &OctreeNode::inside_surface_point_ids)
        .def_readwrite("inside_convex_feature_ids", &OctreeNode::inside_convex_feature_ids)
        .def_readwrite("strict_inside_convex_feature_ids", &OctreeNode::strict_inside_convex_feature_ids)

        .def_readwrite("m_depth", &OctreeNode::m_depth)
        .def_readwrite("m_min_corner_id", &OctreeNode::m_min_corner_id)
        .def_readwrite("m_max_corner_id", &OctreeNode::m_max_corner_id)
        .def_readwrite("m_centroid_id", &OctreeNode::m_centroid_id)
        .def_readwrite("m_sample_grids", &OctreeNode::m_sample_grids)
        .def_readwrite("m_edge_span", &OctreeNode::m_edge_span)

        .def_readwrite("m_vertex_idx", &OctreeNode::m_vertex_idx)
        .def_readwrite("quote_leaf_node_index", &OctreeNode::quote_leaf_node_index)
        .def_readwrite("quote_vertex_id", &OctreeNode::quote_vertex_id)
        .def_readwrite("be_quoted_leaf_node_indexes", &OctreeNode::be_quoted_leaf_node_indexes)
        .def_readwrite("be_quoted_vertex_ids", &OctreeNode::be_quoted_vertex_ids)

        .def_readwrite("cell_sample_mask", &OctreeNode::cell_sample_mask)
        .def_readwrite("groups", &OctreeNode::groups)

        .def_readwrite("m_node_type", &OctreeNode::m_node_type)
        .def_readwrite("m_node_cell_type", &OctreeNode::m_node_cell_type)
        .def_readwrite("m_reconstruct_point", &OctreeNode::m_reconstruct_point)

        .def_readwrite("is_cylinder", &OctreeNode::is_cylinder)
        .def_readwrite("drop_stage", &OctreeNode::drop_stage)
        .def_readwrite("decision_pid", &OctreeNode::decision_pid)
        .def_readwrite("decision_sid", &OctreeNode::decision_sid);

    py::class_<Octree>(m, "Octree")
        .def(
            py::init<int, Eigen::Vector3f, Eigen::Vector3f, int>(),
            py::arg("max_depth"),
            py::arg("min_corner") = Eigen::Vector3f::Constant(-1.0f),
            py::arg("max_corner") = Eigen::Vector3f::Constant(1.0f),
            py::arg("sampling_depth") = 1
        )

        // -------------------------------
        // Python create_octree_by_sqef call
        // -------------------------------
        .def("grow_by_max_depth", &Octree::grow_by_max_depth)
        .def("build_spoint_id_maps", &Octree::build_spoint_id_maps)
        .def("scan_occupied", &Octree::scan_occupied)
        .def("compute_node_strict_contain_feature_ids", &Octree::compute_node_strict_contain_feature_ids)
        .def("scan_occupied_by_index", &Octree::scan_occupied_by_index)

        .def(
            "run_leaf_node_mat_search",
            &Octree::run_leaf_node_mat_search,
            py::arg("spoints"),
            py::arg("sns"),
            py::arg("sp_feature_mask"),
            py::arg("patch_ids"),
            py::arg("sp_convex_feature_mask"),
            py::arg("dihedral_angle_threshold") = 135.0,
            py::arg("min_dihedral_angle_degree") = 0.0
        )

        // -------------------------------
        // extract_octree_by_sqef main flow functions
        // -------------------------------
        .def("build_leaf_nodes_adjacency", &Octree::build_leaf_nodes_adjacency)
        .def("get_centroids_of_leaf_nodes", &Octree::get_centroids_of_leaf_nodes)

        .def(
            "solve_and_locate_vertices",
            &Octree::solve_and_locate_vertices,
            py::arg("spoints"),
            py::arg("sp_convex_feature_mask"),
            py::arg("cloud_sdf_grad_mask_func"),
            py::arg("centroids_coords"),
            py::arg("singular_value_threshold")
        )

        .def("build_mesh_and_topology", &Octree::build_mesh_and_topology)

        .def(
            "compute_face_udf_and_triangulate",
            &Octree::compute_face_udf_and_triangulate,
            py::arg("projFunc")
        )

        .def("clean_and_post_process", &Octree::clean_and_post_process)
        .def("visualize_octree_cells", &Octree::visualize_octree_cells)
        .def("save_ma_mesh", &Octree::save_ma_mesh)
        .def("save_medial_mesh", &Octree::save_medial_mesh)
        // -------------------------------
        // Python debug / visualize_octree_cells data
        // -------------------------------
        .def_property_readonly(
            "m_root_node",
            [](Octree& self) -> OctreeNode& {
                return self.get_root_node();
            },
            py::return_value_policy::reference_internal
        )

        .def_property_readonly(
            "m_leaf_nodes",
            [](Octree& self) {
                return leaf_nodes_as_list(self);
            }
        )

        .def_property_readonly(
            "drop_nodes",
            [](Octree& self) {
                return drop_nodes_as_list(self);
            }
        )

        .def_property_readonly(
            "results",
            [](const Octree& self) -> const std::vector<DualPoint>& {
                return self.get_results();
            },
            py::return_value_policy::reference_internal
        )

        .def_property_readonly("m_max_depth", &Octree::get_max_depth)
        .def_property_readonly("m_sampling_depth", &Octree::get_sampling_depth)
        .def_property_readonly("m_unit_edge_length", &Octree::get_unit_edge_length)
        .def_property_readonly("box_edge_length", &Octree::get_box_edge_length)
        .def_property_readonly("m_min_corner", &Octree::get_min_corner)
        .def_property_readonly("m_max_corner", &Octree::get_max_corner)
        .def_property_readonly("m_capacity_per_dim", &Octree::get_capacity_per_dim)
        .def_property_readonly("resolution", &Octree::get_resolution)

        // -------------------------------
        // public member
        // -------------------------------
        .def_readwrite("m_mesh_v", &Octree::m_mesh_v)
        .def_readwrite("m_mesh_v_radii", &Octree::m_mesh_v_radii)
        .def_readwrite("m_mesh_v_to_leaf_node_indexes", &Octree::m_mesh_v_to_leaf_node_indexes)
        .def_readwrite("m_mesh_v_simulated", &Octree::m_mesh_v_simulated)

        .def_readwrite("m_v_type", &Octree::m_v_type)
        .def_readwrite("m_mesh_f", &Octree::m_mesh_f)
        .def_readwrite("m_mesh_f_unique", &Octree::m_mesh_f_unique)
        .def_readwrite("m_mesh_f_tri", &Octree::m_mesh_f_tri)
        .def_readwrite("m_mesh_f_area2", &Octree::m_mesh_f_area2)
        .def_readwrite("m_mesh_f_udfs", &Octree::m_mesh_f_udfs)

        .def_readwrite("filtered_mesh_f", &Octree::filtered_mesh_f)
        .def_readwrite("filtered_mesh_f_ids", &Octree::filtered_mesh_f_ids)
        .def_readwrite("filtered_mesh_f_tri", &Octree::filtered_mesh_f_tri)

        .def_readwrite("m_mesh_cell_id", &Octree::m_mesh_cell_id)
        .def_readwrite("m_mesh_cell_type", &Octree::m_mesh_cell_type)

        .def_readwrite("cylinder_node_ids", &Octree::cylinder_node_ids)
        .def_readwrite("alone_edges", &Octree::alone_edges)

        .def_readwrite("feature_flags", &Octree::feature_flags)

        .def_readwrite("cavity_face_ids", &Octree::cavity_face_ids)
        .def_readwrite("cavity_leaf_node_indexes", &Octree::cavity_leaf_node_indexes)
        .def_readwrite("cavity_vertex_indexes", &Octree::cavity_vertex_indexes)
        .def_readwrite("cavity_flags", &Octree::cavity_flags)

        .def_readwrite("origin_cavity_face_ids", &Octree::origin_cavity_face_ids)
        .def_readwrite("origin_cavity_leaf_node_indexes", &Octree::origin_cavity_leaf_node_indexes)
        .def_readwrite("origin_cavity_vertex_indexes", &Octree::origin_cavity_vertex_indexes)
        .def_readwrite("origin_cavity_flags", &Octree::origin_cavity_flags)

        .def_readwrite("deleted_face_flag", &Octree::deleted_face_flag)
        .def_readwrite("static_face_flag", &Octree::static_face_flag)
        .def_readwrite("first_deleted_face_flag", &Octree::first_deleted_face_flag)
        .def_readwrite("second_deleted_face_flag", &Octree::second_deleted_face_flag)
        .def_readwrite("third_deleted_face_flag", &Octree::third_deleted_face_flag)
        .def_readwrite("deleted_alone_edge_flag", &Octree::deleted_alone_edge_flag)

        .def_readwrite("convex_feature_points", &Octree::convex_feature_points)
        .def_readwrite("convex_corner_points", &Octree::convex_corner_points)
        .def_readwrite("position_eps", &Octree::position_eps)
        .def_readwrite("norm_eps", &Octree::norm_eps)

        .def_readwrite("angle_threshold0", &Octree::angle_threshold0)
        .def_readwrite("angle_threshold", &Octree::angle_threshold)

        .def_readwrite("deleted_face_map", &Octree::deleted_face_map);
}

PYBIND11_MODULE(GRAPH_GROUPING, m) {
    m.doc() = "RVD grouping library";
    bind_octree(m);
    m.def(
        "save_obj",
        &save_obj,
        py::arg("obj_filename"),
        py::arg("vertices"),
        py::arg("edges"),
        py::arg("triangles"),
        "Save vertices, edges and triangles to an OBJ file"
    );
    m.def(
        "save_mat_obj",
        &save_mat_obj,
        py::arg("filename"),
        py::arg("cpoint"),
        py::arg("radii"),
        py::arg("cedges"),
        py::arg("cfaces"),
        "Save medial axis vertices, radii, edges and faces to a MAT obj-like file"
    );
    m.def(
        "is_cylindrical_tinf_node",
        [](const MatF3& dirs,
        bool torch_feature,
        float eps,
        float min_proj_norm,
        float gap_threshold_deg)
        {
            CylinderDetectionResult r =
                is_cylindrical_tinf_node(
                    dirs,
                    torch_feature,
                    eps,
                    min_proj_norm,
                    gap_threshold_deg
                );

            py::dict d;
            d["is_cylindrical_tinf"] = r.is_cylindrical_tinf;
            d["reason"] = r.reason;
            d["axis"] = r.axis;
            d["evals"] = r.evals;
            d["max_gap"] = r.max_gap;
            d["num_valid_angle"] = r.num_valid_angle;
            d["num_raw_dirs"] = r.num_raw_dirs;
            d["num_used_dirs"] = r.num_used_dirs;
            d["used_filter"] = r.used_filter;
            d["min_proj_norm"] = r.min_proj_norm;
            d["gap_threshold_deg"] = r.gap_threshold_deg;
            return d;
        },
        py::arg("dirs"),
        py::arg("torch_feature") = false,
        py::arg("eps") = 1e-12f,
        py::arg("min_proj_norm") = 0.5f,
        py::arg("gap_threshold_deg") = 165.0f
    );
    // ============================================================
    // util function：cloud_grouping
    // ============================================================
    m.def("cloud_grouping", &cloud_grouping,
          py::arg("cloud"),
          py::arg("cloudNormals"),
          py::arg("cloudFeatureFlags"),
          py::arg("cellSeedNums"),
          py::arg("nearestCloudIdsFlat"),
          py::arg("cellNodeIds"),
          "Compute cloud grouping");

    // ============================================================
    // Segment occlusion check utilities
    // ============================================================
    m.def("batch_segment_occlusion_check",
          &batch_segment_occlusion_check,
          py::arg("centers"),
          py::arg("nearestPoints"),
          py::arg("selfNodeIds"),
          py::arg("occupiedCellOccluder"),
          py::arg("minCorner"),
          py::arg("cellSize"),
          py::arg("resolution"),
          R"pbdoc(
Batch segment occlusion check.

Parameters
----------
centers : (N,3) float32 row-major array
nearestPoints : (N,3) float32 row-major array
selfNodeIds : (N,) int64 array
occupiedCellOccluder : (resolution^3,) int64 array
    Python side should pass occupied_cell_occluder.reshape(-1)
minCorner : (3,) float32 array
cellSize : float
resolution : int

Returns
-------
occludedFlags : (N,) bool array
)pbdoc");

    m.def("batch_segment_occlusion_check_with_hit_ids",
          &batch_segment_occlusion_check_with_hit_ids,
          py::arg("centers"),
          py::arg("nearestPoints"),
          py::arg("selfNodeIds"),
          py::arg("occupiedCellOccluder"),
          py::arg("minCorner"),
          py::arg("cellSize"),
          py::arg("resolution"),
          R"pbdoc(
Batch segment occlusion check with hit node ids.

Parameters
----------
centers : (N,3) float32 row-major array
nearestPoints : (N,3) float32 row-major array
selfNodeIds : (N,) int64 array
occupiedCellOccluder : (resolution^3,) int64 array
    Python side should pass occupied_cell_occluder.reshape(-1)
minCorner : (3,) float32 array
cellSize : float
resolution : int

Returns
-------
(occludedFlags, hitNodeIds)
    occludedFlags : (N,) bool array
    hitNodeIds    : (N,) int64 array, -1 means no hit
)pbdoc");

    m.def("test_flattened_occupied_cell_order",
          &test_flattened_occupied_cell_order,
          py::arg("flatData"),
          py::arg("resolution"),
          R"pbdoc(
Debug utility for verifying the flatten order of a 3D occupied-cell array.

Python side should typically pass:
    occupied_cell_occluder.reshape(-1)

The function prints the received 1D order and the (x,y,z)->idx->value mapping
using idx = (x * resolution + y) * resolution + z
)pbdoc");
    

    py::class_<SQEF>(m, "SQEF")
    .def(py::init<>())

    .def(
        py::init<const MatF3&, const MatF3&, const MatF3&>(),
        py::arg("normals"),
        py::arg("shape_samples"),
        py::arg("cell_samples")
    )

    .def(
        "solve",
        &SQEF::solve,
        py::arg("svd_threshold") = 0.01,
        py::arg("debug") = false
    )

    .def(
        "solve_in_line",
        &SQEF::solve_in_line,
        py::arg("svd_threshold") = 0.01,
        py::arg("debug") = false
    )

    .def(
        "solve_in_plane",
        &SQEF::solve_in_plane,
        py::arg("debug") = false
    )

    .def(
        "solve_with_group",
        &SQEF::solve_with_group,
        py::arg("groups"),
        py::arg("is_cylinder"),
        py::arg("svd_threshold") = 0.01,
        py::arg("debug") = false
    )

    .def(
        "residual",
        &SQEF::residual,
        py::arg("v")
    )

    .def_readwrite("normals", &SQEF::normals)
    .def_readwrite("shape_samples", &SQEF::shape_samples)
    .def_readwrite("cell_samples", &SQEF::cell_samples)
    .def_readwrite("A", &SQEF::A)
    .def_readwrite("b", &SQEF::b)
    .def_readwrite("is_plane", &SQEF::is_plane)
    .def_readwrite("is_line", &SQEF::is_line)
    .def_readwrite("dir", &SQEF::dir)
    .def_readwrite("point", &SQEF::point);

    py::class_<MulPriorityQueue<int>>(m, "MulPriorityQueueInt")
        .def(py::init<>())

        .def(
            "push",
            &MulPriorityQueue<int>::push,
            py::arg("item"),
            py::arg("priority")
        )

        .def(
            "update_priority",
            &MulPriorityQueue<int>::update_priority,
            py::arg("item"),
            py::arg("new_priority")
        )

        .def(
            "pop",
            &MulPriorityQueue<int>::pop
        )

        .def(
            "contains",
            &MulPriorityQueue<int>::contains,
            py::arg("item")
        )

        .def(
            "is_empty",
            &MulPriorityQueue<int>::is_empty
        )

        .def(
            "size",
            &MulPriorityQueue<int>::size
        )

        .def(
            "traverse_via_map",
            &MulPriorityQueue<int>::traverse_via_map
        )

        .def(
            "items_sorted",
            &MulPriorityQueue<int>::items_sorted
        )

        .def(
            "__len__",
            &MulPriorityQueue<int>::size
        )

        .def(
            "__bool__",
            [](const MulPriorityQueue<int>& q) {
                return static_cast<bool>(q);
            }
        );
    
    py::class_<MulPriorityQueue<std::string>>(m, "MulPriorityQueueString")
        .def(py::init<>())

        .def(
            "push",
            &MulPriorityQueue<std::string>::push,
            py::arg("item"),
            py::arg("priority")
        )

        .def(
            "update_priority",
            &MulPriorityQueue<std::string>::update_priority,
            py::arg("item"),
            py::arg("new_priority")
        )

        .def(
            "pop",
            &MulPriorityQueue<std::string>::pop
        )

        .def(
            "contains",
            &MulPriorityQueue<std::string>::contains,
            py::arg("item")
        )

        .def(
            "is_empty",
            &MulPriorityQueue<std::string>::is_empty
        )

        .def(
            "size",
            &MulPriorityQueue<std::string>::size
        )

        .def(
            "traverse_via_map",
            &MulPriorityQueue<std::string>::traverse_via_map
        )

        .def(
            "items_sorted",
            &MulPriorityQueue<std::string>::items_sorted
        )

        .def(
            "__len__",
            &MulPriorityQueue<std::string>::size
        )

        .def(
            "__bool__",
            [](const MulPriorityQueue<std::string>& q) {
                return static_cast<bool>(q);
            }
        );

    py::class_<FastDBSCAN>(m, "DBSCAN")
        .def(
            py::init<double, int, const std::string&>(),
            py::arg("eps") = 0.5,
            py::arg("min_samples") = 5,
            py::arg("metric") = "euclidean"
        )

        .def(
            "fit_predict",
            &FastDBSCAN::fit_predict,
            py::arg("X")
        )

        .def(
            "fit",
            &FastDBSCAN::fit,
            py::arg("X")
        )

        .def_property_readonly(
            "labels_",
            [](const FastDBSCAN& self) {
                return self.labels();
            }
        )

        .def_property_readonly(
            "clusters_",
            [](const FastDBSCAN& self) {
                return self.clusters();
            }
        );

    py::class_<KNNSearchf>(m, "KNNSearch")
        .def(py::init([](const Eigen::Ref<const MatF>& data,
                         int searchThreads) {
                if (data.cols() <= 0) {
                    throw std::invalid_argument("KNNSearch: input data must have positive number of columns.");
                }
                return std::make_unique<KNNSearchf>(
                    data.data(), data.rows(), data.cols(), searchThreads
                );
            }),
            py::arg("data"),
            py::arg("searchThreads") = 4,
            "Construct KNNSearch from a row-major numpy/eigen matrix")

        .def_property_readonly("rows", &KNNSearchf::rows)
        .def_property_readonly("cols", &KNNSearchf::cols)
        .def_property_readonly("searchThreads", &KNNSearchf::searchThreads)

        .def("searchKNN",
             [](const KNNSearchf& self,
                const Eigen::Ref<const MatF>& pts,
                int K) {
                 std::vector<int> indices;
                 std::vector<float> distances;
                 self.searchKNN(indices, distances, pts.data(), pts.rows(), pts.cols(), K);
                 return py::make_tuple(indices, distances);
             },
             py::arg("pts"),
             py::arg("K") = 1,
             "KNN query, returns (indices, distances). Distances are squared Euclidean distances.")

        .def("searchKNN1",
             [](const KNNSearchf& self,
                const Eigen::Ref<const Eigen::Matrix<float, 1, Eigen::Dynamic, Eigen::RowMajor>>& pt) {
                 int index = -1;
                 float dist2 = 0.0f;
                 self.searchKNN1(index, dist2, pt.data());
                 return py::make_tuple(index, dist2);
             },
             py::arg("pt"),
             "Single-point 1-NN query, returns (index, squared_distance).")

        .def("radiusSearch",
             [](const KNNSearchf& self,
                const Eigen::Ref<const MatF>& pts,
                float radius) {
                 std::vector<std::vector<int>> indices;
                 std::vector<std::vector<float>> distances;
                 self.radiusSearch(indices, distances, pts.data(), pts.rows(), pts.cols(), radius);
                 return py::make_tuple(indices, distances);
             },
             py::arg("pts"),
             py::arg("radius"),
             "Batch radius query, returns (all_indices, all_distances). Distances are squared Euclidean distances.");


    py::class_<RVDf::Result>(m, "Result")
        .def(py::init<>())
        .def(py::init<std::vector<int64_t>, float>(),
             py::arg("labels"),
             py::arg("dist"))
        .def_readwrite("labels", &RVDf::Result::labels)
        .def_readwrite("dist",   &RVDf::Result::dist);


    py::class_<RVDf>(m, "PointCloudRVD")
        .def(py::init([](const Eigen::Ref<const MatF3>& points,
                         const Eigen::Ref<const MatF3>& normals,
                         const Eigen::Ref<const ArrayB1>& featureFlags) {
                if (points.rows() != normals.rows() || points.rows() != featureFlags.rows()) {
                    throw std::invalid_argument("PointCloudRVD: points / normals / featureFlags size mismatch.");
                }
                return std::make_unique<RVDf>(
                    MatF3(points),
                    MatF3(normals),
                    ArrayB1(featureFlags)
                );
            }),
            py::arg("points"),
            py::arg("normals"),
            py::arg("featureFlags"),
            "Construct PointCloudRVD from points, normals, and feature flags")

        .def("buildKnnGraph", &RVDf::buildKnnGraph,
             py::arg("knn"),
             py::arg("K"),
             py::arg("symmetric") = true,
             py::arg("maxSeparateNormalAngleDeg") = 90.0f,
             py::arg("maxConnectNormalAngleDeg") = 90.0f,
             py::arg("candidateMultiplier") = 3,
             "Build graph by KNN with normal constraint")


        .def("label",
             py::overload_cast<const Eigen::VectorXi&, const Eigen::VectorXi&, const VecI64&>(&RVDf::label),
             py::arg("cellSeedNums"),
             py::arg("nearestCloudIdsFlat"),
             py::arg("cellNodeIds"),
             "Multi-source label propagation from flattened cell seeds, using global node ids")

        .def("label_from_nested",
             py::overload_cast<const std::vector<std::vector<int>>&, const VecI64&>(&RVDf::label),
             py::arg("cellNearestCloudIds"),
             py::arg("cellNodeIds"),
             "Multi-source label propagation from nested Python lists, using global node ids")

        .def("computeProjectionClusters",
             py::overload_cast<const Eigen::VectorXi&, const Eigen::VectorXi&, const VecI64&>(&RVDf::computeProjectionClusters, py::const_),
             py::arg("cellSeedNums"),
             py::arg("nearestCloudIdsFlat"),
             py::arg("cellNodeIds"),
             "Compute connected seed clusters for each cell")
             
        .def("computeProjectionClustersNoLabel",
            &RVDf::computeProjectionClustersNoLabel,
            py::arg("cellSeedNums"),
            py::arg("nearestCloudIdsFlat"),
            "Compute connected seed clusters for each cell without using propagated labels")
        .def("computeProjectionClustersNoLabelOneRingBridge",
            &RVDf::computeProjectionClustersNoLabelOneRingBridge,
            py::arg("cellSeedNums"),
            py::arg("nearestCloudIdsFlat"),
            "Compute seed clusters for each cell without labels, allowing one-ring non-feature bridge points")
        
        .def("computeProjectionClustersNoLabelOneRingExpanded",
            &RVDf::computeProjectionClustersNoLabelOneRingExpanded,
            py::arg("cellSeedNums"),
            py::arg("nearestCloudIdsFlat"),
            "Compute seed clusters for each cell without labels, using connectivity in the one-ring expanded subgraph; non-seed bridge points are allowed only if they are not feature points")
        
        .def("computeProjectionClustersNoLabelOneRingExpandedInSamePatch",
            &RVDf::computeProjectionClustersNoLabelOneRingExpandedInSamePatch,
            py::arg("cellSeedNums"),
            py::arg("nearestCloudIdsFlat"),
            py::arg("patchIds"),
            "Compute seed clusters for each cell without labels, using connectivity in the one-ring expanded subgraph; non-seed bridge points are allowed only if they are not feature points and in same patch.")

        .def("computeCellRelatedPointClusters",
            &RVDf::computeCellRelatedPointClusters,
            py::arg("cellSeedNums"),
            py::arg("nearestCloudIdsFlat"),
            py::arg("cellNodeIds"),
            "Compute connected clusters of all globally related points for each cell")
    

        .def("computeCellBlockerNodeIds",
             &RVDf::computeCellBlockerNodeIds,
             py::arg("cellSeedNums"),
             py::arg("nearestCloudIdsFlat"),
             py::arg("cellNodeIds"),
             "Compute merged blocker node ids for each cell")

        .def("computeMutualBlockerPairs",
             &RVDf::computeMutualBlockerPairs,
             py::arg("cellSeedNums"),
             py::arg("nearestCloudIdsFlat"),
             py::arg("cellNodeIds"),
             "Return (cellBlockerNodeIds, mutualBlockerPairs)")

        .def("computeCellRelatedPointIds",
             &RVDf::computeCellRelatedPointIds,
             py::arg("cellSeedNums"),
             py::arg("cellNodeIds"),
             "Return point ids related to each cell label")

        .def("computeSmoothPatchIds",
             &RVDf::computeSmoothPatchIds,
             py::return_value_policy::reference_internal,
             "Get point patch id")

        .def("get_results",
             &RVDf::get_results,
             py::return_value_policy::reference_internal,
             "Get label propagation results")

        .def("extractUniqueUndirectedEdges",
             &RVDf::extractUniqueUndirectedEdges,
             "Extract unique undirected edges from the adjacency graph");


    m.def("compute_groups", &compute_groups,
            py::arg("sdfs"), 
            py::arg("sns"), 
            py::arg("coords"),
            py::arg("nearest_points"),
            py::arg("feature_flags"),
            py::arg("norm_eps"),
            py::arg("position_eps"),
            py::arg("distance_threshold"),
          "compute groups");

    m.def("batch_compute_face_udf_parallel", &batch_compute_face_udf_parallel,
            py::arg("projfunc"),
            py::arg("m_mesh_v"),
            py::arg("m_mesh_f"),
            py::arg("m_mesh_f_unique"),
            py::arg("box_edge_length"), 
            py::arg("octree_norm_eps"), 
            py::arg("octree_position_eps"),
            py::arg("ratio"),
          "batch compute distance to tangent plane");
    // ============================================
    // compute distance to tangent plane
    // ============================================
    m.def("compute_tangent_plane", &compute_tangent_plane,
            py::arg("spoints"),
            py::arg("sp_normals"),
            py::arg("sp_feature_mask"),
            py::arg("center"),
            py::arg("box_edge_length"), 
            py::arg("octree_norm_eps"), 
            py::arg("octree_position_eps"),
          "comput tangent plane");
    m.def("compute_tangent_plane_dis", &compute_tangent_plane_dis,
            py::arg("projfunc"),
            py::arg("center"),
            py::arg("box_edge_length"), 
            py::arg("octree_norm_eps"), 
            py::arg("octree_position_eps"),
            py::arg("ratio"),
          "compute distance to tangent plane");
    
    
    py::class_<SDFProjectorFunc>(m, "SDFProjectorFunc")
        .def(py::init<MatF3&,
                      MatF3&,
                      const ArrayB1&,
                      //const ArrayB1&,
                      float,
                      int
                      >(),
            py::arg("points"),
            py::arg("normals"),
            py::arg("feature_mask") = ArrayB1(),
            //py::arg("convex_feature_mask") = ArrayB1(),
            py::arg("ratio") = std::sin(85.0f * static_cast<float>(M_PI) / 180.0f),
            py::arg("search_threads") = 4,
            R"pbdoc(
Create an SDFProjectorFunc.

Parameters
----------
points : (N,3) float32 array
    Surface points.
normals : (N,3) float32 array
    Surface normals.
feature_mask : (N,) bool array, optional
    Feature-point mask. Empty means no feature mask.
// convex_feature_mask : (N,) bool array, optional
//     Convex-feature-point mask. Empty means no convex-feature filtering.
ratio : float, optional
    Clamp ratio used for non-feature normal projection.
search_threads : int, optional
    Number of threads used by NanoKNNSearch.
)pbdoc"
        )
        
        .def("__call__", &SDFProjectorFunc::operator(),
            py::arg("pts"),
            R"pbdoc(
Evaluate SDF projection on query points.

Parameters
----------
pts : (M,3) float32 array
    Query points.

Returns
-------
(sdfs, normals, nearest_points, indices, feature_flags)
    sdfs : (M,) float32
    normals : (M,3) float32
    nearest_points : (M,3) float32
    indices : (M,) int32
    feature_flags : (M,) bool
)pbdoc"
        );

    py::class_<Plane>(m, "Plane")
        .def(py::init<const Eigen::Vector3f&, const Eigen::Vector3f&>(),
            py::arg("normal"), py::arg("point"),
            R"pbdoc(
            construct plane by normal and points
            parameters:
                normal
                point
            )pbdoc")
        
        .def(py::init<const Eigen::Vector3f&, float>(),
            py::arg("normal"), py::arg("distance"),
            R"pbdoc(
            construct plane by normal and points
            )pbdoc")
        
        .def_readonly("normal", &Plane::normal,
            "normal of plane")
        
        .def_readonly("point", &Plane::point,
            "one point in plane")
        
        .def_readonly("distance", &Plane::distance,
            "equation n·x + d = 0")
        
        .def("to_string", &Plane::toString,
            R"pbdoc(
            equation string such as "0.71x + 0.71y + 0.00z + 1.00 = 0"
            )pbdoc")
        
        .def("distance_to_point", &Plane::distanceToPoint,
            py::arg("point"),
            R"pbdoc(
            point distance to plane
            )pbdoc")
        
        .def_property_readonly("equation", [](const Plane& plane) {
            return plane.toString();
        }, "equation string")
        
        .def("__repr__", [](const Plane& p) {
            std::stringstream ss;
            ss << "Plane(normal=[" 
               << p.normal[0] << ", " << p.normal[1] << ", " << p.normal[2] 
               << "], point=[" 
               << p.point[0] << ", " << p.point[1] << ", " << p.point[2] 
               << "], distance=" << p.distance << ")";
            return ss.str();
        })
        
        .def("__str__", &Plane::toString);
    

    py::class_<PlaneGrouping>(m, "PlaneGrouping")
        .def(py::init<double, double, double, int, bool, bool, double>(),
            py::arg("angle_threshold") = 5.0,
            py::arg("position_eps") = 0.1,
            py::arg("distance_threshold") = 0.01,
            py::arg("min_samples") = 3,
            py::arg("refine") = false,
            py::arg("group_normal_first") = true,
            py::arg("max_assignment_distance") = -1.0,
            R"pbdoc(
            constructor
            )pbdoc")
        
        .def("fit", &PlaneGrouping::fit,
            py::arg("points"), py::arg("normals"), 
            py::arg("need_merge") = true, py::arg("debug") = false,
            R"pbdoc(
            compute plane groups
            )pbdoc")
        
        
        .def("set_angle_threshold", &PlaneGrouping::setAngleThreshold,
            py::arg("angle_degrees"),
            "set degree")
        
        .def("set_distance_threshold", &PlaneGrouping::setDistanceThreshold,
            py::arg("threshold"),
            "set threshold")
        
        .def("set_min_samples", &PlaneGrouping::setMinSamples,
            py::arg("min_samples"),
            "set min samples")
        
        .def("set_position_eps", &PlaneGrouping::setPositionEps,
            py::arg("eps"),
            "set eps")
        
        .def_static("create_plane_from_points", [](const MatF3& points) -> Plane {
            if (points.rows() < 3) {
                throw std::runtime_error("need 3 or more points");
            }
            
            Eigen::Vector3f p1 = points.row(0);
            Eigen::Vector3f p2 = points.row(1);
            Eigen::Vector3f p3 = points.row(2);
            
            Eigen::Vector3f v1 = p2 - p1;
            Eigen::Vector3f v2 = p3 - p1;
            Eigen::Vector3f normal = v1.cross(v2).normalized();
            
            return Plane(normal, p1);
        }, py::arg("points"),
           "create plane by three points")
        
        .def_static("compute_plane_distance", [](const Plane& plane, const MatF3& points) -> Eigen::VectorXd {
            Eigen::VectorXd distances(points.rows());
            for (int i = 0; i < points.rows(); ++i) {
                distances[i] = plane.distanceToPoint(points.row(i));
            }
            return distances;
        }, py::arg("plane"), py::arg("points"),
           "compute all distances of points to plane")
        
        .def("__repr__", [](const PlaneGrouping& self) {
            std::stringstream ss;
            ss << "PlaneGrouping()";
            return ss.str();
        });
        
    m.def("version", []() {
        return "PlaneGrouping 1.0.0";
    }, "version");
    
    
    m.def("compute_point_distances", [](const MatF3& points, const Plane& plane) -> py::array_t<double> {
        Eigen::VectorXd distances(points.rows());
        for (int i = 0; i < points.rows(); ++i) {
            distances[i] = plane.distanceToPoint(points.row(i));
        }
        py::array_t<double> result({points.rows()});
        double* ptr = static_cast<double*>(result.request().ptr);
        Eigen::Map<Eigen::VectorXd>(ptr, points.rows()) = distances;
        
        return result;
    }, py::arg("points"), py::arg("plane"),
       "compute distance of point to plane");

    py::register_exception<std::runtime_error>(m, "PlaneGroupingError");


    m.attr("__doc__") = R"pbdoc(
GRAPH_GROUPING python bindings

Typical usage:

    import numpy as np
    import GRAPH_GROUPING as rg

    points = np.asarray(points, dtype=np.float32, order="C")
    normals = np.asarray(normals, dtype=np.float32, order="C")
    flags = np.asarray(flags, dtype=bool)

    knn = rg.KNNSearch(points, 8)
    rvd = rg.PointCloudRVD(points, normals, flags)

    rvd.buildKnnGraph(knn, 26, True, 60.0, 2)
    # or:
    # rvd.buildRadiusGraph(knn, 0.02, True, 60.0)

    rvd.label(cellSeedNums, nearestCloudIdsFlat, cellNodeIds)
    results = rvd.get_results()
)pbdoc";
}
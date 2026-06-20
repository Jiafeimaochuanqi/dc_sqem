#ifndef __OCTREE_H__
#define __OCTREE_H__
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>
#include <iostream>
#include <Eigen/Dense>
#include <vector>
#include <fstream>
#include <set>
#include <unordered_map>
#include <unordered_set>
#include "octree_node.h"
#include "sqef.h"
#include "NanoKNNSearch.h"
#include "MulPriorityQueue.hpp"
#include "mesh_utils.hpp"         
#include "sdf_projector.hpp" 
namespace py = pybind11;
inline int flatten_cell_index(int x, int y, int z, int resolution)
{
    return (x * resolution + y) * resolution + z;
}

inline void append_to_vectorxi(Eigen::VectorXi& v, int value)
{
    const Eigen::Index old_size = v.size();
    v.conservativeResize(old_size + 1);
    v(old_size) = value;
}
inline bool in_resolution_bounds(int x, int y, int z, int resolution)
{
    return x >= 0 && x < resolution &&
           y >= 0 && y < resolution &&
           z >= 0 && z < resolution;
}


inline MatF3 empty_matf3()
{
    return MatF3(0, 3);
}

class Octree
{
public:
    // mesh vertices
    std::vector<Eigen::Vector3f> m_mesh_v;

    std::vector<float> m_mesh_v_radii;

    std::vector<int> m_mesh_v_to_leaf_node_indexes;

    std::vector<Eigen::Vector3f> m_mesh_v_simulated;

    std::vector<MatF3> cell_samples;
    std::vector<MatF3> mesh_samples;

    // the point types of mesh vertices:0 for CORNER, 1 for EDGE, 2 for PLANE
    std::vector<int> m_v_type;

    // mesh faces (quad)
    std::vector<std::vector<int> > m_mesh_f;
    std::vector<std::vector<int> > m_mesh_f_unique;

    std::vector<std::vector<int> > m_mesh_f_tri;

    std::vector<std::vector<int> >filtered_mesh_f;
    std::vector<int> filtered_mesh_f_ids;
    std::vector<std::vector<int> >filtered_mesh_f_tri;

    std::vector<float> m_mesh_f_area2;

    MatF2 m_mesh_f_udfs;

    //cell id corresponding to the mesh faces
    std::vector<std::vector<int> > m_mesh_cell_id;
    //every four elements in subarray 0 for empty cell, 1 for virtual cell, 2 for real cell
    std::vector<std::vector<int>> m_mesh_cell_type;


    std::vector<int> cylinder_node_ids;

    std::set<std::pair<int,int>> alone_edges;
    //medial axis point feature flags
    ArrayB1 feature_flags;

    std::unordered_map<std::array<int, 4>, int, QuadKeyHash> cell_quad2face_id;
    std::unordered_map<int, std::vector<int>> cell_quad2face_id_inverse;
    std::unordered_map<std::array<int, 4>, std::vector<int>, QuadKeyHash> cell_quad2face;

    std::vector<std::vector<int> > cavity_face_ids;
    std::vector<std::vector<int> > cavity_leaf_node_indexes;
    std::vector<std::vector<int> > cavity_vertex_indexes;
    ArrayB1 cavity_flags;

    std::vector<std::vector<int> > origin_cavity_face_ids;
    std::vector<std::vector<int> > origin_cavity_leaf_node_indexes;
    std::vector<std::vector<int> > origin_cavity_vertex_indexes;
    ArrayB1 origin_cavity_flags;




    ArrayB1 deleted_face_flag;
    ArrayB1 static_face_flag;
    ArrayB1 deleted_alone_edge_flag;
    ArrayB1 first_deleted_face_flag;
    ArrayB1 second_deleted_face_flag;
    ArrayB1 third_deleted_face_flag;
    
    std::vector<int> quote_vertex_ids;
    std::vector<std::vector<int>> be_quoted_vertex_ids;
    std::vector<int> quote_leaf_node_indexes;
    std::vector<std::vector<int>> be_quoted_leaf_node_indexes;
    ArrayB1 m_mesh_v_adj_cylinder;

    MatF3 convex_feature_points=MatF3::Zero(0,3);
    MatF3 convex_corner_points=MatF3::Zero(0,3);
    float position_eps;
    float norm_eps=1e-3;
    std::vector<Eigen::VectorXi> node_contain_corner_ids;
    std::vector<Eigen::VectorXi> strict_node_contain_corner_ids;
    Eigen::VectorXi adj_nearest_leaf_node_indices;
    ArrayB1 adj_contain_outer_corner_flags;
    Eigen::VectorXi adj_corner_node_id;
    std::vector<std::pair<Eigen::Vector3f, Eigen::Vector3f>> bounding_boxes;
    std::unordered_map<int,int> deleted_face_map;
    float angle_threshold0;
    float angle_threshold;
    std::vector<std::pair<int, int>>degree_four_edges;
    // std::unordered_map<int, std::unordered_set<int>> spoint_id_to_node_id_map;
    std::unordered_map<int, std::unordered_set<int>> spoint_id_to_sid_map;
    Octree(int max_depth, Eigen::Vector3f min_corner = Eigen::Vector3f::Constant(3, -1.), Eigen::Vector3f max_corner = Eigen::Vector3f::Constant(3, 1.), int sampling_depth=1);
    ~Octree(){}
public:
    
    std::tuple<
    Eigen::VectorXi,
    Eigen::VectorXf,
    MatF3,
    MatF3,
    MatF3,
    Eigen::VectorXi,
    ArrayB1
    >
    get_group_data(OctreeNode& node);
    void compute_inside_surface_point_ids(const MatF3& spoints,OctreeNode& node);
    bool pre_check_node_build(OctreeNode& node, const MatF3& spoints);

    void scan_occupied();
    void compute_node_strict_contain_feature_ids();
    void scan_occupied_by_index();
    void build_spoint_id_maps();
    void adaptive_subdivide_sqef(const MatF3& spoints,float margin);
    void run_leaf_node_mat_search(
        const MatF3& spoints,
        const MatF3& sns,
        const ArrayB1& sp_feature_mask,
        const Eigen::VectorXi& patch_ids,
        const ArrayB1& sp_convex_feature_mask,
        double dihedral_angle_threshold = 135.0,
        double min_dihedral_angle_degree = 0.0
    );
private:
    // the root node
    OctreeNode m_root_node;

    // the max depth of the octree
    int m_max_depth;

    // the min corner
    Eigen::Vector3f m_min_corner;

    // the max corner
    Eigen::Vector3f m_max_corner;

    // the subdivision depth for sampling in a cell
    int m_sampling_depth;

    // the sampling resolution on each edge
    int m_capacity_per_dim;

    //octree resolution
    int resolution;

    //non empty flags
    ArrayB1 occupied;

    //non empty node indexes
    Eigen::VectorXi occupied_cell;

    std::unordered_map<int,Eigen::Vector3i> occupied_cell_inverse;

    Eigen::Vector3i root_min_corner_id;
    Eigen::Vector3i root_max_corner_id;

    // the unit edge length
    float m_unit_edge_length;
    float box_edge_length;

    // the current leaf nodes
    std::vector<std::reference_wrapper<OctreeNode> > m_leaf_nodes;

    std::vector<std::reference_wrapper<OctreeNode> > drop_nodes;

    std::vector<std::pair<int,int>> rvd_graph_edges;

    // the existence of grids, indexed by the global id
    ArrayB1 m_grid_existence;//Mark whether a global grid point already exists or has been sampled.

    // the map from global id to sampling id
    Eigen::VectorXi m_gid2sid;

    // the sampling pattern
    MatI3 m_sampling_pattern;

    std::vector<MatI3> m_sampling_patterns_for_depth;

    std::vector<float> box_edge_length_for_depth;

    // the coordinate, udf, and gradient values of centroids
    MatF3    m_sample_coords;
    MatF3    m_sample_mesh_sns;
    Eigen::VectorXf    m_sample_mesh_sdfs;
    MatF3    m_sample_mesh_projs;
    Eigen::VectorXi m_sample_mesh_point_ids;
    ArrayB1 m_sample_mesh_masks;
    std::vector<std::optional<SQEF>> sqefs;;
    std::vector<DualPoint>results;
    std::unordered_map<int,std::vector<std::vector<int>> > node_id_map_cell_sample_clusters;
    std::unordered_map<int,std::vector<std::vector<int>> > node_id_map_spoint_clusters;
    std::unique_ptr<NanoKNNSearch<float>> center_kdtree;
    

    std::unordered_map<int, int> vertex_to_leaf_cell;
    std::unordered_map<int, int> vertex_to_leaf_cell_inverse;
    // mesh generation
    void cellProcContour(OctreeNode& node);

    void faceProcContour(const std::array<OctreeNode*, 2>& nodes,int dir);

    void edgeProcContour(const std::array<OctreeNode*, 4>& nodes,int dir);

    void processEdgeWrite(const std::array<OctreeNode*, 4>& nodes,int dir);
public:
    ArrayB1 get_m_grid_existence()
    {
        return m_grid_existence;
    }

    Eigen::VectorXi get_m_gid2sid()
    {
        return m_gid2sid;
    }
    MatF3 get_centroids_of_leaf_nodes();
    void build_leaf_nodes_adjacency()
    {
        const Eigen::Index leaf_count =static_cast<Eigen::Index>(m_leaf_nodes.size());

        m_leaf_node_pos_types =Eigen::VectorXi::Zero(leaf_count);

        m_leaf_nodes_adjacency =Eigen::MatrixXi::Constant(leaf_count, 6, -1);

        two_ring_adjacency.clear();
        two_ring_adjacency.resize(static_cast<std::size_t>(leaf_count));

        const int scale = 1 << m_sampling_depth;

        // -x, +x, -y, +y, -z, +z
        static const int neighbor_offsets[6][3] = {
            {-1,  0,  0},
            { 1,  0,  0},
            { 0, -1,  0},
            { 0,  1,  0},
            { 0,  0, -1},
            { 0,  0,  1}
        };

        static const int pos_bit_masks[6] = {
            0b000001,
            0b000010,
            0b000100,
            0b001000,
            0b010000,
            0b100000
        };
        #pragma omp parallel for schedule(static)
        for (int leaf_node_index = 0;leaf_node_index < leaf_count;++leaf_node_index)
        {
            OctreeNode& node =m_leaf_nodes[static_cast<std::size_t>(leaf_node_index)].get();

            const Eigen::Vector3i coord =node.m_min_corner_id / scale;

            const int x = coord(0);
            const int y = coord(1);
            const int z = coord(2);

            for (int k = 0; k < 6; ++k) {
                const int nx = x + neighbor_offsets[k][0];
                const int ny = y + neighbor_offsets[k][1];
                const int nz = z + neighbor_offsets[k][2];

                if (!in_resolution_bounds(nx, ny, nz, resolution)) {
                    continue;
                }

                const int flat_id = flatten_cell_index(nx, ny, nz, resolution);

                if (occupied(flat_id)) {
                    const int neighbor_index = occupied_cell(flat_id);

                    m_leaf_nodes_adjacency(leaf_node_index, k) =neighbor_index;

                    m_leaf_node_pos_types(leaf_node_index) |=pos_bit_masks[k];
                }
            }
        }

        static const int two_ring_offsets[12][3] = {
            {-1, -1,  0},
            {-1,  0, -1},
            { 0, -1, -1},
            { 1,  1,  0},
            { 1,  0,  1},
            { 0,  1,  1},
            {-1,  1,  0},
            {-1,  0,  1},
            { 0, -1,  1},
            { 1, -1,  0},
            { 1,  0, -1},
            { 0,  1, -1}
        };
        #pragma omp parallel for schedule(static)
        for (Eigen::Index leaf_node_index = 0;leaf_node_index < leaf_count;++leaf_node_index)
        {
            OctreeNode& node =m_leaf_nodes[static_cast<std::size_t>(leaf_node_index)].get();

            const Eigen::Vector3i coord =node.m_min_corner_id / scale;

            const int x = coord(0);
            const int y = coord(1);
            const int z = coord(2);

            std::vector<int>& two_ring_neighbors =two_ring_adjacency[static_cast<std::size_t>(leaf_node_index)];

            two_ring_neighbors.clear();
            two_ring_neighbors.reserve(12);

            for (const auto& offset : two_ring_offsets) {
                const int nx = x + offset[0];
                const int ny = y + offset[1];
                const int nz = z + offset[2];

                if (!in_resolution_bounds(nx, ny, nz, resolution)) {
                    continue;
                }

                const int flat_id = flatten_cell_index(nx, ny, nz, resolution);

                if (occupied(flat_id)) {
                    const int neighbor_index = occupied_cell(flat_id);
                    two_ring_neighbors.push_back(neighbor_index);
                }
            }
        }
    }
public:
    OctreeNode& get_root_node()
    {
        return m_root_node;
    }

    const OctreeNode& get_root_node() const
    {
        return m_root_node;
    }

    std::vector<OctreeNode*> get_leaf_nodes()
    {
        std::vector<OctreeNode*> out;
        out.reserve(m_leaf_nodes.size());

        for (auto& node_ref : m_leaf_nodes) {
            out.push_back(&node_ref.get());
        }

        return out;
    }

    std::vector<OctreeNode*> get_drop_nodes()
    {
        std::vector<OctreeNode*> out;
        out.reserve(drop_nodes.size());

        for (auto& node_ref : drop_nodes) {
            out.push_back(&node_ref.get());
        }

        return out;
    }

    const std::vector<DualPoint>& get_results() const
    {
        return results;
    }

    int get_max_depth() const
    {
        return m_max_depth;
    }

    int get_sampling_depth() const
    {
        return m_sampling_depth;
    }

    float get_unit_edge_length() const
    {
        return m_unit_edge_length;
    }

    float get_box_edge_length() const
    {
        return box_edge_length;
    }

    Eigen::Vector3f get_min_corner() const
    {
        return m_min_corner;
    }

    Eigen::Vector3f get_max_corner() const
    {
        return m_max_corner;
    }

    int get_capacity_per_dim() const
    {
        return m_capacity_per_dim;
    }

    int get_resolution() const
    {
        return resolution;
    }
    MatF3 get_m_sample_coords()const
    {
        return m_sample_coords;
    }
private:
    Eigen::VectorXi m_leaf_node_pos_types;
    Eigen::MatrixXi m_leaf_nodes_adjacency;
    std::vector<std::vector<int>> two_ring_adjacency;
    std::vector<ListNode> leaf_link_nodes;
    
// Generate full 3D grid sampling pattern from 0 to stop-1 with step
    MatI3 generate_sample_pattern(int stop, int step = 1) {
        int n = (stop + step - 1) / step;
        MatI3 sample_grids(n*n*n, 3);
        int idx = 0;
        for (int x = 0; x < stop; x += step) {
            for (int y = 0; y < stop; y += step) {
                for (int z = 0; z < stop; z += step) {
                    sample_grids(idx, 0) = x;
                    sample_grids(idx, 1) = y;
                    sample_grids(idx, 2) = z;
                    idx++;
                }
            }
        }
        return sample_grids;
    }

    // Generate sampling pattern for a given depth
    MatI3 generate_sample_pattern_of_depth(int depth) {
        if (depth > m_max_depth) throw std::invalid_argument("Depth exceeds max depth");
        int stop = static_cast<int>(std::pow(2, m_sampling_depth + m_max_depth - depth) + 1);
        int step = static_cast<int>(std::pow(2, m_max_depth - depth));
        return generate_sample_pattern(stop, step);
    }

    // Compute physical box length for a given depth
    float compute_box_length_of_depth(int depth) {
        if (depth > m_max_depth) throw std::invalid_argument("Depth exceeds max depth");
        return m_unit_edge_length * std::pow(2, m_sampling_depth) * std::pow(2, m_max_depth - depth);
    }
    // convert a 3D grid tuple to a global index
    int encode(const Eigen::Vector3i& grid);

    // convert a global index to a 3D grid tuple
    Eigen::Vector3i decode(int gid);

    // compute the coordinate of a grid
    Eigen::Vector3f convert_grid_to_coords(const Eigen::Vector3i& grid)const;
    MatF3 batch_convert_grid_to_coords(const MatI3& grid)const;
    // get new samples
    std::tuple<Eigen::VectorXi,MatF3> get_samples_of_new_nodes_optimized();

    void set_new_grid_shape_data(
        const Eigen::VectorXi& new_grid_indices,
        const Eigen::VectorXf& new_grid_shape_sdfs,
        const MatF3& new_grid_shape_sns,
        const MatF3& new_grid_shape_projs,
        const Eigen::VectorXi& new_grid_shape_point_ids,
        const ArrayB1& new_grid_shape_masks
    );
    
    bool check_is_corner_type(int leaf_node_index) const
    {
        if (leaf_node_index < 0 ||
            leaf_node_index >= static_cast<int>(m_leaf_nodes.size()))
        {
            throw std::out_of_range("check_is_corner_type: leaf_node_index out of range.");
        }

        const int node_pos_type = m_leaf_node_pos_types(leaf_node_index);
        // bit order: -x, +x, -y, +y, -z, +z

        auto neighbor_has_convex_feature = [&](int direction_index) -> bool {
            const int adj_leaf_node_index =
                m_leaf_nodes_adjacency(leaf_node_index, direction_index);

            if (adj_leaf_node_index < 0 ||
                adj_leaf_node_index >= static_cast<int>(m_leaf_nodes.size()))
            {
                return false;
            }

            const OctreeNode& adj_node =
                m_leaf_nodes[static_cast<std::size_t>(adj_leaf_node_index)].get();

            return adj_node.inside_convex_feature_ids.size() > 0;
        };

        // x direction: -x, +x
        const int x_bits = node_pos_type & 0b000011;

        if (!(x_bits == 0b000001 ||
            x_bits == 0b000010 ||
            x_bits == 0b000011))
        {
            return false;
        }

        if (x_bits == 0b000001) {
            if (!neighbor_has_convex_feature(0)) {
                return false;
            }
        } else if (x_bits == 0b000010) {
            if (!neighbor_has_convex_feature(1)) {
                return false;
            }
        }

        // y direction: -y, +y
        const int y_bits = node_pos_type & 0b001100;

        if (!(y_bits == 0b000100 ||
            y_bits == 0b001000 ||
            y_bits == 0b001100))
        {
            return false;
        }

        if (y_bits == 0b000100) {
            if (!neighbor_has_convex_feature(2)) {
                return false;
            }
        } else if (y_bits == 0b001000) {
            if (!neighbor_has_convex_feature(3)) {
                return false;
            }
        }

        // z direction: -z, +z
        const int z_bits = node_pos_type & 0b110000;

        if (!(z_bits == 0b010000 ||
            z_bits == 0b100000 ||
            z_bits == 0b110000))
        {
            return false;
        }

        if (z_bits == 0b010000) {
            if (!neighbor_has_convex_feature(4)) {
                return false;
            }
        } else if (z_bits == 0b100000) {
            if (!neighbor_has_convex_feature(5)) {
                return false;
            }
        }

        return true;
    }
    Eigen::VectorXi contain_feature_ids(
        const MatF3& feature_points,
        const OctreeNode& node,
        float offset) const
    {
        if (feature_points.rows() == 0) {
            return Eigen::VectorXi(0);
        }

        const Eigen::Vector3f min_corner =
            convert_grid_to_coords(node.m_min_corner_id);

        const Eigen::Vector3f max_corner =
            convert_grid_to_coords(node.m_max_corner_id);

        const Eigen::Vector3f offset_dir =
            max_corner - min_corner;

        const Eigen::Vector3f ext_min_corner =
            min_corner - offset_dir * offset;

        const Eigen::Vector3f ext_max_corner =
            max_corner + offset_dir * offset;

        return indices_in_bbox(
            feature_points,
            ext_min_corner,
            ext_max_corner
        );
    }

    void precompute_node_data(const MatF3& centroids_coords)
    {
        const Eigen::Index leaf_count =
            static_cast<Eigen::Index>(m_leaf_nodes.size());

        if (centroids_coords.rows() != leaf_count || centroids_coords.cols() != 3) {
            throw std::invalid_argument(
                "precompute_node_data: centroids_coords must have shape (num_leaf_nodes, 3)."
            );
        }

        // self.bounding_boxes = []
        bounding_boxes.clear();
        bounding_boxes.reserve(static_cast<std::size_t>(leaf_count));

        for (auto& leaf_ref : m_leaf_nodes) {
            const OctreeNode& node = leaf_ref.get();

            const Eigen::Vector3f min_corner =
                convert_grid_to_coords(node.m_min_corner_id);

            const Eigen::Vector3f max_corner =
                convert_grid_to_coords(node.m_max_corner_id);

            bounding_boxes.emplace_back(min_corner, max_corner);
        }

        // self.node_contain_corner_ids = []
        // self.strict_node_contain_corner_ids = []
        node_contain_corner_ids.clear();
        strict_node_contain_corner_ids.clear();

        node_contain_corner_ids.reserve(static_cast<std::size_t>(leaf_count));
        strict_node_contain_corner_ids.reserve(static_cast<std::size_t>(leaf_count));

        for (auto& leaf_ref : m_leaf_nodes) {
            const OctreeNode& node = leaf_ref.get();

            node_contain_corner_ids.push_back(
                contain_feature_ids(
                    convex_corner_points,
                    node,
                    5e-2f
                )
            );

            strict_node_contain_corner_ids.push_back(
                contain_feature_ids(
                    convex_corner_points,
                    node,
                    0.0f
                )
            );
        }

        // self.adj_nearest_leaf_node_indices = [-1 for _ in range(len(self.m_leaf_nodes))]
        adj_nearest_leaf_node_indices =
            Eigen::VectorXi::Constant(leaf_count, -1);

        for (Eigen::Index i = 0; i < leaf_count; ++i) {
            OctreeNode& node =
                m_leaf_nodes[static_cast<std::size_t>(i)].get();

            if (node_contain_corner_ids[static_cast<std::size_t>(i)].size() > 0 &&
                check_is_corner_type(static_cast<int>(i)))
            {
                const Eigen::VectorXi& corner_ids =
                    node_contain_corner_ids[static_cast<std::size_t>(i)];

                const MatF3 corner_points =
                    convex_corner_points(corner_ids, Eigen::all).eval();

                const Eigen::Vector3f target_center =
                    corner_points.colwise().mean().transpose();

                int best_adj_id = -1;
                float min_dis = std::numeric_limits<float>::infinity();

                for (int k = 0; k < 6; ++k) {
                    const int adj_node_id =
                        m_leaf_nodes_adjacency(i, k);

                    if (adj_node_id == -1) {
                        continue;
                    }

                    if (adj_node_id < 0 ||
                        adj_node_id >= static_cast<int>(m_leaf_nodes.size()))
                    {
                        continue;
                    }

                    const OctreeNode& adj_node =
                        m_leaf_nodes[static_cast<std::size_t>(adj_node_id)].get();

                    const bool adj_has_convex_feature =
                        adj_node.inside_convex_feature_ids.size() > 0;

                    const bool adj_has_corner =
                        node_contain_corner_ids[static_cast<std::size_t>(adj_node_id)].size() > 0;

                    if (adj_has_convex_feature && !adj_has_corner) {
                        const Eigen::Vector3f adj_center =
                            centroids_coords.row(adj_node_id).transpose();

                        const float adj_dis =
                            (adj_center - target_center).norm();

                        if (adj_dis < min_dis) {
                            min_dis = adj_dis;
                            best_adj_id = adj_node_id;
                        }
                    }
                }

                adj_nearest_leaf_node_indices(i) = best_adj_id;
            }
        }

        // self.adj_contain_outer_corner_flags = np.zeros(len(self.m_leaf_nodes), dtype=bool)
        // self.adj_corner_node_id = [-1 for _ in range(len(self.m_leaf_nodes))]
        adj_contain_outer_corner_flags =
            ArrayB1::Constant(leaf_count, false);

        adj_corner_node_id =
            Eigen::VectorXi::Constant(leaf_count, -1);

        for (Eigen::Index i = 0; i < leaf_count; ++i) {
            const OctreeNode& node =
                m_leaf_nodes[static_cast<std::size_t>(i)].get();

            if (node.inside_convex_feature_ids.size() == 0) {
                continue;
            }

            for (int k = 0; k < 6; ++k) {
                const int adj_id =
                    m_leaf_nodes_adjacency(i, k);

                if (adj_id == -1) {
                    continue;
                }

                if (adj_id < 0 ||
                    adj_id >= static_cast<int>(m_leaf_nodes.size()))
                {
                    continue;
                }

                if (node_contain_corner_ids[static_cast<std::size_t>(adj_id)].size() > 0) {
                    adj_contain_outer_corner_flags(i) = true;
                    adj_corner_node_id(i) = adj_id;
                    break;
                }
            }
        }
    }

    int query_nearest_center_index(const Eigen::Vector3f& p) const
    {
        if (!center_kdtree) {
            throw std::runtime_error("query_nearest_center_index: center_kdtree is null.");
        }

        int index = -1;
        float dist2 = 0.0f;

        center_kdtree->searchKNN1(index, dist2, p.data());

        return index;
    }
private:
    std::unordered_map<int, std::vector<int>> face2cavity;
    std::unordered_map<std::pair<int, int>, std::vector<int>, PairHash> edge2cavity;
    std::unordered_map<int, std::vector<int>> vertex2cavity;

    std::unordered_map<int, std::vector<int>> origin_face2cavity;
    std::unordered_map<std::pair<int, int>, std::vector<int>, PairHash> origin_edge2cavity;
    std::unordered_map<int, std::vector<int>> origin_vertex2cavity;


    

    std::unordered_map<int, std::unordered_map<int, int>> cavity_adjacenty;
private:
    std::unordered_map<int, std::vector<std::pair<int, int>>> face_to_edge;

    std::unordered_map<
        std::pair<int, int>,
        std::vector<int>,
        PairHash
    > edge_to_faces;

    std::unordered_map<
        int,
        std::vector<std::pair<int, int>>
    > vertex_to_edge;

    std::unordered_map<
        int,
        std::vector<std::vector<int>>
    > face_adjacency;

private:
    bool check_edge_exist(const std::pair<int, int>& edge) const;

    void add_edge_to_vertex_map(const std::pair<int, int>& edge);
public:
    void batch_pre_solve_sqef(
        const MatF3& spoints,
        const ArrayB1& sp_convex_feature_mask,
        SDFProjectorFunc& cloud_sdf_grad_mask_func,
        const MatF3& centroids_coords,
        float singular_value_threshold,
        float corner_factor=1e-2, 
        float edge_factor=1e-2, 
        float plane_factor=1e-2
    );
    void batch_locate(); 
    void dfs_relate();
    // get the number of the leaf nodes
    int get_num_of_leaf_nodes() { return m_leaf_nodes.size(); }

    // generate quad mesh
    void generate_mesh();
    void scan_cavity();
    void backup_cavity_data();
    void initial_topology_data();
    ArrayB1 setup_cavity_flags(const std::vector<int>* group = nullptr);
    void make_vertex_edge_face_topology();
    void make_cavity_adjacency_dict();
    void generate_alone_edges();
    void triangulate_faces_by_udf();
    void update_cavity_data();
    void clean_cavity_face();
    void post_process();
    
    
public:
    int f_num() const;
    int v_num() const;
    int e_num() const;
    int cavity_num(const std::vector<int>* group = nullptr) const;
    int eular_characteristic() const;
    //Check whether this face is a pocket face.
    bool is_avaliable_pocket(int face_id) const {
        const auto& face = m_mesh_f[face_id];
        const int n = static_cast<int>(face.size());
        for (int i = 0; i < n; ++i) {
            int v0 = face[i];
            int v1 = face[(i + 1) % n];
            auto edge = make_sorted_edge(v0, v1);
            auto it = edge_to_faces.find(edge);
            if (it != edge_to_faces.end() && it->second.size() == 1) {
                if (!deleted_face_flag[face_id]) {
                    return true;
                }
            }
        }
        return false;
    }

    //Check whether there is an available pocket in the queue, return the first available face id or -1.
    int has_avaliable_pocket(const MulPriorityQueue<int>& queue) const {
        std::vector<int> face_ids = queue.traverse_via_map();
        for (int face_id : face_ids) {
            if (is_avaliable_pocket(face_id)) {
                return face_id;
            }
        }
        return -1;
    }
private:
    void update_cavity_tables(int face_id);
    void update_relate_face_edge_vertex_table(int face_id);

    std::vector<float> compute_face_rank_global(int face_id);
    std::vector<float> compute_face_rank_local(int face_id);

    std::vector<std::pair<int, std::vector<float>>>
    compute_all_cavity_face_rank_global(const std::vector<int>& all_cavity_face_id);

    std::vector<std::pair<int, std::vector<float>>> compute_all_cavity_face_rank_local(const std::vector<int>& all_cavity_face_id);
    std::vector<float> compute_face_rank_impl(int face_id);
    void update_relate_face_rank_global(int face_id,MulPriorityQueue<int>& queue);
    void update_relate_face_rank_local(int face_id, MulPriorityQueue<int>& queue);
    void update_relate_table_and_rank_global(
        int face_id,
        MulPriorityQueue<int>& queue
    );

    void update_relate_table_and_rank_local(
        int face_id,
        MulPriorityQueue<int>& queue
    );
private:
    bool check_is_feature_bd_edge(const std::pair<int, int>& bd_edge) const;
    void connect_edge_to_adj_node(int cur_vertex_idx, int adj_node_id0, int adj_node_id1);
    void generate_alone_edges_for_cad();
    bool try_delete_face(int fid, int before_eular);
    using Edge = std::pair<int, int>;
    using FaceEdgeType = std::pair<std::vector<std::pair<int, int>>, std::vector<std::pair<int, int>>>;
    void update_face_edge_type_map(std::unordered_map<int, FaceEdgeType>& face_edge_type_map,int face_id);
    int filter_isolate_patch_one_batch(std::unordered_map<int, FaceEdgeType>& face_edge_type_map);
public:
    void grow_by_max_depth(const MatF3&spoints,SDFProjectorFunc& cloud_sdf_grad_mask_func,float margin_threshold);
    void solve_and_locate_vertices(
        const MatF3& spoints,
        const ArrayB1& sp_convex_feature_mask,
        SDFProjectorFunc& cloud_sdf_grad_mask_func,
        const MatF3& centroids_coords,
        float singular_value_threshold);
    void build_mesh_and_topology();
    void compute_face_udf_and_triangulate(SDFProjectorFunc& projFunc);
    void clean_and_post_process();
    py::dict visualize_octree_cells() const;
    void save_ma_mesh(const std::string& savepath)const;
    void save_medial_mesh(std::string savepath)const;
};
#endif // __OCTREE_H__
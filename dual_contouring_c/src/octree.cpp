
#include <iostream>
#include <stdexcept>
#include <omp.h>
#include <numeric>
#include "octree.h"

#include "plane_grouping/dbscan.hpp"
#include "cloud_rvd.h"
#include "tangent.hpp"
#include "MeshConnectivityAnalyzer.hpp"
#include "cylinder_util.hpp"
// #include "cnpy.h"

Eigen::VectorXi g_child_2_vtk_order=[]
{
    Eigen::VectorXi m(8);
    m << 0,4,6,2,1,5,7,3;
    return m;
}();


const int cellProcFaceMask[12][3] = { 
    {0,4,0},{1,5,0},{2,6,0},{3,7,0},
    {0,2,1},{4,6,1},{1,3,1},{5,7,1},
    {0,1,2},{2,3,2},{4,5,2},{6,7,2} 
};

const int cellProcEdgeMask[6][5] = { 
    {0,1,2,3,0},
    {4,5,6,7,0},
    {0,4,1,5,1},
    {2,6,3,7,1},
    {0,2,4,6,2},
    {1,3,5,7,2} 
};

const int faceProcFaceMask[3][4][3] = {
    {{4,0,0},{5,1,0},{6,2,0},{7,3,0}},
    {{2,0,1},{6,4,1},{3,1,1},{7,5,1}},
    {{1,0,2},{3,2,2},{5,4,2},{7,6,2}}
};

const int faceProcEdgeMask[3][4][6] = {
    {{1,4,0,5,1,1},{1,6,2,7,3,1},{0,4,6,0,2,2},{0,5,7,1,3,2}},
    {{0,2,3,0,1,0},{0,6,7,4,5,0},{1,2,0,6,4,2},{1,3,1,7,5,2}},
    {{1,1,0,3,2,0},{1,5,4,7,6,0},{0,1,5,0,4,1},{0,3,7,2,6,1}}
};

const int edgeProcEdgeMask[3][2][5] = {
    {{3,2,1,0,0},{7,6,5,4,0}},
    {{5,1,4,0,1},{7,3,6,2,1}},
    {{6,4,2,0,2},{7,5,3,1,2}},
};




inline DualPoint make_dual_point(
    const Eigen::Vector3f& point,
    DualPointType type,
    bool is_inside,
    float radius = 0.0f)
{
    DualPoint out;
    out.m_point = point;
    out.m_type = type;
    out.m_is_inside = is_inside;
    out.radius = radius;
    return out;
}


inline std::vector<std::vector<int>> clusters_from_labels(
    const Eigen::VectorXi& labels)
{
    std::map<int, std::vector<int>> label_to_cluster;

    for (Eigen::Index i = 0; i < labels.size(); ++i) {
        label_to_cluster[labels(i)].push_back(static_cast<int>(i));
    }

    std::vector<std::vector<int>> clusters;
    clusters.reserve(label_to_cluster.size());

    for (auto& kv : label_to_cluster) {
        clusters.push_back(std::move(kv.second));
    }

    return clusters;
}

inline MatF3 rows_by_ids(const MatF3& mat, const Eigen::VectorXi& ids)
{
    if (ids.size() == 0) {
        return MatF3(0, 3);
    }

    return mat(ids, Eigen::all).eval();
}

inline Eigen::VectorXf vector_by_ids(
    const Eigen::VectorXf& v,
    const Eigen::VectorXi& ids)
{
    if (ids.size() == 0) {
        return Eigen::VectorXf(0);
    }

    return v(ids).eval();
}

inline Eigen::VectorXi vectorxi_by_ids(
    const Eigen::VectorXi& v,
    const Eigen::VectorXi& ids)
{
    if (ids.size() == 0) {
        return Eigen::VectorXi(0);
    }

    return v(ids).eval();
}

inline ArrayB1 arrayb_by_ids(
    const ArrayB1& v,
    const Eigen::VectorXi& ids)
{
    if (ids.size() == 0) {
        return ArrayB1::Constant(0, false);
    }

    return v(ids).eval();
}

inline Eigen::Vector3f nearest_point_to_center(
    const MatF3& feature_points,
    const Eigen::Vector3f& center)
{
    Eigen::VectorXf distances_sq =
        (feature_points.rowwise() - center.transpose()).rowwise().squaredNorm();

    Eigen::Index min_idx = 0;
    distances_sq.minCoeff(&min_idx);

    return feature_points.row(min_idx).transpose();
}

inline MatF3 compute_sqef_normals(
    const MatF3& nearest_points,
    const MatF3& coords,
    const Eigen::VectorXf& sdfs)
{
    MatF3 normals = nearest_points - coords;

    for (Eigen::Index i = 0; i < normals.rows(); ++i) {
        Eigen::Vector3f n = normals.row(i).transpose();
        n.stableNormalize();

        const float sign =
            (sdfs(i) > 0.0f) ? 1.0f :
            (sdfs(i) < 0.0f) ? -1.0f :
                                0.0f;

        normals.row(i) = (-sign * n).transpose();
    }

    return normals;
}

inline std::vector<std::vector<int>> filter_groups_by_negative_sdf(
    const std::vector<std::vector<int>>& groups,
    const Eigen::VectorXf& sdfs)
{
    std::vector<std::vector<int>> new_groups;
    new_groups.reserve(groups.size());

    for (const auto& group : groups) {
        std::vector<int> new_group;
        new_group.reserve(group.size());

        for (int local_id : group) {
            if (local_id >= 0 &&
                local_id < sdfs.size() &&
                sdfs(local_id) < 0.0f)
            {
                new_group.push_back(local_id);
            }
        }

        if (!new_group.empty()) {
            new_groups.push_back(std::move(new_group));
        }
    }

    return new_groups;
}

inline std::vector<std::pair<Eigen::Vector3f, Eigen::Vector3f>>
collect_one_ring_boxes(
    const Eigen::MatrixXi& adjacency,
    const std::vector<std::pair<Eigen::Vector3f, Eigen::Vector3f>>& bounding_boxes,
    Eigen::Index leaf_index)
{
    std::vector<std::pair<Eigen::Vector3f, Eigen::Vector3f>> boxes;
    boxes.reserve(6);

    for (int k = 0; k < adjacency.cols(); ++k) {
        const int adj_id = adjacency(leaf_index, k);
        if (adj_id != -1) {
            boxes.push_back(bounding_boxes[static_cast<std::size_t>(adj_id)]);
        }
    }

    return boxes;
}

inline std::vector<std::pair<Eigen::Vector3f, Eigen::Vector3f>>
collect_two_ring_boxes(
    const std::vector<std::vector<int>>& two_ring_adjacency,
    const std::vector<std::pair<Eigen::Vector3f, Eigen::Vector3f>>& bounding_boxes,
    Eigen::Index leaf_index)
{
    std::vector<std::pair<Eigen::Vector3f, Eigen::Vector3f>> boxes;

    const auto& adj_ids = two_ring_adjacency[static_cast<std::size_t>(leaf_index)];
    boxes.reserve(adj_ids.size());

    for (int adj_id : adj_ids) {
        if (adj_id != -1) {
            boxes.push_back(bounding_boxes[static_cast<std::size_t>(adj_id)]);
        }
    }

    return boxes;
}



Octree::Octree(int max_depth, Eigen::Vector3f min_corner, Eigen::Vector3f max_corner, int sampling_depth)
{
    m_max_depth = max_depth;
    m_min_corner = min_corner;
    m_max_corner = max_corner;
    m_sampling_depth = sampling_depth;

    m_capacity_per_dim = std::pow(2, max_depth + sampling_depth) + 1;
    resolution=(m_capacity_per_dim - 1) / (1 << m_sampling_depth); 
    occupied=Eigen::Array<bool, Eigen::Dynamic, 1>::Constant(resolution*resolution*resolution, false);
    occupied_cell=Eigen::VectorXi::Constant(resolution*resolution*resolution, -1);
    
    std::cout<<"Resolution: "<<resolution<<std::endl;
    std::cout<<"Samples per cell: "<<std::pow(2*sampling_depth + 1,3)<<std::endl;
    Eigen::Vector3i root_node_min_corner_id = Eigen::VectorXi::Zero(3);
    Eigen::Vector3i root_node_max_corner_id = Eigen::VectorXi::Constant(3, m_capacity_per_dim - 1);
    m_root_node = OctreeNode(root_node_min_corner_id, root_node_max_corner_id);

    // compute the unit edge length
    m_unit_edge_length = (max_corner[0] - min_corner[0]) / double(m_capacity_per_dim - 1);
    box_edge_length = m_unit_edge_length*std::pow(2, sampling_depth);
    // initialize the nodes vector and push the root node into it
    m_leaf_nodes.clear();
    m_leaf_nodes.push_back(std::ref(m_root_node));
    drop_nodes.clear();

    const Eigen::Index total_grid_num =
    static_cast<Eigen::Index>(m_capacity_per_dim) *
    static_cast<Eigen::Index>(m_capacity_per_dim) *
    static_cast<Eigen::Index>(m_capacity_per_dim);

    m_grid_existence = ArrayB1::Constant(total_grid_num, false);
    m_gid2sid = Eigen::VectorXi::Constant(total_grid_num, -1);

    // generate sampling pattern
    m_sampling_pattern = generate_sample_pattern(std::pow(2, sampling_depth) + 1);

    m_sampling_patterns_for_depth.resize(m_max_depth + 1);
    box_edge_length_for_depth.resize(m_max_depth + 1);
    for (int d = 0; d <= m_max_depth; ++d) {
        m_sampling_patterns_for_depth[d] = generate_sample_pattern_of_depth(d);
        box_edge_length_for_depth[d] = compute_box_length_of_depth(d);
    }
    m_mesh_f.clear();
}


Eigen::Vector3f Octree::convert_grid_to_coords(const Eigen::Vector3i& grid)const
{
    return m_min_corner + grid.cast<float>() * m_unit_edge_length;
}
MatF3 Octree::batch_convert_grid_to_coords(const MatI3& grid_ids) const
{
    MatF3 coords(grid_ids.rows(), 3);

    coords =grid_ids.cast<float>() * static_cast<float>(m_unit_edge_length);

    coords.rowwise() += m_min_corner.transpose();

    return coords;
}

MatF3 Octree::get_centroids_of_leaf_nodes()
{
    std::cout<<"m_leaf_nodes.size():"<<m_leaf_nodes.size()<<std::endl;
    MatF3 centroids(m_leaf_nodes.size(), 3);
    #pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < m_leaf_nodes.size(); ++i)
    {
        // get the centroid grid
        Eigen::Vector3i centroid_grid = m_leaf_nodes[i].get().m_centroid_id;
        centroids.row(i) = convert_grid_to_coords(centroid_grid).transpose();
    }
    return centroids;
}
int Octree::encode(const Eigen::Vector3i& grid)
{
    return (grid[0] * m_capacity_per_dim + grid[1]) * m_capacity_per_dim + grid[2];
}


Eigen::Vector3i Octree::decode(int gid)
{
    Eigen::Vector3i grid = Eigen::VectorXi::Zero(3);
    const int cap2 = m_capacity_per_dim * m_capacity_per_dim;
    grid[0] = gid / cap2;
    grid[1] = (gid / m_capacity_per_dim) % m_capacity_per_dim;
    grid[2] = gid % m_capacity_per_dim;
    return grid;
}


std::tuple<Eigen::VectorXi,MatF3> Octree::get_samples_of_new_nodes_optimized() {
    Eigen::VectorXi new_grid_indices(m_leaf_nodes.size()*m_sampling_pattern.size());
    MatF3 new_grid_coords(m_leaf_nodes.size()*m_sampling_pattern.size(), 3);
    int id_counter=0;
    const Eigen::Index old_sample_size = m_sample_coords.rows();
    for (auto& leaf_ref : m_leaf_nodes) {
        OctreeNode& leaf_node = leaf_ref.get();

        const MatI3& sampling_pattern =m_sampling_patterns_for_depth[leaf_node.m_depth];

        MatI3 sampling_grids =leaf_node.generate_sample_grids(sampling_pattern);

        leaf_node.m_sample_grids = sampling_grids;
        leaf_node.cell_sample_mask =ArrayB1::Constant(sampling_grids.rows(), false);

        Eigen::VectorXi gids =
            ((sampling_grids.col(0).array() * m_capacity_per_dim +
            sampling_grids.col(1).array()) * m_capacity_per_dim +
            sampling_grids.col(2).array()).matrix();

        std::vector<int> new_local_ids_vec;
        new_local_ids_vec.reserve(static_cast<std::size_t>(gids.size()));

        for (Eigen::Index i = 0; i < gids.size(); ++i) {
            if (!m_grid_existence(gids(i))) {
                new_local_ids_vec.push_back(static_cast<int>(i));
            }
        }

        if (new_local_ids_vec.empty()) {
            continue;
        }

        Eigen::Map<const Eigen::VectorXi> new_local_ids(
            new_local_ids_vec.data(),
            static_cast<Eigen::Index>(new_local_ids_vec.size())
        );

        Eigen::VectorXi new_gids = gids(new_local_ids).eval();

        MatI3 new_grids=sampling_grids(new_local_ids, Eigen::all).eval();

        MatF3 new_coords = batch_convert_grid_to_coords(new_grids);


        const Eigen::Index add_count = new_coords.rows();
        new_grid_indices.segment(id_counter,add_count) = new_gids;
        new_grid_coords.middleRows(id_counter,add_count) = new_coords;

        Eigen::VectorXi new_sids =
            Eigen::VectorXi::LinSpaced(
                add_count,
                static_cast<int>(old_sample_size) + id_counter,
                static_cast<int>(old_sample_size) + id_counter + static_cast<int>(add_count) - 1
            );

        m_grid_existence(new_gids) = ArrayB1::Constant(add_count, true);

        m_gid2sid(new_gids) = new_sids;

        id_counter += static_cast<int>(add_count);
    }
    new_grid_indices.conservativeResize(id_counter);
    new_grid_coords.conservativeResize(id_counter,3);
    const Eigen::Index add_sample_size = new_grid_coords.rows();
    const Eigen::Index new_sample_size = old_sample_size + add_sample_size;

    MatF3 origin_m_sample_coords = m_sample_coords;
    MatF3 origin_m_sample_mesh_sns = m_sample_mesh_sns;
    Eigen::VectorXf origin_m_sample_mesh_sdfs = m_sample_mesh_sdfs;
    MatF3 origin_m_sample_mesh_projs = m_sample_mesh_projs;
    Eigen::VectorXi origin_m_sample_mesh_point_ids = m_sample_mesh_point_ids;
    ArrayB1 origin_m_sample_mesh_masks = m_sample_mesh_masks;

    m_sample_coords.conservativeResize(new_sample_size, 3);
    m_sample_mesh_sns.conservativeResize(new_sample_size, 3);
    m_sample_mesh_sdfs.conservativeResize(new_sample_size);
    m_sample_mesh_projs.conservativeResize(new_sample_size, 3);
    m_sample_mesh_point_ids.conservativeResize(new_sample_size);
    m_sample_mesh_masks.conservativeResize(new_sample_size);

    if (add_sample_size>0) {
        m_sample_coords.bottomRows(new_grid_coords.rows())=new_grid_coords;
    }
    return std::make_tuple(new_grid_indices, new_grid_coords);
}

void Octree::set_new_grid_shape_data(
    const Eigen::VectorXi& new_grid_indices,
    const Eigen::VectorXf& new_grid_shape_sdfs,
    const MatF3& new_grid_shape_sns,
    const MatF3& new_grid_shape_projs,
    const Eigen::VectorXi& new_grid_shape_point_ids,
    const ArrayB1& new_grid_shape_masks)
{
    const Eigen::Index n = new_grid_indices.size();

    if (n == 0) {
        return;
    }

    if (new_grid_shape_sdfs.size() != n) {
        throw std::invalid_argument(
            "set_new_grid_shape_data: new_grid_shape_sdfs size mismatch."
        );
    }

    if (new_grid_shape_sns.rows() != n || new_grid_shape_sns.cols() != 3) {
        throw std::invalid_argument(
            "set_new_grid_shape_data: new_grid_shape_sns shape mismatch."
        );
    }

    if (new_grid_shape_projs.rows() != n || new_grid_shape_projs.cols() != 3) {
        throw std::invalid_argument(
            "set_new_grid_shape_data: new_grid_shape_projs shape mismatch."
        );
    }

    if (new_grid_shape_point_ids.size() != n) {
        throw std::invalid_argument(
            "set_new_grid_shape_data: new_grid_shape_point_ids size mismatch."
        );
    }

    if (new_grid_shape_masks.size() != n) {
        throw std::invalid_argument(
            "set_new_grid_shape_data: new_grid_shape_masks size mismatch."
        );
    }

    Eigen::VectorXi data_ids = m_gid2sid(new_grid_indices).eval();

    if ((data_ids.array() < 0).any()) {
        throw std::runtime_error(
            "set_new_grid_shape_data: some grid ids have not been assigned sample ids."
        );
    }

    if ((data_ids.array() >= m_sample_coords.rows()).any()) {
        throw std::runtime_error(
            "set_new_grid_shape_data: some sample ids are out of range."
        );
    }

    m_sample_mesh_sdfs(data_ids) = new_grid_shape_sdfs;
    m_sample_mesh_sns(data_ids, Eigen::all) = new_grid_shape_sns;
    m_sample_mesh_projs(data_ids, Eigen::all) = new_grid_shape_projs;
    m_sample_mesh_point_ids(data_ids) = new_grid_shape_point_ids;
    m_sample_mesh_masks(data_ids) = new_grid_shape_masks;
}


void Octree::scan_occupied()
{
    const Eigen::Index cell_count =
        static_cast<Eigen::Index>(resolution) *
        static_cast<Eigen::Index>(resolution) *
        static_cast<Eigen::Index>(resolution);

    occupied = ArrayB1::Constant(cell_count, false);
    occupied_cell = Eigen::VectorXi::Constant(cell_count, -1);
    occupied_cell_inverse.clear();

    const int scale = 1 << m_sampling_depth;

    for (Eigen::Index i = 0; i < static_cast<Eigen::Index>(m_leaf_nodes.size()); ++i) {
        OctreeNode& node = m_leaf_nodes[static_cast<std::size_t>(i)].get();

        node.leaf_node_index = static_cast<int>(i);

        const Eigen::Vector3i coord = node.m_min_corner_id / scale;

        const int x = coord(0);
        const int y = coord(1);
        const int z = coord(2);
        if (!in_resolution_bounds(x, y, z, resolution)) {
            std::ostringstream oss;
            oss << "scan_occupied: node.id=" << node.id
                << " has out-of-bound grid coord ("
                << x << ", " << y << ", " << z << ")";
            throw std::runtime_error(oss.str());
        }

        const int flat_id = flatten_cell_index(x, y, z, resolution);

        occupied(flat_id) = true;
        occupied_cell(flat_id) = node.id;
        occupied_cell_inverse[node.id] = coord;
    }
}
void Octree::compute_node_strict_contain_feature_ids()
{
    if (convex_feature_points.rows() <= 0) {
        return;
    }
    const int leaf_count =m_leaf_nodes.size();


    for (int i=0;i<leaf_count ; ++i) {
        OctreeNode& node = m_leaf_nodes[i].get();
        node.strict_inside_convex_feature_ids.resize(0);
        if(node.inside_convex_feature_ids.size()>0)
        {
            const Eigen::Vector3f min_corner =convert_grid_to_coords(node.m_min_corner_id);
            const Eigen::Vector3f max_corner =convert_grid_to_coords(node.m_max_corner_id);
            MatF3 cadidate_points=convex_feature_points(node.inside_convex_feature_ids,Eigen::all).eval();
            const Eigen::VectorXi local_inside_ids=indices_in_bbox(cadidate_points,min_corner,max_corner);
            if (local_inside_ids.size() > 0) {
                node.inside_convex_feature_ids = node.inside_convex_feature_ids(local_inside_ids).eval();
            }
        }
    }
}
// void Octree::compute_node_strict_contain_feature_ids()
// {
//     if (convex_feature_points.rows() <= 0) {
//         return;
//     }

//     std::unordered_map<int, OctreeNode*> id2node;
//     id2node.reserve(m_leaf_nodes.size());

//     for (auto& leaf_ref : m_leaf_nodes) {
//         OctreeNode& node = leaf_ref.get();
//         id2node[node.id] = &node;
//         node.strict_inside_convex_feature_ids.resize(0);
//     }
    
//     for (Eigen::Index convex_feature_id = 0;convex_feature_id < convex_feature_points.rows();++convex_feature_id)
//     {
//         const Eigen::Vector3f p =convex_feature_points.row(convex_feature_id).transpose();

//         const Eigen::Array3f coord_f =(p - m_min_corner.cast<float>()).array()
//             / static_cast<float>(box_edge_length);

//         const int x = static_cast<int>(std::floor(coord_f(0)));
//         const int y = static_cast<int>(std::floor(coord_f(1)));
//         const int z = static_cast<int>(std::floor(coord_f(2)));

//         if (x < 0 || y < 0 || z < 0 ||
//             x >= resolution || y >= resolution || z >= resolution) {
//             continue;
//         }

//         const int flat_id = flatten_cell_index(x, y, z, resolution);

//         const int node_id = occupied_cell(flat_id);

//         if (node_id < 0) {
//             continue;
//         }

//         auto it = id2node.find(node_id);

//         if (it == id2node.end()) {
//             throw std::runtime_error(
//                 "compute_node_strict_contain_feature_ids: node_id not found in leaf nodes."
//             );
//         }

//         OctreeNode* node = it->second;

//         append_to_vectorxi(node->strict_inside_convex_feature_ids,static_cast<int>(convex_feature_id));
//     }
// }

void Octree::scan_occupied_by_index()
{
    const Eigen::Index cell_count =
        static_cast<Eigen::Index>(resolution) *
        static_cast<Eigen::Index>(resolution) *
        static_cast<Eigen::Index>(resolution);

    occupied = ArrayB1::Constant(cell_count, false);
    occupied_cell = Eigen::VectorXi::Constant(cell_count, -1);
    occupied_cell_inverse.clear();

    const int scale = 1 << m_sampling_depth;

    for (Eigen::Index leaf_node_index = 0;
         leaf_node_index < static_cast<Eigen::Index>(m_leaf_nodes.size());
         ++leaf_node_index)
    {
        OctreeNode& node =m_leaf_nodes[static_cast<std::size_t>(leaf_node_index)].get();

        node.leaf_node_index = static_cast<int>(leaf_node_index);

        const Eigen::Vector3i coord = node.m_min_corner_id / scale;

        const int x = coord(0);
        const int y = coord(1);
        const int z = coord(2);

        if (!in_resolution_bounds(x, y, z, resolution)) {
            std::ostringstream oss;
            oss << "scan_occupied_by_index: node.id=" << node.id
                << " has out-of-bound grid coord ("
                << x << ", " << y << ", " << z << ")";
            throw std::runtime_error(oss.str());
        }

        const int flat_id = flatten_cell_index(x, y, z, resolution);

        occupied(flat_id) = true;
        occupied_cell(flat_id) = static_cast<int>(leaf_node_index);
        occupied_cell_inverse[static_cast<int>(leaf_node_index)] =coord;
    }
}

void Octree::build_spoint_id_maps()
{
    // spoint_id_to_node_id_map.clear();
    spoint_id_to_sid_map.clear();

    for (auto& leaf_ref : m_leaf_nodes) {
        OctreeNode& node = leaf_ref.get();

        // const int node_id = node.id;
        const MatI3& sampling_grids = node.m_sample_grids;

        if (sampling_grids.rows() == 0) {
            continue;
        }

        Eigen::VectorXi gids =
            ((sampling_grids.col(0).array() * m_capacity_per_dim +
              sampling_grids.col(1).array()) * m_capacity_per_dim +
              sampling_grids.col(2).array()).matrix();

        if ((gids.array() < 0).any() ||
            (gids.array() >= m_gid2sid.size()).any()) {
            throw std::runtime_error(
                "build_spoint_id_maps: some gids are out of range."
            );
        }

        Eigen::VectorXi sids = m_gid2sid(gids).eval();

        if ((sids.array() < 0).any() ||
            (sids.array() >= m_sample_mesh_point_ids.size()).any()) {
            throw std::runtime_error(
                "build_spoint_id_maps: some sample ids are invalid."
            );
        }

        Eigen::VectorXi shape_point_ids = m_sample_mesh_point_ids(sids).eval();

        for (Eigen::Index i = 0; i < shape_point_ids.size(); ++i) {
            const int pid = shape_point_ids(i);
            const int sid = sids(i);

            // spoint_id_to_node_id_map[pid].insert(node_id);
            spoint_id_to_sid_map[pid].insert(sid);
        }
    }
}

std::tuple<
    Eigen::VectorXi,
    Eigen::VectorXf,
    MatF3,
    MatF3,
    MatF3,
    Eigen::VectorXi,
    ArrayB1
>
Octree::get_group_data(OctreeNode& node)
{
    const MatI3& sampling_grids = node.m_sample_grids;

    if (sampling_grids.rows() == 0) {
        node.cell_sample_mask = ArrayB1::Constant(0, false);

        return std::make_tuple(
            Eigen::VectorXi(0),
            Eigen::VectorXf(0),
            MatF3(0, 3),
            MatF3(0, 3),
            MatF3(0, 3),
            Eigen::VectorXi(0),
            ArrayB1::Constant(0, false)
        );
    }

    Eigen::VectorXi gids =
        ((sampling_grids.col(0).array() * m_capacity_per_dim +
          sampling_grids.col(1).array()) * m_capacity_per_dim +
          sampling_grids.col(2).array()).matrix();

    if ((gids.array() < 0).any() ||
        (gids.array() >= m_gid2sid.size()).any()) {
        throw std::runtime_error("get_group_data: gids out of range.");
    }

    Eigen::VectorXi sids = m_gid2sid(gids).eval();

    if ((sids.array() < 0).any() ||
        (sids.array() >= m_sample_mesh_sdfs.size()).any()) {
        throw std::runtime_error("get_group_data: invalid sample ids.");
    }

    Eigen::VectorXf origin_sdfs =m_sample_mesh_sdfs(sids).eval();

    MatF3 origin_coords =m_sample_coords(sids, Eigen::all).eval();

    MatF3 origin_sns =m_sample_mesh_sns(sids, Eigen::all).eval();

    MatF3 origin_proj_points =m_sample_mesh_projs(sids, Eigen::all).eval();

    Eigen::VectorXi origin_shape_point_ids =m_sample_mesh_point_ids(sids).eval();

    ArrayB1 origin_feature_flags =m_sample_mesh_masks(sids).eval();

    ArrayB1 mask =(origin_sdfs.array() != 0.0f);

    //correspoind to Python: node.cell_sample_mask = mask
    node.cell_sample_mask = mask;

    // Python:
    // if np.all(origin_sdfs > 0) and len(node.inside_surface_point_ids)==0:
    //     return [],[],[],[],[],[],[]
    const bool all_positive =(origin_sdfs.array() > 0.0f).all();

    if (all_positive && node.inside_surface_point_ids.size() == 0) {
        return std::make_tuple(
            Eigen::VectorXi(0),
            Eigen::VectorXf(0),
            MatF3(0, 3),
            MatF3(0, 3),
            MatF3(0, 3),
            Eigen::VectorXi(0),
            ArrayB1::Constant(0, false)
        );
    }

    Eigen::VectorXi valid_ids = indices_from_mask(mask);

    if (valid_ids.size() == 0) {
        return std::make_tuple(
            Eigen::VectorXi(0),
            Eigen::VectorXf(0),
            MatF3(0, 3),
            MatF3(0, 3),
            MatF3(0, 3),
            Eigen::VectorXi(0),
            ArrayB1::Constant(0, false)
        );
    }

    Eigen::VectorXi out_sids =sids(valid_ids).eval();

    Eigen::VectorXf sdfs=origin_sdfs(valid_ids).eval();

    MatF3 sns=origin_sns(valid_ids, Eigen::all).eval();

    MatF3 coords=origin_coords(valid_ids, Eigen::all).eval();

    MatF3 proj_points=origin_proj_points(valid_ids, Eigen::all).eval();

    Eigen::VectorXi shape_point_ids=origin_shape_point_ids(valid_ids).eval();

    ArrayB1 feature_flags=origin_feature_flags(valid_ids).eval();

    return std::make_tuple(
        out_sids,
        sdfs,
        sns,
        coords,
        proj_points,
        shape_point_ids,
        feature_flags
    );
}

void Octree::adaptive_subdivide_sqef(const MatF3& spoints,float margin)
{
    const std::size_t old_leaf_count = m_leaf_nodes.size();

    ArrayB1 keep_flags=ArrayB1::Constant(old_leaf_count, false);

    constexpr std::size_t parallel_threshold = 8192;
    #pragma omp parallel for schedule(dynamic) if(old_leaf_count > parallel_threshold)
    for (int i = 0; i < static_cast<int>(old_leaf_count); ++i) {
        OctreeNode& node = m_leaf_nodes[static_cast<std::size_t>(i)].get();
        if (node.m_parent != nullptr) {
            const Eigen::VectorXi& parent_ids =node.m_parent->inside_convex_feature_ids;

            if (parent_ids.size() > 0) {
                const Eigen::Vector3f min_corner =
                convert_grid_to_coords(node.m_min_corner_id);

                const Eigen::Vector3f max_corner =
                    convert_grid_to_coords(node.m_max_corner_id);

                const Eigen::Vector3f offset_dir = max_corner - min_corner;

                const Eigen::Vector3f ext_min_corner =
                    min_corner - offset_dir * margin;

                const Eigen::Vector3f ext_max_corner =
                    max_corner + offset_dir * margin;

                // points = self.convex_feature_points[parent_inside_convex_feature_ids]
                const MatF3 points =this->convex_feature_points(parent_ids, Eigen::all).eval();

                // local_inside_convex_feature_ids = np.where(mask)[0]
                const Eigen::VectorXi local_inside_ids =indices_in_bbox(points, ext_min_corner, ext_max_corner);

                if (local_inside_ids.size() > 0) {
                    node.inside_convex_feature_ids = parent_ids(local_inside_ids).eval();
                }
            }
        }

        keep_flags(i)= pre_check_node_build(node, spoints);
        if (keep_flags(i) && node.m_depth < m_max_depth) {
            node.subdivide();
            node.m_node_type = "interval";
        }
    }

    std::vector<std::reference_wrapper<OctreeNode>> new_nodes;
    new_nodes.reserve(keep_flags.count()*8);

    drop_nodes.clear();
    drop_nodes.reserve(old_leaf_count-keep_flags.count());
    for (std::size_t i = 0; i < old_leaf_count; ++i) {
        OctreeNode& node = m_leaf_nodes[i].get();

        if (keep_flags(i)) {
            if (node.m_depth < m_max_depth) {

                for (auto& child_ptr : node.m_children) {
                    new_nodes.emplace_back(*child_ptr);
                }
            } else {
                new_nodes.emplace_back(node);
            }
        } else {
            drop_nodes.emplace_back(node);
        }
    }

    m_leaf_nodes = std::move(new_nodes);
}

void Octree::compute_inside_surface_point_ids(
    const MatF3& spoints,
    OctreeNode& node)
{
    if (node.m_parent == nullptr) {
        return;
    }

    const Eigen::VectorXi& parent_ids =
        node.m_parent->inside_surface_point_ids;

    if (parent_ids.size() == 0) {
        return;
    }

    const Eigen::Vector3f min_corner =
        convert_grid_to_coords(node.m_min_corner_id);

    const Eigen::Vector3f max_corner =
        convert_grid_to_coords(node.m_max_corner_id);

    // points = spoints[parent_ids]
    const MatF3 points =
        spoints(parent_ids, Eigen::all).eval();

    // local ids inside this node bbox
    const Eigen::VectorXi local_inside_ids =
        indices_in_bbox(points, min_corner, max_corner);

    if (local_inside_ids.size() == 0) {
        node.inside_surface_point_ids.resize(0);
        return;
    }

    // map local ids back to global surface point ids
    node.inside_surface_point_ids=parent_ids(local_inside_ids).eval();
}
bool Octree::pre_check_node_build(OctreeNode& node, const MatF3& spoints)
{
    node.cell_sample_mask =ArrayB1::Constant(node.m_sample_grids.rows(), true);

    if (node.m_parent == nullptr) {
        return true;
    }

    const double box_edge_length=box_edge_length_for_depth[node.m_depth];

    const double diag_length=std::sqrt(3.0) * box_edge_length;

    const Eigen::Vector3i centroid_grid = node.m_centroid_id;

    const int centroid_gid = encode(centroid_grid);
    const int centroid_sid = m_gid2sid(centroid_gid);

    if (centroid_sid < 0 || centroid_sid >= m_sample_mesh_sdfs.size()) {
        throw std::runtime_error(
            "pre_check_node_build: invalid centroid sample id."
        );
    }

    const float centroid_sdf = m_sample_mesh_sdfs(centroid_sid);

    if (centroid_sdf > static_cast<float>(diag_length * 0.5) &&
        node.inside_convex_feature_ids.size() == 0)
    {
        return false;
    }

    const MatI3& sampling_grids = node.m_sample_grids;

    Eigen::VectorXi gids =
        ((sampling_grids.col(0).array() * m_capacity_per_dim +
          sampling_grids.col(1).array()) * m_capacity_per_dim +
          sampling_grids.col(2).array()).matrix();

    if ((gids.array() < 0).any() ||
        (gids.array() >= m_gid2sid.size()).any())
    {
        throw std::runtime_error(
            "pre_check_node_build: gids out of range."
        );
    }

    Eigen::VectorXi sids = m_gid2sid(gids).eval();

    if ((sids.array() < 0).any() ||
        (sids.array() >= m_sample_mesh_sdfs.size()).any())
    {
        throw std::runtime_error(
            "pre_check_node_build: invalid sample ids."
        );
    }

    Eigen::VectorXf origin_sdfs =
        m_sample_mesh_sdfs(sids).eval();

    if (node.m_depth < m_max_depth) {
        if (origin_sdfs.maxCoeff() > 0.0f) {
            compute_inside_surface_point_ids(spoints, node);
        }
    } else {
        if (origin_sdfs.minCoeff() > 0.0f) {
            compute_inside_surface_point_ids(spoints, node);
        }
    }

    if (node.inside_convex_feature_ids.size() > 0) {
        return true;
    }

    Eigen::VectorXi group_sids;
    Eigen::VectorXf sdfs;
    MatF3 sns;
    MatF3 coords;
    MatF3 nearest_points;
    Eigen::VectorXi shape_point_ids;
    ArrayB1 feature_flags;

    std::tie(
        group_sids,
        sdfs,
        sns,
        coords,
        nearest_points,
        shape_point_ids,
        feature_flags
    ) = get_group_data(node);

    if ((sns.array() != sns.array()).any()) {
        std::cout << "pre_check_node_build: sns contains NaN:\n"
                  << sns << std::endl;
    }

    if (coords.rows() > 1) {
        for (Eigen::Index i = 0; i < sns.rows(); ++i) {
            Eigen::Vector3f n = sns.row(i).transpose();
            n.stableNormalize();
            sns.row(i) = n.transpose();
        }

        const bool torch_feature =feature_flags.count() > 0;

        const bool need_normal_clustering =
            (!torch_feature &&
             node.m_depth < m_max_depth &&
             convex_feature_points.rows() > 0)
            ||
            (node.m_depth == m_max_depth &&
             convex_feature_points.rows() == 0);

        if (need_normal_clustering) {
            // constexpr double PI = 3.14159265358979323846;

            double radian_threshold =static_cast<double>(angle_threshold0) * M_PI / 180.0;
            if(convex_feature_points.size()==0){
                radian_threshold =static_cast<double>(angle_threshold) * M_PI / 180.0;
            }

            const double cosine_eps =1.0 - std::cos(radian_threshold);

            FastDBSCAN normal_clustering(
                cosine_eps,
                1,
                "cosine"
            );

            Eigen::VectorXi normal_labels0 =
                normal_clustering.fit_predict(sns);

            return has_more_than_one_label(normal_labels0);
        }

        return true;
    }

    return false;
}

void Octree::run_leaf_node_mat_search(
    const MatF3& spoints,
    const MatF3& sns,
    const ArrayB1& sp_feature_mask,
    const Eigen::VectorXi& patch_ids,
    const ArrayB1& sp_convex_feature_mask,
    double dihedral_angle_threshold,
    double min_dihedral_angle_degree)
{
    auto flatten_records_from_nodes =
        [&](const std::vector<std::reference_wrapper<OctreeNode>>& nodes)
            -> std::tuple<Eigen::VectorXi, Eigen::VectorXi, Eigen::VectorXi>
    {
        std::vector<int> cell_seed_nums_vec;
        std::vector<int> nearest_cloud_ids_flat_vec;
        std::vector<int> cell_node_ids_vec;

        for (auto& node_ref : nodes) {
            OctreeNode& node = node_ref.get();

            Eigen::VectorXi sids;
            Eigen::VectorXf sdfs;
            MatF3 node_sns;
            MatF3 coords;
            MatF3 proj_points;
            Eigen::VectorXi shape_point_ids;
            ArrayB1 feature_flags;

            std::tie(
                sids,
                sdfs,
                node_sns,
                coords,
                proj_points,
                shape_point_ids,
                feature_flags
            ) = get_group_data(node);

            cell_seed_nums_vec.push_back(static_cast<int>(shape_point_ids.size()));

            for (Eigen::Index i = 0; i < shape_point_ids.size(); ++i) {
                nearest_cloud_ids_flat_vec.push_back(shape_point_ids(i));
            }

            cell_node_ids_vec.push_back(node.id);
        }

        Eigen::VectorXi cell_seed_nums =
            Eigen::Map<const Eigen::VectorXi>(
                cell_seed_nums_vec.data(),
                static_cast<Eigen::Index>(cell_seed_nums_vec.size())
            ).eval();

        Eigen::VectorXi nearest_cloud_ids_flat =
            Eigen::Map<const Eigen::VectorXi>(
                nearest_cloud_ids_flat_vec.data(),
                static_cast<Eigen::Index>(nearest_cloud_ids_flat_vec.size())
            ).eval();

        Eigen::VectorXi cell_node_ids =
            Eigen::Map<const Eigen::VectorXi>(
                cell_node_ids_vec.data(),
                static_cast<Eigen::Index>(cell_node_ids_vec.size())
            ).eval();

        return std::make_tuple(
            cell_seed_nums,
            nearest_cloud_ids_flat,
            cell_node_ids
        );
    };

    PointCloudRVD<float> rvd(spoints, sns, sp_feature_mask);
    NanoKNNSearch<float> knn(spoints.data(),static_cast<int64_t>(spoints.rows()),static_cast<int64_t>(spoints.cols()), 16);

    if (convex_feature_points.rows() > 0) {
        rvd.buildKnnGraph(
            knn,
            26,
            true,
            180.0 - dihedral_angle_threshold - 1.0,
            std::min((180.0 - min_dihedral_angle_degree) / 2.0 + 0.1, 90.0),
            1
        );
    } else {
        rvd.buildKnnGraph(
            knn,
            8,
            true,
            180.0 - dihedral_angle_threshold - 1.0,
            std::min((180.0 - min_dihedral_angle_degree) / 2.0 + 0.1, 90.0),
            1
        );
    }

    rvd_graph_edges = rvd.extractUniqueUndirectedEdges();

    std::vector<std::reference_wrapper<OctreeNode>> direct_keep_nodes;
    std::vector<std::reference_wrapper<OctreeNode>> judge_nodes;

    direct_keep_nodes.reserve(m_leaf_nodes.size());
    judge_nodes.reserve(m_leaf_nodes.size());

    for (auto& leaf_ref : m_leaf_nodes) {
        OctreeNode& node = leaf_ref.get();

        const bool has_feature =
            node.inside_convex_feature_ids.size() > 0;

        if (has_feature) {
            node.decision_pid = -1;
            node.decision_sid = -1;
            node.groups.clear();
            direct_keep_nodes.emplace_back(node);
        } else {
            judge_nodes.emplace_back(node);
        }
    }

    if (judge_nodes.empty()) {
        m_leaf_nodes = std::move(direct_keep_nodes);
        node_id_map_spoint_clusters.clear();

        std::cout
            << "[run_leaf_node_mat_search] direct_keep=" << m_leaf_nodes.size()
            << ", judged=0, final_kept=0, final_dropped=0"
            << std::endl;

        return;
    }

    Eigen::VectorXi cellSeedNums;
    Eigen::VectorXi nearestCloudIdsFlat;
    Eigen::VectorXi cellNodeIds;

    std::tie(
        cellSeedNums,
        nearestCloudIdsFlat,
        cellNodeIds
    ) = flatten_records_from_nodes(judge_nodes);

    std::vector<std::reference_wrapper<OctreeNode>> final_kept_judge_nodes;
    std::vector<std::reference_wrapper<OctreeNode>> final_dropped_judge_nodes;

    final_kept_judge_nodes.reserve(judge_nodes.size());
    final_dropped_judge_nodes.reserve(judge_nodes.size());

    if (cellNodeIds.size() > 0) {
        auto projection_clusters =
            rvd.computeProjectionClustersNoLabelOneRingExpandedInSamePatch(
                cellSeedNums,
                nearestCloudIdsFlat,
                patch_ids
            );

        node_id_map_cell_sample_clusters.clear();
        node_id_map_spoint_clusters.clear();
        const int judge_count = static_cast<int>(judge_nodes.size());
        for (int judge_i = 0; judge_i < judge_count; ++judge_i) {
            OctreeNode& node =judge_nodes[static_cast<std::size_t>(judge_i)].get();
            const int node_id = node.id;
            std::vector<std::vector<int>>& clusters= projection_clusters[static_cast<std::size_t>(judge_i)];
            node.groups=clusters;
            node_id_map_cell_sample_clusters[node_id] =clusters; 

            Eigen::VectorXi shape_point_ids;
            std::tie(
                std::ignore,
                std::ignore,
                std::ignore,
                std::ignore,
                std::ignore,
                shape_point_ids,
                std::ignore
            ) = get_group_data(node);
            node_id_map_spoint_clusters[node_id] =clusters_to_spoint_ids(clusters,shape_point_ids);
        }

        ArrayB1 non_empty_flags= ArrayB1::Constant(judge_count, false);
        #pragma omp parallel for schedule(dynamic)
        for (int judge_i = 0; judge_i < judge_count; ++judge_i) {
            OctreeNode& node =judge_nodes[static_cast<std::size_t>(judge_i)].get();
            const int node_id = node.id;
            std::vector<std::vector<int>>& clusters = node.groups;

            Eigen::VectorXi group_sids;
            Eigen::VectorXf sdfs;
            MatF3 node_sns;
            MatF3 coords;
            MatF3 proj_points;
            Eigen::VectorXi shape_point_ids;
            ArrayB1 feature_flags;

            std::tie(
                group_sids,
                sdfs,
                node_sns,
                coords,
                proj_points,
                shape_point_ids,
                feature_flags
            ) = get_group_data(node);

            Eigen::VectorXi node_patch_ids = patch_ids(shape_point_ids).eval();

            ArrayB1 valid_patch_mask = (node_patch_ids.array() >= 0);

            ArrayB1 feature_patch_mask =(node_patch_ids.array() == -1);

            Eigen::VectorXi valid_patch_ids =  node_patch_ids(indices_where_true(valid_patch_mask)).eval();

            Eigen::VectorXi feature_patch_ids = node_patch_ids(indices_where_true(feature_patch_mask)).eval();

            if (valid_patch_ids.size() == 0) {
                //constexpr double PI = 3.14159265358979323846;
                const double radian_threshold = angle_threshold0 * M_PI / 180.0;
                const double cosine_eps = 1.0 - std::cos(radian_threshold);

                FastDBSCAN normal_clustering(cosine_eps, 1, "cosine");
                Eigen::VectorXi normal_labels0 =
                    normal_clustering.fit_predict(node_sns);

                if (unique_label_count(normal_labels0) > 5 ||
                    node.groups.size() > 4)
                {
                    auto ret = is_cylindrical_tinf_node(node_sns, true);
                    node.is_cylinder = ret.is_cylindrical_tinf;
                }
            } else {
                std::map<int, int> patch_count_map;
                for (Eigen::Index i = 0; i < valid_patch_ids.size(); ++i) {
                    patch_count_map[valid_patch_ids(i)]++;
                }
                const int unique_patch_count =static_cast<int>(patch_count_map.size());

                if (unique_patch_count < 3 && unique_patch_count > 0) {
                    int dominant_patch_id = -1;
                    int dominant_count = -1;

                    for (const auto& kv : patch_count_map) {
                        if (kv.second > dominant_count) {
                            dominant_patch_id = kv.first;
                            dominant_count = kv.second;
                        }
                    }

                    ArrayB1 dominant_mask = (node_patch_ids.array() == dominant_patch_id) || (node_patch_ids.array() == -1);

                    if (unique_patch_count == 1 &&
                        feature_patch_ids.size() > valid_patch_ids.size())
                    {
                        dominant_mask = (node_patch_ids.array() == -1);
                    }

                    Eigen::VectorXi dominant_ids = indices_where_true(dominant_mask);

                    MatF3 dominant_sns = node_sns(dominant_ids, Eigen::all).eval();

                    if (dominant_sns.rows() >= 4) {
                        //constexpr double PI = 3.14159265358979323846;
                        const double radian_threshold = angle_threshold0 * M_PI / 180.0;
                        const double cosine_eps = 1.0 - std::cos(radian_threshold);
                        FastDBSCAN normal_clustering(cosine_eps, 1, "cosine");
                        Eigen::VectorXi normal_labels0 = normal_clustering.fit_predict(dominant_sns);

                        if (unique_label_count(normal_labels0) > 5 ||
                            node.groups.size() > 4)
                        {
                            auto ret = is_cylindrical_tinf_node(
                                dominant_sns,
                                feature_patch_ids.size() > 0
                            );
                            node.is_cylinder = ret.is_cylindrical_tinf;
                        }
                    }
                }
            }//if else end


    
            if (clusters.size() > 1 || node.is_cylinder) {
                //final_kept_judge_nodes.emplace_back(node);
                non_empty_flags[judge_i]=true;
            } else {
                if (feature_patch_ids.size() > 0 &&valid_patch_ids.size() > 0)
                {
                    ArrayB1 sp_convex_flags = sp_convex_feature_mask(shape_point_ids).eval();

                    Eigen::VectorXf adjusted_sdfs = sdfs;

                    for (Eigen::Index i = 0; i < adjusted_sdfs.size(); ++i) {
                        if (sp_convex_flags(i)) {
                            adjusted_sdfs(i) = std::abs(adjusted_sdfs(i));
                        }
                    }

                    ArrayB1 negative_mask =(adjusted_sdfs.array() < 0.0f);

                    if (negative_mask.count() > 1) {
                        Eigen::VectorXi ids = indices_where_true(negative_mask);

                        // constexpr double PI = 3.14159265358979323846;
                        const double radian_threshold =
                            std::min(
                                (180.0 - min_dihedral_angle_degree) / 2.0 + 0.1,
                                90.0
                            ) * M_PI / 180.0;

                        const double cosine_eps =
                            1.0 - std::cos(radian_threshold);

                        MatF3 selected_sns = node_sns(ids, Eigen::all).eval();

                        FastDBSCAN normal_clustering(cosine_eps, 1, "cosine");
                        Eigen::VectorXi normal_labels0 =normal_clustering.fit_predict(selected_sns);

                        if (unique_label_count(normal_labels0) > 1) {
                            clusters =clusters_from_labels_and_ids(normal_labels0, ids);

                            node.groups = clusters;
                            #pragma omp critical(node_id_map_cell_sample_clusters_update)
                            {
                                node_id_map_cell_sample_clusters[node_id] =clusters;
                            }
                            //node_id_map_spoint_clusters[node_id] = clusters_to_spoint_ids(clusters, shape_point_ids);
                            auto spoint_id_clusters =clusters_to_spoint_ids(clusters,shape_point_ids);
                            #pragma omp critical(node_id_map_spoint_clusters_update)
                            {   
                                node_id_map_spoint_clusters[node_id] =spoint_id_clusters;
                            }    

                            //final_kept_judge_nodes.emplace_back(node);
                            non_empty_flags[judge_i]=true;
                            continue;
                        }
                    }
                }

                node.drop_stage = 2;
                //final_dropped_judge_nodes.emplace_back(node);
            }
        }
        for(int judge_i = 0; judge_i < judge_count; ++judge_i)
        {
            OctreeNode& node =judge_nodes[static_cast<std::size_t>(judge_i)].get();
            if(non_empty_flags(judge_i)){
                final_kept_judge_nodes.emplace_back(node);
            }else{
                final_dropped_judge_nodes.emplace_back(node);
            }
        }
    } else {
        node_id_map_spoint_clusters.clear();

        for (auto& node_ref : judge_nodes) {
            OctreeNode& node = node_ref.get();
            node.groups.clear();
            node.drop_stage = 2;
            final_dropped_judge_nodes.emplace_back(node);
        }
    }

    drop_nodes = std::move(final_dropped_judge_nodes);

    std::vector<std::reference_wrapper<OctreeNode>> new_leaf_nodes;
    new_leaf_nodes.reserve(
        direct_keep_nodes.size() + final_kept_judge_nodes.size()
    );

    for (auto& node_ref : direct_keep_nodes) {
        new_leaf_nodes.emplace_back(node_ref.get());
    }

    for (auto& node_ref : final_kept_judge_nodes) {
        new_leaf_nodes.emplace_back(node_ref.get());
    }

    m_leaf_nodes = std::move(new_leaf_nodes);


    std::cout
        << "[run_leaf_node_mat_search] direct_keep="
        << direct_keep_nodes.size()
        << ", judged=" << judge_nodes.size()
        << ", final_kept=" << final_kept_judge_nodes.size()
        << ", final_dropped=" << drop_nodes.size()
        << std::endl;
}



void Octree::batch_pre_solve_sqef(
    const MatF3& spoints,
    const ArrayB1& sp_convex_feature_mask,
    SDFProjectorFunc& cloud_sdf_grad_mask_func,
    const MatF3& centroids_coords,
    float singular_value_threshold,
    float corner_factor,
    float edge_factor,
    float plane_factor)
{
    const Eigen::Index leaf_count =
        static_cast<Eigen::Index>(m_leaf_nodes.size());

    if (centroids_coords.rows() != leaf_count || centroids_coords.cols() != 3) {
        throw std::invalid_argument(
            "batch_pre_solve_sqef: centroids_coords must have shape (num_leaf_nodes, 3)."
        );
    }
    
    center_kdtree = std::make_unique<NanoKNNSearch<float>>(
        centroids_coords.data(),
        static_cast<int64_t>(centroids_coords.rows()),
        static_cast<int64_t>(centroids_coords.cols()),
        1
    );
    sqefs.clear();
    sqefs.resize(static_cast<std::size_t>(leaf_count));

    results.clear();
    results.resize(static_cast<std::size_t>(leaf_count));

    feature_flags =
        ArrayB1::Constant(leaf_count, false);

    m_mesh_v.clear();
    m_mesh_v_radii.clear();
    m_mesh_v_simulated.clear();
    m_v_type.clear();

    cell_samples.clear();
    mesh_samples.clear();
    cell_samples.resize(static_cast<std::size_t>(leaf_count));
    mesh_samples.resize(static_cast<std::size_t>(leaf_count));

    precompute_node_data(centroids_coords);

    const bool has_feature =
        convex_feature_points.rows() > 0;
    //constexpr std::size_t parallel_threshold = 4096;

    #pragma omp parallel for schedule(dynamic)
    for (int ii = 0; ii < static_cast<int>(leaf_count); ++ii) {
        const Eigen::Index i = static_cast<Eigen::Index>(ii);

        OctreeNode& node =
            m_leaf_nodes[static_cast<std::size_t>(i)].get();

        const Eigen::Vector3f min_corner =
            bounding_boxes[static_cast<std::size_t>(i)].first;

        const Eigen::Vector3f max_corner =
            bounding_boxes[static_cast<std::size_t>(i)].second;

        const Eigen::Vector3f center =
            centroids_coords.row(i).transpose();

        // ============================================================
        // 1. corner branch:
        // if len(self.node_contain_corner_ids[i]) > 0
        // ============================================================
        const Eigen::VectorXi& contain_corner_ids =
            node_contain_corner_ids[static_cast<std::size_t>(i)];

        const Eigen::VectorXi& strict_contain_corner_ids =
            strict_node_contain_corner_ids[static_cast<std::size_t>(i)];

        if (contain_corner_ids.size() > 0) {
            feature_flags(i) = true;

            const bool strict_inside =
                strict_contain_corner_ids.size() > 0;

            const Eigen::VectorXi& corner_ids =
                strict_inside ? strict_contain_corner_ids : contain_corner_ids;

            const Eigen::Vector3f corner_point =
                convex_corner_points.row(corner_ids(0)).transpose();

            results[static_cast<std::size_t>(i)] =
                make_dual_point(
                    corner_point,
                    DualPointType::CORNER,
                    strict_inside
                );

            node.m_reconstruct_point =
                results[static_cast<std::size_t>(i)].m_point;

            cell_samples[static_cast<std::size_t>(i)] = empty_matf3();
            mesh_samples[static_cast<std::size_t>(i)] = empty_matf3();

            continue;
        }

        // ============================================================
        // 2. feature edge branch:
        // elif len(node.inside_convex_feature_ids)>0
        //      and self.adj_contain_outer_corner_flags[i] == False
        // ============================================================
        if (node.inside_convex_feature_ids.size() > 0 &&
            !adj_contain_outer_corner_flags(i))
        {
            feature_flags(i) = true;

            const bool strict_inside =
                node.strict_inside_convex_feature_ids.size() > 0;

            const Eigen::VectorXi& feature_ids =
                strict_inside
                    ? node.strict_inside_convex_feature_ids
                    : node.inside_convex_feature_ids;

            const MatF3 feature_points =
                rows_by_ids(convex_feature_points, feature_ids);

            const Eigen::Vector3f nearest_point =
                nearest_point_to_center(feature_points, center);

            results[static_cast<std::size_t>(i)] =
                make_dual_point(
                    nearest_point,
                    DualPointType::EDGE,
                    strict_inside
                );

            node.m_reconstruct_point =
                results[static_cast<std::size_t>(i)].m_point;

            cell_samples[static_cast<std::size_t>(i)] = empty_matf3();
            mesh_samples[static_cast<std::size_t>(i)] = empty_matf3();

            continue;
        }

        // ============================================================
        // 3. feature edge + adjacent outer corner branch
        // ============================================================
        if (node.inside_convex_feature_ids.size() > 0 &&
            adj_contain_outer_corner_flags(i))
        {
            feature_flags(i) = true;

            const int corner_node_id =
                adj_corner_node_id(i);

            if (corner_node_id >= 0 &&
                corner_node_id < leaf_count &&
                adj_nearest_leaf_node_indices(corner_node_id) == ii)
            {
                const Eigen::VectorXi& corner_ids =
                    node_contain_corner_ids[
                        static_cast<std::size_t>(corner_node_id)
                    ];

                const MatF3 corner_points =
                    rows_by_ids(convex_corner_points, corner_ids);

                const Eigen::Vector3f best_v =
                    corner_points.colwise().mean().transpose();

                results[static_cast<std::size_t>(i)] =
                    make_dual_point(
                        best_v,
                        DualPointType::CORNER,
                        false
                    );

                node.m_reconstruct_point =
                    results[static_cast<std::size_t>(i)].m_point;

                cell_samples[static_cast<std::size_t>(i)] = empty_matf3();
                mesh_samples[static_cast<std::size_t>(i)] = empty_matf3();

                continue;
            }

            const bool strict_inside =
                node.strict_inside_convex_feature_ids.size() > 0;

            const Eigen::VectorXi& feature_ids =
                strict_inside
                    ? node.strict_inside_convex_feature_ids
                    : node.inside_convex_feature_ids;

            const MatF3 feature_points =
                rows_by_ids(convex_feature_points, feature_ids);

            const Eigen::Vector3f nearest_point =
                nearest_point_to_center(feature_points, center);

            results[static_cast<std::size_t>(i)] =
                make_dual_point(
                    nearest_point,
                    DualPointType::EDGE,
                    strict_inside
                );

            node.m_reconstruct_point =
                results[static_cast<std::size_t>(i)].m_point;

            cell_samples[static_cast<std::size_t>(i)] = empty_matf3();
            mesh_samples[static_cast<std::size_t>(i)] = empty_matf3();

            continue;
        }

        // ============================================================
        // 4. common SQEF branch
        // ============================================================
        Eigen::VectorXi sids;
        Eigen::VectorXf sdfs;
        MatF3 node_sns;
        MatF3 coords;
        MatF3 proj_points;
        Eigen::VectorXi shape_point_ids;
        ArrayB1 local_feature_flags;

        std::tie(
            sids,
            sdfs,
            node_sns,
            coords,
            proj_points,
            shape_point_ids,
            local_feature_flags
        ) = get_group_data(node);

        MatF3 nearest_points =
            rows_by_ids(spoints, shape_point_ids);

        if (coords.rows() > 1) {
            std::vector<std::vector<int>> groups =
                node.groups;

            const bool torch_feature =
                local_feature_flags.count() > 0;

            // Python:
            // if sp_convex_feature_mask is not None:
            //     sdfs[sp_convex_flags] = abs(...)
            if (sp_convex_feature_mask.size() > 0 &&
                shape_point_ids.size() > 0)
            {
                const ArrayB1 sp_convex_flags =
                    sp_convex_feature_mask(shape_point_ids).eval();

                for (Eigen::Index k = 0; k < sdfs.size(); ++k) {
                    if (sp_convex_flags(k)) {
                        sdfs(k) = std::abs(sdfs(k));
                    }
                }
            }

            // Python: if np.sum(sdfs > 0) > 0: filter groups by negative sdfs
            if ((sdfs.array() > 0.0f).count() > 0) {
                std::vector<std::vector<int>> new_groups =
                    filter_groups_by_negative_sdf(groups, sdfs);

                if (new_groups.size() > 1) {
                    groups = std::move(new_groups);
                }
            }

            // Python feature/non-feature normal clustering branch
            if (has_feature) {
                if (!node.is_cylinder && !torch_feature) {
                    const double cosine_eps =
                        1.0 - std::cos(
                            static_cast<double>(angle_threshold) * M_PI / 180.0
                        );

                    FastDBSCAN normal_clustering(cosine_eps, 1, "cosine");

                    const Eigen::VectorXi normal_labels =normal_clustering.fit_predict(node_sns);

                    const int uniq_count =unique_label_count(normal_labels);

                    if (uniq_count >= 2 &&uniq_count < static_cast<int>(groups.size()))
                    {
                        groups = clusters_from_labels(normal_labels);
                    }
                }
            } else {
                if (!node.is_cylinder) {
                    const double cosine_eps =
                        1.0 - std::cos(
                            static_cast<double>(angle_threshold) * M_PI / 180.0
                        );

                    FastDBSCAN normal_clustering(cosine_eps, 1, "cosine");

                    const Eigen::VectorXi normal_labels =normal_clustering.fit_predict(node_sns);

                    const int uniq_count =unique_label_count(normal_labels);

                    if (uniq_count >= 2 &&uniq_count < static_cast<int>(groups.size()))
                    {
                        groups = clusters_from_labels(normal_labels);
                    }
                }
            }

            // Python:
            // if len(coords)>4 and not node.is_cylinder and (len(groups)==2 or len(groups)==3)
            if (coords.rows() > 4 &&
                !node.is_cylinder &&
                (groups.size() == 2 || groups.size() == 3))
            {
                //Eigen::VectorXi central_indices;
                std::vector<int> central_indices_Vec;
                std::vector<std::vector<int>> new_groups;
                
                std::tie(central_indices_Vec, new_groups) =find_central_indices(nearest_points, groups, 1);
                Eigen::VectorXi central_indices=Eigen::Map<Eigen::VectorXi>(central_indices_Vec.data(), central_indices_Vec.size());
                sdfs =vector_by_ids(sdfs, central_indices);

                node_sns =rows_by_ids(node_sns, central_indices);

                coords =rows_by_ids(coords, central_indices);

                proj_points =rows_by_ids(proj_points, central_indices);

                nearest_points =rows_by_ids(nearest_points, central_indices);

                local_feature_flags =arrayb_by_ids(local_feature_flags, central_indices);

                groups =std::move(new_groups);
            }

            MatF3 normals = compute_sqef_normals(nearest_points, coords, sdfs);

            if (has_feature) {
                for (Eigen::Index k = 0; k < normals.rows(); ++k) {
                    if (!local_feature_flags(k)) {
                        normals.row(k) = node_sns.row(k);
                    }
                }
            }

            sqefs[static_cast<std::size_t>(i)] =SQEF(normals, nearest_points, coords);

            SQEF& sqef = sqefs[static_cast<std::size_t>(i)].value();

            std::vector<std::pair<Eigen::Vector3f, Eigen::Vector3f>>
                one_ring_boxes =
                    collect_one_ring_boxes(
                        m_leaf_nodes_adjacency,
                        bounding_boxes,
                        i
                    );

            std::vector<std::pair<Eigen::Vector3f, Eigen::Vector3f>>
                two_ring_boxes =
                    collect_two_ring_boxes(
                        two_ring_adjacency,
                        bounding_boxes,
                        i
                    );

            results[static_cast<std::size_t>(i)] =
                solve_point_pre_sqef_with_group(
                    sqef,
                    min_corner,
                    max_corner,
                    groups,
                    node.is_cylinder,
                    singular_value_threshold,
                    one_ring_boxes,
                    two_ring_boxes,
                    corner_factor,
                    edge_factor,
                    plane_factor
                );

            DualPoint& result =
                results[static_cast<std::size_t>(i)];

            // Python:
            // if result.m_point is not None and result.radius < 0
            if (result.m_point.has_value() &&
                result.radius < 0.0f)
            {
                result.m_is_inside = false;
                node.m_reconstruct_point = result.m_point;
                result.m_point = std::nullopt;
            }

            // Python:
            // if not result.m_is_inside: node.is_cylinder = False
            if (!result.m_is_inside) {
                node.is_cylinder = false;
            }

            // Python:
            // query sdf of result point; if sdf[0] > 5e-3 invalidate
            if (result.m_is_inside &&
                result.m_point.has_value())
            {
                MatF3 query_point(1, 3);
                query_point.row(0) =
                    result.m_point.value().transpose();

                Eigen::VectorXf query_sdfs;
                MatF3 query_sns;
                MatF3 query_projs;
                Eigen::VectorXi query_ids;
                ArrayB1 query_masks;

                #pragma omp critical(sdf_query_projector)
                {
                    std::tie(
                        query_sdfs,
                        query_sns,
                        query_projs,
                        query_ids,
                        query_masks
                    ) = cloud_sdf_grad_mask_func(query_point);
                }

                if (query_sdfs.size() > 0 &&
                    query_sdfs(0) > 5e-3f)
                {
                    result.m_is_inside = false;
                    node.m_reconstruct_point = result.m_point;
                    result.m_point = std::nullopt;
                }
            }

            node.m_reconstruct_point =
                result.m_point;
        }

        // Python: self.cell_samples.append(coords)
        //         self.mesh_samples.append(nearest_points)
        cell_samples[static_cast<std::size_t>(i)] =
            coords;

        mesh_samples[static_cast<std::size_t>(i)] =
            nearest_points;
    }
}

void Octree::batch_locate()
{
    m_mesh_v.clear();
    m_mesh_v_radii.clear();
    m_mesh_v_simulated.clear();
    m_v_type.clear();
    m_mesh_v_to_leaf_node_indexes.clear();

    vertex_to_leaf_cell.clear();
    vertex_to_leaf_cell_inverse.clear();

    cylinder_node_ids.clear();

    const Eigen::Index leaf_count =
        static_cast<Eigen::Index>(m_leaf_nodes.size());

    if (results.size() != static_cast<std::size_t>(leaf_count)) {
        throw std::runtime_error(
            "batch_locate: results size does not match leaf node count."
        );
    }

    if (sqefs.size() != static_cast<std::size_t>(leaf_count)) {
        throw std::runtime_error(
            "batch_locate: sqefs size does not match leaf node count."
        );
    }

    if (!center_kdtree) {
        throw std::runtime_error(
            "batch_locate: center_kdtree is null. "
            "Call batch_pre_solve_sqef first."
        );
    }

    leaf_link_nodes.clear();
    leaf_link_nodes.resize(static_cast<std::size_t>(leaf_count));

    const int scale = 1 << m_sampling_depth;

    for (Eigen::Index leaf_node_index = 0;
         leaf_node_index < leaf_count;
         ++leaf_node_index)
    {
        OctreeNode& node =
            m_leaf_nodes[static_cast<std::size_t>(leaf_node_index)].get();

        DualPoint& result =
            results[static_cast<std::size_t>(leaf_node_index)];

        ListNode& link_node =
            leaf_link_nodes[static_cast<std::size_t>(leaf_node_index)];

        // self.leaf_link_nodes[leaf_node_index].val = leaf_node_index
        // self.leaf_link_nodes[leaf_node_index].next = None
        link_node.val = static_cast<int>(leaf_node_index);
        link_node.next = nullptr;

        // ============================================================
        // if result.m_is_inside and result.m_point is not None:
        // ============================================================
        if (result.m_is_inside && result.m_point.has_value()) {
            // add cylinder node
            if (node.is_cylinder) {
                cylinder_node_ids.push_back(
                    static_cast<int>(leaf_node_index)
                );
            } else {
                const int node_pos_type =
                    m_leaf_node_pos_types(leaf_node_index);

                const int x_bits = node_pos_type & 0b000011;
                const int y_bits = node_pos_type & 0b001100;
                const int z_bits = node_pos_type & 0b110000;

                int adj_node_id0 = -1;
                int adj_node_id1 = -1;

                if (x_bits == 0b000011) {
                    adj_node_id0 =m_leaf_nodes_adjacency(leaf_node_index, 0);
                    adj_node_id1 =m_leaf_nodes_adjacency(leaf_node_index, 1);
                } else if (y_bits == 0b001100) {
                    adj_node_id0 =m_leaf_nodes_adjacency(leaf_node_index, 2);
                    adj_node_id1 =m_leaf_nodes_adjacency(leaf_node_index, 3);
                } else if (z_bits == 0b110000) {
                    adj_node_id0 =m_leaf_nodes_adjacency(leaf_node_index, 4);
                    adj_node_id1 =m_leaf_nodes_adjacency(leaf_node_index, 5);
                }

                if (adj_node_id0 != -1 && adj_node_id1 != -1) {
                    if (adj_node_id0 >= 0 &&
                        adj_node_id0 < leaf_count &&
                        adj_node_id1 >= 0 &&
                        adj_node_id1 < leaf_count)
                    {
                        const OctreeNode& adj_node0 =
                            m_leaf_nodes[
                                static_cast<std::size_t>(adj_node_id0)
                            ].get();

                        const OctreeNode& adj_node1 =
                            m_leaf_nodes[
                                static_cast<std::size_t>(adj_node_id1)
                            ].get();

                        if (adj_node0.is_cylinder &&
                            adj_node1.is_cylinder)
                        {
                            node.is_cylinder = true;
                            cylinder_node_ids.push_back(
                                static_cast<int>(leaf_node_index)
                            );
                        }
                    }
                }
            }

            node.m_node_type = "leaf";
            node.m_vertex_idx =static_cast<int>(m_mesh_v.size());

            m_mesh_v.push_back(result.m_point.value());
            m_mesh_v_radii.push_back(result.radius);
            m_v_type.push_back(static_cast<int>(result.m_type));
            m_mesh_v_to_leaf_node_indexes.push_back(
                static_cast<int>(leaf_node_index)
            );

            vertex_to_leaf_cell[node.m_vertex_idx] =static_cast<int>(leaf_node_index);

            vertex_to_leaf_cell_inverse[static_cast<int>(leaf_node_index)] = node.m_vertex_idx;

            continue;
        }

        // ============================================================
        // elif result.m_point is not None:
        // ============================================================
        if (result.m_point.has_value()) {
            int nearest_leaf_index = -1;
            float dist2 = 0.0f;

            center_kdtree->searchKNN1(
                nearest_leaf_index,
                dist2,
                result.m_point.value().data()
            );

            bool check = true;

            if (nearest_leaf_index < 0 ||
                nearest_leaf_index >= leaf_count)
            {
                check = false;
            } else {
                const Eigen::Vector3i cur_coord =
                    node.m_min_corner_id / scale;

                const OctreeNode& nearest_node =m_leaf_nodes[static_cast<std::size_t>(nearest_leaf_index)].get();

                const Eigen::Vector3i nearest_coord =nearest_node.m_min_corner_id / scale;

                const int dx =std::abs(cur_coord(0) - nearest_coord(0));
                const int dy =std::abs(cur_coord(1) - nearest_coord(1));
                const int dz =std::abs(cur_coord(2) - nearest_coord(2));

                if (dx > 1 || dy > 1 || dz > 1) {
                    check = false;
                }


                //
                // const bool has_sqef =
                //     sqefs[static_cast<std::size_t>(leaf_node_index)].has_value();
                //
                // if (has_sqef) {
                //     const SQEF& sqef =
                //         sqefs[static_cast<std::size_t>(leaf_node_index)].value();
                //     ...
                // }
            }

            if (!check) {
                link_node.next = nullptr;
                continue;
            }

            const OctreeNode& target_node =m_leaf_nodes[static_cast<std::size_t>(nearest_leaf_index)].get();

            const Eigen::Vector3f min_corner =
                convert_grid_to_coords(target_node.m_min_corner_id);

            const Eigen::Vector3f max_corner =
                convert_grid_to_coords(target_node.m_max_corner_id);

            if (is_inside(
                    result.m_point.value(),
                    min_corner,
                    max_corner))
            {
                link_node.next =&leaf_link_nodes[static_cast<std::size_t>(nearest_leaf_index)];
            } else {
                link_node.next = nullptr;
            }

            continue;
        }

        // ============================================================
        // else:
        // result.m_point is None
        // ============================================================
    }
}

void Octree::dfs_relate()
{
    const int leaf_count = static_cast<int>(m_leaf_nodes.size());

    if (results.size() != static_cast<std::size_t>(leaf_count)) {
        throw std::runtime_error(
            "dfs_relate: results size does not match leaf node count."
        );
    }

    if (leaf_link_nodes.size() != static_cast<std::size_t>(leaf_count)) {
        throw std::runtime_error(
            "dfs_relate: leaf_link_nodes size does not match leaf node count."
        );
    }

    auto add_mesh_vertex = [&](int leaf_node_index,
                               const Eigen::Vector3f& point,
                               float radius,
                               int v_type) -> int
    {
        OctreeNode& node =m_leaf_nodes[static_cast<std::size_t>(leaf_node_index)].get();

        const int vertex_id = static_cast<int>(m_mesh_v.size());

        node.m_vertex_idx = vertex_id;

        m_mesh_v.push_back(point);
        m_mesh_v_radii.push_back(radius);
        m_v_type.push_back(v_type);
        m_mesh_v_to_leaf_node_indexes.push_back(leaf_node_index);

        vertex_to_leaf_cell[vertex_id] = leaf_node_index;
        vertex_to_leaf_cell_inverse[leaf_node_index] = vertex_id;

        return vertex_id;
    };

    for (int leaf_node_index = 0; leaf_node_index < leaf_count; ++leaf_node_index) {
        DualPoint& result =results[static_cast<std::size_t>(leaf_node_index)];

        OctreeNode& node =m_leaf_nodes[static_cast<std::size_t>(leaf_node_index)].get();

        // ============================================================
        // if result.m_is_inside and result.m_point is not None:
        // ============================================================
        if (result.m_is_inside && result.m_point.has_value()) {
            node.m_node_cell_type = 2;
            continue;
        }

        // ============================================================
        // elif result.m_point is not None and leaf_link_nodes[i].next is not None:
        // ============================================================
        if (result.m_point.has_value() &&
            leaf_link_nodes[static_cast<std::size_t>(leaf_node_index)].next != nullptr)
        {
            std::vector<char> visited(static_cast<std::size_t>(leaf_count), 0);

            ListNode* cur_list_node =&leaf_link_nodes[static_cast<std::size_t>(leaf_node_index)];

            while (cur_list_node->next != nullptr &&
                   cur_list_node->val >= 0 &&
                   cur_list_node->val < leaf_count &&
                   visited[static_cast<std::size_t>(cur_list_node->val)] == 0)
            {
                visited[static_cast<std::size_t>(cur_list_node->val)] = 1;
                cur_list_node = cur_list_node->next;
            }

            const int cur_index = cur_list_node->val;

            if (cur_index < 0 || cur_index >= leaf_count) {
                node.m_node_cell_type = 0;
                node.m_node_type = "empty";
                node.m_vertex_idx = -1;
                continue;
            }

            DualPoint& cur_result =results[static_cast<std::size_t>(cur_index)];

            OctreeNode& cur_node =m_leaf_nodes[static_cast<std::size_t>(cur_index)].get();

            // ------------------------------------------------------------
            // if self.results[cur_index].m_is_inside:
            // ------------------------------------------------------------
            if (cur_result.m_is_inside) {
                node.m_node_type = "leaf";

                const std::optional<Eigen::Vector3f> target =cur_result.m_point;

                if (cur_result.m_point.has_value()) {
                    if (static_cast<int>(cur_result.m_type) > static_cast<int>(result.m_type) &&
                        static_cast<int>(result.m_type) < static_cast<int>(DualPointType::PLANE))
                    {
                        // target_node = self.m_leaf_nodes[cur_index]
                        // self.results[cur_index].m_point = result.m_point
                        // self.m_mesh_v[target_node.m_vertex_idx] = result.m_point
                        // self.m_mesh_v_radii[target_node.m_vertex_idx] = result.radius
                        cur_result.m_point = result.m_point;

                        if (cur_node.m_vertex_idx >= 0 &&
                            cur_node.m_vertex_idx < static_cast<int>(m_mesh_v.size()))
                        {
                            m_mesh_v[static_cast<std::size_t>(cur_node.m_vertex_idx)] =
                                result.m_point.value();

                            m_mesh_v_radii[static_cast<std::size_t>(cur_node.m_vertex_idx)] =
                                result.radius;
                        }
                    }
                    else if (static_cast<int>(cur_result.m_type) <= static_cast<int>(result.m_type)) {
                        // result.m_point = target
                        // result.radius = self.results[cur_index].radius
                        result.m_point = target;
                        result.radius = cur_result.radius;
                    }
                }

                node.m_node_cell_type = 1; // virtual

                // add virtual mesh vertex for current node
                add_mesh_vertex(
                    leaf_node_index,
                    result.m_point.value(),
                    result.radius,
                    static_cast<int>(result.m_type)
                );

                node.quote_leaf_node_index = cur_index;
                node.quote_vertex_id = cur_node.m_vertex_idx;

                cur_node.be_quoted_leaf_node_indexes.push_back(node.leaf_node_index);
                cur_node.be_quoted_vertex_ids.push_back(node.m_vertex_idx);
            }
            // ------------------------------------------------------------
            // else: target chain ends at a non-inside result
            // ------------------------------------------------------------
            else {
                node.m_node_type = "leaf";
                node.m_node_cell_type = 1; // virtual

                // add data for node of cur_index
                cur_result.m_point = result.m_point;
                cur_node.m_vertex_idx = static_cast<int>(m_mesh_v.size());

                vertex_to_leaf_cell[cur_node.m_vertex_idx] = cur_index;
                vertex_to_leaf_cell_inverse[cur_index] = cur_node.m_vertex_idx;

                m_mesh_v.push_back(result.m_point.value());
                m_mesh_v_radii.push_back(result.radius);
                m_v_type.push_back(static_cast<int>(result.m_type));
                m_mesh_v_to_leaf_node_indexes.push_back(cur_index);

                cur_node.m_reconstruct_point = result.m_point;
                cur_node.m_node_cell_type = 2;
                cur_node.m_node_type = "leaf";
                cur_result.m_is_inside = true;

                // add virtual mesh vertex for current node
                add_mesh_vertex(
                    leaf_node_index,
                    result.m_point.value(),
                    result.radius,
                    static_cast<int>(result.m_type)
                );

                node.quote_leaf_node_index = cur_index;
                node.quote_vertex_id = cur_node.m_vertex_idx;

                cur_node.be_quoted_leaf_node_indexes.push_back(node.leaf_node_index);
                cur_node.be_quoted_vertex_ids.push_back(node.m_vertex_idx);
            }

            // add cylinder node
            if (node.is_cylinder) {
                cylinder_node_ids.push_back(leaf_node_index);
            }

            continue;
        }

        // ============================================================
        // else:
        // ============================================================
        node.m_node_cell_type = 0;
        node.m_node_type = "empty";
        node.m_vertex_idx = -1;
    }

    // Python:
    // self.m_mesh_v_to_leaf_node_indexes = np.array(...)
    // C++ already keeps std::vector<int>.

    // ================================================================
    // quote_leaf_node_indexes / be_quoted_leaf_node_indexes
    // ================================================================
    quote_leaf_node_indexes.assign(static_cast<std::size_t>(leaf_count), -1);
    be_quoted_leaf_node_indexes.clear();
    be_quoted_leaf_node_indexes.resize(static_cast<std::size_t>(leaf_count));

    for (int i = 0; i < leaf_count; ++i) {
        OctreeNode& node =
            m_leaf_nodes[static_cast<std::size_t>(i)].get();

        if (node.quote_leaf_node_index != -1) {
            quote_leaf_node_indexes[static_cast<std::size_t>(i)] =
                node.quote_leaf_node_index;
        }

        if (!node.be_quoted_leaf_node_indexes.empty()) {
            be_quoted_leaf_node_indexes[static_cast<std::size_t>(i)].insert(
                be_quoted_leaf_node_indexes[static_cast<std::size_t>(i)].end(),
                node.be_quoted_leaf_node_indexes.begin(),
                node.be_quoted_leaf_node_indexes.end()
            );
        }
    }

    // ================================================================
    // quote_vertex_ids / be_quoted_vertex_ids / m_mesh_v_simulated
    // ================================================================
    const int vertex_count = static_cast<int>(m_mesh_v.size());

    quote_vertex_ids.assign(static_cast<std::size_t>(vertex_count), -1);

    be_quoted_vertex_ids.clear();
    be_quoted_vertex_ids.resize(static_cast<std::size_t>(vertex_count));

    m_mesh_v_simulated.clear();
    m_mesh_v_simulated.reserve(static_cast<std::size_t>(vertex_count));

    for (int vertex_id = 0; vertex_id < vertex_count; ++vertex_id) {
        auto it = vertex_to_leaf_cell.find(vertex_id);
        if (it == vertex_to_leaf_cell.end()) {
            throw std::runtime_error(
                "dfs_relate: vertex_to_leaf_cell missing vertex id."
            );
        }

        const int leaf_node_index = it->second;

        if (leaf_node_index < 0 || leaf_node_index >= leaf_count) {
            throw std::runtime_error(
                "dfs_relate: invalid leaf_node_index from vertex_to_leaf_cell."
            );
        }

        OctreeNode& node =
            m_leaf_nodes[static_cast<std::size_t>(leaf_node_index)].get();

        m_mesh_v_simulated.push_back(
            convert_grid_to_coords(node.m_centroid_id)
        );

        if (node.quote_vertex_id != -1) {
            quote_vertex_ids[static_cast<std::size_t>(vertex_id)] =
                node.quote_vertex_id;
        }

        if (!node.be_quoted_vertex_ids.empty()) {
            be_quoted_vertex_ids[static_cast<std::size_t>(vertex_id)].insert(
                be_quoted_vertex_ids[static_cast<std::size_t>(vertex_id)].end(),
                node.be_quoted_vertex_ids.begin(),
                node.be_quoted_vertex_ids.end()
            );
        }
    }

    // ================================================================
    // m_mesh_v_adj_cylinder
    // ================================================================
    m_mesh_v_adj_cylinder =ArrayB1::Constant(vertex_count, false);

    for (int leaf_node_index : cylinder_node_ids) {
        if (leaf_node_index < 0 || leaf_node_index >= leaf_count) {
            continue;
        }

        OctreeNode& node =m_leaf_nodes[static_cast<std::size_t>(leaf_node_index)].get();

        const int cur_vertex_idx = node.m_vertex_idx;

        if (cur_vertex_idx == -1) {
            continue;
        }

        for (int k = 0; k < m_leaf_nodes_adjacency.cols(); ++k) {
            const int adj_leaf_node_index =
                m_leaf_nodes_adjacency(leaf_node_index, k);

            if (adj_leaf_node_index == -1) {
                continue;
            }

            if (adj_leaf_node_index < 0 || adj_leaf_node_index >= leaf_count) {
                continue;
            }

            OctreeNode& adj_node =m_leaf_nodes[static_cast<std::size_t>(adj_leaf_node_index)].get();

            const int adj_vertex_idx = adj_node.m_vertex_idx;

            if (adj_vertex_idx == -1) {
                continue;
            }

            if (m_mesh_v_adj_cylinder(adj_vertex_idx)) {
                continue;
            }

            m_mesh_v_adj_cylinder(adj_vertex_idx) = true;
        }
    }
}

void Octree::generate_mesh()
{
    cell_quad2face_id.clear();
    cell_quad2face_id_inverse.clear();
    cell_quad2face.clear();

    m_mesh_f.clear();
    m_mesh_f_unique.clear();
    m_mesh_cell_type.clear();
    m_mesh_cell_id.clear();
    cellProcContour(m_root_node);
}


void Octree::cellProcContour(OctreeNode& node)
{
    if (node.m_node_type != "interval") {
        return;
    }

    if (node.m_children.size() != 8) {
        throw std::runtime_error(
            "cellProcContour: interval node must have 8 children."
        );
    }

    for (int i = 0; i < 8; ++i) {
        cellProcContour(*node.m_children[static_cast<std::size_t>(i)]);
    }

    // 12 face calls
    for (int i = 0; i < 12; ++i) {
        const int c0 = cellProcFaceMask[i][0];
        const int c1 = cellProcFaceMask[i][1];
        const int dir = cellProcFaceMask[i][2];

        std::array<OctreeNode*, 2> fcd = {
            node.m_children[static_cast<std::size_t>(c0)].get(),
            node.m_children[static_cast<std::size_t>(c1)].get()
        };

        faceProcContour(fcd, dir);
    }

    // 6 edge calls
    for (int i = 0; i < 6; ++i) {
        std::array<OctreeNode*, 4> ecd = {
            node.m_children[static_cast<std::size_t>(cellProcEdgeMask[i][0])].get(),
            node.m_children[static_cast<std::size_t>(cellProcEdgeMask[i][1])].get(),
            node.m_children[static_cast<std::size_t>(cellProcEdgeMask[i][2])].get(),
            node.m_children[static_cast<std::size_t>(cellProcEdgeMask[i][3])].get()
        };

        edgeProcContour(ecd, cellProcEdgeMask[i][4]);
    }
}


void Octree::faceProcContour(
    const std::array<OctreeNode*, 2>& nodes,
    int dir)
{
    if (nodes[0] == nullptr || nodes[1] == nullptr) {
        return;
    }

    if (nodes[0]->m_node_type == "empty" ||
        nodes[1]->m_node_type == "empty")
    {
        return;
    }

    if (nodes[0]->m_node_type == "interval" ||
        nodes[1]->m_node_type == "interval")
    {
        for (int i = 0; i < 4; ++i) {
            const int c0 = faceProcFaceMask[dir][i][0];
            const int c1 = faceProcFaceMask[dir][i][1];
            const int next_dir = faceProcFaceMask[dir][i][2];

            std::array<OctreeNode*, 2> fcd;

            if (nodes[0]->m_node_type == "leaf") {
                fcd[0] = nodes[0];
            } else {
                fcd[0] = nodes[0]->m_children[static_cast<std::size_t>(c0)].get();
            }

            if (nodes[1]->m_node_type == "leaf") {
                fcd[1] = nodes[1];
            } else {
                fcd[1] = nodes[1]->m_children[static_cast<std::size_t>(c1)].get();
            }

            faceProcContour(fcd, next_dir);
        }

        static const int orders[2][4] = {
            {0, 0, 1, 1},
            {0, 1, 0, 1}
        };

        for (int i = 0; i < 4; ++i) {
            const int order_id = faceProcEdgeMask[dir][i][0];

            const int c[4] = {
                faceProcEdgeMask[dir][i][1],
                faceProcEdgeMask[dir][i][2],
                faceProcEdgeMask[dir][i][3],
                faceProcEdgeMask[dir][i][4]
            };

            std::array<OctreeNode*, 4> ecd;

            for (int j = 0; j < 4; ++j) {
                const int node_id = orders[order_id][j];

                if (nodes[node_id]->m_node_type == "leaf") {
                    ecd[j] = nodes[node_id];
                } else {
                    ecd[j] =
                        nodes[node_id]->m_children[static_cast<std::size_t>(c[j])].get();
                }
            }

            edgeProcContour(ecd, faceProcEdgeMask[dir][i][5]);
        }
    }
}


void Octree::edgeProcContour(
    const std::array<OctreeNode*, 4>& nodes,
    int dir)
{
    for (int i = 0; i < 4; ++i) {
        if (nodes[i] == nullptr) {
            return;
        }

        if (nodes[i]->m_node_type == "empty") {
            return;
        }
    }

    const bool all_leaf =
        nodes[0]->m_node_type == "leaf" &&
        nodes[1]->m_node_type == "leaf" &&
        nodes[2]->m_node_type == "leaf" &&
        nodes[3]->m_node_type == "leaf";

    if (all_leaf) {
        processEdgeWrite(nodes, dir);
        return;
    }

    for (int i = 0; i < 2; ++i) {
        const int c[4] = {
            edgeProcEdgeMask[dir][i][0],
            edgeProcEdgeMask[dir][i][1],
            edgeProcEdgeMask[dir][i][2],
            edgeProcEdgeMask[dir][i][3]
        };

        std::array<OctreeNode*, 4> ecd;

        for (int j = 0; j < 4; ++j) {
            if (nodes[j]->m_node_type == "leaf") {
                ecd[j] = nodes[j];
            } else {
                ecd[j] =
                    nodes[j]->m_children[static_cast<std::size_t>(c[j])].get();
            }
        }

        edgeProcContour(ecd, edgeProcEdgeMask[dir][i][4]);
    }
}


void Octree::processEdgeWrite(
    const std::array<OctreeNode*, 4>& nodes,
    int dir)
{
    std::vector<Eigen::Vector3i> min_corner_ids;
    std::vector<Eigen::Vector3i> max_corner_ids;

    min_corner_ids.reserve(4);
    max_corner_ids.reserve(4);

    std::array<int, 2> indexes;
    int idx_count = 0;
    for (int k = 0; k < 3; ++k) {
        if (k != dir) {
            indexes[idx_count++] = k;
        }
    }

    std::vector<int> inds;
    std::vector<int> inds_unique;
    std::vector<int> leaf_node_indexes;
    std::vector<int> node_cell_types;
    // std::vector<int> map_node_ids;

    inds.reserve(4);
    inds_unique.reserve(4);
    leaf_node_indexes.reserve(4);
    node_cell_types.reserve(4);
    // map_node_ids.reserve(4);

    int feature_count = 0;

    for (int i = 0; i < 4; ++i) {
        OctreeNode* node = nodes[i];

        inds.push_back(node->m_vertex_idx);

        if (node->quote_vertex_id != -1) {
            inds_unique.push_back(node->quote_vertex_id);
        } else {
            inds_unique.push_back(node->m_vertex_idx);
        }

        const int leaf_idx = node->leaf_node_index;

        if (leaf_idx >= 0 &&
            leaf_idx < feature_flags.size() &&
            feature_flags(leaf_idx))
        {
            feature_count += 1;
        }

        leaf_node_indexes.push_back(leaf_idx);

        // auto it = vertex_to_leaf_cell.find(node->m_vertex_idx);
        // if (it != vertex_to_leaf_cell.end()) {
        //     map_node_ids.push_back(it->second);
        // } else {
        //     map_node_ids.push_back(-1);
        // }

        node_cell_types.push_back(node->m_node_cell_type);
        min_corner_ids.push_back(node->m_min_corner_id);
        max_corner_ids.push_back(node->m_max_corner_id);
    }

    // Eigen::Vector3i min_corner_id = min_corner_ids[0];
    // Eigen::Vector3i max_corner_id = max_corner_ids[0];

    // for (int i = 1; i < 4; ++i) {
    //     min_corner_id = min_corner_id.cwiseMin(min_corner_ids[i]);
    //     max_corner_id = max_corner_id.cwiseMax(max_corner_ids[i]);
    // }

    // MatI3 edge_ids;
    // edge_ids.row(0) = min_corner_id.transpose();
    // edge_ids.row(1) = max_corner_id.transpose();

    // for (int k = 0; k < 2; ++k) {
    //     const int axis = indexes[k];
    //     const int mid = (min_corner_id(axis) + max_corner_id(axis)) / 2;
    //     edge_ids(0, axis) = mid;
    //     edge_ids(1, axis) = mid;
    // }


    // MatF3 edge =convert_grid_to_coords(edge_ids);

    // (void)edge;
    // (void)feature_count;
    // (void)map_node_ids;

    std::vector<int> face_quad = {
        inds[0], inds[1], inds[3], inds[2]
    };

    std::vector<int> face_quad_unique = {
        inds_unique[0], inds_unique[1], inds_unique[3], inds_unique[2]
    };

    std::vector<int> cell_quad = {
        leaf_node_indexes[0],
        leaf_node_indexes[1],
        leaf_node_indexes[3],
        leaf_node_indexes[2]
    };

    std::vector<int> cell_quad_types = {
        node_cell_types[0],
        node_cell_types[1],
        node_cell_types[3],
        node_cell_types[2]
    };

    const int face_id = static_cast<int>(m_mesh_f.size());

    const std::array<int, 4> cell_key =
        sorted_quad_key(cell_quad);

    cell_quad2face_id[cell_key] = face_id;
    cell_quad2face_id_inverse[face_id] = cell_quad;
    cell_quad2face[cell_key] = face_quad;

    m_mesh_f.push_back(std::move(face_quad));
    m_mesh_f_unique.push_back(std::move(face_quad_unique));

    // every four elements in subarray:
    // 0 empty cell, 1 virtual cell, 2 real cell
    m_mesh_cell_type.push_back(std::move(cell_quad_types));

    // every element in subarray represents a leaf node index
    m_mesh_cell_id.push_back(std::move(cell_quad));
}
void Octree::scan_cavity()
{
    cavity_face_ids.clear();
    cavity_leaf_node_indexes.clear();
    cavity_vertex_indexes.clear();

    const int leaf_count = static_cast<int>(m_leaf_nodes.size());
    const int scale = 1 << m_sampling_depth;

    for (auto& node_ref : m_leaf_nodes) {
        const OctreeNode& node = node_ref.get();

        const Eigen::Vector3i coord =
            node.m_min_corner_id / scale;

        const int x = coord(0);
        const int y = coord(1);
        const int z = coord(2);

        if (!(x < resolution - 1 &&
              y < resolution - 1 &&
              z < resolution - 1))
        {
            continue;
        }

        bool check = true;

        std::vector<int> cavity_leaf_node_index;
        std::vector<int> cavity_vertex_index;

        cavity_leaf_node_index.reserve(8);
        cavity_vertex_index.reserve(8);

        for (int k = 0; k < 8; ++k) {
            const int nx = x + g_child_order(k, 0);
            const int ny = y + g_child_order(k, 1);
            const int nz = z + g_child_order(k, 2);

            if (!in_resolution_bounds(nx, ny, nz, resolution)) {
                check = false;
                break;
            }

            const int flat_id =flatten_cell_index(nx, ny, nz, resolution);

            if (!occupied(flat_id)) {
                check = false;
                break;
            }

            const int leaf_node_index =occupied_cell(flat_id);

            if (leaf_node_index < 0 ||
                leaf_node_index >= leaf_count)
            {
                check = false;
                break;
            }

            const OctreeNode& leaf_node =m_leaf_nodes[static_cast<std::size_t>(leaf_node_index)].get();

            if (leaf_node.m_node_type == "empty") {
                check = false;
                break;
            }

            cavity_leaf_node_index.push_back(leaf_node_index);
            cavity_vertex_index.push_back(leaf_node.m_vertex_idx);
        }

        if (!check) {
            continue;
        }

        // correspond to Python:
        // quad0=[0,1,2,3] x-neg
        // quad1=[4,5,6,7] x-pos
        // quad2=[0,1,5,4] y-neg
        // quad3=[2,3,7,6] y-pos
        // quad4=[0,2,6,4] z-neg
        // quad5=[1,3,7,5] z-pos
        const std::array<std::array<int, 4>, 6> quads = {{
            {{
                cavity_leaf_node_index[0],
                cavity_leaf_node_index[1],
                cavity_leaf_node_index[2],
                cavity_leaf_node_index[3]
            }},
            {{
                cavity_leaf_node_index[4],
                cavity_leaf_node_index[5],
                cavity_leaf_node_index[6],
                cavity_leaf_node_index[7]
            }},
            {{
                cavity_leaf_node_index[0],
                cavity_leaf_node_index[1],
                cavity_leaf_node_index[5],
                cavity_leaf_node_index[4]
            }},
            {{
                cavity_leaf_node_index[2],
                cavity_leaf_node_index[3],
                cavity_leaf_node_index[7],
                cavity_leaf_node_index[6]
            }},
            {{
                cavity_leaf_node_index[0],
                cavity_leaf_node_index[2],
                cavity_leaf_node_index[6],
                cavity_leaf_node_index[4]
            }},
            {{
                cavity_leaf_node_index[1],
                cavity_leaf_node_index[3],
                cavity_leaf_node_index[7],
                cavity_leaf_node_index[5]
            }}
        }};

        std::vector<int> cavity_face_id;
        cavity_face_id.reserve(6);

        bool skip = false;

        for (const auto& quad : quads) {
            const std::array<int, 4> key =
                sorted_quad_key_from_array(quad);

            auto it = cell_quad2face_id.find(key);

            if (it == cell_quad2face_id.end()) {
                std::cout
                    << "cavity face id not in cell_quad2face_id, skip it"
                    << std::endl;

                skip = true;
                break;
            }

            cavity_face_id.push_back(it->second);
        }

        if (skip) {
            continue;
        }

        cavity_face_ids.push_back(std::move(cavity_face_id));
        cavity_leaf_node_indexes.push_back(std::move(cavity_leaf_node_index));
        cavity_vertex_indexes.push_back(std::move(cavity_vertex_index));
    }

    std::cout
        << "scan " << cavity_face_ids.size()
        << " cavity done!"
        << std::endl;
}
ArrayB1 Octree::setup_cavity_flags(
    const std::vector<int>* group)
{
    ArrayB1 cavity_flags=ArrayB1::Constant(m_mesh_f.size(), false);

    face2cavity.clear();
    edge2cavity.clear();
    vertex2cavity.clear();

    std::unordered_set<int> group_set;
    if (group != nullptr) {
        group_set.reserve(group->size());
        for (int cavity_id : *group) {
            group_set.insert(cavity_id);
        }
    }

    for (std::size_t cavity_id_st = 0;cavity_id_st < cavity_face_ids.size();++cavity_id_st)
    {
        const int cavity_id = static_cast<int>(cavity_id_st);
        const std::vector<int>& cavity_face_id =
            cavity_face_ids[cavity_id_st];

        if (cavity_face_id.size() != 6) {
            continue;
        }

        if (group != nullptr &&
            group_set.find(cavity_id) == group_set.end())
        {
            continue;
        }

        for (int face_id : cavity_face_id) {
            if (face_id < 0 || face_id >= static_cast<int>(m_mesh_f.size()))
            {
                continue;
            }

            const std::vector<int>& face =  m_mesh_f[static_cast<std::size_t>(face_id)];

            if (face.size() != 4) {
                throw std::runtime_error(
                    "setup_cavity_flags: mesh face is not a quad."
                );
            }

            const std::pair<int, int> e0 = sorted_edge_key(face[0], face[1]);
            const std::pair<int, int> e1 = sorted_edge_key(face[1], face[2]);
            const std::pair<int, int> e2 = sorted_edge_key(face[2], face[3]);
            const std::pair<int, int> e3 = sorted_edge_key(face[3], face[0]);

            const std::array<std::pair<int, int>, 4> edges = {
                e0, e1, e2, e3
            };

            cavity_flags(face_id) = true;

            auto& face_cavities = face2cavity[face_id];
            if (!vector_contains(face_cavities, cavity_id)) {
                face_cavities.push_back(cavity_id);
            }

            for (const auto& edge : edges) {
                auto& edge_cavities = edge2cavity[edge];
                if (!vector_contains(edge_cavities, cavity_id)) {
                    edge_cavities.push_back(cavity_id);
                }
            }

            for (int vid : face) {
                auto& vertex_cavities = vertex2cavity[vid];
                if (!vector_contains(vertex_cavities, cavity_id)) {
                    vertex_cavities.push_back(cavity_id);
                }
            }
        }
    }

    return cavity_flags;
}
void Octree::backup_cavity_data()
{
    origin_cavity_flags = setup_cavity_flags();

    origin_cavity_leaf_node_indexes = cavity_leaf_node_indexes;
    origin_cavity_vertex_indexes = cavity_vertex_indexes;
    origin_cavity_face_ids = cavity_face_ids;

    origin_face2cavity = face2cavity;
    origin_edge2cavity = edge2cavity;
    origin_vertex2cavity = vertex2cavity;
}
void Octree::initial_topology_data()
{
    const Eigen::Index vertex_count =
        static_cast<Eigen::Index>(m_mesh_v.size());

    const Eigen::Index face_count =
        static_cast<Eigen::Index>(m_mesh_f.size());


    deleted_face_flag =
        ArrayB1::Constant(face_count, false);

    static_face_flag =
        ArrayB1::Constant(face_count, true);

    // Python:
    // for fid in self.face2cavity:
    //     self.static_face_flag[fid] = False
    for (const auto& kv : face2cavity) {
        const int fid = kv.first;

        if (fid >= 0 && fid < face_count) {
            static_face_flag(fid) = false;
        }
    }

    // Python:
    // self.face_adjacency = {}
    // for fid in range(len(self.m_mesh_f)):
    //     self.face_adjacency[fid] = {}
    face_adjacency.clear();
    face_adjacency.reserve(face_count);
    for (int fid = 0; fid < face_count; ++fid) {
        face_adjacency.emplace(fid, std::vector<std::vector<int>>{});
    }

    // Python:
    // self.cavity_adjacenty = {}
    // for cavity_id in range(len(self.cavity_face_ids)):
    //     self.cavity_adjacenty[cavity_id] = {}
    cavity_adjacenty.clear();
    cavity_adjacenty.reserve(cavity_face_ids.size());
    for (int cid = 0; cid < static_cast<int>(cavity_face_ids.size()); ++cid) {
        cavity_adjacenty.emplace(cid, std::unordered_map<int, int>{});
    }
}
void Octree::make_vertex_edge_face_topology()
{
    face_to_edge.clear();
    edge_to_faces.clear();
    vertex_to_edge.clear();

    // Python:
    // self.face_adjacency = {}
    face_adjacency.clear();

    const int face_count = static_cast<int>(m_mesh_f.size());

    for (int face_id = 0; face_id < face_count; ++face_id) {
        if (deleted_face_flag.size() == face_count &&
            deleted_face_flag(face_id))
        {
            continue;
        }

        const std::vector<int>& face =
            m_mesh_f[static_cast<std::size_t>(face_id)];

        const std::vector<std::pair<int, int>> edges =
            extract_edges(face);

        face_to_edge[face_id] = edges;

        // Python:
        // self.face_adjacency[face_id] = {}
        // self.face_adjacency[face_id][edge_idx] = adj_faces
        face_adjacency[face_id] =
            std::vector<std::vector<int>>(edges.size());

        for (std::size_t edge_idx = 0; edge_idx < edges.size(); ++edge_idx) {
            const std::pair<int, int>& edge = edges[edge_idx];

            auto& faces_on_edge = edge_to_faces[edge];

            if (!vector_contains(faces_on_edge, face_id)) {
                faces_on_edge.push_back(face_id);
            }

            const int id0 = edge.first;
            const int id1 = edge.second;

            auto& edges_of_v0 = vertex_to_edge[id0];
            if (!edge_vector_contains(edges_of_v0, edge)) {
                edges_of_v0.push_back(edge);
            }

            auto& edges_of_v1 = vertex_to_edge[id1];
            if (!edge_vector_contains(edges_of_v1, edge)) {
                edges_of_v1.push_back(edge);
            }

            std::vector<int> adj_faces;
            adj_faces.reserve(faces_on_edge.size());

            for (int f : faces_on_edge) {
                if (f != face_id) {
                    adj_faces.push_back(f);
                }
            }

            face_adjacency[face_id][edge_idx] = std::move(adj_faces);
        }
    }

    // Python:
    // for edge in self.alone_edges:
    //     add to vertex_to_edge
    for (const auto& raw_edge : alone_edges) {
        const std::pair<int, int> edge =
            make_sorted_edge(raw_edge.first, raw_edge.second);

        const int id0 = edge.first;
        const int id1 = edge.second;

        auto& edges_of_v0 = vertex_to_edge[id0];
        if (!edge_vector_contains(edges_of_v0, edge)) {
            edges_of_v0.push_back(edge);
        }

        auto& edges_of_v1 = vertex_to_edge[id1];
        if (!edge_vector_contains(edges_of_v1, edge)) {
            edges_of_v1.push_back(edge);
        }
    }
}

void Octree::make_cavity_adjacency_dict()
{
    cavity_adjacenty.clear();

    for (std::size_t cavity_id = 0; cavity_id < cavity_face_ids.size(); ++cavity_id) {
        cavity_adjacenty.emplace(
            static_cast<int>(cavity_id),
            std::unordered_map<int, int>{}
        );
    }

    for (const auto& kv : face2cavity) {
        const int face_id = kv.first;
        const std::vector<int>& face_cavity_ids = kv.second;

        for (int cur_cavity_id : face_cavity_ids) {
            int adj_cavity_id = -1;

            for (int candidate_id : face_cavity_ids) {
                if (candidate_id != cur_cavity_id) {
                    adj_cavity_id = candidate_id;
                    break;
                }
            }

            cavity_adjacenty[cur_cavity_id][face_id] = adj_cavity_id;
        }
    }
}

bool Octree::check_edge_exist(const std::pair<int, int>& edge_raw) const
{
    const std::pair<int, int> edge =
        make_sorted_edge(edge_raw.first, edge_raw.second);

    if (alone_edges.find(edge) != alone_edges.end()) {
        return true;
    }

    auto it = edge_to_faces.find(edge);

    if (it != edge_to_faces.end() && !it->second.empty()) {
        const std::vector<int>& face_ids = it->second;

        for (int fid : face_ids) {
            if (fid < 0 || fid >= deleted_face_flag.size()) {
                continue;
            }

            if (!deleted_face_flag(fid)) {
                return true;
            }
        }

        return false;
    }

    return false;
}
void Octree::add_edge_to_vertex_map(const std::pair<int, int>& edge_raw)
{
    const std::pair<int, int> edge =
        make_sorted_edge(edge_raw.first, edge_raw.second);

    const int v0 = edge.first;
    const int v1 = edge.second;

    {
        auto& edges0 = vertex_to_edge[v0];

        if (!edge_vector_contains(edges0, edge)) {
            edges0.push_back(edge);
        }
    }

    {
        auto& edges1 = vertex_to_edge[v1];

        if (!edge_vector_contains(edges1, edge)) {
            edges1.push_back(edge);
        }
    }
}
void Octree::generate_alone_edges()
{
    if(convex_feature_points.size()>0){
        return;
    }
    const int leaf_count =
        static_cast<int>(m_leaf_nodes.size());

    for (int leaf_node_index = 0;
         leaf_node_index < leaf_count;
         ++leaf_node_index)
    {
        OctreeNode& node =
            m_leaf_nodes[static_cast<std::size_t>(leaf_node_index)].get();

        if (node.m_vertex_idx == -1) {
            continue;
        }

        if (!node.is_cylinder) {
            continue;
        }

        const int vid = node.m_vertex_idx;

        for (int k = 0; k < m_leaf_nodes_adjacency.cols(); ++k) {
            const int adj_leaf_node_index =
                m_leaf_nodes_adjacency(leaf_node_index, k);

            if (adj_leaf_node_index == -1) {
                continue;
            }

            if (adj_leaf_node_index < 0 ||
                adj_leaf_node_index >= leaf_count)
            {
                continue;
            }

            OctreeNode& adj_node =
                m_leaf_nodes[static_cast<std::size_t>(adj_leaf_node_index)].get();

            const int adj_leaf_node_vertex_id =
                adj_node.m_vertex_idx;

            if (adj_leaf_node_vertex_id == -1) {
                continue;
            }

            const std::pair<int, int> edge =
                make_sorted_edge(vid, adj_leaf_node_vertex_id);

            if (check_edge_exist(edge)) {
                continue;
            }

            if (adj_node.is_cylinder) {
                alone_edges.insert(edge);
                add_edge_to_vertex_map(edge);
            } else {
                const bool adj_vertex_has_edges =
                    vertex_to_edge.find(adj_leaf_node_vertex_id) != vertex_to_edge.end();

                if (!adj_vertex_has_edges) {
                    if (!adj_node.be_quoted_vertex_ids.empty()) {
                        const int be_quoted_vertex_id =
                            adj_node.be_quoted_vertex_ids[0];

                        alone_edges.insert(edge);
                        add_edge_to_vertex_map(edge);

                        const std::pair<int, int> connect_edge =
                            make_sorted_edge(
                                be_quoted_vertex_id,
                                adj_leaf_node_vertex_id
                            );

                        if (!check_edge_exist(connect_edge)) {
                            alone_edges.insert(connect_edge);
                            add_edge_to_vertex_map(connect_edge);
                        }
                    }
                } else {
                    alone_edges.insert(edge);
                    add_edge_to_vertex_map(edge);
                }
            }
        }
    }
}

void Octree::triangulate_faces_by_udf()
{
    const int face_count = static_cast<int>(m_mesh_f.size());
    m_mesh_f_tri.resize(static_cast<std::size_t>(face_count) * 2);
    m_mesh_f_area2.resize(static_cast<std::size_t>(face_count));
    #pragma omp parallel for schedule(static) if (face_count > 1024)
    for (int fid = 0; fid < face_count; ++fid) {
        const std::vector<int>& face = m_mesh_f[static_cast<std::size_t>(fid)];

        if (face.size() != 4) {
            throw std::runtime_error(
                "triangulate_faces_by_udf: only quad faces are supported."
            );
        }

        const int f0 = face[0];
        const int f1 = face[1];
        const int f2 = face[2];
        const int f3 = face[3];

        const Eigen::Vector3f& v0 = m_mesh_v[static_cast<std::size_t>(f0)];
        const Eigen::Vector3f& v1 = m_mesh_v[static_cast<std::size_t>(f1)];
        const Eigen::Vector3f& v2 = m_mesh_v[static_cast<std::size_t>(f2)];
        const Eigen::Vector3f& v3 = m_mesh_v[static_cast<std::size_t>(f3)];

        const int t0 = m_v_type[static_cast<std::size_t>(f0)];
        const int t1 = m_v_type[static_cast<std::size_t>(f1)];
        const int t2 = m_v_type[static_cast<std::size_t>(f2)];
        const int t3 = m_v_type[static_cast<std::size_t>(f3)];

        if (t0 == static_cast<int>(DualPointType::EDGE) &&
            t1 == static_cast<int>(DualPointType::PLANE) &&
            t2 == static_cast<int>(DualPointType::EDGE) &&
            t3 == static_cast<int>(DualPointType::PLANE))
        {
            m_mesh_f_tri[fid*2]={f0, f2, f3};
            m_mesh_f_tri[fid*2+1]={f0, f2, f1};

            const float double_area0 = doublearea_single(v0, v1, v2);

            const float double_area1 = doublearea_single(v2, v3, v0);

            m_mesh_f_area2[fid] = std::abs(double_area0) + std::abs(double_area1);

            continue;
        }


        if (t0 == static_cast<int>(DualPointType::PLANE) &&
            t1 == static_cast<int>(DualPointType::EDGE) &&
            t2 == static_cast<int>(DualPointType::PLANE) &&
            t3 == static_cast<int>(DualPointType::EDGE))
        {
            m_mesh_f_tri[fid*2] ={f0, f1, f3};
            m_mesh_f_tri[fid*2+1] ={f2, f3, f1};

            const float double_area0 = doublearea_single(v0, v1, v3);

            const float double_area1 = doublearea_single(v1, v2, v3);

            m_mesh_f_area2[fid] = std::abs(double_area0) + std::abs(double_area1);

            continue;
        }

        const float udf01 = m_mesh_f_udfs(fid, 0);
        const float udf23 = m_mesh_f_udfs(fid, 1);

        if (udf01 < udf23) {
            m_mesh_f_tri[fid*2] ={f0, f1, f2};
            m_mesh_f_tri[fid*2+1] ={f2, f3, f0};

            const float double_area0 = doublearea_single(v0, v1, v2);

            const float double_area1 = doublearea_single(v2, v3, v0);

            m_mesh_f_area2[static_cast<std::size_t>(fid)] =
                std::abs(double_area0) + std::abs(double_area1);
        } else {
            m_mesh_f_tri[fid*2] ={f0, f1, f3};
            m_mesh_f_tri[fid*2+1] ={f1, f2, f3};

            const float double_area0 = doublearea_single(v0, v1, v3);

            const float double_area1 = doublearea_single(v1, v2, v3);

            m_mesh_f_area2[fid] = std::abs(double_area0) + std::abs(double_area1);
        }
    }
}

void Octree::update_cavity_data()
{
    static const std::array<std::pair<int, int>, 12> edge_indexes = {{
        {0, 1}, {0, 2}, {2, 3}, {3, 1},
        {4, 5}, {4, 6}, {6, 7}, {7, 5},
        {0, 4}, {1, 5}, {2, 6}, {3, 7}
    }};

    const int cavity_count =
        static_cast<int>(cavity_face_ids.size());

    for (int cavity_id = 0; cavity_id < cavity_count; ++cavity_id) {
        std::vector<int>& cavity_face_id =
            cavity_face_ids[static_cast<std::size_t>(cavity_id)];

        // Python: cavity_face_id_origin = copy.deepcopy(cavity_face_id)
        const std::vector<int> cavity_face_id_origin = cavity_face_id;

        bool cavity_break = false;

        for (int fid : cavity_face_id_origin) {
            if (fid < 0 || fid >= deleted_face_flag.size()) {
                continue;
            }

            if (deleted_face_flag(fid)) {
                cavity_break = true;

                // Python: cavity_face_id.remove(fid)
                remove_value_from_vector(cavity_face_id, fid);

                // Python:
                // if fid in self.face2cavity and cavity_id in self.face2cavity[fid]:
                //     self.face2cavity[fid].remove(cavity_id)
                auto face_it = face2cavity.find(fid);
                if (face_it != face2cavity.end()) {
                    remove_value_from_vector(face_it->second, cavity_id);
                }
            }
        }

        if (!cavity_break) {
            continue;
        }

        if (cavity_id < 0 ||
            cavity_id >= static_cast<int>(origin_cavity_vertex_indexes.size()))
        {
            continue;
        }

        const std::vector<int>& cavity_vertex_indexes =
            origin_cavity_vertex_indexes[static_cast<std::size_t>(cavity_id)];

        if (cavity_vertex_indexes.size() < 8) {
            continue;
        }

        for (const auto& edge_index : edge_indexes) {
            const int i0 = edge_index.first;
            const int i1 = edge_index.second;

            const int v0 = cavity_vertex_indexes[static_cast<std::size_t>(i0)];
            const int v1 = cavity_vertex_indexes[static_cast<std::size_t>(i1)];

            const std::pair<int, int> edge =
                make_sorted_edge(v0, v1);

            auto edge_it = edge2cavity.find(edge);
            if (edge_it == edge2cavity.end()) {
                continue;
            }

            // Python:
            // if cavity_id in self.edge2cavity[edge]:
            //     self.edge2cavity[edge].remove(cavity_id)
            remove_value_from_vector(edge_it->second, cavity_id);
        }
    }
}

int Octree::f_num() const
{
    if (deleted_face_flag.size() == 0) {
        return 0;
    }

    return static_cast<int>(
        deleted_face_flag.size() - deleted_face_flag.count()
    );
}

int Octree::v_num() const
{
    int count = 0;

    const int vertex_count = static_cast<int>(m_mesh_v.size());

    for (int i = 0; i < vertex_count; ++i) {
        auto it = vertex_to_edge.find(i);

        if (it == vertex_to_edge.end()) {
            continue;
        }

        if (it->second.empty()) {
            continue;
        }

        ++count;
    }

    return count;
}

int Octree::e_num() const
{
    int count = 0;

    for (const auto& kv : edge_to_faces) {
        const std::vector<int>& face_ids = kv.second;

        if (face_ids.empty()) {
            continue;
        }

        bool exist_flag = false;

        for (int fid : face_ids) {
            if (fid < 0 || fid >= deleted_face_flag.size()) {
                continue;
            }

            if (!deleted_face_flag(fid)) {
                exist_flag = true;
                break;
            }
        }

        if (exist_flag) {
            ++count;
        }
    }

    count += static_cast<int>(alone_edges.size());

    return count;
}

int Octree::cavity_num(const std::vector<int>* group) const
{
    int count = 0;

    std::unordered_set<int> group_set;
    if (group != nullptr) {
        group_set.reserve(group->size());
        for (int cavity_id : *group) {
            group_set.insert(cavity_id);
        }
    }

    const int cavity_count =
        static_cast<int>(cavity_face_ids.size());

    for (int i = 0; i < cavity_count; ++i) {
        if (group != nullptr &&
            group_set.find(i) == group_set.end())
        {
            continue;
        }

        if (cavity_face_ids[static_cast<std::size_t>(i)].size() == 6) {
            ++count;
        }
    }

    return count;
}

int Octree::eular_characteristic() const
{
    return v_num() - e_num() + f_num() - cavity_num();
}

void Octree::update_cavity_tables(int face_id)
{
    auto origin_it = origin_face2cavity.find(face_id);
    if (origin_it == origin_face2cavity.end()) {
        return;
    }

    if (face_id < 0 || face_id >= deleted_face_flag.size()) {
        return;
    }

    static const std::array<std::pair<int, int>, 12> edge_indexes = {{
        {0, 1}, {0, 2}, {2, 3}, {3, 1},
        {4, 5}, {4, 6}, {6, 7}, {7, 5},
        {0, 4}, {1, 5}, {2, 6}, {3, 7}
    }};

    // ============================================================
    // delete face
    // ============================================================
    if (deleted_face_flag(face_id)) {
        const std::vector<int>& origin_cavity_ids = origin_it->second;

        for (int cavity_id : origin_cavity_ids) {
            if (cavity_id < 0 ||
                cavity_id >= static_cast<int>(cavity_face_ids.size()))
            {
                continue;
            }

            remove_value_from_vector(
                cavity_face_ids[static_cast<std::size_t>(cavity_id)],
                face_id
            );
        }

        // Python:
        // cur_cavity_ids = self.face2cavity[face_id]
        // for cavity_id in cur_cavity_ids:
        std::vector<int> cur_cavity_ids;
        auto cur_it = face2cavity.find(face_id);
        if (cur_it != face2cavity.end()) {
            cur_cavity_ids = cur_it->second;
        }

        for (int cavity_id : cur_cavity_ids) {
            remove_value_from_vector(face2cavity[face_id], cavity_id);

            if (cavity_id < 0 ||
                cavity_id >= static_cast<int>(cavity_face_ids.size()))
            {
                continue;
            }

            if (cavity_face_ids[static_cast<std::size_t>(cavity_id)].size() < 6) {
                for (int fid : cavity_face_ids[static_cast<std::size_t>(cavity_id)]) {
                    auto fit = face2cavity.find(fid);
                    if (fit != face2cavity.end()) {
                        remove_value_from_vector(fit->second, cavity_id);
                    }
                }

                if (cavity_id < static_cast<int>(origin_cavity_vertex_indexes.size())) {
                    const std::vector<int>& cavity_vertex_indexes =
                        origin_cavity_vertex_indexes[static_cast<std::size_t>(cavity_id)];

                    if (cavity_vertex_indexes.size() == 8) {
                        for (const auto& edge_index : edge_indexes) {
                            const int i0 = edge_index.first;
                            const int i1 = edge_index.second;

                            const int v0 =
                                cavity_vertex_indexes[static_cast<std::size_t>(i0)];

                            const int v1 =
                                cavity_vertex_indexes[static_cast<std::size_t>(i1)];

                            const std::pair<int, int> edge =
                                make_sorted_edge(v0, v1);

                            auto eit = edge2cavity.find(edge);
                            if (eit != edge2cavity.end()) {
                                remove_value_from_vector(eit->second, cavity_id);
                            }
                        }
                    }
                }
            }
        }

        return;
    }

    // ============================================================
    // add face
    // ============================================================
    const std::vector<int>& origin_cavity_ids = origin_it->second;

    for (int cavity_id : origin_cavity_ids) {
        if (cavity_id < 0 ||
            cavity_id >= static_cast<int>(cavity_face_ids.size()))
        {
            continue;
        }

        std::vector<int>& cavity_faces =
            cavity_face_ids[static_cast<std::size_t>(cavity_id)];

        if (!vector_contains(cavity_faces, face_id)) {
            cavity_faces.push_back(face_id);
        }

        // cavity is complete
        if (cavity_faces.size() == 6) {
            if (!vector_contains(face2cavity[face_id], cavity_id)) {
                face2cavity[face_id].push_back(cavity_id);
            }

            for (int fid : cavity_faces) {
                if (!vector_contains(face2cavity[fid], cavity_id)) {
                    face2cavity[fid].push_back(cavity_id);
                }
            }

            if (cavity_id < static_cast<int>(origin_cavity_vertex_indexes.size())) {
                const std::vector<int>& cavity_vertex_indexes =
                    origin_cavity_vertex_indexes[static_cast<std::size_t>(cavity_id)];

                if (cavity_vertex_indexes.size() == 8) {
                    for (const auto& edge_index : edge_indexes) {
                        const int i0 = edge_index.first;
                        const int i1 = edge_index.second;

                        const int v0 =
                            cavity_vertex_indexes[static_cast<std::size_t>(i0)];

                        const int v1 =
                            cavity_vertex_indexes[static_cast<std::size_t>(i1)];

                        const std::pair<int, int> edge =
                            make_sorted_edge(v0, v1);

                        if (!vector_contains(edge2cavity[edge], cavity_id)) {
                            edge2cavity[edge].push_back(cavity_id);
                        }
                    }
                }
            }
        }
    }
}

void Octree::update_relate_face_edge_vertex_table(int face_id)
{
    if (deleted_face_flag.size() != static_cast<Eigen::Index>(m_mesh_f.size())) {
        throw std::runtime_error(
            "update_relate_face_edge_vertex_table: deleted_face_flag size mismatch."
        );
    }

    const std::vector<int>& face = m_mesh_f[static_cast<std::size_t>(face_id)];

    const std::vector<std::pair<int, int>> edges = extract_edges(face);

    for (const auto& edge : edges) {
        if (deleted_face_flag(face_id) && edge_to_faces.find(edge) != edge_to_faces.end()) {
            // delete face
            std::vector<int>& faces_on_edge = edge_to_faces[edge];

            remove_value_from_vector(faces_on_edge, face_id);

            faces_on_edge.erase(
                std::remove_if(
                    faces_on_edge.begin(),
                    faces_on_edge.end(),
                    [&](int fid) {
                        if (fid < 0 || fid >= deleted_face_flag.size()) {
                            return true;
                        }
                        return deleted_face_flag(fid);
                    }
                ),
                faces_on_edge.end()
            );

            if (faces_on_edge.empty()) {
                const int id0 = edge.first;
                const int id1 = edge.second;

                auto it0 = vertex_to_edge.find(id0);
                if (it0 != vertex_to_edge.end()) {
                    auto& edges0 = it0->second;
                    edges0.erase(
                        std::remove(edges0.begin(), edges0.end(), edge),
                        edges0.end()
                    );
                }

                auto it1 = vertex_to_edge.find(id1);
                if (it1 != vertex_to_edge.end()) {
                    auto& edges1 = it1->second;
                    edges1.erase(
                        std::remove(edges1.begin(), edges1.end(), edge),
                        edges1.end()
                    );
                }
            }
        } else {
            // add face
            std::vector<int>& faces_on_edge = edge_to_faces[edge];

            if (!vector_contains(faces_on_edge, face_id)) {
                faces_on_edge.push_back(face_id);
            }

            const int id0 = edge.first;
            const int id1 = edge.second;

            {
                auto& edges0 = vertex_to_edge[id0];

                if (!edge_vector_contains(edges0, edge)) {
                    edges0.push_back(edge);
                }
            }

            {
                auto& edges1 = vertex_to_edge[id1];

                if (!edge_vector_contains(edges1, edge)) {
                    edges1.push_back(edge);
                }
            }
        }
    }
}

std::vector<float> Octree::compute_face_rank_impl(int face_id)
{
    const float INF = std::numeric_limits<float>::infinity();

    if (face_id < 0 || face_id >= static_cast<int>(m_mesh_f.size())) {
        throw std::out_of_range("compute_face_rank_impl: invalid face_id.");
    }

    if (m_mesh_f_udfs.rows() != static_cast<Eigen::Index>(m_mesh_f.size()) ||
        m_mesh_f_udfs.cols() != 2)
    {
        throw std::runtime_error(
            "compute_face_rank_impl: m_mesh_f_udfs must have shape (num_faces, 2)."
        );
    }

    // Python:
    // geo_rank = -min(self.m_mesh_f_udfs[face_id])
    const float geo_rank =-m_mesh_f_udfs.row(face_id).minCoeff();

    // Python:
    // if face_id in self.face2cavity:
    //     if len(self.face2cavity[face_id]) >= 2:
    //         return (np.inf, geo_rank)
    auto face_cavity_it = face2cavity.find(face_id);
    if (face_cavity_it != face2cavity.end()) {
        if (face_cavity_it->second.size() >= 2) {
            return {INF, geo_rank};
        }
    }

    const std::vector<int>& face =m_mesh_f[static_cast<std::size_t>(face_id)];

    const std::vector<std::pair<int, int>> edges =extract_edges(face);

    std::vector<std::pair<int, int>> adj_static_face_edges;

    // Python:
    // for edge in edges:
    //     for fid in self.edge_to_faces[edge]:
    //         if fid != face_id and len(self.edge_to_faces[edge]) == 2
    //            and self.static_face_flag[fid] == True:
    //             adj_static_face_edges.append(edge)
    //             break
    for (const auto& edge : edges) {
        auto edge_it = edge_to_faces.find(edge);
        if (edge_it == edge_to_faces.end()) {
            continue;
        }

        const std::vector<int>& faces_on_edge = edge_it->second;

        for (int fid : faces_on_edge) {
            if (fid < 0 || fid >= static_cast<int>(static_face_flag.size())) {
                continue;
            }

            if (fid != face_id &&faces_on_edge.size() == 2 &&
                static_face_flag(fid))
            {
                adj_static_face_edges.push_back(edge);
                break;
            }
        }
    }

    // Python:
    // if len(adj_static_face_edges) == 4:
    //     self.static_face_flag[face_id] = True
    //     return (np.inf, geo_rank)
    if (adj_static_face_edges.size() == 4) {
        static_face_flag(face_id) = true;
        return {INF, geo_rank};
    }

    // Python:
    // for edge in adj_static_face_edges:
    //     if len(self.edge_to_faces[edge]) == 2:
    //         return (np.inf, geo_rank)
    for (const auto& edge : adj_static_face_edges) {
        auto edge_it = edge_to_faces.find(edge);
        if (edge_it != edge_to_faces.end() &&
            edge_it->second.size() == 2)
        {
            return {INF, geo_rank};
        }
    }

    bool is_pocket = false;
    float topo_rank = 0.0f;

    // Python:
    // for vid in face:
    //     leaf_node_index = self.vertex_to_leaf_cell[vid]
    //     if self.m_leaf_nodes[leaf_node_index].is_cylinder:
    //         topo_rank += 1
    for (int vid : face) {
        auto it = vertex_to_leaf_cell.find(vid);
        if (it == vertex_to_leaf_cell.end()) {
            throw std::runtime_error(
                "compute_face_rank_impl: vertex id missing in vertex_to_leaf_cell."
            );
        }

        const int leaf_node_index = it->second;

        if (leaf_node_index < 0 ||
            leaf_node_index >= static_cast<int>(m_leaf_nodes.size()))
        {
            throw std::runtime_error(
                "compute_face_rank_impl: invalid leaf node index."
            );
        }

        const OctreeNode& node =
            m_leaf_nodes[static_cast<std::size_t>(leaf_node_index)].get();

        if (node.is_cylinder) {
            topo_rank += 1.0f;
        }
    }

    // Python:
    // for edge in edges:
    //     if len(self.edge_to_faces[edge]) == 1:
    //         ...
    for (const auto& edge : edges) {
        auto edge_it = edge_to_faces.find(edge);
        if (edge_it == edge_to_faces.end()) {
            continue;
        }

        const std::vector<int>& faces_on_edge = edge_it->second;

        if (faces_on_edge.size() == 1) {
            const int i0 = edge.first;
            const int i1 = edge.second;

            auto it0 = vertex_to_leaf_cell.find(i0);
            auto it1 = vertex_to_leaf_cell.find(i1);

            if (it0 == vertex_to_leaf_cell.end() ||
                it1 == vertex_to_leaf_cell.end())
            {
                throw std::runtime_error(
                    "compute_face_rank_impl: edge vertex missing in vertex_to_leaf_cell."
                );
            }

            const int leaf_node_index0 = it0->second;
            const int leaf_node_index1 = it1->second;

            if (leaf_node_index0 < 0 ||
                leaf_node_index0 >= static_cast<int>(m_leaf_nodes.size()) ||
                leaf_node_index1 < 0 ||
                leaf_node_index1 >= static_cast<int>(m_leaf_nodes.size()))
            {
                throw std::runtime_error(
                    "compute_face_rank_impl: invalid leaf node index on boundary edge."
                );
            }

            const bool feature0 = feature_flags(leaf_node_index0);

            const bool feature1 = feature_flags(leaf_node_index1);

            const OctreeNode& node0 = m_leaf_nodes[static_cast<std::size_t>(leaf_node_index0)].get();

            const OctreeNode& node1 = m_leaf_nodes[static_cast<std::size_t>(leaf_node_index1)].get();

            // Python:
            // if self.feature_flags[leaf_node_index0] and self.feature_flags[leaf_node_index1]:
            //     return (np.inf, geo_rank)
            if (feature0 && feature1) {
                return {INF, geo_rank};
            }

            // Python:
            // elif self.m_leaf_nodes[leaf_node_index0].is_cylinder
            //      or self.m_leaf_nodes[leaf_node_index1].is_cylinder:
            //     return (np.inf, geo_rank)
            if (node0.is_cylinder || node1.is_cylinder) {
                return {INF, geo_rank};
            }

            is_pocket = true;
        }
    }

    // Python:
    // if not is_pocket:
    //     if face_id not in self.face2cavity or len(self.face2cavity[face_id]) == 0:
    //         return (np.inf, geo_rank)
    if (!is_pocket) {
        auto it = face2cavity.find(face_id);

        if (it == face2cavity.end() || it->second.empty()) {
            return {INF, geo_rank};
        }
    }

    return {topo_rank, geo_rank};
}

std::vector<float> Octree::compute_face_rank_global(int face_id)
{
    return compute_face_rank_impl(face_id);
}

std::vector<float> Octree::compute_face_rank_local(int face_id)
{
    return compute_face_rank_impl(face_id);
}

std::vector<std::pair<int, std::vector<float>>>
Octree::compute_all_cavity_face_rank_local(
    const std::vector<int>& all_cavity_face_id)
{
    std::vector<std::pair<int, std::vector<float>> > results;
    results.reserve(all_cavity_face_id.size());

    for (int face_id : all_cavity_face_id) {
        if (deleted_face_flag(face_id)) {
            continue;
        }

        std::vector<float> priority_tuple =compute_face_rank_local(face_id);

        results.emplace_back(face_id, priority_tuple);
    }

    return results;
}

std::vector<std::pair<int, std::vector<float>>>
Octree::compute_all_cavity_face_rank_global(
    const std::vector<int>& all_cavity_face_id)
{
    std::vector<std::pair<int, std::vector<float> >> results;
    results.reserve(all_cavity_face_id.size());

    for (int face_id : all_cavity_face_id) {
        if (face_id < 0 ||
            face_id >= static_cast<int>(deleted_face_flag.size()))
        {
            continue;
        }

        if (deleted_face_flag(face_id)) {
            continue;
        }

        std::vector<float> priority_tuple =
            compute_face_rank_global(face_id);

        results.emplace_back(face_id, priority_tuple);
    }

    return results;
}

void Octree::clean_cavity_face() {

    std::tie(edge_to_faces, vertex_to_edge) = make_dict_edge2face(m_mesh_f, deleted_face_flag);

    degree_four_edges.clear();
    for (const auto& kv : edge_to_faces) {
        if (kv.second.size() == 4) {
            degree_four_edges.push_back(kv.first);
        }
    }


    ArrayB1 cavity_flags = origin_cavity_flags;


    std::vector<int> all_cavity_face_id;
    for (size_t i = 0; i < cavity_flags.size(); ++i) {
        if (cavity_flags[i]) {
            all_cavity_face_id.push_back(static_cast<int>(i));
        }
    }

    int count = 0;
    int before_euler = eular_characteristic();
    for (size_t cavity_id = 0; cavity_id < cavity_face_ids.size(); ++cavity_id) {
        std::vector<int>& face_ids = cavity_face_ids[cavity_id];
        std::vector<int> new_face_ids;
        for (int fid : face_ids) {
            if (deleted_face_flag[fid] || face2cavity[fid].size() == 2) continue;
            new_face_ids.push_back(fid);
        }
        if (new_face_ids.empty()) continue;

        auto results = compute_all_cavity_face_rank_local(new_face_ids);
        MulPriorityQueue<int> cavity_deleted_queue;
        for (const auto& r : results) {
            cavity_deleted_queue.push(r.first, r.second);
        }

        while (!cavity_deleted_queue.is_empty()) {
            auto [face_id, priority_tuple] = cavity_deleted_queue.pop();
            if (priority_tuple[0] == std::numeric_limits<float>::infinity()) break;
            if (deleted_face_flag[face_id]) continue;

            ++count;
            deleted_face_flag[face_id] = true;
            update_cavity_tables(face_id);
            update_relate_table_and_rank_local(face_id, cavity_deleted_queue);

            int after_euler = eular_characteristic();
            if (after_euler != before_euler) {
                //std::cerr << face_id << " eular characteristic not equal\n";
                deleted_face_flag[face_id] = false;
                update_cavity_tables(face_id);
                update_relate_table_and_rank_local(face_id, cavity_deleted_queue);
                --count;
                after_euler = eular_characteristic();
                if (after_euler != before_euler) {
                    std::cerr << "Final euler mismatch: " << face_id << "\n";
                    //cnpy::npy_save("deleted_face_flag.npy", deleted_face_flag.data(), {deleted_face_flag.size()});
                    throw std::runtime_error("eular characteristic not equal after rollback!");
                }
            } else {
                deleted_face_map[face_id] = static_cast<int>(deleted_face_map.size());
            }
        }
    }

    std::vector<int> new_all_cavity_face_id;
    for (int fid : all_cavity_face_id) {
        if (!deleted_face_flag[fid]) new_all_cavity_face_id.push_back(fid);
    }
    all_cavity_face_id = new_all_cavity_face_id;

    auto results = compute_all_cavity_face_rank_global(all_cavity_face_id);
    MulPriorityQueue<int> deleted_queue;
    for (const auto& r : results) {
        deleted_queue.push(r.first, r.second);
    }

    while (!deleted_queue.is_empty()) {
        auto [face_id, priority_tuple] = deleted_queue.pop();
        if (priority_tuple[0] == std::numeric_limits<float>::infinity()) break;
        if (deleted_face_flag[face_id]) continue;

        ++count;
        deleted_face_flag[face_id] = true;
        update_cavity_tables(face_id);
        update_relate_table_and_rank_global(face_id, deleted_queue);

        int after_euler = eular_characteristic();
        if (after_euler != before_euler) {
            //std::cerr << face_id << " eular characteristic not equal\n";
            deleted_face_flag[face_id] = false;
            update_cavity_tables(face_id);
            update_relate_table_and_rank_global(face_id, deleted_queue);
            --count;
            after_euler = eular_characteristic();
            if (after_euler != before_euler) {
                std::cerr << "Final euler mismatch: " << face_id << "\n";
                //cnpy::npy_save("deleted_face_flag.npy", deleted_face_flag.data(), {deleted_face_flag.size()});
                throw std::runtime_error("eular characteristic not equal");
            }
        } else {
            deleted_face_map[face_id] = static_cast<int>(deleted_face_map.size());
        }
    }

    bool flag = (cavity_num() > 0 || has_avaliable_pocket(deleted_queue)!=-1) && !deleted_queue.is_empty();
    if (flag) {
        std::cout << "has_available_pocket: " << has_avaliable_pocket(deleted_queue) << "\n";
    }
    std::cout << "filter " << count << " face done\n";
    std::cout << "remaining cavity num: " << cavity_num() << "\n";
}

void Octree::update_relate_face_rank_global(int face_id,MulPriorityQueue<int>& queue)
{
    if (face_id < 0 || face_id >= static_cast<int>(m_mesh_f.size())) {
        return;
    }
    const std::vector<int>& face =m_mesh_f[static_cast<std::size_t>(face_id)];
    if (face.size() != 4) {
        throw std::runtime_error(
            "update_relate_face_rank_global: only quad faces are supported."
        );
    }
    const std::vector<std::pair<int, int>> edges =extract_edges(face);
    if (!deleted_face_flag(face_id)) {
        std::vector<float> priority_tuple =compute_face_rank_global(face_id);
        queue.update_priority(face_id, priority_tuple);
    }
    //1.Update the faces adjacent to the shared edge.
    for (const auto& edge : edges) {
        auto edge_it = edge_to_faces.find(edge);

        if (edge_it == edge_to_faces.end()) {
            continue;
        }

        const std::vector<int>& adjacent_faces = edge_it->second;

        for (int fid : adjacent_faces) {
            if (fid == face_id) {
                continue;
            }

            if (fid < 0 || fid >= deleted_face_flag.size()) {
                continue;
            }

            if (deleted_face_flag(fid)) {
                continue;
            }

            if (queue.contains(fid)) {
                std::vector<float> priority_tuple =compute_face_rank_global(fid);
                queue.update_priority(fid, priority_tuple);
            }
        }
    }

    //2. Update the faces in the same original cavity.
    if (face2cavity.find(face_id) != face2cavity.end()) {
        auto origin_it = origin_face2cavity.find(face_id);

        if (origin_it == origin_face2cavity.end()) {
            return;
        }

        const std::vector<int>& old_cavity_ids =
            origin_it->second;

        for (int old_cavity_id : old_cavity_ids) {
            if (old_cavity_id < 0 ||
                old_cavity_id >= static_cast<int>(cavity_face_ids.size()))
            {
                continue;
            }

            const std::vector<int>& cavity_faces =
                cavity_face_ids[static_cast<std::size_t>(old_cavity_id)];

            for (int fid : cavity_faces) {
                if (fid == face_id) {
                    continue;
                }

                if (fid < 0 || fid >= deleted_face_flag.size()) {
                    continue;
                }

                if (deleted_face_flag(fid)) {
                    continue;
                }

                if (queue.contains(fid)) {
                    std::vector<float> priority_tuple =
                        compute_face_rank_global(fid);

                    queue.update_priority(fid, priority_tuple);
                }
            }
        }
    }
}

void Octree::update_relate_face_rank_local(int face_id, MulPriorityQueue<int>& queue)
{
    if (face_id < 0 || face_id >= static_cast<int>(m_mesh_f.size())) return;

    const std::vector<int>& face = m_mesh_f[static_cast<std::size_t>(face_id)];
    if (face.size() != 4) {
        throw std::runtime_error("update_relate_face_rank_local: only quad faces supported");
    }
    const std::vector<std::pair<int, int>> edges =extract_edges(face);
    if (!deleted_face_flag(face_id)) {
        std::vector<float> priority_tuple = compute_face_rank_local(face_id);
        queue.update_priority(face_id, priority_tuple);
    }
    //Update the edge-adjacent faces.
    for (const auto& edge : edges) {
        auto it = edge_to_faces.find(edge);
        if (it == edge_to_faces.end()) continue;

        const std::vector<int>& adj_faces = it->second;
        for (int fid : adj_faces) {
            if (fid == face_id) continue;
            if (deleted_face_flag(fid)) continue;
            if (queue.contains(fid)) {
                std::vector<float> priority_tuple = compute_face_rank_local(fid);
                queue.update_priority(fid, priority_tuple);
            }
        }
    }

    //Update the other faces on the cavity.
    auto cavity_it = face2cavity.find(face_id);
    if (cavity_it != face2cavity.end()) {
        auto origin_it = origin_face2cavity.find(face_id);
        if (origin_it == origin_face2cavity.end()) return;

        const std::vector<int>& old_cavity_ids = origin_it->second;
        for (int old_cavity_id : old_cavity_ids) {
            if (old_cavity_id < 0 || old_cavity_id >= static_cast<int>(cavity_face_ids.size()))
                continue;

            const std::vector<int>& cavity_faces = cavity_face_ids[static_cast<std::size_t>(old_cavity_id)];
            for (int fid : cavity_faces) {
                if (fid == face_id) continue;
                if (deleted_face_flag(fid)) continue;
                if (queue.contains(fid)) {
                    std::vector<float> priority_tuple = compute_face_rank_local(fid);
                    queue.update_priority(fid, priority_tuple);
                }
            }
        }
    }
}

void Octree::update_relate_table_and_rank_global(
    int face_id,
    MulPriorityQueue<int>& queue)
{
    update_relate_face_edge_vertex_table(face_id);
    update_relate_face_rank_global(face_id, queue);
}

void Octree::update_relate_table_and_rank_local(
    int face_id,
    MulPriorityQueue<int>& queue)
{
    update_relate_face_edge_vertex_table(face_id);
    update_relate_face_rank_local(face_id, queue);
}

bool Octree::check_is_feature_bd_edge(
    const std::pair<int, int>& bd_edge) const
{
    const int vid0 = bd_edge.first;
    const int vid1 = bd_edge.second;

    const int leaf_node_index0 = m_mesh_v_to_leaf_node_indexes[static_cast<std::size_t>(vid0)];

    const int leaf_node_index1 = m_mesh_v_to_leaf_node_indexes[static_cast<std::size_t>(vid1)];


    if (feature_flags(leaf_node_index0) &&
        feature_flags(leaf_node_index1))
    {
        return true;
    }

    const OctreeNode& node0 = m_leaf_nodes[static_cast<std::size_t>(leaf_node_index0)].get();

    const OctreeNode& node1 = m_leaf_nodes[static_cast<std::size_t>(leaf_node_index1)].get();

    if (node0.is_cylinder || node1.is_cylinder) {
        return true;
    }

    return false;
}
void Octree::connect_edge_to_adj_node(int cur_vertex_idx, int adj_node_id0, int adj_node_id1) {
    auto process_adj_node = [&](int adj_node_id) {
        if (adj_node_id == -1) return;
        const OctreeNode& adj_node = m_leaf_nodes[static_cast<std::size_t>(adj_node_id)].get();
        int adj_vertex_idx = adj_node.m_vertex_idx;
        if (adj_vertex_idx == -1) return;

        std::pair<int, int> edge = make_sorted_edge(cur_vertex_idx, adj_vertex_idx);
        if (check_edge_exist(edge)) return;

        if (adj_node.is_cylinder) {
            alone_edges.insert(edge);
            add_edge_to_vertex_map(edge);
        } else if (m_v_type[adj_vertex_idx] != 2) {
            auto it = vertex_to_edge.find(adj_vertex_idx);
            if (it == vertex_to_edge.end()) {
                if (!adj_node.be_quoted_vertex_ids.empty()) {
                    int be_quoted_vertex_id = adj_node.be_quoted_vertex_ids[0];
                    alone_edges.insert(edge);
                    add_edge_to_vertex_map(edge);

                    std::pair<int, int> connect_edge = make_sorted_edge(be_quoted_vertex_id, adj_vertex_idx);
                    if (!check_edge_exist(connect_edge)) {
                        alone_edges.insert(connect_edge);
                        add_edge_to_vertex_map(connect_edge);
                    }
                }
            } else {
                alone_edges.insert(edge);
                add_edge_to_vertex_map(edge);
            }
        }
    };

    process_adj_node(adj_node_id0);
    process_adj_node(adj_node_id1);
}
void Octree::generate_alone_edges_for_cad() {
    if (convex_feature_points.rows() == 0) {
        return;
    }

    alone_edges.clear();

    for (int leaf_node_index : cylinder_node_ids) {
        const OctreeNode& node = m_leaf_nodes[static_cast<std::size_t>(leaf_node_index)].get();
        int cur_vertex_idx = node.m_vertex_idx;
        if (cur_vertex_idx == -1) continue;

        int node_pos_type = m_leaf_node_pos_types[leaf_node_index];
        int x_bits = node_pos_type & 0b000011;
        int y_bits = node_pos_type & 0b001100;
        int z_bits = node_pos_type & 0b110000;

        int adj_node_id0 = -1, adj_node_id1 = -1;

        // X-direction
        if (x_bits == 0b000011) {
            adj_node_id0 = m_leaf_nodes_adjacency(leaf_node_index, 0);
            adj_node_id1 = m_leaf_nodes_adjacency(leaf_node_index, 1);
            if (adj_node_id0 != -1 && adj_node_id1 != -1) {
                if (m_leaf_nodes[adj_node_id0].get().is_cylinder || m_leaf_nodes[adj_node_id1].get().is_cylinder) {
                    connect_edge_to_adj_node(cur_vertex_idx, adj_node_id0, adj_node_id1);
                }
            }
        }

        // Y-direction
        if (y_bits == 0b001100) {
            adj_node_id0 = m_leaf_nodes_adjacency(leaf_node_index, 2);
            adj_node_id1 = m_leaf_nodes_adjacency(leaf_node_index, 3);
            if (adj_node_id0 != -1 && adj_node_id1 != -1) {
                if (m_leaf_nodes[adj_node_id0].get().is_cylinder || m_leaf_nodes[adj_node_id1].get().is_cylinder) {
                    connect_edge_to_adj_node(cur_vertex_idx, adj_node_id0, adj_node_id1);
                }
            }
        }

        // Z-direction
        if (z_bits == 0b110000) {
            adj_node_id0 = m_leaf_nodes_adjacency(leaf_node_index, 4);
            adj_node_id1 = m_leaf_nodes_adjacency(leaf_node_index, 5);
            if (adj_node_id0 != -1 && adj_node_id1 != -1) {
                if (m_leaf_nodes[adj_node_id0].get().is_cylinder || m_leaf_nodes[adj_node_id1].get().is_cylinder) {
                    connect_edge_to_adj_node(cur_vertex_idx, adj_node_id0, adj_node_id1);
                }
            }
        }
    }
}

bool Octree::try_delete_face(int fid, int before_eular) {
    deleted_face_flag(fid) = true;

    update_relate_face_edge_vertex_table(fid);
    update_cavity_tables(fid);

    int after_eular = eular_characteristic();
    if (before_eular == after_eular) {
        return true;
    } else {
        // rollback
        deleted_face_flag(fid) = false;
        update_relate_face_edge_vertex_table(fid);
        update_cavity_tables(fid);

        after_eular = eular_characteristic();
        assert(before_eular == after_eular);
        return false;
    }
}

void Octree::update_face_edge_type_map(
    std::unordered_map<int, FaceEdgeType>& face_edge_type_map,
    int face_id)
{
    if (!deleted_face_flag(face_id)) return;

    std::set<int> related_face_ids;
    for (auto& edge : face_to_edge[face_id]) {
        for (auto& fid : edge_to_faces[edge]) {
            related_face_ids.insert(fid);
        }
    }

    for (int fid : related_face_ids) {
        if (fid == face_id) continue;

        std::vector<std::pair<int,int>> bd_edges_fid;
        std::vector<std::pair<int,int>> non_manifold_edges_fid;

        for (auto& edge_fid : face_to_edge[fid]) {
            auto& faces_for_edge = edge_to_faces[edge_fid];
            if (faces_for_edge.size() == 1) {
                bd_edges_fid.push_back(edge_fid);
            } else if (faces_for_edge.size() >= 3) {
                non_manifold_edges_fid.push_back(edge_fid);
            }
        }
        face_edge_type_map[fid] = {bd_edges_fid, non_manifold_edges_fid};
    }
}

int Octree::filter_isolate_patch_one_batch(
    std::unordered_map<int, FaceEdgeType>& face_edge_type_map)
{
    int filter_isolate_patch_count = 0;

    int cur_eular = eular_characteristic();

    const int face_count = static_cast<int>(m_mesh_f.size());

    for (int face_id = 0; face_id < face_count; ++face_id) {
        if (static_face_flag(face_id) &&origin_cavity_flags(face_id))
        {
            continue;
        }

        if (deleted_face_flag(face_id))
        {
            continue;
        }

        auto type_it = face_edge_type_map.find(face_id);
        if (type_it == face_edge_type_map.end()) {
            continue;
        }

        const std::vector<Edge>& bd_edges =type_it->second.first;

        const std::vector<Edge>& non_manifold_edges =type_it->second.second;

        const int boundary_count =static_cast<int>(bd_edges.size());

        const int non_manifold_count =static_cast<int>(non_manifold_edges.size());

        auto mark_deleted_success = [&]() {
            ++filter_isolate_patch_count;
            deleted_face_map[face_id] =static_cast<int>(deleted_face_map.size());
        };

        if (boundary_count == 4) {
            if (try_delete_face(face_id, cur_eular)) {
                mark_deleted_success();
            }

            update_face_edge_type_map(face_edge_type_map, face_id);

            cur_eular = eular_characteristic();
        }
        else if (boundary_count == 3 && non_manifold_count == 1) {
            if (try_delete_face(face_id, cur_eular)) {
                mark_deleted_success();
            }

            update_face_edge_type_map(face_edge_type_map, face_id);
        }
        else if (boundary_count == 3 && non_manifold_count == 0) {
            if (bd_edges.size() >= 3) {
                if (check_is_feature_bd_edge(bd_edges[0]) ||
                    check_is_feature_bd_edge(bd_edges[1]) ||
                    check_is_feature_bd_edge(bd_edges[2]))
                {
                    continue;
                }
            }

            if (try_delete_face(face_id, cur_eular)) {
                mark_deleted_success();
            }

            update_face_edge_type_map(face_edge_type_map, face_id);
        }
        else if (boundary_count == 2 && non_manifold_count == 2) {
            if (bd_edges.size() >= 2) {
                if (check_is_feature_bd_edge(bd_edges[0]) ||
                    check_is_feature_bd_edge(bd_edges[1]))
                {
                    continue;
                }
            }

            if (try_delete_face(face_id, cur_eular)) {
                mark_deleted_success();
            }

            update_face_edge_type_map(face_edge_type_map, face_id);
        }
        else if (boundary_count == 2 && non_manifold_count == 1) {
            if (bd_edges.size() >= 2) {
                if (check_is_feature_bd_edge(bd_edges[0]) ||
                    check_is_feature_bd_edge(bd_edges[1]))
                {
                    continue;
                }
            }

            if (try_delete_face(face_id, cur_eular)) {
                mark_deleted_success();
            }

            update_face_edge_type_map(face_edge_type_map, face_id);
        }
        else if (boundary_count == 1 && non_manifold_count == 3) {
            if (try_delete_face(face_id, cur_eular)) {
                mark_deleted_success();
            }

            update_face_edge_type_map(face_edge_type_map, face_id);
        }
        else if (boundary_count == 1 && non_manifold_count == 2) {
            if (bd_edges.empty() || non_manifold_edges.size() < 2) {
                continue;
            }

            const Edge& bd_edge = bd_edges[0];

            if (check_is_feature_bd_edge(bd_edge)) {
                continue;
            }

            {
                const Edge& non_manifold_edge = non_manifold_edges[0];
                const int vid0 = non_manifold_edge.first;
                const int vid1 = non_manifold_edge.second;

                auto edge_it = edge_to_faces.find(non_manifold_edge);
                const int edge_face_count =
                    edge_it == edge_to_faces.end()
                        ? 0
                        : static_cast<int>(edge_it->second.size());

                if (m_v_type[vid0] != static_cast<int>(DualPointType::PLANE) &&
                    m_v_type[vid1] != static_cast<int>(DualPointType::PLANE) &&
                    edge_face_count == 3)
                {
                    continue;
                }
            }

            {
                const Edge& non_manifold_edge = non_manifold_edges[1];
                const int vid0 = non_manifold_edge.first;
                const int vid1 = non_manifold_edge.second;

                auto edge_it = edge_to_faces.find(non_manifold_edge);
                const int edge_face_count =
                    edge_it == edge_to_faces.end()
                        ? 0
                        : static_cast<int>(edge_it->second.size());

                if (m_v_type[vid0] != static_cast<int>(DualPointType::PLANE) &&
                    m_v_type[vid1] != static_cast<int>(DualPointType::PLANE) &&
                    edge_face_count == 3)
                {
                    continue;
                }
            }

            if (try_delete_face(face_id, cur_eular)) {
                mark_deleted_success();
            }

            update_face_edge_type_map(face_edge_type_map, face_id);
        }
        else if (boundary_count == 1 && non_manifold_count == 1) {
            if (bd_edges.empty() || non_manifold_edges.empty()) {
                continue;
            }

            const Edge& bd_edge = bd_edges[0];

            if (face_id < static_cast<int>(m_mesh_f_area2.size()) &&
                m_mesh_f_area2[face_id] != 0.0f &&
                check_is_feature_bd_edge(bd_edge))
            {
                continue;
            }

            const Edge& non_manifold_edge = non_manifold_edges[0];
            const int vid0 = non_manifold_edge.first;
            const int vid1 = non_manifold_edge.second;

            auto edge_it = edge_to_faces.find(non_manifold_edge);
            const int edge_face_count =
                edge_it == edge_to_faces.end()? 0:static_cast<int>(edge_it->second.size());

            if (m_v_type[vid0] != static_cast<int>(DualPointType::PLANE) &&
                m_v_type[vid1] != static_cast<int>(DualPointType::PLANE) &&
                edge_face_count == 3)
            {
                continue;
            }

            if (try_delete_face(face_id, cur_eular)) {
                mark_deleted_success();
            }

            update_face_edge_type_map(face_edge_type_map, face_id);
        }
    }

    return filter_isolate_patch_count;
}

void Octree::post_process()
{
    const bool has_feature =convex_feature_points.rows() > 0;

    // ============================================================
    // Feature-aware local cleanup
    // ============================================================
    if (has_feature) {
        // Python:
        // self.edge_to_faces,self.vertex_to_edge =
        //     make_dict_edge2face(self.m_mesh_f,self.deleted_face_flag)
        std::tie(edge_to_faces, std::ignore) =
            make_dict_edge2face(m_mesh_f, deleted_face_flag);

        std::unordered_map<int, FaceEdgeType> face_edge_type_map;

        const int face_count =static_cast<int>(m_mesh_f.size());

        for (int face_id = 0; face_id < face_count; ++face_id) {
            if (face_id < static_cast<int>(static_face_flag.size()) &&
                face_id < static_cast<int>(origin_cavity_flags.size()) &&
                static_face_flag(face_id) &&
                static_cast<bool>(origin_cavity_flags(face_id)))
            {
                continue;
            }

            if (face_id < static_cast<int>(deleted_face_flag.size()) &&
                deleted_face_flag(face_id))
            {
                continue;
            }

            std::vector<Edge> bd_edges;
            std::vector<Edge> non_manifold_edges;

            auto face_edge_it = face_to_edge.find(face_id);
            if (face_edge_it == face_to_edge.end()) {
                continue;
            }

            for (const Edge& edge : face_edge_it->second) {
                auto edge_face_it = edge_to_faces.find(edge);
                if (edge_face_it == edge_to_faces.end()) {
                    continue;
                }

                const int degree =static_cast<int>(edge_face_it->second.size());

                if (degree == 1) {
                    bd_edges.push_back(edge);
                } else if (degree >= 3) {
                    non_manifold_edges.push_back(edge);
                }
            }

            face_edge_type_map[face_id] =
                std::make_pair(
                    std::move(bd_edges),
                    std::move(non_manifold_edges)
                );
        }

        while (true) {
            int count = filter_isolate_patch_one_batch(face_edge_type_map);

            if (count == 0) {
                break;
            }
        }

        int cur_eular = eular_characteristic();

        for (int face_id = 0; face_id < face_count; ++face_id) {
            if (face_id < static_cast<int>(static_face_flag.size()) &&
                face_id < static_cast<int>(origin_cavity_flags.size()) &&
                static_face_flag(face_id) &&
                static_cast<bool>(origin_cavity_flags(face_id)))
            {
                continue;
            }

            if (face_id < static_cast<int>(deleted_face_flag.size()) &&
                deleted_face_flag(face_id))
            {
                continue;
            }

            auto type_it = face_edge_type_map.find(face_id);
            if (type_it == face_edge_type_map.end()) {
                continue;
            }

            const std::vector<Edge>& bd_edges =
                type_it->second.first;

            const std::vector<Edge>& non_manifold_edges =
                type_it->second.second;

            (void)non_manifold_edges;

            const int boundary_count =
                static_cast<int>(bd_edges.size());

            if (face_id < static_cast<int>(m_mesh_f_area2.size()) &&
                m_mesh_f_area2[static_cast<std::size_t>(face_id)] == 0.0f &&
                boundary_count > 0)
            {
                if (boundary_count == 1) {
                    const Edge& bd_edge = bd_edges[0];

                    const int vid0 = bd_edge.first;
                    const int vid1 = bd_edge.second;

                    if (vid0 >= 0 && vid0 < static_cast<int>(m_v_type.size()) &&
                        vid1 >= 0 && vid1 < static_cast<int>(m_v_type.size()))
                    {
                        if (m_v_type[static_cast<std::size_t>(vid0)] ==
                                static_cast<int>(DualPointType::CORNER) ||
                            m_v_type[static_cast<std::size_t>(vid1)] ==
                                static_cast<int>(DualPointType::CORNER))
                        {
                            continue;
                        }
                    }

                    if (vid0 < 0 ||
                        vid0 >= static_cast<int>(m_mesh_v_to_leaf_node_indexes.size()) ||
                        vid1 < 0 ||
                        vid1 >= static_cast<int>(m_mesh_v_to_leaf_node_indexes.size()))
                    {
                        continue;
                    }

                    const int leaf_node_index0 =
                        m_mesh_v_to_leaf_node_indexes[
                            static_cast<std::size_t>(vid0)
                        ];

                    const int leaf_node_index1 =
                        m_mesh_v_to_leaf_node_indexes[
                            static_cast<std::size_t>(vid1)
                        ];

                    if (leaf_node_index0 < 0 ||
                        leaf_node_index0 >= static_cast<int>(m_leaf_nodes.size()) ||
                        leaf_node_index1 < 0 ||
                        leaf_node_index1 >= static_cast<int>(m_leaf_nodes.size()))
                    {
                        continue;
                    }

                    const OctreeNode& node0 =
                        m_leaf_nodes[
                            static_cast<std::size_t>(leaf_node_index0)
                        ].get();

                    const OctreeNode& node1 =
                        m_leaf_nodes[
                            static_cast<std::size_t>(leaf_node_index1)
                        ].get();

                    if (node0.is_cylinder || node1.is_cylinder) {
                        continue;
                    }
                }

                if (try_delete_face(face_id, cur_eular)) {
                    deleted_face_map[face_id] =
                        static_cast<int>(deleted_face_map.size());
                }

                update_face_edge_type_map(face_edge_type_map, face_id);
                continue;
            }
        }

        make_vertex_edge_face_topology();
        generate_alone_edges_for_cad();
    }

    // ============================================================
    // Remove redundant connected components
    // ============================================================

    const int face_count =
        static_cast<int>(m_mesh_f.size());

    std::vector<int> origin_filtered_mesh_f_ids;
    origin_filtered_mesh_f_ids.reserve(static_cast<std::size_t>(face_count));

    for (int fid = 0; fid < face_count; ++fid) {
        if (fid < static_cast<int>(deleted_face_flag.size()) &&
            !deleted_face_flag(fid))
        {
            origin_filtered_mesh_f_ids.push_back(fid);
        }
    }

    std::vector<int> filtered_mesh_f_ids_local =origin_filtered_mesh_f_ids;

    // Python:
    // alone_edges_list = list(self.alone_edges)
    std::vector<std::pair<int,int>> alone_edges_list;
    alone_edges_list.reserve(alone_edges.size());

    for (const Edge& edge : alone_edges) {
        alone_edges_list.push_back(edge);
    }


    std::vector<bool> edge_flags(
        alone_edges_list.size(),
        true
    );

    std::vector<bool> local_face_marks(
        origin_filtered_mesh_f_ids.size(),
        true
    );

    third_deleted_face_flag = deleted_face_flag;

    std::vector<std::vector<int>> candidate_faces;
    candidate_faces.reserve(origin_filtered_mesh_f_ids.size());

    for (int fid : origin_filtered_mesh_f_ids) {
        candidate_faces.push_back(
            m_mesh_f[static_cast<std::size_t>(fid)]
        );
    }

    std::vector<std::vector<int>> redundant_filtered_faces;
    std::vector<Edge> redundant_filtered_edges;

    std::tie(
        redundant_filtered_faces,
        redundant_filtered_edges,
        local_face_marks,
        edge_flags
    ) = remove_redundant_patch(
        candidate_faces,
        alone_edges_list
    );

    (void)redundant_filtered_faces;
    (void)redundant_filtered_edges;

    filtered_mesh_f_ids_local.clear();
    filtered_mesh_f_ids_local.reserve(origin_filtered_mesh_f_ids.size());

    for (std::size_t i = 0; i < origin_filtered_mesh_f_ids.size(); ++i) {
        if (i < local_face_marks.size() && local_face_marks[i]) {
            filtered_mesh_f_ids_local.push_back(
                origin_filtered_mesh_f_ids[i]
            );
        }
    }

    // Python:
    // new_deleted_face_ids =
    //     np.setdiff1d(origin_filtered_mesh_f_ids, filtered_mesh_f_ids)
    // self.deleted_face_flag[new_deleted_face_ids] = True
    std::unordered_set<int> kept_face_set;
    kept_face_set.reserve(filtered_mesh_f_ids_local.size());

    for (int fid : filtered_mesh_f_ids_local) {
        kept_face_set.insert(fid);
    }

    for (int fid : origin_filtered_mesh_f_ids) {
        if (kept_face_set.find(fid) == kept_face_set.end()) {
            if (fid >= 0 && fid < deleted_face_flag.size()) {
                deleted_face_flag(fid) = true;
            }
        }
    }

    // Python:
    // self.deleted_alone_edge_flag = ~edge_flags
    deleted_alone_edge_flag =
        ArrayB1::Constant(
            static_cast<Eigen::Index>(edge_flags.size()),
            false
        );

    for (Eigen::Index i = 0;
         i < static_cast<Eigen::Index>(edge_flags.size());
         ++i)
    {
        deleted_alone_edge_flag(i) =!edge_flags[static_cast<std::size_t>(i)];
    }

    // Python:
    // self.filtered_mesh_f = np.array(self.m_mesh_f[filtered_mesh_f_ids])
    // self.filtered_mesh_f_ids = filtered_mesh_f_ids
    filtered_mesh_f.clear();
    filtered_mesh_f.reserve(filtered_mesh_f_ids_local.size());

    for (int fid : filtered_mesh_f_ids_local) {
        filtered_mesh_f.push_back(
            m_mesh_f[static_cast<std::size_t>(fid)]
        );
    }

    filtered_mesh_f_ids =
        filtered_mesh_f_ids_local;

    // ============================================================
    // Output filtered triangles
    // ============================================================

    filtered_mesh_f_tri.clear();
    filtered_mesh_f_tri.reserve(filtered_mesh_f_ids.size() * 2);

    for (int fid : filtered_mesh_f_ids) {
        const int tri_id0 = fid * 2;
        const int tri_id1 = fid * 2 + 1;
        if (tri_id0 < 0 || tri_id1 >= static_cast<int>(m_mesh_f_tri.size()))
        {
            continue;
        }
        filtered_mesh_f_tri.push_back(m_mesh_f_tri[static_cast<std::size_t>(tri_id0)]);
        filtered_mesh_f_tri.push_back(m_mesh_f_tri[static_cast<std::size_t>(tri_id1)]);
    }
}
void Octree::grow_by_max_depth(const MatF3&spoints,SDFProjectorFunc& cloud_sdf_grad_mask_func,float margin_threshold){   
    m_root_node.id=0;
    int cur_depth = 0;

    while(cur_depth <= m_max_depth){
        Eigen::VectorXi new_grid_indices;
        MatF3 new_grid_coords;
        std::tie(new_grid_indices, new_grid_coords) = get_samples_of_new_nodes_optimized();
        
        if(new_grid_indices.size()>0){
            Eigen::VectorXf new_grid_shape_sdfs;
            MatF3 new_grid_shape_sns;
            MatF3 new_grid_shape_projs;
            Eigen::VectorXi new_grid_shape_point_ids;
            ArrayB1 new_grid_shape_masks;
            std::tie(
                new_grid_shape_sdfs, new_grid_shape_sns, new_grid_shape_projs, new_grid_shape_point_ids, new_grid_shape_masks
            ) = cloud_sdf_grad_mask_func(new_grid_coords);
            set_new_grid_shape_data(
                new_grid_indices,
                new_grid_shape_sdfs,
                new_grid_shape_sns,
                new_grid_shape_projs,
                new_grid_shape_point_ids,
                new_grid_shape_masks
            );
        }   

        adaptive_subdivide_sqef(spoints,margin_threshold);

        cur_depth += 1;
    }
    build_spoint_id_maps();
    scan_occupied();
    compute_node_strict_contain_feature_ids();
}
void Octree::solve_and_locate_vertices(
    const MatF3& spoints,
    const ArrayB1& sp_convex_feature_mask,
    SDFProjectorFunc& cloud_sdf_grad_mask_func,
    const MatF3& centroids_coords,
    float singular_value_threshold)
{
    scan_occupied_by_index();
    build_leaf_nodes_adjacency();
    batch_pre_solve_sqef(
        spoints,
        sp_convex_feature_mask,
        cloud_sdf_grad_mask_func,
        centroids_coords,
        singular_value_threshold
    );

    batch_locate();
    dfs_relate();
}

void Octree::build_mesh_and_topology()
{
    generate_mesh();

    scan_cavity();
    backup_cavity_data();

    initial_topology_data();
    make_vertex_edge_face_topology();
    make_cavity_adjacency_dict();

    generate_alone_edges();
}

void Octree::compute_face_udf_and_triangulate(
    SDFProjectorFunc& projFunc)
{
    MatF3 points=vector_vec3f_to_matf3(m_mesh_v);
    MatI faces=vector_vector_int_to_mati(m_mesh_f);
    MatI faces_unique=vector_vector_int_to_mati(m_mesh_f_unique);
    m_mesh_f_udfs = batch_compute_face_udf_parallel(
        projFunc,
        points,
        faces,
        faces_unique,
        box_edge_length,
        norm_eps,
        position_eps,
        0.5f
    );

    triangulate_faces_by_udf();
}

void Octree::clean_and_post_process()
{
    first_deleted_face_flag = deleted_face_flag;

    update_cavity_data();
    make_vertex_edge_face_topology();
    make_cavity_adjacency_dict();

    clean_cavity_face();

    second_deleted_face_flag = deleted_face_flag;

    update_cavity_data();
    make_vertex_edge_face_topology();

    post_process();
}

py::dict Octree::visualize_octree_cells() const
{
    std::cout << "m_leaf_nodes.size():"
              << m_leaf_nodes.size()
              << std::endl;

    py::dict data;

    std::vector<Eigen::RowVector3f> common_points;
    std::vector<Eigen::VectorXi> common_cells;
    int common_cell_offset = 0;

    std::vector<Eigen::RowVector3f> empty_points;
    std::vector<Eigen::VectorXi> empty_cells;
    int empty_cell_offset = 0;

    std::map<int, int> leaf_to_empty;
    std::map<int, int> leaf_to_empty_inverse;
    std::map<int, int> leaf_to_inter;
    std::map<int, int> leaf_to_inter_inverse;

    std::vector<int> empty_cell_m_type;
    std::vector<int> common_cell_m_type;
    std::vector<int> node_cell_types;
    std::vector<int> flags;

    const int leaf_count =static_cast<int>(m_leaf_nodes.size());

    common_points.reserve(static_cast<std::size_t>(leaf_count) * 8);
    empty_points.reserve(static_cast<std::size_t>(leaf_count) * 8);

    common_cells.reserve(static_cast<std::size_t>(leaf_count));
    empty_cells.reserve(static_cast<std::size_t>(leaf_count));

    common_cell_m_type.reserve(static_cast<std::size_t>(leaf_count));
    empty_cell_m_type.reserve(static_cast<std::size_t>(leaf_count));
    node_cell_types.reserve(static_cast<std::size_t>(leaf_count));
    flags.reserve(static_cast<std::size_t>(leaf_count));

    for (int i = 0; i < leaf_count; ++i) {
        const OctreeNode& node =m_leaf_nodes[static_cast<std::size_t>(i)].get();

        node_cell_types.push_back(node.m_node_cell_type);

        const int depth_scale =1 << (m_max_depth + m_sampling_depth - node.m_depth);

        MatI3 node_corner_id =g_child_order * depth_scale;

        node_corner_id.rowwise() += node.m_min_corner_id.transpose();

        MatF3 node_corners =node_corner_id.cast<float>() *static_cast<float>(m_unit_edge_length);

        node_corners.rowwise() += m_min_corner.transpose();

        const int result_type =static_cast<int>(results[static_cast<std::size_t>(i)].m_type);
    
        if (node.m_node_type == "empty") {
            const int empty_cell_index =static_cast<int>(empty_cells.size());

            leaf_to_empty[i] = empty_cell_index;
            leaf_to_empty_inverse[empty_cell_index] = i;

            for (int r = 0; r < 8; ++r) {
                empty_points.emplace_back(node_corners.row(r));
            }

            Eigen::VectorXi cell =(g_child_2_vtk_order.array() + empty_cell_offset).matrix();
            empty_cells.push_back(cell);
            empty_cell_offset += 8;

            empty_cell_m_type.push_back(result_type);
        } else {
            flags.push_back(
                node.m_node_cell_type == 1 ? 0 : 1
            );

            const int common_cell_index =static_cast<int>(common_cells.size());

            leaf_to_inter[i] = common_cell_index;
            leaf_to_inter_inverse[common_cell_index] = i;

            for (int r = 0; r < 8; ++r) {
                common_points.emplace_back(node_corners.row(r));
            }

            Eigen::VectorXi cell =(g_child_2_vtk_order.array() + common_cell_offset).matrix();

            common_cells.push_back(cell);
            common_cell_offset += 8;

            common_cell_m_type.push_back(result_type);
        }
    }

    data["common_point_mat"] =vector_rowvec3f_to_matf3(common_points);

    data["common_cell_mat"] =vector_vectorxi_int_to_mati(common_cells);

    data["empty_point_mat"] =vector_rowvec3f_to_matf3(empty_points);

    data["empty_cell_mat"] =vector_vectorxi_int_to_mati(empty_cells);

    data["leaf_to_empty"] = leaf_to_empty;
    data["leaf_to_empty_inverse"] = leaf_to_empty_inverse;
    data["leaf_to_inter"] = leaf_to_inter;
    data["leaf_to_inter_inverse"] = leaf_to_inter_inverse;

    data["empty_cell_m_type"] =empty_cell_m_type;

    data["common_cell_m_type"] =common_cell_m_type;

    data["flags"] =flags;

    data["node_cell_types"] =node_cell_types;

    return data;
}

void Octree::save_ma_mesh(const std::string& savepath)const
{
    std::vector<std::pair<int,int>> alone_edges_list(alone_edges.begin(), alone_edges.end());

    std::vector<std::pair<int,int>> valid_edges;
    for (size_t i = 0; i < alone_edges_list.size(); ++i) {
        if (!deleted_alone_edge_flag(i)) {
            valid_edges.push_back(alone_edges_list[i]);
        }
    }

    //Add all triangle edges to the set.
    std::set<std::pair<int,int>> all_mat_edges(valid_edges.begin(), valid_edges.end());

    for (int i = 0; i < filtered_mesh_f_tri.size(); ++i) {
        int v0 = filtered_mesh_f_tri[i][0];
        int v1 = filtered_mesh_f_tri[i][1];
        int v2 = filtered_mesh_f_tri[i][2];

        auto edge01 = sorted_edge_key(v0, v1);
        auto edge02 = sorted_edge_key(v0, v2);
        auto edge12 = sorted_edge_key(v1, v2);

        all_mat_edges.insert(edge01);
        all_mat_edges.insert(edge02);
        all_mat_edges.insert(edge12);
    }
    std::vector<std::vector<int>> edges_vec;
    edges_vec.reserve(all_mat_edges.size());
    for (auto& e : all_mat_edges) {
        edges_vec.push_back({e.first, e.second});
    }
    MatF3 vertices=vector_vec3f_to_matf3(m_mesh_v);
    Eigen::VectorXf radii_vec=Eigen::Map<const Eigen::VectorXf> (m_mesh_v_radii.data(),static_cast<Eigen::Index>(m_mesh_v_radii.size()));
    save_mat_obj(savepath, vertices, radii_vec, edges_vec, filtered_mesh_f_tri);
}
void Octree::save_medial_mesh(std::string savepath)const{
    MatF3 vertices=vector_vec3f_to_matf3(m_mesh_v);
    std::vector<std::vector<int>> edges;
    edges.reserve(alone_edges.size());
    for(const std::pair<int,int> &edge:alone_edges)
    {
        edges.push_back({edge.first,edge.second});
    }
    save_obj(savepath, vertices,edges, filtered_mesh_f_tri);
}
    
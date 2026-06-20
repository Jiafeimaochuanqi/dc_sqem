#ifndef __OCTREE_NODE_H__
#define __OCTREE_NODE_H__
#include <optional>
#include <Eigen/Dense>
#include <vector>
#include <array>
#include <algorithm>
#include <memory>
#include <stdexcept>
#include <utility>
#include <string>

typedef Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> MatF;
typedef Eigen::Matrix<float, Eigen::Dynamic, 2, Eigen::RowMajor> MatF2;
typedef Eigen::Matrix<float, Eigen::Dynamic, 3, Eigen::RowMajor> MatF3;
typedef Eigen::Matrix<int, Eigen::Dynamic, 3, Eigen::RowMajor> MatI3;
typedef Eigen::Matrix<float, 1, 3, Eigen::RowMajor> RowVector3f;
typedef Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> MatI;
typedef Eigen::Array<bool, Eigen::Dynamic, 1> ArrayB1;
extern const MatI3 g_child_order;
struct ListNode
{
    int val = -1;
    ListNode* next = nullptr;
};
struct QuadKeyHash
{
    std::size_t operator()(const std::array<int, 4>& q) const noexcept
    {
        std::size_t h = 0;
        for (int v : q) {
            const std::size_t hv = std::hash<int>{}(v);
            h ^= hv + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
        }
        return h;
    }
};

inline std::array<int, 4> sorted_quad_key(const std::vector<int>& quad)
{
    if (quad.size() != 4) {
        throw std::invalid_argument("sorted_quad_key: quad size must be 4.");
    }

    std::array<int, 4> key = {
        quad[0], quad[1], quad[2], quad[3]
    };

    std::sort(key.begin(), key.end());
    return key;
}
inline std::array<int, 4> sorted_quad_key_from_array(
    const std::array<int, 4>& quad)
{
    std::array<int, 4> key = quad;
    std::sort(key.begin(), key.end());
    return key;
}
inline std::pair<int, int> sorted_edge_key(int a, int b)
{
    return (a < b) ? std::make_pair(a, b) : std::make_pair(b, a);
}

inline bool vector_contains(const std::vector<int>& v, int value)
{
    return std::find(v.begin(), v.end(), value) != v.end();
}
inline bool edge_vector_contains(
    const std::vector<std::pair<int, int>>& edges,
    const std::pair<int, int>& edge)
{
    return std::find(edges.begin(), edges.end(), edge) != edges.end();
}
inline void remove_value_from_vector(std::vector<int>& v, int value)
{
    v.erase(
        std::remove(v.begin(), v.end(), value),
        v.end()
    );
}
inline MatF3 vector_vec3f_to_matf3(
    const std::vector<Eigen::Vector3f>& points)
{
    MatF3 mat(static_cast<Eigen::Index>(points.size()), 3);

    for (Eigen::Index i = 0; i < mat.rows(); ++i) {
        mat.row(i) = points[static_cast<std::size_t>(i)].transpose();
    }

    return mat;
}

inline MatF3 vector_rowvec3f_to_matf3(
    const std::vector<Eigen::RowVector3f>& points)
{
    MatF3 mat(static_cast<Eigen::Index>(points.size()), 3);

    for (Eigen::Index i = 0; i < mat.rows(); ++i) {
        mat.row(i) = points[static_cast<std::size_t>(i)];
    }

    return mat;
}

inline MatI vector_vectorxi_int_to_mati(
    const std::vector<Eigen::VectorXi>& rows)
{
    if (rows.empty()) {
        return MatI(0, 0);
    }

    const Eigen::Index row_count =
        static_cast<Eigen::Index>(rows.size());

    const Eigen::Index col_count =
        static_cast<Eigen::Index>(rows[0].size());

    MatI mat(row_count, col_count);

    for (Eigen::Index i = 0; i < row_count; ++i) {
        const auto& row = rows[static_cast<std::size_t>(i)];

        if (static_cast<Eigen::Index>(row.size()) != col_count) {
            throw std::runtime_error(
                "vector_vector_int_to_mati: inconsistent row size."
            );
        }

        for (Eigen::Index j = 0; j < col_count; ++j) {
            mat(i, j) = row[static_cast<std::size_t>(j)];
        }
    }

    return mat;
}

inline MatI vector_vector_int_to_mati(
    const std::vector<std::vector<int>>& rows)
{
    if (rows.empty()) {
        return MatI(0, 0);
    }

    const Eigen::Index row_count =
        static_cast<Eigen::Index>(rows.size());

    const Eigen::Index col_count =
        static_cast<Eigen::Index>(rows[0].size());

    MatI mat(row_count, col_count);

    for (Eigen::Index i = 0; i < row_count; ++i) {
        const auto& row = rows[static_cast<std::size_t>(i)];

        if (static_cast<Eigen::Index>(row.size()) != col_count) {
            throw std::runtime_error(
                "vector_vector_int_to_mati: inconsistent row size."
            );
        }

        for (Eigen::Index j = 0; j < col_count; ++j) {
            mat(i, j) = row[static_cast<std::size_t>(j)];
        }
    }

    return mat;
}
class OctreeNode
{
public:
    //global unique node id
    int id=-1;
    //index of leaf nodes set
    int leaf_node_index=-1;
    // the parent node of the current node
    OctreeNode* m_parent=nullptr;

    //contain cloud point ids
    Eigen::VectorXi inside_surface_point_ids;

    //contain convex feature cloud point ids
    Eigen::VectorXi inside_convex_feature_ids;

    //strict contain convex feature cloud point ids
    Eigen::VectorXi strict_inside_convex_feature_ids;

    // the children nodes of the current node
    std::vector<std::unique_ptr<OctreeNode>> m_children;

    // the current depth
    int m_depth;

    // the grid id of the min and the max corners
    Eigen::Vector3i m_min_corner_id;
    Eigen::Vector3i m_max_corner_id;

    // the grid id of the centroid point
    Eigen::Vector3i m_centroid_id;
    
    // grid ids of the sampling data
    MatI3 m_sample_grids;

    // the edge length of the current node
    int m_edge_span;

    // vertex id in the reconstructed mesh
    int m_vertex_idx=-1;
    
    //index of the node quoted by the current node
    int quote_leaf_node_index=-1;

    //vertex id of the node quoted by the current node
    int quote_vertex_id=-1;
    //indexes of all the node quote the current node
    std::vector<int> be_quoted_leaf_node_indexes;
    //vertex ids of all the node quote the current node
    std::vector<int> be_quoted_vertex_ids;

    ArrayB1 cell_sample_mask;

    std::vector<std::vector<int>> groups;

    // node type: empty interval leaf
    std::string m_node_type="empty";

    // node cell type: 0 is empty,1 is virtual cell, 2 is real cell
    int m_node_cell_type=0;

    // reconstruction point
    std::optional<Eigen::Vector3f> m_reconstruct_point = std::nullopt;

    bool is_cylinder=false;

    int drop_stage = 0;
    int decision_pid = -1;
    int decision_sid = -1;

    OctreeNode(Eigen::Vector3i min_corner_id, Eigen::Vector3i max_corner_id, OctreeNode* parent = nullptr, int depth = 0);
    OctreeNode()
        : m_depth(0),
        m_min_corner_id(Eigen::Vector3i::Zero()),
        m_max_corner_id(Eigen::Vector3i::Zero()),
        m_centroid_id(Eigen::Vector3i::Zero()),
        m_edge_span(0)
    {}
    ~OctreeNode() = default;

    OctreeNode(const OctreeNode&) = delete;
    OctreeNode& operator=(const OctreeNode&) = delete;

    OctreeNode(OctreeNode&&) noexcept = default;
    OctreeNode& operator=(OctreeNode&&) noexcept = default;

    // subdive the current node
    void subdivide();

    inline MatI3 generate_sample_grids(const MatI3& sampling_pattern)
    {
        return sampling_pattern.rowwise() + m_min_corner_id.transpose();
    }
};
struct OctreeVisualData {
    std::vector<float> common_points;
    std::vector<int> common_cells;
    std::vector<float> empty_points;
    std::vector<int> empty_cells;
    std::vector<float> far_points;
    std::vector<int> far_cells;

    std::vector<int> empty_cell_m_type;
    std::vector<int> common_cell_m_type;
    std::vector<int> flags;
    std::vector<int> node_cell_types;

    std::vector<int> leaf_to_empty_keys;
    std::vector<int> leaf_to_empty_values;
    std::vector<int> leaf_to_inter_keys;
    std::vector<int> leaf_to_inter_values;
};
#endif // __OCTREE_NODE_H__
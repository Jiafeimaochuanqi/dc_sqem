#include "octree_node.h"

const MatI3 g_child_order = []
{
    MatI3 m(8, 3);
    m << 0, 0, 0,//0
    0, 0, 1, //1
    0, 1, 0, //2
    0, 1, 1, //3
    1, 0, 0, //4
    1, 0, 1, //5
    1, 1, 0, //6
    1, 1, 1;//7
    return m;
}();

OctreeNode::OctreeNode(Eigen::Vector3i min_corner_id, Eigen::Vector3i max_corner_id, OctreeNode* parent, int depth)
{
    m_parent = parent;
    m_depth = depth;

    m_min_corner_id = min_corner_id;
    m_max_corner_id = max_corner_id;

    m_centroid_id = (m_min_corner_id + m_max_corner_id) / 2;

    m_edge_span = max_corner_id[0] - min_corner_id[0];
    
    m_vertex_idx = -1;
    m_node_type = "empty";
    inside_surface_point_ids.resize(0);
    inside_convex_feature_ids.resize(0);
    strict_inside_convex_feature_ids.resize(0);
}


void OctreeNode::subdivide()
{
    if (!m_children.empty()) {
        throw std::invalid_argument("Node already has children!");
    }

    const int child_edge_span = m_edge_span / 2;
    const Eigen::Vector3i max_corner_id_offset = Eigen::Vector3i::Constant(child_edge_span);
    const int parent_id = id;

    m_children.reserve(8);

    for (int child_index = 0; child_index < 8; ++child_index) {
        const int child_id = parent_id * 8 + 1 + child_index;

        const Eigen::Vector3i offset = g_child_order.row(child_index).transpose();
        const Eigen::Vector3i child_min_corner_id = m_min_corner_id + offset * child_edge_span;
        const Eigen::Vector3i child_max_corner_id = child_min_corner_id + max_corner_id_offset;

        auto child = std::make_unique<OctreeNode>(
            child_min_corner_id,
            child_max_corner_id,
            this,
            m_depth + 1
        );
        child->id = child_id;

        m_children.push_back(std::move(child));
    }
}


#ifndef __MESH_UTILS_H__
#define __MESH_UTILS_H__

#include <Eigen/Dense>
#include <algorithm>
#include <array>
#include <cmath>
#include <fstream>
#include <limits>
#include <numeric>
#include <random>
#include <set>
#include <stdexcept>
#include <string>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>
#include <map>
#include "igl/doublearea.hpp"
typedef Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> MatF;
typedef Eigen::Matrix<float, Eigen::Dynamic, 3, Eigen::RowMajor> MatF3;
typedef Eigen::Matrix<float, Eigen::Dynamic, 1> VectorXf;
typedef Eigen::Matrix<float,1, Eigen::Dynamic, Eigen::RowMajor> RowVectorXf;
typedef Eigen::Matrix<int, Eigen::Dynamic, 1> VectorXi;
typedef Eigen::Matrix<int,1, Eigen::Dynamic, Eigen::RowMajor> RowVectorXi;
typedef Eigen::Matrix<float, 3, 1> Vector3f;
typedef Eigen::Matrix<float, 1, 3, Eigen::RowMajor> RowVector3f;
typedef Eigen::Matrix<int, 1, 3, Eigen::RowMajor> RowVector3i;
typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> MatD;
typedef Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor> MatD3;
typedef Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> MatI;
typedef Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> MatFC;
typedef Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> MatIC;
typedef Eigen::Matrix<bool, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> MatB;
typedef Eigen::Array<bool, Eigen::Dynamic, 1> ArrayB1;


inline int unique_label_count(const Eigen::VectorXi& labels)
{
    std::unordered_set<int> s;
    s.reserve(static_cast<std::size_t>(labels.size()));

    for (Eigen::Index i = 0; i < labels.size(); ++i) {
        s.insert(labels(i));
    }

    return static_cast<int>(s.size());
}

inline std::vector<std::vector<int>> clusters_to_spoint_ids(
    const std::vector<std::vector<int>>& clusters,
    const Eigen::VectorXi& shape_point_ids)
{
    std::vector<std::vector<int>> spoint_ids_cluster;
    spoint_ids_cluster.reserve(clusters.size());

    for (const auto& cluster : clusters) {
        std::unordered_set<int> spoint_ids;

        for (int cid : cluster) {
            if (cid >= 0 && cid < shape_point_ids.size()) {
                spoint_ids.insert(shape_point_ids(cid));
            }
        }

        std::vector<int> ids;
        ids.reserve(spoint_ids.size());

        for (int id : spoint_ids) {
            ids.push_back(id);
        }

        spoint_ids_cluster.push_back(std::move(ids));
    }

    return spoint_ids_cluster;
}

inline Eigen::VectorXi indices_where_true(const ArrayB1& mask)
{
    const int count = static_cast<int>(mask.count());
    Eigen::VectorXi ids(count);

    int k = 0;
    for (Eigen::Index i = 0; i < mask.size(); ++i) {
        if (mask(i)) {
            ids(k++) = static_cast<int>(i);
        }
    }

    return ids;
}
inline std::vector<std::vector<int>> clusters_to_vector(
    const std::map<int, std::vector<int>>& clusters_map)
{
    std::vector<std::vector<int>> clusters_vec;
    clusters_vec.reserve(clusters_map.size());
    for (const auto& kv : clusters_map) {
        clusters_vec.push_back(kv.second);
    }
    return clusters_vec;
} 

inline std::vector<std::vector<int>> clusters_from_labels_and_ids(
    const Eigen::VectorXi& labels,
    const Eigen::VectorXi& ids)
{
    std::map<int, std::vector<int>> label_to_cluster;

    for (Eigen::Index i = 0; i < labels.size(); ++i) {
        label_to_cluster[labels(i)].push_back(ids(i));
    }

    std::vector<std::vector<int>> clusters;
    clusters.reserve(label_to_cluster.size());

    for (auto& kv : label_to_cluster) {
        clusters.push_back(std::move(kv.second));
    }

    return clusters;
}

inline bool has_more_than_one_label(const Eigen::VectorXi& labels)
{
    if (labels.size() <= 1) {
        return false;
    }

    const int first = labels(0);

    for (Eigen::Index i = 1; i < labels.size(); ++i) {
        if (labels(i) != first) {
            return true;
        }
    }

    return false;
}

inline Eigen::VectorXi indices_from_mask(const ArrayB1& mask)
{
    const int count = static_cast<int>(mask.count());

    Eigen::VectorXi ids(count);

    int k = 0;
    for (Eigen::Index i = 0; i < mask.size(); ++i) {
        if (mask(i)) {
            ids(k++) = static_cast<int>(i);
        }
    }

    return ids;
}


struct PairHash {
    std::size_t operator()(const std::pair<int, int>& p) const noexcept {
        const std::size_t h1 = std::hash<int>{}(p.first);
        const std::size_t h2 = std::hash<int>{}(p.second);
        return h1 ^ (h2 + 0x9e3779b97f4a7c15ULL + (h1 << 6) + (h1 >> 2));
    }
};

using EdgeToFaces = std::unordered_map<std::pair<int, int>, std::vector<int>, PairHash>;
using VertexToEdges = std::unordered_map<int, std::vector<std::pair<int, int>>>;

inline std::pair<int, int> make_sorted_edge(int u, int v) {
    return (u <= v) ? std::pair<int, int>{u, v} : std::pair<int, int>{v, u};
}

// Python: indices_in_bbox(points, min_corner, max_corner)
// Uses Eigen array operations to build the inside mask. A final loop is still needed
// to compact variable-length indices into a VectorXi.
inline Eigen::VectorXi indices_in_bbox(
    const MatF3& points,
    const Vector3f& min_corner,
    const Vector3f& max_corner)
{
     const ArrayB1 inside =
        (points.col(0).array() >= min_corner(0)) &&
        (points.col(0).array() <= max_corner(0)) &&
        (points.col(1).array() >= min_corner(1)) &&
        (points.col(1).array() <= max_corner(1)) &&
        (points.col(2).array() >= min_corner(2)) &&
        (points.col(2).array() <= max_corner(2));
    const int count = static_cast<int>(inside.count());
    VectorXi ids(count);

    int k = 0;
    for (Eigen::Index i = 0; i < inside.size(); ++i) {
        if (inside(i)) {
            ids(k++) = static_cast<int>(i);
        }
    }
    return ids;
}

// Python: get_k_smallest_indices(arr, k)
inline VectorXi get_k_smallest_indices(const VectorXf& arr, int k) {
    if (k < 0) {
        throw std::invalid_argument("get_k_smallest_indices: k must be non-negative.");
    }

    const int n = static_cast<int>(arr.size());
    k = std::min(k, n);

    std::vector<int> ids(n);
    std::iota(ids.begin(), ids.end(), 0);

    std::stable_sort(ids.begin(), ids.end(), [&](int a, int b) {
        return arr(a) < arr(b);
    });

    if (k == 0) {
        return Eigen::VectorXi(0);
    }

    return Eigen::Map<const Eigen::VectorXi>(ids.data(), k);
}

// Python: box_intersection(box1, box2)
// C++ interface: pass min/max corners directly.
inline std::tuple<bool, Vector3f, Vector3f> box_intersection(
    const Vector3f& min1,
    const Vector3f& max1,
    const Vector3f& min2,
    const Vector3f& max2)
{
    const Vector3f intersect_min = min1.cwiseMax(min2);
    const Vector3f intersect_max = max1.cwiseMin(max2);
    const bool intersects = (intersect_min.array() <= intersect_max.array()).all();
    return std::make_tuple(intersects, intersect_min, intersect_max);
}

// Python: sample_points_on_edge(edge_vertices, n_samples)
inline MatF3 sample_points_on_edge(const MatF3& edge_vertices, int n_samples) {
    if (n_samples <= 0) {
        return MatF3(0, 3);
    }
    const Eigen::RowVector3f a = edge_vertices.row(0);
    const Eigen::RowVector3f b = edge_vertices.row(1);
    const Eigen::RowVector3f ab = b - a;

    Eigen::VectorXf t_values =
        Eigen::VectorXf::LinSpaced(
            n_samples,
            1.0f / static_cast<float>(n_samples + 1),
            static_cast<float>(n_samples) / static_cast<float>(n_samples + 1)
        );

    MatF3 samples(n_samples, 3);
    samples = t_values * ab + a.replicate(n_samples, 1);
    return samples;
}

// Python: extract_edges(face)
inline std::vector<std::pair<int, int>> extract_edges(const std::vector<int>& face) {
    std::vector<std::pair<int, int>> edges;
    const int n = static_cast<int>(face.size());
    edges.reserve(static_cast<std::size_t>(n));

    for (int i = 0; i < n; ++i) {
        const int u = face[static_cast<std::size_t>(i)];
        const int v = face[static_cast<std::size_t>((i + 1) % n)];
        edges.push_back(make_sorted_edge(u, v));
    }
    return edges;
}

inline std::vector<std::pair<int, int>> extract_edges(const RowVectorXi& face) {
    std::vector<std::pair<int, int>> edges;
    const int n = static_cast<int>(face.size());
    edges.reserve(static_cast<std::size_t>(n));

    for (int i = 0; i < n; ++i) {
        const int u = face(i);
        const int v = face((i + 1) % n);
        edges.push_back(make_sorted_edge(u, v));
    }
    return edges;
}

inline std::vector<std::pair<int, int>> extract_edges_from_face_row(const MatI& faces, Eigen::Index face_id) {
    return extract_edges(faces.row(face_id));
}

// Python: make_dict_edge2face(faces, deleted_flags=None)
inline std::pair<EdgeToFaces, VertexToEdges> make_dict_edge2face(
    const MatI& faces,
    const ArrayB1& deleted_flags=ArrayB1())
{
    if (deleted_flags.size() != 0 &&
        deleted_flags.size() != static_cast<std::size_t>(faces.rows())) {
        throw std::invalid_argument("make_dict_edge2face: deleted_flags size mismatch.");
    }

    std::unordered_map<std::pair<int, int>, std::set<int>, PairHash> edge_to_faces_set;
    std::unordered_map<int, std::set<std::pair<int, int>>> vertex_to_edge_set;

    for (Eigen::Index fid = 0; fid < faces.rows(); ++fid) {
        if (deleted_flags.size() != 0 && deleted_flags(fid)) {
            continue;
        }

        const auto edges = extract_edges_from_face_row(faces, fid);
        for (const auto& e : edges) {
            edge_to_faces_set[e].insert(static_cast<int>(fid));
            vertex_to_edge_set[e.first].insert(e);
            vertex_to_edge_set[e.second].insert(e);
        }
    }

    EdgeToFaces edge_to_faces;
    VertexToEdges vertex_to_edge;

    for (const auto& kv : edge_to_faces_set) {
        edge_to_faces[kv.first] = std::vector<int>(kv.second.begin(), kv.second.end());
    }
    for (const auto& kv : vertex_to_edge_set) {
        vertex_to_edge[kv.first] = std::vector<std::pair<int, int>>(kv.second.begin(), kv.second.end());
    }

    return {edge_to_faces, vertex_to_edge};
}

inline std::pair<EdgeToFaces, VertexToEdges> make_dict_edge2face(
    const std::vector<std::vector<int> >& faces,
    const ArrayB1& deleted_flags=ArrayB1())
{
    if (deleted_flags.size() != 0 &&
        deleted_flags.size() != static_cast<std::size_t>(faces.size())) {
        throw std::invalid_argument("make_dict_edge2face: deleted_flags size mismatch.");
    }

    std::unordered_map<std::pair<int, int>, std::set<int>, PairHash> edge_to_faces_set;
    std::unordered_map<int, std::set<std::pair<int, int>>> vertex_to_edge_set;

    for (size_t fid = 0; fid < faces.size(); ++fid) {
        if (deleted_flags.size() != 0 && deleted_flags(fid)) {
            continue;
        }

        const auto edges = extract_edges(faces[fid]);
        for (const auto& e : edges) {
            edge_to_faces_set[e].insert(static_cast<int>(fid));
            vertex_to_edge_set[e.first].insert(e);
            vertex_to_edge_set[e.second].insert(e);
        }
    }

    EdgeToFaces edge_to_faces;
    VertexToEdges vertex_to_edge;

    for (const auto& kv : edge_to_faces_set) {
        edge_to_faces[kv.first] = std::vector<int>(kv.second.begin(), kv.second.end());
    }
    for (const auto& kv : vertex_to_edge_set) {
        vertex_to_edge[kv.first] = std::vector<std::pair<int, int>>(kv.second.begin(), kv.second.end());
    }

    return {edge_to_faces, vertex_to_edge};
}

// Python: point_projection_on_plane(point, plane_point, plane_normal)
inline Vector3f point_projection_on_plane(
    const Vector3f& point,
    const Vector3f& plane_point,
    const Vector3f& plane_normal)
{
    const float n_norm = plane_normal.norm();
    if (n_norm == 0.0f) {
        throw std::invalid_argument("point_projection_on_plane: zero plane normal.");
    }
    const Vector3f n = plane_normal.stableNormalized();
    const float distance = (point - plane_point).dot(n);
    return point - distance * n;
}

// Python: point_projection_on_line(point, line_point, line_vector)
inline Vector3f point_projection_on_line(
    const Vector3f& point,
    const Vector3f& line_point,
    const Vector3f& line_vector)
{
    const float denom = line_vector.squaredNorm();
    if (denom == 0.0f) {
        throw std::invalid_argument("point_projection_on_line: zero line vector.");
    }
    const float t = (point - line_point).dot(line_vector) / denom;
    return line_point + t * line_vector;
}

// Python: sample_points_in_triangle(vertices, n)
inline MatF3 sample_points_in_triangle(
    const MatF3& vertices,
    int n,
    unsigned int seed = std::random_device{}())
{
    if (n <= 0) {
        return MatF3(0, 3);
    }

    MatF3 points(n,3);

    int random_count = n;
    int offset=0;
    if (n > 7) {
        points.topRows(3)=vertices;
        points.row(3)=(vertices.row(0) + vertices.row(1)) * 0.5f;
        points.row(4)=(vertices.row(0) + vertices.row(2)) * 0.5f;
        points.row(5)=(vertices.row(1) + vertices.row(2)) * 0.5f;
        points.row(6)=(vertices.row(0) + vertices.row(1) + vertices.row(2)) / 3.0f;
        random_count = n - 7;
        offset=7;
    }

    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);

    for (int i = 0; i < random_count; ++i) {
        const float r1 = dist(rng);
        const float r2 = dist(rng);
        const float sr1 = std::sqrt(r1);
        const float u = 1.0f - sr1;
        const float v = sr1 * (1.0f - r2);
        const float w = sr1 * r2;
        points.row(i+offset)=(u * vertices.row(0) + v * vertices.row(1) + w * vertices.row(2));
    }
    return points;
}



// Python: sample_points_in_quad(vertices, n)
inline MatF3 sample_points_in_quad(
    const MatF3& vertices,
    int n,
    unsigned int seed = std::random_device{}())
{
    if (n <= 0) {
        return MatF3(0, 3);
    }

    MatF3 tri0(3, 3);
    tri0.row(0) = vertices.row(0);
    tri0.row(1) = vertices.row(1);
    tri0.row(2) = vertices.row(2);

    MatF3 tri1(3, 3);
    tri1.row(0) = vertices.row(0);
    tri1.row(1) = vertices.row(2);
    tri1.row(2) = vertices.row(3);

    const float s0 = doublearea_single(tri0.row(0), tri0.row(1), tri0.row(2));
    const float s1 = doublearea_single(tri1.row(0), tri1.row(1), tri1.row(2));

    if (s0 + s1 == 0.0f) {
        return vertices;
    }

    const int n0 = static_cast<int>(std::lround((s0 * static_cast<float>(n)) / (s0 + s1)));
    const int n1 = n - n0;

    MatF3 p0 = sample_points_in_triangle(tri0, n0, seed);
    MatF3 p1 = sample_points_in_triangle(tri1, n1, seed + 1U);

    MatF3 out(p0.rows() + p1.rows(), 3);
    if (p0.rows() > 0) {
        out.topRows(p0.rows()) = p0;
    }
    if (p1.rows() > 0) {
        out.bottomRows(p1.rows()) = p1;
    }
    return out;
}

// Python: is_inside(point, min_corner, max_corner, offset=0.0)
inline bool is_inside(
    const Vector3f& point,
    const Vector3f& min_corner,
    const Vector3f& max_corner,
    float offset = 0.0f)
{
    const Vector3f offset_dir = max_corner - min_corner;
    const Vector3f ext_min = min_corner - offset_dir * offset;
    const Vector3f ext_max = max_corner + offset_dir * offset;
    return (point.array() <= ext_max.array()).all() &&
           (point.array() >= ext_min.array()).all();
}

// Python: line_aabb_intersection(point, dir, min_corner, max_corner, offset=0.0)
inline std::tuple<bool, Vector3f, Vector3f> line_aabb_intersection(
    const Vector3f& point,
    const Vector3f& dir,
    const Vector3f& min_corner,
    const Vector3f& max_corner,
    float offset = 0.0f)
{
    const Vector3f offset_dir = max_corner - min_corner;
    const Vector3f mn = min_corner - offset_dir * offset;
    const Vector3f mx = max_corner + offset_dir * offset;

    Eigen::Array3f t_min;
    Eigen::Array3f t_max;
    t_min.setConstant(-std::numeric_limits<float>::infinity());
    t_max.setConstant(std::numeric_limits<float>::infinity());

    bool intersects = true;

    for (int i = 0; i < 3; ++i) {
        if (dir(i) == 0.0f) {
            if (point(i) < mn(i) || point(i) > mx(i)) {
                intersects = false;
            }
            continue;
        }

        const float t1 = (mn(i) - point(i)) / dir(i);
        const float t2 = (mx(i) - point(i)) / dir(i);
        t_min(i) = std::min(t1, t2);
        t_max(i) = std::max(t1, t2);
    }

    const float t_enter = t_min.maxCoeff();
    const float t_exit = t_max.minCoeff();
    const bool final_intersects = intersects && (t_enter <= t_exit);
    const Vector3f enter_point = point + t_enter * dir;
    const Vector3f exit_point = point + t_exit * dir;
    return std::make_tuple(final_intersects, enter_point, exit_point);
}

// Python: point_to_plane_distance(point, plane_point, plane_normal)
inline float point_to_plane_distance(
    const Vector3f& point,
    const Vector3f& plane_point,
    const Vector3f& plane_normal)
{
    return std::abs(plane_normal.dot(point - plane_point));
}

// Python: plane_aabb_intersection(point, dir, min_corner, max_corner, offset=0.0, tol=1e-10)
inline MatF3 plane_aabb_intersection(
    const Vector3f& point,
    const Vector3f& dir,
    const Vector3f& min_corner,
    const Vector3f& max_corner,
    float offset = 0.0f,
    float tol = 1e-6f)
{
    const float norm = dir.norm();
    if (norm == 0.0f) {
        throw std::invalid_argument("plane_aabb_intersection: zero plane normal.");
    }
    const Vector3f norm_dir = dir.stableNormalized();

    const Vector3f offset_dir = max_corner - min_corner;
    const Vector3f mn = min_corner - offset_dir * offset;
    const Vector3f mx = max_corner + offset_dir * offset;
    //The plane exactly coincides with one of the faces of the AABB.
    for (int axis = 0; axis < 3; ++axis) {
        for (int side = 0; side < 2; ++side) {
            const float val = (side == 0) ? mn(axis) : mx(axis);

            if (std::abs(norm_dir(axis)) < 1.0f - tol) {
                continue;
            }

            Vector3f face_point = Vector3f::Zero();
            face_point(axis) = val;

            if (std::abs(dir.dot(point) - dir.dot(face_point)) < tol) {
                MatF3 vertices(4, 3);
                int row = 0;
                if (axis == 0) {
                    for (float y : {mn(1), mx(1)}) {
                        for (float z : {mn(2), mx(2)}) {
                            vertices.row(row++) = Vector3f(val, y, z).transpose();
                        }
                    }
                } else if (axis == 1) {
                    for (float x : {mn(0), mx(0)}) {
                        for (float z : {mn(2), mx(2)}) {
                            vertices.row(row++) = Vector3f(x, val, z).transpose();
                        }
                    }
                } else {
                    for (float x : {mn(0), mx(0)}) {
                        for (float y : {mn(1), mx(1)}) {
                            vertices.row(row++) = Vector3f(x, y, val).transpose();
                        }
                    }
                }
                return vertices;
            }
        }
    }

    std::vector<std::pair<Vector3f, Vector3f>> edges;
    edges.reserve(12);

    const std::array<float, 2> xs{mn(0), mx(0)};
    const std::array<float, 2> ys{mn(1), mx(1)};
    const std::array<float, 2> zs{mn(2), mx(2)};
    //The 12 edges of the AABB
    for (float y : ys) {
        for (float z : zs) {
            edges.push_back({Vector3f(mn(0), y, z), Vector3f(mx(0), y, z)});
        }
    }
    for (float x : xs) {
        for (float z : zs) {
            edges.push_back({Vector3f(x, mn(1), z), Vector3f(x, mx(1), z)});
        }
    }
    for (float x : xs) {
        for (float y : ys) {
            edges.push_back({Vector3f(x, y, mn(2)), Vector3f(x, y, mx(2))});
        }
    }

    std::vector<Vector3f> intersections;

    for (const auto& e : edges) {
        //dir⋅(p1+t*u−point)=0
        const Vector3f& p1 = e.first;
        const Vector3f& p2 = e.second;
        const Vector3f u = p2 - p1;
        const float denom = dir.dot(u);
        if (std::abs(denom) < tol) {
            //The line segment is parallel to the plane.
            continue;
        }
        //t=dir⋅(point-p1)/(dir⋅u)
        const float t = dir.dot(point - p1) / denom;
        if (t >= 0 && t <= 1.0f) {
            const Vector3f p = p1 + t * u;

            bool exists = false;
            for (const Vector3f& q : intersections) {
                if ((p - q).cwiseAbs().maxCoeff() <= tol) {
                    exists = true;
                    break;
                }
            }

            if (!exists) {
                intersections.push_back(p);
            }
        }
    }

    MatF3 out(static_cast<Eigen::Index>(intersections.size()), 3);
    for (Eigen::Index i = 0; i < out.rows(); ++i) {
        out.row(i) = intersections[static_cast<std::size_t>(i)].transpose();
    }
    return out;
}

// Python: save_obj(obj_filename, vertices, edges, triangles)
// Pass an empty (0,2) matrix for edges when there are no edges.
inline void save_obj(
    const std::string& obj_filename,
    const MatF3& vertices,
    const std::vector<std::vector<int>>& edges,
    const std::vector<std::vector<int>>& triangles)
{
    std::ofstream obj_file(obj_filename);
    if (!obj_file) {
        throw std::runtime_error("save_obj: failed to open output file.");
    }

    for (int i = 0; i < vertices.rows(); ++i) {
        obj_file << "v " << vertices(i,0) << " " << vertices(i,1) << " " << vertices(i,2) << "\n";
    }

    for (int i = 0; i < edges.size(); ++i) {
        if(edges[i].size()==2){
            obj_file << "l " << edges[i][0] + 1 << " " << edges[i][1] + 1 << "\n";
        }
    }
    

    for (int i = 0; i < triangles.size(); ++i) {
        if (triangles[i].size() == 3) {
            obj_file << "f " << triangles[i][0] + 1 << " " << triangles[i][1] + 1 << " " << triangles[i][2] + 1 << "\n";
        } else if (triangles[i].size() == 4) {
            obj_file << "f " << triangles[i][0] + 1 << " " << triangles[i][1] + 1 << " "
                     << triangles[i][2] + 1 << " " << triangles[i][3] + 1 << "\n";
        }
    }
}


inline void save_mat_obj(
    const std::string& filename,
    const MatF3& cpoint,
    const Eigen::VectorXf& radii,
    const std::vector<std::vector<int>>& cedges,
    const std::vector<std::vector<int>>& cfaces)
{
    const size_t n = cpoint.size();
    std::vector<bool> used(n, false);
    for (const auto& tri : cfaces) {
        for (int v : tri) {
            used[v] = true;
        }
    }
    for (const auto& edge : cedges) {
        for (int v : edge) {
            used[v] = true;
        }
    }
    std::unordered_map<int, int> old_to_new;
    int new_idx = 0;
    for (size_t i = 0; i < n; ++i) {
        if (used[i]) {
            old_to_new[static_cast<int>(i)] = new_idx++;
        }
    }
    std::vector<Eigen::Vector3f> new_vertices;
    std::vector<float> new_radii;
    new_vertices.reserve(new_idx);
    new_radii.reserve(new_idx);
    for (size_t i = 0; i < n; ++i) {
        if (used[i]) {
            new_vertices.push_back(cpoint.row(i));
            new_radii.push_back(radii(i));
        }
    }
    std::vector<std::vector<int>> new_triangles;
    new_triangles.reserve(cfaces.size());
    for (const auto& tri : cfaces) {
        std::vector<int> new_tri;
        new_tri.reserve(tri.size());
        for (int v : tri) {
            new_tri.push_back(old_to_new[v]);
        }
        new_triangles.push_back(new_tri);
    }

    //construct new edges
    std::vector<std::vector<int>> new_edges;
    new_edges.reserve(cedges.size());
    for (const auto& edge : cedges) {
        std::vector<int> new_edge;
        new_edge.reserve(edge.size());
        for (int v : edge) {
            new_edge.push_back(old_to_new[v]);
        }
        new_edges.push_back(new_edge);
    }

    //output
    std::ofstream fout(filename);
    if (!fout) {
        throw std::runtime_error("Failed to open file: " + filename);
    }

    fout << new_vertices.size() << " "
         << new_edges.size() << " "
         << new_triangles.size() << "\n";

    for (size_t i = 0; i < new_vertices.size(); ++i) {
        const auto& v = new_vertices[i];
        fout << "v " << v[0] << " " << v[1] << " " << v[2] << " "
             << new_radii[i] << "\n";
    }

    for (const auto& edge : new_edges) {
        fout << "e " << edge[0] << " " << edge[1] << "\n";
    }

    for (const auto& tri : new_triangles) {
        fout << "f " << tri[0] << " " << tri[1] << " " << tri[2] << "\n";
    }

    fout.close();
}

// Python: save_obj_with_normals(obj_filename, vertices, edges, triangles, normals)
inline void save_obj_with_normals(
    const std::string& obj_filename,
    const MatF3& vertices,
    const MatI& edges,
    const MatI& triangles,
    const MatF3& normals)
{
    std::ofstream obj_file(obj_filename);
    if (!obj_file) {
        throw std::runtime_error("save_obj_with_normals: failed to open output file.");
    }

    for (Eigen::Index i = 0; i < vertices.rows(); ++i) {
        obj_file << "v " << vertices(i, 0) << " " << vertices(i, 1) << " " << vertices(i, 2) << "\n";
    }

    // Match original Python code: writes "n" rather than standard OBJ "vn".
    for (Eigen::Index i = 0; i < normals.rows(); ++i) {
        obj_file << "n " << normals(i, 0) << " " << normals(i, 1) << " " << normals(i, 2) << "\n";
    }

    if (edges.cols() == 2) {
        for (Eigen::Index i = 0; i < edges.rows(); ++i) {
            obj_file << "l " << edges(i, 0) + 1 << " " << edges(i, 1) + 1 << "\n";
        }
    }

    for (Eigen::Index i = 0; i < triangles.rows(); ++i) {
        if (triangles.cols() == 3) {
            obj_file << "f " << triangles(i, 0) + 1 << " " << triangles(i, 1) + 1 << " " << triangles(i, 2) + 1 << "\n";
        } else if (triangles.cols() == 4) {
            obj_file << "f " << triangles(i, 0) + 1 << " " << triangles(i, 1) + 1 << " "
                     << triangles(i, 2) + 1 << " " << triangles(i, 3) + 1 << "\n";
        }
    }
}
#endif // __MESH_UTILS_H__
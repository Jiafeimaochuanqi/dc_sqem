#ifndef __SQEF_H__
#define __SQEF_H__
#define _USE_MATH_DEFINES
#include <iostream>
#include <Eigen/Dense>
#include <Eigen/SVD>
#include <cmath>
#include <optional>

enum class DualPointType : int {
    CORNER   = 0,
    EDGE     = 1,
    PLANE    = 2,
    NONETYPE = 4
};

inline std::string dual_point_type_to_string(DualPointType type)
{
    switch (type) {
        case DualPointType::CORNER:   return "CORNER";
        case DualPointType::EDGE:     return "EDGE";
        case DualPointType::PLANE:    return "PLANE";
        case DualPointType::NONETYPE: return "NONETYPE";
        default:                      return "UNKNOWN";
    }
}

class DualPoint {
public:
    // Python equivalent:
    //   m_point = None  <=>  m_point == std::nullopt
    //   m_point = p     <=>  m_point.has_value()
    std::optional<Eigen::Vector3f> m_point = std::nullopt;
    DualPointType m_type = DualPointType::NONETYPE;
    Eigen::Vector3f m_dir = Eigen::Vector3f::Zero();
    bool m_is_inside = false;
    float radius = 0.0f;

    DualPoint() = default;

    DualPoint(
        std::optional<Eigen::Vector3f> point,
        DualPointType type = DualPointType::NONETYPE,
        const Eigen::Vector3f& dir = Eigen::Vector3f::Zero(),
        bool is_inside = false,
        float r = 0.0f)
        : m_point(std::move(point)),
          m_type(type),
          m_dir(dir),
          m_is_inside(is_inside),
          radius(r)
    {}

    DualPoint(
        const Eigen::Vector3f& point,
        DualPointType type,
        const Eigen::Vector3f& dir = Eigen::Vector3f::Zero(),
        bool is_inside = false,
        float r = 0.0f)
        : m_point(point),
          m_type(type),
          m_dir(dir),
          m_is_inside(is_inside),
          radius(r)
    {}

    static DualPoint none(
        DualPointType type = DualPointType::NONETYPE,
        const Eigen::Vector3f& dir = Eigen::Vector3f::Zero(),
        bool is_inside = false)
    {
        return DualPoint(std::nullopt, type, dir, is_inside, 0.0f);
    }

    bool has_point() const
    {
        return m_point.has_value();
    }

    const Eigen::Vector3f& point() const
    {
        return m_point.value();
    }
};

typedef Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> MatF;
typedef Eigen::Matrix<float, Eigen::Dynamic, 3, Eigen::RowMajor> MatF3;
typedef Eigen::Matrix<int, Eigen::Dynamic, 3, Eigen::RowMajor> MatI3;
typedef Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> MatI;
typedef Eigen::Array<bool, Eigen::Dynamic, 1> ArrayB1;
class SQEF {
public:
    MatF3 normals;
    MatF3 shape_samples;
    MatF3 cell_samples;
    MatF A;
    Eigen::VectorXf b;
    bool is_plane = false;
    bool is_line = false;
    Eigen::Vector3f dir;
    Eigen::Vector4f point;
    // Default constructor
    SQEF() {}
    SQEF(const MatF3& normals , const MatF3& shape_samples,const MatF3& cell_samples);
    std::optional<Eigen::Vector4f> solve_with_group(std::vector<std::vector<int>>&groups,bool is_cylinder, double svd_threshold=0.01,bool debug=false);
    std::optional<Eigen::Vector4f> solve_in_corner(bool debug=false);
    std::optional<Eigen::Vector4f> solve_in_line(double svd_threshold = 0.01,bool debug = false);
    std::optional<Eigen::Vector4f> solve_in_plane(bool debug = false);
    std::optional<Eigen::Vector4f> solve(double svd_threshold = 0.01, bool debug = false);
    float residual(Eigen::Vector4f v){
        return (A*v-b).norm()/b.size();
    }

};


using AABBBox = std::pair<Eigen::Vector3f, Eigen::Vector3f>;
using AABBBoxList = std::vector<AABBBox>;

// Helper functions corresponding to the Python inner functions in
// solve_point_pre_sqef_with_group.
void sqef_clear_type_state(SQEF& sqef);

std::tuple<Eigen::Vector4f, Eigen::MatrixXf>
sqef_factor_once(SQEF& sqef);

float solve_x3(
    const MatF& A,
    const Eigen::VectorXf& b,
    const Eigen::Vector3f& x_known);

DualPoint solve_result_from_current_type(
    const SQEF& sqef,
    const Eigen::Vector3f& min_corner,
    const Eigen::Vector3f& max_corner,
    float corner_factor = 1e-3f,
    float edge_factor = 1e-3f,
    float plane_factor = 1e-3f);

DualPoint get_neighbor_solution_of_current_type(
    const SQEF& sqef,
    const Eigen::Vector3f& center,
    const AABBBoxList& neighbor_boxes);

void force_line(SQEF& sqef);
void force_plane(SQEF& sqef);

DualPoint solve_point_pre_sqef_with_group(
    SQEF& sqef,
    const Eigen::Vector3f& min_corner,
    const Eigen::Vector3f& max_corner,
    std::vector<std::vector<int>>& groups,
    bool is_cylinder,
    double svd_threshold,
    const AABBBoxList& one_ring_boxes,
    const AABBBoxList& two_ring_boxes,
    float corner_factor = 1e-3f,
    float edge_factor = 1e-3f,
    float plane_factor = 1e-3f);      



#endif // __SQEF_H__
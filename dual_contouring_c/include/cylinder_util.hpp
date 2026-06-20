#ifndef __CYLINDER_UTIL_H__
#define __CYLINDER_UTIL_H__
#define _USE_MATH_DEFINES
#include <cmath>
#include <Eigen/Dense>
#include <algorithm>
#include <limits>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>


using Mat3f = Eigen::Matrix<float, 3, 3>;
using MatF3 = Eigen::Matrix<float, Eigen::Dynamic, 3, Eigen::RowMajor>;
using VectorXf = Eigen::VectorXf;
using Vector3f = Eigen::Vector3f;
using RowVector3f = Eigen::RowVector3f;
using ArrayB1 = Eigen::Array<bool, Eigen::Dynamic, 1>;

// Return fields:
// (is_cylindrical_tinf, reason, axis, max_gap, num_raw_dirs,
//  num_used_dirs, used_filter, min_proj_norm, gap_threshold_deg)


struct CylinderDetectionResult {
    bool is_cylindrical_tinf = false;
    std::string reason = "";

    Vector3f axis = Vector3f::Zero();
    Vector3f evals = Vector3f::Zero();

    float max_gap = std::numeric_limits<float>::quiet_NaN();

    int num_valid_angle = 0;
    int num_raw_dirs = 0;
    int num_used_dirs = 0;

    bool used_filter = false;

    float min_proj_norm = 0.5f;
    float gap_threshold_deg = 165.0f;
};

inline std::tuple<MatF3, ArrayB1> normalize_rows(
    const MatF3& v,
    float eps = 1e-12f)
{
    const Eigen::Index n = v.rows();
    MatF3 out(n, 3);
    out.setZero();

    ArrayB1 valid(n);
    valid.setConstant(false);

    for (Eigen::Index i = 0; i < n; ++i) {
        const float norm = v.row(i).norm();
        if (norm > eps) {
            out.row(i) = v.row(i).stableNormalized();
            valid(i) = true;
        }
    }

    return std::make_tuple(out, valid);
}

inline MatF3 filter_valid_rows(const MatF3& v, const ArrayB1& valid)
{
    if (v.rows() != valid.size()) {
        throw std::invalid_argument("filter_valid_rows: size mismatch.");
    }

    Eigen::Index count = 0;
    for (Eigen::Index i = 0; i < valid.size(); ++i) {
        if (valid(i)) {
            ++count;
        }
    }

    MatF3 out(count, 3);
    Eigen::Index row = 0;
    for (Eigen::Index i = 0; i < valid.size(); ++i) {
        if (valid(i)) {
            out.row(row++) = v.row(i);
        }
    }

    return out;
}

inline float circular_max_gap(VectorXf angles)
{
    const Eigen::Index n = angles.size();

    if (n == 0) {
        return 2.0f * static_cast<float>(M_PI);
    }

    std::vector<float> a(static_cast<std::size_t>(n));
    for (Eigen::Index i = 0; i < n; ++i) {
        a[static_cast<std::size_t>(i)] = angles(i);
    }

    std::sort(a.begin(), a.end());

    if (a.size() == 1) {
        return 2.0f * static_cast<float>(M_PI);
    }

    float max_gap = 0.0f;
    for (std::size_t i = 1; i < a.size(); ++i) {
        max_gap = std::max(max_gap, a[i] - a[i - 1]);
    }

    const float wrap_gap =
        a.front() + 2.0f * static_cast<float>(M_PI) - a.back();

    return std::max(max_gap, wrap_gap);
}

inline std::tuple<Vector3f, Vector3f, Mat3f, Vector3f, Mat3f>
estimate_axis_from_unit_dirs(
    const MatF3& unit_dirs,
    float eps = 1e-12f)
{
    if (unit_dirs.cols() != 3) {
        throw std::invalid_argument(
            "estimate_axis_from_unit_dirs: unit_dirs must have shape (N, 3)."
        );
    }

    if (unit_dirs.rows() < 3) {
        throw std::invalid_argument(
            "estimate_axis_from_unit_dirs: too few directions."
        );
    }

    // mean_dir = unit_dirs.mean(axis=0)
    const Vector3f mean_dir = unit_dirs.colwise().mean().transpose();

    // centered = unit_dirs - mean_dir[None, :]
    MatF3 centered = unit_dirs;
    centered.rowwise() -= mean_dir.transpose();

    // cov = centered.T @ centered
    const Mat3f cov = centered.transpose() * centered;

    // evals, evecs = np.linalg.eigh(cov)
    Eigen::SelfAdjointEigenSolver<Mat3f> solver(cov);

    if (solver.info() != Eigen::Success) {
        throw std::runtime_error(
            "estimate_axis_from_unit_dirs: eigen decomposition failed."
        );
    }

    Vector3f evals = solver.eigenvalues();
    Mat3f evecs = solver.eigenvectors();

    // axis = evecs[:, 0]
    // smallest eigenvalue direction
    Vector3f axis = evecs.col(0).stableNormalized();

    if (axis(2) < 0.0f) {
        axis = -axis;
    }

    return std::make_tuple(axis, evals, evecs, mean_dir, cov);
}


// Return fields:
// (proj_unit, proj_norm, valid_mask)
inline std::tuple<MatF3, VectorXf, ArrayB1> project_dirs_to_axis_plane(
    const MatF3& dirs,
    const Vector3f& axis_in,
    float eps = 1e-12f)
{
    if (axis_in.norm()<eps){
        throw std::runtime_error(
            "zero axis."
        );
    }
    Vector3f axis = axis_in.stableNormalized();

    const Eigen::Index n = dirs.rows();
    MatF3 proj(n, 3);
    VectorXf proj_norm(n);
    ArrayB1 valid_mask(n);
    valid_mask.setConstant(false);

    for (Eigen::Index i = 0; i < n; ++i) {
        const Vector3f d = dirs.row(i).transpose();
        const Vector3f p = d - d.dot(axis) * axis;
        const float pn = p.norm();

        proj_norm(i) = pn;
        if (pn > eps) {
            proj.row(i) = (p / pn).transpose();
            valid_mask(i) = true;
        } else {
            proj.row(i).setZero();
        }
    }

    return std::make_tuple(proj, proj_norm, valid_mask);
}

// Return fields:
// (angles, e1, e2)
inline std::tuple<VectorXf, Vector3f, Vector3f> plane_angles_from_proj(
    const MatF3& proj_unit,
    const Vector3f& axis_in,
    float eps = 1e-12f)
{
    if (axis_in.norm()<eps){
        throw std::runtime_error(
            "zero axis."
        );
    }
    Vector3f axis = axis_in.stableNormalized();

    Vector3f tmp(1.0f, 0.0f, 0.0f);
    if (std::abs(tmp.dot(axis)) > 0.9f) {
        tmp = Vector3f(0.0f, 1.0f, 0.0f);
    }

    Vector3f e1 = tmp - tmp.dot(axis) * axis;
    e1 = e1.stableNormalized();

    Vector3f e2 = axis.cross(e1);
    e2 = e2.stableNormalized();

    VectorXf angles(proj_unit.rows());
    for (Eigen::Index i = 0; i < proj_unit.rows(); ++i) {
        const Vector3f p = proj_unit.row(i).transpose();
        const float x = p.dot(e1);
        const float y = p.dot(e2);
        float ang = std::atan2(y, x);
        if (ang < 0.0f) {
            ang += 2.0f * static_cast<float>(M_PI);
        }
        angles(i) = ang;
    }

    return std::make_tuple(angles, e1, e2);
}

inline CylinderDetectionResult is_cylindrical_tinf_node(
    const MatF3& dirs,
    bool torch_feature = false,
    float eps = 1e-12f,
    float min_proj_norm = 0.5f,
    float gap_threshold_deg = 165.0f)
{
    CylinderDetectionResult ret;
    ret.min_proj_norm = min_proj_norm;
    ret.gap_threshold_deg = gap_threshold_deg;

    MatF3 dirs_unit;
    ArrayB1 valid;
    std::tie(dirs_unit, valid) = normalize_rows(dirs, eps);

    MatF3 valid_dirs = filter_valid_rows(dirs_unit, valid);
    ret.num_raw_dirs = static_cast<int>(valid_dirs.rows());

    if (valid_dirs.rows() < 3) {
        ret.is_cylindrical_tinf = false;
        ret.reason = "too_few_valid_dirs_for_axis";
        return ret;
    }

    Vector3f axis = Vector3f::Zero();
    Vector3f evals = Vector3f::Zero();
    Mat3f evecs = Mat3f::Zero();
    Vector3f mean_dir = Vector3f::Zero();
    Mat3f cov = Mat3f::Zero();

    std::tie(axis, evals, evecs, mean_dir, cov) =
        estimate_axis_from_unit_dirs(valid_dirs, eps);

    ret.axis = axis;
    ret.evals = evals;

    if (evals(1) * 4.0f < evals(2)) {
        ret.is_cylindrical_tinf = false;
        ret.reason = "no_valid_axis";
        return ret;
    }

    if (!torch_feature && evals(0) * 20.0f > evals(1)) {
        ret.is_cylindrical_tinf = false;
        ret.reason = "no_valid_axis";
        return ret;
    }

    MatF3 proj_unit_all;
    VectorXf proj_norm_all;
    ArrayB1 valid_proj_mask;

    std::tie(proj_unit_all, proj_norm_all, valid_proj_mask) =
        project_dirs_to_axis_plane(valid_dirs, axis, eps);

    ArrayB1 valid_for_angle(valid_dirs.rows());
    valid_for_angle.setConstant(false);

    int num_valid_angle = 0;

    for (Eigen::Index i = 0; i < valid_dirs.rows(); ++i) {
        const bool ok =
            valid_proj_mask(i) &&
            proj_norm_all(i) > min_proj_norm;

        valid_for_angle(i) = ok;

        if (ok) {
            ++num_valid_angle;
        }
    }

    ret.num_valid_angle = num_valid_angle;

    const int required =
        std::max(static_cast<int>(dirs.rows() * 0.5f), 4);

    if (num_valid_angle < required) {
        ret.is_cylindrical_tinf = false;
        ret.reason = "too_few_valid_projections";
        return ret;
    }

    MatF3 use_dirs(num_valid_angle, 3);

    Eigen::Index row = 0;
    for (Eigen::Index i = 0; i < valid_dirs.rows(); ++i) {
        if (valid_for_angle(i)) {
            use_dirs.row(row++) = proj_unit_all.row(i);
        }
    }

    VectorXf angles;
    Vector3f e1;
    Vector3f e2;

    std::tie(angles, e1, e2) =
        plane_angles_from_proj(use_dirs, axis, eps);

    const float max_gap = circular_max_gap(angles);
    const float gap_threshold_rad =
        gap_threshold_deg * static_cast<float>(M_PI) / 180.0f;

    const bool is_cyl = max_gap < gap_threshold_rad;

    ret.is_cylindrical_tinf = is_cyl;
    ret.reason = "ok";
    ret.max_gap = max_gap;
    ret.num_used_dirs = static_cast<int>(use_dirs.rows());
    ret.used_filter = true;

    return ret;
}


#endif // __CYLINDER_UTIL_H__
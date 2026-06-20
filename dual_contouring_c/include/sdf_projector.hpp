#ifndef SDF_PROJECTOR_HPP
#define SDF_PROJECTOR_HPP
#define _USE_MATH_DEFINES
#include <cmath>
#include <Eigen/Dense>
#include <vector>
#include <memory>
#include <stdexcept>
#include <tuple>

#include "NanoKNNSearch.h"

typedef Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> MatF;
typedef Eigen::Matrix<float, Eigen::Dynamic, 3, Eigen::RowMajor> MatF3;
typedef Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> MatI;
typedef Eigen::Array<bool, Eigen::Dynamic, 1> ArrayB1;

class SDFProjectorFunc {
private:
    const MatF3 surface_points_;
    const MatF3 surface_points_ns_;

    ArrayB1 feature_mask_;         // sp_feature_mask
    ArrayB1 convex_feature_mask_;  // sp_convex_feature_mask

    bool has_feature_mask_ = false;
    //bool has_convex_feature_mask_ = false;

    // Python: ratio=np.sin(np.radians(85.0))
    float ratio_ = std::sin(85.0f * static_cast<float>(M_PI) / 180.0f);

    int search_threads_ = 4;
    std::unique_ptr<NanoKNNSearch<float>> full_knn_;

    std::unique_ptr<NanoKNNSearch<float>> non_convex_knn_;

    std::vector<int> non_convex_global_ids_;

public:
    SDFProjectorFunc(const MatF3& points,
                     const MatF3& normals,
                     const ArrayB1& feature_mask = ArrayB1(),
                     //const ArrayB1& convex_feature_mask = ArrayB1(),
                     float ratio = std::sin(85.0f * static_cast<float>(M_PI) / 180.0f),
                     int search_threads = 4)
        : surface_points_(points),
          surface_points_ns_(normals),
          ratio_(ratio),
          search_threads_(search_threads)
    {
        if (surface_points_.rows() <= 0) {
            throw std::invalid_argument("SDFProjectorFunc: points is empty.");
        }
        if (surface_points_.rows() != surface_points_ns_.rows()) {
            throw std::invalid_argument("SDFProjectorFunc: points / normals row mismatch.");
        }
        if (surface_points_.cols() != 3 || surface_points_ns_.cols() != 3) {
            throw std::invalid_argument("SDFProjectorFunc: points / normals must have 3 columns.");
        }
        if (search_threads_ <= 0) {
            throw std::invalid_argument("SDFProjectorFunc: search_threads must be positive.");
        }

        if (feature_mask.size() > 0) {
            if (feature_mask.rows() != surface_points_.rows()) {
                throw std::invalid_argument("SDFProjectorFunc: feature_mask size mismatch.");
            }
            feature_mask_ = feature_mask;
            has_feature_mask_ = true;
        } else {
            feature_mask_ = ArrayB1::Constant(surface_points_.rows(), false);
            has_feature_mask_ = false;
        }

        // if (convex_feature_mask.size() > 0) {
        //     if (convex_feature_mask.rows() != surface_points_.rows()) {
        //         throw std::invalid_argument("SDFProjectorFunc: convex_feature_mask size mismatch.");
        //     }
        //     convex_feature_mask_ = convex_feature_mask;
        //     has_convex_feature_mask_ = true;
        // } else {
        //     convex_feature_mask_ = ArrayB1::Constant(surface_points_.rows(), false);
        //     has_convex_feature_mask_ = false;
        // }

        full_knn_ = std::make_unique<NanoKNNSearch<float>>(
            surface_points_.data(),
            surface_points_.rows(),
            3,
            search_threads_);

        // if (has_convex_feature_mask_) {
        //     build_non_convex_knn();
        // }
    }

    std::tuple<Eigen::VectorXf, MatF3, MatF3, Eigen::VectorXi, ArrayB1>
    operator()(const MatF3& pts) const
    {
        Eigen::VectorXf sdfs;
        MatF3 normals;
        MatF3 nearest_points;
        Eigen::VectorXi indices;
        ArrayB1 feature_flags;

        if (has_feature_mask_) {
            compute_with_feature_mask(pts, sdfs, normals, nearest_points, indices, feature_flags);
        } else {
            compute_without_feature_mask(pts, sdfs, normals, nearest_points, indices, feature_flags);
        }

        return std::make_tuple(sdfs, normals, nearest_points, indices, feature_flags);
    }

private:
    void build_non_convex_knn()
    {
        non_convex_global_ids_.clear();
        non_convex_global_ids_.reserve(static_cast<size_t>(surface_points_.rows()));

        for (int i = 0; i < surface_points_.rows(); ++i) {
            if (!convex_feature_mask_(i)) {
                non_convex_global_ids_.push_back(i);
            }
        }

        if (non_convex_global_ids_.empty()) {
            throw std::runtime_error(
                "SDFProjectorFunc: all points are marked as convex feature points; cannot build non-convex knn.");
        }

        MatF3 non_convex_points = surface_points_(non_convex_global_ids_, Eigen::all).eval();

        non_convex_knn_ = std::make_unique<NanoKNNSearch<float>>(
            non_convex_points.data(),
            non_convex_points.rows(),
            3,
            search_threads_);
    }

    void query_nearest_indices(const MatF3& pts, Eigen::VectorXi& indices) const
    {
        if (pts.cols() != 3) {
            throw std::invalid_argument("SDFProjectorFunc::query_nearest_indices: pts must have 3 columns.");
        }

        const int n = pts.rows();
        indices.resize(n);

        if (n == 0) {
            return;
        }

        std::vector<int> nn_indices;
        std::vector<float> nn_dists_dummy;

        // if (has_convex_feature_mask_) {
        //     if (!non_convex_knn_) {
        //         throw std::runtime_error("SDFProjectorFunc: non_convex_knn_ is not initialized.");
        //     }

        //     non_convex_knn_->searchKNN(nn_indices, nn_dists_dummy, pts.data(), n, 3, 1);

        //     Eigen::Map<const Eigen::VectorXi> global_id_vec(
        //         non_convex_global_ids_.data(),
        //         static_cast<int>(non_convex_global_ids_.size())
        //     );

        //     indices = global_id_vec(nn_indices);
        // } else {
        //     full_knn_->searchKNN(nn_indices, nn_dists_dummy, pts.data(), n, 3, 1);
        //     indices = Eigen::Map<const Eigen::VectorXi>(nn_indices.data(), n);
        // }
        full_knn_->searchKNN(nn_indices, nn_dists_dummy, pts.data(), n, 3, 1);
        indices = Eigen::Map<const Eigen::VectorXi>(nn_indices.data(), n);
    }

    void compute_without_feature_mask(
        const MatF3& pts,
        Eigen::VectorXf& sdfs,
        MatF3& normals,
        MatF3& nearest_points,
        Eigen::VectorXi& indices,
        ArrayB1& feature_flags) const
    {
        if (pts.cols() != 3) {
            throw std::invalid_argument("SDFProjectorFunc::compute_without_feature_mask: pts must have 3 columns.");
        }

        const int n = pts.rows();
        sdfs.resize(n);
        normals.resize(n, 3);
        nearest_points.resize(n, 3);
        feature_flags = ArrayB1::Constant(n, false);

        if (n == 0) {
            indices.resize(0);
            return;
        }

        query_nearest_indices(pts, indices);

        MatF3 target_pts = surface_points_(indices, Eigen::all).eval();
        MatF3 target_ns  = surface_points_ns_(indices, Eigen::all).eval();

        MatF3 diff = target_pts - pts;
        Eigen::VectorXf proj = (diff.array() * target_ns.array()).rowwise().sum();

        sdfs = -proj;
        normals = target_ns;

        MatF3 std_nearest = pts.array() - target_ns.array().colwise() * sdfs.array();
        MatF3 offset = std_nearest - target_pts;

        Eigen::VectorXf dis_offset = offset.rowwise().norm();
        Eigen::VectorXf dis = diff.rowwise().norm();

        ArrayB1 need_clamp = dis_offset.array() > ratio_ * dis.array();
        nearest_points = need_clamp.replicate(1, 3).select(target_pts, std_nearest).eval();
    }

    void compute_with_feature_mask(
        const MatF3& pts,
        Eigen::VectorXf& sdfs,
        MatF3& normals,
        MatF3& nearest_points,
        Eigen::VectorXi& indices,
        ArrayB1& feature_flags) const
    {
        if (pts.cols() != 3) {
            throw std::invalid_argument("SDFProjectorFunc::compute_with_feature_mask: pts must have 3 columns.");
        }

        const int n = pts.rows();
        sdfs.resize(n);
        normals.resize(n, 3);
        nearest_points.resize(n, 3);
        feature_flags.resize(n);

        if (n == 0) {
            indices.resize(0);
            return;
        }

        query_nearest_indices(pts, indices);

        feature_flags = feature_mask_(indices).eval();

        MatF3 target_pts = surface_points_(indices, Eigen::all).eval();
        MatF3 target_ns  = surface_points_ns_(indices, Eigen::all).eval();

        MatF3 diff = target_pts - pts;
        Eigen::VectorXf proj = (diff.array() * target_ns.array()).rowwise().sum();
        Eigen::VectorXf dist = diff.rowwise().norm();

        Eigen::VectorXf base_sdf = -proj;
        MatF3 std_nearest = pts.array() - target_ns.array().colwise() * base_sdf.array();
        MatF3 offset = std_nearest - target_pts;
        Eigen::VectorXf dis_offset = offset.rowwise().norm();

        Eigen::ArrayXf sign = Eigen::ArrayXf::Zero(n);
        sign = (proj.array() > 0.0f).select(1.0f, sign);
        sign = (proj.array() < 0.0f).select(-1.0f, sign);
        Eigen::VectorXf feature_sdf = -dist.array() * sign;

        sdfs = feature_flags.select(feature_sdf, base_sdf).eval();

        normals = target_ns;

        ArrayB1 need_clamp_non_feature =
            (!feature_flags) && (dis_offset.array() > ratio_ * dist.array());

        nearest_points = std_nearest;
        nearest_points = need_clamp_non_feature.replicate(1, 3).select(target_pts, nearest_points).eval();
        nearest_points = feature_flags.replicate(1, 3).select(target_pts, nearest_points).eval();
    }
};

#endif // SDF_PROJECTOR_HPP
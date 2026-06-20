#include "plane_grouping/dbscan.hpp"
#define _USE_MATH_DEFINES
#include <Eigen/Dense>
#include <algorithm>
#include <cmath>
#include <map>
#include <queue>
#include <stdexcept>
#include <vector>

MatF computeSquaredDistanceMatrix(const MatF3& vertices) {
    int n = vertices.rows();
    
    Eigen::VectorXf sq_norms = vertices.rowwise().squaredNorm();
    
    /**
     * -2*A*A -2*A*B -2*A*C ... -2*A*E
     * -2*B*A -2*B*B -2*B*C ... -2*B*E
     * .
     * .
     * -2*E*A -2*E*B -2*E*C ... -2*E*E
     */
    MatF sq_distances = 
        (-2.0 * vertices) * vertices.transpose();
    
    /**
     * A^{2} A^{2} A^{2} ... A^{2}
     * B^{2} B^{2} B^{2}...B^{2}
     * .
     * .
     * E^{2} E^{2} E^{2}... E^{2}
     */
    sq_distances.colwise() += sq_norms;
    /**
     * A^{2}+A^{2} A^{2}+B^{2} A^{2}+C^{2} ... A^{2}+E^{2}
     * B^{2}+A^{2} B^{2}+B^{2} B^{2}+C^{2} ... B^{2}+E^{2}
     * .
     * .
     * E^{2}+A^{2} E^{2}+B^{2} E^{2}+C^{2}... E^{2}+E^{2}
     */
    sq_distances.rowwise() += sq_norms.transpose();
    
    sq_distances.diagonal().setZero();
    
    return sq_distances;
}

void radiusSearch(KDTree& kdtree,
                  const Eigen::Vector3f& query,
                  float radius,
                  std::vector<int>& indices) {    
    std::vector<nanoflann::ResultItem<int, float>> matches;
    nanoflann::SearchParameters params;
    
    float sq_radius = radius * radius;
    kdtree.radiusSearch(query.data(), sq_radius, matches, params);
    for (const auto& match : matches) {
        indices.push_back(match.first);
    }
}


FastDBSCAN::FastDBSCAN(
    double eps,
    int min_samples,
    const std::string& metric
)
    : eps_(eps),
      min_samples_(min_samples),
      metric_(metric)
{
    if (eps_ < 0.0) {
        throw std::invalid_argument("DBSCAN eps must be non-negative.");
    }

    if (min_samples_ <= 0) {
        throw std::invalid_argument("DBSCAN min_samples must be positive.");
    }

    if (metric_ != "euclidean" && metric_ != "cosine") {
        throw std::invalid_argument(
            "FastDBSCAN only supports metric='euclidean' or metric='cosine'."
        );
    }
}

Eigen::VectorXi FastDBSCAN::fit_predict(const MatF3& points)
{
    auto result = fit(points);
    return result.first;
}

std::pair<Eigen::VectorXi, std::map<int, std::vector<int>>>
FastDBSCAN::fit(const MatF3& points)
{
    if (metric_ == "euclidean") {
        auto result = clusterEuclidean(points);
        labels_ = result.first;
        clusters_ = result.second;
        return result;
    }

    if (metric_ == "cosine") {
        auto result = clusterCosine(points);
        labels_ = result.first;
        clusters_ = result.second;
        return result;
    }

    throw std::runtime_error("Unsupported metric.");
}

const Eigen::VectorXi& FastDBSCAN::labels() const
{
    return labels_;
}

const std::map<int, std::vector<int>>& FastDBSCAN::clusters() const
{
    return clusters_;
}

std::pair<Eigen::VectorXi, std::map<int, std::vector<int>>>
FastDBSCAN::clusterFromNeighbors(
    int n,
    const std::vector<std::vector<int>>& all_neighbors
) const
{
    Eigen::VectorXi labels(n);
    labels.fill(-1);

    std::map<int, std::vector<int>> clusters;

    if (n == 0) {
        return {labels, clusters};
    }

    std::vector<char> is_core(n, 0);

    for (int i = 0; i < n; ++i) {
        if (static_cast<int>(all_neighbors[i].size()) >= min_samples_) {
            is_core[i] = 1;
        }
    }

    int cluster_id = 0;

    for (int i = 0; i < n; ++i) {
        if (labels[i] != -1) {
            continue;
        }

        if (!is_core[i]) {
            continue;
        }

        labels[i] = cluster_id;

        std::queue<int> q;
        q.push(i);

        while (!q.empty()) {
            int u = q.front();
            q.pop();

            for (int v : all_neighbors[u]) {
                if (labels[v] != -1) {
                    continue;
                }

                labels[v] = cluster_id;

                if (is_core[v]) {
                    q.push(v);
                }
            }
        }

        ++cluster_id;
    }

    for (int i = 0; i < n; ++i) {
        clusters[labels[i]].push_back(i);
    }

    return {labels, clusters};
}

std::pair<Eigen::VectorXi, std::map<int, std::vector<int>>>
FastDBSCAN::clusterEuclidean(const MatF3& points) const
{
    const int n = static_cast<int>(points.rows());

    std::vector<std::vector<int>> all_neighbors(n);

    if (n == 0) {
        Eigen::VectorXi labels(0);
        std::map<int, std::vector<int>> clusters;
        return {labels, clusters};
    }

    if (n < 5000) {
        MatF squareDistances = computeSquaredDistanceMatrix(points);
        const double eps2 = eps_ * eps_;

        #pragma omp parallel for schedule(static)
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
                if (static_cast<double>(squareDistances(i, j)) <= eps2) {
                    all_neighbors[i].push_back(j);
                }
            }
        }
    } else {
        PointCloudAdaptor adaptor(points);
        KDTree index(3, adaptor, nanoflann::KDTreeSingleIndexAdaptorParams(10));
        index.buildIndex();

        #pragma omp parallel for schedule(static)
        for (int i = 0; i < n; ++i) {
            radiusSearch(index, points.row(i), static_cast<float>(eps_), all_neighbors[i]);
            std::sort(all_neighbors[i].begin(), all_neighbors[i].end());
        }
    }

    return clusterFromNeighbors(n, all_neighbors);
}

std::pair<Eigen::VectorXi, std::map<int, std::vector<int>>>
FastDBSCAN::clusterCosine(const MatF3& dirs) const
{
    const int n = static_cast<int>(dirs.rows());

    std::vector<std::vector<int>> all_neighbors(n);

    if (n == 0) {
        Eigen::VectorXi labels(0);
        std::map<int, std::vector<int>> clusters;
        return {labels, clusters};
    }

    MatF3 unit_dirs(n, 3);
    std::vector<char> valid(n, 0);

    const float norm_eps = 1e-12f;

    for (int i = 0; i < n; ++i) {
        float norm = dirs.row(i).norm();

        if (norm > norm_eps) {
            unit_dirs.row(i) = dirs.row(i) / norm;
            valid[i] = 1;
        } else {
            unit_dirs.row(i).setZero();
            valid[i] = 0;
        }
    }

    #pragma omp parallel for schedule(static)
    for (int i = 0; i < n; ++i) {
        all_neighbors[i].reserve(8);

        for (int j = 0; j < n; ++j) {
            double cosine_distance = 1.0;

            if (i == j) {
                cosine_distance = 0.0;
            } else if (valid[i] && valid[j]) {
                double cosv = static_cast<double>(unit_dirs.row(i).dot(unit_dirs.row(j)));

                if (cosv > 1.0) {
                    cosv = 1.0;
                } else if (cosv < -1.0) {
                    cosv = -1.0;
                }

                cosine_distance = 1.0 - cosv;
            }

            if (cosine_distance <= eps_) {
                all_neighbors[i].push_back(j);
            }
        }
    }

    return clusterFromNeighbors(n, all_neighbors);
}
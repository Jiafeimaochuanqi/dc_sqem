#ifndef DBSCAN_HPP
#define DBSCAN_HPP
#include <omp.h>
#include <Eigen/Dense>
#include <string>
#include <map>
#include <sstream>
#include <iomanip>
#include <nanoflann.hpp>
typedef Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> MatF;
typedef Eigen::Matrix<float, Eigen::Dynamic, 3, Eigen::RowMajor> MatF3;
MatF computeSquaredDistanceMatrix(const MatF3& vertices);
class PointCloudAdaptor {
    const MatF3& points_;
    
public:
    PointCloudAdaptor(const MatF3& points) : points_(points) {
        
    }
    
    inline size_t kdtree_get_point_count() const { return points_.rows(); }
    
    inline float kdtree_get_pt(const size_t idx, const size_t dim) const {
        return points_(idx,dim);
    }
    
    template <class BBOX>
    bool kdtree_get_bbox(BBOX&) const { return false; }
};

typedef nanoflann::KDTreeSingleIndexAdaptor<
        nanoflann::L2_Simple_Adaptor<float, PointCloudAdaptor>,
        PointCloudAdaptor, 
        3,
        int
        > KDTree;

void radiusSearch(KDTree& kdtree,
                  const Eigen::Vector3f& query,
                  float radius,
                  std::vector<int>& indices); 



class FastDBSCAN {
public:
    FastDBSCAN(
        double eps = 0.5,
        int min_samples = 5,
        const std::string& metric = "euclidean"
    );

    Eigen::VectorXi fit_predict(const MatF3& points);

    std::pair<Eigen::VectorXi, std::map<int, std::vector<int>>>
    fit(const MatF3& points);

    const Eigen::VectorXi& labels() const;

    const std::map<int, std::vector<int>>& clusters() const;

private:
    std::pair<Eigen::VectorXi, std::map<int, std::vector<int>>>
    clusterEuclidean(const MatF3& points) const;

    std::pair<Eigen::VectorXi, std::map<int, std::vector<int>>>
    clusterCosine(const MatF3& dirs) const;

    std::pair<Eigen::VectorXi, std::map<int, std::vector<int>>>
    clusterFromNeighbors(
        int n,
        const std::vector<std::vector<int>>& all_neighbors
    ) const;

private:
    double eps_;
    int min_samples_;
    std::string metric_;

    Eigen::VectorXi labels_;
    std::map<int, std::vector<int>> clusters_;
};

#endif // DBSCAN_HPP
#ifndef PLANE_GROUPING_H
#define PLANE_GROUPING_H
#define _USE_MATH_DEFINES
#include <cmath>
#include "plane.hpp"
#include <vector>
#include <memory>
#include <cmath>
#include <algorithm>
#include <random>
#include <set>
#include <map>
#include <iostream>
#include <stdexcept>
typedef Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> MatF;
typedef Eigen::Matrix<float, Eigen::Dynamic, 3, Eigen::RowMajor> MatF3;
typedef Eigen::Matrix<float, 1, 3, Eigen::RowMajor> RowVector3f;

Eigen::Matrix<float, Eigen::Dynamic, 3,Eigen::RowMajor> vectorToMatrixMap(
    const std::vector<Eigen::Vector3f>& points) ;

class PlaneGrouping {
private:
    double radian_threshold_;
    double position_eps_;
    double distance_threshold_;
    int min_samples_;
    bool refine_;
    bool group_normal_first_;
    double max_assignment_distance_;
    
    MatF3 points_;
    MatF3 normals_;

    mutable std::mt19937 rng_;
    
public:
    struct GroupResult {
        std::vector<std::vector<int>> initial_groups;
        std::vector<std::vector<int>> final_groups;
        std::vector<Plane> planes;
        void printStats() const {
            
            int total_points = 0;
            for (const auto& group : final_groups) {
                total_points += group.size();
            }
            std::cout << "total points: " << total_points << std::endl;
        }
    };
    
    PlaneGrouping(double angle_threshold = 5.0, 
                  double position_eps = 0.1,
                  double distance_threshold = 0.01,
                  int min_samples = 3,
                  bool refine = false,
                  bool group_normal_first = true,
                  double max_assignment_distance = -1.0);
    
    std::tuple<std::vector<std::vector<int>>,std::vector<std::vector<int>>,std::vector<Plane> > fit(const MatF3& points, 
                   const MatF3& normals,
                   bool need_merge = true,
                   bool debug = false);

    void setAngleThreshold(double angle_degrees) {
        radian_threshold_ = angle_degrees * M_PI / 180.0;
    }
    
    void setDistanceThreshold(double threshold) {
        distance_threshold_ = threshold;
    }
    
    void setMinSamples(int min_samples) {
        min_samples_ = min_samples;
    }
    
    void setPositionEps(double eps) {
        position_eps_ = eps;
    }
    
    void saveResults(const GroupResult& result, const std::string& filename) const;
    
private:
    bool arePointsCollinear(const std::vector<Eigen::Vector3f>& points, double tolerance = 1e-8) const;
    
    std::vector<std::vector<int>> groupByNormalsAndPositions(bool debug = false) const;
    std::vector<std::vector<int>> groupByPositionsAndNormals(bool debug = false) const;
    std::vector<std::vector<int>> refineGroupsByPosition(const std::vector<std::vector<int>>& initial_groups) const;
    std::vector<std::vector<int>> ransacPlaneFitting(const std::vector<Eigen::Vector3f>& points,
                                                    const std::vector<Eigen::Vector3f>& normals,
                                                    const std::vector<int>& indices,
                                                    int max_iterations = 100) const;
    std::shared_ptr<Plane> fitPlaneToPoints(const std::vector<Eigen::Vector3f>& points,
                                           const std::vector<Eigen::Vector3f>& normals) const;
    std::shared_ptr<Plane> fitPlaneToPointsFinal(const std::vector<Eigen::Vector3f>& points,
                                                const std::vector<Eigen::Vector3f>& normals) const;
    double pointToPlaneDistance(const Eigen::Vector3f& point, 
                               const Eigen::Vector3f& plane_normal, 
                               double plane_distance) const;
    void mergePlanes(const std::vector<std::vector<int>>& groups,
                    std::vector<std::vector<int>>& final_groups,
                    std::vector<Plane>& final_planes,
                    bool need_merge) const;
    bool isGroupOnPlane(const std::vector<int>& point_indices, const Plane& plane) const;
    void assignUngroupedPoints(std::vector<std::vector<int>>& groups,
                              std::vector<Plane>& planes,
                              bool debug) const;
};

#endif // PLANE_GROUPING_H
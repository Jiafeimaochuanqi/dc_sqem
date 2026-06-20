#include "plane_grouping/plane_grouping.hpp"
#include <Eigen/Dense>
#include <Eigen/Eigenvalues>
#include <random>
#include <fstream>
#include <iomanip>
#include "plane_grouping/dbscan.hpp"
Eigen::Matrix<float, Eigen::Dynamic, 3,Eigen::RowMajor> vectorToMatrixMap(
    const std::vector<Eigen::Vector3f>& points) {
    return Eigen::Map<const Eigen::Matrix<float, Eigen::Dynamic, 3, Eigen::RowMajor>>(
        reinterpret_cast<const float*>(points.data()),  
        points.size(),                                   
        3                                               
    );
}
PlaneGrouping::PlaneGrouping(double angle_threshold, 
                             double position_eps,
                             double distance_threshold,
                             int min_samples,
                             bool refine,
                             bool group_normal_first,
                             double max_assignment_distance)
    : rng_(std::random_device{}()) {
    
    radian_threshold_ = angle_threshold * M_PI / 180.0;
    distance_threshold_ = distance_threshold;
    min_samples_ = min_samples;
    position_eps_ = position_eps;
    refine_ = refine;
    group_normal_first_ = group_normal_first;
    
    if (max_assignment_distance < 0) {
        max_assignment_distance_ = distance_threshold * 2;
    } else {
        max_assignment_distance_ = max_assignment_distance;
    }
}

std::tuple<std::vector<std::vector<int>>,std::vector<std::vector<int>>,std::vector<Plane> > PlaneGrouping::fit(const MatF3& points, 
                                             const MatF3& normals,
                                             bool need_merge,
                                             bool debug) {
    
    if (points.size() != normals.size()) {
        throw std::invalid_argument("Points and normals must have the same size");
    }
    
    if (points.rows()==0) {
        throw std::invalid_argument("Points cannot be empty");
    }
    //std::cout<<"========step1========"<<std::endl;
    points_ = points;
    normals_ = normals.rowwise().normalized();
    //std::cout<<normals_<<std::endl;
    GroupResult result;
    
    if (group_normal_first_) {
        result.initial_groups = groupByNormalsAndPositions(debug);
    } else {
        result.initial_groups = groupByPositionsAndNormals(debug);
    }
    
    if (debug) {
        std::cout << "initial groups has: " << result.initial_groups.size() << " groups" << std::endl;
        for (size_t i = 0; i < result.initial_groups.size(); ++i) {
            std::cout << "group " << i << ": " << result.initial_groups[i].size() 
                     << " points" << std::endl;
            for(size_t j=0;j<result.initial_groups[i].size();j++){
                std::cout<<result.initial_groups[i][j]<<" ";
            }
            std::cout<<std::endl;
        }
    }
    
    if (refine_) {
        result.initial_groups = refineGroupsByPosition(result.initial_groups);
    }

    mergePlanes(result.initial_groups, result.final_groups, result.planes, need_merge);

    assignUngroupedPoints(result.final_groups, result.planes, debug);
    
    return std::make_tuple(result.initial_groups, result.final_groups, result.planes);
}

bool PlaneGrouping::arePointsCollinear(const std::vector<Eigen::Vector3f>& points, double tolerance) const {
    if (points.size() < 3) return true;
    
    Eigen::Vector3f p0 = points[0];
    Eigen::Vector3f direction = Eigen::Vector3f::Zero();
    
    for (size_t i = 1; i < points.size(); i++) {
        Eigen::Vector3f v = points[i] - p0;
        if (v.norm() > tolerance * 10) {
            direction = v;
            break;
        }
    }
    
    if (direction.norm() < tolerance) return true;
    
    for (size_t i = 1; i < points.size(); i++) {
        Eigen::Vector3f v = points[i] - p0;
        if (v.norm() > tolerance * 10) {
            Eigen::Vector3f cross = direction.cross(v);
            if (cross.norm() > tolerance) {
                return false;
            }
        }
    }
    
    return true;
}


std::vector<std::vector<int>> PlaneGrouping::groupByNormalsAndPositions(bool debug) const {
    std::vector<std::vector<int>> groups;
    
    // float normal_epsilon = 2.0 * sin(radian_threshold_ / 2.0);
    
    // if (debug) {
    //     std::cout << "normal group epsilon: " << normal_epsilon << std::endl;
    // }
    

    // Eigen::VectorXi normal_labels;
    // std::map<int, std::vector<int>> normal_clusters;
    // std::tie(normal_labels,normal_clusters)=FastDBSCAN::cluster(normals_, normal_epsilon, min_samples_);
    double cosine_eps = 1.0 - std::cos(radian_threshold_);

    if (debug) {
        std::cout << "normal group cosine epsilon: " << cosine_eps << std::endl;
    }

    FastDBSCAN normal_clustering(cosine_eps, min_samples_, "cosine");

    Eigen::VectorXi normal_labels;
    std::map<int, std::vector<int>> normal_clusters;

    std::tie(normal_labels, normal_clusters) = normal_clustering.fit(normals_);
    //std::cout<<normal_labels<<std::endl;
    
    for (const auto& cluster : normal_clusters) {
        int label = cluster.first;
        const std::vector<int>& global_indices = cluster.second;
        
        if (label == -1) {
            for (int idx : global_indices) {
                groups.push_back({idx});
            }
            continue;
        }
        
        if (global_indices.size() < min_samples_) {
            groups.push_back(global_indices);
            continue;
        }

        const Eigen::VectorXi global_eigen_indices = Eigen::Map<const Eigen::VectorXi>(global_indices.data(), global_indices.size());
        MatF3 selected_points = points_(global_eigen_indices, Eigen::all);

        Eigen::VectorXi position_labels;
        std::map<int, std::vector<int>> position_clusters;
        // std::tie(position_labels,position_clusters)  = FastDBSCAN::cluster(selected_points, position_eps_, min_samples_);
        FastDBSCAN position_clustering(position_eps_, min_samples_, "euclidean");

        std::tie(position_labels, position_clusters) =position_clustering.fit(selected_points);

        for (const auto& pos_cluster : position_clusters) {
            int pos_label = pos_cluster.first;
            const std::vector<int>& local_pos_indices = pos_cluster.second;
            
            if (pos_label == -1) {
                for (int local_idx : local_pos_indices) {
                    int idx=global_indices[local_idx];
                    groups.push_back({idx});
                }
            } else {
                if (local_pos_indices.size() >= min_samples_) {
                    const Eigen::VectorXi local_eigen_indices = Eigen::Map<const Eigen::VectorXi>(local_pos_indices.data(), local_pos_indices.size());
                    Eigen::VectorXi pos_eigen_indices=global_eigen_indices(local_eigen_indices);
                    std::vector<int> pos_indices(pos_eigen_indices.data(), pos_eigen_indices.data() + pos_eigen_indices.rows());
                    groups.push_back(pos_indices);
                } 
            }
        }
    }
    
    return groups;
}


std::vector<std::vector<int>> PlaneGrouping::groupByPositionsAndNormals(bool debug) const {
    std::vector<std::vector<int>> groups;
    
    Eigen::VectorXi position_labels;
    std::map<int, std::vector<int>> position_clusters;
    //std::tie(position_labels,position_clusters) = FastDBSCAN::cluster(points_, position_eps_, min_samples_);
    FastDBSCAN position_clustering(position_eps_, min_samples_, "euclidean");

    std::tie(position_labels, position_clusters) =position_clustering.fit(points_);

    for (const auto& cluster : position_clusters) {
        int label = cluster.first;
        const std::vector<int>& global_indices = cluster.second;
        
        if (label == -1) {
            for (int idx : global_indices) {
                groups.push_back({idx});
            }
            continue;
        }
        
        if (global_indices.size() < min_samples_) {
            groups.push_back(global_indices);
            continue;
        }

        const Eigen::VectorXi global_eigen_indices = Eigen::Map<const Eigen::VectorXi>(global_indices.data(), global_indices.size());
        MatF3 selected_normals = normals_(global_eigen_indices, Eigen::all);

        // double normal_epsilon = 2.0 * sin(radian_threshold_ / 2.0);
        // Eigen::VectorXi normal_labels;
        // std::map<int, std::vector<int>> normal_clusters;
        // std::tie(normal_labels,normal_clusters)=FastDBSCAN::cluster(selected_normals, normal_epsilon, min_samples_);
        double normal_epsilon = 1.0 - std::cos(radian_threshold_);

        if (debug) {
            std::cout << "normal group cosine epsilon: " << normal_epsilon << std::endl;
        }

        FastDBSCAN normal_clustering(normal_epsilon, min_samples_, "cosine");

        Eigen::VectorXi normal_labels;
        std::map<int, std::vector<int>> normal_clusters;

        std::tie(normal_labels, normal_clusters) = normal_clustering.fit(selected_normals);

        for (const auto& norm_cluster : normal_clusters) {
            int norm_label = norm_cluster.first;
            const std::vector<int>& local_norm_indices = norm_cluster.second;
            
            if (norm_label == -1) {
                for (int local_idx : local_norm_indices) {
                    int global_idx=global_indices[local_idx];
                    groups.push_back({global_idx});
                }
            } else {
                if (local_norm_indices.size() >= min_samples_) {
                    const Eigen::VectorXi local_eigen_indices = Eigen::Map<const Eigen::VectorXi>(local_norm_indices.data(), local_norm_indices.size());
                    Eigen::VectorXi normal_eigen_indices=global_eigen_indices(local_eigen_indices);
                    std::vector<int> normal_indices(normal_eigen_indices.data(), normal_eigen_indices.data() + normal_eigen_indices.rows());
                    groups.push_back(normal_indices);
                } 

            }
        }
    }
    
    return groups;
}

std::vector<std::vector<int>> PlaneGrouping::refineGroupsByPosition(const std::vector<std::vector<int>>& initial_groups) const {
    std::vector<std::vector<int>> refined_groups;
    
    for (const auto& group : initial_groups) {
        if (group.size() < min_samples_) continue;
        
        std::vector<Eigen::Vector3f> group_points;
        std::vector<Eigen::Vector3f> group_normals;
        for (int idx : group) {
            group_points.push_back(points_.row(idx));
            group_normals.push_back(normals_.row(idx));
        }

        auto ransac_groups = ransacPlaneFitting(group_points, group_normals, group);
        refined_groups.insert(refined_groups.end(), ransac_groups.begin(), ransac_groups.end());
    }
    
    return refined_groups;
}

std::vector<std::vector<int>> PlaneGrouping::ransacPlaneFitting(const std::vector<Eigen::Vector3f>& points,
                                                              const std::vector<Eigen::Vector3f>& normals,
                                                              const std::vector<int>& indices,
                                                              int max_iterations) const {
    
    if (points.size() < 3) {
        return {indices};
    }
    
    std::vector<Eigen::Vector3f> remaining_points = points;
    std::vector<Eigen::Vector3f> remaining_normals = normals;
    std::vector<int> remaining_indices = indices;
    
    std::vector<std::vector<int>> groups;
    
    std::uniform_int_distribution<int> dist(0, static_cast<int>(remaining_points.size()) - 1);
    
    while (remaining_points.size() >= min_samples_ && groups.size() <= 2) {
        std::shared_ptr<Plane> best_plane;
        std::vector<int> best_inlier_indices;
        
        for (int iter = 0; iter < max_iterations; ++iter) {
            if (remaining_points.size() < 3) break;

            std::set<int> sample_indices;
            while (sample_indices.size() < 3) {
                sample_indices.insert(dist(rng_));
            }
            
            std::vector<Eigen::Vector3f> sample_points;
            std::vector<Eigen::Vector3f> sample_normals;
            for (int idx : sample_indices) {
                sample_points.push_back(remaining_points[idx]);
                sample_normals.push_back(remaining_normals[idx]);
            }

            auto plane = fitPlaneToPoints(sample_points, sample_normals);
            if (!plane) continue;

            std::vector<int> inlier_indices;
            for (size_t i = 0; i < remaining_points.size(); ++i) {
                double distance = pointToPlaneDistance(remaining_points[i], 
                                                      plane->normal, 
                                                      plane->distance);
                if (distance < distance_threshold_) {
                    inlier_indices.push_back(remaining_indices[i]);
                }
            }
            
            if (inlier_indices.size() > best_inlier_indices.size()) {
                best_plane = plane;
                best_inlier_indices = inlier_indices;
            }
        }
        
        if (best_inlier_indices.size() >= min_samples_) {
            groups.push_back(best_inlier_indices);

            std::set<int> best_inlier_set(best_inlier_indices.begin(), 
                                         best_inlier_indices.end());
            std::vector<int> new_remaining_indices;
            std::vector<Eigen::Vector3f> new_remaining_points;
            std::vector<Eigen::Vector3f> new_remaining_normals;
            
            for (size_t i = 0; i < remaining_indices.size(); ++i) {
                if (best_inlier_set.find(remaining_indices[i]) == best_inlier_set.end()) {
                    new_remaining_indices.push_back(remaining_indices[i]);
                    new_remaining_points.push_back(remaining_points[i]);
                    new_remaining_normals.push_back(remaining_normals[i]);
                }
            }
            
            remaining_indices = std::move(new_remaining_indices);
            remaining_points = std::move(new_remaining_points);
            remaining_normals = std::move(new_remaining_normals);
        } else {
            break;
        }
    }
    
    if (!remaining_indices.empty()) {
        groups.push_back(remaining_indices);
    }
    
    return groups;
}

std::shared_ptr<Plane> PlaneGrouping::fitPlaneToPoints(const std::vector<Eigen::Vector3f>& points,
                                                     const std::vector<Eigen::Vector3f>& normals) const {
    if (points.empty()) return nullptr;
    
    if (points.size() == 1) {
        return std::make_shared<Plane>(normals[0], points[0]);
    }
    
    if (points.size() == 2) {
        auto plane = std::make_shared<Plane>(normals[0], points[0]);
        double distance = pointToPlaneDistance(points[1], plane->normal, plane->distance);
        if (distance > distance_threshold_) {
            return nullptr;
        }
        return plane;
    }

    if (arePointsCollinear(points)) {
        return nullptr;
    }

    Eigen::Vector3f centroid = Eigen::Vector3f::Zero();
    for (const auto& p : points) {
        centroid += p;
    }
    centroid /= points.size();
    
    Eigen::Matrix3f covariance = Eigen::Matrix3f::Zero();
    for (const auto& p : points) {
        Eigen::Vector3f centered = p - centroid;
        covariance += centered * centered.transpose();
    }
    covariance /= points.size();
    
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> solver(covariance);
    Eigen::Vector3f normal = solver.eigenvectors().col(0);
    
    if (normals.size() >= 3) {
        Eigen::Vector3f avg_normal = Eigen::Vector3f::Zero();
        for (const auto& n : normals) {
            avg_normal += n.normalized();
        }
        avg_normal.normalize();
        
        if (normal.dot(avg_normal) < 0) {
            normal = -normal;
        }
    }
    
    return std::make_shared<Plane>(normal, centroid);
}

std::shared_ptr<Plane> PlaneGrouping::fitPlaneToPointsFinal(const std::vector<Eigen::Vector3f>& points,
                                                          const std::vector<Eigen::Vector3f>& normals) const {
    return fitPlaneToPoints(points, normals);
}

double PlaneGrouping::pointToPlaneDistance(const Eigen::Vector3f& point, 
                                         const Eigen::Vector3f& plane_normal, 
                                         double plane_distance) const {
    return std::abs(plane_normal.dot(point) + plane_distance) / plane_normal.norm();
}

void PlaneGrouping::mergePlanes(const std::vector<std::vector<int>>& groups,
                              std::vector<std::vector<int>>& final_groups,
                              std::vector<Plane>& final_planes,
                              bool need_merge) const {
    
    if (groups.empty()) {
        final_groups.clear();
        final_planes.clear();
        return;
    }

    std::vector<std::shared_ptr<Plane>> group_planes(groups.size(),nullptr);
    //#pragma omp parallel for
    for (size_t i=0;i<groups.size();i++) {
        const std::vector<int>& group = groups[i];
        if (group.size() >= min_samples_) {
            std::vector<Eigen::Vector3f> group_points;
            std::vector<Eigen::Vector3f> group_normals;
            for (int idx : group) {
                group_points.push_back(points_.row(idx));
                group_normals.push_back(normals_.row(idx));
            }
            group_planes[i]=fitPlaneToPoints(group_points, group_normals);
        } 
    }
    
    if (!need_merge) {
        final_groups = groups;
        final_planes=std::vector<Plane>(group_planes.size(),Plane());
        for (size_t i=0;i< group_planes.size();i++) {
            auto& plane = group_planes[i];
            if (plane) {
                final_planes[i]=(*plane);
            }
        }
        return;
    }

    std::vector<int> merge_target(groups.size(), -1);
    for (size_t i = 0; i < groups.size(); ++i) {
        if (merge_target[i] != -1) continue;
        //if (!group_planes[i]) continue;
        
        for (size_t j = 0; j < groups.size(); ++j) {
            if (i == j) continue;
            if (!group_planes[j]) continue;
            if (merge_target[j] != -1) continue;
            
            if (isGroupOnPlane(groups[i], *group_planes[j])) {
                merge_target[i] = static_cast<int>(j);
                break;
            }
        }
    }

    std::vector<int> root_groups;
    for (size_t i = 0; i < groups.size(); ++i) {
        if (merge_target[i] == -1) {
            root_groups.push_back(static_cast<int>(i));
        }
    }

    //#pragma omp parallel for schedule(dynamic)
    for (int j = 0; j < root_groups.size(); ++j) {
        int root_idx = root_groups[j];
        std::vector<int> merged_indices = {root_idx};
        for (size_t i = 0; i < groups.size(); ++i) {
            if (merge_target[i] == root_idx) {
                merged_indices.push_back(static_cast<int>(i));
            }
        }
        
        std::vector<int> merged_group;
        for (int idx : merged_indices) {
            merged_group.insert(merged_group.end(), 
                               groups[idx].begin(), 
                               groups[idx].end());
        }
        
        if (merged_group.size() >= min_samples_) {
            std::vector<Eigen::Vector3f> merged_points;
            std::vector<Eigen::Vector3f> merged_normals;
            for (int idx : merged_group) {
                merged_points.push_back(points_.row(idx));
                merged_normals.push_back(normals_.row(idx));
            }
            
            auto plane = fitPlaneToPointsFinal(merged_points, merged_normals);
            if (plane) {
                //#pragma omp critical
                {
                    final_groups.push_back(merged_group);
                    final_planes.push_back(*plane);
                }
            }
        }
    }
}

bool PlaneGrouping::isGroupOnPlane(const std::vector<int>& point_indices, const Plane& plane) const {
    for (int idx : point_indices) {
        double distance = pointToPlaneDistance(points_.row(idx), 
                                              plane.normal, 
                                              plane.distance);
        if (distance > distance_threshold_) {
            return false;
        }
    }
    return true;
}

void PlaneGrouping::assignUngroupedPoints(std::vector<std::vector<int>>& groups,
                                        std::vector<Plane>& planes,
                                        bool debug) const {
    
    if (groups.empty()) return;
    
    std::set<int> grouped_indices;
    for (const auto& group : groups) {
        grouped_indices.insert(group.begin(), group.end());
    }
    
    std::vector<int> ungrouped_indices;
    for (size_t i = 0; i < points_.rows(); ++i) {
        if (grouped_indices.find(static_cast<int>(i)) == grouped_indices.end()) {
            ungrouped_indices.push_back(static_cast<int>(i));
        }
    }
    
    if (ungrouped_indices.empty()) return;
    
    if (debug) {
        std::cout << "assign " << ungrouped_indices.size() 
                 << " ungrouped points" << std::endl;
    }
    
    for (int idx : ungrouped_indices) {
        double min_distance = std::numeric_limits<double>::max();
        int best_group_idx = -1;
        
        for (size_t i = 0; i < planes.size(); ++i) {
            double distance = pointToPlaneDistance(points_.row(idx), 
                                                  planes[i].normal, 
                                                  planes[i].distance);
            if (distance < min_distance) {
                min_distance = distance;
                best_group_idx = static_cast<int>(i);
            }
        }
        
        if (best_group_idx >= 0 && min_distance <= max_assignment_distance_) {
            groups[best_group_idx].push_back(idx);
        } else {
            groups.push_back({idx});
            planes.push_back(Plane(normals_.row(idx), points_.row(idx)));
        }
    }
}

void PlaneGrouping::saveResults(const GroupResult& result, const std::string& filename) const {
    std::ofstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("unable to open file: " + filename);
    }
    
    file << "Plane Grouping Results\n";
    file << "=====================\n\n";
    
    file << "Initial Groups: " << result.initial_groups.size() << "\n";
    for (size_t i = 0; i < result.initial_groups.size(); ++i) {
        file << "  Group " << i << ": " << result.initial_groups[i].size() << " points\n";
    }
    
    file << "\nFinal Groups and Planes: " << result.final_groups.size() << "\n";
    for (size_t i = 0; i < result.final_groups.size(); ++i) {
        file << "  Group " << i << ":\n";
        file << "    Plane: " << result.planes[i].toString() << "\n";
        file << "    Points: " << result.final_groups[i].size() << "\n";
        file << "    Point indices: ";
        for (size_t j = 0; j < result.final_groups[i].size(); ++j) {
            file << result.final_groups[i][j];
            if (j < result.final_groups[i].size() - 1) file << ", ";
        }
        file << "\n";
    }
    
    file.close();
}
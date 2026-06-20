#ifndef TANGENT_HPP
#define TANGENT_HPP
#include <Eigen/Dense>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include "plane_grouping/plane_grouping.hpp"
#include "sdf_projector.hpp"


std::vector<int> get_k_smallest_indices_nth(const std::vector<float>& arr, int k);
MatF3 generate_linespace_points(const Eigen::Vector3f &lower_left,
                                const Eigen::Vector3f &upper_right,int n);
std::tuple<bool, std::vector<std::vector<int>>> 
compute_groups(const Eigen::VectorXf& sdfs,
                        const MatF3& sns,
                        const MatF3& coords,
                        const MatF3& nearest_points,
                        const Eigen::Array<bool, Eigen::Dynamic, 1>& feature_flags,
                        double norm_eps,
                        double position_eps,
                        double distance_threshold); 
std::pair<std::vector<int>, std::vector<std::vector<int>>> 
find_central_indices(const MatF3& vertices, const std::vector<std::vector<int>>& groups, int k = 1);
double compute_tangent_plane(MatF3& spoints,
                    MatF3& sp_normals,
                    const ArrayB1& sp_feature_mask,
                    Eigen::Vector3f& center,
                    double box_edge_length,double octree_norm_eps,double octree_position_eps);
double compute_tangent_plane_dis(SDFProjectorFunc& projfunc,
                    Eigen::Vector3f& center,double box_edge_length,
                    double octree_norm_eps,double octree_position_eps,double ratio=0.125);
MatF batch_compute_face_udf_parallel(SDFProjectorFunc& projfunc,MatF3&m_mesh_v,MatI& m_mesh_f,MatI& m_mesh_f_unique,double box_edge_length,
                    double octree_norm_eps,double octree_position_eps,double ratio=0.125);
#endif //TANGENT_HPP


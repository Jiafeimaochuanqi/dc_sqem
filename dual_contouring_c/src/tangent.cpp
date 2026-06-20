#include "tangent.hpp"
#include "sqef.h"
#include <unsupported/Eigen/KroneckerProduct>
#include <vector>
#include <algorithm>
#include <utility>
#include <tuple>

std::vector<int> get_k_smallest_indices_nth(const Eigen::VectorXf& vec, int k)
{
    std::vector<float> std_vec(vec.data(), vec.data() + vec.size());
    return get_k_smallest_indices_nth(std_vec,k);
}
std::vector<int> get_k_smallest_indices_nth(const std::vector<float>& arr, int k) {
    if (k <= 0) return {};
    if (k >= arr.size()) {
        k = arr.size();
    }
    
    std::vector<std::pair<float, int>> indexed_arr;
    indexed_arr.reserve(arr.size());
    
    for (size_t i = 0; i < arr.size(); ++i) {
        indexed_arr.emplace_back(arr[i], i);
    }
    
    std::nth_element(indexed_arr.begin(), indexed_arr.begin() + k, indexed_arr.end(),
                     [](const auto& a, const auto& b) {
                         if (a.first != b.first) return a.first < b.first;
                         return a.second < b.second;
                     });

    std::sort(indexed_arr.begin(), indexed_arr.begin() + k,
              [](const auto& a, const auto& b) {
                  if (a.first != b.first) return a.first < b.first;
                  return a.second < b.second;
              });
    
    std::vector<int> result(k);
    for (int i = 0; i < k; ++i) {
        result[i] = indexed_arr[i].second;
    }
    
    return result;
}

MatF3 generate_linespace_points(
    const Eigen::Vector3f& lower_left,
    const Eigen::Vector3f& upper_right,
    int n
) {
    if (n <= 0) throw std::invalid_argument("n must be positive");

    const Eigen::VectorXf x = Eigen::VectorXf::LinSpaced(n, lower_left.x(), upper_right.x());
    const Eigen::VectorXf y = Eigen::VectorXf::LinSpaced(n, lower_left.y(), upper_right.y());
    const Eigen::VectorXf z = Eigen::VectorXf::LinSpaced(n, lower_left.z(), upper_right.z());

    const int n2 = n * n;
    const int N  = n * n * n;

    const Eigen::VectorXf ones_n  = Eigen::VectorXf::Ones(n);
    const Eigen::VectorXf ones_n2 = Eigen::VectorXf::Ones(n2);

    Eigen::VectorXf Z = Eigen::kroneckerProduct(ones_n2, z);
    Eigen::VectorXf Y = Eigen::kroneckerProduct(ones_n, Eigen::kroneckerProduct(y, ones_n));
    Eigen::VectorXf X = Eigen::kroneckerProduct(x, ones_n2);

    MatF3 V(N, 3);
    V.col(0) = X;
    V.col(1) = Y;
    V.col(2) = Z;
    return V; // (n^3, 3)
}

Eigen::VectorXf np_sign(const Eigen::VectorXf& arr) {
    return arr.array().sign();
}

std::tuple<bool, std::vector<std::vector<int>>> 
compute_groups(const Eigen::VectorXf& sdfs,
                        const MatF3& sns,
                        const MatF3& coords,
                        const MatF3& nearest_points,
                        const Eigen::Array<bool, Eigen::Dynamic, 1>& feature_flags,
                        double norm_eps,
                        double position_eps,
                        double distance_threshold) {
    
    MatF3 diff = nearest_points - coords;
    Eigen::VectorXf norms = diff.rowwise().norm();
    
    MatF3 normals=diff.normalized();
    Eigen::ArrayXf sign = (sdfs.array() > 0).cast<float>() * 2.0 - 1.0;                        
    normals.array().colwise() *=-sign.array();
    
    Eigen::Matrix<bool, Eigen::Dynamic, 3> mask=feature_flags.replicate(1, 3);
    normals = mask.select(sns, normals);
    
    Eigen::Matrix<bool, Eigen::Dynamic, 3> mask1=(norms.array()<norm_eps).replicate(1,3);

    normals=mask1.select(sns,normals);
    
    bool torch_feature = feature_flags.count()>0;
    double angle_threshold=20.0;
    if (torch_feature)
        angle_threshold=30.0;
    PlaneGrouping plane_group(angle_threshold, position_eps, distance_threshold, 1, false);
    auto [initial_groups, final_groups, _] = plane_group.fit(nearest_points, normals, torch_feature);

    bool enough = false;
    std::vector<std::vector<int>> groups;
    
    if (torch_feature) {
        if (initial_groups.size() <= 2 && final_groups.size() == 1) {
            enough = false;
            groups = final_groups;
        } 
        else if (initial_groups.size() > 2 && final_groups.size() == 1) {
            enough = true;
            groups = initial_groups;
        } 
        else {
            enough = final_groups.size() > 1;
            groups = final_groups;
        }
    } 
    else {
        enough = final_groups.size() > 1;
        groups = final_groups;
    }
    
    return {enough, groups};
}

Eigen::VectorXi find_indices(const Eigen::Array<bool, Eigen::Dynamic, 1>& mask) {
    Eigen::VectorXi indices(mask.count());
    int index=0;
    for (int i = 0; i < mask.size(); ++i) {
        if (mask(i)) {
            indices(index)=i;
            index++;
        }
    }
    return indices;
}
double radians(double degrees) {
    return degrees * (M_PI / 180.0);
}
MatF computeDistanceMatrix(const MatF3& vertices) {
    int n = vertices.rows();
    MatF distances(n, n);

    for (int i = 0; i < n; ++i) {
        distances(i, i) = 0.0f;
        
        for (int j = i + 1; j < n; ++j) {
            float dist = (vertices.row(i) - vertices.row(j)).norm();
            distances(i, j) = dist;
            distances(j, i) = dist;
        }
    }
    
    return distances;
}

std::pair<std::vector<int>, std::vector<std::vector<int>>> 
find_central_indices(const MatF3& vertices, const std::vector<std::vector<int>>& groups,int k) {
    
    std::vector<int> central_indices;
    std::vector<std::vector<int>> new_groups;

    central_indices.reserve(groups.size() * (k > 1 ? k : 1));
    new_groups.reserve(groups.size());

    std::vector<double> distance_buffer;
    
    for (const auto& group : groups) {
        if (group.empty()) continue;
        
        const size_t group_size = group.size();
        
        if (group_size == 1) {
            new_groups.emplace_back(std::initializer_list<int>{static_cast<int>(central_indices.size())});
            central_indices.push_back(group[0]);
            continue;
        }

        MatF3 group_points(group_size, vertices.cols());
        for (size_t i = 0; i < group_size; ++i) {
            group_points.row(i) = vertices.row(group[i]);
        }
        // MatF distance_matrix=computeSquaredDistanceMatrix(group_points);
        // Eigen::VectorXf distance_buffer = distance_matrix.rowwise().sum();
        Eigen::MatrixXf distances(group.size(), group.size());
        for (size_t i = 0; i < group.size(); ++i) {
            for (size_t j = 0; j < group.size(); ++j) {
                distances(i, j) = (group_points.row(i) - group_points.row(j)).norm();
            }
        }
        Eigen::VectorXf total_distances = distances.rowwise().sum();
        if (k == 1) {

            Eigen::Index min_index;
            total_distances.minCoeff(&min_index);
            int central_index = group[min_index];
            new_groups.push_back({static_cast<int>(central_indices.size())});
            central_indices.push_back(central_index);
        } 
        else {
            int actual_k = std::min(k, static_cast<int>(group_size));
            
            //std::vector<int> k_min_index_in_group=get_k_smallest_indices_nth(distance_buffer,actual_k);
            std::vector<int> k_min_indices=get_k_smallest_indices_nth(total_distances,actual_k);
            
            std::vector<int> k_central_indices;
            k_central_indices.reserve(k_min_indices.size());
            for (int idx : k_min_indices) {
                k_central_indices.push_back(group[idx]);
            }
            std::vector<int> new_group;
            int start_idx = static_cast<int>(central_indices.size());
            int end_idx = start_idx + static_cast<int>(k_central_indices.size());
            for (int i = start_idx; i < end_idx; ++i) {
                new_group.push_back(i);
            }
            new_groups.push_back(new_group);
            //central_indices.extend(k_central_index)
            central_indices.insert(central_indices.end(), k_central_indices.begin(), k_central_indices.end());
        }
    }
    
    return {central_indices, new_groups};
}


double compute_tangent_plane(MatF3& spoints,
                    MatF3& sp_normals,
                    const ArrayB1& sp_feature_mask,
                    Eigen::Vector3f& center,
                    double box_edge_length,double octree_norm_eps,double octree_position_eps)
{
    bool has_feature=sp_feature_mask.count()>0;
    SDFProjectorFunc projfunc(spoints,sp_normals,sp_feature_mask);
    Eigen::Vector3f min_corner=center-Eigen::Vector3f::Constant(box_edge_length*0.125);
    Eigen::Vector3f max_corner=center+Eigen::Vector3f::Constant(box_edge_length*0.125);
    MatF3 origin_coords=generate_linespace_points(min_corner,max_corner,3);
    Eigen::VectorXf origin_sdfs;
    MatF3 origin_sns;
    MatF3 origin_nearest_points;
    Eigen::VectorXi origin_spoint_ids;
    Eigen::Array<bool,Eigen::Dynamic,1> origin_feature_flags;
    std::tie(origin_sdfs, origin_sns,origin_nearest_points,origin_spoint_ids,origin_feature_flags)=projfunc(origin_coords);

    //filter
    Eigen::Array<bool,Eigen::Dynamic,1> maskV=(origin_sdfs.array()<0.0f);
    Eigen::VectorXi selectIndices=find_indices(maskV);
    if(selectIndices.rows()==0){
        return std::numeric_limits<double>::infinity();
    }

    Eigen::VectorXf sdfs = origin_sdfs(selectIndices);
   
    MatF3 sns=origin_sns(selectIndices,Eigen::all);
    MatF3 coords = origin_coords(selectIndices,Eigen::all);
    MatF3 nearest_points = origin_nearest_points(selectIndices,Eigen::all);
    Eigen::Array<bool,Eigen::Dynamic,1> feature_flags = origin_feature_flags(selectIndices);

    Eigen::ArrayXf sdfs_sign = (sdfs.array() > 0).cast<float>() * 2.0 - 1.0;   
    MatF3 diff = nearest_points - coords;
    Eigen::VectorXf norms = diff.rowwise().norm();
    norms = norms.array().max(1e-12);
    MatF3 normals=diff.rowwise().normalized();
    
    normals.array().colwise() *= -sdfs_sign.array();

    Eigen::Matrix<bool, Eigen::Dynamic, 3> mask=feature_flags.replicate(1, 3);
    normals = mask.select(sns, normals);
    
    Eigen::Matrix<bool, Eigen::Dynamic, 3> mask1=(norms.array()<octree_norm_eps).replicate(1,3);
    normals=mask1.select(sns,normals);

    double position_eps=radians(40.0)*norms.maxCoeff();
    double distance_threshold=box_edge_length*0.1;
    auto [enough,groups]=compute_groups(sdfs,sns,coords,nearest_points,feature_flags,octree_norm_eps,std::max(octree_position_eps,position_eps),distance_threshold);

    if(enough){
        std::sort(groups.begin(), groups.end(), 
        [](const std::vector<int>& a, const std::vector<int>& b) {
            return a.size() > b.size(); 
        });

        std::vector<std::vector<int>> plane_groups(
            groups.begin(), 
            groups.begin() + std::min(2, (int)groups.size())
        );
        if (coords.rows()>4 && has_feature){
            std::vector<std::vector<int>>new_groups;
            std::vector<int> central_indices;
            std::tie(central_indices,new_groups)=find_central_indices(nearest_points,plane_groups);
            //Eigen::Map<Eigen::VectorXi> eigen_central_indices(central_indices.data(), central_indices.size());
            Eigen::VectorXi eigen_central_indices(central_indices.size());
            for (size_t i = 0; i < central_indices.size(); ++i) {
                eigen_central_indices(i) = central_indices[i];
            }
            sdfs=sdfs(eigen_central_indices).eval();
            sns=sns(eigen_central_indices,Eigen::all).eval();
            coords=coords(eigen_central_indices,Eigen::all).eval();
            nearest_points=nearest_points(eigen_central_indices,Eigen::all).eval();
            feature_flags=feature_flags(eigen_central_indices).eval();
        }else{
            std::vector<int> central_indices;

            central_indices.insert(central_indices.end(), 
                                plane_groups[0].begin(), 
                                plane_groups[0].end());
            
            central_indices.insert(central_indices.end(), 
                                plane_groups[1].begin(), 
                                plane_groups[1].end());
            Eigen::Map<Eigen::VectorXi> eigen_central_indices(central_indices.data(), central_indices.size()); 
            sdfs=sdfs(eigen_central_indices);
            sns=sns(eigen_central_indices,Eigen::all);
            coords=coords(eigen_central_indices,Eigen::all);
            nearest_points=nearest_points(eigen_central_indices,Eigen::all);
            feature_flags=feature_flags(eigen_central_indices);      
        }
        diff = nearest_points - coords;
        norms = (nearest_points - coords).rowwise().norm();
        normals=diff.rowwise().normalized();
        sdfs_sign = (sdfs.array() > 0).cast<float>() * 2.0 - 1.0; 
        normals.array().colwise() *= -sdfs_sign.array();

        
        mask1=(norms.array()<octree_norm_eps).replicate(1,3);
        normals=mask1.select(sns,normals);
        SQEF sqef(normals,nearest_points,coords);
        sqef.solve_in_plane();
        Plane plane(sqef.dir,sqef.point.topRows(3));
        return plane.distanceToPoint(center);
    }
    return std::numeric_limits<double>::infinity();
}
        
        

double compute_tangent_plane_dis(SDFProjectorFunc& projfunc,Eigen::Vector3f& center,double box_edge_length,
    double octree_norm_eps,double octree_position_eps,double ratio)
{
    Eigen::Vector3f min_corner=center-Eigen::Vector3f::Constant(box_edge_length*ratio);
    Eigen::Vector3f max_corner=center+Eigen::Vector3f::Constant(box_edge_length*ratio);
    MatF3 origin_coords=generate_linespace_points(min_corner,max_corner,3);
    Eigen::VectorXf origin_sdfs;
    MatF3 origin_sns;
    MatF3 origin_proj_points;
    Eigen::VectorXi origin_spoint_ids;
    Eigen::Array<bool,Eigen::Dynamic,1> origin_feature_flags;
    std::tie(origin_sdfs, origin_sns,origin_proj_points,origin_spoint_ids,origin_feature_flags)=projfunc(origin_coords);

    Eigen::Array<bool,Eigen::Dynamic,1> maskV=(origin_sdfs.array()<0.0f);
    Eigen::VectorXi selectIndices=find_indices(maskV);
    if(selectIndices.rows()==0){
        return std::numeric_limits<double>::infinity();
    }

    Eigen::VectorXf sdfs = origin_sdfs(selectIndices);
   
    MatF3 sns=origin_sns(selectIndices,Eigen::all);
    MatF3 coords = origin_coords(selectIndices,Eigen::all);
    MatF3 proj_points = origin_proj_points(selectIndices,Eigen::all);
    Eigen::VectorXi spoint_ids=origin_spoint_ids(selectIndices);
    Eigen::Array<bool,Eigen::Dynamic,1> feature_flags = origin_feature_flags(selectIndices);

    Eigen::ArrayXf sdfs_sign = (sdfs.array() > 0).cast<float>() * 2.0 - 1.0;   
    MatF3 diff = proj_points - coords;
    Eigen::VectorXf norms = diff.rowwise().norm();
    norms = norms.array().max(1e-12);
    MatF3 normals=diff.rowwise().normalized();
    
    normals.array().colwise() *= -sdfs_sign.array();

    Eigen::Matrix<bool, Eigen::Dynamic, 3> mask=feature_flags.replicate(1, 3);
    normals = mask.select(normals,sns);
    
    Eigen::Matrix<bool, Eigen::Dynamic, 3> mask1=(norms.array()<octree_norm_eps).replicate(1,3);
    normals=mask1.select(sns,normals);

    double position_eps=radians(40.0)*norms.maxCoeff();
    double distance_threshold=box_edge_length*0.1;
    auto [enough,groups]=compute_groups(sdfs,sns,coords,proj_points,feature_flags,octree_norm_eps,std::max(octree_position_eps,position_eps),distance_threshold);

    if(enough){
        std::sort(groups.begin(), groups.end(), 
        [](const std::vector<int>& a, const std::vector<int>& b) {
            return a.size() > b.size();
        });

        std::vector<std::vector<int>> plane_groups(
            groups.begin(), 
            groups.begin() + std::min(2, (int)groups.size())
        );
        if (coords.rows()>4){
            std::vector<std::vector<int>>new_groups;
            std::vector<int> central_indices;
            std::tie(central_indices,new_groups)=find_central_indices(proj_points,plane_groups);
            //Eigen::Map<Eigen::VectorXi> eigen_central_indices(central_indices.data(), central_indices.size());
            Eigen::VectorXi eigen_central_indices(central_indices.size());
            for (size_t i = 0; i < central_indices.size(); ++i) {
                eigen_central_indices(i) = central_indices[i];
            }
            sdfs=sdfs(eigen_central_indices).eval();
            sns=sns(eigen_central_indices,Eigen::all).eval();
            coords=coords(eigen_central_indices,Eigen::all).eval();
            proj_points=proj_points(eigen_central_indices,Eigen::all).eval();
            feature_flags=feature_flags(eigen_central_indices).eval();
        }else{
            std::vector<int> central_indices;

            central_indices.insert(central_indices.end(), 
                                plane_groups[0].begin(), 
                                plane_groups[0].end());

            central_indices.insert(central_indices.end(), 
                                plane_groups[1].begin(), 
                                plane_groups[1].end());
            Eigen::Map<Eigen::VectorXi> eigen_central_indices(central_indices.data(), central_indices.size()); 
            sdfs=sdfs(eigen_central_indices);
            sns=sns(eigen_central_indices,Eigen::all);
            coords=coords(eigen_central_indices,Eigen::all);
            proj_points=proj_points(eigen_central_indices,Eigen::all);
            feature_flags=feature_flags(eigen_central_indices); 
            //groups.swap(plane_groups);         
        }
        diff = proj_points - coords;
        norms = (proj_points - coords).rowwise().norm();
        normals=diff.rowwise().normalized();
        sdfs_sign = (sdfs.array() > 0).cast<float>() * 2.0 - 1.0; 
        normals.array().colwise() *= -sdfs_sign.array();

        mask=feature_flags.replicate(1, 3);
        normals = mask.select(normals,sns);
        
        mask1=(norms.array()<octree_norm_eps).replicate(1,3);
        normals=mask1.select(sns,normals);
        SQEF sqef(normals,proj_points,coords);
        sqef.solve_in_plane();
        Plane plane(sqef.dir,sqef.point.topRows(3));
        return plane.distanceToPoint(center);
    }
    return std::numeric_limits<double>::infinity();
}
              
int countUniqueElements(const Eigen::RowVectorXi& vec) {
    std::set<int> unique_set;
    
    for (int i = 0; i < vec.size(); ++i) {
        unique_set.insert(vec(i));
    }
    
    return unique_set.size();
}        
MatF batch_compute_face_udf_parallel(SDFProjectorFunc& projfunc,MatF3&m_mesh_v,MatI& m_mesh_f,MatI& m_mesh_f_unique,double box_edge_length,
    double octree_norm_eps,double octree_position_eps,double ratio)
{
    int64_t num_face=m_mesh_f.rows();
    MatF m_mesh_f_udfs=MatF::Zero(num_face,2);
    Eigen::Vector3i idx00={0,1,2};
    Eigen::Vector3i idx01={0,2,3};

    Eigen::Vector3i idx10={0,1,3};
    Eigen::Vector3i idx11={1,2,3};
    MatI tri00_mat=m_mesh_f_unique(Eigen::all, idx00);
    MatI tri01_mat=m_mesh_f_unique(Eigen::all, idx01);

    MatI tri10_mat=m_mesh_f_unique(Eigen::all, idx10);
    MatI tri11_mat=m_mesh_f_unique(Eigen::all, idx11);

    #pragma omp parallel for schedule(static)
    for(int64_t fid=0;fid<num_face;fid++)
    {
        const auto& face_unique=m_mesh_f_unique.row(fid);
        if(countUniqueElements(face_unique)<=2){
            continue;
        }
        //Eigen::RowVectorXi face=m_mesh_f_unique.row(fid);
        //MatF3 face_vertices=m_mesh_v(face,Eigen::all).eval();
        Eigen::RowVectorXi tri00=tri00_mat.row(fid);
        Eigen::RowVectorXi tri01=tri01_mat.row(fid);
        double dis00=0;
        double dis01=0;


        if(countUniqueElements(tri00)==3){
            MatF3 tri_vertices=m_mesh_v(tri00,Eigen::all);
            Eigen::Vector3f center00 =tri_vertices.colwise().mean().transpose();
            dis00=compute_tangent_plane_dis(projfunc,center00,box_edge_length,octree_norm_eps,octree_position_eps,ratio);
        }
        if(countUniqueElements(tri01)==3){
            MatF3 tri_vertices=m_mesh_v(tri01,Eigen::all);
            Eigen::Vector3f center01 = tri_vertices.colwise().mean().transpose();
            dis01=compute_tangent_plane_dis(projfunc,center01,box_edge_length,octree_norm_eps,octree_position_eps,ratio);
        }
        m_mesh_f_udfs(fid,0)=std::max(dis00,dis01);
        Eigen::RowVectorXi tri10=tri10_mat.row(fid);
        Eigen::RowVectorXi tri11=tri11_mat.row(fid);
        double dis10=0;
        double dis11=0;
        if(countUniqueElements(tri10)==3){
            MatF3 tri_vertices= m_mesh_v(tri10,Eigen::all);
            Eigen::Vector3f center10 =tri_vertices.colwise().mean().transpose();
            dis10=compute_tangent_plane_dis(projfunc,center10,box_edge_length,octree_norm_eps,octree_position_eps,ratio);
        }
        if(countUniqueElements(tri11)==3){
            MatF3 tri_vertices=m_mesh_v(tri11,Eigen::all);
            Eigen::Vector3f center11 = tri_vertices.colwise().mean().transpose();
            dis11=compute_tangent_plane_dis(projfunc,center11,box_edge_length,octree_norm_eps,octree_position_eps,ratio);
        }
        m_mesh_f_udfs(fid,1)=std::max(dis10,dis11);
    }
    return m_mesh_f_udfs;
}        
        
        
        

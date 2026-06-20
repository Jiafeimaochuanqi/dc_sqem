import numpy as np
from .mesh_utils import stableNormalized
# from sklearn.cluster import DBSCAN
from cpp_module.GRAPH_GROUPING import DBSCAN
import sys
IS_LINUX = sys.platform.startswith("linux")
IS_WINDOWS = sys.platform.startswith("win")
try:
    if IS_LINUX or IS_WINDOWS:
        from cpp_module.GRAPH_GROUPING import PlaneGrouping as CPlaneGrouping
    else:
        raise NotImplementedError("not supported system")
except ImportError as e:
    print(f"import library fail:{e}")
    sys.exit(1)

def compute_groups(sdfs,sns,coords,nearest_points,feature_flags,norm_eps,position_eps,distance_threshold):
    #check group
    normals = stableNormalized(nearest_points - coords) * (-np.sign(sdfs))
    #normals[feature_flags]=sns[feature_flags]
    norms = np.linalg.norm(nearest_points - coords, axis=1)
    mask1=norms<norm_eps
    normals[mask1]=sns[mask1]
    torch_feature = np.count_nonzero(feature_flags) > 0

    if IS_LINUX:
        plane_group=CPlaneGrouping(40,position_eps,distance_threshold,1,False)
    else:
        plane_group=PlaneGrouping(40,position_eps,distance_threshold,1,False)
    initial_groups,final_groups, _=plane_group.fit(nearest_points.astype(np.float64),normals.astype(np.float64),torch_feature)
    enough=True
    groups=[]
    if torch_feature and len(initial_groups)<=2 and len(final_groups)==1:
        enough=False
        groups=final_groups
    elif torch_feature and len(initial_groups)>2 and len(final_groups)==1:
        enough=True
        groups=initial_groups
    else:
        enough=len(final_groups)>1
        groups=final_groups
    return enough,groups
def are_points_collinear(points, tolerance=1e-8):
    if len(points) < 3:
        return True
    
    p0 = points[0]
    direction = None
    
    for i in range(1, len(points)):
        v = points[i] - p0
        if not np.allclose(v, [0, 0, 0], atol=tolerance*10):
            direction = v
            break

    if direction is None:
        return True

    for i in range(1, len(points)):
        v = points[i] - p0
        if not np.allclose(v, [0, 0, 0], atol=tolerance*10):
            cross = np.cross(direction, v)
            if not np.allclose(cross, [0, 0, 0], atol=tolerance):
                return False
    
    return True

    
    
class Plane:
    def __init__(self,normal,point):
        self.normal=normal
        self.point=point
        self.distance = -np.dot(normal, point)
    def __str__(self):
        str=""
        if self.normal[0]!=0:
            str+=f"{self.normal[0]:.2f}x"
        if self.normal[1]!=0:
            if len(str)>0 and self.normal[1]>0:
                str+="+"
            str+=f"{self.normal[1]:.2f}y"
        if self.normal[2]!=0:
            if len(str)>0 and self.normal[2]>0:
                str+="+"
            str+=f"{self.normal[2]:.2f}z"
        if self.distance!=0:
            if self.distance>0:
                str+="+"
            str+=f"{self.distance:.2f}=0"
        return str
    def __repr__(self):
        str=""
        if self.normal[0]!=0:
            str+=f"{self.normal[0]:.2f}x"
        if self.normal[1]!=0:
            if len(str)>0 and self.normal[1]>0:
                str+="+"
            str+=f"{self.normal[1]:.2f}y"
        if self.normal[2]!=0:
            if len(str)>0 and self.normal[2]>0:
                str+="+"
            str+=f"{self.normal[2]:.2f}z"
        if self.distance!=0:
            if self.distance>0:
                str+="+"
            str+=f"{self.distance:.2f}=0"
        return str
class PlaneGrouping:
    def __init__(self, angle_threshold=5.0, position_eps=0.1, distance_threshold=0.01, min_samples=3,refine=False,group_normal_first=True, max_assignment_distance=None):

        self.radian_threshold = np.radians(angle_threshold)
        self.distance_threshold = distance_threshold
        self.min_samples = min_samples
        self.max_assignment_distance = max_assignment_distance if max_assignment_distance is not None else distance_threshold * 2
        self.position_eps = position_eps
        self.refine=refine
        self.group_normal_first=group_normal_first
    def fit(self, points, normals,need_merge=True,debug=False):

        self.points = np.array(points)
        self.normals = np.array(normals)
        
        if self.group_normal_first:
            initial_groups = self._group_by_normals_and_positions()
        else:
            initial_groups = self._group_by_positions_and_normals()
        if debug:
            with open('_group_by_normals.txt', 'w') as f:
                for i,group_indices in enumerate(initial_groups):
                    group_points = self.points[group_indices]
                    group_normals = self.normals[group_indices]
                    min_corner = np.min(group_points, axis=0)
                    max_corner = np.max(group_points, axis=0)
                    print(f"group {i}:{min_corner}-->{max_corner}",file=f)
                    print(f"{group_points}",file=f)
                    print(f"{group_normals}",file=f)
        if self.refine:
            initial_groups=self._refine_groups_by_position(initial_groups)


        final_groups, planes = self._merge_planes(initial_groups,need_merge)

        if debug:
            for plane in planes:
                print(plane)

        final_groups, planes = self._assign_ungrouped_points(final_groups, planes,debug)
        
        return initial_groups,final_groups, planes
    
    def _group_by_normals_and_positions(self):

        cosine_eps = 1 - np.cos(self.radian_threshold)
        normal_clustering = DBSCAN(eps=cosine_eps,
                                    min_samples=self.min_samples,
                                    metric='cosine')

        normals_norm = self.normals / np.linalg.norm(self.normals, axis=1, keepdims=True)
            
            
        normal_labels = normal_clustering.fit_predict(normals_norm)
        
        
        groups = []

        for normal_label in np.unique(normal_labels):
            if normal_label == -1:
                noise_mask = (normal_labels == normal_label)
                noise_indices = np.where(noise_mask)[0]
                for idx in noise_indices:
                    groups.append([idx])
                continue
              
            normal_mask = (normal_labels == normal_label)
            cluster_points = self.points[normal_mask]
            cluster_normals = self.normals[normal_mask]
            cluster_indices = np.where(normal_mask)[0]
            
            if len(cluster_points) < self.min_samples:
                groups.append(cluster_indices.tolist())
                continue

            position_clustering = DBSCAN(eps=self.position_eps,
                                min_samples=self.min_samples)
            position_labels = position_clustering.fit_predict(cluster_points)  

            for position_label in np.unique(position_labels):
                position_mask = (position_labels == position_label)
                subgroup_indices = cluster_indices[position_mask].tolist()
                
                if position_label == -1:
                    for idx in subgroup_indices:
                        groups.append([idx])
                else:
                    if len(subgroup_indices) >= self.min_samples:
                        groups.append(subgroup_indices)
        
        return groups
    
    def _group_by_positions_and_normals(self):
        position_clustering = DBSCAN(eps=self.position_eps,
                                min_samples=self.min_samples)
        position_labels = position_clustering.fit_predict(self.points)
        
        groups = []
        for position_label in np.unique(position_labels):
            if position_label == -1:
                noise_mask = (position_labels == position_label)
                noise_indices = np.where(noise_mask)[0]
                for idx in noise_indices:
                    groups.append([idx])
                continue
            position_mask = (position_labels == position_label)
            cluster_points = self.points[position_mask]
            cluster_normals = self.normals[position_mask]
            cluster_indices = np.where(position_mask)[0]
            
            if len(cluster_points) < self.min_samples:
                groups.append(cluster_indices.tolist())
                continue

            normals_norm = cluster_normals / np.linalg.norm(cluster_normals, axis=1, keepdims=True)
            cosine_eps = 1 - np.cos(self.radian_threshold)
            normal_clustering = DBSCAN(eps=cosine_eps,
                                    min_samples=self.min_samples,
                                    metric='cosine')
            normal_labels = normal_clustering.fit_predict(normals_norm)

            for normal_label in np.unique(normal_labels):
                normal_mask = (normal_labels == normal_label)
                subgroup_indices = cluster_indices[normal_mask].tolist()
                
                if normal_label == -1:
                    for idx in subgroup_indices:
                        groups.append([idx])
                else:
                    if len(subgroup_indices) >= self.min_samples:
                        groups.append(subgroup_indices)
        
        return groups
    
    def _refine_groups_by_position(self, initial_groups):
        refined_groups = []
        
        for group_indices in initial_groups:
            if len(group_indices) < self.min_samples:
                continue
                
            group_points = self.points[group_indices]
            group_normals = self.normals[group_indices]

            try:
                ransac_groups = self._ransac_plane_fitting(group_points, group_normals, group_indices)
                refined_groups.extend(ransac_groups)
            except Exception as e:
                print(e)
                self._fallback_refinement(group_points, group_normals, group_indices, refined_groups)

        return refined_groups
    
    def _ransac_plane_fitting(self, points, normals, indices, max_iterations=100, inlier_threshold=None):
        if inlier_threshold is None:
            inlier_threshold = self.distance_threshold
        n_points=len(points)
        if n_points == 1:
            return [indices]

        if n_points == 2:
            return [indices]
        
        remaining_points = points.copy()
        remaining_normals = normals.copy()
        remaining_indices = indices.copy()
        
        groups = []
        
        while len(remaining_points) >= self.min_samples and len(groups) <= 2:
            best_plane = None
            best_inliers = []
            best_inlier_indices = []
            
            for _ in range(max_iterations):
                if len(remaining_points) < 3:
                    break
                    
                sample_indices = np.random.choice(len(remaining_points), 3, replace=False)
                sample_points = remaining_points[sample_indices]
                sample_normals = remaining_normals[sample_indices]

                plane = self._fit_plane_to_points(sample_points, sample_normals)
                if plane is None:
                    continue

                inliers = []
                inlier_indices = []
                for i, point in enumerate(remaining_points):
                    distance = self._point_to_plane_distance(point, plane.normal, plane.distance)
                    if distance < inlier_threshold:
                        inliers.append(point)
                        inlier_indices.append(remaining_indices[i])

                if len(inliers) > len(best_inliers):
                    best_plane = plane
                    best_inliers = inliers
                    best_inlier_indices = inlier_indices

            if len(best_inliers) >= self.min_samples:
                groups.append(best_inlier_indices)

                best_inlier_set = set(best_inlier_indices)
                remaining_indices = [idx for idx in remaining_indices if idx not in best_inlier_set]
                remaining_points = self.points[remaining_indices]
                remaining_normals = self.normals[remaining_indices]
            else:
                break
        return groups
    
    def _fallback_refinement(self, points, normals, indices, refined_groups):
        avg_normal = np.median(normals, axis=0)
        avg_normal = avg_normal / np.linalg.norm(avg_normal)
        distances = np.dot(points, avg_normal)
        if len(distances) > 0:
            distance_clustering = DBSCAN(
                eps=self.distance_threshold, 
                min_samples=self.min_samples
            )
            distance_labels = distance_clustering.fit_predict(distances.reshape(-1, 1))
            for label in set(distance_labels):
                if label != -1:
                    subgroup_mask = (distance_labels == label)
                    subgroup_indices = [indices[i] for i in np.where(subgroup_mask)[0]]
                    refined_groups.append(subgroup_indices)
    def _fit_plane_to_points_final(self, points, normals):
        n_points = len(points)
        if n_points==0:
            return None
        elif n_points==1:
            point=points[0]
            normal=normals[0]
            return Plane(normal, point)
        elif n_points==2:
            point=points[0]
            normal=normals[0]
            plane= Plane(normal, point)
            distance = self._point_to_plane_distance(points[1], normal, plane.distance)
            if distance > self.distance_threshold:
                return None
            else:
                return plane
        centroid = np.mean(points, axis=0)
        centered = points - centroid
        covariance = np.cov(centered.T)
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        normal = eigenvectors[:, np.argmin(eigenvalues)]
        normal = normal / np.linalg.norm(normal)
        if normals is not None and len(normals) >= 3:
            avg_point_normal = np.mean(normals, axis=0)
            avg_point_normal = avg_point_normal / np.linalg.norm(avg_point_normal)
            
            if np.dot(normal, avg_point_normal) < 0:
                normal = -normal
        return Plane(normal, centroid)
    
    def _fit_plane_to_points(self, points, normals):
        unique_points,indexes=np.unique(points,return_index=True,axis=0)
        unique_normals=normals[indexes]
        n_points = len(unique_points)
        if n_points==0:
            return None
        elif n_points==1:
            point=points[0]
            normal=normals[0]
            return Plane(normal, point)
        elif n_points==2:
            point=unique_points[0]
            normal=unique_normals[0]
            plane= Plane(normal, point)
            distance = self._point_to_plane_distance(unique_points[1], normal, plane.distance)
            if distance > self.distance_threshold:
                return None
            else:
                return plane

        if are_points_collinear(points):
            return None

        centroid = np.mean(points, axis=0)
        centered = points - centroid

        covariance = np.cov(centered.T)
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        normal = eigenvectors[:, np.argmin(eigenvalues)]
        normal = normal / np.linalg.norm(normal)

        if normals is not None and len(normals) >= 3:
            avg_point_normal = np.mean(normals, axis=0)
            avg_point_normal = avg_point_normal / np.linalg.norm(avg_point_normal)
            
            if np.dot(normal, avg_point_normal) < 0:
                normal = -normal

        return Plane(normal, centroid)
    
    def _point_to_plane_distance(self, point, plane_normal, plane_distance):
        return abs(np.dot(plane_normal, point) + plane_distance) / np.linalg.norm(plane_normal)
    
    def _merge_planes(self, groups,need_merge):
        group_planes = []
        
        for group in groups:
            if len(group) >= self.min_samples:
                points = self.points[group]
                normals = self.normals[group]
                plane = self._fit_plane_to_points(points, normals)
                group_planes.append(plane)
            else:
                group_planes.append(None)
        
        n_groups = len(groups)
        
        if n_groups == 0:
            return [], []
        if not need_merge:
            return groups,group_planes
        merge_target = [None] * n_groups

        for i in range(n_groups):
            if merge_target[i] is not None:
                continue

            for j in range(n_groups):
                if i == j:
                    continue

                if group_planes[j] is None:
                    continue

                if merge_target[j] is not None:
                    continue
                if self._is_group_on_plane(groups[i], group_planes[j]):
                    merge_target[i] = j
                    break
        root_groups = []
        for i in range(n_groups):
            if merge_target[i] is None:
                root_groups.append(i)

        final_groups = []
        final_planes = []

        for root_idx in root_groups:
            merged_indices = [root_idx]
            for i in range(n_groups):
                if merge_target[i] == root_idx:
                    merged_indices.append(i)

            merged_group = []
            for idx in merged_indices:
                merged_group.extend(groups[idx])
            if len(merged_group) >= self.min_samples:
                points = self.points[merged_group]
                normals = self.normals[merged_group]
                plane = self._fit_plane_to_points_final(points, normals)
                if plane is not None:
                    final_groups.append(merged_group)
                    final_planes.append(plane)
        
        return final_groups, final_planes
    
    def _is_group_on_plane(self, point_indices, plane:Plane):
        plane_normal, plane_distance = plane.normal,plane.distance
        
        for idx in point_indices:
            point = self.points[idx]
            distance = self._point_to_plane_distance(point, plane_normal, plane_distance)
            if distance > self.distance_threshold:
                return False
        return True
    
    def _assign_ungrouped_points(self, groups, planes,debug=False):
        if not groups:
            return groups, planes
        grouped_indices = set()
        for group in groups:
            grouped_indices.update(group)
        all_indices = set(range(len(self.points)))
        ungrouped_indices = list(all_indices - grouped_indices)
        
        if not ungrouped_indices:
            return groups, planes

        for idx in ungrouped_indices:
            point = self.points[idx]
            min_distance = float('inf')
            best_group_idx = -1
            for i, (group, plane) in enumerate(zip(groups, planes)):
                plane_normal, plane_distance = plane.normal,plane.distance
                distance = self._point_to_plane_distance(point, plane_normal, plane_distance)
                if distance < min_distance:
                    min_distance = distance
                    best_group_idx = i

            if best_group_idx >= 0 and min_distance <= self.max_assignment_distance:
                groups[best_group_idx].append(idx)
            else:
                groups.append([idx])
                point = self.points[idx]
                normal = self.normals[idx]      
                plane= Plane(normal, point)
                
                planes.append(plane)
        if debug:
            for i, group in enumerate(groups):
                if len(group) >= self.min_samples:
                    points = self.points[group]
                    normals = self.normals[group]
                    planes[i] = self._fit_plane_to_points_final(points, normals)
        
        return groups, planes
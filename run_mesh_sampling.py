#%%
import os
from dual_contouring.mesh_utils import load_mesh,compute_octree_node_edge_length,save_obj_with_normals
import numpy as np
from config import get_sampling_args
from sampling import sampling_from_mesh
import open3d as o3d
import matplotlib
from cpp_module.GRAPH_GROUPING import PointCloudRVD,KNNSearch
name_color_strings=['Yellow','cyan','Magenta','pink','orange','brown','blue','green','Teal','Silver','Black','Gray','Olive','Coral','Lavender','Lime']
hex_color_strings=["#FF00D480"]
def main(args):
    color_rgb_array=[]
    for color_string in name_color_strings:
        color_rgb=matplotlib.colors.to_rgb(color_string)
        color_rgb_array.append(np.array(color_rgb))
    # for hex_color_string in hex_color_strings:
    #     color_rgb=matplotlib.colors.to_rgb(hex_color_string)
    #     color_rgb_array.append(np.array(color_rgb))

    mesh= load_mesh(args.dataset, args.datasets_dir)

    os.makedirs(args.output_dir, exist_ok=True)
    m_unit_edge_length=compute_octree_node_edge_length(args.max_depth)
    args.m_unit_edge_length=m_unit_edge_length
    args.debug=True
    data=sampling_from_mesh(args,mesh)

    savepath=os.path.join(args.output_dir, args.dataset+".sampling.npz")
    sp_feature_mask=data.get('sp_feature_mask',None)
    if sp_feature_mask is None:
        sp_feature_mask=np.zeros(len(data['spoints']),dtype=bool)
    rvd = PointCloudRVD(data['spoints'], data['ns'], sp_feature_mask)
    knn = KNNSearch(data['spoints'], 16)
    if not args.skip_feature:
        min_dihedral_angle_degree=data['min_dihedral_angle_degree']
        rvd.buildKnnGraph(knn, 26, True, 180.0-args.angle_threshold_degrees-1.0,min((180.0-min_dihedral_angle_degree)/2.0+0.1,90.0), 1)
        patchIds =rvd.computeSmoothPatchIds()
        patchIds=np.array(patchIds)
        data['patchIds']=patchIds.astype(np.int32)
        data['angle_threshold_degrees']=np.float32(args.angle_threshold_degrees)
        print(np.max(patchIds))

    
    np.savez(savepath,**data)
    print(f"generate {savepath} finished!")
    if args.debug:
        #================================
        
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(data['spoints'])
        pcd.normals = o3d.utility.Vector3dVector(data['ns'])
        if not args.skip_feature:
            colors = np.zeros((len(data['spoints']), 3)).astype(np.float32)
            colors[data['sp_feature_mask']]=np.array([1.0,0,0])
            non_feature_ids=np.where(~data['sp_feature_mask'])[0]
            for id in non_feature_ids:
                color_rgb=color_rgb_array[patchIds[id]%len(color_rgb_array)]
                colors[id]= np.array(color_rgb)
            pcd.colors = o3d.utility.Vector3dVector(colors)
        savepath=os.path.join(args.output_dir, args.dataset+".sampling_surface_normals.ply")
        o3d.io.write_point_cloud(savepath, pcd, write_ascii=False)
        print(f"generate {savepath} finished!")
        
        surface_graph_edges = rvd.extractUniqueUndirectedEdges()
        rvd_graph_edges = surface_graph_edges
        savepath=os.path.join(args.output_dir, args.dataset+".sampling_surface_normals.obj")
        save_obj_with_normals(savepath,data['spoints'],rvd_graph_edges,[],data['ns'])
        print(f"generate {savepath} finished!")
        #================================


if __name__ == "__main__":
    main(get_sampling_args())

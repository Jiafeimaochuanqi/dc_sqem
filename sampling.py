import numpy as np
from pyigl.sharp_feature_detector import *
from dual_contouring.mesh_utils import make_dict_edge2face,sample_points_on_edge,stableNormalized
from pyigl.barycentric_lattice import sample_mesh_uniform_lattice,extract_unique_edges
def angle_between_normals(n1, n2):
    dot = n1[0]*n2[0] + n1[1]*n2[1] + n1[2]*n2[2]
    mod1 = math.sqrt(n1[0]**2 + n1[1]**2 + n1[2]**2)
    mod2 = math.sqrt(n2[0]**2 + n2[1]**2 + n2[2]**2)
    cos_theta = dot / (mod1 * mod2)
    cos_theta = max(min(cos_theta, 1.0), -1.0)
    rad = math.acos(cos_theta)
    deg = math.degrees(rad)
    
    return rad, deg

def sampling_from_mesh(args,mesh):
    n_samples=args.n_samples
    ratio=args.ratio
    mesh.fix_normals()
    data={}
    convex_sharp_edges=[]
    concave_sharp_edges=[]
    if not args.skip_feature:
        detector = SharpFeatureDetector(mesh.vertices, mesh.faces, args.angle_threshold_degrees)
        edge_types, vertex_to_edges,min_dihedral_angle_degree,max_shape_edge = detector.detect_sharp_edges()
        print(f"min_dihedral_angle_degree:{min_dihedral_angle_degree},max_shape_edge:{max_shape_edge}")
        data['min_dihedral_angle_degree']=np.float32(min_dihedral_angle_degree)
        corner_points = detector.detect_corner_points(edge_types, vertex_to_edges)
        feature_id_set=set()
        convex_corner_id_set=set()
        convex_feature_id_set=set()
        concave_feature_id_set=set()

        other_edges=[]
        for i, corner in enumerate(corner_points):
            if corner.type==ConerType.OUTER:
                convex_corner_id_set.add(corner.vertex_id)
                convex_feature_id_set.add(corner.vertex_id)
            else:
                concave_feature_id_set.add(corner.vertex_id)
            feature_id_set.add(corner.vertex_id)
        edge_vertices=mesh.vertices[detector.uE]#MX2X3
        edge_vector=edge_vertices[:,1,:]-edge_vertices[:,0,:]#MX3
        edge_lengths=np.linalg.norm(edge_vector,axis=1)
        for i, et in enumerate(edge_types):
            if et== EdgeType.CONVEX:
                convex_feature_id_set.add(detector.uE[i, 0])
                convex_feature_id_set.add(detector.uE[i, 1])
                convex_sharp_edges.append(detector.uE[i])

                feature_id_set.add(detector.uE[i, 0])
                feature_id_set.add(detector.uE[i, 1])
            elif et == EdgeType.CONCAVE:
                concave_feature_id_set.add(detector.uE[i, 0])
                concave_feature_id_set.add(detector.uE[i, 1])
                concave_sharp_edges.append(detector.uE[i])

                feature_id_set.add(detector.uE[i, 0])
                feature_id_set.add(detector.uE[i, 1])
            else:
                other_edges.append(detector.uE[i])
        convex_sharp_edges=np.array(convex_sharp_edges)
        concave_sharp_edges=np.array(concave_sharp_edges)

        feature_ids=np.array(list(feature_id_set)).astype(int)
        convex_corner_ids=np.array(list(convex_corner_id_set)).astype(int)
        concave_feature_id_set=concave_feature_id_set-convex_feature_id_set
        convex_feature_ids=np.array(list(convex_feature_id_set)).astype(int)
        concave_feature_ids=np.array(list(concave_feature_id_set)).astype(int)

        data['convex_corner_points']=mesh.vertices[convex_corner_ids]

        data['feature_ids']=feature_ids
        data['convex_feature_ids']=convex_feature_ids
        data['concave_feature_ids']=concave_feature_ids

        
        data['convex_sharp_edges']=convex_sharp_edges
        data['concave_sharp_edges']=concave_sharp_edges
        data['edge_lengths']=edge_lengths
    else:
        other_edges=extract_unique_edges(mesh.faces)
        
    

    edge_to_faces,vertex_to_edge=make_dict_edge2face(mesh.faces)
    
    total_feature_edge_interior_sample_num=0
    for edge in convex_sharp_edges:
        v1=mesh.vertices[edge[0]]
        v2=mesh.vertices[edge[1]]
        n_edge_samples=int(np.linalg.norm(v2-v1)//(args.m_unit_edge_length*ratio))+1
        total_feature_edge_interior_sample_num+=n_edge_samples
    for edge in concave_sharp_edges:
        v1=mesh.vertices[edge[0]]
        v2=mesh.vertices[edge[1]]
        n_edge_samples=int(np.linalg.norm(v2-v1)//(args.m_unit_edge_length*ratio))+1
        total_feature_edge_interior_sample_num+=n_edge_samples

    sp_feature_mask=np.zeros(len(mesh.vertices),dtype=bool)
    feature_ids=data.get('feature_ids',np.array([]))
    if len(feature_ids)>0:
        sp_feature_mask[feature_ids]=True
    non_feature_mask=~sp_feature_mask
    non_feature_edge_interior_sample_num=0
    concave_edge_interior_sample_num=0
    if n_samples>0:
        result=sample_mesh_uniform_lattice(mesh.vertices,mesh.faces,n_samples,len(convex_sharp_edges)+len(concave_sharp_edges),total_feature_edge_interior_sample_num)
        x_gt = result["all_samples"]      
        n_gt = result["all_normals"]
        tmp_points=np.vstack((x_gt,mesh.vertices[non_feature_mask]))
        tmp_normals=np.vstack((n_gt,mesh.vertex_normals[non_feature_mask]))

        
        #=============================
        other_edge_spoints=[]
        other_edge_ns=[]
        m=result['m']
        non_feature_edge_interior_sample_num = max(0, m - 1)
        #concave_edge_interior_sample_num = max(0, m - 1)
        if non_feature_edge_interior_sample_num>0:
            for edge in other_edges:
                v1=mesh.vertices[edge[0]]
                v2=mesh.vertices[edge[1]]
                edge_vertice=np.vstack((v1,v2))
                #n_edge_samples=edge_sample_count[edge]
                samples_on_edge=sample_points_on_edge(edge_vertice,non_feature_edge_interior_sample_num)
                if len(samples_on_edge)>0:
                    related_face_ids=edge_to_faces[tuple(sorted(edge))]
                    f0=related_face_ids[0]
                    f1=related_face_ids[1]
                    n0=mesh.face_normals[f0]
                    n1=mesh.face_normals[f1]
                    edge_normal=stableNormalized(n0+n1)[0]
                    if np.isnan(edge_normal).any():
                        print(edge_normal)
                    other_edge_spoints.append(samples_on_edge)
                    other_edge_ns.append(np.tile(edge_normal, (len(samples_on_edge), 1)))
    else:
        tmp_points=mesh.vertices[non_feature_mask]
        tmp_normals=mesh.vertex_normals[non_feature_mask]

    

    lengths = np.linalg.norm(tmp_normals, axis=-1)

    eps = 1e-12
    valid_mask = lengths > eps

    tmp_points = tmp_points[valid_mask]
    tmp_normals = tmp_normals[valid_mask]
    #================
    non_feature_spoints=tmp_points
    non_feature_ns=tmp_normals
    #-----------------------------


    print('step1 done!')
    
    #-----------------------------
    if(len(other_edge_spoints)>0):
        other_edge_spoints=np.vstack(other_edge_spoints)
        other_edge_ns=np.vstack(other_edge_ns)
    else:
        other_edge_spoints=np.zeros((0,3))
        other_edge_ns=np.zeros((0,3))
    
    print('step2 done!')

    
    #=============================
    convex_feature_spoints=[]
    convex_feature_ns=[]

    convex_feature_ids=data.get('convex_feature_ids',np.array([]))
    if len(convex_feature_ids)>0:
        convex_feature_points=mesh.vertices[convex_feature_ids]
        convex_feaure_normals=mesh.vertex_normals[convex_feature_ids]
    else:
        convex_feature_points=[]
        convex_feaure_normals=[]

    if len(convex_feature_points)>0:
        convex_feature_spoints.append(convex_feature_points)
        convex_feature_ns.append(convex_feaure_normals)

    convex_sharp_edges=data.get('convex_sharp_edges',np.array([]))
    if len(convex_sharp_edges)>0:
        edge_spoints=[]
        edge_ns=[]
        for edge in convex_sharp_edges:
            v1=mesh.vertices[edge[0]]
            v2=mesh.vertices[edge[1]]
            edge_vertice=np.vstack((v1,v2))
            n_edge_samples=int(np.linalg.norm(v2-v1)//(args.m_unit_edge_length*ratio))+1
            #n_edge_samples=non_concave_edge_interior_sample_num
            samples_on_edge=sample_points_on_edge(edge_vertice,n_edge_samples)
            if len(samples_on_edge)>0:
                related_face_ids=edge_to_faces[tuple(sorted(edge))]
                f0=related_face_ids[0]
                f1=related_face_ids[1]
                n0=mesh.face_normals[f0]
                n1=mesh.face_normals[f1]
                edge_normal=stableNormalized(n0+n1)[0]
                if np.isnan(edge_normal).any():
                    print(edge_normal)
                edge_spoints.append(samples_on_edge)
                edge_ns.append(np.tile(edge_normal, (len(samples_on_edge), 1)))
        if len(edge_spoints)>0:
            edge_spoints=np.vstack(edge_spoints)
            edge_ns=np.vstack(edge_ns)
            #====
            convex_feature_spoints.append(edge_spoints)
            convex_feature_ns.append(edge_ns)
    if len(convex_feature_spoints)>0:
        convex_feature_spoints = np.concatenate(convex_feature_spoints, axis=0)
        convex_feature_ns = np.concatenate(convex_feature_ns, axis=0)
    else:
        convex_feature_spoints = np.zeros((0,3))
        convex_feature_ns = np.zeros((0,3))
    #-----------------------------

    print('step3 done!')
    #=============================
    concave_feature_spoints=[]
    concave_feature_ns=[]

    #=============================
    concave_feature_ids=data.get('concave_feature_ids',np.array([]))
    if len(concave_feature_ids)>0:
        concave_feature_points=mesh.vertices[concave_feature_ids]
        concave_feaure_normals=mesh.vertex_normals[concave_feature_ids]
    else:
        concave_feature_points=[]
        concave_feaure_normals=[]
    #concave_fixed_sdfs=np.full((len(concave_feature_points),1),-1e-5)
    if len(concave_feature_points)>0:
        concave_feature_spoints.append(concave_feature_points)#surface points, not need to filter
        concave_feature_ns.append(concave_feaure_normals)#surface normals, not need to filter


    concave_sharp_edges=data.get('concave_sharp_edges',np.array([]))
    if len(concave_sharp_edges)>0:
        edge_spoints=[]
        edge_cpoints=[]
        edge_ns=[]
        
        for edge in concave_sharp_edges:
            v1=mesh.vertices[edge[0]]
            v2=mesh.vertices[edge[1]]
            edge_vertice=np.vstack((v1,v2))
            n_edge_samples=int(np.linalg.norm(v2-v1)//(args.m_unit_edge_length*ratio))+1
            #n_edge_samples=concave_edge_interior_sample_num
            samples_on_edge=sample_points_on_edge(edge_vertice,n_edge_samples)
            if len(samples_on_edge)>0:
                related_face_ids=edge_to_faces[tuple(sorted(edge))]
                f0=related_face_ids[0]
                f1=related_face_ids[1]
                n0=mesh.face_normals[f0]
                n1=mesh.face_normals[f1]
                edge_normal=stableNormalized(n0+n1)[0]
                if np.isnan(edge_normal).any():
                    print(edge_normal)
                edge_spoints.append(samples_on_edge)
                edge_ns.append(np.tile(edge_normal, (len(samples_on_edge), 1)))
        if len(edge_spoints)>0:
            edge_spoints=np.vstack(edge_spoints)
            edge_ns=np.vstack(edge_ns)

            concave_feature_spoints.append(edge_spoints)#surface points, not need to filter
            concave_feature_ns.append(edge_ns)#surface normals, not need to filter



    if len(concave_feature_spoints)>0:
        concave_feature_spoints = np.concatenate(concave_feature_spoints, axis=0)
        concave_feature_ns = np.concatenate(concave_feature_ns, axis=0)
        #-----------------------------
    else:
        concave_feature_spoints = np.zeros((0,3))
        concave_feature_ns = np.zeros((0,3))
        #-----------------------------
    if not args.skip_feature:
        spoints=np.vstack((non_feature_spoints,other_edge_spoints,convex_feature_spoints,concave_feature_spoints))
        ns=np.vstack((non_feature_ns,other_edge_ns,convex_feature_ns,concave_feature_ns))
        if np.isnan(non_feature_ns).any():
            print(non_feature_ns)


        sp_feature_mask=np.zeros(len(spoints),dtype=bool)
        sp_feature_mask[len(non_feature_spoints)+len(other_edge_spoints):]=True

        sp_convex_feature_mask=np.zeros(len(spoints),dtype=bool)
        sp_convex_feature_mask[len(non_feature_spoints)+len(other_edge_spoints):len(non_feature_spoints)+len(other_edge_spoints)+len(convex_feature_spoints)]=True

        sp_concave_feature_mask=np.zeros(len(spoints),dtype=bool)
        sp_concave_feature_mask[len(non_feature_spoints)+len(other_edge_spoints)+len(convex_feature_spoints):]=True
        data['sp_feature_mask']=sp_feature_mask
        data['sp_convex_feature_mask']=sp_convex_feature_mask
        data['sp_concave_feature_mask']=sp_concave_feature_mask
    else:
        spoints=np.vstack((non_feature_spoints,other_edge_spoints))
        ns=np.vstack((non_feature_ns,other_edge_ns))
    data['spoints']=spoints.astype(np.float32)    
    data['ns']=ns.astype(np.float32)
    #-----------------------------
    return data


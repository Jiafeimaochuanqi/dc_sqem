import numpy as np

from sklearn.neighbors import KDTree

def query_sdf_and_sn_and_proj_and_feature_mask(
    sdf_grad_proj_func,
    coords,
    batch_size=-1
):
    """
    coords should be N*M*3 or N*3.
    Return:
        d   shape coords.shape[:-1]
        g   shape coords.shape
        p   shape coords.shape
        ids shape coords.shape[:-1]
        m   shape coords.shape[:-1]
    """
    input_shape = list(coords.shape)
    query_points = coords.reshape(-1, 3)

    if batch_size > 0 and query_points.shape[0] > batch_size:
        batch_num = int(np.ceil(query_points.shape[0] / batch_size))
        pts_list = np.array_split(query_points, batch_num)

        d = []
        g = []
        p = []
        ids = []
        m = []

        for q_per_batch in pts_list:
            temp_d, temp_g, temp_proj, temp_ids, temp_m = sdf_grad_proj_func(q_per_batch)

            d.append(np.asarray(temp_d).reshape(-1))
            g.append(np.asarray(temp_g).reshape(-1, 3))
            p.append(np.asarray(temp_proj).reshape(-1, 3))
            ids.append(np.asarray(temp_ids).reshape(-1))
            m.append(np.asarray(temp_m).reshape(-1))

        d = np.concatenate(d, axis=0)
        g = np.vstack(g)
        p = np.vstack(p)
        ids = np.concatenate(ids, axis=0)
        m = np.concatenate(m, axis=0)

    else:
        d, g, p, ids, m = sdf_grad_proj_func(query_points)

        d = np.asarray(d).reshape(-1)
        g = np.asarray(g).reshape(-1, 3)
        p = np.asarray(p).reshape(-1, 3)
        ids = np.asarray(ids).reshape(-1)
        m = np.asarray(m).reshape(-1)

    scalar_shape = input_shape[:-1]

    g = g.reshape(input_shape)
    p = p.reshape(input_shape)
    d = d.reshape(scalar_shape)
    ids = ids.reshape(scalar_shape)
    m = m.reshape(scalar_shape)

    return d, g, p, ids, m
        
def sdf_sn_proj_from_oriented_points_mask(
    data,
    return_feature_flags=False,
    ratio=np.sin(np.radians(85.0))
):
    surface_points = data['spoints']
    surface_points_ns = data['ns']

    sp_feature_mask = data.get('sp_feature_mask', None)
    sdf_kdtree = KDTree(surface_points)

    def sdf_sn_proj(pts):
        pts = np.asarray(pts)
        _, indexes = sdf_kdtree.query(pts, k=1)
        indexes = indexes.reshape(-1)

        if sp_feature_mask is None:
            target_pts = surface_points[indexes]
            target_ns = surface_points_ns[indexes]

            proj = np.sum((target_pts - pts) * target_ns, axis=1)
            sdf_p = -proj                              # (M,)

            nearest_sdf_points = pts - target_ns * sdf_p[:, np.newaxis]

            dis_offset = np.linalg.norm(nearest_sdf_points - target_pts, axis=1)
            dis = np.linalg.norm(target_pts - pts, axis=1)

            check_mask = dis_offset > ratio * dis
            if np.count_nonzero(check_mask) > 0:
                nearest_sdf_points[check_mask] = target_pts[check_mask]

            feature_flags = np.zeros((pts.shape[0],), dtype=bool)  # (M,)

        else:
            feature_flags = sp_feature_mask[indexes]  # (M,)
            non_feature_flags = ~feature_flags

            target_ns = np.zeros_like(pts)
            sdf_p = np.zeros((pts.shape[0],), dtype=pts.dtype)     # (M,)
            nearest_sdf_points = np.zeros_like(pts)

            # non-feature branch
            non_feature_indexes = indexes[non_feature_flags]

            if np.count_nonzero(non_feature_flags) > 0:
                non_feature_target_pts = surface_points[non_feature_indexes]
                non_feature_target_ns = surface_points_ns[non_feature_indexes]
                non_feature_pts = pts[non_feature_flags]

                non_feature_proj = np.sum(
                    (non_feature_target_pts - non_feature_pts) * non_feature_target_ns,
                    axis=1
                )
                non_feature_sdf_p = -non_feature_proj

                non_feature_nearest_points = (
                    non_feature_pts
                    - non_feature_target_ns * non_feature_sdf_p[:, np.newaxis]
                )

                dis_offset = np.linalg.norm(
                    non_feature_nearest_points - non_feature_target_pts,
                    axis=1
                )
                dis = np.linalg.norm(
                    non_feature_target_pts - non_feature_pts,
                    axis=1
                )

                check_mask = dis_offset > ratio * dis
                if np.count_nonzero(check_mask) > 0:
                    non_feature_nearest_points[check_mask] = non_feature_target_pts[check_mask]

                target_ns[non_feature_flags] = non_feature_target_ns
                sdf_p[non_feature_flags] = non_feature_sdf_p
                nearest_sdf_points[non_feature_flags] = non_feature_nearest_points

            # feature branch
            feature_indexes = indexes[feature_flags]

            if np.count_nonzero(feature_flags) > 0:
                feature_target_pts = surface_points[feature_indexes]
                feature_target_ns = surface_points_ns[feature_indexes]
                feature_pts = pts[feature_flags]

                feature_proj = np.sum(
                    (feature_target_pts - feature_pts) * feature_target_ns,
                    axis=1
                )
                feature_dis = np.linalg.norm(feature_target_pts - feature_pts, axis=1)
                feature_sdf_p = -feature_dis * np.sign(feature_proj)

                target_ns[feature_flags] = feature_target_ns
                sdf_p[feature_flags] = feature_sdf_p
                nearest_sdf_points[feature_flags] = feature_target_pts

        if return_feature_flags:
            return sdf_p, target_ns, nearest_sdf_points, indexes, feature_flags
        else:
            return sdf_p, target_ns, nearest_sdf_points, indexes

    return sdf_sn_proj
import numpy as np

def _normalize_rows(v: np.ndarray, eps: float = 1e-12):
    v = np.asarray(v, dtype=np.float64)
    if v.ndim != 2 or v.shape[1] != 3:
        raise ValueError("input must have shape (N,3)")
    n = np.linalg.norm(v, axis=1, keepdims=True)
    mask = n[:, 0] > eps
    out = np.zeros_like(v, dtype=np.float64)
    out[mask] = v[mask] / n[mask]
    return out, mask

def _circular_max_gap(angles: np.ndarray):
    angles = np.asarray(angles, dtype=np.float64)
    if len(angles) == 0:
        return 2.0 * np.pi
    a = np.sort(angles)
    if len(a) == 1:
        return 2.0 * np.pi
    gaps = np.diff(a)
    wrap_gap = a[0] + 2.0 * np.pi - a[-1]
    return float(max(np.max(gaps), wrap_gap))



def _project_dirs_to_axis_plane(dirs, axis, eps=1e-12):
    axis = np.asarray(axis, dtype=np.float64)
    axis_norm = np.linalg.norm(axis)
    if axis_norm <= eps:
        raise ValueError("zero axis")
    axis = axis / axis_norm
    proj = dirs - np.outer(dirs @ axis, axis)
    proj_norm = np.linalg.norm(proj, axis=1)
    valid_mask = proj_norm > eps
    proj_unit = np.zeros_like(proj)
    proj_unit[valid_mask] = proj[valid_mask] / proj_norm[valid_mask,None]
    return proj_unit, proj_norm, valid_mask

def _plane_angles_from_proj(proj_unit, axis, eps=1e-12):
    axis = np.asarray(axis, dtype=np.float64)
    axis_norm = np.linalg.norm(axis)
    if axis_norm <= eps:
        raise ValueError("zero axis")
    axis = axis / axis_norm
    tmp = np.array([1.0,0.0,0.0], dtype=np.float64)
    if abs(np.dot(tmp, axis)) > 0.9:
        tmp = np.array([0.0,1.0,0.0], dtype=np.float64)
    e1 = tmp - np.dot(tmp, axis)*axis
    e1 = e1 / (np.linalg.norm(e1)+eps)
    e2 = np.cross(axis, e1)
    e2 = e2 / (np.linalg.norm(e2)+eps)
    x = proj_unit @ e1
    y = proj_unit @ e2
    ang = np.mod(np.arctan2(y, x), 2.0*np.pi)
    return ang, e1, e2

def _estimate_axis_from_unit_dirs(unit_dirs, eps=1e-12):
    unit_dirs = np.asarray(unit_dirs, dtype=np.float64)

    if unit_dirs.ndim != 2 or unit_dirs.shape[1] != 3:
        raise ValueError("unit_dirs must have shape (N,3)")

    if len(unit_dirs) < 3:
        raise ValueError("too few representative directions")

    mean_dir = unit_dirs.mean(axis=0)
    centered = unit_dirs - mean_dir[None, :]
    cov = centered.T @ centered

    evals, evecs = np.linalg.eigh(cov)

    axis = evecs[:, 0]
    axis = axis / (np.linalg.norm(axis) + eps)

    if axis[2] < 0:
        axis = -axis

    return axis, evals, evecs, mean_dir, cov

def is_cylindrical_tinf_node(
    dirs: np.ndarray,
    torch_feature=False,
    eps: float = 1e-12,
    min_proj_norm: float = 0.5,
    gap_threshold_deg: float = 165.0,
):
    dirs_unit, valid_dir_mask = _normalize_rows(dirs, eps=eps)
    valid_dirs = dirs_unit[valid_dir_mask]

    if len(valid_dirs) < 3:
        return {
            "is_cylindrical_tinf": False,
            "reason": "too_few_valid_dirs_for_axis",
            "num_raw_dirs": len(valid_dirs),
        }

    axis, evals, evecs, mean_dir, cov = _estimate_axis_from_unit_dirs(valid_dirs, eps)

    if evals[1] * 4 < evals[2]:
        return {
            "is_cylindrical_tinf": False,
            "reason": "no_valid_axis",
            "axis": axis,
            "evals": evals,
        }

    if not torch_feature and evals[0] * 20 > evals[1]:
        return {
            "is_cylindrical_tinf": False,
            "reason": "no_valid_axis",
            "axis": axis,
            "evals": evals,
        }

    proj_unit_all, proj_norm_all, valid_proj_mask = _project_dirs_to_axis_plane(
        valid_dirs, axis, eps
    )

    long_enough = proj_norm_all > min_proj_norm
    valid_for_angle = valid_proj_mask & long_enough
    num_valid_angle = np.sum(valid_for_angle)

    if num_valid_angle >= max(int(len(dirs) * 0.5), 4):
        use_dirs = proj_unit_all[valid_for_angle]
        used_filter = True
    else:
        return {
            "is_cylindrical_tinf": False,
            "reason": "too_few_valid_projections",
            "num_valid_angle": int(num_valid_angle),
            "num_raw_dirs": len(valid_dirs),
            "axis": axis,
        }

    angles, e1, e2 = _plane_angles_from_proj(use_dirs, axis, eps)
    max_gap = _circular_max_gap(angles)
    is_cyl = bool(max_gap < gap_threshold_deg * np.pi / 180.0)

    return {
        "is_cylindrical_tinf": is_cyl,
        "reason": "ok",
        "axis": axis,
        "max_gap": float(max_gap),
        "num_raw_dirs": len(valid_dirs),
        "num_used_dirs": len(use_dirs),
        "used_filter": used_filter,
        "min_proj_norm": min_proj_norm,
        "gap_threshold_deg": gap_threshold_deg,
    }

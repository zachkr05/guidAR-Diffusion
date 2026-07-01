"""
Adaptive Prediction Set (APS) class selection -- which obstacle class(es) is a user's
path edit responsible for.

This is the unchanged, pure-NumPy heart of the old `MoE/offset_finetune.py`: per edit
region (`get_edit_regions`), the region's distance-softmax class responsibilities
(`obtain_probabilities`) are accumulated, largest first, until the cumulative mass
reaches `aps_threshold` -- that set is the affected classes (the conformal method).

In the GPF + Bayesian-inverse-planning pipeline we only consume the *class names*; the
magnitude of the preference is then inferred as (sigma, mu) by `bayesian_inverse.py`,
not read off here. `measure_class_offsets` (the brittle closest-approach selector) is
kept for the `--select geometric` option.

`get_edit_regions` / `obtain_probabilities` were ported verbatim from the old
`utils/utils.py` (which pulled in torch) so this module is torch-free.

Coordinate convention: positions/goal are (row, col); paths are (x, y) = (col, row).
"""

import numpy as np
from scipy.interpolate import interp1d
from skimage.draw import polygon


# --------------------------------------------------------------------------- #
#  Class responsibilities + edit-region segmentation (ported from utils.utils)
# --------------------------------------------------------------------------- #
def obtain_probabilities(obstacle_classes, positions, radii,
                         height=128, width=128, temperature=5.0):
    """Per-cell, per-class responsibility = softmax over (-distance-to-nearest-obstacle).
    Returns {class: (H, W) array summing to 1 across classes at each cell}."""
    rows, cols = np.ogrid[:height, :width]

    pos_dict = positions[0] if isinstance(positions, list) else positions
    rad_dict = radii[0] if isinstance(radii, list) else radii  # noqa: F841 (parity)

    distances = {}
    valid_classes = []
    for cls in obstacle_classes:
        cls_positions = pos_dict.get(cls, [])
        if len(cls_positions) == 0:
            distances[cls] = np.full((height, width), np.inf)
        else:
            min_dist = np.full((height, width), np.inf)
            for pos in cls_positions:
                r, c = pos[0], pos[1]
                dist = np.sqrt((rows - r) ** 2 + (cols - c) ** 2)
                min_dist = np.minimum(min_dist, dist)
            distances[cls] = min_dist
            valid_classes.append(cls)

    dist_stack = np.stack([distances[cls] for cls in obstacle_classes], axis=0)
    min_dist_all = np.min(dist_stack, axis=0, keepdims=True)
    min_dist_all = np.where(np.isinf(min_dist_all), 0, min_dist_all)
    rel_dist = dist_stack - min_dist_all

    closeness_stack = np.exp(-rel_dist / temperature)
    mask = np.array([1.0 if cls in valid_classes else 0.0 for cls in obstacle_classes])
    closeness_stack = closeness_stack * mask[:, None, None]

    sum_closeness = np.sum(closeness_stack, axis=0, keepdims=True) + 1e-8
    resp_stack = closeness_stack / sum_closeness
    return {cls: resp_stack[i] for i, cls in enumerate(obstacle_classes)}


def get_edit_regions(orig_path, user_path, obstacle_classes, batch,
                     height=128, width=128, min_area=50):
    """Identify distinct edit regions by sign changes of the signed area between the
    original and user paths. Returns a list of [class_contributions, edit_points,
    area_mask] tuples. `batch` is (_, _, positions, radii, _, _)."""
    _, _, positions, radii, _, _ = batch
    prob_dict = obtain_probabilities(obstacle_classes, positions, radii, height, width)

    n_samples = max(len(orig_path), len(user_path), 800)

    def resample_path(path, n):
        t_orig = np.linspace(0, 1, len(path))
        t_new = np.linspace(0, 1, n)
        fx = interp1d(t_orig, path[:, 0], kind="linear")
        fy = interp1d(t_orig, path[:, 1], kind="linear")
        return np.column_stack([fx(t_new), fy(t_new)])

    orig_resampled = resample_path(orig_path, n_samples)
    user_resampled = resample_path(user_path, n_samples)

    diff = user_resampled - orig_resampled
    tangent = np.gradient(orig_resampled, axis=0)
    cross = diff[:, 0] * tangent[:, 1] - diff[:, 1] * tangent[:, 0]
    distances = np.linalg.norm(diff, axis=1)

    threshold = 2.0
    is_together = distances < threshold
    sign = np.sign(cross)
    sign[is_together] = 0

    regions = []
    in_region = False
    region_start = 0
    current_sign = 0
    for i in range(n_samples):
        if not in_region:
            if not is_together[i]:
                in_region = True
                region_start = i
                current_sign = sign[i]
        else:
            sign_changed = (sign[i] != 0 and sign[i] != current_sign)
            if is_together[i] or sign_changed:
                regions.append((region_start, i))
                if sign_changed and not is_together[i]:
                    region_start = i
                    current_sign = sign[i]
                    in_region = True
                else:
                    in_region = False
    if in_region:
        regions.append((region_start, n_samples - 1))

    results = []
    for start_idx, end_idx in regions:
        if end_idx - start_idx < 5:
            continue
        orig_segment = orig_resampled[start_idx:end_idx + 1]
        user_segment = user_resampled[start_idx:end_idx + 1]
        if len(orig_segment) < 2:
            continue
        polygon_pts = np.vstack([orig_segment, user_segment[::-1]])
        cluster_mask = np.zeros((height, width), dtype=np.float32)
        rr, cc = polygon(polygon_pts[:, 1], polygon_pts[:, 0], shape=(height, width))
        if len(rr) == 0:
            continue
        cluster_mask[rr, cc] = 1.0
        if cluster_mask.sum() < min_area:
            continue
        ys, xs = np.where(cluster_mask > 0)
        cluster_points = np.column_stack([xs, ys])
        contributions = {}
        for cls in obstacle_classes:
            contributions[cls] = (prob_dict[cls] * cluster_mask).sum() / (cluster_mask.sum() + 1e-8)
        total = sum(contributions.values()) + 1e-8
        contributions = {cls: v / total for cls, v in contributions.items()}
        results.append([contributions, cluster_points, cluster_mask])
    return results


# --------------------------------------------------------------------------- #
#  Geometry helpers
# --------------------------------------------------------------------------- #
def _min_dist_path_to_center(path_xy, center_rc):
    """Closest approach of an (x, y)=(col, row) polyline to an obstacle (row, col)."""
    px = np.asarray(path_xy, dtype=np.float64)
    dx = px[:, 0] - float(center_rc[1])     # x(col) - col
    dy = px[:, 1] - float(center_rc[0])     # y(row) - row
    return float(np.min(np.hypot(dx, dy)))


def _region_displacement(mask, base_path_xy, pct=90.0):
    """How far the path was dragged in an edit region (px): a robust (90th-pct) peak of
    the distance from the region's pixels to the baseline path."""
    ys, xs = np.where(np.asarray(mask) > 0)
    if len(xs) == 0:
        return 0.0
    pts = np.column_stack([xs, ys]).astype(np.float64)            # (n, 2) = (col, row)
    bp = np.asarray(base_path_xy, dtype=np.float64)
    d = np.min(np.linalg.norm(pts[:, None, :] - bp[None, :, :], axis=2), axis=1)
    return float(np.percentile(d, pct))


# --------------------------------------------------------------------------- #
#  Selectors
# --------------------------------------------------------------------------- #
def measure_class_offsets(orig_path_xy, user_path_xy, positions, radii,
                          min_delta=1.0, return_all=False):
    """Geometric selector: per-class largest increase in path clearance (px). Brittle --
    only fires when the path's nearest point to an obstacle moves. Kept for parity."""
    offsets, all_deltas = {}, {}
    for cls, centers in positions.items():
        rads = radii.get(cls, [1] * len(centers))
        best = -np.inf
        for center, rad in zip(centers, rads):
            c_orig = _min_dist_path_to_center(orig_path_xy, center) - float(rad)
            c_user = _min_dist_path_to_center(user_path_xy, center) - float(rad)
            best = max(best, c_user - c_orig)
        best = float(best) if np.isfinite(best) else 0.0
        all_deltas[cls] = best
        if best >= min_delta:
            offsets[cls] = best
    return (offsets, all_deltas) if return_all else offsets


def aps_class_offsets(orig_path_xy, user_path_xy, positions, radii, obstacle_classes,
                      aps_threshold=0.86, H=128, W=128, return_info=False):
    """Adaptive Prediction Set selector over distance-softmax responsibilities.

    Per edit region, accumulate class responsibilities (largest first) until the
    cumulative mass reaches `aps_threshold`; that set is the affected classes. Returns
    {class: displacement_px} (the displacement is informational; the GPF pipeline only
    uses the keys). Robust where `measure_class_offsets` is brittle.
    """
    batch = (None, None, positions, radii, None, None)
    regions = get_edit_regions(orig_path_xy, user_path_xy, list(obstacle_classes),
                               batch, height=H, width=W)

    offsets, info = {}, []
    for contrib, points, mask in regions:
        curr, sel = 0.0, []
        while curr < aps_threshold and len(sel) < len(contrib):
            remaining = {k: v for k, v in contrib.items() if k not in sel}
            best = max(remaining, key=remaining.get)
            curr += float(remaining[best])
            sel.append(best)
        disp = _region_displacement(mask, orig_path_xy)
        for cls in sel:
            offsets[cls] = max(offsets.get(cls, 0.0), disp)
        info.append({"selected": list(sel), "mask_px": int(mask.sum()),
                     "displacement_px": round(disp, 2),
                     "contrib": {k: float(v) for k, v in contrib.items()}})

    offsets = {k: round(v, 2) for k, v in offsets.items() if v > 0.1}
    return (offsets, info) if return_info else offsets

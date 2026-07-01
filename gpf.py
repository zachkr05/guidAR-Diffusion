"""
Gaussian Potential Field (GPF) planner -- the NN-free replacement for the diffusion
generator.

A scene of typed circular obstacles induces, per class, a repulsive Gaussian field
(`Costmap.cost_from_mask_gaussian`, reused from the old data generator). The fields are
fused with a per-class amplitude `mu`, goal attraction is added, and a min-cost path is
routed with skimage's `route_through_array`. That planner is the "approximately rational
agent" whose parameters Bayesian inverse planning inverts (see `bayesian_inverse.py`).

Per-class parameters
    sigma_c : spread of the repulsion (px). Larger -> influence reaches further.
    mu_c    : amplitude / how strongly class c repels (its weight in the fused sum).
Defaults (sigma=6, mu=1) reproduce the original costmap byte-for-byte (regression-tested
in `test_gpf_bip.py`).

Coordinate conventions (same seams as the old diffusion demo):
    scene arrays are indexed [row, col]; positions / goal are (row, col);
    paths handed to the UI / spline layer are (x, y) = (col, row).
"""

import numpy as np
from scipy.ndimage import distance_transform_edt
from skimage.graph import route_through_array

from DataGenerator.sim import Costmap

DEFAULT_SIGMA = 6.0      # repulsion spread (px)
DEFAULT_MU = 1.0         # repulsion strength / per-class amplitude


def start_rc(H, W):
    """Robot start = bottom-right corner (the Costmap convention)."""
    return (H - 1, W - 1)


# --------------------------------------------------------------------------- #
#  Scene sampling (NumPy port of CostmapDataset._sample_scene, torch dropped)
# --------------------------------------------------------------------------- #
def sample_scene(H, W, classes, rng=None,
                 min_num_obstacles=2, max_num_obstacles=4):
    """Randomly sample one scene. Returns a dict:
        {positions:{cls:[(r,c)...]}, radii:{cls:[int...]}, angles:{cls:[float...]},
         goal:(r,c), H, W}.
    """
    rng = np.random.default_rng() if rng is None else rng
    positions, radii, angles = {}, {}, {}
    for cls in classes:
        n = int(rng.integers(min_num_obstacles, max_num_obstacles))
        positions[cls], radii[cls], angles[cls] = [], [], []
        for _ in range(n):
            r = int(rng.integers(10, H - 7))
            c = int(rng.integers(10, W - 7))
            positions[cls].append((r, c))
            radii[cls].append(int(rng.integers(1, 3)))
            angles[cls].append(float(rng.uniform(0, 2 * np.pi)))
    goal = (int(rng.integers(5, H - 5)), int(rng.integers(5, W - 5)))
    return {"positions": positions, "radii": radii, "angles": angles,
            "goal": goal, "H": int(H), "W": int(W)}


# --------------------------------------------------------------------------- #
#  Cost field
# --------------------------------------------------------------------------- #
def _class_mask(centers, radii_c, H, W):
    """Boolean occupancy of one class's true-radius disks."""
    rows, cols = np.ogrid[:H, :W]
    mask = np.zeros((H, W), dtype=bool)
    for (r, c), rad in zip(centers, radii_c):
        mask |= (rows - r) ** 2 + (cols - c) ** 2 <= float(rad) ** 2
    return mask


GOAL_WEIGHT = 0.5        # weight of goal attraction relative to the per-class fields


def build_costmap(scene, params=None):
    """Fused, normalized GPF costmap for a scene.

    params = {cls: (sigma, mu)}; classes absent from `params` use
    (DEFAULT_SIGMA, DEFAULT_MU). Per-class Gaussian fields are summed with weight mu, goal
    attraction is added, and the WHOLE grid is min-max normalized to [0, 1] (plus a small
    positive floor for route_through_array).

    Normalizing keeps the cost scale consistent regardless of mu / number of classes (so the
    inverse-planning rationality beta has a scene-independent meaning). Doing it AFTER goal
    attraction is deliberate: the goal term is independent of mu, so it anchors the scale and
    keeps the obstacle-vs-goal balance -- and hence mu -- identifiable (otherwise a single
    dominant class's mu would cancel under min-max normalization).
    """
    H, W = scene["H"], scene["W"]
    params = params or {}

    fused = np.zeros((H, W), dtype=np.float32)
    for cls, centers in scene["positions"].items():
        sigma, mu = params.get(cls, (DEFAULT_SIGMA, DEFAULT_MU))
        own_mask = _class_mask(centers, scene["radii"][cls], H, W)
        # reuse the proven per-class Gaussian (2*exp(-d^2/2 sigma^2) - 1)
        field = Costmap.cost_from_mask_gaussian(own_mask, sigma=float(sigma))
        fused += float(mu) * field

    # goal attraction: pull the path toward the goal
    gr, gc = int(scene["goal"][0]), int(scene["goal"][1])
    goal_mask = np.zeros((H, W), dtype=np.float32)
    goal_mask[gr, gc] = 1.0
    goal_dist = distance_transform_edt(1.0 - goal_mask).astype(np.float32)
    goal_dist /= goal_dist.max()
    fused = fused + GOAL_WEIGHT * goal_dist

    # normalize the whole grid to [0, 1], then floor strictly positive for the planner
    lo, hi = float(fused.min()), float(fused.max())
    fused = (fused - lo) / (hi - lo + 1e-12)
    return fused.astype(np.float32) + 1e-3


# --------------------------------------------------------------------------- #
#  Planning + path scoring
# --------------------------------------------------------------------------- #
def plan_path(costmap, start, goal):
    """Min-cost path through `costmap`. Returns (path_rc (M,2) int, total_cost)."""
    path_rc, cost = route_through_array(
        costmap, list(start), list(goal), fully_connected=True, geometric=True
    )
    return np.asarray(path_rc, dtype=np.int64), float(cost)


def path_cost_under(costmap, path_rc):
    """Cost of an ARBITRARY (row, col) cell path on `costmap`, using the same
    trapezoidal-times-Euclidean-step accumulation as skimage's MCP_Geometric -- so the
    user path and the optimal path are scored on identical terms."""
    p = np.asarray(path_rc, dtype=np.int64)
    if len(p) < 2:
        return 0.0
    a, b = p[:-1], p[1:]
    ca = costmap[a[:, 0], a[:, 1]]
    cb = costmap[b[:, 0], b[:, 1]]
    step = np.sqrt(((b - a) ** 2).sum(axis=1))
    return float(np.sum(0.5 * (ca + cb) * step))


def rasterize_xy_to_cells(path_xy, H, W):
    """(x, y)=(col, row) polyline -> sequence of unique consecutive (row, col) cells."""
    p = np.asarray(path_xy, dtype=np.float64)
    rc = np.column_stack([
        np.clip(np.round(p[:, 1]), 0, H - 1),
        np.clip(np.round(p[:, 0]), 0, W - 1),
    ]).astype(np.int64)
    keep = np.ones(len(rc), dtype=bool)
    keep[1:] = np.any(rc[1:] != rc[:-1], axis=1)
    return rc[keep]


def plan_path_xy(scene, params=None):
    """Convenience: build the costmap and return the planned path as (x, y)=(col, row)."""
    cm = build_costmap(scene, params)
    path_rc, _ = plan_path(cm, start_rc(scene["H"], scene["W"]), scene["goal"])
    return path_rc[:, ::-1].astype(float), cm


# --------------------------------------------------------------------------- #
#  Smoke test
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    H = W = 128
    rng = np.random.default_rng(0)
    scene = sample_scene(H, W, ["chair", "table", "bomb"], rng=rng)
    cm = build_costmap(scene, {})
    s, g = start_rc(H, W), scene["goal"]
    path_rc, cost = plan_path(cm, s, g)

    assert tuple(path_rc[0]) == s, "path must start at the robot corner"
    assert tuple(path_rc[-1]) == tuple(g), "path must end at the goal"
    print("scene goal=%s  path len=%d  cost=%.3f  -> OK" % (g, len(path_rc), cost))

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(cm, cmap="viridis", origin="upper")
    ax.plot(path_rc[:, 1], path_rc[:, 0], "w-", lw=2)
    ax.plot(g[1], g[0], "g*", ms=16)
    ax.plot(s[1], s[0], "cs", ms=10)
    ax.set_title("GPF costmap + min-cost path (default params)")
    plt.savefig("gpf_smoke.png", dpi=150)
    print("Saved gpf_smoke.png")

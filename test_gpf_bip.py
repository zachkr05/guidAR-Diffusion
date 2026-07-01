"""
Headless verification for the GPF + Bayesian-inverse-planning pipeline.

    python test_gpf_bip.py

  1. regression : gpf.build_costmap(scene, {}) reproduces the legacy fused costmap
                  (independent reconstruction via DataGenerator.sim.Costmap).
  2. bip        : a path that is optimal under a STRONGER chair field is recovered by
                  inverse planning as an elevated chair mu (with ~zero residual), and a
                  wrong-class hypothesis explains it worse.
"""

import numpy as np
from scipy.ndimage import distance_transform_edt

import gpf
import bayesian_inverse
from DataGenerator.sim import Costmap


def _reconstruct_costmap(scene):
    """Independent reconstruction of the normalized GPF costmap at default sigma=6, mu=1,
    built from sim.Costmap's per-class Gaussian -- cross-checks gpf.build_costmap's fusion +
    goal + normalization wiring."""
    H, W = scene["H"], scene["W"]
    rows, cols = np.ogrid[:H, :W]
    fused = np.zeros((H, W), dtype=np.float32)
    for cls, centers in scene["positions"].items():
        own = np.zeros((H, W), dtype=bool)
        for (r, c), rad in zip(centers, scene["radii"][cls]):
            own |= (rows - r) ** 2 + (cols - c) ** 2 <= float(rad) ** 2
        fused += Costmap.cost_from_mask_gaussian(own, sigma=6.0)
    gr, gc = scene["goal"]
    gm = np.zeros((H, W), dtype=np.float32)
    gm[gr, gc] = 1.0
    gd = distance_transform_edt(1.0 - gm).astype(np.float32)
    gd /= gd.max()
    fused = fused + 0.5 * gd
    lo, hi = float(fused.min()), float(fused.max())
    fused = (fused - lo) / (hi - lo + 1e-12)
    return fused.astype(np.float32) + 1e-3


def test_regression():
    scene = gpf.sample_scene(128, 128, ["chair", "table", "bomb"],
                             rng=np.random.default_rng(3))
    got = gpf.build_costmap(scene, {})
    ref = _reconstruct_costmap(scene)
    assert got.shape == ref.shape
    assert np.allclose(got, ref, atol=1e-5), \
        "default GPF costmap diverges from the independent reconstruction"
    assert got.min() >= 1e-3 - 1e-6 and got.max() <= 1.0 + 1e-3 + 1e-6, \
        "normalized grid must lie in [~1e-3, ~1.001]"
    print("[regression] default GPF costmap == independent reconstruction, normalized  (OK)")


def test_bip_recovers_elevated_mu():
    # chair sits squarely on the start->goal diagonal so its strength shapes the path.
    scene = {
        "positions": {"chair": [(32, 32)], "table": [(12, 50)], "bomb": [(50, 12)]},
        "radii": {"chair": [3], "table": [2], "bomb": [2]},
        "angles": {"chair": [0.0], "table": [0.0], "bomb": [0.0]},
        "goal": (5, 5), "H": 64, "W": 64,
    }

    true_params = {"chair": (6.0, 4.0)}                       # a stronger-repelling chair
    user_xy, _ = gpf.plan_path_xy(scene, true_params)         # path the "user" would draw
    base_xy, _ = gpf.plan_path_xy(scene, {})                  # default path

    # the stronger field must actually bend the path, else the test is vacuous
    n = min(len(user_xy), len(base_xy))
    assert np.abs(user_xy[:n] - base_xy[:n]).max() > 1.0, "scene did not produce a real edit"

    params, info = bayesian_inverse.infer(scene, user_xy, ["chair"],
                                          beta=10.0, return_info=True)
    sigma_c, mu_c = params["chair"]
    print("[bip] inferred chair (sigma, mu) = (%.2f, %.2f)  nll=%.4f" %
          (sigma_c, mu_c, info["nll"]))
    assert mu_c > 1.5, "chair repulsion strength should be inferred above the default 1.0"
    assert info["nll"] < 1.0, "the user path should be ~optimal under the inferred field"

    # a wrong-class hypothesis (table, off the path) explains the edit worse.
    _, info_wrong = bayesian_inverse.infer(scene, user_xy, ["table"],
                                           beta=10.0, return_info=True)
    print("[bip] wrong-class (table) residual nll=%.4f vs chair nll=%.4f" %
          (info_wrong["nll"], info["nll"]))
    assert info_wrong["nll"] > info["nll"] + 1e-6, \
        "the responsible class should explain the edit better than an off-path class"
    print("[bip] elevated-mu recovery + discriminative check  (OK)")


if __name__ == "__main__":
    test_regression()
    test_bip_recovers_elevated_mu()
    print("\nAll GPF + BIP checks passed.")

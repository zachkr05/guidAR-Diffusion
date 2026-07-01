"""
Interactive preference-learning demo -- Gaussian Potential Field + Bayesian Inverse Planning
(no neural networks).

Flow:
  1. Sample a scene, build the default GPF costmap, plan the ORIGINAL path.
  2. Show that path as a draggable B-spline over the obstacles. Drag the control points,
     then close the window.
  3. APS picks which obstacle class(es) the edit is responsible for.
  4. Bayesian inverse planning infers those classes' (sigma, mu) so the user's path
     becomes the rational/optimal path; re-plan and overlay original vs. edit vs. adapted.
  5. Generalization: apply the same per-class (sigma, mu) to a FRESH scene and show the
     path bends around the same class -> offset_generalization.png.

Coordinate convention used throughout the UI / spline layer:
    (x, y) = (col, row).  Scene arrays are [row, col]; positions / goal are (row, col).
"""

import argparse
import numpy as np

import matplotlib
matplotlib.use("TkAgg")            # interactive backend; needs a display
import matplotlib.pyplot as plt
import matplotlib.patches as patches

import gpf
from spline import generate_clamped_spline, DraggableBSpline
from aps import aps_class_offsets, measure_class_offsets
import bayesian_inverse


def draw_scene(ax, positions, radii, goal_rc, robot_rc, H, W):
    classes = list(positions.keys())
    colors = plt.cm.Set1(np.linspace(0, 1, max(1, len(classes))))
    for ci, cls in enumerate(classes):
        for (r, c), rad in zip(positions[cls], radii[cls]):
            ax.add_patch(patches.Circle((c, r), float(rad) + 0.5,
                                        color=colors[ci], alpha=0.35))
            ax.add_patch(patches.Circle((c, r), float(rad) + 0.5,
                                        color=colors[ci], fill=False, lw=2))
        ax.plot([], [], "o", color=colors[ci], label=cls)
    ax.plot(goal_rc[1], goal_rc[0], "g*", ms=18, label="goal")
    ax.plot(robot_rc[1], robot_rc[0], "cs", ms=10, label="start")
    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)              # image convention: row increases downward
    ax.set_aspect("equal")
    ax.legend(loc="upper left", fontsize=8)


def show_generalization(classes, params, H, W, seed):
    """Draw a FRESH scene; overlay the default path vs. the path under the inferred
    per-class (sigma, mu) -- the real test that the learned rule transfers."""
    scene = gpf.sample_scene(H, W, classes, rng=np.random.default_rng(seed + 1))
    robot_rc = gpf.start_rc(H, W)
    goal_rc = scene["goal"]

    base_path, _ = gpf.plan_path_xy(scene, {})
    adapt_path, cm = gpf.plan_path_xy(scene, params)

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(cm, cmap="hot", origin="upper", alpha=0.35, extent=[0, W, H, 0])
    draw_scene(ax, scene["positions"], scene["radii"], goal_rc, robot_rc, H, W)
    ax.plot(base_path[:, 0], base_path[:, 1], "w--", lw=2, label="base (default)")
    ax.plot(adapt_path[:, 0], adapt_path[:, 1], color="lime", lw=2.5,
            label="adapted (inferred sigma,mu)")
    ax.legend(loc="upper left", fontsize=8)
    ax.set_title("Generalization to a NEW scene: default vs. inferred field")
    plt.savefig("offset_generalization.png", dpi=150)
    print("Saved offset_generalization.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--obstacle_classes", nargs="+", default=["chair", "table", "bomb"])
    ap.add_argument("--H", type=int, default=128)
    ap.add_argument("--W", type=int, default=128)
    ap.add_argument("--num_ctrl_pts", type=int, default=10)
    ap.add_argument("--degree", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--select", choices=["aps", "geometric"], default="aps",
                    help="how to pick affected classes: 'aps' (distance-softmax + adaptive "
                         "prediction set) or 'geometric' (closest-approach + threshold)")
    ap.add_argument("--aps_threshold", type=float, default=0.86,
                    help="APS cumulative-responsibility cutoff for class selection")
    ap.add_argument("--offset_threshold", type=float, default=1.0,
                    help="geometric selector: min clearance increase (px) to count a class as edited")
    ap.add_argument("--beta", type=float, default=10.0,
                    help="Boltzmann rationality: weight on path-suboptimality in the BIP likelihood")
    args = ap.parse_args()

    H, W = args.H, args.W
    classes = args.obstacle_classes

    # ---- one scene + default GPF path ----
    scene = gpf.sample_scene(H, W, classes, rng=np.random.default_rng(args.seed))
    robot_rc = gpf.start_rc(H, W)
    goal_rc = scene["goal"]
    positions, radii = scene["positions"], scene["radii"]

    orig_path_xy, cm0 = gpf.plan_path_xy(scene, {})

    # ---- fit a clamped B-spline so the user has draggable control points ----
    xs = orig_path_xy[:, 0].astype(float)
    ys = orig_path_xy[:, 1].astype(float)
    ncp = max(args.degree + 1, min(args.num_ctrl_pts, len(xs) - 1))
    x_s0, y_s0, P, U = generate_clamped_spline(xs, ys, k=args.degree, num_ctrl_pts=ncp)
    # spline AS SHOWN (pre-drag) = the baseline for measuring the edit.
    base_path_xy = np.column_stack([x_s0, y_s0])

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(cm0, cmap="hot", origin="upper", alpha=0.35, extent=[0, W, H, 0])
    draw_scene(ax, positions, radii, goal_rc, robot_rc, H, W)
    drag = DraggableBSpline(ax, U, P, args.degree)
    ax.set_title("Drag the CONTROL POINTS (dots on the dashed polygon), then CLOSE the window")
    plt.show()                                  # blocks until the window is closed

    # ---- captured USER path (curve samples reflect final control points) ----
    if drag.x is None or drag.y is None:        # closed without dragging a control point
        user_path_xy = base_path_xy.copy()
    else:
        user_path_xy = np.column_stack([drag.x, drag.y])

    # ---- which class(es) is the edit responsible for? ----
    if args.select == "aps":
        measured, aps_info = aps_class_offsets(
            base_path_xy, user_path_xy, positions, radii, classes,
            aps_threshold=args.aps_threshold, H=H, W=W, return_info=True)
        for r in aps_info:
            print("  edit region (%4d px): contrib=%s | APS@%.2f -> %s" %
                  (r["mask_px"], {k: round(v, 3) for k, v in r["contrib"].items()},
                   args.aps_threshold, r["selected"]))
    else:
        measured, all_deltas = measure_class_offsets(
            base_path_xy, user_path_xy, positions, radii,
            min_delta=args.offset_threshold, return_all=True)
        print("Per-class clearance change (px): %s | threshold=%.2f" %
              ({k: round(v, 2) for k, v in all_deltas.items()}, args.offset_threshold))

    selected = list(measured.keys())
    params = {}
    if not selected:
        print("  -> No affected classes detected. Drag a control point so the path bends "
              "AROUND an obstacle's neighborhood (for 'aps' you can lower --aps_threshold).")
    else:
        print("Affected classes:", selected)
        # ---- Bayesian inverse planning: infer (sigma, mu) for the affected class(es) ----
        params, info = bayesian_inverse.infer(
            scene, user_path_xy, selected, beta=args.beta, return_info=True)
        print("Inferred GPF params {class: (sigma, mu)}:",
              {k: (round(v[0], 2), round(v[1], 2)) for k, v in params.items()})
        print("  BIP residual: nll=%.4f  objective=%.4f" % (info["nll"], info["fun"]))

    # ---- ADAPTED path on the SAME scene (under the inferred field) ----
    adapted_path_xy, cm1 = gpf.plan_path_xy(scene, params)

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(cm1, cmap="hot", origin="upper", alpha=0.35, extent=[0, W, H, 0])
    draw_scene(ax, positions, radii, goal_rc, robot_rc, H, W)
    ax.plot(orig_path_xy[:, 0], orig_path_xy[:, 1], "w--", lw=2, label="original")
    ax.plot(user_path_xy[:, 0], user_path_xy[:, 1], "c-", lw=2, label="user edit")
    ax.plot(adapted_path_xy[:, 0], adapted_path_xy[:, 1],
            color="lime", lw=2.5, label="after inverse planning")
    ax.legend(loc="upper left", fontsize=8)
    ax.set_title("Before vs. user edit vs. inferred field (same scene)")
    plt.savefig("online_finetune_result.png", dpi=150)
    print("Saved online_finetune_result.png")

    # ---- generalization to a NEW scene (the real test of the learned rule) ----
    if params:
        show_generalization(classes, params, H, W, args.seed)

    plt.show()


if __name__ == "__main__":
    main()

# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A **neural-network-free** robot-trajectory preference learner. A scene of typed circular
obstacles plus a goal induces a **Gaussian Potential Field (GPF)**; a min-cost path is
routed through it with `skimage.graph.route_through_array`. The research focus is
**learning a user's preference from a single edit**: the user drags a B-spline over the
path, **Adaptive Prediction Sets (APS)** decide *which obstacle class* the edit concerns,
and **Bayesian Inverse Planning (BIP)** infers that class's GPF parameters
**(σ = spread, μ = repulsion strength)** so the user's path becomes the rational/optimal
path. There are **no neural networks, no torch** — only NumPy/SciPy/scikit-image.

> History: this repo was previously a conditional DDPM that diffused trajectory heatmaps
> and adapted via LoRA+FiLM. That entire stack (`MoE/`, `utils/`, `train.py`,
> `DataGenerator/dataset.py`, checkpoints) was **deleted** in the switch to GPF+BIP. If you
> see references to diffusion/FiLM/LoRA anywhere, they are stale.

## Commands

Install: `pip install -r requirements.txt` (numpy, scipy, scikit-image, matplotlib).

Interactive demo (needs a display — uses the `TkAgg` matplotlib backend):
```
python main.py                                   # default classes: chair table bomb
python main.py --select geometric --beta 10      # geometric class selector; tune rationality
```
Drag the **control-point dots**, then **close the window**. Outputs:
`online_finetune_result.png` (original vs. user edit vs. inferred-field path, same scene)
and `offset_generalization.png` (the inferred per-class (σ, μ) applied to a fresh scene).

Headless checks (no display):
```
python test_gpf_bip.py     # regression (GPF == legacy field) + BIP recovery + discriminative
python -m gpf              # sample a scene, build the field, plan a path -> gpf_smoke.png
python -m DataGenerator.sim # legacy costmap/trajectory viz (kept as the regression reference)
```

## Architecture

The pipeline is four small modules; `main.py` wires them together.

### 1. GPF planner — `gpf.py` (the forward model)
- `sample_scene(H, W, classes, rng)` → `{positions:{cls:[(r,c)…]}, radii, angles, goal, H, W}`.
- `build_costmap(scene, params)` → fused costmap. `params = {cls:(σ, μ)}`; classes absent
  from `params` use `(DEFAULT_SIGMA=6, DEFAULT_MU=1)`. The per-class field is the proven
  `Costmap.cost_from_mask_gaussian` (`2·exp(−d²/2σ²) − 1`, reused from `DataGenerator/sim.py`),
  fused as `Σ_c μ_c · field_c`, plus `0.5·goal_dist` attraction, then the **whole grid is
  min-max normalized to [0,1]** (+ a `1e-3` floor for the planner) so the cost scale is
  consistent regardless of μ / class count. Normalizing *after* goal attraction is deliberate:
  the goal term anchors the scale so μ stays identifiable (test-verified against an independent
  reconstruction).
- `plan_path(costmap, start, goal)` → `(path_rc, cost)` via `route_through_array`;
  `plan_path_xy(scene, params)` is the convenience that returns the path as `(x, y)`.
- `path_cost_under(costmap, path_rc)` scores an **arbitrary** cell path with the same
  trapezoidal × Euclidean-step accumulation as `MCP_Geometric`, so the user path and the
  optimal path are comparable (this is what BIP's likelihood needs).

### 2. APS class selection — `aps.py` (pure NumPy)
`aps_class_offsets(orig_xy, user_xy, positions, radii, classes, aps_threshold=0.86, …)`
segments the edit into regions (`get_edit_regions`), computes per-class distance-softmax
responsibilities (`obtain_probabilities`), and accumulates them largest-first until the
cumulative mass hits `aps_threshold` — that set is the affected classes. The GPF pipeline
consumes **only the class names**; the magnitude is inferred by BIP, not read off here.
`get_edit_regions`/`obtain_probabilities` were ported from the old `utils/utils.py` (which
pulled in torch). `measure_class_offsets` is the brittle geometric `--select geometric` alt.

### 3. Bayesian Inverse Planning — `bayesian_inverse.py` (the inversion)
`infer(scene, user_path_xy, selected, beta=10.0, …)` → `{cls:(σ*, μ*)}` for the selected
classes (others stay at defaults). Following Baker, Saxe & Tenenbaum (2009), the agent is
Boltzmann-rational, so the trajectory likelihood is
`P(user | θ) ∝ exp(−β·[C_θ(user) − C_θ(optimal_θ)])` — the user path is penalized by how
**suboptimal** it is under the candidate field. The MAP posterior (Gaussian prior centered
at the defaults, so ties break toward the *smallest* (σ, μ) change that explains the edit)
is found by **continuous derivative-free optimization** — multi-start `scipy.optimize.minimize`
(Nelder-Mead), because the objective is continuous but kinked (the `max(0,·)` and where the
optimal path switches). `beta` trades fit vs. prior; per-class bounds are `SIGMA_BOUNDS`,
`MU_BOUNDS`.

### 4. Edit UI — `spline.py`
`generate_clamped_spline(x, y, k, num_ctrl_pts)` fits a clamped B-spline; `DraggableBSpline`
makes its control points draggable on a Matplotlib axes. `.x`/`.y` hold the sampled curve and
stay `None` until a control point is first dragged (main.py falls back to the baseline then).

### `DataGenerator/`
`sim.py` `Costmap` is kept as the **legacy GPF/geometry reference**: `cost_from_mask_gaussian`
(reused by `gpf.build_costmap`) and the original `calculateCost` (used only by the regression
test and its own `__main__`). It is torch-free. `dataset.py` was deleted.

### Coordinate conventions (a frequent source of bugs)
- Scene/costmap arrays are indexed `[row, col]`; `positions`, `goal`, and `route_through_array`
  use `(row, col)`. The matplotlib UI, `DraggableBSpline`, and path-xy use `(x, y) = (col, row)`.
- Conversions happen explicitly at the seams (a planned `path_rc[:, ::-1]` becomes `path_xy`;
  `rasterize_xy_to_cells` goes back). Default robot start is the bottom-right corner `(H-1, W-1)`.

There is no linter or CI; `test_gpf_bip.py` is the test suite.

"""
Bayesian Inverse Planning over Gaussian-Potential-Field parameters.

Given a user's single edit to a planned path and the class(es) APS flagged as
responsible, infer those classes' GPF parameters -- sigma (spread) and mu (repulsion
strength) -- that make the user's path the (approximately) rational/optimal path. This
inverts `gpf`'s forward planner.

Following Baker, Saxe & Tenenbaum (2009), the agent is Boltzmann-rational: it prefers
low-cost paths, so the trajectory likelihood is

    P(user_path | theta) proportional to exp(-beta * [C_theta(user) - C_theta(optimal)]),

i.e. the user path is penalized by how *suboptimal* it is under the candidate field.
The posterior P(theta | user_path) ~ P(user_path | theta) * P(theta) is maximized (MAP)
by continuous optimization. A Gaussian prior centered at the defaults (sigma=6, mu=1)
breaks ties toward the smallest (sigma, mu) change that explains the edit (Occam).

Returns {class: (sigma*, mu*)} for the selected classes; non-selected classes keep the
GPF defaults.
"""

import numpy as np
from scipy.optimize import minimize

from gpf import (
    build_costmap, plan_path, path_cost_under, rasterize_xy_to_cells,
    start_rc, DEFAULT_SIGMA, DEFAULT_MU,
)

# per-class search bounds
SIGMA_BOUNDS = (1.5, 20.0)
MU_BOUNDS = (0.1, 8.0)


def _clip_theta(theta):
    t = np.array(theta, dtype=float).reshape(-1, 2)
    t[:, 0] = np.clip(t[:, 0], *SIGMA_BOUNDS)
    t[:, 1] = np.clip(t[:, 1], *MU_BOUNDS)
    return t


def _unpack(theta, selected):
    t = _clip_theta(theta)
    return {cls: (float(t[i, 0]), float(t[i, 1])) for i, cls in enumerate(selected)}


def _data_nll(theta, scene, user_cells, selected, beta):
    """beta * max(0, C_theta(user) - C_theta(optimal)) >= 0."""
    cm = build_costmap(scene, _unpack(theta, selected))
    opt_rc, _ = plan_path(cm, start_rc(scene["H"], scene["W"]), scene["goal"])
    c_user = path_cost_under(cm, user_cells)
    c_opt = path_cost_under(cm, opt_rc)
    return beta * max(0.0, c_user - c_opt)


def _neg_log_post(theta, scene, user_cells, selected, beta,
                  prior_sigma, prior_mu, sigma_sd, mu_sd):
    nll = _data_nll(theta, scene, user_cells, selected, beta)
    t = _clip_theta(theta)
    # -log Gaussian prior (constants dropped)
    nlp = 0.5 * np.sum(((t[:, 0] - prior_sigma) / sigma_sd) ** 2
                       + ((t[:, 1] - prior_mu) / mu_sd) ** 2)
    return nll + nlp


def infer(scene, user_path_xy, selected, beta=10.0,
          prior_sigma=DEFAULT_SIGMA, prior_mu=DEFAULT_MU,
          sigma_sd=4.0, mu_sd=2.0, restarts=None, return_info=False):
    """MAP estimate of per-selected-class (sigma, mu).

    The objective is continuous but kinked (at the max(0, .) and where the optimal path
    switches), so we use derivative-free Nelder-Mead from a few starts and keep the best.
    """
    selected = list(selected)
    if not selected:
        return ({}, {"nll": 0.0, "fun": 0.0}) if return_info else {}

    user_cells = rasterize_xy_to_cells(user_path_xy, scene["H"], scene["W"])
    args = (scene, user_cells, selected, beta,
            prior_sigma, prior_mu, sigma_sd, mu_sd)

    if restarts is None:
        n = len(selected)
        restarts = [
            np.tile([DEFAULT_SIGMA, DEFAULT_MU], n),   # defaults
            np.tile([9.0, 3.0], n),                    # wider + stronger
            np.tile([14.0, 5.0], n),                   # much stronger
        ]

    best = None
    for x0 in restarts:
        res = minimize(_neg_log_post, np.asarray(x0, dtype=float), args=args,
                       method="Nelder-Mead",
                       options={"xatol": 1e-2, "fatol": 1e-4, "maxiter": 600})
        if best is None or res.fun < best.fun:
            best = res

    params = _unpack(best.x, selected)
    if return_info:
        info = {"fun": float(best.fun),
                "nll": float(_data_nll(best.x, scene, user_cells, selected, beta)),
                "iters": int(best.get("nit", 0))}
        return params, info
    return params

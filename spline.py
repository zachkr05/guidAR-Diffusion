"""
Clamped B-spline fitting + an interactive draggable-control-point spline for the edit UI.
Pure NumPy/SciPy/Matplotlib (moved out of MoE/, torch dependency dropped).

Convention: control points and curve samples are (x, y) = (col, row).
`DraggableBSpline.x` / `.y` hold the sampled curve; they stay None until a control point
is first dragged.
"""

import numpy as np
from scipy.interpolate import BSpline


def find_span(n, p, u, U):
    """Find the knot span index for parameter u (n = n_ctrl - 1)."""
    if u >= U[n + 1]:
        return n
    if u <= U[p]:
        return p
    low, high = p, n + 1
    mid = (low + high) // 2
    while u < U[mid] or u >= U[mid + 1]:
        if u < U[mid]:
            high = mid
        else:
            low = mid
        mid = (low + high) // 2
    return mid


def basis_funs(span, u, p, U):
    """Compute the nonzero B-spline basis functions N_{span-p...span,p}(u)."""
    N = np.zeros(p + 1, dtype=float)
    left = np.zeros(p + 1, dtype=float)
    right = np.zeros(p + 1, dtype=float)
    N[0] = 1.0
    for j in range(1, p + 1):
        left[j] = u - U[span + 1 - j]
        right[j] = U[span + j] - u
        saved = 0.0
        for r in range(j):
            denom = right[r + 1] + left[j - r]
            temp = 0.0 if denom == 0 else N[r] / denom
            N[r] = saved + right[r + 1] * temp
            saved = left[j - r] * temp
        N[j] = saved
    return N


def bspline_design_matrix(u, n_ctrl, degree, U):
    p = degree
    n = n_ctrl - 1
    A = np.zeros((len(u), n_ctrl), dtype=float)
    for j, uj in enumerate(u):
        span = find_span(n, p, uj, U)
        N = basis_funs(span, uj, p, U)  # length p+1
        i0 = span - p
        A[j, i0:i0 + p + 1] = N
    return A


def clamped_knots(num_ctrl_pts, degree):
    p = degree
    m = num_ctrl_pts + p
    U = np.zeros(m + 1, dtype=float)
    U[:p + 1] = 0.0
    U[m - p:] = 1.0
    num_interior = num_ctrl_pts - p - 1
    U[p + 1:m - p] = np.linspace(0.0, 1.0, num_interior + 2)[1:-1]
    return U


def reparam_curve(Q):
    d = np.linalg.norm(Q[1:] - Q[:-1], axis=1)
    total = d.sum()
    u = np.concatenate(([0.0], np.cumsum(d) / total))
    u[-1] = 1.0
    return u


def generate_clamped_spline(x_np, y_np, k, num_ctrl_pts):
    Q = np.column_stack([x_np, y_np])
    u = reparam_curve(Q)
    U = clamped_knots(num_ctrl_pts, k)
    A = bspline_design_matrix(u, num_ctrl_pts, k, U)

    P, *_ = np.linalg.lstsq(A, Q, rcond=None)
    P[0] = Q[0]
    P[-1] = Q[-1]
    P = np.asarray(P)

    t0, t1 = U[k], U[-k - 1]
    t = np.linspace(t0, t1, 800)

    splx = BSpline(U, P[:, 0], k)
    sply = BSpline(U, P[:, 1], k)
    return splx(t), sply(t), P, U


class DraggableBSpline:
    """Draggable control-point B-spline overlaid on a Matplotlib axes."""

    def __init__(self, ax, U, P, k, n_samples=600):
        self.ax = ax
        self.U = np.asarray(U, float)
        self.P = np.asarray(P, float).copy()
        self.k = int(k)
        self.n_samples = int(n_samples)

        self.t0, self.t1 = self.U[self.k], self.U[-self.k - 1]
        self.t = np.linspace(self.t0, self.t1, self.n_samples)
        self.t[-1] = self.t1

        self.curve_line, = ax.plot([], [], linewidth=2, label="B-spline curve")
        self.poly_line, = ax.plot(self.P[:, 0], self.P[:, 1], "o--", alpha=0.7,
                                  label="control polygon")

        self._active_idx = None
        self._pick_tol = 8  # pixels

        self.x = None
        self.y = None
        self._update_curve()

        self.cid_press = ax.figure.canvas.mpl_connect("button_press_event", self.on_press)
        self.cid_release = ax.figure.canvas.mpl_connect("button_release_event", self.on_release)
        self.cid_move = ax.figure.canvas.mpl_connect("motion_notify_event", self.on_move)

    def _update_curve(self):
        splx = BSpline(self.U, self.P[:, 0], self.k)
        sply = BSpline(self.U, self.P[:, 1], self.k)
        x_s = splx(self.t)
        y_s = sply(self.t)
        self.x = x_s
        self.y = y_s
        self.curve_line.set_data(x_s, y_s)
        self.poly_line.set_data(self.P[:, 0], self.P[:, 1])
        self.ax.figure.canvas.draw_idle()

    def _closest_point_index(self, event):
        if event.xdata is None or event.ydata is None:
            return None
        pts_disp = self.ax.transData.transform(self.P)
        mouse_disp = np.array([event.x, event.y])
        d2 = np.sum((pts_disp - mouse_disp) ** 2, axis=1)
        idx = int(np.argmin(d2))
        if np.sqrt(d2[idx]) <= self._pick_tol:
            return idx
        return None

    def on_press(self, event):
        if event.inaxes != self.ax or event.button != 1:
            return
        idx = self._closest_point_index(event)
        if idx is not None:
            self._active_idx = idx

    def on_release(self, event):
        self._active_idx = None

    def on_move(self, event):
        if self._active_idx is None:
            return
        if event.inaxes != self.ax:
            return
        if event.xdata is None or event.ydata is None:
            return
        self.P[self._active_idx, 0] = event.xdata
        self.P[self._active_idx, 1] = event.ydata
        self._update_curve()

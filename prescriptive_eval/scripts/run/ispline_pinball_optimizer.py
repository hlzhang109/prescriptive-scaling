#!/usr/bin/env python3
"""
ispline_pinball_optimizer.py

Fit a monotone, saturating I-spline frontier to (z, y) data by minimizing a smoothed pinball
(quantile) loss plus light regularization.

Frontier model (strictly more general than a 4-parameter sigmoid):

    q(z) = y0 + L * sigmoid( a0 + X(z)^T w ),
    w = softplus(beta) >= 0

where X(z) is a clamped cubic I-spline feature vector over z=log10(FLOPs). Each I-spline basis is
monotone nondecreasing; nonnegative weights make the linear predictor monotone, and the logistic
link makes the boundary monotone and saturating in [y0, y0+L] ⊂ [0,1].

Key implementation details:
- Cubic I-splines are built by integrating normalized M-splines using SciPy's BSpline(...).antiderivative().
- Constraints: y0 = expit(u), L = (1 - y0) * expit(v), w = softplus(beta).
- Objective: mean smoothed pinball loss + λ_beta * ||beta||^2 (+ optional λ_a0 * a0^2).
- Optimization: multi-start over knot-midpoint "centers" and scales + random restarts, solved with L-BFGS-B
  on unconstrained params (u, v, a0, beta), with a0 bounded to [-8, 8].

CLI:
  --csv PATH              input CSV containing z and task columns (y in [0,1])
  --tasks t1,t2,...       comma-separated task column names
  --z_col NAME            column for z (default: auto-detect; expects log10 FLOPs)
  --tau FLOAT             quantile level (default 0.98)
  --kappa_final FLOAT     smoothing kappa (default 50)
  --lambda_beta FLOAT     L2 regularization on beta (default 1e-4)
  --lambda_a0 FLOAT       optional L2 regularization on a0 (default 0)
  --n_knot_grid INT       number of equal-mass knot intervals (default 8)
  --n_random INT          number of random restarts (default 20)
  --maxiter INT           max optimizer iterations per start (default 700)
  --seed INT              RNG seed (default 0)
  --out_dir DIR           output directory (writes results.{csv,json} and optional plots/)
  --plot                  save quick scatter+fit plots to out_dir/plots

Outputs:
- results.csv: one row per task with loss, coverage, coverage MAE, and fitted parameters
- results.json: full parameter vectors and knot edges per task
- plots/<task>.png (optional)

Only NumPy, SciPy, and Matplotlib are used (plus Python stdlib).
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.interpolate import BSpline
from scipy.optimize import minimize
from scipy.special import expit

# ----------------------------
# Numerics helpers
# ----------------------------


def softplus(x: np.ndarray) -> np.ndarray:
    """Stable softplus."""
    return np.logaddexp(0.0, x)


def inv_softplus(y: np.ndarray) -> np.ndarray:
    """Stable-ish inverse of softplus for y>=0 (used only for initialization)."""
    y = np.asarray(y, float)
    # For large y, softplus(x) ~ x, so inverse ~ y.
    out = np.where(y > 20.0, y, np.log(np.expm1(np.maximum(y, 1e-12))))
    # For exactly 0, inverse is -inf; cap.
    out = np.where(np.isfinite(out), out, -30.0)
    return out


def logit(p: float, eps: float = 1e-9) -> float:
    p = float(np.clip(p, eps, 1.0 - eps))
    return float(np.log(p) - np.log1p(-p))


def smooth_pinball_loss(r: np.ndarray, tau: float, kappa: float) -> np.ndarray:
    """Smoothed pinball loss (elementwise) for residual r = y - yhat."""
    r = np.asarray(r, float)
    return np.logaddexp(0.0, kappa * r) / kappa + (tau - 1.0) * r


def smooth_pinball_grad_r(r: np.ndarray, tau: float, kappa: float) -> np.ndarray:
    """Derivative d/d r of smooth_pinball_loss."""
    r = np.asarray(r, float)
    return expit(kappa * r) + (tau - 1.0)


# ----------------------------
# I-spline basis construction
# ----------------------------


def build_knot_edges_equal_mass(z: np.ndarray, n_bins: int) -> np.ndarray:
    """Equal-mass knot edges (quantiles) over z. Returns unique sorted edges."""
    z = np.asarray(z, float)
    if z.size == 0:
        return np.array([], dtype=float)
    qs = np.linspace(0.0, 1.0, int(n_bins) + 1)
    edges = np.quantile(z, qs)
    edges = np.unique(edges)
    if edges.size < 2:
        zmin = float(np.min(z))
        zmax = float(np.max(z))
        if zmax > zmin:
            edges = np.array([zmin, zmax], dtype=float)
        else:
            edges = np.array([zmin, zmin + 1.0], dtype=float)
    return edges


def build_ispline_features(z: np.ndarray, edges: np.ndarray, degree: int = 3) -> np.ndarray:
    """Clamped cubic I-spline features via integrated normalized M-splines.

    - edges are treated as knot locations; clamped boundary knots at edges[0], edges[-1].
    - Returns X shape (N, n_basis). Each column is monotone nondecreasing and saturates to 0/1
      outside the knot range.
    """
    z = np.asarray(z, float)
    edges = np.asarray(edges, float)

    if z.size == 0:
        return np.zeros((0, 0), dtype=float)

    edges = np.unique(edges)
    if edges.size < 2:
        return np.zeros((z.size, 0), dtype=float)

    p = int(degree)
    z0, z1 = float(edges[0]), float(edges[-1])
    if not np.isfinite(z0) or not np.isfinite(z1) or z1 <= z0:
        return np.zeros((z.size, 0), dtype=float)

    # Clamped knot vector
    t = np.r_[np.full(p + 1, z0), edges[1:-1], np.full(p + 1, z1)]
    n_basis = len(t) - p - 1
    if n_basis <= 0:
        return np.zeros((z.size, 0), dtype=float)

    # Evaluate inside domain; saturate outside
    zc = np.clip(z, z0, z1)
    k = p + 1  # M-spline order

    X = np.empty((z.size, n_basis), dtype=float)
    for i in range(n_basis):
        denom = float(t[i + k] - t[i])
        if denom <= 0.0:
            X[:, i] = 0.0
            continue

        # M_i(x) = k * B_i(x) / (t[i+k] - t[i])  (normalized to integrate to 1)
        c = np.zeros(n_basis, dtype=float)
        c[i] = k / denom
        M = BSpline(t, c, p, extrapolate=False)

        # I_i(x) = integral M_i from left boundary to x
        I = M.antiderivative()
        base = float(I(z0))
        vals = I(zc) - base

        # Saturate / clamp numerically
        vals = np.clip(vals, 0.0, 1.0)
        vals = np.where(z <= z0, 0.0, vals)
        vals = np.where(z >= z1, 1.0, vals)
        X[:, i] = vals

    return X


# ----------------------------
# Convex (y0, L) solve given shape s(z)
# ----------------------------


def solve_y0_L_given_s(y: np.ndarray, s: np.ndarray, tau: float, kappa: float) -> Tuple[float, float]:
    """Solve for (y0, L) given fixed s in [0,1], minimizing mean smoothed pinball loss.

    Constraints:
      y0 >= 0
      L  >= 0
      y0 + L <= 1

    This subproblem is convex because predictions are affine in (y0, L) and the loss is convex in predictions.
    """
    y = np.asarray(y, float)
    s = np.asarray(s, float)

    lo = float(np.clip(np.quantile(y, 0.05), 0.0, 1.0))
    hi = float(np.clip(np.quantile(y, 0.95), 0.0, 1.0))
    y0_init = float(np.clip(lo, 0.0, 0.99))
    L_init = float(np.clip(hi - y0_init, 1e-3, 1.0 - y0_init))
    x0 = np.array([y0_init, L_init], dtype=float)

    def fun(x: np.ndarray) -> float:
        y0, L = float(x[0]), float(x[1])
        yhat = y0 + L * s
        r = y - yhat
        return float(np.mean(smooth_pinball_loss(r, tau=tau, kappa=kappa)))

    def jac(x: np.ndarray) -> np.ndarray:
        y0, L = float(x[0]), float(x[1])
        yhat = y0 + L * s
        r = y - yhat
        g_r = smooth_pinball_grad_r(r, tau=tau, kappa=kappa)  # dℓ/dr
        g_q = -g_r / y.size  # d(meanℓ)/dq
        d_y0 = float(np.sum(g_q))              # dq/dy0 = 1
        d_L = float(np.sum(g_q * s))           # dq/dL = s
        return np.array([d_y0, d_L], dtype=float)

    cons = (
        {"type": "ineq", "fun": lambda x: x[0]},                  # y0 >= 0
        {"type": "ineq", "fun": lambda x: x[1]},                  # L  >= 0
        {"type": "ineq", "fun": lambda x: 1.0 - x[0] - x[1]},     # y0 + L <= 1
    )
    bounds = [(0.0, 1.0), (0.0, 1.0)]

    res = minimize(
        fun,
        x0,
        method="SLSQP",
        jac=jac,
        bounds=bounds,
        constraints=cons,
        options={"maxiter": 200, "ftol": 1e-10},
    )
    if res.success and np.all(np.isfinite(res.x)):
        y0, L = float(res.x[0]), float(res.x[1])
    else:
        y0, L = float(x0[0]), float(x0[1])

    y0 = float(np.clip(y0, 0.0, 1.0))
    L = float(np.clip(L, 0.0, 1.0 - y0))
    return y0, L


# ----------------------------
# Fitting
# ----------------------------


@dataclass
class FitResult:
    params: np.ndarray  # [u, v, a0, beta...]
    loss: float
    success: bool
    message: str
    edges: np.ndarray


def _objective_and_grad(
    theta: np.ndarray,
    X: np.ndarray,
    y: np.ndarray,
    tau: float,
    kappa: float,
    lambda_beta: float,
    lambda_a0: float,
) -> Tuple[float, np.ndarray]:
    """Return (objective, gradient) for unconstrained params theta = [u, v, a0, beta...]"""
    u = float(theta[0])
    v = float(theta[1])
    a0 = float(theta[2])
    beta = np.asarray(theta[3:], float)

    y0 = expit(u)
    lv = expit(v)
    L = (1.0 - y0) * lv
    w = softplus(beta)

    t = a0 + X.dot(w)
    s = expit(t)
    q = y0 + L * s

    r = y - q
    loss = float(np.mean(smooth_pinball_loss(r, tau=tau, kappa=kappa)))
    reg = lambda_beta * float(beta.dot(beta)) + (lambda_a0 * (a0 * a0) if lambda_a0 != 0.0 else 0.0)
    obj = loss + reg

    # Gradients
    g_r = smooth_pinball_grad_r(r, tau=tau, kappa=kappa)  # dℓ/dr
    g_q = -g_r / y.size  # d(meanℓ)/dq

    dy0_du = y0 * (1.0 - y0)
    dlv_dv = lv * (1.0 - lv)
    dL_du = -dy0_du * lv
    dL_dv = (1.0 - y0) * dlv_dv

    ds_dt = s * (1.0 - s)

    dq_du = dy0_du + dL_du * s
    dq_dv = dL_dv * s
    dq_da0 = L * ds_dt

    # beta: dq/dbeta_j = L * ds_dt * X[:,j] * d softplus / d beta_j
    dw_dbeta = expit(beta)  # derivative of softplus
    tmp = (L * ds_dt)[:, None] * X  # (n, J)
    dq_dbeta = tmp * dw_dbeta[None, :]  # (n, J)

    grad_u = float(np.sum(g_q * dq_du))
    grad_v = float(np.sum(g_q * dq_dv))
    grad_a0 = float(np.sum(g_q * dq_da0)) + (2.0 * lambda_a0 * a0 if lambda_a0 != 0.0 else 0.0)
    grad_beta = dq_dbeta.T.dot(g_q) + 2.0 * lambda_beta * beta

    grad = np.concatenate(([grad_u, grad_v, grad_a0], grad_beta.astype(float)))
    return obj, grad


def _predict_from_params(z: np.ndarray, params: np.ndarray, edges: np.ndarray) -> np.ndarray:
    z = np.asarray(z, float)
    X = build_ispline_features(z, edges, degree=3)
    u, v, a0 = float(params[0]), float(params[1]), float(params[2])
    beta = np.asarray(params[3:], float)

    y0 = expit(u)
    L = (1.0 - y0) * expit(v)
    w = softplus(beta)
    s = expit(a0 + X.dot(w))
    return y0 + L * s


def fit_ispline_enhanced(
    z: np.ndarray,
    y: np.ndarray,
    tau: float = 0.98,
    kappa_final: float = 50.0,
    lambda_beta: float = 1e-4,
    n_knot_grid: int = 8,
    n_random: int = 20,
    seed: int = 0,
    maxiter: int = 700,
    lambda_a0: float = 0.0,
) -> FitResult:
    """Robust multi-start I-spline fit. Returns best solution among starts."""
    z = np.asarray(z, float)
    y = np.asarray(y, float)

    mask = np.isfinite(z) & np.isfinite(y)
    z = z[mask]
    y = y[mask]

    if z.size < 10:
        return FitResult(
            params=np.array([], dtype=float),
            loss=float("nan"),
            success=False,
            message="not enough valid data",
            edges=np.array([], dtype=float),
        )

    edges = build_knot_edges_equal_mass(z, n_knot_grid)
    X = build_ispline_features(z, edges, degree=3)
    J = X.shape[1]
    if J == 0:
        return FitResult(
            params=np.array([], dtype=float),
            loss=float("nan"),
            success=False,
            message="failed to build I-spline basis",
            edges=edges,
        )

    rng = np.random.default_rng(seed)

    mids = 0.5 * (edges[:-1] + edges[1:]) if edges.size >= 2 else np.array([float(np.median(z))])
    z_range = float(np.max(z) - np.min(z))
    eps = max(1e-4, 1e-3 * (z_range if z_range > 0 else 1.0))
    scales = [0.0, 0.5, 1.0, 2.0, 4.0]  # shape steepness of g(z)

    starts: List[np.ndarray] = []

    # Structured starts: concentrate slope around a knot midpoint
    for z_star in mids:
        Xp = build_ispline_features(np.array([z_star + eps]), edges, degree=3)
        Xm = build_ispline_features(np.array([z_star - eps]), edges, degree=3)
        d = (Xp - Xm)[0] / (2.0 * eps)  # approximate derivative of each I-spline at z_star
        d = np.maximum(d, 0.0)
        norm = float(np.linalg.norm(d))
        if norm > 0.0:
            d = d / norm

        Xs = build_ispline_features(np.array([z_star]), edges, degree=3)[0]

        for sscale in scales:
            w0 = sscale * d
            beta0 = np.clip(inv_softplus(w0 + 1e-6), -10.0, 10.0)
            a0_0 = float(-Xs.dot(w0))  # makes t(z_star) ~= 0
            a0_0 = float(np.clip(a0_0, -2.0, 2.0))

            svec = expit(a0_0 + X.dot(w0))
            y0_0, L_0 = solve_y0_L_given_s(y, svec, tau=tau, kappa=kappa_final)
            u0 = logit(y0_0)
            ratio = float(np.clip(L_0 / max(1e-12, (1.0 - y0_0)), 1e-6, 1.0 - 1e-6))
            v0 = logit(ratio)

            starts.append(np.concatenate(([u0, v0, a0_0], beta0.astype(float))))

    # Random starts
    for _ in range(int(n_random)):
        beta0 = rng.normal(loc=-2.0, scale=1.5, size=J)
        a0_0 = float(rng.uniform(-2.0, 2.0))
        w0 = softplus(beta0)
        svec = expit(a0_0 + X.dot(w0))
        y0_0, L_0 = solve_y0_L_given_s(y, svec, tau=tau, kappa=kappa_final)
        u0 = logit(y0_0)
        ratio = float(np.clip(L_0 / max(1e-12, (1.0 - y0_0)), 1e-6, 1.0 - 1e-6))
        v0 = logit(ratio)
        starts.append(np.concatenate(([u0, v0, a0_0], beta0.astype(float))))

    bounds = [(None, None), (None, None), (-8.0, 8.0)] + [(None, None)] * J

    best_res = None
    best_obj = float("inf")

    for theta0 in starts:
        def fun(th: np.ndarray) -> float:
            obj, _ = _objective_and_grad(th, X, y, tau, kappa_final, lambda_beta, lambda_a0)
            return obj

        def jac(th: np.ndarray) -> np.ndarray:
            _, g = _objective_and_grad(th, X, y, tau, kappa_final, lambda_beta, lambda_a0)
            return g

        res = minimize(
            fun,
            theta0,
            method="L-BFGS-B",
            jac=jac,
            bounds=bounds,
            options={"maxiter": int(maxiter), "ftol": 1e-10},
        )
        obj = float(res.fun) if np.isfinite(res.fun) else float("inf")
        if obj < best_obj:
            best_obj = obj
            best_res = res

    if best_res is None:
        return FitResult(params=np.array([], dtype=float), loss=float("nan"), success=False, message="no runs", edges=edges)

    return FitResult(
        params=np.asarray(best_res.x, dtype=float),
        loss=float(best_obj),
        success=bool(best_res.success),
        message=str(best_res.message),
        edges=edges,
    )


# ----------------------------
# Diagnostics
# ----------------------------


def coverage_and_calibration_mae(
    z: np.ndarray,
    y: np.ndarray,
    yhat: np.ndarray,
    edges: np.ndarray,
    tau: float,
) -> Tuple[float, float, float]:
    """Return (coverage, macro_mae, micro_mae) using bins defined by edges."""
    z = np.asarray(z, float)
    y = np.asarray(y, float)
    yhat = np.asarray(yhat, float)
    edges = np.asarray(edges, float)

    cov = float(np.mean(y <= yhat))

    if edges.size < 2:
        return cov, float("nan"), float("nan")

    # bins: 0..B-1
    bins = np.digitize(z, edges[1:-1], right=False)
    B = edges.size - 1
    errs: List[float] = []
    counts: List[int] = []

    for b in range(B):
        m = bins == b
        nb = int(np.sum(m))
        if nb <= 0:
            continue
        cb = float(np.mean(y[m] <= yhat[m]))
        errs.append(abs(cb - tau))
        counts.append(nb)

    if not errs:
        return cov, float("nan"), float("nan")

    macro = float(np.mean(errs))
    micro = float(np.average(errs, weights=np.asarray(counts, float)))
    return cov, macro, micro


# ----------------------------
# CSV IO
# ----------------------------


def _parse_float(x: str) -> float:
    x = x.strip()
    if x == "" or x.lower() in {"nan", "none", "null"}:
        return float("nan")
    try:
        return float(x)
    except Exception:
        return float("nan")


def read_csv_z_and_tasks(
    path: str,
    tasks: Sequence[str],
    z_col: Optional[str] = None,
) -> Tuple[np.ndarray, Dict[str, np.ndarray], str]:
    """Read z and task columns from CSV. Returns (z, {task: y}, resolved_z_col)."""
    with open(path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError("CSV has no header row")

        fieldnames = [c.strip() for c in reader.fieldnames]

        resolved_z = z_col
        if resolved_z is None:
            for cand in ["z", "log10_flops", "log10_FLOPs", "log10_compute", "logC", "log10_C"]:
                if cand in fieldnames:
                    resolved_z = cand
                    break
        if resolved_z is None:
            raise ValueError("Could not infer z column; please pass --z_col")

        missing = [t for t in tasks if t not in fieldnames]
        if missing:
            raise ValueError(f"Missing task columns in CSV: {missing}")

        z_list: List[float] = []
        y_lists: Dict[str, List[float]] = {t: [] for t in tasks}

        for row in reader:
            z_val = _parse_float(row.get(resolved_z, ""))
            z_list.append(z_val)
            for t in tasks:
                y_lists[t].append(_parse_float(row.get(t, "")))

    z = np.asarray(z_list, float)
    ys = {t: np.asarray(v, float) for t, v in y_lists.items()}
    return z, ys, resolved_z


# ----------------------------
# Plotting
# ----------------------------


def save_debug_plot(out_path: str, z: np.ndarray, y: np.ndarray, yhat: np.ndarray, title: str) -> None:
    import matplotlib.pyplot as plt

    z = np.asarray(z, float)
    y = np.asarray(y, float)
    yhat = np.asarray(yhat, float)

    order = np.argsort(z)
    z = z[order]
    y = y[order]
    yhat = yhat[order]

    z_grid = np.linspace(float(np.min(z)), float(np.max(z)), 300)
    # interpolate yhat on a grid via refit prediction
    # (we'll just use a simple line here if caller passed grid, but typically caller recomputes)
    plt.figure(figsize=(7.0, 4.2))
    plt.scatter(z, y, s=12, alpha=0.35, label="data")
    plt.plot(z, yhat, linewidth=2.2, label="fit (on data order)")
    plt.ylim(-0.02, 1.02)
    plt.xlabel(r"$z=\log_{10}(\mathrm{FLOPs})$")
    plt.ylabel("task score $y$")
    plt.title(title)
    plt.legend(frameon=False)
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=200)
    plt.close()


# ----------------------------
# Main / CLI
# ----------------------------


def main() -> None:
    ap = argparse.ArgumentParser(description="Fit monotone I-spline quantile frontier with smoothed pinball loss.")
    ap.add_argument("--csv", type=str, required=True, help="Input CSV containing z and task columns.")
    ap.add_argument("--tasks", type=str, required=True, help="Comma-separated list of task columns to fit.")
    ap.add_argument("--z_col", type=str, default=None, help="Column name for z=log10(FLOPs). If omitted, auto-detect.")
    ap.add_argument("--tau", type=float, default=0.98)
    ap.add_argument("--kappa_final", type=float, default=50.0)
    ap.add_argument("--lambda_beta", type=float, default=1e-4)
    ap.add_argument("--lambda_a0", type=float, default=0.0)
    ap.add_argument("--n_knot_grid", type=int, default=8)
    ap.add_argument("--n_random", type=int, default=20)
    ap.add_argument("--maxiter", type=int, default=700)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out_dir", type=str, required=True, help="Output directory to write results.")
    ap.add_argument("--plot", action="store_true", help="Save quick scatter+fit plots to out_dir/plots.")
    args = ap.parse_args()

    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
    if not tasks:
        raise ValueError("No tasks provided")

    os.makedirs(args.out_dir, exist_ok=True)
    z_all, ys_all, resolved_z_col = read_csv_z_and_tasks(args.csv, tasks=tasks, z_col=args.z_col)

    results_rows: List[Dict[str, Any]] = []
    results_json: Dict[str, Any] = {
        "csv": args.csv,
        "z_col": resolved_z_col,
        "tau": args.tau,
        "kappa_final": args.kappa_final,
        "lambda_beta": args.lambda_beta,
        "lambda_a0": args.lambda_a0,
        "n_knot_grid": args.n_knot_grid,
        "n_random": args.n_random,
        "maxiter": args.maxiter,
        "seed": args.seed,
        "tasks": {},
    }

    plots_dir = os.path.join(args.out_dir, "plots")
    if args.plot:
        os.makedirs(plots_dir, exist_ok=True)

    for task in tasks:
        y_all = ys_all[task]
        mask = np.isfinite(z_all) & np.isfinite(y_all)
        z = z_all[mask]
        y = y_all[mask]
        n = int(z.size)

        if n < 10:
            results_rows.append(
                {"task": task, "n": n, "success": False, "loss": np.nan, "pinball": np.nan, "coverage": np.nan,
                 "cal_macro_mae": np.nan, "cal_micro_mae": np.nan, "y0": np.nan, "L": np.nan, "a0": np.nan,
                 "message": "not enough data"}
            )
            continue

        fit = fit_ispline_enhanced(
            z=z,
            y=y,
            tau=float(args.tau),
            kappa_final=float(args.kappa_final),
            lambda_beta=float(args.lambda_beta),
            n_knot_grid=int(args.n_knot_grid),
            n_random=int(args.n_random),
            seed=int(args.seed),
            maxiter=int(args.maxiter),
            lambda_a0=float(args.lambda_a0),
        )

        if fit.params.size == 0:
            results_rows.append(
                {"task": task, "n": n, "success": False, "loss": np.nan, "pinball": np.nan, "coverage": np.nan,
                 "cal_macro_mae": np.nan, "cal_micro_mae": np.nan, "y0": np.nan, "L": np.nan, "a0": np.nan,
                 "message": fit.message}
            )
            continue

        yhat = _predict_from_params(z, fit.params, fit.edges)
        pin = float(np.mean(smooth_pinball_loss(y - yhat, tau=float(args.tau), kappa=float(args.kappa_final))))
        cov, cal_macro, cal_micro = coverage_and_calibration_mae(z, y, yhat, fit.edges, tau=float(args.tau))

        u, v, a0 = float(fit.params[0]), float(fit.params[1]), float(fit.params[2])
        y0 = float(expit(u))
        L = float((1.0 - y0) * expit(v))
        beta = fit.params[3:].astype(float)
        w = softplus(beta)

        results_rows.append(
            {
                "task": task,
                "n": n,
                "success": bool(fit.success),
                "loss": float(fit.loss),
                "pinball": pin,
                "coverage": cov,
                "cal_macro_mae": cal_macro,
                "cal_micro_mae": cal_micro,
                "y0": y0,
                "L": L,
                "a0": a0,
                "beta_json": json.dumps(beta.tolist()),
            }
        )

        results_json["tasks"][task] = {
            "n": n,
            "success": bool(fit.success),
            "loss": float(fit.loss),
            "pinball": pin,
            "coverage": cov,
            "cal_macro_mae": cal_macro,
            "cal_micro_mae": cal_micro,
            "message": fit.message,
            "edges": fit.edges.tolist(),
            "params": {"u": u, "v": v, "a0": a0, "beta": beta.tolist()},
            "derived": {"y0": y0, "L": L, "w": w.tolist()},
        }

        if args.plot:
            # Plot fitted curve on a grid
            z_grid = np.linspace(float(np.min(z)), float(np.max(z)), 400)
            y_grid = _predict_from_params(z_grid, fit.params, fit.edges)
            import matplotlib.pyplot as plt

            order = np.argsort(z)
            plt.figure(figsize=(7.2, 4.2))
            plt.scatter(z[order], y[order], s=12, alpha=0.35, label="data")
            plt.plot(z_grid, y_grid, linewidth=2.4, label="I-spline fit")
            plt.ylim(-0.02, 1.02)
            plt.xlabel(r"$z=\log_{10}(\mathrm{FLOPs})$")
            plt.ylabel("task score $y$")
            plt.title(f"{task} (tau={args.tau})")
            plt.legend(frameon=False)
            plt.tight_layout()
            out_path = os.path.join(plots_dir, f"{task}.png")
            plt.savefig(out_path, dpi=200)
            plt.close()

    # Write outputs
    csv_path = os.path.join(args.out_dir, "results.csv")
    json_path = os.path.join(args.out_dir, "results.json")

    # CSV
    fieldnames = list(results_rows[0].keys()) if results_rows else []
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in results_rows:
            w.writerow(row)

    # JSON
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results_json, f, indent=2)

    print(f"Wrote: {csv_path}")
    print(f"Wrote: {json_path}")
    if args.plot:
        print(f"Plots: {plots_dir}")


if __name__ == "__main__":
    main()


#!/usr/bin/env python3
"""
sigmoid_quantile_enhanced_fit.py

Self-contained implementation + demo: robust fitting of a 4-parameter monotone, saturating sigmoid
to a high-quantile frontier using a smoothed pinball loss.

Why this exists:
- The smoothed pinball loss is convex in the prediction \hat{y}, but composing it with a 4-parameter sigmoid
  makes the objective non-convex in parameters.
- A single local optimizer run can converge to a poor local minimum depending on initialization.

Mitigation implemented here:
  1) Multi-start search over (z*, b) = (inflection, slope)
  2) For each (z*, b), solve a convex subproblem for (y0, L) with linear constraints
  3) Continuation over (tau, kappa) with constrained refinement from the best candidates

Outputs:
- Prints parameters and train/test metrics (pinball loss, coverage, binned coverage MAE)
- Optionally runs a random-init single-start baseline to illustrate local minima sensitivity
- Saves a diagnostic plot (scatter + fitted curves)

Dependencies: numpy, scipy, matplotlib
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Any, Dict, Tuple, List

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
import matplotlib.pyplot as plt


# ----------------------------
# Loss + model
# ----------------------------
def smooth_pinball_loss(u: np.ndarray, tau: float, kappa: float) -> np.ndarray:
    """Smoothed pinball loss, elementwise, stable implementation."""
    u = np.asarray(u, float)
    return np.logaddexp(0.0, kappa * u) / kappa + (tau - 1.0) * u


def smooth_pinball_grad_u(u: np.ndarray, tau: float, kappa: float) -> np.ndarray:
    """d/du of smooth_pinball_loss."""
    u = np.asarray(u, float)
    return expit(kappa * u) + (tau - 1.0)


def sigmoid_pred(params: np.ndarray, z: np.ndarray) -> np.ndarray:
    """
    4-parameter sigmoid frontier under the parameterization:
      yhat = y0 + L * sigmoid(b*(z - z_star))
    where params = [y0, L, z_star, log_b] and b = exp(log_b).
    """
    y0, L, z_star, log_b = [float(x) for x in params]
    b = float(np.exp(log_b))
    s = expit(b * (z - z_star))
    return y0 + L * s


def objective_and_grad(
    params: np.ndarray,
    z: np.ndarray,
    y: np.ndarray,
    tau: float,
    kappa: float,
    lambda_b: float = 0.0,
    weights: np.ndarray | None = None,
) -> Tuple[float, np.ndarray]:
    """
    Mean smooth pinball loss + ridge penalty on b^2, along with analytic gradient.
    params = [y0, L, z_star, log_b]
    """
    y0, L, z_star, log_b = params
    b = np.exp(log_b)

    t = b * (z - z_star)
    s = expit(t)
    yhat = y0 + L * s
    u = y - yhat

    if weights is None:
        obj = float(np.mean(smooth_pinball_loss(u, tau=tau, kappa=kappa)) + lambda_b * (b ** 2))
    else:
        w = np.asarray(weights, float)
        if w.shape != u.shape:
            raise ValueError(f"weights must have shape {u.shape}, got {w.shape}")
        w_sum = float(np.sum(w))
        if not np.isfinite(w_sum) or w_sum <= 0.0:
            raise ValueError("weights must sum to a positive finite value")
        obj = float(np.sum(w * smooth_pinball_loss(u, tau=tau, kappa=kappa)) / w_sum + lambda_b * (b ** 2))

    r = smooth_pinball_grad_u(u, tau=tau, kappa=kappa)  # dl/du
    sp = s * (1.0 - s)

    # d obj / d params = mean(-r * dyhat/dparam) + penalty
    if weights is None:
        g_y0 = -np.mean(r)
        g_L = -np.mean(r * s)
        g_zstar = np.mean(r * L * b * sp)
        g_logb = -np.mean(r * L * b * sp * (z - z_star))
    else:
        w = np.asarray(weights, float)
        w_sum = float(np.sum(w))
        g_y0 = -np.sum(w * r) / w_sum
        g_L = -np.sum(w * r * s) / w_sum
        g_zstar = np.sum(w * r * L * b * sp) / w_sum
        g_logb = -np.sum(w * r * L * b * sp * (z - z_star)) / w_sum

    if lambda_b != 0.0:
        # d(b^2)/dlogb = 2 b^2
        g_logb += 2.0 * lambda_b * (b ** 2)

    grad = np.array([g_y0, g_L, g_zstar, g_logb], dtype=float)
    return obj, grad


# ----------------------------
# Convex subproblem: (y0, L) given (z*, b)
# ----------------------------
def solve_y0_L_given_shape(
    z: np.ndarray,
    y: np.ndarray,
    s: np.ndarray,
    tau: float,
    kappa: float,
    weights: np.ndarray | None = None,
) -> Tuple[np.ndarray, float, bool]:
    """
    Given fixed sigmoid shape s_i = sigmoid(b*(z_i - z_star)),
    solve for (y0, L) under linear constraints:
      0 <= y0 <= 1
      0 <= L <= 1
      y0 + L <= 1
    Objective is convex in (y0, L) because it's convex in predictions and predictions are affine in (y0, L).
    """

    def fun(v):
        y0, L = v
        yhat = y0 + L * s
        u = y - yhat
        if weights is None:
            return float(np.mean(smooth_pinball_loss(u, tau=tau, kappa=kappa)))
        w = np.asarray(weights, float)
        if w.shape != u.shape:
            raise ValueError(f"weights must have shape {u.shape}, got {w.shape}")
        w_sum = float(np.sum(w))
        if not np.isfinite(w_sum) or w_sum <= 0.0:
            raise ValueError("weights must sum to a positive finite value")
        return float(np.sum(w * smooth_pinball_loss(u, tau=tau, kappa=kappa)) / w_sum)

    def jac(v):
        y0, L = v
        yhat = y0 + L * s
        u = y - yhat
        r = smooth_pinball_grad_u(u, tau=tau, kappa=kappa)
        if weights is None:
            g_y0 = -np.mean(r)
            g_L = -np.mean(r * s)
        else:
            w = np.asarray(weights, float)
            w_sum = float(np.sum(w))
            g_y0 = -np.sum(w * r) / w_sum
            g_L = -np.sum(w * r * s) / w_sum
        return np.array([g_y0, g_L], dtype=float)

    cons = [
        {
            "type": "ineq",
            "fun": lambda v: 1.0 - v[0] - v[1],
            "jac": lambda v: np.array([-1.0, -1.0]),
        }
    ]
    bounds = [(0.0, 1.0), (0.0, 1.0)]

    # conservative init
    y0_init = float(np.clip(np.quantile(y, 0.1), 0.0, 1.0))
    L_init = float(np.clip(np.quantile(y, tau) - y0_init, 0.0, 1.0 - y0_init))
    v0 = np.array([y0_init, L_init], dtype=float)

    res = minimize(
        fun,
        v0,
        method="SLSQP",
        jac=jac,
        bounds=bounds,
        constraints=cons,
        options={"maxiter": 250, "ftol": 1e-9, "disp": False},
    )

    v_hat = res.x
    ok = bool(res.success) and np.all(np.isfinite(v_hat))
    # project to feasible set
    y0_hat = float(np.clip(v_hat[0], 0.0, 1.0))
    L_hat = float(np.clip(v_hat[1], 0.0, 1.0 - y0_hat))
    v_hat = np.array([y0_hat, L_hat], dtype=float)
    obj = float(fun(v_hat))
    return v_hat, obj, ok


# ----------------------------
# Enhanced fitter: multi-start + continuation
# ----------------------------
@dataclass
class FitResult:
    params: np.ndarray
    success: bool
    objective: float
    info: Dict[str, Any]


def fit_sigmoid_enhanced(
    z: np.ndarray,
    y: np.ndarray,
    weights: np.ndarray | None = None,
    tau: float = 0.98,
    kappa_final: float = 50.0,
    lambda_b: float = 1e-2,
    n_zstar_grid: int = 9,
    n_b_grid: int = 9,
    n_random: int = 12,
    seed: int = 0,
    b_min: float = 0.05,
    b_max: float = 50.0,
    maxiter: int = 700,
) -> FitResult:
    rng = np.random.default_rng(seed)
    z = np.asarray(z, float)
    y = np.asarray(y, float)
    if weights is not None:
        w = np.asarray(weights, float)
        if w.shape != z.shape:
            raise ValueError(f"weights must have shape {z.shape}, got {w.shape}")
        w_sum = float(np.sum(w))
        if not np.isfinite(w_sum) or w_sum <= 0.0:
            raise ValueError("weights must sum to a positive finite value")

    z_min, z_max = float(np.min(z)), float(np.max(z))
    if not np.isfinite(z_min) or not np.isfinite(z_max) or z_max <= z_min:
        return FitResult(np.full(4, np.nan), False, np.inf, {"error": "degenerate z"})

    # Continuation schedules (increasing difficulty).
    #
    # NOTE: We avoid *decreasing* kappa when increasing tau, because that can
    # pull the optimizer into a high-τ / low-κ local minimum (near-constant
    # upper envelope) and prevent recovery at larger κ. We therefore:
    #   - run the full κ schedule only at the first τ stage (typically τ≈0.90),
    #   - then, for subsequent τ stages, run only at the largest κ (κ_final).
    tau_sched = [min(0.90, tau), min(0.95, tau), tau]
    tau_sched = [t for i, t in enumerate(tau_sched) if i == 0 or t > tau_sched[i - 1] + 1e-12]
    kappa_sched = [10.0, 25.0, float(kappa_final)]
    kappa_sched = [k for i, k in enumerate(kappa_sched) if i == 0 or k > kappa_sched[i - 1] + 1e-12]

    # candidate grids
    zstar_grid = np.linspace(z_min, z_max, n_zstar_grid)
    zstar_grid = np.unique(np.concatenate([zstar_grid, np.quantile(z, [0.15, 0.35, 0.5, 0.65, 0.85])]))
    b_grid = np.exp(np.linspace(np.log(b_min), np.log(b_max), n_b_grid))

    # random candidates
    zstar_rand = rng.uniform(z_min, z_max, size=n_random)
    b_rand = np.exp(rng.uniform(np.log(b_min), np.log(b_max), size=n_random))

    # stage-0 selection with smoother objective
    tau0, kappa0 = tau_sched[0], kappa_sched[0]
    candidates: List[Tuple[float, bool, np.ndarray]] = []

    def add_candidate(z_star: float, b: float):
        s = expit(b * (z - z_star))
        (y0L, _, ok) = solve_y0_L_given_shape(z, y, s, tau=tau0, kappa=kappa0, weights=weights)
        y0, L = float(y0L[0]), float(y0L[1])
        p = np.array([y0, L, float(z_star), float(np.log(b))], dtype=float)
        obj, _ = objective_and_grad(p, z, y, tau=tau0, kappa=kappa0, lambda_b=lambda_b, weights=weights)
        candidates.append((float(obj), bool(ok), p))

    for z_star in zstar_grid:
        for b in b_grid:
            add_candidate(float(z_star), float(b))
    for z_star, b in zip(zstar_rand, b_rand):
        add_candidate(float(z_star), float(b))

    candidates.sort(key=lambda t: t[0])
    top = candidates[: min(10, len(candidates))]

    # full constrained refinement
    cons = [
        {
            "type": "ineq",
            "fun": lambda p: 1.0 - p[0] - p[1],
            "jac": lambda p: np.array([-1.0, -1.0, 0.0, 0.0]),
        }
    ]
    bounds = [(0.0, 1.0), (0.0, 1.0), (z_min, z_max), (np.log(b_min), np.log(b_max))]

    def run_full(p0: np.ndarray, tau_stage: float, kappa_stage: float):
        def fun(p):
            obj, _ = objective_and_grad(p, z, y, tau=tau_stage, kappa=kappa_stage, lambda_b=lambda_b, weights=weights)
            return obj

        def jac(p):
            _, g = objective_and_grad(p, z, y, tau=tau_stage, kappa=kappa_stage, lambda_b=lambda_b, weights=weights)
            return g

        res = minimize(
            fun,
            p0,
            method="SLSQP",
            jac=jac,
            bounds=bounds,
            constraints=cons,
            options={"maxiter": maxiter, "ftol": 1e-9, "disp": False},
        )

        p = np.array(res.x, dtype=float)
        # light projection (guards against tiny constraint violations)
        y0 = float(np.clip(p[0], 0.0, 1.0))
        L = float(np.clip(p[1], 0.0, 1.0 - y0))
        z_star = float(np.clip(p[2], z_min, z_max))
        log_b = float(np.clip(p[3], np.log(b_min), np.log(b_max)))
        p = np.array([y0, L, z_star, log_b], dtype=float)

        obj, _ = objective_and_grad(p, z, y, tau=tau_stage, kappa=kappa_stage, lambda_b=lambda_b, weights=weights)
        return p, float(obj), bool(res.success), str(res.message)

    best_obj = np.inf
    best_p = None
    best_info: Dict[str, Any] = {}

    for (obj0, ok0, p0) in top:
        p = p0.copy()
        trace = []
        for tau_i, tau_stage in enumerate(tau_sched):
            kappa_stages = kappa_sched if tau_i == 0 else [kappa_sched[-1]]
            for kappa_stage in kappa_stages:
                p, obj, ok, msg = run_full(p, tau_stage, kappa_stage)
                trace.append({"tau": tau_stage, "kappa": kappa_stage, "obj": obj, "ok": ok, "msg": msg})
        final_obj, _ = objective_and_grad(p, z, y, tau=tau, kappa=kappa_final, lambda_b=lambda_b, weights=weights)
        if final_obj < best_obj:
            best_obj = float(final_obj)
            best_p = p.copy()
            best_info = {"stage0_obj": float(obj0), "stage0_ok": bool(ok0), "trace": trace}

    if best_p is None:
        return FitResult(np.full(4, np.nan), False, np.inf, {"error": "no candidates"})
    return FitResult(best_p, True, best_obj, best_info)


# ----------------------------
# Baselines: naive init / random init single start
# ----------------------------
def fit_sigmoid_naive(
    z: np.ndarray,
    y: np.ndarray,
    tau: float = 0.98,
    kappa: float = 50.0,
    lambda_b: float = 1e-2,
    b_min: float = 0.05,
    b_max: float = 50.0,
    maxiter: int = 700,
) -> FitResult:
    z = np.asarray(z, float)
    y = np.asarray(y, float)
    z_min, z_max = float(np.min(z)), float(np.max(z))
    z_rng = max(1e-6, z_max - z_min)

    # heuristic init
    y0_init = float(np.clip(np.quantile(y[z <= np.quantile(z, 0.2)], 0.2), 0.0, 1.0))
    yhi_init = float(np.clip(np.quantile(y[z >= np.quantile(z, 0.8)], tau), 0.0, 1.0))
    L_init = float(np.clip(yhi_init - y0_init, 1e-3, 1.0 - y0_init))
    z_star_init = float(np.median(z))
    b_init = float(np.clip(4.0 / z_rng, b_min, b_max))
    p0 = np.array([y0_init, L_init, z_star_init, np.log(b_init)], dtype=float)

    cons = [
        {
            "type": "ineq",
            "fun": lambda p: 1.0 - p[0] - p[1],
            "jac": lambda p: np.array([-1.0, -1.0, 0.0, 0.0]),
        }
    ]
    bounds = [(0.0, 1.0), (0.0, 1.0), (z_min, z_max), (np.log(b_min), np.log(b_max))]

    def fun(p):
        obj, _ = objective_and_grad(p, z, y, tau=tau, kappa=kappa, lambda_b=lambda_b)
        return obj

    def jac(p):
        _, g = objective_and_grad(p, z, y, tau=tau, kappa=kappa, lambda_b=lambda_b)
        return g

    res = minimize(
        fun,
        p0,
        method="SLSQP",
        jac=jac,
        bounds=bounds,
        constraints=cons,
        options={"maxiter": maxiter, "ftol": 1e-9, "disp": False},
    )

    p = np.array(res.x, dtype=float)
    y0 = float(np.clip(p[0], 0.0, 1.0))
    L = float(np.clip(p[1], 0.0, 1.0 - y0))
    z_star = float(np.clip(p[2], z_min, z_max))
    log_b = float(np.clip(p[3], np.log(b_min), np.log(b_max)))
    p = np.array([y0, L, z_star, log_b], dtype=float)

    obj, _ = objective_and_grad(p, z, y, tau=tau, kappa=kappa, lambda_b=lambda_b)
    return FitResult(p, bool(res.success), float(obj), {"message": str(res.message), "nit": int(getattr(res, "nit", -1))})


def fit_sigmoid_random_start(
    z: np.ndarray,
    y: np.ndarray,
    seed: int,
    tau: float = 0.98,
    kappa: float = 50.0,
    lambda_b: float = 1e-2,
    b_min: float = 0.05,
    b_max: float = 50.0,
    maxiter: int = 700,
) -> FitResult:
    rng = np.random.default_rng(seed)
    z = np.asarray(z, float)
    y = np.asarray(y, float)
    z_min, z_max = float(np.min(z)), float(np.max(z))

    y0 = rng.uniform(0.0, 0.8)
    L = rng.uniform(0.0, 1.0 - y0)
    z_star = rng.uniform(z_min, z_max)
    log_b = rng.uniform(np.log(b_min), np.log(b_max))
    p0 = np.array([y0, L, z_star, log_b], dtype=float)

    cons = [
        {
            "type": "ineq",
            "fun": lambda p: 1.0 - p[0] - p[1],
            "jac": lambda p: np.array([-1.0, -1.0, 0.0, 0.0]),
        }
    ]
    bounds = [(0.0, 1.0), (0.0, 1.0), (z_min, z_max), (np.log(b_min), np.log(b_max))]

    def fun(p):
        obj, _ = objective_and_grad(p, z, y, tau=tau, kappa=kappa, lambda_b=lambda_b)
        return obj

    def jac(p):
        _, g = objective_and_grad(p, z, y, tau=tau, kappa=kappa, lambda_b=lambda_b)
        return g

    res = minimize(
        fun,
        p0,
        method="SLSQP",
        jac=jac,
        bounds=bounds,
        constraints=cons,
        options={"maxiter": maxiter, "ftol": 1e-9, "disp": False},
    )

    p = np.array(res.x, dtype=float)
    y0 = float(np.clip(p[0], 0.0, 1.0))
    L = float(np.clip(p[1], 0.0, 1.0 - y0))
    z_star = float(np.clip(p[2], z_min, z_max))
    log_b = float(np.clip(p[3], np.log(b_min), np.log(b_max)))
    p = np.array([y0, L, z_star, log_b], dtype=float)

    obj, _ = objective_and_grad(p, z, y, tau=tau, kappa=kappa, lambda_b=lambda_b)
    return FitResult(p, bool(res.success), float(obj), {"message": str(res.message), "nit": int(getattr(res, "nit", -1))})


# ----------------------------
# Coverage utilities
# ----------------------------
def equal_mass_bins(z: np.ndarray, B: int, min_bin: int = 25) -> np.ndarray:
    z = np.asarray(z, float)
    n = z.size
    if n == 0:
        return np.array([])
    B_eff = min(B, max(1, n // min_bin))
    qs = np.linspace(0, 1, B_eff + 1)
    edges = np.quantile(z, qs)
    edges = np.unique(edges)
    if edges.size < 2:
        edges = np.array([float(np.min(z)), float(np.max(z))])
    return edges


def bin_indices(z: np.ndarray, edges: np.ndarray) -> np.ndarray:
    return np.searchsorted(edges[1:-1], z, side="right")


def calibration_by_bins(z: np.ndarray, y: np.ndarray, yhat: np.ndarray, tau: float, edges: np.ndarray) -> Dict[str, Any]:
    idx = bin_indices(z, edges)
    B = edges.size - 1
    counts = np.zeros(B, dtype=int)
    cov = np.full(B, np.nan, dtype=float)
    err_abs = np.full(B, np.nan, dtype=float)

    for b in range(B):
        m = idx == b
        nb = int(np.sum(m))
        counts[b] = nb
        if nb == 0:
            continue
        cov_b = float(np.mean(y[m] <= yhat[m]))
        err_abs[b] = abs(cov_b - tau)

    nonempty = counts > 0
    macro_mae = float(np.mean(err_abs[nonempty])) if np.any(nonempty) else float("nan")
    micro_mae = float(np.sum(err_abs[nonempty] * counts[nonempty]) / np.sum(counts[nonempty])) if np.any(nonempty) else float("nan")
    return {"macro_mae": macro_mae, "micro_mae": micro_mae}


def eval_metrics(z: np.ndarray, y: np.ndarray, params: np.ndarray, tau: float, kappa: float, edges: np.ndarray) -> Dict[str, float]:
    yhat = sigmoid_pred(params, z)
    u = y - yhat
    pin = float(np.mean(smooth_pinball_loss(u, tau=tau, kappa=kappa)))
    cov = float(np.mean(y <= yhat))
    cal = calibration_by_bins(z, y, yhat, tau, edges)
    return {"pinball": pin, "coverage": cov, "cal_macro_mae": float(cal["macro_mae"]), "cal_micro_mae": float(cal["micro_mae"])}


# ----------------------------
# Synthetic data generator (imperfect sigmoid with outliers)
# ----------------------------
def generate_synthetic(n: int = 600, seed: int = 0) -> Dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    z_min, z_max = 1.0, 5.0

    # more mass at high compute
    mix = rng.uniform(size=n)
    z = np.where(mix < 0.6, rng.uniform(z_min, z_max, size=n), rng.normal(loc=4.0, scale=0.4, size=n))
    z = np.clip(z, z_min, z_max)

    # generator sigmoid (not necessarily the tau-quantile under noise)
    y0_true, L_true, z_star_true, b_true = 0.12, 0.78, 3.2, 2.0
    y_base = y0_true + L_true * expit(b_true * (z - z_star_true))

    # heteroscedastic noise + heavy tails + outliers
    sigma = 0.03 + 0.06 * (1.0 - expit(1.2 * (z - 3.0)))
    eps = rng.normal(0.0, sigma)

    shock = rng.uniform(size=n) < 0.05
    eps[shock] += rng.standard_t(df=2, size=np.sum(shock)) * 0.10

    pos = rng.uniform(size=n) < 0.03
    eps[pos] += rng.uniform(0.08, 0.25, size=np.sum(pos))

    neg = rng.uniform(size=n) < 0.02
    eps[neg] -= rng.uniform(0.08, 0.20, size=np.sum(neg))

    y = np.clip(y_base + eps, 0.0, 1.0)

    return {
        "z": z,
        "y": y,
        "y_base": y_base,
        "params_true": np.array([y0_true, L_true, z_star_true, np.log(b_true)], dtype=float),
    }


# ----------------------------
# Main
# ----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=600)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tau", type=float, default=0.98)
    ap.add_argument("--kappa", type=float, default=50.0)
    ap.add_argument("--lambda_b", type=float, default=1e-2)
    ap.add_argument("--plot_path", type=str, default="sigmoid_quantile_fit_demo.png")
    ap.add_argument("--random_baseline_restarts", type=int, default=0,
                    help="If >0, run this many random-init single-start fits and report best/worst objective.")
    args = ap.parse_args()

    data = generate_synthetic(n=args.n, seed=args.seed)
    z, y = data["z"], data["y"]
    true_params = data["params_true"]

    rng = np.random.default_rng(42)
    perm = rng.permutation(z.size)
    tr = perm[: int(0.7 * z.size)]
    te = perm[int(0.7 * z.size) :]

    z_tr, y_tr = z[tr], y[tr]
    z_te, y_te = z[te], y[te]

    naive = fit_sigmoid_naive(z_tr, y_tr, tau=args.tau, kappa=args.kappa, lambda_b=args.lambda_b)
    enh = fit_sigmoid_enhanced(z_tr, y_tr, tau=args.tau, kappa_final=args.kappa, lambda_b=args.lambda_b)

    edges = equal_mass_bins(z_tr, B=6, min_bin=30)

    m_naive_tr = eval_metrics(z_tr, y_tr, naive.params, args.tau, args.kappa, edges)
    m_naive_te = eval_metrics(z_te, y_te, naive.params, args.tau, args.kappa, edges)
    m_enh_tr = eval_metrics(z_tr, y_tr, enh.params, args.tau, args.kappa, edges)
    m_enh_te = eval_metrics(z_te, y_te, enh.params, args.tau, args.kappa, edges)

    def fmt_params(p):
        y0, L, z_star, log_b = p
        return f"y0={y0:.3f}, L={L:.3f}, z*={z_star:.3f}, b={np.exp(log_b):.3f}"

    print("\n=== Fit parameters ===")
    print(f"True (generator):      {fmt_params(true_params)}")
    print(f"Naive (single start):  {fmt_params(naive.params)}  success={naive.success}")
    print(f"Enhanced (robust):     {fmt_params(enh.params)}    success={enh.success}")

    print("\n=== Train metrics (tau, kappa) ===")
    print(f"Naive:    pinball={m_naive_tr['pinball']:.6f}  coverage={m_naive_tr['coverage']:.4f}  calMAE={m_naive_tr['cal_macro_mae']:.4f}")
    print(f"Enhanced: pinball={m_enh_tr['pinball']:.6f}  coverage={m_enh_tr['coverage']:.4f}  calMAE={m_enh_tr['cal_macro_mae']:.4f}")

    print("\n=== Test metrics (OOS) ===")
    print(f"Naive:    pinball={m_naive_te['pinball']:.6f}  coverage={m_naive_te['coverage']:.4f}  calMAE={m_naive_te['cal_macro_mae']:.4f}")
    print(f"Enhanced: pinball={m_enh_te['pinball']:.6f}  coverage={m_enh_te['coverage']:.4f}  calMAE={m_enh_te['cal_macro_mae']:.4f}")

    if args.random_baseline_restarts > 0:
        objs = []
        for s in range(args.random_baseline_restarts):
            r = fit_sigmoid_random_start(z_tr, y_tr, seed=s, tau=args.tau, kappa=args.kappa, lambda_b=args.lambda_b)
            objs.append(r.objective)
        print("\n=== Random-init single-start baseline (illustrating nonconvexity) ===")
        print(f"R={args.random_baseline_restarts} restarts on train objective:")
        print(f"  best={np.min(objs):.6f}  median={np.median(objs):.6f}  worst={np.max(objs):.6f}")

    # Plot: data + true generator curve + fits
    z_grid = np.linspace(np.min(z), np.max(z), 400)
    true_curve = sigmoid_pred(true_params, z_grid)
    naive_curve = sigmoid_pred(naive.params, z_grid)
    enh_curve = sigmoid_pred(enh.params, z_grid)

    plt.figure(figsize=(8.6, 5.1))
    plt.scatter(z_tr, y_tr, s=12, alpha=0.45, label="train", rasterized=True)
    plt.scatter(z_te, y_te, s=12, alpha=0.45, label="test", rasterized=True)
    plt.plot(z_grid, true_curve, linewidth=2.5, label="generator sigmoid")
    plt.plot(z_grid, naive_curve, linewidth=2.5, label="naive fit")
    plt.plot(z_grid, enh_curve, linewidth=2.5, label="enhanced fit")

    plt.ylim(-0.02, 1.02)
    plt.xlabel(r"$z=\log_{10}(\mathrm{FLOPs})$")
    plt.ylabel("task score $y$")
    plt.title(rf"High-quantile sigmoid fit (tau={args.tau}, kappa={args.kappa})")
    plt.legend(frameon=False, ncols=2)
    plt.tight_layout()
    plt.savefig(args.plot_path, dpi=200)
    print(f"\nSaved plot to: {args.plot_path}")


if __name__ == "__main__":
    main()

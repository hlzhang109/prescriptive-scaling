"""Core enhanced sigmoid quantile optimizer utilities.

This is the reusable subset of `scripts/run/sigmoid_quantile_optimizer.py`
used by `skill_frontier.core.sigmoid.fit_sigmoid_frontier`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit


def smooth_pinball_loss(u: np.ndarray, tau: float, kappa: float) -> np.ndarray:
    """Smoothed pinball loss, elementwise, stable implementation."""
    u = np.asarray(u, float)
    return np.logaddexp(0.0, kappa * u) / kappa + (tau - 1.0) * u


def smooth_pinball_grad_u(u: np.ndarray, tau: float, kappa: float) -> np.ndarray:
    """d/du of smooth_pinball_loss."""
    u = np.asarray(u, float)
    return expit(kappa * u) + (tau - 1.0)


def sigmoid_pred(params: np.ndarray, z: np.ndarray) -> np.ndarray:
    """4-parameter sigmoid frontier prediction."""
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
    """Mean smooth pinball loss + ridge penalty on b^2, with analytic gradient."""
    y0, L, z_star, log_b = params
    b = np.exp(log_b)

    t = b * (z - z_star)
    s = expit(t)
    yhat = y0 + L * s
    u = y - yhat

    if weights is None:
        obj = float(np.mean(smooth_pinball_loss(u, tau=tau, kappa=kappa)) + lambda_b * (b**2))
    else:
        w = np.asarray(weights, float)
        if w.shape != u.shape:
            raise ValueError(f"weights must have shape {u.shape}, got {w.shape}")
        w_sum = float(np.sum(w))
        if not np.isfinite(w_sum) or w_sum <= 0.0:
            raise ValueError("weights must sum to a positive finite value")
        obj = float(np.sum(w * smooth_pinball_loss(u, tau=tau, kappa=kappa)) / w_sum + lambda_b * (b**2))

    r = smooth_pinball_grad_u(u, tau=tau, kappa=kappa)
    sp = s * (1.0 - s)

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
        g_logb += 2.0 * lambda_b * (b**2)

    grad = np.array([g_y0, g_L, g_zstar, g_logb], dtype=float)
    return obj, grad


def solve_y0_L_given_shape(
    z: np.ndarray,
    y: np.ndarray,
    s: np.ndarray,
    tau: float,
    kappa: float,
    weights: np.ndarray | None = None,
) -> Tuple[np.ndarray, float, bool]:
    """Solve convex subproblem for (y0, L) given fixed sigmoid shape."""

    def fun(v: np.ndarray) -> float:
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

    def jac(v: np.ndarray) -> np.ndarray:
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
    y0_hat = float(np.clip(v_hat[0], 0.0, 1.0))
    L_hat = float(np.clip(v_hat[1], 0.0, 1.0 - y0_hat))
    v_hat = np.array([y0_hat, L_hat], dtype=float)
    obj = float(fun(v_hat))
    return v_hat, obj, ok


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
    """Robust multi-start + continuation enhanced sigmoid fit."""
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

    tau_sched = [min(0.90, tau), min(0.95, tau), tau]
    tau_sched = [t for i, t in enumerate(tau_sched) if i == 0 or t > tau_sched[i - 1] + 1e-12]
    kappa_sched = [10.0, 25.0, float(kappa_final)]
    kappa_sched = [k for i, k in enumerate(kappa_sched) if i == 0 or k > kappa_sched[i - 1] + 1e-12]

    zstar_grid = np.linspace(z_min, z_max, n_zstar_grid)
    zstar_grid = np.unique(np.concatenate([zstar_grid, np.quantile(z, [0.15, 0.35, 0.5, 0.65, 0.85])]))
    b_grid = np.exp(np.linspace(np.log(b_min), np.log(b_max), n_b_grid))

    zstar_rand = rng.uniform(z_min, z_max, size=n_random)
    b_rand = np.exp(rng.uniform(np.log(b_min), np.log(b_max), size=n_random))

    tau0, kappa0 = tau_sched[0], kappa_sched[0]
    candidates: List[Tuple[float, bool, np.ndarray]] = []

    def add_candidate(z_star: float, b: float) -> None:
        s = expit(b * (z - z_star))
        y0L, _, ok = solve_y0_L_given_shape(z, y, s, tau=tau0, kappa=kappa0, weights=weights)
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

    cons = [
        {
            "type": "ineq",
            "fun": lambda p: 1.0 - p[0] - p[1],
            "jac": lambda p: np.array([-1.0, -1.0, 0.0, 0.0]),
        }
    ]
    bounds = [(0.0, 1.0), (0.0, 1.0), (z_min, z_max), (np.log(b_min), np.log(b_max))]

    def run_full(p0: np.ndarray, tau_stage: float, kappa_stage: float) -> Tuple[np.ndarray, float, bool, str]:
        def fun(p: np.ndarray) -> float:
            obj, _ = objective_and_grad(p, z, y, tau=tau_stage, kappa=kappa_stage, lambda_b=lambda_b, weights=weights)
            return obj

        def jac(p: np.ndarray) -> np.ndarray:
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

    for obj0, ok0, p0 in top:
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


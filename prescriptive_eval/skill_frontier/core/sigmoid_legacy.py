"""Legacy sigmoid frontier fitter used for reproducible paper artifacts.

This module preserves the pre-refactor fitting routine so scripts that require
exact historical outputs can opt into it explicitly.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np


def _sigmoid(u: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-u))


def _softplus(u: np.ndarray | float) -> np.ndarray | float:
    return np.log1p(np.exp(u))


def fit_sigmoid_frontier_legacy(
    x: np.ndarray,
    y: np.ndarray,
    tau: float = 0.98,
    use_log10_x: bool = True,
    grid_points: int = 400,
) -> Tuple[np.ndarray, np.ndarray]:
    """Fit the legacy monotone logistic frontier (pre-enhanced optimizer)."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    if x.size < 3:
        return np.array([]), np.array([])
    z = np.log10(np.maximum(x, 1e-300)) if use_log10_x else x.copy()
    order = np.argsort(z)
    z = z[order]
    y = y[order]

    y0_init = float(np.nanmax([0.0, np.nanpercentile(y, 5)]))
    ymax = float(np.nanmin([1.0, np.nanpercentile(y, 99)]))
    L_init = max(1e-3, ymax - y0_init)
    z_med = float(np.nanmedian(z))
    y_med = float(np.nanmedian(y))
    frac = np.clip((y_med - y0_init) / max(L_init, 1e-6), 1e-6, 1 - 1e-6)
    a_init = float(np.log(frac / (1 - frac)))
    b_init = 1.0

    def pack(theta_raw: np.ndarray) -> Tuple[float, float, float, float]:
        g0, gL, a, bb = theta_raw
        y0 = float(_sigmoid(np.array([g0]))[0])
        L_cap = max(1e-6, 1.0 - y0 - 1e-6)
        L = float(_sigmoid(np.array([gL]))[0]) * L_cap
        b = float(_softplus(bb))
        return y0, L, a, b

    def loss(theta_raw: np.ndarray) -> float:
        y0, L, a, b = pack(theta_raw)
        yh = y0 + L * _sigmoid(a + b * z)
        r = y - yh
        k = 50.0
        rho = (1.0 / k) * np.log1p(np.exp(k * r)) + (tau - 1.0) * r
        return float(np.sum(rho) + 1e-3 * (a * a + b * b))

    def _logit(p: float) -> float:
        p = float(np.clip(p, 1e-6, 1 - 1e-6))
        return float(np.log(p / (1 - p)))

    theta0 = np.array(
        [
            _logit(min(0.99, max(0.01, y0_init))),
            _logit(min(0.99, max(0.01, L_init / max(1e-6, 1 - y0_init - 1e-6)))),
            a_init - b_init * z_med,
            np.log(np.expm1(b_init) + 1e-6),
        ]
    )

    best_theta = theta0.copy()
    best_val = loss(theta0)

    try:
        from scipy.optimize import minimize  # type: ignore

        res = minimize(loss, theta0, method="L-BFGS-B", options={"maxiter": 10000})
        if res.success and np.isfinite(res.fun):
            best_theta = res.x
            best_val = float(res.fun)
    except Exception:
        rng = np.random.default_rng(0)
        for _ in range(32):
            cand = theta0 + rng.normal(scale=[0.5, 0.5, 1.0, 0.5])
            val = loss(cand)
            if val < best_val:
                best_val = val
                best_theta = cand

    y0, L, a, b = pack(best_theta)
    z_grid = np.linspace(float(z.min()), float(z.max()), num=grid_points)
    y_hat = y0 + L * _sigmoid(a + b * z_grid)
    y_hat = np.clip(y_hat, 0.0, 1.0)
    xs_sample = (10.0 ** z_grid) if use_log10_x else z_grid
    return xs_sample.astype(float), y_hat.astype(float)

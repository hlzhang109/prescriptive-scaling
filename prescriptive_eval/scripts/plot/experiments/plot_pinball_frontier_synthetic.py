#!/usr/bin/env python3
"""
Synthetic Frontier Comparison: Squared Loss vs. Pinball Loss
------------------------------------------------------------

Generates a toy 1D scaling-law dataset where the true frontier is a sigmoid
in log-compute, then fits:
  - an OLS frontier using squared loss, and
  - a high-quantile frontier using the same smoothed pinball loss as the
    main codebase (via skill_frontier.core.sigmoid.fit_sigmoid_frontier).

Outputs:
  outputs/pinball_viz/synthetic_frontier_comparison.(png|pdf)
"""

from __future__ import annotations

import os
import sys
from typing import Tuple

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

# Ensure repo root on sys.path so skill_frontier imports work when run as a script
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, os.pardir))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from skill_frontier.core.sigmoid import _sigmoid, _softplus, fit_sigmoid_frontier  # type: ignore


def _ensure_tex_serif() -> None:
    try:
        mpl.rcParams["font.family"] = "serif"
        mpl.rcParams["text.usetex"] = True
        mpl.rcParams["text.latex.preamble"] = r"\usepackage{amsmath}"
    except Exception:
        pass


def logistic_frontier(z: np.ndarray, theta: Tuple[float, float, float, float]) -> np.ndarray:
    y0, L, a, b = theta
    return y0 + L * _sigmoid(a + b * z)


def sample_synthetic_data(
    n: int = 200,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, Tuple[float, float, float, float]]:
    rng = np.random.default_rng(seed)
    # True frontier parameters (y0, L, a, b)
    theta_true = (0.25, 0.6, -4.0, 1.0)
    z = np.linspace(0.0, 6.0, n)
    y_frontier = logistic_frontier(z, theta_true)
    # Mostly below-frontier noise with much larger variance than above-frontier bumps
    # (heavy-tailed downward noise, small/tight upward outliers).
    eps = rng.gamma(shape=2.0, scale=0.12, size=n)  # var ≈ 2 * 0.12^2 >> var(delta)
    y = y_frontier - eps
    # A small fraction of above-frontier outliers
    n_out = max(1, int(0.05 * n))
    idx_out = rng.choice(n, size=n_out, replace=False)
    y[idx_out] = y_frontier[idx_out] + rng.uniform(0.01, 0.03, size=n_out)
    # Clip to [0,1]
    y = np.clip(y, 0.0, 1.0)
    return z, y, theta_true


def _pack_raw_to_params(theta_raw: np.ndarray) -> Tuple[float, float, float, float]:
    g0, gL, a, bb = theta_raw
    y0 = float(_sigmoid(np.array([g0]))[0])
    L_cap = max(1e-6, 1.0 - y0 - 1e-6)
    L = float(_sigmoid(np.array([gL]))[0]) * L_cap
    b = float(_softplus(bb))
    return y0, L, a, b


def _encode_initial(theta_true: Tuple[float, float, float, float]) -> np.ndarray:
    y0, L, a, b = theta_true

    def _logit(p: float) -> float:
        p = float(np.clip(p, 1e-6, 1 - 1e-6))
        return float(np.log(p / (1 - p)))

    L_cap = max(1e-6, 1.0 - y0 - 1e-6)
    theta0 = np.array(
        [
            _logit(min(0.99, max(0.01, y0))),
            _logit(min(0.99, max(0.01, L / max(L_cap, 1e-6)))),
            a,
            np.log(np.expm1(max(b, 1e-6)) + 1e-6),
        ]
    )
    return theta0


def fit_ols_frontier(
    z: np.ndarray,
    y: np.ndarray,
    theta_init: Tuple[float, float, float, float],
    l2: float = 1e-3,
) -> Tuple[float, float, float, float]:
    from scipy.optimize import minimize  # type: ignore

    theta0_raw = _encode_initial(theta_init)

    def loss(theta_raw: np.ndarray) -> float:
        y0, L, a, b = _pack_raw_to_params(theta_raw)
        yh = logistic_frontier(z, (y0, L, a, b))
        r = y - yh
        return float(np.sum(r * r) + l2 * (a * a + b * b))

    res = minimize(loss, theta0_raw, method="L-BFGS-B", options={"maxiter": 5000})
    theta_best = theta0_raw if not res.success or not np.isfinite(res.fun) else res.x
    return _pack_raw_to_params(theta_best)


def fit_pinball_frontier_from_core(
    z: np.ndarray,
    y: np.ndarray,
    tau: float = 0.98,
) -> Tuple[float, float, float, float]:
    # Core fitter takes x in original scale and applies log10 internally
    x = 10.0**z
    xs_curve, y_curve = fit_sigmoid_frontier(x, y, tau=float(tau), use_log10_x=True)
    if xs_curve.size == 0 or y_curve.size == 0:
        raise RuntimeError("Core sigmoid fitter returned empty curve on synthetic data")
    # Recover parameters by refitting a sigmoid in closed form to the curve itself
    # (cheap least squares in z-y space, same parameterization as above).
    z_curve = np.log10(xs_curve)

    from scipy.optimize import minimize  # type: ignore

    theta_init = (float(y_curve.min()), float(y_curve.max() - y_curve.min()), 0.0, 1.0)
    theta0_raw = _encode_initial(theta_init)

    def loss(theta_raw: np.ndarray) -> float:
        y0, L, a, b = _pack_raw_to_params(theta_raw)
        yh = logistic_frontier(z_curve, (y0, L, a, b))
        r = y_curve - yh
        return float(np.sum(r * r) + 1e-6 * (a * a + b * b))

    res = minimize(loss, theta0_raw, method="L-BFGS-B", options={"maxiter": 5000})
    theta_best = theta0_raw if not res.success or not np.isfinite(res.fun) else res.x
    return _pack_raw_to_params(theta_best)


def plot_synthetic_frontier_comparison(out_dir: str) -> None:
    _ensure_tex_serif()
    os.makedirs(out_dir, exist_ok=True)

    z, y, theta_true = sample_synthetic_data()

    theta_ols = fit_ols_frontier(z, y, theta_true)
    theta_pinball = fit_pinball_frontier_from_core(z, y, tau=0.98)

    z_dense = np.linspace(z.min(), z.max(), 400)
    y_true = logistic_frontier(z_dense, theta_true)
    y_ols = logistic_frontier(z_dense, theta_ols)
    y_pin = logistic_frontier(z_dense, theta_pinball)

    fig, ax = plt.subplots(figsize=(7.5, 4.5))

    ax.scatter(
        z,
        y,
        s=15,
        alpha=0.35,
        color="#1f77b4",
        edgecolors="none",
        label="Synthetic models",
    )
    ax.plot(
        z_dense,
        y_true,
        color="black",
        linewidth=2.4,
        linestyle="-",
        label="True frontier",
    )
    ax.plot(
        z_dense,
        y_ols,
        color="#d62728",
        linewidth=2.0,
        linestyle="--",
        label="OLS frontier",
    )
    ax.plot(
        z_dense,
        y_pin,
        color="#2ca02c",
        linewidth=2.0,
        linestyle="-.",
        label=rf"Pinball frontier ($\tau=0.98$)",
    )

    ax.set_xlabel(r"$z = \log_{10}(\mathrm{compute})$", fontsize=18, fontweight="bold")
    ax.set_ylabel("Accuracy", fontsize=18, fontweight="bold")
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.5)
    ax.legend(fontsize=12, loc="lower right")

    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "synthetic_frontier_comparison.png"), dpi=300, bbox_inches="tight")
    fig.savefig(os.path.join(out_dir, "synthetic_frontier_comparison.pdf"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    out_dir = os.path.join("outputs", "pinball_viz")
    plot_synthetic_frontier_comparison(out_dir)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Pinball Loss Visualizations (1D)
--------------------------------

Figure A: smoothed vs. true pinball loss as a function of residual r.
Figure B: corresponding gradients, showing the asymmetric weighting that
drives high-quantile frontier fits.

Outputs:
  outputs/pinball_viz/loss_vs_residual.(png|pdf)
  outputs/pinball_viz/gradient_vs_residual.(png|pdf)
"""

from __future__ import annotations

import os
from typing import Iterable, Sequence

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl


def _ensure_tex_serif() -> None:
    try:
        mpl.rcParams["font.family"] = "serif"
        mpl.rcParams["text.usetex"] = True
        mpl.rcParams["text.latex.preamble"] = r"\usepackage{amsmath}"
    except Exception:
        pass


def true_pinball_loss(r: np.ndarray, tau: float) -> np.ndarray:
    r = np.asarray(r, float)
    return np.where(r >= 0.0, tau * r, (tau - 1.0) * r)


def smooth_pinball_loss(r: np.ndarray, tau: float, k: float = 50.0) -> np.ndarray:
    r = np.asarray(r, float)
    return (1.0 / k) * np.log1p(np.exp(k * r)) + (tau - 1.0) * r


def smooth_pinball_grad(r: np.ndarray, tau: float, k: float = 50.0) -> np.ndarray:
    r = np.asarray(r, float)
    sig = 1.0 / (1.0 + np.exp(-k * r))
    return sig + (tau - 1.0)


def true_pinball_grad(r: np.ndarray, tau: float) -> np.ndarray:
    r = np.asarray(r, float)
    g = np.empty_like(r)
    g[r > 0.0] = tau
    g[r < 0.0] = tau - 1.0
    g[r == 0.0] = 0.5 * (tau + (tau - 1.0))
    return g


def plot_loss_vs_residual(
    r: np.ndarray,
    taus: Sequence[float],
    out_dir: str,
) -> None:
    _ensure_tex_serif()
    os.makedirs(out_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7.0, 4.2))

    colors = ["#1f77b4", "#d62728", "#2ca02c"]
    for tau, color in zip(taus, colors):
        ax.plot(
            r,
            true_pinball_loss(r, tau),
            linestyle="--",
            linewidth=1.8,
            color=color,
            label=rf"Pinball, $\tau={tau:.2f}$",
        )
        ax.plot(
            r,
            smooth_pinball_loss(r, tau),
            linestyle="-",
            linewidth=2.2,
            color=color,
            alpha=0.9,
            label=rf"Smoothed, $\tau={tau:.2f}$",
        )

    ax.plot(
        r,
        r**2,
        linestyle=":",
        linewidth=1.8,
        color="black",
        label=r"Squared loss $r^2$",
    )

    ax.set_xlabel(r"Residual $r = y - \hat{y}$", fontsize=18, fontweight="bold")
    ax.set_ylabel(r"Loss $\rho(r)$", fontsize=18, fontweight="bold")
    ax.axvline(0.0, color="#888888", linestyle=":", linewidth=1.0)
    ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.5)
    ax.legend(fontsize=12, loc="upper left")

    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "loss_vs_residual.png"), dpi=300, bbox_inches="tight")
    fig.savefig(os.path.join(out_dir, "loss_vs_residual.pdf"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_grad_vs_residual(
    r: np.ndarray,
    taus: Sequence[float],
    out_dir: str,
) -> None:
    _ensure_tex_serif()
    os.makedirs(out_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7.0, 4.2))

    colors = ["#1f77b4", "#d62728", "#2ca02c"]
    for tau, color in zip(taus, colors):
        ax.plot(
            r,
            true_pinball_grad(r, tau),
            linestyle="--",
            linewidth=1.8,
            color=color,
            label=rf"Pinball grad, $\tau={tau:.2f}$",
        )
        ax.plot(
            r,
            smooth_pinball_grad(r, tau),
            linestyle="-",
            linewidth=2.2,
            color=color,
            alpha=0.9,
            label=rf"Smoothed grad, $\tau={tau:.2f}$",
        )

    ax.set_xlabel(r"Residual $r = y - \hat{y}$", fontsize=18, fontweight="bold")
    ax.set_ylabel(r"Gradient $\partial \tilde{\rho}_\tau / \partial r$", fontsize=18, fontweight="bold")
    ax.axhline(0.0, color="#888888", linestyle=":", linewidth=1.0)
    ax.axvline(0.0, color="#888888", linestyle=":", linewidth=1.0)
    ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.5)
    ax.legend(fontsize=12, loc="lower right")

    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "gradient_vs_residual.png"), dpi=300, bbox_inches="tight")
    fig.savefig(os.path.join(out_dir, "gradient_vs_residual.pdf"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    out_dir = os.path.join("outputs", "pinball_viz")
    r = np.linspace(-0.5, 0.5, 1001)
    taus: Iterable[float] = (0.90, 0.95, 0.98)
    plot_loss_vs_residual(r, taus, out_dir)
    plot_grad_vs_residual(r, taus, out_dir)


if __name__ == "__main__":
    main()

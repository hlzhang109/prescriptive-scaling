"""Common evaluation utilities shared across evaluation scripts.

This module provides wrapper functions for fitting sigmoid frontiers and
interpolating curves, used in various evaluation workflows.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np


def fit_sigmoid_predictor(
    x_train: np.ndarray,
    y_train: np.ndarray,
    tau: float = 0.98,
    *,
    frontier_fit_mode: str = "quantile_per_point",
    bins_for_fit: int | None = None,
    min_bin_size_for_fit: int | None = None,
    bin_frontier_quantile: float = 0.98,
    bin_trim_fraction: float = 0.01,
    bin_edges_for_fit: np.ndarray | None = None,
    kappa_final: float = 50.0,
    lambda_b: float | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Fit a parametric sigmoid frontier on training data.

    This is a wrapper around the existing sigmoid fitter. Returns curve
    coordinates for interpolation.

    Args:
        x_train: Training compute values (FLOPs)
        y_train: Training accuracy values
        tau: Quantile parameter for frontier estimation

    Returns:
        Tuple of (xs_curve, y_curve) arrays for curve interpolation
    """
    # Import lazily to avoid import-order cycles in CLI scripts.
    try:
        from skill_frontier.core.sigmoid import fit_sigmoid_frontier  # type: ignore
    except Exception:
        from ..core.sigmoid import fit_sigmoid_frontier  # type: ignore

    xs_curve, y_curve = fit_sigmoid_frontier(
        x_train,
        y_train,
        tau=float(tau),
        use_log10_x=True,
        fit_mode=str(frontier_fit_mode),
        bins_for_fit=bins_for_fit,
        min_bin_size_for_fit=min_bin_size_for_fit,
        bin_frontier_quantile=float(bin_frontier_quantile),
        bin_trim_fraction=float(bin_trim_fraction),
        bin_edges_for_fit=bin_edges_for_fit,
        kappa_final=float(kappa_final),
        lambda_b=lambda_b,
    )
    return xs_curve, y_curve


def interpolate_curve(
    xs_curve: np.ndarray, y_curve: np.ndarray, x: np.ndarray
) -> np.ndarray:
    """Interpolate fitted curve at new x values.

    Args:
        xs_curve: X coordinates of fitted curve
        y_curve: Y coordinates of fitted curve
        x: New x values to interpolate at

    Returns:
        Interpolated y values at x
    """
    if xs_curve.size == 0:
        return np.zeros_like(x)
    return np.interp(x, xs_curve, y_curve, left=y_curve[0], right=y_curve[-1])

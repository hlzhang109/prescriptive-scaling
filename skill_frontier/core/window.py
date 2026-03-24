#!/usr/bin/env python3
"""
Kernel windowing utilities for skill frontier estimation.

Provides `KernelWindow.build(...)` to precompute per-grid local weights over
models in logC space using simple kernels and Silverman's rule bandwidth.

This module is dependency-light (NumPy only) and standalone.
"""

from __future__ import annotations

import dataclasses
import math
from typing import List, Optional

import numpy as np


def _gaussian_kernel(x: np.ndarray) -> np.ndarray:
    return np.exp(-0.5 * np.square(x))


def _epanechnikov_kernel(x: np.ndarray) -> np.ndarray:
    out = 1.0 - np.square(x)
    out[out < 0.0] = 0.0
    return out


def _silverman_bandwidth(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    n = max(len(x), 1)
    sigma = np.std(x) if n > 1 else 0.0
    if sigma <= 1e-12:
        sigma = 1e-3
    return 1.06 * sigma * (n ** (-1.0 / 5.0))


@dataclasses.dataclass
class KernelWindow:
    """Precompute kernel windows over a grid in logC.

    For each grid value logC0, compute weights w_j(C0) = K((logC_j - logC0)/h).
    Weights are normalized to sum to 1 within numerical tolerance.
    """

    grid_logC: np.ndarray
    weights: List[np.ndarray]  # per-grid normalized weights over selected indices
    indices: List[np.ndarray]  # per-grid indices (support set)

    @staticmethod
    def build(
        logC: np.ndarray,
        grid_logC: np.ndarray,
        bandwidth: Optional[float] = None,
        kernel: str = "gaussian",
        support_threshold_frac: float = 1e-3,
        boundary_correction: bool = True,
        reflection_multiple: float = 2.0,
    ) -> "KernelWindow":
        logC = np.asarray(logC, dtype=float)
        if bandwidth is None or bandwidth <= 0:
            bandwidth = _silverman_bandwidth(logC)
        if kernel == "gaussian":
            K = _gaussian_kernel
        elif kernel == "epanechnikov":
            K = _epanechnikov_kernel
        else:
            raise ValueError(f"Unknown kernel '{kernel}'")
        weights: List[np.ndarray] = []
        indices: List[np.ndarray] = []
        # Precompute boundaries for reflection-based bias correction near edges
        L = float(np.nanmin(logC))
        U = float(np.nanmax(logC))
        for lc0 in grid_logC:
            z = (logC - lc0) / float(bandwidth)
            w_full = K(z)
            if boundary_correction:
                h = float(bandwidth)
                # Left boundary reflection
                if (lc0 - L) <= reflection_multiple * h + 1e-12:
                    zL = ((2.0 * L - logC) - lc0) / h
                    w_full = w_full + K(zL)
                # Right boundary reflection
                if (U - lc0) <= reflection_multiple * h + 1e-12:
                    zR = ((2.0 * U - logC) - lc0) / h
                    w_full = w_full + K(zR)
            # Support thresholding
            wmax = float(np.max(w_full)) if len(w_full) > 0 else 0.0
            if wmax <= 0:
                idx = np.arange(0, len(logC))
                w = np.ones_like(logC) / float(max(len(logC), 1))
            else:
                mask = w_full >= (support_threshold_frac * wmax)
                idx = np.nonzero(mask)[0]
                w = w_full[idx]
                s = float(w.sum())
                if s <= 0:
                    w = np.ones_like(idx, dtype=float) / float(max(len(idx), 1))
                else:
                    w /= s
            weights.append(w.astype(float))
            indices.append(idx.astype(int))
        return KernelWindow(grid_logC=np.asarray(grid_logC), weights=weights, indices=indices)


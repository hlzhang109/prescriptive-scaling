#!/usr/bin/env python3
"""
Utilities for skill frontier estimation: kernels, bandwidth rule, weighted
quantiles, isotonic regression (PAV), simplex directions, and ModelPanel.

Kept dependency-light (NumPy only).
"""

from __future__ import annotations

import dataclasses
import numpy as np
from typing import List, Optional


def gaussian_kernel(x: np.ndarray) -> np.ndarray:
    """Standard normal-like kernel, max normalized to 1."""
    return np.exp(-0.5 * np.square(x))


def epanechnikov_kernel(x: np.ndarray) -> np.ndarray:
    """Epanechnikov kernel with compact support [-1,1]."""
    out = 1.0 - np.square(x)
    out[out < 0.0] = 0.0
    return out


def silverman_bandwidth(x: np.ndarray) -> float:
    """Silverman's rule-of-thumb bandwidth for univariate data."""
    x = np.asarray(x, dtype=float)
    n = max(len(x), 1)
    sigma = np.std(x) if n > 1 else 0.0
    if sigma <= 1e-12:
        sigma = 1e-3
    return 1.06 * sigma * (n ** (-1.0 / 5.0))


def weighted_quantile(
    values: np.ndarray,
    quantile: float,
    sample_weight: Optional[np.ndarray] = None,
    values_already_sorted: bool = False,
) -> float:
    """Compute a weighted quantile of `values`. Non-interpolated step definition."""
    assert 0.0 <= quantile <= 1.0, "quantile must be in [0,1]"
    v = np.asarray(values, dtype=float)
    if sample_weight is None:
        w = np.ones_like(v)
    else:
        w = np.asarray(sample_weight, dtype=float).copy()
        w[w < 0] = 0.0
    if not values_already_sorted:
        sorter = np.argsort(v)
        v = v[sorter]
        w = w[sorter]
    cum_w = np.cumsum(w)
    if cum_w[-1] <= 0:
        return float(v[-1])
    target = quantile * cum_w[-1]
    idx = np.searchsorted(cum_w, target, side="left")
    idx = min(max(idx, 0), len(v) - 1)
    return float(v[idx])


def isotonic_regression_monotone_increasing(y: np.ndarray, w: Optional[np.ndarray] = None) -> np.ndarray:
    """Project `y` onto non-decreasing sequences via PAV (weights optional)."""
    y = np.asarray(y, dtype=float)
    n = y.shape[0]
    if w is None:
        w = np.ones(n, dtype=float)
    else:
        w = np.asarray(w, dtype=float)
    mean = y.copy()
    weight = w.copy()
    start = np.arange(n)
    end = np.arange(n)
    k = 0
    tol = 1e-12
    while k < n - 1:
        if mean[k] <= mean[k + 1] + tol:
            k += 1
            continue
        num = mean[k] * weight[k] + mean[k + 1] * weight[k + 1]
        den = weight[k] + weight[k + 1]
        mean[k] = num / den
        weight[k] = den
        end[k] = end[k + 1]
        mean = np.delete(mean, k + 1)
        weight = np.delete(weight, k + 1)
        start = np.delete(start, k + 1)
        end = np.delete(end, k + 1)
        n -= 1
        while k > 0 and mean[k - 1] > mean[k] + tol:
            num = mean[k - 1] * weight[k - 1] + mean[k] * weight[k]
            den = weight[k - 1] + weight[k]
            mean[k - 1] = num / den
            weight[k - 1] = den
            end[k - 1] = end[k]
            mean = np.delete(mean, k)
            weight = np.delete(weight, k)
            start = np.delete(start, k)
            end = np.delete(end, k)
            n -= 1
            k -= 1
    fitted = np.zeros_like(y)
    for m, s, e in zip(mean, start, end):
        fitted[s : e + 1] = m
    return fitted


def generate_simplex_directions(n_dim: int, num_directions: int, seed: int = 0) -> np.ndarray:
    """Generate directions on the positive simplex (axis directions first)."""
    rng = np.random.default_rng(seed)
    axes = []
    for i in range(n_dim):
        e = np.zeros(n_dim)
        e[i] = 1.0
        axes.append(e)
    axes = np.asarray(axes, dtype=float)
    balanced = np.ones(n_dim) / float(n_dim)
    remaining = max(0, num_directions - (n_dim + 1))
    rand = rng.dirichlet(np.ones(n_dim), size=remaining) if remaining > 0 else np.zeros((0, n_dim))
    others = np.vstack([balanced.reshape(1, -1), rand]).astype(float)
    rounded = np.round(others, 6)
    axes_rounded = np.round(axes, 6)
    mask_keep = [not np.any(np.all(axes_rounded == row, axis=1)) for row in rounded]
    others = others[np.array(mask_keep, dtype=bool)]
    seen = set()
    uniq_others = []
    for row in np.round(others, 6):
        key = tuple(row.tolist())
        if key in seen:
            continue
        seen.add(key)
        uniq_others.append(row.astype(float))
    others = np.array(uniq_others, dtype=float) if uniq_others else np.zeros((0, n_dim))
    out = np.vstack([axes, others])
    return out


@dataclasses.dataclass
class ModelPanel:
    """Holds per-model compute and accuracy vectors."""

    models: List[str]
    logC: np.ndarray
    A: np.ndarray
    tasks: List[str]

    def __post_init__(self) -> None:
        self.logC = np.asarray(self.logC, dtype=float)
        self.A = np.asarray(self.A, dtype=float)
        assert self.A.ndim == 2, "A must be 2D"
        assert self.A.shape[0] == len(self.models), "row count mismatch"
        assert self.A.shape[1] == len(self.tasks), "task count mismatch"
        self.A = np.clip(self.A, 0.0, 1.0)

    @property
    def num_models(self) -> int:
        return len(self.models)

    @property
    def num_tasks(self) -> int:
        return self.A.shape[1]

    def task_index(self, name: str) -> int:
        return self.tasks.index(name)


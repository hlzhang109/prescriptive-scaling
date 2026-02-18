"""Shared pinball loss utilities."""

from __future__ import annotations

import numpy as np


def smooth_pinball_loss(r: np.ndarray, tau: float, k_smooth: float = 50.0) -> np.ndarray:
    """Smoothed pinball loss applied elementwise."""
    r = np.asarray(r, float)
    return (np.logaddexp(0.0, k_smooth * r) / k_smooth) + (tau - 1.0) * r

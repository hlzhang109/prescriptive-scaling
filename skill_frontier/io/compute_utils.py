"""Compute conversion utilities shared across plotting scripts."""

from __future__ import annotations

from typing import Any, Union

import numpy as np
import pandas as pd


def compute_flops_from_tokens_params(
    tokens: Any, params: Any, multiplier: float = 6.0
) -> Union[pd.Series, np.ndarray]:
    """Convert tokens/params (T, B) into FLOPs (not zFLOPs)."""
    tokens_num = pd.to_numeric(tokens, errors="coerce")
    params_num = pd.to_numeric(params, errors="coerce")
    return multiplier * tokens_num * params_num * 1e21


def compute_flops_from_zflops(zflops: Any) -> np.ndarray:
    """Convert zFLOPs into FLOPs as float array."""
    z = pd.to_numeric(zflops, errors="coerce")
    if hasattr(z, "to_numpy"):
        z = z.to_numpy(dtype=float)
    else:
        z = np.asarray(z, dtype=float)
    return z * 1e21

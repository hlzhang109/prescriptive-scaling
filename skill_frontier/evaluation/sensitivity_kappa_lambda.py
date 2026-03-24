"""κ/λ sensitivity utilities for sigmoid frontier fitting.

This module provides helpers to:
  - split a period's data into grouped, z-stratified train/val folds
  - evaluate smoothed pinball loss with a fixed κ_eval
  - compute binned coverage error (τ_hat − τ) using train-defined bins

It is intentionally lightweight and reuses existing fitting/binning primitives.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Tuple

import numpy as np

from skill_frontier.evaluation.binning import compute_bin_statistics


def smooth_pinball_loss(u: np.ndarray, tau: float, kappa: float) -> np.ndarray:
    """Stable smoothed pinball loss (matches the enhanced sigmoid fitter).

    Args:
        u: residuals u = y - yhat
        tau: target quantile
        kappa: smoothing parameter κ
    """
    u = np.asarray(u, float)
    return np.logaddexp(0.0, float(kappa) * u) / float(kappa) + (float(tau) - 1.0) * u


def compute_overlap_edges(edges_train: np.ndarray, z_train: np.ndarray, z_eval: np.ndarray) -> np.ndarray:
    """Clip train bin edges to the z-overlap region between train and eval sets."""
    edges_train = np.asarray(edges_train, float)
    z_train = np.asarray(z_train, float)
    z_eval = np.asarray(z_eval, float)
    if edges_train.size < 2 or z_train.size == 0:
        return np.array([], float)

    if z_eval.size >= 1:
        z_lo = float(max(np.min(z_train), np.min(z_eval)))
        z_hi = float(min(np.max(z_train), np.max(z_eval)))
    else:
        z_lo = float(np.min(z_train))
        z_hi = float(np.max(z_train))

    e = edges_train.copy()
    e[0] = max(float(e[0]), z_lo)
    e[-1] = min(float(e[-1]), z_hi)
    keep = [i for i in range(len(e) - 1) if float(e[i + 1]) > float(e[i]) + 1e-12]
    if not keep:
        return np.array([], float)
    return np.array([e[i] for i in keep] + [e[keep[-1] + 1]], float)


def mask_in_edges(z: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """Mask for values inside [edges[0], edges[-1]] (inclusive)."""
    z = np.asarray(z, float)
    edges = np.asarray(edges, float)
    if edges.size < 2:
        return np.zeros_like(z, dtype=bool)
    return (z >= float(edges[0])) & (z <= float(edges[-1]))


@dataclass(frozen=True)
class CalibrationSummary:
    signed_micro: float
    abs_micro: float
    signed_macro: float
    abs_macro: float
    n: int


def calibration_summary(
    z: np.ndarray,
    y: np.ndarray,
    yhat: np.ndarray,
    *,
    edges: np.ndarray,
    tau: float,
) -> CalibrationSummary:
    """Compute binned coverage error summaries using existing binning logic."""
    rows = compute_bin_statistics(z, y, yhat, np.asarray(edges, float), tau=float(tau))
    ns = []
    signed = []
    abs_err = []
    for (_bid, _lo, _hi, n, hat_tau, ae) in rows:
        if n <= 0:
            continue
        if not (np.isfinite(hat_tau) and np.isfinite(ae)):
            continue
        ns.append(int(n))
        signed.append(float(hat_tau) - float(tau))
        abs_err.append(float(ae))

    if not ns:
        nan = float("nan")
        return CalibrationSummary(nan, nan, nan, nan, 0)

    n_vec = np.asarray(ns, float)
    signed_vec = np.asarray(signed, float)
    abs_vec = np.asarray(abs_err, float)
    denom = float(np.sum(n_vec))
    signed_micro = float(np.sum(n_vec * signed_vec) / denom) if denom > 0 else float("nan")
    abs_micro = float(np.sum(n_vec * abs_vec) / denom) if denom > 0 else float("nan")
    signed_macro = float(np.mean(signed_vec))
    abs_macro = float(np.mean(abs_vec))
    return CalibrationSummary(signed_micro, abs_micro, signed_macro, abs_macro, int(denom))


def pinball_mean(y: np.ndarray, yhat: np.ndarray, *, tau: float, kappa: float) -> float:
    y = np.asarray(y, float)
    yhat = np.asarray(yhat, float)
    m = np.isfinite(y) & np.isfinite(yhat)
    if not np.any(m):
        return float("nan")
    u = y[m] - yhat[m]
    return float(np.mean(smooth_pinball_loss(u, tau=float(tau), kappa=float(kappa))))


def split_train_val_group_stratified(
    *,
    group_ids: np.ndarray,
    z: np.ndarray,
    seed: int,
    frac_train: float = 0.5,
    n_bins: int = 10,
) -> Tuple[np.ndarray, np.ndarray]:
    """Stratified grouped split for internal CV within a period.

    Groups are defined by exact `group_ids` (e.g., pretraining compute Ci).
    Groups are binned by group-level median z, then groups are assigned to
    train/val within each bin targeting ~`frac_train` of points in train.
    """
    group_ids = np.asarray(group_ids)
    z = np.asarray(z, float)
    if group_ids.size != z.size:
        raise ValueError("group_ids and z must have the same length")
    if group_ids.size == 0:
        return np.array([], bool), np.array([], bool)

    rng = np.random.default_rng(int(seed))

    uniq, inv = np.unique(group_ids, return_inverse=True)
    n_groups = int(uniq.size)
    if n_groups <= 1:
        # Degenerate: all points share one group; fall back to random point split.
        m = rng.random(group_ids.size) < float(frac_train)
        return m, ~m

    group_z = np.full(n_groups, np.nan, float)
    group_w = np.zeros(n_groups, int)
    for g in range(n_groups):
        m = inv == g
        group_w[g] = int(np.sum(m))
        group_z[g] = float(np.nanmedian(z[m]))

    # Quantile bins over groups (drop duplicates if group_z has ties).
    q = np.linspace(0.0, 1.0, num=int(max(1, min(n_bins, n_groups))) + 1)
    edges = np.unique(np.quantile(group_z, q))
    if edges.size < 2:
        bin_idx = np.zeros(n_groups, int)
    else:
        bin_idx = np.searchsorted(edges[1:-1], group_z, side="right")

    group_is_train = np.zeros(n_groups, bool)
    for b in range(int(np.max(bin_idx)) + 1):
        g_idx = np.where(bin_idx == b)[0]
        if g_idx.size == 0:
            continue
        rng.shuffle(g_idx)
        total = int(np.sum(group_w[g_idx]))
        target = float(frac_train) * float(total)
        acc = 0.0
        for gi in g_idx:
            if acc < target:
                group_is_train[gi] = True
                acc += float(group_w[gi])

        # Ensure both sides non-empty within the bin when possible.
        if np.all(group_is_train[g_idx]):
            group_is_train[int(g_idx[-1])] = False
        if not np.any(group_is_train[g_idx]):
            group_is_train[int(g_idx[0])] = True

    mask_train = group_is_train[inv]
    mask_val = ~mask_train
    if not np.any(mask_train) or not np.any(mask_val):
        # Global fallback: random group split
        g_perm = np.arange(n_groups)
        rng.shuffle(g_perm)
        cut = int(round(float(frac_train) * float(n_groups)))
        train_g = set(int(g) for g in g_perm[:cut])
        mask_train = np.array([int(g) in train_g for g in inv], bool)
        mask_val = ~mask_train
    return mask_train, mask_val


def select_best_hyperparams(
    candidates: Iterable[Dict[str, float]],
    *,
    tau: float,
    prefer_kappa: float = 50.0,
) -> Dict[str, float]:
    """Select best (kappa, lambda) with the manuscript tie-break rules."""
    best = None
    best_key = None

    for c in candidates:
        pin = float(c.get("val_pinball", float("inf")))
        calib_abs = float(c.get("val_calib_abs", float("inf")))
        lam = float(c.get("lambda_b", float("inf")))
        kap = float(c.get("kappa_train", float("inf")))

        pin_key = pin if np.isfinite(pin) else float("inf")
        calib_key = calib_abs if np.isfinite(calib_abs) else float("inf")
        lam_key = lam if np.isfinite(lam) else float("inf")
        kappa_key = abs(kap - float(prefer_kappa)) if np.isfinite(kap) else float("inf")
        key = (pin_key, calib_key, lam_key, kappa_key)
        if best_key is None or key < best_key:
            best_key = key
            best = c

    return dict(best or {})


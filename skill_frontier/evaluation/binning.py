"""Binning utilities for evaluation and analysis.

This module provides functions for creating equal-mass bins and computing
statistics within bins, primarily used for coverage-based evaluation metrics.
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np


def create_equal_mass_bins(
    z_train: np.ndarray, K: int, min_bin: int
) -> np.ndarray:
    """Group-aware equal-mass binning on z=log10(FLOPs).

    This function creates bins with approximately equal mass (number of samples)
    while respecting the constraint that identical z values are never split
    across bins. Boundaries occur only at unique-z group edges.

    Args:
        z_train: Array of log10(FLOPs) values to bin
        K: Target number of bins
        min_bin: Minimum number of samples per bin (bins violating this are merged)

    Returns:
        Strictly increasing edges array of length B+1 (B bins). If there is
        only one unique z, returns two identical edges [z0, z0] to form one bin.
    """
    z = np.sort(np.asarray(z_train, float))
    if z.size == 0:
        return np.array([])

    # Compress to groups of identical values
    uniq = []
    counts = []
    last = z[0]
    cnt = 1
    for v in z[1:]:
        if v == last:
            cnt += 1
        else:
            uniq.append(float(last))
            counts.append(int(cnt))
            last = v
            cnt = 1
    uniq.append(float(last))
    counts.append(int(cnt))

    G = len(uniq)
    N = int(z.size)
    if G == 1:
        u = float(uniq[0])
        return np.array([u, u], float)

    # Effective max bins is at most number of groups
    K_eff = int(max(1, min(int(K), G)))
    target = max(int(min_bin), int(np.ceil(N / K_eff)))
    edges: List[float] = [float(uniq[0])]
    acc = 0
    for i in range(G):
        acc += counts[i]
        at_last_group = i == G - 1
        if acc >= target or at_last_group:
            edges.append(float(uniq[i]))
            acc = 0

    # Merge bins violating min_bin by removing the boundary towards the smaller neighbor
    def _counts_for_edges(z_sorted: np.ndarray, E: List[float]) -> List[int]:
        if len(E) < 2:
            return []
        # For bins [E[i], E[i+1]) except last [E[-2], E[-1]] inclusive
        L = np.searchsorted(z_sorted, np.array(E[:-1], float), side="left")
        R = np.searchsorted(z_sorted, np.array(E[1:-1], float), side="left")
        cnt = [int(r - l) for l, r in zip(L, R)]
        last_left = int(L[-1])
        last_right = int(np.searchsorted(z_sorted, float(E[-1]), side="right"))
        cnt.append(int(last_right - last_left))
        return cnt

    while True:
        cnt = _counts_for_edges(z, edges)
        if not cnt or len(cnt) <= 1:
            break
        bad = next((i for i, c in enumerate(cnt) if c < min_bin), None)
        if bad is None:
            break
        if bad == 0:
            # merge bin 0 with next: remove edges[1]
            del edges[1]
        elif bad == len(cnt) - 1:
            # merge last with previous: remove edges[-2]
            del edges[-2]
        else:
            left_c = cnt[bad - 1]
            right_c = cnt[bad + 1]
            # remove boundary towards the smaller neighbor
            if left_c <= right_c:
                # merge with left: remove edges[bad]
                del edges[bad]
            else:
                # merge with right: remove edges[bad + 1]
                del edges[bad + 1]

    # Ensure strictly increasing (remove accidental duplicates if any)
    cleaned = [edges[0]]
    for e in edges[1:]:
        if e > cleaned[-1]:
            cleaned.append(e)
    if len(cleaned) < 2:
        # fallback to single bin
        u = float(uniq[-1])
        cleaned = [u, u]

    return np.asarray(cleaned, float)


def compute_bin_statistics(
    z: np.ndarray,
    y: np.ndarray,
    yhat: np.ndarray,
    edges: np.ndarray,
    tau: float,
) -> List[Tuple[int, float, float, int, float, float]]:
    """Compute coverage statistics for each bin.

    For each bin defined by edges, computes the empirical coverage (fraction
    of y values below yhat) and the absolute error from target tau.

    Args:
        z: Array of log10(FLOPs) values
        y: Array of true accuracy values
        yhat: Array of predicted accuracy values (frontier)
        edges: Bin edges (length B+1 for B bins)
        tau: Target coverage quantile

    Returns:
        List of tuples (bin_id, z_lo, z_hi, n, hat_tau, abs_err) for each bin:
            - bin_id: Integer bin index
            - z_lo: Lower bin edge
            - z_hi: Upper bin edge
            - n: Number of samples in bin (with finite y)
            - hat_tau: Empirical coverage in bin
            - abs_err: |hat_tau - tau|
    """
    out = []
    for i in range(len(edges) - 1):
        lo, hi = float(edges[i]), float(edges[i + 1])
        if i < len(edges) - 2:
            mask = (z >= lo) & (z < hi)
        else:
            mask = (z >= lo) & (z <= hi)
        # Only count rows with finite y for coverage; yhat is finite by construction
        mask = mask & np.isfinite(y)
        n = int(np.sum(mask))
        if n == 0:
            hat_tau = float("nan")
            err = float("nan")
        else:
            cov = np.mean(y[mask] <= yhat[mask])
            hat_tau = float(cov)
            err = float(abs(hat_tau - tau))
        out.append((i, lo, hi, n, hat_tau, err))
    return out

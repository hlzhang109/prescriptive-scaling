#!/usr/bin/env python3
"""
Smooth Single‑Skill Frontier (Accuracy vs FLOPs)
------------------------------------------------

Implements a simple, dependency‑light frontier smoother for the 1D setting
(single skill vs compute) described in discussion:

  - Sort by compute x (defaults to log10 FLOPs).
  - Bin x into fixed slices; compute a high quantile (tau≈0.98) of y in each bin.
  - Enforce monotonicity by cumulative max across bins.
  - Smooth via PCHIP if SciPy is available; otherwise linear interpolation.
  - Final envelope applies an FDH guard: max(smooth, running‑max(y)).

This file adds a standalone CLI and reusable function without changing any
existing scripts/commands in the repo. It is safe to import elsewhere.

Usage examples:
  # Open LLM Leaderboard (with_tokens schema) — auto detect Raw tasks
  python scripts/smooth_single_skill_frontier.py \
      --csv tables/open_llm_leaderboard/open_llm_leaderboard_with_tokens.csv \
      --compute_product_cols "Pretraining tokens (T)" "#Params (B)" \
      --compute_multiplier 6.0 \
      --out_dir outputs/smooth_frontier_oll

  # Explicit task list and logC column already present
  python scripts/smooth_single_skill_frontier.py \
      --csv tables/merged_livebench.csv --logC_col logC \
      --tasks a_reasoning a_math a_code --out_dir outputs/smooth_livebench

The output directory gets one PNG+PDF per task named
  smooth_frontier__<task>.(png|pdf)
and a compact CSV with sampled curve points for each task.
"""

from __future__ import annotations

import argparse
import csv as _csv
import math
import os
from typing import Iterable, List, Optional, Sequence, Tuple, Dict, Set

import numpy as np
from skill_frontier.io.output_paths import sanitize_task_name  # type: ignore
from skill_frontier.io.csv_utils import maybe_scale_task_values  # type: ignore

# Centralized period-4 scheme definitions.
from skill_frontier.core.period_scheme import (  # type: ignore
    PERIOD4_BOUNDS_NEW as _PERIOD4_BOUNDS_NEW,
    PERIOD4_SPLITS_NEW as _PERIOD4_SPLITS_NEW,
    PERIOD4_SPLITS_SINGLE_NEW as _PERIOD4_SPLITS_SINGLE_NEW,
    PERIOD4_BOUNDS_OLL_OLD as _PERIOD4_BOUNDS_OLL_OLD,
    PERIOD4_SPLITS_OLL_OLD as _PERIOD4_SPLITS_OLL_OLD,
    PERIOD4_SPLITS_SINGLE_OLL_OLD as _PERIOD4_SPLITS_SINGLE_OLL_OLD,
    PERIOD4_SCHEME as _PERIOD4_SCHEME,
    PERIOD4_BOUNDS,
    PERIOD4_SPLITS,
    PERIOD4_SPLITS_SINGLE,
)

# Default L2 penalty on the sigmoid slope parameter `b` in the enhanced fitter.
DEFAULT_LAMBDA_B: float = 1e-3
# Runtime override (e.g. via CLI); used when callers don't pass lambda_b explicitly.
LAMBDA_B: float = DEFAULT_LAMBDA_B

# Equal-mass binning shared with evaluation code.
try:
    from skill_frontier.evaluation.binning import create_equal_mass_bins  # type: ignore
except Exception:  # pragma: no cover
    create_equal_mass_bins = None  # type: ignore


def smooth_frontier(
    x: np.ndarray,
    y: np.ndarray,
    tau: float = 0.98,
    bins: int = 120,
    use_log10_x: bool = True,
    guard_fdh: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return a smooth, monotone upper frontier y_hat(x).

    Parameters
    - x: compute values (FLOPs). If `use_log10_x`, these are transformed.
    - y: accuracy in [0,1].
    - tau: high quantile per bin to approximate the upper envelope.
    - bins: number of equal‑width bins in x‑space (after log if enabled).
    - use_log10_x: map x -> log10(x) for stability.

    Returns
    - xs_sample: x grid (original x scale) where the curve is sampled
    - y_hat: smoothed, monotone, FDH‑guarded upper frontier at xs_sample
    """
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    if x.size == 0:
        return np.array([]), np.array([])
    if use_log10_x:
        lx = np.log10(np.maximum(x, 1e-300))
    else:
        lx = x.copy()
    order = np.argsort(lx)
    lx = lx[order]
    y = y[order]

    # FDH staircase guard (running max in original order along lx)
    fdh_y = np.maximum.accumulate(y)

    # Build bins in lx and take quantiles
    edges = np.linspace(float(lx.min()), float(lx.max()), bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    q = np.full(bins, np.nan, dtype=float)
    for i in range(bins):
        lo, hi = edges[i], edges[i + 1]
        if i < bins - 1:
            mask = (lx >= lo) & (lx < hi)
        else:
            mask = (lx >= lo) & (lx <= hi)
        yi = y[mask]
        if yi.size:
            q[i] = float(np.nanquantile(yi, tau))
        else:
            # Backfill from previous if empty
            q[i] = float(q[i - 1]) if i > 0 else float("nan")
    if np.isnan(q[0]):
        # pick a small baseline (min non‑nan) to start
        finite = q[np.isfinite(q)]
        q[0] = float(finite[0]) if finite.size else float(np.nanmin(y))
    # Fill any remaining nans
    for i in range(1, bins):
        if not np.isfinite(q[i]):
            q[i] = q[i - 1]
    # Enforce monotone non‑decreasing
    q = np.maximum.accumulate(q)

    # Smooth with PCHIP, fallback to linear interp
    try:
        from scipy.interpolate import PchipInterpolator  # type: ignore

        base = PchipInterpolator(centers, q, extrapolate=True)
        def eval_base(xx: np.ndarray) -> np.ndarray:
            return base(xx)
    except Exception:
        def eval_base(xx: np.ndarray) -> np.ndarray:
            return np.interp(xx, centers, q, left=q[0], right=q[-1])

    # Sample on a dense grid in lx
    lx_grid = np.linspace(float(lx.min()), float(lx.max()), num=400)
    y_smooth = eval_base(lx_grid)
    # Clip to [0,1]
    y_smooth = np.clip(y_smooth, 0.0, 1.0)
    # FDH guard: ensure never below the running max curve (interp in lx)
    if guard_fdh:
        fdh_on_grid = np.interp(lx_grid, lx, fdh_y, left=fdh_y[0], right=fdh_y[-1])
        y_hat = np.maximum(y_smooth, fdh_on_grid)
    else:
        y_hat = y_smooth

    # Map grid back to original x scale
    if use_log10_x:
        xs_sample = (10.0 ** lx_grid)
    else:
        xs_sample = lx_grid
    return xs_sample.astype(float), y_hat.astype(float)


# --------------------------------------------------------------------------------------
# Parametric sigmoid frontier (logistic on log10-compute)
# --------------------------------------------------------------------------------------


def _sigmoid(u: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-u))


def _softplus(u: np.ndarray | float) -> np.ndarray | float:
    return np.log1p(np.exp(u))


def fit_sigmoid_frontier(
    x: np.ndarray,
    y: np.ndarray,
    tau: float = 0.98,
    use_log10_x: bool = True,
    grid_points: int = 400,
    fit_mode: str = "quantile_per_point",
    bins_for_fit: Optional[int] = None,
    min_bin_size_for_fit: Optional[int] = None,
    bin_frontier_quantile: float = 0.90,
    bin_trim_fraction: float = 0.05,
    bin_edges_for_fit: Optional[np.ndarray] = None,
    lambda_b: float | None = None,
    kappa_final: float = 50.0,
    curve_x_limits: Optional[Tuple[float, float]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Fit a monotone sigmoid frontier using the enhanced pinball optimizer."""
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

    def _compute_bin_frontier_points() -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Aggregate per-point data into robust bin-level frontier targets."""
        if bin_edges_for_fit is not None:
            edges_local = np.asarray(bin_edges_for_fit, float)
            if edges_local.size < 2:
                return np.array([]), np.array([]), np.array([])
        else:
            if create_equal_mass_bins is None:
                return np.array([]), np.array([]), np.array([])
            if bins_for_fit is None or bins_for_fit <= 0:
                return np.array([]), np.array([]), np.array([])
            mb = int(max(1, min_bin_size_for_fit if min_bin_size_for_fit is not None else 30))
            try:
                edges_local = create_equal_mass_bins(z, int(max(1, bins_for_fit)), mb)
            except Exception:
                return np.array([]), np.array([]), np.array([])
            if edges_local.size < 2:
                return np.array([]), np.array([]), np.array([])

        B = int(edges_local.size - 1)
        bin_idx = np.searchsorted(edges_local, z, side="right") - 1
        bin_idx = np.clip(bin_idx, 0, B - 1)
        z_bin = np.zeros(B, dtype=float)
        y_bin = np.zeros(B, dtype=float)
        w_bin = np.zeros(B, dtype=float)
        for b in range(B):
            m_b = bin_idx == b
            if not np.any(m_b):
                continue
            z_b = z[m_b]
            y_b = y[m_b]
            n_b = int(y_b.size)
            z_bin[b] = 0.5 * (edges_local[b] + edges_local[b + 1])
            y_sorted = np.sort(y_b)[::-1]
            trim_k = int(np.floor(float(bin_trim_fraction) * n_b))
            trim_k = min(trim_k, max(n_b - 1, 0))
            y_trimmed = y_sorted[trim_k:] if trim_k > 0 else y_sorted
            if y_trimmed.size == 0:
                continue
            y_bin[b] = float(np.quantile(y_trimmed, float(bin_frontier_quantile)))
            w_bin[b] = 1.0 if min_bin_size_for_fit is None else float(n_b)
        m_keep = w_bin > 0
        return z_bin[m_keep], y_bin[m_keep], w_bin[m_keep]

    mode = (fit_mode or "quantile_per_point").lower()
    z_bin: np.ndarray
    y_bin: np.ndarray
    w_bin: np.ndarray
    if mode == "robust_bin_frontier":
        z_bin, y_bin, w_bin = _compute_bin_frontier_points()
        if z_bin.size < 3:
            mode = "quantile_per_point"
    else:
        z_bin = y_bin = w_bin = np.array([])

    # Build training targets (weighted) for the enhanced optimizer
    if mode == "robust_bin_frontier":
        w_int = np.maximum(1, np.round(w_bin).astype(int))
        z_fit = np.repeat(z_bin, w_int)
        y_fit = np.repeat(y_bin, w_int)
    else:
        z_fit, y_fit = z, y
    if z_fit.size < 3:
        return np.array([]), np.array([])

    from skill_frontier.core.sigmoid_quantile_optimizer import (  # type: ignore
        fit_sigmoid_enhanced,
        sigmoid_pred,
    )

    res = fit_sigmoid_enhanced(
        z_fit,
        y_fit,
        tau=float(tau),
        kappa_final=float(kappa_final),
        lambda_b=float(LAMBDA_B if lambda_b is None else lambda_b),
        n_zstar_grid=10,
        n_b_grid=10,
        n_random=100,
        seed=0,
    )
    if not res.success or not np.all(np.isfinite(res.params)):
        return np.array([]), np.array([])

    z_grid_min = float(z.min())
    z_grid_max = float(z.max())
    if curve_x_limits is not None:
        lo, hi = curve_x_limits
        try:
            lo_f = float(lo)
            hi_f = float(hi)
        except Exception:
            lo_f = hi_f = float("nan")
        if np.isfinite(lo_f) and np.isfinite(hi_f) and hi_f > lo_f:
            if use_log10_x:
                if lo_f > 0.0:
                    z_grid_min = float(np.log10(max(lo_f, 1e-300)))
                    z_grid_max = float(np.log10(max(hi_f, 1e-300)))
            else:
                z_grid_min = lo_f
                z_grid_max = hi_f

    z_grid = np.linspace(z_grid_min, z_grid_max, num=grid_points)
    y_hat = sigmoid_pred(res.params, z_grid)
    y_hat = np.clip(y_hat, 0.0, 1.0)
    xs_sample = (10.0 ** z_grid) if use_log10_x else z_grid
    return xs_sample.astype(float), y_hat.astype(float)


def _detect_oll_raw_tasks(headers: Iterable[str]) -> List[str]:
    """Return known OLL Raw task columns present in headers (preserve order)."""
    canonical_new = [
        "IFEval Raw",
        "BBH Raw",
        "MATH Lvl 5 Raw",
        "GPQA Raw",
        "MUSR Raw",
        "MMLU-PRO Raw",
    ]
    canonical_old = [
        "ARC",
        "HellaSwag",
        "MMLU",
        "TruthfulQA",
        "Winogrande",
        "GSM8K",
    ]
    hset = set(h.strip() for h in headers)
    if any(c in hset for c in canonical_new):
        return [c for c in canonical_new if c in hset]
    return [c for c in canonical_old if c in hset]


def _normalize_task_matrix(Y: np.ndarray) -> np.ndarray:
    """Scale legacy percentage task columns to [0, 1] when detected."""
    arr = np.asarray(Y, dtype=float)
    if arr.ndim != 2 or arr.size == 0:
        return arr
    out = arr.copy()
    for j in range(out.shape[1]):
        out[:, j] = maybe_scale_task_values(out[:, j])
    return out


def _resolve_period4_runtime_config(
    date_col: Optional[str],
) -> Tuple[str, List[Tuple[str, Tuple[int, int], Tuple[int, int]]], List[Dict[str, str | List[str]]], List[Dict[str, str | List[str]]], str]:
    """Resolve period4 splits at runtime.

    For legacy OLL CSVs (``date`` column), default to the old 4-period scheme
    unless the environment already selected ``oll_old`` explicitly.
    """
    if _PERIOD4_SCHEME in {"oll_old", "old", "old_oll"}:
        return "oll_old", _PERIOD4_BOUNDS_OLL_OLD, _PERIOD4_SPLITS_OLL_OLD, _PERIOD4_SPLITS_SINGLE_OLL_OLD, "k"
    if date_col == "date":
        return "oll_old", _PERIOD4_BOUNDS_OLL_OLD, _PERIOD4_SPLITS_OLL_OLD, _PERIOD4_SPLITS_SINGLE_OLL_OLD, "k"
    return "new", _PERIOD4_BOUNDS_NEW, _PERIOD4_SPLITS_NEW, _PERIOD4_SPLITS_SINGLE_NEW, "t"


def _read_model_id(row: Dict[str, str]) -> str:
    """Return a model identifier from common columns.

    Prefers 'model', then 'Model sha', then 'name' if present.
    """
    from skill_frontier.io.csv_utils import extract_model_id  # type: ignore

    return extract_model_id(row)


def _load_xy_from_csv(
    path: str,
    task_cols: List[str],
    logC_col: Optional[str] = None,
    compute_product_cols: Optional[Tuple[str, str]] = None,
    compute_multiplier: float = 6.0,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Load compute x and multiple task columns y from CSV.

    Returns (x, Y, tasks_kept) where Y has shape (n, k) and tasks_kept is the
    list of task names actually returned (subset of task_cols present).
    """
    with open(path, "r", newline="") as f:
        reader = _csv.DictReader(f)
        rows = list(reader)
        headers = list(reader.fieldnames or [])
    if not rows:
        raise RuntimeError(f"No rows in {path}")
    # Determine x (compute)
    x_list: List[float] = []
    Y_list: List[List[float]] = []
    kept: List[str] = []
    # Verify tasks presence
    tasks = [t for t in task_cols if t in headers]
    if not tasks:
        raise RuntimeError("None of the requested task columns are in the CSV")
    for r in rows:
        # compute
        if logC_col and logC_col in r and (r[logC_col] not in (None, "", "nan", "NaN")):
            try:
                lc = float(r[logC_col])
                x_val = float(math.exp(lc))
            except Exception:
                x_val = float("nan")
        elif compute_product_cols and all(c in r for c in compute_product_cols):
            v1 = r.get(compute_product_cols[0], None)
            v2 = r.get(compute_product_cols[1], None)
            try:
                t = float(v1) if v1 not in (None, "", "nan", "NaN") else float("nan")
            except Exception:
                t = float("nan")
            try:
                b = float(v2) if v2 not in (None, "", "nan", "NaN") else float("nan")
            except Exception:
                b = float("nan")
            x_val = float(compute_multiplier * t * b) if (np.isfinite(t) and np.isfinite(b)) else float("nan")
        else:
            x_val = float("nan")
        if not np.isfinite(x_val) or x_val <= 0:
            continue
        # tasks
        row_vals: List[float] = []
        ok_any = False
        for t in tasks:
            v = r.get(t, None)
            if v in (None, "", "nan", "NaN"):
                row_vals.append(float("nan"))
            else:
                try:
                    row_vals.append(float(v))
                    ok_any = True
                except Exception:
                    row_vals.append(float("nan"))
        if ok_any:
            x_list.append(x_val)
            Y_list.append(row_vals)
    X = np.asarray(x_list, dtype=float)
    Y = np.asarray(Y_list, dtype=float)
    if Y.ndim != 2 or Y.shape[0] == 0:
        return X, np.zeros((0, 0), dtype=float), tasks
    Y = _normalize_task_matrix(Y)
    # Drop rows where all tasks are NaN
    keep_row = np.any(np.isfinite(Y), axis=1)
    X = X[keep_row]
    Y = Y[keep_row]
    # Identify tasks with enough finite values
    keep_task = np.sum(np.isfinite(Y), axis=0) >= 3
    Y = Y[:, keep_task]
    kept = [t for t, k in zip(tasks, keep_task) if k]
    return X, Y, kept


def _load_xy_with_year_filtered(
    path: str,
    task_cols: List[str],
    logC_col: Optional[str] = None,
    compute_product_cols: Optional[Tuple[str, str]] = None,
    compute_multiplier: float = 6.0,
    date_col: Optional[str] = None,
    manifest: Optional[Set[str]] = None,
    year_filter: Optional[Set[int]] = None,
) -> Tuple[np.ndarray, np.ndarray, List[str], np.ndarray]:
    """Load compute x, tasks Y, and years, filtering by manifest and year set.

    - manifest: if provided, only rows whose model id is in the set are kept
    - year_filter: if provided, only rows with year in the set are kept

    Returns (X, Y, tasks_kept, years).
    """
    with open(path, "r", newline="") as f:
        reader = _csv.DictReader(f)
        rows = list(reader)
        headers = list(reader.fieldnames or [])
    if not rows:
        return np.array([]), np.zeros((0, 0)), [], np.array([], dtype=int)
    if date_col is None:
        if "Upload To Hub Date" in headers:
            date_col = "Upload To Hub Date"
        elif "date" in headers:
            date_col = "date"
        else:
            raise RuntimeError("Could not detect date column; pass a CSV with 'Upload To Hub Date' or 'date'.")
    # Verify tasks presence
    tasks = [t for t in task_cols if t in headers]
    if not tasks:
        return np.array([]), np.zeros((0, 0)), [], np.array([], dtype=int)
    x_list: List[float] = []
    Y_list: List[List[float]] = []
    years: List[int] = []
    for r in rows:
        mid = _read_model_id(r)
        if manifest is not None and (mid == "" or mid not in manifest):
            continue
        # compute
        if logC_col and logC_col in r and (r[logC_col] not in (None, "", "nan", "NaN")):
            try:
                lc = float(r[logC_col])
                x_val = float(math.exp(lc))
            except Exception:
                x_val = float("nan")
        elif compute_product_cols and all(c in r for c in compute_product_cols):
            v1 = r.get(compute_product_cols[0], None)
            v2 = r.get(compute_product_cols[1], None)
            try:
                t = float(v1) if v1 not in (None, "", "nan", "NaN") else float("nan")
            except Exception:
                t = float("nan")
            try:
                b = float(v2) if v2 not in (None, "", "nan", "NaN") else float("nan")
            except Exception:
                b = float("nan")
            x_val = float(compute_multiplier * t * b) if (np.isfinite(t) and np.isfinite(b)) else float("nan")
        else:
            x_val = float("nan")
        if not np.isfinite(x_val) or x_val <= 0:
            continue
        # year
        yr = _parse_year(r.get(date_col, ""))
        if yr is None:
            continue
        if year_filter is not None and int(yr) not in year_filter:
            continue
        # tasks
        row_vals: List[float] = []
        ok_any = False
        for t in tasks:
            v = r.get(t, None)
            if v in (None, "", "nan", "NaN"):
                row_vals.append(float("nan"))
            else:
                try:
                    row_vals.append(float(v))
                    ok_any = True
                except Exception:
                    row_vals.append(float("nan"))
        if ok_any:
            x_list.append(x_val)
            Y_list.append(row_vals)
            years.append(int(yr))
    X = np.asarray(x_list, dtype=float)
    Y = np.asarray(Y_list, dtype=float)
    yrs = np.asarray(years, dtype=int)
    if Y.ndim != 2 or Y.shape[0] == 0:
        return X, np.zeros((0, 0), dtype=float), tasks, yrs
    Y = _normalize_task_matrix(Y)
    # Drop rows where all tasks NaN
    keep_row = np.any(np.isfinite(Y), axis=1)
    X = X[keep_row]
    Y = Y[keep_row]
    yrs = yrs[keep_row]
    # Keep tasks with enough finite values
    keep_task = np.sum(np.isfinite(Y), axis=0) >= 3
    Y = Y[:, keep_task]
    kept = [t for t, k in zip(tasks, keep_task) if k]
    return X, Y, kept, yrs


def _parse_year(s: str) -> Optional[int]:
    s = (s or "").strip()
    if not s or s.lower() in ("nan", "na", "none"):
        return None
    import datetime, re
    for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%d/%m/%Y", "%m/%d/%Y", "%Y-%m", "%Y/%m", "%Y"):
        try:
            return datetime.datetime.strptime(s, fmt).year
        except Exception:
            pass
    m = re.search(r"(20\d{2})", s)
    if m:
        try:
            return int(m.group(1))
        except Exception:
            return None
    return None


def _parse_year_month(s: str) -> Optional[Tuple[int, int]]:
    s = (s or "").strip()
    if not s or s.lower() in ("nan", "na", "none"):
        return None
    import datetime, re
    formats = [
        "%Y-%m-%d",
        "%Y/%m/%d",
        "%d/%m/%Y",
        "%m/%d/%Y",
        "%Y-%m",
        "%Y/%m",
        "%Y",
    ]
    for fmt in formats:
        try:
            dt = datetime.datetime.strptime(s, fmt)
            if fmt in ("%Y-%m", "%Y/%m"):
                return (dt.year, dt.month)
            if fmt == "%Y":
                return (dt.year, 1)
            return (dt.year, dt.month)
        except Exception:
            continue
    # Fallback: regex year-month
    m = re.search(r"(20\d{2})-(\d{1,2})", s)
    if m:
        try:
            return (int(m.group(1)), int(m.group(2)))
        except Exception:
            return None
    m = re.search(r"(20\d{2})", s)
    if m:
        try:
            return (int(m.group(1)), 1)
        except Exception:
            return None
    return None


def _assign_period_index(
    year: int,
    month: int,
    period4_bounds: Optional[List[Tuple[str, Tuple[int, int], Tuple[int, int]]]] = None,
) -> int:
    bounds = PERIOD4_BOUNDS if period4_bounds is None else period4_bounds
    for idx, (_, (y_start, m_start), (y_end, m_end)) in enumerate(bounds):
        start = (y_start, m_start)
        end = (y_end, m_end)
        if (year, month) < start:
            continue
        if (year, month) > end:
            continue
        return idx
    return -1


def _load_xy_with_year(
    path: str,
    task_cols: List[str],
    logC_col: Optional[str] = None,
    compute_product_cols: Optional[Tuple[str, str]] = None,
    compute_multiplier: float = 6.0,
    date_col: Optional[str] = None,
) -> Tuple[np.ndarray, np.ndarray, List[str], np.ndarray]:
    """Load compute x, tasks Y, and year vector from CSV for OLL splits.

    Returns (X, Y, tasks_kept, years), where years aligns with X rows.
    """
    with open(path, "r", newline="") as f:
        reader = _csv.DictReader(f)
        rows = list(reader)
        headers = list(reader.fieldnames or [])
    if not rows:
        raise RuntimeError(f"No rows in {path}")
    if date_col is None:
        # Attempt to auto-detect OLL date column
        if "Upload To Hub Date" in headers:
            date_col = "Upload To Hub Date"
        elif "date" in headers:
            date_col = "date"
        else:
            raise RuntimeError("Could not detect date column; pass a CSV with 'Upload To Hub Date' or 'date'.")
    x_list: List[float] = []
    Y_list: List[List[float]] = []
    years: List[int] = []
    tasks = [t for t in task_cols if t in headers]
    if not tasks:
        raise RuntimeError("None of the requested task columns are in the CSV")
    for r in rows:
        # compute
        if logC_col and logC_col in r and (r[logC_col] not in (None, "", "nan", "NaN")):
            try:
                lc = float(r[logC_col])
                x_val = float(math.exp(lc))
            except Exception:
                x_val = float("nan")
        elif compute_product_cols and all(c in r for c in compute_product_cols):
            v1 = r.get(compute_product_cols[0], None)
            v2 = r.get(compute_product_cols[1], None)
            try:
                t = float(v1) if v1 not in (None, "", "nan", "NaN") else float("nan")
            except Exception:
                t = float("nan")
            try:
                b = float(v2) if v2 not in (None, "", "nan", "NaN") else float("nan")
            except Exception:
                b = float("nan")
            x_val = float(compute_multiplier * t * b) if (np.isfinite(t) and np.isfinite(b)) else float("nan")
        else:
            x_val = float("nan")
        if not np.isfinite(x_val) or x_val <= 0:
            continue
        # parse year
        yr = _parse_year(r.get(date_col, ""))
        if yr is None:
            continue
        # tasks
        row_vals: List[float] = []
        ok_any = False
        for t in tasks:
            v = r.get(t, None)
            if v in (None, "", "nan", "NaN"):
                row_vals.append(float("nan"))
            else:
                try:
                    row_vals.append(float(v))
                    ok_any = True
                except Exception:
                    row_vals.append(float("nan"))
        if ok_any:
            x_list.append(x_val)
            Y_list.append(row_vals)
            years.append(int(yr))
    X = np.asarray(x_list, dtype=float)
    Y = np.asarray(Y_list, dtype=float)
    yrs = np.asarray(years, dtype=int)
    if Y.ndim != 2 or Y.shape[0] == 0:
        return X, np.zeros((0, 0), dtype=float), tasks, yrs
    Y = _normalize_task_matrix(Y)
    keep_row = np.any(np.isfinite(Y), axis=1)
    X = X[keep_row]
    Y = Y[keep_row]
    yrs = yrs[keep_row]
    keep_task = np.sum(np.isfinite(Y), axis=0) >= 3
    Y = Y[:, keep_task]
    kept = [t for t, k in zip(tasks, keep_task) if k]
    return X, Y, kept, yrs


def _load_xy_with_periods(
    path: str,
    task_cols: List[str],
    logC_col: Optional[str],
    compute_product_cols: Optional[Tuple[str, str]],
    compute_multiplier: float,
    date_col: Optional[str],
    period4_bounds: Optional[List[Tuple[str, Tuple[int, int], Tuple[int, int]]]] = None,
) -> Tuple[np.ndarray, np.ndarray, List[str], np.ndarray, np.ndarray]:
    with open(path, "r", newline="") as f:
        reader = _csv.DictReader(f)
        rows = list(reader)
        headers = list(reader.fieldnames or [])
    if not rows:
        raise RuntimeError(f"No rows in {path}")
    if date_col is None:
        if "Upload To Hub Date" in headers:
            date_col = "Upload To Hub Date"
        elif "date" in headers:
            date_col = "date"
        else:
            raise RuntimeError("Could not detect date column; pass --date_col explicitly.")
    tasks = [t for t in task_cols if t in headers]
    if not tasks:
        raise RuntimeError("None of the requested task columns are in the CSV")
    x_list: List[float] = []
    Y_list: List[List[float]] = []
    periods: List[int] = []
    mids: List[str] = []
    for r in rows:
        # Compute
        if logC_col and logC_col in r and (r[logC_col] not in (None, "", "nan", "NaN")):
            try:
                lc = float(r[logC_col])
                x_val = float(math.exp(lc))
            except Exception:
                x_val = float("nan")
        elif compute_product_cols and all(c in r for c in compute_product_cols):
            v1 = r.get(compute_product_cols[0], None)
            v2 = r.get(compute_product_cols[1], None)
            try:
                t = float(v1) if v1 not in (None, "", "nan", "NaN") else float("nan")
            except Exception:
                t = float("nan")
            try:
                b = float(v2) if v2 not in (None, "", "nan", "NaN") else float("nan")
            except Exception:
                b = float("nan")
            x_val = float(compute_multiplier * t * b) if (np.isfinite(t) and np.isfinite(b)) else float("nan")
        else:
            x_val = float("nan")
        if not np.isfinite(x_val) or x_val <= 0:
            continue
        # Period
        ym = _parse_year_month(r.get(date_col, ""))
        if ym is None:
            continue
        period_idx = _assign_period_index(*ym, period4_bounds=period4_bounds)
        if period_idx < 0:
            continue
        # Tasks
        row_vals: List[float] = []
        ok_any = False
        for t in tasks:
            v = r.get(t, None)
            if v in (None, "", "nan", "NaN"):
                row_vals.append(float("nan"))
            else:
                try:
                    row_vals.append(float(v))
                    ok_any = True
                except Exception:
                    row_vals.append(float("nan"))
        if not ok_any:
            continue
        x_list.append(x_val)
        Y_list.append(row_vals)
        periods.append(period_idx)
        mids.append(_read_model_id(r))
    X = np.asarray(x_list, dtype=float)
    Y = np.asarray(Y_list, dtype=float)
    per = np.asarray(periods, dtype=int)
    mid_arr = np.asarray(mids, dtype=object)
    if Y.ndim != 2 or Y.shape[0] == 0:
        return X, np.zeros((0, 0), dtype=float), [], per, mid_arr
    Y = _normalize_task_matrix(Y)
    keep_row = np.any(np.isfinite(Y), axis=1)
    X = X[keep_row]
    Y = Y[keep_row]
    per = per[keep_row]
    mid_arr = mid_arr[keep_row]
    keep_task = np.sum(np.isfinite(Y), axis=0) >= 3
    Y = Y[:, keep_task]
    kept = [t for t, k in zip(tasks, keep_task) if k]
    return X, Y, kept, per, mid_arr


def _plot_curves(
    out_dir: str,
    task: str,
    xs: np.ndarray,
    ys: np.ndarray,
    xs_curve: np.ndarray,
    y_curve: np.ndarray,
) -> None:
    try:
        import matplotlib.pyplot as plt  # type: ignore
        import matplotlib as mpl  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError("matplotlib is required for plotting") from e
    # Use serif font family for all text in the figure
    try:
        mpl.rcParams["font.family"] = "serif"
    except Exception:
        pass
    os.makedirs(out_dir, exist_ok=True)
    plt.figure(figsize=(7.0, 4.6))
    plt.scatter(xs, ys, s=12, alpha=0.20, color="#1f77b4", label="points")
    plt.plot(xs_curve, y_curve, color="#1f77b4", linewidth=2.4, label=f"Smooth τ={0.98:.2f}")
    plt.xscale("log")
    # Dynamic y-limits from data and curve with small padding
    y_min = float(np.nanmin([np.nanmin(ys) if ys.size else np.nan, np.nanmin(y_curve) if y_curve.size else np.nan]))
    y_max = float(np.nanmax([np.nanmax(ys) if ys.size else np.nan, np.nanmax(y_curve) if y_curve.size else np.nan]))
    if not np.isfinite(y_min) or not np.isfinite(y_max) or y_min == y_max:
        y_min, y_max = 0.0, 1.0
    pad = 0.02 * max(1e-6, (y_max - y_min))
    plt.ylim(y_min - pad, y_max + pad)
    plt.xlabel("Pretraining Compute (FLOPs)", fontweight='bold', fontsize=15)
    try:
        from skill_frontier.plotting.axis_formatting import (  # type: ignore
            apply_pretraining_compute_tick_multiplier,
        )

        apply_pretraining_compute_tick_multiplier(plt.gca())
    except Exception:
        pass
    plt.ylabel("Accuracy", fontweight='bold', fontsize=15)
    plt.title(f"Sigmoid Scaling — {task}", fontweight='bold', fontsize=18)
    leg = plt.legend(loc="best", fontsize=12)
    if leg and leg.get_title():
        leg.get_title().set_fontweight("bold")
    plt.tight_layout()
    base = os.path.join(out_dir, f"smooth_frontier__{task}")
    plt.savefig(base + ".png", dpi=300, bbox_inches="tight")
    plt.savefig(base + ".pdf", dpi=300, bbox_inches="tight")
    plt.close()


def _plot_curves_split(
    out_dir: str,
    task: str,
    group_points: Dict[str, Tuple[np.ndarray, np.ndarray]],
    curves: Dict[str, Tuple[np.ndarray, np.ndarray]],
) -> None:
    try:
        import matplotlib.pyplot as plt  # type: ignore
        import matplotlib as mpl  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError("matplotlib is required for plotting") from e
    # Use serif font family for all text in the figure
    try:
        mpl.rcParams["font.family"] = "serif"
    except Exception:
        pass
    os.makedirs(out_dir, exist_ok=True)
    plt.figure(figsize=(7.0, 4.6))
    # Consistent colors for groups across scatters and curves
    palette = ["#1f77b4", "#d62728", "#2ca02c", "#ff7f0e"]
    labels = list(group_points.keys())
    color_map: Dict[str, str] = {lab: palette[i % len(palette)] for i, lab in enumerate(labels)}
    # Scatter per group with distinct colors
    for lab in labels:
        xs, ys = group_points[lab]
        plt.scatter(xs, ys, s=10, alpha=0.20, color=color_map[lab])
    # Overlay curves using same colors and labeled legend entries
    for lab, (xc, yc) in curves.items():
        plt.plot(xc, yc, linewidth=2.2, label=lab, color=color_map.get(lab, "#1f77b4"))
    plt.xscale("log")
    # Dynamic y-limits from all points and curves
    all_y = []
    for lab in labels:
        _, ys = group_points[lab]
        if ys.size:
            all_y.append(np.nanmin(ys))
            all_y.append(np.nanmax(ys))
    for _, (_, yc) in curves.items():
        if yc.size:
            all_y.append(np.nanmin(yc))
            all_y.append(np.nanmax(yc))
    if all_y:
        y_min = float(np.nanmin(all_y))
        y_max = float(np.nanmax(all_y))
        if not np.isfinite(y_min) or not np.isfinite(y_max) or y_min == y_max:
            y_min, y_max = 0.0, 1.0
    else:
        y_min, y_max = 0.0, 1.0
    pad = 0.02 * max(1e-6, (y_max - y_min))
    plt.ylim(y_min - pad, y_max + pad)
    plt.xlabel("Pretraining Compute (FLOPs)", fontweight='bold', fontsize=15)
    try:
        from skill_frontier.plotting.axis_formatting import (  # type: ignore
            apply_pretraining_compute_tick_multiplier,
        )

        apply_pretraining_compute_tick_multiplier(plt.gca())
    except Exception:
        pass
    plt.ylabel("Accuracy", fontweight='bold', fontsize=15)
    plt.title(f"Sigmoid Scaling — {task}", fontweight='bold', fontsize=18)
    plt.legend(loc="best", fontsize=12)
    plt.tight_layout()
    base = os.path.join(out_dir, f"smooth_frontier_split__{task}")
    plt.savefig(base + ".png", dpi=300, bbox_inches="tight")
    plt.savefig(base + ".pdf", dpi=300, bbox_inches="tight")
    plt.close()


def _write_curve_csv(path: str, x: np.ndarray, y: np.ndarray, task: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        w = _csv.writer(f)
        w.writerow(["x", "y_hat", "task"])
        for xi, yi in zip(x, y):
            w.writerow([float(xi), float(yi), task])


def _get_plot_path(out_dir: str, task: str, *, suffix: str = "") -> str:
    """Return structured plot path base (without extension)."""
    task_clean = sanitize_task_name(task)
    plots_dir = os.path.join(out_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    if suffix:
        return os.path.join(plots_dir, f"{task_clean}_{suffix}")
    return os.path.join(plots_dir, task_clean)


def _get_curve_path(out_dir: str, task: str, *, k: Optional[int] = None) -> str:
    """Return structured curve csv path."""
    task_clean = sanitize_task_name(task)
    curves_dir = os.path.join(out_dir, "curves")
    os.makedirs(curves_dir, exist_ok=True)
    if k is not None:
        return os.path.join(curves_dir, f"{task_clean}_k{k}.csv")
    return os.path.join(curves_dir, f"{task_clean}.csv")


def main(argv: Optional[Sequence[str]] = None) -> None:
    p = argparse.ArgumentParser(description="Smooth 1D frontier (accuracy vs FLOPs) with high-quantile + monotone + PCHIP + FDH guard")
    p.add_argument("--csv", required=True, help="Input CSV path")
    p.add_argument("--logC_col", default=None, help="Column containing log compute (optional)")
    p.add_argument("--compute_product_cols", nargs=2, default=None, metavar=("TOKENS_COL", "PARAMS_COL"), help="Two columns whose product (times multiplier) is compute")
    p.add_argument("--compute_multiplier", type=float, default=6.0, help="Multiplier for product compute (default 6.0)")
    p.add_argument("--tasks", nargs="*", default=None, help="Task columns to smooth; if omitted, detect OLL Raw task columns")
    p.add_argument("--tau", type=float, default=0.98, help="High quantile per bin (0.95..0.99)")
    p.add_argument("--bins", type=int, default=120, help="Number of x-bins for quantiles")
    p.add_argument("--no_logx", action="store_true", help="Do not log10 the x-axis (use raw compute)")
    p.add_argument("--out_dir", default=os.path.join("outputs", "smooth_frontier"), help="Output directory for plots and CSVs")
    p.add_argument("--no_fdh_guard", action="store_true", help="Disable FDH guard (use only smoothed high-quantile curve)")
    p.add_argument("--sigmoid", action="store_true", help="Use parametric sigmoid frontier instead of binned quantile smoother")
    p.add_argument("--split_oll_years", action="store_true", help="Legacy flag: split by year (overrides --split_mode)")
    p.add_argument("--split_mode", choices=["none", "year", "period4"], default="period4", help="Split strategy: none, year (<2025 vs 2025), or period4 (default) with sequential train/validation periods")
    p.add_argument("--period4_train_mode", choices=["cumulative", "single_k"], default="cumulative", help="For period4: train on first k periods (cumulative) or only the k-th period (single_k)")
    # Optional manifests: filter rows by selected model IDs so that scatter shows only fitted points
    p.add_argument("--manifest_pre", default=None, help="Path to a text file with model IDs for the pre-2025 group (one per line)")
    p.add_argument("--manifest_2025", default=None, help="Path to a text file with model IDs for the 2025 group (one per line)")
    # Optional manifests for period4: base dir with k*/manifest__train.txt and manifest__val.txt
    p.add_argument("--period4_manifest_base", default=None, help="Base dir with k*/manifest__train.txt and manifest__val.txt to filter period4 panels")
    p.add_argument("--per_year", action="store_true", help="Additionally generate per-year frontiers (one curve per distinct year)")
    args = p.parse_args(argv)

    # Read CSV to arrays
    # Inspect header to auto-detect task columns if needed
    with open(args.csv, "r", newline="") as f:
        reader = _csv.reader(f)
        header = next(reader)
    tasks = args.tasks or _detect_oll_raw_tasks(header)
    if not tasks:
        raise RuntimeError("Could not infer tasks; pass --tasks explicitly")

    # Year split (legacy flag) or explicit --split_mode year
    if args.split_oll_years or args.split_mode == "year":
        # Determine date column from header
        with open(args.csv, "r", newline="") as f:
            r = _csv.reader(f)
            header = next(r)
        date_col = None
        if "Upload To Hub Date" in header:
            date_col = "Upload To Hub Date"
            scheme = "new"
        elif "date" in header:
            date_col = "date"
            scheme = "old"
        else:
            raise RuntimeError("--split_oll_years set but no recognized date column found")
        # Optional manifests for each group
        def _read_manifest(path: Optional[str]) -> Optional[Set[str]]:
            if path is None:
                return None
            s: Set[str] = set()
            try:
                with open(path, "r") as f:
                    for line in f:
                        v = line.strip()
                        if v:
                            s.add(v)
                return s
            except Exception:
                return None

        manifest_pre = _read_manifest(args.manifest_pre)
        manifest_2025 = _read_manifest(args.manifest_2025)

        # If any manifest is provided, load per group with filtering so that scatter exactly matches fitted points
        if manifest_pre is not None or manifest_2025 is not None:
            # Build group filters depending on scheme
            if scheme == "old":
                year_sets = {"2023": {2023}, "2024": {2024}}
            else:
                year_sets = {"< 2025": set(range(0, 2025)), "2025": {2025}}
            # For each task, gather points/curves across available group manifests and plot together
            for task in tasks:
                curves: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
                group_points: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
                for group_label, yset in year_sets.items():
                    man = manifest_pre if group_label in ("2023", "< 2025") else manifest_2025
                    if man is None:
                        continue
                    Xg, Yg, kept_g, _ = _load_xy_with_year_filtered(
                        args.csv,
                        task_cols=[task],
                        logC_col=args.logC_col,
                        compute_product_cols=tuple(args.compute_product_cols) if args.compute_product_cols else None,
                        compute_multiplier=float(args.compute_multiplier),
                        date_col=date_col,
                        manifest=man,
                        year_filter=yset,
                    )
                    if Yg.size == 0:
                        continue
                    yv = Yg[:, 0]
                    m = np.isfinite(Xg) & np.isfinite(yv)
                    if np.sum(m) < 3:
                        continue
                    if args.sigmoid:
                        xs_curve, y_curve = fit_sigmoid_frontier(Xg[m], yv[m], tau=float(args.tau), use_log10_x=not args.no_logx)
                    else:
                        xs_curve, y_curve = smooth_frontier(Xg[m], yv[m], tau=float(args.tau), bins=int(args.bins), use_log10_x=not args.no_logx, guard_fdh=(not args.no_fdh_guard))
                    group_points[group_label] = (Xg[m], yv[m])
                    curves[group_label] = (xs_curve, y_curve)
                    _write_curve_csv(os.path.join(args.out_dir, f"smooth_frontier__{task}__{group_label.replace(' ','_')}.csv"), xs_curve, y_curve, task)
                if curves:
                    _plot_curves_split(args.out_dir, task, group_points, curves)
        else:
            # No manifests provided: default behavior (all points in each group)
            X, Y, kept, yrs = _load_xy_with_year(
                args.csv,
                task_cols=tasks,
                logC_col=args.logC_col,
                compute_product_cols=tuple(args.compute_product_cols) if args.compute_product_cols else None,
                compute_multiplier=float(args.compute_multiplier),
                date_col=date_col,
            )
            if Y.size == 0:
                raise RuntimeError("No usable data rows found for the requested tasks under the year split")
            if scheme == "old":
                groups = {"2023": (yrs == 2023), "2024": (yrs == 2024)}
            else:
                groups = {"< 2025": (yrs < 2025), "2025": (yrs == 2025)}
            for j, task in enumerate(kept):
                y_all = Y[:, j]
                mask_all = np.isfinite(X) & np.isfinite(y_all)
                curves: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
                group_points: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
                for label, gmask in groups.items():
                    m = mask_all & gmask
                    if np.sum(m) < 3:
                        continue
                    group_points[label] = (X[m], y_all[m])
                    if args.sigmoid:
                        xs_curve, y_curve = fit_sigmoid_frontier(X[m], y_all[m], tau=float(args.tau), use_log10_x=not args.no_logx)
                    else:
                        xs_curve, y_curve = smooth_frontier(X[m], y_all[m], tau=float(args.tau), bins=int(args.bins), use_log10_x=not args.no_logx, guard_fdh=(not args.no_fdh_guard))
                    curves[label] = (xs_curve, y_curve)
                    _write_curve_csv(os.path.join(args.out_dir, f"smooth_frontier__{task}__{label.replace(' ','_')}.csv"), xs_curve, y_curve, task)
                if curves:
                    _plot_curves_split(args.out_dir, task, group_points, curves)

        # Optionally, also produce per-year frontiers (one curve per year)
        if args.per_year:
            # Identify sorted unique years present
            uniq_years = sorted({int(y) for y in yrs.tolist()})
            year_groups = {str(y): (yrs == y) for y in uniq_years}
            for j, task in enumerate(kept):
                y_all = Y[:, j]
                mask_all = np.isfinite(X) & np.isfinite(y_all)
                curves_y: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
                pts_y: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
                for label, gmask in year_groups.items():
                    m = mask_all & gmask
                    if np.sum(m) < 3:
                        continue
                    pts_y[label] = (X[m], y_all[m])
                    if args.sigmoid:
                        xs_curve, y_curve = fit_sigmoid_frontier(X[m], y_all[m], tau=float(args.tau), use_log10_x=not args.no_logx)
                    else:
                        xs_curve, y_curve = smooth_frontier(X[m], y_all[m], tau=float(args.tau), bins=int(args.bins), use_log10_x=not args.no_logx, guard_fdh=(not args.no_fdh_guard))
                    curves_y[label] = (xs_curve, y_curve)
                    _write_curve_csv(os.path.join(args.out_dir, f"smooth_frontier__{task}__YEAR_{label}.csv"), xs_curve, y_curve, task)
                if curves_y:
                    # Save with a distinct basename to avoid confusion with the two-subset figure
                    _plot_curves_split(args.out_dir, f"{task} (per-year)", pts_y, curves_y)
    # Four-period sequential train→val triptychs
    elif args.split_mode == "period4":
        # Default output folder for period4 if user didn't override
        default_out = os.path.join("outputs", "sigmoid", "period4", "cumulative")
        default_out_single = os.path.join("outputs", "sigmoid", "period4", "single_k")
        if args.out_dir == os.path.join("outputs", "smooth_frontier"):
            args.out_dir = default_out if args.period4_train_mode == "cumulative" else default_out_single

        # Determine date column from header
        with open(args.csv, "r", newline="") as f:
            r = _csv.reader(f)
            header = next(r)
        date_col = None
        if "Upload To Hub Date" in header:
            date_col = "Upload To Hub Date"
        elif "date" in header:
            date_col = "date"
        else:
            raise RuntimeError("--split_mode period4 but no recognized date column found")

        (
            period4_scheme,
            period4_bounds,
            period4_splits,
            period4_splits_single,
            period_symbol,
        ) = _resolve_period4_runtime_config(date_col)
        if date_col == "date" and _PERIOD4_SCHEME not in {"oll_old", "old", "old_oll"}:
            print("[period4] Detected legacy `date` schema; auto-using `oll_old` period boundaries/splits.")

        X, Y, kept, per, mids = _load_xy_with_periods(
            args.csv,
            task_cols=tasks,
            logC_col=args.logC_col,
            compute_product_cols=tuple(args.compute_product_cols) if args.compute_product_cols else None,
            compute_multiplier=float(args.compute_multiplier),
            date_col=date_col,
            period4_bounds=period4_bounds,
        )
        if Y.size == 0:
            raise RuntimeError("No usable data rows found for the requested tasks (period4)")

        # Build mapping: label -> index and index -> label
        label_to_idx: Dict[str, int] = {lab: i for i, (lab, _, _) in enumerate(period4_bounds)}
        idx_to_label: Dict[int, str] = {i: lab for i, (lab, _, _) in enumerate(period4_bounds)}

        # Helper: render one triptych figure per task.
        def _plot_period4_triptych(out_dir: str, task: str, panels: List[Dict[str, np.ndarray]]) -> None:
            try:
                import matplotlib.pyplot as plt  # type: ignore
                import matplotlib as mpl  # type: ignore
                from matplotlib.lines import Line2D  # type: ignore
                from skill_frontier.plotting.axis_formatting import apply_pretraining_compute_tick_multiplier  # type: ignore
                if period4_scheme == "oll_old":
                    from skill_frontier.plotting.configs import frontier_period4_triptych_legacy as period4_triptych_cfg  # type: ignore
                else:
                    from skill_frontier.plotting.configs import frontier_period4_triptych as period4_triptych_cfg  # type: ignore
            except Exception as e:  # pragma: no cover
                raise RuntimeError("matplotlib and plotting configs are required for plotting") from e

            os.makedirs(out_dir, exist_ok=True)
            with mpl.rc_context(getattr(period4_triptych_cfg, "RCPARAMS", {})):
                fig, axes = plt.subplots(1, 3, figsize=period4_triptych_cfg.FIGSIZE, dpi=300, sharey=True)
                fig.patch.set_facecolor("white")
                try:
                    fig.subplots_adjust(**period4_triptych_cfg.SUBPLOTS_ADJUST)
                except Exception:
                    fig.subplots_adjust(wspace=period4_triptych_cfg.PANEL_WSPACE)

                palette = period4_triptych_cfg.PALETTE
                y_vals: List[float] = []
                for p in panels:
                    for key in ("train_y", "val_y", "curve_y", "curve_val_y"):
                        arr = p.get(key)
                        if arr is not None and arr.size:
                            y_vals.append(float(np.nanmin(arr)))
                            y_vals.append(float(np.nanmax(arr)))
                y_min, y_max = (0.0, 1.0)
                if y_vals:
                    y_min = float(np.nanmin(y_vals))
                    y_max = float(np.nanmax(y_vals))
                    if not (np.isfinite(y_min) and np.isfinite(y_max)) or y_min == y_max:
                        y_min, y_max = 0.0, 1.0
                pad = 0.02 * max(1e-6, (y_max - y_min))

                for idx, (ax, panel) in enumerate(zip(axes, panels), start=1):
                    ax.set_facecolor("white")
                    ax.set_axisbelow(True)

                    ax.scatter(
                        panel["train_x"],
                        panel["train_y"],
                        s=period4_triptych_cfg.SCATTER_SIZE,
                        alpha=period4_triptych_cfg.TRAIN_ALPHA,
                        color=palette["train"],
                        edgecolors=period4_triptych_cfg.MARKER_EDGECOLOR,
                        linewidths=period4_triptych_cfg.MARKER_EDGE_LINEWIDTH,
                        rasterized=True,
                        marker=period4_triptych_cfg.TRAIN_MARKER,
                        label="_nolegend_",
                        zorder=period4_triptych_cfg.TRAIN_ZORDER,
                    )
                    ax.scatter(
                        panel["val_x"],
                        panel["val_y"],
                        s=period4_triptych_cfg.SCATTER_SIZE,
                        alpha=period4_triptych_cfg.VAL_ALPHA,
                        color=palette["val"],
                        edgecolors=period4_triptych_cfg.MARKER_EDGECOLOR,
                        linewidths=period4_triptych_cfg.MARKER_EDGE_LINEWIDTH,
                        rasterized=True,
                        marker=period4_triptych_cfg.VAL_MARKER,
                        label="_nolegend_",
                        zorder=period4_triptych_cfg.VAL_ZORDER,
                    )

                    ax.plot(
                        panel["curve_x"],
                        panel["curve_y"],
                        color=palette["curve"],
                        linewidth=period4_triptych_cfg.CURVE_LINEWIDTH,
                        alpha=period4_triptych_cfg.CURVE_ALPHA,
                        linestyle="-",
                        zorder=5,
                    )
                    if panel.get("curve_val_x") is not None and panel.get("curve_val_y") is not None and np.asarray(panel["curve_val_x"]).size:
                        ax.plot(
                            panel["curve_val_x"],
                            panel["curve_val_y"],
                            color=palette["curve_val"],
                            linewidth=period4_triptych_cfg.CURVE_LINEWIDTH,
                            alpha=period4_triptych_cfg.CURVE_ALPHA,
                            linestyle="--",
                            zorder=5,
                        )

                    ax.set_xscale("log")
                    if period4_scheme == "oll_old":
                        ax.set_xlim(1e21, 1e25)
                        ax.set_xticks([1e21, 1e22, 1e23, 1e24, 1e25])
                    else:
                        ax.set_xlim(1e0, 1e4)
                        ax.set_xticks([1e0, 1e1, 1e2, 1e3, 1e4])
                        apply_pretraining_compute_tick_multiplier(ax, require_label_match=False)
                    ax.set_ylim(y_min - pad, y_max + pad)

                    ax.grid(
                        True,
                        color=period4_triptych_cfg.GRID_MAJOR_COLOR,
                        alpha=period4_triptych_cfg.GRID_MAJOR_ALPHA,
                        linestyle=period4_triptych_cfg.GRID_MAJOR_LINESTYLE,
                        linewidth=period4_triptych_cfg.GRID_MAJOR_LINEWIDTH,
                        zorder=0,
                    )
                    ax.spines["top"].set_visible(False)
                    ax.spines["right"].set_visible(False)
                    ax.spines["left"].set_linewidth(period4_triptych_cfg.SPINE_LINEWIDTH)
                    ax.spines["bottom"].set_linewidth(period4_triptych_cfg.SPINE_LINEWIDTH)
                    ax.spines["left"].set_color(period4_triptych_cfg.SPINE_COLOR)
                    ax.spines["bottom"].set_color(period4_triptych_cfg.SPINE_COLOR)

                    ax.tick_params(
                        axis="both",
                        which="major",
                        labelsize=period4_triptych_cfg.TICK_LABELSIZE,
                        length=period4_triptych_cfg.TICK_LENGTH,
                        width=period4_triptych_cfg.TICK_WIDTH,
                        direction=period4_triptych_cfg.TICK_DIRECTION,
                        color=period4_triptych_cfg.SPINE_COLOR,
                    )
                    ax.tick_params(
                        axis="both",
                        which="minor",
                        labelsize=period4_triptych_cfg.TICK_LABELSIZE,
                        length=period4_triptych_cfg.TICK_MINOR_LENGTH,
                        width=period4_triptych_cfg.TICK_MINOR_WIDTH,
                        direction=period4_triptych_cfg.TICK_DIRECTION,
                        color=period4_triptych_cfg.SPINE_COLOR,
                    )
                    if idx != 1:
                        ax.tick_params(axis="y", which="both", labelleft=False)

                    ax.text(
                        0.5,
                        0.93,
                        f"${period_symbol} = {idx}$",
                        transform=ax.transAxes,
                        ha="center",
                        va="top",
                        fontsize=period4_triptych_cfg.BADGE_FONTSIZE,
                        fontweight=period4_triptych_cfg.BADGE_WEIGHT,
                        bbox=dict(
                            boxstyle=period4_triptych_cfg.BADGE_BOXSTYLE,
                            facecolor=period4_triptych_cfg.BADGE_BOX_FACE,
                            edgecolor=period4_triptych_cfg.BADGE_BOX_EDGE,
                            alpha=period4_triptych_cfg.BADGE_BOX_ALPHA,
                        ),
                    )

                axes[0].set_ylabel("Accuracy", fontweight="bold", fontsize=period4_triptych_cfg.Y_LABEL_FONTSIZE)
                fig.text(
                    0.5,
                    0.02,
                    "Pretraining Compute (FLOPs)",
                    ha="center",
                    va="center",
                    fontsize=period4_triptych_cfg.X_LABEL_FONTSIZE,
                    fontweight="bold",
                )

                legend_handles = [
                    Line2D(
                        [0],
                        [0],
                        marker="o",
                        linestyle="None",
                        markersize=6.5,
                        markerfacecolor=palette["train"],
                        markeredgecolor=period4_triptych_cfg.MARKER_EDGECOLOR,
                        markeredgewidth=period4_triptych_cfg.MARKER_EDGE_LINEWIDTH,
                        label="Train",
                    ),
                    Line2D(
                        [0],
                        [0],
                        marker="o",
                        linestyle="None",
                        markersize=6.5,
                        markerfacecolor=palette["val"],
                        markeredgecolor=period4_triptych_cfg.MARKER_EDGECOLOR,
                        markeredgewidth=period4_triptych_cfg.MARKER_EDGE_LINEWIDTH,
                        label="Val",
                    ),
                    Line2D([0], [0], color=palette["curve"], linewidth=period4_triptych_cfg.CURVE_LINEWIDTH, linestyle="-", label="Fit (Train)"),
                    Line2D([0], [0], color=palette["curve_val"], linewidth=period4_triptych_cfg.CURVE_LINEWIDTH, linestyle="--", label="Fit (Val)"),
                ]
                fig.legend(
                    handles=legend_handles,
                    loc="upper center",
                    bbox_to_anchor=(0.5, 0.985),
                    ncol=4,
                    frameon=False,
                    columnspacing=1.2,
                    handletextpad=0.6,
                    handlelength=2.2,
                    borderaxespad=0.0,
                )

                base = _get_plot_path(out_dir, task, suffix="period4")
                fig.savefig(base + ".png", dpi=300, bbox_inches="tight", pad_inches=0.05, facecolor="white", edgecolor="none")
                fig.savefig(base + ".pdf", dpi=300, bbox_inches="tight", pad_inches=0.05, facecolor="white", edgecolor="none")
                plt.close(fig)

        # Choose split spec set based on training mode
        split_specs = period4_splits if args.period4_train_mode == "cumulative" else period4_splits_single

        # Build panels and write CSVs
        for j, task in enumerate(kept):
            y_all = Y[:, j]
            mask_all = np.isfinite(X) & np.isfinite(y_all) & (per >= 0)
            panels: List[Dict[str, np.ndarray]] = []
            for ki, spec in enumerate(split_specs, start=1):
                train_inds = [label_to_idx[lbl] for lbl in spec["train_labels"] if lbl in label_to_idx]
                val_ind = label_to_idx.get(spec["val_label"], None)
                if val_ind is None:
                    continue
                m_train = mask_all & np.isin(per, np.array(train_inds, dtype=int))
                m_val = mask_all & (per == val_ind)
                # Optional: apply period4 manifests per k
                if args.period4_manifest_base:
                    kdir = os.path.join(args.period4_manifest_base, f"k{ki}")
                    def _load_set(p: str) -> Optional[Set[str]]:
                        try:
                            s: Set[str] = set()
                            if not os.path.isfile(p):
                                return None
                            with open(p, "r") as f:
                                for line in f:
                                    m = line.strip()
                                    if m:
                                        s.add(m)
                            return s
                        except Exception:
                            return None
                    man_tr = _load_set(os.path.join(kdir, "manifest__train.txt"))
                    # For budgeted visualizations we restrict training points to the
                    # manifest but keep all validation points so that the next-period
                    # frontier reflects the full period rather than only budgeted
                    # samples.
                    if man_tr is not None:
                        m_train = m_train & np.isin(mids, np.array(list(man_tr), dtype=object))
                if np.sum(m_train) < 3:
                    # Skip panel if insufficient training points
                    continue
                # Fit on train only
                if args.sigmoid:
                    xs_curve, y_curve = fit_sigmoid_frontier(X[m_train], y_all[m_train], tau=float(args.tau), use_log10_x=not args.no_logx)
                    xs_curve_val: Optional[np.ndarray]
                    y_curve_val: Optional[np.ndarray]
                    if np.sum(m_val) >= 3:
                        xs_curve_val, y_curve_val = fit_sigmoid_frontier(X[m_val], y_all[m_val], tau=float(args.tau), use_log10_x=not args.no_logx)
                    else:
                        xs_curve_val, y_curve_val = None, None
                else:
                    xs_curve, y_curve = smooth_frontier(X[m_train], y_all[m_train], tau=float(args.tau), bins=int(args.bins), use_log10_x=not args.no_logx, guard_fdh=(not args.no_fdh_guard))
                    if np.sum(m_val) >= 3:
                        xs_curve_val, y_curve_val = smooth_frontier(X[m_val], y_all[m_val], tau=float(args.tau), bins=int(args.bins), use_log10_x=not args.no_logx, guard_fdh=(not args.no_fdh_guard))
                    else:
                        xs_curve_val, y_curve_val = None, None
                panels.append({
                    "train_x": X[m_train],
                    "train_y": y_all[m_train],
                    "val_x": X[m_val],
                    "val_y": y_all[m_val],
                    "curve_x": xs_curve,
                    "curve_y": y_curve,
                    "curve_val_x": xs_curve_val if xs_curve_val is not None else np.array([]),
                    "curve_val_y": y_curve_val if y_curve_val is not None else np.array([]),
                    # no per-subplot titles per user request
                })
                # Save per-k CSV of the curve
                _write_curve_csv(_get_curve_path(args.out_dir, task, k=ki), xs_curve, y_curve, task)
            if panels:
                _plot_period4_triptych(args.out_dir, task, panels)

    else:
        x, Y, kept = _load_xy_from_csv(
            args.csv,
            task_cols=tasks,
            logC_col=args.logC_col,
            compute_product_cols=tuple(args.compute_product_cols) if args.compute_product_cols else None,
            compute_multiplier=float(args.compute_multiplier),
        )
        if Y.size == 0:
            raise RuntimeError("No usable data rows found for the requested tasks")
        for j, task in enumerate(kept):
            y = Y[:, j]
            if args.sigmoid:
                xs_curve, y_curve = fit_sigmoid_frontier(x, y, tau=float(args.tau), use_log10_x=not args.no_logx)
            else:
                xs_curve, y_curve = smooth_frontier(x, y, tau=float(args.tau), bins=int(args.bins), use_log10_x=not args.no_logx, guard_fdh=(not args.no_fdh_guard))
            mask = np.isfinite(x) & np.isfinite(y)
            _plot_curves(args.out_dir, task, x[mask], y[mask], xs_curve, y_curve)
            _write_curve_csv(os.path.join(args.out_dir, f"smooth_frontier__{task}.csv"), xs_curve, y_curve, task)


if __name__ == "__main__":  # pragma: no cover
    main()

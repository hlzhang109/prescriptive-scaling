#!/usr/bin/env python3
"""
Sweep alpha% budgets and plot OOS MAE vs alpha for 2-period and 4-period (single_k).

For each alpha in the provided list:
  - 2-period: build manifests in-process (alpha% of param count per group),
              evaluate with scripts/evaluate/sigmoid_binned_mae.py (pre2025_vs_2025).
  - 4-period: build manifests in-process (alpha% of param count per group/k),
              evaluate with scripts/evaluate/sigmoid_binned_mae.py (period4 single_k).

Outputs:
  - Plots under --out_dir showing task-wise OOS coverage error vs alpha for
    (a) 2-period and (b) 4-period (averaged over k=1..3).
  - Intermediate artifacts under structured subfolders per alpha.
"""

from __future__ import annotations

import argparse
import csv as _csv
import logging
import os
import subprocess
import sys
from typing import Dict, List, Tuple

import numpy as np

# Import bootstrap from `scripts/` by putting the scripts dir on sys.path.
SCRIPTS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)
from _bootstrap import ensure_repo_root_on_path  # type: ignore

REPO_ROOT = ensure_repo_root_on_path(__file__)

from skill_frontier.core.budget_design import FrontierParams, Weighting, design_budget_only  # type: ignore
from skill_frontier.core.sigmoid import (
    PERIOD4_BOUNDS,
    PERIOD4_SPLITS_SINGLE,
    _assign_period_index,
)  # type: ignore
from skill_frontier.io.csv_utils import (  # type: ignore
    read_csv_rows,
    compute_flops,
    detect_date_col,
    detect_oll_raw_tasks,
    maybe_scale_task_values,
    parse_date,
    parse_year_month,
    extract_model_id,
    sanitize_name,
)
from skill_frontier.io.manifest_utils import read_manifest  # type: ignore
from skill_frontier.evaluation.common import fit_sigmoid_predictor, interpolate_curve  # type: ignore
from skill_frontier.plotting.configs import mpl_rc as mpl_rc_cfg  # type: ignore
from skill_frontier.plotting.configs import sweep_alpha as sweep_alpha_cfg  # type: ignore

LOGGER = logging.getLogger(__name__)


def _read_task_mae(eval_dir: str) -> Dict[str, Tuple[float, float]]:
    """Read per-task IS and OOS coverage error from summary CSVs in eval_dir."""
    out: Dict[str, Tuple[float, float]] = {}
    summaries_dir = os.path.join(eval_dir, "summaries")
    scan_dir = summaries_dir if os.path.isdir(summaries_dir) else eval_dir
    for fn in os.listdir(scan_dir):
        if not (fn.endswith("__summary.csv") or fn.endswith("_summary.csv")):
            continue
        path = os.path.join(scan_dir, fn)
        with open(path, "r", newline="") as f:
            r = _csv.DictReader(f)
            row = next(r)
        task = row["task"]
        try:
            is_macro = float(row["MAE_IS_macro"]) if row["MAE_IS_macro"] != "" else float("nan")
        except Exception:
            is_macro = float("nan")
        try:
            oos_macro = float(row["MAE_OOS_macro"]) if row["MAE_OOS_macro"] != "" else float("nan")
        except Exception:
            oos_macro = float("nan")
        out[task] = (is_macro, oos_macro)
    return out


def _read_task_mae_period4(base: str) -> Dict[str, Tuple[float, float]]:
    """Average IS/OOS coverage error over k=1..3 for each task under a period4 eval base."""
    is_accum: Dict[str, List[float]] = {}
    oos_accum: Dict[str, List[float]] = {}
    for k in (1, 2, 3):
        kdir = os.path.join(base, f"k{k}")
        if not os.path.isdir(kdir):
            continue
        vals = _read_task_mae(kdir)
        for t, (is_v, oos_v) in vals.items():
            is_accum.setdefault(t, []).append(is_v)
            oos_accum.setdefault(t, []).append(oos_v)
    out: Dict[str, Tuple[float, float]] = {}
    for t in set(list(is_accum.keys()) + list(oos_accum.keys())):
        a_is = np.array(is_accum.get(t, []), float)
        a_oos = np.array(oos_accum.get(t, []), float)
        is_mean = float(np.nanmean(a_is)) if a_is.size else float("nan")
        oos_mean = float(np.nanmean(a_oos)) if a_oos.size else float("nan")
        out[t] = (is_mean, oos_mean)
    return out


def _read_task_pinball(kdir: str) -> Dict[str, Tuple[float, float]]:
    """Read per-task IS and OOS pinball loss from pinball_sigmoid CSVs in kdir."""
    out: Dict[str, Tuple[float, float]] = {}
    if not os.path.isdir(kdir):
        return out
    for fn in os.listdir(kdir):
        if not fn.endswith("__pinball_sigmoid.csv"):
            continue
        path = os.path.join(kdir, fn)
        try:
            rows = list(_csv.DictReader(open(path, "r", newline="")))
        except Exception:
            continue
        train = next((float(r["L_sigmoid"]) for r in rows if r.get("split") == "train"), float("nan"))
        val = next((float(r["L_sigmoid"]) for r in rows if r.get("split") == "val"), float("nan"))
        task = fn.replace("__pinball_sigmoid.csv", "")
        out[task] = (train, val)
    return out


def _read_task_pinball_period4(base: str) -> Dict[str, Tuple[float, float]]:
    """Average IS/OOS pinball over k=1..3 for each task under a period4 eval base."""
    is_accum: Dict[str, List[float]] = {}
    oos_accum: Dict[str, List[float]] = {}
    for k in (1, 2, 3):
        kdir = os.path.join(base, f"k{k}")
        vals = _read_task_pinball(kdir)
        for t, (is_v, oos_v) in vals.items():
            is_accum.setdefault(t, []).append(is_v)
            oos_accum.setdefault(t, []).append(oos_v)
    out: Dict[str, Tuple[float, float]] = {}
    for t in set(list(is_accum.keys()) + list(oos_accum.keys())):
        a_is = np.array(is_accum.get(t, []), float)
        a_oos = np.array(oos_accum.get(t, []), float)
        is_mean = float(np.nanmean(a_is)) if a_is.size else float("nan")
        oos_mean = float(np.nanmean(a_oos)) if a_oos.size else float("nan")
        out[t] = (is_mean, oos_mean)
    return out


def _design_year_split(
    rows: List[dict],
    headers: List[str],
    compute_product_cols: Tuple[str, str],
    compute_multiplier: float,
    alpha_percent: float,
    out_dir: str,
    y0: float = 0.20,
    L: float = 0.75,
    a: float = -9.0,
    b: float = 1.0,
    m: int = 100,
    exchange_passes: int = 0,
    objective: str = "d_optimal",
    balance_lambda: float = 0.0,
    num_bins: int = 10,
    min_bin_size: int = 1,
) -> Tuple[str, str]:
    """Select manifests for pre-2025 and 2025 groups using size (#Params B) as cost."""
    date_col = detect_date_col(headers)
    theta0 = FrontierParams(y0=y0, L=L, a=a, b=max(b, 1e-6))
    w = Weighting(mode="constant", m=m)

    mids: List[str] = []
    z_all: List[float] = []
    size_all: List[float] = []
    year_all: List[int] = []
    for r in rows:
        mid = extract_model_id(r)
        if not mid:
            continue
        C = compute_flops(
            r,
            headers,
            logC_col=None,
            prod_cols=compute_product_cols,
            mult=float(compute_multiplier),
        )
        if not (np.isfinite(C) and C > 0):
            continue
        size_raw = r.get("#Params (B)", None)
        try:
            sz = float(size_raw) if size_raw not in (None, "", "nan", "NaN") else float("nan")
        except Exception:
            sz = float("nan")
        yr = parse_date(r.get(date_col, "")) if date_col else None
        mids.append(mid)
        z_all.append(float(np.log10(C)))
        size_all.append(float(sz))
        year_all.append(int(yr) if yr is not None else -1)

    z_arr = np.asarray(z_all, float)
    size_arr = np.asarray(size_all, float)
    year_arr = np.asarray(year_all, int)

    def _select(mask: np.ndarray) -> List[int]:
        if not mask.any():
            return []
        sz = size_arr[mask]
        z = z_arr[mask]
        sum_sz = float(np.sum(sz[np.isfinite(sz) & (sz > 0)]))
        U = (alpha_percent / 100.0) * sum_sz
        if U <= 0:
            return []
        idxs = design_budget_only(
            z,
            sz,
            U,
            theta0,
            weighting=w,
            seed_c=1.543,
            exchange_passes=int(max(0, exchange_passes)),
            objective=str(objective),
            balance_lambda=float(balance_lambda),
            num_bins=int(max(1, num_bins)),
            min_bin_size=int(max(1, min_bin_size)),
        )
        base_idx = np.nonzero(mask)[0]
        return [int(base_idx[i]) for i in idxs]

    sel_pre = _select(year_arr < 2025)
    sel_2025 = _select(year_arr == 2025)

    os.makedirs(out_dir, exist_ok=True)
    man_pre = os.path.join(out_dir, "manifest__< 2025.txt")
    man_2025 = os.path.join(out_dir, "manifest__2025.txt")
    for path, indices in ((man_pre, sel_pre), (man_2025, sel_2025)):
        with open(path, "w") as f:
            for i in indices:
                f.write(f"{mids[i]}\n")
    return man_pre, man_2025


def _design_period4_singlek(
    rows: List[dict],
    headers: List[str],
    compute_product_cols: Tuple[str, str],
    compute_multiplier: float,
    alpha_percent: float,
    out_base: str,
    y0: float = 0.20,
    L: float = 0.75,
    a: float = -9.0,
    b: float = 1.0,
    m: int = 100,
    exchange_passes: int = 0,
    objective: str = "d_optimal",
    balance_lambda: float = 0.0,
    num_bins: int = 10,
    min_bin_size: int = 1,
) -> str:
    """Select manifests for period4 single_k splits using size (#Params B) as cost."""
    date_col = detect_date_col(headers)
    theta0 = FrontierParams(y0=y0, L=L, a=a, b=max(b, 1e-6))
    w = Weighting(mode="constant", m=m)

    mids: List[str] = []
    z_all: List[float] = []
    size_all: List[float] = []
    per_all: List[int] = []

    def _assign_period(year: int, month: int) -> int:
        for i, bound in enumerate(PERIOD4_BOUNDS):
            if len(bound) == 3:
                _, (y_lo, m_lo), (y_hi, m_hi) = bound
            else:
                y_lo, m_lo, y_hi, m_hi = bound
            if (year, month) >= (y_lo, m_lo) and (year, month) <= (y_hi, m_hi):
                return i
        return -1

    for r in rows:
        mid = extract_model_id(r)
        if not mid:
            continue
        C = compute_flops(
            r,
            headers,
            logC_col=None,
            prod_cols=compute_product_cols,
            mult=float(compute_multiplier),
        )
        if not (np.isfinite(C) and C > 0):
            continue
        ym = parse_year_month(r.get(date_col, "")) if date_col else None
        if ym is None:
            continue
        pid = _assign_period(*ym)
        if pid < 0:
            continue
        size_raw = r.get("#Params (B)", None)
        try:
            sz = float(size_raw) if size_raw not in (None, "", "nan", "NaN") else float("nan")
        except Exception:
            sz = float("nan")
        mids.append(mid)
        z_all.append(float(np.log10(C)))
        size_all.append(float(sz))
        per_all.append(pid)

    z_arr = np.asarray(z_all, float)
    size_arr = np.asarray(size_all, float)
    per_arr = np.asarray(per_all, int)

    os.makedirs(out_base, exist_ok=True)

    for k_idx, split in enumerate(PERIOD4_SPLITS_SINGLE, start=1):
        out_dir = os.path.join(out_base, f"k{k_idx}")
        os.makedirs(out_dir, exist_ok=True)
        train_labels = split.get("train_labels", [])
        val_label = split.get("val_label", None)
        label_to_idx = {}
        for i, bound in enumerate(PERIOD4_BOUNDS):
            if len(bound) == 3:
                label = bound[0]
            else:
                label = f"k{i+1}"
            label_to_idx[label] = i
        train_idx = [label_to_idx[l] for l in train_labels if l in label_to_idx]
        val_idx = [label_to_idx[val_label]] if val_label in label_to_idx else []

        mask_tr = np.isin(per_arr, np.array(train_idx, dtype=int))
        mask_val = np.isin(per_arr, np.array(val_idx, dtype=int))

        def _select(mask: np.ndarray) -> List[int]:
            if not mask.any():
                return []
            z = z_arr[mask]
            sz = size_arr[mask]
            sum_sz = float(np.sum(sz[np.isfinite(sz) & (sz > 0)]))
            U = (alpha_percent / 100.0) * sum_sz
            if U <= 0:
                return []
            idxs = design_budget_only(
                z,
                sz,
                U,
                theta0,
                weighting=w,
                seed_c=1.543,
                exchange_passes=int(max(0, exchange_passes)),
                objective=str(objective),
                balance_lambda=float(balance_lambda),
                num_bins=int(max(1, num_bins)),
                min_bin_size=int(max(1, min_bin_size)),
            )
            base_idx = np.nonzero(mask)[0]
            return [int(base_idx[i]) for i in idxs]

        sel_tr = _select(mask_tr)
        sel_val = _select(mask_val)

        man_tr = os.path.join(out_dir, "manifest__train.txt")
        man_val = os.path.join(out_dir, "manifest__val.txt")
        for path, indices in ((man_tr, sel_tr), (man_val, sel_val)):
            with open(path, "w") as f:
                for i in indices:
                    f.write(f"{mids[i]}\n")
    return out_base


def _pinball_loss_mean(y_true: np.ndarray, y_pred: np.ndarray, tau: float) -> float:
    """Mean smoothed pinball loss (same smoothing as baseline scripts)."""
    if y_true.size == 0:
        return float("nan")
    r = np.asarray(y_true, float) - np.asarray(y_pred, float)
    loss = (np.logaddexp(0.0, 50.0 * r) / 50.0) + (tau - 1.0) * r
    return float(np.mean(loss))


def _build_model_records(
    rows: List[dict],
    headers: List[str],
    tasks: List[str],
    compute_product_cols: Tuple[str, str],
    compute_multiplier: float,
) -> List[dict]:
    """Precompute per-model compute (FLOPs) and task scores for reuse."""
    recs: List[dict] = []
    date_col = detect_date_col(headers)
    for r in rows:
        mid = extract_model_id(r)
        if not mid:
            continue
        C = compute_flops(
            r,
            headers,
            logC_col=None,
            prod_cols=compute_product_cols,
            mult=float(compute_multiplier),
        )
        if not (np.isfinite(C) and C > 0):
            continue
        task_vals: Dict[str, float] = {}
        for t in tasks:
            try:
                v = float(r.get(t, "nan"))
            except Exception:
                v = float("nan")
            task_vals[t] = v
        # Period index for period4 single_k
        per_idx = -1
        if date_col:
            ym = parse_year_month(r.get(date_col, "")) if date_col else None
            if ym is not None:
                y, m = ym
                per_idx = _assign_period_index(y, m)
        recs.append({"id": mid, "x": float(C), "tasks": task_vals, "period": per_idx})

    # Auto-scale per-task values if they look like percentages (0..100) so that downstream
    # fitting/evaluation operates on [0, 1] accuracies.
    for t in tasks:
        raw = np.array([rec["tasks"].get(t, float("nan")) for rec in recs], dtype=float)
        scaled = maybe_scale_task_values(raw)
        for rec, v in zip(recs, scaled):
            rec["tasks"][t] = float(v)
    return recs


def _compute_pinball_for_period4_singlek(
    records: List[dict],
    tasks: List[str],
    bud_base: str,
    eval_base: str,
    tau: float,
    frontier_fit_mode: str,
    bins_for_fit: int,
    min_bin_size_for_fit: int,
    bin_frontier_quantile: float,
    bin_trim_fraction: float,
) -> None:
    """Compute train/val pinball loss for each k using manifests for fit, full pools for eval."""
    for k in (1, 2, 3):
        man_dir = os.path.join(bud_base, f"k{k}")
        eval_dir = os.path.join(eval_base, f"k{k}")
        os.makedirs(eval_dir, exist_ok=True)
        man_train = os.path.join(man_dir, "manifest__train.txt")
        man_val = os.path.join(man_dir, "manifest__val.txt")
        ids_train = read_manifest(man_train)
        ids_val = read_manifest(man_val)
        if not ids_train and not ids_val:
            continue

        # Determine which period indices correspond to this k
        split = PERIOD4_SPLITS_SINGLE[k - 1]
        label_to_idx = {}
        for i, bound in enumerate(PERIOD4_BOUNDS):
            if len(bound) == 3:
                label = bound[0]
            else:
                label = f"k{i+1}"
            label_to_idx[label] = i
        train_idx = [label_to_idx[l] for l in split.get("train_labels", []) if l in label_to_idx]
        val_idx = [label_to_idx[split["val_label"]]] if split.get("val_label") in label_to_idx else []

        # Full pools for evaluation (not restricted to manifests)
        full_tr = [rec for rec in records if rec.get("period", -1) in train_idx]
        full_val = [rec for rec in records if rec.get("period", -1) in val_idx]

        for task in tasks:
            # Fit on selected subset
            x_fit: List[float] = []
            y_fit: List[float] = []
            for rec in records:
                y = rec["tasks"].get(task, float("nan"))
                if not np.isfinite(y):
                    continue
                mid = rec["id"]
                if mid in ids_train:
                    x_fit.append(rec["x"])
                    y_fit.append(y)

            # Evaluate on full train/val pools
            x_tr_arr = np.asarray([rec["x"] for rec in full_tr if np.isfinite(rec["tasks"].get(task, float("nan")))], float)
            y_tr_arr = np.asarray([rec["tasks"][task] for rec in full_tr if np.isfinite(rec["tasks"].get(task, float("nan")))], float)
            x_val_arr = np.asarray([rec["x"] for rec in full_val if np.isfinite(rec["tasks"].get(task, float("nan")))], float)
            y_val_arr = np.asarray([rec["tasks"][task] for rec in full_val if np.isfinite(rec["tasks"].get(task, float("nan")))], float)

            x_fit_arr = np.asarray(x_fit, float)
            y_fit_arr = np.asarray(y_fit, float)

            if x_fit_arr.size == 0:
                L_tr = float("nan")
                L_val = float("nan")
            else:
                xs_curve, y_curve = fit_sigmoid_predictor(
                    x_fit_arr,
                    y_fit_arr,
                    tau=float(tau),
                    frontier_fit_mode=str(frontier_fit_mode),
                    bins_for_fit=bins_for_fit,
                    min_bin_size_for_fit=min_bin_size_for_fit,
                    bin_frontier_quantile=float(bin_frontier_quantile),
                    bin_trim_fraction=float(bin_trim_fraction),
                )
                yhat_tr = interpolate_curve(xs_curve, y_curve, x_tr_arr) if x_tr_arr.size else np.array([])
                yhat_val = interpolate_curve(xs_curve, y_curve, x_val_arr) if x_val_arr.size else np.array([])
                L_tr = _pinball_loss_mean(y_tr_arr, yhat_tr, tau=float(tau)) if yhat_tr.size else float("nan")
                L_val = _pinball_loss_mean(y_val_arr, yhat_val, tau=float(tau)) if yhat_val.size else float("nan")

            out_path = os.path.join(eval_dir, f"{sanitize_name(task)}__pinball_sigmoid.csv")
            with open(out_path, "w", newline="") as f:
                w = _csv.writer(f)
                w.writerow(["split", "L_sigmoid"])
                w.writerow(["train", L_tr])
                w.writerow(["val", L_val])

        # summary over tasks
        train_vals = []
        val_vals = []
        for task in tasks:
            path = os.path.join(eval_dir, f"{sanitize_name(task)}__pinball_sigmoid.csv")
            if not os.path.isfile(path):
                continue
            with open(path, "r", newline="") as f:
                r = list(_csv.DictReader(f))
            for row in r:
                if row.get("split") == "train":
                    try:
                        train_vals.append(float(row["L_sigmoid"]))
                    except Exception as exc:
                        LOGGER.debug("Failed to parse train pinball value for %s: %s", task, exc)
                if row.get("split") == "val":
                    try:
                        val_vals.append(float(row["L_sigmoid"]))
                    except Exception as exc:
                        LOGGER.debug("Failed to parse val pinball value for %s: %s", task, exc)
        summary_path = os.path.join(eval_dir, "summary_over_tasks_pinball_sigmoid.csv")
        with open(summary_path, "w", newline="") as f:
            w = _csv.writer(f)
            w.writerow(["split", "L_sigmoid"])
            w.writerow(["train", float(np.nanmean(train_vals)) if train_vals else float("nan")])
            w.writerow(["val", float(np.nanmean(val_vals)) if val_vals else float("nan")])

def _plot_task_curves(
    alpha_list: List[float],
    series_is: Dict[str, List[float]],
    series_oos: Dict[str, List[float]],
    title: str,
    out_path: str,
    k_value: int = None,
    style: str = "colored",
) -> None:
    import matplotlib.pyplot as plt  # type: ignore
    import matplotlib as mpl  # type: ignore
    import matplotlib.ticker as mticker  # type: ignore

    out_name = os.path.basename(out_path)
    is_period4_plot = out_name.startswith("period4_")
    is_pinball_plot = "pinball" in out_name

    target_alphas = list(alpha_list)
    pos = np.arange(len(target_alphas))
    use_highlight = str(style).lower() in {"highlight", "highlight_mean", "gray_highlight"}
    # We intentionally plot alphas on a *uniformly spaced* x-axis (pos) even if the
    # alpha values are not uniformly spaced numerically; tick labels show the true alphas.
    x = pos

    # Task order: sorted for determinism
    tasks = sorted(series_oos.keys())
    is_bbh_subtask_sweep = any(t.startswith("leaderboard_bbh_") for t in tasks)
    suppress_legend = is_bbh_subtask_sweep
    # For BBH subtasks there are too many curves for per-task legends/linestyles.
    if is_bbh_subtask_sweep:
        use_highlight = True

    # Period4 sweeps in `outputs/sweeps_fullrange`: match the style of Figure 6 (main paper)
    # while keeping the per-task legend entries (gray lines) used in the current sweeps plots.
    use_fig6_style = bool(is_period4_plot and use_highlight and not is_bbh_subtask_sweep)
    if use_fig6_style:
        dpi = 300
        mpl.rcParams["figure.dpi"] = dpi
        mpl.rcParams["savefig.dpi"] = dpi
        mpl.rcParams["pdf.fonttype"] = 42
        mpl.rcParams["ps.fonttype"] = 42
        mpl.rcParams["text.usetex"] = False
        plt.rcParams.update(
            {
                "font.family": "serif",
                "font.serif": ["Times New Roman"],
                "font.size": 9,
                "axes.labelsize": 10,
                "axes.titlesize": 11,
                "xtick.labelsize": 8.5,
                "ytick.labelsize": 8.5,
                "legend.fontsize": 8,
                "axes.linewidth": 0.8,
                "figure.facecolor": "white",
                "axes.facecolor": "white",
                "savefig.facecolor": "white",
                "mathtext.fontset": "cm",
            }
        )
    else:
        # Serif font for academic style and enable LaTeX rendering
        try:
            mpl.rcParams["font.family"] = mpl_rc_cfg.FONT_FAMILY
            mpl.rcParams["text.usetex"] = sweep_alpha_cfg.USETEX
            mpl.rcParams["text.latex.preamble"] = sweep_alpha_cfg.LATEX_PREAMBLE
        except Exception as exc:
            LOGGER.debug("Unable to apply LaTeX rcParams: %s", exc)

    # Figure: two subplots (ID left, OOD right)
    if use_fig6_style:
        fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.6), dpi=300, sharey=True)
        plt.subplots_adjust(left=0.09, right=0.98, bottom=0.15, top=0.93, wspace=0.28)
    else:
        fig, axes = plt.subplots(1, 2, figsize=sweep_alpha_cfg.FIGSIZE, sharey=True)
    ax_is, ax_oos = axes

    markers = sweep_alpha_cfg.MARKERS  # circle, square, tri-up, diamond, tri-down, pentagon

    # Plot IS and OOS per task with consistent styling
    if use_highlight:
        # Highlight-style: gray family for individual tasks + firebrick mean curve.
        if use_fig6_style:
            task_colors = ["#999999"] * max(len(tasks), 1)
            task_alpha = 0.55
            task_lw = 1.6
            # Use distinct line styles for each task (as requested).
            # Matplotlib supports custom dash sequences via (offset, on_off_seq).
            base_linestyles = [
                "-",
                "--",
                "-.",
                ":",
                (0, (5, 1)),
                (0, (3, 1, 1, 1)),
                (0, (7, 2)),
                (0, (4, 2, 1, 2)),
            ]
            if len(tasks) > len(base_linestyles):
                for n in range(8, 8 + (len(tasks) - len(base_linestyles))):
                    base_linestyles.append((0, (max(2, n), 2)))
            task_linestyles = base_linestyles[: max(len(tasks), 1)]
        else:
            cmap = plt.get_cmap("Greys")
            if is_bbh_subtask_sweep:
                mid_grey = float(
                    0.5 * (sweep_alpha_cfg.HIGHLIGHT_GREYS_LO + sweep_alpha_cfg.HIGHLIGHT_GREYS_HI)
                )
                task_colors = [cmap(mid_grey)] * max(len(tasks), 1)
            else:
                # Avoid extremely light grays (hard to see under transparency).
                levels = np.linspace(
                    sweep_alpha_cfg.HIGHLIGHT_GREYS_LO,
                    sweep_alpha_cfg.HIGHLIGHT_GREYS_HI,
                    num=max(len(tasks), 2),
                )
                task_colors = [cmap(float(lv)) for lv in levels[: max(len(tasks), 1)]]
            task_alpha = sweep_alpha_cfg.HIGHLIGHT_TASK_ALPHA
            task_lw = sweep_alpha_cfg.HIGHLIGHT_TASK_LINEWIDTH
            task_linestyles = ["-"] if is_bbh_subtask_sweep else sweep_alpha_cfg.HIGHLIGHT_TASK_LINESTYLES
        for idx, task in enumerate(tasks):
            alpha_to_is = {a: series_is.get(task, [np.nan] * len(alpha_list))[i] for i, a in enumerate(alpha_list)}
            alpha_to_oos = {a: series_oos.get(task, [np.nan] * len(alpha_list))[i] for i, a in enumerate(alpha_list)}
            ys_is = [alpha_to_is.get(a, np.nan) for a in target_alphas]
            ys_oos = [alpha_to_oos.get(a, np.nan) for a in target_alphas]
            legend_label = task.replace(" Raw", "")
            color = task_colors[idx % len(task_colors)]
            linestyle = task_linestyles[idx % len(task_linestyles)]
            ax_is.plot(
                x,
                ys_is,
                color=color,
                linestyle=linestyle,
                linewidth=task_lw,
                alpha=task_alpha,
                label=legend_label if not suppress_legend else None,
                zorder=3,
            )
            ax_oos.plot(
                x,
                ys_oos,
                color=color,
                linestyle=linestyle,
                linewidth=task_lw,
                alpha=task_alpha,
                label=legend_label if not suppress_legend else None,
                zorder=3,
            )
    else:
        # Colored style (legacy): colorblind-friendly palette (6 distinct colors from tab10)
        colors = sweep_alpha_cfg.COLORED_TASK_COLORS
        for idx, task in enumerate(tasks):
            alpha_to_is = {a: series_is.get(task, [np.nan] * len(alpha_list))[i] for i, a in enumerate(alpha_list)}
            alpha_to_oos = {a: series_oos.get(task, [np.nan] * len(alpha_list))[i] for i, a in enumerate(alpha_list)}
            ys_is = [alpha_to_is.get(a, np.nan) for a in target_alphas]
            ys_oos = [alpha_to_oos.get(a, np.nan) for a in target_alphas]
            # Clean legend label: drop trailing " Raw" if present
            legend_label = task.replace(" Raw", "")
            color = colors[idx % len(colors)]
            marker = markers[idx % len(markers)]
            # In-sample (left)
            ax_is.plot(
                pos,
                ys_is,
                marker=marker,
                color=color,
                linewidth=sweep_alpha_cfg.COLORED_TASK_LINEWIDTH,
                markersize=sweep_alpha_cfg.COLORED_TASK_MARKERSIZE,
                markeredgewidth=sweep_alpha_cfg.COLORED_TASK_MARKEREDGEWIDTH,
                markeredgecolor=sweep_alpha_cfg.COLORED_TASK_MARKEREDGECOLOR,
                label=legend_label if not suppress_legend else None,
                zorder=3,
            )
            # Out-of-sample (right)
            ax_oos.plot(
                pos,
                ys_oos,
                marker=marker,
                color=color,
                linewidth=sweep_alpha_cfg.COLORED_TASK_LINEWIDTH,
                markersize=sweep_alpha_cfg.COLORED_TASK_MARKERSIZE,
                markeredgewidth=sweep_alpha_cfg.COLORED_TASK_MARKEREDGEWIDTH,
                markeredgecolor=sweep_alpha_cfg.COLORED_TASK_MARKEREDGECOLOR,
                label=legend_label if not suppress_legend else None,
                zorder=3,
            )

    # Aggregate average coverage error across all tasks (per alpha)
    avg_is: List[float] = []
    avg_oos: List[float] = []
    if tasks and alpha_list:
        for j in range(len(target_alphas)):
            vals_is_j: List[float] = []
            vals_oos_j: List[float] = []
            for t in tasks:
                alpha_to_is = {a: series_is.get(t, [np.nan] * len(alpha_list))[i] for i, a in enumerate(alpha_list)}
                alpha_to_oos = {a: series_oos.get(t, [np.nan] * len(alpha_list))[i] for i, a in enumerate(alpha_list)}
                vals_is_j.append(alpha_to_is.get(target_alphas[j], np.nan))
                vals_oos_j.append(alpha_to_oos.get(target_alphas[j], np.nan))
            def _nanmean_safe(vals: List[float]) -> float:
                arr = np.asarray([v for v in vals if np.isfinite(v)], dtype=float)
                return float(arr.mean()) if arr.size else float("nan")
            avg_is.append(_nanmean_safe(vals_is_j))
            avg_oos.append(_nanmean_safe(vals_oos_j))

        # Plot aggregated averages with a common style on both subplots
        if use_highlight:
            if use_fig6_style:
                avg_color = "#e41a1c"
                ax_is.plot(
                    x,
                    avg_is,
                    color=avg_color,
                    alpha=0.95,
                    linewidth=2.4,
                    marker="o",
                    markersize=6.0,
                    markeredgecolor="white",
                    markeredgewidth=0.6,
                    label="Average" if not suppress_legend else None,
                    zorder=6,
                )
                ax_oos.plot(
                    x,
                    avg_oos,
                    color=avg_color,
                    alpha=0.95,
                    linewidth=2.4,
                    marker="o",
                    markersize=6.0,
                    markeredgecolor="white",
                    markeredgewidth=0.6,
                    label="Average" if not suppress_legend else None,
                    zorder=6,
                )
            else:
                ax_is.plot(
                    x,
                    avg_is,
                    color=sweep_alpha_cfg.HIGHLIGHT_AVG_COLOR,
                    linewidth=sweep_alpha_cfg.HIGHLIGHT_AVG_LINEWIDTH,
                    alpha=1.0,
                    label="Average" if not suppress_legend else None,
                    zorder=6,
                )
                ax_oos.plot(
                    x,
                    avg_oos,
                    color=sweep_alpha_cfg.HIGHLIGHT_AVG_COLOR,
                    linewidth=sweep_alpha_cfg.HIGHLIGHT_AVG_LINEWIDTH,
                    alpha=1.0,
                    label="Average" if not suppress_legend else None,
                    zorder=6,
                )
        else:
            ax_is.plot(
                pos,
                avg_is,
                color=sweep_alpha_cfg.COLORED_AVG_COLOR,
                linestyle=sweep_alpha_cfg.COLORED_AVG_LINESTYLE,
                linewidth=sweep_alpha_cfg.COLORED_AVG_LINEWIDTH,
                marker=sweep_alpha_cfg.COLORED_AVG_MARKER,
                markersize=sweep_alpha_cfg.COLORED_AVG_MARKERSIZE,
                markeredgewidth=sweep_alpha_cfg.COLORED_AVG_MARKEREDGEWIDTH,
                markeredgecolor=sweep_alpha_cfg.COLORED_AVG_MARKEREDGECOLOR,
                label="Average" if not suppress_legend else None,
                zorder=5,
            )
            ax_oos.plot(
                pos,
                avg_oos,
                color=sweep_alpha_cfg.COLORED_AVG_COLOR,
                linestyle=sweep_alpha_cfg.COLORED_AVG_LINESTYLE,
                linewidth=sweep_alpha_cfg.COLORED_AVG_LINEWIDTH,
                marker=sweep_alpha_cfg.COLORED_AVG_MARKER,
                markersize=sweep_alpha_cfg.COLORED_AVG_MARKERSIZE,
                markeredgewidth=sweep_alpha_cfg.COLORED_AVG_MARKEREDGEWIDTH,
                markeredgecolor=sweep_alpha_cfg.COLORED_AVG_MARKEREDGECOLOR,
                label="Average" if not suppress_legend else None,
                zorder=5,
            )

    if use_fig6_style:
        # Figure 6-style panel chrome (no minor ticks, light dotted grid, clean spines).
        for ax in (ax_is, ax_oos):
            ax.set_axisbelow(True)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.spines["left"].set_linewidth(0.8)
            ax.spines["bottom"].set_linewidth(0.8)
            ax.grid(True, alpha=0.2, linestyle=":", linewidth=0.5, zorder=0)
            ax.tick_params(axis="both", which="major", length=4, width=0.8, direction="out")
            ax.tick_params(axis="both", which="minor", length=0)
    else:
        # Multi-level grid
        for ax in (ax_is, ax_oos):
            ax.set_axisbelow(True)
            ax.grid(
                True,
                which="major",
                linestyle=sweep_alpha_cfg.GRID_MAJOR_LINESTYLE,
                linewidth=sweep_alpha_cfg.GRID_MAJOR_LINEWIDTH,
                alpha=sweep_alpha_cfg.GRID_MAJOR_ALPHA,
                color=sweep_alpha_cfg.GRID_MAJOR_COLOR,
                zorder=1,
            )
            ax.grid(
                True,
                which="minor",
                linestyle=sweep_alpha_cfg.GRID_MINOR_LINESTYLE,
                linewidth=sweep_alpha_cfg.GRID_MINOR_LINEWIDTH,
                alpha=sweep_alpha_cfg.GRID_MINOR_ALPHA,
                color=sweep_alpha_cfg.GRID_MINOR_COLOR,
                zorder=1,
            )
            ax.minorticks_on()

        # Refined background and spines
        for ax in (ax_is, ax_oos):
            ax.set_facecolor(sweep_alpha_cfg.AX_FACE_COLOR)
            for spine in ax.spines.values():
                spine.set_edgecolor(sweep_alpha_cfg.SPINE_EDGE_COLOR)
                spine.set_linewidth(sweep_alpha_cfg.SPINE_LINEWIDTH)
            # Remove top and right spines for modern minimal style
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

    # Typography with hierarchy
    if use_fig6_style:
        xlabel = r"$\alpha$ (% of evaluation budget)"
        ylabel = "Pinball loss" if is_pinball_plot else "Coverage error"
        ax_is.set_xlabel(xlabel, fontsize=12, fontweight="bold", labelpad=4)
        ax_oos.set_xlabel(xlabel, fontsize=12, fontweight="bold", labelpad=4)
        ax_is.set_ylabel(ylabel, fontsize=12, fontweight="bold", labelpad=4)
        ax_oos.set_ylabel("")

        def _format_alpha_label(a: float) -> str:
            try:
                return str(int(a)) if float(a).is_integer() else str(a)
            except Exception:
                return str(a)

        for ax in (ax_is, ax_oos):
            ax.set_xticks(pos)
            ax.set_xticklabels([_format_alpha_label(a) for a in target_alphas])
            ax.set_xlim(pos[0] - 0.2, pos[-1] + 0.2)

        title_bbox = dict(
            boxstyle="round,pad=0.35",
            facecolor="#f0f0f0",
            edgecolor="#cccccc",
            linewidth=0.8,
            alpha=0.95,
        )
        title_obj = ax_is.set_title("In-distribution (ID)", fontweight="bold", pad=12)
        title_obj.set_bbox(title_bbox)
        title_obj = ax_oos.set_title("Out-of-distribution (OOD)", fontweight="bold", pad=12)
        title_obj.set_bbox(title_bbox)
    else:
        # Using LaTeX bold commands since text.usetex = True
        ylabel_size = sweep_alpha_cfg.YLABEL_FONTSIZE
        title_size = sweep_alpha_cfg.TITLE_FONTSIZE
        ax_is.set_xlabel(
            r"$\boldsymbol{\alpha}$ \textbf{(\% of evaluation budget)}", fontsize=sweep_alpha_cfg.XLABEL_FONTSIZE
        )
        ax_oos.set_xlabel(
            r"$\boldsymbol{\alpha}$ \textbf{(\% of evaluation budget)}", fontsize=sweep_alpha_cfg.XLABEL_FONTSIZE
        )
        ax_is.set_ylabel(r"\textbf{coverage error}", fontsize=ylabel_size)
        ax_is.tick_params(axis="both", labelsize=sweep_alpha_cfg.TICK_LABELSIZE)
        ax_oos.tick_params(axis="both", labelsize=sweep_alpha_cfg.TICK_LABELSIZE)
        for ax in (ax_is, ax_oos):
            ax.set_xticks(pos)
            ax.set_xticklabels(target_alphas)
            ax.set_xlim(pos[0] - 0.2, pos[-1] + 0.2)

        # Subfigure titles only (no global figure title), nudged slightly upward
        ax_is.set_title(r"\textbf{In-distribution (ID)}", fontsize=title_size, pad=sweep_alpha_cfg.SUBTITLE_PAD)
        ax_oos.set_title(r"\textbf{Out-of-distribution (OOD)}", fontsize=title_size, pad=sweep_alpha_cfg.SUBTITLE_PAD)

    # Add k-value text box if provided (on both subplots for clarity)
    if k_value is not None:
        textstr = rf"$t={k_value}$"
        props = (
            dict(
                boxstyle="round,pad=0.25",
                facecolor="#f0f0f0",
                edgecolor="#cccccc",
                linewidth=0.8,
                alpha=0.95,
            )
            if use_fig6_style
            else dict(
                boxstyle=sweep_alpha_cfg.KBOX_STYLE,
                facecolor=sweep_alpha_cfg.KBOX_FACE,
                edgecolor=sweep_alpha_cfg.KBOX_EDGE,
                alpha=sweep_alpha_cfg.KBOX_ALPHA,
            )
        )
        fontsize = 12 if use_fig6_style else sweep_alpha_cfg.KBOX_FONTSIZE
        for ax in (ax_is, ax_oos):
            ax.text(
                0.05,
                0.95,
                textstr,
                transform=ax.transAxes,
                fontsize=fontsize,
                verticalalignment="top",
                bbox=props,
            )

    # Dynamic y-axis limits with padding (shared for IS/OOS)
    all_vals = [
        v
        for ys in list(series_is.values()) + list(series_oos.values())
        for v in ys
        if np.isfinite(v) and v > 0
    ]
    # Include aggregated averages if present
    if avg_is and avg_oos:
        all_vals.extend([v for v in avg_is + avg_oos if np.isfinite(v) and v > 0])
    if all_vals:
        ymin, ymax = min(all_vals) * 0.95, max(all_vals) * 1.05
        ax_is.set_ylim(ymin, ymax)
        ax_oos.set_ylim(ymin, ymax)
        # Log scale (shared y-axis)
        if use_fig6_style:
            ax_is.set_yscale("log")
        else:
            # Log scale for OOS to separate well-performing models
            ax_oos.set_yscale("log")
    # Ensure x-axis spans the full alpha list, even if some y are NaN at endpoints
    if alpha_list:
        try:
            x_right = pos[-1] + 0.2 if len(pos) else len(alpha_list) - 0.8
            ax_is.set_xlim(-0.2, x_right)
            ax_oos.set_xlim(-0.2, x_right)
        except Exception as exc:
            LOGGER.debug("Unable to set x-limits for alpha sweep plot: %s", exc)
    if suppress_legend:
        if use_fig6_style:
            plt.subplots_adjust(left=0.09, right=0.98, bottom=0.15, top=0.93, wspace=0.28)
        else:
            fig.tight_layout()
    else:
        if use_fig6_style:
            plt.subplots_adjust(left=0.09, right=0.98, bottom=0.15, top=0.93, wspace=0.28)
        else:
            # Adjust layout to leave room for legend below
            if use_highlight:
                fig.tight_layout(rect=sweep_alpha_cfg.TIGHT_LAYOUT_RECT_HIGHLIGHT)
            else:
                fig.tight_layout(rect=sweep_alpha_cfg.TIGHT_LAYOUT_RECT_COLORED)

        # Legend below both subplots in a single row (deduplicate labels across axes)
        handles_is, labels_is = ax_is.get_legend_handles_labels()
        handles_oos, labels_oos = ax_oos.get_legend_handles_labels()
        all_handles = list(handles_is) + list(handles_oos)
        all_labels = list(labels_is) + list(labels_oos)
        dedup: dict = {}
        for h, lab in zip(all_handles, all_labels):
            if lab not in dedup:
                dedup[lab] = h
        handles = list(dedup.values())
        labels = list(dedup.keys())
        if handles:
            if use_fig6_style:
                # Keep the legend in a single row (as requested).
                ncol = len(labels)
                legend = fig.legend(
                    handles,
                    labels,
                    loc="lower center",
                    bbox_to_anchor=(0.5, -0.18),
                    ncol=ncol,
                    fontsize=8,
                    framealpha=0.9,
                    edgecolor="#999999",
                    fancybox=False,
                    handlelength=1.5,
                    handletextpad=0.5,
                    borderpad=0.5,
                    columnspacing=1.2,
                )
                # Ensure the figure-level legend doesn't overlap the x-axis labels.
                # (We keep the subplot geometry/figsize fixed and push the legend
                # further down; savefig(bbox_inches="tight") will include it.)
                try:
                    fig.canvas.draw()
                    renderer = fig.canvas.get_renderer()
                    xlabels = [ax.xaxis.label for ax in (ax_is, ax_oos)]
                    xlbl_bboxes = [
                        t.get_window_extent(renderer=renderer).transformed(fig.transFigure.inverted())
                        for t in xlabels
                        if t.get_text()
                    ]
                    if xlbl_bboxes:
                        xlbl_bottom = min(bb.y0 for bb in xlbl_bboxes)
                        pad = 0.01
                        y_anchor = -0.18
                        for _ in range(8):
                            fig.canvas.draw()
                            leg_bb = legend.get_window_extent(renderer=renderer).transformed(
                                fig.transFigure.inverted()
                            )
                            if leg_bb.y1 < (xlbl_bottom - pad):
                                break
                            shift = (leg_bb.y1 - (xlbl_bottom - pad)) + 0.005
                            y_anchor -= shift
                            legend.set_bbox_to_anchor((0.5, y_anchor), transform=fig.transFigure)
                except Exception as exc:
                    LOGGER.debug("Legend anchor adjustment failed: %s", exc)
            else:
                ncol = len(labels)
                fig.legend(
                    handles,
                    labels,
                    loc=sweep_alpha_cfg.LEGEND_LOC,
                    bbox_to_anchor=(
                        sweep_alpha_cfg.LEGEND_BBOX_TO_ANCHOR_HIGHLIGHT
                        if use_highlight
                        else sweep_alpha_cfg.LEGEND_BBOX_TO_ANCHOR_COLORED
                    ),
                    ncol=ncol,
                    fontsize=sweep_alpha_cfg.LEGEND_FONTSIZE,
                    framealpha=sweep_alpha_cfg.LEGEND_FRAMEALPHA,
                    edgecolor=sweep_alpha_cfg.LEGEND_EDGECOLOR,
                    fancybox=sweep_alpha_cfg.LEGEND_FANCYBOX,
                    markerscale=sweep_alpha_cfg.LEGEND_MARKERSCALE,
                    labelspacing=sweep_alpha_cfg.LEGEND_LABELSPACING,
                    handlelength=sweep_alpha_cfg.LEGEND_HANDLELENGTH,
                    handleheight=sweep_alpha_cfg.LEGEND_HANDLEHEIGHT,
                )

    # Save with tight layout including legend
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    save_kws = dict(dpi=300, bbox_inches="tight")
    if use_fig6_style:
        save_kws.update({"pad_inches": 0.05, "facecolor": "white", "edgecolor": "none"})
    plt.savefig(out_path, **save_kws)
    # Also save PDF version
    out_path_pdf = out_path.replace(".png", ".pdf")
    plt.savefig(out_path_pdf, **save_kws)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description="Sweep alpha% budgets and plot OOS MAE vs alpha")
    ap.add_argument("--csv", required=True)
    ap.add_argument("--compute_product_cols", nargs=2, default=None)
    ap.add_argument("--compute_multiplier", type=float, default=6.0)
    ap.add_argument("--alphas", nargs="*", type=float, default=[10,20,30,40,50])
    ap.add_argument(
        "--design_objective",
        choices=["d_optimal", "i_optimal_predvar", "i_optimal_predvar_balanced"],
        default="d_optimal",
        help="Design objective for budget selection (default: d_optimal)",
    )
    ap.add_argument(
        "--balance_lambda",
        type=float,
        default=0.0,
        help="Bin-balance lambda for balanced I-optimal (default 0.0)",
    )
    ap.add_argument(
        "--frontier_fit_mode",
        choices=["quantile_per_point", "robust_bin_frontier"],
        default="quantile_per_point",
        help="How to fit sigmoid frontiers when evaluating sweeps (default: quantile_per_point)",
    )
    ap.add_argument(
        "--bins",
        type=int,
        default=10,
        help="Target number of equal-mass bins used in evaluation (default: 10).",
    )
    ap.add_argument(
        "--min_bin_size",
        type=int,
        default=30,
        help="Minimum samples per bin used in evaluation (default: 30).",
    )
    ap.add_argument(
        "--oos_bins",
        choices=["train_overlap", "test_fixed"],
        default="test_fixed",
        help="How to define OOS bins during evaluation (default: test_fixed).",
    )
    ap.add_argument(
        "--bin_frontier_quantile",
        type=float,
        default=0.98,
        help="Quantile level for bin-level robust frontier targets when using robust_bin_frontier mode",
    )
    ap.add_argument(
        "--bin_trim_fraction",
        type=float,
        default=0.01,
        help="Fraction of top points per bin to trim before computing the bin-level frontier quantile",
    )
    ap.add_argument(
        "--tasks",
        nargs="*",
        default=None,
        help=(
            "Optional explicit task columns to evaluate in sweeps; "
            "if omitted, uses canonical OLL Raw tasks present in the CSV"
        ),
    )
    ap.add_argument(
        "--plot_only",
        action="store_true",
        help="Only read existing eval_* folders and regenerate plots (skip design/eval).",
    )
    ap.add_argument("--out_dir", default=os.path.join("outputs", "sweeps"))
    ap.add_argument("--exchange_passes", type=int, default=0, help="Optional 1-exchange polishing passes in selection (default 0 = greedy only)")
    ap.add_argument(
        "--write_pinball",
        action="store_true",
        help="If set, also compute sigmoid pinball losses for sweeps (train/val) and write per-alpha summaries.",
    )
    args = ap.parse_args()

    if not args.compute_product_cols or len(args.compute_product_cols) != 2:
        raise SystemExit("--compute_product_cols is required (two columns)")

    alphas = [float(a) for a in args.alphas]
    os.makedirs(args.out_dir, exist_ok=True)
    rows, headers = read_csv_rows(args.csv)
    if not rows:
        raise SystemExit(f"No rows in CSV: {args.csv}")

    # Detect tasks from CSV header once so every alpha yields aligned series.
    # If --tasks is provided, restrict to those columns; otherwise use canonical OLL Raw tasks.
    #
    # In plot-only mode, prefer inferring tasks from the existing eval_* summaries, since
    # the current CSV may contain many additional columns (e.g., BBH subtasks) that are
    # not part of the canonical Raw task set.
    tasks_seen: List[str] = []
    if not (args.plot_only and not args.tasks):
        try:
            with open(args.csv, "r", newline="") as f:
                import csv as _csv

                rd = _csv.reader(f)
                header = next(rd)
            hset = set(h.strip() for h in header)
            if args.tasks:
                tasks_seen = [t for t in args.tasks if t in hset]
            else:
                tasks_seen = detect_oll_raw_tasks(list(hset))
        except Exception:
            tasks_seen = list(args.tasks) if args.tasks else []

    python_bin = sys.executable or "python3"
    # Precompute model records (compute + task scores) for pinball reuse
    model_records = _build_model_records(
        rows,
        headers,
        tasks_seen,
        compute_product_cols=(args.compute_product_cols[0], args.compute_product_cols[1]),
        compute_multiplier=float(args.compute_multiplier),
    )

    # 2-period sweep
    series_year_is: Dict[str, List[float]] = {}
    series_year_oos: Dict[str, List[float]] = {}
    for a in alphas:
        eval_dir = os.path.join(args.out_dir, f"eval_year_alpha{int(a)}")
        if not args.plot_only:
            bud_dir = os.path.join(args.out_dir, f"year_alpha{int(a)}")
            os.makedirs(bud_dir, exist_ok=True)
            man_pre, man_2025 = _design_year_split(
                rows,
                headers,
                compute_product_cols=(args.compute_product_cols[0], args.compute_product_cols[1]),
                compute_multiplier=float(args.compute_multiplier),
                alpha_percent=float(a),
                out_dir=bud_dir,
                exchange_passes=int(max(0, args.exchange_passes)),
                objective=str(args.design_objective),
                balance_lambda=float(args.balance_lambda),
            )
            # Evaluate
            os.makedirs(eval_dir, exist_ok=True)
            cmd = [
                python_bin,
                os.path.join("scripts", "evaluate", "sigmoid_binned_mae.py"),
                "--csv",
                args.csv,
                "--compute_product_cols",
                args.compute_product_cols[0],
                args.compute_product_cols[1],
                "--compute_multiplier",
                str(args.compute_multiplier),
                "--split_mode",
                "pre2025_vs_2025",
                "--tau",
                "0.98",
                "--bins",
                str(int(args.bins)),
                "--min_bin_size",
                str(int(args.min_bin_size)),
                "--frontier_fit_mode",
                args.frontier_fit_mode,
                "--bin_frontier_quantile",
                str(args.bin_frontier_quantile),
                "--bin_trim_fraction",
                str(args.bin_trim_fraction),
                "--out_dir",
                eval_dir,
                "--manifest_pre",
                man_pre,
                "--manifest_2025",
                man_2025,
                "--oos_bins",
                args.oos_bins,
            ]
            if args.tasks:
                cmd.extend(["--tasks", *args.tasks])
            subprocess.check_call(cmd)
        else:
            if not os.path.isdir(eval_dir):
                print(f"[sweep_alpha] plot_only: missing eval_dir for alpha={a}: {eval_dir}")
                continue
        # Read per-task IS/OOS coverage error
        maes = _read_task_mae(eval_dir)
        if not tasks_seen:
            tasks_seen = sorted(maes.keys())
        for t in tasks_seen:
            is_v, oos_v = maes.get(t, (float("nan"), float("nan")))
            series_year_is.setdefault(t, []).append(is_v)
            series_year_oos.setdefault(t, []).append(oos_v)
    # (no baseline override at alpha=0)

    # 4-period (single_k) sweep: average OOS Macro across k per alpha
    series_p4_is: Dict[str, List[float]] = {}
    series_p4_oos: Dict[str, List[float]] = {}
    # Also keep separate per-k series
    series_p4_k1_is: Dict[str, List[float]] = {}
    series_p4_k1_oos: Dict[str, List[float]] = {}
    series_p4_k2_is: Dict[str, List[float]] = {}
    series_p4_k2_oos: Dict[str, List[float]] = {}
    series_p4_k3_is: Dict[str, List[float]] = {}
    series_p4_k3_oos: Dict[str, List[float]] = {}
    for a in alphas:
        eval_base = os.path.join(args.out_dir, f"eval_p4_alpha{int(a)}")
        if not args.plot_only:
            bud_base = os.path.join(args.out_dir, f"p4_alpha{int(a)}")
            os.makedirs(bud_base, exist_ok=True)
            _design_period4_singlek(
                rows,
                headers,
                compute_product_cols=(args.compute_product_cols[0], args.compute_product_cols[1]),
                compute_multiplier=float(args.compute_multiplier),
                alpha_percent=float(a),
                out_base=bud_base,
                exchange_passes=int(max(0, args.exchange_passes)),
                objective=str(args.design_objective),
                balance_lambda=float(args.balance_lambda),
            )
            # Evaluate per k with manifests
            cmd = [
                python_bin,
                os.path.join("scripts", "evaluate", "sigmoid_binned_mae.py"),
                "--csv",
                args.csv,
                "--compute_product_cols",
                args.compute_product_cols[0],
                args.compute_product_cols[1],
                "--compute_multiplier",
                str(args.compute_multiplier),
                "--split_mode",
                "period4",
                "--train_mode",
                "single_k",
                "--tau",
                "0.98",
                "--bins",
                str(int(args.bins)),
                "--min_bin_size",
                str(int(args.min_bin_size)),
                "--frontier_fit_mode",
                args.frontier_fit_mode,
                "--bin_frontier_quantile",
                str(args.bin_frontier_quantile),
                "--bin_trim_fraction",
                str(args.bin_trim_fraction),
                "--out_base",
                eval_base,
                "--manifest_base",
                bud_base,
                "--manifest_apply",
                "train_only",
                "--oos_bins",
                args.oos_bins,
            ]
            if args.tasks:
                cmd.extend(["--tasks", *args.tasks])
            subprocess.check_call(cmd)
            if args.write_pinball:
                _compute_pinball_for_period4_singlek(
                    model_records,
                    tasks_seen,
                    bud_base,
                    eval_base,
                    tau=0.98,
                    frontier_fit_mode=args.frontier_fit_mode,
                    bins_for_fit=int(args.bins),
                    min_bin_size_for_fit=int(args.min_bin_size),
                    bin_frontier_quantile=float(args.bin_frontier_quantile),
                    bin_trim_fraction=float(args.bin_trim_fraction),
                )
        else:
            if not os.path.isdir(eval_base):
                print(f"[sweep_alpha] plot_only: missing eval_base for alpha={a}: {eval_base}")
                continue
        vals = _read_task_mae_period4(eval_base)
        for t in tasks_seen:
            is_v, oos_v = vals.get(t, (float("nan"), float("nan")))
            series_p4_is.setdefault(t, []).append(is_v)
            series_p4_oos.setdefault(t, []).append(oos_v)
        # Per-k values
        for k, series_k_is, series_k_oos in (
            (1, series_p4_k1_is, series_p4_k1_oos),
            (2, series_p4_k2_is, series_p4_k2_oos),
            (3, series_p4_k3_is, series_p4_k3_oos),
        ):
            kdir = os.path.join(eval_base, f"k{k}")
            vals_k = _read_task_mae(kdir) if os.path.isdir(kdir) else {}
            for t in tasks_seen:
                is_v, oos_v = vals_k.get(t, (float("nan"), float("nan")))
                series_k_is.setdefault(t, []).append(is_v)
                series_k_oos.setdefault(t, []).append(oos_v)
    # (no baseline override at alpha=0)

    # Plot
    _plot_task_curves(
        alphas,
        series_year_is,
        series_year_oos,
        title=r"coverage error vs. $\alpha$ (2-period: pre-2025 vs 2025)",
        out_path=os.path.join(args.out_dir, "year_tasks_vs_alpha.png"),
    )
    _plot_task_curves(
        alphas,
        series_p4_is,
        series_p4_oos,
        title=r"coverage error vs. $\alpha$ (averaged over $t$)",
        out_path=os.path.join(args.out_dir, "period4_singlek_tasks_vs_alpha.png"),
        style="highlight",
    )
    # Separate plots per k with k-value text boxes
    _plot_task_curves(
        alphas,
        series_p4_k1_is,
        series_p4_k1_oos,
        title=r"coverage error vs. $\alpha$ ($t=1$)",
        out_path=os.path.join(args.out_dir, "period4_singlek_k1_tasks_vs_alpha.png"),
        k_value=1,
        style="highlight",
    )
    _plot_task_curves(
        alphas,
        series_p4_k2_is,
        series_p4_k2_oos,
        title=r"coverage error vs. $\alpha$ ($t=2$)",
        out_path=os.path.join(args.out_dir, "period4_singlek_k2_tasks_vs_alpha.png"),
        k_value=2,
        style="highlight",
    )
    _plot_task_curves(
        alphas,
        series_p4_k3_is,
        series_p4_k3_oos,
        title=r"coverage error vs. $\alpha$ ($t=3$)",
        out_path=os.path.join(args.out_dir, "period4_singlek_k3_tasks_vs_alpha.png"),
        k_value=3,
        style="highlight",
    )

    # Pinball task plots (period4 single_k), using pinball summaries if present
    series_pin_is: Dict[str, List[float]] = {}
    series_pin_oos: Dict[str, List[float]] = {}
    series_pin_k1_is: Dict[str, List[float]] = {}
    series_pin_k1_oos: Dict[str, List[float]] = {}
    series_pin_k2_is: Dict[str, List[float]] = {}
    series_pin_k2_oos: Dict[str, List[float]] = {}
    series_pin_k3_is: Dict[str, List[float]] = {}
    series_pin_k3_oos: Dict[str, List[float]] = {}

    for a in alphas:
        eval_base = os.path.join(args.out_dir, f"eval_p4_alpha{int(a)}")
        vals = _read_task_pinball_period4(eval_base)
        for t in tasks_seen:
            is_v, oos_v = vals.get(t, (float("nan"), float("nan")))
            series_pin_is.setdefault(t, []).append(is_v)
            series_pin_oos.setdefault(t, []).append(oos_v)
        for k, series_k_is, series_k_oos in (
            (1, series_pin_k1_is, series_pin_k1_oos),
            (2, series_pin_k2_is, series_pin_k2_oos),
            (3, series_pin_k3_is, series_pin_k3_oos),
        ):
            kdir = os.path.join(eval_base, f"k{k}")
            vals_k = _read_task_pinball(kdir) if os.path.isdir(kdir) else {}
            for t in tasks_seen:
                is_v, oos_v = vals_k.get(t, (float("nan"), float("nan")))
                series_k_is.setdefault(t, []).append(is_v)
                series_k_oos.setdefault(t, []).append(oos_v)

    _plot_task_curves(
        alphas,
        series_pin_is,
        series_pin_oos,
        title=r"Pinball loss vs. $\alpha$ (avg over $t$)",
        out_path=os.path.join(args.out_dir, "period4_singlek_tasks_vs_alpha_pinball.png"),
        style="highlight",
    )
    _plot_task_curves(
        alphas,
        series_pin_k1_is,
        series_pin_k1_oos,
        title=r"Pinball loss vs. $\alpha$ ($t=1$)",
        out_path=os.path.join(args.out_dir, "period4_singlek_k1_tasks_vs_alpha_pinball.png"),
        k_value=1,
        style="highlight",
    )
    _plot_task_curves(
        alphas,
        series_pin_k2_is,
        series_pin_k2_oos,
        title=r"Pinball loss vs. $\alpha$ ($t=2$)",
        out_path=os.path.join(args.out_dir, "period4_singlek_k2_tasks_vs_alpha_pinball.png"),
        k_value=2,
        style="highlight",
    )
    _plot_task_curves(
        alphas,
        series_pin_k3_is,
        series_pin_k3_oos,
        title=r"Pinball loss vs. $\alpha$ ($t=3$)",
        out_path=os.path.join(args.out_dir, "period4_singlek_k3_tasks_vs_alpha_pinball.png"),
        k_value=3,
        style="highlight",
    )


if __name__ == "__main__":
    main()

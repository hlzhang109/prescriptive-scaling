#!/usr/bin/env python3
"""
Plot evaluation summaries for sigmoid frontier (pre-2025 train vs 2025 test).

Reads per-task summaries and bin CSVs produced by `scripts/evaluate/sigmoid_binned_mae.py`
and renders:
  - Grouped bars: ID vs OOD coverage error per task
  - Scatter: OOD vs ID coverage error with diagonal
  - Heatmaps: |hat_tau - tau| per bin (train and test-overlap), tasks on y-axis

Outputs go to --out_dir (default: outputs/evaluation/sigmoid/year_split/no_budget/plots).
"""

from __future__ import annotations

import argparse
import csv as _csv
import os
from typing import Dict, List, Optional, Tuple

import sys

# Import bootstrap from `scripts/` by putting the scripts dir on sys.path.
SCRIPTS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)
from _bootstrap import ensure_repo_root_on_path  # type: ignore

REPO_ROOT = ensure_repo_root_on_path(__file__)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt  # type: ignore
import matplotlib as mpl  # type: ignore
from matplotlib.colors import TwoSlopeNorm  # type: ignore

from skill_frontier.core.sigmoid import PERIOD4_SPLITS_SINGLE  # type: ignore
from skill_frontier.io.csv_utils import detect_date_col, parse_year_month  # type: ignore
from skill_frontier.io.output_paths import sanitize_task_name  # type: ignore
from skill_frontier.io.task_mappings import format_task_label  # type: ignore

import re

from skill_frontier.plotting.configs import eval_sigmoid as eval_sigmoid_cfg  # type: ignore
from skill_frontier.plotting.configs import mpl_rc as mpl_rc_cfg  # type: ignore
from skill_frontier.plotting.labels import BIN_UPPER_FLOPS_1E21_LABEL  # type: ignore

# Default x-axis label for bin upper-bound; can be overridden via CLI to support
# alternative x definitions (e.g. model size instead of FLOPs).
BIN_X_LABEL = BIN_UPPER_FLOPS_1E21_LABEL

# Tau used only for *signed* annotations/diagnostics (hat_tau - tau).
# Absolute error heatmap colors use the precomputed `abs_err` from the evaluation outputs.
PLOT_TAU = eval_sigmoid_cfg.PLOT_TAU


def _task_legacy_filename(task: str) -> str:
    """Legacy filename mapping used by older evaluation outputs."""
    return task.replace("/", "_").replace("\\", "_")


def _summary_scan_dir(eval_dir: str) -> str:
    summaries_dir = os.path.join(eval_dir, "summaries")
    return summaries_dir if os.path.isdir(summaries_dir) else eval_dir


def _bins_scan_dir(eval_dir: str) -> str:
    bins_dir = os.path.join(eval_dir, "bins")
    return bins_dir if os.path.isdir(bins_dir) else eval_dir


def _list_k_eval_dirs(base_dir: str) -> List[Tuple[int, str]]:
    """Return sorted [(k, path)] for subfolders named k<int> that exist under base_dir."""
    out: List[Tuple[int, str]] = []
    try:
        names = os.listdir(base_dir)
    except Exception:
        return out
    for name in names:
        m = re.fullmatch(r"k(\d+)", name)
        if not m:
            continue
        k = int(m.group(1))
        path = os.path.join(base_dir, name)
        if os.path.isdir(path):
            out.append((k, path))
    out.sort(key=lambda t: t[0])
    return out


def _find_summary_paths(eval_dir: str) -> List[str]:
    """Return list of per-task summary CSV paths (structured or legacy)."""
    scan_dir = _summary_scan_dir(eval_dir)
    try:
        names = os.listdir(scan_dir)
    except Exception:
        return []
    out: List[str] = []
    for fn in names:
        if fn.endswith("__summary.csv") or fn.endswith("_summary.csv"):
            out.append(os.path.join(scan_dir, fn))
    out.sort()
    return out


def _get_summary_path(eval_dir: str, task: str) -> Optional[str]:
    """Return path to a task summary CSV if it exists, else None."""
    task_clean = sanitize_task_name(task)
    task_legacy = _task_legacy_filename(task)

    candidates: List[str] = []
    summaries_dir = os.path.join(eval_dir, "summaries")
    if os.path.isdir(summaries_dir):
        candidates.extend(
            [
                os.path.join(summaries_dir, f"{task_clean}_summary.csv"),
                os.path.join(summaries_dir, f"{task_legacy}__summary.csv"),
                os.path.join(summaries_dir, f"{task}__summary.csv"),
            ]
        )
    candidates.extend(
        [
            os.path.join(eval_dir, f"{task_clean}_summary.csv"),
            os.path.join(eval_dir, f"{task_legacy}__summary.csv"),
            os.path.join(eval_dir, f"{task}__summary.csv"),
        ]
    )
    for path in candidates:
        if os.path.isfile(path):
            return path
    return None


def _find_bins_path(eval_dir: str, task: str, which: str) -> Optional[str]:
    """Return path to a bins CSV (structured or legacy) if it exists, else None."""
    task_clean = sanitize_task_name(task)
    task_legacy = _task_legacy_filename(task)
    era_candidates = ["train"] if which == "train" else ["test_fixed", "test_overlap", "test"]

    candidates_dirs: List[str] = []
    bins_dir = os.path.join(eval_dir, "bins")
    if os.path.isdir(bins_dir):
        candidates_dirs.append(bins_dir)
    candidates_dirs.append(eval_dir)

    for era in era_candidates:
        for d in candidates_dirs:
            candidates: List[str] = [
                os.path.join(d, f"{task_clean}_bins_{era}.csv"),
                os.path.join(d, f"{task_legacy}__bins_{era}.csv"),
                os.path.join(d, f"{task}__bins_{era}.csv"),
            ]
            # Historical outputs: some runs always used a *_bins_test_overlap filename.
            if which != "train" and era != "test_overlap":
                candidates.extend(
                    [
                        os.path.join(d, f"{task_clean}_bins_test_overlap.csv"),
                        os.path.join(d, f"{task_legacy}__bins_test_overlap.csv"),
                        os.path.join(d, f"{task}__bins_test_overlap.csv"),
                    ]
                )
            for path in candidates:
                if os.path.isfile(path):
                    return path
    return None


def _find_task_summaries(eval_dir: str) -> List[str]:
    return _find_summary_paths(eval_dir)


def _read_summary(path: str) -> Tuple[str, float, float, float, float]:
    with open(path, "r", newline="") as f:
        r = _csv.DictReader(f)
        row = next(r)
    task = row["task"]
    mae_is_macro = float(row["MAE_IS_macro"]) if row["MAE_IS_macro"] != "" else float("nan")
    mae_oos_macro = float(row["MAE_OOS_macro"]) if row["MAE_OOS_macro"] != "" else float("nan")
    mae_is_micro = float(row["MAE_IS_micro"]) if row["MAE_IS_micro"] != "" else float("nan")
    mae_oos_micro = float(row["MAE_OOS_micro"]) if row["MAE_OOS_micro"] != "" else float("nan")
    return task, mae_is_macro, mae_oos_macro, mae_is_micro, mae_oos_micro


def _detect_tasks(eval_dir: str) -> List[str]:
    tasks: List[str] = []
    for path in _find_task_summaries(eval_dir):
        try:
            task, _, _, _, _ = _read_summary(path)
        except Exception:
            continue
        tasks.append(task)
    tasks.sort()
    return tasks


def _detect_bbh_subtasks(eval_dir: str) -> List[str]:
    """Detect BBH subtasks (columns containing 'leaderboard_bbh_')."""
    tasks = _detect_tasks(eval_dir)
    return [t for t in tasks if "leaderboard_bbh_" in t]


def _detect_main_tasks(eval_dir: str) -> List[str]:
    """Detect main tasks (exclude BBH subtasks)."""
    tasks = _detect_tasks(eval_dir)
    return [t for t in tasks if "leaderboard_bbh_" not in t]


def _read_bins(eval_dir: str, task: str, which: str) -> List[Tuple[int, float]]:
    path = _find_bins_path(eval_dir, task, which)
    rows: List[Tuple[int, float]] = []
    if not path or not os.path.isfile(path):
        return rows
    with open(path, "r", newline="") as f:
        r = _csv.DictReader(f)
        for row in r:
            try:
                bid = int(row["bin_id"]) if row["bin_id"] != "" else -1
            except Exception:
                continue
            ae_str = row.get("abs_err", "")
            ae = float(ae_str) if (ae_str not in ("", "nan", "NaN")) else float("nan")
            rows.append((bid, ae))
    rows.sort(key=lambda t: t[0])
    return rows


def _read_bins_full(eval_dir: str, task: str, which: str) -> List[Tuple[int, float, float]]:
    """Return list of (bin_id, hat_tau, abs_err) for a task/which, ordered by bin_id.

    Falls back gracefully if CSV missing; returns empty list.
    """
    path = _find_bins_path(eval_dir, task, which)
    rows: List[Tuple[int, float, float]] = []
    if not path or not os.path.isfile(path):
        return rows
    with open(path, "r", newline="") as f:
        r = _csv.DictReader(f)
        for row in r:
            try:
                bid = int(row.get("bin_id", ""))
            except Exception:
                continue
            ht_s = row.get("hat_tau", "")
            ae_s = row.get("abs_err", "")
            ht = float(ht_s) if ht_s not in ("", "nan", "NaN") else float("nan")
            ae = float(ae_s) if ae_s not in ("", "nan", "NaN") else float("nan")
            rows.append((bid, ht, ae))
    rows.sort(key=lambda t: t[0])
    return rows


def _read_bin_z_hi(eval_dir: str, task: str, which: str) -> List[float]:
    # Reads z_hi (log10 x upper bound) per bin, ordered by bin_id
    path = _find_bins_path(eval_dir, task, which)
    vals: List[Tuple[int, float]] = []
    if not path or not os.path.isfile(path):
        return []
    with open(path, "r", newline="") as f:
        r = _csv.DictReader(f)
        for row in r:
            bid_s = row.get("bin_id", "")
            zhi_s = row.get("z_hi", "")
            if bid_s == "" or zhi_s == "":
                continue
            try:
                bid = int(bid_s)
                zhi = float(zhi_s)
            except Exception:
                continue
            vals.append((bid, zhi))
    vals.sort(key=lambda t: t[0])
    return [zhi for (_, zhi) in vals]


def _display_name(task: str) -> str:
    """Human-readable task label for plots.

    - Strips ' Raw' suffix.
    - For BBH subtasks, also strips the 'leaderboard_bbh_' prefix to save space.
    """
    return format_task_label(task)


def _plot_bars(eval_dir: str, out_dir: str, tasks: List[str], display_names: List[str], is_vals: np.ndarray, oos_vals: np.ndarray) -> None:
    import matplotlib.pyplot as plt  # type: ignore
    import matplotlib as mpl  # type: ignore
    import matplotlib.ticker as mticker  # type: ignore
    try:
        mpl.rcParams["font.family"] = mpl_rc_cfg.FONT_FAMILY
    except Exception:
        pass
    os.makedirs(out_dir, exist_ok=True)
    x = np.arange(len(tasks))
    w = eval_sigmoid_cfg.BARS_BAR_WIDTH
    fig, ax = plt.subplots(figsize=eval_sigmoid_cfg.BARS_FIGSIZE)
    ax.bar(x - w/2, is_vals, width=w, label="ID (train)")
    ax.bar(x + w/2, oos_vals, width=w, label="OOD (test)")
    ax.set_xticks(x)
    ax.set_xticklabels(
        display_names,
        rotation=eval_sigmoid_cfg.BARS_XTICK_ROTATION,
        ha="right",
        fontsize=eval_sigmoid_cfg.BARS_XTICK_LABELSIZE,
    )
    ax.set_ylabel("Binned Coverage MAE", fontweight="bold", fontsize=eval_sigmoid_cfg.BARS_YLABEL_FONTSIZE)
    ax.set_title(
        "Sigmoid Frontier — ID vs OOD coverage error",
        fontweight="bold",
        fontsize=eval_sigmoid_cfg.BARS_TITLE_FONTSIZE,
    )
    ax.legend(loc="best", fontsize=eval_sigmoid_cfg.BARS_LEGEND_FONTSIZE)
    fig.tight_layout(rect=eval_sigmoid_cfg.BARS_TIGHT_LAYOUT_RECT)
    fig.savefig(os.path.join(out_dir, "summary_bars_IS_OOS.png"), dpi=300, bbox_inches="tight")
    fig.savefig(os.path.join(out_dir, "summary_bars_IS_OOS.pdf"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def _plot_scatter(eval_dir: str, out_dir: str, display_names: List[str], is_vals: np.ndarray, oos_vals: np.ndarray) -> None:
    import matplotlib.pyplot as plt  # type: ignore
    import matplotlib as mpl  # type: ignore
    try:
        mpl.rcParams["font.family"] = mpl_rc_cfg.FONT_FAMILY
    except Exception:
        pass
    os.makedirs(out_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=eval_sigmoid_cfg.SCATTER_FIGSIZE)
    ax.scatter(
        is_vals,
        oos_vals,
        s=eval_sigmoid_cfg.SCATTER_POINT_SIZE,
        alpha=eval_sigmoid_cfg.SCATTER_POINT_ALPHA,
        color=eval_sigmoid_cfg.SCATTER_POINT_COLOR,
    )
    # Diagonal
    mn = float(np.nanmin([np.nanmin(is_vals), np.nanmin(oos_vals)]))
    mx = float(np.nanmax([np.nanmax(is_vals), np.nanmax(oos_vals)]))
    pad = eval_sigmoid_cfg.SCATTER_DIAG_PAD_FRAC * (mx - mn if mx > mn else 1.0)
    ax.plot(
        [mn - pad, mx + pad],
        [mn - pad, mx + pad],
        color=eval_sigmoid_cfg.SCATTER_DIAG_COLOR,
        linestyle=eval_sigmoid_cfg.SCATTER_DIAG_LINESTYLE,
        linewidth=eval_sigmoid_cfg.SCATTER_DIAG_LINEWIDTH,
    )
    # Labels
    ax.set_xlabel("ID coverage error", fontweight="bold", fontsize=eval_sigmoid_cfg.SCATTER_XLABEL_FONTSIZE)
    ax.set_ylabel("OOD coverage error", fontweight="bold", fontsize=eval_sigmoid_cfg.SCATTER_YLABEL_FONTSIZE)
    ax.set_title(
        "Sigmoid Frontier — OOD vs ID (coverage error)",
        fontweight="bold",
        fontsize=eval_sigmoid_cfg.SCATTER_TITLE_FONTSIZE,
    )
    # Optional: annotate points with task names
    for label, xi, yi in zip(display_names, is_vals, oos_vals):
        if not (np.isfinite(xi) and np.isfinite(yi)):
            continue
        ax.annotate(
            label,
            (xi, yi),
            textcoords="offset points",
            xytext=eval_sigmoid_cfg.SCATTER_ANNOT_XYTEXT,
            fontsize=eval_sigmoid_cfg.SCATTER_ANNOT_FONTSIZE,
        )
    fig.tight_layout(rect=eval_sigmoid_cfg.SCATTER_TIGHT_LAYOUT_RECT)
    fig.savefig(os.path.join(out_dir, "summary_scatter_OOS_vs_IS.png"), dpi=300, bbox_inches="tight")
    fig.savefig(os.path.join(out_dir, "summary_scatter_OOS_vs_IS.pdf"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def _plot_heatmap(eval_dir: str, out_dir: str, tasks: List[str], display_names: List[str], which: str, annotate_cells: bool = True) -> None:
    import matplotlib.pyplot as plt  # type: ignore
    import matplotlib as mpl  # type: ignore
    try:
        mpl.rcParams["font.family"] = mpl_rc_cfg.FONT_FAMILY
    except Exception:
        pass
    # collect per-task bin arrays (abs_err) and signed errors for annotations
    per_task: List[np.ndarray] = []
    per_task_signed: List[np.ndarray] = []
    zhi_per_task: List[List[float]] = []
    max_bins = 0
    for t in tasks:
        rows = _read_bins(eval_dir, t, which)
        rows_full = _read_bins_full(eval_dir, t, which)
        zhi = _read_bin_z_hi(eval_dir, t, which)
        if not rows:
            per_task.append(np.full((0,), np.nan))
            per_task_signed.append(np.full((0,), np.nan))
            zhi_per_task.append([])
            continue
        aerr = [ae for (_, ae) in rows]
        a = np.array(aerr, float)
        # Signed = hat_tau - tau (tau is only used for annotations).
        TAU = float(PLOT_TAU)
        signed_list: List[float] = []
        if rows_full:
            for (_, ht, _ae) in rows_full:
                if np.isfinite(ht):
                    signed_list.append(float(ht - TAU))
                else:
                    signed_list.append(float("nan"))
        else:
            signed_list = [float("nan")] * len(aerr)
        s = np.array(signed_list, float)
        per_task.append(a)
        per_task_signed.append(s)
        zhi_per_task.append(zhi)
        if a.shape[0] > max_bins:
            max_bins = a.shape[0]
    if max_bins == 0:
        return
    # pad to max_bins with NaN
    M = np.full((len(tasks), max_bins), np.nan, dtype=float)
    S = np.full((len(tasks), max_bins), np.nan, dtype=float)
    for i, a in enumerate(per_task):
        if a.size:
            M[i, :a.size] = a
    for i, s in enumerate(per_task_signed):
        if s.size:
            S[i, :s.size] = s
    # plot
    os.makedirs(out_dir, exist_ok=True)
    fig, ax = plt.subplots(
        figsize=(
            max(eval_sigmoid_cfg.HEATMAP_FIG_MIN_WIDTH, eval_sigmoid_cfg.HEATMAP_FIG_WIDTH_PER_BIN * max_bins),
            eval_sigmoid_cfg.HEATMAP_FIG_HEIGHT_PER_TASK * len(tasks) + eval_sigmoid_cfg.HEATMAP_FIG_HEIGHT_PAD,
        )
    )
    # Use a shared 'Blues' colormap for all heatmaps
    _cmap = plt.get_cmap(eval_sigmoid_cfg.HEATMAP_CMAP_NAME).copy()
    try:
        _cmap.set_bad(eval_sigmoid_cfg.HEATMAP_NAN_COLOR)
    except Exception:
        pass
    im = ax.imshow(
        np.ma.masked_invalid(M),
        aspect=eval_sigmoid_cfg.HEATMAP_IMSHOW_ASPECT,
        interpolation=eval_sigmoid_cfg.HEATMAP_IMSHOW_INTERPOLATION,
        cmap=_cmap,
    )
    ax.set_yticks(np.arange(len(tasks)))
    ax.set_yticklabels(display_names)
    ax.set_xticks(np.arange(max_bins))
    # Derive x tick labels from median z_hi across tasks for each bin index (10**z_hi)
    ticks_text: List[str] = []
    for j in range(max_bins):
        vals = []
        for zhi in zhi_per_task:
            if j < len(zhi):
                vals.append(zhi[j])
        if vals:
            zhi_med = float(np.median(np.array(vals, float)))
            flops_hi = 10.0 ** zhi_med
            # compact engineering/scientific display
            ticks_text.append(f"{flops_hi:.2e}")
        else:
            ticks_text.append("")
    ax.set_xticklabels(ticks_text, rotation=eval_sigmoid_cfg.HEATMAP_XTICK_ROTATION, ha="right")
    # Increase tick label padding and sizes to avoid overlaps with axis labels
    ax.tick_params(
        axis="x",
        labelsize=eval_sigmoid_cfg.HEATMAP_TICK_LABELSIZE,
        pad=eval_sigmoid_cfg.HEATMAP_TICK_PAD,
    )
    ax.tick_params(
        axis="y",
        labelsize=eval_sigmoid_cfg.HEATMAP_TICK_LABELSIZE,
        pad=eval_sigmoid_cfg.HEATMAP_TICK_PAD,
    )
    # Unified title for both train and test-overlap heatmaps; add padding to lift it away from the plot
    ax.set_title(
        r"ID vs. OOD Coverage Error $\hat{\tau}-\tau$",
        fontweight="bold",
        fontsize=eval_sigmoid_cfg.HEATMAP_TITLE_FONTSIZE,
        pad=eval_sigmoid_cfg.HEATMAP_TITLE_PAD,
    )
    ax.set_xlabel(BIN_X_LABEL, fontweight="bold", fontsize=eval_sigmoid_cfg.HEATMAP_XLABEL_FONTSIZE)
    ax.set_ylabel("Task", fontweight="bold", fontsize=eval_sigmoid_cfg.HEATMAP_YLABEL_FONTSIZE)
    ax.xaxis.labelpad = eval_sigmoid_cfg.HEATMAP_XAXIS_LABELPAD
    ax.yaxis.labelpad = eval_sigmoid_cfg.HEATMAP_YAXIS_LABELPAD
    fig.subplots_adjust(right=eval_sigmoid_cfg.HEATMAP_SUBPLOTS_ADJUST_RIGHT)
    cbar = fig.colorbar(im, ax=ax, pad=eval_sigmoid_cfg.HEATMAP_COLORBAR_PAD)
    cbar.ax.tick_params(labelsize=eval_sigmoid_cfg.HEATMAP_CBAR_TICK_LABELSIZE)
    # No colorbar label per request
    # Optional numeric annotation inside each cell
    if annotate_cells:
        # Use black/white depending on perceived luminance of the cell color
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                val = M[i, j]              # absolute error for color/threshold
                if not np.isfinite(val):
                    continue
                # Mark cells where the metric (val) exceeds 0.01 for coverage heatmaps
                if val >= eval_sigmoid_cfg.HEATMAP_ANNOT_THRESHOLD:
                    try:
                        rgba = im.cmap(im.norm(val))
                    except Exception:
                        # Fallback: assume linear 0..1
                        rgba = im.cmap(val)
                    r, g, b, _ = rgba
                    luminance = 0.2126 * r + 0.7152 * g + 0.0722 * b
                    txt_color = "white" if luminance < 0.5 else "black"
                    ax.text(
                        j,
                        i,
                        f"{val * 100.0:.1f}",
                        ha="center",
                        va="center",
                        fontsize=eval_sigmoid_cfg.HEATMAP_ANNOT_FONTSIZE,
                        fontweight="bold",
                        color=txt_color,
                    )
    fig.tight_layout(rect=eval_sigmoid_cfg.HEATMAP_TIGHT_LAYOUT_RECT)
    suffix = "train" if which == "train" else "test_overlap"
    fig.savefig(os.path.join(out_dir, f"heatmap_bins_{suffix}.png"), dpi=300, bbox_inches="tight")
    fig.savefig(os.path.join(out_dir, f"heatmap_bins_{suffix}.pdf"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def _prepare_matrix(eval_dir: str, tasks: List[str], which: str) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    # Build matrices: absolute error (for color) and signed error (for annotations).
    # Return (M_abs, S_signed, xticklabels)
    per_task: List[np.ndarray] = []
    per_task_signed: List[np.ndarray] = []
    zhi_per_task: List[List[float]] = []
    max_bins = 0
    for t in tasks:
        rows = _read_bins(eval_dir, t, which)
        rows_full = _read_bins_full(eval_dir, t, which)
        zhi = _read_bin_z_hi(eval_dir, t, which)
        if not rows:
            per_task.append(np.full((0,), np.nan))
            per_task_signed.append(np.full((0,), np.nan))
            zhi_per_task.append([])
            continue
        aerr = [ae for (_, ae) in rows]
        a = np.array(aerr, float)
        # Signed = hat_tau - tau (tau is only used for annotations).
        TAU = float(PLOT_TAU)
        signed_list: List[float] = []
        if rows_full:
            for (_, ht, _ae) in rows_full:
                if np.isfinite(ht):
                    signed_list.append(float(ht - TAU))
                else:
                    signed_list.append(float("nan"))
        else:
            signed_list = [float("nan")] * len(aerr)
        s = np.array(signed_list, float)
        per_task.append(a)
        per_task_signed.append(s)
        zhi_per_task.append(zhi)
        if a.shape[0] > max_bins:
            max_bins = a.shape[0]
    if max_bins == 0:
        return np.full((len(tasks), 0), np.nan), np.full((len(tasks), 0), np.nan), []
    M = np.full((len(tasks), max_bins), np.nan, dtype=float)
    S = np.full((len(tasks), max_bins), np.nan, dtype=float)
    for i, a in enumerate(per_task):
        if a.size:
            M[i, :a.size] = a
    for i, s in enumerate(per_task_signed):
        if s.size:
            S[i, :s.size] = s
    # xticks from median z_hi across tasks
    ticks_text: List[str] = []
    for j in range(max_bins):
        vals = []
        for zhi in zhi_per_task:
            if j < len(zhi):
                vals.append(zhi[j])
        if vals:
            zhi_med = float(np.median(np.array(vals, float)))
            flops_hi = 10.0 ** zhi_med
            if flops_hi > 0.0:
                # Scientific notation with mantissa × 10^{exponent}, LaTeX style
                s = f"{flops_hi:.2e}"  # e.g., '3.48e+01'
                mantissa, exp_str = s.split("e")
                exp_val = int(exp_str)
                ticks_text.append(rf"${mantissa}\times 10^{{{exp_val}}}$")
            else:
                ticks_text.append("")
        else:
            ticks_text.append("")
    return M, S, ticks_text


def _make_heatmap_cmap():
    """Return a shared heatmap colormap with a neutral NaN color."""
    import matplotlib.pyplot as plt  # type: ignore

    cmap = plt.get_cmap(eval_sigmoid_cfg.HEATMAP_CMAP_NAME).copy()
    try:
        cmap.set_bad(eval_sigmoid_cfg.HEATMAP_NAN_COLOR)
    except Exception:
        pass
    return cmap


def _annotate_signed_cells(
    ax,
    im,
    M_abs: np.ndarray,
    S_signed: np.ndarray,
    threshold_abs: float,
    fontsize: int,
) -> None:
    """Annotate heatmap cells with signed error (×100); fill all finite cells.

    Colors are based on M_abs (absolute error); displayed text prefers S_signed
    (hat_tau - tau). If S_signed is NaN, falls back to M_abs (positive).
    """
    if not np.isfinite(threshold_abs) or threshold_abs < 0.0:
        threshold_abs = 0.0
    for i in range(M_abs.shape[0]):
        for j in range(M_abs.shape[1]):
            val_abs = M_abs[i, j]
            signed = S_signed[i, j]
            if not np.isfinite(val_abs):
                continue
            use_signed = signed if np.isfinite(signed) else 0.0
            if abs(use_signed) < threshold_abs and threshold_abs > 0.0:
                continue
            disp = (use_signed if np.isfinite(signed) else val_abs) * 100.0
            rgba = im.cmap(im.norm(val_abs))
            r, g, b, _ = rgba
            luminance = 0.2126 * r + 0.7152 * g + 0.0722 * b
            txt_color = "white" if luminance < 0.5 else "black"
            ax.text(
                j,
                i,
                rf"\textbf{{{disp:.1f}}}",
                ha="center",
                va="center",
                fontsize=fontsize,
                color=txt_color,
            )


def _annotate_pinball_cells(
    ax,
    im,
    M: np.ndarray,
    threshold_display: float,
    fontsize: int,
) -> None:
    """Annotate cells with pinball loss × 1000 above a display threshold.

    Colors are based on M (per-bin pinball loss); annotations show
    loss * 1000 with one decimal place.
    """
    if not np.isfinite(threshold_display) or threshold_display <= 0.0:
        threshold_loss = 0.0
    else:
        threshold_loss = threshold_display / 1000.0
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            val = M[i, j]
            if not np.isfinite(val) or val < threshold_loss:
                continue
            disp = val * 1000.0
            rgba = im.cmap(im.norm(val))
            r, g, b, _ = rgba
            luminance = 0.2126 * r + 0.7152 * g + 0.0722 * b
            txt_color = "white" if luminance < 0.5 else "black"
            ax.text(
                j,
                i,
                rf"\textbf{{{disp:.1f}}}",
                ha="center",
                va="center",
                fontsize=fontsize,
                color=txt_color,
            )


def _plot_heatmap_grid_period4_singlek(base_dir: str, out_dir: str, annotate_cells: bool = True) -> None:
    import matplotlib.pyplot as plt  # type: ignore
    import matplotlib as mpl  # type: ignore
    try:
        mpl.rcParams["text.usetex"] = True
        mpl.rcParams["text.latex.preamble"] = mpl_rc_cfg.LATEX_PREAMBLE
        mpl.rcParams["font.family"] = mpl_rc_cfg.FONT_FAMILY
    except Exception:
        pass
    k_dirs = _list_k_eval_dirs(base_dir)
    if not k_dirs:
        return
    eval_dirs = [p for (_k, p) in k_dirs]
    # Tasks from first dir
    tasks = _detect_main_tasks(eval_dirs[0])
    display_names = [_display_name(t) for t in tasks]
    # Prepare matrices and xticks for 2 * len(k_dirs) panels
    panels = []  # list of tuples (M_abs, S_signed, xticks_text, which_label, k)
    all_vals = []
    for k, ed in k_dirs:
        for which, label in (("train", "ID"), ("test", "OOD")):
            M, S, xt = _prepare_matrix(ed, tasks, which)
            panels.append((M, S, xt, label, k))
            if M.size:
                all_vals.append(M[np.isfinite(M)])
    vmax = float(np.nanmax(np.concatenate(all_vals))) if all_vals else 1.0
    vmin = 0.0
    is_pinball_base = "evaluation_pinball" in os.path.abspath(base_dir)
    ncols = len(panels)
    # Allocate subplot widths proportional to the number of x-bins in each panel so
    # that each heatmap cell has the same physical width across panels.
    panel_bin_counts = [int(p[0].shape[1] or 6) for p in panels]
    total_bins = int(sum(panel_bin_counts)) if panel_bin_counts else 6
    fig_width = max(
        eval_sigmoid_cfg.P4_GRID_MIN_WIDTH,
        eval_sigmoid_cfg.P4_GRID_WIDTH_PER_COL * total_bins,
    )
    # Add a small width bump so we can afford a bit more inter-panel whitespace
    # without squeezing heatmap cells.
    fig_width *= 1.04
    # Figure with 2 * (#k) subplots in a row, plus a dedicated colorbar axis on the
    # right. We use per-panel width ratios so each heatmap cell has the same width
    # across all panels, even when bin counts differ by k.
    fig = plt.figure(figsize=(fig_width, eval_sigmoid_cfg.P4_GRID_HEIGHT))
    cbar_ratio = 0.9  # ~one bin-width, to accommodate large tick labels.
    # Increase blank space between panels (wspace is a fraction of average axis width).
    wspace = max(float(eval_sigmoid_cfg.P4_GRID_WSPACE), 0.05)
    gs = fig.add_gridspec(
        1,
        ncols + 1,
        width_ratios=[*panel_bin_counts, cbar_ratio],
        wspace=wspace,
    )
    axes = []
    for i in range(ncols):
        sharey = axes[0] if axes else None
        axes.append(fig.add_subplot(gs[0, i], sharey=sharey))
    cax = fig.add_subplot(gs[0, -1])
    # Shared heatmap colormap for all heatmaps
    _cmap = _make_heatmap_cmap()
    ims = []
    for ax_i, (ax, (M, S, xt, label, k)) in enumerate(zip(axes, panels)):
        # Add cell borders with subtle gridlines
        im = ax.imshow(np.ma.masked_invalid(M), aspect="auto", interpolation="nearest",
                       cmap=_cmap, vmin=vmin, vmax=vmax)
        ims.append(im)
        # Add gridlines between cells (imshow doesn't support edgecolors)
        ax.set_xticks(np.arange(M.shape[1] + 1) - 0.5, minor=True)
        ax.set_yticks(np.arange(M.shape[0] + 1) - 0.5, minor=True)
        ax.grid(
            which="minor",
            color=eval_sigmoid_cfg.P4_GRID_CELL_BORDER_COLOR,
            linewidth=eval_sigmoid_cfg.P4_GRID_CELL_BORDER_LINEWIDTH,
            linestyle=eval_sigmoid_cfg.P4_GRID_CELL_BORDER_LINESTYLE,
        )
        ax.tick_params(which='minor', size=0)  # Hide minor tick marks
        ax.set_yticks(np.arange(len(tasks)))
        if ax_i == 0:
            ax.set_yticklabels(display_names)
        else:
            ax.tick_params(axis="y", labelleft=False)
        ax.set_xticks(np.arange(M.shape[1]))
        ax.set_xticklabels(xt, rotation=30, ha="right")
        ax.tick_params(axis="x", labelsize=eval_sigmoid_cfg.P4_GRID_XTICK_LABELSIZE, pad=eval_sigmoid_cfg.P4_GRID_TICK_PAD)
        ax.tick_params(axis="y", labelsize=eval_sigmoid_cfg.P4_GRID_YTICK_LABELSIZE, pad=eval_sigmoid_cfg.P4_GRID_TICK_PAD)
        # Embedded badge (top-left), not bold, white rounded box (style of attached example)
        try:
            with mpl.rc_context({"text.usetex": True, "text.latex.preamble": mpl_rc_cfg.LATEX_PREAMBLE}):
                from matplotlib.offsetbox import AnchoredText  # type: ignore
                txt = fr"$t={k}\mid$ {label}"
                badge = AnchoredText(
                    txt,
                    loc="upper left",
                    prop=dict(
                        size=eval_sigmoid_cfg.P4_GRID_BADGE_FONTSIZE,
                        color=eval_sigmoid_cfg.P4_GRID_BADGE_COLOR,
                        fontweight=eval_sigmoid_cfg.P4_GRID_BADGE_WEIGHT,
                    ),
                )
        except Exception:
            from matplotlib.offsetbox import AnchoredText  # type: ignore
            txt = fr"$t={k}\mid$ {label}"
            badge = AnchoredText(
                txt,
                loc="upper left",
                prop=dict(
                    size=eval_sigmoid_cfg.P4_GRID_BADGE_FONTSIZE,
                    color=eval_sigmoid_cfg.P4_GRID_BADGE_COLOR,
                    fontweight=eval_sigmoid_cfg.P4_GRID_BADGE_WEIGHT,
                ),
            )
        try:
            badge.patch.set_facecolor(eval_sigmoid_cfg.P4_GRID_BADGE_FACE)
            badge.patch.set_alpha(eval_sigmoid_cfg.P4_GRID_BADGE_ALPHA)
            badge.patch.set_edgecolor(eval_sigmoid_cfg.P4_GRID_BADGE_EDGE)
            badge.patch.set_linewidth(eval_sigmoid_cfg.P4_GRID_BADGE_LINEWIDTH)
        except Exception:
            pass
        ax.add_artist(badge)
        # Cell annotation: pinball vs coverage depending on base_dir
        if annotate_cells:
            if is_pinball_base:
                # Display per-bin pinball loss × 1000 (annotate all finite cells)
                _annotate_pinball_cells(ax, im, M, threshold_display=0.0, fontsize=eval_sigmoid_cfg.P4_GRID_ANNOT_FONTSIZE)
            else:
                # Coverage grids: annotate signed coverage error (percentage points)
                _annotate_signed_cells(ax, im, M, S, threshold_abs=0.0, fontsize=eval_sigmoid_cfg.P4_GRID_ANNOT_FONTSIZE)
    # Shared labels: y once on the left; restore global x label for heatmap grid
    # Place the super y-label a bit further left to avoid overlap with large y-tick labels.
    fig.supylabel(
        r"\textbf{Task}",
        fontsize=eval_sigmoid_cfg.P4_GRID_SUPYLABEL_FONTSIZE,
        x=0.01,
    )
    fig.supxlabel(
        fr"\textbf{{{BIN_X_LABEL}}}",
        fontsize=eval_sigmoid_cfg.P4_GRID_SUPXLABEL_FONTSIZE,
        y=eval_sigmoid_cfg.P4_GRID_SUPXLABEL_Y,
    )
    # Single colorbar to the right of the subplot group.
    cbar = fig.colorbar(
        ims[-1],
        cax=cax,
    )
    cbar.ax.tick_params(labelsize=eval_sigmoid_cfg.P4_GRID_CBAR_TICK_LABELSIZE)
    # Title at top with larger font size
    if is_pinball_base:
        title = r"\textbf{ID vs. OOD Pinball Loss}"
    else:
        title = r"\textbf{ID vs. OOD Coverage Error $\hat{\tau}-\tau$}"
    fig.suptitle(title, fontsize=eval_sigmoid_cfg.P4_GRID_TITLE_FONTSIZE, y=eval_sigmoid_cfg.P4_GRID_TITLE_Y)
    fig.tight_layout(rect=eval_sigmoid_cfg.P4_GRID_TIGHT_LAYOUT_RECT)
    os.makedirs(out_dir, exist_ok=True)
    fig.savefig(os.path.join(out_dir, "heatmaps_period4_singlek_grid.png"), dpi=300, bbox_inches="tight")
    fig.savefig(os.path.join(out_dir, "heatmaps_period4_singlek_grid.pdf"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def _plot_heatmap_grid_period4_singlek_bbh(base_dir: str, out_dir: str, annotate_cells: bool = True) -> None:
    """ID-only heatmap grid for BBH subtasks (available k folders).

    Style is intentionally matched to `_plot_heatmap_grid_period4_singlek_oos_only_bbh`.
    """
    import matplotlib.pyplot as plt  # type: ignore
    import matplotlib as mpl  # type: ignore
    try:
        mpl.rcParams["text.usetex"] = True
        mpl.rcParams["text.latex.preamble"] = mpl_rc_cfg.LATEX_PREAMBLE
        mpl.rcParams["font.family"] = mpl_rc_cfg.FONT_FAMILY
    except Exception:
        pass
    k_dirs = _list_k_eval_dirs(base_dir)
    if not k_dirs:
        return
    eval_dirs = [p for (_k, p) in k_dirs]
    tasks = _detect_bbh_subtasks(eval_dirs[0])
    if not tasks:
        return
    display_names = [_display_name(t) for t in tasks]
    panels = []
    all_vals = []
    for k, ed in k_dirs:
        M, S, xt = _prepare_matrix(ed, tasks, "train")
        panels.append((M, S, xt, k))
        if M.size:
            all_vals.append(M[np.isfinite(M)])
    vmax = float(np.nanmax(np.concatenate(all_vals))) if all_vals else 1.0
    vmin = 0.0
    is_pinball_base = "evaluation_pinball" in os.path.abspath(base_dir)
    fig, axes = plt.subplots(
        1,
        len(panels),
        figsize=(
            max(eval_sigmoid_cfg.P4_BBH_GRID_MIN_WIDTH, eval_sigmoid_cfg.P4_BBH_GRID_WIDTH_PER_COL * sum(p[0].shape[1] or 6 for p in panels)),
            eval_sigmoid_cfg.P4_BBH_GRID_HEIGHT_PER_TASK * len(tasks) + eval_sigmoid_cfg.P4_BBH_GRID_HEIGHT_PAD,
        ),
        sharey=True,
    )
    axes = np.atleast_1d(axes)
    # Shared heatmap colormap for all heatmaps (aligned with main grid and OOS-only plots)
    _cmap = _make_heatmap_cmap()
    ims = []
    for ax, (M, S, xt, k) in zip(axes, panels):
        im = ax.imshow(np.ma.masked_invalid(M), aspect="auto", interpolation="nearest",
                       cmap=_cmap, vmin=vmin, vmax=vmax)
        ims.append(im)
        ax.set_xticks(np.arange(M.shape[1] + 1) - 0.5, minor=True)
        ax.set_yticks(np.arange(M.shape[0] + 1) - 0.5, minor=True)
        ax.grid(
            which="minor",
            color=eval_sigmoid_cfg.P4_GRID_CELL_BORDER_COLOR,
            linewidth=eval_sigmoid_cfg.P4_GRID_CELL_BORDER_LINEWIDTH,
            linestyle=eval_sigmoid_cfg.P4_GRID_CELL_BORDER_LINESTYLE,
        )
        ax.tick_params(which='minor', size=0)
        ax.set_yticks(np.arange(len(tasks)))
        ax.set_yticklabels(display_names)
        ax.set_xticks(np.arange(M.shape[1]))
        ax.set_xticklabels(xt, rotation=30, ha="right")
        ax.tick_params(axis="x", labelsize=eval_sigmoid_cfg.P4_BBH_GRID_XTICK_LABELSIZE, pad=eval_sigmoid_cfg.P4_BBH_GRID_TICK_PAD)
        ax.tick_params(axis="y", labelsize=eval_sigmoid_cfg.P4_BBH_GRID_YTICK_LABELSIZE, pad=eval_sigmoid_cfg.P4_BBH_GRID_TICK_PAD)
        try:
            with mpl.rc_context({"text.usetex": True, "text.latex.preamble": mpl_rc_cfg.LATEX_PREAMBLE}):
                from matplotlib.offsetbox import AnchoredText  # type: ignore
                txt = fr"$k={k}\mid$ ID"
                badge = AnchoredText(
                    txt,
                    loc="upper left",
                    prop=dict(
                        size=eval_sigmoid_cfg.P4_BBH_GRID_BADGE_FONTSIZE,
                        color=eval_sigmoid_cfg.P4_BBH_GRID_BADGE_COLOR,
                        fontweight=eval_sigmoid_cfg.P4_BBH_GRID_BADGE_WEIGHT,
                    ),
                )
        except Exception:
            from matplotlib.offsetbox import AnchoredText  # type: ignore
            txt = fr"$k={k}\mid$ ID"
            badge = AnchoredText(
                txt,
                loc="upper left",
                prop=dict(
                    size=eval_sigmoid_cfg.P4_BBH_GRID_BADGE_FONTSIZE,
                    color=eval_sigmoid_cfg.P4_BBH_GRID_BADGE_COLOR,
                    fontweight=eval_sigmoid_cfg.P4_BBH_GRID_BADGE_WEIGHT,
                ),
            )
        try:
            badge.patch.set_facecolor(eval_sigmoid_cfg.P4_BBH_GRID_BADGE_FACE)
            badge.patch.set_alpha(eval_sigmoid_cfg.P4_BBH_GRID_BADGE_ALPHA)
            badge.patch.set_edgecolor(eval_sigmoid_cfg.P4_BBH_GRID_BADGE_EDGE)
            badge.patch.set_linewidth(eval_sigmoid_cfg.P4_BBH_GRID_BADGE_LINEWIDTH)
        except Exception:
            pass
        ax.add_artist(badge)
        if annotate_cells:
            if is_pinball_base:
                _annotate_pinball_cells(ax, im, M, threshold_display=0.0, fontsize=eval_sigmoid_cfg.P4_BBH_GRID_ANNOT_FONTSIZE)
            else:
                _annotate_signed_cells(ax, im, M, S, threshold_abs=0.0, fontsize=eval_sigmoid_cfg.P4_BBH_GRID_ANNOT_FONTSIZE)
    fig.supylabel(r"\textbf{BBH Subtask}", fontsize=eval_sigmoid_cfg.P4_BBH_GRID_SUPYLABEL_FONTSIZE)
    fig.supxlabel(fr"\textbf{{{BIN_X_LABEL}}}", fontsize=eval_sigmoid_cfg.P4_BBH_GRID_SUPXLABEL_FONTSIZE)
    # Colorbar aligned to heatmap height
    cbar = fig.colorbar(
        ims[-1],
        ax=axes[-1],
        location="right",
        pad=eval_sigmoid_cfg.P4_BBH_GRID_CBAR_PAD,
        fraction=eval_sigmoid_cfg.P4_BBH_GRID_CBAR_FRACTION,
        aspect=eval_sigmoid_cfg.P4_BBH_GRID_CBAR_ASPECT,
    )
    cbar.ax.tick_params(labelsize=eval_sigmoid_cfg.P4_BBH_GRID_CBAR_TICK_LABELSIZE)
    if is_pinball_base:
        title = r"\textbf{ID Pinball Loss (BBH subtasks)}"
    else:
        title = r"\textbf{ID Coverage Error $|\hat{\tau} - \tau|$ (BBH subtasks)}"
    fig.suptitle(title, fontsize=eval_sigmoid_cfg.P4_BBH_GRID_TITLE_FONTSIZE, y=eval_sigmoid_cfg.P4_BBH_GRID_TITLE_Y)
    fig.subplots_adjust(wspace=eval_sigmoid_cfg.P4_BBH_GRID_WSPACE)
    fig.tight_layout(rect=eval_sigmoid_cfg.P4_BBH_GRID_TIGHT_LAYOUT_RECT)
    os.makedirs(out_dir, exist_ok=True)
    fig.savefig(os.path.join(out_dir, "heatmaps_period4_singlek_grid_bbh.png"), dpi=300, bbox_inches="tight")
    fig.savefig(os.path.join(out_dir, "heatmaps_period4_singlek_grid_bbh.pdf"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def _plot_heatmap_grid_period4_singlek_oos_only(base_dir: str, out_dir: str, annotate_cells: bool = True) -> None:
    """Create a smaller 1xK grid with only OOD heatmaps for available k folders."""
    import matplotlib.pyplot as plt  # type: ignore
    import matplotlib as mpl  # type: ignore
    try:
        mpl.rcParams["font.family"] = mpl_rc_cfg.FONT_FAMILY
    except Exception:
        pass
    k_dirs = _list_k_eval_dirs(base_dir)
    if not k_dirs:
        return
    eval_dirs = [p for (_k, p) in k_dirs]
    # Tasks from first dir
    tasks = _detect_main_tasks(eval_dirs[0])
    display_names = [_display_name(t) for t in tasks]
    # Prepare matrices and xticks for OOS panels only
    panels = []  # list of tuples (M_abs, S_signed, xticks_text, k)
    all_vals = []
    is_pinball_base = "evaluation_pinball" in os.path.abspath(base_dir)
    for k, ed in k_dirs:
        # Only OOS panels
        M_abs, S_signed, xt = _prepare_matrix(ed, tasks, "test")
        panels.append((M_abs, S_signed, xt, k))
        if M_abs.size:
            all_vals.append(M_abs[np.isfinite(M_abs)])
    if all_vals:
        concat = np.concatenate(all_vals)
        vmax = float(np.nanmax(concat))
        if not np.isfinite(vmax) or vmax <= 0.0:
            vmax = 1.0
    else:
        vmax = 1.0
    vmin = 0.0
    # Figure with K subplots in a row, shared y. Width ratios are proportional
    # to the number of bins in each panel so that individual heatmap cells have
    # (approximately) the same width across subplots.
    width_ratios = [
        max(M_abs.shape[1], 1) for (M_abs, _S, _xt, _k) in panels
    ]
    total_cols = float(sum(width_ratios)) if width_ratios else 1.0
    fig_width = max(eval_sigmoid_cfg.P4_OOS_ONLY_GRID_MIN_WIDTH, eval_sigmoid_cfg.P4_OOS_ONLY_GRID_WIDTH_PER_COL * total_cols)
    fig, axes = plt.subplots(
        1,
        len(panels),
        figsize=(fig_width, eval_sigmoid_cfg.P4_OOS_ONLY_GRID_HEIGHT),
        sharey=True,
        gridspec_kw={"width_ratios": width_ratios},
    )
    axes = np.atleast_1d(axes)
    # Use the shared heatmap colormap for all OOS heatmaps
    _cmap = _make_heatmap_cmap()
    ims = []
    for ax, (M_abs, S_signed, xt, k) in zip(axes, panels):
        # Add cell borders with subtle gridlines
        data = M_abs
        im = ax.imshow(
            np.ma.masked_invalid(data),
            aspect="auto",
            interpolation="nearest",
            cmap=_cmap,
            vmin=vmin,
            vmax=vmax,
        )
        ims.append(im)
        # Add gridlines between cells (imshow doesn't support edgecolors)
        ax.set_xticks(np.arange(data.shape[1] + 1) - 0.5, minor=True)
        ax.set_yticks(np.arange(data.shape[0] + 1) - 0.5, minor=True)
        ax.grid(
            which="minor",
            color=eval_sigmoid_cfg.P4_GRID_CELL_BORDER_COLOR,
            linewidth=eval_sigmoid_cfg.P4_GRID_CELL_BORDER_LINEWIDTH,
            linestyle=eval_sigmoid_cfg.P4_GRID_CELL_BORDER_LINESTYLE,
        )
        ax.tick_params(which='minor', size=0)  # Hide minor tick marks
        ax.set_yticks(np.arange(len(tasks)))
        ax.set_yticklabels(display_names)
        ax.set_xticks(np.arange(data.shape[1]))
        ax.set_xticklabels(xt, rotation=30, ha="right")
        # Larger font sizes for axis tick labels
        ax.tick_params(axis="x", labelsize=eval_sigmoid_cfg.P4_OOS_ONLY_GRID_XTICK_LABELSIZE, pad=eval_sigmoid_cfg.P4_OOS_ONLY_GRID_TICK_PAD)
        ax.tick_params(axis="y", labelsize=eval_sigmoid_cfg.P4_OOS_ONLY_GRID_YTICK_LABELSIZE, pad=eval_sigmoid_cfg.P4_OOS_ONLY_GRID_TICK_PAD)
        # Embedded badge (top-left)
        try:
            with mpl.rc_context({"text.usetex": True, "text.latex.preamble": mpl_rc_cfg.LATEX_PREAMBLE}):
                from matplotlib.offsetbox import AnchoredText  # type: ignore
                txt = fr"$k={k}\mid$ OOD"
                badge = AnchoredText(
                    txt,
                    loc="upper left",
                    prop=dict(
                        size=eval_sigmoid_cfg.P4_OOS_ONLY_GRID_BADGE_FONTSIZE,
                        color=eval_sigmoid_cfg.P4_OOS_ONLY_GRID_BADGE_COLOR,
                        fontweight=eval_sigmoid_cfg.P4_OOS_ONLY_GRID_BADGE_WEIGHT,
                    ),
                )
        except Exception:
            from matplotlib.offsetbox import AnchoredText  # type: ignore
            txt = fr"$k={k}\mid$ OOD"
            badge = AnchoredText(
                txt,
                loc="upper left",
                prop=dict(
                    size=eval_sigmoid_cfg.P4_OOS_ONLY_GRID_BADGE_FONTSIZE,
                    color=eval_sigmoid_cfg.P4_OOS_ONLY_GRID_BADGE_COLOR,
                    fontweight=eval_sigmoid_cfg.P4_OOS_ONLY_GRID_BADGE_WEIGHT,
                ),
            )
        try:
            badge.patch.set_facecolor(eval_sigmoid_cfg.P4_OOS_ONLY_GRID_BADGE_FACE)
            badge.patch.set_alpha(eval_sigmoid_cfg.P4_OOS_ONLY_GRID_BADGE_ALPHA)
            badge.patch.set_edgecolor(eval_sigmoid_cfg.P4_OOS_ONLY_GRID_BADGE_EDGE)
            badge.patch.set_linewidth(eval_sigmoid_cfg.P4_OOS_ONLY_GRID_BADGE_LINEWIDTH)
        except Exception:
            pass
        ax.add_artist(badge)
        # Cell annotation: pinball vs coverage depending on base_dir.
        if annotate_cells:
            if is_pinball_base:
                # Per-bin pinball loss × 1000; annotate all finite cells
                _annotate_pinball_cells(ax, im, M_abs, threshold_display=0.0, fontsize=eval_sigmoid_cfg.P4_OOS_ONLY_GRID_ANNOT_FONTSIZE)
            else:
                # Coverage grids: signed coverage error (percentage points)
                _annotate_signed_cells(ax, im, M_abs, S_signed, threshold_abs=0.0, fontsize=eval_sigmoid_cfg.P4_OOS_ONLY_GRID_ANNOT_FONTSIZE)
    # Shared labels: y once on the left; restore global x label for heatmap grid
    fig.supylabel(r"\textbf{Task}", fontsize=eval_sigmoid_cfg.P4_OOS_ONLY_GRID_SUPYLABEL_FONTSIZE)
    fig.supxlabel(fr"\textbf{{{BIN_X_LABEL}}}", fontsize=eval_sigmoid_cfg.P4_OOS_ONLY_GRID_SUPXLABEL_FONTSIZE)
    # Single colorbar to the right of the rightmost subplot - match heatmap height exactly
    cbar = fig.colorbar(
        ims[-1],
        ax=axes[-1],
        location="right",
        pad=eval_sigmoid_cfg.P4_OOS_ONLY_GRID_CBAR_PAD,
        fraction=eval_sigmoid_cfg.P4_OOS_ONLY_GRID_CBAR_FRACTION,
        aspect=eval_sigmoid_cfg.P4_OOS_ONLY_GRID_CBAR_ASPECT,
    )
    cbar.ax.tick_params(labelsize=eval_sigmoid_cfg.P4_OOS_ONLY_GRID_CBAR_TICK_LABELSIZE)
    # Title at top with larger font size
    if is_pinball_base:
        fig.suptitle(
            r"\textbf{OOD Pinball Loss}",
            fontsize=eval_sigmoid_cfg.P4_OOS_ONLY_GRID_TITLE_FONTSIZE,
            y=eval_sigmoid_cfg.P4_OOS_ONLY_GRID_TITLE_Y,
        )
    else:
        fig.suptitle(
            r"\textbf{OOD Coverage Error $\hat{\tau} - \tau$}",
            fontsize=eval_sigmoid_cfg.P4_OOS_ONLY_GRID_TITLE_FONTSIZE,
            y=eval_sigmoid_cfg.P4_OOS_ONLY_GRID_TITLE_Y,
        )
    # Reduce spacing between adjacent subplots
    fig.subplots_adjust(wspace=eval_sigmoid_cfg.P4_OOS_ONLY_GRID_WSPACE)
    fig.tight_layout(rect=eval_sigmoid_cfg.P4_OOS_ONLY_GRID_TIGHT_LAYOUT_RECT)
    os.makedirs(out_dir, exist_ok=True)
    fig.savefig(os.path.join(out_dir, "heatmaps_period4_singlek_oos.png"), dpi=300, bbox_inches="tight")
    fig.savefig(os.path.join(out_dir, "heatmaps_period4_singlek_oos.pdf"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def _plot_heatmap_grid_period4_singlek_oos_only_bbh(base_dir: str, out_dir: str, annotate_cells: bool = True) -> None:
    """OOD-only heatmap grid for BBH subtasks (available k folders)."""
    import matplotlib.pyplot as plt  # type: ignore
    import matplotlib as mpl  # type: ignore
    try:
        mpl.rcParams["font.family"] = mpl_rc_cfg.FONT_FAMILY
    except Exception:
        pass
    k_dirs = _list_k_eval_dirs(base_dir)
    if not k_dirs:
        return
    eval_dirs = [p for (_k, p) in k_dirs]
    tasks = _detect_bbh_subtasks(eval_dirs[0])
    if not tasks:
        return
    display_names = [_display_name(t) for t in tasks]
    panels = []
    all_vals = []
    for k, ed in k_dirs:
        M, S, xt = _prepare_matrix(ed, tasks, "test")
        panels.append((M, S, xt, k))
        if M.size:
            all_vals.append(M[np.isfinite(M)])
    vmax = float(np.nanmax(np.concatenate(all_vals))) if all_vals else 1.0
    vmin = 0.0
    is_pinball_base = "evaluation_pinball" in os.path.abspath(base_dir)
    fig, axes = plt.subplots(
        1,
        len(panels),
        figsize=(
            max(eval_sigmoid_cfg.P4_BBH_GRID_MIN_WIDTH, eval_sigmoid_cfg.P4_BBH_GRID_WIDTH_PER_COL * sum(p[0].shape[1] or 6 for p in panels)),
            eval_sigmoid_cfg.P4_BBH_GRID_HEIGHT_PER_TASK * len(tasks) + eval_sigmoid_cfg.P4_BBH_GRID_HEIGHT_PAD,
        ),
        sharey=True,
    )
    axes = np.atleast_1d(axes)
    # Shared heatmap colormap for all BBH OOS heatmaps (consistent with main OOS-only plots)
    _cmap = _make_heatmap_cmap()
    ims = []
    for ax, (M, S, xt, k) in zip(axes, panels):
        im = ax.imshow(np.ma.masked_invalid(M), aspect="auto", interpolation="nearest",
                       cmap=_cmap, vmin=vmin, vmax=vmax)
        ims.append(im)
        ax.set_xticks(np.arange(M.shape[1] + 1) - 0.5, minor=True)
        ax.set_yticks(np.arange(M.shape[0] + 1) - 0.5, minor=True)
        ax.grid(
            which="minor",
            color=eval_sigmoid_cfg.P4_GRID_CELL_BORDER_COLOR,
            linewidth=eval_sigmoid_cfg.P4_GRID_CELL_BORDER_LINEWIDTH,
            linestyle=eval_sigmoid_cfg.P4_GRID_CELL_BORDER_LINESTYLE,
        )
        ax.tick_params(which='minor', size=0)
        ax.set_yticks(np.arange(len(tasks)))
        ax.set_yticklabels(display_names)
        ax.set_xticks(np.arange(M.shape[1]))
        ax.set_xticklabels(xt, rotation=30, ha="right")
        ax.tick_params(axis="x", labelsize=eval_sigmoid_cfg.P4_BBH_GRID_XTICK_LABELSIZE, pad=eval_sigmoid_cfg.P4_BBH_GRID_TICK_PAD)
        ax.tick_params(axis="y", labelsize=eval_sigmoid_cfg.P4_BBH_GRID_YTICK_LABELSIZE, pad=eval_sigmoid_cfg.P4_BBH_GRID_TICK_PAD)
        try:
            with mpl.rc_context({"text.usetex": True, "text.latex.preamble": mpl_rc_cfg.LATEX_PREAMBLE}):
                from matplotlib.offsetbox import AnchoredText  # type: ignore
                txt = fr"$k={k}\mid$ OOD"
                badge = AnchoredText(
                    txt,
                    loc="upper left",
                    prop=dict(
                        size=eval_sigmoid_cfg.P4_BBH_GRID_BADGE_FONTSIZE,
                        color=eval_sigmoid_cfg.P4_BBH_GRID_BADGE_COLOR,
                        fontweight=eval_sigmoid_cfg.P4_BBH_GRID_BADGE_WEIGHT,
                    ),
                )
        except Exception:
            from matplotlib.offsetbox import AnchoredText  # type: ignore
            txt = fr"$k={k}\mid$ OOD"
            badge = AnchoredText(
                txt,
                loc="upper left",
                prop=dict(
                    size=eval_sigmoid_cfg.P4_BBH_GRID_BADGE_FONTSIZE,
                    color=eval_sigmoid_cfg.P4_BBH_GRID_BADGE_COLOR,
                    fontweight=eval_sigmoid_cfg.P4_BBH_GRID_BADGE_WEIGHT,
                ),
            )
        try:
            badge.patch.set_facecolor(eval_sigmoid_cfg.P4_BBH_GRID_BADGE_FACE)
            badge.patch.set_alpha(eval_sigmoid_cfg.P4_BBH_GRID_BADGE_ALPHA)
            badge.patch.set_edgecolor(eval_sigmoid_cfg.P4_BBH_GRID_BADGE_EDGE)
            badge.patch.set_linewidth(eval_sigmoid_cfg.P4_BBH_GRID_BADGE_LINEWIDTH)
        except Exception:
            pass
        ax.add_artist(badge)
        if annotate_cells:
            if is_pinball_base:
                _annotate_pinball_cells(ax, im, M, threshold_display=0.0, fontsize=eval_sigmoid_cfg.P4_BBH_GRID_ANNOT_FONTSIZE)
            else:
                _annotate_signed_cells(ax, im, M, S, threshold_abs=0.0, fontsize=eval_sigmoid_cfg.P4_BBH_GRID_ANNOT_FONTSIZE)
    fig.supylabel(r"\textbf{BBH Subtask}", fontsize=eval_sigmoid_cfg.P4_BBH_GRID_SUPYLABEL_FONTSIZE)
    fig.supxlabel(fr"\textbf{{{BIN_X_LABEL}}}", fontsize=eval_sigmoid_cfg.P4_BBH_GRID_SUPXLABEL_FONTSIZE)
    cbar = fig.colorbar(
        ims[-1],
        ax=axes[-1],
        location="right",
        pad=eval_sigmoid_cfg.P4_BBH_GRID_CBAR_PAD,
        fraction=eval_sigmoid_cfg.P4_BBH_GRID_CBAR_FRACTION,
        aspect=eval_sigmoid_cfg.P4_BBH_GRID_CBAR_ASPECT,
    )
    cbar.ax.tick_params(labelsize=eval_sigmoid_cfg.P4_BBH_GRID_CBAR_TICK_LABELSIZE)
    if is_pinball_base:
        fig.suptitle(
            r"\textbf{OOD Pinball Loss (BBH subtasks)}",
            fontsize=eval_sigmoid_cfg.P4_BBH_GRID_TITLE_FONTSIZE,
            y=eval_sigmoid_cfg.P4_BBH_GRID_TITLE_Y,
        )
    else:
        fig.suptitle(
            r"\textbf{OOD Coverage Error $|\hat{\tau} - \tau|$ (BBH subtasks)}",
            fontsize=eval_sigmoid_cfg.P4_BBH_GRID_TITLE_FONTSIZE,
            y=eval_sigmoid_cfg.P4_BBH_GRID_TITLE_Y,
        )
    fig.subplots_adjust(wspace=eval_sigmoid_cfg.P4_BBH_GRID_WSPACE)
    fig.tight_layout(rect=eval_sigmoid_cfg.P4_BBH_GRID_TIGHT_LAYOUT_RECT)
    os.makedirs(out_dir, exist_ok=True)
    fig.savefig(os.path.join(out_dir, "heatmaps_period4_singlek_oos_bbh.png"), dpi=300, bbox_inches="tight")
    fig.savefig(os.path.join(out_dir, "heatmaps_period4_singlek_oos_bbh.pdf"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def _plot_bars_grid_period4_singlek(base_dir: str, out_dir: str) -> None:
    import matplotlib.pyplot as plt  # type: ignore
    import matplotlib as mpl  # type: ignore
    try:
        mpl.rcParams["font.family"] = mpl_rc_cfg.FONT_FAMILY
    except Exception:
        pass
    k_dirs = _list_k_eval_dirs(base_dir)
    if not k_dirs:
        return
    tasks = _detect_main_tasks(k_dirs[0][1])
    display_names = [_display_name(t) for t in tasks]

    # Collect values and determine common y-limit
    panels = []  # list of dicts with keys: is_vals, oos_vals
    y_max = 0.0
    for _k, ed in k_dirs:
        is_vals = []
        oos_vals = []
        for t in tasks:
            path = _get_summary_path(ed, t)
            if not path:
                is_macro = float("nan")
                oos_macro = float("nan")
            else:
                _t, is_macro, oos_macro, _, _ = _read_summary(path)
            is_vals.append(is_macro)
            oos_vals.append(oos_macro)
        is_arr = np.array(is_vals, float)
        oos_arr = np.array(oos_vals, float)
        panels.append({"is": is_arr, "oos": oos_arr})
        m = float(np.nanmax([np.nanmax(is_arr), np.nanmax(oos_arr)]))
        if np.isfinite(m):
            y_max = max(y_max, m)
    if y_max <= 0:
        y_max = 0.1

    # Plot K subplots in a row, shared y
    # Slightly reduced height and shared y-axis; increase width
    ncols = len(k_dirs)
    fig_width = max(
        eval_sigmoid_cfg.P4_BARS_GRID_BASE_WIDTH,
        (eval_sigmoid_cfg.P4_BARS_GRID_BASE_WIDTH * ncols) / eval_sigmoid_cfg.P4_BARS_GRID_WIDTH_DIVISOR,
    )
    fig, axes = plt.subplots(1, ncols, figsize=(fig_width, eval_sigmoid_cfg.P4_BARS_GRID_HEIGHT), sharey=True)
    axes = np.atleast_1d(axes)
    w = eval_sigmoid_cfg.P4_BARS_GRID_BAR_WIDTH
    x = np.arange(len(tasks))

    # Define color palette: teal for ID, coral for OOD
    color_is = eval_sigmoid_cfg.P4_BARS_GRID_COLOR_IS
    color_oos = eval_sigmoid_cfg.P4_BARS_GRID_COLOR_OOS

    for idx, ((k, _ed), panel, ax) in enumerate(zip(k_dirs, panels, axes), start=1):
        is_arr = panel["is"]
        oos_arr = panel["oos"]
        bars_is = ax.bar(x - w/2, is_arr, width=w, label="ID (train)", color=color_is, zorder=3)
        bars_oos = ax.bar(x + w/2, oos_arr, width=w, label="OOD (test)", color=color_oos, zorder=3)

        # Set explicit x-limits with padding to keep bars inside subplot frame
        # Bars at x=0 and x=len(tasks)-1; with width w, they extend ±w/2 from center
        # Add small padding (e.g., 0.5) beyond the bar edges
        ax.set_xlim(
            eval_sigmoid_cfg.P4_BARS_GRID_XLIM_LEFT,
            len(tasks) - eval_sigmoid_cfg.P4_BARS_GRID_XLIM_RIGHT_PAD,
        )

        # Vertical gradient background (light gray at top to white at bottom) - after setting limits
        try:
            from matplotlib.colors import LinearSegmentedColormap
            grad_data = np.linspace(0, 1, 256).reshape(256, 1)
            grad_cmap = LinearSegmentedColormap.from_list(
                eval_sigmoid_cfg.P4_BARS_BG_GRAD_NAME,
                [eval_sigmoid_cfg.P4_BARS_BG_GRAD_TOP_COLOR, eval_sigmoid_cfg.P4_BARS_BG_GRAD_BOTTOM_COLOR],
            )
            # Use the explicit limits we just set
            xlims = ax.get_xlim()
            ax.imshow(
                grad_data,
                extent=[xlims[0], xlims[1], 0, y_max * eval_sigmoid_cfg.P4_BARS_GRID_YLIM_MULT],
                aspect="auto",
                cmap=grad_cmap,
                alpha=eval_sigmoid_cfg.P4_BARS_BG_GRAD_ALPHA,
                zorder=eval_sigmoid_cfg.P4_BARS_BG_GRAD_ZORDER,
                interpolation=eval_sigmoid_cfg.P4_BARS_BG_GRAD_INTERPOLATION,
            )
        except Exception:
            ax.set_facecolor(eval_sigmoid_cfg.P4_BARS_BG_GRAD_TOP_COLOR)

        # Apply gradient fills to bars (darker at bottom, lighter at top)
        try:
            from matplotlib.colors import LinearSegmentedColormap
            import matplotlib.patches as mpatches
            for bar, color in [(bars_is, color_is), (bars_oos, color_oos)]:
                for rect in bar:
                    height = rect.get_height()
                    if height <= 0 or not np.isfinite(height):
                        continue
                    # Create gradient colormap from darker to lighter
                    from matplotlib.colors import to_rgb
                    rgb = to_rgb(color)
                    darker = tuple(c * eval_sigmoid_cfg.P4_BARS_BAR_GRAD_DARKEN_FACTOR for c in rgb)
                    grad_cmap = LinearSegmentedColormap.from_list(eval_sigmoid_cfg.P4_BARS_BAR_GRAD_NAME, [darker, color])
                    # Create gradient patch
                    grad = grad_cmap(np.linspace(0, 1, 256).reshape(256, 1))
                    im = ax.imshow(grad, extent=[rect.get_x(), rect.get_x() + rect.get_width(),
                                                  0, height], aspect='auto', zorder=eval_sigmoid_cfg.P4_BARS_BAR_GRAD_ZORDER, interpolation=eval_sigmoid_cfg.P4_BARS_BAR_GRAD_INTERPOLATION)
                    im.set_clip_path(rect)
        except Exception:
            pass
        ax.set_xticks(x)
        ax.set_xticklabels(display_names, rotation=eval_sigmoid_cfg.P4_BARS_GRID_XTICK_ROTATION, ha="right", fontsize=(fig_width / 2))
        ax.set_ylim(0.0, y_max * eval_sigmoid_cfg.P4_BARS_GRID_YLIM_MULT)
        # Embedded badge matching triptych style and LaTeX rendering (set to 24 pt)
        try:
            with mpl.rc_context({"text.usetex": True, "text.latex.preamble": r"\usepackage{amsmath}"}):
                label_txt = fr"$k={k}$"
                from matplotlib.offsetbox import AnchoredText  # type: ignore
                badge = AnchoredText(
                    label_txt,
                    loc="upper left",
                    prop=dict(size=eval_sigmoid_cfg.P4_BARS_GRID_BADGE_FONTSIZE, color=eval_sigmoid_cfg.P4_BARS_GRID_BADGE_COLOR),
                )
        except Exception:
            label_txt = fr"$k={k}$"
            from matplotlib.offsetbox import AnchoredText  # type: ignore
            badge = AnchoredText(
                label_txt,
                loc="upper left",
                prop=dict(size=eval_sigmoid_cfg.P4_BARS_GRID_BADGE_FONTSIZE, color=eval_sigmoid_cfg.P4_BARS_GRID_BADGE_COLOR),
            )
        try:
            badge.patch.set_facecolor(eval_sigmoid_cfg.P4_BARS_GRID_BADGE_FACE)
            badge.patch.set_alpha(eval_sigmoid_cfg.P4_BARS_GRID_BADGE_ALPHA)
            badge.patch.set_edgecolor(eval_sigmoid_cfg.P4_BARS_GRID_BADGE_EDGE)
            badge.patch.set_linewidth(eval_sigmoid_cfg.P4_BARS_GRID_BADGE_LINEWIDTH)
        except Exception:
            pass
        ax.add_artist(badge)
        if idx == 1:
            ax.set_ylabel("MAE", fontweight="bold", fontsize=eval_sigmoid_cfg.P4_BARS_GRID_YLABEL_FONTSIZE)
        # No x-axis label on the bars grid (keep tick labels only)
        # Legend only on the last axis
        if idx == len(axes):
            leg = ax.legend(loc="upper right", fontsize=eval_sigmoid_cfg.P4_BARS_GRID_LEGEND_FONTSIZE)
            if leg and leg.get_title():
                leg.get_title().set_fontweight("bold")
        # Grid styling similar to triptych (y-grid with major/minor) - softer opacity
        try:
            import matplotlib.ticker as mticker
            ax.yaxis.set_minor_locator(mticker.AutoMinorLocator(2))
        except Exception:
            pass
        ax.yaxis.grid(
            True,
            which="major",
            linestyle=eval_sigmoid_cfg.P4_BARS_GRID_MAJOR_GRID_LINESTYLE,
            linewidth=eval_sigmoid_cfg.P4_BARS_GRID_MAJOR_GRID_LINEWIDTH,
            color=eval_sigmoid_cfg.P4_BARS_GRID_MAJOR_GRID_COLOR,
            alpha=eval_sigmoid_cfg.P4_BARS_GRID_MAJOR_GRID_ALPHA,
            zorder=eval_sigmoid_cfg.P4_BARS_GRID_GRID_ZORDER,
        )
        ax.yaxis.grid(
            True,
            which="minor",
            linestyle=eval_sigmoid_cfg.P4_BARS_GRID_MINOR_GRID_LINESTYLE,
            linewidth=eval_sigmoid_cfg.P4_BARS_GRID_MINOR_GRID_LINEWIDTH,
            color=eval_sigmoid_cfg.P4_BARS_GRID_MINOR_GRID_COLOR,
            alpha=eval_sigmoid_cfg.P4_BARS_GRID_MINOR_GRID_ALPHA,
            zorder=eval_sigmoid_cfg.P4_BARS_GRID_GRID_ZORDER,
        )
    # Set the title font size for the bars grid - move closer to subplots
    fig.suptitle(
        eval_sigmoid_cfg.P4_BARS_GRID_TITLE,
        fontsize=eval_sigmoid_cfg.P4_BARS_GRID_TITLE_FONTSIZE,
        fontweight="bold",
        y=eval_sigmoid_cfg.P4_BARS_GRID_TITLE_Y,
    )
    # Increase horizontal spacing between subplots
    fig.subplots_adjust(wspace=eval_sigmoid_cfg.P4_BARS_GRID_WSPACE)
    fig.tight_layout(rect=eval_sigmoid_cfg.P4_BARS_GRID_TIGHT_LAYOUT_RECT)
    os.makedirs(out_dir, exist_ok=True)
    fig.savefig(os.path.join(out_dir, "bars_period4_singlek_grid.png"), dpi=300, bbox_inches="tight")
    fig.savefig(os.path.join(out_dir, "bars_period4_singlek_grid.pdf"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def _plot_bars_grid_period4_singlek_bbh(base_dir: str, out_dir: str) -> None:
    """BBH-only version of _plot_bars_grid_period4_singlek with panels stacked vertically."""
    import matplotlib.pyplot as plt  # type: ignore
    import matplotlib as mpl  # type: ignore
    try:
        mpl.rcParams["font.family"] = mpl_rc_cfg.FONT_FAMILY
    except Exception:
        pass
    k_dirs = _list_k_eval_dirs(base_dir)
    if not k_dirs:
        return
    tasks = _detect_bbh_subtasks(k_dirs[0][1])
    if not tasks:
        return
    display_names = [_display_name(t) for t in tasks]

    panels = []
    y_max = 0.0
    for _k, ed in k_dirs:
        is_vals = []
        oos_vals = []
        for t in tasks:
            path = _get_summary_path(ed, t)
            if not path:
                is_macro = float("nan")
                oos_macro = float("nan")
            else:
                _t, is_macro, oos_macro, _, _ = _read_summary(path)
            is_vals.append(is_macro)
            oos_vals.append(oos_macro)
        is_arr = np.array(is_vals, float)
        oos_arr = np.array(oos_vals, float)
        panels.append({"is": is_arr, "oos": oos_arr})
        m = float(np.nanmax([np.nanmax(is_arr), np.nanmax(oos_arr)]))
        if np.isfinite(m):
            y_max = max(y_max, m)
    if y_max <= 0:
        y_max = 0.1

    # Add a final panel with mean over k
    is_mean = np.nanmean(np.vstack([p["is"] for p in panels]), axis=0)
    oos_mean = np.nanmean(np.vstack([p["oos"] for p in panels]), axis=0)
    panels.append({"is": is_mean, "oos": oos_mean})

    # Arrange panels vertically: one row per k, plus the mean row
    fig, axes = plt.subplots(
        len(panels),
        1,
        figsize=(
            eval_sigmoid_cfg.P4_BARS_BBH_FIGSIZE_WIDTH,
            eval_sigmoid_cfg.P4_BARS_BBH_FIGSIZE_HEIGHT_PER_ROW * len(panels),
        ),
        sharex=True,
    )
    axes = np.atleast_1d(axes)
    w = eval_sigmoid_cfg.P4_BARS_BBH_BAR_WIDTH  # Narrower bars to add spacing when many subtasks
    x = np.arange(len(tasks))

    color_is = eval_sigmoid_cfg.P4_BARS_GRID_COLOR_IS
    color_oos = eval_sigmoid_cfg.P4_BARS_GRID_COLOR_OOS

    labels = [f"k={k}" for (k, _ed) in k_dirs] + ["mean k"]

    for idx, (ax, panel, lbl) in enumerate(zip(axes, panels, labels), start=1):
        is_arr = panel["is"]
        oos_arr = panel["oos"]
        bars_is = ax.bar(x - w/2, is_arr, width=w, label="ID (train)", color=color_is, zorder=3)
        bars_oos = ax.bar(x + w/2, oos_arr, width=w, label="OOD (test)", color=color_oos, zorder=3)

        ax.set_xlim(
            eval_sigmoid_cfg.P4_BARS_GRID_XLIM_LEFT,
            len(tasks) - eval_sigmoid_cfg.P4_BARS_GRID_XLIM_RIGHT_PAD,
        )
        try:
            from matplotlib.colors import LinearSegmentedColormap
            grad_data = np.linspace(0, 1, 256).reshape(256, 1)
            grad_cmap = LinearSegmentedColormap.from_list(
                eval_sigmoid_cfg.P4_BARS_BG_GRAD_NAME,
                [eval_sigmoid_cfg.P4_BARS_BG_GRAD_TOP_COLOR, eval_sigmoid_cfg.P4_BARS_BG_GRAD_BOTTOM_COLOR],
            )
            xlims = ax.get_xlim()
            ax.imshow(
                grad_data,
                extent=[xlims[0], xlims[1], 0, y_max * eval_sigmoid_cfg.P4_BARS_GRID_YLIM_MULT],
                aspect="auto",
                cmap=grad_cmap,
                alpha=eval_sigmoid_cfg.P4_BARS_BG_GRAD_ALPHA,
                zorder=eval_sigmoid_cfg.P4_BARS_BG_GRAD_ZORDER,
                interpolation=eval_sigmoid_cfg.P4_BARS_BG_GRAD_INTERPOLATION,
            )
        except Exception:
            ax.set_facecolor(eval_sigmoid_cfg.P4_BARS_BG_GRAD_TOP_COLOR)

        try:
            from matplotlib.colors import LinearSegmentedColormap
            for bar, color in [(bars_is, color_is), (bars_oos, color_oos)]:
                for rect in bar:
                    height = rect.get_height()
                    if height <= 0 or not np.isfinite(height):
                        continue
                    from matplotlib.colors import to_rgb
                    rgb = to_rgb(color)
                    darker = tuple(c * eval_sigmoid_cfg.P4_BARS_BAR_GRAD_DARKEN_FACTOR for c in rgb)
                    grad_cmap = LinearSegmentedColormap.from_list(eval_sigmoid_cfg.P4_BARS_BAR_GRAD_NAME, [darker, color])
                    grad = grad_cmap(np.linspace(0, 1, 256).reshape(256, 1))
                    im = ax.imshow(grad, extent=[rect.get_x(), rect.get_x() + rect.get_width(), 0, height],
                                   aspect='auto', zorder=eval_sigmoid_cfg.P4_BARS_BAR_GRAD_ZORDER, interpolation=eval_sigmoid_cfg.P4_BARS_BAR_GRAD_INTERPOLATION)
                    im.set_clip_path(rect)
        except Exception:
            pass

        ax.set_xticks(x)
        # Only show x-axis tick labels on the last (bottom) subplot
        if idx == len(axes):
            ax.set_xticklabels(
                display_names,
                rotation=eval_sigmoid_cfg.P4_BARS_BBH_XTICK_ROTATION,
                ha="right",
                fontsize=eval_sigmoid_cfg.P4_BARS_BBH_XTICK_FONTSIZE,
            )
        else:
            ax.set_xticklabels([])
        ax.set_ylim(0.0, y_max * eval_sigmoid_cfg.P4_BARS_GRID_YLIM_MULT)
        try:
            with mpl.rc_context({"text.usetex": True, "text.latex.preamble": r"\usepackage{amsmath}"}):
                from matplotlib.offsetbox import AnchoredText  # type: ignore
                label_txt = lbl
                badge = AnchoredText(
                    label_txt,
                    loc="upper left",
                    prop=dict(size=eval_sigmoid_cfg.P4_BARS_BBH_BADGE_FONTSIZE, color=eval_sigmoid_cfg.P4_BARS_GRID_BADGE_COLOR),
                )
        except Exception:
            from matplotlib.offsetbox import AnchoredText  # type: ignore
            label_txt = lbl
            badge = AnchoredText(
                label_txt,
                loc="upper left",
                prop=dict(size=eval_sigmoid_cfg.P4_BARS_BBH_BADGE_FONTSIZE, color=eval_sigmoid_cfg.P4_BARS_GRID_BADGE_COLOR),
            )
        try:
            badge.patch.set_facecolor(eval_sigmoid_cfg.P4_BARS_GRID_BADGE_FACE)
            badge.patch.set_alpha(eval_sigmoid_cfg.P4_BARS_GRID_BADGE_ALPHA)
            badge.patch.set_edgecolor(eval_sigmoid_cfg.P4_BARS_GRID_BADGE_EDGE)
            badge.patch.set_linewidth(eval_sigmoid_cfg.P4_BARS_GRID_BADGE_LINEWIDTH)
        except Exception:
            pass
        ax.add_artist(badge)
        if idx == len(axes):
            leg = ax.legend(loc="upper right", fontsize=eval_sigmoid_cfg.P4_BARS_BBH_LEGEND_FONTSIZE)
            if leg and leg.get_title():
                leg.get_title().set_fontweight("bold")
        try:
            import matplotlib.ticker as mticker
            ax.yaxis.set_minor_locator(mticker.AutoMinorLocator(2))
        except Exception:
            pass
        ax.yaxis.grid(
            True,
            which="major",
            linestyle=eval_sigmoid_cfg.P4_BARS_GRID_MAJOR_GRID_LINESTYLE,
            linewidth=eval_sigmoid_cfg.P4_BARS_GRID_MAJOR_GRID_LINEWIDTH,
            color=eval_sigmoid_cfg.P4_BARS_GRID_MAJOR_GRID_COLOR,
            alpha=eval_sigmoid_cfg.P4_BARS_GRID_MAJOR_GRID_ALPHA,
            zorder=eval_sigmoid_cfg.P4_BARS_GRID_GRID_ZORDER,
        )
        ax.yaxis.grid(
            True,
            which="minor",
            linestyle=eval_sigmoid_cfg.P4_BARS_GRID_MINOR_GRID_LINESTYLE,
            linewidth=eval_sigmoid_cfg.P4_BARS_GRID_MINOR_GRID_LINEWIDTH,
            color=eval_sigmoid_cfg.P4_BARS_GRID_MINOR_GRID_COLOR,
            alpha=eval_sigmoid_cfg.P4_BARS_GRID_MINOR_GRID_ALPHA,
            zorder=eval_sigmoid_cfg.P4_BARS_GRID_GRID_ZORDER,
        )
        ax.tick_params(axis="y", labelsize=eval_sigmoid_cfg.P4_BARS_BBH_YTICK_LABELSIZE)

    # Figure-level y-axis label centered vertically across all subplots
    fig.supylabel(
        "MAE",
        fontweight="bold",
        fontsize=eval_sigmoid_cfg.P4_BARS_BBH_SUPYLABEL_FONTSIZE,
        x=eval_sigmoid_cfg.P4_BARS_BBH_SUPYLABEL_X,
    )
    fig.suptitle(
        eval_sigmoid_cfg.P4_BARS_BBH_TITLE,
        fontsize=eval_sigmoid_cfg.P4_BARS_BBH_TITLE_FONTSIZE,
        fontweight="bold",
        y=eval_sigmoid_cfg.P4_BARS_BBH_TITLE_Y,
    )
    # Use hspace for vertical spacing between subplots
    fig.subplots_adjust(hspace=eval_sigmoid_cfg.P4_BARS_BBH_HSPACE)
    fig.tight_layout(rect=eval_sigmoid_cfg.P4_BARS_BBH_TIGHT_LAYOUT_RECT)
    os.makedirs(out_dir, exist_ok=True)
    fig.savefig(os.path.join(out_dir, "bars_period4_singlek_grid_bbh.png"), dpi=300, bbox_inches="tight")
    fig.savefig(os.path.join(out_dir, "bars_period4_singlek_grid_bbh.pdf"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def _plot_bars_grid_period4_singlek_oos_only(base_dir: str, out_dir: str) -> None:
    """Create a smaller 1xK bars grid with only OOD values for available k folders."""
    import matplotlib.pyplot as plt  # type: ignore
    import matplotlib as mpl  # type: ignore
    try:
        mpl.rcParams["font.family"] = mpl_rc_cfg.FONT_FAMILY
    except Exception:
        pass
    k_dirs = _list_k_eval_dirs(base_dir)
    if not k_dirs:
        return
    tasks = _detect_main_tasks(k_dirs[0][1])
    display_names = [_display_name(t) for t in tasks]

    # Collect OOD values only
    panels = []  # list of OOD arrays
    y_max = 0.0
    for _k, ed in k_dirs:
        oos_vals = []
        for t in tasks:
            path = _get_summary_path(ed, t)
            if not path:
                oos_macro = float("nan")
            else:
                _t, _is_macro, oos_macro, _, _ = _read_summary(path)
            oos_vals.append(oos_macro)
        oos_arr = np.array(oos_vals, float)
        panels.append(oos_arr)
        m = float(np.nanmax(oos_arr))
        if np.isfinite(m):
            y_max = max(y_max, m)
    if y_max <= 0:
        y_max = 0.1

    # Plot K subplots in a row, shared y. Increase height for larger fonts.
    ncols = len(k_dirs)
    fig_width = max(
        eval_sigmoid_cfg.P4_BARS_GRID_BASE_WIDTH,
        (eval_sigmoid_cfg.P4_BARS_GRID_BASE_WIDTH * ncols) / eval_sigmoid_cfg.P4_BARS_GRID_WIDTH_DIVISOR,
    )
    fig, axes = plt.subplots(1, ncols, figsize=(fig_width, eval_sigmoid_cfg.P4_BARS_OOS_GRID_HEIGHT), sharey=True)
    axes = np.atleast_1d(axes)
    w = eval_sigmoid_cfg.P4_BARS_OOS_GRID_BAR_WIDTH  # Wider bars since we only have one set
    x = np.arange(len(tasks))

    # Use coral color for OOD
    color_oos = eval_sigmoid_cfg.P4_BARS_GRID_COLOR_OOS

    for idx, ((k, _ed), oos_arr, ax) in enumerate(zip(k_dirs, panels, axes), start=1):
        bars_oos = ax.bar(x, oos_arr, width=w, label="OOD (test)", color=color_oos, zorder=3)

        # Set explicit x-limits with padding to keep bars inside subplot frame
        ax.set_xlim(
            eval_sigmoid_cfg.P4_BARS_GRID_XLIM_LEFT,
            len(tasks) - eval_sigmoid_cfg.P4_BARS_GRID_XLIM_RIGHT_PAD,
        )

        # Vertical gradient background (light gray at top to white at bottom)
        try:
            from matplotlib.colors import LinearSegmentedColormap
            grad_data = np.linspace(0, 1, 256).reshape(256, 1)
            grad_cmap = LinearSegmentedColormap.from_list(
                eval_sigmoid_cfg.P4_BARS_BG_GRAD_NAME,
                [eval_sigmoid_cfg.P4_BARS_BG_GRAD_TOP_COLOR, eval_sigmoid_cfg.P4_BARS_BG_GRAD_BOTTOM_COLOR],
            )
            xlims = ax.get_xlim()
            ax.imshow(
                grad_data,
                extent=[xlims[0], xlims[1], 0, y_max * eval_sigmoid_cfg.P4_BARS_GRID_YLIM_MULT],
                aspect="auto",
                cmap=grad_cmap,
                alpha=eval_sigmoid_cfg.P4_BARS_BG_GRAD_ALPHA,
                zorder=eval_sigmoid_cfg.P4_BARS_BG_GRAD_ZORDER,
                interpolation=eval_sigmoid_cfg.P4_BARS_BG_GRAD_INTERPOLATION,
            )
        except Exception:
            ax.set_facecolor(eval_sigmoid_cfg.P4_BARS_BG_GRAD_TOP_COLOR)

        # Apply gradient fills to bars (darker at bottom, lighter at top)
        try:
            from matplotlib.colors import LinearSegmentedColormap
            import matplotlib.patches as mpatches
            for rect in bars_oos:
                height = rect.get_height()
                if height <= 0 or not np.isfinite(height):
                    continue
                # Create gradient colormap from darker to lighter
                from matplotlib.colors import to_rgb
                rgb = to_rgb(color_oos)
                darker = tuple(c * eval_sigmoid_cfg.P4_BARS_BAR_GRAD_DARKEN_FACTOR for c in rgb)
                grad_cmap = LinearSegmentedColormap.from_list(eval_sigmoid_cfg.P4_BARS_BAR_GRAD_NAME, [darker, color_oos])
                # Create gradient patch
                grad = grad_cmap(np.linspace(0, 1, 256).reshape(256, 1))
                im = ax.imshow(grad, extent=[rect.get_x(), rect.get_x() + rect.get_width(),
                                              0, height], aspect='auto', zorder=eval_sigmoid_cfg.P4_BARS_BAR_GRAD_ZORDER, interpolation=eval_sigmoid_cfg.P4_BARS_BAR_GRAD_INTERPOLATION)
                im.set_clip_path(rect)
        except Exception:
            pass
        ax.set_xticks(x)
        ax.set_xticklabels(display_names, rotation=eval_sigmoid_cfg.P4_BARS_GRID_XTICK_ROTATION, ha="right", fontsize=(fig_width / 2))
        ax.set_ylim(0.0, y_max * eval_sigmoid_cfg.P4_BARS_GRID_YLIM_MULT)
        # Embedded badge with larger font
        try:
            with mpl.rc_context({"text.usetex": True, "text.latex.preamble": r"\usepackage{amsmath}"}):
                label_txt = fr"$k={k}\mid$ OOD"
                from matplotlib.offsetbox import AnchoredText  # type: ignore
                badge = AnchoredText(
                    label_txt,
                    loc="upper left",
                    prop=dict(
                        size=eval_sigmoid_cfg.P4_BARS_OOS_GRID_BADGE_FONTSIZE,
                        color=eval_sigmoid_cfg.P4_BARS_GRID_BADGE_COLOR,
                        fontweight=eval_sigmoid_cfg.P4_BARS_OOS_GRID_BADGE_WEIGHT,
                    ),
                )
        except Exception:
            label_txt = fr"$k={k}\mid$ OOD"
            from matplotlib.offsetbox import AnchoredText  # type: ignore
            badge = AnchoredText(
                label_txt,
                loc="upper left",
                prop=dict(
                    size=eval_sigmoid_cfg.P4_BARS_OOS_GRID_BADGE_FONTSIZE,
                    color=eval_sigmoid_cfg.P4_BARS_GRID_BADGE_COLOR,
                    fontweight=eval_sigmoid_cfg.P4_BARS_OOS_GRID_BADGE_WEIGHT,
                ),
            )
        try:
            badge.patch.set_facecolor(eval_sigmoid_cfg.P4_BARS_GRID_BADGE_FACE)
            badge.patch.set_alpha(eval_sigmoid_cfg.P4_BARS_GRID_BADGE_ALPHA)
            badge.patch.set_edgecolor(eval_sigmoid_cfg.P4_BARS_GRID_BADGE_EDGE)
            badge.patch.set_linewidth(eval_sigmoid_cfg.P4_BARS_GRID_BADGE_LINEWIDTH)
        except Exception:
            pass
        ax.add_artist(badge)
        if idx == 1:
            # Larger font for y-axis label
            ax.set_ylabel("MAE", fontweight="bold", fontsize=eval_sigmoid_cfg.P4_BARS_OOS_GRID_YLABEL_FONTSIZE)
        # Grid styling
        try:
            import matplotlib.ticker as mticker
            ax.yaxis.set_minor_locator(mticker.AutoMinorLocator(2))
        except Exception:
            pass
        ax.yaxis.grid(
            True,
            which="major",
            linestyle=eval_sigmoid_cfg.P4_BARS_GRID_MAJOR_GRID_LINESTYLE,
            linewidth=eval_sigmoid_cfg.P4_BARS_GRID_MAJOR_GRID_LINEWIDTH,
            color=eval_sigmoid_cfg.P4_BARS_GRID_MAJOR_GRID_COLOR,
            alpha=eval_sigmoid_cfg.P4_BARS_GRID_MAJOR_GRID_ALPHA,
            zorder=eval_sigmoid_cfg.P4_BARS_GRID_GRID_ZORDER,
        )
        ax.yaxis.grid(
            True,
            which="minor",
            linestyle=eval_sigmoid_cfg.P4_BARS_GRID_MINOR_GRID_LINESTYLE,
            linewidth=eval_sigmoid_cfg.P4_BARS_GRID_MINOR_GRID_LINEWIDTH,
            color=eval_sigmoid_cfg.P4_BARS_GRID_MINOR_GRID_COLOR,
            alpha=eval_sigmoid_cfg.P4_BARS_GRID_MINOR_GRID_ALPHA,
            zorder=eval_sigmoid_cfg.P4_BARS_GRID_GRID_ZORDER,
        )
        # Larger tick labels
        ax.tick_params(axis="y", labelsize=eval_sigmoid_cfg.P4_BARS_OOS_GRID_YTICK_LABELSIZE)
    # Set the title - changed from "IS vs OOS" to "OOS"
    fig.suptitle(
        eval_sigmoid_cfg.P4_BARS_OOS_GRID_TITLE,
        fontsize=eval_sigmoid_cfg.P4_BARS_OOS_GRID_TITLE_FONTSIZE,
        fontweight="bold",
        y=eval_sigmoid_cfg.P4_BARS_OOS_GRID_TITLE_Y,
    )
    # Increase horizontal spacing between subplots
    fig.subplots_adjust(wspace=eval_sigmoid_cfg.P4_BARS_OOS_GRID_WSPACE)
    fig.tight_layout(rect=eval_sigmoid_cfg.P4_BARS_OOS_GRID_TIGHT_LAYOUT_RECT)
    os.makedirs(out_dir, exist_ok=True)
    fig.savefig(os.path.join(out_dir, "bars_period4_singlek_oos.png"), dpi=300, bbox_inches="tight")
    fig.savefig(os.path.join(out_dir, "bars_period4_singlek_oos.pdf"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def _plot_bars_grid_period4_singlek_oos_only_bbh(base_dir: str, out_dir: str) -> None:
    """OOD-only bars grid for BBH subtasks (available k folders)."""
    import matplotlib.pyplot as plt  # type: ignore
    import matplotlib as mpl  # type: ignore
    try:
        mpl.rcParams["font.family"] = mpl_rc_cfg.FONT_FAMILY
    except Exception:
        pass
    k_dirs = _list_k_eval_dirs(base_dir)
    if not k_dirs:
        return
    tasks = _detect_bbh_subtasks(k_dirs[0][1])
    if not tasks:
        return
    display_names = [_display_name(t) for t in tasks]

    panels = []
    y_max = 0.0
    for _k, ed in k_dirs:
        oos_vals = []
        for t in tasks:
            path = _get_summary_path(ed, t)
            if not path:
                oos_macro = float("nan")
            else:
                _t, _is_macro, oos_macro, _, _ = _read_summary(path)
            oos_vals.append(oos_macro)
        oos_arr = np.array(oos_vals, float)
        panels.append(oos_arr)
        m = float(np.nanmax(oos_arr))
        if np.isfinite(m):
            y_max = max(y_max, m)
    if y_max <= 0:
        y_max = 0.1

    # Vertical layout: one column, multiple rows
    fig, axes = plt.subplots(
        len(panels),
        1,
        figsize=(
            eval_sigmoid_cfg.P4_BARS_BBH_FIGSIZE_WIDTH,
            eval_sigmoid_cfg.P4_BARS_BBH_FIGSIZE_HEIGHT_PER_ROW * len(panels),
        ),
        sharex=True,
    )
    axes = np.atleast_1d(axes)
    w = eval_sigmoid_cfg.P4_BARS_OOS_BBH_BAR_WIDTH  # Narrower bars to add spacing when many subtasks
    x = np.arange(len(tasks))
    color_oos = eval_sigmoid_cfg.P4_BARS_GRID_COLOR_OOS

    for idx, ((k, _ed), oos_arr, ax) in enumerate(zip(k_dirs, panels, axes), start=1):
        bars_oos = ax.bar(x, oos_arr, width=w, label="OOD (test)", color=color_oos, zorder=3)
        ax.set_xlim(
            eval_sigmoid_cfg.P4_BARS_GRID_XLIM_LEFT,
            len(tasks) - eval_sigmoid_cfg.P4_BARS_GRID_XLIM_RIGHT_PAD,
        )
        try:
            from matplotlib.colors import LinearSegmentedColormap
            grad_data = np.linspace(0, 1, 256).reshape(256, 1)
            grad_cmap = LinearSegmentedColormap.from_list(
                eval_sigmoid_cfg.P4_BARS_BG_GRAD_NAME,
                [eval_sigmoid_cfg.P4_BARS_BG_GRAD_TOP_COLOR, eval_sigmoid_cfg.P4_BARS_BG_GRAD_BOTTOM_COLOR],
            )
            xlims = ax.get_xlim()
            ax.imshow(
                grad_data,
                extent=[xlims[0], xlims[1], 0, y_max * eval_sigmoid_cfg.P4_BARS_GRID_YLIM_MULT],
                aspect="auto",
                cmap=grad_cmap,
                alpha=eval_sigmoid_cfg.P4_BARS_BG_GRAD_ALPHA,
                zorder=eval_sigmoid_cfg.P4_BARS_BG_GRAD_ZORDER,
                interpolation=eval_sigmoid_cfg.P4_BARS_BG_GRAD_INTERPOLATION,
            )
        except Exception:
            ax.set_facecolor(eval_sigmoid_cfg.P4_BARS_BG_GRAD_TOP_COLOR)

        try:
            from matplotlib.colors import LinearSegmentedColormap
            from matplotlib.colors import to_rgb
            for rect in bars_oos:
                height = rect.get_height()
                if height <= 0 or not np.isfinite(height):
                    continue
                rgb = to_rgb(color_oos)
                darker = tuple(c * eval_sigmoid_cfg.P4_BARS_BAR_GRAD_DARKEN_FACTOR for c in rgb)
                grad_cmap = LinearSegmentedColormap.from_list(eval_sigmoid_cfg.P4_BARS_BAR_GRAD_NAME, [darker, color_oos])
                grad = grad_cmap(np.linspace(0, 1, 256).reshape(256, 1))
                im = ax.imshow(grad, extent=[rect.get_x(), rect.get_x() + rect.get_width(), 0, height],
                               aspect='auto', zorder=eval_sigmoid_cfg.P4_BARS_BAR_GRAD_ZORDER, interpolation=eval_sigmoid_cfg.P4_BARS_BAR_GRAD_INTERPOLATION)
                im.set_clip_path(rect)
        except Exception:
            pass
        ax.set_xticks(x)
        if idx == len(axes):
            ax.set_xticklabels(
                display_names,
                rotation=eval_sigmoid_cfg.P4_BARS_BBH_XTICK_ROTATION,
                ha="right",
                fontsize=eval_sigmoid_cfg.P4_BARS_BBH_XTICK_FONTSIZE,
            )
        else:
            ax.set_xticklabels([])
        ax.set_ylim(0.0, y_max * eval_sigmoid_cfg.P4_BARS_GRID_YLIM_MULT)
        try:
            with mpl.rc_context({"text.usetex": True, "text.latex.preamble": r"\usepackage{amsmath}"}):
                from matplotlib.offsetbox import AnchoredText  # type: ignore
                label_txt = fr"$k={k}\mid$ OOD"
                badge = AnchoredText(
                    label_txt,
                    loc="upper left",
                    prop=dict(size=eval_sigmoid_cfg.P4_BARS_OOS_BBH_BADGE_FONTSIZE, color=eval_sigmoid_cfg.P4_BARS_GRID_BADGE_COLOR),
                )
        except Exception:
            from matplotlib.offsetbox import AnchoredText  # type: ignore
            label_txt = fr"$k={k}\mid$ OOD"
            badge = AnchoredText(
                label_txt,
                loc="upper left",
                prop=dict(size=eval_sigmoid_cfg.P4_BARS_OOS_BBH_BADGE_FONTSIZE, color=eval_sigmoid_cfg.P4_BARS_GRID_BADGE_COLOR),
            )
        try:
            badge.patch.set_facecolor(eval_sigmoid_cfg.P4_BARS_GRID_BADGE_FACE)
            badge.patch.set_alpha(eval_sigmoid_cfg.P4_BARS_GRID_BADGE_ALPHA)
            badge.patch.set_edgecolor(eval_sigmoid_cfg.P4_BARS_GRID_BADGE_EDGE)
            badge.patch.set_linewidth(eval_sigmoid_cfg.P4_BARS_GRID_BADGE_LINEWIDTH)
        except Exception:
            pass
        ax.add_artist(badge)
        try:
            import matplotlib.ticker as mticker
            ax.yaxis.set_minor_locator(mticker.AutoMinorLocator(2))
        except Exception:
            pass
        ax.yaxis.grid(
            True,
            which="major",
            linestyle=eval_sigmoid_cfg.P4_BARS_GRID_MAJOR_GRID_LINESTYLE,
            linewidth=eval_sigmoid_cfg.P4_BARS_GRID_MAJOR_GRID_LINEWIDTH,
            color=eval_sigmoid_cfg.P4_BARS_GRID_MAJOR_GRID_COLOR,
            alpha=eval_sigmoid_cfg.P4_BARS_GRID_MAJOR_GRID_ALPHA,
            zorder=eval_sigmoid_cfg.P4_BARS_GRID_GRID_ZORDER,
        )
        ax.yaxis.grid(
            True,
            which="minor",
            linestyle=eval_sigmoid_cfg.P4_BARS_GRID_MINOR_GRID_LINESTYLE,
            linewidth=eval_sigmoid_cfg.P4_BARS_GRID_MINOR_GRID_LINEWIDTH,
            color=eval_sigmoid_cfg.P4_BARS_GRID_MINOR_GRID_COLOR,
            alpha=eval_sigmoid_cfg.P4_BARS_GRID_MINOR_GRID_ALPHA,
            zorder=eval_sigmoid_cfg.P4_BARS_GRID_GRID_ZORDER,
        )
        ax.tick_params(axis="y", labelsize=eval_sigmoid_cfg.P4_BARS_OOS_BBH_YTICK_LABELSIZE)

    # Figure-level y-axis label centered vertically across all subplots
    fig.supylabel(
        "MAE",
        fontweight="bold",
        fontsize=eval_sigmoid_cfg.P4_BARS_BBH_SUPYLABEL_FONTSIZE,
        x=eval_sigmoid_cfg.P4_BARS_BBH_SUPYLABEL_X,
    )
    fig.suptitle(
        eval_sigmoid_cfg.P4_BARS_OOS_BBH_TITLE,
        fontsize=eval_sigmoid_cfg.P4_BARS_OOS_BBH_TITLE_FONTSIZE,
        fontweight="bold",
        y=eval_sigmoid_cfg.P4_BARS_OOS_BBH_TITLE_Y,
    )
    # Use hspace for vertical spacing between subplots
    fig.subplots_adjust(hspace=eval_sigmoid_cfg.P4_BARS_OOS_BBH_HSPACE)
    fig.tight_layout(rect=eval_sigmoid_cfg.P4_BARS_OOS_BBH_TIGHT_LAYOUT_RECT)
    os.makedirs(out_dir, exist_ok=True)
    fig.savefig(os.path.join(out_dir, "bars_period4_singlek_oos_bbh.png"), dpi=300, bbox_inches="tight")
    fig.savefig(os.path.join(out_dir, "bars_period4_singlek_oos_bbh.pdf"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def _plot_calib_summary_panels(base_dir: str) -> None:
    """Create summary panels A–C for period4 single_k evaluation (main tasks).

    Saves three separate publication-ready figures sized for horizontal layout.
    """
    from matplotlib.ticker import FuncFormatter  # type: ignore

    is_pinball_base = "evaluation_pinball" in os.path.abspath(base_dir)
    k_dirs = _list_k_eval_dirs(base_dir)
    if not k_dirs:
        return
    eval_dirs = [p for (_k, p) in k_dirs]
    k_vals = [k for (k, _p) in k_dirs]

    all_tasks = _detect_main_tasks(eval_dirs[0])
    # Subset highlighted as higher-drift tasks (thicker lines in Panel A).
    drift_candidates = ["IFEval Raw", "MATH Lvl 5 Raw", "MMLU-PRO Raw"]
    drift_tasks = [t for t in drift_candidates if t in all_tasks]
    drift_set = set(drift_tasks)
    stable_tasks = [t for t in all_tasks if t not in drift_set]
    tau = float(PLOT_TAU)
    out_dir = os.path.join(base_dir, "plots")
    os.makedirs(out_dir, exist_ok=True)

    # Configure matplotlib for publication quality with serif fonts and LaTeX
    mpl.rcParams.update(eval_sigmoid_cfg.CALIB_SUMMARY_RC_PARAMS)

    def _scaled_formatter(scale: float) -> FuncFormatter:
        def _fmt(y: float, _pos: float) -> str:
            v = y / scale
            if abs(v) < 0.05:
                v = 0.0
            return f"{v:.1f}"

        return FuncFormatter(_fmt)

    def mean_signed_err(task: str, ed: str) -> float:
        path = _find_bins_path(ed, task, which="test")
        if not path or not os.path.exists(path):
            return float("nan")
        df = pd.read_csv(path)
        if is_pinball_base:
            if "abs_err" in df.columns:
                return float(df["abs_err"].mean())
            return float("nan")
        if "hat_tau" in df.columns:
            return float((df["hat_tau"] - tau).mean())
        return float("nan")

    stable_series = {t: np.array([mean_signed_err(t, ed) for ed in eval_dirs]) for t in stable_tasks}
    drift_series = {t: np.array([mean_signed_err(t, ed) for ed in eval_dirs]) for t in drift_tasks}

    drift_heat: Dict[str, List[np.ndarray]] = {}
    drift_edges: Dict[str, List[np.ndarray]] = {}
    for t in drift_tasks:
        mats = []
        edges_list = []
        for ed in eval_dirs:
            path = _find_bins_path(ed, t, which="test")
            if not path or not os.path.exists(path):
                mats.append(np.array([]))
                edges_list.append(np.array([]))
                continue
            df = pd.read_csv(path)
            if df.empty or any(c not in df.columns for c in ("z_lo", "z_hi", "hat_tau")):
                mats.append(np.array([]))
                edges_list.append(np.array([]))
                continue
            edges = np.r_[df["z_lo"].values[0], df["z_hi"].values]
            vals = df["abs_err"].values if is_pinball_base else (df["hat_tau"].values - tau)
            mats.append(vals)
            edges_list.append(edges)
        drift_heat[t] = mats
        drift_edges[t] = edges_list

    # Density for Panel C (combined train bins across all k, using k1 bin edges)
    bins_ref = None
    scan_dir = _bins_scan_dir(eval_dirs[0])
    for fn in os.listdir(scan_dir):
        if fn.endswith("__bins_train.csv") or fn.endswith("_bins_train.csv"):
            dfb = pd.read_csv(os.path.join(scan_dir, fn))
            if dfb.empty or any(c not in dfb.columns for c in ("z_lo", "z_hi")):
                continue
            bins_ref = np.r_[dfb["z_lo"].values[0], dfb["z_hi"].values]
            break
    density_qwen = density_other = None
    if bins_ref is not None:
        csv_path = os.path.join("tables", "open_llm_leaderboard", "open_llm_leaderboard_with_tokens.csv")
        if os.path.exists(csv_path):
            df_all = pd.read_csv(csv_path)
            date_col = detect_date_col(df_all)
            df_all["period"] = df_all[date_col].fillna("").apply(parse_year_month)
            compute = df_all["Pretraining tokens (T)"] * df_all["#Params (B)"] * 6
            df_all["logC"] = np.log10(compute)

            def cmp_le(a, cutoff):
                if a is None:
                    return False
                y, m = a
                cy, cm = map(int, cutoff.split("-"))
                return (y, m) <= (cy, cm)

            # combine train labels across all periods (k=1..4)
            all_train_labels: List[str] = []
            for split in PERIOD4_SPLITS_SINGLE:
                all_train_labels.extend(split["train_labels"])

            mtrain = pd.Series(False, index=df_all.index)
            for lab in all_train_labels:
                if lab.startswith("<="):
                    cutoff = lab[2:]
                    mtrain |= df_all["period"].apply(lambda x: cmp_le(x, cutoff))
                else:
                    lo, hi = lab.split("..")

                    def between(a):
                        if a is None:
                            return False
                        y, m = a
                        ly, lm = map(int, lo.split("-"))
                        hy, hm = map(int, hi.split("-"))
                        return (y, m) >= (ly, lm) and (y, m) <= (hy, hm)

                    mtrain |= df_all["period"].apply(between)
            df_sel = df_all[mtrain & df_all["logC"].notna()].copy()
            bins_idx = pd.cut(df_sel["logC"], bins=bins_ref, labels=False, include_lowest=True, right=False)
            df_sel["bin"] = bins_idx
            df_sel["is_qwen"] = (
                df_sel["Identified base model"]
                .fillna("")
                .astype(str)
                .str.lower()
                .str.startswith("qwen")
            )
            nbins = len(bins_ref) - 1
            density_qwen = np.zeros(nbins, dtype=int)
            density_other = np.zeros(nbins, dtype=int)
            for b in range(nbins):
                mask = df_sel["bin"] == b
                density_qwen[b] = int(df_sel[mask & df_sel["is_qwen"]].shape[0])
                density_other[b] = int(df_sel[mask & (~df_sel["is_qwen"])].shape[0])

    # =========================================================================
    # PANEL A: Line plot (OOD metric by period)
    # =========================================================================
    figA = plt.figure(figsize=(5.0, 3.2))
    axA = figA.add_subplot(111)
    xk = np.array(k_vals, dtype=float)

    # Plot exactly the six main tasks once each (avoid duplicate legend entries).
    task_to_vals: Dict[str, np.ndarray] = {}
    task_to_style: Dict[str, str] = {}
    for t in stable_tasks:
        task_to_vals[t] = stable_series[t]
        task_to_style[t] = "stable"
    for t in drift_tasks:
        task_to_vals[t] = drift_series[t]
        task_to_style[t] = "drift"

    for t in all_tasks:
        vals = task_to_vals.get(t)
        if vals is None:
            continue
        if task_to_style.get(t) == "drift":
            axA.plot(xk, vals, marker="o", markersize=9, label=t, linewidth=3.0)
        else:
            axA.plot(
                xk, vals, marker="o", markersize=8, label=t, alpha=0.8, linewidth=2.0
            )

    axA.set_title(
        r"\textbf{OOD pinball loss}" if is_pinball_base else r"\textbf{OOD coverage error}",
        fontsize=14,
        pad=8,
    )
    axA.set_xlabel(r"\textbf{Period} $k$", fontsize=13)
    if is_pinball_base:
        axA.set_ylabel(r"$\mathbf{mean}\,\rho_\tau\; (\times 10^{-3})$", fontsize=13)
        axA.yaxis.set_major_formatter(_scaled_formatter(1e-3))
    else:
        axA.set_ylabel(r"$\mathbf{mean}\,\hat{\tau}-\tau\; (\times 10^{-2})$", fontsize=13)
        axA.yaxis.set_major_formatter(_scaled_formatter(1e-2))
    axA.grid(axis="y", alpha=0.3, linestyle="--", linewidth=0.7)
    axA.set_xticks(k_vals)
    axA.tick_params(labelsize=11)

    # Legend: single boxed legend spanning the bottom of the figure (2 rows).
    handles, labels = axA.get_legend_handles_labels()
    seen = set()
    uniq_handles = []
    uniq_labels = []
    for h, lab in zip(handles, labels):
        if lab in seen:
            continue
        seen.add(lab)
        uniq_handles.append(h)
        uniq_labels.append(lab)

    legend = None
    # Start large and shrink until the two-row legend comfortably fits the figure width.
    for fs in range(12, 6, -1):
        if legend is not None:
            legend.remove()
        legend = figA.legend(
            uniq_handles,
            uniq_labels,
            loc="lower center",
            bbox_to_anchor=(0.5, 0.02),
            ncol=3,
            fontsize=fs,
            frameon=True,
            fancybox=True,
            borderpad=0.6,
            handlelength=2.0,
            handletextpad=0.6,
            columnspacing=1.4,
        )
        frame = legend.get_frame()
        frame.set_facecolor("white")
        frame.set_edgecolor("#c0c0c0")
        frame.set_linewidth(0.8)
        frame.set_alpha(0.95)

        figA.canvas.draw()
        renderer = figA.canvas.get_renderer()
        leg_bbox = legend.get_window_extent(renderer=renderer)
        fig_bbox = figA.get_window_extent(renderer=renderer)
        if leg_bbox.width <= 0.98 * fig_bbox.width:
            break

    # Reserve vertical space for the bottom legend box.
    figA.canvas.draw()
    renderer = figA.canvas.get_renderer()
    leg_bbox_fig = legend.get_window_extent(renderer=renderer).transformed(
        figA.transFigure.inverted()
    )
    bottom = min(0.35, float(leg_bbox_fig.y1) + 0.04)
    figA.tight_layout(rect=[0.0, bottom, 1.0, 1.0], pad=0.3)
    figA.savefig(os.path.join(out_dir, "panel_a_lineplot.png"), dpi=300, bbox_inches="tight")
    figA.savefig(os.path.join(out_dir, "panel_a_lineplot.pdf"), dpi=300, bbox_inches="tight")
    plt.close(figA)

    # Optional: if both calibration-eval and pinball-eval bases exist, also render a
    # merged Panel A figure with coverage (left) and pinball (right).
    def _compute_panel_a_task_values(
        base: str, *, is_pin: bool
    ) -> Tuple[List[int], List[str], Dict[str, np.ndarray]]:
        k_dirs_local = _list_k_eval_dirs(base)
        if not k_dirs_local:
            return [], [], {}
        eval_dirs_local = [p for (_k, p) in k_dirs_local]
        k_vals_local = [k for (k, _p) in k_dirs_local]
        tasks_local = _detect_main_tasks(eval_dirs_local[0])

        def _mean_signed_err_local(task: str, ed: str) -> float:
            path = _find_bins_path(ed, task, which="test")
            if not path or not os.path.exists(path):
                return float("nan")
            df_local = pd.read_csv(path)
            if is_pin:
                if "abs_err" in df_local.columns:
                    return float(df_local["abs_err"].mean())
                return float("nan")
            if "hat_tau" in df_local.columns:
                return float((df_local["hat_tau"] - tau).mean())
            return float("nan")

        task_to_vals_local = {
            t: np.array([_mean_signed_err_local(t, ed) for ed in eval_dirs_local])
            for t in tasks_local
        }
        return k_vals_local, tasks_local, task_to_vals_local

    calib_base_default = os.path.join(
        REPO_ROOT, "outputs", "evaluation", "sigmoid", "period4", "single_k", "no_budget"
    )
    pinball_base_default = os.path.join(
        REPO_ROOT, "evaluation_pinball", "period4_singlek_no_budget"
    )
    have_both_bases = os.path.isdir(calib_base_default) and os.path.isdir(pinball_base_default)
    if have_both_bases:
        k_cal, tasks_cal, vals_cal = _compute_panel_a_task_values(
            calib_base_default, is_pin=False
        )
        k_pin, tasks_pin, vals_pin = _compute_panel_a_task_values(
            pinball_base_default, is_pin=True
        )
        common_tasks = [t for t in all_tasks if t in tasks_cal and t in tasks_pin]
        common_k = [k for k in k_vals if k in k_cal and k in k_pin]
        if common_tasks and common_k:
            k_to_idx_cal = {k: i for i, k in enumerate(k_cal)}
            k_to_idx_pin = {k: i for i, k in enumerate(k_pin)}

            figC, (axL, axR) = plt.subplots(1, 2, figsize=(5.0, 3.2))
            xk_common = np.array(common_k, dtype=float)

            for t in common_tasks:
                y_cal = vals_cal.get(t)
                y_pin = vals_pin.get(t)
                if y_cal is None or y_pin is None:
                    continue
                y_cal_aligned = np.array([y_cal[k_to_idx_cal[k]] for k in common_k], dtype=float)
                y_pin_aligned = np.array([y_pin[k_to_idx_pin[k]] for k in common_k], dtype=float)
                if t in drift_set:
                    axL.plot(xk_common, y_cal_aligned, marker="o", markersize=9, label=t, linewidth=3.0)
                    axR.plot(xk_common, y_pin_aligned, marker="o", markersize=9, label=t, linewidth=3.0)
                else:
                    axL.plot(
                        xk_common,
                        y_cal_aligned,
                        marker="o",
                        markersize=8,
                        label=t,
                        alpha=0.8,
                        linewidth=2.0,
                    )
                    axR.plot(
                        xk_common,
                        y_pin_aligned,
                        marker="o",
                        markersize=8,
                        label=t,
                        alpha=0.8,
                        linewidth=2.0,
                    )

            axL.set_title(r"\textbf{OOD coverage error}", fontsize=14, pad=8)
            axR.set_title(r"\textbf{OOD pinball loss}", fontsize=14, pad=8)
            axL.set_xlabel(r"\textbf{Period} $k$", fontsize=13)
            axR.set_xlabel(r"\textbf{Period} $k$", fontsize=13)
            axL.set_ylabel(r"$\mathbf{mean}\,\hat{\tau}-\tau\; (\times 10^{-2})$", fontsize=13)
            axR.set_ylabel(r"$\mathbf{mean}\,\rho_\tau\; (\times 10^{-3})$", fontsize=13)
            axL.yaxis.set_major_formatter(_scaled_formatter(1e-2))
            axR.yaxis.set_major_formatter(_scaled_formatter(1e-3))
            for ax in (axL, axR):
                ax.grid(axis="y", alpha=0.3, linestyle="--", linewidth=0.7)
                ax.set_xticks(common_k)
                ax.tick_params(labelsize=11)

            # Legend: single boxed legend spanning the bottom of the figure (2 rows).
            handles, labels = axL.get_legend_handles_labels()
            seen = set()
            uniq_handles = []
            uniq_labels = []
            for h, lab in zip(handles, labels):
                if lab in seen:
                    continue
                seen.add(lab)
                uniq_handles.append(h)
                uniq_labels.append(lab)

            legend = None
            for fs in range(12, 6, -1):
                if legend is not None:
                    legend.remove()
                legend = figC.legend(
                    uniq_handles,
                    uniq_labels,
                    loc="lower center",
                    bbox_to_anchor=(0.5, 0.02),
                    ncol=3,
                    fontsize=fs,
                    frameon=True,
                    fancybox=True,
                    borderpad=0.6,
                    handlelength=2.0,
                    handletextpad=0.6,
                    columnspacing=1.4,
                )
                frame = legend.get_frame()
                frame.set_facecolor("white")
                frame.set_edgecolor("#c0c0c0")
                frame.set_linewidth(0.8)
                frame.set_alpha(0.95)

                figC.canvas.draw()
                renderer = figC.canvas.get_renderer()
                leg_bbox = legend.get_window_extent(renderer=renderer)
                fig_bbox = figC.get_window_extent(renderer=renderer)
                if leg_bbox.width <= 0.98 * fig_bbox.width:
                    break

            figC.canvas.draw()
            renderer = figC.canvas.get_renderer()
            leg_bbox_fig = legend.get_window_extent(renderer=renderer).transformed(
                figC.transFigure.inverted()
            )
            bottom = min(0.35, float(leg_bbox_fig.y1) + 0.04)
            figC.tight_layout(rect=[0.0, bottom, 1.0, 1.0], pad=0.3)

            for out_base in (calib_base_default, pinball_base_default):
                out_dir_target = os.path.join(out_base, "plots")
                os.makedirs(out_dir_target, exist_ok=True)
                figC.savefig(
                    os.path.join(out_dir_target, "panel_a_lineplot.png"),
                    dpi=300,
                    bbox_inches="tight",
                )
                figC.savefig(
                    os.path.join(out_dir_target, "panel_a_lineplot.pdf"),
                    dpi=300,
                    bbox_inches="tight",
                )
            plt.close(figC)

    # =========================================================================
    # PANEL B: Heatmaps (select rows from main OOS heatmap)
    # =========================================================================
    panel_tasks = [t for t in drift_candidates if t in all_tasks]
    if not panel_tasks:
        # Still write the legacy combined panel if possible, but skip panel B/C.
        if len(k_dirs) == 3:
            _plot_calib_summary_panels_legacy(
                base_dir,
                stable_series,
                drift_series,
                drift_heat,
                drift_edges,
                density_qwen,
                density_other,
                bins_ref,
            )
        return
    # Align color scale and colormap with the main OOS heatmap
    panel_data = []
    width_ratios = []
    all_vals = []
    for k, ed in k_dirs:
        M_abs, S_signed, xt = _prepare_matrix(ed, panel_tasks, "test")
        panel_data.append((k, M_abs, S_signed, xt))
        width_ratios.append(max(M_abs.shape[1], 1))
        if M_abs.size:
            all_vals.append(M_abs[np.isfinite(M_abs)])
    vmax = float(np.nanmax(np.concatenate(all_vals))) if all_vals else 1.0
    vmin = 0.0
    figB = plt.figure(figsize=(max(12.0, 4.0 * len(panel_data)), 3.2))
    gs = figB.add_gridspec(1, len(panel_data), hspace=0.10, wspace=0.15,
                           left=0.07, right=0.88, top=0.90, bottom=0.20,
                           width_ratios=width_ratios)
    mesh = None
    cmap_shared = _make_heatmap_cmap()

    for c, (k, M_abs, S_signed, xt) in enumerate(panel_data):
        ax = figB.add_subplot(gs[0, c])
        if M_abs.size == 0:
            ax.axis("off")
            continue
        n_tasks, n_bins = M_abs.shape
        if n_bins == 0:
            ax.axis("off")
            continue
        x_edges = np.arange(n_bins + 1)
        y_edges = np.arange(n_tasks + 1)
        mesh = ax.pcolormesh(x_edges, y_edges, M_abs,
                             cmap=cmap_shared, vmin=vmin, vmax=vmax, shading="auto")

        # Cell annotations
        for i in range(n_tasks):
            for j in range(n_bins):
                val_abs = M_abs[i, j]
                if not np.isfinite(val_abs):
                    continue
                if is_pinball_base:
                    disp = val_abs * 1000.0
                    if abs(disp) < 5.0:
                        continue
                    # Match legacy panel-B annotation behavior for pinball outputs.
                    disp = np.ceil(disp * 10.0) / 10.0
                else:
                    signed = S_signed[i, j] if (S_signed.size and np.isfinite(S_signed[i, j])) else val_abs
                    if abs(signed) * 100.0 < 5.0:
                        continue
                    disp = signed * 100.0
                rgba = mesh.cmap(mesh.norm(val_abs))
                r_lum, g_lum, b_lum, _ = rgba
                luminance = 0.2126 * r_lum + 0.7152 * g_lum + 0.0722 * b_lum
                txt_color = "white" if luminance < 0.5 else "black"
                ax.text(j + 0.5, i + 0.5, rf"\textbf{{{disp:.1f}}}",
                        ha="center", va="center", fontsize=9.5, color=txt_color)

        ax.set_xlim(0, n_bins)
        ax.set_ylim(0, n_tasks)
        ax.set_aspect("equal", adjustable="box")

        # X ticks from heatmap ticks (xt)
        xticklabels = xt[:n_bins] if xt else ["" for _ in range(n_bins)]
        ax.set_xticks(np.arange(n_bins) + 0.5)
        ax.set_xticklabels(xticklabels, rotation=38, ha="right", fontsize=9.5)

        # Y ticks as task labels only for leftmost subplot
        if c == 0:
            ylabels = [t.replace(" Raw", "") for t in panel_tasks]
            ax.set_yticks(np.arange(n_tasks) + 0.5)
            ax.set_yticklabels([rf"\textbf{{{y}}}" for y in ylabels], fontsize=11)
        else:
            ax.set_yticks([])

        # k badge consistent with main OOD heatmap
        from matplotlib.offsetbox import AnchoredText  # type: ignore
        badge = AnchoredText(fr"$t={k}\mid$ OOD", loc="upper left",
                             prop=dict(size=16, color="#333", fontweight="bold"))
        try:
            badge.patch.set_facecolor("white")
            badge.patch.set_alpha(0.85)
            badge.patch.set_edgecolor("#a0a0a0")
            badge.patch.set_linewidth(1.0)
        except Exception:
            pass
        ax.add_artist(badge)

        # Subtle separators
        for i in range(1, n_bins):
            ax.axvline(i, color="white", linewidth=0.6, alpha=0.5)
        for i in range(1, n_tasks):
            ax.axhline(i, color="white", linewidth=0.6, alpha=0.5)

    title_fs = 22

    if mesh is not None:
        cax = figB.add_axes([0.90, 0.25, 0.020, 0.60])
        cb = figB.colorbar(mesh, cax=cax)
        cb.set_label(
            r"$\rho_\tau$" if is_pinball_base else r"$\hat{\tau}-\tau$",
            fontsize=title_fs,
            labelpad=10,
            weight="bold",
        )
        cb.ax.tick_params(labelsize=10)

    figB.supxlabel(fr"\textbf{{{BIN_X_LABEL}}}", fontsize=20, y=0.02)
    figB.suptitle(
        r"\textbf{OOD Pinball Loss}"
        if is_pinball_base
        else r"\textbf{OOD Coverage Error $\hat{\tau}-\tau$}",
        fontsize=title_fs,
        y=0.97,
    )

    figB.savefig(os.path.join(out_dir, "panel_b_heatmaps.png"), dpi=300, bbox_inches="tight")
    figB.savefig(os.path.join(out_dir, "panel_b_heatmaps.pdf"), dpi=300, bbox_inches="tight")
    plt.close(figB)

    # =========================================================================
    # PANEL C: Model density histogram (all train periods combined)
    # =========================================================================
    if density_qwen is not None and density_other is not None and bins_ref is not None:
        figC = plt.figure(figsize=(5.6, 2.8))
        axC = figC.add_subplot(111)

        nbins = len(bins_ref) - 1
        x_pos = np.arange(nbins)
        width = 0.8
        xticklabels = []
        for hi_log in bins_ref[1:]:
            try:
                hi_val = 10.0 ** float(hi_log)
                s = f"{hi_val:.2e}"
                mantissa, exp_str = s.split("e")
                exp_val = int(exp_str)
                xticklabels.append(rf"${mantissa}\times 10^{{{exp_val}}}$")
            except Exception:
                xticklabels.append("")

        axC.bar(x_pos, density_other, width=width, color="#6A9BC3",
                label="Other", align="center", edgecolor="white", linewidth=0.6)
        axC.bar(x_pos, density_qwen, width=width, color="#D64545",
                label="Qwen", align="center", bottom=density_other,
                edgecolor="white", linewidth=0.6)

        axC.set_xticks(x_pos)
        axC.set_xticklabels(xticklabels, rotation=32, ha="right", fontsize=10)
        axC.set_xlabel(rf"\textbf{{{BIN_X_LABEL}}}", fontsize=12)
        axC.set_ylabel(r"\textbf{\# models}", fontsize=12)
        axC.legend(fontsize=10, loc="upper right")
        axC.set_title(r"\textbf{Model density by compute bin (train $k=1..4$)}",
                     fontsize=13, pad=6)
        axC.grid(axis="y", alpha=0.3, linestyle="--", linewidth=0.7)
        axC.tick_params(labelsize=10)

        figC.tight_layout(pad=0.2)
        figC.savefig(os.path.join(out_dir, "panel_c_density.png"), dpi=300, bbox_inches="tight")
        figC.savefig(os.path.join(out_dir, "panel_c_density.pdf"), dpi=300, bbox_inches="tight")
        plt.close(figC)

    # Also save legacy combined version for backward compatibility
    if len(k_dirs) == 3:
        _plot_calib_summary_panels_legacy(
            base_dir,
            stable_series,
            drift_series,
            drift_heat,
            drift_edges,
            density_qwen,
            density_other,
            bins_ref,
        )


def _plot_calib_summary_panels_legacy(base_dir: str, stable_series: Dict, drift_series: Dict,
                                      drift_heat: Dict, drift_edges: Dict,
                                      density_qwen, density_other, bins_ref) -> None:
    """Legacy combined panel layout (kept for backward compatibility)."""
    is_pinball_base = "evaluation_pinball" in os.path.abspath(base_dir)
    out_dir = os.path.join(base_dir, "plots")
    eval_dirs = [os.path.join(base_dir, f"k{i}") for i in (1, 2, 3)]
    drift_tasks = ["IFEval Raw", "MATH Lvl 5 Raw"]
    tau = float(PLOT_TAU)

    mpl.rcParams["axes.spines.top"] = False
    mpl.rcParams["axes.spines.right"] = False
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(4, 3, height_ratios=[1.0, 0.7, 0.7, 0.9], hspace=0.35, wspace=0.3)

    # Panel A
    axA = fig.add_subplot(gs[0, :])
    xk = np.array([1, 2, 3])
    for t, vals in stable_series.items():
        axA.plot(xk, vals, marker="o", markersize=7, label=t, alpha=0.8, linewidth=1.8)
    for t, vals in drift_series.items():
        axA.plot(xk, vals, marker="o", markersize=8, label=t, linewidth=2.8)
    axA.set_title(
        r"OOD macro pinball loss by period" if is_pinball_base else r"OOD macro coverage error by period",
        fontweight="bold",
        fontsize=12,
        pad=10,
    )
    axA.set_xlabel(r"Period $k$", fontsize=11)
    axA.set_ylabel(
        r"$\mathrm{mean}\,\rho_\tau$" if is_pinball_base else r"$\mathrm{mean}\,|\hat{\tau}-\tau|$",
        fontsize=11,
    )
    axA.legend(frameon=False, fontsize=10, ncol=3, loc="upper right")
    axA.grid(axis="y", alpha=0.3, linestyle="--", linewidth=0.7)
    axA.set_xticks([1, 2, 3])

    # Panel B
    vmax = 0.05
    if is_pinball_base:
        vmax_candidates = []
        for t in drift_tasks:
            for vals in drift_heat.get(t, []):
                if isinstance(vals, np.ndarray) and vals.size:
                    vmax_candidates.append(vals[np.isfinite(vals)])
        if vmax_candidates:
            vmax = float(np.nanmax(np.concatenate(vmax_candidates)))
            if not np.isfinite(vmax) or vmax <= 0.0:
                vmax = 1.0
    norm = TwoSlopeNorm(vcenter=0.0, vmin=-0.05, vmax=0.05)
    mesh = None
    axes_B = [[fig.add_subplot(gs[1, c]) for c in range(3)], [fig.add_subplot(gs[2, c]) for c in range(3)]]

    for r, task in enumerate(drift_tasks):
        task_short = task.replace(" Raw", "")
        for c, ed in enumerate(eval_dirs):
            ax = axes_B[r][c]
            vals = drift_heat[task][c]
            edges = drift_edges[task][c]
            if vals.size == 0 or edges.size == 0:
                ax.axis("off")
                continue
            if is_pinball_base:
                cmap_shared = _make_heatmap_cmap()
                mesh = ax.pcolormesh(
                    edges, [0, 1], vals.reshape(1, -1), cmap=cmap_shared, vmin=0.0, vmax=vmax, shading="auto"
                )
            else:
                mesh = ax.pcolormesh(edges, [0, 1], vals.reshape(1, -1), cmap="RdBu_r", norm=norm, shading="auto")
                for i, v in enumerate(vals):
                    if abs(v) > 0.05:
                        ax.add_patch(plt.Rectangle((edges[i], 0), edges[i + 1] - edges[i], 1,
                                                  fill=False, edgecolor="k", linewidth=2.0))
            ax.set_xlim(edges[0], edges[-1])
            ax.set_ylim(0, 1)
            if len(edges) > 8:
                tick_indices = np.linspace(0, len(edges) - 1, 7, dtype=int)
                ax.set_xticks(edges[tick_indices])
                ax.set_xticklabels([f"{edges[i]:.2f}" for i in tick_indices], rotation=40, ha="right", fontsize=8.5)
            else:
                ax.set_xticks(edges)
                ax.set_xticklabels([f"{e:.2f}" for e in edges], rotation=40, ha="right", fontsize=8.5)
            ax.set_yticks([])
            ax.set_aspect("auto")
            if c == 0:
                ax.set_ylabel(task_short, fontweight="bold", fontsize=10)
            ax.set_title(rf"$k={c+1}$", fontweight="bold", fontsize=10)
            for edge in edges:
                ax.axvline(edge, color="white", linewidth=0.5, alpha=0.4)

    if mesh is not None:
        cax = fig.add_axes([0.92, 0.38, 0.015, 0.22])
        cb = fig.colorbar(mesh, cax=cax)
        cb.set_label(r"$\rho_\tau$" if is_pinball_base else r"$\hat{\tau}-\tau$", fontsize=11, labelpad=8)
        cb.ax.tick_params(labelsize=9)

    # Panel C
    if density_qwen is not None and density_other is not None and bins_ref is not None:
        axC = fig.add_subplot(gs[3, :])
        centers = 0.5 * (bins_ref[:-1] + bins_ref[1:])
        width = np.diff(bins_ref) * 0.95
        axC.bar(centers, density_other, width=width, color="#7FA8D1",
                label="Other", align="center", edgecolor="white", linewidth=0.5)
        axC.bar(centers, density_qwen, width=width, color="#E74C3C",
                label="Qwen", align="center", bottom=density_other, edgecolor="white", linewidth=0.5)
        axC.set_xlabel(r"Bin Upper FLOPs $(\times 10^{21})$", fontsize=11)
        axC.set_ylabel(r"\# models", fontsize=11)
        axC.legend(frameon=False, fontsize=10, loc="upper right")
        axC.set_title(r"Model density by compute bin (train $k=1$)", fontweight="bold", fontsize=12, pad=10)
        axC.grid(axis="y", alpha=0.3, linestyle="--", linewidth=0.7)
        axC.tick_params(labelsize=9)

    fig.savefig(os.path.join(out_dir, "calib_summary_panels.png"), dpi=300, bbox_inches="tight")
    fig.savefig(os.path.join(out_dir, "calib_summary_panels.pdf"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description="Plot evaluation summaries for sigmoid frontier")
    ap.add_argument(
        "--eval_dir",
        default=os.path.join("outputs", "evaluation", "sigmoid", "year_split", "no_budget"),
    )
    ap.add_argument(
        "--out_dir",
        default=os.path.join(
            "outputs", "evaluation", "sigmoid", "year_split", "no_budget", "plots"
        ),
    )
    ap.add_argument("--no_cell_values", action="store_true", help="Disable numeric annotations inside heatmap cells")
    ap.add_argument("--period4_singlek_base", default=None, help="If set, combine k*/ heatmaps (train+OOD) into a single 1x(2K) figure. Pass base dir like outputs/eval/period4_singlek")
    ap.add_argument("--bin_x_label", default=None, help="Override x-axis label for bin upper bound (e.g. 'Bin Upper Params (B)')")
    ap.add_argument("--tau", type=float, default=None, help="Tau used for signed annotations (hat_tau - tau); colors use abs_err from CSVs.")
    ap.add_argument("--make_calib_summary_panels", action="store_true", help="Generate summary panels A–C (period stability lines, minimal bin heatmaps for IFEval/MATH, model density by bin)")
    args = ap.parse_args()

    global BIN_X_LABEL
    if args.bin_x_label:
        BIN_X_LABEL = args.bin_x_label
    global PLOT_TAU
    if args.tau is not None:
        PLOT_TAU = float(args.tau)

    # Combined 1x6 grid mode for period4_singlek
    if args.period4_singlek_base:
        if args.make_calib_summary_panels:
            _plot_calib_summary_panels(args.period4_singlek_base)
        out_dir_main = os.path.join(args.period4_singlek_base, "plots")
        os.makedirs(out_dir_main, exist_ok=True)
        # Main 6-task grids
        _plot_heatmap_grid_period4_singlek(args.period4_singlek_base, out_dir_main, annotate_cells=(not args.no_cell_values))
        _plot_bars_grid_period4_singlek(args.period4_singlek_base, out_dir_main)
        _plot_heatmap_grid_period4_singlek_oos_only(args.period4_singlek_base, out_dir_main, annotate_cells=(not args.no_cell_values))
        _plot_bars_grid_period4_singlek_oos_only(args.period4_singlek_base, out_dir_main)
        # BBH subtask grids in dedicated subfolder
        out_dir_sub = os.path.join(args.period4_singlek_base, "plots_subtasks")
        os.makedirs(out_dir_sub, exist_ok=True)
        _plot_heatmap_grid_period4_singlek_bbh(args.period4_singlek_base, out_dir_sub, annotate_cells=(not args.no_cell_values))
        _plot_bars_grid_period4_singlek_bbh(args.period4_singlek_base, out_dir_sub)
        _plot_heatmap_grid_period4_singlek_oos_only_bbh(args.period4_singlek_base, out_dir_sub, annotate_cells=(not args.no_cell_values))
        _plot_bars_grid_period4_singlek_oos_only_bbh(args.period4_singlek_base, out_dir_sub)
        return

    tasks = _detect_tasks(args.eval_dir)
    if not tasks:
        raise SystemExit(f"No task summaries found in {args.eval_dir}")
    # read summaries
    is_vals = []
    oos_vals = []
    for t in tasks:
        path = _get_summary_path(args.eval_dir, t)
        if not path:
            raise SystemExit(f"Missing summary CSV for task '{t}' under {args.eval_dir}")
        _t, is_macro, oos_macro, _, _ = _read_summary(path)
        is_vals.append(is_macro)
        oos_vals.append(oos_macro)
    is_arr = np.array(is_vals, float)
    oos_arr = np.array(oos_vals, float)

    display_names = [_display_name(t) for t in tasks]

    os.makedirs(args.out_dir, exist_ok=True)
    _plot_bars(args.eval_dir, args.out_dir, tasks, display_names, is_arr, oos_arr)
    _plot_scatter(args.eval_dir, args.out_dir, display_names, is_arr, oos_arr)
    annotate = (not args.no_cell_values)
    _plot_heatmap(args.eval_dir, args.out_dir, tasks, display_names, which="train", annotate_cells=annotate)
    _plot_heatmap(args.eval_dir, args.out_dir, tasks, display_names, which="test", annotate_cells=annotate)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Plot baseline comparisons for pinball-loss frontiers in the period4 single-k, no-budget setting.

Figure 1:
  - Global pinball loss (×1e3) for Null, Sigmoid, Oracle per task and period k.

Inputs:
  evaluation_pinball_baselines/period4_singlek_no_budget/k{1,2,3}/*__pinball_baselines.csv

Outputs:
  evaluation_pinball_baselines/period4_singlek_no_budget/plots/fig1_global_pinball_bars.(png|pdf)
"""

from __future__ import annotations

import os
import argparse
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np

import matplotlib.pyplot as plt  # type: ignore
import matplotlib as mpl  # type: ignore

# Import bootstrap from `scripts/` by putting the scripts dir on sys.path.
SCRIPTS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)
from _bootstrap import ensure_repo_root_on_path  # type: ignore

REPO_ROOT = ensure_repo_root_on_path(__file__)

from skill_frontier.io.task_mappings import format_task_label  # type: ignore


BASE_DIR_MAIN = os.path.join("evaluation_pinball_baselines", "period4_singlek_no_budget")
BASE_DIR_BBH = os.path.join("evaluation_pinball_baselines", "period4_singlek_no_budget_bbh_subtasks")
PLOTS_DIR_MAIN = os.path.join(BASE_DIR_MAIN, "plots")
PLOTS_DIR_BBH = os.path.join(BASE_DIR_MAIN, "plots_subtasks")

# Styling constants for the "relative vs constant" / "coverage relative vs constant" line plots.
FIG_LINE_WIDTH = 5.0  # was 7.0
FIG_LINE_HEIGHT = 4.0
AXIS_LABEL_FONTSIZE = 18
XTICK_LABEL_FONTSIZE = 16
TICK_LABEL_FONTSIZE = 16
RIGHT_LABEL_MIN_SEP = 5.0  # increase to avoid overlap
RIGHT_LABEL_X_OFFSET = 0.12  # slightly more breathing room for right-side labels
X_LABEL_PERIOD_PK = r"\textbf{Period} $\mathcal{P}_k$"


def _setup_rcparams() -> None:
    """Configure Matplotlib global style for publication-quality plots."""
    try:
        mpl.rcParams["text.usetex"] = True
        mpl.rcParams["text.latex.preamble"] = r"\usepackage{amsmath}"
    except Exception:
        pass
    try:
        mpl.rcParams["font.family"] = "serif"
    except Exception:
        pass
    mpl.rcParams["axes.spines.top"] = False
    mpl.rcParams["axes.spines.right"] = False


def _load_pinball_baselines_for_k(
    k_idx: int, base_dir: str, allowed_tasks: Optional[set] = None
) -> Tuple[List[str], Dict[str, Dict[str, Dict[str, float]]]]:
    """Load per-task pinball baseline CSVs for a given k.

    Returns:
        tasks: ordered list of task names.
        metrics: metrics[task][split] -> dict with keys:
            L_sigmoid, L_ispline, L_null, L_oracle, ratio_sigmoid_oracle, R2_pinball,
            and (if present) CE_sigmoid, CE_ispline, CE_null, CE_oracle.
    """
    k_dir = os.path.join(base_dir, f"k{k_idx}")
    metrics: Dict[str, Dict[str, Dict[str, float]]] = {}
    tasks: List[str] = []
    for fn in sorted(os.listdir(k_dir)):
        if not fn.endswith("__pinball_baselines.csv"):
            continue
        task = fn[: -len("__pinball_baselines.csv")]
        if allowed_tasks is not None and task not in allowed_tasks:
            continue
        path = os.path.join(k_dir, fn)
        with open(path, "r", newline="") as f:
            import csv  # type: ignore

            r = csv.DictReader(f)
            per_split: Dict[str, Dict[str, float]] = {}
            for row in r:
                split = row["split"]
                per_split[split] = {
                    "L_sigmoid": float(row["L_sigmoid"]),
                    "L_ispline": float(row.get("L_ispline", "nan")),
                    "L_null": float(row["L_null"]),
                    "L_oracle": float(row["L_oracle"]),
                    "CE_sigmoid": float(row.get("CE_sigmoid", "nan")),
                    "CE_ispline": float(row.get("CE_ispline", "nan")),
                    "CE_null": float(row.get("CE_null", "nan")),
                    "CE_oracle": float(row.get("CE_oracle", "nan")),
                    "ratio_sigmoid_oracle": float(row["ratio_sigmoid_oracle"]),
                    "R2_pinball": float(row["R2_pinball"]),
                }
        metrics[task] = per_split
        tasks.append(task)
    return tasks, metrics


def _collect_baseline_matrices(base_dir: str, allowed_tasks: Optional[set] = None):
    """Aggregate per-task/k pinball losses into dense matrices.

    Returns:
        tasks: list of raw task names (from k=1).
        display_names: pretty task labels.
        data: nested dict data[method][split] -> array of shape (K, T),
              where K=3 (k=1,2,3) and T=len(tasks).
              Methods: 'const', 'binwise', 'sigmoid', 'ispline'.
    """
    tasks_k1, metrics_k1 = _load_pinball_baselines_for_k(1, base_dir, allowed_tasks=allowed_tasks)
    tasks = tasks_k1
    display_names = [_display_task_name(t) for t in tasks]
    metrics = {
        1: metrics_k1,
        2: _load_pinball_baselines_for_k(2, base_dir, allowed_tasks=allowed_tasks)[1],
        3: _load_pinball_baselines_for_k(3, base_dir, allowed_tasks=allowed_tasks)[1],
    }

    K = 3
    T = len(tasks)
    methods = ("const", "binwise", "sigmoid", "ispline")
    splits = ("train", "val")
    data: Dict[str, Dict[str, np.ndarray]] = {
        m: {s: np.full((K, T), np.nan, dtype=float) for s in splits} for m in methods
    }

    for k_idx in (1, 2, 3):
        mk = metrics[k_idx]
        row = k_idx - 1
        for t_idx, task in enumerate(tasks):
            per_split = mk.get(task)
            if not per_split:
                continue
            for split in splits:
                row_dict = per_split.get(split)
                if not row_dict:
                    continue
                data["const"][split][row, t_idx] = row_dict["L_null"]
                data["binwise"][split][row, t_idx] = row_dict["L_oracle"]
                data["sigmoid"][split][row, t_idx] = row_dict["L_sigmoid"]
                data["ispline"][split][row, t_idx] = row_dict.get("L_ispline", float("nan"))

    return tasks, display_names, data


def _collect_calibration_matrices(base_dir: str, allowed_tasks: Optional[set] = None):
    """Aggregate per-task/k coverage errors into dense matrices.

    Uses CE_* columns written by scripts/evaluate/sigmoid_pinball_baselines.py.
    Returns:
        tasks, display_names, data_ce with same structure as _collect_baseline_matrices.
    """
    tasks_k1, metrics_k1 = _load_pinball_baselines_for_k(1, base_dir, allowed_tasks=allowed_tasks)
    tasks = tasks_k1
    display_names = [_display_task_name(t) for t in tasks]
    metrics = {
        1: metrics_k1,
        2: _load_pinball_baselines_for_k(2, base_dir, allowed_tasks=allowed_tasks)[1],
        3: _load_pinball_baselines_for_k(3, base_dir, allowed_tasks=allowed_tasks)[1],
    }

    K = 3
    T = len(tasks)
    methods = ("const", "binwise", "sigmoid", "ispline")
    splits = ("train", "val")
    data: Dict[str, Dict[str, np.ndarray]] = {
        m: {s: np.full((K, T), np.nan, dtype=float) for s in splits} for m in methods
    }

    for k_idx in (1, 2, 3):
        mk = metrics[k_idx]
        row = k_idx - 1
        for t_idx, task in enumerate(tasks):
            per_split = mk.get(task)
            if not per_split:
                continue
            for split in splits:
                row_dict = per_split.get(split)
                if not row_dict:
                    continue
                data["const"][split][row, t_idx] = row_dict.get("CE_null", float("nan"))
                data["binwise"][split][row, t_idx] = row_dict.get("CE_oracle", float("nan"))
                data["sigmoid"][split][row, t_idx] = row_dict.get("CE_sigmoid", float("nan"))
                data["ispline"][split][row, t_idx] = row_dict.get("CE_ispline", float("nan"))

    return tasks, display_names, data


def _display_task_name(task: str) -> str:
    """Human-readable task label (strip ' Raw' and BBH prefixes)."""
    return format_task_label(task)


def _plot_for_base(base_dir: str, plots_dir: str, suffix: str = "", allowed_tasks: Optional[set] = None) -> None:
    """Baseline comparison plots for a given base directory."""
    os.makedirs(plots_dir, exist_ok=True)

    tasks, display_names, data = _collect_baseline_matrices(base_dir, allowed_tasks=allowed_tasks)
    _, _, data_ce = _collect_calibration_matrices(base_dir, allowed_tasks=allowed_tasks)

    # Determine task ordering (by mean constant OOS loss over k)
    const_oos = data["const"]["val"]  # (K, T)
    mean_const_oos = np.nanmean(const_oos, axis=0)
    order = np.argsort(mean_const_oos)
    display_names_ord = [display_names[i] for i in order]

    # Method metadata
    methods = ["const", "binwise", "sigmoid", "ispline"]
    method_labels_panelA = {
        "const": "Constant",
        "binwise": "Binwise",
        "sigmoid": "Sigmoid",
        "ispline": "I-spline",
    }
    colors = {
        "const": "#b0b0b0",
        "binwise": "#636363",
        "sigmoid": "#b22222",
        "ispline": "#1f77b4",
    }

    # ---------------- Figure A: Avg OOS relative loss vs constant ----------------
    figA, axA = plt.subplots(figsize=(FIG_LINE_WIDTH, FIG_LINE_HEIGHT))
    x_k = np.array([1, 2, 3], dtype=float)
    const_oos_ord = const_oos[:, order]

    placed_y: List[float] = []
    min_sep = RIGHT_LABEL_MIN_SEP  # enforce minimum vertical separation between labels

    def _place_label(x_pos: float, y_val: float, text: str, color: str) -> None:
        y_target = y_val
        for existing in placed_y:
            if abs(y_target - existing) < min_sep:
                # Nudge upward just enough to clear the closest label
                y_target = existing + min_sep
        placed_y.append(y_target)
        axA.text(
            x_pos,
            y_target,
            text,
            color=color,
            fontsize=14,
            va="center",
        )

    for m in methods:
        L_m = data[m]["val"][:, order]
        rel = np.full_like(L_m, np.nan, dtype=float)
        mask = np.isfinite(L_m) & np.isfinite(const_oos_ord) & (const_oos_ord > 0.0)
        rel[mask] = 100.0 * (L_m[mask] - const_oos_ord[mask]) / const_oos_ord[mask]
        rel_mean = np.nanmean(rel, axis=1)
        if m == "const":
            rel_mean = np.zeros_like(rel_mean)

        axA.plot(
            x_k,
            rel_mean,
            marker="o",
            linewidth=2.0 if m in ("sigmoid", "ispline") else 1.5,
            markersize=6 if m in ("sigmoid", "ispline") else 5,
            color=colors[m],
            alpha=0.9,
        )

        y_last = rel_mean[-1]
        _place_label(x_k[-1] + RIGHT_LABEL_X_OFFSET, y_last, method_labels_panelA[m], colors[m])

    axA.set_xticks(x_k)
    axA.set_xticklabels([r"$k=1$", r"$k=2$", r"$k=3$"], fontsize=XTICK_LABEL_FONTSIZE)
    axA.set_xlabel(X_LABEL_PERIOD_PK, fontsize=AXIS_LABEL_FONTSIZE)
    axA.set_ylabel(r"\textbf{Relative loss (\%)}", fontsize=AXIS_LABEL_FONTSIZE)
    axA.tick_params(axis="both", which="major", labelsize=TICK_LABEL_FONTSIZE)
    axA.set_title(
        r"\textbf{Avg. OOD pinball loss vs constant (\%)}",
        fontsize=18,
        pad=15,
    )
    axA.grid(axis="y", linestyle="-", linewidth=0.5, alpha=0.3)
    figA.tight_layout()
    figA.savefig(os.path.join(plots_dir, f"fig_oos_relative_vs_constant{suffix}.png"), dpi=300, bbox_inches="tight")
    figA.savefig(os.path.join(plots_dir, f"fig_oos_relative_vs_constant{suffix}.pdf"), dpi=300, bbox_inches="tight")
    plt.close(figA)

    # ---------------- Figure A (IS): Avg IS relative loss vs constant ----------------
    figA_is, axA_is = plt.subplots(figsize=(FIG_LINE_WIDTH, FIG_LINE_HEIGHT))
    x_k = np.array([1, 2, 3], dtype=float)
    const_is = data["const"]["train"][:, order]

    placed_y = []
    min_sep = RIGHT_LABEL_MIN_SEP  # enforce minimum vertical separation between labels

    def _place_label_is(x_pos: float, y_val: float, text: str, color: str) -> None:
        y_target = y_val
        for existing in placed_y:
            if abs(y_target - existing) < min_sep:
                y_target = existing + min_sep
        placed_y.append(y_target)
        axA_is.text(
            x_pos,
            y_target,
            text,
            color=color,
            fontsize=14,
            va="center",
        )

    for m in methods:
        L_m = data[m]["train"][:, order]
        rel = np.full_like(L_m, np.nan, dtype=float)
        mask = np.isfinite(L_m) & np.isfinite(const_is) & (const_is > 0.0)
        rel[mask] = 100.0 * (L_m[mask] - const_is[mask]) / const_is[mask]
        rel_mean = np.nanmean(rel, axis=1)
        if m == "const":
            rel_mean = np.zeros_like(rel_mean)

        axA_is.plot(
            x_k,
            rel_mean,
            marker="o",
            linewidth=2.0 if m in ("sigmoid", "ispline") else 1.5,
            markersize=6 if m in ("sigmoid", "ispline") else 5,
            color=colors[m],
            alpha=0.9,
        )

        y_last = rel_mean[-1]
        _place_label_is(x_k[-1] + RIGHT_LABEL_X_OFFSET, y_last, method_labels_panelA[m], colors[m])

    axA_is.set_xticks(x_k)
    axA_is.set_xticklabels([r"$k=1$", r"$k=2$", r"$k=3$"], fontsize=XTICK_LABEL_FONTSIZE)
    axA_is.set_xlabel(X_LABEL_PERIOD_PK, fontsize=AXIS_LABEL_FONTSIZE)
    axA_is.set_ylabel(r"\textbf{Relative loss (\%)}", fontsize=AXIS_LABEL_FONTSIZE)
    axA_is.tick_params(axis="both", which="major", labelsize=TICK_LABEL_FONTSIZE)
    axA_is.set_title(
        r"\textbf{Avg. ID pinball loss vs constant (\%)}",
        fontsize=18,
        pad=15,
    )
    axA_is.grid(axis="y", linestyle="-", linewidth=0.5, alpha=0.3)
    figA_is.tight_layout()
    figA_is.savefig(os.path.join(plots_dir, f"fig_is_relative_vs_constant{suffix}.png"), dpi=300, bbox_inches="tight")
    figA_is.savefig(os.path.join(plots_dir, f"fig_is_relative_vs_constant{suffix}.pdf"), dpi=300, bbox_inches="tight")
    plt.close(figA_is)

    # ---------------- Figure A (IS + OOS): Avg relative loss vs constant ----------------
    figA_both, axA_both = plt.subplots(figsize=(FIG_LINE_WIDTH, FIG_LINE_HEIGHT))
    x_k = np.array([1, 2, 3], dtype=float)
    const_is = data["const"]["train"][:, order]
    const_oos_ord = const_oos[:, order]

    placed_y = []
    min_sep = RIGHT_LABEL_MIN_SEP  # enforce minimum vertical separation between labels

    def _place_label_both(x_pos: float, y_val: float, text: str, color: str) -> None:
        y_target = y_val
        for existing in placed_y:
            if abs(y_target - existing) < min_sep:
                y_target = existing + min_sep
        placed_y.append(y_target)
        axA_both.text(
            x_pos,
            y_target,
            text,
            color=color,
            fontsize=14,
            va="center",
        )

    for m in methods:
        # IS
        L_is = data[m]["train"][:, order]
        rel_is = np.full_like(L_is, np.nan, dtype=float)
        mask_is = np.isfinite(L_is) & np.isfinite(const_is) & (const_is > 0.0)
        rel_is[mask_is] = 100.0 * (L_is[mask_is] - const_is[mask_is]) / const_is[mask_is]
        rel_is_mean = np.nanmean(rel_is, axis=1)
        if m == "const":
            rel_is_mean = np.zeros_like(rel_is_mean)

        # OOS
        L_oos = data[m]["val"][:, order]
        rel_oos = np.full_like(L_oos, np.nan, dtype=float)
        mask_oos = np.isfinite(L_oos) & np.isfinite(const_oos_ord) & (const_oos_ord > 0.0)
        rel_oos[mask_oos] = 100.0 * (L_oos[mask_oos] - const_oos_ord[mask_oos]) / const_oos_ord[mask_oos]
        rel_oos_mean = np.nanmean(rel_oos, axis=1)
        if m == "const":
            rel_oos_mean = np.zeros_like(rel_oos_mean)

        axA_both.plot(
            x_k,
            rel_is_mean,
            marker="o",
            linewidth=2.0 if m in ("sigmoid", "ispline") else 1.5,
            markersize=6 if m in ("sigmoid", "ispline") else 5,
            color=colors[m],
            alpha=0.9,
            linestyle="-",
        )
        axA_both.plot(
            x_k,
            rel_oos_mean,
            marker="o",
            linewidth=2.0 if m in ("sigmoid", "ispline") else 1.5,
            markersize=6 if m in ("sigmoid", "ispline") else 5,
            color=colors[m],
            alpha=0.9,
            linestyle="--",
        )

        y_last = rel_is_mean[-1]
        _place_label_both(x_k[-1] + RIGHT_LABEL_X_OFFSET, y_last, method_labels_panelA[m], colors[m])

    from matplotlib.lines import Line2D  # type: ignore
    axA_both.legend(
        handles=[
            Line2D([0], [0], color="#333333", linestyle="-", linewidth=2.0, label="ID"),
            Line2D([0], [0], color="#333333", linestyle="--", linewidth=2.0, label="OOD"),
        ],
        loc="upper right",
        fontsize=14,
        frameon=False,
        handlelength=2.0,
        borderaxespad=0.3,
    )

    axA_both.set_xticks(x_k)
    axA_both.set_xticklabels([r"$k=1$", r"$k=2$", r"$k=3$"], fontsize=XTICK_LABEL_FONTSIZE)
    axA_both.set_xlabel(X_LABEL_PERIOD_PK, fontsize=AXIS_LABEL_FONTSIZE)
    axA_both.set_ylabel(r"\textbf{Relative loss (\%)}", fontsize=AXIS_LABEL_FONTSIZE)
    axA_both.tick_params(axis="both", which="major", labelsize=TICK_LABEL_FONTSIZE)
    axA_both.set_title(
        r"\textbf{Avg. pinball loss vs constant (\%)}",
        fontsize=18,
        pad=15,
    )
    axA_both.grid(axis="y", linestyle="-", linewidth=0.5, alpha=0.3)
    figA_both.tight_layout()
    figA_both.savefig(os.path.join(plots_dir, f"fig_is_oos_relative_vs_constant{suffix}.png"), dpi=300, bbox_inches="tight")
    figA_both.savefig(os.path.join(plots_dir, f"fig_is_oos_relative_vs_constant{suffix}.pdf"), dpi=300, bbox_inches="tight")
    plt.close(figA_both)

    # ---------------- Figure A2: Avg OOS relative coverage error vs constant ----------------
    figA2, axA2 = plt.subplots(figsize=(FIG_LINE_WIDTH, FIG_LINE_HEIGHT))
    x_k = np.array([1, 2, 3], dtype=float)
    const_oos_ce = data_ce["const"]["val"][:, order]

    placed_y = []
    min_sep = RIGHT_LABEL_MIN_SEP  # enforce minimum vertical separation between labels

    def _place_label2(x_pos: float, y_val: float, text: str, color: str) -> None:
        y_target = y_val
        for existing in placed_y:
            if abs(y_target - existing) < min_sep:
                y_target = existing + min_sep
        placed_y.append(y_target)
        axA2.text(
            x_pos,
            y_target,
            text,
            color=color,
            fontsize=14,
            va="center",
        )

    for m in methods:
        E_m = data_ce[m]["val"][:, order]
        rel = np.full_like(E_m, np.nan, dtype=float)
        mask = np.isfinite(E_m) & np.isfinite(const_oos_ce) & (const_oos_ce > 0.0)
        rel[mask] = 100.0 * (E_m[mask] - const_oos_ce[mask]) / const_oos_ce[mask]
        rel_mean = np.nanmean(rel, axis=1)
        if m == "const":
            rel_mean = np.zeros_like(rel_mean)

        axA2.plot(
            x_k,
            rel_mean,
            marker="o",
            linewidth=2.0 if m in ("sigmoid", "ispline") else 1.5,
            markersize=6 if m in ("sigmoid", "ispline") else 5,
            color=colors[m],
            alpha=0.9,
        )

        y_last = rel_mean[-1]
        if np.isfinite(y_last):
            _place_label2(x_k[-1] + RIGHT_LABEL_X_OFFSET, y_last, method_labels_panelA[m], colors[m])

    axA2.set_xticks(x_k)
    axA2.set_xticklabels([r"$k=1$", r"$k=2$", r"$k=3$"], fontsize=XTICK_LABEL_FONTSIZE)
    axA2.set_xlabel(X_LABEL_PERIOD_PK, fontsize=AXIS_LABEL_FONTSIZE)
    axA2.set_ylabel(r"\textbf{Relative error (\%)}", fontsize=AXIS_LABEL_FONTSIZE)
    axA2.tick_params(axis="both", which="major", labelsize=TICK_LABEL_FONTSIZE)
    axA2.set_title(
        r"\textbf{Avg. OOD coverage error vs constant (\%)}",
        fontsize=18,
        pad=15,
    )
    axA2.grid(axis="y", linestyle="-", linewidth=0.5, alpha=0.3)
    figA2.tight_layout()
    figA2.savefig(
        os.path.join(plots_dir, f"fig_oos_calibration_relative_vs_constant{suffix}.png"),
        dpi=300,
        bbox_inches="tight",
    )
    figA2.savefig(
        os.path.join(plots_dir, f"fig_oos_calibration_relative_vs_constant{suffix}.pdf"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(figA2)

    # ---------------- Figure A2 (IS): Avg IS relative coverage error vs constant ----------------
    figA2_is, axA2_is = plt.subplots(figsize=(FIG_LINE_WIDTH, FIG_LINE_HEIGHT))
    x_k = np.array([1, 2, 3], dtype=float)
    const_is_ce = data_ce["const"]["train"][:, order]

    placed_y = []
    min_sep = RIGHT_LABEL_MIN_SEP  # enforce minimum vertical separation between labels

    def _place_label2_is(x_pos: float, y_val: float, text: str, color: str) -> None:
        y_target = y_val
        for existing in placed_y:
            if abs(y_target - existing) < min_sep:
                y_target = existing + min_sep
        placed_y.append(y_target)
        axA2_is.text(
            x_pos,
            y_target,
            text,
            color=color,
            fontsize=14,
            va="center",
        )

    for m in methods:
        E_m = data_ce[m]["train"][:, order]
        rel = np.full_like(E_m, np.nan, dtype=float)
        mask = np.isfinite(E_m) & np.isfinite(const_is_ce) & (const_is_ce > 0.0)
        rel[mask] = 100.0 * (E_m[mask] - const_is_ce[mask]) / const_is_ce[mask]
        rel_mean = np.nanmean(rel, axis=1)
        if m == "const":
            rel_mean = np.zeros_like(rel_mean)

        axA2_is.plot(
            x_k,
            rel_mean,
            marker="o",
            linewidth=2.0 if m in ("sigmoid", "ispline") else 1.5,
            markersize=6 if m in ("sigmoid", "ispline") else 5,
            color=colors[m],
            alpha=0.9,
        )

        y_last = rel_mean[-1]
        if np.isfinite(y_last):
            _place_label2_is(x_k[-1] + RIGHT_LABEL_X_OFFSET, y_last, method_labels_panelA[m], colors[m])

    axA2_is.set_xticks(x_k)
    axA2_is.set_xticklabels([r"$k=1$", r"$k=2$", r"$k=3$"], fontsize=XTICK_LABEL_FONTSIZE)
    axA2_is.set_xlabel(X_LABEL_PERIOD_PK, fontsize=AXIS_LABEL_FONTSIZE)
    axA2_is.set_ylabel(r"\textbf{Relative error (\%)}", fontsize=AXIS_LABEL_FONTSIZE)
    axA2_is.tick_params(axis="both", which="major", labelsize=TICK_LABEL_FONTSIZE)
    axA2_is.set_title(
        r"\textbf{Avg. ID coverage error vs constant (\%)}",
        fontsize=18,
        pad=15,
    )
    axA2_is.grid(axis="y", linestyle="-", linewidth=0.5, alpha=0.3)
    figA2_is.tight_layout()
    figA2_is.savefig(
        os.path.join(plots_dir, f"fig_is_calibration_relative_vs_constant{suffix}.png"),
        dpi=300,
        bbox_inches="tight",
    )
    figA2_is.savefig(
        os.path.join(plots_dir, f"fig_is_calibration_relative_vs_constant{suffix}.pdf"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(figA2_is)

    # ---------------- Figure A2 (IS + OOS): Avg relative coverage error vs constant ----------------
    figA2_both, axA2_both = plt.subplots(figsize=(FIG_LINE_WIDTH, FIG_LINE_HEIGHT))
    x_k = np.array([1, 2, 3], dtype=float)
    const_is_ce = data_ce["const"]["train"][:, order]
    const_oos_ce = data_ce["const"]["val"][:, order]

    placed_y = []
    min_sep = RIGHT_LABEL_MIN_SEP  # enforce minimum vertical separation between labels

    def _place_label_ce_both(x_pos: float, y_val: float, text: str, color: str) -> None:
        y_target = y_val
        for existing in placed_y:
            if abs(y_target - existing) < min_sep:
                y_target = existing + min_sep
        placed_y.append(y_target)
        axA2_both.text(
            x_pos,
            y_target,
            text,
            color=color,
            fontsize=14,
            va="center",
        )

    for m in methods:
        # IS
        E_is = data_ce[m]["train"][:, order]
        rel_is = np.full_like(E_is, np.nan, dtype=float)
        mask_is = np.isfinite(E_is) & np.isfinite(const_is_ce) & (const_is_ce > 0.0)
        rel_is[mask_is] = 100.0 * (E_is[mask_is] - const_is_ce[mask_is]) / const_is_ce[mask_is]
        rel_is_mean = np.nanmean(rel_is, axis=1)
        if m == "const":
            rel_is_mean = np.zeros_like(rel_is_mean)

        # OOS
        E_oos = data_ce[m]["val"][:, order]
        rel_oos = np.full_like(E_oos, np.nan, dtype=float)
        mask_oos = np.isfinite(E_oos) & np.isfinite(const_oos_ce) & (const_oos_ce > 0.0)
        rel_oos[mask_oos] = 100.0 * (E_oos[mask_oos] - const_oos_ce[mask_oos]) / const_oos_ce[mask_oos]
        rel_oos_mean = np.nanmean(rel_oos, axis=1)
        if m == "const":
            rel_oos_mean = np.zeros_like(rel_oos_mean)

        axA2_both.plot(
            x_k,
            rel_is_mean,
            marker="o",
            linewidth=2.0 if m in ("sigmoid", "ispline") else 1.5,
            markersize=6 if m in ("sigmoid", "ispline") else 5,
            color=colors[m],
            alpha=0.9,
            linestyle="-",
        )
        axA2_both.plot(
            x_k,
            rel_oos_mean,
            marker="o",
            linewidth=2.0 if m in ("sigmoid", "ispline") else 1.5,
            markersize=6 if m in ("sigmoid", "ispline") else 5,
            color=colors[m],
            alpha=0.9,
            linestyle="--",
        )

        y_last = rel_is_mean[-1]
        if np.isfinite(y_last):
            _place_label_ce_both(x_k[-1] + RIGHT_LABEL_X_OFFSET, y_last, method_labels_panelA[m], colors[m])

    from matplotlib.lines import Line2D  # type: ignore
    axA2_both.legend(
        handles=[
            Line2D([0], [0], color="#333333", linestyle="-", linewidth=2.0, label="ID"),
            Line2D([0], [0], color="#333333", linestyle="--", linewidth=2.0, label="OOD"),
        ],
        loc="upper right",
        fontsize=14,
        frameon=False,
        handlelength=2.0,
        borderaxespad=0.3,
    )

    axA2_both.set_xticks(x_k)
    axA2_both.set_xticklabels([r"$k=1$", r"$k=2$", r"$k=3$"], fontsize=XTICK_LABEL_FONTSIZE)
    axA2_both.set_xlabel(X_LABEL_PERIOD_PK, fontsize=AXIS_LABEL_FONTSIZE)
    axA2_both.set_ylabel(r"\textbf{Relative error (\%)}", fontsize=AXIS_LABEL_FONTSIZE)
    axA2_both.tick_params(axis="both", which="major", labelsize=TICK_LABEL_FONTSIZE)
    axA2_both.set_title(
        r"\textbf{Avg. coverage error vs constant (\%)}",
        fontsize=18,
        pad=15,
    )
    axA2_both.grid(axis="y", linestyle="-", linewidth=0.5, alpha=0.3)
    figA2_both.tight_layout()
    figA2_both.savefig(
        os.path.join(plots_dir, f"fig_is_oos_calibration_relative_vs_constant{suffix}.png"),
        dpi=300,
        bbox_inches="tight",
    )
    figA2_both.savefig(
        os.path.join(plots_dir, f"fig_is_oos_calibration_relative_vs_constant{suffix}.pdf"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(figA2_both)

    # ---------------- Figure B: per-task OOS relative improvement heatmaps ----------------
    col_labels = [r"$k=1$", r"$k=2$", r"$k=3$", r"$\mathrm{mean\ over\ }k$"]

    def _relative_matrix(method: str) -> np.ndarray:
        L_m = data[method]["val"][:, order]
        L_const = const_oos_ord
        rel = np.full_like(L_m, np.nan, dtype=float)
        mask = np.isfinite(L_m) & np.isfinite(L_const) & (L_const > 0.0)
        rel[mask] = 100.0 * (L_m[mask] - L_const[mask]) / L_const[mask]
        mean_over_k = np.nanmean(rel, axis=0, keepdims=True)
        mat = np.concatenate([rel, mean_over_k], axis=0)  # (K+1, T)
        mat = mat.T  # (T, K+1)
        avg_row = np.nanmean(mat, axis=0, keepdims=True)
        mat = np.vstack([avg_row, mat])  # (T+1, K+1)
        return mat

    mats_B = {
        "binwise": _relative_matrix("binwise"),
        "sigmoid": _relative_matrix("sigmoid"),
        "ispline": _relative_matrix("ispline"),
    }
    all_vals_B = np.concatenate([m.flatten() for m in mats_B.values()])
    finite_B = all_vals_B[np.isfinite(all_vals_B)]
    if finite_B.size:
        vmin_B = float(np.nanmin(finite_B))
        vmax_B = float(np.nanmax(finite_B))
    else:
        vmin_B, vmax_B = 0.0, 1.0
    cmap_B = mpl.cm.Blues

    method_titles_B = {
        "binwise": r"\textbf{Binwise}",
        "sigmoid": r"\textbf{Sigmoid}",
        "ispline": r"\textbf{I-spline}",
    }

    row_labels = ["Avg"] + display_names_ord
    n_tasks = len(row_labels)

    # Adaptive sizing based on number of tasks
    if n_tasks > 10:
        fig_height = max(8.0, min(n_tasks * 0.35, 16.0))
        fig_width = 18.0
        ytick_fontsize = 12
        ylabel_fontsize = 24
        cell_fontsize = 10
    else:
        fig_height = 5.0
        fig_width = 14.0
        ytick_fontsize = 16
        ylabel_fontsize = 24
        cell_fontsize = 11

    figB, axes_B = plt.subplots(1, 3, figsize=(fig_width, fig_height), sharey=True, constrained_layout=True)
    axes_B = np.atleast_1d(axes_B)

    for ax, (m, mat) in zip(axes_B, mats_B.items()):
        im = ax.imshow(mat, aspect="auto", cmap=cmap_B, vmin=vmin_B, vmax=vmax_B)
        ax.set_title(method_titles_B[m], fontsize=24, fontweight="bold", pad=10)
        ax.set_xticks(np.arange(len(col_labels)))
        ax.set_xticklabels(col_labels, rotation=30, ha="right", fontsize=16)
        ax.set_yticks(np.arange(len(row_labels)))
        if ax is axes_B[0]:
            ax.set_yticklabels(row_labels, fontsize=ytick_fontsize)
        else:
            ax.set_yticklabels([])

        # Add gridlines for cell separation
        ax.set_xticks(np.arange(len(col_labels) + 1) - 0.5, minor=True)
        ax.set_yticks(np.arange(len(row_labels) + 1) - 0.5, minor=True)
        ax.grid(which="minor", color="white", linestyle='-', linewidth=1.5)
        ax.tick_params(which="minor", size=0)

        n_rows, n_cols = mat.shape
        for i in range(n_rows):
            for j in range(n_cols):
                val = mat[i, j]
                if not np.isfinite(val):
                    continue
                text = f"{val:.1f}"
                normed = (val - vmin_B) / (vmax_B - vmin_B + 1e-12)
                rgba = cmap_B(normed)
                # Perceptual luminance for readability
                lum = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2]
                color = "black" if lum > 0.5 else "white"
                ax.text(
                    j,
                    i,
                    text,
                    ha="center",
                    va="center",
                    fontsize=cell_fontsize,
                    fontweight="bold",
                    color=color,
                )

    # Y-axis label and tick labels on leftmost heatmap
    axes_B[0].set_ylabel(r"\textbf{Task}", fontsize=ylabel_fontsize, fontweight="bold")
    axes_B[0].set_yticks(np.arange(len(row_labels)))
    axes_B[0].set_yticklabels(row_labels, fontsize=ytick_fontsize)

    # colorbar for figure B
    cbar_B = figB.colorbar(
        mpl.cm.ScalarMappable(norm=mpl.colors.Normalize(vmin=vmin_B, vmax=vmax_B), cmap=cmap_B),
        ax=axes_B.ravel().tolist(),
        location="right",
        fraction=0.046,
        pad=0.04,
    )
    cbar_B.set_label(r"\textbf{\% change in OOD loss vs constant}", fontsize=14, fontweight="bold")
    cbar_B.ax.tick_params(labelsize=11)
    figB.savefig(os.path.join(plots_dir, f"fig_oos_relative_heatmaps{suffix}.png"), dpi=300, bbox_inches="tight")
    figB.savefig(os.path.join(plots_dir, f"fig_oos_relative_heatmaps{suffix}.pdf"), dpi=300, bbox_inches="tight")
    plt.close(figB)

    # ---------------- Figure C: generalization gap heatmaps ----------------
    def _gap_matrix(method: str) -> np.ndarray:
        L_tr = data[method]["train"][:, order]
        L_val = data[method]["val"][:, order]
        gap = (L_val - L_tr) * 1000.0
        mean_over_k = np.nanmean(gap, axis=0, keepdims=True)
        mat = np.concatenate([gap, mean_over_k], axis=0)
        mat = mat.T
        avg_row = np.nanmean(mat, axis=0, keepdims=True)
        mat = np.vstack([avg_row, mat])
        return mat

    mats_C = {
        "binwise": _gap_matrix("binwise"),
        "sigmoid": _gap_matrix("sigmoid"),
        "ispline": _gap_matrix("ispline"),
    }
    all_vals_C = np.concatenate([m.flatten() for m in mats_C.values()])
    finite_C = all_vals_C[np.isfinite(all_vals_C)]
    if finite_C.size:
        vmin_C = float(np.nanmin(finite_C))
        vmax_C = float(np.nanmax(finite_C))
    else:
        vmin_C, vmax_C = 0.0, 1.0
    cmap_C = mpl.cm.Blues

    method_titles_C = {
        "binwise": r"\textbf{Binwise}",
        "sigmoid": r"\textbf{Sigmoid}",
        "ispline": r"\textbf{I-spline}",
    }
    figC, axes_C = plt.subplots(1, 3, figsize=(fig_width, fig_height), sharey=True, constrained_layout=True)
    axes_C = np.atleast_1d(axes_C)
    for ax, (m, mat) in zip(axes_C, mats_C.items()):
        im = ax.imshow(mat, aspect="auto", cmap=cmap_C, vmin=vmin_C, vmax=vmax_C)
        ax.set_title(method_titles_C[m], fontsize=18, fontweight="bold", pad=10)
        ax.set_xticks(np.arange(len(col_labels)))
        ax.set_xticklabels(col_labels, rotation=30, ha="right", fontsize=16)
        ax.set_yticks(np.arange(len(row_labels)))
        ax.set_yticklabels([])  # we set labels once after the loop

        # Add gridlines for cell separation
        ax.set_xticks(np.arange(len(col_labels) + 1) - 0.5, minor=True)
        ax.set_yticks(np.arange(len(row_labels) + 1) - 0.5, minor=True)
        ax.grid(which="minor", color="white", linestyle='-', linewidth=1.5)
        ax.tick_params(which="minor", size=0)

        # annotate all cells with numerical gap values
        n_rows, n_cols = mat.shape
        for i in range(n_rows):
            for j in range(n_cols):
                val = mat[i, j]
                if not np.isfinite(val):
                    continue
                text = f"{val:.1f}"
                # choose text color based on background intensity
                normed = (val - vmin_C) / (vmax_C - vmin_C + 1e-12)
                color = "white" if normed > 0.6 else "black"
                ax.text(
                    j,
                    i,
                    text,
                    ha="center",
                    va="center",
                    fontsize=cell_fontsize,
                    fontweight="bold",
                    color=color,
                )

    # Y-axis label and tick labels on leftmost gap heatmap
    axes_C[0].set_ylabel(r"\textbf{Task}", fontsize=max(14, ylabel_fontsize - 6), fontweight="bold")
    axes_C[0].set_yticks(np.arange(len(row_labels)))
    axes_C[0].set_yticklabels(row_labels, fontsize=ytick_fontsize, fontweight="bold")

    cbar_C = figC.colorbar(
        mpl.cm.ScalarMappable(norm=mpl.colors.Normalize(vmin=vmin_C, vmax=vmax_C), cmap=cmap_C),
        ax=axes_C.ravel().tolist(),
        location="right",
        fraction=0.046,
        pad=0.04,
    )
    cbar_C.set_label(r"\textbf{OOD vs. Train loss $\times 10^3$}", fontsize=14, fontweight="bold")
    cbar_C.ax.tick_params(labelsize=11)

    figC.savefig(os.path.join(plots_dir, f"fig_gap_heatmaps{suffix}.png"), dpi=300, bbox_inches="tight")
    figC.savefig(os.path.join(plots_dir, f"fig_gap_heatmaps{suffix}.pdf"), dpi=300, bbox_inches="tight")
    plt.close(figC)


def plot_figure1_global_bars(base_dir_main: str, base_dir_bbh: str) -> None:
    """Baseline comparison plots for main tasks and BBH subtasks."""
    _setup_rcparams()

    plots_dir_main = os.path.join(base_dir_main, "plots")
    os.makedirs(plots_dir_main, exist_ok=True)

    main_task_set = {
        "BBH Raw",
        "GPQA Raw",
        "IFEval Raw",
        "MATH Lvl 5 Raw",
        "MMLU-PRO Raw",
        "MUSR Raw",
    }

    # Main six tasks
    _plot_for_base(base_dir_main, plots_dir_main, suffix="", allowed_tasks=main_task_set)

    # BBH subtasks (if available)
    if os.path.isdir(os.path.join(base_dir_bbh, "k1")):
        plots_dir_bbh = os.path.join(base_dir_bbh, "plots")
        os.makedirs(plots_dir_bbh, exist_ok=True)
        _plot_for_base(base_dir_bbh, plots_dir_bbh, suffix="", allowed_tasks=None)


def main() -> None:
    ap = argparse.ArgumentParser(description="Plot pinball baseline comparisons (fig1)")
    ap.add_argument(
        "--base_dir_main",
        default=BASE_DIR_MAIN,
        help="Base dir with k*/__pinball_baselines.csv for main tasks.",
    )
    ap.add_argument(
        "--base_dir_bbh",
        default=BASE_DIR_BBH,
        help="Base dir with k*/__pinball_baselines.csv for BBH subtasks.",
    )
    args = ap.parse_args()
    plot_figure1_global_bars(args.base_dir_main, args.base_dir_bbh)


if __name__ == "__main__":
    main()

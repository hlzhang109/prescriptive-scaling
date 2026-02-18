#!/usr/bin/env python3
"""
Highlight-style sweep plot for BBH subtasks under balanced I-optimal design.

Reads OOS coverage error vs alpha from
  outputs/sweeps_iopt_balanced_fullrange/bbh_subtasks/eval_p4_alpha*/k{1,2,3}
and produces a publication-quality figure where:
  - All 24 BBH subtasks are plotted as thin, gray, semi-transparent lines.
  - The mean curve across subtasks is overlaid as a thick, opaque line.
  - The legend is built from proxy Line2D artists so legend markers are solid.
"""

from __future__ import annotations

import os
from typing import Dict, List, Tuple

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, os.pardir))


def _read_task_mae(eval_dir: str) -> Dict[str, Tuple[float, float]]:
    """Read per-task IS/OOS coverage error from __summary.csv files in eval_dir."""
    out: Dict[str, Tuple[float, float]] = {}
    summaries_dir = os.path.join(eval_dir, "summaries")
    scan_dir = summaries_dir if os.path.isdir(summaries_dir) else eval_dir
    for fn in os.listdir(scan_dir):
        if not (fn.endswith("__summary.csv") or fn.endswith("_summary.csv")):
            continue
        path = os.path.join(scan_dir, fn)
        import csv as _csv

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
        a_is = np.asarray(is_accum.get(t, []), float)
        a_oos = np.asarray(oos_accum.get(t, []), float)
        is_mean = float(np.nanmean(a_is)) if a_is.size else float("nan")
        oos_mean = float(np.nanmean(a_oos)) if a_oos.size else float("nan")
        out[t] = (is_mean, oos_mean)
    return out


def _plot_highlight(
    alphas: List[float],
    Y_is: np.ndarray,
    Y_oos: np.ndarray,
    out_base: str,
    title_suffix: str = "",
) -> None:
    """Plot highlight-style IS/OOS curves with average overlay."""
    alpha_arr = np.asarray(alphas, float)
    if Y_is.size == 0 or Y_oos.size == 0:
        return

    mean_is = np.nanmean(Y_is, axis=0)
    mean_oos = np.nanmean(Y_oos, axis=0)

    plt.rcParams["font.family"] = "serif"
    fig, (ax_is, ax_oos) = plt.subplots(1, 2, figsize=(14.0, 5.0), sharey=True)

    # Individual curves: thin, gray, semi-transparent
    for row_is, row_oos in zip(Y_is, Y_oos):
        ax_is.plot(alpha_arr, row_is, color="gray", linewidth=1.2, alpha=0.25)
        ax_oos.plot(alpha_arr, row_oos, color="gray", linewidth=1.2, alpha=0.25)

    # Mean curves: thick, contrasting color
    mean_color = "firebrick"
    ax_is.plot(alpha_arr, mean_is, color=mean_color, linewidth=2.8, alpha=1.0)
    ax_oos.plot(alpha_arr, mean_oos, color=mean_color, linewidth=2.8, alpha=1.0)

    ax_is.set_xlabel(r"Budget $\alpha$ (%)", fontsize=16)
    ax_oos.set_xlabel(r"Budget $\alpha$ (%)", fontsize=16)
    ax_is.set_ylabel("coverage error", fontsize=16)
    ax_is.set_title("In-sample", fontsize=18, fontweight="bold")
    ax_oos.set_title("Out-of-sample", fontsize=18, fontweight="bold")

    for ax in (ax_is, ax_oos):
        ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.5)

    if title_suffix:
        fig.suptitle(title_suffix, fontsize=18, fontweight="bold", y=0.98)

    # Proxy legend handles (solid, non-transparent)
    individual_handle = mlines.Line2D(
        [], [], color="gray", linewidth=1.2, alpha=1.0, label="Individual subtasks"
    )
    avg_handle = mlines.Line2D(
        [], [], color=mean_color, linewidth=2.8, alpha=1.0, label="Average"
    )
    ax_oos.legend(
        handles=[individual_handle, avg_handle],
        loc="upper right",
        frameon=True,
        framealpha=0.9,
        fontsize=12,
    )

    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.97))
    fig.savefig(out_base + ".png", dpi=300, bbox_inches="tight")
    fig.savefig(out_base + ".pdf", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    sweeps_dir = os.path.join(REPO_ROOT, "outputs", "sweeps_iopt_balanced_fullrange", "bbh_subtasks")

    # Determine alpha values from eval_p4_alpha* folders
    alphas: List[float] = []
    for name in os.listdir(sweeps_dir):
        if name.startswith("eval_p4_alpha"):
            try:
                a = float(name.replace("eval_p4_alpha", ""))
                alphas.append(a)
            except ValueError:
                continue
    if not alphas:
        raise SystemExit(f"No eval_p4_alpha* folders found under {sweeps_dir}")
    alphas = sorted(set(alphas))

    # Canonical list of 24 BBH subtasks from the leaderboard CSV header
    import csv as _csv

    header_path = os.path.join(REPO_ROOT, "tables", "open_llm_leaderboard", "open_llm_leaderboard_with_tokens.csv")
    with open(header_path, "r", newline="") as f:
        rd = _csv.reader(f)
        header = next(rd)
    bbh_tasks = [c for c in header if "leaderboard_bbh_" in c]

    # Aggregated over k=1..3: collect IS/OOS curves per task
    curves_is_agg: Dict[str, List[float]] = {t: [] for t in bbh_tasks}
    curves_oos_agg: Dict[str, List[float]] = {t: [] for t in bbh_tasks}
    for a in alphas:
        eval_base = os.path.join(sweeps_dir, f"eval_p4_alpha{int(a)}")
        vals = _read_task_mae_period4(eval_base)
        for t in bbh_tasks:
            is_v, oos_v = vals.get(t, (float("nan"), float("nan")))
            curves_is_agg[t].append(is_v)
            curves_oos_agg[t].append(oos_v)
    Y_is_agg = np.vstack([np.asarray(curves_is_agg[t], float) for t in bbh_tasks])
    Y_oos_agg = np.vstack([np.asarray(curves_oos_agg[t], float) for t in bbh_tasks])
    _plot_highlight(
        alphas,
        Y_is_agg,
        Y_oos_agg,
        os.path.join(sweeps_dir, "period4_singlek_bbh_highlight"),
        title_suffix="Balanced I-optimal design (BBH subtasks, averaged over k)",
    )

    # Per-k highlight plots
    for k in (1, 2, 3):
        curves_is_k: Dict[str, List[float]] = {t: [] for t in bbh_tasks}
        curves_oos_k: Dict[str, List[float]] = {t: [] for t in bbh_tasks}
        for a in alphas:
            eval_dir = os.path.join(sweeps_dir, f"eval_p4_alpha{int(a)}", f"k{k}")
            if not os.path.isdir(eval_dir):
                for t in bbh_tasks:
                    curves_is_k[t].append(float("nan"))
                    curves_oos_k[t].append(float("nan"))
                continue
            vals = _read_task_mae(eval_dir)
            for t in bbh_tasks:
                is_v, oos_v = vals.get(t, (float("nan"), float("nan")))
                curves_is_k[t].append(is_v)
                curves_oos_k[t].append(oos_v)
        Y_is_k = np.vstack([np.asarray(curves_is_k[t], float) for t in bbh_tasks])
        Y_oos_k = np.vstack([np.asarray(curves_oos_k[t], float) for t in bbh_tasks])
        _plot_highlight(
            alphas,
            Y_is_k,
            Y_oos_k,
            os.path.join(sweeps_dir, f"period4_singlek_k{k}_bbh_highlight"),
            title_suffix=f"Balanced I-optimal design (BBH subtasks, $k={k}$)",
        )


if __name__ == "__main__":
    main()

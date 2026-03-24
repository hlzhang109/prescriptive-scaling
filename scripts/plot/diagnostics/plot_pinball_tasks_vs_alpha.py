#!/usr/bin/env python3
"""
Plot per-task sigmoid pinball loss vs alpha for period4 single-k sweeps.

Inputs (per base_dir):
  eval_p4_alphaXX/k{1,2,3}/<task>__pinball_sigmoid.csv (train/val rows)

Outputs:
  <base_dir>/plots_pinball_tasks/
    pinball_tasks_k1.{png,pdf}
    pinball_tasks_k2.{png,pdf}
    pinball_tasks_k3.{png,pdf}
    pinball_tasks_avg.{png,pdf}   (per-task averaged over k)
"""
from __future__ import annotations

import argparse
import os
import re
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import matplotlib as mpl  # type: ignore
import matplotlib.pyplot as plt  # type: ignore


def _find_alpha_dirs(base: Path) -> List[Tuple[int, Path]]:
    alphas: List[Tuple[int, Path]] = []
    for p in base.glob("eval_p4_alpha*"):
        m = re.match(r"eval_p4_alpha(\d+)", p.name)
        if m:
            alphas.append((int(m.group(1)), p))
    return sorted(alphas, key=lambda x: x[0])


def _discover_tasks(alpha_dir: Path, k: int) -> List[str]:
    kdir = alpha_dir / f"k{k}"
    tasks = []
    if not kdir.is_dir():
        return tasks
    for fn in kdir.glob("*__pinball_sigmoid.csv"):
        name = fn.name.replace("__pinball_sigmoid.csv", "")
        tasks.append(name)
    return sorted(tasks)


def _load_task_series(
    alpha_dirs: List[Tuple[int, Path]], tasks: List[str]
) -> Tuple[Dict[int, Dict[str, List[float]]], Dict[int, Dict[str, List[float]]]]:
    """Return per-k IS/OOS series: {k: {task: [vals per alpha]}}."""
    series_is: Dict[int, Dict[str, List[float]]] = {1: {}, 2: {}, 3: {}}
    series_oos: Dict[int, Dict[str, List[float]]] = {1: {}, 2: {}, 3: {}}
    for alpha, p in alpha_dirs:
        for k in (1, 2, 3):
            kdir = p / f"k{k}"
            for task in tasks:
                path = kdir / f"{task}__pinball_sigmoid.csv"
                L_tr = float("nan")
                L_val = float("nan")
                if path.is_file():
                    try:
                        import pandas as pd  # type: ignore

                        df = pd.read_csv(path)
                        tr_row = df[df["split"] == "train"]
                        val_row = df[df["split"] == "val"]
                        if not tr_row.empty:
                            L_tr = float(tr_row.iloc[0]["L_sigmoid"])
                        if not val_row.empty:
                            L_val = float(val_row.iloc[0]["L_sigmoid"])
                    except Exception:
                        pass
                series_is[k].setdefault(task, []).append(L_tr)
                series_oos[k].setdefault(task, []).append(L_val)
    return series_is, series_oos


def _plot_task_curves(
    alphas: List[int],
    series_is: Dict[str, List[float]],
    series_oos: Dict[str, List[float]],
    title: str,
    out_path: Path,
) -> None:
    try:
        mpl.rcParams["font.family"] = "serif"
        mpl.rcParams["text.usetex"] = True
        mpl.rcParams["text.latex.preamble"] = r"\usepackage{amsmath}"
    except Exception:
        pass

    target_alphas = [0, 5, 10, 20, 50, 100]
    pos = np.arange(len(target_alphas))

    fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=True)
    ax_is, ax_oos = axes

    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]
    markers = ["o", "s", "^", "D", "v", "p"]
    tasks = sorted(series_oos.keys())

    for idx, task in enumerate(tasks):
        alpha_to_is = {a: series_is.get(task, [np.nan]*len(alphas))[i] for i, a in enumerate(alphas)}
        alpha_to_oos = {a: series_oos.get(task, [np.nan]*len(alphas))[i] for i, a in enumerate(alphas)}
        ys_is = [alpha_to_is.get(a, np.nan) for a in target_alphas]
        ys_oos = [alpha_to_oos.get(a, np.nan) for a in target_alphas]
        label = task.replace(" Raw", "")
        color = colors[idx % len(colors)]
        marker = markers[idx % len(markers)]
        ax_is.plot(
            pos,
            ys_is,
            marker=marker,
            color=color,
            linewidth=2.2,
            markersize=7,
            markeredgewidth=1.4,
            markeredgecolor="white",
            label=label,
            zorder=3,
        )
        ax_oos.plot(
            pos,
            ys_oos,
            marker=marker,
            color=color,
            linewidth=2.2,
            markersize=7,
            markeredgewidth=1.4,
            markeredgecolor="white",
            label=label,
            zorder=3,
        )

    for ax, lbl in ((ax_is, "In-sample pinball loss"), (ax_oos, "Out-of-sample pinball loss")):
        ax.set_xlabel(r"$\alpha$", fontsize=14)
        ax.set_ylabel(lbl, fontsize=14)
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=12)
        ax.set_xticks(pos)
        ax.set_xticklabels(target_alphas)
        ax.set_xlim(-0.5, len(target_alphas) - 0.5)
    handles, labels = ax_oos.get_legend_handles_labels()
    ax_oos.legend(handles, labels, ncol=2, fontsize=12, frameon=False)
    fig.suptitle(title, fontsize=16)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path.with_suffix(".png"), dpi=300)
    plt.savefig(out_path.with_suffix(".pdf"), dpi=300)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description="Plot per-task pinball loss vs alpha (period4 single-k).")
    ap.add_argument("--base_dir", required=True, help="Sweep base directory (e.g., outputs/sweeps_iopt_balanced_fullrange)")
    ap.add_argument("--out_dir", default=None, help="Output directory for plots (default: <base_dir>/plots_pinball_tasks)")
    args = ap.parse_args()

    base = Path(args.base_dir)
    out_dir = Path(args.out_dir) if args.out_dir else base / "plots_pinball_tasks"

    alpha_dirs = _find_alpha_dirs(base)
    if not alpha_dirs:
        raise SystemExit(f"No eval_p4_alpha* directories found under {base}")

    tasks: List[str] = []
    for _, p in alpha_dirs:
        tasks = _discover_tasks(p, k=1)
        if tasks:
            break
    if not tasks:
        raise SystemExit("Could not discover tasks (no pinball pinball_sigmoid files found).")

    alphas = [a for a, _ in alpha_dirs]
    series_is_k, series_oos_k = _load_task_series(alpha_dirs, tasks)

    # Per-k plots
    for k in (1, 2, 3):
        _plot_task_curves(
            alphas,
            series_is_k[k],
            series_oos_k[k],
            title=fr"Pinball loss vs. $\alpha$ ($k={k}$)",
            out_path=out_dir / f"pinball_tasks_k{k}",
        )

    # Average over k per task at each alpha
    series_is_avg: Dict[str, List[float]] = {t: [] for t in tasks}
    series_oos_avg: Dict[str, List[float]] = {t: [] for t in tasks}
    for idx in range(len(alphas)):
        for t in tasks:
            vals_is = [series_is_k[k].get(t, [np.nan] * len(alphas))[idx] for k in (1, 2, 3)]
            vals_oos = [series_oos_k[k].get(t, [np.nan] * len(alphas))[idx] for k in (1, 2, 3)]
            series_is_avg[t].append(float(np.nanmean(vals_is)) if np.any(np.isfinite(vals_is)) else np.nan)
            series_oos_avg[t].append(float(np.nanmean(vals_oos)) if np.any(np.isfinite(vals_oos)) else np.nan)

    _plot_task_curves(
        alphas,
        series_is_avg,
        series_oos_avg,
        title=r"Pinball loss vs. $\alpha$ (avg over $k$)",
        out_path=out_dir / "pinball_tasks_avg",
    )


if __name__ == "__main__":
    main()

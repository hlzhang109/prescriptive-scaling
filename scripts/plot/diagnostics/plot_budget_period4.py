#!/usr/bin/env python3
"""
Generate per-task budget-frontier plots for period4 designs.

This script does not recompute the budget design. It reads the existing
`manifest__train.txt` / `manifest__val.txt` files produced by:
  - scripts/run/budget_only.py period4 --train_mode <cumulative|single_k>

and then, for each k in {1,2,3}:
  - Extracts (compute, accuracy) points for each task for the selected models
  - Fits a high-quantile sigmoid frontier on the selected points
  - Writes curve CSVs and PNG/PDF plots under:
      outputs/budget/period4/<train_mode>/{curves,plots}/k{k}/
"""

from __future__ import annotations

import argparse
import csv as _csv
import os
import sys
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np

# When executed as `python scripts/plot/diagnostics/plot_budget_period4.py`, Python's import root
# becomes `scripts/plot/diagnostics/`, so we must add the repo root (three levels up)
# to import `skill_frontier.*` and `scripts.*` consistently.
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, os.pardir))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from skill_frontier.io.csv_utils import (  # type: ignore
    compute_flops,
    detect_oll_raw_tasks,
    maybe_scale_task_values,
    extract_model_id,
    read_csv_rows,
    sanitize_name,
)
from skill_frontier.io.manifest_utils import read_manifest  # type: ignore
from skill_frontier.io.output_paths import get_budget_output_paths  # type: ignore
from skill_frontier.plotting.model_families import (  # type: ignore
    FAMILY_ORDER,
    color_for_family,
    extract_base_model_name,
    family_from_base_model,
)
from skill_frontier.plotting.axis_formatting import (  # type: ignore
    apply_pretraining_compute_tick_multiplier,
)
from skill_frontier.plotting.configs import frontier_1d as frontier_1d_cfg  # type: ignore
from skill_frontier.plotting.configs import mpl_rc as mpl_rc_cfg  # type: ignore
from skill_frontier.plotting.labels import PRETRAINING_COMPUTE_FLOPS_LABEL  # type: ignore

try:
    from skill_frontier.core.sigmoid import fit_sigmoid_frontier  # type: ignore
except Exception:  # pragma: no cover
    from scripts.smooth_single_skill_frontier import fit_sigmoid_frontier  # type: ignore


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Plot period4 budget selections (no re-design).")
    default_csv = os.path.join(
        REPO_ROOT,
        "tables",
        "open_llm_leaderboard",
        "open_llm_leaderboard_with_tokens.csv",
    )
    p.add_argument("--csv", default=default_csv, help="Input CSV (default: OLL with tokens).")
    p.add_argument(
        "--compute_product_cols",
        nargs=2,
        default=["Pretraining tokens (T)", "#Params (B)"],
        help="Two columns to multiply for compute (FLOPs proxy).",
    )
    p.add_argument(
        "--compute_multiplier",
        type=float,
        default=6.0,
        help="Multiplier for product compute (default: 6.0).",
    )
    p.add_argument(
        "--tau",
        type=float,
        default=0.98,
        help="Quantile tau for the fitted sigmoid frontier on selected points.",
    )
    p.add_argument(
        "--out_base",
        default=os.path.join(REPO_ROOT, "outputs"),
        help="Outputs base directory (default: repo_root/outputs).",
    )
    p.add_argument(
        "--train_mode",
        choices=["cumulative", "single_k"],
        default="single_k",
        help="Which period4 budget design variant to plot.",
    )
    p.add_argument(
        "--tasks",
        nargs="*",
        default=None,
        help="Task columns to plot (default: auto-detect OLL Raw tasks).",
    )
    p.add_argument(
        "--splits",
        nargs="*",
        choices=["train", "val"],
        default=["train", "val"],
        help="Which manifests to plot per k (default: train and val).",
    )
    p.add_argument(
        "--scatter_style",
        choices=["two_color", "family"],
        default="two_color",
        help="Scatter styling: 'two_color' (default) disables model-family coloring; 'family' colors points by base-model family.",
    )
    return p


def _iter_selected_points(
    rows: List[dict],
    headers: List[str],
    selected: set[str],
    task: str,
    compute_product_cols: Tuple[str, str],
    compute_multiplier: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    xs: List[float] = []
    ys: List[float] = []
    base_models: List[str] = []
    for row in rows:
        mid = extract_model_id(row)
        if not mid or mid not in selected:
            continue
        c = compute_flops(
            row,
            headers,
            logC_col=None,
            prod_cols=compute_product_cols,
            mult=float(compute_multiplier),
        )
        if not (np.isfinite(c) and c > 0.0):
            continue
        v = row.get(task, None)
        try:
            y = float(v) if v not in (None, "", "nan", "NaN") else float("nan")
        except Exception:
            y = float("nan")
        if not np.isfinite(y):
            continue
        xs.append(float(c))
        ys.append(float(y))
        base_models.append(extract_base_model_name(row))
    y_arr = maybe_scale_task_values(np.asarray(ys, float))
    return np.asarray(xs, float), y_arr, np.asarray(base_models, dtype=object)


def _write_curve_csv(path: str, xs: np.ndarray, ys: np.ndarray, task: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="") as f:
        w = _csv.writer(f)
        w.writerow(["x", "y_hat", "task"])
        for xi, yi in zip(xs, ys):
            w.writerow([float(xi), float(yi), task])


def _plot_one(
    out_base: str,
    plots_dir: str,
    curves_dir: str,
    k_idx: int,
    split: str,
    task: str,
    X: np.ndarray,
    Y: np.ndarray,
    base_models: np.ndarray,
    xs_curve: np.ndarray,
    y_curve: np.ndarray,
    tau: float,
    scatter_style: str,
) -> None:
    try:
        import matplotlib.pyplot as plt  # type: ignore
        import matplotlib as mpl  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError("matplotlib is required for plotting") from e

    try:
        mpl.rcParams["font.family"] = mpl_rc_cfg.FONT_FAMILY
    except Exception:
        pass

    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(curves_dir, exist_ok=True)

    safe_task = sanitize_name(task)
    suffix = f"__{split}"
    curve_csv = os.path.join(curves_dir, f"budget_frontier__{safe_task}{suffix}.csv")
    _write_curve_csv(curve_csv, xs_curve, y_curve, task)

    fig, ax = plt.subplots(figsize=frontier_1d_cfg.FIGSIZE)
    if str(scatter_style) == "family" and base_models is not None and base_models.size == X.size:
        fams = np.asarray([family_from_base_model(b) for b in base_models], dtype=object)
        for fam in FAMILY_ORDER:
            m = fams == fam
            if not np.any(m):
                continue
            ax.scatter(
                X[m],
                Y[m],
                s=frontier_1d_cfg.SCATTER_SIZE,
                alpha=frontier_1d_cfg.SCATTER_ALPHA,
                color=color_for_family(str(fam)),
                linewidths=frontier_1d_cfg.SCATTER_LINEWIDTHS,
                rasterized=True,
                label="_nolegend_",
            )
    else:
        ax.scatter(
            X,
            Y,
            s=frontier_1d_cfg.SCATTER_SIZE,
            alpha=frontier_1d_cfg.SCATTER_ALPHA,
            color="#1f77b4",
            label="_nolegend_",
            linewidths=frontier_1d_cfg.SCATTER_LINEWIDTHS,
        )
    ax.plot(
        xs_curve,
        y_curve,
        color="#1f77b4",
        linewidth=frontier_1d_cfg.CURVE_LINEWIDTH,
        label=rf"Sigmoid $\tau={tau:.2f}$",
    )
    ax.set_xscale("log")
    y_min = float(np.nanmin([np.nanmin(Y), np.nanmin(y_curve)]))
    y_max = float(np.nanmax([np.nanmax(Y), np.nanmax(y_curve)]))
    if not np.isfinite(y_min) or not np.isfinite(y_max) or y_min == y_max:
        y_min, y_max = 0.0, 1.0
    pad = 0.02 * max(1e-6, (y_max - y_min))
    ax.set_ylim(y_min - pad, y_max + pad)
    ax.set_xlabel(PRETRAINING_COMPUTE_FLOPS_LABEL, fontweight="bold", fontsize=frontier_1d_cfg.LABEL_FONTSIZE_X)
    apply_pretraining_compute_tick_multiplier(ax)
    ax.set_ylabel("Accuracy", fontweight="bold", fontsize=frontier_1d_cfg.LABEL_FONTSIZE_Y)
    ax.set_title(
        f"Budget Frontier — k={k_idx} ({split}) — {task}",
        fontweight="bold",
        fontsize=frontier_1d_cfg.TITLE_FONTSIZE,
    )
    from matplotlib.lines import Line2D  # type: ignore

    point_handle = Line2D(
        [0],
        [0],
        marker="o",
        linestyle="None",
        color="#333333",
        markersize=7,
        label="selected points",
    )
    line_handles, line_labels = ax.get_legend_handles_labels()
    ax.legend(
        [point_handle] + line_handles,
        [point_handle.get_label()] + line_labels,
        loc=frontier_1d_cfg.LEGEND_LOC,
        fontsize=frontier_1d_cfg.LEGEND_FONTSIZE,
    )
    fig.tight_layout()

    base = os.path.join(plots_dir, f"budget_frontier__{safe_task}{suffix}")
    fig.savefig(base + ".png", dpi=300, bbox_inches="tight")
    fig.savefig(base + ".pdf", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = _build_argparser().parse_args(argv)

    rows, headers = read_csv_rows(args.csv)
    if not rows:
        raise SystemExit("No rows in CSV")

    tasks = list(args.tasks) if args.tasks else detect_oll_raw_tasks(headers)
    if not tasks:
        raise SystemExit("No tasks provided and none auto-detected")

    base_paths = get_budget_output_paths(
        args.out_base,
        mode="period4",
        train_mode=args.train_mode,
        legacy=False,
    )
    manifests_base = base_paths["manifests"]
    plots_base = base_paths["plots"]
    curves_base = base_paths["curves"]

    for k_idx in (1, 2, 3):
        for split in args.splits:
            manifest_path = os.path.join(
                manifests_base, f"k{k_idx}", f"manifest__{split}.txt"
            )
            if not os.path.isfile(manifest_path):
                print(f"[plot budget] skip k{k_idx} {split}: missing {manifest_path}")
                continue
            selected = read_manifest(manifest_path)
            if not selected:
                print(f"[plot budget] skip k{k_idx} {split}: empty manifest")
                continue

            plots_dir = os.path.join(plots_base, f"k{k_idx}")
            curves_dir = os.path.join(curves_base, f"k{k_idx}")

            for task in tasks:
                X, Y, base_models = _iter_selected_points(
                    rows,
                    headers,
                    selected=set(selected),
                    task=task,
                    compute_product_cols=tuple(args.compute_product_cols),
                    compute_multiplier=float(args.compute_multiplier),
                )
                if X.size < 3:
                    print(f"[plot budget] skip k{k_idx} {split} {task}: only {X.size} points")
                    continue

                xs_curve, y_curve = fit_sigmoid_frontier(
                    X, Y, tau=float(args.tau), use_log10_x=True
                )
                _plot_one(
                    out_base=args.out_base,
                    plots_dir=plots_dir,
                    curves_dir=curves_dir,
                    k_idx=k_idx,
                    split=split,
                    task=task,
                    X=X,
                    Y=Y,
                    base_models=base_models,
                    xs_curve=xs_curve,
                    y_curve=y_curve,
                    tau=float(args.tau),
                    scatter_style=str(args.scatter_style),
                )


if __name__ == "__main__":
    main()

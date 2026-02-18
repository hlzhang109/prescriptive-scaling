#!/usr/bin/env python3
"""
Overlay sigmoid frontiers with top-50%-by-Hub vs rest scatter.

This script:
  - Reads a FULL Open LLM Leaderboard CSV and a filtered TOP CSV.
  - Fits a sigmoid frontier on ALL data for each task.
  - Plots all models as scatter, with distinct colors for:
      * Top models (present in TOP CSV),
      * Remaining models.
  - Overlays the sigmoid frontier fitted on all data.

The resulting figures are written to --out_dir (typically
outputs_top_models/sigmoid_no_split).
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import List, Tuple

import numpy as np

# Ensure repo root is on path for local imports
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, os.pardir))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from skill_frontier.io.csv_utils import (  # type: ignore
    read_csv_rows,
    compute_flops,
    detect_oll_raw_tasks,
    extract_model_id,
)
from skill_frontier.io.csv_utils import collect_model_ids  # type: ignore
from skill_frontier.plotting.plot_utils import ensure_dir, save_figure  # type: ignore
from skill_frontier.io.output_paths import sanitize_task_name  # type: ignore
from skill_frontier.core.fit_imports import import_fit_sigmoid_frontier_basic  # type: ignore
from skill_frontier.plotting.axis_formatting import (  # type: ignore
    apply_pretraining_compute_tick_multiplier,
)


def _load_top_ids(top_csv: str) -> set:
    return collect_model_ids(top_csv)


def _fit_and_plot(
    full_csv: str,
    top_csv: str,
    compute_product_cols: Tuple[str, str],
    compute_multiplier: float,
    tasks: List[str],
    tau_all: float,
    tau_top: float,
    out_dir: str,
) -> None:
    rows, headers = read_csv_rows(full_csv)
    if not rows:
        raise SystemExit(f"No rows in {full_csv}")
    top_ids = _load_top_ids(top_csv)
    fit_sigmoid_frontier = import_fit_sigmoid_frontier_basic()

    try:
        import matplotlib.pyplot as plt  # type: ignore
        import matplotlib as mpl  # type: ignore
    except Exception as e:  # pragma: no cover
        raise SystemExit("matplotlib is required for plotting") from e

    try:
        mpl.rcParams["font.family"] = "serif"
    except Exception:
        pass

    ensure_dir(out_dir)
    plots_dir = os.path.join(out_dir, "plots")
    ensure_dir(plots_dir)

    def _row_compute(row: dict) -> float:
        return compute_flops(
            row,
            headers,
            logC_col=None,
            prod_cols=compute_product_cols,
            mult=float(compute_multiplier),
        )

    for task in tasks:
        x_all: List[float] = []
        y_all: List[float] = []
        mids: List[str] = []

        for r in rows:
            mid = extract_model_id(r)
            C = _row_compute(r)
            if not (np.isfinite(C) and C > 0.0):
                continue
            v = r.get(task, None)
            try:
                y = float(v) if v not in (None, "", "nan", "NaN") else float("nan")
            except Exception:
                y = float("nan")
            if not np.isfinite(y):
                continue
            x_all.append(float(C))
            y_all.append(float(y))
            mids.append(mid)

        if len(x_all) < 3:
            print(f"[top_hub] skip {task}: only {len(x_all)} points")
            continue

        x_arr = np.asarray(x_all, float)
        y_arr = np.asarray(y_all, float)

        # Fit sigmoid frontier on ALL points (always using tau_all)
        xs_curve_all, y_curve_all = fit_sigmoid_frontier(
            x_arr, y_arr, tau=float(tau_all), use_log10_x=True
        )
        if xs_curve_all.size == 0:
            print(f"[top_hub] skip {task}: fit returned empty curve")
            continue

        # Build top vs rest masks
        mids_arr = np.asarray(mids, dtype=object)
        top_mask = np.isin(mids_arr, np.array(list(top_ids), dtype=object))

        # Fit sigmoid frontier on TOP models only (if enough points)
        xs_curve_top: np.ndarray
        y_curve_top: np.ndarray
        xs_curve_top = np.array([], dtype=float)
        y_curve_top = np.array([], dtype=float)
        if np.sum(top_mask) >= 3:
            xs_curve_top, y_curve_top = fit_sigmoid_frontier(
                x_arr[top_mask], y_arr[top_mask], tau=float(tau_top), use_log10_x=True
            )

        # Plot
        plt.figure(figsize=(7.0, 4.6))
        # Other models: light gray
        if np.any(~top_mask):
            plt.scatter(
                x_arr[~top_mask],
                y_arr[~top_mask],
                s=10,
                alpha=0.20,
                color="#c0c0c0",
                label="Other models",
            )
        # Top models: stronger color
        if np.any(top_mask):
            plt.scatter(
                x_arr[top_mask],
                y_arr[top_mask],
                s=16,
                alpha=0.35,
                color="#1f77b4",
                label="Top 50% by Hub ❤️",
            )

        # Sigmoid frontier (all data)
        plt.plot(
            xs_curve_all,
            y_curve_all,
            color="#000000",
            linewidth=2.4,
            label=f"Sigmoid frontier (all data, τ={float(tau_all):.2f})",
        )

        # Sigmoid frontier (top models only), if available
        if xs_curve_top.size:
            plt.plot(
                xs_curve_top,
                y_curve_top,
                color="#ff7f0e",
                linewidth=2.0,
                linestyle="--",
                label=f"Sigmoid frontier (top 50% only, τ={float(tau_top):.2f})",
            )

        plt.xscale("log")
        y_min = float(
            np.nanmin(
                [
                    np.nanmin(y_arr) if y_arr.size else np.nan,
                    np.nanmin(y_curve_all) if y_curve_all.size else np.nan,
                ]
            )
        )
        y_max = float(
            np.nanmax(
                [
                    np.nanmax(y_arr) if y_arr.size else np.nan,
                    np.nanmax(y_curve_all) if y_curve_all.size else np.nan,
                ]
            )
        )
        if not np.isfinite(y_min) or not np.isfinite(y_max) or y_min == y_max:
            y_min, y_max = 0.0, 1.0
        pad = 0.02 * max(1e-6, (y_max - y_min))
        plt.ylim(y_min - pad, y_max + pad)
        plt.xlabel("Pretraining Compute (FLOPs)", fontweight="bold", fontsize=15)
        apply_pretraining_compute_tick_multiplier(plt.gca())
        plt.ylabel("Accuracy", fontweight="bold", fontsize=15)
        plt.title(f"Sigmoid Frontier (Top vs Rest) — {task}", fontweight="bold", fontsize=18)
        leg = plt.legend(loc="best", fontsize=11)
        if leg and leg.get_title():
            leg.get_title().set_fontweight("bold")
        plt.tight_layout()

        save_figure(
            plt.gcf(),
            plots_dir,
            sanitize_task_name(task),
            dpi=300,
            bbox_inches="tight",
            png_first=True,
        )
        plt.close()


def main(argv: List[str] | None = None) -> None:
    ap = argparse.ArgumentParser(
        description="Overlay sigmoid frontier with top-50%-by-Hub vs rest scatter"
    )
    ap.add_argument("--full_csv", required=True, help="Full OLL CSV path")
    ap.add_argument("--top_csv", required=True, help="Filtered top-Hub CSV path")
    ap.add_argument(
        "--compute_product_cols",
        nargs=2,
        required=True,
        help="Two columns whose product (times multiplier) is compute",
    )
    ap.add_argument(
        "--compute_multiplier",
        type=float,
        default=6.0,
        help="Multiplier for product compute (default 6.0)",
    )
    ap.add_argument(
        "--tasks",
        nargs="*",
        default=None,
        help="Task columns to plot; if omitted, detect OLL Raw tasks",
    )
    ap.add_argument(
        "--tau_all",
        type=float,
        default=0.98,
        help="Quantile parameter for the all-data sigmoid frontier (default: 0.98)",
    )
    ap.add_argument(
        "--tau_top",
        type=float,
        default=0.98,
        help="Quantile parameter for the top-model-only sigmoid frontier (default: 0.98)",
    )
    ap.add_argument(
        "--out_dir",
        default=os.path.join("outputs_top_models", "sigmoid", "no_split"),
        help="Output directory for plots (default: outputs_top_models/sigmoid/no_split)",
    )
    args = ap.parse_args(argv)

    rows, headers = read_csv_rows(args.full_csv)
    if not rows:
        raise SystemExit(f"No rows in {args.full_csv}")
    tasks = args.tasks or detect_oll_raw_tasks(headers)
    if not tasks:
        raise SystemExit("No tasks provided and none auto-detected from headers")

    _fit_and_plot(
        full_csv=args.full_csv,
        top_csv=args.top_csv,
        compute_product_cols=(args.compute_product_cols[0], args.compute_product_cols[1]),
        compute_multiplier=float(args.compute_multiplier),
        tasks=tasks,
        tau_all=float(args.tau_all),
        tau_top=float(args.tau_top),
        out_dir=args.out_dir,
    )


if __name__ == "__main__":
    main()

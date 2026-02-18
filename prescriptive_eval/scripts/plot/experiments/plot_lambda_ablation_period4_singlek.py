#!/usr/bin/env python3
"""
Plot coverage error vs lambda for period-4 single-k, no-budget sigmoid frontiers.

This script expects evaluation outputs produced by
`scripts/run/ablate_lambda_period4_singlek.py`, which creates directories:

    outputs/lambda_ablation/period4_singlek_no_budget/lambda_<tag>/k{1,2,3}/

Each k-directory must contain `summary_over_tasks.csv` with metrics:
  - IS_MAE_macro_mean
  - OOS_MAE_macro_mean

The script aggregates these and generates a 1x3 subplot figure showing
in-sample and out-of-sample coverage error as a function of lambda for
k=1,2,3 respectively.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np


REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, os.pardir)
)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Plot coverage error vs lambda for period4 single-k (no budget)."
    )
    default_root = os.path.join(
        REPO_ROOT, "outputs", "lambda_ablation", "period4_singlek_no_budget"
    )
    p.add_argument(
        "--root",
        default=default_root,
        help="Root directory containing lambda_<tag>/ subdirectories.",
    )
    p.add_argument(
        "--lambdas",
        type=float,
        nargs="*",
        default=None,
        help=(
            "Optional explicit list of lambdas to plot. If omitted, "
            "all lambda_* subdirectories under --root are used."
        ),
    )
    return p


def _discover_lambda_dirs(root: str) -> List[Tuple[float, str]]:
    """Return sorted list of (lambda_value, path) discovered under root."""
    if not os.path.isdir(root):
        raise SystemExit(f"Lambda root directory not found: {root}")

    items: List[Tuple[float, str]] = []
    for name in os.listdir(root):
        if not name.startswith("lambda_"):
            continue
        tag = name[len("lambda_") :]
        try:
            lam_val = float(tag)
        except ValueError:
            continue
        items.append((lam_val, os.path.join(root, name)))
    if not items:
        raise SystemExit(f"No lambda_* subdirectories found under {root}")
    items.sort(key=lambda t: t[0])
    return items


def _metrics_from_summary(path: str) -> Dict[str, float]:
    """Load metric -> value mapping from summary_over_tasks.csv."""
    metrics: Dict[str, float] = {}
    with open(path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            m = row.get("metric", "")
            v = row.get("value", "")
            if not m:
                continue
            try:
                metrics[m] = float(v)
            except Exception:
                continue
    return metrics


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)

    # Discover lambda directories
    discovered = _discover_lambda_dirs(args.root)
    if args.lambdas is not None and len(args.lambdas) > 0:
        # Filter discovered list to the requested lambdas (by numeric value)
        requested = set(float(l) for l in args.lambdas)
        discovered = [t for t in discovered if t[0] in requested]
        if not discovered:
            raise SystemExit("No matching lambda_* directories for requested lambdas.")

    lambdas, lambda_paths = zip(*discovered)
    lambdas = list(lambdas)
    num_lam = len(lambdas)
    num_k = 3

    mae_is = np.full((num_k, num_lam), np.nan, dtype=float)
    mae_oos = np.full((num_k, num_lam), np.nan, dtype=float)

    for j, (lam, lam_path) in enumerate(discovered):
        for k_idx in range(num_k):
            k = k_idx + 1
            summary_candidates = [
                os.path.join(lam_path, f"k{k}", "aggregate", "summary_over_tasks.csv"),
                os.path.join(lam_path, f"k{k}", "summary_over_tasks.csv"),
            ]
            summary_path = next((p for p in summary_candidates if os.path.isfile(p)), None)
            if not summary_path:
                continue
            metrics = _metrics_from_summary(summary_path)
            mae_is[k_idx, j] = metrics.get("IS_MAE_macro_mean", float("nan"))
            mae_oos[k_idx, j] = metrics.get("OOS_MAE_macro_mean", float("nan"))

    # Plot configuration (match general paper style, but slightly more polished)
    plt.rc("text", usetex=True)
    plt.rc("font", family="serif")

    fig, axes = plt.subplots(1, num_k, figsize=(16, 4.5), sharey=True)

    lambdas_arr = np.array(lambdas, float)
    # For log-spacing on the x-axis we only plot strictly positive lambdas.
    mask_pos = lambdas_arr > 0.0
    lambdas_pos = lambdas_arr[mask_pos]

    # Global y-limits across all k (so sharey=True is consistent)
    is_all = mae_is[:, mask_pos]
    oos_all = mae_oos[:, mask_pos]
    finite_all = np.concatenate(
        [is_all[np.isfinite(is_all)], oos_all[np.isfinite(oos_all)]]
    )
    if finite_all.size:
        y_lo = float(np.min(finite_all))
        y_hi = float(np.max(finite_all))
        pad = 0.05 * (y_hi - y_lo if y_hi > y_lo else 1.0)
        y_min = y_lo - pad
        y_max = y_hi + pad
    else:
        y_min, y_max = 0.0, 1.0

    is_color = "firebrick"
    oos_color = "0.35"  # dark gray

    for k_idx, ax in enumerate(axes):
        k = k_idx + 1
        is_vals = mae_is[k_idx, mask_pos]
        oos_vals = mae_oos[k_idx, mask_pos]
        ax.set_ylim(y_min, y_max)
        ax.plot(
            lambdas_pos,
            is_vals,
            marker="o",
            markersize=8,
            linewidth=2.0,
            color=is_color,
            markerfacecolor="white",
            markeredgecolor=is_color,
            label="In-sample pinball loss",
        )
        ax.plot(
            lambdas_pos,
            oos_vals,
            marker="D",
            markersize=7,
            linewidth=2.0,
            color=oos_color,
            markerfacecolor="white",
            markeredgecolor=oos_color,
            label="Out-of-sample pinball loss",
        )
        ax.set_xlabel(
            r"\textbf{$\lambda$ (L2 penalty)}", fontsize=20
        )
        if k_idx == 0:
            ax.set_ylabel(
                r"\textbf{Average pinball loss}", fontsize=20
            )
        ax.grid(True, linestyle=":", alpha=0.4)
        ax.tick_params(axis="both", labelsize=14)
        # Log scale on lambda so spacing is uniform in log10(lambda)
        ax.set_xscale("log")
        xticks = lambdas_pos
        ax.set_xticks(xticks)
        ax.set_xticklabels(
            [
                r"$0$"
                if val == 0.0  # kept for completeness; mask_pos ensures val>0 here
                else rf"$10^{{{int(np.log10(val))}}}$"
                for val in xticks
            ],
            rotation=0,
        )
        # Place k label inside subplot in a small textbox (top-left)
        bbox_props = dict(
            boxstyle="round,pad=0.2",
            facecolor="white",
            edgecolor="0.5",
            alpha=0.7,
            linewidth=0.8,
        )
        ax.text(
            0.03,
            0.95,
            rf"$k={k}$",
            transform=ax.transAxes,
            fontsize=18,
            fontweight="bold",
            va="top",
            ha="left",
            bbox=bbox_props,
        )

    # Shared legend
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=2,
        fontsize=18,
        markerscale=1.4,
        frameon=False,
        bbox_to_anchor=(0.5, -0.02),
    )
    fig.tight_layout(rect=(0.0, 0.08, 1.0, 1.0))

    out_plots_dir = os.path.join(args.root, "plots")
    os.makedirs(out_plots_dir, exist_ok=True)
    out_png = os.path.join(out_plots_dir, "period4_singlek_mae_vs_lambda.png")
    out_pdf = os.path.join(out_plots_dir, "period4_singlek_mae_vs_lambda.pdf")
    fig.savefig(out_png, dpi=300)
    fig.savefig(out_pdf)
    print(f"[lambda_ablation] Saved plots to {out_png} and {out_pdf}")


if __name__ == "__main__":
    main()

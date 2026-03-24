#!/usr/bin/env python3
"""
Regenerate the 6 overlay figures for the pretrain-vs-posttrain analysis to match
the visual style of the first two subplots in:
  outputs/figures_main_paper/figure4_refined.pdf

Overwrites (in place):
  - Compute-x overlays:
      outputs/sigmoid/no_split/pretrain_vs_posttrain/plots/overlay*.pdf
      outputs/sigmoid/no_split/pretrain_vs_posttrain/plots/overlay*.png
  - Size-x overlays:
      outputs/sigmoid_size/no_split/pretrain_vs_posttrain/plots/overlay*.pdf
      outputs/sigmoid_size/no_split/pretrain_vs_posttrain/plots/overlay*.png

Usage:
  python scripts/plot/overlays/regenerate_overlays_pretrain_vs_posttrain.py
  python scripts/plot/overlays/regenerate_overlays_pretrain_vs_posttrain.py --x-axis size
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Optional, Sequence, Tuple

import numpy as np

try:
    import pandas as pd  # type: ignore
except Exception as e:  # pragma: no cover
    raise SystemExit("This script requires pandas.") from e

try:
    import matplotlib.pyplot as plt  # type: ignore
    import matplotlib.ticker as mticker  # type: ignore
except Exception as e:  # pragma: no cover
    raise SystemExit("This script requires matplotlib.") from e

# Ensure repo root is on sys.path.
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, os.pardir))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from scripts.plot.figure4_main_paper import plot_fig4 as fig4  # noqa: E402


TASKS: Tuple[str, ...] = (
    "BBH Raw",
    "GPQA Raw",
    "IFEval Raw",
    "MATH Lvl 5 Raw",
    "MMLU-PRO Raw",
    "MUSR Raw",
)

XLIM_PARAMS = (1e6, 2e11)
XTICKS_PARAMS = [1e7, 1e8, 1e9, 1e10, 1e11]


def _maybe_scale_scores(y: np.ndarray) -> np.ndarray:
    y = np.asarray(y, float)
    if y.size and np.nanmax(y) > 1.0:
        return y / 100.0
    return y


def _load_points_df(tasks: Sequence[str]) -> pd.DataFrame:
    usecols = ["Pretraining tokens (T)", "#Params (B)", "Type", "Official Providers", *list(tasks)]
    df = pd.read_csv(fig4.OLL_CSV, usecols=usecols)

    tokens = pd.to_numeric(df["Pretraining tokens (T)"], errors="coerce")
    params = pd.to_numeric(df["#Params (B)"], errors="coerce")
    compute_scaled = 6.0 * tokens * params
    df["compute_flops"] = compute_scaled * 1e21
    df["model_size_params"] = params * 1e9
    df["is_official_pretrained"] = (
        df["Type"].astype(str).str.lower().str.contains("pretrained", na=False)
        & df["Official Providers"].apply(fig4._is_true)
    )
    return df


def _get_points_for_task(
    df: pd.DataFrame, task: str, *, x_col: str
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x = pd.to_numeric(df.get(x_col, np.nan), errors="coerce").to_numpy(dtype=float)
    y = pd.to_numeric(df.get(task, np.nan), errors="coerce").to_numpy(dtype=float)
    y = _maybe_scale_scores(y)

    keep = np.isfinite(x) & (x > 0.0) & np.isfinite(y)
    x_all = x[keep]
    y_all = y[keep]

    is_pre = np.asarray(df.get("is_official_pretrained", False), dtype=bool)
    keep_pre = keep & is_pre
    x_pre = x[keep_pre]
    y_pre = y[keep_pre]
    return x_all, y_all, x_pre, y_pre


def _compute_dynamic_ylim(y_points: np.ndarray, y_curve: np.ndarray) -> Tuple[float, float]:
    candidates = []
    if y_points.size:
        candidates.extend([float(np.nanmin(y_points)), float(np.nanmax(y_points))])
    if y_curve.size:
        candidates.extend([float(np.nanmin(y_curve)), float(np.nanmax(y_curve))])

    if not candidates or not np.all(np.isfinite(candidates)):
        return (0.0, 1.0)

    y_min = float(np.nanmin(candidates))
    y_max = float(np.nanmax(candidates))
    if not np.isfinite(y_min) or not np.isfinite(y_max) or y_min == y_max:
        return (0.0, 1.0)

    pad = 0.02 * max(1e-6, (y_max - y_min))
    return (y_min - pad, y_max + pad)


def _configure_overlay_axes(
    ax,
    *,
    x_label: str,
    xlim: Tuple[float, float],
    xticks: Sequence[float],
) -> None:
    fig4.style_axes(ax)
    ax.set_xscale("log")
    ax.set_xlim(xlim)

    ax.set_xlabel(x_label, fontsize=10, fontweight="bold", labelpad=4)
    ax.set_ylabel("Accuracy", fontsize=10, fontweight="bold", labelpad=4)

    ax.set_xticks(list(xticks))
    ax.set_xticklabels([f"$10^{{{int(np.log10(x))}}}$" for x in xticks])

    ax.yaxis.set_major_locator(mticker.MultipleLocator(0.2))
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f"))

    ax.tick_params(axis="both", which="minor", length=0)
    ax.minorticks_off()


def _load_fit_json(*, fits_dir: Path, task: str, subset: str) -> dict:
    path = fits_dir / f"{fig4._task_slug(task)}_{subset}.json"
    with open(path, "r") as f:
        return json.load(f)


def _reference_subplot_bbox() -> Tuple[float, float, float, float]:
    """Return the (left, bottom, width, height) bbox of Figure 4's first subplot.

    This is used to match the physical axes size (in inches) so fonts/legend
    appear identical to the reference subplots.
    """

    fig4._configure_rcparams()
    fig, axes = plt.subplots(1, 3, figsize=(fig4.FIG_WIDTH, fig4.FIG_HEIGHT), dpi=fig4.DPI)
    plt.subplots_adjust(left=0.08, right=0.955, bottom=0.18, top=0.96, wspace=0.52)
    bbox = axes[0].get_position()
    plt.close(fig)
    return (float(bbox.x0), float(bbox.y0), float(bbox.width), float(bbox.height))


def _plot_overlay(
    task: str,
    df_points: pd.DataFrame,
    *,
    fits_dir: Path,
    out_dir: Path,
    ax_bbox: Tuple[float, float, float, float],
    pad_inches: float,
    x_col: str,
    x_label: str,
    xlim: Tuple[float, float],
    xticks: Sequence[float],
) -> None:
    x_all, y_all, x_pre, y_pre = _get_points_for_task(df_points, task, x_col=x_col)

    fit_post = _load_fit_json(fits_dir=fits_dir, task=task, subset="post")
    fit_pre = _load_fit_json(fits_dir=fits_dir, task=task, subset="pre")
    x_post, y_post = fig4._sample_curve_from_fit(fit_post)
    x_pre_curve, y_pre_curve = fig4._sample_curve_from_fit(fit_pre)

    fig = plt.figure(figsize=(fig4.FIG_WIDTH, fig4.FIG_HEIGHT), dpi=fig4.DPI)
    ax = fig.add_axes(ax_bbox)
    _configure_overlay_axes(ax, x_label=x_label, xlim=xlim, xticks=xticks)

    ax.scatter(
        x_all,
        y_all,
        s=fig4.STYLE["marker_size"] ** 2,
        alpha=fig4.STYLE["marker_alpha"],
        color=fig4.COLORS["points"],
        edgecolors=fig4.STYLE["marker_edgecolor"],
        linewidths=fig4.STYLE["marker_edgewidth"],
        zorder=2,
        rasterized=True,
    )
    if x_pre.size:
        ax.scatter(
            x_pre,
            y_pre,
            s=(fig4.STYLE["marker_size"] * 1.5) ** 2,
            alpha=fig4.STYLE["marker_alpha_pretrained"],
            color=fig4.COLORS["pretrained_points"],
            edgecolors=fig4.STYLE["marker_edgecolor"],
            linewidths=fig4.STYLE["marker_edgewidth_pretrained"],
            marker="^",
            zorder=3,
            rasterized=True,
        )

    if x_post.size:
        ax.plot(
            x_post,
            y_post,
            linewidth=fig4.STYLE["line_width"],
            alpha=fig4.STYLE["line_alpha"],
            color=fig4.COLORS["smooth_98_posttrain"],
            linestyle="-",
            zorder=4,
        )
    if x_pre_curve.size:
        ax.plot(
            x_pre_curve,
            y_pre_curve,
            linewidth=fig4.STYLE["line_width"],
            alpha=fig4.STYLE["line_alpha"],
            color=fig4.COLORS["smooth_98_pretrain"],
            linestyle="--",
            dashes=(5, 3),
            zorder=4,
        )

    y_lo, y_hi = _compute_dynamic_ylim(y_all, y_post)
    ax.set_ylim(y_lo, y_hi)
    ax.set_title("")

    handles, labels = fig4._create_scatter_legend_handles()
    legend = ax.legend(
        handles,
        labels,
        loc="upper left",
        bbox_to_anchor=(0.02, 0.76),
        frameon=True,
        framealpha=0.95,
        edgecolor="#999999",
        fancybox=False,
        fontsize=7,
        ncol=1,
        handlelength=1.5,
        handletextpad=0.4,
        labelspacing=0.35,
        borderpad=0.5,
    )
    legend.set_zorder(10)

    out_base = out_dir / f"overlay_{fig4._task_slug(task)}"
    fig.savefig(
        out_base.with_suffix(".pdf"),
        dpi=fig4.DPI,
        bbox_inches="tight",
        pad_inches=float(pad_inches),
        facecolor="white",
        edgecolor="none",
    )
    fig.savefig(
        out_base.with_suffix(".png"),
        dpi=fig4.DPI,
        bbox_inches="tight",
        pad_inches=float(pad_inches),
        facecolor="white",
        edgecolor="none",
    )
    plt.close(fig)


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Regenerate pretrain-vs-posttrain overlay figures.")
    parser.add_argument(
        "--x-axis",
        choices=("compute", "size"),
        default="compute",
        help="Which x-axis variant to regenerate (default: compute).",
    )
    parser.add_argument(
        "--pad-inches",
        type=float,
        default=0.08,
        help="Padding for bbox_inches='tight' (in inches). Default matches figure4_refined export.",
    )
    args = parser.parse_args(argv)

    fig4._configure_rcparams()

    if str(args.x_axis) == "size":
        base_dir = Path(REPO_ROOT) / "outputs" / "sigmoid_size" / "no_split" / "pretrain_vs_posttrain"
        x_col = "model_size_params"
        x_label = "Model Size (#Params)"
        xlim = XLIM_PARAMS
        xticks = XTICKS_PARAMS
    else:
        base_dir = Path(fig4.BASE_DIR)
        x_col = "compute_flops"
        x_label = "Pretraining Compute (FLOPs)"
        xlim = fig4.XLIM_FLOPS
        xticks = fig4.XTICKS_FLOPS

    fits_dir = base_dir / "fits"
    plots_dir = base_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    df_points = _load_points_df(TASKS)

    ax_bbox = _reference_subplot_bbox()
    for task in TASKS:
        _plot_overlay(
            task,
            df_points,
            fits_dir=fits_dir,
            out_dir=plots_dir,
            ax_bbox=ax_bbox,
            pad_inches=float(args.pad_inches),
            x_col=x_col,
            x_label=x_label,
            xlim=xlim,
            xticks=xticks,
        )

    print(f"Updated overlays in: {plots_dir}")
    print("Files: overlay_*.pdf and overlay_*.png")


if __name__ == "__main__":
    main()

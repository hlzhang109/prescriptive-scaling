#!/usr/bin/env python3
"""
Create Figure 7 (main paper): temporal trends with density heatmaps (2-panel layout).

Data source:
  outputs/open_llm_leaderboard/current/v2_scaling_metrics_*.csv

Outputs:
  outputs/figures_main_paper/figure7_main_paper.pdf
  outputs/figures_main_paper/figure7_main_paper.png
"""

from __future__ import annotations

import argparse
import os
from typing import Dict, Optional, Sequence, Tuple

import numpy as np

try:
    import pandas as pd  # type: ignore
except Exception as e:  # pragma: no cover
    raise SystemExit("This script requires pandas.") from e

try:
    import matplotlib.pyplot as plt  # type: ignore
    import matplotlib as mpl  # type: ignore
    import matplotlib.dates as mdates  # type: ignore
    from matplotlib.colors import LinearSegmentedColormap, LogNorm  # type: ignore
except Exception as e:  # pragma: no cover
    raise SystemExit("This script requires matplotlib.") from e

from skill_frontier.plotting.plot_utils import apply_font_embedding  # type: ignore

# -----------------------------------------------------------------------------
# CONFIG (match Figure 1)
# -----------------------------------------------------------------------------

FIG_WIDTH = 7.0
FIG_HEIGHT = 2.4
DPI = 300

OUT_DIR = os.path.join("outputs", "figures_main_paper")
OUT_PDF = os.path.join(OUT_DIR, "figure7_main_paper.pdf")

DATA_DIR = os.path.join("outputs", "open_llm_leaderboard", "current")

TASK_FILES: Dict[str, str] = {
    "MMLU-Pro": "v2_scaling_metrics_mmlu_pro.csv",
    "Math Lvl 5": "v2_scaling_metrics_math_lvl_5.csv",
    "BBH": "v2_scaling_metrics_bbh.csv",
    "GPQA": "v2_scaling_metrics_gpqa.csv",
    "IFEval": "v2_scaling_metrics_ifeval.csv",
    "MUSR": "v2_scaling_metrics_musr.csv",
}

DEFAULT_TASK_A = "MMLU-Pro"
DEFAULT_TASK_B = "Math Lvl 5"

METRIC_COL = "dominance_rate_frontier"
METRIC_LABEL = "Frontier Dominance Rate"

STYLE = {
    "line_width": 2.0,
    "grid_alpha": 0.2,
    "grid_linestyle": ":",
    "grid_linewidth": 0.5,
    "spine_width": 0.8,
}

colors_density = [
    "#deebf7",
    "#c6dbef",
    "#9ecae1",
    "#6baed6",
    "#4292c6",
    "#2171b5",
    "#08519c",
    "#08306b",
]
cmap_density = LinearSegmentedColormap.from_list("density_blue", colors_density, N=256)


def _configure_rcparams() -> None:
    mpl.rcParams["figure.dpi"] = DPI
    mpl.rcParams["savefig.dpi"] = DPI
    apply_font_embedding(42)
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 10,
            "axes.labelsize": 11,
            "axes.titlesize": 12,
            "axes.labelweight": "bold",
            "xtick.labelsize": 9.5,
            "ytick.labelsize": 9.5,
            "legend.fontsize": 9,
            "axes.linewidth": 0.8,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "mathtext.fontset": "dejavusans",
        }
    )


def apply_panel_style(ax) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(STYLE["spine_width"])
    ax.spines["bottom"].set_linewidth(STYLE["spine_width"])
    ax.grid(
        True,
        alpha=STYLE["grid_alpha"],
        linestyle=STYLE["grid_linestyle"],
        linewidth=STYLE["grid_linewidth"],
        zorder=0,
    )
    ax.tick_params(axis="both", which="major", length=4, width=0.8, direction="out")
    ax.tick_params(axis="both", which="minor", length=0)
    ax.set_facecolor("white")


def _load_task_df(task_label: str) -> pd.DataFrame:
    filename = TASK_FILES.get(task_label)
    if not filename:
        raise ValueError(f"Unknown task label: {task_label}")
    path = os.path.join(DATA_DIR, filename)
    df = pd.read_csv(path)
    df["month"] = pd.to_datetime(df["month"], errors="coerce")
    df[METRIC_COL] = pd.to_numeric(df[METRIC_COL], errors="coerce")
    df = df.dropna(subset=["month", METRIC_COL]).copy()
    return df


def _density_hexbin(
    ax,
    x,
    y,
    *,
    weights: Optional[np.ndarray] = None,
    gridsize: int = 18,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
):
    return ax.hexbin(
        x,
        y,
        gridsize=gridsize,
        cmap=cmap_density,
        mincnt=1,
        edgecolors="none",
        linewidths=0.2,
        alpha=0.95,
        C=weights,
        reduce_C_function=np.sum if weights is not None else None,
        vmin=vmin,
        vmax=vmax,
        zorder=2,
    )


def _panel_title(ax, text: str) -> None:
    ax.text(
        0.5,
        0.96,
        text,
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=10,
        fontweight="bold",
        bbox=dict(
            boxstyle="round,pad=0.35",
            facecolor="white",
            edgecolor="#cccccc",
            linewidth=0.5,
            alpha=0.95,
        ),
        zorder=10,
    )


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Create Figure 7 (temporal density heatmaps).")
    parser.add_argument("--task-a", default=DEFAULT_TASK_A, choices=sorted(TASK_FILES.keys()))
    parser.add_argument("--task-b", default=DEFAULT_TASK_B, choices=sorted(TASK_FILES.keys()))
    parser.add_argument("--out", default=OUT_PDF, help="Output PDF path.")
    args = parser.parse_args(argv)

    _configure_rcparams()

    df_a = _load_task_df(args.task_a)
    df_b = _load_task_df(args.task_b)

    x_a = mdates.date2num(df_a["month"])
    y_a = df_a[METRIC_COL].to_numpy(dtype=float)
    w_a = pd.to_numeric(df_a["n_total"], errors="coerce").fillna(1).to_numpy(dtype=float)
    x_b = mdates.date2num(df_b["month"])
    y_b = df_b[METRIC_COL].to_numpy(dtype=float)
    w_b = pd.to_numeric(df_b["n_total"], errors="coerce").fillna(1).to_numpy(dtype=float)

    x_min = float(np.nanmin(np.concatenate([x_a, x_b])))
    x_max = float(np.nanmax(np.concatenate([x_a, x_b])))

    fig, axes = plt.subplots(1, 2, figsize=(FIG_WIDTH, FIG_HEIGHT), dpi=DPI)
    plt.subplots_adjust(left=0.09, right=0.97, bottom=0.16, top=0.92, wspace=0.28)

    for ax in axes:
        apply_panel_style(ax)
        ax.set_xlim(x_min, x_max)
        ax.xaxis.set_major_locator(mdates.YearLocator(1))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        ax.set_ylim(0.0, 1.0)
        ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])

    hb_a = _density_hexbin(axes[0], x_a, y_a, weights=w_a, gridsize=18, vmin=1)
    hb_b = _density_hexbin(axes[1], x_b, y_b, weights=w_b, gridsize=18, vmin=1)
    def _max_count(hb) -> float:
        arr = hb.get_array()
        if hasattr(arr, "compressed"):
            arr = arr.compressed()
        arr = np.asarray(arr)
        return float(arr.max()) if arr.size else 1.0

    vmax = float(max(_max_count(hb_a), _max_count(hb_b)))
    norm = LogNorm(vmin=1, vmax=max(2.0, vmax))
    hb_a.set_norm(norm)
    hb_b.set_norm(norm)

    axes[0].set_xlabel("Time Period", fontsize=10, fontweight="bold", labelpad=4)
    axes[0].set_ylabel(METRIC_LABEL, fontsize=10, fontweight="bold", labelpad=4)
    axes[1].set_xlabel("Time Period", fontsize=10, fontweight="bold", labelpad=4)
    axes[1].set_ylabel("")
    axes[1].set_yticklabels([])

    axes[0].text(-0.15, 1.05, "(a)", transform=axes[0].transAxes, ha="left", va="top", fontsize=11, fontweight="bold")
    axes[1].text(-0.15, 1.05, "(b)", transform=axes[1].transAxes, ha="left", va="top", fontsize=11, fontweight="bold")

    _panel_title(axes[0], args.task_a)
    _panel_title(axes[1], args.task_b)

    cbar = fig.colorbar(hb_b, ax=axes, pad=0.02, fraction=0.046)
    cbar.set_label("Count", fontsize=9, fontweight="normal", labelpad=6)
    cbar.ax.tick_params(labelsize=8)

    out_pdf = os.path.abspath(str(args.out))
    os.makedirs(os.path.dirname(out_pdf), exist_ok=True)
    fig.savefig(out_pdf, dpi=DPI, bbox_inches="tight", pad_inches=0.05, facecolor="white", edgecolor="none")
    fig.savefig(os.path.splitext(out_pdf)[0] + ".png", dpi=DPI, bbox_inches="tight", pad_inches=0.05,
                facecolor="white", edgecolor="none")
    print(f"Saved: {out_pdf}")
    plt.close(fig)


if __name__ == "__main__":
    main()

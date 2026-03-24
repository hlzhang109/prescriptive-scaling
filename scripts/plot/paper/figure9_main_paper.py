#!/usr/bin/env python3
"""
Create Figure 9 (main paper): gap-to-boundary + post-training lift (1×2).

The provided "source data files" are rendered PDFs:
  - outputs/sigmoid/no_split/pretrain_vs_posttrain/plots/gap_all_tasks.pdf
  - outputs/sigmoid/no_split/pretrain_vs_posttrain/plots/lift_all_tasks.pdf

Rather than attempting to extract data from PDFs, this script loads the exact
underlying tables used to generate those plots:
  - outputs/sigmoid/no_split/pretrain_vs_posttrain/tables/gap_bins_{TASK}.csv
  - outputs/sigmoid/no_split/pretrain_vs_posttrain/tables/lift_{TASK}.csv

Output (overwritten if exists):
  - outputs/figures_main_paper/figure9_main_paper.pdf

Usage:
  python scripts/plot/paper/figure9_main_paper.py
  python scripts/plot/paper/figure9_main_paper.py --out outputs/figures_main_paper/figure9_main_paper.pdf
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import numpy as np

try:
    import pandas as pd  # type: ignore
except Exception as e:  # pragma: no cover
    raise SystemExit("This script requires pandas.") from e

try:
    import matplotlib.pyplot as plt  # type: ignore
    from matplotlib.lines import Line2D  # type: ignore
    from matplotlib.ticker import FuncFormatter  # type: ignore
except Exception as e:  # pragma: no cover
    raise SystemExit("This script requires matplotlib.") from e


# -----------------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------------

BASE_DIR = Path("outputs/sigmoid/no_split/pretrain_vs_posttrain")
PLOTS_DIR = BASE_DIR / "plots"
TABLES_DIR = BASE_DIR / "tables"

GAP_SOURCE_PDF = PLOTS_DIR / "gap_all_tasks.pdf"
LIFT_SOURCE_PDF = PLOTS_DIR / "lift_all_tasks.pdf"

DEFAULT_OUT = Path("outputs/figures_main_paper/figure9_main_paper.pdf")


# -----------------------------------------------------------------------------
# Style config (as requested)
# -----------------------------------------------------------------------------

FIG_WIDTH = 7.0
FIG_HEIGHT = 3.5
DPI = 300

TASK_COLORS = {
    "BBH": "#377eb8",
    "GPQA": "#ff7f00",
    "IFEval": "#4daf4a",
    "MATH Lvl 5": "#e41a1c",
    "MMLU-PRO": "#984ea3",
    "MUSR": "#a65628",
}

STYLE = {
    "line_width": 2.0,
    "marker_size": 4.0,
    "marker_alpha": 0.7,
    "line_alpha": 0.9,
    "grid_alpha": 0.2,
    "grid_linestyle": ":",
    "grid_linewidth": 0.5,
    "spine_width": 0.8,
}

TITLE_BBOX = dict(
    boxstyle="round,pad=0.35",
    facecolor="#f0f0f0",
    edgecolor="#cccccc",
    linewidth=0.8,
    alpha=0.95,
)


RAW_SUFFIX_RE = re.compile(r"\s*Raw\s*$")


def _task_slug(task_raw: str) -> str:
    return str(task_raw).replace(" ", "_")


def _strip_raw(task_raw: str) -> str:
    return RAW_SUFFIX_RE.sub("", str(task_raw)).strip()


def _log_formatter(x: float, pos: Optional[int] = None) -> str:  # noqa: ARG001
    if x is None or not np.isfinite(x) or x <= 0:
        return ""
    exponent = int(np.round(np.log10(float(x))))
    return f"$10^{{{exponent}}}$"


def _configure_matplotlib() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": DPI,
            "savefig.dpi": DPI,
            "font.family": "serif",
            "font.serif": ["Times New Roman"],
            "font.size": 11,
            "axes.labelsize": 12,
            "axes.titlesize": 13,
            "axes.labelweight": "bold",
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 9.5,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def _load_tasks_order() -> Tuple[str, ...]:
    summary_path = TABLES_DIR / "pretrain_posttrain_heterogeneity_summary.csv"
    if summary_path.exists():
        df = pd.read_csv(summary_path)
        tasks = [str(t) for t in df.get("task", []).tolist()]
        tasks = [t for t in tasks if t]
        if tasks:
            return tuple(tasks)
    return (
        "BBH Raw",
        "GPQA Raw",
        "IFEval Raw",
        "MATH Lvl 5 Raw",
        "MMLU-PRO Raw",
        "MUSR Raw",
    )


def load_gap_data(*, tasks_order_raw: Sequence[str]) -> Dict[str, pd.DataFrame]:
    """Load binned gap-to-boundary summaries for each task (median + IQR)."""
    out: Dict[str, pd.DataFrame] = {}
    for task_raw in tasks_order_raw:
        path = TABLES_DIR / f"gap_bins_{_task_slug(task_raw)}.csv"
        if not path.exists():
            continue
        df = pd.read_csv(path)
        required = {"x_center_scaled", "median_gap", "q25_gap", "q75_gap"}
        if not required.issubset(set(df.columns)):
            continue
        df = df.copy()
        df["compute_flops"] = pd.to_numeric(df["x_center_scaled"], errors="coerce") * 1e21
        df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=["compute_flops", "median_gap"])
        df = df[df["compute_flops"] > 0].sort_values("compute_flops")
        out[_strip_raw(task_raw)] = df
    return out


def load_lift_data(*, tasks_order_raw: Sequence[str]) -> Dict[str, pd.DataFrame]:
    """Load per-base lift points for each task (robust vs fallback)."""
    out: Dict[str, pd.DataFrame] = {}
    for task_raw in tasks_order_raw:
        path = TABLES_DIR / f"lift_{_task_slug(task_raw)}.csv"
        if not path.exists():
            continue
        df = pd.read_csv(path)
        required = {"compute_scaled", "lift", "lift_is_robust"}
        if not required.issubset(set(df.columns)):
            continue
        df = df.copy()
        df["compute_flops"] = pd.to_numeric(df["compute_scaled"], errors="coerce") * 1e21
        df["lift"] = pd.to_numeric(df["lift"], errors="coerce")
        df["lift_is_robust"] = df["lift_is_robust"].astype(str).str.lower().isin(["1", "true", "t", "yes", "y"])
        df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=["compute_flops", "lift"])
        df = df[df["compute_flops"] > 0].sort_values("compute_flops")
        out[_strip_raw(task_raw)] = df
    return out


def _style_axes(ax) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(STYLE["spine_width"])
    ax.spines["bottom"].set_linewidth(STYLE["spine_width"])
    ax.tick_params(axis="both", which="major", length=4, width=STYLE["spine_width"], direction="out")
    ax.grid(True, alpha=STYLE["grid_alpha"], linestyle=STYLE["grid_linestyle"], linewidth=STYLE["grid_linewidth"])
    ax.set_xscale("log")
    ax.xaxis.set_major_formatter(FuncFormatter(_log_formatter))


def _compute_shared_xlim(gap_by_task: Dict[str, pd.DataFrame], lift_by_task: Dict[str, pd.DataFrame]) -> Tuple[float, float]:
    xs = []
    for df in list(gap_by_task.values()) + list(lift_by_task.values()):
        if "compute_flops" in df.columns:
            vals = pd.to_numeric(df["compute_flops"], errors="coerce").to_numpy(dtype=float)
            vals = vals[np.isfinite(vals) & (vals > 0)]
            if vals.size:
                xs.append(float(np.nanmin(vals)))
                xs.append(float(np.nanmax(vals)))
    if not xs:
        return (2e20, 1e25)
    x_min = float(np.nanmin(xs))
    x_max = float(np.nanmax(xs))
    x_min = max(x_min * 0.95, 1e-12)
    x_max = max(1e25, x_max * 1.05)
    return (x_min, x_max)


def create_figure(*, out_path: Path) -> None:
    tasks_order_raw = _load_tasks_order()
    gap_by_task = load_gap_data(tasks_order_raw=tasks_order_raw)
    lift_by_task = load_lift_data(tasks_order_raw=tasks_order_raw)

    fig, axes = plt.subplots(1, 2, figsize=(FIG_WIDTH, FIG_HEIGHT), dpi=DPI)
    plt.subplots_adjust(left=0.10, right=0.96, bottom=0.14, top=0.92, wspace=0.30)

    # Panel (a): Gap to boundary
    ax_gap = axes[0]
    for task in [_strip_raw(t) for t in tasks_order_raw]:
        df = gap_by_task.get(task)
        if df is None or df.empty or task not in TASK_COLORS:
            continue
        x = df["compute_flops"].to_numpy(dtype=float)
        y = pd.to_numeric(df["median_gap"], errors="coerce").to_numpy(dtype=float)
        y25 = pd.to_numeric(df["q25_gap"], errors="coerce").to_numpy(dtype=float)
        y75 = pd.to_numeric(df["q75_gap"], errors="coerce").to_numpy(dtype=float)
        ax_gap.plot(
            x,
            y,
            linewidth=STYLE["line_width"],
            alpha=STYLE["line_alpha"],
            color=TASK_COLORS[task],
            marker="o",
            markersize=STYLE["marker_size"],
            label=task,
            zorder=3,
        )
        ax_gap.fill_between(x, y25, y75, color=TASK_COLORS[task], alpha=0.12, linewidth=0, zorder=2)
    ax_gap.axhline(0.0, color="#666666", linewidth=1.0, linestyle="--", alpha=0.9, zorder=1)
    ax_gap.set_xlabel("Pretraining Compute (FLOPs)", fontweight="bold")
    ax_gap.set_ylabel("Gap to boundary", fontweight="bold")
    ax_gap.set_title("(a) Gap to post-trained boundary", fontweight="bold", pad=8).set_bbox(TITLE_BBOX)

    # Panel (b): Post-training lift
    ax_lift = axes[1]
    for task in [_strip_raw(t) for t in tasks_order_raw]:
        df = lift_by_task.get(task)
        if df is None or df.empty or task not in TASK_COLORS:
            continue
        robust = df["lift_is_robust"].astype(bool).to_numpy()
        ax_lift.scatter(
            df.loc[robust, "compute_flops"].to_numpy(dtype=float),
            df.loc[robust, "lift"].to_numpy(dtype=float),
            s=STYLE["marker_size"] ** 2,
            alpha=STYLE["marker_alpha"],
            color=TASK_COLORS[task],
            edgecolors="none",
            zorder=3,
            rasterized=True,
        )
        ax_lift.scatter(
            df.loc[~robust, "compute_flops"].to_numpy(dtype=float),
            df.loc[~robust, "lift"].to_numpy(dtype=float),
            s=STYLE["marker_size"] ** 2,
            alpha=STYLE["marker_alpha"],
            facecolors="none",
            edgecolors=TASK_COLORS[task],
            linewidths=1.1,
            zorder=3,
            rasterized=True,
        )
    ax_lift.axhline(0.0, color="#666666", linewidth=1.0, linestyle="--", alpha=0.9, zorder=1)
    ax_lift.set_xlabel("Pretraining Compute (FLOPs)", fontweight="bold")
    ax_lift.set_ylabel("Post-training lift", fontweight="bold")
    ax_lift.set_title("(b) Post-training Lift", fontweight="bold", pad=8).set_bbox(TITLE_BBOX)

    # Common styling + shared x-limits/ticks.
    xlim = _compute_shared_xlim(gap_by_task, lift_by_task)
    xticks = [1e21, 1e22, 1e23, 1e24, 1e25]
    for ax in axes:
        _style_axes(ax)
        ax.set_xlim(xlim)
        ax.set_xticks(xticks)

    # Y-limits based on data (avoid clipping negatives in lift).
    if gap_by_task:
        gap_max = max(
            float(np.nanmax(pd.to_numeric(df["q75_gap"], errors="coerce").to_numpy(dtype=float)))
            for df in gap_by_task.values()
            if "q75_gap" in df.columns and df.shape[0]
        )
        ax_gap.set_ylim(0.0, max(0.6, gap_max * 1.05))
    if lift_by_task:
        all_lifts = np.concatenate(
            [
                pd.to_numeric(df["lift"], errors="coerce").to_numpy(dtype=float)
                for df in lift_by_task.values()
                if "lift" in df.columns and df.shape[0]
            ]
        )
        all_lifts = all_lifts[np.isfinite(all_lifts)]
        if all_lifts.size:
            y_min = float(np.nanmin(all_lifts))
            y_max = float(np.nanmax(all_lifts))
            pad = 0.06 * max(1e-6, y_max - y_min)
            ax_lift.set_ylim(y_min - pad, y_max + pad)

    # Legends
    ax_gap.legend(
        loc="upper left",
        frameon=True,
        framealpha=0.9,
        edgecolor="gray",
        fancybox=False,
        fontsize=8.5,
        ncol=1,
        handlelength=1.5,
        handletextpad=0.5,
    )
    estimator_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor="#666666",
            markeredgecolor="#666666",
            markersize=6,
            label="robust (q0.95)",
            linestyle="",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor="none",
            markeredgecolor="#666666",
            markersize=6,
            label="fallback (max)",
            linestyle="",
        ),
    ]
    ax_lift.legend(
        handles=estimator_handles,
        loc="upper left",
        frameon=True,
        framealpha=0.9,
        edgecolor="gray",
        fancybox=False,
        fontsize=8.5,
        ncol=1,
        handlelength=1.5,
        handletextpad=0.5,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight", pad_inches=0.05, facecolor="white", edgecolor="none")
    plt.close(fig)


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Create Figure 9 (main paper).")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help="Output PDF path.")
    args = parser.parse_args(argv)

    _configure_matplotlib()
    create_figure(out_path=args.out)
    print(f"Saved: {args.out}")
    print(f"Reference PDFs: {GAP_SOURCE_PDF} and {LIFT_SOURCE_PDF}")


if __name__ == "__main__":
    main()

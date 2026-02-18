#!/usr/bin/env python3
"""
Generate Figure 1 for the main paper: 2x3 grid showing MMLU-Pro and Math Lvl 5
across periods k=1, 2, 3.

Usage:
    python scripts/plot/paper/figure1_main_paper.py
"""

import os
import sys
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Import bootstrap from `scripts/` by putting the scripts dir on sys.path.
SCRIPTS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)
from _bootstrap import ensure_repo_root_on_path  # type: ignore

REPO_ROOT = ensure_repo_root_on_path(__file__)

# Use the legacy sigmoid fitter to preserve the published Figure 1 artifact.
from skill_frontier.core.sigmoid_legacy import fit_sigmoid_frontier_legacy as fit_sigmoid_frontier
from skill_frontier.io.compute_utils import compute_flops_from_tokens_params
from skill_frontier.io.csv_utils import detect_date_col_flexible, load_leaderboard_results
from skill_frontier.io.period_utils import parse_year_month
from skill_frontier.core.period_scheme import PERIOD4_SPLITS_NEW
from skill_frontier.plotting.paper_style import apply_rcparams, STYLE_FIGURE1
from skill_frontier.plotting.plot_utils import ensure_dir, save_figure
from skill_frontier.plotting.plot_paths import FIGURES_MAIN_PAPER_DIR

# ============================================================================
# CONFIGURATION
# ============================================================================

# Tasks and periods to plot
TASKS = ['MMLU-PRO_Raw', 'MATH_Lvl_5_Raw']
TASK_LABELS = ['MMLU-Pro', 'Math Lvl 5']
PERIODS = [1, 2, 3]

# Period definitions for train/val split (same as in project).
# Figure 1 uses the cumulative train split with a single held-out validation period.
PERIOD4_SPLITS = [
    {"train_labels": list(spec["train_labels"]), "test_labels": [str(spec["val_label"])]}
    for spec in PERIOD4_SPLITS_NEW
]

# Paths
DATA_PATH = os.path.join(REPO_ROOT, "tables/open_llm_leaderboard/open_llm_leaderboard_with_tokens.csv")
CURVES_DIR = os.path.join(REPO_ROOT, "outputs/sigmoid/period4/single_k/curves")
OUT_DIR = os.path.join(REPO_ROOT, str(FIGURES_MAIN_PAPER_DIR))

# Task column mapping
TASK_COLUMN_MAP = {
    'MMLU-PRO_Raw': 'MMLU-PRO Raw',
    'MATH_Lvl_5_Raw': 'MATH Lvl 5 Raw',
}

# ============================================================================
# STYLING
# ============================================================================

apply_rcparams("figure1")

# Visual styling config (shared)
STYLE = STYLE_FIGURE1

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def is_in_period(date_tuple: Optional[Tuple[int, int]], period_labels: List[str]) -> bool:
    """Check if a date falls within the given period labels."""
    if date_tuple is None:
        return False

    y, m = date_tuple

    for label in period_labels:
        if label.startswith("<="):
            cutoff = label[2:]
            cy, cm = map(int, cutoff.split("-"))
            if (y, m) <= (cy, cm):
                return True
        elif ".." in label:
            lo, hi = label.split("..")
            ly, lm = map(int, lo.split("-"))
            hy, hm = map(int, hi.split("-"))
            if (y, m) >= (ly, lm) and (y, m) <= (hy, hm):
                return True

    return False


def load_raw_data() -> pd.DataFrame:
    """Load the main leaderboard data with compute values."""
    df = load_leaderboard_results(DATA_PATH)

    # Parse submission date
    date_col = detect_date_col_flexible(df.columns)
    if date_col is None:
        raise ValueError("Could not find a suitable date column in the input CSV.")

    df['period'] = df[date_col].apply(parse_year_month)

    # Compute log10(FLOPs) where FLOPs = 6 * tokens * params
    flops = compute_flops_from_tokens_params(
        df['Pretraining tokens (T)'],
        df['#Params (B)'],
        multiplier=6.0,
    )  # 6 * (T tokens) * (B params) in FLOPs
    flops = flops.where(flops > 0, np.nan)
    df['logC'] = np.log10(flops)

    return df


def load_curve_data(task: str, k: int) -> pd.DataFrame:
    """Load fitted curve data for a specific task and period."""
    curve_file = os.path.join(CURVES_DIR, f"{task}_k{k}.csv")
    if os.path.exists(curve_file):
        return pd.read_csv(curve_file)
    return pd.DataFrame()


def fit_curve_on_data(x_flops: np.ndarray, y_acc: np.ndarray, tau: float = 0.98) -> Tuple[np.ndarray, np.ndarray]:
    """Fit a sigmoid frontier curve on data points.

    Args:
        x_flops: Compute values in FLOPs
        y_acc: Accuracy values
        tau: Quantile for frontier fitting (default 0.98)

    Returns:
        (curve_x, curve_y): Arrays of fitted curve points in FLOPs and accuracy
    """
    # Filter valid data
    valid = np.isfinite(x_flops) & np.isfinite(y_acc) & (x_flops > 0)
    if np.sum(valid) < 3:
        return np.array([]), np.array([])

    x_valid = x_flops[valid]
    y_valid = y_acc[valid]

    # Use the project's sigmoid fitting function
    curve_x, curve_y = fit_sigmoid_frontier(
        x_valid, y_valid, tau=tau, use_log10_x=True, grid_points=400
    )

    return curve_x, curve_y


def get_train_val_data(df: pd.DataFrame, task: str, k: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Get train and validation data for a specific task and period k."""
    split = PERIOD4_SPLITS[k - 1]  # k is 1-indexed

    task_col = TASK_COLUMN_MAP.get(task, task.replace('_', ' '))

    # Filter to rows with valid data
    mask_valid = df['logC'].notna() & df[task_col].notna()
    df_valid = df[mask_valid].copy()

    # Classify into train/val based on period
    is_train = df_valid['period'].apply(lambda p: is_in_period(p, split['train_labels']))
    is_val = df_valid['period'].apply(lambda p: is_in_period(p, split['test_labels']))

    df_train = df_valid[is_train].copy()
    df_val = df_valid[is_val].copy()

    return df_train, df_val




# ============================================================================
# PLOTTING FUNCTIONS
# ============================================================================

def create_figure():
    """Create the 2x3 figure with all subplots."""
    from matplotlib.lines import Line2D

    # Load raw data
    print("Loading raw data...")
    df = load_raw_data()

    # Figure dimensions
    fig_width = 7.0
    fig_height = 4.2
    dpi = 300

    fig, axes = plt.subplots(2, 3, figsize=(fig_width, fig_height), dpi=dpi)
    fig.patch.set_facecolor("white")

    # Subplot spacing
    plt.subplots_adjust(
        left=0.10,
        right=0.985,
        bottom=0.12,
        top=0.90,
        hspace=0.20,
        wspace=0.18,
    )

    # Y-axis limits (task-specific, will be adjusted based on data)
    ylim_mmlu = (0.1, 0.7)
    ylim_math = (0.0, 0.7)
    ylims = [ylim_mmlu, ylim_math]

    # X-axis limits (FLOPs) - matching original plot range
    xlim = (1e21, 1e25)
    xticks = [1e21, 1e22, 1e23, 1e24, 1e25]

    for row, (task, task_label, ylim) in enumerate(zip(TASKS, TASK_LABELS, ylims)):
        for col, k in enumerate(PERIODS):
            ax = axes[row, col]
            ax.set_facecolor("white")
            ax.set_axisbelow(True)

            print(f"Plotting {task_label}, k={k}...")

            # Get train/val data
            df_train, df_val = get_train_val_data(df, task, k)
            task_col = TASK_COLUMN_MAP.get(task, task.replace('_', ' '))

            # Convert logC to FLOPs for scatter points
            train_flops = 10 ** df_train['logC'].values
            train_acc = df_train[task_col].values

            val_flops = 10 ** df_val['logC'].values
            val_acc = df_val[task_col].values

            # Plot scatter points
            ax.scatter(train_flops, train_acc,
                      s=STYLE['marker_size']**2,
                      alpha=STYLE['marker_alpha'],
                      color=STYLE['train_points_color'],
                      edgecolors=STYLE['marker_edgecolor'],
                      linewidths=STYLE['marker_edgewidth'],
                      zorder=3,
                      rasterized=True)

            ax.scatter(val_flops, val_acc,
                      s=STYLE['marker_size']**2,
                      alpha=STYLE['marker_alpha'],
                      color=STYLE['val_points_color'],
                      edgecolors=STYLE['marker_edgecolor'],
                      linewidths=STYLE['marker_edgewidth'],
                      zorder=3,
                      rasterized=True)

            # Fit and plot curves for both train and val data
            # Train curve (solid)
            curve_train_x, curve_train_y = fit_curve_on_data(train_flops, train_acc, tau=0.98)
            if curve_train_x.size > 0:
                ax.plot(curve_train_x, curve_train_y,
                       linewidth=STYLE['line_width_fit'],
                       alpha=STYLE['line_alpha'],
                       color=STYLE['fit_train_color'],
                       linestyle='-',
                       zorder=5)

            # Val curve (dashed)
            curve_val_x, curve_val_y = fit_curve_on_data(val_flops, val_acc, tau=0.98)
            if curve_val_x.size > 0:
                ax.plot(curve_val_x, curve_val_y,
                       linewidth=STYLE['line_width_fit'],
                       alpha=STYLE['line_alpha'],
                       color=STYLE['fit_val_color'],
                       linestyle='--',
                       zorder=5)

            # Styling
            ax.set_xscale('log')
            ax.set_xlim(xlim)
            ax.set_ylim(ylim)

            # Grid
            ax.grid(
                True,
                color=STYLE['grid_color'],
                alpha=STYLE['grid_alpha'],
                linestyle=STYLE['grid_linestyle'],
                linewidth=STYLE['grid_linewidth'],
                zorder=0,
            )

            # Spines
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_linewidth(STYLE['spine_width'])
            ax.spines['bottom'].set_linewidth(STYLE['spine_width'])
            ax.spines['left'].set_color(STYLE['spine_color'])
            ax.spines['bottom'].set_color(STYLE['spine_color'])

            # Tick params
            ax.tick_params(axis='both', which='major',
                          length=4, width=0.8, direction='out', color=STYLE['spine_color'])
            ax.tick_params(axis='both', which='minor',
                          length=2.5, width=0.6, direction='out', color=STYLE['spine_color'])

            # X-axis ticks for all, but labels only for bottom row
            ax.set_xticks(xticks)
            if row == 1:  # bottom row
                ax.set_xticklabels([f'$10^{{{int(np.log10(x))}}}$' for x in xticks])
            else:  # top row
                ax.set_xticklabels([])

            # Leftmost column only: show y-axis labels
            if col != 0:
                ax.tick_params(axis="y", which="both", labelleft=False)


    # Column titles (t = 1, 2, 3)
    for col, k in enumerate(PERIODS):
        axes[0, col].text(
            0.5,
            0.93,
            f'$t = {k}$',
            transform=axes[0, col].transAxes,
            ha='center',
            va='top',
            fontsize=13,
            fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='none', alpha=0.8),
        )

    # Row labels (task names) - positioned outside left edge
    row_labels = ['MMLU-Pro', 'Math Lvl 5']
    for row, label in enumerate(row_labels):
        axes[row, 0].text(
            -0.20,
            0.5,
            label,
            transform=axes[row, 0].transAxes,
            ha='right',
            va='center',
            fontsize=13,
            fontweight='bold',
            rotation=90,
        )

    # Shared x-axis label at bottom
    fig.text(0.5, 0.02, 'Pretraining Compute (FLOPs)',
            ha='center', va='center', fontsize=16, fontweight='bold')

    # Unified legend across the top (avoids occluding any subplot)
    legend_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="None",
            markersize=6.5,
            markerfacecolor=STYLE["train_points_color"],
            markeredgecolor=STYLE["marker_edgecolor"],
            markeredgewidth=STYLE["marker_edgewidth"],
            label="Train",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="None",
            markersize=6.5,
            markerfacecolor=STYLE["val_points_color"],
            markeredgecolor=STYLE["marker_edgecolor"],
            markeredgewidth=STYLE["marker_edgewidth"],
            label="Val",
        ),
        Line2D(
            [0],
            [0],
            color=STYLE["fit_train_color"],
            linewidth=STYLE["line_width_fit"],
            linestyle="-",
            label="Fit (Train)",
        ),
        Line2D(
            [0],
            [0],
            color=STYLE["fit_val_color"],
            linewidth=STYLE["line_width_fit"],
            linestyle="--",
            label="Fit (Val)",
        ),
    ]
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.985),
        ncol=4,
        frameon=False,
        columnspacing=1.2,
        handletextpad=0.6,
        handlelength=2.2,
        borderaxespad=0.0,
    )

    return fig


def main():
    """Main function to create and save the figure."""

    # Create output directory
    ensure_dir(OUT_DIR)

    # Create figure
    fig = create_figure()

    # Save as PDF
    pdf_path, png_path = save_figure(
        fig,
        OUT_DIR,
        "figure1_main_paper",
        dpi=300,
        bbox_inches="tight",
        pad_inches=0.05,
        facecolor="white",
        edgecolor="none",
    )
    print(f"Saved: {pdf_path}")

    # Save as PNG
    print(f"Saved: {png_path}")

    print(f"\nFigure saved to {OUT_DIR}/")
    plt.close(fig)


if __name__ == "__main__":
    main()

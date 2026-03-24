"""Shared Matplotlib styling for main-paper figures.

This module centralizes rcParams and style dictionaries for the main-paper
figure scripts without changing any values.
"""

from __future__ import annotations

from typing import Dict

# -----------------------------------------------------------------------------
# rcParams presets (literal copies from existing figure scripts)
# -----------------------------------------------------------------------------

RCPARAMS_PAPER_FIGURE1: Dict[str, object] = {
    "font.family": "serif",
    "font.serif": ["Times New Roman"],
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
    "font.size": 15,
    "axes.labelsize": 16,
    "axes.titlesize": 16,
    "axes.labelweight": "bold",
    "xtick.labelsize": 13,
    "ytick.labelsize": 13,
    "legend.fontsize": 12,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "mathtext.fontset": "cm",
}

RCPARAMS_PAPER_FIGURE3_MPL: Dict[str, object] = {
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
}

RCPARAMS_PAPER_FIGURE3: Dict[str, object] = {
    "font.family": "serif",
    "font.serif": ["Times New Roman"],
    "font.size": 13,
    "axes.labelsize": 16,
    "axes.titlesize": 16,
    "axes.labelweight": "bold",
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 10,
    "axes.linewidth": 0.8,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
    "mathtext.fontset": "cm",
}

RCPARAMS_PAPER_FIGURE5: Dict[str, object] = {
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "font.family": "serif",
    "font.serif": ["Times New Roman"],
    "font.size": 14,
    "axes.labelsize": 14,
    "axes.titlesize": 14,
    "axes.labelweight": "bold",
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "axes.linewidth": 0.8,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
    "mathtext.fontset": "cm",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
}

RCPARAMS_PAPER_FIGURE6_MPL: Dict[str, object] = {
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
}

RCPARAMS_PAPER_FIGURE6: Dict[str, object] = {
    "font.family": "serif",
    "font.serif": ["Times New Roman"],
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
    "mathtext.fontset": "cm",
}

RCPARAMS_MAIN_PAPER: Dict[str, Dict[str, object]] = {
    "figure1": RCPARAMS_PAPER_FIGURE1,
    "figure3": RCPARAMS_PAPER_FIGURE3,
    "figure5": RCPARAMS_PAPER_FIGURE5,
    "figure6": RCPARAMS_PAPER_FIGURE6,
}

RCPARAMS_MAIN_PAPER_MPL: Dict[str, Dict[str, object]] = {
    "figure3": RCPARAMS_PAPER_FIGURE3_MPL,
    "figure6": RCPARAMS_PAPER_FIGURE6_MPL,
}

# -----------------------------------------------------------------------------
# Style dicts (literal copies from existing figure scripts)
# -----------------------------------------------------------------------------

STYLE_FIGURE1: Dict[str, object] = {
    "line_width_fit": 2.5,
    "marker_size": 4.2,
    "marker_alpha": 0.55,
    "marker_edgecolor": "white",
    "marker_edgewidth": 0.3,
    "line_alpha": 0.95,
    "train_points_color": "#377eb8",
    "val_points_color": "#e41a1c",
    "fit_train_color": "#377eb8",
    "fit_val_color": "#e41a1c",
    "grid_alpha": 0.2,
    "grid_linestyle": ":",
    "grid_linewidth": 0.6,
    "grid_color": "gray",
    "spine_width": 0.8,
    "spine_color": "#4d4d4d",
}

STYLE_FIGURE3: Dict[str, object] = {
    "grid_alpha": 0.2,
    "grid_linestyle": ":",
    "grid_linewidth": 0.5,
    "spine_width": 0.8,
    "line_width": 2.0,
    "line_width_drift": 2.6,
    "marker_size": 6.0,
    "marker_size_drift": 7.0,
    "alpha": 0.85,
    "alpha_drift": 0.95,
}

STYLE_FIGURE5: Dict[str, object] = {
    "marker_size": 3.5,
    "marker_alpha": 0.6,
    "line_width": 2.0,
    "line_alpha": 0.9,
    "grid_alpha": 0.2,
    "grid_linestyle": ":",
    "grid_linewidth": 0.5,
    "spine_width": 0.8,
}

STYLE_FIGURE6: Dict[str, object] = {
    "grid_alpha": 0.2,
    "grid_linestyle": ":",
    "grid_linewidth": 0.5,
    "spine_width": 0.8,
    "task_line_color": "#999999",
    "task_line_alpha": 0.55,
    "task_line_width": 1.6,
    "avg_color": "#e41a1c",
    "avg_alpha": 0.95,
    "avg_line_width": 2.4,
    "avg_marker": "o",
    "avg_markersize": 6.0,
    "avg_markeredgecolor": "white",
    "avg_markeredgewidth": 0.6,
}

STYLE_MAIN_PAPER: Dict[str, Dict[str, object]] = {
    "figure1": STYLE_FIGURE1,
    "figure3": STYLE_FIGURE3,
    "figure5": STYLE_FIGURE5,
    "figure6": STYLE_FIGURE6,
}

TITLE_BBOX_MAIN_PAPER: Dict[str, object] = {
    "boxstyle": "round,pad=0.35",
    "facecolor": "#f0f0f0",
    "edgecolor": "#cccccc",
    "linewidth": 0.8,
    "alpha": 0.95,
}


def apply_rcparams(style_name: str = "paper") -> None:
    """Apply rcParams for a named main-paper style preset."""
    from matplotlib import rcParams as mpl_rcparams  # type: ignore
    import matplotlib.pyplot as plt  # type: ignore

    name = style_name.strip().lower()
    aliases = {
        "paper": "figure1",
        "fig1": "figure1",
        "fig3": "figure3",
        "fig5": "figure5",
        "fig6": "figure6",
    }
    name = aliases.get(name, name)

    if name not in RCPARAMS_MAIN_PAPER:
        raise ValueError(f"Unknown main-paper style: {style_name!r}")

    if name in RCPARAMS_MAIN_PAPER_MPL:
        mpl_rcparams.update(RCPARAMS_MAIN_PAPER_MPL[name])

    plt.rcParams.update(RCPARAMS_MAIN_PAPER[name])

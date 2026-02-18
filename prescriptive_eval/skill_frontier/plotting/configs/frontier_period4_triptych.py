from __future__ import annotations

# Period4 single_k triptych style: used by
# - scripts/smooth_single_skill_frontier.py (period4 triptych plots)

# Slightly taller than Figure 1's per-row height to avoid any overlap between the
# shared x-label and the subplots when saved with tight bounding boxes.
FIGSIZE = (7.0, 2.0)

import matplotlib.pyplot as plt
plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Computer Modern Roman']

# Match the subplot styling used in outputs/figures_main_paper/figure1_main_paper.*
RCPARAMS = {
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"],
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
    "font.size": 10,
    "axes.labelsize": 13,
    "axes.titlesize": 13,
    "axes.labelweight": "bold",
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 9,
    # Use a custom mathtext fontset so Times New Roman is accepted.
    "mathtext.fontset": "custom",
    "mathtext.rm": "Times New Roman",
    "mathtext.it": "Times New Roman:italic",
    "mathtext.bf": "Times New Roman:bold",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
}

PALETTE = {
    "train": "#377eb8",      # Train points
    "val": "#e41a1c",        # Val points
    "curve": "#377eb8",      # Fit (Train)
    "curve_val": "#e41a1c",  # Fit (Val)
}

TRAIN_MARKER = "o"
VAL_MARKER = "o"
# Matplotlib uses area for scatter size; match Figure 1's marker_size=4.2.
SCATTER_SIZE = 4.2**2
TRAIN_ALPHA = 0.55
VAL_ALPHA = 0.55
TRAIN_ZORDER = 3
VAL_ZORDER = 3
MARKER_EDGECOLOR = "white"
MARKER_EDGE_LINEWIDTH = 0.3

CURVE_LINEWIDTH = 2.5
CURVE_ALPHA = 0.95

X_LABEL_FONTSIZE = 13
Y_LABEL_FONTSIZE = 13

TICK_LABELSIZE = 10
TICK_LENGTH = 4.0
TICK_WIDTH = 0.8
TICK_MINOR_LENGTH = 2.5
TICK_MINOR_WIDTH = 0.6
TICK_DIRECTION = "out"

SPINE_LINEWIDTH = 0.8
SPINE_COLOR = "#4d4d4d"
SPINE_ALPHA = 1.0

BADGE_FONTSIZE = 13
BADGE_WEIGHT = "bold"
BADGE_COLOR = "#000000"
BADGE_BOX_FACE = "white"
BADGE_BOX_ALPHA = 0.8
BADGE_BOX_EDGE = "none"
BADGE_BOX_LINEWIDTH = 0.0
BADGE_BOXSTYLE = "round,pad=0.3"

GRID_MAJOR_LINESTYLE = ":"
GRID_MAJOR_LINEWIDTH = 0.6
GRID_MAJOR_COLOR = "gray"
GRID_MAJOR_ALPHA = 0.2

GRID_MINOR_LINESTYLE = ":"
GRID_MINOR_LINEWIDTH = 0.6
GRID_MINOR_COLOR = "#dcdcdc"
GRID_MINOR_ALPHA = 0.6

LEGEND_FONTSIZE = 9

PANEL_WSPACE = 0.18

# Subplot spacing (Figure 1-inspired)
SUBPLOTS_ADJUST = {
    "left": 0.10,
    "right": 0.985,
    "bottom": 0.2,
    "top": 0.90,
    "wspace": PANEL_WSPACE,
}

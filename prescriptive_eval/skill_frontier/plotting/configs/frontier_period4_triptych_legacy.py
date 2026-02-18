from __future__ import annotations

# Legacy period4 single_k triptych style used for outputs_old reproduction.
# This matches the historical restyle settings in scripts/plot/restyle.

FIGSIZE = (7.0, 2.1)

RCPARAMS = {
    "font.family": "serif",
    "font.sans-serif": ["Times New Roman"],
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
    "mathtext.fontset": "cm",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
}

PALETTE = {
    "train": "#377eb8",
    "val": "#e41a1c",
    "curve": "#377eb8",
    "curve_val": "#e41a1c",
}

TRAIN_MARKER = "o"
VAL_MARKER = "o"
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
SUBPLOTS_ADJUST = {
    "left": 0.10,
    "right": 0.985,
    "bottom": 0.24,
    "top": 0.90,
    "wspace": PANEL_WSPACE,
}

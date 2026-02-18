from __future__ import annotations

# Period4 overlay panels (old/train vs new/val): used by
# - scripts/plot/plot_period4_frontiers_old_vs_new.py

# Base width/height matches the period4 triptych; width scales linearly with number of panels.
BASE_FIGSIZE = (14.5, 4.2)

PALETTE = {"train": "firebrick", "val": "#1f77b4", "curve": "firebrick", "curve_val": "#1f77b4"}
EXTRA_COLOR = "#A6D96A"
EXTRA_FAMILY_ORDER = ["Olmo", "Hermes", "Nemotron", "OpenThinker", "Others"]
EXTRA_FAMILY_COLORS = {
    "Olmo": "#A6D96A",
    "Hermes": "#7570b3",
    "Nemotron": "#e66101",
    "OpenThinker": "#1b9e77",
    "Others": "#bdbdbd",
}

TRAIN_MARKER = "o"
VAL_MARKER = "^"
SCATTER_SIZE = 45
TRAIN_ALPHA = 0.12
VAL_ALPHA = 0.35
TRAIN_ZORDER = 2
VAL_ZORDER = 3
VAL_EDGE_LINEWIDTH = 0.9

CURVE_LINEWIDTH = 2.6
CURVE_ALPHA = 0.95

X_LABEL_FONTSIZE = 24
Y_LABEL_FONTSIZE = 20

TICK_LABELSIZE = 14
TICK_LENGTH = 4.5
TICK_WIDTH = 0.9
TICK_DIRECTION = "out"

SPINE_LINEWIDTH = 1.2
SPINE_COLOR = "#6e6e6e"
SPINE_ALPHA = 0.95

BADGE_FONTSIZE = 24
BADGE_WEIGHT = "semibold"
BADGE_COLOR = "#333"
BADGE_BOX_FACE = "white"
BADGE_BOX_ALPHA = 0.4
BADGE_BOX_EDGE = "#6e6e6e"
BADGE_BOX_LINEWIDTH = 0.8

GRID_MAJOR_LINESTYLE = "-"
GRID_MAJOR_LINEWIDTH = 0.9
GRID_MAJOR_COLOR = "#cfcfcf"
GRID_MAJOR_ALPHA = 0.6

GRID_MINOR_LINESTYLE = ":"
GRID_MINOR_LINEWIDTH = 0.6
GRID_MINOR_COLOR = "#dcdcdc"
GRID_MINOR_ALPHA = 0.6

LEGEND_LOC = "lower right"
LEGEND_FONTSIZE = 15
LEGEND_FRAMEALPHA = 0.2
LEGEND_FANCYBOX = True
LEGEND_BORDERPAD = 0.6

PANEL_WSPACE = 0.25

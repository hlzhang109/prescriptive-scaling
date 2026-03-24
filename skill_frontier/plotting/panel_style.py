"""Shared panel styling helpers for paper figures."""

from __future__ import annotations

from typing import Dict


def apply_panel_style(ax, style: Dict[str, float]) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(style["spine_width"])
    ax.spines["bottom"].set_linewidth(style["spine_width"])
    ax.grid(
        True,
        alpha=style["grid_alpha"],
        linestyle=style["grid_linestyle"],
        linewidth=style["grid_linewidth"],
        zorder=0,
    )
    ax.tick_params(axis="both", which="major", length=4, width=0.8, direction="out")
    ax.tick_params(axis="both", which="minor", length=0)

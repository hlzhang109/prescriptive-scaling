#!/usr/bin/env python3
"""
Recreate Figure 4 (main paper): pretraining vs post-training frontiers + monotonicity.

Vertical layout (1 column × 3 rows):
  (a) MMLU-PRO: all points + official pretrained + pretrained/post-trained boundaries
  (b) MATH Lvl 5: all points + official pretrained + pretrained/post-trained boundaries
  (c) Monotonicity: horizontal grouped bar chart (adjacent decrease rate)

Inputs:
  outputs/sigmoid/no_split/pretrain_vs_posttrain/
    - fits/*_{pre,post}.json
    - tables/lift_{TASK}.csv
    - tables/pretrain_posttrain_heterogeneity_summary.csv

Outputs:
  outputs/figures_main_paper/figure4_refined.pdf (+ .png)
  outputs/figures_main_paper/figure4_main_paper.pdf (+ figure4_main_paper_preview.png)

Usage:
  python scripts/plot/figure4_main_paper/plot_fig4.py
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Optional, Sequence, Tuple

import numpy as np
import matplotlib as mpl

try:
    import pandas as pd  # type: ignore
except Exception as e:  # pragma: no cover
    raise SystemExit("This script requires pandas.") from e

try:
    import matplotlib.pyplot as plt  # type: ignore
except Exception as e:  # pragma: no cover
    raise SystemExit("This script requires matplotlib.") from e

# Import bootstrap from `scripts/` by putting the scripts dir on sys.path.
SCRIPTS_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, os.pardir)
)
if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)
from _bootstrap import ensure_repo_root_on_path  # type: ignore

REPO_ROOT = ensure_repo_root_on_path(__file__)


# -----------------------------------------------------------------------------
# CONFIG (publication-style single-column figure)
# -----------------------------------------------------------------------------

FIG_WIDTH = 7.4
FIG_HEIGHT = 1.92
DPI = 300

XLIM_FLOPS = (1e21, 1e25)
XTICKS_FLOPS = [1e21, 1e22, 1e23, 1e24, 1e25]

COLORS = {
    "points": "#999999",  # gray for all points
    "pretrained_points": "#2ca02c",  # green for official pretrained
    "smooth_98_posttrain": "#d62728",  # red (solid)
    "smooth_98_pretrain": "#1f77b4",  # blue (dashed)
    "pretrained_bars": "#6495ED",  # cornflower blue
    "posttrained_bars": "#DC143C",  # crimson
}

STYLE = {
    "line_width": 2.3,
    "marker_size": 2.8,
    "marker_alpha": 0.35,
    "marker_alpha_pretrained": 0.75,
    "marker_edgecolor": "white",
    "marker_edgewidth": 0.3,
    "marker_edgewidth_pretrained": 0.3,
    "line_alpha": 0.95,
    "grid_alpha": 0.15,
    "grid_linestyle": ":",
    "grid_linewidth": 0.4,
    "grid_color": "#cccccc",
    "spine_width": 0.9,
    "spine_color": "#000000",
    "bar_height": 0.38,
}

BASE_DIR = os.path.join(REPO_ROOT, "outputs", "sigmoid", "no_split", "pretrain_vs_posttrain")
FITS_DIR = os.path.join(BASE_DIR, "fits")
TABLES_DIR = os.path.join(BASE_DIR, "tables")
OUT_DIR = os.path.join(REPO_ROOT, "outputs", "figures_main_paper")

OLL_CSV = os.path.join(REPO_ROOT, "tables", "open_llm_leaderboard", "open_llm_leaderboard_with_tokens.csv")

TASKS_SCATTER = (
    ("MMLU-PRO Raw", "(a)", "MMLU-Pro"),
    ("MATH Lvl 5 Raw", "(b)", "Math Lvl 5"),
)

BAR_SUMMARY_CSV = os.path.join(TABLES_DIR, "pretrain_posttrain_heterogeneity_summary.csv")
BAR_TASK_LABELS = {
    "BBH Raw": "BBH",
    "GPQA Raw": "GPQA",
    "IFEval Raw": "IFEval",
    "MATH Lvl 5 Raw": "MATH Lvl 5",
    "MMLU-PRO Raw": "MMLU-PRO",
    "MUSR Raw": "MUSR",
}


def _configure_rcparams() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": DPI,
            "savefig.dpi": DPI,
            "font.family": "serif",
            "font.serif": ["Times New Roman"],
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "font.size": 10,
            "axes.labelsize": 11,
            "axes.titlesize": 12,
            "axes.labelweight": "bold",
            "xtick.labelsize": 9.5,
            "ytick.labelsize": 9.5,
            "legend.fontsize": 8,
            "figure.titlesize": 12,
            "axes.linewidth": 0.9,
            "grid.linewidth": STYLE["grid_linewidth"],
            "lines.linewidth": STYLE["line_width"],
            "patch.linewidth": 0.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "mathtext.fontset": "cm",
        }
    )


def _sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, float)
    out = np.empty_like(x, dtype=float)
    pos = x >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    expx = np.exp(x[~pos])
    out[~pos] = expx / (1.0 + expx)
    return out


def _sigmoid_pred(params: np.ndarray, z_scaled: np.ndarray) -> np.ndarray:
    """Match scripts/run/sigmoid_quantile_optimizer.py::sigmoid_pred."""
    y0, L, z_star, log_b = [float(v) for v in np.asarray(params, float).tolist()]
    b = float(np.exp(log_b))
    return y0 + L * _sigmoid(b * (np.asarray(z_scaled, float) - float(z_star)))


def _task_slug(task: str) -> str:
    return str(task).replace(" ", "_")


def _load_fit_json(task: str, subset: str) -> dict:
    path = os.path.join(FITS_DIR, f"{_task_slug(task)}_{subset}.json")
    with open(path, "r") as f:
        return json.load(f)


def _sample_curve_from_fit(fit: dict, *, grid_points: int = 400) -> Tuple[np.ndarray, np.ndarray]:
    if str(fit.get("status", "")).lower() != "ok":
        return np.array([], dtype=float), np.array([], dtype=float)

    params = np.asarray(fit.get("params", []), dtype=float)
    if params.size != 4 or not np.all(np.isfinite(params)):
        return np.array([], dtype=float), np.array([], dtype=float)

    z_min = float(fit.get("z_scaled_min", np.nan))
    z_max = float(fit.get("z_scaled_max", np.nan))
    if not (np.isfinite(z_min) and np.isfinite(z_max) and z_max > z_min):
        return np.array([], dtype=float), np.array([], dtype=float)

    x_tick_multiplier = float(fit.get("x_tick_multiplier", 1.0))

    z_grid = np.linspace(z_min, z_max, num=int(grid_points))
    y_hat = _sigmoid_pred(params, z_grid)
    y_hat = np.clip(y_hat, 0.0, 1.0)

    # The fit operates over z_scaled = log10(compute_scaled); x_scaled is in the same units.
    x_scaled = 10.0 ** z_grid
    x_flops = x_scaled * x_tick_multiplier
    return x_flops.astype(float), y_hat.astype(float)


def _load_lift_table(task: str) -> pd.DataFrame:
    path = os.path.join(TABLES_DIR, f"lift_{_task_slug(task)}.csv")
    df = pd.read_csv(path)
    return df


from skill_frontier.io.boolean_utils import is_true  # type: ignore


def style_axes(ax) -> None:
    """Apply unified styling to any axis."""
    ax.set_facecolor("white")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(STYLE["spine_width"])
    ax.spines["bottom"].set_linewidth(STYLE["spine_width"])
    ax.spines["left"].set_color(STYLE["spine_color"])
    ax.spines["bottom"].set_color(STYLE["spine_color"])

    ax.tick_params(
        axis="both",
        which="major",
        length=4,
        width=0.9,
        direction="out",
        colors=STYLE["spine_color"],
    )
    ax.tick_params(axis="both", which="minor", length=0)
    ax.minorticks_off()

    ax.grid(
        True,
        alpha=STYLE["grid_alpha"],
        linestyle=STYLE["grid_linestyle"],
        linewidth=STYLE["grid_linewidth"],
        color=STYLE["grid_color"],
        zorder=0,
        which="major",
    )
    ax.grid(False, which="minor")


def _add_panel_label(ax, label: str, title: str, *, label_y: float = 1.05) -> None:
    return ax.text(
        0.5,
        label_y,
        f"{label} {title}",
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=11,
        fontweight="bold",
        color="black",
        bbox=dict(
            boxstyle="round,pad=0.35",
            facecolor="#f0f0f0",
            edgecolor="#cccccc",
            linewidth=0.8,
            alpha=0.95,
        ),
        clip_on=False,
        zorder=10,
    )

def _get_text_bbox_axes(ax, text, renderer) -> mpl.transforms.Bbox:
    bbox_disp = text.get_window_extent(renderer=renderer)
    (x0, y0), (x1, y1) = ax.transAxes.inverted().transform(bbox_disp.get_points())
    return mpl.transforms.Bbox.from_extents(float(x0), float(y0), float(x1), float(y1))


def _get_textbox_bbox_axes(ax, text, renderer) -> mpl.transforms.Bbox:
    patch = getattr(text, "get_bbox_patch", lambda: None)()
    if patch is not None:
        bbox_disp = patch.get_window_extent(renderer=renderer)
    else:
        bbox_disp = text.get_window_extent(renderer=renderer)
    (x0, y0), (x1, y1) = ax.transAxes.inverted().transform(bbox_disp.get_points())
    return mpl.transforms.Bbox.from_extents(float(x0), float(y0), float(x1), float(y1))


def _get_artist_bbox_axes(ax, artist, renderer) -> mpl.transforms.Bbox:
    bbox_disp = artist.get_window_extent(renderer=renderer)
    (x0, y0), (x1, y1) = ax.transAxes.inverted().transform(bbox_disp.get_points())
    return mpl.transforms.Bbox.from_extents(float(x0), float(y0), float(x1), float(y1))


def _center_title_and_callout(
    ax,
    *,
    title_text,
    callout_text,
    pad: float = 0.02,
    center_x: float = 0.5,
    callout_dx: float = 0.0,
) -> None:
    """Place callout to the right of the title and center them jointly."""
    fig = ax.figure
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()

    title_bbox = _get_text_bbox_axes(ax, title_text, renderer)
    x = float(title_bbox.x1) + float(pad)
    y = float(callout_text.get_position()[1])

    callout_text.set_ha("left")
    callout_text.set_va("bottom")
    callout_text.set_position((x, y))

    # Center the union of title + callout at `center_x`.
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    title_bbox = _get_text_bbox_axes(ax, title_text, renderer)
    callout_bbox = _get_text_bbox_axes(ax, callout_text, renderer)
    group_bbox = mpl.transforms.Bbox.union([title_bbox, callout_bbox])
    group_center = float((group_bbox.x0 + group_bbox.x1) / 2.0)
    delta = float(center_x - group_center)
    if abs(delta) > 1e-6:
        tx, ty = title_text.get_position()
        cx, cy = callout_text.get_position()
        title_text.set_position((float(tx) + delta, float(ty)))
        callout_text.set_position((float(cx) + delta, float(cy)))

    if callout_dx:
        cx, cy = callout_text.get_position()
        callout_text.set_position((float(cx) + float(callout_dx), float(cy)))


def _center_title_and_legend(
    ax,
    *,
    title_text,
    legend,
    pad: float = 0.02,
    center_x: float = 0.5,
    legend_dx: float = 0.0,
    legend_dy: float = 0.0,
) -> None:
    """Place legend to the right of the title and center them jointly."""
    fig = ax.figure
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()

    title_bbox = _get_textbox_bbox_axes(ax, title_text, renderer)
    x = float(title_bbox.x1) + float(pad)
    # Match the visual baseline of the title/callout text boxes: align the *box bottoms*.
    # `legend_dy` is in axis-fraction units; positive moves the legend downward.
    y = float(title_bbox.y0) - float(legend_dy)
    legend.set_bbox_to_anchor((x, y), transform=ax.transAxes)

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    title_bbox = _get_textbox_bbox_axes(ax, title_text, renderer)
    legend_bbox = _get_artist_bbox_axes(ax, legend, renderer)
    group_bbox = mpl.transforms.Bbox.union([title_bbox, legend_bbox])
    group_center = float((group_bbox.x0 + group_bbox.x1) / 2.0)
    delta = float(center_x - group_center)
    tx, ty = title_text.get_position()
    if abs(delta) > 1e-6:
        title_text.set_position((float(tx) + delta, float(ty)))
    legend.set_bbox_to_anchor((x + delta + float(legend_dx), y), transform=ax.transAxes)


def _load_all_points_df() -> pd.DataFrame:
    usecols = [
        "Pretraining tokens (T)",
        "#Params (B)",
        "Type",
        "Official Providers",
        TASKS_SCATTER[0][0],
        TASKS_SCATTER[1][0],
    ]
    df = pd.read_csv(OLL_CSV, usecols=usecols)

    tokens = pd.to_numeric(df["Pretraining tokens (T)"], errors="coerce")
    params = pd.to_numeric(df["#Params (B)"], errors="coerce")
    compute_scaled = 6.0 * tokens * params
    df["compute_flops"] = compute_scaled * 1e21
    df["is_official_pretrained"] = (
        df["Type"].astype(str).str.lower().str.contains("pretrained", na=False)
        & df["Official Providers"].apply(is_true)
    )
    return df


def _maybe_scale_scores(y: np.ndarray) -> np.ndarray:
    y = np.asarray(y, float)
    if y.size and np.nanmax(y) > 1.0:
        return y / 100.0
    return y


def _get_points_for_task(df: pd.DataFrame, task: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x = pd.to_numeric(df.get("compute_flops", np.nan), errors="coerce").to_numpy(dtype=float)
    y = pd.to_numeric(df.get(task, np.nan), errors="coerce").to_numpy(dtype=float)
    y = _maybe_scale_scores(y)
    keep = np.isfinite(x) & (x > 0.0) & np.isfinite(y)
    x_all = x[keep]
    y_all = y[keep]

    is_pre = df.get("is_official_pretrained", False)
    is_pre = np.asarray(is_pre, dtype=bool)
    keep_pre = keep & is_pre
    x_pre = x[keep_pre]
    y_pre = y[keep_pre]
    return x_all, y_all, x_pre, y_pre


def _configure_scatter_axes(ax, *, show_xlabel: bool) -> None:
    ax.set_xscale("log")
    ax.set_xlim(XLIM_FLOPS)
    ax.set_ylim(0.0, 0.68)
    yticks_shared = [0.0, 0.2, 0.4, 0.6]
    ax.set_yticks(yticks_shared)
    ax.set_yticklabels([f"{y:.1f}" for y in yticks_shared])
    ax.set_xticks(XTICKS_FLOPS)
    ax.set_xticklabels([f"$10^{{{int(np.log10(x))}}}$" for x in XTICKS_FLOPS])
    ax.set_xlabel(
        "Pretraining Compute (FLOPs)" if show_xlabel else "",
        fontsize=10,
        fontweight="bold",
        labelpad=4,
    )
    ax.set_ylabel("Accuracy", fontsize=10, fontweight="bold", labelpad=4)
    ax.tick_params(axis="both", which="minor", length=0)
    ax.minorticks_off()


def _create_scatter_legend_handles() -> Tuple[list, list]:
    handles: list = []
    labels: list = []

    handles.append(
        mpl.lines.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=COLORS["points"],
            markersize=5.5,
            alpha=STYLE["marker_alpha"],
            linestyle="None",
        )
    )
    labels.append("All")

    handles.append(
        mpl.lines.Line2D(
            [0],
            [0],
            marker="^",
            color="w",
            markerfacecolor=COLORS["pretrained_points"],
            markersize=6.5,
            alpha=STYLE["marker_alpha_pretrained"],
            linestyle="None",
            markeredgecolor=STYLE["marker_edgecolor"],
            markeredgewidth=STYLE["marker_edgewidth_pretrained"],
        )
    )
    labels.append("Pretrained")

    handles.append(
        mpl.lines.Line2D(
            [0],
            [0],
            color=COLORS["smooth_98_posttrain"],
            linewidth=STYLE["line_width"],
            linestyle="-",
        )
    )
    labels.append("Post-trained")

    handles.append(
        mpl.lines.Line2D(
            [0],
            [0],
            color=COLORS["smooth_98_pretrain"],
            linewidth=STYLE["line_width"],
            linestyle="--",
            dashes=(5, 3),
        )
    )
    labels.append("Pretrained")

    return handles, labels


def _plot_monotonicity_bars(ax) -> None:
    df = pd.read_csv(BAR_SUMMARY_CSV)
    df = df.dropna(subset=["task", "adjdec_pre", "adjdec_post"]).copy()

    tasks = df["task"].astype(str).tolist()
    pretrained_rates = pd.to_numeric(df["adjdec_pre"], errors="coerce").to_numpy(dtype=float)
    posttrained_rates = pd.to_numeric(df["adjdec_post"], errors="coerce").to_numpy(dtype=float)

    order = np.argsort(pretrained_rates)[::-1]
    tasks_sorted = [tasks[i] for i in order]
    pretrained_sorted = [float(pretrained_rates[i]) for i in order]
    posttrained_sorted = [float(posttrained_rates[i]) for i in order]

    y_pos = np.arange(len(tasks_sorted))
    bar_height = float(STYLE.get("bar_height", 0.38))

    ax.barh(
        y_pos - bar_height / 2,
        pretrained_sorted,
        height=bar_height,
        color=COLORS["pretrained_bars"],
        alpha=0.85,
        label="Pretrained",
        edgecolor="black",
        linewidth=0.6,
        zorder=2,
    )
    ax.barh(
        y_pos + bar_height / 2,
        posttrained_sorted,
        height=bar_height,
        color=COLORS["posttrained_bars"],
        alpha=0.85,
        label="Post-trained",
        edgecolor="black",
        linewidth=0.6,
        zorder=3,
    )

    ax.set_yticks(y_pos)
    ax.set_yticklabels([BAR_TASK_LABELS.get(t, t.replace(" Raw", "")) for t in tasks_sorted], fontsize=9.0)
    for tick in ax.get_yticklabels():
        tick.set_fontweight("bold")
    # Pull tick labels closer to the axis to avoid inter-panel overlap.
    ax.tick_params(axis="y", which="major", pad=0)

    xlabel = ax.set_xlabel("Adjacent Decrease Rate", fontsize=10, fontweight="bold", labelpad=4)
    # Right-align to prevent truncation in tight exports.
    xlabel.set_horizontalalignment("right")
    xlabel.set_x(0.98)

    max_rate = float(np.nanmax([np.nanmax(pretrained_rates), np.nanmax(posttrained_rates)]))
    ax.set_xlim(0.0, max_rate * 1.30 if np.isfinite(max_rate) else 1.0)
    ax.invert_yaxis()
    ax.axvline(x=0.0, color="black", linewidth=1.0, zorder=2)
    # Keep y-ticks uniformly spaced across the full panel height (no extra headroom).
    ax.set_ylim(len(tasks_sorted) - 0.5, -0.5)

    ax.grid(
        axis="x",
        alpha=STYLE["grid_alpha"],
        linestyle=STYLE["grid_linestyle"],
        linewidth=STYLE["grid_linewidth"],
        color=STYLE["grid_color"],
        zorder=0,
    )
    ax.grid(axis="y", visible=False)

    # Value labels on bars
    offset = 0.012
    for i, (pre_val, post_val) in enumerate(zip(pretrained_sorted, posttrained_sorted)):
        ax.text(
            float(pre_val) + offset,
            float(y_pos[i] - bar_height / 2),
            f"{float(pre_val):.2f}",
            va="center",
            ha="left",
            fontsize=7.5,
            fontweight="normal",
            color="#1f4788",
            zorder=4,
        )
        ax.text(
            float(post_val) + offset,
            float(y_pos[i] + bar_height / 2),
            f"{float(post_val):.2f}",
            va="center",
            ha="left",
            fontsize=7.5,
            fontweight="normal",
            color="#b22222",
            zorder=4,
        )
    # Limits above provide the desired headroom; avoid extra autoscale padding.


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Recreate Figure 4 (refined, horizontal 1×3).")
    parser.add_argument(
        "--out",
        default=os.path.join(OUT_DIR, "figure4_refined.pdf"),
        help="Output PDF path (refined).",
    )
    args = parser.parse_args(argv)

    _configure_rcparams()

    fig, axes = plt.subplots(1, 3, figsize=(FIG_WIDTH, FIG_HEIGHT), dpi=DPI)
    # Keep figsize fixed, but shrink the subplot height to make room for the boxed titles above each panel.
    plt.subplots_adjust(left=0.08, right=0.955, bottom=0.18, top=0.82, wspace=0.52)

    for ax in axes:
        style_axes(ax)

    df_points = _load_all_points_df()

    # Panels (a) and (b): scatter + curves
    title_texts = []
    for ax, (task, panel_label, panel_title) in zip(axes[:2], TASKS_SCATTER):
        all_compute, all_acc, pre_compute, pre_acc = _get_points_for_task(df_points, task)

        ax.scatter(
            all_compute,
            all_acc,
            s=STYLE["marker_size"] ** 2,
            alpha=STYLE["marker_alpha"],
            color=COLORS["points"],
            edgecolors=STYLE["marker_edgecolor"],
            linewidths=STYLE["marker_edgewidth"],
            label="All",
            zorder=2,
            rasterized=True,
        )
        ax.scatter(
            pre_compute,
            pre_acc,
            s=(STYLE["marker_size"] * 1.5) ** 2,
            alpha=STYLE["marker_alpha_pretrained"],
            color=COLORS["pretrained_points"],
            edgecolors=STYLE["marker_edgecolor"],
            linewidths=STYLE["marker_edgewidth_pretrained"],
            label="Pretrained",
            marker="^",
            zorder=3,
            rasterized=True,
        )

        fit_post = _load_fit_json(task, "post")
        fit_pre = _load_fit_json(task, "pre")
        x_post, y_post = _sample_curve_from_fit(fit_post)
        x_pre, y_pre = _sample_curve_from_fit(fit_pre)

        if x_post.size:
            ax.plot(
                x_post,
                y_post,
                linewidth=STYLE["line_width"],
                alpha=STYLE["line_alpha"],
                color=COLORS["smooth_98_posttrain"],
                linestyle="-",
                label=r"Post-trained $\tau$ = 0.98",
                zorder=4,
            )
        if x_pre.size:
            ax.plot(
                x_pre,
                y_pre,
                linewidth=STYLE["line_width"],
                alpha=STYLE["line_alpha"],
                color=COLORS["smooth_98_pretrain"],
                linestyle="--",
                dashes=(5, 3),
                label=r"Pretrained $\tau$ = 0.98",
                zorder=4,
            )

        _configure_scatter_axes(ax, show_xlabel=True)
        ax.set_title("")
        title_texts.append(_add_panel_label(ax, panel_label, panel_title))

        handles, labels = _create_scatter_legend_handles()
        legend = ax.legend(
            handles,
            labels,
            loc="upper left",
            bbox_to_anchor=(0.02, 0.98),
            frameon=True,
            framealpha=0.75,
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

    # Subtle annotations: key insights
    callout_texts = []
    callout_texts.append(
        axes[0].text(
        0.5,
        1.05,
        "Marginal gain\nfrom post-training",
        transform=axes[0].transAxes,
        ha="left",
        va="bottom",
        fontsize=7.5,
        style="italic",
        color="#444444",
        bbox=dict(
            boxstyle="round,pad=0.4",
            facecolor="#fffacd",
            edgecolor="#999999",
            linewidth=0.5,
            alpha=0.92,
        ),
        clip_on=False,
        zorder=10,
        )
    )
    callout_texts.append(
        axes[1].text(
        0.5,
        1.05,
        "Significant gain\nfrom post-training",
        transform=axes[1].transAxes,
        ha="left",
        va="bottom",
        fontsize=7.5,
        style="italic",
        color="#444444",
        bbox=dict(
            boxstyle="round,pad=0.4",
            facecolor="#fffacd",
            edgecolor="#999999",
            linewidth=0.5,
            alpha=0.92,
        ),
        clip_on=False,
        zorder=10,
        )
    )
    for ax, title_text, callout_text in zip(axes[:2], title_texts, callout_texts):
        _center_title_and_callout(
            ax,
            title_text=title_text,
            callout_text=callout_text,
            pad=0.02,
            callout_dx=0.06,
        )

    # Panel (c): monotonicity horizontal bars
    _plot_monotonicity_bars(axes[2])
    # Shift panel (c) slightly rightward.
    pos = axes[2].get_position()
    axes[2].set_position([pos.x0 + 0.015, pos.y0, pos.width, pos.height])

    # Panel labels for (c) and remove title-based labels.
    axes[2].set_title("")
    title_text_c = _add_panel_label(axes[2], "(c)", "Monotonicity")

    # Legend for panel (c): to the right of the boxed title, matching the annotation style.
    handles, labels = axes[2].get_legend_handles_labels()
    legend_c = axes[2].legend(
        handles,
        labels,
        loc="lower left",
        bbox_to_anchor=(0.0, float(title_text_c.get_position()[1])),
        bbox_transform=axes[2].transAxes,
        frameon=True,
        framealpha=0.95,
        edgecolor="#cccccc",
        fancybox=True,
        fontsize=7.5,
        ncol=1,
        handlelength=1.2,
        handletextpad=0.4,
        labelspacing=0.35,
        borderpad=0.4,
    )
    legend_c.get_frame().set_facecolor("#f0f0f0")
    legend_c.get_frame().set_linewidth(0.8)
    legend_c.set_zorder(10)
    _center_title_and_legend(
        axes[2],
        title_text=title_text_c,
        legend=legend_c,
        pad=0.0,
        legend_dx=-0.003,
        legend_dy=0.045,
    )

    out_pdf = os.path.abspath(str(args.out))
    os.makedirs(os.path.dirname(out_pdf), exist_ok=True)
    fig.savefig(out_pdf, dpi=DPI, bbox_inches="tight", pad_inches=0.08, facecolor="white", edgecolor="none")

    png_path = os.path.splitext(out_pdf)[0] + ".png"
    fig.savefig(png_path, dpi=DPI, bbox_inches="tight", pad_inches=0.08, facecolor="white", edgecolor="none")

    # Also write the canonical main-paper filenames.
    canonical_pdf = os.path.join(OUT_DIR, "figure4_main_paper.pdf")
    canonical_png = os.path.join(OUT_DIR, "figure4_main_paper_preview.png")
    if os.path.abspath(canonical_pdf) != out_pdf:
        fig.savefig(
            canonical_pdf,
            dpi=DPI,
            bbox_inches="tight",
            pad_inches=0.08,
            facecolor="white",
            edgecolor="none",
        )
        fig.savefig(
            canonical_png,
            dpi=DPI,
            bbox_inches="tight",
            pad_inches=0.08,
            facecolor="white",
            edgecolor="none",
        )

    print("Figure 4 saved to outputs/figures_main_paper/")
    print(f"Saved: {out_pdf}")
    print(f"Saved: {png_path}")
    if os.path.abspath(canonical_pdf) != out_pdf:
        print(f"Saved: {canonical_pdf}")
        print(f"Saved: {canonical_png}")
    print(f"Dimensions: {FIG_WIDTH}\" × {FIG_HEIGHT}\"")
    print("Layout: 1 row × 3 columns (horizontal)")
    print("Panel (c): Horizontal bars for compactness")
    plt.close(fig)


if __name__ == "__main__":
    main()

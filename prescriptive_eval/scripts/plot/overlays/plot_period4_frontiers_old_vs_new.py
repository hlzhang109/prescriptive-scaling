#!/usr/bin/env python3
"""
Plot period4/single_k sigmoid frontier overlays using:
  - "old" Open LLM Leaderboard (OLL) models, and
  - "new" newly-evaluated models (metrics from validation_leaderboard.csv, compute from new_eval_leaderboard.csv).

Modes
  - same_period: old Pk vs new Pk for k=1..4  -> outputs/.../new_models/plots/
  - p4_to_p5:    old P4 vs new P5 (after P4)  -> outputs/.../new_models_p5/plots/

The style mirrors `outputs/sigmoid/period4/single_k/plots/*_period4.png`.
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

try:
    import pandas as pd  # type: ignore
except Exception as e:  # pragma: no cover
    raise SystemExit("This script requires pandas.") from e

# Ensure repo root is on sys.path when running as a script.
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, os.pardir))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from skill_frontier.plotting.fig5_data_utils import (  # type: ignore  # noqa: E402
    _assign_period_index,
    _compute_old_compute_zflops,
    _make_period_ids_from_dates,
)
from skill_frontier.io.task_mappings import MAIN_TASK_MAP_OLL_TO_NEW  # type: ignore  # noqa: E402
from skill_frontier.plotting.model_families import (  # type: ignore  # noqa: E402
    FAMILY_ORDER_WITH_GPT,
    color_for_family,
    family_from_base_model,
)
from skill_frontier.plotting.axis_formatting import (  # type: ignore  # noqa: E402
    apply_pretraining_compute_tick_multiplier,
)
from skill_frontier.plotting.configs import frontier_period4_old_vs_new as period4_old_vs_new_cfg  # type: ignore  # noqa: E402
from skill_frontier.plotting.configs import mpl_rc as mpl_rc_cfg  # type: ignore  # noqa: E402
from skill_frontier.plotting.labels import PRETRAINING_COMPUTE_FLOPS_LABEL  # type: ignore  # noqa: E402

# Reuse the same sigmoid fitter + filename conventions used by the existing plots.
from scripts.smooth_single_skill_frontier import (  # type: ignore  # noqa: E402
    _get_plot_path,
    fit_sigmoid_frontier,
)

FAMILY_MARKERS: Dict[str, str] = {
    "Qwen": "o",
    "Llama": "s",
    "Mistral": "^",
    "Gemma": "D",
    "Phi": "P",
    "GPT": "X",
    "Others": "v",
}

# Explicit panel-(d) exclusion used by Figure 5 style p4_to_p5 plots.
PANEL_D_EXCLUDED_MODEL_IDS = {
    "allenai/olmo-3-7b-think-sft",
}

# -----------------------------------------------------------------------------
# Paper style (match Figure 1 main paper exactly)
# -----------------------------------------------------------------------------

FIG1_RCPARAMS = {
    "font.family": "serif",
    "font.serif": ["Times New Roman"],
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
    "font.size": 12,
    "axes.labelsize": 15,
    "axes.titlesize": 15,
    "axes.labelweight": "bold",
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 9,
    "mathtext.fontset": "cm",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
}

FIG1_STYLE = {
    "line_width_fit": 2.5,
    "marker_size": 4.2,  # sqrt(s) => s ~ 17.6
    "marker_alpha": 0.55,
    "marker_edgecolor": "white",
    "marker_edgewidth": 0.3,
    "line_alpha": 0.95,
    "train_points_color": "#377eb8",  # blue
    "val_points_color": "#e41a1c",  # red
    "fit_train_color": "#377eb8",
    "fit_val_color": "#e41a1c",
    "grid_alpha": 0.2,
    "grid_linestyle": ":",
    "grid_linewidth": 0.6,
    "grid_color": "gray",
    "spine_width": 0.8,
    "spine_color": "#4d4d4d",
}

FIG1_LAYOUT = {
    "fig_width": 7.0,
    "fig_height": 4.2,
    "left": 0.10,
    "right": 0.985,
    "bottom": 0.12,
    "top": 0.90,
    "wspace": 0.18,
    "hspace": 0.20,
    "nrows": 2,
    "ncols": 3,
}

# Figure 5 (panels (c)/(d)) style: match outputs/figures_main_paper/figure5_main_paper.png
FIG5_RCPARAMS = {
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "font.family": "serif",
    "font.serif": ["Times New Roman"],
    "font.size": 14,
    "axes.labelsize": 14,
    "axes.titlesize": 14,
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

FIG5_STYLE = {
    "marker_size": 3.5,
    "marker_alpha": 0.6,
    "line_width": 2.0,
    "line_alpha": 0.9,
    "grid_alpha": 0.2,
    "grid_linestyle": ":",
    "grid_linewidth": 0.5,
    "spine_width": 0.8,
}

FIG5_COLORS = {
    "p4_frontier": "#377eb8",  # blue (val)
    "p5_frontier": "#e41a1c",  # red (train)
}

FIG5_LAYOUT = {
    "fig_width": 11.5,
    "fig_height": 2.6,
    "left": 0.06,
    "right": 0.98,
    "bottom": 0.18,
    "top": 0.88,
    "wspace": 0.32,
    "hspace": 0.0,
    "nrows": 1,
    "ncols": 4,
}


def _panel_inches_from_layout(layout: Dict[str, float]) -> Tuple[float, float]:
    """Return (panel_width_in, panel_height_in) for a Matplotlib gridspec-like layout."""
    fig_w = float(layout["fig_width"])
    fig_h = float(layout["fig_height"])
    left = float(layout["left"])
    right = float(layout["right"])
    bottom = float(layout["bottom"])
    top = float(layout["top"])
    wspace = float(layout["wspace"])
    hspace = float(layout["hspace"])
    nrows = int(layout["nrows"])
    ncols = int(layout["ncols"])

    avail_w = max(1e-9, right - left)
    avail_h = max(1e-9, top - bottom)
    denom_w = float(ncols + max(0, ncols - 1) * wspace)
    denom_h = float(nrows + max(0, nrows - 1) * hspace)
    panel_w = fig_w * avail_w / max(1e-9, denom_w)
    panel_h = fig_h * avail_h / max(1e-9, denom_h)
    return float(panel_w), float(panel_h)


def _figsize_for_panels(
    *,
    ncols: int,
    panel_w: float,
    panel_h: float,
    left: float,
    right: float,
    bottom: float,
    top: float,
    wspace: float,
) -> Tuple[float, float]:
    avail_w = max(1e-9, right - left)
    avail_h = max(1e-9, top - bottom)
    denom_w = float(ncols + max(0, ncols - 1) * wspace)
    fig_w = panel_w * denom_w / avail_w
    fig_h = panel_h / avail_h  # 1 row
    return float(fig_w), float(fig_h)


def _apply_fig1_axes_style(ax) -> None:
    ax.set_facecolor("white")
    ax.set_axisbelow(True)

    ax.grid(
        True,
        color=FIG1_STYLE["grid_color"],
        alpha=FIG1_STYLE["grid_alpha"],
        linestyle=FIG1_STYLE["grid_linestyle"],
        linewidth=FIG1_STYLE["grid_linewidth"],
        zorder=0,
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(FIG1_STYLE["spine_width"])
    ax.spines["bottom"].set_linewidth(FIG1_STYLE["spine_width"])
    ax.spines["left"].set_color(FIG1_STYLE["spine_color"])
    ax.spines["bottom"].set_color(FIG1_STYLE["spine_color"])

    ax.tick_params(
        axis="both",
        which="major",
        length=4,
        width=0.8,
        direction="out",
        color=FIG1_STYLE["spine_color"],
    )
    ax.tick_params(
        axis="both",
        which="minor",
        length=2.5,
        width=0.6,
        direction="out",
        color=FIG1_STYLE["spine_color"],
    )


def _save_fig1(fig, *, base_path: str) -> None:
    fig.savefig(
        base_path + ".png",
        dpi=300,
        bbox_inches="tight",
        pad_inches=0.05,
        facecolor="white",
        edgecolor="none",
    )
    fig.savefig(
        base_path + ".pdf",
        dpi=300,
        bbox_inches="tight",
        pad_inches=0.05,
        facecolor="white",
        edgecolor="none",
    )


def _apply_fig5_axes_style(ax) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(FIG5_STYLE["spine_width"])
    ax.spines["bottom"].set_linewidth(FIG5_STYLE["spine_width"])

    ax.grid(
        True,
        alpha=FIG5_STYLE["grid_alpha"],
        linestyle=FIG5_STYLE["grid_linestyle"],
        linewidth=FIG5_STYLE["grid_linewidth"],
        zorder=0,
    )
    ax.tick_params(axis="both", which="major", length=4, width=0.8, direction="out")
    ax.tick_params(axis="both", which="minor", length=0)
    ax.set_facecolor("white")


def _fig5_panel_label(ax, text: str) -> None:
    ax.text(
        0.5,
        1.06,
        text,
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=14,
        fontweight="bold",
        color="black",
        bbox=dict(
            boxstyle="round,pad=0.35",
            facecolor="#f0f0f0",
            edgecolor="#cccccc",
            linewidth=0.8,
            alpha=0.95,
        ),
        zorder=10,
        clip_on=False,
    )


def _save_fig5(fig, *, base_path: str) -> None:
    fig.savefig(
        base_path + ".png",
        dpi=300,
        bbox_inches="tight",
        pad_inches=0.05,
        facecolor="white",
        edgecolor="none",
    )
    fig.savefig(
        base_path + ".pdf",
        dpi=300,
        bbox_inches="tight",
        pad_inches=0.05,
        facecolor="white",
        edgecolor="none",
    )


def _extra_family_from_model_id(model_id: str) -> str:
    s = str(model_id).strip().lower()
    if not s or s == "nan":
        return "Others"
    if "openthinker" in s or "open-thoughts" in s or "open_thoughts" in s:
        return "OpenThinker"
    if "nemotron" in s or s.startswith("nvidia/") or "nvidia" in s:
        return "Nemotron"
    if "hermes" in s:
        return "Hermes"
    if "olmo" in s:
        return "Olmo"
    return "Others"


def _period_base_membership_key(base_model: str) -> str:
    """Normalize base-model IDs for P1–P4 vs P5 membership checks."""
    s = str(base_model).strip().lower()
    if not s or s == "nan":
        return ""
    # Treat major versions as a family for period membership (per docs/fix.txt).
    if "llama-3" in s or "llama3" in s:
        return "llama-3"
    if "qwen3" in s:
        return "qwen3"
    return s


def _p5_family_from_base_model(base_model: str) -> str:
    """Family labels for P5-only base models in the right subplot."""
    s = str(base_model).strip().lower()
    if not s or s == "nan":
        return "Others"
    if "qwen3" in s:
        return "Qwen3"
    if "gpt-oss" in s:
        return "GPT-OSS"
    if "gemma-3" in s:
        return "Gemma-3"
    if "smollm3" in s:
        return "SmolLM3"
    if "llama-3" in s or "llama3" in s:
        return "Llama-3"
    return "Others"


def _load_old_oll(
    csv_path: str,
    tasks: Sequence[str],
) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    out = pd.DataFrame()
    out["compute_zflops"] = _compute_old_compute_zflops(df)
    out["pid"] = _make_period_ids_from_dates(
        df.get("Upload To Hub Date", pd.Series([np.nan] * len(df))),
        after_p4_as_p5=False,
    )
    base = df.get("Base model family")
    if base is None:
        base = df.get("Identified base model")
    if base is None:
        base = df.get("Base Model")
    if base is None:
        base = df.get("Model")
    out["base_model"] = base if base is not None else np.nan
    for t in tasks:
        out[t] = pd.to_numeric(df.get(t, np.nan), errors="coerce")
    return out


def _load_new_eval(
    metrics_csv: str,
    compute_csv: str,
    top_models_csv: str,
    tasks: Sequence[str],
) -> pd.DataFrame:
    df_metrics = pd.read_csv(metrics_csv)
    df_compute = pd.read_csv(compute_csv, usecols=["model_id", "pretrain_compute_zflops"])
    df_meta = pd.read_csv(top_models_csv)

    df_new = df_metrics.merge(
        df_compute,
        on="model_id",
        how="left",
        validate="one_to_one",
    )
    df_new = df_new.merge(
        df_meta[["model_id", "last_modified"]],
        on="model_id",
        how="left",
        validate="one_to_one",
    )
    out = pd.DataFrame()
    out["model_id"] = df_new.get("model_id", np.nan)
    out["compute_zflops"] = pd.to_numeric(df_new.get("pretrain_compute_zflops", np.nan), errors="coerce")
    out["pid"] = _make_period_ids_from_dates(
        df_new.get("last_modified", pd.Series([np.nan] * len(df_new))),
        after_p4_as_p5=True,
    )
    out["base_model"] = df_new.get("mapped_base_model", np.nan)
    for oll_name, new_col in MAIN_TASK_MAP_OLL_TO_NEW.items():
        if oll_name in tasks:
            out[oll_name] = pd.to_numeric(df_new.get(new_col, np.nan), errors="coerce")
    return out


def _load_extra_leaderboard(extra_csv: str, tasks: Sequence[str]) -> pd.DataFrame:
    df = pd.read_csv(extra_csv)
    # Drop EvoLM series (and any other entries explicitly requested to be excluded).
    model_id = df.get("model_id", pd.Series([np.nan] * len(df))).astype(str)
    base_id = df.get("mapped_base_model", pd.Series([np.nan] * len(df))).astype(str)
    is_evolm = model_id.str.contains("evolm", case=False, na=False) | base_id.str.contains("evolm", case=False, na=False)
    if is_evolm.any():
        df = df.loc[~is_evolm].copy()

    out = pd.DataFrame()
    out["model_id"] = df.get("model_id", np.nan)
    tokens = pd.to_numeric(df.get("Pretraining tokens (T)", np.nan), errors="coerce")
    params = pd.to_numeric(df.get("#Params (B)", np.nan), errors="coerce")
    out["compute_zflops"] = 6.0 * tokens * params
    out["base_model"] = df.get("mapped_base_model", np.nan)
    for oll_name, col in MAIN_TASK_MAP_OLL_TO_NEW.items():
        if oll_name in tasks:
            out[oll_name] = pd.to_numeric(df.get(col, np.nan), errors="coerce")
    return out


def _load_additional_post_trained_model_ids(csv_path: str) -> List[str]:
    df = pd.read_csv(csv_path, dtype=str)
    raw = [str(v).strip() for v in df.to_numpy().ravel().tolist()]
    out: List[str] = []
    for v in raw:
        if not v or v.lower() == "nan":
            continue
        if v.lower() in ("model name",):
            continue
        if v.lower().startswith("zhenting"):
            continue
        if v not in out:
            out.append(v)
    return out


def _plot_period_panels(
    out_dir: str,
    task: str,
    panels: List[Dict[str, object]],
    scatter_style: str,
    *,
    plots_subdir: str = "plots",
    show_k_badge: bool = True,
    x_label_fontsize: Optional[int] = None,
    legend_loc: Optional[str] = None,
    base_scatter_scale: float = 1.0,
    sharex: bool = False,
    legend_axis_index: int = -1,
    show_extra_family_legend: bool = False,
    supxlabel_y: Optional[float] = None,
) -> None:
    try:
        import matplotlib as mpl  # type: ignore
        import matplotlib.pyplot as plt  # type: ignore
        import matplotlib.ticker as mticker  # type: ignore
        from matplotlib.offsetbox import AnchoredText  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError("matplotlib is required for plotting") from e

    # Match paper-facing styling based on output directory name.
    use_fig1_style = os.path.basename(os.path.normpath(out_dir)) == "new_models"
    use_fig5_style = os.path.basename(os.path.normpath(out_dir)) == "new_models_p5"
    if use_fig1_style:
        try:
            plt.rcParams.update(dict(FIG1_RCPARAMS))
        except Exception:
            pass
    elif use_fig5_style:
        try:
            plt.rcParams.update(dict(FIG5_RCPARAMS))
        except Exception:
            pass
    else:
        try:
            mpl.rcParams["font.family"] = mpl_rc_cfg.FONT_FAMILY
        except Exception:
            pass

    n = len(panels)
    if n == 0:
        return

    if use_fig5_style:
        # ---------------------------------------------------------------------
        # Figure 5-style rendering for p4_to_p5 plots:
        # match panels (c) "New Models" and (d) "New Families".
        # ---------------------------------------------------------------------
        if n != 2:
            raise RuntimeError(f"Expected 2 panels for new_models_p5, got {n}")

        from matplotlib.lines import Line2D  # type: ignore

        ref_panel_w, ref_panel_h = _panel_inches_from_layout(FIG5_LAYOUT)
        left = float(FIG5_LAYOUT["left"])
        right = float(FIG5_LAYOUT["right"])
        bottom = float(FIG5_LAYOUT["bottom"])
        top = float(FIG5_LAYOUT["top"])
        wspace = float(FIG5_LAYOUT["wspace"])

        fig_w, fig_h = _figsize_for_panels(
            ncols=n,
            panel_w=ref_panel_w,
            panel_h=ref_panel_h,
            left=left,
            right=right,
            bottom=bottom,
            top=top,
            wspace=wspace,
        )

        fig, axes = plt.subplots(1, n, figsize=(fig_w, fig_h), dpi=300, sharey=True)
        fig.patch.set_facecolor("white")
        axes = np.atleast_1d(axes)
        fig.subplots_adjust(left=left, right=right, bottom=bottom, top=top, wspace=wspace)

        # Shared y-limits (Figure 5 uses y-min anchored at 0 for these panels).
        y_vals: List[float] = []
        for p in panels:
            for key in ("train_y", "val_y", "extra_y", "curve_y", "curve_val_y"):
                arr = p.get(key)
                if arr is not None and getattr(arr, "size", 0):
                    y_vals.append(float(np.nanmin(arr)))
                    y_vals.append(float(np.nanmax(arr)))
        ymin, ymax = 0.0, 0.9
        if y_vals:
            dmin = float(np.nanmin(y_vals))
            dmax = float(np.nanmax(y_vals))
            if np.isfinite(dmin) and np.isfinite(dmax):
                ymin = min(ymin, dmin)
                ymax = max(ymax, dmax)
        pad = 0.03 * max(1e-6, (ymax - ymin))
        y_lo = max(0.0, ymin - pad)
        y_hi = min(1.0, ymax + pad)

        x_scale = 1e21  # panel dict stores zFLOPs; Figure 5 plots FLOPs
        xlim = (1e21, 1e25)
        xticks = [1e21, 1e22, 1e23, 1e24, 1e25]
        xticklabels = [f"$10^{{{int(np.log10(x))}}}$" for x in xticks]

        # Left panel: (c) New Models (old P4 vs P5 models with seen base families).
        ax = axes[0]
        _apply_fig5_axes_style(ax)
        ax.set_xscale("log")
        ax.set_xlim(*xlim)
        ax.set_xticks(xticks)
        ax.set_xticklabels(xticklabels)
        ax.set_ylim(y_lo, y_hi)
        ax.set_yticks([t for t in (0.0, 0.2, 0.4, 0.6, 0.8) if y_lo - 1e-6 <= t <= y_hi + 1e-6])
        ax.tick_params(axis="y", labelleft=False)
        ax.set_yticklabels([])
        ax.set_xlabel(PRETRAINING_COMPUTE_FLOPS_LABEL, fontsize=12, fontweight="bold", labelpad=4)
        _fig5_panel_label(ax, "(a) New Models")

        train_alpha = 0.20
        val_alpha = 0.35

        p_left = panels[0]
        train_x = p_left.get("train_x")
        train_y = p_left.get("train_y")
        if isinstance(train_x, np.ndarray) and isinstance(train_y, np.ndarray) and train_x.size:
            ax.scatter(
                train_x * x_scale,
                train_y,
                s=FIG5_STYLE["marker_size"] ** 2,
                alpha=train_alpha,
                color=FIG5_COLORS["p5_frontier"],
                edgecolors="none",
                linewidths=0.0,
                zorder=2,
                rasterized=True,
            )

        val_x = p_left.get("val_x")
        val_y = p_left.get("val_y")
        if isinstance(val_x, np.ndarray) and isinstance(val_y, np.ndarray) and val_x.size:
            ax.scatter(
                val_x * x_scale,
                val_y,
                s=FIG5_STYLE["marker_size"] ** 2,
                alpha=val_alpha,
                facecolors="none",
                edgecolors=FIG5_COLORS["p4_frontier"],
                linewidths=0.9,
                marker="^",
                zorder=3,
                rasterized=True,
            )

        curve_x = p_left.get("curve_x")
        curve_y = p_left.get("curve_y")
        if isinstance(curve_x, np.ndarray) and isinstance(curve_y, np.ndarray) and curve_x.size:
            ax.plot(
                curve_x * x_scale,
                curve_y,
                linewidth=FIG5_STYLE["line_width"],
                alpha=FIG5_STYLE["line_alpha"],
                color=FIG5_COLORS["p5_frontier"],
                linestyle="-",
                zorder=4,
            )

        curve_val_x = p_left.get("curve_val_x")
        curve_val_y = p_left.get("curve_val_y")
        if isinstance(curve_val_x, np.ndarray) and isinstance(curve_val_y, np.ndarray) and curve_val_x.size:
            ax.plot(
                curve_val_x * x_scale,
                curve_val_y,
                linewidth=FIG5_STYLE["line_width"],
                alpha=FIG5_STYLE["line_alpha"],
                color=FIG5_COLORS["p4_frontier"],
                linestyle="--",
                dashes=(5, 3),
                zorder=4,
            )

        ax.legend(
            handles=[
                Line2D(
                    [0],
                    [0],
                    marker="o",
                    linestyle="None",
                    markersize=5.5,
                    markerfacecolor=FIG5_COLORS["p5_frontier"],
                    markeredgecolor="none",
                    alpha=train_alpha,
                    label="train",
                ),
                Line2D(
                    [0],
                    [0],
                    marker="^",
                    linestyle="None",
                    markersize=6.5,
                    markerfacecolor="none",
                    markeredgecolor=FIG5_COLORS["p4_frontier"],
                    markeredgewidth=0.9,
                    alpha=val_alpha,
                    label="val",
                ),
                Line2D([0], [0], color=FIG5_COLORS["p5_frontier"], linewidth=FIG5_STYLE["line_width"], label="fit (train)"),
                Line2D(
                    [0],
                    [0],
                    color=FIG5_COLORS["p4_frontier"],
                    linewidth=FIG5_STYLE["line_width"],
                    linestyle="--",
                    dashes=(5, 3),
                    label="fit (val)",
                ),
            ],
            loc="upper left",
            bbox_to_anchor=(0.03, 0.98),
            frameon=True,
            framealpha=0.9,
            edgecolor="gray",
            fancybox=False,
            fontsize=9,
            ncol=1,
            handlelength=1.3,
            handletextpad=0.4,
            labelspacing=0.3,
            borderpad=0.4,
        )

        # Right panel: (d) New Families (unseen P5 base families + additional post-trained models).
        ax = axes[1]
        _apply_fig5_axes_style(ax)
        ax.set_xscale("log")
        ax.set_xlim(*xlim)
        ax.set_xticks(xticks)
        ax.set_xticklabels(xticklabels)
        ax.set_ylim(y_lo, y_hi)
        ax.set_yticks([t for t in (0.0, 0.2, 0.4, 0.6, 0.8) if y_lo - 1e-6 <= t <= y_hi + 1e-6])
        ax.tick_params(axis="y", labelleft=False)
        ax.set_yticklabels([])
        ax.set_xlabel(PRETRAINING_COMPUTE_FLOPS_LABEL, fontsize=12, fontweight="bold", labelpad=4)
        _fig5_panel_label(ax, "(b) New Families")

        p_right = panels[1]
        family_colors = p_right.get("family_colors")
        family_order = p_right.get("family_order")
        val_family = p_right.get("val_family")
        extra_family = p_right.get("extra_family")

        val_x = p_right.get("val_x")
        val_y = p_right.get("val_y")
        if (
            isinstance(val_x, np.ndarray)
            and isinstance(val_y, np.ndarray)
            and isinstance(val_family, np.ndarray)
            and isinstance(family_colors, dict)
            and isinstance(family_order, (list, tuple))
            and val_x.size
        ):
            for fam in family_order:
                m = val_family == fam
                if not np.any(m):
                    continue
                c = family_colors.get(str(fam), "#333333")
                ax.scatter(
                    val_x[m] * x_scale,
                    val_y[m],
                    s=(FIG5_STYLE["marker_size"] * 1.15) ** 2,
                    alpha=FIG5_STYLE["marker_alpha"],
                    facecolors="none",
                    edgecolors=c,
                    linewidths=0.9,
                    marker="^",
                    zorder=2,
                    rasterized=True,
                )

        extra_x = p_right.get("extra_x")
        extra_y = p_right.get("extra_y")
        if (
            isinstance(extra_x, np.ndarray)
            and isinstance(extra_y, np.ndarray)
            and isinstance(extra_family, np.ndarray)
            and isinstance(family_colors, dict)
            and isinstance(family_order, (list, tuple))
            and extra_x.size
        ):
            for fam in family_order:
                m = extra_family == fam
                if not np.any(m):
                    continue
                c = family_colors.get(str(fam), "#333333")
                ax.scatter(
                    extra_x[m] * x_scale,
                    extra_y[m],
                    s=(FIG5_STYLE["marker_size"] * 1.15) ** 2,
                    alpha=max(FIG5_STYLE["marker_alpha"], 0.6),
                    color=c,
                    edgecolors="none",
                    marker="D",
                    zorder=2,
                    rasterized=True,
                )

        curve_x = p_right.get("curve_x")
        curve_y = p_right.get("curve_y")
        if isinstance(curve_x, np.ndarray) and isinstance(curve_y, np.ndarray) and curve_x.size:
            ax.plot(
                curve_x * x_scale,
                curve_y,
                linewidth=FIG5_STYLE["line_width"],
                alpha=FIG5_STYLE["line_alpha"],
                color=FIG5_COLORS["p5_frontier"],
                linestyle="-",
                zorder=3,
            )

        # Family legend on this panel.
        if (
            isinstance(val_family, np.ndarray)
            and isinstance(extra_family, np.ndarray)
            and isinstance(family_order, (list, tuple))
            and isinstance(family_colors, dict)
        ):
            val_set = set(str(v) for v in val_family.tolist())
            extra_set = set(str(v) for v in extra_family.tolist())
            fams_for_legend = [str(f) for f in family_order if (str(f) in val_set or str(f) in extra_set)]
            handles = []
            for fam in fams_for_legend:
                c = family_colors.get(fam, "#333333")
                if fam in val_set:
                    handles.append(
                        Line2D(
                            [0],
                            [0],
                            marker="^",
                            linestyle="None",
                            color=c,
                            markerfacecolor="none",
                            markeredgecolor=c,
                            markeredgewidth=1.0,
                            markersize=7.5,
                            label=fam,
                        )
                    )
                else:
                    handles.append(
                        Line2D(
                            [0],
                            [0],
                            marker="D",
                            linestyle="None",
                            color=c,
                            markerfacecolor=c,
                            markeredgecolor=c,
                            markersize=6.5,
                            label=fam,
                        )
                    )
            if handles:
                ax.legend(
                    handles,
                    [h.get_label() for h in handles],
                    loc="upper left",
                    bbox_to_anchor=(0.03, 0.98),
                    frameon=True,
                    framealpha=0.9,
                    edgecolor="gray",
                    fancybox=False,
                    fontsize=9,
                    ncol=1,
                    handlelength=1.2,
                    handletextpad=0.4,
                    labelspacing=0.3,
                    borderpad=0.4,
                )

        base = _get_plot_path(out_dir, task, legacy=False, suffix="period4", plots_subdir=plots_subdir)
        _save_fig5(fig, base_path=base)
        plt.close(fig)
        return

    if use_fig1_style:
        # ---------------------------------------------------------------------
        # Figure 1-style rendering (fonts, grid, spines, k-badges, legend).
        # Also match subplot physical size: same height as a Figure 1 subplot,
        # and width increased by 20%.
        # ---------------------------------------------------------------------
        ref_panel_w, ref_panel_h = _panel_inches_from_layout(FIG1_LAYOUT)
        target_panel_w = 1.2 * ref_panel_w
        target_panel_h = ref_panel_h

        left = float(FIG1_LAYOUT["left"])
        right = float(FIG1_LAYOUT["right"])
        top = float(FIG1_LAYOUT["top"])
        # Extra bottom margin for a single-row layout (prevents xlabel overlap while
        # keeping the *axes* height identical to Figure 1's subplot height).
        bottom = 0.18
        wspace = float(FIG1_LAYOUT["wspace"])

        fig_w, fig_h = _figsize_for_panels(
            ncols=n,
            panel_w=target_panel_w,
            panel_h=target_panel_h,
            left=left,
            right=right,
            bottom=bottom,
            top=top,
            wspace=wspace,
        )

        fig, axes = plt.subplots(1, n, figsize=(fig_w, fig_h), dpi=300, sharey=True, sharex=bool(sharex))
        fig.patch.set_facecolor("white")
        axes = np.atleast_1d(axes)

        # Shared y-limits across panels for consistency (preserve content).
        y_vals: List[float] = []
        for p in panels:
            for key in ("train_y", "val_y", "extra_y", "curve_y", "curve_val_y"):
                arr = p.get(key)
                if arr is not None and getattr(arr, "size", 0):
                    y_vals.append(float(np.nanmin(arr)))
                    y_vals.append(float(np.nanmax(arr)))
        y_min, y_max = (0.0, 1.0)
        if y_vals:
            y_min = float(np.nanmin(y_vals))
            y_max = float(np.nanmax(y_vals))
            if not (np.isfinite(y_min) and np.isfinite(y_max)) or y_min == y_max:
                y_min, y_max = 0.0, 1.0
        pad = 0.02 * max(1e-6, (y_max - y_min))

        x_scale = 1e21  # convert zFLOPs -> FLOPs for plotting (matches Figure 1 axis units)
        xticks = [1e21, 1e22, 1e23, 1e24, 1e25]
        xticklabels = [f"$10^{{{int(np.log10(x))}}}$" for x in xticks]

        for idx, (ax, panel) in enumerate(zip(axes, panels), start=1):
            k_raw = panel.get("k_label", idx)
            if isinstance(k_raw, np.ndarray):
                k_label = int(k_raw.flat[0]) if k_raw.size else idx
            else:
                k_label = int(k_raw)

            _apply_fig1_axes_style(ax)
            ax.set_xscale("log")
            ax.set_ylim(y_min - pad, y_max + pad)

            # Force decade ticks like Figure 1 (labels stay consistent even if x-lims
            # include smaller values due to new-model overlays).
            ax.set_xticks(xticks)
            ax.set_xticklabels(xticklabels)

            train_x = panel.get("train_x")
            train_y = panel.get("train_y")
            if isinstance(train_x, np.ndarray) and isinstance(train_y, np.ndarray) and train_x.size:
                ax.scatter(
                    train_x * x_scale,
                    train_y,
                    s=FIG1_STYLE["marker_size"] ** 2,
                    alpha=FIG1_STYLE["marker_alpha"],
                    color=FIG1_STYLE["train_points_color"],
                    edgecolors=FIG1_STYLE["marker_edgecolor"],
                    linewidths=FIG1_STYLE["marker_edgewidth"],
                    zorder=3,
                    rasterized=True,
                )

            val_x = panel.get("val_x")
            val_y = panel.get("val_y")
            if isinstance(val_x, np.ndarray) and isinstance(val_y, np.ndarray) and val_x.size:
                ax.scatter(
                    val_x * x_scale,
                    val_y,
                    s=FIG1_STYLE["marker_size"] ** 2,
                    alpha=FIG1_STYLE["marker_alpha"],
                    color=FIG1_STYLE["val_points_color"],
                    edgecolors=FIG1_STYLE["marker_edgecolor"],
                    linewidths=FIG1_STYLE["marker_edgewidth"],
                    zorder=3,
                    rasterized=True,
                )

            curve_x = panel.get("curve_x")
            curve_y = panel.get("curve_y")
            if isinstance(curve_x, np.ndarray) and isinstance(curve_y, np.ndarray) and curve_x.size:
                ax.plot(
                    curve_x * x_scale,
                    curve_y,
                    linewidth=FIG1_STYLE["line_width_fit"],
                    alpha=FIG1_STYLE["line_alpha"],
                    color=FIG1_STYLE["fit_train_color"],
                    linestyle="-",
                    zorder=5,
                )

            curve_val_x = panel.get("curve_val_x")
            curve_val_y = panel.get("curve_val_y")
            if isinstance(curve_val_x, np.ndarray) and isinstance(curve_val_y, np.ndarray) and curve_val_x.size:
                ax.plot(
                    curve_val_x * x_scale,
                    curve_val_y,
                    linewidth=FIG1_STYLE["line_width_fit"],
                    alpha=FIG1_STYLE["line_alpha"],
                    color=FIG1_STYLE["fit_val_color"],
                    linestyle="--",
                    zorder=5,
                )

            if show_k_badge:
                ax.text(
                    0.5,
                    0.93,
                    f"$t = {k_label}$",
                    transform=ax.transAxes,
                    ha="center",
                    va="top",
                    fontsize=13,
                    fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="none", alpha=0.8),
                )

        axes[0].set_ylabel("Accuracy", fontsize=13, fontweight="bold")
        fig.text(0.5, 0.02, PRETRAINING_COMPUTE_FLOPS_LABEL, ha="center", va="center", fontsize=13, fontweight="bold")

        from matplotlib.lines import Line2D  # type: ignore

        legend_handles = [
            Line2D(
                [0],
                [0],
                marker="o",
                linestyle="None",
                markersize=6.5,
                markerfacecolor=FIG1_STYLE["train_points_color"],
                markeredgecolor=FIG1_STYLE["marker_edgecolor"],
                markeredgewidth=FIG1_STYLE["marker_edgewidth"],
                label="Train",
            ),
            Line2D(
                [0],
                [0],
                marker="o",
                linestyle="None",
                markersize=6.5,
                markerfacecolor=FIG1_STYLE["val_points_color"],
                markeredgecolor=FIG1_STYLE["marker_edgecolor"],
                markeredgewidth=FIG1_STYLE["marker_edgewidth"],
                label="Val",
            ),
            Line2D(
                [0],
                [0],
                color=FIG1_STYLE["fit_train_color"],
                linewidth=FIG1_STYLE["line_width_fit"],
                linestyle="-",
                label="Fit (Train)",
            ),
            Line2D(
                [0],
                [0],
                color=FIG1_STYLE["fit_val_color"],
                linewidth=FIG1_STYLE["line_width_fit"],
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

        fig.subplots_adjust(left=left, right=right, bottom=bottom, top=top, wspace=wspace)

        base = _get_plot_path(out_dir, task, legacy=False, suffix="period4", plots_subdir=plots_subdir)
        _save_fig1(fig, base_path=base)
        plt.close(fig)
        return
    # Match the existing 3-panel figure aspect by scaling width linearly with n.
    fig_w = period4_old_vs_new_cfg.BASE_FIGSIZE[0] * (n / 3.0)
    fig, axes = plt.subplots(
        1,
        n,
        figsize=(fig_w, period4_old_vs_new_cfg.BASE_FIGSIZE[1]),
        sharey=True,
        sharex=bool(sharex),
    )
    axes = np.atleast_1d(axes)

    palette = period4_old_vs_new_cfg.PALETTE
    x_label_fontsize = int(x_label_fontsize) if x_label_fontsize is not None else period4_old_vs_new_cfg.X_LABEL_FONTSIZE
    legend_loc = str(legend_loc) if legend_loc is not None else period4_old_vs_new_cfg.LEGEND_LOC

    # Shared y-limits across panels for consistency.
    y_vals: List[float] = []
    for p in panels:
        for key in ("train_y", "val_y", "extra_y", "curve_y", "curve_val_y"):
            arr = p.get(key)
            if arr is not None and arr.size:
                y_vals.append(float(np.nanmin(arr)))
                y_vals.append(float(np.nanmax(arr)))
    y_min, y_max = (0.0, 1.0)
    if y_vals:
        y_min = float(np.nanmin(y_vals))
        y_max = float(np.nanmax(y_vals))
        if not (np.isfinite(y_min) and np.isfinite(y_max)) or y_min == y_max:
            y_min, y_max = 0.0, 1.0
    pad = 0.02 * max(1e-6, (y_max - y_min))

    x_lower, x_upper = None, None
    if sharex:
        x_vals: List[float] = []
        for p in panels:
            for key in ("train_x", "val_x", "extra_x", "curve_x", "curve_val_x"):
                arr = p.get(key)
                if isinstance(arr, np.ndarray) and arr.size:
                    x_vals.append(float(np.nanmin(arr)))
                    x_vals.append(float(np.nanmax(arr)))
        if x_vals:
            x_min = float(np.nanmin(x_vals))
            x_max = float(np.nanmax(x_vals))
            if np.isfinite(x_min) and np.isfinite(x_max) and x_min > 0 and x_max > 0 and x_min != x_max:
                x_lower = x_min * 0.9
                x_upper = x_max * 1.05

    for idx, (ax, panel) in enumerate(zip(axes, panels), start=1):
        k_raw = panel.get("k_label", idx)
        if isinstance(k_raw, np.ndarray):
            k_label = int(k_raw.flat[0]) if k_raw.size else idx
        else:
            k_label = int(k_raw)
        # Scatter points: color encodes base-model family; marker encodes old/train vs new/val.
        train_marker = period4_old_vs_new_cfg.TRAIN_MARKER
        val_marker = period4_old_vs_new_cfg.VAL_MARKER
        extra_marker = "D"
        train_alpha = period4_old_vs_new_cfg.TRAIN_ALPHA
        val_alpha = period4_old_vs_new_cfg.VAL_ALPHA
        train_zorder = period4_old_vs_new_cfg.TRAIN_ZORDER
        val_zorder = period4_old_vs_new_cfg.VAL_ZORDER
        val_linewidth = period4_old_vs_new_cfg.VAL_EDGE_LINEWIDTH
        color_by_family = str(scatter_style) == "family"
        extra_color = getattr(period4_old_vs_new_cfg, "EXTRA_COLOR", "#A6D96A")
        base_scatter_size = float(period4_old_vs_new_cfg.SCATTER_SIZE) * float(base_scatter_scale)
        extra_scatter_size = float(period4_old_vs_new_cfg.SCATTER_SIZE)
        use_family_markers = bool(panel.get("use_family_markers", False))
        custom_family_colors = panel.get("family_colors")
        custom_family_order = panel.get("family_order")

        train_x = panel.get("train_x")
        train_y = panel.get("train_y")
        train_base = panel.get("train_base")
        if isinstance(train_x, np.ndarray) and isinstance(train_y, np.ndarray) and train_x.size:
            if color_by_family and isinstance(train_base, np.ndarray) and train_base.size == train_x.size:
                fams = np.asarray([family_from_base_model(b, include_gpt=True) for b in train_base], dtype=object)
                for fam in FAMILY_ORDER_WITH_GPT:
                    m = fams == fam
                    if not np.any(m):
                        continue
                    ax.scatter(
                        train_x[m],
                        train_y[m],
                        s=base_scatter_size,
                        alpha=train_alpha,
                        color=color_for_family(str(fam)),
                        linewidths=0,
                        rasterized=True,
                        marker=train_marker,
                        label="_nolegend_",
                        zorder=train_zorder,
                    )
            else:
                ax.scatter(
                    train_x,
                    train_y,
                    s=base_scatter_size,
                    alpha=train_alpha,
                    color=(palette["train"] if not color_by_family else "0.5"),
                    linewidths=0,
                    rasterized=True,
                    marker=train_marker,
                    label="_nolegend_",
                    zorder=train_zorder,
                )

        val_x = panel.get("val_x")
        val_y = panel.get("val_y")
        val_base = panel.get("val_base")
        val_family = panel.get("val_family")
        if isinstance(val_x, np.ndarray) and isinstance(val_y, np.ndarray) and val_x.size:
            if (
                use_family_markers
                and isinstance(val_family, np.ndarray)
                and val_family.size == val_x.size
                and isinstance(custom_family_colors, dict)
            ):
                fams = val_family.astype(str)
                order = (
                    [str(f) for f in custom_family_order]
                    if isinstance(custom_family_order, (list, tuple))
                    else sorted(set(fams.tolist()))
                )
                for fam in order:
                    m = fams == fam
                    if not np.any(m):
                        continue
                    ax.scatter(
                        val_x[m],
                        val_y[m],
                        s=base_scatter_size,
                        alpha=val_alpha,
                        facecolors="none",
                        edgecolors=custom_family_colors.get(fam, "#333333"),
                        linewidths=val_linewidth,
                        rasterized=True,
                        marker=val_marker,
                        label="_nolegend_",
                        zorder=val_zorder,
                    )
            elif use_family_markers and isinstance(val_base, np.ndarray) and val_base.size == val_x.size:
                fams = np.asarray([family_from_base_model(b, include_gpt=True) for b in val_base], dtype=object)
                for fam in FAMILY_ORDER_WITH_GPT:
                    m = fams == fam
                    if not np.any(m):
                        continue
                    ax.scatter(
                        val_x[m],
                        val_y[m],
                        s=base_scatter_size,
                        alpha=val_alpha,
                        facecolors="none",
                        edgecolors=color_for_family(str(fam)),
                        linewidths=val_linewidth,
                        rasterized=True,
                        marker=FAMILY_MARKERS.get(str(fam), val_marker),
                        label="_nolegend_",
                        zorder=val_zorder,
                    )
            elif color_by_family and isinstance(val_base, np.ndarray) and val_base.size == val_x.size:
                fams = np.asarray([family_from_base_model(b, include_gpt=True) for b in val_base], dtype=object)
                for fam in FAMILY_ORDER_WITH_GPT:
                    m = fams == fam
                    if not np.any(m):
                        continue
                    ax.scatter(
                        val_x[m],
                        val_y[m],
                        s=base_scatter_size,
                        alpha=val_alpha,
                        facecolors="none",
                        edgecolors=color_for_family(str(fam)),
                        linewidths=val_linewidth,
                        rasterized=True,
                        marker=val_marker,
                        label="_nolegend_",
                        zorder=val_zorder,
                    )
            else:
                ax.scatter(
                    val_x,
                    val_y,
                    s=base_scatter_size,
                    alpha=val_alpha,
                    facecolors="none",
                    edgecolors=(palette["val"] if not color_by_family else "0.5"),
                    linewidths=val_linewidth,
                    rasterized=True,
                    marker=val_marker,
                    label="_nolegend_",
                    zorder=val_zorder,
                )

        extra_x = panel.get("extra_x")
        extra_y = panel.get("extra_y")
        extra_base = panel.get("extra_base")
        extra_id = panel.get("extra_id")
        extra_family = panel.get("extra_family")
        if isinstance(extra_x, np.ndarray) and isinstance(extra_y, np.ndarray) and extra_x.size:
            if (
                use_family_markers
                and isinstance(extra_family, np.ndarray)
                and extra_family.size == extra_x.size
                and isinstance(custom_family_colors, dict)
            ):
                fams = extra_family.astype(str)
                order = (
                    [str(f) for f in custom_family_order]
                    if isinstance(custom_family_order, (list, tuple))
                    else sorted(set(fams.tolist()))
                )
                for fam in order:
                    m = fams == fam
                    if not np.any(m):
                        continue
                    ax.scatter(
                        extra_x[m],
                        extra_y[m],
                        s=extra_scatter_size,
                        alpha=max(val_alpha, 0.6),
                        color=custom_family_colors.get(fam, extra_color),
                        linewidths=0,
                        rasterized=True,
                        marker=extra_marker,
                        label="_nolegend_",
                        zorder=val_zorder + 1,
                    )
            elif use_family_markers:
                fam_source = None
                if isinstance(extra_base, np.ndarray) and extra_base.size == extra_x.size:
                    fam_source = extra_base
                elif isinstance(extra_id, np.ndarray) and extra_id.size == extra_x.size:
                    fam_source = extra_id
                if fam_source is not None:
                    fams = np.asarray([family_from_base_model(v, include_gpt=True) for v in fam_source], dtype=object)
                    for fam in FAMILY_ORDER_WITH_GPT:
                        m = fams == fam
                        if not np.any(m):
                            continue
                        ax.scatter(
                            extra_x[m],
                            extra_y[m],
                            s=extra_scatter_size,
                            alpha=max(val_alpha, 0.6),
                            color=color_for_family(str(fam)),
                            linewidths=0,
                            rasterized=True,
                            marker=FAMILY_MARKERS.get(str(fam), extra_marker),
                            label="_nolegend_",
                            zorder=val_zorder + 1,
                        )
                else:
                    ax.scatter(
                        extra_x,
                        extra_y,
                        s=extra_scatter_size,
                        alpha=max(val_alpha, 0.6),
                        color=extra_color,
                        linewidths=0,
                        rasterized=True,
                        marker=extra_marker,
                        label="_nolegend_",
                        zorder=val_zorder + 1,
                    )
            else:
                ax.scatter(
                    extra_x,
                    extra_y,
                    s=extra_scatter_size,
                    alpha=max(val_alpha, 0.6),
                    color=extra_color,
                    linewidths=0,
                    rasterized=True,
                    marker=extra_marker,
                    label="_nolegend_",
                    zorder=val_zorder + 1,
                )
        # Curves.
        if isinstance(panel.get("curve_x"), np.ndarray) and panel["curve_x"].size:
            ax.plot(
                panel["curve_x"],
                panel["curve_y"],
                color=palette["curve"],
                linewidth=period4_old_vs_new_cfg.CURVE_LINEWIDTH,
                alpha=period4_old_vs_new_cfg.CURVE_ALPHA,
                label="fit (train)",
            )
        if isinstance(panel.get("curve_val_x"), np.ndarray) and isinstance(panel.get("curve_val_y"), np.ndarray):
            if panel["curve_val_x"].size:
                ax.plot(
                    panel["curve_val_x"],
                    panel["curve_val_y"],
                    color=palette["curve_val"],
                    linewidth=period4_old_vs_new_cfg.CURVE_LINEWIDTH,
                    alpha=period4_old_vs_new_cfg.CURVE_ALPHA,
                    linestyle="--",
                    label="fit (val)",
                )

        ax.set_xscale("log")
        apply_pretraining_compute_tick_multiplier(ax, require_label_match=False)
        if x_lower is not None and x_upper is not None:
            ax.set_xlim(x_lower, x_upper)
        ax.set_ylim(y_min - pad, y_max + pad)

        ax.tick_params(
            axis="both",
            labelsize=period4_old_vs_new_cfg.TICK_LABELSIZE,
            length=period4_old_vs_new_cfg.TICK_LENGTH,
            width=period4_old_vs_new_cfg.TICK_WIDTH,
            direction=period4_old_vs_new_cfg.TICK_DIRECTION,
        )
        for spine in ax.spines.values():
            spine.set_linewidth(period4_old_vs_new_cfg.SPINE_LINEWIDTH)
            try:
                spine.set_color(period4_old_vs_new_cfg.SPINE_COLOR)
                spine.set_alpha(period4_old_vs_new_cfg.SPINE_ALPHA)
            except Exception:
                pass

        if show_k_badge:
            # k-badge.
            try:
                with mpl.rc_context({"text.usetex": True, "text.latex.preamble": r"\usepackage{amsmath}"}):
                    badge = AnchoredText(
                        fr"$t={k_label}$",
                        loc="upper left",
                        prop=dict(
                            size=period4_old_vs_new_cfg.BADGE_FONTSIZE,
                            weight=period4_old_vs_new_cfg.BADGE_WEIGHT,
                            color=period4_old_vs_new_cfg.BADGE_COLOR,
                        ),
                    )
            except Exception:
                badge = AnchoredText(
                    fr"$t={k_label}$",
                    loc="upper left",
                    prop=dict(
                        size=period4_old_vs_new_cfg.BADGE_FONTSIZE,
                        weight=period4_old_vs_new_cfg.BADGE_WEIGHT,
                        color=period4_old_vs_new_cfg.BADGE_COLOR,
                    ),
                )
            try:
                badge.patch.set_facecolor(period4_old_vs_new_cfg.BADGE_BOX_FACE)
                badge.patch.set_alpha(period4_old_vs_new_cfg.BADGE_BOX_ALPHA)
                badge.patch.set_edgecolor(period4_old_vs_new_cfg.BADGE_BOX_EDGE)
                badge.patch.set_linewidth(period4_old_vs_new_cfg.BADGE_BOX_LINEWIDTH)
            except Exception:
                pass
            ax.add_artist(badge)

        # Grid styling (major + minor).
        try:
            ax.yaxis.set_minor_locator(mticker.AutoMinorLocator(2))
            if ax.get_xscale() == "log":
                ax.xaxis.set_minor_locator(
                    mticker.LogLocator(base=10.0, subs=(0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9))
                )
            else:
                ax.xaxis.set_minor_locator(mticker.AutoMinorLocator(2))
        except Exception:
            pass
        ax.yaxis.grid(
            True,
            which="major",
            linestyle=period4_old_vs_new_cfg.GRID_MAJOR_LINESTYLE,
            linewidth=period4_old_vs_new_cfg.GRID_MAJOR_LINEWIDTH,
            color=period4_old_vs_new_cfg.GRID_MAJOR_COLOR,
            alpha=period4_old_vs_new_cfg.GRID_MAJOR_ALPHA,
        )
        ax.yaxis.grid(
            True,
            which="minor",
            linestyle=period4_old_vs_new_cfg.GRID_MINOR_LINESTYLE,
            linewidth=period4_old_vs_new_cfg.GRID_MINOR_LINEWIDTH,
            color=period4_old_vs_new_cfg.GRID_MINOR_COLOR,
            alpha=period4_old_vs_new_cfg.GRID_MINOR_ALPHA,
        )
        ax.xaxis.grid(
            True,
            which="major",
            linestyle=period4_old_vs_new_cfg.GRID_MAJOR_LINESTYLE,
            linewidth=period4_old_vs_new_cfg.GRID_MAJOR_LINEWIDTH,
            color=period4_old_vs_new_cfg.GRID_MAJOR_COLOR,
            alpha=period4_old_vs_new_cfg.GRID_MAJOR_ALPHA,
        )
        ax.xaxis.grid(
            True,
            which="minor",
            linestyle=period4_old_vs_new_cfg.GRID_MINOR_LINESTYLE,
            linewidth=period4_old_vs_new_cfg.GRID_MINOR_LINEWIDTH,
            color=period4_old_vs_new_cfg.GRID_MINOR_COLOR,
            alpha=period4_old_vs_new_cfg.GRID_MINOR_ALPHA,
        )

    if sharex and len(axes) > 1:
        for ax in axes[1:]:
            ax.tick_params(axis="y", labelleft=True)

    axes[0].set_ylabel("Accuracy", fontweight="bold", fontsize=period4_old_vs_new_cfg.Y_LABEL_FONTSIZE)

    # Centered x-label: use per-axis xlabel for odd n (matches existing), supxlabel for even n.
    if n % 2 == 1:
        axes[(n // 2)].set_xlabel(PRETRAINING_COMPUTE_FLOPS_LABEL, fontweight="bold", fontsize=x_label_fontsize)
    else:
        if supxlabel_y is not None:
            fig.supxlabel(PRETRAINING_COMPUTE_FLOPS_LABEL, y=float(supxlabel_y), fontweight="bold", fontsize=x_label_fontsize)
        else:
            fig.supxlabel(PRETRAINING_COMPUTE_FLOPS_LABEL, fontweight="bold", fontsize=x_label_fontsize)

    from matplotlib.lines import Line2D  # type: ignore

    marker_handles = [
        Line2D([0], [0], marker="o", linestyle="None", color="#333333", markersize=8, label="train"),
        Line2D(
            [0],
            [0],
            marker="^",
            linestyle="None",
            color="#333333",
            markerfacecolor="none",
            markeredgecolor="#333333",
            markeredgewidth=1.2,
            markersize=9,
            label="val",
        ),
    ]
    legend_ax = axes[int(legend_axis_index)]
    line_handles, line_labels = legend_ax.get_legend_handles_labels()
    handles = marker_handles + line_handles
    labels = [h.get_label() for h in marker_handles] + line_labels
    leg = legend_ax.legend(
        handles,
        labels,
        loc=legend_loc,
        fontsize=period4_old_vs_new_cfg.LEGEND_FONTSIZE,
        frameon=True,
        framealpha=period4_old_vs_new_cfg.LEGEND_FRAMEALPHA,
        fancybox=period4_old_vs_new_cfg.LEGEND_FANCYBOX,
        borderpad=period4_old_vs_new_cfg.LEGEND_BORDERPAD,
    )
    if leg and leg.get_title():
        leg.get_title().set_fontweight("bold")

    if show_extra_family_legend:
        if len(axes) >= 2 and panels:
            right_panel = panels[-1]
            custom_colors = right_panel.get("family_colors")
            custom_order = right_panel.get("family_order")
            val_family = right_panel.get("val_family")
            extra_family = right_panel.get("extra_family")
            if (
                isinstance(custom_colors, dict)
                and isinstance(custom_order, (list, tuple))
                and (isinstance(val_family, np.ndarray) or isinstance(extra_family, np.ndarray))
            ):
                val_set = set(str(v) for v in (val_family.tolist() if isinstance(val_family, np.ndarray) else []))
                extra_set = set(str(v) for v in (extra_family.tolist() if isinstance(extra_family, np.ndarray) else []))
                fams_for_legend = [str(f) for f in custom_order if (str(f) in val_set or str(f) in extra_set)]
                if fams_for_legend:
                    fam_handles = []
                    for fam in fams_for_legend:
                        c = custom_colors.get(fam, "#333333")
                        if fam in val_set:
                            fam_handles.append(
                                Line2D(
                                    [0],
                                    [0],
                                    marker="^",
                                    linestyle="None",
                                    color=c,
                                    markerfacecolor="none",
                                    markeredgecolor=c,
                                    markeredgewidth=1.2,
                                    markersize=9,
                                    label=fam,
                                )
                            )
                        else:
                            fam_handles.append(
                                Line2D(
                                    [0],
                                    [0],
                                    marker="D",
                                    linestyle="None",
                                    color=c,
                                    markerfacecolor=c,
                                    markeredgecolor=c,
                                    markersize=8,
                                    label=fam,
                                )
                            )
                    axes[-1].legend(
                        fam_handles,
                        [h.get_label() for h in fam_handles],
                        loc="upper left",
                        fontsize=period4_old_vs_new_cfg.LEGEND_FONTSIZE,
                        frameon=True,
                        framealpha=period4_old_vs_new_cfg.LEGEND_FRAMEALPHA,
                        fancybox=period4_old_vs_new_cfg.LEGEND_FANCYBOX,
                        borderpad=period4_old_vs_new_cfg.LEGEND_BORDERPAD,
                    )
            else:
                fams_present: List[str] = []
                for key in ("val_base", "extra_base"):
                    vals = right_panel.get(key)
                    if isinstance(vals, np.ndarray) and vals.size:
                        for v in vals.tolist():
                            fam = family_from_base_model(str(v), include_gpt=True)
                            if fam not in fams_present:
                                fams_present.append(fam)
                fams_for_legend = [f for f in FAMILY_ORDER_WITH_GPT if f in fams_present]
                if fams_for_legend:
                    fam_handles = [
                        Line2D(
                            [0],
                            [0],
                            marker=FAMILY_MARKERS.get(fam, "o"),
                            linestyle="None",
                            color=color_for_family(fam),
                            markersize=8,
                            label=fam,
                        )
                        for fam in fams_for_legend
                    ]
                    axes[-1].legend(
                        fam_handles,
                        [h.get_label() for h in fam_handles],
                        loc="upper left",
                        fontsize=period4_old_vs_new_cfg.LEGEND_FONTSIZE,
                        frameon=True,
                        framealpha=period4_old_vs_new_cfg.LEGEND_FRAMEALPHA,
                        fancybox=period4_old_vs_new_cfg.LEGEND_FANCYBOX,
                        borderpad=period4_old_vs_new_cfg.LEGEND_BORDERPAD,
                    )

    fig.subplots_adjust(wspace=period4_old_vs_new_cfg.PANEL_WSPACE)
    fig.tight_layout()

    base = _get_plot_path(out_dir, task, legacy=False, suffix="period4", plots_subdir=plots_subdir)
    fig.savefig(base + ".png", dpi=300, bbox_inches="tight")
    fig.savefig(base + ".pdf", dpi=300, bbox_inches="tight")
    plt.close(fig)


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Plot period4/single_k sigmoid frontiers: old OLL vs new models."
    )
    p.add_argument(
        "--oll_csv",
        default=os.path.join("tables", "open_llm_leaderboard", "open_llm_leaderboard_with_tokens.csv"),
        help="OLL CSV (with_tokens schema).",
    )
    p.add_argument(
        "--new_eval_csv",
        default=os.path.join("tables", "open_llm_leaderboard", "validation_leaderboard.csv"),
        help="Validation leaderboard CSV containing evaluation metrics (new models).",
    )
    p.add_argument(
        "--new_compute_csv",
        default=os.path.join("tables", "open_llm_leaderboard", "new_eval_leaderboard.csv"),
        help="CSV containing pretraining compute (pretrain_compute_zflops) for new models.",
    )
    p.add_argument(
        "--top_models_csv",
        default=os.path.join("tables", "top_models_by_base.csv"),
        help="Top-models metadata CSV (for last_modified).",
    )
    p.add_argument(
        "--out_dir",
        default=None,
        help="Output directory root (plots/ will be created under this).",
    )
    p.add_argument(
        "--mode",
        choices=["same_period", "p4_to_p5"],
        default="same_period",
        help=(
            "Which comparison to plot. "
            "'same_period' plots old Pk vs new Pk for k=1..4; "
            "'p4_to_p5' plots old P4 vs new P5 only."
        ),
    )
    p.add_argument("--tau", type=float, default=0.98, help="Sigmoid frontier quantile tau.")
    p.add_argument(
        "--lambda_b",
        type=float,
        default=1e-3,
        help="L2 penalty on sigmoid slope b in the frontier fitter (default: 1e-3)",
    )
    p.add_argument(
        "--frontier_fit_mode",
        choices=["quantile_per_point", "robust_bin_frontier"],
        default="quantile_per_point",
        help="Sigmoid fitting mode (match scripts/smooth_single_skill_frontier.py defaults).",
    )
    p.add_argument("--bins_for_fit", type=int, default=120, help="Bins used for robust-bin mode targets.")
    p.add_argument("--bin_frontier_quantile", type=float, default=0.98)
    p.add_argument("--bin_trim_fraction", type=float, default=0.01)
    p.add_argument(
        "--scatter_style",
        choices=["two_color", "family"],
        default="two_color",
        help="Scatter styling: 'two_color' (default) disables model-family coloring; 'family' colors points by base-model family.",
    )
    p.add_argument(
        "--extra_points_csv",
        default=None,
        help=(
            "Optional CSV (new_leaderboard_results_with_tokens.csv schema) to overlay as extra points "
            "in p4_to_p5 plots."
        ),
    )
    return p


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = build_argparser().parse_args(argv)
    if args.mode == "p4_to_p5" and not args.extra_points_csv:
        default_extra = os.path.join("tables", "new_leaderboard_results_with_tokens.csv")
        if os.path.exists(default_extra):
            args.extra_points_csv = default_extra
            print(f"[plot_period4_old_vs_new] --extra_points_csv not set; using {default_extra}")

    if args.out_dir is None:
        if args.mode == "p4_to_p5":
            args.out_dir = os.path.join("outputs", "sigmoid", "period4", "single_k", "new_models_p5")
        else:
            args.out_dir = os.path.join("outputs", "sigmoid", "period4", "single_k", "new_models")

    # Task list = canonical OLL Raw tasks present in old CSV, intersected with new mapping.
    df_oll_head = pd.read_csv(args.oll_csv, nrows=1)
    tasks = [t for t in MAIN_TASK_MAP_OLL_TO_NEW.keys() if t in df_oll_head.columns]
    if not tasks:
        raise SystemExit("No supported tasks found in OLL CSV.")

    df_old = _load_old_oll(args.oll_csv, tasks=tasks)
    df_new = _load_new_eval(args.new_eval_csv, args.new_compute_csv, args.top_models_csv, tasks=tasks)
    df_extra = None
    if args.extra_points_csv:
        df_extra = _load_extra_leaderboard(args.extra_points_csv, tasks=tasks)

    # Filter to finite compute. Period filtering happens per mode below.
    df_old = df_old[np.isfinite(df_old["compute_zflops"]) & (df_old["compute_zflops"] > 0)]
    df_new = df_new[np.isfinite(df_new["compute_zflops"]) & (df_new["compute_zflops"] > 0)]
    if df_extra is not None:
        df_extra = df_extra[np.isfinite(df_extra["compute_zflops"]) & (df_extra["compute_zflops"] > 0)]

    df_additional: Optional[pd.DataFrame] = None
    prev_base_set: set[str] = set()
    if args.mode == "p4_to_p5":
        prev_bases = df_new[df_new["pid"].isin([0, 1, 2, 3])].get("base_model", pd.Series([], dtype=str)).astype(str)
        prev_base_set = {k for k in (_period_base_membership_key(b) for b in prev_bases.tolist()) if k}
    if args.mode == "p4_to_p5" and df_extra is not None:
        additional_csv = os.path.join("tables", "additional_post_trained_models.csv")
        additional_ids = _load_additional_post_trained_model_ids(additional_csv)
        df_additional = df_extra[df_extra.get("model_id", pd.Series([], dtype=str)).astype(str).isin(additional_ids)].copy()
        if not df_additional.empty:
            extra_ids_norm = df_additional["model_id"].astype(str).str.strip().str.lower()
            df_additional = df_additional.loc[~extra_ids_norm.isin(PANEL_D_EXCLUDED_MODEL_IDS)].copy()
        found = set(df_additional.get("model_id", pd.Series([], dtype=str)).astype(str).tolist())
        missing = [m for m in additional_ids if m not in found]
        if missing:
            print(f"[plot_period4_old_vs_new] WARNING: {len(missing)} additional models missing from {args.extra_points_csv}")

    os.makedirs(args.out_dir, exist_ok=True)

    for task in tasks:
        panels: List[Dict[str, object]] = []
        if args.mode == "p4_to_p5":
            pairs = [(4, 3, 4)]  # k_label, old_pid, new_pid
        else:
            pairs = [(1, 0, 0), (2, 1, 1), (3, 2, 2), (4, 3, 3)]
        for k_label, old_pid, new_pid in pairs:
            old_k = df_old[df_old["pid"] == old_pid]
            new_k = df_new[df_new["pid"] == new_pid]
            if args.mode == "p4_to_p5":
                # Split P5 models by whether their base-model family appeared in P1..P4 (validation leaderboard).
                bases = new_k.get("base_model", pd.Series([np.nan] * len(new_k))).astype(str)
                base_keys = bases.map(_period_base_membership_key)
                is_prev_base = base_keys.isin(prev_base_set)
                new_left = new_k.loc[is_prev_base].copy()
                new_right = new_k.loc[~is_prev_base].copy()
                if not new_right.empty and "model_id" in new_right.columns:
                    right_ids_norm = new_right["model_id"].astype(str).str.strip().str.lower()
                    new_right = new_right.loc[~right_ids_norm.isin(PANEL_D_EXCLUDED_MODEL_IDS)].copy()

                extra_right = None
                if df_additional is not None and not df_additional.empty:
                    val_ids_right = set(new_right.get("model_id", pd.Series([], dtype=str)).astype(str).tolist())
                    extra_right = df_additional[~df_additional["model_id"].astype(str).isin(val_ids_right)].copy()

                def _panel_for(old_df: pd.DataFrame, new_df: pd.DataFrame, extra_df: Optional[pd.DataFrame]) -> Dict[str, object]:
                    x_train = pd.to_numeric(old_df["compute_zflops"], errors="coerce").to_numpy(dtype=float)
                    y_train = pd.to_numeric(old_df[task], errors="coerce").to_numpy(dtype=float)
                    b_train = old_df.get("base_model", pd.Series([np.nan] * len(old_df))).astype(str).to_numpy(dtype=object)
                    mtr = np.isfinite(x_train) & (x_train > 0.0) & np.isfinite(y_train)
                    x_train = x_train[mtr]
                    y_train = y_train[mtr]
                    b_train = b_train[mtr]

                    x_val = pd.to_numeric(new_df["compute_zflops"], errors="coerce").to_numpy(dtype=float)
                    y_val = pd.to_numeric(new_df.get(task, np.nan), errors="coerce").to_numpy(dtype=float)
                    b_val = new_df.get("base_model", pd.Series([np.nan] * len(new_df))).astype(str).to_numpy(dtype=object)
                    mval = np.isfinite(x_val) & (x_val > 0.0) & np.isfinite(y_val)
                    x_val = x_val[mval]
                    y_val = y_val[mval]
                    b_val = b_val[mval]

                    x_extra = np.array([], dtype=float)
                    y_extra = np.array([], dtype=float)
                    b_extra = np.array([], dtype=object)
                    id_extra = np.array([], dtype=object)
                    if extra_df is not None:
                        x_extra = pd.to_numeric(extra_df["compute_zflops"], errors="coerce").to_numpy(dtype=float)
                        y_extra = pd.to_numeric(extra_df.get(task, np.nan), errors="coerce").to_numpy(dtype=float)
                        b_extra = extra_df.get("base_model", pd.Series([np.nan] * len(extra_df))).astype(str).to_numpy(dtype=object)
                        id_extra = extra_df.get("model_id", pd.Series([np.nan] * len(extra_df))).astype(str).to_numpy(dtype=object)
                        mex = np.isfinite(x_extra) & (x_extra > 0.0) & np.isfinite(y_extra)
                        x_extra = x_extra[mex]
                        y_extra = y_extra[mex]
                        b_extra = b_extra[mex]
                        id_extra = id_extra[mex]

                    if x_train.size >= 3:
                        xs_curve, y_curve = fit_sigmoid_frontier(
                            x_train,
                            y_train,
                            tau=float(args.tau),
                            use_log10_x=True,
                            fit_mode=str(args.frontier_fit_mode),
                            bins_for_fit=int(args.bins_for_fit),
                            min_bin_size_for_fit=30,
                            bin_frontier_quantile=float(args.bin_frontier_quantile),
                            bin_trim_fraction=float(args.bin_trim_fraction),
                            lambda_b=float(args.lambda_b),
                        )
                    else:
                        xs_curve, y_curve = np.array([]), np.array([])

                    if x_val.size >= 3:
                        xs_curve_val, y_curve_val = fit_sigmoid_frontier(
                            x_val,
                            y_val,
                            tau=float(args.tau),
                            use_log10_x=True,
                            fit_mode=str(args.frontier_fit_mode),
                            bins_for_fit=int(args.bins_for_fit),
                            min_bin_size_for_fit=30,
                            bin_frontier_quantile=float(args.bin_frontier_quantile),
                            bin_trim_fraction=float(args.bin_trim_fraction),
                            lambda_b=float(args.lambda_b),
                        )
                    else:
                        xs_curve_val, y_curve_val = np.array([]), np.array([])

                    return {
                        "k_label": int(k_label),
                        "train_x": x_train,
                        "train_y": y_train,
                        "train_base": b_train,
                        "val_x": x_val,
                        "val_y": y_val,
                        "val_base": b_val,
                        "extra_x": x_extra,
                        "extra_y": y_extra,
                        "extra_base": b_extra,
                        "extra_id": id_extra,
                        "curve_x": xs_curve,
                        "curve_y": y_curve,
                        "curve_val_x": xs_curve_val,
                        "curve_val_y": y_curve_val,
                    }

                left = _panel_for(old_k, new_left, None)
                right = _panel_for(old_k, new_right, extra_right)
                # Keep curves, but hide train scatter in the right subplot.
                right["train_x"] = np.array([], dtype=float)
                right["train_y"] = np.array([], dtype=float)
                right["train_base"] = np.array([], dtype=object)
                right["use_family_markers"] = True
                right["curve_val_x"] = np.array([], dtype=float)
                right["curve_val_y"] = np.array([], dtype=float)
                # Right subplot family coloring/legend: additional post-trained families + unseen P5 base families.
                p5_colors = {
                    "Qwen3": "#d62728",
                    "GPT-OSS": "#bcbd22",
                    "Gemma-3": "#17becf",
                    "SmolLM3": "#e377c2",
                }
                family_colors = dict(getattr(period4_old_vs_new_cfg, "EXTRA_FAMILY_COLORS", {}))
                family_colors.update(p5_colors)
                family_order = list(getattr(period4_old_vs_new_cfg, "EXTRA_FAMILY_ORDER", [])) + list(p5_colors.keys())
                right["family_colors"] = family_colors
                right["family_order"] = family_order
                val_base = right.get("val_base")
                if isinstance(val_base, np.ndarray) and val_base.size:
                    right["val_family"] = np.asarray([_p5_family_from_base_model(b) for b in val_base.tolist()], dtype=object)
                extra_id = right.get("extra_id")
                if isinstance(extra_id, np.ndarray) and extra_id.size:
                    right["extra_family"] = np.asarray([_extra_family_from_model_id(m) for m in extra_id.tolist()], dtype=object)
                panels.extend([left, right])
                continue

            extra_k = None
            if df_extra is not None and args.mode == "p4_to_p5":
                val_ids = set(new_k.get("model_id", pd.Series([], dtype=str)).astype(str).tolist())
                extra_k = df_extra[~df_extra["model_id"].astype(str).isin(val_ids)]

            x_train = pd.to_numeric(old_k["compute_zflops"], errors="coerce").to_numpy(dtype=float)
            y_train = pd.to_numeric(old_k[task], errors="coerce").to_numpy(dtype=float)
            b_train = old_k.get("base_model", pd.Series([np.nan] * len(old_k))).astype(str).to_numpy(dtype=object)
            mtr = np.isfinite(x_train) & (x_train > 0.0) & np.isfinite(y_train)
            x_train = x_train[mtr]
            y_train = y_train[mtr]
            b_train = b_train[mtr]

            x_val = pd.to_numeric(new_k["compute_zflops"], errors="coerce").to_numpy(dtype=float)
            y_val = pd.to_numeric(new_k.get(task, np.nan), errors="coerce").to_numpy(dtype=float)
            b_val = new_k.get("base_model", pd.Series([np.nan] * len(new_k))).astype(str).to_numpy(dtype=object)
            mval = np.isfinite(x_val) & (x_val > 0.0) & np.isfinite(y_val)
            x_val = x_val[mval]
            y_val = y_val[mval]
            b_val = b_val[mval]

            x_extra = np.array([], dtype=float)
            y_extra = np.array([], dtype=float)
            b_extra = np.array([], dtype=object)
            id_extra = np.array([], dtype=object)
            if extra_k is not None:
                x_extra = pd.to_numeric(extra_k["compute_zflops"], errors="coerce").to_numpy(dtype=float)
                y_extra = pd.to_numeric(extra_k.get(task, np.nan), errors="coerce").to_numpy(dtype=float)
                b_extra = extra_k.get("base_model", pd.Series([np.nan] * len(extra_k))).astype(str).to_numpy(dtype=object)
                id_extra = extra_k.get("model_id", pd.Series([np.nan] * len(extra_k))).astype(str).to_numpy(dtype=object)
                mex = np.isfinite(x_extra) & (x_extra > 0.0) & np.isfinite(y_extra)
                x_extra = x_extra[mex]
                y_extra = y_extra[mex]
                b_extra = b_extra[mex]
                id_extra = id_extra[mex]

            if x_train.size >= 3:
                xs_curve, y_curve = fit_sigmoid_frontier(
                    x_train,
                    y_train,
                    tau=float(args.tau),
                    use_log10_x=True,
                    fit_mode=str(args.frontier_fit_mode),
                    bins_for_fit=int(args.bins_for_fit),
                    min_bin_size_for_fit=30,
                    bin_frontier_quantile=float(args.bin_frontier_quantile),
                    bin_trim_fraction=float(args.bin_trim_fraction),
                    lambda_b=float(args.lambda_b),
                )
            else:
                xs_curve, y_curve = np.array([]), np.array([])

            if x_val.size >= 3:
                xs_curve_val, y_curve_val = fit_sigmoid_frontier(
                    x_val,
                    y_val,
                    tau=float(args.tau),
                    use_log10_x=True,
                    fit_mode=str(args.frontier_fit_mode),
                    bins_for_fit=int(args.bins_for_fit),
                    min_bin_size_for_fit=30,
                    bin_frontier_quantile=float(args.bin_frontier_quantile),
                    bin_trim_fraction=float(args.bin_trim_fraction),
                    lambda_b=float(args.lambda_b),
                )
            else:
                xs_curve_val, y_curve_val = np.array([]), np.array([])

            panels.append(
                {
                    "k_label": int(k_label),
                    "train_x": x_train,
                    "train_y": y_train,
                    "train_base": b_train,
                    "val_x": x_val,
                    "val_y": y_val,
                    "val_base": b_val,
                    "extra_x": x_extra,
                    "extra_y": y_extra,
                    "extra_base": b_extra,
                    "extra_id": id_extra,
                    "curve_x": xs_curve,
                    "curve_y": y_curve,
                    "curve_val_x": xs_curve_val,
                    "curve_val_y": y_curve_val,
                }
            )

        is_p4_to_p5 = str(args.mode) == "p4_to_p5"
        panels_to_plot = panels

        _plot_period_panels(
            args.out_dir,
            task,
            panels_to_plot,
            str(args.scatter_style),
            plots_subdir="plots",
            show_k_badge=(not is_p4_to_p5),
            x_label_fontsize=(24 if is_p4_to_p5 else None),
            legend_loc=("upper left" if is_p4_to_p5 else None),
            base_scatter_scale=1.0,
            sharex=is_p4_to_p5,
            legend_axis_index=(0 if is_p4_to_p5 else -1),
            show_extra_family_legend=is_p4_to_p5,
            supxlabel_y=None,
        )


if __name__ == "__main__":
    main()

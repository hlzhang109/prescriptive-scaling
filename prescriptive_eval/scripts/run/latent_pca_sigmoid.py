#!/usr/bin/env python3
"""
Latent PCA Capability Factors + Sigmoid Prescriptive Scaling (Appendix Analysis)
------------------------------------------------------------------------------
This script computes low-dimensional *latent capability factors* by applying PCA
to a model×benchmark score matrix, then fits the same τ-quantile sigmoid frontier
vs pretraining compute on those latent factors.

Latent capability factors (PCA components)
----------------------------------------
Let each model i have a score vector over T tasks:
  x_i = [y_i(task_1), ..., y_i(task_T)]  ∈ R^T

We fit PCA on TRAIN PERIOD ONLY to obtain K orthonormal directions:
  W ∈ R^{T×K}, columns w_1,...,w_K.
The latent factor (PC) scores are:
  s_i = (x_i - μ) @ W            (or standardized: (x_i-μ)/σ before PCA)

Important implementation rules (reviewer-safe):
- PCA is fit on TRAIN PERIOD ONLY (P_k) and then applied to P_k and P_{k+1}
  to avoid temporal leakage across periods.
- Missingness is handled explicitly:
    * complete_case: drop rows with any missing task.
    * mean (default): impute missing entries using TRAIN means only.
- PCA sign is arbitrary. We align each PC so it correlates positively with an
  "overall capability" proxy (mean standardized task score) on the TRAIN period,
  making monotone scaling vs compute interpretable.

Sigmoid frontier on PCA factors
-------------------------------
The sigmoid τ-quantile optimizer in this repo expects targets in [0,1]. PCA
scores are unbounded, so we apply a monotone rescaling per split/component:
  q_low  = quantile(S_train[:,j], 0.01)
  q_high = quantile(S_train[:,j], 0.99)
  y_pc   = clip((S - q_low)/(q_high-q_low), 0, 1)

We then fit the standard sigmoid frontier (same fitter as manuscript figures)
on (compute, y_pc) and evaluate OOS on the next period.

Data sources / schema (paper pipeline compatibility)
----------------------------------------------------
- Default input CSV: tables/open_llm_leaderboard/open_llm_leaderboard_with_tokens.csv
- Task columns (OLL v2 raw tasks):
    "IFEval Raw", "BBH Raw", "MATH Lvl 5 Raw", "GPQA Raw", "MUSR Raw", "MMLU-PRO Raw"
- Compute proxy:
    compute_scaled = 6 * tokens(T) * params(B)    (units: 1e21 FLOPs)
    z_scaled       = log10(compute_scaled)
- Period assignment uses skill_frontier.core.sigmoid.PERIOD4_BOUNDS on the
  date column (new OLL uses "Upload To Hub Date").

Example:
  python scripts/run/latent_pca_sigmoid.py \\
    --input_csv tables/open_llm_leaderboard/open_llm_leaderboard_with_tokens.csv \\
    --out_dir outputs/latent_pca_sigmoid \\
    --missing mean --standardize 1 --pca_k 3 \\
    --tau 0.98 --kappa 50 --lambda_reg 1e-3 --seed 0
"""

from __future__ import annotations

import argparse
import csv as _csv
import math
import os
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

# Ensure repo root on sys.path when running as a script.
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from skill_frontier.core.period_utils import assign_period_index_period4_one_based as _assign_period_index_period4  # type: ignore
from skill_frontier.core.sigmoid import PERIOD4_BOUNDS, PERIOD4_SPLITS_SINGLE  # type: ignore
from skill_frontier.evaluation.binning import create_equal_mass_bins  # type: ignore
from skill_frontier.evaluation.common import fit_sigmoid_predictor, interpolate_curve  # type: ignore
from skill_frontier.evaluation.sensitivity_kappa_lambda import (  # type: ignore
    calibration_summary,
    compute_overlap_edges,
    mask_in_edges,
    pinball_mean,
)
from skill_frontier.io.csv_utils import (  # type: ignore
    compute_flops,
    detect_date_col,
    detect_oll_raw_tasks,
    maybe_scale_task_values,
    parse_year_month,
    read_csv_rows,
)
from skill_frontier.plotting.axis_formatting import apply_model_size_tick_multiplier, apply_pretraining_compute_tick_multiplier  # type: ignore
from skill_frontier.plotting.configs import frontier_1d as frontier_1d_cfg  # type: ignore
from skill_frontier.plotting.configs import mpl_rc as mpl_rc_cfg  # type: ignore
from skill_frontier.plotting.labels import PRETRAINING_COMPUTE_FLOPS_LABEL  # type: ignore
from skill_frontier.io.task_utils import parse_tasks_arg  # type: ignore
from skill_frontier.io.hf_utils import extract_hf_repo_from_model_html as _extract_hf_repo_from_model_html  # type: ignore
from skill_frontier.io.boolean_utils import is_true  # type: ignore

# Match overlay styling used in:
# outputs/sigmoid_size/no_split/pretrain_vs_posttrain/plots/overlay_BBH_Raw.png
from scripts.plot.figure4_main_paper import plot_fig4 as fig4  # type: ignore

def _extract_model_id(row: Dict[str, str]) -> str:
    eval_name = (row.get("eval_name", "") or "").strip()
    if eval_name:
        return eval_name
    model_sha = (row.get("Model sha", "") or "").strip()
    if model_sha:
        return model_sha
    model_html = row.get("Model", "") or ""
    model_repo = _extract_hf_repo_from_model_html(model_html)
    if model_repo:
        return model_repo
    return ""


def _is_official_pretrained_row(row: Dict[str, str]) -> bool:
    type_str = str(row.get("Type", "") or "").lower()
    is_pretrained = "pretrained" in type_str
    is_official_provider = is_true(row.get("Official Providers", "False"))
    return bool(is_pretrained and is_official_provider)


def _safe_float(value: object) -> float:
    try:
        if value is None:
            return float("nan")
        s = str(value).strip()
        if s in {"", "nan", "NaN"}:
            return float("nan")
        return float(s)
    except Exception:
        return float("nan")


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _normalize_period4_single_splits() -> List[Tuple[int, int, str]]:
    """Return [(train_idx, val_idx, split_name)] for P1→P2, P2→P3, P3→P4."""
    label_to_idx: Dict[str, int] = {}
    for i, bound in enumerate(PERIOD4_BOUNDS):
        label = bound[0] if len(bound) == 3 else f"k{i+1}"
        label_to_idx[str(label)] = int(i) + 1

    splits: List[Tuple[int, int, str]] = []
    for spec in PERIOD4_SPLITS_SINGLE:
        train_labels = spec.get("train_labels", [])
        val_label = spec.get("val_label", None)
        if not train_labels or val_label is None:
            continue
        train_idx = [label_to_idx[l] for l in train_labels if l in label_to_idx]
        val_idx = label_to_idx.get(str(val_label), None)
        if not train_idx or val_idx is None:
            continue
        k = int(train_idx[-1])
        splits.append((k, int(val_idx), f"P{k}→P{int(val_idx)}"))
    # Stable order by train period
    splits.sort(key=lambda t: t[0])
    return splits


@dataclass(frozen=True)
class SplitPCAResult:
    split_k: int
    split_name: str
    tasks: List[str]
    mu: np.ndarray
    sigma: np.ndarray
    components: np.ndarray  # (T, K)
    explained_variance_ratio: np.ndarray  # (K,)


def _fit_pca_train_only(
    *,
    X_train: np.ndarray,
    X_val: np.ndarray,
    standardize: bool,
    k: int,
) -> Tuple[SplitPCAResult, np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[bool]]:
    if X_train.ndim != 2 or X_val.ndim != 2:
        raise ValueError("X_train/X_val must be 2D")
    if X_train.shape[1] != X_val.shape[1]:
        raise ValueError("X_train/X_val must have the same number of task columns")

    mu = np.mean(X_train, axis=0)
    if standardize:
        sigma = np.std(X_train, axis=0)
        sigma = np.maximum(sigma, 1e-6)
        X_train_std = (X_train - mu) / sigma
        X_val_std = (X_val - mu) / sigma
    else:
        sigma = np.ones_like(mu)
        X_train_std = X_train - mu
        X_val_std = X_val - mu

    # SVD-based PCA on train only.
    U, S, Vt = np.linalg.svd(X_train_std, full_matrices=False)
    components = Vt.T[:, : int(k)]

    # Explained variance ratio from singular values.
    if X_train_std.shape[0] >= 2:
        evals = (S**2) / float(X_train_std.shape[0] - 1)
    else:
        evals = np.zeros_like(S)
    denom = float(np.sum(evals)) if np.isfinite(np.sum(evals)) else 0.0
    evr = (evals[: int(k)] / denom) if denom > 0 else np.full(int(k), np.nan, float)

    # Project to PC scores.
    S_train = X_train_std @ components
    S_val = X_val_std @ components

    # Sign alignment: corr(PC_j, overall_capability_train) >= 0.
    overall = np.mean(X_train_std, axis=1)
    flipped: List[bool] = []
    for j in range(int(k)):
        s = S_train[:, j]
        corr = float("nan")
        if np.std(s) > 1e-12 and np.std(overall) > 1e-12:
            corr = float(np.corrcoef(s, overall)[0, 1])
        flip = bool(np.isfinite(corr) and corr < 0.0)
        flipped.append(flip)
        if flip:
            components[:, j] *= -1.0
            S_train[:, j] *= -1.0
            S_val[:, j] *= -1.0

    return (
        SplitPCAResult(
            split_k=-1,  # caller fills
            split_name="",
            tasks=[],
            mu=mu.astype(float),
            sigma=sigma.astype(float),
            components=components.astype(float),
            explained_variance_ratio=evr.astype(float),
        ),
        X_train_std.astype(float),
        X_val_std.astype(float),
        S_train.astype(float),
        S_val.astype(float),
        flipped,
    )


def _scale_to_unit_interval(
    s_train: np.ndarray,
    s_val: np.ndarray,
    *,
    q_lo: float = 0.01,
    q_hi: float = 0.99,
) -> Tuple[np.ndarray, np.ndarray, float, float]:
    s_train = np.asarray(s_train, float)
    s_val = np.asarray(s_val, float)
    lo = float(np.quantile(s_train, float(q_lo)))
    hi = float(np.quantile(s_train, float(q_hi)))
    if not (np.isfinite(lo) and np.isfinite(hi) and hi > lo + 1e-12):
        raise ValueError("Degenerate PCA score quantiles; cannot rescale to [0,1].")
    y_train = np.clip((s_train - lo) / (hi - lo), 0.0, 1.0)
    y_val = np.clip((s_val - lo) / (hi - lo), 0.0, 1.0)
    return y_train.astype(float), y_val.astype(float), lo, hi


def _configure_matplotlib() -> None:
    try:
        import matplotlib as mpl  # type: ignore

        mpl.rcParams["font.family"] = mpl_rc_cfg.FONT_FAMILY
        mpl.rcParams["mathtext.fontset"] = mpl_rc_cfg.MATH_FONTSET
    except Exception:
        pass


def _overlay_reference_subplot_bbox() -> Tuple[float, float, float, float]:
    """Return the bbox of Figure 4's first subplot (left, bottom, width, height).

    We reuse this to ensure the latent PCA sigmoid plots match the physical axes
    size used by the overlay reference (and therefore match fonts/legend scaling).
    """

    try:
        import matplotlib as mpl  # type: ignore
        import matplotlib.pyplot as plt  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError("matplotlib is required for plotting") from e

    with mpl.rc_context():
        fig4._configure_rcparams()
        fig, axes = plt.subplots(1, 3, figsize=(fig4.FIG_WIDTH, fig4.FIG_HEIGHT), dpi=fig4.DPI)
        plt.subplots_adjust(left=0.08, right=0.955, bottom=0.18, top=0.96, wspace=0.52)
        bbox = axes[0].get_position()
        plt.close(fig)
    return (float(bbox.x0), float(bbox.y0), float(bbox.width), float(bbox.height))


def _compute_dynamic_ylim(y_points: np.ndarray, y_curves: Sequence[np.ndarray]) -> Tuple[float, float]:
    candidates: List[float] = []
    y_points = np.asarray(y_points, float)
    if y_points.size:
        candidates.extend([float(np.nanmin(y_points)), float(np.nanmax(y_points))])
    for y_curve in y_curves:
        y_curve = np.asarray(y_curve, float)
        if y_curve.size:
            candidates.extend([float(np.nanmin(y_curve)), float(np.nanmax(y_curve))])

    if not candidates or not np.all(np.isfinite(candidates)):
        return (0.0, 1.0)

    y_min = float(np.nanmin(candidates))
    y_max = float(np.nanmax(candidates))
    if not np.isfinite(y_min) or not np.isfinite(y_max) or y_min == y_max:
        return (0.0, 1.0)

    pad = 0.02 * max(1e-6, (y_max - y_min))
    return (y_min - pad, y_max + pad)


def _plot_pc_nosplit(
    *,
    out_path_base: str,
    x_all: np.ndarray,
    y_all: np.ndarray,
    x_pretrained: np.ndarray,
    y_pretrained: np.ndarray,
    x_post_curve: np.ndarray,
    y_post_curve: np.ndarray,
    x_pre_curve: np.ndarray,
    y_pre_curve: np.ndarray,
    ylabel: str,
    ax_bbox: Tuple[float, float, float, float],
    pad_inches: float = 0.08,
) -> None:
    """Plot no-split PC scaling with the Figure 4 overlay style + pretrained subset fit."""

    try:
        import matplotlib as mpl  # type: ignore
        import matplotlib.pyplot as plt  # type: ignore
        import matplotlib.ticker as mticker  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError("matplotlib is required for plotting") from e

    x_all = np.asarray(x_all, float)
    y_all = np.asarray(y_all, float)
    x_pretrained = np.asarray(x_pretrained, float)
    y_pretrained = np.asarray(y_pretrained, float)
    x_post_curve = np.asarray(x_post_curve, float)
    y_post_curve = np.asarray(y_post_curve, float)
    x_pre_curve = np.asarray(x_pre_curve, float)
    y_pre_curve = np.asarray(y_pre_curve, float)

    with mpl.rc_context():
        fig4._configure_rcparams()

        fig = plt.figure(figsize=(fig4.FIG_WIDTH, fig4.FIG_HEIGHT), dpi=fig4.DPI)
        ax = fig.add_axes(ax_bbox)

        fig4.style_axes(ax)
        ax.set_xscale("log")
        ax.set_xlim(fig4.XLIM_FLOPS)

        ax.set_xlabel(PRETRAINING_COMPUTE_FLOPS_LABEL, fontsize=10, fontweight="bold", labelpad=4)
        ax.set_ylabel(str(ylabel), fontsize=10, fontweight="bold", labelpad=4)

        ax.set_xticks(list(fig4.XTICKS_FLOPS))
        ax.set_xticklabels([f"$10^{{{int(np.log10(x))}}}$" for x in fig4.XTICKS_FLOPS])

        ax.yaxis.set_major_locator(mticker.MultipleLocator(0.2))
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f"))

        ax.tick_params(axis="both", which="minor", length=0)
        ax.minorticks_off()

        ax.scatter(
            x_all,
            y_all,
            s=fig4.STYLE["marker_size"] ** 2,
            alpha=fig4.STYLE["marker_alpha"],
            color=fig4.COLORS["points"],
            edgecolors=fig4.STYLE["marker_edgecolor"],
            linewidths=fig4.STYLE["marker_edgewidth"],
            zorder=2,
            rasterized=True,
        )
        if x_pretrained.size:
            ax.scatter(
                x_pretrained,
                y_pretrained,
                s=(fig4.STYLE["marker_size"] * 1.5) ** 2,
                alpha=fig4.STYLE["marker_alpha_pretrained"],
                color=fig4.COLORS["pretrained_points"],
                edgecolors=fig4.STYLE["marker_edgecolor"],
                linewidths=fig4.STYLE["marker_edgewidth_pretrained"],
                marker="^",
                zorder=3,
                rasterized=True,
            )

        if x_post_curve.size:
            ax.plot(
                x_post_curve,
                y_post_curve,
                linewidth=fig4.STYLE["line_width"],
                alpha=fig4.STYLE["line_alpha"],
                color=fig4.COLORS["smooth_98_posttrain"],
                linestyle="-",
                zorder=4,
            )
        if x_pre_curve.size:
            ax.plot(
                x_pre_curve,
                y_pre_curve,
                linewidth=fig4.STYLE["line_width"],
                alpha=fig4.STYLE["line_alpha"],
                color=fig4.COLORS["smooth_98_pretrain"],
                linestyle="--",
                dashes=(5, 3),
                zorder=4,
            )

        y_lo, y_hi = _compute_dynamic_ylim(y_all, [y_post_curve, y_pre_curve])
        ax.set_ylim(y_lo, y_hi)
        ax.set_title("")

        handles, labels = fig4._create_scatter_legend_handles()
        legend = ax.legend(
            handles,
            labels,
            loc="upper left",
            bbox_to_anchor=(0.02, 0.76),
            frameon=True,
            framealpha=0.95,
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

        fig.savefig(
            out_path_base + ".pdf",
            dpi=fig4.DPI,
            bbox_inches="tight",
            pad_inches=float(pad_inches),
            facecolor="white",
            edgecolor="none",
        )
        fig.savefig(
            out_path_base + ".png",
            dpi=fig4.DPI,
            bbox_inches="tight",
            pad_inches=float(pad_inches),
            facecolor="white",
            edgecolor="none",
        )
        plt.close(fig)


def _apply_frontier_style(ax, *, xlabel: str, ylabel: str, title: str) -> None:
    ax.set_xscale("log")
    ax.set_xlabel(xlabel, fontweight="bold", fontsize=frontier_1d_cfg.LABEL_FONTSIZE_X)
    if xlabel == PRETRAINING_COMPUTE_FLOPS_LABEL:
        apply_pretraining_compute_tick_multiplier(ax)
    ax.set_ylabel(ylabel, fontweight="bold", fontsize=frontier_1d_cfg.LABEL_FONTSIZE_Y)
    if title:
        ax.set_title(title, fontweight="bold", fontsize=frontier_1d_cfg.TITLE_FONTSIZE)
    ax.tick_params(
        axis="both",
        labelsize=frontier_1d_cfg.TICK_LABELSIZE,
        length=frontier_1d_cfg.TICK_LENGTH,
        width=frontier_1d_cfg.TICK_WIDTH,
        direction=frontier_1d_cfg.TICK_DIRECTION,
    )
    ax.yaxis.grid(
        True,
        which="major",
        linestyle=frontier_1d_cfg.GRID_MAJOR_LINESTYLE,
        linewidth=frontier_1d_cfg.GRID_MAJOR_LINEWIDTH,
        color=frontier_1d_cfg.GRID_MAJOR_COLOR,
        alpha=frontier_1d_cfg.GRID_MAJOR_ALPHA,
    )
    ax.yaxis.grid(
        True,
        which="minor",
        linestyle=frontier_1d_cfg.GRID_MINOR_LINESTYLE,
        linewidth=frontier_1d_cfg.GRID_MINOR_LINEWIDTH,
        color=frontier_1d_cfg.GRID_MINOR_COLOR,
        alpha=frontier_1d_cfg.GRID_MINOR_ALPHA,
    )
    ax.xaxis.grid(
        True,
        which="major",
        linestyle=frontier_1d_cfg.GRID_MAJOR_LINESTYLE,
        linewidth=frontier_1d_cfg.GRID_MAJOR_LINEWIDTH,
        color=frontier_1d_cfg.GRID_MAJOR_COLOR,
        alpha=frontier_1d_cfg.GRID_MAJOR_ALPHA,
    )
    ax.xaxis.grid(
        True,
        which="minor",
        linestyle=frontier_1d_cfg.GRID_MINOR_LINESTYLE,
        linewidth=frontier_1d_cfg.GRID_MINOR_LINEWIDTH,
        color=frontier_1d_cfg.GRID_MINOR_COLOR,
        alpha=frontier_1d_cfg.GRID_MINOR_ALPHA,
    )
    for spine in ax.spines.values():
        spine.set_linewidth(frontier_1d_cfg.SPINE_LINEWIDTH)
        spine.set_color(frontier_1d_cfg.SPINE_COLOR)


def _write_outputs_readme(
    *,
    out_dir: str,
    cmd_example: str,
    tasks: List[str],
    pca_k: int,
    standardize: bool,
    missing: str,
    tau: float,
    kappa: float,
    lambda_reg: float,
) -> None:
    text = f"""# Latent PCA Capability Factors + Sigmoid Prescriptive Scaling

This folder contains an **appendix analysis** connecting the repo's sigmoid frontier framework to
the “observational scaling laws” line of work that extracts **low-dimensional latent capability
factors** from benchmark score matrices (via PCA).

## Definition: latent capability factors (PCA)
For each model `i`, let `x_i ∈ R^T` be its vector of task scores over the task list below.
We fit PCA on the **training period only** to get orthonormal components `W ∈ R^(T×K)` and define
PC scores `s_i = (x_i - μ) @ W` (or standardized coordinates if enabled).

### Train-only protocol (no leakage)
For each split `P_k → P_(k+1)`:
1. Build `X_train` and `X_val` from the task scores for models in `P_k` and `P_(k+1)`.
2. Handle missingness using **train-only** statistics.
3. (Optional) Standardize using **train-only** `(μ, σ)`.
4. Fit PCA on `X_train` only; apply it to both `X_train` and `X_val`.
5. Align PCA signs so each PC correlates **positively** with average standardized score on train.

## Sigmoid scaling on PCA factors (prescriptive frontier)
The sigmoid frontier fitter expects targets in `[0,1]`. Since PCA scores are unbounded, we apply a
monotone rescaling per split/component using train quantiles:
`q_low = Q_0.01(S_train)`, `q_high = Q_0.99(S_train)`, then
`pc_scaled = clip((S - q_low)/(q_high-q_low), 0, 1)`.

We then fit the **same** τ-quantile sigmoid frontier vs pretraining compute on `pc_scaled`.

### No-split visualization (all periods combined)
For appendix-style single-panel visualizations that match `outputs/sigmoid/no_split/plots`, we also:
- fit PCA on **all periods combined** (using the same missingness handling and standardization),
- rescale PCs to `[0,1]` using global quantiles,
- fit a single sigmoid frontier on all models (no train/val split),
- and plot the resulting curve and points in `firebrick`.

## Configuration used
- Tasks: {", ".join(tasks)}
- `pca_k`: {int(pca_k)}
- `standardize`: {bool(standardize)}
- `missing`: `{missing}` (train-only imputation if `mean`)
- Sigmoid fit: `tau={tau}`, `kappa={kappa}`, `lambda_reg={lambda_reg}`

## Outputs
### Tables
- `tables/pca_scores_split_*.csv`: per-model PCA scores (raw + scaled) for each split.
- `tables/pca_loadings_by_split.csv`: PCA loadings (`W`) per split and component.
- `tables/pca_explained_variance_by_split.csv`: explained variance ratios per split.
- `tables/pca_oos_metrics_summary.csv`: OOS pinball + calibration summaries (PC1..PC{int(pca_k)}).

### Plots
- `plots/pca_explained_variance.(pdf|png)`: explained variance ratio for PC1..PC{int(pca_k)} (by split).
- `plots/pca_loadings_pc1_pc2_k3.(pdf|png)`: PC1/PC2 loadings (split k=3 by default).
- `plots/sigmoid_pca_pc*_all.(pdf|png)`: sigmoid scaling on PC1..PC{int(pca_k)} fit on **all periods combined** (no split).
- `plots/pca_pc1_oos_metrics.(pdf|png)`: OOS pinball + |coverage error| summary for PC1 across splits.

## How to regenerate
```bash
{cmd_example}
```

## How to include in paper
See `latex_appendix_snippet.tex` for ready-to-paste figure environments.
"""
    _ensure_dir(out_dir)
    with open(os.path.join(out_dir, "README.md"), "w") as f:
        f.write(text)


def _write_latex_snippet(*, out_dir: str) -> None:
    snippet = r"""% Appendix figures: latent PCA capability factors + sigmoid scaling
% (generated by scripts/run/latent_pca_sigmoid.py)

\begin{figure}[t]
  \centering
  \includegraphics[width=0.62\linewidth]{outputs/latent_pca_sigmoid/plots/pca_explained_variance.pdf}
  \caption{Explained variance ratios of PCA latent capability factors (PC1--PC3) by period split.}
  \label{fig:latent_pca_explained_variance}
\end{figure}

\begin{figure}[t]
  \centering
  \includegraphics[width=0.85\linewidth]{outputs/latent_pca_sigmoid/plots/pca_loadings_pc1_pc2_k3.pdf}
  \caption{PCA loadings for PC1 and PC2 on the six OLL v2 raw tasks (shown for split $k=3$).}
  \label{fig:latent_pca_loadings}
\end{figure}

\begin{figure}[t]
  \centering
  \includegraphics[width=0.62\linewidth]{outputs/latent_pca_sigmoid/plots/sigmoid_pca_pc1_all.pdf}
  \caption{Sigmoid prescriptive scaling on the first latent capability factor (PC1), fit on all periods combined.}
  \label{fig:latent_pca_pc1_scaling}
\end{figure}

\begin{figure}[t]
  \centering
  \includegraphics[width=\linewidth]{outputs/latent_pca_sigmoid/plots/pca_pc1_oos_metrics.pdf}
  \caption{Out-of-sample metrics across period shifts for PC1: smoothed pinball loss and absolute coverage error.}
  \label{fig:latent_pca_pc1_metrics}
\end{figure}
"""
    _ensure_dir(out_dir)
    with open(os.path.join(out_dir, "latex_appendix_snippet.tex"), "w") as f:
        f.write(snippet)


def _load_period4_dataframe(
    *,
    csv_path: str,
    tasks: Optional[List[str]],
    compute_product_cols: Tuple[str, str],
    compute_multiplier: float,
    include_official_pretrained: bool,
) -> pd.DataFrame:
    rows, headers = read_csv_rows(csv_path)
    if not rows:
        raise RuntimeError(f"No rows in {csv_path}")

    tasks_local = tasks or detect_oll_raw_tasks(headers)
    if not tasks_local:
        raise RuntimeError("No task columns detected.")
    for t in tasks_local:
        if t not in headers:
            raise RuntimeError(f"Task column {t!r} not found in CSV.")

    date_col = detect_date_col(headers)
    if date_col is None:
        raise RuntimeError("Could not detect date column (expected 'Upload To Hub Date' or 'date').")

    records: List[dict] = []
    for row in rows:
        compute_scaled = compute_flops(
            row,
            headers,
            logC_col=None,
            prod_cols=compute_product_cols,
            mult=float(compute_multiplier),
        )
        if not (np.isfinite(compute_scaled) and float(compute_scaled) > 0.0):
            continue

        ym = parse_year_month(str(row.get(date_col, "") or ""))
        if ym is None:
            continue
        period_idx = _assign_period_index_period4(int(ym[0]), int(ym[1]))
        if period_idx <= 0:
            continue

        model_id = _extract_model_id(row)
        if not model_id:
            continue

        is_official_pretrained = _is_official_pretrained_row(row)
        if (not include_official_pretrained) and is_official_pretrained:
            continue

        task_vals: Dict[str, float] = {t: _safe_float(row.get(t, None)) for t in tasks_local}
        if not any(np.isfinite(v) for v in task_vals.values()):
            continue

        records.append(
            {
                "model_id": model_id,
                "period_idx": int(period_idx),
                "compute_scaled": float(compute_scaled),
                "z_scaled": float(math.log10(float(compute_scaled))),
                "is_official_pretrained": bool(is_official_pretrained),
                **task_vals,
            }
        )

    df = pd.DataFrame.from_records(records)
    if df.empty:
        raise RuntimeError("No usable rows after filtering for compute + tasks + date.")

    # Auto-scale task values that look like percentages to [0,1].
    for t in tasks_local:
        arr = pd.to_numeric(df[t], errors="coerce").to_numpy(dtype=float)
        df[t] = maybe_scale_task_values(arr)

    return df.reset_index(drop=True)


def main(argv: Optional[Sequence[str]] = None) -> None:
    ap = argparse.ArgumentParser(description="Appendix: PCA latent capability factors + sigmoid scaling (period4).")
    ap.add_argument(
        "--input_csv",
        default=os.path.join("tables", "open_llm_leaderboard", "open_llm_leaderboard_with_tokens.csv"),
        help="Input OLL CSV (with_tokens schema).",
    )
    ap.add_argument(
        "--tasks",
        nargs="*",
        default=None,
        help="Task columns (default: auto-detect OLL Raw tasks). Accepts comma-separated list.",
    )
    ap.add_argument(
        "--compute_product_cols",
        nargs=2,
        default=("Pretraining tokens (T)", "#Params (B)"),
        metavar=("TOKENS_COL", "PARAMS_COL"),
    )
    ap.add_argument("--compute_multiplier", type=float, default=6.0)
    ap.add_argument("--tau", type=float, default=0.98)
    ap.add_argument("--kappa", type=float, default=50.0)
    ap.add_argument("--lambda_reg", type=float, default=1e-3)
    ap.add_argument("--pca_k", type=int, default=3)
    ap.add_argument("--standardize", type=int, default=1, help="1=standardize tasks on train (default), 0=center only.")
    ap.add_argument(
        "--missing",
        choices=("mean", "complete_case"),
        default="mean",
        help="Missingness handling: mean (train-only) or complete_case (drop rows with any missing task).",
    )
    ap.add_argument(
        "--period_split",
        choices=("period4",),
        default="period4",
        help="Period split protocol (currently only period4).",
    )
    ap.add_argument("--include_official_pretrained", action="store_true", help="Include official pretrained rows (default: excluded).")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out_dir", default=os.path.join("outputs", "latent_pca_sigmoid"))
    args = ap.parse_args(list(argv) if argv is not None else None)

    tasks = parse_tasks_arg(args.tasks)
    df = _load_period4_dataframe(
        csv_path=str(args.input_csv),
        tasks=tasks,
        compute_product_cols=(str(args.compute_product_cols[0]), str(args.compute_product_cols[1])),
        compute_multiplier=float(args.compute_multiplier),
        include_official_pretrained=bool(args.include_official_pretrained),
    )
    tasks = tasks or [c for c in detect_oll_raw_tasks(df.columns.tolist()) if c in df.columns]  # type: ignore[arg-type]
    if not tasks:
        raise RuntimeError("Could not determine tasks list after loading.")

    out_dir = str(args.out_dir)
    plots_dir = os.path.join(out_dir, "plots")
    tables_dir = os.path.join(out_dir, "tables")
    for d in (out_dir, plots_dir, tables_dir):
        _ensure_dir(d)

    splits = _normalize_period4_single_splits()
    if not splits:
        raise RuntimeError("Could not derive period4 single-k splits.")

    pca_loadings_rows: List[dict] = []
    pca_evr_rows: List[dict] = []
    metrics_rows: List[dict] = []

    missing_report: List[dict] = []

    for (k_train, k_val, split_name) in splits:
        df_tr = df[df["period_idx"] == int(k_train)].copy().reset_index(drop=True)
        df_va = df[df["period_idx"] == int(k_val)].copy().reset_index(drop=True)
        if df_tr.empty or df_va.empty:
            continue

        X_tr = df_tr[tasks].to_numpy(dtype=float)
        X_va = df_va[tasks].to_numpy(dtype=float)
        C_tr = df_tr["compute_scaled"].to_numpy(dtype=float)
        C_va = df_va["compute_scaled"].to_numpy(dtype=float)
        z_tr = df_tr["z_scaled"].to_numpy(dtype=float)
        z_va = df_va["z_scaled"].to_numpy(dtype=float)

        # Missingness handling (explicit, train-only for imputation).
        miss_tr = int(np.sum(~np.isfinite(X_tr)))
        miss_va = int(np.sum(~np.isfinite(X_va)))
        if str(args.missing) == "complete_case":
            m_tr = np.all(np.isfinite(X_tr), axis=1)
            m_va = np.all(np.isfinite(X_va), axis=1)
            df_tr = df_tr.loc[m_tr].reset_index(drop=True)
            df_va = df_va.loc[m_va].reset_index(drop=True)
            X_tr = X_tr[m_tr]
            X_va = X_va[m_va]
            C_tr = C_tr[m_tr]
            C_va = C_va[m_va]
            z_tr = z_tr[m_tr]
            z_va = z_va[m_va]
        else:
            means = np.nanmean(X_tr, axis=0)
            means = np.where(np.isfinite(means), means, 0.0)
            X_tr = np.where(np.isfinite(X_tr), X_tr, means.reshape(1, -1))
            X_va = np.where(np.isfinite(X_va), X_va, means.reshape(1, -1))

        missing_report.append(
            {
                "split": split_name,
                "train_period": int(k_train),
                "val_period": int(k_val),
                "train_n_raw": int((df[df["period_idx"] == int(k_train)]).shape[0]),
                "val_n_raw": int((df[df["period_idx"] == int(k_val)]).shape[0]),
                "train_missing_entries_raw": int(miss_tr),
                "val_missing_entries_raw": int(miss_va),
                "train_n_used": int(X_tr.shape[0]),
                "val_n_used": int(X_va.shape[0]),
            }
        )

        if X_tr.shape[0] < 3 or X_va.shape[0] < 3:
            continue

        k_pca = int(max(1, min(int(args.pca_k), X_tr.shape[1])))
        (
            pca_res,
            X_tr_std,
            X_va_std,
            S_tr,
            S_va,
            flipped,
        ) = _fit_pca_train_only(X_train=X_tr, X_val=X_va, standardize=bool(int(args.standardize)), k=k_pca)
        pca_res = SplitPCAResult(
            split_k=int(k_train),
            split_name=str(split_name),
            tasks=list(tasks),
            mu=pca_res.mu,
            sigma=pca_res.sigma,
            components=pca_res.components,
            explained_variance_ratio=pca_res.explained_variance_ratio,
        )

        # Save loadings + explained variance rows.
        for j in range(k_pca):
            pca_evr_rows.append(
                {
                    "split_k": int(k_train),
                    "split": str(split_name),
                    "component": int(j + 1),
                    "explained_variance_ratio": float(pca_res.explained_variance_ratio[j]),
                    "sign_flipped": int(bool(flipped[j])),
                }
            )
            row = {
                "split_k": int(k_train),
                "split": str(split_name),
                "component": int(j + 1),
                "explained_variance_ratio": float(pca_res.explained_variance_ratio[j]),
                "sign_flipped": int(bool(flipped[j])),
            }
            for t_i, t in enumerate(tasks):
                row[f"loading__{t}"] = float(pca_res.components[t_i, j])
            pca_loadings_rows.append(row)

        # Per-model scores table (train + val).
        scores_rows: List[dict] = []
        for split_tag, df_part, C_part, z_part, S_part in [
            ("train", df_tr, C_tr, z_tr, S_tr),
            ("val", df_va, C_va, z_va, S_va),
        ]:
            for i in range(df_part.shape[0]):
                row_i = df_part.iloc[int(i)]
                base = {
                    "model_id": str(row_i["model_id"]),
                    "split": str(split_tag),
                    "period_idx": int(row_i["period_idx"]),
                    "compute_scaled": float(C_part[i]),
                    "z_scaled": float(z_part[i]),
                    "is_official_pretrained": bool(row_i["is_official_pretrained"]),
                }
                for t in tasks:
                    v = _safe_float(row_i.get(t, float("nan")))
                    base[t] = float(v) if np.isfinite(v) else float("nan")
                for j in range(k_pca):
                    base[f"pc{j+1}_raw"] = float(S_part[i, j])
                scores_rows.append(base)
        df_scores = pd.DataFrame(scores_rows)

        # Rescale each PC to [0,1] using train quantiles, then fit sigmoid vs compute.
        for j in range(k_pca):
            try:
                y_tr_pc, y_va_pc, q_low, q_high = _scale_to_unit_interval(S_tr[:, j], S_va[:, j])
            except Exception:
                continue

            # Write scaled scores back to per-model table (for this component only).
            df_scores.loc[df_scores["split"] == "train", f"pc{j+1}_scaled"] = y_tr_pc
            df_scores.loc[df_scores["split"] == "val", f"pc{j+1}_scaled"] = y_va_pc

            xs_curve, y_curve = fit_sigmoid_predictor(
                C_tr,
                y_tr_pc,
                tau=float(args.tau),
                kappa_final=float(args.kappa),
                lambda_b=float(args.lambda_reg),
            )
            if xs_curve.size == 0:
                continue

            xs_curve_val = np.array([], dtype=float)
            y_curve_val = np.array([], dtype=float)
            if C_va.size >= 3:
                xs_curve_val, y_curve_val = fit_sigmoid_predictor(
                    C_va,
                    y_va_pc,
                    tau=float(args.tau),
                    kappa_final=float(args.kappa),
                    lambda_b=float(args.lambda_reg),
                )

            yhat_tr = interpolate_curve(xs_curve, y_curve, C_tr)
            yhat_va = interpolate_curve(xs_curve, y_curve, C_va)

            edges_tr = create_equal_mass_bins(z_tr, K=10, min_bin=30)
            edges_va = compute_overlap_edges(edges_tr, z_tr, z_va)
            m_va = mask_in_edges(z_va, edges_va) & np.isfinite(y_va_pc) & np.isfinite(yhat_va)
            pin_tr = float(pinball_mean(y_tr_pc, yhat_tr, tau=float(args.tau), kappa=float(args.kappa)))
            pin_va = float(pinball_mean(y_va_pc[m_va], yhat_va[m_va], tau=float(args.tau), kappa=float(args.kappa))) if np.any(m_va) else float("nan")
            calib_va = (
                calibration_summary(z_va[m_va], y_va_pc[m_va], yhat_va[m_va], edges=edges_va, tau=float(args.tau))
                if np.any(m_va)
                else None
            )

            metrics_rows.append(
                {
                    "split_k": int(k_train),
                    "split": str(split_name),
                    "component": int(j + 1),
                    "train_n": int(C_tr.size),
                    "val_n": int(C_va.size),
                    "val_n_overlap": int(np.sum(m_va)),
                    "tau": float(args.tau),
                    "kappa_eval": float(args.kappa),
                    "lambda_reg": float(args.lambda_reg),
                    "pc_q_low": float(q_low),
                    "pc_q_high": float(q_high),
                    "train_pinball": float(pin_tr),
                    "val_pinball": float(pin_va),
                    "val_calib_signed_micro": float(calib_va.signed_micro) if calib_va is not None else float("nan"),
                    "val_calib_abs_micro": float(calib_va.abs_micro) if calib_va is not None else float("nan"),
                    "explained_variance_ratio": float(pca_res.explained_variance_ratio[j]),
                    "sign_flipped": int(bool(flipped[j])),
                }
            )

        # Write per-split per-model table.
        out_scores_path = os.path.join(tables_dir, f"pca_scores_split_{int(k_train)}.csv")
        df_scores.to_csv(out_scores_path, index=False)

    # Save PCA metadata tables.
    pd.DataFrame(pca_loadings_rows).to_csv(os.path.join(tables_dir, "pca_loadings_by_split.csv"), index=False)
    pd.DataFrame(pca_evr_rows).to_csv(os.path.join(tables_dir, "pca_explained_variance_by_split.csv"), index=False)
    pd.DataFrame(metrics_rows).to_csv(os.path.join(tables_dir, "pca_oos_metrics_summary.csv"), index=False)
    pd.DataFrame(missing_report).to_csv(os.path.join(tables_dir, "missingness_summary.csv"), index=False)

    # ---------------- Plots ----------------
    _configure_matplotlib()
    import matplotlib.pyplot as plt  # type: ignore

    # P1) Explained variance ratios by split.
    evr_df = pd.DataFrame(pca_evr_rows)
    if not evr_df.empty:
        fig, ax = plt.subplots(figsize=(7.2, 4.2))
        for split_k in sorted(evr_df["split_k"].unique().tolist()):
            sub = evr_df[evr_df["split_k"] == split_k].sort_values("component")
            ax.plot(
                sub["component"].to_numpy(),
                sub["explained_variance_ratio"].to_numpy(),
                marker="o",
                linewidth=2.0,
                markersize=5,
                label=f"P{int(split_k)}→P{int(split_k)+1}",
            )
        ax.set_xlabel("PC index", fontweight="bold")
        ax.set_ylabel("Explained variance ratio", fontweight="bold")
        ax.set_title("PCA explained variance (train-only, by split)", fontweight="bold")
        ax.grid(True, which="major", alpha=0.25)
        ax.legend(loc="best", frameon=True, framealpha=0.2)
        fig.tight_layout()
        base = os.path.join(plots_dir, "pca_explained_variance")
        fig.savefig(base + ".png", dpi=300)
        fig.savefig(base + ".pdf", dpi=300)
        plt.close(fig)

    # P2) PCA loadings bar plot for PC1/PC2 on split k=3 (default).
    load_df = pd.DataFrame(pca_loadings_rows)
    if not load_df.empty and int(args.pca_k) >= 2:
        sub = load_df[load_df["split_k"] == 3].copy()
        if not sub.empty:
            pc1 = sub[sub["component"] == 1].iloc[0]
            pc2 = sub[sub["component"] == 2].iloc[0]
            vals1 = np.array([float(pc1[f"loading__{t}"]) for t in tasks], float)
            vals2 = np.array([float(pc2[f"loading__{t}"]) for t in tasks], float)
            x = np.arange(len(tasks))
            w = 0.38
            fig, ax = plt.subplots(figsize=(8.4, 4.6))
            ax.bar(x - w / 2, vals1, width=w, label="PC1", color="#1f77b4", alpha=0.9)
            ax.bar(x + w / 2, vals2, width=w, label="PC2", color="#ff7f0e", alpha=0.9)
            ax.axhline(0.0, color="#555555", linewidth=1.0)
            ax.set_xticks(x)
            ax.set_xticklabels([t.replace(" Raw", "") for t in tasks], rotation=20, ha="right")
            ax.set_ylabel("Loading", fontweight="bold")
            ax.set_title("PCA loadings (split k=3, train-only)", fontweight="bold")
            ax.legend(loc="best", frameon=True, framealpha=0.2)
            ax.grid(True, axis="y", which="major", alpha=0.25)
            fig.tight_layout()
            base = os.path.join(plots_dir, "pca_loadings_pc1_pc2_k3")
            fig.savefig(base + ".png", dpi=300)
            fig.savefig(base + ".pdf", dpi=300)
            plt.close(fig)

    # P3) No-split scaling plots: fit on all periods combined, matching `outputs/sigmoid/no_split/plots`.
    for pc_idx in range(1, int(args.pca_k) + 1):
        for ext in (".png", ".pdf"):
            try:
                os.remove(os.path.join(plots_dir, f"sigmoid_pca_pc{pc_idx}_periods{ext}"))
            except FileNotFoundError:
                pass
            except Exception:
                pass

    # Match overlay style (Figure 4) and additionally highlight + fit on official pretrained points.
    # Post-trained set = all non-official-pretrained rows.
    df_post = df.loc[~df["is_official_pretrained"].astype(bool)].copy().reset_index(drop=True)

    # Pretrained set = official-pretrained rows. If the main df excluded them, load a small view including them.
    df_pre = df.loc[df["is_official_pretrained"].astype(bool)].copy().reset_index(drop=True)
    if df_pre.empty:
        df_with_pre = _load_period4_dataframe(
            csv_path=str(args.input_csv),
            tasks=list(tasks),
            compute_product_cols=(str(args.compute_product_cols[0]), str(args.compute_product_cols[1])),
            compute_multiplier=float(args.compute_multiplier),
            include_official_pretrained=True,
        )
        df_pre = df_with_pre.loc[df_with_pre["is_official_pretrained"].astype(bool)].copy().reset_index(drop=True)

    # Prepare post-trained matrices (PCA + post-trained fit).
    X_post_raw = df_post[tasks].to_numpy(dtype=float)
    C_post_scaled = df_post["compute_scaled"].to_numpy(dtype=float)
    keep_post = np.isfinite(C_post_scaled) & (C_post_scaled > 0.0)
    X_post_raw = X_post_raw[keep_post]
    C_post_scaled = C_post_scaled[keep_post]

    if str(args.missing) == "complete_case":
        keep_cc = np.all(np.isfinite(X_post_raw), axis=1)
        X_post = X_post_raw[keep_cc].astype(float)
        C_post_scaled = C_post_scaled[keep_cc].astype(float)
        means = None
    else:
        X_post_nan = X_post_raw.astype(float, copy=True)
        X_post_nan[~np.isfinite(X_post_nan)] = np.nan
        means = np.nanmean(X_post_nan, axis=0)
        means = np.where(np.isfinite(means), means, 0.0)
        X_post = np.where(np.isfinite(X_post_raw), X_post_raw, means[None, :]).astype(float)
        C_post_scaled = C_post_scaled.astype(float)

    # Prepare official-pretrained matrices (project into post-trained PCA basis).
    X_pre_raw = df_pre[tasks].to_numpy(dtype=float)
    C_pre_scaled = df_pre["compute_scaled"].to_numpy(dtype=float)
    keep_pre = np.isfinite(C_pre_scaled) & (C_pre_scaled > 0.0)
    X_pre_raw = X_pre_raw[keep_pre]
    C_pre_scaled = C_pre_scaled[keep_pre]

    if str(args.missing) == "complete_case":
        keep_cc_pre = np.all(np.isfinite(X_pre_raw), axis=1)
        X_pre = X_pre_raw[keep_cc_pre].astype(float)
        C_pre_scaled = C_pre_scaled[keep_cc_pre].astype(float)
    else:
        # Impute using the post-trained means to keep the PCA basis comparable.
        if means is None:
            means = np.zeros((len(tasks),), dtype=float)
        X_pre = np.where(np.isfinite(X_pre_raw), X_pre_raw, means[None, :]).astype(float)
        C_pre_scaled = C_pre_scaled.astype(float)

    # Convert compute proxy (1e21 units) to FLOPs for plotting/fit to match overlay style.
    C_post = C_post_scaled * 1e21
    C_pre = C_pre_scaled * 1e21

    if X_post.shape[0] >= 3 and X_post.shape[1] >= 1:
        k_pca_all = int(max(1, min(int(args.pca_k), X_post.shape[1])))
        pca_all, _X_post_std, _X_post_std2, S_post, _S_post2, _flipped_all = _fit_pca_train_only(
            X_train=X_post,
            X_val=X_post,
            standardize=bool(int(args.standardize)),
            k=k_pca_all,
        )

        # Project official pretrained points into the same PCA basis.
        if X_pre.shape[0] >= 1:
            X_pre_std = (X_pre - pca_all.mu[None, :]) / pca_all.sigma[None, :]
            S_pre = (X_pre_std @ pca_all.components).astype(float)
        else:
            S_pre = np.zeros((0, int(k_pca_all)), dtype=float)

        ax_bbox = _overlay_reference_subplot_bbox()

        for j in range(int(k_pca_all)):
            try:
                y_post_pc, y_pre_pc, _q_low, _q_high = _scale_to_unit_interval(S_post[:, j], S_pre[:, j] if S_pre.size else np.array([], float))
            except Exception:
                continue

            m_fit_post = np.isfinite(C_post) & (C_post > 0.0) & np.isfinite(y_post_pc)
            if int(np.sum(m_fit_post)) < 3:
                continue

            xs_post_curve, y_post_curve = fit_sigmoid_predictor(
                C_post[m_fit_post],
                y_post_pc[m_fit_post],
                float(args.tau),
                kappa_final=float(args.kappa),
                lambda_b=float(args.lambda_reg),
            )
            if xs_post_curve.size == 0:
                continue

            xs_pre_curve = np.array([], dtype=float)
            y_pre_curve = np.array([], dtype=float)
            m_fit_pre = np.isfinite(C_pre) & (C_pre > 0.0) & np.isfinite(y_pre_pc)
            if int(np.sum(m_fit_pre)) >= 3:
                xs_pre_curve, y_pre_curve = fit_sigmoid_predictor(
                    C_pre[m_fit_pre],
                    y_pre_pc[m_fit_pre],
                    float(args.tau),
                    kappa_final=float(args.kappa),
                    lambda_b=float(args.lambda_reg),
                )

            x_all = np.concatenate([C_post[m_fit_post], C_pre[m_fit_pre]]) if int(np.sum(m_fit_pre)) else C_post[m_fit_post]
            y_all = np.concatenate([y_post_pc[m_fit_post], y_pre_pc[m_fit_pre]]) if int(np.sum(m_fit_pre)) else y_post_pc[m_fit_post]

            out_base = os.path.join(plots_dir, f"sigmoid_pca_pc{int(j+1)}_all")
            _plot_pc_nosplit(
                out_path_base=out_base,
                x_all=x_all,
                y_all=y_all,
                x_pretrained=C_pre[m_fit_pre],
                y_pretrained=y_pre_pc[m_fit_pre],
                x_post_curve=xs_post_curve,
                y_post_curve=y_post_curve,
                x_pre_curve=xs_pre_curve,
                y_pre_curve=y_pre_curve,
                ylabel=f"PC{int(j+1)} (scaled)",
                ax_bbox=ax_bbox,
            )

    # P4) Metrics summary plot (PC1, k=1..3).
    met_df = pd.DataFrame(metrics_rows)
    if not met_df.empty:
        pc1 = met_df[met_df["component"] == 1].sort_values("split_k")
        if not pc1.empty:
            x = pc1["split_k"].to_numpy(dtype=int)
            fig, ax = plt.subplots(figsize=(7.2, 4.2))
            ax.plot(x, pc1["val_pinball"].to_numpy(dtype=float), marker="o", linewidth=2.0, label="OOS pinball")
            ax.plot(x, pc1["val_calib_abs_micro"].to_numpy(dtype=float), marker="s", linewidth=2.0, label=r"OOS $|\hat{\tau}-\tau|$ (micro)")
            ax.set_xticks(x)
            ax.set_xticklabels([f"P{k}→P{k+1}" for k in x])
            ax.set_xlabel("Period shift", fontweight="bold")
            ax.set_ylabel("Metric value", fontweight="bold")
            ax.set_title("PC1 OOS metrics across period shifts", fontweight="bold")
            ax.grid(True, which="major", alpha=0.25)
            ax.legend(loc="best", frameon=True, framealpha=0.2)
            fig.tight_layout()
            base = os.path.join(plots_dir, "pca_pc1_oos_metrics")
            fig.savefig(base + ".png", dpi=300)
            fig.savefig(base + ".pdf", dpi=300)
            plt.close(fig)

    # Write explanation + LaTeX snippet into the output directory.
    cmd_example = (
        "python scripts/run/latent_pca_sigmoid.py "
        f"--input_csv {str(args.input_csv)} "
        f"--out_dir {out_dir} "
        f"--missing {str(args.missing)} --standardize {int(args.standardize)} --pca_k {int(args.pca_k)} "
        f"--tau {float(args.tau)} --kappa {float(args.kappa)} --lambda_reg {float(args.lambda_reg)} --seed {int(args.seed)}"
    )
    _write_outputs_readme(
        out_dir=out_dir,
        cmd_example=cmd_example,
        tasks=list(tasks),
        pca_k=int(args.pca_k),
        standardize=bool(int(args.standardize)),
        missing=str(args.missing),
        tau=float(args.tau),
        kappa=float(args.kappa),
        lambda_reg=float(args.lambda_reg),
    )
    _write_latex_snippet(out_dir=out_dir)

    # ---------------- Stdout summary ----------------
    print("[latent_pca_sigmoid] outputs:", out_dir)
    if missing_report:
        print("[latent_pca_sigmoid] split sizes:")
        for r in missing_report:
            print(
                f"  {r['split']}: train_n_used={r['train_n_used']} val_n_used={r['val_n_used']} "
                f"(raw train={r['train_n_raw']} val={r['val_n_raw']}; missing entries raw train={r['train_missing_entries_raw']} val={r['val_missing_entries_raw']})"
            )
    if not evr_df.empty:
        k3 = evr_df[evr_df["split_k"] == 3].sort_values("component")
        if not k3.empty:
            ratios = ", ".join([f"PC{int(r.component)}={float(r.explained_variance_ratio):.3f}" for r in k3.itertuples()])
            print("[latent_pca_sigmoid] explained variance (k=3):", ratios)


if __name__ == "__main__":
    main()

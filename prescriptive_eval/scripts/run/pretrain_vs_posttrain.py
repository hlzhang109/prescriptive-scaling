#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

# Import bootstrap from `scripts/` by putting the scripts dir on sys.path.
SCRIPTS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)
from _bootstrap import ensure_repo_root_on_path  # type: ignore

REPO_ROOT = ensure_repo_root_on_path(__file__)

from scipy.stats import spearmanr  # type: ignore

from skill_frontier.evaluation.binning import create_equal_mass_bins  # type: ignore
from skill_frontier.io.csv_utils import (  # type: ignore
    compute_flops,
    detect_oll_raw_tasks,
    maybe_scale_task_values,
    read_csv_rows,
)
from skill_frontier.io.output_paths import sanitize_task_name  # type: ignore
from skill_frontier.plotting.axis_formatting import (  # type: ignore
    apply_pretraining_compute_tick_multiplier,
    apply_model_size_tick_multiplier,
)
from skill_frontier.plotting.configs import frontier_1d as frontier_1d_cfg  # type: ignore
from skill_frontier.plotting.configs import mpl_rc as mpl_rc_cfg  # type: ignore
from skill_frontier.plotting.labels import (  # type: ignore
    MODEL_SIZE_PARAMS_LABEL,
    PRETRAINING_COMPUTE_FLOPS_LABEL,
)

from scripts.run.sigmoid_quantile_optimizer import (  # type: ignore
    fit_sigmoid_enhanced,
    sigmoid_pred,
)

X_LABEL: str = PRETRAINING_COMPUTE_FLOPS_LABEL
X_TICK_MULTIPLIER: float = 1e21
LOG10_X_TICK_MULTIPLIER: float = 21.0

TASK_COLORS: Dict[str, str] = {
    "BBH Raw": "#1f77b4",
    "GPQA Raw": "#ff7f0e",
    "IFEval Raw": "#2ca02c",
    "MATH Lvl 5 Raw": "#d62728",
    "MMLU-PRO Raw": "#9467bd",
    "MUSR Raw": "#8c564b",
}


TASK_GROUPS: Dict[str, List[str]] = {
    "knowledge": ["MMLU-PRO Raw", "GPQA Raw"],
    "reasoning": ["BBH Raw", "MATH Lvl 5 Raw", "MUSR Raw"],
    "instruction": ["IFEval Raw"],
}

_RAW_WORD_RE = re.compile(r"\bRaw\b")


def _strip_raw(text: str) -> str:
    out = _RAW_WORD_RE.sub("", str(text))
    out = re.sub(r"\s+", " ", out).strip()
    return out


from skill_frontier.io.boolean_utils import is_true  # type: ignore


def _is_official_pretrained_row(row: Dict[str, str]) -> bool:
    type_str = str(row.get("Type", "") or "").lower()
    is_pretrained = "pretrained" in type_str
    is_official_provider = is_true(row.get("Official Providers", "False"))
    return bool(is_pretrained and is_official_provider)


from skill_frontier.io.hf_utils import extract_hf_repo_from_model_html as _extract_hf_repo_from_model_html  # type: ignore


def _normalize_base_model_name(raw: str) -> str:
    s = (raw or "").strip()
    if not s or s.lower() in {"nan", "none", "removed"}:
        return ""
    return s.split(" (", 1)[0].strip()


def _extract_base_model_name(row: Dict[str, str]) -> str:
    identified = (row.get("Identified base model", "") or "").strip()
    if identified:
        return identified.strip().lower()
    base_model = _normalize_base_model_name(row.get("Base Model", "") or "")
    if base_model:
        return base_model
    model_html = row.get("Model", "") or ""
    model_repo = _extract_hf_repo_from_model_html(model_html)
    return _normalize_base_model_name(model_repo)


def _extract_model_name(row: Dict[str, str]) -> str:
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


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _prepare_out_dirs(out_dir: str) -> Dict[str, str]:
    plots_dir = os.path.join(out_dir, "plots")
    tables_dir = os.path.join(out_dir, "tables")
    fits_dir = os.path.join(out_dir, "fits")
    for d in (out_dir, plots_dir, tables_dir, fits_dir):
        _ensure_dir(d)
    return {"root": out_dir, "plots": plots_dir, "tables": tables_dir, "fits": fits_dir}


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


def load_analysis_dataframe(
    *,
    csv_path: str,
    compute_product_cols: Tuple[str, str],
    compute_multiplier: float,
    x_axis: str = "compute",
    size_col: str = "#Params (B)",
    tasks: Optional[List[str]] = None,
) -> pd.DataFrame:
    rows, headers = read_csv_rows(csv_path)
    if not rows:
        raise RuntimeError(f"No rows in {csv_path}")

    tasks_local = tasks or detect_oll_raw_tasks(headers)
    if not tasks_local:
        raise RuntimeError("No task columns detected.")

    records: List[dict] = []
    for row in rows:
        if str(x_axis) == "size":
            compute_scaled = _safe_float(row.get(size_col, None))
        else:
            compute_scaled = compute_flops(
                row,
                headers,
                logC_col=None,
                prod_cols=compute_product_cols,
                mult=float(compute_multiplier),
            )
        if not (np.isfinite(compute_scaled) and float(compute_scaled) > 0.0):
            continue
        compute_flops_abs = float(compute_scaled) * float(X_TICK_MULTIPLIER)

        params_b = _safe_float(row.get(size_col, None))
        tokens_t = _safe_float(row.get(str(compute_product_cols[0]), None))

        model_name = _extract_model_name(row)
        base_model_name = _extract_base_model_name(row)
        is_official_pretrained = _is_official_pretrained_row(row)

        for task in tasks_local:
            score = _safe_float(row.get(task, None))
            if not np.isfinite(score):
                continue
            records.append(
                {
                    "task": str(task),
                    "model_name": model_name,
                    "base_model_name": base_model_name,
                    "params_b": float(params_b),
                    "tokens_t": float(tokens_t),
                    "compute_scaled": float(compute_scaled),
                    "compute_flops": compute_flops_abs,
                    "z": float(math.log10(compute_flops_abs)),
                    "z_scaled": float(math.log10(float(compute_scaled))),
                    "score": float(score),
                    "is_official_pretrained": bool(is_official_pretrained),
                    "is_post_trained": bool(not is_official_pretrained),
                }
            )

    df = pd.DataFrame.from_records(records)
    if df.empty:
        raise RuntimeError("No usable rows after filtering for compute + scores.")

    for task in tasks_local:
        mask = df["task"] == task
        if not bool(mask.any()):
            continue
        df.loc[mask, "score"] = maybe_scale_task_values(df.loc[mask, "score"].to_numpy())

    df = df.dropna(subset=["compute_scaled", "score"]).reset_index(drop=True)
    df = df[(df["compute_scaled"] > 0.0) & np.isfinite(df["compute_scaled"])].reset_index(drop=True)
    return df


@dataclass(frozen=True)
class SigmoidFit:
    params: np.ndarray
    z_min: float
    z_max: float
    n: int
    tau: float
    lambda_b: float


def fit_sigmoid_params(
    z_scaled: np.ndarray,
    y: np.ndarray,
    *,
    tau: float,
    lambda_b: float,
) -> Optional[SigmoidFit]:
    z_scaled = np.asarray(z_scaled, float)
    y = np.asarray(y, float)
    keep = np.isfinite(z_scaled) & np.isfinite(y)
    z_scaled = z_scaled[keep]
    y = y[keep]
    if z_scaled.size < 3:
        return None

    result = fit_sigmoid_enhanced(
        z_scaled,
        y,
        tau=float(tau),
        kappa_final=50.0,
        lambda_b=float(lambda_b),
        n_zstar_grid=10,
        n_b_grid=10,
        n_random=100,
        seed=0,
    )
    if not bool(result.success) or not np.all(np.isfinite(result.params)):
        return None

    return SigmoidFit(
        params=np.asarray(result.params, float),
        z_min=float(np.min(z_scaled)),
        z_max=float(np.max(z_scaled)),
        n=int(z_scaled.size),
        tau=float(tau),
        lambda_b=float(lambda_b),
    )


def override_fit_range(fit: SigmoidFit, *, z_min: float, z_max: float) -> SigmoidFit:
    return SigmoidFit(
        params=np.asarray(fit.params, float),
        z_min=float(z_min),
        z_max=float(z_max),
        n=int(fit.n),
        tau=float(fit.tau),
        lambda_b=float(fit.lambda_b),
    )


def sample_sigmoid_curve(fit: SigmoidFit, *, grid_points: int = 400) -> Tuple[np.ndarray, np.ndarray]:
    z_grid = np.linspace(float(fit.z_min), float(fit.z_max), num=int(grid_points))
    y_hat = sigmoid_pred(fit.params, z_grid)
    y_hat = np.clip(y_hat, 0.0, 1.0)
    x_scaled = 10.0 ** z_grid
    return x_scaled.astype(float), y_hat.astype(float)


def predict_sigmoid(fit: SigmoidFit, z_scaled: np.ndarray) -> np.ndarray:
    z_scaled = np.asarray(z_scaled, float)
    y_hat = sigmoid_pred(fit.params, z_scaled)
    return np.clip(y_hat, 0.0, 1.0)


def write_fit_json(path: str, *, fit: Optional[SigmoidFit], task: str, subset: str, reason: str = "") -> None:
    payload: dict = {
        "task": str(task),
        "subset": str(subset),
        "x_tick_multiplier": X_TICK_MULTIPLIER,
        "z_scaled_offset": LOG10_X_TICK_MULTIPLIER,
    }
    if fit is None:
        payload.update({"status": "skipped", "reason": str(reason)})
    else:
        payload.update(
            {
                "status": "ok",
                "tau": float(fit.tau),
                "lambda_b": float(fit.lambda_b),
                "n": int(fit.n),
                "z_scaled_min": float(fit.z_min),
                "z_scaled_max": float(fit.z_max),
                "params": [float(x) for x in fit.params],
            }
        )
    with open(path, "w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def _apply_frontier_style(ax, *, xlabel: str, ylabel: str, title: str) -> None:
    ax.set_xscale("log")
    ax.set_xlabel(xlabel, fontweight="bold", fontsize=frontier_1d_cfg.LABEL_FONTSIZE_X)
    if xlabel == PRETRAINING_COMPUTE_FLOPS_LABEL:
        apply_pretraining_compute_tick_multiplier(ax)
    if xlabel == MODEL_SIZE_PARAMS_LABEL:
        apply_model_size_tick_multiplier(ax)
    ax.set_ylabel(ylabel, fontweight="bold", fontsize=frontier_1d_cfg.LABEL_FONTSIZE_Y)
    ax.set_title(title, fontweight="bold", fontsize=frontier_1d_cfg.TITLE_FONTSIZE)
    ax.tick_params(
        axis="both",
        labelsize=frontier_1d_cfg.TICK_LABELSIZE,
        length=frontier_1d_cfg.TICK_LENGTH,
        width=frontier_1d_cfg.TICK_WIDTH,
        direction=frontier_1d_cfg.TICK_DIRECTION,
    )
    try:
        import matplotlib.ticker as mticker  # type: ignore

        ax.yaxis.set_minor_locator(mticker.AutoMinorLocator(2))
        if ax.get_xscale() == "log":
            ax.xaxis.set_minor_locator(mticker.LogLocator(base=10.0, subs=(0.2, 0.4, 0.6, 0.8)))
    except Exception:
        pass

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


def plot_overlay(
    *,
    task: str,
    df_task: pd.DataFrame,
    fit_post: SigmoidFit,
    fit_pre: Optional[SigmoidFit],
    tau: float,
    out_plots_dir: str,
    post_points_label: str = "points",
    pre_points_label: str = "official pretrained",
    post_boundary_label: str = "Boundary (all)",
    pre_boundary_label: str = "Boundary (pretrained)",
    xlim_scaled: Optional[Tuple[float, float]] = None,
) -> None:
    import matplotlib as mpl  # type: ignore
    import matplotlib.pyplot as plt  # type: ignore

    mpl.rcParams["font.family"] = mpl_rc_cfg.FONT_FAMILY

    task_slug = sanitize_task_name(task)
    fig, ax = plt.subplots(figsize=frontier_1d_cfg.FIGSIZE)

    df_post_pts = df_task[df_task["is_post_trained"]].copy()
    df_pre_pts = df_task[df_task["is_official_pretrained"]].copy()

    ax.scatter(
        df_post_pts["compute_scaled"].to_numpy(),
        df_post_pts["score"].to_numpy(),
        s=frontier_1d_cfg.SCATTER_SIZE,
        alpha=frontier_1d_cfg.SCATTER_ALPHA,
        color="firebrick",
        label=str(post_points_label),
        linewidths=frontier_1d_cfg.SCATTER_LINEWIDTHS,
        rasterized=True,
    )
    if not df_pre_pts.empty:
        ax.scatter(
            df_pre_pts["compute_scaled"].to_numpy(),
            df_pre_pts["score"].to_numpy(),
            s=max(frontier_1d_cfg.SCATTER_SIZE * 2, frontier_1d_cfg.SCATTER_SIZE + 8),
            alpha=0.85,
            facecolors="none",
            edgecolors="#1f77b4",
            marker="^",
            linewidths=1.2,
            label=str(pre_points_label),
            rasterized=True,
            zorder=3,
        )

    x_post, y_post = sample_sigmoid_curve(fit_post)
    ax.plot(
        x_post,
        y_post,
        color="firebrick",
        linewidth=frontier_1d_cfg.CURVE_LINEWIDTH,
        label=str(post_boundary_label),
        alpha=frontier_1d_cfg.CURVE_ALPHA,
        zorder=4,
    )

    if fit_pre is not None:
        x_pre, y_pre = sample_sigmoid_curve(fit_pre)
        ax.plot(
            x_pre,
            y_pre,
            color="#1f77b4",
            linewidth=frontier_1d_cfg.CURVE_LINEWIDTH,
            linestyle="--",
            label=str(pre_boundary_label),
            alpha=frontier_1d_cfg.CURVE_ALPHA,
            zorder=4,
        )

    y_min = float(
        np.nanmin(
            [
                float(df_task["score"].min()) if not df_task.empty else np.nan,
                float(np.nanmin(y_post)) if y_post.size else np.nan,
            ]
        )
    )
    y_max = float(
        np.nanmax(
            [
                float(df_task["score"].max()) if not df_task.empty else np.nan,
                float(np.nanmax(y_post)) if y_post.size else np.nan,
            ]
        )
    )
    if not np.isfinite(y_min) or not np.isfinite(y_max) or y_min == y_max:
        y_min, y_max = 0.0, 1.0
    pad = 0.02 * max(1e-6, (y_max - y_min))
    ax.set_ylim(y_min - pad, y_max + pad)

    _apply_frontier_style(
        ax,
        xlabel=X_LABEL,
        ylabel="Accuracy",
        title="",
    )

    if xlim_scaled is not None:
        try:
            x_lo, x_hi = xlim_scaled
            if np.isfinite(x_lo) and np.isfinite(x_hi) and float(x_lo) > 0.0 and float(x_lo) < float(x_hi):
                ax.set_xlim(float(x_lo), float(x_hi))
        except Exception:
            pass

    leg = ax.legend(
        loc=frontier_1d_cfg.LEGEND_LOC,
        fontsize=frontier_1d_cfg.LEGEND_FONTSIZE,
        frameon=True,
        framealpha=frontier_1d_cfg.LEGEND_FRAMEALPHA,
    )
    if leg and leg.get_title():
        leg.get_title().set_fontweight("bold")
    fig.tight_layout()

    base = os.path.join(out_plots_dir, f"overlay_{task_slug}")
    fig.savefig(base + ".png", dpi=300)
    fig.savefig(base + ".pdf", dpi=300)
    plt.close(fig)


def _assign_bins(z_scaled: np.ndarray, edges: np.ndarray) -> np.ndarray:
    z_scaled = np.asarray(z_scaled, float)
    edges = np.asarray(edges, float)
    if edges.size < 2:
        return np.full(z_scaled.shape, -1, dtype=int)
    b = int(edges.size - 1)
    idx = np.searchsorted(edges, z_scaled, side="right") - 1
    return np.clip(idx, 0, b - 1)


def compute_gap_bins(
    *,
    df_pre: pd.DataFrame,
    fit_post: SigmoidFit,
    bins_gap: int,
) -> pd.DataFrame:
    z_pre = df_pre["z_scaled"].to_numpy()
    edges = create_equal_mass_bins(z_pre, int(bins_gap), 1)
    bin_idx = _assign_bins(z_pre, edges)
    gaps = predict_sigmoid(fit_post, z_pre) - df_pre["score"].to_numpy()

    out_rows: List[dict] = []
    for b in range(int(edges.size - 1)):
        mask = bin_idx == b
        if not bool(np.any(mask)):
            continue
        gaps_b = gaps[mask]
        z_lo = float(edges[b])
        z_hi = float(edges[b + 1])
        x_center = float(10.0 ** (0.5 * (z_lo + z_hi)))
        out_rows.append(
            {
                "bin": int(b),
                "z_lo_scaled": z_lo,
                "z_hi_scaled": z_hi,
                "x_center_scaled": x_center,
                "n": int(np.sum(mask)),
                "median_gap": float(np.nanmedian(gaps_b)),
                "mean_gap": float(np.nanmean(gaps_b)),
                "q25_gap": float(np.nanquantile(gaps_b, 0.25)),
                "q75_gap": float(np.nanquantile(gaps_b, 0.75)),
            }
        )
    return pd.DataFrame(out_rows)


def plot_gap(
    *,
    task: str,
    gap_bins: pd.DataFrame,
    out_plots_dir: str,
) -> None:
    import matplotlib as mpl  # type: ignore
    import matplotlib.pyplot as plt  # type: ignore

    mpl.rcParams["font.family"] = mpl_rc_cfg.FONT_FAMILY
    task_slug = sanitize_task_name(task)

    fig, ax = plt.subplots(figsize=frontier_1d_cfg.FIGSIZE)
    if not gap_bins.empty:
        x = gap_bins["x_center_scaled"].to_numpy()
        y = gap_bins["median_gap"].to_numpy()
        yerr_low = y - gap_bins["q25_gap"].to_numpy()
        yerr_high = gap_bins["q75_gap"].to_numpy() - y
        ax.errorbar(
            x,
            y,
            yerr=np.vstack([yerr_low, yerr_high]),
            fmt="o-",
            color="#1f77b4",
            linewidth=1.8,
            markersize=5,
            capsize=3,
            label="median gap (IQR)",
        )
    ax.axhline(0.0, color="#555555", linewidth=1.0, linestyle="--", alpha=0.8)

    _apply_frontier_style(
        ax,
        xlabel=X_LABEL,
        ylabel="Gap to boundary",
        title=f"Gap to Boundary — {_strip_raw(task)}",
    )
    ax.legend(loc="best", fontsize=frontier_1d_cfg.LEGEND_FONTSIZE, frameon=True, framealpha=frontier_1d_cfg.LEGEND_FRAMEALPHA)
    fig.tight_layout()

    base = os.path.join(out_plots_dir, f"gap_{task_slug}")
    fig.savefig(base + ".png", dpi=300)
    fig.savefig(base + ".pdf", dpi=300)
    plt.close(fig)


def plot_gap_all_tasks(
    *,
    gap_bins_by_task: Dict[str, pd.DataFrame],
    tasks_order: List[str],
    out_plots_dir: str,
) -> None:
    import matplotlib as mpl  # type: ignore
    import matplotlib.pyplot as plt  # type: ignore

    mpl.rcParams["font.family"] = mpl_rc_cfg.FONT_FAMILY

    fig, ax = plt.subplots(figsize=frontier_1d_cfg.FIGSIZE)
    for task in tasks_order:
        gap_bins = gap_bins_by_task.get(task)
        if gap_bins is None or gap_bins.empty:
            continue
        color = TASK_COLORS.get(task, None)
        x = gap_bins["x_center_scaled"].to_numpy()
        y = gap_bins["median_gap"].to_numpy()
        ax.plot(
            x,
            y,
            marker="o",
            linewidth=1.8,
            markersize=4,
            color=color,
            label=_strip_raw(task),
        )
        ax.fill_between(
            x,
            gap_bins["q25_gap"].to_numpy(),
            gap_bins["q75_gap"].to_numpy(),
            color=color,
            alpha=0.12,
            linewidth=0,
        )

    ax.axhline(0.0, color="#555555", linewidth=1.0, linestyle="--", alpha=0.8)
    _apply_frontier_style(
        ax,
        xlabel=X_LABEL,
        ylabel="Gap to boundary",
        title="Gap to Boundary (All Tasks)",
    )
    ax.legend(
        loc="best",
        fontsize=frontier_1d_cfg.LEGEND_FONTSIZE,
        frameon=True,
        framealpha=frontier_1d_cfg.LEGEND_FRAMEALPHA,
    )
    fig.tight_layout()

    base = os.path.join(out_plots_dir, "gap_all_tasks")
    fig.savefig(base + ".png", dpi=300)
    fig.savefig(base + ".pdf", dpi=300)
    plt.close(fig)


def compute_heterogeneity_bins(
    *,
    df_task: pd.DataFrame,
    bins_hetero: int,
) -> pd.DataFrame:
    z_all = df_task["z_scaled"].to_numpy()
    edges = create_equal_mass_bins(z_all, int(bins_hetero), 1)
    bin_idx = _assign_bins(z_all, edges)

    out_rows: List[dict] = []
    for b in range(int(edges.size - 1)):
        mask_bin = bin_idx == b
        if not bool(np.any(mask_bin)):
            continue
        z_lo = float(edges[b])
        z_hi = float(edges[b + 1])
        x_center = float(10.0 ** (0.5 * (z_lo + z_hi)))

        df_bin = df_task.loc[mask_bin]
        pre_scores = df_bin.loc[df_bin["is_official_pretrained"], "score"].to_numpy()
        post_scores = df_bin.loc[df_bin["is_post_trained"], "score"].to_numpy()

        def _iqr(values: np.ndarray) -> float:
            values = values[np.isfinite(values)]
            if values.size < 2:
                return float("nan")
            return float(np.nanquantile(values, 0.75) - np.nanquantile(values, 0.25))

        def _std(values: np.ndarray) -> float:
            values = values[np.isfinite(values)]
            if values.size < 2:
                return float("nan")
            return float(np.nanstd(values, ddof=1))

        out_rows.append(
            {
                "bin": int(b),
                "z_lo_scaled": z_lo,
                "z_hi_scaled": z_hi,
                "x_center_scaled": x_center,
                "n_pre": int(np.isfinite(pre_scores).sum()),
                "iqr_pre": _iqr(pre_scores),
                "std_pre": _std(pre_scores),
                "n_post": int(np.isfinite(post_scores).sum()),
                "iqr_post": _iqr(post_scores),
                "std_post": _std(post_scores),
            }
        )
    return pd.DataFrame(out_rows)


def plot_heterogeneity_iqr(
    *,
    task: str,
    hetero_bins: pd.DataFrame,
    out_plots_dir: str,
    post_label: str = "post-trained",
    pre_label: str = "pretrained",
) -> None:
    import matplotlib as mpl  # type: ignore
    import matplotlib.pyplot as plt  # type: ignore

    mpl.rcParams["font.family"] = mpl_rc_cfg.FONT_FAMILY
    task_slug = sanitize_task_name(task)

    fig, ax = plt.subplots(figsize=frontier_1d_cfg.FIGSIZE)
    if not hetero_bins.empty:
        x = hetero_bins["x_center_scaled"].to_numpy()
        ax.plot(x, hetero_bins["iqr_post"].to_numpy(), "o-", color="firebrick", linewidth=1.8, markersize=5, label=str(post_label))
        ax.plot(x, hetero_bins["iqr_pre"].to_numpy(), "o--", color="#1f77b4", linewidth=1.8, markersize=5, label=str(pre_label))

    _apply_frontier_style(
        ax,
        xlabel=X_LABEL,
        ylabel="Within-bin IQR",
        title=f"Heterogeneity (IQR) — {_strip_raw(task)}",
    )
    ax.legend(loc="best", fontsize=frontier_1d_cfg.LEGEND_FONTSIZE, frameon=True, framealpha=frontier_1d_cfg.LEGEND_FRAMEALPHA)
    fig.tight_layout()

    base = os.path.join(out_plots_dir, f"heterogeneity_iqr_by_bin_{task_slug}")
    fig.savefig(base + ".png", dpi=300)
    fig.savefig(base + ".pdf", dpi=300)
    plt.close(fig)


def plot_heterogeneity_iqr_all_tasks(
    *,
    hetero_bins_by_task: Dict[str, pd.DataFrame],
    tasks_order: List[str],
    out_plots_dir: str,
    post_label: str = "post-trained",
    pre_label: str = "pretrained",
) -> None:
    import matplotlib as mpl  # type: ignore
    import matplotlib.pyplot as plt  # type: ignore
    from matplotlib.lines import Line2D  # type: ignore

    mpl.rcParams["font.family"] = mpl_rc_cfg.FONT_FAMILY

    fig, ax = plt.subplots(figsize=frontier_1d_cfg.FIGSIZE)
    for task in tasks_order:
        hetero_bins = hetero_bins_by_task.get(task)
        if hetero_bins is None or hetero_bins.empty:
            continue
        color = TASK_COLORS.get(task, None)
        x = hetero_bins["x_center_scaled"].to_numpy()
        ax.plot(
            x,
            hetero_bins["iqr_post"].to_numpy(),
            color=color,
            linewidth=1.8,
            marker="o",
            markersize=3,
            label=_strip_raw(task),
        )
        ax.plot(
            x,
            hetero_bins["iqr_pre"].to_numpy(),
            color=color,
            linewidth=1.6,
            linestyle="--",
            marker="o",
            markersize=3,
            alpha=0.9,
            label="_nolegend_",
        )

    _apply_frontier_style(
        ax,
        xlabel=X_LABEL,
        ylabel="Within-bin IQR",
        title="Heterogeneity (IQR) (All Tasks)",
    )
    task_legend = ax.legend(
        loc="upper left",
        fontsize=frontier_1d_cfg.LEGEND_FONTSIZE,
        frameon=True,
        framealpha=frontier_1d_cfg.LEGEND_FRAMEALPHA,
        title="Task",
    )
    if task_legend and task_legend.get_title():
        task_legend.get_title().set_fontweight("bold")
    ax.add_artist(task_legend)

    style_handles = [
        Line2D([0], [0], color="#333333", linewidth=1.8, linestyle="-", label=str(post_label)),
        Line2D([0], [0], color="#333333", linewidth=1.8, linestyle="--", label=str(pre_label)),
    ]
    ax.legend(
        handles=style_handles,
        loc="upper right",
        fontsize=frontier_1d_cfg.LEGEND_FONTSIZE,
        frameon=True,
        framealpha=frontier_1d_cfg.LEGEND_FRAMEALPHA,
        title="Subset",
    )
    fig.tight_layout()

    base = os.path.join(out_plots_dir, "heterogeneity_iqr_by_bin_all_tasks")
    fig.savefig(base + ".png", dpi=300)
    fig.savefig(base + ".pdf", dpi=300)
    plt.close(fig)


def adjacent_decrease_rate(z: np.ndarray, y: np.ndarray, *, eps: float) -> float:
    z = np.asarray(z, float)
    y = np.asarray(y, float)
    keep = np.isfinite(z) & np.isfinite(y)
    z = z[keep]
    y = y[keep]
    if z.size < 2:
        return float("nan")
    order = np.lexsort((y, z))
    y_sorted = y[order]
    violations = (y_sorted[1:] + float(eps)) < y_sorted[:-1]
    return float(np.mean(violations)) if violations.size else float("nan")


def spearman_rho(z: np.ndarray, y: np.ndarray) -> float:
    z = np.asarray(z, float)
    y = np.asarray(y, float)
    keep = np.isfinite(z) & np.isfinite(y)
    z = z[keep]
    y = y[keep]
    if z.size < 3:
        return float("nan")
    rho, _ = spearmanr(z, y)
    try:
        return float(rho)
    except Exception:
        return float("nan")


def plot_task_bars(
    *,
    tasks: List[str],
    pre_values: List[float],
    post_values: List[float],
    ylabel: str,
    title: str,
    out_path_base: str,
    fig_size: Tuple[float, float] = frontier_1d_cfg.FIGSIZE,
    pre_label: str = "pretrained",
    post_label: str = "post-trained",
) -> None:
    import matplotlib as mpl  # type: ignore
    import matplotlib.pyplot as plt  # type: ignore

    mpl.rcParams["font.family"] = mpl_rc_cfg.FONT_FAMILY

    x_pos = np.arange(len(tasks), dtype=float)
    width = 0.38

    fig, ax = plt.subplots(figsize=fig_size)
    ax.bar(
        x_pos - width / 2,
        pre_values,
        width=width,
        facecolor="none",
        edgecolor="#1f77b4",
        linewidth=1.2,
        hatch="///",
        label=str(pre_label),
    )
    ax.bar(
        x_pos + width / 2,
        post_values,
        width=width,
        color="firebrick",
        alpha=0.85,
        label=str(post_label),
    )
    ax.set_xticks(x_pos)
    ax.set_xticklabels([_strip_raw(t) for t in tasks], rotation=25, ha="right", fontsize=frontier_1d_cfg.TICK_LABELSIZE)
    ax.set_ylabel(ylabel, fontweight="bold", fontsize=frontier_1d_cfg.LABEL_FONTSIZE_Y)
    ax.set_title(title, fontweight="bold", fontsize=frontier_1d_cfg.TITLE_FONTSIZE)
    ax.tick_params(axis="y", labelsize=frontier_1d_cfg.TICK_LABELSIZE)
    ax.yaxis.grid(True, which="major", linestyle=frontier_1d_cfg.GRID_MAJOR_LINESTYLE, linewidth=frontier_1d_cfg.GRID_MAJOR_LINEWIDTH, color=frontier_1d_cfg.GRID_MAJOR_COLOR, alpha=frontier_1d_cfg.GRID_MAJOR_ALPHA)
    for spine in ax.spines.values():
        spine.set_linewidth(frontier_1d_cfg.SPINE_LINEWIDTH)
        spine.set_color(frontier_1d_cfg.SPINE_COLOR)
    ax.legend(loc="best", fontsize=frontier_1d_cfg.LEGEND_FONTSIZE, frameon=True, framealpha=frontier_1d_cfg.LEGEND_FRAMEALPHA)
    fig.tight_layout()
    fig.savefig(out_path_base + ".png", dpi=300)
    fig.savefig(out_path_base + ".pdf", dpi=300)
    plt.close(fig)


def compute_lift_table(
    *,
    task: str,
    df_task: pd.DataFrame,
    min_derived_for_robust: int,
    robust_quantile: float,
) -> pd.DataFrame:
    df_pre = df_task[df_task["is_official_pretrained"]].copy()
    df_post = df_task[df_task["is_post_trained"]].copy()
    df_pre = df_pre[df_pre["base_model_name"].astype(str) != ""]
    df_post = df_post[df_post["base_model_name"].astype(str) != ""]
    if df_pre.empty or df_post.empty:
        return pd.DataFrame([])

    idx = df_pre.groupby("base_model_name")["score"].idxmax()
    bases = df_pre.loc[idx].copy()

    out_rows: List[dict] = []
    for _, base_row in bases.iterrows():
        base_name = str(base_row["base_model_name"])
        derived = df_post[df_post["base_model_name"] == base_name]
        if derived.empty:
            continue
        derived_scores = derived["score"].to_numpy()
        n_derived = int(np.isfinite(derived_scores).sum())
        if n_derived == 0:
            continue

        robust = n_derived >= int(min_derived_for_robust)
        if robust:
            post_best = float(np.nanquantile(derived_scores, float(robust_quantile)))
        else:
            post_best = float(np.nanmax(derived_scores))

        pre_score = float(base_row["score"])
        out_rows.append(
            {
                "task": str(task),
                "base_model_name": base_name,
                "compute_scaled": float(base_row["compute_scaled"]),
                "z_scaled": float(base_row["z_scaled"]),
                "pretrained_score": pre_score,
                "post_best_score": post_best,
                "n_derived": n_derived,
                "lift": float(post_best - pre_score),
                "lift_is_robust": bool(robust),
            }
        )

    return pd.DataFrame(out_rows)


def plot_lift(
    *,
    task: str,
    lift_df: pd.DataFrame,
    out_plots_dir: str,
) -> None:
    import matplotlib as mpl  # type: ignore
    import matplotlib.pyplot as plt  # type: ignore

    mpl.rcParams["font.family"] = mpl_rc_cfg.FONT_FAMILY
    task_slug = sanitize_task_name(task)

    fig, ax = plt.subplots(figsize=frontier_1d_cfg.FIGSIZE)
    if not lift_df.empty:
        robust_mask = lift_df["lift_is_robust"].astype(bool).to_numpy()
        ax.scatter(
            lift_df.loc[robust_mask, "compute_scaled"].to_numpy(),
            lift_df.loc[robust_mask, "lift"].to_numpy(),
            s=32,
            alpha=0.85,
            color="firebrick",
            label="robust (q0.95)",
            rasterized=True,
        )
        ax.scatter(
            lift_df.loc[~robust_mask, "compute_scaled"].to_numpy(),
            lift_df.loc[~robust_mask, "lift"].to_numpy(),
            s=32,
            alpha=0.85,
            facecolors="none",
            edgecolors="firebrick",
            linewidths=1.2,
            label="fallback (max)",
            rasterized=True,
        )
    ax.axhline(0.0, color="#555555", linewidth=1.0, linestyle="--", alpha=0.8)

    _apply_frontier_style(
        ax,
        xlabel=X_LABEL,
        ylabel="Post-training lift",
        title=f"Post-training Lift — {_strip_raw(task)}",
    )
    ax.legend(loc="best", fontsize=frontier_1d_cfg.LEGEND_FONTSIZE, frameon=True, framealpha=frontier_1d_cfg.LEGEND_FRAMEALPHA)
    fig.tight_layout()

    base = os.path.join(out_plots_dir, f"lift_{task_slug}")
    fig.savefig(base + ".png", dpi=300)
    fig.savefig(base + ".pdf", dpi=300)
    plt.close(fig)


def plot_lift_all_tasks(
    *,
    lift_by_task: Dict[str, pd.DataFrame],
    tasks_order: List[str],
    out_plots_dir: str,
) -> None:
    import matplotlib as mpl  # type: ignore
    import matplotlib.pyplot as plt  # type: ignore
    from matplotlib.lines import Line2D  # type: ignore

    mpl.rcParams["font.family"] = mpl_rc_cfg.FONT_FAMILY

    fig, ax = plt.subplots(figsize=frontier_1d_cfg.FIGSIZE)
    for task in tasks_order:
        lift_df = lift_by_task.get(task)
        if lift_df is None or lift_df.empty:
            continue
        color = TASK_COLORS.get(task, None)
        robust_mask = lift_df["lift_is_robust"].astype(bool).to_numpy()
        ax.scatter(
            lift_df.loc[robust_mask, "compute_scaled"].to_numpy(),
            lift_df.loc[robust_mask, "lift"].to_numpy(),
            s=28,
            alpha=0.85,
            color=color,
            rasterized=True,
            label=_strip_raw(task),
        )
        ax.scatter(
            lift_df.loc[~robust_mask, "compute_scaled"].to_numpy(),
            lift_df.loc[~robust_mask, "lift"].to_numpy(),
            s=28,
            alpha=0.85,
            facecolors="none",
            edgecolors=color,
            linewidths=1.1,
            rasterized=True,
            label="_nolegend_",
        )

    ax.axhline(0.0, color="#555555", linewidth=1.0, linestyle="--", alpha=0.8)
    _apply_frontier_style(
        ax,
        xlabel=X_LABEL,
        ylabel="Post-training lift",
        title="Post-training Lift (All Tasks)",
    )

    task_legend = ax.legend(
        loc="upper left",
        fontsize=frontier_1d_cfg.LEGEND_FONTSIZE,
        frameon=True,
        framealpha=frontier_1d_cfg.LEGEND_FRAMEALPHA,
        title="Task",
    )
    if task_legend and task_legend.get_title():
        task_legend.get_title().set_fontweight("bold")
    ax.add_artist(task_legend)

    style_handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor="#333333", markeredgecolor="#333333", markersize=6, label="robust (q0.95)"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor="none", markeredgecolor="#333333", markersize=6, label="fallback (max)"),
    ]
    ax.legend(
        handles=style_handles,
        loc="upper right",
        fontsize=frontier_1d_cfg.LEGEND_FONTSIZE,
        frameon=True,
        framealpha=frontier_1d_cfg.LEGEND_FRAMEALPHA,
        title="Lift estimator",
    )
    fig.tight_layout()

    base = os.path.join(out_plots_dir, "lift_all_tasks")
    fig.savefig(base + ".png", dpi=300)
    fig.savefig(base + ".pdf", dpi=300)
    plt.close(fig)


def compute_group_gap_summary(
    *,
    gap_bins_by_task: Dict[str, pd.DataFrame],
    groups: Dict[str, List[str]],
) -> pd.DataFrame:
    out_rows: List[dict] = []
    for group_name, group_tasks in groups.items():
        available = [t for t in group_tasks if t in gap_bins_by_task and not gap_bins_by_task[t].empty]
        if len(available) != len(group_tasks):
            continue
        min_bins = min(int(gap_bins_by_task[t].shape[0]) for t in available)
        for b in range(min_bins):
            centers = []
            medians = []
            for t in available:
                row = gap_bins_by_task[t].iloc[b]
                centers.append(float(row["x_center_scaled"]))
                medians.append(float(row["median_gap"]))
            if not centers:
                continue
            out_rows.append(
                {
                    "group": group_name,
                    "bin": int(b),
                    "x_center_scaled": float(10.0 ** float(np.mean(np.log10(centers)))),
                    "mean_median_gap": float(np.mean(medians)),
                }
            )
    return pd.DataFrame(out_rows)


def plot_group_gap_summary(
    *,
    summary_df: pd.DataFrame,
    out_plots_dir: str,
) -> None:
    import matplotlib as mpl  # type: ignore
    import matplotlib.pyplot as plt  # type: ignore

    mpl.rcParams["font.family"] = mpl_rc_cfg.FONT_FAMILY
    fig, ax = plt.subplots(figsize=frontier_1d_cfg.FIGSIZE)
    if not summary_df.empty:
        for group_name in sorted(summary_df["group"].unique().tolist()):
            sub = summary_df[summary_df["group"] == group_name].sort_values("bin")
            ax.plot(
                sub["x_center_scaled"].to_numpy(),
                sub["mean_median_gap"].to_numpy(),
                marker="o",
                linewidth=1.8,
                markersize=5,
                label=group_name,
            )
    ax.axhline(0.0, color="#555555", linewidth=1.0, linestyle="--", alpha=0.8)
    _apply_frontier_style(
        ax,
        xlabel=X_LABEL,
        ylabel="Mean gap to boundary",
        title="Summary Gap by Group",
    )
    ax.legend(loc="best", fontsize=frontier_1d_cfg.LEGEND_FONTSIZE, frameon=True, framealpha=frontier_1d_cfg.LEGEND_FRAMEALPHA)
    fig.tight_layout()
    base = os.path.join(out_plots_dir, "summary_gap_by_group")
    fig.savefig(base + ".png", dpi=300)
    fig.savefig(base + ".pdf", dpi=300)
    plt.close(fig)


def compute_group_lift_summary(
    *,
    lift_by_task: Dict[str, pd.DataFrame],
    groups: Dict[str, List[str]],
) -> pd.DataFrame:
    out_rows: List[dict] = []
    for group_name, group_tasks in groups.items():
        lifts: List[float] = []
        for t in group_tasks:
            df = lift_by_task.get(t)
            if df is None or df.empty:
                continue
            lifts.extend([float(x) for x in df["lift"].to_numpy() if np.isfinite(x)])
        if not lifts:
            continue
        out_rows.append(
            {
                "group": group_name,
                "n": int(len(lifts)),
                "mean_lift": float(np.mean(lifts)),
                "median_lift": float(np.median(lifts)),
            }
        )
    return pd.DataFrame(out_rows)


def plot_group_lift_summary(
    *,
    summary_df: pd.DataFrame,
    out_plots_dir: str,
) -> None:
    import matplotlib as mpl  # type: ignore
    import matplotlib.pyplot as plt  # type: ignore

    mpl.rcParams["font.family"] = mpl_rc_cfg.FONT_FAMILY
    fig, ax = plt.subplots(figsize=frontier_1d_cfg.FIGSIZE)
    if not summary_df.empty:
        summary_df = summary_df.sort_values("group")
        ax.bar(summary_df["group"].to_list(), summary_df["mean_lift"].to_numpy(), color="firebrick", alpha=0.85)
        ax.set_ylabel("Mean lift", fontweight="bold", fontsize=frontier_1d_cfg.LABEL_FONTSIZE_Y)
        ax.set_title("Summary Lift by Group", fontweight="bold", fontsize=frontier_1d_cfg.TITLE_FONTSIZE)
        ax.tick_params(axis="both", labelsize=frontier_1d_cfg.TICK_LABELSIZE)
        ax.yaxis.grid(True, which="major", linestyle=frontier_1d_cfg.GRID_MAJOR_LINESTYLE, linewidth=frontier_1d_cfg.GRID_MAJOR_LINEWIDTH, color=frontier_1d_cfg.GRID_MAJOR_COLOR, alpha=frontier_1d_cfg.GRID_MAJOR_ALPHA)
        for spine in ax.spines.values():
            spine.set_linewidth(frontier_1d_cfg.SPINE_LINEWIDTH)
            spine.set_color(frontier_1d_cfg.SPINE_COLOR)
    fig.tight_layout()
    base = os.path.join(out_plots_dir, "summary_lift_by_group")
    fig.savefig(base + ".png", dpi=300)
    fig.savefig(base + ".pdf", dpi=300)
    plt.close(fig)


def compute_overlap_z_scaled_range(*, df_below: pd.DataFrame, df_above: pd.DataFrame) -> Optional[Tuple[float, float]]:
    """Return a z_scaled range where both groups have support.

    Uses a robust intersection of the groups' ranges on z_scaled, trimming
    extreme tails to avoid single-point outliers dominating the overlap.
    """
    if df_below.empty or df_above.empty:
        return None

    z_below = pd.to_numeric(df_below.get("z_scaled", np.nan), errors="coerce")
    z_above = pd.to_numeric(df_above.get("z_scaled", np.nan), errors="coerce")
    z_below = z_below[np.isfinite(z_below.to_numpy())]
    z_above = z_above[np.isfinite(z_above.to_numpy())]
    if z_below.empty or z_above.empty:
        return None

    raw_lo = float(max(float(z_below.min()), float(z_above.min())))
    raw_hi = float(min(float(z_below.max()), float(z_above.max())))
    if not (np.isfinite(raw_lo) and np.isfinite(raw_hi) and raw_lo < raw_hi):
        return None

    def _trimmed_bounds(z: pd.Series, *, trim_quantile: float, min_trim_points: int) -> Tuple[float, float]:
        z_np = np.sort(z.to_numpy(dtype=float))
        z_np = z_np[np.isfinite(z_np)]
        if z_np.size < 2:
            return (float("nan"), float("nan"))
        k = max(int(min_trim_points), int(math.floor(float(trim_quantile) * float(z_np.size))))
        k = min(int(k), int((z_np.size - 1) // 2))
        return (float(z_np[k]), float(z_np[-1 - k]))

    # Trim a small amount from both tails to keep overlap in regions where both
    # groups have non-trivial support (e.g. avoid a single extreme outlier).
    trim_q = 0.01
    trim_min_pts = 5
    lo_below, hi_below = _trimmed_bounds(z_below, trim_quantile=trim_q, min_trim_points=trim_min_pts)
    lo_above, hi_above = _trimmed_bounds(z_above, trim_quantile=trim_q, min_trim_points=trim_min_pts)
    z_lo = float(max(lo_below, lo_above))
    z_hi = float(min(hi_below, hi_above))
    if np.isfinite(z_lo) and np.isfinite(z_hi) and z_lo < z_hi:
        return (float(z_lo), float(z_hi))

    # Fallback: simple min/max overlap.
    return (float(raw_lo), float(raw_hi))


def main(argv: Optional[Sequence[str]] = None) -> None:
    ap = argparse.ArgumentParser(description="Pretrain vs posttrain analysis suite (no_split)")
    ap.add_argument("--tau", type=float, default=0.98)
    ap.add_argument("--bins_gap", type=int, default=8)
    ap.add_argument("--bins_hetero", type=int, default=10)
    ap.add_argument("--eps_adjdec", type=float, default=0.01)
    ap.add_argument("--min_pretrained_fit", type=int, default=20)
    ap.add_argument("--min_derived_for_robust", type=int, default=20)
    ap.add_argument("--robust_quantile", type=float, default=0.95)
    default_out_dir = os.path.join("outputs", "sigmoid", "no_split", "pretrain_vs_posttrain")
    ap.add_argument("--out_dir", default=default_out_dir)
    ap.add_argument(
        "--compare",
        choices=("pretrain_vs_posttrain", "size_effect", "training_token_effect"),
        default="pretrain_vs_posttrain",
        help="Which comparison to run (default: pretrain_vs_posttrain).",
    )
    ap.add_argument(
        "--size_threshold_b",
        type=float,
        default=5.0,
        help="Size cutoff for --compare size_effect (models with #Params (B) < threshold).",
    )
    ap.add_argument(
        "--tokens_threshold_t",
        type=float,
        default=10.0,
        help="Pretraining tokens cutoff for --compare training_token_effect (rows with tokens(T) < threshold).",
    )
    ap.add_argument("--x_axis", choices=("compute", "size"), default="compute")
    ap.add_argument("--size_col", default="#Params (B)")

    ap.add_argument(
        "--csv",
        default=os.path.join("tables", "open_llm_leaderboard", "open_llm_leaderboard_with_tokens.csv"),
        help="Input OLL CSV (with_tokens schema).",
    )
    ap.add_argument(
        "--compute_product_cols",
        nargs=2,
        default=("Pretraining tokens (T)", "#Params (B)"),
        metavar=("TOKENS_COL", "PARAMS_COL"),
    )
    ap.add_argument("--compute_multiplier", type=float, default=6.0)
    ap.add_argument("--lambda_b", type=float, default=1e-3)
    ap.add_argument("--tasks", nargs="*", default=None)
    ap.add_argument(
        "--only",
        choices=("all", "overlay"),
        default="all",
        help="Generate only a subset of outputs (default: all).",
    )

    args = ap.parse_args(list(argv) if argv is not None else None)

    global X_LABEL, X_TICK_MULTIPLIER, LOG10_X_TICK_MULTIPLIER
    base_out_dir = os.path.join("outputs", "sigmoid", "no_split")
    if str(args.x_axis) == "size":
        X_LABEL = MODEL_SIZE_PARAMS_LABEL
        X_TICK_MULTIPLIER = 1e9
        LOG10_X_TICK_MULTIPLIER = 9.0
        base_out_dir = os.path.join("outputs", "sigmoid_size", "no_split")
    else:
        X_LABEL = PRETRAINING_COMPUTE_FLOPS_LABEL
        X_TICK_MULTIPLIER = 1e21
        LOG10_X_TICK_MULTIPLIER = 21.0

    default_out_dir_resolved = os.path.join(base_out_dir, "pretrain_vs_posttrain")
    # If user didn't override --out_dir, map it to the correct base dir (sigmoid vs sigmoid_size).
    if str(args.out_dir) == str(default_out_dir):
        args.out_dir = default_out_dir_resolved
    # For threshold comparisons, default output location is `outputs/sigmoid/no_split/<compare>`.
    if str(args.compare) != "pretrain_vs_posttrain" and str(args.out_dir) == str(default_out_dir_resolved):
        args.out_dir = os.path.join(base_out_dir, str(args.compare))

    out_dirs = _prepare_out_dirs(str(args.out_dir))

    include_lift = str(args.compare) == "pretrain_vs_posttrain"
    post_points_label = "points"
    post_boundary_label = "Boundary (all)"
    if str(args.compare) == "size_effect":
        thresh = float(args.size_threshold_b)
        pre_points_label = f"small models (<{thresh:g}B)"
        post_points_label = f"large models (>={thresh:g}B)"
        pre_boundary_label = "Boundary (small)"
        pre_bar_label = f"small (<{thresh:g}B)"
        post_bar_label = f"large (>={thresh:g}B)"
        fit_pre_suffix = "small"
        fit_post_suffix = "large"
        post_boundary_label = "Boundary (large)"
    elif str(args.compare) == "training_token_effect":
        thresh = float(args.tokens_threshold_t)
        thresh_disp = f"{thresh:g}T"
        pre_points_label = f"small pretraining tokens (<{thresh_disp})"
        post_points_label = f"large pretraining tokens (>={thresh_disp})"
        pre_boundary_label = "Boundary (small tokens)"
        pre_bar_label = f"tokens < {thresh_disp}"
        post_bar_label = f"tokens >= {thresh_disp}"
        fit_pre_suffix = "lowtokens"
        fit_post_suffix = "hightokens"
        post_boundary_label = "Boundary (large tokens)"
    else:
        pre_points_label = "official pretrained"
        pre_boundary_label = "Boundary (pretrained)"
        pre_bar_label = "pretrained"
        post_bar_label = "post-trained"
        fit_pre_suffix = "pre"
        fit_post_suffix = "post"
        post_points_label = "post-trained / derived"
        post_boundary_label = "Boundary (post-trained)"

    if str(args.only) == "all":
        # Remove stale fit JSONs (especially important when compare-mode suffixes change).
        for fname in list(os.listdir(out_dirs["fits"])):
            if not fname.endswith(".json"):
                continue
            try:
                os.remove(os.path.join(out_dirs["fits"], fname))
            except Exception:
                pass

        # Remove per-task plots that are now combined into single figures.
        for prefix in ("gap_", "heterogeneity_iqr_by_bin_", "lift_"):
            for fname in list(os.listdir(out_dirs["plots"])):
                if not fname.startswith(prefix):
                    continue
                if not (fname.endswith(".png") or fname.endswith(".pdf")):
                    continue
                try:
                    os.remove(os.path.join(out_dirs["plots"], fname))
                except Exception:
                    pass

    df = load_analysis_dataframe(
        csv_path=str(args.csv),
        compute_product_cols=(str(args.compute_product_cols[0]), str(args.compute_product_cols[1])),
        compute_multiplier=float(args.compute_multiplier),
        x_axis=str(args.x_axis),
        size_col=str(args.size_col),
        tasks=list(args.tasks) if args.tasks else None,
    )

    # Override the subset definition for alternate comparisons.
    if str(args.compare) == "size_effect":
        thresh = float(args.size_threshold_b)
        df["is_official_pretrained"] = (pd.to_numeric(df["params_b"], errors="coerce") < thresh).astype(bool)
        df["is_post_trained"] = (~df["is_official_pretrained"]).astype(bool)
    elif str(args.compare) == "training_token_effect":
        thresh = float(args.tokens_threshold_t)
        df["is_official_pretrained"] = (pd.to_numeric(df["tokens_t"], errors="coerce") < thresh).astype(bool)
        df["is_post_trained"] = (~df["is_official_pretrained"]).astype(bool)

    tasks = sorted(df["task"].unique().tolist())

    if str(args.only) == "overlay":
        for task in tasks:
            df_task_full = df[df["task"] == task].copy()

            df_task_fit = df_task_full
            fit_z_range: Optional[Tuple[float, float]] = None
            fit_xlim_scaled: Optional[Tuple[float, float]] = None
            if str(args.compare) in {"size_effect", "training_token_effect"}:
                df_below = df_task_full[df_task_full["is_official_pretrained"]].copy()
                df_above = df_task_full[~df_task_full["is_official_pretrained"]].copy()
                z_range = compute_overlap_z_scaled_range(df_below=df_below, df_above=df_above)
                if z_range is not None:
                    z_min, z_max = z_range
                    fit_z_range = (float(z_min), float(z_max))
                    fit_xlim_scaled = (float(10.0 ** float(z_min)), float(10.0 ** float(z_max)))
                    df_task_fit = df_task_full[
                        (df_task_full["z_scaled"] >= z_min) & (df_task_full["z_scaled"] <= z_max)
                    ].copy()

            df_post = df_task_fit[df_task_fit["is_post_trained"]]
            df_pre = df_task_fit[df_task_fit["is_official_pretrained"]]
            print(f"[{task}] n_post_fit={int(df_post.shape[0])} n_pre_fit={int(df_pre.shape[0])}")

            fit_post = fit_sigmoid_params(
                df_post["z_scaled"].to_numpy(),
                df_post["score"].to_numpy(),
                tau=float(args.tau),
                lambda_b=float(args.lambda_b),
            )
            if fit_post is None:
                raise RuntimeError(f"[{task}] post-trained fit failed unexpectedly.")

            pretrained_reason = ""
            if int(df_pre.shape[0]) < int(args.min_pretrained_fit):
                pretrained_reason = f"too_few_points(n={int(df_pre.shape[0])})"

            fit_pre = fit_sigmoid_params(
                df_pre["z_scaled"].to_numpy(),
                df_pre["score"].to_numpy(),
                tau=float(args.tau),
                lambda_b=float(args.lambda_b),
            )
            if fit_pre is None and pretrained_reason:
                print(f"[{task}] pretrained fit skipped ({pretrained_reason})")

            if fit_z_range is not None:
                z_min, z_max = fit_z_range
                fit_post = override_fit_range(fit_post, z_min=z_min, z_max=z_max)
                if fit_pre is not None:
                    fit_pre = override_fit_range(fit_pre, z_min=z_min, z_max=z_max)

            plot_overlay(
                task=task,
                df_task=df_task_full,
                fit_post=fit_post,
                fit_pre=fit_pre,
                tau=float(args.tau),
                out_plots_dir=out_dirs["plots"],
                post_points_label=post_points_label,
                pre_points_label=pre_points_label,
                post_boundary_label=post_boundary_label,
                pre_boundary_label=pre_boundary_label,
                xlim_scaled=fit_xlim_scaled,
            )

        print(f"{str(args.compare)}: done (overlay only)")
        print(f"tasks: {', '.join(tasks)}")
        return

    gap_bins_by_task: Dict[str, pd.DataFrame] = {}
    lift_by_task: Dict[str, pd.DataFrame] = {}
    hetero_by_task: Dict[str, pd.DataFrame] = {}

    hetero_summary_rows: List[dict] = []
    lift_corr_rows: List[dict] = []
    skipped_pretrained_fits: List[str] = []

    for task in tasks:
        df_task_full = df[df["task"] == task].copy()

        df_task = df_task_full
        fit_z_range: Optional[Tuple[float, float]] = None
        fit_xlim_scaled: Optional[Tuple[float, float]] = None
        if str(args.compare) in {"size_effect", "training_token_effect"}:
            df_below = df_task_full[df_task_full["is_official_pretrained"]].copy()
            df_above = df_task_full[~df_task_full["is_official_pretrained"]].copy()
            z_range = compute_overlap_z_scaled_range(df_below=df_below, df_above=df_above)
            if z_range is not None:
                z_min, z_max = z_range
                fit_z_range = (float(z_min), float(z_max))
                fit_xlim_scaled = (float(10.0 ** float(z_min)), float(10.0 ** float(z_max)))
                df_task = df_task_full[(df_task_full["z_scaled"] >= z_min) & (df_task_full["z_scaled"] <= z_max)].copy()

        df_post = df_task[df_task["is_post_trained"]]
        df_pre = df_task[df_task["is_official_pretrained"]]
        print(f"[{task}] n_post_fit={int(df_post.shape[0])} n_pre_fit={int(df_pre.shape[0])}")

        fit_post = fit_sigmoid_params(
            df_post["z_scaled"].to_numpy(),
            df_post["score"].to_numpy(),
            tau=float(args.tau),
            lambda_b=float(args.lambda_b),
        )
        if fit_post is None:
            raise RuntimeError(f"[{task}] post-trained fit failed unexpectedly.")

        fit_pre: Optional[SigmoidFit] = None
        pretrained_reason = ""
        if int(df_pre.shape[0]) < int(args.min_pretrained_fit):
            pretrained_reason = f"too_few_points(n={int(df_pre.shape[0])})"
        fit_pre = fit_sigmoid_params(
            df_pre["z_scaled"].to_numpy(),
            df_pre["score"].to_numpy(),
            tau=float(args.tau),
            lambda_b=float(args.lambda_b),
        )
        if fit_pre is None:
            skipped_pretrained_fits.append(f"{task} ({pretrained_reason or 'fit_failed'})")
            print(f"[{task}] pretrained fit skipped ({pretrained_reason or 'fit_failed'})")

        if fit_z_range is not None:
            z_min, z_max = fit_z_range
            fit_post = override_fit_range(fit_post, z_min=z_min, z_max=z_max)
            if fit_pre is not None:
                fit_pre = override_fit_range(fit_pre, z_min=z_min, z_max=z_max)

        write_fit_json(
            os.path.join(out_dirs["fits"], f"{sanitize_task_name(task)}_{fit_post_suffix}.json"),
            fit=fit_post,
            task=task,
            subset=fit_post_suffix,
        )
        write_fit_json(
            os.path.join(out_dirs["fits"], f"{sanitize_task_name(task)}_{fit_pre_suffix}.json"),
            fit=fit_pre,
            task=task,
            subset=fit_pre_suffix,
            reason=pretrained_reason,
        )

        plot_overlay(
            task=task,
            df_task=df_task_full,
            fit_post=fit_post,
            fit_pre=fit_pre,
            tau=float(args.tau),
            out_plots_dir=out_dirs["plots"],
            post_points_label=post_points_label,
            pre_points_label=pre_points_label,
            post_boundary_label=post_boundary_label,
            pre_boundary_label=pre_boundary_label,
            xlim_scaled=fit_xlim_scaled,
        )

        if not df_pre.empty:
            gap_bins = compute_gap_bins(df_pre=df_pre, fit_post=fit_post, bins_gap=int(args.bins_gap))
            gap_bins.to_csv(os.path.join(out_dirs["tables"], f"gap_bins_{sanitize_task_name(task)}.csv"), index=False)
            gap_bins_by_task[task] = gap_bins
        else:
            gap_bins_by_task[task] = pd.DataFrame([])

        hetero_bins = compute_heterogeneity_bins(df_task=df_task, bins_hetero=int(args.bins_hetero))
        hetero_bins.to_csv(os.path.join(out_dirs["tables"], f"heterogeneity_{sanitize_task_name(task)}.csv"), index=False)
        hetero_by_task[task] = hetero_bins

        def _mean_iqr(df_bins: pd.DataFrame, col: str) -> float:
            if df_bins.empty or col not in df_bins.columns:
                return float("nan")
            vals = df_bins[col].to_numpy()
            vals = vals[np.isfinite(vals)]
            return float(np.mean(vals)) if vals.size else float("nan")

        hetero_summary_rows.append(
            {
                "task": task,
                "n_pre": int(df_pre.shape[0]),
                "n_post": int(df_post.shape[0]),
                "spearman_pre": spearman_rho(df_pre["z"].to_numpy(), df_pre["score"].to_numpy()),
                "spearman_post": spearman_rho(df_post["z"].to_numpy(), df_post["score"].to_numpy()),
                "adjdec_pre": adjacent_decrease_rate(df_pre["z"].to_numpy(), df_pre["score"].to_numpy(), eps=float(args.eps_adjdec)),
                "adjdec_post": adjacent_decrease_rate(df_post["z"].to_numpy(), df_post["score"].to_numpy(), eps=float(args.eps_adjdec)),
                "mean_iqr_pre": _mean_iqr(hetero_bins, "iqr_pre"),
                "mean_iqr_post": _mean_iqr(hetero_bins, "iqr_post"),
            }
        )

        if include_lift:
            lift_df = compute_lift_table(
                task=task,
                df_task=df_task,
                min_derived_for_robust=int(args.min_derived_for_robust),
                robust_quantile=float(args.robust_quantile),
            )
            lift_df.to_csv(os.path.join(out_dirs["tables"], f"lift_{sanitize_task_name(task)}.csv"), index=False)
            lift_by_task[task] = lift_df

            if not lift_df.empty:
                lift_corr_rows.append(
                    {
                        "task": task,
                        "n_bases": int(lift_df.shape[0]),
                        "rho_pre": spearman_rho(lift_df["z_scaled"].to_numpy(), lift_df["pretrained_score"].to_numpy()),
                        "rho_postbest": spearman_rho(lift_df["z_scaled"].to_numpy(), lift_df["post_best_score"].to_numpy()),
                    }
                )

    hetero_summary_df = pd.DataFrame(hetero_summary_rows)
    hetero_summary_df.to_csv(os.path.join(out_dirs["tables"], "pretrain_posttrain_heterogeneity_summary.csv"), index=False)

    tasks_order = hetero_summary_df["task"].tolist()
    plot_gap_all_tasks(gap_bins_by_task=gap_bins_by_task, tasks_order=tasks_order, out_plots_dir=out_dirs["plots"])
    plot_heterogeneity_iqr_all_tasks(
        hetero_bins_by_task=hetero_by_task,
        tasks_order=tasks_order,
        out_plots_dir=out_dirs["plots"],
        post_label=post_bar_label,
        pre_label=pre_bar_label,
    )
    if include_lift:
        plot_lift_all_tasks(lift_by_task=lift_by_task, tasks_order=tasks_order, out_plots_dir=out_dirs["plots"])
    plot_task_bars(
        tasks=tasks_order,
        pre_values=[float(x) for x in hetero_summary_df["adjdec_pre"].to_numpy()],
        post_values=[float(x) for x in hetero_summary_df["adjdec_post"].to_numpy()],
        ylabel=fr"Adj. decrease rate ($\epsilon={float(args.eps_adjdec):.2f}$)",
        title="Monotonicity Violations by Task",
        out_path_base=os.path.join(out_dirs["plots"], "monotonicity_adjdec_by_task"),
        fig_size=frontier_1d_cfg.FIGSIZE,
        pre_label=pre_bar_label,
        post_label=post_bar_label,
    )
    plot_task_bars(
        tasks=tasks_order,
        pre_values=[float(x) for x in hetero_summary_df["spearman_pre"].to_numpy()],
        post_values=[float(x) for x in hetero_summary_df["spearman_post"].to_numpy()],
        ylabel=r"Spearman $\rho$",
        title="Spearman Correlation by Task",
        out_path_base=os.path.join(out_dirs["plots"], "spearman_by_task"),
        pre_label=pre_bar_label,
        post_label=post_bar_label,
    )

    if include_lift:
        lift_corr_df = pd.DataFrame(lift_corr_rows)
        lift_corr_df.to_csv(os.path.join(out_dirs["tables"], "lift_correlation_summary.csv"), index=False)
        if not lift_corr_df.empty:
            plot_task_bars(
                tasks=lift_corr_df["task"].tolist(),
                pre_values=[float(x) for x in lift_corr_df["rho_pre"].to_numpy()],
                post_values=[float(x) for x in lift_corr_df["rho_postbest"].to_numpy()],
                ylabel=r"Spearman $\rho$",
                title="Spearman: Pretrained vs Post-trained-best",
                out_path_base=os.path.join(out_dirs["plots"], "spearman_pre_vs_postbest_by_task"),
                pre_label=pre_bar_label,
                post_label=post_bar_label,
            )

    gap_group_df = compute_group_gap_summary(gap_bins_by_task=gap_bins_by_task, groups=TASK_GROUPS)
    gap_group_df.to_csv(os.path.join(out_dirs["tables"], "summary_gap_by_group.csv"), index=False)
    plot_group_gap_summary(summary_df=gap_group_df, out_plots_dir=out_dirs["plots"])

    if include_lift:
        lift_group_df = compute_group_lift_summary(lift_by_task=lift_by_task, groups=TASK_GROUPS)
        lift_group_df.to_csv(os.path.join(out_dirs["tables"], "summary_lift_by_group.csv"), index=False)
        plot_group_lift_summary(summary_df=lift_group_df, out_plots_dir=out_dirs["plots"])

    print(f"{str(args.compare)}: done")
    print(f"tasks: {', '.join(tasks)}")
    if skipped_pretrained_fits:
        print("subset fits skipped:", "; ".join(skipped_pretrained_fits))


if __name__ == "__main__":
    main()

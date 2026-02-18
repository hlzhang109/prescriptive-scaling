#!/usr/bin/env python3
"""
Create Figure 5 (main paper): compact 4-panel horizontal layout (k=3, k=4, New Models, New Families).

Data sources:
  - Base (old) models: tables/open_llm_leaderboard/open_llm_leaderboard_with_tokens.csv
  - New/P5 models: tables/open_llm_leaderboard/validation_leaderboard.csv +
                  tables/open_llm_leaderboard/new_eval_leaderboard.csv +
                  tables/top_models_by_base.csv

These are the same sources used to generate the plots under:
  outputs/sigmoid/period4/single_k/new_models/plots
  outputs/sigmoid/period4/single_k/new_models_p5/plots

Outputs:
  outputs/figures_main_paper/figure5_main_paper.pdf
  outputs/figures_main_paper/figure5_main_paper.png
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from typing import Dict, Optional, Sequence, Tuple

import numpy as np

try:
    import pandas as pd  # type: ignore
except Exception as e:  # pragma: no cover
    raise SystemExit("This script requires pandas.") from e

try:
    import matplotlib.pyplot as plt  # type: ignore
    from matplotlib.lines import Line2D  # type: ignore
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

from skill_frontier.core.sigmoid import PERIOD4_BOUNDS  # type: ignore  # noqa: E402
from skill_frontier.io.csv_utils import parse_year_month  # type: ignore  # noqa: E402
from skill_frontier.plotting.model_families import (  # type: ignore  # noqa: E402
    FAMILY_ORDER_WITH_GPT,
    color_for_family,
    family_from_base_model,
)
from skill_frontier.plotting.configs import frontier_period4_old_vs_new as period4_old_vs_new_cfg  # type: ignore  # noqa: E402
from scripts.smooth_single_skill_frontier import fit_sigmoid_frontier  # type: ignore  # noqa: E402

# -----------------------------------------------------------------------------
# CONFIG (match Figure 1 style)
# -----------------------------------------------------------------------------

FIG_WIDTH = 11.5
FIG_HEIGHT = 2.6
DPI = 300

XLIM_FLOPS = (1e21, 1e25)
XTICKS_FLOPS = [1e21, 1e22, 1e23, 1e24, 1e25]

COLORS = {
    # Keep Figure 1 palette for the fit/curve semantics.
    "frontier_fit": "#377eb8",
    "p4_frontier": "#377eb8",
    "p5_frontier": "#e41a1c",
}

STYLE = {
    "marker_size": 3.5,
    "marker_alpha": 0.6,
    "line_width": 2.0,
    "line_alpha": 0.9,
    "grid_alpha": 0.2,
    "grid_linestyle": ":",
    "grid_linewidth": 0.5,
    "spine_width": 0.8,
}

OUT_DIR = os.path.join(REPO_ROOT, "outputs", "figures_main_paper")
OUT_PDF = os.path.join(OUT_DIR, "figure5_main_paper.pdf")

OLL_CSV = os.path.join(REPO_ROOT, "tables", "open_llm_leaderboard", "open_llm_leaderboard_with_tokens.csv")
NEW_METRICS_CSV = os.path.join(REPO_ROOT, "tables", "open_llm_leaderboard", "validation_leaderboard.csv")
NEW_COMPUTE_CSV = os.path.join(REPO_ROOT, "tables", "open_llm_leaderboard", "new_eval_leaderboard.csv")
TOP_MODELS_CSV = os.path.join(REPO_ROOT, "tables", "top_models_by_base.csv")
EXTRA_LEADERBOARD_CSV = os.path.join(REPO_ROOT, "tables", "new_leaderboard_results_with_tokens.csv")
ADDITIONAL_POSTTRAIN_CSV = os.path.join(REPO_ROOT, "tables", "additional_post_trained_models.csv")

DEFAULT_TASK = "MATH Lvl 5 Raw"
MAIN_TASKS: Sequence[str] = (
    "IFEval Raw",
    "BBH Raw",
    "MATH Lvl 5 Raw",
    "GPQA Raw",
    "MUSR Raw",
    "MMLU-PRO Raw",
)

MAIN_TASK_MAP_OLL_TO_NEW: Dict[str, str] = {
    "IFEval Raw": "leaderboard_ifeval_inst_level_strict_acc_none",
    "BBH Raw": "leaderboard_bbh_acc_norm_none",
    "MATH Lvl 5 Raw": "leaderboard_math_hard_exact_match_none",
    "GPQA Raw": "leaderboard_gpqa_acc_norm_none",
    "MUSR Raw": "leaderboard_musr_acc_norm_none",
    "MMLU-PRO Raw": "leaderboard_mmlu_pro_acc_none",
}

# Frontier fit defaults (same as period4 overlay plots).
TAU = 0.98
LAMBDA_B = 1e-3
KAPPA_FINAL = 50.0
FIT_MODE = "quantile_per_point"

# Explicit panel-(d) exclusions requested for the main-paper Figure 5 rendering.
PANEL_D_EXCLUDED_MODEL_IDS = {
    "allenai/olmo-3-7b-think-sft",
}


def _configure_rcparams() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": DPI,
            "savefig.dpi": DPI,
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
    )


def _assign_period_index(bounds: Sequence[Tuple[str, Tuple[int, int], Tuple[int, int]]], ym: Tuple[int, int]) -> int:
    y, m = ym
    for idx, (_lab, (y_lo, m_lo), (y_hi, m_hi)) in enumerate(bounds):
        if (y, m) >= (y_lo, m_lo) and (y, m) <= (y_hi, m_hi):
            return idx
    return -1


def _make_period_ids_from_dates(date_series: pd.Series, *, after_p4_as_p5: bool) -> np.ndarray:
    out = np.full(shape=(len(date_series),), fill_value=-1, dtype=int)
    p4_end = PERIOD4_BOUNDS[-1][2]
    for i, v in enumerate(date_series.tolist()):
        if v is None or (isinstance(v, float) and math.isnan(v)):
            continue
        ym = parse_year_month(str(v))
        if ym is None:
            continue
        if after_p4_as_p5 and ym > p4_end:
            out[i] = 4
        else:
            out[i] = _assign_period_index(PERIOD4_BOUNDS, ym)
    return out


def _compute_old_compute_zflops(df_oll: pd.DataFrame) -> pd.Series:
    tokens = pd.to_numeric(df_oll.get("Pretraining tokens (T)", np.nan), errors="coerce")
    params = pd.to_numeric(df_oll.get("#Params (B)", np.nan), errors="coerce")
    return 6.0 * tokens * params


def load_old_oll(oll_csv: str, *, task: str) -> pd.DataFrame:
    df = pd.read_csv(oll_csv)
    out = pd.DataFrame()
    out["compute_zflops"] = _compute_old_compute_zflops(df)
    out["pid"] = _make_period_ids_from_dates(
        df.get("Upload To Hub Date", pd.Series([np.nan] * len(df))),
        after_p4_as_p5=False,
    )
    out[task] = pd.to_numeric(df.get(task, np.nan), errors="coerce")
    base = df.get("Base model family")
    if base is None:
        base = df.get("Identified base model")
    if base is None:
        base = df.get("Base Model")
    out["base_model"] = base if base is not None else np.nan
    return out


def load_new_models(metrics_csv: str, compute_csv: str, top_models_csv: str, *, task: str) -> pd.DataFrame:
    metric_col = MAIN_TASK_MAP_OLL_TO_NEW.get(task)
    if metric_col is None:
        raise ValueError(f"Unsupported task for new-models CSV: {task!r}")

    df_metrics = pd.read_csv(metrics_csv)
    df_compute = pd.read_csv(compute_csv, usecols=["model_id", "pretrain_compute_zflops"])
    # NOTE: validation_leaderboard.csv already contains `mapped_base_model`. We only need
    # `last_modified` here for period assignment; including `mapped_base_model` would
    # create a merge collision (pandas suffixes) and drop the unsuffixed column.
    df_meta = pd.read_csv(top_models_csv, usecols=["model_id", "last_modified"])

    df_new = df_metrics.merge(df_compute, on="model_id", how="left", validate="one_to_one")
    df_new = df_new.merge(df_meta, on="model_id", how="left", validate="one_to_one")

    out = pd.DataFrame()
    out["model_id"] = df_new.get("model_id", np.nan)
    out["compute_zflops"] = pd.to_numeric(df_new.get("pretrain_compute_zflops", np.nan), errors="coerce")
    out["pid"] = _make_period_ids_from_dates(
        df_new.get("last_modified", pd.Series([np.nan] * len(df_new))),
        after_p4_as_p5=True,
    )
    out[task] = pd.to_numeric(df_new.get(metric_col, np.nan), errors="coerce")
    out["base_model"] = df_new.get("mapped_base_model", np.nan)
    return out


def _load_extra_leaderboard(extra_csv: str, *, task: str) -> pd.DataFrame:
    metric_col = MAIN_TASK_MAP_OLL_TO_NEW.get(task)
    if metric_col is None:
        raise ValueError(f"Unsupported task for extra-leaderboard CSV: {task!r}")

    df = pd.read_csv(extra_csv)
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
    out[task] = pd.to_numeric(df.get(metric_col, np.nan), errors="coerce")
    return out


def _load_additional_post_trained_model_ids(csv_path: str) -> list[str]:
    df = pd.read_csv(csv_path, dtype=str)
    raw = [str(v).strip() for v in df.to_numpy().ravel().tolist()]
    out: list[str] = []
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
    s = str(base_model).strip().lower()
    if not s or s == "nan":
        return ""
    if "llama-3" in s or "llama3" in s:
        return "llama-3"
    if "qwen3" in s:
        return "qwen3"
    return s


def _p5_family_from_base_model(base_model: str) -> str:
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


def _family_labels(base_models: np.ndarray) -> np.ndarray:
    base_models = np.asarray(base_models, dtype=object)
    return np.asarray([family_from_base_model(str(v), include_gpt=True) for v in base_models.tolist()], dtype=object)


def _xy_for_pid(df: pd.DataFrame, *, task: str, pid: int) -> Tuple[np.ndarray, np.ndarray]:
    df_k = df[df["pid"] == int(pid)]
    x_z = pd.to_numeric(df_k.get("compute_zflops", np.nan), errors="coerce").to_numpy(dtype=float)
    y = pd.to_numeric(df_k.get(task, np.nan), errors="coerce").to_numpy(dtype=float)
    x = x_z * 1e21
    m = np.isfinite(x) & (x > 0.0) & np.isfinite(y)
    return x[m], y[m]


def _xyf_for_period(df: pd.DataFrame, *, task: str, k: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    pid = int(k) - 1
    df_k = df[df["pid"] == pid]
    x_z = pd.to_numeric(df_k.get("compute_zflops", np.nan), errors="coerce").to_numpy(dtype=float)
    y = pd.to_numeric(df_k.get(task, np.nan), errors="coerce").to_numpy(dtype=float)
    base = df_k.get("base_model", pd.Series([np.nan] * len(df_k))).astype(str).to_numpy(dtype=object)
    x = x_z * 1e21
    m = np.isfinite(x) & (x > 0.0) & np.isfinite(y)
    return x[m], y[m], _family_labels(base[m])


def _xyf_for_pid(df: pd.DataFrame, *, task: str, pid: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    df_k = df[df["pid"] == int(pid)]
    x_z = pd.to_numeric(df_k.get("compute_zflops", np.nan), errors="coerce").to_numpy(dtype=float)
    y = pd.to_numeric(df_k.get(task, np.nan), errors="coerce").to_numpy(dtype=float)
    base = df_k.get("base_model", pd.Series([np.nan] * len(df_k))).astype(str).to_numpy(dtype=object)
    x = x_z * 1e21
    m = np.isfinite(x) & (x > 0.0) & np.isfinite(y)
    return x[m], y[m], _family_labels(base[m])


def _fit_frontier(x_flops: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    return fit_sigmoid_frontier(
        x_flops,
        y,
        tau=float(TAU),
        use_log10_x=True,
        grid_points=400,
        fit_mode=str(FIT_MODE),
        bins_for_fit=120,
        min_bin_size_for_fit=30,
        bin_frontier_quantile=float(TAU),
        bin_trim_fraction=0.01,
        lambda_b=float(LAMBDA_B),
        kappa_final=float(KAPPA_FINAL),
    )


def apply_panel_style(ax) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(STYLE["spine_width"])
    ax.spines["bottom"].set_linewidth(STYLE["spine_width"])

    ax.grid(
        True,
        alpha=STYLE["grid_alpha"],
        linestyle=STYLE["grid_linestyle"],
        linewidth=STYLE["grid_linewidth"],
        zorder=0,
    )
    ax.tick_params(axis="both", which="major", length=4, width=0.8, direction="out")
    ax.tick_params(axis="both", which="minor", length=0)
    ax.set_facecolor("white")


def _configure_axis(ax, *, show_ylabel: bool) -> None:
    ax.set_xscale("log")
    ax.set_xlim(*XLIM_FLOPS)
    ax.set_xticks(XTICKS_FLOPS)
    ax.set_xticklabels([f"$10^{{{int(np.log10(x))}}}$" for x in XTICKS_FLOPS])
    if not show_ylabel:
        ax.tick_params(axis="y", labelleft=False)


def _panel_label(ax, text: str) -> None:
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


def _resolve_ylim(y_vals: Sequence[np.ndarray], *, default: Tuple[float, float]) -> Tuple[float, float]:
    ymin, ymax = default
    data = [arr for arr in y_vals if arr is not None and np.asarray(arr).size]
    if not data:
        return default
    dmin = float(np.nanmin([float(np.nanmin(arr)) for arr in data]))
    dmax = float(np.nanmax([float(np.nanmax(arr)) for arr in data]))
    if np.isfinite(dmin) and np.isfinite(dmax):
        ymin = min(ymin, dmin)
        ymax = max(ymax, dmax)
    pad = 0.03 * max(1e-6, (ymax - ymin))
    return max(0.0, ymin - pad), min(1.0, ymax + pad)


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Create Figure 5 (compact 4-panel horizontal).")
    parser.add_argument("--task", default=DEFAULT_TASK, choices=list(MAIN_TASKS), help="Task to plot.")
    parser.add_argument("--out", default=OUT_PDF, help="Output PDF path.")
    args = parser.parse_args(argv)

    _configure_rcparams()

    task = str(args.task)
    out_pdf = os.path.abspath(str(args.out))

    df_old = load_old_oll(OLL_CSV, task=task)
    df_new = load_new_models(NEW_METRICS_CSV, NEW_COMPUTE_CSV, TOP_MODELS_CSV, task=task)

    # Panels (a)/(b): old OLL ("train") vs newly-evaluated ("val") within the same period.
    x3_train, y3_train = _xy_for_pid(df_old, task=task, pid=2)
    x3_val, y3_val = _xy_for_pid(df_new, task=task, pid=2)
    cx3_train, cy3_train = _fit_frontier(x3_train, y3_train)
    cx3_val, cy3_val = _fit_frontier(x3_val, y3_val)

    x4_train, y4_train = _xy_for_pid(df_old, task=task, pid=3)
    x4_val, y4_val = _xy_for_pid(df_new, task=task, pid=3)
    cx4_train, cy4_train = _fit_frontier(x4_train, y4_train)
    cx4_val, cy4_val = _fit_frontier(x4_val, y4_val)

    x_p5, y_p5, fam_p5 = _xyf_for_pid(df_new, task=task, pid=4)
    cx_p5, cy_p5 = _fit_frontier(x_p5, y_p5)

    # Right panel of the period4 (p4_to_p5) plot: unseen P5 base families + additional post-trained models.
    df_p5 = df_new[df_new["pid"] == 4].copy()
    prev_bases = df_new[df_new["pid"].isin([0, 1, 2, 3])].get("base_model", pd.Series([], dtype=str)).astype(str)
    prev_base_set = {k for k in (_period_base_membership_key(b) for b in prev_bases.tolist()) if k}
    base_keys = df_p5.get("base_model", pd.Series([np.nan] * len(df_p5))).astype(str).map(_period_base_membership_key)
    df_p5_seen = df_p5.loc[base_keys.isin(prev_base_set)].copy()
    df_p5_unseen = df_p5.loc[~base_keys.isin(prev_base_set)].copy()

    # Extra overlay points (post-trained models) as in plot_period4_frontiers_old_vs_new.py.
    df_extra = _load_extra_leaderboard(EXTRA_LEADERBOARD_CSV, task=task)
    additional_ids = _load_additional_post_trained_model_ids(ADDITIONAL_POSTTRAIN_CSV)
    df_additional = df_extra[df_extra.get("model_id", pd.Series([], dtype=str)).astype(str).isin(additional_ids)].copy()
    val_ids_unseen = set(df_p5_unseen.get("model_id", pd.Series([], dtype=str)).astype(str).tolist())
    df_extra_right = df_additional[~df_additional.get("model_id", pd.Series([], dtype=str)).astype(str).isin(val_ids_unseen)].copy()
    if not df_extra_right.empty:
        extra_ids_norm = df_extra_right.get("model_id", pd.Series([np.nan] * len(df_extra_right))).astype(str).str.strip().str.lower()
        df_extra_right = df_extra_right.loc[~extra_ids_norm.isin(PANEL_D_EXCLUDED_MODEL_IDS)].copy()

    # Build arrays in FLOPs and precompute the right-panel family labels.
    unseen_x_z = pd.to_numeric(df_p5_unseen.get("compute_zflops", np.nan), errors="coerce").to_numpy(dtype=float)
    unseen_y = pd.to_numeric(df_p5_unseen.get(task, np.nan), errors="coerce").to_numpy(dtype=float)
    unseen_base = df_p5_unseen.get("base_model", pd.Series([np.nan] * len(df_p5_unseen))).astype(str).to_numpy(dtype=object)
    m_unseen = np.isfinite(unseen_x_z) & (unseen_x_z > 0.0) & np.isfinite(unseen_y)
    unseen_x = unseen_x_z[m_unseen] * 1e21
    unseen_y = unseen_y[m_unseen]
    unseen_family = np.asarray([_p5_family_from_base_model(b) for b in unseen_base[m_unseen].tolist()], dtype=object)

    seen_x_z = pd.to_numeric(df_p5_seen.get("compute_zflops", np.nan), errors="coerce").to_numpy(dtype=float)
    seen_y = pd.to_numeric(df_p5_seen.get(task, np.nan), errors="coerce").to_numpy(dtype=float)
    m_seen = np.isfinite(seen_x_z) & (seen_x_z > 0.0) & np.isfinite(seen_y)
    seen_x = seen_x_z[m_seen] * 1e21
    seen_y = seen_y[m_seen]

    cx_seen, cy_seen = _fit_frontier(seen_x, seen_y) if seen_x.size >= 3 else (np.array([]), np.array([]))

    extra_x_z = pd.to_numeric(df_extra_right.get("compute_zflops", np.nan), errors="coerce").to_numpy(dtype=float)
    extra_y = pd.to_numeric(df_extra_right.get(task, np.nan), errors="coerce").to_numpy(dtype=float)
    extra_id = df_extra_right.get("model_id", pd.Series([np.nan] * len(df_extra_right))).astype(str).to_numpy(dtype=object)
    m_extra = np.isfinite(extra_x_z) & (extra_x_z > 0.0) & np.isfinite(extra_y)
    extra_x = extra_x_z[m_extra] * 1e21
    extra_y = extra_y[m_extra]
    extra_family = np.asarray([_extra_family_from_model_id(mid) for mid in extra_id[m_extra].tolist()], dtype=object)

    ylim_k = _resolve_ylim(
        [y3_train, y3_val, y4_train, y4_val, cy3_train, cy3_val, cy4_train, cy4_val],
        default=(0.1, 0.7),
    )
    ylim_p4_to_p5 = _resolve_ylim(
        [y4_train, seen_y, unseen_y, extra_y, cy4_train, cy_seen],
        default=(0.0, 0.9),
    )

    fig, axes = plt.subplots(1, 4, figsize=(FIG_WIDTH, FIG_HEIGHT), dpi=DPI)
    plt.subplots_adjust(left=0.06, right=0.98, bottom=0.18, top=0.88, wspace=0.32)
    for ax in axes:
        apply_panel_style(ax)

    # Panel (a): k=3
    ax = axes[0]
    train_alpha = 0.20
    val_alpha = 0.35
    ax.scatter(
        x3_train,
        y3_train,
        s=STYLE["marker_size"] ** 2,
        alpha=train_alpha,
        color=COLORS["p5_frontier"],
        edgecolors="none",
        linewidths=0.0,
        zorder=2,
        rasterized=True,
    )
    ax.scatter(
        x3_val,
        y3_val,
        s=STYLE["marker_size"] ** 2,
        alpha=val_alpha,
        facecolors="none",
        edgecolors=COLORS["p4_frontier"],
        linewidths=0.9,
        marker="^",
        zorder=3,
        rasterized=True,
    )
    if cx3_train.size:
        ax.plot(
            cx3_train,
            cy3_train,
            linewidth=STYLE["line_width"],
            alpha=STYLE["line_alpha"],
            color=COLORS["p5_frontier"],
            linestyle="-",
            zorder=4,
        )
    if cx3_val.size:
        ax.plot(
            cx3_val,
            cy3_val,
            linewidth=STYLE["line_width"],
            alpha=STYLE["line_alpha"],
            color=COLORS["p4_frontier"],
            linestyle="--",
            dashes=(5, 3),
            zorder=4,
        )
    _configure_axis(ax, show_ylabel=True)
    ax.set_ylim(*ylim_k)
    ax.set_yticks([t for t in (0.0, 0.2, 0.4, 0.6, 0.8) if ylim_k[0] - 1e-6 <= t <= ylim_k[1] + 1e-6])
    ax.set_xlabel("Pretraining Compute (FLOPs)", fontsize=12, fontweight="bold", labelpad=4)
    ax.set_ylabel("Accuracy", fontsize=14, fontweight="bold", labelpad=4)
    _panel_label(ax, r"(a) $t = 3$")
    ax.legend(
        handles=[
            Line2D(
                [0],
                [0],
                marker="o",
                linestyle="None",
                markersize=5.5,
                markerfacecolor=COLORS["p5_frontier"],
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
                markeredgecolor=COLORS["p4_frontier"],
                markeredgewidth=0.9,
                alpha=val_alpha,
                label="val",
            ),
            Line2D([0], [0], color=COLORS["p5_frontier"], linewidth=STYLE["line_width"], label="fit (train)"),
            Line2D(
                [0],
                [0],
                color=COLORS["p4_frontier"],
                linewidth=STYLE["line_width"],
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

    # Panel (b): k=4
    ax = axes[1]
    ax.scatter(
        x4_train,
        y4_train,
        s=STYLE["marker_size"] ** 2,
        alpha=train_alpha,
        color=COLORS["p5_frontier"],
        edgecolors="none",
        linewidths=0.0,
        zorder=2,
        rasterized=True,
    )
    ax.scatter(
        x4_val,
        y4_val,
        s=STYLE["marker_size"] ** 2,
        alpha=val_alpha,
        facecolors="none",
        edgecolors=COLORS["p4_frontier"],
        linewidths=0.9,
        marker="^",
        zorder=3,
        rasterized=True,
    )
    if cx4_train.size:
        ax.plot(
            cx4_train,
            cy4_train,
            linewidth=STYLE["line_width"],
            alpha=STYLE["line_alpha"],
            color=COLORS["p5_frontier"],
            linestyle="-",
            zorder=4,
        )
    if cx4_val.size:
        ax.plot(
            cx4_val,
            cy4_val,
            linewidth=STYLE["line_width"],
            alpha=STYLE["line_alpha"],
            color=COLORS["p4_frontier"],
            linestyle="--",
            dashes=(5, 3),
            zorder=4,
        )
    _configure_axis(ax, show_ylabel=False)
    ax.set_ylim(*ylim_k)
    ax.set_yticks([t for t in (0.0, 0.2, 0.4, 0.6, 0.8) if ylim_k[0] - 1e-6 <= t <= ylim_k[1] + 1e-6])
    ax.set_yticklabels([])
    ax.set_xlabel("Pretraining Compute (FLOPs)", fontsize=12, fontweight="bold", labelpad=4)
    _panel_label(ax, r"(b) $t = 4$")
    ax.legend(
        handles=[
            Line2D(
                [0],
                [0],
                marker="o",
                linestyle="None",
                markersize=5.5,
                markerfacecolor=COLORS["p5_frontier"],
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
                markeredgecolor=COLORS["p4_frontier"],
                markeredgewidth=0.9,
                alpha=val_alpha,
                label="val",
            ),
            Line2D([0], [0], color=COLORS["p5_frontier"], linewidth=STYLE["line_width"], label="fit (train)"),
            Line2D(
                [0],
                [0],
                color=COLORS["p4_frontier"],
                linewidth=STYLE["line_width"],
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

    # Panel (c): p4_to_p5 (left subplot): old P4 ("train") vs P5 models on seen base families ("val").
    ax = axes[2]
    ax.scatter(
        x4_train,
        y4_train,
        s=STYLE["marker_size"] ** 2,
        alpha=train_alpha,
        color=COLORS["p5_frontier"],
        edgecolors="none",
        linewidths=0.0,
        zorder=2,
        rasterized=True,
    )
    ax.scatter(
        seen_x,
        seen_y,
        s=STYLE["marker_size"] ** 2,
        alpha=val_alpha,
        facecolors="none",
        edgecolors=COLORS["p4_frontier"],
        linewidths=0.9,
        marker="^",
        zorder=3,
        rasterized=True,
    )
    if cx4_train.size:
        ax.plot(
            cx4_train,
            cy4_train,
            linewidth=STYLE["line_width"],
            alpha=STYLE["line_alpha"],
            color=COLORS["p5_frontier"],
            linestyle="-",
            zorder=4,
        )
    if cx_seen.size:
        ax.plot(
            cx_seen,
            cy_seen,
            linewidth=STYLE["line_width"],
            alpha=STYLE["line_alpha"],
            color=COLORS["p4_frontier"],
            linestyle="--",
            dashes=(5, 3),
            zorder=4,
        )
    _configure_axis(ax, show_ylabel=False)
    ax.set_ylim(*ylim_p4_to_p5)
    ax.set_yticks([t for t in (0.0, 0.2, 0.4, 0.6, 0.8) if ylim_p4_to_p5[0] - 1e-6 <= t <= ylim_p4_to_p5[1] + 1e-6])
    ax.set_yticklabels([])
    ax.set_xlabel("Pretraining Compute (FLOPs)", fontsize=12, fontweight="bold", labelpad=4)
    _panel_label(ax, "(c) New Models")

    ax.legend(
        handles=[
            Line2D(
                [0],
                [0],
                marker="o",
                linestyle="None",
                markersize=5.5,
                markerfacecolor=COLORS["p5_frontier"],
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
                markeredgecolor=COLORS["p4_frontier"],
                markeredgewidth=0.9,
                alpha=val_alpha,
                label="val",
            ),
            Line2D([0], [0], color=COLORS["p5_frontier"], linewidth=STYLE["line_width"], label="fit (train)"),
            Line2D(
                [0],
                [0],
                color=COLORS["p4_frontier"],
                linewidth=STYLE["line_width"],
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

    # Panel (d): unseen P5 base families (right subplot of the period4 p4_to_p5 plot).
    ax = axes[3]
    # Colors/order follow plot_period4_frontiers_old_vs_new.py.
    p5_colors = {
        "Qwen3": "#d62728",
        "GPT-OSS": "#bcbd22",
        "Gemma-3": "#17becf",
        "SmolLM3": "#e377c2",
    }
    family_colors = dict(getattr(period4_old_vs_new_cfg, "EXTRA_FAMILY_COLORS", {}))
    family_colors.update(p5_colors)
    family_order = list(getattr(period4_old_vs_new_cfg, "EXTRA_FAMILY_ORDER", [])) + list(p5_colors.keys())

    # P5 unseen families (triangles, unfilled).
    for fam in family_order:
        m = unseen_family == fam
        if not np.any(m):
            continue
        c = family_colors.get(str(fam), "#333333")
        ax.scatter(
            unseen_x[m],
            unseen_y[m],
            s=(STYLE["marker_size"] * 1.15) ** 2,
            alpha=STYLE["marker_alpha"],
            facecolors="none",
            edgecolors=c,
            linewidths=0.9,
            marker="^",
            zorder=2,
            rasterized=True,
        )

    # Additional post-trained models (diamonds, filled).
    for fam in family_order:
        m = extra_family == fam
        if not np.any(m):
            continue
        c = family_colors.get(str(fam), "#333333")
        ax.scatter(
            extra_x[m],
            extra_y[m],
            s=(STYLE["marker_size"] * 1.15) ** 2,
            alpha=max(STYLE["marker_alpha"], 0.6),
            color=c,
            edgecolors="none",
            marker="D",
            zorder=2,
            rasterized=True,
        )

    # Baseline P4 frontier (fit train).
    if cx4_train.size:
        ax.plot(
            cx4_train,
            cy4_train,
            linewidth=STYLE["line_width"],
            alpha=STYLE["line_alpha"],
            color=COLORS["p5_frontier"],
            linestyle="-",
            zorder=3,
        )

    _configure_axis(ax, show_ylabel=False)
    ax.set_ylim(*ylim_p4_to_p5)
    ax.set_yticks([t for t in (0.0, 0.2, 0.4, 0.6, 0.8) if ylim_p4_to_p5[0] - 1e-6 <= t <= ylim_p4_to_p5[1] + 1e-6])
    ax.set_yticklabels([])
    ax.set_xlabel("Pretraining Compute (FLOPs)", fontsize=12, fontweight="bold", labelpad=4)
    _panel_label(ax, "(d) New Families")

    # Family legend for this panel, on the right side.
    val_set = set(str(v) for v in unseen_family.tolist())
    extra_set = set(str(v) for v in extra_family.tolist())
    fams_for_legend = [str(f) for f in family_order if (str(f) in val_set or str(f) in extra_set)]
    unseen_handles = []
    for fam in fams_for_legend:
        c = family_colors.get(fam, "#333333")
        if fam in val_set:
            unseen_handles.append(
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
            unseen_handles.append(
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
    if unseen_handles:
        ax.legend(
            unseen_handles,
            [h.get_label() for h in unseen_handles],
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

    os.makedirs(os.path.dirname(out_pdf), exist_ok=True)
    fig.savefig(out_pdf, dpi=DPI, bbox_inches="tight", pad_inches=0.05, facecolor="white", edgecolor="none")
    fig.savefig(os.path.splitext(out_pdf)[0] + ".png", dpi=DPI, bbox_inches="tight", pad_inches=0.05,
                facecolor="white", edgecolor="none")
    print(f"Saved: {out_pdf}")
    plt.close(fig)


if __name__ == "__main__":
    main()

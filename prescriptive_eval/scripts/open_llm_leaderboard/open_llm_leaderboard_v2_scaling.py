#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd

# Ensure repo root is on sys.path when running as a script.
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from skill_frontier.plotting.configs import frontier_1d as frontier_1d_cfg  # type: ignore
from skill_frontier.plotting.configs import mpl_rc as mpl_rc_cfg  # type: ignore

_RAW_WORD_RE = re.compile(r"\bRaw\b")
_RAW_TASK_COLUMNS: tuple[str, ...] = (
    "IFEval Raw",
    "BBH Raw",
    "MATH Lvl 5 Raw",
    "GPQA Raw",
    "MUSR Raw",
    "MMLU-PRO Raw",
)
_OLD_TASK_COLUMNS: tuple[str, ...] = (
    "ARC",
    "HellaSwag",
    "MMLU",
    "TruthfulQA",
    "Winogrande",
    "GSM8K",
    "Maintainers Choice",
)

# Mapping: canonical OLL Raw task columns -> columns in `tables/open_llm_leaderboard/new_eval_leaderboard.csv`.
# Keep this aligned with other "new eval" utilities (e.g., scripts/plot/plot_period4_frontiers_old_vs_new.py).
_NEW_LEADERBOARD_TASK_MAP: dict[str, str] = {
    "IFEval Raw": "leaderboard_ifeval_inst_level_strict_acc_none",
    "BBH Raw": "leaderboard_bbh_acc_norm_none",
    "MATH Lvl 5 Raw": "leaderboard_math_hard_exact_match_none",
    "GPQA Raw": "leaderboard_gpqa_acc_norm_none",
    "MUSR Raw": "leaderboard_musr_acc_norm_none",
    "MMLU-PRO Raw": "leaderboard_mmlu_pro_acc_none",
}


def _strip_raw(text: str) -> str:
    out = _RAW_WORD_RE.sub("", str(text))
    out = re.sub(r"\s+", " ", out).strip()
    return out


def _display_metric_name(name: str) -> str:
    # Strip common leaderboard arrow glyphs and "Raw".
    out = str(name).replace("⬆️", "").replace("⬇️", "")
    out = _strip_raw(out)
    return out


def _slugify(text: str) -> str:
    s = _display_metric_name(text)
    s = re.sub(r"[^A-Za-z0-9]+", "_", s).strip("_")
    return s.lower() or "metric"


def resolve_col(df: pd.DataFrame, aliases: List[str]) -> str:
    """Return the first matching column name from aliases (case-insensitive)."""
    cols = list(df.columns)
    lower_map = {str(c).strip().lower(): c for c in cols}
    for a in aliases:
        if a in cols:
            return a
        key = str(a).strip().lower()
        if key in lower_map:
            return str(lower_map[key])
    raise KeyError(f"Missing required column; tried aliases={aliases}. Available columns={cols}")


def _ensure_out_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _coerce_bool(series: pd.Series) -> pd.Series:
    def _to_bool(v: object) -> bool:
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return False
        if isinstance(v, bool):
            return bool(v)
        s = str(v).strip().lower()
        return s in {"1", "true", "t", "yes", "y"}

    return series.apply(_to_bool)


def _log10_positive(values: pd.Series) -> pd.Series:
    v = pd.to_numeric(values, errors="coerce")
    v = v.where(v > 0)
    return np.log10(v)


def _load_last_modified_map(top_models_csv: str) -> dict[str, str]:
    """Return a mapping model_id -> last_modified (string) from top_models_by_base.csv."""
    df = pd.read_csv(top_models_csv, usecols=["model_id", "last_modified"])
    df["model_id"] = df["model_id"].astype(str).str.strip()
    df["last_modified"] = df["last_modified"].astype(str).str.strip()
    df = df[(df["model_id"] != "") & (df["model_id"].str.lower() != "nan")].copy()
    df = df[(df["last_modified"] != "") & (df["last_modified"].str.lower() != "nan")].copy()
    if df.empty:
        return {}
    # Deduplicate by taking the max timestamp per model_id.
    dt = pd.to_datetime(df["last_modified"], errors="coerce", utc=True)
    df = df.assign(_dt=dt).dropna(subset=["_dt"]).copy()
    if df.empty:
        return {}
    df = df.sort_values("_dt").groupby("model_id", as_index=False).tail(1)
    return {str(k): str(v) for k, v in zip(df["model_id"], df["last_modified"])}


def _fetch_hf_last_modified(model_id: str, *, timeout_s: float = 20.0) -> Optional[str]:
    """Fetch Hugging Face model metadata and return its `lastModified` timestamp."""
    model_id = str(model_id).strip()
    if not model_id:
        return None
    url = f"https://huggingface.co/api/models/{model_id}"
    req = Request(url, headers={"User-Agent": "skill-frontier/open_llm_leaderboard_v2_scaling"})
    try:
        with urlopen(req, timeout=float(timeout_s)) as resp:  # nosec - URL is fixed to HF API
            data = json.loads(resp.read().decode("utf-8"))
    except (HTTPError, URLError, TimeoutError, ValueError):
        return None
    if not isinstance(data, dict):
        return None
    # HF uses camelCase in the public API.
    lm = data.get("lastModified") or data.get("last_modified")
    return str(lm).strip() if lm else None


def _load_extra_leaderboard_points(
    *,
    extra_csv: str,
    top_models_csv: str,
    extra_compute_csv: Optional[str],
    tasks: Sequence[str],
    compute_multiplier: float,
    fetch_missing_last_modified: bool,
    hf_timeout_s: float,
) -> pd.DataFrame:
    """Load extra eval points and align them to the OLL v2 schema used by this script.

    Supports two schemas:
      1) `new_leaderboard_results_with_tokens.csv` (already has tokens/params).
      2) `open_llm_leaderboard/validation_leaderboard.csv` (metrics only; we join
         compute from `open_llm_leaderboard/new_eval_leaderboard.csv` and
         params/date from `top_models_by_base.csv`).
    """
    df = pd.read_csv(extra_csv)
    out = pd.DataFrame()
    out["model_id"] = df.get("model_id", np.nan)
    out["mapped_base_model"] = df.get("mapped_base_model", np.nan)

    has_tokens = ("Pretraining tokens (T)" in df.columns) and ("#Params (B)" in df.columns)
    if has_tokens:
        out["Pretraining tokens (T)"] = pd.to_numeric(df.get("Pretraining tokens (T)", np.nan), errors="coerce")
        out["#Params (B)"] = pd.to_numeric(df.get("#Params (B)", np.nan), errors="coerce")
        pretrain_compute_zflops = pd.to_numeric(df.get("pretrain_compute_zflops", np.nan), errors="coerce")
        # Prefer a pre-enriched HF timestamp if present.
        if "lastModified" in df.columns:
            out["last_modified"] = df.get("lastModified", np.nan)
        elif "last_modified" in df.columns:
            out["last_modified"] = df.get("last_modified", np.nan)
        elif "Upload To Hub Date" in df.columns:
            out["last_modified"] = df.get("Upload To Hub Date", np.nan)
        elif "Submission Date" in df.columns:
            out["last_modified"] = df.get("Submission Date", np.nan)
    else:
        if not extra_compute_csv or not os.path.isfile(str(extra_compute_csv)):
            raise FileNotFoundError(
                "Extra leaderboard CSV does not contain tokens/params; "
                "provide --extra_compute_csv (tables/open_llm_leaderboard/new_eval_leaderboard.csv)."
            )

        df_compute = pd.read_csv(str(extra_compute_csv), usecols=["model_id", "pretrain_compute_zflops"])
        df_compute["model_id"] = df_compute["model_id"].astype(str).str.strip()
        df_compute["pretrain_compute_zflops"] = pd.to_numeric(df_compute["pretrain_compute_zflops"], errors="coerce")
        df_compute = df_compute.groupby("model_id", as_index=False)["pretrain_compute_zflops"].max()

        df_meta = pd.read_csv(top_models_csv, usecols=["model_id", "last_modified", "num_params_B"])
        df_meta["model_id"] = df_meta["model_id"].astype(str).str.strip()
        df_meta["num_params_B"] = pd.to_numeric(df_meta["num_params_B"], errors="coerce")
        df_meta["last_modified"] = pd.to_datetime(df_meta["last_modified"], errors="coerce", utc=True)
        df_meta = df_meta.dropna(subset=["model_id"]).copy()
        df_meta = df_meta.groupby("model_id", as_index=False).agg({"last_modified": "max", "num_params_B": "max"})

        df_join = df[["model_id", "mapped_base_model"]].copy()
        df_join["model_id"] = df_join["model_id"].astype(str).str.strip()
        df_join = df_join.merge(df_compute, on="model_id", how="left")
        df_join = df_join.merge(df_meta, on="model_id", how="left")

        out["#Params (B)"] = pd.to_numeric(df_join["num_params_B"], errors="coerce")
        pretrain_compute_zflops = pd.to_numeric(df_join["pretrain_compute_zflops"], errors="coerce")
        out["Pretraining tokens (T)"] = pretrain_compute_zflops / (float(compute_multiplier) * out["#Params (B)"])
        out["last_modified"] = df_join["last_modified"].dt.strftime("%Y-%m-%d %H:%M:%S%z")

    # Task columns: create canonical OLL names so downstream plotting can reuse the same code paths.
    for oll_col in tasks:
        src = _NEW_LEADERBOARD_TASK_MAP.get(oll_col)
        out[oll_col] = pd.to_numeric(df.get(src, np.nan), errors="coerce") if src else np.nan

    # Release time (used as "Date" in the figure3 replication plots).
    last_modified_map: dict[str, str] = {}
    if top_models_csv and os.path.isfile(top_models_csv):
        last_modified_map = _load_last_modified_map(top_models_csv)
    model_id_series = out["model_id"].astype(str).str.strip()
    if "last_modified" not in out.columns:
        out["last_modified"] = model_id_series.map(last_modified_map)

    # For missing entries, optionally fetch from HF API (needed for models not in top_models_by_base.csv).
    missing = out["last_modified"].isna() | (out["last_modified"].astype(str).str.lower().isin({"", "nan"}))
    if bool(fetch_missing_last_modified) and bool(missing.any()):
        unique_missing = sorted(set(model_id_series[missing].tolist()))
        fetched: dict[str, str] = {}
        for mid in unique_missing:
            lm = _fetch_hf_last_modified(mid, timeout_s=float(hf_timeout_s))
            if lm:
                fetched[mid] = lm
        if fetched:
            out.loc[missing, "last_modified"] = model_id_series[missing].map(fetched)

    # Store in the same column name the main table uses by default.
    out["Upload To Hub Date"] = out["last_modified"]
    out["Submission Date"] = out["last_modified"]
    out["pretrain_compute_zflops"] = pretrain_compute_zflops

    return out


@dataclass(frozen=True)
class ResolvedCols:
    model: str
    score: str
    params_b: str
    date: str
    upload_date: Optional[str]
    flagged: Optional[str]
    merged: Optional[str]
    moe: Optional[str]
    tokens_t: Optional[str]
    compute: Optional[str]


def _resolve_columns(df: pd.DataFrame, *, date_override: Optional[str] = None) -> ResolvedCols:
    model = resolve_col(df, ["eval_name", "model", "Model", "model_id", "model_name", "fullname"])
    score = resolve_col(df, ["Average ⬆️", "average", "Average", "avg_score", "score", "leaderboard_score"])
    params_b = resolve_col(df, ["#Params (B)", "Params (B)", "params_b", "params", "parameters_b", "n_params_b"])

    upload_date = None
    for a in ["Upload To Hub Date", "upload_date", "uploaded_at"]:
        try:
            upload_date = resolve_col(df, [a])
            break
        except KeyError:
            continue

    if date_override:
        date = resolve_col(df, [date_override])
    elif upload_date is not None:
        # Default to upload date when present (preferred timeline).
        date = upload_date
    else:
        date = resolve_col(df, ["Submission Date", "submission_date", "submitted_at", "date"])

    flagged = None
    for a in ["Flagged", "flagged"]:
        try:
            flagged = resolve_col(df, [a])
            break
        except KeyError:
            continue

    merged = None
    for a in ["Merged", "merged"]:
        try:
            merged = resolve_col(df, [a])
            break
        except KeyError:
            continue

    moe = None
    for a in ["MoE", "moe", "is_moe"]:
        try:
            moe = resolve_col(df, [a])
            break
        except KeyError:
            continue

    tokens_t = None
    for a in ["Pretraining tokens (T)", "tokens", "token_count", "training_tokens", "pretraining_tokens"]:
        try:
            tokens_t = resolve_col(df, [a])
            break
        except KeyError:
            continue

    compute = None
    for a in ["pretraining_compute", "compute", "training_compute", "flops", "pretrain_flops"]:
        try:
            compute = resolve_col(df, [a])
            break
        except KeyError:
            continue

    return ResolvedCols(
        model=model,
        score=score,
        params_b=params_b,
        date=date,
        upload_date=upload_date,
        flagged=flagged,
        merged=merged,
        moe=moe,
        tokens_t=tokens_t,
        compute=compute,
    )


def _linear_fit_slope_r2(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    keep = np.isfinite(x) & np.isfinite(y)
    x = x[keep]
    y = y[keep]
    if x.size < 2:
        return float("nan"), float("nan")
    slope, intercept = np.polyfit(x, y, 1)
    y_hat = slope * x + intercept
    ss_res = float(np.sum((y - y_hat) ** 2))
    ss_tot = float(np.sum((y - float(np.mean(y))) ** 2))
    r2 = float("nan") if ss_tot <= 0 else float(1.0 - ss_res / ss_tot)
    return float(slope), r2


def _spearman(x: pd.Series, y: pd.Series) -> float:
    df = pd.DataFrame({"x": x, "y": y}).dropna()
    if df.shape[0] < 3:
        return float("nan")
    try:
        return float(df["x"].corr(df["y"], method="spearman"))
    except Exception:
        return float("nan")


def _compute_small_frontier(
    *,
    df_small: pd.DataFrame,
    date_col: str,
    score_col: str,
) -> tuple[pd.Series, pd.Series]:
    day = df_small[date_col].dt.floor("D")
    daily_best = df_small.assign(date_day=day).groupby("date_day")[score_col].max().sort_index()
    frontier = daily_best.cummax()
    return daily_best, frontier


def _mark_large_dominated(
    *,
    df_large: pd.DataFrame,
    date_col: str,
    score_col: str,
    daily_best_small: pd.Series,
    frontier_small: pd.Series,
    dominance_mode: str,
) -> pd.DataFrame:
    df_large = df_large.copy()
    df_large["date_day"] = df_large[date_col].dt.floor("D")

    if dominance_mode == "global_best_small":
        ref = float(daily_best_small.max()) if not daily_best_small.empty else float("nan")
        df_large["small_ref"] = ref
    else:
        ref_series = frontier_small if dominance_mode == "frontier" else daily_best_small
        df_large["small_ref"] = ref_series.reindex(df_large["date_day"]).ffill().to_numpy()

    df_large["dominated"] = (df_large[score_col] < df_large["small_ref"]) & np.isfinite(df_large["small_ref"])
    return df_large


def _build_monthly_metrics(
    *,
    df_all: pd.DataFrame,
    df_small: pd.DataFrame,
    df_large_eval: pd.DataFrame,
    threshold_params_b: float,
    score_col: str,
    params_col: str,
    tokens_col: Optional[str],
    compute_col: Optional[str],
    dominance_mode: str,
    min_samples: int = 10,
) -> pd.DataFrame:
    df_all = df_all.copy()
    df_all["month"] = df_all["date"].dt.to_period("M").dt.to_timestamp()
    df_small = df_small.copy()
    df_small["month"] = df_small["date"].dt.to_period("M").dt.to_timestamp()
    df_large_eval = df_large_eval.copy()
    df_large_eval["month"] = df_large_eval["date"].dt.to_period("M").dt.to_timestamp()

    df_all["log_params_b"] = _log10_positive(df_all[params_col])
    if tokens_col is not None:
        df_all["log_tokens"] = _log10_positive(df_all[tokens_col])
    if compute_col is not None:
        df_all["log_compute"] = _log10_positive(df_all[compute_col])

    out_rows: List[dict] = []
    for month in sorted(df_all["month"].dropna().unique().tolist()):
        df_m = df_all[df_all["month"] == month]
        n_total = int(df_m.shape[0])
        n_small = int((df_m[params_col] < float(threshold_params_b)).sum())
        n_large = int((df_m[params_col] > float(threshold_params_b)).sum())

        df_large_m = df_large_eval[df_large_eval["month"] == month]
        dom_rate = float("nan")
        if df_large_m.shape[0] > 0:
            dom_rate = float(df_large_m["dominated"].mean())

        def _subset_metrics(sub: pd.DataFrame, prefix: str) -> dict:
            if sub.shape[0] < int(min_samples):
                return {
                    f"spearman_log_params_{prefix}": float("nan"),
                    f"slope_log_params_{prefix}": float("nan"),
                    f"r2_log_params_{prefix}": float("nan"),
                    f"spearman_log_compute_{prefix}": float("nan"),
                    f"slope_log_compute_{prefix}": float("nan"),
                    f"r2_log_compute_{prefix}": float("nan"),
                    f"spearman_log_tokens_{prefix}": float("nan"),
                    f"slope_log_tokens_{prefix}": float("nan"),
                    f"r2_log_tokens_{prefix}": float("nan"),
                }

            row: dict = {}
            row[f"spearman_log_params_{prefix}"] = _spearman(sub["log_params_b"], sub[score_col])
            slope, r2 = _linear_fit_slope_r2(sub["log_params_b"].to_numpy(), sub[score_col].to_numpy())
            row[f"slope_log_params_{prefix}"] = slope
            row[f"r2_log_params_{prefix}"] = r2

            if compute_col is not None and "log_compute" in sub.columns and sub["log_compute"].notna().sum() >= int(min_samples):
                row[f"spearman_log_compute_{prefix}"] = _spearman(sub["log_compute"], sub[score_col])
                slope, r2 = _linear_fit_slope_r2(sub["log_compute"].to_numpy(), sub[score_col].to_numpy())
                row[f"slope_log_compute_{prefix}"] = slope
                row[f"r2_log_compute_{prefix}"] = r2
            else:
                row[f"spearman_log_compute_{prefix}"] = float("nan")
                row[f"slope_log_compute_{prefix}"] = float("nan")
                row[f"r2_log_compute_{prefix}"] = float("nan")

            if tokens_col is not None and "log_tokens" in sub.columns and sub["log_tokens"].notna().sum() >= int(min_samples):
                row[f"spearman_log_tokens_{prefix}"] = _spearman(sub["log_tokens"], sub[score_col])
                slope, r2 = _linear_fit_slope_r2(sub["log_tokens"].to_numpy(), sub[score_col].to_numpy())
                row[f"slope_log_tokens_{prefix}"] = slope
                row[f"r2_log_tokens_{prefix}"] = r2
            else:
                row[f"spearman_log_tokens_{prefix}"] = float("nan")
                row[f"slope_log_tokens_{prefix}"] = float("nan")
                row[f"r2_log_tokens_{prefix}"] = float("nan")

            return row

        row = {
            "month": month,
            "n_total": n_total,
            "n_small": n_small,
            "n_large": n_large,
            f"dominance_rate_{dominance_mode}": dom_rate,
        }

        row.update(_subset_metrics(df_m, "all"))
        row.update(_subset_metrics(df_m[df_m[params_col] < float(threshold_params_b)], "small"))
        row.update(_subset_metrics(df_m[df_m[params_col] > float(threshold_params_b)], "large"))

        out_rows.append(row)

    return pd.DataFrame(out_rows)


def _plot_figure3(
    *,
    df_small: pd.DataFrame,
    df_large_eval: pd.DataFrame,
    date_col: str,
    score_col: str,
    metric_label: str,
    threshold_params_b: float,
    daily_best_small: pd.Series,
    frontier_small: pd.Series,
    dominance_mode: str,
    out_path_base: str,
    cutoff_date: Optional[pd.Timestamp],
) -> None:
    import matplotlib as mpl  # type: ignore
    import matplotlib.dates as mdates  # type: ignore
    import matplotlib.pyplot as plt  # type: ignore

    mpl.rcParams["font.family"] = mpl_rc_cfg.FONT_FAMILY
    mpl.rcParams["mathtext.fontset"] = mpl_rc_cfg.MATH_FONTSET

    # Point styling (user-specified):
    #   top-to-bottom zorder: large > small > dominated large
    large_color = "#1f77b4"
    small_color = "firebrick"
    dominated_color = "gray"
    large_color_newer = "#0d5aa7"
    small_color_newer = "#ff7f0e"
    dominated_color_newer = "#3d3d3d"
    alpha_large = 0.8
    alpha_small = 0.7
    alpha_dominated = 0.5

    # Match the font sizes used in outputs/sigmoid/no_split/pretrain_vs_posttrain/plots/overlay_BBH_Raw.png.
    label_fs_x = float(frontier_1d_cfg.LABEL_FONTSIZE_X)
    label_fs_y = float(frontier_1d_cfg.LABEL_FONTSIZE_Y)
    tick_fs = float(frontier_1d_cfg.TICK_LABELSIZE)
    legend_fs = float(frontier_1d_cfg.LEGEND_FONTSIZE) * 0.75
    # Match the pixel size of outputs/sigmoid/no_split/plots/BBH_Raw.png (2069x1344 @ 300 dpi).
    panel_figsize = (0.75 * (2069.0 / 300.0), 1344.0 / 300.0)

    score_is_percent = float(df_small[score_col].max()) > 1.0
    if score_is_percent:
        ylabel = "Average (%)" if metric_label.lower() == "average" else f"{metric_label} (%)"
    else:
        ylabel = "Average" if metric_label.lower() == "average" else str(metric_label)

    cutoff_day = None
    cutoff_label = None
    if cutoff_date is not None and pd.notna(cutoff_date):
        cutoff_day = pd.to_datetime(cutoff_date, errors="coerce").floor("D")
        if pd.notna(cutoff_day):
            cutoff_label = f"cutoff ({cutoff_day.date().isoformat()})"

    # Remove legacy combined outputs (v1).
    for ext in (".png", ".pdf"):
        try:
            os.remove(out_path_base + ext)
        except FileNotFoundError:
            pass
        except Exception:
            pass

    # Panel (a): small models over time (saved as its own figure)
    fig_a, ax_a = plt.subplots(figsize=panel_figsize)
    small_is_newer = pd.Series(False, index=df_small.index)
    if cutoff_day is not None:
        small_is_newer = df_small[date_col].dt.floor("D") > cutoff_day
        if cutoff_label:
            ax_a.axvline(cutoff_day, color="black", linestyle="--", linewidth=1.6, alpha=0.65, zorder=0, label=cutoff_label)

    df_small_old = df_small[~small_is_newer]
    df_small_new = df_small[small_is_newer]
    if not df_small_old.empty:
        ax_a.scatter(
            df_small_old[date_col].to_numpy(),
            df_small_old[score_col].to_numpy(),
            s=14,
            alpha=alpha_small,
            color=small_color,
            label=f"small models < {threshold_params_b:g}B",
            rasterized=True,
            zorder=2,
        )
    if not df_small_new.empty:
        ax_a.scatter(
            df_small_new[date_col].to_numpy(),
            df_small_new[score_col].to_numpy(),
            s=14,
            alpha=alpha_small,
            color=small_color_newer,
            label="_nolegend_",
            rasterized=True,
            zorder=2,
        )
    if not daily_best_small.empty:
        ax_a.plot(
            daily_best_small.index.to_numpy(),
            daily_best_small.to_numpy(),
            color=small_color,
            linewidth=2.0,
            alpha=0.95,
            label="best daily (small)",
            zorder=3,
        )
    if dominance_mode == "frontier" and not frontier_small.empty:
        ax_a.plot(
            frontier_small.index.to_numpy(),
            frontier_small.to_numpy(),
            color=small_color,
            linewidth=2.0,
            linestyle="--",
            alpha=0.9,
            label="frontier (cummax)",
            zorder=3,
        )

    ax_a.set_xlabel("Date", fontweight="bold", fontsize=label_fs_x)
    ax_a.set_ylabel(ylabel, fontweight="bold", fontsize=label_fs_y)
    ax_a.tick_params(axis="both", labelsize=tick_fs)
    ax_a.grid(
        True,
        which="major",
        axis="y",
        linestyle=frontier_1d_cfg.GRID_MAJOR_LINESTYLE,
        linewidth=frontier_1d_cfg.GRID_MAJOR_LINEWIDTH,
        color=frontier_1d_cfg.GRID_MAJOR_COLOR,
        alpha=frontier_1d_cfg.GRID_MAJOR_ALPHA,
    )
    ax_a.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=4, maxticks=8))
    ax_a.xaxis.set_major_formatter(mdates.ConciseDateFormatter(ax_a.xaxis.get_major_locator()))
    for spine in ax_a.spines.values():
        spine.set_linewidth(frontier_1d_cfg.SPINE_LINEWIDTH)
        spine.set_color(frontier_1d_cfg.SPINE_COLOR)
    ax_a.legend(loc="best", fontsize=legend_fs, frameon=True, framealpha=frontier_1d_cfg.LEGEND_FRAMEALPHA)
    fig_a.subplots_adjust(left=0.18, right=0.98, bottom=0.28, top=0.98)
    save_kwargs = {"dpi": 300, "bbox_inches": "tight", "pad_inches": 0.02}
    fig_a.savefig(out_path_base + "_small.png", **save_kwargs)
    fig_a.savefig(out_path_base + "_small.pdf", **save_kwargs)
    plt.close(fig_a)

    # Panel (b): large models vs small; highlight dominated (saved as its own figure)
    fig_b, ax_b = plt.subplots(figsize=panel_figsize)
    dominated_mask = df_large_eval["dominated"].astype(bool)
    df_large_main = df_large_eval[~dominated_mask]

    large_is_newer = pd.Series(False, index=df_large_main.index)
    if cutoff_day is not None:
        large_is_newer = df_large_main[date_col].dt.floor("D") > cutoff_day
        if cutoff_label:
            ax_b.axvline(cutoff_day, color="black", linestyle="--", linewidth=1.6, alpha=0.65, zorder=0, label=cutoff_label)

    df_large_old = df_large_main[~large_is_newer]
    df_large_new = df_large_main[large_is_newer]
    if not df_large_old.empty:
        ax_b.scatter(
            df_large_old[date_col].to_numpy(),
            df_large_old[score_col].to_numpy(),
            s=26,
            alpha=alpha_large,
            color=large_color,
            label=f"large models > {threshold_params_b:g}B",
            rasterized=True,
            zorder=3,
        )
    if not df_large_new.empty:
        ax_b.scatter(
            df_large_new[date_col].to_numpy(),
            df_large_new[score_col].to_numpy(),
            s=26,
            alpha=alpha_large,
            color=large_color_newer,
            label="_nolegend_",
            rasterized=True,
            zorder=3,
        )

    if not df_small_old.empty:
        ax_b.scatter(
            df_small_old[date_col].to_numpy(),
            df_small_old[score_col].to_numpy(),
            s=10,
            alpha=alpha_small,
            color=small_color,
            label=f"small models < {threshold_params_b:g}B",
            rasterized=True,
            zorder=2,
        )
    if not df_small_new.empty:
        ax_b.scatter(
            df_small_new[date_col].to_numpy(),
            df_small_new[score_col].to_numpy(),
            s=10,
            alpha=alpha_small,
            color=small_color_newer,
            label="_nolegend_",
            rasterized=True,
            zorder=2,
        )
    dominated = df_large_eval[dominated_mask]
    if not dominated.empty:
        dominated_is_newer = pd.Series(False, index=dominated.index)
        if cutoff_day is not None:
            dominated_is_newer = dominated[date_col].dt.floor("D") > cutoff_day

        dominated_old = dominated[~dominated_is_newer]
        dominated_new = dominated[dominated_is_newer]
        if not dominated_old.empty:
            ax_b.scatter(
                dominated_old[date_col].to_numpy(),
                dominated_old[score_col].to_numpy(),
                s=38,
                alpha=alpha_dominated,
                facecolors="none",
                edgecolors=dominated_color,
                linewidths=1.2,
                label="dominated large",
                rasterized=True,
                zorder=1,
            )
        if not dominated_new.empty:
            ax_b.scatter(
                dominated_new[date_col].to_numpy(),
                dominated_new[score_col].to_numpy(),
                s=38,
                alpha=alpha_dominated,
                facecolors="none",
                edgecolors=dominated_color_newer,
                linewidths=1.2,
                label="_nolegend_",
                rasterized=True,
                zorder=1,
            )

    ax_b.set_xlabel("Date", fontweight="bold", fontsize=label_fs_x)
    ax_b.set_ylabel(ylabel, fontweight="bold", fontsize=label_fs_y)
    ax_b.tick_params(axis="both", labelsize=tick_fs)
    ax_b.grid(
        True,
        which="major",
        axis="y",
        linestyle=frontier_1d_cfg.GRID_MAJOR_LINESTYLE,
        linewidth=frontier_1d_cfg.GRID_MAJOR_LINEWIDTH,
        color=frontier_1d_cfg.GRID_MAJOR_COLOR,
        alpha=frontier_1d_cfg.GRID_MAJOR_ALPHA,
    )
    ax_b.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=4, maxticks=8))
    ax_b.xaxis.set_major_formatter(mdates.ConciseDateFormatter(ax_b.xaxis.get_major_locator()))
    for spine in ax_b.spines.values():
        spine.set_linewidth(frontier_1d_cfg.SPINE_LINEWIDTH)
        spine.set_color(frontier_1d_cfg.SPINE_COLOR)
    ax_b.legend(loc="best", fontsize=legend_fs, frameon=True, framealpha=frontier_1d_cfg.LEGEND_FRAMEALPHA)
    fig_b.subplots_adjust(left=0.18, right=0.98, bottom=0.28, top=0.98)
    fig_b.savefig(out_path_base + "_large.png", **save_kwargs)
    fig_b.savefig(out_path_base + "_large.pdf", **save_kwargs)
    plt.close(fig_b)


def _run_once(
    *,
    input_csv: Optional[str] = None,
    input_df: Optional[pd.DataFrame] = None,
    out_dir: str,
    date_col_override: Optional[str],
    score_col_override: Optional[str],
    dominance_mode: str,
    threshold_params_b: float,
    exclude_flagged: bool,
    exclude_merged: bool,
    exclude_moe: bool,
    compute_multiplier: float,
    suffix: str,
    cutoff_date: Optional[pd.Timestamp],
) -> tuple[str, str]:
    if input_df is None and input_csv is None:
        raise ValueError("Provide input_df or input_csv")
    df = input_df.copy() if input_df is not None else pd.read_csv(str(input_csv))
    cols = _resolve_columns(df, date_override=date_col_override)
    score_col = str(score_col_override) if score_col_override else cols.score
    metric_label = _display_metric_name(score_col)
    metric_slug = _slugify(metric_label)

    df = df.rename(columns={cols.date: "date"}).copy()
    # Mixed date formats can appear when appending extra eval points (e.g. YYYY-MM-DD vs full ISO timestamps).
    df["date"] = pd.to_datetime(df["date"], errors="coerce", utc=True, format="mixed").dt.tz_convert(None)
    df["_score"] = pd.to_numeric(df[score_col], errors="coerce")
    df["_params_b"] = pd.to_numeric(df[cols.params_b], errors="coerce")

    if cols.flagged is not None:
        df["_flagged"] = _coerce_bool(df[cols.flagged])
    else:
        df["_flagged"] = False
    if cols.merged is not None:
        df["_merged"] = _coerce_bool(df[cols.merged])
    else:
        df["_merged"] = False
    if cols.moe is not None:
        df["_moe"] = _coerce_bool(df[cols.moe])
    else:
        df["_moe"] = False

    dropped_before = int(df.shape[0])
    df = df.dropna(subset=["date", "_score", "_params_b"]).copy()
    df = df[df["_params_b"] > 0].copy()
    if exclude_flagged and cols.flagged is not None:
        df_unflagged = df[~df["_flagged"]].copy()
        if df_unflagged.empty:
            in_name = os.path.basename(str(input_csv)) if input_csv else "in-memory"
            print(f"[warn] exclude_flagged would drop all rows for {in_name}; keeping flagged rows")
        else:
            df = df_unflagged
    if exclude_merged and cols.merged is not None:
        df = df[~df["_merged"]].copy()
    if exclude_moe and cols.moe is not None:
        df = df[~df["_moe"]].copy()
    dropped_after = int(df.shape[0])

    tokens_col = None
    if cols.tokens_t is not None and cols.tokens_t in df.columns:
        tokens_col = cols.tokens_t
        df["_tokens_t"] = pd.to_numeric(df[tokens_col], errors="coerce")
    else:
        df["_tokens_t"] = np.nan

    compute_col = None
    if cols.compute is not None and cols.compute in df.columns:
        compute_col = cols.compute
        df["_compute_flops"] = pd.to_numeric(df[compute_col], errors="coerce")
    elif tokens_col is not None:
        # Compute FLOPs from tokens (T) and params (B): 6 * T * B * 1e21.
        df["_compute_flops"] = float(compute_multiplier) * df["_tokens_t"] * df["_params_b"] * 1e21
        compute_col = "_compute_flops"
    else:
        df["_compute_flops"] = np.nan

    df_small = df[df["_params_b"] < float(threshold_params_b)].copy()
    df_large = df[df["_params_b"] > float(threshold_params_b)].copy()

    daily_best_small, frontier_small = _compute_small_frontier(df_small=df_small, date_col="date", score_col="_score")
    df_large_eval = _mark_large_dominated(
        df_large=df_large,
        date_col="date",
        score_col="_score",
        daily_best_small=daily_best_small,
        frontier_small=frontier_small,
        dominance_mode=dominance_mode,
    )

    _ensure_out_dir(out_dir)
    out_plot = os.path.join(out_dir, f"v2_figure3_replication_{metric_slug}{suffix}")
    out_metrics = os.path.join(out_dir, f"v2_scaling_metrics_{metric_slug}{suffix}.csv")

    _plot_figure3(
        df_small=df_small,
        df_large_eval=df_large_eval,
        date_col="date",
        score_col="_score",
        metric_label=metric_label,
        threshold_params_b=float(threshold_params_b),
        daily_best_small=daily_best_small,
        frontier_small=frontier_small,
        dominance_mode=dominance_mode,
        out_path_base=out_plot,
        cutoff_date=cutoff_date,
    )

    metrics_df = _build_monthly_metrics(
        df_all=df,
        df_small=df_small,
        df_large_eval=df_large_eval,
        threshold_params_b=float(threshold_params_b),
        score_col="_score",
        params_col="_params_b",
        tokens_col="_tokens_t" if tokens_col is not None else None,
        compute_col=compute_col,
        dominance_mode=dominance_mode,
        min_samples=10,
    )
    metrics_df.to_csv(out_metrics, index=False)

    min_date = df["date"].min()
    max_date = df["date"].max()
    print(
        f"[run{suffix or ''}] rows_loaded={dropped_before} rows_after_clean={dropped_after} "
        f"small={int(df_small.shape[0])} large={int(df_large.shape[0])} "
        f"date_range={min_date.date() if pd.notna(min_date) else None}..{max_date.date() if pd.notna(max_date) else None}"
    )
    print(f"[run{suffix or ''}] wrote plots: {out_plot}_small.png|.pdf and {out_plot}_large.png|.pdf")
    print(f"[run{suffix or ''}] wrote metrics: {out_metrics}")

    return out_plot, out_metrics


def main(argv: Optional[Sequence[str]] = None) -> None:
    ap = argparse.ArgumentParser(description="Replicate Figure 3 logic on Open LLM Leaderboard v2 CSV.")
    ap.add_argument(
        "--input_csv",
        default=os.path.join("tables", "open_llm_leaderboard", "open_llm_leaderboard_with_tokens.csv"),
        help="Path to the local Open LLM Leaderboard v2 CSV.",
    )
    ap.add_argument(
        "--extra_csv",
        action="append",
        default=None,
        help="Optional extra eval points to append to the current plots (repeatable; overrides defaults when provided).",
    )
    ap.add_argument(
        "--extra_compute_csv",
        default=os.path.join("tables", "open_llm_leaderboard", "new_eval_leaderboard.csv"),
        help=(
            "Compute table for extra points (expects columns: model_id, pretrain_compute_zflops). "
            "Used when --extra_csv does not include tokens/params columns."
        ),
    )
    ap.add_argument(
        "--top_models_csv",
        default=os.path.join("tables", "top_models_by_base.csv"),
        help="Optional metadata CSV containing last_modified for some model IDs (used to date extra points).",
    )
    ap.add_argument(
        "--fetch_missing_last_modified",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="If True, fetch missing lastModified timestamps from the Hugging Face API for extra points.",
    )
    ap.add_argument("--hf_timeout_s", type=float, default=20.0, help="Timeout (seconds) per HF API request.")
    ap.add_argument("--out_dir", default=os.path.join("outputs", "open_llm_leaderboard"))
    ap.add_argument("--date_col", default=None, help="Override date column (otherwise prefers Upload To Hub Date when available).")
    ap.add_argument("--also_run_upload_date", action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument("--dominance_mode", choices=("frontier", "global_best_small", "same_day"), default="frontier")
    ap.add_argument("--threshold_params_b", type=float, default=13.0)
    ap.add_argument("--exclude_flagged", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--exclude_merged", action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument("--exclude_moe", action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument("--compute_multiplier", type=float, default=6.0, help="Compute FLOPs = multiplier * tokens(T) * params(B) * 1e21.")

    args = ap.parse_args(list(argv) if argv is not None else None)

    out_root = str(args.out_dir)
    out_current = os.path.join(out_root, "current")
    out_old = os.path.join(out_root, "old")

    # Current leaderboard: plot all six "Raw" tasks, using upload date by default.
    df_current = pd.read_csv(str(args.input_csv))
    raw_task_cols = [resolve_col(df_current, [c]) for c in _RAW_TASK_COLUMNS]

    cutoff_date = None
    try:
        cols_for_cutoff = _resolve_columns(df_current, date_override=str(args.date_col) if args.date_col else None)
        dt_cutoff = pd.to_datetime(df_current[cols_for_cutoff.date], errors="coerce", utc=True, format="mixed")
        if dt_cutoff.notna().any():
            cutoff_date = dt_cutoff.max().tz_convert(None)
    except Exception:
        cutoff_date = None

    df_combined = df_current

    default_extra_csvs = [
        os.path.join("tables", "open_llm_leaderboard", "validation_leaderboard.csv"),
        os.path.join("tables", "new_leaderboard_results_with_tokens.csv"),
    ]
    extra_csvs: List[str] = list(default_extra_csvs) if args.extra_csv is None else [str(p) for p in args.extra_csv]
    extra_frames: List[pd.DataFrame] = []

    # Drop entries already present in the official current CSV (avoid double-counting).
    cur_ids: set[str] = set()
    for col in ("fullname", "Base Model", "model_id"):
        if col in df_current.columns:
            cur_ids = set(df_current[col].dropna().astype(str).str.strip().tolist())
            break

    dropped_overlap_total = 0
    missing_date_total = 0
    for extra_csv in extra_csvs:
        if not extra_csv:
            continue
        if not os.path.isfile(extra_csv):
            print(f"[current] extra_points skipped (file not found): {extra_csv}")
            continue
        df_extra = _load_extra_leaderboard_points(
            extra_csv=str(extra_csv),
            top_models_csv=str(args.top_models_csv),
            extra_compute_csv=str(args.extra_compute_csv) if args.extra_compute_csv else None,
            tasks=list(raw_task_cols),
            compute_multiplier=float(args.compute_multiplier),
            fetch_missing_last_modified=bool(args.fetch_missing_last_modified),
            hf_timeout_s=float(args.hf_timeout_s),
        )

        if cur_ids and "model_id" in df_extra.columns:
            before = int(df_extra.shape[0])
            df_extra = df_extra[~df_extra["model_id"].astype(str).str.strip().isin(cur_ids)].copy()
            dropped_overlap_total += before - int(df_extra.shape[0])

        missing_date = (
            int(pd.to_datetime(df_extra["Upload To Hub Date"], errors="coerce").isna().sum())
            if "Upload To Hub Date" in df_extra.columns
            else int(df_extra.shape[0])
        )
        missing_date_total += missing_date
        print(f"[current] extra_points rows={int(df_extra.shape[0])} missing_date={missing_date} source={extra_csv}")
        extra_frames.append(df_extra)

    if extra_frames:
        df_extra_all = pd.concat(extra_frames, ignore_index=True, sort=False)
        if "model_id" in df_extra_all.columns:
            df_extra_all["model_id"] = df_extra_all["model_id"].astype(str).str.strip()
            # Prefer later sources when duplicates exist.
            df_extra_all = df_extra_all.drop_duplicates(subset=["model_id"], keep="last").copy()
        df_combined = pd.concat([df_current, df_extra_all], ignore_index=True, sort=False)
        print(
            f"[current] extra_points merged_rows={int(df_extra_all.shape[0])} dropped_overlap_total={dropped_overlap_total} "
            f"missing_date_total={missing_date_total}"
        )

    for score_col in raw_task_cols:
        _run_once(
            input_csv=str(args.input_csv),
            input_df=df_combined,
            out_dir=out_current,
            date_col_override=str(args.date_col) if args.date_col else None,
            score_col_override=score_col,
            dominance_mode=str(args.dominance_mode),
            threshold_params_b=float(args.threshold_params_b),
            exclude_flagged=bool(args.exclude_flagged),
            exclude_merged=bool(args.exclude_merged),
            exclude_moe=bool(args.exclude_moe),
            compute_multiplier=float(args.compute_multiplier),
            suffix="",
            cutoff_date=cutoff_date,
        )

    # Old leaderboard: run the same analysis on the overall average column.
    old_csv = os.path.join("tables", "open_llm_leaderboard", "open_llm_leaderboard_old_with_tokens.csv")
    df_old_head = pd.read_csv(old_csv, nrows=1)
    old_task_cols = []
    for c in _OLD_TASK_COLUMNS:
        try:
            old_task_cols.append(resolve_col(df_old_head, [c]))
        except KeyError:
            continue
    if not old_task_cols:
        old_task_cols = [resolve_col(df_old_head, ["Average ⬆️", "average", "Average", "avg_score", "score", "leaderboard_score"])]
    for score_col in old_task_cols:
        _run_once(
            input_csv=old_csv,
            out_dir=out_old,
            date_col_override=str(args.date_col) if args.date_col else None,
            score_col_override=score_col,
            dominance_mode=str(args.dominance_mode),
            threshold_params_b=float(args.threshold_params_b),
            exclude_flagged=bool(args.exclude_flagged),
            exclude_merged=bool(args.exclude_merged),
            exclude_moe=bool(args.exclude_moe),
            compute_multiplier=float(args.compute_multiplier),
            suffix="",
            cutoff_date=None,
        )

    if bool(args.also_run_upload_date):
        # Compatibility knob: also run an alternative timeline using submission date when available.
        df = pd.read_csv(str(args.input_csv), nrows=1)
        try:
            submission_col = resolve_col(df, ["Submission Date", "submission_date", "submitted_at", "date"])
        except KeyError:
            submission_col = None
        if submission_col is None:
            print("[run_submission_date] skipped (submission date column not found)")
        else:
            for score_col in raw_task_cols:
                _run_once(
                    input_csv=str(args.input_csv),
                    out_dir=out_current,
                    date_col_override=submission_col,
                    score_col_override=score_col,
                    dominance_mode=str(args.dominance_mode),
                    threshold_params_b=float(args.threshold_params_b),
                    exclude_flagged=bool(args.exclude_flagged),
                    exclude_merged=bool(args.exclude_merged),
                    exclude_moe=bool(args.exclude_moe),
                    compute_multiplier=float(args.compute_multiplier),
                    suffix="_submission_date",
                    cutoff_date=cutoff_date,
                )


if __name__ == "__main__":
    main()

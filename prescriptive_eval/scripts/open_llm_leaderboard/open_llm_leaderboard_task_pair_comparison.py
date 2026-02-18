#!/usr/bin/env python3

from __future__ import annotations

import argparse
import os
import sys
from typing import Optional, Sequence, Tuple

import numpy as np
import pandas as pd

# Ensure repo root is on sys.path when running as a script.
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from skill_frontier.plotting.configs import frontier_1d as frontier_1d_cfg  # type: ignore
from skill_frontier.plotting.configs import mpl_rc as mpl_rc_cfg  # type: ignore


PAIR_SPECS: Tuple[Tuple[str, str, str], ...] = (
    ("MMLU", "MMLU-PRO", "mmlu_vs_mmlu_pro"),
    ("GSM8K", "MATH Lvl 5", "gsm8k_vs_math_lvl_5"),
)


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


def _load_and_filter(
    *,
    path: str,
    exclude_flagged: bool,
    exclude_merged: bool,
    exclude_moe: bool,
) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "Flagged" in df.columns:
        df["_flagged"] = _coerce_bool(df["Flagged"])
    else:
        df["_flagged"] = False

    if "Merged" in df.columns:
        df["_merged"] = _coerce_bool(df["Merged"])
    else:
        df["_merged"] = False

    if "MoE" in df.columns:
        df["_moe"] = _coerce_bool(df["MoE"])
    else:
        df["_moe"] = False

    if exclude_flagged and "Flagged" in df.columns:
        df_unflagged = df[~df["_flagged"]].copy()
        if not df_unflagged.empty:
            df = df_unflagged
    if exclude_merged and "Merged" in df.columns:
        df = df[~df["_merged"]].copy()
    if exclude_moe and "MoE" in df.columns:
        df = df[~df["_moe"]].copy()

    return df


def _mark_dominated_large(
    *,
    small_x: np.ndarray,
    small_y: np.ndarray,
    large_x: np.ndarray,
    large_y: np.ndarray,
) -> np.ndarray:
    if small_x.size == 0 or small_y.size == 0 or large_x.size == 0 or large_y.size == 0:
        return np.zeros((large_x.size,), dtype=bool)

    # Dominated if there exists a small model strictly better on both axes.
    dom = (small_x[None, :] > large_x[:, None]) & (small_y[None, :] > large_y[:, None])
    return np.any(dom, axis=1)


def _plot_pair(
    *,
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    out_path_base: str,
    threshold_params_b: float,
) -> None:
    import matplotlib as mpl  # type: ignore
    import matplotlib.pyplot as plt  # type: ignore

    mpl.rcParams["font.family"] = mpl_rc_cfg.FONT_FAMILY
    mpl.rcParams["mathtext.fontset"] = mpl_rc_cfg.MATH_FONTSET

    # Match outputs/open_llm_leaderboard/old/v2_figure3_replication_arc_large.png
    panel_figsize = (0.75 * (2069.0 / 300.0), 1344.0 / 300.0)
    label_fs_x = float(frontier_1d_cfg.LABEL_FONTSIZE_X)
    label_fs_y = float(frontier_1d_cfg.LABEL_FONTSIZE_Y)
    tick_fs = float(frontier_1d_cfg.TICK_LABELSIZE)
    legend_fs = float(frontier_1d_cfg.LEGEND_FONTSIZE) * 0.75

    # Point styling (Figure 3 replication style)
    large_color = "#1f77b4"
    small_color = "firebrick"
    dominated_color = "gray"
    alpha_large = 0.8
    alpha_small = 0.7
    alpha_dominated = 0.5

    df = df.copy()
    df["_x"] = pd.to_numeric(df[x_col], errors="coerce")
    df["_y"] = pd.to_numeric(df[y_col], errors="coerce")
    df["_params_b"] = pd.to_numeric(df["params_b"], errors="coerce")
    df = df.dropna(subset=["_x", "_y", "_params_b"]).copy()
    df = df[df["_params_b"] > 0].copy()
    if df.empty:
        raise RuntimeError(f"No rows available for plot ({x_col} vs {y_col}).")

    df_small = df[df["_params_b"] < float(threshold_params_b)].copy()
    df_large = df[df["_params_b"] > float(threshold_params_b)].copy()

    dominated_mask = np.zeros((df_large.shape[0],), dtype=bool)
    if not df_small.empty and not df_large.empty:
        dominated_mask = _mark_dominated_large(
            small_x=df_small["_x"].to_numpy(),
            small_y=df_small["_y"].to_numpy(),
            large_x=df_large["_x"].to_numpy(),
            large_y=df_large["_y"].to_numpy(),
        )

    df_large_main = df_large.iloc[~dominated_mask].copy() if not df_large.empty else df_large
    dominated_large = df_large.iloc[dominated_mask].copy() if not df_large.empty else df_large

    fig, ax = plt.subplots(figsize=panel_figsize)
    if not df_large_main.empty:
        ax.scatter(
            df_large_main["_x"].to_numpy(),
            df_large_main["_y"].to_numpy(),
            s=26,
            alpha=alpha_large,
            color=large_color,
            label=f"large models > {threshold_params_b:g}B",
            rasterized=True,
            zorder=3,
        )
    if not df_small.empty:
        ax.scatter(
            df_small["_x"].to_numpy(),
            df_small["_y"].to_numpy(),
            s=10,
            alpha=alpha_small,
            color=small_color,
            label=f"small models < {threshold_params_b:g}B",
            rasterized=True,
            zorder=2,
        )
    if not dominated_large.empty:
        ax.scatter(
            dominated_large["_x"].to_numpy(),
            dominated_large["_y"].to_numpy(),
            s=38,
            alpha=alpha_dominated,
            facecolors="none",
            edgecolors=dominated_color,
            linewidths=1.2,
            label="dominated large",
            rasterized=True,
            zorder=1,
        )

    def _maybe_percent_label(col: str, series: pd.Series) -> str:
        try:
            return f"{col} (%)" if float(series.max()) > 1.0 else str(col)
        except Exception:
            return str(col)

    ax.set_xlabel(_maybe_percent_label(x_col, df["_x"]), fontweight="bold", fontsize=label_fs_x)
    ax.set_ylabel(_maybe_percent_label(y_col, df["_y"]), fontweight="bold", fontsize=label_fs_y)
    ax.tick_params(axis="both", labelsize=tick_fs)
    ax.grid(
        True,
        which="major",
        axis="y",
        linestyle=frontier_1d_cfg.GRID_MAJOR_LINESTYLE,
        linewidth=frontier_1d_cfg.GRID_MAJOR_LINEWIDTH,
        color=frontier_1d_cfg.GRID_MAJOR_COLOR,
        alpha=frontier_1d_cfg.GRID_MAJOR_ALPHA,
    )
    for spine in ax.spines.values():
        spine.set_linewidth(frontier_1d_cfg.SPINE_LINEWIDTH)
        spine.set_color(frontier_1d_cfg.SPINE_COLOR)
    ax.legend(loc="best", fontsize=legend_fs, frameon=True, framealpha=frontier_1d_cfg.LEGEND_FRAMEALPHA)
    fig.subplots_adjust(left=0.18, right=0.98, bottom=0.28, top=0.98)

    fig.savefig(out_path_base + ".png", dpi=300)
    fig.savefig(out_path_base + ".pdf", dpi=300)
    plt.close(fig)


def main(argv: Optional[Sequence[str]] = None) -> None:
    ap = argparse.ArgumentParser(description="Compare old vs current Open LLM Leaderboard tasks for shared models.")
    ap.add_argument(
        "--old_csv",
        default=os.path.join("tables", "open_llm_leaderboard", "open_llm_leaderboard_old_with_tokens.csv"),
    )
    ap.add_argument(
        "--new_csv",
        default=os.path.join("tables", "open_llm_leaderboard", "open_llm_leaderboard_with_tokens.csv"),
    )
    ap.add_argument("--out_dir", default=os.path.join("outputs", "open_llm_leaderboard", "comparison"))
    ap.add_argument("--join_key", default="eval_name")
    ap.add_argument("--threshold_params_b", type=float, default=13.0)
    ap.add_argument("--exclude_flagged", action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument("--exclude_merged", action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument("--exclude_moe", action=argparse.BooleanOptionalAction, default=False)

    args = ap.parse_args(list(argv) if argv is not None else None)

    out_dir = str(args.out_dir)
    _ensure_out_dir(out_dir)

    df_old = _load_and_filter(
        path=str(args.old_csv),
        exclude_flagged=bool(args.exclude_flagged),
        exclude_merged=bool(args.exclude_merged),
        exclude_moe=bool(args.exclude_moe),
    )
    df_new = _load_and_filter(
        path=str(args.new_csv),
        exclude_flagged=bool(args.exclude_flagged),
        exclude_merged=bool(args.exclude_merged),
        exclude_moe=bool(args.exclude_moe),
    )

    join_key = str(args.join_key)
    if join_key not in df_old.columns:
        raise KeyError(f"join_key={join_key!r} not in old CSV columns")
    if join_key not in df_new.columns:
        raise KeyError(f"join_key={join_key!r} not in new CSV columns")

    shared = sorted(set(df_old[join_key].dropna().astype(str)) & set(df_new[join_key].dropna().astype(str)))
    if not shared:
        raise RuntimeError("No shared models found between the two tables.")

    # Build one combined dataframe with the required columns for each plot.
    old_sub = df_old[df_old[join_key].astype(str).isin(shared)].copy()
    new_sub = df_new[df_new[join_key].astype(str).isin(shared)].copy()

    # Use the max of old/new params as a stable size proxy for the threshold split.
    merged = old_sub[[join_key, "#Params (B)"]].merge(
        new_sub[[join_key, "#Params (B)"]],
        on=join_key,
        how="inner",
        suffixes=("_old", "_new"),
    )
    merged["params_b"] = np.nanmax(
        np.stack(
            [
                pd.to_numeric(merged["#Params (B)_old"], errors="coerce").to_numpy(),
                pd.to_numeric(merged["#Params (B)_new"], errors="coerce").to_numpy(),
            ],
            axis=0,
        ),
        axis=0,
    )

    for x_col, y_col, slug in PAIR_SPECS:
        if x_col not in old_sub.columns:
            raise KeyError(f"Missing {x_col!r} in old table")
        if y_col not in new_sub.columns:
            raise KeyError(f"Missing {y_col!r} in new table")

        df_pair = merged[[join_key, "params_b"]].merge(old_sub[[join_key, x_col]], on=join_key, how="inner")
        df_pair = df_pair.merge(new_sub[[join_key, y_col]], on=join_key, how="inner")

        out_base = os.path.join(out_dir, f"task_pair_{slug}")
        _plot_pair(
            df=df_pair,
            x_col=x_col,
            y_col=y_col,
            out_path_base=out_base,
            threshold_params_b=float(args.threshold_params_b),
        )
        print(f"[ok] wrote {out_base}.png|.pdf (n={int(df_pair.shape[0])})")


if __name__ == "__main__":
    main()

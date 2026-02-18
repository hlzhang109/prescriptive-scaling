#!/usr/bin/env python3
"""
docs/aime_sigmoid tasks — AIME-2025 + sigmoid frontier.

Task 1:
  - Read `tables/artificial_analysis/aime-2025.csv`
  - Add:
      * `Pretraining compute` = 6 * params(B) * tokens(T)
      * `Source` (URL to an authoritative primary source)
    Only fill when params/tokens are explicitly documented by a primary source.

Task 1.2:
  - Fit a high-quantile sigmoid frontier on rows with pretraining compute available,
    matching the methodology and plotting style of:
      `scripts/epoch_ai/fit_benchmarks_runs_sigmoid_frontiers.py`
  - Write outputs to:
      `outputs/artificial_analysis/aime-2025/sigmoid/no_split/{points,curves,plots}/`
"""

from __future__ import annotations

import argparse
import datetime as _dt
import math
import os
import re
import sys
from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
import pandas as pd

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

try:
    from skill_frontier.io.output_paths import sanitize_task_name
except Exception:

    def sanitize_task_name(task: str) -> str:  # type: ignore[misc]
        name = task.replace(" ", "_")
        for ch in ['/', '\\', ':', '*', '?', '"', '<', '>', '|']:
            name = name.replace(ch, "_")
        return name


try:
    from scripts.smooth_single_skill_frontier import fit_sigmoid_frontier
except Exception:
    from smooth_single_skill_frontier import fit_sigmoid_frontier  # type: ignore

try:
    from skill_frontier.plotting.axis_formatting import apply_pretraining_compute_tick_multiplier
    from skill_frontier.plotting.configs import frontier_1d as frontier_1d_cfg
    from skill_frontier.plotting.configs import mpl_rc as mpl_rc_cfg
    from skill_frontier.plotting.labels import PRETRAINING_COMPUTE_FLOPS_LABEL
except Exception:  # pragma: no cover
    apply_pretraining_compute_tick_multiplier = None  # type: ignore

    class _Frontier1DCfg:  # type: ignore
        FIGSIZE = (7.0, 4.6)
        SCATTER_SIZE = 12
        SCATTER_ALPHA = 0.20
        SCATTER_LINEWIDTHS = 0.0
        CURVE_LINEWIDTH = 2.4
        CURVE_ALPHA = 0.95
        LABEL_FONTSIZE_X = 18
        LABEL_FONTSIZE_Y = 18
        TITLE_FONTSIZE = 18
        TICK_LABELSIZE = 15
        TICK_LENGTH = 4.0
        TICK_WIDTH = 0.9
        TICK_DIRECTION = "out"
        GRID_MAJOR_LINESTYLE = "-"
        GRID_MAJOR_LINEWIDTH = 0.9
        GRID_MAJOR_COLOR = "#d0d0d0"
        GRID_MAJOR_ALPHA = 0.7
        GRID_MINOR_LINESTYLE = ":"
        GRID_MINOR_LINEWIDTH = 0.6
        GRID_MINOR_COLOR = "#e0e0e0"
        GRID_MINOR_ALPHA = 0.7
        SPINE_LINEWIDTH = 1.1
        SPINE_COLOR = "#555555"
        LEGEND_FONTSIZE = 15
        LEGEND_FRAMEALPHA = 0.15
        LEGEND_LOC = "best"

    frontier_1d_cfg = _Frontier1DCfg()  # type: ignore

    class _MplRcCfg:  # type: ignore
        FONT_FAMILY = "serif"

    mpl_rc_cfg = _MplRcCfg()  # type: ignore
    PRETRAINING_COMPUTE_FLOPS_LABEL = "Pretraining Compute (FLOPs)"


X_TICK_MULTIPLIER: float = 1e21
TARGET_FIGSIZE: tuple[float, float] = (0.75 * (2069.0 / 300.0), 1344.0 / 300.0)


def _apply_frontier_style(ax, *, xlabel: str, ylabel: str, title: str) -> None:
    ax.set_xscale("log")
    ax.set_xlabel(xlabel, fontweight="bold", fontsize=frontier_1d_cfg.LABEL_FONTSIZE_X)
    if apply_pretraining_compute_tick_multiplier is not None and xlabel == PRETRAINING_COMPUTE_FLOPS_LABEL:
        apply_pretraining_compute_tick_multiplier(ax)
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


@dataclass(frozen=True)
class ComputeSource:
    params_b: float
    tokens_t: float
    source_url: str
    notes: str = ""

    @property
    def pretraining_compute(self) -> float:
        return float(6.0 * self.params_b * self.tokens_t)


_RE_FLOAT_B = re.compile(r"(?P<num>\d+(?:\.\d+)?)\s*B", re.IGNORECASE)
_RE_FLOAT_A_B = re.compile(r"A(?P<num>\d+(?:\.\d+)?)\s*B", re.IGNORECASE)


def _maybe_parse_single_b(name: str) -> Optional[float]:
    m = _RE_FLOAT_B.search(name)
    if not m:
        return None
    try:
        return float(m.group("num"))
    except Exception:
        return None


def _maybe_parse_activated_b(name: str) -> Optional[float]:
    m = _RE_FLOAT_A_B.search(name)
    if not m:
        return None
    try:
        return float(m.group("num"))
    except Exception:
        return None


def infer_compute_source(model_name: str) -> Optional[ComputeSource]:
    name = str(model_name or "").strip()
    low = name.lower()

    # Llama 2 (paper reports 2T tokens for pretraining)
    if low.startswith("llama 2") and "chat" in low:
        p = _maybe_parse_single_b(name)
        if p is None:
            return None
        return ComputeSource(
            params_b=float(p),
            tokens_t=2.0,
            source_url="https://arxiv.org/abs/2307.09288",
            notes="Llama 2 paper: trained on 2T tokens.",
        )

    # Llama 3 / 3.1 Instruct (avoid mapping other variants without explicit evidence)
    if low.startswith("llama 3 instruct") or low.startswith("llama 3.1 instruct"):
        p = _maybe_parse_single_b(name)
        if p is None:
            return None
        tokens_t = 15.0
        notes = "Llama 3 paper: pre-trained on about 15T tokens."
        if abs(p - 405.0) < 1e-6:
            tokens_t = 15.6
            notes = "Llama 3 paper: 405B trained on 15.6T tokens."
        return ComputeSource(
            params_b=float(p),
            tokens_t=float(tokens_t),
            source_url="https://arxiv.org/abs/2407.21783",
            notes=notes,
        )

    # Qwen2 Instruct: over 7T tokens
    if low.startswith("qwen2 ") and "instruct" in low:
        p = _maybe_parse_single_b(name)
        if p is None:
            return None
        return ComputeSource(
            params_b=float(p),
            tokens_t=7.0,
            source_url="https://arxiv.org/abs/2407.10671",
            notes="Qwen2 paper: over 7T tokens.",
        )

    # Qwen2.5 Instruct: 18T tokens
    if low.startswith("qwen2.5 ") and "instruct" in low and "coder" not in low:
        p = _maybe_parse_single_b(name)
        if p is None:
            return None
        return ComputeSource(
            params_b=float(p),
            tokens_t=18.0,
            source_url="https://arxiv.org/abs/2412.15115",
            notes="Qwen2.5 report: 18T tokens.",
        )

    # Qwen2.5 Coder Instruct: 5.5T tokens
    if low.startswith("qwen2.5 ") and "coder" in low and "instruct" in low:
        p = _maybe_parse_single_b(name)
        if p is None:
            return None
        return ComputeSource(
            params_b=float(p),
            tokens_t=5.5,
            source_url="https://arxiv.org/abs/2409.12186",
            notes="Qwen2.5-Coder report: 5.5T tokens.",
        )

    # Qwen3: 36T tokens; use activated params for MoE variants when specified as AxxB.
    if low.startswith("qwen3 "):
        p_act = _maybe_parse_activated_b(name)
        if p_act is not None:
            return ComputeSource(
                params_b=float(p_act),
                tokens_t=36.0,
                source_url="https://arxiv.org/abs/2505.09388",
                notes="Qwen3 report: 36T tokens; MoE uses activated params (AxxB).",
            )
        p = _maybe_parse_single_b(name)
        if p is None:
            return None
        return ComputeSource(
            params_b=float(p),
            tokens_t=36.0,
            source_url="https://arxiv.org/abs/2505.09388",
            notes="Qwen3 report: 36T tokens.",
        )

    # Gemma 3: token budgets by size (paper)
    if low.startswith("gemma 3 ") and "instruct" in low and "gemma 3n" not in low:
        p = _maybe_parse_single_b(name)
        if p is None:
            return None
        tokens_map = {1.0: 2.0, 4.0: 4.0, 12.0: 12.0, 27.0: 14.0}
        tokens_t = tokens_map.get(float(p))
        if tokens_t is None:
            return None
        return ComputeSource(
            params_b=float(p),
            tokens_t=float(tokens_t),
            source_url="https://arxiv.org/abs/2503.19786",
            notes="Gemma 3 paper: token budgets by size.",
        )

    # Phi-3 mini 3.8B: 3.3T tokens (paper)
    if low.startswith("phi-3") and "mini" in low and "3.8b" in low:
        return ComputeSource(
            params_b=3.8,
            tokens_t=3.3,
            source_url="https://arxiv.org/abs/2404.14219",
            notes="Phi-3 report: phi-3-mini 3.8B trained on 3.3T tokens.",
        )

    # Phi-4: ~10T tokens (paper)
    if low.strip() == "phi-4":
        return ComputeSource(
            params_b=14.0,
            tokens_t=10.0,
            source_url="https://arxiv.org/abs/2412.08905",
            notes="Phi-4 report: 14B pretrained for ~10T tokens.",
        )

    return None


def augment_aime_csv(in_csv: str, out_csv: str, record_md: str) -> None:
    df = pd.read_csv(in_csv)
    if "Pretraining compute" not in df.columns:
        df["Pretraining compute"] = np.nan
    if "Source" not in df.columns:
        df["Source"] = ""

    filled = 0
    used_sources: dict[str, str] = {}

    for idx, row in df.iterrows():
        model_name = str(row.get("model_name", "") or "").strip()
        cs = infer_compute_source(model_name)
        if cs is None:
            continue
        c = cs.pretraining_compute
        if not (np.isfinite(c) and c > 0):
            continue
        df.at[idx, "Pretraining compute"] = float(c)
        df.at[idx, "Source"] = str(cs.source_url)
        used_sources[str(cs.source_url)] = cs.notes
        filled += 1

    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    df.to_csv(out_csv, index=False)

    os.makedirs(os.path.dirname(record_md) or ".", exist_ok=True)
    now = _dt.datetime.now().astimezone().strftime("%Y-%m-%d %H:%M %Z")
    lines = [
        "# AIME-2025 pretraining-compute sources",
        "",
        "- Goal: Fill `Pretraining compute = 6 * params(B) * tokens(T)` for AIME-2025 models using primary sources only.",
        "- When generated: " + now,
        "- Notes: For Qwen3 MoE variants `AxxB`, we use **activated parameters** (AxxB) per the Qwen3 report.",
        "",
        "## Sources used",
    ]
    for url in sorted(used_sources.keys()):
        note = used_sources[url]
        lines.append(f"- {url}")
        if note:
            lines.append(f"  - Evidence: {note}")
    lines.append("")
    lines.append("## Filled rows")
    lines.append(f"- Filled `Pretraining compute` for {filled}/{len(df)} rows.")
    with open(record_md, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def fit_and_plot(
    *,
    csv_path: str,
    out_dir: str,
    task_name: str,
    tau: float,
    kappa: float,
    lambda_b: float,
    min_points: int,
    score_col: str,
    compute_col: str,
    model_col: str,
) -> None:
    df = pd.read_csv(csv_path)
    missing = [c for c in (compute_col, score_col, model_col) if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    x = pd.to_numeric(df[compute_col], errors="coerce").to_numpy(float)
    y = pd.to_numeric(df[score_col], errors="coerce").to_numpy(float)
    keep = np.isfinite(x) & (x > 0) & np.isfinite(y)
    x = x[keep]
    y = y[keep]
    models = df.loc[keep, model_col].astype(str).tolist()

    if x.size < int(min_points):
        raise RuntimeError(f"Not enough points with compute to fit: n={x.size} < {min_points}")

    y = np.clip(y, 0.0, 1.0)

    safe_task = sanitize_task_name(task_name)
    out_points = os.path.join(out_dir, "sigmoid", "no_split", "points")
    out_curves = os.path.join(out_dir, "sigmoid", "no_split", "curves")
    out_plots = os.path.join(out_dir, "sigmoid", "no_split", "plots")
    for d in (out_points, out_curves, out_plots):
        os.makedirs(d, exist_ok=True)

    points_path = os.path.join(out_points, f"{safe_task}.csv")
    with open(points_path, "w", encoding="utf-8") as f:
        f.write("model,training_compute_flop,log10_compute,score\n")
        for m, x_scaled, yv in zip(models, x, y):
            c_abs = float(x_scaled) * float(X_TICK_MULTIPLIER)
            z = math.log10(c_abs) if c_abs > 0 else float("nan")
            f.write(f"{m},{c_abs:.12g},{z:.12g},{float(yv):.6g}\n")

    xs_curve_scaled, y_curve = fit_sigmoid_frontier(
        x,
        y,
        tau=float(tau),
        use_log10_x=True,
        fit_mode="quantile_per_point",
        kappa_final=float(kappa),
        lambda_b=float(lambda_b),
    )
    if xs_curve_scaled.size == 0:
        raise RuntimeError("Sigmoid fit failed (empty curve)")

    curve_path = os.path.join(out_curves, f"{safe_task}.csv")
    with open(curve_path, "w", encoding="utf-8") as f:
        f.write("training_compute_flop,y_hat,task\n")
        for x_scaled, yv in zip(xs_curve_scaled, y_curve):
            c_abs = float(x_scaled) * float(X_TICK_MULTIPLIER)
            f.write(f"{c_abs:.12g},{float(yv):.12g},{task_name}\n")

    import matplotlib as mpl  # type: ignore
    import matplotlib.pyplot as plt  # type: ignore

    mpl.rcParams["font.family"] = mpl_rc_cfg.FONT_FAMILY
    fig, ax = plt.subplots(figsize=TARGET_FIGSIZE)
    ax.scatter(
        x,
        y,
        s=float(frontier_1d_cfg.SCATTER_SIZE) * 4.5,
        alpha=0.6,
        color="black",
        marker="X",
        label="points",
        linewidths=frontier_1d_cfg.SCATTER_LINEWIDTHS,
        rasterized=True,
    )
    ax.plot(
        xs_curve_scaled,
        y_curve,
        color="firebrick",
        linewidth=frontier_1d_cfg.CURVE_LINEWIDTH,
        label=f"Smooth τ={float(tau):.2f}",
        alpha=frontier_1d_cfg.CURVE_ALPHA,
    )
    _apply_frontier_style(
        ax,
        xlabel=PRETRAINING_COMPUTE_FLOPS_LABEL,
        ylabel="Best score",
        title="",
    )
    y_min = float(np.nanmin([np.nanmin(y), np.nanmin(y_curve)]))
    y_max = float(np.nanmax([np.nanmax(y), np.nanmax(y_curve)]))
    if not np.isfinite(y_min) or not np.isfinite(y_max) or y_min == y_max:
        y_min, y_max = 0.0, 1.0
    pad = 0.02 * max(1e-6, (y_max - y_min))
    ax.set_ylim(y_min - pad, y_max + pad)
    ax.legend(
        loc=frontier_1d_cfg.LEGEND_LOC,
        fontsize=frontier_1d_cfg.LEGEND_FONTSIZE,
        frameon=True,
        framealpha=frontier_1d_cfg.LEGEND_FRAMEALPHA,
    )
    fig.tight_layout()
    out_base = os.path.join(out_plots, safe_task)
    fig.savefig(out_base + ".png", dpi=300)
    fig.savefig(out_base + ".pdf", dpi=300)
    plt.close(fig)


def main(argv: Optional[Sequence[str]] = None) -> None:
    ap = argparse.ArgumentParser(description="AIME-2025 compute augmentation + sigmoid frontier")
    ap.add_argument("--in_csv", default=os.path.join("tables", "artificial_analysis", "aime-2025.csv"))
    ap.add_argument(
        "--out_csv",
        default=os.path.join("tables", "artificial_analysis", "aime-2025_with_tokens.csv"),
    )
    ap.add_argument(
        "--out_dir",
        default=os.path.join("outputs", "artificial_analysis", "aime-2025"),
        help="Output directory root (will create sigmoid/no_split subfolders)",
    )
    ap.add_argument("--tau", type=float, default=0.98)
    ap.add_argument("--kappa", type=float, default=50.0)
    ap.add_argument("--lambda_b", type=float, default=1e-4)
    ap.add_argument("--min_points", type=int, default=20)
    args = ap.parse_args(argv)

    record_md = os.path.join(str(args.out_dir), "source_record.md")
    augment_aime_csv(str(args.in_csv), str(args.out_csv), record_md)
    fit_and_plot(
        csv_path=str(args.out_csv),
        out_dir=str(args.out_dir),
        task_name="AIME_2025",
        tau=float(args.tau),
        kappa=float(args.kappa),
        lambda_b=float(args.lambda_b),
        min_points=int(args.min_points),
        score_col="aime_2025",
        compute_col="Pretraining compute",
        model_col="model_name",
    )
    print(f"[aime_sigmoid] wrote: {args.out_csv}")
    print(f"[aime_sigmoid] outputs in: {args.out_dir}")


if __name__ == "__main__":  # pragma: no cover
    main()

#!/usr/bin/env python3
"""
Generate pairwise envelope plots for LiveBench or Open LLM Leaderboard.

This script detects task columns automatically based on repository conventions
used in scripts/run_frontier_from_csv.py and scripts/skill_frontier.py:
  - LiveBench: tasks are columns starting with prefix 'a_' in tables/merged_livebench.csv.
  - Open LLM Leaderboard: tasks are the 'Raw' summary columns present among
    ['IFEval Raw','BBH Raw','MATH Lvl 5 Raw','GPQA Raw','MUSR Raw','MMLU-PRO Raw']
    in tables/open_llm_leaderboard/open_llm_leaderboard_old_with_tokens.csv by default.

Outputs are saved under pairwise_envelope/<dataset>/ as both PDF and PNG with
filename 'envelope_pairwise__<TaskX>__<TaskY>.(pdf|png)', dpi=300, tight layout.

Usage examples:
  # LiveBench
  python scripts/data/generate_envelopes.py --dataset livebench

  # Open LLM Leaderboard (old_with_tokens)
  python scripts/data/generate_envelopes.py --dataset oll

You can override CSV paths and the automatic task detection via flags.
"""

from __future__ import annotations

import argparse
import csv as _csv
import os
import itertools
from typing import Dict, List, Sequence, Tuple, Optional

import numpy as np

# Local import with robust fallback: learned monotone envelopes + isoquant
import sys
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
try:
    from skill_frontier.core.envelope import (
        fit_upper_lower_with_tradeoff,
        fit_isoquant,
    )
except Exception:  # pragma: no cover
    try:
        from scripts.monotone_envelope import (
            fit_upper_lower_with_tradeoff,
            fit_isoquant,
        )
    except Exception:
        from monotone_envelope import (  # type: ignore
            fit_upper_lower_with_tradeoff,
            fit_isoquant,
        )

from skill_frontier.io.csv_utils import detect_oll_raw_tasks, maybe_scale_task_values  # type: ignore
from skill_frontier.plotting.model_families import (  # type: ignore
    FAMILY_ORDER,
    color_for_family,
    extract_base_model_name,
    family_from_base_model,
)
from skill_frontier.plotting.configs import envelopes as envelopes_cfg  # type: ignore


def _read_csv_header(path: str) -> List[str]:
    with open(path, "r", newline="") as f:
        reader = _csv.reader(f)
        header = next(reader)
    return [h.strip() for h in header]


def _detect_livebench_tasks(csv_path: str, prefix: str = "a_") -> List[str]:
    headers = _read_csv_header(csv_path)
    tasks = [c for c in headers if c.startswith(prefix)]
    if not tasks:
        raise RuntimeError(
            f"No LiveBench tasks found with prefix '{prefix}' in {csv_path}. Columns: {headers}"
        )
    return tasks


def _detect_oll_tasks(csv_path: str) -> List[str]:
    """Detect OLL task columns for supported schemas (new and old)."""
    headers = _read_csv_header(csv_path)
    tasks = detect_oll_raw_tasks(headers)
    if not tasks:
        raise RuntimeError(
            f"No supported OLL task columns auto-detected in {csv_path}. "
            "Pass --tasks explicitly to override."
        )
    return tasks


def _read_two_cols(csv_path: str, xcol: str, ycol: str) -> tuple[np.ndarray, np.ndarray]:
    xs: List[float] = []
    ys: List[float] = []
    with open(csv_path, "r", newline="") as f:
        reader = _csv.DictReader(f)
        for row in reader:
            vx = row.get(xcol, None)
            vy = row.get(ycol, None)
            if vx is None or vy is None:
                continue
            try:
                x = float(vx)
                y = float(vy)
            except Exception:
                continue
            if np.isfinite(x) and np.isfinite(y):
                xs.append(x)
                ys.append(y)
    return np.array(xs, dtype=float), np.array(ys, dtype=float)


def _read_two_cols_with_stats(csv_path: str, xcol: str, ycol: str) -> Tuple[np.ndarray, np.ndarray, Dict[str, int]]:
    rows = 0
    finite_x = 0
    finite_y = 0
    finite_both = 0
    xs: List[float] = []
    ys: List[float] = []
    with open(csv_path, "r", newline="") as f:
        reader = _csv.DictReader(f)
        for row in reader:
            rows += 1
            vx = row.get(xcol, None)
            vy = row.get(ycol, None)
            okx = oky = False
            try:
                x = float(vx) if vx is not None else float("nan")
                okx = np.isfinite(x)
            except Exception:
                x = float("nan"); okx = False
            try:
                y = float(vy) if vy is not None else float("nan")
                oky = np.isfinite(y)
            except Exception:
                y = float("nan"); oky = False
            if okx:
                finite_x += 1
            if oky:
                finite_y += 1
            if okx and oky:
                finite_both += 1
                xs.append(x)
                ys.append(y)
    stats = {
        "rows": rows,
        "finite_x": finite_x,
        "finite_y": finite_y,
        "finite_both": finite_both,
        "used_points": len(xs),
    }
    return np.array(xs, dtype=float), np.array(ys, dtype=float), stats


def _csv_shape(path: str) -> Tuple[int, int]:
    cols = 0
    rows = 0
    with open(path, "r", newline="") as f:
        reader = _csv.reader(f)
        try:
            header = next(reader)
        except StopIteration:
            return 0, 0
        cols = len(header)
        for _ in reader:
            rows += 1
    return rows, cols


def _read_two_cols_with_compute(
    path: str,
    xcol: str,
    ycol: str,
    compute_multiplier: float = 6.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Read x,y plus compute C for rows with all three finite.

    Compute logic mirrors scripts/skill_frontier.py:
      - If 'logC' column exists: C = exp(logC).
      - Else, if both 'Pretraining tokens (T)' and '#Params (B)' exist: C = 6*T*B.
    Rows missing any of x,y,C are dropped.
    """
    # Detect header
    with open(path, "r", newline="") as f:
        reader = _csv.DictReader(f)
        headers = reader.fieldnames or []
        has_logC = "logC" in headers
        has_prod = ("Pretraining tokens (T)" in headers) and ("#Params (B)" in headers)
        xs: List[float] = []
        ys: List[float] = []
        Cs: List[float] = []
        bases: List[str] = []
        for row in reader:
            vx = row.get(xcol, None); vy = row.get(ycol, None)
            try:
                x = float(vx) if vx not in (None, "", "nan", "NaN") else float("nan")
                y = float(vy) if vy not in (None, "", "nan", "NaN") else float("nan")
            except Exception:
                x = y = float("nan")
            if not (np.isfinite(x) and np.isfinite(y)):
                continue
            C = float("nan")
            if has_logC:
                vlc = row.get("logC", None)
                try:
                    lc = float(vlc) if vlc not in (None, "", "nan", "NaN") else float("nan")
                    C = float(np.exp(lc)) if np.isfinite(lc) else float("nan")
                except Exception:
                    C = float("nan")
            elif has_prod:
                vt = row.get("Pretraining tokens (T)", None)
                vb = row.get("#Params (B)", None)
                try:
                    T = float(vt) if vt not in (None, "", "nan", "NaN") else float("nan")
                except Exception:
                    T = float("nan")
                try:
                    B = float(vb) if vb not in (None, "", "nan", "NaN") else float("nan")
                except Exception:
                    B = float("nan")
                if np.isfinite(T) and np.isfinite(B):
                    C = float(compute_multiplier * T * B)
            if not np.isfinite(C):
                continue
            xs.append(x)
            ys.append(y)
            Cs.append(C)
            bases.append(extract_base_model_name(row))
    x_arr = maybe_scale_task_values(np.array(xs, dtype=float))
    y_arr = maybe_scale_task_values(np.array(ys, dtype=float))
    return x_arr, y_arr, np.array(Cs, dtype=float), np.array(bases, dtype=object)


def _sanitize(name: str) -> str:
    # Replace path separators; allow spaces and common characters for readability
    out = name.replace("/", "_").replace("\\", "_")
    return out

def _plot_and_save(
    x: np.ndarray,
    y: np.ndarray,
    base_models: np.ndarray,
    task_x: str,
    task_y: str,
    out_base: str,
    scatter_style: str,
) -> None:
    # Fit models
    g, f, iso, comb = fit_upper_lower_with_tradeoff(x, y, label=f"{task_x} x {task_y} (overall)", K=10)
    # Plot with matplotlib directly to control dpi/layout and save both formats
    import matplotlib.pyplot as plt  # type: ignore
    try:
        from scipy.interpolate import PchipInterpolator  # type: ignore
    except Exception:
        PchipInterpolator = None  # type: ignore

    # Axes limits: min/max of scatter points
    try:
        x_lo = float(np.nanmin(x)) if x.size else 0.0
        x_hi = float(np.nanmax(x)) if x.size else 1.0
        y_lo = float(np.nanmin(y)) if y.size else 0.0
        y_hi = float(np.nanmax(y)) if y.size else 1.0
    except Exception:
        x_lo, x_hi, y_lo, y_hi = 0.0, 1.0, 0.0, 1.0

    xs = np.linspace(x_lo, x_hi, num=400)
    # Components
    y_upper_raw = g.predict(xs)
    y_lower_raw = f.predict(xs)
    # Smooth the boundaries with a monotone cubic interpolator (PCHIP) if available
    if PchipInterpolator is not None:
        try:
            y_upper_raw = PchipInterpolator(xs, y_upper_raw, extrapolate=False)(xs)
            y_lower_raw = PchipInterpolator(xs, y_lower_raw, extrapolate=False)(xs)
        except Exception:
            pass
    # Compute multiple trade-off isoquants (tau in {0.90, 0.95, 0.98})
    trade_taus = [0.90, 0.95, 0.98]
    iso_list = []
    for t in trade_taus:
        try:
            iso_t = fit_isoquant(x, y, tau=t, g_model=g)
            iso_list.append((t, iso_t))
        except Exception:
            pass
    plt.figure(figsize=envelopes_cfg.PAIRWISE_FIGSIZE)
    # Plot all points (no window cropping), colored by base-model family.
    if str(scatter_style) == "family" and base_models is not None and base_models.size == x.size:
        fams = np.asarray([family_from_base_model(b) for b in base_models], dtype=object)
        for fam in FAMILY_ORDER:
            m = fams == fam
            if not np.any(m):
                continue
            plt.scatter(
                x[m],
                y[m],
                s=envelopes_cfg.PAIRWISE_SCATTER_SIZE,
                alpha=envelopes_cfg.PAIRWISE_SCATTER_ALPHA,
                color=color_for_family(str(fam)),
                label="_nolegend_",
            )
        plt.scatter(
            [],
            [],
            s=envelopes_cfg.PAIRWISE_SCATTER_SIZE,
            alpha=envelopes_cfg.PAIRWISE_SCATTER_ALPHA,
            color="#7f7f7f",
            label="points",
        )
    else:
        plt.scatter(
            x,
            y,
            s=envelopes_cfg.PAIRWISE_SCATTER_SIZE,
            alpha=envelopes_cfg.PAIRWISE_SCATTER_ALPHA,
            color="#1f77b4",
            label="points",
        )
    # Upper envelope (this is the upper boundary)
    plt.plot(
        xs,
        y_upper_raw,
        label=f"Upper envelope g (τ={g.tau:.2f})",
        color="#1f77b4",
        linewidth=envelopes_cfg.UPPER_LINE_WIDTH,
        linestyle=envelopes_cfg.UPPER_LINESTYLE,
        zorder=4,
    )
    # Trade-off boundaries for multiple taus
    trade_colors = {0.90: "#d62728", 0.95: "#ff7f0e", 0.98: "#2ca02c"}
    trade_styles = {0.90: (0, (4, 2)), 0.95: (0, (2, 2)), 0.98: (0, (1, 2))}
    for t, iso_t in iso_list:
        y_tradeoff = iso_t.y_of_x(xs)
        plt.plot(
            xs,
            y_tradeoff,
            label=f"Trade-off y_h(x), τ={t:.2f}",
            color=trade_colors.get(t, "#d62728"),
            linewidth=envelopes_cfg.TRADEOFF_LINE_WIDTH,
            linestyle=trade_styles.get(t, (0, (4, 2))),
        )
    # Lower boundary component
    plt.plot(
        xs,
        y_lower_raw,
        label=f"Lower boundary f (τ={f.tau:.2f})",
        color="#2ca02c",
        linewidth=envelopes_cfg.LOWER_LINE_WIDTH,
        linestyle=envelopes_cfg.LOWER_LINESTYLE,
    )
    # Elbow marker
    plt.scatter(
        [comb.x_star],
        [comb.y_star],
        color="#6a3d9a",
        s=envelopes_cfg.ELBOW_MARKER_SIZE,
        zorder=5,
        label="elbow",
    )
    plt.xlabel(task_x)
    plt.ylabel(task_y)
    plt.title(f"Envelope — {task_x} × {task_y}")
    plt.xlim(x_lo, x_hi)
    plt.ylim(y_lo, y_hi)
    leg = plt.legend(loc="best", fontsize=envelopes_cfg.LEGEND_FONTSIZE, title=envelopes_cfg.LEGEND_TITLE)
    if leg and leg.get_title():
        leg.get_title().set_fontweight("bold")
    plt.tight_layout()
    png_path = out_base + ".png"
    pdf_path = out_base + ".pdf"
    plt.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.savefig(pdf_path, dpi=300, bbox_inches="tight")
    plt.close()


def _plot_bands(
    x: np.ndarray,
    y: np.ndarray,
    C: np.ndarray,
    base_models: np.ndarray,
    task_x: str,
    task_y: str,
    out_base: str,
    scatter_style: str,
    K_sub: int = 10,
) -> None:
    import matplotlib.pyplot as plt  # type: ignore
    import numpy as np
    # Band thresholds as in scripts/skill_frontier.plot_pairwise_frontiers
    c1, c2 = 54.0, 756.0
    masks = [
        (C <= c1),
        (C > c1) & (C <= c2),
        (C > c2),
    ]
    titles = [f"C ≤ {c1:.3g}", f"{c1:.3g} < C ≤ {c2:.3g}", f"C > {c2:.3g}"]
    # Prepare figure with three subplots side by side
    fig, axes = plt.subplots(1, 3, figsize=envelopes_cfg.TRIPTYCH_FIGSIZE, squeeze=False)
    axes = axes[0]
    # Per-band axes set by each band's scatter min/max

    for k in range(3):
        ax = axes[k]
        mk = masks[k] & np.isfinite(x) & np.isfinite(y)
        xk, yk = x[mk], y[mk]
        bk = base_models[mk] if base_models is not None and base_models.size == x.size else None
        # Default axes if we cannot infer from data
        x_lo, x_hi, y_lo, y_hi = 0.0, 1.0, 0.0, 1.0
        if xk.size > 0:
            try:
                x_lo = float(np.nanmin(xk)); x_hi = float(np.nanmax(xk))
                y_lo = float(np.nanmin(yk)); y_hi = float(np.nanmax(yk))
            except Exception:
                pass
        if xk.size >= max(3, K_sub):
            # Compute per-band limits based on points in this band
            try:
                x_lo = float(np.nanmin(xk)); x_hi = float(np.nanmax(xk))
                y_lo = float(np.nanmin(yk)); y_hi = float(np.nanmax(yk))
            except Exception:
                x_lo, x_hi, y_lo, y_hi = 0.0, 1.0, 0.0, 1.0

            try:
                gk, fk, isok, combk = fit_upper_lower_with_tradeoff(
                    xk, yk, K=10, label=f"{task_x} x {task_y} band={k+1}"
                )
            except Exception as exc:
                print(f"[bands] skip {task_x} x {task_y} band {k+1}: {exc}")
                ax.set_xlim(x_lo, x_hi)
                ax.set_ylim(y_lo, y_hi)
                continue
            xs = np.linspace(x_lo, x_hi, num=400)
            y_upper_raw = gk.predict(xs)
            # Compute multiple trade-off isoquants (reuse gk)
            trade_taus = [0.90, 0.95, 0.98]
            iso_list = []
            for t in trade_taus:
                try:
                    iso_t = fit_isoquant(xk, yk, tau=t, g_model=gk)
                    iso_list.append((t, iso_t))
                except Exception:
                    pass
            y_lower_raw = fk.predict(xs)
            try:
                from scipy.interpolate import PchipInterpolator  # type: ignore
                y_upper_raw = PchipInterpolator(xs, y_upper_raw, extrapolate=False)(xs)
                y_lower_raw = PchipInterpolator(xs, y_lower_raw, extrapolate=False)(xs)
            except Exception:
                pass
            # Upper envelope = upper boundary
            ax.plot(xs, y_upper_raw, color="#1f77b4", linewidth=2.2, linestyle="solid", label=f"Upper envelope g (τ={getattr(gk,'tau',0.90):.2f})")
            trade_colors = {0.90: "#d62728", 0.95: "#ff7f0e", 0.98: "#2ca02c"}
            trade_styles = {0.90: (0, (4, 2)), 0.95: (0, (2, 2)), 0.98: (0, (1, 2))}
            for t, iso_t in iso_list:
                y_trade = iso_t.y_of_x(xs)
                ax.plot(xs, y_trade, color=trade_colors.get(t, "#d62728"), linewidth=1.6, linestyle=trade_styles.get(t, (4, 2)), label=f"Trade-off y_h, τ={t:.2f}")
            ax.plot(xs, y_lower_raw, color="#2ca02c", linewidth=1.6, linestyle="dashed", label=f"Lower boundary f (τ={getattr(fk,'tau',0.10):.2f})")
        # Plot all points in this band (colored by base-model family).
        if str(scatter_style) == "family" and bk is not None and isinstance(bk, np.ndarray) and bk.size == xk.size:
            fams = np.asarray([family_from_base_model(b) for b in bk], dtype=object)
            for fam in FAMILY_ORDER:
                m = fams == fam
                if not np.any(m):
                    continue
                ax.scatter(xk[m], yk[m], s=10, alpha=0.25, color=color_for_family(str(fam)), label="_nolegend_")
            ax.scatter([], [], s=10, alpha=0.25, color="#7f7f7f", label="points")
        else:
            ax.scatter(xk, yk, s=10, alpha=0.25, color="#1f77b4", label="points")
        ax.set_xlabel(task_x)
        if k == 0:
            ax.set_ylabel(task_y)
        ax.set_title(titles[k])
        ax.set_xlim(x_lo, x_hi)
        ax.set_ylim(y_lo, y_hi)
        if k == 0:
            leg = ax.legend(loc="best", fontsize=envelopes_cfg.TRIPTYCH_LEGEND_FONTSIZE)
    plt.tight_layout()
    plt.savefig(out_base + ".png", dpi=300, bbox_inches="tight")
    plt.savefig(out_base + ".pdf", dpi=300, bbox_inches="tight")
    plt.close()


def main(argv: Sequence[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Generate pairwise monotone envelopes for LiveBench or OLL")
    p.add_argument("--dataset", choices=["livebench", "oll", "open_llm_leaderboard"], required=True)
    p.add_argument("--livebench_csv", default="tables/merged_livebench.csv", help="LiveBench merged CSV path")
    p.add_argument(
        "--oll_csv",
        default="tables/open_llm_leaderboard/open_llm_leaderboard_with_tokens.csv",
        help="Open LLM Leaderboard CSV path (with_tokens schema; has Raw columns)",
    )
    p.add_argument("--out_root", default="pairwise_envelope", help="Base output directory")
    p.add_argument("--tasks", nargs="*", default=None, help="Optional explicit task list to use for pairwise plots (overrides auto resolution for the dataset)")
    p.add_argument(
        "--scatter_style",
        choices=["two_color", "family"],
        default="two_color",
        help="Scatter styling: 'two_color' (default) disables model-family coloring; 'family' colors points by base-model family.",
    )
    args = p.parse_args(argv)

    os.makedirs(args.out_root, exist_ok=True)

    if args.dataset == "livebench":
        csv_path = args.livebench_csv
        tasks = args.tasks if args.tasks else _detect_livebench_tasks(csv_path)
        out_dir = os.path.join(args.out_root, "livebench")
    else:
        csv_path = args.oll_csv
        tasks = args.tasks if args.tasks else _detect_oll_tasks(csv_path)
        out_dir = os.path.join(args.out_root, "open_llm_leaderboard")

    os.makedirs(out_dir, exist_ok=True)

    # All unordered task pairs
    pairs = list(itertools.combinations(tasks, 2))
    for a, b in pairs:
        # Read data and compute
        x, y, C, base_models = _read_two_cols_with_compute(csv_path, a, b)
        if x.size == 0:
            continue
        pair_dir = os.path.join(out_dir, f"{_sanitize(a)}__{_sanitize(b)}")
        os.makedirs(pair_dir, exist_ok=True)
        # Overall plot
        _plot_and_save(x, y, base_models, a, b, os.path.join(pair_dir, "overall"), str(args.scatter_style))
        # Three-band plot using compute bands
        _plot_bands(x, y, C, base_models, a, b, os.path.join(pair_dir, "bands"), str(args.scatter_style), K_sub=10)


if __name__ == "__main__":  # pragma: no cover
    main()

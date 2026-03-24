#!/usr/bin/env python3
"""
2D Sigmoid Frontier Visualization (Compute + Ratio)
--------------------------------------------------

Fits and visualizes a nested "compute+ratio" extension of the compute-only
sigmoid frontier, using *no split* (all rows).

Definitions (log-space):
  - tokens: T (column)
  - params: P (column)
  - z      = log10(mult * T * P)               # compute proxy
  - r      = log10(T) - log10(P)               # log ratio (tokens vs params)
  - z_eff  = z + alpha * r                     # effective compute

Model (upper-quantile frontier):
  y_hat = y0 + L * sigmoid(b * (z_eff - z_star))

This is a strict generalization of the compute-only model: alpha = 0 recovers
the original 1D compute-based frontier (up to the same pinball objective).

Outputs (--out_dir):
  - fit_params.csv                # per-task best alpha and sigmoid params
  - pred_grid__<task>.csv         # grid predictions in (logT, logP) space
  - plots/surface__<task>.png/pdf # 2D contour visualization
  - plots/slices__<task>.png/pdf  # predicted y vs ratio at fixed compute levels
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

# When executed as `python scripts/plot/diagnostics/plot_sigmoid_frontier_2d.py`, Python's import root
# becomes `scripts/plot/diagnostics/`, so we must add the repo root (three levels up)
# to import `skill_frontier.*` and `scripts.*` consistently.
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, os.pardir))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from skill_frontier.io.csv_utils import (  # type: ignore
    detect_oll_raw_tasks,
    maybe_scale_task_values,
    read_csv_rows,
)
from skill_frontier.io.output_paths import sanitize_task_name  # type: ignore
from skill_frontier.plotting.model_families import (  # type: ignore
    FAMILY_ORDER,
    color_for_family,
    extract_base_model_name,
    family_from_base_model,
)
from skill_frontier.plotting.configs import frontier_1d as frontier_1d_cfg  # type: ignore
from skill_frontier.plotting.configs import mpl_rc as mpl_rc_cfg  # type: ignore
from skill_frontier.plotting.configs import sigmoid2d as sigmoid2d_cfg  # type: ignore


def _configure_matplotlib(mpl, use_tex: bool) -> None:
    """Apply common matplotlib rcParams (optionally enabling LaTeX)."""
    try:
        mpl.rcParams["font.family"] = mpl_rc_cfg.FONT_FAMILY
        # LaTeX-like math rendering even when not using external LaTeX.
        mpl.rcParams["mathtext.fontset"] = mpl_rc_cfg.MATH_FONTSET
        if use_tex:
            mpl.rcParams["text.usetex"] = True
            mpl.rcParams["text.latex.preamble"] = mpl_rc_cfg.LATEX_PREAMBLE
    except Exception:
        pass


def _import_sigmoid_optimizer():
    try:
        from scripts.run.sigmoid_quantile_optimizer import fit_sigmoid_enhanced, sigmoid_pred  # type: ignore
    except Exception:
        try:
            from sigmoid_quantile_optimizer import fit_sigmoid_enhanced, sigmoid_pred  # type: ignore
        except Exception as e:  # pragma: no cover
            raise SystemExit("Could not import sigmoid_quantile_optimizer (need numpy+scipy)") from e
    return fit_sigmoid_enhanced, sigmoid_pred


def _to_float(v: object) -> float:
    try:
        x = float(v)  # type: ignore[arg-type]
        return float(x)
    except Exception:
        return float("nan")


@dataclass
class TaskData:
    logT: np.ndarray
    logP: np.ndarray
    z: np.ndarray
    r: np.ndarray
    y: np.ndarray
    base_models: np.ndarray


def _load_task_data(
    rows: List[dict],
    task: str,
    tokens_col: str,
    params_col: str,
    compute_multiplier: float,
) -> TaskData:
    logT_list: List[float] = []
    logP_list: List[float] = []
    z_list: List[float] = []
    r_list: List[float] = []
    y_list: List[float] = []
    base_list: List[str] = []

    log_mult = float(np.log10(max(float(compute_multiplier), 1e-300)))
    for row in rows:
        T = _to_float(row.get(tokens_col, "nan"))
        P = _to_float(row.get(params_col, "nan"))
        yv = _to_float(row.get(task, "nan"))
        if not (np.isfinite(T) and np.isfinite(P) and np.isfinite(yv)):
            continue
        if T <= 0.0 or P <= 0.0:
            continue
        logT = float(np.log10(T))
        logP = float(np.log10(P))
        z = log_mult + logT + logP
        r = logT - logP
        logT_list.append(logT)
        logP_list.append(logP)
        z_list.append(z)
        r_list.append(r)
        y_list.append(float(yv))
        base_list.append(extract_base_model_name(row))

    logT_arr = np.asarray(logT_list, float)
    logP_arr = np.asarray(logP_list, float)
    z_arr = np.asarray(z_list, float)
    r_arr = np.asarray(r_list, float)
    y_arr = np.asarray(y_list, float)
    base_arr = np.asarray(base_list, dtype=object)

    y_arr = maybe_scale_task_values(y_arr)
    y_arr = np.clip(y_arr, 0.0, 1.0)

    keep = np.isfinite(logT_arr) & np.isfinite(logP_arr) & np.isfinite(z_arr) & np.isfinite(r_arr) & np.isfinite(y_arr)
    return TaskData(
        logT=logT_arr[keep],
        logP=logP_arr[keep],
        z=z_arr[keep],
        r=r_arr[keep],
        y=y_arr[keep],
        base_models=base_arr[keep],
    )


def _quantile_range(x: np.ndarray, lo: float = 0.01, hi: float = 0.99) -> Tuple[float, float]:
    if x.size == 0:
        return 0.0, 1.0
    q = np.quantile(x, [lo, hi])
    a, b = float(q[0]), float(q[1])
    if not (np.isfinite(a) and np.isfinite(b)):
        a, b = float(np.nanmin(x)), float(np.nanmax(x))
    if not np.isfinite(a):
        a = 0.0
    if not np.isfinite(b):
        b = 1.0
    if a == b:
        a -= 1.0
        b += 1.0
    pad = 0.05 * (b - a)
    return a - pad, b + pad


def _write_fit_params(path: str, rows: List[Dict[str, object]]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not rows:
        return
    # Stable column order
    fieldnames = [
        "task",
        "n_points",
        "tau",
        "alpha",
        "objective",
        "y0",
        "L",
        "z_star",
        "b",
        "log_b",
    ]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fieldnames})


def _write_pred_grid(
    out_csv: str,
    task: str,
    logT_grid: np.ndarray,
    logP_grid: np.ndarray,
    yhat: np.ndarray,
    z: np.ndarray,
    r: np.ndarray,
    z_eff: np.ndarray,
) -> None:
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["log10_tokens_T", "log10_params_B", "log10_flops", "ratio_logT_minus_logP", "z_eff", "y_hat", "task"])
        for lt, lp, zz, rr, ze, yy in zip(
            logT_grid.ravel(),
            logP_grid.ravel(),
            z.ravel(),
            r.ravel(),
            z_eff.ravel(),
            yhat.ravel(),
        ):
            w.writerow([float(lt), float(lp), float(zz), float(rr), float(ze), float(yy), task])


def _plot_surface(
    out_base: str,
    task: str,
    data: TaskData,
    logT_lin: np.ndarray,
    logP_lin: np.ndarray,
    yhat_grid: np.ndarray,
    alpha: float,
    tau: float,
    use_tex: bool,
    scatter_style: str,
) -> None:
    try:
        import matplotlib.pyplot as plt  # type: ignore
        import matplotlib as mpl  # type: ignore
    except Exception as e:  # pragma: no cover
        raise SystemExit("matplotlib is required for plotting") from e

    _configure_matplotlib(mpl, use_tex=use_tex)

    os.makedirs(os.path.dirname(out_base), exist_ok=True)

    fig, ax = plt.subplots(figsize=sigmoid2d_cfg.SURFACE_FIGSIZE)
    norm = mpl.colors.Normalize(vmin=0.0, vmax=1.0)
    levels = np.linspace(0.0, 1.0, sigmoid2d_cfg.SURFACE_LEVELS)
    cf = ax.contourf(
        logP_lin,
        logT_lin,
        yhat_grid,
        levels=levels,
        cmap="viridis",
        norm=norm,
        alpha=sigmoid2d_cfg.SURFACE_CONTOUR_ALPHA,
    )

    # Color scatter points by base-model family (contour already encodes accuracy).
    if str(scatter_style) == "family" and data.base_models is not None and data.base_models.size == data.logP.size:
        fams = np.asarray([family_from_base_model(b) for b in data.base_models], dtype=object)
        for fam in FAMILY_ORDER:
            m = fams == fam
            if not np.any(m):
                continue
            ax.scatter(
                data.logP[m],
                data.logT[m],
                s=sigmoid2d_cfg.SURFACE_SCATTER_SIZE,
                alpha=sigmoid2d_cfg.SURFACE_SCATTER_ALPHA,
                linewidths=sigmoid2d_cfg.SURFACE_SCATTER_LINEWIDTHS,
                color=color_for_family(str(fam)),
            )
    elif str(scatter_style) != "family":
        ax.scatter(
            data.logP,
            data.logT,
            s=sigmoid2d_cfg.SURFACE_SCATTER_SIZE,
            alpha=sigmoid2d_cfg.SURFACE_SCATTER_ALPHA,
            linewidths=sigmoid2d_cfg.SURFACE_SCATTER_LINEWIDTHS,
            color="#7f7f7f",
        )
    else:
        ax.scatter(
            data.logP,
            data.logT,
            c=data.y,
            cmap="viridis",
            norm=norm,
            s=sigmoid2d_cfg.SURFACE_SCATTER_SIZE,
            alpha=sigmoid2d_cfg.SURFACE_SCATTER_ALPHA,
            linewidths=sigmoid2d_cfg.SURFACE_SCATTER_LINEWIDTHS,
        )

    # Overlay iso-compute (logT + logP) and iso-ratio (logT - logP) lines for context.
    s = data.logT + data.logP
    rr = data.logT - data.logP
    s_levels = np.quantile(s, [0.10, 0.50, 0.90]) if s.size else np.array([])
    r_levels = np.quantile(rr, [0.10, 0.50, 0.90]) if rr.size else np.array([])
    xline = np.linspace(float(logP_lin.min()), float(logP_lin.max()), 200)
    for s0 in s_levels:
        yline = float(s0) - xline
        ax.plot(
            xline,
            yline,
            color=sigmoid2d_cfg.SURFACE_ISO_LINE_COLOR,
            linestyle=sigmoid2d_cfg.SURFACE_ISO_COMPUTE_LINESTYLE,
            linewidth=sigmoid2d_cfg.SURFACE_ISO_LINEWIDTH,
            alpha=sigmoid2d_cfg.SURFACE_ISO_LINE_ALPHA,
        )
    for r0 in r_levels:
        yline = float(r0) + xline
        ax.plot(
            xline,
            yline,
            color=sigmoid2d_cfg.SURFACE_ISO_LINE_COLOR,
            linestyle=sigmoid2d_cfg.SURFACE_ISO_RATIO_LINESTYLE,
            linewidth=sigmoid2d_cfg.SURFACE_ISO_LINEWIDTH,
            alpha=sigmoid2d_cfg.SURFACE_ISO_LINE_ALPHA,
        )

    cbar = fig.colorbar(cf, ax=ax, pad=sigmoid2d_cfg.SURFACE_COLORBAR_PAD)
    cbar.set_label("Accuracy", fontsize=sigmoid2d_cfg.SURFACE_COLORBAR_LABEL_FONTSIZE)

    ax.set_xlabel("log10(#Params (B))", fontsize=sigmoid2d_cfg.SURFACE_XLABEL_FONTSIZE)
    ax.set_ylabel("log10(Pretraining tokens (T))", fontsize=sigmoid2d_cfg.SURFACE_YLABEL_FONTSIZE)
    ax.set_title(
        f"{task} — 2D sigmoid frontier (τ={tau:.2f}, α={alpha:.3f})",
        fontsize=sigmoid2d_cfg.SURFACE_TITLE_FONTSIZE,
    )
    ax.set_xlim(float(logP_lin.min()), float(logP_lin.max()))
    ax.set_ylim(float(logT_lin.min()), float(logT_lin.max()))
    fig.tight_layout()
    fig.savefig(out_base + ".png", dpi=300, bbox_inches="tight")
    fig.savefig(out_base + ".pdf", dpi=300, bbox_inches="tight")
    plt.close(fig)


def _plot_slices(
    out_base: str,
    task: str,
    data: TaskData,
    alpha: float,
    params: np.ndarray,
    tau: float,
    slice_quantiles: Sequence[float],
    slice_bandwidth: float,
    sigmoid_pred,
    use_tex: bool,
    scatter_style: str,
) -> None:
    try:
        import matplotlib.pyplot as plt  # type: ignore
        import matplotlib as mpl  # type: ignore
    except Exception as e:  # pragma: no cover
        raise SystemExit("matplotlib is required for plotting") from e

    _configure_matplotlib(mpl, use_tex=use_tex)

    os.makedirs(os.path.dirname(out_base), exist_ok=True)
    fig, ax = plt.subplots(figsize=sigmoid2d_cfg.SLICES_FIGSIZE)

    r_min, r_max = _quantile_range(data.r, 0.01, 0.99)
    r_grid = np.linspace(float(r_min), float(r_max), 260)
    z_levels = np.quantile(data.z, slice_quantiles).tolist() if data.z.size else []
    yhat_all: List[np.ndarray] = []

    # Use a stable palette consistent across lines and points.
    colors = (
        mpl.rcParams.get("axes.prop_cycle", None).by_key().get("color", [])  # type: ignore[union-attr]
        if mpl.rcParams.get("axes.prop_cycle", None) is not None
        else []
    )
    if not colors:
        colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]

    # Ensure consistent ordering by compute level (handles unsorted quantiles).
    pairs = list(zip(list(slice_quantiles), list(z_levels)))
    pairs.sort(key=lambda t: float(t[1]))
    slice_quantiles_sorted = [float(q) for (q, _) in pairs]
    z_levels_sorted = [float(z0) for (_, z0) in pairs]

    # Color points by the nearest slice (partitioned by midpoints between z-levels).
    if len(z_levels_sorted) == 1:
        group_idx = np.zeros_like(data.z, dtype=int)
    else:
        bounds = 0.5 * (np.asarray(z_levels_sorted[:-1]) + np.asarray(z_levels_sorted[1:]))
        group_idx = np.searchsorted(bounds, data.z, side="right")
    markers = sigmoid2d_cfg.SLICES_MARKERS
    for i in range(len(z_levels_sorted)):
        m = group_idx == i
        if not np.any(m):
            continue
        if str(scatter_style) == "family" and data.base_models is not None and data.base_models.size == data.r.size:
            fams = np.asarray([family_from_base_model(b) for b in data.base_models[m]], dtype=object)
            for fam in FAMILY_ORDER:
                mf = fams == fam
                if not np.any(mf):
                    continue
                ax.scatter(
                    data.r[m][mf],
                    data.y[m][mf],
                    s=sigmoid2d_cfg.SLICES_SCATTER_SIZE,
                    alpha=sigmoid2d_cfg.SLICES_SCATTER_ALPHA,
                    color=color_for_family(str(fam)),
                    marker=markers[i % len(markers)],
                    linewidths=sigmoid2d_cfg.SLICES_SCATTER_LINEWIDTHS,
                    rasterized=True,
                    label="_nolegend_",
                    zorder=1,
                )
        else:
            ax.scatter(
                data.r[m],
                data.y[m],
                s=sigmoid2d_cfg.SLICES_SCATTER_SIZE,
                alpha=sigmoid2d_cfg.SLICES_SCATTER_ALPHA,
                color=colors[i % len(colors)],
                linewidths=sigmoid2d_cfg.SLICES_SCATTER_LINEWIDTHS,
                rasterized=True,
                label="_nolegend_",
                zorder=1,
            )

    for i, z0 in enumerate(z_levels_sorted):
        yhat = sigmoid_pred(params, z0 + float(alpha) * r_grid)
        yhat_all.append(np.asarray(yhat, float))
        q = float(slice_quantiles_sorted[i])
        pct = int(round(100 * q))
        if abs(q - 0.10) < 1e-9:
            prefix = "Low compute"
        elif abs(q - 0.50) < 1e-9:
            prefix = "Median compute"
        elif abs(q - 0.90) < 1e-9:
            prefix = "High compute"
        else:
            prefix = "Compute"
        label = f"{prefix} ({pct}th pct; $\\log_{{10}}(\\mathrm{{FLOPs}})$={float(z0):.2f})"
        ax.plot(
            r_grid,
            yhat,
            linewidth=sigmoid2d_cfg.SLICES_CURVE_LINEWIDTH,
            alpha=sigmoid2d_cfg.SLICES_CURVE_ALPHA,
            label=label,
            color=colors[i % len(colors)],
            zorder=2,
        )

    # Match styling of the 1D sigmoid no-split plots (except for colors).
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

    # Dynamic y-limits from points and curves with small padding (same rule as 1D plots).
    y_min = float(np.nanmin(data.y)) if data.y.size else float("nan")
    y_max = float(np.nanmax(data.y)) if data.y.size else float("nan")
    if yhat_all:
        yh = np.concatenate(yhat_all) if len(yhat_all) > 1 else yhat_all[0]
        if yh.size:
            y_min = float(np.nanmin([y_min, float(np.nanmin(yh))]))
            y_max = float(np.nanmax([y_max, float(np.nanmax(yh))]))
    if not np.isfinite(y_min) or not np.isfinite(y_max) or y_min == y_max:
        y_min, y_max = 0.0, 1.0
    pad = 0.02 * max(1e-6, (y_max - y_min))
    ax.set_ylim(y_min - pad, y_max + pad)

    ax.set_xlim(float(r_min), float(r_max))
    ax.set_xlabel(r"$\log_{10}(T) - \log_{10}(P)$", fontweight="bold", fontsize=sigmoid2d_cfg.SLICES_XLABEL_FONTSIZE)
    ax.set_ylabel("Accuracy", fontweight="bold", fontsize=sigmoid2d_cfg.SLICES_YLABEL_FONTSIZE)
    ax.set_title(
        f"{task} ($\\tau={tau:.2f},\\ \\alpha={alpha:.3f}$)",
        fontweight="bold",
        fontsize=sigmoid2d_cfg.SLICES_TITLE_FONTSIZE,
    )
    leg = ax.legend(
        loc=sigmoid2d_cfg.SLICES_LEGEND_LOC,
        fontsize=sigmoid2d_cfg.SLICES_LEGEND_FONTSIZE,
        frameon=True,
        framealpha=sigmoid2d_cfg.SLICES_LEGEND_FRAMEALPHA,
        title=r"Compute level (percentile of $\log_{10}(\mathrm{FLOPs})$)",
    )
    if leg and leg.get_title():
        leg.get_title().set_fontweight("bold")
    fig.tight_layout()
    fig.savefig(out_base + ".png", dpi=300, bbox_inches="tight")
    fig.savefig(out_base + ".pdf", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main(argv: Optional[Sequence[str]] = None) -> None:
    p = argparse.ArgumentParser(description="Fit and visualize a 2D sigmoid frontier (compute + ratio) on OLL CSVs.")
    p.add_argument("--csv", required=True, help="Input CSV path")
    p.add_argument(
        "--compute_product_cols",
        nargs=2,
        default=("Pretraining tokens (T)", "#Params (B)"),
        metavar=("TOKENS_COL", "PARAMS_COL"),
        help="Tokens and params columns (used for compute and ratio)",
    )
    p.add_argument("--compute_multiplier", type=float, default=6.0, help="Compute multiplier (default 6.0)")
    p.add_argument("--tasks", nargs="*", default=None, help="Task columns (default: auto-detect OLL Raw tasks)")
    p.add_argument("--tau", type=float, default=0.98, help="Upper-quantile τ for frontier fit")
    p.add_argument(
        "--lambda_b",
        type=float,
        default=1e-2,
        help="L2 penalty on sigmoid slope b in the frontier fitter (default: 1e-2)",
    )
    p.add_argument("--alpha_min", type=float, default=-1.0)
    p.add_argument("--alpha_max", type=float, default=1.0)
    p.add_argument("--alpha_steps", type=int, default=21, help="Grid size for alpha search (odd recommended)")
    p.add_argument("--grid_n", type=int, default=120, help="Grid resolution per axis for 2D surface")
    p.add_argument("--slice_bandwidth", type=float, default=0.0, help="|log10(C)-z0| bandwidth for slice scatter overlay (0 disables)")
    p.add_argument(
        "--slice_quantiles",
        type=float,
        nargs="*",
        default=[0.10, 0.50, 0.90],
        help="Compute quantiles for slice curves",
    )
    p.add_argument("--usetex", action="store_true", help="Render text with LaTeX (requires a LaTeX installation)")
    p.add_argument("--out_dir", default=os.path.join("outputs", "sigmoid2d", "no_split"))
    p.add_argument("--seed", type=int, default=0, help="Random seed for optimizer")
    p.add_argument(
        "--scatter_style",
        choices=["two_color", "family"],
        default="two_color",
        help="Scatter styling: 'two_color' (default) disables model-family coloring; 'family' colors points by base-model family.",
    )
    args = p.parse_args(argv)

    rows, headers = read_csv_rows(args.csv)
    if not rows:
        raise SystemExit(f"No rows in {args.csv}")
    tasks = args.tasks or detect_oll_raw_tasks(headers)
    if not tasks:
        raise SystemExit("Could not infer tasks; pass --tasks explicitly")

    tokens_col, params_col = args.compute_product_cols
    fit_sigmoid_enhanced, sigmoid_pred = _import_sigmoid_optimizer()

    os.makedirs(args.out_dir, exist_ok=True)
    plots_dir = os.path.join(args.out_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    alpha_grid = np.linspace(float(args.alpha_min), float(args.alpha_max), int(args.alpha_steps))
    fit_rows: List[Dict[str, object]] = []

    for task in tasks:
        data = _load_task_data(
            rows,
            task=task,
            tokens_col=tokens_col,
            params_col=params_col,
            compute_multiplier=float(args.compute_multiplier),
        )
        if data.y.size < 3:
            print(f"[sigmoid2d] skip {task}: only {int(data.y.size)} points")
            continue

        best = None
        best_alpha = float("nan")
        best_obj = float("inf")

        for alpha in alpha_grid:
            z_eff = data.z + float(alpha) * data.r
            res = fit_sigmoid_enhanced(
                z_eff,
                data.y,
                tau=float(args.tau),
                kappa_final=50.0,
                lambda_b=float(args.lambda_b),
                n_zstar_grid=10,
                n_b_grid=10,
                n_random=100,
                seed=int(args.seed),
            )
            if not res.success:
                continue
            if float(res.objective) < best_obj:
                best_obj = float(res.objective)
                best_alpha = float(alpha)
                best = res

        if best is None:
            print(f"[sigmoid2d] skip {task}: all alpha fits failed")
            continue

        params = np.asarray(best.params, float)
        y0, L, z_star, log_b = [float(x) for x in params]
        b = float(np.exp(log_b))
        fit_rows.append(
            {
                "task": task,
                "n_points": int(data.y.size),
                "tau": float(args.tau),
                "alpha": float(best_alpha),
                "objective": float(best_obj),
                "y0": y0,
                "L": L,
                "z_star": z_star,
                "b": b,
                "log_b": log_b,
            }
        )

        # Build prediction grid in (logT, logP) space
        logT_min, logT_max = _quantile_range(data.logT, 0.01, 0.99)
        logP_min, logP_max = _quantile_range(data.logP, 0.01, 0.99)
        logT_lin = np.linspace(logT_min, logT_max, int(args.grid_n))
        logP_lin = np.linspace(logP_min, logP_max, int(args.grid_n))
        LP, LT = np.meshgrid(logP_lin, logT_lin)
        z_grid = np.log10(max(float(args.compute_multiplier), 1e-300)) + LT + LP
        r_grid = LT - LP
        z_eff_grid = z_grid + float(best_alpha) * r_grid
        yhat_grid = sigmoid_pred(params, z_eff_grid.ravel()).reshape(LT.shape)
        yhat_grid = np.clip(yhat_grid, 0.0, 1.0)

        # Save prediction grid (CSV)
        task_clean = sanitize_task_name(task)
        grid_csv = os.path.join(args.out_dir, f"pred_grid__{task_clean}.csv")
        _write_pred_grid(
            grid_csv,
            task=task,
            logT_grid=LT,
            logP_grid=LP,
            yhat=yhat_grid,
            z=z_grid,
            r=r_grid,
            z_eff=z_eff_grid,
        )

        # Plots
        surface_base = os.path.join(plots_dir, f"surface__{task_clean}")
        _plot_surface(
            surface_base,
            task=task,
            data=data,
            logT_lin=logT_lin,
            logP_lin=logP_lin,
            yhat_grid=yhat_grid,
            alpha=float(best_alpha),
            tau=float(args.tau),
            use_tex=bool(args.usetex),
            scatter_style=str(args.scatter_style),
        )

        slices_base = os.path.join(plots_dir, f"slices__{task_clean}")
        _plot_slices(
            slices_base,
            task=task,
            data=data,
            alpha=float(best_alpha),
            params=params,
            tau=float(args.tau),
            slice_quantiles=list(args.slice_quantiles),
            slice_bandwidth=float(args.slice_bandwidth),
            sigmoid_pred=sigmoid_pred,
            use_tex=bool(args.usetex),
            scatter_style=str(args.scatter_style),
        )

        print(f"[sigmoid2d] {task}: alpha={best_alpha:.3f} obj={best_obj:.6g} n={int(data.y.size)}")

    _write_fit_params(os.path.join(args.out_dir, "fit_params.csv"), fit_rows)


if __name__ == "__main__":
    main()

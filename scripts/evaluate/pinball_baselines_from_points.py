#!/usr/bin/env python3
"""
Compare sigmoid vs I-spline frontiers on existing "points/*.csv" datasets.

Motivation: a good-looking sigmoid fit alone does not justify that the sigmoid is a
good functional form for a frontier boundary. This script mirrors the spirit of
`scripts/evaluate/sigmoid_pinball_baselines.py` (pinball + coverage error) but for
no-split point clouds already saved under output folders like:
  <root>/sigmoid/no_split/points/*.csv

It fits the frontier on a random train split and evaluates on a held-out split:
  - Sigmoid frontier (repo fitter)
  - Monotone, saturating I-spline frontier (scripts/run/ispline_pinball_optimizer.py)
  - Null (best constant)
  - Oracle binwise (per-bin best constant)

By default, it writes results next to the sigmoid outputs:
  <root>/pinball_baselines/no_split/{results.csv,results.json,tasks/,plots/}

This script does NOT run any new benchmark evaluation; it only uses existing points.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from skill_frontier.evaluation.binning import create_equal_mass_bins  # type: ignore
from skill_frontier.evaluation.common import fit_sigmoid_predictor, interpolate_curve  # type: ignore
from skill_frontier.io.output_paths import sanitize_task_name  # type: ignore
from skill_frontier.plotting.axis_formatting import apply_pretraining_compute_tick_multiplier  # type: ignore
from skill_frontier.plotting.configs import frontier_1d as frontier_1d_cfg  # type: ignore
from skill_frontier.plotting.configs import mpl_rc as mpl_rc_cfg  # type: ignore
from skill_frontier.plotting.labels import PRETRAINING_COMPUTE_FLOPS_LABEL  # type: ignore
from scripts.run import ispline_pinball_optimizer as iso  # type: ignore


X_TICK_MULTIPLIER_DEFAULT: float = 1e21
TARGET_FIGSIZE: tuple[float, float] = (0.75 * (2069.0 / 300.0), 1344.0 / 300.0)


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Compare sigmoid vs I-spline frontiers on existing points CSVs."
    )
    p.add_argument(
        "--points_dir",
        action="append",
        default=[],
        help="Directory containing points CSVs (model,training_compute_flop,log10_compute,score). Can be passed multiple times.",
    )
    p.add_argument(
        "--roots",
        nargs="*",
        default=[],
        help="One or more roots to recursively search for sigmoid/no_split/points directories.",
    )
    p.add_argument(
        "--out_dir",
        default=None,
        help="Optional output directory (only valid when exactly one --points_dir is provided). If omitted, uses <root>/pinball_baselines/no_split.",
    )
    p.add_argument("--tau", type=float, default=0.98)
    p.add_argument("--kappa_final", type=float, default=50.0)
    p.add_argument("--lambda_b", type=float, default=1e-4, help="Sigmoid L2 penalty on slope b.")
    p.add_argument(
        "--frontier_fit_mode",
        choices=["quantile_per_point", "robust_bin_frontier"],
        default="quantile_per_point",
    )
    p.add_argument("--bins", type=int, default=10, help="Equal-mass bins for oracle/coverage.")
    p.add_argument("--min_bin_size", type=int, default=5, help="Min bin size after merging.")
    p.add_argument("--train_frac", type=float, default=0.7, help="Fraction of points used for training.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--x_tick_multiplier", type=float, default=X_TICK_MULTIPLIER_DEFAULT)
    p.add_argument(
        "--ispline_lambda_beta",
        type=float,
        default=1e-4,
        help="I-spline L2 penalty on beta (controls smoothness).",
    )
    p.add_argument(
        "--ispline_knots",
        type=int,
        default=3,
        help="I-spline equal-mass knot bins (n_knot_grid). Match evaluation_pinball_baselines default (=3).",
    )
    p.add_argument("--ispline_random", type=int, default=20, help="I-spline random restarts.")
    p.add_argument("--ispline_maxiter", type=int, default=700)
    p.add_argument("--no_plots", action="store_true", help="Skip writing overlay frontier plots.")
    return p


def _pinball_loss(y: np.ndarray, yhat: np.ndarray, *, tau: float, kappa: float) -> float:
    y = np.asarray(y, float)
    yhat = np.asarray(yhat, float)
    mask = np.isfinite(y) & np.isfinite(yhat)
    if not np.any(mask):
        return float("nan")
    r = y[mask] - yhat[mask]
    vals = np.logaddexp(0.0, kappa * r) / kappa + (tau - 1.0) * r
    return float(np.mean(vals))


def _best_constant_pinball(y: np.ndarray, *, tau: float, kappa: float) -> float:
    y = np.asarray(y, float)
    y = y[np.isfinite(y)]
    if y.size == 0:
        return float("nan")
    q0 = float(np.clip(np.quantile(y, tau), 0.0, 1.0))

    def _obj(c: float) -> float:
        return _pinball_loss(y, np.full_like(y, c), tau=float(tau), kappa=float(kappa))

    try:
        res = minimize_scalar(_obj, bounds=(0.0, 1.0), method="bounded", options={"xatol": 1e-6})
        if res.success and np.isfinite(res.x):
            return float(res.x)
    except Exception:
        pass
    return q0


def _macro_coverage_error(
    z: np.ndarray,
    y: np.ndarray,
    yhat: np.ndarray,
    edges: np.ndarray,
    *,
    tau: float,
) -> float:
    z = np.asarray(z, float)
    y = np.asarray(y, float)
    yhat = np.asarray(yhat, float)
    edges = np.asarray(edges, float)
    if edges.size < 2:
        return float("nan")
    errs: List[float] = []
    for i in range(len(edges) - 1):
        lo, hi = float(edges[i]), float(edges[i + 1])
        if i < len(edges) - 2:
            mask = (z >= lo) & (z < hi)
        else:
            mask = (z >= lo) & (z <= hi)
        mask = mask & np.isfinite(y) & np.isfinite(yhat)
        n = int(np.sum(mask))
        if n == 0:
            continue
        cov = float(np.mean(y[mask] <= yhat[mask]))
        errs.append(abs(cov - float(tau)))
    if not errs:
        return float("nan")
    return float(np.mean(np.asarray(errs, float)))


def _compute_bin_pinball(
    z: np.ndarray,
    y: np.ndarray,
    yhat: np.ndarray,
    edges: np.ndarray,
    *,
    tau: float,
    kappa: float,
) -> List[Tuple[int, float, float, int, float]]:
    z = np.asarray(z, float)
    y = np.asarray(y, float)
    yhat = np.asarray(yhat, float)
    edges = np.asarray(edges, float)
    out: List[Tuple[int, float, float, int, float]] = []
    for i in range(len(edges) - 1):
        lo, hi = float(edges[i]), float(edges[i + 1])
        if i < len(edges) - 2:
            mask = (z >= lo) & (z < hi)
        else:
            mask = (z >= lo) & (z <= hi)
        mask = mask & np.isfinite(y) & np.isfinite(yhat)
        n = int(np.sum(mask))
        loss = float("nan") if n == 0 else _pinball_loss(y[mask], yhat[mask], tau=float(tau), kappa=float(kappa))
        out.append((i, lo, hi, n, loss))
    return out


def _apply_frontier_style(
    ax, *, xlabel: str, ylabel: str, title: str, x_tick_multiplier: float
) -> None:
    ax.set_xscale("log")
    ax.set_xlabel(xlabel, fontweight="bold", fontsize=frontier_1d_cfg.LABEL_FONTSIZE_X)
    if xlabel == PRETRAINING_COMPUTE_FLOPS_LABEL:
        apply_pretraining_compute_tick_multiplier(ax, multiplier=float(x_tick_multiplier))
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


def _iter_points_dirs_from_roots(roots: Sequence[str]) -> Iterable[Path]:
    for root in roots:
        base = Path(root)
        if not base.exists():
            continue
        for p in base.rglob("sigmoid/no_split/points"):
            if p.is_dir():
                yield p


def _default_out_dir_for_points_dir(points_dir: Path) -> Path:
    # points_dir = <root>/sigmoid/no_split/points
    # root = points_dir.parent.parent.parent
    root = points_dir.parent.parent.parent
    return root / "pinball_baselines" / "no_split"


def _load_points_csv(path: Path) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    df = pd.read_csv(path)
    if "training_compute_flop" not in df.columns or "score" not in df.columns:
        raise ValueError(f"Missing required columns in {path}: expected training_compute_flop, score")
    c = pd.to_numeric(df["training_compute_flop"], errors="coerce").to_numpy(float)
    y = pd.to_numeric(df["score"], errors="coerce").to_numpy(float)
    z = None
    if "log10_compute" in df.columns:
        z = pd.to_numeric(df["log10_compute"], errors="coerce").to_numpy(float)
    mask = np.isfinite(c) & (c > 0.0) & np.isfinite(y)
    c = c[mask]
    y = y[mask]
    if z is not None:
        z = z[mask]
    return c, y, z


def _load_sigmoid_curve_csv(path: Path, *, x_tick_multiplier: float) -> Tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(path)
    if "training_compute_flop" not in df.columns or "y_hat" not in df.columns:
        raise ValueError(f"Missing required columns in {path}: expected training_compute_flop, y_hat")
    x_abs = pd.to_numeric(df["training_compute_flop"], errors="coerce").to_numpy(float)
    y_hat = pd.to_numeric(df["y_hat"], errors="coerce").to_numpy(float)
    mask = np.isfinite(x_abs) & (x_abs > 0.0) & np.isfinite(y_hat)
    x_abs = x_abs[mask]
    y_hat = y_hat[mask]
    x_scaled = x_abs / float(x_tick_multiplier)
    return x_scaled.astype(float), y_hat.astype(float)


def _safe_r2(L_model: float, L_null: float) -> float:
    if not (np.isfinite(L_model) and np.isfinite(L_null)):
        return float("nan")
    if L_null == 0.0:
        return float("nan")
    return float(1.0 - (L_model / L_null))


def _write_task_metrics_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def _run_one_points_dir(args: argparse.Namespace, points_dir: Path, out_dir: Path) -> Dict[str, Any]:
    csv_paths = sorted([p for p in points_dir.glob("*.csv") if p.is_file()])
    if not csv_paths:
        raise ValueError(f"No CSV files found under: {points_dir}")

    curves_dir = points_dir.parent / "curves"

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "tasks").mkdir(parents=True, exist_ok=True)
    (out_dir / "plots").mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(int(args.seed))

    results_rows: List[Dict[str, Any]] = []
    results_json: Dict[str, Any] = {
        "points_dir": str(points_dir),
        "out_dir": str(out_dir),
        "config": {
            "tau": float(args.tau),
            "kappa_final": float(args.kappa_final),
            "lambda_b": float(args.lambda_b),
            "frontier_fit_mode": str(args.frontier_fit_mode),
            "bins": int(args.bins),
            "min_bin_size": int(args.min_bin_size),
            "train_frac": float(args.train_frac),
            "seed": int(args.seed),
            "x_tick_multiplier": float(args.x_tick_multiplier),
            "ispline_lambda_beta": float(args.ispline_lambda_beta),
            "ispline_knots": int(args.ispline_knots),
            "ispline_random": int(args.ispline_random),
            "ispline_maxiter": int(args.ispline_maxiter),
        },
        "tasks": {},
    }

    for task_idx, csv_path in enumerate(csv_paths):
        task = csv_path.stem
        safe_task = sanitize_task_name(task)

        c_abs, y_all, _z_raw = _load_points_csv(csv_path)
        n_total = int(y_all.size)
        if n_total < 12:
            print(f"[points_baselines] skip {task}: only n={n_total} points")
            continue

        x_mult = float(args.x_tick_multiplier)
        c = c_abs / x_mult
        z = np.log10(c)
        mask = np.isfinite(z) & np.isfinite(y_all)
        c = c[mask]
        z = z[mask]
        y = y_all[mask]
        n_total = int(y.size)
        if n_total < 12:
            print(f"[points_baselines] skip {task}: only n={n_total} valid points after filtering")
            continue

        perm = rng.permutation(n_total)
        n_train = int(round(float(args.train_frac) * n_total))
        n_train = int(np.clip(n_train, 10, n_total - 2))
        tr_idx = perm[:n_train]
        va_idx = perm[n_train:]

        c_tr, z_tr, y_tr = c[tr_idx], z[tr_idx], y[tr_idx]
        c_va, z_va, y_va = c[va_idx], z[va_idx], y[va_idx]

        z_lo, z_hi = float(np.min(z_tr)), float(np.max(z_tr))
        overlap = (z_va >= z_lo) & (z_va <= z_hi)
        c_va_o, z_va_o, y_va_o = c_va[overlap], z_va[overlap], y_va[overlap]
        if y_va_o.size < 2:
            # If overlap is too small, fall back to evaluating on all held-out points.
            c_va_o, z_va_o, y_va_o = c_va, z_va, y_va

        # Bins for oracle + coverage (train z only)
        edges = create_equal_mass_bins(z_tr, int(max(1, args.bins)), int(max(1, args.min_bin_size)))
        if edges.size < 2:
            edges = np.array([float(np.min(z_tr)), float(np.max(z_tr))], dtype=float)

        # Sigmoid frontier fit on train
        xs_curve, y_curve = fit_sigmoid_predictor(
            c_tr,
            y_tr,
            tau=float(args.tau),
            frontier_fit_mode=str(args.frontier_fit_mode),
            bins_for_fit=int(args.bins),
            min_bin_size_for_fit=int(args.min_bin_size),
            kappa_final=float(args.kappa_final),
            lambda_b=float(args.lambda_b),
        )
        yhat_tr_sig = interpolate_curve(xs_curve, y_curve, c_tr)
        yhat_va_sig = interpolate_curve(xs_curve, y_curve, c_va_o)

        # I-spline frontier fit on train (monotone, saturating)
        fit_is = iso.fit_ispline_enhanced(
            z=z_tr,
            y=y_tr,
            tau=float(args.tau),
            kappa_final=float(args.kappa_final),
            lambda_beta=float(args.ispline_lambda_beta),
            n_knot_grid=int(args.ispline_knots),
            n_random=int(args.ispline_random),
            seed=int(args.seed),
            maxiter=int(args.ispline_maxiter),
        )
        yhat_tr_is = (
            iso._predict_from_params(z_tr, fit_is.params, fit_is.edges)
            if fit_is.params.size and fit_is.edges.size
            else np.full_like(y_tr, np.nan, dtype=float)
        )
        yhat_va_is = (
            iso._predict_from_params(z_va_o, fit_is.params, fit_is.edges)
            if fit_is.params.size and fit_is.edges.size
            else np.full_like(y_va_o, np.nan, dtype=float)
        )

        # Null constant frontier
        q_null = _best_constant_pinball(y_tr, tau=float(args.tau), kappa=float(args.kappa_final))
        yhat_tr_null = np.full_like(y_tr, q_null, dtype=float)
        yhat_va_null = np.full_like(y_va_o, q_null, dtype=float)

        # Oracle per-bin constants
        B = int(max(1, edges.size - 1))
        q_bins = np.full((B,), float("nan"), dtype=float)
        for i in range(B):
            lo, hi = float(edges[i]), float(edges[i + 1])
            in_bin = (z_tr >= lo) & (z_tr < hi) if i < B - 1 else (z_tr >= lo) & (z_tr <= hi)
            y_bin = y_tr[in_bin]
            if y_bin.size:
                q_bins[i] = _best_constant_pinball(y_bin, tau=float(args.tau), kappa=float(args.kappa_final))

        def _oracle_predict(zv: np.ndarray) -> np.ndarray:
            idx = np.searchsorted(edges, zv, side="right") - 1
            idx = np.clip(idx, 0, B - 1)
            return q_bins[idx]

        yhat_tr_orc = _oracle_predict(z_tr)
        yhat_va_orc = _oracle_predict(z_va_o)

        # Metrics
        L_tr_sig = _pinball_loss(y_tr, yhat_tr_sig, tau=float(args.tau), kappa=float(args.kappa_final))
        L_tr_is = _pinball_loss(y_tr, yhat_tr_is, tau=float(args.tau), kappa=float(args.kappa_final))
        L_tr_null = _pinball_loss(y_tr, yhat_tr_null, tau=float(args.tau), kappa=float(args.kappa_final))
        L_tr_orc = _pinball_loss(y_tr, yhat_tr_orc, tau=float(args.tau), kappa=float(args.kappa_final))

        L_va_sig = _pinball_loss(y_va_o, yhat_va_sig, tau=float(args.tau), kappa=float(args.kappa_final))
        L_va_is = _pinball_loss(y_va_o, yhat_va_is, tau=float(args.tau), kappa=float(args.kappa_final))
        L_va_null = _pinball_loss(y_va_o, yhat_va_null, tau=float(args.tau), kappa=float(args.kappa_final))
        L_va_orc = _pinball_loss(y_va_o, yhat_va_orc, tau=float(args.tau), kappa=float(args.kappa_final))

        ce_tr_sig = _macro_coverage_error(z_tr, y_tr, yhat_tr_sig, edges, tau=float(args.tau))
        ce_tr_is = _macro_coverage_error(z_tr, y_tr, yhat_tr_is, edges, tau=float(args.tau))
        ce_tr_null = _macro_coverage_error(z_tr, y_tr, yhat_tr_null, edges, tau=float(args.tau))
        ce_tr_orc = _macro_coverage_error(z_tr, y_tr, yhat_tr_orc, edges, tau=float(args.tau))

        ce_va_sig = _macro_coverage_error(z_va_o, y_va_o, yhat_va_sig, edges, tau=float(args.tau))
        ce_va_is = _macro_coverage_error(z_va_o, y_va_o, yhat_va_is, edges, tau=float(args.tau))
        ce_va_null = _macro_coverage_error(z_va_o, y_va_o, yhat_va_null, edges, tau=float(args.tau))
        ce_va_orc = _macro_coverage_error(z_va_o, y_va_o, yhat_va_orc, edges, tau=float(args.tau))

        ratio_tr_sig_orc = float(L_tr_sig / L_tr_orc) if np.isfinite(L_tr_sig) and np.isfinite(L_tr_orc) and L_tr_orc != 0 else float("nan")
        ratio_va_sig_orc = float(L_va_sig / L_va_orc) if np.isfinite(L_va_sig) and np.isfinite(L_va_orc) and L_va_orc != 0 else float("nan")
        r2_tr_sig = _safe_r2(L_tr_sig, L_tr_null)
        r2_va_sig = _safe_r2(L_va_sig, L_va_null)

        # Per-task CSV (mirrors sigmoid_pinball_baselines)
        task_csv = out_dir / "tasks" / f"{safe_task}__pinball_baselines.csv"
        _write_task_metrics_csv(
            task_csv,
            [
                {
                    "split": "train",
                    "L_sigmoid": L_tr_sig,
                    "L_ispline": L_tr_is,
                    "L_null": L_tr_null,
                    "L_oracle": L_tr_orc,
                    "CE_sigmoid": ce_tr_sig,
                    "CE_ispline": ce_tr_is,
                    "CE_null": ce_tr_null,
                    "CE_oracle": ce_tr_orc,
                    "ratio_sigmoid_oracle": ratio_tr_sig_orc,
                    "R2_pinball": r2_tr_sig,
                },
                {
                    "split": "val",
                    "L_sigmoid": L_va_sig,
                    "L_ispline": L_va_is,
                    "L_null": L_va_null,
                    "L_oracle": L_va_orc,
                    "CE_sigmoid": ce_va_sig,
                    "CE_ispline": ce_va_is,
                    "CE_null": ce_va_null,
                    "CE_oracle": ce_va_orc,
                    "ratio_sigmoid_oracle": ratio_va_sig_orc,
                    "R2_pinball": r2_va_sig,
                },
            ],
        )

        # Per-bin train pinball (include I-spline for this no-split setting)
        bins_sig = _compute_bin_pinball(z_tr, y_tr, yhat_tr_sig, edges, tau=float(args.tau), kappa=float(args.kappa_final))
        bins_is = _compute_bin_pinball(z_tr, y_tr, yhat_tr_is, edges, tau=float(args.tau), kappa=float(args.kappa_final))
        bins_null = _compute_bin_pinball(z_tr, y_tr, yhat_tr_null, edges, tau=float(args.tau), kappa=float(args.kappa_final))
        bins_orc = _compute_bin_pinball(z_tr, y_tr, yhat_tr_orc, edges, tau=float(args.tau), kappa=float(args.kappa_final))
        bins_csv = out_dir / "tasks" / f"{safe_task}__bins_train_pinball_baselines.csv"
        bins_csv.parent.mkdir(parents=True, exist_ok=True)
        with bins_csv.open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(
                [
                    "bin_id",
                    "z_lo",
                    "z_hi",
                    "n",
                    "loss_sigmoid",
                    "loss_ispline",
                    "loss_null",
                    "loss_oracle",
                ]
            )
            for (i, lo, hi, n, ls), (_, _, _, _, li), (_, _, _, _, ln), (_, _, _, _, lo2) in zip(
                bins_sig, bins_is, bins_null, bins_orc
            ):
                w.writerow([i, lo, hi, n, ls, li, ln, lo2])

        # Summary row (one per task)
        row = {
            "task": task,
            "n_total": n_total,
            "n_train": int(y_tr.size),
            "n_val": int(y_va.size),
            "n_val_eval": int(y_va_o.size),
            "tau": float(args.tau),
            "L_train_sigmoid": L_tr_sig,
            "L_train_ispline": L_tr_is,
            "L_train_null": L_tr_null,
            "L_train_oracle": L_tr_orc,
            "L_val_sigmoid": L_va_sig,
            "L_val_ispline": L_va_is,
            "L_val_null": L_va_null,
            "L_val_oracle": L_va_orc,
            "CE_train_sigmoid": ce_tr_sig,
            "CE_train_ispline": ce_tr_is,
            "CE_train_null": ce_tr_null,
            "CE_train_oracle": ce_tr_orc,
            "CE_val_sigmoid": ce_va_sig,
            "CE_val_ispline": ce_va_is,
            "CE_val_null": ce_va_null,
            "CE_val_oracle": ce_va_orc,
            "ratio_val_sigmoid_oracle": ratio_va_sig_orc,
            "R2_val_sigmoid": r2_va_sig,
            "ispline_success": bool(fit_is.success),
            "ispline_loss_train_reported": float(fit_is.loss) if np.isfinite(fit_is.loss) else float("nan"),
            "z_train_min": z_lo,
            "z_train_max": z_hi,
        }
        results_rows.append(row)

        results_json["tasks"][task] = {
            "metrics": row,
            "sigmoid_curve": {
                "x_train_units": "training_compute_flop / x_tick_multiplier",
                "xs_curve": xs_curve.tolist(),
                "y_curve": y_curve.tolist(),
            },
            "ispline_fit": {
                "success": bool(fit_is.success),
                "message": str(fit_is.message),
                "loss": float(fit_is.loss) if np.isfinite(fit_is.loss) else None,
                "edges": fit_is.edges.tolist() if fit_is.edges.size else [],
                "params": fit_is.params.tolist() if fit_is.params.size else [],
            },
            "oracle": {"edges": edges.tolist(), "q_bins": q_bins.tolist(), "q_null": float(q_null)},
        }

        if not args.no_plots:
            try:
                import matplotlib as mpl  # type: ignore
                import matplotlib.pyplot as plt  # type: ignore
            except Exception:
                plt = None  # type: ignore
                mpl = None  # type: ignore
            if plt is not None:
                try:
                    if mpl is not None:
                        mpl.rcParams["font.family"] = mpl_rc_cfg.FONT_FAMILY
                except Exception:
                    pass

                fig, ax = plt.subplots(figsize=TARGET_FIGSIZE)
                ax.scatter(
                    c,
                    y,
                    s=float(frontier_1d_cfg.SCATTER_SIZE) * 4.5,
                    alpha=0.6,
                    color="black",
                    marker="X",
                    linewidths=frontier_1d_cfg.SCATTER_LINEWIDTHS,
                    rasterized=True,
                )

                # Sigmoid curve: for plot fidelity, prefer the precomputed curve next to the
                # original sigmoid outputs (if available), else fall back to the train-fit curve.
                xs_curve_plot, y_curve_plot = xs_curve, y_curve
                try:
                    curve_path = curves_dir / csv_path.name
                    if curve_path.exists():
                        xs_curve_plot, y_curve_plot = _load_sigmoid_curve_csv(
                            curve_path, x_tick_multiplier=float(args.x_tick_multiplier)
                        )
                except Exception:
                    pass

                ax.plot(
                    xs_curve_plot,
                    y_curve_plot,
                    color="firebrick",
                    linewidth=frontier_1d_cfg.CURVE_LINEWIDTH,
                    alpha=frontier_1d_cfg.CURVE_ALPHA,
                    label=f"Sigmoid (loss = {float(L_va_sig):.4f})",
                )

                # I-spline curve evaluated on a dense z grid
                z_grid = np.linspace(float(np.min(z)), float(np.max(z)), 300)
                c_grid = np.power(10.0, z_grid)
                if fit_is.params.size and fit_is.edges.size:
                    y_grid_is = iso._predict_from_params(z_grid, fit_is.params, fit_is.edges)
                    ax.plot(
                        c_grid,
                        y_grid_is,
                        color="#1f77b4",
                        linewidth=frontier_1d_cfg.CURVE_LINEWIDTH,
                        alpha=frontier_1d_cfg.CURVE_ALPHA,
                        label=f"I-spline (loss = {float(L_va_is):.4f})",
                    )

                _apply_frontier_style(
                    ax,
                    xlabel=PRETRAINING_COMPUTE_FLOPS_LABEL,
                    ylabel="Score",
                    title="",
                    x_tick_multiplier=float(args.x_tick_multiplier),
                )
                y_min = float(np.nanmin([np.nanmin(y), np.nanmin(y_curve)]))
                y_max = float(np.nanmax([np.nanmax(y), np.nanmax(y_curve)]))
                if np.isfinite(y_min) and np.isfinite(y_max) and y_min != y_max:
                    pad = 0.02 * (y_max - y_min)
                    ax.set_ylim(y_min - pad, y_max + pad)
                ax.legend(
                    loc=frontier_1d_cfg.LEGEND_LOC,
                    fontsize=frontier_1d_cfg.LEGEND_FONTSIZE,
                    frameon=True,
                    framealpha=frontier_1d_cfg.LEGEND_FRAMEALPHA,
                )
                fig.tight_layout()
                out_base = out_dir / "plots" / f"{safe_task}__frontier_compare"
                fig.savefig(str(out_base) + ".png", dpi=300)
                fig.savefig(str(out_base) + ".pdf", dpi=300)
                plt.close(fig)

        print(
            f"[points_baselines] {points_dir.parent.parent.parent.name}:{task} n={n_total} "
            f"L_val(sigmoid)={L_va_sig:.4g} L_val(ispline)={L_va_is:.4g}"
        )

    # Write summary CSV/JSON
    results_csv = out_dir / "results.csv"
    if results_rows:
        with results_csv.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(results_rows[0].keys()))
            w.writeheader()
            for r in results_rows:
                w.writerow(r)

    results_json_path = out_dir / "results.json"
    with results_json_path.open("w") as f:
        json.dump(results_json, f, indent=2, sort_keys=True)

    return {
        "points_dir": str(points_dir),
        "out_dir": str(out_dir),
        "n_tasks": len(results_rows),
        "results_csv": str(results_csv),
        "results_json": str(results_json_path),
    }


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = _build_argparser().parse_args(argv)

    points_dirs: List[Path] = []
    points_dirs.extend([Path(p) for p in args.points_dir])
    points_dirs.extend(list(_iter_points_dirs_from_roots(args.roots)))
    # de-dup while preserving order
    seen = set()
    points_dirs_unique: List[Path] = []
    for p in points_dirs:
        rp = str(p.resolve())
        if rp in seen:
            continue
        seen.add(rp)
        points_dirs_unique.append(p)
    points_dirs = points_dirs_unique

    if not points_dirs:
        raise SystemExit("No points_dir found. Use --points_dir or --roots.")

    if args.out_dir is not None and len(points_dirs) != 1:
        raise SystemExit("--out_dir is only allowed when exactly one --points_dir is provided.")

    summary: List[Dict[str, Any]] = []
    for points_dir in points_dirs:
        if not points_dir.exists():
            print(f"[points_baselines] skip missing points_dir: {points_dir}")
            continue
        out_dir = Path(args.out_dir) if args.out_dir is not None else _default_out_dir_for_points_dir(points_dir)
        res = _run_one_points_dir(args, points_dir=points_dir, out_dir=out_dir)
        summary.append(res)

    print("\n[points_baselines] done")
    for r in summary:
        print(f"  - {r['points_dir']} -> {r['out_dir']} (tasks={r['n_tasks']})")


if __name__ == "__main__":
    main()

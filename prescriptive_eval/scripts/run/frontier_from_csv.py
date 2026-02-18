#!/usr/bin/env python3
"""
Run skill frontier estimation from an arbitrary CSV by specifying column names.

Flexible inputs:
  - CSV must contain model identifiers and either a log-compute column or a raw compute column.
  - Tasks can be specified via prefix or an explicit list of columns.

Examples:
  # Use merged LiveBench (from merge_livebench_to_merged.py):
  python scripts/run/frontier_from_csv.py \
    --csv tables/merged_livebench.csv --model_col model --logC_col logC \
    --task_prefix a_ --output_dir outputs/frontier_livebench_generic \
    --num_C_grid 16 --num_directions 128 --write_vertices --log_level INFO

  # Another CSV with raw compute column and explicit task columns:
  python scripts/run/frontier_from_csv.py \
    --csv tables/mydata.csv --model_col name --compute_col pretrain_compute_zflops \
    --task_cols accuracy_math accuracy_code accuracy_reason --output_dir outputs/frontier_mydata
"""

from __future__ import annotations

import argparse
import csv as _csv
import math
import os
import sys
from typing import List, Optional

import numpy as np

# Robust import: add repo root then use packaged modules
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
from skill_frontier.core.frontier import (
    FrontierConfig,
    SkillFrontier,
    DEAConfig,
    QuantileConfig,
)
from skill_frontier.core.utils import ModelPanel


def _read_csv(
    path: str,
    model_col: str,
    label_col: Optional[str],
    logC_col: Optional[str],
    compute_col: Optional[str],
    compute_product_cols: Optional[List[str]],
    compute_multiplier: float,
    task_prefix: Optional[str],
    task_cols: Optional[List[str]],
    task_contains: Optional[str],
    task_scale: float,
    auto_task_scale: bool,
    min_models_per_task: int,
    min_values_per_row: int,
    impute: str,
    clip01: bool,
    compute_eps: float,
) -> ModelPanel:
    # Read rows as dicts
    with open(path, "r", newline="") as f:
        reader = _csv.DictReader(f)
        rows = list(reader)
    if not rows:
        raise ValueError(f"No rows in {path}")
    cols = list(rows[0].keys())
    if model_col not in cols:
        raise ValueError(f"model_col '{model_col}' not found. Columns: {cols}")
    if label_col is None and 'eval_name' in cols:
        label_col = 'eval_name'
    if label_col is not None and label_col not in cols:
        label_col = None  # silently fall back to model_col
    # Determine compute column usage
    use_logC = None
    compute_mode = None  # 'logC' | 'raw' | 'product'
    if logC_col and logC_col in cols:
        use_logC = logC_col
        compute_mode = 'logC'
    elif compute_col and compute_col in cols:
        compute_mode = 'raw'
    elif compute_product_cols and len(compute_product_cols) == 2 and all(c in cols for c in compute_product_cols):
        compute_mode = 'product'
    else:
        raise ValueError("Provide --logC_col, or --compute_col, or --compute_product_cols (two columns) present in CSV")
    # Determine task columns
    chosen_tasks: List[str]
    if task_cols:
        for c in task_cols:
            if c not in cols:
                raise ValueError(f"Task column '{c}' not found in CSV")
        chosen_tasks = list(task_cols)
    else:
        if task_contains:
            chosen_tasks = [c for c in cols if task_contains in c]
            if not chosen_tasks:
                raise ValueError(f"No task columns contain substring '{task_contains}'")
        elif task_prefix:
            chosen_tasks = [c for c in cols if c.startswith(task_prefix)]
            if not chosen_tasks:
                raise ValueError(f"No task columns start with prefix '{task_prefix}'")
        else:
            # Default: treat all columns except model and compute columns as tasks
            excluded = {model_col}
            if logC_col:
                excluded.add(logC_col)
            if compute_col:
                excluded.add(compute_col)
            if compute_product_cols:
                excluded.update(compute_product_cols)
            chosen_tasks = [c for c in cols if c not in excluded]
            if not chosen_tasks:
                raise ValueError("Could not infer task columns; pass --task_prefix or --task_cols")

    # Parse into arrays
    models_raw: List[str] = []
    logC_raw: List[float] = []
    A_rows: List[List[float]] = []
    for r in rows:
        m_id = str(r.get(model_col, "") or "").strip()
        m_label = str(r.get(label_col, "") or "").strip() if label_col is not None else ""
        m = m_label or m_id
        if not m:
            continue
        # compute logC
        if compute_mode == 'logC':
            try:
                lc = float(r[use_logC])
            except Exception:
                lc = float("nan")
        elif compute_mode == 'raw':
            v = r.get(compute_col, None)
            try:
                c = float(v) if v not in (None, "", "nan", "NaN") else float("nan")
            except Exception:
                c = float("nan")
            # Use compute_eps while still rejecting non-positive compute
            lc = float(math.log(c + compute_eps)) if (np.isfinite(c) and c > 0.0) else float("nan")
        elif compute_mode == 'product':
            v1 = r.get(compute_product_cols[0], None)
            v2 = r.get(compute_product_cols[1], None)
            try:
                x1 = float(v1) if v1 not in (None, "", "nan", "NaN") else float("nan")
            except Exception:
                x1 = float("nan")
            try:
                x2 = float(v2) if v2 not in (None, "", "nan", "NaN") else float("nan")
            except Exception:
                x2 = float("nan")
            c = compute_multiplier * x1 * x2
            lc = float(math.log(c + compute_eps)) if (np.isfinite(c) and c > 0.0) else float("nan")
        else:
            lc = float("nan")
        if not np.isfinite(lc):
            continue
        # task vector
        vec: List[float] = []
        for c in chosen_tasks:
            v = r.get(c, None)
            if v in (None, "", "nan", "NaN"):
                vec.append(float("nan"))
            else:
                try:
                    vec.append(float(v) * task_scale)
                except Exception:
                    vec.append(float("nan"))
        models_raw.append(m)
        logC_raw.append(lc)
        A_rows.append(vec)

    A = np.array(A_rows, dtype=float)
    # Optional per-task auto scaling: if a task appears to be in percent (typical values > 1), scale by 0.01
    if auto_task_scale and A.size > 0:
        scales = np.ones(len(chosen_tasks), dtype=float)
        for j in range(len(chosen_tasks)):
            col = A[:, j]
            finite = col[np.isfinite(col)]
            if finite.size == 0:
                continue
            p95 = float(np.nanpercentile(finite, 95))
            if p95 > 1.0:
                scales[j] = 0.01
        if not np.all(scales == 1.0):
            A = A * scales.reshape(1, -1)
    logC = np.array(logC_raw, dtype=float)

    # Task selection by coverage (optional if threshold>0)
    if min_models_per_task > 0:
        non_missing_counts = np.sum(np.isfinite(A), axis=0)
        keep = non_missing_counts >= min_models_per_task
        if not keep.any():
            raise ValueError("No tasks meet coverage threshold; lower --min_models_per_task.")
        A = A[:, keep]
        chosen_tasks = [t for t, k in zip(chosen_tasks, keep) if k]

    # Row selection by valid values among kept tasks (optional)
    if min_values_per_row > 0:
        valid_counts = np.sum(np.isfinite(A), axis=1)
        row_ok = valid_counts >= min_values_per_row
        A = A[row_ok, :]
        logC = logC[row_ok]
        models_raw = [m for m, ok in zip(models_raw, row_ok) if ok]

    # Imputation
    if impute.lower() == "median":
        for j in range(A.shape[1]):
            col = A[:, j]
            mask = np.isfinite(col)
            med = float(np.nanmedian(col)) if mask.any() else 0.0
            col[~mask] = med
            A[:, j] = col
    elif impute.lower() == "none":
        # Drop any remaining rows with NaN
        ok = np.all(np.isfinite(A), axis=1)
        A = A[ok, :]
        logC = logC[ok]
        models_raw = [m for m, k in zip(models_raw, ok) if k]
    else:
        raise ValueError("--impute must be one of: none, median")

    if clip01:
        A = np.clip(A, 0.0, 1.0)

    panel = ModelPanel(models=models_raw, logC=logC, A=A, tasks=chosen_tasks)
    return panel


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run skill frontiers from a flexible CSV")
    # Input mapping
    p.add_argument("--csv", required=True, help="Input CSV path")
    p.add_argument("--model_col", default="model", help="Name of the model ID column")
    p.add_argument("--label_col", default=None, help="Optional display-name column (e.g., eval_name); falls back to --model_col if missing")
    p.add_argument("--logC_col", default="logC", help="Name of log-compute column (if present)")
    p.add_argument("--compute_col", default=None, help="Name of raw compute column (if no logC). Uses natural log.")
    p.add_argument("--compute_eps", type=float, default=1e-12, help="Epsilon added before log")
    p.add_argument("--compute_product_cols", nargs=2, default=None, help="Two column names to multiply for raw compute")
    p.add_argument("--compute_multiplier", type=float, default=1.0, help="Multiplier applied to compute (e.g., 6 for 6*T*B)")
    p.add_argument("--task_prefix", default="a_", help="Prefix to select task columns (ignored if --task_cols provided)")
    p.add_argument("--task_cols", nargs="*", default=None, help="Explicit list of task columns")
    p.add_argument("--task_contains", default=None, help="Substring to select task columns (ignored if --task_cols provided)")
    p.add_argument("--task_scale", type=float, default=1.0, help="Scale factor applied to all task values (e.g., 0.01 for percentages)")
    p.add_argument("--auto_task_scale", action="store_true", help="Auto-scale tasks that look like percentages (divide by 100 per column)")
    # Optional data curation
    p.add_argument("--min_models_per_task", type=int, default=0, help="Minimum models per task to keep the task (0=disable)")
    p.add_argument("--min_values_per_row", type=int, default=0, help="Minimum valid task values per row to keep the row (0=disable)")
    p.add_argument("--impute", choices=["none", "median"], default="none", help="Imputation strategy for missing task values")
    p.add_argument("--no_clip01", action="store_true", help="Disable clipping task values to [0,1]")
    # Frontier config
    p.add_argument("--output_dir", required=True, help="Base directory to write outputs")
    p.add_argument("--split_output_dir", action="store_true", help="Write CSVs to <output_dir>/<csv_subdir> and plots to <output_dir>/<plots_subdir>")
    p.add_argument("--csv_subdir", type=str, default="csv", help="Subfolder for CSVs when --split_output_dir is set")
    p.add_argument("--plots_subdir", type=str, default="plots", help="Subfolder for plots when --split_output_dir is set")
    p.add_argument("--num_C_grid", type=int, default=24, help="Number of compute grid points")
    p.add_argument("--num_directions", type=int, default=128, help="Number of directions on simplex")
    p.add_argument("--kernel", type=str, default="gaussian", choices=["gaussian", "epanechnikov"], help="Kernel type")
    p.add_argument("--bandwidth", type=float, default=None, help="Kernel bandwidth in logC (None => Silverman)")
    p.add_argument("--alpha_cap_multiple", type=float, default=4.0, help="DEA alpha cap multiple for regularization")
    p.add_argument("--crs", action="store_true", help="Use CRS (drop VRS constraint) for DEA")
    p.add_argument("--write_vertices", action="store_true", help="Write DEA/FDH approximate frontier vertices per grid point")
    p.add_argument("--no_smooth_C", action="store_true", help="Disable monotone smoothing across compute (for diagnostics)")
    p.add_argument("--log_level", type=str, default="INFO", help="Logging level (DEBUG, INFO, WARNING)")
    # Plotting
    p.add_argument("--plot_max_per_task", action="store_true", help="Generate per-task plots of max accuracy vs compute")
    p.add_argument("--auto_plots", action="store_true", help="Always generate per-task plots to the plots subdir (or --plot_dir)")
    p.add_argument(
        "--plot_methods",
        nargs="*",
        default=None,
        help="Methods to plot (subset of: DEA FDH Q0.95 Q0.99). Default: all available",
    )
    p.add_argument("--plot_points", type=int, default=200, help="Number of x points for plot interpolation grid")
    p.add_argument("--plot_dir", type=str, default=None, help="Directory to save plots (defaults to output_dir/figures)")
    p.add_argument("--plot_pairwise_frontier", action="store_true", help="Generate pairwise 2D frontiers at selected computes")
    p.add_argument("--pairwise_points", type=int, default=128, help="Number of 2D directions for frontier reconstruction")
    p.add_argument("--pairwise_C_count", type=int, default=3, help="Number of compute grid points to overlay per figure")
    p.add_argument("--pairwise_dir", type=str, default=None, help="Subdirectory for pairwise plots (defaults to '<base>/frontiers_pairwise')")
    # Lower frontier
    p.add_argument("--compute_lower", action="store_true", help="Also compute lower-frontier floors via sign-flip and write min_per_task.csv")
    # Robust outlier filtering
    p.add_argument("--robust_enable", action="store_true", help="Enable robust outlier filtering per compute window (MCD/MAD)")
    p.add_argument("--robust_support_fraction", type=float, default=0.9, help="MCD support fraction (default 0.9)")
    p.add_argument("--robust_alpha", type=float, default=0.005, help="Tail probability for chi-square cutoff (default 0.005)")
    p.add_argument("--robust_alpha_lower", type=float, default=0.05, help="Tail probability for lower-curve 1D filtering in plots (default 0.05)")
    p.add_argument("--robust_min_task_cov", type=float, default=0.9, help="Minimum per-model task coverage within window (default 0.9)")
    p.add_argument("--robust_scale_floor", type=float, default=0.05, help="MAD scale floor per task (default 0.05)")
    p.add_argument("--robust_action", choices=["drop", "winsorize"], default="drop", help="Outlier handling action (default drop)")
    p.add_argument("--robust_winsor_q_low", type=float, default=0.05, help="Winsorization lower quantile (if action=winsorize)")
    p.add_argument("--robust_k_bandwidth", type=float, default=3.0, help="Window half-width multiplier in bandwidth units (unused with kernels; default 3.0)")
    p.add_argument("--chance_baseline_csv", type=str, default=None, help="Optional CSV with a single line of per-task chance baselines in [0,1]")
    return p


def main(argv: Optional[List[str]] = None) -> None:
    args = build_argparser().parse_args(argv)
    import logging
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO), format="%(levelname)s:%(name)s:%(message)s")

    panel = _read_csv(
        path=args.csv,
        model_col=args.model_col,
        label_col=args.label_col,
        logC_col=args.logC_col,
        compute_col=args.compute_col,
        task_prefix=args.task_prefix,
        task_cols=args.task_cols,
        task_contains=args.task_contains,
        task_scale=args.task_scale,
        auto_task_scale=args.auto_task_scale,
        min_models_per_task=args.min_models_per_task,
        min_values_per_row=args.min_values_per_row,
        impute=args.impute,
        clip01=(not args.no_clip01),
        compute_eps=args.compute_eps,
        compute_product_cols=args.compute_product_cols,
        compute_multiplier=args.compute_multiplier,
    )
    logging.info(
        "Loaded CSV with %d models and %d tasks. Tasks: %s",
        panel.num_models,
        panel.num_tasks,
        ", ".join(panel.tasks),
    )

    # Choose CSV output directory (optionally split)
    base_out = args.output_dir
    csv_out_dir = os.path.join(base_out, args.csv_subdir) if args.split_output_dir else base_out

    cfg = FrontierConfig(
        num_C_grid=args.num_C_grid,
        kernel=args.kernel,
        bandwidth=args.bandwidth,
        num_directions=args.num_directions,
        dea=DEAConfig(var_returns_to_scale=(not args.crs), alpha_cap_multiple=args.alpha_cap_multiple, enforce_monotone_in_C=(not args.no_smooth_C)),
        write_vertices=args.write_vertices,
    )
    # Quantile config mirrors smoothing flag
    cfg.quantile = QuantileConfig(enforce_monotone_in_C=(not args.no_smooth_C))
    # Global smoother toggle
    if args.no_smooth_C:
        cfg.smooth_monotone_in_C = False
    if args.compute_lower:
        cfg.compute_lower = True
    if args.robust_enable:
        cfg.robust_enable = True
        cfg.robust_support_fraction = args.robust_support_fraction
        cfg.robust_alpha = args.robust_alpha
        cfg.robust_alpha_lower = args.robust_alpha_lower
        cfg.robust_min_task_cov = args.robust_min_task_cov
        cfg.robust_scale_floor = args.robust_scale_floor
        cfg.robust_action = args.robust_action
        cfg.robust_winsor_q_low = args.robust_winsor_q_low
        cfg.robust_k_bandwidth = args.robust_k_bandwidth
        if args.chance_baseline_csv is not None:
            try:
                with open(args.chance_baseline_csv, "r") as bf:
                    line = bf.readline().strip()
                vals = [float(x) for x in line.split(',') if x.strip() != ""]
                cfg.chance_baseline = vals
            except Exception:
                pass
    # Record a human-readable compute formula for plotting
    try:
        if args.logC_col:
            cfg.compute_formula = f"C = exp({args.logC_col})"
        elif args.compute_col:
            cfg.compute_formula = f"C = {args.compute_col}"
        elif args.compute_product_cols:
            c1, c2 = args.compute_product_cols[0], args.compute_product_cols[1]
            cfg.compute_formula = f"C = {args.compute_multiplier:g} × {c1} × {c2}"
    except Exception:
        pass
    runner = SkillFrontier(panel, cfg)
    results = runner.run(output_dir=csv_out_dir)
    grid_logC = results["grid_logC"]
    supports = results["supports"]
    logging.info("Computed support functions for methods: %s", ", ".join(supports.keys()))
    logging.info(
        "Grid logC range: [%.3f, %.3f] with %d points",
        float(np.nanmin(grid_logC)), float(np.nanmax(grid_logC)), len(grid_logC)
    )
    logging.info("Wrote CSV outputs to %s", csv_out_dir)

    # Determine plots directory (optionally split)
    if args.split_output_dir:
        default_plots_dir = os.path.join(base_out, args.plots_subdir)
    else:
        default_plots_dir = os.path.join(base_out, "figures")
    plot_dir = args.plot_dir or default_plots_dir

    if args.plot_max_per_task or args.auto_plots:
        runner.plot_max_per_task(output_dir=plot_dir, num_points=args.plot_points, methods=args.plot_methods)
        logging.info("Saved max-per-task plots to %s", plot_dir)
    if args.plot_pairwise_frontier:
        # Use a dedicated subdir unless user explicitly set plot_dir
        if args.pairwise_dir is not None:
            pair_dir = args.pairwise_dir if os.path.isabs(args.pairwise_dir) else os.path.join(base_out, args.pairwise_dir)
        else:
            pair_dir = os.path.join(base_out, "frontiers_pairwise")
        runner.plot_pairwise_frontiers(output_dir=pair_dir, pairwise_points=args.pairwise_points, C_count=args.pairwise_C_count, methods=args.plot_methods)
        logging.info("Saved pairwise frontier plots to %s", pair_dir)


if __name__ == "__main__":
    main()

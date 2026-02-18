#!/usr/bin/env python3
"""
Unified Sigmoid Frontier Evaluation using Binned Coverage MAE.

Supports multiple train/test splitting strategies:
  - pre2025_vs_2025: Train on pre-2025 data, test on 2025 data
  - period4: 4-period cumulative or single-k splits

For each task:
  - Fit a parametric sigmoid (accuracy vs FLOPs) on train data
  - Build equal-mass bins on train z=log10(FLOPs) with minimum count per bin
  - Compute per-bin coverage on train (IS) and test (OOS)
  - Report Macro/Micro MAE and per-bin diagnostics

Dependencies: numpy; uses new skill_frontier package structure
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from typing import List, Optional, Sequence, Tuple

import numpy as np

# Add repo root to path for imports
REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), os.pardir, os.pardir)
)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

# Import from new package structure
from skill_frontier.io.csv_utils import (
    read_csv_rows,
    detect_date_col,
    detect_oll_raw_tasks,
    maybe_scale_task_values,
    compute_flops,
    parse_date,
    extract_model_id,
)
from skill_frontier.io.manifest_utils import read_manifest
from skill_frontier.io.output_paths import EvaluationRunPaths
from skill_frontier.evaluation.binning import (
    create_equal_mass_bins,
    compute_bin_statistics,
)
from skill_frontier.evaluation.metrics import (
    write_bin_results,
    write_task_summary,
    aggregate_task_metrics,
)
from skill_frontier.evaluation.common import (
    fit_sigmoid_predictor,
    interpolate_curve,
)
from skill_frontier.core.period_utils import assign_period_index_period4


# Period4 helpers (imported from sigmoid.py when needed)
def _parse_year_month(s: str) -> Optional[Tuple[int, int]]:
    """Parse year-month from date string."""
    from skill_frontier.io.csv_utils import parse_year_month

    return parse_year_month(s)


def run_pre2025_vs_2025(args) -> None:
    """Run evaluation with pre-2025 train vs 2025 test split."""
    rows, headers = read_csv_rows(args.csv)
    if not rows:
        raise SystemExit(f"No rows in CSV: {args.csv}")

    date_col = detect_date_col(headers)
    tasks = args.tasks or detect_oll_raw_tasks(headers)
    if not tasks:
        raise SystemExit("No tasks provided or auto-detected")

    # Build arrays for x (compute or size) and years
    z_list: List[float] = []
    C_list: List[float] = []
    years: List[int] = []
    mids: List[str] = []
    use_size = bool(args.size_col and args.size_col in headers)
    for r in rows:
        mid = extract_model_id(r)
        # x: prefer explicit size column if provided and valid, otherwise compute
        x_val = float("nan")
        if use_size:
            raw_sz = r.get(args.size_col, None)
            try:
                x_val = float(raw_sz) if raw_sz not in (None, "", "nan", "NaN") else float("nan")
            except Exception:
                x_val = float("nan")
        if not (np.isfinite(x_val) and x_val > 0.0):
            C = compute_flops(
                r,
                headers,
                logC_col=args.logC_col,
                prod_cols=(
                    tuple(args.compute_product_cols) if args.compute_product_cols else None
                ),
                mult=float(args.compute_multiplier),
            )
            if not (np.isfinite(C) and C > 0.0):
                mids.append(mid)
                z_list.append(float("nan"))
                C_list.append(float("nan"))
                years.append(-1)
                continue
            x_val = float(C)
        z_list.append(float(math.log10(x_val)))
        C_list.append(float(x_val))
        yr = parse_date(r.get(date_col, "")) if date_col else None
        years.append(int(yr) if yr is not None else -1)
        mids.append(mid)

    z_all = np.asarray(z_list, float)
    C_all = np.asarray(C_list, float)
    years = np.asarray(years, int)
    mids_arr = np.asarray(mids, dtype=object)

    # Full train/test masks (split-defined, independent of manifests)
    mask_train_full = years < 2025
    mask_test = years == 2025

    # Fit-time train mask (may be further restricted by manifests)
    mask_train_fit = mask_train_full.copy()

    # Optional manifest filtering
    if args.manifest_pre:
        sel_pre = read_manifest(args.manifest_pre)
        mask_train_fit = mask_train_fit & np.isin(
            mids_arr, np.array(list(sel_pre), dtype=object)
        )
    if args.manifest_2025:
        sel_te = read_manifest(args.manifest_2025)
        mask_test = mask_test & np.isin(
            mids_arr, np.array(list(sel_te), dtype=object)
        )

    os.makedirs(args.out_dir, exist_ok=True)
    paths = EvaluationRunPaths(args.out_dir, legacy=bool(getattr(args, "legacy_output", False)))
    summary_rows: List[Tuple[str, float, float, float, float]] = []

    for task in tasks:
        # Extract y for train/test
        y_list = []
        for r in rows:
            v = r.get(task, None)
            try:
                y = float(v) if v not in (None, "", "nan", "NaN") else float("nan")
            except Exception:
                y = float("nan")
            y_list.append(y)
        y_all = maybe_scale_task_values(np.asarray(y_list, float))

        # Train subset used for fitting (may be restricted by manifest)
        mtr_fit = mask_train_fit & np.isfinite(z_all) & np.isfinite(y_all)
        z_tr_fit = z_all[mtr_fit]
        C_tr_fit = C_all[mtr_fit]
        y_tr_fit = y_all[mtr_fit]
        if z_tr_fit.size < 3:
            print(f"[eval] skip {task}: only {z_tr_fit.size} train points")
            continue

        # Evaluation on full train split (independent of manifests) and bin edges
        mtr_eval = mask_train_full & np.isfinite(z_all) & np.isfinite(y_all)
        z_tr = z_all[mtr_eval]
        C_tr = C_all[mtr_eval]
        y_tr = y_all[mtr_eval]

        edges_tr = create_equal_mass_bins(
            z_tr, int(max(1, args.bins)), int(max(1, args.min_bin_size))
        )
        if edges_tr.size < 2:
            print(f"[eval] skip {task}: insufficient bins after merging")
            continue

        # Fit on (possibly budgeted) train subset using train-bin edges for robust mode
        xs_curve, y_curve = fit_sigmoid_predictor(
            C_tr_fit,
            y_tr_fit,
            tau=float(args.tau),
            frontier_fit_mode=str(args.frontier_fit_mode),
            bins_for_fit=int(args.bins),
            min_bin_size_for_fit=int(args.min_bin_size),
            bin_frontier_quantile=float(args.bin_frontier_quantile),
            bin_trim_fraction=float(args.bin_trim_fraction),
            bin_edges_for_fit=edges_tr,
        )

        # Predictions on full train split
        yhat_tr = interpolate_curve(xs_curve, y_curve, C_tr)

        # Train bins (built on full train z, not subset; edges_tr already computed)
        bins_tr = compute_bin_statistics(z_tr, y_tr, yhat_tr, edges_tr, tau=float(args.tau))

        # Train MAE macro/micro
        vals = [(n, ae) for (_, _, _, n, _, ae) in bins_tr if n > 0 and np.isfinite(ae)]
        if vals:
            n_vec = np.array([v[0] for v in vals], float)
            ae_vec = np.array([v[1] for v in vals], float)
            mae_is_macro = float(np.mean(ae_vec))
            mae_is_micro = float(np.sum(n_vec * ae_vec) / np.sum(n_vec))
        else:
            mae_is_macro = float("nan")
            mae_is_micro = float("nan")

        # Write train bins CSV
        write_bin_results(
            paths.get_bins_path(task, era="train"),
            bins_tr,
            era="train",
        )

        # Test evaluation
        mte = mask_test & np.isfinite(z_all) & np.isfinite(y_all)
        z_te = z_all[mte]
        C_te = C_all[mte]
        y_te = y_all[mte]

        # Build OOS bin edges
        if args.oos_bins == "test_fixed":
            edges_te = (
                create_equal_mass_bins(
                    z_te, int(max(1, args.bins)), int(max(1, args.min_bin_size))
                )
                if z_te.size
                else np.array([])
            )
        else:
            # Train-overlap bins
            if z_te.size >= 1:
                z_lo = float(max(np.min(z_tr), np.min(z_te)))
                z_hi = float(min(np.max(z_tr), np.max(z_te)))
            else:
                z_lo = float(np.min(z_tr))
                z_hi = float(np.max(z_tr))
            e = edges_tr.copy()
            e[0] = max(e[0], z_lo)
            e[-1] = min(e[-1], z_hi)
            keep = [i for i in range(len(e) - 1) if e[i + 1] > e[i] + 1e-12]
            edges_te = (
                np.array([e[i] for i in keep] + [e[keep[-1] + 1]] if keep else [], float)
            )

        mae_oos_macro = float("nan")
        mae_oos_micro = float("nan")
        if edges_te.size >= 2 and z_te.size > 0:
            yhat_te = interpolate_curve(xs_curve, y_curve, C_te)
            bins_te = compute_bin_statistics(z_te, y_te, yhat_te, edges_te, tau=float(args.tau))
            era_label = "test_fixed" if args.oos_bins == "test_fixed" else "test_overlap"
            write_bin_results(
                paths.get_bins_path(task, era=era_label),
                bins_te,
                era=era_label,
            )
            vals_te = [
                (n, ae) for (_, _, _, n, _, ae) in bins_te if n > 0 and np.isfinite(ae)
            ]
            if vals_te:
                n_vec = np.array([v[0] for v in vals_te], float)
                ae_vec = np.array([v[1] for v in vals_te], float)
                mae_oos_macro = float(np.mean(ae_vec))
                mae_oos_micro = float(np.sum(n_vec * ae_vec) / np.sum(n_vec))
        else:
            era_label = "test_fixed" if args.oos_bins == "test_fixed" else "test_overlap"
            write_bin_results(
                paths.get_bins_path(task, era=era_label),
                [],
                era=era_label,
            )

        # Task summary
        write_task_summary(
            paths.get_summary_path(task),
            task=task,
            mae_is_macro=mae_is_macro,
            mae_is_micro=mae_is_micro,
            mae_oos_macro=mae_oos_macro,
            mae_oos_micro=mae_oos_micro,
            K_used=int(len(edges_tr) - 1),
            overlap_used=bool(edges_te.size >= 2),
        )
        summary_rows.append((task, mae_is_macro, mae_is_micro, mae_oos_macro, mae_oos_micro))

    # Aggregate summary
    aggregate_task_metrics(paths.get_aggregate_path(), summary_rows)


def run_period4(args) -> None:
    """Run evaluation with 4-period cumulative or single-k splits."""
    # Import period definitions
    try:
        from skill_frontier.core.sigmoid import PERIOD4_BOUNDS, PERIOD4_SPLITS, PERIOD4_SPLITS_SINGLE
    except Exception:
        try:
            from scripts.smooth_single_skill_frontier import PERIOD4_BOUNDS, PERIOD4_SPLITS, PERIOD4_SPLITS_SINGLE
        except Exception:
            from smooth_single_skill_frontier import PERIOD4_BOUNDS, PERIOD4_SPLITS, PERIOD4_SPLITS_SINGLE

    rows, headers = read_csv_rows(args.csv)
    if not rows:
        raise SystemExit(f"No rows in CSV: {args.csv}")
    date_col = detect_date_col(headers)
    if date_col is None:
        raise SystemExit("Could not detect date column for period split")
    tasks = args.tasks or detect_oll_raw_tasks(headers)
    if not tasks:
        raise SystemExit("No tasks provided or auto-detected")

    # Build arrays (model id, x, z, period index); x can be compute or size
    C_list: List[float] = []
    z_list: List[float] = []
    per_list: List[int] = []
    mid_list: List[str] = []
    use_size = bool(args.size_col and args.size_col in headers)
    for r in rows:
        mid = extract_model_id(r)
        if mid == "":
            mid = ""
        # x: prefer explicit size column if provided and valid, otherwise compute
        x_val = float("nan")
        if use_size:
            raw_sz = r.get(args.size_col, None)
            try:
                x_val = float(raw_sz) if raw_sz not in (None, "", "nan", "NaN") else float("nan")
            except Exception:
                x_val = float("nan")
        if not (np.isfinite(x_val) and x_val > 0.0):
            C = compute_flops(
                r,
                headers,
                logC_col=args.logC_col,
                prod_cols=(
                    tuple(args.compute_product_cols) if args.compute_product_cols else None
                ),
                mult=float(args.compute_multiplier),
            )
            if not (np.isfinite(C) and C > 0.0):
                mid_list.append(mid)
                C_list.append(float("nan"))
                z_list.append(float("nan"))
                per_list.append(-1)
                continue
            x_val = float(C)
        ym = _parse_year_month(r.get(date_col, ""))
        if ym is None:
            mid_list.append(mid)
            C_list.append(float("nan"))
            z_list.append(float("nan"))
            per_list.append(-1)
            continue
        pid = assign_period_index_period4(*ym)
        if pid < 0:
            mid_list.append(mid)
            C_list.append(float("nan"))
            z_list.append(float("nan"))
            per_list.append(-1)
            continue
        mid_list.append(mid)
        C_list.append(float(x_val))
        z_list.append(float(math.log10(x_val)))
        per_list.append(int(pid))

    C_all = np.asarray(C_list, float)
    z_all = np.asarray(z_list, float)
    per_all = np.asarray(per_list, int)
    mid_all = np.asarray(mid_list, dtype=object)

    # Gather task arrays
    y_mat = {}
    for task in tasks:
        y_vals = []
        for r in rows:
            v = r.get(task, None)
            try:
                y = float(v) if v not in (None, "", "nan", "NaN") else float("nan")
            except Exception:
                y = float("nan")
            y_vals.append(y)
        y_mat[task] = maybe_scale_task_values(np.asarray(y_vals, float))

    # Build label->index map to interpret splits when defined by labels
    label_to_idx = {}
    for i, bound in enumerate(PERIOD4_BOUNDS):
        if len(bound) == 3:
            label = bound[0]
        else:
            label = f"k{i+1}"
        label_to_idx[label] = i

    splits_src = PERIOD4_SPLITS if args.train_mode == "cumulative" else PERIOD4_SPLITS_SINGLE

    def _normalize_split(split):
        if isinstance(split, dict):
            train_labels = split.get("train_labels", [])
            val_label = split.get("val_label", None)
            train_idx = [label_to_idx[l] for l in train_labels if l in label_to_idx]
            val_idx = [label_to_idx[val_label]] if val_label in label_to_idx else []
            return train_idx, val_idx
        return split

    # Process each period split
    for k_idx, split in enumerate(splits_src, start=1):
        train_per, val_per = _normalize_split(split)
        # Build full train/val masks (split-defined)
        if args.train_mode == "cumulative":
            mask_tr_full = np.isin(per_all, train_per)
        else:  # single_k
            mask_tr_full = np.zeros_like(per_all, dtype=bool) if not train_per else (per_all == train_per[-1])
        mask_val_eval = np.isin(per_all, val_per)

        # Fit-time train mask (may be further restricted by manifests)
        mask_tr_fit = mask_tr_full.copy()

        # Optional manifest filtering
        if args.manifest_base:
            k_dir = os.path.join(args.manifest_base, f"k{k_idx}")
            train_manifest = os.path.join(k_dir, "manifest__train.txt")
            val_manifest = os.path.join(k_dir, "manifest__val.txt")
            if os.path.exists(train_manifest):
                sel_tr = read_manifest(train_manifest)
                mask_tr_fit = mask_tr_fit & np.isin(mid_all, np.array(list(sel_tr), dtype=object))
            if os.path.exists(val_manifest) and args.manifest_apply == "both":
                sel_val = read_manifest(val_manifest)
                mask_val_eval = mask_val_eval & np.isin(
                    mid_all, np.array(list(sel_val), dtype=object)
                )

        out_dir = os.path.join(args.out_base, f"k{k_idx}")
        os.makedirs(out_dir, exist_ok=True)
        paths = EvaluationRunPaths(out_dir, legacy=bool(getattr(args, "legacy_output", False)))
        summary_rows = []

        for task in tasks:
            y_all = y_mat[task]
            # Train subset for fitting (may be budgeted)
            mtr_fit = mask_tr_fit & np.isfinite(z_all) & np.isfinite(y_all)
            z_tr_fit = z_all[mtr_fit]
            C_tr_fit = C_all[mtr_fit]
            y_tr_fit = y_all[mtr_fit]
            if z_tr_fit.size < 3:
                print(f"[k{k_idx}] skip {task}: only {z_tr_fit.size} train points")
                continue

            # Evaluation on full train split (independent of manifests) and bin edges
            mtr_eval = mask_tr_full & np.isfinite(z_all) & np.isfinite(y_all)
            z_tr = z_all[mtr_eval]
            C_tr = C_all[mtr_eval]
            y_tr = y_all[mtr_eval]

            edges_tr = create_equal_mass_bins(
                z_tr, int(max(1, args.bins)), int(max(1, args.min_bin_size))
            )
            if edges_tr.size < 2:
                print(f"[k{k_idx}] skip {task}: insufficient bins after merging")
                continue

            # Fit on train subset using train-bin edges for robust mode
            xs_curve, y_curve = fit_sigmoid_predictor(
                C_tr_fit,
                y_tr_fit,
                tau=float(args.tau),
                frontier_fit_mode=str(args.frontier_fit_mode),
                bins_for_fit=int(args.bins),
                min_bin_size_for_fit=int(args.min_bin_size),
                bin_frontier_quantile=float(args.bin_frontier_quantile),
                bin_trim_fraction=float(args.bin_trim_fraction),
                bin_edges_for_fit=edges_tr,
            )
            yhat_tr = interpolate_curve(xs_curve, y_curve, C_tr)

            # Train bins from full train z (edges_tr already computed)
            bins_tr = compute_bin_statistics(
                z_tr, y_tr, yhat_tr, edges_tr, tau=float(args.tau)
            )

            # Train MAE
            vals = [
                (n, ae) for (_, _, _, n, _, ae) in bins_tr if n > 0 and np.isfinite(ae)
            ]
            if vals:
                n_vec = np.array([v[0] for v in vals], float)
                ae_vec = np.array([v[1] for v in vals], float)
                mae_is_macro = float(np.mean(ae_vec))
                mae_is_micro = float(np.sum(n_vec * ae_vec) / np.sum(n_vec))
            else:
                mae_is_macro = float("nan")
                mae_is_micro = float("nan")

            write_bin_results(
                paths.get_bins_path(task, era="train"),
                bins_tr,
                era="train",
            )

            # Val evaluation (mask_val_eval may be manifest-filtered if apply='both')
            mval = mask_val_eval & np.isfinite(z_all) & np.isfinite(y_all)
            z_val = z_all[mval]
            C_val = C_all[mval]
            y_val = y_all[mval]

            # Build OOS bin edges
            if args.oos_bins == "test_fixed":
                edges_val = (
                    create_equal_mass_bins(
                        z_val, int(max(1, args.bins)), int(max(1, args.min_bin_size))
                    )
                    if z_val.size
                    else np.array([])
                )
            else:
                if z_val.size >= 1:
                    z_lo = float(max(np.min(z_tr), np.min(z_val)))
                    z_hi = float(min(np.max(z_tr), np.max(z_val)))
                else:
                    z_lo = float(np.min(z_tr))
                    z_hi = float(np.max(z_tr))
                e = edges_tr.copy()
                e[0] = max(e[0], z_lo)
                e[-1] = min(e[-1], z_hi)
                keep = [i for i in range(len(e) - 1) if e[i + 1] > e[i] + 1e-12]
                edges_val = (
                    np.array([e[i] for i in keep] + [e[keep[-1] + 1]] if keep else [], float)
                )

            mae_oos_macro = float("nan")
            mae_oos_micro = float("nan")
            if edges_val.size >= 2 and z_val.size > 0:
                yhat_val = interpolate_curve(xs_curve, y_curve, C_val)
                bins_val = compute_bin_statistics(
                    z_val, y_val, yhat_val, edges_val, tau=float(args.tau)
                )
                era_label = "test_fixed" if args.oos_bins == "test_fixed" else "test_overlap"
                write_bin_results(
                    paths.get_bins_path(task, era=era_label),
                    bins_val,
                    era=era_label,
                )
                vals_val = [
                    (n, ae)
                    for (_, _, _, n, _, ae) in bins_val
                    if n > 0 and np.isfinite(ae)
                ]
                if vals_val:
                    n_vec = np.array([v[0] for v in vals_val], float)
                    ae_vec = np.array([v[1] for v in vals_val], float)
                    mae_oos_macro = float(np.mean(ae_vec))
                    mae_oos_micro = float(np.sum(n_vec * ae_vec) / np.sum(n_vec))
            else:
                era_label = "test_fixed" if args.oos_bins == "test_fixed" else "test_overlap"
                write_bin_results(
                    paths.get_bins_path(task, era=era_label),
                    [],
                    era=era_label,
                )

            # Task summary
            write_task_summary(
                paths.get_summary_path(task),
                task=task,
                mae_is_macro=mae_is_macro,
                mae_is_micro=mae_is_micro,
                mae_oos_macro=mae_oos_macro,
                mae_oos_micro=mae_oos_micro,
                K_used=int(len(edges_tr) - 1),
                overlap_used=bool(edges_val.size >= 2),
            )
            summary_rows.append(
                (task, mae_is_macro, mae_is_micro, mae_oos_macro, mae_oos_micro)
            )

        # Aggregate summary for this period
        aggregate_task_metrics(paths.get_aggregate_path(), summary_rows)


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Unified Sigmoid Frontier Evaluation via Binned Coverage MAE"
    )
    p.add_argument("--csv", required=True, help="Input CSV path")
    p.add_argument(
        "--split_mode",
        choices=["pre2025_vs_2025", "period4"],
        required=True,
        help="Split strategy to use",
    )
    p.add_argument("--logC_col", default=None, help="Log-compute column name")
    p.add_argument(
        "--compute_product_cols", nargs=2, default=None, help="Two columns to multiply for compute"
    )
    p.add_argument(
        "--compute_multiplier", type=float, default=6.0, help="Multiplier for product compute"
    )
    p.add_argument(
        "--size_col",
        default=None,
        help="Optional column to use as x (e.g. '#Params (B)'); if set, overrides compute-based x",
    )
    p.add_argument("--tasks", nargs="*", default=None, help="Task columns to evaluate")
    p.add_argument("--tau", type=float, default=0.98, help="Quantile parameter")
    p.add_argument("--bins", type=int, default=10, help="Target number of bins")
    p.add_argument(
        "--min_bin_size", type=int, default=30, help="Minimum samples per bin"
    )
    p.add_argument(
        "--frontier_fit_mode",
        choices=["quantile_per_point", "robust_bin_frontier"],
        default="quantile_per_point",
        help="How to fit the sigmoid frontier (default: quantile_per_point)",
    )
    p.add_argument(
        "--bin_frontier_quantile",
        type=float,
        default=0.98,
        help="Quantile level for bin-level robust frontier targets when using robust_bin_frontier mode",
    )
    p.add_argument(
        "--bin_trim_fraction",
        type=float,
        default=0.01,
        help="Fraction of top points per bin to trim before computing the bin-level frontier quantile",
    )
    p.add_argument(
        "--oos_bins",
        choices=["train_overlap", "test_fixed"],
        default="train_overlap",
        help="How to define OOS bins",
    )
    p.add_argument(
        "--legacy_output",
        action="store_true",
        help="Write legacy flat files (no bins/ or summaries/ subdirectories)",
    )

    # pre2025_vs_2025 specific
    p.add_argument(
        "--out_dir",
        default=None,
        help="Output directory (for pre2025_vs_2025 mode)",
    )
    p.add_argument(
        "--manifest_pre",
        default=None,
        help="Manifest file for pre-2025 train models",
    )
    p.add_argument(
        "--manifest_2025",
        default=None,
        help="Manifest file for 2025 test models",
    )

    # period4 specific
    p.add_argument(
        "--train_mode",
        choices=["cumulative", "single_k"],
        default="cumulative",
        help="Training mode for period4 (cumulative or single_k)",
    )
    p.add_argument(
        "--out_base",
        default=None,
        help="Output base directory (for period4 mode)",
    )
    p.add_argument(
        "--manifest_base",
        default=None,
        help="Base directory for period4 manifests",
    )
    p.add_argument(
        "--manifest_apply",
        choices=["both", "train_only"],
        default="both",
        help="Apply manifests to both train and val, or only train",
    )

    return p


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = build_argparser().parse_args(argv)

    # Set default output directories based on split mode
    if args.split_mode == "pre2025_vs_2025":
        if args.out_dir is None:
            args.out_dir = os.path.join(
                "outputs", "evaluation", "sigmoid", "year_split", "no_budget"
            )
        run_pre2025_vs_2025(args)
    elif args.split_mode == "period4":
        if args.out_base is None:
            args.out_base = os.path.join(
                "outputs", "evaluation", "sigmoid", "period4", args.train_mode, "no_budget"
            )
        run_period4(args)


if __name__ == "__main__":
    main()

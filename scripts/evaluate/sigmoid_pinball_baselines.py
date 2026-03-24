#!/usr/bin/env python3
"""
Compare sigmoid frontier pinball loss against simple baselines
for the period4 single-k, no-budget setting.

For each task and period k in {1,2,3}, we compute:
  - Per-bin and global smoothed pinball loss for:
      * Sigmoid frontier (existing fit)
      * Null frontier (global tau-quantile, no compute dependence)
      * Oracle binwise frontier (per-bin tau-quantiles on train)
  - Summary metrics such as:
      * L_sigmoid, L_null, L_oracle
      * L_sigmoid / L_oracle
      * R2_pinball = 1 - L_sigmoid / L_null

Outputs are written under:
    evaluation_pinball_baselines/period4_singlek_no_budget/
mirroring the period4_singlek_no_budget structure.

This script does not modify existing evaluation code or outputs.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.optimize import minimize, minimize_scalar  
from scipy.special import expit

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from skill_frontier.io.csv_utils import (  # type: ignore
    read_csv_rows,
    detect_date_col,
    detect_oll_raw_tasks,
    compute_flops,
    extract_model_id,
    parse_year_month,
    sanitize_name,
)
from skill_frontier.core.sigmoid import (  # type: ignore
    PERIOD4_BOUNDS,
    PERIOD4_SPLITS_SINGLE,
)
from skill_frontier.core.period_utils import assign_period_index_period4  # type: ignore
from skill_frontier.evaluation.binning import create_equal_mass_bins  # type: ignore
from skill_frontier.evaluation.common import (  # type: ignore
    fit_sigmoid_predictor,
    interpolate_curve,
)
from skill_frontier.evaluation.pinball_utils import smooth_pinball_loss  # type: ignore
from skill_frontier.evaluation.metrics import aggregate_task_metrics  # type: ignore
from scripts.run import ispline_pinball_optimizer as iso  # type: ignore


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Baseline comparison for sigmoid frontiers using smoothed pinball loss "
            "(period4 single-k, no budget)."
        )
    )
    default_csv = os.path.join(
        REPO_ROOT,
        "tables",
        "open_llm_leaderboard",
        "open_llm_leaderboard_with_tokens.csv",
    )
    p.add_argument("--csv", default=default_csv, help="Input OLL CSV (with tokens schema).")
    p.add_argument(
        "--compute_product_cols",
        nargs=2,
        default=["Pretraining tokens (T)", "#Params (B)"],
        help="Two columns to multiply for compute FLOPs proxy.",
    )
    p.add_argument(
        "--compute_multiplier",
        type=float,
        default=6.0,
        help="Multiplier applied to product of compute columns.",
    )
    p.add_argument(
        "--size_col",
        default=None,
        help="Optional column to use as x (e.g. '#Params (B)'); if set, overrides compute-based x.",
    )
    p.add_argument(
        "--tau",
        type=float,
        default=0.98,
        help="Quantile parameter tau used for pinball loss and baselines.",
    )
    p.add_argument(
        "--kappa_final",
        type=float,
        default=50.0,
        help="Sigmoid fitter smoothing parameter kappa (κ).",
    )
    p.add_argument(
        "--lambda_b",
        type=float,
        default=None,
        help="Sigmoid regularization weight lambda (λ); if omitted, uses the repo default.",
    )
    p.add_argument(
        "--bins",
        type=int,
        default=10,
        help="Target number of equal-mass bins in log-compute.",
    )
    p.add_argument(
        "--min_bin_size",
        type=int,
        default=30,
        help="Minimum number of samples per bin (after merging).",
    )
    p.add_argument(
        "--frontier_fit_mode",
        choices=["quantile_per_point", "robust_bin_frontier"],
        default="quantile_per_point",
        help="How to fit the sigmoid frontier (kept consistent with main evaluation).",
    )
    p.add_argument(
        "--bin_frontier_quantile",
        type=float,
        default=0.98,
        help="Quantile for bin-level robust targets when using robust_bin_frontier.",
    )
    p.add_argument(
        "--bin_trim_fraction",
        type=float,
        default=0.01,
        help="Trim fraction for robust bin-level frontier.",
    )
    p.add_argument(
        "--task_group",
        choices=["main", "bbh_subtasks"],
        default="main",
        help="Which task set to evaluate: 'main' six tasks or BBH subtasks.",
    )
    p.add_argument(
        "--out_base",
        default=os.path.join(
            REPO_ROOT, "evaluation_pinball_baselines", "period4_singlek_no_budget"
        ),
        help="Output base directory for baseline comparison results.",
    )
    p.add_argument(
        "--small_max_size",
        type=float,
        default=None,
        help="If set, additionally evaluate a size-only subset (e.g., models with size <= this value, in the same units as size_col). "
             "Frontiers are still fitted on the full train split; only the evaluation masks and bins are restricted.",
    )
    p.add_argument(
        "--out_base_small",
        default=None,
        help="Optional output base for the small-model evaluation; if not set but --small_max_size is provided, defaults to out_base + '_small'.",
    )
    return p


def _best_constant_pinball(y: np.ndarray, tau: float) -> float:
    """Return scalar c in [0,1] minimizing mean smoothed pinball loss on y."""
    y = np.asarray(y, float)
    y = y[np.isfinite(y)]
    if y.size == 0:
        return float("nan")

    # Use tau-quantile as a robust initial guess / fallback.
    q0 = float(np.quantile(y, tau))

    def _obj(c: float) -> float:
        r = y - c
        return float(np.mean(smooth_pinball_loss(r, tau=tau)))

    try:
        res = minimize_scalar(_obj, bounds=(0.0, 1.0), method="bounded", options={"xatol": 1e-6})
        if res.success and np.isfinite(res.x):
            return float(res.x)
    except Exception:
        pass
    return q0


def _compute_bin_pinball(
    z: np.ndarray,
    y: np.ndarray,
    yhat: np.ndarray,
    edges: np.ndarray,
    tau: float,
) -> List[Tuple[int, float, float, int, float]]:
    """Per-bin pinball statistics: (bin_id, z_lo, z_hi, n, loss_mean)."""
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
        if n == 0:
            loss_mean = float("nan")
        else:
            r = y[mask] - yhat[mask]
            loss_vals = smooth_pinball_loss(r, tau=tau)
            loss_mean = float(np.mean(loss_vals))
        out.append((i, lo, hi, n, loss_mean))
    return out


def _macro_calibration_error(
    z: np.ndarray,
    y: np.ndarray,
    yhat: np.ndarray,
    edges: np.ndarray,
    tau: float,
) -> float:
    """Macro-average coverage error (mean |hat_tau - tau| over non-empty bins)."""
    z = np.asarray(z, float)
    y = np.asarray(y, float)
    yhat = np.asarray(yhat, float)
    edges = np.asarray(edges, float)
    if edges.size < 2:
        return float("nan")
    # Use the same binning/coverage metric as sigmoid_binned_mae, but guard against non-finite yhat.
    out = []
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
        out.append(abs(cov - float(tau)))
    if not out:
        return float("nan")
    return float(np.mean(np.asarray(out, float)))


def _fit_ispline_frontier(
    z_tr: np.ndarray,
    y_tr: np.ndarray,
    tau: float,
    edges_ispline: np.ndarray,
) -> iso.FitResult:
    """Wrapper to fit monotone I-spline frontier using the enhanced optimizer."""
    z_tr = np.asarray(z_tr, float)
    y_tr = np.asarray(y_tr, float)
    mask = np.isfinite(z_tr) & np.isfinite(y_tr)
    z_tr = z_tr[mask]
    y_tr = y_tr[mask]
    if z_tr.size == 0 or edges_ispline.size < 2:
        return iso.FitResult(
            params=np.array([], dtype=float),
            loss=float("nan"),
            success=False,
            message="not enough data or edges",
            edges=np.asarray(edges_ispline, float),
        )

    n_bins = max(2, int(edges_ispline.size - 1))
    return iso.fit_ispline_enhanced(
        z=z_tr,
        y=y_tr,
        tau=float(tau),
        n_knot_grid=int(n_bins),
        n_random=20,
        seed=0,
        maxiter=700,
    )


def _predict_ispline_frontier(fit: iso.FitResult, z: np.ndarray) -> np.ndarray:
    """Evaluate a fitted I-spline frontier."""
    if fit is None or fit.params.size == 0 or fit.edges.size < 2:
        return np.full_like(np.asarray(z, float), np.nan, dtype=float)
    return iso._predict_from_params(np.asarray(z, float), fit.params, fit.edges)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = _build_argparser().parse_args(argv)

    out_base_small = args.out_base_small
    if args.small_max_size is not None and out_base_small is None:
        out_base_small = args.out_base + "_small"

    rows, headers = read_csv_rows(args.csv)
    if not rows:
        raise SystemExit("No rows in CSV")

    date_col = detect_date_col(headers)
    if date_col is None:
        raise SystemExit("Could not detect date column for period split")

    if args.task_group == "main":
        tasks = detect_oll_raw_tasks(headers)
    else:
        # BBH subtasks: columns containing 'leaderboard_bbh_'
        tasks = [h for h in headers if "leaderboard_bbh_" in h]
    if not tasks:
        raise SystemExit("No task columns detected for the requested task_group")

    # Build arrays for x (compute or size) and period indices
    C_list: List[float] = []
    z_list: List[float] = []
    per_list: List[int] = []
    mid_list: List[str] = []
    use_size = bool(args.size_col and args.size_col in headers)

    for r in rows:
        mid = extract_model_id(r)
        if mid == "":
            mid = ""
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
                logC_col=None,
                prod_cols=tuple(args.compute_product_cols),
                mult=float(args.compute_multiplier),
            )
            if not (np.isfinite(C) and C > 0.0):
                mid_list.append(mid)
                C_list.append(float("nan"))
                z_list.append(float("nan"))
                per_list.append(-1)
                continue
            x_val = float(C)
        ym = parse_year_month(r.get(date_col, ""))
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

    # Gather task arrays
    y_mat: Dict[str, np.ndarray] = {}
    for task in tasks:
        y_vals: List[float] = []
        for r in rows:
            v = r.get(task, None)
            try:
                y = float(v) if v not in (None, "", "nan", "NaN") else float("nan")
            except Exception:
                y = float("nan")
            y_vals.append(y)
        y_mat[task] = np.asarray(y_vals, float)

    # Map label -> index to interpret PERIOD4_SPLITS_SINGLE
    label_to_idx: Dict[str, int] = {}
    for i, bound in enumerate(PERIOD4_BOUNDS):
        if len(bound) == 3:
            label = bound[0]
        else:
            label = f"k{i+1}"
        label_to_idx[label] = i

    def _normalize_split(split_obj):
        if isinstance(split_obj, dict):
            train_labels = split_obj.get("train_labels", [])
            val_label = split_obj.get("val_label", None)
            train_idx = [label_to_idx[l] for l in train_labels if l in label_to_idx]
            val_idx = [label_to_idx[val_label]] if val_label in label_to_idx else []
            return train_idx, val_idx
        return split_obj

    os.makedirs(args.out_base, exist_ok=True)

    # Process each single-k split
    for k_idx, split in enumerate(PERIOD4_SPLITS_SINGLE, start=1):
        train_per, val_per = _normalize_split(split)
        # For single_k, train_per should contain exactly one index
        if not train_per:
            continue
        train_pid = train_per[-1]

        mask_tr = (per_all == train_pid)
        mask_val = np.isin(per_all, val_per)

        out_dir_k = os.path.join(args.out_base, f"k{k_idx}")
        os.makedirs(out_dir_k, exist_ok=True)
        out_dir_k_small = None
        if out_base_small:
            out_dir_k_small = os.path.join(out_base_small, f"k{k_idx}")
            os.makedirs(out_dir_k_small, exist_ok=True)

        summary_rows: List[Tuple[str, float, float, float, float, float, float]] = []

        for task in tasks:
            y_all = y_mat[task]

            mtr = mask_tr & np.isfinite(z_all) & np.isfinite(y_all)
            z_tr = z_all[mtr]
            C_tr = C_all[mtr]
            y_tr = y_all[mtr]
            if z_tr.size < 3:
                print(f"[baseline k{k_idx}] skip {task}: only {z_tr.size} train points")
                continue

            # Train bins (equal-mass) from full train z
            edges_tr = create_equal_mass_bins(
                z_tr, int(max(1, args.bins)), int(max(1, args.min_bin_size))
            )
            if edges_tr.size < 2:
                print(f"[baseline k{k_idx}] skip {task}: insufficient bins after merging")
                continue

            # Sigmoid frontier fitted on train subset
            xs_curve, y_curve = fit_sigmoid_predictor(
                C_tr,
                y_tr,
                tau=float(args.tau),
                frontier_fit_mode=str(args.frontier_fit_mode),
                bins_for_fit=int(args.bins),
                min_bin_size_for_fit=int(args.min_bin_size),
                bin_frontier_quantile=float(args.bin_frontier_quantile),
                bin_trim_fraction=float(args.bin_trim_fraction),
                bin_edges_for_fit=edges_tr,
                kappa_final=float(args.kappa_final),
                lambda_b=args.lambda_b,
            )
            yhat_tr_sig = interpolate_curve(xs_curve, y_curve, C_tr)

            # I-spline frontier driven by a 3-bin equal-mass partition on z_tr
            edges_ispline = create_equal_mass_bins(z_tr, K=3, min_bin=1)
            theta_mc = _fit_ispline_frontier(
                z_tr, y_tr, tau=float(args.tau), edges_ispline=edges_ispline
            )
            yhat_tr_mc = _predict_ispline_frontier(theta_mc, z_tr)

            # Null frontier: global tau-quantile on train y
            y_tr_valid = y_tr[np.isfinite(y_tr)]
            if y_tr_valid.size == 0:
                print(f"[baseline k{k_idx}] skip {task}: no valid train labels")
                continue
            q_null = float(np.quantile(y_tr_valid, args.tau))
            yhat_tr_null = np.full_like(y_tr, q_null)

            # Oracle frontier: per-bin tau-quantiles on train
            B = len(edges_tr) - 1
            q_bins = np.full((B,), np.nan, dtype=float)
            for i in range(B):
                lo, hi = float(edges_tr[i]), float(edges_tr[i + 1])
                if i < B - 1:
                    mask_bin = (z_tr >= lo) & (z_tr < hi)
                else:
                    mask_bin = (z_tr >= lo) & (z_tr <= hi)
                y_bin = y_tr[mask_bin & np.isfinite(y_tr)]
                if y_bin.size == 0:
                    continue
                # Binwise constant that minimises the smoothed pinball loss on this bin.
                q_bins[i] = _best_constant_pinball(y_bin, tau=float(args.tau))

            # Helper to map z to oracle predictions
            def _predict_oracle(z_vec: np.ndarray) -> np.ndarray:
                z_vec = np.asarray(z_vec, float)
                idx = np.searchsorted(edges_tr, z_vec, side="right") - 1
                idx = np.clip(idx, 0, B - 1)
                yhat = q_bins[idx]
                return yhat

            yhat_tr_oracle = _predict_oracle(z_tr)

            # Validation/OOS subset: restrict to overlapping z-range with train
            mval = mask_val & np.isfinite(z_all) & np.isfinite(y_all)
            z_val_full = z_all[mval]
            C_val_full = C_all[mval]
            y_val_full = y_all[mval]
            if z_val_full.size > 0:
                z_lo = float(max(np.min(z_tr), np.min(z_val_full)))
                z_hi = float(min(np.max(z_tr), np.max(z_val_full)))
                m_overlap = (z_val_full >= z_lo) & (z_val_full <= z_hi)
                z_val = z_val_full[m_overlap]
                C_val = C_val_full[m_overlap]
                y_val = y_val_full[m_overlap]
            else:
                z_val = np.array([], float)
                C_val = np.array([], float)
                y_val = np.array([], float)

            yhat_val_sig = (
                interpolate_curve(xs_curve, y_curve, C_val) if z_val.size else np.array([], float)
            )
            yhat_val_null = np.full_like(y_val, q_null)
            yhat_val_oracle = _predict_oracle(z_val) if z_val.size else np.array([], float)
            yhat_val_mc = _predict_ispline_frontier(theta_mc, z_val) if z_val.size else np.array([], float)

            # Coverage error (macro) in-sample / out-of-sample (binned coverage MAE)
            # Train bins: edges_tr (built on train z). OOS bins: test_fixed (built on val z) for consistency
            # with scripts/evaluate/sigmoid_binned_mae.py default.
            edges_val = (
                create_equal_mass_bins(
                    z_val, int(max(1, args.bins)), int(max(1, args.min_bin_size))
                )
                if z_val.size
                else np.array([])
            )
            ce_tr_sig = _macro_calibration_error(z_tr, y_tr, yhat_tr_sig, edges_tr, tau=float(args.tau))
            ce_tr_null = _macro_calibration_error(z_tr, y_tr, yhat_tr_null, edges_tr, tau=float(args.tau))
            ce_tr_oracle = _macro_calibration_error(z_tr, y_tr, yhat_tr_oracle, edges_tr, tau=float(args.tau))
            ce_tr_mc = _macro_calibration_error(z_tr, y_tr, yhat_tr_mc, edges_tr, tau=float(args.tau))

            ce_val_sig = _macro_calibration_error(z_val, y_val, yhat_val_sig, edges_val, tau=float(args.tau))
            ce_val_null = _macro_calibration_error(z_val, y_val, yhat_val_null, edges_val, tau=float(args.tau))
            ce_val_oracle = _macro_calibration_error(z_val, y_val, yhat_val_oracle, edges_val, tau=float(args.tau))
            ce_val_mc = _macro_calibration_error(z_val, y_val, yhat_val_mc, edges_val, tau=float(args.tau))

            # Global pinball losses (train and val)
            def _global_loss(y: np.ndarray, yhat: np.ndarray) -> float:
                mask = np.isfinite(y) & np.isfinite(yhat)
                if not np.any(mask):
                    return float("nan")
                r = y[mask] - yhat[mask]
                return float(np.mean(smooth_pinball_loss(r, tau=float(args.tau))))

            L_tr_sig = _global_loss(y_tr, yhat_tr_sig)
            L_tr_null = _global_loss(y_tr, yhat_tr_null)
            L_tr_oracle = _global_loss(y_tr, yhat_tr_oracle)
            L_tr_mc = _global_loss(y_tr, yhat_tr_mc)

            L_val_sig = _global_loss(y_val, yhat_val_sig)
            L_val_null = _global_loss(y_val, yhat_val_null)
            L_val_oracle = _global_loss(y_val, yhat_val_oracle)
            L_val_mc = _global_loss(y_val, yhat_val_mc)

            # Ratios and pseudo-R2 (per split)
            def _safe_ratio(num: float, den: float) -> float:
                if not (np.isfinite(num) and np.isfinite(den)) or den <= 0.0:
                    return float("nan")
                return float(num / den)

            def _safe_r2(L_model: float, L_null: float) -> float:
                if not (np.isfinite(L_model) and np.isfinite(L_null)) or L_null <= 0.0:
                    return float("nan")
                return float(1.0 - L_model / L_null)

            ratio_tr_sig_oracle = _safe_ratio(L_tr_sig, L_tr_oracle)
            ratio_val_sig_oracle = _safe_ratio(L_val_sig, L_val_oracle)
            r2_tr_sig = _safe_r2(L_tr_sig, L_tr_null)
            r2_val_sig = _safe_r2(L_val_sig, L_val_null)

            summary_rows.append(
                (
                    task,
                    L_tr_sig,
                    L_tr_null,
                    L_tr_oracle,
                    ratio_tr_sig_oracle,
                    r2_tr_sig,
                    float(L_tr_sig * 1000.0),
                )
            )

            # Write per-task CSV for both splits
            out_task = os.path.join(out_dir_k, f"{sanitize_name(task)}__pinball_baselines.csv")
            with open(out_task, "w", newline="") as f:
                import csv

                w = csv.writer(f)
                w.writerow(
                    [
                        "split",
                        "L_sigmoid",
                        "L_ispline",
                        "L_null",
                        "L_oracle",
                        "CE_sigmoid",
                        "CE_ispline",
                        "CE_null",
                        "CE_oracle",
                        "ratio_sigmoid_oracle",
                        "R2_pinball",
                    ]
                )
                w.writerow(
                    [
                        "train",
                        L_tr_sig,
                        L_tr_mc,
                        L_tr_null,
                        L_tr_oracle,
                        ce_tr_sig,
                        ce_tr_mc,
                        ce_tr_null,
                        ce_tr_oracle,
                        ratio_tr_sig_oracle,
                        r2_tr_sig,
                    ]
                )
                w.writerow(
                    [
                        "val",
                        L_val_sig,
                        L_val_mc,
                        L_val_null,
                        L_val_oracle,
                        ce_val_sig,
                        ce_val_mc,
                        ce_val_null,
                        ce_val_oracle,
                        ratio_val_sig_oracle,
                        r2_val_sig,
                    ]
                )

            # Optional: per-bin pinball CSV for inspection (train only, to keep size small)
            bins_tr_sig = _compute_bin_pinball(z_tr, y_tr, yhat_tr_sig, edges_tr, tau=float(args.tau))
            bins_tr_null = _compute_bin_pinball(z_tr, y_tr, yhat_tr_null, edges_tr, tau=float(args.tau))
            bins_tr_oracle = _compute_bin_pinball(
                z_tr, y_tr, yhat_tr_oracle, edges_tr, tau=float(args.tau)
            )
            out_bins = os.path.join(
                out_dir_k, f"{sanitize_name(task)}__bins_train_pinball_baselines.csv"
            )
            with open(out_bins, "w", newline="") as f:
                import csv

                w = csv.writer(f)
                w.writerow(
                    [
                        "bin_id",
                        "z_lo",
                        "z_hi",
                        "n_sigmoid",
                        "loss_sigmoid",
                        "loss_null",
                        "loss_oracle",
                    ]
                )
                for (i, lo, hi, n_sig, loss_sig), (_, _, _, n_null, loss_null), (
                    _,
                    _,
                    _,
                    n_orc,
                    loss_orc,
                ) in zip(bins_tr_sig, bins_tr_null, bins_tr_oracle):
                    n_consistent = n_sig  # all three use same mask definition
                    w.writerow(
                        [
                            i,
                            lo,
                            hi,
                            n_consistent,
                            loss_sig,
                            loss_null,
                            loss_orc,
                        ]
                    )

            # Small-model evaluation (no refit; reuse predictions, recompute bins on small subset)
            if out_dir_k_small and args.small_max_size is not None and use_size:
                C_tr = C_all[mtr]
                mask_tr_small = (C_tr <= float(args.small_max_size))
                z_tr_s = z_tr[mask_tr_small]
                y_tr_s = y_tr[mask_tr_small]
                # Validation (already overlap-restricted), use same overlap mask then size filter
                C_val = C_val if 'C_val' in locals() else np.array([], float)
                mask_val_small = (C_val <= float(args.small_max_size)) if C_val.size else np.array([], bool)
                z_val_s = z_val[mask_val_small] if mask_val_small.size else np.array([], float)
                y_val_s = y_val[mask_val_small] if mask_val_small.size else np.array([], float)
                yhat_tr_sig_s = yhat_tr_sig[mask_tr_small]
                yhat_val_sig_s = yhat_val_sig[mask_val_small] if mask_val_small.size else np.array([], float)
                yhat_tr_mc_s = yhat_tr_mc[mask_tr_small]
                yhat_val_mc_s = yhat_val_mc[mask_val_small] if mask_val_small.size else np.array([], float)

                if z_tr_s.size >= 2:
                    edges_tr_s = create_equal_mass_bins(
                        z_tr_s, int(max(1, args.bins)), int(max(1, args.min_bin_size))
                    )
                else:
                    edges_tr_s = np.array([float("-inf"), float("inf")])

                # Null and oracle recomputed on the small train subset for fairness
                y_tr_valid_s = y_tr_s[np.isfinite(y_tr_s)]
                q_null_s = float(np.quantile(y_tr_valid_s, args.tau)) if y_tr_valid_s.size else float("nan")
                yhat_tr_null_s = np.full_like(y_tr_s, q_null_s)
                yhat_val_null_s = np.full_like(y_val_s, q_null_s)

                yhat_tr_oracle_s = np.full_like(y_tr_s, float("nan"))
                yhat_val_oracle_s = np.full_like(y_val_s, float("nan"))
                if edges_tr_s.size >= 2:
                    B = edges_tr_s.size - 1
                    q_bins_s = np.full(B, float("nan"))
                    for i, (lo, hi) in enumerate(zip(edges_tr_s[:-1], edges_tr_s[1:])):
                        in_bin = (z_tr_s >= lo) & (z_tr_s < hi if i < B - 1 else z_tr_s <= hi)
                        if np.any(in_bin):
                            # Binwise constant that minimises smoothed pinball loss
                            q_bins_s[i] = _best_constant_pinball(y_tr_s[in_bin], tau=float(args.tau))
                    # Map via searchsorted to avoid gaps/edge effects
                    idx_tr = np.clip(np.searchsorted(edges_tr_s, z_tr_s, side="right") - 1, 0, B - 1)
                    yhat_tr_oracle_s = q_bins_s[idx_tr]
                    idx_val = np.clip(np.searchsorted(edges_tr_s, z_val_s, side="right") - 1, 0, B - 1) if z_val_s.size else np.array([], int)
                    yhat_val_oracle_s = q_bins_s[idx_val] if z_val_s.size else np.array([], float)

                def _mean_pinball(y, yhat):
                    r = y - yhat
                    return float(np.mean(smooth_pinball_loss(r, tau=float(args.tau)))) if y.size else float("nan")

                L_tr_sig_s = _mean_pinball(y_tr_s, yhat_tr_sig_s)
                L_tr_mc_s = _mean_pinball(y_tr_s, yhat_tr_mc_s)
                L_tr_null_s = _mean_pinball(y_tr_s, yhat_tr_null_s)
                L_tr_oracle_s = _mean_pinball(y_tr_s, yhat_tr_oracle_s)

                L_val_sig_s = _mean_pinball(y_val_s, yhat_val_sig_s)
                L_val_mc_s = _mean_pinball(y_val_s, yhat_val_mc_s)
                L_val_null_s = _mean_pinball(y_val_s, yhat_val_null_s)
                L_val_oracle_s = _mean_pinball(y_val_s, yhat_val_oracle_s)

                ratio_tr_sig_oracle_s = float(L_tr_sig_s / L_tr_oracle_s) if np.isfinite(L_tr_sig_s) and np.isfinite(L_tr_oracle_s) and L_tr_oracle_s != 0 else float("nan")
                ratio_val_sig_oracle_s = float(L_val_sig_s / L_val_oracle_s) if np.isfinite(L_val_sig_s) and np.isfinite(L_val_oracle_s) and L_val_oracle_s != 0 else float("nan")
                r2_tr_sig_s = _safe_r2(L_tr_sig_s, L_tr_null_s)
                r2_val_sig_s = _safe_r2(L_val_sig_s, L_val_null_s)

                # Write per-task CSV for small subset
                out_task_s = os.path.join(out_dir_k_small, f"{sanitize_name(task)}__pinball_baselines.csv")
                with open(out_task_s, "w", newline="") as f:
                    import csv

                    w = csv.writer(f)
                    w.writerow(
                        [
                            "split",
                            "L_sigmoid",
                            "L_ispline",
                            "L_null",
                            "L_oracle",
                            "ratio_sigmoid_oracle",
                            "R2_pinball",
                        ]
                    )
                    w.writerow(
                        [
                            "train",
                            L_tr_sig_s,
                            L_tr_mc_s,
                            L_tr_null_s,
                            L_tr_oracle_s,
                            ratio_tr_sig_oracle_s,
                            r2_tr_sig_s,
                        ]
                    )
                    w.writerow(
                        [
                            "val",
                            L_val_sig_s,
                            L_val_mc_s,
                            L_val_null_s,
                            L_val_oracle_s,
                            ratio_val_sig_oracle_s,
                            r2_val_sig_s,
                        ]
                    )

                # Per-bin (train) for small subset
                bins_tr_sig_s = _compute_bin_pinball(z_tr_s, y_tr_s, yhat_tr_sig_s, edges_tr_s, tau=float(args.tau))
                bins_tr_null_s = _compute_bin_pinball(z_tr_s, y_tr_s, yhat_tr_null_s, edges_tr_s, tau=float(args.tau))
                bins_tr_oracle_s = _compute_bin_pinball(z_tr_s, y_tr_s, yhat_tr_oracle_s, edges_tr_s, tau=float(args.tau))
                out_bins_s = os.path.join(out_dir_k_small, f"{sanitize_name(task)}__bins_train_pinball_baselines.csv")
                with open(out_bins_s, "w", newline="") as f:
                    import csv

                    w = csv.writer(f)
                    w.writerow(
                        [
                            "bin_id",
                            "z_lo",
                            "z_hi",
                            "n_sigmoid",
                            "loss_sigmoid",
                            "loss_null",
                            "loss_oracle",
                        ]
                    )
                    for (i, lo, hi, n_sig, loss_sig), (_, _, _, n_null, loss_null), (
                        _,
                        _,
                        _,
                        n_orc,
                        loss_orc,
                    ) in zip(bins_tr_sig_s, bins_tr_null_s, bins_tr_oracle_s):
                        n_consistent = n_sig
                        w.writerow([i, lo, hi, n_consistent, loss_sig, loss_null, loss_orc])

        # Aggregate summary across tasks for this k (using sigmoid train loss as metric)
        if summary_rows:
            agg_path = os.path.join(out_dir_k, "summary_over_tasks_pinball.csv")
            # aggregate_task_metrics expects (task, mae_is_macro, mae_is_micro, mae_oos_macro, mae_oos_micro)
            # We repurpose it by feeding sigmoid train/val losses as "IS/OOS macro".
            rows_for_agg: List[Tuple[str, float, float, float, float]] = []
            for (task, L_tr_sig, _L_tr_null, _L_tr_oracle, _ratio, _r2, _) in summary_rows:
                rows_for_agg.append((task, L_tr_sig, float("nan"), float("nan"), float("nan")))
            aggregate_task_metrics(agg_path, rows_for_agg)


if __name__ == "__main__":
    main()

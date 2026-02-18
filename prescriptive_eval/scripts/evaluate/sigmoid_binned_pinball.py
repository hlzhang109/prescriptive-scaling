#!/usr/bin/env python3
"""
Sigmoid Frontier Evaluation using Per-Bin Pinball Loss (Period4 Single-k).

This script mirrors the outputs of `scripts/evaluate/sigmoid_binned_mae.py`,
but:
  - For train bins, `abs_err` is the mean smoothed pinball loss in that bin.
  - For OOS bins (k=1..3), `abs_err` is a *delta* between two fits:
        loss(fit on period k, eval on period k+1)
      - loss(fit on period k+1, eval on period k+1)

Key properties:
  * Existing code and outputs are not modified.
  * Results are written into a new root directory:
        evaluation_pinball/period4_singlek_no_budget/
    mirroring the structure of:
        outputs/evaluation/period4_singlek_no_budget/
  * The plotting script `scripts/plot/eval_sigmoid.py` can be pointed at
    this base directory via `--period4_singlek_base` to produce figures
    with identical style, now driven by pinball loss instead of MAE.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), os.pardir, os.pardir)
)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from skill_frontier.core.period_utils import assign_period_index_period4
from skill_frontier.evaluation.pinball_utils import smooth_pinball_loss  # type: ignore


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Evaluate sigmoid frontiers with per-bin pinball loss "
            "(period4 single-k, no budget)."
        )
    )
    default_csv = os.path.join(
        REPO_ROOT,
        "tables",
        "open_llm_leaderboard",
        "open_llm_leaderboard_with_tokens.csv",
    )
    p.add_argument(
        "--csv",
        default=default_csv,
        help="Input OLL CSV (default: with_tokens schema).",
    )
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
        "--tau",
        type=float,
        default=0.98,
        help="Quantile parameter tau used for pinball loss.",
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
        "--oos_bins",
        choices=["train_overlap", "test_fixed"],
        default="train_overlap",
        help="How to define OOS bins (same semantics as sigmoid_binned_mae.py).",
    )
    p.add_argument(
        "--frontier_fit_mode",
        choices=["quantile_per_point", "robust_bin_frontier"],
        default="quantile_per_point",
        help="How to fit the sigmoid frontier (default: quantile_per_point).",
    )
    p.add_argument("--bin_frontier_quantile", type=float, default=0.98)
    p.add_argument("--bin_trim_fraction", type=float, default=0.01)
    p.add_argument(
        "--out_base",
        default=os.path.join(
            REPO_ROOT, "evaluation_pinball", "period4_singlek_no_budget"
        ),
        help="Output base directory for pinball evaluation.",
    )
    return p


def _compute_bin_pinball_stats(
    z: np.ndarray,
    y: np.ndarray,
    yhat: np.ndarray,
    edges: np.ndarray,
    tau: float,
) -> List[Tuple[int, float, float, int, float, float]]:
    """
    Compute per-bin statistics using smoothed pinball loss.

    For each bin:
      * hat_tau = empirical coverage (fraction of y <= yhat) as before.
      * abs_err = average smoothed pinball loss in that bin.

    The pinball loss matches the fitter:
        r = y - yhat
        rho_tilde(r) = (1/k) log(1 + exp(k r)) + (tau - 1) r,  k=50.
    """
    z = np.asarray(z, float)
    y = np.asarray(y, float)
    yhat = np.asarray(yhat, float)
    edges = np.asarray(edges, float)
    out: List[Tuple[int, float, float, int, float, float]] = []

    for i in range(len(edges) - 1):
        lo, hi = float(edges[i]), float(edges[i + 1])
        if i < len(edges) - 2:
            mask = (z >= lo) & (z < hi)
        else:
            mask = (z >= lo) & (z <= hi)
        mask = mask & np.isfinite(y)
        n = int(np.sum(mask))
        if n == 0:
            hat_tau = float("nan")
            loss = float("nan")
        else:
            y_bin = y[mask]
            yhat_bin = yhat[mask]
            # Coverage (for optional inspection / annotations)
            cov = np.mean(y_bin <= yhat_bin)
            hat_tau = float(cov)
            # Smoothed pinball loss
            r = y_bin - yhat_bin
            loss = float(np.mean(smooth_pinball_loss(r, tau=float(tau))))
        out.append((i, lo, hi, n, hat_tau, loss))
    return out


def _compute_bin_statistics_pinball(
    z: np.ndarray,
    y: np.ndarray,
    yhat: np.ndarray,
    edges: np.ndarray,
    tau: float,
) -> List[Tuple[int, float, float, int, float, float]]:
    """Backwards-compatible alias used by downstream ablation utilities."""
    return _compute_bin_pinball_stats(z=z, y=y, yhat=yhat, edges=edges, tau=tau)


def _macro_micro_from_bins(bins_rows: List[Tuple[int, float, float, int, float, float]]) -> Tuple[float, float]:
    vals = [(n, ae) for (_bid, _lo, _hi, n, _ht, ae) in bins_rows if n > 0 and np.isfinite(ae)]
    if not vals:
        return float("nan"), float("nan")
    n_vec = np.array([v[0] for v in vals], float)
    ae_vec = np.array([v[1] for v in vals], float)
    macro = float(np.mean(ae_vec))
    micro = float(np.sum(n_vec * ae_vec) / np.sum(n_vec))
    return macro, micro


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = _build_argparser().parse_args(argv)

    from skill_frontier.core.sigmoid import PERIOD4_BOUNDS, PERIOD4_SPLITS_SINGLE  # type: ignore
    from skill_frontier.evaluation.binning import create_equal_mass_bins
    from skill_frontier.evaluation.common import fit_sigmoid_predictor, interpolate_curve
    from skill_frontier.evaluation.metrics import (
        aggregate_task_metrics,
        write_bin_results,
        write_task_summary,
    )
    from skill_frontier.io.csv_utils import (
        compute_flops,
        detect_date_col,
        detect_oll_raw_tasks,
        maybe_scale_task_values,
        parse_year_month,
        read_csv_rows,
    )
    from skill_frontier.io.output_paths import EvaluationRunPaths

    rows, headers = read_csv_rows(args.csv)
    if not rows:
        raise SystemExit("No rows in CSV")
    date_col = detect_date_col(headers)
    if date_col is None:
        raise SystemExit("Could not detect date column for period split")
    tasks = detect_oll_raw_tasks(headers)
    if not tasks:
        raise SystemExit("No tasks auto-detected")

    # Build arrays for compute and period index.
    C_list: List[float] = []
    z_list: List[float] = []
    per_list: List[int] = []
    for r in rows:
        C = compute_flops(
            r,
            headers,
            logC_col=None,
            prod_cols=(args.compute_product_cols[0], args.compute_product_cols[1]),
            mult=float(args.compute_multiplier),
        )
        if not (np.isfinite(C) and C > 0.0):
            C_list.append(float("nan"))
            z_list.append(float("nan"))
            per_list.append(-1)
            continue
        z_list.append(float(math.log10(float(C))))
        C_list.append(float(C))
        ym = parse_year_month(r.get(date_col, "")) if date_col else None
        if ym is None:
            per_list.append(-1)
        else:
            y, m = ym
            per_list.append(int(assign_period_index_period4(int(y), int(m))))

    C_all = np.asarray(C_list, float)
    z_all = np.asarray(z_list, float)
    per_all = np.asarray(per_list, int)

    # Pre-load y arrays per task.
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
        y_mat[task] = maybe_scale_task_values(np.asarray(y_vals, float))

    # Build label->index map to interpret splits.
    label_to_idx = {}
    for i, bound in enumerate(PERIOD4_BOUNDS):
        if len(bound) == 3:
            label = bound[0]
        else:
            label = f"k{i+1}"
        label_to_idx[label] = i

    def _normalize_split(split):
        train_labels = split.get("train_labels", [])
        val_label = split.get("val_label", None)
        train_idx = [label_to_idx[l] for l in train_labels if l in label_to_idx]
        val_idx = [label_to_idx[val_label]] if val_label in label_to_idx else []
        return train_idx, val_idx

    os.makedirs(args.out_base, exist_ok=True)

    # Process each period split: k uses train period k and evaluates on next period (k+1).
    for k_idx, split in enumerate(PERIOD4_SPLITS_SINGLE, start=1):
        train_per, val_per = _normalize_split(split)
        if not train_per or not val_per:
            continue
        mask_tr_full = per_all == train_per[-1]
        mask_val = np.isin(per_all, val_per)

        out_dir = os.path.join(args.out_base, f"k{k_idx}")
        os.makedirs(out_dir, exist_ok=True)
        paths = EvaluationRunPaths(out_dir, legacy=False)
        summary_rows: List[Tuple[str, float, float, float, float]] = []

        for task in tasks:
            y_all = y_mat[task]

            # Train subset for fitting and evaluation.
            mtr = mask_tr_full & np.isfinite(z_all) & np.isfinite(y_all)
            z_tr = z_all[mtr]
            C_tr = C_all[mtr]
            y_tr = y_all[mtr]
            if z_tr.size < 3:
                continue

            edges_tr = create_equal_mass_bins(
                z_tr, int(max(1, args.bins)), int(max(1, args.min_bin_size))
            )
            if edges_tr.size < 2:
                continue

            xs_curve_tr, y_curve_tr = fit_sigmoid_predictor(
                C_tr,
                y_tr,
                tau=float(args.tau),
                frontier_fit_mode=str(args.frontier_fit_mode),
                bins_for_fit=int(args.bins),
                min_bin_size_for_fit=int(args.min_bin_size),
                bin_frontier_quantile=float(args.bin_frontier_quantile),
                bin_trim_fraction=float(args.bin_trim_fraction),
                bin_edges_for_fit=edges_tr,
            )
            yhat_tr = interpolate_curve(xs_curve_tr, y_curve_tr, C_tr)
            bins_tr = _compute_bin_pinball_stats(z_tr, y_tr, yhat_tr, edges_tr, tau=float(args.tau))
            loss_is_macro, loss_is_micro = _macro_micro_from_bins(bins_tr)
            write_bin_results(paths.get_bins_path(task, era="train"), bins_tr, era="train")

            # OOS evaluation (next period).
            mval = mask_val & np.isfinite(z_all) & np.isfinite(y_all)
            z_val = z_all[mval]
            C_val = C_all[mval]
            y_val = y_all[mval]

            # Build OOS bin edges (same as sigmoid_binned_mae.py semantics).
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

            era_label = "test_fixed" if args.oos_bins == "test_fixed" else "test_overlap"
            loss_oos_macro = float("nan")
            loss_oos_micro = float("nan")

            if edges_val.size >= 2 and z_val.size > 0:
                # Fit on period k (train) and evaluate on period k+1 (val).
                yhat_val_prev = interpolate_curve(xs_curve_tr, y_curve_tr, C_val)
                bins_prev = _compute_bin_pinball_stats(
                    z_val, y_val, yhat_val_prev, edges_val, tau=float(args.tau)
                )

                # Fit directly on period k+1 and evaluate on period k+1 (val).
                edges_val_fit = create_equal_mass_bins(
                    z_val, int(max(1, args.bins)), int(max(1, args.min_bin_size))
                )
                if edges_val_fit.size >= 2 and z_val.size >= 3:
                    xs_curve_val, y_curve_val = fit_sigmoid_predictor(
                        C_val,
                        y_val,
                        tau=float(args.tau),
                        frontier_fit_mode=str(args.frontier_fit_mode),
                        bins_for_fit=int(args.bins),
                        min_bin_size_for_fit=int(args.min_bin_size),
                        bin_frontier_quantile=float(args.bin_frontier_quantile),
                        bin_trim_fraction=float(args.bin_trim_fraction),
                        bin_edges_for_fit=edges_val_fit,
                    )
                    yhat_val_self = interpolate_curve(xs_curve_val, y_curve_val, C_val)
                    bins_self = _compute_bin_pinball_stats(
                        z_val, y_val, yhat_val_self, edges_val, tau=float(args.tau)
                    )

                    # Write delta: (prev-fit loss) - (self-fit loss), keeping hat_tau from prev-fit.
                    bins_delta: List[Tuple[int, float, float, int, float, float]] = []
                    for (bid, lo, hi, n, ht_prev, loss_prev), (_bid2, _lo2, _hi2, _n2, _ht2, loss_self) in zip(
                        bins_prev, bins_self
                    ):
                        delta = float(loss_prev - loss_self) if (np.isfinite(loss_prev) and np.isfinite(loss_self)) else float("nan")
                        bins_delta.append((int(bid), float(lo), float(hi), int(n), float(ht_prev), delta))
                else:
                    bins_delta = [(bid, lo, hi, n, ht, float("nan")) for (bid, lo, hi, n, ht, _loss) in bins_prev]

                write_bin_results(paths.get_bins_path(task, era=era_label), bins_delta, era=era_label)
                loss_oos_macro, loss_oos_micro = _macro_micro_from_bins(bins_delta)
            else:
                write_bin_results(paths.get_bins_path(task, era=era_label), [], era=era_label)

            write_task_summary(
                paths.get_summary_path(task),
                task=task,
                mae_is_macro=loss_is_macro,
                mae_is_micro=loss_is_micro,
                mae_oos_macro=loss_oos_macro,
                mae_oos_micro=loss_oos_micro,
                K_used=int(len(edges_tr) - 1),
                overlap_used=bool(edges_val.size >= 2),
            )
            summary_rows.append((task, loss_is_macro, loss_is_micro, loss_oos_macro, loss_oos_micro))

        aggregate_task_metrics(paths.get_aggregate_path(), summary_rows)


if __name__ == "__main__":
    main()

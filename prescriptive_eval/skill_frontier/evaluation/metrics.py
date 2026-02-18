"""Evaluation metrics calculation and CSV output utilities.

This module provides functions for writing evaluation results (bin statistics,
task summaries, and aggregate metrics) to CSV files.
"""

from __future__ import annotations

import csv as _csv
import os
from typing import List, Tuple

import numpy as np


def write_bin_results(
    path: str,
    rows: List[Tuple[int, float, float, int, float, float]],
    era: str,
) -> None:
    """Write per-bin coverage statistics to CSV.

    Args:
        path: Output CSV path
        rows: List of tuples (bin_id, z_lo, z_hi, n, hat_tau, abs_err)
        era: Label for the era/split (e.g., "train", "test_overlap", "test_fixed")
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="") as f:
        w = _csv.writer(f)
        w.writerow(["bin_id", "z_lo", "z_hi", "n", "hat_tau", "abs_err", "era"])
        for i, lo, hi, n, ht, ae in rows:
            w.writerow(
                [
                    i,
                    f"{lo:.12g}",
                    f"{hi:.12g}",
                    n,
                    f"{ht:.6g}" if np.isfinite(ht) else "",
                    f"{ae:.6g}" if np.isfinite(ae) else "",
                    era,
                ]
            )


def write_task_summary(
    path: str,
    task: str,
    mae_is_macro: float,
    mae_is_micro: float,
    mae_oos_macro: float,
    mae_oos_micro: float,
    K_used: int,
    overlap_used: bool,
) -> None:
    """Write per-task summary statistics to CSV.

    Args:
        path: Output CSV path
        task: Task name
        mae_is_macro: In-sample MAE (macro average over bins)
        mae_is_micro: In-sample MAE (micro average weighted by bin size)
        mae_oos_macro: Out-of-sample MAE (macro)
        mae_oos_micro: Out-of-sample MAE (micro)
        K_used: Number of bins used
        overlap_used: Whether overlap binning was used
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="") as f:
        w = _csv.writer(f)
        w.writerow(
            [
                "task",
                "MAE_IS_macro",
                "MAE_IS_micro",
                "MAE_OOS_macro",
                "MAE_OOS_micro",
                "K_used",
                "overlap_used",
            ]
        )
        w.writerow(
            [
                task,
                f"{mae_is_macro:.6g}",
                f"{mae_is_micro:.6g}",
                f"{mae_oos_macro:.6g}",
                f"{mae_oos_micro:.6g}",
                K_used,
                int(overlap_used),
            ]
        )


def aggregate_task_metrics(
    path: str, rows: List[Tuple[str, float, float, float, float]]
) -> None:
    """Aggregate metrics across multiple tasks and write to CSV.

    Args:
        path: Output CSV path
        rows: List of tuples (task, mae_is_macro, mae_is_micro, mae_oos_macro, mae_oos_micro)
    """
    if not rows:
        return
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    is_macro = np.array([r[1] for r in rows], float)
    is_micro = np.array([r[2] for r in rows], float)
    oos_macro = np.array([r[3] for r in rows], float)
    oos_micro = np.array([r[4] for r in rows], float)

    with open(path, "w", newline="") as f:
        w = _csv.writer(f)
        w.writerow(["metric", "value"])
        w.writerow(["IS_MAE_macro_mean", f"{np.nanmean(is_macro):.6g}"])
        w.writerow(["IS_MAE_micro_mean", f"{np.nanmean(is_micro):.6g}"])
        w.writerow(["OOS_MAE_macro_mean", f"{np.nanmean(oos_macro):.6g}"])
        w.writerow(["OOS_MAE_micro_mean", f"{np.nanmean(oos_micro):.6g}"])

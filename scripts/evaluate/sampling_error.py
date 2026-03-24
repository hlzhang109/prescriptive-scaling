#!/usr/bin/env python3
"""
Compare sampling-induced frontier error vs full-data fits.

Two complementary views:
  1) Evaluation error delta/ratio (IS/OOS Macro/Micro MAE) using existing
     evaluator outputs per k (e.g., period4 single_k). Compare per-task and
     aggregate summary between FULL and SAMPLED runs.
  2) Curve discrepancy (optional): mean absolute difference between y_hat curves
     on a common x-grid over the overlap of x-domains; and difference in the
     compute required to achieve a target accuracy (elbow at y*).

Inputs are folders that already exist from your pipeline:
  - --full_eval_base, --sampled_eval_base: roots containing k1..k3/<task>__summary.csv
  - --full_frontier_dir, --sampled_frontier_dir: roots containing
       smooth_frontier__<TASK>__k{1,2,3}.csv (optional for curve metrics)

Outputs:
  - A CSV with per-(k,task) metrics and deltas/ratios (and curve metrics if available)
"""

from __future__ import annotations

import argparse
import csv as _csv
import os
from typing import Dict, List, Optional, Tuple

import numpy as np


from skill_frontier.io.output_paths import sanitize_task_name  # type: ignore


def _task_legacy_filename(task: str) -> str:
    return task.replace("/", "_").replace("\\", "_")


def _summary_scan_dir(eval_k_dir: str) -> str:
    summaries_dir = os.path.join(eval_k_dir, "summaries")
    return summaries_dir if os.path.isdir(summaries_dir) else eval_k_dir


def _iter_summary_paths(eval_k_dir: str) -> List[str]:
    scan_dir = _summary_scan_dir(eval_k_dir)
    try:
        names = os.listdir(scan_dir)
    except Exception:
        return []
    out: List[str] = []
    for fn in names:
        if fn.endswith("__summary.csv") or fn.endswith("_summary.csv"):
            out.append(os.path.join(scan_dir, fn))
    out.sort()
    return out


def _get_summary_path(eval_k_dir: str, task: str) -> Optional[str]:
    task_clean = sanitize_task_name(task)
    task_legacy = _task_legacy_filename(task)
    candidates: List[str] = []
    summaries_dir = os.path.join(eval_k_dir, "summaries")
    if os.path.isdir(summaries_dir):
        candidates.extend(
            [
                os.path.join(summaries_dir, f"{task_clean}_summary.csv"),
                os.path.join(summaries_dir, f"{task_legacy}__summary.csv"),
                os.path.join(summaries_dir, f"{task}__summary.csv"),
            ]
        )
    candidates.extend(
        [
            os.path.join(eval_k_dir, f"{task_clean}_summary.csv"),
            os.path.join(eval_k_dir, f"{task_legacy}__summary.csv"),
            os.path.join(eval_k_dir, f"{task}__summary.csv"),
        ]
    )
    for path in candidates:
        if os.path.isfile(path):
            return path
    return None


def _find_curve_path(frontier_dir: str, task: str, k: int) -> Optional[str]:
    task_clean = sanitize_task_name(task)
    task_legacy = _task_legacy_filename(task)
    candidates = [
        os.path.join(frontier_dir, f"smooth_frontier__{task}__k{k}.csv"),
        os.path.join(frontier_dir, f"smooth_frontier__{task_legacy}__k{k}.csv"),
        os.path.join(frontier_dir, "curves", f"{task_clean}_k{k}.csv"),
        os.path.join(frontier_dir, "curves", f"{task_legacy}_k{k}.csv"),
    ]
    for path in candidates:
        if path and os.path.isfile(path):
            return path
    return None


def _detect_tasks(eval_k_dir: str) -> List[str]:
    tasks: List[str] = []
    for path in _iter_summary_paths(eval_k_dir):
        try:
            with open(path, "r", newline="") as f:
                r = _csv.DictReader(f)
                row = next(r)
            task = row.get("task", "")
        except Exception:
            continue
        if task:
            tasks.append(task)
    tasks.sort()
    return tasks


def _read_summary(path: str) -> Tuple[float, float, float, float]:
    with open(path, "r", newline="") as f:
        r = _csv.DictReader(f)
        row = next(r)
    is_macro = float(row["MAE_IS_macro"]) if row["MAE_IS_macro"] != "" else float("nan")
    is_micro = float(row["MAE_IS_micro"]) if row["MAE_IS_micro"] != "" else float("nan")
    oos_macro = float(row["MAE_OOS_macro"]) if row["MAE_OOS_macro"] != "" else float("nan")
    oos_micro = float(row["MAE_OOS_micro"]) if row["MAE_OOS_micro"] != "" else float("nan")
    return is_macro, is_micro, oos_macro, oos_micro


def _read_curve(path: str) -> Tuple[np.ndarray, np.ndarray]:
    xs: List[float] = []
    ys: List[float] = []
    with open(path, "r", newline="") as f:
        r = _csv.DictReader(f)
        for row in r:
            try:
                xs.append(float(row["x"]))
                ys.append(float(row["y_hat"]))
            except Exception:
                continue
    return np.asarray(xs, float), np.asarray(ys, float)


def _curve_L1(yf: np.ndarray, ys: np.ndarray) -> float:
    if yf.size == 0 or ys.size == 0:
        return float("nan")
    return float(np.mean(np.abs(yf - ys)))


def _curve_interp_diff(full_xy: Tuple[np.ndarray, np.ndarray], samp_xy: Tuple[np.ndarray, np.ndarray], grid: int = 256) -> float:
    xf, yf = full_xy
    xs, ys = samp_xy
    if xf.size == 0 or xs.size == 0:
        return float("nan")
    lo = max(float(np.min(xf)), float(np.min(xs)))
    hi = min(float(np.max(xf)), float(np.max(xs)))
    if not (np.isfinite(lo) and np.isfinite(hi)) or hi <= lo:
        return float("nan")
    z = np.logspace(np.log10(lo), np.log10(hi), num=grid)
    yf_i = np.interp(z, xf, yf, left=yf[0], right=yf[-1])
    ys_i = np.interp(z, xs, ys, left=ys[0], right=ys[-1])
    return _curve_L1(yf_i, ys_i)


def _x_at_y(x: np.ndarray, y: np.ndarray, target: float) -> float:
    if x.size == 0 or y.size == 0:
        return float("nan")
    yy = np.clip(y, 0.0, 1.0)
    # find smallest x s.t. y>=target via interpolation
    # monotone but clip anyway
    try:
        # invert by scanning
        idx = np.searchsorted(yy, target, side="left")
        if idx <= 0:
            return float(x[0])
        if idx >= yy.size:
            return float(x[-1])
        # linear interp
        x0, x1 = float(x[idx - 1]), float(x[idx])
        y0, y1 = float(yy[idx - 1]), float(yy[idx])
        if y1 == y0:
            return x1
        t = (target - y0) / (y1 - y0)
        return float(x0 + t * (x1 - x0))
    except Exception:
        return float("nan")


def main() -> None:
    ap = argparse.ArgumentParser(description="Compare sampling vs full frontier error")
    ap.add_argument("--full_eval_base", required=True, help="Full-data eval base with k*/<task>__summary.csv")
    ap.add_argument("--sampled_eval_base", required=True, help="Sampled eval base with k*/<task>__summary.csv")
    ap.add_argument("--full_frontier_dir", default=None, help="Optional: full frontiers dir with smooth_frontier__<TASK>__k*.csv")
    ap.add_argument("--sampled_frontier_dir", default=None, help="Optional: sampled frontiers dir with smooth_frontier__<TASK>__k*.csv")
    ap.add_argument("--y_target", type=float, default=0.80, help="Target accuracy to compute x@y elbow difference")
    ap.add_argument("--out_csv", required=True)
    args = ap.parse_args()

    rows_out: List[List[object]] = []
    header = [
        "k", "task",
        "IS_macro_full", "IS_macro_samp", "OOS_macro_full", "OOS_macro_samp",
        "d_OOS_macro", "r_OOS_macro",
        "IS_micro_full", "IS_micro_samp", "OOS_micro_full", "OOS_micro_samp",
        "d_OOS_micro", "r_OOS_micro",
        "curve_L1", "dx_at_y_target",
    ]

    for k in (1, 2, 3):
        full_k = os.path.join(args.full_eval_base, f"k{k}")
        samp_k = os.path.join(args.sampled_eval_base, f"k{k}")
        tasks = _detect_tasks(full_k)
        for task in tasks:
            f_sum = _get_summary_path(full_k, task)
            s_sum = _get_summary_path(samp_k, task)
            if not (f_sum and s_sum and os.path.isfile(f_sum) and os.path.isfile(s_sum)):
                continue
            ISm_f, ISu_f, OOSm_f, OOSu_f = _read_summary(f_sum)
            ISm_s, ISu_s, OOSm_s, OOSu_s = _read_summary(s_sum)
            d_OOSm = float(OOSm_s - OOSm_f) if np.isfinite(OOSm_s) and np.isfinite(OOSm_f) else float("nan")
            r_OOSm = float(OOSm_s / OOSm_f) if np.isfinite(OOSm_s) and np.isfinite(OOSm_f) and OOSm_f != 0 else float("nan")
            d_OOSu = float(OOSu_s - OOSu_f) if np.isfinite(OOSu_s) and np.isfinite(OOSu_f) else float("nan")
            r_OOSu = float(OOSu_s / OOSu_f) if np.isfinite(OOSu_s) and np.isfinite(OOSu_f) and OOSu_f != 0 else float("nan")

            curve_L1 = float("nan"); dx_at_y = float("nan")
            if args.full_frontier_dir and args.sampled_frontier_dir:
                f_curve = _find_curve_path(args.full_frontier_dir, task, k)
                s_curve = _find_curve_path(args.sampled_frontier_dir, task, k)
                if f_curve and s_curve and os.path.isfile(f_curve) and os.path.isfile(s_curve):
                    xf, yf = _read_curve(f_curve)
                    xs, ys = _read_curve(s_curve)
                    curve_L1 = _curve_interp_diff((xf, yf), (xs, ys), grid=256)
                    # elbow x-diff at y_target
                    x_f = _x_at_y(xf, yf, float(args.y_target))
                    x_s = _x_at_y(xs, ys, float(args.y_target))
                    if np.isfinite(x_f) and np.isfinite(x_s):
                        dx_at_y = float(x_s - x_f)

            rows_out.append([
                k, task,
                ISm_f, ISm_s, OOSm_f, OOSm_s,
                d_OOSm, r_OOSm,
                ISu_f, ISu_s, OOSu_f, OOSu_s,
                d_OOSu, r_OOSu,
                curve_L1, dx_at_y,
            ])

    os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)
    with open(args.out_csv, "w", newline="") as f:
        w = _csv.writer(f)
        w.writerow(header)
        for row in rows_out:
            w.writerow(row)
    print(f"[compare] wrote: {args.out_csv} ({len(rows_out)} rows)")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Sweep `balance_lambda` for the balanced I-optimal objective at a fixed alpha.

Goal
----
For each balance_lambda value, we:
  1) Construct period4/single_k budget manifests at alpha% (cost = #Params (B)).
  2) Evaluate coverage error (|hat_tau - tau|) in-sample on the full train pool
     using scripts/evaluate/sigmoid_binned_mae.py.
  3) Aggregate the in-sample coverage error across the six canonical OLL tasks.

Outputs
-------
Creates an output folder (default: outputs/sweeps_fullrange/lambda_sweep_alpha50/) containing:
  - per-lambda artifacts:
      lambda_<val>/p4_alpha50/k{1,2,3}/manifest__train.txt (+ val manifest)
      lambda_<val>/eval_p4_alpha50/k{1,2,3}/summaries/*_summary.csv
  - a top-level CSV summary and a simple PNG/PDF plot.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import subprocess
import sys
from typing import Dict, List, Sequence, Tuple

import numpy as np


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from skill_frontier.core.budget_design import FrontierParams, Weighting, design_budget_only  # type: ignore  # noqa: E402
from skill_frontier.core.sigmoid import PERIOD4_BOUNDS, PERIOD4_SPLITS_SINGLE  # type: ignore  # noqa: E402
from skill_frontier.io.csv_utils import (  # type: ignore  # noqa: E402
    compute_flops,
    detect_date_col,
    extract_model_id,
    parse_year_month,
    read_csv_rows,
)


TASKS_6 = [
    "IFEval Raw",
    "BBH Raw",
    "MATH Lvl 5 Raw",
    "GPQA Raw",
    "MMLU-PRO Raw",
    "MUSR Raw",
]


def _format_lambda_for_path(v: float) -> str:
    if v == 0.0:
        return "0"
    # Use scientific formatting but filesystem-friendly.
    s = f"{v:.3g}"
    s = s.replace("+", "").replace(".", "p")
    s = s.replace("e", "e")
    return s


def _read_task_is_mae(k_dir: str) -> Dict[str, float]:
    out: Dict[str, float] = {}
    summaries_dir = os.path.join(k_dir, "summaries")
    scan_dir = summaries_dir if os.path.isdir(summaries_dir) else k_dir
    if not os.path.isdir(scan_dir):
        return out
    for fn in os.listdir(scan_dir):
        if not (fn.endswith("__summary.csv") or fn.endswith("_summary.csv")):
            continue
        path = os.path.join(scan_dir, fn)
        try:
            with open(path, "r", newline="") as f:
                r = csv.DictReader(f)
                row = next(r)
        except Exception:
            continue
        task = str(row.get("task", "")).strip()
        try:
            is_macro = float(row.get("MAE_IS_macro", "nan"))
        except Exception:
            is_macro = float("nan")
        out[task] = is_macro
    return out


def _read_task_is_mae_period4(eval_base: str) -> Dict[str, float]:
    is_accum: Dict[str, List[float]] = {}
    for k in (1, 2, 3):
        vals = _read_task_is_mae(os.path.join(eval_base, f"k{k}"))
        for t, v in vals.items():
            is_accum.setdefault(t, []).append(v)
    out: Dict[str, float] = {}
    for t, arr in is_accum.items():
        a = np.asarray(arr, float)
        out[t] = float(np.nanmean(a)) if a.size else float("nan")
    return out


def _assign_period(year: int, month: int) -> int:
    for i, bound in enumerate(PERIOD4_BOUNDS):
        if len(bound) == 3:
            _, (y_lo, m_lo), (y_hi, m_hi) = bound
        else:
            y_lo, m_lo, y_hi, m_hi = bound
        if (year, month) >= (y_lo, m_lo) and (year, month) <= (y_hi, m_hi):
            return i
    return -1


def _design_period4_singlek(
    rows: List[dict],
    headers: List[str],
    *,
    compute_product_cols: Tuple[str, str],
    compute_multiplier: float,
    alpha_percent: float,
    out_base: str,
    objective: str,
    balance_lambda: float,
    exchange_passes: int,
    num_bins: int,
    min_bin_size: int,
) -> None:
    date_col = detect_date_col(headers)
    if date_col is None:
        raise RuntimeError("Could not detect date column in CSV for period4 split")

    theta0 = FrontierParams(y0=0.20, L=0.75, a=-9.0, b=1.0)
    w = Weighting(mode="constant", m=100)

    mids: List[str] = []
    z_all: List[float] = []
    size_all: List[float] = []
    per_all: List[int] = []

    for r in rows:
        mid = extract_model_id(r)
        if not mid:
            continue
        C = compute_flops(
            r,
            headers,
            logC_col=None,
            prod_cols=compute_product_cols,
            mult=float(compute_multiplier),
        )
        if not (np.isfinite(C) and C > 0):
            continue
        ym = parse_year_month(r.get(date_col, "")) if date_col else None
        if ym is None:
            continue
        pid = _assign_period(*ym)
        if pid < 0:
            continue
        size_raw = r.get("#Params (B)", None)
        try:
            sz = float(size_raw) if size_raw not in (None, "", "nan", "NaN") else float("nan")
        except Exception:
            sz = float("nan")
        mids.append(mid)
        z_all.append(float(np.log10(float(C))))
        size_all.append(float(sz))
        per_all.append(int(pid))

    z_arr = np.asarray(z_all, float)
    size_arr = np.asarray(size_all, float)
    per_arr = np.asarray(per_all, int)

    os.makedirs(out_base, exist_ok=True)

    label_to_idx: Dict[str, int] = {}
    for i, bound in enumerate(PERIOD4_BOUNDS):
        if len(bound) == 3:
            label = bound[0]
        else:
            label = f"k{i+1}"
        label_to_idx[label] = i

    for k_idx, split in enumerate(PERIOD4_SPLITS_SINGLE, start=1):
        out_dir = os.path.join(out_base, f"k{k_idx}")
        os.makedirs(out_dir, exist_ok=True)
        train_labels = split.get("train_labels", [])
        val_label = split.get("val_label", None)
        train_idx = [label_to_idx[l] for l in train_labels if l in label_to_idx]
        val_idx = [label_to_idx[val_label]] if val_label in label_to_idx else []

        mask_tr = np.isin(per_arr, np.asarray(train_idx, dtype=int))
        mask_val = np.isin(per_arr, np.asarray(val_idx, dtype=int))

        def _select(mask: np.ndarray) -> List[int]:
            if not mask.any():
                return []
            z = z_arr[mask]
            sz = size_arr[mask]
            sum_sz = float(np.sum(sz[np.isfinite(sz) & (sz > 0)]))
            U = (alpha_percent / 100.0) * sum_sz
            if U <= 0:
                return []
            idxs = design_budget_only(
                z,
                sz,
                U,
                theta0,
                weighting=w,
                seed_c=1.543,
                exchange_passes=int(max(0, exchange_passes)),
                objective=str(objective),
                balance_lambda=float(balance_lambda),
                num_bins=int(max(1, num_bins)),
                min_bin_size=int(max(1, min_bin_size)),
            )
            base_idx = np.nonzero(mask)[0]
            return [int(base_idx[i]) for i in idxs]

        sel_tr = _select(mask_tr)
        sel_val = _select(mask_val)

        for path, indices in (
            (os.path.join(out_dir, "manifest__train.txt"), sel_tr),
            (os.path.join(out_dir, "manifest__val.txt"), sel_val),
        ):
            with open(path, "w") as f:
                for i in indices:
                    f.write(f"{mids[i]}\n")


def _run_period4_eval(
    *,
    csv_path: str,
    compute_product_cols: Tuple[str, str],
    compute_multiplier: float,
    tasks: Sequence[str],
    bud_base: str,
    eval_base: str,
    frontier_fit_mode: str,
    bin_frontier_quantile: float,
    bin_trim_fraction: float,
) -> None:
    python_bin = sys.executable or "python3"
    os.makedirs(eval_base, exist_ok=True)
    cmd = [
        python_bin,
        os.path.join("scripts", "evaluate", "sigmoid_binned_mae.py"),
        "--csv",
        csv_path,
        "--compute_product_cols",
        compute_product_cols[0],
        compute_product_cols[1],
        "--compute_multiplier",
        str(compute_multiplier),
        "--split_mode",
        "period4",
        "--train_mode",
        "single_k",
        "--tau",
        "0.98",
        "--bins",
        "10",
        "--min_bin_size",
        "30",
        "--frontier_fit_mode",
        frontier_fit_mode,
        "--bin_frontier_quantile",
        str(bin_frontier_quantile),
        "--bin_trim_fraction",
        str(bin_trim_fraction),
        "--out_base",
        eval_base,
        "--manifest_base",
        bud_base,
        "--manifest_apply",
        "train_only",
        "--oos_bins",
        "test_fixed",
        "--tasks",
        *list(tasks),
    ]
    subprocess.check_call(cmd)


def _write_summary_csv(path: str, rows: List[dict], fieldnames: List[str]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def _plot_summary(
    out_dir: str,
    lambdas: List[float],
    avg_is: List[float],
    *,
    title: str,
    filename_base: str,
) -> None:
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception:
        return
    xs = np.asarray(lambdas, float)
    ys = np.asarray(avg_is, float)
    order = np.argsort(xs)
    xs, ys = xs[order], ys[order]

    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    ax.plot(xs, ys, "-o", color="0.2", linewidth=2.0, markersize=6)
    ax.set_xscale("symlog", linthresh=1e-6)
    ax.set_xlabel(r"balance $\lambda$")
    ax.set_ylabel("Avg. IS coverage error")
    ax.set_title(title)
    ax.grid(True, which="major", linestyle="-", alpha=0.35)
    ax.grid(True, which="minor", linestyle=":", alpha=0.25)
    fig.tight_layout()
    out_png = os.path.join(out_dir, filename_base + ".png")
    out_pdf = os.path.join(out_dir, filename_base + ".pdf")
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    fig.savefig(out_pdf, dpi=300, bbox_inches="tight")
    plt.close(fig)


def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Sweep balance_lambda at fixed alpha=50 and summarize IS coverage error.")
    ap.add_argument(
        "--csv",
        default=os.path.join("tables", "open_llm_leaderboard", "open_llm_leaderboard_with_tokens.csv"),
        help="Training CSV (OLL with_tokens schema).",
    )
    ap.add_argument("--compute_product_cols", nargs=2, default=["Pretraining tokens (T)", "#Params (B)"])
    ap.add_argument("--compute_multiplier", type=float, default=6.0)
    ap.add_argument("--alpha", type=float, default=50.0)
    ap.add_argument(
        "--lambdas",
        nargs="*",
        type=float,
        default=[0.0, 1e-3, 1e-2, 1e-1, 1.0],
        help="Values of balance_lambda to sweep (default: 0,1e-3,1e-2,1e-1,1).",
    )
    ap.add_argument(
        "--out_dir",
        default=os.path.join("outputs", "sweeps_fullrange", "lambda_sweep_alpha50"),
        help="Output directory (created if missing).",
    )
    ap.add_argument(
        "--objective",
        default="i_optimal_predvar_balanced",
        help="Design objective (default: i_optimal_predvar_balanced).",
    )
    ap.add_argument("--exchange_passes", type=int, default=2)
    ap.add_argument("--num_bins", type=int, default=10)
    ap.add_argument("--min_bin_size", type=int, default=1)
    ap.add_argument(
        "--frontier_fit_mode",
        choices=["quantile_per_point", "robust_bin_frontier"],
        default="quantile_per_point",
    )
    ap.add_argument("--bin_frontier_quantile", type=float, default=0.98)
    ap.add_argument("--bin_trim_fraction", type=float, default=0.01)
    return ap


def main(argv: Sequence[str] | None = None) -> None:
    args = build_argparser().parse_args(list(argv) if argv is not None else None)

    rows, headers = read_csv_rows(args.csv)
    if not rows:
        raise SystemExit(f"No rows in CSV: {args.csv}")

    # Ensure all 6 tasks exist in the CSV header.
    missing = [t for t in TASKS_6 if t not in headers]
    if missing:
        raise SystemExit(f"Missing required task columns in CSV: {missing}")

    alpha_int = int(round(float(args.alpha)))
    out_dir = str(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    results: List[dict] = []
    avg_list: List[float] = []
    lambda_list: List[float] = []

    for lam in list(args.lambdas):
        lam_f = float(lam)
        lam_tag = _format_lambda_for_path(lam_f)
        run_dir = os.path.join(out_dir, f"lambda_{lam_tag}")
        os.makedirs(run_dir, exist_ok=True)

        bud_base = os.path.join(run_dir, f"p4_alpha{alpha_int}")
        eval_base = os.path.join(run_dir, f"eval_p4_alpha{alpha_int}")

        _design_period4_singlek(
            rows,
            headers,
            compute_product_cols=(args.compute_product_cols[0], args.compute_product_cols[1]),
            compute_multiplier=float(args.compute_multiplier),
            alpha_percent=float(args.alpha),
            out_base=bud_base,
            objective=str(args.objective),
            balance_lambda=lam_f,
            exchange_passes=int(args.exchange_passes),
            num_bins=int(args.num_bins),
            min_bin_size=int(args.min_bin_size),
        )

        _run_period4_eval(
            csv_path=str(args.csv),
            compute_product_cols=(args.compute_product_cols[0], args.compute_product_cols[1]),
            compute_multiplier=float(args.compute_multiplier),
            tasks=TASKS_6,
            bud_base=bud_base,
            eval_base=eval_base,
            frontier_fit_mode=str(args.frontier_fit_mode),
            bin_frontier_quantile=float(args.bin_frontier_quantile),
            bin_trim_fraction=float(args.bin_trim_fraction),
        )

        per_task_is = _read_task_is_mae_period4(eval_base)
        is_vals = np.array([per_task_is.get(t, float("nan")) for t in TASKS_6], float)
        avg_is = float(np.nanmean(is_vals)) if np.isfinite(is_vals).any() else float("nan")

        row = {"alpha": alpha_int, "balance_lambda": lam_f, "avg_is_calibration_error": avg_is}
        for t in TASKS_6:
            row[f"is_{t}"] = per_task_is.get(t, float("nan"))
        results.append(row)
        lambda_list.append(lam_f)
        avg_list.append(avg_is)

    # Sort by lambda
    order = np.argsort(np.asarray(lambda_list, float))
    results_sorted = [results[i] for i in order.tolist()]

    fieldnames = ["alpha", "balance_lambda", "avg_is_calibration_error"] + [f"is_{t}" for t in TASKS_6]
    out_csv = os.path.join(out_dir, f"avg_is_calibration_error_vs_lambda_alpha{alpha_int}.csv")
    _write_summary_csv(out_csv, results_sorted, fieldnames)

    _plot_summary(
        out_dir,
        [lambda_list[i] for i in order.tolist()],
        [avg_list[i] for i in order.tolist()],
        title=f"Avg. IS coverage error vs balance_lambda (alpha={alpha_int})",
        filename_base=f"avg_is_calibration_error_vs_lambda_alpha{alpha_int}",
    )

    print(f"Wrote summary: {out_csv}")


if __name__ == "__main__":
    main()


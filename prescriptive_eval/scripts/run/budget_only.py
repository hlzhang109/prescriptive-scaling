#!/usr/bin/env python3
"""
Unified budget-only design tool for frontier model selection.

Supports three modes:
  - general: General-purpose design/fit/plot with flexible CSV input
  - pre2025_vs_2025: Budget-constrained selection for pre-2025 vs 2025 split
  - period4: Budget-constrained selection for 4-period splits (cumulative/single_k)

The budget-only design selects models to maximize information about the frontier
using only compute constraints (no accuracy data needed during selection).
"""

from __future__ import annotations

import argparse
import csv as _csv
import math
import os
import sys
from typing import List, Optional, Tuple

import numpy as np

# Add repo root to path
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
    compute_flops,
    parse_date,
    parse_year_month,
    extract_model_id,
)
from skill_frontier.io.manifest_utils import write_manifest_from_indices, write_manifest

# Import output path utilities
try:
    from skill_frontier.io.output_paths import get_budget_output_paths
except ImportError:
    # Fallback: create manifests directory manually
    def get_budget_output_paths(base_dir, mode="year_split", train_mode=None, legacy=False):
        if legacy:
            return {"root": base_dir, "manifests": base_dir, "plots": os.path.join(base_dir, "plots"), "curves": os.path.join(base_dir, "plots")}
        if mode == "no_split":
            root = base_dir
        elif mode == "year_split":
            root = os.path.join(base_dir, "budget", "year_split")
        elif mode == "period4":
            train_suffix = train_mode or "cumulative"
            root = os.path.join(base_dir, "budget", "period4", train_suffix)
        else:
            root = os.path.join(base_dir, "budget")
        return {"root": root, "manifests": os.path.join(root, "manifests"), "plots": os.path.join(root, "plots"), "curves": os.path.join(root, "curves")}


def _prepare_output_dirs(paths: dict, extra: Optional[List[str]] = None) -> None:
    """Create standard output subdirectories for manifests/plots/curves."""
    for key in ("root", "manifests", "plots", "curves"):
        if key in paths and paths[key]:
            os.makedirs(paths[key], exist_ok=True)
    for path in extra or []:
        os.makedirs(path, exist_ok=True)

# Import budget design from core
try:
    from skill_frontier.core.budget_design import (
        FrontierParams,
        Weighting,
        design_budget_only,
    )
except Exception:
    try:
        from scripts.budget_only_design import (
            FrontierParams,
            Weighting,
            design_budget_only,
        )
    except Exception:
        from budget_only_design import FrontierParams, Weighting, design_budget_only

# Import sigmoid fitter for fit/plot functionality
try:
    from skill_frontier.core.sigmoid import fit_sigmoid_frontier
except Exception:
    try:
        from scripts.smooth_single_skill_frontier import fit_sigmoid_frontier
    except Exception:
        from smooth_single_skill_frontier import fit_sigmoid_frontier


def read_candidates_general(
    path: str,
    logC_col: Optional[str],
    compute_product_cols: Optional[Tuple[str, str]],
    compute_multiplier: float,
    size_col: Optional[str] = None,
) -> Tuple[List[str], np.ndarray, np.ndarray]:
    """Read candidates CSV and return (model_ids, C_cost, z_x) for rows with usable data.

    C is always the compute cost (FLOPs) used for budgeting. z is log10(x) where x
    is either compute or an explicit size column if provided (e.g. '#Params (B)').
    """
    rows, headers = read_csv_rows(path)
    mids: List[str] = []
    C: List[float] = []  # compute cost
    z: List[float] = []  # design coordinate (log10 x)
    for r in rows:
        m = extract_model_id(r)
        if not m:
            continue
        c = compute_flops(
            r,
            headers,
            logC_col=logC_col,
            prod_cols=compute_product_cols,
            mult=compute_multiplier,
        )
        if not (np.isfinite(c) and c > 0.0):
            continue
        # Design x: size if available, otherwise compute
        if size_col and size_col in r and (r[size_col] not in (None, "", "nan", "NaN")):
            try:
                x = float(r[size_col])
            except Exception:
                x = float("nan")
        else:
            x = float(c)
        if not (np.isfinite(x) and x > 0.0):
            continue
        mids.append(m)
        C.append(c)
        z.append(float(math.log10(x)))
    return mids, np.asarray(C, float), np.asarray(z, float)


def design_main(args) -> None:
    """Run general-purpose budget-only design."""
    mids, C, z = read_candidates_general(
        args.csv,
        logC_col=args.logC_col,
        compute_product_cols=(
            tuple(args.compute_product_cols) if args.compute_product_cols else None
        ),
        compute_multiplier=float(args.compute_multiplier),
        size_col=args.size_col,
    )
    if z.size == 0:
        raise SystemExit("No usable candidates")

    theta0 = FrontierParams(
        y0=args.y0, L=args.L, a=args.a, b=max(args.b, 1e-6)
    )
    w = Weighting(mode=("binomial" if args.binomial else "constant"), m=args.m)
    idxs = design_budget_only(
        z,
        C,
        float(args.budget),
        theta0,
        weighting=w,
        seed_c=args.c,
        exchange_passes=int(max(0, getattr(args, "exchange_passes", 0))),
        objective=str(getattr(args, "objective", "d_optimal")),
        balance_lambda=float(getattr(args, "balance_lambda", 0.0)),
    )
    print(
        f"[design] selected={len(idxs)} total_cost={float(np.sum(C[idxs])):.6g} (budget={args.budget:g})"
    )
    write_manifest_from_indices(args.out_manifest, mids, idxs)
    print(f"[design] wrote manifest: {args.out_manifest}")


def fit_main(args) -> None:
    """Fit sigmoid frontier on evaluated subset for a single task."""
    rows, headers = read_csv_rows(args.subset_csv)
    x: List[float] = []
    y: List[float] = []
    for row in rows:
        c = compute_flops(
            row,
            headers,
            logC_col=args.logC_col,
            prod_cols=(
                tuple(args.compute_product_cols) if args.compute_product_cols else None
            ),
            mult=float(args.compute_multiplier),
        )
        if not (np.isfinite(c) and c > 0.0):
            continue
        v = row.get(args.task, None)
        try:
            yy = float(v) if v not in (None, "", "nan", "NaN") else float("nan")
        except Exception:
            yy = float("nan")
        if np.isfinite(yy):
            x.append(c)
            y.append(yy)
    if len(x) < 3:
        raise SystemExit("Need at least 3 (x,y) points to fit")

    xs, yhat = fit_sigmoid_frontier(
        np.asarray(x, float), np.asarray(y, float), tau=float(args.tau), use_log10_x=True
    )
    os.makedirs(os.path.dirname(args.out_curve) or ".", exist_ok=True)
    with open(args.out_curve, "w", newline="") as f:
        w = _csv.writer(f)
        w.writerow(["x", "y_hat", "task"])
        for xi, yi in zip(xs, yhat):
            w.writerow([float(xi), float(yi), args.task])
    print(f"[fit] wrote curve CSV: {args.out_curve}")


def plot_main(args) -> None:
    """Plot selected points and fitted sigmoid curves per task."""
    # Load manifest
    from skill_frontier.io.manifest_utils import read_manifest

    sel = read_manifest(args.manifest)

    # Read CSV
    rows, headers = read_csv_rows(args.csv)
    tasks = args.tasks or detect_oll_raw_tasks(headers)
    if not tasks:
        raise SystemExit("No tasks provided and none auto-detected.")

    # Prepare plotting
    try:
        import matplotlib.pyplot as plt
        import matplotlib as mpl
    except Exception:
        raise SystemExit("matplotlib is required for plotting")
    try:
        mpl.rcParams["font.family"] = "serif"
    except Exception:
        pass

    paths = get_budget_output_paths(args.out_dir, mode="no_split", legacy=False)
    points_dir = os.path.join(paths["root"], "selected_points")
    _prepare_output_dirs(paths, extra=[points_dir])

    # Iterate tasks
    for task in tasks:
        xs: List[float] = []
        ys: List[float] = []
        mids: List[str] = []
        for row in rows:
            m = extract_model_id(row)
            if not m or m not in sel:
                continue
            c = compute_flops(
                row,
                headers,
                logC_col=args.logC_col,
                prod_cols=(
                    tuple(args.compute_product_cols)
                    if args.compute_product_cols
                    else None
                ),
                mult=float(args.compute_multiplier),
            )
            if not (np.isfinite(c) and c > 0):
                continue
            v = row.get(task, None)
            try:
                y = float(v) if v not in (None, "", "nan", "NaN") else float("nan")
            except Exception:
                y = float("nan")
            if np.isfinite(y):
                xs.append(float(c))
                ys.append(float(y))
                mids.append(m)

        X = np.asarray(xs, float)
        Y = np.asarray(ys, float)
        if X.size < 3:
            print(f"[plot] skip {task}: only {X.size} points")
            continue

        # Fit
        xs_curve, y_curve = fit_sigmoid_frontier(
            X, Y, tau=float(args.tau), use_log10_x=True
        )

        # Write selected points CSV
        from skill_frontier.io.csv_utils import sanitize_name

        sel_csv = os.path.join(points_dir, f"selected_points__{sanitize_name(task)}.csv")
        with open(sel_csv, "w", newline="") as f:
            w = _csv.writer(f)
            w.writerow(["model", "x", "y", "task"])
            for m, xi, yi in zip(mids, X, Y):
                w.writerow([m, float(xi), float(yi), task])

        # Write curve CSV
        curve_csv = os.path.join(paths["curves"], f"budget_frontier__{sanitize_name(task)}.csv")
        with open(curve_csv, "w", newline="") as f:
            w = _csv.writer(f)
            w.writerow(["x", "y_hat", "task"])
            for xi, yi in zip(xs_curve, y_curve):
                w.writerow([float(xi), float(yi), task])

        # Plot
        plt.figure(figsize=(7.0, 4.6))
        plt.scatter(X, Y, s=12, alpha=0.20, color="#1f77b4", label="points")
        plt.plot(
            xs_curve,
            y_curve,
            color="#1f77b4",
            linewidth=2.4,
            label=f"Sigmoid τ={float(args.tau):.2f}",
        )
        plt.xscale("log")
        y_min = float(np.nanmin([np.nanmin(Y), np.nanmin(y_curve)]))
        y_max = float(np.nanmax([np.nanmax(Y), np.nanmax(y_curve)]))
        if not np.isfinite(y_min) or not np.isfinite(y_max) or y_min == y_max:
            y_min, y_max = 0.0, 1.0
        pad = 0.02 * max(1e-6, (y_max - y_min))
        plt.ylim(y_min - pad, y_max + pad)
        plt.xlabel("Pretraining Compute (ZFLOPs)", fontweight="bold", fontsize=15)
        plt.ylabel("Accuracy", fontweight="bold", fontsize=15)
        plt.title(f"Budget Frontier — {task}", fontweight="bold", fontsize=18)
        plt.legend(loc="best", fontsize=12)
        plt.tight_layout()
        from skill_frontier.io.csv_utils import sanitize_name

        base = os.path.join(paths["plots"], f"budget_frontier__{sanitize_name(task)}")
        plt.savefig(base + ".png", dpi=300, bbox_inches="tight")
        plt.savefig(base + ".pdf", dpi=300, bbox_inches="tight")
        plt.close()


def run_pre2025_vs_2025(args) -> None:
    """Budget-constrained selection for pre-2025 vs 2025 split."""
    rows, headers = read_csv_rows(args.csv)
    date_col = detect_date_col(headers)
    if date_col is None:
        raise SystemExit("Could not detect date column")

    # Split by year
    mids_pre: List[str] = []
    C_pre: List[float] = []
    z_pre: List[float] = []
    mids_2025: List[str] = []
    C_2025: List[float] = []
    z_2025: List[float] = []

    for r in rows:
        mid = extract_model_id(r)
        if not mid:
            continue
        c = compute_flops(
            r,
            headers,
            logC_col=args.logC_col,
            prod_cols=(
                tuple(args.compute_product_cols) if args.compute_product_cols else None
            ),
            mult=float(args.compute_multiplier),
        )
        if not (np.isfinite(c) and c > 0.0):
            continue
        # Design x: model size if provided, otherwise compute
        if args.size_col and args.size_col in r and (r[args.size_col] not in (None, "", "nan", "NaN")):
            try:
                x_val = float(r[args.size_col])
            except Exception:
                x_val = float("nan")
        else:
            x_val = float(c)
        if not (np.isfinite(x_val) and x_val > 0.0):
            continue
        yr = parse_date(r.get(date_col, ""))
        if yr is None:
            continue
        if yr < 2025:
            mids_pre.append(mid)
            C_pre.append(c)
            z_pre.append(float(math.log10(x_val)))
        elif yr == 2025:
            mids_2025.append(mid)
            C_2025.append(c)
            z_2025.append(float(math.log10(x_val)))

    C_pre_arr = np.asarray(C_pre, float)
    z_pre_arr = np.asarray(z_pre, float)
    C_2025_arr = np.asarray(C_2025, float)
    z_2025_arr = np.asarray(z_2025, float)

    # Compute budgets
    U_pre = args.budget_train_factor * float(np.mean(C_pre_arr)) if C_pre_arr.size else 0.0
    U_2025 = args.budget_val_factor * float(np.mean(C_2025_arr)) if C_2025_arr.size else 0.0

    print(f"[pre2025] candidates={len(mids_pre)} budget={U_pre:.6g}")
    print(f"[2025] candidates={len(mids_2025)} budget={U_2025:.6g}")

    paths = get_budget_output_paths(
        args.out_dir,
        mode="year_split",
        legacy=getattr(args, "legacy_output", False),
    )
    _prepare_output_dirs(paths)
    manifest_dir = paths["manifests"]

    # Design for each group
    theta0 = FrontierParams(y0=args.y0, L=args.L, a=args.a, b=max(args.b, 1e-6))
    w = Weighting(mode=("binomial" if args.binomial else "constant"), m=args.m)

    if z_pre_arr.size > 0 and U_pre > 0:
        idxs_pre = design_budget_only(
            z_pre_arr, C_pre_arr, U_pre, theta0, weighting=w, seed_c=args.c
        )
        write_manifest_from_indices(
            os.path.join(manifest_dir, "manifest__< 2025.txt"), mids_pre, idxs_pre
        )
        print(f"[pre2025] selected={len(idxs_pre)}")

    if z_2025_arr.size > 0 and U_2025 > 0:
        idxs_2025 = design_budget_only(
            z_2025_arr, C_2025_arr, U_2025, theta0, weighting=w, seed_c=args.c
        )
        write_manifest_from_indices(
            os.path.join(manifest_dir, "manifest__2025.txt"), mids_2025, idxs_2025
        )
        print(f"[2025] selected={len(idxs_2025)}")


def run_period4(args) -> None:
    """Budget-constrained selection for 4-period splits."""
    # Import period definitions
    try:
        from skill_frontier.core.sigmoid import PERIOD4_BOUNDS, PERIOD4_SPLITS
    except Exception:
        try:
            from scripts.smooth_single_skill_frontier import PERIOD4_BOUNDS, PERIOD4_SPLITS
        except Exception:
            from smooth_single_skill_frontier import PERIOD4_BOUNDS, PERIOD4_SPLITS

    def _assign_period(year: int, month: int) -> int:
        """Map (year, month) to period index using PERIOD4_BOUNDS.

        PERIOD4_BOUNDS supports two shapes:
          - (label, (y_lo, m_lo), (y_hi, m_hi))
          - (y_lo, m_lo, y_hi, m_hi)
        """
        for i, bound in enumerate(PERIOD4_BOUNDS):
            if len(bound) == 3:
                _, (y_lo, m_lo), (y_hi, m_hi) = bound
            else:
                y_lo, m_lo, y_hi, m_hi = bound
            if (year, month) >= (y_lo, m_lo) and (year, month) <= (y_hi, m_hi):
                return i
        return -1

    rows, headers = read_csv_rows(args.csv)
    date_col = detect_date_col(headers)
    if date_col is None:
        raise SystemExit("Could not detect date column")

    # Build arrays with period assignments
    mids: List[str] = []
    C_list: List[float] = []
    z_list: List[float] = []
    per_list: List[int] = []

    for r in rows:
        mid = extract_model_id(r)
        if not mid:
            continue
        c = compute_flops(
            r,
            headers,
            logC_col=args.logC_col,
            prod_cols=(
                tuple(args.compute_product_cols) if args.compute_product_cols else None
            ),
            mult=float(args.compute_multiplier),
        )
        if not (np.isfinite(c) and c > 0.0):
            continue
        # Design x: model size if provided, otherwise compute
        if args.size_col and args.size_col in r and (r[args.size_col] not in (None, "", "nan", "NaN")):
            try:
                x_val = float(r[args.size_col])
            except Exception:
                x_val = float("nan")
        else:
            x_val = float(c)
        if not (np.isfinite(x_val) and x_val > 0.0):
            continue
        ym = parse_year_month(r.get(date_col, ""))
        if ym is None:
            continue
        pid = _assign_period(*ym)
        if pid < 0:
            continue
        mids.append(mid)
        C_list.append(c)
        z_list.append(float(math.log10(x_val)))
        per_list.append(pid)

    C_all = np.asarray(C_list, float)
    z_all = np.asarray(z_list, float)
    per_all = np.asarray(per_list, int)

    theta0 = FrontierParams(y0=args.y0, L=args.L, a=args.a, b=max(args.b, 1e-6))
    w = Weighting(mode=("binomial" if args.binomial else "constant"), m=args.m)

    base_paths = get_budget_output_paths(
        args.out_base,
        mode="period4",
        train_mode=args.train_mode,
        legacy=getattr(args, "legacy_output", False),
    )
    _prepare_output_dirs(base_paths)

    # Build label->index map for PERIOD4_BOUNDS
    label_to_idx = {}
    for i, bound in enumerate(PERIOD4_BOUNDS):
        if len(bound) == 3:
            label = bound[0]
        else:
            label = f"k{i+1}"
        label_to_idx[label] = i

    # Choose splits (cumulative vs single_k) and normalize to index lists
    try:
        from skill_frontier.core.sigmoid import PERIOD4_SPLITS_SINGLE
    except Exception:
        try:
            from scripts.smooth_single_skill_frontier import PERIOD4_SPLITS_SINGLE
        except Exception:
            from smooth_single_skill_frontier import PERIOD4_SPLITS_SINGLE

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
        # Build train/val masks
        if args.train_mode == "cumulative":
            mask_tr = np.isin(per_all, train_per)
        else:  # single_k
            mask_tr = np.zeros_like(per_all, dtype=bool) if not train_per else (per_all == train_per[-1])
        mask_val = np.isin(per_all, val_per)

        mids_tr = [m for m, ok in zip(mids, mask_tr) if ok]
        C_tr = C_all[mask_tr]
        z_tr = z_all[mask_tr]
        mids_val = [m for m, ok in zip(mids, mask_val) if ok]
        C_val = C_all[mask_val]
        z_val = z_all[mask_val]

        # Compute budgets
        U_tr = args.budget_train_factor * float(np.mean(C_tr)) if C_tr.size else 0.0
        U_val = args.budget_val_factor * float(np.mean(C_val)) if C_val.size else 0.0

        print(f"[k{k_idx} train] candidates={len(mids_tr)} budget={U_tr:.6g}")
        print(f"[k{k_idx} val] candidates={len(mids_val)} budget={U_val:.6g}")

        k_paths = {
            "root": os.path.join(base_paths["root"], f"k{k_idx}"),
            "manifests": os.path.join(base_paths["manifests"], f"k{k_idx}"),
            "plots": os.path.join(base_paths["plots"], f"k{k_idx}"),
            "curves": os.path.join(base_paths["curves"], f"k{k_idx}"),
        }
        _prepare_output_dirs(k_paths)
        manifest_dir = k_paths["manifests"]

        if z_tr.size > 0 and U_tr > 0:
            idxs_tr = design_budget_only(z_tr, C_tr, U_tr, theta0, weighting=w, seed_c=args.c)
            # Map back to original mids_tr indices
            write_manifest(os.path.join(manifest_dir, "manifest__train.txt"), [mids_tr[i] for i in idxs_tr])
            print(f"[k{k_idx} train] selected={len(idxs_tr)}")

        if z_val.size > 0 and U_val > 0:
            idxs_val = design_budget_only(z_val, C_val, U_val, theta0, weighting=w, seed_c=args.c)
            write_manifest(os.path.join(manifest_dir, "manifest__val.txt"), [mids_val[i] for i in idxs_val])
            print(f"[k{k_idx} val] selected={len(idxs_val)}")


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Unified budget-only design tool")
    sub = p.add_subparsers(dest="cmd", required=True)

    # General-purpose design subcommand
    pd = sub.add_parser("design", help="General-purpose budget-only design")
    pd.add_argument("--csv", required=True)
    pd.add_argument("--logC_col", default=None)
    pd.add_argument("--compute_product_cols", nargs=2, default=None)
    pd.add_argument("--compute_multiplier", type=float, default=6.0)
    pd.add_argument("--size_col", default=None, help="Optional column to use as design x (e.g. '#Params (B)'); compute still used for budget cost")
    pd.add_argument("--budget", type=float, required=True)
    pd.add_argument("--out_manifest", required=True)
    pd.add_argument("--y0", type=float, default=0.20)
    pd.add_argument("--L", type=float, default=0.75)
    pd.add_argument("--a", type=float, default=-9.0)
    pd.add_argument("--b", type=float, default=1.0)
    pd.add_argument("--c", type=float, default=1.543)
    pd.add_argument("--binomial", action="store_true")
    pd.add_argument("--m", type=int, default=100)
    pd.add_argument("--exchange_passes", type=int, default=0)
    pd.add_argument(
        "--objective",
        choices=["d_optimal", "i_optimal_predvar", "i_optimal_predvar_balanced"],
        default="d_optimal",
        help="Design objective (default: d_optimal)",
    )
    pd.add_argument(
        "--balance_lambda",
        type=float,
        default=0.0,
        help="Bin-balance weight lambda for balanced I-optimal (default 0.0)",
    )

    # Fit subcommand
    pf = sub.add_parser("fit", help="Fit sigmoid on evaluated subset")
    pf.add_argument("--subset_csv", required=True)
    pf.add_argument("--task", required=True)
    pf.add_argument("--logC_col", default=None)
    pf.add_argument("--compute_product_cols", nargs=2, default=None)
    pf.add_argument("--compute_multiplier", type=float, default=6.0)
    pf.add_argument("--tau", type=float, default=0.98)
    pf.add_argument("--out_curve", required=True)

    # Plot subcommand
    pp = sub.add_parser("plot", help="Plot selected points and fitted curves")
    pp.add_argument("--csv", required=True)
    pp.add_argument("--manifest", required=True)
    pp.add_argument("--logC_col", default=None)
    pp.add_argument("--compute_product_cols", nargs=2, default=None)
    pp.add_argument("--compute_multiplier", type=float, default=6.0)
    pp.add_argument("--tasks", nargs="*", default=None)
    pp.add_argument("--tau", type=float, default=0.98)
    pp.add_argument("--out_dir", required=True)

    # Pre2025 vs 2025 split mode
    ps = sub.add_parser("pre2025_vs_2025", help="Budget design for pre-2025 vs 2025 split")
    ps.add_argument("--csv", required=True)
    ps.add_argument("--logC_col", default=None)
    ps.add_argument("--compute_product_cols", nargs=2, default=None)
    ps.add_argument("--compute_multiplier", type=float, default=6.0)
    ps.add_argument("--size_col", default=None, help="Optional column to use as design x (e.g. '#Params (B)'); compute still used for budget cost")
    ps.add_argument("--budget_train_factor", type=float, default=200.0)
    ps.add_argument("--budget_val_factor", type=float, default=200.0)
    ps.add_argument("--out_dir", default="outputs")
    ps.add_argument("--y0", type=float, default=0.20)
    ps.add_argument("--L", type=float, default=0.75)
    ps.add_argument("--a", type=float, default=-9.0)
    ps.add_argument("--b", type=float, default=1.0)
    ps.add_argument("--c", type=float, default=1.543)
    ps.add_argument("--binomial", action="store_true")
    ps.add_argument("--m", type=int, default=100)
    ps.add_argument("--legacy_output", action="store_true", help="Use legacy flat output structure")

    # Period4 mode
    p4 = sub.add_parser("period4", help="Budget design for 4-period splits")
    p4.add_argument("--csv", required=True)
    p4.add_argument("--logC_col", default=None)
    p4.add_argument("--compute_product_cols", nargs=2, default=None)
    p4.add_argument("--compute_multiplier", type=float, default=6.0)
    p4.add_argument("--size_col", default=None, help="Optional column to use as design x (e.g. '#Params (B)'); compute still used for budget cost")
    p4.add_argument("--budget_train_factor", type=float, default=200.0)
    p4.add_argument("--budget_val_factor", type=float, default=200.0)
    p4.add_argument("--train_mode", choices=["cumulative", "single_k"], default="cumulative")
    p4.add_argument("--out_base", default="outputs")
    p4.add_argument("--y0", type=float, default=0.20)
    p4.add_argument("--L", type=float, default=0.75)
    p4.add_argument("--a", type=float, default=-9.0)
    p4.add_argument("--b", type=float, default=1.0)
    p4.add_argument("--c", type=float, default=1.543)
    p4.add_argument("--binomial", action="store_true")
    p4.add_argument("--m", type=int, default=100)
    p4.add_argument("--legacy_output", action="store_true", help="Use legacy flat output structure")

    return p


def main(argv: Optional[List[str]] = None) -> None:
    args = build_argparser().parse_args(argv)
    if args.cmd == "design":
        design_main(args)
    elif args.cmd == "fit":
        fit_main(args)
    elif args.cmd == "plot":
        plot_main(args)
    elif args.cmd == "pre2025_vs_2025":
        run_pre2025_vs_2025(args)
    elif args.cmd == "period4":
        run_period4(args)


if __name__ == "__main__":
    main()

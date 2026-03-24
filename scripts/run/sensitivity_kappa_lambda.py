#!/usr/bin/env python3
"""κ/λ sensitivity analysis for sigmoid frontier fitting (period4 single-k).

This script runs three related analyses:
  1) Baseline: fit on P_k with default (κ=50, λ=lambda_default), evaluate on P_{k+1}
  2) CV-selected: inside each P_k, do grouped stratified train/val splits to select (κ, λ),
     refit on full P_k, evaluate on P_{k+1}

Evaluation metrics:
  - Smoothed pinball loss ρ̃_τ with a FIXED κ_eval=50 (comparable across κ_train)
  - Signed coverage error (τ_hat − τ) and absolute coverage error |τ_hat − τ|
    computed with the same train-defined equal-mass bins and overlap restriction used
    in the existing Section 4 period4 pipeline.

Outputs are written under:
  outputs/sensitivity/kappa_lambda_cv/
    - results_raw.csv
    - summary_by_task_period.csv
    - plots/*.png + *.pdf
"""

from __future__ import annotations

import argparse
import csv as _csv
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from skill_frontier.core.sigmoid import PERIOD4_BOUNDS  # type: ignore
from skill_frontier.core.period_utils import (  # type: ignore
    assign_period_index_period4,
    normalize_period4_splits_single,
)
from skill_frontier.evaluation.binning import create_equal_mass_bins  # type: ignore
from skill_frontier.evaluation.common import fit_sigmoid_predictor, interpolate_curve  # type: ignore
from skill_frontier.evaluation.sensitivity_kappa_lambda import (
    CalibrationSummary,
    calibration_summary,
    compute_overlap_edges,
    mask_in_edges,
    pinball_mean,
    select_best_hyperparams,
    split_train_val_group_stratified,
)
from skill_frontier.io.csv_utils import (
    compute_flops,
    detect_date_col,
    detect_oll_raw_tasks,
    maybe_scale_task_values,
    parse_year_month,
    read_csv_rows,
)


def _default_lambda_b() -> float:
    try:
        from scripts.smooth_single_skill_frontier import DEFAULT_LAMBDA_B
    except Exception:
        from smooth_single_skill_frontier import DEFAULT_LAMBDA_B  # type: ignore
    return float(DEFAULT_LAMBDA_B)


def _normalize_splits() -> List[Tuple[int, int]]:
    return normalize_period4_splits_single()


def _eval_on_split(
    *,
    C_train: np.ndarray,
    z_train: np.ndarray,
    edges_train: np.ndarray,
    xs_curve: np.ndarray,
    y_curve: np.ndarray,
    C_eval: np.ndarray,
    z_eval: np.ndarray,
    y_eval: np.ndarray,
    tau: float,
    kappa_eval: float,
) -> Tuple[float, CalibrationSummary, int]:
    edges_eval = compute_overlap_edges(edges_train, z_train, z_eval)
    if edges_eval.size < 2 or z_eval.size == 0:
        nan = float("nan")
        return nan, CalibrationSummary(nan, nan, nan, nan, 0), 0

    yhat = interpolate_curve(xs_curve, y_curve, C_eval)
    m = mask_in_edges(z_eval, edges_eval) & np.isfinite(y_eval) & np.isfinite(yhat)
    if not np.any(m):
        nan = float("nan")
        return nan, CalibrationSummary(nan, nan, nan, nan, 0), 0

    pin = pinball_mean(y_eval[m], yhat[m], tau=float(tau), kappa=float(kappa_eval))
    calib = calibration_summary(z_eval[m], y_eval[m], yhat[m], edges=edges_eval, tau=float(tau))
    return float(pin), calib, int(np.sum(m))


def _fit_curve(
    *,
    C: np.ndarray,
    y: np.ndarray,
    edges_train: np.ndarray,
    tau: float,
    frontier_fit_mode: str,
    bins: int,
    min_bin_size: int,
    bin_frontier_quantile: float,
    bin_trim_fraction: float,
    kappa_train: float,
    lambda_b: float,
) -> Tuple[np.ndarray, np.ndarray]:
    xs_curve, y_curve = fit_sigmoid_predictor(
        C,
        y,
        tau=float(tau),
        frontier_fit_mode=str(frontier_fit_mode),
        bins_for_fit=int(bins),
        min_bin_size_for_fit=int(min_bin_size),
        bin_frontier_quantile=float(bin_frontier_quantile),
        bin_trim_fraction=float(bin_trim_fraction),
        bin_edges_for_fit=np.asarray(edges_train, float),
        kappa_final=float(kappa_train),
        lambda_b=float(lambda_b),
    )
    return xs_curve, y_curve


def _run_task_period_job(job: Dict) -> List[Dict]:
    task = str(job["task"])
    k = int(job["k"])
    seeds = list(job["seeds"])

    tau = float(job["tau"])
    kappa_eval = float(job["kappa_eval"])
    bins = int(job["bins"])
    min_bin_size = int(job["min_bin_size"])
    cv_frac_train = float(job["cv_frac_train"])
    cv_z_bins = int(job["cv_z_bins"])
    frontier_fit_mode = str(job["frontier_fit_mode"])
    bin_frontier_quantile = float(job["bin_frontier_quantile"])
    bin_trim_fraction = float(job["bin_trim_fraction"])

    kappa_grid = [float(x) for x in job["kappa_grid"]]
    lambda_grid = [float(x) for x in job["lambda_grid"]]
    lambda_default = float(job["lambda_default"])

    C_pk = np.asarray(job["C_pk"], float)
    z_pk = np.asarray(job["z_pk"], float)
    y_pk = np.asarray(job["y_pk"], float)
    C_ok = np.asarray(job["C_ok"], float)
    z_ok = np.asarray(job["z_ok"], float)
    y_ok = np.asarray(job["y_ok"], float)

    if C_pk.size < 3 or C_ok.size < 1:
        return []

    edges_pk = create_equal_mass_bins(z_pk, int(max(1, bins)), int(max(1, min_bin_size)))
    if edges_pk.size < 2:
        return []

    # Baseline fit + OOS eval (independent of CV splits)
    xs_base, y_base = _fit_curve(
        C=C_pk,
        y=y_pk,
        edges_train=edges_pk,
        tau=tau,
        frontier_fit_mode=frontier_fit_mode,
        bins=bins,
        min_bin_size=min_bin_size,
        bin_frontier_quantile=bin_frontier_quantile,
        bin_trim_fraction=bin_trim_fraction,
        kappa_train=50.0,
        lambda_b=lambda_default,
    )
    baseline_oos_pin, baseline_oos_calib, baseline_n_overlap = _eval_on_split(
        C_train=C_pk,
        z_train=z_pk,
        edges_train=edges_pk,
        xs_curve=xs_base,
        y_curve=y_base,
        C_eval=C_ok,
        z_eval=z_ok,
        y_eval=y_ok,
        tau=tau,
        kappa_eval=kappa_eval,
    )

    out_rows: List[Dict] = []
    for seed in seeds:
        mask_tr, mask_val = split_train_val_group_stratified(
            group_ids=C_pk,
            z=z_pk,
            seed=int(seed),
            frac_train=float(cv_frac_train),
            n_bins=int(cv_z_bins),
        )
        if mask_tr.size != C_pk.size:
            continue
        if np.sum(mask_tr) < 3 or np.sum(mask_val) < 3:
            continue

        C_tr = C_pk[mask_tr]
        z_tr = z_pk[mask_tr]
        y_tr = y_pk[mask_tr]
        C_val = C_pk[mask_val]
        z_val = z_pk[mask_val]
        y_val = y_pk[mask_val]

        edges_tr = create_equal_mass_bins(z_tr, int(max(1, bins)), int(max(1, min_bin_size)))
        if edges_tr.size < 2:
            continue

        grid_candidates: List[Dict[str, float]] = []
        for kappa_train in kappa_grid:
            for lam in lambda_grid:
                xs, yy = _fit_curve(
                    C=C_tr,
                    y=y_tr,
                    edges_train=edges_tr,
                    tau=tau,
                    frontier_fit_mode=frontier_fit_mode,
                    bins=bins,
                    min_bin_size=min_bin_size,
                    bin_frontier_quantile=bin_frontier_quantile,
                    bin_trim_fraction=bin_trim_fraction,
                    kappa_train=kappa_train,
                    lambda_b=lam,
                )
                if xs.size == 0:
                    grid_candidates.append(
                        {
                            "kappa_train": float(kappa_train),
                            "lambda_b": float(lam),
                            "val_pinball": float("inf"),
                            "val_calib_abs": float("inf"),
                            "val_calib_signed": float("nan"),
                        }
                    )
                    continue

                val_pin, val_calib, _n_val_overlap = _eval_on_split(
                    C_train=C_tr,
                    z_train=z_tr,
                    edges_train=edges_tr,
                    xs_curve=xs,
                    y_curve=yy,
                    C_eval=C_val,
                    z_eval=z_val,
                    y_eval=y_val,
                    tau=tau,
                    kappa_eval=kappa_eval,
                )
                grid_candidates.append(
                    {
                        "kappa_train": float(kappa_train),
                        "lambda_b": float(lam),
                        "val_pinball": float(val_pin),
                        "val_calib_abs": float(val_calib.abs_micro),
                        "val_calib_signed": float(val_calib.signed_micro),
                    }
                )

        best = select_best_hyperparams(grid_candidates, tau=tau, prefer_kappa=50.0)
        if not best:
            continue

        kappa_sel = float(best["kappa_train"])
        lam_sel = float(best["lambda_b"])

        # Refit on full P_k using selected hyperparams, then evaluate OOS.
        xs_sel, y_sel = _fit_curve(
            C=C_pk,
            y=y_pk,
            edges_train=edges_pk,
            tau=tau,
            frontier_fit_mode=frontier_fit_mode,
            bins=bins,
            min_bin_size=min_bin_size,
            bin_frontier_quantile=bin_frontier_quantile,
            bin_trim_fraction=bin_trim_fraction,
            kappa_train=kappa_sel,
            lambda_b=lam_sel,
        )
        if xs_sel.size == 0:
            continue

        oos_pin, oos_calib, n_overlap = _eval_on_split(
            C_train=C_pk,
            z_train=z_pk,
            edges_train=edges_pk,
            xs_curve=xs_sel,
            y_curve=y_sel,
            C_eval=C_ok,
            z_eval=z_ok,
            y_eval=y_ok,
            tau=tau,
            kappa_eval=kappa_eval,
        )

        out_rows.append(
            {
                "task": task,
                "k": k,
                "seed": int(seed),
                "kappa_selected": kappa_sel,
                "lambda_selected": lam_sel,
                "val_pinball": float(best.get("val_pinball", float("nan"))),
                "val_calib_signed": float(best.get("val_calib_signed", float("nan"))),
                "val_calib_abs": float(best.get("val_calib_abs", float("nan"))),
                "oos_pinball": float(oos_pin),
                "oos_calib_signed": float(oos_calib.signed_micro),
                "oos_calib_abs": float(oos_calib.abs_micro),
                "baseline_oos_pinball": float(baseline_oos_pin),
                "baseline_oos_calib_signed": float(baseline_oos_calib.signed_micro),
                "baseline_oos_calib_abs": float(baseline_oos_calib.abs_micro),
                "n_pk": int(C_pk.size),
                "n_ok": int(C_ok.size),
                "n_overlap": int(n_overlap),
                "baseline_n_overlap": int(baseline_n_overlap),
            }
        )

    return out_rows


def _write_csv(path: str, rows: List[Dict], fieldnames: List[str]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="") as f:
        w = _csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fieldnames})


def _plot_outputs(
    out_dir: str,
    results: List[Dict],
    *,
    csv_path: str,
    compute_product_cols: Sequence[str],
    compute_multiplier: float,
    tau: float,
    bins: int,
    min_bin_size: int,
    frontier_fit_mode: str,
    bin_frontier_quantile: float,
    bin_trim_fraction: float,
    lambda_default: float,
) -> None:
    import pandas as pd
    import matplotlib.pyplot as plt  # type: ignore
    import matplotlib as mpl  # type: ignore

    try:
        from skill_frontier.plotting.configs.eval_sigmoid import CALIB_SUMMARY_RC_PARAMS  # type: ignore
    except Exception:
        CALIB_SUMMARY_RC_PARAMS = {}  # type: ignore

    plots_dir = os.path.join(out_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    df = pd.DataFrame(results)
    if df.empty:
        return

    with mpl.rc_context(CALIB_SUMMARY_RC_PARAMS):
        # Match baseline figure typography (see scripts/plot/plot_pinball_baselines.py)
        FIG_W = 5.0
        FIG_H = 4.0
        AXIS_LABEL_FONTSIZE = 18
        TICK_LABEL_FONTSIZE = 16
        TITLE_FONTSIZE = 18
        CELL_FONTSIZE = 16
        CMAP = mpl.cm.Blues
        X_LABEL_PERIOD_PK = r"\textbf{Period} $\mathcal{P}_k$"

        # Plot A: selected κ and log10(λ) heatmaps (task x k)
        agg = (
            df.groupby(["task", "k"], as_index=False)
            .agg(
                kappa_selected_median=("kappa_selected", "median"),
                lambda_selected_median=("lambda_selected", "median"),
            )
            .copy()
        )
        agg["log10_lambda_median"] = np.log10(np.maximum(agg["lambda_selected_median"].astype(float), 1e-16))
        tasks = sorted(df["task"].unique().tolist())
        task_labels = [str(t).replace(" Raw", "").replace(" raw", "") for t in tasks]
        ks = sorted(df["k"].unique().tolist())
        mat_kappa = np.full((len(tasks), len(ks)), np.nan, float)
        mat_loglam = np.full((len(tasks), len(ks)), np.nan, float)
        for _i, row in agg.iterrows():
            ti = tasks.index(row["task"])
            ki = ks.index(int(row["k"]))
            mat_kappa[ti, ki] = float(row["kappa_selected_median"])
            mat_loglam[ti, ki] = float(row["log10_lambda_median"])

        fig, axes = plt.subplots(1, 2, figsize=(2.0 * FIG_W, FIG_H), constrained_layout=True)
        ax0, ax1 = axes
        im0 = ax0.imshow(mat_kappa, aspect="auto", cmap=CMAP)
        ax0.set_title(r"\textbf{Selected $\kappa$ (median)}", fontsize=TITLE_FONTSIZE, pad=10)
        ax0.set_xticks(range(len(ks)))
        ax0.set_xticklabels([f"k={k}" for k in ks], rotation=0, fontsize=TICK_LABEL_FONTSIZE)
        ax0.set_yticks(range(len(tasks)))
        ax0.set_yticklabels(task_labels, fontsize=TICK_LABEL_FONTSIZE)
        ax0.set_xlabel(X_LABEL_PERIOD_PK, fontsize=AXIS_LABEL_FONTSIZE)
        ax0.set_ylabel(r"\textbf{Task}", fontsize=AXIS_LABEL_FONTSIZE)

        # gridlines for cell separation
        ax0.set_xticks(np.arange(len(ks) + 1) - 0.5, minor=True)
        ax0.set_yticks(np.arange(len(tasks) + 1) - 0.5, minor=True)
        ax0.grid(which="minor", color="white", linestyle="-", linewidth=1.5)
        ax0.tick_params(which="minor", size=0)

        fig.colorbar(im0, ax=ax0, fraction=0.046, pad=0.04)

        im1 = ax1.imshow(mat_loglam, aspect="auto", cmap=CMAP)
        ax1.set_title(
            r"\textbf{Selected $\log_{10}(\lambda)$ (median)}",
            fontsize=TITLE_FONTSIZE,
            pad=10,
        )
        ax1.set_xticks(range(len(ks)))
        ax1.set_xticklabels([f"k={k}" for k in ks], rotation=0, fontsize=TICK_LABEL_FONTSIZE)
        ax1.set_yticks(range(len(tasks)))
        ax1.set_yticklabels([])
        ax1.set_xlabel(X_LABEL_PERIOD_PK, fontsize=AXIS_LABEL_FONTSIZE)

        ax1.set_xticks(np.arange(len(ks) + 1) - 0.5, minor=True)
        ax1.set_yticks(np.arange(len(tasks) + 1) - 0.5, minor=True)
        ax1.grid(which="minor", color="white", linestyle="-", linewidth=1.5)
        ax1.tick_params(which="minor", size=0)
        fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)

        # Annotate with adaptive text color for readability.
        def _annotate(ax, mat, fmt: str) -> None:
            finite = mat[np.isfinite(mat)]
            if finite.size:
                vmin = float(np.nanmin(finite))
                vmax = float(np.nanmax(finite))
            else:
                vmin, vmax = 0.0, 1.0
            denom = (vmax - vmin) + 1e-12
            for (rr, cc), val in np.ndenumerate(mat):
                if not np.isfinite(val):
                    continue
                normed = (float(val) - vmin) / denom
                rgba = CMAP(normed)
                lum = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2]
                color = "black" if lum > 0.5 else "white"
                ax.text(
                    cc,
                    rr,
                    fmt.format(val),
                    ha="center",
                    va="center",
                    fontsize=CELL_FONTSIZE,
                    fontweight="bold",
                    color=color,
                )

        _annotate(ax0, mat_kappa, "{:.0f}")
        _annotate(ax1, mat_loglam, "{:.1f}")

        for ext in ("png", "pdf"):
            fig.savefig(
                os.path.join(plots_dir, f"selected_kappa_lambda.{ext}"),
                dpi=300,
                bbox_inches="tight",
            )
        plt.close(fig)

        # Plot B: baseline vs CV OOS diagnostics aggregated over tasks (mean across tasks)
        summ = (
            df.groupby(["task", "k"], as_index=False)
            .agg(
                oos_pinball=("oos_pinball", "mean"),
                baseline_oos_pinball=("baseline_oos_pinball", "mean"),
                oos_calib_abs=("oos_calib_abs", "mean"),
                baseline_oos_calib_abs=("baseline_oos_calib_abs", "mean"),
            )
            .copy()
        )
        by_k = (
            summ.groupby("k", as_index=False)
            .agg(
                oos_pinball=("oos_pinball", "mean"),
                baseline_oos_pinball=("baseline_oos_pinball", "mean"),
                oos_calib_abs=("oos_calib_abs", "mean"),
                baseline_oos_calib_abs=("baseline_oos_calib_abs", "mean"),
            )
            .sort_values("k")
        )
        ks = by_k["k"].astype(int).tolist()

        fig, axes = plt.subplots(1, 2, figsize=(12.0, 5.2), constrained_layout=True)
        ax0, ax1 = axes

        ax0.plot(ks, by_k["baseline_oos_pinball"], color="#1f77b4", linewidth=3, label="baseline")
        ax0.plot(ks, by_k["oos_pinball"], color="#1f77b4", linewidth=3, linestyle="--", label="CV-selected")
        ax0.set_title("OOS pinball loss (mean over tasks)", fontsize=18, fontweight="bold")
        ax0.set_xlabel("Period k", fontsize=16, fontweight="bold")
        ax0.set_ylabel(r"$\\tilde{\\rho}_{\\tau}$", fontsize=16, fontweight="bold")
        ax0.set_xticks(ks)
        ax0.grid(True, which="major", linestyle="-", linewidth=0.8, alpha=0.3)
        ax0.grid(True, which="minor", linestyle=":", linewidth=0.5, alpha=0.2)
        ax0.legend(fontsize=12, frameon=False)

        ax1.plot(ks, by_k["baseline_oos_calib_abs"], color="#1f77b4", linewidth=3, label="baseline")
        ax1.plot(ks, by_k["oos_calib_abs"], color="#1f77b4", linewidth=3, linestyle="--", label="CV-selected")
        ax1.set_title("OOS |coverage error| (mean over tasks)", fontsize=18, fontweight="bold")
        ax1.set_xlabel("Period k", fontsize=16, fontweight="bold")
        ax1.set_ylabel(r"$|\\hat{\\tau}-\\tau|$", fontsize=16, fontweight="bold")
        ax1.set_xticks(ks)
        ax1.grid(True, which="major", linestyle="-", linewidth=0.8, alpha=0.3)
        ax1.grid(True, which="minor", linestyle=":", linewidth=0.5, alpha=0.2)
        ax1.legend(fontsize=12, frameon=False)

        for ext in ("png", "pdf"):
            fig.savefig(os.path.join(plots_dir, f"oos_baseline_vs_cv.{ext}"), dpi=200)
        plt.close(fig)

        # Plot C: representative fitted curves (baseline vs CV-selected)
        rep_task = "MATH Lvl 5 Raw"
        rep_ks = [1, 3]
        if rep_task in df["task"].unique().tolist():
            # Use median selected hyperparams across seeds.
            rep_sel = (
                df[df["task"] == rep_task]
                .groupby("k", as_index=False)
                .agg(
                    kappa_selected=("kappa_selected", "median"),
                    lambda_selected=("lambda_selected", "median"),
                )
            )
            rep_map = {int(r["k"]): (float(r["kappa_selected"]), float(r["lambda_selected"])) for _, r in rep_sel.iterrows()}

            rows, headers = read_csv_rows(csv_path)
            date_col = detect_date_col(headers)
            if date_col is not None and rows:
                # Build arrays for the representative task only.
                C_list: List[float] = []
                z_list: List[float] = []
                per_list: List[int] = []
                y_list: List[float] = []
                for r in rows:
                    C = compute_flops(
                        r,
                        headers,
                        logC_col=None,
                        prod_cols=tuple(compute_product_cols),
                        mult=float(compute_multiplier),
                    )
                    ym = parse_year_month(r.get(date_col, ""))
                    v = r.get(rep_task, None)
                    try:
                        yv = float(v) if v not in (None, "", "nan", "NaN") else float("nan")
                    except Exception:
                        yv = float("nan")
                    if not (np.isfinite(C) and C > 0.0 and ym is not None):
                        C_list.append(float("nan"))
                        z_list.append(float("nan"))
                        per_list.append(-1)
                        y_list.append(yv)
                        continue
                    pid = assign_period_index_period4(int(ym[0]), int(ym[1]))
                    if pid < 0:
                        C_list.append(float("nan"))
                        z_list.append(float("nan"))
                        per_list.append(-1)
                        y_list.append(yv)
                        continue
                    C_list.append(float(C))
                    z_list.append(float(np.log10(C)))
                    per_list.append(int(pid))
                    y_list.append(yv)

                C_all = np.asarray(C_list, float)
                z_all = np.asarray(z_list, float)
                per_all = np.asarray(per_list, int)
                y_all = maybe_scale_task_values(np.asarray(y_list, float))

                splits = _normalize_splits()
                k_to_period = {i + 1: (p_tr, p_val) for i, (p_tr, p_val) in enumerate(splits)}

                fig, axes = plt.subplots(1, len(rep_ks), figsize=(14.0, 5.2), constrained_layout=True)
                if len(rep_ks) == 1:
                    axes = [axes]
                for ax, k in zip(axes, rep_ks):
                    if k not in k_to_period or k not in rep_map:
                        ax.axis("off")
                        continue
                    p_tr, _p_val = k_to_period[int(k)]
                    m_pk = (per_all == int(p_tr)) & np.isfinite(C_all) & (C_all > 0) & np.isfinite(y_all)
                    C_pk = C_all[m_pk]
                    z_pk = z_all[m_pk]
                    y_pk = y_all[m_pk]
                    if C_pk.size < 3:
                        ax.axis("off")
                        continue

                    edges_pk = create_equal_mass_bins(z_pk, int(max(1, bins)), int(max(1, min_bin_size)))
                    if edges_pk.size < 2:
                        ax.axis("off")
                        continue

                    kappa_sel, lam_sel = rep_map[int(k)]

                    xs_base, y_base = _fit_curve(
                        C=C_pk,
                        y=y_pk,
                        edges_train=edges_pk,
                        tau=float(tau),
                        frontier_fit_mode=str(frontier_fit_mode),
                        bins=int(bins),
                        min_bin_size=int(min_bin_size),
                        bin_frontier_quantile=float(bin_frontier_quantile),
                        bin_trim_fraction=float(bin_trim_fraction),
                        kappa_train=50.0,
                        lambda_b=float(lambda_default),
                    )
                    xs_cv, y_cv = _fit_curve(
                        C=C_pk,
                        y=y_pk,
                        edges_train=edges_pk,
                        tau=float(tau),
                        frontier_fit_mode=str(frontier_fit_mode),
                        bins=int(bins),
                        min_bin_size=int(min_bin_size),
                        bin_frontier_quantile=float(bin_frontier_quantile),
                        bin_trim_fraction=float(bin_trim_fraction),
                        kappa_train=float(kappa_sel),
                        lambda_b=float(lam_sel),
                    )

                    ax.scatter(C_pk, y_pk, s=18, alpha=0.20, color="#777777", label="points", zorder=2)
                    ax.plot(xs_base, y_base, color="firebrick", linewidth=3.2, label="baseline", zorder=3)
                    ax.plot(xs_cv, y_cv, color="firebrick", linewidth=3.2, linestyle="--", label="CV-selected", zorder=3)
                    ax.set_xscale("log")
                    ax.set_ylim(0.0, 1.0)
                    rep_task_title = str(rep_task).replace(" Raw", "").replace(" raw", "")
                    ax.set_title(f"{rep_task_title} (k={int(k)})", fontsize=16, fontweight="bold")
                    ax.set_xlabel("Pretraining Compute (FLOPs)", fontsize=16, fontweight="bold")
                    ax.set_ylabel("Accuracy", fontsize=16, fontweight="bold")
                    ax.grid(True, which="major", linestyle="-", linewidth=0.8, alpha=0.3)
                    ax.grid(True, which="minor", linestyle=":", linewidth=0.5, alpha=0.2)

                axes[0].legend(fontsize=12, frameon=False, loc="upper left")
                for ext in ("png", "pdf"):
                    fig.savefig(os.path.join(plots_dir, f"representative_fits_{rep_task.replace(' ', '_')}.{ext}"), dpi=200)
                plt.close(fig)


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="κ/λ sensitivity analysis for sigmoid frontier fitting.")
    default_csv = os.path.join(
        REPO_ROOT, "tables", "open_llm_leaderboard", "open_llm_leaderboard_with_tokens.csv"
    )
    p.add_argument("--csv", default=default_csv)
    p.add_argument(
        "--compute_product_cols",
        nargs=2,
        default=["Pretraining tokens (T)", "#Params (B)"],
        help="Two columns multiplied for compute proxy (default: tokens(T) and params(B)).",
    )
    p.add_argument("--compute_multiplier", type=float, default=6.0)
    p.add_argument("--tau", type=float, default=0.98)
    p.add_argument("--kappa_eval", type=float, default=50.0)
    p.add_argument("--bins", type=int, default=10)
    p.add_argument("--min_bin_size", type=int, default=30)
    p.add_argument(
        "--frontier_fit_mode",
        choices=["quantile_per_point", "robust_bin_frontier"],
        default="quantile_per_point",
    )
    p.add_argument("--bin_frontier_quantile", type=float, default=0.98)
    p.add_argument("--bin_trim_fraction", type=float, default=0.01)
    p.add_argument(
        "--kappa_grid",
        nargs="+",
        type=float,
        default=[20, 50, 100, 200, 1000],
        help="Grid of kappa values used during CV selection.",
    )
    p.add_argument(
        "--lambda_grid",
        nargs="+",
        type=float,
        default=[1e-4, 1e-3, 1e-2, 1e-1],
        help="Grid of lambda_b values used during CV selection.",
    )
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    p.add_argument("--cv_frac_train", type=float, default=0.5)
    p.add_argument("--cv_z_bins", type=int, default=10)
    p.add_argument("--jobs", type=int, default=max(1, os.cpu_count() or 1))
    p.add_argument(
        "--out_dir",
        default=os.path.join(REPO_ROOT, "outputs", "sensitivity", "kappa_lambda_cv"),
    )
    p.add_argument("--tasks", nargs="*", default=None)
    return p


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = _build_argparser().parse_args(argv)

    lambda_default = _default_lambda_b()
    lambda_grid = list(float(x) for x in args.lambda_grid)
    tol = 1e-15 * max(1.0, abs(lambda_default))
    if not any(abs(float(x) - lambda_default) <= tol for x in lambda_grid):
        lambda_grid.append(lambda_default)
        lambda_grid = sorted(set(lambda_grid))

    rows, headers = read_csv_rows(args.csv)
    if not rows:
        raise SystemExit(f"No rows in CSV: {args.csv}")
    date_col = detect_date_col(headers)
    if date_col is None:
        raise SystemExit("Could not detect date column for period split")

    tasks = args.tasks or detect_oll_raw_tasks(headers)
    if not tasks:
        raise SystemExit("No tasks provided or auto-detected")

    C_list: List[float] = []
    z_list: List[float] = []
    per_list: List[int] = []
    for r in rows:
        C = compute_flops(
            r,
            headers,
            logC_col=None,
            prod_cols=tuple(args.compute_product_cols),
            mult=float(args.compute_multiplier),
        )
        ym = parse_year_month(r.get(date_col, ""))
        if not (np.isfinite(C) and C > 0.0 and ym is not None):
            C_list.append(float("nan"))
            z_list.append(float("nan"))
            per_list.append(-1)
            continue
        pid = assign_period_index_period4(int(ym[0]), int(ym[1]))
        if pid < 0:
            C_list.append(float("nan"))
            z_list.append(float("nan"))
            per_list.append(-1)
            continue
        C_list.append(float(C))
        z_list.append(float(np.log10(C)))
        per_list.append(int(pid))

    C_all = np.asarray(C_list, float)
    z_all = np.asarray(z_list, float)
    per_all = np.asarray(per_list, int)

    # Read and scale task arrays once.
    y_mat: Dict[str, np.ndarray] = {}
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

    splits = _normalize_splits()  # list of (train_period_idx, val_period_idx) for k=1..3

    jobs: List[Dict] = []
    for k_idx, (p_tr, p_val) in enumerate(splits, start=1):
        mask_pk = per_all == int(p_tr)
        mask_ok = per_all == int(p_val)
        for task in tasks:
            y_all = y_mat[task]
            m_pk = mask_pk & np.isfinite(C_all) & (C_all > 0) & np.isfinite(y_all)
            m_ok = mask_ok & np.isfinite(C_all) & (C_all > 0) & np.isfinite(y_all)
            C_pk = C_all[m_pk]
            z_pk = z_all[m_pk]
            y_pk = y_all[m_pk]
            C_ok = C_all[m_ok]
            z_ok = z_all[m_ok]
            y_ok = y_all[m_ok]
            if C_pk.size < 3 or C_ok.size < 1:
                continue
            jobs.append(
                {
                    "task": task,
                    "k": int(k_idx),
                    "seeds": list(int(s) for s in args.seeds),
                    "tau": float(args.tau),
                    "kappa_eval": float(args.kappa_eval),
                    "bins": int(args.bins),
                    "min_bin_size": int(args.min_bin_size),
                    "cv_frac_train": float(args.cv_frac_train),
                    "cv_z_bins": int(args.cv_z_bins),
                    "frontier_fit_mode": str(args.frontier_fit_mode),
                    "bin_frontier_quantile": float(args.bin_frontier_quantile),
                    "bin_trim_fraction": float(args.bin_trim_fraction),
                    "kappa_grid": list(float(x) for x in args.kappa_grid),
                    "lambda_grid": list(float(x) for x in lambda_grid),
                    "lambda_default": float(lambda_default),
                    "C_pk": C_pk,
                    "z_pk": z_pk,
                    "y_pk": y_pk,
                    "C_ok": C_ok,
                    "z_ok": z_ok,
                    "y_ok": y_ok,
                }
            )

    os.makedirs(args.out_dir, exist_ok=True)
    print(
        f"[sensitivity] jobs={len(jobs)} tasks={len(tasks)} seeds={len(args.seeds)} "
        f"kappa_grid={list(args.kappa_grid)} lambda_grid={lambda_grid} lambda_default={lambda_default:g}"
    )

    results: List[Dict] = []
    if int(args.jobs) <= 1:
        for j in jobs:
            results.extend(_run_task_period_job(j))
    else:
        with ProcessPoolExecutor(max_workers=int(args.jobs)) as ex:
            futs = [ex.submit(_run_task_period_job, j) for j in jobs]
            for fut in as_completed(futs):
                res = fut.result()
                if res:
                    results.extend(res)

    if not results:
        raise SystemExit("No results computed (all tasks skipped?)")

    fieldnames = [
        "task",
        "k",
        "seed",
        "kappa_selected",
        "lambda_selected",
        "val_pinball",
        "val_calib_signed",
        "val_calib_abs",
        "oos_pinball",
        "oos_calib_signed",
        "oos_calib_abs",
        "baseline_oos_pinball",
        "baseline_oos_calib_signed",
        "baseline_oos_calib_abs",
        "n_pk",
        "n_ok",
        "n_overlap",
        "baseline_n_overlap",
    ]
    _write_csv(os.path.join(args.out_dir, "results_raw.csv"), results, fieldnames)

    # Summary table
    try:
        import pandas as pd

        df = pd.DataFrame(results)
        df["delta_oos_pinball"] = df["oos_pinball"] - df["baseline_oos_pinball"]
        df["delta_oos_calib_abs"] = df["oos_calib_abs"] - df["baseline_oos_calib_abs"]
        summ = (
            df.groupby(["task", "k"], as_index=False)
            .agg(
                n_seeds=("seed", "count"),
                kappa_selected_mean=("kappa_selected", "mean"),
                kappa_selected_std=("kappa_selected", "std"),
                lambda_selected_median=("lambda_selected", "median"),
                oos_pinball_mean=("oos_pinball", "mean"),
                oos_pinball_std=("oos_pinball", "std"),
                oos_calib_abs_mean=("oos_calib_abs", "mean"),
                oos_calib_abs_std=("oos_calib_abs", "std"),
                baseline_oos_pinball=("baseline_oos_pinball", "mean"),
                baseline_oos_calib_abs=("baseline_oos_calib_abs", "mean"),
                delta_oos_pinball_mean=("delta_oos_pinball", "mean"),
                delta_oos_calib_abs_mean=("delta_oos_calib_abs", "mean"),
            )
            .sort_values(["k", "task"])
        )
        summ.to_csv(os.path.join(args.out_dir, "summary_by_task_period.csv"), index=False)
    except Exception as e:
        print(f"[sensitivity] WARN: failed to write summary_by_task_period.csv: {e}")

    _plot_outputs(
        args.out_dir,
        results,
        csv_path=str(args.csv),
        compute_product_cols=tuple(args.compute_product_cols),
        compute_multiplier=float(args.compute_multiplier),
        tau=float(args.tau),
        bins=int(args.bins),
        min_bin_size=int(args.min_bin_size),
        frontier_fit_mode=str(args.frontier_fit_mode),
        bin_frontier_quantile=float(args.bin_frontier_quantile),
        bin_trim_fraction=float(args.bin_trim_fraction),
        lambda_default=float(lambda_default),
    )
    print(f"[sensitivity] wrote: {os.path.join(args.out_dir, 'results_raw.csv')}")
    print(f"[sensitivity] plots: {os.path.join(args.out_dir, 'plots')}")


if __name__ == "__main__":
    main()

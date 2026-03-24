#!/usr/bin/env python3
"""κ/λ sweep (plateau check) for sigmoid frontier fitting.

Sweeps κ_train × λ over a small grid on representative tasks and periods,
fits on P_k, evaluates OOS on P_{k+1} with:
  - fixed κ_eval=50 for smoothed pinball loss
  - train-defined bins clipped to the train–OOS z-overlap for calibration

Outputs are written under:
  outputs/sensitivity/kappa_lambda_sweep/
    - sweep_results.csv
    - plots/*.png + *.pdf
"""

from __future__ import annotations

import argparse
import csv as _csv
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, List, Optional, Sequence, Tuple

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
    calibration_summary,
    compute_overlap_edges,
    mask_in_edges,
    pinball_mean,
)
from skill_frontier.io.csv_utils import (
    compute_flops,
    detect_date_col,
    maybe_scale_task_values,
    parse_year_month,
    read_csv_rows,
)


def _normalize_splits() -> List[Tuple[int, int]]:
    return normalize_period4_splits_single()


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
    return fit_sigmoid_predictor(
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


def _eval_oos(
    *,
    C_train: np.ndarray,
    z_train: np.ndarray,
    edges_train: np.ndarray,
    xs_curve: np.ndarray,
    y_curve: np.ndarray,
    C_oos: np.ndarray,
    z_oos: np.ndarray,
    y_oos: np.ndarray,
    tau: float,
    kappa_eval: float,
) -> Tuple[float, float, float, int]:
    edges_oos = compute_overlap_edges(edges_train, z_train, z_oos)
    if edges_oos.size < 2:
        nan = float("nan")
        return nan, nan, nan, 0
    yhat = interpolate_curve(xs_curve, y_curve, C_oos)
    m = mask_in_edges(z_oos, edges_oos) & np.isfinite(y_oos) & np.isfinite(yhat)
    if not np.any(m):
        nan = float("nan")
        return nan, nan, nan, 0
    pin = float(pinball_mean(y_oos[m], yhat[m], tau=float(tau), kappa=float(kappa_eval)))
    calib = calibration_summary(z_oos[m], y_oos[m], yhat[m], edges=edges_oos, tau=float(tau))
    return pin, float(calib.signed_micro), float(calib.abs_micro), int(np.sum(m))


def _run_task_period_sweep(job: Dict) -> List[Dict]:
    task = str(job["task"])
    k = int(job["k"])
    tau = float(job["tau"])
    kappa_eval = float(job["kappa_eval"])
    bins = int(job["bins"])
    min_bin_size = int(job["min_bin_size"])
    frontier_fit_mode = str(job["frontier_fit_mode"])
    bin_frontier_quantile = float(job["bin_frontier_quantile"])
    bin_trim_fraction = float(job["bin_trim_fraction"])
    kappa_grid = [float(x) for x in job["kappa_grid"]]
    lambda_grid = [float(x) for x in job["lambda_grid"]]

    C_pk = np.asarray(job["C_pk"], float)
    z_pk = np.asarray(job["z_pk"], float)
    y_pk = np.asarray(job["y_pk"], float)
    C_ok = np.asarray(job["C_ok"], float)
    z_ok = np.asarray(job["z_ok"], float)
    y_ok = np.asarray(job["y_ok"], float)

    edges_pk = create_equal_mass_bins(z_pk, int(max(1, bins)), int(max(1, min_bin_size)))
    if edges_pk.size < 2:
        return []

    out: List[Dict] = []
    for kappa_train in kappa_grid:
        for lam in lambda_grid:
            xs, yy = _fit_curve(
                C=C_pk,
                y=y_pk,
                edges_train=edges_pk,
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
                continue
            oos_pin, oos_calib_signed, oos_calib_abs, n_overlap = _eval_oos(
                C_train=C_pk,
                z_train=z_pk,
                edges_train=edges_pk,
                xs_curve=xs,
                y_curve=yy,
                C_oos=C_ok,
                z_oos=z_ok,
                y_oos=y_ok,
                tau=tau,
                kappa_eval=kappa_eval,
            )
            out.append(
                {
                    "task": task,
                    "k": int(k),
                    "kappa_train": float(kappa_train),
                    "lambda_b": float(lam),
                    "oos_pinball": float(oos_pin),
                    "oos_calib_signed": float(oos_calib_signed),
                    "oos_calib_abs": float(oos_calib_abs),
                    "n_pk": int(C_pk.size),
                    "n_ok": int(C_ok.size),
                    "n_overlap": int(n_overlap),
                }
            )
    return out


def _write_csv(path: str, rows: List[Dict], fieldnames: List[str]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="") as f:
        w = _csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fieldnames})


def _plot_heatmaps(out_dir: str, rows: List[Dict]) -> None:
    import pandas as pd
    import matplotlib.pyplot as plt  # type: ignore
    import matplotlib as mpl  # type: ignore

    try:
        from skill_frontier.plotting.configs.eval_sigmoid import CALIB_SUMMARY_RC_PARAMS  # type: ignore
    except Exception:
        CALIB_SUMMARY_RC_PARAMS = {}  # type: ignore

    plots_dir = os.path.join(out_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    df = pd.DataFrame(rows)
    if df.empty:
        return

    # Match baseline figure typography (see scripts/plot/plot_pinball_baselines.py)
    FIG_W = 5.0
    FIG_H = 4.0
    AXIS_LABEL_FONTSIZE = 18
    TICK_LABEL_FONTSIZE = 16
    TITLE_FONTSIZE = 18
    CELL_FONTSIZE = 16
    CMAP = mpl.cm.Blues

    with mpl.rc_context(CALIB_SUMMARY_RC_PARAMS):
        for (task, k), dfg in df.groupby(["task", "k"]):
            kappa_vals = sorted(dfg["kappa_train"].unique().tolist())
            lambda_vals = sorted(dfg["lambda_b"].unique().tolist())
            mat_pin = np.full((len(kappa_vals), len(lambda_vals)), np.nan, float)
            mat_cal = np.full((len(kappa_vals), len(lambda_vals)), np.nan, float)
            for _, r in dfg.iterrows():
                i = kappa_vals.index(float(r["kappa_train"]))
                j = lambda_vals.index(float(r["lambda_b"]))
                mat_pin[i, j] = float(r["oos_pinball"])
                mat_cal[i, j] = float(r["oos_calib_abs"])

            lam_labels = []
            for lam in lambda_vals:
                if lam <= 0.0:
                    lam_labels.append("0")
                else:
                    lam_labels.append(f"{np.log10(lam):.0f}")

            for metric_name, mat in [("oos_pinball", mat_pin), ("oos_calib_abs", mat_cal)]:
                fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
                im = ax.imshow(mat, aspect="auto", cmap=CMAP)

                task_title = str(task).replace(" Raw", "").replace(" raw", "")
                metric_title = (
                    r"OOS pinball loss"
                    if metric_name == "oos_pinball"
                    else r"OOS coverage $|\hat{\tau}-\tau|$"
                )
                ax.set_title(
                    rf"\textbf{{{task_title} (k={int(k)}) - {metric_title}}}",
                    fontsize=TITLE_FONTSIZE,
                    pad=10,
                )
                ax.set_xlabel(r"\textbf{$\log_{10}(\lambda)$}", fontsize=AXIS_LABEL_FONTSIZE)
                ax.set_ylabel(r"\textbf{$\kappa_{\mathrm{train}}$}", fontsize=AXIS_LABEL_FONTSIZE)
                ax.set_xticks(range(len(lambda_vals)))
                ax.set_xticklabels(lam_labels, rotation=0, fontsize=TICK_LABEL_FONTSIZE)
                ax.set_yticks(range(len(kappa_vals)))
                ax.set_yticklabels([f"{int(v)}" for v in kappa_vals], fontsize=TICK_LABEL_FONTSIZE)
                ax.tick_params(axis="both", which="major", labelsize=TICK_LABEL_FONTSIZE)

                # Gridlines for cell separation.
                ax.set_xticks(np.arange(len(lambda_vals) + 1) - 0.5, minor=True)
                ax.set_yticks(np.arange(len(kappa_vals) + 1) - 0.5, minor=True)
                ax.grid(which="minor", color="white", linestyle="-", linewidth=1.5)
                ax.tick_params(which="minor", size=0)

                cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                cbar.ax.tick_params(labelsize=TICK_LABEL_FONTSIZE)

                finite = mat[np.isfinite(mat)]
                if finite.size:
                    vmin = float(np.nanmin(finite))
                    vmax = float(np.nanmax(finite))
                else:
                    vmin, vmax = 0.0, 1.0
                denom = (vmax - vmin) + 1e-12
                for (i, j), v in np.ndenumerate(mat):
                    if np.isfinite(v):
                        normed = (float(v) - vmin) / denom
                        rgba = CMAP(normed)
                        lum = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2]
                        color = "black" if lum > 0.5 else "white"
                        ax.text(
                            j,
                            i,
                            f"{v:.3g}",
                            ha="center",
                            va="center",
                            fontsize=CELL_FONTSIZE,
                            fontweight="bold",
                            color=color,
                        )

                safe_task = str(task).replace("/", "_").replace("\\", "_").replace(" ", "_")
                for ext in ("png", "pdf"):
                    fig.savefig(
                        os.path.join(plots_dir, f"heatmap_{safe_task}_k{int(k)}_{metric_name}.{ext}"),
                        dpi=300,
                        bbox_inches="tight",
                    )
                plt.close(fig)


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="κ/λ sweep for sigmoid frontier fitting.")
    default_csv = os.path.join(
        REPO_ROOT, "tables", "open_llm_leaderboard", "open_llm_leaderboard_with_tokens.csv"
    )
    p.add_argument("--csv", default=default_csv)
    p.add_argument(
        "--compute_product_cols",
        nargs=2,
        default=["Pretraining tokens (T)", "#Params (B)"],
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
    )
    p.add_argument(
        "--lambda_grid",
        nargs="+",
        type=float,
        default=[1e-4, 1e-3, 1e-2, 1e-1],
    )
    p.add_argument("--tasks", nargs="+", default=["MATH Lvl 5 Raw", "BBH Raw"])
    p.add_argument("--ks", nargs="+", type=int, default=[1, 3])
    p.add_argument("--jobs", type=int, default=max(1, os.cpu_count() or 1))
    p.add_argument(
        "--out_dir",
        default=os.path.join(REPO_ROOT, "outputs", "sensitivity", "kappa_lambda_sweep"),
    )
    return p


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = _build_argparser().parse_args(argv)
    rows, headers = read_csv_rows(args.csv)
    if not rows:
        raise SystemExit(f"No rows in CSV: {args.csv}")
    date_col = detect_date_col(headers)
    if date_col is None:
        raise SystemExit("Could not detect date column for period split")

    tasks = list(args.tasks)
    splits = _normalize_splits()
    k_to_period = {i + 1: (p_tr, p_val) for i, (p_tr, p_val) in enumerate(splits)}

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

    jobs: List[Dict] = []
    for task in tasks:
        y_all = y_mat[task]
        for k in args.ks:
            if int(k) not in k_to_period:
                continue
            p_tr, p_val = k_to_period[int(k)]
            mask_pk = per_all == int(p_tr)
            mask_ok = per_all == int(p_val)
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
                    "k": int(k),
                    "tau": float(args.tau),
                    "kappa_eval": float(args.kappa_eval),
                    "bins": int(args.bins),
                    "min_bin_size": int(args.min_bin_size),
                    "frontier_fit_mode": str(args.frontier_fit_mode),
                    "bin_frontier_quantile": float(args.bin_frontier_quantile),
                    "bin_trim_fraction": float(args.bin_trim_fraction),
                    "kappa_grid": list(float(x) for x in args.kappa_grid),
                    "lambda_grid": list(float(x) for x in args.lambda_grid),
                    "C_pk": C_pk,
                    "z_pk": z_pk,
                    "y_pk": y_pk,
                    "C_ok": C_ok,
                    "z_ok": z_ok,
                    "y_ok": y_ok,
                }
            )

    os.makedirs(args.out_dir, exist_ok=True)
    print(f"[sweep] jobs={len(jobs)} kappa_grid={list(args.kappa_grid)} lambda_grid={list(args.lambda_grid)}")

    out_rows: List[Dict] = []
    if int(args.jobs) <= 1:
        for j in jobs:
            out_rows.extend(_run_task_period_sweep(j))
    else:
        with ProcessPoolExecutor(max_workers=int(args.jobs)) as ex:
            futs = [ex.submit(_run_task_period_sweep, j) for j in jobs]
            for fut in as_completed(futs):
                res = fut.result()
                if res:
                    out_rows.extend(res)

    if not out_rows:
        raise SystemExit("No sweep results computed.")

    fieldnames = [
        "task",
        "k",
        "kappa_train",
        "lambda_b",
        "oos_pinball",
        "oos_calib_signed",
        "oos_calib_abs",
        "n_pk",
        "n_ok",
        "n_overlap",
    ]
    _write_csv(os.path.join(args.out_dir, "sweep_results.csv"), out_rows, fieldnames)
    _plot_heatmaps(args.out_dir, out_rows)
    print(f"[sweep] wrote: {os.path.join(args.out_dir, 'sweep_results.csv')}")
    print(f"[sweep] plots: {os.path.join(args.out_dir, 'plots')}")


if __name__ == "__main__":
    main()

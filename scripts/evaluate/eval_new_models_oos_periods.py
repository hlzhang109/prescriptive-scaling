#!/usr/bin/env python3
"""
Evaluate sigmoid frontiers on OLL (old) periods with OOS validation on newly-evaluated models.

Goal
  - Use the Open LLM Leaderboard (OLL) data to fit sigmoid frontiers on period P_k (old models).
  - Evaluate those frontiers on newly-evaluated models under one of the supported mappings:
      * same_period: old P_k -> new P_k
      * next_period: old P_k -> new P_{k+1} with an extra new-only P5 after old P4
      * p4_to_p5: old P4 -> new P5 only

Outputs
  Writes a period4/single_k-like directory with k1..k4 subfolders, each containing:
    - bins/<task>_bins_train.csv
    - bins/<task>_bins_test_overlap.csv (or *_bins_test_fixed.csv)
    - summaries/<task>_summary.csv
    - aggregate/summary_over_tasks.csv

This output structure is compatible with scripts/plot/eval_sigmoid.py via --period4_singlek_base.
"""

from __future__ import annotations

import argparse
import math
import os
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

try:
    import pandas as pd  # type: ignore
except Exception as e:  # pragma: no cover
    raise SystemExit("This script requires pandas (already used elsewhere in this repo).") from e

# Repo imports
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from skill_frontier.core.sigmoid import PERIOD4_BOUNDS  # type: ignore
from skill_frontier.evaluation.binning import create_equal_mass_bins, compute_bin_statistics  # type: ignore
from skill_frontier.evaluation.common import fit_sigmoid_predictor, interpolate_curve  # type: ignore
from skill_frontier.evaluation.metrics import write_bin_results, write_task_summary, aggregate_task_metrics  # type: ignore
from skill_frontier.io.csv_utils import parse_year_month  # type: ignore
from skill_frontier.io.output_paths import EvaluationRunPaths  # type: ignore


MAIN_TASK_MAP_NEW_TO_OLD: Dict[str, str] = {
    # Canonical OLL Raw column -> new leaderboard column
    "IFEval Raw": "leaderboard_ifeval_inst_level_strict_acc_none",
    "BBH Raw": "leaderboard_bbh_acc_norm_none",
    "MATH Lvl 5 Raw": "leaderboard_math_hard_exact_match_none",
    "GPQA Raw": "leaderboard_gpqa_acc_norm_none",
    "MUSR Raw": "leaderboard_musr_acc_norm_none",
    "MMLU-PRO Raw": "leaderboard_mmlu_pro_acc_none",
}


def _assign_period_index(bounds: Sequence[Tuple[str, Tuple[int, int], Tuple[int, int]]], ym: Tuple[int, int]) -> int:
    y, m = ym
    for idx, (_lab, (y_lo, m_lo), (y_hi, m_hi)) in enumerate(bounds):
        if (y, m) >= (y_lo, m_lo) and (y, m) <= (y_hi, m_hi):
            return idx
    return -1


def _parse_year_month_series(values: Iterable[object]) -> List[Optional[Tuple[int, int]]]:
    out: List[Optional[Tuple[int, int]]] = []
    for v in values:
        out.append(parse_year_month("" if v is None else str(v)))
    return out


def _pinball_loss(r: np.ndarray, tau: float, k_smooth: float = 50.0) -> np.ndarray:
    r = np.asarray(r, float)
    return (np.logaddexp(0.0, k_smooth * r) / k_smooth) + (tau - 1.0) * r


def _compute_bin_statistics_pinball(
    z: np.ndarray,
    y: np.ndarray,
    yhat: np.ndarray,
    edges: np.ndarray,
    tau: float,
) -> List[Tuple[int, float, float, int, float, float]]:
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
        mask = mask & np.isfinite(y) & np.isfinite(yhat)
        n = int(np.sum(mask))
        if n == 0:
            hat_tau = float("nan")
            loss = float("nan")
        else:
            y_bin = y[mask]
            yhat_bin = yhat[mask]
            hat_tau = float(np.mean(y_bin <= yhat_bin))
            r = y_bin - yhat_bin
            loss = float(np.mean(_pinball_loss(r, tau=tau)))
        out.append((i, lo, hi, n, hat_tau, loss))
    return out


def _edges_overlap(edges_tr: np.ndarray, z_tr: np.ndarray, z_val: np.ndarray) -> np.ndarray:
    if z_tr.size == 0 or edges_tr.size < 2:
        return np.array([], float)
    if z_val.size >= 1:
        z_lo = float(max(np.min(z_tr), np.min(z_val)))
        z_hi = float(min(np.max(z_tr), np.max(z_val)))
    else:
        z_lo = float(np.min(z_tr))
        z_hi = float(np.max(z_tr))
    e = np.asarray(edges_tr, float).copy()
    e[0] = max(e[0], z_lo)
    e[-1] = min(e[-1], z_hi)
    keep = [i for i in range(len(e) - 1) if e[i + 1] > e[i] + 1e-12]
    if not keep:
        return np.array([], float)
    return np.array([e[i] for i in keep] + [e[keep[-1] + 1]], float)


def _build_datasets(
    oll_csv: str,
    new_eval_csv: str,
    new_compute_csv: str,
    top_models_csv: str,
    include_bbh_subtasks: bool,
) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    """Return (df_old, df_new, tasks) with tasks using the old OLL column names."""
    df_oll = pd.read_csv(oll_csv)
    df_metrics = pd.read_csv(new_eval_csv)
    df_compute = pd.read_csv(new_compute_csv, usecols=["model_id", "pretrain_compute_zflops"])
    df_new = df_metrics.merge(
        df_compute,
        on="model_id",
        how="left",
        validate="one_to_one",
    )
    df_meta = pd.read_csv(top_models_csv)

    # Join metadata for last_modified and (optional) parameter count.
    df_new = df_new.merge(
        df_meta[["model_id", "last_modified", "num_params_B"]],
        on="model_id",
        how="left",
        validate="one_to_one",
    )

    # Identify BBH subtask columns from the OLL schema.
    bbh_subtasks_oll: List[str] = []
    if include_bbh_subtasks:
        bbh_subtasks_oll = [
            c for c in df_oll.columns if c.startswith("leaderboard_bbh_")
        ]
        # Old CSV includes exactly 24 subtasks; keep deterministic ordering.
        bbh_subtasks_oll = sorted(bbh_subtasks_oll)

    # Build task list (main tasks present in both datasets, plus optional subtasks).
    main_tasks = [t for t in MAIN_TASK_MAP_NEW_TO_OLD.keys() if MAIN_TASK_MAP_NEW_TO_OLD[t] in df_new.columns and t in df_oll.columns]
    tasks = list(main_tasks) + list(bbh_subtasks_oll)

    # Old dataset (OLL)
    df_old = pd.DataFrame()
    # Use a unified model id column for downstream selection
    if "Model sha" in df_oll.columns:
        df_old["model"] = df_oll["Model sha"].astype(str)
    elif "model" in df_oll.columns:
        df_old["model"] = df_oll["model"].astype(str)
    else:
        df_old["model"] = df_oll.get("Model", "").astype(str)
    df_old["source"] = "oll"
    df_old["mapped_base_model"] = df_oll.get("Identified base model", np.nan)
    df_old["date"] = df_oll.get("Upload To Hub Date", np.nan)
    df_old["compute_zflops"] = (
        6.0 * pd.to_numeric(df_oll.get("Pretraining tokens (T)", np.nan), errors="coerce")
        * pd.to_numeric(df_oll.get("#Params (B)", np.nan), errors="coerce")
    )
    for t in tasks:
        df_old[t] = pd.to_numeric(df_oll.get(t, np.nan), errors="coerce")

    # New dataset (new eval)
    df_new_out = pd.DataFrame()
    df_new_out["model"] = df_new["model_id"].astype(str)
    df_new_out["source"] = "new_eval"
    df_new_out["mapped_base_model"] = df_new.get("mapped_base_model", np.nan)
    df_new_out["date"] = df_new.get("last_modified", np.nan)
    df_new_out["compute_zflops"] = pd.to_numeric(df_new.get("pretrain_compute_zflops", np.nan), errors="coerce")
    for old_name, new_col in MAIN_TASK_MAP_NEW_TO_OLD.items():
        if old_name in tasks:
            df_new_out[old_name] = pd.to_numeric(df_new.get(new_col, np.nan), errors="coerce")
    for sub in bbh_subtasks_oll:
        new_col = f"{sub}_acc_norm_none"
        if new_col in df_new.columns:
            df_new_out[sub] = pd.to_numeric(df_new.get(new_col, np.nan), errors="coerce")
        else:
            df_new_out[sub] = np.nan

    return df_old, df_new_out, tasks


def _assign_periods_new(
    df_new: pd.DataFrame,
    period4_bounds: Sequence[Tuple[str, Tuple[int, int], Tuple[int, int]]],
) -> np.ndarray:
    """Assign P1..P5 for new models, where P5 is strictly after the end of the last P4 bound."""
    ym_list = _parse_year_month_series(df_new["date"].tolist())
    # End of original P4
    _p4_end = period4_bounds[-1][2]
    pids: List[int] = []
    for ym in ym_list:
        if ym is None:
            pids.append(-1)
            continue
        if ym > _p4_end:
            pids.append(4)  # P5
            continue
        pids.append(_assign_period_index(period4_bounds, ym))
    return np.asarray(pids, int)


def _assign_periods_old(
    df_old: pd.DataFrame,
    period4_bounds: Sequence[Tuple[str, Tuple[int, int], Tuple[int, int]]],
) -> np.ndarray:
    ym_list = _parse_year_month_series(df_old["date"].tolist())
    pids: List[int] = []
    for ym in ym_list:
        if ym is None:
            pids.append(-1)
        else:
            pids.append(_assign_period_index(period4_bounds, ym))
    return np.asarray(pids, int)


def run_eval(
    df_old: pd.DataFrame,
    df_new: pd.DataFrame,
    tasks: Sequence[str],
    out_base: str,
    metric: str,
    tau: float,
    bins: int,
    min_bin_size: int,
    frontier_fit_mode: str,
    bin_frontier_quantile: float,
    bin_trim_fraction: float,
    oos_bins: str,
    eval_mode: str,
) -> None:
    if metric not in ("calibration", "pinball"):
        raise ValueError(f"Unknown metric: {metric}")
    if oos_bins not in ("train_overlap", "test_fixed"):
        raise ValueError(f"Unknown oos_bins: {oos_bins}")
    if eval_mode not in ("same_period", "next_period", "p4_to_p5"):
        raise ValueError(f"Unknown eval_mode: {eval_mode}")

    pid_old = _assign_periods_old(df_old, PERIOD4_BOUNDS)
    pid_new = _assign_periods_new(df_new, PERIOD4_BOUNDS)

    # Define the evaluation mapping as (k_label, train_pid_old, val_pid_new).
    # Period ids: 0..3 correspond to old/new P1..P4 (PERIOD4_BOUNDS).
    # New-only P5 is represented as pid=4 (strictly after P4 end).
    if eval_mode == "same_period":
        split_specs = [(1, 0, 0), (2, 1, 1), (3, 2, 2), (4, 3, 3)]
    elif eval_mode == "next_period":
        split_specs = [(1, 0, 1), (2, 1, 2), (3, 2, 3), (4, 3, 4)]
    else:  # p4_to_p5
        split_specs = [(4, 3, 4)]

    # Precompute compute/z arrays (filter per task later for finite y).
    C_old = pd.to_numeric(df_old["compute_zflops"], errors="coerce").to_numpy(dtype=float)
    C_new = pd.to_numeric(df_new["compute_zflops"], errors="coerce").to_numpy(dtype=float)
    z_old = np.log10(np.maximum(C_old, 1e-300))
    z_new = np.log10(np.maximum(C_new, 1e-300))

    for k_label, train_pid, val_pid in split_specs:
        out_dir = os.path.join(out_base, f"k{k_label}")
        os.makedirs(out_dir, exist_ok=True)
        paths = EvaluationRunPaths(out_dir, legacy=False)

        mask_tr_base = (pid_old == train_pid) & np.isfinite(C_old) & (C_old > 0.0) & np.isfinite(z_old)
        mask_val_base = (pid_new == val_pid) & np.isfinite(C_new) & (C_new > 0.0) & np.isfinite(z_new)

        summary_rows: List[Tuple[str, float, float, float, float]] = []

        for task in tasks:
            y_old = pd.to_numeric(df_old[task], errors="coerce").to_numpy(dtype=float)
            y_new = pd.to_numeric(df_new[task], errors="coerce").to_numpy(dtype=float)

            mtr = mask_tr_base & np.isfinite(y_old)
            mval = mask_val_base & np.isfinite(y_new)

            C_tr = C_old[mtr]
            z_tr = z_old[mtr]
            y_tr = y_old[mtr]
            C_val = C_new[mval]
            z_val = z_new[mval]
            y_val = y_new[mval]

            if C_tr.size < 3:
                # Write empty outputs (so plotting can still discover the task list if other k's have it).
                write_bin_results(paths.get_bins_path(task, era="train"), [], era="train")
                era_label = "test_fixed" if oos_bins == "test_fixed" else "test_overlap"
                write_bin_results(paths.get_bins_path(task, era=era_label), [], era=era_label)
                write_task_summary(
                    paths.get_summary_path(task),
                    task=task,
                    mae_is_macro=float("nan"),
                    mae_is_micro=float("nan"),
                    mae_oos_macro=float("nan"),
                    mae_oos_micro=float("nan"),
                    K_used=0,
                    overlap_used=False,
                )
                summary_rows.append((task, float("nan"), float("nan"), float("nan"), float("nan")))
                continue

            edges_tr = create_equal_mass_bins(z_tr, int(max(1, bins)), int(max(1, min_bin_size)))
            if edges_tr.size < 2:
                write_bin_results(paths.get_bins_path(task, era="train"), [], era="train")
                era_label = "test_fixed" if oos_bins == "test_fixed" else "test_overlap"
                write_bin_results(paths.get_bins_path(task, era=era_label), [], era=era_label)
                write_task_summary(
                    paths.get_summary_path(task),
                    task=task,
                    mae_is_macro=float("nan"),
                    mae_is_micro=float("nan"),
                    mae_oos_macro=float("nan"),
                    mae_oos_micro=float("nan"),
                    K_used=0,
                    overlap_used=False,
                )
                summary_rows.append((task, float("nan"), float("nan"), float("nan"), float("nan")))
                continue

            xs_curve, y_curve = fit_sigmoid_predictor(
                C_tr,
                y_tr,
                tau=float(tau),
                frontier_fit_mode=str(frontier_fit_mode),
                bins_for_fit=int(bins),
                min_bin_size_for_fit=int(min_bin_size),
                bin_frontier_quantile=float(bin_frontier_quantile),
                bin_trim_fraction=float(bin_trim_fraction),
                bin_edges_for_fit=edges_tr,
            )

            yhat_tr = interpolate_curve(xs_curve, y_curve, C_tr)
            if metric == "calibration":
                bins_tr = compute_bin_statistics(z_tr, y_tr, yhat_tr, edges_tr, tau=float(tau))
            else:
                bins_tr = _compute_bin_statistics_pinball(z_tr, y_tr, yhat_tr, edges_tr, tau=float(tau))

            vals = [(n, ae) for (_bid, _lo, _hi, n, _ht, ae) in bins_tr if n > 0 and np.isfinite(ae)]
            if vals:
                n_vec = np.array([v[0] for v in vals], float)
                ae_vec = np.array([v[1] for v in vals], float)
                mae_is_macro = float(np.mean(ae_vec))
                mae_is_micro = float(np.sum(n_vec * ae_vec) / np.sum(n_vec))
            else:
                mae_is_macro = float("nan")
                mae_is_micro = float("nan")

            write_bin_results(paths.get_bins_path(task, era="train"), bins_tr, era="train")

            # OOS bin edges
            if oos_bins == "test_fixed":
                edges_val = (
                    create_equal_mass_bins(z_val, int(max(1, bins)), int(max(1, min_bin_size)))
                    if z_val.size
                    else np.array([], float)
                )
            else:
                edges_val = _edges_overlap(edges_tr, z_tr, z_val)

            mae_oos_macro = float("nan")
            mae_oos_micro = float("nan")
            era_label = "test_fixed" if oos_bins == "test_fixed" else "test_overlap"
            if edges_val.size >= 2 and z_val.size > 0:
                yhat_val = interpolate_curve(xs_curve, y_curve, C_val)
                if metric == "calibration":
                    bins_val = compute_bin_statistics(z_val, y_val, yhat_val, edges_val, tau=float(tau))
                else:
                    bins_val = _compute_bin_statistics_pinball(z_val, y_val, yhat_val, edges_val, tau=float(tau))
                write_bin_results(paths.get_bins_path(task, era=era_label), bins_val, era=era_label)
                vals_val = [(n, ae) for (_bid, _lo, _hi, n, _ht, ae) in bins_val if n > 0 and np.isfinite(ae)]
                if vals_val:
                    n_vec = np.array([v[0] for v in vals_val], float)
                    ae_vec = np.array([v[1] for v in vals_val], float)
                    mae_oos_macro = float(np.mean(ae_vec))
                    mae_oos_micro = float(np.sum(n_vec * ae_vec) / np.sum(n_vec))
            else:
                write_bin_results(paths.get_bins_path(task, era=era_label), [], era=era_label)

            write_task_summary(
                paths.get_summary_path(task),
                task=task,
                mae_is_macro=mae_is_macro,
                mae_is_micro=mae_is_micro,
                mae_oos_macro=mae_oos_macro,
                mae_oos_micro=mae_oos_micro,
                K_used=int(max(0, len(edges_tr) - 1)),
                overlap_used=bool(edges_val.size >= 2),
            )
            summary_rows.append((task, mae_is_macro, mae_is_micro, mae_oos_macro, mae_oos_micro))

        aggregate_task_metrics(paths.get_aggregate_path(), summary_rows)


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Evaluate old OLL frontiers on new models as OOS validation (period4 + P5)."
    )
    p.add_argument(
        "--oll_csv",
        default=os.path.join("tables", "open_llm_leaderboard", "open_llm_leaderboard_with_tokens.csv"),
        help="OLL CSV (with_tokens schema).",
    )
    p.add_argument(
        "--new_eval_csv",
        default=os.path.join("tables", "open_llm_leaderboard", "validation_leaderboard.csv"),
        help="Validation leaderboard CSV containing evaluation metrics (new models).",
    )
    p.add_argument(
        "--new_compute_csv",
        default=os.path.join("tables", "open_llm_leaderboard", "new_eval_leaderboard.csv"),
        help="CSV containing pretraining compute (pretrain_compute_zflops) for new models.",
    )
    p.add_argument(
        "--top_models_csv",
        default=os.path.join("tables", "top_models_by_base.csv"),
        help="Top-models metadata CSV containing last_modified for new models.",
    )
    p.add_argument(
        "--metric",
        choices=["calibration", "pinball"],
        default="calibration",
        help="Which per-bin error metric to write as abs_err.",
    )
    p.add_argument(
        "--include_bbh_subtasks",
        action="store_true",
        help="If set, include the 24 BBH subtasks in addition to main tasks.",
    )
    p.add_argument(
        "--out_base",
        required=True,
        help="Output base directory; writes k1..k4 subfolders under this path.",
    )
    p.add_argument("--tau", type=float, default=0.98, help="Tau used for fitting and evaluation.")
    p.add_argument("--bins", type=int, default=10, help="Target number of equal-mass bins.")
    p.add_argument("--min_bin_size", type=int, default=30, help="Minimum samples per bin.")
    p.add_argument(
        "--frontier_fit_mode",
        choices=["quantile_per_point", "robust_bin_frontier"],
        default="quantile_per_point",
        help="How to fit the sigmoid frontier (kept consistent with main evaluation).",
    )
    p.add_argument("--bin_frontier_quantile", type=float, default=0.98)
    p.add_argument("--bin_trim_fraction", type=float, default=0.01)
    p.add_argument(
        "--oos_bins",
        choices=["train_overlap", "test_fixed"],
        default="train_overlap",
        help="How to define OOS bins (same semantics as sigmoid_binned_mae.py).",
    )
    p.add_argument(
        "--eval_mode",
        choices=["same_period", "next_period", "p4_to_p5"],
        default="same_period",
        help=(
            "How to map old periods to new-model evaluation periods. "
            "'same_period' evaluates old Pk -> new Pk; "
            "'next_period' evaluates old Pk -> new P{k+1} with an extra new-only P5; "
            "'p4_to_p5' evaluates only old P4 -> new P5."
        ),
    )
    return p


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = build_argparser().parse_args(argv)

    df_old, df_new, tasks = _build_datasets(
        oll_csv=args.oll_csv,
        new_eval_csv=args.new_eval_csv,
        new_compute_csv=args.new_compute_csv,
        top_models_csv=args.top_models_csv,
        include_bbh_subtasks=bool(args.include_bbh_subtasks),
    )

    os.makedirs(args.out_base, exist_ok=True)
    run_eval(
        df_old=df_old,
        df_new=df_new,
        tasks=tasks,
        out_base=args.out_base,
        metric=str(args.metric),
        tau=float(args.tau),
        bins=int(args.bins),
        min_bin_size=int(args.min_bin_size),
        frontier_fit_mode=str(args.frontier_fit_mode),
        bin_frontier_quantile=float(args.bin_frontier_quantile),
        bin_trim_fraction=float(args.bin_trim_fraction),
        oos_bins=str(args.oos_bins),
        eval_mode=str(args.eval_mode),
    )


if __name__ == "__main__":
    main()

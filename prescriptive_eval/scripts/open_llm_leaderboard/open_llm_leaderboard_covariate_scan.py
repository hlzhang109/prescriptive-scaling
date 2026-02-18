#!/usr/bin/env python3

from __future__ import annotations

"""
Scan Open LLM Leaderboard metadata for systematic shifts vs a compute-only frontier.

This is intended as an exploratory tool to answer questions like:
  - Which *leaderboard-accessible* variables (model family, training regime, MoE, precision, ...)
    correlate with being above/below the compute->performance boundary?

Workflow (per task):
  1) Compute proxy: C = compute_multiplier * tokens_T * params_B  (units: zettaFLOPs).
  2) Fit a sigmoid quantile frontier y_hat(z) on a chosen reference period.
  3) Compute residuals r = y - y_hat(z) on an analysis subset.
  4) For each covariate:
       - categorical: group stats + within-z-bin adjusted mean residuals
       - numeric: within-z-bin partial Spearman + OLS slope (demeaned within bins)

Outputs go under --out_dir:
  - residuals/<task>.csv
  - covariates/<task>_groups.csv
  - covariates/<task>_summary.csv
  - covariates/summary_all_tasks.csv
"""

import argparse
import math
import os
import sys
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

# Ensure repo root is on sys.path when running as a script.
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from scipy.stats import spearmanr  # type: ignore

from skill_frontier.core.sigmoid import PERIOD4_BOUNDS  # type: ignore
from skill_frontier.evaluation.binning import create_equal_mass_bins  # type: ignore
from skill_frontier.io.csv_utils import maybe_scale_task_values, parse_year_month  # type: ignore
from scripts.run.sigmoid_quantile_optimizer import fit_sigmoid_enhanced, sigmoid_pred  # type: ignore


_DEFAULT_TASKS: Tuple[str, ...] = (
    "IFEval Raw",
    "BBH Raw",
    "MATH Lvl 5 Raw",
    "GPQA Raw",
    "MUSR Raw",
    "MMLU-PRO Raw",
)


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _safe_log10(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    out = np.full_like(x, np.nan, dtype=float)
    m = np.isfinite(x) & (x > 0)
    out[m] = np.log10(x[m])
    return out


def _coerce_bool_series(series: pd.Series) -> pd.Series:
    def _to_bool(v: object) -> bool:
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return False
        if isinstance(v, bool):
            return bool(v)
        s = str(v).strip().lower()
        return s in {"1", "true", "t", "yes", "y"}

    return series.apply(_to_bool)


def _assign_period(ym: Optional[Tuple[int, int]]) -> Tuple[Optional[int], Optional[str]]:
    if ym is None:
        return (None, None)
    y, m = int(ym[0]), int(ym[1])
    for idx, (label, (y0, m0), (y1, m1)) in enumerate(PERIOD4_BOUNDS, start=1):
        if (y, m) < (int(y0), int(m0)):
            continue
        if (y, m) > (int(y1), int(m1)):
            continue
        return (idx, f"P{idx}")
    return (None, None)


def _infer_org(model_id: str) -> str:
    s = str(model_id or "").strip()
    if "/" in s:
        return s.split("/", 1)[0].strip() or "unknown"
    return "unknown"


def _slugify(text: str) -> str:
    s = str(text).strip().lower()
    s = "".join(ch if ch.isalnum() else "_" for ch in s)
    s = "_".join(p for p in s.split("_") if p)
    return s or "task"


def _parse_period_list(spec: str, *, ref_period: int) -> Optional[List[int]]:
    s = str(spec or "").strip().lower()
    if s in {"same", "ref"}:
        return [int(ref_period)]
    if s in {"all", "*"}:
        return None
    out: List[int] = []
    for tok in str(spec).split(","):
        tok = tok.strip().lower()
        if not tok:
            continue
        if tok.startswith("p"):
            tok = tok[1:]
        try:
            p = int(tok)
        except Exception:
            continue
        if p not in out:
            out.append(p)
    return out or [int(ref_period)]


def _top_k_or_other(series: pd.Series, *, k: int) -> pd.Series:
    if k <= 0:
        return series
    vc = series.value_counts(dropna=False)
    keep = set(str(x) for x in vc.head(int(k)).index)
    return series.where(series.astype(str).isin(keep), other="OTHER")


@dataclass(frozen=True)
class NumericScanResult:
    variable: str
    n: int
    spearman_within_bins: float
    spearman_p_value: float
    beta_within_bins: float


def _scan_numeric_within_bins(
    *,
    variable: str,
    values: pd.Series,
    residuals: pd.Series,
    bin_id: pd.Series,
) -> Optional[NumericScanResult]:
    v = pd.to_numeric(values, errors="coerce")
    r = pd.to_numeric(residuals, errors="coerce")
    b = pd.to_numeric(bin_id, errors="coerce")
    m = np.isfinite(v) & np.isfinite(r) & np.isfinite(b)
    if int(m.sum()) < 50:
        return None

    df = pd.DataFrame({"v": v[m], "r": r[m], "b": b[m]})
    # Demean within bins (fixed effects for compute).
    df["v_tilde"] = df["v"] - df.groupby("b")["v"].transform("mean")
    df["r_tilde"] = df["r"] - df.groupby("b")["r"].transform("mean")

    vv = df["v_tilde"].to_numpy(dtype=float)
    rr = df["r_tilde"].to_numpy(dtype=float)
    # Spearman on demeaned variables.
    rho, p = spearmanr(vv, rr)
    # OLS slope on demeaned variables.
    denom = float(np.dot(vv, vv))
    beta = float(np.dot(vv, rr) / denom) if denom > 0 else float("nan")
    return NumericScanResult(
        variable=str(variable),
        n=int(df.shape[0]),
        spearman_within_bins=float(rho) if np.isfinite(rho) else float("nan"),
        spearman_p_value=float(p) if np.isfinite(p) else float("nan"),
        beta_within_bins=float(beta) if np.isfinite(beta) else float("nan"),
    )


def _bin_ids_from_edges(z: np.ndarray, edges: np.ndarray) -> np.ndarray:
    if edges.size < 2:
        return np.zeros_like(z, dtype=int)
    if edges.size == 2 and float(edges[0]) == float(edges[1]):
        return np.zeros_like(z, dtype=int)
    # digitize against interior edges -> [0, B-1]
    interior = edges[1:-1]
    return np.digitize(z, interior, right=False).astype(int)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--csv",
        default=os.path.join("tables", "open_llm_leaderboard", "open_llm_leaderboard_with_tokens.csv"),
        help="Path to Open LLM Leaderboard v2 CSV with tokens/params.",
    )
    ap.add_argument(
        "--tasks",
        default=",".join(_DEFAULT_TASKS),
        help="Comma-separated task columns to scan (default: the 6 main 'Raw' tasks).",
    )
    ap.add_argument("--ref_period", type=int, default=3, help="Reference period index (1..4) used to fit y_hat.")
    ap.add_argument(
        "--analyze_periods",
        default="same",
        help="Which periods to analyze residuals on: 'same' (default), 'all', or e.g. '2,3,4'.",
    )
    ap.add_argument("--tau", type=float, default=0.98, help="Quantile for the sigmoid frontier fit.")
    ap.add_argument("--kappa_final", type=float, default=50.0, help="Smoothed pinball sharpness for the final stage.")
    ap.add_argument("--lambda_b", type=float, default=1e-2, help="Ridge penalty on sigmoid slope b.")
    ap.add_argument("--compute_multiplier", type=float, default=6.0, help="Compute proxy multiplier (default 6*T*P).")
    ap.add_argument("--n_bins", type=int, default=20, help="Equal-mass bins in z for within-bin adjustment.")
    ap.add_argument("--min_bin", type=int, default=40, help="Minimum samples per z-bin (bins violating are merged).")
    ap.add_argument("--min_group_n", type=int, default=50, help="Minimum group size for categorical breakdowns.")
    ap.add_argument("--max_groups", type=int, default=12, help="Max categories to keep for high-cardinality variables.")
    ap.add_argument(
        "--out_dir",
        default=os.path.join("outputs", "open_llm_leaderboard", "covariate_scan"),
        help="Output directory.",
    )
    ap.add_argument(
        "--exclude_flagged",
        action="store_true",
        help="Drop rows where 'Flagged' is truthy (if the column exists).",
    )
    ap.add_argument(
        "--require_hub_available",
        action="store_true",
        help="Keep only rows where 'Available on the hub' is truthy (if the column exists).",
    )
    args = ap.parse_args()

    tasks = [t.strip() for t in str(args.tasks).split(",") if t.strip()]
    if not tasks:
        raise SystemExit("No tasks specified.")

    df = pd.read_csv(str(args.csv))
    for col in ("Pretraining tokens (T)", "#Params (B)"):
        if col not in df.columns:
            raise SystemExit(f"Missing required compute column {col!r} in {args.csv}")

    if bool(args.exclude_flagged) and "Flagged" in df.columns:
        df = df[~_coerce_bool_series(df["Flagged"])].copy()
    if bool(args.require_hub_available) and "Available on the hub" in df.columns:
        df = df[_coerce_bool_series(df["Available on the hub"])].copy()

    tokens_t = pd.to_numeric(df["Pretraining tokens (T)"], errors="coerce")
    params_b = pd.to_numeric(df["#Params (B)"], errors="coerce")
    compute_zflops = float(args.compute_multiplier) * tokens_t * params_b
    z = _safe_log10(compute_zflops.to_numpy(dtype=float))
    m_compute = np.isfinite(z)
    df = df[m_compute].copy()
    df["compute_zflops"] = compute_zflops[m_compute].to_numpy(dtype=float)
    df["z"] = z[m_compute]
    # Derived covariates
    df["tokens_per_param"] = (tokens_t[m_compute] / params_b[m_compute]).to_numpy(dtype=float)
    df["log_tokens_per_param"] = _safe_log10(df["tokens_per_param"].to_numpy(dtype=float))
    df["log_params_b"] = _safe_log10(params_b[m_compute].to_numpy(dtype=float))
    df["log_tokens_t"] = _safe_log10(tokens_t[m_compute].to_numpy(dtype=float))

    # Period assignment (from Upload To Hub Date)
    ym = df.get("Upload To Hub Date", pd.Series([None] * len(df))).astype(str).map(parse_year_month)
    period_info = ym.map(_assign_period)
    df["period_idx"] = period_info.map(lambda t: t[0])
    df["period"] = period_info.map(lambda t: t[1])

    # Model identifiers (prefer fullname; fall back to whatever is present).
    if "fullname" in df.columns:
        df["model_id"] = df["fullname"].astype(str).str.strip()
    elif "model_id" in df.columns:
        df["model_id"] = df["model_id"].astype(str).str.strip()
    else:
        df["model_id"] = ""
    df["org"] = df["model_id"].map(_infer_org)

    analyze_periods = _parse_period_list(str(args.analyze_periods), ref_period=int(args.ref_period))
    if analyze_periods is None:
        df_analyze = df.copy()
    else:
        df_analyze = df[df["period_idx"].isin([int(p) for p in analyze_periods])].copy()
    if df_analyze.empty:
        raise SystemExit("No rows available after period filtering.")

    out_dir = str(args.out_dir)
    residuals_dir = os.path.join(out_dir, "residuals")
    cov_dir = os.path.join(out_dir, "covariates")
    _ensure_dir(residuals_dir)
    _ensure_dir(cov_dir)

    # Covariates to scan (categorical + numeric).
    categorical_vars = [
        "Base model family",
        "Architecture",
        "Type",
        "Precision",
        "Weight type",
        "MoE",
        "Merged",
        "Official Providers",
        "Hub License",
        "org",
    ]
    numeric_vars = [
        "Hub ❤️",
        "CO₂ cost (kg)",
        "log_params_b",
        "log_tokens_t",
        "log_tokens_per_param",
    ]

    all_summaries: List[pd.DataFrame] = []

    for task in tasks:
        if task not in df.columns:
            print(f"[skip] Missing task column: {task!r}")
            continue

        df_task = df_analyze.copy()
        y_raw = pd.to_numeric(df_task[task], errors="coerce").to_numpy(dtype=float)
        y = maybe_scale_task_values(y_raw)
        df_task = df_task[np.isfinite(y)].copy()
        df_task["y"] = y[np.isfinite(y)]
        if df_task.empty:
            print(f"[skip] No finite scores for {task!r} after filtering.")
            continue

        df_fit = df_task.copy()
        df_fit = df_fit[df_fit["period_idx"] == int(args.ref_period)].copy()
        if df_fit.empty:
            raise SystemExit(f"No rows in reference period P{int(args.ref_period)} for task {task!r}.")

        fit = fit_sigmoid_enhanced(
            df_fit["z"].to_numpy(dtype=float),
            df_fit["y"].to_numpy(dtype=float),
            tau=float(args.tau),
            kappa_final=float(args.kappa_final),
            lambda_b=float(args.lambda_b),
            seed=0,
        )
        if not bool(fit.success) or not np.all(np.isfinite(fit.params)):
            raise SystemExit(f"Sigmoid fit failed for task {task!r}: {fit.info}")
        params = fit.params

        df_task["yhat"] = sigmoid_pred(params, df_task["z"].to_numpy(dtype=float))
        df_task["residual"] = df_task["y"].to_numpy(dtype=float) - df_task["yhat"].to_numpy(dtype=float)
        df_task["above"] = df_task["residual"] > 0.0

        # Build compute bins on the analysis set for within-bin adjustment.
        edges = create_equal_mass_bins(df_task["z"].to_numpy(dtype=float), int(args.n_bins), int(args.min_bin))
        df_task["bin_id"] = _bin_ids_from_edges(df_task["z"].to_numpy(dtype=float), edges)

        task_slug = _slugify(task)
        residuals_path = os.path.join(residuals_dir, f"{task_slug}.csv")
        keep_cols = [
            "model_id",
            "eval_name" if "eval_name" in df_task.columns else None,
            "period_idx",
            "period",
            "z",
            "compute_zflops",
            "y",
            "yhat",
            "residual",
            "above",
            "bin_id",
        ]
        # Add covariate columns (if present)
        keep_cols += [c for c in categorical_vars + numeric_vars if c in df_task.columns]
        keep_cols = [c for c in keep_cols if c is not None and c in df_task.columns]
        df_task.to_csv(residuals_path, index=False, columns=keep_cols)

        # -------------------------
        # Categorical scan
        # -------------------------
        group_rows: List[dict] = []
        summary_rows: List[dict] = []

        # Precompute bin-demeaned residuals for adjusted group means.
        r_mean_by_bin = df_task.groupby("bin_id")["residual"].mean()
        df_task["_resid_tilde"] = df_task["residual"] - df_task["bin_id"].map(r_mean_by_bin)

        for var in categorical_vars:
            if var not in df_task.columns:
                continue
            s = df_task[var]
            # Normalize booleans consistently.
            if var in {"MoE", "Merged", "Official Providers"}:
                s = _coerce_bool_series(s).map({True: "True", False: "False"})
            else:
                s = s.fillna("").astype(str).str.strip().replace({"": "NA"})

            # Reduce high-cardinality variables.
            if var in {"Base model family", "Architecture", "Hub License", "org"}:
                s = _top_k_or_other(s, k=int(args.max_groups))

            tmp = df_task.assign(_g=s)
            g = (
                tmp.groupby("_g", dropna=False)
                .agg(
                    n=("residual", "size"),
                    mean_residual=("residual", "mean"),
                    median_residual=("residual", "median"),
                    mean_adj_residual=("_resid_tilde", "mean"),
                    above_rate=("above", "mean"),
                    mean_z=("z", "mean"),
                )
                .reset_index()
                .rename(columns={"_g": "group"})
            )
            g = g[g["n"] >= int(args.min_group_n)].copy()
            if g.empty:
                continue
            g["task"] = task
            g["variable"] = var
            g = g.sort_values("mean_adj_residual", ascending=False, kind="mergesort")

            for _, row in g.iterrows():
                group_rows.append(
                    {
                        "task": task,
                        "variable": var,
                        "group": row["group"],
                        "n": int(row["n"]),
                        "mean_residual": float(row["mean_residual"]),
                        "median_residual": float(row["median_residual"]),
                        "mean_adj_residual": float(row["mean_adj_residual"]),
                        "above_rate": float(row["above_rate"]),
                        "mean_z": float(row["mean_z"]),
                    }
                )

            top = g.iloc[0]
            bottom = g.iloc[-1]
            summary_rows.append(
                {
                    "task": task,
                    "variable": var,
                    "kind": "categorical",
                    "n_total": int(g["n"].sum()),
                    "n_groups": int(g.shape[0]),
                    "effect_adj_mean_residual": float(top["mean_adj_residual"] - bottom["mean_adj_residual"]),
                    "top_group": str(top["group"]),
                    "top_group_adj_mean_residual": float(top["mean_adj_residual"]),
                    "bottom_group": str(bottom["group"]),
                    "bottom_group_adj_mean_residual": float(bottom["mean_adj_residual"]),
                }
            )

        # -------------------------
        # Numeric scan
        # -------------------------
        for var in numeric_vars:
            if var not in df_task.columns:
                continue
            scan = _scan_numeric_within_bins(
                variable=var, values=df_task[var], residuals=df_task["residual"], bin_id=df_task["bin_id"]
            )
            if scan is None:
                continue
            summary_rows.append(
                {
                    "task": task,
                    "variable": var,
                    "kind": "numeric",
                    "n_total": int(scan.n),
                    "n_groups": 0,
                    "effect_adj_mean_residual": float("nan"),
                    "top_group": "",
                    "top_group_adj_mean_residual": float("nan"),
                    "bottom_group": "",
                    "bottom_group_adj_mean_residual": float("nan"),
                    "spearman_within_bins": float(scan.spearman_within_bins),
                    "spearman_p_value": float(scan.spearman_p_value),
                    "beta_within_bins": float(scan.beta_within_bins),
                }
            )

        # Write outputs
        df_groups = pd.DataFrame.from_records(group_rows)
        df_summary = pd.DataFrame.from_records(summary_rows)
        groups_path = os.path.join(cov_dir, f"{task_slug}_groups.csv")
        summary_path = os.path.join(cov_dir, f"{task_slug}_summary.csv")
        if not df_groups.empty:
            df_groups.to_csv(groups_path, index=False)
        if not df_summary.empty:
            df_summary.to_csv(summary_path, index=False)

        if not df_summary.empty:
            all_summaries.append(df_summary.assign(task=task))

        # Concise console output (top categorical effects by adjusted mean residual spread)
        if not df_summary.empty:
            cat = df_summary[df_summary["kind"] == "categorical"].copy()
            cat = cat.sort_values("effect_adj_mean_residual", ascending=False, kind="mergesort")
            print(f"[{task}] wrote {residuals_path}")
            if not cat.empty:
                show = cat.head(5)[
                    [
                        "variable",
                        "effect_adj_mean_residual",
                        "top_group",
                        "top_group_adj_mean_residual",
                        "bottom_group",
                        "bottom_group_adj_mean_residual",
                    ]
                ]
                print(show.to_string(index=False))

    if all_summaries:
        df_all = pd.concat(all_summaries, ignore_index=True)
        out_all = os.path.join(cov_dir, "summary_all_tasks.csv")
        df_all.to_csv(out_all, index=False)
        print(f"Wrote: {out_all}")


if __name__ == "__main__":
    main()

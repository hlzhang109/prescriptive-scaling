#!/usr/bin/env python3

from __future__ import annotations

"""
Model-selection pipeline for a contamination check experiment (e.g., AIME 2025).

This script selects ~N models to evaluate on a NEW benchmark (selection only; no evaluation code).

-----------------------
Step-0 discoveries (repo)
-----------------------
Data sources already present in this repo (no downloads required):
  (A) Canonical Open LLM Leaderboard v2 table used by Section 4 sigmoid plots:
      - tables/open_llm_leaderboard/open_llm_leaderboard_with_tokens.csv
      - Key columns:
          * model identifier (HF repo): `fullname` (e.g., "01-ai/Yi-1.5-34B")
          * row identifier: `eval_name` (contains precision suffix, not ideal for eval harness)
          * base model: `Identified base model` (fallback: `Base Model`)
          * date for period split: `Upload To Hub Date`
          * compute inputs: `Pretraining tokens (T)` and `#Params (B)`
          * task scores: e.g. `MATH Lvl 5 Raw`, `MMLU-PRO Raw` (already in [0,1])
      - Period assignment is the repo's PERIOD4_BOUNDS (P1..P4) via `Upload To Hub Date`.

  (B) Newly-evaluated (extra) models with compute/tokens/params (used in period4 old-vs-new overlays):
      - tables/new_leaderboard_results_with_tokens.csv
      - Key columns:
          * model identifier: `model_id` (HF repo)
          * base model: `mapped_base_model`
          * compute inputs: `Pretraining tokens (T)`, `#Params (B)`
          * task scores in "leaderboard_*" columns; mapping matches scripts/evaluate/eval_new_models_oos_periods.py:
              "MATH Lvl 5 Raw" -> leaderboard_math_hard_exact_match_none
              "MMLU-PRO Raw"  -> leaderboard_mmlu_pro_acc_none
      - No date column: we label these as "P5" (a new-only period after PERIOD4_BOUNDS).

Sigmoid frontier parameters:
  - Existing plotting pipeline writes sampled curves to outputs/sigmoid/**/curves/*.csv,
    but does not persist fitted param vectors.
  - We therefore re-fit the reference-period sigmoid using the same optimizer used for plots:
      scripts/run/sigmoid_quantile_optimizer.py::fit_sigmoid_enhanced
    with smoothed pinball parameters (tau, kappa_final) and ridge penalty (lambda_b).

Model identifier for downstream evaluation:
  - The inference/eval harness in this repo ecosystem (e.g., LiveBench) uses Hugging Face repo ids
    passed to `from_pretrained(...)` (see LiveBench-main/livebench/model/model_adapter.py).
  - Therefore, we output `model_id` as the HF repo id:
      * `fullname` for OLL v2 rows
      * `model_id` for the new leaderboard extra table
"""

import argparse
import json
import math
import os
import sys
from collections import Counter
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

# Ensure repo root is on sys.path when running as a script.
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from skill_frontier.core.period_utils import (  # type: ignore
    assign_period_index_period4_one_based,
    parse_period_list as _parse_period_list,
    parse_period_token as _parse_period_token,
)
from skill_frontier.core.sigmoid import PERIOD4_BOUNDS  # type: ignore
from skill_frontier.io.csv_utils import maybe_scale_task_values, parse_year_month  # type: ignore


OLL_V2_WITH_TOKENS = os.path.join(
    "tables", "open_llm_leaderboard", "open_llm_leaderboard_with_tokens.csv"
)
NEW_LB_WITH_TOKENS = os.path.join("tables", "new_leaderboard_results_with_tokens.csv")

# Matches scripts/evaluate/eval_new_models_oos_periods.py::MAIN_TASK_MAP_NEW_TO_OLD
NEW_LB_TASK_MAP: Dict[str, str] = {
    "IFEval Raw": "leaderboard_ifeval_inst_level_strict_acc_none",
    "BBH Raw": "leaderboard_bbh_acc_norm_none",
    "MATH Lvl 5 Raw": "leaderboard_math_hard_exact_match_none",
    "GPQA Raw": "leaderboard_gpqa_acc_norm_none",
    "MUSR Raw": "leaderboard_musr_acc_norm_none",
    "MMLU-PRO Raw": "leaderboard_mmlu_pro_acc_none",
}


def _sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, float)
    # Stable expit without SciPy.
    out = np.empty_like(x, dtype=float)
    pos = x >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    exp_x = np.exp(x[~pos])
    out[~pos] = exp_x / (1.0 + exp_x)
    return out


def _assign_period_index(ym: Tuple[int, int]) -> Optional[int]:
    idx = assign_period_index_period4_one_based(int(ym[0]), int(ym[1]))
    return None if idx < 0 else idx


def _infer_org(model_id: str) -> str:
    s = str(model_id or "").strip()
    if "/" in s:
        return s.split("/", 1)[0].strip() or "unknown"
    return "unknown"


def _coerce_bool_series(series: pd.Series) -> pd.Series:
    def _to_bool(v: object) -> bool:
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return False
        if isinstance(v, bool):
            return bool(v)
        s = str(v).strip().lower()
        return s in {"1", "true", "t", "yes", "y"}

    return series.apply(_to_bool)


def _compute_pretraining_zflops(tokens_t: pd.Series, params_b: pd.Series, *, multiplier: float = 6.0) -> pd.Series:
    t = pd.to_numeric(tokens_t, errors="coerce")
    p = pd.to_numeric(params_b, errors="coerce")
    zflops = multiplier * t * p
    zflops = zflops.where(zflops > 0)
    return zflops


def _safe_log10(series: pd.Series) -> pd.Series:
    v = pd.to_numeric(series, errors="coerce")
    v = v.where(v > 0)
    return np.log10(v)


def _dedupe_by_model_id(df: pd.DataFrame) -> pd.DataFrame:
    """De-duplicate by model_id, keeping the newest (period_idx, date) record."""
    if df.empty:
        return df
    df = df.copy()
    df["period_idx"] = pd.to_numeric(df.get("period_idx", np.nan), errors="coerce")
    # Prefer higher period index (e.g., P5 new-only rows), then newer date, then higher y_math.
    df["_date_sort"] = pd.to_datetime(df.get("date", pd.Series([pd.NaT] * len(df))), errors="coerce")
    df["_y_sort"] = pd.to_numeric(df.get("y_math", pd.Series([np.nan] * len(df))), errors="coerce")
    df = df.sort_values(
        by=["period_idx", "_date_sort", "_y_sort", "model_id"],
        ascending=[False, False, False, True],
        kind="mergesort",
    )
    df = df.drop_duplicates(subset=["model_id"], keep="first").copy()
    df = df.drop(columns=["_date_sort", "_y_sort"], errors="ignore")
    return df


def load_model_table(
    *,
    task: str,
    include_new_lb: bool = True,
    compute_multiplier: float = 6.0,
) -> pd.DataFrame:
    """Load and normalize model metadata + task scores into a single DataFrame.

    Output columns (at least):
      model_id, base_model_id, org, period, period_idx, flops_zflops, z, y_math
    """
    frames: List[pd.DataFrame] = []

    # -----------------------------
    # (A) OLL v2 with tokens
    # -----------------------------
    if not os.path.exists(OLL_V2_WITH_TOKENS):
        raise FileNotFoundError(f"Missing required table: {OLL_V2_WITH_TOKENS}")

    usecols_oll = [
        "eval_name",
        "fullname",
        "Upload To Hub Date",
        "Pretraining tokens (T)",
        "#Params (B)",
        task,
        "MMLU-PRO Raw",
        "Identified base model",
        "Base Model",
        "Available on the hub",
        "Flagged",
        "Merged",
        "MoE",
    ]
    df_oll = pd.read_csv(
        OLL_V2_WITH_TOKENS,
        usecols=lambda c: c in set(usecols_oll),
    )
    if "fullname" not in df_oll.columns:
        raise KeyError(f"Expected column 'fullname' in {OLL_V2_WITH_TOKENS}")
    if task not in df_oll.columns:
        raise KeyError(f"Expected task column {task!r} in {OLL_V2_WITH_TOKENS}")

    df_oll = df_oll.rename(
        columns={
            "fullname": "model_id",
            "Upload To Hub Date": "date",
            task: "y_math",
            "MMLU-PRO Raw": "y_mmlu",
        }
    )
    df_oll["source"] = "oll_v2"
    df_oll["org"] = df_oll["model_id"].astype(str).map(_infer_org)
    # base model id
    base = df_oll.get("Identified base model")
    if base is None:
        base = pd.Series([""] * len(df_oll))
    base2 = df_oll.get("Base Model")
    if base2 is None:
        base2 = pd.Series([""] * len(df_oll))
    base = base.fillna("").astype(str).str.strip()
    base2 = base2.fillna("").astype(str).str.strip()
    df_oll["base_model_id"] = base.where(base != "", base2)
    df_oll["base_model_id"] = df_oll["base_model_id"].where(df_oll["base_model_id"] != "", df_oll["model_id"])

    # Open-weight filter: only keep models available on HF hub when that flag is present.
    if "Available on the hub" in df_oll.columns:
        df_oll["available_on_hub"] = _coerce_bool_series(df_oll["Available on the hub"])
        df_oll = df_oll[df_oll["available_on_hub"]].copy()
    else:
        df_oll["available_on_hub"] = True

    for col in ("Flagged", "Merged", "MoE"):
        if col in df_oll.columns:
            df_oll[col.lower()] = _coerce_bool_series(df_oll[col])

    # Compute + z
    if "Pretraining tokens (T)" not in df_oll.columns or "#Params (B)" not in df_oll.columns:
        raise KeyError(f"Missing compute columns in {OLL_V2_WITH_TOKENS} (need 'Pretraining tokens (T)' and '#Params (B)')")
    df_oll["flops_zflops"] = _compute_pretraining_zflops(
        df_oll["Pretraining tokens (T)"], df_oll["#Params (B)"], multiplier=float(compute_multiplier)
    )
    df_oll["z"] = _safe_log10(df_oll["flops_zflops"])

    # Period assignment: P1..P4 based on PERIOD4_BOUNDS + Upload To Hub Date.
    df_oll["date"] = pd.to_datetime(df_oll["date"], errors="coerce")
    ym = df_oll["date"].astype(str).map(parse_year_month)
    period_idx = ym.map(lambda t: _assign_period_index(t) if t is not None else None)
    df_oll["period_idx"] = period_idx
    df_oll["period"] = df_oll["period_idx"].map(lambda p: f"P{int(p)}" if pd.notna(p) else None)

    # Scale (robust to percent-encoded tables).
    df_oll["y_math"] = maybe_scale_task_values(pd.to_numeric(df_oll["y_math"], errors="coerce").to_numpy())
    df_oll["y_mmlu"] = maybe_scale_task_values(pd.to_numeric(df_oll.get("y_mmlu", np.nan), errors="coerce").to_numpy())
    df_oll["y_math"] = pd.to_numeric(df_oll["y_math"], errors="coerce")
    df_oll["y_mmlu"] = pd.to_numeric(df_oll["y_mmlu"], errors="coerce")

    # Keep only models with compute and y_math.
    df_oll = df_oll[np.isfinite(df_oll["flops_zflops"]) & np.isfinite(df_oll["y_math"])].copy()
    df_oll = df_oll[np.isfinite(df_oll["z"]) & df_oll["period_idx"].notna()].copy()
    frames.append(df_oll)

    # -----------------------------
    # (B) New leaderboard results with tokens (extra models; label as P5)
    # -----------------------------
    if include_new_lb and os.path.exists(NEW_LB_WITH_TOKENS):
        new_task_col = NEW_LB_TASK_MAP.get(task, "")
        usecols_new = [
            "mapped_base_model",
            "model_id",
            "Pretraining tokens (T)",
            "#Params (B)",
            new_task_col,
            NEW_LB_TASK_MAP.get("MMLU-PRO Raw", ""),
        ]
        df_new = pd.read_csv(
            NEW_LB_WITH_TOKENS,
            usecols=lambda c: c in set(usecols_new),
        )
        if "model_id" in df_new.columns and new_task_col in df_new.columns:
            df_new = df_new.rename(
                columns={
                    "mapped_base_model": "base_model_id",
                    new_task_col: "y_math",
                    NEW_LB_TASK_MAP.get("MMLU-PRO Raw", ""): "y_mmlu",
                }
            )
            df_new["source"] = "new_lb"
            df_new["org"] = df_new["model_id"].astype(str).map(_infer_org)
            df_new["date"] = pd.NaT
            df_new["period_idx"] = len(PERIOD4_BOUNDS) + 1  # new-only P5
            df_new["period"] = df_new["period_idx"].map(lambda p: f"P{int(p)}")

            df_new["base_model_id"] = df_new.get("base_model_id", "").fillna("").astype(str).str.strip()
            df_new["base_model_id"] = df_new["base_model_id"].where(df_new["base_model_id"] != "", df_new["model_id"])
            df_new["flops_zflops"] = _compute_pretraining_zflops(
                df_new["Pretraining tokens (T)"], df_new["#Params (B)"], multiplier=float(compute_multiplier)
            )
            df_new["z"] = _safe_log10(df_new["flops_zflops"])

            df_new["y_math"] = maybe_scale_task_values(pd.to_numeric(df_new["y_math"], errors="coerce").to_numpy())
            df_new["y_mmlu"] = maybe_scale_task_values(pd.to_numeric(df_new.get("y_mmlu", np.nan), errors="coerce").to_numpy())
            df_new["y_math"] = pd.to_numeric(df_new["y_math"], errors="coerce")
            df_new["y_mmlu"] = pd.to_numeric(df_new["y_mmlu"], errors="coerce")

            df_new = df_new[np.isfinite(df_new["flops_zflops"]) & np.isfinite(df_new["y_math"])].copy()
            df_new = df_new[np.isfinite(df_new["z"])].copy()
            frames.append(df_new)

    df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if df.empty:
        raise RuntimeError("No rows after loading/normalizing model tables.")

    # Final de-duplication by HF repo id.
    df = _dedupe_by_model_id(df)
    df = df.reset_index(drop=True)
    return df


@dataclass
class SigmoidParams:
    y0: float
    L: float
    z_star: float
    log_b: float

    @property
    def b(self) -> float:
        return float(np.exp(self.log_b))

    def as_theta_y0_L_a_b(self) -> Dict[str, float]:
        b = self.b
        return {"y0": float(self.y0), "L": float(self.L), "a": float(-b * self.z_star), "b": float(b)}


def fit_sigmoid_params(
    z: np.ndarray,
    y: np.ndarray,
    *,
    tau: float,
    kappa_final: float,
    lambda_b: float,
    seed: int,
) -> SigmoidParams:
    z = np.asarray(z, float)
    y = np.asarray(y, float)
    mask = np.isfinite(z) & np.isfinite(y)
    z = z[mask]
    y = y[mask]
    if z.size < 10:
        raise RuntimeError(f"Not enough points to fit sigmoid (n={int(z.size)})")

    # Import here to keep this script lightweight unless fitting is needed.
    try:
        from scripts.run.sigmoid_quantile_optimizer import fit_sigmoid_enhanced  # type: ignore
    except Exception:
        from run.sigmoid_quantile_optimizer import fit_sigmoid_enhanced  # type: ignore

    res = fit_sigmoid_enhanced(
        z,
        y,
        tau=float(tau),
        kappa_final=float(kappa_final),
        lambda_b=float(lambda_b),
        n_zstar_grid=10,
        n_b_grid=10,
        n_random=100,
        seed=int(seed),
    )
    if (not bool(res.success)) or (not np.all(np.isfinite(res.params))):
        raise RuntimeError("Sigmoid fit failed (non-finite parameters).")
    y0, L, z_star, log_b = [float(v) for v in res.params]
    return SigmoidParams(y0=y0, L=L, z_star=z_star, log_b=log_b)


def predict_sigmoid(z: np.ndarray, params: SigmoidParams) -> np.ndarray:
    z = np.asarray(z, float)
    t = params.b * (z - float(params.z_star))
    return float(params.y0) + float(params.L) * _sigmoid(t)


def _compute_bin_edges_qcut(z: pd.Series, n_bins: int) -> np.ndarray:
    z = pd.to_numeric(z, errors="coerce")
    z = z[np.isfinite(z)]
    if z.empty:
        return np.array([], dtype=float)
    try:
        _labels, edges = pd.qcut(z, q=int(n_bins), labels=False, retbins=True, duplicates="drop")
    except Exception:
        return np.array([], dtype=float)
    edges = np.asarray(edges, dtype=float)
    edges = np.unique(edges)
    if edges.size < 2:
        return np.array([], dtype=float)
    return edges


def _assign_bins(z: pd.Series, edges: np.ndarray) -> pd.Series:
    if edges.size < 2:
        return pd.Series([np.nan] * len(z), index=z.index)
    return pd.cut(z, bins=edges, labels=False, include_lowest=True)


def _default_bin_weights(n_bins: int) -> np.ndarray:
    weights = np.ones(int(n_bins), dtype=float)
    if n_bins <= 0:
        return weights
    top = min(3, int(n_bins))
    if top > 0:
        weights[-top:] *= 2.0
    return weights


def _allocate_quotas(total: int, weights: np.ndarray) -> np.ndarray:
    weights = np.asarray(weights, float)
    if total <= 0 or weights.size == 0 or float(np.sum(weights)) <= 0:
        return np.zeros_like(weights, dtype=int)
    raw = total * (weights / float(np.sum(weights)))
    quota = np.floor(raw).astype(int)
    remainder = int(total - int(np.sum(quota)))
    if remainder > 0:
        frac = raw - quota
        # Highest fractional part first; tie-break to higher bin index (higher compute).
        order = sorted(range(weights.size), key=lambda i: (float(frac[i]), i), reverse=True)
        for i in order[:remainder]:
            quota[i] += 1
    return quota.astype(int)


def _compute_group_sizes(n_total: int) -> Tuple[int, int, int, int]:
    n_total = int(n_total)
    n_a = int(round(0.40 * n_total))
    n_b = int(round(0.30 * n_total))
    n_c = int(round(0.20 * n_total))
    n_d = int(n_total - n_a - n_b - n_c)
    if n_d < 0:
        n_d = 0
    # Fix rounding drift if needed.
    while (n_a + n_b + n_c + n_d) < n_total:
        n_a += 1
    while (n_a + n_b + n_c + n_d) > n_total and n_a > 0:
        n_a -= 1
    n_d = int(n_total - n_a - n_b - n_c)
    return n_a, n_b, n_c, n_d


def _matches_any_family(model_id: str, patterns_lower: List[str]) -> bool:
    s = str(model_id).lower()
    return any(p in s for p in patterns_lower if p)


@dataclass
class SelectionCaps:
    max_per_base: int
    max_per_org: int


class SelectionState:
    def __init__(self, caps: SelectionCaps):
        self.caps = caps
        self.selected: List[Dict[str, object]] = []
        self.selected_ids: set[str] = set()
        self.base_counts: Counter[str] = Counter()
        self.org_counts: Counter[str] = Counter()

    def can_add(self, model_id: str, base_model_id: str, org: str) -> bool:
        if model_id in self.selected_ids:
            return False
        if self.base_counts[base_model_id] >= int(self.caps.max_per_base):
            return False
        if self.org_counts[org] >= int(self.caps.max_per_org):
            return False
        return True

    def add_record(self, record: Dict[str, object]) -> None:
        model_id = str(record.get("model_id", ""))
        base_model_id = str(record.get("base_model_id", ""))
        org = str(record.get("org", "unknown"))
        self.selected.append(record)
        self.selected_ids.add(model_id)
        self.base_counts[base_model_id] += 1
        self.org_counts[org] += 1

    def remove_record(self, record: Dict[str, object]) -> None:
        model_id = str(record.get("model_id", ""))
        base_model_id = str(record.get("base_model_id", ""))
        org = str(record.get("org", "unknown"))
        self.selected.remove(record)
        self.selected_ids.remove(model_id)
        self.base_counts[base_model_id] -= 1
        if self.base_counts[base_model_id] <= 0:
            self.base_counts.pop(base_model_id, None)
        self.org_counts[org] -= 1
        if self.org_counts[org] <= 0:
            self.org_counts.pop(org, None)


def _row_to_record(
    row: pd.Series,
    *,
    selection_group: str,
    rationale: str,
    matched_to_model_id: Optional[str] = None,
    matched_delta_z: Optional[float] = None,
) -> Dict[str, object]:
    def _get(name: str, default: object = None) -> object:
        return row[name] if name in row.index else default

    return {
        "model_id": str(_get("model_id", "")),
        "eval_name": str(_get("eval_name", "")) if _get("eval_name", None) is not None else "",
        "source": str(_get("source", "")),
        "base_model_id": str(_get("base_model_id", "")),
        "org": str(_get("org", "unknown")),
        "period": str(_get("period", "")),
        "period_idx": int(_get("period_idx", -1)) if pd.notna(_get("period_idx", np.nan)) else -1,
        "flops_zflops": float(_get("flops_zflops", float("nan"))),
        "z": float(_get("z", float("nan"))),
        "y_math": float(_get("y_math", float("nan"))),
        "pred_ref": float(_get("pred_ref", float("nan"))),
        "residual_ref": float(_get("residual_ref", float("nan"))),
        "bin_id": int(_get("bin_id", -1)) if pd.notna(_get("bin_id", np.nan)) else -1,
        "y_mmlu": float(_get("y_mmlu", float("nan"))),
        "pred_mmlu_ref": float(_get("pred_mmlu_ref", float("nan"))),
        "residual_mmlu_ref": float(_get("residual_mmlu_ref", float("nan"))),
        "specialist_score": float(_get("specialist_score", float("nan"))),
        "selection_group": str(selection_group),
        "rationale": str(rationale),
        "matched_to_model_id": str(matched_to_model_id) if matched_to_model_id is not None else "",
        "matched_delta_z": float(matched_delta_z) if matched_delta_z is not None and np.isfinite(matched_delta_z) else float("nan"),
    }


def select_models(
    df: pd.DataFrame,
    *,
    n_total: int,
    ref_period: int,
    target_periods: List[int],
    bins: int,
    delta_z: float,
    max_per_base: int,
    max_per_org: int,
    include_families: List[str],
    min_per_family: int,
) -> pd.DataFrame:
    df = df.copy()
    df["model_id"] = df["model_id"].astype(str)
    df["base_model_id"] = df["base_model_id"].astype(str)
    df["org"] = df["org"].astype(str)
    df["period_idx"] = pd.to_numeric(df["period_idx"], errors="coerce")

    # Candidate pools
    df_target = df[df["period_idx"].isin([int(p) for p in target_periods])].copy()
    if df_target.empty:
        raise RuntimeError(f"No rows found for target_periods={target_periods}.")
    df_ref_primary = df[df["period_idx"] == int(ref_period)].copy()
    if df_ref_primary.empty:
        raise RuntimeError(f"No rows found for ref_period=P{ref_period}.")
    df_ref_fallback = pd.DataFrame()
    if int(ref_period) > 1:
        df_ref_fallback = df[df["period_idx"] == int(ref_period) - 1].copy()

    # Compute bins on the target pool; assign bin_id to ALL rows so B/C can be compared on same bins.
    edges = _compute_bin_edges_qcut(df_target["z"], int(bins))
    if edges.size < 2:
        raise RuntimeError("Could not compute compute-bin edges (insufficient unique z values).")
    n_bins_eff = int(edges.size - 1)
    df["bin_id"] = _assign_bins(df["z"], edges)
    df_target["bin_id"] = _assign_bins(df_target["z"], edges)
    df_ref_primary["bin_id"] = _assign_bins(df_ref_primary["z"], edges)
    if not df_ref_fallback.empty:
        df_ref_fallback["bin_id"] = _assign_bins(df_ref_fallback["z"], edges)

    n_a, n_b, n_c, n_d = _compute_group_sizes(int(n_total))
    caps = SelectionCaps(max_per_base=int(max_per_base), max_per_org=int(max_per_org))
    state = SelectionState(caps)

    # Helper: stable candidate ordering
    def _sort_desc_resid(frame: pd.DataFrame) -> pd.DataFrame:
        return frame.sort_values(["residual_ref", "model_id"], ascending=[False, True], kind="mergesort")

    def _sort_asc_resid(frame: pd.DataFrame) -> pd.DataFrame:
        return frame.sort_values(["residual_ref", "model_id"], ascending=[True, True], kind="mergesort")

    # -----------------------------
    # Group A: frontier advancers
    # -----------------------------
    patterns_lower = [p.strip().lower() for p in include_families if p.strip()]
    required_any = lambda mid: _matches_any_family(mid, patterns_lower)

    df_a = _sort_desc_resid(df_target)
    weights = _default_bin_weights(n_bins_eff)
    quota = _allocate_quotas(int(n_a), weights)

    selected_a: List[Dict[str, object]] = []
    for b in range(n_bins_eff):
        if int(quota[b]) <= 0:
            continue
        df_b = df_a[df_a["bin_id"] == b]
        if df_b.empty:
            continue
        for _, row in df_b.iterrows():
            if len(selected_a) >= int(n_a):
                break
            if sum(1 for r in selected_a if int(r.get("bin_id", -1)) == b) >= int(quota[b]):
                break
            model_id = str(row["model_id"])
            base_model_id = str(row["base_model_id"])
            org = str(row["org"])
            if not state.can_add(model_id, base_model_id, org):
                continue
            rec = _row_to_record(
                row,
                selection_group="A_frontier_advancer",
                rationale="High exceedance above P{ref} sigmoid boundary; bin-stratified".format(ref=int(ref_period)),
            )
            state.add_record(rec)
            selected_a.append(rec)

    # Fill any remaining A slots (global residual order)
    if len(selected_a) < int(n_a):
        for _, row in df_a.iterrows():
            if len(selected_a) >= int(n_a):
                break
            model_id = str(row["model_id"])
            base_model_id = str(row["base_model_id"])
            org = str(row["org"])
            if not state.can_add(model_id, base_model_id, org):
                continue
            rec = _row_to_record(
                row,
                selection_group="A_frontier_advancer",
                rationale="High exceedance above P{ref} sigmoid boundary; filled after bin pass".format(ref=int(ref_period)),
            )
            state.add_record(rec)
            selected_a.append(rec)

    # Soft family coverage top-up (replace lowest-priority A picks not in any required family).
    if patterns_lower and int(min_per_family) > 0:
        required_set = list(patterns_lower)

        def _family_count(records: List[Dict[str, object]], pat: str) -> int:
            return sum(1 for r in records if pat in str(r.get("model_id", "")).lower())

        df_a_all = _sort_desc_resid(df_target)
        non_required_a = sorted(
            [r for r in selected_a if not required_any(str(r.get("model_id", "")))],
            key=lambda r: (float(r.get("residual_ref", float("inf"))), str(r.get("model_id", ""))),
        )

        for pat in required_set:
            need = int(min_per_family) - _family_count(selected_a, pat)
            if need <= 0:
                continue
            cand_pat = df_a_all[df_a_all["model_id"].str.lower().str.contains(pat, na=False)]
            cand_pat = _sort_desc_resid(cand_pat)
            for _, row in cand_pat.iterrows():
                if need <= 0:
                    break
                model_id = str(row["model_id"])
                base_model_id = str(row["base_model_id"])
                org = str(row["org"])
                if model_id in state.selected_ids:
                    continue
                # If we still have space in A (rare), add directly.
                if len(selected_a) < int(n_a) and state.can_add(model_id, base_model_id, org):
                    rec = _row_to_record(
                        row,
                        selection_group="A_frontier_advancer",
                        rationale=f"Family coverage top-up for pattern={pat!r}",
                    )
                    state.add_record(rec)
                    selected_a.append(rec)
                    need -= 1
                    continue
                # Otherwise, attempt replacement.
                replaced = False
                for drop in list(non_required_a):
                    state.remove_record(drop)
                    if state.can_add(model_id, base_model_id, org):
                        rec = _row_to_record(
                            row,
                            selection_group="A_frontier_advancer",
                            rationale=f"Family coverage top-up for pattern={pat!r} (replaced {drop.get('model_id','')})",
                        )
                        state.add_record(rec)
                        selected_a.remove(drop)
                        selected_a.append(rec)
                        non_required_a.remove(drop)
                        # newly added might still be non-required for other patterns (unlikely), but keep list conservative
                        if not required_any(model_id):
                            non_required_a.append(rec)
                        replaced = True
                        need -= 1
                        break
                    state.add_record(drop)  # revert
                if not replaced:
                    break

    # -----------------------------
    # Group B: matched baselines (ref period)
    # -----------------------------
    selected_b: List[Dict[str, object]] = []

    def _best_baseline_for_a(z_a: float, df_ref_pool: pd.DataFrame, *, max_delta: float) -> Optional[pd.Series]:
        pool = df_ref_pool[np.abs(df_ref_pool["z"] - float(z_a)) <= float(max_delta)].copy()
        if pool.empty:
            return None
        pool = pool.sort_values(["y_math", "model_id"], ascending=[False, True], kind="mergesort")
        for _, r in pool.iterrows():
            model_id = str(r["model_id"])
            base_model_id = str(r["base_model_id"])
            org = str(r["org"])
            if state.can_add(model_id, base_model_id, org):
                return r
        return None

    deltas = [float(delta_z), 0.10, 0.15]
    deltas = sorted(set([d for d in deltas if d > 0]))
    selected_a_sorted = sorted(selected_a, key=lambda r: float(r.get("residual_ref", -np.inf)), reverse=True)

    def _fill_baselines_from_pool(df_ref_pool: pd.DataFrame, pool_label: str) -> None:
        for max_delta in deltas:
            if len(selected_b) >= int(n_b):
                return
            for a_rec in selected_a_sorted:
                if len(selected_b) >= int(n_b):
                    return
                z_a = float(a_rec.get("z", float("nan")))
                if not np.isfinite(z_a):
                    continue
                best = _best_baseline_for_a(z_a, df_ref_pool, max_delta=max_delta)
                if best is None:
                    continue
                model_id = str(best["model_id"])
                base_model_id = str(best["base_model_id"])
                org = str(best["org"])
                if not state.can_add(model_id, base_model_id, org):
                    continue
                dz = float(abs(float(best["z"]) - z_a))
                rec = _row_to_record(
                    best,
                    selection_group="B_matched_baseline",
                    rationale=(
                        f"Compute-matched baseline to {a_rec.get('model_id','')} within ±{max_delta:.2f} log10 FLOPs "
                        f"(source={pool_label})"
                    ),
                    matched_to_model_id=str(a_rec.get("model_id", "")),
                    matched_delta_z=dz,
                )
                state.add_record(rec)
                selected_b.append(rec)

    _fill_baselines_from_pool(df_ref_primary, f"P{int(ref_period)}")
    if len(selected_b) < int(n_b) and (not df_ref_fallback.empty):
        _fill_baselines_from_pool(df_ref_fallback, f"P{int(ref_period) - 1}")

    # If still short: nearest-neighbor fallback.
    if len(selected_b) < int(n_b):
        df_nn_pool = (
            pd.concat([df_ref_primary, df_ref_fallback], ignore_index=True)
            if not df_ref_fallback.empty
            else df_ref_primary
        )
        for a_rec in selected_a_sorted:
            if len(selected_b) >= int(n_b):
                break
            z_a = float(a_rec.get("z", float("nan")))
            if not np.isfinite(z_a):
                continue
            pool = df_nn_pool.copy()
            pool["abs_dz"] = np.abs(pool["z"] - z_a)
            pool = pool.sort_values(["abs_dz", "y_math", "model_id"], ascending=[True, False, True], kind="mergesort")
            for _, r in pool.iterrows():
                model_id = str(r["model_id"])
                base_model_id = str(r["base_model_id"])
                org = str(r["org"])
                if not state.can_add(model_id, base_model_id, org):
                    continue
                dz = float(r["abs_dz"])
                rec = _row_to_record(
                    r,
                    selection_group="B_matched_baseline",
                    rationale=f"Nearest-z baseline to {a_rec.get('model_id','')} (fallback)",
                    matched_to_model_id=str(a_rec.get("model_id", "")),
                    matched_delta_z=dz,
                )
                state.add_record(rec)
                selected_b.append(rec)
                break

    # -----------------------------
    # Group C: below-frontier controls (target periods)
    # -----------------------------
    selected_c: List[Dict[str, object]] = []
    df_c_pool = df_target.copy()
    # Prefer truly below-boundary controls; if too few, fall back to bottom-half residuals.
    df_c_below = df_c_pool[df_c_pool["residual_ref"] <= 0.0].copy()
    if df_c_below.shape[0] < int(n_c):
        thr = float(df_c_pool["residual_ref"].quantile(0.5))
        df_c_below = df_c_pool[df_c_pool["residual_ref"] <= thr].copy()
    df_c_below = _sort_asc_resid(df_c_below)

    # Match the bin distribution of Group A as closely as possible.
    count_a_bin = Counter(int(r.get("bin_id", -1)) for r in selected_a if int(r.get("bin_id", -1)) >= 0)
    counts = np.array([count_a_bin.get(b, 0) for b in range(n_bins_eff)], dtype=float)
    if float(np.sum(counts)) <= 0:
        counts = np.ones(n_bins_eff, dtype=float)
    quota_c = _allocate_quotas(int(n_c), counts)

    for b in range(n_bins_eff):
        if int(quota_c[b]) <= 0:
            continue
        df_b = df_c_below[df_c_below["bin_id"] == b]
        if df_b.empty:
            continue
        for _, row in df_b.iterrows():
            if len(selected_c) >= int(n_c):
                break
            if sum(1 for r in selected_c if int(r.get("bin_id", -1)) == b) >= int(quota_c[b]):
                break
            model_id = str(row["model_id"])
            base_model_id = str(row["base_model_id"])
            org = str(row["org"])
            if not state.can_add(model_id, base_model_id, org):
                continue
            rec = _row_to_record(
                row,
                selection_group="C_below_frontier_control",
                rationale="Compute-matched control (below prior frontier); bin distribution matched to Group A",
            )
            state.add_record(rec)
            selected_c.append(rec)

    if len(selected_c) < int(n_c):
        for _, row in df_c_below.iterrows():
            if len(selected_c) >= int(n_c):
                break
            model_id = str(row["model_id"])
            base_model_id = str(row["base_model_id"])
            org = str(row["org"])
            if not state.can_add(model_id, base_model_id, org):
                continue
            rec = _row_to_record(
                row,
                selection_group="C_below_frontier_control",
                rationale="Compute-matched control (below prior frontier); filled after bin pass",
            )
            state.add_record(rec)
            selected_c.append(rec)

    # -----------------------------
    # Group D: fill/diversity anchors
    # -----------------------------
    selected_d: List[Dict[str, object]] = []
    df_remaining = df[~df["model_id"].isin(list(state.selected_ids))].copy()
    n_d_effective = int(max(0, int(n_total) - len(state.selected)))
    if not df_remaining.empty and n_d_effective > 0:
        df_remaining["is_target_period"] = df_remaining["period_idx"].isin([int(p) for p in target_periods]).astype(int)
        df_remaining["abs_resid"] = np.abs(df_remaining["residual_ref"].to_numpy())

        remaining_rows = [r for _, r in df_remaining.iterrows()]
        for _ in range(n_d_effective):
            best_row: Optional[pd.Series] = None
            best_key: Optional[Tuple] = None
            for row in remaining_rows:
                model_id = str(row["model_id"])
                base_model_id = str(row["base_model_id"])
                org = str(row["org"])
                if not state.can_add(model_id, base_model_id, org):
                    continue
                is_target = int(row.get("is_target_period", 0))
                base_ct = int(state.base_counts.get(base_model_id, 0))
                org_ct = int(state.org_counts.get(org, 0))
                abs_resid = float(row.get("abs_resid", float("inf")))
                key = (-is_target, base_ct, org_ct, abs_resid, model_id)
                if best_key is None or key < best_key:
                    best_key = key
                    best_row = row
            if best_row is None:
                break
            model_id = str(best_row["model_id"])
            base_model_id = str(best_row["base_model_id"])
            org = str(best_row["org"])
            base_ct = int(state.base_counts.get(base_model_id, 0))
            org_ct = int(state.org_counts.get(org, 0))
            rationale = "Diversity/coverage fill to reach N_total"
            if base_ct == 0:
                rationale += " (new base_model_id)"
            if org_ct == 0:
                rationale += " (new org)"
            if int(best_row.get("is_target_period", 0)) == 0:
                rationale += " (non-target fallback)"
            rec = _row_to_record(best_row, selection_group="D_fill", rationale=rationale)
            state.add_record(rec)
            selected_d.append(rec)
            remaining_rows = [r for r in remaining_rows if str(r["model_id"]) != model_id]

    selected_df = pd.DataFrame(state.selected)
    # Stable output ordering: by group then residual_ref desc then model_id.
    group_order = {
        "A_frontier_advancer": 0,
        "B_matched_baseline": 1,
        "C_below_frontier_control": 2,
        "D_fill": 3,
    }
    selected_df["_g"] = selected_df["selection_group"].map(lambda g: group_order.get(str(g), 99))
    selected_df = selected_df.sort_values(["_g", "residual_ref", "model_id"], ascending=[True, False, True], kind="mergesort")
    selected_df = selected_df.drop(columns=["_g"], errors="ignore")
    return selected_df.reset_index(drop=True)


def main(argv: Optional[Sequence[str]] = None) -> None:
    ap = argparse.ArgumentParser(description="Select models for AIME 2025 contamination-check evaluation (selection only).")
    ap.add_argument("--task", default="MATH Lvl 5 Raw", help="Canonical task name (e.g., 'MATH Lvl 5 Raw').")
    ap.add_argument("--tau", type=float, default=0.98, help="Sigmoid frontier quantile tau.")
    ap.add_argument("--kappa", type=float, default=50.0, help="Smooth pinball sharpness kappa.")
    ap.add_argument("--lambda_reg", type=float, default=1e-3, help="Ridge penalty on sigmoid slope b (lambda_b).")
    ap.add_argument("--n_total", type=int, default=100, help="Total number of models to select.")
    ap.add_argument("--seed", type=int, default=0, help="RNG seed (used for fitting init; selection is deterministic).")
    ap.add_argument("--ref_period", default="P3", help="Reference period (e.g., 'P3' or '3').")
    ap.add_argument("--target_periods", default="P4,P5", help="Comma-separated target periods (e.g., 'P4,P5').")
    ap.add_argument("--bins", type=int, default=10, help="Number of compute bins for stratified selection.")
    ap.add_argument("--delta_z", type=float, default=0.07, help="Compute matching tolerance in log10 FLOPs.")
    ap.add_argument("--max_per_base", type=int, default=2, help="Max selected per base_model_id.")
    ap.add_argument("--max_per_org", type=int, default=4, help="Max selected per org (HF namespace).")
    ap.add_argument("--out_csv", default=os.path.join("outputs", "aime2025_selection", "models_math_lvl5.csv"))
    ap.add_argument("--out_json", default=os.path.join("outputs", "aime2025_selection", "models_math_lvl5.json"))
    ap.add_argument(
        "--include_families",
        default="Nemotron,OpenThinker,Olmo",
        help="Comma-separated family substrings to (soft) enforce coverage for.",
    )
    ap.add_argument("--min_per_family", type=int, default=5, help="Minimum per requested family substring if possible.")
    args = ap.parse_args(list(argv) if argv is not None else None)

    ref_period = _parse_period_token(args.ref_period)
    if ref_period is None:
        raise SystemExit(f"Could not parse --ref_period={args.ref_period!r}")
    target_periods = _parse_period_list(args.target_periods)
    if not target_periods:
        raise SystemExit(f"Could not parse --target_periods={args.target_periods!r}")

    include_families = [p.strip() for p in str(args.include_families).split(",") if p.strip()]

    df = load_model_table(task=str(args.task), include_new_lb=True, compute_multiplier=6.0)
    # Filter to relevant periods (ref + targets) for fitting/selection diagnostics.
    relevant_periods = sorted(set([int(ref_period), *[int(p) for p in target_periods]]))
    df_rel = df[df["period_idx"].isin(relevant_periods)].copy()
    if df_rel.empty:
        raise SystemExit("No rows after restricting to ref/target periods.")

    # Fit reference sigmoid on ref_period using existing MATH scores.
    df_ref = df_rel[df_rel["period_idx"] == int(ref_period)].copy()
    if df_ref.shape[0] < 10:
        raise SystemExit(f"Not enough reference-period rows to fit frontier for P{ref_period} (n={int(df_ref.shape[0])}).")
    params_ref = fit_sigmoid_params(
        df_ref["z"].to_numpy(),
        df_ref["y_math"].to_numpy(),
        tau=float(args.tau),
        kappa_final=float(args.kappa),
        lambda_b=float(args.lambda_reg),
        seed=int(args.seed),
    )
    df_rel["pred_ref"] = predict_sigmoid(df_rel["z"].to_numpy(), params_ref)
    df_rel["residual_ref"] = df_rel["y_math"].to_numpy(dtype=float) - df_rel["pred_ref"].to_numpy(dtype=float)

    # Optional MMLU specialist/generalist signals.
    have_mmlu = "y_mmlu" in df_rel.columns and np.isfinite(df_ref["y_mmlu"]).sum() >= 10
    if have_mmlu:
        mmlu_ref = df_ref[np.isfinite(df_ref["y_mmlu"])].copy()
        try:
            params_mmlu_ref = fit_sigmoid_params(
                mmlu_ref["z"].to_numpy(),
                mmlu_ref["y_mmlu"].to_numpy(),
                tau=float(args.tau),
                kappa_final=float(args.kappa),
                lambda_b=float(args.lambda_reg),
                seed=int(args.seed),
            )
            df_rel["pred_mmlu_ref"] = predict_sigmoid(df_rel["z"].to_numpy(), params_mmlu_ref)
            df_rel["residual_mmlu_ref"] = df_rel["y_mmlu"].to_numpy(dtype=float) - df_rel["pred_mmlu_ref"].to_numpy(dtype=float)
            df_rel["specialist_score"] = df_rel["residual_ref"] - df_rel["residual_mmlu_ref"]
        except Exception:
            df_rel["pred_mmlu_ref"] = np.nan
            df_rel["residual_mmlu_ref"] = np.nan
            df_rel["specialist_score"] = np.nan
    else:
        df_rel["pred_mmlu_ref"] = np.nan
        df_rel["residual_mmlu_ref"] = np.nan
        df_rel["specialist_score"] = np.nan

    selected = select_models(
        df_rel,
        n_total=int(args.n_total),
        ref_period=int(ref_period),
        target_periods=[int(p) for p in target_periods],
        bins=int(args.bins),
        delta_z=float(args.delta_z),
        max_per_base=int(args.max_per_base),
        max_per_org=int(args.max_per_org),
        include_families=include_families,
        min_per_family=int(args.min_per_family),
    )

    out_csv = str(args.out_csv)
    out_json = str(args.out_json)
    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(out_json) or ".", exist_ok=True)
    selected.to_csv(out_csv, index=False)

    # Diagnostics + JSON payload.
    counts_group = selected["selection_group"].value_counts().to_dict()
    counts_period = selected["period"].value_counts().to_dict()
    counts_org = selected["org"].value_counts().head(10).to_dict()
    counts_base = selected["base_model_id"].value_counts().head(10).to_dict()

    # Bin histogram: selected vs target pool.
    df_target = df_rel[df_rel["period_idx"].isin([int(p) for p in target_periods])].copy()
    # Ensure bin_id exists by reusing the same qcut edges as selection used.
    edges = _compute_bin_edges_qcut(df_target["z"], int(args.bins))
    df_target["bin_id"] = _assign_bins(df_target["z"], edges)
    selected_bins = selected["bin_id"].value_counts().sort_index().to_dict()
    target_bins = df_target["bin_id"].value_counts().sort_index().to_dict()

    fam_counts: Dict[str, int] = {}
    for pat in [p.strip() for p in include_families if p.strip()]:
        fam_counts[pat] = int(selected["model_id"].astype(str).str.contains(pat, case=False, na=False).sum())

    payload = {
        "config": {
            "task": str(args.task),
            "tau": float(args.tau),
            "kappa": float(args.kappa),
            "lambda_reg": float(args.lambda_reg),
            "n_total": int(args.n_total),
            "seed": int(args.seed),
            "ref_period": f"P{int(ref_period)}",
            "target_periods": [f"P{int(p)}" for p in target_periods],
            "bins": int(args.bins),
            "delta_z": float(args.delta_z),
            "max_per_base": int(args.max_per_base),
            "max_per_org": int(args.max_per_org),
            "include_families": include_families,
            "min_per_family": int(args.min_per_family),
            "out_csv": out_csv,
            "out_json": out_json,
        },
        "data_sources": {
            "oll_v2_with_tokens": OLL_V2_WITH_TOKENS,
            "new_leaderboard_results_with_tokens": NEW_LB_WITH_TOKENS if os.path.exists(NEW_LB_WITH_TOKENS) else None,
        },
        "period4_bounds": PERIOD4_BOUNDS,
        "sigmoid_ref": {
            "paramization": "yhat = y0 + L * sigmoid(b*(z - z_star)), z=log10(pretrain_compute_zflops)",
            "raw_params": {"y0": params_ref.y0, "L": params_ref.L, "z_star": params_ref.z_star, "log_b": params_ref.log_b},
            "theta_y0_L_a_b": params_ref.as_theta_y0_L_a_b(),
        },
        "summary": {
            "n_selected": int(selected.shape[0]),
            "counts_by_group": counts_group,
            "counts_by_period": counts_period,
            "top_org": counts_org,
            "top_base_model_id": counts_base,
            "bin_hist_selected": {str(k): int(v) for k, v in selected_bins.items()},
            "bin_hist_target_pool": {str(k): int(v) for k, v in target_bins.items()},
            "family_coverage": fam_counts,
        },
        "selected_models": selected.to_dict(orient="records"),
    }
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")

    print(f"[ok] wrote {out_csv}")
    print(f"[ok] wrote {out_json}")
    print(f"[summary] n_selected={int(selected.shape[0])} (target={int(args.n_total)})")
    print(f"[summary] counts_by_group={counts_group}")
    print(f"[summary] counts_by_period={counts_period}")
    print(f"[summary] top_org={counts_org}")
    print(f"[summary] top_base_model_id={counts_base}")
    print(f"[summary] family_coverage={fam_counts}")
    print(f"[summary] bin_hist_selected={ {str(k): int(v) for k, v in selected_bins.items()} }")
    print(f"[summary] bin_hist_target_pool={ {str(k): int(v) for k, v in target_bins.items()} }")

    min_ok = int(math.floor(0.9 * float(args.n_total)))
    if int(selected.shape[0]) < min_ok:
        raise SystemExit(
            f"Selected only {int(selected.shape[0])} models (< 0.9*N={min_ok}); "
            "data scarcity or caps too strict. See JSON for diagnostics."
        )


if __name__ == "__main__":
    main()

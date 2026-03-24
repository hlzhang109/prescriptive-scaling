#!/usr/bin/env python3
"""
Merge LiveBench wide results with numeric metadata into an Option-A-style merged CSV.

Inputs:
  - --wide_csv: tables/livebench/livebench_subcategory_results_wide.csv
  - --metadata_csv: tables/livebench/livebench_model_metadata_numeric.csv

Behavior (mirrors Option B selection so downstream Option A is clean):
  - Keep tasks with ≥ --min_models_per_task valid entries
  - Keep models with ≥ --min_values_per_row valid values among kept tasks
  - Join metadata; compute logC = log(pretrain_compute_zflops + eps); drop rows missing compute
  - Impute remaining missing values per task by the task median across retained rows
  - Clip values into [0,1]
  - Write merged CSV with columns: model, logC, a_<task1>, ..., a_<taskN>

Example:
  python scripts/data/merge_livebench.py \
    --wide_csv tables/livebench/livebench_subcategory_results_wide.csv \
    --metadata_csv tables/livebench/livebench_model_metadata_numeric.csv \
    --out_csv tables/merged_livebench.csv \
    --min_models_per_task 20 --min_values_per_row 5
"""

from __future__ import annotations

import argparse
import csv as _csv
import math
import os
from typing import List, Tuple

import numpy as np


def _sanitize_task_name(name: str) -> str:
    # Keep alnum and underscores; replace others with '_'
    out = []
    for ch in name:
        if ch.isalnum() or ch == "_":
            out.append(ch)
        else:
            out.append("_")
    return "".join(out)


def _read_wide_csv(path: str) -> Tuple[List[str], List[str], np.ndarray]:
    with open(path, "r", newline="") as f:
        reader = _csv.DictReader(f)
        rows = list(reader)
    if not rows:
        raise ValueError(f"No rows in {path}")
    all_cols = list(rows[0].keys())
    if "model" not in all_cols:
        raise ValueError("Expected 'model' column in wide CSV")
    task_cols = [c for c in all_cols if c != "model"]
    models: List[str] = []
    A_rows: List[List[float]] = []
    for r in rows:
        m = str(r.get("model", "") or "").strip()
        if not m:
            continue
        vec: List[float] = []
        for c in task_cols:
            v_raw = r.get(c, None)
            if v_raw in (None, "", "nan", "NaN"):
                vec.append(float("nan"))
            else:
                try:
                    vec.append(float(v_raw))
                except Exception:
                    vec.append(float("nan"))
        models.append(m)
        A_rows.append(vec)
    return models, task_cols, np.array(A_rows, dtype=float)


def _read_metadata_compute(path: str) -> dict:
    with open(path, "r", newline="") as f:
        reader = _csv.DictReader(f)
        rows = list(reader)
    if not rows:
        raise ValueError(f"No rows in {path}")
    if "model" not in rows[0] or "pretrain_compute_zflops" not in rows[0]:
        raise ValueError("Expected columns 'model' and 'pretrain_compute_zflops' in metadata CSV")
    mp = {}
    for r in rows:
        m = str(r.get("model", "") or "").strip()
        v = r.get("pretrain_compute_zflops", None)
        try:
            c = float(v) if v not in (None, "", "nan", "NaN") else float("nan")
        except Exception:
            c = float("nan")
        if m:
            mp[m] = c
    return mp


def main(argv: List[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Merge LiveBench wide results + numeric metadata to Option-A merged CSV.")
    p.add_argument("--wide_csv", required=True, help="Path to livebench_subcategory_results_wide.csv")
    p.add_argument("--metadata_csv", required=True, help="Path to livebench_model_metadata_numeric.csv")
    p.add_argument("--out_csv", default="tables/merged_livebench.csv", help="Output merged CSV path")
    p.add_argument("--min_models_per_task", type=int, default=20, help="Min models per task to keep the task")
    p.add_argument("--min_values_per_row", type=int, default=5, help="Min kept-task values per model to keep the row")
    p.add_argument("--task_prefix", type=str, default="a_", help="Prefix for task columns in output merged CSV")
    p.add_argument("--clip01", action="store_true", help="Clip task values to [0,1] in output")
    args = p.parse_args(argv)

    models_raw, task_cols_raw, A_raw = _read_wide_csv(args.wide_csv)
    # Task selection by coverage
    non_missing_counts = np.sum(np.isfinite(A_raw), axis=0)
    keep_task_mask = non_missing_counts >= max(1, args.min_models_per_task)
    kept_tasks_raw = [t for t, k in zip(task_cols_raw, keep_task_mask) if k]
    if not kept_tasks_raw:
        raise ValueError("No tasks meet coverage threshold. Consider lowering --min_models_per_task.")
    A_kept = A_raw[:, keep_task_mask]

    # Row selection by valid counts
    valid_counts = np.sum(np.isfinite(A_kept), axis=1)
    row_ok = valid_counts >= max(1, args.min_values_per_row)
    models_kept = [m for m, ok in zip(models_raw, row_ok) if ok]
    A_kept = A_kept[row_ok, :]

    # Join compute metadata
    comp_map = _read_metadata_compute(args.metadata_csv)
    eps = 1e-12
    models_final: List[str] = []
    logC_list: List[float] = []
    A_fin_nans: List[List[float]] = []
    for m, row in zip(models_kept, A_kept.tolist()):
        c = comp_map.get(m, float("nan"))
        if not np.isfinite(c):
            continue
        models_final.append(m)
        logC_list.append(float(math.log(c + eps)))
        A_fin_nans.append(row)
    if not models_final:
        raise ValueError("After join, no models have valid compute. Check metadata CSV.")

    # Impute missing per task by median over retained rows; optional clipping
    A = np.array(A_fin_nans, dtype=float)
    for j in range(A.shape[1]):
        col = A[:, j]
        mask = np.isfinite(col)
        med = float(np.nanmedian(col)) if mask.any() else 0.0
        col[~mask] = med
        if args.clip01:
            col = np.clip(col, 0.0, 1.0)
        A[:, j] = col

    # Build output header and write
    os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)
    safe_tasks = [_sanitize_task_name(t) for t in kept_tasks_raw]
    out_task_cols = [f"{args.task_prefix}{t}" for t in safe_tasks]
    with open(args.out_csv, "w", newline="") as f:
        w = _csv.writer(f)
        w.writerow(["model", "logC", *out_task_cols])
        for i, m in enumerate(models_final):
            w.writerow([m, f"{logC_list[i]:.12g}", *[f"{float(x):.12g}" for x in A[i, :].tolist()]])

    print(f"Wrote merged CSV with {len(models_final)} models and {len(out_task_cols)} tasks:")
    print(f"  - {args.out_csv}")


if __name__ == "__main__":
    main()

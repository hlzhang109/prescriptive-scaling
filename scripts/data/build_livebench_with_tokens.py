#!/usr/bin/env python3
"""
Build a LiveBench "with tokens" CSV suitable for sigmoid frontier plotting.

Inputs:
  - --wide_csv: tables/livebench/livebench_subcategory_results_wide.csv
  - --metadata_csv: tables/livebench/livebench_model_metadata_numeric.csv

Output:
  - --out_csv (default): tables/livebench/livebench_with_tokens.csv
    Columns:
      - model
      - Pretraining tokens (T)
      - #Params (B)
      - <task1>, <task2>, ... (tasks with >= --min_records models having both score and compute)
  - --out_task_counts_json (default): tables/livebench/livebench_task_counts.json
  - --out_tasks_txt (default): tables/livebench/livebench_tasks_min<min_records>.txt
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List

import pandas as pd


TOKENS_T_COL = "Pretraining tokens (T)"
PARAMS_B_COL = "#Params (B)"


def main() -> None:
    p = argparse.ArgumentParser(description="Join LiveBench results with params/tokens metadata for plotting.")
    p.add_argument("--wide_csv", required=True, help="LiveBench wide results CSV")
    p.add_argument("--metadata_csv", required=True, help="LiveBench numeric model metadata CSV")
    p.add_argument(
        "--min_records",
        type=int,
        default=30,
        help="Minimum #models per task with known compute + score to keep the task",
    )
    p.add_argument(
        "--out_csv",
        default=os.path.join("tables", "livebench", "livebench_with_tokens.csv"),
        help="Output CSV path",
    )
    p.add_argument(
        "--out_task_counts_json",
        default=os.path.join("tables", "livebench", "livebench_task_counts.json"),
        help="Output JSON with per-task valid counts",
    )
    p.add_argument(
        "--out_tasks_txt",
        default=None,
        help="Optional output text file listing kept tasks (one per line). Defaults to tables/livebench/livebench_tasks_min<min_records>.txt",
    )
    args = p.parse_args()

    wide = pd.read_csv(args.wide_csv)
    meta = pd.read_csv(args.metadata_csv)

    if "model" not in wide.columns:
        raise ValueError("wide_csv must contain a 'model' column")
    if "model" not in meta.columns:
        raise ValueError("metadata_csv must contain a 'model' column")

    if "params_b" not in meta.columns or "pretrain_tokens" not in meta.columns:
        raise ValueError("metadata_csv must contain 'params_b' and 'pretrain_tokens' columns")

    df = wide.merge(meta[["model", "params_b", "pretrain_tokens"]], on="model", how="left")
    df[PARAMS_B_COL] = pd.to_numeric(df["params_b"], errors="coerce")
    df[TOKENS_T_COL] = pd.to_numeric(df["pretrain_tokens"], errors="coerce") / 1e12

    task_cols = [c for c in wide.columns if c != "model"]
    if not task_cols:
        raise ValueError("No task columns found in wide_csv")

    # Count valid records per task: needs score + known params + known tokens.
    counts: Dict[str, int] = {}
    keep_tasks: List[str] = []
    has_compute = df[PARAMS_B_COL].notna() & df[TOKENS_T_COL].notna()
    for t in task_cols:
        valid = has_compute & df[t].notna()
        n = int(valid.sum())
        counts[t] = n
        if n >= int(args.min_records):
            keep_tasks.append(t)

    os.makedirs(os.path.dirname(args.out_task_counts_json) or ".", exist_ok=True)
    with open(args.out_task_counts_json, "w") as f:
        json.dump({"min_records": int(args.min_records), "counts": counts}, f, indent=2, sort_keys=True)
        f.write("\n")

    if args.out_tasks_txt is None:
        args.out_tasks_txt = os.path.join(
            os.path.dirname(args.out_csv) or ".", f"livebench_tasks_min{int(args.min_records)}.txt"
        )
    os.makedirs(os.path.dirname(args.out_tasks_txt) or ".", exist_ok=True)
    with open(args.out_tasks_txt, "w") as f:
        for t in keep_tasks:
            f.write(f"{t}\n")

    if not keep_tasks:
        raise RuntimeError(
            f"No tasks have >= {int(args.min_records)} valid records with known compute. "
            f"See {args.out_task_counts_json}."
        )

    out_cols = ["model", TOKENS_T_COL, PARAMS_B_COL, *keep_tasks]
    out = df[out_cols].copy()
    # Keep only rows with compute and at least one kept task score
    any_task = out[keep_tasks].notna().any(axis=1)
    out = out[has_compute & any_task]

    os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)
    out.to_csv(args.out_csv, index=False)

    print("LiveBench with-tokens table built.")
    print(f"  - out_csv: {args.out_csv} ({len(out)} models)")
    print(f"  - kept tasks ({len(keep_tasks)}): {', '.join(keep_tasks)}")
    print(f"  - task counts: {args.out_task_counts_json}")
    print(f"  - tasks list:  {args.out_tasks_txt}")


if __name__ == "__main__":
    main()


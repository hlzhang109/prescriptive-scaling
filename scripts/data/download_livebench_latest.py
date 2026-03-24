#!/usr/bin/env python3
"""
Download latest LiveBench leaderboard judgments and rebuild derived tables.

Data source: Hugging Face dataset `livebench/model_judgment` (split: `leaderboard`).

Outputs (under --out_dir, default: tables/livebench):
  - leaderboard.parquet                     Raw judgments
  - livebench_subcategory_results_long.csv  Per-model per-task mean scores (long)
  - livebench_subcategory_results_wide.csv  Per-model mean scores (wide)
  - livebench_category_and_overall.csv      Category means + overall mean-of-categories
  - livebench_download_info.json            Download metadata for reproducibility
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
from typing import Dict, Optional

import pandas as pd
from datasets import load_dataset
from huggingface_hub import HfApi


TASK_TO_METRIC: Dict[str, str] = {
    "LCB_generation": "pass@1",
    "coding_completion": "pass@1",
    "connections": "accuracy",
    "paraphrase": "combined_score",
    "plot_unscrambling": "score",
    "story_generation": "combined_score",
    "typos": "accuracy",
}


def _safe_mkdir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _dataset_sha(dataset_id: str, revision: Optional[str]) -> Optional[str]:
    try:
        info = HfApi().dataset_info(dataset_id, revision=revision)
        return getattr(info, "sha", None)
    except Exception:
        return None


def main() -> None:
    p = argparse.ArgumentParser(description="Download latest LiveBench judgments and rebuild derived tables.")
    p.add_argument("--dataset_id", default="livebench/model_judgment", help="Hugging Face dataset id")
    p.add_argument("--split", default="leaderboard", help="Dataset split")
    p.add_argument(
        "--revision",
        default=None,
        help="Optional dataset revision (commit SHA or tag) to pin; default downloads latest",
    )
    p.add_argument("--out_dir", default=os.path.join("tables", "livebench"), help="Output directory")
    args = p.parse_args()

    _safe_mkdir(args.out_dir)

    ds = load_dataset(args.dataset_id, split=args.split, revision=args.revision)
    df = ds.to_pandas()

    required_cols = {"question_id", "task", "model", "score", "category"}
    missing = sorted(required_cols.difference(df.columns))
    if missing:
        raise RuntimeError(f"Dataset missing required columns: {missing}")

    # Raw dump
    out_parquet = os.path.join(args.out_dir, "leaderboard.parquet")
    df.to_parquet(out_parquet, index=False)

    # Aggregate: per-model, per-task mean score
    agg = (
        df.groupby(["model", "category", "task"], as_index=False)["score"]
        .mean()
        .rename(columns={"score": "value"})
    )
    agg["metric"] = agg["task"].map(TASK_TO_METRIC).fillna("score")

    out_long = os.path.join(args.out_dir, "livebench_subcategory_results_long.csv")
    agg[["model", "category", "task", "metric", "value"]].to_csv(out_long, index=False)

    out_wide = os.path.join(args.out_dir, "livebench_subcategory_results_wide.csv")
    wide = agg.pivot_table(index="model", columns="task", values="value", aggfunc="first").reset_index()
    wide.to_csv(out_wide, index=False)

    # Category means and overall (mean-of-categories)
    cat = (
        agg.groupby(["model", "category"], as_index=False)["value"]
        .mean()
        .rename(columns={"value": "category_mean"})
    )
    overall = cat.groupby("model", as_index=False)["category_mean"].mean().rename(
        columns={"category_mean": "overall_mean_of_categories"}
    )
    cat = cat.merge(overall, on="model", how="left")
    out_cat = os.path.join(args.out_dir, "livebench_category_and_overall.csv")
    cat.to_csv(out_cat, index=False)

    info = {
        "dataset_id": args.dataset_id,
        "split": args.split,
        "revision_requested": args.revision,
        "dataset_sha": _dataset_sha(args.dataset_id, args.revision),
        "downloaded_at_utc": _dt.datetime.now(tz=_dt.timezone.utc).isoformat(),
        "n_rows": int(df.shape[0]),
        "n_models": int(df["model"].nunique()),
        "n_tasks": int(df["task"].nunique()),
        "n_categories": int(df["category"].nunique()),
        "tasks": sorted(map(str, df["task"].unique().tolist())),
        "categories": sorted(map(str, df["category"].unique().tolist())),
        "models": sorted(map(str, df["model"].unique().tolist())),
    }
    out_info = os.path.join(args.out_dir, "livebench_download_info.json")
    with open(out_info, "w") as f:
        json.dump(info, f, indent=2, sort_keys=True)
        f.write("\n")

    print("LiveBench download complete.")
    print(f"  - raw:      {out_parquet}")
    print(f"  - long:     {out_long}")
    print(f"  - wide:     {out_wide}")
    print(f"  - category: {out_cat}")
    print(f"  - info:     {out_info}")
    print(f"Tasks ({info['n_tasks']}): {', '.join(info['tasks'])}")
    print(f"Models: {info['n_models']}, Rows: {info['n_rows']}")


if __name__ == "__main__":
    main()

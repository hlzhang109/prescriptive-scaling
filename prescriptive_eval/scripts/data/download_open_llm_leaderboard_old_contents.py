#!/usr/bin/env python3
"""
Download the full *v1 / old* Open LLM Leaderboard table from Hugging Face.

Authoritative source:
  - dataset repo: open-llm-leaderboard-old/contents

We validate the column names to ensure we downloaded the correct table
(v1 tasks: ARC/HellaSwag/MMLU/TruthfulQA/Winogrande/GSM8K).
"""

from __future__ import annotations

import argparse
import datetime as _dt
import shutil
from pathlib import Path
from typing import Optional

import pandas as pd
from huggingface_hub import HfApi, hf_hub_download


EXPECTED_COLUMNS_V1: list[str] = [
    "eval_name",
    "Precision",
    "Type",
    "T",
    "Weight type",
    "Architecture",
    "Model",
    "fullname",
    "Model sha",
    "Average ⬆️",
    "Hub License",
    "Hub ❤️",
    "#Params (B)",
    "Available on the hub",
    "Merged",
    "MoE",
    "Flagged",
    "date",
    "Chat Template",
    "ARC",
    "HellaSwag",
    "MMLU",
    "TruthfulQA",
    "Winogrande",
    "GSM8K",
    "Maintainers Choice",
]


def _backup_existing(path: Path) -> Optional[Path]:
    if not path.exists():
        return None
    ts = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    backup = path.with_name(f"{path.stem}.bak_{ts}{path.suffix}")
    shutil.copy2(path, backup)
    return backup


def _discover_train_parquet(repo_id: str, *, revision: Optional[str]) -> str:
    files = HfApi().list_repo_files(repo_id=repo_id, repo_type="dataset", revision=revision)
    parquet_files = [f for f in files if f.startswith("data/train-") and f.endswith(".parquet")]
    if not parquet_files:
        raise SystemExit(f"No train parquet files found in {repo_id}. Files: {files[:20]}")
    if len(parquet_files) > 1:
        # Prefer the first shard if sharded (still deterministic after sort).
        parquet_files = sorted(parquet_files)
    return parquet_files[0]


def main(argv: Optional[list[str]] = None) -> None:
    ap = argparse.ArgumentParser(description="Download full Open LLM Leaderboard v1 (old) contents table and write CSV.")
    ap.add_argument("--repo_id", default="open-llm-leaderboard-old/contents", help="HF dataset repo id.")
    ap.add_argument(
        "--revision",
        default=None,
        help="Optional git revision/commit SHA to pin to (default: latest).",
    )
    ap.add_argument(
        "--parquet_file",
        default=None,
        help="Optional parquet filename within the dataset repo (default: auto-discover data/train-*.parquet).",
    )
    ap.add_argument(
        "--output_csv",
        default="tables/open_llm_leaderboard/open_llm_leaderboard_old.csv",
        help="Where to write the downloaded v1 leaderboard CSV.",
    )
    ap.add_argument(
        "--backup",
        action="store_true",
        help="If set, create a timestamped backup of the existing output CSV before overwriting.",
    )
    args = ap.parse_args(argv)

    out_path = Path(args.output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if args.backup:
        backup = _backup_existing(out_path)
        if backup:
            print(f"[download] backed up existing CSV -> {backup}")

    parquet_file = args.parquet_file or _discover_train_parquet(args.repo_id, revision=args.revision)
    parquet_path = hf_hub_download(
        repo_id=args.repo_id,
        repo_type="dataset",
        filename=parquet_file,
        revision=args.revision,
    )

    df = pd.read_parquet(parquet_path)

    missing = [c for c in EXPECTED_COLUMNS_V1 if c not in df.columns]
    extra = [c for c in df.columns if c not in EXPECTED_COLUMNS_V1]
    if missing or extra:
        raise SystemExit(
            "Downloaded table schema does not match expected v1 leaderboard columns.\n"
            f"Missing columns ({len(missing)}): {missing}\n"
            f"Extra columns ({len(extra)}): {extra}\n"
            f"Downloaded columns: {list(df.columns)}"
        )

    df = df[EXPECTED_COLUMNS_V1]
    df.to_csv(out_path, index=False)

    info = HfApi().dataset_info(repo_id=args.repo_id, revision=args.revision)
    rev = getattr(info, "sha", None) or args.revision or "unknown"
    print(f"[download] source: {args.repo_id}@{rev}")
    print(f"[download] wrote {len(df)} rows × {len(df.columns)} cols -> {out_path}")


if __name__ == "__main__":
    main()


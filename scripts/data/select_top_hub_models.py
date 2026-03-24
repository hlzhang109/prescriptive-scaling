#!/usr/bin/env python3
"""
Select top-k% models by Hub score from the Open LLM Leaderboard CSV.

This is a small utility to create a filtered CSV consisting of the
top fraction of rows sorted by the 'Hub ❤️' column.

Example:
  python scripts/data/select_top_hub_models.py \
    --input tables/open_llm_leaderboard/open_llm_leaderboard_with_tokens.csv \
    --output tables/open_llm_leaderboard/open_llm_leaderboard_with_tokens_top50hub.csv \
    --fraction 0.5
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import List


def select_top_hub(
    input_path: str,
    output_path: str,
    fraction: float = 0.5,
    hub_col: str = "Hub ❤️",
) -> None:
    in_path = Path(input_path)
    out_path = Path(output_path)
    if not in_path.is_file():
        raise SystemExit(f"Input CSV not found: {in_path}")

    with in_path.open() as f:
        reader = csv.DictReader(f)
        rows: List[dict] = list(reader)
        header = reader.fieldnames or []

    if not rows:
        raise SystemExit(f"No rows in {in_path}")
    if hub_col not in header:
        raise SystemExit(f"Column '{hub_col}' not found in {in_path}")

    scored = []
    for idx, row in enumerate(rows):
        v = row.get(hub_col, "")
        try:
            score = float(v) if v not in ("", "nan", "NaN") else float("nan")
        except Exception:
            score = float("nan")
        if math.isfinite(score):
            scored.append((score, idx))
    if not scored:
        raise SystemExit(f"No finite values in '{hub_col}'")

    scored.sort(reverse=True)  # highest Hub score first
    k = max(1, int(math.ceil(fraction * len(scored))))
    keep_idx = {i for _, i in scored[:k]}
    filtered = [rows[i] for i in sorted(keep_idx)]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        for row in filtered:
            writer.writerow(row)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Select top-fraction models by Hub score")
    p.add_argument("--input", required=True, help="Input CSV path")
    p.add_argument("--output", required=True, help="Output CSV path")
    p.add_argument(
        "--fraction",
        type=float,
        default=0.5,
        help="Fraction of rows to keep (default: 0.5 for top 50%%)",
    )
    p.add_argument(
        "--hub_col",
        default="Hub ❤️",
        help="Column name for Hub score (default: 'Hub ❤️')",
    )
    args = p.parse_args(argv)
    if not (0.0 < args.fraction <= 1.0):
        raise SystemExit("--fraction must be in (0,1]")
    select_top_hub(args.input, args.output, fraction=float(args.fraction), hub_col=str(args.hub_col))


if __name__ == "__main__":
    main()

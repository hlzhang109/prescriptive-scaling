#!/usr/bin/env python3
"""
Summarize Open LLM Leaderboard base-model usage by the repo's 4-period scheme.

Reads:
  tables/open_llm_leaderboard/open_llm_leaderboard_with_tokens.csv

Uses:
  - "Identified base model" as the base-model key
  - PERIOD4_BOUNDS from skill_frontier.core.sigmoid to assign each row to a period

Outputs:
  - period4_base_models_counts.csv (long format: period, base_model, count)
  - period4_base_models_pivot.csv  (wide format: base_model x period counts)
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Optional, Tuple

import pandas as pd

# Ensure repo root is on sys.path ahead of `scripts/`, since `scripts/skill_frontier.py`
# would otherwise shadow the `skill_frontier/` package.
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from skill_frontier.core.sigmoid import PERIOD4_BOUNDS  # type: ignore
from skill_frontier.io.csv_utils import detect_date_col, parse_year_month  # type: ignore


def _assign_period_label(year_month: Optional[Tuple[int, int]]) -> Optional[str]:
    if year_month is None:
        return None
    year, month = year_month
    for label, (y_start, m_start), (y_end, m_end) in PERIOD4_BOUNDS:
        if (year, month) < (y_start, m_start):
            continue
        if (year, month) > (y_end, m_end):
            continue
        return label
    return None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--csv",
        default=os.path.join(
            "tables", "open_llm_leaderboard", "open_llm_leaderboard_with_tokens.csv"
        ),
        help="Input Open LLM Leaderboard CSV.",
    )
    ap.add_argument(
        "--out_dir",
        default=os.path.join("outputs", "open_llm_leaderboard", "current"),
        help="Output directory for tables.",
    )
    ap.add_argument(
        "--min_count",
        type=int,
        default=10,
        help="Only include base models appearing at least this many times in a period.",
    )
    ap.add_argument(
        "--date_col",
        default=None,
        help="Optional override for the column used to assign periods.",
    )
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    base_col = "Identified base model"
    if base_col not in df.columns:
        raise SystemExit(f"Missing required column {base_col!r} in {args.csv}")

    date_col = args.date_col or detect_date_col(list(df.columns))
    if not date_col or date_col not in df.columns:
        raise SystemExit(
            f"Could not detect a usable date column in {args.csv}; "
            "pass --date_col (e.g. 'Upload To Hub Date' or 'date')."
        )

    year_month = df[date_col].astype(str).map(parse_year_month)
    period_label = year_month.map(_assign_period_label)

    df = df.assign(_period_label=period_label)
    df = df[df["_period_label"].notna()].copy()
    df[base_col] = df[base_col].fillna("").astype(str).str.strip()
    df = df[df[base_col] != ""].copy()

    counts = (
        df.groupby(["_period_label", base_col], dropna=False)
        .size()
        .reset_index(name="n")
    )
    counts = counts[counts["n"] >= int(args.min_count)].copy()

    label_to_idx = {lab: i + 1 for i, (lab, _, _) in enumerate(PERIOD4_BOUNDS)}
    counts["period_idx"] = counts["_period_label"].map(label_to_idx)
    counts = counts.rename(columns={"_period_label": "period_label", base_col: "base_model"})
    counts = counts[["period_idx", "period_label", "base_model", "n"]].sort_values(
        ["period_idx", "n", "base_model"], ascending=[True, False, True]
    )

    os.makedirs(args.out_dir, exist_ok=True)
    out_counts = os.path.join(args.out_dir, "period4_base_models_counts.csv")
    counts.to_csv(out_counts, index=False)

    pivot = counts.pivot_table(
        index="base_model",
        columns="period_label",
        values="n",
        aggfunc="first",
    ).sort_index()
    # Ensure chronological period ordering (matches PERIOD4_BOUNDS), instead of pandas'
    # default column sort order.
    ordered_cols = [lab for (lab, _, _) in PERIOD4_BOUNDS if lab in pivot.columns]
    remaining = [c for c in pivot.columns if c not in ordered_cols]
    pivot = pivot.reindex(columns=[*ordered_cols, *remaining])
    # Keep blanks (NaN) where base models don't meet min_count for that period.
    out_pivot = os.path.join(args.out_dir, "period4_base_models_pivot.csv")
    pivot.to_csv(out_pivot)

    print(f"Wrote: {out_counts}")
    print(f"Wrote: {out_pivot}")
    print(f"Periods: {[lab for (lab, _, _) in PERIOD4_BOUNDS]}")


if __name__ == "__main__":
    main()

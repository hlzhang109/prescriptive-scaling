#!/usr/bin/env python3
"""
Plot year-wise distribution of models for both Open LLM Leaderboard CSVs
excluding rows with missing compute.

Compute presence criterion: both 'Pretraining tokens (T)' and '#Params (B)'
must be present and finite; compute is taken as 6 * T * B > 0 by default.

Years are extracted from:
  - New (with_tokens): 'Upload To Hub Date'
  - Old (old_with_tokens): 'date'

Outputs a single bar chart (PDF+PNG) with grouped counts by year per dataset.
"""

from __future__ import annotations

import argparse
import csv as _csv
import os
from collections import Counter
from typing import Dict, Tuple

import numpy as np


def _parse_year(s: str) -> int | None:
    s = (s or "").strip()
    if not s or s.lower() in ("nan", "na", "none"):
        return None
    import datetime, re
    for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%d/%m/%Y", "%m/%d/%Y", "%Y-%m", "%Y/%m", "%Y"):
        try:
            return datetime.datetime.strptime(s, fmt).year
        except Exception:
            pass
    m = re.search(r"(20\d{2})", s)
    if m:
        try:
            return int(m.group(1))
        except Exception:
            return None
    return None


def _count_years(path: str, date_col: str, compute_multiplier: float = 6.0) -> Counter:
    ctr = Counter()
    with open(path, "r", newline="") as f:
        reader = _csv.DictReader(f)
        for r in reader:
            # Compute presence check
            vt = r.get("Pretraining tokens (T)", None)
            vb = r.get("#Params (B)", None)
            try:
                T = float(vt) if vt not in (None, "", "nan", "NaN") else float("nan")
            except Exception:
                T = float("nan")
            try:
                B = float(vb) if vb not in (None, "", "nan", "NaN") else float("nan")
            except Exception:
                B = float("nan")
            if not (np.isfinite(T) and np.isfinite(B) and (compute_multiplier * T * B) > 0.0):
                continue
            yr = _parse_year(r.get(date_col, ""))
            if yr is not None:
                ctr[yr] += 1
    return ctr


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Year distribution for OLL datasets (excluding missing compute)")
    p.add_argument("--new_csv", default=os.path.join("tables", "open_llm_leaderboard", "open_llm_leaderboard_with_tokens.csv"))
    p.add_argument("--old_csv", default=os.path.join("tables", "open_llm_leaderboard", "open_llm_leaderboard_old_with_tokens.csv"))
    p.add_argument("--out", default=os.path.join("outputs", "oll_year_distribution.png"))
    p.add_argument("--compute_multiplier", type=float, default=6.0)
    args = p.parse_args(argv)

    new_counts = _count_years(args.new_csv, date_col="Upload To Hub Date", compute_multiplier=float(args.compute_multiplier))
    old_counts = _count_years(args.old_csv, date_col="date", compute_multiplier=float(args.compute_multiplier))

    # Unified year axis covering 2022..2025 (common focus)
    years = list(range(2022, 2026))
    new_vals = [new_counts.get(y, 0) for y in years]
    old_vals = [old_counts.get(y, 0) for y in years]

    # Plot grouped bars
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError("matplotlib is required for plotting") from e
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    x = np.arange(len(years))
    w = 0.36
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    ax.bar(x - w / 2, old_vals, width=w, label="OLL old (date)", color="#1f77b4")
    ax.bar(x + w / 2, new_vals, width=w, label="OLL new (Upload To Hub)", color="#ff7f0e")
    ax.set_xticks(x)
    ax.set_xticklabels([str(y) for y in years])
    ax.set_ylabel("Model count (compute present)")
    ax.set_title("Open LLM Leaderboard — Year Distribution (compute present)")
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()
    fig.savefig(args.out, dpi=300, bbox_inches="tight")
    base, _ = os.path.splitext(args.out)
    fig.savefig(base + ".pdf", dpi=300, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":  # pragma: no cover
    main()


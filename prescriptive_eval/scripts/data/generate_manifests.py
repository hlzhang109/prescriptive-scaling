#!/usr/bin/env python3
"""
Generate one-shot (10% budget) manifests with UNIQUE model IDs per group.

- Splits Open LLM Leaderboard (with_tokens) by year: pre-2025 and 2025.
- Designs K = ceil(0.1 * N_group) replicate targets via four-point + exchange
  using only compute and year (no y/accuracy).
- Assigns each replicate to the nearest UNUSED candidate by compute so that
  the final manifest has K distinct model IDs per group.

Usage:
  python scripts/data/generate_manifests.py \
    --csv tables/open_llm_leaderboard/open_llm_leaderboard_with_tokens.csv \
    --out_dir outputs/one_shot_manifests_unique

Produces:
  - outputs/one_shot_manifests_unique/manifest__< 2025.txt
  - outputs/one_shot_manifests_unique/manifest__2025.txt
"""

from __future__ import annotations

import argparse
import csv as _csv
import math
import os
from typing import List, Tuple

import numpy as np

# Import design primitives with robust path handling
import sys
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
DOCS_DIR = os.path.join(REPO_ROOT, 'docs')
if DOCS_DIR not in sys.path:
    sys.path.insert(0, DOCS_DIR)
try:
    from frontier_one_shot_design import (
        FrontierParams, WeightingModel, make_design_exchange
    )
except Exception as e:
    raise SystemExit(f"Failed to import frontier_one_shot_design from {DOCS_DIR}: {e}")


def _parse_year(s: str) -> int | None:
    s = (s or "").strip()
    if not s or s.lower() in ("nan", "none", "na"):
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


def _read_pool_oll_with_tokens(path: str) -> Tuple[List[str], np.ndarray, np.ndarray]:
    TOK = "Pretraining tokens (T)"
    PAR = "#Params (B)"
    MUL = 6.0
    mids: List[str] = []
    C: List[float] = []
    years: List[int] = []
    with open(path, "r", newline="") as f:
        r = _csv.DictReader(f)
        rows = list(r)
    headers = list(rows[0].keys()) if rows else []
    date_col = "Upload To Hub Date" if "Upload To Hub Date" in headers else ("date" if "date" in headers else None)
    if date_col is None:
        raise SystemExit("Expected a date column ('Upload To Hub Date' or 'date') in the CSV header")
    for row in rows:
        m = str(row.get("model", "") or row.get("Model sha", "") or row.get("name", "") or "").strip()
        if not m:
            continue
        v1 = row.get(TOK, None)
        v2 = row.get(PAR, None)
        try:
            t = float(v1) if v1 not in (None, "", "nan", "NaN") else float("nan")
        except Exception:
            t = float("nan")
        try:
            b = float(v2) if v2 not in (None, "", "nan", "NaN") else float("nan")
        except Exception:
            b = float("nan")
        if not (np.isfinite(t) and np.isfinite(b)):
            continue
        c = float(MUL * t * b)
        if not (np.isfinite(c) and c > 0.0):
            continue
        yr = _parse_year(row.get(date_col, ""))
        if yr is None:
            continue
        mids.append(m)
        C.append(c)
        years.append(int(yr))
    return mids, np.asarray(C, float), np.asarray(years, int)


def _unique_assignment_by_nearest(z: np.ndarray, targets: List[float], used: np.ndarray | None = None) -> List[int]:
    """Assign each target to the nearest UNUSED index in z (greedy).

    - z: array of compute z=log10(C) per candidate in the group.
    - targets: list of target z values (one entry per replicate).
    - used: optional boolean mask of pre-used indices.
    Returns a list of distinct indices with length <= len(targets) (stops if pool exhausted).
    """
    n = z.shape[0]
    taken = np.zeros(n, dtype=bool) if used is None else used.copy()
    out: List[int] = []
    order = np.argsort(np.asarray(targets, float))  # small heuristic for locality
    for k in order:
        t = float(targets[k])
        # sorted by distance
        idx = np.argsort(np.abs(z - t))
        pick = None
        for i in idx:
            if not taken[i]:
                pick = int(i)
                break
        if pick is None:
            break
        taken[pick] = True
        out.append(pick)
    return out


def _write_manifest(path: str, mids: List[str], indices: List[int]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        for i in indices:
            f.write(f"{mids[i]}\n")


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate one-shot (10%) manifests with unique model IDs per year group")
    ap.add_argument("--csv", required=True, help="Open LLM Leaderboard with_tokens CSV path")
    ap.add_argument("--out_dir", default=os.path.join("outputs", "one_shot_manifests_unique"))
    ap.add_argument("--K_frac", type=float, default=0.10)
    ap.add_argument("--y0", type=float, default=0.20)
    ap.add_argument("--L", type=float, default=0.75)
    ap.add_argument("--a", type=float, default=-9.0)
    ap.add_argument("--b", type=float, default=1.0)
    ap.add_argument("--c", type=float, default=1.543)
    ap.add_argument("--max_swaps", type=int, default=200)
    args = ap.parse_args()

    mids, C, years = _read_pool_oll_with_tokens(args.csv)
    if C.size == 0:
        raise SystemExit("No candidates with usable compute/year")
    z = np.log10(C)
    theta0 = FrontierParams(y0=args.y0, L=args.L, a=args.a, b=max(args.b, 1e-6))
    weighting = WeightingModel(noise="binomial", m=100)

    def design_unique(mask: np.ndarray, label: str) -> None:
        idx = np.nonzero(mask)[0]
        if idx.size == 0:
            print(f"[design] {label}: empty")
            return
        zg = z[idx]
        N = int(zg.size)
        K = int(max(4, min(N, math.ceil(args.K_frac * N))))
        # exchange design returns replicate targets in coordinate space
        targets = make_design_exchange(zg.tolist(), K, theta0, weighting=weighting, c=args.c, max_swaps=args.max_swaps, freeze_extremes=True)
        # assign to unique nearest indices within the group
        local_indices = _unique_assignment_by_nearest(zg, targets)
        if len(local_indices) < K:
            print(f"[design] {label}: only {len(local_indices)} unique neighbors found (K={K})")
        # map local indices to global
        global_indices = [int(idx[i]) for i in local_indices]
        outp = os.path.join(args.out_dir, f"manifest__{label}.txt")
        _write_manifest(outp, mids, global_indices)
        print(f"[design] {label}: N={N} K={K} unique={len(global_indices)} -> {outp}")

    os.makedirs(args.out_dir, exist_ok=True)
    design_unique(years < 2025, "< 2025")
    design_unique(years == 2025, "2025")


if __name__ == "__main__":
    main()

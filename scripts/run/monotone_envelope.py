#!/usr/bin/env python3
"""
Run the monotone envelope + isoquant pipeline on a CSV with two columns.

Example:
  python scripts/run/monotone_envelope.py \
    --csv tables/merged_livebench.csv \
    --x_col "a_BBH Raw" --y_col "a_MATH Lvl 5 Raw" \
    --out outputs/monotone_envelope_BBHxMATH.png

Dependencies: numpy, scipy, matplotlib.
"""

from __future__ import annotations

import argparse
import csv as _csv
import os
from typing import List

import numpy as np

# Robust import: allow running both as module and as a script path
try:
    from scripts.monotone_envelope import (  # type: ignore
        fit_upper_lower_with_tradeoff,
        quick_plot,
    )
except Exception:
    try:
        from monotone_envelope import (  # type: ignore
            fit_upper_lower_with_tradeoff,
            quick_plot,
        )
    except Exception:
        import os as _os, sys as _sys
        REPO_ROOT = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), _os.pardir))
        if REPO_ROOT not in _sys.path:
            _sys.path.insert(0, REPO_ROOT)
        from scripts.monotone_envelope import (  # type: ignore
            fit_upper_lower_with_tradeoff,
            quick_plot,
        )


def _read_two_columns(path: str, x_col: str, y_col: str) -> tuple[np.ndarray, np.ndarray]:
    with open(path, "r", newline="") as f:
        r = _csv.DictReader(f)
        xs: List[float] = []
        ys: List[float] = []
        for row in r:
            if x_col not in row or y_col not in row:
                raise ValueError(f"Columns not found: {x_col}, {y_col}. Available: {list(row.keys())}")
            vx = row.get(x_col, "")
            vy = row.get(y_col, "")
            try:
                x = float(vx)
                y = float(vy)
            except Exception:
                x, y = float("nan"), float("nan")
            if np.isfinite(x) and np.isfinite(y):
                xs.append(x)
                ys.append(y)
    if not xs:
        raise ValueError("No finite values read from CSV")
    return np.array(xs, dtype=float), np.array(ys, dtype=float)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Monotone envelope + isoquant for a 2D scatter from CSV")
    p.add_argument("--csv", required=True, help="Input CSV path")
    p.add_argument("--x_col", required=True, help="Column for x")
    p.add_argument("--y_col", required=True, help="Column for y")
    p.add_argument("--out", default=None, help="Output plot path (PNG). If omitted, show interactively")
    p.add_argument("--tau_hi", type=float, default=0.98, help="Upper quantile level for g(x)")
    p.add_argument("--tau_lo", type=float, default=0.02, help="Lower quantile level for f(x)")
    p.add_argument("--K", type=int, default=20, help="Max interior knots (min with sqrt(n))")
    p.add_argument("--degree", type=int, default=3, help="Spline degree (cubic=3)")
    p.add_argument("--lam_grid", nargs="*", type=float, default=[1e-2], help="Lambda grid for curvature penalty")
    args = p.parse_args(argv)

    x, y = _read_two_columns(args.csv, args.x_col, args.y_col)
    g, f, iso, comb = fit_upper_lower_with_tradeoff(
        x, y, tau_hi=args.tau_hi, tau_lo=args.tau_lo, K=args.K, degree=args.degree, lam_grid=args.lam_grid
    )
    title = f"Monotone envelope: {args.x_col} vs {args.y_col}\n(elbow at x={comb.x_star:.3g}, y={comb.y_star:.3g})"
    if args.out is not None:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    quick_plot(x, y, g_model=g, f_model=f, iso_params=iso, combined=comb, title=title, save_path=args.out)


if __name__ == "__main__":  # pragma: no cover
    main()

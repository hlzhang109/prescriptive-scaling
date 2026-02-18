#!/usr/bin/env python3
"""
Plot pinball-loss vs alpha for balanced I-optimal sweeps.

Expected input layout (per base_dir):
  eval_p4_alphaXX/
    k1/
      <task>__pinball_baselines.csv  (train/val rows with L_* columns)
    k2/
    k3/

Outputs:
  plots_pinball/
    pinball_vs_alpha_k{1,2,3}.png/pdf   (avg over tasks per method)
    pinball_vs_alpha_avg.png/pdf        (avg over tasks and k)
"""
from __future__ import annotations

import argparse
import os
import re
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import matplotlib.pyplot as plt  # type: ignore


def _find_alpha_dirs(base: Path) -> List[Tuple[int, Path]]:
    alphas: List[Tuple[int, Path]] = []
    for p in base.glob("eval_p4_alpha*"):
        m = re.match(r"eval_p4_alpha(\d+)", p.name)
        if m:
            alphas.append((int(m.group(1)), p))
    return sorted(alphas, key=lambda x: x[0])


def _load_avg_val_losses(alpha_dir: Path, methods: List[str]) -> Dict[int, Dict[str, float]]:
    """Return {k: {method: avg_val_loss_over_tasks}} for this alpha."""
    out: Dict[int, Dict[str, float]] = {}
    for k in (1, 2, 3):
        kdir = alpha_dir / f"k{k}"
        if not kdir.is_dir():
            continue
        vals: Dict[str, List[float]] = {m: [] for m in methods}
        for fn in kdir.glob("*__pinball_baselines.csv"):
            try:
                import pandas as pd  # type: ignore
            except Exception:
                raise RuntimeError("pandas is required to parse pinball baseline CSVs for plotting.")
            df = pd.read_csv(fn)
            val_row = df[df["split"] == "val"]
            if val_row.empty:
                continue
            row = val_row.iloc[0]
            for m in methods:
                col = None
                if m == "sigmoid":
                    col = "L_sigmoid"
                elif m == "ispline":
                    col = "L_ispline"
                elif m == "binwise":
                    col = "L_oracle"
                elif m == "const":
                    col = "L_null"
                if col and col in row and np.isfinite(row[col]):
                    vals[m].append(float(row[col]))
        # Also support sigmoid-only summaries produced by --write_pinball
        summary_sig = kdir / "summary_over_tasks_pinball_sigmoid.csv"
        if summary_sig.is_file() and "sigmoid" in methods:
            try:
                import pandas as pd  # type: ignore
                df_sig = pd.read_csv(summary_sig)
                val_row = df_sig[df_sig["split"] == "val"]
                if not val_row.empty:
                    v = float(val_row.iloc[0]["L_sigmoid"])
                    vals.setdefault("sigmoid", []).append(v)
            except Exception:
                pass
        if any(len(v) for v in vals.values()):
            out[k] = {m: (float(np.mean(v)) if len(v) else np.nan) for m, v in vals.items()}
    return out


def _plot_series(alphas: List[int], data: Dict[int, List[float]], title: str, out_path: Path) -> None:
    # Target tick labels with equal spacing
    target_alphas = [0, 5, 10, 20, 50, 100]
    pos = np.arange(len(target_alphas))

    plt.figure(figsize=(8, 5))
    for label, ys in data.items():
        # Map existing series onto target ticks
        alpha_to_val = {a: ys[i] for i, a in enumerate(alphas)}
        aligned = [alpha_to_val.get(a, np.nan) for a in target_alphas]
        plt.plot(pos, aligned, marker="o", label=label)
    plt.xlabel(r"$\alpha$")
    plt.ylabel("Pinball loss (val)")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.xticks(pos, target_alphas)
    plt.xlim(-0.5, len(target_alphas) - 0.5)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path.with_suffix(".png"), dpi=300)
    plt.savefig(out_path.with_suffix(".pdf"), dpi=300)
    plt.close()


def main() -> None:
    ap = argparse.ArgumentParser(description="Plot pinball loss vs alpha from sweep outputs.")
    ap.add_argument("--base_dir", required=True, help="Base sweep directory (e.g., outputs/sweeps_iopt_balanced_fullrange)")
    ap.add_argument("--out_dir", default=None, help="Output directory for plots (default: <base_dir>/plots_pinball)")
    args = ap.parse_args()

    base = Path(args.base_dir)
    out_dir = Path(args.out_dir) if args.out_dir else base / "plots_pinball"
    # Focus on sigmoid frontier pinball loss; other methods can be added later.
    methods = ["sigmoid"]

    alpha_dirs = _find_alpha_dirs(base)
    if not alpha_dirs:
        raise SystemExit(f"No eval_p4_alpha* directories found under {base}")

    series_per_k: Dict[int, Dict[str, List[float]]] = {1: {m: [] for m in methods},
                                                       2: {m: [] for m in methods},
                                                       3: {m: [] for m in methods}}
    alphas_used: List[int] = []
    for alpha, p in alpha_dirs:
        per_k = _load_avg_val_losses(p, methods)
        # require all ks? we fill per available k, use NaN if missing
        alphas_used.append(alpha)
        for k in (1, 2, 3):
            for m in methods:
                val = per_k.get(k, {}).get(m, np.nan)
                series_per_k[k][m].append(val)

    # Plot per k
    for k in (1, 2, 3):
        data = {m: series_per_k[k][m] for m in methods}
        _plot_series(alphas_used, data, f"Pinball loss vs $\\alpha$ (k={k})", out_dir / f"pinball_vs_alpha_k{k}")

    # Plot averaged over k
    avg_data: Dict[str, List[float]] = {m: [] for m in methods}
    for idx in range(len(alphas_used)):
        for m in methods:
            vals = [series_per_k[k][m][idx] for k in (1, 2, 3)]
            avg_data[m].append(float(np.nanmean(vals)) if np.any(np.isfinite(vals)) else np.nan)
    _plot_series(alphas_used, avg_data, "Pinball loss vs $\\alpha$ (avg k)", out_dir / "pinball_vs_alpha_avg")


if __name__ == "__main__":
    main()

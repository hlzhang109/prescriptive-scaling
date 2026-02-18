#!/usr/bin/env python3
"""
3D pairwise frontier plot: (x, y) are two task accuracies; z is pretraining compute.

At each selected compute level, we reconstruct the convex attainable set in
the 2D task plane from directional supports, then stack these convex polygons
along the z axis (compute) to visualize the joint frontier surface.

Fully general: accepts arbitrary CSVs via the same flexible mapping used by
run_frontier_from_csv.py. You must specify exactly two task columns via
--pair_tasks; scaling and compute mapping follow the same conventions as the
single-task runner.

Examples (Open LLM Leaderboard, Raw metrics):
  python scripts/plot/diagnostics/pairwise_3d.py \
    --csv tables/open_llm_leaderboard/open_llm_leaderboard_with_tokens.csv \
    --model_col 'Model sha' \
    --compute_product_cols 'Pretraining tokens (T)' '#Params (B)' --compute_multiplier 6 \
    --pair_tasks 'BBH Raw' 'GPQA Raw' \
    --output 'outputs/pairwise3d_BBH_Raw__GPQA_Raw.png'

Old leaderboard with percentages (scale 0.01):
  python scripts/plot/diagnostics/pairwise_3d.py \
    --csv tables/open_llm_leaderboard/open_llm_leaderboard_old_with_tokens.csv \
    --model_col 'Model sha' \
    --compute_product_cols 'Pretraining tokens (T)' '#Params (B)' --compute_multiplier 6 \
    --pair_tasks ARC HellaSwag --task_scale 0.01 \
    --output 'outputs/pairwise3d_ARC__HellaSwag.png'
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import List, Optional, Sequence, Tuple
import math

import numpy as np

# Ensure repo root is importable when running as a script.
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, os.pardir))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from skill_frontier.core.frontier import (  # noqa: E402
    FrontierConfig,
    SkillFrontier,
    DEAConfig,
    DEAEstimator,
    QuantileConfig,
    QuantileFrontierEstimator,
    isotonic_regression_monotone_increasing,
)
from skill_frontier.core.utils import ModelPanel  # noqa: E402
from scripts.run.frontier_from_csv import _read_csv as _read_csv_flexible  # noqa: E402


def _safe(name: str) -> str:
    return "".join(ch if (ch.isalnum() or ch in ("_","-")) else "_" for ch in name)


def two_simplex_directions(k: int) -> np.ndarray:
    t = np.linspace(0.0, 1.0, k)
    return np.stack([t, 1 - t], axis=1)


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="3D pairwise frontier plot (two tasks vs compute)")
    # Input mapping (same style as flexible runner)
    p.add_argument("--csv", required=True, help="Input CSV path")
    p.add_argument("--model_col", default="model", help="Name of the model ID column")
    p.add_argument("--logC_col", default="logC", help="Name of log-compute column (if present)")
    p.add_argument("--label_col", default=None, help="Optional display-name column (e.g., eval_name)")
    p.add_argument("--compute_col", default=None, help="Name of raw compute column (if no logC). Uses natural log internally.")
    p.add_argument("--compute_eps", type=float, default=1e-12, help="Epsilon added before log (only for positivity check in loader)")
    p.add_argument("--compute_product_cols", nargs=2, default=None, help="Two column names to multiply for raw compute")
    p.add_argument("--compute_multiplier", type=float, default=1.0, help="Multiplier applied to compute product (e.g., 6 for 6*T*B)")
    p.add_argument("--task_prefix", default=None, help="Prefix to select task columns if not using --pair_tasks")
    p.add_argument("--task_contains", default=None, help="Substring to select task columns if not using --pair_tasks")
    p.add_argument("--task_scale", type=float, default=1.0, help="Scale factor for task values (e.g., 0.01 for percentages)")
    p.add_argument("--pair_tasks", nargs=2, default=None, help="Exactly two task column names to form (x, y)")
    p.add_argument("--tasks", nargs="*", default=None, help="Candidate task columns; if --pair_tasks not set and this list has >=2, generate all pairs")
    p.add_argument("--pair_all", action="store_true", help="Generate 3D plots for all pairs from --tasks/--task_contains/--task_prefix")
    # Data curation
    p.add_argument("--min_models_per_task", type=int, default=0)
    p.add_argument("--min_values_per_row", type=int, default=0)
    p.add_argument("--impute", choices=["none", "median"], default="none")
    p.add_argument("--no_clip01", action="store_true")
    # Frontier config
    p.add_argument("--num_C_grid", type=int, default=24)
    p.add_argument("--kernel", type=str, default="gaussian", choices=["gaussian", "epanechnikov"])
    p.add_argument("--bandwidth", type=float, default=None)
    p.add_argument("--alpha_cap_multiple", type=float, default=4.0)
    p.add_argument("--crs", action="store_true")
    p.add_argument("--no_smooth_C", action="store_true", help="Disable monotone smoothing across compute")
    p.add_argument("--pairwise_points", type=int, default=128, help="Number of 2D directions for halfspace reconstruction")
    p.add_argument("--C_count", type=int, default=10, help="Number of compute slices to plot across the grid")
    p.add_argument("--method", default="Q0.99", choices=["DEA", "Q0.95", "Q0.99"], help="Convex method for reconstruction")
    p.add_argument("--min_slice_coverage", type=float, default=0.3, help="Minimum fraction of directions with finite supports to include a slice")
    # Output
    p.add_argument("--output", default=None, help="Output image path (.png/.pdf) for single pair")
    p.add_argument("--out_dir", default=None, help="Directory to save multiple pair images when --pair_all is set")
    p.add_argument("--dpi", type=int, default=300)
    return p


def _render_pair(args, pair: Tuple[str, str]) -> None:
    # Build panel using the flexible loader with exactly the pair tasks
    panel: ModelPanel = _read_csv_flexible(
        path=args.csv,
        model_col=args.model_col,
        label_col=args.label_col,
        logC_col=args.logC_col,
        compute_col=args.compute_col,
        compute_product_cols=args.compute_product_cols,
        compute_multiplier=args.compute_multiplier,
        task_prefix=args.task_prefix,
        task_cols=list(pair),
        task_contains=args.task_contains,
        task_scale=args.task_scale,
        auto_task_scale=False,
        min_models_per_task=args.min_models_per_task,
        min_values_per_row=args.min_values_per_row,
        impute=args.impute,
        clip01=(not args.no_clip01),
        compute_eps=args.compute_eps,
    )
    # Indices for the pair
    try:
        i = panel.task_index(pair[0])
        j = panel.task_index(pair[1])
    except ValueError as e:
        raise SystemExit(f"Pair task not found in panel: {e}")

    # Frontier configuration and runner
    cfg = FrontierConfig(
        num_C_grid=args.num_C_grid,
        kernel=args.kernel,
        bandwidth=args.bandwidth,
        num_directions=64,  # we compute pairwise supports separately below
        dea=DEAConfig(var_returns_to_scale=(not args.crs), alpha_cap_multiple=args.alpha_cap_multiple, enforce_monotone_in_C=(not args.no_smooth_C)),
        write_vertices=False,
    )
    cfg.quantile = QuantileConfig(enforce_monotone_in_C=(not args.no_smooth_C))
    if args.no_smooth_C:
        cfg.smooth_monotone_in_C = False

    runner = SkillFrontier(panel, cfg)
    runner.run(output_dir=None)  # we only need grid/windows

    # Prepare estimators and 2D directions
    dea_est = DEAEstimator(panel, cfg.dea)
    q_est = QuantileFrontierEstimator(panel, cfg.quantile)
    U2 = two_simplex_directions(args.pairwise_points)

    # Choose compute slices with adequate coverage
    nC = len(runner.grid_logC)  # type: ignore[arg-type]

    # Precompute supports along the entire grid per direction (for optional smoothing)
    m = U2.shape[0]
    H = np.full((m, nC), np.nan, dtype=float)
    for uidx, u2 in enumerate(U2):
        u_full = np.zeros(panel.num_tasks)
        u_full[i] = float(u2[0])
        u_full[j] = float(u2[1])
        for iC in range(nC):
            lc0 = float(runner.grid_logC[iC])  # type: ignore[index]
            idx_w = runner.window.indices[iC]  # type: ignore[union-attr]
            w_w = runner.window.weights[iC]    # type: ignore[union-attr]
            if args.method == "DEA":
                if not dea_est.is_available():
                    continue
                val, _ = dea_est.support_value(u_full, idx_w, w_w, lc0)
            else:
                vals = q_est.support_values(u_full, idx_w, w_w)
                key = 0.95 if args.method == "Q0.95" else 0.99
                val = vals[key]
            H[uidx, iC] = float(val) if np.isfinite(val) else np.nan

    # Monotone smoothing across compute if enabled (like SkillFrontier)
    if not args.no_smooth_C:
        for uidx in range(m):
            col = H[uidx, :]
            mask = np.isfinite(col)
            if mask.sum() >= 2:
                H[uidx, mask] = isotonic_regression_monotone_increasing(col[mask])

    # Slice coverage and selection
    coverage = np.mean(np.isfinite(H), axis=0)  # (nC,)
    feasible_idxs = [i for i in range(nC) if coverage[i] >= float(args.min_slice_coverage)]
    if not feasible_idxs:
        # fallback: pick mid-slices uniformly even if coverage is low
        feasible_idxs = list(range(nC))
    if args.C_count >= len(feasible_idxs):
        C_indices = feasible_idxs
    else:
        # sample evenly from feasible indices
        pos = np.linspace(0, len(feasible_idxs)-1, num=args.C_count).round().astype(int).tolist()
        C_indices = [feasible_idxs[k] for k in pos]

    # Reconstruct polygons at selected slices using (optionally smoothed) supports
    polys: List[Tuple[float, np.ndarray]] = []  # list of (C_value, P[*,2])
    for iC in C_indices:
        lc0 = float(runner.grid_logC[iC])  # type: ignore[index]
        C0 = float(np.exp(lc0))
        A_half: List[np.ndarray] = []
        b_half: List[float] = []
        for uidx, u2 in enumerate(U2):
            h = H[uidx, iC]
            if not np.isfinite(h):
                continue
            A_half.append(np.array([u2[0], u2[1]], dtype=float))
            b_half.append(float(h))
        # If insufficient directional halfspaces, skip this slice
        if len(A_half) < 3:
            continue
        # Bounding box [0,1]^2
        A_half.extend([
            np.array([1.0, 0.0]),  # x <= 1
            np.array([0.0, 1.0]),  # y <= 1
            np.array([-1.0, 0.0]), # -x <= 0
            np.array([0.0, -1.0]), # -y <= 0
        ])
        b_half.extend([1.0, 1.0, 0.0, 0.0])
        P = runner._halfspace_intersection_polygon(A_half, b_half)
        if P.shape[0] > 0:
            polys.append((C0, P))

    # Plot 3D
    try:
        import matplotlib.pyplot as plt  # type: ignore
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection  # type: ignore
    except Exception as e:
        raise RuntimeError("matplotlib (with mplot3d) is required for 3D plotting") from e

    import matplotlib as mpl  # type: ignore
    mpl.rcParams["font.family"] = "serif"
    fig = plt.figure(figsize=(8.5, 6.5))
    ax = fig.add_subplot(111, projection='3d')

    facecolor = {
        "DEA": (0.95, 0.35, 0.55, 0.25),
        "Q0.95": (0.20, 0.70, 0.80, 0.20),
        "Q0.99": (0.40, 0.50, 0.95, 0.20),
    }.get(args.method, (0.6, 0.6, 0.6, 0.25))
    edgecolor = (0.2, 0.2, 0.2, 0.6)

    for C0, P in polys:
        z = float(math.log10(C0))
        verts3d = [(float(x), float(y), z) for x, y in P]
        coll = Poly3DCollection([verts3d], facecolors=[facecolor], edgecolors=[edgecolor], linewidths=1.0)
        ax.add_collection3d(coll)

    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    # Z uses log10(C) values; we reverse z-limits below so smaller compute is on top
    ax.set_xlabel(pair[0], fontsize=12, fontweight='bold')
    ax.set_ylabel(pair[1], fontsize=12, fontweight='bold')
    ax.set_zlabel('Pretraining Compute (FLOPs)', fontsize=12, fontweight='bold', labelpad=14)
    title = f"Acc. Boundaries — {args.method}: {pair[0]} vs {pair[1]}"
    ax.set_title(title, fontsize=14, fontweight='bold')

    # Autoscale z to data
    if polys:
        zmin = min(math.log10(C) for C, _ in polys)
        zmax = max(math.log10(C) for C, _ in polys)
        # Reverse z so smaller compute (lower log10) is on top
        ax.set_zlim(zmax, zmin)

    plt.tight_layout()
    # Resolve output path
    if args.output:
        out_path = args.output
    else:
        out_dir = args.out_dir or os.path.join("outputs", "pairwise3d")
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f"pair3d__{_safe(pair[0])}__{_safe(pair[1])}.png")
    plt.savefig(out_path, dpi=args.dpi, bbox_inches='tight')
    base, ext = os.path.splitext(out_path)
    if ext.lower() != '.pdf':
        plt.savefig(base + '.pdf', dpi=args.dpi, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved 3D pairwise frontier to {out_path}")


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = build_argparser().parse_args(argv)

    if args.pair_tasks is not None:
        _render_pair(args, tuple(args.pair_tasks))
        return

    # Generate all pairs
    # Discover candidate tasks from provided list or CSV header using contains/prefix
    tasks: List[str]
    if args.tasks:
        tasks = list(args.tasks)
    else:
        with open(args.csv, 'r') as f:
            import csv as _csv
            reader = _csv.DictReader(f)
            cols = reader.fieldnames or []
        if args.task_contains:
            tasks = [c for c in cols if args.task_contains in c]
        elif args.task_prefix:
            tasks = [c for c in cols if c.startswith(args.task_prefix)]
        else:
            raise SystemExit("Provide --pair_tasks, or --tasks, or one of --task_contains/--task_prefix to discover tasks for --pair_all.")
    if len(tasks) < 2:
        raise SystemExit("Need at least two tasks to form pairs")
    # Output directory
    if args.out_dir is None and args.output is not None:
        # If user mistakenly set --output for pair_all, derive out_dir from it
        args.out_dir = os.path.dirname(args.output) or os.path.join("outputs", "pairwise3d")
    if args.out_dir is None:
        args.out_dir = os.path.join("outputs", "pairwise3d")
    os.makedirs(args.out_dir, exist_ok=True)

    for a in range(len(tasks)):
        for b in range(a + 1, len(tasks)):
            pair = (tasks[a], tasks[b])
            # Set output to None so _render_pair writes into out_dir with auto name
            args.output = None
            _render_pair(args, pair)


if __name__ == '__main__':
    main()

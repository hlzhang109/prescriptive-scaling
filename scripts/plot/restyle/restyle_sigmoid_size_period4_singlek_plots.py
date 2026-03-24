#!/usr/bin/env python3
"""Restyle sigmoid_size period4 single-k triptych plots using shared restyle logic."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    import matplotlib.pyplot as plt  # type: ignore
except Exception as exc:  # pragma: no cover
    raise SystemExit("This script requires matplotlib.") from exc

from scripts.plot.restyle._period4_restyle_common import (  # type: ignore
    FIG1_RCPARAMS,
    RestyleConfig,
    list_plot_slugs,
    restyle_slug_set,
)
from skill_frontier.core.period_scheme import (  # type: ignore
    PERIOD4_BOUNDS_NEW,
    PERIOD4_SPLITS_SINGLE_NEW,
)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Restyle sigmoid_size period4 single_k plots to match Figure 1 style."
    )
    parser.add_argument(
        "--plots_dir",
        type=Path,
        default=Path("outputs/sigmoid_size/period4/single_k/plots"),
        help="Directory containing *_period4.{png,pdf} (and bbh_subtasks/).",
    )
    parser.add_argument(
        "--curves_dir",
        type=Path,
        default=Path("outputs/sigmoid_size/period4/single_k/curves"),
        help="Directory containing *_k{1,2,3}.csv curves (train fits).",
    )
    parser.add_argument(
        "--oll_csv",
        type=Path,
        default=Path("tables/open_llm_leaderboard/open_llm_leaderboard_with_tokens.csv"),
        help="Main OLL with-tokens CSV (for Raw tasks).",
    )
    parser.add_argument(
        "--oll_bbh_csv",
        type=Path,
        default=Path("tables/open_llm_leaderboard/open_llm_leaderboard_with_tokens_top50hub.csv"),
        help="Top-hub OLL CSV (for leaderboard_bbh_* subtasks).",
    )
    parser.add_argument("--tau", type=float, default=0.98)
    parser.add_argument("--lambda_b", type=float, default=1e-3)
    parser.add_argument("--kappa_final", type=float, default=50.0)
    parser.add_argument("--fig_width", type=float, default=7.0)
    parser.add_argument("--fig_height", type=float, default=2.1)
    parser.add_argument("--dpi", type=int, default=300)
    args = parser.parse_args(list(argv) if argv is not None else None)

    plt.rcParams.update(dict(FIG1_RCPARAMS))

    plots_dir = args.plots_dir
    curves_dir = args.curves_dir
    if not plots_dir.exists():
        raise FileNotFoundError(f"Missing plots dir: {plots_dir}")
    if not curves_dir.exists():
        raise FileNotFoundError(f"Missing curves dir: {curves_dir}")

    main_slugs = list_plot_slugs(plots_dir)
    bbh_dir = plots_dir / "bbh_subtasks"
    bbh_slugs = list_plot_slugs(bbh_dir) if bbh_dir.exists() else []

    cfg = RestyleConfig(
        period_bounds=PERIOD4_BOUNDS_NEW,
        period_splits_single=PERIOD4_SPLITS_SINGLE_NEW,
        date_col="Upload To Hub Date",
        x_mode="size",
        x_scale_to_plot=1e9,
        x_ticks=[1e7, 1e8, 1e9, 1e10, 1e11],
        x_label="Model Size (#Params)",
        x_label_y=-0.07,
        bottom=0.12,
        period_symbol="t",
    )

    updated_main = restyle_slug_set(
        slugs=main_slugs,
        plots_dir=plots_dir,
        curves_dir=curves_dir,
        csv_path=args.oll_csv,
        cfg=cfg,
        tau=float(args.tau),
        lambda_b=float(args.lambda_b),
        kappa_final=float(args.kappa_final),
        fig_width=float(args.fig_width),
        fig_height=float(args.fig_height),
        dpi=int(args.dpi),
    )

    updated_bbh = restyle_slug_set(
        slugs=bbh_slugs,
        plots_dir=bbh_dir,
        curves_dir=curves_dir,
        csv_path=args.oll_bbh_csv,
        cfg=cfg,
        tau=float(args.tau),
        lambda_b=float(args.lambda_b),
        kappa_final=float(args.kappa_final),
        fig_width=float(args.fig_width),
        fig_height=float(args.fig_height),
        dpi=int(args.dpi),
    )

    print(f"Updated {updated_main} main plots in: {plots_dir}")
    print(f"Updated {updated_bbh} BBH-subtask plots in: {bbh_dir}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

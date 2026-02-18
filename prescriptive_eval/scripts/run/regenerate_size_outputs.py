#!/usr/bin/env python3
"""
Regenerate size-axis outputs to mirror the compute-axis pipeline.

Outputs:
  - outputs/sigmoid_size/** (1D sigmoid frontiers vs model size)
  - evaluation_pinball_baselines_size/** (pinball-loss baseline comparisons vs model size)

This script intentionally reuses the existing generator scripts without refactoring them.
"""

from __future__ import annotations

import argparse
import csv
import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import List


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _read_lines(path: Path) -> List[str]:
    if not path.exists():
        return []
    lines: List[str] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            lines.append(s)
    return lines


def _bbh_subtask_cols(csv_path: Path) -> List[str]:
    if not csv_path.exists():
        return []
    with csv_path.open("r", newline="", encoding="utf-8") as f:
        header = next(csv.reader(f))
    return [c for c in header if c.startswith("leaderboard_bbh_")]


def _run(cmd: List[str], *, env: dict[str, str]) -> None:
    pretty = " ".join(shlex.quote(s) for s in cmd)
    print(f"+ {pretty}", flush=True)
    subprocess.run(cmd, check=True, env=env)


def main() -> int:
    repo_root = _repo_root()

    p = argparse.ArgumentParser(
        description="Regenerate sigmoid_size and evaluation_pinball_baselines_size outputs."
    )
    p.add_argument(
        "--python",
        default=sys.executable,
        help="Python executable used to invoke the underlying scripts (default: current).",
    )
    p.add_argument(
        "--oll_csv",
        type=Path,
        default=repo_root
        / "tables/open_llm_leaderboard/open_llm_leaderboard_with_tokens.csv",
        help="Main Open LLM Leaderboard with-tokens CSV.",
    )
    p.add_argument(
        "--oll_bbh_csv",
        type=Path,
        default=repo_root
        / "tables/open_llm_leaderboard/open_llm_leaderboard_with_tokens_top50hub.csv",
        help="Top-hub Open LLM CSV containing leaderboard_bbh_* columns.",
    )
    p.add_argument(
        "--livebench_csv",
        type=Path,
        default=repo_root / "tables/livebench/livebench_with_tokens.csv",
        help="LiveBench with-tokens CSV (optional).",
    )
    p.add_argument(
        "--livebench_tasks_file",
        type=Path,
        default=repo_root / "tables/livebench/livebench_tasks_min30.txt",
        help="Task list (one per line) for LiveBench sigmoid plotting (optional).",
    )
    p.add_argument(
        "--size_col",
        default="#Params (B)",
        help="Column name used as x-axis for size-based runs.",
    )
    p.add_argument("--tau", type=float, default=0.98)
    p.add_argument("--lambda_b", type=float, default=1e-3)
    p.add_argument("--scatter_style", choices=["two_color", "family"], default="two_color")

    p.add_argument(
        "--out_sigmoid_dir",
        type=Path,
        default=repo_root / "outputs/sigmoid_size",
        help="Output root for sigmoid_size outputs.",
    )
    p.add_argument(
        "--baselines_main_dir",
        type=Path,
        default=repo_root / "evaluation_pinball_baselines_size/period4_singlek_no_budget",
    )
    p.add_argument(
        "--baselines_bbh_dir",
        type=Path,
        default=repo_root
        / "evaluation_pinball_baselines_size/period4_singlek_no_budget_bbh_subtasks",
    )
    p.add_argument(
        "--baselines_main_small_dir",
        type=Path,
        default=repo_root / "evaluation_pinball_baselines_size/period4_singlek_no_budget_small",
    )
    p.add_argument(
        "--baselines_bbh_small_dir",
        type=Path,
        default=repo_root
        / "evaluation_pinball_baselines_size/period4_singlek_no_budget_small_bbh_subtasks",
    )
    p.add_argument("--bins", type=int, default=10)
    p.add_argument("--min_bin_size", type=int, default=30)
    p.add_argument("--small_max_size", type=float, default=10.0)

    args = p.parse_args()

    env = os.environ.copy()
    env.setdefault("MPLCONFIGDIR", str(repo_root / ".cache/matplotlib"))
    env.setdefault("XDG_CACHE_HOME", str(repo_root / ".cache"))
    (repo_root / ".cache/matplotlib").mkdir(parents=True, exist_ok=True)

    smooth = repo_root / "scripts/smooth_single_skill_frontier.py"
    eval_baselines = repo_root / "scripts/evaluate/sigmoid_pinball_baselines.py"
    plot_baselines = repo_root / "scripts/plot/plot_pinball_baselines.py"

    # -------------------------------------------------------------------------
    # 1) Sigmoid size outputs (mirrors regenerate_all.sh Section 2)
    # -------------------------------------------------------------------------
    if not args.oll_csv.exists():
        raise FileNotFoundError(f"Missing --oll_csv: {args.oll_csv}")

    common_sigmoid = [
        args.python,
        str(smooth),
        "--csv",
        str(args.oll_csv),
        "--size_col",
        str(args.size_col),
        "--sigmoid",
        "--scatter_style",
        str(args.scatter_style),
        "--lambda_b",
        str(args.lambda_b),
        "--tau",
        str(args.tau),
    ]

    _run(
        common_sigmoid + ["--split_mode", "none", "--out_dir", str(args.out_sigmoid_dir / "no_split")],
        env=env,
    )
    _run(
        common_sigmoid + ["--split_mode", "year", "--out_dir", str(args.out_sigmoid_dir / "year_split")],
        env=env,
    )
    _run(
        common_sigmoid
        + [
            "--split_mode",
            "period4",
            "--period4_train_mode",
            "cumulative",
            "--out_dir",
            str(args.out_sigmoid_dir / "period4/cumulative"),
        ],
        env=env,
    )
    _run(
        common_sigmoid
        + [
            "--split_mode",
            "period4",
            "--period4_train_mode",
            "single_k",
            "--out_dir",
            str(args.out_sigmoid_dir / "period4/single_k"),
        ],
        env=env,
    )

    bbh_cols = _bbh_subtask_cols(args.oll_bbh_csv)
    if bbh_cols:
        _run(
            [
                args.python,
                str(smooth),
                "--csv",
                str(args.oll_bbh_csv),
                "--size_col",
                str(args.size_col),
                "--sigmoid",
                "--split_mode",
                "period4",
                "--period4_train_mode",
                "single_k",
                "--scatter_style",
                str(args.scatter_style),
                "--lambda_b",
                str(args.lambda_b),
                "--tau",
                str(args.tau),
                "--tasks",
                *bbh_cols,
                "--plots_subdir",
                "plots/bbh_subtasks",
                "--out_dir",
                str(args.out_sigmoid_dir / "period4/single_k"),
            ],
            env=env,
        )
    else:
        print(
            f"WARNING: skipping BBH subtasks sigmoid_size (no leaderboard_bbh_* cols in {args.oll_bbh_csv})",
            flush=True,
        )

    livebench_tasks = _read_lines(args.livebench_tasks_file)
    if args.livebench_csv.exists() and livebench_tasks:
        _run(
            [
                args.python,
                str(smooth),
                "--csv",
                str(args.livebench_csv),
                "--size_col",
                str(args.size_col),
                "--sigmoid",
                "--split_mode",
                "none",
                "--scatter_style",
                str(args.scatter_style),
                "--lambda_b",
                str(args.lambda_b),
                "--tau",
                str(args.tau),
                "--tasks",
                *livebench_tasks,
                "--out_dir",
                str(args.out_sigmoid_dir / "livebench/no_split"),
            ],
            env=env,
        )
    else:
        print("WARNING: skipping LiveBench sigmoid_size (missing CSV or tasks list).", flush=True)

    # -------------------------------------------------------------------------
    # 2) Pinball baseline comparisons (size-axis)
    # -------------------------------------------------------------------------
    _run(
        [
            args.python,
            str(eval_baselines),
            "--csv",
            str(args.oll_csv),
            "--size_col",
            str(args.size_col),
            "--task_group",
            "main",
            "--out_base",
            str(args.baselines_main_dir),
            "--small_max_size",
            str(args.small_max_size),
            "--out_base_small",
            str(args.baselines_main_small_dir),
            "--bins",
            str(args.bins),
            "--min_bin_size",
            str(args.min_bin_size),
        ],
        env=env,
    )

    if args.oll_bbh_csv.exists() and bbh_cols:
        _run(
            [
                args.python,
                str(eval_baselines),
                "--csv",
                str(args.oll_bbh_csv),
                "--size_col",
                str(args.size_col),
                "--task_group",
                "bbh_subtasks",
                "--out_base",
                str(args.baselines_bbh_dir),
                "--small_max_size",
                str(args.small_max_size),
                "--out_base_small",
                str(args.baselines_bbh_small_dir),
                "--bins",
                str(args.bins),
                "--min_bin_size",
                str(args.min_bin_size),
            ],
            env=env,
        )
    else:
        print("WARNING: skipping BBH subtasks baselines_size (missing CSV or columns).", flush=True)

    _run(
        [
            args.python,
            str(plot_baselines),
            "--base_dir_main",
            str(args.baselines_main_dir),
            "--base_dir_bbh",
            str(args.baselines_bbh_dir),
        ],
        env=env,
    )
    _run(
        [
            args.python,
            str(plot_baselines),
            "--base_dir_main",
            str(args.baselines_main_small_dir),
            "--base_dir_bbh",
            str(args.baselines_bbh_small_dir),
        ],
        env=env,
    )

    print("\n✓ Size-axis regeneration complete.", flush=True)
    print(f"  - sigmoid_size: {args.out_sigmoid_dir}", flush=True)
    print(f"  - pinball_baselines_size: {args.baselines_main_dir.parent}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


"""Shared argparse helpers for plotting scripts."""

from __future__ import annotations

from typing import Sequence


def add_task_out_args(
    parser,
    *,
    default_task: str,
    tasks: Sequence[str],
    default_out: str,
    task_help: str = "Task to plot.",
    out_help: str = "Output PDF path.",
) -> None:
    parser.add_argument("--task", default=default_task, choices=list(tasks), help=task_help)
    parser.add_argument("--out", default=default_out, help=out_help)

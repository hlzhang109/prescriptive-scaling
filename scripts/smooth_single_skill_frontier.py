#!/usr/bin/env python3
"""Compatibility wrapper for ``skill_frontier.core.sigmoid``.

This preserves the legacy CLI/module path:
  - ``python scripts/smooth_single_skill_frontier.py ...``
  - ``from scripts.smooth_single_skill_frontier import ...``
"""

from __future__ import annotations

import os
import sys
from typing import Optional

# Import bootstrap from `scripts/` by putting the scripts dir on sys.path.
SCRIPTS_DIR = os.path.abspath(os.path.dirname(__file__))
if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)
from _bootstrap import ensure_repo_root_on_path  # type: ignore

ensure_repo_root_on_path(__file__)

from skill_frontier.core import sigmoid as _impl  # type: ignore  # noqa: E402
from skill_frontier.io.output_paths import sanitize_task_name  # type: ignore  # noqa: E402

# Common explicit re-exports used throughout the repo.
DEFAULT_LAMBDA_B = _impl.DEFAULT_LAMBDA_B
LAMBDA_B = _impl.LAMBDA_B
PERIOD4_BOUNDS = _impl.PERIOD4_BOUNDS
PERIOD4_SPLITS = _impl.PERIOD4_SPLITS
PERIOD4_SPLITS_SINGLE = _impl.PERIOD4_SPLITS_SINGLE
fit_sigmoid_frontier = _impl.fit_sigmoid_frontier
smooth_frontier = _impl.smooth_frontier
main = _impl.main


def _get_plot_path(
    out_dir: str,
    task: str,
    legacy: bool = False,
    suffix: str = "",
    plots_subdir: str = "plots",
) -> str:
    """Get path for plot file (without extension)."""
    task_clean = sanitize_task_name(task)

    if legacy:
        if suffix:
            return os.path.join(out_dir, f"smooth_frontier_{suffix}__{task}")
        return os.path.join(out_dir, f"smooth_frontier__{task}")

    plots_dir = os.path.join(out_dir, plots_subdir)
    os.makedirs(plots_dir, exist_ok=True)
    if suffix:
        return os.path.join(plots_dir, f"{task_clean}_{suffix}")
    return os.path.join(plots_dir, task_clean)


def _get_curve_path(
    out_dir: str,
    task: str,
    legacy: bool = False,
    group: Optional[str] = None,
    k: Optional[int] = None,
) -> str:
    """Get path for curve CSV file."""
    task_clean = sanitize_task_name(task)

    if legacy:
        if k is not None:
            return os.path.join(out_dir, f"smooth_frontier__{task}__k{k}.csv")
        if group:
            group_clean = group.replace(" ", "_")
            return os.path.join(out_dir, f"smooth_frontier__{task}__{group_clean}.csv")
        return os.path.join(out_dir, f"smooth_frontier__{task}.csv")

    curves_dir = os.path.join(out_dir, "curves")
    os.makedirs(curves_dir, exist_ok=True)
    if k is not None:
        return os.path.join(curves_dir, f"{task_clean}_k{k}.csv")
    if group:
        group_clean = group.replace(" ", "_").lower()
        return os.path.join(curves_dir, f"{task_clean}_{group_clean}.csv")
    return os.path.join(curves_dir, f"{task_clean}.csv")


def __getattr__(name: str):  # noqa: ANN001
    return getattr(_impl, name)


if __name__ == "__main__":
    raise SystemExit(main())

"""Shared helpers for restyle curve lookup."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Tuple

import pandas as pd  # type: ignore


def task_from_curve(curve_csv: Path) -> str:
    df = pd.read_csv(curve_csv, usecols=["task"], nrows=1)
    if df.empty:
        raise ValueError(f"Empty curve CSV: {curve_csv}")
    return str(df.iloc[0]["task"])


def iter_plot_slugs(plot_dir: Path) -> Iterable[Tuple[str, Path]]:
    # Yield (slug, png_path) for each existing triptych PNG.
    for p in sorted(plot_dir.glob("*_period4.png")):
        slug = p.name[: -len("_period4.png")]
        yield slug, p

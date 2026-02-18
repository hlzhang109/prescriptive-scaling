"""Shared period4 parsing helpers for restyle scripts."""

from __future__ import annotations

from typing import Optional, Sequence, Tuple


def parse_year_month_simple(date_str: object) -> Optional[Tuple[int, int]]:
    if date_str is None:
        return None
    s = str(date_str).strip()
    if not s or s.lower() == "nan":
        return None
    # Allow ISO-like timestamps.
    if "T" in s:
        s = s.split("T", 1)[0]
    parts = s.split("-")
    if len(parts) < 2:
        return None
    try:
        return int(parts[0]), int(parts[1])
    except Exception:
        return None


def _in_range(ym: Tuple[int, int], lo: Tuple[int, int], hi: Tuple[int, int]) -> bool:
    return lo <= ym <= hi


def assign_period_index(
    ym: Optional[Tuple[int, int]],
    bounds: Sequence[Tuple[str, Tuple[int, int], Tuple[int, int]]],
) -> int:
    if ym is None:
        return -1
    for idx, (_, lo, hi) in enumerate(bounds):
        if _in_range(ym, lo, hi):
            return idx
    return -1

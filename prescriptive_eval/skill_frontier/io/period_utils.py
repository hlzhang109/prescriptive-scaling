"""Shared period/date utilities.

Contains behavior-preserving helpers that were previously duplicated in scripts.
"""

from __future__ import annotations

from typing import Optional, Tuple

import pandas as pd


def parse_year_month(date_str: str) -> Optional[Tuple[int, int]]:
    """Parse date string to (year, month) tuple."""
    if not date_str or pd.isna(date_str):
        return None
    try:
        # Handle various date formats
        if 'T' in str(date_str):
            date_str = str(date_str).split('T')[0]
        parts = str(date_str).split('-')
        if len(parts) >= 2:
            return (int(parts[0]), int(parts[1]))
    except Exception:
        pass
    return None

"""Consolidated CSV I/O and data processing utilities.

This module contains all the common CSV reading, parsing, and data extraction
functions that were previously duplicated across multiple scripts.
"""

from __future__ import annotations

import csv as _csv
import datetime
import math
import os
import re
from typing import List, Optional, Sequence, Tuple

import numpy as np

DEFAULT_LEADERBOARD_RESULTS_PATH = os.path.join(
    "tables",
    "open_llm_leaderboard",
    "open_llm_leaderboard_with_tokens.csv",
)
DEFAULT_NEW_LEADERBOARD_RESULTS_PATH = os.path.join(
    "tables",
    "new_leaderboard_results_with_tokens.csv",
)


def load_leaderboard_results(path: str = DEFAULT_LEADERBOARD_RESULTS_PATH):
    import pandas as pd  # type: ignore

    return pd.read_csv(path)


def read_csv_rows(path: str) -> Tuple[List[dict], List[str]]:
    """Read CSV file and return rows as dictionaries along with headers.

    Args:
        path: Path to CSV file

    Returns:
        Tuple of (rows, headers) where rows is a list of dicts and headers is a list of column names
    """
    with open(path, "r", newline="") as f:
        reader = _csv.DictReader(f)
        rows = list(reader)
        headers = list(reader.fieldnames or [])
    return rows, headers


def detect_date_col(headers: List[str]) -> Optional[str]:
    """Detect the date column from CSV headers.

    Args:
        headers: List of column names

    Returns:
        Name of the date column if found, None otherwise
    """
    if "Upload To Hub Date" in headers:
        return "Upload To Hub Date"
    if "date" in headers:
        return "date"
    return None


def parse_date(s: str) -> Optional[int]:
    """Parse a date string and extract the year.

    Tries multiple date formats and falls back to regex extraction.

    Args:
        s: Date string to parse

    Returns:
        Year as integer if successfully parsed, None otherwise
    """
    s = (s or "").strip()
    if not s or s.lower() in ("nan", "none", "na"):
        return None

    # Try common date formats
    for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%d/%m/%Y", "%m/%d/%Y", "%Y-%m", "%Y/%m", "%Y"):
        try:
            return datetime.datetime.strptime(s, fmt).year
        except Exception:
            pass

    # Try regex extraction for 20XX years
    m = re.search(r"(20\d{2})", s)
    if m:
        try:
            return int(m.group(1))
        except Exception:
            return None
    return None


def parse_year_month(s: str) -> Optional[Tuple[int, int]]:
    """Parse a date string and extract year and month.

    Args:
        s: Date string to parse

    Returns:
        Tuple of (year, month) if successfully parsed, None otherwise
    """
    s = (s or "").strip()
    if not s or s.lower() in ("nan", "none", "na"):
        return None

    # Try formats with month information
    for fmt in (
        "%Y-%m-%d",
        "%Y/%m/%d",
        "%Y-%m-%dT%H:%M:%SZ",   # e.g. "2024-03-21T06:05:50Z"
        "%Y-%m-%dT%H:%M:%S%z",  # e.g. "2024-03-21T06:05:50+00:00"
        "%Y-%m-%d %H:%M:%S%z",  # e.g. "2024-09-27 15:52:33+00:00"
        "%Y-%m-%d %H:%M:%S",    # e.g. "2024-09-27 15:52:33"
        "%d/%m/%Y",
        "%m/%d/%Y",
        "%Y-%m",
        "%Y/%m",
    ):
        try:
            dt = datetime.datetime.strptime(s, fmt)
            return (dt.year, dt.month)
        except Exception:
            pass
    return None


def detect_date_col_flexible(headers: Sequence[str]) -> Optional[str]:
    """Detect a date column using flexible heuristics.

    Args:
        headers: List of column names

    Returns:
        Name of the date column if found, None otherwise
    """
    if "Upload To Hub Date" in headers:
        return "Upload To Hub Date"
    if "Submission Date" in headers:
        return "Submission Date"
    for col in headers:
        col_lower = col.lower()
        if "date" in col_lower or "submission" in col_lower:
            return col
    return None


def detect_oll_raw_tasks(headers: List[str]) -> List[str]:
    """Auto-detect Open LLM Leaderboard raw task columns.

    Args:
        headers: List of column names from CSV

    Returns:
        List of canonical OLL task column names that exist in headers
    """
    canonical_new = [
        "IFEval Raw",
        "BBH Raw",
        "MATH Lvl 5 Raw",
        "GPQA Raw",
        "MUSR Raw",
        "MMLU-PRO Raw",
    ]
    # Older OLL schema uses percentage scores (0..100) for these tasks.
    canonical_old = [
        "ARC",
        "HellaSwag",
        "MMLU",
        "TruthfulQA",
        "Winogrande",
        "GSM8K",
    ]
    hset = set(h.strip() for h in headers)
    if any(c in hset for c in canonical_new):
        return [c for c in canonical_new if c in hset]
    return [c for c in canonical_old if c in hset]


def maybe_scale_task_values(values: np.ndarray) -> np.ndarray:
    """Auto-scale task values that look like percentages to [0, 1].

    Heuristic: if the 95th percentile is > 1 and <= 100, treat values as
    percentages and divide by 100. Otherwise return unchanged.
    """
    arr = np.asarray(values, dtype=float)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return arr
    try:
        p95 = float(np.nanpercentile(finite, 95))
    except Exception:
        p95 = float(np.nanmax(finite))
    if (p95 > 1.0 + 1e-9) and (p95 <= 100.0 + 1e-9):
        return arr / 100.0
    return arr


def compute_flops(
    row: dict,
    headers: List[str],
    logC_col: Optional[str] = None,
    prod_cols: Optional[Tuple[str, str]] = None,
    mult: float = 1.0,
) -> float:
    """Compute FLOPs from a CSV row using various column configurations.

    Args:
        row: CSV row as dictionary
        headers: List of all column names
        logC_col: Name of log-compute column (if present)
        prod_cols: Tuple of two column names to multiply for compute (e.g., tokens and batch)
        mult: Multiplier to apply to product (e.g., 6 for 6*T*B)

    Returns:
        Computed FLOPs as float, or NaN if unable to compute
    """
    # Try logC column first
    if logC_col and logC_col in headers:
        try:
            lc = float(row.get(logC_col, "nan"))
            return float(math.exp(lc))
        except Exception:
            return float("nan")

    # Try product of two columns
    if prod_cols and all(c in headers for c in prod_cols):
        v1 = row.get(prod_cols[0], None)
        v2 = row.get(prod_cols[1], None)
        try:
            t = float(v1) if v1 not in (None, "", "nan", "NaN") else float("nan")
        except Exception:
            t = float("nan")
        try:
            b = float(v2) if v2 not in (None, "", "nan", "NaN") else float("nan")
        except Exception:
            b = float("nan")
        if np.isfinite(t) and np.isfinite(b):
            return float(mult * t * b)
        return float("nan")

    return float("nan")


def sanitize_name(name: str) -> str:
    """Sanitize a string to be safe for use in filenames.

    Args:
        name: String to sanitize (e.g., task name)

    Returns:
        Sanitized string with path separators replaced
    """
    return name.replace("/", "_").replace("\\", "_")


def extract_model_id(row: dict) -> str:
    """Extract model ID from a CSV row, trying multiple column names.

    Args:
        row: CSV row as dictionary

    Returns:
        Model ID string, or empty string if not found
    """
    return str(
        row.get("model", "")
        or row.get("Model sha", "")
        or row.get("name", "")
        or ""
    ).strip()


def collect_model_ids(path: str) -> set[str]:
    """Collect model IDs from a CSV file, skipping empty IDs."""
    rows, _ = read_csv_rows(path)
    ids = set()
    for row in rows:
        mid = extract_model_id(row)
        if mid:
            ids.add(mid)
    return ids

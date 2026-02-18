"""Shared period4 split utilities.

Keeps the period index assignment logic centralized while preserving existing
fallback behavior for importing PERIOD4_BOUNDS.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple


def _import_period_scheme():
    try:
        from skill_frontier.core.period_scheme import (  # type: ignore
            PERIOD4_BOUNDS,
            PERIOD4_SPLITS_SINGLE,
        )
    except Exception:
        from .period_scheme import PERIOD4_BOUNDS, PERIOD4_SPLITS_SINGLE  # type: ignore
    return PERIOD4_BOUNDS, PERIOD4_SPLITS_SINGLE


def _load_period4_bounds():
    PERIOD4_BOUNDS, _ = _import_period_scheme()
    return PERIOD4_BOUNDS


def assign_period_index_period4(year: int, month: int) -> int:
    """Assign period index (0,1,2,3) for the 4-period split."""
    PERIOD4_BOUNDS = _load_period4_bounds()

    for i, bound in enumerate(PERIOD4_BOUNDS):
        if len(bound) == 3:
            _, (y_lo, m_lo), (y_hi, m_hi) = bound
        else:
            y_lo, m_lo, y_hi, m_hi = bound
        if (year, month) >= (y_lo, m_lo) and (year, month) <= (y_hi, m_hi):
            return i
    return -1


def assign_period_index_period4_one_based(year: int, month: int) -> int:
    """Assign period index (1,2,3,4) for the 4-period split, or -1 if out of range."""
    idx = assign_period_index_period4(year, month)
    return idx + 1 if idx >= 0 else -1


def normalize_period4_splits_single() -> List[Tuple[int, int]]:
    """Return list of (train_idx, val_idx) for PERIOD4_SPLITS_SINGLE (0-based)."""
    PERIOD4_BOUNDS, PERIOD4_SPLITS_SINGLE = _import_period_scheme()

    label_to_idx: Dict[str, int] = {}
    for i, bound in enumerate(PERIOD4_BOUNDS):
        label = bound[0] if len(bound) == 3 else f"k{i+1}"
        label_to_idx[str(label)] = int(i)

    out: List[Tuple[int, int]] = []
    for split in PERIOD4_SPLITS_SINGLE:
        train_labels = split.get("train_labels", [])
        val_label = split.get("val_label", None)
        if not train_labels or val_label is None:
            continue
        train_idx = [label_to_idx[l] for l in train_labels if l in label_to_idx]
        val_idx = label_to_idx[val_label] if val_label in label_to_idx else None
        if not train_idx or val_idx is None:
            continue
        out.append((int(train_idx[-1]), int(val_idx)))
    return out


def parse_period_token(token: str) -> Optional[int]:
    s = str(token).strip().upper()
    if not s:
        return None
    if s.startswith("P"):
        s = s[1:]
    try:
        return int(s)
    except Exception:
        return None


def parse_period_list(value: str) -> List[int]:
    out: List[int] = []
    for tok in str(value).split(","):
        p = parse_period_token(tok)
        if p is not None and p not in out:
            out.append(p)
    return out

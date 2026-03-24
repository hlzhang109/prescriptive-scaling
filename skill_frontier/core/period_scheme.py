"""Centralized period-4 split configuration.

This module defines both current (new OLL) and legacy (old OLL) period-4
schemes and resolves the active one via ``SKILL_FRONTIER_PERIOD4_SCHEME``.
"""

from __future__ import annotations

import os
from typing import Dict, List, Tuple


PeriodBounds = List[Tuple[str, Tuple[int, int], Tuple[int, int]]]
PeriodSplits = List[Dict[str, str | List[str]]]


PERIOD4_BOUNDS_NEW: PeriodBounds = [
    ("<=2024-06", (1900, 1), (2024, 6)),
    ("2024-07..2024-09", (2024, 7), (2024, 9)),
    ("2024-10..2024-12", (2024, 10), (2024, 12)),
    ("2025-01..2025-03", (2025, 1), (2025, 3)),
]
PERIOD4_SPLITS_NEW: PeriodSplits = [
    {"train_labels": ["<=2024-06"], "val_label": "2024-07..2024-09"},
    {"train_labels": ["<=2024-06", "2024-07..2024-09"], "val_label": "2024-10..2024-12"},
    {"train_labels": ["<=2024-06", "2024-07..2024-09", "2024-10..2024-12"], "val_label": "2025-01..2025-03"},
]
PERIOD4_SPLITS_SINGLE_NEW: PeriodSplits = [
    {"train_labels": ["<=2024-06"], "val_label": "2024-07..2024-09"},
    {"train_labels": ["2024-07..2024-09"], "val_label": "2024-10..2024-12"},
    {"train_labels": ["2024-10..2024-12"], "val_label": "2025-01..2025-03"},
]

PERIOD4_BOUNDS_OLL_OLD: PeriodBounds = [
    ("2023-07..2023-11", (2023, 7), (2023, 11)),
    ("2023-12..2024-01", (2023, 12), (2024, 1)),
    ("2024-02..2024-03", (2024, 2), (2024, 3)),
    ("2024-04..2024-06", (2024, 4), (2024, 6)),
]
PERIOD4_SPLITS_OLL_OLD: PeriodSplits = [
    {"train_labels": ["2023-07..2023-11"], "val_label": "2023-12..2024-01"},
    {"train_labels": ["2023-07..2023-11", "2023-12..2024-01"], "val_label": "2024-02..2024-03"},
    {"train_labels": ["2023-07..2023-11", "2023-12..2024-01", "2024-02..2024-03"], "val_label": "2024-04..2024-06"},
]
PERIOD4_SPLITS_SINGLE_OLL_OLD: PeriodSplits = [
    {"train_labels": ["2023-07..2023-11"], "val_label": "2023-12..2024-01"},
    {"train_labels": ["2023-12..2024-01"], "val_label": "2024-02..2024-03"},
    {"train_labels": ["2024-02..2024-03"], "val_label": "2024-04..2024-06"},
]


def resolve_period4_scheme() -> str:
    """Return the active scheme key: ``new`` (default) or ``oll_old``."""
    value = os.getenv("SKILL_FRONTIER_PERIOD4_SCHEME", "").strip().lower()
    if value in {"oll_old", "old", "old_oll"}:
        return "oll_old"
    return "new"


def resolve_period4_config(
    scheme: str | None = None,
) -> Tuple[str, PeriodBounds, PeriodSplits, PeriodSplits]:
    """Resolve bounds/splits for a scheme (or current env-selected scheme)."""
    active = (scheme or resolve_period4_scheme()).strip().lower()
    if active in {"oll_old", "old", "old_oll"}:
        return (
            "oll_old",
            PERIOD4_BOUNDS_OLL_OLD,
            PERIOD4_SPLITS_OLL_OLD,
            PERIOD4_SPLITS_SINGLE_OLL_OLD,
        )
    return ("new", PERIOD4_BOUNDS_NEW, PERIOD4_SPLITS_NEW, PERIOD4_SPLITS_SINGLE_NEW)


PERIOD4_SCHEME, PERIOD4_BOUNDS, PERIOD4_SPLITS, PERIOD4_SPLITS_SINGLE = resolve_period4_config()

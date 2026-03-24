"""Boolean parsing helpers."""

from __future__ import annotations


def is_true(value: object) -> bool:
    return str(value).strip().lower() in {"1", "true", "t", "yes", "y"}

"""Utilities for reading and writing manifest files.

Manifest files contain lists of model IDs (one per line) used to filter
datasets for train/test splits or budget-constrained selections.
"""

from __future__ import annotations

import os
from typing import List, Optional, Set


def read_manifest(path: str) -> Set[str]:
    """Read a manifest file and return the set of model IDs.

    Args:
        path: Path to manifest file (text file with one model ID per line)

    Returns:
        Set of model ID strings (stripped of whitespace, excluding empty lines)
    """
    if not path or not os.path.isfile(path):
        return set()
    try:
        with open(path, "r") as f:
            return set(line.strip() for line in f if line.strip())
    except (OSError, UnicodeDecodeError):
        return set()


def read_manifest_optional(path: Optional[str]) -> Optional[Set[str]]:
    """Read a manifest file and return the set of model IDs if available.

    Returns None when the path is missing, unreadable, or the file is absent.
    """
    if not path:
        return None
    if not os.path.isfile(path):
        return None
    try:
        with open(path, "r") as f:
            return set(line.strip() for line in f if line.strip())
    except (OSError, UnicodeDecodeError):
        return None


def write_manifest(path: str, model_ids: List[str]) -> None:
    """Write a manifest file with one model ID per line.

    Args:
        path: Path to output manifest file
        model_ids: List of model ID strings to write
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="") as f:
        for mid in model_ids:
            f.write(f"{mid}\n")


def write_manifest_from_indices(
    path: str, model_ids: List[str], indices: List[int]
) -> None:
    """Write a manifest file with model IDs selected by indices.

    Args:
        path: Path to output manifest file
        model_ids: Complete list of model IDs
        indices: List of integer indices to select from model_ids
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="") as f:
        for i in indices:
            f.write(f"{model_ids[i]}\n")

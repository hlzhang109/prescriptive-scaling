"""Bootstrap helpers for running scripts directly via `python scripts/...`.

These helpers keep entrypoints lightweight while ensuring repo-local imports work
regardless of the current working directory.
"""

from __future__ import annotations

import os
import sys
from typing import Optional


def find_repo_root(start_path: str, *, max_hops: int = 6) -> str:
    """Walk upward from start_path to find the repo root.

    Repo root heuristic: contains both `skill_frontier/` and `scripts/`.
    """
    d = os.path.abspath(os.path.dirname(start_path))
    for _ in range(max_hops):
        if os.path.isdir(os.path.join(d, "skill_frontier")) and os.path.isdir(os.path.join(d, "scripts")):
            return d
        parent = os.path.dirname(d)
        if parent == d:
            break
        d = parent
    return os.path.abspath(os.path.join(os.path.dirname(start_path), os.pardir))


def ensure_repo_root_on_path(start_path: str) -> str:
    """Insert the repo root into sys.path if missing; return repo root path."""
    repo_root = find_repo_root(start_path)
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    return repo_root


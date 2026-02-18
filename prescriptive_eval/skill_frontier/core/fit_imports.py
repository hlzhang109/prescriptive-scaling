"""Import helpers for fit_sigmoid_frontier with ordered fallbacks."""

from __future__ import annotations


def import_fit_sigmoid_frontier_basic():
    """Import fit_sigmoid_frontier from the canonical core module."""
    try:
        from skill_frontier.core.sigmoid import fit_sigmoid_frontier  # type: ignore
    except Exception:
        from .sigmoid import fit_sigmoid_frontier  # type: ignore
    return fit_sigmoid_frontier


def import_fit_sigmoid_frontier_extended():
    """Import fit_sigmoid_frontier from the canonical core module."""
    try:
        from skill_frontier.core.sigmoid import fit_sigmoid_frontier  # type: ignore
    except Exception:
        from .sigmoid import fit_sigmoid_frontier  # type: ignore
    return fit_sigmoid_frontier

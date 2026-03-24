"""Compatibility wrapper for legacy imports from ``scripts.skill_frontier_window``.

Canonical implementation lives in ``skill_frontier.core.window``.
"""

from __future__ import annotations

import os
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from skill_frontier.core.window import *  # noqa: F401,F403

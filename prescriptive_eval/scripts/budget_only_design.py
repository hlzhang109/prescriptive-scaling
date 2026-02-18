"""Compatibility wrapper for ``scripts.budget_only_design``.

Canonical implementation lives in ``skill_frontier.core.budget_design``.
"""

from __future__ import annotations

import os
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from skill_frontier.core.budget_design import *  # noqa: F401,F403

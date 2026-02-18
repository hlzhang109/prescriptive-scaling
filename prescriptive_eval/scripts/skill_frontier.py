"""Compatibility wrapper for the legacy ``scripts/skill_frontier.py`` entrypoint.

Canonical implementation lives in ``skill_frontier.core.frontier``.
"""

from __future__ import annotations

import os
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from skill_frontier.core.frontier import *  # noqa: F401,F403
from skill_frontier.core.frontier import main


if __name__ == "__main__":  # pragma: no cover
    main()

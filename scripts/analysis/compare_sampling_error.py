#!/usr/bin/env python3
"""Compatibility wrapper for ``scripts/analysis/compare_sampling_error.py``.

Canonical implementation lives in ``scripts/evaluate/sampling_error.py``.
"""

from __future__ import annotations

import os
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from scripts.evaluate.sampling_error import *  # noqa: F401,F403
from scripts.evaluate.sampling_error import main


if __name__ == "__main__":
    main()


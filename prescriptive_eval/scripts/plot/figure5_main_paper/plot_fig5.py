#!/usr/bin/env python3
"""Compatibility wrapper for the Figure 5 plot script.

Canonical location: `scripts/plot/paper/figure5/plot_fig5.py`.
"""

from __future__ import annotations

import os
import sys

# Import bootstrap from `scripts/` by putting the scripts dir on sys.path.
SCRIPTS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)
from _bootstrap import ensure_repo_root_on_path  # type: ignore

ensure_repo_root_on_path(__file__)

from scripts.plot.paper.figure5.plot_fig5 import main  # type: ignore  # noqa: E402


if __name__ == "__main__":
    main()


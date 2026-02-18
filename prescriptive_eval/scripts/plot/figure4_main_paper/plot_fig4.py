#!/usr/bin/env python3
"""Compatibility wrapper for the Figure 4 plot script.

Canonical location: `scripts/plot/paper/figure4/plot_fig4.py`.

This wrapper also re-exports module-level helpers so other scripts can keep
importing `scripts.plot.figure4_main_paper.plot_fig4` as a module.
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

from scripts.plot.paper.figure4 import plot_fig4 as _impl  # type: ignore  # noqa: E402

# Export common entrypoint explicitly.
main = _impl.main


def __getattr__(name: str):  # noqa: ANN001
    return getattr(_impl, name)


if __name__ == "__main__":
    main()

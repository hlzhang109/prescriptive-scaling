#!/usr/bin/env python3
"""Figure 21 (enhanced): OOS absolute coverage error heatmaps over (kappa, lambda)."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.plot.experiments._kappa_lambda_heatmap import render


def main() -> None:
    render(
        metric="oos_calib_abs",
        suptitle=r"OOS calibration $|\hat{\tau}-\tau|$",
        out_pdf_name="figure21_enhanced.pdf",
    )


if __name__ == "__main__":
    main()


#!/usr/bin/env python3
"""Export the Artificial Analysis MATH-500 leaderboard."""

from __future__ import annotations

import os
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from scripts.data._download_artificial_analysis_common import ExportConfig, run_cli


def main() -> None:
    run_cli(
        ExportConfig(
            label="MATH-500",
            eval_key="math_500",
            score_col="math_500",
            score_pct_col="math_500_pct",
            out_csv_name="math-500.csv",
            out_info_name="math-500_download_info.json",
            include_release_date=False,
        )
    )


if __name__ == "__main__":
    main()

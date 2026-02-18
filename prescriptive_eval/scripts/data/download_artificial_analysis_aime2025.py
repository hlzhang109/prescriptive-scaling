#!/usr/bin/env python3
"""Export the Artificial Analysis AIME 2025 leaderboard."""

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
            label="AIME 2025",
            eval_key="aime_25",
            score_col="aime_2025",
            score_pct_col="aime_2025_pct",
            out_csv_name="aime-2025.csv",
            out_info_name="aime-2025_download_info.json",
            include_release_date=True,
        )
    )


if __name__ == "__main__":
    main()

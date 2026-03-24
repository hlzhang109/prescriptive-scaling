"""Shared Figure-5 data prep helpers.

Contains period assignment utilities and loaders used by Figure 5 scripts.
"""

from __future__ import annotations

import math
from typing import Sequence, Tuple

import numpy as np
import pandas as pd

from skill_frontier.core.sigmoid import PERIOD4_BOUNDS
from skill_frontier.io.csv_utils import load_leaderboard_results, parse_year_month
from skill_frontier.io.task_mappings import MAIN_TASKS, MAIN_TASK_MAP_OLL_TO_NEW

def _assign_period_index(bounds: Sequence[Tuple[str, Tuple[int, int], Tuple[int, int]]], ym: Tuple[int, int]) -> int:
    y, m = ym
    for idx, (_lab, (y_lo, m_lo), (y_hi, m_hi)) in enumerate(bounds):
        if (y, m) >= (y_lo, m_lo) and (y, m) <= (y_hi, m_hi):
            return idx
    return -1


def _make_period_ids_from_dates(date_series: pd.Series, *, after_p4_as_p5: bool) -> np.ndarray:
    out = np.full(shape=(len(date_series),), fill_value=-1, dtype=int)
    p4_end = PERIOD4_BOUNDS[-1][2]
    for i, v in enumerate(date_series.tolist()):
        if v is None or (isinstance(v, float) and math.isnan(v)):
            continue
        ym = parse_year_month(str(v))
        if ym is None:
            continue
        if after_p4_as_p5 and ym > p4_end:
            out[i] = 4
        else:
            out[i] = _assign_period_index(PERIOD4_BOUNDS, ym)
    return out


def _compute_old_compute_zflops(df_oll: pd.DataFrame) -> pd.Series:
    tokens = pd.to_numeric(df_oll.get("Pretraining tokens (T)", np.nan), errors="coerce")
    params = pd.to_numeric(df_oll.get("#Params (B)", np.nan), errors="coerce")
    return 6.0 * tokens * params


def load_old_oll(oll_csv: str, *, task: str) -> pd.DataFrame:
    df = load_leaderboard_results(oll_csv)
    out = pd.DataFrame()
    out["compute_zflops"] = _compute_old_compute_zflops(df)
    out["pid"] = _make_period_ids_from_dates(
        df.get("Upload To Hub Date", pd.Series([np.nan] * len(df))),
        after_p4_as_p5=False,
    )
    out[task] = pd.to_numeric(df.get(task, np.nan), errors="coerce")
    base = df.get("Base model family")
    if base is None:
        base = df.get("Identified base model")
    if base is None:
        base = df.get("Base Model")
    out["base_model"] = base if base is not None else np.nan
    return out


def load_new_models(metrics_csv: str, compute_csv: str, top_models_csv: str, *, task: str) -> pd.DataFrame:
    metric_col = MAIN_TASK_MAP_OLL_TO_NEW.get(task)
    if metric_col is None:
        raise ValueError(f"Unsupported task for new-models CSV: {task!r}")

    df_metrics = pd.read_csv(metrics_csv)
    df_compute = pd.read_csv(compute_csv, usecols=["model_id", "pretrain_compute_zflops"])
    # NOTE: validation_leaderboard.csv already contains `mapped_base_model`. We only need
    # `last_modified` here for period assignment; including `mapped_base_model` would
    # create a merge collision (pandas suffixes) and drop the unsuffixed column.
    df_meta = pd.read_csv(top_models_csv, usecols=["model_id", "last_modified"])

    df_new = df_metrics.merge(df_compute, on="model_id", how="left", validate="one_to_one")
    df_new = df_new.merge(df_meta, on="model_id", how="left", validate="one_to_one")

    out = pd.DataFrame()
    out["model_id"] = df_new.get("model_id", np.nan)
    out["compute_zflops"] = pd.to_numeric(df_new.get("pretrain_compute_zflops", np.nan), errors="coerce")
    out["pid"] = _make_period_ids_from_dates(
        df_new.get("last_modified", pd.Series([np.nan] * len(df_new))),
        after_p4_as_p5=True,
    )
    out[task] = pd.to_numeric(df_new.get(metric_col, np.nan), errors="coerce")
    out["base_model"] = df_new.get("mapped_base_model", np.nan)
    return out

"""Shared exporter for Artificial Analysis leaderboard slices."""

from __future__ import annotations

import argparse
import csv
import datetime as _dt
import json
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import requests


DEFAULT_URL = "https://artificialanalysis.ai/api/v2/data/llms/models"


@dataclass(frozen=True)
class ExportConfig:
    label: str
    eval_key: str
    score_col: str
    score_pct_col: str
    out_csv_name: str
    out_info_name: str
    include_release_date: bool = False


def _safe_mkdir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _hours_since_mtime(path: str) -> Optional[float]:
    try:
        mtime = os.path.getmtime(path)
    except OSError:
        return None
    return (time.time() - float(mtime)) / 3600.0


def _load_json(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        return json.load(f)


def _write_json(path: str, obj: Dict[str, Any]) -> None:
    with open(path, "w") as f:
        json.dump(obj, f, separators=(",", ":"))
        f.write("\n")


def _fetch_models_json(url: str, api_key: str, *, timeout_s: int) -> Dict[str, Any]:
    resp = requests.get(url, headers={"x-api-key": api_key}, timeout=timeout_s)
    resp.raise_for_status()
    payload = resp.json()
    if not isinstance(payload, dict) or "data" not in payload:
        raise RuntimeError("Unexpected response schema: missing 'data' field.")
    return payload


def _extract_rows(payload: Dict[str, Any], cfg: ExportConfig) -> List[Dict[str, Any]]:
    data = payload.get("data")
    if not isinstance(data, list):
        raise RuntimeError("Unexpected response schema: 'data' is not a list.")

    rows: List[Dict[str, Any]] = []
    for model in data:
        if not isinstance(model, dict):
            continue

        evals = model.get("evaluations") or {}
        if not isinstance(evals, dict):
            continue

        value = evals.get(cfg.eval_key)
        if value is None:
            continue
        try:
            score = float(value)
        except Exception:
            continue

        creator = None
        mc = model.get("model_creator")
        if isinstance(mc, dict):
            creator = mc.get("name")

        pricing = model.get("pricing") or {}
        if not isinstance(pricing, dict):
            pricing = {}

        row: Dict[str, Any] = {
            "model_id": model.get("id"),
            "model_name": model.get("name"),
            "creator": creator,
            cfg.score_col: score,
            cfg.score_pct_col: round(score * 100.0, 2),
            "price_1m_blended_3_to_1": pricing.get("price_1m_blended_3_to_1"),
            "median_output_tokens_per_second": model.get("median_output_tokens_per_second"),
            "median_time_to_first_token_seconds": model.get("median_time_to_first_token_seconds"),
        }
        if cfg.include_release_date:
            row["release_date"] = model.get("release_date")
        rows.append(row)

    rows.sort(key=lambda r: float(r[cfg.score_col]), reverse=True)
    for i, row in enumerate(rows, start=1):
        row["rank"] = i
    return rows


def _fieldnames(cfg: ExportConfig) -> List[str]:
    names = [
        "rank",
        "model_id",
        "model_name",
        "creator",
    ]
    if cfg.include_release_date:
        names.append("release_date")
    names.extend(
        [
            cfg.score_col,
            cfg.score_pct_col,
            "price_1m_blended_3_to_1",
            "median_output_tokens_per_second",
            "median_time_to_first_token_seconds",
        ]
    )
    return names


def run_cli(cfg: ExportConfig) -> None:
    ap = argparse.ArgumentParser(
        description=f"Download Artificial Analysis models data and export {cfg.label} leaderboard CSV."
    )
    ap.add_argument("--out_dir", default=os.path.join("tables", "artificial_analysis"), help="Output directory")
    ap.add_argument("--url", default=DEFAULT_URL, help="AA Data API URL")
    ap.add_argument(
        "--api_key_env",
        default="AA_API_KEY",
        help="Environment variable name that stores the AA API key (default: AA_API_KEY)",
    )
    ap.add_argument(
        "--max_age_hours",
        type=float,
        default=24.0,
        help="Reuse cached JSON when newer than this many hours (default: 24)",
    )
    ap.add_argument(
        "--force_refresh",
        action="store_true",
        help="Force a network refresh even if cache is fresh (still only one request).",
    )
    ap.add_argument("--timeout_s", type=int, default=30, help="HTTP timeout in seconds (default: 30)")
    args = ap.parse_args()

    _safe_mkdir(args.out_dir)

    cache_json = os.path.join(args.out_dir, "llms_models.json")
    out_csv = os.path.join(args.out_dir, cfg.out_csv_name)
    out_info = os.path.join(args.out_dir, cfg.out_info_name)

    cache_age_h = _hours_since_mtime(cache_json)
    cache_fresh = cache_age_h is not None and cache_age_h <= float(args.max_age_hours)

    source = "cache"
    payload: Dict[str, Any]
    if (not args.force_refresh) and cache_fresh:
        payload = _load_json(cache_json)
    else:
        api_key = os.environ.get(str(args.api_key_env))
        if not api_key:
            if os.path.exists(cache_json):
                payload = _load_json(cache_json)
                source = "stale_cache"
            else:
                raise SystemExit(
                    f"Missing {args.api_key_env} and no cache available at {cache_json}. "
                    f"Set {args.api_key_env} in your environment or provide a cache file."
                )
        else:
            payload = _fetch_models_json(str(args.url), str(api_key), timeout_s=int(args.timeout_s))
            _write_json(cache_json, payload)
            source = "api"
            cache_age_h = _hours_since_mtime(cache_json)

    rows = _extract_rows(payload, cfg)
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=_fieldnames(cfg))
        w.writeheader()
        w.writerows(rows)

    info = {
        "url": str(args.url),
        "source": source,
        "cache_json": cache_json,
        "cache_age_hours": None if cache_age_h is None else round(float(cache_age_h), 3),
        "downloaded_at_utc": _dt.datetime.now(tz=_dt.timezone.utc).isoformat(),
        "n_rows": len(rows),
        "api_key_env": str(args.api_key_env),
    }
    _write_json(out_info, info)

    print(f"Artificial Analysis {cfg.label} export complete.")
    print(f"  - csv:   {out_csv}")
    print(f"  - info:  {out_info}")
    print(f"  - cache: {cache_json} (source={source})")
    print(f"Rows: {len(rows)}")


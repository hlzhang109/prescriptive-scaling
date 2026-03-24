#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import time
from typing import Dict, Optional
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import pandas as pd


def _load_top_models_last_modified(top_models_csv: str) -> Dict[str, str]:
    if not top_models_csv or not os.path.isfile(top_models_csv):
        return {}
    df = pd.read_csv(top_models_csv, usecols=["model_id", "last_modified"])
    df["model_id"] = df["model_id"].astype(str).str.strip()
    df["last_modified"] = df["last_modified"].astype(str).str.strip()
    df = df[(df["model_id"] != "") & (df["model_id"].str.lower() != "nan")].copy()
    df = df[(df["last_modified"] != "") & (df["last_modified"].str.lower() != "nan")].copy()
    if df.empty:
        return {}
    df["_dt"] = pd.to_datetime(df["last_modified"], errors="coerce", utc=True, format="mixed")
    df = df.dropna(subset=["_dt"]).copy()
    if df.empty:
        return {}
    df = df.sort_values("_dt").groupby("model_id", as_index=False).tail(1)
    return {str(mid): str(dt.isoformat()) for mid, dt in zip(df["model_id"], df["_dt"])}


def _read_cache(cache_json: Optional[str]) -> Dict[str, str]:
    if not cache_json or not os.path.isfile(cache_json):
        return {}
    try:
        with open(cache_json, "r") as f:
            data = json.load(f)
        if isinstance(data, dict):
            return {str(k): str(v) for k, v in data.items() if v is not None}
    except Exception:
        pass
    return {}


def _write_cache(cache_json: Optional[str], cache: Dict[str, str]) -> None:
    if not cache_json:
        return
    os.makedirs(os.path.dirname(os.path.abspath(cache_json)), exist_ok=True)
    with open(cache_json, "w") as f:
        json.dump(cache, f, indent=2, sort_keys=True)


def _fetch_hf_last_modified(model_id: str, *, timeout_s: float) -> Optional[str]:
    model_id = str(model_id).strip()
    if not model_id:
        return None
    url = f"https://huggingface.co/api/models/{model_id}"
    req = Request(url, headers={"User-Agent": "skill-frontier/enrich_new_leaderboard_results_with_last_modified"})
    try:
        with urlopen(req, timeout=float(timeout_s)) as resp:  # nosec - URL is fixed to HF API
            data = json.loads(resp.read().decode("utf-8"))
    except (HTTPError, URLError, TimeoutError, ValueError):
        return None
    if not isinstance(data, dict):
        return None
    lm = data.get("lastModified")
    if not lm:
        return None
    dt = pd.to_datetime(str(lm), errors="coerce", utc=True, format="mixed")
    if pd.isna(dt):
        return None
    return str(dt.isoformat())


def main() -> None:
    ap = argparse.ArgumentParser(description="Enrich new_leaderboard_results_with_tokens.csv with HF lastModified timestamps.")
    ap.add_argument(
        "--in_csv",
        default=os.path.join("tables", "new_leaderboard_results_with_tokens.csv"),
        help="Input CSV (expects 'model_id').",
    )
    ap.add_argument(
        "--out_csv",
        default=None,
        help="Output CSV path (default: overwrite --in_csv).",
    )
    ap.add_argument(
        "--top_models_csv",
        default=os.path.join("tables", "top_models_by_base.csv"),
        help="Optional CSV containing last_modified for some model_ids.",
    )
    ap.add_argument(
        "--cache_json",
        default=os.path.join("tables", "_cache_hf_lastModified_new_leaderboard.json"),
        help="Cache file to avoid re-querying HF API across runs.",
    )
    ap.add_argument("--timeout_s", type=float, default=20.0, help="Timeout (seconds) per HF API request.")
    ap.add_argument("--sleep_s", type=float, default=0.05, help="Sleep between HF API requests.")
    ap.add_argument("--max_retries", type=int, default=3, help="Retries per model_id when the HF API errors.")
    args = ap.parse_args()

    in_csv = str(args.in_csv)
    out_csv = str(args.out_csv) if args.out_csv else in_csv
    df = pd.read_csv(in_csv)
    if "model_id" not in df.columns:
        raise SystemExit(f"Input CSV missing required column: model_id ({in_csv})")

    model_ids = df["model_id"].astype(str).str.strip()
    unique_ids = [m for m in model_ids.dropna().unique().tolist() if m and m.lower() != "nan"]

    from_top = _load_top_models_last_modified(str(args.top_models_csv))
    cache = _read_cache(str(args.cache_json) if args.cache_json else None)

    resolved: Dict[str, str] = {}
    n_top = 0
    n_cache = 0
    n_fetch = 0
    n_fail = 0

    for mid in unique_ids:
        if mid in resolved:
            continue
        if mid in from_top:
            resolved[mid] = from_top[mid]
            n_top += 1
            continue
        if mid in cache:
            resolved[mid] = cache[mid]
            n_cache += 1
            continue
        last = None
        for _ in range(max(1, int(args.max_retries))):
            last = _fetch_hf_last_modified(mid, timeout_s=float(args.timeout_s))
            if last:
                break
            time.sleep(0.2)
        if last:
            resolved[mid] = last
            cache[mid] = last
            n_fetch += 1
        else:
            n_fail += 1
        if float(args.sleep_s) > 0:
            time.sleep(float(args.sleep_s))

    _write_cache(str(args.cache_json) if args.cache_json else None, cache)

    df["lastModified"] = model_ids.map(resolved)
    df.to_csv(out_csv, index=False)

    print(
        f"[enrich_lastModified] wrote={out_csv} rows={len(df)} unique_model_ids={len(unique_ids)} "
        f"resolved_top={n_top} resolved_cache={n_cache} fetched={n_fetch} failed={n_fail}"
    )


if __name__ == "__main__":
    main()


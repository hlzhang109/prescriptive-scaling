#!/usr/bin/env python3
"""
Download the full *new* Open LLM Leaderboard table from Hugging Face.

Why this exists
---------------
`tables/open_llm_leaderboard/open_llm_leaderboard.csv` is easy to end up with a
partial export. The authoritative "contents" dataset on Hugging Face provides a
complete table with the expected schema (IFEval/BBH/MATH/GPQA/MUSR/MMLU-PRO).

Important: BBH subtasks
-----------------------
The "contents" dataset does *not* include BBH subtask columns (e.g.
`leaderboard_bbh_boolean_expressions`). Those live in the public dataset
`open-llm-leaderboard/results` as per-model `results_*.json` files.

If `--bbh-subtasks` is enabled (default), this script will:
  1) download `open-llm-leaderboard/contents`
  2) join in the 24 BBH subtask scores by matching:
       - `fullname` -> results JSON subfolder path
       - `Precision` -> `config.model_dtype` in results JSON (best-effort)
     and selecting the candidate results file whose mean BBH-subtask score best
     matches the table's `BBH Raw` value (robust to occasional upstream SHA
     inconsistencies).

This script downloads the parquet from:
  - dataset repo: open-llm-leaderboard/contents
  - file: data/train-00000-of-00001.parquet

and writes a CSV locally after validating the base column names.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import shutil
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

import pandas as pd
from huggingface_hub import HfApi, hf_hub_download


EXPECTED_COLUMNS: list[str] = [
    "eval_name",
    "Precision",
    "Type",
    "T",
    "Weight type",
    "Architecture",
    "Model",
    "fullname",
    "Model sha",
    "Average ⬆️",
    "Hub License",
    "Hub ❤️",
    "#Params (B)",
    "Available on the hub",
    "MoE",
    "Flagged",
    "Chat Template",
    "CO₂ cost (kg)",
    "IFEval Raw",
    "IFEval",
    "BBH Raw",
    "BBH",
    "MATH Lvl 5 Raw",
    "MATH Lvl 5",
    "GPQA Raw",
    "GPQA",
    "MUSR Raw",
    "MUSR",
    "MMLU-PRO Raw",
    "MMLU-PRO",
    "Merged",
    "Official Providers",
    "Upload To Hub Date",
    "Submission Date",
    "Generation",
    "Base Model",
]

BBH_SUBTASK_COLUMNS: list[str] = [
    "leaderboard_bbh_boolean_expressions",
    "leaderboard_bbh_causal_judgement",
    "leaderboard_bbh_date_understanding",
    "leaderboard_bbh_disambiguation_qa",
    "leaderboard_bbh_formal_fallacies",
    "leaderboard_bbh_geometric_shapes",
    "leaderboard_bbh_hyperbaton",
    "leaderboard_bbh_logical_deduction_five_objects",
    "leaderboard_bbh_logical_deduction_seven_objects",
    "leaderboard_bbh_logical_deduction_three_objects",
    "leaderboard_bbh_movie_recommendation",
    "leaderboard_bbh_navigate",
    "leaderboard_bbh_object_counting",
    "leaderboard_bbh_penguins_in_a_table",
    "leaderboard_bbh_reasoning_about_colored_objects",
    "leaderboard_bbh_ruin_names",
    "leaderboard_bbh_salient_translation_error_detection",
    "leaderboard_bbh_snarks",
    "leaderboard_bbh_sports_understanding",
    "leaderboard_bbh_temporal_sequences",
    "leaderboard_bbh_tracking_shuffled_objects_five_objects",
    "leaderboard_bbh_tracking_shuffled_objects_seven_objects",
    "leaderboard_bbh_tracking_shuffled_objects_three_objects",
    "leaderboard_bbh_web_of_lies",
]


def _backup_existing(path: Path) -> Optional[Path]:
    if not path.exists():
        return None
    ts = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    backup = path.with_name(f"{path.stem}.bak_{ts}{path.suffix}")
    shutil.copy2(path, backup)
    return backup


def _precision_matches(row_precision: str, model_dtype: str) -> bool:
    rp = (row_precision or "").strip().lower()
    md = (model_dtype or "").strip().lower()
    if not rp:
        return True
    # Quantized runs may still report float/bfloat in config; don't hard-fail.
    if "4bit" in rp or "8bit" in rp or "int" in rp:
        return True
    if "bfloat16" in rp or rp in ("bf16",):
        return ("bfloat16" in md) or ("bf16" in md)
    if "float16" in rp or rp in ("fp16",):
        return ("float16" in md) or ("fp16" in md)
    return True


def _index_results_files(api: HfApi, repo_id: str, revision: Optional[str]) -> dict[str, list[str]]:
    files = api.list_repo_files(repo_id=repo_id, repo_type="dataset", revision=revision)
    results_files = [f for f in files if f.endswith(".json") and "/results_" in f]
    by_fullname: dict[str, list[str]] = defaultdict(list)
    for f in results_files:
        # Example: "01-ai/Yi-1.5-34B/results_2024-06-19T03-56-15.142347.json"
        dirname = f.rsplit("/results_", 1)[0]
        by_fullname[dirname].append(f)
    for k in by_fullname:
        # Filenames start with ISO-ish timestamps; lexicographic sort == chronological.
        by_fullname[k].sort(reverse=True)
    return dict(by_fullname)


def _fetch_bbh_subtasks(
    *,
    fullname: str,
    model_sha: str,
    precision: str,
    target_bbh_raw: float,
    files_by_fullname: dict[str, list[str]],
    repo_id: str,
    revision: Optional[str],
    metric_key: str,
) -> Optional[dict[str, float]]:
    candidates = files_by_fullname.get(fullname)
    if not candidates:
        return None

    want_sha = (model_sha or "").strip()
    try:
        target = float(target_bbh_raw)
    except Exception:
        target = float("nan")

    best: Optional[dict[str, float]] = None
    best_diff = float("inf")
    best_sha_match = False

    for fname in candidates:
        try:
            local = hf_hub_download(repo_id=repo_id, repo_type="dataset", filename=fname, revision=revision)
        except Exception:
            continue
        try:
            with open(local, "r") as f:
                data = json.load(f)
        except Exception:
            continue
        cfg = data.get("config", {}) if isinstance(data, dict) else {}
        got_sha = str(cfg.get("model_sha", "")).strip()
        if not _precision_matches(precision, str(cfg.get("model_dtype", ""))):
            continue

        res = data.get("results", {}) if isinstance(data, dict) else {}
        out: dict[str, float] = {}
        vals: list[float] = []
        for col in BBH_SUBTASK_COLUMNS:
            v = None
            if isinstance(res, dict) and col in res and isinstance(res[col], dict):
                v = res[col].get(metric_key)
                if v is None and metric_key != "acc_norm,none":
                    v = res[col].get("acc_norm,none")
            try:
                fv = float(v) if v is not None else float("nan")
            except Exception:
                fv = float("nan")
            out[col] = fv
            if fv == fv:  # not NaN
                vals.append(fv)

        # Prefer the candidate whose mean BBH subtask score matches the table's BBH Raw.
        if vals and target == target:
            mean_val = sum(vals) / float(len(vals))
            diff = abs(mean_val - target)
        else:
            diff = float("inf")

        sha_match = bool(want_sha and got_sha and got_sha == want_sha)
        if (diff < best_diff) or (diff == best_diff and sha_match and not best_sha_match):
            best = out
            best_diff = diff
            best_sha_match = sha_match
            # Fast path: perfect (or near-perfect) match.
            if diff <= 1e-6:
                break

    # Fallback: if we couldn't compute a diff (e.g., missing BBH Raw), try strict sha match.
    if best is None and want_sha:
        for fname in candidates:
            try:
                local = hf_hub_download(repo_id=repo_id, repo_type="dataset", filename=fname, revision=revision)
            except Exception:
                continue
            try:
                with open(local, "r") as f:
                    data = json.load(f)
            except Exception:
                continue
            cfg = data.get("config", {}) if isinstance(data, dict) else {}
            got_sha = str(cfg.get("model_sha", "")).strip()
            if got_sha != want_sha:
                continue
            if not _precision_matches(precision, str(cfg.get("model_dtype", ""))):
                continue
            res = data.get("results", {}) if isinstance(data, dict) else {}
            out: dict[str, float] = {}
            for col in BBH_SUBTASK_COLUMNS:
                v = None
                if isinstance(res, dict) and col in res and isinstance(res[col], dict):
                    v = res[col].get(metric_key)
                    if v is None and metric_key != "acc_norm,none":
                        v = res[col].get("acc_norm,none")
                try:
                    out[col] = float(v) if v is not None else float("nan")
                except Exception:
                    out[col] = float("nan")
            return out

    return best


def main(argv: Optional[list[str]] = None) -> None:
    ap = argparse.ArgumentParser(description="Download full Open LLM Leaderboard 'contents' table and write CSV.")
    ap.add_argument("--repo_id", default="open-llm-leaderboard/contents", help="HF dataset repo id.")
    ap.add_argument("--repo_type", default="dataset", help="HF repo type (must be 'dataset').")
    ap.add_argument(
        "--parquet_file",
        default="data/train-00000-of-00001.parquet",
        help="Parquet filename within the dataset repo.",
    )
    ap.add_argument(
        "--revision",
        default=None,
        help="Optional git revision/commit SHA to pin to (default: latest).",
    )
    ap.add_argument(
        "--bbh-subtasks",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="If set, augment the downloaded table with 24 BBH subtask columns (default: enabled).",
    )
    ap.add_argument(
        "--bbh_results_repo_id",
        default="open-llm-leaderboard/results",
        help="HF dataset repo id containing per-model results JSONs.",
    )
    ap.add_argument(
        "--bbh_results_revision",
        default=None,
        help="Optional git revision/commit SHA to pin BBH results to (default: latest).",
    )
    ap.add_argument(
        "--bbh_metric_key",
        default="acc_norm,none",
        help="Metric key to extract for BBH subtasks inside results JSON (default: 'acc_norm,none').",
    )
    ap.add_argument(
        "--bbh_workers",
        type=int,
        default=8,
        help="Thread pool size for BBH subtask downloads (default: 8; set <=1 for sequential).",
    )
    ap.add_argument(
        "--output_csv",
        default="tables/open_llm_leaderboard/open_llm_leaderboard.csv",
        help="Where to write the downloaded leaderboard CSV.",
    )
    ap.add_argument(
        "--backup",
        action="store_true",
        help="If set, create a timestamped backup of the existing output CSV before overwriting.",
    )
    args = ap.parse_args(argv)

    if args.repo_type != "dataset":
        raise SystemExit("--repo_type must be 'dataset'")

    out_path = Path(args.output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if args.backup:
        backup = _backup_existing(out_path)
        if backup:
            print(f"[download] backed up existing CSV -> {backup}")

    parquet_path = hf_hub_download(
        repo_id=args.repo_id,
        repo_type=args.repo_type,
        filename=args.parquet_file,
        revision=args.revision,
    )

    df = pd.read_parquet(parquet_path)

    missing = [c for c in EXPECTED_COLUMNS if c not in df.columns]
    extra = [c for c in df.columns if c not in EXPECTED_COLUMNS]
    if missing or extra:
        raise SystemExit(
            "Downloaded table schema does not match expected leaderboard columns.\n"
            f"Missing columns ({len(missing)}): {missing}\n"
            f"Extra columns ({len(extra)}): {extra}\n"
            f"Downloaded columns: {list(df.columns)}"
        )

    df = df[EXPECTED_COLUMNS]

    if args.bbh_subtasks:
        print("[download] indexing BBH results files...")
        api = HfApi()
        files_by_fullname = _index_results_files(api, repo_id=args.bbh_results_repo_id, revision=args.bbh_results_revision)
        for c in BBH_SUBTASK_COLUMNS:
            df[c] = float("nan")

        fullnames = df["fullname"].fillna("").astype(str).tolist()
        model_shas = df["Model sha"].fillna("").astype(str).tolist()
        precisions = df["Precision"].fillna("").astype(str).tolist()

        total = len(df)
        found = 0
        missing_ct = 0

        def _job(i: int) -> tuple[int, Optional[dict[str, float]]]:
            out = _fetch_bbh_subtasks(
                fullname=fullnames[i],
                model_sha=model_shas[i],
                precision=precisions[i],
                target_bbh_raw=df.at[i, "BBH Raw"],
                files_by_fullname=files_by_fullname,
                repo_id=args.bbh_results_repo_id,
                revision=args.bbh_results_revision,
                metric_key=str(args.bbh_metric_key),
            )
            return i, out

        if int(args.bbh_workers) > 1:
            with ThreadPoolExecutor(max_workers=int(args.bbh_workers)) as ex:
                futures = [ex.submit(_job, i) for i in range(total)]
                for j, fut in enumerate(as_completed(futures), start=1):
                    try:
                        i, out = fut.result()
                    except Exception:
                        missing_ct += 1
                        continue
                    if out is None:
                        missing_ct += 1
                    else:
                        found += 1
                        for k, v in out.items():
                            df.at[i, k] = v
                    if j % 200 == 0 or j == total:
                        print(f"[download] BBH subtasks: {j}/{total} rows processed (matched={found}, missing={missing_ct})")
        else:
            for i in range(total):
                _, out = _job(i)
                if out is None:
                    missing_ct += 1
                else:
                    found += 1
                    for k, v in out.items():
                        df.at[i, k] = v
                if (i + 1) % 200 == 0 or (i + 1) == total:
                    print(f"[download] BBH subtasks: {i+1}/{total} rows processed (matched={found}, missing={missing_ct})")

        print(f"[download] BBH subtasks complete: matched={found}/{total}, missing={missing_ct}")

        # Sanity check: mean(BBH subtasks) should agree with "BBH Raw" when enough subtasks are present.
        try:
            bbh_raw = pd.to_numeric(df["BBH Raw"], errors="coerce")
            sub_mean = df[BBH_SUBTASK_COLUMNS].mean(axis=1, skipna=True)
            ok = sub_mean.notna() & bbh_raw.notna()
            if int(ok.sum()) > 0:
                max_abs = float((sub_mean[ok] - bbh_raw[ok]).abs().max())
                if max_abs > 1e-6:
                    print(f"[WARN] max |mean(subtasks)-BBH Raw| = {max_abs:.6g} (expected ~0)")
        except Exception:
            pass

    df.to_csv(out_path, index=False)

    info = HfApi().dataset_info(repo_id=args.repo_id, revision=args.revision)
    rev = getattr(info, "sha", None) or args.revision or "unknown"
    print(f"[download] source: {args.repo_id}@{rev}")
    print(f"[download] wrote {len(df)} rows × {len(df.columns)} cols -> {out_path}")


if __name__ == "__main__":
    main()

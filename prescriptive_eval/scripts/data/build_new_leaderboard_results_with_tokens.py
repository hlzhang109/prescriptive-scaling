#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import re
from typing import Dict, Tuple

import numpy as np
import pandas as pd


def _infer_params_b(base_model: str) -> float:
    m = re.findall(r"([0-9]+(?:\.[0-9]+)?)\s*[Bb](?![A-Za-z])", base_model)
    if not m:
        return float("nan")
    return float(m[-1])


def _infer_tokens_t(base_model: str, *, fixed_tokens_t: Dict[str, float]) -> float:
    if base_model in fixed_tokens_t:
        return float(fixed_tokens_t[base_model])

    evolm = re.match(r"^zhenting/evolm-(?P<params>[0-9.]+)B-(?P<tokens>[0-9]+)BT$", base_model)
    if evolm:
        tokens_b = float(evolm.group("tokens"))
        return tokens_b / 1000.0

    return float("nan")


def _build_fixed_tokens_t() -> Dict[str, float]:
    # See docs/web_search/new_leaderboard_results_with_tokens.md for sources.
    fixed: Dict[str, float] = {
        # Qwen
        "Qwen/Qwen2.5-1.5B": 18.0,
        "Qwen/Qwen2.5-7B": 18.0,
        "Qwen/Qwen2.5-32B": 18.0,
        "Qwen/Qwen3-8B-Base": 36.0,
        "Qwen/Qwen3-14B-Base": 36.0,
        # ByteDance Seed
        "ByteDance-Seed/Seed-OSS-36B-Base": 12.0,
        # NVIDIA Nemotron Nano v2
        "nvidia/NVIDIA-Nemotron-Nano-9B-v2-Base": 20.0,
        "nvidia/NVIDIA-Nemotron-Nano-12B-v2-Base": 20.0,
        # AllenAI OLMo (token counts differ by checkpoint)
        "allenai/OLMo-1B": 3.0,
        "allenai/OLMo-1B-0724-hf": 3.05,
        "allenai/OLMo-7B": 2.5,
        "allenai/OLMo-7B-0424-hf": 2.05,
        "allenai/OLMo-7B-0724-hf": 2.75,
        "allenai/OLMo-2-0425-1B": 4.0,
        "allenai/OLMo-2-1124-7B": 4.0,
        "allenai/OLMo-2-1124-13B": 5.0,
        "allenai/OLMo-2-0325-32B": 6.0,
        "allenai/Olmo-3-1025-7B": 5.93,
        "allenai/Olmo-3-1125-32B": 5.50,
        # AllenAI digital-socrates is a fine-tune of Llama 2 Chat (use base-model pretrain tokens)
        "allenai/digital-socrates-7b": 2.0,
        "allenai/digital-socrates-13b": 2.0,
        # Meta Llama
        "meta-llama/Llama-2-7b-hf": 2.0,
        "meta-llama/Llama-2-13b-hf": 2.0,
        "meta-llama/Llama-2-70b-hf": 2.0,
        # Use Llama-3-scale token count as proxy for 3.1 base models.
        "meta-llama/Llama-3.1-8B": 15.0,
        "meta-llama/Llama-3.1-70B": 15.0,
        # Not an LLM / no OLL scores in the input table; keep missing.
        "allenai/OlmoEarth-v1-Base": float("nan"),
    }
    return fixed


def main() -> None:
    ap = argparse.ArgumentParser(description="Add tokens/params columns to new_leaderboard_results.csv")
    ap.add_argument(
        "--in_csv",
        default="tables/new_leaderboard_results.csv",
        help="Input CSV (expects 'mapped_base_model').",
    )
    ap.add_argument(
        "--out_csv",
        default="tables/new_leaderboard_results_with_tokens.csv",
        help="Output CSV with added columns.",
    )
    args = ap.parse_args()

    df = pd.read_csv(args.in_csv)
    if "mapped_base_model" not in df.columns:
        raise SystemExit("Input CSV missing required column: mapped_base_model")

    fixed_tokens_t = _build_fixed_tokens_t()

    tokens_t = []
    params_b = []
    for base in df["mapped_base_model"].astype(str).tolist():
        tok = _infer_tokens_t(base, fixed_tokens_t=fixed_tokens_t)
        prm = _infer_params_b(base)
        tokens_t.append(tok)
        params_b.append(prm)

    df["Pretraining tokens (T)"] = pd.to_numeric(tokens_t, errors="coerce")
    df["#Params (B)"] = pd.to_numeric(params_b, errors="coerce")

    # Sanity check coverage (allow NaNs only where explicitly intended).
    by_base = df[["mapped_base_model", "Pretraining tokens (T)", "#Params (B)"]].drop_duplicates()
    missing = by_base[
        (~np.isfinite(by_base["Pretraining tokens (T)"])) | (~np.isfinite(by_base["#Params (B)"]))
    ]["mapped_base_model"].tolist()

    allowed_missing = {"allenai/OlmoEarth-v1-Base"}
    unexpected = sorted(set(missing) - allowed_missing)
    if unexpected:
        raise SystemExit(
            "Missing tokens/params for base models:\n  - " + "\n  - ".join(unexpected)
        )

    df.to_csv(args.out_csv, index=False)


if __name__ == "__main__":
    main()

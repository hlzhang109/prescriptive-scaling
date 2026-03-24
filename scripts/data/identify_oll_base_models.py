#!/usr/bin/env python3
"""
Identify base models + pretraining token counts for Open LLM Leaderboard exports.

This script is a standalone, reproducible replacement for `tables/find_token_size.ipynb`.
It supports both:
  - the *new* OLL export (has a `Base Model` column; tasks like `IFEval Raw`, `BBH Raw`, ...)
  - the *v1/old* OLL export (no `Base Model` column; tasks like `ARC`, `HellaSwag`, ...)

It reads a leaderboard CSV (default: new) and writes:
  - `tables/open_llm_leaderboard/open_llm_leaderboard_with_tokens.csv`
  - a separate CSV of unassigned rows (when we are not confident)

It adds/overwrites three columns:
  - `Pretraining tokens (T)` (float, in trillions)
  - `Base model family` (string)
  - `Identified base model` (string)

Design principles:
  - Prefer the *official* `Base Model` column (which is often a fine-tuned/merge model);
    we infer the *base model of that entry* via string/architecture/params heuristics.
  - Enforce strong constraints to avoid systematic mislabels:
      * If `Architecture` contains "Qwen2", the model must be in Qwen2/Qwen2.5.
      * If the name/base contains "Falcon3", do NOT infer Llama just because the
        architecture is `LlamaForCausalLM`.
    For the v1/old export, the `Qwen2ForCausalLM` class is also used for many Qwen1.5
    fine-tunes; we relax the Qwen constraint in that case to avoid misclassification.
  - If unsure: leave blank and export to the unassigned CSV for manual review.

Token-count sources (accessed 2025-12-25):
  - Qwen2: "over 7 trillion tokens" (Qwen2 Technical Report, arXiv:2407.10671)
    https://arxiv.org/pdf/2407.10671.pdf
  - Qwen2.5: "up to 18 trillion tokens" (Qwen blog)
    https://qwenlm.github.io/blog/qwen2.5/
  - Llama 3: "over 15T tokens" (Meta blog)
    https://ai.meta.com/blog/meta-llama-3/
  - Llama 3.1: "over 15 trillion tokens" (Meta blog)
    https://ai.meta.com/blog/meta-llama-3-1/
  - Llama 2: "trained on 2 trillion tokens" (Llama 2 paper, arXiv:2307.09288)
    https://arxiv.org/pdf/2307.09288.pdf
  - Gemma 2: "27B on 13T, 9B on 8T, 2B on 2T tokens" (Gemma 2 paper, arXiv:2408.00118)
    https://arxiv.org/pdf/2408.00118.pdf
  - Phi-3: "trained on 3.3 trillion tokens" (Phi-3 Technical Report, arXiv:2404.14219)
    https://arxiv.org/pdf/2404.14219.pdf
  - Phi-4: "pretrained for approximately 10T tokens" (Phi-4 Technical Report, arXiv:2412.08905)
    https://arxiv.org/pdf/2412.08905.pdf
  - Falcon3: base model list + training-token notes (Hugging Face blog)
    https://huggingface.co/blog/falcon3

Notes:
  - For Falcon3 token counts, we use the explicit values in the blog:
      * Transformer Falcon3-7B: 14T tokens
      * Falcon3-10B: 14T + 2T continuation => 16T (approximate total)
      * Falcon3-1B/3B: "<100GT" => 0.1T (upper bound used as proxy)
      * Falcon3-Mamba-7B: blog reports +1.5T additional tokens; total is unclear, so we
        conservatively set 1.5T to avoid overstating compute.
    If you prefer different conventions, adjust `BASE_MODEL_TOKENS`.
"""

from __future__ import annotations

import argparse
import re
from typing import Optional, Tuple

import numpy as np
import pandas as pd


BASE_MODEL_TOKENS: dict[str, float] = {
    # Llama
    "llama-3": 15.0,
    "llama-3.1": 15.0,
    "llama-3.2": 15.0,
    "llama-2": 2.0,
    "llama-1": 1.4,
    # Mistral / Mixtral
    "mistral-7b": 0.8,
    "mixtral": 1.0,
    # Gemma (size-specific in the Gemma 2 paper)
    "gemma-2-2b": 2.0,
    "gemma-2-9b": 8.0,
    "gemma-2-27b": 13.0,
    # Gemma 3 (token counts inferred from `tables/open_llm_leaderboard/new_eval_leaderboard.csv`)
    "gemma-3-270m": 6.0,
    "gemma-3-1b": 2.0,
    "gemma-3-4b": 4.0,
    "gemma-3-12b": 12.0,
    "gemma-3-27b": 14.0,
    "gemma-1": 2.0,
    # Qwen
    "qwen2.5": 18.0,
    "qwen2": 7.0,
    # Qwen3 (token counts inferred from `tables/open_llm_leaderboard/new_eval_leaderboard.csv`)
    "qwen3": 36.0,
    "qwen1": 2.0,
    # Yi
    "yi-1.5": 3.0,
    "yi-1": 3.0,
    # Phi
    "phi-4": 10.0,
    "phi-3": 3.3,
    "phi-2": 1.4,
    "phi-1.5": 0.03,
    "phi-1": 0.006,
    # Falcon (older Falcon family)
    "falcon-180b": 3.5,
    "falcon-40b": 1.0,
    "falcon-7b": 1.5,
    # Falcon3 (new)
    "falcon3-10b": 16.0,
    "falcon3-7b": 14.0,
    "falcon3-3b": 0.1,
    "falcon3-1b": 0.1,
    "falcon3-mamba-7b": 1.5,
    # Other (kept from the notebook for backwards compatibility)
    # SmolLM3 (token counts inferred from `tables/open_llm_leaderboard/new_eval_leaderboard.csv`)
    "smollm3-3b": 11.2,
    # OpenAI GPT-OSS (token counts inferred from `tables/open_llm_leaderboard/new_eval_leaderboard.csv`)
    "gpt-oss-20b": 4.167,
    "glm3": 3.9,
    "glm2": 1.4,
    "deepseek2": 8.0,
    "deepseek": 2.0,
    "mpt-30b": 1.5,
    "mpt-7b": 1.0,
    "stablelm": 1.5,
    "bloom": 0.366,
    "baichuan-3": 3.2,
    "baichuan-2": 2.6,
}


# Relaxed parameter ranges copied from `tables/find_token_size.ipynb`.
PARAM_SIZE_RANGES: list[tuple[float, float, str]] = [
    # Llama-2
    (6.7, 7.5, "llama-2-7b"),
    (12.5, 13.5, "llama-2-13b"),
    (64.0, 71.0, "llama-2-70b"),
    # Llama-3
    (7.8, 8.3, "llama-3-8b"),
    (69.0, 72.0, "llama-3-70b"),
    # Mistral
    (7.0, 7.5, "mistral-7b"),
    # Mixtral
    (45.0, 48.0, "mixtral-8x7b"),
    # Gemma-2
    (1.8, 2.2, "gemma-2-2b"),
    (8.9, 9.5, "gemma-2-9b"),
    (26.5, 29.0, "gemma-2-27b"),
    # Qwen2 (note: Qwen2.5 shares the Qwen2 architecture; series disambiguation is done elsewhere)
    (0.45, 0.55, "qwen2-0.5b"),
    (1.4, 1.7, "qwen2-1.5b"),
    (7.4, 7.8, "qwen2-7b"),
    (14.0, 15.0, "qwen2-14b"),
    (72.0, 73.5, "qwen2-72b"),
    # Phi-3
    (3.7, 4.0, "phi-3-4b"),
    (6.8, 7.2, "phi-3-7b"),
    (13.0, 15.0, "phi-3-14b"),
]


def _norm(s: object) -> str:
    return str(s or "").strip()


def _norm_lc(s: object) -> str:
    return _norm(s).lower()


def _strip_trailing_parens(s: str) -> str:
    # Remove a single trailing "(...)" like "(Merge)" while keeping internal parentheses.
    return re.sub(r"\s*\([^)]*\)\s*$", "", s).strip()


def extract_size_from_name(name: str) -> Optional[str]:
    """Extract a size token like '7b', '0.5b', '72b' from a name."""
    size_patterns = [
        r"[\-_](\d+\.?\d*)b(?!\w)",
        r"(\d+\.?\d*)b[\-_]",
        r"[\-_](\d+\.?\d*)B(?!\w)",
        r"(\d+\.?\d*)B[\-_]",
        r"/(\d+\.?\d*)[bB](?![a-zA-Z0-9])",
    ]
    for pattern in size_patterns:
        m = re.search(pattern, name)
        if m:
            return f"{m.group(1)}b"
    return None


def match_param_size_to_model(param_size: float) -> Optional[str]:
    if not np.isfinite(param_size):
        return None
    for lo, hi, model_id in PARAM_SIZE_RANGES:
        if lo <= param_size <= hi:
            return model_id
    return None


def _qwen_series_from_context(name_lc: str, param_size: float, *, allow_qwen1: bool = False) -> str:
    # In the v1 ("old") OLL export, many Qwen1.5 models are labeled with the
    # `Qwen2ForCausalLM` architecture class. Treat explicit Qwen1.5 strings as
    # Qwen1 family to avoid misclassifying them as Qwen2/Qwen2.5.
    if allow_qwen1 and ("qwen1.5" in name_lc or "qwen1_5" in name_lc or "qwen1-5" in name_lc):
        return "qwen1"

    # Qwen3: Hugging Face may still tag these as `Qwen2ForCausalLM`; prefer explicit strings.
    #
    # IMPORTANT: avoid misclassifying Qwen2.5 3B fine-tunes named like "...Qwen-3b...".
    # Require a delimiter after "3" (e.g., Qwen3-4B, Qwen-3-4B) or end-of-string.
    if re.search(r"qwen[\-_]?3(?:[\-_]|$)", name_lc):
        return "qwen3"

    # Qwen2.5 sizes not present in Qwen2: 3B, 14B, 32B. Use those as a strong signal.
    if "qwen2.5" in name_lc or "qwen2_5" in name_lc or "v2.5" in name_lc:
        return "qwen2.5"
    if 2.8 <= param_size <= 3.8 or 13.0 <= param_size <= 16.0 or 29.0 <= param_size <= 36.0:
        return "qwen2.5"
    # Qwen2 (not 2.5) has a 57B base model.
    if 54.0 <= param_size <= 60.0:
        return "qwen2"
    if "qwen2" in name_lc:
        return "qwen2"
    if allow_qwen1 and (re.search(r"qwen[\-_]?1(?!\.)", name_lc) or "qwen1.5" in name_lc):
        return "qwen1"
    return "qwen2"


def _identify_falcon3(name_lc: str, param_size: float) -> Tuple[Optional[str], Optional[str]]:
    if "falcon3" not in name_lc and "falconthink3" not in name_lc:
        return None, None

    if "mamba" in name_lc:
        return "falcon3-mamba-7b", "falcon3-mamba-7b"

    size = extract_size_from_name(name_lc)
    if size:
        return f"falcon3-{size}", f"falcon3-{size}"

    # Param-size fallback for common Falcon3 sizes.
    if 9.5 <= param_size <= 11.5:
        return "falcon3-10b", "falcon3-10b"
    if 6.8 <= param_size <= 8.2:
        return "falcon3-7b", "falcon3-7b"
    if 2.8 <= param_size <= 3.6:
        return "falcon3-3b", "falcon3-3b"
    if 1.2 <= param_size <= 2.2:
        return "falcon3-1b", "falcon3-1b"

    return None, None


def _identify_phi4(name_lc: str, param_size: float) -> Tuple[Optional[str], Optional[str]]:
    if "phi-4" not in name_lc and "phi4" not in name_lc:
        return None, None

    # If the (official) base-model entry explicitly says microsoft/phi-4, treat it as Phi-4
    # even if the leaderboard's parameter count is distorted by quantization artifacts.
    if "microsoft/phi-4" in name_lc:
        if "mini" in name_lc:
            return "phi-4", "phi-4-mini-4b"
        return "phi-4", "phi-4-14b"

    size = extract_size_from_name(name_lc)
    if size == "14b":
        return "phi-4", "phi-4-14b"
    if size in {"4b", "3.8b"} and "mini" in name_lc:
        return "phi-4", "phi-4-mini-4b"

    # Phi-4-mini (≈3.8B) appears in the leaderboard; keep a stable identifier.
    if "mini" in name_lc and 3.3 <= param_size <= 4.2:
        return "phi-4", "phi-4-mini-4b"

    # Default Phi-4 (≈14B).
    if 12.0 <= param_size <= 16.0:
        return "phi-4", "phi-4-14b"

    # If the name says phi-4 but we can't infer the size, keep the family and leave size blank.
    return "phi-4", None


def extract_base_model_from_name(
    model_name: str,
    *,
    architecture: str,
    param_size: float,
    allow_qwen1: bool = False,
) -> Tuple[Optional[str], Optional[str]]:
    """Return (base_model_family, identified_base_model) or (None, None)."""
    if not model_name:
        return None, None

    name_lc = _strip_trailing_parens(model_name.lower())
    arch_lc = architecture.lower() if architecture else ""

    # 0) Falcon3 (must come before generic Falcon / Llama logic)
    fam, ident = _identify_falcon3(name_lc, param_size)
    if fam and ident:
        return fam, ident

    # 1) Phi-4 (must come before generic Phi-* logic)
    fam, ident = _identify_phi4(name_lc, param_size)
    if fam:
        return fam, ident

    # 2) Qwen2 architecture hard constraint
    if "qwen2" in arch_lc:
        base_family = _qwen_series_from_context(name_lc, param_size, allow_qwen1=allow_qwen1)
        # Size
        size = extract_size_from_name(name_lc)
        if size:
            # Preserve MoE active-parameter tags like "A14B" / "A3B" when present.
            active = None
            m_active = re.search(r"[\-_]a(\d+\.?\d*)b(?!\w)", name_lc)
            if m_active:
                active = f"a{m_active.group(1)}b"
            if active:
                return base_family, f"{base_family}-{size}-{active}"
            return base_family, f"{base_family}-{size}"
        # Fallback to param-size buckets
        if 0.35 <= param_size <= 0.70:
            return base_family, f"{base_family}-0.5b"
        if 1.2 <= param_size <= 2.2:
            return base_family, f"{base_family}-1.5b"
        if 2.8 <= param_size <= 3.8:
            return base_family, f"{base_family}-3b"
        if 6.8 <= param_size <= 8.6:
            return base_family, f"{base_family}-7b"
        if 13.0 <= param_size <= 16.0:
            return base_family, f"{base_family}-14b"
        if 29.0 <= param_size <= 36.0:
            return base_family, f"{base_family}-32b"
        if 54.0 <= param_size <= 60.0:
            return base_family, f"{base_family}-57b"
        if 70.0 <= param_size <= 80.0:
            return base_family, f"{base_family}-72b"
        # Unknown Qwen2 size (e.g., MoE); leave blank.
        return base_family, None

    # 3) Llama family (model name / base model usually contains version)
    if re.search(r"llama[\-_]?3\.2|llama[/\\]3\.2", name_lc):
        base_family = "llama-3.2"
    elif re.search(r"llama[\-_]?3\.1|llama[/\\]3\.1", name_lc):
        base_family = "llama-3.1"
    elif re.search(r"llama[\-_]?3(?!\.)|llama[/\\]3(?!\.)", name_lc):
        base_family = "llama-3"
    elif re.search(r"llama[\-_]?2|llama[/\\]2", name_lc):
        base_family = "llama-2"
    elif re.search(r"llama[\-_]?1|llama[/\\]1", name_lc):
        base_family = "llama-1"
    # Mistral / Mixtral
    elif re.search(r"mistral[\-_]?7b|mistral[/\\]7b", name_lc):
        base_family = "mistral-7b"
    elif "mixtral" in name_lc:
        base_family = "mixtral"
    # Gemma
    elif re.search(r"gemma[\-_]?3(?![0-9.])|gemma[/\\]3(?![0-9.])", name_lc):
        if "270m" in name_lc:
            return "gemma-3-270m", "gemma-3-270m"
        size = extract_size_from_name(name_lc)
        if size:
            fam = f"gemma-3-{size}"
            return fam, fam
        return None, None
    elif re.search(r"gemma[\-_]?2[\-_]?2b|gemma[/\\]2[\-_]?2b", name_lc):
        base_family = "gemma-2-2b"
    elif re.search(r"gemma[\-_]?2[\-_]?9b|gemma[/\\]2[\-_]?9b", name_lc):
        base_family = "gemma-2-9b"
    elif re.search(r"gemma[\-_]?2[\-_]?27b|gemma[/\\]2[\-_]?27b", name_lc):
        base_family = "gemma-2-27b"
    # Gemma v1 models often appear as "gemma-2b" / "gemma-7b" (without "gemma-1" in the name).
    elif re.search(r"gemma[\-_]?2b|gemma[/\\]2b", name_lc):
        return "gemma-1", "gemma-1-2b"
    elif re.search(r"gemma[\-_]?7b|gemma[/\\]7b", name_lc):
        return "gemma-1", "gemma-1-7b"
    elif re.search(r"gemma[\-_]?2|gemma[/\\]2", name_lc):
        # If gemma-2 without size, infer from params.
        if 1.8 <= param_size <= 2.2:
            base_family = "gemma-2-2b"
        elif 8.9 <= param_size <= 9.5:
            base_family = "gemma-2-9b"
        elif 26.5 <= param_size <= 29.0:
            base_family = "gemma-2-27b"
        else:
            return None, None
    elif re.search(r"gemma[\-_]?1|gemma[/\\]1", name_lc):
        base_family = "gemma-1"
    # Many Gemma fine-tunes put the size before the family token (e.g., "zephyr-7b-gemma");
    # if we see "gemma" anywhere, treat it as Gemma v1 unless the name explicitly matched Gemma-2 above.
    elif "gemma" in name_lc:
        base_family = "gemma-1"
    # Qwen (non-Qwen2 architectures)
    elif re.search(r"qwen[\-_]?3(?:[\-_]|$)|qwen[/\\]3(?:[\-_]|$)", name_lc):
        base_family = "qwen3"
    elif re.search(r"qwen[\-_]?2\.5|qwen[/\\]2\.5", name_lc):
        base_family = "qwen2.5"
    elif re.search(r"qwen[\-_]?2(?!\.)|qwen[/\\]2(?!\.)", name_lc):
        base_family = "qwen2"
    elif re.search(r"qwen[\-_]?1|qwen[/\\]1", name_lc):
        base_family = "qwen1"
    # Yi
    elif re.search(r"yi[\-_]?1\.5|yi[/\\]1\.5", name_lc):
        base_family = "yi-1.5"
    elif re.search(r"yi[\-_]?1(?!\.)|yi[/\\]1(?!\.)", name_lc):
        base_family = "yi-1"
    elif re.search(r"(^|[/\\])yi[\-_]?\d", name_lc):
        # Older Yi base models (e.g., 01-ai/Yi-34B) omit the explicit version token.
        base_family = "yi-1"
    # SmolLM3
    elif "smollm3" in name_lc:
        size = extract_size_from_name(name_lc)
        if size:
            fam = f"smollm3-{size}"
            return fam, fam
        return None, None
    # GPT-OSS
    elif "gpt-oss" in name_lc or "gpt_oss" in name_lc:
        size = extract_size_from_name(name_lc)
        if size:
            fam = f"gpt-oss-{size}"
            return fam, fam
        return None, None
    # Phi (pre-phi-4 handled above)
    elif re.search(r"phi[\-_]?3|phi[/\\]3", name_lc):
        base_family = "phi-3"
    elif re.search(r"phi[\-_]?2|phi[/\\]2", name_lc):
        base_family = "phi-2"
    elif re.search(r"phi[\-_]?1\.5|phi[/\\]1\.5", name_lc):
        base_family = "phi-1.5"
    elif re.search(r"phi[\-_]?1(?!\.)|phi[/\\]1(?!\.)", name_lc):
        base_family = "phi-1"
    # Falcon (older)
    elif re.search(r"falcon[\-_]?180b|falcon[/\\]180b", name_lc):
        base_family = "falcon-180b"
    elif re.search(r"falcon[\-_]?40b|falcon[/\\]40b", name_lc):
        base_family = "falcon-40b"
    elif re.search(r"falcon[\-_]?7b|falcon[/\\]7b", name_lc):
        base_family = "falcon-7b"
    # Other
    elif re.search(r"chatglm3|glm3", name_lc):
        base_family = "glm3"
    elif re.search(r"chatglm2|glm2", name_lc):
        base_family = "glm2"
    elif re.search(r"deepseek[\-_]?llm[\-_]?2|deepseek2", name_lc):
        base_family = "deepseek2"
    elif "deepseek" in name_lc:
        base_family = "deepseek"
    elif re.search(r"mpt[\-_]?30b|mpt[/\\]30b", name_lc):
        base_family = "mpt-30b"
    elif re.search(r"mpt[\-_]?7b|mpt[/\\]7b", name_lc):
        base_family = "mpt-7b"
    elif "stablelm" in name_lc:
        base_family = "stablelm"
    elif "bloom" in name_lc:
        base_family = "bloom"
    elif re.search(r"baichuan[\-_]?3|baichuan[/\\]3", name_lc):
        base_family = "baichuan-3"
    elif re.search(r"baichuan[\-_]?2|baichuan[/\\]2", name_lc):
        base_family = "baichuan-2"
    else:
        return None, None

    # Determine identified size string
    if re.search(r"\d+b$", base_family):
        return base_family, base_family

    size = extract_size_from_name(name_lc)
    if size:
        return base_family, f"{base_family}-{size}"

    # Fall back to parameter ranges where safe
    pm = match_param_size_to_model(param_size)
    if pm and pm.startswith(base_family):
        return base_family, pm

    # Special-case: "microsoft/phi-2" often appears without an explicit size; phi-2 is ~2.7B.
    if base_family == "phi-2" and 2.3 <= param_size <= 3.3:
        return base_family, "phi-2-2b"

    # If we can't determine the size reliably, do not guess.
    return None, None


def _token_count(base_family: Optional[str], identified: Optional[str]) -> Optional[float]:
    if not base_family:
        return None

    # Qwen2: the technical report notes Qwen2-0.5B used a 12T token dataset.
    if base_family == "qwen2" and (identified or "").lower().startswith("qwen2-0.5b"):
        return 12.0

    return BASE_MODEL_TOKENS.get(base_family)


def determine_pretraining_tokens(row: pd.Series) -> Tuple[Optional[float], Optional[str], Optional[str]]:
    model_name = _norm(row.get("fullname", ""))
    base_model_field = _norm(row.get("Base Model", ""))
    architecture = _norm(row.get("Architecture", ""))
    param_size = float(row.get("#Params (B)", float("nan")) or float("nan"))
    has_base_model_col = "Base Model" in row.index
    allow_qwen1 = not has_base_model_col

    # 1) Prefer Base Model field (official from leaderboard).
    for candidate in ([base_model_field, model_name] if has_base_model_col else [model_name]):
        if not candidate or candidate.lower() in {"nan", "none", "removed"}:
            continue
        family, ident = extract_base_model_from_name(
            candidate,
            architecture=architecture,
            param_size=param_size,
            allow_qwen1=allow_qwen1,
        )
        if family:
            tok = _token_count(family, ident)
            return (float(tok) if tok is not None else None), family, ident

    # 2) Architecture-only fallback (conservative; avoid Llama mislabels).
    arch_lc = architecture.lower()
    pm = match_param_size_to_model(param_size)

    if "qwen2" in arch_lc:
        ctx = (_norm_lc(base_model_field) + " " + _norm_lc(model_name)).strip() if has_base_model_col else _norm_lc(model_name)
        base_family = _qwen_series_from_context(ctx, param_size, allow_qwen1=allow_qwen1)
        if 0.35 <= param_size <= 0.70:
            ident = f"{base_family}-0.5b"
        elif 1.2 <= param_size <= 2.2:
            ident = f"{base_family}-1.5b"
        elif 2.8 <= param_size <= 3.8:
            ident = f"{base_family}-3b"
        elif 6.8 <= param_size <= 8.6:
            ident = f"{base_family}-7b"
        elif 13.0 <= param_size <= 16.0:
            ident = f"{base_family}-14b"
        elif 29.0 <= param_size <= 36.0:
            ident = f"{base_family}-32b"
        elif 54.0 <= param_size <= 60.0:
            ident = f"{base_family}-57b"
        elif 70.0 <= param_size <= 80.0:
            ident = f"{base_family}-72b"
        else:
            ident = None
        tok = _token_count(base_family, ident)
        return (float(tok) if tok is not None else None), base_family, ident

    # Mistral is architecture-specific in the OLL exports; safe to use as a fallback.
    if "mistralforcausallm" in arch_lc or arch_lc == "mistralmodel":
        if 6.0 <= param_size <= 9.0 or "7b" in _norm_lc(model_name):
            tok = _token_count("mistral-7b", "mistral-7b")
            return (float(tok) if tok is not None else None), "mistral-7b", "mistral-7b"
        return None, None, None

    # Mixtral is also architecture-specific.
    if "mixtralforcausallm" in arch_lc:
        ident = pm if pm and pm.startswith("mixtral") else None
        tok = _token_count("mixtral", ident)
        return (float(tok) if tok is not None else None), "mixtral", ident

    # Phi: many fine-tunes omit 'phi' in the model name; use architecture + rough size.
    if "phiforcausallm" in arch_lc or "phimultipleheadsforcasuallm" in arch_lc:
        if 1.0 <= param_size < 2.0:
            tok = _token_count("phi-1.5", None)
            return (float(tok) if tok is not None else None), "phi-1.5", None
        if 2.0 <= param_size <= 4.0:
            tok = _token_count("phi-2", "phi-2-2b")
            return (float(tok) if tok is not None else None), "phi-2", "phi-2-2b"
        return None, None, None

    if "phi3forcausallm" in arch_lc:
        fam = "phi-3"
        ident = pm if pm and pm.startswith("phi-3") else None
        tok = _token_count(fam, ident)
        return (float(tok) if tok is not None else None), fam, ident

    # If we see Falcon3 in names but couldn't parse it, don't fallback to Llama.
    if "falcon3" in _norm_lc(base_model_field) or "falcon3" in _norm_lc(model_name):
        return None, None, None

    # Llama: only accept if params match a known Llama bucket.
    if "llama" in arch_lc:
        if pm and (pm.startswith("llama-2") or pm.startswith("llama-3")):
            fam = pm.rsplit("-", 1)[0]
            tok = _token_count(fam, pm)
            return (float(tok) if tok is not None else None), fam, pm
        return None, None, None

    return None, None, None


def main(argv: Optional[list[str]] = None) -> None:
    ap = argparse.ArgumentParser(description="Annotate Open LLM Leaderboard with base-model + token columns.")
    ap.add_argument(
        "--input_csv",
        default="tables/open_llm_leaderboard/open_llm_leaderboard.csv",
        help="Input leaderboard CSV (no token/base columns).",
    )
    ap.add_argument(
        "--output_csv",
        default="tables/open_llm_leaderboard/open_llm_leaderboard_with_tokens.csv",
        help="Output CSV with added columns.",
    )
    ap.add_argument(
        "--unassigned_csv",
        default="tables/open_llm_leaderboard/open_llm_leaderboard_unassigned_base_models.csv",
        help="Write rows where `Identified base model` is blank.",
    )
    args = ap.parse_args(argv)

    df = pd.read_csv(args.input_csv)

    # Drop the pandas index column if present.
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])

    results = df.apply(determine_pretraining_tokens, axis=1)
    df["Pretraining tokens (T)"] = [r[0] for r in results]
    df["Base model family"] = [r[1] for r in results]
    df["Identified base model"] = [r[2] for r in results]

    df.to_csv(args.output_csv, index=False)

    unassigned = df[df["Identified base model"].isna() | (df["Identified base model"].astype(str).str.strip() == "")]
    unassigned.to_csv(args.unassigned_csv, index=False)

    matched = int(df["Identified base model"].notna().sum())
    total = int(len(df))
    print(f"[base-model] matched {matched}/{total} ({matched/total*100:.2f}%) -> {args.output_csv}")
    print(f"[base-model] unassigned {int(len(unassigned))} -> {args.unassigned_csv}")

    # Guardrail checks for known failure modes (new OLL export only).
    if "Base Model" in df.columns:
        qwen2_bad = df[
            df["Architecture"].astype(str).str.contains("Qwen2", case=False, na=False)
            & df["Base model family"].astype(str).str.lower().isin(["qwen1", "qwen1.5"])
        ]
        if len(qwen2_bad) > 0:
            print(f"[WARN] Found {len(qwen2_bad)} rows with Qwen2 architecture mapped to Qwen1; check patterns.")


if __name__ == "__main__":
    main()

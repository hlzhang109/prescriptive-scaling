from __future__ import annotations

import re
from typing import Any, Mapping, Optional, Sequence

# Public family names used in plot legends.
FAMILY_QWEN = "Qwen"
FAMILY_LLAMA = "Llama"
FAMILY_MISTRAL = "Mistral"
FAMILY_GEMMA = "Gemma"
FAMILY_PHI = "Phi"
FAMILY_GPT = "GPT"
FAMILY_OTHERS = "Others"

FAMILY_ORDER: tuple[str, ...] = (
    FAMILY_QWEN,
    FAMILY_LLAMA,
    FAMILY_MISTRAL,
    FAMILY_GEMMA,
    FAMILY_PHI,
    FAMILY_OTHERS,
)

FAMILY_ORDER_WITH_GPT: tuple[str, ...] = (
    FAMILY_QWEN,
    FAMILY_LLAMA,
    FAMILY_MISTRAL,
    FAMILY_GEMMA,
    FAMILY_PHI,
    FAMILY_GPT,
    FAMILY_OTHERS,
)

# Stable, readable palette (tab10-like) for categorical base model families.
FAMILY_COLORS: dict[str, str] = {
    FAMILY_QWEN: "#d62728",  # red
    FAMILY_LLAMA: "#1f77b4",  # blue
    FAMILY_MISTRAL: "#2ca02c",  # green
    FAMILY_GEMMA: "#9467bd",  # purple
    FAMILY_PHI: "#ff7f0e",  # orange
    FAMILY_OTHERS: "#7f7f7f",  # gray
    FAMILY_GPT: "#8c564b",  # brown
}

DEFAULT_BASE_MODEL_COLS: tuple[str, ...] = (
    "Base model family",
    "Identified base model",
    "mapped_base_model",
    "Base Model",
    "Base model",
    "Base Model family",
    "Base Model Family",
    "base_model",
    "model",
    "Model",
    "model_id",
    "name",
    "eval_name",
)


def _norm_str(value: object) -> str:
    if value is None:
        return ""
    try:
        s = str(value).strip()
    except Exception:
        return ""
    if s in ("", "nan", "NaN", "None"):
        return ""
    return s


def extract_base_model_name(
    row: Mapping[str, Any],
    *,
    cols: Sequence[str] = DEFAULT_BASE_MODEL_COLS,
) -> str:
    for col in cols:
        try:
            v = row.get(col)  # type: ignore[attr-defined]
        except Exception:
            try:
                v = row[col]  # type: ignore[index]
            except Exception:
                continue
        s = _norm_str(v)
        if s:
            return s
    return ""


_PHI_RE = re.compile(r"(^|[^a-z])phi([\\-\\s0-9]|$)")


def family_from_base_model(base_model_name: str, *, include_gpt: bool = False) -> str:
    """Map a base-model identifier string to a coarse family bucket.

    Buckets are chosen for plotting (Qwen/Llama/Mistral/Gemma/Phi/Others, plus GPT optionally).
    """
    s = _norm_str(base_model_name).lower()
    if not s:
        return FAMILY_OTHERS

    if "qwen" in s:
        return FAMILY_QWEN
    if "llama" in s:
        return FAMILY_LLAMA
    if "mistral" in s or "mixtral" in s:
        return FAMILY_MISTRAL
    if "gemma" in s:
        return FAMILY_GEMMA
    if s.startswith("phi") or _PHI_RE.search(s) is not None:
        return FAMILY_PHI
    if include_gpt:
        if "gpt" in s or "openai" in s:
            return FAMILY_GPT
    return FAMILY_OTHERS


def color_for_family(family: str) -> str:
    return FAMILY_COLORS.get(family, FAMILY_COLORS[FAMILY_OTHERS])


def colors_for_base_models(
    base_models: Sequence[str],
    *,
    include_gpt: bool = False,
) -> list[str]:
    return [color_for_family(family_from_base_model(b, include_gpt=include_gpt)) for b in base_models]


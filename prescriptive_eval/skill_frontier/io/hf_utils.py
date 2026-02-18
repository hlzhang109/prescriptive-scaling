"""Helpers for HuggingFace model links."""

from __future__ import annotations

import re

_HF_RE = re.compile(r"huggingface\\.co/([^\"\\s>]+)")


def extract_hf_repo_from_model_html(model_html: str) -> str:
    match = _HF_RE.search(model_html or "")
    if match:
        return match.group(1).strip()
    return (model_html or "").strip()

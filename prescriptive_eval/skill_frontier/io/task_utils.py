"""Helpers for task argument parsing."""

from __future__ import annotations

from typing import List, Optional


def parse_tasks_arg(tasks_arg: Optional[List[str]]) -> Optional[List[str]]:
    if not tasks_arg:
        return None
    if len(tasks_arg) == 1 and "," in str(tasks_arg[0]):
        return [t.strip() for t in str(tasks_arg[0]).split(",") if t.strip()]
    out: List[str] = []
    for t in tasks_arg:
        for part in str(t).split(","):
            part = part.strip()
            if part:
                out.append(part)
    return out or None

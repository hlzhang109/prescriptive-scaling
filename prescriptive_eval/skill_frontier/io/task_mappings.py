"""Task name mappings shared across plotting scripts."""

from __future__ import annotations

from typing import Dict, Iterable, List, Sequence

MAIN_TASKS: Sequence[str] = (
    "IFEval Raw",
    "BBH Raw",
    "MATH Lvl 5 Raw",
    "GPQA Raw",
    "MUSR Raw",
    "MMLU-PRO Raw",
)

MAIN_TASK_MAP_OLL_TO_NEW: Dict[str, str] = {
    "IFEval Raw": "leaderboard_ifeval_inst_level_strict_acc_none",
    "BBH Raw": "leaderboard_bbh_acc_norm_none",
    "MATH Lvl 5 Raw": "leaderboard_math_hard_exact_match_none",
    "GPQA Raw": "leaderboard_gpqa_acc_norm_none",
    "MUSR Raw": "leaderboard_musr_acc_norm_none",
    "MMLU-PRO Raw": "leaderboard_mmlu_pro_acc_none",
}

MAIN_TASKS_FIG3_ORDER: Sequence[str] = (
    "BBH Raw",
    "GPQA Raw",
    "IFEval Raw",
    "MATH Lvl 5 Raw",
    "MMLU-PRO Raw",
    "MUSR Raw",
)


def format_task_label(task: str) -> str:
    """Human-readable task label (strip ' Raw' and BBH prefixes)."""
    name = task
    if name.startswith("leaderboard_bbh_"):
        name = name[len("leaderboard_bbh_") :]
    name = name.replace(" Raw", "").replace(" raw", "")
    return " ".join(name.split())


def order_tasks(tasks: Iterable[str], order: Sequence[str], *, drop_missing: bool = True) -> List[str]:
    tasks_set = set(tasks)
    ordered = [t for t in order if t in tasks_set]
    if drop_missing:
        return ordered
    ordered.extend(sorted(tasks_set.difference(order)))
    return ordered

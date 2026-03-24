#!/usr/bin/env python3
"""
Fit single-skill sigmoid frontiers for Epoch AI `benchmarks_runs.csv`.

Data sources:
  - epoch_ai/benchmarks_runs.csv: per (task, model) evaluation records with a numeric
    "Best score (across scorers)" column.
  - epoch_ai/models_info.csv: model metadata with "Training compute (FLOP)" and optional
    "Base model" pointers used to resolve missing compute.

For each task with at least `--min_models` unique models with resolved training compute
and a valid score, this script:
  - Fits a high-quantile sigmoid frontier using the repo's current fitter
    (`fit_sigmoid_frontier` from scripts/smooth_single_skill_frontier.py).
  - Writes structured outputs under `--out_dir`:
      sigmoid/no_split/{points,curves,plots}/
      metadata/{model_mapping.csv,task_summary.csv,unmatched_models.txt}
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import re
import sys
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

try:
    from skill_frontier.io.output_paths import sanitize_task_name
except Exception:

    def sanitize_task_name(task: str) -> str:
        name = task.replace(" ", "_")
        for ch in ['/', '\\', ':', '*', '?', '"', '<', '>', '|']:
            name = name.replace(ch, "_")
        return name


try:
    from scripts.smooth_single_skill_frontier import fit_sigmoid_frontier
except Exception:
    from smooth_single_skill_frontier import fit_sigmoid_frontier  # type: ignore

try:
    from skill_frontier.plotting.axis_formatting import apply_pretraining_compute_tick_multiplier
    from skill_frontier.plotting.configs import frontier_1d as frontier_1d_cfg
    from skill_frontier.plotting.configs import mpl_rc as mpl_rc_cfg
    from skill_frontier.plotting.labels import PRETRAINING_COMPUTE_FLOPS_LABEL
except Exception:  # pragma: no cover
    apply_pretraining_compute_tick_multiplier = None  # type: ignore

    class _Frontier1DCfg:  # type: ignore
        FIGSIZE = (7.0, 4.6)
        SCATTER_SIZE = 12
        SCATTER_ALPHA = 0.20
        SCATTER_LINEWIDTHS = 0.0
        CURVE_LINEWIDTH = 2.4
        CURVE_ALPHA = 0.95
        LABEL_FONTSIZE_X = 18
        LABEL_FONTSIZE_Y = 18
        TITLE_FONTSIZE = 18
        TICK_LABELSIZE = 15
        TICK_LENGTH = 4.0
        TICK_WIDTH = 0.9
        TICK_DIRECTION = "out"
        GRID_MAJOR_LINESTYLE = "-"
        GRID_MAJOR_LINEWIDTH = 0.9
        GRID_MAJOR_COLOR = "#d0d0d0"
        GRID_MAJOR_ALPHA = 0.7
        GRID_MINOR_LINESTYLE = ":"
        GRID_MINOR_LINEWIDTH = 0.6
        GRID_MINOR_COLOR = "#e0e0e0"
        GRID_MINOR_ALPHA = 0.7
        SPINE_LINEWIDTH = 1.1
        SPINE_COLOR = "#555555"
        LEGEND_FONTSIZE = 15
        LEGEND_FRAMEALPHA = 0.15
        LEGEND_LOC = "best"

    frontier_1d_cfg = _Frontier1DCfg()  # type: ignore

    class _MplRcCfg:  # type: ignore
        FONT_FAMILY = "serif"

    mpl_rc_cfg = _MplRcCfg()  # type: ignore
    PRETRAINING_COMPUTE_FLOPS_LABEL = "Pretraining Compute (FLOPs)"

X_TICK_MULTIPLIER: float = 1e21
TARGET_FIGSIZE: tuple[float, float] = (0.75 * (2069.0 / 300.0), 1344.0 / 300.0)


TOKEN_RE = re.compile(r"[a-z]+|\d+(?:\.\d+)?[a-z]*", re.IGNORECASE)
SIZE_TOKEN_RE = re.compile(r"^\d+(?:\.\d+)?b$", re.IGNORECASE)
MIX_SIZE_TOKEN_RE = re.compile(r"^\d+x\d+(?:\.\d+)?b$", re.IGNORECASE)
CTX_TOKEN_RE = re.compile(r"^\d+k$", re.IGNORECASE)
DATE_TOKEN_RE = re.compile(r"^\d{4}(?:\d{2}(?:\d{2})?)?$")  # 4,6,8 digits
VERSION_TOKEN_RE = re.compile(r"^v\d+(?:\.\d+)*$", re.IGNORECASE)


DROP_TOKENS_ANYWHERE: Set[str] = {
    "instruct",
    "chat",
    "hf",
    "dpo",
    "fp8",
    "preview",
    "thinking",
    "beta",
    "exp",
    "experimental",
    "it",
    # Note: keep family tokens like 'pro'/'flash' because they can be part of a model name.
}

DROP_TOKENS_TRAILING: Set[str] = {
    "high",
    "medium",
    "low",
    "xhigh",
    "xlow",
    "xl",
    "xs",
}

ORG_PREFIXES: Set[str] = {"meta", "openai", "anthropic", "google", "microsoft", "xai"}


def _canon_key(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(s).strip().lower())


def _tokens(s: str) -> List[str]:
    return [t.lower() for t in TOKEN_RE.findall(str(s).strip().lower()) if t.strip() != ""]


def _size_tokens(tokens: Iterable[str]) -> Set[str]:
    out: Set[str] = set()
    for t in tokens:
        if SIZE_TOKEN_RE.match(t) or MIX_SIZE_TOKEN_RE.match(t):
            out.add(t.lower())
    return out


def _safe_float(x: object) -> float:
    try:
        if x is None:
            return float("nan")
        s = str(x).strip()
        if s == "" or s.lower() in {"nan", "none", "na"}:
            return float("nan")
        return float(s)
    except Exception:
        return float("nan")


@dataclass
class ModelInfoIndex:
    # canonical key -> list of names (for display/debug)
    names: Dict[str, List[str]]
    # canonical key -> list of observed training compute values
    compute_vals: Dict[str, List[float]]
    # canonical key -> set of canonical base-model keys
    base_keys: Dict[str, Set[str]]
    # canonical key -> representative token set (from shortest model name)
    token_sets: Dict[str, Set[str]]
    # canonical key -> computed final training compute (resolved through base model)
    compute_final: Dict[str, Optional[float]]


def build_model_info_index(models_csv: str) -> ModelInfoIndex:
    names: Dict[str, List[str]] = {}
    compute_vals: Dict[str, List[float]] = {}
    base_keys: Dict[str, Set[str]] = {}

    # Use utf-8-sig to robustly handle UTF-8 BOM-prefixed CSVs (Epoch AI exports).
    with open(models_csv, "r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            model = str(row.get("Model", "") or "").strip()
            if not model:
                continue
            key = _canon_key(model)
            if not key:
                continue
            names.setdefault(key, []).append(model)
            comp = _safe_float(row.get("Training compute (FLOP)", ""))
            if np.isfinite(comp) and comp > 0.0:
                compute_vals.setdefault(key, []).append(float(comp))
            base = str(row.get("Base model", "") or "").strip()
            if base and base.lower() not in {"nan", "none", "na"}:
                base_k = _canon_key(base)
                if base_k:
                    base_keys.setdefault(key, set()).add(base_k)

    # Representative tokens per canonical key
    token_sets: Dict[str, Set[str]] = {}
    for key, ns in names.items():
        rep = min(ns, key=len)
        token_sets[key] = set(_tokens(rep))

    # Resolve final compute via base model pointers
    compute_final: Dict[str, Optional[float]] = {}
    visiting: Set[str] = set()

    def resolve(k: str) -> Optional[float]:
        if k in compute_final:
            return compute_final[k]
        if k in visiting:
            return None
        visiting.add(k)
        vals = compute_vals.get(k, [])
        if vals:
            vals_sorted = sorted(vals)
            out = float(vals_sorted[len(vals_sorted) // 2])
            compute_final[k] = out
            visiting.remove(k)
            return out
        for bk in base_keys.get(k, set()):
            v = resolve(bk)
            if v is not None and np.isfinite(v) and v > 0.0:
                compute_final[k] = float(v)
                visiting.remove(k)
                return compute_final[k]
        visiting.remove(k)
        compute_final[k] = None
        return None

    for k in names.keys():
        resolve(k)

    return ModelInfoIndex(
        names=names,
        compute_vals=compute_vals,
        base_keys=base_keys,
        token_sets=token_sets,
        compute_final=compute_final,
    )


def _candidate_keys_for_run_model(model: str) -> List[str]:
    toks = _tokens(model)
    if not toks:
        return []

    # Remove obvious context tokens (e.g., 16k/32k/128k) early.
    toks = [t for t in toks if not CTX_TOKEN_RE.match(t)]

    keys: List[str] = []

    def add(ts: Sequence[str]) -> None:
        if not ts:
            return
        keys.append(_canon_key(" ".join(ts)))

    add(toks)

    # Strip org prefix
    if toks and toks[0] in ORG_PREFIXES:
        add(toks[1:])

    # Remove common tokens anywhere (e.g., instruct/chat/hf)
    add([t for t in toks if t not in DROP_TOKENS_ANYWHERE])

    # Iteratively strip trailing tokens (e.g., dates, effort suffixes)
    for k in range(len(toks), 0, -1):
        ts = list(toks[:k])
        while ts and (
            ts[-1] in DROP_TOKENS_ANYWHERE
            or ts[-1] in DROP_TOKENS_TRAILING
            or VERSION_TOKEN_RE.match(ts[-1])
            or DATE_TOKEN_RE.match(ts[-1])
        ):
            ts = ts[:-1]
        add(ts)
        if ts and ts[0] in ORG_PREFIXES:
            add(ts[1:])
        # Also try removing DROP_TOKENS_ANYWHERE after trailing-strip
        add([t for t in ts if t not in DROP_TOKENS_ANYWHERE])

    # Unique, preserve order
    out: List[str] = []
    seen: Set[str] = set()
    for k in keys:
        if k and k not in seen:
            seen.add(k)
            out.append(k)
    return out


@dataclass
class ModelMatch:
    run_model: str
    matched_model: Optional[str]
    matched_key: Optional[str]
    compute_flop: Optional[float]
    method: str
    score: Optional[float] = None  # similarity score for fuzzy matches


def match_run_model_to_compute(model: str, index: ModelInfoIndex) -> ModelMatch:
    run_model = str(model)
    # Direct: exact canonical keys (and simple variants)
    for k in _candidate_keys_for_run_model(run_model):
        comp = index.compute_final.get(k)
        if comp is not None and np.isfinite(comp) and comp > 0.0:
            rep = min(index.names.get(k, [run_model]), key=len)
            return ModelMatch(
                run_model=run_model,
                matched_model=rep,
                matched_key=k,
                compute_flop=float(comp),
                method="direct",
                score=None,
            )

    # Fuzzy: token-set match with conservative constraints to avoid size mismatches.
    run_tokens = set(_tokens(run_model))
    run_sizes = _size_tokens(run_tokens)

    best: Optional[Tuple[float, str]] = None  # (score, matched_key)
    for k, comp in index.compute_final.items():
        if comp is None or not np.isfinite(comp) or comp <= 0.0:
            continue
        cand_tokens = index.token_sets.get(k, set())
        if not cand_tokens:
            continue
        cand_sizes = _size_tokens(cand_tokens)
        cand_core = cand_tokens - cand_sizes
        if not cand_core:
            continue

        # Size constraint:
        # - If the run name contains explicit size tokens (e.g., 70b), require that the candidate includes them.
        # - If the run name lacks size tokens, allow size-bearing candidates only if their *core* tokens are a subset of the run tokens.
        if run_sizes:
            if not run_sizes.issubset(cand_sizes):
                continue
        else:
            if not cand_core.issubset(run_tokens):
                continue

        inter = len(run_tokens & cand_core)
        if inter < max(2, min(3, len(cand_core))):
            continue
        union = len(run_tokens | cand_core)
        jacc = inter / float(max(1, union))

        subset_bonus = 0.10 if cand_core.issubset(run_tokens) else 0.0
        score = float(jacc + subset_bonus + 0.01 * inter)
        if best is None or score > best[0]:
            best = (score, k)

    if best is None:
        return ModelMatch(
            run_model=run_model,
            matched_model=None,
            matched_key=None,
            compute_flop=None,
            method="unmatched",
            score=None,
        )

    score, k = best
    comp = index.compute_final.get(k)
    rep = min(index.names.get(k, [run_model]), key=len)
    return ModelMatch(
        run_model=run_model,
        matched_model=rep,
        matched_key=k,
        compute_flop=float(comp) if comp is not None else None,
        method="fuzzy",
        score=float(score),
    )


def read_runs_best_scores(runs_csv: str) -> Dict[Tuple[str, str], float]:
    """Return {(task, model): best_score} for successful runs."""
    best: Dict[Tuple[str, str], float] = {}
    # Use utf-8-sig to robustly handle UTF-8 BOM-prefixed CSVs (Epoch AI exports).
    with open(runs_csv, "r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            status = str(row.get("Status", "") or "").strip().lower()
            if status != "success":
                continue
            task = str(row.get("task", "") or "").strip()
            model = str(row.get("model", "") or "").strip()
            if not task or not model:
                continue
            score = _safe_float(row.get("Best score (across scorers)", ""))
            if not np.isfinite(score):
                continue
            key = (task, model)
            prev = best.get(key)
            if prev is None or score > prev:
                best[key] = float(score)
    return best


def _write_csv(path: str, header: Sequence[str], rows: Iterable[Sequence[object]]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(list(header))
        for r in rows:
            w.writerow(list(r))


def _apply_frontier_style(ax, *, xlabel: str, ylabel: str, title: str) -> None:
    ax.set_xscale("log")
    ax.set_xlabel(xlabel, fontweight="bold", fontsize=frontier_1d_cfg.LABEL_FONTSIZE_X)
    if apply_pretraining_compute_tick_multiplier is not None and xlabel == PRETRAINING_COMPUTE_FLOPS_LABEL:
        apply_pretraining_compute_tick_multiplier(ax)
    ax.set_ylabel(ylabel, fontweight="bold", fontsize=frontier_1d_cfg.LABEL_FONTSIZE_Y)
    ax.set_title(title, fontweight="bold", fontsize=frontier_1d_cfg.TITLE_FONTSIZE)
    ax.tick_params(
        axis="both",
        labelsize=frontier_1d_cfg.TICK_LABELSIZE,
        length=frontier_1d_cfg.TICK_LENGTH,
        width=frontier_1d_cfg.TICK_WIDTH,
        direction=frontier_1d_cfg.TICK_DIRECTION,
    )
    try:
        import matplotlib.ticker as mticker  # type: ignore

        ax.yaxis.set_minor_locator(mticker.AutoMinorLocator(2))
        if ax.get_xscale() == "log":
            ax.xaxis.set_minor_locator(mticker.LogLocator(base=10.0, subs=(0.2, 0.4, 0.6, 0.8)))
    except Exception:
        pass

    ax.yaxis.grid(
        True,
        which="major",
        linestyle=frontier_1d_cfg.GRID_MAJOR_LINESTYLE,
        linewidth=frontier_1d_cfg.GRID_MAJOR_LINEWIDTH,
        color=frontier_1d_cfg.GRID_MAJOR_COLOR,
        alpha=frontier_1d_cfg.GRID_MAJOR_ALPHA,
    )
    ax.yaxis.grid(
        True,
        which="minor",
        linestyle=frontier_1d_cfg.GRID_MINOR_LINESTYLE,
        linewidth=frontier_1d_cfg.GRID_MINOR_LINEWIDTH,
        color=frontier_1d_cfg.GRID_MINOR_COLOR,
        alpha=frontier_1d_cfg.GRID_MINOR_ALPHA,
    )
    ax.xaxis.grid(
        True,
        which="major",
        linestyle=frontier_1d_cfg.GRID_MAJOR_LINESTYLE,
        linewidth=frontier_1d_cfg.GRID_MAJOR_LINEWIDTH,
        color=frontier_1d_cfg.GRID_MAJOR_COLOR,
        alpha=frontier_1d_cfg.GRID_MAJOR_ALPHA,
    )
    ax.xaxis.grid(
        True,
        which="minor",
        linestyle=frontier_1d_cfg.GRID_MINOR_LINESTYLE,
        linewidth=frontier_1d_cfg.GRID_MINOR_LINEWIDTH,
        color=frontier_1d_cfg.GRID_MINOR_COLOR,
        alpha=frontier_1d_cfg.GRID_MINOR_ALPHA,
    )
    for spine in ax.spines.values():
        spine.set_linewidth(frontier_1d_cfg.SPINE_LINEWIDTH)
        spine.set_color(frontier_1d_cfg.SPINE_COLOR)


def main(argv: Optional[Sequence[str]] = None) -> None:
    p = argparse.ArgumentParser(description="Fit Epoch AI single-skill sigmoid frontiers from benchmarks_runs.csv")
    p.add_argument("--runs_csv", default=os.path.join("epoch_ai", "benchmarks_runs.csv"))
    p.add_argument("--models_csv", default=os.path.join("epoch_ai", "models_info.csv"))
    p.add_argument("--out_dir", default="outputs_epoch_ai_benchmarks_runs")
    p.add_argument("--min_models", type=int, default=20, help="Minimum unique models per task to fit")
    p.add_argument("--tau", type=float, default=0.98, help="Quantile level for sigmoid frontier fit")
    p.add_argument("--kappa", type=float, default=50.0, help="Smoothing parameter κ for the pinball loss")
    p.add_argument("--lambda_b", type=float, default=1e-4, help="L2 penalty λ on sigmoid slope parameter b")
    p.add_argument("--no_plots", action="store_true", help="Skip plot generation (write CSVs only)")
    args = p.parse_args(argv)

    out_dir = str(args.out_dir)
    out_points = os.path.join(out_dir, "sigmoid", "no_split", "points")
    out_curves = os.path.join(out_dir, "sigmoid", "no_split", "curves")
    out_plots = os.path.join(out_dir, "sigmoid", "no_split", "plots")
    out_meta = os.path.join(out_dir, "metadata")
    for d in (out_points, out_curves, out_plots, out_meta):
        os.makedirs(d, exist_ok=True)

    print(f"[epoch_ai] loading models: {args.models_csv}")
    model_index = build_model_info_index(str(args.models_csv))

    print(f"[epoch_ai] loading runs: {args.runs_csv}")
    best_scores = read_runs_best_scores(str(args.runs_csv))
    tasks = sorted(set(t for (t, _m) in best_scores.keys()))
    models = sorted(set(m for (_t, m) in best_scores.keys()))
    print(f"[epoch_ai] unique tasks={len(tasks)} unique models={len(models)} (successful runs, best per task/model)")

    # Build model mapping once
    mapping: Dict[str, ModelMatch] = {}
    for m in models:
        mapping[m] = match_run_model_to_compute(m, model_index)

    # Write model mapping
    map_rows = []
    for m in sorted(models):
        mm = mapping[m]
        map_rows.append(
            [
                mm.run_model,
                mm.method,
                mm.matched_model or "",
                mm.matched_key or "",
                f"{mm.compute_flop:.12g}" if mm.compute_flop is not None else "",
                f"{mm.score:.6g}" if mm.score is not None else "",
            ]
        )
    _write_csv(
        os.path.join(out_meta, "model_mapping.csv"),
        ["run_model", "match_method", "matched_model_info", "matched_key", "training_compute_flop", "match_score"],
        map_rows,
    )

    unmatched = [m for m in models if mapping[m].compute_flop is None]
    with open(os.path.join(out_meta, "unmatched_models.txt"), "w") as f:
        for m in unmatched:
            f.write(m + "\n")
    print(f"[epoch_ai] compute resolved for {len(models) - len(unmatched)}/{len(models)} models")

    # Prepare per-task point tables
    task_to_points: Dict[str, List[Tuple[str, float, float]]] = {t: [] for t in tasks}
    for (task, model), score in best_scores.items():
        mm = mapping.get(model)
        if mm is None or mm.compute_flop is None or not np.isfinite(mm.compute_flop):
            continue
        c = float(mm.compute_flop)
        if not (np.isfinite(c) and c > 0.0):
            continue
        y = float(score)
        if not np.isfinite(y):
            continue
        # Clamp score to [0,1] for stability
        y = float(np.clip(y, 0.0, 1.0))
        task_to_points[task].append((model, c, y))

    # Task summary + fitting
    summary_rows = []
    fitted_tasks = 0
    for task in tasks:
        pts = task_to_points.get(task, [])
        # Ensure uniqueness by model (take max y if duplicates remain)
        by_model: Dict[str, Tuple[float, float]] = {}
        for m, c, y in pts:
            prev = by_model.get(m)
            if prev is None or y > prev[1]:
                by_model[m] = (c, y)
        pts_u = [(m, c, y) for m, (c, y) in by_model.items()]
        pts_u.sort(key=lambda t: t[1])
        n = len(pts_u)

        c_vals = np.array([c for (_m, c, _y) in pts_u], float)
        y_vals = np.array([y for (_m, _c, y) in pts_u], float)

        c_min = float(np.nanmin(c_vals)) if c_vals.size else float("nan")
        c_max = float(np.nanmax(c_vals)) if c_vals.size else float("nan")

        can_fit = n >= int(max(1, args.min_models))
        summary_rows.append(
            [
                task,
                n,
                f"{c_min:.6g}" if np.isfinite(c_min) else "",
                f"{c_max:.6g}" if np.isfinite(c_max) else "",
                int(can_fit),
            ]
        )
        if not can_fit:
            continue

        safe_task = sanitize_task_name(task)

        # Write points CSV
        points_path = os.path.join(out_points, f"{safe_task}.csv")
        point_rows = []
        for m, c, y in pts_u:
            z = math.log10(c) if c > 0 else float("nan")
            point_rows.append([m, f"{c:.12g}", f"{z:.12g}" if np.isfinite(z) else "", f"{y:.6g}"])
        _write_csv(points_path, ["model", "training_compute_flop", "log10_compute", "score"], point_rows)

        # Fit sigmoid frontier (match default paper hyperparameters)
        c_vals_scaled = c_vals / float(X_TICK_MULTIPLIER)
        xs_curve_scaled, y_curve = fit_sigmoid_frontier(
            c_vals_scaled,
            y_vals,
            tau=float(args.tau),
            use_log10_x=True,
            fit_mode="quantile_per_point",
            kappa_final=float(args.kappa),
            lambda_b=float(args.lambda_b),
        )
        if xs_curve_scaled.size == 0:
            print(f"[epoch_ai] WARN: fit failed for task={task} (n={n})")
            continue
        xs_curve_abs = xs_curve_scaled * float(X_TICK_MULTIPLIER)

        # Write curve CSV
        curve_path = os.path.join(out_curves, f"{safe_task}.csv")
        curve_rows = [[f"{float(x):.12g}", f"{float(y):.12g}", task] for x, y in zip(xs_curve_abs, y_curve)]
        _write_csv(curve_path, ["training_compute_flop", "y_hat", "task"], curve_rows)

        if not args.no_plots:
            try:
                import matplotlib.pyplot as plt  # type: ignore
                import matplotlib as mpl  # type: ignore
            except Exception:
                plt = None  # type: ignore
                mpl = None  # type: ignore
            if plt is not None:
                try:
                    if mpl is not None:
                        mpl.rcParams["font.family"] = mpl_rc_cfg.FONT_FAMILY
                except Exception:
                    pass
                fig, ax = plt.subplots(figsize=TARGET_FIGSIZE)
                ax.scatter(
                    c_vals_scaled,
                    y_vals,
                    s=float(frontier_1d_cfg.SCATTER_SIZE) * 4.5,
                    alpha=0.6,
                    color="black",
                    marker="X",
                    label="points",
                    linewidths=frontier_1d_cfg.SCATTER_LINEWIDTHS,
                    rasterized=True,
                )
                tau_label = float(args.tau)
                ax.plot(
                    xs_curve_scaled,
                    y_curve,
                    color="firebrick",
                    linewidth=frontier_1d_cfg.CURVE_LINEWIDTH,
                    label=f"Smooth τ={tau_label:.2f}",
                    alpha=frontier_1d_cfg.CURVE_ALPHA,
                )

                _apply_frontier_style(
                    ax,
                    xlabel=PRETRAINING_COMPUTE_FLOPS_LABEL,
                    ylabel="Best score",
                    title="",
                )
                # y-limits with padding
                y_min = float(np.nanmin([np.nanmin(y_vals), np.nanmin(y_curve)]))
                y_max = float(np.nanmax([np.nanmax(y_vals), np.nanmax(y_curve)]))
                if not np.isfinite(y_min) or not np.isfinite(y_max) or y_min == y_max:
                    y_min, y_max = 0.0, 1.0
                pad = 0.02 * max(1e-6, (y_max - y_min))
                ax.set_ylim(y_min - pad, y_max + pad)
                ax.legend(
                    loc=frontier_1d_cfg.LEGEND_LOC,
                    fontsize=frontier_1d_cfg.LEGEND_FONTSIZE,
                    frameon=True,
                    framealpha=frontier_1d_cfg.LEGEND_FRAMEALPHA,
                )
                fig.tight_layout()
                out_base = os.path.join(out_plots, safe_task)
                fig.savefig(out_base + ".png", dpi=300)
                fig.savefig(out_base + ".pdf", dpi=300)
                plt.close(fig)

        fitted_tasks += 1
        print(f"[epoch_ai] fitted task={task} n={n}")

    _write_csv(
        os.path.join(out_meta, "task_summary.csv"),
        ["task", "n_models_complete", "compute_min", "compute_max", "fit"],
        summary_rows,
    )

    print(f"[epoch_ai] done. fitted_tasks={fitted_tasks}. outputs in: {out_dir}")


if __name__ == "__main__":  # pragma: no cover
    main()

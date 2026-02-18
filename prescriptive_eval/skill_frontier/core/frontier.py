"""
Skill Frontier Estimation vs. Pretraining Compute
=================================================

This module implements the algorithm outlined in
docs/LLM_Eval_Frontier_Implementation_Plan.pdf to estimate skill
frontiers (Pareto boundaries) as a function of pretraining compute.

Key capabilities:
- Local-window frontier estimation at fixed compute (C0) via:
  - DEA (convex envelope, output-oriented, VRS) using LP
  - FDH (non-convex inner hull) without LP
  - Probabilistic frontier via directional weighted quantiles
- Support-function smoothing across compute with monotonicity
- Outputs support functions and per-task maxima; optional vertex export

Design goals:
- Object-oriented, modular, and easy to extend
- Detailed comments to aid debugging and comprehension
- Light dependencies by default (NumPy); optional SciPy/PuLP if installed

I/O contract (preferred minimal contract):
    - Input option A: a single CSV `merged.csv` with columns:
      ['model', 'logC', 'a_task1', ..., 'a_taskN']
      where accuracies are floats in [0,1] and `logC = log(pretrain_compute_zflops + eps)`
    - Input option B (LiveBench-native):
      - `livebench_subcategory_results_wide.csv` with columns ['model', <subcategory1>, ...]
      - `livebench_model_metadata_numeric.csv` in the same folder (or provided via flag) containing
        'pretrain_compute_zflops'. The loader will join on 'model', select sufficiently covered tasks,
        and drop rows missing compute or with missing task values.
- Outputs (optional, written if `output_dir` is provided):
    - support_functions.csv: per (C, direction, method) support values
    - max_per_task.csv: per (C, task, method) max attainable accuracy
    - frontier_vertices_{C}.csv: optional approximate vertices for DEA/FDH

Notes on numerical backends:
- DEA LP uses `scipy.optimize.linprog` if available; otherwise tries `pulp`.
- FDH requires no solver.
- Probabilistic frontier uses local weighted quantiles (no heavy regression).

This file can be used as a library or as a CLI script.
"""

from __future__ import annotations

import argparse
import dataclasses
import logging
import math
import os
import sys
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import csv as _csv

try:
    import numpy as np
except Exception as e:  # pragma: no cover - numpy is expected
    raise RuntimeError("NumPy is required for this module") from e


# Optional backends (scipy/pulp). These may not be installed; code handles gracefully.
try:  # SciPy LP solver
    from scipy.optimize import linprog as _scipy_linprog  # type: ignore
except Exception:  # pragma: no cover - optional
    _scipy_linprog = None

try:  # PuLP LP solver
    import pulp as _pulp  # type: ignore
except Exception:  # pragma: no cover - optional
    _pulp = None

# Optional robust statistics backends (may be unavailable)
try:
    from sklearn.covariance import MinCovDet as _MinCovDet  # type: ignore
except Exception:  # pragma: no cover - optional
    _MinCovDet = None
try:
    from scipy.stats import chi2 as _scipy_chi2  # type: ignore
except Exception:  # pragma: no cover - optional
    _scipy_chi2 = None

# Utilities moved into submodule to simplify this file
# Import from new package structure with fallbacks for backward compatibility
try:
    from skill_frontier.core.utils import (
        gaussian_kernel,
        epanechnikov_kernel,
        silverman_bandwidth,
        weighted_quantile,
        isotonic_regression_monotone_increasing,
        generate_simplex_directions,
        ModelPanel,
    )
except Exception:
    try:
        from .utils import (
            gaussian_kernel,
            epanechnikov_kernel,
            silverman_bandwidth,
            weighted_quantile,
            isotonic_regression_monotone_increasing,
            generate_simplex_directions,
            ModelPanel,
        )
    except Exception:
        from skill_frontier_utils import (
            gaussian_kernel,
            epanechnikov_kernel,
            silverman_bandwidth,
            weighted_quantile,
            isotonic_regression_monotone_increasing,
            generate_simplex_directions,
            ModelPanel,
        )


 # --------------------------------------------------------------------------------------
 # Data container and windowing
 # --------------------------------------------------------------------------------------


def load_merged_csv(path: str, task_prefix: str = "a_") -> ModelPanel:
    """Load a 'merged.csv' like file (Appendix B) without depending on pandas.

    Expected columns:
      - 'model' (str)
      - 'logC' (float)
      - task columns starting with `task_prefix` (default: 'a_')

    Returns a ModelPanel.
    """
    with open(path, "r", newline="") as f:
        reader = _csv.DictReader(f)
        rows = list(reader)
    if not rows:
        raise ValueError(f"No rows in {path}")
    # Detect task columns
    all_cols = rows[0].keys()
    task_cols = [c for c in all_cols if c.startswith(task_prefix)]
    if not task_cols:
        raise ValueError(
            f"No task columns found with prefix '{task_prefix}'. Columns: {list(all_cols)}"
        )
    models: List[str] = []
    logC: List[float] = []
    A_rows: List[List[float]] = []
    for r in rows:
        try:
            m = str(r["model"]) if r["model"] is not None else ""
        except KeyError as ke:
            raise ValueError("Missing required column 'model'") from ke
        try:
            lc = float(r["logC"]) if r["logC"] is not None else float("nan")
        except KeyError as ke:
            raise ValueError("Missing required column 'logC'") from ke
        if not m or not np.isfinite(lc):
            continue  # drop incomplete rows
        a_vec: List[float] = []
        ok = True
        for tc in task_cols:
            try:
                v = float(r[tc]) if r[tc] not in (None, "", "nan", "NaN") else float("nan")
            except Exception:
                v = float("nan")
            if not np.isfinite(v):
                ok = False
                break
            a_vec.append(v)
        if not ok:
            continue
        models.append(m)
        logC.append(lc)
        A_rows.append(a_vec)
    if not models:
        raise ValueError("All rows filtered out due to missing/invalid values")
    return ModelPanel(models=models, logC=np.array(logC), A=np.array(A_rows), tasks=task_cols)


def load_livebench_wide_csv(
    wide_path: str,
    metadata_path: Optional[str] = None,
    min_models_per_task: int = 20,
    min_values_per_row: int = 5,
) -> ModelPanel:
    """Load LiveBench subcategory results (wide) and join with numeric metadata.

    - wide_path: path to 'livebench_subcategory_results_wide.csv'
      expected columns: ['model', <subcategory_1>, ..., <subcategory_n>]
    - metadata_path: path to 'livebench_model_metadata_numeric.csv'. If None, will attempt to
      locate a sibling file in the same directory as `wide_path`.
    - min_models_per_task: minimum number of models that must have a valid score for a task to keep it.
    - min_values_per_row: keep any row (model) that has at least this many valid task values among kept tasks.

    Behavior per the implementation plan:
    - Select a core panel of tasks by coverage (>= min_models_per_task non-missing entries).
    - Join metadata; compute logC from 'pretrain_compute_zflops' (drop rows missing compute).
    - Drop models with any missing task value among the selected tasks.
    """
    # Resolve metadata path if not provided
    if metadata_path is None:
        candidate = os.path.join(os.path.dirname(wide_path), "livebench_model_metadata_numeric.csv")
        metadata_path = candidate if os.path.isfile(candidate) else None
    if metadata_path is None:
        raise ValueError(
            "metadata_path not provided and could not auto-locate 'livebench_model_metadata_numeric.csv'"
        )

    # Read wide CSV
    with open(wide_path, "r", newline="") as f:
        reader = _csv.DictReader(f)
        rows = list(reader)
    if not rows:
        raise ValueError(f"No rows in {wide_path}")
    # Identify task columns (all non-'model')
    all_cols = list(rows[0].keys())
    if "model" not in all_cols:
        raise ValueError("Expected column 'model' in wide CSV")
    task_cols = [c for c in all_cols if c != "model"]
    # Parse into arrays with NaN for missing entries
    models_raw: List[str] = []
    A_raw: List[List[float]] = []
    for r in rows:
        m = str(r.get("model", "") or "").strip()
        if not m:
            continue
        vec: List[float] = []
        for c in task_cols:
            v_raw = r.get(c, None)
            if v_raw is None or v_raw == "":
                vec.append(float("nan"))
            else:
                try:
                    vec.append(float(v_raw))
                except Exception:
                    vec.append(float("nan"))
        models_raw.append(m)
        A_raw.append(vec)
    A_raw_arr = np.array(A_raw, dtype=float)

    # Select tasks by coverage
    non_missing_counts = np.sum(np.isfinite(A_raw_arr), axis=0)
    keep_task_mask = non_missing_counts >= max(1, int(min_models_per_task))
    kept_tasks = [t for t, k in zip(task_cols, keep_task_mask) if k]
    if len(kept_tasks) == 0:
        raise ValueError(
            f"No tasks meet coverage threshold >= {min_models_per_task}. Non-missing counts: {dict(zip(task_cols, non_missing_counts.tolist()))}"
        )
    A_kept = A_raw_arr[:, keep_task_mask]

    # Keep models with at least `min_values_per_row` valid values among kept tasks
    valid_counts = np.sum(np.isfinite(A_kept), axis=1)
    row_ok = valid_counts >= max(1, int(min_values_per_row))
    models_kept = [m for m, ok in zip(models_raw, row_ok) if ok]
    A_kept = A_kept[row_ok, :]

    # Load metadata numeric and compute logC
    with open(metadata_path, "r", newline="") as f:
        mreader = _csv.DictReader(f)
        mrows = list(mreader)
    # Build mapping model -> pretrain_compute_zflops (float or nan)
    compute_map: Dict[str, float] = {}
    for r in mrows:
        m = str(r.get("model", "") or "").strip()
        v = r.get("pretrain_compute_zflops", None)
        try:
            c = float(v) if v not in (None, "", "nan", "NaN") else float("nan")
        except Exception:
            c = float("nan")
        compute_map[m] = c
    # Create compute vector for kept models; drop those without compute
    logC_list: List[float] = []
    models_final: List[str] = []
    A_final_nans: List[List[float]] = []
    eps = 1e-12
    for m, arow in zip(models_kept, A_kept.tolist()):
        c = compute_map.get(m, float("nan"))
        if not np.isfinite(c):
            continue
        logC_list.append(float(math.log(c + eps)))
        models_final.append(m)
        A_final_nans.append(arow)

    if len(models_final) == 0:
        raise ValueError("After joining metadata and filtering, no models remain with compute and full task coverage.")

    # Impute remaining missing task values per column (task-wise median over the retained models)
    A_arr = np.array(A_final_nans, dtype=float)
    for j in range(A_arr.shape[1]):
        col = A_arr[:, j]
        mask = np.isfinite(col)
        if mask.any():
            med = float(np.nanmedian(col))
        else:
            med = 0.0
        col[~mask] = med
        A_arr[:, j] = np.clip(col, 0.0, 1.0)

    panel = ModelPanel(models=models_final, logC=np.array(logC_list), A=A_arr, tasks=kept_tasks)
    logging.getLogger("load_livebench_wide_csv").info(
        "Loaded LiveBench wide: %d models, %d tasks (kept from %d); coverage threshold=%d",
        panel.num_models,
        panel.num_tasks,
        len(task_cols),
        min_models_per_task,
    )
    return panel


try:
    from skill_frontier.core.window import KernelWindow
except Exception:
    try:
        from .window import KernelWindow
    except Exception:
        from skill_frontier_window import KernelWindow


# --------------------------------------------------------------------------------------
# DEA (LP) and FDH estimators
# --------------------------------------------------------------------------------------


@dataclasses.dataclass
class DEAConfig:
    var_returns_to_scale: bool = True  # VRS: sum alpha_j = 1; CRS if False
    alpha_cap_multiple: float = 4.0  # regularization: alpha_j <= cap_multiple * w_j
    # If True, enforce monotone non-decreasing support in C via isotonic smoothing later
    enforce_monotone_in_C: bool = True


class DEAEstimator:
    """DEA support-function estimator under a local kernel window.

    Solves, for a fixed direction u and target compute C0 (expressed in logC space via constraint
    over actual C), the LP:

      maximize   sum_j alpha_j * (u^T a_j)
      subject to sum_j alpha_j * C_j <= C0
                 sum_j alpha_j == 1                 [VRS]
                 0 <= alpha_j <= cap_j              [regularization cap via local weights]

    where cap_j = alpha_cap_multiple * w_j, with w_j the normalized kernel weights at C0.

    Notes:
    - We operate in logC space for grid/widow creation. For the LP capacity constraint we need
      actual C; as a proxy we use exp(logC). The relative scale is what matters.
    - If `var_returns_to_scale` is False, we drop the sum alpha_j = 1 constraint (CRS variant).
    """

    def __init__(self, panel: ModelPanel, cfg: Optional[DEAConfig] = None) -> None:
        self.panel = panel
        self.cfg = cfg or DEAConfig()
        self._logger = logging.getLogger(self.__class__.__name__)
        self._has_scipy = _scipy_linprog is not None
        self._has_pulp = _pulp is not None
        if not (self._has_scipy or self._has_pulp):
            self._logger.warning(
                "No LP backend found (scipy or pulp). DEA will be unavailable."
            )

    def is_available(self) -> bool:
        return self._has_scipy or self._has_pulp

    def support_value(
        self,
        u: np.ndarray,
        idx: np.ndarray,
        weights: np.ndarray,
        target_logC: float,
    ) -> Tuple[float, Optional[np.ndarray]]:
        """Solve the DEA LP and return (support_value, solution_alpha_on_panel_or_None).

        - u: direction vector in R_+^n, L1-normalized (recommended)
        - idx: indices of models in the local window
        - weights: normalized kernel weights aligned to idx
        - target_logC: scalar logC target for the compute constraint
        """
        if not self.is_available():
            return float("nan"), None
        a_w = self.panel.A[idx, :]  # (k, n)
        logC_w = self.panel.logC[idx]
        C_w = np.exp(logC_w)
        # Objective: maximize sum_j alpha_j * (u^T a_j)
        prof = a_w @ u  # (k,)
        # Bounds 0 <= alpha_j <= cap_j
        cap = self.cfg.alpha_cap_multiple * weights
        cap = np.maximum(cap, 1e-12)

        if self._has_scipy:
            # Convert to minimization for linprog
            c = -prof  # minimize -objective
            # Inequality: sum_j alpha_j * C_j <= C0
            A_ub = np.expand_dims(C_w, axis=0)
            b_ub = np.array([math.exp(target_logC)], dtype=float)

            A_eq = None
            b_eq = None
            if self.cfg.var_returns_to_scale:
                A_eq = np.ones((1, len(idx)))
                b_eq = np.array([1.0], dtype=float)

            bounds = [(0.0, float(cap_i)) for cap_i in cap]

            res = _scipy_linprog(c=c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs")
            if not res.success:
                self._logger.debug(f"linprog failed at logC0={target_logC:.3f}, status={res.status} {res.message}")
                return float("nan"), None
            alpha_w = res.x
            value = float(prof @ alpha_w)
        else:
            # Fallback to PuLP
            prob = _pulp.LpProblem("DEA_support", _pulp.LpMaximize)
            alpha_vars = [_pulp.LpVariable(f"a_{j}", lowBound=0.0, upBound=float(cap[j])) for j in range(len(idx))]
            # Objective
            prob += _pulp.lpSum([float(prof[j]) * alpha_vars[j] for j in range(len(idx))])
            # Capacity constraint
            prob += _pulp.lpSum([float(C_w[j]) * alpha_vars[j] for j in range(len(idx))]) <= float(math.exp(target_logC))
            # VRS
            if self.cfg.var_returns_to_scale:
                prob += _pulp.lpSum(alpha_vars) == 1.0
            # Solve
            status = prob.solve(_pulp.PULP_CBC_CMD(msg=False))
            if status != 1:  # 1 == optimal
                self._logger.debug(f"PuLP solve failed at logC0={target_logC:.3f}, status={status}")
                return float("nan"), None
            alpha_w = np.array([v.value() for v in alpha_vars], dtype=float)
            value = float(prof @ alpha_w)
        # Map alphas back to full panel index space if needed by caller
        alpha_full = np.zeros(self.panel.num_models, dtype=float)
        alpha_full[idx] = alpha_w
        return value, alpha_full


class FDHEstimator:
    """FDH (Free Disposal Hull) inner frontier.

    For a given window at C0, consider the subset S = {j: C_j <= C0 and weight_j >= threshold}.
    The FDH support in direction u is simply max_{j in S} u^T a_j.
    Undominated points are useful for reporting vertices/diagnostics.
    """

    def __init__(self, panel: ModelPanel, weight_threshold_frac: float = 1e-6) -> None:
        self.panel = panel
        self.weight_threshold_frac = weight_threshold_frac

    def support_value(
        self, u: np.ndarray, idx: np.ndarray, weights: np.ndarray, target_logC: float
    ) -> Tuple[float, Optional[int]]:
        logC_w = self.panel.logC[idx]
        C_w = np.exp(logC_w)
        a_w = self.panel.A[idx, :]
        # Restrict to models with C_j <= C0 and significant weight
        wmax = float(np.max(weights)) if len(weights) else 0.0
        mask = (C_w <= math.exp(target_logC)) & (weights >= self.weight_threshold_frac * wmax)
        cand_idx_local = np.nonzero(mask)[0]
        if cand_idx_local.size == 0:
            return float("nan"), None
        scores = (a_w[cand_idx_local, :] @ u)
        best_local = int(cand_idx_local[int(np.argmax(scores))])
        best_global_idx = int(idx[best_local])
        best_value = float(scores.max())
        return best_value, best_global_idx

    def undominated_indices(
        self, idx: np.ndarray, target_logC: float, weights: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Return global indices of undominated points within C_j <= C0 among the given window.

        A point p dominates q if A[p] >= A[q] coordinatewise and A[p] != A[q].
        """
        logC_w = self.panel.logC[idx]
        C_w = np.exp(logC_w)
        a_w = self.panel.A[idx, :]
        mask = C_w <= math.exp(target_logC)
        cand = np.nonzero(mask)[0]
        if cand.size == 0:
            return np.array([], dtype=int)
        A = a_w[cand, :]
        keep = np.ones(A.shape[0], dtype=bool)
        for i in range(A.shape[0]):
            if not keep[i]:
                continue
            # any j that dominates i -> drop i
            dominated = np.all(A >= A[i, :], axis=1) & np.any(A > A[i, :], axis=1)
            dominated[i] = False
            if np.any(dominated):
                keep[i] = False
        return idx[cand[keep]].astype(int)


# --------------------------------------------------------------------------------------
# Probabilistic frontier via directional weighted quantiles
# --------------------------------------------------------------------------------------


@dataclasses.dataclass
class QuantileConfig:
    quantiles: Tuple[float, ...] = (0.95, 0.99)
    enforce_monotone_in_C: bool = True


class QuantileFrontierEstimator:
    """Directional local-quantile support estimator.

    For each direction u and compute grid C0, compute weighted quantile of y_j = u^T a_j
    using kernel weights at C0. This is a local approximation to conditional quantile regression.
    """

    def __init__(self, panel: ModelPanel, cfg: Optional[QuantileConfig] = None) -> None:
        self.panel = panel
        self.cfg = cfg or QuantileConfig()

    def support_values(
        self, u: np.ndarray, idx: np.ndarray, weights: np.ndarray
    ) -> Dict[float, float]:
        a_w = self.panel.A[idx, :]
        y = a_w @ u
        out: Dict[float, float] = {}
        for q in self.cfg.quantiles:
            out[q] = weighted_quantile(y, q, weights)
        return out


# --------------------------------------------------------------------------------------
# Configuration and Orchestrator
# --------------------------------------------------------------------------------------


@dataclasses.dataclass
class FrontierConfig:
    # Grid in logC
    num_C_grid: int = 24
    # Kernel settings
    kernel: str = "gaussian"  # or 'epanechnikov'
    bandwidth: Optional[float] = None  # if None, Silverman rule on logC
    support_threshold_frac: float = 1e-3
    # Directions
    num_directions: int = 128
    direction_seed: int = 0
    # DEA/FDH/Quantile
    dea: DEAConfig = dataclasses.field(default_factory=DEAConfig)
    quantile: QuantileConfig = dataclasses.field(default_factory=QuantileConfig)
    # Smoothing across C
    smooth_monotone_in_C: bool = True
    # Output
    write_vertices: bool = False  # if True, write frontier_vertices_{C}.csv for DEA/FDH
    # Lower-frontier (floors) computation via sign-flip
    compute_lower: bool = False
    lower_suffix: str = "_LOWER"
    # Robust outlier filtering within each compute window
    robust_enable: bool = False
    robust_support_fraction: float = 0.9
    robust_alpha: float = 0.005
    robust_alpha_lower: float = 0.05
    robust_min_task_cov: float = 0.9
    robust_scale_floor: float = 0.05
    robust_action: str = "drop"  # or 'winsorize' (drop-only supported in supports)
    robust_winsor_q_low: float = 0.05
    robust_k_bandwidth: float = 3.0
    chance_baseline: Optional[Sequence[float]] = None
    # Overlay raw points on single-task plots
    overlay_raw_points: bool = True


class SkillFrontier:
    """Main entry point for estimating skill frontiers over compute.

    Usage pattern:
      panel = load_merged_csv(path)
      frontier = SkillFrontier(panel, cfg)
      results = frontier.run(output_dir=...)

    The `run` method returns a dictionary with numpy arrays and also writes CSVs if requested.
    """

    def __init__(self, panel: ModelPanel, cfg: Optional[FrontierConfig] = None) -> None:
        self.panel = panel
        self.cfg = cfg or FrontierConfig()
        self._logger = logging.getLogger(self.__class__.__name__)

        # Will be filled during run
        self.grid_logC: Optional[np.ndarray] = None
        self.window: Optional[KernelWindow] = None
        self.directions: Optional[np.ndarray] = None
        self.supports: Optional[Dict[str, np.ndarray]] = None
        self.max_per_task_cache: Optional[Dict[str, np.ndarray]] = None
        self.min_per_task_cache: Optional[Dict[str, np.ndarray]] = None

    def _build_grid(self) -> np.ndarray:
        # Use data min/max logC as grid bounds
        lo = float(np.min(self.panel.logC))
        hi = float(np.max(self.panel.logC))
        # Slightly expand bounds to cover edge models
        pad = 0.01 * (hi - lo + 1e-9)
        lo -= pad
        hi += pad
        grid = np.linspace(lo, hi, num=self.cfg.num_C_grid)
        self._logger.debug(f"Grid logC in [{lo:.3f}, {hi:.3f}] with {len(grid)} points")
        return grid

    def _precompute_windows(self, grid_logC: np.ndarray) -> KernelWindow:
        window = KernelWindow.build(
            logC=self.panel.logC,
            grid_logC=grid_logC,
            bandwidth=self.cfg.bandwidth,
            kernel=self.cfg.kernel,
            support_threshold_frac=self.cfg.support_threshold_frac,
        )
        return window

    def _compute_support_tables(
        self,
        directions: np.ndarray,
        grid_logC: np.ndarray,
        window: KernelWindow,
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, Optional[List[object]]]]:
        """Compute support functions h_C(u) for DEA, FDH, and quantile frontiers.

        Returns:
            supports: dict mapping method -> (num_C_grid, num_dirs) array
            extras: dict with method-specific extras (e.g., solutions/indices) per grid
        """
        nC = len(grid_logC)
        nU = directions.shape[0]
        supports: Dict[str, np.ndarray] = {}
        extras: Dict[str, Optional[List[object]]] = {}

        # DEA
        dea_est = DEAEstimator(self.panel, self.cfg.dea)
        if dea_est.is_available():
            H_dea = np.full((nC, nU), np.nan, dtype=float)
            # Store solution alphas for vertex extraction later (list of lists per C)
            solu: List[List[Optional[np.ndarray]]] = [[] for _ in range(nC)]
            for iC, lc0 in enumerate(grid_logC):
                idx = window.indices[iC]
                w = window.weights[iC]
                if self.cfg.robust_enable:
                    idx, w = self._robust_filter_window(idx, w)
                for iU, u in enumerate(directions):
                    val, alpha_full = dea_est.support_value(u, idx, w, lc0)
                    H_dea[iC, iU] = val
                    solu[iC].append(alpha_full)
            # Monotone smoothing across C (per direction)
            if self.cfg.smooth_monotone_in_C and self.cfg.dea.enforce_monotone_in_C:
                self._smooth_monotone_columns(H_dea)
            supports["DEA"] = H_dea
            extras["DEA"] = solu
        else:
            self._logger.warning("Skipping DEA: no LP backend available.")

        # FDH
        fdh_est = FDHEstimator(self.panel)
        H_fdh = np.full((nC, nU), np.nan, dtype=float)
        # Store best indices for vertex extraction
        best_idx: List[List[Optional[int]]] = [[] for _ in range(nC)]
        for iC, lc0 in enumerate(grid_logC):
            idx = window.indices[iC]
            w = window.weights[iC]
            if self.cfg.robust_enable:
                idx, w = self._robust_filter_window(idx, w)
            for iU, u in enumerate(directions):
                val, j = fdh_est.support_value(u, idx, w, lc0)
                H_fdh[iC, iU] = val
                best_idx[iC].append(j)
        if self.cfg.smooth_monotone_in_C:
            self._smooth_monotone_columns(H_fdh)
        supports["FDH"] = H_fdh
        extras["FDH"] = best_idx

        # Quantile frontier(s)
        q_est = QuantileFrontierEstimator(self.panel, self.cfg.quantile)
        for q in self.cfg.quantile.quantiles:
            key = f"Q{q:.2f}"
            Hq = np.full((nC, nU), np.nan, dtype=float)
            for iC, _lc0 in enumerate(grid_logC):
                idx = window.indices[iC]
                w = window.weights[iC]
                if self.cfg.robust_enable:
                    idx, w = self._robust_filter_window(idx, w)
                for iU, u in enumerate(directions):
                    vals = q_est.support_values(u, idx, w)
                    Hq[iC, iU] = vals[q]
            if self.cfg.smooth_monotone_in_C and self.cfg.quantile.enforce_monotone_in_C:
                self._smooth_monotone_columns(Hq)
            supports[key] = Hq
            extras[key] = None

        return supports, extras

    def _smooth_monotone_columns(self, H: np.ndarray) -> None:
        """Apply isotonic regression per column across compute grid where finite.

        In-place modification; skips columns with <2 finite values.
        """
        for i in range(H.shape[1]):
            col = H[:, i]
            mask = np.isfinite(col)
            if mask.sum() >= 2:
                H[mask, i] = isotonic_regression_monotone_increasing(col[mask])

    def _robust_filter_window(self, idx: np.ndarray, weights: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Apply robust outlier filtering to a window's indices and weights.

        Uses MinCovDet if available; falls back to diagonal-MAD distances.
        Keeps behavior graceful on small windows by skipping filtering.
        """
        if idx.size == 0:
            return idx, weights
        A_w = self.panel.A[idx, :]
        n_tasks = A_w.shape[1]
        # Coverage filter per model
        cov = np.mean(np.isfinite(A_w), axis=1)
        keep_cov = cov >= float(self.cfg.robust_min_task_cov)
        if keep_cov.any():
            idx = idx[keep_cov]
            weights = weights[keep_cov]
            A_w = A_w[keep_cov, :]
        # Skip if too few points
        if A_w.shape[0] < max(3, n_tasks + 2):
            s = float(weights.sum())
            if s > 0:
                weights = weights / s
            return idx, weights
        # Optional chance baseline normalization
        X = A_w.copy()
        if self.cfg.chance_baseline is not None:
            b = np.array(self.cfg.chance_baseline, dtype=float)
            if b.shape[0] == n_tasks:
                denom = np.clip(1.0 - b, 1e-6, 1.0)
                X = (X - b.reshape(1, -1)) / denom.reshape(1, -1)
                X = np.clip(X, 0.0, 1.0)
        # Robust standardization via median/MAD
        m = np.nanmedian(X, axis=0)
        mad = np.nanmedian(np.abs(X - m.reshape(1, -1)), axis=0)
        s = 1.4826 * mad
        s = np.maximum(s, float(self.cfg.robust_scale_floor))
        Z = (np.nan_to_num(X, nan=m) - m.reshape(1, -1)) / s.reshape(1, -1)
        # Distances
        if _MinCovDet is not None:
            try:
                mcd = _MinCovDet(support_fraction=float(self.cfg.robust_support_fraction), random_state=0)
                mcd.fit(Z)
                D2 = mcd.mahalanobis(Z)
            except Exception:
                D2 = np.sum(Z * Z, axis=1)
        else:
            D2 = np.sum(Z * Z, axis=1)
        # Cutoff
        if _scipy_chi2 is not None:
            try:
                cutoff = float(_scipy_chi2.ppf(1.0 - float(self.cfg.robust_alpha), df=n_tasks))
            except Exception:
                cutoff = float(np.nanquantile(D2, 1.0 - float(self.cfg.robust_alpha)))
        else:
            cutoff = float(np.nanquantile(D2, 1.0 - float(self.cfg.robust_alpha)))
        outlier_mask = D2 > cutoff
        if not np.any(outlier_mask):
            s = float(weights.sum())
            if s > 0:
                weights = weights / s
            return idx, weights
        # Only 'drop' action supported for now
        keep = ~outlier_mask
        if keep.sum() < max(2, n_tasks):
            # Too few after drop; keep all
            keep = np.ones_like(keep, dtype=bool)
        idx2 = idx[keep]
        w2 = weights[keep]
        s = float(w2.sum())
        if s > 0:
            w2 = w2 / s
        else:
            if len(w2) > 0:
                w2 = np.ones_like(w2, dtype=float) / float(len(w2))
        return idx2, w2

    def _max_per_task(
        self, supports: Dict[str, np.ndarray], directions: np.ndarray
    ) -> Dict[str, np.ndarray]:
        """Compute max-per-task curves by projecting the frontier onto each axis.

        For each method and C, use the support in direction e_i to read off max a_i.
        Assumes the first `num_tasks` rows in `directions` are axis directions e_i.
        Returns dict method -> array (num_C_grid, num_tasks).
        """
        num_tasks = self.panel.num_tasks
        axis_slice = slice(0, num_tasks)
        out: Dict[str, np.ndarray] = {}
        for method, H in supports.items():
            # H: (num_C, num_dirs)
            Amax = H[:, axis_slice]
            out[method] = Amax
        return out

    def _write_support_functions_csv(
        self,
        path: str,
        grid_logC: np.ndarray,
        directions: np.ndarray,
        supports: Dict[str, np.ndarray],
    ) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", newline="") as f:
            w = _csv.writer(f)
            # header
            n = directions.shape[1]
            w.writerow(["logC", "direction_id", *(f"u_{i+1}" for i in range(n)), "h", "method"])
            for method, H in supports.items():
                for iC, lc0 in enumerate(grid_logC):
                    for iU, u in enumerate(directions):
                        h = float(H[iC, iU])
                        row = [float(lc0), int(iU), *[float(x) for x in u], h, method]
                        w.writerow(row)

    def _write_max_per_task_csv(
        self,
        path: str,
        grid_logC: np.ndarray,
        tasks: List[str],
        max_per_task: Dict[str, np.ndarray],
    ) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", newline="") as f:
            w = _csv.writer(f)
            w.writerow(["logC", "task", "max_accuracy", "method"])
            for method, Amax in max_per_task.items():
                for iC, lc0 in enumerate(grid_logC):
                    for iT, t in enumerate(tasks):
                        w.writerow([float(lc0), t, float(Amax[iC, iT]), method])

    def _write_min_per_task_csv(
        self,
        path: str,
        grid_logC: np.ndarray,
        tasks: List[str],
        min_per_task: Dict[str, np.ndarray],
    ) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", newline="") as f:
            w = _csv.writer(f)
            w.writerow(["logC", "task", "min_accuracy", "method"])
            for method, Amin in min_per_task.items():
                for iC, lc0 in enumerate(grid_logC):
                    for iT, t in enumerate(tasks):
                        w.writerow([float(lc0), t, float(Amin[iC, iT]), method])

    def _write_vertices_for_C(
        self,
        out_dir: str,
        iC: int,
        lc0: float,
        directions: np.ndarray,
        supports: Dict[str, np.ndarray],
        extras: Dict[str, Optional[List[object]]],
    ) -> None:
        """Write approximate vertices for DEA and FDH at a given C grid point.

        - DEA: reconstruct a* = sum_j alpha_j a_j from LP solutions across directions
        - FDH: take the best points (indices) across directions; deduplicate and undominated subset
        """
        # Collect candidate points per method
        method_points: Dict[str, List[np.ndarray]] = {k: [] for k in supports.keys()}
        # DEA points
        if "DEA" in supports and extras.get("DEA") is not None:
            sol_list = extras["DEA"][iC]  # type: ignore[index]
            for alpha_full in sol_list:  # type: ignore[assignment]
                if alpha_full is None:
                    continue
                a_star = alpha_full @ self.panel.A  # (N,) @ (N,n) -> (n,)
                method_points["DEA"].append(a_star)
        # FDH points
        if "FDH" in supports and extras.get("FDH") is not None:
            idx_list = extras["FDH"][iC]  # type: ignore[index]
            for j in idx_list:  # type: ignore[assignment]
                if j is None:
                    continue
                method_points["FDH"].append(self.panel.A[int(j), :])
        # Write CSV per method if there are points
        for method, pts in method_points.items():
            if not pts:
                continue
            # Deduplicate and select undominated set
            P = np.unique(np.round(np.vstack(pts), decimals=6), axis=0)
            keep = np.ones(P.shape[0], dtype=bool)
            for i in range(P.shape[0]):
                if not keep[i]:
                    continue
                dominated = np.all(P >= P[i, :], axis=1) & np.any(P > P[i, :], axis=1)
                dominated[i] = False
                if np.any(dominated):
                    keep[i] = False
            P = P[keep, :]
            out_path = os.path.join(out_dir, f"frontier_vertices_{iC:03d}_{method}.csv")
            with open(out_path, "w", newline="") as f:
                w = _csv.writer(f)
                w.writerow(["vertex_id", *self.panel.tasks])
                for vi, row in enumerate(P):
                    w.writerow([vi, *[float(x) for x in row]])

    def run(self, output_dir: Optional[str] = None) -> Dict[str, object]:
        # Build compute grid and windows
        grid_logC = self._build_grid()
        self.grid_logC = grid_logC
        window = self._precompute_windows(grid_logC)
        self.window = window
        # Directions (includes axis+balanced)
        dirs = generate_simplex_directions(
            n_dim=self.panel.num_tasks,
            num_directions=self.cfg.num_directions,
            seed=self.cfg.direction_seed,
        )
        self.directions = dirs

        # Compute supports (upper)
        supports, extras = self._compute_support_tables(dirs, grid_logC, window)

        # Optionally compute lower-frontier via sign-flip
        min_per_task: Dict[str, np.ndarray] = {}
        if self.cfg.compute_lower:
            # Flipped panel A -> -A
            panel_neg = ModelPanel(models=self.panel.models, logC=self.panel.logC.copy(), A=(-self.panel.A).copy(), tasks=self.panel.tasks)
            # Disable smoothing for flipped pass; we'll smooth after negation
            cfg_neg = dataclasses.replace(
                self.cfg,
                smooth_monotone_in_C=False,
                dea=dataclasses.replace(self.cfg.dea, enforce_monotone_in_C=False),
                quantile=dataclasses.replace(self.cfg.quantile, enforce_monotone_in_C=False),
            )
            runner_neg = SkillFrontier(panel_neg, cfg_neg)
            supp_neg, _ = runner_neg._compute_support_tables(dirs, grid_logC, window)
            for method, Hneg in supp_neg.items():
                Hlower = (-Hneg).copy()
                # Enforce non-decreasing floors across compute if enabled
                if self.cfg.smooth_monotone_in_C:
                    nC, nU = Hlower.shape
                    for iU in range(nU):
                        col = Hlower[:, iU]
                        mask = np.isfinite(col)
                        if mask.sum() >= 2:
                            Hlower[mask, iU] = isotonic_regression_monotone_increasing(col[mask])
                Hlower = np.clip(Hlower, 0.0, 1.0)
                lname = f"{method}{self.cfg.lower_suffix}"
                supports[lname] = Hlower
            # Build per-task floors from axis directions for lower methods
            num_tasks = self.panel.num_tasks
            axis_slice = slice(0, num_tasks)
            for m, H in supports.items():
                if m.endswith(self.cfg.lower_suffix):
                    min_per_task[m] = H[:, axis_slice]

        self.supports = supports

        # Per-task curves (axis supports)
        max_per_task = self._max_per_task(supports, dirs)
        self.max_per_task_cache = max_per_task
        self.min_per_task_cache = min_per_task if self.cfg.compute_lower else None

        # Optional outputs
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            self._write_support_functions_csv(
                os.path.join(output_dir, "support_functions.csv"), grid_logC, dirs, supports
            )
            self._write_max_per_task_csv(
                os.path.join(output_dir, "max_per_task.csv"), grid_logC, self.panel.tasks, max_per_task
            )
            if self.cfg.compute_lower and min_per_task:
                self._write_min_per_task_csv(
                    os.path.join(output_dir, "min_per_task.csv"), grid_logC, self.panel.tasks, min_per_task
                )
            if self.cfg.write_vertices:
                for iC, lc0 in enumerate(grid_logC):
                    self._write_vertices_for_C(output_dir, iC, lc0, dirs, supports, extras)

        return {
            "grid_logC": grid_logC,
            "directions": dirs,
            "supports": supports,
            "max_per_task": max_per_task,
            "min_per_task": min_per_task if self.cfg.compute_lower else None,
            "extras": extras,
        }

    # ------------------------------------------------------------------
    # Interpolation helpers: predict per-task maxima at arbitrary logC
    # ------------------------------------------------------------------

    def predict_max_per_task_at(self, logC_value: float, method: str) -> np.ndarray:
        """Predict per-task maximum accuracy at a given logC via monotone linear interpolation.

        - logC_value: target compute in log-space
        - method: one of the keys in `supports` (e.g., 'DEA', 'FDH', 'Q0.95', 'Q0.99')

        Returns a 1D array of length `num_tasks` with predicted maxima for each task.
        """
        if self.grid_logC is None or self.directions is None or self.supports is None:
            raise RuntimeError("Run the estimator first (call run()) before prediction")
        if method not in self.supports:
            raise ValueError(f"Unknown method '{method}'. Available: {list(self.supports.keys())}")
        H = self.supports[method]  # (num_C, num_dirs)
        num_tasks = self.panel.num_tasks
        axis_slice = slice(0, num_tasks)
        # Extract axis-direction supports per task
        H_axis = H[:, axis_slice]  # (num_C, num_tasks)
        x = self.grid_logC
        # Interpolate per task with robustness to NaNs (use nearest valid as fallback)
        out = np.zeros(num_tasks, dtype=float)
        for i in range(num_tasks):
            y = H_axis[:, i]
            mask = np.isfinite(y)
            if mask.sum() == 0:
                out[i] = float("nan")
                continue
            # Clamp outside range to boundary values
            if logC_value <= x[mask][0]:
                out[i] = float(y[mask][0])
                continue
            if logC_value >= x[mask][-1]:
                out[i] = float(y[mask][-1])
                continue
            out[i] = float(np.interp(logC_value, x[mask], y[mask]))
        return out

    def predict_max_per_task_curve(self, logC_values: np.ndarray, method: str) -> np.ndarray:
        """Predict per-task maxima for multiple logC values.

        Returns an array of shape (len(logC_values), num_tasks).
        """
        logC_values = np.asarray(logC_values, dtype=float)
        curves = np.vstack([self.predict_max_per_task_at(lc, method) for lc in logC_values])
        return curves

    # ------------------------------------------------------------------
    # Plotting: per-task curves across methods
    # ------------------------------------------------------------------

    def plot_max_per_task(
        self,
        output_dir: str,
        num_points: int = 200,
        methods: Optional[Sequence[str]] = None,
        dpi: int = 300,
    ) -> None:
        """Generate per-task plots of max accuracy vs. pretraining compute for each method.

        - output_dir: directory to save figures (PDF and PNG)
        - num_points: number of x points in the dense logC grid
        - methods: which methods to plot; defaults to available among ['DEA','FDH','Q0.95','Q0.99']
        - dpi: output DPI (applies to PNG; PDF is vector but we pass dpi for consistency)
        """
        if self.grid_logC is None or self.directions is None or self.supports is None:
            raise RuntimeError("Run the estimator first (call run()) before plotting")

        # Optional imports (fall back gracefully)
        try:
            import matplotlib.pyplot as plt  # type: ignore
            import matplotlib as mpl  # type: ignore
        except Exception as e:  # pragma: no cover
            raise RuntimeError("matplotlib is required for plotting") from e
        try:
            import seaborn as sns  # type: ignore
            sns.set_theme(style="darkgrid", context="talk")
        except Exception:
            # If seaborn is not available, continue with matplotlib defaults
            pass
        # Use serif fonts globally in figures
        try:
            mpl.rcParams["font.family"] = "serif"
        except Exception:
            pass

        os.makedirs(output_dir, exist_ok=True)
        # Dense grid in logC and corresponding compute values C = exp(logC)
        lc_min = float(np.nanmin(self.grid_logC))
        lc_max = float(np.nanmax(self.grid_logC))
        lc_grid = np.linspace(lc_min, lc_max, num_points)
        C_grid = np.exp(lc_grid)

        # Choose methods to plot (include LOWER variants by default if present)
        suffix = getattr(self.cfg, "lower_suffix", "_LOWER")
        avail = set(self.supports.keys())
        if methods is None:
            base_order = ("DEA", "FDH", "Q0.95", "Q0.99")
            bases = [b for b in base_order if (b in avail or f"{b}{suffix}" in avail)]
            methods = []
            for b in bases:
                if b in avail:
                    methods.append(b)
                lower_name = f"{b}{suffix}"
                if lower_name in avail:
                    methods.append(lower_name)
        else:
            methods = list(methods)
            # Build bases seen for color mapping
            base_order = tuple(dict.fromkeys([m[:-len(suffix)] if m.endswith(suffix) else m for m in methods]))
            bases = list(base_order)
        if not methods:
            raise RuntimeError("No available methods to plot.")

        # Color palette per base method; LOWER reuses base color with dashed style
        try:
            import seaborn as sns  # type: ignore
            base_palette = sns.color_palette("husl", n_colors=len(bases))
        except Exception:
            base_palette = None

        # Helpers for 1D per-task robust filtering in plots
        def _filter_1d(idx: np.ndarray, w: np.ndarray, values: np.ndarray, alpha: float) -> Tuple[np.ndarray, np.ndarray]:
            # Keep only finite values
            mask_fin = np.isfinite(values)
            idx = idx[mask_fin]
            w = w[mask_fin]
            v = values[mask_fin]
            if v.size == 0:
                return idx, w
            # Median/MAD z-scores
            m = float(np.nanmedian(v))
            mad = float(np.nanmedian(np.abs(v - m)))
            s = 1.4826 * mad
            s = max(s, float(self.cfg.robust_scale_floor))
            z = (v - m) / s
            # Chi2 cutoff df=1 -> z^2 > cutoff
            try:
                from math import sqrt
                if _scipy_chi2 is not None:
                    zthr = float(np.sqrt(_scipy_chi2.ppf(1.0 - float(alpha), df=1)))
                else:
                    zthr = 3.0
            except Exception:
                zthr = 3.0
            keep = np.abs(z) <= zthr
            if not np.any(keep):
                return idx, w
            idx2 = idx[keep]
            w2 = w[keep]
            sW = float(w2.sum())
            if sW > 0:
                w2 = w2 / sW
            else:
                w2 = np.ones_like(w2, dtype=float) / float(len(w2))
            return idx2, w2

        # Estimators for axis-direction supports in plots
        dea_est = DEAEstimator(self.panel, self.cfg.dea)
        fdh_est = FDHEstimator(self.panel)

        for ti, task in enumerate(self.panel.tasks):
            plt.figure(figsize=(7, 4.5))
            # Track y-range across all plotted methods for dynamic ylim
            y_min, y_max = float("inf"), float("-inf")
            for mi, method in enumerate(methods):
                # Build discrete values along the runner grid using per-task 1D filtering
                y_grid = np.full(len(self.grid_logC), np.nan, dtype=float)
                axis_u = np.zeros(self.panel.num_tasks)
                axis_u[ti] = 1.0
                is_lower = method.endswith(suffix)
                base_name = method[:-len(suffix)] if is_lower else method
                for iC, lc0 in enumerate(self.grid_logC):
                    idx = self.window.indices[iC]
                    w_win = self.window.weights[iC]
                    # 1D filter on this task only
                    v = self.panel.A[idx, ti]
                    alpha_use = self.cfg.robust_alpha_lower if is_lower else self.cfg.robust_alpha
                    idx_f, w_f = _filter_1d(idx, w_win, v, alpha_use) if len(idx) > 0 else (idx, w_win)
                    if idx_f.size == 0:
                        y_grid[iC] = float('nan')
                        continue
                    if base_name == "DEA" and dea_est.is_available():
                        u = -axis_u if is_lower else axis_u
                        val, _ = dea_est.support_value(u, idx_f, w_f, float(lc0))
                        y_grid[iC] = (-float(val) if is_lower else float(val))
                    elif base_name == "FDH":
                        u = -axis_u if is_lower else axis_u
                        val, _ = fdh_est.support_value(u, idx_f, w_f, float(lc0))
                        y_grid[iC] = (-float(val) if is_lower else float(val))
                    elif base_name.startswith("Q"):
                        # Parse quantile level from name like Q0.95
                        try:
                            q = float(base_name[1:])
                        except Exception:
                            q = 0.95
                        q_eff = (1.0 - q) if is_lower else q
                        # Weighted quantile of task values
                        y_grid[iC] = float(weighted_quantile(self.panel.A[idx_f, ti], q_eff, w_f))
                    else:
                        y_grid[iC] = float('nan')
                # Optional monotone smoothing across compute
                if base_name == "DEA" and self.cfg.smooth_monotone_in_C and self.cfg.dea.enforce_monotone_in_C:
                    mask = np.isfinite(y_grid)
                    if mask.sum() >= 2:
                        y_grid[mask] = isotonic_regression_monotone_increasing(y_grid[mask])
                if base_name == "FDH" and self.cfg.smooth_monotone_in_C:
                    mask = np.isfinite(y_grid)
                    if mask.sum() >= 2:
                        y_grid[mask] = isotonic_regression_monotone_increasing(y_grid[mask])
                if base_name.startswith("Q") and self.cfg.smooth_monotone_in_C and self.cfg.quantile.enforce_monotone_in_C:
                    mask = np.isfinite(y_grid)
                    if mask.sum() >= 2:
                        y_grid[mask] = isotonic_regression_monotone_increasing(y_grid[mask])
                # Interpolate to dense grid respecting NaNs similar to predict helper
                mask = np.isfinite(y_grid)
                if mask.sum() == 0:
                    y = np.full_like(lc_grid, np.nan, dtype=float)
                else:
                    xg = self.grid_logC[mask]
                    yg = y_grid[mask]
                    y = np.interp(lc_grid, xg, yg, left=yg[0], right=yg[-1])
                # Update y-range ignoring NaNs
                if np.isfinite(y).any():
                    y_min = min(y_min, float(np.nanmin(y)))
                    y_max = max(y_max, float(np.nanmax(y)))
                # Determine color/style from base method
                is_lower = method.endswith(suffix)
                base_name = method[:-len(suffix)] if is_lower else method
                try:
                    bi = bases.index(base_name)
                except ValueError:
                    bi = mi
                color = (base_palette[bi] if base_palette is not None else None)
                plt.plot(
                    C_grid,
                    y,
                    label=method,
                    linewidth=2.2,
                    color=color,
                    linestyle=("--" if is_lower else "-"),
                )
            # Optionally overlay raw points from the current panel for this task
            if getattr(self.cfg, "overlay_raw_points", False):
                C_raw = np.exp(self.panel.logC)
                y_raw = self.panel.A[:, ti]
                mask = np.isfinite(C_raw) & np.isfinite(y_raw)
                if mask.any():
                    # Downsample if very dense for visibility
                    idxs = np.nonzero(mask)[0]
                    max_pts = 3000
                    if idxs.size > max_pts:
                        rng = np.random.default_rng(0)
                        idxs = rng.choice(idxs, size=max_pts, replace=False)
                    plt.scatter(
                        C_raw[idxs],
                        y_raw[idxs],
                        s=14,
                        alpha=0.25,
                        c="#444444",
                        edgecolors="none",
                        label="raw points",
                        zorder=4,
                    )
                    y_min = min(y_min, float(np.nanmin(y_raw[idxs])))
                    y_max = max(y_max, float(np.nanmax(y_raw[idxs])))
            # Aesthetics
            plt.xscale("log")
            # Set y-limits based on data with slight padding, clamped to [0,1]
            if np.isfinite(y_min) and np.isfinite(y_max):
                span = max(1e-6, y_max - y_min)
                pad = max(0.02, 0.05 * span)
                y_lo = max(0.0, y_min - pad)
                y_hi = min(1.0, y_max + pad)
                if y_hi <= y_lo:  # degenerate case (flat line)
                    y_lo = max(0.0, y_min - 0.05)
                    y_hi = min(1.0, y_max + 0.05)
                plt.ylim(y_lo, y_hi)
            else:
                plt.ylim(0.0, 1.0)
            # Bold, large labels and title
            plt.xlabel("Pretraining Compute (FLOPs)", fontsize=14, fontweight="bold")
            try:
                from skill_frontier.plotting.axis_formatting import (  # type: ignore
                    apply_pretraining_compute_tick_multiplier,
                )

                apply_pretraining_compute_tick_multiplier(plt.gca())
            except Exception:
                pass
            plt.ylabel("Accuracy (upper/lower)", fontsize=14, fontweight="bold")
            title = f"Acc. Boundaries vs Compute — {task}"
            plt.title(title, fontsize=16, fontweight="bold")
            # Ticks styling
            plt.tick_params(axis='both', which='major', labelsize=12)
            # Compact legend at top-left
            handles, labels = plt.gca().get_legend_handles_labels()
            uniq = {}
            for h, l in zip(handles, labels):
                if l not in uniq:
                    uniq[l] = h
            if uniq:
                leg = plt.legend(
                    list(uniq.values()),
                    list(uniq.keys()),
                    loc="upper left",
                    fontsize=8,
                    frameon=True,
                    fancybox=False,
                    framealpha=0.85,
                    borderpad=0.25,
                    labelspacing=0.2,
                    handlelength=1.0,
                    handletextpad=0.3,
                    borderaxespad=0.2,
                    markerscale=0.9,
                    title="Method (solid=upper, dashed=lower)",
                    title_fontsize=8,
                )
                if leg and leg.get_title():
                    leg.get_title().set_fontweight("bold")
            plt.tight_layout()
            # Save both PDF and PNG (300 dpi)
            base = os.path.join(output_dir, f"max_accuracy_vs_compute__{task}")
            plt.savefig(base + ".pdf", dpi=dpi, bbox_inches="tight")
            plt.savefig(base + ".png", dpi=dpi, bbox_inches="tight")
            plt.close()

    @staticmethod
    def _halfspace_intersection_polygon(A: List[np.ndarray], b: List[float]) -> np.ndarray:
        """Compute 2D convex polygon from halfspaces a_k^T x <= b_k.

        Returns vertices ordered around the hull. Adds a small tolerance for feasibility.
        """
        tol = 1e-9
        pts: List[Tuple[float, float]] = []
        m = len(A)
        for i in range(m):
            ai = A[i]
            bi = b[i]
            for j in range(i + 1, m):
                aj = A[j]
                bj = b[j]
                M = np.array([[ai[0], ai[1]], [aj[0], aj[1]]], dtype=float)
                det = np.linalg.det(M)
                if abs(det) < 1e-12:
                    continue
                x = np.linalg.solve(M, np.array([bi, bj], dtype=float))
                # Check feasibility
                if all((A[k][0] * x[0] + A[k][1] * x[1]) <= b[k] + 1e-7 for k in range(m)):
                    pts.append((float(x[0]), float(x[1])))
        if not pts:
            return np.zeros((0, 2), dtype=float)
        # Deduplicate approximately
        P = np.unique(np.round(np.array(pts), 6), axis=0)
        # Remove points outside [0,1]^2 with tolerance
        mask = (
            (P[:, 0] >= -tol)
            & (P[:, 1] >= -tol)
            & (P[:, 0] <= 1 + tol)
            & (P[:, 1] <= 1 + tol)
        )
        P = P[mask]
        if P.shape[0] == 0:
            return np.zeros((0, 2), dtype=float)
        # Order around centroid
        c = P.mean(axis=0)
        ang = np.arctan2(P[:, 1] - c[1], P[:, 0] - c[0])
        order = np.argsort(ang)
        return P[order]

    def plot_pairwise_frontiers(
        self,
        output_dir: str,
        pairwise_points: int = 128,
        C_count: int = 3,
        methods: Optional[Sequence[str]] = None,
        dpi: int = 300,
    ) -> None:
        """Plot true 2D frontiers for each task pair at selected compute grid points.

        - pairwise_points: number of 2D directions on the positive simplex for halfspaces
        - C_count: number of logC grid points to sample (evenly across the grid)
        """
        if self.grid_logC is None or self.window is None or self.supports is None:
            raise RuntimeError("Run the estimator first (call run()) before plotting")
        try:
            import matplotlib.pyplot as plt  # type: ignore
            import matplotlib as mpl  # type: ignore
        except Exception as e:
            raise RuntimeError("matplotlib is required for plotting") from e
        try:
            import seaborn as sns  # type: ignore
            sns.set_theme(style="darkgrid", context="talk")
        except Exception:
            pass
        # Serif fonts in all figures
        try:
            mpl.rcParams["font.family"] = "serif"
        except Exception:
            pass

        os.makedirs(output_dir, exist_ok=True)
        # Prepare extreme-points log file (one per task pair and compute slice)
        extreme_path = os.path.join(output_dir, "extreme_models.txt")
        try:
            extreme_f = open(extreme_path, "w", newline="")
            extreme_f.write("task_x,task_y,C_index,logC,C,model,a_x,a_y\n")
        except Exception:
            extreme_f = None

        # Choose methods (exclude FDH by default; can still be passed explicitly)
        default_methods = [m for m in ("DEA", "Q0.95", "Q0.99") if m in self.supports]
        methods = list(methods) if methods is not None else default_methods
        methods = [m for m in methods if m != "FDH"]
        if not methods:
            raise RuntimeError("No available methods to plot.")
        # Colors per method
        try:
            import seaborn as sns  # type: ignore
            method_palette = {m: c for m, c in zip(methods, sns.color_palette("husl", n_colors=len(methods)))}
        except Exception:
            method_palette = {m: None for m in methods}
        # Line styles per compute index
        linestyles = ["solid", "dashed", "dotted", "dashdot"]

        # Prepare estimators
        dea_est = DEAEstimator(self.panel, self.cfg.dea)
        fdh_est = FDHEstimator(self.panel)
        q_est = QuantileFrontierEstimator(self.panel, self.cfg.quantile)

        # Compute selection: if 3 slices, use fixed FLOPs cutoffs (54 and 756) for bands
        nC = len(self.grid_logC)
        band_masks: Optional[List[np.ndarray]] = None
        bands_label: Optional[List[str]] = None
        band_colors: Optional[List[Tuple[float, float, float]]] = None
        if C_count == 3:
            C_all = np.exp(self.panel.logC)
            c1, c2 = 54.0, 756.0
            # Masks in C-space
            band_masks = [
                C_all <= c1,
                (C_all > c1) & (C_all <= c2),
                C_all > c2,
            ]
            bands_label = [
                f"C ≤ {c1:.3g}",
                f"{c1:.3g} < C ≤ {c2:.3g}",
                f"C > {c2:.3g}",
            ]
            # Representative grid indices near band medians (fallback to geometric centers)
            def nearest_idx(val: float) -> int:
                return int(np.argmin(np.abs(self.grid_logC - val)))
            Cmin = float(np.nanmin(C_all)) if C_all.size else 1.0
            Cmax = float(np.nanmax(C_all)) if C_all.size else 1.0
            centers_C: List[float] = []
            for k, mask in enumerate(band_masks):
                if np.any(mask):
                    centers_C.append(float(np.exp(np.nanmedian(self.panel.logC[mask]))))
                else:
                    if k == 0:
                        centers_C.append(max(Cmin, 0.5 * c1))
                    elif k == 1:
                        centers_C.append(float(np.sqrt(c1 * c2)))
                    else:
                        centers_C.append(min(Cmax, 2.0 * c2))
            C_indices = [nearest_idx(float(np.log(c))) for c in centers_C]
            try:
                import seaborn as sns  # type: ignore
                band_colors = list(sns.color_palette("Set2", n_colors=3))
            except Exception:
                band_colors = [(0.30, 0.68, 0.95), (0.20, 0.80, 0.60), (0.98, 0.55, 0.35)]
        else:
            preferred = [3, 13, 23]
            if all((i < nC) for i in preferred) and C_count == 3:
                C_indices = preferred
            else:
                if C_count >= nC:
                    C_indices = list(range(nC))
                else:
                    C_indices = np.linspace(0, nC - 1, num=C_count).round().astype(int).tolist()

        # 2D directions on simplex for (i,j)
        def two_directions(k: int) -> np.ndarray:
            # include endpoints and balanced; L1-normalized
            t = np.linspace(0.0, 1.0, num=k)
            return np.stack([t, 1 - t], axis=1)

        T = self.panel.tasks
        nT = len(T)
        for i in range(nT):
            for j in range(i + 1, nT):
                ti, tj = T[i], T[j]
                # Create one row of subplots, one per selected compute slice
                try:
                    import matplotlib.pyplot as plt  # type: ignore
                    fig, axes = plt.subplots(1, len(C_indices), figsize=(6.5 * len(C_indices), 6.0), squeeze=False)
                    axes = axes[0]
                except Exception as e:
                    raise
                # Prepare scatter marker styles per compute slice
                slice_markers = ["o", "s", "^", "D", "P", "X"]
                # Track dynamic axis ranges per subplot
                x_los = [np.nan] * len(C_indices)
                x_his = [np.nan] * len(C_indices)
                y_los = [np.nan] * len(C_indices)
                y_his = [np.nan] * len(C_indices)
                for idx_pos, iC in enumerate(C_indices):
                    ax = axes[idx_pos]
                    lc0 = float(self.grid_logC[iC])
                    idx = self.window.indices[iC]
                    w = self.window.weights[iC]
                    U2 = two_directions(pairwise_points)
                    # Build halfspaces per method (convex methods)
                    halfspaces: Dict[str, Tuple[List[np.ndarray], List[float]]] = {}
                    for method in methods:
                        halfspaces[method] = ([], [])
                    for u2 in U2:
                        # Full direction in R^n concentrated on (i,j)
                        u_full = np.zeros(nT)
                        u_full[i] = float(u2[0])
                        u_full[j] = float(u2[1])
                        # DEA
                        if "DEA" in methods and dea_est.is_available():
                            val, _ = dea_est.support_value(u_full, idx, w, lc0)
                            if np.isfinite(val):
                                halfspaces["DEA"][0].append(np.array([u2[0], u2[1]]))
                                halfspaces["DEA"][1].append(float(val))
                        # Quantiles
                        if q_est and any(m.startswith("Q") for m in methods):
                            qvals = q_est.support_values(u_full, idx, w)
                            for q in self.cfg.quantile.quantiles:
                                key = f"Q{q:.2f}"
                                if key in methods and np.isfinite(qvals[q]):
                                    halfspaces[key][0].append(np.array([u2[0], u2[1]]))
                                    halfspaces[key][1].append(float(qvals[q]))
                    # Bounding box halfspaces (x>=0, y>=0, x<=1, y<=1)
                    bbox_A = [
                        np.array([1.0, 0.0]),  # x <= 1
                        np.array([0.0, 1.0]),  # y <= 1
                        np.array([-1.0, 0.0]), # -x <= 0
                        np.array([0.0, -1.0]), # -y <= 0
                    ]
                    bbox_b = [1.0, 1.0, 0.0, 0.0]
                    # Plot convex polygons
                    for method in methods:
                        if method == "FDH":
                            continue
                        A_list, b_list = halfspaces.get(method, ([], []))
                        A_all = A_list + bbox_A
                        b_all = b_list + bbox_b
                        P = self._halfspace_intersection_polygon(A_all, b_all)
                        if P.shape[0] >= 3:
                            color = method_palette.get(method)
                            ls = linestyles[0]  # single boundary per subplot
                            ax.plot(
                                np.append(P[:, 0], P[0, 0]),
                                np.append(P[:, 1], P[0, 1]),
                                label=f"{method}",
                                color=color,
                                linewidth=2.0,
                                linestyle=ls,
                            )
                    # Scatter raw points for this slice (finite pairs only) with larger markers
                    if band_masks is not None:
                        mask_models = band_masks[idx_pos]
                        finite_pair = np.isfinite(self.panel.A[:, i]) & np.isfinite(self.panel.A[:, j])
                        sel = np.nonzero(mask_models & finite_pair)[0]
                        pts = self.panel.A[sel][:, [i, j]] if sel.size > 0 else np.zeros((0, 2))
                    else:
                        if len(idx) > 0:
                            fin = np.isfinite(self.panel.A[idx, i]) & np.isfinite(self.panel.A[idx, j])
                            sel = idx[fin]
                            pts = self.panel.A[sel][:, [i, j]] if sel.size > 0 else np.zeros((0, 2))
                        else:
                            pts = np.zeros((0, 2))
                    mk = slice_markers[idx_pos % len(slice_markers)]
                    point_color = (band_colors[idx_pos] if band_colors is not None else (0.27, 0.27, 0.27))
                    ax.scatter(
                        pts[:, 0],
                        pts[:, 1],
                        s=30,
                        alpha=0.55,
                        c=[point_color],
                        edgecolors="none",
                        marker=mk,
                        zorder=5,
                        label="points",
                    )
                    if pts.shape[0] > 0:
                        x_los[idx_pos] = float(np.nanmin(pts[:, 0]))
                        x_his[idx_pos] = float(np.nanmax(pts[:, 0]))
                        y_los[idx_pos] = float(np.nanmin(pts[:, 1]))
                        y_his[idx_pos] = float(np.nanmax(pts[:, 1]))
                    # Log extreme model at upper-right using FDH (best in direction (1,1))
                    try:
                        fdh_tmp = FDHEstimator(self.panel)
                        u_full = np.zeros(nT)
                        u_full[i] = 1.0
                        u_full[j] = 1.0
                        _val, best_idx = fdh_tmp.support_value(u_full, idx, w, lc0)
                        if best_idx is not None and extreme_f is not None:
                            bx = float(self.panel.A[best_idx, i])
                            by = float(self.panel.A[best_idx, j])
                            name = self.panel.models[best_idx]
                            extreme_f.write(f"{ti},{tj},{iC},{lc0:.12g},{math.exp(lc0):.12g},{name},{bx:.6g},{by:.6g}\n")
                        elif extreme_f is not None:
                            extreme_f.write(f"{ti},{tj},{iC},{lc0:.12g},{math.exp(lc0):.12g},NA,NA,NA\n")
                    except Exception:
                        pass
                # Aesthetics
                for idx_pos, iC in enumerate(C_indices):
                    ax = axes[idx_pos]
                    lc0 = float(self.grid_logC[iC])
                    C0 = float(np.exp(lc0))
                    # Dynamic limits from scatter, with padding and clamped to [0,1]
                    xlo, xhi = x_los[idx_pos], x_his[idx_pos]
                    ylo, yhi = y_los[idx_pos], y_his[idx_pos]
                    if np.isfinite(xlo) and np.isfinite(xhi):
                        xspan = max(1e-6, xhi - xlo)
                        xpad = max(0.01, 0.05 * xspan)
                        ax.set_xlim(max(0.0, xlo - xpad), min(1.0, xhi + xpad))
                    else:
                        ax.set_xlim(0.0, 1.0)
                    if np.isfinite(ylo) and np.isfinite(yhi):
                        yspan = max(1e-6, yhi - ylo)
                        ypad = max(0.01, 0.05 * yspan)
                        ax.set_ylim(max(0.0, ylo - ypad), min(1.0, yhi + ypad))
                    else:
                        ax.set_ylim(0.0, 1.0)
                    ax.set_xlabel(f"Accuracy — {ti}", fontsize=14, fontweight="bold")
                    if idx_pos == 0:
                        ax.set_ylabel(f"Accuracy — {tj}", fontsize=14, fontweight="bold")
                    subtitle = bands_label[idx_pos] if bands_label is not None else f"C ≈ {C0:.3g}"
                    ax.set_title(f"Acc. Boundaries — {ti} × {tj}  |  {subtitle}", fontsize=14, fontweight="bold")
                    # Annotate compute formula
                    comp_str = self.cfg.compute_formula or "C = exp(logC)"
                    try:
                        ax.text(0.01, 0.02, comp_str, transform=ax.transAxes, fontsize=8, alpha=0.85)
                    except Exception:
                        pass
                    # Compact legend in upper-left corner
                    handles, labels = ax.get_legend_handles_labels()
                    # Deduplicate labels while preserving order
                    uniq = {}
                    for h, l in zip(handles, labels):
                        if l not in uniq:
                            uniq[l] = h
                    if uniq:
                        leg = ax.legend(
                            list(uniq.values()),
                            list(uniq.keys()),
                            loc="upper left",
                            fontsize=8,
                            frameon=True,
                            fancybox=False,
                            framealpha=0.85,
                            borderpad=0.25,
                            labelspacing=0.2,
                            handlelength=1.0,
                            handletextpad=0.3,
                            borderaxespad=0.2,
                            markerscale=0.9,
                        )
                fig.tight_layout()
                base = os.path.join(output_dir, f"pairwise_frontier__{ti}__{tj}")
                fig.savefig(base + ".pdf", dpi=dpi, bbox_inches="tight")
                fig.savefig(base + ".png", dpi=dpi, bbox_inches="tight")
                plt.close(fig)
        # Close extreme log file if open
        if extreme_f is not None:
            try:
                extreme_f.close()
            except Exception as exc:
                logging.debug("Failed to close extreme_models log file: %s", exc)


# --------------------------------------------------------------------------------------
# CLI, example usage, and synthetic validation hooks
# --------------------------------------------------------------------------------------


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Estimate skill frontiers vs pretraining compute")
    p.add_argument("--merged_csv", type=str, required=False, default=None, help="Path to merged.csv input (Option A)")
    p.add_argument(
        "--wide_csv",
        type=str,
        required=False,
        default=None,
        help="Path to LiveBench wide CSV (livebench_subcategory_results_wide.csv) (Option B)",
    )
    p.add_argument(
        "--metadata_csv",
        type=str,
        required=False,
        default=None,
        help="Path to LiveBench numeric metadata (livebench_model_metadata_numeric.csv) for Option B",
    )
    p.add_argument(
        "--min_models_per_task",
        type=int,
        default=20,
        help="Minimum models per task to keep task (Option B)",
    )
    p.add_argument(
        "--min_values_per_row",
        type=int,
        default=5,
        help="Minimum valid task values per model row to keep (Option B)",
    )
    p.add_argument("--output_dir", type=str, required=False, default=None, help="Directory to write CSV outputs")
    p.add_argument("--num_C_grid", type=int, default=24, help="Number of logC grid points")
    p.add_argument("--num_directions", type=int, default=128, help="Number of directions on simplex")
    p.add_argument("--kernel", type=str, default="gaussian", choices=["gaussian", "epanechnikov"], help="Kernel type")
    p.add_argument("--bandwidth", type=float, default=None, help="Bandwidth in logC (None => Silverman)")
    p.add_argument("--alpha_cap_multiple", type=float, default=4.0, help="DEA alpha cap multiple vs normalized weights")
    p.add_argument("--crs", action="store_true", help="Use CRS (drop VRS sum alpha=1) for DEA")
    p.add_argument("--write_vertices", action="store_true", help="Write approximate frontier vertices CSVs")
    p.add_argument("--log_level", type=str, default="INFO", help="Logging level (DEBUG, INFO, WARNING)")
    p.add_argument("--synthetic", action="store_true", help="Run on synthetic data if no merged_csv is provided")
    # Plotting
    p.add_argument("--plot_max_per_task", action="store_true", help="Generate per-task plots of max accuracy vs compute")
    p.add_argument("--plot_points", type=int, default=200, help="Number of x points for plot interpolation grid")
    p.add_argument("--plot_dir", type=str, default=None, help="Directory to save plots (defaults to output_dir or ./outputs/plots)")
    p.add_argument("--plot_pairwise_frontier", action="store_true", help="Generate true 2D frontiers for each task pair at selected computes")
    p.add_argument("--pairwise_points", type=int, default=128, help="Number of 2D directions for frontier reconstruction")
    p.add_argument("--pairwise_C_count", type=int, default=3, help="Number of compute grid points to overlay per figure")
    # Lower frontier
    p.add_argument("--compute_lower", action="store_true", help="Also compute lower-frontier floors via sign-flip and write min_per_task.csv")
    # Robust outlier filtering
    p.add_argument("--robust_enable", action="store_true", help="Enable robust outlier filtering per compute window (MCD/MAD)")
    p.add_argument("--robust_support_fraction", type=float, default=0.9, help="MCD support fraction (default 0.9)")
    p.add_argument("--robust_alpha", type=float, default=0.005, help="Tail probability for chi-square cutoff (default 0.005)")
    p.add_argument("--robust_alpha_lower", type=float, default=0.05, help="Tail probability for lower-curve 1D filtering in plots (default 0.05)")
    p.add_argument("--robust_min_task_cov", type=float, default=0.9, help="Minimum per-model task coverage within window (default 0.9)")
    p.add_argument("--robust_scale_floor", type=float, default=0.05, help="MAD scale floor per task (default 0.05)")
    p.add_argument("--robust_action", choices=["drop", "winsorize"], default="drop", help="Outlier handling action (default drop)")
    p.add_argument("--robust_winsor_q_low", type=float, default=0.05, help="Winsorization lower quantile (if action=winsorize)")
    p.add_argument("--robust_k_bandwidth", type=float, default=3.0, help="Window half-width multiplier in bandwidth units (unused with kernels; default 3.0)")
    p.add_argument("--chance_baseline_csv", type=str, default=None, help="Optional CSV with one row of per-task chance baselines in [0,1]")
    return p


def _generate_synthetic_panel(n_models: int = 200, n_tasks: int = 3, seed: int = 0) -> ModelPanel:
    """Generate a simple synthetic dataset with sigmoidal task progress in logC.

    Each task i has a max level and steepness; models have random logC and noise.
    """
    rng = np.random.default_rng(seed)
    logC = rng.uniform(0.0, 10.0, size=n_models)
    # Task ceilings and steepness
    max_levels = np.linspace(0.7, 0.95, n_tasks)
    k = np.linspace(0.6, 1.4, n_tasks)  # steepness
    mid = np.linspace(3.0, 7.0, n_tasks)  # midpoint in logC
    A = np.zeros((n_models, n_tasks))
    for i in range(n_tasks):
        A[:, i] = max_levels[i] / (1.0 + np.exp(-k[i] * (logC - mid[i])))
    # Add small noise and clip
    A += 0.02 * rng.normal(size=A.shape)
    A = np.clip(A, 0.0, 1.0)
    models = [f"M{j:03d}" for j in range(n_models)]
    tasks = [f"a_task{i+1}" for i in range(n_tasks)]
    return ModelPanel(models=models, logC=logC, A=A, tasks=tasks)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = build_argparser().parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO), format="%(levelname)s:%(name)s:%(message)s")

    if args.merged_csv is not None:
        if not os.path.isfile(args.merged_csv):
            print(f"File not found: {args.merged_csv}")
            sys.exit(2)
        panel = load_merged_csv(args.merged_csv)
        logging.info(
            "Loaded merged CSV with %d models, %d tasks. Tasks: %s",
            panel.num_models,
            panel.num_tasks,
            ", ".join(panel.tasks),
        )
    elif args.wide_csv is not None:
        if not os.path.isfile(args.wide_csv):
            print(f"File not found: {args.wide_csv}")
            sys.exit(2)
        panel = load_livebench_wide_csv(
            args.wide_csv,
            metadata_path=args.metadata_csv,
            min_models_per_task=args.min_models_per_task,
            min_values_per_row=args.min_values_per_row,
        )
        logging.info(
            "Loaded LiveBench wide CSV with %d models, %d tasks. Tasks: %s",
            panel.num_models,
            panel.num_tasks,
            ", ".join(panel.tasks),
        )
    else:
        if not args.synthetic:
            print("Provide --merged_csv, or --wide_csv (+ optional --metadata_csv), or pass --synthetic for a demo.")
            sys.exit(2)
        panel = _generate_synthetic_panel()
        logging.info("Using synthetic panel (demo mode)")

    cfg = FrontierConfig(
        num_C_grid=args.num_C_grid,
        kernel=args.kernel,
        bandwidth=args.bandwidth,
        num_directions=args.num_directions,
        dea=DEAConfig(var_returns_to_scale=(not args.crs), alpha_cap_multiple=args.alpha_cap_multiple),
        write_vertices=args.write_vertices,
    )
    if args.compute_lower:
        cfg.compute_lower = True
    # Robust options
    if args.robust_enable:
        cfg.robust_enable = True
        cfg.robust_support_fraction = args.robust_support_fraction
        cfg.robust_alpha = args.robust_alpha
        cfg.robust_alpha_lower = args.robust_alpha_lower
        cfg.robust_min_task_cov = args.robust_min_task_cov
        cfg.robust_scale_floor = args.robust_scale_floor
        cfg.robust_action = args.robust_action
        cfg.robust_winsor_q_low = args.robust_winsor_q_low
        cfg.robust_k_bandwidth = args.robust_k_bandwidth
        # Optional baseline CSV: expect a single headerless line with per-task baselines aligned to panel.tasks
        if args.chance_baseline_csv is not None:
            try:
                with open(args.chance_baseline_csv, "r") as bf:
                    line = bf.readline().strip()
                vals = [float(x) for x in line.split(',') if x.strip() != ""]
                cfg.chance_baseline = vals
            except Exception as exc:
                logging.debug("Ignoring invalid chance baseline CSV '%s': %s", args.chance_baseline_csv, exc)
    runner = SkillFrontier(panel, cfg)
    results = runner.run(output_dir=args.output_dir)
    # Small summary
    grid_logC = results["grid_logC"]
    supports = results["supports"]
    logging.info("Computed support functions for methods: %s", ", ".join(supports.keys()))
    logging.info("Grid logC range: [%.3f, %.3f] with %d points", float(grid_logC.min()), float(grid_logC.max()), len(grid_logC))
    if args.output_dir:
        logging.info("Wrote outputs to %s", args.output_dir)
    # Optional plotting
    if args.plot_max_per_task:
        plot_dir = args.plot_dir or (os.path.join(args.output_dir, "figures") if args.output_dir else os.path.join("outputs", "plots"))
        runner.plot_max_per_task(output_dir=plot_dir, num_points=args.plot_points)
        logging.info("Saved max-per-task plots to %s", plot_dir)
    if args.plot_pairwise_frontier:
        plot_dir = args.plot_dir or (os.path.join(args.output_dir, "frontiers_pairwise") if args.output_dir else os.path.join("outputs", "frontiers_pairwise"))
        runner.plot_pairwise_frontiers(output_dir=plot_dir, pairwise_points=args.pairwise_points, C_count=args.pairwise_C_count)
        logging.info("Saved pairwise frontier plots to %s", plot_dir)


if __name__ == "__main__":  # pragma: no cover
    main()

#!/usr/bin/env python3
"""
Budget-only one-shot design for a single-skill sigmoid frontier.

Implements a K-independent, budget-constrained selection of models to evaluate.
Given a candidate pool with compute C (ZFLOPs), we work in z=log10(C); each
evaluation costs C (or 10**z). We choose a subset S maximizing D-optimal
information under sum_{i in S} C_i <= U, using a greedy per-unit-cost rule.

Key pieces:
- FrontierParams, sigmoid frontier and Jacobian row j(z;theta0).
- Information matrix M(S) = sum w(z) j j^T, with optional heteroscedastic weights.
- Rank-one logdet and inverse updates to score marginal gains efficiently:
    Δ = log(1 + w * j^T Minv j)

Dependencies: numpy only.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Callable, Tuple
import numpy as np

try:
    # Used only for constructing I-optimal predictive-variance grids and bins
    from skill_frontier.evaluation.binning import create_equal_mass_bins  # type: ignore
except Exception:  # pragma: no cover
    create_equal_mass_bins = None  # type: ignore


def _sigmoid(u: float) -> float:
    return 1.0 / (1.0 + np.exp(-u))


@dataclass
class FrontierParams:
    y0: float
    L: float
    a: float
    b: float


def jacobian_row(z: float, theta: FrontierParams) -> np.ndarray:
    s = _sigmoid(theta.a + theta.b * z)
    return np.array([
        1.0,
        s,
        theta.L * s * (1.0 - s),
        theta.L * s * (1.0 - s) * z,
    ], dtype=float)


@dataclass
class Weighting:
    """Optional heteroscedastic weights.

    weight(z) = 1 / noise_var(z)
    noise models: 'constant' or 'binomial' with m items and p≈frontier.
    """
    mode: str = "constant"
    m: int = 100

    def noise_var(self, z: float, theta: FrontierParams) -> float:
        if self.mode == "binomial":
            p = theta.y0 + theta.L * _sigmoid(theta.a + theta.b * z)
            p = float(np.clip(p, 1e-6, 1 - 1e-6))
            return p * (1.0 - p) / max(self.m, 1)
        return 1.0

    def weight(self, z: float, theta: FrontierParams) -> float:
        nv = self.noise_var(z, theta)
        return 1.0 / nv


@dataclass
class InfoState:
    M: np.ndarray       # 4x4
    Minv: np.ndarray    # 4x4
    logdetM: float


def _init_info_state(eps: float = 1e-9) -> InfoState:
    M = eps * np.eye(4, dtype=float)
    Minv = (1.0 / eps) * np.eye(4, dtype=float)
    logdetM = 4.0 * np.log(eps)
    return InfoState(M=M, Minv=Minv, logdetM=float(logdetM))


def _rank_one_update(state: InfoState, j: np.ndarray, w: float) -> InfoState:
    """Return new InfoState after M <- M + w * j j^T using Sherman-Morrison.

    Δlog|M| = log(1 + w * j^T Minv j)
    Minv_new = Minv - (Minv u u^T Minv) / (1 + u^T Minv u), u = sqrt(w) * j
    """
    Minv = state.Minv
    u = np.sqrt(max(w, 0.0)) * j.reshape(-1, 1)  # (4,1)
    denom = float(1.0 + (u.T @ Minv @ u)[0, 0])
    if denom <= 1e-15:
        # Degenerate; return numerically stable-ish no-op
        return state
    Minv_u = Minv @ u
    Minv_new = Minv - (Minv_u @ Minv_u.T) / denom
    logdet_new = state.logdetM + float(np.log(denom))
    # We do not need M explicitly for scoring; keep Minv+logdet only
    return InfoState(M=state.M, Minv=Minv_new, logdetM=logdet_new)


def _marginal_gain_per_cost(state: InfoState, j: np.ndarray, w: float, cost: float) -> Tuple[float, float]:
    """Return (gain, denom) where gain = Δlogdet / cost and denom = 1 + w * j^T Minv j."""
    if cost <= 0:
        return (-np.inf, 1.0)
    Minv = state.Minv
    val = float((j.reshape(1, -1) @ Minv @ j.reshape(-1, 1))[0, 0])
    denom = float(1.0 + max(w, 0.0) * val)
    dlog = float(np.log(denom)) if denom > 0 else -np.inf
    return (dlog / cost, denom)


def _build_i_optimal_grid(
    z: np.ndarray,
    num_bins: int = 10,
    min_bin_size: int = 1,
) -> Tuple[np.ndarray, np.ndarray]:
    """Construct a z-grid and weights for I-optimal predictive-variance design.

    Uses the same equal-mass binning logic as evaluation (if available), with
    bin midpoints as grid points and uniform weights over bins.
    """
    z = np.asarray(z, float)
    z = z[np.isfinite(z)]
    if z.size == 0 or create_equal_mass_bins is None:
        return np.array([], dtype=float), np.array([], dtype=float)
    try:
        edges = create_equal_mass_bins(z, int(max(1, num_bins)), int(max(1, min_bin_size)))  # type: ignore[arg-type]
    except Exception:
        return np.array([], dtype=float), np.array([], dtype=float)
    if edges.size < 2:
        return np.array([], dtype=float), np.array([], dtype=float)
    mids = 0.5 * (edges[:-1] + edges[1:])
    if mids.size == 0:
        return np.array([], dtype=float), np.array([], dtype=float)
    w = np.full(mids.shape, 1.0 / float(mids.size), dtype=float)
    return mids.astype(float), w.astype(float)


def _build_A_matrix(
    z_grid: np.ndarray,
    w_grid: np.ndarray,
    theta0: FrontierParams,
) -> np.ndarray:
    """Build A = sum_k w_k j(z_k)^T j(z_k) used in I-optimal predictive variance.

    Note: We intentionally do *not* include measurement weights here; those
    already enter through M(S) via Weighting.weight and thus Sigma.
    """
    p = 4
    A = np.zeros((p, p), dtype=float)
    for z_k, w_k in zip(z_grid, w_grid):
        j_k = jacobian_row(float(z_k), theta0).reshape(-1)
        A += float(w_k) * np.outer(j_k, j_k)
    return A


def _assign_bins(
    z: np.ndarray,
    num_bins: int = 10,
    min_bin_size: int = 1,
) -> Tuple[np.ndarray, int]:
    """Assign each z to an equal-mass bin, reusing evaluation binning logic.

    Returns:
        bin_index: shape (N,) with values in [0, B-1] (or -1 for invalid),
        num_bins:  number of bins B.
    """
    z = np.asarray(z, float)
    n = int(z.size)
    bin_index = np.full(n, -1, dtype=int)
    if n == 0 or create_equal_mass_bins is None:
        return bin_index, 0
    z_finite = z[np.isfinite(z)]
    if z_finite.size == 0:
        return bin_index, 0
    try:
        edges = create_equal_mass_bins(  # type: ignore[arg-type]
            z_finite, int(max(1, num_bins)), int(max(1, min_bin_size))
        )
    except Exception:
        return bin_index, 0
    if edges.size < 2:
        return bin_index, 0
    B = int(edges.size - 1)
    for i, zi in enumerate(z):
        if not np.isfinite(zi):
            continue
        # Bins [edges[b], edges[b+1]) except last inclusive, as in compute_bin_statistics.
        b = int(np.searchsorted(edges, float(zi), side="right") - 1)
        if b < 0:
            b = 0
        if b >= B:
            b = B - 1
        bin_index[i] = b
    return bin_index, B


def design_budget_only(
    z_pool: np.ndarray,
    C_pool: np.ndarray,
    budget: float,
    theta0: FrontierParams,
    weighting: Optional[Weighting] = None,
    seed_c: float = 1.543,
    exchange_passes: int = 0,
    repulsion: float = 0.0,
    objective: str = "d_optimal",
    balance_lambda: float = 0.0,
    num_bins: Optional[int] = None,
    min_bin_size: Optional[int] = None,
) -> List[int]:
    """Return indices (distinct) selected under sum C <= budget.

    - Seed with up to four points: [zmin, zmax, z*±c/b] snapped to nearest unique indices.
    - Greedy add: at each step, among remaining candidates that fit the residual
      budget, pick argmax Δlog|M| per cost; stop when no feasible positive gain.
    """
    z = np.asarray(z_pool, float)
    C = np.asarray(C_pool, float)
    n = int(z.size)
    assert C.shape == z.shape
    if budget <= 0 or n == 0:
        return []
    wmodel = weighting or Weighting(mode="constant")

    # Bin parameters for I-optimal grid and balanced design. Defaults preserve
    # previous behavior (10 bins, effectively no min-bin constraint) if not
    # provided by the caller.
    nb_eff = int(num_bins) if num_bins is not None and num_bins > 0 else 10
    mb_eff = int(min_bin_size) if min_bin_size is not None and min_bin_size > 0 else 1

    # Fixed RNG for reproducible but randomized tie-breaking
    rng = np.random.default_rng(42)

    objective_key = str(objective).lower()

    # Precompute per-candidate (j(z), w(z))
    J = np.zeros((n, 4), dtype=float)
    W = np.zeros(n, dtype=float)
    for i in range(n):
        J[i, :] = jacobian_row(float(z[i]), theta0)
        W[i] = float(wmodel.weight(float(z[i]), theta0))

    # If using I-optimal predictive-variance, build A and z-grid
    A = None
    if objective_key in ("i_optimal", "i_optimal_predvar", "i_optimal_predvar_balanced"):
        z_grid, w_grid = _build_i_optimal_grid(z, num_bins=nb_eff, min_bin_size=mb_eff)
        if z_grid.size > 0:
            A = _build_A_matrix(z_grid, w_grid, theta0)
        else:
            # Fallback to D-optimal if we cannot build a sensible grid
            objective_key = "d_optimal"

    # If using bin-balanced I-optimal, construct bin assignments
    balanced = objective_key in ("i_optimal_predvar_balanced", "i_optimal_balanced") and balance_lambda > 0.0
    bin_index = None
    bin_counts = None
    if balanced:
        bin_index, num_bins_eff = _assign_bins(z, num_bins=nb_eff, min_bin_size=mb_eff)
        if num_bins_eff <= 0:
            balanced = False
        else:
            bin_counts = np.zeros(num_bins_eff, dtype=int)

    selected: List[int] = []
    used = np.zeros(n, dtype=bool)
    cost_used = 0.0
    state = _init_info_state()

    # Seed points: extremes and around z* = -a/b
    b = max(theta0.b, 1e-8)
    zmin = float(np.min(z))
    zmax = float(np.max(z))
    zstar = -theta0.a / b
    targets = [zmin, zmax, zstar - seed_c / b, zstar + seed_c / b]
    # Snap in order while respecting uniqueness and budget; break ties randomly
    base_perm = rng.permutation(n)
    for t in targets:
        dist = np.abs(z - t)
        idx_order = np.lexsort((base_perm, dist))
        for i in idx_order:
            if used[i]:
                continue
            if repulsion > 0.0 and any(abs(z[i] - z[j]) < repulsion for j in selected):
                continue
            if cost_used + C[i] > budget:
                continue
            # accept
            used[i] = True
            selected.append(int(i))
            cost_used += float(C[i])
            state = _rank_one_update(state, J[i, :], W[i])
            if balanced and bin_index is not None and bin_counts is not None:
                b_idx = int(bin_index[i])
                if b_idx >= 0:
                    bin_counts[b_idx] += 1
            break

    # Greedy additions
    # For I-optimal, we track predictive-variance objective via A and Minv.
    if objective_key in ("i_optimal", "i_optimal_predvar", "i_optimal_predvar_balanced") and A is not None:
        phi_pred = -float(np.trace(A @ state.Minv))
    else:
        phi_pred = 0.0

    while True:
        resid = float(budget - cost_used)
        if resid <= 0:
            break
        best_idx = -1
        best_score = 0.0
        best_denom = 1.0
        # Score remaining candidates
        if objective_key == "d_optimal":
            # Randomize candidate scan order for tie-breaking
            for i in rng.permutation(n):
                if used[i] or C[i] > resid:
                    continue
                if repulsion > 0.0 and any(abs(z[i] - z[j]) < repulsion for j in selected):
                    continue
                score, denom = _marginal_gain_per_cost(state, J[i, :], W[i], float(C[i]))
                if score > best_score:
                    best_score = score
                    best_idx = i
                    best_denom = denom
        else:
            # I-optimal predictive-variance (optionally bin-balanced) greedy step
            # Current predictive-variance objective
            phi_cur = -float(np.trace(A @ state.Minv))  # type: ignore[arg-type]
            eps_bal = 1e-6
            # Randomize candidate scan order for tie-breaking
            for i in rng.permutation(n):
                if used[i] or C[i] > resid:
                    continue
                if repulsion > 0.0 and any(abs(z[i] - z[j]) < repulsion for j in selected):
                    continue
                cost_i = float(C[i])
                if cost_i <= 0.0:
                    continue
                # Prospective Sherman–Morrison update
                Minv = state.Minv
                u = np.sqrt(max(W[i], 0.0)) * J[i, :].reshape(-1, 1)
                v = Minv @ u
                denom = float(1.0 + (u.T @ v)[0, 0])
                if denom <= 1e-15:
                    continue
                Minv_new = Minv - (v @ v.T) / denom
                phi_new = -float(np.trace(A @ Minv_new))  # type: ignore[arg-type]
                gain_info = phi_new - phi_cur

                gain_total = gain_info
                if balanced and bin_index is not None and bin_counts is not None:
                    b_idx = int(bin_index[i])
                    if b_idx >= 0:
                        n_b = int(bin_counts[b_idx])
                        delta_bal = float(
                            np.log(n_b + 1.0 + eps_bal) - np.log(n_b + eps_bal)
                        )
                        gain_total = gain_info + float(balance_lambda) * delta_bal

                score = gain_total / cost_i
                if score > best_score:
                    best_score = score
                    best_idx = i
                    best_denom = denom
        if best_idx < 0 or best_score <= 0:
            break
        # Commit best add
        used[best_idx] = True
        selected.append(int(best_idx))
        cost_used += float(C[best_idx])
        state = _rank_one_update(state, J[best_idx, :], W[best_idx])
        if balanced and bin_index is not None and bin_counts is not None:
            b_idx = int(bin_index[best_idx])
            if b_idx >= 0:
                bin_counts[b_idx] += 1
        if objective_key in ("i_optimal", "i_optimal_predvar", "i_optimal_predvar_balanced") and A is not None:
            phi_pred = -float(np.trace(A @ state.Minv))

    # Optional 1-exchange polishing: try delete/add swaps under budget (D-optimal only)
    if exchange_passes and selected and objective_key == "d_optimal":
        selected = _polish_exchange_1(
            z, C, J, W, selected, float(budget), passes=int(max(1, exchange_passes))
        )

    return selected


def _build_info_state(J: np.ndarray, W: np.ndarray, sel: List[int]) -> InfoState:
    state = _init_info_state()
    for i in sel:
        state = _rank_one_update(state, J[int(i), :], float(W[int(i)]))
    return state


def _logdet_for_set(J: np.ndarray, W: np.ndarray, sel: List[int]) -> float:
    st = _build_info_state(J, W, sel)
    return float(st.logdetM)


def _polish_exchange_1(
    z: np.ndarray,
    C: np.ndarray,
    J: np.ndarray,
    W: np.ndarray,
    selected: List[int],
    budget: float,
    passes: int = 1,
    eps_improve: float = 1e-12,
) -> List[int]:
    """Simple Fedorov 1-exchange polish under a hard budget.

    For a few passes: for each selected i, try replacing it with the best
    non-selected j that fits the residual budget and improves logdet.
    """
    sel: List[int] = [int(i) for i in selected]
    used = np.zeros(J.shape[0], dtype=bool)
    used[sel] = True
    total_cost = float(np.sum(C[sel]))
    for _ in range(max(1, passes)):
        improved = False
        # Precompute current logdet
        cur_logdet = _logdet_for_set(J, W, sel)
        for pos in range(len(sel)):
            i = int(sel[pos])
            # Build state excluding i
            sel_minus = sel[:pos] + sel[pos+1:]
            st_minus = _build_info_state(J, W, sel_minus)
            logdet_minus = float(st_minus.logdetM)
            # Removal penalty
            delta_remove = float(cur_logdet - logdet_minus)
            # Residual budget if we remove i
            resid = float(budget - (total_cost - float(C[i])))
            if resid <= 0:
                continue
            # Scan candidates that fit
            best_j = -1
            best_gain = 0.0
            Minv = st_minus.Minv
            for j in range(J.shape[0]):
                if used[j]:
                    continue
                if C[j] > resid:
                    continue
                # gain from adding j on top of sel_minus
                jj = J[j, :].reshape(-1, 1)
                val = float((jj.T @ Minv @ jj)[0, 0])
                denom = 1.0 + float(max(W[j], 0.0)) * val
                if denom <= 0:
                    continue
                dlog = float(np.log(denom))
                if dlog > best_gain:
                    best_gain = dlog
                    best_j = j
            # Commit swap if net improvement
            if best_j >= 0 and (best_gain - delta_remove) > eps_improve:
                used[i] = False
                used[best_j] = True
                total_cost = total_cost - float(C[i]) + float(C[best_j])
                sel[pos] = int(best_j)
                # Update baseline logdet
                cur_logdet = cur_logdet - delta_remove + best_gain
                improved = True
        if not improved:
            break
    return sel

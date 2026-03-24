#!/usr/bin/env python3
"""
Monotone envelope fitting for 2D scatter (upper/lower + trade‑off isoquant).

Implements the blueprint:
  - Smooth monotone upper/lower envelopes via I‑spline basis with sign‑weighted
    least squares (IRLS) and nonnegative coefficients (monotone increasing in x).
  - A smooth, monotone isoquant trade‑off curve y_h(x) from an L_p aggregator.
  - A combined upper envelope min(g(x), y_h(x)) with an explicit elbow.

Dependencies: numpy; optional SciPy (BSpline) for smooth splines; matplotlib (for optional plotting).
If SciPy is unavailable, the fitter falls back to an equal‑mass piecewise
quantile with isotonic smoothing (still monotone, less smooth).

All public predictors accept raw x and return raw y; internally, data are robust‑scaled
by median/IQR. This file is self‑contained and does not modify other repo code.
"""

from __future__ import annotations

import dataclasses
import math
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np

# Note: Previous versions relied on cvxpy. The current implementation uses
# IRLS + NNLS and does not require cvxpy. Dependency removed for simplicity.

def _get_BSpline():
    """Return scipy.interpolate.BSpline if available, else None.

    This makes SciPy optional: callers can branch to a simpler fallback when
    the spline basis is not available.
    """
    try:  # pragma: no cover - optional dependency
        from scipy.interpolate import BSpline  # type: ignore
        return BSpline
    except Exception:
        return None


# --------------------------------------------------------------------------------------
# Robust scaling utilities
# --------------------------------------------------------------------------------------


@dataclasses.dataclass
class RobustScaler1D:
    median: float
    iqr: float

    @staticmethod
    def fit(x: np.ndarray) -> "RobustScaler1D":
        x = np.asarray(x, dtype=float)
        m = float(np.nanmedian(x))
        q25 = float(np.nanpercentile(x, 25))
        q75 = float(np.nanpercentile(x, 75))
        iqr = q75 - q25
        if not np.isfinite(iqr) or abs(iqr) < 1e-8:
            # Fallback to MAD or std to avoid degeneracy
            mad = float(np.nanmedian(np.abs(x - m)))
            iqr = 1.4826 * mad
            if not np.isfinite(iqr) or abs(iqr) < 1e-8:
                s = float(np.nanstd(x))
                iqr = s if s > 0 else 1.0
        return RobustScaler1D(median=m, iqr=float(iqr))

    def transform(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        return (x - self.median) / float(self.iqr)

    def inverse(self, z: np.ndarray) -> np.ndarray:
        z = np.asarray(z, dtype=float)
        return z * float(self.iqr) + self.median


# --------------------------------------------------------------------------------------
# I-spline / M-spline basis (via BSpline normalization + antiderivative)
# --------------------------------------------------------------------------------------


def _augmented_knots(x: np.ndarray, degree: int, K: int) -> np.ndarray:
    """Construct an open knot vector with interior knots adapted to data.

    Requirements implemented:
      - Intervals should split the sample into (approximately) equal-mass subsets.
      - Intervals must be non-degenerate (strictly increasing edges with width>eps).

    Implementation details:
      - Start from r_target = K interior knots, capped by sqrt(n) and unique_x-1.
      - Build r+1 quantile edges at j/(r+1); if any consecutive edges collide (<= eps),
        decrease r until all edges are strictly increasing.
      - Return open knot vector with boundaries repeated degree+1 times.
    """
    x = np.asarray(x, dtype=float)
    degree = int(degree)
    K = max(0, int(K))
    n = int(x.size)
    if n == 0:
        # fallback safe knots
        return np.array([0, 0, 0, 0, 1, 1, 1, 1], dtype=float)  # degree=3 default shape
    xmin = float(np.nanmin(x))
    xmax = float(np.nanmax(x))
    if not np.isfinite(xmin) or not np.isfinite(xmax) or xmin == xmax:
        xmin, xmax = (0.0, 1.0)
    # Unique x guard
    xu = np.unique(x[np.isfinite(x)])
    unique_x = int(xu.size)
    # Target number of interior knots r
    r_cap = max(0, unique_x - 1)  # cannot split into more non-empty intervals than unique gaps
    r_target = min(K, max(0, int(np.sqrt(max(n, 1))) - 1), r_cap)
    eps = 1e-10

    def try_build(r: int) -> np.ndarray:
        if r <= 0:
            return np.array([], dtype=float)
        # Equal-mass edges including boundaries
        q_edges = np.linspace(0.0, 1.0, num=r + 2)
        edges = np.quantile(x, q_edges)
        # Ensure strictly increasing edges by removing collisions
        diffs = np.diff(edges)
        if np.all(diffs > eps):
            return edges[1:-1].astype(float)
        # If collisions exist, fail (caller will reduce r)
        return np.array([], dtype=float)

    r = r_target
    interior = try_build(r)
    while r > 0 and interior.size != r:
        r -= 1
        interior = try_build(r)
    # Build open knot vector
    t = np.concatenate([
        np.repeat(xmin, degree + 1),
        interior,
        np.repeat(xmax, degree + 1),
    ]).astype(float)
    return t


def i_m_spline_design(
    x: np.ndarray, knots: np.ndarray, degree: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return I-spline and M-spline design matrices evaluated at x.

    Uses the identity that an M-spline of degree d is a B-spline B_{i,d}
    normalized to have unit integral over its support. An I-spline is the
    antiderivative of the M-spline, shifted so I(x_min)=0 and I(x_max)=1.

    Returns:
        I: (n, p) matrix of I-spline basis values
        M: (n, p) matrix of M-spline basis values (g'(x) = M @ theta)
        Mprime: (n, p) matrix of d/dx M-spline (for diagnostics; g'' = Mprime @ theta)
    """
    x = np.asarray(x, dtype=float)
    BSpline = _get_BSpline()
    if BSpline is None:
        raise RuntimeError("SciPy BSpline unavailable; cannot build spline design. Use fallback path.")
    t = np.asarray(knots, dtype=float)
    n = x.shape[0]
    p = len(t) - degree - 1  # number of basis functions
    if p <= 0:
        raise ValueError("Invalid knot/degree configuration: no basis functions")

    I = np.zeros((n, p), dtype=float)
    M = np.zeros((n, p), dtype=float)
    Mp = np.zeros((n, p), dtype=float)
    L = float(t[0])
    # Loop over basis elements; p is small (<= ~K+degree)
    for i in range(p):
        # Build the i-th B-spline basis using the full knot vector.
        #
        # IMPORTANT: BSpline.basis_element(t_piece, extrapolate=False) returns NaNs
        # outside its local support in older SciPy versions. Using the full-knot
        # representation ensures values are 0 outside support (within the global
        # knot range), which is what we want for a design matrix.
        c = np.zeros(p, dtype=float)
        c[i] = 1.0
        b_i = BSpline(t, c, degree, extrapolate=False)
        # Guard: derivative may fail for repeats; compute inside try/except.
        try:
            db_i = b_i.derivative(nu=1)
        except Exception:
            db_i = None
        ab_i = b_i.antiderivative(nu=1)
        denom = float(t[i + degree + 1] - t[i])
        denom = denom if denom > 1e-12 else 1e-12
        factor = float(degree + 1) / denom
        # Evaluate
        bi_x = np.nan_to_num(b_i(x), nan=0.0, posinf=0.0, neginf=0.0)
        dbi_x = np.nan_to_num(db_i(x), nan=0.0, posinf=0.0, neginf=0.0) if db_i is not None else np.zeros_like(bi_x)
        abi_x = np.nan_to_num(ab_i(x), nan=0.0, posinf=0.0, neginf=0.0)
        abi_L = float(ab_i(L))
        # M-spline and I-spline basis columns
        M[:, i] = factor * bi_x
        Mp[:, i] = factor * dbi_x
        Ii = factor * (abi_x - abi_L)
        # Numerical clipping to [0,1]
        Ii = np.clip(Ii, 0.0, 1.0)
        I[:, i] = Ii
    return I, M, Mp


def _report_knot_intervals(x: np.ndarray, knots: np.ndarray, degree: int, label: str = "") -> None:
    """Silenced diagnostics (no-op).

    The project previously printed interval coverage. Per request, we now
    suppress all routine prints; only fallback events are reported elsewhere.
    """
    return None


def _second_difference_matrix(p: int) -> np.ndarray:
    """(p-2) x p second-difference operator for curvature control."""
    if p < 3:
        return np.zeros((0, p), dtype=float)
    D = np.zeros((p - 2, p), dtype=float)
    for i in range(p - 2):
        D[i, i] = 1.0
        D[i, i + 1] = -2.0
        D[i, i + 2] = 1.0
    return D


# --------------------------------------------------------------------------------------
# Monotone quantile spline fitting
# --------------------------------------------------------------------------------------


@dataclasses.dataclass
class SplineModel:
    beta0: float
    theta: np.ndarray
    knots: np.ndarray
    degree: int
    tau: float
    scaler_x: RobustScaler1D
    scaler_y: RobustScaler1D

    def predict(self, x_raw: np.ndarray) -> np.ndarray:
        """Predict y (raw units) for raw x.

        y_hat = beta0 + I(x_scaled) @ theta, then inverse-scale to raw y.
        """
        xs = self.scaler_x.transform(np.asarray(x_raw, dtype=float))
        I, _, _ = i_m_spline_design(xs, self.knots, self.degree)
        ys = float(self.beta0) + I @ self.theta
        return self.scaler_y.inverse(ys)

    # Diagnostics helpers
    def derivatives(self, x_raw: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Return g'(x) and g''(x) in scaled space for diagnostics."""
        xs = self.scaler_x.transform(np.asarray(x_raw, dtype=float))
        _, M, Mp = i_m_spline_design(xs, self.knots, self.degree)
        g1 = M @ self.theta
        g2 = Mp @ self.theta
        return g1, g2


# (removed) _quantile_loss was used only in an earlier cvxpy formulation.


def _train_valid_split_by_quantiles(x: np.ndarray, valid_frac: float = 0.2, bins: int = 10, seed: int = 0) -> Tuple[np.ndarray, np.ndarray]:
    """Stratified split along x-quantiles for stable coverage evaluation."""
    rng = np.random.default_rng(seed)
    x = np.asarray(x, dtype=float)
    n = len(x)
    q = np.linspace(0, 1, bins + 1)
    edges = np.quantile(x, q)
    edges[0] -= 1e-9
    edges[-1] += 1e-9
    train_mask = np.ones(n, dtype=bool)
    for i in range(bins):
        lo, hi = edges[i], edges[i + 1]
        idx = np.nonzero((x >= lo) & (x <= hi))[0]
        if idx.size == 0:
            continue
        k = max(1, int(round(valid_frac * idx.size)))
        choose = rng.choice(idx, size=k, replace=False)
        train_mask[choose] = False
    train_idx = np.nonzero(train_mask)[0]
    valid_idx = np.nonzero(~train_mask)[0]
    if valid_idx.size == 0:
        # Fallback: last 20%
        s = int(math.floor(0.8 * n))
        train_idx = np.arange(0, s)
        valid_idx = np.arange(s, n)
    return train_idx, valid_idx


def fit_monotone_quantile_spline(
    x_raw: np.ndarray,
    y_raw: np.ndarray,
    tau: float,
    K: int = 20,
    degree: int = 3,
    lam_grid: Sequence[float] = (1e-4, 1e-3, 1e-2, 1e-1),
    valid_frac: float = 0.2,
    seed: int = 0,
    fit_label: Optional[str] = None,
) -> SplineModel:
    """Fit monotone increasing quantile spline y ≈ g(x) under I‑spline basis.

    - Robust-scales x and y by median/IQR internally.
    - Enforces monotonicity via theta >= 0.
    - Selects lambda by validation coverage error |cov - tau| with a small
      smoothness tie-break using mean(g''^2) on a grid.
    """
    x_raw = np.asarray(x_raw, dtype=float)
    y_raw = np.asarray(y_raw, dtype=float)
    mask = np.isfinite(x_raw) & np.isfinite(y_raw)
    if not np.any(mask):
        raise ValueError("No finite (x,y) pairs")
    x_raw = x_raw[mask]
    y_raw = y_raw[mask]

    sx = RobustScaler1D.fit(x_raw)
    sy = RobustScaler1D.fit(y_raw)
    x = sx.transform(x_raw)
    y = sy.transform(y_raw)

    # Knots and basis
    K_eff = int(min(K, max(1, int(np.sqrt(len(x))))))
    knots = _augmented_knots(x, degree=degree, K=K_eff)
    # Report interval diagnostics for transparency
    _report_knot_intervals(x, knots, degree=degree, label=f"tau={tau:.2f}")
    BS = _get_BSpline()
    if BS is None:
        # Fallback to piecewise monotone model if SciPy BSpline is unavailable
        return _make_piecewise_model(x, y, knots, tau, sx, sy)
    I, M, Mp = i_m_spline_design(x, knots, degree)
    p = I.shape[1]
    D2 = _second_difference_matrix(p)

    tr_idx, va_idx = _train_valid_split_by_quantiles(x, valid_frac=valid_frac, bins=10, seed=seed)
    Xtr_I, ytr = I[tr_idx], y[tr_idx]
    Xva_I, yva = I[va_idx], y[va_idx]

    best = None
    best_lam = None
    best_score = float("inf")

    # Helper: projected NNLS solver on augmented normal equations
    def _proj_nnls(A_aug: np.ndarray, b_aug: np.ndarray, pdim: int, ridge_diag: float = 1e-8) -> Tuple[float, np.ndarray]:
        # Columns: [intercept | theta (pdim)]
        ncols = 1 + pdim
        active = np.ones(pdim, dtype=bool)
        b0 = 0.0
        th = np.zeros(pdim, dtype=float)
        for _ in range(25):
            # Build subset of columns
            cols = np.concatenate([np.array([0]), 1 + np.nonzero(active)[0]])
            A_sub = A_aug[:, cols]
            # Normal equations with ridge
            M = A_sub.T @ A_sub
            M.flat[:: M.shape[0] + 1] += ridge_diag
            rhs = A_sub.T @ b_aug
            try:
                z_sub = np.linalg.solve(M, rhs)
            except Exception:
                # escalate ridge and retry
                M.flat[:: M.shape[0] + 1] += 1e-6
                z_sub = np.linalg.lstsq(M, rhs, rcond=None)[0]
            # Map back
            z_full = np.zeros(ncols, dtype=float)
            z_full[cols] = z_sub
            b0_new = float(z_full[0])
            th_new = z_full[1:]
            # Enforce nonnegativity by deactivating negative thetas
            neg = th_new < -1e-10
            if np.any(neg):
                active[neg] = False
                # If all deactivated, keep zeros
                b0, th = b0_new, np.maximum(th_new, 0.0)
                continue
            b0, th = b0_new, np.maximum(th_new, 0.0)
            break
        return b0, th

    # Primary solver: IRLS + projected NNLS (expectile P-spline)
    last_pair = None
    for lam in lam_grid:
        lam_eff = max(float(lam), 1e-8)
        b0 = float(np.nanmedian(ytr))
        th = np.zeros(p, dtype=float)
        for _ in range(12):
            r = ytr - (b0 + Xtr_I @ th)
            w = np.where(r >= 0, float(tau), float(1.0 - tau))
            sqrtw = np.sqrt(w)
            A_data = np.hstack([np.ones((Xtr_I.shape[0], 1)), Xtr_I])
            A_w = (sqrtw.reshape(-1, 1)) * A_data
            b_w = sqrtw * ytr
            P = np.hstack([np.zeros((D2.shape[0], 1)), D2])
            A_aug = np.vstack([A_w, np.sqrt(lam_eff) * P])
            b_aug = np.concatenate([b_w, np.zeros(D2.shape[0], dtype=float)])
            b0_new, th_new = _proj_nnls(A_aug, b_aug, p)
            if np.allclose(th, th_new, rtol=1e-5, atol=1e-6) and abs(b0_new - b0) < 1e-6:
                b0, th = b0_new, th_new
                break
            b0, th = b0_new, th_new
        last_pair = (b0, th, lam)
        # Validation coverage + curvature tie-break
        yhat_va = b0 + Xva_I @ th
        cov = float(np.mean(yva <= yhat_va)) if yva.size > 0 else float("nan")
        cov_err = abs(cov - tau) if np.isfinite(cov) else 1.0
        # Curvature penalty on grid in scaled x
        xs_grid = np.linspace(float(np.min(x)), float(np.max(x)), num=100)
        _, _, Mp_grid = i_m_spline_design(xs_grid, knots, degree)
        gpp = Mp_grid @ th
        curv = float(np.mean(np.square(gpp)))
        score = cov_err + 0.01 * curv
        if score < best_score:
            best_score = score
            best = (b0, th)
            best_lam = lam

    if best is None:
        # Use the last IRLS solution as a conservative default to avoid fallback
        if last_pair is not None:
            b0, th, best_lam = last_pair
            best = (b0, th)
        else:
        # Fallback: piecewise quantile with isotonic smoothing (no solver)
            try:
                n = int(len(x))
                xmn = float(np.nanmin(x)) if n else float('nan')
                xmx = float(np.nanmax(x)) if n else float('nan')
                lbl = fit_label or ""
                print(f"[ENVELOPE] fallback=True tau={tau:.2f} n={n} x_range=[{xmn:.6g},{xmx:.6g}] label=<{lbl}>")
            except Exception:
                pass
        return _make_piecewise_model(x, y, knots, tau, sx, sy)

    # Refit on the full set with the chosen lambda using IRLS/NNLS
    lam = float(best_lam if best_lam is not None else 1e-2)
    b0 = float(np.nanmedian(y))
    th = np.zeros(p, dtype=float)
    lam_eff = max(float(lam), 1e-8)
    for _ in range(16):
        r = y - (b0 + I @ th)
        w = np.where(r >= 0, float(tau), float(1.0 - tau))
        sqrtw = np.sqrt(w)
        A_data = np.hstack([np.ones((I.shape[0], 1)), I])
        A_w = (sqrtw.reshape(-1, 1)) * A_data
        b_w = sqrtw * y
        P = np.hstack([np.zeros((D2.shape[0], 1)), D2])
        A_aug = np.vstack([A_w, np.sqrt(lam_eff) * P])
        b_aug = np.concatenate([b_w, np.zeros(D2.shape[0], dtype=float)])
        b0_new, th_new = _proj_nnls(A_aug, b_aug, p)
        if np.allclose(th, th_new, rtol=1e-5, atol=1e-6) and abs(b0_new - b0) < 1e-6:
            b0, th = b0_new, th_new
            break
        b0, th = b0_new, th_new

    model = SplineModel(
        beta0=float(b0),
        theta=th,
        knots=knots,
        degree=degree,
        tau=float(tau),
        scaler_x=sx,
        scaler_y=sy,
    )
    # Optional: emit a single-line indicator that no fallback was needed
    # (kept quiet by default to avoid verbosity). Uncomment if desired.
    # try:
    #     print(f"[ENVELOPE] fallback=False tau={tau:.2f} n={len(x)}")
    # except Exception:
    #     pass
    return model


# Fallback: piecewise equal-mass quantile + isotonic regression, no QP solver
@dataclasses.dataclass
class PiecewiseMonotoneModel:
    x_mid: np.ndarray  # scaled midpoints (increasing)
    y_mid: np.ndarray  # scaled monotone values at midpoints
    tau: float
    scaler_x: RobustScaler1D
    scaler_y: RobustScaler1D

    def predict(self, x_raw: np.ndarray) -> np.ndarray:
        xs = self.scaler_x.transform(np.asarray(x_raw, dtype=float))
        # Linear interpolation between midpoints; clamp to ends
        y_scaled = np.interp(xs, self.x_mid, self.y_mid, left=self.y_mid[0], right=self.y_mid[-1])
        return self.scaler_y.inverse(y_scaled)


def _make_piecewise_model(
    x: np.ndarray,
    y: np.ndarray,
    knots: np.ndarray,
    tau: float,
    sx: RobustScaler1D,
    sy: RobustScaler1D,
) -> "PiecewiseMonotoneModel":
    """Build a monotone piecewise-quantile model on equal-mass intervals.

    Shared fallback used when SciPy is unavailable or when spline fitting fails.
    """
    t = np.asarray(knots, dtype=float)
    brk = np.unique(t)
    mids = 0.5 * (brk[:-1] + brk[1:])
    yq: List[float] = []
    wts: List[int] = []
    eps = 1e-10
    for i in range(len(brk) - 1):
        lo, hi = float(brk[i]), float(brk[i + 1])
        if i < len(brk) - 2:
            mask = (x >= lo - eps) & (x < hi - eps)
        else:
            mask = (x >= lo - eps) & (x <= hi + eps)
        yi = y[mask]
        if yi.size == 0:
            yq.append(np.nan)
            wts.append(0)
        else:
            yq.append(float(np.nanquantile(yi, tau)))
            wts.append(int(yi.size))
    yq_arr = np.asarray(yq, dtype=float)
    # Fill missing bins by nearest neighbor interpolation
    if np.any(~np.isfinite(yq_arr)):
        finite = np.isfinite(yq_arr)
        if finite.any():
            yq_arr[~finite] = np.interp(np.flatnonzero(~finite), np.flatnonzero(finite), yq_arr[finite])
        else:
            yq_arr[:] = float(np.nanmedian(y))
    yq_iso = _isotonic_increasing(yq_arr, np.asarray(wts, dtype=float))
    return PiecewiseMonotoneModel(
        x_mid=mids.astype(float),
        y_mid=yq_iso.astype(float),
        tau=float(tau),
        scaler_x=sx,
        scaler_y=sy,
    )


def _isotonic_increasing(values: np.ndarray, weights: Optional[np.ndarray] = None) -> np.ndarray:
    v = np.asarray(values, dtype=float)
    n = int(v.size)
    if n == 0:
        return v
    if weights is None:
        w = np.ones(n, dtype=float)
    else:
        w = np.asarray(weights, dtype=float)
    # Ensure positive weights to avoid divisions by zero
    w = np.where(w <= 0, 1.0, w)
    # PAV using block lists (stable for non-integer weights when we track block length separately)
    level = [float(vi) for vi in v]
    weight = [float(wi) for wi in w]
    blen = [1 for _ in range(n)]  # number of bins represented by each block
    i = 0
    tol = 1e-12
    while i < len(level) - 1:
        if level[i] <= level[i + 1] + tol:
            i += 1
            continue
        # Merge i and i+1
        new_w = weight[i] + weight[i + 1]
        new_lvl = (level[i] * weight[i] + level[i + 1] * weight[i + 1]) / new_w
        level[i] = new_lvl
        weight[i] = new_w
        blen[i] += blen[i + 1]
        del level[i + 1]
        del weight[i + 1]
        del blen[i + 1]
        # Walk back as needed
        while i > 0 and level[i - 1] > level[i] + tol:
            new_w = weight[i - 1] + weight[i]
            new_lvl = (level[i - 1] * weight[i - 1] + level[i] * weight[i]) / new_w
            level[i - 1] = new_lvl
            weight[i - 1] = new_w
            blen[i - 1] += blen[i]
            del level[i]
            del weight[i]
            del blen[i]
            i -= 1
        if i < 0:
            i = 0
    # Expand to length n by repeating block averages by block length
    out = np.empty(n, dtype=float)
    idx = 0
    for lvl, ln in zip(level, blen):
        out[idx : idx + ln] = lvl
        idx += ln
    return out


# --------------------------------------------------------------------------------------
# Isoquant trade-off (h(x,y) = c) and combined envelope
# --------------------------------------------------------------------------------------


@dataclasses.dataclass
class IsoquantParams:
    p: float  # 2,4,8, or math.inf
    sx: float
    sy: float
    c: float
    scaler_x: RobustScaler1D
    scaler_y: RobustScaler1D

    def y_of_x(self, x_raw: np.ndarray) -> np.ndarray:
        """Evaluate y_h(x) in raw units on raw x."""
        xs = self.scaler_x.transform(np.asarray(x_raw, dtype=float))
        if math.isinf(self.p):
            ys = self.sy * np.maximum(self.c - xs / self.sx, 0.0)
        else:
            # Avoid negative inside power due to float error
            inner = np.maximum(self.c ** self.p - (xs / self.sx) ** self.p, 0.0)
            ys = self.sy * np.power(inner, 1.0 / self.p)
        return self.scaler_y.inverse(ys)


def _lp_aggregator(x: np.ndarray, y: np.ndarray, p: float, sx: float, sy: float) -> np.ndarray:
    xs = x / float(sx)
    ys = y / float(sy)
    if math.isinf(p):
        return np.maximum(xs, ys)
    return np.power(np.power(np.abs(xs), p) + np.power(np.abs(ys), p), 1.0 / p)


def fit_isoquant(
    x_raw: np.ndarray,
    y_raw: np.ndarray,
    tau: float = 0.90,
    p_grid: Sequence[float] = (2.0, 4.0, 8.0, math.inf),
    g_model: Optional[SplineModel] = None,
) -> IsoquantParams:
    """Fit monotone isoquant h(x,y)=c and expose explicit y_h(x) curve.

    - Uses robust scaling (same as spline) and fixes sx, sy to the scaled IQRs (=1).
    - Sets c as the tau-quantile of h over the train set.
    - Selects p to match validation coverage and, as a tie-breaker, the MSE
      to the current upper spline g(x) near the upper envelope.
    """
    x_raw = np.asarray(x_raw, dtype=float)
    y_raw = np.asarray(y_raw, dtype=float)
    mask = np.isfinite(x_raw) & np.isfinite(y_raw)
    if not np.any(mask):
        raise ValueError("No finite (x,y) pairs")
    x_raw = x_raw[mask]
    y_raw = y_raw[mask]

    sxr = RobustScaler1D.fit(x_raw)
    syr = RobustScaler1D.fit(y_raw)
    x = sxr.transform(x_raw)
    y = syr.transform(y_raw)

    tr_idx, va_idx = _train_valid_split_by_quantiles(x, valid_frac=0.2, bins=10, seed=0)
    xt, yt = x[tr_idx], y[tr_idx]
    xv, yv = x[va_idx], y[va_idx]

    best = None
    best_score = float("inf")

    for p in p_grid:
        h_tr = _lp_aggregator(xt, yt, p, sx=1.0, sy=1.0)
        c = float(np.quantile(h_tr, tau))
        # Validation coverage
        h_va = _lp_aggregator(xv, yv, p, sx=1.0, sy=1.0)
        cov = float(np.mean(h_va <= c)) if h_va.size else float("nan")
        cov_err = abs(cov - tau) if np.isfinite(cov) else 1.0
        mse = 0.0
        if g_model is not None:
            # Compare to the upper spline near the elbow region on a grid
            xg = np.linspace(float(np.min(x)), float(np.max(x)), num=100)
            yg_from_iso = syr.inverse(
                IsoquantParams(p=p, sx=1.0, sy=1.0, c=c, scaler_x=sxr, scaler_y=syr).y_of_x(sxr.inverse(xg))
            )  # returns raw; convert back to scaled
            yg_from_iso = syr.transform(yg_from_iso)
            yg_from_g = syr.transform(g_model.predict(sxr.inverse(xg)))
            mse = float(np.mean(np.square(np.minimum(yg_from_iso, yg_from_g) - yg_from_g)))
        score = cov_err + 0.01 * mse
        if score < best_score:
            best_score = score
            best = (p, c)

    if best is None:
        raise RuntimeError("Isoquant fitting failed")

    p_sel, c_sel = best
    params = IsoquantParams(
        p=float(p_sel), sx=1.0, sy=1.0, c=float(c_sel), scaler_x=sxr, scaler_y=syr
    )
    return params


@dataclasses.dataclass
class CombinedUpper:
    g_model: SplineModel
    iso_params: IsoquantParams
    x_star: float
    y_star: float

    def predict(self, x_raw: np.ndarray, smooth_min_kappa: Optional[float] = None) -> np.ndarray:
        gx = self.g_model.predict(x_raw)
        hy = self.iso_params.y_of_x(x_raw)
        if smooth_min_kappa is None or smooth_min_kappa <= 0:
            return np.minimum(gx, hy)
        k = float(smooth_min_kappa)
        # Smooth-min in raw space
        return -(1.0 / k) * np.log(np.exp(-k * gx) + np.exp(-k * hy))


def _find_intersection_scalar(
    g_model: SplineModel,
    iso_params: IsoquantParams,
    x_lo: float,
    x_hi: float,
    max_iter: int = 60,
    tol: float = 1e-6,
) -> Tuple[float, float]:
    """Find x* with g(x*) ≈ y_h(x*) via bisection; fall back to min-gap point."""
    def phi(xval: float) -> float:
        return float(g_model.predict(np.array([xval]))[0] - iso_params.y_of_x(np.array([xval]))[0])

    a, b = float(x_lo), float(x_hi)
    fa, fb = phi(a), phi(b)
    if math.isnan(fa) or math.isnan(fb):
        fa, fb = 0.0, 0.0
    if fa == 0.0:
        return a, float(g_model.predict(np.array([a]))[0])
    if fb == 0.0:
        return b, float(g_model.predict(np.array([b]))[0])
    # If different signs, bisection
    if fa * fb < 0:
        for _ in range(max_iter):
            m = 0.5 * (a + b)
            fm = phi(m)
            if abs(fm) < tol:
                y = float(g_model.predict(np.array([m]))[0])
                return m, y
            if fa * fm <= 0:
                b, fb = m, fm
            else:
                a, fa = m, fm
        m = 0.5 * (a + b)
        y = float(g_model.predict(np.array([m]))[0])
        return m, y
    # Else: pick the point of minimum absolute gap on a fine grid
    xs = np.linspace(a, b, num=400)
    gx = g_model.predict(xs)
    hy = iso_params.y_of_x(xs)
    i = int(np.argmin(np.abs(gx - hy)))
    return float(xs[i]), float(gx[i])


def fit_upper_lower_with_tradeoff(
    x_raw: np.ndarray,
    y_raw: np.ndarray,
    tau_hi: float = 0.90,
    tau_lo: float = 0.10,
    K: int = 20,
    degree: int = 3,
    lam_grid: Sequence[float] = (1e-4, 1e-3, 1e-2, 1e-1, 1e0, 1e1),
    p_grid: Sequence[float] = (2.0, 4.0, 8.0, math.inf),
    label: Optional[str] = None,
) -> Tuple[SplineModel, SplineModel, IsoquantParams, CombinedUpper]:
    """Full pipeline: fit upper/lower splines, isoquant, and combined upper."""
    g_model = fit_monotone_quantile_spline(
        x_raw, y_raw, tau=tau_hi, K=K, degree=degree, lam_grid=lam_grid, fit_label=(f"{label} upper" if label else None)
    )
    f_model = fit_monotone_quantile_spline(
        x_raw, y_raw, tau=tau_lo, K=K, degree=degree, lam_grid=lam_grid, fit_label=(f"{label} lower" if label else None)
    )
    iso_params = fit_isoquant(x_raw, y_raw, tau=tau_hi, p_grid=p_grid, g_model=g_model)
    # Intersection on data range
    xmin = float(np.nanmin(x_raw))
    xmax = float(np.nanmax(x_raw))
    x_star, y_star = _find_intersection_scalar(g_model, iso_params, xmin, xmax)
    combined = CombinedUpper(g_model=g_model, iso_params=iso_params, x_star=x_star, y_star=y_star)
    return g_model, f_model, iso_params, combined


# --------------------------------------------------------------------------------------
# Optional plotting helper for quick QA
# --------------------------------------------------------------------------------------


def quick_plot(
    x_raw: np.ndarray,
    y_raw: np.ndarray,
    g_model: SplineModel,
    f_model: Optional[SplineModel] = None,
    iso_params: Optional[IsoquantParams] = None,
    combined: Optional[CombinedUpper] = None,
    title: Optional[str] = None,
    save_path: Optional[str] = None,
):
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError("matplotlib is required for plotting") from e

    x_raw = np.asarray(x_raw, dtype=float)
    y_raw = np.asarray(y_raw, dtype=float)
    xs = np.linspace(float(np.nanmin(x_raw)), float(np.nanmax(x_raw)), num=300)
    plt.figure(figsize=(6.5, 5.0))
    plt.scatter(x_raw, y_raw, s=12, alpha=0.25, label="points")
    plt.plot(xs, g_model.predict(xs), label=f"g (tau={g_model.tau:.2f})", color="#1f77b4", linewidth=2.0)
    if f_model is not None:
        plt.plot(xs, f_model.predict(xs), label=f"f (tau={f_model.tau:.2f})", color="#2ca02c", linewidth=2.0)
    if iso_params is not None:
        plt.plot(xs, iso_params.y_of_x(xs), label="isoquant y_h(x)", color="#d62728", linewidth=2.0)
    if combined is not None:
        plt.plot(xs, combined.predict(xs), label="combined upper", color="#9467bd", linewidth=2.0)
        plt.scatter([combined.x_star], [combined.y_star], color="#9467bd", s=36, zorder=5, label="elbow")
    if title:
        plt.title(title)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend(loc="best", fontsize=9)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


__all__ = [
    "RobustScaler1D",
    "SplineModel",
    "IsoquantParams",
    "CombinedUpper",
    "fit_monotone_quantile_spline",
    "fit_isoquant",
    "fit_upper_lower_with_tradeoff",
    "quick_plot",
]

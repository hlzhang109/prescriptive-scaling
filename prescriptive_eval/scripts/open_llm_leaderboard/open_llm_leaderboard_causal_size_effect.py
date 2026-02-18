#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

# Ensure repo root is on sys.path when running as a script, so the
# `skill_frontier/` package takes precedence over `scripts/skill_frontier.py`.
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from skill_frontier.core.envelope import i_m_spline_design

try:
    from scipy.interpolate import BSpline
    from scipy.optimize import minimize
except Exception as e:  # pragma: no cover
    BSpline = None
    minimize = None
    _SCIPY_IMPORT_ERROR = e
else:
    _SCIPY_IMPORT_ERROR = None


_RAW_TASK_COLUMNS: tuple[str, ...] = (
    "IFEval Raw",
    "BBH Raw",
    "MATH Lvl 5 Raw",
    "GPQA Raw",
    "MUSR Raw",
    "MMLU-PRO Raw",
)

# Mapping: canonical OLL Raw task columns -> columns in `tables/open_llm_leaderboard/validation_leaderboard.csv`.
_NEW_LEADERBOARD_TASK_MAP: dict[str, str] = {
    "IFEval Raw": "leaderboard_ifeval_inst_level_strict_acc_none",
    "BBH Raw": "leaderboard_bbh_acc_norm_none",
    "MATH Lvl 5 Raw": "leaderboard_math_hard_exact_match_none",
    "GPQA Raw": "leaderboard_gpqa_acc_norm_none",
    "MUSR Raw": "leaderboard_musr_acc_norm_none",
    "MMLU-PRO Raw": "leaderboard_mmlu_pro_acc_none",
}


def _slugify(text: str) -> str:
    out = str(text).strip().lower()
    out = "".join(ch if ch.isalnum() else "_" for ch in out)
    out = "_".join([p for p in out.split("_") if p])
    return out or "task"


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _safe_to_datetime(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce", utc=True, format="mixed").dt.tz_convert(None)


def _coerce_bool(series: pd.Series) -> pd.Series:
    def _to_bool(v: object) -> bool:
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return False
        if isinstance(v, bool):
            return bool(v)
        s = str(v).strip().lower()
        return s in {"1", "true", "t", "yes", "y"}

    return series.apply(_to_bool)


def _pinball_loss(y: np.ndarray, yhat: np.ndarray, tau: float) -> float:
    """Pinball loss for quantile tau."""
    y = np.asarray(y, dtype=float)
    yhat = np.asarray(yhat, dtype=float)
    u = y - yhat
    return float(np.mean(np.maximum(float(tau) * u, (float(tau) - 1.0) * u)))


def _coverage(y: np.ndarray, yhat: np.ndarray) -> float:
    y = np.asarray(y, dtype=float)
    yhat = np.asarray(yhat, dtype=float)
    return float(np.mean(y <= yhat))


def _sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    out = np.empty_like(x, dtype=float)
    pos = x >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    expx = np.exp(x[~pos])
    out[~pos] = expx / (1.0 + expx)
    return out


def _softplus(x: np.ndarray) -> np.ndarray:
    """Stable softplus: log(1+exp(x))."""
    x = np.asarray(x, dtype=float)
    return np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0.0)


def _augmented_knots(x: np.ndarray, degree: int, K: int) -> np.ndarray:
    """Open knot vector with (approximately) equal-mass interior knots."""
    x = np.asarray(x, dtype=float)
    degree = int(degree)
    K = max(0, int(K))
    n = int(x.size)
    if n == 0:
        return np.array([0, 0, 0, 0, 1, 1, 1, 1], dtype=float)
    xmin = float(np.nanmin(x))
    xmax = float(np.nanmax(x))
    if not np.isfinite(xmin) or not np.isfinite(xmax) or xmin == xmax:
        xmin, xmax = (0.0, 1.0)
    xu = np.unique(x[np.isfinite(x)])
    unique_x = int(xu.size)
    r_cap = max(0, unique_x - 1)
    r_target = min(K, max(0, int(np.sqrt(max(n, 1))) - 1), r_cap)
    eps = 1e-10

    def try_build(r: int) -> np.ndarray:
        if r <= 0:
            return np.array([], dtype=float)
        q_edges = np.linspace(0.0, 1.0, num=r + 2)
        edges = np.quantile(x, q_edges)
        if np.all(np.diff(edges) > eps):
            return edges[1:-1].astype(float)
        return np.array([], dtype=float)

    r = r_target
    interior = try_build(r)
    while r > 0 and interior.size != r:
        r -= 1
        interior = try_build(r)
    return np.concatenate(
        [
            np.repeat(xmin, degree + 1),
            interior,
            np.repeat(xmax, degree + 1),
        ]
    ).astype(float)


def _second_difference_matrix(p: int) -> np.ndarray:
    """(p-2) x p matrix such that (D2 @ theta) are second differences."""
    p = int(p)
    if p < 3:
        return np.zeros((0, p), dtype=float)
    D2 = np.zeros((p - 2, p), dtype=float)
    for i in range(p - 2):
        D2[i, i] = 1.0
        D2[i, i + 1] = -2.0
        D2[i, i + 2] = 1.0
    return D2


def _bspline_design(x: np.ndarray, knots: np.ndarray, degree: int) -> np.ndarray:
    if BSpline is None:  # pragma: no cover
        raise RuntimeError(f"SciPy is required for B-splines; import error: {_SCIPY_IMPORT_ERROR!r}")
    x = np.asarray(x, dtype=float)
    knots = np.asarray(knots, dtype=float)
    degree = int(degree)
    p = int(knots.size - degree - 1)
    if p <= 0:
        raise ValueError("Invalid knot vector: not enough knots for requested degree")
    x_clipped = np.clip(x, float(knots[0]), float(knots[-1]))
    basis = np.zeros((x_clipped.size, p), dtype=float)
    for j in range(p):
        c = np.zeros(p, dtype=float)
        c[j] = 1.0
        spl = BSpline(knots, c, degree, extrapolate=False)
        vals = spl(x_clipped)
        basis[:, j] = np.where(np.isfinite(vals), vals, 0.0)
    return basis


def _fit_quantile_boundary_gam(
    *,
    x_raw: np.ndarray,
    t_scaled: np.ndarray,
    y: np.ndarray,
    tau: float,
    K: int,
    degree: int,
    lam: float,
    smooth_eps: float,
    seed: int,
) -> Dict[str, object]:
    """Fit q_tau(y|x,t) via a logit-link GAM with monotone exposure g(t).

    Model (with centered x):
      q_hat = sigmoid(eta)
      eta = alpha + beta x + phi(t) + delta g(t) + theta x g(t)
      phi(t) = B(t) @ gamma (B-spline, centered; last basis dropped for identifiability)
      g(t) = I(t) @ theta_g (I-spline; theta_g >= 0 and sum(theta_g)=1)

    Fit by minimizing a smoothed pinball loss on y-scale + spline roughness penalties.
    """
    if minimize is None:  # pragma: no cover
        raise RuntimeError(f"SciPy optimize is required; import error: {_SCIPY_IMPORT_ERROR!r}")

    rng = np.random.default_rng(int(seed))
    x_raw = np.asarray(x_raw, dtype=float)
    t_scaled = np.asarray(t_scaled, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x_raw) & np.isfinite(t_scaled) & np.isfinite(y)
    if not np.any(mask):
        raise ValueError("No finite (x,t,y) triples")
    x_raw = x_raw[mask]
    t_scaled = np.clip(t_scaled[mask], 0.0, 1.0)
    y = y[mask]

    x_mean = float(np.mean(x_raw))
    x = x_raw - x_mean

    # Bases
    t_for_knots = np.concatenate([t_scaled, np.array([0.0, 1.0], dtype=float)])
    knots_phi = _augmented_knots(t_for_knots, degree=degree, K=K)
    B_full = _bspline_design(t_scaled, knots_phi, degree)
    if B_full.shape[1] < 2:
        raise ValueError("Time spline basis too small; increase --K or check t range")
    Phi0 = B_full[:, :-1]  # drop 1 basis to avoid intercept redundancy
    phi_mean = Phi0.mean(axis=0)
    Phi = Phi0 - phi_mean
    p_phi = int(Phi.shape[1])
    D2_phi = _second_difference_matrix(p_phi)

    knots_g = _augmented_knots(t_for_knots, degree=degree, K=K)
    I_g, _, _ = i_m_spline_design(t_scaled, knots_g, degree)
    p_g = int(I_g.shape[1])
    D2_g = _second_difference_matrix(p_g)

    # Initial params
    q0 = float(np.quantile(y, float(tau)))
    q0 = float(np.clip(q0, 1e-4, 1.0 - 1e-4))
    alpha0 = float(np.log(q0 / (1.0 - q0)))
    beta0 = 0.1
    delta0 = 0.0
    theta0 = 0.0
    gamma0 = np.zeros(p_phi, dtype=float)
    theta_g0 = np.full(p_g, 1.0 / float(max(1, p_g)), dtype=float)
    theta_g0 = np.clip(theta_g0 + 1e-3 * rng.standard_normal(p_g), 0.0, None)
    theta_g0 = theta_g0 / float(np.sum(theta_g0)) if float(np.sum(theta_g0)) > 0 else theta_g0

    p0 = np.concatenate(
        [
            np.array([alpha0, beta0, delta0, theta0], dtype=float),
            gamma0,
            theta_g0,
        ]
    )

    # Bounds: enforce nonnegativity on beta, delta, theta; theta_g >= 0.
    bounds: list[tuple[Optional[float], Optional[float]]] = []
    bounds.append((None, None))  # alpha
    bounds.append((0.0, None))  # beta
    bounds.append((0.0, None))  # delta
    bounds.append((0.0, None))  # theta
    bounds.extend([(None, None)] * p_phi)  # gamma
    bounds.extend([(0.0, 1.0)] * p_g)  # theta_g

    idx_gamma0 = 4
    idx_theta_g0 = 4 + p_phi

    inv_m_phi = 1.0 / float(max(1, D2_phi.shape[0]))
    inv_m_g = 1.0 / float(max(1, D2_g.shape[0]))
    lam = float(max(lam, 0.0))

    def unpack(params: np.ndarray) -> tuple[float, float, float, float, np.ndarray, np.ndarray]:
        alpha = float(params[0])
        beta = float(params[1])
        delta = float(params[2])
        theta = float(params[3])
        gamma = np.asarray(params[idx_gamma0:idx_theta_g0], dtype=float)
        theta_g = np.asarray(params[idx_theta_g0:], dtype=float)
        return alpha, beta, delta, theta, gamma, theta_g

    def predict_internal(params: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        alpha, beta, delta, theta, gamma, theta_g = unpack(params)
        g = I_g @ theta_g
        eta = alpha + beta * x + (Phi @ gamma) + delta * g + theta * (x * g)
        q_hat = _sigmoid(np.clip(eta, -60.0, 60.0))
        return q_hat, g

    def objective(params: np.ndarray) -> float:
        q_hat, _ = predict_internal(params)
        u = y - q_hat
        s = u / float(max(smooth_eps, 1e-8))
        loss = (float(tau) - 1.0) * u + float(smooth_eps) * _softplus(s)
        val = float(np.mean(loss))
        alpha, beta, delta, theta, gamma, theta_g = unpack(params)
        if D2_phi.shape[0] > 0:
            val += lam * inv_m_phi * float(np.sum(np.square(D2_phi @ gamma)))
        if D2_g.shape[0] > 0:
            val += lam * inv_m_g * float(np.sum(np.square(D2_g @ theta_g)))
        val += 1e-6 * (beta * beta + delta * delta + theta * theta)
        return val

    def gradient(params: np.ndarray) -> np.ndarray:
        alpha, beta, delta, theta, gamma, theta_g = unpack(params)
        q_hat, g = predict_internal(params)
        u = y - q_hat
        s = u / float(max(smooth_eps, 1e-8))
        dloss_du = (float(tau) - 1.0) + _sigmoid(np.clip(s, -60.0, 60.0))
        dL_deta = (-dloss_du * (q_hat * (1.0 - q_hat))) / float(y.size)

        grad_alpha = float(np.sum(dL_deta))
        grad_beta = float(np.sum(dL_deta * x))
        grad_delta = float(np.sum(dL_deta * g))
        grad_theta = float(np.sum(dL_deta * (x * g)))
        grad_gamma = Phi.T @ dL_deta
        grad_theta_g = I_g.T @ (dL_deta * (delta + theta * x))

        if D2_phi.shape[0] > 0:
            grad_gamma = grad_gamma + (2.0 * lam * inv_m_phi) * (D2_phi.T @ (D2_phi @ gamma))
        if D2_g.shape[0] > 0:
            grad_theta_g = grad_theta_g + (2.0 * lam * inv_m_g) * (D2_g.T @ (D2_g @ theta_g))
        grad_beta += 2e-6 * beta
        grad_delta += 2e-6 * delta
        grad_theta += 2e-6 * theta

        out = np.zeros_like(params, dtype=float)
        out[0] = grad_alpha
        out[1] = grad_beta
        out[2] = grad_delta
        out[3] = grad_theta
        out[idx_gamma0:idx_theta_g0] = grad_gamma
        out[idx_theta_g0:] = grad_theta_g
        return out

    def constr_sum_theta_g(params: np.ndarray) -> float:
        return float(np.sum(params[idx_theta_g0:]) - 1.0)

    def constr_sum_theta_g_jac(params: np.ndarray) -> np.ndarray:
        out = np.zeros_like(params, dtype=float)
        out[idx_theta_g0:] = 1.0
        return out

    res = minimize(
        objective,
        p0,
        jac=gradient,
        method="SLSQP",
        bounds=bounds,
        constraints=[{"type": "eq", "fun": constr_sum_theta_g, "jac": constr_sum_theta_g_jac}],
        options={"maxiter": 400, "ftol": 1e-9, "disp": False},
    )
    if not bool(res.success):
        raise RuntimeError(f"Optimization failed: {res.message}")

    alpha, beta, delta, theta, gamma, theta_g = unpack(np.asarray(res.x, dtype=float))
    return {
        "alpha": alpha,
        "beta": beta,
        "delta": delta,
        "theta": theta,
        "gamma_phi": gamma,
        "theta_g": theta_g,
        "x_mean": x_mean,
        "knots_phi": knots_phi,
        "phi_mean": phi_mean,
        "degree_phi": int(degree),
        "knots_g": knots_g,
        "degree_g": int(degree),
        "tau": float(tau),
        "lam": float(lam),
        "smooth_eps": float(smooth_eps),
        "opt": {"n_iter": int(getattr(res, "nit", -1)), "status": int(res.status), "message": str(res.message)},
    }


def _predict_quantile_boundary_gam(
    model: Dict[str, object],
    *,
    x_raw: np.ndarray,
    t_scaled: np.ndarray,
) -> np.ndarray:
    x_raw = np.asarray(x_raw, dtype=float)
    t_scaled = np.clip(np.asarray(t_scaled, dtype=float), 0.0, 1.0)
    x = x_raw - float(model["x_mean"])

    knots_phi = np.asarray(model["knots_phi"], dtype=float)
    degree_phi = int(model["degree_phi"])
    B_full = _bspline_design(t_scaled, knots_phi, degree_phi)
    Phi0 = B_full[:, :-1]
    Phi = Phi0 - np.asarray(model["phi_mean"], dtype=float)

    knots_g = np.asarray(model["knots_g"], dtype=float)
    degree_g = int(model["degree_g"])
    I_g, _, _ = i_m_spline_design(t_scaled, knots_g, degree_g)
    theta_g = np.asarray(model["theta_g"], dtype=float)
    g = I_g @ theta_g

    eta = (
        float(model["alpha"])
        + float(model["beta"]) * x
        + Phi @ np.asarray(model["gamma_phi"], dtype=float)
        + float(model["delta"]) * g
        + float(model["theta"]) * (x * g)
    )
    return _sigmoid(np.clip(eta, -60.0, 60.0))


def _load_top_models_meta(top_models_csv: str) -> pd.DataFrame:
    df = pd.read_csv(top_models_csv, usecols=["model_id", "last_modified", "num_params_B"])
    df["model_id"] = df["model_id"].astype(str).str.strip()
    df["last_modified"] = _safe_to_datetime(df["last_modified"])
    df["num_params_B"] = pd.to_numeric(df["num_params_B"], errors="coerce")
    df = df.dropna(subset=["model_id"]).copy()
    return df.groupby("model_id", as_index=False).agg({"last_modified": "max", "num_params_B": "max"})


def _load_extra_leaderboard_points(
    *,
    extra_csv: str,
    top_models_csv: str,
    extra_compute_csv: Optional[str],
    tasks: Sequence[str],
    compute_multiplier: float,
) -> pd.DataFrame:
    """Load extra eval points and align them to the OLL v2 schema used by the plots.

    Supports two schemas:
      1) `tables/new_leaderboard_results_with_tokens.csv` (has tokens/params and a HF timestamp column).
      2) `tables/open_llm_leaderboard/validation_leaderboard.csv` (metrics only; join params/date via top_models_by_base.csv).
    """
    df = pd.read_csv(extra_csv)
    out = pd.DataFrame()
    out["model_id"] = df.get("model_id", np.nan)

    has_tokens = ("Pretraining tokens (T)" in df.columns) and ("#Params (B)" in df.columns)
    if has_tokens:
        out["Pretraining tokens (T)"] = pd.to_numeric(df.get("Pretraining tokens (T)", np.nan), errors="coerce")
        out["#Params (B)"] = pd.to_numeric(df.get("#Params (B)", np.nan), errors="coerce")
        if "lastModified" in df.columns:
            out["Upload To Hub Date"] = df.get("lastModified", np.nan)
        elif "Upload To Hub Date" in df.columns:
            out["Upload To Hub Date"] = df.get("Upload To Hub Date", np.nan)
        elif "Submission Date" in df.columns:
            out["Upload To Hub Date"] = df.get("Submission Date", np.nan)
        else:
            out["Upload To Hub Date"] = np.nan

        pretrain_compute_zflops = pd.to_numeric(df.get("pretrain_compute_zflops", np.nan), errors="coerce")
        out["pretrain_compute_zflops"] = pretrain_compute_zflops
    else:
        if not extra_compute_csv or not os.path.isfile(str(extra_compute_csv)):
            raise FileNotFoundError(
                "Extra leaderboard CSV does not contain tokens/params; "
                "provide --extra_compute_csv (tables/open_llm_leaderboard/new_eval_leaderboard.csv)."
            )
        df_compute = pd.read_csv(str(extra_compute_csv), usecols=["model_id", "pretrain_compute_zflops"])
        df_compute["model_id"] = df_compute["model_id"].astype(str).str.strip()
        df_compute["pretrain_compute_zflops"] = pd.to_numeric(df_compute["pretrain_compute_zflops"], errors="coerce")
        df_compute = df_compute.groupby("model_id", as_index=False)["pretrain_compute_zflops"].max()

        df_meta = _load_top_models_meta(top_models_csv)
        df_join = df[["model_id"]].copy()
        df_join["model_id"] = df_join["model_id"].astype(str).str.strip()
        df_join = df_join.merge(df_compute, on="model_id", how="left")
        df_join = df_join.merge(df_meta, on="model_id", how="left")

        out["#Params (B)"] = pd.to_numeric(df_join["num_params_B"], errors="coerce")
        pretrain_compute_zflops = pd.to_numeric(df_join["pretrain_compute_zflops"], errors="coerce")
        out["Pretraining tokens (T)"] = pretrain_compute_zflops / (float(compute_multiplier) * out["#Params (B)"])
        out["Upload To Hub Date"] = df_join["last_modified"].dt.strftime("%Y-%m-%d %H:%M:%S%z")
        out["pretrain_compute_zflops"] = pretrain_compute_zflops

    for task_col in tasks:
        src = _NEW_LEADERBOARD_TASK_MAP.get(task_col)
        out[task_col] = pd.to_numeric(df.get(src, np.nan), errors="coerce") if src else np.nan

    return out


def load_open_llm_plot_points(
    *,
    input_csv: str,
    extra_csvs: Sequence[str],
    top_models_csv: str,
    extra_compute_csv: str,
    compute_multiplier: float,
) -> tuple[pd.DataFrame, pd.Timestamp]:
    """Return the exact point set used by `outputs/open_llm_leaderboard/current` plots."""
    df_current = pd.read_csv(input_csv)
    if "Upload To Hub Date" not in df_current.columns:
        raise SystemExit(f"Missing required column in {input_csv}: Upload To Hub Date")
    if "#Params (B)" not in df_current.columns:
        raise SystemExit(f"Missing required column in {input_csv}: #Params (B)")

    dt_current = _safe_to_datetime(df_current["Upload To Hub Date"])
    if dt_current.notna().any():
        cutoff_date = dt_current.max()
    else:
        raise SystemExit(f"Could not parse any dates from {input_csv} (Upload To Hub Date)")

    cur_ids: set[str] = set()
    for col in ("fullname", "Base Model", "model_id"):
        if col in df_current.columns:
            cur_ids = set(df_current[col].dropna().astype(str).str.strip().tolist())
            break

    tasks = [c for c in _RAW_TASK_COLUMNS if c in df_current.columns]
    if not tasks:
        raise SystemExit(f"None of expected task columns found in {input_csv}: {_RAW_TASK_COLUMNS}")

    extra_frames: List[pd.DataFrame] = []
    for extra_csv in extra_csvs:
        if not extra_csv or not os.path.isfile(extra_csv):
            continue
        df_extra = _load_extra_leaderboard_points(
            extra_csv=str(extra_csv),
            top_models_csv=str(top_models_csv),
            extra_compute_csv=str(extra_compute_csv),
            tasks=tasks,
            compute_multiplier=float(compute_multiplier),
        )
        if cur_ids:
            df_extra = df_extra[~df_extra["model_id"].astype(str).str.strip().isin(cur_ids)].copy()
        extra_frames.append(df_extra)

    if extra_frames:
        df_extra_all = pd.concat(extra_frames, ignore_index=True, sort=False)
        df_extra_all["model_id"] = df_extra_all["model_id"].astype(str).str.strip()
        df_extra_all = df_extra_all.drop_duplicates(subset=["model_id"], keep="last").copy()
        df = pd.concat([df_current, df_extra_all], ignore_index=True, sort=False)
    else:
        df = df_current

    return df, cutoff_date


def _pick_train_mask(
    dates: pd.Series,
    *,
    split_mode: str,
    cutoff_date: pd.Timestamp,
    min_val: int,
    seed: int,
    val_frac: float,
    time_bins: int,
) -> np.ndarray:
    """Train/validation split used for evaluation.

    Modes:
      - random: IID split (default), val_frac held out uniformly at random.
      - cutoff: out-of-time split; train is <= cutoff_date and val is > cutoff_date
                if there are at least min_val post-cutoff points; otherwise falls back to
                a stratified time-quantile split (val_frac per bin).
    """
    split_mode = str(split_mode).lower().strip()
    n = int(dates.shape[0])
    if n <= 1:
        return np.ones(n, dtype=bool)

    if split_mode == "random":
        rng = np.random.default_rng(int(seed))
        n_val = int(np.floor(float(val_frac) * float(n)))
        n_val = min(max(1, n_val), n - 1)
        perm = rng.permutation(n)
        train_mask = np.ones(n, dtype=bool)
        train_mask[perm[:n_val]] = False
        return train_mask

    if split_mode != "cutoff":
        raise ValueError(f"Unknown split_mode={split_mode!r}; expected 'random' or 'cutoff'")

    dt = _safe_to_datetime(dates)
    pre = dt <= cutoff_date
    post = dt > cutoff_date
    if int(post.sum()) >= int(min_val):
        return pre.to_numpy()

    rng = np.random.default_rng(int(seed))
    df = pd.DataFrame({"t": dt}).dropna()
    if df.shape[0] < 50:
        return pre.to_numpy()
    bins = pd.qcut(df["t"], q=min(int(time_bins), df.shape[0]), duplicates="drop")
    train_mask = np.zeros(n, dtype=bool)
    for _, idx in df.groupby(bins, observed=True).groups.items():
        idx = np.asarray(list(idx), dtype=int)
        if idx.size <= 1:
            continue
        n_val = int(np.floor(float(val_frac) * float(idx.size)))
        n_val = min(max(1, n_val), idx.size - 1)
        keep = np.ones(idx.size, dtype=bool)
        val_idx = rng.choice(idx.size, size=n_val, replace=False)
        keep[val_idx] = False
        train_mask[idx] = keep
    return train_mask


def main(argv: Optional[Sequence[str]] = None) -> None:
    ap = argparse.ArgumentParser(
        description="Fit an attainable performance boundary vs model size and time (high-quantile frontier)."
    )
    ap.add_argument(
        "--input_csv",
        default=os.path.join("tables", "open_llm_leaderboard", "open_llm_leaderboard_with_tokens.csv"),
    )
    ap.add_argument(
        "--extra_csv",
        action="append",
        default=[
            os.path.join("tables", "open_llm_leaderboard", "validation_leaderboard.csv"),
            os.path.join("tables", "new_leaderboard_results_with_tokens.csv"),
        ],
        help="Extra eval CSVs (repeatable).",
    )
    ap.add_argument(
        "--extra_compute_csv",
        default=os.path.join("tables", "open_llm_leaderboard", "new_eval_leaderboard.csv"),
    )
    ap.add_argument("--top_models_csv", default=os.path.join("tables", "top_models_by_base.csv"))
    ap.add_argument("--compute_multiplier", type=float, default=6.0)

    ap.add_argument("--tau", type=float, default=0.98, help="Quantile level for the attainable boundary.")
    ap.add_argument("--K", type=int, default=20, help="Max interior knots for time bases (adapted down internally).")
    ap.add_argument("--degree", type=int, default=3, help="Spline degree for time bases.")
    ap.add_argument(
        "--lam_grid",
        default="1e-5,3e-5,1e-4,3e-4,1e-3,3e-3,1e-2,3e-2,1e-1",
        help="Comma-separated smoothing strengths for validation selection.",
    )
    ap.add_argument(
        "--smooth_eps",
        type=float,
        default=0.002,
        help="Smoothing for pinball loss (in y-units); smaller is closer to exact pinball but less smooth.",
    )

    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--val_frac", type=float, default=0.2)
    ap.add_argument(
        "--split_mode",
        choices=("random", "cutoff"),
        default="random",
        help="Train/validation split mode (default: random 80/20; 'cutoff' holds out post-cutoff models).",
    )
    ap.add_argument("--min_val", type=int, default=200, help="Min post-cutoff val points required for cutoff split.")
    ap.add_argument("--time_bins", type=int, default=10)

    ap.add_argument("--exclude_flagged", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--exclude_merged", action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument("--exclude_moe", action=argparse.BooleanOptionalAction, default=False)

    ap.add_argument("--size_threshold_b", type=float, default=13.0, help="Reference size (B params) for reporting.")
    ap.add_argument(
        "--out_dir",
        default=os.path.join("outputs", "open_llm_leaderboard", "current", "size_boundary_effect"),
    )
    args = ap.parse_args(list(argv) if argv is not None else None)

    df, cutoff_date = load_open_llm_plot_points(
        input_csv=str(args.input_csv),
        extra_csvs=list(args.extra_csv),
        top_models_csv=str(args.top_models_csv),
        extra_compute_csv=str(args.extra_compute_csv),
        compute_multiplier=float(args.compute_multiplier),
    )

    lam_grid = [float(x) for x in str(args.lam_grid).split(",") if str(x).strip() != ""]
    if not lam_grid:
        raise SystemExit("--lam_grid parsed empty; provide at least one value")

    out_dir = str(args.out_dir)
    _ensure_dir(out_dir)

    rows: List[Dict[str, object]] = []
    for task in _RAW_TASK_COLUMNS:
        if task not in df.columns:
            continue

        date_col = "Upload To Hub Date"
        params_col = "#Params (B)"
        model_col = "fullname" if "fullname" in df.columns else ("Base Model" if "Base Model" in df.columns else "model_id")
        flagged_col = "Flagged" if "Flagged" in df.columns else None
        merged_col = "Merged" if "Merged" in df.columns else None
        moe_col = "MoE" if "MoE" in df.columns else None

        cols = [model_col, date_col, params_col, task]
        for c in (flagged_col, merged_col, moe_col):
            if c is not None:
                cols.append(c)
        sub = df[cols].copy()
        sub = sub.rename(columns={model_col: "model_id", date_col: "date", params_col: "params_b", task: "y"})
        sub["date"] = _safe_to_datetime(sub["date"])
        sub["params_b"] = pd.to_numeric(sub["params_b"], errors="coerce")
        sub["y"] = pd.to_numeric(sub["y"], errors="coerce")
        sub = sub.dropna(subset=["date", "params_b", "y"]).copy()
        sub = sub[(sub["params_b"] > 0)].copy()

        if bool(args.exclude_flagged) and flagged_col is not None and flagged_col in sub.columns:
            sub["_flagged"] = _coerce_bool(sub[flagged_col])
            sub_unflagged = sub[~sub["_flagged"]].copy()
            sub = sub_unflagged if not sub_unflagged.empty else sub
        if bool(args.exclude_merged) and merged_col is not None and merged_col in sub.columns:
            sub["_merged"] = _coerce_bool(sub[merged_col])
            sub = sub[~sub["_merged"]].copy()
        if bool(args.exclude_moe) and moe_col is not None and moe_col in sub.columns:
            sub["_moe"] = _coerce_bool(sub[moe_col])
            sub = sub[~sub["_moe"]].copy()

        if sub.shape[0] < 300:
            continue

        x_raw = np.log(sub["params_b"].to_numpy(dtype=float))
        y = sub["y"].to_numpy(dtype=float)
        t0 = sub["date"].min()
        t1 = sub["date"].max()
        denom = float((t1 - t0).total_seconds())
        if not np.isfinite(denom) or denom <= 0:
            t_scaled_all = np.zeros(sub.shape[0], dtype=float)
        else:
            t_scaled_all = (sub["date"] - t0).dt.total_seconds().to_numpy(dtype=float) / denom

        mask = np.isfinite(x_raw) & np.isfinite(y) & np.isfinite(t_scaled_all)
        x_raw = x_raw[mask]
        y = y[mask]
        t_scaled_all = t_scaled_all[mask]
        sub = sub.iloc[np.where(mask)[0]].copy()

        train_mask = _pick_train_mask(
            sub["date"],
            split_mode=str(args.split_mode),
            cutoff_date=cutoff_date,
            min_val=int(args.min_val),
            seed=int(args.seed),
            val_frac=float(args.val_frac),
            time_bins=int(args.time_bins),
        )
        val_mask = ~train_mask

        x_tr = x_raw[train_mask]
        y_tr = y[train_mask]
        x_va = x_raw[val_mask]
        y_va = y[val_mask]
        t_tr = t_scaled_all[train_mask]
        t_va = t_scaled_all[val_mask]

        best_model: Optional[Dict[str, object]] = None
        best_lam: Optional[float] = None
        best_cov_err = float("inf")
        best_pinball = float("inf")
        for lam in lam_grid:
            try:
                candidate = _fit_quantile_boundary_gam(
                    x_raw=x_tr,
                    t_scaled=t_tr,
                    y=y_tr,
                    tau=float(args.tau),
                    K=int(args.K),
                    degree=int(args.degree),
                    lam=float(lam),
                    smooth_eps=float(args.smooth_eps),
                    seed=int(args.seed),
                )
            except Exception as e:
                print(f"[size_boundary] {task}: lam={lam:g} fit failed: {e}", file=sys.stderr)
                continue

            yhat_va = (
                _predict_quantile_boundary_gam(candidate, x_raw=x_va, t_scaled=t_va)
                if y_va.size
                else np.array([], dtype=float)
            )
            cov_va = _coverage(y_va, yhat_va) if y_va.size else float("nan")
            cov_err = float(abs(cov_va - float(args.tau))) if np.isfinite(cov_va) else float("inf")
            pin_va = _pinball_loss(y_va, yhat_va, float(args.tau)) if y_va.size else float("inf")

            if (cov_err < best_cov_err - 1e-6) or (abs(cov_err - best_cov_err) <= 1e-6 and pin_va < best_pinball):
                best_cov_err = cov_err
                best_pinball = pin_va
                best_model = candidate
                best_lam = float(lam)

        if best_model is None:
            print(f"[size_boundary] {task}: all lam fits failed; skipping", file=sys.stderr)
            continue

        model = best_model
        yhat_tr = _predict_quantile_boundary_gam(model, x_raw=x_tr, t_scaled=t_tr)
        yhat_va = _predict_quantile_boundary_gam(model, x_raw=x_va, t_scaled=t_va) if y_va.size else np.array([], dtype=float)

        cov_tr = _coverage(y_tr, yhat_tr)
        cov_va = _coverage(y_va, yhat_va) if y_va.size else float("nan")

        # Boundary effect summaries at median time, computed on the training distribution (avoid extrapolation).
        x_p10, x_p50, x_p90 = (float(np.quantile(x_tr, q)) for q in (0.10, 0.50, 0.90))
        t_ref = float(np.quantile(t_tr, 0.50))
        y_p10 = float(_predict_quantile_boundary_gam(model, x_raw=np.array([x_p10]), t_scaled=np.array([t_ref]))[0])
        y_p50 = float(_predict_quantile_boundary_gam(model, x_raw=np.array([x_p50]), t_scaled=np.array([t_ref]))[0])
        y_p90 = float(_predict_quantile_boundary_gam(model, x_raw=np.array([x_p90]), t_scaled=np.array([t_ref]))[0])
        y_gap_p90_p10 = y_p90 - y_p10

        x_thr = float(np.log(float(args.size_threshold_b)))
        y_thr = float(_predict_quantile_boundary_gam(model, x_raw=np.array([x_thr]), t_scaled=np.array([t_ref]))[0])
        y_thr_early = float(_predict_quantile_boundary_gam(model, x_raw=np.array([x_thr]), t_scaled=np.array([0.0]))[0])
        y_thr_late = float(_predict_quantile_boundary_gam(model, x_raw=np.array([x_thr]), t_scaled=np.array([1.0]))[0])

        metrics = {
            "n_total": int(sub.shape[0]),
            "n_train": int(train_mask.sum()),
            "n_val": int(val_mask.sum()),
            "split_mode": str(args.split_mode),
            "cutoff_date": cutoff_date.date().isoformat(),
            "tau": float(args.tau),
            "pinball_train": _pinball_loss(y_tr, yhat_tr, float(args.tau)),
            "pinball_val": _pinball_loss(y_va, yhat_va, float(args.tau)) if y_va.size else float("nan"),
            "coverage_train": cov_tr,
            "coverage_val": cov_va,
            "coverage_error_train": float(abs(cov_tr - float(args.tau))),
            "coverage_error_val": float(abs(cov_va - float(args.tau))) if np.isfinite(cov_va) else float("nan"),
            "x_log_params_p10": x_p10,
            "x_log_params_p50": x_p50,
            "x_log_params_p90": x_p90,
            "t_scaled_ref": t_ref,
            "q_tau_p10": y_p10,
            "q_tau_p50": y_p50,
            "q_tau_p90": y_p90,
            "q_tau_gap_p90_p10": y_gap_p90_p10,
            "q_tau_at_size_threshold": y_thr,
            "q_tau_at_size_threshold_early": y_thr_early,
            "q_tau_at_size_threshold_late": y_thr_late,
            "size_threshold_b": float(args.size_threshold_b),
            "x_log_params_train_min": float(np.min(x_tr)),
            "x_log_params_train_max": float(np.max(x_tr)),
            "coef_beta_logit": float(model["beta"]),
            "coef_delta_logit": float(model["delta"]),
            "coef_theta_logit": float(model["theta"]),
            "size_slope_logit_early": float(model["beta"]),
            "size_slope_logit_late": float(model["beta"]) + float(model["theta"]),
            "lam_selected": float(best_lam) if best_lam is not None else float("nan"),
        }

        out_json = os.path.join(out_dir, f"{_slugify(task)}.json")
        with open(out_json, "w") as f:
            json.dump(
                {
                    "task": task,
                    "metrics": metrics,
                    "model": {
                        "alpha": float(model["alpha"]),
                        "beta": float(model["beta"]),
                        "delta": float(model["delta"]),
                        "theta": float(model["theta"]),
                        "gamma_phi": [float(v) for v in np.asarray(model["gamma_phi"], dtype=float).tolist()],
                        "theta_g": [float(v) for v in np.asarray(model["theta_g"], dtype=float).tolist()],
                        "x_mean": float(model["x_mean"]),
                        "phi": {
                            "degree": int(model["degree_phi"]),
                            "knots": [float(v) for v in np.asarray(model["knots_phi"], dtype=float).tolist()],
                            "center_mean": [float(v) for v in np.asarray(model["phi_mean"], dtype=float).tolist()],
                            "dropped_basis": "last",
                        },
                        "g": {
                            "degree": int(model["degree_g"]),
                            "knots": [float(v) for v in np.asarray(model["knots_g"], dtype=float).tolist()],
                            "sum_theta_g": float(np.sum(np.asarray(model["theta_g"], dtype=float))),
                        },
                        "optimization": dict(model.get("opt", {})),
                    },
                    "hyperparams": {
                        "tau": float(args.tau),
                        "K": int(args.K),
                        "degree": int(args.degree),
                        "lam_grid": lam_grid,
                        "lam_selected": float(best_lam) if best_lam is not None else None,
                        "smooth_eps": float(args.smooth_eps),
                        "split": {
                            "mode": str(args.split_mode),
                            "val_frac": float(args.val_frac),
                            "seed": int(args.seed),
                            "cutoff_date": cutoff_date.date().isoformat(),
                            "min_val": int(args.min_val),
                            "time_bins": int(args.time_bins),
                        },
                        "filters": {
                            "exclude_flagged": bool(args.exclude_flagged),
                            "exclude_merged": bool(args.exclude_merged),
                            "exclude_moe": bool(args.exclude_moe),
                        },
                        "size_threshold_b": float(args.size_threshold_b),
                    },
                },
                f,
                indent=2,
                sort_keys=True,
            )

        rows.append({"task": task, **metrics})
        print(
            f"[size_boundary] {task}: n_train={metrics['n_train']} n_val={metrics['n_val']} "
            f"pinball_val={metrics['pinball_val']:.6f} cov_val={metrics['coverage_val'] if np.isfinite(metrics['coverage_val']) else float('nan'):.3f}"
        )

    out_csv = os.path.join(out_dir, "summary.csv")
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f"[size_boundary] wrote {out_csv}")


if __name__ == "__main__":
    main()

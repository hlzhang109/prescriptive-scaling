#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Optional, Sequence

import numpy as np
import pandas as pd

# Ensure repo root is on sys.path when running as a script.
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from skill_frontier.plotting.configs import frontier_1d as frontier_1d_cfg  # type: ignore
from skill_frontier.plotting.configs import mpl_rc as mpl_rc_cfg  # type: ignore


def _ensure_out_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _load_required_csv(path: str, required_cols: Sequence[str]) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise KeyError(f"Missing columns in {path}: {missing}")
    return df


def _logit(p: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    p = np.clip(p, eps, 1.0 - eps)
    return np.log(p / (1.0 - p))


def _fit_linear(x: np.ndarray, y: np.ndarray, *, intercept_nonneg: bool = False) -> np.ndarray:
    x_design = np.column_stack([np.ones_like(x), x])
    coef, *_ = np.linalg.lstsq(x_design, y, rcond=None)
    if not intercept_nonneg:
        return coef

    # Constrain intercept (a) to be non-negative. If the unconstrained least-squares solution
    # already satisfies a >= 0, it's optimal. Otherwise the constrained optimum lies on the
    # boundary a = 0, and we re-fit b with a fixed.
    if float(coef[0]) >= 0.0:
        return coef

    denom = float(np.sum(x * x))
    b = float(np.sum(x * y) / denom) if denom > 0.0 else 0.0
    return np.array([0.0, b], dtype=float)


def _predict_linear(coef: np.ndarray, x: np.ndarray) -> np.ndarray:
    return coef[0] + coef[1] * x


def _format_cutoff_for_legend(ts: pd.Timestamp) -> str:
    month = ts.strftime("%b") + "."
    day = int(ts.day)
    if 11 <= (day % 100) <= 13:
        suffix = "th"
    else:
        suffix = {1: "st", 2: "nd", 3: "rd"}.get(day % 10, "th")
    return f"{month} {day}{suffix} {int(ts.year)}"


def _r2(y: np.ndarray, yhat: np.ndarray) -> float:
    ss_res = float(np.sum((y - yhat) ** 2))
    ss_tot = float(np.sum((y - float(np.mean(y))) ** 2))
    if ss_tot == 0.0:
        return float("nan")
    return 1.0 - (ss_res / ss_tot)


def _format_shifted_logit_legend(alpha: float, beta: float, gamma: float) -> str:
    sign = "+" if beta >= 0 else "-"
    beta_abs = abs(float(beta))
    return (
        rf"$\hat{{y}} = {float(alpha):.2f} {sign} {beta_abs:.2f}\,"
        rf"\operatorname{{logit}}\!\left(\frac{{x}}{{100}} - {float(gamma):.2f}\right)$"
    )


def _sigmoid(t: np.ndarray) -> np.ndarray:
    t = np.asarray(t, float)
    t = np.clip(t, -60.0, 60.0)
    return 1.0 / (1.0 + np.exp(-t))


def _format_logit_to_logit_legend(a: float, b: float) -> str:
    sign = "+" if b >= 0 else "-"
    b_abs = abs(float(b))
    return (
        rf"$\operatorname{{logit}}\!\left(\frac{{\hat{{y}}}}{{100}}\right)"
        rf" = {float(a):.2f} {sign} {b_abs:.2f}\,"
        rf"\operatorname{{logit}}\!\left(\frac{{x}}{{100}}\right)$"
    )


def _fit_logit_to_logit_linear(*, x_pct: np.ndarray, y_pct: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    x = np.clip(np.asarray(x_pct, float) / 100.0, eps, 1.0 - eps)
    y = np.clip(np.asarray(y_pct, float) / 100.0, eps, 1.0 - eps)
    X = _logit(x, eps=eps)
    Y = _logit(y, eps=eps)
    m = np.isfinite(X) & np.isfinite(Y)
    X = X[m]
    Y = Y[m]
    design = np.column_stack([np.ones_like(X), X])
    coef, _ = _fit_ols(design, Y)
    return coef.astype(float)


def _fit_ols(design: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    coef, *_ = np.linalg.lstsq(design, y, rcond=None)
    resid = y - (design @ coef)
    return coef, resid


def _ols_se(design: np.ndarray, resid: np.ndarray) -> np.ndarray:
    n, p = design.shape
    dof = max(1, int(n - p))
    sse = float(np.sum(resid**2))
    sigma2 = sse / float(dof)
    xtx = design.T @ design
    cov = sigma2 * np.linalg.pinv(xtx)
    se = np.sqrt(np.clip(np.diag(cov), 0.0, np.inf))
    return se.astype(float)


def _contamination_test_logit_to_logit(
    *,
    x_pct: np.ndarray,
    y_pct: np.ndarray,
    group_post: np.ndarray,
    n_bins: int,
    n_perm: int,
    seed: int,
    eps: float = 1e-6,
) -> dict:
    """Test for contamination-induced y inflation using a logit-to-logit link.

    Implements exactly:
      - Transform: X = logit(x), Y = logit(y), with x,y in [0,1] from percents (clipped by eps).
      - OLS: Y = a + b X + gamma * g + e, where g=1 for post-release (red), 0 otherwise (blue).
      - One-sided stratified permutation p-value for gamma > 0 (permute g within X-quantile bins).
      - Guardrail: restrict to the common support of X across groups.
    """
    x_pct = np.asarray(x_pct, float)
    y_pct = np.asarray(y_pct, float)
    group_post = np.asarray(group_post, bool)
    if x_pct.shape != y_pct.shape or x_pct.shape != group_post.shape:
        raise ValueError("x_pct, y_pct, group_post must have the same shape.")

    x = np.clip(x_pct / 100.0, eps, 1.0 - eps)
    y = np.clip(y_pct / 100.0, eps, 1.0 - eps)
    X = _logit(x, eps=eps)
    Y = _logit(y, eps=eps)
    mask = np.isfinite(X) & np.isfinite(Y)
    X = X[mask]
    Y = Y[mask]
    g = group_post[mask]

    pre = ~g
    post = g
    if int(pre.sum()) < 5 or int(post.sum()) < 5:
        raise RuntimeError(
            f"Not enough points for test (n_pre={int(pre.sum())}, n_post={int(post.sum())})."
        )

    x_min_pre, x_max_pre = float(np.min(X[pre])), float(np.max(X[pre]))
    x_min_post, x_max_post = float(np.min(X[post])), float(np.max(X[post]))
    support_lo = max(x_min_pre, x_min_post)
    support_hi = min(x_max_pre, x_max_post)
    if not (np.isfinite(support_lo) and np.isfinite(support_hi) and support_hi > support_lo):
        raise RuntimeError("No overlap in X support between groups.")

    in_support = (X >= support_lo) & (X <= support_hi)
    dropped = int((~in_support).sum())
    Xs = X[in_support]
    Ys = Y[in_support]
    gs = g[in_support]

    pre_s = ~gs
    post_s = gs
    if int(pre_s.sum()) < 5 or int(post_s.sum()) < 5:
        raise RuntimeError(
            f"Not enough points after support restriction (n_pre={int(pre_s.sum())}, n_post={int(post_s.sum())})."
        )

    # OLS: Y = a + b X + gamma g + e
    design = np.column_stack([np.ones_like(Xs), Xs, gs.astype(float)])
    coef_obs, resid_obs = _fit_ols(design, Ys)
    se_obs = _ols_se(design, resid_obs)

    a_hat, b_hat, gamma_hat = float(coef_obs[0]), float(coef_obs[1]), float(coef_obs[2])
    se_gamma = float(se_obs[2]) if se_obs.size >= 3 else float("nan")
    t_gamma = float(gamma_hat / se_gamma) if np.isfinite(se_gamma) and se_gamma > 0 else float("nan")
    r2_obs = _r2(Ys, design @ coef_obs)

    # Stratified permutation by X quantile bins (duplicates="drop" for safety).
    bin_id = pd.qcut(pd.Series(Xs), q=int(n_bins), labels=False, duplicates="drop").to_numpy()
    rng = np.random.default_rng(int(seed))
    gamma_perm = np.empty(int(n_perm), dtype=float)
    for i in range(int(n_perm)):
        g_perm = gs.copy()
        for b in np.unique(bin_id):
            idx = np.flatnonzero(bin_id == b)
            if idx.size <= 1:
                continue
            g_perm[idx] = rng.permutation(g_perm[idx])
        design_p = np.column_stack([np.ones_like(Xs), Xs, g_perm.astype(float)])
        coef_p, _ = _fit_ols(design_p, Ys)
        gamma_perm[i] = float(coef_p[2])

    p_one_sided = float((1.0 + np.sum(gamma_perm >= gamma_hat)) / (len(gamma_perm) + 1.0))
    p_two_sided = float((1.0 + np.sum(np.abs(gamma_perm) >= abs(gamma_hat))) / (len(gamma_perm) + 1.0))

    perm_mean = float(np.mean(gamma_perm))
    perm_std = float(np.std(gamma_perm, ddof=1)) if len(gamma_perm) > 1 else float("nan")
    z_perm = float((gamma_hat - perm_mean) / perm_std) if np.isfinite(perm_std) and perm_std > 0 else float("nan")
    odds_ratio = float(np.exp(gamma_hat)) if np.isfinite(gamma_hat) else float("nan")
    q05, q50, q95 = [float(x) for x in np.quantile(gamma_perm, [0.05, 0.5, 0.95])]

    return {
        "eps_clip": float(eps),
        "n_used_total": int(X.shape[0]),
        "n_used_in_support": int(Xs.shape[0]),
        "n_dropped_outside_support": int(dropped),
        "n_pre": int(pre_s.sum()),
        "n_post": int(post_s.sum()),
        "X_support_lo": float(support_lo),
        "X_support_hi": float(support_hi),
        "X_pre_min": x_min_pre,
        "X_pre_max": x_max_pre,
        "X_post_min": x_min_post,
        "X_post_max": x_max_post,
        "coef_intercept_a": a_hat,
        "coef_slope_b": b_hat,
        "coef_group_shift_gamma": gamma_hat,
        "odds_ratio_group_shift": odds_ratio,
        "se_group_shift_gamma": se_gamma,
        "t_group_shift_gamma": t_gamma,
        "r2_logit_fit": float(r2_obs),
        "n_bins": int(n_bins),
        "n_perm": int(n_perm),
        "seed": int(seed),
        "pvalue_one_sided_gamma_gt_0": p_one_sided,
        "pvalue_two_sided": p_two_sided,
        "gamma_perm_mean": perm_mean,
        "gamma_perm_std": perm_std,
        "gamma_perm_q05": q05,
        "gamma_perm_q50": q50,
        "gamma_perm_q95": q95,
        "gamma_perm_zscore": z_perm,
    }


def _fit_shifted_logit_linear(
    *,
    x_pct: np.ndarray,
    y_pct: np.ndarray,
    gamma_min: float = 0.0,
    eps: float = 1e-6,
    n_iter: int = 32,
) -> tuple[float, np.ndarray]:
    """
    Fit y ≈ a + b * logit(x/100 - gamma), with gamma constrained to [gamma_min, gamma_max),
    where gamma_max = min(x/100) - eps (to avoid clipping on the fit subset).
    """
    x_norm = x_pct / 100.0
    gamma_max = float(np.min(x_norm) - eps)
    if not np.isfinite(gamma_max) or gamma_max <= gamma_min:
        gamma = float(max(gamma_min, 0.0))
        u = _logit(x_norm - gamma, eps=eps)
        return gamma, _fit_linear(u, y_pct, intercept_nonneg=True)

    a = float(max(gamma_min, 0.0))
    b = gamma_max
    phi = (1.0 + 5.0**0.5) / 2.0

    def sse(gamma: float) -> float:
        u = _logit(x_norm - gamma, eps=eps)
        coef = _fit_linear(u, y_pct, intercept_nonneg=True)
        resid = y_pct - _predict_linear(coef, u)
        return float(np.sum(resid**2))

    c = b - (b - a) / phi
    d = a + (b - a) / phi
    fc = sse(c)
    fd = sse(d)
    for _ in range(int(n_iter)):
        if fc < fd:
            b = d
            d = c
            fd = fc
            c = b - (b - a) / phi
            fc = sse(c)
        else:
            a = c
            c = d
            fc = fd
            d = a + (b - a) / phi
            fd = sse(d)

    gamma = float((a + b) / 2.0)
    u = _logit(x_norm - gamma, eps=eps)
    return gamma, _fit_linear(u, y_pct, intercept_nonneg=True)


def _contamination_test_stratified_permutation(
    *,
    x_pct: np.ndarray,
    y_pct: np.ndarray,
    group_post: np.ndarray,
    n_bins: int,
    n_perm: int,
    seed: int,
) -> dict:
    if x_pct.shape != y_pct.shape or x_pct.shape != group_post.shape:
        raise ValueError("x_pct, y_pct, group_post must have the same shape.")

    pre_mask = ~group_post
    post_mask = group_post
    if pre_mask.sum() < 5 or post_mask.sum() < 5:
        raise RuntimeError(
            f"Not enough points for test (n_pre={int(pre_mask.sum())}, n_post={int(post_mask.sum())})."
        )

    # Mapping hypothesis: y ≈ a + b * logit(x/100 - gamma), with gamma >= 0.
    # Here we assume the mapping is "valid" for post-release points, then test whether pre-release points
    # are systematically worse (i.e., have lower residuals conditional on x).
    x_norm = x_pct / 100.0
    gamma_obs, coef_obs = _fit_shifted_logit_linear(x_pct=x_pct[post_mask], y_pct=y_pct[post_mask])
    u_all = _logit(x_norm - gamma_obs)
    resid_obs = y_pct - _predict_linear(coef_obs, u_all)
    delta_obs = float(resid_obs[post_mask].mean() - resid_obs[pre_mask].mean())
    r2_obs = _r2(y_pct[post_mask], _predict_linear(coef_obs, _logit(x_norm[post_mask] - gamma_obs)))

    # Stratified permutation by x (MATH-500) quantile bins.
    bin_id = pd.qcut(pd.Series(x_pct), q=n_bins, labels=False, duplicates="drop").to_numpy()
    rng = np.random.default_rng(seed)
    delta_perm = np.empty(int(n_perm), dtype=float)
    for i in range(int(n_perm)):
        group_perm = group_post.copy()
        for b in np.unique(bin_id):
            idx = np.flatnonzero(bin_id == b)
            if idx.size <= 1:
                continue
            group_perm[idx] = rng.permutation(group_post[idx])

        pre_m = ~group_perm
        post_m = group_perm
        gamma, coef = _fit_shifted_logit_linear(x_pct=x_pct[post_m], y_pct=y_pct[post_m])
        u_perm = _logit(x_norm - gamma)
        resid = y_pct - _predict_linear(coef, u_perm)
        delta_perm[i] = resid[post_m].mean() - resid[pre_m].mean()

    # One-sided test: post-release residuals are larger (contamination advantage).
    p_one_sided = float((1.0 + np.sum(delta_perm >= delta_obs)) / (len(delta_perm) + 1.0))
    p_two_sided = float((1.0 + np.sum(np.abs(delta_perm) >= abs(delta_obs))) / (len(delta_perm) + 1.0))

    return {
        "delta_mean_residual_post_minus_pre": delta_obs,
        "pvalue_one_sided_post_gt_pre": p_one_sided,
        "pvalue_two_sided": p_two_sided,
        "coef_alpha": float(coef_obs[0]),
        "coef_beta": float(coef_obs[1]),
        "coef_gamma": float(gamma_obs),
        "r2_fit_subset": float(r2_obs),
        "perm_delta_mean": float(delta_perm.mean()),
        "perm_delta_std": float(delta_perm.std(ddof=1)) if len(delta_perm) > 1 else float("nan"),
    }


def main(argv: Optional[Sequence[str]] = None) -> None:
    ap = argparse.ArgumentParser(description="Plot MATH-500 vs AIME 2025 from Artificial Analysis exports.")
    ap.add_argument(
        "--aime_csv",
        default=os.path.join("tables", "artificial_analysis", "aime-2025.csv"),
        help="Path to AIME 2025 leaderboard CSV (from download_artificial_analysis_aime2025.py).",
    )
    ap.add_argument(
        "--math500_csv",
        default=os.path.join("tables", "artificial_analysis", "math-500.csv"),
        help="Path to MATH-500 leaderboard CSV (from download_artificial_analysis_math500.py).",
    )
    ap.add_argument(
        "--out_dir",
        default=os.path.join("outputs", "artificial_analysis", "comparison"),
        help="Output directory.",
    )
    ap.add_argument(
        "--release_cutoff",
        default="2025-02-06",
        help="ISO date; models released before this date are treated as pre-release.",
    )
    ap.add_argument("--n_perm", type=int, default=2000, help="Number of stratified permutations for the test.")
    ap.add_argument("--n_bins", type=int, default=10, help="Number of x-quantile bins for stratified permutation.")
    ap.add_argument("--seed", type=int, default=0, help="RNG seed for the permutation test.")
    args = ap.parse_args(list(argv) if argv is not None else None)

    df_aime = _load_required_csv(
        str(args.aime_csv),
        required_cols=("model_id", "aime_2025_pct", "release_date"),
    )
    df_math = _load_required_csv(
        str(args.math500_csv),
        required_cols=("model_id", "math_500_pct"),
    )

    df = df_math[["model_id", "model_name", "math_500_pct"]].merge(
        df_aime[["model_id", "aime_2025_pct", "release_date"]], on="model_id", how="inner"
    )
    df["math_500_pct"] = pd.to_numeric(df["math_500_pct"], errors="coerce")
    df["aime_2025_pct"] = pd.to_numeric(df["aime_2025_pct"], errors="coerce")
    df["release_date"] = pd.to_datetime(df["release_date"], errors="coerce")
    df = df.dropna(subset=["math_500_pct", "aime_2025_pct"]).copy()
    if df.empty:
        raise RuntimeError("No rows after merging MATH-500 and AIME 2025 tables.")

    import matplotlib as mpl  # type: ignore
    import matplotlib.pyplot as plt  # type: ignore

    mpl.rcParams["font.family"] = mpl_rc_cfg.FONT_FAMILY
    mpl.rcParams["mathtext.fontset"] = mpl_rc_cfg.MATH_FONTSET
    mpl.rcParams["text.usetex"] = True
    mpl.rcParams["text.latex.preamble"] = mpl_rc_cfg.LATEX_PREAMBLE

    # Match outputs/open_llm_leaderboard/comparison styling.
    panel_figsize = (0.75 * (2069.0 / 300.0), 1344.0 / 300.0)
    label_fs_x = float(frontier_1d_cfg.LABEL_FONTSIZE_X)
    label_fs_y = float(frontier_1d_cfg.LABEL_FONTSIZE_Y)
    tick_fs = float(frontier_1d_cfg.TICK_LABELSIZE)
    legend_fs = float(frontier_1d_cfg.LEGEND_FONTSIZE) * 0.75

    release_cutoff = pd.Timestamp(str(args.release_cutoff))
    cutoff_label = _format_cutoff_for_legend(release_cutoff)
    is_pre_release = df["release_date"].notna() & (df["release_date"] < release_cutoff)
    pre_mask = is_pre_release.to_numpy()
    post_mask = ~pre_mask

    fig, ax = plt.subplots(figsize=panel_figsize)
    ax.scatter(
        df.loc[pre_mask, "math_500_pct"].to_numpy(),
        df.loc[pre_mask, "aime_2025_pct"].to_numpy(),
        s=18,
        alpha=0.8,
        c="#1f77b4",
        label=f"Before {cutoff_label}",
        rasterized=True,
    )
    ax.scatter(
        df.loc[post_mask, "math_500_pct"].to_numpy(),
        df.loc[post_mask, "aime_2025_pct"].to_numpy(),
        s=18,
        alpha=0.8,
        c="firebrick",
        label=f"After {cutoff_label}",
        rasterized=True,
    )

    x_post = df.loc[post_mask, "math_500_pct"].to_numpy()
    y_post = df.loc[post_mask, "aime_2025_pct"].to_numpy()
    m_post = np.isfinite(x_post) & np.isfinite(y_post)
    if int(np.sum(m_post)) >= 6:
        coef_fit = _fit_logit_to_logit_linear(x_pct=x_post[m_post], y_pct=y_post[m_post])
        x_grid = np.linspace(float(np.min(x_post[m_post])), float(np.max(x_post[m_post])), 250)
        x_grid_norm = np.clip(x_grid / 100.0, 1e-6, 1.0 - 1e-6)
        X_grid = _logit(x_grid_norm)
        y_grid = 100.0 * _sigmoid(_predict_linear(coef_fit, X_grid))
        ax.plot(
            x_grid,
            y_grid,
            color="firebrick",
            linewidth=2.1,
            alpha=0.9,
            label=_format_logit_to_logit_legend(coef_fit[0], coef_fit[1]),
        )

    ax.set_xlabel(r"\textbf{MATH-500 (\%)}", fontsize=label_fs_x)
    ax.set_ylabel(r"\textbf{AIME 2025 (\%)}", fontsize=label_fs_y)
    ax.tick_params(axis="both", labelsize=tick_fs)
    ax.set_ylim(bottom=0.0)

    ax.grid(
        True,
        which="major",
        axis="y",
        linestyle=frontier_1d_cfg.GRID_MAJOR_LINESTYLE,
        linewidth=frontier_1d_cfg.GRID_MAJOR_LINEWIDTH,
        color=frontier_1d_cfg.GRID_MAJOR_COLOR,
        alpha=frontier_1d_cfg.GRID_MAJOR_ALPHA,
    )
    for spine in ax.spines.values():
        spine.set_linewidth(frontier_1d_cfg.SPINE_LINEWIDTH)
        spine.set_color(frontier_1d_cfg.SPINE_COLOR)

    ax.legend(loc="best", fontsize=legend_fs, frameon=True, framealpha=frontier_1d_cfg.LEGEND_FRAMEALPHA)
    fig.subplots_adjust(left=0.18, right=0.98, bottom=0.22, top=0.98)

    out_dir = str(args.out_dir)
    _ensure_out_dir(out_dir)
    out_base = os.path.join(out_dir, "task_pair_math_500_vs_aime_2025")
    fig.savefig(out_base + ".png", dpi=300)
    fig.savefig(out_base + ".pdf", dpi=300)
    plt.close(fig)

    df_test = df[df["release_date"].notna()].copy()
    group_post = (df_test["release_date"] >= release_cutoff).to_numpy()
    test_results = _contamination_test_logit_to_logit(
        x_pct=df_test["math_500_pct"].to_numpy(),
        y_pct=df_test["aime_2025_pct"].to_numpy(),
        group_post=group_post,
        n_bins=int(args.n_bins),
        n_perm=int(args.n_perm),
        seed=int(args.seed),
    )
    stats_path = os.path.join(out_dir, "contamination_test_math_500_vs_aime_2025.json")
    payload = {
        "release_cutoff": str(release_cutoff.date()),
        "n_total_merged": int(df.shape[0]),
        "n_with_release_date": int(df_test.shape[0]),
        "n_missing_release_date": int(df.shape[0] - df_test.shape[0]),
        "n_pre_release": int((~group_post).sum()),
        "n_post_release": int(group_post.sum()),
        "test_name": "logit_to_logit_group_shift",
        **test_results,
    }
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")

    print(f"[ok] wrote {out_base}.png|.pdf (n={int(df.shape[0])})")
    print(f"[ok] wrote {stats_path}")


if __name__ == "__main__":
    main()

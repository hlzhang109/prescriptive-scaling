#!/usr/bin/env python3
"""
Lambda ablation for period-4 single-k sigmoid frontiers (no budget).

This script is **additive**: it does not modify any existing code or outputs.
It:

  * Reuses the existing evaluation pipeline in
    `scripts/evaluate/sigmoid_binned_mae.py` (period4, single_k, no manifests).
  * Temporarily monkey-patches `scripts.smooth_single_skill_frontier.fit_sigmoid_frontier`
    inside this process to change only the L2 regularization coefficient
    on the sigmoid slope parameters (a, b).
  * Runs the evaluation once per lambda value and writes results into
    lambda-tagged subdirectories under a new root:

        outputs/lambda_ablation/period4_singlek_no_budget/lambda_<value>/

Existing results in `outputs/` are not touched.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np


REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), os.pardir, os.pardir)
)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Run a lambda (L2) ablation for period-4 single-k, no-budget "
            "sigmoid frontier evaluation."
        )
    )
    # Data / compute configuration (kept compatible with regenerate_all.sh)
    default_csv = os.path.join(
        REPO_ROOT,
        "tables",
        "open_llm_leaderboard",
        "open_llm_leaderboard_with_tokens.csv",
    )
    p.add_argument(
        "--csv",
        default=default_csv,
        help="Input OLL CSV (default: with_tokens schema).",
    )
    p.add_argument(
        "--compute_product_cols",
        nargs=2,
        default=["Pretraining tokens (T)", "#Params (B)"],
        help="Two columns to multiply to obtain FLOPs proxy.",
    )
    p.add_argument(
        "--compute_multiplier",
        type=float,
        default=6.0,
        help="Multiplier applied to product of compute columns.",
    )

    # Evaluation configuration (matched to period4_singlek_no_budget baseline)
    p.add_argument(
        "--tau",
        type=float,
        default=0.98,
        help="Quantile parameter tau used in evaluation.",
    )
    p.add_argument(
        "--bins",
        type=int,
        default=10,
        help="Target number of equal-mass bins in log-compute.",
    )
    p.add_argument(
        "--min_bin_size",
        type=int,
        default=30,
        help="Minimum number of points per bin (after merging).",
    )

    # Lambda grid
    p.add_argument(
        "--lambdas",
        type=float,
        nargs="+",
        default=[0.0, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0],
        help=(
            "List of L2 regularization strengths to sweep over. "
            "Each value produces its own lambda_<value> subdirectory."
        ),
    )

    # Output root
    p.add_argument(
        "--out_root",
        default=os.path.join(
            REPO_ROOT, "outputs", "lambda_ablation", "period4_singlek_no_budget"
        ),
        help="Root directory for lambda-tagged evaluation outputs.",
    )

    return p


def _make_fit_sigmoid_frontier_with_lambda(lambda_reg: float):
    """
    Return a drop-in replacement for scripts.smooth_single_skill_frontier.fit_sigmoid_frontier
    that uses a configurable L2 coefficient `lambda_reg` instead of the hard-coded 1e-3.

    The implementation is intentionally kept as close as possible to the original
    quantile-per-point frontier fitter; robust-bin mode is also supported so that
    call sites can reuse this function transparently.
    """
    import scripts.smooth_single_skill_frontier as ssf  # type: ignore

    _sigmoid = ssf._sigmoid  # type: ignore[attr-defined]
    _softplus = ssf._softplus  # type: ignore[attr-defined]
    create_equal_mass_bins = getattr(ssf, "create_equal_mass_bins", None)

    def fit_sigmoid_frontier(  # type: ignore[override]
        x: np.ndarray,
        y: np.ndarray,
        tau: float = 0.98,
        use_log10_x: bool = True,
        grid_points: int = 400,
        fit_mode: str = "quantile_per_point",
        bins_for_fit: Optional[int] = None,
        min_bin_size_for_fit: Optional[int] = None,
        bin_frontier_quantile: float = 0.90,
        bin_trim_fraction: float = 0.05,
        bin_edges_for_fit: Optional[np.ndarray] = None,
        lambda_b: float | None = None,
        kappa_final: float = 50.0,
        curve_x_limits: Optional[Tuple[float, float]] = None,
        **_ignored: object,
    ) -> Tuple[np.ndarray, np.ndarray]:
        x_arr = np.asarray(x, float)
        y_arr = np.asarray(y, float)
        m = np.isfinite(x_arr) & np.isfinite(y_arr)
        if curve_x_limits is not None:
            x_lo, x_hi = curve_x_limits
            if x_lo is not None:
                m = m & (x_arr >= float(x_lo))
            if x_hi is not None:
                m = m & (x_arr <= float(x_hi))
        x_f = x_arr[m]
        y_f = y_arr[m]
        if x_f.size < 3:
            return np.array([]), np.array([])
        if use_log10_x:
            z = np.log10(np.maximum(x_f, 1e-300))
        else:
            z = x_f.copy()
        order = np.argsort(z)
        z = z[order]
        y_sorted = y_f[order]

        # Initial guesses (copied from original implementation)
        y0_init = float(np.nanmax([0.0, np.nanpercentile(y_sorted, 5)]))
        ymax = float(np.nanmin([1.0, np.nanpercentile(y_sorted, 99)]))
        L_init = max(1e-3, ymax - y0_init)
        z_med = float(np.nanmedian(z))
        y_med = float(np.nanmedian(y_sorted))
        frac = np.clip((y_med - y0_init) / max(L_init, 1e-6), 1e-6, 1 - 1e-6)
        a_init = float(np.log(frac / (1 - frac)))  # logit(frac)
        b_init = 1.0

        def pack(theta_raw: np.ndarray) -> Tuple[float, float, float, float]:
            g0, gL, a, bb = theta_raw
            y0 = float(_sigmoid(np.array([g0]))[0])
            L_cap = max(1e-6, 1.0 - y0 - 1e-6)
            L = float(_sigmoid(np.array([gL]))[0]) * L_cap
            b = float(_softplus(bb))
            return y0, L, a, b

        # Optional: precompute robust bin-level summaries for alternative fit mode
        def _compute_bin_frontier_points() -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
            if bin_edges_for_fit is not None:
                edges_local = np.asarray(bin_edges_for_fit, float)
                if edges_local.size < 2:
                    return np.array([]), np.array([]), np.array([])
            else:
                if create_equal_mass_bins is None:
                    return np.array([]), np.array([]), np.array([])
                if bins_for_fit is None or bins_for_fit <= 0:
                    return np.array([]), np.array([]), np.array([])
                mb = int(
                    max(
                        1,
                        min_bin_size_for_fit if min_bin_size_for_fit is not None else 30,
                    )
                )
                try:
                    edges_local = create_equal_mass_bins(
                        z, int(max(1, bins_for_fit)), mb
                    )
                except Exception:
                    return np.array([]), np.array([]), np.array([])
                if edges_local.size < 2:
                    return np.array([]), np.array([]), np.array([])

            B = int(edges_local.size - 1)
            bin_idx = np.searchsorted(edges_local, z, side="right") - 1
            bin_idx = np.clip(bin_idx, 0, B - 1)
            z_bin = np.zeros(B, dtype=float)
            y_bin = np.zeros(B, dtype=float)
            w_bin = np.zeros(B, dtype=float)
            for b in range(B):
                m_b = bin_idx == b
                if not np.any(m_b):
                    continue
                z_b = z[m_b]
                y_b = y_sorted[m_b]
                n_b = int(y_b.size)
                left, right = float(edges_local[b]), float(edges_local[b + 1])
                z_bin[b] = 0.5 * (left + right)
                y_sorted_b = np.sort(y_b)[::-1]
                trim_k = int(math.floor(float(bin_trim_fraction) * n_b))
                trim_k = min(trim_k, max(n_b - 1, 0))
                if trim_k > 0:
                    y_trimmed = y_sorted_b[trim_k:]
                else:
                    y_trimmed = y_sorted_b
                if y_trimmed.size == 0:
                    continue
                y_bin[b] = float(
                    np.quantile(y_trimmed, float(bin_frontier_quantile))
                )
                w_bin[b] = 1.0
            m_keep = w_bin > 0
            return z_bin[m_keep], y_bin[m_keep], w_bin[m_keep]

        mode = (fit_mode or "quantile_per_point").lower()
        z_bin: np.ndarray
        y_bin: np.ndarray
        w_bin: np.ndarray
        if mode == "robust_bin_frontier":
            z_bin, y_bin, w_bin = _compute_bin_frontier_points()
            if z_bin.size < 3:
                mode = "quantile_per_point"

        def loss(theta_raw: np.ndarray) -> float:
            y0, L, a, b = pack(theta_raw)
            if mode == "robust_bin_frontier":
                yh_bin = y0 + L * _sigmoid(a + b * z_bin)
                r_bin = y_bin - yh_bin
                loss_val = float(np.sum(w_bin * (r_bin**2)))
            else:
                yh = y0 + L * _sigmoid(a + b * z)
                r = y_sorted - yh
                k = float(kappa_final) if float(kappa_final) > 0 else 50.0
                rho = (1.0 / k) * np.log1p(np.exp(k * r)) + (tau - 1.0) * r
                loss_val = float(np.sum(rho))
            return float(loss_val + lambda_reg * (a * a + b * b))

        def _logit(p: float) -> float:
            p = float(np.clip(p, 1e-6, 1 - 1e-6))
            return float(np.log(p / (1 - p)))

        theta0 = np.array(
            [
                _logit(min(0.99, max(0.01, y0_init))),
                _logit(
                    min(0.99, max(0.01, L_init / max(1e-6, 1 - y0_init - 1e-6)))
                ),
                a_init - b_init * z_med,
                np.log(np.expm1(b_init) + 1e-6),
            ]
        )

        best_theta = theta0.copy()
        best_val = loss(theta0)

        try:
            from scipy.optimize import minimize  # type: ignore

            res = minimize(
                loss, theta0, method="L-BFGS-B", options={"maxiter": 10000}
            )
            if res.success and np.isfinite(res.fun):
                best_theta = res.x
                best_val = float(res.fun)
        except Exception:
            rng = np.random.default_rng(0)
            for _ in range(32):
                cand = theta0 + rng.normal(scale=[0.5, 0.5, 1.0, 0.5])
                val = loss(cand)
                if val < best_val:
                    best_val = val
                    best_theta = cand

        y0, L, a, b = pack(best_theta)
        z_grid = np.linspace(float(z.min()), float(z.max()), num=grid_points)
        y_hat = y0 + L * _sigmoid(a + b * z_grid)
        y_hat = np.clip(y_hat, 0.0, 1.0)
        xs_sample = (10.0**z_grid) if use_log10_x else z_grid
        return xs_sample.astype(float), y_hat.astype(float)

    return fit_sigmoid_frontier


def _run_single_lambda(
    lambda_reg: float,
    csv: str,
    compute_product_cols: Sequence[str],
    compute_multiplier: float,
    tau: float,
    bins: int,
    min_bin_size: int,
    out_root: str,
) -> None:
    """
    Patch the sigmoid fitter with the given lambda, then run period4 single_k
    evaluation and write results into a lambda-specific subdirectory.

    The evaluation is modified to use **per-bin pinball loss** (matching the
    training loss) by monkey-patching `compute_bin_statistics` inside the
    MAE evaluation module.
    """
    import scripts.smooth_single_skill_frontier as ssf  # type: ignore
    from scripts.evaluate import sigmoid_binned_mae as sbm  # type: ignore
    from scripts.evaluate.sigmoid_binned_pinball import (  # type: ignore
        _compute_bin_statistics_pinball,
    )

    if not os.path.isfile(csv):
        raise SystemExit(f"CSV not found: {csv}")

    os.makedirs(out_root, exist_ok=True)
    # Directory name based on lambda (compact but unambiguous)
    lam_tag = f"{lambda_reg:.0e}" if lambda_reg != 0.0 else "0"
    out_base = os.path.join(out_root, f"lambda_{lam_tag}")
    os.makedirs(out_base, exist_ok=True)

    # Install patched fitter into the smooth_single_skill_frontier module
    ssf.fit_sigmoid_frontier = _make_fit_sigmoid_frontier_with_lambda(lambda_reg)  # type: ignore[assignment]

    # Patch evaluation to use pinball loss per bin instead of MAE
    sbm.compute_bin_statistics = _compute_bin_statistics_pinball  # type: ignore[assignment]

    # Build argument list for the existing evaluation driver (period4, single_k)
    argv = [
        "--csv",
        csv,
        "--compute_product_cols",
        compute_product_cols[0],
        compute_product_cols[1],
        "--compute_multiplier",
        str(compute_multiplier),
        "--split_mode",
        "period4",
        "--train_mode",
        "single_k",
        "--tau",
        str(tau),
        "--bins",
        str(bins),
        "--min_bin_size",
        str(min_bin_size),
        "--out_base",
        out_base,
    ]

    sbm.main(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)

    for lam in args.lambdas:
        print(f"[lambda_ablation] Running evaluation for lambda={lam} ...")
        _run_single_lambda(
            lambda_reg=float(lam),
            csv=args.csv,
            compute_product_cols=args.compute_product_cols,
            compute_multiplier=float(args.compute_multiplier),
            tau=float(args.tau),
            bins=int(args.bins),
            min_bin_size=int(args.min_bin_size),
            out_root=args.out_root,
        )
        print(f"[lambda_ablation] Done for lambda={lam}")


if __name__ == "__main__":
    main()

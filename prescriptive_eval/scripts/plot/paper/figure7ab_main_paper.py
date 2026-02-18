#!/usr/bin/env python3
"""
Generate Figure 7a and Figure 7b (main paper) with consistent styling.

Figure 7a: GPQA diamond frontier comparison (sigmoid vs I-spline) based on the
           Epoch AI benchmark runs outputs.
Figure 7b: MATH-500 vs AIME 2025 comparison based on Artificial Analysis exports.

Outputs (in --out_dir):
  - figure7a_main_paper.(pdf|png)
  - figure7b_main_paper.(pdf|png)

Notes:
  - Keeps figure size consistent with the original artifacts.
  - Only changes styling (fonts/colors/axes/legend/save settings).
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Optional, Sequence, Tuple

import numpy as np

try:
    import pandas as pd  # type: ignore
except Exception as e:  # pragma: no cover
    raise SystemExit("This script requires pandas.") from e

try:
    import matplotlib as mpl  # type: ignore
    import matplotlib.pyplot as plt  # type: ignore
    from matplotlib.patches import Rectangle  # type: ignore
except Exception as e:  # pragma: no cover
    raise SystemExit("This script requires matplotlib.") from e

# Import bootstrap from `scripts/` by putting the scripts dir on sys.path.
SCRIPTS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)
from _bootstrap import ensure_repo_root_on_path  # type: ignore

REPO_ROOT = ensure_repo_root_on_path(__file__)

from skill_frontier.evaluation.common import fit_sigmoid_predictor, interpolate_curve  # type: ignore
from skill_frontier.plotting.axis_formatting import apply_pretraining_compute_tick_multiplier  # type: ignore
from skill_frontier.plotting.labels import PRETRAINING_COMPUTE_FLOPS_LABEL  # type: ignore
from skill_frontier.plotting.plot_utils import apply_font_embedding  # type: ignore
from scripts.run import ispline_pinball_optimizer as iso  # type: ignore


# -----------------------------------------------------------------------------
# STYLE (paper spec)
# -----------------------------------------------------------------------------

STYLE_RCPARAMS = {
    "font.family": "serif",
    "font.serif": ["Times New Roman"],
    "font.size": 11,
    "axes.labelsize": 13,
    "axes.titlesize": 13,
    "axes.labelweight": "bold",
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 11,
}

STYLE = {
    "line_width_fit": 2.4,
    "marker_size": 5.0,
    "marker_alpha": 0.55,
    "marker_edgecolor": "white",
    "marker_edgewidth": 0.35,
    "line_alpha": 0.95,
    "color_blue": "#377eb8",
    "color_red": "#e41a1c",
    "grid_alpha": 0.2,
    "grid_linestyle": ":",
    "grid_linewidth": 0.5,
    "spine_width": 0.8,
    "spine_color": "#4d4d4d",
}

FIGSIZE: tuple[float, float] = (4.0, 3.0)


def _configure_rcparams() -> None:
    mpl.rcParams["figure.dpi"] = 300
    mpl.rcParams["savefig.dpi"] = 300
    apply_font_embedding(42)
    plt.rcParams.update(
        {
            **STYLE_RCPARAMS,
            "axes.linewidth": float(STYLE["spine_width"]),
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "mathtext.fontset": "cm",
            "text.latex.preamble": r"\usepackage{amsmath}",
        }
    )


def _apply_axis_style(ax) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(float(STYLE["spine_width"]))
    ax.spines["bottom"].set_linewidth(float(STYLE["spine_width"]))
    ax.spines["left"].set_color(str(STYLE["spine_color"]))
    ax.spines["bottom"].set_color(str(STYLE["spine_color"]))
    ax.tick_params(axis="both", which="major", length=4, width=0.8, direction="out")
    ax.tick_params(axis="both", which="minor", length=0)
    ax.grid(
        True,
        alpha=float(STYLE["grid_alpha"]),
        linestyle=str(STYLE["grid_linestyle"]),
        linewidth=float(STYLE["grid_linewidth"]),
        zorder=0,
    )
    ax.set_axisbelow(True)


def _apply_legend_style(ax, *, loc: str = "best", handles=None, labels=None) -> None:
    kwargs = dict(
        loc=loc,
        frameon=True,
        framealpha=0.85,
        edgecolor="#c0c0c0",
        fancybox=False,
        fontsize=float(STYLE_RCPARAMS["legend.fontsize"]),
        handlelength=1.6,
        handletextpad=0.5,
        borderpad=0.4,
    )
    if handles is None or labels is None:
        ax.legend(**kwargs)
    else:
        ax.legend(handles, labels, **kwargs)


def _save_figure_both(fig, out_base: str, *, tight: bool = True) -> None:
    if not tight:
        fig.savefig(out_base + ".pdf", dpi=300, facecolor="white", edgecolor="none")
        fig.savefig(out_base + ".png", dpi=300, facecolor="white", edgecolor="none")
        return

    # Preserve the original figure size while using bbox_inches='tight' and pad_inches=0.05
    # by adding an invisible inset artist that expands the tight bbox back to the full canvas.
    pad = 0.05
    w_in, h_in = fig.get_size_inches()
    fx = pad / float(w_in) if w_in else 0.0
    fy = pad / float(h_in) if h_in else 0.0
    rect = Rectangle(
        (fx, fy),
        max(0.0, 1.0 - 2.0 * fx),
        max(0.0, 1.0 - 2.0 * fy),
        transform=fig.transFigure,
        fill=False,
        linewidth=0.0,
        edgecolor="none",
    )
    fig.add_artist(rect)
    try:
        fig.savefig(
            out_base + ".pdf",
            dpi=300,
            bbox_inches="tight",
            pad_inches=pad,
            facecolor="white",
            edgecolor="none",
        )
        fig.savefig(
            out_base + ".png",
            dpi=300,
            bbox_inches="tight",
            pad_inches=pad,
            facecolor="white",
            edgecolor="none",
        )
    finally:
        try:
            rect.remove()
        except Exception:
            pass


# -----------------------------------------------------------------------------
# Figure 7a (GPQA diamond frontier compare)
# -----------------------------------------------------------------------------


def _pinball_loss(y: np.ndarray, yhat: np.ndarray, *, tau: float, kappa: float) -> float:
    y = np.asarray(y, float)
    yhat = np.asarray(yhat, float)
    mask = np.isfinite(y) & np.isfinite(yhat)
    if not np.any(mask):
        return float("nan")
    r = y[mask] - yhat[mask]
    vals = np.logaddexp(0.0, kappa * r) / kappa + (tau - 1.0) * r
    return float(np.mean(vals))


def _load_points_csv(path: str) -> Tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(path)
    if "training_compute_flop" not in df.columns or "score" not in df.columns:
        raise ValueError(f"Missing required columns in {path}: expected training_compute_flop, score")
    c_abs = pd.to_numeric(df["training_compute_flop"], errors="coerce").to_numpy(float)
    y = pd.to_numeric(df["score"], errors="coerce").to_numpy(float)
    return c_abs, y


def _load_sigmoid_curve_csv(path: str, *, x_tick_multiplier: float) -> Tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(path)
    if "training_compute_flop" not in df.columns or "y_hat" not in df.columns:
        raise ValueError(f"Missing required columns in {path}: expected training_compute_flop, y_hat")
    x_abs = pd.to_numeric(df["training_compute_flop"], errors="coerce").to_numpy(float)
    y_hat = pd.to_numeric(df["y_hat"], errors="coerce").to_numpy(float)
    x = x_abs / float(x_tick_multiplier)
    return x, y_hat


def plot_figure7a(
    *,
    out_dir: str,
    points_csv: str,
    curve_csv: str,
    seed: int = 0,
    tau: float = 0.98,
    kappa_final: float = 50.0,
    lambda_b: float = 1e-4,
    frontier_fit_mode: str = "quantile_per_point",
    bins: int = 10,
    min_bin_size: int = 5,
    train_frac: float = 0.7,
    x_tick_multiplier: float = 1e21,
    ispline_lambda_beta: float = 1e-4,
    ispline_knots: int = 3,
    ispline_random: int = 20,
    ispline_maxiter: int = 700,
) -> None:
    c_abs, y_all = _load_points_csv(points_csv)
    c = c_abs / float(x_tick_multiplier)
    z = np.log10(c)
    mask = np.isfinite(c) & (c > 0) & np.isfinite(z) & np.isfinite(y_all)
    c = c[mask]
    z = z[mask]
    y = y_all[mask]
    n_total = int(y.size)
    if n_total < 12:
        raise RuntimeError(f"Too few points after filtering (n={n_total}).")

    rng = np.random.default_rng(int(seed))
    perm = rng.permutation(n_total)
    n_train = int(round(float(train_frac) * n_total))
    n_train = int(np.clip(n_train, 10, n_total - 2))
    tr_idx = perm[:n_train]
    va_idx = perm[n_train:]

    c_tr, z_tr, y_tr = c[tr_idx], z[tr_idx], y[tr_idx]
    c_va, z_va, y_va = c[va_idx], z[va_idx], y[va_idx]

    z_lo, z_hi = float(np.min(z_tr)), float(np.max(z_tr))
    overlap = (z_va >= z_lo) & (z_va <= z_hi)
    c_va_o, z_va_o, y_va_o = c_va[overlap], z_va[overlap], y_va[overlap]
    if int(y_va_o.size) < 2:
        c_va_o, z_va_o, y_va_o = c_va, z_va, y_va

    # Train-fit sigmoid (for loss computation); plot curve comes from precomputed curve CSV.
    xs_fit, ys_fit = fit_sigmoid_predictor(
        c_tr,
        y_tr,
        tau=float(tau),
        frontier_fit_mode=str(frontier_fit_mode),
        bins_for_fit=int(bins),
        min_bin_size_for_fit=int(min_bin_size),
        kappa_final=float(kappa_final),
        lambda_b=float(lambda_b),
    )
    yhat_va_sig = interpolate_curve(xs_fit, ys_fit, c_va_o)
    L_va_sig = _pinball_loss(y_va_o, yhat_va_sig, tau=float(tau), kappa=float(kappa_final))

    # I-spline fit on train
    fit_is = iso.fit_ispline_enhanced(
        z=z_tr,
        y=y_tr,
        tau=float(tau),
        kappa_final=float(kappa_final),
        lambda_beta=float(ispline_lambda_beta),
        n_knot_grid=int(ispline_knots),
        n_random=int(ispline_random),
        seed=int(seed),
        maxiter=int(ispline_maxiter),
    )
    yhat_va_is = (
        iso._predict_from_params(z_va_o, fit_is.params, fit_is.edges)
        if fit_is.params.size and fit_is.edges.size
        else np.full_like(y_va_o, np.nan, dtype=float)
    )
    L_va_is = _pinball_loss(y_va_o, yhat_va_is, tau=float(tau), kappa=float(kappa_final))

    # Preserve legacy rendering used by the published artifact.
    from skill_frontier.plotting.configs import frontier_period4_triptych as _legacy_triptych_cfg  # type: ignore
    _ = _legacy_triptych_cfg

    # Curves to plot
    xs_curve_plot, ys_curve_plot = _load_sigmoid_curve_csv(curve_csv, x_tick_multiplier=float(x_tick_multiplier))
    z_grid = np.linspace(float(np.min(z)), float(np.max(z)), 300)
    c_grid = np.power(10.0, z_grid)
    y_grid_is = (
        iso._predict_from_params(z_grid, fit_is.params, fit_is.edges)
        if fit_is.params.size and fit_is.edges.size
        else np.full_like(z_grid, np.nan, dtype=float)
    )

    fig, ax = plt.subplots(figsize=FIGSIZE, constrained_layout=True)

    ax.scatter(
        c,
        y,
        s=float(STYLE["marker_size"]) ** 2,
        alpha=float(STYLE["marker_alpha"]),
        color=str(STYLE["spine_color"]),
        marker="o",
        edgecolors=str(STYLE["marker_edgecolor"]),
        linewidths=float(STYLE["marker_edgewidth"]),
        rasterized=True,
        zorder=2,
    )

    # Plot in the same semantic colors as the paper: sigmoid (red, solid), I-spline (blue, dashed).
    ax.plot(
        xs_curve_plot,
        ys_curve_plot,
        color=str(STYLE["color_red"]),
        linewidth=float(STYLE["line_width_fit"]),
        alpha=float(STYLE["line_alpha"]),
        linestyle="-",
        label=f"Sigmoid (loss = {float(L_va_sig):.4f})",
        zorder=4,
    )
    ax.plot(
        c_grid,
        y_grid_is,
        color=str(STYLE["color_blue"]),
        linewidth=float(STYLE["line_width_fit"]),
        alpha=float(STYLE["line_alpha"]),
        linestyle="--",
        label=f"I-spline (loss = {float(L_va_is):.4f})",
        zorder=4,
    )

    ax.set_xscale("log")
    ax.set_xlabel(PRETRAINING_COMPUTE_FLOPS_LABEL, fontweight="bold")
    apply_pretraining_compute_tick_multiplier(ax, multiplier=float(x_tick_multiplier))
    ax.set_ylabel("GPQA Diamond", fontweight="bold")

    try:
        import matplotlib.ticker as mticker  # type: ignore

        ax.yaxis.set_minor_locator(mticker.AutoMinorLocator(2))
        ax.xaxis.set_minor_locator(mticker.LogLocator(base=10.0, subs=(0.2, 0.4, 0.6, 0.8)))
    except Exception:
        pass

    # Match the original plot's y-limits computation (uses the train-fit curve).
    y_min = float(np.nanmin([np.nanmin(y), np.nanmin(ys_fit)]))
    y_max = float(np.nanmax([np.nanmax(y), np.nanmax(ys_fit)]))
    if np.isfinite(y_min) and np.isfinite(y_max) and y_min != y_max:
        pad_y = 0.02 * (y_max - y_min)
        ax.set_ylim(y_min - pad_y, y_max + pad_y)

    _apply_axis_style(ax)
    _apply_legend_style(ax, loc="best")
    out_base = os.path.join(out_dir, "figure7a_main_paper")
    _save_figure_both(fig, out_base, tight=True)
    plt.close(fig)


# -----------------------------------------------------------------------------
# Figure 7b (MATH-500 vs AIME 2025)
# -----------------------------------------------------------------------------


def _format_cutoff_for_legend(ts: "pd.Timestamp") -> str:
    month = ts.strftime("%b") + "."
    day = int(ts.day)
    if 11 <= (day % 100) <= 13:
        suffix = "th"
    else:
        suffix = {1: "st", 2: "nd", 3: "rd"}.get(day % 10, "th")
    return f"{month} {day}{suffix} {int(ts.year)}"


def _logit(p: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    p = np.clip(p, eps, 1.0 - eps)
    return np.log(p / (1.0 - p))


def _sigmoid(t: np.ndarray) -> np.ndarray:
    t = np.asarray(t, float)
    t = np.clip(t, -60.0, 60.0)
    return 1.0 / (1.0 + np.exp(-t))


def _fit_ols(design: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    coef, *_ = np.linalg.lstsq(design, y, rcond=None)
    resid = y - (design @ coef)
    return coef, resid


def _predict_linear(coef: np.ndarray, x: np.ndarray) -> np.ndarray:
    return coef[0] + coef[1] * x


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


def _format_logit_to_logit_legend(a: float, b: float) -> str:
    sign = "+" if b >= 0 else "-"
    b_abs = abs(float(b))
    return (
        rf"$\operatorname{{logit}}\!\left(\frac{{\hat{{y}}}}{{100}}\right)"
        rf" = {float(a):.2f} {sign} {b_abs:.2f}\,"
        rf"\operatorname{{logit}}\!\left(\frac{{x}}{{100}}\right)$"
    )


def plot_figure7b(
    *,
    out_dir: str,
    aime_csv: str,
    math500_csv: str,
    release_cutoff: str = "2025-02-06",
) -> None:
    df_aime = pd.read_csv(aime_csv)
    df_math = pd.read_csv(math500_csv)
    df = df_math[["model_id", "model_name", "math_500_pct"]].merge(
        df_aime[["model_id", "aime_2025_pct", "release_date"]], on="model_id", how="inner"
    )
    df["math_500_pct"] = pd.to_numeric(df["math_500_pct"], errors="coerce")
    df["aime_2025_pct"] = pd.to_numeric(df["aime_2025_pct"], errors="coerce")
    df["release_date"] = pd.to_datetime(df["release_date"], errors="coerce")
    df = df.dropna(subset=["math_500_pct", "aime_2025_pct"]).copy()
    if df.empty:
        raise RuntimeError("No rows after merging MATH-500 and AIME 2025 tables.")

    cutoff = pd.Timestamp(str(release_cutoff))
    cutoff_label = _format_cutoff_for_legend(cutoff)
    is_pre_release = df["release_date"].notna() & (df["release_date"] < cutoff)
    pre_mask = is_pre_release.to_numpy()
    post_mask = (~is_pre_release).to_numpy()

    fig, ax = plt.subplots(figsize=FIGSIZE, constrained_layout=True)

    # Logit-logit axes: plot in log-odds space so the fitted relationship is linear.
    # Use a slightly larger epsilon to avoid extreme values for 0% / 100% points.
    eps = 1e-3

    # Plot red first, then blue so the "Before" points remain visible on top of overlaps.
    ax.scatter(
        _logit(df.loc[post_mask, "math_500_pct"].to_numpy() / 100.0, eps=eps),
        _logit(df.loc[post_mask, "aime_2025_pct"].to_numpy() / 100.0, eps=eps),
        s=float(STYLE["marker_size"]) ** 2,
        alpha=float(STYLE["marker_alpha"]),
        c=str(STYLE["color_red"]),
        edgecolors=str(STYLE["marker_edgecolor"]),
        linewidths=float(STYLE["marker_edgewidth"]),
        label=f"After {cutoff_label}",
        rasterized=True,
        zorder=2,
    )
    ax.scatter(
        _logit(df.loc[pre_mask, "math_500_pct"].to_numpy() / 100.0, eps=eps),
        _logit(df.loc[pre_mask, "aime_2025_pct"].to_numpy() / 100.0, eps=eps),
        s=float(STYLE["marker_size"]) ** 2,
        alpha=float(STYLE["marker_alpha"]),
        c=str(STYLE["color_blue"]),
        edgecolors=str(STYLE["marker_edgecolor"]),
        linewidths=float(STYLE["marker_edgewidth"]),
        label=f"Before {cutoff_label}",
        rasterized=True,
        zorder=3,
    )

    x_post = df.loc[post_mask, "math_500_pct"].to_numpy()
    y_post = df.loc[post_mask, "aime_2025_pct"].to_numpy()
    m_post = np.isfinite(x_post) & np.isfinite(y_post)
    if int(np.sum(m_post)) >= 6:
        coef_fit = _fit_logit_to_logit_linear(x_pct=x_post[m_post], y_pct=y_post[m_post], eps=eps)
        X_grid = np.linspace(
            float(np.min(_logit(x_post[m_post] / 100.0, eps=eps))),
            float(np.max(_logit(x_post[m_post] / 100.0, eps=eps))),
            250,
        )
        y_grid = _predict_linear(coef_fit, X_grid)
        ax.plot(
            X_grid,
            y_grid,
            color=str(STYLE["color_red"]),
            linewidth=float(STYLE["line_width_fit"]),
            alpha=float(STYLE["line_alpha"]),
            linestyle="-",
            label=_format_logit_to_logit_legend(coef_fit[0], coef_fit[1]),
            zorder=4,
        )

    ax.set_xlabel(r"$\operatorname{logit}(\mathrm{MATH{-}500}\times 0.01)$", fontweight="bold")
    ax.set_ylabel(r"$\operatorname{logit}(\mathrm{AIME}\,2025\times 0.01)$", fontweight="bold")
    _apply_axis_style(ax)

    # Keep legend order consistent: Before, After, then fit.
    handles, labels = ax.get_legend_handles_labels()
    wanted = [f"Before {cutoff_label}", f"After {cutoff_label}"]
    ordered: list[tuple[object, str]] = []
    for w in wanted:
        for h, lab in zip(handles, labels):
            if lab == w:
                ordered.append((h, lab))
                break
    for h, lab in zip(handles, labels):
        if lab not in wanted:
            ordered.append((h, lab))
    _apply_legend_style(ax, loc="best", handles=[h for h, _ in ordered], labels=[lab for _, lab in ordered])

    out_base = os.path.join(out_dir, "figure7b_main_paper")
    _save_figure_both(fig, out_base, tight=False)
    plt.close(fig)


def main(argv: Optional[Sequence[str]] = None) -> None:
    p = argparse.ArgumentParser(description="Generate Figure 7a/7b main-paper plots.")
    p.add_argument("--out_dir", default=os.path.join("outputs", "figures_main_paper"))
    p.add_argument("--only", choices=("7a", "7b", "both"), default="both")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--release_cutoff", default="2025-02-06")
    p.add_argument(
        "--gpqa_points_csv",
        default=os.path.join("outputs_epoch_ai_benchmarks_runs", "sigmoid", "no_split", "points", "GPQA_diamond.csv"),
    )
    p.add_argument(
        "--gpqa_curve_csv",
        default=os.path.join("outputs_epoch_ai_benchmarks_runs", "sigmoid", "no_split", "curves", "GPQA_diamond.csv"),
    )
    p.add_argument("--aime_csv", default=os.path.join("tables", "artificial_analysis", "aime-2025.csv"))
    p.add_argument("--math500_csv", default=os.path.join("tables", "artificial_analysis", "math-500.csv"))
    args = p.parse_args(list(argv) if argv is not None else None)

    _configure_rcparams()
    Path(str(args.out_dir)).mkdir(parents=True, exist_ok=True)

    if args.only in ("7a", "both"):
        plot_figure7a(out_dir=str(args.out_dir), points_csv=str(args.gpqa_points_csv), curve_csv=str(args.gpqa_curve_csv), seed=int(args.seed))
        print(f"[ok] wrote {os.path.join(str(args.out_dir), 'figure7a_main_paper.pdf')}|.png")
    if args.only in ("7b", "both"):
        plot_figure7b(
            out_dir=str(args.out_dir),
            aime_csv=str(args.aime_csv),
            math500_csv=str(args.math500_csv),
            release_cutoff=str(args.release_cutoff),
        )
        print(f"[ok] wrote {os.path.join(str(args.out_dir), 'figure7b_main_paper.pdf')}|.png")


if __name__ == "__main__":
    main()

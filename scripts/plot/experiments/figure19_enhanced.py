#!/usr/bin/env python3
"""
Figure 19 (enhanced): In-period κ/λ tuning + OOD metric comparison.

Inputs:
  - outputs/sensitivity/kappa_lambda_cv/results_raw.csv
  - outputs/sensitivity/kappa_lambda_cv/summary_by_task_period.csv

Outputs (overwrites):
  - outputs/sensitivity/kappa_lambda_cv/plots/figure19_enhanced.pdf
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

try:
    import matplotlib.pyplot as plt  # type: ignore
    from matplotlib.colors import BoundaryNorm, ListedColormap  # type: ignore
    import matplotlib.gridspec as gridspec  # type: ignore
except Exception as e:  # pragma: no cover
    raise SystemExit("This script requires matplotlib.") from e

try:
    import seaborn as sns  # type: ignore
except Exception as e:  # pragma: no cover
    raise SystemExit("This script requires seaborn.") from e


REPO_ROOT = Path(__file__).resolve().parents[3]
CV_DIR = REPO_ROOT / "outputs" / "sensitivity" / "kappa_lambda_cv"
PLOTS_DIR = CV_DIR / "plots"
RAW_CSV = CV_DIR / "results_raw.csv"
SUMMARY_CSV = CV_DIR / "summary_by_task_period.csv"

OUT_PDF = PLOTS_DIR / "figure19_enhanced.pdf"


TASKS_ORDER = [
    "BBH Raw",
    "GPQA Raw",
    "IFEval Raw",
    "MATH Lvl 5 Raw",
    "MMLU-PRO Raw",
    "MUSR Raw",
]
TASK_LABELS = {
    "BBH Raw": "BBH",
    "GPQA Raw": "GPQA",
    "IFEval Raw": "IFEval",
    "MATH Lvl 5 Raw": "MATH Lvl 5",
    "MMLU-PRO Raw": "MMLU-PRO",
    "MUSR Raw": "MUSR",
}
PERIODS = [1, 2, 3]

TITLE_BBOX = dict(
    boxstyle="round,pad=0.35",
    facecolor="#f0f0f0",
    edgecolor="#cccccc",
    linewidth=0.8,
    alpha=0.95,
)


def _configure_matplotlib() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman"],
            "font.size": 11,
            "axes.labelsize": 13,
            "axes.titlesize": 14,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "mathtext.fontset": "cm",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def _set_boxed_title(ax, title: str) -> None:
    ax.set_title(title, fontweight="bold", pad=8).set_bbox(TITLE_BBOX)


def _style_line_ax(ax) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.8)
    ax.spines["bottom"].set_linewidth(0.8)
    ax.tick_params(axis="both", which="major", length=4, width=0.8, direction="out")
    ax.grid(True, alpha=0.25, linestyle=":", linewidth=0.5, zorder=0)


def _discrete_blues_colormap(levels: list[int]) -> tuple[ListedColormap, BoundaryNorm, list[float]]:
    levels_sorted = sorted({int(v) for v in levels})
    if len(levels_sorted) == 1:
        bounds = [levels_sorted[0] - 1.0, levels_sorted[0] + 1.0]
    else:
        mids = [(levels_sorted[i] + levels_sorted[i + 1]) / 2.0 for i in range(len(levels_sorted) - 1)]
        bounds = [levels_sorted[0] - (mids[0] - levels_sorted[0]), *mids, levels_sorted[-1] + (levels_sorted[-1] - mids[-1])]

    cmap = ListedColormap(plt.get_cmap("Blues")(np.linspace(0.35, 0.9, num=len(levels_sorted))))
    norm = BoundaryNorm(bounds, cmap.N)
    return cmap, norm, [float(v) for v in levels_sorted]


def _load_selection_tables() -> tuple[pd.DataFrame, pd.DataFrame]:
    raw = pd.read_csv(RAW_CSV)
    raw = raw[raw["task"].isin(TASKS_ORDER)].copy()
    raw["k"] = pd.to_numeric(raw["k"], errors="coerce").astype("Int64")
    raw["kappa_selected"] = pd.to_numeric(raw["kappa_selected"], errors="coerce")
    raw["lambda_selected"] = pd.to_numeric(raw["lambda_selected"], errors="coerce")

    kappa_med = raw.groupby(["task", "k"], dropna=False)["kappa_selected"].median().unstack("k")
    log10_lambda = np.log10(raw["lambda_selected"].to_numpy(dtype=float))
    raw["log10_lambda_selected"] = log10_lambda
    lambda_med = raw.groupby(["task", "k"], dropna=False)["log10_lambda_selected"].median().unstack("k")

    kappa_med = kappa_med.reindex(index=TASKS_ORDER, columns=PERIODS)
    lambda_med = lambda_med.reindex(index=TASKS_ORDER, columns=PERIODS)

    kappa_med.index = [TASK_LABELS.get(t, t) for t in kappa_med.index]
    lambda_med.index = [TASK_LABELS.get(t, t) for t in lambda_med.index]
    kappa_med.columns = [f"k={int(k)}" for k in kappa_med.columns]
    lambda_med.columns = [f"k={int(k)}" for k in lambda_med.columns]
    return kappa_med, lambda_med


def _load_oos_lines() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    summary = pd.read_csv(SUMMARY_CSV)
    summary = summary[summary["task"].isin(TASKS_ORDER)].copy()
    summary["k"] = pd.to_numeric(summary["k"], errors="coerce").astype(int)

    pinball_fixed = summary.groupby("k")["baseline_oos_pinball"].mean().reindex(PERIODS).to_numpy(dtype=float)
    pinball_cv = summary.groupby("k")["oos_pinball_mean"].mean().reindex(PERIODS).to_numpy(dtype=float)

    calib_fixed = summary.groupby("k")["baseline_oos_calib_abs"].mean().reindex(PERIODS).to_numpy(dtype=float)
    calib_cv = summary.groupby("k")["oos_calib_abs_mean"].mean().reindex(PERIODS).to_numpy(dtype=float)

    x = np.asarray(PERIODS, dtype=float)
    return x, pinball_fixed, pinball_cv, calib_fixed, calib_cv


def main() -> None:
    sns.set_theme(style="white")
    _configure_matplotlib()

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    kappa_tbl, lambda_tbl = _load_selection_tables()
    x_k, pin_fixed, pin_cv, calib_fixed, calib_cv = _load_oos_lines()

    # Wider and with more space between the two line subplots to avoid overlap.
    fig = plt.figure(figsize=(14.0, 3.2), dpi=300)
    gs = gridspec.GridSpec(
        1,
        2,
        figure=fig,
        width_ratios=[1.15, 1.05],
        left=0.07,
        right=0.98,
        bottom=0.15,
        top=0.90,
        wspace=0.34,
    )

    gs_left = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[0], wspace=0.40)
    ax_kappa = fig.add_subplot(gs_left[0])
    ax_lambda = fig.add_subplot(gs_left[1])

    gs_right = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[1], wspace=0.62)
    ax_pinball = fig.add_subplot(gs_right[0])
    ax_calib = fig.add_subplot(gs_right[1])

    # (a) Heatmaps: selected κ and log10(λ)
    kappa_values = [int(v) for v in np.unique(kappa_tbl.to_numpy(dtype=float)[np.isfinite(kappa_tbl.to_numpy(dtype=float))]).tolist()]
    # Ensure stable ordering / presence of expected values
    for v in (20, 50, 100, 200):
        if v not in kappa_values:
            kappa_values.append(v)
    cmap_kappa, norm_kappa, kappa_ticks = _discrete_blues_colormap(kappa_values)

    sns.heatmap(
        kappa_tbl,
        ax=ax_kappa,
        cmap=cmap_kappa,
        norm=norm_kappa,
        annot=True,
        fmt=".0f",
        annot_kws={"fontsize": 10, "fontweight": "bold"},
        cbar_kws={"label": "", "shrink": 0.8, "aspect": 20, "pad": 0.02},
        linewidths=0.5,
        linecolor="white",
        square=False,
    )
    _set_boxed_title(ax_kappa, r"(a) Selected $\kappa$ (median)")
    ax_kappa.set_xlabel(r"Period $\mathcal{P}_k$", fontweight="bold")
    ax_kappa.set_ylabel("Task", fontweight="bold")
    ax_kappa.tick_params(axis="x", rotation=0)
    ax_kappa.tick_params(axis="y", rotation=0)
    cbar_kappa = ax_kappa.collections[0].colorbar
    cbar_kappa.set_ticks(kappa_ticks)
    cbar_kappa.ax.tick_params(labelsize=8)
    cbar_kappa.outline.set_linewidth(0.8)

    sns.heatmap(
        lambda_tbl,
        ax=ax_lambda,
        cmap="Blues",
        annot=True,
        fmt=".2f",
        annot_kws={"fontsize": 10, "fontweight": "bold"},
        cbar_kws={"label": "", "shrink": 0.8, "aspect": 20, "pad": 0.02},
        linewidths=0.5,
        linecolor="white",
        square=False,
    )
    _set_boxed_title(ax_lambda, r"Selected $\log_{10}(\lambda)$ (median)")
    ax_lambda.set_xlabel(r"Period $\mathcal{P}_k$", fontweight="bold")
    ax_lambda.set_ylabel("")
    ax_lambda.set_yticklabels([])
    ax_lambda.tick_params(axis="x", rotation=0)
    cbar_lambda = ax_lambda.collections[0].colorbar
    cbar_lambda.ax.tick_params(labelsize=8)
    cbar_lambda.outline.set_linewidth(0.8)

    # (b) Line plots: mean OOD metrics over tasks
    line_color = "#377eb8"
    line_cfg = {
        "linewidth": 2.0,
        "markersize": 6.0,
        "alpha": 0.9,
        "color": line_color,
    }

    ax_pinball.plot(x_k, pin_fixed, linestyle="-", marker="o", label="fixed", **line_cfg)
    ax_pinball.plot(x_k, pin_cv, linestyle="--", marker="o", label="CV-selected", **line_cfg)
    _set_boxed_title(ax_pinball, "(b) OOD pinball loss")
    ax_pinball.set_xlabel(r"Period $k$", fontweight="bold")
    ax_pinball.set_ylabel("OOD pinball loss\n(mean over tasks)", fontweight="bold")
    ax_pinball.set_xticks(PERIODS)
    _style_line_ax(ax_pinball)
    ax_pinball.legend(
        loc="best",
        frameon=True,
        framealpha=0.9,
        edgecolor="gray",
        fancybox=False,
        fontsize=9,
        handlelength=2.0,
        handletextpad=0.6,
    )

    ax_calib.plot(x_k, calib_fixed, linestyle="-", marker="o", label="fixed", **line_cfg)
    ax_calib.plot(x_k, calib_cv, linestyle="--", marker="o", label="CV-selected", **line_cfg)
    _set_boxed_title(ax_calib, r"OOD calibration $|\hat{\tau}-\tau|$")
    ax_calib.set_xlabel(r"Period $k$", fontweight="bold")
    ax_calib.set_ylabel("OOD calibration error\n(mean over tasks)", fontweight="bold")
    ax_calib.set_xticks(PERIODS)
    _style_line_ax(ax_calib)
    ax_calib.legend(
        loc="best",
        frameon=True,
        framealpha=0.9,
        edgecolor="gray",
        fancybox=False,
        fontsize=9,
        handlelength=2.0,
        handletextpad=0.6,
    )

    fig.savefig(OUT_PDF, dpi=300, facecolor="white", edgecolor="none")
    plt.close(fig)

    print(f"Wrote: {OUT_PDF}")


if __name__ == "__main__":
    main()

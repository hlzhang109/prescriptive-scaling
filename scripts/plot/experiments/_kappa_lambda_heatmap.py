"""Shared renderer for kappa/lambda sweep heatmap figures."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

try:
    import matplotlib.pyplot as plt  # type: ignore
except Exception as e:  # pragma: no cover
    raise SystemExit("This script requires matplotlib.") from e

try:
    import seaborn as sns  # type: ignore
except Exception as e:  # pragma: no cover
    raise SystemExit("This script requires seaborn.") from e


REPO_ROOT = Path(__file__).resolve().parents[3]
SWEEP_DIR = REPO_ROOT / "outputs" / "sensitivity" / "kappa_lambda_sweep"
PLOTS_DIR = SWEEP_DIR / "plots"
SWEEP_CSV = SWEEP_DIR / "sweep_results.csv"

KAPPA_ORDER = [20.0, 50.0, 100.0, 200.0, 1000.0]
LOG10_LAMBDA_ORDER = [-4.0, -3.0, -2.0, -1.0]

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


def _pivot_metric(df: pd.DataFrame, *, task: str, k: int, metric: str) -> pd.DataFrame:
    sub = df[(df["task"] == task) & (df["k"] == int(k))].copy()
    sub["kappa_train"] = pd.to_numeric(sub["kappa_train"], errors="coerce")
    sub["lambda_b"] = pd.to_numeric(sub["lambda_b"], errors="coerce")
    sub["log10_lambda"] = np.log10(sub["lambda_b"].to_numpy(dtype=float))

    pt = sub.pivot_table(index="kappa_train", columns="log10_lambda", values=metric, aggfunc="mean")
    pt = pt.reindex(index=KAPPA_ORDER, columns=LOG10_LAMBDA_ORDER)
    pt.index = [int(v) for v in pt.index]
    pt.columns = [int(v) for v in pt.columns]
    return pt


def _style_colorbar(ax) -> None:
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=9)
    cbar.outline.set_linewidth(0.8)


def render(*, metric: str, suptitle: str, out_pdf_name: str) -> None:
    sns.set_theme(style="white")
    _configure_matplotlib()

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(SWEEP_CSV)
    out_pdf = PLOTS_DIR / out_pdf_name

    panels = [
        ("BBH Raw", 1, "(a) BBH, k=1."),
        ("BBH Raw", 3, "(b) BBH, k=3."),
        ("MATH Lvl 5 Raw", 1, "(c) MATH Lvl 5, k=1."),
        ("MATH Lvl 5 Raw", 3, "(d) MATH Lvl 5, k=3."),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(10.0, 6.4), dpi=300)
    plt.subplots_adjust(left=0.10, right=0.95, bottom=0.10, top=0.90, wspace=0.35, hspace=0.40)

    heatmap_style = {
        "cmap": "Blues",
        "annot": True,
        "fmt": ".2f",
        "annot_kws": {"fontsize": 9.0, "fontweight": "normal"},
        "cbar_kws": {"shrink": 0.9, "aspect": 15, "pad": 0.02},
        "linewidths": 0.5,
        "linecolor": "white",
        "square": False,
    }

    for ax, (task, k, title) in zip(axes.ravel(), panels):
        data = _pivot_metric(df, task=task, k=k, metric=metric)
        sns.heatmap(data, ax=ax, **heatmap_style)
        _set_boxed_title(ax, title)
        ax.set_xlabel(r"$\log_{10}(\lambda)$", fontweight="bold")
        ax.set_ylabel(r"$\kappa_{\mathrm{train}}$", fontweight="bold")
        ax.tick_params(axis="x", rotation=0)
        ax.tick_params(axis="y", rotation=0)
        _style_colorbar(ax)

    supt = fig.suptitle(suptitle, fontweight="bold", fontsize=14, y=0.985)
    supt.set_bbox(TITLE_BBOX)

    fig.savefig(out_pdf, dpi=300, facecolor="white", edgecolor="none")
    plt.close(fig)
    print(f"Wrote: {out_pdf}")


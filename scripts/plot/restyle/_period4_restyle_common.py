"""Shared logic for restyling period4 single-k triptych plots."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np

try:
    import pandas as pd  # type: ignore
except Exception as e:  # pragma: no cover
    raise SystemExit("This script requires pandas.") from e

try:
    import matplotlib.pyplot as plt  # type: ignore
    from matplotlib.lines import Line2D  # type: ignore
except Exception as e:  # pragma: no cover
    raise SystemExit("This script requires matplotlib.") from e

try:
    from skill_frontier.io.csv_utils import maybe_scale_task_values  # type: ignore
except Exception:  # pragma: no cover
    def maybe_scale_task_values(values: np.ndarray) -> np.ndarray:  # type: ignore
        arr = np.asarray(values, dtype=float)
        finite = arr[np.isfinite(arr)]
        if finite.size == 0:
            return arr
        try:
            p95 = float(np.nanpercentile(finite, 95))
        except Exception:
            p95 = float(np.nanmax(finite))
        if (p95 > 1.0 + 1e-9) and (p95 <= 100.0 + 1e-9):
            return arr / 100.0
        return arr

try:
    from scripts.smooth_single_skill_frontier import fit_sigmoid_frontier  # type: ignore
except Exception as e:  # pragma: no cover
    raise SystemExit("Could not import scripts.smooth_single_skill_frontier.fit_sigmoid_frontier") from e

from scripts.plot.restyle.restyle_period4_utils import assign_period_index, parse_year_month_simple  # type: ignore
from scripts.plot.restyle.restyle_curve_utils import iter_plot_slugs, task_from_curve  # type: ignore


FIG1_RCPARAMS = {
    "font.family": "serif",
    "font.sans-serif": ["Times New Roman"],
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
    "font.size": 10,
    "axes.labelsize": 13,
    "axes.titlesize": 13,
    "axes.labelweight": "bold",
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 9,
    "mathtext.fontset": "cm",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
}

STYLE = {
    "line_width_fit": 2.5,
    "marker_size": 4.2,
    "marker_alpha": 0.55,
    "marker_edgecolor": "white",
    "marker_edgewidth": 0.3,
    "line_alpha": 0.95,
    "train_points_color": "#377eb8",
    "val_points_color": "#e41a1c",
    "fit_train_color": "#377eb8",
    "fit_val_color": "#e41a1c",
    "grid_alpha": 0.2,
    "grid_linestyle": ":",
    "grid_linewidth": 0.6,
    "grid_color": "gray",
    "spine_width": 0.8,
    "spine_color": "#4d4d4d",
}


@dataclass(frozen=True)
class RestyleConfig:
    period_bounds: List[Tuple[str, Tuple[int, int], Tuple[int, int]]]
    period_splits_single: List[Dict[str, str | List[str]]]
    date_col: str
    x_mode: str  # "compute" or "size"
    x_scale_to_plot: float
    x_ticks: List[float]
    x_label: str
    x_label_y: float
    bottom: float
    period_symbol: str = "k"


@dataclass(frozen=True)
class PanelData:
    train_x_raw: np.ndarray
    train_y: np.ndarray
    val_x_raw: np.ndarray
    val_y: np.ndarray
    curve_train_x_raw: np.ndarray
    curve_train_y: np.ndarray
    curve_val_x_raw: np.ndarray
    curve_val_y: np.ndarray


def _load_curve_xy(curve_csv: Path) -> Tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(curve_csv, usecols=["x", "y_hat"])
    x = pd.to_numeric(df["x"], errors="coerce").to_numpy(dtype=float)
    y = pd.to_numeric(df["y_hat"], errors="coerce").to_numpy(dtype=float)
    m = np.isfinite(x) & (x > 0) & np.isfinite(y)
    return x[m], y[m]


def _prepare_dataframe(csv_path: Path, *, task_cols: Sequence[str], cfg: RestyleConfig) -> pd.DataFrame:
    if cfg.x_mode == "compute":
        usecols = [cfg.date_col, "Pretraining tokens (T)", "#Params (B)", *list(task_cols)]
    elif cfg.x_mode == "size":
        usecols = [cfg.date_col, "#Params (B)", *list(task_cols)]
    else:
        raise ValueError(f"Unsupported x_mode={cfg.x_mode!r}")

    df = pd.read_csv(csv_path, usecols=usecols)
    df["ym"] = df[cfg.date_col].apply(parse_year_month_simple)
    df["period_idx"] = df["ym"].apply(lambda ym: assign_period_index(ym, cfg.period_bounds)).astype(int)

    if cfg.x_mode == "compute":
        tokens_t = pd.to_numeric(df["Pretraining tokens (T)"], errors="coerce")
        params_b = pd.to_numeric(df["#Params (B)"], errors="coerce")
        df["x_raw"] = 6.0 * tokens_t * params_b
    else:
        df["x_raw"] = pd.to_numeric(df["#Params (B)"], errors="coerce")
    return df


def _build_panels(
    df: pd.DataFrame,
    *,
    task_col: str,
    curves_dir: Path,
    task_slug: str,
    tau: float,
    lambda_b: float,
    kappa_final: float,
    cfg: RestyleConfig,
) -> List[PanelData]:
    y_raw = pd.to_numeric(df[task_col], errors="coerce").to_numpy(dtype=float)
    y = maybe_scale_task_values(y_raw)
    x_raw = pd.to_numeric(df["x_raw"], errors="coerce").to_numpy(dtype=float)
    per = pd.to_numeric(df["period_idx"], errors="coerce").to_numpy(dtype=int)

    finite = np.isfinite(x_raw) & (x_raw > 0) & np.isfinite(y) & (per >= 0)
    label_to_idx: Dict[str, int] = {lab: i for i, (lab, _, _) in enumerate(cfg.period_bounds)}

    panels: List[PanelData] = []
    for k, spec in enumerate(cfg.period_splits_single, start=1):
        train_labels = list(spec.get("train_labels", []))  # type: ignore[arg-type]
        val_label = spec.get("val_label", None)  # type: ignore[assignment]
        train_inds = [label_to_idx[lbl] for lbl in train_labels if lbl in label_to_idx]
        val_ind = label_to_idx.get(str(val_label), None) if val_label is not None else None
        if not train_inds or val_ind is None:
            continue

        m_train = finite & np.isin(per, np.asarray(train_inds, dtype=int))
        m_val = finite & (per == int(val_ind))

        train_x = x_raw[m_train].astype(float, copy=False)
        train_y = y[m_train].astype(float, copy=False)
        val_x = x_raw[m_val].astype(float, copy=False)
        val_y = y[m_val].astype(float, copy=False)

        curve_train_csv = curves_dir / f"{task_slug}_k{k}.csv"
        curve_train_x, curve_train_y = _load_curve_xy(curve_train_csv)

        curve_val_x = np.array([], dtype=float)
        curve_val_y = np.array([], dtype=float)
        if val_x.size >= 3:
            cvx, cvy = fit_sigmoid_frontier(
                val_x,
                val_y,
                tau=float(tau),
                use_log10_x=True,
                grid_points=400,
                fit_mode="quantile_per_point",
                lambda_b=float(lambda_b),
                kappa_final=float(kappa_final),
            )
            if cvx.size and cvy.size:
                curve_val_x = cvx.astype(float)
                curve_val_y = cvy.astype(float)

        panels.append(
            PanelData(
                train_x_raw=train_x,
                train_y=train_y,
                val_x_raw=val_x,
                val_y=val_y,
                curve_train_x_raw=curve_train_x,
                curve_train_y=curve_train_y,
                curve_val_x_raw=curve_val_x,
                curve_val_y=curve_val_y,
            )
        )
    return panels


def _compute_shared_ylim(panels: Sequence[PanelData]) -> Tuple[float, float]:
    y_vals: List[float] = []
    for p in panels:
        for arr in (p.train_y, p.val_y, p.curve_train_y, p.curve_val_y):
            if arr.size:
                y_vals.append(float(np.nanmin(arr)))
                y_vals.append(float(np.nanmax(arr)))
    if not y_vals:
        return (0.0, 1.0)
    y_min = float(np.nanmin(y_vals))
    y_max = float(np.nanmax(y_vals))
    if not (np.isfinite(y_min) and np.isfinite(y_max)) or y_min == y_max:
        return (0.0, 1.0)
    pad = 0.02 * max(1e-6, (y_max - y_min))
    return (y_min - pad, y_max + pad)


def _apply_axes_style(ax) -> None:
    ax.set_facecolor("white")
    ax.set_axisbelow(True)

    ax.grid(
        True,
        color=STYLE["grid_color"],
        alpha=STYLE["grid_alpha"],
        linestyle=STYLE["grid_linestyle"],
        linewidth=STYLE["grid_linewidth"],
        zorder=0,
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(STYLE["spine_width"])
    ax.spines["bottom"].set_linewidth(STYLE["spine_width"])
    ax.spines["left"].set_color(STYLE["spine_color"])
    ax.spines["bottom"].set_color(STYLE["spine_color"])

    ax.tick_params(axis="both", which="major", length=4, width=0.8, direction="out", color=STYLE["spine_color"])
    ax.tick_params(axis="both", which="minor", length=2.5, width=0.6, direction="out", color=STYLE["spine_color"])


def _format_log_xticks(ax, *, xticks: Sequence[float]) -> None:
    ax.set_xticks(list(xticks))
    ax.set_xticklabels([f"$10^{{{int(np.log10(x))}}}$" for x in xticks])


def _save(fig, *, out_pdf: Path, out_png: Path) -> None:
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf, dpi=300, bbox_inches="tight", pad_inches=0.05, facecolor="white", edgecolor="none")
    fig.savefig(out_png, dpi=300, bbox_inches="tight", pad_inches=0.05, facecolor="white", edgecolor="none")


def _restyle_triptych(
    *,
    task_slug: str,
    task_col: str,
    df: pd.DataFrame,
    curves_dir: Path,
    out_pdf: Path,
    out_png: Path,
    fig_width: float,
    fig_height: float,
    dpi: int,
    tau: float,
    lambda_b: float,
    kappa_final: float,
    cfg: RestyleConfig,
) -> None:
    panels = _build_panels(
        df,
        task_col=task_col,
        curves_dir=curves_dir,
        task_slug=task_slug,
        tau=tau,
        lambda_b=lambda_b,
        kappa_final=kappa_final,
        cfg=cfg,
    )
    if len(panels) != 3:
        raise RuntimeError(f"Expected 3 panels for {task_slug}, got {len(panels)}")

    y_lo, y_hi = _compute_shared_ylim(panels)

    fig, axes = plt.subplots(1, 3, figsize=(fig_width, fig_height), dpi=dpi, sharey=True)
    fig.patch.set_facecolor("white")
    fig.subplots_adjust(left=0.10, right=0.985, bottom=float(cfg.bottom), top=0.90, wspace=0.18)

    for idx, (ax, panel) in enumerate(zip(axes, panels), start=1):
        _apply_axes_style(ax)
        ax.set_xscale("log")
        ax.set_ylim(y_lo, y_hi)

        scale = float(cfg.x_scale_to_plot)
        train_x = panel.train_x_raw * scale
        val_x = panel.val_x_raw * scale
        curve_train_x = panel.curve_train_x_raw * scale
        curve_val_x = panel.curve_val_x_raw * scale

        ax.scatter(
            train_x,
            panel.train_y,
            s=STYLE["marker_size"] ** 2,
            alpha=STYLE["marker_alpha"],
            color=STYLE["train_points_color"],
            edgecolors=STYLE["marker_edgecolor"],
            linewidths=STYLE["marker_edgewidth"],
            zorder=3,
            rasterized=True,
        )
        ax.scatter(
            val_x,
            panel.val_y,
            s=STYLE["marker_size"] ** 2,
            alpha=STYLE["marker_alpha"],
            color=STYLE["val_points_color"],
            edgecolors=STYLE["marker_edgecolor"],
            linewidths=STYLE["marker_edgewidth"],
            zorder=3,
            rasterized=True,
        )

        if curve_train_x.size and panel.curve_train_y.size:
            ax.plot(
                curve_train_x,
                panel.curve_train_y,
                linewidth=STYLE["line_width_fit"],
                alpha=STYLE["line_alpha"],
                color=STYLE["fit_train_color"],
                linestyle="-",
                zorder=5,
            )
        if curve_val_x.size and panel.curve_val_y.size:
            ax.plot(
                curve_val_x,
                panel.curve_val_y,
                linewidth=STYLE["line_width_fit"],
                alpha=STYLE["line_alpha"],
                color=STYLE["fit_val_color"],
                linestyle="--",
                zorder=5,
            )

        _format_log_xticks(ax, xticks=cfg.x_ticks)
        ax.text(
            0.5,
            0.93,
            f"${cfg.period_symbol} = {idx}$",
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=13,
            fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="none", alpha=0.8),
        )

    axes[0].set_ylabel("Accuracy", fontsize=13, fontweight="bold")
    fig.text(0.5, float(cfg.x_label_y), cfg.x_label, ha="center", va="center", fontsize=13, fontweight="bold")

    legend_handles = [
        Line2D([0], [0], marker="o", linestyle="None", markersize=6.5, markerfacecolor=STYLE["train_points_color"], markeredgecolor=STYLE["marker_edgecolor"], markeredgewidth=STYLE["marker_edgewidth"], label="Train"),
        Line2D([0], [0], marker="o", linestyle="None", markersize=6.5, markerfacecolor=STYLE["val_points_color"], markeredgecolor=STYLE["marker_edgecolor"], markeredgewidth=STYLE["marker_edgewidth"], label="Val"),
        Line2D([0], [0], color=STYLE["fit_train_color"], linewidth=STYLE["line_width_fit"], linestyle="-", label="Fit (Train)"),
        Line2D([0], [0], color=STYLE["fit_val_color"], linewidth=STYLE["line_width_fit"], linestyle="--", label="Fit (Val)"),
    ]
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.985),
        ncol=4,
        frameon=False,
        columnspacing=1.2,
        handletextpad=0.6,
        handlelength=2.2,
        borderaxespad=0.0,
    )

    _save(fig, out_pdf=out_pdf, out_png=out_png)
    plt.close(fig)


def discover_slug_task_map(*, slugs: Sequence[str], curves_dir: Path) -> Dict[str, str]:
    slug_to_task: Dict[str, str] = {}
    for slug in slugs:
        curve_csv = curves_dir / f"{slug}_k1.csv"
        if not curve_csv.exists():
            raise FileNotFoundError(f"Missing curve CSV for {slug}: {curve_csv}")
        slug_to_task[slug] = task_from_curve(curve_csv)
    return slug_to_task


def restyle_slug_set(
    *,
    slugs: Sequence[str],
    plots_dir: Path,
    curves_dir: Path,
    csv_path: Path,
    cfg: RestyleConfig,
    tau: float,
    lambda_b: float,
    kappa_final: float,
    fig_width: float,
    fig_height: float,
    dpi: int,
) -> int:
    if not slugs:
        return 0
    slug_to_task = discover_slug_task_map(slugs=slugs, curves_dir=curves_dir)
    task_cols = [slug_to_task[s] for s in slugs]
    df = _prepare_dataframe(csv_path, task_cols=task_cols, cfg=cfg)

    for slug in slugs:
        task_col = slug_to_task[slug]
        out_png = plots_dir / f"{slug}_period4.png"
        out_pdf = plots_dir / f"{slug}_period4.pdf"
        _restyle_triptych(
            task_slug=slug,
            task_col=task_col,
            df=df,
            curves_dir=curves_dir,
            out_pdf=out_pdf,
            out_png=out_png,
            fig_width=float(fig_width),
            fig_height=float(fig_height),
            dpi=int(dpi),
            tau=float(tau),
            lambda_b=float(lambda_b),
            kappa_final=float(kappa_final),
            cfg=cfg,
        )
    return len(slugs)


def list_plot_slugs(plots_dir: Path) -> List[str]:
    return [slug for slug, _ in iter_plot_slugs(plots_dir)]

from __future__ import annotations

from typing import Literal

from skill_frontier.plotting.labels import MODEL_SIZE_PARAMS_LABEL


def apply_pretraining_compute_tick_multiplier(
    ax,
    *,
    multiplier: float = 1e21,
    axis: Literal["x"] = "x",
    label: str = "Pretraining Compute (FLOPs)",
    require_label_match: bool = True,
) -> None:
    """Format log-scale tick labels as if values were multiplied by `multiplier`.

    This is used for plots where the x-values are stored in units of 1e21 FLOPs
    (e.g., 6 * tokens(T) * params(B) where T is in trillions and B is in billions),
    but the axis label is "Pretraining Compute (FLOPs)" and we want tick labels in
    absolute FLOPs.
    """

    if axis != "x":
        raise ValueError(f"Unsupported axis: {axis!r}")

    try:
        import matplotlib.ticker as mticker  # type: ignore
    except Exception:
        return

    if ax is None:
        return

    if require_label_match:
        try:
            if str(ax.get_xlabel()) != label:
                return
        except Exception:
            return

    try:
        if str(ax.get_xscale()) != "log":
            return
    except Exception:
        return

    base_formatter = mticker.LogFormatterMathtext(base=10.0)

    def _fmt(val, pos=None):  # type: ignore[no-untyped-def]
        try:
            if val is None or val <= 0:
                return ""
            return base_formatter(val * multiplier, pos)
        except Exception:
            return ""

    ax.xaxis.set_major_formatter(mticker.FuncFormatter(_fmt))


def apply_model_size_tick_multiplier(
    ax,
    *,
    multiplier: float = 1e9,
    axis: Literal["x"] = "x",
    label: str = MODEL_SIZE_PARAMS_LABEL,
    require_label_match: bool = True,
) -> None:
    """Format log-scale tick labels as if values were multiplied by `multiplier`.

    This mirrors `apply_pretraining_compute_tick_multiplier`, but for model size plots
    where the x-values are stored in billions of parameters.
    """

    if axis != "x":
        raise ValueError(f"Unsupported axis: {axis!r}")

    try:
        import matplotlib.ticker as mticker  # type: ignore
    except Exception:
        return

    if ax is None:
        return

    if require_label_match:
        try:
            if str(ax.get_xlabel()) != label:
                return
        except Exception:
            return

    try:
        if str(ax.get_xscale()) != "log":
            return
    except Exception:
        return

    base_formatter = mticker.LogFormatterMathtext(base=10.0)

    def _fmt(val, pos=None):  # type: ignore[no-untyped-def]
        try:
            if val is None or val <= 0:
                return ""
            return base_formatter(val * multiplier, pos)
        except Exception:
            return ""

    ax.xaxis.set_major_formatter(mticker.FuncFormatter(_fmt))

"""Plotting utilities."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Union


def ensure_dir(path: Union[str, Path]) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def save_figure(
    fig,
    out_dir: Union[str, Path],
    stem: str,
    *,
    dpi: Optional[float] = None,
    bbox_inches: Optional[str] = None,
    pad_inches: Optional[float] = None,
    facecolor: Optional[str] = None,
    edgecolor: Optional[str] = None,
    png_first: bool = False,
) -> tuple[str, str]:
    out_dir_str = str(out_dir)
    base = os.path.join(out_dir_str, stem)
    pdf_path = f"{base}.pdf"
    png_path = f"{base}.png"

    save_kwargs = {}
    if dpi is not None:
        save_kwargs["dpi"] = dpi
    if bbox_inches is not None:
        save_kwargs["bbox_inches"] = bbox_inches
    if pad_inches is not None:
        save_kwargs["pad_inches"] = pad_inches
    if facecolor is not None:
        save_kwargs["facecolor"] = facecolor
    if edgecolor is not None:
        save_kwargs["edgecolor"] = edgecolor

    if png_first:
        fig.savefig(png_path, **save_kwargs)
        fig.savefig(pdf_path, **save_kwargs)
    else:
        fig.savefig(pdf_path, **save_kwargs)
        fig.savefig(png_path, **save_kwargs)

    return pdf_path, png_path


def apply_font_embedding(fonttype: int = 42) -> None:
    import matplotlib as mpl  # type: ignore

    mpl.rcParams["pdf.fonttype"] = fonttype
    mpl.rcParams["ps.fonttype"] = fonttype

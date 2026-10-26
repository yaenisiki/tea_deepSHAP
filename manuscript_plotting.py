"""Shared, figure-number-free Matplotlib style and panel exports."""

from __future__ import annotations

import re
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.transforms import Bbox


MM_PER_INCH = 25.4
COLUMN_WIDTH_MM = {"single": 90, "one_half": 140, "double": 190}
PNG_DPI = 600
TIFF_DPI = {"line_art": 1000, "combination": 600, "image": 300}


def preferred_font_path(*, bold: bool = False) -> Path:
    available = {entry.name for entry in font_manager.fontManager.ttflist}
    family = next((name for name in ("Arial", "Helvetica", "DejaVu Sans") if name in available), "DejaVu Sans")
    prop = font_manager.FontProperties(family=family, weight="bold" if bold else "normal")
    return Path(font_manager.findfont(prop, fallback_to_default=True))


def apply_manuscript_style() -> None:
    family = font_manager.FontProperties(fname=str(preferred_font_path())).get_name()
    mpl.rcParams.update({
        "font.family": family,
        "font.sans-serif": [family],
        "font.size": 8,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "axes.titlesize": 8,
        "axes.linewidth": 0.8,
        "lines.linewidth": 1.3,
        "lines.markersize": 4.5,
        "patch.linewidth": 0.8,
        "grid.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.frameon": False,
        "pdf.fonttype": 42,
        "pdf.use14corefonts": False,
        "svg.fonttype": "none",
        "savefig.dpi": PNG_DPI,
        "axes.unicode_minus": False,
    })


def panel_figure(width_mm: float = 90, height_mm: float = 75, **kwargs):
    """Create one panel at its intended physical size, not a later scaled proxy."""
    fig, ax = plt.subplots(
        figsize=(width_mm / MM_PER_INCH, height_mm / MM_PER_INCH),
        layout="constrained", **kwargs,
    )
    fig._manuscript_fixed_canvas = True
    return fig, ax


def content_stem(name: str) -> str:
    stem = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")
    if not stem or re.search(r"(?:^|_)fig(?:ure)?_?\d+[a-z]?(?:_|$)", stem):
        raise ValueError("Panel filenames must describe content, not a Figure number")
    return stem


def save_panel(
    fig: mpl.figure.Figure,
    directory: str | Path,
    name: str,
    *,
    axes: mpl.axes.Axes | None = None,
    tiff: bool = False,
    tiff_kind: str = "auto",
    close: bool = True,
) -> dict[str, Path]:
    """Export one vector PDF and 600-dpi PNG; optionally a submission TIFF.

    ``axes`` crops one axes from an existing composite without rasterizing its
    PDF content. New panels should instead use ``panel_figure`` at final size.
    """
    stem = content_stem(name)
    if tiff_kind == "auto":
        tiff_kind = "line_art" if stem.endswith(("_roc", "_pr", "_calibration")) or "threshold" in stem else "combination"
    if tiff_kind not in TIFF_DPI:
        raise ValueError(f"Unknown TIFF kind: {tiff_kind}")
    if getattr(fig, "_suptitle", None) is not None:
        fig._suptitle.set_text("")
    for ax in fig.axes:
        ax.set_title("")
    destination = Path(directory)
    destination.mkdir(parents=True, exist_ok=True)
    bbox = None if getattr(fig, "_manuscript_fixed_canvas", False) else "tight"
    if axes is not None:
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        boxes = [axes.get_tightbbox(renderer)]
        for artist in [*axes.collections, *axes.images]:
            colorbar = getattr(artist, "colorbar", None)
            if colorbar is not None:
                boxes.append(colorbar.ax.get_tightbbox(renderer))
        for other in fig.axes:
            colorbar = getattr(other, "_colorbar", None)
            if colorbar is not None and getattr(colorbar.mappable, "axes", None) is axes:
                boxes.append(other.get_tightbbox(renderer))
        bbox = Bbox.union(boxes).transformed(fig.dpi_scale_trans.inverted())
    exports = {suffix: destination / f"{stem}.{suffix}" for suffix in ("pdf", "png")}
    fig.savefig(exports["pdf"], bbox_inches=bbox, pad_inches=0.08)
    fig.savefig(exports["png"], dpi=PNG_DPI, bbox_inches=bbox, pad_inches=0.08)
    if tiff:
        exports["tiff"] = destination / f"{stem}.tiff"
        fig.savefig(exports["tiff"], dpi=TIFF_DPI[tiff_kind], bbox_inches=bbox, pad_inches=0.08)
    if close:
        plt.close(fig)
    return exports


def save_axes_panels(fig: mpl.figure.Figure, directory: str | Path, panels: dict[str, mpl.axes.Axes]) -> None:
    """Write independently reusable vector/raster panels from a staging canvas."""
    for name, ax in panels.items():
        save_panel(fig, directory, name, axes=ax, tiff=True, close=False)
    plt.close(fig)

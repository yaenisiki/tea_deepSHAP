"""Compose existing vector-PDF panels without rerunning analysis.

Usage: python compose_panels.py --spec path/to/layout.json
The JSON specifies output, width_mm (90/140/190), height_mm, margin_mm,
gap_mm, optional row_weights, and rows of {file, weight} entries. A row may
contain one wide panel or several unequal-width panels. No Figure number or
theme order is prescribed.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from manuscript_plotting import MM_PER_INCH, TIFF_DPI, preferred_font_path


def export_tiff_from_png(png_path: Path, tiff_path: Path, *, dpi: int = 600) -> None:
    """Write a 600-dpi TIFF atomically from the already-rendered PNG."""
    if dpi != 600:
        raise ValueError("PDF composite TIFF currently requires 600 dpi")
    from PIL import Image

    temporary_tiff = tiff_path.with_name(tiff_path.stem + ".pending.tiff")
    try:
        with Image.open(png_path) as preview:
            preview.save(temporary_tiff, format="TIFF", compression="raw",
                         dpi=(dpi, dpi))
        temporary_tiff.replace(tiff_path)
    finally:
        temporary_tiff.unlink(missing_ok=True)


def compose(spec_path: Path) -> None:
    try:
        import pymupdf as fitz  # Insert source PDF pages as vector content.
    except ImportError as exc:
        raise RuntimeError("PyMuPDF is required for vector-PDF panel composition") from exc
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    width_mm = float(spec.get("width_mm", 190))
    if width_mm not in (90, 140, 190):
        raise ValueError("width_mm must be 90, 140, or 190")
    height_mm = float(spec["height_mm"])
    margin_mm = float(spec.get("margin_mm", 3))
    gap_mm = float(spec.get("gap_mm", 3))
    label_gutter_mm = float(spec.get("label_gutter_mm", 4))
    rows = spec["rows"]
    if not rows or any(not row or len(row) > 3 for row in rows):
        raise ValueError("Supply 1x2, 2x2, 2x3, or unequal rows with 1–3 panels")
    weights = spec.get("row_weights", [1] * len(rows))
    if len(weights) != len(rows) or any(float(value) <= 0 for value in weights):
        raise ValueError("row_weights must contain one positive value per row")
    if min(width_mm, height_mm) <= 2 * margin_mm or gap_mm < 0:
        raise ValueError("Invalid page geometry")

    mm_to_pt = 72 / MM_PER_INCH
    page_width, page_height = width_mm * mm_to_pt, height_mm * mm_to_pt
    margin, gap, label_gutter = (value * mm_to_pt for value in (margin_mm, gap_mm, label_gutter_mm))
    usable_height = page_height - 2 * margin - (len(rows) - 1) * gap
    if usable_height <= 0:
        raise ValueError("Rows do not fit on the page")
    document = fitz.open()
    page = document.new_page(width=page_width, height=page_height)
    page.insert_font(fontname="PanelBold", fontfile=str(preferred_font_path(bold=True)))
    label_index = 0
    top = margin
    source_files: list[str] = []
    for row, row_weight in zip(rows, weights):
        row_height = usable_height * float(row_weight) / sum(map(float, weights))
        widths = [float(item.get("weight", 1)) for item in row]
        if any(value <= 0 for value in widths):
            raise ValueError("Panel weights must be positive")
        usable_width = page_width - 2 * margin - (len(row) - 1) * gap
        left = margin
        for item, panel_weight in zip(row, widths):
            panel_width = usable_width * panel_weight / sum(widths)
            source_path = (spec_path.parent / item["file"]).resolve()
            if source_path.suffix.lower() != ".pdf":
                raise ValueError(f"Vector PDF panel required: {source_path}")
            with fitz.open(source_path) as source:
                if len(source) != 1:
                    raise ValueError(f"Panel PDF must have one page: {source_path}")
                target = fitz.Rect(left, top + label_gutter, left + panel_width, top + row_height)
                if target.is_empty:
                    raise ValueError("Panel target is empty; increase height_mm")
                scale = min(target.width / source[0].rect.width, target.height / source[0].rect.height)
                if not item.get("allow_rescale", False) and not 0.88 <= scale <= 1.12:
                    raise ValueError(
                        f"Panel {source_path.name} would scale {scale:.2f}x; regenerate at final size "
                        "or explicitly set allow_rescale"
                    )
                page.show_pdf_page(target, source, 0, keep_proportion=True)
            source_files.append(str(source_path))
            if spec.get("panel_letters", True):
                if label_index >= 26:
                    raise ValueError("At most 26 automatic panel letters are supported")
                page.insert_text(
                    (left, top + 10.5), chr(ord("A") + label_index),
                    fontname="PanelBold", fontsize=11.5, color=(0, 0, 0),
                )
            label_index += 1
            left += panel_width + gap
        top += row_height + gap

    output = (spec_path.parent / spec["output"]).resolve()
    if any(path.exists() for path in (output.with_suffix(".pdf"), output.with_suffix(".png"), output.with_suffix(".tiff"))):
        raise FileExistsError("Composite output already exists; choose a new output stem")
    output.parent.mkdir(parents=True, exist_ok=True)
    document.save(output.with_suffix(".pdf"), garbage=4, deflate=True)
    raster = page.get_pixmap(matrix=fitz.Matrix(600 / 72, 600 / 72), alpha=False)
    raster.set_dpi(600, 600)
    raster.save(output.with_suffix(".png"))
    del raster
    document.close()
    if spec.get("tiff", False):
        tiff_kind = spec.get("tiff_kind", "combination")
        if tiff_kind not in TIFF_DPI:
            raise ValueError(f"Unknown TIFF kind: {tiff_kind}")
        # The combination TIFF and PNG use the same 600-dpi raster. Reusing the
        # saved PNG avoids a second large pixmap and an Image.frombytes copy.
        export_tiff_from_png(output.with_suffix(".png"), output.with_suffix(".tiff"),
                             dpi=TIFF_DPI[tiff_kind])
    print(f"Composed {len(source_files)} panels: {output.with_suffix('.pdf')}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compose existing vector PDF panels")
    parser.add_argument("--spec", type=Path, required=True, help="JSON layout specification")
    compose(parser.parse_args().spec.resolve())

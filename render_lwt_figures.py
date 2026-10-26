"""Render available LWT panels from frozen outputs and open a local gallery.

This entry point does not run development CV, LORO model fitting, temporal
validation, or SHAP computation. Chemical PCA/effects are calculated solely
for their requested descriptive/inferential panels from the locked workbook.
"""

from __future__ import annotations

import argparse
from datetime import datetime
from html import escape
from pathlib import Path
import subprocess
import sys
import webbrowser


ROOT = Path(__file__).resolve().parent
FIGURE_ROOT = ROOT / "outputs" / "lwt_figures"
LORO_PREDICTIONS = ROOT / "outputs" / "lwt_paper" / "full" / "loro_predictions.csv"


def _new_preview_dir() -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    candidate = FIGURE_ROOT / f"preview_{stamp}"
    suffix = 2
    while candidate.exists():
        candidate = FIGURE_ROOT / f"preview_{stamp}_{suffix}"
        suffix += 1
    candidate.mkdir(parents=True)
    return candidate


def _gallery(directory: Path, notes: list[str]) -> Path:
    sections = []
    supplements = []
    for name in ("chemical", "benchmark", "temporal", "loro", "shap"):
        images = sorted((directory / name).glob("*.png")) if (directory / name).exists() else []
        if images:
            cards = "\n".join(
                f'<figure><a href="{escape(image.relative_to(directory).as_posix(), quote=True)}">'
                f'<img src="{escape(image.relative_to(directory).as_posix(), quote=True)}" '
                f'alt="{escape(image.stem, quote=True)}"></a>'
                f'<figcaption>{escape(image.stem)}</figcaption></figure>'
                for image in images
            )
            sections.append(f"<section><h2>{escape(name.capitalize())}</h2><div class='grid'>{cards}</div></section>")
        for image in sorted((directory / name / "supplementary").glob("*.png")):
            relative = escape(image.relative_to(directory).as_posix(), quote=True)
            supplements.append(
                f'<figure><a href="{relative}"><img src="{relative}" '
                f'alt="{escape(image.stem, quote=True)}"></a>'
                f'<figcaption>{escape(name)} / {escape(image.stem)}</figcaption></figure>'
            )
    messages = "".join(f"<li>{escape(note)}</li>" for note in notes)
    page = (
        "<!doctype html><html lang='en'><head><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width, initial-scale=1'>"
        "<title>LWT panel previews</title>"
        "<style>body{font:15px Arial,sans-serif;color:#263c49;margin:2rem auto;max-width:1400px;"
        "padding:0 1rem}h1,h2{font-weight:600}.grid{display:grid;"
        "grid-template-columns:repeat(auto-fit,minmax(360px,1fr));gap:18px}"
        "figure{margin:0;padding:12px;border:1px solid #d9e3e7;background:white}"
        "img{display:block;width:100%;height:auto}figcaption{font-size:12px;margin-top:8px}"
        "li{margin:5px 0}</style></head><body><h1>LWT panel previews</h1>"
        f"<ul>{messages}</ul>{''.join(sections)}"
        f"<section><h2>Supplementary candidates</h2><div class='grid'>{''.join(supplements)}</div></section>"
        "</body></html>"
    )
    index = directory / "index.html"
    index.write_text(page, encoding="utf-8")
    return index


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-open", action="store_true", help="Do not open the local gallery in a browser")
    parser.add_argument("--loro-source", type=Path,
                        help="Directory containing verified loro_predictions.csv and loro_metrics.csv")
    args = parser.parse_args()
    output = _new_preview_dir()
    jobs = [
        ("chemical", ["plot_lwt_chemical.py", "--output", str(output / "chemical")]),
        ("benchmark", ["plot_lwt_results.py", "--section", "benchmark", "--output", str(output / "benchmark")]),
        ("temporal", ["plot_lwt_results.py", "--section", "temporal", "--output", str(output / "temporal")]),
        ("shap", ["plot_lwt_shap_results.py", "--output", str(output / "shap")]),
    ]
    notes = []
    loro_source = args.loro_source.resolve() if args.loro_source else None
    loro_predictions = loro_source / "loro_predictions.csv" if loro_source else LORO_PREDICTIONS
    loro_metrics = loro_source / "loro_metrics.csv" if loro_source else ROOT / "outputs" / "lwt_paper" / "full" / "loro_metrics.csv"
    if loro_predictions.is_file() and loro_metrics.is_file():
        jobs.insert(3, ("loro", ["plot_lwt_loro.py", "--predictions", str(loro_predictions),
                                "--metrics", str(loro_metrics),
                                "--output", str(output / "loro")]))
    else:
        notes.append("LORO CI panel skipped: saved per-sample held-out probabilities are unavailable; aggregate metrics cannot yield a genuine bootstrap CI.")
    failed = []
    for label, arguments in jobs:
        print(f"Rendering {label}...", flush=True)
        completed = subprocess.run([sys.executable, str(ROOT / arguments[0]), *arguments[1:]], cwd=ROOT)
        if completed.returncode:
            failed.append(label)
            notes.append(f"{label} failed (exit code {completed.returncode}); see terminal traceback.")
    index = _gallery(output, notes)
    print(f"Preview gallery: {index}", flush=True)
    if not args.no_open:
        webbrowser.open(index.resolve().as_uri())
    if failed:
        raise SystemExit(f"Figure sections failed: {', '.join(failed)}")


if __name__ == "__main__":
    main()

"""Run the prespecified LWT fast mode without using 2026 for selection."""

from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import argparse
import json
import platform
import time
from importlib import metadata
from pathlib import Path

import pandas as pd

from lwt_data import DATASET_VERSION, DATA_PATH, FEATURES, REGIONS, ROOT, load_dataset
from lwt_models import MODEL_NAMES, model_specs
from lwt_validation import development_cv, leave_one_region_out


def main() -> None:
    parser = argparse.ArgumentParser(description="LWT development nested CV and six-region LORO")
    parser.add_argument("--mode", choices=["fast", "paper"], default="fast")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--only", choices=MODEL_NAMES)
    parser.add_argument("--outer-limit", type=int, default=None, help="Smoke test only")
    parser.add_argument("--skip-loro", action="store_true", help="Smoke test only")
    args = parser.parse_args()
    max_outer = 25 if args.mode == "paper" else 10
    if args.outer_limit is not None and not (1 <= args.outer_limit <= max_outer):
        parser.error(f"--outer-limit must be between 1 and {max_outer}")
    output = (args.output or ROOT / "outputs" / f"lwt_{args.mode}" / "full").resolve()
    if output.exists():
        parser.error(f"Output already exists; choose a new path: {output}")
    df, digest = load_dataset()
    development = df[df["年份"].isin([2024, 2025])].copy()
    assert len(development) == 186 and not (development["年份"] == 2026).any()
    specs, missing = model_specs(mode=args.mode)
    if args.only:
        specs = {args.only: specs[args.only]} if args.only in specs else {}
        missing = {args.only: missing[args.only]} if args.only in missing else {}
    output.mkdir(parents=True)
    started = time.perf_counter()
    folds, summary, failures = development_cv(
        development, specs, missing, outer_limit=args.outer_limit, mode=args.mode,
    )
    folds.to_csv(output / "development_fold_metrics.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(output / "development_summary.csv", index=False, encoding="utf-8-sig")
    failures.to_csv(output / "development_failures.csv", index=False, encoding="utf-8-sig")
    if not args.skip_loro:
        loro_predictions: list[dict] = []
        loro, loro_failures = leave_one_region_out(
            development, specs, missing, mode=args.mode,
            prediction_records=loro_predictions,
        )
        loro.to_csv(output / "loro_metrics.csv", index=False, encoding="utf-8-sig")
        loro_failures.to_csv(output / "loro_failures.csv", index=False, encoding="utf-8-sig")
        pd.DataFrame(loro_predictions).to_csv(output / "loro_predictions.csv", index=False, encoding="utf-8-sig")
    versions = {}
    for package in ["numpy", "pandas", "scikit-learn", "openpyxl", "xgboost", "lightgbm", "catboost"]:
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            versions[package] = "not installed globally"
    manifest = {
        "mode": f"LWT {args.mode.upper()} MODE", "dataset_version": DATASET_VERSION,
        "data_path": str(DATA_PATH), "data_sha256": digest,
        "features": FEATURES, "positive_class": "特级", "development_years": [2024, 2025],
        "development_n": len(development), "temporal_year": 2026,
        "temporal_n": int((df["年份"] == 2026).sum()),
        "temporal_used_for_training_tuning_or_selection": False,
        "statistical_unit": "one supplied input row; synthetic example rows are artificial",
        "outer_cv": f"RepeatedStratifiedKFold 5 folds x {5 if args.mode == 'paper' else 2} repeats, seed 42",
        "inner_cv": f"StratifiedKFold {5 if args.mode == 'paper' else 3} folds, shuffled, ROC-AUC search",
        "search": "RandomizedSearchCV up to 6 candidates" if args.mode == "paper" else "GridSearchCV fast grid",
        "decision_threshold": 0.5,
        "loro_regions": [] if args.skip_loro else REGIONS,
        "loro_singleton_rule": "The one 略阳 sample remains in the training pool of all six region holdouts.",
        "loro_sample_id_source": "generated analysis_id from year and source Excel row; not an original sample number",
        "requested_models": [args.only] if args.only else MODEL_NAMES,
        "missing_dependencies": missing,
        "outer_limit": args.outer_limit,
        "runtime_seconds": time.perf_counter() - started,
        "python": platform.python_version(), "packages": versions,
    }
    (output / "run_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    print(f"Completed: {output}", flush=True)


if __name__ == "__main__":
    main()

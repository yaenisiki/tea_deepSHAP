"""Redraw LWT SHAP panels from completed saved SHAP tables; never fit models."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from lwt_data import CONCENTRATION_UNIT, FEATURES, ROOT, load_dataset
from run_lwt_shap import (
    PRIMARY, _render_panels, _sha256, assert_premium_positive_class,
)
from run_lwt_temporal import FREEZE_PATH, OUTPUT_DIR as TEMPORAL_DIR


TABLES = {
    "importance": "shap_global_importance.csv",
    "folds": "shap_fold_stability.csv",
    "years": "shap_year_stability.csv",
    "regions": "shap_region_stability.csv",
    "development": "shap_development_oof_values.csv",
    "temporal": "shap_temporal_values.csv",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path,
                        default=ROOT / "outputs" / "lwt_shap_retry")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "outputs" / "lwt_figures" / "shap")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Output exists; choose a new --output: {args.output}")
    manifest_path = args.source / "shap_run_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Completed SHAP manifest not found: {manifest_path}")
    data, digest = load_dataset()
    assert_premium_positive_class(data)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not (
        manifest.get("data_sha256") == digest
        and manifest.get("frozen_config_sha256") == _sha256(FREEZE_PATH)
        and manifest.get("model") == PRIMARY
        and manifest.get("feature_set") == FEATURES
        and manifest.get("development_n") == 186
        and manifest.get("temporal_n") == 201
        and manifest.get("no_tuning_feature_selection_or_model_selection") is True
    ):
        raise ValueError("Saved SHAP tables do not match the frozen LWT data/model")
    additivity = manifest.get("max_additivity_error", {})
    if set(additivity) != {"development", "temporal", "oof"} or any(
        value is None or not np.isfinite(float(value)) or float(value) > 1e-4
        for value in additivity.values()
    ):
        raise ValueError("Completed SHAP run lacks a passing probability-additivity check")
    tables = {}
    for key, filename in TABLES.items():
        path = args.source / filename
        if not path.is_file():
            raise FileNotFoundError(path)
        tables[key] = pd.read_csv(path)
    development = data.loc[data["年份"].isin([2024, 2025])].copy()
    temporal = data.loc[data["年份"].eq(2026)].copy()
    dev_values, temporal_values = tables["development"], tables["temporal"]
    shap_columns = [f"shap_{feature}" for feature in FEATURES]
    if not (
        len(dev_values) == 186 and len(temporal_values) == 201
        and len(tables["importance"]) == 10 and set(tables["importance"].feature) == set(FEATURES)
        and len(tables["folds"]) == 250
        and len(tables["years"]) == 30 and len(tables["regions"]) == 60
        and set(shap_columns).issubset(dev_values.columns)
        and set(shap_columns).issubset(temporal_values.columns)
        and np.array_equal(dev_values.analysis_id.to_numpy(), development.analysis_id.to_numpy())
        and np.array_equal(temporal_values.analysis_id.to_numpy(), temporal.analysis_id.to_numpy())
        and np.array_equal(temporal_values.source_excel_row.to_numpy(dtype=int),
                           temporal.source_excel_row.to_numpy())
    ):
        raise ValueError("Saved SHAP rows or ten feature columns are incomplete/misaligned")
    probability = temporal_values.probability_special_grade.to_numpy(dtype=float)
    temporal_shap = temporal_values[shap_columns].to_numpy(dtype=float)
    base = temporal_values.base_probability.to_numpy(dtype=float)
    if not (np.isfinite(temporal_shap).all() and np.isfinite(base).all()
            and np.all((0 <= base) & (base <= 1))
            and np.allclose(base + temporal_shap.sum(axis=1), probability, atol=1e-4)):
        raise ValueError("Saved temporal SHAP does not add to probability of premium grade")
    predictions_path = TEMPORAL_DIR / "temporal_predictions.csv"
    if _sha256(predictions_path) != manifest.get("temporal_predictions_sha256"):
        raise ValueError("Frozen temporal predictions changed after SHAP calculation")
    predictions = pd.read_csv(predictions_path).set_index("analysis_id").loc[temporal.analysis_id]
    if not np.allclose(probability, predictions.probability_special_grade.to_numpy(dtype=float),
                       atol=1e-10, rtol=0):
        raise ValueError("Saved SHAP probabilities differ from frozen temporal predictions")
    folds = tables["folds"]
    weighted = folds.assign(total=folds.mean_abs_shap * folds.n_test).groupby("feature")[["total", "n_test"]].sum()
    reconstructed = (weighted.total / weighted.n_test).loc[tables["importance"].feature].to_numpy()
    if not np.allclose(reconstructed, tables["importance"].mean_abs_shap.to_numpy(), atol=1e-10):
        raise ValueError("Development OOF importance does not match the saved fold records")
    args.output.mkdir(parents=True)
    _render_panels(
        args.output, tables["importance"], tables["folds"],
        tables["years"], tables["regions"], development, temporal,
        dev_values[shap_columns].to_numpy(dtype=float), temporal_shap, base, probability,
    )
    (args.output / "shap_plot_manifest.json").write_text(
        json.dumps({
            "source_manifest_sha256": _sha256(manifest_path),
            "data_sha256": digest,
            "model": PRIMARY,
            "concentration_unit": CONCENTRATION_UNIT,
            "concentration_unit_source": "Researcher-confirmed for 2024–2026; not inferred from Excel headers",
            "no_model_fitting_or_shap_recomputation": True,
            "relative_importance": "mean absolute SHAP divided by within-group sum",
            "year_groups": [2024, 2025, 2026],
            "region_cohort": "2026 temporal, six core regions",
        }, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    print(f"Saved SHAP panels from existing tables to {args.output}")


if __name__ == "__main__":
    main()

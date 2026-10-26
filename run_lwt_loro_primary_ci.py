"""Recover Extra Trees LORO sample scores and plot genuine bootstrap AUC CIs.

This refits exactly six held-out-region models using the region-specific
hyperparameters already selected in the completed paper-mode run. It does not
retune, rerun development CV, use 2026, or change the primary model.
The primary model type was selected on the full development cohort, so these
holdouts assess conditional regional robustness, not model-selection-independent
external-region validation.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
from sklearn.base import clone

from lwt_data import FEATURES, REGIONS, ROOT, load_dataset
from lwt_models import MODEL_NAMES, model_specs
from lwt_validation import evaluate


PRIMARY = "Extra Trees"
PAPER_DIR = ROOT / "outputs" / "lwt_paper" / "full"
FIGURE_DIR = ROOT / "outputs" / "lwt_figures"
CHECK_METRICS = (
    "ROC_AUC", "PR_AUC", "Accuracy", "Sensitivity", "Specificity",
    "Precision", "F1", "Balanced_Accuracy", "MCC", "Brier",
    "TN", "FP", "FN", "TP",
)


def _new_output() -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output = FIGURE_DIR / f"loro_primary_ci_{stamp}"
    suffix = 2
    while output.exists():
        output = FIGURE_DIR / f"loro_primary_ci_{stamp}_{suffix}"
        suffix += 1
    return output


def main() -> None:
    metrics_path = PAPER_DIR / "loro_metrics.csv"
    manifest_path = PAPER_DIR / "run_manifest.json"
    if not metrics_path.is_file() or not manifest_path.is_file():
        raise FileNotFoundError("Completed paper-mode LORO metrics and manifest are required")
    data, digest = load_dataset()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not (
        manifest.get("mode") == "LWT PAPER MODE"
        and manifest.get("data_sha256") == digest
        and manifest.get("features") == FEATURES
        and manifest.get("development_years") == [2024, 2025]
        and manifest.get("development_n") == 186
        and manifest.get("temporal_used_for_training_tuning_or_selection") is False
        and manifest.get("inner_cv") == "StratifiedKFold 5 folds, shuffled, ROC-AUC search"
        and manifest.get("decision_threshold") == 0.5
        and manifest.get("requested_models") == MODEL_NAMES
        and manifest.get("loro_regions") == REGIONS
    ):
        raise ValueError("Saved paper-mode results do not match the locked development dataset")
    reference = pd.read_csv(metrics_path)
    reference = reference.loc[reference.model.eq(PRIMARY)].copy()
    if (len(reference) != 6 or set(reference.region) != set(REGIONS)
            or not reference.status.eq("ok").all()):
        raise ValueError("Expected six completed Extra Trees LORO records")
    reference = reference.set_index("region").loc[REGIONS].reset_index()
    specs, missing = model_specs(seed=42, mode="paper")
    if PRIMARY in missing or PRIMARY not in specs:
        raise RuntimeError("The original Extra Trees pipeline is unavailable")
    template = specs[PRIMARY][0]
    development = data.loc[data["年份"].isin([2024, 2025])].copy()
    if len(development) != 186 or data.loc[data["年份"].eq(2026)].shape[0] != 201:
        raise ValueError("Locked development/temporal cohort sizes changed")

    records = []
    selection_audit = []
    for region in REGIONS:
        row = reference.loc[reference.region.eq(region)].iloc[0]
        held_out = development["产地"].eq(region)
        train = development.loc[~held_out]
        test = development.loc[held_out]
        if (set(train.analysis_id) & set(test.analysis_id)
                or test["产地"].ne(region).any()
                or train["产地"].eq(region).any()):
            raise ValueError(f"Held-out region entered the training pool: {region}")
        if (len(train) != int(row.n_train) or len(test) != int(row.n_test)
                or int(test.label.sum()) != int(row.positive_test)
                or int((1 - test.label).sum()) != int(row.negative_test)):
            raise ValueError(f"Saved LORO sample counts differ for {region}")
        params = json.loads(row.best_params)
        if not isinstance(params, dict) or not params or any(not key.startswith("model__") for key in params):
            raise ValueError(f"Invalid saved Extra Trees hyperparameters for {region}")
        fitted = clone(template).set_params(**params).fit(train[FEATURES], train.label)
        selection_audit.append({
            "region": region, "n_train": len(train), "n_held_out": len(test),
            "parameter_source": "archived region-specific best_params from training-only inner CV",
            "inner_cv": manifest["inner_cv"], "decision_threshold": 0.5,
            "best_params": params, "held_out_rows_used_for_refit": False,
            "retuning": False,
        })
        if list(fitted.classes_) != [0, 1]:
            raise ValueError("Extra Trees class 1 must be premium grade")
        probability = fitted.predict_proba(test[FEATURES])[:, 1]
        observed = evaluate(test.label.to_numpy(dtype=int), probability)
        for metric in CHECK_METRICS:
            if not np.isclose(float(observed[metric]), float(row[metric]), rtol=0, atol=1e-9):
                raise ValueError(
                    f"Refitted {region} {metric} differs from the completed paper-mode result; "
                    "no CI was exported"
                )
        for (_, sample), score in zip(test.iterrows(), probability):
            records.append({
                "model": PRIMARY, "region": region,
                "sample_id": sample.analysis_id, "analysis_id": sample.analysis_id,
                "source_excel_row": int(sample.source_excel_row), "year": int(sample["年份"]),
                "y_true": int(sample.label), "true_label": int(sample.label),
                "y_prob": float(score), "probability_premium_grade": float(score),
            })

    predictions = pd.DataFrame(records)
    if len(predictions) != 185 or not predictions.analysis_id.is_unique:
        raise ValueError("Expected 185 unique held-out samples across six core regions")
    output = _new_output()
    output.mkdir(parents=True)
    predictions_path = output / "loro_predictions.csv"
    verified_metrics_path = output / "loro_metrics.csv"
    predictions.to_csv(predictions_path, index=False, encoding="utf-8-sig")
    reference.to_csv(verified_metrics_path, index=False, encoding="utf-8-sig")
    (output / "loro_refit_manifest.json").write_text(
        json.dumps({
            "dataset_version": "LWT analysis dataset v1.0",
            "data_sha256": digest,
            "source_paper_metrics_sha256": hashlib.sha256(metrics_path.read_bytes()).hexdigest(),
            "model": PRIMARY,
            "development_years": [2024, 2025],
            "held_out_regions": REGIONS,
            "n_unique_held_out": len(predictions),
            "sample_id_source": "generated analysis_id from year and source Excel row; not an original sample number",
            "refit": "Six region-specific training-only Pipeline fits with archived best_params",
            "retuning": False,
            "temporal_2026_used": False,
            "all_archived_metrics_verified": True,
            "split_parameter_audit": selection_audit,
            "primary_model_selection_caveat": "Extra Trees was selected using development CV covering all regions; LORO held-out regions are independent of each split's parameter search, not of global model-type selection",
        }, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    subprocess.run([
        sys.executable, str(ROOT / "plot_lwt_loro.py"),
        "--predictions", str(predictions_path),
        "--metrics", str(verified_metrics_path),
        "--output", str(output),
    ], cwd=ROOT, check=True)
    print(f"LORO CI panel and verified sample predictions: {output}")


if __name__ == "__main__":
    main()

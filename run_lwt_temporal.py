"""Fit development-frozen LWT models and evaluate 2026 once, without retuning."""

from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from lwt_data import FEATURES, ROOT, load_dataset
from lwt_models import MODEL_NAMES, model_specs
from lwt_validation import evaluate


PAPER_DIR = ROOT / "outputs" / "lwt_paper" / "full"
AUDIT_PATH = Path(os.environ.get("TEA_PROVENANCE_AUDIT", str(ROOT / "private_data" / "provenance_audit.csv")))
FREEZE_PATH = ROOT / "lwt_temporal_frozen.json"
OUTPUT_DIR = ROOT / "outputs" / "lwt_temporal"
BOOTSTRAP_REPLICATES = 5000


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_frozen_configuration(data_sha256: str) -> dict:
    frozen = json.loads(FREEZE_PATH.read_text(encoding="utf-8"))
    inputs = {
        "development_fold_metrics_sha256": PAPER_DIR / "development_fold_metrics.csv",
        "development_summary_sha256": PAPER_DIR / "development_summary.csv",
        "development_manifest_sha256": PAPER_DIR / "run_manifest.json",
        "model_code_sha256": ROOT / "lwt_models.py",
        "validation_code_sha256": ROOT / "lwt_validation.py",
    }
    for key, path in inputs.items():
        if _sha256(path) != frozen[key]:
            raise ValueError(f"Frozen input has changed: {path}")
    audit_digest = _sha256(AUDIT_PATH) if AUDIT_PATH.is_file() else None
    if audit_digest != frozen["provenance_audit_sha256"]:
        raise ValueError("Optional provenance audit changed after freezing")
    manifest = json.loads((PAPER_DIR / "run_manifest.json").read_text(encoding="utf-8"))
    if not (
        manifest["mode"] == "LWT PAPER MODE"
        and manifest["data_sha256"] == data_sha256
        and manifest["development_n"] == 186
        and manifest["temporal_n"] == 201
        and manifest["temporal_used_for_training_tuning_or_selection"] is False
        and manifest["outer_limit"] is None
    ):
        raise ValueError("Completed PAPER MODE manifest does not match the frozen design")
    if frozen["primary_model"] != "Extra Trees" or frozen["decision_threshold"] != 0.5:
        raise ValueError("Primary model or decision threshold differs from the freeze")
    if frozen["development_years"] != [2024, 2025] or frozen["temporal_year"] != 2026:
        raise ValueError("Cohort years differ from the freeze")
    if set(frozen["models"]) != set(MODEL_NAMES):
        raise ValueError("Frozen configurations do not cover exactly the 16 official models")

    summary = pd.read_csv(PAPER_DIR / "development_summary.csv")
    if not (
        set(summary["model"]) == set(MODEL_NAMES)
        and len(summary) == 16
        and summary["status"].eq("ok").all()
        and summary["n_success_folds"].eq(25).all()
    ):
        raise ValueError("PAPER MODE development summary is incomplete")
    folds = pd.read_csv(PAPER_DIR / "development_fold_metrics.csv")
    for name in MODEL_NAMES:
        part = folds.loc[folds["model"].eq(name)]
        if len(part) != 25 or part[["repeat", "fold"]].drop_duplicates().shape[0] != 25:
            raise ValueError(f"PAPER MODE fold record is incomplete: {name}")
        ranked = (
            part.groupby("best_params", as_index=False)
            .agg(frequency=("best_params", "size"), inner_auc=("inner_best_auc", "mean"))
            .sort_values(["frequency", "inner_auc", "best_params"], ascending=[False, False, True])
        )
        if json.loads(ranked.iloc[0]["best_params"]) != frozen["models"][name]:
            raise ValueError(f"Frozen parameters do not match development-only rule: {name}")
    return frozen


def _flagged_rows(temporal: pd.DataFrame) -> set[int]:
    """Use a separately supplied audit; absence does not mean provenance is verified."""
    if not AUDIT_PATH.is_file():
        return set()
    audit = pd.read_csv(AUDIT_PATH)
    required = {"source_excel_row", "analysis_id", "year", "provenance_status"}
    if not required.issubset(audit.columns):
        raise ValueError("Optional provenance audit has missing columns")
    if not (
        audit["source_excel_row"].is_unique
        and audit["analysis_id"].is_unique
        and audit["year"].eq(2026).all()
        and audit["provenance_status"].eq("source_explanation_pending").all()
    ):
        raise ValueError("Invalid optional provenance audit")
    matched = temporal.merge(
        audit[["source_excel_row", "analysis_id"]],
        on=["source_excel_row", "analysis_id"], how="inner",
    )
    if len(matched) != len(audit):
        raise ValueError("Provenance records do not match the supplied temporal rows")
    return set(audit["source_excel_row"].astype(int))


def _auc_bootstrap(y: np.ndarray, probability: np.ndarray, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    negative = np.flatnonzero(y == 0)
    positive = np.flatnonzero(y == 1)
    if not len(negative) or not len(positive):
        raise ValueError("Both classes are required for ROC-AUC bootstrap")
    scores = np.empty(BOOTSTRAP_REPLICATES, dtype=float)
    for index in range(BOOTSTRAP_REPLICATES):
        sampled = np.concatenate([
            rng.choice(negative, len(negative), replace=True),
            rng.choice(positive, len(positive), replace=True),
        ])
        scores[index] = roc_auc_score(y[sampled], probability[sampled])
    lower, upper = np.quantile(scores, [0.025, 0.975])
    return {
        "model": "Extra Trees", "n_test": len(y),
        "positive_test": len(positive), "negative_test": len(negative),
        "ROC_AUC": float(roc_auc_score(y, probability)),
        "CI_lower": float(lower), "CI_upper": float(upper),
        "confidence_level": 0.95, "bootstrap_method": "stratified_percentile",
        "bootstrap_replicates": BOOTSTRAP_REPLICATES, "seed": seed,
    }


def main() -> None:
    if OUTPUT_DIR.exists():
        raise FileExistsError(f"Output already exists; no files will be overwritten: {OUTPUT_DIR}")
    data, data_sha256 = load_dataset()
    frozen = _read_frozen_configuration(data_sha256)
    development = data.loc[data["年份"].isin([2024, 2025])].copy()
    temporal = data.loc[data["年份"].eq(2026)].copy()
    if len(development) != 186 or len(temporal) != 201:
        raise ValueError("Development or temporal cohort size has changed")
    flagged_rows = _flagged_rows(temporal)
    retained = ~temporal["source_excel_row"].isin(flagged_rows)
    if flagged_rows and temporal.loc[retained, "label"].nunique() != 2:
        raise ValueError("Sensitivity cohort requires both classes")

    specs, missing = model_specs(seed=frozen["seed"], mode="paper")
    if missing or set(specs) != set(MODEL_NAMES):
        raise RuntimeError(f"All 16 model dependencies are required: {missing}")
    X_train, y_train = development[FEATURES], development["label"]
    X_test, y_test = temporal[FEATURES], temporal["label"].to_numpy()
    metrics, sensitivity_metrics = [], []
    primary_probability = None
    for name in MODEL_NAMES:
        estimator, grid = specs[name]
        params = frozen["models"][name]
        if set(params) != set(grid) or any(params[key] not in grid[key] for key in params):
            raise ValueError(f"Frozen settings differ from the PAPER MODE search space: {name}")
        estimator.set_params(**params)
        estimator.fit(X_train, y_train)
        if list(estimator.classes_) != [0, 1]:
            raise ValueError(f"Unexpected probability class order: {name}")
        probability = estimator.predict_proba(X_test)[:, 1]
        if not (np.isfinite(probability).all() and np.logical_and(probability >= 0, probability <= 1).all()):
            raise ValueError(f"Invalid predicted probabilities: {name}")
        full = evaluate(y_test, probability)
        evaluations = [(metrics, full, len(temporal))]
        if flagged_rows:
            subset = evaluate(y_test[retained.to_numpy()], probability[retained.to_numpy()])
            evaluations.append((sensitivity_metrics, subset, int(retained.sum())))
        for target, values, count in evaluations:
            target.append({
                "model": name, "n_test": count,
                "positive_test": int(values["TP"] + values["FN"]),
                "negative_test": int(values["TN"] + values["FP"]),
                "threshold": frozen["decision_threshold"],
                **values,
                "confusion_matrix": json.dumps([
                    [values["TN"], values["FP"]],
                    [values["FN"], values["TP"]],
                ]),
            })
        if name == frozen["primary_model"]:
            primary_probability = probability
        print(f"Temporal fit/evaluation complete: {name}", flush=True)
    if primary_probability is None:
        raise RuntimeError("Primary model was not evaluated")

    predictions = pd.DataFrame({
        "analysis_id": temporal["analysis_id"].to_numpy(),
        "source_excel_row": temporal["source_excel_row"].to_numpy(),
        "year": 2026,
        "true_label": y_test,
        "probability_special_grade": primary_probability,
        "predicted_label_at_0_5": (primary_probability >= 0.5).astype(int),
        "provenance_pending": temporal["source_excel_row"].isin(flagged_rows).to_numpy(),
    })
    bootstrap = _auc_bootstrap(y_test, primary_probability, frozen["seed"])
    run_manifest = {
        "dataset_version": frozen["dataset_version"],
        "data_sha256": data_sha256,
        "frozen_config_sha256": _sha256(FREEZE_PATH),
        "development_fold_metrics_sha256": frozen["development_fold_metrics_sha256"],
        "provenance_audit_sha256": frozen["provenance_audit_sha256"],
        "primary_model": frozen["primary_model"],
        "parameter_selection": frozen["parameter_selection"],
        "development_n": len(development), "temporal_n": len(temporal),
        "sensitivity_n": int(retained.sum()) if flagged_rows else None,
        "provenance_audit_supplied": AUDIT_PATH.is_file(),
        "threshold": frozen["decision_threshold"],
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "no_2026_training_tuning_or_model_selection": True,
        "models": frozen["models"],
    }
    OUTPUT_DIR.mkdir(parents=True)
    pd.DataFrame(metrics).to_csv(OUTPUT_DIR / "temporal_metrics.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame([bootstrap]).to_csv(
        OUTPUT_DIR / "temporal_primary_bootstrap.csv", index=False, encoding="utf-8-sig",
    )
    predictions.to_csv(OUTPUT_DIR / "temporal_predictions.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(sensitivity_metrics, columns=pd.DataFrame(metrics).columns).to_csv(
        OUTPUT_DIR / "temporal_sensitivity_metrics.csv", index=False, encoding="utf-8-sig",
    )
    (OUTPUT_DIR / "temporal_run_manifest.json").write_text(
        json.dumps(run_manifest, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    print(f"Completed: {OUTPUT_DIR}", flush=True)


if __name__ == "__main__":
    main()

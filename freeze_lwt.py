"""Freeze development-only tuning results before running temporal evaluation."""

import json
import pandas as pd

from lwt_data import DATASET_VERSION, load_dataset
from lwt_models import MODEL_NAMES
from run_lwt_temporal import AUDIT_PATH, FREEZE_PATH, PAPER_DIR, ROOT, _read_frozen_configuration, _sha256


def select_parameters(folds: pd.DataFrame) -> dict:
    models = {}
    for name in MODEL_NAMES:
        part = folds.loc[folds["model"].eq(name)]
        if len(part) != 25 or part[["repeat", "fold"]].drop_duplicates().shape[0] != 25:
            raise ValueError(f"Expected 25 completed paper-mode outer folds for {name}")
        if part["inner_best_auc"].isna().any():
            raise ValueError(f"Missing inner-CV scores for {name}")
        ranked = (
            part.groupby("best_params", as_index=False)
            .agg(frequency=("best_params", "size"), inner_auc=("inner_best_auc", "mean"))
            .sort_values(["frequency", "inner_auc", "best_params"], ascending=[False, False, True])
        )
        models[name] = json.loads(ranked.iloc[0]["best_params"])
    return models


def main() -> None:
    if FREEZE_PATH.exists():
        raise FileExistsError("Freeze already exists. Archive the previous run before starting a new analysis.")
    _, digest = load_dataset()
    manifest_path = PAPER_DIR / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not (
        manifest.get("mode") == "LWT PAPER MODE"
        and manifest.get("data_sha256") == digest
        and manifest.get("requested_models") == MODEL_NAMES
        and manifest.get("outer_limit") is None
        and manifest.get("temporal_used_for_training_tuning_or_selection") is False
    ):
        raise ValueError("A complete, matching 16-model paper-mode development run is required")
    folds = pd.read_csv(PAPER_DIR / "development_fold_metrics.csv")
    frozen = {
        "dataset_version": DATASET_VERSION,
        "primary_model": "Extra Trees",
        "development_years": [2024, 2025], "temporal_year": 2026,
        "decision_threshold": 0.5, "seed": 42,
        "parameter_selection": "Most frequent inner-CV best parameters across 25 development folds; ties by higher mean inner_best_auc then parameter JSON ascending. Primary algorithm is prespecified as Extra Trees. No temporal score or outer-test score is used to choose parameters.",
        "development_fold_metrics_sha256": _sha256(PAPER_DIR / "development_fold_metrics.csv"),
        "development_summary_sha256": _sha256(PAPER_DIR / "development_summary.csv"),
        "development_manifest_sha256": _sha256(manifest_path),
        "provenance_audit_sha256": _sha256(AUDIT_PATH) if AUDIT_PATH.is_file() else None,
        "model_code_sha256": _sha256(ROOT / "lwt_models.py"),
        "validation_code_sha256": _sha256(ROOT / "lwt_validation.py"),
        "models": select_parameters(folds),
    }
    # Validate a prospective freeze before leaving it on disk.
    FREEZE_PATH.write_text(json.dumps(frozen, ensure_ascii=False, indent=2), encoding="utf-8")
    try:
        _read_frozen_configuration(digest)
    except Exception:
        FREEZE_PATH.unlink()
        raise
    print("Development-only configuration frozen. Temporal evaluation may now run.")


if __name__ == "__main__":
    main()

"""Fast nested development CV and six-region holdout validation."""

from __future__ import annotations

import json
from math import prod
import time
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, average_precision_score, balanced_accuracy_score,
    brier_score_loss, confusion_matrix, f1_score, matthews_corrcoef,
    precision_score, roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, RepeatedStratifiedKFold, StratifiedKFold

from lwt_data import FEATURES, REGIONS
from lwt_models import MODEL_NAMES


METRICS = [
    "ROC_AUC", "PR_AUC", "Accuracy", "Sensitivity", "Specificity",
    "Precision", "F1", "Balanced_Accuracy", "MCC", "Brier",
]


def evaluate(y_true: np.ndarray, probability: np.ndarray) -> dict[str, float | int]:
    predicted = (probability >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, predicted, labels=[0, 1]).ravel()
    both_classes = len(np.unique(y_true)) == 2
    return {
        "ROC_AUC": float(roc_auc_score(y_true, probability)) if both_classes else np.nan,
        "PR_AUC": float(average_precision_score(y_true, probability)) if both_classes else np.nan,
        "Accuracy": float(accuracy_score(y_true, predicted)),
        "Sensitivity": float(tp / (tp + fn)) if tp + fn else np.nan,
        "Specificity": float(tn / (tn + fp)) if tn + fp else np.nan,
        "Precision": float(precision_score(y_true, predicted, zero_division=0)),
        "F1": float(f1_score(y_true, predicted, zero_division=0)),
        "Balanced_Accuracy": float(balanced_accuracy_score(y_true, predicted)),
        "MCC": float(matthews_corrcoef(y_true, predicted)),
        "Brier": float(brier_score_loss(y_true, probability)),
        "TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp),
    }


def fit_inner(spec: tuple, X: pd.DataFrame, y: pd.Series, seed: int, mode: str = "fast"):
    estimator, grid = spec
    inner = StratifiedKFold(n_splits=5 if mode == "paper" else 3, shuffle=True, random_state=seed)
    search_class = RandomizedSearchCV if mode == "paper" else GridSearchCV
    options = (
        {"param_distributions": grid, "n_iter": min(6, prod(len(v) for v in grid.values())), "random_state": seed}
        if mode == "paper" else {"param_grid": grid}
    )
    search = search_class(
        estimator=estimator,
        scoring="roc_auc",
        cv=inner,
        n_jobs=1,
        refit=True,
        error_score="raise",
        **options,
    )
    search.fit(X, y)
    return search


def development_cv(
    development: pd.DataFrame,
    specs: dict,
    missing: dict[str, str],
    seed: int = 42,
    outer_limit: int | None = None,
    mode: str = "fast",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    X, y = development[FEATURES], development["label"]
    splits = list(RepeatedStratifiedKFold(
        n_splits=5, n_repeats=5 if mode == "paper" else 2, random_state=seed,
    ).split(X, y))
    if outer_limit is not None:
        splits = splits[:outer_limit]
    records: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = [
        {"stage": "dependency", "model": name, "repeat": "", "fold": "", "region": "", "error": reason}
        for name, reason in missing.items()
    ]
    for name, spec in specs.items():
        for index, (train_index, test_index) in enumerate(splits):
            started = time.perf_counter()
            repeat, fold = divmod(index, 5)
            try:
                search = fit_inner(
                    spec, X.iloc[train_index], y.iloc[train_index], seed + index + 1, mode,
                )
                probability = search.predict_proba(X.iloc[test_index])[:, 1]
                row = evaluate(y.iloc[test_index].to_numpy(), probability)
                records.append({
                    "model": name, "repeat": repeat + 1, "fold": fold + 1,
                    "n_train": len(train_index), "n_test": len(test_index),
                    "positive_test": int(y.iloc[test_index].sum()),
                    "negative_test": int(len(test_index) - y.iloc[test_index].sum()),
                    "inner_best_auc": float(search.best_score_),
                    "best_params": json.dumps(search.best_params_, ensure_ascii=False, sort_keys=True),
                    "runtime_seconds": time.perf_counter() - started,
                    **row,
                })
            except Exception as exc:
                failures.append({
                    "stage": "outer_cv", "model": name, "repeat": repeat + 1,
                    "fold": fold + 1, "region": "", "error": f"{type(exc).__name__}: {exc}",
                })
        print(f"CV {name}: {sum(r['model'] == name for r in records)}/{len(splits)} folds", flush=True)
    folds = pd.DataFrame(records)
    summary = []
    for name in MODEL_NAMES:
        part = folds[folds["model"] == name] if not folds.empty else pd.DataFrame()
        row: dict[str, Any] = {
            "model": name,
            "status": "missing_dependency" if name in missing else (
                "ok" if len(part) == len(splits) else "fold_failure"
            ),
            "n_expected_folds": len(splits), "n_success_folds": len(part),
            "runtime_seconds": float(part["runtime_seconds"].sum()) if len(part) else 0.0,
        }
        for metric in METRICS:
            row[f"{metric}_mean"] = float(part[metric].mean()) if len(part) else np.nan
            row[f"{metric}_sd"] = float(part[metric].std(ddof=1)) if len(part) > 1 else np.nan
        summary.append(row)
    return folds, pd.DataFrame(summary), pd.DataFrame(failures, columns=[
        "stage", "model", "repeat", "fold", "region", "error",
    ])


def leave_one_region_out(
    development: pd.DataFrame,
    specs: dict,
    missing: dict[str, str],
    seed: int = 42,
    mode: str = "fast",
    prediction_records: list[dict[str, Any]] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for region_index, region in enumerate(REGIONS):
        test = development["产地"].eq(region)
        train = ~test  # Includes the one 略阳 development sample by prespecified rule.
        X_train, y_train = development.loc[train, FEATURES], development.loc[train, "label"]
        X_test, y_test = development.loc[test, FEATURES], development.loc[test, "label"]
        counts = {
            "region": region, "n_train": len(y_train), "n_test": len(y_test),
            "positive_train": int(y_train.sum()), "negative_train": int(len(y_train) - y_train.sum()),
            "positive_test": int(y_test.sum()), "negative_test": int(len(y_test) - y_test.sum()),
        }
        for name in MODEL_NAMES:
            started = time.perf_counter()
            if name in missing:
                rows.append({"model": name, **counts, "status": "missing_dependency", "runtime_seconds": 0.0})
                continue
            try:
                search = fit_inner(specs[name], X_train, y_train, seed + 100 + region_index, mode)
                probability = search.predict_proba(X_test)[:, 1]
                held_out_metrics = evaluate(y_test.to_numpy(), probability)
                if prediction_records is not None:
                    held_out = development.loc[test]
                    for sample_id, excel_row, year, truth, score in zip(
                        held_out.analysis_id, held_out.source_excel_row, held_out["年份"],
                        y_test, probability,
                    ):
                        prediction_records.append({
                            "model": name, "region": region,
                            "sample_id": sample_id, "analysis_id": sample_id,
                            "source_excel_row": int(excel_row), "year": int(year),
                            "y_true": int(truth), "true_label": int(truth),
                            "y_prob": float(score), "probability_premium_grade": float(score),
                        })
                rows.append({
                    "model": name, **counts, "status": "ok",
                    "inner_best_auc": float(search.best_score_),
                    "best_params": json.dumps(search.best_params_, ensure_ascii=False, sort_keys=True),
                    "runtime_seconds": time.perf_counter() - started,
                    **held_out_metrics,
                })
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
                failures.append({"stage": "loro", "model": name, "region": region, "error": error})
                rows.append({"model": name, **counts, "status": "failed", "runtime_seconds": time.perf_counter() - started})
        print(f"LORO {region}: complete", flush=True)
    return pd.DataFrame(rows), pd.DataFrame(failures, columns=["stage", "model", "region", "error"])

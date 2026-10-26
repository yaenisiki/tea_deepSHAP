"""Plot genuine stratified-bootstrap LORO ROC-AUC intervals from saved scores.

Required predictions CSV columns: model, region, analysis_id, true_label,
probability_premium_grade. One row per held-out development sample per region.
This script never fits a model. Existing aggregate metrics alone cannot supply
bootstrap confidence intervals; it fails rather than drawing pseudo-CIs.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from lwt_data import REGIONS, ROOT, load_dataset
from manuscript_plotting import apply_manuscript_style, panel_figure, save_panel


PRIMARY = "Extra Trees"
LABELS = {
    "勉县": "Mianxian", "南郑": "Nanzheng", "城固": "Chenggu",
    "宁强": "Ningqiang", "西乡": "Xixiang", "镇巴": "Zhenba",
}
INK = "#263C49"
SIGNAL = "#087F8C"
MUTED = "#AAB8BF"


def _stratified_auc_ci(y: np.ndarray, probability: np.ndarray,
                       *, replicates: int, seed: int) -> tuple[float, float, float]:
    positive = np.flatnonzero(y == 1)
    negative = np.flatnonzero(y == 0)
    if len(positive) < 2 or len(negative) < 2:
        raise ValueError("A region needs at least two samples per class for bootstrap CI")
    auc = float(roc_auc_score(y, probability))
    rng = np.random.default_rng(seed)
    draws = np.empty(replicates)
    for index in range(replicates):
        sampled = np.r_[rng.choice(negative, len(negative), replace=True),
                        rng.choice(positive, len(positive), replace=True)]
        draws[index] = roc_auc_score(y[sampled], probability[sampled])
    lower, upper = np.quantile(draws, [0.025, 0.975])
    return auc, float(lower), float(upper)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path,
                        default=ROOT / "outputs" / "lwt_paper" / "full" / "loro_predictions.csv")
    parser.add_argument("--metrics", type=Path,
                        default=ROOT / "outputs" / "lwt_paper" / "full" / "loro_metrics.csv")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "outputs" / "lwt_figures" / "loro")
    parser.add_argument("--bootstrap", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.bootstrap < 1000:
        parser.error("Use at least 1000 bootstrap replicates")
    if not args.predictions.is_file():
        raise FileNotFoundError(
            f"Per-sample LORO predictions are required for bootstrap CI: {args.predictions}. "
            "The existing loro_metrics.csv contains only aggregates; no CI can be inferred from it."
        )
    predictions = pd.read_csv(args.predictions)
    required = {"model", "region", "analysis_id", "source_excel_row", "year",
                "true_label", "probability_premium_grade"}
    if not required.issubset(predictions.columns):
        raise ValueError(f"LORO predictions need columns: {sorted(required)}")
    primary_rows = predictions.loc[predictions.model.eq(PRIMARY)]
    if not primary_rows.analysis_id.is_unique:
        raise ValueError("Each development sample must appear in only one held-out region")
    data, _ = load_dataset()
    expected = data.loc[
        data["年份"].isin([2024, 2025]) & data["产地"].isin(REGIONS),
        ["analysis_id", "source_excel_row", "年份", "产地", "label"],
    ].set_index("analysis_id")
    if set(primary_rows.analysis_id) != set(expected.index):
        raise ValueError("Primary LORO predictions must cover exactly the core-region development samples")
    aligned = expected.loc[primary_rows.analysis_id]
    if (not np.array_equal(primary_rows.source_excel_row.to_numpy(dtype=int),
                           aligned.source_excel_row.to_numpy(dtype=int))
            or not np.array_equal(primary_rows.year.to_numpy(dtype=int),
                                  aligned["年份"].to_numpy(dtype=int))
            or not np.array_equal(primary_rows.region.to_numpy(), aligned["产地"].to_numpy())
            or not np.array_equal(primary_rows.true_label.to_numpy(dtype=int),
                                  aligned.label.to_numpy(dtype=int))):
        raise ValueError("Primary LORO prediction regions or labels differ from the locked dataset")
    if not args.metrics.is_file():
        raise FileNotFoundError(args.metrics)
    metrics = pd.read_csv(args.metrics)
    records = []
    for index, region in enumerate(REGIONS):
        part = predictions.loc[predictions.model.eq(PRIMARY) & predictions.region.eq(region)]
        reference = metrics.loc[metrics.model.eq(PRIMARY) & metrics.region.eq(region)]
        if len(reference) != 1 or reference.iloc[0].status != "ok":
            raise ValueError(f"Missing completed primary-model LORO result: {region}")
        if len(part) != int(reference.iloc[0].n_test) or not part.analysis_id.is_unique:
            raise ValueError(f"LORO prediction rows are incomplete or duplicated: {region}")
        y = part.true_label.to_numpy(dtype=int)
        probability = part.probability_premium_grade.to_numpy(dtype=float)
        if (set(y) != {0, 1} or not np.isfinite(probability).all()
                or np.any((probability < 0) | (probability > 1))):
            raise ValueError(f"Invalid held-out labels or probabilities: {region}")
        if (int((y == 1).sum()) != int(reference.iloc[0].positive_test)
                or int((y == 0).sum()) != int(reference.iloc[0].negative_test)):
            raise ValueError(f"LORO class counts do not match completed metrics: {region}")
        auc, lower, upper = _stratified_auc_ci(
            y, probability, replicates=args.bootstrap, seed=args.seed + index,
        )
        if not np.isclose(auc, float(reference.iloc[0].ROC_AUC), atol=1e-10):
            raise ValueError(f"Saved LORO AUC does not match per-sample predictions: {region}")
        records.append({
            "model": PRIMARY, "region": region, "n_test": len(part),
            "positive_test": int((y == 1).sum()), "negative_test": int((y == 0).sum()),
            "ROC_AUC": auc, "CI_lower": lower, "CI_upper": upper,
            "confidence_level": 0.95, "bootstrap_method": "stratified_percentile",
            "bootstrap_replicates": args.bootstrap, "seed": args.seed + index,
        })
    extra = primary_rows.loc[~primary_rows.region.isin(REGIONS)]
    if len(extra):
        raise ValueError("Primary-model prediction file contains an unexpected non-core region")
    table = pd.DataFrame(records)
    args.output.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.output / "loro_auc_bootstrap.csv", index=False, encoding="utf-8-sig")
    apply_manuscript_style()
    fig, ax = panel_figure(90, 76)
    ypos = np.arange(6)
    ax.hlines(ypos, table.CI_lower, table.CI_upper, color=MUTED, linewidth=1.6)
    ax.scatter(table.ROC_AUC, ypos, s=31, color=SIGNAL, zorder=3)
    ax.axvline(0.5, color=MUTED, linestyle="--", linewidth=0.8)
    ax.set_yticks(ypos, [f"{LABELS[row.region]} (n={row.n_test})" for row in table.itertuples()])
    ax.invert_yaxis()
    ax.set_xlim(0, 1.02)
    ax.set_xlabel("Held-out-region ROC-AUC (95% bootstrap CI)")
    ax.tick_params(length=2.5, color=INK)
    save_panel(fig, args.output, "loro_auc_ci", tiff=True)
    secondary = metrics.loc[metrics.model.eq(PRIMARY)].set_index("region").loc[REGIONS]
    fig, ax = panel_figure(90, 76)
    for field, offset, color, label in (
        ("Balanced_Accuracy", -0.12, SIGNAL, "Balanced accuracy"),
        ("MCC", 0.12, INK, "MCC"),
    ):
        ax.scatter(secondary[field].to_numpy(dtype=float), ypos + offset,
                   s=22, color=color, label=label)
    ax.set_yticks(ypos, [LABELS[region] for region in REGIONS])
    ax.invert_yaxis()
    ax.set_xlim(min(-0.05, float(secondary.MCC.min()) - 0.05), 1.02)
    ax.set_xlabel("Held-out-region metric value")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=1)
    ax.tick_params(length=2.5, color=INK)
    save_panel(fig, args.output / "supplementary", "loro_metrics", tiff=True)
    print(f"Saved LORO bootstrap CI panel to {args.output}")


if __name__ == "__main__":
    main()

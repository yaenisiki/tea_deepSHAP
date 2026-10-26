"""Draw benchmark and temporal LWT panels from frozen result CSVs only.

This script never loads the workbook, fits a model, selects a threshold, or
computes SHAP. It creates independent, figure-number-free panels.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
)

from manuscript_plotting import apply_manuscript_style, panel_figure, save_panel


ROOT = Path(__file__).resolve().parent
PRIMARY = "Extra Trees"
INK = "#263C49"
SIGNAL = "#087F8C"
MUTED = "#AAB8BF"
PALE = "#E8EEF0"
GOLD = "#CC9140"
CMAP = LinearSegmentedColormap.from_list("lwt_rank", ["#F2F5F4", "#9AC9C8", SIGNAL])


def _read(path: Path, required: list[str]) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(path)
    data = pd.read_csv(path)
    missing = set(required) - set(data.columns)
    if missing:
        raise ValueError(f"Missing columns in {path.name}: {sorted(missing)}")
    return data


def _baseline(ax, *, grid: bool = True) -> None:
    if grid:
        ax.grid(axis="x", color=PALE, linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(length=2.5, color=INK)


def _save(fig, out: Path, stem: str) -> None:
    save_panel(fig, out, stem, tiff=True)


def plot_benchmark(summary: pd.DataFrame, folds: pd.DataFrame, out: Path) -> None:
    ok = summary.loc[summary.status.eq("ok")].copy()
    if len(ok) != 16 or set(ok.model) != set(folds.model):
        raise ValueError("Expected all 16 successful development models")
    if not (ok.n_success_folds.astype(int) == 25).all():
        raise ValueError("Expected 25 successful outer folds per model")
    names = ok.sort_values("ROC_AUC_mean", ascending=False).model.tolist()
    samples = [folds.loc[folds.model.eq(name), "ROC_AUC"].to_numpy(dtype=float) for name in names]
    if any(len(x) != 25 or not np.isfinite(x).all() for x in samples):
        raise ValueError("Incomplete outer-fold AUC data")

    fig, ax = panel_figure(190, 123)
    bp = ax.boxplot(samples, vert=False, positions=np.arange(16), widths=0.54,
                    patch_artist=True, showfliers=False, medianprops={"color": INK, "linewidth": 1.2},
                    whiskerprops={"color": MUTED}, capprops={"color": MUTED})
    for patch, name in zip(bp["boxes"], names):
        patch.set_facecolor(SIGNAL if name == PRIMARY else "#D9E3E7")
        patch.set_edgecolor(SIGNAL if name == PRIMARY else MUTED)
    for i, values in enumerate(samples):
        jitter = np.linspace(-0.16, 0.16, len(values))
        ax.scatter(values, i + jitter, s=5, color=INK, alpha=0.27, linewidths=0, zorder=3)
    ax.set_yticks(np.arange(16), names)
    ax.invert_yaxis()
    ax.set_xlim(0.25, 1.01)
    ax.set_xlabel("Outer-CV ROC-AUC")
    _baseline(ax)
    _save(fig, out, "model_auc_distribution")

    fig, ax = panel_figure(190, 125)
    ypos = np.arange(len(names))
    ordered = ok.set_index("model").loc[names]
    primary_index = names.index(PRIMARY)
    ax.axhspan(primary_index - 0.45, primary_index + 0.45, color="#EAF3F2", zorder=0)
    for field, offset, color, marker, label in (
        ("ROC_AUC_mean", -0.23, SIGNAL, "o", "ROC-AUC"),
        ("Balanced_Accuracy_mean", 0, GOLD, "s", "Balanced accuracy"),
        ("MCC_mean", 0.23, INK, "^", "MCC"),
    ):
        ax.scatter(ordered[field], ypos + offset, color=color, marker=marker,
                   s=20, alpha=0.43, label=label, zorder=3)
        ax.scatter(float(ordered.loc[PRIMARY, field]), primary_index + offset,
                   color=color, marker=marker, s=32, edgecolors=INK,
                   linewidths=0.45, zorder=4)
    ax.set_yticks(ypos, names)
    for tick in ax.get_yticklabels():
        if tick.get_text() == PRIMARY:
            tick.set_color(SIGNAL)
            tick.set_fontweight("bold")
    ax.invert_yaxis()
    ax.set_xlim(min(-0.05, float(ordered["MCC_mean"].min()) - 0.05), 1.02)
    ax.set_xlabel("Mean outer-CV metric value (25 folds)")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=3)
    _baseline(ax)
    _save(fig, out, "model_multimetric_dotplot")

    supplementary = out / "supplementary"
    fields = ["ROC_AUC_mean", "Balanced_Accuracy_mean", "MCC_mean"]
    labels = ["ROC-AUC", "Balanced accuracy", "MCC"]
    values = ordered[fields].to_numpy(dtype=float)
    ranks = np.column_stack([ordered[field].rank(pct=True).to_numpy() for field in fields])
    fig, ax = panel_figure(140, 116)
    ax.pcolormesh(np.arange(4), np.arange(17), ranks, cmap=CMAP,
                  vmin=0, vmax=1, edgecolors="white", linewidth=1)
    ax.set_xlim(0, 3)
    ax.set_ylim(16, 0)
    ax.set_xticks(np.arange(3) + 0.5, labels)
    ax.set_yticks(np.arange(16) + 0.5, names)
    ax.xaxis.tick_top()
    ax.tick_params(axis="both", length=0)
    for row in range(16):
        for col in range(3):
            ax.text(col + 0.5, row + 0.5, f"{values[row, col]:.2f}",
                    ha="center", va="center", fontsize=8,
                    color="white" if ranks[row, col] > 0.72 else INK)
    ax.text(0.99, -0.055, "Color: within-metric rank; values: mean across 25 folds",
            transform=ax.transAxes, ha="right", va="top", fontsize=8, color=INK)
    _save(fig, supplementary, "model_benchmark_heatmap")


def _validated_temporal(predictions: pd.DataFrame, metrics: pd.DataFrame, bootstrap: pd.DataFrame):
    row = metrics.loc[metrics.model.eq(PRIMARY)]
    boot = bootstrap.loc[bootstrap.model.eq(PRIMARY)]
    if len(row) != 1 or len(boot) != 1 or len(predictions) != 201:
        raise ValueError("Primary temporal records are incomplete")
    if not predictions.year.astype(str).eq("2026").all():
        raise ValueError("Temporal predictions include a non-2026 row")
    if not predictions.analysis_id.is_unique or not predictions.source_excel_row.is_unique:
        raise ValueError("Temporal prediction identifiers are not unique")
    y = predictions.true_label.to_numpy(dtype=int)
    p = predictions.probability_special_grade.to_numpy(dtype=float)
    if set(y) != {0, 1} or not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
        raise ValueError("Invalid labels or probabilities")
    reported = row.iloc[0]
    if not np.isclose(float(reported.threshold), 0.5):
        raise ValueError("Temporal threshold is not the frozen 0.5")
    checks = {
        "ROC_AUC": roc_auc_score(y, p),
        "PR_AUC": average_precision_score(y, p),
        "Brier": brier_score_loss(y, p),
    }
    for key, calculated in checks.items():
        if not np.isclose(calculated, float(reported[key]), atol=1e-10):
            raise ValueError(f"Temporal {key} does not match the frozen metrics")
    cm = confusion_matrix(y, p >= 0.5, labels=[0, 1])
    if not np.array_equal(cm.ravel(), reported[["TN", "FP", "FN", "TP"]].to_numpy(dtype=int)):
        raise ValueError("Temporal confusion matrix does not match the frozen metrics")
    return y, p, reported, boot.iloc[0], cm


def plot_temporal(predictions: pd.DataFrame, metrics: pd.DataFrame,
                  sensitivity: pd.DataFrame, bootstrap: pd.DataFrame, out: Path) -> None:
    y, p, reported, boot, cm = _validated_temporal(predictions, metrics, bootstrap)
    fpr, tpr, _ = roc_curve(y, p)
    fig, ax = panel_figure(90, 75)
    ax.plot(fpr, tpr, color=SIGNAL, linewidth=1.7)
    ax.plot([0, 1], [0, 1], linestyle="--", color=MUTED, linewidth=0.8)
    ax.text(0.98, 0.04, f"AUC {float(reported.ROC_AUC):.3f}\n95% CI {float(boot.CI_lower):.3f}–{float(boot.CI_upper):.3f}",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=8, color=INK)
    ax.set(xlim=(0, 1), ylim=(0, 1.02), xlabel="False positive rate", ylabel="True positive rate")
    _baseline(ax, grid=False)
    _save(fig, out, "temporal_roc")

    precision, recall, _ = precision_recall_curve(y, p)
    fig, ax = panel_figure(90, 75)
    ax.step(recall, precision, where="post", color=SIGNAL, linewidth=1.7)
    ax.axhline(y.mean(), linestyle="--", color=MUTED, linewidth=0.8)
    ax.text(0.98, 0.04, f"AP {float(reported.PR_AUC):.3f}\nPrevalence {y.mean():.3f}",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=8, color=INK)
    ax.set(xlim=(0, 1), ylim=(0, 1.02), xlabel="Recall", ylabel="Precision")
    _baseline(ax, grid=False)
    _save(fig, out, "temporal_pr")

    observed, predicted = calibration_curve(y, p, n_bins=8, strategy="quantile")
    fig, ax = panel_figure(90, 75)
    ax.plot([0, 1], [0, 1], linestyle="--", color=MUTED, linewidth=0.8)
    ax.plot(predicted, observed, marker="o", color=SIGNAL, linewidth=1.5, markersize=4)
    ax.text(0.98, 0.04, f"Brier {float(reported.Brier):.3f}", transform=ax.transAxes,
            ha="right", va="bottom", fontsize=8, color=INK)
    ax.set(xlim=(0, 1), ylim=(0, 1.02), xlabel="Mean probability of premium grade",
           ylabel="Observed premium-grade fraction")
    _baseline(ax, grid=False)
    _save(fig, out / "supplementary", "temporal_calibration")

    fig, ax = panel_figure(90, 75)
    normalized = cm / cm.sum(axis=1, keepdims=True)
    ax.pcolormesh(np.arange(3), np.arange(3), normalized, cmap=CMAP,
                  vmin=0, vmax=1, edgecolors="white", linewidth=1.5)
    ax.set(xlim=(0, 2), ylim=(2, 0), xlabel="Predicted grade", ylabel="Observed grade")
    ax.set_xticks([0.5, 1.5], ["Non-premium grade", "Premium grade"])
    ax.set_yticks([0.5, 1.5], ["Non-premium grade", "Premium grade"])
    ax.tick_params(length=0)
    for row in range(2):
        for col in range(2):
            ax.text(col + 0.5, row + 0.5, f"{cm[row, col]}\n({normalized[row, col]:.0%})",
                    ha="center", va="center", fontsize=8,
                    color="white" if normalized[row, col] > 0.7 else INK)
    _save(fig, out, "temporal_confusion_matrix")

    if sensitivity.empty:
        return  # No exclusion audit was supplied; no sensitivity panel is claimed.
    comparison = sensitivity.loc[sensitivity.model.eq(PRIMARY)]
    if len(comparison) != 1:
        raise ValueError("Expected one provenance-exclusion result for Extra Trees")
    excluded = comparison.iloc[0]
    if not 0 < int(excluded.n_test) < int(reported.n_test):
        raise ValueError("Sensitivity cohort must be a strict subset")
    fields = ["ROC_AUC", "Balanced_Accuracy", "MCC", "Brier"]
    labels = ["ROC-AUC", "Balanced accuracy", "MCC", "Brier (lower better)"]
    all_values = reported[fields].to_numpy(dtype=float)
    excluded_values = excluded[fields].to_numpy(dtype=float)
    if not np.isfinite(np.r_[all_values, excluded_values]).all():
        raise ValueError("Temporal sensitivity metrics contain non-finite values")
    fig, ax = panel_figure(90, 75)
    ypos = np.arange(4)
    delta = excluded_values - all_values
    ax.axvline(0, color=MUTED, linestyle="--", linewidth=0.8)
    ax.scatter(delta, ypos, s=34, color=SIGNAL, zorder=3)
    for position, change in zip(ypos, delta):
        ax.annotate(f"{change:+.3f}", (change, position),
                    xytext=(5 if change >= 0 else -5, 0), textcoords="offset points",
                    ha="left" if change >= 0 else "right", va="center", fontsize=8)
    ax.set_yticks(ypos, labels)
    ax.invert_yaxis()
    limit = max(0.025, float(np.max(np.abs(delta))) * 1.8)
    ax.set_xlim(-limit, limit)
    ax.set_xlabel(f"Change after exclusion (n={int(excluded.n_test)} minus n={int(reported.n_test)})")
    _baseline(ax, grid=False)
    _save(fig, out, "temporal_sensitivity")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--section", choices=("benchmark", "temporal"), required=True)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    apply_manuscript_style()
    source = ROOT / "outputs"
    paper = source / "lwt_paper" / "full"
    temporal = source / "lwt_temporal"
    output = args.output or source / "lwt_figures" / args.section
    if args.section == "benchmark":
        summary = _read(paper / "development_summary.csv", [
            "model", "status", "n_success_folds", "ROC_AUC_mean",
            "Balanced_Accuracy_mean", "MCC_mean",
        ])
        folds = _read(paper / "development_fold_metrics.csv", ["model", "ROC_AUC"])
        plot_benchmark(summary, folds, output)
    else:
        predictions = _read(temporal / "temporal_predictions.csv", [
            "analysis_id", "source_excel_row", "year", "true_label", "probability_special_grade",
        ])
        metric_columns = ["model", "n_test", "threshold", "ROC_AUC", "PR_AUC", "Brier",
                          "Balanced_Accuracy", "MCC", "TN", "FP", "FN", "TP"]
        metrics = _read(temporal / "temporal_metrics.csv", metric_columns)
        sensitivity = _read(temporal / "temporal_sensitivity_metrics.csv", metric_columns)
        bootstrap = _read(temporal / "temporal_primary_bootstrap.csv", ["model", "CI_lower", "CI_upper"])
        plot_temporal(predictions, metrics, sensitivity, bootstrap, output)
    print(f"Saved {args.section} panels to {output}")


if __name__ == "__main__":
    main()

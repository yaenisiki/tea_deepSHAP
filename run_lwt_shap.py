"""Post-validation SHAP for the frozen LWT Extra Trees primary model.

Run only after ``run_lwt_temporal.py`` has completed. This script does not tune,
select features/models/thresholds, or alter the source workbook. Its repeated
CV fits use the already-frozen hyperparameters solely for OOF SHAP stability.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, Normalize
from scipy.stats import spearmanr
from sklearn.base import clone
from sklearn.model_selection import RepeatedStratifiedKFold
from statsmodels.stats.multitest import multipletests

from lwt_data import CONCENTRATION_UNIT_SOURCE, CONCENTRATION_UNIT, FEATURES, REGIONS, ROOT, load_dataset
from lwt_models import model_specs
from manuscript_plotting import apply_manuscript_style, panel_figure, save_panel
from run_lwt_temporal import FREEZE_PATH, OUTPUT_DIR as TEMPORAL_DIR, _read_frozen_configuration
import shap


PRIMARY = "Extra Trees"
INK = "#263C49"
SIGNAL = "#087F8C"
GOLD = "#CC9140"
MUTED = "#AAB8BF"
PALE = "#E8EEF0"
HEAT = LinearSegmentedColormap.from_list("lwt_shap", ["#F3F6F5", "#A9D1D0", SIGNAL])
VALUE_CMAP = LinearSegmentedColormap.from_list("lwt_value", ["#386DA8", "#E7EBE9", "#C77749"])
REGION_ENGLISH = {
    "勉县": "Mianxian", "南郑": "Nanzheng", "城固": "Chenggu",
    "宁强": "Ningqiang", "西乡": "Xixiang", "镇巴": "Zhenba",
}
DEPENDENCE_FEATURES = ("EC", "EGCG", "ECG")
MAIN_DEPENDENCE = {"EC", "EGCG"}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _verified_inputs(data: pd.DataFrame, data_sha256: str):
    frozen = _read_frozen_configuration(data_sha256)
    manifest_path = TEMPORAL_DIR / "temporal_run_manifest.json"
    predictions_path = TEMPORAL_DIR / "temporal_predictions.csv"
    if not manifest_path.is_file() or not predictions_path.is_file():
        raise FileNotFoundError("Run the frozen 2026 temporal validation before SHAP")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not (
        manifest.get("data_sha256") == data_sha256
        and manifest.get("frozen_config_sha256") == _sha256(FREEZE_PATH)
        and manifest.get("primary_model") == PRIMARY
        and manifest.get("development_n") == 186
        and manifest.get("temporal_n") == 201
        and manifest.get("no_2026_training_tuning_or_model_selection") is True
        and manifest.get("models", {}).get(PRIMARY) == frozen["models"][PRIMARY]
    ):
        raise ValueError("Saved temporal validation does not match the frozen design")
    predictions = pd.read_csv(predictions_path)
    required = {
        "analysis_id", "source_excel_row", "year", "true_label",
        "probability_special_grade", "provenance_pending",
    }
    if len(predictions) != 201 or not required.issubset(predictions.columns):
        raise ValueError("Saved 2026 primary-model predictions are incomplete")
    if not predictions.analysis_id.is_unique or not predictions.source_excel_row.is_unique:
        raise ValueError("Duplicate temporal prediction identifiers")
    development = data.loc[data["年份"].isin([2024, 2025])].copy()
    temporal = data.loc[data["年份"].eq(2026)].copy()
    if len(development) != 186 or len(temporal) != 201:
        raise ValueError("Development or temporal cohort size has changed")
    predictions = predictions.set_index("analysis_id").loc[temporal.analysis_id]
    if not (
        predictions.year.astype(int).eq(2026).all()
        and np.array_equal(predictions.source_excel_row.to_numpy(dtype=int), temporal.source_excel_row.to_numpy())
        and np.array_equal(predictions.true_label.to_numpy(dtype=int), temporal.label.to_numpy())
    ):
        raise ValueError("Saved 2026 predictions do not align with source rows")
    return frozen, development, temporal, predictions, manifest_path, predictions_path


def _frozen_estimator(frozen: dict):
    specs, _ = model_specs(seed=frozen["seed"], mode="paper")
    estimator, grid = specs[PRIMARY]
    params = frozen["models"][PRIMARY]
    if set(params) != set(grid) or any(params[key] not in grid[key] for key in params):
        raise ValueError("Frozen Extra Trees parameters differ from the paper-mode search space")
    estimator.set_params(**params)
    return estimator


def assert_premium_positive_class(data: pd.DataFrame) -> None:
    """Lock the interpretation of class 1 to the original premium-grade label."""
    premium = data["品级"].eq("特级")
    non_premium = data["品级"].eq("非特级")
    if not (premium.any() and non_premium.any() and (premium | non_premium).all()):
        raise ValueError("Original grade labels are not exactly premium/non-premium")
    if not (data.loc[premium, "label"].eq(1).all()
            and data.loc[non_premium, "label"].eq(0).all()):
        raise ValueError("Class 1 must map to Premium grade")


def _positive_shap(fitted, X: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, float]:
    """Return class-1 SHAP in probability units, or fail the additivity audit."""
    transformed = fitted[:-1].transform(X)
    model = fitted.named_steps["model"]
    if list(model.classes_) != [0, 1] or list(fitted.classes_) != [0, 1]:
        raise ValueError("Unexpected Extra Trees class order")
    explainer = shap.TreeExplainer(
        model, feature_perturbation="tree_path_dependent", model_output="raw",
    )
    try:
        result = explainer.shap_values(transformed, check_additivity=False)
    except TypeError:
        result = explainer.shap_values(transformed)
    if isinstance(result, list) and len(result) == 2:
        positive = np.asarray(result[1], dtype=float)
    else:
        array = np.asarray(result, dtype=float)
        if array.shape == (len(X), len(FEATURES), 2):
            positive = array[:, :, 1]
        elif array.shape == (2, len(X), len(FEATURES)):
            positive = array[1]
        else:
            raise ValueError(f"Unsupported binary SHAP shape: {array.shape}")
    expected = np.asarray(explainer.expected_value, dtype=float).reshape(-1)
    if len(expected) != 2 or positive.shape != (len(X), len(FEATURES)):
        raise ValueError("Expected one base value per class and ten SHAP features")
    if not 0 <= expected[1] <= 1:
        raise ValueError("Explainer base value is not on the probability scale")
    probability = fitted.predict_proba(X)[:, 1]
    if not np.isfinite(probability).all() or np.any((probability < 0) | (probability > 1)):
        raise ValueError("Extra Trees class-1 output is not a valid probability")
    error = np.abs(expected[1] + positive.sum(axis=1) - probability)
    max_error = float(error.max())
    if not np.isfinite(positive).all() or max_error > 1e-4:
        raise ValueError(f"Class-1 SHAP is not additive to predict_proba: {max_error}")
    return positive, np.full(len(X), expected[1]), max_error


def _oof_shap(estimator, development: pd.DataFrame, seed: int):
    X = development[FEATURES].reset_index(drop=True)
    y = development.label.to_numpy()
    values = np.full((5, len(X), len(FEATURES)), np.nan)
    fold_records = []
    max_error = 0.0
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=seed)
    for index, (train_idx, test_idx) in enumerate(cv.split(X, y)):
        fitted = clone(estimator).fit(X.iloc[train_idx], y[train_idx])
        explained, _, error = _positive_shap(fitted, X.iloc[test_idx])
        repeat, fold = divmod(index, 5)
        values[repeat, test_idx, :] = explained
        max_error = max(max_error, error)
        for feature, importance in zip(FEATURES, np.mean(np.abs(explained), axis=0)):
            fold_records.append({
                "repeat": repeat + 1, "fold": fold + 1, "n_test": len(test_idx),
                "feature": feature, "mean_abs_shap": float(importance),
            })
        print(f"Fixed-parameter OOF SHAP: {index + 1}/25", flush=True)
    if not np.isfinite(values).all():
        raise ValueError("Some development samples lack out-of-fold SHAP values")
    return values, pd.DataFrame(fold_records), max_error


def _save_table(table: pd.DataFrame, output: Path, name: str) -> None:
    table.to_csv(output / name, index=False, encoding="utf-8-sig")


def _shap_table(rows: pd.DataFrame, values: np.ndarray, base: np.ndarray | None = None,
                probability: np.ndarray | None = None) -> pd.DataFrame:
    table = rows[["analysis_id", "source_excel_row", "年份", "产地", "label"]].copy()
    table = table.rename(columns={"年份": "year", "产地": "region", "label": "true_label"})
    if base is not None:
        table["base_probability"] = base
    if probability is not None:
        table["probability_special_grade"] = probability
    for index, feature in enumerate(FEATURES):
        table[f"shap_{feature}"] = values[:, index]
    return table


def _importance_table(values: np.ndarray) -> pd.DataFrame:
    table = pd.DataFrame({
        "feature": FEATURES,
        "mean_abs_shap": np.mean(np.abs(values), axis=0),
    }).sort_values("mean_abs_shap", ascending=False, kind="stable")
    table.insert(0, "rank", np.arange(1, len(table) + 1))
    return table.reset_index(drop=True)


def _stability_tables(development: pd.DataFrame, temporal: pd.DataFrame,
                      development_shap: np.ndarray, temporal_shap: np.ndarray):
    years = []
    for year in [2024, 2025, 2026]:
        rows, values = (development, development_shap) if year != 2026 else (temporal, temporal_shap)
        subset = rows["年份"].to_numpy() == year
        for feature, mean_abs in zip(FEATURES, np.mean(np.abs(values[subset]), axis=0)):
            years.append({"year": year, "cohort": "development" if year != 2026 else "temporal",
                          "n": int(subset.sum()), "feature": feature, "mean_abs_shap": float(mean_abs)})
    regions = []
    for region in REGIONS:
        subset = temporal["产地"].to_numpy() == region
        if not subset.any():
            raise ValueError(f"No 2026 temporal samples in core region: {region}")
        for feature, mean_abs in zip(FEATURES, np.mean(np.abs(temporal_shap[subset]), axis=0)):
            regions.append({"region": region, "cohort": "2026 temporal", "n": int(subset.sum()),
                            "feature": feature, "mean_abs_shap": float(mean_abs)})
    return pd.DataFrame(years), pd.DataFrame(regions)


def _basic_axis(ax) -> None:
    ax.tick_params(length=2.5, color=INK)
    ax.set_axisbelow(True)


def _plot_global(importance: pd.DataFrame, output: Path) -> None:
    ordered = importance.iloc[::-1]
    fig, ax = panel_figure(90, 78)
    ax.barh(ordered.feature, ordered.mean_abs_shap, color=SIGNAL, height=0.62)
    ax.set_xlabel("OOF mean |SHAP| (probability units)")
    ax.set_xlim(left=0)
    _basic_axis(ax)
    save_panel(fig, output, "shap_global_importance", tiff=True)


def _swarm_offsets(values: np.ndarray, width: float = 0.22) -> np.ndarray:
    """Deterministic beeswarm-like packing within small SHAP-value bins."""
    edges = np.linspace(float(values.min()), float(values.max()) + 1e-12, 32)
    buckets = np.digitize(values, edges)
    offset = np.zeros(len(values))
    for bucket in np.unique(buckets):
        indices = np.flatnonzero(buckets == bucket)
        if len(indices) > 1:
            offset[indices] = np.linspace(-width, width, len(indices))
    return offset


def _plot_beeswarm(importance: pd.DataFrame, development: pd.DataFrame,
                   first_repeat_shap: np.ndarray, output: Path) -> None:
    order = importance.feature.tolist()
    fig, ax = panel_figure(140, 100)
    for row, feature in enumerate(order):
        index = FEATURES.index(feature)
        x = first_repeat_shap[:, index]
        raw = development[feature].to_numpy(dtype=float)
        low, high = np.quantile(raw, [0.05, 0.95])
        color = np.clip((raw - low) / max(high - low, 1e-12), 0, 1)
        ax.scatter(x, row + _swarm_offsets(x), c=color, cmap=VALUE_CMAP,
                   vmin=0, vmax=1, s=9, alpha=0.75, edgecolors="none")
    ax.axvline(0, color=MUTED, linewidth=0.8)
    ax.set_yticks(np.arange(len(order)), order)
    ax.invert_yaxis()
    ax.set_xlabel("SHAP value for premium-grade probability")
    colorbar = fig.colorbar(plt.cm.ScalarMappable(norm=Normalize(0, 1), cmap=VALUE_CMAP),
                           ax=ax, fraction=0.035, pad=0.02)
    colorbar.set_label("Relative feature value", fontsize=8)
    colorbar.set_ticks([0, 1])
    colorbar.set_ticklabels(["Low", "High"])
    colorbar.ax.tick_params(labelsize=8)
    _basic_axis(ax)
    save_panel(fig, output, "shap_beeswarm", tiff=True)


def _plot_dependence(importance: pd.DataFrame, development: pd.DataFrame,
                     temporal: pd.DataFrame, oof_shap: np.ndarray,
                     temporal_shap: np.ndarray, output: Path) -> None:
    for feature in DEPENDENCE_FEATURES:
        index = FEATURES.index(feature)
        fig, ax = panel_figure(90, 75)
        ax.scatter(development[feature], oof_shap[:, index], s=15,
                   color=MUTED, alpha=0.65, linewidths=0, label="2024–2025 OOF")
        ax.scatter(temporal[feature], temporal_shap[:, index], s=16,
                   facecolors="none", edgecolors=SIGNAL, linewidths=0.65,
                   alpha=0.8, label="2026 held-out")
        ax.axhline(0, color=PALE, linewidth=0.8)
        ax.set_xlabel(f"{feature} concentration ({CONCENTRATION_UNIT})")
        ax.set_ylabel("SHAP value for premium-grade probability")
        ax.legend(loc="best", fontsize=8, markerscale=1.1)
        _basic_axis(ax)
        destination = output if feature in MAIN_DEPENDENCE else output / "supplementary"
        save_panel(fig, destination, f"shap_dependence_{feature}", tiff=True)


def _plot_fold_stability(folds: pd.DataFrame, importance: pd.DataFrame,
                         output: Path) -> None:
    ordered = importance.feature.iloc[::-1].tolist()
    grouped = folds.groupby("feature").mean_abs_shap.agg(["mean", "std"]).loc[ordered]
    ypos = np.arange(len(ordered))
    fig, ax = panel_figure(90, 78)
    ax.hlines(ypos, np.maximum(0, grouped["mean"] - grouped["std"]),
              grouped["mean"] + grouped["std"], color=MUTED, linewidth=2)
    ax.scatter(grouped["mean"], ypos, color=SIGNAL, s=25, zorder=3)
    ax.set_yticks(ypos, ordered)
    ax.set_xlim(left=0)
    ax.set_xlabel("Fold mean |SHAP| (mean ± SD; 25 folds)")
    _basic_axis(ax)
    save_panel(fig, output, "shap_fold_stability", tiff=True)


def _plot_stability_heatmap(table: pd.DataFrame, row_key: str, row_labels: list[str],
                            row_values: list, feature_order: list[str], output: Path,
                            stem: str, height_mm: float,
                            value_col: str = "relative_importance") -> None:
    matrix = table.pivot(index=row_key, columns="feature", values=value_col)
    values = matrix.loc[row_values, feature_order].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError(f"Non-finite stability values: {stem}")
    if value_col == "relative_importance" and not np.allclose(values.sum(axis=1), 1, atol=1e-10):
        raise ValueError(f"Relative stability rows must each sum to one: {stem}")
    fig, ax = panel_figure(140, height_mm)
    mesh = ax.pcolormesh(np.arange(len(feature_order) + 1), np.arange(len(row_values) + 1),
                         values, cmap=HEAT, vmin=0, vmax=max(float(values.max()), 1e-8),
                         edgecolors="white", linewidth=0.8)
    ax.set_xlim(0, len(feature_order))
    ax.set_ylim(len(row_values), 0)
    ax.set_xticks(np.arange(len(feature_order)) + 0.5, feature_order)
    ax.set_yticks(np.arange(len(row_values)) + 0.5, row_labels)
    ax.tick_params(length=0)
    colorbar = fig.colorbar(mesh, ax=ax, fraction=0.03, pad=0.02)
    colorbar.set_label("Relative mean |SHAP|" if value_col == "relative_importance"
                       else "Mean |SHAP| (probability)", fontsize=8)
    colorbar.ax.tick_params(labelsize=8)
    save_panel(fig, output, stem, tiff=True)


def _plot_local(temporal: pd.DataFrame, values: np.ndarray, base: np.ndarray,
                probabilities: np.ndarray, output: Path) -> None:
    index = int(np.argmin(temporal.source_excel_row.to_numpy()))
    if not (np.isfinite(base[index]) and 0 <= base[index] <= 1
            and np.isclose(base[index] + values[index].sum(), probabilities[index], atol=1e-4)):
        raise ValueError("Local base must come from the explainer and sum to predict_proba")
    contributions = values[index]
    order = np.argsort(-np.abs(contributions))
    cumulative = np.r_[base[index], base[index] + np.cumsum(contributions[order])]
    fig, ax = panel_figure(140, 94)
    for row, feature_index in enumerate(order):
        start, end = cumulative[row:row + 2]
        ax.barh(row, abs(end - start), left=min(start, end), height=0.6,
                color=SIGNAL if end >= start else GOLD)
    ax.axvline(base[index], color=MUTED, linestyle="--", linewidth=0.8,
               label=f"Base {base[index]:.3f}")
    ax.axvline(probabilities[index], color=INK, linewidth=1,
               label=f"Prediction {probabilities[index]:.3f}")
    ax.set_yticks(np.arange(len(FEATURES)), [FEATURES[i] for i in order])
    ax.invert_yaxis()
    ax.set_xlabel("Cumulative probability of premium grade")
    ax.set_xlim(min(-0.03, float(cumulative.min()) - 0.03),
                max(1.03, float(cumulative.max()) + 0.03))
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=2)
    _basic_axis(ax)
    save_panel(fig, output, "shap_local_explanation", tiff=True)


def _relative_importance(table: pd.DataFrame, group_col: str) -> pd.DataFrame:
    result = table.copy()
    denominator = result.groupby(group_col).mean_abs_shap.transform("sum")
    if (denominator <= 0).any() or not np.isfinite(denominator).all():
        raise ValueError(f"Invalid SHAP importance denominator for {group_col}")
    result["relative_importance"] = result.mean_abs_shap / denominator
    sums = result.groupby(group_col).relative_importance.sum()
    if not np.allclose(sums.to_numpy(), 1, atol=1e-10):
        raise ValueError(f"Relative SHAP importance must sum to one within each {group_col}")
    return result


def _rank_correlations(table: pd.DataFrame, group_col: str, order: list) -> pd.DataFrame:
    matrix = table.pivot(index=group_col, columns="feature", values="relative_importance")
    matrix = matrix.loc[order, FEATURES]
    records = []
    for left_index, left in enumerate(order):
        for right in order[left_index + 1:]:
            rho, p_value = spearmanr(matrix.loc[left], matrix.loc[right])
            if not np.isfinite(rho) or not np.isfinite(p_value):
                raise ValueError(f"Undefined SHAP rank correlation: {left} vs {right}")
            records.append({"group_a": left, "group_b": right, "n_features": len(FEATURES),
                            "spearman_rho": float(rho), "raw_p": float(p_value)})
    result = pd.DataFrame(records)
    result["FDR_q"] = multipletests(result.raw_p, method="fdr_bh")[1]
    return result


def _plot_rank_correlations(ranks: pd.DataFrame, order: list, labels: list[str],
                            output: Path, stem: str, size_mm: tuple[int, int]) -> None:
    """A separate numeric panel paired with each relative-importance heatmap."""
    positions = {group: index for index, group in enumerate(order)}
    values = np.full((len(order), len(order)), np.nan)
    np.fill_diagonal(values, 1.0)
    for row in ranks.itertuples():
        values[positions[row.group_b], positions[row.group_a]] = row.spearman_rho
    expected_pairs = len(order) * (len(order) - 1) // 2
    if (len(ranks) != expected_pairs
            or not np.isfinite(values[np.tril_indices(len(order))]).all()):
        raise ValueError(f"Incomplete SHAP rank correlations: {stem}")
    fig, ax = panel_figure(*size_mm)
    mesh = ax.pcolormesh(np.arange(len(order) + 1), np.arange(len(order) + 1),
                         np.ma.masked_invalid(values), cmap=VALUE_CMAP,
                         vmin=-1, vmax=1, edgecolors="white", linewidth=0.8)
    ax.set_xlim(0, len(order))
    ax.set_ylim(len(order), 0)
    ax.set_xticks(np.arange(len(order)) + 0.5, labels, rotation=45, ha="right")
    ax.set_yticks(np.arange(len(order)) + 0.5, labels)
    ax.tick_params(length=0)
    for row in range(len(order)):
        for col in range(row + 1):
            rho = values[row, col]
            ax.text(col + 0.5, row + 0.5, f"{rho:.2f}", ha="center", va="center",
                    fontsize=8, color="white" if abs(rho) > 0.75 else INK)
    colorbar = fig.colorbar(mesh, ax=ax, fraction=0.04, pad=0.025)
    colorbar.set_label("Spearman rank correlation", fontsize=8)
    colorbar.ax.tick_params(labelsize=8)
    save_panel(fig, output, stem, tiff=True)


def _plot_cohort_comparison(importance: pd.DataFrame, temporal_shap: np.ndarray,
                            temporal_n: int, output: Path) -> pd.DataFrame:
    comparison = importance[["feature", "mean_abs_shap"]].rename(
        columns={"mean_abs_shap": "development_oof_mean_abs_shap"},
    ).copy()
    temporal_means = dict(zip(FEATURES, np.mean(np.abs(temporal_shap), axis=0)))
    comparison["temporal_mean_abs_shap"] = comparison.feature.map(temporal_means)
    comparison["development_n"] = 186
    comparison["development_oof_repeats"] = 5
    comparison["temporal_n"] = temporal_n
    values = comparison[["development_oof_mean_abs_shap", "temporal_mean_abs_shap"]].to_numpy()
    if not np.isfinite(values).all():
        raise ValueError("Non-finite development-vs-temporal SHAP importance")
    fig, ax = panel_figure(140, 87)
    ypos = np.arange(len(comparison))
    ax.hlines(ypos, np.minimum(values[:, 0], values[:, 1]),
              np.maximum(values[:, 0], values[:, 1]), color=MUTED, linewidth=1)
    ax.scatter(values[:, 0], ypos - 0.08, s=28, color=SIGNAL,
               label="Development OOF (n=186)", zorder=3)
    ax.scatter(values[:, 1], ypos + 0.08, s=28, facecolors="white",
               edgecolors=GOLD, linewidths=1.2,
               label=f"2026 temporal (n={temporal_n})", zorder=3)
    ax.set_yticks(ypos, comparison.feature)
    ax.invert_yaxis()
    ax.set_xlim(left=0)
    ax.set_xlabel("Mean |SHAP| (probability units)")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=2)
    _basic_axis(ax)
    save_panel(fig, output, "shap_development_temporal_comparison", tiff=True)
    return comparison


def _render_panels(output: Path, importance: pd.DataFrame, fold_table: pd.DataFrame,
                   year_table: pd.DataFrame, region_table: pd.DataFrame,
                   development: pd.DataFrame, temporal: pd.DataFrame,
                   oof_first_repeat: np.ndarray, temporal_shap: np.ndarray,
                   temporal_base: np.ndarray, probability: np.ndarray) -> None:
    apply_manuscript_style()
    supplementary = output / "supplementary"
    year_relative = _relative_importance(year_table, "year")
    region_relative = _relative_importance(region_table, "region")
    _save_table(year_relative, output, "shap_year_relative_importance.csv")
    _save_table(region_relative, output, "shap_region_relative_importance.csv")
    year_ranks = _rank_correlations(year_relative, "year", [2024, 2025, 2026])
    region_ranks = _rank_correlations(region_relative, "region", REGIONS)
    _save_table(year_ranks, output, "shap_year_rank_correlations.csv")
    _save_table(region_ranks, output, "shap_region_rank_correlations.csv")
    _plot_global(importance, supplementary)
    _plot_beeswarm(importance, development, oof_first_repeat, output)
    comparison = _plot_cohort_comparison(importance, temporal_shap, len(temporal), output)
    _save_table(comparison, output, "shap_development_temporal_comparison.csv")
    _plot_dependence(importance, development, temporal, oof_first_repeat, temporal_shap,
                     output)
    _plot_fold_stability(fold_table, importance, supplementary)
    ordered_features = importance.feature.tolist()
    year_counts = year_relative.drop_duplicates("year").set_index("year")["n"]
    _plot_stability_heatmap(
        year_relative, "year", [f"{year} {'dev' if year != 2026 else 'test'} (n={int(year_counts[year])})"
                                for year in [2024, 2025, 2026]],
        [2024, 2025, 2026], ordered_features, output, "shap_year_stability_relative", 55,
    )
    _plot_stability_heatmap(
        year_table, "year", [f"{year} {'dev' if year != 2026 else 'test'} (n={int(year_counts[year])})"
                             for year in [2024, 2025, 2026]],
        [2024, 2025, 2026], ordered_features, supplementary,
        "shap_year_absolute_stability", 55, value_col="mean_abs_shap",
    )
    region_counts = region_relative.drop_duplicates("region").set_index("region")["n"]
    _plot_stability_heatmap(
        region_relative, "region", [f"{REGION_ENGLISH[region]} (n={int(region_counts[region])})" for region in REGIONS],
        REGIONS, ordered_features, output, "shap_region_stability_relative", 80,
    )
    _plot_stability_heatmap(
        region_table, "region", [f"{REGION_ENGLISH[region]} (n={int(region_counts[region])})" for region in REGIONS],
        REGIONS, ordered_features, supplementary,
        "shap_region_absolute_stability", 80, value_col="mean_abs_shap",
    )
    _plot_rank_correlations(year_ranks, [2024, 2025, 2026],
                            ["2024", "2025", "2026"], output,
                            "shap_year_rank_correlation", (90, 80))
    _plot_rank_correlations(region_ranks, REGIONS,
                            [REGION_ENGLISH[region] for region in REGIONS], output,
                            "shap_region_rank_correlation", (110, 105))
    _plot_local(temporal, temporal_shap, temporal_base, probability, supplementary)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs" / "lwt_shap")
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"SHAP output already exists; choose a new --output: {output}")
    data, data_sha256 = load_dataset()
    assert_premium_positive_class(data)
    frozen, development, temporal, saved_predictions, temporal_manifest, predictions_path = _verified_inputs(
        data, data_sha256,
    )
    estimator = _frozen_estimator(frozen)
    X_development, y_development = development[FEATURES], development.label
    X_temporal = temporal[FEATURES]
    fitted = clone(estimator).fit(X_development, y_development)
    probability = fitted.predict_proba(X_temporal)[:, 1]
    if not np.allclose(probability, saved_predictions.probability_special_grade.to_numpy(dtype=float),
                       rtol=0, atol=1e-10):
        raise ValueError("Refitted frozen Extra Trees probabilities differ from saved 2026 validation")
    development_shap, _, dev_error = _positive_shap(fitted, X_development)
    temporal_shap, temporal_base, temporal_error = _positive_shap(fitted, X_temporal)
    oof_shap, fold_table, oof_error = _oof_shap(estimator, development, frozen["seed"])
    importance = _importance_table(oof_shap.reshape(-1, len(FEATURES)))
    year_table, region_table = _stability_tables(
        development, temporal, development_shap, temporal_shap,
    )
    output.mkdir(parents=True)
    _save_table(importance, output, "shap_global_importance.csv")
    _save_table(fold_table, output, "shap_fold_stability.csv")
    _save_table(year_table, output, "shap_year_stability.csv")
    _save_table(region_table, output, "shap_region_stability.csv")
    _save_table(_shap_table(development, oof_shap[0]),
                output, "shap_development_oof_values.csv")
    _save_table(_shap_table(temporal, temporal_shap, temporal_base, probability),
                output, "shap_temporal_values.csv")
    _render_panels(output, importance, fold_table, year_table, region_table,
                   development, temporal, oof_shap[0], temporal_shap,
                   temporal_base, probability)
    run_manifest = {
        "dataset_version": frozen["dataset_version"],
        "data_sha256": data_sha256,
        "frozen_config_sha256": _sha256(FREEZE_PATH),
        "temporal_manifest_sha256": _sha256(temporal_manifest),
        "temporal_predictions_sha256": _sha256(predictions_path),
        "model": PRIMARY,
        "model_params": frozen["models"][PRIMARY],
        "development_n": 186, "temporal_n": 201,
        "feature_set": FEATURES,
        "concentration_unit": CONCENTRATION_UNIT,
        "concentration_unit_source": CONCENTRATION_UNIT_SOURCE,
        "global_importance_source": "2024-2025 fixed-parameter repeated OOF SHAP (5x5)",
        "beeswarm_and_dependence_development_source": "First outer repeat OOF SHAP (n=186)",
        "year_stability_source": "Frozen final model; development in-sample and 2026 held-out",
        "region_stability_source": "Frozen final model; six core regions in 2026 only",
        "stability_panel_scale": "Within each year/region: mean absolute SHAP divided by the sum across ten features",
        "cohort_comparison": "Development fixed-parameter OOF mean absolute SHAP vs 2026 frozen-model mean absolute SHAP",
        "supplementary_candidates": ["global importance", "fold stability", "ECG dependence", "local explanation", "absolute stability"],
        "local_example": "First 2026 source Excel row; analysis ID is generated, not an original sample number",
        "max_additivity_error": {"development": dev_error, "temporal": temporal_error, "oof": oof_error},
        "no_tuning_feature_selection_or_model_selection": True,
        "shap_is_predictive_not_causal": True,
    }
    (output / "shap_run_manifest.json").write_text(
        json.dumps(run_manifest, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    print(f"Saved frozen-model SHAP tables and panels to {output}", flush=True)


if __name__ == "__main__":
    main()

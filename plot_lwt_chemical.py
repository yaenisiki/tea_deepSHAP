"""Chemical phenotype panels; no classifier fitting or model selection.

PCA is fitted on 2024-2025 only and 2026 is projected for display. Adjusted
grade effects and Spearman correlations use 2024-2025 development samples.
The singleton non-core region remains in PCA/descriptive rows but is excluded
from region-adjusted regression because it cannot support HC3 inference.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Ellipse
from scipy.stats import chi2
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.multitest import multipletests

from lwt_data import CONCENTRATION_UNIT_SOURCE, DATASET_VERSION, CONCENTRATION_UNIT, FEATURES, REGIONS, ROOT, load_dataset
from manuscript_plotting import apply_manuscript_style, panel_figure, save_panel


INK = "#263C49"
PREMIUM = "#087F8C"
NON_PREMIUM = "#A77756"
MUTED = "#AAB8BF"
MARKERS = {2024: "o", 2025: "s", 2026: "^"}
GRADES = {1: "Premium grade", 0: "Non-premium grade"}
REGION_STEMS = {
    "勉县": "mianxian", "南郑": "nanzheng", "城固": "chenggu",
    "宁强": "ningqiang", "西乡": "xixiang", "镇巴": "zhenba",
    "略阳": "lueyang",
}
DERIVED = (
    "total_catechins", "esterified_catechins",
    "non_esterified_catechins", "esterified_non_esterified_ratio",
)
CORR_MAP = LinearSegmentedColormap.from_list(
    "chemical_correlation", ["#567A9B", "#F5F5F2", "#B8775B"],
)


def _save(fig, output: Path, stem: str) -> None:
    save_panel(fig, output, stem, tiff=True)


def _plot_feature_distributions(data: pd.DataFrame, output: Path, stem: str) -> None:
    """Ten independent concentration axes; grade colours are consistent throughout."""
    fig, axes = plt.subplots(2, 5, figsize=(190 / 25.4, 130 / 25.4), layout="constrained")
    fig._manuscript_fixed_canvas = True
    rng = np.random.default_rng(42)
    for ax, feature in zip(axes.flat, FEATURES):
        for position, label, color in ((0, 1, PREMIUM), (1, 0, NON_PREMIUM)):
            values = data.loc[data.label.eq(label), feature].to_numpy(dtype=float)
            if not len(values):
                continue
            if len(values) >= 2 and np.ptp(values) > 0:
                violin = ax.violinplot(values, positions=[position], widths=0.62,
                                       showmeans=False, showmedians=False, showextrema=False)
                for body in violin["bodies"]:
                    body.set_facecolor(color)
                    body.set_edgecolor(color)
                    body.set_alpha(0.22)
            ax.boxplot(values, positions=[position], widths=0.14, patch_artist=True,
                       showfliers=False, manage_ticks=False,
                       boxprops={"facecolor": "white", "edgecolor": color, "linewidth": 0.9},
                       medianprops={"color": INK, "linewidth": 1.0},
                       whiskerprops={"color": color, "linewidth": 0.8},
                       capprops={"color": color, "linewidth": 0.8})
            ax.scatter(position + rng.uniform(-0.12, 0.12, len(values)), values,
                       s=4, color=color, alpha=0.38, linewidths=0)
        ax.set_xlim(-0.5, 1.5)
        ax.set_xticks([0, 1], ["", ""])
        ax.set_ylabel(f"{feature} ({CONCENTRATION_UNIT})")
        ax.tick_params(length=2.5, color=INK)
    fig.supxlabel("Premium grade (left)     Non-premium grade (right)", fontsize=9)
    _save(fig, output, stem)


def _derived_chemistry(data: pd.DataFrame, output: Path) -> dict[str, float | int]:
    """Descriptive-only composites; never appended to the ML feature list."""
    result = data[["analysis_id", "source_excel_row", "年份", "产地", "品级", "label"]].copy()
    result = result.rename(columns={"年份": "year", "产地": "region", "品级": "grade_original"})
    result["analyte_and_sum_unit"] = CONCENTRATION_UNIT
    result["ratio_unit"] = "dimensionless"
    result["total_catechins"] = data[["GC", "EGC", "C", "EC", "EGCG", "GCG", "ECG", "CG"]].sum(axis=1)
    result["esterified_catechins"] = data[["EGCG", "GCG", "ECG", "CG"]].sum(axis=1)
    result["non_esterified_catechins"] = data[["GC", "EGC", "C", "EC"]].sum(axis=1)
    denominator = result["non_esterified_catechins"].to_numpy(dtype=float)
    numerator = result["esterified_catechins"].to_numpy(dtype=float)
    positive = denominator[np.isfinite(denominator) & (denominator > 0)]
    if not len(positive):
        raise ValueError("No positive non-esterified catechin denominator")
    # Numerical guard only; this is not an analytical LOD/LOQ threshold.
    threshold = max(float(np.sqrt(np.finfo(float).eps) * np.median(positive)),
                    float(np.finfo(float).tiny))
    valid = np.isfinite(numerator) & np.isfinite(denominator) & (denominator > threshold)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        ratio = np.divide(numerator, denominator, out=np.full(len(data), np.nan), where=valid)
    nonfinite = valid & ~np.isfinite(ratio)
    ratio[nonfinite] = np.nan
    result["esterified_non_esterified_ratio"] = ratio
    result.to_csv(output / "chemical_derived_values.csv", index=False, encoding="utf-8-sig")
    structure = (result.groupby(["year", "region", "grade_original"], dropna=False)
                 .size().rename("n").reset_index())
    structure.to_csv(output / "chemical_sample_structure.csv", index=False, encoding="utf-8-sig")
    strata = [("overall", result)]
    for column in ("grade_original", "year", "region"):
        strata.extend((f"{column}={value}", part) for value, part in result.groupby(column, dropna=False))
    for columns in (("grade_original", "year"), ("grade_original", "region")):
        strata.extend(
            (";".join(f"{column}={value}" for column, value in zip(columns, values)), part)
            for values, part in result.groupby(list(columns), dropna=False)
        )
    rows = []
    for stratum, part in strata:
        for feature in DERIVED:
            values = part[feature].dropna().to_numpy(dtype=float)
            rows.append({
                "stratum": stratum, "feature": feature, "n_samples": len(part),
                "unit": "dimensionless" if feature == "esterified_non_esterified_ratio"
                        else CONCENTRATION_UNIT,
                "n_valid": len(values),
                "mean": float(np.mean(values)) if len(values) else np.nan,
                "sd": float(np.std(values, ddof=1)) if len(values) > 1 else np.nan,
                "median": float(np.median(values)) if len(values) else np.nan,
                "q1": float(np.quantile(values, 0.25)) if len(values) else np.nan,
                "q3": float(np.quantile(values, 0.75)) if len(values) else np.nan,
            })
    pd.DataFrame(rows).to_csv(output / "chemical_derived_descriptives.csv",
                              index=False, encoding="utf-8-sig")
    return {
        "ratio_numerical_denominator_threshold": threshold,
        "ratio_unsafe_denominator_n": int((~np.isfinite(denominator) | (denominator <= threshold)).sum()),
        "ratio_nonfinite_result_n": int(nonfinite.sum()),
    }


def _ellipse(ax, xy: np.ndarray, color: str) -> None:
    """95% data ellipse, not a confidence interval for a group mean."""
    if len(xy) < 3:
        return
    eigenvalues, eigenvectors = np.linalg.eigh(np.cov(xy, rowvar=False))
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues, eigenvectors = eigenvalues[order], eigenvectors[:, order]
    if eigenvalues[-1] <= 0:
        return
    angle = np.degrees(np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0]))
    radius = np.sqrt(chi2.ppf(0.95, df=2))
    width, height = 2 * radius * np.sqrt(eigenvalues)
    ellipse = Ellipse(xy.mean(axis=0), width, height, angle=angle,
                      facecolor=color, edgecolor=color, alpha=0.09, linewidth=0.9)
    ax.add_patch(ellipse)


def _plot_pca(data: pd.DataFrame, development: pd.DataFrame,
              output: Path, ellipses: bool) -> None:
    scaler = StandardScaler().fit(development[FEATURES])
    pca = PCA(n_components=2).fit(scaler.transform(development[FEATURES]))
    scores = pca.transform(scaler.transform(data[FEATURES]))
    table = data[["analysis_id", "source_excel_row", "年份", "产地", "品级", "label"]].copy()
    table = table.rename(columns={"年份": "year", "产地": "region", "品级": "grade_original"})
    table["PC1"], table["PC2"] = scores[:, 0], scores[:, 1]
    table.to_csv(output / "pca_scores.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame({"component": ["PC1", "PC2"],
                  "explained_variance_ratio": pca.explained_variance_ratio_}).to_csv(
        output / "pca_variance.csv", index=False, encoding="utf-8-sig",
    )
    fig, ax = panel_figure(140, 100)
    for year, marker in MARKERS.items():
        for label, color in ((0, NON_PREMIUM), (1, PREMIUM)):
            subset = (data["年份"].to_numpy() == year) & (data.label.to_numpy() == label)
            ax.scatter(scores[subset, 0], scores[subset, 1], s=20, marker=marker,
                       color=color, edgecolors="white", linewidths=0.25,
                       alpha=0.68 if year == 2026 else 0.8)
    if ellipses:
        development_mask = data["年份"].isin([2024, 2025]).to_numpy()
        for label, color in ((0, NON_PREMIUM), (1, PREMIUM)):
            _ellipse(ax, scores[development_mask & data.label.eq(label).to_numpy()], color)
    ax.axhline(0, color="#D6E0E3", linewidth=0.6)
    ax.axvline(0, color="#D6E0E3", linewidth=0.6)
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%} explained variance)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%} explained variance)")
    grade_handles = [Line2D([0], [0], marker="o", linestyle="none", color=color,
                            label=GRADES[label], markersize=5)
                     for label, color in ((0, NON_PREMIUM), (1, PREMIUM))]
    year_handles = [Line2D([0], [0], marker=marker, linestyle="none", color=INK,
                           label=str(year), markersize=5)
                    for year, marker in MARKERS.items()]
    grade_legend = ax.legend(handles=grade_handles, loc="upper left")
    ax.add_artist(grade_legend)
    ax.legend(handles=year_handles, loc="upper right", ncol=3)
    _save(fig, output, "pca_by_grade_year")


def _grade_effects(development: pd.DataFrame, output: Path) -> None:
    inferential = development.loc[development["产地"].isin(REGIONS)].copy()
    excluded = len(development) - len(inferential)
    if len(inferential) < 50 or inferential["产地"].nunique() != 6:
        raise ValueError("Expected six sufficiently represented core regions")
    records = []
    for feature in FEATURES:
        values = inferential[feature].to_numpy(dtype=float)
        sd = float(values.std(ddof=0))
        if not np.isfinite(sd) or sd <= 0:
            raise ValueError(f"Cannot standardize constant analyte: {feature}")
        frame = pd.DataFrame({
            "z": (values - values.mean()) / sd,
            "premium": inferential.label.to_numpy(dtype=int),
            "year": inferential["年份"].astype(str).to_numpy(),
            "region": inferential["产地"].to_numpy(),
        })
        fit = smf.ols("z ~ premium + C(year) + C(region)", data=frame).fit(cov_type="HC3")
        if np.linalg.matrix_rank(fit.model.exog) != fit.model.exog.shape[1]:
            raise ValueError("Grade, year and region design matrix is rank deficient")
        ci = fit.conf_int().loc["premium"]
        records.append({
            "feature": feature, "standardized_effect": float(fit.params["premium"]),
            "CI_lower": float(ci.iloc[0]), "CI_upper": float(ci.iloc[1]),
            "raw_p": float(fit.pvalues["premium"]), "standardization_sd": sd,
            "input_concentration_unit": CONCENTRATION_UNIT,
            "n_included": len(inferential), "n_excluded_noncore": excluded,
            "cohort": "2024-2025 development, six core regions",
            "model": "z(analyte) ~ premium grade + year + region; OLS HC3",
        })
    effects = pd.DataFrame(records)
    if not np.isfinite(effects[["standardized_effect", "CI_lower", "CI_upper", "raw_p"]]).all().all():
        raise ValueError("Non-finite adjusted grade effect or CI")
    effects["FDR_q"] = multipletests(effects.raw_p, method="fdr_bh")[1]
    effects.to_csv(output / "chemical_grade_effects.csv", index=False, encoding="utf-8-sig")
    fig, ax = panel_figure(140, 88)
    ypos = np.arange(len(FEATURES))
    center = effects.standardized_effect.to_numpy()
    ax.hlines(ypos, effects.CI_lower, effects.CI_upper,
              color=MUTED, linewidth=1.5)
    ax.scatter(center, ypos, s=27, color=PREMIUM, zorder=3)
    ax.axvline(0, color=INK, linestyle="--", linewidth=0.8)
    ax.set_yticks(ypos, FEATURES)
    ax.invert_yaxis()
    ax.set_xlabel("Adjusted premium-grade effect (analyte SD units)")
    ax.tick_params(length=2.5)
    _save(fig, output, "chemical_grade_effect_forest")


def _plot_correlation(development: pd.DataFrame, output: Path) -> None:
    corr = development[FEATURES].corr(method="spearman")
    if not np.isfinite(corr.to_numpy()).all():
        raise ValueError("Spearman matrix contains non-finite correlations")
    corr.to_csv(output / "chemical_spearman_correlation.csv", encoding="utf-8-sig")
    fig, ax = panel_figure(140, 112)
    lower_triangle = np.ma.masked_where(np.triu(np.ones((10, 10), dtype=bool), k=1),
                                        corr.to_numpy())
    mesh = ax.pcolormesh(np.arange(11), np.arange(11), lower_triangle,
                         cmap=CORR_MAP, vmin=-1, vmax=1,
                         edgecolors="white", linewidth=0.7)
    ax.set_xlim(0, 10)
    ax.set_ylim(10, 0)
    ax.set_xticks(np.arange(10) + 0.5, FEATURES)
    ax.set_yticks(np.arange(10) + 0.5, FEATURES)
    ax.tick_params(length=0)
    for i in range(10):
        for j in range(10):
            if j > i:
                continue
            value = corr.iloc[i, j]
            ax.text(j + 0.5, i + 0.5, f"{value:.2f}",
                    ha="center", va="center", fontsize=8,
                    color="white" if abs(value) > 0.72 else INK)
    colorbar = fig.colorbar(mesh, ax=ax, fraction=0.035, pad=0.025)
    colorbar.set_label("Spearman rho", fontsize=8)
    colorbar.ax.tick_params(labelsize=8)
    _save(fig, output, "chemical_correlation")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path,
                        default=ROOT / "outputs" / "lwt_figures" / "chemical")
    parser.add_argument("--ellipses", action="store_true",
                        help="Add pale 95% development data ellipses to the PCA plot")
    parser.add_argument("--supplementary-by", nargs="*", choices=("year", "region"),
                        default=[], help="Also export separate ten-analyte distributions by stratum")
    args = parser.parse_args()
    data, digest = load_dataset()
    development = data.loc[data["年份"].isin([2024, 2025])].copy()
    if len(data) != 387 or len(development) != 186:
        raise ValueError("Locked cohort sizes differ from the LWT analysis dataset")
    args.output.mkdir(parents=True, exist_ok=True)
    apply_manuscript_style()
    _plot_feature_distributions(data, args.output, "chemical_feature_distributions")
    ratio_guard = _derived_chemistry(data, args.output)
    _plot_pca(data, development, args.output, args.ellipses)
    _grade_effects(development, args.output)
    _plot_correlation(development, args.output)
    if "year" in args.supplementary_by:
        for year, part in data.groupby("年份"):
            _plot_feature_distributions(part, args.output / "supplementary",
                                        f"chemical_feature_distributions_year_{year}")
    if "region" in args.supplementary_by:
        for region, part in data.groupby("产地"):
            if region not in REGION_STEMS:
                raise ValueError(f"Unknown region filename stem: {region}")
            _plot_feature_distributions(part, args.output / "supplementary",
                                        f"chemical_feature_distributions_region_{REGION_STEMS[region]}")
    (args.output / "chemical_plot_manifest.json").write_text(
        json.dumps({
            "dataset_version": DATASET_VERSION, "data_sha256": digest,
            "concentration_unit": CONCENTRATION_UNIT,
            "concentration_unit_source": CONCENTRATION_UNIT_SOURCE,
            "pca_fit_cohort": "2024-2025 development; 2026 projected only",
            "grade_effect_cohort": "2024-2025 core regions; non-core rows excluded from HC3 regression",
            "grade_effect_noncore_excluded_n": int((~development["产地"].isin(REGIONS)).sum()),
            "spearman_cohort": "2024-2025 development",
            "distribution_cohort": "all 387 supplied rows; synthetic rows are artificial",
            "derived_indicators": list(DERIVED),
            "derived_indicators_used_for_ml": False,
            **ratio_guard,
            "supplementary_strata": args.supplementary_by,
            "ellipse": "95% data ellipse, not a confidence interval" if args.ellipses else "none",
        }, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    print(f"Saved chemical phenotype panels to {args.output}")


if __name__ == "__main__":
    main()

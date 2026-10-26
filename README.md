# Tea grade classification: analysis code

Public analysis code for a green-tea chemical composition and grade classification study. The workflow includes 16 classifiers, repeated nested cross-validation, leave-one-region-out (LORO) evaluation, a development-only parameter freeze, temporal validation, chemical statistics, and SHAP explanations.

**Original observations and trained model weights are not included.** The example CSV is generated entirely from an invented random process. It is for checking the software and must not be used as scientific evidence or as a substitute for the study data. The public code retains the study's cohort and region checks.

中文说明：[使用说明](docs/使用说明.md)。Disclosure templates: [data and code availability](docs/DATA_AND_CODE_AVAILABILITY.md).

## Contents

| Files | Purpose |
| --- | --- |
| `lwt_data.py` | Explicit CSV/XLSX input, schema checks, optional input digest |
| `lwt_models.py`, `lwt_validation.py`, `run_lwt.py` | 16 pipelines, tuning spaces, nested CV and LORO |
| `freeze_lwt.py`, `run_lwt_temporal.py` | Freeze development-derived parameters and evaluate the held-out year |
| `run_lwt_shap.py` | Fixed-parameter OOF SHAP, temporal SHAP and additivity checks |
| `plot_lwt_*.py`, `run_lwt_loro_primary_ci.py` | Chemical panels and result panels, genuine bootstrap AUC intervals |
| `manuscript_plotting.py`, `render_lwt_figures.py`, `compose_panels.py` | Shared export style, local panel gallery and vector PDF composition |
| `examples/` | Independently generated synthetic CSV and its generator |
| `tests/`, `verify_release.py` | Public adapter tests and upload inventory checks |

Historical analyses, third-party reference code, final manuscript layouts that depend on local tools, all original outputs, sample-level predictions/SHAP, source audit records, local dependency folders, and the original frozen configuration are excluded. Model definitions, tuning grids, and analysis logic are public; fitted parameter arrays and serialized models are excluded.

## Environment and quick check

Use Python 3.13 and a separate environment. `requirements.txt` records the versions used for local release checks; a fresh dependency installation on every platform has not been tested.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python verify_release.py
```

The following PowerShell commands run only on artificial data:

```powershell
$env:TEA_DATA_PATH = (Resolve-Path "examples/synthetic_tea.csv").Path
$env:TEA_DATASET_VERSION = "synthetic-example-v1 (not study observations)"
python run_lwt.py --mode fast --only "Extra Trees" --outer-limit 1 --skip-loro --output outputs/demo_smoke
```

In Bash, set the corresponding environment variables with:

```bash
export TEA_DATA_PATH="$PWD/examples/synthetic_tea.csv"
export TEA_DATASET_VERSION="synthetic-example-v1 (not study observations)"
```

The smoke run uses one outer fold and one model. It is a software check, not a complete benchmark. Generated results are local and are ignored by Git. Use a new output path for each run; analysis commands refuse to overwrite completed runs where specified.

## Full workflow on a supplied dataset

Set `TEA_DATA_PATH` before running any script. Without it, the loader looks for `private_data/study_data.xlsx`, which is intentionally absent. A UTF-8 CSV or an XLSX `Sheet1` must contain these columns:

```text
GA,GC,EGC,C,CAF,EC,EGCG,GCG,ECG,CG,年份,产地,品级
```

The fixed study design requires 63 rows for 2024, 123 for 2025 and 201 for 2026. The first two years form the development cohort; 2026 is held out. The six core regions are 勉县, 南郑, 城固, 宁强, 西乡 and 镇巴. The original LORO CI recovery also expects one non-core 略阳 development row. Grades are 特级 (positive class, 1) and 非特级 (0). Ten features must be finite numeric values. Extra input columns, including original sample identifiers, are dropped by the public adapter. Derived row identifiers are still sample-level information and must stay private for real data.

Optional variables: `TEA_SHEET` (XLSX sheet name), `TEA_EXPECTED_SHA256` (an owner-supplied digest), `TEA_DATASET_VERSION`, and `TEA_CONCENTRATION_UNIT` (default `mg/L`; synthetic values are arbitrary).

After setting the input, execute in order:

```powershell
python run_lwt.py --mode paper
python freeze_lwt.py
python run_lwt_temporal.py
python run_lwt_shap.py --output outputs/lwt_shap_retry
python render_lwt_figures.py --no-open
```

Full mode requires all 16 model dependencies. It uses 25 outer folds per model and performs region holdouts; it is substantially more expensive than the smoke check. The primary algorithm is prespecified as Extra Trees in this release. Its tuning parameters are selected only from development inner-CV results; the fixed threshold is 0.5. The new freeze is generated from the supplied input and is not the study's confidential frozen artifact. Do not use temporal results to revise model selection or parameters.

The gallery creates chemical, benchmark, temporal, LORO and SHAP panels. Chemical distributions and local SHAP can expose individual observations; **do not upload generated panels or tables from private data without reviewing them**. `compose_panels.py` accepts a user-supplied layout JSON; the study's private final layouts and original figures are not distributed.

## Optional provenance exclusion analysis

The private study audit is not distributed. A dataset owner may supply a UTF-8 CSV through `TEA_PROVENANCE_AUDIT` before freezing, with columns `source_excel_row`, `analysis_id`, `year`, and `provenance_status`. Each listed row must match the supplied 2026 input and use status `source_explanation_pending`. Row numbers include the header, so the first data row is 2 in either input format.

If no audit is supplied, the manifest records its absence and the provenance-exclusion sensitivity analysis/panel is skipped. This is not a claim that provenance has been verified, and it does not replace the manuscript's obligation to describe actual study limitations. No synthetic audit or private exclusion list is invented for this release.

## Upload boundary

Upload only this directory's approved files, preferably from the clean release ZIP. The `.gitignore` uses an explicit allowlist. Keep real inputs in `private_data/` or outside the repository; outputs and generated freezes remain local. Existing Git-tracked files and `git add -f` can bypass ignore rules, so run `python verify_release.py` after staging and inspect `git status --short` before pushing.

`RELEASE_MANIFEST.json` records SHA256 for the packaged files. The checker verifies those bytes, checks common credential patterns, and rejects prospective Git files outside the inventory. It is a release snapshot check, not a guarantee against every possible secret. After intentionally changing public code, review and regenerate the inventory before a new release. No original repository history is included.

"""Public input adapter. Private study observations are not distributed."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
DATA_PATH = Path(os.environ.get("TEA_DATA_PATH", str(ROOT / "private_data" / "study_data.xlsx"))).expanduser().resolve()
DATASET_VERSION = os.environ.get("TEA_DATASET_VERSION", "user-supplied dataset (not distributed)")
CONCENTRATION_UNIT = os.environ.get("TEA_CONCENTRATION_UNIT", "mg/L")
CONCENTRATION_UNIT_SOURCE = "Supplied by the dataset owner; synthetic example values are arbitrary"
FEATURES = ["GA", "GC", "EGC", "C", "CAF", "EC", "EGCG", "GCG", "ECG", "CG"]
REGIONS = ["勉县", "南郑", "城固", "宁强", "西乡", "镇巴"]


def load_dataset() -> tuple[pd.DataFrame, str]:
    if not DATA_PATH.is_file():
        raise FileNotFoundError(
            "Private study data are not included. Set TEA_DATA_PATH to your authorized "
            "CSV/XLSX input, or to examples/synthetic_tea.csv for the artificial demo."
        )
    digest = hashlib.sha256(DATA_PATH.read_bytes()).hexdigest()
    expected = os.environ.get("TEA_EXPECTED_SHA256")
    if expected and digest.lower() != expected.lower():
        raise ValueError("Input SHA256 differs from TEA_EXPECTED_SHA256")
    if DATA_PATH.suffix.lower() == ".csv":
        frame = pd.read_csv(DATA_PATH, encoding="utf-8-sig")
    elif DATA_PATH.suffix.lower() == ".xlsx":
        frame = pd.read_excel(DATA_PATH, sheet_name=os.environ.get("TEA_SHEET", "Sheet1"), engine="openpyxl")
    else:
        raise ValueError("Input must be a UTF-8 CSV or an XLSX workbook")
    required = FEATURES + ["年份", "产地", "品级"]
    if frame.columns.duplicated().any() or any(column not in frame for column in required):
        raise ValueError(f"Required columns: {required}")
    # Preserve the published study's cohort checks instead of silently changing its design.
    if len(frame) != 387 or frame["年份"].value_counts().to_dict() != {2026: 201, 2025: 123, 2024: 63}:
        raise ValueError("Study design requires 63/123/201 rows for 2024/2025/2026 (387 total)")
    if frame["品级"].isna().any() or set(frame["品级"]) != {"特级", "非特级"}:
        raise ValueError("Grade labels must be 特级 and 非特级, with no missing labels")
    if frame["产地"].isna().any() or not set(frame["产地"]).issubset(set(REGIONS) | {"略阳"}):
        raise ValueError("Missing or unexpected study region")
    numeric = frame[FEATURES].apply(pd.to_numeric, errors="coerce")
    if not np.isfinite(numeric.to_numpy(dtype=float)).all():
        raise ValueError("Missing, nonnumeric or infinite analyte value")
    # Discard names, original IDs and any other extra input columns at the boundary.
    frame = frame[required].copy()
    frame[FEATURES] = numeric
    frame["source_excel_row"] = np.arange(2, len(frame) + 2)
    frame["analysis_id"] = [f"Y{year}_ROW{row:05d}" for year, row in zip(frame["年份"], frame["source_excel_row"])]
    frame["label"] = frame["品级"].map({"非特级": 0, "特级": 1}).astype(int)
    return frame, digest

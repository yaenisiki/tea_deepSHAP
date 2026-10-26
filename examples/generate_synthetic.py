"""Generate artificial demo rows without reading or fitting any study data."""

from pathlib import Path
import numpy as np
import pandas as pd

FEATURES = ["GA", "GC", "EGC", "C", "CAF", "EC", "EGCG", "GCG", "ECG", "CG"]
REGIONS = ["勉县", "南郑", "城固", "宁强", "西乡", "镇巴"]


def generate() -> pd.DataFrame:
    rng = np.random.default_rng(20261026)
    parts = []
    for year, count in [(2024, 63), (2025, 123), (2026, 201)]:
        labels = rng.permutation(np.arange(count) % 2)
        regions = np.array([REGIONS[index % len(REGIONS)] for index in range(count)], dtype=object)
        if year == 2024:
            regions[-1] = "略阳"  # One artificial non-core development row.
        # Entirely invented distribution and effects; no empirical means or model weights.
        values = rng.lognormal(mean=2.0, sigma=0.45, size=(count, len(FEATURES)))
        values[:, 5] *= 1 + 0.55 * labels
        values[:, 6] *= 1 + 0.40 * labels
        part = pd.DataFrame(np.round(values, 6), columns=FEATURES)
        part["年份"] = year
        part["产地"] = regions
        part["品级"] = np.where(labels == 1, "特级", "非特级")
        parts.append(part)
    return pd.concat(parts, ignore_index=True)


if __name__ == "__main__":
    destination = Path(__file__).resolve().with_name("synthetic_tea.csv")
    if destination.exists():
        raise FileExistsError(f"Refusing to replace an existing demo: {destination.name}")
    generate().to_csv(destination, index=False, encoding="utf-8", lineterminator="\n")
    print(f"Created {destination.name}: 387 artificial rows; not research evidence.")

"""
qc.py - Quality control checks for segmentation results
"""

import numpy as np
from skimage.measure import regionprops
import pandas as pd


TARGET_GRAPES = 36

_RATIO_LABELS = ["100:0", "80:20", "60:40", "40:60", "20:80", "0:100"]


def run_qc_checks(
        labeled_mask: np.ndarray,
        df_measurements: pd.DataFrame,
        wb_factors: dict = None,
        grabcut_changes: list = None,
        cell_grape_counts: list = None,
        grid_method: str = None,
) -> dict:
    """
    Run all QC checks on the segmentation result.
    Returns QC report dict.
    """
    props = regionprops(labeled_mask)
    n = len(props)
    areas = [p.area for p in props]
    solidities = [p.solidity for p in props]

    issues = []

    # 1. Count check
    count_ok = n == TARGET_GRAPES
    if not count_ok:
        issues.append(f"Expected {TARGET_GRAPES} grapes, detected {n}")

    # 2. Area outliers (flag grapes with area 3× median or <1/3 median)
    if areas:
        median_area = np.median(areas)
        for i, a in enumerate(areas):
            if a > 3 * median_area:
                issues.append(f"Grape {i+1}: area too large ({a:.0f} px vs median {median_area:.0f})")
            elif a < median_area / 3:
                issues.append(f"Grape {i+1}: area too small ({a:.0f} px vs median {median_area:.0f})")

    # 3. Solidity check (jagged masks = likely bad segmentation)
    for i, s in enumerate(solidities):
        if s < 0.6:
            issues.append(f"Grape {i+1}: low solidity ({s:.2f}) - possible bad mask")

    # 4. Color sanity (grapes should be yellowish-green)
    if not df_measurements.empty and "Mean_H" in df_measurements.columns:
        for _, row in df_measurements.iterrows():
            h = row.get("Mean_H")
            if h is None or not isinstance(h, (int, float)):
                continue
            if h < 10 or h > 120:
                issues.append(f"Grape {row['Grape_ID']}: unusual hue ({h:.1f}) - check mask")

    # 5. Per-cell grape count (should be exactly 3 per cell)
    if cell_grape_counts:
        for i, cnt in enumerate(cell_grape_counts):
            if cnt != 3:
                col_label = "Coated" if i % 2 == 0 else "Control"
                ratio = _RATIO_LABELS[i // 2] if i // 2 < 6 else f"row{i // 2}"
                issues.append(
                    f"Cell {i + 1} ({ratio} {col_label}): {cnt} grapes found (expected 3)"
                )

    # 6. GrabCut large area changes (> 30% shift may indicate bad rough mask)
    if grabcut_changes:
        for ch in grabcut_changes:
            if abs(ch["delta_pct"]) > 30:
                issues.append(
                    f"Grape {ch['grape_id']}: GrabCut changed area by "
                    f"{ch['delta_pct']:+.0f}% — check mask"
                )

    # Overall status
    if n == TARGET_GRAPES and not issues:
        status = "PASS"
        color = "green"
    elif abs(n - TARGET_GRAPES) <= 2 and len(issues) <= 2:
        status = "WARNING"
        color = "orange"
    else:
        status = "FAIL"
        color = "red"

    mean_solidity = float(np.mean(solidities)) if solidities else 0.0

    return {
        "status": status,
        "color": color,
        "n_detected": n,
        "n_expected": TARGET_GRAPES,
        "issues": issues,
        "area_stats": {
            "min": min(areas) if areas else 0,
            "max": max(areas) if areas else 0,
            "mean": np.mean(areas) if areas else 0,
            "median": np.median(areas) if areas else 0,
        },
        "wb_factors": wb_factors or {},
        "grid_method": grid_method or "Unknown",
        "cell_grape_counts": cell_grape_counts or [],
        "mean_solidity": round(mean_solidity, 3),
    }

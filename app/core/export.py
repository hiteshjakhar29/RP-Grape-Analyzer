"""
export.py - Export results to CSV and Excel
"""

import pandas as pd
import cv2
import numpy as np
from pathlib import Path
from datetime import datetime
import os


def export_results(df_measurements: pd.DataFrame,
                   df_comparison: pd.DataFrame,
                   debug_images: dict,
                   output_dir: str,
                   image_name: str) -> dict:
    """
    Export all results:
    - CSV (per grape measurements)
    - Excel (per grape + comparison sheet)
    - Step images (PNG files)
    Returns dict of output file paths.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_name = Path(image_name).stem

    output_paths = {}

    # ── CSV ──
    csv_path = output_dir / f"{base_name}_measurements_{timestamp}.csv"
    df_measurements.to_csv(csv_path, index=False)
    output_paths["csv"] = str(csv_path)

    # ── Excel (multi-sheet) ──
    excel_path = output_dir / f"{base_name}_measurements_{timestamp}.xlsx"
    with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
        df_measurements.to_excel(writer, sheet_name="Per_Grape_Measurements", index=False)

        if not df_comparison.empty:
            df_comparison.to_excel(writer, sheet_name="ImageJ_Comparison", index=False)

        # Summary sheet (group averages)
        if not df_measurements.empty:
            summary = df_measurements.groupby(["Group", "Ratio"]).agg({
                "Area_px2": ["mean", "std"],
                "Mean_R": "mean", "Mean_G": "mean", "Mean_B": "mean",
                "Mean_L": "mean", "Mean_a": "mean", "Mean_b": "mean",
                "Mean_H": "mean", "Mean_S": "mean", "Mean_Brightness": "mean",
            }).round(3)
            summary.to_excel(writer, sheet_name="Group_Summary")

    output_paths["excel"] = str(excel_path)

    # ── Step images ──
    image_keys = ["original", "normalized", "zone_grid", "grape_masks", "final_overlay"]

    for key in image_keys:
        if key in debug_images and debug_images[key] is not None:
            img = debug_images[key]
            if len(img.shape) == 3 and img.shape[2] == 3:
                img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            else:
                img_bgr = img
            img_path = output_dir / f"{base_name}_{key}_{timestamp}.png"
            cv2.imwrite(str(img_path), img_bgr)
            output_paths[key] = str(img_path)

    print(f"[Export] Results saved to: {output_dir}")
    return output_paths

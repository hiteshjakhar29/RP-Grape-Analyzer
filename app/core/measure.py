"""
measure.py - Color measurements per grape
Replicates ImageJ macro measurements: RGB, Lab, HSB + Area
Matches the output format of Whole_Color_Measurements.csv
"""

import numpy as np
import cv2
from skimage import measure, color as skcolor
from skimage.measure import regionprops
import pandas as pd
from pathlib import Path


# Grape position labels (fixed layout - 6 rows × 2 groups × 3 grapes)
RATIO_LABELS = ["100:0", "80:20", "60:40", "40:60", "20:80", "0:100"]
GROUP_LABELS = ["Coated", "Control"]


def measure_all_grapes(image_rgb: np.ndarray, labeled_mask: np.ndarray,
                        image_path: str, px_per_mm: float = None) -> pd.DataFrame:
    """
    Measure Area, RGB, Lab, HSB for each labeled grape region.
    Returns DataFrame with 1 row per grape (36 rows).
    
    Matches ImageJ macro measurement logic exactly:
    - Area: pixel count (and mm² if scale provided)
    - RGB: mean of R, G, B channels inside mask
    - Lab: using skimage.color.rgb2lab (same formula as ImageJ Lab Stack)
    - HSB: using cv2 HSV (same as ImageJ HSB Stack)
    """
    # Convert color spaces once (for whole image, then sample per grape)
    image_lab = skcolor.rgb2lab(image_rgb)
    image_hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)
    image_name = Path(image_path).name

    rows = []
    props = regionprops(labeled_mask)

    # Sort by position (top-to-bottom, left-to-right) to assign consistent IDs
    props_sorted = sorted(props, key=lambda p: (p.centroid[0], p.centroid[1]))

    for idx, prop in enumerate(props_sorted):
        mask = (labeled_mask == prop.label)

        # ── Area ──
        area_px = int(mask.sum())
        area_mm2 = round(area_px / (px_per_mm ** 2), 3) if px_per_mm else None

        # ── RGB measurements (matches ImageJ Make Composite + multimeasure) ──
        R_pixels = image_rgb[:, :, 0][mask].astype(float)
        G_pixels = image_rgb[:, :, 1][mask].astype(float)
        B_pixels = image_rgb[:, :, 2][mask].astype(float)

        mean_R = round(float(R_pixels.mean()), 3)
        mean_G = round(float(G_pixels.mean()), 3)
        mean_B = round(float(B_pixels.mean()), 3)
        std_R  = round(float(R_pixels.std()), 3)
        std_G  = round(float(G_pixels.std()), 3)
        std_B  = round(float(B_pixels.std()), 3)

        # ── Lab measurements (matches ImageJ Lab Stack) ──
        L_pixels = image_lab[:, :, 0][mask]
        a_pixels = image_lab[:, :, 1][mask]
        b_pixels = image_lab[:, :, 2][mask]

        mean_L  = round(float(L_pixels.mean()), 3)
        mean_a  = round(float(a_pixels.mean()), 3)
        mean_b  = round(float(b_pixels.mean()), 3)
        std_L   = round(float(L_pixels.std()), 3)
        std_a   = round(float(a_pixels.std()), 3)
        std_b_s = round(float(b_pixels.std()), 3)

        # ── HSB measurements (matches ImageJ HSB Stack) ──
        # ImageJ HSB Stack: H in [0,255] scaled from [0,360]
        # OpenCV HSV: H in [0,180], S in [0,255], V in [0,255]
        H_pixels = image_hsv[:, :, 0][mask].astype(float)
        S_pixels = image_hsv[:, :, 1][mask].astype(float)
        V_pixels = image_hsv[:, :, 2][mask].astype(float)

        # Scale H to ImageJ range [0, 255] (ImageJ uses 0-255 for Hue)
        H_scaled = H_pixels * (255.0 / 180.0)

        mean_H = round(float(H_scaled.mean()), 3)
        mean_S = round(float(S_pixels.mean()), 3)
        mean_B_val = round(float(V_pixels.mean()), 3)  # Brightness = Value
        std_H  = round(float(H_scaled.std()), 3)
        std_S  = round(float(S_pixels.std()), 3)
        std_B_val = round(float(V_pixels.std()), 3)

        # ── Assign grape position (row, group, ratio) ──
        cy, cx = prop.centroid
        img_h, img_w = labeled_mask.shape
        img_mid_x = img_w / 2

        # Determine group (Coated = left half, Control = right half)
        group = "Coated" if cx < img_mid_x else "Control"

        # Determine ratio row (6 rows divided by image height)
        row_band = img_h / 6.0
        ratio_idx = min(int(cy / row_band), 5)
        ratio = RATIO_LABELS[ratio_idx]

        # Within-group position (1, 2, or 3 - left to right within cell)
        grape_id = idx + 1

        rows.append({
            "Image": image_name,
            "Grape_ID": grape_id,
            "Group": group,
            "Ratio": ratio,
            "Centroid_X": round(cx, 1),
            "Centroid_Y": round(cy, 1),
            # Area
            "Area_px2": area_px,
            "Area_mm2": area_mm2,
            # RGB
            "Mean_R": mean_R, "StdDev_R": std_R,
            "Mean_G": mean_G, "StdDev_G": std_G,
            "Mean_B": mean_B, "StdDev_B": std_B,
            # Lab
            "Mean_L": mean_L,  "StdDev_L": std_L,
            "Mean_a": mean_a,  "StdDev_a": std_a,
            "Mean_b": mean_b,  "StdDev_b": std_b_s,
            # HSB
            "Mean_H": mean_H,  "StdDev_H": std_H,
            "Mean_S": mean_S,  "StdDev_S": std_S,
            "Mean_Brightness": mean_B_val, "StdDev_Brightness": std_B_val,
        })

    df = pd.DataFrame(rows)
    return df


def compute_scale_from_ruler(image_rgb: np.ndarray) -> float | None:
    """
    Auto-detect px/mm scale from the ruler on the left edge of the image.
    Detects dark tick marks on the ruler and measures spacing.
    Returns px_per_mm or None if detection fails.
    """
    try:
        h, w = image_rgb.shape[:2]
        # Ruler is on the left ~8% of image width
        ruler_strip = image_rgb[:, :int(w * 0.08), :]

        gray = cv2.cvtColor(ruler_strip, cv2.COLOR_RGB2GRAY)
        # Tick marks are dark on light ruler
        _, binary = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV)

        # Find horizontal lines (tick marks)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 1))
        lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

        # Find row positions of tick marks
        row_sums = lines.sum(axis=1)
        tick_rows = np.where(row_sums > lines.shape[1] * 0.3)[0]

        if len(tick_rows) < 2:
            return None

        # Find clusters of rows (each tick mark = cluster)
        tick_centers = []
        current_cluster = [tick_rows[0]]
        for r in tick_rows[1:]:
            if r - current_cluster[-1] <= 5:
                current_cluster.append(r)
            else:
                tick_centers.append(np.mean(current_cluster))
                current_cluster = [r]
        tick_centers.append(np.mean(current_cluster))

        if len(tick_centers) < 2:
            return None

        # Each tick = 1 mm
        spacings = np.diff(tick_centers)
        px_per_mm = float(np.median(spacings))

        if 5 < px_per_mm < 200:  # sanity check
            return px_per_mm

        return None

    except Exception as e:
        print(f"[Scale] Ruler detection failed: {e}")
        return None


def measure_grape_python(image_rgb: np.ndarray, mask: np.ndarray,
                         px_per_mm: float = None) -> dict:
    """
    Measure a single grape region: Area, Centroid, RGB, Lab, HSB.
    ImageJ-equivalent formulas:
    - Lab: skimage.color.rgb2lab
    - H scaled ×255/180 to match ImageJ HSB Stack (0-255 range)
    Returns a dict with all metric values. Never returns None.
    """
    pixels_bool = mask > 0
    n_px = int(pixels_bool.sum())

    if n_px == 0:
        # Fallback: treat entire crop as the grape
        pixels_bool = np.ones(mask.shape, dtype=bool)
        n_px = int(pixels_bool.sum())

    ys, xs = np.where(pixels_bool)

    # ── RGB ──
    R = image_rgb[:, :, 0][pixels_bool].astype(float)
    G = image_rgb[:, :, 1][pixels_bool].astype(float)
    B = image_rgb[:, :, 2][pixels_bool].astype(float)

    # ── Lab (skimage matches ImageJ Lab Stack) ──
    image_lab = skcolor.rgb2lab(image_rgb)
    L_px = image_lab[:, :, 0][pixels_bool]
    a_px = image_lab[:, :, 1][pixels_bool]
    b_px = image_lab[:, :, 2][pixels_bool]

    # ── HSB (OpenCV HSV; H scaled ×255/180 to match ImageJ 0-255 range) ──
    image_hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)
    H_px = image_hsv[:, :, 0][pixels_bool].astype(float) * (255.0 / 180.0)
    S_px = image_hsv[:, :, 1][pixels_bool].astype(float)
    V_px = image_hsv[:, :, 2][pixels_bool].astype(float)

    area_mm2 = round(n_px / (px_per_mm ** 2), 3) if px_per_mm else None

    return {
        "Area_px2":        n_px,
        "Area_mm2":        area_mm2,
        "Centroid_X":      round(float(xs.mean()), 1),
        "Centroid_Y":      round(float(ys.mean()), 1),
        "Mean_R":          round(float(R.mean()), 3),
        "Mean_G":          round(float(G.mean()), 3),
        "Mean_B":          round(float(B.mean()), 3),
        "Mean_L":          round(float(L_px.mean()), 3),
        "Mean_a":          round(float(a_px.mean()), 3),
        "Mean_b":          round(float(b_px.mean()), 3),
        "Mean_H":          round(float(H_px.mean()), 3),
        "Mean_S":          round(float(S_px.mean()), 3),
        "Mean_Brightness": round(float(V_px.mean()), 3),
    }


def compare_with_imagej(python_df: pd.DataFrame,
                         imagej_csv_path: str) -> pd.DataFrame:
    """
    Compare Python measurements vs ImageJ baseline CSV.
    Computes % error for each metric.
    Returns comparison DataFrame with error columns.
    """
    try:
        ij_df = pd.read_csv(imagej_csv_path)
    except Exception as e:
        print(f"[Compare] Cannot read ImageJ CSV: {e}")
        return pd.DataFrame()

    # Parse ImageJ whole-image CSV format (rows = channels)
    # Format: rows for R,G,B,L,a,b,H,S,Brightness with Area repeated
    ij_metrics = {}
    label_col = "Label" if "Label" in ij_df.columns else ij_df.columns[1]
    mean_col = "Mean1" if "Mean1" in ij_df.columns else "Mean"
    area_col = "Area1" if "Area1" in ij_df.columns else "Area"

    for _, row in ij_df.iterrows():
        label = str(row.get(label_col, ""))
        mean_val = row.get(mean_col, None)
        area_val = row.get(area_col, None)

        if "_RGB" in label:
            if "Mean_R_ij" not in ij_metrics:
                ij_metrics["Mean_R_ij"] = mean_val
                ij_metrics["Area_ij"] = area_val
            elif "Mean_G_ij" not in ij_metrics:
                ij_metrics["Mean_G_ij"] = mean_val
            elif "Mean_B_ij" not in ij_metrics:
                ij_metrics["Mean_B_ij"] = mean_val
        elif "_LAB" in label or "L* " in label or "a*" in label or "b*" in label:
            if "Mean_L_ij" not in ij_metrics:
                ij_metrics["Mean_L_ij"] = mean_val
            elif "Mean_a_ij" not in ij_metrics:
                ij_metrics["Mean_a_ij"] = mean_val
            elif "Mean_b_ij" not in ij_metrics:
                ij_metrics["Mean_b_ij"] = mean_val
        elif "_HSB" in label or "Hue" in label or "Sat" in label or "Bright" in label:
            if "Mean_H_ij" not in ij_metrics:
                ij_metrics["Mean_H_ij"] = mean_val
            elif "Mean_S_ij" not in ij_metrics:
                ij_metrics["Mean_S_ij"] = mean_val
            elif "Mean_Brightness_ij" not in ij_metrics:
                ij_metrics["Mean_Brightness_ij"] = mean_val

    # Compute global means from Python (all grapes combined = comparable to ImageJ whole-image)
    python_means = {
        "Area_px2_py": python_df["Area_px2"].sum(),
        "Mean_R_py": python_df["Mean_R"].mean(),
        "Mean_G_py": python_df["Mean_G"].mean(),
        "Mean_B_py": python_df["Mean_B"].mean(),
        "Mean_L_py": python_df["Mean_L"].mean(),
        "Mean_a_py": python_df["Mean_a"].mean(),
        "Mean_b_py": python_df["Mean_b"].mean(),
        "Mean_H_py": python_df["Mean_H"].mean(),
        "Mean_S_py": python_df["Mean_S"].mean(),
        "Mean_Brightness_py": python_df["Mean_Brightness"].mean(),
    }

    # Build comparison table
    metric_pairs = [
        ("Area", "Area_ij", "Area_px2_py"),
        ("Mean R", "Mean_R_ij", "Mean_R_py"),
        ("Mean G", "Mean_G_ij", "Mean_G_py"),
        ("Mean B", "Mean_B_ij", "Mean_B_py"),
        ("Mean L*", "Mean_L_ij", "Mean_L_py"),
        ("Mean a*", "Mean_a_ij", "Mean_a_py"),
        ("Mean b*", "Mean_b_ij", "Mean_b_py"),
        ("Mean H", "Mean_H_ij", "Mean_H_py"),
        ("Mean S", "Mean_S_ij", "Mean_S_py"),
        ("Mean Brightness", "Mean_Brightness_ij", "Mean_Brightness_py"),
    ]

    comparison_rows = []
    for metric_name, ij_key, py_key in metric_pairs:
        ij_val = ij_metrics.get(ij_key)
        py_val = python_means.get(py_key)

        if ij_val is not None and py_val is not None:
            try:
                ij_val = float(ij_val)
                py_val = float(py_val)
                if ij_val != 0:
                    pct_error = abs((py_val - ij_val) / ij_val) * 100
                else:
                    pct_error = None
            except:
                pct_error = None
        else:
            pct_error = None

        comparison_rows.append({
            "Metric": metric_name,
            "ImageJ_Value": round(ij_val, 3) if ij_val is not None else "N/A",
            "Python_Value": round(py_val, 3) if py_val is not None else "N/A",
            "Abs_Error_%": round(pct_error, 2) if pct_error is not None else "N/A",
            "Status": ("✅ <5%" if pct_error is not None and pct_error < 5
                       else "⚠️ >5%" if pct_error is not None
                       else "N/A"),
        })

    return pd.DataFrame(comparison_rows)

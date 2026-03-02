"""
imagej_runner.py - Runs Fiji/ImageJ headless from Python
Calls the PerGrape_Color_Macro.ijm automatically
"""

import subprocess
import os
from pathlib import Path
import platform
import pandas as pd


def find_fiji() -> str | None:
    """
    Auto-detect Fiji installation on Mac/Linux/Windows.
    Returns path to Fiji executable or None if not found.
    """
    system = platform.system()

    if system == "Darwin":  # Mac
        candidates = [
            "/Applications/Fiji.app/Contents/MacOS/ImageJ-macosx",
            str(Path.home() / "Applications/Fiji.app/Contents/MacOS/ImageJ-macosx"),
            "/opt/fiji/Fiji.app/Contents/MacOS/ImageJ-macosx",
        ]
    elif system == "Linux":
        candidates = [
            "/opt/fiji/ImageJ-linux64",
            str(Path.home() / "fiji/ImageJ-linux64"),
        ]
    elif system == "Windows":
        candidates = [
            r"C:\Fiji.app\ImageJ-win64.exe",
            str(Path.home() / r"AppData\Local\Fiji.app\ImageJ-win64.exe"),
        ]
    else:
        return None

    for path in candidates:
        if os.path.exists(path):
            return path

    return None


def run_imagej_macro(image_folder: str, fiji_path: str = None) -> str | None:
    """
    Run the PerGrape_Color_Macro.ijm on a folder of images using Fiji headless.
    Returns path to output CSV, or None if failed.
    """
    if fiji_path is None:
        fiji_path = find_fiji()

    if fiji_path is None:
        print("[ImageJ] Fiji not found. Skipping ImageJ measurement.")
        print("[ImageJ] To install: download from https://imagej.net/software/fiji/downloads")
        return None

    macro_path = Path(__file__).parent.parent / "imagej_macros" / "PerGrape_Color_Macro.ijm"

    if not macro_path.exists():
        print(f"[ImageJ] Macro not found: {macro_path}")
        return None

    print(f"[ImageJ] Running macro: {macro_path}")
    print(f"[ImageJ] Image folder: {image_folder}")
    print(f"[ImageJ] Fiji path: {fiji_path}")

    try:
        result = subprocess.run(
            [fiji_path, "--headless", "--console", "-macro", str(macro_path), image_folder],
            capture_output=True, text=True, timeout=300
        )

        print("[ImageJ STDOUT]:", result.stdout[-1000:] if result.stdout else "(none)")
        if result.stderr:
            print("[ImageJ STDERR]:", result.stderr[-500:])

        # Look for output CSV in image folder
        output_files = list(Path(image_folder).glob("*_PerGrape_Raw.csv"))
        if output_files:
            return str(output_files[-1])

        return None

    except subprocess.TimeoutExpired:
        print("[ImageJ] Timed out after 300s")
        return None
    except Exception as e:
        print(f"[ImageJ] Error: {e}")
        return None


def parse_imagej_per_grape_csv(csv_path: str) -> pd.DataFrame:
    """
    Parse the long-format CSV output from PerGrape_Color_Macro.ijm
    and reshape to 1 row per grape.

    ImageJ output format (long): each grape gets 3 rows per colorspace (R,G,B then L,a,b then H,S,B)
    We need to pivot this to 1 row per grape with all channels as columns.
    """
    try:
        df_raw = pd.read_csv(csv_path)
        print(f"[ImageJ Parser] Raw CSV shape: {df_raw.shape}")
        print(f"[ImageJ Parser] Columns: {list(df_raw.columns)}")

        # Detect structure: look for Label column
        if "Label" not in df_raw.columns and df_raw.columns[1] == "Label":
            df_raw.columns = ["idx"] + list(df_raw.columns[1:])

        # The macro outputs ROI data in groups of 3 per grape per colorspace
        # Structure: n_grapes rows for R, n_grapes rows for G, n_grapes rows for B
        #            then n_grapes for L, a, b
        #            then n_grapes for H, S, Brightness
        # Total rows = n_grapes * 9

        n_total = len(df_raw)
        n_grapes = n_total // 9  # 3 colorspaces × 3 channels each

        if n_total % 9 != 0:
            print(f"[ImageJ Parser] Warning: {n_total} rows not divisible by 9")
            n_grapes = n_total // 9

        mean_col = "Mean1" if "Mean1" in df_raw.columns else "Mean"
        area_col = "Area1" if "Area1" in df_raw.columns else "Area"

        rows = []
        for i in range(n_grapes):
            # RGB: rows i, i+n_grapes, i+2*n_grapes
            r_idx = i
            g_idx = i + n_grapes
            b_idx = i + 2 * n_grapes
            # Lab: rows i+3*n, i+4*n, i+5*n
            l_idx = i + 3 * n_grapes
            a_idx = i + 4 * n_grapes
            b_star_idx = i + 5 * n_grapes
            # HSB: rows i+6*n, i+7*n, i+8*n
            h_idx = i + 6 * n_grapes
            s_idx = i + 7 * n_grapes
            br_idx = i + 8 * n_grapes

            def safe_get(idx, col):
                try:
                    if idx < len(df_raw):
                        return float(df_raw.iloc[idx][col])
                except:
                    pass
                return None

            rows.append({
                "Grape_ID_IJ": i + 1,
                "Area_px2_IJ": safe_get(r_idx, area_col),
                "Mean_R_IJ": safe_get(r_idx, mean_col),
                "Mean_G_IJ": safe_get(g_idx, mean_col),
                "Mean_B_IJ": safe_get(b_idx, mean_col),
                "Mean_L_IJ": safe_get(l_idx, mean_col),
                "Mean_a_IJ": safe_get(a_idx, mean_col),
                "Mean_b_IJ": safe_get(b_star_idx, mean_col),
                "Mean_H_IJ": safe_get(h_idx, mean_col),
                "Mean_S_IJ": safe_get(s_idx, mean_col),
                "Mean_Brightness_IJ": safe_get(br_idx, mean_col),
            })

        return pd.DataFrame(rows)

    except Exception as e:
        print(f"[ImageJ Parser] Failed: {e}")
        return pd.DataFrame()

"""
imagej_measure.py - Actual ImageJ measurements via pyimagej
ImageJ runs inside Python — same engine, same formulas, same results as manual Fiji.
"""

import numpy as np
import pandas as pd
import cv2
import tempfile
import os
from pathlib import Path

# ─────────────────────────────────────────────
#  ImageJ Singleton (start once, reuse)
# ─────────────────────────────────────────────

_ij = None

def get_imagej(progress_fn=None):
    """
    Start ImageJ (via pyimagej) once and reuse.
    Downloads Fiji automatically on first run (~300MB, one time only).
    Requires Java (openjdk) installed: brew install openjdk

    progress_fn: optional callable(str) for real-time status updates to the UI.
    """
    global _ij
    if _ij is not None:
        return _ij

    def log(msg):
        print(msg)
        if progress_fn:
            progress_fn(msg)

    try:
        import imagej
        log("🔬 Starting ImageJ engine — first run downloads Fiji (~300 MB), may take several minutes...")
        _ij = imagej.init('sc.fiji:fiji', mode='headless')
        log(f"✅ ImageJ started. Version: {_ij.getVersion()}")
        return _ij
    except Exception as e:
        log(f"⚠️ ImageJ failed to start: {e}")
        log("   Tip: install Java with 'brew install openjdk', then restart the app.")
        return None


def is_imagej_available() -> bool:
    """Check if pyimagej + Java are both installed and working."""
    try:
        import imagej
        import jpype
    except ImportError:
        return False

    # Packages alone are not enough — Java must actually be present
    import subprocess
    try:
        result = subprocess.run(
            ['java', '-version'],
            capture_output=True,
            timeout=5,
        )
        return result.returncode == 0
    except (FileNotFoundError, OSError, subprocess.TimeoutExpired):
        return False


# ─────────────────────────────────────────────
#  Core measurement function
# ─────────────────────────────────────────────

def measure_grapes_imagej(image_rgb: np.ndarray,
                           labeled_mask: np.ndarray,
                           image_path: str,
                           px_per_mm: float = None) -> pd.DataFrame:
    """
    Measure Area + RGB + Lab + HSB for each grape using actual ImageJ engine.

    Process:
    1. Save image + mask to temp files
    2. Run ImageJ macro headlessly (Lab Stack, HSB Stack, RGB Stack, multimeasure)
    3. Parse results back into DataFrame (1 row per grape)

    This produces IDENTICAL results to manually running in Fiji.
    """
    ij = get_imagej(progress_fn=None)
    if ij is None:
        print("[ImageJ] Not available, falling back to Python measurements")
        from app.core.measure import measure_all_grapes
        return measure_all_grapes(image_rgb, labeled_mask, image_path, px_per_mm)

    try:
        import tempfile, json
        from skimage.measure import regionprops

        image_name = Path(image_path).name

        # ── Save image to temp file (ImageJ reads from disk) ──
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_image = os.path.join(tmpdir, "grape_image.tif")
            tmp_mask  = os.path.join(tmpdir, "grape_mask.tif")
            tmp_macro = os.path.join(tmpdir, "measure.ijm")
            tmp_out   = os.path.join(tmpdir, "results.csv")

            # Save image as TIFF (lossless)
            image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
            cv2.imwrite(tmp_image, image_bgr)

            # Save labeled mask
            # Convert labeled mask to binary (255=grape, 0=background)
            # We'll measure each grape ROI separately
            props = regionprops(labeled_mask)
            props_sorted = sorted(props, key=lambda p: (p.centroid[0], p.centroid[1]))

            # Build and run macro for each grape
            all_rows = []

            for idx, prop in enumerate(props_sorted):
                grape_id = idx + 1

                # Save individual grape mask
                single_mask = ((labeled_mask == prop.label).astype(np.uint8)) * 255
                cv2.imwrite(tmp_mask, single_mask)

                # Write ImageJ macro for this grape
                macro = _build_measurement_macro(tmp_image, tmp_mask, tmp_out)
                with open(tmp_macro, 'w') as f:
                    f.write(macro)

                # Run macro via pyimagej
                ij.script().run("measure.ijm", macro, True, {})

                # Parse output CSV
                row_data = _parse_imagej_output(tmp_out, grape_id, prop,
                                                 labeled_mask.shape, image_name, px_per_mm)
                if row_data:
                    all_rows.append(row_data)

                # Clean results for next grape
                try:
                    ij.py.run_macro('run("Clear Results");')
                except:
                    pass

            df = pd.DataFrame(all_rows)
            print(f"[ImageJ] Measured {len(df)} grapes successfully")
            return df

    except Exception as e:
        print(f"[ImageJ] Measurement error: {e}")
        import traceback
        traceback.print_exc()
        print("[ImageJ] Falling back to Python measurements")
        from app.core.measure import measure_all_grapes
        return measure_all_grapes(image_rgb, labeled_mask, image_path, px_per_mm)


def _build_measurement_macro(image_path: str, mask_path: str, output_path: str) -> str:
    """
    Build an ImageJ macro string that:
    1. Opens the grape image
    2. Opens the mask and creates ROI from it
    3. Measures RGB, Lab, HSB inside that ROI
    4. Saves results to CSV

    This is the EXACT same logic as your original macro, per grape.
    """
    # Escape paths for ImageJ macro (use forward slashes)
    img_p  = image_path.replace("\\", "/")
    mask_p = mask_path.replace("\\", "/")
    out_p  = output_path.replace("\\", "/")

    macro = f"""
// Open main image
open("{img_p}");
currentImage = getImageID();

// Open mask and create ROI
open("{mask_p}");
setThreshold(128, 255);
run("Create Selection");
close(); // close mask image

// Transfer selection to main image
selectImage(currentImage);
run("Restore Selection");

// ─── RGB measurements ───
run("Duplicate...", "title=RGB_copy");
selectWindow("RGB_copy");
run("Make Composite");
run("ROI Manager...");
roiManager("Reset");
run("Restore Selection");
roiManager("Add");
run("Set Measurements...", "area mean standard modal min limit decimal=3");
roiManager("multi-measure measure_all one append");
roiManager("Reset");
close("RGB_copy");

// ─── Lab measurements ───
selectImage(currentImage);
run("Duplicate...", "title=LAB_copy");
selectWindow("LAB_copy");
run("Lab Stack");
run("Restore Selection");
roiManager("Add");
run("Set Measurements...", "area mean standard modal min limit decimal=3");
roiManager("multi-measure measure_all one append");
roiManager("Reset");
close("LAB_copy");

// ─── HSB measurements ───
selectImage(currentImage);
run("Duplicate...", "title=HSB_copy");
selectWindow("HSB_copy");
run("HSB Stack");
run("Restore Selection");
roiManager("Add");
run("Set Measurements...", "area mean standard modal min limit decimal=3");
roiManager("multi-measure measure_all one append");
roiManager("Reset");
close("HSB_copy");

// Save results
saveAs("Measurements", "{out_p}");
run("Clear Results");
roiManager("Reset");
selectImage(currentImage);
close();
"""
    return macro


def _parse_imagej_output(csv_path: str, grape_id: int, prop,
                          image_shape: tuple, image_name: str,
                          px_per_mm: float = None) -> dict | None:
    """
    Parse the ImageJ output CSV for one grape.
    ImageJ outputs 9 rows (3 channels × 3 colorspaces).
    We extract: Area, Mean_R/G/B, Mean_L/a/b, Mean_H/S/Brightness
    """
    try:
        if not os.path.exists(csv_path):
            return None

        df = pd.read_csv(csv_path)
        if df.empty or len(df) < 9:
            return None

        mean_col = "Mean1" if "Mean1" in df.columns else ("Mean" if "Mean" in df.columns else df.columns[3])
        area_col = "Area1" if "Area1" in df.columns else ("Area" if "Area" in df.columns else df.columns[2])

        def get_mean(row_idx):
            try:
                return float(df.iloc[row_idx][mean_col])
            except:
                return None

        def get_area(row_idx):
            try:
                return float(df.iloc[row_idx][area_col])
            except:
                return None

        # Row order from macro:
        # 0=R, 1=G, 2=B  (RGB composite = 3 slices)
        # 3=L, 4=a, 5=b  (Lab stack = 3 slices)
        # 6=H, 7=S, 8=B  (HSB stack = 3 slices)
        area_px = get_area(0)
        area_mm2 = round(area_px / (px_per_mm ** 2), 3) if (area_px and px_per_mm) else None

        # Position
        img_h, img_w = image_shape
        cy, cx = prop.centroid
        group = "Coated" if cx < img_w / 2 else "Control"
        ratio_labels = ["100:0", "80:20", "60:40", "40:60", "20:80", "0:100"]
        ratio_idx = min(int(cy / (img_h / 6.0)), 5)
        ratio = ratio_labels[ratio_idx]

        return {
            "Image":            image_name,
            "Grape_ID":         grape_id,
            "Group":            group,
            "Ratio":            ratio,
            "Centroid_X":       round(cx, 1),
            "Centroid_Y":       round(cy, 1),
            # Area
            "Area_px2":         int(area_px) if area_px else None,
            "Area_mm2":         area_mm2,
            # RGB (ImageJ values)
            "Mean_R":           get_mean(0),
            "Mean_G":           get_mean(1),
            "Mean_B":           get_mean(2),
            # Lab (ImageJ values)
            "Mean_L":           get_mean(3),
            "Mean_a":           get_mean(4),
            "Mean_b":           get_mean(5),
            # HSB (ImageJ values)
            "Mean_H":           get_mean(6),
            "Mean_S":           get_mean(7),
            "Mean_Brightness":  get_mean(8),
            # Source
            "Measurement_Engine": "ImageJ (pyimagej)",
        }

    except Exception as e:
        print(f"[ImageJ Parser] Error parsing grape {grape_id}: {e}")
        return None


# ─────────────────────────────────────────────
#  Batch macro approach (faster — all 36 grapes in one ImageJ call)
# ─────────────────────────────────────────────

def measure_grapes_imagej_batch(image_rgb: np.ndarray,
                                 labeled_mask: np.ndarray,
                                 image_path: str,
                                 px_per_mm: float = None,
                                 progress_fn=None) -> pd.DataFrame:
    """
    Faster approach: send all 36 grape ROIs to ImageJ in one call.
    Uses ROI Manager with all masks at once → single multimeasure pass per colorspace.
    This matches your original macro logic most closely.

    progress_fn: optional callable(str) for real-time status updates to the UI.
    """
    ij = get_imagej(progress_fn=progress_fn)
    if ij is None:
        from app.core.measure import measure_all_grapes
        return measure_all_grapes(image_rgb, labeled_mask, image_path, px_per_mm)

    try:
        import tempfile
        from skimage.measure import regionprops
        import zipfile, struct

        image_name = Path(image_path).name
        props = regionprops(labeled_mask)
        props_sorted = sorted(props, key=lambda p: (p.centroid[0], p.centroid[1]))
        n = len(props_sorted)

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_image = os.path.join(tmpdir, "image.tif")
            tmp_out_rgb = os.path.join(tmpdir, "results_rgb.csv")
            tmp_out_lab = os.path.join(tmpdir, "results_lab.csv")
            tmp_out_hsb = os.path.join(tmpdir, "results_hsb.csv")

            # Save image
            image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
            cv2.imwrite(tmp_image, image_bgr)

            # Save individual mask files (one per grape)
            mask_paths = []
            for i, prop in enumerate(props_sorted):
                mask = ((labeled_mask == prop.label).astype(np.uint8)) * 255
                mask_path = os.path.join(tmpdir, f"mask_{i:03d}.tif")
                cv2.imwrite(mask_path, mask)
                mask_paths.append(mask_path)

            # Build batch macro
            macro = _build_batch_macro(
                tmp_image, mask_paths,
                tmp_out_rgb, tmp_out_lab, tmp_out_hsb,
                tmpdir
            )

            tmp_macro = os.path.join(tmpdir, "batch_measure.ijm")
            with open(tmp_macro, 'w') as f:
                f.write(macro)

            # Run macro
            print(f"[ImageJ] Running batch measurement for {n} grapes...")
            ij.script().run("batch_measure.ijm", macro, True, {})

            # Parse results
            df_rgb = _parse_batch_csv(tmp_out_rgb, ["Area_px2", "Mean_R", "Mean_G", "Mean_B"])
            df_lab = _parse_batch_csv(tmp_out_lab, ["_Area_L", "Mean_L", "Mean_a", "Mean_b"])
            df_hsb = _parse_batch_csv(tmp_out_hsb, ["_Area_H", "Mean_H", "Mean_S", "Mean_Brightness"])

            # Combine
            rows = []
            img_h, img_w = labeled_mask.shape

            for i, prop in enumerate(props_sorted):
                cy, cx = prop.centroid
                group = "Coated" if cx < img_w / 2 else "Control"
                ratio_labels = ["100:0", "80:20", "60:40", "40:60", "20:80", "0:100"]
                ratio_idx = min(int(cy / (img_h / 6.0)), 5)

                area_px = df_rgb.get("Area_px2", [None] * n)[i] if df_rgb else (labeled_mask == prop.label).sum()
                area_mm2 = round(area_px / (px_per_mm ** 2), 3) if (area_px and px_per_mm) else None

                rows.append({
                    "Image":            image_name,
                    "Grape_ID":         i + 1,
                    "Group":            group,
                    "Ratio":            ratio_labels[ratio_idx],
                    "Centroid_X":       round(cx, 1),
                    "Centroid_Y":       round(cy, 1),
                    "Area_px2":         int(area_px) if area_px else None,
                    "Area_mm2":         area_mm2,
                    "Mean_R":           _safe_get(df_rgb, "Mean_R", i),
                    "Mean_G":           _safe_get(df_rgb, "Mean_G", i),
                    "Mean_B":           _safe_get(df_rgb, "Mean_B", i),
                    "Mean_L":           _safe_get(df_lab, "Mean_L", i),
                    "Mean_a":           _safe_get(df_lab, "Mean_a", i),
                    "Mean_b":           _safe_get(df_lab, "Mean_b", i),
                    "Mean_H":           _safe_get(df_hsb, "Mean_H", i),
                    "Mean_S":           _safe_get(df_hsb, "Mean_S", i),
                    "Mean_Brightness":  _safe_get(df_hsb, "Mean_Brightness", i),
                    "Measurement_Engine": "ImageJ (pyimagej)",
                })

            return pd.DataFrame(rows)

    except Exception as e:
        print(f"[ImageJ Batch] Error: {e}")
        import traceback
        traceback.print_exc()
        from app.core.measure import measure_all_grapes
        return measure_all_grapes(image_rgb, labeled_mask, image_path, px_per_mm)


def _build_batch_macro(image_path, mask_paths, out_rgb, out_lab, out_hsb, tmpdir):
    """Build a single ImageJ macro that measures all grapes at once."""
    img_p = image_path.replace("\\", "/")
    out_rgb_p = out_rgb.replace("\\", "/")
    out_lab_p = out_lab.replace("\\", "/")
    out_hsb_p = out_hsb.replace("\\", "/")

    # Build mask loading lines
    mask_lines = ""
    for i, mp in enumerate(mask_paths):
        mp_fwd = mp.replace("\\", "/")
        mask_lines += f"""
    open("{mp_fwd}");
    setThreshold(128, 255);
    run("Create Selection");
    close();
    selectImage(mainImage);
    run("Restore Selection");
    roiManager("Add");
"""

    macro = f"""
open("{img_p}");
mainImage = getImageID();

run("ROI Manager...");
roiManager("Reset");

// Load all grape masks as ROIs
{mask_lines}

// ─── RGB measurements ───
selectImage(mainImage);
run("Duplicate...", "title=RGB_batch");
selectWindow("RGB_batch");
run("Make Composite");
run("Set Measurements...", "area mean standard modal min limit decimal=3");
roiManager("multi-measure measure_all one append");
saveAs("Measurements", "{out_rgb_p}");
run("Clear Results");
close("RGB_batch");

// ─── Lab measurements ───
selectImage(mainImage);
run("Duplicate...", "title=LAB_batch");
selectWindow("LAB_batch");
run("Lab Stack");
run("Set Measurements...", "area mean standard modal min limit decimal=3");
roiManager("multi-measure measure_all one append");
saveAs("Measurements", "{out_lab_p}");
run("Clear Results");
close("LAB_batch");

// ─── HSB measurements ───
selectImage(mainImage);
run("Duplicate...", "title=HSB_batch");
selectWindow("HSB_batch");
run("HSB Stack");
run("Set Measurements...", "area mean standard modal min limit decimal=3");
roiManager("multi-measure measure_all one append");
saveAs("Measurements", "{out_hsb_p}");
run("Clear Results");
close("HSB_batch");

roiManager("Reset");
selectImage(mainImage);
close();
"""
    return macro


def _parse_batch_csv(csv_path: str, col_names: list) -> dict:
    """Parse a batch measurement CSV from ImageJ."""
    try:
        if not os.path.exists(csv_path):
            return {}
        df = pd.read_csv(csv_path)
        if df.empty:
            return {}

        mean_col = "Mean1" if "Mean1" in df.columns else "Mean"
        area_col = "Area1" if "Area1" in df.columns else "Area"
        n = len(df) // 3  # 3 channels per colorspace

        result = {
            col_names[0]: [float(df.iloc[i][area_col]) for i in range(n)],
            col_names[1]: [float(df.iloc[i][mean_col]) for i in range(n)],
            col_names[2]: [float(df.iloc[i + n][mean_col]) for i in range(n)],
            col_names[3]: [float(df.iloc[i + 2*n][mean_col]) for i in range(n)],
        }
        return result
    except Exception as e:
        print(f"[Batch CSV Parser] Error: {e}")
        return {}


def _safe_get(d: dict, key: str, idx: int):
    """Safely get value from parsed dict."""
    try:
        return round(d[key][idx], 3)
    except:
        return None

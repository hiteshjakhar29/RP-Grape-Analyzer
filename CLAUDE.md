# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this project is
A Python desktop GUI application that automates image analysis of grapes.
36 grapes are photographed over 300 days to observe how coating affects them.
The app replaces a manual ImageJ workflow — user uploads one image and gets
all measurements automatically.

## Running the app

```bash
./run.sh        # Recommended: handles venv setup and launch
python main.py  # Direct launch (after activating venv manually)
```

`run.sh` handles: venv creation, `pip install -r requirements.txt`, and launches `python main.py`.

**Fiji requirement:** Fiji must be installed separately (fiji.sc). Default search path is `/Applications/FijiWorking/Fiji.app`. If not found, the app prompts at startup to browse for it or fall back to Python formulas.

## Architecture

### Architecture
Single implementation using reference masks:

- `main.py` — PySide6 entry point (dark theme, sets up `MainWindow`)
- `ui/main_window.py` — Main window: `AnalysisWorker` (QThread), toolbar, pipeline viewer, results table
- `ui/pipeline_viewer.py` — 6-step clickable thumbnail strip
- `ui/results_table.py` — 36-row results table with StdDev columns
- `core/image_pipeline.py` — Background removal + HSV binary segmentation
- `core/mask_engine.py` — Reference mask loading, image alignment (ORB), adaptive mask fitting, GrabCut refinement, visualization
- `core/color_engine.py` — Fiji subprocess measurements (means + StdDev) with Python fallback
- `core/exporter.py` — Excel export with styled headers
- `config.py` — Paths + `REF_DAY = 150` (reference image day number)

### Segmentation strategy (reference mask approach)
`AnalysisWorker.run()` in `ui/main_window.py` orchestrates the pipeline:
1. Resize to 2450×2707 reference resolution (`normalize_to_reference`)
2. Align to reference image via ORB feature matching (`align_image`)
3. Generate binary mask: HSV threshold + morphological close/open + fill holes (`generate_binary`)
4. Load 36 pre-computed reference masks from `assetes/reference_masks.npz`
5. For each grape: `adapt_mask()` (day-aware tolerance) → `refine_mask_grabcut()` → final session mask
6. Measure all 36 grapes in one Fiji subprocess call (`measure_all_grapes`)
7. Compute StdDev for all channels in Python, merge with Fiji means

### `adapt_mask()` — day-aware tolerance
Accepts `day` and `ref_day` (from `config.REF_DAY`). Tolerance scales automatically:
- `day < ref_day` → higher `grow_tol` (up to 0.60): grapes are bigger than reference
- `day > ref_day` → higher `shrink_tol` (up to 0.70): grapes have shrunken more
- Falls back to progressively eroding the reference mask if no blob found

### `refine_mask_grabcut()`
After `adapt_mask`, GrabCut refines the exact boundary using the original image. The eroded mask centre is seeded as sure-foreground. Rejected (falls back to adapted mask) if area shifts >40%.

### Measurement engine (`core/color_engine.py`)
`measure_all_grapes()` calls Fiji as a subprocess with `assetes/measure_grapes_batch.ijm` to get all 36 grape means in one call. StdDev is computed in Python from the same pixel data. Falls back to pure Python formulas (mathematically identical Lab/HSB) if Fiji is not found.

Color space notes:
- Lab: `skimage.color.rgb2lab()` (CIE D65, matches ImageJ Lab Stack)
- HSB: OpenCV HSV with H rescaled to 0–255 range to match ImageJ

### QC (`app/core/qc.py`)
`run_qc_checks()` validates: exact count of 36, area outliers (>3× or <1/3 median), solidity <0.6, hue outside 10–120, per-cell count != 3, GrabCut area shifts >30%. Returns `PASS`/`WARNING`/`FAIL`.

## What we are measuring (per grape, 36 grapes per image)
- Area (px² and mm²)
- RGB: Mean R, Mean G, Mean B + StdDev each
- Lab: Mean L*, Mean a*, Mean b* + StdDev each
- HSB: Mean Hue, Mean Saturation, Mean Brightness + StdDev each

## Image layout (always fixed)
- 6 rows × 2 groups (Coated left, Control right) = 12 cells × 3 grapes = 36 total
- Ratio rows: 100:0, 80:20, 60:40, 40:60, 20:80, 0:100
- Ruler on left edge of image (used for mm/px scale)
- Blue grid lines on white background

## Key constraints
- Must always detect exactly 36 grapes — hard requirement
- Measurements must match ImageJ output (professor validation)
- Java (openjdk) must be installed for pyimagej to work
- App targets macOS with Python 3.10+

## Output format
CSV + Excel (3 sheets: `Per_Grape_Measurements`, `ImageJ_Comparison`, `Group_Summary`) with 36 rows:
```
Image, Grape_ID, Group, Ratio, Centroid_X, Centroid_Y,
Area_px2, Area_mm2,
Mean_R, StdDev_R, Mean_G, StdDev_G, Mean_B, StdDev_B,
Mean_L, StdDev_L, Mean_a, StdDev_a, Mean_b, StdDev_b,
Mean_H, StdDev_H, Mean_S, StdDev_S, Mean_Brightness, StdDev_Brightness,
Measurement_Engine
```

## Dependencies
`requirements.txt` lists: PySide6, numpy, opencv-python, Pillow, scipy, scikit-image, openpyxl.
Also required but not yet in requirements.txt: `pandas`, `pyimagej`, `jpype1` (for ImageJ), `ultralytics` (YOLOv8).

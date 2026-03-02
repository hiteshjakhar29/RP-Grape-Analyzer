# Grape Analyzer — Project Context for Claude

## What this project is
A Python desktop GUI application that automates image analysis of grapes.
36 grapes are photographed over 300 days to observe how coating affects them.
The app replaces a manual ImageJ workflow — user uploads one image and gets 
all measurements automatically.

## What we are measuring (per grape, 36 grapes per image)
- Area (px² and mm²)
- RGB: Mean R, Mean G, Mean B
- Lab: Mean L*, Mean a*, Mean b*
- HSB: Mean Hue, Mean Saturation, Mean Brightness

## Image layout (always fixed)
- 6 rows × 2 groups (Coated left, Control right) = 12 cells × 3 grapes = 36 total
- Ratio rows: 100:0, 80:20, 60:40, 40:60, 20:80, 0:100
- Ruler on left edge of image (used for mm/px scale)
- Blue grid lines on white background

## Measurement engine
ImageJ runs inside Python via pyimagej.
Same Lab Stack, HSB Stack, RGB Stack as the original Fiji macro.
Segmentation is done by Python (OpenCV + YOLOv8), NOT ImageJ.
ImageJ only does the measurements after masks are ready.

## Segmentation strategy (hybrid)
1. Classical: RGB threshold (Blue < 136) + watershed (ImageJ-style)
2. ML: YOLOv8-seg detects grapes
3. Whichever gives count closest to 36 is used
4. QC check: if count != 36, flag as warning/fail

## Tech stack
- GUI: PySide6 (dark theme, 4 tabs)
- Segmentation: OpenCV, scikit-image, scipy, ultralytics (YOLOv8)
- Measurements: pyimagej (actual ImageJ engine, Java-based)
- Export: pandas, openpyxl (CSV + Excel)
- Language: Python 3.10+
- Platform: macOS

## Project structure
grape_analyzer/
├── app/
│   ├── main.py                  ← GUI entry point (PySide6)
│   └── core/
│       ├── pipeline.py          ← Segmentation (classical + YOLOv8 hybrid)
│       ├── imagej_measure.py    ← ImageJ measurements via pyimagej
│       ├── measure.py           ← Python fallback measurements
│       ├── export.py            ← CSV + Excel export
│       └── qc.py                ← QC checks
├── imagej_macros/
│   └── PerGrape_Color_Macro.ijm ← Original macro (reference + manual use)
├── requirements.txt
├── run.sh                       ← Mac launcher
└── CLAUDE.md                    ← This file

## Key constraints
- Must always detect exactly 36 grapes — hard requirement
- Measurements must match ImageJ output (professor validation)
- No comparison tab needed — just accurate measurements + export
- App must work on macOS with Python 3.10+
- Java (openjdk) must be installed for pyimagej to work

## Current known issues being fixed
- pipeline.py has 9 bugs identified (return type, YOLO overlap, 
  always-true masks, model caching, regionprops in loop, etc.)

## Output format
CSV + Excel with 36 rows (1 per grape), columns:
Image, Grape_ID, Group, Ratio, Centroid_X, Centroid_Y,
Area_px2, Area_mm2,
Mean_R, StdDev_R, Mean_G, StdDev_G, Mean_B, StdDev_B,
Mean_L, StdDev_L, Mean_a, StdDev_a, Mean_b, StdDev_b,
Mean_H, StdDev_H, Mean_S, StdDev_S, Mean_Brightness, StdDev_Brightness,
Measurement_Engine
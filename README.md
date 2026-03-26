# Grape Analyzer

A Python desktop application that automates color and area measurement of 36 grapes photographed over a 300-day study. Replaces a manual ImageJ workflow — load one image and get all measurements automatically.

**Measurement engine: Fiji/ImageJ (headless subprocess)** — all Mean and StdDev values are measured directly by ImageJ, identical to running the macro manually in Fiji.

---

## What It Does

1. Load a grape session image (any of your 300 days)
2. The app aligns it to a reference image, adapts pre-computed reference masks to the current grape sizes, and refines each boundary with GrabCut
3. Fiji measures all 36 grapes in a single headless call — Area, RGB, Lab (L\*a\*b\*), HSB (Hue/Saturation/Brightness) with full Mean + StdDev
4. Results shown in a 36-row table and exported to Excel

---

## Prerequisites

| Requirement | Version | Notes |
|---|---|---|
| Python | 3.10+ | `python3 --version` to check |
| Fiji (ImageJ) | Any recent | Must be installed manually — see below |
| macOS | Apple Silicon or Intel | Windows/Linux supported with path changes |

---

## Installation Guide

### Step 1 — Install Fiji

Fiji is the ImageJ distribution used for all measurements. It must be installed **before** running the app.

1. Go to **[https://fiji.sc](https://fiji.sc)** and click **Download**
2. Select your platform (macOS, Windows, or Linux)
3. Extract the downloaded archive

**macOS — recommended install location:**
```
/Applications/FijiWorking/Fiji.app
```
Drag `Fiji.app` from the downloaded `.dmg` or extracted folder into `/Applications/FijiWorking/`. Create the `FijiWorking` folder if it does not exist.

> If you install Fiji elsewhere (e.g. `~/Applications/Fiji.app`), you can point the app to it via the Settings dialog on first launch.

**Verify Fiji works** by double-clicking `Fiji.app` — the ImageJ window should open. Close it before using the analyzer.

---

### Step 2 — Install Python 3.10+

**macOS (using Homebrew — recommended):**
```bash
# Install Homebrew if not already installed
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install Python
brew install python@3.12
```

**Verify:**
```bash
python3 --version
# Should print Python 3.10 or higher
```

**Windows:**
Download the installer from [https://www.python.org/downloads/](https://www.python.org/downloads/). Check **"Add Python to PATH"** during install.

---

### Step 3 — Clone the Repository

```bash
git clone https://github.com/hiteshjakhar29/RP-Grape-Analyzer.git
cd RP-Grape-Analyzer
```

---

### Step 4 — Run the App

**macOS / Linux (recommended — handles venv + dependencies automatically):**
```bash
chmod +x run.sh
./run.sh
```

`run.sh` will:
- Create a Python virtual environment (`.venv/`) on first run
- Install all dependencies from `requirements.txt`
- Launch the app

**Manual setup (if `run.sh` doesn't work):**
```bash
# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate        # macOS/Linux
# .venv\Scripts\activate         # Windows

# Install dependencies
pip install -r requirements.txt

# Launch
python main.py
```

---

### Step 5 — Configure Fiji Path (if needed)

On first launch, the app checks for Fiji at:

| Platform | Default path checked |
|---|---|
| macOS | `/Applications/FijiWorking/Fiji.app/Contents/MacOS/fiji-macos-arm64` |
| Windows | `C:/Fiji.app/fiji-windows-x64.exe` |
| Linux | `/opt/Fiji.app/fiji-linux-x64` |

If Fiji is not found at the default path, a **Settings dialog** appears — click **Browse** and navigate to your Fiji executable. The path is saved for all future sessions.

The header bar shows a **green badge** when Fiji is detected, or **amber** when running in Python-fallback mode.

---

## How to Use

1. Click **Load Image** → select any grape session photo (PNG/JPG)
2. Enter the **day number** for that image (1–300)
3. Click **Run Analysis** — the pipeline runs automatically
4. Watch the **6 pipeline thumbnails** update in real time (click any to enlarge)
5. Results appear in the **36-row table** below
6. Click **Export Excel** to save the `.xlsx` file

---

## Output

Excel file with one row per grape (36 total):

| Column | Source |
|---|---|
| Grape ID, Group (Coated/Control), Ratio, Row, Day | Layout metadata |
| Area (px) | ImageJ |
| Mean R/G/B, Std R/G/B | ImageJ — raw 0–255 |
| Mean L\*, Std L\* | ImageJ — L\* range 0–100 |
| Mean a\*, Std a\* | ImageJ — a\* range −128 to +127 |
| Mean b\*, Std b\* | ImageJ — b\* range −128 to +127 |
| Hue (°), Std H | ImageJ — H rescaled to 0–360° |
| Saturation, Std S | ImageJ — S rescaled to 0–1 |
| Brightness, Std Br | ImageJ — Br rescaled to 0–1 |
| Eccentricity | Python (OpenCV `fitEllipse`) |

> If Fiji is not available, all values except eccentricity fall back to Python formulas that are mathematically equivalent to ImageJ (same CIE D65 Lab pipeline, same Java `Color.RGBtoHSB` algorithm).

---

## Image Layout (Fixed)

```
| 6 rows × 2 groups | Coated (left) | Control (right) |
Each cell has 3 grapes → 6 × 2 × 3 = 36 grapes total

Row 1: Ratio 100:0
Row 2: Ratio  80:20
Row 3: Ratio  60:40
Row 4: Ratio  40:60
Row 5: Ratio  20:80
Row 6: Ratio   0:100
```

Grape positions are **fixed** across all 300 days. The app uses pre-computed reference masks from an intermediate reference day (day 150) and adapts them per session.

---

## Project Structure

```
RP-Grape-Analyzer/
├── main.py                    ← App entry point
├── run.sh                     ← One-click launcher (Mac/Linux)
├── config.py                  ← Paths, REF_DAY = 150
├── requirements.txt
│
├── core/
│   ├── image_pipeline.py      ← Background removal, HSV binary segmentation (+ CLAHE shadow fix)
│   ├── mask_engine.py         ← Reference mask loading, alignment, adapt_mask, GrabCut refinement
│   ├── color_engine.py        ← Fiji subprocess call + Python fallback measurements
│   └── exporter.py            ← Excel export
│
├── ui/
│   ├── main_window.py         ← Main window, AnalysisWorker thread, toolbar
│   ├── pipeline_viewer.py     ← 6-step clickable thumbnail strip
│   └── results_table.py       ← 36-row dark-theme results table
│
└── assetes/
    ├── measure_grapes_batch.ijm   ← ImageJ macro (all 36 grapes in one call)
    └── reference_masks.npz        ← Pre-computed reference masks (gitignored — not in repo)
```

> **`reference_masks.npz` is not included in the repository** (gitignored). It must be generated from your own reference image using the mask generation step in the pipeline, or provided separately.

---

## Troubleshooting

**App starts but Fiji badge is amber (fallback mode)**
→ Fiji was not found at the default path. Click Settings on the toolbar and point to your Fiji executable.

**"Permission denied" when running `./run.sh`**
```bash
chmod +x run.sh
```

**`python3: command not found`**
→ Install Python 3.10+ (see Step 2 above). On macOS after Homebrew install, try `python3.12` explicitly.

**`pip install` fails with SSL error**
```bash
pip install --trusted-host pypi.org --trusted-host files.pythonhosted.org -r requirements.txt
```

**Fiji takes a long time on first analysis**
→ Normal — Fiji starts a JVM on first run. Subsequent analyses of the same session are faster. The app has a 5-minute timeout.

**Not all 36 grapes detected**
→ The reference masks (`reference_masks.npz`) may not be present. Ensure this file is in `assetes/` before running.

**Windows: Fiji path not found**
→ Open `config.py` and set `IMAGEJ_PATH` to your Fiji executable path, e.g. `C:/Fiji.app/fiji-windows-x64.exe`. Or use the Settings dialog in-app.

---

## Dependencies

Installed automatically by `run.sh` / `pip install -r requirements.txt`:

```
PySide6        — GUI framework
numpy          — Array operations
opencv-python  — Image processing, GrabCut, watershed
Pillow         — Image I/O
scipy          — Morphological operations
openpyxl       — Excel export
```

No Java installation required — Fiji bundles its own JVM.

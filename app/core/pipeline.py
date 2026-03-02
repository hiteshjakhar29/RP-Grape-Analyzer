"""
pipeline.py - Grape segmentation pipeline (Classical + ML hybrid)
Detects exactly 36 individual grape regions from an image.
"""

import cv2
import numpy as np
import warnings
from skimage import measure, segmentation
from skimage.feature import peak_local_max
from scipy import ndimage as ndi
from pathlib import Path


# ─────────────────────────────────────────────
#  Tunable constants
# ─────────────────────────────────────────────

MIN_GRAPE_SIZE_PX = 15000    # Minimum region area (px²) kept after watershed
WATERSHED_MIN_DISTANCE = 50  # Minimum pixel distance between watershed seeds

# Model singletons — loaded once on first call, reused across images
_dino_processor = None
_dino_model = None
_sam2_predictor = None


# ─────────────────────────────────────────────
#  CLASSICAL SEGMENTATION (mirrors ImageJ macro)
# ─────────────────────────────────────────────

def classical_segmentation(image_bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict, int]:
    """
    Replicates the ImageJ macro threshold logic exactly:
      - RGB threshold: Blue channel < 136
      - Fill holes, open, watershed, analyze particles
    Returns: (labeled_mask, binary_mask, debug_images_dict, n_grapes)
             binary_mask is the cleaned binary used by the caller for fallback watershed.
    """
    debug = {}

    # Work in RGB
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    debug["original"] = image_rgb.copy()

    # ── Step 1: Background removal (white background normalization) ──
    # Detect the white/light background and normalize
    normalized = _normalize_white_background(image_rgb)
    debug["normalized"] = normalized.copy()

    # ── Step 2: Color threshold (Blue < 136, same as ImageJ macro) ──
    # Only the blue channel is actually constrained; R/G span the full uint8 range
    B = normalized[:, :, 2]
    binary = (B <= 136).astype(np.uint8) * 255
    debug["binary_raw"] = binary.copy()

    # ── Step 3: Remove grid lines (blue lines on white background) ──
    binary = _remove_grid_lines(binary, image_rgb)
    debug["grid_removed"] = binary.copy()

    # ── Step 4: Morphological cleanup ──
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=4)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)

    # Fill holes
    binary_bool = binary > 0
    binary_bool = ndi.binary_fill_holes(binary_bool)
    binary = (binary_bool.astype(np.uint8)) * 255
    debug["mask_clean"] = binary.copy()

    # ── Step 5: Watershed to split touching grapes ──
    labeled = _watershed_split(binary)
    debug["labeled"] = labeled.copy()

    # ── Step 6: Filter by size (remove tiny noise, keep grape-sized regions) ──
    labeled = _filter_by_size(labeled, min_size=MIN_GRAPE_SIZE_PX)

    # ── Step 7: Re-split any remaining merged blobs ──
    n_after_filter = len(np.unique(labeled)) - 1
    if n_after_filter > 0:
        fg_area = int((labeled > 0).sum())
        expected_area = fg_area / n_after_filter
        labeled = _resplit_large_regions(labeled, expected_area)

    n_grapes = len(np.unique(labeled)) - 1  # exclude background (0)

    # Create color overlay for visualization
    debug["overlay"] = _create_overlay(image_rgb, labeled)

    return labeled, binary, debug, n_grapes


def _normalize_white_background(image_rgb: np.ndarray) -> np.ndarray:
    """
    Normalize image using white background as reference.
    Finds background pixels (high brightness) and scales channels.
    """
    # Convert to grayscale to find background
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)

    # Background = very bright pixels (white paper)
    bg_mask = gray > 200
    if bg_mask.sum() < 1000:
        return image_rgb  # not enough background pixels, skip normalization

    # Compute background mean per channel
    bg_r = image_rgb[:, :, 0][bg_mask].mean()
    bg_g = image_rgb[:, :, 1][bg_mask].mean()
    bg_b = image_rgb[:, :, 2][bg_mask].mean()

    # Target: neutral white (220, 220, 220)
    target = 220.0
    scale_r = target / max(bg_r, 1)
    scale_g = target / max(bg_g, 1)
    scale_b = target / max(bg_b, 1)

    normalized = image_rgb.astype(np.float32)
    normalized[:, :, 0] = np.clip(normalized[:, :, 0] * scale_r, 0, 255)
    normalized[:, :, 1] = np.clip(normalized[:, :, 1] * scale_g, 0, 255)
    normalized[:, :, 2] = np.clip(normalized[:, :, 2] * scale_b, 0, 255)

    return normalized.astype(np.uint8)


def _remove_grid_lines(binary: np.ndarray, image_rgb: np.ndarray) -> np.ndarray:
    """
    Remove blue grid lines from the binary mask.
    Blue lines: high B, low R/G relative to B.
    """
    R, G, B = image_rgb[:, :, 0], image_rgb[:, :, 1], image_rgb[:, :, 2]

    # Blue line pixels: B is dominant and moderately high
    blue_line_mask = (B.astype(int) - R.astype(int) > 20) & \
                     (B.astype(int) - G.astype(int) > 10) & \
                     (B > 80)

    # Dilate slightly to catch edge effects
    kernel = np.ones((3, 3), np.uint8)
    blue_line_mask = cv2.dilate(blue_line_mask.astype(np.uint8), kernel, iterations=2)

    # Remove blue line pixels from binary mask
    binary = binary.copy()
    binary[blue_line_mask > 0] = 0

    return binary


def _watershed_split(binary: np.ndarray,
                     min_distance: int = WATERSHED_MIN_DISTANCE) -> np.ndarray:
    """
    Apply distance transform + watershed to split touching grapes.
    min_distance: minimum pixel separation between watershed seeds.
    """
    binary_bool = binary > 0

    # Distance transform
    distance = ndi.distance_transform_edt(binary_bool)

    # Find local maxima (grape centers)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        coords = peak_local_max(distance, min_distance=min_distance, labels=binary_bool)
    mask_peaks = np.zeros(distance.shape, dtype=bool)
    mask_peaks[tuple(coords.T)] = True
    markers, _ = ndi.label(mask_peaks)

    # Watershed
    labeled = segmentation.watershed(-distance, markers, mask=binary_bool)

    return labeled


def _filter_by_size(labeled: np.ndarray, min_size: int = 3000) -> np.ndarray:
    """
    Remove regions smaller than min_size pixels (noise, text, grid artifacts).
    Uses np.bincount for O(n_pixels) efficiency instead of per-region scanning.
    """
    counts = np.bincount(labeled.ravel())
    # Build lookup table: keep labels large enough, map everything else to 0
    keep = counts >= min_size
    keep[0] = False  # always zero out background
    lut = np.where(keep, np.arange(len(counts)), 0)
    result = lut[labeled]
    return measure.label(result > 0)


def _resplit_large_regions(labeled: np.ndarray, expected_area: float,
                            max_ratio: float = 1.7) -> np.ndarray:
    """
    Re-run watershed on any region whose area is more than max_ratio × expected_area.
    Uses a min_distance tuned to the expected grape radius so one seed per grape is
    detected even in deeply merged blobs.
    """
    # Radius of a typical grape; use 45% of it as min_distance so two touching
    # grape centres (~2×radius apart) each get their own seed.
    single_radius = max(20, int(np.sqrt(expected_area / np.pi) * 0.45))

    result = labeled.copy()
    next_label = int(labeled.max()) + 1

    for prop in measure.regionprops(labeled):
        if prop.area <= expected_area * max_ratio:
            continue  # normal-sized region, leave it

        submask = (labeled == prop.label).astype(np.uint8) * 255
        binary_bool = submask > 0
        distance = ndi.distance_transform_edt(binary_bool)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            coords = peak_local_max(distance, min_distance=single_radius,
                                    labels=binary_bool)

        if len(coords) < 2:
            continue  # watershed can't split it further — leave as-is

        mask_peaks = np.zeros(distance.shape, dtype=bool)
        mask_peaks[tuple(coords.T)] = True
        markers, _ = ndi.label(mask_peaks)
        sub_labeled = segmentation.watershed(-distance, markers, mask=binary_bool)

        # Replace the original region with the newly split sub-regions
        result[labeled == prop.label] = 0
        for sid in np.unique(sub_labeled):
            if sid == 0:
                continue
            result[sub_labeled == sid] = next_label
            next_label += 1

    return measure.label(result > 0)


def _create_overlay(image_rgb: np.ndarray, labeled: np.ndarray) -> np.ndarray:
    """
    Create a color overlay showing grape boundaries and IDs.
    """
    overlay = image_rgb.copy()
    props = measure.regionprops(labeled)  # called once for all regions

    for region in props:
        region_id = region.label
        mask = (labeled == region_id).astype(np.uint8)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, contours, -1, (0, 255, 0), 2)

        cy, cx = region.centroid
        cv2.putText(overlay, str(region_id),
                    (int(cx) - 8, int(cy) + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    return overlay


# ─────────────────────────────────────────────
#  ML SEGMENTATION (Grounding DINO + SAM2)
# ─────────────────────────────────────────────

def grounded_sam2_segmentation(
        image_bgr: np.ndarray,
        progress_fn=None,
) -> tuple[np.ndarray | None, int]:
    """
    Detect and segment grapes using Grounding DINO (text-prompted detection)
    followed by SAM2 (per-box precise mask refinement).

    Flow:
      1. Load DINO + SAM2 (once; cached in app/models/).
      2. Detect grape bounding boxes with the text prompt "grape."
      3. Retry with box_threshold=0.20 if count != 36.
      4. Refine each box into a pixel mask with SAM2.
      5. Fall back to classical_segmentation if DINO finds 0 boxes or
         count is still != 36 after the retry.

    Returns (labeled_mask, count) or (None, 0) on unrecoverable failure.
    """
    global _dino_processor, _dino_model, _sam2_predictor

    def log(msg: str) -> None:
        print(msg)
        if progress_fn:
            progress_fn(msg)

    models_dir = Path(__file__).parent.parent / "models"
    models_dir.mkdir(exist_ok=True)

    # ── Import guard ──
    try:
        import torch
        from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
        from sam2.sam2_image_predictor import SAM2ImagePredictor
        from PIL import Image as PILImage
    except ImportError as e:
        log(f"[ML] Grounding DINO / SAM2 not installed: {e}")
        return None, 0

    device = (
        "mps" if torch.backends.mps.is_available() else
        "cuda" if torch.cuda.is_available() else
        "cpu"
    )
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    h, w = image_bgr.shape[:2]

    # ── Load Grounding DINO (once) ──
    if _dino_processor is None or _dino_model is None:
        log("Loading Grounding DINO...")
        try:
            dino_cache = str(models_dir / "grounding-dino-base")
            _dino_processor = AutoProcessor.from_pretrained(
                "IDEA-Research/grounding-dino-base", cache_dir=dino_cache
            )
            _dino_model = AutoModelForZeroShotObjectDetection.from_pretrained(
                "IDEA-Research/grounding-dino-base", cache_dir=dino_cache
            ).to(device)
            _dino_model.eval()
        except Exception as e:
            log(f"[ML] Grounding DINO load failed: {e}")
            return None, 0

    # ── Load SAM2 (once) ──
    if _sam2_predictor is None:
        log("Loading SAM2...")
        try:
            sam2_cache = str(models_dir / "sam2-hiera-small")
            _sam2_predictor = SAM2ImagePredictor.from_pretrained(
                "facebook/sam2-hiera-small", cache_dir=sam2_cache
            )
        except Exception as e:
            log(f"[ML] SAM2 load failed: {e}")
            return None, 0

    # ── DINO detection helper ──
    def _detect_boxes(box_thr: float, text_thr: float) -> np.ndarray:
        """Returns (N, 4) xyxy boxes in pixel coords, or empty array on error."""
        try:
            pil_image = PILImage.fromarray(image_rgb)
            inputs = _dino_processor(
                images=pil_image, text="grape.", return_tensors="pt"
            ).to(device)
            with torch.no_grad():
                outputs = _dino_model(**inputs)
            results = _dino_processor.post_process_grounded_object_detection(
                outputs,
                inputs.input_ids,
                box_threshold=box_thr,
                text_threshold=text_thr,
                target_sizes=[(h, w)],
            )
            return results[0]["boxes"].cpu().numpy()  # (N, 4)
        except Exception as e:
            log(f"[ML] DINO detection error: {e}")
            return np.zeros((0, 4), dtype=np.float32)

    # ── Attempt 1: standard thresholds ──
    log("Detecting grapes with text prompt...")
    boxes = _detect_boxes(box_thr=0.25, text_thr=0.25)
    log(f"Found {len(boxes)} grapes")

    # ── Attempt 2: lower thresholds if count is wrong ──
    if len(boxes) != 36:
        log(f"Retrying with lower threshold (got {len(boxes)}, expected 36)...")
        boxes = _detect_boxes(box_thr=0.20, text_thr=0.20)
        log(f"Found {len(boxes)} grapes")

    # ── Fall back to classical if DINO still can't reach 36 ──
    if len(boxes) == 0 or len(boxes) != 36:
        log(f"[ML] DINO found {len(boxes)} grapes — falling back to classical segmentation")
        classical_labeled, _, _, classical_n = classical_segmentation(image_bgr)
        return classical_labeled, classical_n

    # ── Refine each detected box with SAM2 ──
    log("Refining masks with SAM2...")
    try:
        import torch
        with torch.inference_mode():
            _sam2_predictor.set_image(image_rgb)
            labeled = np.zeros((h, w), dtype=np.int32)
            for i, box in enumerate(boxes):
                masks, scores, _ = _sam2_predictor.predict(
                    box=box,
                    multimask_output=False,
                )
                # masks shape: (1, H, W) when multimask_output=False
                best_mask = masks[0].astype(bool)
                labeled[best_mask & (labeled == 0)] = i + 1

        n = int(labeled.max())
        log(f"SAM2 refined {n} grape masks")
        return labeled, n

    except Exception as e:
        log(f"[ML] SAM2 prediction failed: {e}")
        return None, 0


# ─────────────────────────────────────────────
#  GRID-BASED SEGMENTATION (primary method)
# ─────────────────────────────────────────────

_RATIO_LABELS = ["100:0", "80:20", "60:40", "40:60", "20:80", "0:100"]


def _merge_close_lines(positions: list, gap: int = 25) -> list:
    """Merge line positions within `gap` pixels of each other (take mean)."""
    if not positions:
        return []
    positions = sorted(positions)
    merged = []
    cluster = [positions[0]]
    for pos in positions[1:]:
        if pos - cluster[-1] <= gap:
            cluster.append(pos)
        else:
            merged.append(float(np.mean(cluster)))
            cluster = [pos]
    merged.append(float(np.mean(cluster)))
    return merged


def _detect_grid_lines(
        image_rgb: np.ndarray,
) -> tuple[list, list, np.ndarray, str]:
    """
    Detect blue grid lines via HSV colour filter + HoughLinesP.
    Returns (h_lines, v_lines, blue_mask, method_str).

    h_lines — 7 y-positions bounding the 6 grape rows.
              Lines in the top-8% header (Coated/Control text) are excluded.
              Synthetic top/bottom bounds are added when the outer frame is
              not detected by HoughLines.

    v_lines — [center_divider_x]: the single vertical line separating the
              Coated (left) and Control (right) grape columns.
              The leftmost 25% (ruler + label column) is excluded; the line
              closest to x=55% of image width is kept.

    Falls back to equal-division grid if HoughLines finds too few lines.
    """
    h, w = image_rgb.shape[:2]
    hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)

    # H 90-130, S 40-255, V 40-255
    blue_mask = cv2.inRange(hsv,
                             np.array([90, 40, 40]),
                             np.array([130, 255, 255]))
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    blue_mask = cv2.morphologyEx(blue_mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    lines = cv2.HoughLinesP(
        blue_mask, 1, np.pi / 180,
        threshold=80,
        minLineLength=int(w * 0.3),
        maxLineGap=30,
    )

    h_pos: list = []
    v_pos: list = []
    if lines is not None:
        for seg in lines:
            x1, y1, x2, y2 = seg[0]
            angle = abs(float(np.degrees(np.arctan2(y2 - y1, x2 - x1))))
            if angle < 20:
                y_mid = (y1 + y2) / 2.0
                # Exclude header row (top 8%) — contains "Coated"/"Control" text
                if y_mid > h * 0.08:
                    h_pos.append(y_mid)
            elif angle > 70:
                x_mid = (x1 + x2) / 2.0
                # Exclude ruler + label column (leftmost 25% of image)
                if x_mid > w * 0.25:
                    v_pos.append(x_mid)

    h_lines = sorted(_merge_close_lines(h_pos, gap=25))
    v_lines_all = sorted(_merge_close_lines(v_pos, gap=25))

    # Keep only the single vertical line closest to the expected centre divider
    target_v = w * 0.55
    if v_lines_all:
        center_v = min(v_lines_all, key=lambda x: abs(x - target_v))
    else:
        center_v = target_v  # fallback estimate
    v_lines = [center_v]

    # Ensure h_lines brackets the full grape area (add synthetic bounds if absent)
    grape_top    = h * 0.08
    grape_bottom = h * 0.98
    if not h_lines or h_lines[0] > grape_top + h * 0.05:
        h_lines.insert(0, grape_top)
    if not h_lines or h_lines[-1] < grape_bottom - h * 0.05:
        h_lines.append(grape_bottom)

    if len(h_lines) >= 5:
        return h_lines, v_lines, blue_mask, "HoughLines"

    # Fallback: hardcoded proportions based on known image layout
    # Ruler+label col = leftmost 25%; grape cols = 25%-97%; center divider = 61%
    header_bottom = int(h * 0.09)
    footer_top    = int(h * 0.97)
    row_height    = (footer_top - header_bottom) // 6
    h_lines = [float(header_bottom + i * row_height) for i in range(7)]
    v_lines = [float(w * 0.61)]
    return h_lines, v_lines, blue_mask, "Fallback equal division"


def _compute_cells(
        h_lines: list,
        v_lines: list,
        image_shape: tuple,
) -> list:
    """
    Compute 12 cell bounding boxes (6 rows × 2 columns).

    Column layout left → right:
      [ruler ~12%] [label col ~13%] [Coated grapes] [divider] [Control grapes]

    grape_area_left  = image_width × 0.25  (skips ruler + label column)
    grape_area_right = image_width × 0.97  (skips right edge)
    center_divider   = v_lines[0]          (detected or estimated at 55%)

    Returns list of (x1, y1, x2, y2) in row-major, Coated-first order.
    8 px inward padding on every side to avoid grid-line pixels.
    """
    h, w = image_shape[:2]
    PAD = 8

    grape_area_left  = int(w * 0.25)
    grape_area_right = int(w * 0.97)
    center_divider   = int(v_lines[0]) if v_lines else int(w * 0.55)

    h_bounds = sorted(h_lines)
    while len(h_bounds) < 7:
        h_bounds.append(float(h * 0.98))
    h_bounds = h_bounds[:7]

    cells = []
    for row_i in range(6):
        top    = int(h_bounds[row_i])
        bottom = int(h_bounds[row_i + 1])
        # Coated (left column)
        cells.append((
            max(0, grape_area_left + PAD),
            max(0, top + PAD),
            min(w, center_divider - PAD),
            min(h, bottom - PAD),
        ))
        # Control (right column)
        cells.append((
            max(0, center_divider + PAD),
            max(0, top + PAD),
            min(w, grape_area_right - PAD),
            min(h, bottom - PAD),
        ))
    return cells


def save_cell_debug_images(
        image_rgb: np.ndarray,
        cells: list,
        output_dir: str = None,
) -> None:
    """
    Save per-cell debug images so the cell crop, blue mask, B<=136 threshold,
    and Otsu threshold can be inspected to diagnose detection failures.

    Saved under <output_dir>/debug_cells/ as:
        cell_NN_crop.png       — raw RGB crop
        cell_NN_blue_mask.png  — HSV blue-line detection
        cell_NN_thresh_136.png — B <= 136 binary
        cell_NN_thresh_otsu.png — inverted Otsu binary
    """
    if output_dir is None:
        output_dir = str(Path.home() / "grape_analyzer_outputs")
    out = Path(output_dir) / "debug_cells"
    out.mkdir(parents=True, exist_ok=True)

    for i, (x1, y1, x2, y2) in enumerate(cells):
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(image_rgb.shape[1], x2), min(image_rgb.shape[0], y2)
        if x2 <= x1 or y2 <= y1:
            continue

        crop = image_rgb[y1:y2, x1:x2].copy()

        # Raw crop (RGB → BGR for cv2.imwrite)
        cv2.imwrite(str(out / f"cell_{i+1:02d}_crop.png"),
                    cv2.cvtColor(crop, cv2.COLOR_RGB2BGR))

        # Blue line mask (HSV)
        hsv = cv2.cvtColor(crop, cv2.COLOR_RGB2HSV)
        blue_mask = cv2.inRange(hsv,
                                np.array([90, 40, 40]),
                                np.array([130, 255, 255]))
        cv2.imwrite(str(out / f"cell_{i+1:02d}_blue_mask.png"), blue_mask)

        # B <= 136 threshold
        B = crop[:, :, 2]
        thresh_136 = ((B <= 136).astype(np.uint8)) * 255
        cv2.imwrite(str(out / f"cell_{i+1:02d}_thresh_136.png"), thresh_136)

        # Otsu inverted threshold on grayscale
        gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
        _, thresh_otsu = cv2.threshold(gray, 0, 255,
                                       cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        cv2.imwrite(str(out / f"cell_{i+1:02d}_thresh_otsu.png"), thresh_otsu)

    print(f"[Debug] Cell debug images saved to: {out}")


def _find_grapes_in_cell(
        image_rgb: np.ndarray,
        cell_bbox: tuple,
        cell_idx: int = 0,
        progress_fn=None,
) -> list:
    """
    Segment exactly 3 grapes inside a cell bounding box.
    Returns a list of full-image boolean masks (up to 3).

    Strategy:
      1. LAB (L < 220, a < 128) + Otsu combined → morphological cleanup.
      2. Dynamic min area = max(5000, total_foreground/3 × 0.4).
      3. Split any merged blob (area > 1.5× expected) via num_peaks force-split.
      4. Force-split entire mask with num_peaks=3 when still < 3 grapes.
      5. Wide LAB (L < 230, a < 135) re-attempt when ≤ 1 grape found.
      6. HSV (H=10-80) last-resort fallback when all above fail.
      7. Keep the 3 largest valid regions.
    """
    def log(msg: str) -> None:
        print(msg)
        if progress_fn:
            progress_fn(msg)

    x1, y1, x2, y2 = cell_bbox
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(image_rgb.shape[1], x2), min(image_rgb.shape[0], y2)

    if x2 <= x1 or y2 <= y1:
        return []

    crop = image_rgb[y1:y2, x1:x2].copy()
    crop_h, crop_w = crop.shape[:2]

    # Log cell dimensions so the user can verify coordinates
    log(f"  Cell {cell_idx}: x1={x1} y1={y1} x2={x2} y2={y2} "
        f"w={x2 - x1} h={y2 - y1}")

    # Absolute minimum grape area — a real grape is ~55 000 px in a full image;
    # we allow down to 15 000 px to catch partially visible grapes at edges.
    MIN_GRAPE_AREA = 15000

    def _morphology(binary: np.ndarray) -> np.ndarray:
        k9 = np.ones((9, 9), np.uint8)
        k5 = np.ones((5, 5), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, k9, iterations=3)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN,  k5, iterations=2)
        return ndi.binary_fill_holes(binary).astype(np.uint8)

    def _label_valid(binary: np.ndarray) -> tuple:
        lbl = measure.label(binary)
        props = measure.regionprops(lbl)
        valid = sorted([p for p in props if p.area >= MIN_GRAPE_AREA],
                       key=lambda p: p.area, reverse=True)
        return valid, lbl

    # ── Primary detection: LAB + Otsu combined ──
    # Method 1: LAB colour space — grapes are darker (L < 220) and greener (a < 128)
    lab = cv2.cvtColor(crop, cv2.COLOR_RGB2LAB)
    L_ch, a_ch, _ = cv2.split(lab)
    lab_mask = ((L_ch < 220) & (a_ch < 128)).astype(np.uint8) * 255

    # Method 2: Otsu inverted on grayscale
    gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
    _, otsu_raw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Combine: pixel is grape if either method detects it
    combined = cv2.bitwise_or(lab_mask, otsu_raw)
    combined_clean = _morphology(combined)

    # Dynamic expected area: total foreground / 3 grapes per cell
    total_fg = int(combined_clean.sum())
    expected_area = (total_fg / 3) if total_fg > 0 else MIN_GRAPE_AREA
    dyn_min_area  = max(5000, int(expected_area * 0.4))

    valid, labeled_crop = _label_valid(combined_clean)
    log(f"    LAB+Otsu: {len(valid)} grapes "
        f"(expected≈{int(expected_area)} px, dyn_min={dyn_min_area} px)")

    # ── Helper: force-split a binary blob into n_grapes using num_peaks ──
    def _split_merged(bin_mask, n_grapes):
        """Split elongated/touching grapes by forcing exactly n_grapes peaks."""
        dist = ndi.distance_transform_edt(bin_mask > 0)
        grape_w = max(1, crop_w // (n_grapes + 1))
        min_d = max(10, grape_w // 3)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            coords = peak_local_max(dist,
                                    min_distance=min_d,
                                    num_peaks=n_grapes,
                                    labels=(bin_mask > 0))
        if len(coords) == 0:
            return None
        mkrs = np.zeros(bin_mask.shape, dtype=np.int32)
        for k, (r, c) in enumerate(coords):
            mkrs[r, c] = k + 1
        return segmentation.watershed(-dist, mkrs, mask=(bin_mask > 0))

    # ── Split merged blobs (area > 1.5× expected_area) ──
    if valid and expected_area > 0:
        new_lbl  = np.zeros_like(labeled_crop)
        next_id  = 1
        did_split = False
        for prop in valid:
            if prop.area > expected_area * 1.5:
                n_to_split = max(2, round(prop.area / expected_area))
                sub_bin    = (labeled_crop == prop.label).astype(np.uint8)
                sub_split  = _split_merged(sub_bin, n_to_split)
                if sub_split is not None:
                    for sid in np.unique(sub_split):
                        if sid == 0:
                            continue
                        frag = sub_split == sid
                        if frag.sum() >= dyn_min_area:
                            new_lbl[frag] = next_id
                            next_id += 1
                    did_split = True
                    continue   # don't re-add the original merged blob
            new_lbl[labeled_crop == prop.label] = next_id
            next_id += 1
        if did_split:
            valid, labeled_crop = _label_valid(new_lbl > 0)
            log(f"    After merge-split: {len(valid)} grapes")

    # ── Force-split with num_peaks=3 when still < 3 grapes ──
    if len(valid) < 3 and combined_clean.sum() > 0:
        forced = _split_merged(combined_clean, 3)
        if forced is not None:
            forced_props = measure.regionprops(forced)
            forced_valid = sorted([p for p in forced_props if p.area >= dyn_min_area],
                                  key=lambda p: p.area, reverse=True)
            log(f"    Force-split (num_peaks=3): {len(forced_valid)} grapes")
            if len(forced_valid) > len(valid):
                valid, labeled_crop = forced_valid, forced

    # ── Wide LAB fallback when ≤ 1 grape (dehydrated / bottom-row grapes) ──
    if len(valid) <= 1:
        lab_wide = ((L_ch < 230) & (a_ch < 135)).astype(np.uint8) * 255
        combined_wide = cv2.bitwise_or(lab_wide, otsu_raw)
        clean_wide    = _morphology(combined_wide)
        wide_valid, wide_lbl = _label_valid(clean_wide)
        if len(wide_valid) < 3 and clean_wide.sum() > 0:
            forced_wide = _split_merged(clean_wide, 3)
            if forced_wide is not None:
                dyn_min_wide = max(5000, int(clean_wide.sum() / 3 * 0.4))
                fw_props  = measure.regionprops(forced_wide)
                fw_valid  = sorted([p for p in fw_props if p.area >= dyn_min_wide],
                                   key=lambda p: p.area, reverse=True)
                if len(fw_valid) > len(wide_valid):
                    wide_valid, wide_lbl = fw_valid, forced_wide
        if len(wide_valid) > len(valid):
            valid, labeled_crop = wide_valid, wide_lbl
            log(f"    Wide LAB (L<230, a<135): {len(valid)} grapes")

    # ── HSV yellowish-green fallback when still 0 grapes ──
    if len(valid) == 0:
        hsv = cv2.cvtColor(crop, cv2.COLOR_RGB2HSV)
        # Grapes span yellow-green → H 10-80, moderate S and V
        hsv_mask = cv2.inRange(hsv,
                               np.array([10, 30, 50]),
                               np.array([80, 255, 255]))
        hsv_valid, hsv_lbl = _label_valid(_morphology(hsv_mask))
        if hsv_valid:
            valid, labeled_crop = hsv_valid, hsv_lbl
            log(f"    HSV fallback: {len(hsv_valid)} grapes")
        else:
            log(f"  ⚠️  Cell {cell_idx}: 0 grapes found — all methods failed")
            return []

    # Map crop-space masks to full-image coordinates (top-3 by area)
    full_masks = []
    for prop in valid[:3]:
        mask_crop = (labeled_crop == prop.label)
        full_mask = np.zeros(image_rgb.shape[:2], dtype=bool)
        full_mask[y1:y2, x1:x2] = mask_crop
        full_masks.append(full_mask)
    return full_masks


def grid_based_segmentation(
        image_rgb: np.ndarray,
        progress_fn=None,
) -> dict:
    """
    Primary segmentation using the fixed 6×2 grid structure.

    Phase 1 — Detect blue grid lines (HoughLinesP; fallback: equal division).
    Phase 2 — Compute 12 cell bounding boxes.
    Phase 3 — Find exactly 3 grapes per cell.
    Phase 4 — Build a single labeled mask (labels 1…n_grapes).

    Returns dict with: labeled_mask, n_grapes, grid_method,
                       cell_grape_counts, cells, debug_images.
    """
    def log(msg: str) -> None:
        print(msg)
        if progress_fn:
            progress_fn(msg)

    h, w = image_rgb.shape[:2]

    # Phase 1
    h_lines, v_lines, blue_mask, grid_method = _detect_grid_lines(image_rgb)
    log(f"🔲 Grid detected: {len(h_lines)} horizontal, "
        f"{len(v_lines)} vertical lines ({grid_method})")

    # Phase 2
    cells = _compute_cells(h_lines, v_lines, image_rgb.shape)

    # Debug: grid overlay
    #   Horizontal row dividers  → RED
    #   Centre vertical divider  → BLUE
    #   Left grape-area boundary → GREEN (marks start of Coated column)
    grid_overlay = image_rgb.copy()
    for hy in h_lines:
        cv2.line(grid_overlay, (0, int(hy)), (w, int(hy)), (220, 50, 50), 2)
    grape_left_x = int(w * 0.25)
    cv2.line(grid_overlay, (grape_left_x, 0), (grape_left_x, h), (50, 200, 50), 2)
    for vx in v_lines:
        cv2.line(grid_overlay, (int(vx), 0), (int(vx), h), (50, 100, 220), 2)

    # Debug: cell overlay (green = Coated, blue = Control, labeled)
    cell_overlay = image_rgb.copy()
    for i, (cx1, cy1, cx2, cy2) in enumerate(cells):
        color = (0, 200, 100) if i % 2 == 0 else (50, 100, 220)
        cv2.rectangle(cell_overlay, (cx1, cy1), (cx2, cy2), color, 2)
        cv2.putText(cell_overlay, f"C{i + 1}", (cx1 + 4, cy1 + 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

    # Log all cell coordinates so the user can verify them
    log("📐 Cell coordinates:")
    for ci, (cx1, cy1, cx2, cy2) in enumerate(cells):
        side = "Coated" if ci % 2 == 0 else "Control"
        row = ci // 2
        log(f"  Cell {ci + 1} row{row + 1} {side}: "
            f"x={cx1}-{cx2} y={cy1}-{cy2} "
            f"size={cx2 - cx1}x{cy2 - cy1}")

    # Save per-cell debug images (always — for diagnosis)
    save_cell_debug_images(image_rgb, cells)

    # Phases 3 & 4
    labeled_mask = np.zeros((h, w), dtype=np.int32)
    label_id = 1
    cell_grape_counts: list = []

    for cell_idx, cell_bbox in enumerate(cells):
        row_i = cell_idx // 2
        col_i = cell_idx % 2
        ratio = _RATIO_LABELS[row_i] if row_i < 6 else f"row{row_i}"
        group = "Coated" if col_i == 0 else "Control"

        grape_masks = _find_grapes_in_cell(
            image_rgb, cell_bbox,
            cell_idx=cell_idx + 1,
            progress_fn=progress_fn,
        )
        n_found = len(grape_masks)
        cell_grape_counts.append(n_found)
        log(f"📦 Cell {cell_idx + 1}/12 ({ratio} {group}): {n_found} grapes found")

        for mask in grape_masks:
            labeled_mask[mask & (labeled_mask == 0)] = label_id
            label_id += 1

    n_grapes = label_id - 1

    # Debug: grape masks (random colour per grape)
    rng = np.random.default_rng(42)
    grape_colors = rng.integers(60, 230, size=(max(n_grapes + 1, 2), 3)).astype(np.uint8)
    grape_masks_img = image_rgb.copy()
    for lbl in range(1, n_grapes + 1):
        grape_masks_img[labeled_mask == lbl] = grape_colors[lbl]

    final_overlay = _create_overlay(image_rgb, labeled_mask)

    return {
        "labeled_mask": labeled_mask,
        "n_grapes": n_grapes,
        "grid_method": grid_method,
        "cell_grape_counts": cell_grape_counts,
        "cells": cells,
        "debug_images": {
            "blue_mask": blue_mask,
            "grid_overlay": grid_overlay,
            "cell_overlay": cell_overlay,
            "grape_masks": grape_masks_img,
            "final_overlay": final_overlay,
        },
    }


# ─────────────────────────────────────────────
#  WHITE BALANCE NORMALISATION
# ─────────────────────────────────────────────

def normalize_white_balance(
        image_rgb: np.ndarray,
        labeled_mask: np.ndarray,
) -> tuple:
    """
    Scale each channel so the white board → (255, 255, 255).
    Samples only background pixels (R,G,B > 210) that are not grapes.
    Returns (normalized_image, wb_factors_dict).
    Returns (original_image, {}) if sampling fails.
    """
    R, G, B = image_rgb[:, :, 0], image_rgb[:, :, 1], image_rgb[:, :, 2]
    white_px = (R > 210) & (G > 210) & (B > 210) & (labeled_mask == 0)

    if white_px.sum() < 1000:
        print("[WB] Insufficient white pixels — skipping normalisation")
        return image_rgb, {}

    mean_R = float(R[white_px].mean())
    mean_G = float(G[white_px].mean())
    mean_B = float(B[white_px].mean())

    scale_R = 255.0 / max(mean_R, 1.0)
    scale_G = 255.0 / max(mean_G, 1.0)
    scale_B = 255.0 / max(mean_B, 1.0)

    normalized = image_rgb.astype(np.float32)
    normalized[:, :, 0] = np.clip(normalized[:, :, 0] * scale_R, 0, 255)
    normalized[:, :, 1] = np.clip(normalized[:, :, 1] * scale_G, 0, 255)
    normalized[:, :, 2] = np.clip(normalized[:, :, 2] * scale_B, 0, 255)

    return normalized.astype(np.uint8), {
        "scale_R": round(scale_R, 4),
        "scale_G": round(scale_G, 4),
        "scale_B": round(scale_B, 4),
        "mean_white_R": round(mean_R, 2),
        "mean_white_G": round(mean_G, 2),
        "mean_white_B": round(mean_B, 2),
    }


# ─────────────────────────────────────────────
#  GRABCUT MASK REFINEMENT
# ─────────────────────────────────────────────

def refine_mask_grabcut(
        image_rgb: np.ndarray,
        grape_bbox: tuple,
        rough_mask: np.ndarray,
) -> np.ndarray:
    """
    Refine a grape mask using GrabCut for accurate pixel boundaries.

    grape_bbox: (x1, y1, x2, y2) bounding box of the grape in pixel coords.
    rough_mask: full-image boolean array (True = foreground pixel).

    Returns a refined full-image boolean mask.
    Falls back to rough_mask if GrabCut fails or produces an unreasonable result
    (refined area < 50% or > 200% of rough area).
    """
    h_img, w_img = image_rgb.shape[:2]
    x1, y1, x2, y2 = grape_bbox
    PAD = 15
    px1, py1 = max(0, x1 - PAD), max(0, y1 - PAD)
    px2, py2 = min(w_img, x2 + PAD), min(h_img, y2 + PAD)

    crop = image_rgb[py1:py2, px1:px2].copy()
    ch, cw = crop.shape[:2]
    if ch < 10 or cw < 10:
        return rough_mask

    # GrabCut initialisation mask
    gc_mask = np.full((ch, cw), cv2.GC_PR_BGD, dtype=np.uint8)
    rough_crop = rough_mask[py1:py2, px1:px2]
    gc_mask[rough_crop] = cv2.GC_PR_FGD

    # Definite background at the 5-pixel border
    border = 5
    gc_mask[:border, :] = cv2.GC_BGD
    gc_mask[-border:, :] = cv2.GC_BGD
    gc_mask[:, :border] = cv2.GC_BGD
    gc_mask[:, -border:] = cv2.GC_BGD

    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)

    try:
        cv2.grabCut(crop, gc_mask, None, bgd_model, fgd_model,
                    5, cv2.GC_INIT_WITH_MASK)
    except Exception:
        return rough_mask

    refined_crop = (gc_mask == cv2.GC_FGD) | (gc_mask == cv2.GC_PR_FGD)
    rough_area = int(rough_mask.sum())
    refined_area = int(refined_crop.sum())

    # Sanity check: reject if area changed by more than 30% either way.
    # GrabCut expanding beyond 130% usually means it leaked into background.
    # Shrinking below 70% usually means it stripped too much of the grape.
    if rough_area > 0 and not (0.70 <= refined_area / rough_area <= 1.30):
        return rough_mask

    refined_full = np.zeros(image_rgb.shape[:2], dtype=bool)
    refined_full[py1:py2, px1:px2] = refined_crop
    return refined_full


# ─────────────────────────────────────────────
#  ZONE-BASED SEGMENTATION (36 individual zones)
# ─────────────────────────────────────────────

def compute_zones(image_h: int, image_w: int,
                  image_rgb: np.ndarray = None) -> list:
    """
    Compute exactly 36 grape zones scaled from reference pixel measurements
    taken from a 1810 × 2000 px reference image.

    Reference layout:
      coated_left = 407,  center_div = 1052,  right_edge = 1672
      row_ys = [216, 477, 743, 1015, 1301, 1580, 1818]

    If image_rgb is provided, HoughLines is attempted to refine the center
    divider and row boundaries; falls back to proportional defaults otherwise.

    Returns list of 36 zone dicts:
      {"grape_id", "row", "col", "group", "ratio", "bbox": (x1,y1,x2,y2)}
    """
    sx = image_w / 1810
    sy = image_h / 2000

    # Proportional defaults from reference image
    coated_left = int(407  * sx)
    center_div  = int(1052 * sx)
    right_edge  = int(1672 * sx)
    row_ys      = [int(y * sy) for y in [216, 477, 743, 1015, 1301, 1580, 1818]]

    # Optionally refine via HoughLines
    if image_rgb is not None:
        h_lines, v_lines, _, _ = _detect_grid_lines(image_rgb)
        # Accept detected center divider only if close to expected position (±10%)
        tol = int(image_w * 0.10)
        if v_lines and abs(v_lines[0] - center_div) <= tol:
            center_div = int(v_lines[0])
        # Accept h_lines only if we get at least 7 (= 6 rows + outer bounds)
        if len(h_lines) >= 7:
            row_ys = [int(y) for y in sorted(h_lines)[:7]]

    coated_w  = (center_div - coated_left) // 3
    control_w = (right_edge - center_div)  // 3
    _RATIO    = ["100:0", "80:20", "60:40", "40:60", "20:80", "0:100"]
    PAD       = 8

    zones = []
    for row in range(6):
        y1 = row_ys[row]     + PAD
        y2 = row_ys[row + 1] - PAD

        for col in range(3):     # Coated grapes (left group)
            x1 = coated_left + col       * coated_w + PAD
            x2 = coated_left + (col + 1) * coated_w - PAD
            zones.append({
                "grape_id": row * 6 + col + 1,
                "row": row + 1,
                "col": col + 1,
                "group": "Coated",
                "ratio": _RATIO[row],
                "bbox": (max(0, x1), max(0, y1),
                         min(image_w, x2), min(image_h, y2)),
            })

        for col in range(3):     # Control grapes (right group)
            x1 = center_div + col       * control_w + PAD
            x2 = center_div + (col + 1) * control_w - PAD
            zones.append({
                "grape_id": row * 6 + 3 + col + 1,
                "row": row + 1,
                "col": col + 4,
                "group": "Control",
                "ratio": _RATIO[row],
                "bbox": (max(0, x1), max(0, y1),
                         min(image_w, x2), min(image_h, y2)),
            })

    return zones  # always exactly 36


def detect_single_grape_in_zone(crop_rgb: np.ndarray) -> tuple[np.ndarray, bool]:
    """
    Find exactly 1 grape inside a small crop zone.
    Returns (binary_mask_uint8, is_fallback).

    Grape = any pixel that is NOT white background AND NOT a blue grid line.
    Fallback: filled ellipse at zone centre when no foreground is detected.
    """
    img = crop_rgb.astype(int)

    # White background: all channels > 195
    white = (img[:, :, 0] > 195) & (img[:, :, 1] > 195) & (img[:, :, 2] > 195)

    # Blue grid lines: B channel clearly dominates R and G
    blue = (
        (img[:, :, 2] - img[:, :, 0] > 25) &
        (img[:, :, 2] - img[:, :, 1] > 25) &
        (img[:, :, 2] > 80)
    )

    grape_raw = (~white & ~blue).astype(np.uint8) * 255

    # Morphological cleanup: close gaps, remove noise, fill holes
    k_close = np.ones((13, 13), np.uint8)
    k_open  = np.ones((7,  7),  np.uint8)
    cleaned = cv2.morphologyEx(grape_raw, cv2.MORPH_CLOSE, k_close, iterations=3)
    cleaned = cv2.morphologyEx(cleaned,   cv2.MORPH_OPEN,  k_open,  iterations=2)
    filled  = ndi.binary_fill_holes(cleaned > 0).astype(np.uint8) * 255

    # Keep only the largest connected component = the grape
    n_cc, labels, stats, _ = cv2.connectedComponentsWithStats(
        filled, connectivity=8
    )

    if n_cc <= 1:
        # Nothing found — use ellipse at zone centre as fallback
        result = np.zeros(crop_rgb.shape[:2], np.uint8)
        cy = crop_rgb.shape[0] // 2
        cx = crop_rgb.shape[1] // 2
        ry = crop_rgb.shape[0] // 3
        rx = crop_rgb.shape[1] // 4
        cv2.ellipse(result, (cx, cy), (rx, ry), 0, 0, 360, 255, -1)
        return result, True

    largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    result = (labels == largest).astype(np.uint8) * 255
    return result, False


def _draw_all_masks(image_rgb: np.ndarray,
                    labeled_mask: np.ndarray,
                    zones: list) -> np.ndarray:
    """Colour each of the 36 grape regions with a distinct hue."""
    overlay = image_rgb.copy()
    rng = np.random.default_rng(42)
    for zone in zones:
        gid   = zone["grape_id"]
        color = rng.integers(60, 230, size=3).tolist()
        overlay[labeled_mask == gid] = color
    return overlay


def _draw_final_overlay(image_rgb: np.ndarray,
                        labeled_mask: np.ndarray,
                        zones: list) -> np.ndarray:
    """Draw grape contours + ID labels (Coated=green, Control=cyan)."""
    overlay = image_rgb.copy()
    for zone in zones:
        gid  = zone["grape_id"]
        mask = (labeled_mask == gid).astype(np.uint8)
        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        color = (0, 255, 0) if zone["group"] == "Coated" else (0, 200, 255)
        cv2.drawContours(overlay, contours, -1, color, 2)
        ys, xs = np.where(labeled_mask == gid)
        if len(xs):
            cv2.putText(overlay, str(gid),
                        (int(xs.mean()) - 8, int(ys.mean()) + 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
    return overlay


def _draw_zone_grid(image_rgb: np.ndarray, zones: list) -> np.ndarray:
    """Draw all 36 zone bounding boxes (Coated=green, Control=blue)."""
    overlay = image_rgb.copy()
    for zone in zones:
        x1, y1, x2, y2 = zone["bbox"]
        color = (0, 200, 100) if zone["group"] == "Coated" else (50, 100, 220)
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)
        cv2.putText(overlay, str(zone["grape_id"]), (x1 + 2, y1 + 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
    return overlay


# ─────────────────────────────────────────────
#  HYBRID PIPELINE (main entry point)
# ─────────────────────────────────────────────

def run_hybrid_pipeline(image_path: str, use_ml: bool = False,
                        progress_fn=None) -> dict:
    """
    Main pipeline entry point — Zone-First approach (always exactly 36 grapes).

    Step 1: Load image + white balance normalisation
    Step 2: Compute px/mm scale from ruler
    Step 3: Compute 36 proportional grape zones (HoughLines refinement if available)
    Step 4: Detect one grape per zone → labeled_mask
    Step 5: Measure each grape with Python (ImageJ-equivalent formulas)

    Returns dict with: image_rgb, normalized_image, labeled_mask, n_grapes (=36),
                       method_used, wb_factors, px_per_mm, df_measurements,
                       fallback_ids, debug_images, grabcut_changes, cell_grape_counts,
                       grid_method, classical_n, ml_n, image_path.
    """
    import pandas as pd
    from app.core.measure import measure_grape_python, compute_scale_from_ruler

    def log(msg: str) -> None:
        print(msg)
        if progress_fn:
            progress_fn(msg)

    image_bgr = cv2.imread(str(image_path))
    if image_bgr is None:
        raise ValueError(f"Cannot read image: {image_path}")

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    h, w = image_rgb.shape[:2]
    log(f"📷 Image loaded: {w}×{h} px")

    # ── Step 1: White balance normalisation ──
    log("⚖️  Normalising white balance...")
    tmp_mask = np.zeros((h, w), dtype=np.int32)   # no grapes known yet
    normalized_rgb, wb_factors = normalize_white_balance(image_rgb, tmp_mask)
    if wb_factors:
        log(f"⚖️  WB: R×{wb_factors['scale_R']:.3f}  "
            f"G×{wb_factors['scale_G']:.3f}  B×{wb_factors['scale_B']:.3f}")
    else:
        log("⚖️  WB: skipped (insufficient white pixels)")
        normalized_rgb = image_rgb

    # ── Step 2: Ruler scale ──
    log("📐 Detecting scale from ruler...")
    px_per_mm = compute_scale_from_ruler(image_rgb)
    if px_per_mm:
        log(f"📐 Scale: {px_per_mm:.2f} px/mm")
    else:
        log("📐 Scale not detected — Area in px² only")

    # ── Step 3: Compute 36 zones ──
    log("🔲 Computing 36 grape zones...")
    zones = compute_zones(h, w, image_rgb)
    zone_grid_img = _draw_zone_grid(image_rgb, zones)
    log(f"🔲 36 zones ready "
        f"(sx={w/1810:.3f} sy={h/2000:.3f})")

    # ── Step 4: Detect one grape per zone ──
    log("🍇 Detecting grape in each zone...")
    labeled_mask  = np.zeros((h, w), dtype=np.int32)
    fallback_ids: list = []

    for zone in zones:
        gid             = zone["grape_id"]
        x1, y1, x2, y2 = zone["bbox"]
        crop            = normalized_rgb[y1:y2, x1:x2]

        if crop.size == 0:
            fallback_ids.append(gid)
            log(f"  ⚠  Grape {gid:2d} ({zone['ratio']} {zone['group']}): empty crop")
            continue

        mask_crop, is_fallback = detect_single_grape_in_zone(crop)

        # Write mask into full labeled image (no overlap allowed)
        full_mask = np.zeros((h, w), np.uint8)
        full_mask[y1:y2, x1:x2] = mask_crop
        labeled_mask[full_mask > 0] = gid

        if is_fallback:
            fallback_ids.append(gid)
            log(f"  ⚠  Grape {gid:2d} ({zone['ratio']} {zone['group']}): fallback mask")
        else:
            log(f"  ✅ Grape {gid:2d} ({zone['ratio']} {zone['group']}): detected")

    n_real = 36 - len(fallback_ids)
    log(f"✅ Detection: {n_real}/36 real, {len(fallback_ids)}/36 fallback")

    # ── Step 5: Measure each grape ──
    log("📊 Measuring all 36 grapes (Python / ImageJ-equivalent)...")
    image_name = Path(image_path).name
    rows = []

    for zone in zones:
        gid        = zone["grape_id"]
        grape_mask = (labeled_mask == gid).astype(np.uint8) * 255
        m = measure_grape_python(normalized_rgb, grape_mask, px_per_mm)
        if m is None:
            m = {k: 0 for k in [
                "Area_px2", "Area_mm2", "Centroid_X", "Centroid_Y",
                "Mean_R", "Mean_G", "Mean_B",
                "Mean_L", "Mean_a", "Mean_b",
                "Mean_H", "Mean_S", "Mean_Brightness",
            ]}
        rows.append({
            "Image":              image_name,
            "Grape_ID":           gid,
            "Row":                zone["row"],
            "Col":                zone["col"],
            "Group":              zone["group"],
            "Ratio":              zone["ratio"],
            "Area_px2":           m["Area_px2"],
            "Area_mm2":           m["Area_mm2"],
            "Centroid_X":         m["Centroid_X"],
            "Centroid_Y":         m["Centroid_Y"],
            "Mean_R":             m["Mean_R"],
            "Mean_G":             m["Mean_G"],
            "Mean_B":             m["Mean_B"],
            "Mean_L":             m["Mean_L"],
            "Mean_a":             m["Mean_a"],
            "Mean_b":             m["Mean_b"],
            "Mean_H":             m["Mean_H"],
            "Mean_S":             m["Mean_S"],
            "Mean_Brightness":    m["Mean_Brightness"],
            "Fallback":           gid in fallback_ids,
            "Measurement_Engine": "Python (ImageJ-equivalent)",
        })

    df_measurements = pd.DataFrame(rows)
    log("✅ All 36 grapes measured")

    # ── Build debug images ──
    grape_masks_img = _draw_all_masks(image_rgb, labeled_mask, zones)
    final_overlay   = _draw_final_overlay(image_rgb, labeled_mask, zones)

    return {
        "image_rgb":         image_rgb,
        "normalized_image":  normalized_rgb,
        "labeled_mask":      labeled_mask,
        "n_grapes":          36,
        "method_used":       "Zone-First (36 proportional zones)",
        "wb_factors":        wb_factors,
        "px_per_mm":         px_per_mm,
        "df_measurements":   df_measurements,
        "fallback_ids":      fallback_ids,
        "grabcut_changes":   [],
        "cell_grape_counts": [],
        "grid_method":       "Proportional zones",
        "classical_n":       36,
        "ml_n":              0,
        "debug_images": {
            "original":      image_rgb,
            "normalized":    normalized_rgb,
            "zone_grid":     zone_grid_img,
            "grape_masks":   grape_masks_img,
            "final_overlay": final_overlay,
        },
        "image_path": image_path,
    }

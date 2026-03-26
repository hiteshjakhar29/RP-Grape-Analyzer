"""
image_pipeline.py — All image processing steps.

run_pipeline() takes ONE original image and produces ALL pipeline steps.
"""

import cv2
import numpy as np
from scipy.ndimage import binary_fill_holes


def run_pipeline(original_rgb: np.ndarray) -> dict:
    """
    Input:  original_rgb — raw photo as np.ndarray, shape (H, W, 3), dtype uint8, RGB
    Output: dict with keys:
        'step1_original'    — original image
        'step2_bg_removed'  — grapes on black background
        'step3_binary'      — white grapes on black (matches ImageJ binary)
        'step4_day0_overlay'— Day 0 reference boundaries drawn on original
        'step5_adapted'     — adapted boundaries for this session
        'step6_measured'    — filled measurement regions
    (Steps 4-6 are filled in by mask_engine after loading masks.)
    """
    step1 = original_rgb.copy()
    step2 = generate_bg_removed(original_rgb)
    step3 = generate_binary(step2)

    return {
        "step1_original":    step1,
        "step2_bg_removed":  step2,
        "step3_binary":      step3,
    }


def generate_bg_removed(original_rgb: np.ndarray) -> np.ndarray:
    """
    Step 2: Remove white background and dark plastic wrap edges.
    Output: grapes in original color on pure black background.
    Matches ImageJ's background subtraction output.
    """
    R = original_rgb[:, :, 0].astype(np.float32)
    G = original_rgb[:, :, 1].astype(np.float32)
    B = original_rgb[:, :, 2].astype(np.float32)

    # White paper background: all channels high
    is_white_bg = (R > 185) & (G > 185) & (B > 185)

    # Dark plastic wrap border around tray
    is_dark_wrap = (R < 50) & (G < 50) & (B < 50)

    bg_removed = original_rgb.copy()
    bg_removed[is_white_bg | is_dark_wrap] = [0, 0, 0]
    return bg_removed


def generate_binary(bg_removed: np.ndarray) -> np.ndarray:
    """
    Step 3: Convert bg-removed image to binary mask.
    Grapes = white (255), everything else = black (0).
    Matches ImageJ's Make Binary + Fill Holes output.

    Algorithm:
    1. Detect green/yellow grape pixels by HSV color range
    2. Exclude blue grid lines
    3. Morphological close to bridge small gaps within grapes
    4. Fill holes (captures brown stem tips enclosed by green)
    5. Remove small noise blobs
    """
    hsv = cv2.cvtColor(bg_removed, cv2.COLOR_RGB2HSV)

    # ── Shadow correction (segmentation path only) ───────────────────────────
    # CLAHE on V lifts shadow-covered grape pixels above the V>55 threshold.
    # H and S are untouched so hue-based grape/grid discrimination is unaffected.
    # clipLimit=2.0: conservative boost; tileGridSize=(8,8): ~300x340 px per tile
    # at reference resolution — larger than one grape, keeps local context.
    try:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        hsv[:, :, 2] = clahe.apply(hsv[:, :, 2])
    except Exception:
        pass  # fallback: proceed with unenhanced V (pre-improvement behaviour)

    H = hsv[:, :, 0]  # 0-179 in OpenCV
    S = hsv[:, :, 1]  # 0-255
    V = hsv[:, :, 2]  # 0-255 (CLAHE-enhanced)

    # Grape color: yellow-green hue
    is_grape = (H >= 12) & (H <= 100) & (S > 25) & (V > 55)

    # Exclude blue grid lines
    is_grid = (H > 105) & (H < 140) & (S > 60)
    is_grape = is_grape & ~is_grid

    grape_mask = is_grape.astype(np.uint8) * 255

    k7 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    k5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    grape_mask = cv2.morphologyEx(grape_mask, cv2.MORPH_CLOSE, k7, iterations=3)
    grape_mask = cv2.morphologyEx(grape_mask, cv2.MORPH_OPEN,  k5, iterations=1)

    # Fill holes
    grape_mask = binary_fill_holes(grape_mask > 0).astype(np.uint8) * 255

    # Remove small blobs (noise, ruler marks, grid artifacts)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(grape_mask, connectivity=8)
    clean = np.zeros_like(grape_mask)
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] > 8000:
            clean[labels == i] = 255

    return clean

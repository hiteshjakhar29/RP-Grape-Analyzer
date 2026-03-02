"""
mask_engine.py — Mask loading, image alignment, adaptive mask fitting,
and visualization overlays.
"""

import json
import cv2
import numpy as np
from scipy.ndimage import binary_fill_holes


# ─────────────────────────────────────────────
# Reference resolution constants
# ─────────────────────────────────────────────

# cv2.resize dsize = (width, height) as specified: (2450, 2707)
# Resulting numpy shape = (2707, 2450, C)
_REF_DSIZE  = (2450, 2707)
_REF_PIXELS = 2450 * 2707   # total pixel count at reference resolution


def normalize_to_reference(image: np.ndarray) -> tuple[np.ndarray, tuple[int, int]]:
    """
    Step 1 — Resize session image to reference resolution (2450×2707 cv2 dsize).

    Returns
    -------
    normalized    : np.ndarray  — image at reference resolution
    original_dims : (orig_h, orig_w)  — stored for area correction (step 3)
    """
    orig_h, orig_w = image.shape[:2]
    normalized = cv2.resize(image, _REF_DSIZE, interpolation=cv2.INTER_LINEAR)
    return normalized, (orig_h, orig_w)


def area_to_original_scale(area_ref: int | float, original_dims: tuple[int, int]) -> int:
    """
    Step 3 — Correct pixel area measured at reference resolution back to the
    original image scale.

    area_original = area_reference × (orig_h × orig_w) / (2450 × 2707)

    RGB / Lab / HSB values are color-resolution-independent — no correction needed.
    """
    orig_h, orig_w = original_dims
    return round(area_ref * (orig_h * orig_w) / _REF_PIXELS)


# ─────────────────────────────────────────────
# Loading
# ─────────────────────────────────────────────

def load_reference_masks(masks_path: str, meta_path: str) -> tuple:
    """Load pre-computed reference masks and metadata."""
    masks_np = np.load(masks_path)
    with open(meta_path) as f:
        metadata = json.load(f)
    masks = {int(k): masks_np[f"grape_{k}"] for k in metadata.keys()}
    return masks, metadata


def _scale_mask(mask: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    """
    Resize a binary mask to (target_h, target_w) using nearest-neighbour
    interpolation so pixel values stay 0 / 255.
    Returns the resized mask as uint8.
    """
    mh, mw = mask.shape[:2]
    if mh == target_h and mw == target_w:
        return mask.astype(np.uint8)
    resized = cv2.resize(
        mask.astype(np.uint8), (target_w, target_h),
        interpolation=cv2.INTER_NEAREST,
    )
    return resized


# ─────────────────────────────────────────────
# Alignment
# ─────────────────────────────────────────────

def align_image(new_image: np.ndarray, day0_reference: np.ndarray) -> np.ndarray:
    """
    Align new session image to reference using ORB feature matching.
    Handles small camera shifts (±20-50px typical).
    Falls back to identity transform if matching fails.
    """
    gray_ref = cv2.cvtColor(day0_reference, cv2.COLOR_RGB2GRAY)
    gray_new = cv2.cvtColor(new_image, cv2.COLOR_RGB2GRAY)

    orb = cv2.ORB_create(nfeatures=3000)
    kp1, des1 = orb.detectAndCompute(gray_ref, None)
    kp2, des2 = orb.detectAndCompute(gray_new, None)

    if des1 is None or des2 is None or len(kp1) < 20 or len(kp2) < 20:
        return new_image

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = sorted(bf.match(des1, des2), key=lambda x: x.distance)
    good = matches[: min(200, len(matches))]

    if len(good) < 20:
        return new_image

    pts_ref = np.float32([kp1[m.queryIdx].pt for m in good])
    pts_new = np.float32([kp2[m.trainIdx].pt for m in good])

    H, _ = cv2.findHomography(pts_new, pts_ref, cv2.RANSAC, 5.0)
    if H is None:
        return new_image

    h, w = day0_reference.shape[:2]
    aligned = cv2.warpPerspective(
        new_image, H, (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )
    return aligned


# ─────────────────────────────────────────────
# Adaptive mask fitting
# ─────────────────────────────────────────────

def adapt_mask(
    day0_mask: np.ndarray,
    session_binary: np.ndarray,
    grape_id: int,
    shrink_tol: float = 0.35,
    grow_tol: float   = 0.40,
    search_pad: int   = 50,
) -> np.ndarray:
    """
    Adapt reference mask to current session grape shape.

    The grape can:
    - Shrink up to 35% (after reference day — drying over time)
    - Grow up to 40% (before reference day — larger when fresh)
    - Shift slightly within search_pad pixels (default 50 px at reference scale)

    If the reference mask and session binary have different shapes (e.g. different
    scan resolutions), the reference mask is proportionally scaled to match the
    session binary before comparison.

    Strategy:
    1. Find bounding box of (scaled) reference mask + search_pad
    2. Look for grape blob in that region of session binary
    3. If found and area within tolerance → use session blob shape
    4. If shrank beyond tolerance → intersect session blob with reference mask
    5. If no blob found → apply progressive erosion to reference mask
    """
    # ── Scale reference mask to match session binary if needed ──
    bin_h, bin_w = session_binary.shape[:2]
    day0_mask_scaled = _scale_mask(day0_mask, bin_h, bin_w)

    ys, xs = np.where(day0_mask_scaled > 0)
    if len(xs) == 0:
        return day0_mask_scaled.copy()

    h, w = day0_mask_scaled.shape
    # Scale search_pad proportionally to the size change
    ref_h = day0_mask.shape[0]
    pad_scaled = max(20, int(search_pad * (bin_h / ref_h)))

    x1 = max(0, int(xs.min()) - pad_scaled)
    x2 = min(w, int(xs.max()) + pad_scaled)
    y1 = max(0, int(ys.min()) - pad_scaled)
    y2 = min(h, int(ys.max()) + pad_scaled)

    day0_area = int(np.sum(day0_mask_scaled > 0))
    day0_cx   = float(xs.mean())
    day0_cy   = float(ys.mean())

    # Extract session binary in search window
    search = session_binary[y1:y2, x1:x2].copy()
    search_filled = binary_fill_holes(search > 0).astype(np.uint8) * 255

    n, labels, stats, centroids = cv2.connectedComponentsWithStats(
        search_filled, connectivity=8
    )

    best_label = -1
    best_score = float("inf")
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] < 500:
            continue
        cx = centroids[i][0] + x1
        cy = centroids[i][1] + y1
        dist = np.sqrt((cx - day0_cx) ** 2 + (cy - day0_cy) ** 2)
        if dist < best_score:
            best_score = dist
            best_label = i

    if best_label > 0 and best_score < (pad_scaled * 2):
        session_area = stats[best_label, cv2.CC_STAT_AREA]
        ratio = session_area / day0_area

        if ratio >= (1 - shrink_tol):
            blob        = (labels == best_label).astype(np.uint8) * 255
            blob_filled = binary_fill_holes(blob > 0).astype(np.uint8) * 255

            if ratio > 1.0:
                # Grape is LARGER than reference — check whether the blob was
                # clipped at the search-window edge and expand if so.
                blob_ys, blob_xs = np.where(blob_filled > 0)
                win_h, win_w = search_filled.shape
                touches_edge = (
                    blob_xs.min() == 0 or blob_xs.max() == win_w - 1 or
                    blob_ys.min() == 0 or blob_ys.max() == win_h - 1
                )
                if touches_edge:
                    # Expand window around blob's actual bounding box and re-extract.
                    exp_x1 = max(0, x1 + int(blob_xs.min()) - pad_scaled)
                    exp_x2 = min(w, x1 + int(blob_xs.max()) + pad_scaled)
                    exp_y1 = max(0, y1 + int(blob_ys.min()) - pad_scaled)
                    exp_y2 = min(h, y1 + int(blob_ys.max()) + pad_scaled)

                    wide_search = session_binary[exp_y1:exp_y2, exp_x1:exp_x2].copy()
                    wide_filled = binary_fill_holes(wide_search > 0).astype(np.uint8) * 255
                    nw, lw, sw, cw = cv2.connectedComponentsWithStats(
                        wide_filled, connectivity=8
                    )
                    best_lw, best_dw = -1, float("inf")
                    for j in range(1, nw):
                        if sw[j, cv2.CC_STAT_AREA] < 500:
                            continue
                        cx_w = cw[j][0] + exp_x1
                        cy_w = cw[j][1] + exp_y1
                        d = np.sqrt((cx_w - day0_cx) ** 2 + (cy_w - day0_cy) ** 2)
                        if d < best_dw:
                            best_dw, best_lw = d, j
                    if best_lw > 0:
                        blob_w        = (lw == best_lw).astype(np.uint8) * 255
                        blob_w_filled = binary_fill_holes(blob_w > 0).astype(np.uint8) * 255
                        full = np.zeros((h, w), dtype=np.uint8)
                        full[exp_y1:exp_y2, exp_x1:exp_x2] = blob_w_filled
                        return full
                    # Wide extraction found nothing — fall through to narrow blob below

            # Grape is at/below reference size, or wide extraction failed:
            # use the blob from the initial search window.
            full = np.zeros((h, w), dtype=np.uint8)
            full[y1:y2, x1:x2] = blob_filled
            return full
        else:
            # Much smaller than reference — intersect with scaled reference to suppress noise
            blob = (labels == best_label).astype(np.uint8) * 255
            full_blob = np.zeros((h, w), dtype=np.uint8)
            full_blob[y1:y2, x1:x2] = blob
            intersect = ((full_blob > 0) & (day0_mask_scaled > 0)).astype(np.uint8) * 255
            if np.sum(intersect > 0) > 500:
                return intersect

    # Fallback: erode scaled reference mask
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    eroded = cv2.erode(day0_mask_scaled, k, iterations=3)
    if np.sum(eroded > 0) > 500:
        return eroded
    return day0_mask_scaled


# ─────────────────────────────────────────────
# Visualization
# ─────────────────────────────────────────────

_GROUP_COLORS = {
    "Coated":  (0,   200, 255),   # cyan
    "Control": (255, 165,   0),   # orange
}
_GROUP_FILL_COLORS = {
    "Coated":  (100, 200, 255),
    "Control": (255, 200, 100),
}


def draw_boundaries(
    image_rgb: np.ndarray,
    masks: dict,
    metadata: dict,
    style: str = "solid",
) -> np.ndarray:
    """
    Draw grape boundary contours on a copy of image_rgb.
    style='solid'  — continuous contour line
    style='dashed' — dashed contour line (for reference masks)
    Groups are color-coded: Coated=cyan, Control=orange.
    Each grape is labelled with its ID.
    Masks are auto-scaled to match image dimensions if needed.
    """
    ih, iw = image_rgb.shape[:2]
    overlay = image_rgb.copy()

    for gid_str, meta in metadata.items():
        gid   = int(gid_str)
        mask  = masks.get(gid)
        if mask is None:
            continue
        mask_u8 = _scale_mask(mask, ih, iw)
        group   = meta.get("group", "Coated")
        color   = _GROUP_COLORS.get(group, (255, 255, 255))

        contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if style == "dashed":
            for contour in contours:
                pts = contour.squeeze()
                if pts.ndim < 2 or len(pts) < 2:
                    continue
                n = len(pts)
                DASH = 8
                GAP  = 5
                i = 0
                while i < n:
                    j = min(i + DASH, n)
                    for k in range(i, j - 1):
                        cv2.line(overlay, tuple(pts[k % n]), tuple(pts[(k + 1) % n]), color, 2)
                    i += DASH + GAP
        else:
            cv2.drawContours(overlay, contours, -1, color, 2)

        # Grape ID label at centroid
        ys, xs = np.where(mask_u8 > 0)
        if len(xs) > 0:
            cx, cy = int(xs.mean()), int(ys.mean())
            cv2.putText(
                overlay, str(gid), (cx - 8, cy + 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA,
            )

    return overlay


def draw_filled(
    image_rgb: np.ndarray,
    masks: dict,
    metadata: dict,
) -> np.ndarray:
    """
    Draw semi-transparent filled regions on image_rgb.
    Coated=light-blue, Control=light-orange.
    Each grape is labelled with its ID.
    Masks are auto-scaled to match image dimensions if needed.
    """
    ih, iw = image_rgb.shape[:2]
    base       = image_rgb.copy()
    fill_layer = image_rgb.copy()

    for gid_str, meta in metadata.items():
        gid  = int(gid_str)
        mask = masks.get(gid)
        if mask is None:
            continue
        mask_u8   = _scale_mask(mask, ih, iw)
        bool_mask = mask_u8 > 0
        group     = meta.get("group", "Coated")
        fill_col  = _GROUP_FILL_COLORS.get(group, (200, 200, 200))
        line_col  = _GROUP_COLORS.get(group, (255, 255, 255))

        fill_layer[bool_mask] = fill_col

        contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(fill_layer, contours, -1, line_col, 1)

        ys, xs = np.where(bool_mask)
        if len(xs) > 0:
            cx, cy = int(xs.mean()), int(ys.mean())
            cv2.putText(
                fill_layer, str(gid), (cx - 8, cy + 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA,
            )

    # Blend: 40% fill, 60% original
    result = cv2.addWeighted(fill_layer, 0.4, base, 0.6, 0)
    return result

"""
color_engine.py — Color measurements per grape.

Primary:  headless Fiji/ImageJ subprocess — all 36 grapes in ONE call via
          assetes/measure_grapes_batch.ijm
Fallback: pure-Python formulas that are mathematically identical to ImageJ
          (same CIE D65 Lab pipeline, same Java Color.RGBtoHSB algorithm).
"""

import csv
import os
import platform
import subprocess
import tempfile

import cv2
import numpy as np
from PIL import Image as PILImage


# ── Fiji/ImageJ path management ───────────────────────────────────────────────

_IMAGEJ_PATH: str | None = None   # cached after first successful lookup

_FIJI_CANDIDATES: dict[str, list[str]] = {
    "Darwin": [
        "/Applications/FijiWorking/Fiji.app/Contents/MacOS/fiji-macos-arm64",
    ],
    "Windows": [
        "C:/Fiji.app/fiji-windows-x64.exe",
        "C:/Program Files/Fiji.app/fiji-windows-x64.exe",
        "C:/Fiji.app/ImageJ-win64.exe",
        "C:/Program Files/Fiji.app/ImageJ-win64.exe",
    ],
    "Linux": [
        "/opt/Fiji.app/fiji-linux-x64",
        os.path.expanduser("~/Fiji.app/fiji-linux-x64"),
        "/opt/Fiji.app/ImageJ-linux64",
        os.path.expanduser("~/Fiji.app/ImageJ-linux64"),
    ],
}

_MACRO_PATH = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "assetes", "measure_grapes_batch.ijm")
)


def find_imagej() -> str:
    """Return path to Fiji/ImageJ executable, or raise FileNotFoundError."""
    global _IMAGEJ_PATH
    if _IMAGEJ_PATH and os.path.exists(_IMAGEJ_PATH):
        return _IMAGEJ_PATH
    for path in _FIJI_CANDIDATES.get(platform.system(), []):
        if os.path.exists(path):
            _IMAGEJ_PATH = path
            return path
    raise FileNotFoundError(
        "Fiji/ImageJ not found. Install from https://fiji.sc\n"
        f"Searched: {_FIJI_CANDIDATES.get(platform.system(), [])}"
    )


def set_imagej_path(path: str) -> None:
    """Override the auto-detected ImageJ path (called from UI settings dialog)."""
    global _IMAGEJ_PATH
    _IMAGEJ_PATH = path


# ── Batch measurement — primary API ──────────────────────────────────────────

def measure_all_grapes(
    original_rgb: np.ndarray,
    masks: dict,       # {grape_id (int): mask_array (H×W uint8)}
    metadata: dict,    # {"1": {...}, "2": {...}, ...}  keys are str
) -> list[dict]:
    """
    Measure all grapes with a SINGLE ImageJ subprocess call.

    If Fiji is not installed or the call fails for any reason, silently falls
    back to the built-in Python formulas (values are mathematically identical).

    Returns a list of measurement dicts in the same order as metadata.keys().
    """
    gids = [int(k) for k in metadata.keys()]

    try:
        ij = find_imagej()
    except FileNotFoundError:
        return [_python_measure(original_rgb, masks[g], g) for g in gids if g in masks]

    with tempfile.TemporaryDirectory() as tmpdir:
        # ── Save session image ──────────────────────────────────────────────
        img_path = os.path.join(tmpdir, "session.png")
        PILImage.fromarray(original_rgb).save(img_path)

        # ── Save all masks as  1.png … N.png  ──────────────────────────────
        masks_dir = os.path.join(tmpdir, "masks")
        os.makedirs(masks_dir, exist_ok=True)
        for gid in gids:
            m = masks.get(gid)
            if m is not None:
                PILImage.fromarray(
                    (m > 0).astype(np.uint8) * 255, mode="L"
                ).save(os.path.join(masks_dir, f"{gid}.png"))

        # ── Call ImageJ ─────────────────────────────────────────────────────
        csv_path   = os.path.join(tmpdir, "results.csv")
        n          = max(gids) if gids else 36
        macro_args = f"{img_path}|{masks_dir}|{csv_path}|{n}"

        try:
            proc = subprocess.run(
                [ij, "--headless", "--console",
                 "-macro", os.path.abspath(_MACRO_PATH), macro_args],
                capture_output=True,
                text=True,
                timeout=300,   # 5-minute budget for the full batch
            )
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return _fallback_all(original_rgb, masks, gids)

        if proc.returncode != 0 or not os.path.exists(csv_path):
            return _fallback_all(original_rgb, masks, gids)

        # ── Parse CSV ────────────────────────────────────────────────────────
        by_id: dict[int, dict] = {}
        try:
            with open(csv_path, newline="") as f:
                for row in csv.DictReader(f):
                    gid = int(float(row["grape_id"]))
                    by_id[gid] = {
                        "grape_id": gid,
                        "area_px":  int(float(row["area_px"])),
                        # Mean values — from ImageJ
                        "mean_R":   round(float(row["mean_R"]),  4),
                        "mean_G":   round(float(row["mean_G"]),  4),
                        "mean_B":   round(float(row["mean_B"]),  4),
                        "mean_L":   round(float(row["mean_L"]),  4),
                        "mean_a":   round(float(row["mean_a"]),  4),
                        "mean_b":   round(float(row["mean_b"]),  4),
                        "mean_H":   round(float(row["mean_H"]),  4),
                        "mean_S":   round(float(row["mean_S"]),  4),
                        "mean_Br":  round(float(row["mean_Br"]), 4),
                        # StdDev values — from ImageJ (primary path)
                        "std_R":    round(float(row["std_R"]),   4),
                        "std_G":    round(float(row["std_G"]),   4),
                        "std_B":    round(float(row["std_B"]),   4),
                        "std_L":    round(float(row["std_L"]),   4),
                        "std_a":    round(float(row["std_a"]),   4),
                        "std_b":    round(float(row["std_b"]),   4),
                        "std_H":    round(float(row["std_H"]),   4),
                        "std_S":    round(float(row["std_S"]),   4),
                        "std_Br":   round(float(row["std_Br"]),  4),
                    }
        except Exception:
            return _fallback_all(original_rgb, masks, gids)

        # Merge shape metrics (eccentricity — Python only, no ImageJ equivalent)
        results = []
        for gid in gids:
            if gid in by_id:
                by_id[gid].update(_compute_shape_metrics(masks.get(gid)))
                results.append(by_id[gid])
            elif gid in masks:
                results.append(_python_measure(original_rgb, masks[gid], gid))
        return results


def _fallback_all(original_rgb, masks, gids) -> list[dict]:
    return [_python_measure(original_rgb, masks[g], g) for g in gids if g in masks]


def _compute_shape_metrics(mask: np.ndarray | None) -> dict:
    """
    Compute eccentricity from mask shape using OpenCV fitEllipse.

    Uses the same formula as ImageJ's Fit Ellipse measurement:
        eccentricity = sqrt(1 - (minor_axis / major_axis)^2)
    Range: 0.0 (perfect circle) → 1.0 (line).
    """
    if mask is None or not np.any(mask > 0):
        return {"eccentricity": 0.0}
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return {"eccentricity": 0.0}
    cnt = max(contours, key=cv2.contourArea)
    if len(cnt) < 5:
        return {"eccentricity": 0.0}
    try:
        _, (MA, ma), _ = cv2.fitEllipse(cnt)
        major = max(MA, ma)
        minor = min(MA, ma)
        if major < 1e-6:
            return {"eccentricity": 0.0}
        return {"eccentricity": round(float(np.sqrt(1.0 - (minor / major) ** 2)), 4)}
    except Exception:
        return {"eccentricity": 0.0}


def _compute_stddev(original_rgb: np.ndarray, mask: np.ndarray | None, grape_id: int) -> dict:
    """
    Compute per-channel StdDev for a grape mask using the same color-space
    math as _python_measure.  Fallback only — not used when ImageJ succeeds.
    """
    _empty_std = {
        "std_R": 0.0, "std_G": 0.0, "std_B": 0.0,
        "std_L": 0.0, "std_a": 0.0, "std_b": 0.0,
        "std_H": 0.0, "std_S": 0.0, "std_Br": 0.0,
    }
    if mask is None:
        return _empty_std
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return _empty_std

    pixels = original_rgb[ys, xs].astype(np.float64)
    R, G, B = pixels[:, 0], pixels[:, 1], pixels[:, 2]

    # Lab
    def linearize(c):
        n = c / 255.0
        return np.where(n <= 0.04045, n / 12.92, ((n + 0.055) / 1.055) ** 2.4)

    Rl, Gl, Bl = linearize(R), linearize(G), linearize(B)
    X = Rl * 0.4124564 + Gl * 0.3575761 + Bl * 0.1804375
    Y = Rl * 0.2126729 + Gl * 0.7151522 + Bl * 0.0721750
    Z = Rl * 0.0193339 + Gl * 0.1191920 + Bl * 0.9503041

    def cie_f(t):
        return np.where(t > 0.008856, np.cbrt(t), (903.3 * t + 16.0) / 116.0)

    fx, fy, fz = cie_f(X / 0.95047), cie_f(Y / 1.00000), cie_f(Z / 1.08883)
    L_arr = 116.0 * fy - 16.0
    a_arr = 500.0 * (fx - fy)
    b_arr = 200.0 * (fy - fz)

    # HSB
    Rn, Gn, Bn = R / 255.0, G / 255.0, B / 255.0
    Cmax  = np.maximum(np.maximum(Rn, Gn), Bn)
    Cmin  = np.minimum(np.minimum(Rn, Gn), Bn)
    delta = Cmax - Cmin
    hue   = np.zeros(len(R), dtype=np.float64)
    d     = delta > 1e-10
    mr = d & (Cmax == Rn)
    mg = d & (Cmax == Gn)
    mb = d & (Cmax == Bn)
    hue[mr] = 60.0 * (((Gn[mr] - Bn[mr]) / delta[mr]) % 6.0)
    hue[mg] = 60.0 * ((Bn[mg] - Rn[mg]) / delta[mg] + 2.0)
    hue[mb] = 60.0 * ((Rn[mb] - Gn[mb]) / delta[mb] + 4.0)
    S_arr = np.where(Cmax > 1e-10, delta / Cmax, 0.0)

    return {
        "std_R":  round(float(np.std(R)),     4),
        "std_G":  round(float(np.std(G)),     4),
        "std_B":  round(float(np.std(B)),     4),
        "std_L":  round(float(np.std(L_arr)), 4),
        "std_a":  round(float(np.std(a_arr)), 4),
        "std_b":  round(float(np.std(b_arr)), 4),
        "std_H":  round(float(np.std(hue)),   4),
        "std_S":  round(float(np.std(S_arr)), 4),
        "std_Br": round(float(np.std(Cmax)),  4),
    }


# ── Single-grape shim (backward-compat, used nowhere in the new pipeline) ─────

def measure_grape(original_rgb: np.ndarray, mask: np.ndarray, grape_id: int) -> dict:
    """Measure one grape using Python formulas. Kept for backward compatibility."""
    return _python_measure(original_rgb, mask, grape_id)


# ── Python measurement formulas (ImageJ-equivalent) ───────────────────────────

def _python_measure(original_rgb: np.ndarray, mask: np.ndarray, grape_id: int) -> dict:
    """
    Pure-Python measurement — identical math to ImageJ:
    - Lab: D65 illuminant, IEC 61966-2-1 sRGB linearisation, CIE f()
    - HSB: Java Color.RGBtoHSB algorithm (H in 0-360°, S/Br in 0-1)
    Always samples from original_rgb (aligned raw photo), never from binary.
    """
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return _empty(grape_id)

    pixels = original_rgb[ys, xs].astype(np.float64)
    R = pixels[:, 0]
    G = pixels[:, 1]
    B = pixels[:, 2]

    area_px = int(len(xs))
    mean_R  = float(np.mean(R))
    mean_G  = float(np.mean(G))
    mean_B  = float(np.mean(B))

    # ── CIE L*a*b* ────────────────────────────────────────────────────────────
    def linearize(c):
        n = c / 255.0
        return np.where(n <= 0.04045, n / 12.92, ((n + 0.055) / 1.055) ** 2.4)

    Rl = linearize(R)
    Gl = linearize(G)
    Bl = linearize(B)

    X = Rl * 0.4124564 + Gl * 0.3575761 + Bl * 0.1804375
    Y = Rl * 0.2126729 + Gl * 0.7151522 + Bl * 0.0721750
    Z = Rl * 0.0193339 + Gl * 0.1191920 + Bl * 0.9503041

    def cie_f(t):
        return np.where(t > 0.008856, np.cbrt(t), (903.3 * t + 16.0) / 116.0)

    fx = cie_f(X / 0.95047)
    fy = cie_f(Y / 1.00000)
    fz = cie_f(Z / 1.08883)

    mean_L = float(np.mean(116.0 * fy - 16.0))
    mean_a = float(np.mean(500.0 * (fx - fy)))
    mean_b = float(np.mean(200.0 * (fy - fz)))

    # ── HSB (Java Color.RGBtoHSB equivalent) ─────────────────────────────────
    Rn = R / 255.0
    Gn = G / 255.0
    Bn = B / 255.0

    Cmax  = np.maximum(np.maximum(Rn, Gn), Bn)
    Cmin  = np.minimum(np.minimum(Rn, Gn), Bn)
    delta = Cmax - Cmin

    hue = np.zeros(len(R), dtype=np.float64)
    d   = delta > 1e-10
    mr  = d & (Cmax == Rn)
    mg  = d & (Cmax == Gn)
    mb  = d & (Cmax == Bn)
    hue[mr] = 60.0 * (((Gn[mr] - Bn[mr]) / delta[mr]) % 6.0)
    hue[mg] = 60.0 * ((Bn[mg] - Rn[mg]) / delta[mg] + 2.0)
    hue[mb] = 60.0 * ((Rn[mb] - Gn[mb]) / delta[mb] + 4.0)

    S_arr = np.where(Cmax > 1e-10, delta / Cmax, 0.0)

    result = {
        "grape_id": grape_id,
        "area_px":  area_px,
        "mean_R":   round(mean_R,  4),
        "mean_G":   round(mean_G,  4),
        "mean_B":   round(mean_B,  4),
        "mean_L":   round(mean_L,  4),
        "mean_a":   round(mean_a,  4),
        "mean_b":   round(mean_b,  4),
        "mean_H":   round(float(np.mean(hue)),   4),
        "mean_S":   round(float(np.mean(S_arr)),  4),
        "mean_Br":  round(float(np.mean(Cmax)),   4),
        "std_R":    round(float(np.std(R)),       4),
        "std_G":    round(float(np.std(G)),       4),
        "std_B":    round(float(np.std(B)),       4),
        "std_L":    round(float(np.std(116.0 * fy - 16.0)), 4),
        "std_a":    round(float(np.std(500.0 * (fx - fy))), 4),
        "std_b":    round(float(np.std(200.0 * (fy - fz))), 4),
        "std_H":    round(float(np.std(hue)),     4),
        "std_S":    round(float(np.std(S_arr)),   4),
        "std_Br":   round(float(np.std(Cmax)),    4),
    }
    result.update(_compute_shape_metrics(mask))
    return result


def _empty(grape_id: int) -> dict:
    return {
        "grape_id": grape_id, "area_px": 0,
        "mean_R": 0.0, "mean_G": 0.0, "mean_B": 0.0,
        "mean_L": 0.0, "mean_a": 0.0, "mean_b": 0.0,
        "mean_H": 0.0, "mean_S": 0.0, "mean_Br": 0.0,
        "std_R":  0.0, "std_G":  0.0, "std_B":  0.0,
        "std_L":  0.0, "std_a":  0.0, "std_b":  0.0,
        "std_H":  0.0, "std_S":  0.0, "std_Br": 0.0,
        "eccentricity": 0.0,
    }

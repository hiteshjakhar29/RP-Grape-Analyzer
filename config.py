"""config.py — project-wide paths and constants."""
import os

ROOT_DIR   = os.path.dirname(os.path.abspath(__file__))
ASSETS_DIR = os.path.join(ROOT_DIR, "assetes")          # existing folder name

MASKS_PATH              = os.path.join(ASSETS_DIR, "reference_masks.npz")
META_PATH               = os.path.join(ASSETS_DIR, "reference_mask_metadata.json")
REFERENCE_ORIGINAL_PATH = os.path.join(ASSETS_DIR, "reference_original.jpg")
MACRO_PATH              = os.path.join(ASSETS_DIR, "measure_grapes_batch.ijm")

# Day number of the reference image used to generate the reference masks.
# Grapes earlier than REF_DAY are bigger (grow_tol scales up).
# Grapes later than REF_DAY are more shrunken (shrink_tol scales up).
REF_DAY = 150

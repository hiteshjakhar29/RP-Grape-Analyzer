"""
main_window.py — PySide6 main window for Grape Analyzer.

Layout:
┌────────────────────────────────────────────────────────────────┐
│ TOOLBAR: [📂 Load Image]  Day:[___]  [▶ Run Analysis]  [💾 Export] │
├────────────────────────────────────────────────────────────────┤
│ PIPELINE VIEWER (6 clickable thumbnails)                       │
├────────────────────────────────────────────────────────────────┤
│ RESULTS TABLE (36 rows × 15 columns)                           │
├────────────────────────────────────────────────────────────────┤
│ STATUS BAR  ████████░░  "Processing grape 12/36..."            │
└────────────────────────────────────────────────────────────────┘
"""

import os
import sys
import numpy as np

from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QFileDialog, QLabel, QSpinBox,
    QProgressBar, QStatusBar, QDialog, QScrollArea,
    QMessageBox,
)
from PySide6.QtCore import Qt, QThread, Signal, QTimer
from PySide6.QtGui import QPixmap, QImage


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Grape Analyzer — 300 Day Study")
        self.setMinimumSize(1400, 900)
        self._current_image_path: str | None = None
        self._results: list = []
        self._pipeline_steps: dict = {}
        self._worker: "AnalysisWorker | None" = None
        self._setup_ui()
        # Defer ImageJ detection until after the event loop starts so the main
        # window is fully visible before any dialog is shown.
        QTimer.singleShot(200, self._setup_imagej)

    # ─────────────────────────────── UI Setup ────────────────────────────

    def _setup_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        layout.setSpacing(6)

        # ── Toolbar ──
        toolbar = QHBoxLayout()
        toolbar.setSpacing(8)

        self.btn_load = QPushButton("📂 Load Session Image")
        self.btn_load.clicked.connect(self._load_image)

        self.spin_day = QSpinBox()
        self.spin_day.setRange(0, 300)
        self.spin_day.setPrefix("Day: ")
        self.spin_day.setFixedWidth(90)

        self.btn_run = QPushButton("▶ Run Analysis")
        self.btn_run.clicked.connect(self._run_analysis)
        self.btn_run.setEnabled(False)

        self.btn_export = QPushButton("💾 Export Excel")
        self.btn_export.clicked.connect(self._export)
        self.btn_export.setEnabled(False)

        self.lbl_image = QLabel("No image loaded")
        self.lbl_image.setStyleSheet("color: #aaa; font-style: italic;")

        toolbar.addWidget(self.btn_load)
        toolbar.addWidget(self.lbl_image)
        toolbar.addStretch()
        toolbar.addWidget(self.spin_day)
        toolbar.addWidget(self.btn_run)
        toolbar.addWidget(self.btn_export)
        layout.addLayout(toolbar)

        # ── Pipeline Viewer ──
        from ui.pipeline_viewer import PipelineViewer
        self.pipeline_viewer = PipelineViewer()
        self.pipeline_viewer.setFixedHeight(185)
        self.pipeline_viewer.step_clicked.connect(self._show_full_step)
        layout.addWidget(self.pipeline_viewer)

        # ── Results Table ──
        from ui.results_table import ResultsTable
        self.results_table = ResultsTable()
        layout.addWidget(self.results_table)

        # ── Progress bar ──
        self.progress = QProgressBar()
        self.progress.setVisible(False)
        self.progress.setTextVisible(True)
        layout.addWidget(self.progress)

        # ── Status bar ──
        self.status = QStatusBar()
        self.setStatusBar(self.status)
        self.status.showMessage("Ready — load a session image to begin.")

    def _setup_imagej(self):
        """Auto-detect Fiji/ImageJ; prompt the user to browse if not found."""
        try:
            from core.color_engine import find_imagej
            path = find_imagej()
            self.status.showMessage(
                f"ImageJ ready: {os.path.basename(path)}"
                " — measurements will use Fiji"
            )
        except FileNotFoundError:
            box = QMessageBox(self)
            box.setWindowTitle("Fiji/ImageJ Not Found")
            box.setText(
                "Fiji/ImageJ was not found automatically.\n\n"
                "Install from https://fiji.sc  (free, one click)\n"
                "or locate the existing executable manually.\n\n"
                "Without Fiji the app will use built-in Python formulas\n"
                "(mathematically identical results)."
            )
            browse_btn = box.addButton("Browse for Fiji…", QMessageBox.AcceptRole)
            box.addButton("Use Python Formulas", QMessageBox.RejectRole)
            box.exec()
            if box.clickedButton() is browse_btn:
                path, _ = QFileDialog.getOpenFileName(
                    self, "Locate Fiji / ImageJ Executable"
                )
                if path:
                    from core.color_engine import set_imagej_path
                    set_imagej_path(path)
                    self.status.showMessage(
                        f"ImageJ set: {os.path.basename(path)}"
                    )
                    return
            self.status.showMessage(
                "Using Python formulas (Fiji not configured)"
            )

    # ─────────────────────────────── Actions ─────────────────────────────

    def _load_image(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Load Session Image", "",
            "Images (*.jpg *.jpeg *.png *.tif *.tiff)",
        )
        if not path:
            return
        self._current_image_path = path
        name = os.path.basename(path)
        self.lbl_image.setText(name)
        self.btn_run.setEnabled(True)
        self.status.showMessage(f"Loaded: {name}")

    def _run_analysis(self):
        if not self._current_image_path:
            return
        self.btn_run.setEnabled(False)
        self.btn_export.setEnabled(False)
        self.progress.setVisible(True)
        self.progress.setValue(0)

        self._worker = AnalysisWorker(
            self._current_image_path,
            self.spin_day.value(),
        )
        self._worker.progress.connect(self._on_progress)
        self._worker.finished.connect(self._on_finished)
        self._worker.error.connect(self._on_error)
        self._worker.start()

    def _on_progress(self, pct: int, msg: str):
        self.progress.setValue(pct)
        self.status.showMessage(msg)

    def _on_finished(self, pipeline_data: dict):
        self.btn_run.setEnabled(True)
        self.btn_export.setEnabled(True)
        self.progress.setVisible(False)

        # Update pipeline viewer
        step_keys = [
            "step1_original", "step2_bg_removed", "step3_binary",
            "step4_day0",     "step5_session",    "step6_measured",
        ]
        self._pipeline_steps = pipeline_data.get("steps", {})
        for i, key in enumerate(step_keys):
            img = self._pipeline_steps.get(key)
            if img is not None:
                self.pipeline_viewer.update_step(i, img)

        # Update table
        self._results = pipeline_data.get("results", [])
        self.results_table.populate(self._results)

        n = len(self._results)
        self.status.showMessage(
            f"Analysis complete — {n} grapes measured (Day {self.spin_day.value()})"
        )

    def _on_error(self, msg: str):
        self.btn_run.setEnabled(True)
        self.progress.setVisible(False)
        QMessageBox.critical(self, "Analysis Error", msg)
        self.status.showMessage("Error — see dialog for details.")

    def _show_full_step(self, step_idx: int):
        """Open a scrollable full-size view of the selected pipeline step."""
        step_keys = [
            "step1_original", "step2_bg_removed", "step3_binary",
            "step4_day0",     "step5_session",    "step6_measured",
        ]
        if not self._pipeline_steps:
            return
        key = step_keys[step_idx] if step_idx < len(step_keys) else None
        img = self._pipeline_steps.get(key) if key else None
        if img is None:
            return

        dlg = _FullImageDialog(img, parent=self)
        dlg.exec()

    def _export(self):
        if not self._results:
            return
        day  = self.spin_day.value()
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Excel",
            f"grape_analysis_day{day}.xlsx",
            "Excel (*.xlsx)",
        )
        if not path:
            return
        try:
            from core.exporter import export_to_excel
            export_to_excel(self._results, day, path)
            self.status.showMessage(f"Saved: {path}")
        except Exception as e:
            QMessageBox.critical(self, "Export Error", str(e))


# ─────────────────────────────── Full-image dialog ───────────────────────────

class _FullImageDialog(QDialog):
    """Scrollable full-size image viewer."""

    def __init__(self, image_rgb: np.ndarray, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Pipeline Step — Full Size")
        self.resize(900, 700)

        h, w = image_rgb.shape[:2]
        img  = np.ascontiguousarray(image_rgb, dtype=np.uint8)
        qimg = QImage(img.data, w, h, 3 * w, QImage.Format_RGB888)
        pix  = QPixmap.fromImage(qimg)

        lbl = QLabel()
        lbl.setPixmap(pix)
        lbl.setAlignment(Qt.AlignCenter)

        scroll = QScrollArea()
        scroll.setWidget(lbl)
        scroll.setWidgetResizable(False)

        layout = QVBoxLayout(self)
        layout.addWidget(scroll)

        btn_close = QPushButton("Close")
        btn_close.clicked.connect(self.accept)
        layout.addWidget(btn_close)


# ─────────────────────────────── Background worker ───────────────────────────

class AnalysisWorker(QThread):
    progress = Signal(int, str)
    finished = Signal(dict)
    error    = Signal(str)

    def __init__(self, image_path: str, day: int):
        super().__init__()
        self.image_path = image_path
        self.day        = day

    def run(self):
        try:
            import os
            import numpy as np
            from PIL import Image as PILImage

            from config import MASKS_PATH, META_PATH, REFERENCE_ORIGINAL_PATH
            from core.image_pipeline import generate_bg_removed, generate_binary
            from core.mask_engine import (
                load_reference_masks, align_image, adapt_mask,
                draw_boundaries, draw_filled,
                normalize_to_reference, area_to_original_scale,
            )
            from core.color_engine import measure_all_grapes

            # ── 1. Load → normalize → align ──────────────────────────────
            self.progress.emit(5, "Loading image…")
            raw_rgb = np.array(PILImage.open(self.image_path).convert("RGB"))

            # Step 1: resize to reference resolution before anything else
            original_rgb, original_dims = normalize_to_reference(raw_rgb)

            self.progress.emit(10, "Aligning to reference…")
            if os.path.exists(REFERENCE_ORIGINAL_PATH):
                day0_ref = np.array(PILImage.open(REFERENCE_ORIGINAL_PATH).convert("RGB"))
                aligned  = align_image(original_rgb, day0_ref)
            else:
                aligned  = original_rgb

            # ── 2. Background removal ────────────────────────────────────
            self.progress.emit(18, "Removing background…")
            bg_removed = generate_bg_removed(aligned)

            # ── 3. Binary segmentation ───────────────────────────────────
            self.progress.emit(26, "Generating binary segmentation…")
            binary = generate_binary(bg_removed)

            # ── 4. Load reference masks ──────────────────────────────────
            self.progress.emit(35, "Loading reference masks…")
            reference_masks, metadata = load_reference_masks(MASKS_PATH, META_PATH)

            # ── 5a. Adapt all masks ───────────────────────────────────────
            session_masks = {}
            n             = len(metadata)
            for idx, (gid_str, _) in enumerate(metadata.items()):
                gid = int(gid_str)
                pct = 35 + int(30 * idx / n)
                self.progress.emit(pct, f"Adapting mask {gid}/{n}…")
                session_masks[gid] = adapt_mask(reference_masks[gid], binary, gid)

            # ── 5b. Measure all grapes — single ImageJ call ───────────────
            self.progress.emit(65, "Measuring grapes (ImageJ)…")
            measurements = measure_all_grapes(aligned, session_masks, metadata)
            measurements_by_id = {m["grape_id"]: m for m in measurements}

            # ── 5c. Attach metadata + Step 3 area correction ─────────────
            results = []
            for gid_str, meta in metadata.items():
                gid  = int(gid_str)
                meas = dict(measurements_by_id.get(gid, {}))
                # Correct area from reference resolution back to original scale
                if "area_px" in meas:
                    meas["area_px"] = area_to_original_scale(meas["area_px"], original_dims)
                meas.update({
                    "group":     meta.get("group", ""),
                    "ratio":     meta.get("ratio", ""),
                    "grape_row": meta.get("row", 0),
                    "day":       self.day,
                })
                results.append(meas)

            # ── 6. Build overlay images ──────────────────────────────────
            self.progress.emit(88, "Building visualizations…")
            day0_ov = draw_boundaries(aligned, reference_masks, metadata, style="dashed")
            sess_ov = draw_boundaries(aligned, session_masks,   metadata, style="solid")
            meas_ov = draw_filled(aligned, session_masks, metadata)

            self.progress.emit(98, "Done!")
            self.finished.emit({
                "steps": {
                    "step1_original":   aligned,
                    "step2_bg_removed": bg_removed,
                    "step3_binary":     np.stack([binary] * 3, axis=-1),
                    "step4_day0":       day0_ov,
                    "step5_session":    sess_ov,
                    "step6_measured":   meas_ov,
                },
                "results": results,
            })

        except Exception:
            import traceback
            self.error.emit(traceback.format_exc())

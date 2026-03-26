"""
main_window.py — PySide6 main window for Grape Analyzer.

Layout:
┌─────────────────────────────────────────────────────────────────┐
│ HEADER BAR: app title + Fiji status badge                       │
├─────────────────────────────────────────────────────────────────┤
│ TOOLBAR: [Load Image]  <filename>  |  Day:[___]  [Run]  [Export]│
├─────────────────────────────────────────────────────────────────┤
│ PIPELINE VIEWER (6 clickable thumbnails)                        │
├─────────────────────────────────────────────────────────────────┤
│ RESULTS TABLE  ─or─  EMPTY STATE hint                           │
├─────────────────────────────────────────────────────────────────┤
│ PROGRESS BAR (hidden unless running)                            │
├─────────────────────────────────────────────────────────────────┤
│ STATUS BAR                                                      │
└─────────────────────────────────────────────────────────────────┘
"""

import os
import sys
import numpy as np

from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QFileDialog, QLabel, QSpinBox,
    QProgressBar, QStatusBar, QDialog, QScrollArea,
    QMessageBox, QFrame, QStackedWidget,
)
from PySide6.QtCore import Qt, QThread, Signal, QTimer
from PySide6.QtGui import QPixmap, QImage, QFont


# ─────────────────────────────── Colour palette ──────────────────────────────

_C = {
    "bg":          "#1E1E2E",   # main background
    "surface":     "#2A2A3C",   # cards / toolbar
    "border":      "#3A3A52",   # subtle dividers
    "text":        "#CDD6F4",   # primary text
    "subtext":     "#6C7086",   # muted labels
    "accent_blue": "#89B4FA",   # Load button
    "blue_dark":   "#1E66F5",
    "blue_hover":  "#1558D6",
    "accent_green":"#A6E3A1",   # Run button
    "green_dark":  "#40A02B",
    "green_hover": "#368A24",
    "accent_amber":"#FAB387",   # Export button
    "amber_dark":  "#FE640B",
    "amber_hover": "#E55A08",
    "disabled_bg": "#313244",
    "disabled_txt":"#585B70",
    "progress_fg": "#89DCEB",
    "fiji_ok":     "#A6E3A1",
    "fiji_warn":   "#FAB387",
}


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
        QTimer.singleShot(300, self._setup_imagej)

    # ─────────────────────────────── UI Setup ────────────────────────────────

    def _setup_ui(self):
        self.setStyleSheet(f"QMainWindow, QWidget {{ background: {_C['bg']}; color: {_C['text']}; }}")

        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setSpacing(0)
        root.setContentsMargins(0, 0, 0, 0)

        # ── Header bar ────────────────────────────────────────────────────────
        header = QWidget()
        header.setFixedHeight(48)
        header.setStyleSheet(f"""
            QWidget {{
                background: {_C['surface']};
                border-bottom: 1px solid {_C['border']};
            }}
        """)
        h_layout = QHBoxLayout(header)
        h_layout.setContentsMargins(16, 0, 16, 0)

        title_lbl = QLabel("Grape Analyzer")
        title_lbl.setStyleSheet(f"""
            color: {_C['text']};
            font-size: 15px;
            font-weight: bold;
            background: transparent;
            border: none;
        """)

        self.fiji_badge = QLabel("Fiji: checking…")
        self.fiji_badge.setStyleSheet(f"""
            color: {_C['subtext']};
            font-size: 11px;
            background: transparent;
            border: none;
        """)

        h_layout.addWidget(title_lbl)
        h_layout.addStretch()
        h_layout.addWidget(self.fiji_badge)
        root.addWidget(header)

        # ── Toolbar ───────────────────────────────────────────────────────────
        toolbar_frame = QWidget()
        toolbar_frame.setFixedHeight(56)
        toolbar_frame.setStyleSheet(f"""
            QWidget {{
                background: {_C['surface']};
                border-bottom: 1px solid {_C['border']};
            }}
        """)
        toolbar = QHBoxLayout(toolbar_frame)
        toolbar.setContentsMargins(12, 8, 12, 8)
        toolbar.setSpacing(10)

        self.btn_load = QPushButton("  Load Image")
        self.btn_load.setFixedHeight(36)
        self.btn_load.setStyleSheet(self._btn_style(_C['blue_dark'], _C['blue_hover']))
        self.btn_load.clicked.connect(self._load_image)

        self.lbl_image = QLabel("No image selected")
        self.lbl_image.setStyleSheet(f"color: {_C['subtext']}; font-size: 12px; background: transparent; border: none;")

        # vertical divider
        div = QFrame()
        div.setFrameShape(QFrame.VLine)
        div.setStyleSheet(f"color: {_C['border']}; background: {_C['border']};")
        div.setFixedWidth(1)

        lbl_day = QLabel("Day")
        lbl_day.setStyleSheet(f"color: {_C['subtext']}; font-size: 12px; background: transparent; border: none;")

        self.spin_day = QSpinBox()
        self.spin_day.setRange(0, 300)
        self.spin_day.setFixedSize(72, 34)
        self.spin_day.setStyleSheet(f"""
            QSpinBox {{
                background: {_C['bg']};
                color: {_C['text']};
                border: 1px solid {_C['border']};
                border-radius: 6px;
                padding: 2px 6px;
                font-size: 13px;
            }}
            QSpinBox::up-button, QSpinBox::down-button {{
                background: {_C['border']};
                border-radius: 3px;
                width: 16px;
            }}
        """)

        self.btn_run = QPushButton("  Run Analysis")
        self.btn_run.setFixedHeight(36)
        self.btn_run.setStyleSheet(self._btn_style(_C['green_dark'], _C['green_hover'], disabled=True))
        self.btn_run.setEnabled(False)
        self.btn_run.clicked.connect(self._run_analysis)

        self.btn_export = QPushButton("  Export Excel")
        self.btn_export.setFixedHeight(36)
        self.btn_export.setStyleSheet(self._btn_style(_C['amber_dark'], _C['amber_hover'], disabled=True))
        self.btn_export.setEnabled(False)
        self.btn_export.clicked.connect(self._export)

        toolbar.addWidget(self.btn_load)
        toolbar.addWidget(self.lbl_image)
        toolbar.addStretch()
        toolbar.addWidget(div)
        toolbar.addSpacing(4)
        toolbar.addWidget(lbl_day)
        toolbar.addWidget(self.spin_day)
        toolbar.addSpacing(4)
        toolbar.addWidget(self.btn_run)
        toolbar.addWidget(self.btn_export)
        root.addWidget(toolbar_frame)

        # ── Content area ──────────────────────────────────────────────────────
        content = QWidget()
        content_layout = QVBoxLayout(content)
        content_layout.setContentsMargins(12, 10, 12, 6)
        content_layout.setSpacing(8)

        # Pipeline Viewer
        from ui.pipeline_viewer import PipelineViewer
        self.pipeline_viewer = PipelineViewer()
        self.pipeline_viewer.setFixedHeight(190)
        self.pipeline_viewer.step_clicked.connect(self._show_full_step)
        content_layout.addWidget(self.pipeline_viewer)

        # Stacked: empty state vs results table
        self.stack = QStackedWidget()

        # -- Empty state panel --
        empty_panel = QWidget()
        empty_panel.setStyleSheet(f"""
            QWidget {{
                background: {_C['surface']};
                border: 1px solid {_C['border']};
                border-radius: 10px;
            }}
        """)
        ep_layout = QVBoxLayout(empty_panel)
        ep_layout.setAlignment(Qt.AlignCenter)

        icon_lbl = QLabel("🍇")
        icon_lbl.setAlignment(Qt.AlignCenter)
        icon_lbl.setStyleSheet("font-size: 48px; background: transparent; border: none;")

        heading = QLabel("No analysis loaded")
        heading.setAlignment(Qt.AlignCenter)
        heading.setStyleSheet(f"color: {_C['text']}; font-size: 18px; font-weight: bold; background: transparent; border: none;")

        steps_lbl = QLabel(
            "1.  Click  Load Image  and select a session photograph\n"
            "2.  Set the study Day number (0 – 300)\n"
            "3.  Click  Run Analysis  —  results will appear here"
        )
        steps_lbl.setAlignment(Qt.AlignCenter)
        steps_lbl.setStyleSheet(f"color: {_C['subtext']}; font-size: 13px; line-height: 1.8; background: transparent; border: none;")

        ep_layout.addWidget(icon_lbl)
        ep_layout.addSpacing(8)
        ep_layout.addWidget(heading)
        ep_layout.addSpacing(10)
        ep_layout.addWidget(steps_lbl)

        # -- Results table --
        from ui.results_table import ResultsTable
        self.results_table = ResultsTable()

        self.stack.addWidget(empty_panel)   # index 0
        self.stack.addWidget(self.results_table)  # index 1
        self.stack.setCurrentIndex(0)
        content_layout.addWidget(self.stack)

        # Progress bar
        self.progress = QProgressBar()
        self.progress.setVisible(False)
        self.progress.setFixedHeight(6)
        self.progress.setTextVisible(False)
        self.progress.setStyleSheet(f"""
            QProgressBar {{
                background: {_C['surface']};
                border-radius: 3px;
                border: none;
            }}
            QProgressBar::chunk {{
                background: {_C['progress_fg']};
                border-radius: 3px;
            }}
        """)
        content_layout.addWidget(self.progress)

        root.addWidget(content)

        # Status bar
        self.status = QStatusBar()
        self.status.setStyleSheet(f"""
            QStatusBar {{
                background: {_C['surface']};
                color: {_C['subtext']};
                border-top: 1px solid {_C['border']};
                font-size: 12px;
                padding: 2px 10px;
            }}
        """)
        self.setStatusBar(self.status)
        self.status.showMessage("Ready — load a session image to begin.")

    @staticmethod
    def _btn_style(bg: str, hover: str, disabled: bool = False) -> str:
        dis = f"""
            QPushButton:disabled {{
                background: {_C['disabled_bg']};
                color: {_C['disabled_txt']};
                border: none;
            }}
        """ if disabled else ""
        return f"""
            QPushButton {{
                background: {bg};
                color: white;
                border: none;
                border-radius: 7px;
                padding: 0 16px;
                font-size: 13px;
                font-weight: 600;
            }}
            QPushButton:hover {{ background: {hover}; }}
            QPushButton:pressed {{ background: {hover}; opacity: 0.85; }}
            {dis}
        """

    # ─────────────────────────────── Fiji detection ──────────────────────────

    def _setup_imagej(self):
        """Auto-detect Fiji/ImageJ; update badge and optionally prompt user."""
        try:
            from core.color_engine import find_imagej
            path = find_imagej()
            self.fiji_badge.setText(f"Fiji: {os.path.basename(os.path.dirname(path))}")
            self.fiji_badge.setStyleSheet(f"color: {_C['fiji_ok']}; font-size: 11px; background: transparent; border: none;")
            self.status.showMessage(f"Fiji ready — measurements will use ImageJ  ({os.path.basename(path)})")
        except FileNotFoundError:
            self.fiji_badge.setText("Fiji: not found")
            self.fiji_badge.setStyleSheet(f"color: {_C['fiji_warn']}; font-size: 11px; background: transparent; border: none;")

            box = QMessageBox(self)
            box.setWindowTitle("Fiji / ImageJ Not Found")
            box.setText(
                "Fiji was not found automatically.\n\n"
                "Install from  fiji.sc  (free, one click)\n"
                "or locate the existing executable manually.\n\n"
                "Without Fiji the app uses built-in Python formulas\n"
                "(mathematically equivalent results)."
            )
            browse_btn = box.addButton("Browse for Fiji…", QMessageBox.AcceptRole)
            box.addButton("Use Python Formulas", QMessageBox.RejectRole)
            box.exec()
            if box.clickedButton() is browse_btn:
                path, _ = QFileDialog.getOpenFileName(self, "Locate Fiji / ImageJ Executable")
                if path:
                    from core.color_engine import set_imagej_path
                    set_imagej_path(path)
                    self.fiji_badge.setText(f"Fiji: {os.path.basename(path)}")
                    self.fiji_badge.setStyleSheet(f"color: {_C['fiji_ok']}; font-size: 11px; background: transparent; border: none;")
                    self.status.showMessage(f"Fiji set: {os.path.basename(path)}")
                    return
            self.status.showMessage("Using Python formulas — Fiji not configured")

    # ─────────────────────────────── Actions ─────────────────────────────────

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
        self.lbl_image.setStyleSheet(f"color: {_C['text']}; font-size: 12px; background: transparent; border: none;")
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

        step_keys = [
            "step1_original", "step2_bg_removed", "step3_binary",
            "step4_day0",     "step5_session",    "step6_measured",
        ]
        self._pipeline_steps = pipeline_data.get("steps", {})
        for i, key in enumerate(step_keys):
            img = self._pipeline_steps.get(key)
            if img is not None:
                self.pipeline_viewer.update_step(i, img)

        self._results = pipeline_data.get("results", [])
        self.results_table.populate(self._results)
        self.stack.setCurrentIndex(1)   # show table, hide empty state

        n = len(self._results)
        day = self.spin_day.value()
        self.setWindowTitle(f"Grape Analyzer — Day {day}  ({n} grapes)")
        self.status.showMessage(f"Analysis complete — {n} grapes measured   Day {day}")

    def _on_error(self, msg: str):
        self.btn_run.setEnabled(True)
        self.progress.setVisible(False)
        QMessageBox.critical(self, "Analysis Error", msg)
        self.status.showMessage("Error — see dialog for details.")

    def _show_full_step(self, step_idx: int):
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
        day = self.spin_day.value()
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
            self.status.showMessage(f"Saved: {os.path.basename(path)}")
        except Exception as e:
            QMessageBox.critical(self, "Export Error", str(e))


# ─────────────────────────────── Full-image dialog ───────────────────────────

class _FullImageDialog(QDialog):
    """Scrollable full-size image viewer."""

    def __init__(self, image_rgb: np.ndarray, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Pipeline Step — Full Size")
        self.resize(960, 720)
        self.setStyleSheet(f"background: {_C['bg']}; color: {_C['text']};")

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
        scroll.setStyleSheet(f"border: 1px solid {_C['border']}; border-radius: 6px;")

        btn_close = QPushButton("Close")
        btn_close.setFixedHeight(34)
        btn_close.setStyleSheet(MainWindow._btn_style(
            _C['blue_dark'], _C['blue_hover']
        ))
        btn_close.clicked.connect(self.accept)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(8)
        layout.addWidget(scroll)
        layout.addWidget(btn_close, alignment=Qt.AlignRight)


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

            from config import MASKS_PATH, META_PATH, REFERENCE_ORIGINAL_PATH, REF_DAY
            from core.image_pipeline import generate_bg_removed, generate_binary
            from core.mask_engine import (
                load_reference_masks, align_image, adapt_mask,
                refine_mask_grabcut, draw_boundaries, draw_filled,
                normalize_to_reference, area_to_original_scale,
            )
            from core.color_engine import measure_all_grapes

            # ── 1. Load → normalize → align ──────────────────────────────
            self.progress.emit(5, "Loading image…")
            raw_rgb = np.array(PILImage.open(self.image_path).convert("RGB"))
            original_rgb, original_dims = normalize_to_reference(raw_rgb)

            self.progress.emit(10, "Aligning to reference…")
            if os.path.exists(REFERENCE_ORIGINAL_PATH):
                day0_ref = np.array(PILImage.open(REFERENCE_ORIGINAL_PATH).convert("RGB"))
                aligned  = align_image(original_rgb, day0_ref)
            else:
                aligned  = original_rgb

            # ── 2. Background removal ─────────────────────────────────────
            self.progress.emit(18, "Removing background…")
            bg_removed = generate_bg_removed(aligned)

            # ── 3. Binary segmentation ────────────────────────────────────
            self.progress.emit(26, "Generating binary mask…")
            binary = generate_binary(bg_removed)

            # ── 4. Load reference masks ───────────────────────────────────
            self.progress.emit(35, "Loading reference masks…")
            reference_masks, metadata = load_reference_masks(MASKS_PATH, META_PATH)

            # ── 5a. Adapt + refine all masks ──────────────────────────────
            session_masks = {}
            n = len(metadata)
            for idx, (gid_str, _) in enumerate(metadata.items()):
                gid = int(gid_str)
                pct = 35 + int(40 * idx / n)
                self.progress.emit(pct, f"Fitting mask {gid} / {n}…")
                raw_mask = adapt_mask(
                    reference_masks[gid], binary, gid,
                    day=self.day, ref_day=REF_DAY,
                )
                session_masks[gid] = refine_mask_grabcut(aligned, raw_mask)

            # ── 5b. Measure all grapes ────────────────────────────────────
            self.progress.emit(78, "Measuring with ImageJ…")
            measurements = measure_all_grapes(aligned, session_masks, metadata)
            measurements_by_id = {m["grape_id"]: m for m in measurements}

            # ── 5c. Attach metadata + area correction ─────────────────────
            results = []
            for gid_str, meta in metadata.items():
                gid  = int(gid_str)
                meas = dict(measurements_by_id.get(gid, {}))
                if "area_px" in meas:
                    meas["area_px"] = area_to_original_scale(meas["area_px"], original_dims)
                meas.update({
                    "group":     meta.get("group", ""),
                    "ratio":     meta.get("ratio", ""),
                    "grape_row": meta.get("row", 0),
                    "day":       self.day,
                })
                results.append(meas)

            # ── 6. Build overlay images ───────────────────────────────────
            self.progress.emit(90, "Building visualizations…")
            day0_ov = draw_boundaries(aligned, reference_masks, metadata, style="dashed")
            sess_ov = draw_boundaries(aligned, session_masks,   metadata, style="solid")
            meas_ov = draw_filled(aligned, session_masks, metadata)

            self.progress.emit(99, "Done.")
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

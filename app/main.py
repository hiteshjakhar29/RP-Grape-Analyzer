"""
main.py - Grape Analyzer GUI
Upload an image → see intermediate steps → get CSV/Excel with per-grape measurements
"""

import sys
import os
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QFileDialog, QTabWidget, QTableWidget,
    QTableWidgetItem, QProgressBar, QScrollArea,
    QTextEdit, QGroupBox, QMessageBox, QHeaderView
)
from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtGui import QPixmap, QImage, QFont, QColor, QPalette

import numpy as np
import cv2


# ─────────────────────────────────────────────
#  Worker Thread (runs pipeline in background)
# ─────────────────────────────────────────────

class PipelineWorker(QThread):
    progress = Signal(str)
    finished = Signal(dict)
    error = Signal(str)

    def __init__(self, image_path, output_dir):
        super().__init__()
        self.image_path = image_path
        self.output_dir = output_dir

    def run(self):
        try:
            import pandas as pd
            from app.core.pipeline import run_hybrid_pipeline
            from app.core.export import export_results
            from app.core.qc import run_qc_checks

            # ── Run pipeline (detection + measurement in one step) ──
            self.progress.emit("🔍 Running analysis pipeline...")
            result = run_hybrid_pipeline(self.image_path,
                                         progress_fn=self.progress.emit)
            self.progress.emit(f"✅ {result['n_grapes']} grapes analysed")

            df_measurements = result["df_measurements"]

            # ── QC ──
            self.progress.emit("🔬 Running QC checks...")
            qc_report = run_qc_checks(
                result["labeled_mask"],
                df_measurements,
                wb_factors=result.get("wb_factors"),
                grabcut_changes=result.get("grabcut_changes", []),
                cell_grape_counts=result.get("cell_grape_counts", []),
                grid_method=result.get("grid_method"),
            )

            # ── Export ──
            self.progress.emit("💾 Saving CSV + Excel...")
            output_paths = export_results(
                df_measurements,
                pd.DataFrame(),
                result["debug_images"],
                self.output_dir,
                Path(self.image_path).name,
            )

            self.finished.emit({
                "result":          result,
                "df_measurements": df_measurements,
                "qc_report":       qc_report,
                "output_paths":    output_paths,
            })

        except Exception as e:
            import traceback
            self.error.emit(f"{str(e)}\n\n{traceback.format_exc()}")


# ─────────────────────────────────────────────
#  Image Display Widget
# ─────────────────────────────────────────────

class ImageDisplay(QLabel):
    def __init__(self, title=""):
        super().__init__()
        self.setAlignment(Qt.AlignCenter)
        self.setMinimumSize(300, 200)
        self.setStyleSheet("""
            QLabel {
                background-color: #1a1a2e;
                border: 1px solid #444;
                border-radius: 6px;
                color: #888;
            }
        """)
        self.setText(title if title else "No image")
        self._title = title

    def set_numpy_image(self, img_rgb: np.ndarray, max_size=(640, 480)):
        if img_rgb is None:
            self.setText(self._title + "\n(not available)")
            return

        h, w = img_rgb.shape[:2]
        scale = min(max_size[0] / w, max_size[1] / h, 1.0)
        new_w, new_h = int(w * scale), int(h * scale)
        img_resized = cv2.resize(img_rgb, (new_w, new_h))

        if len(img_resized.shape) == 2:
            q_img = QImage(img_resized.data, new_w, new_h, new_w,
                           QImage.Format_Grayscale8)
        else:
            img_cont = np.ascontiguousarray(img_resized)
            q_img = QImage(img_cont.data, new_w, new_h, new_w * 3,
                           QImage.Format_RGB888)

        self.setPixmap(QPixmap.fromImage(q_img))


# ─────────────────────────────────────────────
#  Data Table Widget
# ─────────────────────────────────────────────

def make_table(df) -> QTableWidget:
    if df is None or df.empty:
        t = QTableWidget(0, 1)
        t.setHorizontalHeaderLabels(["No data"])
        return t

    t = QTableWidget(len(df), len(df.columns))
    t.setHorizontalHeaderLabels(list(df.columns))
    t.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
    t.setAlternatingRowColors(True)
    t.setStyleSheet("""
        QTableWidget {
            background-color: #0f0f23;
            color: #e0e0e0;
            gridline-color: #333;
            font-size: 11px;
        }
        QHeaderView::section {
            background-color: #1a1a3e;
            color: #a0c4ff;
            font-weight: bold;
            padding: 4px;
            border: 1px solid #333;
        }
        QTableWidget::item:alternate {
            background-color: #141428;
        }
    """)

    for r, (_, row) in enumerate(df.iterrows()):
        for c, val in enumerate(row):
            item = QTableWidgetItem(str(val))
            item.setFlags(item.flags() & ~Qt.ItemIsEditable)
            if df.columns[c] == "Fallback" and str(val).lower() == "true":
                item.setForeground(QColor("#ffcc00"))
            t.setItem(r, c, item)

    return t


# ─────────────────────────────────────────────
#  Main Window
# ─────────────────────────────────────────────

class GrapeAnalyzerWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("🍇 Grape Analyzer — Color & Area Measurement Tool")
        self.setMinimumSize(1280, 860)
        self._apply_dark_theme()
        self._build_ui()
        self._worker = None
        self._results = None

    def _apply_dark_theme(self):
        palette = QPalette()
        palette.setColor(QPalette.Window,          QColor("#0a0a1a"))
        palette.setColor(QPalette.WindowText,      QColor("#e0e0e0"))
        palette.setColor(QPalette.Base,            QColor("#0f0f23"))
        palette.setColor(QPalette.AlternateBase,   QColor("#141428"))
        palette.setColor(QPalette.Text,            QColor("#e0e0e0"))
        palette.setColor(QPalette.Button,          QColor("#1a1a3e"))
        palette.setColor(QPalette.ButtonText,      QColor("#e0e0e0"))
        palette.setColor(QPalette.Highlight,       QColor("#3a3aff"))
        palette.setColor(QPalette.HighlightedText, QColor("#ffffff"))
        QApplication.setPalette(palette)

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)
        main_layout.setSpacing(8)
        main_layout.setContentsMargins(12, 12, 12, 12)

        # ── Header ──
        header = QLabel("🍇  Grape Analyzer")
        header.setFont(QFont("Arial", 22, QFont.Bold))
        header.setStyleSheet("color: #a0c4ff; padding: 8px 0;")
        main_layout.addWidget(header)

        subtitle = QLabel(
            "Upload an image → Auto-detect 36 grapes → Get Area, RGB, Lab, HSB per grape"
        )
        subtitle.setStyleSheet("color: #888; font-size: 12px;")
        main_layout.addWidget(subtitle)

        # ── Control Panel ──
        controls = QGroupBox("Controls")
        controls.setStyleSheet("""
            QGroupBox { color: #a0c4ff; font-weight: bold; border: 1px solid #333;
                         border-radius: 6px; margin-top: 8px; padding-top: 8px; }
            QGroupBox::title { subcontrol-origin: margin; left: 10px; }
        """)
        ctrl_layout = QHBoxLayout(controls)

        self.btn_upload = QPushButton("📁  Upload Image")
        self.btn_upload.setFixedHeight(40)
        self.btn_upload.setStyleSheet(self._btn_style("#2a4a8a", "#3a6aaa"))
        self.btn_upload.clicked.connect(self._on_upload_image)
        ctrl_layout.addWidget(self.btn_upload)

        self.lbl_image_path = QLabel("No image selected")
        self.lbl_image_path.setStyleSheet("color: #888; font-size: 11px;")
        ctrl_layout.addWidget(self.lbl_image_path, 1)

        self.btn_run = QPushButton("▶  Run Analysis")
        self.btn_run.setFixedHeight(40)
        self.btn_run.setEnabled(False)
        self.btn_run.setStyleSheet(self._btn_style("#5a2a8a", "#7a3aaa"))
        self.btn_run.clicked.connect(self._on_run)
        ctrl_layout.addWidget(self.btn_run)

        self.btn_export = QPushButton("💾  Open Output Folder")
        self.btn_export.setFixedHeight(40)
        self.btn_export.setEnabled(False)
        self.btn_export.setStyleSheet(self._btn_style("#3a5a2a", "#5a8a3a"))
        self.btn_export.clicked.connect(self._on_open_folder)
        ctrl_layout.addWidget(self.btn_export)

        main_layout.addWidget(controls)

        # ── Progress + Status ──
        prog_row = QHBoxLayout()
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 0)
        self.progress_bar.setVisible(False)
        self.progress_bar.setFixedHeight(8)
        self.progress_bar.setStyleSheet("""
            QProgressBar { border: none; background: #1a1a3e; border-radius: 4px; }
            QProgressBar::chunk { background: #3a3aff; border-radius: 4px; }
        """)
        prog_row.addWidget(self.progress_bar)
        main_layout.addLayout(prog_row)

        self.lbl_status = QLabel("Ready. Upload an image to begin.")
        self.lbl_status.setStyleSheet("color: #4eff91; font-size: 11px; padding: 2px;")
        main_layout.addWidget(self.lbl_status)

        # ── QC Badge ──
        self.lbl_qc = QLabel("")
        self.lbl_qc.setStyleSheet("font-size: 13px; font-weight: bold; padding: 4px;")
        self.lbl_qc.setVisible(False)
        main_layout.addWidget(self.lbl_qc)

        # ── Main Tab Area ──
        self.tabs = QTabWidget()
        self.tabs.setStyleSheet("""
            QTabWidget::pane { border: 1px solid #333; background: #0f0f23; }
            QTabBar::tab { background: #1a1a3e; color: #888; padding: 8px 16px;
                           border: 1px solid #333; }
            QTabBar::tab:selected { background: #2a2a5e; color: #ffffff;
                                    border-bottom: 2px solid #3a3aff; }
            QTabBar::tab:hover { background: #242450; color: #e0e0e0; }
        """)
        main_layout.addWidget(self.tabs, 1)

        self._build_tab_steps()
        self._build_tab_measurements()
        self._build_tab_qc()

        # ── Private state ──
        self._image_path  = None
        self._output_dir  = str(Path.home() / "grape_analyzer_outputs")

    def _btn_style(self, bg, hover):
        return f"""
            QPushButton {{
                background-color: {bg};
                color: white;
                border: none;
                border-radius: 6px;
                padding: 0 16px;
                font-size: 12px;
                font-weight: bold;
            }}
            QPushButton:hover {{ background-color: {hover}; }}
            QPushButton:disabled {{ background-color: #333; color: #666; }}
        """

    # ── TABS ──

    def _build_tab_steps(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.addWidget(QLabel(
            "  Processing Steps — each stage of the pipeline",
            styleSheet="color:#888; font-size:11px;"
        ))

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet("QScrollArea { border: none; background: #0a0a1a; }")

        inner = QWidget()
        grid = QHBoxLayout(inner)
        grid.setSpacing(8)

        steps = [
            ("original",      "1. Original Image"),
            ("normalized",    "2. White Balance Corrected"),
            ("zone_grid",     "3. Zone Grid (36 zones)"),
            ("grape_masks",   "4. Detected Grapes"),
            ("final_overlay", "5. ✅ Final Result"),
        ]
        self._step_displays = {}
        for key, title in steps:
            col = QVBoxLayout()
            lbl_title = QLabel(title)
            lbl_title.setStyleSheet(
                "color: #a0c4ff; font-size: 10px; font-weight: bold;"
            )
            lbl_title.setAlignment(Qt.AlignCenter)
            col.addWidget(lbl_title)

            img_disp = ImageDisplay(title)
            img_disp.setFixedSize(280, 220)
            col.addWidget(img_disp)
            self._step_displays[key] = img_disp
            grid.addLayout(col)

        scroll.setWidget(inner)
        layout.addWidget(scroll)
        self.tabs.addTab(tab, "📷 Processing Steps")

    def _build_tab_measurements(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.addWidget(QLabel(
            "  Per-Grape Measurements (1 row per grape, 36 rows total)",
            styleSheet="color:#888; font-size:11px;"
        ))
        self.tabs.addTab(tab, "📋 Measurements")
        self._tab_measurements_widget = tab

    def _build_tab_qc(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)
        self._qc_text = QTextEdit()
        self._qc_text.setReadOnly(True)
        self._qc_text.setStyleSheet("""
            QTextEdit {
                background-color: #0f0f23;
                color: #e0e0e0;
                font-family: monospace;
                font-size: 12px;
                border: 1px solid #333;
            }
        """)
        self._qc_text.setPlainText("QC report will appear here after running analysis.")
        layout.addWidget(self._qc_text)
        self.tabs.addTab(tab, "🔬 QC Report")

    # ── ACTIONS ──

    def _on_upload_image(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select Grape Image", "",
            "Images (*.jpg *.jpeg *.png *.tif *.tiff)"
        )
        if path:
            self._image_path = path
            name = Path(path).name
            self.lbl_image_path.setText(f"✅ {name}")
            self.lbl_image_path.setStyleSheet("color: #4eff91; font-size: 11px;")
            self.btn_run.setEnabled(True)
            self.lbl_status.setText(f"Image loaded: {name}")

            img = cv2.imread(path)
            if img is not None:
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                self._step_displays["original"].set_numpy_image(img_rgb)

    def _on_run(self):
        if not self._image_path:
            return

        self.btn_run.setEnabled(False)
        self.btn_export.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.lbl_qc.setVisible(False)
        self.lbl_status.setText("🔄 Running analysis...")

        self._worker = PipelineWorker(self._image_path, self._output_dir)
        self._worker.progress.connect(self._on_progress)
        self._worker.finished.connect(self._on_finished)
        self._worker.error.connect(self._on_error)
        self._worker.start()

    def _on_progress(self, msg):
        self.lbl_status.setText(msg)

    def _on_finished(self, data):
        self._results = data
        self.progress_bar.setVisible(False)
        self.btn_run.setEnabled(True)
        self.btn_export.setEnabled(True)

        result       = data["result"]
        debug        = result["debug_images"]
        qc           = data["qc_report"]
        df_m         = data["df_measurements"]
        fallback_ids = result.get("fallback_ids", [])
        n_real       = 36 - len(fallback_ids)

        # ── Update step images ──
        for key, disp in self._step_displays.items():
            img = debug.get(key)
            if img is not None and isinstance(img, np.ndarray):
                if img.dtype != np.uint8:
                    img = (
                        (img / img.max() * 255).astype(np.uint8)
                        if img.max() > 0 else img.astype(np.uint8)
                    )
                if len(img.shape) == 2:
                    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
                disp.set_numpy_image(img)
            else:
                disp.setText(f"{key}\n(not available)")

        # ── QC badge ──
        qc_colors = {"PASS": "#4eff91", "WARNING": "#ffcc00", "FAIL": "#ff4e4e"}
        self.lbl_qc.setText(
            f"QC: {qc['status']} — 36/36 grapes analysed "
            f"({n_real} detected, {len(fallback_ids)} fallback) | {result['method_used']}"
        )
        self.lbl_qc.setStyleSheet(
            f"color: {qc_colors.get(qc['status'], '#fff')}; "
            f"font-size: 13px; font-weight: bold;"
        )
        self.lbl_qc.setVisible(True)

        # ── Measurements table ──
        self._refresh_table(self._tab_measurements_widget, df_m,
                            "Per-Grape Measurements")

        # ── QC report text ──
        qc_text  = f"QC Status: {qc['status']}\n"
        qc_text += f"Grapes: 36/36 ({n_real} detected, {len(fallback_ids)} fallback)\n"
        qc_text += f"Segmentation: {result['method_used']}\n"
        qc_text += f"Measurement engine: Python (ImageJ-equivalent)\n"
        qc_text += f"Mean solidity: {qc.get('mean_solidity', 0):.3f}\n\n"

        px = result.get("px_per_mm")
        if px:
            qc_text += f"Scale: {px:.2f} px/mm  →  1 px = {1/px:.4f} mm\n\n"
        else:
            qc_text += "Scale: not detected (Area in px² only)\n\n"

        wb = result.get("wb_factors") or {}
        if wb:
            qc_text += "White Balance Correction:\n"
            qc_text += (f"  R×{wb.get('scale_R', 1):.3f}  "
                        f"G×{wb.get('scale_G', 1):.3f}  "
                        f"B×{wb.get('scale_B', 1):.3f}\n")
            qc_text += (f"  White ref: R={wb.get('mean_white_R', 0):.1f}  "
                        f"G={wb.get('mean_white_G', 0):.1f}  "
                        f"B={wb.get('mean_white_B', 0):.1f}\n\n")
        else:
            qc_text += "White Balance: skipped\n\n"

        if fallback_ids:
            qc_text += (f"Fallback masks used for {len(fallback_ids)} grapes: "
                        f"{fallback_ids}\n\n")
        else:
            qc_text += "✅ All 36 grapes detected without fallback.\n\n"

        if qc["issues"]:
            qc_text += "Issues:\n"
            for issue in qc["issues"]:
                qc_text += f"  ⚠ {issue}\n"
        else:
            qc_text += "✅ No issues found.\n"

        qc_text += "\nArea stats (across all grapes):\n"
        for k, v in qc["area_stats"].items():
            qc_text += f"  {k}: {v:.0f} px²\n"

        self._qc_text.setPlainText(qc_text)
        self.lbl_status.setText(
            f"✅ Analysis complete! Results saved to: {self._output_dir}"
        )
        self.tabs.setCurrentIndex(0)

    def _refresh_table(self, parent_widget, df, label_text):
        layout = parent_widget.layout()
        # Remove previous table widget if it exists
        for i in reversed(range(layout.count())):
            item = layout.itemAt(i)
            if item and item.widget() and isinstance(item.widget(), QTableWidget):
                item.widget().deleteLater()

        if df is None or (hasattr(df, "empty") and df.empty):
            lbl = QLabel(f"  {label_text}")
            lbl.setStyleSheet("color: #888;")
            layout.addWidget(lbl)
            return

        layout.addWidget(make_table(df))

    def _on_error(self, msg):
        self.progress_bar.setVisible(False)
        self.btn_run.setEnabled(True)
        self.lbl_status.setText("❌ Error occurred — see details below")
        self._qc_text.setPlainText(f"ERROR:\n{msg}")
        self.tabs.setCurrentIndex(2)   # QC tab
        QMessageBox.critical(
            self, "Pipeline Error",
            f"An error occurred:\n\n{msg[:300]}...\n\nSee QC tab for full details."
        )

    def _on_open_folder(self):
        import subprocess
        subprocess.Popen(["open", self._output_dir])


# ─────────────────────────────────────────────
#  Entry Point
# ─────────────────────────────────────────────

def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    window = GrapeAnalyzerWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()

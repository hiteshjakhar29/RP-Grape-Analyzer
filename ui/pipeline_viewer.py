"""
pipeline_viewer.py — Horizontal strip of 6 pipeline step thumbnails.
Click any thumbnail to emit step_clicked(index).
"""

import numpy as np
from PySide6.QtWidgets import QWidget, QHBoxLayout, QLabel, QVBoxLayout
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QPixmap, QImage


STEP_LABELS = [
    ("Step 1", "Original Image"),
    ("Step 2", "Background\nRemoved"),
    ("Step 3", "Binary\nSegmentation"),
    ("Step 4", "Reference\nMasks"),
    ("Step 5", "Adapted\nMasks"),
    ("Step 6", "Measurement\nRegions"),
]


class PipelineViewer(QWidget):
    """Clickable 6-thumbnail pipeline viewer."""

    step_clicked = Signal(int)   # emits step index 0-5

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setSpacing(8)
        layout.setContentsMargins(4, 4, 4, 4)
        self.thumbs: list[_ClickableLabel] = []

        for i, (title, subtitle) in enumerate(STEP_LABELS):
            frame = QVBoxLayout()
            frame.setSpacing(2)

            lbl_title = QLabel(f"<b>{title}</b><br><small>{subtitle}</small>")
            lbl_title.setAlignment(Qt.AlignCenter)
            lbl_title.setStyleSheet("color: #ccc; font-size: 11px;")

            thumb = _ClickableLabel(i, self.step_clicked)
            thumb.setFixedSize(180, 135)
            thumb.setStyleSheet(
                "border: 2px solid #555; border-radius: 4px; background: #2a2a2a;"
            )
            thumb.setAlignment(Qt.AlignCenter)
            thumb.setText("—")

            frame.addWidget(lbl_title)
            frame.addWidget(thumb)
            layout.addLayout(frame)
            self.thumbs.append(thumb)

    def update_step(self, step_idx: int, image_rgb: np.ndarray) -> None:
        """Update one pipeline step thumbnail with an RGB numpy array."""
        if step_idx < 0 or step_idx >= len(self.thumbs):
            return
        h, w = image_rgb.shape[:2]
        # Ensure contiguous uint8 C-order array
        img = np.ascontiguousarray(image_rgb, dtype=np.uint8)
        qimg = QImage(img.data, w, h, 3 * w, QImage.Format_RGB888)
        pix  = QPixmap.fromImage(qimg).scaled(
            180, 135, Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        self.thumbs[step_idx].setPixmap(pix)
        self.thumbs[step_idx].setText("")
        self.thumbs[step_idx].setStyleSheet(
            "border: 2px solid #4CAF50; border-radius: 4px;"
        )


class _ClickableLabel(QLabel):
    def __init__(self, idx: int, signal: Signal):
        super().__init__()
        self._idx    = idx
        self._signal = signal
        self.setCursor(Qt.PointingHandCursor)

    def mousePressEvent(self, event):
        self._signal.emit(self._idx)

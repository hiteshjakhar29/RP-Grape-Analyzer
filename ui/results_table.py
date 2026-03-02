"""
results_table.py — 36-row results table with alternating neutral rows.
Group identity is shown through text color in the Group column only.
"""

from PySide6.QtWidgets import QTableWidget, QTableWidgetItem, QHeaderView
from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QFont


COLUMNS = [
    ("ID",         "grape_id"),
    ("Group",      "group"),
    ("Ratio",      "ratio"),
    ("Row",        "grape_row"),
    ("Day",        "day"),
    ("Area (px)",  "area_px"),
    ("Mean R",     "mean_R"),
    ("Mean G",     "mean_G"),
    ("Mean B",     "mean_B"),
    ("Mean L*",    "mean_L"),
    ("Mean a*",    "mean_a"),
    ("Mean b*",    "mean_b"),
    ("Hue (°)",    "mean_H"),
    ("Sat",        "mean_S"),
    ("Brightness", "mean_Br"),
]

# Row backgrounds — alternating white / light gray
_ROW_ODD  = QColor(0xFF, 0xFF, 0xFF)   # white       (0-indexed even rows)
_ROW_EVEN = QColor(0xF5, 0xF5, 0xF5)   # light gray  (0-indexed odd rows)

# Text colors
_TEXT_DARK    = QColor(0x1A, 0x1A, 0x1A)   # near-black for all data cells
_COATED_TEXT  = QColor(0x1A, 0x52, 0x76)   # dark blue  — Group cell, Coated
_CONTROL_TEXT = QColor(0x78, 0x42, 0x12)   # dark orange — Group cell, Control

# Index of the "Group" column (resolved once at module load)
_GROUP_COL = next(i for i, (_, k) in enumerate(COLUMNS) if k == "group")


class ResultsTable(QTableWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setColumnCount(len(COLUMNS))
        self.setHorizontalHeaderLabels([c[0] for c in COLUMNS])
        self.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self.setAlternatingRowColors(False)
        self.setSortingEnabled(True)
        bold = QFont()
        bold.setBold(True)
        self.horizontalHeader().setFont(bold)
        self.setEditTriggers(QTableWidget.NoEditTriggers)
        self.setSelectionBehavior(QTableWidget.SelectRows)

    def populate(self, results: list) -> None:
        """Fill the table with a list of measurement dicts."""
        self.setSortingEnabled(False)
        self.setRowCount(len(results))

        for row, r in enumerate(results):
            bg_color = _ROW_ODD if row % 2 == 0 else _ROW_EVEN
            group    = r.get("group", "")

            for col, (_, key) in enumerate(COLUMNS):
                val  = r.get(key, "")
                text = f"{val:.4f}" if isinstance(val, float) else str(val)

                item = QTableWidgetItem(text)
                item.setBackground(bg_color)
                if col == _GROUP_COL:
                    item.setForeground(_COATED_TEXT if group == "Coated" else _CONTROL_TEXT)
                else:
                    item.setForeground(_TEXT_DARK)
                item.setTextAlignment(Qt.AlignCenter)
                self.setItem(row, col, item)

        self.setSortingEnabled(True)

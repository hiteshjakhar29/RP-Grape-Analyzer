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
    ("Std R",      "std_R"),
    ("Mean G",     "mean_G"),
    ("Std G",      "std_G"),
    ("Mean B",     "mean_B"),
    ("Std B",      "std_B"),
    ("Mean L*",    "mean_L"),
    ("Std L*",     "std_L"),
    ("Mean a*",    "mean_a"),
    ("Std a*",     "std_a"),
    ("Mean b*",    "mean_b"),
    ("Std b*",     "std_b"),
    ("Hue (°)",    "mean_H"),
    ("Std H",      "std_H"),
    ("Sat",        "mean_S"),
    ("Std S",      "std_S"),
    ("Brightness", "mean_Br"),
    ("Std Br",     "std_Br"),
    ("Eccentricity", "eccentricity"),
]

# Row backgrounds — dark theme alternating rows
_ROW_ODD  = QColor(0x2A, 0x2A, 0x3C)   # surface (even rows)
_ROW_EVEN = QColor(0x24, 0x24, 0x34)   # slightly darker (odd rows)

# Text colors
_TEXT_DARK    = QColor(0xCD, 0xD6, 0xF4)   # Catppuccin text
_COATED_TEXT  = QColor(0x89, 0xB4, 0xFA)   # blue  — Group cell, Coated
_CONTROL_TEXT = QColor(0xFA, 0xB3, 0x87)   # peach — Group cell, Control

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
        self.setStyleSheet("""
            QTableWidget {
                background: #2A2A3C;
                color: #CDD6F4;
                border: 1px solid #3A3A52;
                border-radius: 6px;
                gridline-color: #313244;
                font-size: 12px;
            }
            QHeaderView::section {
                background: #1E1E2E;
                color: #89B4FA;
                border: none;
                border-right: 1px solid #3A3A52;
                border-bottom: 1px solid #3A3A52;
                padding: 5px 8px;
                font-weight: bold;
                font-size: 12px;
            }
            QTableWidget::item:selected {
                background: #313244;
                color: #CDD6F4;
            }
            QScrollBar:vertical {
                background: #1E1E2E;
                width: 8px;
                border-radius: 4px;
            }
            QScrollBar::handle:vertical {
                background: #585B70;
                border-radius: 4px;
            }
            QScrollBar:horizontal {
                background: #1E1E2E;
                height: 8px;
                border-radius: 4px;
            }
            QScrollBar::handle:horizontal {
                background: #585B70;
                border-radius: 4px;
            }
        """)

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

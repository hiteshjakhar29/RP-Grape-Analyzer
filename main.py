"""main.py — Entry point for Grape Analyzer."""

import sys
import os

# Ensure project root is on sys.path so 'core' and 'ui' are importable.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from PySide6.QtWidgets import QApplication
from PySide6.QtCore import Qt
from ui.main_window import MainWindow


def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")

    # Dark palette
    from PySide6.QtGui import QPalette, QColor
    palette = QPalette()
    palette.setColor(QPalette.Window,          QColor(45, 45, 45))
    palette.setColor(QPalette.WindowText,      QColor(220, 220, 220))
    palette.setColor(QPalette.Base,            QColor(30, 30, 30))
    palette.setColor(QPalette.AlternateBase,   QColor(50, 50, 50))
    palette.setColor(QPalette.ToolTipBase,     QColor(255, 255, 220))
    palette.setColor(QPalette.ToolTipText,     QColor(0, 0, 0))
    palette.setColor(QPalette.Text,            QColor(220, 220, 220))
    palette.setColor(QPalette.Button,          QColor(60, 60, 60))
    palette.setColor(QPalette.ButtonText,      QColor(220, 220, 220))
    palette.setColor(QPalette.BrightText,      QColor(255, 80, 80))
    palette.setColor(QPalette.Link,            QColor(42, 130, 218))
    palette.setColor(QPalette.Highlight,       QColor(42, 130, 218))
    palette.setColor(QPalette.HighlightedText, QColor(0, 0, 0))
    app.setPalette(palette)

    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()

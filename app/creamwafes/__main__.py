"""Entry point: ``uv run creamwafes [project-folder]``."""

import sys
from pathlib import Path

from PySide6.QtGui import QColor, QPalette
from PySide6.QtWidgets import QApplication

from .diag import Inspector, log, setup_logging
from .paths import DATA_DIR, OLD_DATA_DIR, move_old_songs
from .model import Project
from .ui.main_window import MainWindow


def dark_palette() -> QPalette:
    pal = QPalette()
    for role, color in {
        QPalette.Window: "#1f2329", QPalette.WindowText: "#d7dce4", QPalette.Base: "#16181d",
        QPalette.AlternateBase: "#20242b", QPalette.Text: "#d7dce4", QPalette.Button: "#2a2f38",
        QPalette.ButtonText: "#d7dce4", QPalette.Highlight: "#4f7fd9", QPalette.HighlightedText: "#ffffff",
        QPalette.ToolTipBase: "#2a2f38", QPalette.ToolTipText: "#d7dce4", QPalette.PlaceholderText: "#6c7480",
    }.items():
        pal.setColor(role, QColor(color))
    return pal


def main() -> None:
    moved = move_old_songs()
    setup_logging()
    if moved:
        log.info("moved songs from %s to %s (the app's new name)", OLD_DATA_DIR, DATA_DIR)
    app = QApplication(sys.argv)
    app.setApplicationName("creamWafes")
    app.setStyle("Fusion")
    app.setPalette(dark_palette())
    project = None
    if len(sys.argv) > 1 and (Path(sys.argv[1]) / "project.json").exists():
        project = Project.load(Path(sys.argv[1]))
    window = MainWindow(project)
    window.inspector = Inspector(window)
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()

"""Entry point: ``uv run creamwafes [song-folder]``."""

import sys
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QFont, QIcon, QPainter, QPalette, QPixmap
from PySide6.QtWidgets import QApplication, QSplashScreen

from . import ASSETS, __version__

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


def app_icon() -> QIcon:
    icon = QIcon()
    for size in (128, 256, 512):
        icon.addFile(str(ASSETS / f"app-icon-{size}.png"))
    return icon


def splash() -> QSplashScreen | None:
    """The startup splash, with this version written where the artwork leaves room for it."""
    pixmap = QPixmap(str(ASSETS / "splash-960x600.png"))
    if pixmap.isNull():
        return None
    p = QPainter(pixmap)
    p.fillRect(800, 52, 110, 28, QColor("#1a1210"))  # the "[VERSION]" placeholder
    font = QFont("DM Mono")
    font.setStyleHint(QFont.Monospace)
    font.setPixelSize(15)
    p.setFont(font)
    p.setPen(QColor("#cdbba4"))
    p.drawText(700, 52, 195, 28, Qt.AlignRight | Qt.AlignVCenter, f"v{__version__}")
    p.end()
    screen = QSplashScreen(pixmap)
    screen.show()
    QApplication.processEvents()
    return screen


def main() -> None:
    moved = move_old_songs()
    setup_logging()
    if moved:
        log.info("moved songs from %s to %s (the app's new name)", OLD_DATA_DIR, DATA_DIR)
    app = QApplication(sys.argv)
    app.setApplicationName("creamWafes")
    app.setApplicationVersion(__version__)
    app.setDesktopFileName("creamwafes")  # Wayland/GNOME: the dock uses creamwafes.desktop's icon
    app.setWindowIcon(app_icon())
    shown = splash()
    app.setStyle("Fusion")
    app.setPalette(dark_palette())
    project = None
    if len(sys.argv) > 1 and (Path(sys.argv[1]) / "project.json").exists():
        project = Project.load(Path(sys.argv[1]))
    window = MainWindow(project)
    window.inspector = Inspector(window)
    window.show()
    if shown is not None:
        shown.finish(window)
    sys.exit(app.exec())


if __name__ == "__main__":
    main()

"""Where creamWafes keeps songs, logs and exports: <your Music folder>/creamWafes (override with
CREAMWAFES_HOME)."""

from __future__ import annotations

import os
from pathlib import Path

APP = "creamWafes"


def _music_dir() -> Path:
    try:
        from PySide6.QtCore import QStandardPaths
        found = QStandardPaths.writableLocation(QStandardPaths.MusicLocation)
        if found:
            return Path(found)
    except Exception:  # noqa: BLE001 - no Qt yet / no desktop: the usual place
        pass
    return Path.home() / "Music"


DATA_DIR = Path(os.environ.get("CREAMWAFES_HOME") or _music_dir() / APP)
OLD_DATA_DIR = _music_dir() / "LayerDAW"  # the app's name before it was released


def move_old_songs() -> bool:
    """One time: songs made under the old name move to the new folder (nothing inside changes;
    a song's files are all relative to its own folder)."""
    if OLD_DATA_DIR.is_dir() and not DATA_DIR.exists() and not os.environ.get("CREAMWAFES_HOME"):
        OLD_DATA_DIR.rename(DATA_DIR)
        return True
    return False

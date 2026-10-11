"""Diagnostics: a session log, crash capture, and a local inspect socket for debugging the live app.

    ~/Music/LayerDAW/session.log        everything the user did + errors + a health line every 5 s
    $XDG_RUNTIME_DIR/layerdaw.sock      line commands: state | screenshot <path> | log [n] | restart
                                        (tools/layerdaw-inspect.py is the client; only this user can connect)
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
import traceback
from collections import deque
from pathlib import Path

from PySide6.QtCore import QProcess, QtMsgType, QTimer, qInstallMessageHandler
from PySide6.QtNetwork import QLocalServer

LOG_DIR = Path.home() / "Music" / "LayerDAW"
LOG_FILE = LOG_DIR / "session.log"
# LAYERDAW_SOCKET lets test instances use their own socket instead of taking over the real app's
SOCKET = Path(os.environ.get("LAYERDAW_SOCKET") or Path(os.environ.get("XDG_RUNTIME_DIR", "/tmp")) / "layerdaw.sock")

log = logging.getLogger("layerdaw")
_recent: deque[str] = deque(maxlen=400)


class _Recent(logging.Handler):
    def emit(self, record):
        _recent.append(self.format(record))


def setup_logging() -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    if LOG_FILE.exists():
        LOG_FILE.replace(LOG_FILE.with_suffix(".prev.log"))
    fmt = logging.Formatter("%(asctime)s.%(msecs)03d %(levelname)-5s %(message)s", "%H:%M:%S")
    for handler in (logging.FileHandler(LOG_FILE), _Recent()):
        handler.setFormatter(fmt)
        log.addHandler(handler)
    log.setLevel(logging.INFO)

    def excepthook(kind, value, tb):
        log.error("UNCAUGHT %s: %s\n%s", kind.__name__, value, "".join(traceback.format_tb(tb)))
        window = getattr(excepthook, "window", None)
        if window is not None:  # keep the app alive and tell the user something went wrong
            window.engine_label.setText(f"⚠ internal error: {kind.__name__}: {value} (logged)")

    sys.excepthook = excepthook

    def qt_messages(mode, _context, message):
        if mode in (QtMsgType.QtWarningMsg, QtMsgType.QtCriticalMsg, QtMsgType.QtFatalMsg):
            if not any(noise in message for noise in ("propagateSizeHints", "QWaylandTextInput")):
                log.warning("qt: %s", message)

    qInstallMessageHandler(qt_messages)
    log.info("LayerDAW started (pid %d)", os.getpid())


def recent(n: int = 60) -> list[str]:
    return list(_recent)[-n:]


class Inspector:
    """Health logging + the local inspect socket, attached to the main window."""

    def __init__(self, window):
        self.w = window
        sys.excepthook.window = window
        self._last_callbacks = window.player.callbacks
        self._health = QTimer(window, interval=5000, timeout=self._log_health)
        self._health.start()
        self.server = QLocalServer(window)
        self.server.setSocketOptions(QLocalServer.UserAccessOption)
        self.server.newConnection.connect(self._accept)
        self._listen()

    def _listen(self) -> None:
        self.server.close()
        QLocalServer.removeServer(str(SOCKET))
        if not self.server.listen(str(SOCKET)):
            log.warning("inspect socket unavailable: %s", self.server.errorString())

    def state(self) -> dict:
        w, p = self.w, self.w.player
        focus = w.focusWidget()
        return {
            "song": str(w.project.folder), "bpm": w.project.bpm, "key": w.project.key,
            "tracks": [{"name": t.name, "kind": t.kind, "muted": t.muted, "solo": t.solo,
                        "volume": round(t.volume, 2), "clips": [[round(c.start, 2), round(c.length, 2)] for c in t.clips]}
                       for t in w.project.tracks],
            "pads": [{"i": i, "name": pad.name, "mode": pad.mode, "kind": pad.kind, "play": pad.play,
                      "label": w.pad_label(pad), "sound": pad.clip is not None}
                     for i, pad in enumerate(w.project.pads) if pad.clip is not None or pad.play != "sample"],
            "selected_pads": w.selected_pads,
            "player": {"playing": p.playing, "position": round(p.position, 3), "voices": p.voice_count,
                       "callbacks": p.callbacks, "stream": getattr(p, "_stream_settings", None),
                       "stream_active": bool(getattr(p._stream, "active", False))},
            "engine_ready": w.engine.is_ready, "jobs": [j["name"] for j in w.jobs.values()],
            "pads_visible": w.pads_dock.isVisible(), "recording": w.pad_panel.rec.isChecked(),
            "focus": type(focus).__name__ if focus else None, "active_window": w.isActiveWindow(),
            "status": w.engine_label.text(), "selection": w.state.selection,
            "undo": w.history.next_undo(), "log": recent(40),
        }

    def _restart(self) -> str:
        """Save, start a fresh copy of the app on the same song (picking up new code), then quit."""
        w = self.w
        if w.jobs:
            return "busy: generation in progress — not restarting"
        w.project.save()
        args = [sys.argv[0], str(w.project.folder)]
        self.server.close()  # hand the socket over cleanly: closing later would delete the new copy's
        if not QProcess.startDetached(sys.executable, args):
            self._listen()
            return "error: couldn't start a new copy"
        self._closing = True
        log.info("restarting on request (new code) with %s", w.project.folder)
        QTimer.singleShot(300, w.close)
        return f"restarting with {w.project.folder.name}"

    def _log_health(self) -> None:
        if not SOCKET.exists() and not getattr(self, "_closing", False):
            log.warning("inspect socket went missing — re-creating it")
            self._listen()
        p = self.w.player
        rate = (p.callbacks - self._last_callbacks) / 5.0
        self._last_callbacks = p.callbacks
        log.info("health playing=%s pos=%.2f voices=%d callbacks/s=%.0f jobs=%d",
                 p.playing, p.position, p.voice_count, rate, len(self.w.jobs))

    def _accept(self) -> None:
        conn = self.server.nextPendingConnection()
        conn.readyRead.connect(lambda c=conn: self._serve(c))

    def _serve(self, conn) -> None:
        while conn.canReadLine():
            line = bytes(conn.readLine()).decode().strip()
            cmd, _, arg = line.partition(" ")
            try:
                if cmd == "state":
                    reply = json.dumps(self.state(), default=str)
                elif cmd == "screenshot":
                    path = arg or f"/tmp/layerdaw-{int(time.time())}.png"
                    self.w.grab().save(path)
                    reply = path
                elif cmd == "log":
                    reply = "\n".join(recent(int(arg or 80)))
                elif cmd == "restart":
                    reply = self._restart()
                else:
                    reply = "commands: state | screenshot [path] | log [n] | restart"
            except Exception as exc:  # noqa: BLE001 - report, don't kill the app
                reply = f"error: {exc}"
            conn.write((reply + "\n\x04\n").encode())
            conn.flush()

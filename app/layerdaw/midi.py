"""USB MIDI input: auto-connects to any controller that's plugged in (and ones plugged in later).

Polled from a Qt timer every few ms instead of a thread, so hits arrive on the UI thread.
"""

from __future__ import annotations

from PySide6.QtCore import QObject, QTimer, Signal

try:
    import mido

    mido.set_backend("mido.backends.rtmidi")
except Exception:  # noqa: BLE001 - MIDI is optional; the app works without it
    mido = None

IGNORED = ("Midi Through", "RtMidi", "PipeWire-System")


class MidiInput(QObject):
    note_on = Signal(int, float)   # note, velocity 0..1
    note_off = Signal(int)
    status = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._ports: dict[str, object] = {}
        self._poll = QTimer(self, interval=3, timeout=self._read)
        self._scan = QTimer(self, interval=2000, timeout=self._rescan)
        if mido is None:
            self.status.emit("MIDI unavailable")
            return
        self._rescan()
        self._poll.start()
        self._scan.start()

    def describe(self) -> str:
        if mido is None:
            return "MIDI: unavailable"
        return "MIDI: " + (", ".join(n.split(":")[0] for n in self._ports) if self._ports else "no controller")

    def _rescan(self) -> None:
        try:
            names = {n for n in mido.get_input_names() if not n.startswith(IGNORED)}
        except Exception:  # noqa: BLE001 - backend hiccup: try again next scan
            return
        changed = False
        for gone in set(self._ports) - names:
            self._ports.pop(gone).close()
            changed = True
        for new in names - set(self._ports):
            try:
                self._ports[new] = mido.open_input(new)
                changed = True
            except Exception:  # noqa: BLE001 - busy/unplugged mid-scan
                pass
        if changed:
            self.status.emit(self.describe())

    def _read(self) -> None:
        for port in list(self._ports.values()):
            for msg in port.iter_pending():
                if msg.type == "note_on" and msg.velocity > 0:
                    self.note_on.emit(msg.note, msg.velocity / 127)
                elif msg.type in ("note_off", "note_on"):
                    self.note_off.emit(msg.note)

    def close(self) -> None:
        for port in self._ports.values():
            port.close()
        self._ports.clear()

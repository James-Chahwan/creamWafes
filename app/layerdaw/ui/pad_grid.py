"""The 4×4 pad grid (MPC layout: pad 1 bottom-left): playing, flashing, and multi-selection."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PySide6.QtCore import QPoint, QRectF, Qt, QTimer, Signal
from PySide6.QtGui import QColor, QPainter, QPen
from PySide6.QtWidgets import QGridLayout, QLabel, QScrollArea, QWidget

from ..model import AUDIO_EXTENSIONS, Pad

# Keyboard rows map onto pad rows top→bottom, like the hardware: Z is pad 1, bottom-left.
KEY_ROWS = ["1234", "QWER", "ASDF", "ZXCV"]
KEY_TO_PAD = {getattr(Qt, f"Key_{ch}"): (3 - r) * 4 + c for r, row in enumerate(KEY_ROWS) for c, ch in enumerate(row)}
PAD_TO_KEY = {pad: KEY_ROWS[3 - pad // 4][pad % 4] for pad in range(16)}
MIDI_BASE = 36  # most pad controllers send 36..51 for their 16 pads

CATEGORIES = ["kick", "snare", "clap", "hi-hat", "open hat", "perc", "drums", "808", "bass", "keys", "lead",
              "pad", "stab", "hook", "vocals", "fx", "other"]
_FAMILY = {"kick": "drums", "snare": "drums", "clap": "drums", "hi-hat": "drums", "open hat": "drums",
           "perc": "drums", "808": "bass", "keys": "keyboard", "lead": "synth", "pad": "synth", "stab": "synth",
           "vocal chop": "vocals", "other": "audio"}
CHORD_COLOR = "#a978d6"


KEYED = 16  # the top 4×4: played by the keyboard and a MIDI pad controller; the tray below never ends
COLUMNS = 4


def note_to_pad(note: int) -> int:
    """Which pad a MIDI note plays: 36–51 the 4×4 (any octave), 52 and up continue into the tray."""
    return note - MIDI_BASE if note >= MIDI_BASE + KEYED else (note - MIDI_BASE) % KEYED


def pad_key(index: int) -> str:
    """How a pad is named: its key on the 4×4 ("Q"), its number in the tray ("17")."""
    return PAD_TO_KEY[index] if index < KEYED else str(index + 1)


def family(kind: str) -> str:
    """Colour family for a pad category ("kick" → drums…)."""
    return _FAMILY.get(kind, kind)


def guess_kind(name: str) -> str:
    """A category from a file name: 'Kick_01.wav' → kick, 'bass stem' → bass…"""
    n = name.lower()
    for kind, words in (("808", ("808",)), ("kick", ("kick", "bd")), ("snare", ("snare", "sd")),
                        ("clap", ("clap",)), ("open hat", ("open hat", "openhat", "oh")),
                        ("hi-hat", ("hat", "hh")), ("perc", ("perc", "rim", "shaker", "tom")),
                        ("drums", ("drum", "kit", "loop")), ("bass", ("bass", "sub")),
                        ("vocals", ("vox", "vocal", "voice", "acapella")), ("keys", ("key", "piano", "rhodes", "organ")),
                        ("lead", ("lead", "pluck", "synth")), ("pad", ("pad", "string")), ("fx", ("fx", "riser", "impact"))):
        if any(w in n for w in words):
            return kind
    return "other"


class PadButton(QWidget):
    pressed = Signal(int, float, object)   # index, velocity, keyboard modifiers
    released = Signal(int)
    menu = Signal(int, QPoint)
    file_dropped = Signal(int, str)

    def __init__(self, index: int, parent=None):
        super().__init__(parent)
        self.index = index
        self.pad: Pad | None = None
        self.color = QColor("#e0a14a")
        self.big: tuple[str, str] | None = None  # (chord/note name, numeral) for chord & note pads
        self.flash = 0.0
        self.selected = False
        self.generating: float | None = None
        self.setMinimumSize(92, 62)
        self.setAcceptDrops(True)
        self._env: tuple | None = None

    def set_pad(self, pad: Pad, color: QColor, big: tuple[str, str] | None) -> None:
        self.pad, self.color, self.big, self._env = pad, color, big, None
        what = {"chord": "chord", "note": "note"}.get(pad.play, "sample")
        where = (f"key {PAD_TO_KEY[self.index]} · MIDI {MIDI_BASE + self.index}" if self.index < KEYED
                 else "tray (click to play)")
        self.setToolTip(f"Pad {pad_key(self.index)} · {where}\n"
                        f"{pad.name or 'empty'} ({what})\nclick: select & play · Ctrl/Shift+click: select several")
        self.update()

    def dragEnterEvent(self, ev):
        if any(Path(u.toLocalFile()).suffix.lower() in AUDIO_EXTENSIONS for u in ev.mimeData().urls()):
            ev.acceptProposedAction()

    def dropEvent(self, ev):
        path = next((u.toLocalFile() for u in ev.mimeData().urls()
                     if Path(u.toLocalFile()).suffix.lower() in AUDIO_EXTENSIONS), None)
        if path:
            self.file_dropped.emit(self.index, path)

    def _envelope(self, width: int):
        buf = self.pad.buffer() if self.pad else None
        if buf is None or len(buf) == 0:
            return None
        if self._env is None or self._env[0] != width or self._env[1] is not buf:
            mono = np.abs(buf.mean(axis=1))
            edges = np.linspace(0, len(mono), width + 1).astype(int)
            peaks = np.array([mono[a:b].max() if b > a else 0 for a, b in zip(edges[:-1], edges[1:])])
            self._env = (width, buf, peaks / max(peaks.max(), 1e-6))
        return self._env[2]

    def paintEvent(self, _ev):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        r = QRectF(self.rect()).adjusted(2, 2, -2, -2)
        if self.generating is not None:
            p.setPen(QPen(QColor("#7aa2f7"), 1.5, Qt.DashLine))
            p.setBrush(QColor(122, 162, 247, 40))
            p.drawRoundedRect(r, 7, 7)
            p.fillRect(QRectF(r.left() + 4, r.bottom() - 6, (r.width() - 8) * self.generating, 3), QColor("#7aa2f7"))
            p.setPen(QColor("#c8d3f5"))
            p.drawText(r, Qt.AlignCenter, f"generating… {int(self.generating * 100)}%")
            return
        pad = self.pad
        filled = pad is not None and (pad.clip is not None or pad.play != "sample")
        base = QColor(self.color if filled else QColor("#2a2f38"))
        base.setAlpha(70 if filled else 255)
        if self.flash > 0:
            base = QColor(self.color).lighter(130)
            base.setAlpha(int(90 + 165 * self.flash))
        edge = QColor("#ffffff") if self.selected else (QColor(self.color) if filled else QColor("#3a404b"))
        p.setPen(QPen(edge, 2.5 if self.selected else 1.5))
        p.setBrush(base)
        p.drawRoundedRect(r, 7, 7)
        inner = r.adjusted(7, 4, -6, -5)
        if self.big is not None:
            p.setPen(QColor("#f1ecfa"))
            p.drawText(inner, Qt.AlignRight | Qt.AlignBottom, self.big[1])
            font = p.font()
            font.setPointSize(font.pointSize() + 5)
            font.setBold(True)
            line = p.fontMetrics().height()
            p.save()
            p.setFont(font)
            p.drawText(r.adjusted(0, line * 0.5, 0, -line * 0.9), Qt.AlignCenter, self.big[0])  # clear of the name
            p.restore()
        else:
            env = self._envelope(int(r.width()) - 12) if pad is not None and pad.clip is not None else None
            if env is not None:
                p.setPen(QColor(255, 255, 255, 90))
                mid, half = r.center().y() + 4, r.height() / 4
                for x, v in enumerate(env):
                    p.drawLine(int(r.left() + 6 + x), int(mid - v * half), int(r.left() + 6 + x), int(mid + v * half))
        p.setPen(QColor("#e6e9ee") if filled else QColor("#6c7480"))
        p.drawText(inner, Qt.AlignLeft | Qt.AlignTop, pad_key(self.index))
        if filled:
            tags = [pad.mode if pad.mode != "oneshot" else "", f"{pad.pitch:+d}st" if pad.pitch else "",
                    f"{20 * np.log10(pad.gain):+.0f}dB" if abs(20 * np.log10(pad.gain)) >= 0.5 else ""]
            tag = " ".join(t for t in tags if t)
            if tag:
                p.drawText(inner, Qt.AlignRight | Qt.AlignTop, tag)
            name = p.fontMetrics().elidedText(pad.name or ("synth" if pad.clip is None else ""), Qt.ElideRight,
                                              int(r.width()) - 40 if self.big else int(r.width()) - 12)
            p.drawText(inner, Qt.AlignLeft | Qt.AlignBottom, name)

    def mousePressEvent(self, ev):
        if ev.button() == Qt.LeftButton:
            # top half = full velocity; lower down gets softer (to 60%), like a velocity-sensitive pad
            depth = max(0.0, min(1.0, (ev.position().y() / max(1, self.height()) - 0.5) * 2))
            self.pressed.emit(self.index, 1.0 - 0.4 * depth, ev.modifiers())
        elif ev.button() == Qt.RightButton:
            self.menu.emit(self.index, ev.globalPosition().toPoint())

    def mouseReleaseEvent(self, ev):
        if ev.button() == Qt.LeftButton:
            self.released.emit(self.index)


class PadGrid(QScrollArea):
    """The 4×4 you play with the keyboard/MIDI (pad 1 bottom-left, MPC layout), then a tray of pads
    that goes on forever below it: somewhere to collect sounds and pick from when making parts."""

    hit = Signal(int, float)                 # play it
    released = Signal(int)
    selection_changed = Signal(list)         # selected pad indices, most recent last
    menu = Signal(int, QPoint)
    file_dropped = Signal(int, str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWidgetResizable(True)
        self.setFrameShape(QScrollArea.NoFrame)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        inner = QWidget()
        self._grid = QGridLayout(inner)
        self._grid.setSpacing(4)
        self._grid.setContentsMargins(0, 0, 4, 0)
        self.tray_label = QLabel("TRAY · more pads for collecting sounds — click to play · Ctrl/Shift+click to "
                                 "select · drop a WAV")
        self.tray_label.setStyleSheet("color: #8a93a0; font-size: 11px; margin-top: 4px;")
        self._grid.addWidget(self.tray_label, 4, 0, 1, COLUMNS)
        self.setWidget(inner)
        self.setMinimumHeight(4 * 66 + 46)  # the whole 4×4 and a peek at the tray
        self.buttons: list[PadButton] = []
        self.selected: list[int] = [0]
        self.generating: dict[int, float] = {}
        self.set_count(KEYED + COLUMNS)
        self._decay = QTimer(self, interval=30, timeout=self._fade)
        self._decay.start()

    def set_count(self, count: int) -> None:
        """Show ``count`` pads (the 4×4, then tray rows)."""
        while len(self.buttons) < count:
            b = PadButton(len(self.buttons))
            i = b.index
            row, col = (3 - i // 4, i % 4) if i < KEYED else (5 + (i - KEYED) // COLUMNS, (i - KEYED) % COLUMNS)
            self._grid.addWidget(b, row, col)
            b.pressed.connect(self._pressed)
            b.released.connect(self.released)
            b.menu.connect(self._menu)
            b.file_dropped.connect(self.file_dropped)
            b.selected = i in self.selected
            b.generating = self.generating.get(i)
            self.buttons.append(b)
        while len(self.buttons) > count:
            b = self.buttons.pop()
            self._grid.removeWidget(b)
            b.deleteLater()
        self.tray_label.setVisible(count > KEYED)

    def show_pad(self, index: int) -> None:
        if 0 <= index < len(self.buttons):
            self.ensureWidgetVisible(self.buttons[index], 0, 20)

    def _pressed(self, index: int, velocity: float, mods) -> None:
        if mods & Qt.ControlModifier:  # add/remove from the selection; no sound
            sel = [i for i in self.selected if i != index] if index in self.selected else self.selected + [index]
            self.select(sel or [index])
            return
        if mods & Qt.ShiftModifier and self.selected:
            a, b = sorted((self.selected[-1], index))
            self.select([i for i in self.selected if not a <= i <= b] + list(range(a, b + 1)))
            return
        self.select([index])
        self.hit.emit(index, velocity)

    def _menu(self, index: int, pos: QPoint) -> None:
        # right-click inside a selection keeps it, and makes the clicked pad the one single-pad edits hit
        self.select([i for i in self.selected if i != index] + [index] if index in self.selected else [index])
        self.menu.emit(index, pos)

    def select(self, indices: list[int]) -> None:
        self.selected = list(dict.fromkeys(indices))
        for b in self.buttons:
            b.selected = b.index in self.selected
            b.update()
        self.selection_changed.emit(self.selected)

    def flash(self, index: int) -> None:
        if 0 <= index < len(self.buttons):
            self.buttons[index].flash = 1.0
            self.buttons[index].update()

    def set_generating(self, indices: list[int], progress: float | None) -> None:
        for i in indices:
            if progress is None:
                self.generating.pop(i, None)
            else:
                self.generating[i] = progress
            if 0 <= i < len(self.buttons):
                self.buttons[i].generating = progress
                self.buttons[i].update()

    def _fade(self) -> None:
        for b in self.buttons:
            if b.flash > 0:
                b.flash = max(0.0, b.flash - 0.18)
                b.update()

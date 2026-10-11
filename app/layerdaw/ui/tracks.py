"""Left-hand track headers: name (double-click to rename), mute/solo, volume, pan, menu."""

from __future__ import annotations

import math

from PySide6.QtCore import QPoint, Qt, Signal
from PySide6.QtWidgets import (QFrame, QHBoxLayout, QLabel, QLineEdit, QSlider, QStackedWidget,
                               QToolButton, QVBoxLayout, QWidget)

from .. import fx
from ..model import Project, Track
from .timeline import KIND_COLORS, ROW_H

HEADER_W = 270


def db_text(volume: float) -> str:
    return "-inf dB" if volume <= 0 else f"{20 * math.log10(volume):+.1f} dB"


class ResetSlider(QSlider):
    """Horizontal slider that jumps back to ``default`` on double-click."""

    def __init__(self, lo: int, hi: int, value: int, default: int, parent=None):
        super().__init__(Qt.Horizontal, parent)
        self.setRange(lo, hi)
        self.setValue(value)
        self.default = default

    def mouseDoubleClickEvent(self, _ev):
        self.sliderPressed.emit()
        self.setValue(self.default)


class TrackHeader(QFrame):
    about_to_change = Signal(str, object)    # undo label, coalesce key
    changed = Signal()                       # mute/solo/volume/pan changed -> remix
    renamed = Signal(Track, str)
    regenerate = Signal(Track)
    delete = Signal(Track)
    menu = Signal(Track, QPoint)
    selected = Signal(Track)
    shape = Signal(Track)   # the fx button: open the Layer panel

    def __init__(self, track: Track, selected: bool, busy: bool, parent=None):
        super().__init__(parent)
        self.track = track
        self.setFixedHeight(ROW_H)
        color = KIND_COLORS.get(track.kind, "#8e9bb0")
        self.setStyleSheet(f"TrackHeader {{ background: {'#262b34' if selected else '#1f2329'};"
                           f" border-left: 4px solid {color}; border-bottom: 1px solid #2a2f38; }}")

        self.name = QLabel()
        self.name.setStyleSheet("font-weight: 600;")
        self.name.setToolTip(self._tooltip())
        self.name.setText(self.name.fontMetrics().elidedText(
            ("[busy] " if busy else "") + track.name, Qt.ElideRight, HEADER_W - 156))
        self.editor = QLineEdit(track.name)
        self.editor.editingFinished.connect(self._finish_rename)
        self.name_stack = QStackedWidget()
        self.name_stack.addWidget(self.name)
        self.name_stack.addWidget(self.editor)
        self.name_stack.setFixedHeight(24)

        self.mute = self._toggle("M", track.muted, "Mute", "#c0504d")
        self.solo = self._toggle("S", track.solo, "Solo", "#d9a441")
        redo = self._button("⟳", "New variation of this song part (Ctrl+Z goes back)" if track.gen.get("mode") == "song"
                            else "Regenerate this layer with a new seed (keeps the old take, muted)")
        redo.setEnabled(bool(track.gen) and not busy)
        redo.clicked.connect(lambda: self.regenerate.emit(track))
        self.fx_button = self._button("fx", "")
        self.fx_button.clicked.connect(lambda: self.shape.emit(track))
        self.update_fx()
        more = self._button("⋯", "More: rename, duplicate, export, delete…")
        more.clicked.connect(lambda: self.menu.emit(track, more.mapToGlobal(more.rect().bottomLeft())))

        self.vol = ResetSlider(0, 150, int(track.volume * 100), 100)
        self.vol.setToolTip(f"Volume {db_text(track.volume)} · double-click resets")
        self.vol.sliderPressed.connect(lambda: self.about_to_change.emit("Volume", None))
        self.vol.valueChanged.connect(self._set_volume)
        self.pan = ResetSlider(-100, 100, int(track.pan * 100), 0)
        self.pan.setFixedWidth(60)
        self.pan.setToolTip(self._pan_text())
        self.pan.sliderPressed.connect(lambda: self.about_to_change.emit("Pan", None))
        self.pan.valueChanged.connect(self._set_pan)

        top = QHBoxLayout()
        top.addWidget(self.name_stack, 1)
        for w in (self.mute, self.solo, self.fx_button, redo, more):
            top.addWidget(w)
        bottom = QHBoxLayout()
        bottom.addWidget(QLabel("vol"))
        bottom.addWidget(self.vol, 1)
        bottom.addWidget(QLabel("pan"))
        bottom.addWidget(self.pan)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 4, 6, 4)
        layout.addLayout(top)
        layout.addLayout(bottom)
        for label in self.findChildren(QLabel):
            if label is not self.name:
                label.setStyleSheet("color: #7a828e; font-size: 10px;")

    def _tooltip(self) -> str:
        lines = [self.track.name]
        if self.track.prompt:
            lines.append(f"prompt: {self.track.prompt}")
        if self.track.gen.get("bpm"):
            lines.append(f"made at {self.track.gen['bpm']} BPM, {self.track.gen.get('key', '')}")
        return "\n".join(lines) + "\n(double-click to rename)"

    def _pan_text(self) -> str:
        v = int(self.track.pan * 100)
        return f"Pan {'C' if v == 0 else f'L{-v}' if v < 0 else f'R{v}'} · double-click centres"

    def update_fx(self) -> None:
        """The fx button lights up when the track's sound is shaped."""
        on = fx.is_active(self.track.fx)
        what = fx.describe(self.track.fx)
        self.fx_button.setToolTip(f"Shape this track's sound: {what}" if what else
                                  "Shape this track's sound: warmth, tone, pitch, punch, echo, reverb…")
        self.fx_button.setStyleSheet("QToolButton { background: #5ec2a8; color: black; font-weight: 700; }"
                                     if on else "")

    def mousePressEvent(self, ev):
        if ev.button() == Qt.LeftButton:
            self.selected.emit(self.track)
        super().mousePressEvent(ev)

    def _button(self, text: str, tip: str) -> QToolButton:
        b = QToolButton()
        b.setText(text)
        b.setToolTip(tip)
        b.setFixedSize(24, 22)
        return b

    def _toggle(self, text: str, on: bool, tip: str, color: str) -> QToolButton:
        b = self._button(text, tip)
        b.setCheckable(True)
        b.setChecked(on)
        b.setStyleSheet(f"QToolButton:checked {{ background: {color}; color: black; }}")
        b.toggled.connect(lambda _on, t=tip: self._toggled(t))
        return b

    def _toggled(self, what: str) -> None:
        self.about_to_change.emit(what, None)
        self.track.muted, self.track.solo = self.mute.isChecked(), self.solo.isChecked()
        self.changed.emit()

    def _set_volume(self, value: int) -> None:
        if not self.vol.isSliderDown():
            self.about_to_change.emit("Volume", ("vol", id(self.track)))
        self.track.volume = value / 100
        self.vol.setToolTip(f"Volume {db_text(self.track.volume)} · double-click resets")
        self.changed.emit()

    def _set_pan(self, value: int) -> None:
        if not self.pan.isSliderDown():
            self.about_to_change.emit("Pan", ("pan", id(self.track)))
        self.track.pan = value / 100
        self.pan.setToolTip(self._pan_text())
        self.changed.emit()

    def start_rename(self) -> None:
        self.editor.setText(self.track.name)
        self.name_stack.setCurrentWidget(self.editor)
        self.editor.setFocus()
        self.editor.selectAll()

    def _finish_rename(self) -> None:
        if self.name_stack.currentWidget() is not self.editor:
            return
        self.name_stack.setCurrentWidget(self.name)
        new = self.editor.text().strip()
        if new and new != self.track.name:
            self.renamed.emit(self.track, new)

    def mouseDoubleClickEvent(self, _ev):
        self.start_rename()

    def contextMenuEvent(self, ev):
        self.menu.emit(self.track, ev.globalPos())


class PendingHeader(QFrame):
    """Placeholder header for a layer that's still being generated."""

    cancel = Signal()

    def __init__(self, label: str, parent=None):
        super().__init__(parent)
        self.setFixedHeight(ROW_H)
        self.setStyleSheet("PendingHeader { background: #1f2433; border-left: 4px solid #7aa2f7;"
                           " border-bottom: 1px solid #2a2f38; }")
        text = QLabel(f"Generating {label}…")
        text.setStyleSheet("color: #c8d3f5; font-weight: 600;")
        stop = QToolButton()
        stop.setText("Cancel")
        stop.setToolTip("Stop generating (cancels everything queued; the engine restarts)")
        stop.clicked.connect(self.cancel)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(8, 4, 6, 4)
        layout.addWidget(text, 1)
        layout.addWidget(stop)


class TrackHeaders(QWidget):
    about_to_change = Signal(str, object)
    changed = Signal()
    renamed = Signal(Track, str)
    regenerate = Signal(Track)
    delete = Signal(Track)
    menu = Signal(Track, QPoint)
    cancel = Signal()
    selected = Signal(Track)
    shape = Signal(Track)   # the fx button: open the Layer panel

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedWidth(HEADER_W)
        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(0, 0, 0, 0)
        self._layout.setSpacing(0)
        self.headers: dict[int, TrackHeader] = {}

    def rebuild(self, project: Project, selected: int | None, pending: list[dict]) -> None:
        while (item := self._layout.takeAt(0)) is not None:
            if item.widget():
                item.widget().deleteLater()
        busy_rows = {j["row"] for j in pending if j["row"] is not None}
        self.headers = {}
        for row, track in enumerate(project.tracks):
            header = TrackHeader(track, row == selected, row in busy_rows)
            for name in ("about_to_change", "changed", "renamed", "regenerate", "delete", "menu", "selected", "shape"):
                getattr(header, name).connect(getattr(self, name))
            self.headers[id(track)] = header
            self._layout.addWidget(header)
        new_jobs = [j for j in pending if j["row"] is None and not j.get("pads")]
        for job in new_jobs:
            placeholder = PendingHeader(job["label"])
            placeholder.cancel.connect(self.cancel)
            self._layout.addWidget(placeholder)
        self._layout.addStretch(1)
        self.setFixedHeight(max(len(project.tracks) + len(new_jobs), 1) * ROW_H)

    def start_rename(self, track: Track) -> None:
        if id(track) in self.headers:
            self.headers[id(track)].start_rename()

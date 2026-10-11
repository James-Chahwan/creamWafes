"""Bar ruler + track lanes: waveforms, clip dragging, bar selection, playhead, generation ghosts."""

from __future__ import annotations

import numpy as np
from PySide6.QtCore import QPoint, QRectF, Qt, Signal
from PySide6.QtGui import QBrush, QColor, QPainter, QPen
from PySide6.QtWidgets import QWidget

from ..model import SR, Clip, Project, Track

ROW_H = 76
RULER_H = 26
PEAK_BIN = 256  # samples per cached waveform bin
DRAG_START_PX = 6

KIND_COLORS = {
    "vocals": "#e8697d", "backing_vocals": "#d98aa0", "drums": "#e0a14a", "bass": "#7f8cf0",
    "guitar": "#c9b458", "keyboard": "#5ec2a8", "percussion": "#d9874a", "strings": "#a978d6",
    "synth": "#4fb3e0", "fx": "#9aa3ad", "brass": "#e0c060", "woodwinds": "#8fcf6a",
    "song": "#8e9bb0", "audio": "#8e9bb0", "hook": "#e86bb0",
}
BG, ROW_ALT, GRID, BAR_LINE = QColor("#1b1e24"), QColor("#20242b"), QColor("#2a2f38"), QColor("#3a404b")
SELECTION, PLAYHEAD, TEXT = QColor(120, 170, 255, 45), QColor("#ff5f56"), QColor("#c8ced8")
EMPTY_HELP = [
    "Empty song",
    "",
    "＋ Add layer   make a first part on its own (drums, bass, keys, vocals…)",
    "✦ New idea     make a whole groove from a description",
    "or drag audio files in here",
]


def clip_envelope(clip: Clip, pps: float) -> tuple[np.ndarray, np.ndarray]:
    """Min/max per screen pixel at the given zoom.

    Cached on the clip together with the exact audio array it was computed from, so a
    replaced/cropped array (or a recycled ``id()``) can never show a stale waveform.
    """
    cache = clip.__dict__.get("_wave")
    if cache is None or cache["audio"] is not clip.audio:
        mono = clip.audio.mean(axis=1)
        n = len(mono) // PEAK_BIN
        bins = mono[:n * PEAK_BIN].reshape(n, PEAK_BIN) if n else np.zeros((1, 1), np.float32)
        cache = clip.__dict__["_wave"] = {"audio": clip.audio, "lo": bins.min(axis=1),
                                          "hi": bins.max(axis=1), "env": {}}
    env = cache["env"]
    if pps not in env:
        if len(env) > 16:
            env.clear()
        width = max(1, int(len(clip.audio) / SR * pps))
        idx = np.linspace(0, len(cache["lo"]), width, endpoint=False).astype(int)
        env[pps] = (np.minimum.reduceat(cache["lo"], idx), np.maximum.reduceat(cache["hi"], idx))
    return env[pps]


class TimelineState:
    """View state shared by the ruler, lanes and window."""

    def __init__(self):
        self.pps = 40.0  # pixels per second
        self.selection: tuple[float, float] | None = None
        self.selected_track: int | None = None
        self.selected_clip: Clip | None = None
        self.playhead = 0.0
        self.pending: list[dict] = []  # {"label", "progress", "row" (int|None for a new row)}

    def x(self, seconds: float) -> float:
        return seconds * self.pps

    def t(self, x: float) -> float:
        return max(0.0, x / self.pps)


class _TimeDragMixin:
    """Click = seek, drag = select a bar-snapped range. Hosts define ``seek``/``selection_changed``."""

    def _press(self, x: float) -> None:
        self._drag_from = x

    def _move(self, x: float) -> None:
        if self._drag_from is None or abs(x - self._drag_from) < DRAG_START_PX:
            return
        bar = self.project.bar_seconds
        a, b = sorted((self.state.t(self._drag_from), self.state.t(x)))
        self.state.selection = (np.floor(a / bar) * bar, max(np.ceil(b / bar), np.floor(a / bar) + 1) * bar)
        self.selection_changed.emit()

    def _release(self, x: float) -> None:
        if self._drag_from is not None and abs(x - self._drag_from) < DRAG_START_PX:
            self.seek.emit(self.state.t(x))
        self._drag_from = None


class Ruler(_TimeDragMixin, QWidget):
    seek = Signal(float)
    selection_changed = Signal()

    def __init__(self, project: Project, state: TimelineState, parent=None):
        super().__init__(parent)
        self.project, self.state, self.offset, self._drag_from = project, state, 0, None
        self.setFixedHeight(RULER_H)
        self.setToolTip("Click to move the playhead · drag to select bars")

    def paintEvent(self, _ev):
        p = QPainter(self)
        p.fillRect(self.rect(), QColor("#16181d"))
        bar = self.project.bar_seconds
        first = int(self.state.t(self.offset) / bar)
        p.setPen(TEXT)
        for i in range(first, first + int(self.width() / (bar * self.state.pps)) + 2):
            x = self.state.x(i * bar) - self.offset
            p.drawLine(int(x), RULER_H - 8, int(x), RULER_H)
            if bar * self.state.pps > 22 or i % 4 == 0:
                p.drawText(int(x) + 3, RULER_H - 10, str(i + 1))
        if self.state.selection:
            a, b = (self.state.x(s) - self.offset for s in self.state.selection)
            p.fillRect(QRectF(a, 0, b - a, RULER_H), QColor(120, 170, 255, 110))
        x = self.state.x(self.state.playhead) - self.offset
        p.setPen(QPen(PLAYHEAD, 2))
        p.drawLine(int(x), 0, int(x), RULER_H)

    def mousePressEvent(self, ev): self._press(ev.position().x() + self.offset)
    def mouseMoveEvent(self, ev): self._move(ev.position().x() + self.offset)
    def mouseReleaseEvent(self, ev): self._release(ev.position().x() + self.offset)


class Lanes(_TimeDragMixin, QWidget):
    seek = Signal(float)
    selection_changed = Signal()
    track_selected = Signal(int)
    zoomed = Signal()
    hscroll = Signal(int)                       # wheel delta for horizontal scrolling
    clip_edit_started = Signal()                # about to move a clip: take an undo checkpoint
    clip_moved = Signal()
    context_menu = Signal(object, object, QPoint)  # track | None, clip | None, global pos

    def __init__(self, project: Project, state: TimelineState, parent=None):
        super().__init__(parent)
        self.project, self.state, self._drag_from = project, state, None
        self._clip_drag: dict | None = None
        self.setFocusPolicy(Qt.StrongFocus)

    # --- geometry ----------------------------------------------------------------------

    def rows(self) -> int:
        return len(self.project.tracks) + sum(1 for j in self.state.pending if j["row"] is None and not j.get("pads"))

    def update_size(self, min_width: int, min_height: int = 0) -> None:
        length = max(self.project.length + 4 * self.project.bar_seconds, 60)
        self.setFixedSize(max(int(self.state.x(length)), min_width), max(self.rows() * ROW_H, min_height))

    def hit(self, pos) -> tuple[int | None, Track | None, Clip | None]:
        row = int(pos.y() // ROW_H)
        if not 0 <= row < len(self.project.tracks):
            return None, None, None
        track = self.project.tracks[row]
        t = self.state.t(pos.x())
        clip = next((c for c in reversed(track.clips) if c.start <= t < c.end), None)
        return row, track, clip

    # --- painting ----------------------------------------------------------------------

    def paintEvent(self, ev):
        p = QPainter(self)
        visible = ev.rect()
        p.fillRect(visible, BG)
        for row in range(len(self.project.tracks)):
            y = row * ROW_H
            if row == self.state.selected_track:
                p.fillRect(0, y, self.width(), ROW_H, QColor("#262b34"))
            elif row % 2:
                p.fillRect(0, y, self.width(), ROW_H, ROW_ALT)
        self._paint_grid(p, visible)
        for row, track in enumerate(self.project.tracks):
            color = QColor(KIND_COLORS.get(track.kind, "#8e9bb0"))
            if not self.project.audible(track):
                color.setAlpha(70)
            for clip in track.clips:
                self._paint_clip(p, clip, track, row * ROW_H, color, visible)
        self._paint_pending(p)
        if not self.project.tracks and not any(not j.get("pads") for j in self.state.pending):
            self._paint_empty_help(p)
        if self.state.selection:
            a, b = (self.state.x(s) for s in self.state.selection)
            p.fillRect(QRectF(a, 0, b - a, self.height()), SELECTION)
        p.setPen(QPen(PLAYHEAD, 2))
        x = int(self.state.x(self.state.playhead))
        p.drawLine(x, 0, x, self.height())

    def _paint_grid(self, p: QPainter, rect) -> None:
        beat = self.project.beat_seconds
        i = int(self.state.t(rect.left()) / beat)
        while (x := self.state.x(i * beat)) <= rect.right():
            p.setPen(BAR_LINE if i % self.project.beats_per_bar == 0 else GRID)
            if beat * self.state.pps > 8 or i % self.project.beats_per_bar == 0:
                p.drawLine(int(x), rect.top(), int(x), rect.bottom())
            i += 1

    def _paint_clip(self, p: QPainter, clip: Clip, track: Track, y: int, color: QColor, visible) -> None:
        x0, x1 = self.state.x(clip.start), self.state.x(clip.end)
        if x1 < visible.left() or x0 > visible.right():
            return
        box = QRectF(x0, y + 3, x1 - x0, ROW_H - 6)
        fill = QColor(color)
        fill.setAlpha(min(fill.alpha(), 55))
        selected = clip is self.state.selected_clip
        p.setPen(QPen(QColor("#ffffff") if selected else color, 2 if selected else 1))
        p.setBrush(fill)
        p.drawRoundedRect(box, 4, 4)
        lo, hi = clip_envelope(clip, self.state.pps)
        mid, half = box.center().y() + 5, box.height() / 2 - 9
        first = max(0, visible.left() - int(x0))
        last = min(len(lo), visible.right() + 1 - int(x0))
        if last > first:
            tops = (mid - np.clip(hi[first:last], -1, 1) * half).astype(int).tolist()
            bottoms = (mid - np.clip(lo[first:last], -1, 1) * half).astype(int).tolist()
            p.setPen(color)
            x = int(x0) + first
            for top, bottom in zip(tops, bottoms):
                p.drawLine(x, top, x, bottom)
                x += 1
        if box.width() > 40:
            p.setPen(QColor(230, 235, 240, 200 if color.alpha() > 100 else 90))
            p.drawText(box.adjusted(6, 2, -4, 0), Qt.AlignLeft | Qt.AlignTop, track.name)

    def _paint_pending(self, p: QPainter) -> None:
        new_rows = len(self.project.tracks)
        length = max(self.project.length, 8 * self.project.bar_seconds)
        for job in self.state.pending:
            if job.get("pads"):  # shown on the pads instead
                continue
            row = job["row"]
            if row is None:
                row, new_rows = new_rows, new_rows + 1
            box = QRectF(2, row * ROW_H + 3, self.state.x(length), ROW_H - 6)
            p.setPen(QPen(QColor("#7aa2f7"), 1, Qt.DashLine))
            p.setBrush(QBrush(QColor(122, 162, 247, 40), Qt.BDiagPattern))
            p.drawRoundedRect(box, 4, 4)
            p.fillRect(QRectF(box.left(), box.bottom() - 4, box.width() * job["progress"], 4), QColor("#7aa2f7"))
            p.setPen(QColor("#c8d3f5"))
            p.drawText(box.adjusted(10, 0, 0, 0), Qt.AlignVCenter | Qt.AlignLeft,
                       f"Generating {job['label']}   {int(job['progress'] * 100)}%")

    def _paint_empty_help(self, p: QPainter) -> None:
        p.setPen(QColor("#8a93a0"))
        font = p.font()
        for i, line in enumerate(EMPTY_HELP):
            font.setPointSize(14 if i == 0 else 11)
            font.setBold(i == 0)
            p.setFont(font)
            p.drawText(40, 60 + i * 26, line)

    # --- mouse -------------------------------------------------------------------------

    def mousePressEvent(self, ev):
        self.setFocus()
        row, track, clip = self.hit(ev.position())
        if row is not None:
            self.state.selected_track = row
            self.track_selected.emit(row)
        if ev.button() == Qt.RightButton:
            self.state.selected_clip = clip
            self.update()
            self.context_menu.emit(track, clip, ev.globalPosition().toPoint())
            return
        if ev.button() != Qt.LeftButton:
            return
        self.state.selected_clip = clip
        if clip is not None:
            self._clip_drag = {"clip": clip, "grab": self.state.t(ev.position().x()) - clip.start,
                               "x": ev.position().x(), "moving": False}
        else:
            self._press(ev.position().x())
        self.update()

    def mouseMoveEvent(self, ev):
        drag = self._clip_drag
        if drag is None:
            self._move(ev.position().x())
            return
        if not drag["moving"]:
            if abs(ev.position().x() - drag["x"]) < DRAG_START_PX:
                return
            drag["moving"] = True
            self.clip_edit_started.emit()
        t = self.state.t(ev.position().x()) - drag["grab"]
        if not ev.modifiers() & Qt.AltModifier:  # snap to the beat unless Alt is held
            beat = self.project.beat_seconds
            t = round(t / beat) * beat
        drag["clip"].start = max(0.0, t)
        self.update()

    def mouseReleaseEvent(self, ev):
        drag, self._clip_drag = self._clip_drag, None
        if drag is None:
            self._release(ev.position().x())
        elif drag["moving"]:
            self.clip_moved.emit()
        else:  # a plain click on a clip also moves the playhead there
            self.seek.emit(self.state.t(ev.position().x()))

    def wheelEvent(self, ev):
        if ev.modifiers() & Qt.ControlModifier:  # ctrl+wheel = zoom
            factor = 1.25 if ev.angleDelta().y() > 0 else 0.8
            self.state.pps = float(np.clip(self.state.pps * factor, 4, 600))
            self.zoomed.emit()
        elif ev.modifiers() & Qt.ShiftModifier:  # shift+wheel = scroll sideways
            self.hscroll.emit(-ev.angleDelta().y())
        else:
            ev.ignore()
            return
        ev.accept()

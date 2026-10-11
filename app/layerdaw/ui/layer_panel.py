"""The Layer panel: shape the selected track's sound (warmth, tone, cuts, pitch, punch, duck,
width, echo, reverb). Every change is non-destructive and undoable; the track's audio never changes."""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (QComboBox, QGridLayout, QHBoxLayout, QLabel, QLineEdit, QMenu, QPushButton,
                               QScrollArea, QSpinBox, QToolButton, QVBoxLayout, QWidget)

from .. import fx
from .timeline import KIND_COLORS
from .tracks import ResetSlider

LOW_CUT = (20.0, 1000.0)     # slider 0..100, log scale; 0 = off
HIGH_CUT = (500.0, 20000.0)  # slider 0..100, log scale; 100 = off


def _log(v: int, lo: float, hi: float) -> float:
    return lo * (hi / lo) ** (v / 100)


def _unlog(hz: float, lo: float, hi: float) -> int:
    import math
    return int(round(100 * math.log(max(lo, min(hi, hz)) / lo) / math.log(hi / lo)))


# key: (section, label, slider range, default slider value, slider → value, value → slider, value text, tip)
SLIDERS = {
    "warmth": ("TONE", "Warmth", (0, 100), 0, lambda v: v / 100, lambda x: int(round(x * 100)),
               lambda x: f"{x:.0%}" if x else "off", "Tape/tube saturation: thicker, rounder, softer top"),
    "tone": ("TONE", "Dark ↔ Bright", (-100, 100), 0, lambda v: v / 100, lambda x: int(round(x * 100)),
             lambda x: "neutral" if not x else f"{'brighter' if x > 0 else 'darker'} {abs(x):.0%}",
             "Tilts the sound: more lows and fewer highs, or the other way"),
    "low_cut": ("TONE", "Low cut", (0, 100), 0, lambda v: round(_log(v, *LOW_CUT)), lambda x: _unlog(x, *LOW_CUT),
                lambda x: "off" if x <= LOW_CUT[0] else f"{x:.0f} Hz", "Removes rumble/bass below this"),
    "high_cut": ("TONE", "High cut", (0, 100), 100, lambda v: round(_log(v, *HIGH_CUT)),
                 lambda x: _unlog(x, *HIGH_CUT),
                 lambda x: "off" if x >= HIGH_CUT[1] else (f"{x / 1000:.1f} kHz" if x >= 1000 else f"{x:.0f} Hz"),
                 "Removes highs above this (muffled, distant, lo-fi)"),
    "pitch": ("PITCH", "Pitch", (-12, 12), 0, int, int, lambda x: f"{x:+d} semitones" if x else "as recorded",
              "Changes the pitch, keeps the speed (Rubber Band)"),
    "punch": ("DYNAMICS", "Punch", (0, 100), 0, lambda v: v / 100, lambda x: int(round(x * 100)),
              lambda x: f"{x:.0%}" if x else "off", "Sharper attacks, tighter body (transient shaper + compression)"),
    "duck": ("DYNAMICS", "Duck", (0, 100), 0, lambda v: v / 100, lambda x: int(round(x * 100)),
             lambda x: f"{x:.0%}" if x else "off", "Dips this track every time the kick hits (sidechain pump)"),
    "width": ("STEREO", "Width", (0, 200), 100, lambda v: v / 100, lambda x: int(round(x * 100)),
              lambda x: "mono" if x == 0 else ("as recorded" if x == 1 else f"{x:.0%}"),
              "Narrower (towards mono) or wider than recorded"),
    "delay": ("SPACE", "Echo", (0, 100), 0, lambda v: v / 100, lambda x: int(round(x * 100)),
              lambda x: f"{x:.0%}" if x else "off", "Tempo-synced ping-pong echoes"),
    "reverb": ("SPACE", "Reverb", (0, 100), 0, lambda v: v / 100, lambda x: int(round(x * 100)),
               lambda x: f"{x:.0%}" if x else "off", "Puts the sound in a room"),
    "size": ("SPACE", "Room size", (0, 100), 50, lambda v: v / 100, lambda x: int(round(x * 100)),
             lambda x: "small room" if x < 0.3 else "hall" if x < 0.7 else "huge", "How big the reverb's room is"),
}
LABELS = {k: v[1] for k, v in SLIDERS.items()} | {"duck_to": "Duck to", "delay_time": "Echo time",
                                                  "bypass": "Bypass"}


class LayerPanel(QWidget):
    changed = Signal(str, object)   # fx key, new value
    preset = Signal(str)
    bars_changed = Signal(int, int)  # first, last bar (1-based, inclusive) picked for generating
    generate = Signal(int, int, str)
    find_gap = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumWidth(300)
        self.title = QLabel("No track selected")
        self.title.setStyleSheet("font-weight: 700; font-size: 14px;")
        self.title.setWordWrap(True)
        self.summary = QLabel()
        self.summary.setStyleSheet("color: #8a93a0;")
        self.summary.setWordWrap(True)
        self.empty = QLabel("Click a track's name or one of its clips to shape how it sounds.\n\n"
                            "Nothing here changes the audio files: it's applied as the song plays and "
                            "when you export, and Ctrl+Z undoes it.")
        self.empty.setWordWrap(True)
        self.empty.setStyleSheet("color: #8a93a0;")

        presets = QToolButton(text="Presets ▾")
        presets.setPopupMode(QToolButton.InstantPopup)
        menu = QMenu(presets)
        for name in fx.PRESETS:
            menu.addAction(name).triggered.connect(lambda _=False, n=name: self.preset.emit(n))
        presets.setMenu(menu)
        self.bypass = QPushButton("A/B: bypass")
        self.bypass.setCheckable(True)
        self.bypass.setToolTip("Hear the track without any of this, to compare")
        self.bypass.setStyleSheet("QPushButton:checked { background: #d9a441; color: black; }")
        self.bypass.toggled.connect(lambda on: self.changed.emit("bypass", on))
        reset = QPushButton("Reset")
        reset.setToolTip("Back to the sound as recorded")
        reset.clicked.connect(lambda: self.preset.emit("Clean (reset)"))

        self.sliders: dict[str, ResetSlider] = {}
        self.values: dict[str, QLabel] = {}
        grid = QGridLayout()
        grid.setHorizontalSpacing(6)
        grid.setVerticalSpacing(2)
        row, section = 0, None
        self.duck_to = QComboBox()
        self.duck_to.setToolTip("The track whose hits push this one down")
        self.duck_to.activated.connect(lambda _i: self.changed.emit("duck_to", self.duck_to.currentData()))
        self.delay_time = QComboBox()
        for name in fx.DELAY_BEATS:
            self.delay_time.addItem(name, name)
        self.delay_time.activated.connect(lambda _i: self.changed.emit("delay_time", self.delay_time.currentData()))
        self.pitch_note = QLabel("Install Rubber Band to change pitch: sudo dnf install rubberband")
        self.pitch_note.setStyleSheet("color: #d9a441; font-size: 11px;")
        self.pitch_note.setWordWrap(True)
        for key, (sec, label, (lo, hi), default, to_value, _to_slider, text, tip) in SLIDERS.items():
            if sec != section:
                head = QLabel(sec)
                head.setStyleSheet("color: #8a93a0; font-size: 11px; margin-top: 6px;")
                grid.addWidget(head, row, 0, 1, 3)
                row, section = row + 1, sec
                if sec == "PITCH":
                    grid.addWidget(self.pitch_note, row, 0, 1, 3)
                    row += 1
            name = QLabel(label)
            name.setToolTip(tip)
            slider = ResetSlider(lo, hi, default, default)
            slider.setToolTip(f"{tip} · double-click resets")
            value = QLabel(text(to_value(default)))
            value.setMinimumWidth(92)
            value.setStyleSheet("color: #c8ccd4;")
            slider.valueChanged.connect(lambda v, k=key: self._moved(k, v))
            grid.addWidget(name, row, 0)
            grid.addWidget(slider, row, 1)
            grid.addWidget(value, row, 2)
            self.sliders[key], self.values[key] = slider, value
            row += 1
            if key == "duck":
                grid.addWidget(QLabel("   under"), row, 0)
                grid.addWidget(self.duck_to, row, 1, 1, 2)
                row += 1
            if key == "delay":
                grid.addWidget(QLabel("   every"), row, 0)
                grid.addWidget(self.delay_time, row, 1, 1, 2)
                row += 1
        grid.setColumnStretch(1, 1)

        self.controls = QWidget()
        c = QVBoxLayout(self.controls)
        c.setContentsMargins(0, 0, 0, 0)
        top = QHBoxLayout()
        for w in (presets, self.bypass, reset):
            top.addWidget(w)
        top.addStretch(1)
        c.addLayout(top)
        c.addLayout(grid)
        c.addWidget(self._build_generate())
        c.addStretch(1)

        inner = QWidget()
        lay = QVBoxLayout(inner)
        lay.setContentsMargins(8, 6, 8, 6)
        lay.addWidget(self.title)
        lay.addWidget(self.summary)
        lay.addWidget(self.empty)
        lay.addWidget(self.controls)
        lay.addStretch(1)
        scroll = QScrollArea(widgetResizable=True)
        scroll.setFrameShape(QScrollArea.NoFrame)
        scroll.setWidget(inner)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(scroll)
        self.show_track(None)

    def _build_generate(self) -> QWidget:
        box = QWidget()
        lay = QVBoxLayout(box)
        lay.setContentsMargins(0, 8, 0, 0)
        lay.setSpacing(4)
        head = QLabel("GENERATE PART OF THIS LAYER")
        head.setStyleSheet("color: #8a93a0; font-size: 11px;")
        self.gen_from, self.gen_to = QSpinBox(), QSpinBox()
        for spin, value in ((self.gen_from, 1), (self.gen_to, 4)):
            spin.setRange(1, 999)
            spin.setValue(value)
        self.find_gap_btn = QPushButton("Find a gap")
        self.find_gap_btn.setToolTip("Pick the longest stretch of bars where this layer is silent")
        self.find_gap_btn.clicked.connect(self.find_gap)
        bars = QHBoxLayout()
        for w in (QLabel("Bars"), self.gen_from, QLabel("to"), self.gen_to, self.find_gap_btn):
            bars.addWidget(w)
        bars.addStretch(1)
        self.gen_prompt = QLineEdit()
        self.gen_prompt.setPlaceholderText("what goes there (optional): busier hats, a fill, a drop…")
        self.gen_prompt.returnPressed.connect(self._generate)
        self.gen_btn = QPushButton()
        self.gen_btn.setStyleSheet("QPushButton { background: #4f7fd9; color: white; padding: 5px; font-weight: 700; }"
                                   "QPushButton:disabled { background: #39414d; color: #8a93a0; }")
        self.gen_btn.clicked.connect(self._generate)
        hint = QLabel("Or drag on the ruler. The AI redoes those bars of this layer, hearing the rest of it, and "
                      "fills silent bars too. You get a new take (this one is muted, not lost) with the same "
                      "sound settings.")
        hint.setWordWrap(True)
        hint.setStyleSheet("color: #7a828e; font-size: 11px;")
        lay.addWidget(head)
        lay.addLayout(bars)
        lay.addWidget(self.gen_prompt)
        lay.addWidget(self.gen_btn)
        lay.addWidget(hint)
        self._label_generate()
        for spin in (self.gen_from, self.gen_to):  # only now: editing them updates the button
            spin.valueChanged.connect(self._bars_edited)
        return box

    def _bars_edited(self, _v: int) -> None:
        if self.gen_to.value() < self.gen_from.value():
            spin = self.gen_to if self.sender() is self.gen_from else self.gen_from
            spin.blockSignals(True)
            spin.setValue(self.gen_from.value() if spin is self.gen_to else self.gen_to.value())
            spin.blockSignals(False)
        self._label_generate()
        self.bars_changed.emit(self.gen_from.value(), self.gen_to.value())

    def _label_generate(self) -> None:
        a, b = self.gen_from.value(), self.gen_to.value()
        self.gen_btn.setText(f"✦ Generate bar {a}" if a == b else f"✦ Generate bars {a}–{b}")

    def _generate(self) -> None:
        if self.gen_btn.isEnabled():
            self.generate.emit(self.gen_from.value(), self.gen_to.value(), self.gen_prompt.text().strip())

    def set_bars(self, first: int, last: int) -> None:
        for spin, v in ((self.gen_from, first), (self.gen_to, max(first, last))):
            spin.blockSignals(True)
            spin.setValue(v)
            spin.blockSignals(False)
        self._label_generate()

    def set_engine_ready(self, ready: bool) -> None:
        self.gen_btn.setEnabled(ready)
        self.gen_btn.setToolTip("" if ready else "The AI engine is still loading")

    def _moved(self, key: str, v: int) -> None:
        _sec, _label, _range, _default, to_value, _to_slider, text, _tip = SLIDERS[key]
        value = to_value(v)
        self.values[key].setText(text(value))
        self.changed.emit(key, value)

    def show_track(self, track, others: list[str] | None = None, auto_duck: str | None = None,
                   pitch_ok: bool = True) -> None:
        self.empty.setVisible(track is None)
        self.controls.setVisible(track is not None)
        if track is None:
            self.title.setText("Layer")
            self.summary.setText("")
            return
        color = KIND_COLORS.get(track.kind, "#8e9bb0")
        self.title.setText(f"<span style='color:{color}'>■</span> {track.name}")
        self.summary.setText(fx.describe(track.fx) or "as recorded")
        s = fx.settings(track.fx)
        for key, slider in self.sliders.items():
            to_slider, text = SLIDERS[key][5], SLIDERS[key][6]
            slider.blockSignals(True)
            slider.setValue(to_slider(s[key]))
            slider.blockSignals(False)
            self.values[key].setText(text(s[key]))
        self.bypass.blockSignals(True)
        self.bypass.setChecked(bool(s["bypass"]))
        self.bypass.blockSignals(False)
        self.duck_to.clear()
        self.duck_to.addItem(f"Auto ({auto_duck})" if auto_duck else "Auto (the kick)", "")
        for name in others or []:
            self.duck_to.addItem(name, name)
        self.duck_to.setCurrentIndex(max(0, self.duck_to.findData(s["duck_to"])))
        self.delay_time.setCurrentIndex(max(0, self.delay_time.findData(s["delay_time"])))
        self.sliders["pitch"].setEnabled(pitch_ok)
        self.pitch_note.setVisible(not pitch_ok)

    def set_summary(self, track) -> None:
        self.summary.setText(fx.describe(track.fx) or "as recorded")

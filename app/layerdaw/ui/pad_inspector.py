"""The inspector beside the pads: everything about the selected pad (or pads), in one place."""

from __future__ import annotations

import numpy as np
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (QButtonGroup, QCheckBox, QComboBox, QHBoxLayout, QLabel, QLineEdit, QMenu,
                               QPushButton, QStackedWidget, QToolButton, QVBoxLayout, QWidget)

from ..model import Pad
from .pad_grid import CATEGORIES, PAD_TO_KEY

FLAVOURS = ["triad", "7th", "sus2", "sus4", "add9"]
MODES = {"sample": [("oneshot", "One-shot"), ("hold", "Hold"), ("loop", "Loop"), ("playon", "Play-on")],
         "chord": [("hold", "Hold"), ("oneshot", "One-shot")],
         "note": [("hold", "Hold"), ("oneshot", "One-shot")]}
MODE_TIPS = {"oneshot": "Plays once, however long you hold", "hold": "Sustains while you hold the pad",
             "loop": "Repeats the slice while you hold", "playon": "The original audio keeps playing while you hold"}
REPLACE = [("The hit at the playhead (selected clip)", "playhead"), ("The selected bars", "bars"),
           ("The whole selected clip", "clip"), ("An audio file…", "file"), (None, None),
           ("◀ Previous hit from the same source", "prev"), ("Next hit from the same source ▶", "next"),
           (None, None), ("Clear this pad", "clear")]
CHOPS = [("Kit — one of each drum sound", "kit"), ("At each hit, in order", "hits"), ("Every beat", "beat"),
         ("Every half beat", "half"), ("Every bar (loops)", "bar"), (None, None),
         ("An audio file from disk…", "file")]


def _menu_button(text: str, items, signal, tip: str = "") -> QToolButton:
    b = QToolButton(text=text)
    b.setPopupMode(QToolButton.InstantPopup)
    b.setToolTip(tip)
    menu = QMenu(b)
    for label, value in items:
        if label is None:
            menu.addSeparator()
        else:
            menu.addAction(label).triggered.connect(lambda _=False, v=value: signal.emit(v))
    b.setMenu(menu)
    return b


def _row(*widgets, stretch=True) -> QHBoxLayout:
    row = QHBoxLayout()
    row.setSpacing(4)
    for w in widgets:
        row.addWidget(w) if isinstance(w, QWidget) else row.addLayout(w)
    if stretch:
        row.addStretch(1)
    return row


def _section(text: str) -> QLabel:
    label = QLabel(text)
    label.setStyleSheet("color: #8a93a0; font-size: 11px; margin-top: 4px;")
    return label


class PadInspector(QWidget):
    # one pad
    renamed = Signal(str)
    category_set = Signal(str)
    generate = Signal()
    replace = Signal(str)
    audition = Signal()
    play_set = Signal(str)
    mode_set = Signal(str)
    degree_set = Signal(int)
    flavour_set = Signal(str)
    strum_set = Signal(bool)
    pitch_step = Signal(int)
    gain_step = Signal(float)
    # several pads
    make_song = Signal()
    kit = Signal()
    hooks = Signal()
    progression = Signal()
    chop = Signal(str)
    play_all = Signal(str)
    mode_all = Signal(str)
    clear_all = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumWidth(380)
        self.stack = QStackedWidget()
        self.stack.addWidget(self._build_single())
        self.stack.addWidget(self._build_multi())
        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 0, 0, 0)
        layout.addWidget(self.stack)

    # --- one pad -----------------------------------------------------------------------------
    def _build_single(self) -> QWidget:
        page = QWidget()
        self.title = QLabel()
        self.title.setStyleSheet("font-weight: 700; font-size: 14px;")
        self.name = QLineEdit()
        self.name.setPlaceholderText("name")
        self.name.editingFinished.connect(lambda: self.renamed.emit(self.name.text().strip()))
        self.category = QComboBox()
        self.category.addItems(CATEGORIES)
        self.category.setToolTip("What this sound is (colour, and its part when you make a song)")
        self.category.activated.connect(lambda _i: self.category_set.emit(self.category.currentText()))
        gen = QPushButton("✦ Generate…")
        gen.setToolTip("Describe a sound; pick the best of 4 candidates")
        gen.clicked.connect(self.generate)
        self.replace_btn = _menu_button("Replace ▾", REPLACE, self.replace, "Put a different sound on this pad")
        play = QPushButton("▶")
        play.setFixedWidth(34)
        play.setToolTip("Hear this pad")
        play.clicked.connect(self.audition)
        self.info = QLabel()
        self.info.setStyleSheet("color: #8a93a0;")

        self.play_buttons: dict[str, QToolButton] = {}
        self._play_group = QButtonGroup(self)
        play_row = []
        for value, text, tip in (("sample", "Sample", "Plays the sound itself"),
                                 ("chord", "Chord", "Plays a chord of the song's key, using this sound"),
                                 ("note", "Note", "Plays one note of the key (808s, bass lines, leads)")):
            b = QToolButton(text=text)
            b.setCheckable(True)
            b.setToolTip(tip)
            b.setStyleSheet("QToolButton:checked { background: #4f7fd9; color: white; }")
            b.clicked.connect(lambda _=False, v=value: self.play_set.emit(v))
            self._play_group.addButton(b)
            self.play_buttons[value] = b
            play_row.append(b)

        self.mode_buttons: dict[str, QToolButton] = {}
        self._mode_group = QButtonGroup(self)
        mode_row = []
        for value in ("oneshot", "hold", "loop", "playon"):
            b = QToolButton(text={"oneshot": "One-shot", "hold": "Hold", "loop": "Loop", "playon": "Play-on"}[value])
            b.setCheckable(True)
            b.setToolTip(MODE_TIPS[value])
            b.setStyleSheet("QToolButton:checked { background: #4f7fd9; color: white; }")
            b.clicked.connect(lambda _=False, v=value: self.mode_set.emit(v))
            self._mode_group.addButton(b)
            self.mode_buttons[value] = b
            mode_row.append(b)

        self.degree = QComboBox()
        self.degree.setMinimumWidth(140)
        self.degree.activated.connect(lambda i: self.degree_set.emit(i))
        self.flavour_buttons: dict[str, QToolButton] = {}
        self._flavour_group = QButtonGroup(self)
        flav_row = []
        for f in FLAVOURS:
            b = QToolButton(text=f)
            b.setCheckable(True)
            b.setStyleSheet("QToolButton:checked { background: #a978d6; color: black; }")
            b.clicked.connect(lambda _=False, v=f: self.flavour_set.emit(v))
            self._flavour_group.addButton(b)
            self.flavour_buttons[f] = b
            flav_row.append(b)
        self.strum = QCheckBox("Strum")
        self.strum.toggled.connect(self.strum_set)
        self.degree_label = QLabel("Chord")
        self.degree_label.setMinimumWidth(44)
        self.chord_box = QWidget()
        cb = QVBoxLayout(self.chord_box)
        cb.setContentsMargins(0, 0, 0, 0)
        cb.addLayout(_row(self.degree_label, self.degree, self.strum))
        cb.addLayout(_row(*flav_row))
        self.chord_hint = QLabel("hold G 7th · H sus4 · B sus2 · N add9 while playing")
        self.chord_hint.setStyleSheet("color: #7a828e; font-size: 11px;")
        cb.addWidget(self.chord_hint)

        tune = []
        for text, sig, val, tip in (("♭", self.pitch_step, -1, "Down a semitone"), ("♯", self.pitch_step, 1, "Up a semitone"),
                                    ("−", self.gain_step, 1 / 1.413, "3 dB quieter"), ("+", self.gain_step, 1.413, "3 dB louder")):
            b = QToolButton(text=text)
            b.setToolTip(tip)
            b.clicked.connect(lambda _=False, s_=sig, v=val: s_.emit(v))
            tune.append(b)
        self.levels = QLabel()
        self.levels.setStyleSheet("color: #8a93a0;")

        lay = QVBoxLayout(page)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(3)
        lay.addLayout(_row(self.title, self.name, self.category, stretch=False))
        lay.addWidget(_section("SOUND"))
        lay.addLayout(_row(gen, self.replace_btn, play, self.info))
        lay.addWidget(_section("PLAY AS"))
        lay.addLayout(_row(*play_row))
        lay.addWidget(self.chord_box)
        lay.addLayout(_row(*mode_row))
        lay.addWidget(_section("TUNE · VOLUME"))
        lay.addLayout(_row(*tune, self.levels))
        lay.addStretch(1)
        return page

    # --- several pads ------------------------------------------------------------------------
    def _build_multi(self) -> QWidget:
        page = QWidget()
        self.multi_title = QLabel()
        self.multi_title.setStyleSheet("font-weight: 700; font-size: 14px;")
        self.multi_title.setWordWrap(True)
        song = QPushButton("✦ Make a song from these…")
        song.setStyleSheet("QPushButton { background: #4f7fd9; color: white; padding: 7px; font-weight: 700; }")
        song.setToolTip("Arrange exactly these sounds into a full song: drums, chords, bass, hooks")
        song.clicked.connect(self.make_song)
        kit = QPushButton("✦ Kit onto these…")
        kit.clicked.connect(self.kit)
        hooks = QPushButton("✦ Hooks onto these…")
        hooks.clicked.connect(self.hooks)
        prog = QPushButton("✦ Progression onto these…")
        prog.setToolTip("Write a chord progression for a mood and lay it across these pads as chord pads")
        prog.clicked.connect(self.progression)
        chop = _menu_button("Chop onto these ▾", CHOPS, self.chop,
                            "Slice the clip selected on the timeline (or a file) across these pads")
        play_all = _menu_button("Play as ▾", [("Sample", "sample"), ("Chord", "chord"), ("Note", "note")], self.play_all)
        mode_all = _menu_button("Mode ▾", [(t, v) for v, t in MODES["sample"]], self.mode_all)
        clear = QPushButton("Clear")
        clear.clicked.connect(self.clear_all)
        hint = QLabel("Click a pad to work on just that one · Ctrl+click adds/removes · Shift+click a range")
        hint.setStyleSheet("color: #7a828e; font-size: 11px;")
        hint.setWordWrap(True)
        lay = QVBoxLayout(page)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(self.multi_title)
        lay.addWidget(song)
        lay.addWidget(_section("FILL THESE PADS"))
        lay.addLayout(_row(kit, hooks))
        lay.addLayout(_row(prog))
        lay.addLayout(_row(chop))
        lay.addWidget(_section("ALL OF THEM"))
        lay.addLayout(_row(play_all, mode_all, clear))
        lay.addWidget(hint)
        lay.addStretch(1)
        return page

    # --- showing state ----------------------------------------------------------------------
    def show_single(self, index: int, pad: Pad, degree_labels: list[str], held_flavour: str | None) -> None:
        self.stack.setCurrentIndex(0)
        self.title.setText(f"Pad {PAD_TO_KEY[index]}")
        if not self.name.hasFocus():
            self.name.setText(pad.name)
        if self.category.findText(pad.kind) < 0:  # e.g. "guitar" from a track it was loaded from
            self.category.addItem(pad.kind)
        self.category.setCurrentText(pad.kind)
        self.info.setText(f"{pad.clip.length:.2f}s" if pad.clip is not None else
                          ("built-in synth" if pad.play != "sample" else "empty"))
        self.play_buttons[pad.play].setChecked(True)
        self.chord_box.setVisible(pad.play in ("chord", "note"))
        for b in self.flavour_buttons.values():
            b.setVisible(pad.play == "chord")
        self.strum.setVisible(pad.play == "chord")
        self.chord_hint.setVisible(pad.play == "chord")
        self.degree_label.setText("Chord" if pad.play == "chord" else "Note")
        self.degree.blockSignals(True)
        self.degree.clear()
        self.degree.addItems(degree_labels)
        self.degree.setCurrentIndex(min(pad.degree, len(degree_labels) - 1))
        self.degree.blockSignals(False)
        (self.flavour_buttons.get(held_flavour or pad.flavour) or self.flavour_buttons["triad"]).setChecked(True)
        self.strum.blockSignals(True)
        self.strum.setChecked(pad.strum)
        self.strum.blockSignals(False)
        allowed = {v for v, _ in MODES[pad.play]}
        for mode, b in self.mode_buttons.items():
            b.setVisible(mode in allowed)
            b.setChecked(pad.mode == mode)
        db = 20 * np.log10(max(pad.gain, 1e-6))
        self.levels.setText(f"{pad.pitch:+d} st · {db:+.0f} dB")

    def show_multi(self, indices: list[int]) -> None:
        self.stack.setCurrentIndex(1)
        keys = " ".join(PAD_TO_KEY[i] for i in sorted(indices, key=lambda i: (-(i // 4), i % 4)))
        self.multi_title.setText(f"{len(indices)} pads selected: {keys}")

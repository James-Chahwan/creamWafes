"""The pads dock: a slim transport bar, the 4×4 grid, and the inspector for the selected pad(s)."""

from __future__ import annotations

from PySide6.QtCore import Signal
from PySide6.QtWidgets import QCheckBox, QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget

from .pad_grid import PadGrid
from .pad_inspector import PadInspector


class PadPanel(QWidget):
    record_toggled = Signal(bool)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.grid = PadGrid()
        self.inspector = PadInspector()
        self.rec = QPushButton("● Record pads")
        self.rec.setCheckable(True)
        self.rec.setToolTip("Plays the song and records what you play onto a new track (stop to finish)")
        self.rec.setStyleSheet("QPushButton:checked { background: #c0392b; color: white; }")
        self.rec.toggled.connect(self.record_toggled)
        self.quantize = QCheckBox("Quantize 1/16")
        self.quantize.setChecked(True)
        select_all = QPushButton("Select all")
        select_all.clicked.connect(lambda: self.grid.select(list(range(16))))
        self.midi_label = QLabel("MIDI: …")
        self.midi_label.setStyleSheet("color: #8a93a0;")
        hint = QLabel("keys 1234 · QWER · ASDF · ZXCV  ·  Shift = softer  ·  drop a WAV on a pad to load it")
        hint.setStyleSheet("color: #7a828e; font-size: 11px;")
        bar = QHBoxLayout()
        for w in (self.rec, self.quantize, select_all, self.midi_label):
            bar.addWidget(w)
        bar.addStretch(1)
        bar.addWidget(hint)
        body = QHBoxLayout()
        body.addWidget(self.grid, 3)
        body.addWidget(self.inspector, 2)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 4, 6, 6)
        layout.addLayout(bar)
        layout.addLayout(body, 1)

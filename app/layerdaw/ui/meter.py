"""Stereo peak meter with a clip light (click to reset)."""

from __future__ import annotations

import math

from PySide6.QtCore import QRectF
from PySide6.QtGui import QColor, QPainter
from PySide6.QtWidgets import QWidget

FLOOR_DB = -48.0


class Meter(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(130, 22)
        self.setToolTip("Master level. Red light = clipping (over 0 dB); click to reset.")
        self.levels = [FLOOR_DB, FLOOR_DB]
        self.holds = [FLOOR_DB, FLOOR_DB]
        self.clipped = False

    def push(self, peaks) -> None:
        for ch in range(2):
            db = 20 * math.log10(max(float(peaks[ch]), 1e-6))
            self.levels[ch] = max(db, self.levels[ch] - 1.5)  # fast attack, gentle fall
            self.holds[ch] = max(db, self.holds[ch] - 0.3)
            self.clipped |= peaks[ch] > 1.0
        self.update()

    def mousePressEvent(self, _ev):
        self.clipped = False
        self.update()

    def paintEvent(self, _ev):
        p = QPainter(self)
        p.fillRect(self.rect(), QColor("#121418"))
        width = self.width() - 18
        for ch in range(2):
            y = 3 + ch * 9
            p.fillRect(QRectF(2, y, width, 7), QColor("#22262d"))
            frac = (min(self.levels[ch], 0) - FLOOR_DB) / -FLOOR_DB
            if frac > 0:
                color = "#4cc38a" if self.levels[ch] < -12 else "#d9c34a" if self.levels[ch] < -3 else "#e8834a"
                p.fillRect(QRectF(2, y, width * frac, 7), QColor(color))
            hold = (min(self.holds[ch], 0) - FLOOR_DB) / -FLOOR_DB
            p.fillRect(QRectF(2 + width * hold - 1, y, 2, 7), QColor("#c8ced8"))
        p.fillRect(QRectF(self.width() - 13, 4, 10, 14), QColor("#ff4d4d" if self.clipped else "#3a2226"))

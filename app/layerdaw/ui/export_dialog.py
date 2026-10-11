"""File → Export song…: the whole mix (or part of it) as one MP3/WAV/FLAC/OGG/Opus/AIFF/M4A file."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtWidgets import (QCheckBox, QComboBox, QDialog, QDialogButtonBox, QFileDialog, QFormLayout,
                               QHBoxLayout, QLabel, QLineEdit, QPushButton)

from .. import export

EXPORT_DIR = Path.home() / "Music" / "LayerDAW" / "exports"
RATES = [("44.1 kHz (CD, streaming)", 44100), ("48 kHz (video, as made)", 48000)]
# rough bytes per second, for the size estimate
_SIZE = {"mp3": lambda q, r: q * 125, "m4a": lambda q, r: q * 125, "wav": lambda q, r: r * 2 * _BYTES[q],
         "aiff": lambda q, r: r * 2 * _BYTES[q], "flac": lambda q, r: r * 2 * _BYTES[q] * 0.6,
         "ogg": lambda q, r: {0.1: 32000, 0.4: 20000, 0.7: 12000}.get(q, 20000),
         "opus": lambda q, r: {0.1: 24000, 0.4: 16000, 0.7: 8000}.get(q, 16000)}
_BYTES = {"PCM_16": 2, "PCM_24": 3, "FLOAT": 4}


class ExportDialog(QDialog):
    def __init__(self, window, prefs: dict | None = None):
        super().__init__(window)
        self.w = window
        prefs = prefs or {}
        project = window.project
        bar = project.bar_seconds
        self.setWindowTitle("Export song")
        self.setMinimumWidth(560)
        self.name = QLineEdit(project.name)
        self.folder = QLineEdit(prefs.get("folder", str(EXPORT_DIR)))
        choose = QPushButton("Choose…")
        choose.clicked.connect(self._choose)
        folder_row = QHBoxLayout()
        folder_row.addWidget(self.folder, 1)
        folder_row.addWidget(choose)
        self.format = QComboBox()
        for key in export.available():
            self.format.addItem(export.FORMATS[key][0], key)
        self.format.setCurrentIndex(max(0, self.format.findData(prefs.get("format", "mp3"))))
        self.quality = QComboBox()
        self.rate = QComboBox()
        for text, value in RATES:
            self.rate.addItem(text, value)
        self.rate.setCurrentIndex(max(0, self.rate.findData(prefs.get("rate", 44100))))
        self.range = QComboBox()
        self.range.addItem(f"Whole song (0:00 – {self._clock(project.length)})", ("all", 0.0, project.length))
        if window.state.selection:
            a, b = window.state.selection
            self.range.addItem(f"Selected bars {round(a / bar) + 1}–{round(b / bar)}", ("sel", a, b))
            self.range.setCurrentIndex(1)
        pos = window.player.position
        if 0 < pos < project.length - 0.5:
            start = int(pos // bar) * bar
            self.range.addItem(f"From the playhead (bar {int(pos // bar) + 1}) to the end", ("from", start, project.length))
        self.level = QComboBox()
        for value, text in export.LEVELS:
            self.level.addItem(text, value)
        self.level.setCurrentIndex(max(0, self.level.findData(prefs.get("level", "peak"))))
        self.level.setToolTip("Streaming services play everything at about −14 LUFS: export at that and it "
                              "won't be turned down. Every option except 'as mixed' keeps peaks at −1 dB.")
        self.tails = QCheckBox("Let echoes and reverb ring out at the end")
        self.tails.setChecked(prefs.get("tails", True))
        self.title = QLineEdit(prefs.get("title", project.name))
        self.artist = QLineEdit(prefs.get("artist", ""))
        self.artist.setPlaceholderText("optional")
        self.summary = QLabel()
        self.summary.setWordWrap(True)
        self.summary.setStyleSheet("color: #c8ccd4;")
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.button(QDialogButtonBox.Ok).setText("Export")
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        form = QFormLayout(self)
        form.addRow("File name", self.name)
        form.addRow("Folder", folder_row)
        form.addRow("Format", self.format)
        form.addRow("Quality", self.quality)
        form.addRow("Sample rate", self.rate)
        form.addRow("What", self.range)
        form.addRow("Level", self.level)
        form.addRow("", self.tails)
        form.addRow("Title", self.title)
        form.addRow("Artist", self.artist)
        form.addRow(self.summary)
        form.addRow(buttons)
        for w in (self.quality, self.rate, self.range):  # (format → _format_changed, which updates too)
            w.currentIndexChanged.connect(lambda _i: self._update())
        self.name.textChanged.connect(lambda _t: self._update())
        self.folder.textChanged.connect(lambda _t: self._update())
        self._format_changed(prefs.get("quality"))
        self.format.currentIndexChanged.connect(lambda _i: self._format_changed())

    @staticmethod
    def _clock(seconds: float) -> str:
        return f"{int(seconds // 60)}:{int(seconds % 60):02d}"

    def _choose(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "Export to", self.folder.text())
        if folder:
            self.folder.setText(folder)

    def _format_changed(self, quality=None) -> None:
        fmt = self.format.currentData()
        _label, _ext, qualities, default = export.FORMATS[fmt]
        self.quality.blockSignals(True)
        self.quality.clear()
        for text, value in qualities:
            self.quality.addItem(text, value)
        index = self.quality.findData(quality) if quality is not None else -1
        self.quality.setCurrentIndex(index if index >= 0 else default)
        self.quality.blockSignals(False)
        fixed = export.fixed_rate(fmt)
        self.rate.setEnabled(fixed is None)
        if fixed:
            self.rate.setCurrentIndex(self.rate.findData(fixed))
        self._update()

    def path(self) -> Path:
        fmt = self.format.currentData()
        name = "".join(c if c.isalnum() or c in " -_+().'&" else "_" for c in self.name.text().strip()) or "song"
        return Path(self.folder.text()).expanduser() / f"{name}{export.FORMATS[fmt][1]}"

    def _update(self) -> None:
        kind, a, b = self.range.currentData()
        self.tails.setEnabled(kind != "sel")  # a selection stops where it stops
        fmt, q, rate = self.format.currentData(), self.quality.currentData(), self.rate.currentData()
        size = _SIZE[fmt](q, rate) * (b - a) / 1e6
        self.summary.setText(f"→ {self.path()}   ·   {self._clock(b - a)}   ·   about {size:.1f} MB")

    def options(self) -> dict:
        kind, a, b = self.range.currentData()
        return {"path": self.path(), "format": self.format.currentData(), "quality": self.quality.currentData(),
                "rate": self.rate.currentData(), "range": (kind, a, b), "level": self.level.currentData(),
                "tails": self.tails.isChecked() and kind != "sel", "title": self.title.text().strip(),
                "artist": self.artist.text().strip()}

    def prefs(self) -> dict:
        o = self.options()
        return {"folder": self.folder.text(), "format": o["format"], "quality": o["quality"], "rate": o["rate"],
                "level": o["level"], "tails": self.tails.isChecked(), "artist": o["artist"]}

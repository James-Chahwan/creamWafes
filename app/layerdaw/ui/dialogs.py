"""Dialogs for asking the engine for a new layer or a fresh starting idea."""

from __future__ import annotations

from PySide6.QtWidgets import (QCheckBox, QComboBox, QDialog, QDialogButtonBox, QFormLayout,
                               QLabel, QLineEdit, QPlainTextEdit, QRadioButton, QSpinBox, QVBoxLayout)

from ..model import LAYER_TYPES, STEM_OF, VOCAL_TYPES

LYRICS_HINT = "[Verse]\nfirst line of the verse\nsecond line\n\n[Chorus]\n..."
_MEMORY: dict[str, dict] = {}  # last accepted values per dialog type, for this session


class _GenerateDialog(QDialog):
    def __init__(self, title: str, parent=None):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setMinimumWidth(480)
        self.form = QFormLayout(self)
        self.prompt = QLineEdit()
        self.lyrics = QPlainTextEdit()
        self.lyrics.setPlaceholderText(LYRICS_HINT)
        self.lyrics.setFixedHeight(130)
        self.steps = QSpinBox()
        self.steps.setRange(8, 100)
        self.steps.setValue(32)
        self.steps.setToolTip("More steps = better quality, slower. 32 is a good default.")
        self.seed = QSpinBox()
        self.seed.setRange(-1, 2**31 - 1)
        self.seed.setValue(-1)
        self.seed.setSpecialValueText("random")

    def _fields(self) -> dict:
        return {"prompt": self.prompt, "lyrics": self.lyrics, "steps": self.steps}

    def _restore(self) -> None:
        for name, value in _MEMORY.get(type(self).__name__, {}).items():
            w = self._fields().get(name)
            if isinstance(w, QLineEdit):
                w.setText(value)
            elif isinstance(w, QPlainTextEdit):
                w.setPlainText(value)
            elif isinstance(w, QSpinBox):
                w.setValue(value)
            elif isinstance(w, QComboBox):
                w.setCurrentText(value)
            elif isinstance(w, QCheckBox):
                w.setChecked(value)

    def accept(self) -> None:
        getters = {QLineEdit: "text", QPlainTextEdit: "toPlainText", QSpinBox: "value",
                   QComboBox: "currentText", QCheckBox: "isChecked"}
        _MEMORY[type(self).__name__] = {
            name: getattr(w, next(g for t, g in getters.items() if isinstance(w, t)))()
            for name, w in self._fields().items()}
        super().accept()

    def _finish(self) -> None:
        self.form.addRow("Quality (steps)", self.steps)
        self.form.addRow("Seed", self.seed)
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.button(QDialogButtonBox.Ok).setText("Generate")
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        self.form.addRow(buttons)

    def _lyrics_text(self) -> str:
        return self.lyrics.toPlainText().strip()


class LayerDialog(_GenerateDialog):
    """Add one instrument/vocal part.

    With tracks playing, the engine hears them and returns a new full-mix take with the part
    added. With nothing to listen to, it makes the part from scratch on its own track.
    """

    def __init__(self, selection: tuple[float, float] | None, bar_seconds: float, kind: str | None = None,
                 prompt: str | None = None, empty: bool = False, parent=None):
        super().__init__("Add layer" if not empty else "First layer", parent)
        self.empty = empty
        self.kind = QComboBox()
        self.kind.addItems(LAYER_TYPES)
        self.kind.setCurrentText("keyboard")
        self.kind.currentTextChanged.connect(self._kind_changed)
        self.prompt.setPlaceholderText("e.g. warm Rhodes chords, jazzy and mellow")
        self.whole = QRadioButton("whole song")
        self.only = QRadioButton()
        self.whole.setChecked(True)
        self.bars = QSpinBox()
        self.bars.setRange(2, 128)
        if selection:
            a, b = (round(s / bar_seconds) + 1 for s in selection)
            self.only.setText(f"only change bars {a}–{b - 1} (rest of the song stays)")
            self.bars.setValue(max(2, b - a))
        else:
            self.only.setText("only selected bars (drag on the ruler first)")
            self.only.setEnabled(False)
            self.bars.setValue(16)

        if empty:
            note = QLabel("Nothing is playing yet, so this part is made from scratch on its own track,\n"
                          "at the project tempo and key. Later layers will listen to it.")
        else:
            note = QLabel("The engine listens to every unmuted track and returns a new take of the\n"
                          "whole song with this part added. The tracks it heard get muted —\n"
                          "unless you keep only the new part (vocals, drums, bass).")
        note.setStyleSheet("color: #8a93a0;")
        self.form.addRow(note)
        self.form.addRow("Instrument", self.kind)
        self.form.addRow("Describe it", self.prompt)
        self.lyrics_label = QLabel("Lyrics")
        self.form.addRow(self.lyrics_label, self.lyrics)
        self.isolate = QCheckBox()
        self.isolate.setToolTip("The engine always returns the whole song with the part in it; this separates the "
                                "new part out, so it lands on its own track and your tracks stay as they are")
        if not empty:
            self.form.addRow("", self.isolate)
        if empty:
            self.form.addRow("Length (bars)", self.bars)
        else:
            range_box = QVBoxLayout()
            range_box.addWidget(self.whole)
            range_box.addWidget(self.only)
            self.form.addRow("Range", range_box)
        self._finish()
        self._restore()
        if kind is not None:
            self.kind.setCurrentText(kind)
        if prompt is not None:
            self.prompt.setText(prompt)
        self._kind_changed(self.kind.currentText())

    def _fields(self) -> dict:
        return {**super()._fields(), "kind": self.kind, "bars": self.bars}

    def _kind_changed(self, kind: str) -> None:
        sung = kind in VOCAL_TYPES
        self.lyrics.setVisible(sung)
        self.lyrics_label.setVisible(sung)
        stem = STEM_OF.get(kind)
        self.isolate.setVisible(stem is not None and not self.empty)
        self.isolate.setText(f"Keep only the new {kind.replace('_', ' ')} (your tracks stay unmuted)")
        # vocals: always wanted on their own. Drums/bass: the song's own drums/bass would come along too
        self.isolate.setChecked(sung)
        self.adjustSize()

    def values(self) -> dict:
        kind = self.kind.currentText()
        return {"kind": kind, "prompt": self.prompt.text().strip(),
                "lyrics": self._lyrics_text() if kind in VOCAL_TYPES else "[Instrumental]",
                "steps": self.steps.value(), "seed": self.seed.value(),
                "only_selection": self.only.isChecked(), "bars": self.bars.value(),
                "isolate": STEM_OF.get(kind) if self.isolate.isChecked() and not self.empty else None}


class IdeaDialog(_GenerateDialog):
    """Generate a starting groove/song from text, to layer on top of."""

    def __init__(self, bpm: int, key: str, parent=None):
        super().__init__("New idea from text", parent)
        self.prompt.setPlaceholderText("e.g. lofi hip hop groove, dusty drums and round bass")
        self.bars = QSpinBox()
        self.bars.setRange(4, 128)
        self.bars.setValue(16)
        self.instrumental = QCheckBox("instrumental (no vocals)")
        self.instrumental.setChecked(True)
        self.instrumental.toggled.connect(lambda on: self.lyrics.setEnabled(not on))
        self.lyrics.setEnabled(False)
        self.form.addRow(QLabel(f"Uses the project tempo/key: {bpm} BPM, {key}"))
        self.form.addRow("Describe it", self.prompt)
        self.form.addRow("Length (bars)", self.bars)
        self.form.addRow("", self.instrumental)
        self.form.addRow("Lyrics", self.lyrics)
        self._finish()
        self._restore()
        self.lyrics.setEnabled(not self.instrumental.isChecked())

    def _fields(self) -> dict:
        return {**super()._fields(), "bars": self.bars, "instrumental": self.instrumental}

    def values(self) -> dict:
        sung = not self.instrumental.isChecked()
        return {"prompt": self.prompt.text().strip(), "bars": self.bars.value(),
                "lyrics": self._lyrics_text() if sung else "[Instrumental]", "instrumental": not sung,
                "steps": self.steps.value(), "seed": self.seed.value()}

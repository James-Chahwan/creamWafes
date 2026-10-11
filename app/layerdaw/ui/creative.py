"""Writing tools around the pads: chord progressions, AI hooks, designed sounds, and making a song
from the selected pads."""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QButtonGroup, QCheckBox, QComboBox, QDialog, QDialogButtonBox, QFormLayout,
                               QGridLayout, QHBoxLayout, QLabel, QLineEdit, QPushButton, QRadioButton, QSlider,
                               QSpinBox, QVBoxLayout, QWidget)

from .. import chop, theory
from ..audio import Note, render_note
from ..chords import detect_pitch
from ..diag import log
from ..model import SR, Clip, Track, load_audio
from .pad_grid import family, pad_key

MOODS = ["dark", "sad", "epic", "trap", "chill", "jazzy", "uplifting", "pop"]


# --- picking the useful parts out of generated audio ------------------------------------------

def best_phrases(audio: np.ndarray, bar: float, bars: int, beat: float, count: int = 4) -> list[tuple[int, int]]:
    """Bar-aligned windows of ``bars`` bars that are loud and busy (a melody, not a pause)."""
    win = int(bars * bar * SR)
    if len(audio) < win:
        return [(0, len(audio))]
    starts = list(range(0, len(audio) - win + 1, int(bar * SR)))
    scored = []
    for s in starts:
        seg = audio[s:s + win]
        rms = float(np.sqrt(np.mean(seg ** 2)))
        notes = len(chop.onsets(seg.mean(axis=1), beat / 4 * 0.8))
        scored.append((rms * (0.4 + 0.6 * min(notes, 8) / 8), s))
    picks: list[int] = []
    for _score, s in sorted(scored, reverse=True):
        if all(abs(s - p) >= win for p in picks):
            picks.append(s)
        if len(picks) == count:
            break
    return [(s, s + win) for s in sorted(picks)]


def best_one_shots(audio: np.ndarray, bar: float, beat: float, count: int = 4) -> list[tuple[int, int]]:
    """The loudest clearly-different single sounds (each up to a bar long)."""
    hits = chop.onsets(audio.mean(axis=1), max(0.08, beat / 4))
    found = []
    for i, s in enumerate(hits):
        e = int(min(hits[i + 1] if i + 1 < len(hits) else len(audio), s + bar * SR))
        if e - s < int(0.05 * SR):
            continue
        f = chop.hit_features(audio[s:e])
        found.append((f["peak"], int(s), e, f["bright"]))
    picks: list[tuple[int, int, float]] = []
    for _peak, s, e, bright in sorted(found, reverse=True):
        if all(abs(bright - b) > 0.15 * bright for _, _, b in picks):
            picks.append((s, e, bright))
        if len(picks) == count:
            break
    return [(s, e) for s, e, _ in picks]


def steadiest_note(audio: np.ndarray, length: float = 1.5) -> tuple[float, float, float | None]:
    """(offset, length, pitch) of the most steady, clearly pitched part: a good chord sound."""
    win, hop = int(length * SR), int(0.25 * SR)
    best = (-1.0, 0, None)
    for s in range(int(0.3 * SR), max(int(0.3 * SR) + 1, len(audio) - win), hop):
        seg = audio[s:s + win]
        if len(seg) < win:
            break
        env = np.sqrt((seg[: win // 2400 * 2400].mean(axis=1).reshape(-1, 2400) ** 2).mean(axis=1))
        rms = float(env.mean())
        if rms < 1e-3:
            continue
        pitch = detect_pitch(seg)
        score = rms * (2.0 if pitch is not None else 1.0) / (1 + 3 * float(env.std() / (rms + 1e-9)))
        if score > best[0]:
            best = (score, s, pitch)
    return best[1] / SR, length, best[2]


# --- dialogs --------------------------------------------------------------------------------------

class ProgressionDialog(QDialog):
    def __init__(self, window):
        super().__init__(window)
        self.w = window
        self.setWindowTitle("Chord progression")
        self.setMinimumWidth(520)
        self.rng = random.Random()
        self.mood = QComboBox(editable=True)
        self.mood.addItems(MOODS)
        self.mood.setCurrentText(getattr(window, "_last_mood", "dark"))
        self.mood.lineEdit().setPlaceholderText("a mood: dark, lofi, cinematic, house…")
        self.result = QLabel()
        self.result.setStyleSheet("font-size: 20px; font-weight: 700; color: #c9b6e8; padding: 8px;")
        self.romans = QLabel()
        self.romans.setStyleSheet("color: #8a93a0;")
        self.bars = QSpinBox(minimum=1, maximum=4)
        self.bars.setValue(2 if window.project.bar_seconds < 2 else 1)
        self.repeats = QSpinBox(minimum=1, maximum=8, value=2)
        write = QPushButton("Write")
        write.clicked.connect(self.write)
        another = QPushButton("🎲 Another")
        another.clicked.connect(self.write)
        listen = QPushButton("▶ Audition")
        listen.clicked.connect(lambda: window.audition_progression(self.prog, self.bars.value()))
        to_pads = QPushButton("Put on the chord pads")
        to_pads.clicked.connect(lambda: (window.progression_to_pads(self.prog), self.accept()))
        to_timeline = QPushButton("Add to the timeline")
        to_timeline.clicked.connect(lambda: (window.progression_to_timeline(self.prog, self.bars.value(),
                                                                           self.repeats.value()), self.accept()))
        form = QFormLayout()
        row = QHBoxLayout()
        row.addWidget(self.mood, 1)
        row.addWidget(write)
        row.addWidget(another)
        form.addRow("Mood", row)
        form.addRow(self.result)
        form.addRow(self.romans)
        form.addRow("Bars per chord", self.bars)
        form.addRow("Repeat (timeline)", self.repeats)
        actions = QHBoxLayout()
        for b in (listen, to_pads, to_timeline):
            actions.addWidget(b)
        close = QDialogButtonBox(QDialogButtonBox.Close)
        close.rejected.connect(self.reject)
        layout = QVBoxLayout(self)
        layout.addLayout(form)
        layout.addLayout(actions)
        layout.addWidget(close)
        self.write()

    def write(self) -> None:
        self.w._last_mood = self.mood.currentText()
        self.prog = theory.write_progression(self.w.project.key, self.mood.currentText(), self.rng)
        self.result.setText("  →  ".join(self.prog["names"]))
        self.romans.setText(f"{self.prog['mood']} · {' – '.join(self.prog['romans'])} in {self.w.project.key}")


class HookDialog(QDialog):
    def __init__(self, parent, default: str):
        super().__init__(parent)
        self.setWindowTitle("Generate hooks")
        self.setMinimumWidth(460)
        self.prompt = QLineEdit(default)
        self.prompt.setPlaceholderText("e.g. pitched vocal chop, plucky synth lead, distorted guitar riff")
        self.phrases = QRadioButton("Phrases — short melodic hooks (1–2 bars) in your key")
        self.candy = QRadioButton("Ear candy — one-shots: shouts, risers, stabs, FX")
        self.phrases.setChecked(True)
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.button(QDialogButtonBox.Ok).setText("Generate")
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        form = QFormLayout(self)
        form.addRow("Describe it", self.prompt)
        form.addRow(self.phrases)
        form.addRow(self.candy)
        form.addRow(buttons)

    def values(self) -> tuple[str, str]:
        return self.prompt.text().strip(), "phrases" if self.phrases.isChecked() else "candy"


# --- the mixin ------------------------------------------------------------------------------------

class CreativeMixin:
    """Mixed into MainWindow (after PadsMixin/GenerateMixin)."""

    # progressions ----------------------------------------------------------------------------
    def open_progression_writer(self) -> None:
        dlg = ProgressionDialog(self)
        dlg.exec()
        dlg.deleteLater()

    def _progression_pad(self):
        """The sound a progression is heard with: the selected chord pad's, else the built-in synth."""
        from ..model import Pad
        for i in self.selected_pads:
            if self.project.pads[i].play == "chord":
                return self.project.pads[i]
        return Pad(play="chord", mode="hold")

    def render_progression(self, prog: dict, bars_per_chord: int, repeats: int = 1) -> np.ndarray:
        key = self.project.key
        pad = self._progression_pad()
        bar = int(self.project.bar_seconds * SR)
        span = bars_per_chord * bar
        chords = prog["degrees"] * repeats
        out = np.zeros(((len(chords) * bars_per_chord + 1) * bar, 2), np.float32)
        strum = 0.035 if pad.strum else 0.0
        for k, degree in enumerate(chords):
            notes = theory.chord_notes(key, degree, prog["flavour"])
            gain = 0.8 * pad.gain / np.sqrt(len(notes))
            for j, midi in enumerate(notes):
                buf, loop = self.instrument.note(pad, midi, self.key_root(), sustain=True)
                start = k * span + int(j * strum * SR)
                audio = render_note(Note(buf, gain, loop), span - int(j * strum * SR) - int(0.12 * SR))
                n = min(len(audio), len(out) - start)
                out[start:start + n] += audio[:n]
        nz = np.flatnonzero(np.abs(out).max(axis=1) > 1e-4)
        return out[: nz[-1] + 1] if len(nz) else out

    def audition_progression(self, prog: dict, bars_per_chord: int) -> None:
        self.player.play_note(Note(self.render_progression(prog, bars_per_chord)), ("progression",), choke=True)

    def progression_to_pads(self, prog: dict) -> None:
        """Lay the progression's chords across the selected pads (or 4 pads from the selected one)."""
        sel = sorted(self.selected_pads)
        targets = sel if len(sel) > 1 else [sel[0] + k for k in range(4)]
        self._ensure_pads(max(targets) + 1)
        slots = theory.pad_slots(prog["degrees"])
        self._checkpoint("Progression onto pads")
        for n, i in enumerate(targets):
            pad = self.project.pads[i]
            pad.play, pad.degree, pad.flavour = "chord", slots[n % len(slots)], prog["flavour"]
            if pad.mode in ("loop", "playon", "oneshot") and pad.clip is None:
                pad.mode = "hold"
            if pad.clip is None:
                pad.name, pad.kind = "synth", "keys"
        self.pads_dock.show()
        self.select_pads(targets)
        self.refresh_pads()
        self._changed()
        names = [self.pad_label(self.project.pads[i])[0] for i in targets]
        self.engine_label.setText(f"Chord pads {' '.join(pad_key(i) for i in targets)}: {' · '.join(names)}")
        log.info("progression → pads %s %s", targets, names)

    def progression_to_timeline(self, prog: dict, bars_per_chord: int, repeats: int) -> None:
        audio = self.render_progression(prog, bars_per_chord, repeats)
        path = self.project.write_wav(audio, "progression")
        bar = self.project.bar_seconds
        start = np.floor(self.player.position / bar) * bar
        self._checkpoint("Add progression")
        track = Track(name=self._unique_name("chords · " + " ".join(prog["names"])), kind="keyboard",
                      clips=[Clip(path, start)], prompt=f"{prog['mood']} progression")
        self.project.tracks.append(track)
        self.refresh()
        self._changed()
        self.engine_label.setText(f"Added '{track.name}' at bar {int(start / bar) + 1}")
        log.info("progression → timeline %s ×%d", prog["names"], repeats)

    def generate_chord_sound(self) -> None:
        """A chord pad's sound comes from ✦ Generate on that pad (keys / pad / lead are good kinds)."""
        i = self.primary_pad
        if self.project.pads[i].play != "chord":
            self.engine_label.setText("Select a chord pad first (or set a pad to Play as: Chord)")
            return
        self.design_pad_sound(i)

    # hooks onto pads ----------------------------------------------------------------------------
    def generate_hooks(self, pad: int | None = None) -> None:
        dlg = HookDialog(self, getattr(self, "_last_hook_prompt", "pitched vocal chop"))
        accepted = dlg.exec()
        prompt, kind = dlg.values()
        dlg.deleteLater()
        if not accepted:
            return
        if not prompt:
            return
        self._last_hook_prompt = prompt
        if kind == "phrases":
            caption = f"{prompt}, catchy melodic hook, memorable riff, solo lead instrument"
            seconds = max(10.5, 8 * self.project.bar_seconds)
        else:
            caption = f"{prompt}, one-shot sound effects, vocal shouts, risers, impacts, stabs, sparse"
            seconds = 12.0
        params = self._base_params(caption, "[Instrumental]", 32, -1)
        params.update(duration=seconds, instrumental=True)
        sel = sorted(self.selected_pads)
        pads = [pad] if pad is not None else (sel if len(sel) > 1 else [sel[0]] if sel else [12])
        gen = {"mode": "hooks", "kind": "hook", "prompt": prompt, "hook_type": kind, "hook_pads": pads}
        label = f"hook → pad {pad_key(pad)}" if pad is not None else "hooks"
        self._submit("text2music", params, gen, name=label, mute=[], pads=pads)

    def _hooks_ready(self, path: Path, prompt: str, kind: str, pads: list[int]) -> None:
        audio = load_audio(path)
        bar, beat = self.project.bar_seconds, self.project.beat_seconds
        if kind == "phrases":
            ranges = best_phrases(audio, bar, 2 if bar < 1.6 else 1, beat, len(pads))
        else:
            ranges = best_one_shots(audio, bar, beat, len(pads))
        self._checkpoint("Generate hooks")
        for n, (index, (s, e)) in enumerate(zip(pads, ranges), 1):
            p = self.project.pads[index]
            p.clear()
            p.clip = Clip(path, 0.0, s / SR, (e - s) / SR)
            p.name = f"{prompt[:12]} {n}" if len(pads) > 1 else prompt[:16]
            p.kind, p.mode = "hook", "oneshot"
        self.refresh_pads()
        self._changed()
        self.engine_label.setText(f"{len(ranges)} hook{'s' if len(ranges) != 1 else ''} on the top row — "
                                  f"keys {' '.join(pad_key(i) for i in pads[:len(ranges)])}")
        log.info("hooks (%s) → pads %s from %s", kind, pads[:len(ranges)], path.name)


# --- single-sound design: describe → generate → pick from candidates --------------------------------

class _Chips(QWidget):
    """Toggle buttons in rows. ``exclusive``: at most one on (click it again to clear)."""

    def __init__(self, items: list[str], selected=(), exclusive: bool = False, columns: int = 6, parent=None):
        super().__init__(parent)
        grid = QGridLayout(self)
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setSpacing(3)
        self.exclusive = exclusive
        self.buttons: dict[str, QPushButton] = {}
        for n, item in enumerate(items):
            b = QPushButton(item.replace("&", "&&"))  # "R&B", not a keyboard shortcut on B
            b.setCheckable(True)
            b.setChecked(item in selected)
            b.setStyleSheet("QPushButton { padding: 2px 6px; } QPushButton:checked { background: #4f7fd9; color: white; }")
            b.toggled.connect(lambda on, it=item: self._toggled(it, on))
            grid.addWidget(b, n // columns, n % columns)
            self.buttons[item] = b

    def _toggled(self, item: str, on: bool) -> None:
        if self.exclusive and on:
            for other, b in self.buttons.items():
                if other != item and b.isChecked():
                    b.blockSignals(True)
                    b.setChecked(False)
                    b.blockSignals(False)

    def selected(self) -> list[str]:
        return [item for item, b in self.buttons.items() if b.isChecked()]


def _segmented(options: list[tuple[str, str]], current: str) -> tuple[QWidget, dict[str, QPushButton]]:
    box = QWidget()
    row = QHBoxLayout(box)
    row.setContentsMargins(0, 0, 0, 0)
    row.setSpacing(3)
    group = QButtonGroup(box)
    buttons = {}
    for value, text in options:
        b = QPushButton(text)
        b.setCheckable(True)
        b.setChecked(value == current)
        b.setStyleSheet("QPushButton:checked { background: #4f7fd9; color: white; }")
        group.addButton(b)
        row.addWidget(b)
        buttons[value] = b
    row.addStretch(1)
    return box, buttons


class SoundDialog(QDialog):
    """What sound(s) to make: a kind (kick, 808, chord loop…), your words, genre and mood, how long,
    how many takes — or a variation of the pad's own sound."""

    def __init__(self, parent, pad_keys, default_text: str, pad=None, prefs: dict | None = None,
                 vary: bool = False):
        from ..sounds import GENRES, GROUPS, MOODS, guess_type
        super().__init__(parent)
        keys = [pad_keys] if isinstance(pad_keys, str) else list(pad_keys)
        prefs = prefs or {}
        self.many = len(keys) > 1
        self.setWindowTitle(f"New sounds for {len(keys)} pads" if self.many else f"New sound for pad {keys[0]}")
        self.setMinimumWidth(560)
        self.text = QLineEdit(default_text)
        self.text.setPlaceholderText("e.g. distorted cyberpunk 808, dusty vinyl snare, glassy Rhodes")
        self.kind = QComboBox()
        self.kind.setMaxVisibleItems(40)
        for group, kinds in GROUPS:
            self.kind.addItem(f"— {group} —", None)
            self.kind.model().item(self.kind.count() - 1).setEnabled(False)
            for k in kinds:
                self.kind.addItem(f"   {k}", k)
        self.set_kind(guess_type(default_text))
        self.text.textEdited.connect(lambda t: self.set_kind(guess_type(t)))
        self._kind_touched = False
        self.kind.activated.connect(lambda _i: setattr(self, "_kind_touched", True))
        self.genre = _Chips(GENRES, [prefs.get("genre", "")], exclusive=True)
        self.moods = _Chips(MOODS, prefs.get("moods", []))
        length_box, self.length_buttons = _segmented([("short", "Short"), ("medium", "Medium"), ("long", "Long")],
                                                     prefs.get("length", "medium"))
        takes_box, self.takes_buttons = _segmented([("4", "4 takes"), ("8", "8 takes")], str(prefs.get("takes", 4)))
        self._takes_box = takes_box  # owned here: with several pads it isn't shown (one take per pad)
        self.vary = QCheckBox()
        self.how_different = QSlider(Qt.Horizontal)
        self.how_different.setRange(0, 100)
        self.how_different.setValue(50)
        self.how_different.setToolTip("A little: close to the pad's sound · A lot: only loosely like it")
        has_sound = pad is not None and pad.clip is not None and not self.many
        if has_sound:
            self.vary.setText(f"Start from this pad's sound ({pad.name or 'sound'})")
            self.vary.setChecked(vary)
            if vary:
                self.set_kind(guess_type(f"{pad.name} {pad.kind}"))
        self.vary.toggled.connect(lambda on: self.how_different.setEnabled(on))
        self.how_different.setEnabled(self.vary.isChecked())
        self.spread = QCheckBox("Put every take on its own pad (in the tray)")
        self.spread.setToolTip("Keep all of them to play with: they go onto empty tray pads below the 4×4"
                               + (" — this pad stays as it is" if has_sound else ""))
        self.spread.setChecked(vary)  # variations: keep them all next to the original by default
        self.vary.toggled.connect(self.spread.setChecked)
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.button(QDialogButtonBox.Ok).setText(f"✦ Generate {len(keys)} sounds" if self.many else "✦ Generate")
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        form = QFormLayout(self)
        form.addRow("Describe it", self.text)
        form.addRow("Kind of sound", self.kind)
        form.addRow("Genre", self.genre)
        form.addRow("Mood", self.moods)
        form.addRow("Length", length_box)
        if self.many:
            form.addRow("Takes", QLabel(f"a different one on each pad ({' '.join(keys)})"))
        else:
            takes_row = QHBoxLayout()
            takes_row.addWidget(takes_box)
            takes_row.addWidget(self.spread)
            form.addRow("Takes", takes_row)
        if has_sound:
            slider_row = QHBoxLayout()
            for w in (QLabel("a little"), self.how_different, QLabel("a lot")):
                slider_row.addWidget(w)
            form.addRow(self.vary)
            form.addRow("How different", slider_row)
        self.note = QLabel()
        self.note.setStyleSheet("color: #8a93a0;")
        self.note.setWordWrap(True)
        form.addRow(self.note)
        form.addRow(buttons)
        self.kind.currentIndexChanged.connect(lambda _i: self._kind_changed())  # (now that the note exists)
        self._kind_changed()

    def set_kind(self, kind: str) -> None:
        i = self.kind.findData(kind)
        if i >= 0:
            self.kind.setCurrentIndex(i)

    def sound_type(self) -> str:
        return self.kind.currentData() or "lead"

    def _kind_changed(self) -> None:
        from ..sounds import LENGTHS, is_loop
        loop = is_loop(self.sound_type())
        for value, b in self.length_buttons.items():
            b.setText(LENGTHS[value][1 if loop else 0])
        self.note.setText("Loops are whole bars at the song's tempo and play as loop pads." if loop else
                          "One sound at a time: the AI makes a take of just this sound and you pick the best.")

    @staticmethod
    def _picked(buttons: dict) -> str:
        return next((v for v, b in buttons.items() if b.isChecked()), next(iter(buttons)))

    def spec(self) -> dict:
        genre = self.genre.selected()
        return {"text": self.text.text().strip(), "type": self.sound_type(), "genre": genre[0] if genre else "",
                "moods": self.moods.selected(), "length": self._picked(self.length_buttons),
                "takes": int(self._picked(self.takes_buttons)), "vary": self.vary.isChecked(),
                "spread": self.spread.isChecked() and not self.many,
                "strength": round(1.0 - 0.7 * self.how_different.value() / 100, 2)}

    def accept(self) -> None:
        from ..sounds import guess_type
        if not self._kind_touched and not self.vary.isChecked():  # however the text got there, read it again
            self.set_kind(guess_type(self.text.text()))
        super().accept()


class SoundPicker(QDialog):
    """Audition the best candidates from one generation and pick one (or ask for more, or keep them all)."""

    MORE = 2
    ALL = 3

    def __init__(self, window, audio: np.ndarray, picks: list[tuple[int, int, str]], pad_key: str):
        super().__init__(window)
        from ..model import declick
        self.w = window
        self.setWindowTitle(f"Pick a sound for pad {pad_key}")
        self.choice: int | None = None
        self.slices = [declick(audio[s:e]) for s, e, _ in picks]
        grid = QGridLayout()
        self.buttons = []
        for i, (s, e, label) in enumerate(picks):
            b = QPushButton(f"▶  {label}\n{(e - s) / SR:.2f} s")
            b.setCheckable(True)
            b.setMinimumSize(120, 64)
            b.clicked.connect(lambda _=False, i=i: self._audition(i))
            self.buttons.append(b)
            grid.addWidget(b, i // 4, i % 4)
        use = QPushButton("Put on pad")
        use.setDefault(True)
        use.clicked.connect(self._use)
        more = QPushButton("✦ More like this")
        more.clicked.connect(lambda: self.done(self.MORE))
        keep_all = QPushButton("All into the tray")
        keep_all.setToolTip("Keep every take: each goes on an empty pad below the 4×4")
        keep_all.clicked.connect(lambda: self.done(self.ALL))
        keep_all.setEnabled(bool(picks))
        cancel = QPushButton("Cancel")
        cancel.clicked.connect(self.reject)
        actions = QHBoxLayout()
        for b in (more, keep_all, cancel, use):
            actions.addWidget(b)
        layout = QVBoxLayout(self)
        hint = QLabel("Click a candidate to hear it, then Put on pad." if picks else
                      "Nothing clean enough in that take — try ✦ More like this, or describe it differently.")
        layout.addWidget(hint)
        layout.addLayout(grid)
        layout.addLayout(actions)
        if picks:
            self._audition(0)

    def _audition(self, i: int) -> None:
        self.choice = i
        for j, b in enumerate(self.buttons):
            b.setChecked(j == i)
        self.w.player.play_note(Note(self.slices[i]), ("audition",), choke=True)

    def _use(self) -> None:
        if self.choice is not None:
            self.accept()


BANK_PARTS = [("kit", "Drum kit (kick, snare, hats, clap…)"), ("808", "808"), ("chords", "Chords (chord pads)"),
              ("melody", "Melody (lead)"), ("hook", "Hook (vocal chops)"), ("fx", "FX (riser)"),
              ("loop", "Drum loop"), ("perc", "Percussion")]
BANK_TYPES = {"808": "808", "perc": "perc", "loop": "drum loop", "melody": "lead", "hook": "vocal chop",
              "fx": "riser", "chords": "keys"}


class BankDialog(QDialog):
    """Fill several pads in one go from one vibe: kit, 808, chords, melody, hook…"""

    def __init__(self, window, targets: list[int], prefs: dict | None = None):
        from ..sounds import GENRES, MOODS
        super().__init__(window)
        prefs = prefs or {}
        self.targets = sorted(targets)
        self.setWindowTitle(f"Fill {len(targets)} pads")
        self.setMinimumWidth(560)
        self.text = QLineEdit(prefs.get("bank_text", ""))
        self.text.setPlaceholderText("the vibe, e.g. cyberpunk night drive, dusty soul, hard UK drill")
        self.genre = _Chips(GENRES, [prefs.get("genre", "")], exclusive=True)
        self.moods = _Chips(MOODS, prefs.get("moods", []))
        self.parts: dict[str, QCheckBox] = {}
        parts_box = QGridLayout()
        for n, (part, text) in enumerate(BANK_PARTS):
            box = QCheckBox(text)
            box.setChecked(True)
            box.toggled.connect(lambda _on: self._preview())
            parts_box.addWidget(box, n // 2, n % 2)
            self.parts[part] = box
        self.preview = QLabel()
        self.preview.setWordWrap(True)
        self.preview.setStyleSheet("color: #c8ccd4;")
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.button(QDialogButtonBox.Ok).setText(f"✦ Fill {len(targets)} pads")
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        form = QFormLayout(self)
        form.addRow("Vibe", self.text)
        form.addRow("Genre", self.genre)
        form.addRow("Mood", self.moods)
        form.addRow("Make", parts_box)
        form.addRow(self.preview)
        note = QLabel("Each part is its own generation (one after another, ~10–20 s each). The best take goes on "
                      "each pad; use ✦ Generate or Replace on a pad afterwards to change just that one.")
        note.setWordWrap(True)
        note.setStyleSheet("color: #8a93a0;")
        form.addRow(note)
        form.addRow(buttons)
        self._preview()

    def include(self) -> list[str]:
        return [p for p, box in self.parts.items() if box.isChecked()]

    def _preview(self) -> None:
        from ..sounds import plan_bank
        plan = plan_bank(self.targets, self.include())
        names = dict(BANK_PARTS)
        self.preview.setText("  ·  ".join(f"<b>{' '.join(pad_key(i) for i in pads)}</b> {names[p].split(' (')[0]}"
                                          for p, pads in plan.items()) or "<i>pick something to make</i>")

    def spec(self) -> dict:
        genre = self.genre.selected()
        return {"text": self.text.text().strip(), "genre": genre[0] if genre else "", "moods": self.moods.selected(),
                "include": self.include()}


class SoundMixin:
    """Mixed into MainWindow: generating sounds for pads (one, several, variations, a whole bank)."""

    def _sound_prefs(self) -> dict:
        return getattr(self, "_prefs", {"genre": "", "moods": [], "length": "medium", "takes": 4})

    def design_pad_sound(self, index: int, text: str | None = None, sound_type: str | None = None,
                         targets: list[int] | None = None, vary: bool = False) -> None:
        """✦ Generate for a pad (or ``targets``). With ``text`` given, no dialog (✦ More like this)."""
        targets = targets or [index]
        if text is None:
            pad = self.project.pads[targets[0]]
            dlg = SoundDialog(self, [pad_key(i) for i in targets], getattr(self, "_last_sound_text", ""),
                              pad if len(targets) == 1 else None, self._sound_prefs(), vary=vary)
            accepted = dlg.exec()
            spec = dlg.spec()
            dlg.deleteLater()
            if not accepted:
                return
            self._last_sound_text = spec["text"]
            self._prefs = {**self._sound_prefs(), **{k: spec[k] for k in ("genre", "moods", "length", "takes")}}
        else:
            spec = {**self._sound_prefs(), "text": text, "type": sound_type, "vary": False, "strength": 0.6}
        self._submit_sound(spec, targets)

    def generate_sounds(self, targets: list[int] | None = None) -> None:
        """Multi-pad ✦ Sounds onto these: one dialog, a different take on each pad."""
        targets = sorted(targets or self.selected_pads)
        self.design_pad_sound(targets[0], targets=targets)

    def _submit_sound(self, spec: dict, targets: list[int], auto: bool = False, extra: dict | None = None) -> None:
        from ..sounds import caption, take_seconds, variation_source
        st = spec["type"]
        source = targets[0]  # the pad the dialog was opened on (a variation starts from its sound)
        if spec.get("spread") and len(targets) == 1:  # every take on its own pad
            n = int(spec.get("takes", 4))
            keep = [] if spec.get("vary") or self._has_sound(self.project.pads[source]) else [source]
            targets = keep + self.tray_slots(n - len(keep))
            auto = True
            self.pad_panel.grid.show_pad(targets[-1])
        takes = max(int(spec.get("takes", 4)), len(targets))
        cap = caption(st, spec.get("text", ""), spec.get("genre", ""), spec.get("moods", ()), spec.get("length", "medium"))
        params = self._base_params(cap, "[Instrumental]", 8, -1)
        params["instrumental"] = True
        gen = {"mode": "sound", "kind": st, "prompt": spec.get("text", ""), "sound_type": st,
               "sound_pad": targets[0], "targets": list(targets), "spec": spec,
               "auto": auto or len(targets) > 1, **(extra or {})}
        pad = self.project.pads[source]
        keys = " ".join(pad_key(i) for i in targets)
        if spec.get("vary") and pad.clip is not None:
            src = variation_source(pad.buffer(), st, self.project.beat_seconds)
            wav = self.project.write_wav(src, "variation-src")
            params.update(src_audio=str(wav), audio_cover_strength=float(spec.get("strength", 0.6)))
            self._submit("cover", params, gen, name=f"variation of {pad.name or st} → pad {keys}", mute=[],
                         pads=targets)
            return
        params["duration"] = take_seconds(st, takes, spec.get("length", "medium"), self.project.bar_seconds)
        shown = gen.get("chords") or targets  # bank chords: one take, shown coming on every chord pad
        self._submit("text2music", params, gen, name=f"{st} → pad {keys}", mute=[], pads=shown)

    def _sound_ready(self, path: Path, gen: dict) -> None:
        from ..sounds import SOUND_TYPES, candidates
        audio = load_audio(path)
        spec = gen.get("spec") or {"length": "medium", "takes": 4}
        targets = gen.get("targets") or [gen["sound_pad"]]
        sound_type = gen["sound_type"]
        count = max(int(spec.get("takes", 4)), len(targets))
        picks = candidates(audio, sound_type, self.project.beat_seconds, self.project.bar_seconds, count,
                           spec.get("length", "medium"))
        log.info("sound %s → %d candidates %s", sound_type, len(picks), [p[2] for p in picks])
        if gen.get("chords"):
            self._chords_from_take(path, picks, gen["chords"], spec)
            return
        if gen.get("auto"):
            self._picks_onto_pads(path, picks, targets, sound_type)
            return
        index = targets[0]
        picker = SoundPicker(self, audio, picks, pad_key(index))
        result = picker.exec()
        picker.deleteLater()
        self.player._cut.append(("audition",))
        if result == SoundPicker.ALL:
            self._picks_onto_pads(path, picks, self.tray_slots(len(picks)), sound_type)
            return
        if result == SoundPicker.MORE:
            self._submit_sound(spec if gen.get("spec") else {**self._sound_prefs(), "text": gen["prompt"],
                                                              "type": sound_type}, targets)
            return
        if result != QDialog.Accepted or picker.choice is None:
            return
        s, e, label = picks[picker.choice]
        _caps, _how, kind, mode = SOUND_TYPES[sound_type]
        self._put_on_pad(index, path, s / SR, (e - s) / SR, label, kind)
        pad = self.project.pads[index]
        if pad.mode != mode:
            pad.mode = mode
            self.refresh_pads()

    def _picks_onto_pads(self, path: Path, picks: list, targets: list[int], sound_type: str) -> None:
        """A different take on each pad, as one undo step."""
        from ..sounds import SOUND_TYPES
        if not picks:
            self.engine_label.setText(f"Nothing clean enough came out for {sound_type} — try again or describe it "
                                      "differently")
            return
        _caps, _how, kind, mode = SOUND_TYPES[sound_type]
        self._checkpoint(f"{sound_type} onto pads")
        filled = []
        for i, (s, e, label) in zip(targets, picks):
            pad = self.project.pads[i]
            pad.clear()
            pad.clip = Clip(path, 0.0, s / SR, (e - s) / SR)
            pad.name, pad.kind, pad.mode = label, kind, mode
            filled.append(i)
        self.pads_dock.show()
        self.refresh_pads()
        self._changed()
        self.pad_panel.grid.show_pad(filled[-1])
        missing = [pad_key(i) for i in targets[len(filled):]]
        self.engine_label.setText(f"{sound_type} on pads {' '.join(pad_key(i) for i in filled)}"
                                  + (f" — only {len(filled)} good takes, {' '.join(missing)} left as they were"
                                     if missing else ""))
        log.info("%s → pads %s", sound_type, filled)

    def _chords_from_take(self, path: Path, picks: list, chord_pads: list[int], spec: dict) -> None:
        """Bank chords: the best keys take becomes the sound of every chord pad, in a progression."""
        if not picks:
            self.engine_label.setText("No clean keys sound came out for the chords — try ✦ Generate on those pads")
            return
        s, e, label = picks[0]
        prog = theory.write_progression(self.project.key, " ".join([*spec.get("moods", []), spec.get("genre", "")]))
        slots = theory.pad_slots(prog["degrees"])
        self._checkpoint("Chords onto pads")
        for n, i in enumerate(chord_pads):
            pad = self.project.pads[i]
            pad.clear()
            pad.clip = Clip(path, 0.0, s / SR, (e - s) / SR)
            pad.name, pad.kind, pad.mode = label, "keys", "hold"
            pad.play, pad.degree, pad.flavour = "chord", slots[n % len(slots)], prog["flavour"]
        self.pads_dock.show()
        self.refresh_pads()
        self._changed()
        names = [self.pad_label(self.project.pads[i])[0] for i in chord_pads]
        self.engine_label.setText(f"Chord pads {' '.join(pad_key(i) for i in chord_pads)}: {' · '.join(names)}")

    # --- a whole bank ---------------------------------------------------------------------------
    def fill_bank(self, targets: list[int] | None = None, spec: dict | None = None) -> dict:
        """One vibe → kit, 808, chords, melody, hook, fx… across the selected pads. Returns the plan."""
        from ..sounds import plan_bank
        targets = sorted(targets or self.selected_pads)
        if spec is None:
            dlg = BankDialog(self, targets, {**self._sound_prefs(), "bank_text": getattr(self, "_last_bank", "")})
            accepted = dlg.exec()
            spec = dlg.spec()
            dlg.deleteLater()
            if not accepted:
                return {}
            self._last_bank = spec["text"]
            self._prefs = {**self._sound_prefs(), "genre": spec["genre"], "moods": spec["moods"]}
        plan = plan_bank(targets, spec["include"])
        style = {"text": spec["text"], "genre": spec["genre"], "moods": spec["moods"], "length": "medium", "takes": 4}
        for part, pads in plan.items():
            if part == "kit":
                words = ", ".join(x for x in (spec["text"], " ".join([*spec["moods"], spec["genre"]])) if x)
                self._submit_kit(words or "punchy drum kit", pads)
            elif part == "chords":
                self._submit_sound({**style, "type": BANK_TYPES[part]}, pads[:1], auto=True, extra={"chords": pads})
            else:
                self._submit_sound({**style, "type": BANK_TYPES[part]}, pads, auto=True)
        log.info("bank %s → %s", spec, plan)
        self.engine_label.setText(f"Filling {len(targets)} pads: {', '.join(plan)} (one after another)")
        return plan
# --- a song from the selected pads (arranged from exactly those sounds) ---------------------------

ROLE_TEXT = {"kick": "kick pattern", "snare": "snare pattern", "clap": "claps", "hat": "hi-hat pattern",
             "openhat": "open hats", "perc": "percussion", "bass": "bass line on the chord roots",
             "lead": "arpeggios (chorus)", "chords": "chord progression", "stab": "stabs on the beat",
             "texture": "held texture", "hook": "hooks in the choruses", "fx": "risers into choruses"}
# how each hit of a pad's part sounds
NOTES = [("auto", "As the pad is set"), ("short", "Short notes"), ("full", "Full sample, every hit"),
         ("hold", "Hold to the next hit")]
NOTE_TIPS = {"auto": "Whatever the pad's own mode does (One-shot, Hold, Loop, Play-on)",
             "short": "Quick, clipped hits (an 8th note at most)",
             "full": "Each hit plays the whole sound; the next hit cuts it off",
             "hold": "Each hit sustains until the next one: no gaps"}
MODE_TEXT = {"oneshot": "one-shot", "hold": "hold", "loop": "loop", "playon": "play-on"}


def _cut(audio: np.ndarray, frames: int) -> np.ndarray:
    """Stop ``audio`` after ``frames`` with a short fade (no click)."""
    if len(audio) <= frames:
        return audio
    out = audio[:max(1, frames)].copy()
    f = min(len(out), int(0.012 * SR))
    out[-f:] *= np.linspace(1.0, 0.0, f, dtype=np.float32)[:, None]
    return out


class SongDialog(QDialog):
    def __init__(self, window, indices: list[int]):
        from .. import arrange
        super().__init__(window)
        self.w, self.indices = window, sorted(indices)
        self.setWindowTitle("Make a song from these pads")
        self.setMinimumWidth(640)
        self.style = QComboBox()
        auto = arrange.auto_style(window.project.bpm)
        self.style.addItem(f"Auto ({auto}, from {window.project.bpm} BPM)", "auto")
        for st in arrange.STYLES:
            self.style.addItem(st, st)
        self.short = QRadioButton("Short song — about 32 bars (intro, verse, chorus, break, chorus)")
        self.full = QRadioButton("Full song — about 72 bars (… plus an outro, longer sections)")
        self.full.setChecked(True)
        self.part = QRadioButton("Just a part:")
        self.part_name = QComboBox()
        self.part_name.addItems(arrange.PART_NAMES)
        self.part_name.setCurrentText("verse")
        self.part_bars = QSpinBox()
        self.part_bars.setRange(1, 64)
        self.part_bars.setValue(8)
        self.part_bars.setSuffix(" bars")
        for w in (self.part_name, self.part_bars):
            w.setEnabled(False)
        self.part.toggled.connect(lambda on: (self.part_name.setEnabled(on), self.part_bars.setEnabled(on)))
        part_row = QHBoxLayout()
        for w in (self.part, self.part_name, self.part_bars):
            part_row.addWidget(w)
        part_row.addStretch(1)
        self.auto_roles = {i: arrange.role_of(window.project.pads[i], window._classify_pad) for i in self.indices}
        grid = QGridLayout()
        grid.setHorizontalSpacing(8)
        for col, text in enumerate(("Pad", "", "Plays", "Notes")):
            head = QLabel(text)
            head.setStyleSheet("color: #8a93a0; font-size: 11px;")
            grid.addWidget(head, 0, col)
        self.part_boxes: dict[int, QComboBox] = {}
        self.note_boxes: dict[int, QComboBox] = {}
        for row, i in enumerate(self.indices, start=1):
            pad = window.project.pads[i]
            label = window.pad_label(pad)
            what = QLabel(f"<b>{pad_key(i)}</b>")
            name = QLabel(f"{pad.name or 'synth'}" + (f" ({label[0]})" if label else ""))
            part = QComboBox()
            for role in arrange.ROLES:
                if role == "chords" and pad.play != "chord":
                    continue  # chords come from chord pads (they know which chord)
                part.addItem(ROLE_TEXT[role], role)
            part.setCurrentIndex(part.findData(self.auto_roles[i]))
            notes = QComboBox()
            for value, text in NOTES:
                notes.addItem(f"{text} ({MODE_TEXT.get(pad.mode, pad.mode)})" if value == "auto" else text, value)
                notes.setItemData(notes.count() - 1, NOTE_TIPS[value], Qt.ToolTipRole)
            self.part_boxes[i], self.note_boxes[i] = part, notes
            for col, widget in enumerate((what, name, part, notes)):
                grid.addWidget(widget, row, col)
        grid.setColumnStretch(1, 1)
        self.no_chords = QLabel("<i>no chord pads selected: bass and leads follow i – VI – iv – v</i>")
        self.no_chords.setVisible(not any(r == "chords" for r in self.auto_roles.values()))
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.button(QDialogButtonBox.Ok).setText("Make the song")
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        form = QFormLayout(self)
        form.addRow("Style", self.style)
        form.addRow(self.short)
        form.addRow(self.full)
        form.addRow(part_row)
        form.addRow(QLabel("What each pad plays, and how its notes sound:"))
        form.addRow(grid)
        form.addRow(self.no_chords)
        note = QLabel("Every pad lands on its own track (at the playhead's bar). A part goes in alongside what's "
                      "there; a song mutes what it overlaps. Right-click a track later → Song part to change what "
                      "it plays or how its notes sound, or ⟳ for a new variation.")
        note.setStyleSheet("color: #8a93a0;")
        note.setWordWrap(True)
        form.addRow(note)
        form.addRow(buttons)

    def length(self) -> str:
        if self.part.isChecked():
            return f"part:{self.part_name.currentText()}:{self.part_bars.value()}"
        return "short" if self.short.isChecked() else "full"

    def choices(self) -> tuple[dict[int, str], dict[int, str]]:
        """(parts changed from what the pad would play anyway, notes other than 'as the pad is set')."""
        roles = {i: b.currentData() for i, b in self.part_boxes.items() if b.currentData() != self.auto_roles[i]}
        notes = {i: b.currentData() for i, b in self.note_boxes.items() if b.currentData() != "auto"}
        return roles, notes


class SongMixin:
    """Mixed into MainWindow: arrange the selected pads into a song on the timeline."""

    def _classify_pad(self, pad) -> str | None:
        """For a generic drum pad, tell kick/snare/hat… from the sound itself."""
        buf = pad.buffer()
        if buf is None:
            return None
        role = chop.drum_role(chop.hit_features(buf))
        return {"hat": "hat", "open hat": "openhat", "tom": "perc"}.get(role, role)

    def open_song_maker(self) -> None:
        indices = [i for i in self.selected_pads
                   if self.project.pads[i].clip is not None or self.project.pads[i].play != "sample"]
        if len(indices) < 2:
            self.engine_label.setText("Select a few pads with sounds first (Ctrl+click), then make a song")
            return
        dlg = SongDialog(self, indices)
        accepted = dlg.exec()
        style, length = dlg.style.currentData(), dlg.length()
        roles, notes = dlg.choices()
        dlg.deleteLater()
        if accepted:
            self.make_song(indices, style, length, roles=roles, notes=notes)

    def _song_pads(self, indices) -> dict:
        return {i: self.project.pads[i] for i in indices          # a sample pad with no sound has nothing to play
                if self.project.pads[i].clip is not None or self.project.pads[i].play != "sample"}

    def _song_plan(self, pads: dict, style: str, length: str, roles=None, variations=None):
        from .. import arrange
        hook_beats = {i: (p.clip.length if p.clip else 1.0) / self.project.beat_seconds for i, p in pads.items()}
        return arrange.make_plan(pads, bpm=self.project.bpm, style=style, length=length, hook_beats=hook_beats,
                                 classify=self._classify_pad, roles=roles, variations=variations)

    def _song_total(self, plan) -> int:
        return int((plan.bars * 4 + 8) * self.project.beat_seconds * SR)

    def _part_voices(self, index: int, pad, role: str, ev, notes: str) -> list[tuple[Note, float]]:
        """The sounds one hit of a part starts: [(note, delay in seconds)]."""
        key, key_root = self.project.key, self.key_root()
        pc, mode = theory.parse_key(key)
        sustain = notes != "full"  # a full sample plays out as recorded (pitched); the rest sustain while held
        if role in ("chords", "bass", "lead"):
            if role == "chords":
                midis = theory.chord_notes(key, ev.degree, pad.flavour)
                gain = ev.velocity * pad.gain * 0.9 / np.sqrt(len(midis))
                strum = 0.035 if pad.strum else 0.0
            else:
                midis = [theory.degree_midi(pc, mode, ev.degree) - 24 if role == "bass" else
                         theory.chord_notes(key, ev.degree)[ev.tone % 3] + 12]
                gain, strum = ev.velocity * pad.gain, 0.0
            out = []
            for k, m in enumerate(midis):
                buf, loop = self.instrument.note(pad, m, key_root, sustain=sustain)
                out.append((Note(buf, gain, loop), k * strum))
            return out
        if notes in ("short", "full") and pad.clip is not None:
            return [(Note(pad.buffer(), ev.velocity * pad.gain), 0.0)]
        if notes == "hold" and pad.clip is not None:
            buf, loop = pad.freeze_buffer()
            return [(Note(buf, ev.velocity * pad.gain, loop), 0.0)]
        return [(n, n.delay) for n in self._pad_voices(index, ev.velocity)]

    def _render_part(self, plan, index: int, pad, notes: str, total: int) -> np.ndarray:
        """One pad's whole part as audio (``total`` frames from the song's start)."""
        from bisect import bisect_right
        beat = self.project.beat_seconds
        role = plan.roles[index]
        # when this part plays next (any pad playing the same part): where "hold" ends and "full" is cut
        hits = sorted({e.beat for j, evs in plan.events.items() if plan.roles[j] == role for e in evs})
        mix = np.zeros((total, 2), np.float32)
        for ev in plan.events[index]:
            k = bisect_right(hits, ev.beat + 1e-6)
            until = (hits[k] if k < len(hits) else plan.bars * 4) - ev.beat
            held = {"short": min(ev.length, 0.5), "hold": max(ev.length, until)}.get(notes, ev.length)
            limit = {"short": min(ev.length, 0.5), "full": until}.get(notes)  # beats
            at = int(ev.beat * beat * SR)
            for note, delay in self._part_voices(index, pad, role, ev, notes):
                frames = int(held * beat * SR) - int(delay * SR)
                audio = render_note(Note(note.buf, note.gain, note.loop, 0.0, note.gate), max(1, frames))
                if limit is not None:
                    audio = _cut(audio, int(limit * beat * SR) - int(delay * SR))
                a = at + int(delay * SR)
                n = min(len(audio), total - a)
                if n > 0:
                    mix[a:a + n] += audio[:n]
        return mix

    @staticmethod
    def _part_name(pad, role: str) -> str:
        name = pad.name or "synth"
        return name if role in name.lower().replace("-", "") else f"{name} · {role}"  # "Kick" not "Kick · kick"

    def make_song(self, indices: list[int], style: str = "auto", length: str = "full",
                  roles: dict[int, str] | None = None, notes: dict[int, str] | None = None) -> list[Track]:
        pads = self._song_pads(indices)
        if not pads:
            self.engine_label.setText("Nothing to arrange — those pads have no sound")
            return []
        roles, notes = dict(roles or {}), dict(notes or {})
        plan = self._song_plan(pads, style, length, roles)
        total = self._song_total(plan)
        start_bar = int(self.player.position // self.project.bar_seconds)
        start = start_bar * self.project.bar_seconds
        rendered = [(i, self._render_part(plan, i, pads[i], notes.get(i, "auto"), total))
                    for i, events in plan.events.items() if events]
        if not rendered:
            self.engine_label.setText("Nothing to arrange — those pads have no sound")
            return []
        # level the parts together (same balance) so the new song peaks at -1 dBFS instead of clipping
        together = np.zeros((total, 2), np.float32)
        for _, mix in rendered:
            together += mix
        peak = float(np.abs(together).max())
        scale = min(1.0, 0.89 / peak) if peak > 0 else 1.0
        del together
        part = length.split(":")[1] if length.startswith("part:") else None
        self._checkpoint(f"Make a {part} from pads" if part else "Make a song from pads")
        end = start + total / SR
        muted = [] if part else [t for t in self.project.tracks if self.project.audible(t)  # a part slots in
                                 and any(cl.start < end and cl.start + cl.length > start for cl in t.clips)]
        for t in muted:  # what was already there would play over (and clash with) the new song
            t.muted = True
        tracks = []
        for i, mix in rendered:
            nz = np.flatnonzero(np.abs(mix).max(axis=1) > 1e-4)
            if not len(nz):
                continue  # this pad never actually sounded in the arrangement
            mix = mix[: nz[-1] + 1] * scale
            pad, role = pads[i], plan.roles[i]
            path = self.project.write_wav(mix, f"song-{pad_key(i)}")
            # everything needed to re-render just this part later (Song part menu / ⟳)
            gen = {"mode": "song", "pad": i, "pads": sorted(pads), "style": plan.style, "length": length,
                   "roles": {str(k): v for k, v in roles.items()}, "notes": {str(k): v for k, v in notes.items()},
                   "variations": {}, "role": role, "scale": scale, "start": start, "bpm": self.project.bpm,
                   "key": self.project.key}
            name = self._part_name(pad, role) + (f" ({part})" if part else "")
            track = Track(name=self._unique_name(name), kind=family(pad.kind), clips=[Clip(path, start)],
                          prompt=f"{plan.style} arrangement", gen=gen)
            self.project.tracks.append(track)
            tracks.append(track)
        self.refresh()
        self._changed()
        what = f"{plan.bars}-bar {plan.style} {part}" if part else f"{plan.bars}-bar {plan.style} song"
        self.engine_label.setText(f"Made a {what} from {len(tracks)} pads (from bar {start_bar + 1}) — press Space"
                                  + (f" · muted {len(muted)} track{'s' * (len(muted) > 1)} it overlapped "
                                     "(unmute to layer them)" if muted else ""))
        log.info("song from pads %s: %s, %d bars, %d tracks, level %.2f, roles %s, notes %s, muted %s",
                 sorted(indices), plan.style, plan.bars, len(tracks), scale, roles, notes, [t.name for t in muted])
        return tracks

    # --- changing one part afterwards ---------------------------------------------------------------
    def render_song_part(self, track: Track, role: str | None = None, notes: str | None = None,
                         vary: bool = False) -> bool:
        """Re-render one song track in place: another part, another way of playing its notes, a new
        variation, or just picking up edits to its pad. Same level as the rest of the song."""
        g = track.gen
        i = g["pad"]
        if g.get("bpm") and g["bpm"] != self.project.bpm:
            self.engine_label.setText(f"This song was made at {g['bpm']} BPM — set the tempo back to redo its parts")
            return False
        pads = self._song_pads(g["pads"])
        if i not in pads:
            self.engine_label.setText(f"Pad {pad_key(i)} has no sound any more — put one back to redo this part")
            return False
        roles = {int(k): v for k, v in g.get("roles", {}).items()}
        notes_by = {int(k): v for k, v in g.get("notes", {}).items()}
        variations = {int(k): v for k, v in g.get("variations", {}).items()}
        if role is not None:
            roles[i] = role
        if notes is not None:
            notes_by[i] = notes
        if vary:
            variations[i] = variations.get(i, 0) + 1
        plan = self._song_plan(pads, g["style"], g["length"], roles, variations)
        mix = self._render_part(plan, i, pads[i], notes_by.get(i, "auto"), self._song_total(plan))
        nz = np.flatnonzero(np.abs(mix).max(axis=1) > 1e-4)
        if not len(nz):
            self.engine_label.setText(f"As {ROLE_TEXT[plan.roles[i]]}, pad {pad_key(i)} wouldn't play at all "
                                      "in this song — kept it as it was")
            return False
        old_role = track.gen.get("role") or self._part_role(track)
        path = self.project.write_wav(mix[: nz[-1] + 1] * g.get("scale", 1.0), f"song-{pad_key(i)}")
        start = track.clips[0].start - track.clips[0].offset if track.clips else g.get("start", 0.0)
        what = ("new variation" if vary else f"plays {ROLE_TEXT[role]}" if role else
                f"notes: {dict(NOTES)[notes].lower()}" if notes else "re-rendered from the pad")
        self._checkpoint(f"Song part: {what}")
        track.clips = [Clip(path, start)]
        track.gen = {**g, "roles": {str(k): v for k, v in roles.items()},
                     "notes": {str(k): v for k, v in notes_by.items()},
                     "variations": {str(k): v for k, v in variations.items()}, "role": plan.roles[i]}
        if role is not None and track.name.split(" #")[0] == self._part_name(pads[i], old_role):
            track.name = self._unique_name(self._part_name(pads[i], role))
        self.refresh()
        self._changed()
        self.engine_label.setText(f"{track.name}: {what}")
        log.info("song part %s pad %d: %s", track.name, i, what)
        return True

    def _part_role(self, track: Track) -> str:
        """The part a song track plays (worked out again for tracks made before it was stored)."""
        g = track.gen
        pads = self._song_pads(g["pads"])
        roles = {int(k): v for k, v in g.get("roles", {}).items()}
        if g["pad"] in roles:
            return roles[g["pad"]]
        from .. import arrange
        pad = self.project.pads[g["pad"]]
        return arrange.role_of(pad, self._classify_pad) if g["pad"] in pads else "stab"

    def song_part_items(self, track: Track) -> list:
        """Right-click → Song part ▸ for a track made by Make a song."""
        from .. import arrange
        g = track.gen
        i = g["pad"]
        pad = self.project.pads[i]
        role = g.get("role") or self._part_role(track)
        note = g.get("notes", {}).get(str(i), "auto")
        parts = [(("● " if r == role else "   ") + ROLE_TEXT[r], lambda r=r: self.render_song_part(track, role=r))
                 for r in arrange.ROLES if r != "chords" or pad.play == "chord"]
        notes = [(("● " if v == note else "   ") + text + (f" ({MODE_TEXT.get(pad.mode, pad.mode)})" if v == "auto" else ""),
                  lambda v=v: self.render_song_part(track, notes=v)) for v, text in NOTES]
        return [("Notes", notes), ("Plays", parts),
                ("⟳ New variation", lambda: self.render_song_part(track, vary=True)),
                (f"Re-render from pad {pad_key(i)} (after editing the pad)", lambda: self.render_song_part(track))]

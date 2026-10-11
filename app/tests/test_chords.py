"""End-to-end: any pad can play chords or notes of the key, set up from the inspector like a person
would (offscreen, silent). Also: songs from the old fixed-chord-row layout still open correctly."""

import json
import os

os.environ["LAYERDAW_NO_AUDIO"] = "1"

import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402
from PySide6.QtCore import QPoint, Qt  # noqa: E402
from PySide6.QtTest import QTest  # noqa: E402
from PySide6.QtWidgets import QApplication, QFileDialog, QInputDialog, QToolButton  # noqa: E402

from harness import activate, check, cleanup, demo_song, fake_engine, finish, open_window  # noqa: E402
from layerdaw import theory  # noqa: E402
from layerdaw.chords import detect_pitch  # noqa: E402
from layerdaw.model import SR, Project  # noqa: E402
from layerdaw.ui import main_window as mw  # noqa: E402
from layerdaw.ui.timeline import ROW_H  # noqa: E402

check([theory.chord_name("F# minor", d) for d in range(8)] == ["F#m", "G#°", "A", "Bm", "C#m", "D", "E", "F#m"],
      "F# minor chord names")
check([theory.chord_name("C major", d, "7th") for d in range(7)] == ["Cmaj7", "Dm7", "Em7", "Fmaj7", "G7", "Am7", "Bø7"],
      "C major sevenths")

fake_engine()
mw.MainWindow._show_menu = lambda self, items, pos: None
song = demo_song()  # A minor, 90 BPM
app, w = open_window(song, (1400, 820))
w.pads_dock.show()
QApplication.processEvents()
grid, ins = w.pad_panel.grid, w.pad_panel.inspector


def pump(times=1):
    for _ in range(times):
        w.player._callback(np.zeros((256, 2), np.float32), 256, None, None)


def settle():
    """Let an edit's preview (released by a 450 ms timer) finish, like real time passing."""
    QTest.qWait(500)
    pump(40)


def click_pad(i, mods=Qt.NoModifier):
    b = grid.buttons[i]
    QTest.mouseClick(b, Qt.LeftButton, mods, QPoint(b.width() // 2, 8))
    pump(30)


def menu_item(button_text, item_text):
    """Pick ``item_text`` from one of the inspector's ▾ menus."""
    b = next(b for b in ins.findChildren(QToolButton) if b.text() == button_text)
    next(a for a in b.menu().actions() if a.text() == item_text).trigger()


check(all(p.play == "sample" and p.clip is None for p in w.project.pads), "a new song's pads start empty")

# --- turn the Q pad into a chord pad from the inspector
click_pad(8)
check(grid.selected == [8] and ins.title.text() == "Pad Q", "clicking Q selects it in the inspector")
check(not ins.chord_box.isVisibleTo(ins), "a sample pad shows no chord controls")
ins.play_buttons["chord"].click()
pad8 = w.project.pads[8]
check(pad8.play == "chord" and pad8.mode == "hold", "Play as Chord makes Q a held chord pad")
check(grid.buttons[8].big == ("Am", "i"), f"Q shows its chord and numeral {grid.buttons[8].big}")
check(ins.chord_box.isVisibleTo(ins) and ins.degree.count() == 8 and ins.degree.itemText(3).endswith("Dm"),
      "the inspector lists the key's chords to pick from")
check(ins.info.text() == "built-in synth", "with no sound on it, a chord pad uses the built-in synth")
pump(80)

settle()
QTest.keyPress(w.lanes, Qt.Key_Q)
pump()  # the audio thread picks the new voices up
check(w.player.voice_count == 3, "Q plays a 3-note chord")
out = np.zeros((256, 2), np.float32)
w.player._callback(out, 256, None, None)
check(np.abs(out).max() > 0.01, "the chord is audible")
pump(400)
check(w.player.voice_count == 3, "the chord sustains while Q is held")
QTest.keyRelease(w.lanes, Qt.Key_Q)
pump(30)
check(w.player.voice_count == 0, "releasing Q fades the chord out")

# --- the next three pads in one go: shift-click a range, Play as ▾ Chord, then a chord each
click_pad(11, Qt.ShiftModifier)
check(grid.selected == [8, 9, 10, 11] and ins.stack.currentIndex() == 1 and "QWER" in ins.multi_title.text().replace(" ", ""),
      f"shift-click selects Q..R and shows the multi-pad page ({ins.multi_title.text()})")
menu_item("Play as ▾", "Chord")
check(all(w.project.pads[i].play == "chord" for i in range(8, 12)), "Play as ▾ Chord turns all four into chord pads")
for i, degree in ((9, 3), (10, 4), (11, 5)):
    click_pad(i)
    ins.degree.activated.emit(degree)  # what picking from the combo does
labels = [grid.buttons[i].big[0] for i in range(8, 12)]
check(labels == ["Am", "Dm", "Em", "F"], f"each pad got its own chord {labels}")
check(ins.degree.currentIndex() == 5, "the combo shows the chosen chord")
pump(80)

settle()
QTest.keyPress(w.lanes, Qt.Key_G)
check(grid.buttons[8].big[0] == "Am7", "holding G shows 7th chords on the chord pads")
QTest.keyPress(w.lanes, Qt.Key_Q)
pump()  # the audio thread picks the new voices up
check(w.player.voice_count == 4, "G + Q plays Am7 (4 notes)")
QTest.keyRelease(w.lanes, Qt.Key_Q)
QTest.keyRelease(w.lanes, Qt.Key_G)
check(grid.buttons[8].big[0] == "Am", "letting go of G goes back to triads")
pump(30)

# --- flavour, strum and mode belong to one pad
click_pad(8)
pump(40)
ins.flavour_buttons["sus4"].click()
check(pad8.flavour == "sus4" and grid.buttons[8].big[0] == "Asus4", "sus4 latches on Q")
check(w.project.pads[9].flavour == "triad" and grid.buttons[9].big[0] == "Dm", "…and only on Q")
ins.mode_buttons["oneshot"].click()
check(pad8.mode == "oneshot" and ins.flavour_buttons["sus4"].isChecked(), "changing the mode keeps the flavour")
check(not ins.mode_buttons["loop"].isVisibleTo(ins), "chord pads only offer Hold / One-shot")
ins.mode_buttons["hold"].click()
ins.flavour_buttons["triad"].click()
ins.strum.setChecked(True)
check(pad8.strum and [round(v.delay, 3) for v in w._pad_voices(8, 1.0)] == [0.0, 0.035, 0.07],
      "Strum rolls Q's notes in")
check(all(v.delay == 0 for v in w._pad_voices(9, 1.0)), "W isn't strummed")
ins.strum.setChecked(False)
pump(80)

# --- a tonal sound on a chord pad: its pitch is detected and the chords are built from it
stem = song.parent / "Rhodes_A3.wav"
t = np.arange(2 * SR) / SR
tone = 0.4 * np.sin(2 * np.pi * 220 * t) + 0.15 * np.sin(2 * np.pi * 440 * t)
sf.write(stem, np.stack([tone, tone], 1).astype(np.float32), SR)
w.pad_file_dropped(8, str(stem))
pad8 = w.project.pads[8]
check(pad8.play == "chord" and pad8.name == "Rhodes_A3" and grid.buttons[8].big[0] == "Am",
      "dropping a WAV on a chord pad keeps it the same chord, with the new sound")
root = w.instrument.root_of(pad8, w.key_root())
check(abs(root - 57) < 0.2, f"the sound's pitch is detected (A3 = 57, got {root:.2f})")
check(ins.info.text().endswith("s"), f"the inspector shows the sound's length ({ins.info.text()})")
pump(60)
kr = w.key_root()
buf_a, buf_c = w.instrument.note(pad8, 57, kr, sustain=False)[0], w.instrument.note(pad8, 60, kr, sustain=False)[0]
check(abs(len(buf_c) / len(buf_a) - 2 ** (-3 / 12)) < 0.01, "C is the A3 sound pitched up 3 semitones")
played = detect_pitch(w.instrument.note(pad8, 64, kr, sustain=False)[0])
check(played is not None and abs(((played - 64) + 6) % 12 - 6) < 0.3, f"the E note really sounds as E ({played:.2f})")
shifts = [round(((m - root + 6) % 12) - 6) for m in w.chord_midis(pad8)]
check(all(abs(s) <= 6 for s in shifts), f"no note is pitched more than a tritone from the sound ({shifts})")

# --- note pads: one note of the key; an 808 sits two octaves down
click_pad(4)
ins.play_buttons["note"].click()
pad4 = w.project.pads[4]
note = theory.note_name(w.note_midi(pad4))
check(pad4.play == "note" and grid.buttons[4].big == (note, "i") and ins.degree_label.text() == "Note",
      f"Play as Note shows the note ({grid.buttons[4].big})")
check(not ins.flavour_buttons["7th"].isVisibleTo(ins) and not ins.strum.isVisibleTo(ins), "note pads hide the chord flavours")
settle()
QTest.keyPress(w.lanes, Qt.Key_A)
pump()  # the audio thread picks the new voices up
check(w.player.voice_count == 1, "A plays one note")
QTest.keyRelease(w.lanes, Qt.Key_A)
pump(40)
lead = w.note_midi(pad4)
ins.category.setCurrentText("808")
ins.category.activated.emit(ins.category.currentIndex())
check(w.note_midi(pad4) == lead - 24, "an 808 note pad plays two octaves lower")
ins.degree.activated.emit(4)
check(grid.buttons[4].big[0] == theory.note_name(lead - 24 + 7), f"picking the v makes it the fifth ({grid.buttons[4].big})")
pump(80)

# --- chopping a clip onto selected pads only touches those
tk = np.arange(int(60 / 90 * SR)) / SR
kick_t = np.sin(2 * np.pi * (45 + 100 * np.exp(-tk / 0.03)) * tk) * np.exp(-tk / 0.09)
QTest.mouseClick(w.lanes, Qt.LeftButton, pos=QPoint(60, ROW_H // 2))
click_pad(0)
click_pad(3, Qt.ShiftModifier)
menu_item("Chop onto these ▾", "At each hit, in order")
filled = [i for i, p in enumerate(w.project.pads) if p.clip]
check(filled == [0, 1, 2, 3, 8], f"chopping fills just the selected pads ({filled})")
check(all(w.project.pads[i].play == "chord" for i in range(8, 12)), "the chord pads are untouched")
QFileDialog.getOpenFileName = staticmethod(lambda *a, **k: (str(song.parent / "loop.wav"), ""))
QInputDialog.getItem = staticmethod(lambda *a, **k: ("Every beat", True))
sf.write(song.parent / "loop.wav", np.tile(np.stack([kick_t, kick_t], 1), (8, 1)).astype(np.float32), SR)
click_pad(13)
click_pad(15, Qt.ShiftModifier)
menu_item("Chop onto these ▾", "An audio file from disk…")
check([w.project.pads[i].name for i in (13, 14, 15)] == ["loop 1", "loop 2", "loop 3"],
      "Chop onto these ▾ → An audio file slices a file from disk across the selected pads")
pump(80)

# --- recording a held chord
w.player.seek(0.0)
w.pad_panel.rec.setChecked(True)
QTest.keyPress(w.lanes, Qt.Key_W)
w.player.seek(1.0)
QTest.keyRelease(w.lanes, Qt.Key_W)
w.player.stop()
w._tick()
take = w.project.tracks[-1]
check(take.name.startswith("pads take") and 1.0 <= take.clips[0].length <= 1.25,
      f"a held chord records for as long as it was held ({take.clips[0].length:.2f}s)")

settle()
w.midi.note_on.emit(44, 0.9)
check(w.player.voice_count >= 3, "MIDI note 44 (pad Q) plays its chord")
w.midi.note_off.emit(44)
pump(40)

# --- undo, save, reload
click_pad(12)
ins.play_buttons["chord"].click()
check(w.project.pads[12].play == "chord", "pad 1 becomes a chord pad")
QTest.keyClick(w.lanes, Qt.Key_Z, Qt.ControlModifier)
check(w.project.pads[12].play == "sample" and grid.buttons[12].big is None, "Ctrl+Z turns it back")
pump(40)

w.project.save()
again = Project.load(song)
p8, p11, p4 = again.pads[8], again.pads[11], again.pads[4]
check(p8.play == "chord" and p8.clip is not None and p8.name == "Rhodes_A3" and abs(p8.root - 57) < 0.2
      and p11.play == "chord" and p11.degree == 5 and p11.clip is None,
      "chord pads (sound, chord, detected pitch) are saved with the song")
check(p4.play == "note" and p4.kind == "808" and p4.degree == 4, "the 808 note pad is saved too")

w.bpm.setFocus()
QTest.keyClick(w.bpm, Qt.Key_G)
check(w._held_flavours == [], "G typed into a field isn't a flavour key")
activate(w)
w.close()

# --- a song saved before per-pad chords: QWER were chord slots sharing one sound, 1234 hooks
legacy = song.parent / "legacy"
(legacy / "audio").mkdir(parents=True)
d = 0.4 * np.sin(2 * np.pi * 293.66 * t)
sf.write(legacy / "audio" / "Pad_D.wav", np.stack([d, d], 1).astype(np.float32), SR)
old_pad = {"name": "vox", "gain": 1.0, "pitch": 0, "kind": "vocals", "mode": "oneshot",
           "clip": {"path": "audio/Pad_D.wav", "offset": 0.0, "length": 1.0}}
(legacy / "project.json").write_text(json.dumps({
    "bpm": 90, "key": "A minor", "beats_per_bar": 4, "tracks": [],
    "pads": [None] * 12 + [old_pad] + [None] * 3,
    "chords": {"enabled": True, "slots": [0, 3, 4, 6], "flavour": "7th", "strum": True,
               "source": {"path": "audio/Pad_D.wav", "offset": 0.0, "length": 2.0, "name": "Pad_D", "root": 62.0}}}))
old = Project.load(legacy)
check([p.play for p in old.pads[8:12]] == ["chord"] * 4 and [p.degree for p in old.pads[8:12]] == [0, 3, 4, 6],
      "an old song's chord row becomes four chord pads with the same chords")
check(all(p.flavour == "7th" and p.strum and p.name == "Pad_D" and p.root == 62.0 and p.clip is not None
          and p.kind == "keys" for p in old.pads[8:12]), "…each with the old shared chord sound, flavour and strum")
check(old.pads[12].kind == "hook" and old.pads[12].clip is not None, "the old hook row keeps its sounds, as hooks")
check(all(p.clip is None and p.play == "sample" for p in old.pads[:8]), "the drum pads are untouched")
app, w2 = open_window(legacy)
labels = [w2.pad_panel.grid.buttons[i].big[0] for i in range(8, 12)]
check(labels == ["Am7", "Dm7", "Em7", "G7"], f"it opens showing those chords {labels}")
w2.select_pad(8)
check(w2.pad_panel.inspector.category.currentText() == "keys", "…as keys in the inspector")
w2.project.pads[9].kind = "guitar"
w2.select_pad(9)
check(w2.pad_panel.inspector.category.currentText() == "guitar", "a category from a track (guitar) shows as itself, not 'other'")
w2.project.pads[8].play = "sample"
w2.project.save()
check(Project.load(legacy).pads[8].play == "sample", "once saved, it isn't migrated again")
w2.close()

cleanup(song)
finish()

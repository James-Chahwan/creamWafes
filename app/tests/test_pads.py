"""End-to-end: the pad sampler, used like a person would (offscreen, fake engine, silent audio)."""

import os

os.environ["LAYERDAW_NO_AUDIO"] = "1"

import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402
from PySide6.QtCore import QMimeData, QPoint, QPointF, Qt, QUrl  # noqa: E402
from PySide6.QtGui import QDropEvent  # noqa: E402
from PySide6.QtTest import QTest  # noqa: E402
from PySide6.QtWidgets import QApplication, QFileDialog, QInputDialog  # noqa: E402

from harness import SENT, activate, check, cleanup, deliver, demo_song, fake_engine, finish, open_window  # noqa: E402
from layerdaw.model import SR, Project  # noqa: E402
from layerdaw.ui import main_window as mw  # noqa: E402
from layerdaw.ui.timeline import ROW_H  # noqa: E402

song = demo_song()
drums = sf.read(song / "audio" / next(p.name for p in (song / "audio").iterdir() if "drums" in p.name),
                dtype="float32")[0]
fake_engine(drums)
menus = []
mw.MainWindow._show_menu = lambda self, items, pos: menus.append(self._build_menu(items))
app, w = open_window(song, (1300, 820))


def pump(frames=256, times=1):
    for _ in range(times):
        w.player._callback(np.zeros((frames, 2), np.float32), frames, None, None)


check(not w.pads_dock.isVisible(), "pads stay hidden until used")
QTest.keyClick(w.lanes, Qt.Key_P)
check(w.pads_dock.isVisible(), "P shows the pads")
QTest.mouseClick(w.lanes, Qt.LeftButton, pos=QPoint(60, ROW_H // 2))
check(w.state.selected_clip is w.project.tracks[0].clips[0], "click the drums clip to select it")
w.chop_to_pads("hits")
check(sum(1 for p in w.project.pads if p.clip) == 16, "Chop → at each hit fills 16 pads")

QTest.keyClick(w.lanes, Qt.Key_Z)
QTest.keyClick(w.lanes, Qt.Key_X)
check(w.player.voice_count == 2, "keys Z and X trigger two voices")
out = np.zeros((256, 2), np.float32)
w.player._callback(out, 256, None, None)
check(np.abs(out).max() > 0.01, "the voices are audible in the audio callback")
pump(times=400)
check(w.player.voice_count == 0, "finished voices are released")
QTest.mouseClick(w.pad_panel.grid.buttons[3], Qt.LeftButton)
check(w.player.voice_count == 1, "clicking a pad with the mouse plays it")
pump(times=400)

w.pad_menu(0, QPoint())
pitch = next(a for a in menus[-1].actions() if a.text() == "Pitch").menu()
before = len(w.project.pads[0].buffer())
next(a for a in pitch.actions() if a.text().strip(" ●").startswith("+12")).trigger()
check(w.project.pads[0].pitch == 12 and abs(len(w.project.pads[0].buffer()) - before / 2) < 50,
      "right-click → Pitch +12 = an octave up")

w.player.seek(0.0)
w.pad_panel.rec.setChecked(True)
check(w.player.playing, "● Record starts playback")
for t, key in ((0.01, Qt.Key_Z), (0.68, Qt.Key_X), (1.30, Qt.Key_C)):
    w.player.seek(t)
    QTest.keyClick(w.lanes, key)
w.player.stop()
w._tick()
take = w.project.tracks[-1]
check(take.name == "pads take" and not w.pad_panel.rec.isChecked(), "stopping playback finishes the pad take")
check(abs(take.clips[0].start) < 1e-6 and take.clips[0].length > 1.3, "the take starts at bar 1 and spans the hits")
QTest.keyClick(w.lanes, Qt.Key_Z, Qt.ControlModifier)
check(take not in w.project.tracks, "Ctrl+Z removes the pad take")

# --- hold mode: sustain while the key is down, fade on release; one-shots ignore key-up
from layerdaw.audio import RELEASE  # noqa: E402

pad4 = w.project.pads[4]
check(pad4.mode == "oneshot", "drum chops default to one-shot")
slice_len = len(pad4.buffer())
from layerdaw.model import load_audio  # noqa: E402
src = load_audio(pad4.clip.path)
a = int(round(pad4.clip.offset * SR))


def hold_and_capture(mode: str, key=Qt.Key_A) -> np.ndarray:
    pad4.mode = mode
    QTest.keyPress(w.lanes, key)
    blocks = []
    for _ in range(int(slice_len / 256) * 4):  # hold for 4× the slice length, capturing the output
        b = np.zeros((256, 2), np.float32)
        w.player._callback(b, 256, None, None)
        blocks.append(b)
    QTest.keyRelease(w.lanes, key)
    pump(times=RELEASE // 256 + 2)
    return np.concatenate(blocks)[:, 0]


after = slice(slice_len + 2000, None)  # well past where the slice would have ended
played = hold_and_capture("playon")
match = np.corrcoef(played[after], (src[a:a + len(played), 0] * pad4.gain)[after])[0, 1]
check(match > 0.99, f"Play-on keeps playing the original audio past the slice (match {match:.3f})")

held = hold_and_capture("hold")
bleed = abs(np.corrcoef(held[after], src[a:a + len(held), 0][after])[0, 1])
env = np.sqrt((held[after][: len(held[after]) // 2400 * 2400].reshape(-1, 2400) ** 2).mean(axis=1))
check(env.min() > 1e-3 and bleed < 0.2,
      f"Hold sustains this sound only — nothing that came after it (bleed {bleed:.2f})")
check(env.std() / env.mean() < 0.3, f"Hold sustains at a steady level (wobble {env.std() / env.mean():.2f})")
check(w.player.voice_count == 0, "letting go fades it out within 0.12 s")

pad4.mode = "loop"
QTest.keyPress(w.lanes, Qt.Key_A)
pump(times=int(slice_len / 256) * 3)
check(w.player.voice_count == 1, "Loop mode repeats the slice while held")
QTest.keyRelease(w.lanes, Qt.Key_A)
pump(times=RELEASE // 256 + 2)
_, loop = pad4.hold_buffer()
hb = pad4.hold_buffer()[0].mean(axis=1)
jump = abs(hb[loop[1] - 1] - hb[loop[0] - 1])
check(jump < 0.05, f"the loop seam is crossfaded (end→start step {jump:.4f})")

pad0 = w.project.pads[1]
QTest.keyPress(w.lanes, Qt.Key_X)
QTest.keyRelease(w.lanes, Qt.Key_X)
pump(times=(RELEASE // 256) + 4)
check(len(pad0.buffer()) < RELEASE * 1.5 or w.player.voice_count == 1, "a quick tap on a one-shot still plays the whole hit")
pump(times=400)

pad4.mode = "hold"
w.player.seek(0.0)
w.pad_panel.rec.setChecked(True)
QTest.keyPress(w.lanes, Qt.Key_A)
w.player.seek(1.5)
QTest.keyRelease(w.lanes, Qt.Key_A)
w.player.stop()
w._tick()
held_take = w.project.tracks[-1]
check(held_take.name.startswith("pads take") and 1.5 <= held_take.clips[0].length <= 1.5 + 0.2,
      f"a recorded held note lasts as long as it was held ({held_take.clips[0].length:.2f}s for 1.5 s)")
w.undo()
pad4.mode = "oneshot"

# --- the pad editor: one pad at a time
QTest.mouseClick(w.pad_panel.grid.buttons[2], Qt.LeftButton)
pump(times=400)
check(w.primary_pad == 2 and w.pad_panel.grid.buttons[2].selected and "Pad C" in w.pad_panel.inspector.title.text(),
      "clicking a pad selects it in the editor")
w.pad_panel.inspector.mode_buttons["loop"].click()
check(w.project.pads[2].mode == "loop", "editor: Loop button sets the mode")
w.pad_panel.inspector.mode_buttons["oneshot"].click()
p_before, g_before = w.project.pads[2].pitch, w.project.pads[2].gain
w.pad_panel.inspector.pitch_step.emit(1)
w.pad_panel.inspector.gain_step.emit(1.413)
check(w.project.pads[2].pitch == p_before + 1 and abs(w.project.pads[2].gain - g_before * 1.413) < 1e-6,
      "editor: ♯ and + change pitch and volume")
pump(times=400)
start = w.project.pads[2].clip.offset
w.pad_panel.inspector.replace.emit("next")
nxt = w.project.pads[2].clip.offset
check(nxt > start + 0.05, f"Replace ▸ Next hit moves to the following hit ({start:.2f}s → {nxt:.2f}s)")
w.pad_panel.inspector.replace.emit("prev")
check(abs(w.project.pads[2].clip.offset - start) < 0.02, "Replace ▸ Previous hit goes back")
pump(times=400)
clip = w.project.tracks[0].clips[0]
w.state.selected_clip = clip
beat = w.project.beat_seconds
w.player.seek(clip.start + 2 * beat + 0.01)  # just after a hit
w.pad_panel.inspector.replace.emit("playhead")
got = w.project.pads[2].clip.offset - clip.offset
check(abs(got - 2 * beat) < 0.03 or abs(got - 2.5 * beat) < 0.03,
      f"Replace ▸ hit at the playhead grabs that hit ({got:.3f}s, beat 2 = {2 * beat:.3f}s)")
w.state.selection = (w.project.bar_seconds, 2 * w.project.bar_seconds)
w.pad_panel.inspector.replace.emit("bars")
check(abs(w.project.pads[2].clip.length - w.project.bar_seconds) < 0.01, "Replace ▸ the selected bars (1 bar)")
w.state.selection = None
pump(times=600)
from PySide6.QtCore import QTimer  # noqa: E402
from layerdaw.ui import creative  # noqa: E402


def drive(dialog_cls, action):
    def go():
        d = next(x for x in QApplication.topLevelWidgets() if isinstance(x, dialog_cls) and x.isVisible())
        action(d)
    QTimer.singleShot(150, go)


others = [w.project.pads[i].clip.offset if w.project.pads[i].clip else None for i in (0, 1, 3)]
drive(creative.SoundDialog, lambda d: (d.text.setText("deep punchy kick"), d.accept()))
w.pad_panel.inspector.replace.emit("generate")
check(SENT[-1][1]["duration"] >= 10 and SENT[-1][1]["caption"].startswith("deep punchy kick, kick drum"),
      "Replace ▸ Generate asks the engine for just that one sound")
drive(creative.SoundPicker, lambda d: d._use())
deliver(w)
check(w.project.pads[2].name.startswith("kick") and w.project.pads[2].kind == "drums",
      f"the picked sound lands on that pad ({w.project.pads[2].name})")
check([w.project.pads[i].clip.offset if w.project.pads[i].clip else None for i in (0, 1, 3)] == others,
      "…and only that pad")
pump(times=600)
w.select_pad(9)
w.set_pad_play("chord")
check(w.pad_panel.inspector.chord_box.isVisible() or w.pad_panel.inspector.play_buttons["chord"].isChecked(),
      "switching a pad to Chord shows its chord settings in the inspector")
w.set_pad_play("sample")
w.select_pad(0)

w.project.save()
again = Project.load(song)
check(sum(1 for p in again.pads if p.clip) == 16 and again.pads[0].pitch == 12, "pads are saved with the song")

# a stem file dropped onto a pad
stem = song.parent / "Bass_Stem.wav"
sf.write(stem, (0.3 * np.sin(2 * np.pi * 55 * np.arange(SR) / SR)).astype(np.float32), SR)
mime = QMimeData()
mime.setUrls([QUrl.fromLocalFile(str(stem))])
btn = w.pad_panel.grid.buttons[5]
drop = QDropEvent(QPointF(10, 10), Qt.CopyAction, mime, Qt.LeftButton, Qt.NoModifier)
btn.dropEvent(drop)
pad = w.project.pads[5]
check(pad.clip is not None and pad.name == "Bass_Stem" and pad.kind == "bass", "drop a WAV onto a pad loads it")
check(pad.clip.path.parent == w.project.audio_dir, "the dropped file is copied into the song folder")

QFileDialog.getOpenFileName = staticmethod(lambda *a, **k: (str(stem), ""))
w.load_pad_from_file(6)
check(w.project.pads[6].name == "Bass_Stem", "right-click → Load audio file… works")
QInputDialog.getItem = staticmethod(lambda *a, **k: ("Every beat", True))
w.select_pad(0)  # one pad selected → the chop fills all 16
w.chop_file_to_pads()
check(sum(1 for p in w.project.pads if p.clip) >= 2 and w.project.pads[0].name.startswith("Bass_Stem"),
      "Chop an audio file (stem) → pads")
bad = song.parent / "broken.wav"
bad.write_bytes(b"not audio")
w.load_pad_from_file(7, str(bad))
check("Couldn't load" in w.engine_label.text(), "a broken file shows a message instead of crashing")

w.undo()  # undo the stem chop → the drum pads come back, with the dropped stem still on pad 6
check(w.project.pads[0].name.startswith("drums") and w.project.pads[5].name == "Bass_Stem",
      "Ctrl+Z after chopping a stem restores the previous pads")
w.undo()  # undo "Load audio file" on pad 7
check(w.project.pads[6].name.startswith("drums"), "…and the next Ctrl+Z undoes the file load")

# a kit = one clean example of each drum sound, named, in drum-machine order
from harness import drum_loop  # noqa: E402
from layerdaw import chop  # noqa: E402

loop = drum_loop()
roles = [r for _, _, r, _ in chop.kit(loop, 60 / 120, 8)]
check(roles[:4] == ["kick", "snare", "hat", "clap"], f"kit picks kick, snare, hat, clap in order ({roles})")
gains = [g for _, _, _, g in chop.kit(loop, 60 / 120, 8)]
check(gains[2] > 1.5, f"the quiet hat gets levelled up (×{gains[2]:.1f})")

import harness  # noqa: E402

harness.fake_engine(loop)
QInputDialog.getText = staticmethod(lambda *a, **k: ("dusty boom bap", True))
n = len(w.project.tracks)
w.generate_kit()
check(SENT[-1][1]["duration"] >= 10, "kit request is at least 10 s (engine minimum)")
check(w.lanes.rows() == len(w.project.tracks) and w.pad_panel.grid.buttons[0].generating == 0.0,
      "a generating kit shows on the pads, not as a timeline row")
deliver(w)
names = [p.name for p in w.project.pads if p.clip]
check(len(w.project.tracks) == n and names[:4] == ["kick", "snare", "hat", "clap"] and
      w.pad_panel.grid.buttons[0].generating is None, f"Generate kit lands a named kit on the pads ({names})")

w.midi.note_on.emit(37, 0.8)
check(w.player.voice_count >= 1, "MIDI note 37 plays pad 2")
pump(times=400)

hdr = w.headers.headers[id(w.project.tracks[0])]
hdr.start_rename()
hdr.editor.setFocus()
QApplication.processEvents()
v = w.player.voice_count
QTest.keyClick(hdr.editor, Qt.Key_Z)
check(w.player.voice_count == v, "typing a track name doesn't fire pads")
activate(w)
w.bpm.setFocus()
w.bpm.lineEdit().selectAll()
QTest.keyClicks(w.bpm.lineEdit(), "124")
QTest.keyClick(w.bpm.lineEdit(), Qt.Key_Return)
check(w.player.voice_count == v and w.bpm.value() == 124, "typing digits in the BPM box doesn't fire pads")
w.lanes.setFocus()
QTest.keyClick(w.lanes, Qt.Key_Space)
check(w.player.playing, "Space plays after leaving the BPM box")

w.close()
cleanup(song)
finish()

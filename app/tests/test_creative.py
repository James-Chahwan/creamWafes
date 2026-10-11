"""End-to-end: progression writer, AI chord sound and the hook row (offscreen, fake engine, silent)."""

import os
import random

os.environ["CREAMWAFES_NO_AUDIO"] = "1"

import numpy as np  # noqa: E402
from PySide6.QtCore import QTimer  # noqa: E402
from PySide6.QtWidgets import QApplication, QDialog, QInputDialog  # noqa: E402

import harness  # noqa: E402
from harness import SENT, check, cleanup, deliver, demo_song, finish, open_window  # noqa: E402
from creamwafes import theory  # noqa: E402
from creamwafes.model import SR  # noqa: E402
from creamwafes.ui import creative  # noqa: E402
from creamwafes.ui import main_window as mw  # noqa: E402

mw.MainWindow._show_menu = lambda self, items, pos: None
harness.fake_engine()
song = demo_song()  # A minor, 90 BPM
app, w = open_window(song, (1400, 860))
w.pads_dock.show()
QApplication.processEvents()


def pump(times=1):
    for _ in range(times):
        w.player._callback(np.zeros((256, 2), np.float32), 256, None, None)


# --- progressions
prog = theory.write_progression("A minor", "dark", random.Random(1))
check(prog["mood"] == "dark" and len(prog["degrees"]) == 4 and all(0 <= d < 7 for d in prog["degrees"]),
      f"a dark progression in A minor: {' → '.join(prog['names'])}")
check(theory.mood_of("lofi chill beat") == "chill" and theory.mood_of("cinematic trailer") == "epic",
      "moods are understood from everyday words")
w.select_pads([8, 9, 10, 11])
w.progression_to_pads(prog)
want = [theory.chord_name("A minor", d) for d in theory.pad_slots(prog["degrees"])]
got = [w.pad_panel.grid.buttons[i].big[0] for i in range(8, 12)]
check(got == want and all(w.project.pads[i].play == "chord" for i in range(8, 12)),
      f"Progression onto the selected pads → {got}")
w.audition_progression(prog, 1)
check(w.player.voice_count == 1, "▶ Audition plays the progression")
pump(30)
n = len(w.project.tracks)
w.progression_to_timeline(prog, 1, 2)
track = w.project.tracks[-1]
expect = 4 * 2 * w.project.bar_seconds
check(len(w.project.tracks) == n + 1 and track.name.startswith("chords ·") and abs(track.clips[0].length - expect) < 0.4,
      f"Add to the timeline → an 8-bar chord track ({track.clips[0].length:.1f}s ≈ {expect:.1f}s)")

dlg_seen = []


def drive_progression_dialog():
    d = next(x for x in QApplication.topLevelWidgets() if isinstance(x, creative.ProgressionDialog) and x.isVisible())
    d.mood.setCurrentText("uplifting")
    d.write()
    dlg_seen.append(d.prog["mood"])
    d.reject()


QTimer.singleShot(150, drive_progression_dialog)
w.open_progression_writer()
check(dlg_seen == ["uplifting"], "the ✦ Progression… dialog writes for the mood you type")

# --- a chord pad's own timbre: load a held A3 tone onto it; its pitch is detected
import soundfile as sf  # noqa: E402

t = np.arange(3 * SR) / SR
tone = (0.4 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
tone_path = song.parent / "Rhodes_A3.wav"
sf.write(tone_path, np.stack([tone, tone], 1), SR)
w.select_pad(8)
w.pad_file_dropped(8, str(tone_path))
pad8 = w.project.pads[8]
check(pad8.play == "chord" and pad8.name == "Rhodes_A3", "dropping a sound on a chord pad keeps it a chord pad (new timbre)")
w.hit_pad(8)
check(pad8.root is not None and abs(pad8.root - 57) < 0.3, f"…and its pitch is detected ({pad8.root:.2f} ≈ A3)")
pump(400)

# --- hooks: phrases (bar-aligned, busy parts) and ear candy (distinct one-shots)
bar = w.project.bar_seconds
mel = np.zeros(int(8 * bar * SR), np.float32)
rng = np.random.default_rng(0)
for k in range(32):  # 8 bars of quarter-note melody, louder in bars 3–6
    f = 220 * 2 ** (rng.integers(0, 12) / 12)
    a0 = int(k * bar / 4 * SR)
    tt = np.arange(int(bar / 4 * SR)) / SR
    mel[a0:a0 + len(tt)] = np.sin(2 * np.pi * f * tt) * np.exp(-tt / 0.2) * (0.8 if 8 <= k < 24 else 0.2)
harness.fake_engine(np.stack([mel, mel], 1))


def answer_hooks(kind):
    def go():
        d = next(x for x in QApplication.topLevelWidgets() if isinstance(x, creative.HookDialog) and x.isVisible())
        d.prompt.setText("plucky synth lead")
        (d.phrases if kind == "phrases" else d.candy).setChecked(True)
        d.accept()
    QTimer.singleShot(150, go)


answer_hooks("phrases")
w.select_pads([12, 13, 14, 15])
w.generate_hooks()
check(all(w.pad_panel.grid.buttons[i].generating == 0.0 for i in range(12, 16)), "generating hooks shows on the selected pads")
deliver(w)
hooks = [w.project.pads[i] for i in range(12, 16)]
lengths = {round(p.clip.length / bar, 2) for p in hooks if p.clip}
starts = [round(p.clip.offset / bar, 2) for p in hooks if p.clip]
check(len([p for p in hooks if p.clip]) == 4 and lengths == {1.0}, f"4 one-bar phrases on the 4 selected pads ({lengths})")
check(all(abs(s - round(s)) < 1e-6 for s in starts) and all(2 <= s < 6 for s in starts),
      f"phrases are bar-aligned and taken from the busy part (bars {starts})")
before = w.player.voice_count  # the chord-sound preview may still be holding (its release is on a timer)
w.hit_pad(12)
check(w.player.voice_count == before + 1, "a hook pad plays its phrase")
pump(400)

boom = np.zeros(int(12 * SR), np.float32)
for k, (f, d) in enumerate(((80, 0.4), (600, 0.3), (2500, 0.2), (7000, 0.15), (300, 0.3))):
    a0 = int((1 + 2 * k) * SR)
    tt = np.arange(int(0.6 * SR)) / SR
    boom[a0:a0 + len(tt)] = np.sin(2 * np.pi * f * tt) * np.exp(-tt / d) * 0.8
harness.fake_engine(np.stack([boom, boom], 1))
answer_hooks("candy")
w.select_pads([12, 13, 14, 15])
w.generate_hooks()
deliver(w)
picked = [w.project.pads[i].clip for i in range(12, 16) if w.project.pads[i].clip]
check(len(picked) == 4 and len({round(c.offset) for c in picked}) == 4, "ear candy: 4 different one-shots")

w.project.save()
w.close()
cleanup(song)
finish()

"""End-to-end: the 4×4 you play plus a tray of pads that never runs out — collecting sounds there,
keeping every take, and making just a part of a song from what you pick (offscreen, silent)."""

import os

os.environ["LAYERDAW_NO_AUDIO"] = "1"

import numpy as np  # noqa: E402
from PySide6.QtCore import QPoint, Qt, QTimer  # noqa: E402
from PySide6.QtTest import QTest  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

import harness  # noqa: E402
from harness import SENT, activate, check, cleanup, deliver, demo_song, drum_loop, finish, open_window  # noqa: E402
from layerdaw.model import Project  # noqa: E402
from layerdaw.ui import creative  # noqa: E402
from layerdaw.ui.pad_grid import KEYED, note_to_pad, pad_key  # noqa: E402
from layerdaw.ui.timeline import ROW_H  # noqa: E402

SHOTS = os.environ.get("LAYERDAW_SHOTS")
harness.fake_engine(drum_loop(90, 8))
mw_menus = []
song = demo_song()
app, w = open_window(song, (1400, 900))
w.pads_dock.show()
QApplication.processEvents()
grid = w.pad_panel.grid
seen = {}


def silence():
    """Stop whatever is still sounding (e.g. a long clip's audition) before counting voices."""
    w.player._voices.clear()
    w.player._incoming.clear()


def pump(n=1):
    for _ in range(n):
        w.player._callback(np.zeros((256, 2), np.float32), 256, None, None)


def drive(cls, action):
    def go():
        d = next(x for x in QApplication.topLevelWidgets() if isinstance(x, cls) and x.isVisible())
        action(d)
    QTimer.singleShot(150, go)


check(len(grid.buttons) == KEYED + 4 and grid.tray_label.isVisible(), "the 4×4, then a tray row below it")
check([pad_key(i) for i in (0, 15, 16, 41)] == ["Z", "4", "17", "42"], "4×4 pads go by key, tray pads by number")
check(note_to_pad(36) == 0 and note_to_pad(51) == 15 and note_to_pad(52) == 16 and note_to_pad(24) == 4,
      "MIDI 36–51 play the 4×4, 52+ go on into the tray")
check(not any(a.text().startswith("✦ New idea") for a in w.findChildren(type(w.a_layer))),
      "no ✦ New idea (no whole songs from text)")

# --- collecting: a clip from the timeline into the tray
QTest.mouseClick(w.lanes, Qt.LeftButton, pos=QPoint(60, ROW_H // 2))
clip, track = w.state.selected_clip, w.project.tracks[0]
items = w._tray_items(track, clip)
check([i[0] for i in items if i][:1] == ["The whole clip"] and any(i and i[0].startswith("Chop it") for i in items),
      "right-click a clip → Add to the pad tray ▸ whole clip / chop it")
slot = w.add_to_tray(track, clip)
check(slot == 16 and w.project.pads[16].clip is not None and w.selected_pads == [16],
      "the whole clip lands on pad 17, the first tray pad")
check(len(grid.buttons) == KEYED + 8, "…and there's another empty row below it (the tray never runs out)")
silence()
QTest.keyPress(w.lanes, Qt.Key_Z)
pump()
check(w.player.voice_count == 0, "the keyboard still plays the 4×4 (Z is empty here), not the tray")
QTest.keyRelease(w.lanes, Qt.Key_Z)
b = grid.buttons[16]
silence()
QTest.mouseClick(b, Qt.LeftButton, Qt.NoModifier, QPoint(b.width() // 2, 8))
pump()
check(w.player.voice_count == 1, "clicking a tray pad plays it")
silence()
w.midi.note_on.emit(52, 0.9)
pump()
check(w.player.voice_count == 1, "MIDI note 52 plays pad 17")
w.midi.note_off.emit(52)
pump(400)

w.state.selection = (w.project.bar_seconds, 2 * w.project.bar_seconds)
before = len(w.project.tracks)
n_slots = w.add_to_tray(track, clip, bars=True)
check(n_slots == 17 and abs(w.project.pads[17].clip.length - w.project.bar_seconds) < 0.01, "…or just the selected bars")
w.state.selection = None
filled_before = sum(p.clip is not None for p in w.project.pads)
w.chop_to_pads("beat", track, clip, targets=w.tray_slots(16))
tray_used = [i for i, p in enumerate(w.project.pads) if p.clip is not None and i >= KEYED]
check(len(tray_used) == 18 and tray_used[:2] == [16, 17] and min(tray_used[2:]) == 18,
      f"Chop it: every beat fills the next 16 tray pads ({pad_key(tray_used[2])}–{pad_key(tray_used[-1])})")
check(len(grid.buttons) == KEYED + 4 * ((tray_used[-1] - KEYED) // 4 + 2), "the tray grew to fit, plus an empty row")
w.undo()
check(sum(p.clip is not None for p in w.project.pads) == filled_before and len(grid.buttons) == KEYED + 8,
      "Ctrl+Z takes the chops back out, and the tray shrinks again")

# --- keep every take: a new sound onto the 4×4 pad + the tray
def spread(d):
    seen.update(spread_default=d.spread.isChecked())
    d.text.setText("punchy kick")
    d.spread.setChecked(True)
    d.takes_buttons["4"].click()
    d.accept()
drive(creative.SoundDialog, spread)
w.design_pad_sound(0)
check(not seen.get("spread_default") and SENT[-1][1]["caption"].startswith("punchy kick"),
      "Generate: 'every take on its own pad' is there (off by default)")
deliver(w)
kicks = [i for i, p in enumerate(w.project.pads) if p.name.startswith("kick")]
check(kicks[0] == 0 and len(kicks) == 4 and all(i >= KEYED for i in kicks[1:]),
      f"…ticked: the first take goes on Z, the rest into the tray ({[pad_key(i) for i in kicks]})")

# --- variations of a pad: all of them into the tray, the original stays
original = w.project.pads[0].clip.offset
w.select_pad(0)
drive(creative.SoundDialog, lambda d: (seen.update(vary_spread=d.spread.isChecked()), d.accept()))
w.design_pad_sound(0, vary=True)
task, params, _ = SENT[-1]
deliver(w)
variations = [i for i, p in enumerate(w.project.pads) if i >= KEYED and p.clip is not None
              and p.clip.path == SENT[-1][2]]
check(task == "cover" and seen.get("vary_spread") and len(variations) == 4 and w.project.pads[0].clip.offset == original,
      f"✦ Variation: 4 variations into the tray ({[pad_key(i) for i in variations]}), Z untouched")

# --- the picker can keep them all too
w.select_pad(1)
drive(creative.SoundDialog, lambda d: (d.text.setText("crisp snare"), d.spread.setChecked(False), d.accept()))
w.design_pad_sound(1)
drive(creative.SoundPicker, lambda d: d.done(creative.SoundPicker.ALL))
deliver(w)
snares = [i for i, p in enumerate(w.project.pads) if p.clip is not None and p.clip.path == SENT[-1][2]]
check(len(snares) >= 2 and all(i >= KEYED for i in snares) and w.project.pads[1].clip is None,
      f"picker → All into the tray keeps every take ({[pad_key(i) for i in snares]})")
activate(w)

# --- the tray as a selection tray: pick some, make just a part
w.select_all_pads()
check(set(w.selected_pads) == {i for i, p in enumerate(w.project.pads) if p.clip is not None},
      "Select all = every pad with a sound, 4×4 and tray")
picked = [0, variations[0], snares[0]]
w.select_pads(picked)
w.player.seek(8 * w.project.bar_seconds)
n_tracks = len(w.project.tracks)


def part(d):
    seen["parts"] = [d.part_name.itemText(k) for k in range(d.part_name.count())]
    d.part.setChecked(True)
    d.part_name.setCurrentText("chorus")
    d.part_bars.setValue(4)
    d.accept()


drive(creative.SongDialog, part)
w.open_song_maker()
new = w.project.tracks[n_tracks:]
check(seen.get("parts") == ["intro", "verse", "chorus", "break", "outro"] and len(new) == 3,
      f"Make a song ▸ Just a part: a 4-bar chorus from the 3 picked pads ({[t.name for t in new]})")
check(all("(chorus)" in t.name and abs(t.clips[0].start - 8 * w.project.bar_seconds) < 1e-6
          and t.clips[0].length <= 5 * w.project.bar_seconds for t in new),
      "…at the playhead's bar, about 4 bars long, named as the chorus")
check(not any(t.muted for t in w.project.tracks[:n_tracks]), "a part goes in alongside what's there (nothing muted)")
check(new[0].gen["length"] == "part:chorus:4", "…and its tracks can still be re-rendered as that part")
if SHOTS:
    QApplication.processEvents()
    grid.show_pad(snares[-1])
    QApplication.processEvents()
    w.grab().save(f"{SHOTS}/tray.png")

# --- saved and reopened
w.project.save()
again = Project.load(song)
last = max(i for i, p in enumerate(w.project.pads) if p.clip is not None)
check(len(again.pads) == last + 1 and again.pads[last].clip is not None, f"the tray is saved, up to pad {pad_key(last)}")
w.close()
cleanup(song)
finish()

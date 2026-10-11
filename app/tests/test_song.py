"""End-to-end: select some pads and make a song from exactly those sounds (offscreen, silent)."""

import os
import time

os.environ["LAYERDAW_NO_AUDIO"] = "1"

import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402
from PySide6.QtCore import QPoint, Qt, QTimer  # noqa: E402
from PySide6.QtTest import QTest  # noqa: E402
from PySide6.QtWidgets import QApplication, QDialogButtonBox, QPushButton  # noqa: E402

from harness import activate, check, cleanup, demo_song, fake_engine, finish, open_window  # noqa: E402
from layerdaw import arrange, theory  # noqa: E402
from layerdaw.chords import detect_pitch  # noqa: E402
from layerdaw.model import SR, Pad, load_audio  # noqa: E402
from layerdaw.ui import main_window as mw  # noqa: E402

SHOTS = os.environ.get("LAYERDAW_SHOTS")  # a folder: save screenshots there to look at

# --- the arranger on its own
pads = {0: Pad(name="kick", kind="kick"), 1: Pad(name="snare", kind="snare"), 2: Pad(name="hats", kind="hi-hat"),
        3: Pad(kind="808", play="note"), 6: Pad(name="riser", kind="fx"), 12: Pad(name="vox", kind="vocals"),
        **{8 + k: Pad(kind="keys", play="chord", degree=d) for k, d in enumerate([0, 5, 3, 4])}}
for p in pads.values():
    p.clip = None
plan = arrange.make_plan(pads, bpm=140, style="trap", length="full")
check(plan.bars == 72 and [s[0] for s in plan.sections] == ["intro", "verse", "chorus", "break", "chorus", "outro"],
      f"a full song is 72 bars: {[(s[0], s[2]) for s in plan.sections]}")
short = arrange.make_plan(pads, bpm=140, style="trap", length="short")
check(short.bars == 32 and "outro" not in [s[0] for s in short.sections], "a short one is 32 bars, no outro")
check(plan.roles == {0: "kick", 1: "snare", 2: "hat", 3: "bass", 6: "fx", 12: "hook",
                     8: "chords", 9: "chords", 10: "chords", 11: "chords"}, f"each pad gets a part {plan.roles}")
check(plan.progression == [0, 5, 3, 4], "the chord pads, in order, are the progression")
check(all(0 <= e.beat < plan.bars * 4 for evs in plan.events.values() for e in evs), "every hit lands inside the song")
intro_end = plan.sections[1][1] * 4
check(not [e for e in plan.events[0] if e.beat < intro_end], "no kick in the intro")
check(min(e.beat for e in plan.events[2]) >= intro_end / 2, "hats come in halfway through the intro")
check(all(bool([e for e in plan.events[0] if s * 4 <= e.beat < (s + n) * 4]) == (name in ("verse", "chorus"))
          for name, s, n in plan.sections if name != "outro"), "the kick plays in verses and choruses only")
bars_per_chord = 2 if 240 / 140 < 2.0 else 1
check(all(e.degree == pads[i].degree for i in range(8, 12) for e in plan.events[i]),
      "each chord pad only ever plays its own chord")
underneath = {int(e.beat // (4 * bars_per_chord)): e.degree for i in range(8, 12) for e in plan.events[i]}
check(all(underneath.get(int(e.beat // (4 * bars_per_chord)), e.degree) == e.degree for e in plan.events[3]),
      "the 808 follows the chord underneath it")
chorus_starts = [s for name, s, _ in plan.sections if name == "chorus"]
check(sorted(int(e.beat // 4) for e in plan.events[6]) == [s - 1 for s in chorus_starts], "a riser leads into every chorus")
check(plan.events[12] and all(any(s <= e.beat / 4 < s + n for name, s, n in plan.sections if name in ("chorus", "break"))
                              for e in plan.events[12]), "hooks only play in choruses and the break (with chords there)")
# the user's real pick (3 kicks, 2 hats, a snare, a synth stab, 2 hooks; no chords/perc/fx): a lofi song
# with that used to go silent for the whole 8-bar break
theirs = {1: Pad(name="k1", kind="kick"), 5: Pad(name="k2", kind="kick"), 7: Pad(name="k3", kind="kick"),
          2: Pad(name="h1", kind="hi-hat"), 6: Pad(name="h2", kind="hi-hat"), 3: Pad(name="s", kind="snare"),
          11: Pad(name="lead F5", kind="synth"), 14: Pad(name="w3", kind="hook"), 15: Pad(name="w4", kind="hook")}
lofi = arrange.make_plan(theirs, bpm=176, style="lofi", length="full")
busy = [len({i for i, evs in lofi.events.items() for e in evs if b * 4 <= e.beat < b * 4 + 4}) for b in range(lofi.bars)]
silent = [b + 1 for b, k in enumerate(busy) if k == 0]
check(not silent, f"no silent bars in the whole song (silent: {silent})")
brk = next(s for s in lofi.sections if s[0] == "break")
in_break = {lofi.roles[i] for i, evs in lofi.events.items() for e in evs if brk[1] * 4 <= e.beat < (brk[1] + brk[2]) * 4}
check(len(in_break) >= 3 and "kick" not in in_break, f"the break keeps 3+ parts going, still without the kick {sorted(in_break)}")
roll = [e for e in lofi.events[3] if (brk[1] + brk[2] - 1) * 4 <= e.beat < (brk[1] + brk[2]) * 4]
check(len(roll) == 8 and roll[-1].velocity > roll[0].velocity, "a snare roll builds out of the break into the chorus")
sets = [theirs, {0: Pad(kind="kick"), 1: Pad(kind="snare")}, {0: Pad(kind="kick")}, {0: Pad(kind="kick"), 12: Pad(kind="hook")},
        {12: Pad(kind="hook")}, {3: Pad(kind="808", play="note"), 4: Pad(kind="lead", play="note")}, pads]
gaps = []
for k, chosen in enumerate(sets):
    for style in arrange.STYLES:
        for length in ("short", "full"):
            p = arrange.make_plan(chosen, bpm=176, style=style, length=length)
            gaps += [(k, style, length, b + 1) for b in range(p.bars) if not any(
                e.beat < b * 4 + 4 and e.beat + e.length > b * 4 for evs in p.events.values() for e in evs)]
check(not gaps, f"whatever pads are picked, no style/length leaves a silent bar {gaps[:5]}")
again = arrange.make_plan(pads, bpm=140, style="trap", length="full")
check([(e.beat, e.velocity) for e in again.events[2]] == [(e.beat, e.velocity) for e in plan.events[2]],
      "the same pads give the same song")
check([arrange.auto_style(b) for b in (85, 100, 124, 140, 176)] == ["lofi", "pop", "house", "trap", "dnb"],
      "Auto picks a style from the tempo")
check(arrange.role_of(Pad(name="Kick 2", kind="drums")) == "kick"
      and arrange.role_of(Pad(kind="drums"), classify=lambda p: "snare") == "snare"
      and arrange.role_of(Pad(kind="bass", play="note")) == "bass"
      and arrange.role_of(Pad(kind="keys", play="note")) == "lead", "roles come from how a pad plays, its category, its name, or its sound")
nochords = arrange.make_plan({0: Pad(kind="kick"), 3: Pad(kind="808", play="note")}, bpm=100, style="pop")
check(nochords.progression == [0, 5, 3, 4], "with no chord pads the bass follows i – VI – iv – v")

# --- in the app: real sounds on pads, select them, make the song
fake_engine()
mw.MainWindow._show_menu = lambda self, items, pos: None
song = demo_song()  # A minor, 90 BPM
app, w = open_window(song, (1400, 860))
w.pads_dock.show()
grid, ins = w.pad_panel.grid, w.pad_panel.inspector
beat_s, bar_s = w.project.beat_seconds, w.project.bar_seconds


def write(name, audio):
    path = song.parent / f"{name}.wav"
    sf.write(path, np.stack([audio, audio], 1).astype(np.float32), SR)
    return str(path)


rng = np.random.default_rng(3)
t = np.arange(int(0.4 * SR)) / SR
kick = np.sin(2 * np.pi * (45 + 100 * np.exp(-t / 0.03)) * t) * np.exp(-t / 0.09)
snare = (rng.standard_normal(len(t)) * 0.5 + np.sin(2 * np.pi * 190 * t)) * np.exp(-t / 0.06)
hat = np.diff(rng.standard_normal(len(t) + 1)) * np.exp(-t / 0.015) * 0.5
tv = np.arange(2 * SR) / SR
vox = 0.4 * np.sin(2 * np.pi * 440 * tv + 3 * np.sin(2 * np.pi * 5 * tv)) * np.minimum(1, tv * 20)
for i, (name, audio) in enumerate((("Kick", kick), ("Snare", snare), ("HiHat", hat))):
    w.pad_file_dropped(i, write(name, audio))
w.pad_file_dropped(12, write("vox chop", vox))
w.select_pad(3)
w.set_pad_play("note")
w.set_pad_category("808")
for k, degree in enumerate([0, 5, 3, 4]):
    w.select_pad(8 + k)
    w.set_pad_play("chord")
    w.set_pad_degree(degree)
QTest.qWait(500)
check([w.project.pads[i].kind for i in (0, 1, 2, 3, 12)] == ["kick", "snare", "hi-hat", "808", "vocals"],
      "the dropped sounds got their categories from their names")

# fewer than two sounds: nothing to arrange
w.select_pads([0, 5])
w.open_song_maker()
check("Select a few pads" in w.engine_label.text() and QApplication.activeModalWidget() is None,
      "with only one sound selected, Make a song says what to do")

# Ctrl+click the pads (pad 5 is empty: it should just be skipped)
w.select_pad(0)
for i in (1, 2, 3, 5, 8, 9, 10, 11, 12):
    b = grid.buttons[i]
    QTest.mouseClick(b, Qt.LeftButton, Qt.ControlModifier, QPoint(b.width() // 2, 8))
check(sorted(grid.selected) == [0, 1, 2, 3, 5, 8, 9, 10, 11, 12] and ins.stack.currentIndex() == 1,
      "Ctrl+click builds up a selection; the inspector shows the multi-pad actions")
if SHOTS:
    w.pads_dock.grab().save(f"{SHOTS}/song-multi.png")
w.player.seek(2.5 * bar_s)  # the song goes in at the playhead's bar
n_tracks = len(w.project.tracks)
seen = {}


def drive_dialog():
    dlg = QApplication.activeModalWidget()
    if dlg is None or not hasattr(dlg, "part_boxes"):
        QTimer.singleShot(100, drive_dialog)
        return
    seen["parts"] = " | ".join(f"{k} {dlg.part_boxes[k].currentText()} / {dlg.note_boxes[k].currentText()}"
                               for k in sorted(dlg.part_boxes))
    seen["chords_note"] = dlg.no_chords.isVisible()
    hook = dlg.note_boxes[12]
    hook.setCurrentIndex(hook.findData("full"))  # the vox chop: play the whole phrase each time
    seen["styles"] = [dlg.style.itemData(k) for k in range(dlg.style.count())]
    seen["auto"] = dlg.style.itemText(0)
    dlg.style.setCurrentIndex(dlg.style.findData("trap"))
    dlg.short.setChecked(True)
    if SHOTS:
        dlg.grab().save(f"{SHOTS}/song-dialog.png")
    seen["t0"] = time.monotonic()
    dlg.findChild(QDialogButtonBox).button(QDialogButtonBox.Ok).click()


QTimer.singleShot(150, drive_dialog)
next(b for b in ins.findChildren(QPushButton) if b.text().startswith("✦ Make a song")).click()
took = time.monotonic() - seen.get("t0", time.monotonic())
parts = seen.get("parts", "")
check(all(s in parts for s in ("kick pattern", "snare pattern", "hi-hat pattern", "bass line", "chord progression",
                               "hooks in the choruses")) and not seen.get("chords_note"),
      f"the dialog shows what each pad will play ({parts[:90]}…)")
check("0 kick pattern / As the pad is set (one-shot)" in parts and "12 hooks in the choruses / As the pad is set (hold)" in parts,
      "…and how its notes will sound (the pad's own mode, until you pick another)")
check(seen.get("styles", [])[:1] == ["auto"] and "lofi" in seen.get("auto", "") and "dnb" in seen.get("styles", []),
      f"Auto picks from the tempo ({seen.get('auto')})")
new = w.project.tracks[n_tracks:]
names = [tr.name for tr in new]
check(len(new) == 9, f"one track per pad with a sound, the empty one skipped ({names})")
check(took < 15, f"rendering a 32-bar song is quick ({took:.1f}s)")
by_role = {arrange.role_of(w.project.pads[i]): tr for i, tr in zip([0, 1, 2, 3, 8, 9, 10, 11, 12], new)}
check(names[:4] == ["Kick", "Snare", "HiHat", "synth · bass"] and names[-1] == "vox chop · hook"
      and sum(" · chords" in n for n in names) == 4, "tracks are named by pad (and part, unless the name says it)")
check(all(abs(tr.clips[0].start - 2 * bar_s) < 1e-6 for tr in new), "every part starts at the playhead's bar (bar 3)")
check(all(tr.clips[0].length <= 33 * bar_s for tr in new), "nothing runs past the 32 bars (plus a tail)")
check([tr.kind for tr in new[:4]] == ["drums", "drums", "drums", "bass"] and new[-1].kind == "vocals",
      "tracks are coloured by their pad's sound family")

kick_audio = load_audio(by_role["kick"].clips[0].path).mean(axis=1)
intro, verse = kick_audio[: int(4 * bar_s * SR)], kick_audio[int(4 * bar_s * SR): int(12 * bar_s * SR)]
check(np.abs(intro).max() < 1e-3 < np.abs(verse).max(), "the kick sits out the intro and comes in on the verse")
bass = load_audio(by_role["bass"].clips[0].path).mean(axis=1)
root = theory.root_midi(theory.parse_key("A minor")[0]) - 24
progression = [0, 5, 3, 4]
pitches = []
for bar in range(4, 8):  # verse: one chord a bar at 90 BPM
    a = int(bar * bar_s * SR)
    got = detect_pitch(bass[a + int(0.05 * SR): a + int(beat_s * SR)])
    want = theory.degree_midi(theory.parse_key("A minor")[0], "minor", progression[bar % 4]) - 24
    pitches.append((None if got is None else round(got), want))
check(all(g is not None and (g - want) % 12 == 0 for g, want in pitches), f"the 808 plays the chord roots {pitches}")
chord_tracks = [tr for tr in new if " · chords" in tr.name]
check(all(np.abs(load_audio(tr.clips[0].path)).max() > 0.01 for tr in chord_tracks), "every chord pad is heard")
check(all(w.project.tracks[k].muted for k in range(n_tracks)) and "muted 2 tracks" in w.engine_label.text(),
      f"the tracks it overlaps are muted, and it says so ({w.engine_label.text()})")
together = w.project.render(new)
check(0.5 < np.abs(together).max() <= 0.9, f"the new parts are levelled together, no clipping (peak {np.abs(together).max():.2f})")
QApplication.processEvents()
check(len(w.headers.headers) == len(w.project.tracks) and all(h.isVisible() for h in w.headers.headers.values()),
      "every new track has its header (mute/solo/volume)")
if SHOTS:
    w.lanes_scroll.verticalScrollBar().setValue(10 ** 6)
    QApplication.processEvents()
    w.grab().save(f"{SHOTS}/song-timeline.png")

activate(w)  # focus comes back to the window after the dialog
QTest.keyClick(w.lanes, Qt.Key_Z, Qt.ControlModifier)
check(len(w.project.tracks) == n_tracks and not any(t.muted for t in w.project.tracks),
      "one Ctrl+Z takes the whole song back out (and unmutes what was there)")
QTest.keyClick(w.lanes, Qt.Key_Z, Qt.ControlModifier | Qt.ShiftModifier)
check(len(w.project.tracks) == n_tracks + 9, "…and Ctrl+Shift+Z puts it back")

# --- changing one part afterwards (right-click a song track → Song part)
from PySide6.QtWidgets import QMenu  # noqa: E402


def song_part(track, *path):
    """Click through the track's real right-click menu: Song part ▸ path…"""
    menu = w._build_menu(w._track_items(track, None))
    for label in ("Song part", *path[:-1]):
        menu = next(a.menu() for a in menu.actions() if a.menu() is not None and a.text().strip(" ●") == label)
    next(a for a in menu.actions() if a.text().strip(" ●").startswith(path[-1])).trigger()
    QApplication.processEvents()


def runs(track):
    """Lengths (seconds) of the stretches where the track is sounding."""
    audio = np.abs(load_audio(track.clips[0].path)).max(axis=1)
    block = int(0.01 * SR)
    on = audio[: len(audio) // block * block].reshape(-1, block).max(axis=1) > 1e-3  # 10 ms envelope
    on = np.repeat(on, block)
    edges = np.flatnonzero(np.diff(np.r_[0, on.astype(int), 0]))
    return (edges[1::2] - edges[::2]) / SR


new = w.project.tracks[n_tracks:]
kick_t, hat_t, snare_t, hook_t = new[0], new[2], new[1], new[-1]
check(hook_t.gen.get("notes") == {"12": "full"} and hook_t.gen["role"] == "hook", "the dialog's Notes choice is used")
check(max(runs(hook_t)) > 1.9, f"Full sample: every hook hit plays the whole 2 s phrase ({max(runs(hook_t)):.2f}s)")
check(kick_t.gen["mode"] == "song" and kick_t.gen["pads"] == [0, 1, 2, 3, 8, 9, 10, 11, 12],
      "each song track remembers how it was made")
before, start = kick_t.clips[0].path, kick_t.clips[0].start
song_part(kick_t, "Notes", "Short notes")
check(kick_t.clips[0].path != before and kick_t.clips[0].start == start and kick_t.gen["notes"].get("0") == "short",
      "Song part ▸ Notes ▸ Short re-renders just that track, in place")
check(max(runs(kick_t)) <= 0.5 * beat_s + 0.02, f"…every kick is now at most an 8th note ({max(runs(kick_t)):.2f}s)")
check("Short notes" in w.engine_label.text().capitalize() or "short notes" in w.engine_label.text(),
      f"…and the status bar says what changed ({w.engine_label.text()})")
QTest.keyClick(w.lanes, Qt.Key_Z, Qt.ControlModifier)
check(kick_t.clips[0].path == before and "0" not in kick_t.gen.get("notes", {}), "Ctrl+Z puts the kick back as it was")

song_part(hook_t, "Notes", "Hold to the next hit")
chorus = next(st for name, st, _ in arrange.make_plan(w._song_pads(hook_t.gen["pads"]), bpm=90, style="trap",
                                                      length="short").sections if name == "chorus")
check(max(runs(hook_t)) >= 7.5 * bar_s, f"Hold: the hook sustains from hit to hit, no gaps ({max(runs(hook_t)) / bar_s:.1f} bars)")

song_part(hat_t, "Plays", "percussion")
check(hat_t.name == "HiHat · perc" and hat_t.gen["role"] == "perc", f"Plays ▸ percussion turns the hats into a perc part ({hat_t.name})")

snare_before = load_audio(snare_t.clips[0].path)
w.regenerate(snare_t)  # the ⟳ button
snare_after = load_audio(snare_t.clips[0].path)
check(snare_t.gen["variations"] == {"1": 1} and (len(snare_after) != len(snare_before)
                                                 or np.abs(snare_after - snare_before).max() > 0.05),
      "⟳ on a song track gives that part a new variation")

w.project.pads[0].pitch = 12
song_part(kick_t, "Re-render from pad Z")
check(max(runs(kick_t)) < 0.3, "Re-render from pad picks up edits to the pad (an octave up = shorter kicks)")
w.project.pads[0].pitch = 0

w.project.bpm = 120
check(not w.render_song_part(kick_t, vary=True) and "made at 90 BPM" in w.engine_label.text(),
      "after a tempo change it won't redo a part out of time with the rest")
w.project.bpm = 90
w.project.save()
from layerdaw.model import Project  # noqa: E402
reloaded = next(t for t in Project.load(song).tracks if t.name == kick_t.name)
check(reloaded.gen["mode"] == "song" and reloaded.gen["pad"] == 0, "song parts stay editable after reopening the song")
if SHOTS:
    w._build_menu(w._track_items(hook_t, None)).grab().save(f"{SHOTS}/song-part-menu.png")

# drums only, full length, auto style
w.player.seek(0.0)
t0 = time.monotonic()
drums = w.make_song([0, 1, 2], "auto", "full")
check(len(drums) == 3 and all(tr.clips[0].start == 0 for tr in drums), "a drums-only song works too")
check(time.monotonic() - t0 < 15, f"…72 bars in {time.monotonic() - t0:.1f}s")
check(w.make_song([5, 6]) == [] and "no sound" in w.engine_label.text(), "empty pads: nothing to arrange, and it says so")

w.close()
cleanup(song)
finish()

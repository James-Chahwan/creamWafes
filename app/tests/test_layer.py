"""End-to-end: the Layer panel — shape a track's sound like a person would (offscreen, silent)."""

import os
import time

os.environ["LAYERDAW_NO_AUDIO"] = "1"

import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402
from PySide6.QtCore import QPoint, Qt  # noqa: E402
from PySide6.QtTest import QTest  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from harness import SENT, check, cleanup, deliver, demo_song, fake_engine, finish, open_window  # noqa: E402
from layerdaw import fx  # noqa: E402
from layerdaw.model import SR, Clip, Project, Track  # noqa: E402

SHOTS = os.environ.get("LAYERDAW_SHOTS")
fake_engine()
song = demo_song()  # drums + bass, A minor, 90 BPM
app, w = open_window(song, (1500, 900))
panel = w.layer_panel
drums, bass = w.project.tracks


def settle(timeout=20.0):
    """Wait for the background fx work (the status bar says "Shaping …" meanwhile)."""
    end = time.monotonic() + timeout
    QTest.qWait(200)
    while (w._fx_pending() or w._fx_timer.isActive()) and time.monotonic() < end:
        QTest.qWait(50)
    QApplication.processEvents()


def header(track):
    QApplication.processEvents()
    return w.headers.headers[id(track)]


def level(audio, a, b):
    return float(np.abs(audio[int(a * SR):int(b * SR)]).max())


dry = w.project.render().copy()
check(w.layer_dock.isVisible() and not panel.controls.isVisible() and panel.empty.isVisible(),
      "the Layer panel is there, asking you to pick a track")
QTest.mouseClick(header(bass), Qt.LeftButton, pos=QPoint(120, 50))
check(w.state.selected_track == 1 and panel.controls.isVisible() and "bass" in panel.title.text(),
      "clicking a track's name opens it in the Layer panel")
check(panel.summary.text() == "as recorded" and panel.values["low_cut"].text() == "off", "a fresh track is as recorded")
header(drums).fx_button.click()
check(w.state.selected_track == 0 and "drums" in panel.title.text(), "the fx button on a track opens that track")
QTest.mouseClick(header(bass), Qt.LeftButton, pos=QPoint(120, 50))

# --- warmth + reverb, worked out in the background
before = w.history.next_undo()
for v in (10, 20, 35, 40):  # a drag: several values, one undo step
    panel.sliders["warmth"].setValue(v)
panel.sliders["reverb"].setValue(50)
check(bass.fx == {"warmth": 0.4, "reverb": 0.5} and panel.values["warmth"].text() == "40%",
      f"sliders set the track's sound ({bass.fx})")
check(w.engine_label.text().startswith("Shaping") or w._fx_timer.isActive(), "it works on it in the background")
settle()
wet = w.project.render(ready_only=True)
check(np.abs(wet - dry[:len(wet)]).max() > 0.01 and w.engine_label.text() == "Ready", "…and the mix picks it up")
check(header(bass).fx_button.styleSheet() != "" and "reverb 50%" in header(bass).fx_button.toolTip(),
      "the track's fx button lights up and says what's on")
check(panel.summary.text() == "warm 40% · reverb 50%", f"the panel sums it up ({panel.summary.text()})")
bass_end = max(c.end for c in bass.clips)
_, tail = fx.output(w.project, bass)
check(len(tail) / SR > bass_end + 1.0, f"the reverb rings on after the clip ({len(tail) / SR - bass_end:.1f}s tail)")

panel.bypass.setChecked(True)
settle()
check(np.abs(w.project.render(ready_only=True) - dry).max() < 1e-6, "A/B bypass = exactly the dry track")
panel.bypass.setChecked(False)
settle()

# --- undo keeps you on the track, with the old values
w.undo()  # bypass off
w.undo()  # bypass on
w.undo()  # reverb
check(bass.fx == {"warmth": 0.4} and w.state.selected_track == 1 and panel.values["reverb"].text() == "off",
      f"Ctrl+Z undoes one change and stays on the track ({bass.fx})")
w.undo()  # the whole warmth drag
check(bass.fx == {} and w.history.next_undo() == before, "a slider drag is one undo step")
settle()
QApplication.processEvents()
check(header(bass).fx_button.styleSheet() == "", "the fx button goes dark again")
w.redo()
check(bass.fx == {"warmth": 0.4}, "Ctrl+Shift+Z brings it back")

# --- every control lands in the right units
panel.sliders["low_cut"].setValue(50)
panel.sliders["high_cut"].setValue(50)
panel.sliders["tone"].setValue(-40)
panel.sliders["width"].setValue(0)
check(abs(bass.fx["low_cut"] - 141) <= 2 and abs(bass.fx["high_cut"] - 3162) <= 5 and bass.fx["tone"] == -0.4
      and bass.fx["width"] == 0.0, f"cuts in Hz, tone and width in % ({bass.fx})")
check(panel.values["width"].text() == "mono" and panel.values["high_cut"].text() == "3.2 kHz", "values read like a person would say them")
panel.sliders["low_cut"].mouseDoubleClickEvent(None)
check("low_cut" not in bass.fx, "double-click a slider resets it")

# --- presets
panel.duck_to.setCurrentIndex(panel.duck_to.findData("drums"))
panel.duck_to.activated.emit(panel.duck_to.currentIndex())
w.apply_layer_preset("Lo-fi radio")
check(bass.fx == {**fx.PRESETS["Lo-fi radio"], "duck_to": "drums"} and panel.values["high_cut"].text() == "3.8 kHz",
      "a preset sets several controls at once (and keeps who it ducks under)")
w.apply_layer_preset("Clean (reset)")
check(bass.fx == {"duck_to": "drums"} and not fx.is_active(bass.fx), "Reset puts it back as recorded")

# --- duck under the drums: the bass dips on each hit
check(panel.duck_to.itemText(0) == "Auto (drums)" and panel.duck_to.findData("drums") > 0,
      f"Duck lists the other tracks; Auto would pick the drums ({panel.duck_to.itemText(0)})")
panel.sliders["duck"].setValue(80)
settle()
_, ducked = fx.output(w.project, bass)
start, raw = fx.raw(bass)
beat = w.project.beat_seconds


def gain_at(b):  # how much of the dry bass is left, 20 ms from beat b
    a = int(b * beat * SR)
    return np.abs(ducked[a:a + 960]).max() / max(np.abs(raw[a:a + 960]).max(), 1e-6)


on_hit, between = gain_at(2.03), gain_at(2.8)
check(on_hit < 0.6 and between > 0.9, f"Duck: the bass dips right on the kick and recovers before the next ({on_hit:.2f} → {between:.2f})")
panel.sliders["duck"].setValue(0)

# --- pitch keeps the speed
if fx.pitch_available():
    panel.sliders["pitch"].setValue(12)
    settle(60)
    _, up = fx.output(w.project, bass)
    seg = up[int(1 * SR): int(2 * SR), 0]
    peak = np.fft.rfftfreq(len(seg), 1 / SR)[np.argmax(np.abs(np.fft.rfft(seg))[5:]) + 5]
    check(abs(peak - 110) < 4 and abs(len(up) - len(raw)) < SR // 10,
          f"Pitch +12: the 55 Hz bass plays at 110 Hz, same length ({peak:.1f} Hz)")
    panel.sliders["pitch"].setValue(0)
else:
    check(not panel.sliders["pitch"].isEnabled() and panel.pitch_note.isVisible(), "no Rubber Band: pitch says how to get it")

# --- echo
panel.sliders["delay"].setValue(60)
panel.delay_time.setCurrentIndex(panel.delay_time.findData("1/4"))
panel.delay_time.activated.emit(panel.delay_time.currentIndex())
check(bass.fx["delay"] == 0.6 and bass.fx["delay_time"] == "1/4", "Echo amount and time")
settle()
if SHOTS:
    w.grab().save(f"{SHOTS}/layer-panel.png")

# --- it's what you export, what you save, and what the AI hears
out = song.parent / "bass.wav"
w.project.export_track(bass, out)
exported, _ = sf.read(out, dtype="float32")
plain = np.clip(w.project.render([bass], w.project.length, ready_only=False), -1, 1)
check(np.abs(exported - plain).max() < 1e-3 and np.abs(exported - np.clip(dry, -1, 1)).max() > 0.01,
      "Export track includes its sound settings")
w.project.save()
again = Project.load(song)
check(again.tracks[1].fx == bass.fx, "sound settings are saved with the song")
check(np.abs(again.render([again.tracks[1]], again.length) - plain).max() < 1e-3,
      "…and a reopened song sounds the same")

# --- generate part of a layer: pick bars (ruler or boxes), find a gap, fill it
bar = w.project.bar_seconds
src_clip = bass.clips[0]
gappy = Track(name="keys", kind="keyboard", clips=[Clip(src_clip.path, 0.0, 0.0, 2 * bar),
                                                   Clip(src_clip.path, 5 * bar, 5 * bar, 3 * bar)])
w._checkpoint("test: add a layer with a gap")
w.project.tracks.append(gappy)
w.refresh()
QTest.mouseClick(header(gappy), Qt.LeftButton, pos=QPoint(120, 50))
check(panel.gen_btn.isEnabled() and panel.gen_btn.text().startswith("✦ Generate bars"),
      f"the Layer panel has a Generate section ({panel.gen_btn.text()})")
w.state.selection = (3 * bar, 6 * bar)  # what dragging bars 4–6 on the ruler does
w._selection_changed()
check((panel.gen_from.value(), panel.gen_to.value()) == (4, 6) and panel.gen_btn.text() == "✦ Generate bars 4–6",
      "dragging on the ruler fills in the bars")
panel.gen_to.setValue(7)
check(w.state.selection == (3 * bar, 7 * bar), "…and typing bars moves the ruler selection")
panel.gen_from.setValue(9)
check(panel.gen_to.value() == 9, "the range can't go backwards")
panel.find_gap_btn.click()
check((panel.gen_from.value(), panel.gen_to.value()) == (3, 5) and "Bars 3–5 of keys are empty" in w.engine_label.text(),
      f"Find a gap picks the silent bars between its sounds ({w.engine_label.text()})")
gappy.fx = {"warmth": 0.5, "reverb": 0.3}
gappy.volume, gappy.pan = 0.7, -0.4
sent = len(SENT)
panel.gen_prompt.setText("soft chords filling the gap")
panel.gen_btn.click()
task, params, _ = SENT[-1]
check(len(SENT) == sent + 1 and task == "repaint" and abs(params["repainting_start"] - 2 * bar) < 1e-6
      and abs(params["repainting_end"] - 5 * bar) < 1e-6 and params["caption"] == "soft chords filling the gap",
      f"✦ Generate sends just those bars, with what you asked for ({task} {params['repainting_start']:.2f}–{params['repainting_end']:.2f}s)")
src_audio, _ = sf.read(params["src_audio"], dtype="float32", always_2d=True)
dry_keys = w.project.render([gappy], len(src_audio) / SR, dry=True)
check(np.abs(src_audio - dry_keys).max() < 1e-4, "the AI hears the layer dry (its reverb isn't baked into the new take)")
check(w.state.pending[-1].get("track") is gappy and w.lanes.rows() == len(w.project.tracks),
      "while it generates, the progress sits on that track over those bars (no extra row)")
n_tracks = len(w.project.tracks)
if SHOTS:
    w.state.pending[-1]["progress"] = 0.6
    w.lanes.update()
    QApplication.processEvents()
    w.grab().save(f"{SHOTS}/layer-generating.png")
deliver(w)
fresh = w.state.selected_clip
if SHOTS:
    QApplication.processEvents()
    w.grab().save(f"{SHOTS}/layer-spliced.png")
check(len(w.project.tracks) == n_tracks and not gappy.muted and fresh in gappy.clips,
      "the new bars go right into the same track (no new take)")
check(abs(fresh.start - (2 * bar - 0.02)) < 1e-3 and abs(fresh.end - (5 * bar + 0.02)) < 1e-3
      and gappy.fx == {"warmth": 0.5, "reverb": 0.3} and (gappy.volume, gappy.pan) == (0.7, -0.4),
      f"…exactly in bars 3–5 ({fresh.start:.2f}–{fresh.end:.2f}s), keeping the track's sound settings")
others = [c for c in gappy.clips if c is not fresh]
check(all(c.end <= fresh.start + 1e-6 or c.start >= fresh.end - 1e-6 for c in others),
      "…and nothing else overlaps it")
mix = w.project.render([gappy], dry=True)
a = int(round(fresh.start * SR))
check(np.abs(np.diff(mix[a - 50:a + 50, 0])).max() < 0.05, "the join is crossfaded (no click)")
check(w.layer_track() is gappy and "Bars 3–5 of keys done" in w.engine_label.text(),
      f"the panel stays on the track ({w.engine_label.text()})")
w.undo()
check(len(gappy.clips) == 2 and fresh not in gappy.clips, "Ctrl+Z puts the old bars back")
panel.gen_prompt.clear()
panel.gen_btn.click()
check(SENT[-1][1]["caption"] == "keyboard, same sound and style as the rest of this part",
      f"with nothing typed it asks for more of the same ({SENT[-1][1]['caption']})")
w.jobs.clear()
w.state.pending.clear()
w.engine.is_ready = False
w.refresh_layer()
check(not panel.gen_btn.isEnabled(), "Generate waits for the engine to load")
w.engine.is_ready = True
w.state.selection = None
w.refresh()

# --- a setting that blows up doesn't take the app down or loop forever
real = fx.process
fx.process = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom"))
jobs = w._fx_job
panel.sliders["warmth"].setValue(77)
settle()
QTest.qWait(600)
check("Couldn't apply" in w.engine_label.text() and w._fx_job - jobs <= 2,
      f"a failing effect is reported once, not retried forever ({w.engine_label.text()})")
check(np.abs(w.project.render(ready_only=True)).max() > 0, "…and the song still plays (that track dry)")
fx.process = real

w.close()
cleanup(song)
finish()

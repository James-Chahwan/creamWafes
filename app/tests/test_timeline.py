"""End-to-end: use the timeline like a person (offscreen, fake engine, silent audio)."""

import json
import os

os.environ["LAYERDAW_NO_AUDIO"] = "1"

from PySide6.QtCore import QPoint, Qt, QTimer  # noqa: E402
from PySide6.QtTest import QTest  # noqa: E402
from PySide6.QtWidgets import QApplication, QDialog  # noqa: E402

from harness import SENT, activate, check, cleanup, deliver, demo_song, fake_engine, finish, open_window  # noqa: E402
from layerdaw.model import Project  # noqa: E402
from layerdaw.ui import main_window as mw  # noqa: E402
from layerdaw.ui.timeline import ROW_H  # noqa: E402

fake_engine()
menus = []
mw.MainWindow._show_menu = lambda self, items, pos: menus.append(self._build_menu(items))


def labels(menu, depth=0):
    out = []
    for a in menu.actions():
        if not a.isSeparator():
            out.append(a.text())
            if a.menu():
                out += labels(a.menu(), depth + 1)
    return out


def answer_dialog(**vals):
    def go():
        d = next(x for x in QApplication.topLevelWidgets() if isinstance(x, QDialog) and x.isVisible())
        for k, v in vals.items():
            w = getattr(d, k)
            w.setText(v) if k == "prompt" else w.setCurrentText(v) if k == "kind" else w.setValue(v)
        d.accept()
    QTimer.singleShot(150, go)


song = demo_song()
proj = Project.load(song)
proj.tracks = []          # start from an empty song
proj.save()
app, w = open_window(song)

QTest.keyClick(w.lanes, Qt.Key_Space)
check(w.player.playing, "Space plays on a fresh window")
w.toggle_play()

answer_dialog(kind="drums", prompt="boom bap", bars=8)
w.add_layer()
check(len(w.state.pending) == 1, "a ghost row shows while generating")
deliver(w); activate(w)
check(len(w.project.tracks) == 1 and SENT[-1][0] == "text2music", "first layer on an empty song = solo text2music")

hdr = w.headers.headers[id(w.project.tracks[0])]
QTest.mouseDClick(hdr, Qt.LeftButton)
hdr.editor.setText("Drums v1"); hdr.editor.editingFinished.emit(); activate(w)
check(w.project.tracks[0].name == "Drums v1", "double-click a track name to rename it")

bar_px = w.state.x(w.project.bar_seconds)
y = ROW_H // 2
QTest.mousePress(w.lanes, Qt.LeftButton, pos=QPoint(30, y))
QTest.mouseMove(w.lanes, QPoint(int(30 + 2 * bar_px), y))
QTest.mouseRelease(w.lanes, Qt.LeftButton, pos=QPoint(int(30 + 2 * bar_px), y))
check(abs(w.project.tracks[0].clips[0].start - 2 * w.project.bar_seconds) < 0.01, "drag a clip to move it 2 bars")
QTest.keyClick(w.lanes, Qt.Key_Z, Qt.ControlModifier)
check(w.project.tracks[0].clips[0].start == 0.0, "Ctrl+Z undoes the move")
QTest.keyClick(w.lanes, Qt.Key_Z, Qt.ControlModifier | Qt.ShiftModifier)
check(w.project.tracks[0].clips[0].start > 0, "Ctrl+Shift+Z redoes it")

w.player.seek(w.project.tracks[0].clips[0].start + 4.0)
w.state.selected_clip, w.state.selected_track = None, 0
QTest.keyClick(w.lanes, Qt.Key_E, Qt.ControlModifier)
check(len(w.project.tracks[0].clips) == 2, "Ctrl+E splits at the playhead")
w.state.selected_clip = w.project.tracks[0].clips[1]
QTest.keyClick(w.lanes, Qt.Key_D, Qt.ControlModifier)
check(len(w.project.tracks[0].clips) == 3, "Ctrl+D duplicates the clip")
QTest.keyClick(w.lanes, Qt.Key_Delete)
check(len(w.project.tracks[0].clips) == 2, "Del deletes the selected clip")

w.bpm.setFocus(); QApplication.processEvents()
QTest.keyClick(w.bpm, Qt.Key_Space)
check(w.player.playing, "Space plays even with the BPM box focused")
w.toggle_play()

answer_dialog(kind="synth", prompt="acid lead")
w.add_layer(); deliver(w); activate(w)
t = w.project.tracks
check(t[0].muted and not t[1].muted and t[1].name.startswith("+synth"), "add layer = new take, source muted")
QTest.keyClick(w.lanes, Qt.Key_Z, Qt.ControlModifier)
check(len(w.project.tracks) == 1 and not w.project.tracks[0].muted, "undo removes the take and unmutes")
QTest.keyClick(w.lanes, Qt.Key_Y, Qt.ControlModifier)
check(len(w.project.tracks) == 2, "Ctrl+Y redoes it")

take = w.project.tracks[1]
QTest.mouseClick(w.lanes, Qt.RightButton, pos=QPoint(60, ROW_H + 20))
items = labels(menus[-1])
check(any(i.startswith("Extend '") for i in items) and any(i.startswith("+4 bars") for i in items),
      "right-click ON a clip offers Extend +4/+8/+16")
check(any("Chop to pads" == i for i in items), "right-click a clip offers Chop to pads")
w.extend_to(take, take.end + 4 * w.project.bar_seconds)
task, params, _ = SENT[-1]
check(task == "repaint" and params["chunk_mask_mode"] == "explicit" and params["repainting_start"] < take.end,
      "extend = explicit-mask repaint starting before the end")
deliver(w); activate(w)
new = w.project.tracks[-1]
check("→ bar" in new.name and abs(new.clips[0].end - (take.end + 4 * w.project.bar_seconds)) <= w.project.bar_seconds,
      f"extended take trimmed to the target ({new.name})")

w.state.selection = (w.project.bar_seconds * 2, w.project.bar_seconds * 4)
w.redo_bars(new)
check(SENT[-1][0] == "repaint" and abs(SENT[-1][1]["repainting_start"] - w.project.bar_seconds * 2) < 1e-6,
      "redo bars = repaint of the selection")
w.cancel_generation()
check(not w.jobs and not w.state.pending and getattr(w.engine, "restarted", False), "cancel clears pending work")

n = len(w.project.tracks)
w.regenerate(w.project.tracks[1]); deliver(w); activate(w)
check(len(w.project.tracks) == n + 1 and w.project.tracks[1].muted, "regenerate keeps the old take (muted)")

w.bpm.setValue(120); QApplication.processEvents()
check("90" in w.tempo_warning.text(), "changing BPM warns about tracks made at the old tempo")
w.bpm.setValue(90)
w.headers.headers[id(w.project.tracks[0])].pan.setValue(-50)
QTest.qWait(1000)
saved = json.loads((song / "project.json").read_text())
check(saved["tracks"][0]["pan"] == -0.5, "autosaves after a change, no Save needed")
reloaded = Project.load(song)
check([len(t.clips) for t in reloaded.tracks] == [len(t.clips) for t in w.project.tracks], "reload matches")

w.state.selected_clip, w.state.selected_track = None, 0
QTest.keyClick(w.lanes, Qt.Key_Delete)
check(w.project.tracks[0].name != "Drums v1", "Del with no clip selected deletes the track")
QTest.keyClick(w.lanes, Qt.Key_Z, Qt.ControlModifier)
check(w.project.tracks[0].name == "Drums v1", "…and Ctrl+Z brings it back")

w.close()
cleanup(song)
finish()

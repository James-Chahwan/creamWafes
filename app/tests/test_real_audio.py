"""Real audio: PortAudio → ALSA → PipeWire into a null sink, recorded back with parec.

The other tests use a silent fake stream, which is how a stalled-stream bug once shipped. Run
via tools/test.sh, which creates the null sink and sets PIPEWIRE_NODE so nothing is audible.
"""

import os
import subprocess
import tempfile
import time

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

from harness import check, cleanup, demo_song, fake_engine, finish, open_window

SINK = os.environ.get("PIPEWIRE_NODE")
if not SINK:
    raise SystemExit("run me through tools/test.sh (needs PIPEWIRE_NODE=<null sink>)")

fake_engine()
song = demo_song()
capture = tempfile.NamedTemporaryFile(suffix=".raw", delete=False).name
t_cap = time.monotonic()  # the recording's time zero; every mark below is relative to it
rec = subprocess.Popen(["parec", "-d", f"{SINK}.monitor", "--format=float32le", "--rate=48000",
                        "--channels=2", "--raw", "--latency-msec=10"], stdout=open(capture, "wb"))
time.sleep(0.4)
app, w = open_window(song)
t0 = t_cap
w.pads_dock.show()
marks: dict[str, float] = {}


def wait(sec):
    end = time.monotonic() + sec
    while time.monotonic() < end:
        app.processEvents()
        time.sleep(0.005)


check(w.player._stream.active, "the output stream opens")
w.player.seek(1.0)
w.toggle_play()
p0 = w.player.position
marks["song"] = time.monotonic() - t0
wait(1.0)
moved = w.player.position - p0
check(0.85 < moved < 1.15, f"the playhead moves in real time ({moved:.2f}s in 1.00s)")
w.toggle_play()
wait(0.3)

w.state.selected_clip = w.project.tracks[0].clips[0]
w.chop_to_pads("hits")
marks["pads"] = time.monotonic() - t0
for key in (Qt.Key_Z, Qt.Key_X, Qt.Key_C, Qt.Key_V):
    QTest.keyClick(w.lanes, key)
    wait(0.4)
check(len(w.player._incoming) == 0 and w.player.voice_count == 0, "pad hits get played (nothing stuck in the queue)")

w.player.seek(4.0)
w.pad_panel.rec.setChecked(True)
wait(0.2)
for key in (Qt.Key_Z, Qt.Key_X, Qt.Key_Z):
    QTest.keyClick(w.lanes, key)
    wait(0.35)
times = [round(h["t"], 2) for h in w._pad_hits]
check(len(times) == 3 and times[1] - times[0] > 0.25 and times[2] - times[1] > 0.25,
      f"recorded hits land at real, increasing times {times}")
w.toggle_play()
wait(0.3)
w.project.pads[4].mode = "hold"
marks["hold"] = time.monotonic() - t0
QTest.keyPress(w.lanes, Qt.Key_A)
wait(1.2)
QTest.keyRelease(w.lanes, Qt.Key_A)
marks["release"] = time.monotonic() - t0
wait(0.5)
end_mark = time.monotonic() - t0
wait(1.0)  # keep streaming so parec's buffer reaches the file
rec.send_signal(subprocess.signal.SIGINT)
rec.wait()
w.close()

audio = np.fromfile(capture, dtype=np.float32).reshape(-1, 2).mean(axis=1)
os.unlink(capture)
win = 4800  # 100 ms
check(len(audio) / 48000 >= end_mark, f"the capture covers the whole run ({len(audio) / 48000:.1f}s ≥ {end_mark:.1f}s)")


def loud_between(a: float, b: float) -> float:
    seg = audio[int(a * 48000):int(b * 48000)]
    return 20 * np.log10(np.sqrt(np.mean(seg ** 2)) + 1e-9) if len(seg) else -200


check(loud_between(marks["song"] + 0.1, marks["song"] + 0.9) > -40, "the song is actually audible")
hits = [loud_between(marks["pads"] + i * 0.4, marks["pads"] + i * 0.4 + 0.25) for i in range(4)]
check(all(h > -45 for h in hits), f"each pad hit is audible (dB: {', '.join(f'{h:.0f}' for h in hits)})")

held = [loud_between(marks["hold"] + 0.1 + k * 0.25, marks["hold"] + 0.35 + k * 0.25) for k in range(4)]
check(all(h > -45 for h in held), f"holding a key keeps the pad sounding for 1.2 s (dB: {', '.join(f'{h:.0f}' for h in held)})")
after = loud_between(marks["release"] + 0.25, marks["release"] + 0.45)
check(after < -60, f"releasing the key silences it ({after:.0f} dB after 0.25 s)")

# the watchdog: kill the output mid-playback (what a stalled device looks like) and make sure
# playback comes back by itself
app, w = open_window(song)
w.player.seek(1.0)
w.player.play()
wait(0.5)
w.player._stream.stop()  # no more callbacks from here on
wait(1.2)
p = w.player.position
wait(1.0)
moved = w.player.position - p
check("stalled" in w.engine_label.text() and 0.85 < moved < 1.15,
      f"a stalled output is detected and restarted automatically (moved {moved:.2f}s/1.00s; "
      f"status: {w.engine_label.text()!r})")
check("512" in w.engine_label.text(), "a one-off stall retries the fast (low-latency) setting first")
w.close()

cleanup(song)
finish()

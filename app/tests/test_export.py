"""End-to-end: File → Export song… in every format, level option and range (offscreen, silent)."""

import json
import os
import shutil
import subprocess

os.environ["LAYERDAW_NO_AUDIO"] = "1"

import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402
from PySide6.QtCore import QTimer  # noqa: E402
from PySide6.QtWidgets import QApplication, QMessageBox  # noqa: E402

from harness import check, cleanup, demo_song, fake_engine, finish, open_window  # noqa: E402
from layerdaw import export  # noqa: E402
from layerdaw.model import SR  # noqa: E402
from layerdaw.ui.export_dialog import ExportDialog  # noqa: E402


def probe(path):
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=bit_rate,duration:format_tags=title,artist"
                          ":stream=sample_rate,channels,codec_name:stream_tags=title,artist", "-of", "json", str(path)],
                         capture_output=True, text=True).stdout
    return json.loads(out or "{}")


# --- loudness, level, limiter
t = np.arange(20 * SR) / SR
tone = 10 ** (-23 / 20) * np.sin(2 * np.pi * 1000 * t)
check(abs(export.lufs(np.stack([tone, tone], 1)) + 23.0) < 0.05, "the loudness meter reads the EBU test tone as −23 LUFS")
rng = np.random.default_rng(0)
mix = (rng.standard_normal((20 * SR, 2)) * 0.2).astype(np.float32)
mix[SR * 3:SR * 3 + 50] = 1.7  # a clip
for level, want in (("stream", -14.0), ("club", -9.0)):
    out, rep = export.master(mix, level)
    check(abs(rep["lufs"] - want) < 0.3 and rep["peak_db"] <= -0.99,
          f"'{level}' lands at {rep['lufs']:.1f} LUFS with peaks held at {rep['peak_db']:.2f} dB")
out, rep = export.master(mix, "peak")
check(abs(rep["peak_db"] + 1.0) < 0.01, "'normalise peaks' puts the loudest moment at −1 dB")
out, rep = export.master(mix, "mixed")
check(rep["gain_db"] == 0 and np.abs(out).max() < 1.0 and np.allclose(out[:SR], mix[:SR], atol=1e-6),
      "'as mixed' leaves it alone except for keeping it from clipping")

# --- every format writes and reads back
song = demo_song()
tmp = song.parent / "out"
short = (np.stack([np.sin(2 * np.pi * 220 * t[:3 * SR])] * 2, 1) * 0.5).astype(np.float32)
for fmt in export.available():
    label, ext, qualities, default = export.FORMATS[fmt]
    p = export.write(tmp / "f", short, fmt, qualities[default][1], 44100, title="Night Drive", artist="Test Artist")
    info = probe(p)
    stream = (info.get("streams") or [{}])[0]
    dur = float(info.get("format", {}).get("duration", 0))
    rate = int(stream.get("sample_rate", 0))
    tags = {k.lower(): v for k, v in [*info.get("format", {}).get("tags", {}).items(),
                                      *stream.get("tags", {}).items()]}  # OGG keeps them on the stream
    ok = p.suffix == ext and abs(dur - 3.0) < 0.15 and rate == (48000 if fmt == "opus" else 44100)
    extra = ""
    if fmt == "mp3":
        kbps = int(info["format"]["bit_rate"]) // 1000
        ok &= abs(kbps - 320) <= 5  # (tags add a little on a 3 s file)
        extra = f", {kbps} kbps"
    if fmt in ("mp3", "flac", "ogg", "m4a"):
        ok &= tags.get("title") == "Night Drive" and tags.get("artist") == "Test Artist"
        extra += ", tagged"
    check(ok, f"{label}: {p.name} plays back {dur:.2f}s at {rate} Hz{extra}")

# --- the dialog and the real export
fake_engine()
app, w = open_window(song)
w.pads_dock.hide()
bar = w.project.bar_seconds
check(w.a_export_song.text() == "⤓ Export song…" and any(a is w.a_export_song for a in w.findChildren(type(w.a_layer)))
      and w.a_export_song.shortcut().toString() == "Ctrl+Shift+X", "File → Export song… (Ctrl+Shift+X), on the toolbar too")
w.state.selection = (2 * bar, 4 * bar)
dlg = ExportDialog(w)
check(dlg.format.currentData() == "mp3" and dlg.quality.currentText() == "320 kbps" and dlg.rate.currentData() == 44100,
      "defaults: MP3 320 kbps, 44.1 kHz")
check(dlg.range.currentData()[0] == "sel" and dlg.range.count() >= 2 and not dlg.tails.isEnabled(),
      "with bars selected it offers just those (no ring-out for a selection)")
dlg.format.setCurrentIndex(dlg.format.findData("flac"))
check([dlg.quality.itemText(i) for i in range(dlg.quality.count())] == ["24-bit", "16-bit (CD)"],
      "FLAC offers 24/16-bit")
dlg.format.setCurrentIndex(dlg.format.findData("opus"))
check(not dlg.rate.isEnabled() and dlg.rate.currentData() == 48000, "Opus is always 48 kHz")
dlg.format.setCurrentIndex(dlg.format.findData("mp3"))
dlg.folder.setText(str(tmp))
check(str(tmp / f"{w.project.name}.mp3") in dlg.summary.text() and "MB" in dlg.summary.text(),
      f"it shows where it's going and how big ({dlg.summary.text()})")
dlg.deleteLater()

w._quiet_export = True
w.state.selection = None
w.project.tracks[1].fx = {"width": 0.0}  # the bass, mono: sound settings must be in the export
opts = {"path": tmp / "whole.wav", "format": "wav", "quality": "PCM_24", "rate": 48000, "range": ("all", 0.0, w.project.length),
        "level": "stream", "tails": True, "title": "t", "artist": ""}
path = w.export_song(opts)
audio, sr = sf.read(path, dtype="float32")
check(sr == 48000 and w.project.length - 0.01 <= len(audio) / sr <= w.project.length + 6.0,
      f"whole song: {len(audio) / sr:.2f}s of a {w.project.length:.2f}s song (plus any ring-out)")
check(abs(export.lufs(audio, sr) + 14) < 0.5 and np.abs(audio).max() <= 10 ** (-1 / 20) + 1e-3,
      f"…at streaming loudness ({export.lufs(audio, sr):.1f} LUFS), peaks under −1 dB")
again = w.export_song(opts)
check(again.name == "whole (2).wav" and path.exists(), "exporting again never overwrites the earlier file")
w.state.selection = (2 * bar, 4 * bar)
sel = w.export_song({**opts, "path": tmp / "bars.flac", "format": "flac", "quality": "PCM_16", "rate": 44100,
                     "range": ("sel", 2 * bar, 4 * bar), "tails": False, "level": "mixed"})
a2, sr2 = sf.read(sel, dtype="float32")
check(sr2 == 44100 and abs(len(a2) / sr2 - 2 * bar) < 0.01, f"selected bars 3–4: exactly {2 * bar:.2f}s at 44.1 kHz")
check("Exported" in w.engine_label.text() and "LUFS" in w.engine_label.text(), f"the status bar says what it made ({w.engine_label.text()})")

w._quiet_export = False
seen = {}


def answer():
    box = next((x for x in QApplication.topLevelWidgets() if isinstance(x, QMessageBox) and x.isVisible()), None)
    if box is None:
        QTimer.singleShot(100, answer)
        return
    seen["text"] = box.text()
    seen["buttons"] = [b.text() for b in box.buttons()]
    box.button(QMessageBox.Ok).click()


QTimer.singleShot(200, answer)
w.export_song({**opts, "path": tmp / "done.mp3", "format": "mp3", "quality": 192, "rate": 44100})
check("done.mp3" in seen.get("text", "") and "▶ Play it" in seen.get("buttons", []) and "Open folder" in seen.get("buttons", []),
      "afterwards: what was made, with ▶ Play it / Open folder")
check(abs(int(probe(tmp / "done.mp3")["format"]["bit_rate"]) // 1000 - 192) <= 5, "192 kbps when asked")

w.close()
shutil.rmtree(tmp, ignore_errors=True)
cleanup(song)
finish()

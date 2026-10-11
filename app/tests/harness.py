"""Shared bits for the creamWafes end-to-end tests: fake engine, checks, a demo song."""

from __future__ import annotations

import os
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
# never take over the real app's inspect socket (that would cut the debug hook off from it)
os.environ["CREAMWAFES_SOCKET"] = str(Path(tempfile.gettempdir()) / f"creamwafes-test-{os.getpid()}.sock")

from PySide6.QtTest import QTest  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from creamwafes.engine import Engine  # noqa: E402
from creamwafes.model import SR, Project  # noqa: E402
from creamwafes.ui import main_window as mw  # noqa: E402

FAILS: list[str] = []
SENT: list[tuple] = []  # (task, params, output path) for every fake generation


def check(cond: bool, what: str) -> None:
    print(("PASS " if cond else "FAIL ") + what, flush=True)
    if not cond:
        FAILS.append(what)


def finish() -> None:
    print(f"\n{len(FAILS)} failure(s)" if FAILS else "\nall passed", flush=True)
    for widget in QApplication.topLevelWidgets():
        widget.close()
    QApplication.processEvents()
    Path(os.environ["CREAMWAFES_SOCKET"]).unlink(missing_ok=True)
    sys.stdout.flush()
    os._exit(1 if FAILS else 0)  # skip interpreter teardown: PySide can segfault there after modal dialogs


def demo_song(bpm: int = 90, bars: int = 8) -> Path:
    """A throwaway song folder with synthetic drums + bass, so tests never touch real songs."""
    root = Path(tempfile.mkdtemp(prefix="creamwafes-test-"))
    beat, n = 60 / bpm, int(bars * 4 * 60 / bpm * SR)
    rng = np.random.default_rng(1)
    t = np.arange(int(0.3 * SR)) / SR
    kick = np.sin(2 * np.pi * (45 + 90 * np.exp(-t / 0.03)) * t) * np.exp(-t / 0.08)
    hat = rng.standard_normal(int(0.05 * SR)) * np.exp(-np.arange(int(0.05 * SR)) / SR / 0.012) * 0.3
    drums, bass = np.zeros(n), np.zeros(n)
    for b in range(bars * 8):
        i = int(b * beat / 2 * SR)
        hit = kick if b % 2 == 0 else hat
        drums[i:i + len(hit)] += hit[:n - i]
    tb = np.arange(n) / SR
    bass[:] = 0.4 * np.sin(2 * np.pi * 55 * tb) * (0.6 + 0.4 * np.sign(np.sin(2 * np.pi * tb / beat)))
    project = Project(root / "song", bpm=bpm, key="A minor")
    for name, audio in (("drums", drums), ("bass", bass)):
        src = root / f"{name}.wav"
        sf.write(src, (audio / np.abs(audio).max() * 0.7).astype(np.float32), SR)
        project.add_file(src, name, name)
    project.save()
    return root / "song"


def drum_loop(bpm: int = 120, bars: int = 4) -> np.ndarray:
    """A loop with four clearly different drums (kick, snare, hat, clap) in a busy pattern."""
    rng = np.random.default_rng(2)
    t = lambda sec: np.arange(int(sec * SR)) / SR  # noqa: E731

    def band(noise, lo, hi):
        spec = np.fft.rfft(noise)
        f = np.fft.rfftfreq(len(noise), 1 / SR)
        spec[(f < lo) | (f > hi)] = 0
        return np.fft.irfft(spec, len(noise))

    tk, ts, th, tc = t(0.3), t(0.25), t(0.06), t(0.15)
    kick = np.sin(2 * np.pi * (40 + 80 * np.exp(-tk / 0.03)) * tk) * np.exp(-tk / 0.09)
    noise = band(rng.standard_normal(len(ts)), 200, 9000)
    snare = (noise / np.abs(noise).max() * 0.6 + 0.5 * np.sin(2 * np.pi * 190 * ts)) * np.exp(-ts / 0.07)
    hat = band(rng.standard_normal(len(th)), 7000, 16000) * np.exp(-th / 0.012) * 3
    burst = np.exp(-(tc % 0.012) / 0.003) * (tc < 0.036) + np.exp(-np.maximum(tc - 0.036, 0) / 0.03) * (tc >= 0.036)
    clap = band(rng.standard_normal(len(tc)), 1800, 4500) * burst * 2
    beat = 60 / bpm
    n = int(bars * 4 * beat * SR)
    out = np.zeros(n)
    pattern = {0: kick, 1: hat, 2: snare, 3: hat, 4: kick, 5: clap, 6: snare, 7: hat}
    for step in range(bars * 8):
        i = int(step * beat / 2 * SR)
        hit = pattern[step % 8]
        out[i:i + len(hit)] += hit[:n - i] / np.abs(hit).max() * (0.3 if hit is hat else 0.8)
    return np.stack([out, out], 1).astype(np.float32)


def fake_engine(source: np.ndarray | None = None) -> None:
    """Generations return instantly: ``source`` audio (or noise) of the requested length."""
    Engine.start = lambda self: (setattr(self, "is_ready", True), self.status.emit("Engine ready (fake)"))
    Engine.restart = lambda self: (self.busy.clear(), setattr(self, "restarted", True))

    def submit(self, task, params, out_dir):
        out_dir.mkdir(parents=True, exist_ok=True)
        n = int(params.get("duration", 10.0) * SR)
        audio = np.resize(source, (n, 2)) if source is not None else np.random.default_rng(0).normal(0, .1, (n, 2))
        path = out_dir / f"fake{len(SENT)}.wav"
        sf.write(path, audio.astype(np.float32), SR)
        SENT.append((task, params, path))
        return f"j{len(SENT)}"

    Engine.submit = submit


def deliver(window) -> None:
    """Pretend the engine finished the most recent request."""
    window._job_done(f"j{len(SENT)}", {"path": str(SENT[-1][2]), "seconds": 40.0})


def open_window(song: Path, size=(1300, 760)):
    app = QApplication.instance()
    if app is None:  # look exactly like the real app, so screenshots are representative
        from creamwafes.__main__ import dark_palette
        app = QApplication([])
        app.setStyle("Fusion")
        app.setPalette(dark_palette())
    w = mw.MainWindow(Project.load(song))
    w.resize(*size)
    w.show()
    activate(w)
    return app, w


def activate(w) -> None:
    """What the window manager does after a dialog closes; without it shortcuts don't fire offscreen."""
    w.activateWindow()
    QTest.qWaitForWindowActive(w)
    w.lanes.setFocus()
    QApplication.processEvents()


def cleanup(song: Path) -> None:
    shutil.rmtree(song.parent, ignore_errors=True)

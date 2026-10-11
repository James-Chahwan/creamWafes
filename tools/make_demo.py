"""Make a tiny synthetic demo project (drums + bass, 8 bars, 90 BPM, A minor) to try layering on.

    cd app && uv run python ../tools/make_demo.py ~/Music/creamWafes/demo
"""

import sys
from pathlib import Path

import numpy as np
import soundfile as sf

from creamwafes.model import SR, Project

BPM, BARS = 90, 8
beat = 60 / BPM
n = int(BARS * 4 * beat * SR)
rng = np.random.default_rng(1)


def place(buf, sound, t):
    i = int(t * SR)
    m = min(len(sound), len(buf) - i)
    buf[i:i + m] += sound[:m]


def env(length, decay):
    t = np.arange(int(length * SR)) / SR
    return t, np.exp(-t / decay)


t, e = env(0.4, 0.08)
kick = np.sin(2 * np.pi * (45 + 90 * np.exp(-t / 0.03)) * t) * e
t, e = env(0.25, 0.06)
snare = (rng.standard_normal(len(t)) * 0.6 + np.sin(2 * np.pi * 190 * t) * 0.4) * e
t, e = env(0.06, 0.015)
hat = rng.standard_normal(len(t)) * e * 0.25

drums = np.zeros(n)
for b in range(BARS * 4):
    place(drums, hat, b * beat)
    place(drums, hat, (b + 0.5) * beat)
    place(drums, snare if b % 2 else kick, b * beat)
    if b % 4 == 3:
        place(drums, kick, (b + 0.5) * beat)

bass = np.zeros(n)
roots = [55.0, 55.0, 43.65, 49.0]  # A, A, F, G
for bar in range(BARS):
    f = roots[bar % 4]
    for step, length in ((0, 1.5), (1.5, 0.5), (2, 1.5), (3.5, 0.5)):
        t, e = env(length * beat, 0.35)
        note = (np.sin(2 * np.pi * f * t) + 0.3 * np.sin(4 * np.pi * f * t)) * np.minimum(e, 1) * np.minimum(t * 200, 1)
        place(bass, note * 0.5, (bar * 4 + step) * beat)

folder = Path(sys.argv[1]).expanduser()
src = folder / "src"
src.mkdir(parents=True, exist_ok=True)
project = Project(folder, bpm=BPM, key="A minor")
for name, kind, audio in (("drums", "drums", drums), ("bass", "bass", bass)):
    path = src / f"{name}.wav"
    sf.write(path, (audio / np.abs(audio).max() * 0.7).astype(np.float32), SR)
    project.add_file(path, name, kind)
project.save()
print(f"demo project: {folder}")

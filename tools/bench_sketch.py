"""Real-engine check of "sketch → song part": cover (produce it) and complete (add instruments).

    cd engine/ACE-Step-1.5 && uv run --no-sync python ../../tools/bench_sketch.py
"""

import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "engine" / "ACE-Step-1.5"))
sys.path.insert(0, str(ROOT / "app" / "layerdaw"))
sys.path.insert(0, str(ROOT / "app" / "tests"))
sys.path.insert(0, str(ROOT / "app"))

import engine_worker  # noqa: E402
from acestep.inference import GenerationConfig, GenerationParams, generate_music  # noqa: E402
from harness import drum_loop  # noqa: E402  (synthetic kick/snare/hat/clap loop)
from layerdaw.chords import synth_note  # noqa: E402

SR, BPM, BARS = 48000, 120, 8
OUT = ROOT / "test_out" / "sketch"
OUT.mkdir(parents=True, exist_ok=True)
bar = 240 / BPM


def sketch() -> np.ndarray:
    drums = drum_loop(BPM, BARS)
    chords = np.zeros_like(drums)
    prog = [[57, 60, 64], [53, 57, 60], [48, 52, 55], [55, 59, 62]]  # Am F C G
    for b in range(BARS):
        for beat in (0, 2):  # chord stabs on beats 1 and 3, like pad hits
            a = int((b * bar + beat * bar / 4) * SR)
            stab = sum(synth_note(m, 0.4)[:, 0] for m in prog[b % 4]) * 0.25
            chords[a:a + len(stab), 0] += stab[: len(chords) - a]
    chords[:, 1] = chords[:, 0]
    return np.clip(drums * 0.8 + chords, -1, 1).astype(np.float32)


def onset_env(x):
    m = x.mean(axis=1)
    hop = 512
    e = np.sqrt((m[: len(m) // hop * hop].reshape(-1, hop) ** 2).mean(axis=1))
    return np.maximum(np.diff(e), 0)


def run(dit, name, **kw):
    t0 = time.perf_counter()
    params = GenerationParams(thinking=False, seed=11, bpm=BPM, keyscale="A minor", timesignature="4",
                              inference_steps=32, lyrics="[Instrumental]", **kw)
    r = generate_music(dit, None, params, GenerationConfig(batch_size=1, use_random_seed=False, audio_format="wav"),
                       save_dir=str(OUT / name))
    if not r.success:
        print(f"[{name}] FAILED {r.error}", flush=True)
        return None
    out, _ = sf.read(r.audios[0]["path"], dtype="float32")
    print(f"[{name}] {time.perf_counter() - t0:.1f}s → {r.audios[0]['path']}", flush=True)
    return out


src = sketch()
path = OUT / "sketch.wav"
sf.write(path, src, SR)
dit, _ = engine_worker.load_handler("acestep-v15-base")
results = {
    "cover_0.7": run(dit, "cover_0.7", task_type="cover", src_audio=str(path), audio_cover_strength=0.7,
                     caption="dark cyberpunk trap, heavy 808 bass, distorted drums, dark synth chords"),
    "complete": run(dit, "complete", task_type="complete", src_audio=str(path),
                    instruction="Complete the input track with BASS | SYNTH | STRINGS:",
                    caption="dark cyberpunk trap, heavy 808 bass, dark synth pads, cinematic strings"),
}
se = onset_env(src)
for name, out in results.items():
    if out is None:
        continue
    oe = onset_env(out)
    n = min(len(se), len(oe))
    rhythm = np.corrcoef(se[:n], oe[:n])[0, 1]
    corr = np.corrcoef(src[: len(out), 0], out[: len(src), 0])[0, 1]
    spec = lambda x: np.abs(np.fft.rfft(x[:, 0][: SR * 8]))  # noqa: E731
    f = np.fft.rfftfreq(SR * 8, 1 / SR)
    band = lambda s, lo, hi: 10 * np.log10(np.sum(s[(f >= lo) & (f < hi)] ** 2) + 1e-9)  # noqa: E731
    s_in, s_out = spec(src), spec(out)
    print(f"{name:10} len {len(out) / SR:5.1f}s (sketch {len(src) / SR:.1f}s) | rhythm follows sketch {rhythm:+.2f} | "
          f"same audio {corr:+.2f} | rms {np.sqrt(np.mean(out ** 2)):.3f} | low-end change "
          f"{band(s_out, 30, 120) - band(s_in, 30, 120):+.1f} dB | mids change {band(s_out, 300, 3000) - band(s_in, 300, 3000):+.1f} dB")

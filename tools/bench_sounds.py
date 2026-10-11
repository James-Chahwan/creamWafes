"""Base vs turbo for single sounds: speed, and how much of each take is the sound that was asked for.

    cd engine/ACE-Step-1.5 && uv run --no-sync python ../../tools/bench_sounds.py
"""

import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "engine" / "ACE-Step-1.5"))
sys.path.insert(0, str(ROOT / "app" / "layerdaw"))
sys.path.insert(0, str(ROOT / "app"))

import engine_worker  # noqa: E402
from acestep.inference import GenerationConfig, GenerationParams, generate_music  # noqa: E402
from layerdaw import chop, sounds  # noqa: E402
from layerdaw.chords import detect_pitch  # noqa: E402

SR, BPM, KEY = 48000, 140, "F minor"
beat, bar = 60 / BPM, 240 / BPM
OUT = ROOT / "test_out" / "sounds"
TESTS = [("kick", "hard punchy techno"), ("snare", "crisp trap"), ("808", "distorted cyberpunk"),
         ("keys", "warm Rhodes"), ("vocal chop", "pitched female")]


def score(audio: np.ndarray, kind: str) -> str:
    how = sounds.SOUND_TYPES[kind][1]
    if how.startswith("drum:"):
        want = how.split(":")[1]
        roles = [chop.drum_role(chop.hit_features(audio[s:e], audio[max(0, s - 3840):s]))
                 for s, e in chop.by_hits(audio, beat, count=10_000)]
        hit = sum(r == want for r in roles)
        return f"{hit}/{len(roles)} hits are {want}s"
    hits = chop.onsets(audio.mean(axis=1), beat / 4)
    pitched = 0
    for i, s in enumerate(hits):
        e = hits[i + 1] if i + 1 < len(hits) else len(audio)
        if e - s >= int(0.15 * SR) and detect_pitch(audio[s:e]) is not None:
            pitched += 1
    return f"{pitched}/{len(hits)} sounds clearly pitched"


handlers = {"base": engine_worker.load_handler("acestep-v15-base")[0],
            "turbo": engine_worker.load_handler(engine_worker.TURBO)[0]}
for kind, words in TESTS:
    for model, h in handlers.items():
        steps = 8 if model == "turbo" else 32
        extra = {"shift": 3.0} if model == "turbo" else {}
        params = GenerationParams(task_type="text2music", thinking=False, seed=5, caption=sounds.caption(kind, words),
                                  lyrics="[Instrumental]", instrumental=True, bpm=BPM, keyscale=KEY, timesignature="4",
                                  duration=10.5, inference_steps=steps, **extra)
        t0 = time.perf_counter()
        r = generate_music(h, None, params, GenerationConfig(batch_size=1, use_random_seed=False, audio_format="wav"),
                           save_dir=str(OUT / f"{kind}-{model}"))
        took = time.perf_counter() - t0
        if not r.success:
            print(f"{kind:10} {model:5} FAILED {r.error}", flush=True)
            continue
        audio, _ = sf.read(r.audios[0]["path"], dtype="float32")
        cands = sounds.candidates(audio, kind, beat, bar)
        print(f"{kind:10} {model:5} {took:5.1f}s | {score(audio, kind):28} | candidates: "
              f"{', '.join(c[2] for c in cands) or 'none'}", flush=True)

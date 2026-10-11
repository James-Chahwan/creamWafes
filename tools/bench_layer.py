"""Benchmark ACE-Step 1.5 add-layer ("lego") on this machine.

Generates a short backing track, then layers keys and vocals on top of it,
timing each step and checking the output isn't silent/NaN (fp16 issues on
pre-Ampere cards like the GTX 1660 tend to show up that way).

Run from the musicwork root:
    uv run --project engine/ACE-Step-1.5 python tools/bench_layer.py
"""

import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

ROOT = Path(__file__).resolve().parents[1]
ENGINE = ROOT / "engine" / "ACE-Step-1.5"
OUT = ROOT / "test_out"
sys.path.insert(0, str(ENGINE))
sys.path.insert(0, str(ROOT / "app" / "layerdaw"))

import engine_worker  # noqa: E402  (same low-VRAM loader + quantization fix the app uses)
from acestep.inference import GenerationConfig, GenerationParams, generate_music  # noqa: E402

BPM, KEY, DURATION, STEPS = 90, "A minor", 30, 32


def describe(path: str) -> str:
    audio, sr = sf.read(path)
    if np.isnan(audio).any():
        return "BROKEN: contains NaN"
    rms = float(np.sqrt(np.mean(audio**2)))
    return f"{len(audio) / sr:.1f}s @ {sr}Hz, rms={rms:.4f}" + ("  (SILENT?)" if rms < 1e-3 else "")


def run(dit, name: str, **kwargs) -> str:
    params = GenerationParams(bpm=BPM, keyscale=KEY, timesignature="4", duration=DURATION,
                              inference_steps=STEPS, thinking=False, seed=42, **kwargs)
    config = GenerationConfig(batch_size=1, use_random_seed=False, audio_format="wav")
    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    result = generate_music(dit, None, params, config, save_dir=str(OUT / name))
    elapsed = time.perf_counter() - t0
    if not result.success:
        raise RuntimeError(f"{name} failed: {result.error}")
    path = result.audios[0]["path"]
    peak = torch.cuda.max_memory_allocated() / 2**30
    print(f"[{name}] {elapsed:6.1f}s  peak VRAM {peak:.2f} GiB  -> {path}\n        {describe(path)}", flush=True)
    return path


def main() -> None:
    OUT.mkdir(exist_ok=True)
    t0 = time.perf_counter()
    dit, _ = engine_worker.load_handler("acestep-v15-base")
    print(f"[init] {time.perf_counter() - t0:.1f}s", flush=True)

    backing = run(dit, "1_backing", task_type="text2music", instrumental=True,
                  caption="lofi hip hop groove, dusty drums and round electric bass, no melody",
                  lyrics="[Instrumental]")
    run(dit, "2_keys_layer", task_type="lego", src_audio=backing,
        instruction="Generate the KEYBOARD track based on the audio context:",
        caption="warm Rhodes electric piano chords, jazzy, mellow", lyrics="[Instrumental]")
    run(dit, "3_vocal_layer", task_type="lego", src_audio=backing, vocal_language="en",
        instruction="Generate the VOCALS track based on the audio context:",
        caption="soft female R&B vocal, intimate, breathy",
        lyrics="[Verse]\nLate night city, neon on the rain\nI keep on walking, calling out your name")


if __name__ == "__main__":
    main()

"""Second pass after bench_layer.py: pull the new part back out of each lego mix with `extract`.

    cd engine/ACE-Step-1.5 && uv run --no-sync python ../../tools/bench_extract.py
"""

import glob
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "test_out"
sys.path.insert(0, str(ROOT / "engine" / "ACE-Step-1.5"))
sys.path.insert(0, str(ROOT / "app" / "layerdaw"))

import engine_worker  # noqa: E402
from acestep.inference import GenerationConfig, GenerationParams, generate_music  # noqa: E402


def mono(path) -> np.ndarray:
    return sf.read(path, dtype="float32")[0].mean(axis=1)


def main() -> None:
    dit, _ = engine_worker.load_handler("acestep-v15-base")
    backing = mono(glob.glob(str(OUT / "1_backing" / "*.wav"))[0])
    for mix_dir, track in (("2_keys_layer", "KEYBOARD"), ("3_vocal_layer", "VOCALS")):
        src = glob.glob(str(OUT / mix_dir / "*.wav"))[0]
        params = GenerationParams(task_type="extract", src_audio=src, thinking=False, seed=42,
                                  instruction=f"Extract the {track} track from the audio:",
                                  inference_steps=32, bpm=90, keyscale="A minor", timesignature="4")
        t0 = time.perf_counter()
        result = generate_music(dit, None, params, GenerationConfig(batch_size=1, use_random_seed=False,
                                                                    audio_format="wav"),
                                save_dir=str(OUT / f"4_extract_{track.lower()}"))
        if not result.success:
            print(f"[extract {track}] FAILED: {result.error}", flush=True)
            continue
        stem = mono(result.audios[0]["path"])
        rms = float(np.sqrt(np.mean(stem**2)))
        corr = float(np.corrcoef(stem, backing)[0, 1])
        print(f"[extract {track}] {time.perf_counter() - t0:.1f}s rms={rms:.4f} corr_with_backing={corr:+.3f}"
              f" -> {result.audios[0]['path']}", flush=True)


if __name__ == "__main__":
    main()

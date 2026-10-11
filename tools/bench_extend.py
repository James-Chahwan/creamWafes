"""Real-engine check of "Extend to here": repaint with repainting_end past the end of the source.

    cd engine/ACE-Step-1.5 && uv run --no-sync python ../../tools/bench_extend.py SRC.wav BPM
"""

import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "engine" / "ACE-Step-1.5"))
sys.path.insert(0, str(ROOT / "app" / "creamwafes"))

import engine_worker  # noqa: E402
from acestep.inference import GenerationConfig, GenerationParams, generate_music  # noqa: E402


def main() -> None:
    src, bpm = Path(sys.argv[1]), int(sys.argv[2])
    audio, sr = sf.read(src, dtype="float32")
    end, bar = len(audio) / sr, 240 / bpm
    target = end + 4 * bar
    from creamwafes_generate import last_loud_bar_end
    start = max(0.0, min(last_loud_bar_end(audio, bar), end) - bar)
    print(f"[extend] repaint {start:.2f}s -> {target:.2f}s (song ends {end:.2f}s)", flush=True)
    padded = ROOT / "test_out" / "extend_src_padded.wav"
    padded.parent.mkdir(exist_ok=True)
    sf.write(padded, np.concatenate([audio, np.zeros((int((target - end) * sr), audio.shape[1]), np.float32)]), sr)
    src = padded  # same trick as the app: pre-pad so the model is told the real target length
    dit, _ = engine_worker.load_handler("acestep-v15-base")
    params = GenerationParams(task_type="repaint", src_audio=str(src), thinking=False, seed=7,
                              caption=sys.argv[4] if len(sys.argv) > 4 else "continue the song in the same style", lyrics="[Instrumental]",
                              bpm=bpm, timesignature="4", inference_steps=32,
                              repainting_start=start, repainting_end=target, duration=target,
                              chunk_mask_mode=sys.argv[3] if len(sys.argv) > 3 else "auto")
    t0 = time.perf_counter()
    result = generate_music(dit, None, params, GenerationConfig(batch_size=1, use_random_seed=False,
                                                                audio_format="wav"),
                            save_dir=str(ROOT / "test_out" / "5_extend"))
    if not result.success:
        print(f"[extend] FAILED: {result.error}", flush=True)
        return
    out, osr = sf.read(result.audios[0]["path"], dtype="float32")
    mono_out, mono_src = out.mean(axis=1), audio.mean(axis=1)
    keep = int(start * sr)
    corr = float(np.corrcoef(mono_out[:keep], mono_src[:keep])[0, 1])
    tail = mono_out[int(end * osr):int((target - bar) * osr)]  # new bars, minus the last one
    tail_rms = float(np.sqrt(np.mean(tail**2))) if len(tail) else 0.0
    src_rms = float(np.sqrt(np.mean(mono_src**2)))
    print(f"[extend] {time.perf_counter() - t0:.1f}s  src {end:.2f}s -> out {len(out) / osr:.2f}s "
          f"(asked {target:.2f}s) | kept part corr {corr:+.3f} | new tail rms {tail_rms:.3f} vs src {src_rms:.3f}"
          f" -> {result.audios[0]['path']}", flush=True)


if __name__ == "__main__":
    main()

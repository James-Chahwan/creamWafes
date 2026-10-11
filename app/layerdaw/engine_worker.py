"""ACE-Step worker process. Runs inside the engine's own venv, not the app's.

Loads the model once, then reads one JSON job per line on stdin and answers
with lines on stdout prefixed by ``@@ ``. Everything else the engine prints is
pushed to stderr so it can't corrupt the protocol.

Job:    {"id": "...", "task": "lego", "out_dir": "...", "params": {...GenerationParams fields}}
Events: ready | started | progress | done | error
"""

import json
import os
import sys
import time
import traceback
from pathlib import Path
from unittest.mock import patch

_protocol = os.fdopen(os.dup(sys.stdout.fileno()), "w", buffering=1)
os.dup2(sys.stderr.fileno(), sys.stdout.fileno())
sys.stdout = sys.stderr

ENGINE = Path(os.environ.get("LAYERDAW_ENGINE", Path(__file__).resolve().parents[2] / "engine" / "ACE-Step-1.5"))
sys.path.insert(0, str(ENGINE))


def emit(event: str, **fields) -> None:
    _protocol.write("@@ " + json.dumps({"event": event, **fields}) + "\n")


def load_handler(model: str):
    from acestep.handler import AceStepHandler

    import torch

    vram_gb = torch.cuda.get_device_properties(0).total_memory / 2**30 if torch.cuda.is_available() else 0
    low_vram = vram_gb < 12
    if vram_gb:
        prefer_bf16_on_pre_ampere()
    handler = AceStepHandler()
    # The main-model check insists on turbo + the 1.7B LM; we don't use either on small cards.
    with patch("acestep.core.generation.handler.init_service_downloads.check_main_model_exists",
               return_value=True):
        status, ok = handler.initialize_service(
            project_root=str(ENGINE), config_path=model,
            device="cuda" if vram_gb else "cpu",
            compile_model=low_vram, offload_to_cpu=low_vram, offload_dit_to_cpu=low_vram,
            quantization="int8_weight_only" if low_vram else None,
        )
    if not ok:
        raise RuntimeError(status)
    if low_vram:
        ensure_dit_quantized(handler)
    return handler, vram_gb


def prefer_bf16_on_pre_ampere() -> None:
    """Run pre-Ampere cards (GTX 16xx/RTX 20xx) in emulated bf16 instead of fp16.

    ACE-Step picks fp16 there, and the DiT overflows: every latent comes back NaN. bf16 has
    fp32's range so it can't overflow, costs the same memory, and on a GTX 1660 its emulated
    matmuls measured ~5x faster than fp16 ones. Set LAYERDAW_DTYPE=fp16 to opt out.
    """
    import torch

    from acestep import gpu_config
    from acestep.core.generation.handler import memory_utils

    if os.environ.get("LAYERDAW_DTYPE") == "fp16" or torch.cuda.get_device_capability()[0] >= 8:
        return
    if not torch.cuda.is_bf16_supported(including_emulation=True):
        return
    gpu_config.cuda_supports_bfloat16 = lambda *_a, **_k: True
    memory_utils.cuda_supports_bfloat16 = lambda *_a, **_k: True
    print("[layerdaw] pre-Ampere GPU: forcing emulated bfloat16 (fp16 overflows to NaN)", file=sys.stderr)


def ensure_dit_quantized(handler) -> None:
    """Work around ACE-Step compiling before quantizing.

    torch.compile wraps the model, so every module name gains an ``_orig_mod.`` prefix and
    the DiT filter (which wants names starting with ``decoder``) matches nothing: the log
    says "quantized" but the weights stay fp16 and the 2B model no longer fits in 6 GB.
    Quantizing the unwrapped module in place fixes it; the compiled wrapper shares it.
    """
    wrapper = handler.model
    inner = getattr(wrapper, "_orig_mod", None)
    if inner is None:
        return
    handler.model = inner
    try:
        handler._apply_dit_quantization("int8_weight_only")
    finally:
        handler.model = wrapper
    kinds = {type(m.weight).__name__ for n, m in inner.named_modules()
             if n.startswith("decoder.layers") and hasattr(m, "weight") and m.weight is not None}
    print(f"[layerdaw] decoder weight types after quantization: {sorted(kinds)}", file=sys.stderr)


# Only the base model can add/extract/complete a track; everything else sounds better (and is ~4×
# faster) on turbo, ACE-Step's "very high" quality model, when it's installed.
BASE_ONLY = {"lego", "complete", "extract"}
TURBO = "acestep-v15-turbo"


def pick_model(handlers: dict, task: str) -> str:
    return "turbo" if task not in BASE_ONLY and "turbo" in handlers else "base"


def release_gpu(handlers: dict) -> None:
    """Give the GPU back between jobs. ACE-Step offloads each model to the CPU after use, but its VAE
    uses (old-style) weight norm, which recomputes ``weight`` before every forward and keeps it as a
    plain attribute — so ``.to("cpu")`` leaves ~330 MB of those copies on the GPU after every job,
    and a long add-layer right after another runs out of memory on a 6 GB card. They're rebuilt on
    the next forward anyway, so move any such stray CUDA tensor off the GPU, then empty the cache."""
    import gc
    import torch
    if not torch.cuda.is_available():
        return
    for handler in handlers.values():
        for part in vars(handler).values():
            if not isinstance(part, torch.nn.Module):
                continue
            for module in part.modules():
                owned = set(module._parameters) | set(module._buffers)
                on_cpu = all(not p.is_cuda for p in module.parameters(recurse=False))
                for name, value in list(vars(module).items()):
                    if torch.is_tensor(value) and value.is_cuda and name not in owned and on_cpu:
                        setattr(module, name, value.detach().to("cpu"))
    gc.collect()
    torch.cuda.empty_cache()


_separator = None


def separate(path: str, stem: str, out_dir: str) -> str:
    """Pull one stem ("vocals", "drums", "bass" or "other") out of a mix with Demucs htdemucs (MIT).
    ACE-Step's lego returns the whole song with the new part in it; this keeps just the part."""
    global _separator
    import julius
    import numpy as np
    import soundfile as sf
    import torch
    try:
        from demucs.apply import apply_model
        from demucs.pretrained import get_model
    except ImportError as exc:
        raise RuntimeError("Stem separation isn't installed in the engine — run tools/setup_engine.sh") from exc
    if _separator is None:
        _separator = get_model("htdemucs").eval()
    model = _separator
    audio, sr = sf.read(path, dtype="float32", always_2d=True)
    x = torch.from_numpy(np.ascontiguousarray(audio.T))
    if x.shape[0] == 1:
        x = x.repeat(2, 1)
    if sr != model.samplerate:
        x = julius.resample_frac(x, sr, model.samplerate)
    ref = x.mean(0)
    mean, std = ref.mean(), ref.std() + 1e-8

    def run(device: str):
        with torch.no_grad():
            return apply_model(model.to(device), ((x - mean) / std)[None].to(device), split=True,
                               overlap=0.25, progress=False)[0].cpu()
    try:
        out = run("cuda" if torch.cuda.is_available() else "cpu")
    except torch.cuda.OutOfMemoryError:  # the generator is still on the GPU: do it on the CPU instead
        torch.cuda.empty_cache()
        out = run("cpu")
    finally:
        model.to("cpu")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    y = out[model.sources.index(stem)] * std + mean
    if sr != model.samplerate:
        y = julius.resample_frac(y, model.samplerate, sr)
    dest = Path(out_dir) / f"{Path(path).stem}-{stem}.wav"
    sf.write(dest, y.numpy().T[:len(audio)], sr, subtype="FLOAT")
    return str(dest)


def run_job(handlers: dict, job: dict) -> None:
    from acestep.inference import GenerationConfig, GenerationParams, generate_music

    job_id = job["id"]
    if job["task"] == "separate":  # no generating: split existing audio
        emit("started", id=job_id, model="demucs")
        t0 = time.perf_counter()
        path = separate(job["params"]["src"], job["params"]["stem"], job["out_dir"])
        emit("done", id=job_id, path=path, seconds=round(time.perf_counter() - t0, 1), model="demucs")
        return
    which = pick_model(handlers, job["task"])
    handler = handlers[which]
    emit("started", id=job_id, model=which)
    fields = dict(job["params"])
    isolate = fields.pop("isolate", None)  # keep only this stem of the result (see separate())
    if which == "turbo":  # turbo is distilled for 8 steps; its docs recommend shift 3
        fields["inference_steps"] = min(int(fields.get("inference_steps", 8)), 8)
        fields.setdefault("shift", 3.0)
    params = GenerationParams(task_type=job["task"], thinking=False, **fields)
    config = GenerationConfig(batch_size=1, use_random_seed=params.seed < 0, audio_format="wav")

    def progress(value, desc=""):
        emit("progress", id=job_id, value=float(value), desc=str(desc))

    t0 = time.perf_counter()
    result = generate_music(handler, None, params, config, save_dir=job["out_dir"], progress=progress)
    if not result.success and "out of memory" in str(result.error).lower():  # clear the GPU and try once more
        release_gpu(handlers)
        progress(0.0, "GPU was full: cleared it, trying again")
        result = generate_music(handler, None, params, config, save_dir=job["out_dir"], progress=progress)
    if not result.success:
        emit("error", id=job_id, message=str(result.error))
        return
    audio = result.audios[0]
    path = audio["path"]
    if isolate:
        progress(0.98, f"keeping only the {isolate}")
        path = separate(path, isolate, job["out_dir"])
    emit("done", id=job_id, path=path, mix_path=audio["path"], seed=audio["params"].get("seed"),
         seconds=round(time.perf_counter() - t0, 1), model=which)


def main() -> None:
    model = os.environ.get("LAYERDAW_MODEL", "acestep-v15-base")
    handlers: dict = {}
    try:
        t0 = time.perf_counter()
        handlers["base"], vram_gb = load_handler(model)
        emit("ready", model=model, vram_gb=round(vram_gb, 1), seconds=round(time.perf_counter() - t0, 1))
    except Exception as exc:  # noqa: BLE001 - report any startup failure to the app
        traceback.print_exc()
        emit("fatal", message=str(exc))
        sys.exit(1)
    if (ENGINE / "checkpoints" / TURBO / "model.safetensors").exists() and not os.environ.get("LAYERDAW_NO_TURBO"):
        try:  # both stay in system RAM; only the one generating moves onto the GPU
            t0 = time.perf_counter()
            handlers["turbo"], _ = load_handler(TURBO)
            emit("models", turbo=True, seconds=round(time.perf_counter() - t0, 1))
        except Exception as exc:  # noqa: BLE001 - base still works on its own
            traceback.print_exc()
            emit("models", turbo=False, message=str(exc))

    for line in sys.stdin:
        if not line.strip():
            continue
        job = {}
        try:
            job = json.loads(line)
            run_job(handlers, job)
        except Exception as exc:  # noqa: BLE001 - one bad job shouldn't kill the worker
            traceback.print_exc()
            emit("error", id=job.get("id"), message=str(exc))
        finally:
            release_gpu(handlers)


if __name__ == "__main__":
    main()

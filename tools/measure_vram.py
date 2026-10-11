"""Load the engine like the worker does, then measure what the DiT bundle costs on the GPU.

    cd engine/ACE-Step-1.5 && uv run --no-sync python ../../tools/measure_vram.py
"""

import os
import sys
from collections import defaultdict
from pathlib import Path

import torch

APP = Path(__file__).resolve().parents[1] / "app" / "creamwafes"
sys.path.insert(0, str(APP))
os.environ.setdefault("CREAMWAFES_ENGINE", str(APP.parents[1] / "engine" / "ACE-Step-1.5"))
import engine_worker  # noqa: E402  (redirects stdout to stderr; fine here)

GiB = 2**30


def tensor_bytes(t: torch.Tensor) -> int:
    if hasattr(t, "__tensor_flatten__"):
        names, _ = t.__tensor_flatten__()
        return sum(tensor_bytes(getattr(t, n)) for n in names)
    return t.numel() * t.element_size()


def breakdown(model) -> None:
    by = defaultdict(int)
    for name, p in model.named_parameters():
        name = name.replace("_orig_mod.", "")
        top = ".".join(name.split(".")[:2]) if name.startswith("decoder") else name.split(".")[0]
        by[(top, p.device.type, type(p.data).__name__ if hasattr(p, "data") else "?")] += tensor_bytes(p)
    for (top, dev, kind), b in sorted(by.items(), key=lambda kv: -kv[1])[:12]:
        print(f"  {b / GiB:5.2f} GiB  {dev:4}  {kind:24} {top}", file=sys.stderr)


handler, _ = engine_worker.load_handler("acestep-v15-base")
print(f"after init: allocated {torch.cuda.memory_allocated() / GiB:.2f} GiB", file=sys.stderr)
breakdown(handler.model)
torch.cuda.reset_peak_memory_stats()
try:
    with handler._load_model_context("model"):
        print(f"model on GPU: allocated {torch.cuda.memory_allocated() / GiB:.2f} GiB, "
              f"peak {torch.cuda.max_memory_allocated() / GiB:.2f} GiB", file=sys.stderr)
        breakdown(handler.model)
except torch.OutOfMemoryError as exc:
    print(f"OOM: peak {torch.cuda.max_memory_allocated() / GiB:.2f} GiB — {str(exc)[:120]}", file=sys.stderr)
    breakdown(handler.model)

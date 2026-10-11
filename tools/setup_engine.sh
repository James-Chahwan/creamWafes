#!/usr/bin/env bash
# Set up the AI engine LayerDAW talks to: ACE-Step 1.5 (MIT) at the commit LayerDAW was built
# against, its own Python environment, and the model weights (~10.5 GB download, ~18 GB on disk
# with the environment). Safe to re-run: finished steps are skipped.
set -euo pipefail
cd "$(dirname "$0")/.."

ACE_COMMIT=ca1e85f   # the ACE-Step 1.5 commit LayerDAW is tested with
ENGINE=engine/ACE-Step-1.5

command -v uv >/dev/null || { echo "Needs uv: https://docs.astral.sh/uv/ (curl -LsSf https://astral.sh/uv/install.sh | sh)"; exit 1; }

if [ ! -d "$ENGINE/.git" ]; then
    git clone https://github.com/ace-step/ACE-Step-1.5 "$ENGINE"
fi
git -C "$ENGINE" fetch --quiet origin
git -C "$ENGINE" checkout --quiet "$ACE_COMMIT"

(cd "$ENGINE" && UV_HTTP_TIMEOUT=300 uv sync --python 3.12)

cd "$ENGINE"
hf() { uvx --from huggingface_hub hf download "$@"; }
[ -f checkpoints/vae/diffusion_pytorch_model.safetensors ] ||
    hf ACE-Step/Ace-Step1.5 --include "vae/*" "Qwen3-Embedding-0.6B/*" "config.json" --local-dir ./checkpoints
[ -f checkpoints/acestep-v15-base/model.safetensors ] ||
    hf ACE-Step/acestep-v15-base --local-dir ./checkpoints/acestep-v15-base
[ -f checkpoints/acestep-v15-turbo/model.safetensors ] ||
    hf ACE-Step/Ace-Step1.5 acestep-v15-turbo/config.json acestep-v15-turbo/configuration_acestep_v15.py \
        acestep-v15-turbo/model.safetensors acestep-v15-turbo/modeling_acestep_v15_turbo.py \
        acestep-v15-turbo/silence_latent.pt --local-dir ./checkpoints
du -sh checkpoints/* .venv
echo "Engine ready. Start the app: cd app && uv run layerdaw"

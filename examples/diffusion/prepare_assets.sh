#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# Download the model + reward assets needed for the Phase 1 SD1.5 GRPO PoC:
#   1. SD1.5 base pipeline          (AI-ModelScope/stable-diffusion-v1-5)
#   2. CLIP ViT-L/14 image encoder  (openai/clip-vit-large-patch14)
#   3. LAION aesthetic predictor    (sac+logos+ava1-l14-linearMSE.pth)
#
# Source selection (SD15_SOURCE):
#   - "modelscope" (default): pull SD1.5 from ModelScope via its Python SDK.
#       The original `runwayml/stable-diffusion-v1-5` HF repo was removed by
#       RunwayML, and the HF community mirror is heavily rate-limited / blocked
#       from many regions (notably mainland China, where SD1.5 large files crawl
#       at ~50KB/s). ModelScope's CDN is fast and unmetered for these weights,
#       so it is the default. Note: the ModelScope repo ships fp32 safetensors
#       only (no .fp16 variants); loading with torch_dtype=float16 still works
#       and uses fp16 memory -- it just takes 2x disk.
#   - "hf": pull SD1.5 from the HuggingFace community mirror
#       (stable-diffusion-v1-5/stable-diffusion-v1-5) via huggingface-cli.
#
# Override the exact repos via SD15_REPO / CLIP_REPO if you have your own copies.
#
# Idempotent: existing files are skipped. Nothing here requires sudo or a GPU.
#
# Usage:
#   bash examples/diffusion/prepare_assets.sh [ASSET_DIR]
#   SD15_SOURCE=hf bash examples/diffusion/prepare_assets.sh [ASSET_DIR]
#
# ASSET_DIR defaults to ./assets/diffusion. After running, train with:
#   python examples/diffusion/sd15_grpo.py \
#       --model_path  "$ASSET_DIR/stable-diffusion-v1-5" \
#       --clip_model  "$ASSET_DIR/clip-vit-large-patch14" \
#       --aesthetic_weights "$ASSET_DIR/sac+logos+ava1-l14-linearMSE.pth" \
#       --prompt_file examples/diffusion/prompts/aesthetic_prompts.txt
# (these are also the script defaults, so a bare run picks up the local copies).

set -euo pipefail

ASSET_DIR="${1:-./assets/diffusion}"
SD15_SOURCE="${SD15_SOURCE:-modelscope}"
CLIP_REPO="${CLIP_REPO:-openai/clip-vit-large-patch14}"
AESTHETIC_URL="https://github.com/christophschuhmann/improved-aesthetic-predictor/raw/main/sac+logos+ava1-l14-linearMSE.pth"
AESTHETIC_FILE="$ASSET_DIR/sac+logos+ava1-l14-linearMSE.pth"

SD15_DEST="$ASSET_DIR/stable-diffusion-v1-5"
CLIP_DEST="$ASSET_DIR/clip-vit-large-patch14"

mkdir -p "$ASSET_DIR"
echo "[prepare_assets] target dir: $ASSET_DIR"
echo "[prepare_assets] SD1.5 source: $SD15_SOURCE"

dir_has_files() {
  local d="$1"
  [ -d "$d" ] && [ -n "$(ls -A "$d" 2>/dev/null || true)" ]
}

# ---- 1: SD1.5 pipeline ----
if dir_has_files "$SD15_DEST"; then
  echo "[prepare_assets] skip SD1.5 (already at $SD15_DEST)"
elif [ "$SD15_SOURCE" = "modelscope" ]; then
  SD15_REPO="${SD15_REPO:-AI-ModelScope/stable-diffusion-v1-5}"
  if ! python -c "import modelscope" >/dev/null 2>&1; then
    echo "[prepare_assets] ERROR: modelscope SDK not found." >&2
    echo "  Install it with:  uv pip install modelscope" >&2
    echo "  (or)              pip install modelscope" >&2
    echo "  Or use the HF mirror instead:  SD15_SOURCE=hf bash $0 $ASSET_DIR" >&2
    exit 1
  fi
  echo "[prepare_assets] downloading $SD15_REPO (ModelScope) -> $SD15_DEST"
  python - "$SD15_REPO" "$SD15_DEST" <<'PY'
import sys
from modelscope import snapshot_download

repo, dest = sys.argv[1], sys.argv[2]
snapshot_download(repo, local_dir=dest)
PY
elif [ "$SD15_SOURCE" = "hf" ]; then
  SD15_REPO="${SD15_REPO:-stable-diffusion-v1-5/stable-diffusion-v1-5}"
  if ! command -v huggingface-cli >/dev/null 2>&1; then
    echo "[prepare_assets] ERROR: huggingface-cli not found." >&2
    echo "  Install it with:  uv pip install 'huggingface_hub[cli]'" >&2
    exit 1
  fi
  echo "[prepare_assets] downloading $SD15_REPO (HuggingFace) -> $SD15_DEST"
  huggingface-cli download "$SD15_REPO" --local-dir "$SD15_DEST"
else
  echo "[prepare_assets] ERROR: unknown SD15_SOURCE='$SD15_SOURCE' (use 'modelscope' or 'hf')." >&2
  exit 1
fi

# ---- 2: CLIP ViT-L/14 (HuggingFace) ----
if dir_has_files "$CLIP_DEST"; then
  echo "[prepare_assets] skip CLIP (already at $CLIP_DEST)"
else
  if ! command -v huggingface-cli >/dev/null 2>&1; then
    echo "[prepare_assets] ERROR: huggingface-cli not found (needed for CLIP)." >&2
    echo "  Install it with:  uv pip install 'huggingface_hub[cli]'" >&2
    exit 1
  fi
  echo "[prepare_assets] downloading $CLIP_REPO (HuggingFace) -> $CLIP_DEST"
  huggingface-cli download "$CLIP_REPO" --local-dir "$CLIP_DEST"
fi

# ---- 3: LAION aesthetic predictor head ----
if [ -f "$AESTHETIC_FILE" ]; then
  echo "[prepare_assets] skip aesthetic weights (already at $AESTHETIC_FILE)"
else
  echo "[prepare_assets] downloading aesthetic predictor -> $AESTHETIC_FILE"
  if command -v curl >/dev/null 2>&1; then
    curl -fL "$AESTHETIC_URL" -o "$AESTHETIC_FILE"
  elif command -v wget >/dev/null 2>&1; then
    wget -O "$AESTHETIC_FILE" "$AESTHETIC_URL"
  else
    echo "[prepare_assets] ERROR: neither curl nor wget found for the .pth download." >&2
    echo "  Manually fetch: $AESTHETIC_URL" >&2
    echo "  and save it to: $AESTHETIC_FILE" >&2
    exit 1
  fi
fi

echo "[prepare_assets] done. Assets ready under: $ASSET_DIR"
echo
echo "Next (these paths match the script defaults, so flags are optional):"
echo "  python examples/diffusion/sd15_grpo.py \\"
echo "      --model_path  \"$SD15_DEST\" \\"
echo "      --clip_model  \"$CLIP_DEST\" \\"
echo "      --aesthetic_weights \"$AESTHETIC_FILE\" \\"
echo "      --prompt_file examples/diffusion/prompts/aesthetic_prompts.txt"

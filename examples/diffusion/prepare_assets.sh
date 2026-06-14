#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# Download the model + reward assets needed for the Phase 1 SD1.5 GRPO PoC:
#   1. SD1.5 base pipeline          (stable-diffusion-v1-5/stable-diffusion-v1-5)
#   2. CLIP ViT-L/14 image encoder  (openai/clip-vit-large-patch14)
#   3. LAION aesthetic predictor    (sac+logos+ava1-l14-linearMSE.pth)
#
# NOTE: the original `runwayml/stable-diffusion-v1-5` repo was removed from the
# Hub by RunwayML. The community-maintained mirror
# `stable-diffusion-v1-5/stable-diffusion-v1-5` is byte-identical and is what we
# download here. Override via the SD15_REPO env var if you have your own copy.
#
# Idempotent: existing files are skipped. Nothing here requires sudo or a GPU.
#
# Usage:
#   bash examples/diffusion/prepare_assets.sh [ASSET_DIR]
#
# ASSET_DIR defaults to ./assets/diffusion. After running, train with:
#   python examples/diffusion/sd15_grpo.py \
#       --model_path  "$ASSET_DIR/stable-diffusion-v1-5" \
#       --clip_model  "$ASSET_DIR/clip-vit-large-patch14" \
#       --aesthetic_weights "$ASSET_DIR/sac+logos+ava1-l14-linearMSE.pth" \
#       --prompt_file examples/diffusion/prompts/aesthetic_prompts.txt
#
# You can also skip the HF snapshot downloads and let diffusers/transformers
# pull the repos by their hub IDs at runtime; the local copies just make runs
# offline-friendly and reproducible.

set -euo pipefail

ASSET_DIR="${1:-./assets/diffusion}"
SD15_REPO="${SD15_REPO:-stable-diffusion-v1-5/stable-diffusion-v1-5}"
CLIP_REPO="${CLIP_REPO:-openai/clip-vit-large-patch14}"
AESTHETIC_URL="https://github.com/christophschuhmann/improved-aesthetic-predictor/raw/main/sac+logos+ava1-l14-linearMSE.pth"
AESTHETIC_FILE="$ASSET_DIR/sac+logos+ava1-l14-linearMSE.pth"

mkdir -p "$ASSET_DIR"
echo "[prepare_assets] target dir: $ASSET_DIR"

# ---- 1 & 2: HuggingFace snapshots (SD1.5 + CLIP) ----
if ! command -v huggingface-cli >/dev/null 2>&1; then
  echo "[prepare_assets] ERROR: huggingface-cli not found." >&2
  echo "  Install it with:  uv pip install 'huggingface_hub[cli]'" >&2
  echo "  (or pip install)  pip install 'huggingface_hub[cli]'" >&2
  exit 1
fi

download_snapshot() {
  local repo="$1" dest="$2"
  if [ -d "$dest" ] && [ -n "$(ls -A "$dest" 2>/dev/null || true)" ]; then
    echo "[prepare_assets] skip $repo (already at $dest)"
  else
    echo "[prepare_assets] downloading $repo -> $dest"
    huggingface-cli download "$repo" --local-dir "$dest"
  fi
}

download_snapshot "$SD15_REPO" "$ASSET_DIR/stable-diffusion-v1-5"
download_snapshot "$CLIP_REPO" "$ASSET_DIR/clip-vit-large-patch14"

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
echo "Next:"
echo "  python examples/diffusion/sd15_grpo.py \\"
echo "      --model_path  \"$ASSET_DIR/stable-diffusion-v1-5\" \\"
echo "      --clip_model  \"$ASSET_DIR/clip-vit-large-patch14\" \\"
echo "      --aesthetic_weights \"$AESTHETIC_FILE\" \\"
echo "      --prompt_file examples/diffusion/prompts/aesthetic_prompts.txt"

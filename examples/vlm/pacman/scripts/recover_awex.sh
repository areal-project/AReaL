#!/usr/bin/env bash
# GPU environment only; not run locally. Run one update, restart, run the next.
# Keep the same artifact root, experiment/trial names and topology across launches.
set -euo pipefail
: "${AREAL_ROOT:?Export the current AReaL checkout}"
: "${PACMAN_ARTIFACT_ROOT:?Use a fresh absolute output directory}"
cd "$AREAL_ROOT"
mkdir -p "$PACMAN_ARTIFACT_ROOT"

check_snapshot() {
    python - "$PACMAN_ARTIFACT_ROOT/curriculum1/training" "$1" <<'PY'
import json
import sys
from pathlib import Path

root, expected = Path(sys.argv[1]), int(sys.argv[2])
infos = list(root.rglob("recover_info/step_info.json"))
if expected == -1:
    if infos:
        raise RuntimeError("Use a fresh artifact root for this recovery check")
else:
    if len(infos) != 1:
        raise RuntimeError(f"Expected one recovery snapshot, found {infos}")
    info = json.loads(infos[0].read_text())
    if info["global_step"] != expected:
        raise RuntimeError(f"Unexpected saved step: {info}")
    checkpoint = infos[0].parent.parent / "default" / "recover_checkpoint"
    if not checkpoint.is_dir() or not any(checkpoint.iterdir()):
        raise RuntimeError(f"Missing DCP checkpoint: {checkpoint}")
    print(f"Recovery snapshot validated: {infos[0]}, global_step={expected}")
PY
}

check_snapshot -1
bash examples/vlm/pacman/scripts/train_c1_awex.sh "$@" \
    recover.mode=auto recover.no_save_optim=false recover.no_load_optim=false \
    actor.megatron.async_save=false total_train_steps=1
check_snapshot 0
bash examples/vlm/pacman/scripts/train_c1_awex.sh "$@" \
    recover.mode=auto recover.no_save_optim=false recover.no_load_optim=false \
    actor.megatron.async_save=false total_train_steps=2 \
    2>&1 | tee "$PACMAN_ARTIFACT_ROOT/recovery-resumed.log"
grep -F 'Recovering from ' "$PACMAN_ARTIFACT_ROOT/recovery-resumed.log"
check_snapshot 1

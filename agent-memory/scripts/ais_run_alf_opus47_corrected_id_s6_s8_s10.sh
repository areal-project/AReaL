#!/bin/bash
# Corrected ID-only validation for original Opus S6/S8/S10, one run each.
# Loading each old snapshot triggers canonical hard-membership + summary rebuild.
set -euo pipefail

MEMRL_DIR=/storage/openpsi/users/yl/agent-memory/MemRL
SNAP_ROOT=/storage/openpsi/experiments/checkpoints/admin/yl-mem-region/alfworld/alfworld/exp_alfworld_region_opus47_20260623-100806/local_cache/snapshot
RUN_TAG=${RUN_TAG:-$(date +%Y%m%d-%H%M%S)}
LOGFILE="$MEMRL_DIR/logs/aistudio_alf_opus47_corrected_id_s6_s8_s10_${RUN_TAG}.log"
exec > >(tee -a "$LOGFILE") 2>&1
cd "$MEMRL_DIR"

VENV_SP=/AReaL/.venv/lib/python3.12/site-packages
pip install -e . --no-deps --target "$VENV_SP" -i https://pypi.antfin-inc.com/simple/ 2>&1 | tail -2
pip install mem0ai "chonkie==1.2.1" tensorboard pandas tqdm hdbscan concurrent-log-handler textworld alfworld \
  --target "$VENV_SP" -i https://pypi.antfin-inc.com/simple/ 2>&1 | tail -5 || true

export PYTHONPATH=/storage/openpsi/users/yl/agent-memory/.local/lib/python3.12/site-packages:${PYTHONPATH:-}
export PYTHONDONTWRITEBYTECODE=1
export HF_HOME=/storage/openpsi/users/yl/agent-memory/.cache/huggingface
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export TMPDIR=/dev/shm/alf_opus47_corrected_id TEMP=/dev/shm/alf_opus47_corrected_id TMP=/dev/shm/alf_opus47_corrected_id
mkdir -p "$TMPDIR"
export MEMRL_LLM_MIN_INTERVAL=1.0 MEMRL_EMBED_MIN_INTERVAL=1.5 MEMRL_EMBED_THROTTLE=0.5
export MEMRL_UPDATE_MAX_WORKERS=2 MEMRL_ALFWORLD_LLM_CONCURRENCY=16

for S in 6 8 10; do
  SNAP="$SNAP_ROOT/$S"
  [[ -f "$SNAP/local_cache/region_manager.json" ]] || { echo "[FATAL] missing $SNAP"; exit 1; }
  CFG="$TMPDIR/opus47_corrected_id_s${S}.yaml"
  python3 - "$CFG" "$SNAP" "$S" "$RUN_TAG" <<'PY'
import sys, yaml
from pathlib import Path
out, snap, sec, tag = sys.argv[1:]
cfg = yaml.safe_load(Path('configs/rl_alf_config.opus47_region.yaml').read_text())
cfg['memory']['build_strategy'] = 'trajectory' if int(sec) >= 10 else cfg['memory']['build_strategy']
cfg['memory']['load_from_checkpoint'] = True
cfg['memory']['checkpoint_path'] = snap
cfg['experiment']['experiment_name'] = f'alfworld_region_opus47_corrected_id_s{sec}_{tag}'
cfg['experiment']['mode'] = 'test'
cfg['experiment']['n_eval_runs'] = 1
cfg['experiment']['ckpt_resume_enabled'] = False
cfg['experiment']['ckpt_resume_path'] = ''
cfg['experiment']['ckpt_resume_epoch'] = None
cfg['experiment']['save_memories'] = False
cfg['experiment']['save_trajectories'] = True
Path(out).write_text(yaml.safe_dump(cfg, sort_keys=False))
PY
  export MEMRL_RUN_ID="corrected-id-s${S}-${RUN_TAG}"
  echo "============================================================"
  echo "[EVAL] Corrected ID-only S${S}, one deterministic run"
  echo "[EVAL] canonical hard memberships and summaries rebuild on snapshot load"
  echo "============================================================"
  python3 run/run_alfworld.py \
    --config "$CFG" --region --region_gating_mode additive \
    --shrinkage_confidence_k 3.0 --propagation_eta 0.12 --val_lambda_max 0.15 \
    --failure_summary_n_slots 1 --id_eval_only
  echo "[EVAL] S${S} completed at $(date)"
done

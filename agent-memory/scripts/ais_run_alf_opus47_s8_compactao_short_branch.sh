#!/bin/bash
# Isolated one-epoch continuation from the original Opus Region+FS S8 checkpoint.
# The original experiment directory is read-only input; all new snapshots go to a new run.
set -euo pipefail

MEMRL_DIR=/storage/openpsi/users/yl/agent-memory/MemRL
SOURCE_SNAPSHOT_ROOT=/storage/openpsi/experiments/checkpoints/admin/yl-mem-region/alfworld/alfworld/exp_alfworld_region_opus47_s8_compactao_v2_20260722/local_cache/snapshot
SOURCE_SNAPSHOT="$SOURCE_SNAPSHOT_ROOT/8"
MATRIX_CREDENTIAL_CONFIG=/storage/openpsi/users/yl/cfuse/rq3_v11_clean_20260719/config_multisurface_isolated.yaml
RUN_TAG=${RUN_TAG:-$(date +%Y%m%d-%H%M%S)}
BRANCH_NAME=alfworld_region_opus47_s8_compactao_short
LOGFILE="$MEMRL_DIR/logs/aistudio_alf_opus47_s8_compactao_short_${RUN_TAG}.log"
exec > >(tee -a "$LOGFILE") 2>&1

cd "$MEMRL_DIR"
[[ -f "$SOURCE_SNAPSHOT/local_cache/region_manager.json" ]] || { echo "[FATAL] missing $SOURCE_SNAPSHOT"; exit 1; }
[[ -f "$MATRIX_CREDENTIAL_CONFIG" ]] || { echo "[FATAL] missing Matrix credential config"; exit 1; }

VENV_SP=/AReaL/.venv/lib/python3.12/site-packages
pip install -e . --no-deps --target "$VENV_SP" -i https://pypi.antfin-inc.com/simple/ 2>&1 | tail -2
pip install mem0ai "chonkie==1.2.1" tensorboard pandas tqdm hdbscan concurrent-log-handler textworld alfworld \
  --target "$VENV_SP" -i https://pypi.antfin-inc.com/simple/ 2>&1 | tail -5 || true

export PYTHONPATH=/storage/openpsi/users/yl/agent-memory/.local/lib/python3.12/site-packages:${PYTHONPATH:-}
export PYTHONDONTWRITEBYTECODE=1
export HF_HOME=/storage/openpsi/users/yl/agent-memory/.cache/huggingface
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export TMPDIR=/dev/shm/alf_opus47_s8_compactao_short TEMP=/dev/shm/alf_opus47_s8_compactao_short TMP=/dev/shm/alf_opus47_s8_compactao_short
mkdir -p "$TMPDIR"
export MEMRL_LLM_MIN_INTERVAL=1.0 MEMRL_EMBED_MIN_INTERVAL=1.5 MEMRL_EMBED_THROTTLE=0.5
export MEMRL_UPDATE_MAX_WORKERS=2 MEMRL_ALFWORLD_LLM_CONCURRENCY=16
# Unique run ID: never auto-resume into or write under the original experiment.
export MEMRL_RUN_ID="s8-compactao-short-${RUN_TAG}"

CFG="$TMPDIR/opus47_s8_short.yaml"
trap 'rm -f "$CFG"' EXIT
python3 - "$CFG" "$SOURCE_SNAPSHOT_ROOT" "$BRANCH_NAME" "$MATRIX_CREDENTIAL_CONFIG" <<'PY'
import os
import sys
from pathlib import Path

import yaml

out, snap_root, name, credential_config = sys.argv[1:]
cfg = yaml.safe_load(Path('configs/rl_alf_config.opus47_region.yaml').read_text())
credential_map = yaml.safe_load(Path(credential_config).read_text())

def credential(model):
    for item in credential_map.get('model_list', []):
        if item.get('model_name') != model:
            continue
        params = item.get('litellm_params') or {}
        value = params.get('api_key')
        if isinstance(value, str) and value.startswith('os.environ/'):
            value = os.environ.get(value.split('/', 1)[1])
        if not value:
            raise RuntimeError(f'Matrix credential for {model!r} is unavailable')
        return value, (params.get('api_base') or 'https://matrixllm.alipay.com/v1/')
    raise RuntimeError(f'No Matrix credential mapping for {model!r}')

llm_key, llm_base = credential('claude-opus-4-7')
embed_key, embed_base = credential('text-embedding-3-large')
cfg['llm']['api_key'] = llm_key
cfg['llm']['base_url'] = llm_base
cfg['embedding']['api_key'] = embed_key
cfg['embedding']['base_url'] = embed_base
cfg['memory']['build_strategy'] = 'trajectory'
cfg['experiment']['experiment_name'] = name
cfg['experiment']['num_sections'] = 9       # resume S8, run one isolated continuation section
cfg['experiment']['dataset_ratio'] = float(os.environ.get('DATASET_RATIO', '0.10'))  # short smoke/diagnostic branch
cfg['experiment']['valid_interval'] = 1
cfg['experiment']['test_interval'] = 0      # no OOD access during tuning
cfg['experiment']['ckpt_resume_enabled'] = True
cfg['experiment']['ckpt_resume_path'] = snap_root
cfg['experiment']['ckpt_resume_epoch'] = 8
Path(out).write_text(yaml.safe_dump(cfg, sort_keys=False))
os.chmod(out, 0o600)
PY

echo "[INFO] Matrix credentials loaded from verified config (values not logged)"
echo "[INFO] New isolated branch: $BRANCH_NAME / $MEMRL_RUN_ID"
echo "[INFO] Source snapshot (read only): $SOURCE_SNAPSHOT"
echo "[INFO] Short continuation: DATASET_RATIO=${DATASET_RATIO:-0.10}, compact action/observation trajectory, FS=1, concurrency=16"
echo "[INFO] Tuning: propagation_eta=0.06; source is compact-S8 v2; OOD/test disabled"
python3 run/run_alfworld.py \
  --config "$CFG" --region --region_gating_mode additive \
  --shrinkage_confidence_k 3.0 --propagation_eta 0.06 --val_lambda_max 0.10 \
  --failure_summary_n_slots 1 --skip_initial_eval

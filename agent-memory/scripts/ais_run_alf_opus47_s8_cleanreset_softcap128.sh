#!/bin/bash
# Single clean S8 continuation: reset legacy aggregate evidence, accumulate only
# new source-conserving evidence, freeze topology, and cap each sim-key bucket.
set -euo pipefail

MEMRL_DIR=/storage/openpsi/users/yl/agent-memory/MemRL
SOURCE_SNAPSHOT_ROOT=/storage/openpsi/experiments/checkpoints/admin/yl-mem-region/alfworld/alfworld/exp_alfworld_region_opus47_s8_compactao_v2_20260722/local_cache/snapshot
SOURCE_SNAPSHOT="$SOURCE_SNAPSHOT_ROOT/8"
# Optional explicit resume root (the directory containing local_cache/snapshot).
# Without it, discover the newest complete s9_bN checkpoint from a prior clean-reset run.
RESUME_ROOT=${RESUME_ROOT:-}
MATRIX_CREDENTIAL_CONFIG=/storage/openpsi/users/yl/cfuse/rq3_v11_clean_20260719/config_multisurface_isolated.yaml
RUN_TAG=${RUN_TAG:-$(date +%Y%m%d-%H%M%S)}
BRANCH_NAME=alfworld_region_opus47_s8_cleanreset_softsource_cap128
RUN_ID="s8-cleanreset-softsource-cap128-${RUN_TAG}"
LOGFILE="$MEMRL_DIR/logs/aistudio_alf_opus47_${RUN_ID}.log"
TMPROOT="/dev/shm/alf_opus47_${RUN_ID}"

exec > >(tee -a "$LOGFILE") 2>&1
cd "$MEMRL_DIR"
[[ -f "$SOURCE_SNAPSHOT/local_cache/region_manager.json" ]] || { echo "[FATAL] missing $SOURCE_SNAPSHOT"; exit 1; }
[[ -f "$MATRIX_CREDENTIAL_CONFIG" ]] || { echo "[FATAL] missing credential config"; exit 1; }

# Select a consistent snapshot/cum_state pair.  A batch snapshot includes its
# own local_cache/cum_state.json, so the runner will resume at N+1 exactly.
if [[ -z "$RESUME_ROOT" ]]; then
  latest_batch=$(find /storage/openpsi/experiments/checkpoints/admin/yl-mem-region/alfworld/alfworld \
    -type d -path '*exp_alfworld_region_opus47_s8_cleanreset_softsource_cap128_*/local_cache/snapshot/s9_b*' \
    -printf '%T@ %p\n' 2>/dev/null | sort -nr | head -1 | awk '{print $2}')
  if [[ -n "${latest_batch:-}" && -f "$latest_batch/local_cache/cum_state.json" ]]; then
    RESUME_ROOT="$(dirname "$latest_batch")"
  fi
fi
if [[ -n "$RESUME_ROOT" ]]; then
  [[ -d "$RESUME_ROOT" ]] || { echo "[FATAL] RESUME_ROOT does not exist: $RESUME_ROOT"; exit 1; }
  RESUME_KIND=batch
  RESUME_PATH="$RESUME_ROOT"
  RESUME_EPOCH=
  RESET_LEGACY_EVIDENCE=false
else
  RESUME_KIND=legacy_s8
  RESUME_PATH="$SOURCE_SNAPSHOT_ROOT"
  RESUME_EPOCH=8
  RESET_LEGACY_EVIDENCE=true
fi

VENV_SP=/AReaL/.venv/lib/python3.12/site-packages
pip install -e . --no-deps --target "$VENV_SP" -i https://pypi.antfin-inc.com/simple/ 2>&1 | tail -2
pip install mem0ai 'chonkie==1.2.1' tensorboard pandas tqdm hdbscan concurrent-log-handler textworld alfworld \
  --target "$VENV_SP" -i https://pypi.antfin-inc.com/simple/ 2>&1 | tail -5 || true

export PYTHONPATH=/storage/openpsi/users/yl/agent-memory/.local/lib/python3.12/site-packages:${PYTHONPATH:-}
export PYTHONDONTWRITEBYTECODE=1
export HF_HOME=/storage/openpsi/users/yl/agent-memory/.cache/huggingface
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export TMPDIR="$TMPROOT" TEMP="$TMPROOT" TMP="$TMPROOT"
mkdir -p "$TMPROOT"
# One job owns the shared queue; still retain a cross-process key for all MemOS and MemRL paths.
export MEMRL_EMBED_RATE_LIMIT_KEY=opus47-s8-cleanreset-cap128
export MEMRL_EMBED_GLOBAL_MIN_INTERVAL=2.5 MEMRL_EMBED_MIN_INTERVAL=2.5 MEMRL_EMBED_THROTTLE=2.5
export MEMRL_EMBED_429_BASE_DELAY=15 MEMRL_EMBED_429_MAX_DELAY=180
export MEMRL_UPDATE_MAX_WORKERS=1 MEMRL_ALFWORLD_LLM_CONCURRENCY=4
export MEMRL_LLM_MIN_INTERVAL=2.0 MEMRL_LLM_MAX_RETRIES=5 MEMRL_LLM_TENACITY_ATTEMPTS=5
export MEMRL_RUN_ID="$RUN_ID"

CFG="$TMPROOT/config.yaml"
trap 'rm -f "$CFG"' EXIT
python3 - "$CFG" "$RESUME_PATH" "$RESUME_EPOCH" "$BRANCH_NAME" "$MATRIX_CREDENTIAL_CONFIG" <<'PY'
import os, sys
from pathlib import Path
import yaml
out, snap_root, resume_epoch, name, credential_config = sys.argv[1:]
cfg = yaml.safe_load(Path('configs/rl_alf_config.opus47_region.yaml').read_text())
creds = yaml.safe_load(Path(credential_config).read_text())
def credential(model):
    for item in creds.get('model_list', []):
        if item.get('model_name') == model:
            params = item.get('litellm_params') or {}
            value = params.get('api_key')
            if isinstance(value, str) and value.startswith('os.environ/'):
                value = os.environ.get(value.split('/', 1)[1])
            if not value:
                raise RuntimeError(f'Matrix credential for {model!r} unavailable')
            return value, params.get('api_base') or 'https://matrixllm.alipay.com/v1/'
    raise RuntimeError(f'No credential mapping for {model!r}')
llm_key, llm_base = credential('claude-opus-4-7')
embed_key, embed_base = credential('text-embedding-3-large')
cfg['llm']['api_key'], cfg['llm']['base_url'] = llm_key, llm_base
cfg['embedding']['api_key'], cfg['embedding']['base_url'] = embed_key, embed_base
cfg['memory']['build_strategy'] = 'trajectory'
cfg['experiment']['experiment_name'] = name
cfg['experiment']['num_sections'] = 9
cfg['experiment']['dataset_ratio'] = float(os.environ.get('DATASET_RATIO', '1.0'))
cfg['experiment']['valid_interval'] = 1
cfg['experiment']['test_interval'] = 0
cfg['experiment']['ckpt_resume_enabled'] = True
cfg['experiment']['ckpt_resume_path'] = snap_root
if resume_epoch:
    cfg['experiment']['ckpt_resume_epoch'] = int(resume_epoch)
else:
    cfg['experiment'].pop('ckpt_resume_epoch', None)
Path(out).write_text(yaml.safe_dump(cfg, sort_keys=False))
os.chmod(out, 0o600)
PY

echo "[INFO] resume kind=$RESUME_KIND path=$RESUME_PATH epoch=${RESUME_EPOCH:-<latest batch>}"
if [[ "$RESET_LEGACY_EVIDENCE" == true ]]; then
  echo "[INFO] first legacy-S8 start: retain Q/geometry/membership/prior; reset aggregate observed evidence once"
  RESET_ARGS=(--region_reset_legacy_evidence_on_resume)
else
  echo "[INFO] batch resume: preserve already accumulated source-conserving evidence; DO NOT reset it"
  RESET_ARGS=()
fi
echo "[INFO] soft source-conserving, topology frozen, cap=128 candidates/sim-key, full S9 train + ID validation"
python3 run/run_alfworld.py --config "$CFG" --region --region_gating_mode additive \
  --region_split_evidence_migration_mode soft_source_conserving \
  --region_freeze_topology "${RESET_ARGS[@]}" \
  --max_candidates_per_sim_key 128 \
  --shrinkage_confidence_k 3.0 --propagation_eta 0.06 --val_lambda_max 0.10 \
  --failure_summary_n_slots 1 --skip_initial_eval

#!/bin/bash
# Two isolated Opus S8 continuations in one Ais job. The source S8 snapshot is read-only.
set -euo pipefail

MEMRL_DIR=/storage/openpsi/users/yl/agent-memory/MemRL
SOURCE_SNAPSHOT_ROOT=/storage/openpsi/experiments/checkpoints/admin/yl-mem-region/alfworld/alfworld/exp_alfworld_region_opus47_s8_compactao_v2_20260722/local_cache/snapshot
SOURCE_SNAPSHOT="$SOURCE_SNAPSHOT_ROOT/8"
MATRIX_CREDENTIAL_CONFIG=/storage/openpsi/users/yl/cfuse/rq3_v11_clean_20260719/config_multisurface_isolated.yaml
RUN_TAG=${RUN_TAG:-$(date +%Y%m%d-%H%M%S)}
BASE_TMP=/dev/shm/alf_opus47_s8_splitfix_dual_25s
mkdir -p "$BASE_TMP"

cd "$MEMRL_DIR"
[[ -f "$SOURCE_SNAPSHOT/local_cache/region_manager.json" ]] || { echo "[FATAL] missing $SOURCE_SNAPSHOT"; exit 1; }
[[ -f "$MATRIX_CREDENTIAL_CONFIG" ]] || { echo "[FATAL] missing credential config"; exit 1; }

VENV_SP=/AReaL/.venv/lib/python3.12/site-packages
pip install -e . --no-deps --target "$VENV_SP" -i https://pypi.antfin-inc.com/simple/ 2>&1 | tail -2
pip install mem0ai 'chonkie==1.2.1' tensorboard pandas tqdm hdbscan concurrent-log-handler textworld alfworld \
  --target "$VENV_SP" -i https://pypi.antfin-inc.com/simple/ 2>&1 | tail -5 || true

export PYTHONPATH=/storage/openpsi/users/yl/agent-memory/.local/lib/python3.12/site-packages:${PYTHONPATH:-}
export PYTHONDONTWRITEBYTECODE=1
export HF_HOME=/storage/openpsi/users/yl/agent-memory/.cache/huggingface
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
# Both children share this file-backed limiter and its 429 cooldowns.
export MEMRL_EMBED_RATE_LIMIT_KEY=opus47-s8-splitfix-dual-25s
export MEMRL_EMBED_GLOBAL_MIN_INTERVAL=2.5
export MEMRL_EMBED_MIN_INTERVAL=2.5
export MEMRL_EMBED_THROTTLE=2.5
export MEMRL_EMBED_429_BASE_DELAY=15
export MEMRL_EMBED_429_MAX_DELAY=180
export MEMRL_UPDATE_MAX_WORKERS=1
export MEMRL_ALFWORLD_LLM_CONCURRENCY=4
export MEMRL_LLM_MIN_INTERVAL=2.0
export MEMRL_LLM_MAX_RETRIES=5
export MEMRL_LLM_TENACITY_ATTEMPTS=5

make_config() {
  local out="$1" name="$2" migration_mode="$3" topology_enabled="$4"
  python3 - "$out" "$SOURCE_SNAPSHOT_ROOT" "$name" "$MATRIX_CREDENTIAL_CONFIG" "$migration_mode" "$topology_enabled" <<'PY'
import os, sys
from pathlib import Path
import yaml
out, snap_root, name, credential_config, migration_mode, topology_enabled = sys.argv[1:]
cfg = yaml.safe_load(Path('configs/rl_alf_config.opus47_region.yaml').read_text())
credential_map = yaml.safe_load(Path(credential_config).read_text())
def credential(model):
    for item in credential_map.get('model_list', []):
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
cfg['experiment']['ckpt_resume_epoch'] = 8
Path(out).write_text(yaml.safe_dump(cfg, sort_keys=False))
os.chmod(out, 0o600)
PY
}

HARD_TAG="s8-splitfix-hard-rebase-25s-${RUN_TAG}"
SOFT_TAG="s8-splitfix-softsource-warmup-25s-${RUN_TAG}"
HARD_CFG="$BASE_TMP/${HARD_TAG}.yaml"
SOFT_CFG="$BASE_TMP/${SOFT_TAG}.yaml"
trap 'rm -f "$HARD_CFG" "$SOFT_CFG"' EXIT
make_config "$HARD_CFG" alfworld_region_opus47_s8_splitfix_hard_rebase_25s hard_member_rebase true
make_config "$SOFT_CFG" alfworld_region_opus47_s8_splitfix_softsource_warmup_25s soft_source_conserving false

run_branch() {
  local tag="$1" cfg="$2" mode="$3" topology="$4"
  local log="$MEMRL_DIR/logs/aistudio_alf_opus47_${tag}.log"
  (
    export MEMRL_RUN_ID="$tag"
    export TMPDIR="$BASE_TMP/$tag" TEMP="$BASE_TMP/$tag" TMP="$BASE_TMP/$tag"
    mkdir -p "$TMPDIR"
    echo "[INFO] branch=$tag mode=$mode topology_updates=$topology source=$SOURCE_SNAPSHOT"
    extra_args=()
    if [[ "$topology" == "false" ]]; then
      extra_args+=(--region_freeze_topology)
    fi
    python3 run/run_alfworld.py --config "$cfg" --region --region_gating_mode additive \
      --region_split_evidence_migration_mode "$mode" "${extra_args[@]}" \
      --shrinkage_confidence_k 3.0 --propagation_eta 0.06 --val_lambda_max 0.10 \
      --failure_summary_n_slots 1 --skip_initial_eval
  ) > >(tee -a "$log") 2>&1
}

echo "[INFO] starting hard-rebase branch first; both branches use one shared 2.5s embedding slot queue"
run_branch "$HARD_TAG" "$HARD_CFG" hard_member_rebase true &
HARD_PID=$!
sleep 90
echo "[INFO] starting soft-source warm-up branch after 90s; topology is frozen only for this S8->S9 warm-up"
run_branch "$SOFT_TAG" "$SOFT_CFG" soft_source_conserving false &
SOFT_PID=$!

set +e
wait "$HARD_PID"; HARD_STATUS=$?
wait "$SOFT_PID"; SOFT_STATUS=$?
set -e
echo "[INFO] dual branch complete: hard=$HARD_STATUS soft_warmup=$SOFT_STATUS"
exit $(( HARD_STATUS != 0 || SOFT_STATUS != 0 ))

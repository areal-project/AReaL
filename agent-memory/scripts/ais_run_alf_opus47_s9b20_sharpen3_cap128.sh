#!/bin/bash
# Matched ablation fork from clean-reset S9 batch-20 checkpoint.
# Only change vs source branch: online top-3 region evidence sharpening alpha 2 -> 3.
set -euo pipefail

MEMRL_DIR=/storage/openpsi/users/yl/agent-memory/MemRL
# This continuation is intentionally pinned to the pure sharpen=3 lineage.
# Do not auto-discover or fall back to the baseline snapshot: doing so can
# silently create a w2 -> w3 late-switch run instead of a pure w3 continuation.
RESUME_ROOT=/storage/openpsi/experiments/checkpoints/admin/yl-mem-region/alfworld/alfworld/exp_alfworld_region_opus47_s9b20_softsource_sharpen3_cap128_s9b20-softsource-sharpen3-cap128-20260724-022104/local_cache/snapshot
EXPECTED_BATCH="$RESUME_ROOT/s9_b30"
MATRIX_CREDENTIAL_CONFIG=/storage/openpsi/users/yl/cfuse/rq3_v11_clean_20260719/config_multisurface_isolated.yaml
RUN_TAG=${RUN_TAG:-$(date +%Y%m%d-%H%M%S)}
BRANCH_NAME=${BRANCH_NAME:-alfworld_region_opus47_s9b20_softsource_sharpen3_cap128}
RUN_ID="s9b20-softsource-sharpen3-cap128-${RUN_TAG}"
LOGFILE="$MEMRL_DIR/logs/aistudio_alf_opus47_${RUN_ID}.log"
TMPROOT="/dev/shm/alf_opus47_${RUN_ID}"
exec > >(tee -a "$LOGFILE") 2>&1

cd "$MEMRL_DIR"
[[ -d "$RESUME_ROOT" ]] || { echo "[FATAL] resume root missing: $RESUME_ROOT"; exit 1; }
[[ -d "$EXPECTED_BATCH" ]] || { echo "[FATAL] expected pure-w3 checkpoint missing: $EXPECTED_BATCH"; exit 1; }
[[ -f "$MATRIX_CREDENTIAL_CONFIG" ]] || { echo "[FATAL] missing credential config"; exit 1; }
python3 - "$EXPECTED_BATCH" <<'PY_PREFLIGHT'
import json, sys
from pathlib import Path
p = Path(sys.argv[1])
meta = json.loads((p / 'snapshot_meta.json').read_text())
region = json.loads((p / 'local_cache' / 'region_manager.json').read_text())
required = [
    p / 'local_cache' / 'cum_state.json',
    p / 'local_cache' / 'region_manager.json',
    p / 'cube' / 'textual_memory.json',
    p / 'qdrant' / 'meta.json',
]
missing = [str(x) for x in required if not x.is_file() or x.stat().st_size <= 0]
if missing:
    raise SystemExit(f'incomplete pure-w3 checkpoint: {missing}')
if meta.get('checkpoint_id') != 's9_b30':
    raise SystemExit(f"unexpected checkpoint_id: {meta.get('checkpoint_id')!r}")
if float(region.get('region_evidence_sharpen_alpha', -1)) != 3.0:
    raise SystemExit('pure-w3 checkpoint does not have sharpen alpha=3.0')
if not region.get('has_complete_region_source_evidence_ledger'):
    raise SystemExit('pure-w3 checkpoint lacks complete region source ledger')
if region.get('region_topology_updates_enabled') is not False:
    raise SystemExit('pure-w3 checkpoint topology is not frozen')
print(f"[PREFLIGHT] pure-w3 checkpoint verified: {p}")
print('[PREFLIGHT] checkpoint_id=s9_b30 alpha=3.0 source_ledger=complete topology=frozen')
PY_PREFLIGHT

VENV_SP=/AReaL/.venv/lib/python3.12/site-packages
pip install -e . --no-deps --target "$VENV_SP" -i https://pypi.antfin-inc.com/simple/ 2>&1 | tail -2
pip install mem0ai 'chonkie==1.2.1' tensorboard pandas tqdm hdbscan concurrent-log-handler textworld alfworld \
  --target "$VENV_SP" -i https://pypi.antfin-inc.com/simple/ 2>&1 | tail -5 || true

export PYTHONPATH=/storage/openpsi/users/yl/agent-memory/.local/lib/python3.12/site-packages:${PYTHONPATH:-}
export PYTHONDONTWRITEBYTECODE=1 HF_HOME=/storage/openpsi/users/yl/agent-memory/.cache/huggingface
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export TMPDIR="$TMPROOT" TEMP="$TMPROOT" TMP="$TMPROOT"; mkdir -p "$TMPROOT"
# Separate job, shared API key: a conservative independent limiter key avoids accidental
# coupling with the continuing baseline while keeping all MemRL/MemOS calls serialized.
export MEMRL_EMBED_RATE_LIMIT_KEY=opus47-s8-cleanreset-cap128
export MEMRL_EMBED_GLOBAL_MIN_INTERVAL=2.5 MEMRL_EMBED_MIN_INTERVAL=2.5 MEMRL_EMBED_THROTTLE=2.5
export MEMRL_EMBED_429_BASE_DELAY=15 MEMRL_EMBED_429_MAX_DELAY=180
export MEMRL_UPDATE_MAX_WORKERS=1 MEMRL_ALFWORLD_LLM_CONCURRENCY=4
export MEMRL_LLM_MIN_INTERVAL=2.0 MEMRL_LLM_MAX_RETRIES=5 MEMRL_LLM_TENACITY_ATTEMPTS=5
export MEMRL_RUN_ID="$RUN_ID"
# Controlled B31-B40 short-window experiment. Deferred repair is the only
# behavioral change in this first cell; later cells reuse the same source/window.
export MEMRL_ALFWORLD_STOP_AFTER_BATCH=${MEMRL_ALFWORLD_STOP_AFTER_BATCH:-40}
export MEMRL_ALFWORLD_DEFERRED_REPAIR=${MEMRL_ALFWORLD_DEFERRED_REPAIR:-1}
export MEMRL_ALFWORLD_DEFERRED_REPAIR_COOLDOWN_S=${MEMRL_ALFWORLD_DEFERRED_REPAIR_COOLDOWN_S:-30}
export MEMRL_ALFWORLD_DEFERRED_REPAIR_MAX_GAMES=${MEMRL_ALFWORLD_DEFERRED_REPAIR_MAX_GAMES:-8}

CFG="$TMPROOT/config.yaml"
trap 'rm -f "$CFG"' EXIT
python3 - "$CFG" "$RESUME_ROOT" "$BRANCH_NAME" "$MATRIX_CREDENTIAL_CONFIG" <<'PY'
import os,sys
from pathlib import Path
import yaml
out,resume_root,name,credential_config=sys.argv[1:]
cfg=yaml.safe_load(Path('configs/rl_alf_config.opus47_region.yaml').read_text())
creds=yaml.safe_load(Path(credential_config).read_text())
def credential(model):
 for item in creds.get('model_list',[]):
  if item.get('model_name')==model:
   params=item.get('litellm_params') or {}; key=params.get('api_key')
   if isinstance(key,str) and key.startswith('os.environ/'): key=os.environ.get(key.split('/',1)[1])
   if not key: raise RuntimeError(f'Matrix credential for {model!r} unavailable')
   return key,params.get('api_base') or 'https://matrixllm.alipay.com/v1/'
 raise RuntimeError(f'No credential mapping for {model!r}')
llm_key,llm_base=credential('claude-opus-4-7'); emb_key,emb_base=credential('text-embedding-3-large')
cfg['llm']['api_key'],cfg['llm']['base_url']=llm_key,llm_base
cfg['embedding']['api_key'],cfg['embedding']['base_url']=emb_key,emb_base
cfg['memory']['build_strategy']='trajectory'
cfg['experiment']['experiment_name']=name
cfg['experiment']['num_sections']=9
cfg['experiment']['dataset_ratio']=float(os.environ.get('DATASET_RATIO','1.0'))
cfg['experiment']['valid_interval']=1; cfg['experiment']['test_interval']=0
cfg['experiment']['ckpt_resume_enabled']=True; cfg['experiment']['ckpt_resume_path']=resume_root
cfg['experiment'].pop('ckpt_resume_epoch',None)
Path(out).write_text(yaml.safe_dump(cfg,sort_keys=False));os.chmod(out,0o600)
PY

echo "[INFO] matched resume root: $RESUME_ROOT"
echo "[INFO] preserves accumulated source ledger + cap128 + frozen topology; no legacy reset"
echo "[INFO] only ablation change: region_evidence_sharpen_alpha=3.0 (baseline source alpha=2.0)"
read -r -a EXTRA_ALF_ARGS <<< "${MEMRL_ALFWORLD_EXTRA_ARGS:-}"
echo "[INFO] experiment branch: $BRANCH_NAME"
echo "[INFO] extra ALFWorld args: ${EXTRA_ALF_ARGS[*]:-<none>}"
python3 run/run_alfworld.py --config "$CFG" --region --region_gating_mode additive \
 --region_split_evidence_migration_mode soft_source_conserving --region_freeze_topology \
 --max_candidates_per_sim_key 128 --region_evidence_sharpen_alpha 3.0 \
 --shrinkage_confidence_k 3.0 --propagation_eta 0.06 --val_lambda_max 0.10 \
 --failure_summary_n_slots 1 --skip_initial_eval "${EXTRA_ALF_ARGS[@]}"

#!/bin/bash
# S8 fixed-checkpoint ID-only 3-point Region calibration. No train/OOD/memory writes.
set -euo pipefail
MEMRL_DIR=/storage/openpsi/users/yl/agent-memory/MemRL
SNAP=/storage/openpsi/experiments/checkpoints/admin/yl-mem-region/alfworld/alfworld/exp_alfworld_region_opus47_20260623-100806/local_cache/snapshot/8
MATRIX_CREDENTIAL_CONFIG=/storage/openpsi/users/yl/cfuse/rq3_v11_clean_20260719/config_multisurface_isolated.yaml
RUN_TAG=${RUN_TAG:-$(date +%Y%m%d-%H%M%S)}
LOGFILE="$MEMRL_DIR/logs/aistudio_alf_opus47_s8_id_3point_${RUN_TAG}.log"
exec > >(tee -a "$LOGFILE") 2>&1
cd "$MEMRL_DIR"
[[ -f "$SNAP/local_cache/region_manager.json" ]] || { echo "[FATAL] missing S8 snapshot"; exit 1; }
[[ -f "$MATRIX_CREDENTIAL_CONFIG" ]] || { echo "[FATAL] missing credential config"; exit 1; }
python3 - "$SNAP" <<'PY'
import json,sys
from pathlib import Path
p=Path(sys.argv[1]);m=json.loads((p/'snapshot_meta.json').read_text());r=json.loads((p/'local_cache/region_manager.json').read_text())
assert str(m.get('checkpoint_id'))=='8',m
assert r.get('is_clustered') and r.get('regions') and r.get('membership_weights')
print('[PREFLIGHT] S8 fixed snapshot verified: checkpoint_id=8, regions=%d, memberships=%d' % (len(r['regions']),len(r['membership_weights'])))
PY
VENV_SP=/AReaL/.venv/lib/python3.12/site-packages
pip install -e . --no-deps --target "$VENV_SP" -i https://pypi.antfin-inc.com/simple/ 2>&1 | tail -2
pip install mem0ai 'chonkie==1.2.1' tensorboard pandas tqdm hdbscan concurrent-log-handler textworld alfworld --target "$VENV_SP" -i https://pypi.antfin-inc.com/simple/ 2>&1 | tail -5 || true
export PYTHONPATH=/storage/openpsi/users/yl/agent-memory/.local/lib/python3.12/site-packages:${PYTHONPATH:-}
export PYTHONDONTWRITEBYTECODE=1 HF_HOME=/storage/openpsi/users/yl/agent-memory/.cache/huggingface
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export TMPDIR=/dev/shm/alf_opus47_s8_id_3point
export TEMP="$TMPDIR" TMP="$TMPDIR"
mkdir -p "$TMPDIR"
export MEMRL_LLM_MIN_INTERVAL=2.0 MEMRL_LLM_MAX_RETRIES=5 MEMRL_LLM_TENACITY_ATTEMPTS=5
export MEMRL_ALFWORLD_LLM_CONCURRENCY=4
export MEMRL_EMBED_RATE_LIMIT_KEY=opus47-s8-id-3point
export MEMRL_EMBED_GLOBAL_MIN_INTERVAL=2.5 MEMRL_EMBED_MIN_INTERVAL=2.5 MEMRL_EMBED_THROTTLE=2.5
export MEMRL_ALFWORLD_DEFERRED_REPAIR=1
export MEMRL_ALFWORLD_DEFERRED_REPAIR_COOLDOWN_S=30
export MEMRL_ALFWORLD_DEFERRED_REPAIR_MAX_GAMES=8
# Explicitly disable experimental prompt/routing changes.
export MEMRL_ALFWORLD_STATE_GUARD_PROMPT=0 MEMRL_ALFWORLD_PROGRAM_GUIDE=0
export MEMRL_REGION_RETEMPERATURE=
export MEMRL_UTILITY_ANCHOR_CALIBRATED=0

for SPEC in current:0.10:3 conservative:0.05:8 very_conservative:0.03:15; do
  IFS=: read -r NAME LMAX K <<< "$SPEC"
  CFG="$TMPDIR/s8_id_${NAME}.yaml"
  python3 - "$CFG" "$SNAP" "$NAME" "$RUN_TAG" "$MATRIX_CREDENTIAL_CONFIG" <<'PY'
import os,sys,yaml
from pathlib import Path
out,snap,name,tag,cred_path=sys.argv[1:]
cfg=yaml.safe_load(Path('configs/rl_alf_config.opus47_region.yaml').read_text())
creds=yaml.safe_load(Path(cred_path).read_text())
def cred(model):
 for item in creds.get('model_list',[]):
  if item.get('model_name')==model:
   p=item.get('litellm_params') or {}; key=p.get('api_key')
   if isinstance(key,str) and key.startswith('os.environ/'): key=os.environ.get(key.split('/',1)[1])
   if not key: raise RuntimeError(model)
   return key,p.get('api_base') or 'https://matrixllm.alipay.com/v1/'
 raise RuntimeError(model)
llmk,llmb=cred('claude-opus-4-7');embk,embb=cred('text-embedding-3-large')
cfg['llm']['api_key'],cfg['llm']['base_url']=llmk,llmb
cfg['embedding']['api_key'],cfg['embedding']['base_url']=embk,embb
cfg['memory']['load_from_checkpoint']=True;cfg['memory']['checkpoint_path']=snap
cfg['experiment']['experiment_name']=f'alfworld_region_opus47_s8_id3_{name}_{tag}'
cfg['experiment']['mode']='test';cfg['experiment']['n_eval_runs']=1
cfg['experiment']['ckpt_resume_enabled']=False;cfg['experiment']['ckpt_resume_path']='';cfg['experiment']['ckpt_resume_epoch']=None
cfg['experiment']['save_memories']=False;cfg['experiment']['save_trajectories']=True
Path(out).write_text(yaml.safe_dump(cfg,sort_keys=False));os.chmod(out,0o600)
PY
  export MEMRL_RUN_ID="s8-id3-${NAME}-${RUN_TAG}"
  echo "============================================================"
  echo "[ID-ONLY] S8 setting=$NAME lambda_max=$LMAX confidence_k=$K"
  echo "[ID-ONLY] global retrieval, z-norm ON, temperature=0.10, no memory writes, no OOD"
  echo "============================================================"
  python3 run/run_alfworld.py --config "$CFG" --region --region_gating_mode additive \
    --shrinkage_confidence_k "$K" --val_lambda_max "$LMAX" --propagation_eta 0.12 \
    --failure_summary_n_slots 1 --id_eval_only
  echo "[ID-ONLY] $NAME completed at $(date)"
done

#!/bin/bash
# Fixed S8 ID-only: best conservative Region params + max_steps=40 + 3-round deferred repair.
set -euo pipefail
MEMRL_DIR=/storage/openpsi/users/yl/agent-memory/MemRL
SNAP=/storage/openpsi/experiments/checkpoints/admin/yl-mem-region/alfworld/alfworld/exp_alfworld_region_opus47_20260623-100806/local_cache/snapshot/8
MATRIX_CREDENTIAL_CONFIG=/storage/openpsi/users/yl/cfuse/rq3_v11_clean_20260719/config_multisurface_isolated.yaml
RUN_TAG=${RUN_TAG:-$(date +%Y%m%d-%H%M%S)}
LOGFILE="$MEMRL_DIR/logs/aistudio_alf_opus47_s8_id_max40_${RUN_TAG}.log"
exec > >(tee -a "$LOGFILE") 2>&1
cd "$MEMRL_DIR"
[[ -f "$SNAP/local_cache/region_manager.json" ]] || exit 1
[[ -f "$MATRIX_CREDENTIAL_CONFIG" ]] || exit 1
python3 - "$SNAP" <<'PY'
import json,sys
from pathlib import Path
p=Path(sys.argv[1]);m=json.loads((p/'snapshot_meta.json').read_text());assert str(m.get('checkpoint_id'))=='8'
print('[PREFLIGHT] fixed S8 checkpoint verified')
PY
VENV_SP=/AReaL/.venv/lib/python3.12/site-packages
pip install -e . --no-deps --target "$VENV_SP" -i https://pypi.antfin-inc.com/simple/ 2>&1 | tail -2
pip install mem0ai 'chonkie==1.2.1' tensorboard pandas tqdm hdbscan concurrent-log-handler textworld alfworld --target "$VENV_SP" -i https://pypi.antfin-inc.com/simple/ 2>&1 | tail -5 || true
export PYTHONPATH=/storage/openpsi/users/yl/agent-memory/.local/lib/python3.12/site-packages:${PYTHONPATH:-}
export PYTHONDONTWRITEBYTECODE=1 HF_HOME=/storage/openpsi/users/yl/agent-memory/.cache/huggingface
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export TMPDIR=/dev/shm/alf_opus47_s8_id_max40
export TEMP="$TMPDIR" TMP="$TMPDIR"; mkdir -p "$TMPDIR"
export MEMRL_LLM_MIN_INTERVAL=2.0 MEMRL_LLM_MAX_RETRIES=5 MEMRL_LLM_TENACITY_ATTEMPTS=5
export MEMRL_ALFWORLD_LLM_CONCURRENCY=4
export MEMRL_EMBED_RATE_LIMIT_KEY=opus47-s8-id-max40
export MEMRL_EMBED_GLOBAL_MIN_INTERVAL=2.5 MEMRL_EMBED_MIN_INTERVAL=2.5 MEMRL_EMBED_THROTTLE=2.5
export MEMRL_ALFWORLD_DEFERRED_REPAIR=1
export MEMRL_ALFWORLD_DEFERRED_REPAIR_ROUNDS=3
export MEMRL_ALFWORLD_DEFERRED_REPAIR_COOLDOWNS=30,60,120
export MEMRL_ALFWORLD_DEFERRED_REPAIR_MAX_GAMES=3
export MEMRL_ALFWORLD_STATE_GUARD_PROMPT=0 MEMRL_ALFWORLD_PROGRAM_GUIDE=0
export MEMRL_REGION_RETEMPERATURE=
export MEMRL_UTILITY_ANCHOR_CALIBRATED=0
CFG="$TMPDIR/s8_id_max40.yaml"
python3 - "$CFG" "$SNAP" "$RUN_TAG" "$MATRIX_CREDENTIAL_CONFIG" <<'PY'
import os,sys,yaml
from pathlib import Path
out,snap,tag,cred_path=sys.argv[1:]
cfg=yaml.safe_load(Path('configs/rl_alf_config.opus47_region.yaml').read_text())
creds=yaml.safe_load(Path(cred_path).read_text())
def cred(model):
 for item in creds.get('model_list',[]):
  if item.get('model_name')==model:
   p=item.get('litellm_params') or {}; key=p.get('api_key')
   if isinstance(key,str) and key.startswith('os.environ/'): key=os.environ.get(key.split('/',1)[1])
   return key,p.get('api_base') or 'https://matrixllm.alipay.com/v1/'
 raise RuntimeError(model)
llmk,llmb=cred('claude-opus-4-7');embk,embb=cred('text-embedding-3-large')
cfg['llm']['api_key'],cfg['llm']['base_url']=llmk,llmb
cfg['embedding']['api_key'],cfg['embedding']['base_url']=embk,embb
cfg['memory']['load_from_checkpoint']=True;cfg['memory']['checkpoint_path']=snap
cfg['experiment']['experiment_name']=f'alfworld_region_opus47_s8_full140_max60_mrt120_{tag}'
cfg['experiment']['mode']='test';cfg['experiment']['n_eval_runs']=1;cfg['experiment']['max_steps']=60;cfg['experiment']['max_recent_turns']=120
cfg['experiment']['ckpt_resume_enabled']=False;cfg['experiment']['ckpt_resume_path']='';cfg['experiment']['ckpt_resume_epoch']=None
cfg['experiment']['save_memories']=False;cfg['experiment']['save_trajectories']=True
Path(out).write_text(yaml.safe_dump(cfg,sort_keys=False));os.chmod(out,0o600)
PY
export MEMRL_RUN_ID="s8-full140-max60-mrt120-${RUN_TAG}"
echo '[ID-ONLY] S8 full 140, lambda=0.03 k=15 max_steps=60 max_recent_turns=120, history fix'
python3 run/run_alfworld.py --config "$CFG" --region --region_gating_mode additive \
  --shrinkage_confidence_k 15 --val_lambda_max 0.03 --propagation_eta 0.12 \
  --failure_summary_n_slots 1 --id_eval_only

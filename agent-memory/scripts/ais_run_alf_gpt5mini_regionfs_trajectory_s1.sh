#!/bin/bash
# GPT-5 mini ALFWorld Region+FS trajectory: resume completed S1 and run through S10.
# Fresh memory, S1->S10, standard max_steps=30.
set -euo pipefail
MEMRL_DIR=/storage/openpsi/users/yl/agent-memory/MemRL
MATRIX_CREDENTIAL_CONFIG=/storage/openpsi/users/yl/cfuse/rq3_v11_clean_20260719/config_multisurface_isolated.yaml
# Stable experiment identity MUST survive AIS retry. Do not derive this from the
# retry container clock: output-dir identity is what lets a retry find sN_bM.
# AIS injects RUN_TAG on every retry. This experiment must keep one identity,
# otherwise each retry creates a fresh output tree and cannot find sN_bM.
RUN_TAG=20260802-232000
RUN_ID=gpt5mini-regionfs-trajectory-s1-20260802-232000
BRANCH_NAME=alfworld_regionfs_gpt5mini_trajectory_topology_stable_s1
OUTPUT_ROOT=/storage/openpsi/experiments/checkpoints/admin/yl-mem-region/alfworld_gpt5mini_regionfs_trajectory_s1_20260802b
EXP_DIR="$OUTPUT_ROOT/alfworld/exp_${BRANCH_NAME}_${RUN_ID}"
SNAPSHOT_ROOT="$EXP_DIR/local_cache/snapshot"
LOGFILE="$MEMRL_DIR/logs/aistudio_alf_gpt5mini_regionfs_trajectory_s1_${RUN_TAG}.log"
TMPROOT="/dev/shm/alf_gpt5mini_regionfs_trajectory_s1_${RUN_TAG}"
exec > >(tee -a "$LOGFILE") 2>&1
cd "$MEMRL_DIR"
[[ -f "$MATRIX_CREDENTIAL_CONFIG" ]] || { echo '[FATAL] credential config missing'; exit 1; }
mkdir -p "$OUTPUT_ROOT" "$TMPROOT"; chmod 700 "$TMPROOT"
VENV_SP=/AReaL/.venv/lib/python3.12/site-packages
pip install -e . --no-deps --target "$VENV_SP" -i https://pypi.antfin-inc.com/simple/ 2>&1 | tail -2
pip install mem0ai 'chonkie==1.2.1' tensorboard pandas tqdm hdbscan concurrent-log-handler textworld alfworld --target "$VENV_SP" -i https://pypi.antfin-inc.com/simple/ 2>&1 | tail -5 || true
export PYTHONPATH=/storage/openpsi/users/yl/agent-memory/.local/lib/python3.12/site-packages:${PYTHONPATH:-}
export PYTHONDONTWRITEBYTECODE=1 HF_HOME=/storage/openpsi/users/yl/agent-memory/.cache/huggingface
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export TMPDIR="$TMPROOT" TEMP="$TMPROOT" TMP="$TMPROOT"
export MEMRL_RUN_ID="$RUN_ID"
# Shared API safety: one Opus train job, serialized memory writes and conservative request spacing.
# Batch-32 staggered concurrency: allow 32 in-flight workers, start one new chat request every 1s.
export MEMRL_LLM_MIN_INTERVAL=1.0 MEMRL_LLM_MAX_RETRIES=5 MEMRL_LLM_TENACITY_ATTEMPTS=5
export MEMRL_ALFWORLD_LLM_CONCURRENCY=32
export MEMRL_EMBED_RATE_LIMIT_KEY=gpt5mini-regionfs-trajectory-s1
export MEMRL_EMBED_GLOBAL_MIN_INTERVAL=4.0 MEMRL_EMBED_MIN_INTERVAL=4.0 MEMRL_EMBED_THROTTLE=4.0
export MEMRL_EMBED_429_BASE_DELAY=15 MEMRL_EMBED_429_MAX_DELAY=180
export MEMRL_UPDATE_MAX_WORKERS=1
# Keep reliability fixes; standard benchmark action budget remains 30.
export MEMRL_ALFWORLD_DEFERRED_REPAIR=1
export MEMRL_ALFWORLD_DEFERRED_REPAIR_ROUNDS=3
export MEMRL_ALFWORLD_DEFERRED_REPAIR_COOLDOWNS=30,60,120
export MEMRL_ALFWORLD_DEFERRED_REPAIR_MAX_GAMES=8
export MEMRL_ALFWORLD_STATE_GUARD_PROMPT=0 MEMRL_ALFWORLD_PROGRAM_GUIDE=0
export MEMRL_REGION_RETEMPERATURE=
export MEMRL_UTILITY_ANCHOR_CALIBRATED=0
CFG="$TMPROOT/config.yaml"
trap 'rm -f "$CFG"' EXIT
python3 - "$CFG" "$BRANCH_NAME" "$OUTPUT_ROOT" "$MATRIX_CREDENTIAL_CONFIG" "$SNAPSHOT_ROOT" <<'PY'
import os,sys,yaml
from pathlib import Path
out,name,output_root,cred_path,snapshot_root=sys.argv[1:]
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
llmk,llmb=cred('gpt-5-mini');embk,embb=cred('text-embedding-3-large')
cfg['llm']['api_key'],cfg['llm']['base_url']=llmk,llmb
cfg['llm']['model']='gpt-5-mini-2025-08-07'
cfg['embedding']['api_key'],cfg['embedding']['base_url']=embk,embb
cfg['memory']['build_strategy']='trajectory'
cfg['memory']['retrieve_strategy']='query';cfg['memory']['update_strategy']='adjustment'
cfg['memory']['k_retrieve']=3
cfg['memory']['max_keywords']=5
cfg['memory']['add_similarity_threshold']=0.90
cfg['experiment']['experiment_name']=name
cfg['experiment']['mode']='train';cfg['experiment']['num_sections']=10
cfg['experiment']['batch_size']=32;cfg['experiment']['dataset_ratio']=1.0
cfg['experiment']['max_steps']=30;cfg['experiment']['max_recent_turns']=60
cfg['experiment']['output_dir']=output_root
cfg['experiment']['valid_interval']=1;cfg['experiment']['test_interval']=1
# Retry safety: only resume from this experiment's own latest complete batch checkpoint.
# A first launch has no snapshot root and remains an empty-memory S1 start.
root=Path(snapshot_root)
def complete_batch(p):
    return (
        p.is_dir()
        and (p/'snapshot_meta.json').is_file()
        and (p/'cube'/'textual_memory.json').is_file()
        and (p/'cube'/'textual_memory.json').stat().st_size > 0
        and (p/'qdrant'/'meta.json').is_file()
        and (p/'local_cache'/'cum_state.json').is_file()
    )
latest=[]
if root.is_dir():
    for child in root.iterdir():
        if complete_batch(child):
            import re
            m=re.fullmatch(r's(\d+)_b(\d+)',child.name)
            if m: latest.append((int(m.group(1)),int(m.group(2)),child))
if latest:
    _,_,batch=max(latest)
    cfg['experiment']['ckpt_resume_enabled']=True
    cfg['experiment']['ckpt_resume_path']=str(root)
    cfg['experiment'].pop('ckpt_resume_epoch',None)
    print(f'[RESUME-CONFIG] own latest complete checkpoint={batch}')
else:
    cfg['experiment']['ckpt_resume_enabled']=False
    cfg['experiment']['ckpt_resume_path']=''
    cfg['experiment'].pop('ckpt_resume_epoch',None)
    print('[RESUME-CONFIG] no own complete checkpoint: fresh S1')
cfg['experiment']['save_memories']=True;cfg['experiment']['save_trajectories']=True
# 72B Cell-A stability scoring params.
cfg['rl_config']['tau']=0.60
cfg['rl_config']['weight_sim']=0.45
cfg['rl_config']['weight_q']=0.55
Path(out).write_text(yaml.safe_dump(cfg,sort_keys=False));os.chmod(out,0o600)
PY
python3 - "$CFG" "$SNAPSHOT_ROOT" <<'PY'
import inspect,os,sys,yaml
from pathlib import Path
from memrl.service.region_manager import RegionManager
cfg=yaml.safe_load(Path(sys.argv[1]).read_text())
snapshot_root=sys.argv[2]
assert cfg['memory']['build_strategy']=='trajectory'
assert isinstance(cfg['experiment']['ckpt_resume_enabled'], bool)
assert cfg['experiment']['max_steps']==30 and cfg['experiment']['num_sections']==10
assert cfg['llm']['model']=='gpt-5-mini-2025-08-07'
assert os.environ.get('MEMRL_ALFWORLD_LLM_CONCURRENCY') == '32'
assert float(os.environ.get('MEMRL_LLM_MIN_INTERVAL', '0')) == 1.0
if cfg['experiment']['ckpt_resume_enabled']:
    assert Path(cfg['experiment']['ckpt_resume_path']).resolve() == Path(snapshot_root).resolve()
src=inspect.getsource(RegionManager)
for token in ['region_source_success_by_region','region_source_total_by_region','soft_source_conserving','_split_child_routing_weights']:
 assert token in src,token
assert 'q_val * float(q_count)' not in src
print('[PREFLIGHT] gpt-5-mini trajectory; current source-ledger split; S1->S10 resume; standard 30 steps')
PY
echo '================================================================'
echo '[CELL] GPT-5 mini trajectory + Region+FS topology stability (resume S1, finish S2-S10)'
echo '[CELL] resume own latest checkpoint, total S1-S10, batch=32, max_steps=30, max_recent_turns=60'
echo '[CELL] eta=0.03 shrinkage_k=2.5 tau=0.60 ws/wq=0.45/0.55 FS=1'
echo '[CELL] topology init=3000, no mid-epoch edits, merge_interval=3553, cooldown=1; LLM concurrency=32 + 1s chat stagger; embedding=4s'
echo "[CELL] output=$OUTPUT_ROOT run_id=$RUN_ID exp_dir=$EXP_DIR"
echo "[CELL] snapshot_root=$SNAPSHOT_ROOT"
echo '================================================================'
python3 scripts/run_alfworld_resume_boundary.py --config "$CFG" \
  --region --region_gating_mode additive --region_utility_mode beta \
  --region_split_evidence_migration_mode soft_source_conserving \
  --region_cluster_init_step 3000 --region_merge_interval 3553 \
  --region_disable_mid_epoch_topology --region_topology_cooldown_sections 1 \
  --shrinkage_confidence_k 2.5 --propagation_eta 0.03 \
  --val_lambda_max 0.45 --no_z_norm \
  --explore_schedule '0,1,1,1,0,0,0,0,0,0' \
  --failure_summary_n_slots 1

#!/usr/bin/env bash
# Read-only corrected Region checkpoint validation grid. Source snapshot is copied to /tmp.
set -euo pipefail
PROJECT_DIR=/storage/openpsi/users/yl/agent-memory/MemRL
LOCAL_SP=/storage/openpsi/users/yl/agent-memory/.local/lib/python3.12/site-packages
SOURCE_SNAPSHOT=${SOURCE_SNAPSHOT:?}
EVAL_LABEL=${EVAL_LABEL:?}
EVAL_SECTION=${EVAL_SECTION:-9}
EVAL_WEIGHT_SIM=${EVAL_WEIGHT_SIM:?}
EVAL_WEIGHT_Q=${EVAL_WEIGHT_Q:?}
EVAL_FAILURE_SLOTS=${EVAL_FAILURE_SLOTS:?}
TRACE_OUTPUT=${TRACE_OUTPUT:-}
HOST_SHORT=$(hostname | sed 's/^gpulingjun//; s/\..*$//')
TS=$(date +%Y%m%d_%H%M%S)
LOGFILE="$PROJECT_DIR/logs/llb_db_${EVAL_LABEL}_region_readonly_valeval_${HOST_SHORT}_${TS}.log"
exec > >(tee -a "$LOGFILE") 2>&1
cd "$PROJECT_DIR"
echo "READONLY_REGION_EVAL label=$EVAL_LABEL source=$SOURCE_SNAPSHOT sim=$EVAL_WEIGHT_SIM q=$EVAL_WEIGHT_Q fs=$EVAL_FAILURE_SLOTS"
[[ -f "$SOURCE_SNAPSHOT/snapshot_meta.json" && -d "$SOURCE_SNAPSHOT/cube" && -d "$SOURCE_SNAPSHOT/qdrant" && -f "$SOURCE_SNAPSHOT/local_cache/region_manager.json" ]] || { echo 'ERROR: unhealthy Region source snapshot' >&2; exit 2; }
export PYTHONPATH="$PROJECT_DIR:$LOCAL_SP:${PYTHONPATH:-}"
export MEMRL_OS_BACKEND=local MEMRL_DB_BACKEND=auto MEMRL_UPDATE_MAX_WORKERS=1
export MEMRL_EMBED_THROTTLE=3.0 MEMRL_EMBED_GLOBAL_MIN_INTERVAL=3.0 MEMRL_LLM_MIN_INTERVAL=1.0
export MEMRL_LLM_MODEL=gpt-4.1-mini-2025-04-14 MEMRL_LLB_REFLECTION_PROMPT=v2 MEMRL_LLB_SCRIPT_DETAIL=db_pattern
export MEMRL_EVAL_ONLY_SECTION="$EVAL_SECTION"
unset MEMRL_DB_MULTI_AXIS || true
unset MEMRL_REGION_SPLIT_RANGE_FRACTION || true
unset MEMRL_REGION_SPLIT_ON_RESUME || true
unset MEMRL_REGION_RETRIEVE_MODE || true
unset MEMRL_WEIGHTED_REGION_QUOTA || true
unset MEMRL_RETRIEVAL_AUDIT_PATH || true
unset MEMRL_FINAL_MEMORY_DEDUP || true
export HF_ENDPOINT=https://hf-mirror.com HF_HOME=/tmp/huggingface
apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq mariadb-server >/dev/null 2>&1 || { echo 'ERROR: mariadb install failed' >&2; exit 3; }
VENV_SP=/AReaL/.venv/lib/python3.12/site-packages
pip install mem0ai 'chonkie==1.2.1' tensorboard pandas tqdm concurrent-log-handler mysql-connector-python hdbscan --target "$VENV_SP" -i https://pypi.antfin-inc.com/simple/
WORKDIR=$(mktemp -d /tmp/llb_db_region_grid.XXXXXX)
trap 'rm -rf "$WORKDIR"' EXIT
LOCAL_SNAPSHOT="$WORKDIR/snapshot"
LOCAL_CONFIG="$WORKDIR/config.yaml"
LOCAL_OUTPUT="$WORKDIR/out"
cp -a "$SOURCE_SNAPSHOT" "$LOCAL_SNAPSHOT"
python3 - "$LOCAL_CONFIG" "$LOCAL_SNAPSHOT" "$LOCAL_OUTPUT" "$EVAL_WEIGHT_SIM" "$EVAL_WEIGHT_Q" <<'PYCFG'
import sys,yaml
from pathlib import Path
dst,snapshot,output=map(Path,sys.argv[1:4]); ws,wq=map(float,sys.argv[4:6])
cfg=yaml.safe_load(Path('configs/rl_llb_db_region_fs_hardfix_s3.yaml').read_text())
cfg['llm']['api_key']='runtime-injected'; cfg['embedding']['api_key']='runtime-injected'
cfg['memory']['load_from_checkpoint']=True; cfg['memory']['checkpoint_path']=str(snapshot); cfg['memory']['user_id']='readonly_region_grid'
cfg['experiment']['experiment_name']='readonly_region_grid'; cfg['experiment']['output_dir']=str(output)
cfg['experiment']['save_trajectories']=False; cfg['experiment']['save_memories']=False
cfg['experiment']['ckpt_save_every_n_batches']=0; cfg['experiment']['eval_runs']=1; cfg['experiment']['eval_temperature']=0.0
cfg['experiment']['trace_jsonl_path']=str(output/'trace.jsonl')
cfg['experiment']['trace_sample_filter']=None
cfg['rl_config']['weight_sim']=ws; cfg['rl_config']['weight_q']=wq
dst.write_text(yaml.safe_dump(cfg,sort_keys=False))
PYCFG
python3 scripts/run_llb_with_rotated_matrix_credentials.py \
 --config "$LOCAL_CONFIG" --output_dir "$LOCAL_OUTPUT" \
 --region --region_gating_mode additive --propagation_eta 0.12 \
 --shrinkage_confidence_k 3.0 --no_z_norm \
 --explore_schedule '0,4,3,2,2,1,1,1,1,0' \
 --failure_summary_n_slots "$EVAL_FAILURE_SLOTS" --resume_eval_section 0
if [[ -n "$TRACE_OUTPUT" ]]; then
  mkdir -p "$(dirname "$TRACE_OUTPUT")"
  cp "$LOCAL_OUTPUT/trace.jsonl" "$TRACE_OUTPUT"
  echo "READONLY_REGION_TRACE_SAVED path=$TRACE_OUTPUT"
fi
echo "READONLY_REGION_EVAL_COMPLETE label=$EVAL_LABEL"

#!/usr/bin/env bash
# Strict read-only dense-only validation of the original Mem0 E7 checkpoint.
set -euo pipefail
PROJECT_DIR=/storage/openpsi/users/yl/agent-memory/MemRL
SOURCE_SNAPSHOT=/storage/openpsi/experiments/checkpoints/admin/yl-mem-region/llb_baselines/exp_llb_db_mem0_gpt41mini_mem0-db-gpt41mini-20260719-v2/snapshot/7
LOCAL_SP=/storage/openpsi/users/yl/agent-memory/.local/lib/python3.12/site-packages
HOST_SHORT=$(hostname | sed 's/^gpulingjun//; s/\..*$//')
TS=$(date +%Y%m%d_%H%M%S)
LOGFILE="$PROJECT_DIR/logs/llb_db_mem0_e7_dense_readonly_valeval_${HOST_SHORT}_${TS}.log"
exec > >(tee -a "$LOGFILE") 2>&1
cd "$PROJECT_DIR"
echo "READONLY_DENSE_MEM0_E7 source=$SOURCE_SNAPSHOT"
[[ -f "$SOURCE_SNAPSHOT/snapshot_meta.json" && -d "$SOURCE_SNAPSHOT/mem0_qdrant" ]] || { echo 'ERROR: unhealthy E7 snapshot' >&2; exit 2; }
export PYTHONPATH="$PROJECT_DIR:$LOCAL_SP:${PYTHONPATH:-}"
export MEMRL_OS_BACKEND=local MEMRL_DB_BACKEND=auto MEMRL_UPDATE_MAX_WORKERS=1
export MEMRL_MEM0_MIN_INTERVAL=1.0 MEMRL_LLM_MODEL=gpt-4.1-mini-2025-04-14
export MEMRL_LLB_REFLECTION_PROMPT=v2 MEMRL_LLB_SCRIPT_DETAIL=db_pattern
export MEM0_TELEMETRY=False ANONYMIZED_TELEMETRY=False POSTHOG_DISABLED=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export FASTEMBED_CACHE_PATH="$PROJECT_DIR/scripts/fastembed_cache"
export HF_ENDPOINT=https://hf-mirror.com HF_HOME=/tmp/huggingface
export MEMRL_EVAL_ONLY_SECTION=7
apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq mariadb-server >/dev/null 2>&1 || { echo 'ERROR: mariadb install failed' >&2; exit 3; }
command -v mysqld >/dev/null || { echo 'ERROR: mysqld unavailable' >&2; exit 3; }
MEM0_DEPS=/tmp/llb_db_mem0_dense_e7_site
rm -rf "$MEM0_DEPS"; mkdir -p "$MEM0_DEPS"
pip install 'mem0ai==2.0.12' 'qdrant-client[fastembed]>=1.17,<1.18' 'huggingface-hub>=0.34,<1.0' 'tokenizers>=0.22,<0.23' --target "$MEM0_DEPS" -i https://pypi.antfin-inc.com/simple/
export PYTHONPATH="$PROJECT_DIR:$MEM0_DEPS:$LOCAL_SP:${PYTHONPATH:-}"
WORKDIR=$(mktemp -d /tmp/llb_db_mem0_e7_dense_eval.XXXXXX)
trap 'rm -rf "$WORKDIR"' EXIT
LOCAL_SNAPSHOT="$WORKDIR/snapshot_7"; LOCAL_CONFIG="$WORKDIR/config.yaml"; LOCAL_OUTPUT="$WORKDIR/out"
cp -a "$SOURCE_SNAPSHOT" "$LOCAL_SNAPSHOT"
python3 - "$LOCAL_CONFIG" "$LOCAL_SNAPSHOT" "$LOCAL_OUTPUT" <<'PYCFG'
import sys
from pathlib import Path
import yaml
out,snap,output=map(Path,sys.argv[1:])
cfg=yaml.safe_load(Path('configs/rl_llb_db_mem0.yaml').read_text())
cfg['llm']['api_key']='runtime-injected'; cfg['embedding']['api_key']='runtime-injected'
cfg['llm']['model']='gpt-4.1-mini-2025-04-14'
cfg['memory']['load_from_checkpoint']=True; cfg['memory']['checkpoint_path']=str(snap)
cfg['memory']['user_id']='llb_db_mem0_e7_dense_readonly_eval'
cfg['experiment']['output_dir']=str(output); cfg['experiment']['save_trajectories']=False; cfg['experiment']['save_memories']=False
cfg['experiment']['ckpt_save_every_n_batches']=0; cfg['experiment']['ckpt_max_keep']=1; cfg['experiment']['eval_runs']=1; cfg['experiment']['eval_temperature']=0.0
out.write_text(yaml.safe_dump(cfg,sort_keys=False))
PYCFG
# Dense-only audit intentionally bypasses BM25 wrapper. It retains the original
# E7 behavior while still injecting rotated Matrix credentials and eval-only mode.
python3 scripts/run_llb_with_rotated_matrix_credentials.py --config "$LOCAL_CONFIG" --output_dir "$LOCAL_OUTPUT" --mem0 --mem0_infer true --mem0_collection llb_db_mem0_mem0_db_gpt41mini_20260719_v2
echo 'READONLY_DENSE_MEM0_E7_COMPLETE'

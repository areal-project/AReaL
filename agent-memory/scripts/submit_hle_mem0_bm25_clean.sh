#!/usr/bin/env bash
set -euo pipefail
ROOT=/storage/openpsi/users/yl/agent-memory
RID="yl-hle-mem0-bm25-clean-$(date +%Y%m%d-%H%M%S)"
echo "[submit] clean Mem0+BM25 fresh S1 run: $RID"
REMOTE="/tmp/${RID}.sh"
cat > "$REMOTE" <<'REMOTE'
#!/usr/bin/env bash
set -euo pipefail
ROOT=/storage/openpsi/users/yl/agent-memory
cd "$ROOT/MemRL"
LOG="$ROOT/MemRL/logs/aistudio_hle_mem0_bm25_clean_${HLE_RID}.log"
exec > >(tee -a "$LOG") 2>&1
MEM0_SP="/tmp/hle_mem0_clean_site_${HLE_RID}"
rm -rf "$MEM0_SP"; mkdir -p "$MEM0_SP"
export PYTHONPATH="$ROOT/MemRL:$ROOT/.local/lib/python3.12/site-packages:${PYTHONPATH:-}"
echo "[Mem0 clean] bootstrap rid=$HLE_RID host=$(hostname)"
if command -v uv >/dev/null 2>&1; then
  uv pip install --python python3 --target "$MEM0_SP" -i https://pypi.antfin-inc.com/simple/ mem0ai "chonkie==1.2.1" "qdrant-client[fastembed]>=1.17,<1.18" "tokenizers>=0.22,<0.23" "huggingface-hub>=0.34,<1.0" ollama
else
  pip install --target "$MEM0_SP" -i https://pypi.antfin-inc.com/simple/ mem0ai "chonkie==1.2.1" "qdrant-client[fastembed]>=1.17,<1.18" "tokenizers>=0.22,<0.23" "huggingface-hub>=0.34,<1.0" ollama
fi
export PYTHONPATH="$MEM0_SP:$PYTHONPATH"
export HF_HOME="$ROOT/.cache/huggingface" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export FASTEMBED_CACHE_PATH="$ROOT/MemRL/scripts/fastembed_cache"
export MEMRL_REASONING_EFFORT=high MEMRL_RUNNER_MAX_RETRIES=1 MEMRL_RUN_ID="$HLE_RID"
export MEMRL_LLM_MIN_INTERVAL=4.0 MEMRL_EMBED_MIN_INTERVAL=4.0 MEMRL_MEM0_MIN_INTERVAL=4.0 MEMRL_UPDATE_MAX_WORKERS=1
export MEMRL_LLM_CLIENT_TIMEOUT_S=600 MEMRL_LLM_GEN_TIMEOUT_S=600 MEMRL_RESUME_PENDING_TIMEOUT_S=600
export MEMRL_INFRA_RETRY_ROUNDS=3 MEMRL_INFRA_RETRY_WAIT_S=240
export MEMRL_LLM_GLOBAL_MAX_INFLIGHT=64 MEMRL_LLM_INFLIGHT_DIR=/storage/openpsi/experiments/checkpoints/admin/yl-mem-region/.llm_inflight MEMRL_LLM_INFLIGHT_KEY=hle-region-selfrag-gemini35flash
export MEM0_TELEMETRY=false MEM0_TELEMETRY_SAMPLE_RATE=0 POSTHOG_DISABLED=true
python3 -c "import memrl, mem0, qdrant_client, fastembed; from memrl.service.mem0_memory_service import Mem0MemoryService; print('[Mem0 clean] isolated dependencies OK')"
SAFE_ROOT="/storage/openpsi/experiments/checkpoints/admin/yl-mem-region/hle/exp_hle_mem0_bm25_clean_${HLE_RID}"
TMP_CFG="/tmp/hle_mem0_bm25_clean_${HLE_RID}.yaml"
export SAFE_ROOT TMP_CFG
python3 - <<'PY'
import os,yaml
from pathlib import Path
src=Path('configs/rl_hle_config.rag_gemini35flash_resume_1b9.yaml')
cfg=yaml.safe_load(src.read_text())
cred=yaml.safe_load(Path('/storage/openpsi/users/yl/cfuse/rq3_v11_clean_20260719/config_multisurface_isolated.yaml').read_text())
keys={x['model_name']:x['litellm_params']['api_key'] for x in cred['model_list']}
cfg['llm']['api_key']=keys['gemini-3.5-flash']; cfg['embedding']['api_key']=keys['text-embedding-3-large']
cfg['memory'].update(build_strategy='proceduralization',user_id='hle_mem0_bm25_clean_user')
root=Path(os.environ['SAFE_ROOT']); snap=root/'snapshot'
has=snap.is_dir() and any(p.is_dir() and (p/'snapshot_meta.json').is_file() for p in snap.iterdir())
cfg['experiment'].update(experiment_name='hle_mem0_bm25_clean',ckpt_resume_enabled=has,ckpt_resume_path=str(root) if has else '',ckpt_resume_epoch=0,ckpt_resume_prefer_current_run=True,ckpt_save_every_n_batches=5,ckpt_max_keep=3)
yaml.safe_dump(cfg,open(os.environ['TMP_CFG'],'w'),sort_keys=False)
print('[Mem0 clean] remote-start mode='+('resume' if has else 'fresh')+' root='+str(root))
Path('/tmp/judge_'+os.environ['HLE_RID']).write_text(keys['gpt-4o-2024-11-20'])
PY
JUDGE=$(cat "/tmp/judge_${HLE_RID}"); rm -f "/tmp/judge_${HLE_RID}"
echo "[Mem0 clean] interval=4.0s update_workers=1 save_every=5 dynamic-own-resume=true"
COLLECTION="hle_mem0_bm25_clean_${HLE_RID//-/_}"
python3 scripts/run_hle_mem0_bm25.py --config "$TMP_CFG" --train data/hle/hle_test.parquet --mem0 --mem0_infer true --mem0_collection "$COLLECTION" --judge_model gpt-4o-2024-11-20 --judge_base_url https://matrixllm.alipay.com/v1/ --judge_api_key "$JUDGE"
REMOTE
PAYLOAD=$(base64 -w0 "$REMOTE")
export PYTHONPATH=/tmp/yl_pypai AISTUDIO_LOGIN_NAME=aistudio AISTUDIO_USERNUMBER=477578 WORKER_NUM=0 JOB_TAG='' JOB_NAME="$RID" KM_IMAGE='acr-sh-ant-registry-vpc.cn-shanghai.cr.aliyuncs.com/gpu/areal-runtime:dev-sglang-20260401'
export JOB_COMMAND="HLE_RID=$(printf '%q' "$RID") bash -c $(printf '%q' "echo $PAYLOAD | base64 -d | bash")"
export LAUNCH_CONTAINER_MODE=dev_local PYPAI_HOME="/tmp/pypai_${RID}" TMPDIR=/tmp
cd /tmp
python3 "$ROOT/MemRL/scripts/submit_hle_template.py"

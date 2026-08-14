#!/usr/bin/env bash
set -euo pipefail
: "${HLE_MODE:?}" "${HLE_RID:?}" "${HLE_CKPT:?}"
ROOT=/storage/openpsi/users/yl/agent-memory
cd "$ROOT/MemRL"
CFG=/tmp/hle_403_recovery_${HLE_RID}.yaml
export CFG
python3 - <<'PY'
import os
from pathlib import Path
import yaml
mode=os.environ['HLE_MODE']; ckpt=os.environ['HLE_CKPT']
root=Path('/storage/openpsi/users/yl/agent-memory')
cred=yaml.safe_load((root/'cfuse/rq3_v11_clean_20260719/config_multisurface_isolated.yaml').read_text())
keys={x['model_name']:x['litellm_params']['api_key'] for x in cred['model_list']}
need=('gemini-3.5-flash','text-embedding-3-large','gpt-4o-2024-11-20')
missing=[x for x in need if not keys.get(x)]
if missing: raise RuntimeError(f'missing credential mapping: {missing}')
src={'region':'configs/rl_hle_config.region_gemini35flash_conservative_resume_1b4.yaml','rag':'configs/rl_hle_config.rag_gemini35flash_resume_1b9.yaml','selfrag':'configs/rl_hle_config.selfrag_gemini35flash_resume_1b9.yaml','mem0':'configs/rl_hle_config.rag_gemini35flash_resume_1b9.yaml'}[mode]
cfg=yaml.safe_load(open(src))
cfg['llm']['api_key']=keys['gemini-3.5-flash']; cfg['embedding']['api_key']=keys['text-embedding-3-large']
cfg['experiment'].update(ckpt_resume_enabled=True,ckpt_resume_path=ckpt,ckpt_resume_epoch=0)
if mode=='mem0': cfg['memory'].update(build_strategy='proceduralization',user_id='hle_mem0_gemini35flash_user')
yaml.safe_dump(cfg,open(os.environ['CFG'],'w'),sort_keys=False)
Path('/tmp/hle_403_recovery_judge_key').write_text(keys['gpt-4o-2024-11-20'])
PY
JUDGE_KEY=$(cat /tmp/hle_403_recovery_judge_key); rm -f /tmp/hle_403_recovery_judge_key
export PYTHONPATH="$ROOT/MemRL:$ROOT/.local/lib/python3.12/site-packages:${PYTHONPATH:-}"
export HF_HOME="$ROOT/.cache/huggingface" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 FASTEMBED_CACHE_PATH="$ROOT/MemRL/scripts/fastembed_cache"
export MEMRL_RUN_ID="$HLE_RID" MEMRL_REASONING_EFFORT=high MEMRL_RUNNER_MAX_RETRIES=1
if [[ "$HLE_MODE" == mem0 ]]; then
 SP=/tmp/hle_mem0_site_${HLE_RID}; rm -rf "$SP"; mkdir -p "$SP"
 if command -v uv >/dev/null 2>&1; then uv pip install --python python3 --target "$SP" -i https://pypi.antfin-inc.com/simple/ mem0ai 'chonkie==1.2.1' 'qdrant-client[fastembed]>=1.17,<1.18' 'tokenizers>=0.22,<0.23' 'huggingface-hub>=0.34,<1.0' ollama; else pip install --target "$SP" -i https://pypi.antfin-inc.com/simple/ mem0ai 'chonkie==1.2.1' 'qdrant-client[fastembed]>=1.17,<1.18' 'tokenizers>=0.22,<0.23' 'huggingface-hub>=0.34,<1.0' ollama; fi
 export PYTHONPATH="$SP:$PYTHONPATH" MEMRL_LLM_MIN_INTERVAL=5.0 MEMRL_EMBED_MIN_INTERVAL=5.0 MEMRL_MEM0_MIN_INTERVAL=5.0 MEMRL_UPDATE_MAX_WORKERS=1 MEM0_TELEMETRY=false POSTHOG_DISABLED=true
 LOGFILE="$ROOT/MemRL/logs/aistudio_hle_mem0_gemini35flash_${HLE_RID}.log"; exec > >(tee -a "$LOGFILE") 2>&1
 python3 -c "import mem0,qdrant_client,fastembed; print('[Mem0] precheck imports OK; resuming pre-403 checkpoint')"
 python3 scripts/run_hle_mem0_bm25.py --config "$CFG" --train data/hle/hle_test.parquet --mem0 --mem0_infer true --mem0_collection "hle_mem0_${HLE_RID//-/_}" --judge_model gpt-4o-2024-11-20 --judge_base_url https://matrixllm.alipay.com/v1/ --judge_api_key "$JUDGE_KEY"
else
 SP=/tmp/hle_baseline_site_${HLE_RID}; rm -rf "$SP"; mkdir -p "$SP"; pip install --target "$SP" -i https://pypi.antfin-inc.com/simple/ tensorboard pandas tqdm concurrent-log-handler >/tmp/hle_install_baseline_deps.log 2>&1; export PYTHONPATH="$SP:$PYTHONPATH"
 export MEMRL_LLM_MIN_INTERVAL=5.0 MEMRL_EMBED_MIN_INTERVAL=5.0 MEMRL_UPDATE_MAX_WORKERS=1
 LOGFILE="$ROOT/MemRL/logs/aistudio_hle_${HLE_MODE}_gemini35flash_${HLE_RID}.log"; exec > >(tee -a "$LOGFILE") 2>&1
 if [[ "$HLE_MODE" == region ]]; then python3 run/run_hle_region.py --config "$CFG" --train data/hle/hle_test.parquet --judge_model gpt-4o-2024-11-20 --judge_base_url https://matrixllm.alipay.com/v1/ --judge_api_key "$JUDGE_KEY"; elif [[ "$HLE_MODE" == selfrag ]]; then python3 run/run_hle.py --config "$CFG" --train data/hle/hle_test.parquet --self_rag --self_rag_inject_k 3 --judge_model gpt-4o-2024-11-20 --judge_base_url https://matrixllm.alipay.com/v1/ --judge_api_key "$JUDGE_KEY"; else python3 run/run_hle.py --config "$CFG" --train data/hle/hle_test.parquet --judge_model gpt-4o-2024-11-20 --judge_base_url https://matrixllm.alipay.com/v1/ --judge_api_key "$JUDGE_KEY"; fi
fi

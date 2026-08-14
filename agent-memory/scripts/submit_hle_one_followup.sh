#!/bin/bash
set -euo pipefail
MODE="${1:?usage: $0 rag|selfrag|mem0 RUN_ID}"
RID="${2:?usage: $0 rag|selfrag|mem0 RUN_ID}"
case "$MODE" in rag|selfrag|mem0) ;; *) echo "invalid mode: $MODE" >&2; exit 2;; esac
export AISTUDIO_LOGIN_NAME="aistudio"
export AISTUDIO_USERNUMBER="477578"
: "${AISTUDIO_TOKEN:=}"
export WORKER_NUM=0 JOB_TAG=""
export JOB_NAME="${JOB_NAME_OVERRIDE:-$RID}"
export KM_IMAGE="acr-sh-ant-registry-vpc.cn-shanghai.cr.aliyuncs.com/gpu/areal-runtime:dev-sglang-20260401"
export HLE_MODE="$MODE" HLE_RID="$RID"
export JOB_COMMAND='bash -c '\''
set -euo pipefail
ROOT=/storage/openpsi/users/yl/agent-memory
cd "$ROOT/MemRL"
if [[ "$HLE_MODE" == mem0 ]]; then
  LOGFILE="$ROOT/MemRL/logs/aistudio_hle_mem0_gemini35flash_${HLE_RID}.log"
  exec > >(tee -a "$LOGFILE") 2>&1
  echo "[Mem0 bootstrap] start host=$(hostname) python=$(python3 --version 2>&1) rid=$HLE_RID"
fi
# Import MemRL directly from the mounted source tree. Building a wheel from
# CPFS is unnecessary and has intermittently failed with ESTALE during wheel
# finalization. Keep all third-party bootstrap writes under node-local /tmp.
MEM0_SP=/tmp/hle_mem0_site_${HLE_RID}
BASELINE_SP=/tmp/hle_baseline_site_${HLE_RID}
export PYTHONPATH="$ROOT/MemRL:$ROOT/.local/lib/python3.12/site-packages:${PYTHONPATH:-}"
if [[ "$HLE_MODE" == mem0 ]]; then
  # Isolate Mem0-only dependencies from the AReaL environment used by RAG/Self-RAG.
  rm -rf "$MEM0_SP"
  mkdir -p "$MEM0_SP"
  echo "[Mem0 bootstrap] installing mem0/fastembed dependencies"
  if command -v uv >/dev/null 2>&1; then
    uv pip install --python python3 --target "$MEM0_SP" -i https://pypi.antfin-inc.com/simple/ mem0ai "chonkie==1.2.1" "qdrant-client[fastembed]>=1.17,<1.18" "tokenizers>=0.22,<0.23" "huggingface-hub>=0.34,<1.0" ollama
  else
    pip install --target "$MEM0_SP" -i https://pypi.antfin-inc.com/simple/ mem0ai "chonkie==1.2.1" "qdrant-client[fastembed]>=1.17,<1.18" "tokenizers>=0.22,<0.23" "huggingface-hub>=0.34,<1.0" ollama
  fi
  export PYTHONPATH="$MEM0_SP:$PYTHONPATH"
  python3 -c "import memrl, mem0, qdrant_client, fastembed; from memrl.service.mem0_memory_service import Mem0MemoryService; print(\"mem0 isolated dependency imports OK (fastembed present)\")"
else
  rm -rf "$BASELINE_SP"
  mkdir -p "$BASELINE_SP"
  pip install --target "$BASELINE_SP" -i https://pypi.antfin-inc.com/simple/ tensorboard pandas tqdm concurrent-log-handler >/tmp/hle_install_baseline_deps.log 2>&1
  export PYTHONPATH="$BASELINE_SP:$PYTHONPATH"
  python3 -c "import memrl, memos; print(\"baseline dependency imports OK\")"
fi
export HF_HOME="$ROOT/.cache/huggingface" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export FASTEMBED_CACHE_PATH="$ROOT/MemRL/scripts/fastembed_cache"
export MEMRL_REASONING_EFFORT=high MEMRL_RUNNER_MAX_RETRIES=1 MEMRL_RUN_ID="$HLE_RID"
API_KEY=$(python3 -c "import yaml; print(yaml.safe_load(open(\"configs/rl_hle_config.rag_gemini35flash_resume_1b9.yaml\"))[\"llm\"][\"api_key\"])")
if [[ "$HLE_MODE" == rag ]]; then
  export MEMRL_LLM_MIN_INTERVAL=1.1 MEMRL_EMBED_MIN_INTERVAL=1.7 MEMRL_UPDATE_MAX_WORKERS=2 MEMRL_REQUEST_PROFILE=rag
  LOGFILE="$ROOT/MemRL/logs/aistudio_hle_rag_gemini35flash_${HLE_RID}.log"
  exec > >(tee -a "$LOGFILE") 2>&1
  echo "[RAG] PRECHECK_OK resume=1_b9 llm_interval=1.1 embed_interval=1.7"
  python3 run/run_hle.py --config configs/rl_hle_config.rag_gemini35flash_resume_1b9.yaml --train data/hle/hle_test.parquet --judge_model gpt-4o-2024-11-20 --judge_base_url https://matrixllm.alipay.com/v1/ --judge_api_key "$API_KEY"
elif [[ "$HLE_MODE" == selfrag ]]; then
  export MEMRL_LLM_MIN_INTERVAL=1.9 MEMRL_EMBED_MIN_INTERVAL=2.9 MEMRL_UPDATE_MAX_WORKERS=2 MEMRL_REQUEST_PROFILE=selfrag
  LOGFILE="$ROOT/MemRL/logs/aistudio_hle_selfrag_gemini35flash_${HLE_RID}.log"
  exec > >(tee -a "$LOGFILE") 2>&1
  echo "[Self-RAG] PRECHECK_OK enabled=true inject_top_k=3 resume=1_b9 llm_interval=1.9 embed_interval=2.9"
  python3 run/run_hle.py --config configs/rl_hle_config.selfrag_gemini35flash_resume_1b9.yaml --train data/hle/hle_test.parquet --self_rag --self_rag_inject_k 3 --judge_model gpt-4o-2024-11-20 --judge_base_url https://matrixllm.alipay.com/v1/ --judge_api_key "$API_KEY"
else
  export MEMRL_LLM_MIN_INTERVAL=5.0 MEMRL_EMBED_MIN_INTERVAL=5.0 MEMRL_MEM0_MIN_INTERVAL=5.0 MEMRL_UPDATE_MAX_WORKERS=1 MEMRL_REQUEST_PROFILE=mem0 MEM0_TELEMETRY=false MEM0_TELEMETRY_SAMPLE_RATE=0 POSTHOG_DISABLED=true
  TMP_CFG=/tmp/hle_mem0_${HLE_RID}.yaml
  python3 - "$TMP_CFG" <<"PY"
import sys, yaml
src="configs/rl_hle_config.rag_gemini35flash_resume_1b9.yaml"
cfg=yaml.safe_load(open(src))
cfg["memory"]["build_strategy"]="proceduralization"
cfg["memory"]["user_id"]="hle_mem0_gemini35flash_user"
e=cfg["experiment"]
import os
from pathlib import Path
rid = os.environ.get("HLE_RID")
if not rid:
    raise RuntimeError("HLE_RID is required")
ckpt_root=f"/storage/openpsi/experiments/checkpoints/admin/yl-mem-region/hle/exp_hle_mem0_gemini35flash_{rid}"
snapshot_root=Path(ckpt_root) / "snapshot"
has_snapshot=snapshot_root.is_dir() and any(
    p.is_dir() and ((p / "snapshot_meta.json").is_file() or (p / "mem0_qdrant").is_dir())
    for p in snapshot_root.iterdir()
)
e.update(
    experiment_name="hle_mem0_gemini35flash",
    ckpt_resume_enabled=bool(has_snapshot),
    ckpt_resume_path=ckpt_root if has_snapshot else "",
    ckpt_resume_epoch=0,
    ckpt_save_every_n_batches=10,
)
cfg["_mem0_resume_detected"] = bool(has_snapshot)
yaml.safe_dump(cfg, open(sys.argv[1],"w"), sort_keys=False)
PY
  RESUME_MODE=$(python3 - "$TMP_CFG" <<"PY"
import sys,yaml
cfg=yaml.safe_load(open(sys.argv[1]))
print("resume" if cfg.pop("_mem0_resume_detected", False) else "fresh")
yaml.safe_dump(cfg, open(sys.argv[1],"w"), sort_keys=False)
PY
)
  echo "[Mem0] PRECHECK_OK env=isolated imports=mem0,qdrant_client,fastembed bm25=required infer=true mode=${RESUME_MODE} llm_interval=5.0 embed_interval=5.0 mem0_op_interval=5.0"
  COLLECTION="hle_mem0_${HLE_RID//-/_}"
  python3 scripts/run_hle_mem0_bm25.py --config "$TMP_CFG" --train data/hle/hle_test.parquet --mem0 --mem0_infer true --mem0_collection "$COLLECTION" --judge_model gpt-4o-2024-11-20 --judge_base_url https://matrixllm.alipay.com/v1/ --judge_api_key "$API_KEY"
fi
'\'''
# HLE_MODE/HLE_RID are local submitter variables; inject them into the remote command environment.
export JOB_COMMAND="HLE_MODE=$(printf '%q' "$MODE") HLE_RID=$(printf '%q' "$RID") $JOB_COMMAND"
if command -v uv >/dev/null 2>&1; then
  uv pip install "aistudio-common>=0.0.28.75" aii-pypai -i https://pypi.antfin-inc.com/simple/ >/dev/null
elif ! python -c 'import pypai, aistudio_common' >/dev/null 2>&1; then
  python -m pip install "aistudio-common>=0.0.28.75" aii-pypai -i https://pypi.antfin-inc.com/simple/ >/dev/null
fi
export LAUNCH_CONTAINER_MODE=dev_local
if [[ -n "$AISTUDIO_TOKEN" ]]; then
  aistudio_user login --name "$AISTUDIO_LOGIN_NAME" --usernumber "$AISTUDIO_USERNUMBER" --token "$AISTUDIO_TOKEN" >/dev/null
else
  echo "AISTUDIO_TOKEN not set; using existing cached login" >&2
fi
python /storage/openpsi/users/yl/agent-memory/MemRL/scripts/submit_hle_template.py

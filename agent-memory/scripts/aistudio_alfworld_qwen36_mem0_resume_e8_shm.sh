#!/bin/bash
# Qwen3.6 ALFWorld Mem0: resume E8 with /dev/shm fast_downward temp copies.
set -euo pipefail
MEMRL_DIR=/storage/openpsi/users/yl/agent-memory/MemRL
QWEN36_PATH=/storage/openpsi/models/Qwen__Qwen3.6-35B-A3B
EMBED_PATH=/storage/openpsi/models/Qwen3-Embedding-8B
RUN_STAMP="${1:-$(date +%Y%m%d_%H%M%S)}"
LOGFILE="$MEMRL_DIR/logs/aistudio_qwen36_mem0_${RUN_STAMP}.log"
WORK_TMP="/dev/shm/q36m_${RUN_STAMP}_$$"
MEM0_CKPT=/storage/openpsi/experiments/checkpoints/admin/yl-mem-region/alfworld/alfworld/exp_alfworld_mem0_qwen36_qwen36_mem0_v1/local_cache
export MEMRL_RUN_ID=qwen36_mem0_v1_resume_e8_shm
# Scope recorded in the immutable s3_b10 checkpoint (14754 memories).
export MEMRL_MEM0_USER_ID=alf_5912
# Retrieval is explicitly scoped by Mem0MemoryService.
export MEMRL_MEM0_SEARCH_ALL_SCOPES=0
export MEMRL_ALFWORLD_LLM_CONCURRENCY="${MEMRL_ALFWORLD_LLM_CONCURRENCY:-8}"
export MEMRL_LLM_CLIENT_TIMEOUT_S="${MEMRL_LLM_CLIENT_TIMEOUT_S:-600}"
export MEMRL_LLM_MAX_RETRIES="${MEMRL_LLM_MAX_RETRIES:-1}"
export MEM0_TELEMETRY=False POSTHOG_DISABLED=True
export MEMRL_MEM0_OMIT_EMBED_DIMENSIONS=1
# fast_downward copies libdownward.so into Python's tempfile directory once per
# environment instance. /tmp is CPFS quota-limited for this job; use tmpfs.
export TMPDIR="$WORK_TMP"
export TMP="$WORK_TMP"
export TEMP="$WORK_TMP"
mkdir -p "$WORK_TMP" "$(dirname "$LOGFILE")"
exec > >(tee -a "$LOGFILE") 2>&1
export HF_HOME=/storage/openpsi/users/yl/agent-memory/.cache/huggingface HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1
export FASTEMBED_CACHE_PATH="$MEMRL_DIR/scripts/fastembed_cache"
VENV_SP=/AReaL/.venv/lib/python3.12/site-packages
export FASTEMBED_SITE_PACKAGES="$VENV_SP"
export PYTHONPATH="$MEMRL_DIR:/storage/openpsi/users/yl/agent-memory/.local/lib/python3.12/site-packages:${PYTHONPATH:-}"
cd "$MEMRL_DIR"
pip install mem0ai fastembed "chonkie==1.2.1" tensorboard hdbscan pandas tqdm concurrent-log-handler textworld alfworld --target "$VENV_SP" -i https://pypi.antfin-inc.com/simple/ 2>&1 | tail -5
python3 - <<'PYIMPORT'
import pathlib, memrl
actual = pathlib.Path(memrl.__file__).resolve()
expected = pathlib.Path.cwd().joinpath("memrl").resolve()
if expected not in actual.parents:
    raise SystemExit(f"memrl imported from unexpected path: {actual}")
print(f"[OK] source memrl imported from {actual}")
PYIMPORT
python3 - <<'PYBM25CACHE'
import os
from pathlib import Path
cache = Path(os.environ["FASTEMBED_CACHE_PATH"])
snapshots = list(cache.glob("models--Qdrant--bm25/snapshots/*"))
required = ("config.json", "english.txt")
if not snapshots or not any(all((snap / name).is_file() for name in required) for snap in snapshots):
    raise SystemExit(f"Offline Qdrant/bm25 cache is incomplete: {cache}")
print(f"[OK] offline Qdrant/bm25 cache available at {cache}")
PYBM25CACHE
python3 - <<'PYFASTEMBED'
import os, sys
sys.path.append(os.environ["FASTEMBED_SITE_PACKAGES"])
import fastembed
from fastembed import SparseTextEmbedding
print(f"[OK] fastembed import available from {fastembed.__file__}")
PYFASTEMBED
MEM0_SNAPSHOT="${MEM0_CKPT}/snapshot/8"
python3 - "$MEM0_SNAPSHOT" <<'PYCHECKPOINT'
import json, pathlib, sys
root = pathlib.Path(sys.argv[1])
required = [
    root / "mem0_qdrant",
    root / "mem0_id_metadata.json",
    root / "local_cache" / "cum_state.json",
]
missing = [str(p) for p in required if not p.exists() or (p.is_file() and p.stat().st_size == 0)]
if missing or not any((root / "mem0_qdrant").rglob("*")):
    raise SystemExit("Mem0 resume checkpoint incomplete: " + ", ".join(missing))
state = json.loads((root / "local_cache" / "cum_state.json").read_text())
print(f"[CHECKPOINT] Mem0 snapshot E8 valid; global_step={state.get('global_step')}")
PYCHECKPOINT
SERVER_PIDS=()
cleanup(){ status=$?; echo "[INFO] cleanup status=$status $(date)"; ((${#SERVER_PIDS[@]})) && kill "${SERVER_PIDS[@]}" 2>/dev/null || true; wait 2>/dev/null || true; rm -rf "$WORK_TMP"; exit "$status"; }
trap cleanup EXIT INT TERM
PORT_BASE=$(python3 - <<'PY'
import os,random,socket
ps=list(range(20000,44000,3)); random.Random(f"{os.getenv('HOSTNAME')}:{os.getpid()}").shuffle(ps)
for b in ps:
 s=[]
 try:
  for p in (b,b+1,b+2): x=socket.socket(); x.bind(('0.0.0.0',p)); s.append(x)
 except OSError: pass
 else: print(b); break
 finally:
  for x in s:x.close()
else: raise SystemExit('no free ports')
PY
)
EMBED_PORT=$PORT_BASE; ACT_PORT=$((PORT_BASE+1)); EXTRACT_PORT=$((PORT_BASE+2))
echo "[PORTS] embed=$EMBED_PORT act=$ACT_PORT extract=$EXTRACT_PORT"
wait_server(){ local n=$1 p=$2 pid=$3 model=$4 limit=$5 r stable=0; for((i=1;i<=limit;i++));do kill -0 "$pid" 2>/dev/null||return 1; r=$(curl -fsS "http://127.0.0.1:$p/v1/models" 2>/dev/null||true); if R="$r" M="$model" python3 -c 'import os,json,sys;sys.exit(0 if os.environ["M"] in {x.get("id") for x in json.loads(os.environ["R"]).get("data",[])} else 1)' 2>/dev/null;then stable=$((stable+1));((stable>=5))&&{ echo "[READY] $n model=$model pid=$pid";return;};else stable=0;fi;sleep 1;done;echo "[ERROR] $n timeout";return 1; }
TMPDIR="$WORK_TMP" CUDA_VISIBLE_DEVICES=0 python3 -m vllm.entrypoints.openai.api_server --model "$EMBED_PATH" --served-model-name Qwen/Qwen3-Embedding-8B --host 127.0.0.1 --port "$EMBED_PORT" --max-model-len 8192 --trust-remote-code --convert embed & P=$!;SERVER_PIDS+=("$P");PID_E=$P
TMPDIR="$WORK_TMP" CUDA_VISIBLE_DEVICES=1 python3 -m vllm.entrypoints.openai.api_server --model "$QWEN36_PATH" --served-model-name Qwen3.6-35B-A3B --host 127.0.0.1 --port "$ACT_PORT" --max-model-len 32768 --trust-remote-code --reasoning-parser qwen3 & P=$!;SERVER_PIDS+=("$P");PID_A=$P
TMPDIR="$WORK_TMP" CUDA_VISIBLE_DEVICES=2 python3 -m vllm.entrypoints.openai.api_server --model "$QWEN36_PATH" --served-model-name Qwen3.6-35B-A3B-extract --host 127.0.0.1 --port "$EXTRACT_PORT" --max-model-len 32768 --trust-remote-code & P=$!;SERVER_PIDS+=("$P");PID_X=$P
wait_server embedding "$EMBED_PORT" "$PID_E" Qwen/Qwen3-Embedding-8B 1200
wait_server action "$ACT_PORT" "$PID_A" Qwen3.6-35B-A3B 1800
wait_server extraction "$EXTRACT_PORT" "$PID_X" Qwen3.6-35B-A3B-extract 1800
CFG="$WORK_TMP/alf_mem0.yaml"
cat > "$CFG" <<EOF
llm: {provider: openai, api_key: EMPTY, base_url: "http://localhost:${ACT_PORT}/v1/", model: Qwen3.6-35B-A3B, temperature: 0, max_tokens: 4096}
embedding: {provider: openai, api_key: EMPTY, base_url: "http://localhost:${EMBED_PORT}/v1/", model: Qwen/Qwen3-Embedding-8B, max_text_len: 8196, dimension: 4096}
memory: {build_strategy: proceduralization, retrieve_strategy: query, update_strategy: adjustment, k_retrieve: 3, max_keywords: 5, add_similarity_threshold: 0.9, memory_budget_tokens: 0, sim_norm_mean: 0.5187, sim_norm_std: 0.1203}
environment: {alfworld_config_path: configs/envs/alfworld.yaml, alfworld_env_type: AlfredTWEnv}
experiment:
  random_seed: 42
  enable_value_driven: false
  experiment_name: alfworld_mem0_qwen36
  mode: train
  num_sections: 10
  batch_size: 128
  dataset_ratio: 1.0
  few_shot_path: data/alfworld/alfworld_examples.json
  baseline_mode: null
  baseline_k: 10
  output_dir: /storage/openpsi/experiments/checkpoints/admin/yl-mem-region/alfworld
  max_steps: 30
  max_recent_turns: 20
  strip_thinking: true
  max_trajectory_len: 6000
  max_history_response_chars: 4000
  force_think: false
  save_trajectories: true
  save_memories: true
  ckpt_resume_enabled: true
  ckpt_resume_path: ${MEM0_CKPT}/snapshot/8
  ckpt_resume_epoch: null
  n_eval_runs: 4
  eval_temperature: 0.2
rl_config: {epsilon: 0, tau: 0.0, alpha: 0.3, gamma: 0.0, q_init_pos: 0, q_init_neg: 0, success_reward: 1.0, failure_reward: -1.0, topk: 3, novelty_threshold: 0.85, recency_boost: 0.0, reward_merge_gain: 0.1, q_min_threshold: -10, weight_sim: 1.0, weight_q: 0.0}
EOF
export MEMRL_MEM0_LLM_BASE_URL="http://localhost:${EXTRACT_PORT}/v1/" MEMRL_MEM0_LLM_MODEL=Qwen3.6-35B-A3B-extract
echo "[RESUME] Mem0 E8 checkpoint with /dev/shm fast_downward temp: ${MEM0_SNAPSHOT}"
TMPDIR="$WORK_TMP" TMP="$WORK_TMP" TEMP="$WORK_TMP" python3 scripts/run_alfworld_qwen36_resume.py --config "$CFG" --mem0 --skip_initial_eval

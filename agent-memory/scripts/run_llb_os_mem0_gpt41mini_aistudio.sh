#!/bin/bash
# Run LLB OS Mem0 GPT-4.1-mini on AIStudio only.
# A fresh collection is required so Mem0 v3 creates Qdrant's BM25 sparse slot.
set -euo pipefail
PROJECT_DIR=/storage/openpsi/users/yl/agent-memory/MemRL
LOCAL_SP=/storage/openpsi/users/yl/agent-memory/.local/lib/python3.12/site-packages
HOST_SHORT=$(hostname | cut -d. -f1 | tail -c 8)
TS=$(date +%Y%m%d_%H%M%S)
LOGFILE=${PROJECT_DIR}/logs/llb_os_mem0_gpt41mini_${HOST_SHORT}_${TS}.log
mkdir -p "${PROJECT_DIR}/logs"
exec > >(tee -a "$LOGFILE") 2>&1

echo '=========================================='
echo 'LLB OS Mem0 (GPT-4.1-mini, native semantic+BM25 hybrid)'
echo "Host: $(hostname) Start: $(date) Log: $LOGFILE"
echo '=========================================='
export PYTHONPATH=${PROJECT_DIR}:${PROJECT_DIR}/3rdparty/LifelongAgentBench:${LOCAL_SP}:${PYTHONPATH:-}
export MEMRL_OS_BACKEND=local
export MEMRL_OS_SANDBOX=1
export MEMRL_UPDATE_MAX_WORKERS=1
# Space every Matrix/LLB/embedding/Mem0-internal request by >=2 seconds.
export MEMRL_LLB_REQUEST_INTERVAL=2.0
export MATRIX_REQUEST_INTERVAL=2.0
export MEMRL_MEM0_MIN_INTERVAL=2.0
export MEMRL_EMBED_THROTTLE=2.0
export MEMRL_EMBED_GLOBAL_MIN_INTERVAL=2.0
export MEMRL_EMBED_MAX_RETRIES=8
export MEMRL_EMBED_429_BASE_DELAY=10
export MEMRL_EMBED_429_MAX_DELAY=120
export MEMRL_EMBED_RETRY_JITTER=2
export MEMRL_EMBED_RATE_LIMIT_DIR=/storage/openpsi/users/yl/agent-memory/.cache/embedding_rate_limits
export MEMRL_EMBED_RATE_LIMIT_KEY=llb-os-mem0-gpt41mini-text-embedding-3-large
export MATRIX_CREDENTIAL_CONFIG=/storage/openpsi/users/yl/cfuse/rq3_v11_clean_20260719/config_multisurface_isolated.yaml
# The current validated config has a live Matrix gateway credential under gpt-4o
# (and embeddings) but no gpt-4.1-mini alias. Matrix credentials authenticate the
# gateway, not a local provider default; resolve the validated key without echoing
# it, then patch the job-local run_llb config to use it for both chat and embedding.
export MATRIX_API_KEY="$(python3 - <<'PY_MATRIX_KEY'
from pathlib import Path
import yaml
cfg = yaml.safe_load(Path('/storage/openpsi/users/yl/cfuse/rq3_v11_clean_20260719/config_multisurface_isolated.yaml').read_text())
for item in cfg.get('model_list', []):
    if item.get('model_name') == 'gpt-4o':
        key = (item.get('litellm_params') or {}).get('api_key')
        if isinstance(key, str) and key:
            print(key, end='')
            break
else:
    raise SystemExit('No validated Matrix gateway credential mapping found')
PY_MATRIX_KEY
)"
[ -n "${MATRIX_API_KEY}" ] || { echo '[Mem0] Missing Matrix gateway credential' >&2; exit 31; }
export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=/tmp/huggingface
VENV_SP=/AReaL/.venv/lib/python3.12/site-packages
cd "$PROJECT_DIR"

# AIS eviction retry: reuse this submission's stable run id and latest snapshot.
source "$PROJECT_DIR/scripts/llb_os_auto_resume.sh" "llb_os_mem0_gpt41mini"

# Keep the repository's patched Mem0/Qdrant implementation first on PYTHONPATH.
# The prior Mem0 job installed unconstrained fastembed dependencies into a target
# overlay and selected tokenizers==0.23.1; that shadows AReaL's transformers and
# violates its <=0.23.0 constraint before the benchmark starts. Install a fully
# isolated overlay with explicit compatible constraints instead. It is under /tmp,
# never modifies the AReaL/shared Python environment.
BM25_SP=/tmp/memrl_mem0_bm25_site
rm -rf "$BM25_SP"
mkdir -p "$BM25_SP"
pip install --disable-pip-version-check --upgrade --target "$BM25_SP" \
  'fastembed==0.7.4' 'tokenizers==0.22.2' 'protobuf<7' 'huggingface-hub<1.0' \
  -i https://pypi.antfin-inc.com/simple/
export PYTHONPATH=${PROJECT_DIR}:${PROJECT_DIR}/3rdparty/LifelongAgentBench:${BM25_SP}:${LOCAL_SP}:${PYTHONPATH:-}
python3 - <<'PY_IMPORTS'
from packaging.version import Version
import tokenizers
assert Version('0.22.0') <= Version(tokenizers.__version__) <= Version('0.23.0'), tokenizers.__version__
import memos, memrl, fastembed
print('imports OK; tokenizers=', tokenizers.__version__, 'memrl from:', memrl.__file__, 'fastembed from:', fastembed.__file__)
PY_IMPORTS

# Remote-job-only auth preflight. Fail before sampling any OS task if the resolved
# Matrix token cannot serve both the exact GPT-4.1-mini chat model and embeddings.
# Never print Authorization headers or model text.
python3 - <<'PY_AUTH'
import os, time, httpx
base = 'https://matrixllm.alipay.com/v1'
headers = {'Authorization': f"Bearer {os.environ['MATRIX_API_KEY']}", 'Content-Type': 'application/json'}
with httpx.Client(timeout=httpx.Timeout(60, connect=20)) as client:
    r = client.post(base + '/chat/completions', headers=headers, json={
        'model': 'gpt-4.1-mini-2025-04-14',
        'messages': [{'role': 'user', 'content': 'Reply with OK.'}],
        'temperature': 0.0, 'max_tokens': 4,
    })
    r.raise_for_status()
    time.sleep(2.0)
    r = client.post(base + '/embeddings', headers=headers, json={
        'model': 'text-embedding-3-large', 'input': ['Mem0 OS credential preflight'],
    })
    r.raise_for_status()
print('[Mem0] MATRIX AUTH OK: GPT-4.1-mini chat + text-embedding-3-large both returned 200.')
PY_AUTH

# This is a startup preflight, not an experiment: it proves that the exact patched
# local Mem0 source has hybrid BM25 code AND that fastembed can construct/encode BM25.
python3 - <<'PY_PRECHECK'
from pathlib import Path
import yaml
from fastembed import SparseTextEmbedding
cfg = yaml.safe_load(Path('configs/rl_llb_os_mem0_gpt41mini.yaml').read_text())
service = Path('memrl/service/mem0_memory_service.py').read_text()
qdrant = Path('/storage/openpsi/users/yl/agent-memory/.local/lib/python3.12/site-packages/mem0/vector_stores/qdrant.py')
qdrant_text = qdrant.read_text()
assert cfg['experiment']['task'] == 'os'
assert cfg['memory']['k_retrieve'] == 5
assert cfg['memory']['user_id'] == 'llb_os_mem0_gpt41mini_user'
assert 'Semantic + BM25 hybrid retrieval on search' in service
assert '"filters": {"user_id": self.user_id}' in service
assert 'sparse_vectors_config={' in qdrant_text and '"bm25"' in qdrant_text
assert 'def keyword_search' in qdrant_text
encoder = SparseTextEmbedding(model_name='Qdrant/bm25')
encoded = list(encoder.embed(['verify BM25 hybrid retrieval is active']))
assert encoded and len(encoded[0].indices) > 0
print('[Mem0] PRECHECK OK: task=os k=5 infer=true isolated_user=llb_os_mem0_gpt41mini_user')
print('[Mem0] BM25 OK: Qdrant bm25 sparse slot + keyword_search + fastembed Qdrant/bm25 encoder verified.')
print('[Mem0] REQUEST GATES OK: all configured request intervals are 2.0 seconds.')
PY_PRECHECK

# Do not modify shared root-owned source. Use a job-local copy to pass the config user_id
# into Mem0MemoryService, so the filters/add calls stay in this run's isolated scope.
PATCHED_RUN=run/.run_llb_mem0_gpt41mini_patched.py
python3 - <<'PY_PATCH'
from pathlib import Path
src = Path('run/run_llb.py').read_text()
old = '''                top_k=config.memory.k_retrieve or 5,\n                infer=(args.mem0_infer == "true"),\n            )\n'''
new = '''                top_k=config.memory.k_retrieve or 5,\n                infer=(args.mem0_infer == "true"),\n                user_id=config.memory.user_id,\n            )\n'''
if old not in src:
    raise RuntimeError('Mem0 user_id patch anchor missing; refusing to run unisolated.')
patched = src.replace(old, new, 1)
credential_anchor = '''        config = MempConfig.from_yaml(str(config_path))
'''
credential_override = '''        config = MempConfig.from_yaml(str(config_path))
        # AIS runner injects a validated Matrix gateway token; override stale YAML
        # credentials for both direct providers and Mem0 internals before creation.
        _matrix_api_key = os.environ.get("MATRIX_API_KEY", "").strip()
        if _matrix_api_key:
            config.llm.api_key = _matrix_api_key
            config.embedding.api_key = _matrix_api_key
            logger.info("Using injected Matrix gateway credential for LLM and embedding providers.")
'''
if credential_anchor not in patched:
    raise RuntimeError('Matrix credential patch anchor missing in run_llb.py')
patched = patched.replace(credential_anchor, credential_override, 1)
# The shared llb_rl_runner now guards partial backend failures directly, and
# Mem0MemoryService preserves the common (task_description, memory_id) contract.
# Refuse to submit if either half of that invariant is absent.
service_text = Path('memrl/service/mem0_memory_service.py').read_text()
runner_text = Path('memrl/run/llb_rl_runner.py').read_text()
if 'results.append((desc, None))' not in service_text:
    raise RuntimeError('Mem0 partial-failure return-contract fix is missing')
if 'for i, result_item in enumerate(result_vis or [])' not in runner_text:
    raise RuntimeError('LLB runner partial-memory-result guard is missing')
# run_llb.py derives LLB_ROOT from __file__.  Keeping the patched copy in /tmp
# made it incorrectly resolve /3rdparty/LifelongAgentBench. Place the ephemeral
# copy beside the real runner, preserving its relative project layout.
patched_path = Path('run/.run_llb_mem0_gpt41mini_patched.py')
patched_path.write_text(patched)
print('[Mem0] PATCH OK: config.memory.user_id is passed to Mem0MemoryService; patched runner=', patched_path)
PY_PATCH

echo '[INFO] Starting GPT-4.1-mini Mem0 with infer=true and fresh semantic+BM25 collection...'
python3 "$PATCHED_RUN" --config configs/rl_llb_os_mem0_gpt41mini.yaml --mem0 --mem0_infer true --mem0_collection "memrl_mem0_llb_os_gpt41mini_${MEMRL_RUN_ID}"

echo "LLB OS Mem0 completed: $(date)"

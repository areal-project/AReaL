#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$ROOT_DIR"
source /storage/openpsi/users/zzy/.zzy-enc-2/bin/activate
python -m vllm.entrypoints.openai.api_server --host 0.0.0.0 --port 8000 --model /storage/openpsi/models/zzy/Qwen__Qwen3-4B-new-tok --tokenizer /storage/openpsi/models/zzy/Qwen__Qwen3-4B-new-tok --served-model-name qwen3-4b-newenc --tensor-parallel-size 8 --hf-overrides '{"architectures": ["Qwen3ChunkedForCausalLM"], "chunk_start_token_id": 151669, "chunk_end_token_id": 151670}' --trust-remote-code & SERVER_PID=$!
trap 'kill "$SERVER_PID" 2>/dev/null || true; wait "$SERVER_PID" 2>/dev/null || true' EXIT INT TERM
until curl -fsS http://127.0.0.1:8000/v1/models >/dev/null 2>&1; do sleep 2; done
export OPENAI_API_BASE=http://127.0.0.1:8000/v1 OPENAI_API_KEY=dummy
RESULT_FILE="$(find results/test_time_compute -type f -name 'direct_generation_imobench_AIME25_*.json' -print 2>/dev/null | xargs -r ls -1t | head -n1)"
if [[ -z "${RESULT_FILE:-}" ]]; then
	echo "Error: no AIME25 direct-generation result found under results/test_time_compute." >&2
	echo "Run scripts/run_imobench_experiment.py first, or pass a fixed file path by editing this script." >&2
	exit 2
fi
python3 -m scripts.run_aggregation_experiment --result-files "$RESULT_FILE" --output-dir results/test_time_compute/sglang_batch_with_aggregation/aggregation/4b/AIME25/new --model_name openai/qwen3-4b-newenc --pairwise-template aggregation_pairwise_new_encoding_comparison --max-concurrent 40
 
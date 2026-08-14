#!/bin/bash
# HLE MemRL baseline with gemini-3.1-pro-preview
# Resume from checkpoint exp_hle_reproduce_gemini31_20260516-154351 (964/2500 done in sec1)
# batch_size=32, streaming, full 2500 dataset (including image rows)
#
# Usage: bash scripts/run_hle_gemini31_resume.sh
#
# Before first run: fix cum_state.json next_batch for new batch_size
# (see PATCH section below)

set -e

cd /storage/openpsi/users/yl/agent-memory/MemRL

CKPT_PATH="/storage/openpsi/experiments/checkpoints/admin/yl-mem-region/cross_benchmark/hle/exp_hle_reproduce_gemini31_20260516-154351"
CUM_STATE="$CKPT_PATH/snapshot/1_b482/local_cache/cum_state.json"

# ---- PATCH: Fix next_batch for batch_size=32 (only needed on first run) ----
# Old: next_batch=483 (batch_size=2, 483*2=966 questions tracked)
# New: next_batch=30 (batch_size=32, 30*32=960 questions, runner loads 964 from batch_all_recs)
if grep -q '"next_batch": 483' "$CUM_STATE" 2>/dev/null; then
    echo "[PATCH] Fixing next_batch: 483 -> 30 for batch_size=32"
    python3 -c "
import json
with open('$CUM_STATE', 'r') as f:
    d = json.load(f)
d['next_batch'] = 30
with open('$CUM_STATE', 'w') as f:
    json.dump(d, f, ensure_ascii=False)
print(f'Patched: next_batch={d[\"next_batch\"]}, batch_all_recs={len(d.get(\"batch_all_recs\",[]))} entries')
"
fi

# ---- Rate limiter: space out requests to avoid 429 at bs=32 ----
# With 32 concurrent streaming requests, spacing by 0.3s avoids QPS bursts.
# The provider's _SendRateLimiter reads this env var.
export MEMRL_LLM_MIN_INTERVAL=0.3

# ---- Run ----
python3 run/run_hle.py \
    --config configs/rl_hle_config.gemini31_resume.yaml \
    --judge_model gpt-4o-2024-11-20 \
    --judge_base_url "https://matrixllm.alipay.com/v1/" \
    --judge_api_key "sk-43dd5f664179406d92fec42a9364f8a5"

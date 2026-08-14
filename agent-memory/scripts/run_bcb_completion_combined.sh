#!/bin/bash
# Run all missing BCB completion work serially while reusing one LLM and one
# embedding server started by aistudio_bcb_runner_strict.sh.
set -euo pipefail

MEMRL_DIR="${MEMRL_DIR:-/storage/openpsi/users/yl/agent-memory/MemRL}"
OUT_ROOT="${BCB_ALL_OUT_ROOT:-/storage/openpsi/experiments/checkpoints/admin/yl-mem-region/bigcodebench}"
STATE_DIR="${BCB_COMBINED_STATE_DIR:-$OUT_ROOT/completion_20260722_state}"
LLM_BASE_URL="${BCB_LLM_BASE_URL:?BCB_LLM_BASE_URL must be set by the parent runner}"
mkdir -p "$STATE_DIR/configs" "$STATE_DIR/logs"
cd "$MEMRL_DIR"

make_config() {
    local src="$1"
    local name="$2"
    local dst="$STATE_DIR/configs/${name}.yaml"
    cp "$src" "$dst"
    sed -i "s#http://localhost:8000/v1/#${LLM_BASE_URL}#g" "$dst"
    grep -qF "$LLM_BASE_URL" "$dst" || { echo "[FATAL] failed to set LLM URL in $dst"; exit 1; }
    printf '%s\n' "$dst"
}

run_stage() {
    local name="$1"; shift
    local done="$STATE_DIR/${name}.done"
    if [ -f "$done" ]; then
        echo "[COMBINED] SKIP completed stage: $name"
        return 0
    fi
    echo "=========================================="
    echo "[COMBINED] START stage=$name at $(date)"
    echo "[COMBINED] CMD: $*"
    echo "=========================================="
    "$@" 2>&1 | tee -a "$STATE_DIR/logs/${name}.log"
    local rc=${PIPESTATUS[0]}
    if [ "$rc" -ne 0 ]; then
        echo "[COMBINED] FAIL stage=$name rc=$rc"
        return "$rc"
    fi
    printf 'completed_at=%s\n' "$(date -Iseconds)" > "$done"
    echo "[COMBINED] DONE stage=$name at $(date)"
}

COMMON=(--split instruct --subset full --checkpoint_interval 100 --max_checkpoints 3 --eval_timeout 240 --untrusted_hard_timeout 300)
MEMP_CFG=$(make_config configs/rl_bcb_config.memp_local.yaml memp)
RAG_CFG=$(make_config configs/rl_bcb_config.rag_local.yaml rag)
MEMRL_CFG=$(make_config configs/rl_bcb_config.deepseek_v3_local.yaml memrl)
LEAF_CFG=$(make_config configs/rl_bcb_config.deepseek_v3_local_region.yaml leaf)

MEMP_E9="$OUT_ROOT/deepseek_v3_memp/bigcodebench_eval/instruct_full/memory/20260710_032817_deepseek-ai_DeepSeek-V3_rl-off/epoch9/snapshot/9"
RAG_E8="$OUT_ROOT/deepseek_v3_rag/bigcodebench_eval/instruct_full/memory/20260712_071437_deepseek-ai_DeepSeek-V3_rl-off/epoch8/snapshot/8"
MEMRL_E10="$OUT_ROOT/deepseek_v3_local_memrl/bigcodebench_eval/instruct_full/memory/20260619_165450_deepseek-ai_DeepSeek-V3_rl-on/epoch10/snapshot/10"
if [ "${BCB_LEAF_ONLY:-0}" != "1" ]; then
    if [ "${BCB_SKIP_MEMP:-0}" != "1" ]; then
        [ -f "$MEMP_E9/snapshot_meta.json" ] || { echo "[FATAL] missing source snapshot: $MEMP_E9"; exit 1; }
        run_stage memp_e10_multi python run/run_bcb.py \
            --config "$MEMP_CFG" --epochs 10 --output_dir "$OUT_ROOT/deepseek_v3_memp_completion" \
            --resume_from "$MEMP_E9" --resume_epoch 9 \
            --n_eval_runs 3 --eval_temperature 0.2 --multi_eval_epochs last "${COMMON[@]}"
    else
        echo "[COMBINED] BCB_SKIP_MEMP=1; preserving completed MemP E10 artifacts and starting at RAG."
    fi

    for p in "$RAG_E8" "$MEMRL_E10"; do
        [ -f "$p/snapshot_meta.json" ] || { echo "[FATAL] missing source snapshot: $p"; exit 1; }
    done

    run_stage rag_e9_e10_multi python run/run_bcb.py \
        --config "$RAG_CFG" --epochs 10 --output_dir "$OUT_ROOT/deepseek_v3_rag_completion" \
        --resume_from "$RAG_E8" --resume_epoch 8 \
        --n_eval_runs 3 --eval_temperature 0.2 --multi_eval_epochs last "${COMMON[@]}"

    run_stage memrl_e10_multi python scripts/run_bcb_multi_eval_only.py \
        --config "$MEMRL_CFG" --epochs 10 --output_dir "$OUT_ROOT/deepseek_v3_local_memrl_multieval" \
        --resume_from "$MEMRL_E10" --resume_epoch 10 \
        --n_eval_runs 3 --eval_temperature 0.2 --multi_eval_epochs last "${COMMON[@]}"
fi

if [ "${BCB_SKIP_LEAF:-0}" = "1" ]; then
    echo "[COMBINED] BCB_SKIP_LEAF=1; non-Leaf stages completed, leaving Leaf untouched."
    printf 'completed_at=%s\n' "$(date -Iseconds)" > "$STATE_DIR/NON_LEAF.done"
    exit 0
fi

LEAF_OUT="$OUT_ROOT/deepseek_v3_local_region_fs_leaf_splitprior_fix"
LEAF_ARGS=(
    --task_cluster_k 0
    --region_gating_mode additive
    --region_utility_mode beta
    --region_split_evidence_migration_mode soft_source_conserving
    --region_temperature 0.1
    --shrinkage_top_n 1
    --region_min_cluster_size 12
    --region_min_samples 0
    --region_cluster_selection_method leaf
    --region_max_region_share 0.30
    --region_smoothing_C 0.5
    --propagation_eta 0.12
    --propagation_k 30
    --propagation_sim_min 0.40
    --explore_schedule 0,4,3,2,2,1,1,1,1,0
    --explore_success_ratio 0.7
    --shrinkage_confidence_k 3.0
    --val_lambda_max 0.15
    --failure_summary_n_slots 1
)

# The method changed from epoch 1 onward, so this stage intentionally starts
# fresh. A platform retry may resume only from a snapshot produced by this fixed
# output root.
leaf_train() {
    local resume=()
    local best
    best=$(python - "$LEAF_OUT" <<'PY'
import glob, os, re, sys
best = (0, "")
for meta in glob.glob(os.path.join(sys.argv[1], "**/epoch*/snapshot/*/snapshot_meta.json"), recursive=True):
    d = os.path.dirname(meta)
    m = re.search(r"/epoch(\d+)/snapshot/(\d+)$", d)
    if m and m.group(1) == m.group(2) and int(m.group(1)) > best[0]:
        best = (int(m.group(1)), d)
if best[1]: print(f"{best[0]}\t{best[1]}")
PY
)
    if [ -n "$best" ]; then
        local ep snap
        ep=${best%%$'\t'*}; snap=${best#*$'\t'}
        if [ "$ep" -lt 10 ]; then
            resume=(--resume_from "$snap" --resume_epoch "$ep")
            echo "[COMBINED] Leaf fixed-run retry resumes E$ep: $snap"
        fi
    fi
    python run/run_bcb_region.py \
        --config "$LEAF_CFG" --epochs 10 --output_dir "$LEAF_OUT" \
        "${resume[@]}" "${LEAF_ARGS[@]}" "${COMMON[@]}"
}
run_stage leaf_fixed_e1_e10 leaf_train

LEAF_E10=$(python - "$LEAF_OUT" <<'PY'
import glob, os, sys
c = glob.glob(os.path.join(sys.argv[1], "**/epoch10/snapshot/10/snapshot_meta.json"), recursive=True)
if not c: raise SystemExit("no completed fixed Leaf E10 snapshot found")
print(os.path.dirname(max(c, key=os.path.getmtime)))
PY
)
run_stage leaf_fixed_e10_multi python scripts/run_bcb_region_multi_eval_only.py \
    --config "$LEAF_CFG" --epochs 10 --output_dir "$OUT_ROOT/deepseek_v3_local_region_fs_leaf_splitprior_fix_multieval" \
    --resume_from "$LEAF_E10" --resume_epoch 10 --eval_only \
    "${LEAF_ARGS[@]}" "${COMMON[@]}"

echo "[COMBINED] ALL STAGES COMPLETED at $(date)"
printf 'completed_at=%s\n' "$(date -Iseconds)" > "$STATE_DIR/ALL.done"

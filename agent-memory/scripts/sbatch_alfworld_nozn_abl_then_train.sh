#!/bin/bash
#SBATCH --job-name=yl-alf-nozn-abl-then-train
#SBATCH --partition=all
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=400G
#SBATCH --gres=gpu:5
#SBATCH --output=logs/alf_nozn_abl_then_train_%j.log
#SBATCH --error=logs/alf_nozn_abl_then_train_%j.log

# Combined sbatch: Phase 1 runs --no_z_norm ablation grid (4 cells, ~40min) on
# 928122 S7 snapshot, Phase 2 runs full region s10 training (vlmax=0.05 +
# --no_z_norm, ~66h). Shares one vLLM startup (~30min) across both phases.
#
# Background: 929089 ablation (z-norm ON) found vlmax=0.05 best (OOD +4.48pp)
# but only 13% of oracle 35pp lift released. Codex review says z-norm in
# retrieve_query absorbs region utility's absolute differences. Phase 1
# validates whether --no_z_norm releases more signal; Phase 2 trains a full
# run with the best config either way (committed: vlmax=0.05 + --no_z_norm).

MEMRL_DIR="/storage/openpsi/users/yl/agent-memory/MemRL"
SINGULARITY_IMG="/storage/openpsi/images/areal-latest.sif"
MODEL_PATH="/storage/openpsi/models/Qwen2.5-72B-Instruct"
EMBED_MODEL_PATH="/storage/openpsi/models/Qwen3-Embedding-8B"
LLM_PORT=$((8000 + ($SLURM_JOB_ID % 100) * 2))
EMBED_PORT=$((LLM_PORT + 1))

echo "=========================================="
echo "ALFWorld --no_z_norm ABLATION (Phase 1) → REGION s10 TRAINING (Phase 2)"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Start time: $(date)"
echo "=========================================="

INNER_SCRIPT=$(mktemp /tmp/alf_nozn_combined_XXXXXX.sh)
cat > "$INNER_SCRIPT" << 'INNEREOF'
#!/bin/bash

MEMRL_DIR="$1"; MODEL_PATH="$2"; LLM_PORT="$3"; EMBED_MODEL_PATH="$4"; EMBED_PORT="$5"
cd "$MEMRL_DIR"
echo "[INFO] Starting on $(hostname)"

git checkout region-dev --quiet 2>/dev/null || true
# git pull disabled: always use local working-tree code

find . -name "*.pyc" -delete 2>/dev/null || true
find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
rm -rf *.egg-info 2>/dev/null || true
pip uninstall -y memrl 2>/dev/null || true
pip install --no-cache-dir -e . --quiet 2>/dev/null || true
export PYTHONDONTWRITEBYTECODE=1
pip install memoryos memos mem0ai 'chonkie==1.2.1' tensorboard hdbscan --quiet 2>/dev/null || true
pip install vllm --quiet 2>/dev/null || true
pip install textworld alfworld --quiet 2>/dev/null || true

python -c "import memrl; print('[VERIFY] memrl loaded from:', memrl.__file__)"

export HF_HOME=/storage/openpsi/users/yl/agent-memory/.cache/huggingface

# --- Start embedding vLLM on GPU 4 ---
CUDA_VISIBLE_DEVICES=4 python -m vllm.entrypoints.openai.api_server \
    --model "$EMBED_MODEL_PATH" --served-model-name Qwen/Qwen3-Embedding-8B \
    --port "$EMBED_PORT" --max-model-len 8192 --gpu-memory-utilization 0.30 \
    --trust-remote-code --disable-log-requests --seed 42 &
EMBED_PID=$!

for i in $(seq 1 1200); do
    curl -s "http://localhost:${EMBED_PORT}/health" > /dev/null 2>&1 && echo '[INFO] Embedding vLLM is ready!' && break
    [ "$i" -eq 1200 ] && echo '[ERROR] Embedding vLLM failed' && kill $EMBED_PID 2>/dev/null && exit 1
    sleep 1
done

# --- Start LLM vLLM ---
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_IB_TIMEOUT=22
LLM_PID_FILE=/tmp/vllm_llm_pid_$$

start_llm() {
    if [ -f "$LLM_PID_FILE" ]; then
        local old_pid=$(cat "$LLM_PID_FILE")
        if kill -0 "$old_pid" 2>/dev/null; then
            kill "$old_pid" 2>/dev/null || true
            sleep 2
        fi
    fi
    CUDA_VISIBLE_DEVICES=0,1,2,3 python -m vllm.entrypoints.openai.api_server \
        --model "$MODEL_PATH" --served-model-name Qwen2.5-72B-Instruct \
        --tensor-parallel-size 4 --port "$LLM_PORT" --trust-remote-code \
        --max-model-len 32768 --gpu-memory-utilization 0.90 \
        --disable-log-requests --seed 42 --disable-frontend-multiprocessing &
    echo $! > "$LLM_PID_FILE"
    echo "[WATCHDOG] Started vLLM pid=$(cat $LLM_PID_FILE) at $(date)"
}

start_llm
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

for i in $(seq 1 1800); do
    curl -s "http://localhost:${LLM_PORT}/health" > /dev/null 2>&1 && echo '[INFO] LLM vLLM server is ready!' && break
    if ! kill -0 "$(cat $LLM_PID_FILE)" 2>/dev/null; then
        echo "[ERROR] LLM vLLM process died. Aborting."
        kill "$EMBED_PID" 2>/dev/null
        exit 1
    fi
    if [ "$i" -eq 1800 ]; then
        echo '[ERROR] LLM vLLM failed to become ready within 1800s.'
        kill "$(cat $LLM_PID_FILE)" 2>/dev/null
        kill "$EMBED_PID" 2>/dev/null
        exit 1
    fi
    sleep 1
done

# --- Watchdog covers both phases ---
WATCHDOG_KEEP_RUNNING=/tmp/watchdog_keep_$$
touch "$WATCHDOG_KEEP_RUNNING"
(
    fails=0
    while [ -f "$WATCHDOG_KEEP_RUNNING" ]; do
        if curl -s "http://localhost:${LLM_PORT}/health" > /dev/null 2>&1; then
            fails=0
        else
            fails=$((fails+1))
            echo "[WATCHDOG] health check failed ($fails/3) at $(date)"
            if [ "$fails" -ge 3 ]; then
                echo "[WATCHDOG] restarting vLLM at $(date)"
                start_llm
                restarted=0
                for w in $(seq 1 300); do
                    if curl -s "http://localhost:${LLM_PORT}/health" > /dev/null 2>&1; then
                        echo "[WATCHDOG] vLLM restarted at $(date)"
                        restarted=1; break
                    fi
                    sleep 1
                done
                [ "$restarted" -eq 1 ] && fails=0 || echo "[WATCHDOG] restart FAILED at $(date)"
            fi
        fi
        sleep 60
    done
) &
WD_PID=$!

# ============ PHASE 1: --no_z_norm ablation (4 cells, ~40min) ============
echo ""
echo "=========================================="
echo "[PHASE 1] Ablation grid with --no_z_norm (vlmax × wq on 928122 S7 snapshot)"
echo "=========================================="

for vlmax in 0.15 0.05; do
    for wq in 0.5 0.7; do
        config="configs/rl_alf_config.qwen72b_region_abl_vlmax${vlmax}_wq${wq}.yaml"
        TEMP_CONFIG=/tmp/alf_region_abl_nozn_${vlmax}_${wq}_$$.yaml
        sed "s|localhost:8000|localhost:${LLM_PORT}|g; s|localhost:8001|localhost:${EMBED_PORT}|g; s|alfworld_region_abl_vlmax${vlmax}_wq${wq}|alfworld_region_abl_nozn_vlmax${vlmax}_wq${wq}|g" \
            "$config" > "$TEMP_CONFIG"
        echo ""
        echo "[ABLATION-NOZN] val_lambda_max=${vlmax}, weight_q=${wq}, --no_z_norm"
        python run/run_alfworld.py \
            --config "$TEMP_CONFIG" --region --region_gating_mode additive \
            --shrinkage_confidence_k 3.0 --propagation_eta 0.12 \
            --val_lambda_max ${vlmax} --no_z_norm
        echo "[INFO] Cell (vlmax=${vlmax}, wq=${wq}, no_z_norm) done."
    done
done
PHASE1_EXIT=$?
echo "[INFO] Phase 1 complete with exit code: $PHASE1_EXIT"

# ============ PHASE 2: full region s10 training (vlmax=0.05 + --no_z_norm, ~66h) ============
echo ""
echo "=========================================="
echo "[PHASE 2] Region s10 training (val_lambda_max=0.05 + --no_z_norm)"
echo "explore_schedule=0,2,2,1,1,1,1,0,0,0 (soft) + additive + confgate"
echo "=========================================="

TRAIN_CONFIG=/tmp/alf_region_s10_vlmax005_nozn_train_$$.yaml
sed "s|localhost:8000|localhost:${LLM_PORT}|g; s|localhost:8001|localhost:${EMBED_PORT}|g" \
    configs/rl_alf_config.qwen72b_region_10section_softexplore_vlmax005_nozn.yaml > "$TRAIN_CONFIG"

python run/run_alfworld.py \
    --config "$TRAIN_CONFIG" --region --region_gating_mode additive \
    --shrinkage_confidence_k 3.0 --propagation_eta 0.12 --val_lambda_max 0.05 \
    --explore_schedule "0,2,2,1,1,1,1,0,0,0" \
    --no_z_norm \
    --skip_initial_eval &
MAIN_PID=$!
wait "$MAIN_PID"; PHASE2_EXIT=$?
echo "[INFO] Phase 2 (training) exited with code: $PHASE2_EXIT"

# Cleanup
rm -f "$WATCHDOG_KEEP_RUNNING"
kill "$WD_PID" 2>/dev/null; wait "$WD_PID" 2>/dev/null
[ -f "$LLM_PID_FILE" ] && kill "$(cat $LLM_PID_FILE)" 2>/dev/null && rm -f "$LLM_PID_FILE"
kill "$EMBED_PID" 2>/dev/null; wait "$EMBED_PID" 2>/dev/null
echo '[INFO] Done. Phase1 exit=' "$PHASE1_EXIT" ' Phase2 exit=' "$PHASE2_EXIT"
INNEREOF

chmod +x "$INNER_SCRIPT"

singularity exec --nv --no-home --writable-tmpfs \
    --bind /storage:/storage \
    $SINGULARITY_IMG \
    bash "$INNER_SCRIPT" "$MEMRL_DIR" "$MODEL_PATH" "$LLM_PORT" "$EMBED_MODEL_PATH" "$EMBED_PORT"

rm -f "$INNER_SCRIPT"

echo "=========================================="
echo "End time: $(date)"
echo "Exit code: $?"
echo "=========================================="

#!/bin/bash
# MemRL Experiment Runner - Submit to compute node via srun
# 直接调用 matrixllm API，无需 LiteLLM 代理
# Usage: bash scripts/run_memrl_srun.sh [bcb|alf|hle|llb_os|llb_db]

set -e

MEMRL_DIR="/storage/openpsi/users/yl/agent-memory/MemRL"
SINGULARITY_IMG="/storage/openpsi/images/areal-latest.sif"
EXPERIMENT=${1:-bcb}
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

cd $MEMRL_DIR
mkdir -p logs

case "$EXPERIMENT" in
    bcb)
        JOB_NAME="yl-memrl-bcb"
        CONFIG_FILE="configs/rl_bcb_config.local.yaml"
        RUN_CMD="python run/run_bcb.py --config \${CONFIG_FILE} --split instruct --subset full --epochs 10"
        ;;
    alf)
        JOB_NAME="yl-memrl-alf"
        CONFIG_FILE="configs/rl_alf_config.local.yaml"
        RUN_CMD="python run/run_alfworld.py --config \${CONFIG_FILE}"
        ;;
    hle)
        JOB_NAME="yl-memrl-hle"
        CONFIG_FILE="configs/rl_hle_config.local.yaml"
        RUN_CMD="python run/run_hle.py --config \${CONFIG_FILE} --train data/hle/hle_test.parquet"
        ;;
    hle31)
        JOB_NAME="yl-memrl-hle31"
        CONFIG_FILE="configs/rl_hle_config.gemini31.yaml"
        RUN_CMD="python run/run_hle.py --config \${CONFIG_FILE} --train data/hle/hle_test.parquet"
        ;;
    llb_os)
        JOB_NAME="yl-llb-memrl"
        CONFIG_FILE="configs/rl_llb_os_config.local.yaml"
        RUN_CMD="python run/run_llb.py --config \${CONFIG_FILE}"
        ;;
    llb_db)
        JOB_NAME="yl-memrl-llb-db"
        CONFIG_FILE="configs/rl_llb_db_config.local.yaml"
        RUN_CMD="python run/run_llb.py --config \${CONFIG_FILE}"
        ;;
    *)
        echo "Usage: $0 [bcb|alf|hle|llb_os|llb_db]"
        echo "  bcb    - BigCodeBench (gpt-4o), target: 0.595/0.627"
        echo "  alf    - ALFWorld (gpt-5-mini), target: 0.949/0.981"
        echo "  hle    - HLE (gemini-3-pro-image-preview), target: 0.570/0.606"
        echo "  llb_os - LLB OS Task (gpt-4o), target: 0.788/0.804"
        echo "  llb_db - LLB DB Task (gpt-4o), target: 0.960/0.972"
        exit 1
        ;;
esac

LOG_FILE="logs/${EXPERIMENT}_${TIMESTAMP}.log"

echo "=========================================="
echo "Submitting $EXPERIMENT experiment"
echo "Job name: $JOB_NAME"
echo "Log file: $LOG_FILE"
echo "=========================================="

# 完整的运行脚本 - 在计算节点内执行
read -r -d '' FULL_SCRIPT << 'SCRIPT_EOF' || true
#!/bin/bash
# Note: removed set -e to allow pip warnings without failing

MEMRL_DIR="__MEMRL_DIR__"
CONFIG_FILE="__CONFIG_FILE__"
EXPERIMENT="__EXPERIMENT__"

cd ${MEMRL_DIR}

echo "[INFO] Starting ${EXPERIMENT} on $(hostname)"

# 1. 设置HuggingFace镜像
export HF_ENDPOINT="https://hf-mirror.com"
export HF_HOME="/tmp/huggingface"

# 2. 安装依赖（不安装litellm，直接调用matrixllm API）
echo "[INFO] Installing dependencies..."
pip install -e . --quiet || echo "Warning: pip install -e . failed"
pip install memoryos memos mem0ai 'chonkie==1.2.1' tensorboard docker --quiet || echo "Warning: pip install deps failed"

# 2. 直接运行实验（不启动LiteLLM代理，直接调用matrixllm API）
echo "[INFO] Starting ${EXPERIMENT} experiment..."
__RUN_CMD__

echo "[INFO] ${EXPERIMENT} experiment completed!"
SCRIPT_EOF

# 替换占位符
FULL_SCRIPT="${FULL_SCRIPT//__MEMRL_DIR__/$MEMRL_DIR}"
FULL_SCRIPT="${FULL_SCRIPT//__CONFIG_FILE__/$CONFIG_FILE}"
FULL_SCRIPT="${FULL_SCRIPT//__EXPERIMENT__/$EXPERIMENT}"
FULL_SCRIPT="${FULL_SCRIPT//__RUN_CMD__/$RUN_CMD}"

# 提交到计算节点
srun --mpi=pmi2 \
    --job-name=$JOB_NAME \
    --ntasks=1 \
    --gres=gpu:0 \
    --chdir=$MEMRL_DIR \
    --cpus-per-task=8 \
    --mem=64G \
    singularity exec --nv --no-home --writable-tmpfs \
    --bind /storage:/storage \
    $SINGULARITY_IMG \
    bash -c "$FULL_SCRIPT" \
    2>&1 | tee $LOG_FILE

echo "Experiment $EXPERIMENT completed! Log saved to $LOG_FILE"

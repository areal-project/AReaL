#!/bin/bash
# ============================================================================
# BCB Region+FS leaf: gpt-4o-2024-11-20 + text-embedding-3-large (MatrixLLM API)
# RESUME E5-E10 from the E4 snapshot (run 20260705_161843 stopped at E4).
# Pure API — no GPU server needed; gpu_num=1 only for scheduling to /storage node.
# ============================================================================
export AISTUDIO_LOGIN_NAME="aistudio"
export AISTUDIO_USERNUMBER="477578"
export AISTUDIO_TOKEN="7371e433-4755-44b1-b410-319ab4024990"

export WORKER_NUM="0"
export JOB_TAG=""
TS=$(date +%Y%m%d-%H%M%S)
export JOB_NAME="${JOB_NAME:-yl-bcb-gpt4o-leaf-resume-${TS}}"
RUN_ID="${JOB_NAME}"
export KM_IMAGE="acr-sh-ant-registry-vpc.cn-shanghai.cr.aliyuncs.com/gpu/areal-runtime:dev-sglang-20260401"

# E4 snapshot of the original leaf run (verified: cube/qdrant/local_cache/meta present)
RESUME_SNAP="/storage/openpsi/experiments/checkpoints/admin/yl-mem-region/bigcodebench/gpt4o_region_fs_leaf/bigcodebench_eval/instruct_full/region/20260705_161843_gpt-4o-2024-11-20_region/epoch4/snapshot/4"

export JOB_COMMAND="bash -c '
set -e
LOGFILE=/storage/openpsi/users/yl/agent-memory/MemRL/logs/aistudio_bcb_gpt4o_leaf_resume_${RUN_ID}.log
exec > >(tee -a \$LOGFILE) 2>&1

echo \"=== Start: \$(date) ===\"
cd /storage/openpsi/users/yl/agent-memory/MemRL

VENV_SP=/AReaL/.venv/lib/python3.12/site-packages
pip install mem0ai \"chonkie==1.2.1\" tensorboard pandas tqdm concurrent-log-handler ollama hdbscan --target \$VENV_SP -i https://pypi.antfin-inc.com/simple/ 2>&1 | tail -3

export PYTHONPATH=/storage/openpsi/users/yl/agent-memory/MemRL:/storage/openpsi/users/yl/agent-memory/.local/lib/python3.12/site-packages:\$PYTHONPATH
export HF_HOME=/storage/openpsi/users/yl/agent-memory/.cache/huggingface
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
# API rate limits — larger intervals since many API jobs are running concurrently
# (avoid 429s on MatrixLLM). Overridable via env before calling this script.
export MEMRL_LLM_MIN_INTERVAL=\"\${MEMRL_LLM_MIN_INTERVAL:-6.0}\"
export MEMRL_EMBED_MIN_INTERVAL=\"\${MEMRL_EMBED_MIN_INTERVAL:-4.0}\"
export MEMRL_UPDATE_MAX_WORKERS=\"\${MEMRL_UPDATE_MAX_WORKERS:-1}\"
python3 -c \"import memos; import memrl; print(\\\"imports OK:\\\", memrl.__file__)\"

# BCB eval-time deps (numba==0.55 will fail on py3.12, harmless; others install)
pip install -r 3rdparty/bigcodebench-main/Requirements/requirements.txt --target \$VENV_SP -i https://pypi.antfin-inc.com/simple/ 2>&1 | tail -2 || true
pip install -r 3rdparty/bigcodebench-main/Requirements/requirements-eval.txt --target \$VENV_SP -i https://pypi.antfin-inc.com/simple/ 2>&1 | tail -2 || true
pip install faker statsmodels xlwt python-docx sendgrid openpyxl xlrd seaborn pyarrow shapely geopandas folium \
    django scikit-image pyquery geopy xmltodict Flask-Mail flask_login pyfakefs texttable textblob gensim \
    pytesseract holidays pycryptodome mechanize wikipedia wordcloud wordninja requests_mock python-Levenshtein \
    natsort librosa Flask-WTF flask-restful --target \$VENV_SP -i https://pypi.antfin-inc.com/simple/ 2>&1 | tail -3 || true

echo \"=== [gpt4o leaf RESUME] E5-E10 from E4 snapshot ===\"
python3 run/run_bcb_region.py \
    --config configs/rl_bcb_config.gpt4o_region.yaml \
    --split instruct --subset full --epochs 10 \
    --split_file configs/bigcodebench/splits/full_seed42_dlp_clean_ids.json \
    --checkpoint_interval 100 --max_checkpoints 3 \
    --eval_timeout 240 --untrusted_hard_timeout 300 \
    --task_cluster_k 0 \
    --region_gating_mode additive --region_utility_mode beta \
    --region_temperature 0.1 --shrinkage_top_n 1 --region_min_cluster_size 12 \
    --region_min_samples 0 --region_cluster_selection_method leaf \
    --region_max_region_share 0.30 \
    --region_smoothing_C 0.5 --propagation_eta 0.12 --propagation_k 30 \
    --propagation_sim_min 0.40 --explore_schedule 0,4,3,2,2,1,1,1,1,0 \
    --explore_success_ratio 0.7 --shrinkage_confidence_k 3.0 --val_lambda_max 0.15 \
    --failure_summary_n_slots 1 \
    --resume_from ${RESUME_SNAP} --resume_epoch 4 \
    --output_dir /storage/openpsi/experiments/checkpoints/admin/yl-mem-region/bigcodebench/gpt4o_region_fs_leaf

echo \"=== Done: \$(date) ===\"
'"

pip install "aistudio-common>=0.0.28.75" -i https://pypi.antfin-inc.com/simple/ >/dev/null 2>&1
pip install aii-pypai -i https://pypi.antfin-inc.com/simple/ >/dev/null 2>&1
export LAUNCH_CONTAINER_MODE=dev_local
aistudio_user login --name "${AISTUDIO_LOGIN_NAME}" --usernumber "${AISTUDIO_USERNUMBER}" --token "${AISTUDIO_TOKEN}"
python /storage/openpsi/users/yl/agent-memory/MemRL/scripts/submit_hle_template.py

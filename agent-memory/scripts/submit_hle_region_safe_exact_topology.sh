#!/usr/bin/env bash
set -euo pipefail
ROOT=/storage/openpsi/users/yl/agent-memory
RID="yl-hle-region-safe-exact-topology-$(date +%Y%m%d-%H%M%S)"
SOURCE_CKPT=/storage/openpsi/experiments/checkpoints/admin/yl-mem-region/hle/exp_hle_group1_region_trajectory_yl-hle-region-selfrag-two-20260727-003308-regiontraj/snapshot/1_incomplete_final
REGION_CKPT="$SOURCE_CKPT"
echo "[submit] intentional clean S2 fork source: $REGION_CKPT"
REMOTE="/tmp/${RID}.sh"
cat > "$REMOTE" <<'REMOTE'
#!/usr/bin/env bash
set -euo pipefail
ROOT=/storage/openpsi/users/yl/agent-memory
cd "$ROOT/MemRL"
LOG="$ROOT/MemRL/logs/aistudio_hle_region_safe_exact_topology_${HLE_RID}.log"
exec > >(tee -a "$LOG") 2>&1
SITE="/tmp/hle_region_safe_site_${HLE_RID}"
rm -rf "$SITE"; mkdir -p "$SITE"
pip install --target "$SITE" -i https://pypi.antfin-inc.com/simple/ tensorboard pandas tqdm concurrent-log-handler hdbscan >/tmp/hle_region_safe_deps.log 2>&1
export MEMRL_HLE_SAFE_SOURCE="$ROOT/MemRL"
export PYTHONPATH="$ROOT/MemRL/scripts/hle_region_safe_overlay:$SITE:$ROOT/MemRL:$ROOT/.local/lib/python3.12/site-packages:${PYTHONPATH:-}"
export HF_HOME="$ROOT/.cache/huggingface" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
unset MEMRL_LLM_GLOBAL_MIN_INTERVAL MEMRL_EMBED_GLOBAL_MIN_INTERVAL MEMRL_LLM_RATE_LIMIT_DIR MEMRL_EMBED_RATE_LIMIT_DIR MEMRL_LLM_RATE_LIMIT_KEY MEMRL_EMBED_RATE_LIMIT_KEY MEMRL_HLE_TIMEOUT_POLICY
export MEMRL_UPDATE_MAX_WORKERS=4 MEMRL_RUNNER_MAX_RETRIES=1 MEMRL_LLM_CLIENT_TIMEOUT_S=600 MEMRL_LLM_GEN_TIMEOUT_S=600
export MEMRL_EMBED_REQUEST_TIMEOUT_S=30 MEMRL_EMBED_SDK_MAX_RETRIES=0 MEMRL_EMBED_CONNECTION_MAX_RETRIES=2 MEMRL_EMBED_CONNECTION_RETRY_BUDGET_S=75 MEMRL_EMBED_CONNECTION_BASE_DELAY=1
export MEMRL_RESUME_PENDING_TIMEOUT_S=600 MEMRL_INFRA_RETRY_ROUNDS=3 MEMRL_INFRA_RETRY_WAIT_S=240
export MEMRL_LLM_GLOBAL_MAX_INFLIGHT=64 MEMRL_LLM_INFLIGHT_DIR=/storage/openpsi/experiments/checkpoints/admin/yl-mem-region/.llm_inflight MEMRL_LLM_INFLIGHT_KEY=hle-region-selfrag-gemini35flash
# On every platform/container restart, prefer this safe run's own newest checkpoint.
# The explicit S1 source is only a fallback before the fork has produced a checkpoint.
SAFE_GLOB="/storage/openpsi/experiments/checkpoints/admin/yl-mem-region/hle/exp_hle_regionfs_exact_topostable_${HLE_RID}-regiontraj/snapshot/*"
SOURCE_CKPT=/storage/openpsi/experiments/checkpoints/admin/yl-mem-region/hle/exp_hle_group1_region_trajectory_yl-hle-region-selfrag-two-20260727-003308-regiontraj/snapshot/1_incomplete_final
REGION_CKPT=$(python3 "$ROOT/MemRL/scripts/select_latest_hle_checkpoint.py" --glob "$SAFE_GLOB" --glob "$SOURCE_CKPT" --max-no-id 1 --require-region-manager)
echo "[remote-start] dynamically selected Region safe checkpoint: $REGION_CKPT"
echo "[HLE REGION SAFE] rid=$HLE_RID interval=2.5s exact_anchor=true FS=1 exploration=off topology_mid=1600 max_actual_changes_per_section=1 min_change_gap=2000"
python3 -c "import sys; sys.path.insert(0, '/storage/openpsi/users/yl/agent-memory/MemRL/scripts/hle_region_safe_overlay'); import memrl.service.region_memory_service as sm, memrl.run.hle_region_runner as rm; assert 'hle_region_safe_overlay' in sm.__file__, sm.__file__; assert 'hle_region_safe_overlay' in rm.__file__, rm.__file__; assert hasattr(sm.RegionMemoryService, '_apply_exact_success_anchor'); print('[HLE REGION SAFE] OVERLAY_PRECHECK_OK', sm.__file__, rm.__file__)"
CFG="/tmp/hle_region_safe_${HLE_RID}.yaml"
export HLE_CKPT="$REGION_CKPT" CFG
python3 - <<'PY'
import os,yaml
from pathlib import Path
cred=yaml.safe_load(Path('/storage/openpsi/users/yl/cfuse/rq3_v11_clean_20260719/config_multisurface_isolated.yaml').read_text())
keys={x['model_name']:x['litellm_params']['api_key'] for x in cred['model_list']}
assert all(keys.get(x) for x in ('gemini-3.5-flash','text-embedding-3-large','gpt-4o-2024-11-20'))
cfg=yaml.safe_load(Path('configs/rl_hle_config.region_gemini35flash_conservative.yaml').read_text())
cfg['memory'].update(build_strategy='trajectory',user_id='hle_region_exact_topostable_user')
cfg['experiment'].update(experiment_name='hle_regionfs_exact_topostable',ckpt_resume_enabled=True,ckpt_resume_path=os.environ['HLE_CKPT'],ckpt_resume_epoch=0,ckpt_resume_prefer_current_run=True,ckpt_save_every_n_batches=5)
cfg['llm']['api_key']=keys['gemini-3.5-flash']; cfg['embedding']['api_key']=keys['text-embedding-3-large']
yaml.safe_dump(cfg,open(os.environ['CFG'],'w'),sort_keys=False)
Path('/tmp/judge_'+os.environ['HLE_RID']).write_text(keys['gpt-4o-2024-11-20'])
PY
JUDGE=$(cat "/tmp/judge_${HLE_RID}"); rm -f "/tmp/judge_${HLE_RID}"
export MEMRL_RUN_ID="${HLE_RID}-regiontraj" MEMRL_REASONING_EFFORT=high MEMRL_LLM_MIN_INTERVAL=2.5 MEMRL_EMBED_MIN_INTERVAL=2.5
python3 "$ROOT/MemRL/scripts/hle_region_safe_overlay/run_hle_region.py" \
  --config "$CFG" --train data/hle/hle_test.parquet \
  --judge_model gpt-4o-2024-11-20 --judge_base_url https://matrixllm.alipay.com/v1/ --judge_api_key "$JUDGE" \
  --region_gating_mode additive --region_retrieve_mode global --k_global 30 --k_local 10 \
  --shrinkage_top_n 1 --shrinkage_lambda_max 0.9 --shrinkage_confidence_k 0.5 \
  --region_temperature 0.025 --region_utility_mode beta --region_smoothing_C 0.5 \
  --region_cluster_init_step 500 --region_merge_interval 400 \
  --topology_mid_section_step 1600 --topology_min_change_gap 2000 --protect_exact_success_memory \
  --propagation_eta 0.03 --propagation_k 10 --propagation_sim_min 0.60 \
  --explore_schedule '0,0,0,0,0,0,0,0,0,0' --failure_summary_n_slots 1 \
  --region_split_evidence_migration_mode soft_source_conserving
REMOTE
PAYLOAD=$(base64 -w0 "$REMOTE")
export PYTHONPATH=/tmp/yl_pypai AISTUDIO_LOGIN_NAME=aistudio AISTUDIO_USERNUMBER=477578 WORKER_NUM=0 JOB_TAG='' JOB_NAME="$RID" KM_IMAGE='acr-sh-ant-registry-vpc.cn-shanghai.cr.aliyuncs.com/gpu/areal-runtime:dev-sglang-20260401'
export JOB_COMMAND="HLE_RID=$(printf '%q' "$RID") bash -c $(printf '%q' "echo $PAYLOAD | base64 -d | bash")"
export LAUNCH_CONTAINER_MODE=dev_local PYPAI_HOME="/tmp/pypai_${RID}" TMPDIR=/tmp
cd /tmp
python3 "$ROOT/MemRL/scripts/submit_hle_template.py"

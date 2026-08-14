#!/usr/bin/env bash
set -euo pipefail
ROOT=/storage/openpsi/users/yl/agent-memory
RID="yl-hle-region-only-$(date +%Y%m%d-%H%M%S)"
REGION_CKPT=$(python3 "$ROOT/MemRL/scripts/select_latest_hle_checkpoint.py" --glob '/storage/openpsi/experiments/checkpoints/admin/yl-mem-region/hle/exp_hle_group1_region_trajectory_*/snapshot/*' --max-no-id 1 --require-region-manager)
echo "[submit] selected Region checkpoint: $REGION_CKPT"
REMOTE="/tmp/${RID}.sh"
cat > "$REMOTE" <<'REMOTE'
#!/usr/bin/env bash
set -euo pipefail
ROOT=/storage/openpsi/users/yl/agent-memory
cd "$ROOT/MemRL"
LOG="$ROOT/MemRL/logs/aistudio_hle_region_only_${HLE_RID}.log"
exec > >(tee -a "$LOG") 2>&1
STATE="/tmp/hle_region_only_${HLE_RID}"
mkdir -p "$STATE"
SITE="/tmp/hle_region_only_site_${HLE_RID}"
rm -rf "$SITE"; mkdir -p "$SITE"
pip install --target "$SITE" -i https://pypi.antfin-inc.com/simple/ tensorboard pandas tqdm concurrent-log-handler hdbscan >/tmp/hle_group1_deps.log 2>&1
export PYTHONPATH="$SITE:$ROOT/MemRL:$ROOT/.local/lib/python3.12/site-packages:${PYTHONPATH:-}"
export HF_HOME="$ROOT/.cache/huggingface" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
unset MEMRL_LLM_GLOBAL_MIN_INTERVAL MEMRL_EMBED_GLOBAL_MIN_INTERVAL MEMRL_LLM_RATE_LIMIT_DIR MEMRL_EMBED_RATE_LIMIT_DIR MEMRL_LLM_RATE_LIMIT_KEY MEMRL_EMBED_RATE_LIMIT_KEY MEMRL_HLE_TIMEOUT_POLICY
export MEMRL_UPDATE_MAX_WORKERS=4 MEMRL_RUNNER_MAX_RETRIES=1 MEMRL_LLM_CLIENT_TIMEOUT_S=600 MEMRL_LLM_GEN_TIMEOUT_S=600
export MEMRL_EMBED_REQUEST_TIMEOUT_S=30 MEMRL_EMBED_SDK_MAX_RETRIES=0 MEMRL_EMBED_CONNECTION_MAX_RETRIES=2 MEMRL_EMBED_CONNECTION_RETRY_BUDGET_S=75 MEMRL_EMBED_CONNECTION_BASE_DELAY=1
export MEMRL_RESUME_PENDING_TIMEOUT_S=600
export MEMRL_INFRA_RETRY_ROUNDS=3 MEMRL_INFRA_RETRY_WAIT_S=240
export MEMRL_LLM_GLOBAL_MAX_INFLIGHT=64 MEMRL_LLM_INFLIGHT_DIR=/storage/openpsi/experiments/checkpoints/admin/yl-mem-region/.llm_inflight MEMRL_LLM_INFLIGHT_KEY=hle-region-selfrag-gemini35flash
python3 -c "import hdbscan; print('[HLE group1] HDBSCAN_PRECHECK_OK', getattr(hdbscan, '__version__', 'installed'))"
echo "[HLE region-only] rid=$HLE_RID RegionTraj=2.0s update_workers=4 shared_inflight=64"
launch(){
 local mode="$1" delay="$2" interval="$3" ckpt="$4" child="$5"
 (
  sleep "$delay"
  LOG="$ROOT/MemRL/logs/aistudio_hle_${mode}_gemini35flash_${child}.log"
  exec > >(tee -a "$LOG") 2>&1
  export HLE_MODE="$mode" HLE_CKPT="$ckpt" HLE_CHILD="$child" CFG="/tmp/hle_${child}.yaml"
  echo "[HLE region-only child] mode=$mode delay=${delay}s interval=${interval}s ckpt=$ckpt update_workers=$MEMRL_UPDATE_MAX_WORKERS"
  python3 - <<'PY'
import os,yaml
from pathlib import Path
cred=yaml.safe_load(Path('/storage/openpsi/users/yl/cfuse/rq3_v11_clean_20260719/config_multisurface_isolated.yaml').read_text())
keys={x['model_name']:x['litellm_params']['api_key'] for x in cred['model_list']}
assert all(keys.get(x) for x in ('gemini-3.5-flash','text-embedding-3-large','gpt-4o-2024-11-20'))
m=os.environ['HLE_MODE']
if m in ('rag','selfrag'):
    src={'rag':'configs/rl_hle_config.rag_gemini35flash_resume_1b9.yaml','selfrag':'configs/rl_hle_config.selfrag_gemini35flash_resume_1b9.yaml'}[m]
    cfg=yaml.safe_load(Path(src).read_text())
    cfg['experiment'].update(experiment_name='hle_group1_'+m,ckpt_resume_enabled=True,ckpt_resume_path=os.environ['HLE_CKPT'],ckpt_resume_epoch=0,ckpt_resume_prefer_current_run=True,ckpt_save_every_n_batches=5)
else:
    cfg=yaml.safe_load(Path('configs/rl_hle_config.region_gemini35flash_conservative.yaml').read_text())
    cfg['memory'].update(build_strategy='trajectory',user_id='hle_region_trajectory_user')
    cfg['experiment'].update(experiment_name='hle_group1_region_trajectory',ckpt_resume_enabled=True,ckpt_resume_path=os.environ['HLE_CKPT'],ckpt_resume_epoch=0,ckpt_resume_prefer_current_run=True,ckpt_save_every_n_batches=5)
cfg['llm']['api_key']=keys['gemini-3.5-flash']
cfg['embedding']['api_key']=keys['text-embedding-3-large']
yaml.safe_dump(cfg,open(os.environ['CFG'],'w'),sort_keys=False)
Path('/tmp/judge_'+os.environ['HLE_CHILD']).write_text(keys['gpt-4o-2024-11-20'])
PY
  JUDGE=$(cat "/tmp/judge_${child}"); rm -f "/tmp/judge_${child}"
  export MEMRL_RUN_ID="$child" MEMRL_REASONING_EFFORT=high MEMRL_LLM_MIN_INTERVAL="$interval" MEMRL_EMBED_MIN_INTERVAL="$interval"
  if [ "$mode" = selfrag ]; then
    python3 run/run_hle.py --config "$CFG" --train data/hle/hle_test.parquet --self_rag --self_rag_inject_k 3 --judge_model gpt-4o-2024-11-20 --judge_base_url https://matrixllm.alipay.com/v1/ --judge_api_key "$JUDGE"
  elif [ "$mode" = regiontraj ]; then
    echo "[HLE region-only] HDBSCAN_REQUIRED=true checkpointV2=enabled"
    python3 run/run_hle_region.py --config "$CFG" --train data/hle/hle_test.parquet --judge_model gpt-4o-2024-11-20 --judge_base_url https://matrixllm.alipay.com/v1/ --judge_api_key "$JUDGE" --region_gating_mode additive --region_retrieve_mode global --k_global 30 --k_local 10 --shrinkage_top_n 1 --shrinkage_lambda_max 0.9 --shrinkage_confidence_k 0.5 --region_temperature 0.025 --region_utility_mode beta --region_smoothing_C 0.5 --region_cluster_init_step 500 --region_merge_interval 400 --propagation_eta 0.03 --propagation_k 10 --propagation_sim_min 0.60 --explore_schedule '0,4,3,2,2,1,1,1,0' --failure_summary_n_slots 1 --region_split_evidence_migration_mode soft_source_conserving
  else
    python3 run/run_hle.py --config "$CFG" --train data/hle/hle_test.parquet --judge_model gpt-4o-2024-11-20 --judge_base_url https://matrixllm.alipay.com/v1/ --judge_api_key "$JUDGE"
  fi
 ) &
 echo "$! $mode $child" >> "$STATE/children"
}
launch regiontraj 0 2.0 "$REGION_CKPT" "${HLE_RID}-regiontraj"
status=0
while read -r pid mode child; do wait "$pid" || status=1; done < "$STATE/children"
exit "$status"
REMOTE
PAYLOAD=$(base64 -w0 "$REMOTE")
export PYTHONPATH=/tmp/yl_pypai AISTUDIO_LOGIN_NAME=aistudio AISTUDIO_USERNUMBER=477578 WORKER_NUM=0 JOB_TAG='' JOB_NAME="$RID" KM_IMAGE='acr-sh-ant-registry-vpc.cn-shanghai.cr.aliyuncs.com/gpu/areal-runtime:dev-sglang-20260401'
export JOB_COMMAND="HLE_RID=$(printf '%q' "$RID") REGION_CKPT=$(printf '%q' "$REGION_CKPT") bash -c $(printf '%q' "echo $PAYLOAD | base64 -d | bash")"
export LAUNCH_CONTAINER_MODE=dev_local PYPAI_HOME="/tmp/pypai_${RID}" TMPDIR=/tmp
cd /tmp
python3 "$ROOT/MemRL/scripts/submit_hle_template.py"

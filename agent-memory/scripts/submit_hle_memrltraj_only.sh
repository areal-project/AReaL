#!/usr/bin/env bash
set -euo pipefail
ROOT=/storage/openpsi/users/yl/agent-memory
RID="yl-hle-memrltraj-only-$(date +%Y%m%d-%H%M%S)"
MEMRL_CKPT=$(python3 "$ROOT/MemRL/scripts/select_latest_hle_checkpoint.py" --glob '/storage/openpsi/experiments/checkpoints/admin/yl-mem-region/hle/exp_hle_memrltraj_concurrent_control_*/snapshot/*' --max-no-id 0)
echo "[submit] selected MemRL trajectory checkpoint: $MEMRL_CKPT"
REMOTE="/tmp/${RID}.sh"
cat > "$REMOTE" <<'REMOTE'
#!/usr/bin/env bash
set -euo pipefail
ROOT=/storage/openpsi/users/yl/agent-memory
cd "$ROOT/MemRL"
LOG="$ROOT/MemRL/logs/aistudio_hle_memrltraj_only_${HLE_RID}.log"
exec > >(tee -a "$LOG") 2>&1
STATE="/tmp/hle_memrltraj_only_${HLE_RID}"
mkdir -p "$STATE"
SITE="/tmp/hle_memrltraj_only_site_${HLE_RID}"
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
# Re-evaluate on every remote container start. AIStudio retries re-run this script,
# so a checkpoint chosen only by the local submitter can become stale.
MEMRL_CKPT=$(python3 "$ROOT/MemRL/scripts/select_latest_hle_checkpoint.py" --glob '/storage/openpsi/experiments/checkpoints/admin/yl-mem-region/hle/exp_hle_memrltraj_concurrent_control_*/snapshot/*' --max-no-id 0)
echo "[remote-start] dynamically selected MemRL trajectory checkpoint: $MEMRL_CKPT"
echo "[HLE memrltraj-only] rid=$HLE_RID MemRLTraj=4.0s update_workers=4 shared_inflight=64"
launch(){
 local mode="$1" delay="$2" interval="$3" ckpt="$4" child="$5"
 (
  sleep "$delay"
  LOG="$ROOT/MemRL/logs/aistudio_hle_${mode}_gemini35flash_${child}.log"
  exec > >(tee -a "$LOG") 2>&1
  export HLE_MODE="$mode" HLE_CKPT="$ckpt" HLE_CHILD="$child" CFG="/tmp/hle_${child}.yaml"
  echo "[HLE memrltraj-only child] mode=$mode delay=${delay}s interval=${interval}s ckpt=$ckpt update_workers=$MEMRL_UPDATE_MAX_WORKERS"
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
    cfg=yaml.safe_load(Path('configs/rl_hle_config.memrl_gemini35flash.yaml').read_text())
    cfg['memory'].update(build_strategy='trajectory',user_id='hle_memrl_trajectory_gemini35flash_user')
    cfg['experiment'].update(experiment_name='hle_memrltraj_concurrent_control',ckpt_resume_enabled=True,ckpt_resume_path=os.environ['HLE_CKPT'],ckpt_resume_epoch=0,ckpt_resume_prefer_current_run=True,ckpt_save_every_n_batches=5)
cfg['llm']['api_key']=keys['gemini-3.5-flash']
cfg['embedding']['api_key']=keys['text-embedding-3-large']
yaml.safe_dump(cfg,open(os.environ['CFG'],'w'),sort_keys=False)
Path('/tmp/judge_'+os.environ['HLE_CHILD']).write_text(keys['gpt-4o-2024-11-20'])
PY
  JUDGE=$(cat "/tmp/judge_${child}"); rm -f "/tmp/judge_${child}"
  export MEMRL_RUN_ID="$child" MEMRL_REASONING_EFFORT=high MEMRL_LLM_MIN_INTERVAL="$interval" MEMRL_EMBED_MIN_INTERVAL="$interval"
  python3 run/run_hle.py --config "$CFG" --train data/hle/hle_test.parquet --judge_model gpt-4o-2024-11-20 --judge_base_url https://matrixllm.alipay.com/v1/ --judge_api_key "$JUDGE"
 ) &
 echo "$! $mode $child" >> "$STATE/children"
}
launch memrltraj 0 4.0 "$MEMRL_CKPT" "${HLE_RID}-memrltraj"
status=0
while read -r pid mode child; do wait "$pid" || status=1; done < "$STATE/children"
exit "$status"
REMOTE
PAYLOAD=$(base64 -w0 "$REMOTE")
export PYTHONPATH=/tmp/yl_pypai AISTUDIO_LOGIN_NAME=aistudio AISTUDIO_USERNUMBER=477578 WORKER_NUM=0 JOB_TAG='' JOB_NAME="$RID" KM_IMAGE='acr-sh-ant-registry-vpc.cn-shanghai.cr.aliyuncs.com/gpu/areal-runtime:dev-sglang-20260401'
export JOB_COMMAND="HLE_RID=$(printf '%q' "$RID") MEMRL_CKPT=$(printf '%q' "$MEMRL_CKPT") bash -c $(printf '%q' "echo $PAYLOAD | base64 -d | bash")"
export LAUNCH_CONTAINER_MODE=dev_local PYPAI_HOME="/tmp/pypai_${RID}" TMPDIR=/tmp
cd /tmp
python3 "$ROOT/MemRL/scripts/submit_hle_template.py"

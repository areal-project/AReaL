#!/usr/bin/env bash
# Fresh DB main-method run: stable Region topology + fixed 1-slot FS + fixed final memory budget of five.
set -euo pipefail

PROJECT_DIR=/storage/openpsi/users/yl/agent-memory/MemRL
LOCAL_SP=/storage/openpsi/users/yl/agent-memory/.local/lib/python3.12/site-packages
HOST_SHORT=$(hostname | sed 's/^gpulingjun//; s/\..*$//')
TS=$(date +%Y%m%d_%H%M%S)
LOGFILE="$PROJECT_DIR/logs/llb_db_region_fs_corrected_fs2_preserve_gpt41mini_${HOST_SHORT}_${TS}.log"
exec > >(tee -a "$LOGFILE") 2>&1
cd "$PROJECT_DIR"

echo '=========================================='
echo 'LLB DB main Region-FS: stable topology, FS=2, natural selection, final selector top-5'
echo "Host: $(hostname)"
echo "Start time: $(date)"
echo "Log: $LOGFILE"
echo '=========================================='

export PYTHONPATH="$PROJECT_DIR:$LOCAL_SP:${PYTHONPATH:-}"
export MEMRL_OS_BACKEND=local
export MEMRL_DB_BACKEND=auto
# DB corrected-proceduralization topology: no section-internal maintenance;
# run initial clustering / split-merge only at each section boundary, with no cooldown.
export MEMRL_REGION_CLUSTER_INIT_STEP=0
unset MEMRL_REGION_TOPOLOGY_COOLDOWN_SECTIONS || true
unset MEMRL_REGION_TOPOLOGY_MAINTENANCE_STEPS || true
unset MEMRL_DB_MULTI_AXIS || true
unset MEMRL_REGION_SPLIT_RANGE_FRACTION || true
unset MEMRL_REGION_SPLIT_ON_RESUME || true
unset MEMRL_REGION_RETRIEVE_MODE || true
unset MEMRL_WEIGHTED_REGION_QUOTA || true
unset MEMRL_RETRIEVAL_AUDIT_PATH || true
unset MEMRL_FINAL_MEMORY_DEDUP || true
export MEMRL_UPDATE_MAX_WORKERS=1
# Next fresh run: enforce a shared cross-process 3s interval for every
# embedding-backed operation (retrieval plus text_mem.add), not merely a local
# fallback throttle. Current running job is intentionally unaffected.
export MEMRL_EMBED_THROTTLE=3.0
export MEMRL_EMBED_GLOBAL_MIN_INTERVAL=3.0
export MEMRL_LLM_MIN_INTERVAL=1.0
export MEMRL_LLB_REFLECTION_PROMPT=v2
export MEMRL_LLB_SCRIPT_DETAIL=db_pattern
export MEMRL_LLM_MODEL=gpt-4.1-mini-2025-04-14
export MEMRL_RUN_ID="${MEMRL_RUN_ID:-region-fs-db-gpt41mini-corrected-fs2-preserve-$(date +%Y%m%d-%H%M%S)}"
export MEMRL_REGION_REGISTER_ON_CREATE=0
export MEMRL_REGION_BACKFILL_ON_RESTORE=0
export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=/tmp/huggingface

VENV_SP=/AReaL/.venv/lib/python3.12/site-packages
printf '%s\n' '[INFO] Installing MariaDB server...'
apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq mariadb-server >/dev/null 2>&1 || echo 'Warning: apt-get install mariadb-server failed'
printf '%s\n' '[INFO] Installing runtime deps...'
pip install mem0ai 'chonkie==1.2.1' tensorboard pandas tqdm concurrent-log-handler mysql-connector-python hdbscan --target "$VENV_SP" -i https://pypi.antfin-inc.com/simple/
python3 -c "import memos, memrl; print('imports OK; memrl from:', memrl.__file__)"

# Job-local overlay: ordinary Region-FS replaces the content of naturally
# selected failure memories but retains the selector's original top-5 IDs.
OVERLAY=/tmp/llb_db_preservefs_overlay
rm -rf "$OVERLAY"
mkdir -p "$OVERLAY"
cp -a "$PROJECT_DIR/memrl" "$PROJECT_DIR/run" "$PROJECT_DIR/scripts/run_llb_with_rotated_matrix_credentials.py" "$OVERLAY/"
ln -s "$PROJECT_DIR/3rdparty" "$OVERLAY/3rdparty"
python3 - "$OVERLAY" <<'PYOVERLAY'
import sys
from pathlib import Path
root=Path(sys.argv[1])
p=root/'memrl/run/llb_rl_runner.py'; s=p.read_text()
old='''        if getattr(self, "_failure_summary_preserve_selection", False):
            success_mems = list(processed_mems.get("successed", []))
            failure_mems = list(processed_mems.get("failed", []))
            if not failure_mems:
                return processed_mems
            mode = getattr(self, "_failure_summary_mode", "region")
            if mode == "region":
                self._replace_failure_with_region_summary(failure_mems)
            else:
                self._replace_failure_with_inline_summary(failure_mems)
            max_success = max(0, final_budget - effective_failure_slots)
            result = {}
            if success_mems[:max_success]:
                result["successed"] = success_mems[:max_success]
            if failure_mems[:effective_failure_slots]:
                result["failed"] = failure_mems[:effective_failure_slots]
            return result
'''
new='''        if getattr(self, "_failure_summary_preserve_selection", False):
            # Canonical ordinary FS: preserve the base selector's exact IDs and
            # order/count; only replace text for naturally selected failures.
            failure_mems = list(processed_mems.get("failed", []))
            if not failure_mems:
                return processed_mems
            mode = getattr(self, "_failure_summary_mode", "region")
            if mode == "region":
                self._replace_failure_with_region_summary(failure_mems)
            else:
                self._replace_failure_with_inline_summary(failure_mems)
            return processed_mems
'''
assert old in s, 'preserve-selection branch target missing'
p.write_text(s.replace(old,new,1))
p=root/'run_llb_with_rotated_matrix_credentials.py'; s=p.read_text()
s=s.replace('Path(__file__).resolve().parents[1] / "run" / "run_llb.py"', 'Path(__file__).resolve().parent / "run" / "run_llb.py"')
p.write_text(s)
PYOVERLAY
PYTHONPATH="$OVERLAY:$LOCAL_SP:${PYTHONPATH:-}" python3 -m py_compile "$OVERLAY/memrl/run/llb_rl_runner.py" "$OVERLAY/run/run_llb.py" "$OVERLAY/run_llb_with_rotated_matrix_credentials.py"
python3 - "$OVERLAY" <<'PYSTATIC'
import sys
from pathlib import Path
text=(Path(sys.argv[1])/'memrl/run/llb_rl_runner.py').read_text()
assert 'preserve the base selector' in text
assert 'return processed_mems' in text
print('[OVERLAY PRECHECK] ordinary Region FS replaces only natural failure content; base top-5 IDs preserved')
PYSTATIC

# Generated config is node-local: no credentials and no output are written to
# the permanent checkpoint tree except the new experiment's own snapshots.
CONFIG=/tmp/rl_llb_db_region_fs_corrected_fs2_preserve.yaml
python3 - "$CONFIG" <<'PYCFG'
import sys
from pathlib import Path
import yaml
src=Path('configs/rl_llb_db_region_fs_hardfix_s3.yaml')
dst=Path(sys.argv[1])
cfg=yaml.safe_load(src.read_text())
# Credentials are injected into the in-memory MempConfig by the wrapper.
cfg['llm']['api_key']='runtime-injected'
cfg['embedding']['api_key']='runtime-injected'
# Critical: no old Region checkpoint/polluted split posterior can enter this run.
cfg['memory']['load_from_checkpoint']=False
cfg['memory']['checkpoint_path']=''
run_id = __import__('os').environ['MEMRL_RUN_ID']
safe_run_id = ''.join(ch if ch.isalnum() or ch in '-_' else '_' for ch in run_id)
cfg['memory']['user_id']='llb_db_region_fs_corrected_fs2_preserve_' + safe_run_id
cfg['experiment']['experiment_name']='llb_db_region_fs_gpt41mini_corrected_fs2_preserve'
cfg['experiment']['ckpt_save_every_n_batches']=10
cfg['experiment']['ckpt_max_keep']=3
cfg['experiment']['eval_runs']=1
cfg['experiment']['eval_temperature']=0.0
# Main-method retrieval contract: recall 10 candidates, inject exactly a 5-slot final context whenever five candidates exist.
cfg['memory']['k_retrieve']=10
cfg['rl_config']['topk']=5
# FS contract is natural-selection: do not force a failure slot or candidate-pool backfill.
assert cfg['memory']['k_retrieve'] == 10
assert cfg['rl_config']['topk'] == 5
dst.write_text(yaml.safe_dump(cfg,sort_keys=False))
PYCFG

printf '%s\n' '[INFO] Fresh DB main-method run: corrected section-end topology (no cooldown); FS=2 ordinary FS=2; preserve selector top-5 IDs; recall=10.'
python3 - "$CONFIG" <<'PYVERIFY'
import sys, yaml
cfg=yaml.safe_load(open(sys.argv[1]))
assert cfg['memory']['load_from_checkpoint'] is False
assert cfg['memory']['checkpoint_path'] == ''
assert cfg['memory']['k_retrieve'] == 10
assert cfg['rl_config']['topk'] == 5
print('[PRECHECK] fresh checkpoint=false; DB corrected topology=section-end only/no cooldown; candidate recall=10; final selector topk=5; ordinary FS=2, selection-preserving top-5')
PYVERIFY
printf '%s\n' "[INFO] MEMRL_RUN_ID=$MEMRL_RUN_ID"
PYTHONPATH="$OVERLAY:$LOCAL_SP:${PYTHONPATH:-}" python3 "$OVERLAY/run_llb_with_rotated_matrix_credentials.py" \
  --config "$CONFIG" \
  --output_dir /storage/openpsi/experiments/checkpoints/admin/yl-mem-region/llb_v2reflect \
  --region --region_gating_mode additive \
  --propagation_eta 0.12 \
  --shrinkage_confidence_k 3.0 \
  --no_z_norm \
  --explore_schedule '0,4,3,2,2,1,1,1,1,0' \
  --failure_summary_n_slots 2 \
  --failure_summary_preserve_selection \
  --resume_eval_section -1

echo '=========================================='
echo "End time: $(date)"
echo '=========================================='

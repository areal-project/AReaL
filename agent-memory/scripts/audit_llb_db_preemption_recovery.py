#!/usr/bin/env python3
"""Static audit for the default LLB-DB preemption recovery contract."""
from pathlib import Path
import re
import sys
import yaml

ROOT = Path(__file__).resolve().parents[1]
errors = []

def require(ok, message):
    if not ok:
        errors.append(message)

submit = (ROOT / 'scripts/submit_llb_db_memrl_haiku.py').read_text()
require('RetryPolicy.ON_EVICTION' in submit, 'submit template lacks ON_EVICTION')
require(re.search(r'max_attempt\s*=\s*10', submit) is not None, 'submit template max_attempt != 10')

run_llb = (ROOT / 'run/run_llb.py').read_text()
require('"--resume_eval_section", type=int, default=-1' in run_llb,
        'resume_eval_section default is not -1')
require('_maybe_resume_from_ckpt_if_needed' in run_llb, 'auto-resume scanner missing')

runner = (ROOT / 'memrl/run/llb_rl_runner.py').read_text()
for token in ('task_outcomes.jsonl', 'cum_state.json', '_append_llb_task_outcomes',
              '_persist_llb_cum_state', '_validation_done_marker'):
    require(token in runner, f'LLB runner missing recovery token: {token}')

configs = {
    'memp': ('configs/rl_llb_db_memp.yaml', 3),
    'rag': ('configs/rl_llb_db_rag.yaml', 3),
    'selfrag': ('configs/rl_llb_db_selfrag.yaml', 3),
    'mem0': ('configs/rl_llb_db_mem0.yaml', 1000),
    'region': ('configs/rl_llb_db_region_fs.yaml', 3),
    'memrl': ('configs/rl_llb_db_memrl_haiku_v2reflect.yaml', 3),
}
for name, (relative, expected_keep) in configs.items():
    cfg = yaml.safe_load((ROOT / relative).read_text())
    exp = cfg.get('experiment', {})
    require(int(exp.get('ckpt_save_every_n_batches', 0)) == 10,
            f'{name}: batch checkpoint interval != 10')
    require(int(exp.get('ckpt_max_keep', 0)) == expected_keep,
            f'{name}: ckpt_max_keep != {expected_keep}')

launchers = [
    'scripts/run_llb_db_baselines_gpt41mini_aistudio.sh',
    'scripts/run_llb_db_memp_gpt41mini_aistudio.sh',
    'scripts/run_llb_db_memrl_gpt41mini_aistudio_v2reflect.sh',
    'scripts/run_llb_db_region_fs_splitprior_fix_aistudio.sh',
]
for relative in launchers:
    text = (ROOT / relative).read_text()
    require('MEMRL_RUN_ID' in text, f'{relative}: stable MEMRL_RUN_ID missing')

if errors:
    print('LLB_DB_PREEMPTION_AUDIT_FAILED')
    for error in errors:
        print(' -', error)
    sys.exit(1)
print('LLB_DB_PREEMPTION_AUDIT_OK')

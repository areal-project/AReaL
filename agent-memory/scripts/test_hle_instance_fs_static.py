#!/usr/bin/env python3
from pathlib import Path
r=Path('scripts/hle_instance_fs_overlay/memrl/run/hle_region_runner.py').read_text()
e=Path('scripts/hle_instance_fs_overlay/run_hle_region.py').read_text()
score=Path('scripts/hle_instance_fs_overlay/memrl/service/region_memory_service.py').read_text()
for token in ['_failure_summary_require_exact_task','_fs_task_compatible','exact_task_gate=%s','task_description']:
 assert token in r,token
for token in ['failure_summary_require_exact_task','require_exact_task']:
 assert token in e,token
for token in ['Scoring: hybrid_score = sim * w_sim + Q[target_subtask] * w_q','final_score = hybrid_score * region_gating_score']:
 assert token in score,token
print('PASS hle instance FS static; score formula unchanged')

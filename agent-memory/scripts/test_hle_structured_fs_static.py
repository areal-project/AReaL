#!/usr/bin/env python3
from pathlib import Path
runner=Path('scripts/hle_structured_fs_overlay/memrl/run/hle_region_runner.py').read_text()
entry=Path('scripts/hle_structured_fs_overlay/run_hle_region.py').read_text()
for token in ['self._failure_summary_mode != "hle_structured"','_failure_summary_independent_pool','_failure_summary_min_success','_failure_summary_min_similarity','_failure_summary_structured_min_evidence','_failure_summary_signature_fields','[HLE STRUCTURED FS]','abstained_no_evidence','baseline_selected = list(selected_mems)','out = baseline_selected','baseline_ids=%s final_ids=%s']:
    assert token in runner, token
for token in ['raw_cfg = yaml.safe_load','failure_summary_mode','failure_summary_independent_pool','failure_summary_min_success','failure_summary_min_similarity','failure_summary_structured_min_evidence','[HLE FS EFFECTIVE CONFIG]']:
    assert token in entry, token
print('PASS hle structured FS static contract')

#!/usr/bin/env python3
from pathlib import Path
svc=Path('scripts/hle_region_safe_overlay/memrl/service/region_memory_service.py').read_text()
run=Path('scripts/hle_region_safe_overlay/memrl/run/hle_region_runner.py').read_text()
entry=Path('scripts/hle_region_safe_overlay/run_hle_region.py').read_text()
for token in ['exact_success_anchor(', '[EXACT ANCHOR]', 'protect_exact_success_memory']:
    assert token in svc, token
for token in ['_topology_mid_section_step', '_topology_changed_this_section', 'phase="mid_late"', 'phase="section_end"', 'end skipped: topology already changed this section']:
    assert token in run, token
assert run.count('def _region_clustering_step')==1
assert run.count('def _region_end_of_section')==1
for token in ['topology_mid_section_step', 'topology_min_change_gap', 'protect_exact_success_memory']:
    assert token in entry, token
print('PASS hle region safe overlay static contract')

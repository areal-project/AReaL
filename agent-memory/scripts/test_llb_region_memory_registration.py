#!/usr/bin/env python3
"""Regression: all newly stored LLB memories enter Region geometry without fake evidence."""
from pathlib import Path
import importlib.util,sys
import numpy as np
ROOT=Path('/storage/openpsi/users/yl/agent-memory/MemRL')
spec=importlib.util.spec_from_file_location('rmtest',ROOT/'memrl/service/region_manager.py');m=importlib.util.module_from_spec(spec);sys.modules[spec.name]=m;spec.loader.exec_module(m)
R=m.Region;RM=m.RegionManager
rm=RM(task_hierarchy={},temperature=.1)
rm._known_subtasks=['llb_db/select_simple']
rm._is_clustered=True
rm.regions=[R(region_id=0,centroid=np.array([.8]),member_ids=[]),R(region_id=1,centroid=np.array([.1]),member_ids=[])]
assert rm.register_memory('success','llb_db/select_simple',.9)
assert rm.register_memory('failure','llb_db/select_simple',0.0)
assert set(rm.membership_weights)=={'success','failure'}
assert 'failure' in rm.regions[1].member_ids and 'success' in rm.regions[0].member_ids
assert rm.subtask_q_counts['failure']=={}
assert 'failure' not in rm.memory_total_count_by_subtask
assert 'failure' not in rm.memory_success_sum_by_subtask
print('LLB_REGION_MEMORY_REGISTRATION_TEST_OK')

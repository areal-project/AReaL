#!/usr/bin/env python3
import importlib.util, pathlib, sys, types
import numpy as np
# Deterministic split on first dimension median.
cluster=types.ModuleType('sklearn.cluster')
class KM:
 def __init__(self,*a,**k): pass
 def fit_predict(self,X): return (np.asarray(X)[:,0]>=np.median(np.asarray(X)[:,0])).astype(int)
cluster.KMeans=KM
sk=types.ModuleType('sklearn'); sk.cluster=cluster
sys.modules['sklearn']=sk; sys.modules['sklearn.cluster']=cluster
spec=importlib.util.spec_from_file_location('rm_prog',pathlib.Path('/tmp/llbdb_progressive_topology/region_manager.py'))
mod=importlib.util.module_from_spec(spec);sys.modules[spec.name]=mod;spec.loader.exec_module(mod)
R,M=mod.Region,mod.RegionManager
m=M(task_hierarchy={},min_cluster_size=3,region_utility_mode='beta',region_split_evidence_migration_mode='hard_member_rebase',region_max_variance_splits_per_epoch=1,region_split_min_effective_evidence=200,region_progressive_best_split=True,region_max_merges_per_epoch=1,region_split_min_child_size=3,region_protect_new_split_children=True)
m._known_subtasks=['a','b'];m._is_clustered=True
regs=[]
# Region 0 variance lower; Region 1 variance higher. Both evidence=240.
for rid,(lo,hi) in enumerate(((.35,.65),(.05,.95))):
 members=[f'r{rid}_{i}' for i in range(12)]
 for i,mid in enumerate(members):
  v=lo if i<6 else hi
  m.subtask_q[mid]={'a':v,'b':v};m.subtask_q_counts[mid]={'a':4,'b':4}
  m.memory_success_sum_by_subtask[mid]={'a':2,'b':2};m.memory_total_count_by_subtask[mid]={'a':10,'b':10}
 regs.append(R(region_id=rid,member_ids=members,centroid=np.array([(lo+hi)/2]*2),utility_by_subtask={'a':.2+rid*.6,'b':.2+rid*.6},counts_by_subtask={'a':12,'b':12},success_sum_by_subtask={'a':120,'b':120},total_count_by_subtask={'a':120,'b':120},prior_alpha_by_subtask={'a':2.5,'b':2.5},prior_beta_by_subtask={'a':2.5,'b':2.5}))
m.regions=regs
assert m.maybe_split_merge()
# Only high-variance region 1 splits; region 0 remains whole; new children cannot merge same cycle.
sets=[set(r.member_ids) for r in m.regions]
assert set(regs[0].member_ids) in sets
assert len(m.regions)==3, [len(r.member_ids) for r in m.regions]
assert sorted(len(r.member_ids) for r in m.regions)==[6,6,12]
print('DB_PROGRESSIVE_TOPOLOGY_TEST_OK')

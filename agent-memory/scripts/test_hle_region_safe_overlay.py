#!/usr/bin/env python3
import os, sys
from pathlib import Path
ROOT=Path('/storage/openpsi/users/yl/agent-memory/MemRL')
OV=ROOT/'scripts/hle_region_safe_overlay'
os.environ['MEMRL_HLE_SAFE_SOURCE']=str(ROOT)
sys.path[:0]=[str(OV),str(ROOT)]

from memrl.service.region_memory_service import RegionMemoryService
from memrl.run.hle_region_runner import HLERegionRunner

# Exact anchor: successful exact task must become slot 0 and retain top-k uniqueness.
svc=RegionMemoryService.__new__(RegionMemoryService)
svc.exact_match_similarity=0.9999
svc._exact_anchor_stats={'calls':0,'hits':0,'already_selected':0}
q='same question'
exact={'memory_id':'exact','similarity':1.0,'content':'Task: same question\n\nCOMPACT','metadata':{'success':True}}
selected=[{'memory_id':'a','similarity':.8,'content':'Task: other','metadata':{'success':True}},
          {'memory_id':'b','similarity':.7,'content':'Task: other2','metadata':{'success':True}},
          {'memory_id':'c','similarity':.6,'content':'Task: other3','metadata':{'success':False}}]
out=svc._apply_exact_success_anchor(selected,[*selected,exact],q,3)
assert [x['memory_id'] for x in out]==['exact','a','b'],out
# Failed or non-exact candidates must not anchor.
failed=dict(exact,memory_id='failed',metadata={'success':False})
out2=svc._apply_exact_success_anchor(selected,[*selected,failed],q,3)
assert [x['memory_id'] for x in out2]==['a','b','c']

class RM:
    def __init__(self, changes):
        self._is_clustered=True; self.regions=[1,2]; self.subtask_q={}; self.membership_weights={}; self.changes=list(changes); self.calls=0
    def maybe_split_merge(self): self.calls+=1; return self.changes.pop(0) if self.changes else False
    def classify_transfer_patterns(self): pass

# One actual change at mid/late => section end cannot change again.
r=HLERegionRunner.__new__(HLERegionRunner); r.memory_service=type('M',(),{})(); r.memory_service.region_manager=RM([True,True])
r._region_cluster_init_step=500; r._region_merge_interval=400; r._topology_mid_section_step=1600; r._topology_min_change_gap=0
r._topology_section=None; r._topology_section_new=0; r._topology_mid_checked=False; r._topology_changed_this_section=False; r._last_topology_change_global_step=0; r._global_step=500
r._region_clustering_step(1500,2); assert r.memory_service.region_manager.calls==0
r._region_clustering_step(100,2); assert r.memory_service.region_manager.calls==1 and r._topology_changed_this_section
r._region_end_of_section(2); assert r.memory_service.region_manager.calls==1

# No mid change => section end gets one second check; still max one actual change.
r2=HLERegionRunner.__new__(HLERegionRunner); r2.memory_service=type('M',(),{})(); r2.memory_service.region_manager=RM([False,True])
r2._region_cluster_init_step=500; r2._region_merge_interval=400; r2._topology_mid_section_step=1600; r2._topology_min_change_gap=0
r2._topology_section=None; r2._topology_section_new=0; r2._topology_mid_checked=False; r2._topology_changed_this_section=False; r2._last_topology_change_global_step=0; r2._global_step=500
r2._region_clustering_step(1600,2); assert r2.memory_service.region_manager.calls==1 and not r2._topology_changed_this_section
r2._region_end_of_section(2); assert r2.memory_service.region_manager.calls==2 and r2._topology_changed_this_section
print('PASS hle region safe overlay')

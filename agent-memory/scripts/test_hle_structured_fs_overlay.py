#!/usr/bin/env python3
import os,sys
from pathlib import Path
ROOT=Path('/storage/openpsi/users/yl/agent-memory/MemRL')
OV=ROOT/'scripts/hle_structured_fs_overlay'
os.environ['MEMRL_HLE_SAFE_SOURCE']=str(ROOT)
sys.path[:0]=[str(OV),str(ROOT)]
from memrl.run.hle_region_runner import HLERegionRunner

class MD:
    def __init__(self, **kw): self.model_extra=kw
class Obj:
    def __init__(self, **kw): self.metadata=MD(**kw)
class Region:
    def __init__(self): self.region_id=7; self.member_ids=['f1','f2']
class RM:
    def __init__(self):
        self.regions=[Region()]; self.membership_weights={'cand':[1.0]}
class MS:
    def __init__(self):
        self.region_manager=RM()
        common=dict(success=False,category='Math',answer_type='exactMatch',raw_subject='Algebra')
        self._mem_cache={
          'f1':Obj(**common,full_content='FAILURE_MODE: algebraic sign error\nMISTAKES:\n- lost a minus sign during substitution\nFIXES:\n- verify every algebraic substitution carefully'),
          'f2':Obj(**common,full_content='FAILURE_MODE: algebraic sign error\nMISTAKES:\n- expanded the expression with the wrong sign\nFIXES:\n- check each symbolic transformation carefully'),
        }

def mem(mid, success, sim, cat='Math', at='exactMatch'):
    return {'memory_id':mid,'similarity':sim,'content':'memory-'+mid,'metadata':{'success':success,'category':cat,'answer_type':at}}

r=HLERegionRunner.__new__(HLERegionRunner)
r.memory_service=MS(); r.retrieve_k=3
r._failure_inject_log_counter=0; r._region_failure_summaries=None
r._failure_index=None;r._failure_index_size=0
r._structured_fs_stats={'calls':0,'injected':0,'abstained_no_candidate':0,'abstained_no_evidence':0,'abstained_min_success':0}
r.configure_failure_summary(n_slots=1,mode='hle_structured',independent_pool=True,min_success=2,min_similarity=.5,structured_min_evidence=2,signature_fields=['category','answer_type'])
selected=[mem('s1',True,.9),mem('s2',True,.8),mem('badselected',False,.7)]
pool=selected+[mem('cand',False,.75),mem('wrongcat',False,.99,'Physics')]
out=r._inject_failure_summary(selected,'q',candidate_mems=pool,query_metadata={'category':'Math','answer_type':'exactMatch'})
assert len(out)==3,out
assert sum(r._mem_success_flag(x) for x in out)==2,out
assert out[-1].get('_hle_structured_failure_summary'),out
assert 'Evidence count: 2' in out[-1]['content']
# Incompatible target => abstain and preserve all-success fallback.
out2=r._inject_failure_summary(selected,'q',candidate_mems=pool,query_metadata={'category':'Chemistry','answer_type':'multipleChoice'})
assert not any(x.get('_hle_structured_failure_summary') for x in out2),out2
assert [x['memory_id'] for x in out2] == [x['memory_id'] for x in selected], out2

selected_weak=[mem('only_success',True,.9),mem('fsel1',False,.8),mem('fsel2',False,.7)]
out3=r._inject_failure_summary(selected_weak,'q',candidate_mems=selected_weak+[mem('cand',False,.75)],query_metadata={'category':'Math','answer_type':'exactMatch'})
assert [x['memory_id'] for x in out3] == [x['memory_id'] for x in selected_weak], out3
print('PASS hle structured FS overlay',r._structured_fs_stats)

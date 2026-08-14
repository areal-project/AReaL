#!/usr/bin/env python3
import os,sys
from pathlib import Path
ROOT=Path('/storage/openpsi/users/yl/agent-memory/MemRL')
OV=ROOT/'scripts/hle_instance_fs_overlay'
os.environ['MEMRL_HLE_SAFE_SOURCE']=str(ROOT)
sys.path[:0]=[str(OV),str(ROOT)]
from memrl.run.hle_region_runner import HLERegionRunner

class MD:
    def __init__(self, **kw): self.model_extra=kw
class Obj:
    def __init__(self, **kw): self.metadata=MD(**kw)
class Region:
    def __init__(self): self.region_id=7; self.member_ids=['f1','f2','other']
class RM:
    def __init__(self): self.regions=[Region()]; self.membership_weights={'cand':[1.0],'othercand':[1.0]}
class RL: topk=3
class MS:
    def __init__(self):
        self.region_manager=RM(); self.rl_config=RL()
        common=dict(success=False,category='Math',raw_subject='Mathematics')
        self._mem_cache={
          'f1':Obj(**common,task_description='same question',full_content='FAILURE_MODE: sign error\nMISTAKES:\n- lost a minus sign during substitution\nFIXES:\n- verify every algebraic substitution carefully'),
          'f2':Obj(**common,task_description='same question',full_content='FAILURE_MODE: sign error\nMISTAKES:\n- expanded the expression with the wrong sign\nFIXES:\n- check each symbolic transformation carefully'),
          'other':Obj(**common,task_description='different question',full_content='FAILURE_MODE: unrelated\nMISTAKES:\n- made an unrelated mistake in another task\nFIXES:\n- solve the other task carefully'),
        }

def mem(mid, success, sim, task='same question'):
    return {'memory_id':mid,'similarity':sim,'content':'memory-'+mid,'metadata':{'success':success,'category':'Math','raw_subject':'Mathematics','task_description':task}}

def runner():
    r=HLERegionRunner.__new__(HLERegionRunner);r.memory_service=MS();r.retrieve_k=5
    r._failure_inject_log_counter=0;r._region_failure_summaries=None;r._failure_index=None;r._failure_index_size=0
    r._structured_fs_stats={'calls':0,'injected':0,'abstained_no_candidate':0,'abstained_no_evidence':0,'abstained_min_success':0}
    r.configure_failure_summary(n_slots=1,mode='hle_structured',independent_pool=True,min_success=2,min_similarity=.5,structured_min_evidence=2,signature_fields=['category','raw_subject'],require_exact_task=True)
    return r

selected=[mem('s1',True,.9),mem('s2',True,.8),mem('basefail',False,.7)]
r=runner();pool=selected+[mem('cand',False,.75)]
out=r._inject_failure_summary(selected,'same question',candidate_mems=pool,query_metadata={'category':'Math','raw_subject':'Mathematics'})
assert len(out)==3 and sum(r._fs_success_flag(x) for x in out)==2,out
assert any(x.get('_hle_structured_failure_summary') for x in out),out
assert 'Evidence count: 2' in next(x['content'] for x in out if x.get('_hle_structured_failure_summary'))

# Same subject but different task cannot trigger FS; fallback is exact baseline IDs/order.
r2=runner();otherpool=selected+[mem('othercand',False,.95,task='different question')]
out2=r2._inject_failure_summary(selected,'same question',candidate_mems=otherpool,query_metadata={'category':'Math','raw_subject':'Mathematics'})
assert [x['memory_id'] for x in out2]==[x['memory_id'] for x in selected],out2
assert not any(x.get('_hle_structured_failure_summary') for x in out2)
print('PASS hle instance-gated FS overlay',r._structured_fs_stats,r2._structured_fs_stats)

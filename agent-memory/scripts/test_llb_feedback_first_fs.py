#!/usr/bin/env python3
"""Feedback-first cold-start FS: no membership required; source ID survives for Q update."""
import ast,sys
from pathlib import Path
from types import SimpleNamespace
from typing import Dict,List,Optional
import numpy as np
ROOT=Path('/storage/openpsi/users/yl/agent-memory/MemRL');sys.path.insert(0,str(ROOT))
path=ROOT/'memrl/run/llb_rl_runner.py';tree=ast.parse(path.read_text())
cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='LLBRunner')
names={'process_retrieve_mems','_inject_failure_summary','_argmax_region_for_memory','_replace_failure_with_region_summary','_replace_failure_with_db_structured_summary','_replace_failure_with_inline_db_structured_summary','_replace_failure_with_inline_summary'}
methods=[n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name in names]
mini=ast.Module(body=[ast.ClassDef(name='R',bases=[],keywords=[],body=methods,decorator_list=[])],type_ignores=[]);ast.fix_missing_locations(mini)
ns={'Dict':Dict,'List':List,'Optional':Optional,'np':np,'os':__import__('os')};exec(compile(mini,str(path),'exec'),ns);R=ns['R']
skills=['select','subquery_nested','where_single_condition']
def mem(mid,task,ok,text,sim=.8):return {'memory_id':mid,'content':text,'similarity':sim,'metadata':{'source_benchmark':'llb_db','task_id':task,'success':ok,'skill_list':skills,'full_content':text}}
text='FAILURE_MODE: wrong answer\nMISTAKES:\n- Used the wrong global average subquery.\nFIXES:\n- Compute global average before outer filtering.'
r=R();r.task='db';r.rl_config=SimpleNamespace(topk=5);r.retrieve_k=10;r._failure_summary_n_slots=1;r._failure_summary_independent_pool=True;r._failure_summary_min_success=4;r._failure_summary_min_similarity=.6;r._failure_summary_min_evidence=3;r._failure_summary_db_structured=True;r._failure_summary_mode='region';r._failure_summary_inline_k=None
# No failure has Region membership.
r.memory_service=SimpleNamespace(region_manager=SimpleNamespace(regions=[],membership_weights={}),_mem_cache={})
succ=[mem(f's{i}',i,True,f'success {i}',.9-i*.01) for i in range(4)]
fails=[mem(f'f{i}',100+i,False,text,.8-i*.01) for i in range(3)]
out=r._inject_failure_summary(r.process_retrieve_mems(succ),'task',succ+fails,target_skill_list=skills)
assert len(out['successed'])==4 and len(out['failed'])==1,out
fm=out['failed'][0]
assert fm['memory_id'] in {'f0','f1','f2'}
assert fm['_db_structured_cold_start'] is True
assert 'SQL failure guardrails' in fm['content']
# This is exactly how training builds the Q-update ID list.
ids=[m['memory_id'] for bucket in out.values() for m in bucket]
assert fm['memory_id'] in ids
print('LLB_FEEDBACK_FIRST_FS_TEST_OK')

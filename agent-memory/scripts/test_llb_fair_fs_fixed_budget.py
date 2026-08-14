#!/usr/bin/env python3
import ast,sys
from pathlib import Path
from types import SimpleNamespace
from typing import Dict,List,Optional
import numpy as np
P=Path('/storage/openpsi/users/yl/agent-memory/MemRL/memrl/run/llb_rl_runner.py');t=ast.parse(P.read_text());c=next(x for x in t.body if isinstance(x,ast.ClassDef) and x.name=='LLBRunner');names={'process_retrieve_mems','_inject_failure_summary','_replace_failure_with_region_summary','_replace_failure_with_inline_summary','_argmax_region_for_memory'};ms=[x for x in c.body if isinstance(x,ast.FunctionDef) and x.name in names];tree=ast.Module(body=[ast.ClassDef(name='R',bases=[],keywords=[],body=ms,decorator_list=[])],type_ignores=[]);ast.fix_missing_locations(tree);ns={'Dict':Dict,'List':List,'Optional':Optional,'np':np,'os':__import__('os')};exec(compile(tree,str(P),'exec'),ns);R=ns['R']
def m(mid,ok):return {'memory_id':mid,'content':mid,'similarity':.8,'metadata':{'success':ok,'task_id':mid}}
r=R();r.rl_config=SimpleNamespace(topk=5);r.retrieve_k=10;r._failure_summary_n_slots=1;r._failure_summary_fixed_budget=True
reg=SimpleNamespace(region_id=0,failure_summary='DIRECT REGION FS');r.memory_service=SimpleNamespace(region_manager=SimpleNamespace(regions=[reg],membership_weights={'f':np.array([1.0])}))
sel=[m(f's{i}',True) for i in range(5)];pool=sel+[m('f',False)]
out=r._inject_failure_summary(r.process_retrieve_mems(sel),'q',candidate_mems=pool)
assert len(out['successed'])==4 and len(out['failed'])==1 and out['failed'][0]['content']=='DIRECT REGION FS',out
assert len(out['successed'])+len(out['failed'])==5
r.memory_service=SimpleNamespace(region_manager=SimpleNamespace(regions=[],membership_weights={}))
out2=r._inject_failure_summary(r.process_retrieve_mems(sel),'q',candidate_mems=pool)
assert len(out2['successed'])==5 and 'failed' not in out2
print('LLB_FAIR_FS_FIXED_BUDGET_TEST_OK')
